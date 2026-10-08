#!/usr/bin/env python3
"""Resident RAG server. Loads the vision-language model and the embedder once, holds the
corpus index in memory, and answers app.py over a unix socket.

Start order: models load -> warm-up -> socket opens -> /app/corpus auto-indexed in the
background (an explicit `app.py --index` waits for that run instead of repeating it)."""
import json
import os
import pickle
import re
import select
import socketserver
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter

import numpy as np

import retrieval as R

LLM_DIR = os.environ.get("LLM_DIR", "/models/qwen3-vl-8b")
EMB_DIR = os.environ.get("EMB_DIR", "/models/bge-small")
SOCK = os.environ.get("RAG_SOCKET", "/tmp/rag.sock")
DEFAULT_CORPUS = os.environ.get("RAG_CORPUS", "/app/corpus")
INDEX_CACHE = os.environ.get("RAG_INDEX_CACHE", "/tmp/rag_index.pkl")
INDEX_BUDGET_S = float(os.environ.get("RAG_INDEX_BUDGET_S", "480"))
CTX_CHARS = int(os.environ.get("RAG_CTX_CHARS", "24000"))
PARSE_TIMEOUT_S = 60
HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE_MAX_DF = 3  # an identifier shared by more files than this is a topic, not a link


def log(*a):
    print("[server]", *a, flush=True)


# --------------------------------------------------------------------------- models
OCR_PROMPT = """Transcribe every piece of text visible in this image, exactly as printed. Keep codes, part numbers, revisions, units and symbols such as # or ° unchanged.
Start with one line "IMAGE: <what the image shows, in a few words>".
Then write one item per line. Keep things that belong together on the same line: a pin and its signal, a label and its value, a table row. For example "B14: THERM_ALERT#" or "Board revision: REV-C2".
Do not describe the picture beyond the first line and do not add anything that is not printed."""


def load_image(src, max_side=1600, min_side=448):
    """Path or PIL image -> RGB PIL image sized for the vision model."""
    from PIL import Image, ImageOps
    img = src if isinstance(src, Image.Image) else Image.open(src)
    try:
        img.seek(0)
    except Exception:  # noqa: BLE001
        pass
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img).astype("float32")
        a = (a - a.min()) / (a.max() - a.min() + 1e-6) * 255.0
        img = Image.fromarray(a.astype("uint8"))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    s = max_side / max(w, h) if max(w, h) > max_side else (
        min_side / max(w, h) if max(w, h) < min_side else 1.0)
    if s != 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


class Models:
    def __init__(self):
        import torch
        from transformers import (AutoModel, AutoModelForImageTextToText, AutoProcessor,
                                  AutoTokenizer)
        self.torch = torch
        self.lock = threading.Lock()  # one GPU, one generate at a time
        t0 = time.time()
        log(f"torch {torch.__version__} hip={torch.version.hip} "
            f"device={torch.cuda.get_device_name(0)}")
        self.proc = AutoProcessor.from_pretrained(LLM_DIR)
        # Load on CPU then move: avoids caching_allocator_warmup failures on MI300X VFs.
        self.llm = AutoModelForImageTextToText.from_pretrained(
            LLM_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True).to("cuda").eval()
        try:
            self.etok = AutoTokenizer.from_pretrained(EMB_DIR)
            self.emb = AutoModel.from_pretrained(EMB_DIR).to("cuda").eval()
        except Exception:  # noqa: BLE001  (BM25 alone still works)
            traceback.print_exc()
            self.emb = None
        log(f"models loaded in {time.time() - t0:.1f}s")

    def generate(self, messages, images=None, max_new_tokens=300):
        torch = self.torch
        prompt = self.proc.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
        kw = {"text": [prompt], "return_tensors": "pt"}
        if images:
            kw["images"] = images
        with self.lock, torch.inference_mode():
            inputs = self.proc(**kw).to("cuda")
            out = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def ocr(self, img):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": OCR_PROMPT}]}]
        return self.generate(msgs, images=[img], max_new_tokens=768).strip()

    def embed(self, texts, query=False):
        if self.emb is None or not texts:
            return None
        torch = self.torch
        if query:
            texts = ["Represent this sentence for searching relevant passages: " + t
                     for t in texts]
        vecs = []
        with self.lock, torch.inference_mode():
            for i in range(0, len(texts), 64):
                enc = self.etok(texts[i:i + 64], padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to("cuda")
                v = self.emb(**enc).last_hidden_state[:, 0]
                vecs.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(vecs)

    def warmup(self):
        from PIL import Image, ImageDraw
        self.generate([{"role": "user", "content": [{"type": "text", "text": "Reply OK."}]}],
                      max_new_tokens=4)
        img = Image.new("RGB", (640, 320), "white")
        ImageDraw.Draw(img).text((200, 140), "REV-A1 B14", fill="black")
        self.ocr(img)
        self.embed(["warm-up"])


# --------------------------------------------------------------------------- parsing
class ParseWorker:
    """parsers.py runs in a separate plain-Python process (no torch, no GPU)."""

    def __init__(self):
        self.p = None

    def _start(self):
        self.p = subprocess.Popen([sys.executable, "-c", "import parsers; parsers.serve()"],
                                  cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)

    def kill(self):
        if self.p is not None:
            try:
                self.p.kill()
                self.p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        self.p = None

    def parse(self, path):
        if self.p is None or self.p.poll() is not None:
            self._start()
        try:
            self.p.stdin.write(json.dumps({"path": path}) + "\n")
            self.p.stdin.flush()
            ready, _, _ = select.select([self.p.stdout], [], [], PARSE_TIMEOUT_S)
            if not ready:
                raise TimeoutError(f"parse exceeded {PARSE_TIMEOUT_S}s")
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("parser process died")
            return json.loads(line)
        except Exception as e:  # noqa: BLE001
            self.kill()
            return {"skip": f"parser failure: {e}"}


def corpus_signature(root):
    sig = []
    for dp, dns, fns in os.walk(root, onerror=lambda e: None):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            try:
                st = os.stat(p)
                sig.append((rel, st.st_size, st.st_mtime_ns))
            except OSError:
                sig.append((rel, -1, -1))
    return sig


def render_pdf_pages(path, max_pages=8):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(path)
    try:
        for i in range(min(len(pdf), max_pages)):
            yield i + 1, pdf[i].render(scale=2).to_pil().convert("RGB")
    finally:
        pdf.close()


# --------------------------------------------------------------------------- index
class Index:
    def __init__(self, root, signature, files, chunks, emb):
        self.root, self.signature = root, signature
        self.files, self.chunks, self.emb = files, chunks, emb
        self.bm25 = R.BM25([R.tokens(f"{c['title']} {c['loc']} {c['text']}") for c in chunks])
        self.file_chunks = {}
        for i, c in enumerate(chunks):
            self.file_chunks.setdefault(c["file"], []).append(i)
        self.file_ids = {f: {R.norm_value(x) for x in R.ids_in(d["text"].upper())}
                         for f, d in files.items()}
        self.id_df = Counter(i for ids in self.file_ids.values() for i in ids)

    def contains(self, f, value):
        pat = R.value_pattern(value)
        return bool(pat and pat.search(self.files[f]["text"]))

    def contains_chunk(self, i, value):
        pat = R.value_pattern(value)
        return bool(pat and pat.search(self.chunks[i]["text"]))

    def save(self, path):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"root": self.root, "signature": self.signature, "files": self.files,
                         "chunks": self.chunks, "emb": self.emb}, fh)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["root"], d["signature"], d["files"], d["chunks"], d["emb"])


def build_index(root, models, worker):
    t0 = time.monotonic()
    deadline = t0 + INDEX_BUDGET_S
    signature = corpus_signature(root)
    files, chunks, skipped = {}, [], []
    for rel, size, _ in signature:
        path = os.path.join(root, rel)
        if size < 0:
            skipped.append((rel, "cannot stat"))
            continue
        res = worker.parse(path)
        if "skip" in res:
            skipped.append((rel, res["skip"]))
            continue
        units = res.get("units", [])
        try:
            if res["kind"] == "image":
                if time.monotonic() > deadline:
                    skipped.append((rel, "index budget exhausted before OCR"))
                    continue
                units = [{"text": models.ocr(load_image(path)), "loc": "text in image",
                          "row": False}]
            elif res["kind"] == "pdf_scanned":
                units = []
                for n, img in render_pdf_pages(path):
                    if time.monotonic() > deadline:
                        break
                    units.append({"text": models.ocr(load_image(img)),
                                  "loc": f"page {n}", "row": False})
        except Exception as e:  # noqa: BLE001
            skipped.append((rel, f"ocr failed: {e}"))
            continue
        text = "\n".join(u["text"] for u in units if u.get("text"))
        if not text.strip():
            skipped.append((rel, "no text"))
            continue
        files[rel] = {"kind": res["kind"], "withdrawn": bool(res.get("withdrawn")),
                      "text": text}
        chunks.extend(R.make_chunks(rel, {"units": units}))
    worker.kill()

    # An older revision of the same document is superseded by a newer one.
    fams = {}
    for rel in files:
        fams.setdefault(R.family_key(rel), []).append(rel)
    for members in fams.values():
        revs = [(R.revision_number(m), m) for m in members]
        live = [r for r, m in revs if r >= 0 and not files[m]["withdrawn"]]
        if len(members) > 1 and live:
            top = max(live)
            for r, m in revs:
                if 0 <= r < top:
                    files[m]["withdrawn"] = True
    for c in chunks:
        c["withdrawn"] = files[c["file"]]["withdrawn"]

    emb = models.embed([f"{c['title']} | {c['loc']}\n{c['text']}"[:2000] for c in chunks])
    idx = Index(root, signature, files, chunks, emb)
    log(f"indexed {len(files)} files / {len(chunks)} chunks in {time.monotonic() - t0:.1f}s; "
        f"withdrawn={[f for f, d in files.items() if d['withdrawn']]}")
    for rel, why in skipped:
        log(f"  skipped {rel}: {why}")
    return idx, skipped


# --------------------------------------------------------------------------- retrieval
OLD_REV_Q = re.compile(r"withdrawn|superseded|obsolete|previous|older|original|earlier|"
                       r"\br1\b|\brev(?:ision)?\s*1\b", re.I)


def search(idx, models, question, top=40):
    n = len(idx.chunks)
    if n == 0:
        return []
    allow_old = bool(OLD_REV_Q.search(question))
    bm = idx.bm25.scores(R.tokens(question))
    rankings = [(sorted(bm, key=bm.get, reverse=True)[:200], 1.0)]
    if idx.emb is not None:
        qv = models.embed([question], query=True)
        if qv is not None:
            sims = idx.emb @ qv[0]
            rankings.append((list(np.argsort(-sims)[:200]), 1.0))
    q_ids = R.ids_in(question.upper())
    if q_ids:
        hit = [i for i, c in enumerate(idx.chunks)
               if any(x.lower() in c["text"].lower() for x in q_ids)]
        hit.sort(key=lambda i: -bm.get(i, 0.0))
        rankings.append((hit, 1.5))
    fused = R.rrf(rankings)
    ranked = [i for i in sorted(fused, key=fused.get, reverse=True)
              if allow_old or not idx.chunks[i].get("withdrawn")]

    # Multi-hop: identifiers in the best chunks that the question did not name (a ticket
    # number found in a log line) pull in the chunks that define them.
    qn = {R.norm_value(x) for x in q_ids}
    extra = []
    for i in ranked[:6]:
        for raw in R.ids_in(idx.chunks[i]["text"].upper()):
            k = R.norm_value(raw)
            if k in qn or idx.id_df.get(k, 99) > BRIDGE_MAX_DF or k in {e[0] for e in extra}:
                continue
            extra.append((k, raw))
    added = []
    for k, raw in extra[:6]:
        cand = [i for i, c in enumerate(idx.chunks) if raw.lower() in c["text"].lower()]
        cand = [i for i in cand if (allow_old or not idx.chunks[i].get("withdrawn"))
                and i not in ranked[:3] and i not in added]
        cand.sort(key=lambda i: -fused.get(i, 0.0))
        added.extend(cand[:3])
    head = ranked[:3]
    tail = [i for i in ranked[3:] if i not in added]
    return (head + added + tail)[:top + len(added)]


def build_context(idx, ranked, question):
    allow_old = bool(OLD_REV_Q.search(question))
    usable = [f for f, d in idx.files.items() if allow_old or not d["withdrawn"]]
    order = []
    for i in ranked:
        f = idx.chunks[i]["file"]
        if f not in order:
            order.append(f)
    total = sum(len(idx.files[f]["text"]) for f in usable)
    picked = {}
    if total <= CTX_CHARS:  # small corpus: show every usable file whole
        order += [f for f in usable if f not in order]
        for f in order:
            picked[f] = list(idx.file_chunks.get(f, []))
    else:
        used = 0
        for i in ranked:
            c = idx.chunks[i]
            if used + len(c["text"]) > CTX_CHARS:
                continue
            picked.setdefault(c["file"], []).append(i)
            used += len(c["text"]) + 40
        for f, ids in picked.items():  # tables: always show the column header row
            hdr = [i for i in idx.file_chunks.get(f, []) if idx.chunks[i]["loc"].endswith("header")]
            for h in hdr[:2]:
                if h not in ids:
                    ids.append(h)
        order = [f for f in order if f in picked]
    parts = []
    for f in order:
        ids = sorted(set(picked.get(f, [])))
        if not ids:
            continue
        head = f"### FILE: {f}"
        if idx.files[f]["withdrawn"]:
            head += "  (WITHDRAWN / SUPERSEDED - do not use)"
        body = "\n".join(f"[{idx.chunks[i]['loc']}] {idx.chunks[i]['text']}" for i in ids)
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts), order


# --------------------------------------------------------------------------- answering
SYSTEM = """You answer questions about a private document collection. The documents describe fictional products, so you know nothing about them beyond what is written below. Use ONLY the documents provided.

How to answer:
- Give the value only: a number, part number, version, code, pin, date or quarter. No sentence, no explanation, no units (write 94, not 94 °C).
- Give the complete value, including any qualifier that identifies it: "Q3 FY27" not "Q3", "REV-C2" not "C2", a part number with every suffix.
- Copy the value exactly as it is written in the document.
- Some questions need two steps: one document gives an identifier (ticket number, error code, part number, version) and another document gives the value for that identifier. Follow the chain.
- If the documents do not contain the answer, the answer is "". Never guess, never use outside knowledge, and never use a document marked WITHDRAWN.

How to cite ("sources"):
- List only the files you actually needed: a file belongs in sources only if, without it, the answer could not be found.
- If you followed a chain, list every file in the chain: the file that gave the identifier AND the file that gave the value.
- Do not list files that only discuss the same topic or repeat the value.
- Put the file that states the value first.

Reply with one JSON object and nothing else:
{"steps": "<one short sentence: which file gave what>", "answer": "<value, or empty string>", "sources": ["<file path>", ...]}"""

REFUSALS = {"", "none", "null", "n/a", "na", "unknown", "not found", "not available",
            "not specified", "not mentioned", "not provided", "not stated", "no answer",
            "not applicable", "-"}
UNIT_RE = re.compile(
    r"^[$€£]?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:°\s*[CFK]?|deg(?:rees?)?\s*[CF]?|[CF]|"
    r"seconds?|secs?|s|ms|milliseconds?|minutes?|mins?|hours?|hrs?|h|days?|weeks?|wks?|"
    r"months?|years?|%|percent|v|volts?|mv|w|watts?|a|amps?|ma|mm|cm|kg|g|"
    r"mhz|ghz|khz|hz|gb|mb|kb|tb|gib|mib|units?|pcs|pieces|usd|eur)?\s*$", re.I)


def parse_reply(raw):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        try:
            d = json.loads(s[a:b + 1])
            if isinstance(d, dict):
                return d
        except ValueError:
            pass
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', s)
    srcs = re.findall(r'"([^"]+\.[A-Za-z0-9]{1,5})"', s)
    return {"answer": m.group(1) if m else "", "sources": srcs}


def clean_answer(a):
    if isinstance(a, (int, float)):
        a = str(a)
    if not isinstance(a, str):
        return ""
    a = a.strip().strip("`\"'“”‘’").strip()
    a = re.sub(r"^(?:answer|value|result)\s*[:：]\s*", "", a, flags=re.I).strip()
    a = re.sub(r"\s*\([^)]*\)$", "", a).strip()  # drop a trailing parenthetical
    a = a.rstrip(".;,").strip() if not re.search(r"\d\.$", a) else a
    if a.lower() in REFUSALS or len(a.split()) > 8 or len(a) > 80:
        return ""
    m = UNIT_RE.match(a)
    if m and m.group(0).strip() != m.group(1):
        a = m.group(1)
    return a


def resolve_files(idx, names, allowed):
    out = []
    for n in names if isinstance(names, list) else []:
        if not isinstance(n, str):
            continue
        n = n.strip().lstrip("./")
        n = n.split("/app/corpus/", 1)[-1] if "/app/corpus/" in n else n
        hit = [f for f in allowed if f == n] or [f for f in allowed if f.endswith("/" + n)] or \
            [f for f in allowed if os.path.basename(f) == os.path.basename(n)]
        if hit and hit[0] not in out:
            out.append(hit[0])
    return out


def choose_citations(idx, question, answer, model_files, ctx_files, ranked, pick=None):
    """Necessity, not relevance: keep the file(s) that hold the value, plus any file that
    supplied a rare identifier the value's own row/line depends on."""
    a_files = [f for f in model_files if idx.contains(f, answer)]
    if not a_files:
        cand = [f for f in ctx_files if idx.contains(f, answer)]
        if not cand:
            return None  # value appears in no document we showed: treat as invented
        a_files = cand[:1]
    q_ids = {R.norm_value(x) for x in R.ids_in(question.upper())}
    ans_n = R.norm_value(answer)
    # identifiers that sit next to the value (same chunk) in each answer file
    near = {}
    for f in a_files:
        ids = set()
        for i in idx.file_chunks.get(f, []):
            if idx.contains_chunk(i, answer):
                ids |= {R.norm_value(x) for x in R.ids_in(idx.chunks[i]["text"].upper())}
        near[f] = {x for x in ids - q_ids if x != ans_n and len(x) >= 3
                   and idx.id_df.get(x, 99) <= BRIDGE_MAX_DF}
    near_all = set().union(*near.values()) if near else set()
    links, bridges = [], set()
    for f in model_files:
        if f in a_files:
            continue
        shared = idx.file_ids.get(f, set()) & near_all
        if shared:
            links.append(f)
            bridges |= shared
    if len(a_files) > 1:
        linked = [f for f in a_files if near[f] & bridges] if bridges else []
        if linked:
            a_files = linked
        else:  # same value in several files: keep the one the question is about
            chosen = pick(a_files) if pick else None
            if chosen not in a_files:
                pos = {i: n for n, i in enumerate(ranked)}

                def fit(f):
                    hits = [pos[i] for i in idx.file_chunks.get(f, [])
                            if i in pos and idx.contains_chunk(i, answer)]
                    return min(hits) if hits else len(pos) + ctx_files.index(f)
                chosen = min(a_files, key=fit)
            a_files = [chosen]
    return a_files + links


PICK_PROMPT = """Question: {q}
Answer: {a}

Each of these files contains that answer:
{files}

Which ONE file is the direct source the question is asking about? Consider what the question says (for example "logged" points to a log, "printed on the label" to an image, "in the ingest service" to its code). Reply with the file path only."""


def make_picker(idx, models, question, answer):
    def pick(files):
        listing = "\n\n".join(
            f"### {f}\n" + "\n".join(idx.chunks[i]["text"][:600] for i in idx.file_chunks[f]
                                     if idx.contains_chunk(i, answer))[:1200] for f in files)
        msg = [{"role": "user", "content": [{"type": "text", "text": PICK_PROMPT.format(
            q=question, a=answer, files=listing)}]}]
        reply = models.generate(msg, max_new_tokens=40)
        got = resolve_files(idx, re.findall(r"[\w./\-]+\.\w{1,5}", reply), files)
        return got[0] if got else None
    return pick


def answer_question(idx, models, question):
    empty = {"answer": "", "citations": [], "confidence": 0.0}
    if idx is None or not idx.chunks or not question.strip():
        return empty
    t0 = time.monotonic()
    ranked = search(idx, models, question)
    ctx, ctx_files = build_context(idx, ranked, question)
    if not ctx:
        return empty
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text":
                                          f"Documents:\n\n{ctx}\n\nQuestion: {question}"}]}]
    raw = models.generate(msgs, max_new_tokens=300)
    d = parse_reply(raw)
    ans = clean_answer(d.get("answer", ""))
    cites = []
    if ans:
        model_files = resolve_files(idx, d.get("sources", []), ctx_files)
        cites = choose_citations(idx, question, ans, model_files, ctx_files, ranked,
                                 pick=make_picker(idx, models, question, ans))
        if not cites:
            ans, cites = "", []
    log(f"Q: {question!r} -> {ans!r} {cites} ({time.monotonic() - t0:.1f}s) "
        f"steps={d.get('steps', '')!r} model_sources={d.get('sources')}")
    return {"answer": ans, "citations": cites, "confidence": 0.9 if ans else 0.0}


# --------------------------------------------------------------------------- service
class State:
    models = None
    worker = None
    index = None
    index_lock = threading.Lock()
    last_skipped = []


def ensure_index(path):
    path = os.path.abspath(path)
    with State.index_lock:
        sig = corpus_signature(path) if os.path.isdir(path) else []
        if State.index is not None and State.index.root == path and State.index.signature == sig:
            return State.index
        if os.path.exists(INDEX_CACHE):
            try:
                cached = Index.load(INDEX_CACHE)
                if cached.root == path and cached.signature == sig:
                    State.index = cached
                    log(f"index restored from cache ({len(cached.files)} files)")
                    return cached
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        idx, skipped = build_index(path, State.models, State.worker)
        State.index, State.last_skipped = idx, skipped
        try:
            idx.save(INDEX_CACHE)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        return idx


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline().decode("utf-8"))
            cmd = req.get("cmd")
            if cmd == "index":
                idx = ensure_index(req.get("path") or DEFAULT_CORPUS)
                resp = {"ok": True, "files": len(idx.files), "chunks": len(idx.chunks),
                        "skipped": State.last_skipped}
            elif cmd == "query":
                if State.index_lock.acquire(timeout=20):  # wait out an index in progress
                    State.index_lock.release()
                resp = answer_question(State.index, State.models, req.get("question", ""))
            else:
                resp = {"ok": True}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            resp = {"answer": "", "citations": [], "confidence": 0.0, "error": str(e)}
        self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))


def main():
    State.models = Models()
    State.models.warmup()
    State.worker = ParseWorker()
    if os.path.exists(SOCK):
        os.remove(SOCK)
    server = socketserver.ThreadingUnixStreamServer(SOCK, Handler)
    server.daemon_threads = True
    os.chmod(SOCK, 0o777)
    log(f"READY on {SOCK}")
    if os.path.isdir(DEFAULT_CORPUS):
        threading.Thread(target=lambda: _safe(ensure_index, DEFAULT_CORPUS), daemon=True).start()
    server.serve_forever()


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception:  # noqa: BLE001
        traceback.print_exc()


if __name__ == "__main__":
    main()
