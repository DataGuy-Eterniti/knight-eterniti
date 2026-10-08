"""Chunking, identifier-aware BM25, and helpers for multi-hop and citation logic.
Standard library only."""
import math
import os
import re
from collections import Counter, defaultdict

STOP = set("""a an the of to in on for and or is are was were be been by with at from as that
this which what who whom whose when where why how does do did it its into than then there
these those can could should would will shall may might must not no yes if per vs via
about above below between under over any all each some such only also our your their
his her they them we you i me my he she him one two first last value""".split())

# Identifiers: ORR-1847, ORR-FAN-2214-B, TQ-40, E7731, REV-C2, THERM_ALERT#, 0x1F, v4.3.2
ID_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9]*(?:[-_][A-Z0-9#]+)+#?"     # ORR-1847, THERM_ALERT#, REV-C2
    r"|[A-Z]{1,4}\d{2,}[A-Z0-9]*"                    # E7731, TQ40, B14
    r"|0x[0-9A-Fa-f]+"                               # hex codes
    r"|v?\d+\.\d+(?:\.\d+)+)"                        # versions 4.3.2
    r"(?![A-Za-z0-9])")


def norm_value(s):
    """The grader's normalisation: uppercase, no whitespace, no - . · _"""
    s = (s or "").upper()
    return re.sub(r"[\s\-\.·_]", "", s)


def stem(t):
    """Tiny suffix stripper so logged/logs/log and engages/engaged meet."""
    if len(t) <= 4 or not t.isalpha():
        return t
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[:-len(suf)]
            if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "aeiouls":
                t = t[:-1]
            break
    return t


def value_pattern(v):
    """Regex that finds a value the way the grader compares it (case-insensitive,
    separators optional) but only as a whole token, so 4.3.2 never matches ...243 2026."""
    atoms = [a for a in re.split(r"[\s\-\.·_]+", (v or "").strip()) if a]
    if not atoms:
        return None
    sep = r"[\s\-\.·_/]*"
    parts = []
    for a in atoms:
        # allow a gap at letter/digit transitions inside an atom: Q3FY27 ~ Q3 FY27
        pieces = re.findall(r"\d+|[^\W\d_]+|[^\w\s]", a)
        parts.append(r"[\s\-]?".join(re.escape(x) for x in pieces) if pieces else re.escape(a))
    # a version written v4.3.2 / V4.3.2 still counts as 4.3.2
    lead = r"(?:(?<![0-9A-Za-z])|(?<=[vV]))" if atoms[0][0].isdigit() else r"(?<![0-9A-Za-z])"
    return re.compile(lead + sep.join(parts) + r"(?![0-9A-Za-z])", re.I)


def tokens(s):
    s = (s or "").lower()
    out = []
    for m in re.finditer(r"[a-z0-9]+(?:[-_./#][a-z0-9]+)*#?", s):
        t = m.group()
        if t in STOP:
            continue
        out.append(stem(t))
        parts = [p for p in re.split(r"[-_./#]", t) if p]
        if len(parts) > 1:
            out.extend(stem(p) for p in parts if p not in STOP)
            out.append("".join(parts))          # tq-40 -> tq40
        elif re.fullmatch(r"[a-z]+\d+", t):     # tq40 -> tq, 40
            m2 = re.fullmatch(r"([a-z]+)(\d+)", t)
            out.extend([m2.group(1), m2.group(2)])
    return out


def ids_in(text):
    return {m.group() for m in ID_RE.finditer(text or "")}


def family_key(rel):
    """specs/tq40_datasheet_r2.pdf and specs/tq40_datasheet_r1_WITHDRAWN.pdf -> same family."""
    d, b = os.path.split(rel.lower())
    b = os.path.splitext(b)[0]
    b = re.sub(r"(?:^|(?<=[_\-. ]))(?:withdrawn|superseded|obsolete|deprecated|draft|final"
               r"|old|new|latest|current)(?=$|[_\-. ])", "", b)
    b = re.sub(r"(?:^|[_\-. ])(?:r|rev|v|version)[_\-. ]?\d+[a-z]?(?=$|[_\-. ])", "", b)
    b = re.sub(r"[_\-. ]+", "_", b).strip("_")
    return f"{d}/{b}"


def revision_number(rel):
    m = re.search(r"(?:^|[_\-. ])(?:r|rev|v|version)[_\-. ]?(\d+)", os.path.basename(rel.lower()))
    return int(m.group(1)) if m else -1


def make_chunks(rel, parsed, window_lines=14, overlap=4, small_doc_chars=2500):
    """Turn parser units into retrievable chunks. Every chunk carries its file path
    and a readable location so the model (and BM25) can see where it came from."""
    title = re.sub(r"[_\-.]+", " ", os.path.splitext(rel)[0])
    chunks = []
    for u in parsed.get("units", []):
        text = (u.get("text") or "").strip()
        if not text:
            continue
        if u.get("row") or len(text) <= small_doc_chars:
            chunks.append({"file": rel, "loc": u.get("loc", ""), "text": text, "title": title})
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        step = max(1, window_lines - overlap)
        for i in range(0, len(lines), step):
            seg = "\n".join(lines[i:i + window_lines])
            loc = f"{u.get('loc', '')} lines {i + 1}-{min(len(lines), i + window_lines)}"
            chunks.append({"file": rel, "loc": loc.strip(), "text": seg, "title": title})
            if i + window_lines >= len(lines):
                break
    return chunks


class BM25:
    def __init__(self, docs, k1=1.4, b=0.7):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df = defaultdict(int)
        for c in self.tf:
            for t in c:
                df[t] += 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)

    def scores(self, query_tokens):
        sc = defaultdict(float)
        for t in set(query_tokens):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                sc[i] += idf * f * (self.k1 + 1) / denom
        return sc


def rrf(rankings, k=60):
    out = defaultdict(float)
    for ranking, weight in rankings:
        for r, i in enumerate(ranking):
            out[i] += weight / (k + r + 1)
    return out
