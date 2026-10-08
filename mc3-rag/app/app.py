#!/usr/bin/env python3
"""Grader entry point. Standard library only, so it starts in milliseconds.

  python3 /app/app.py --index /app/corpus
  python3 /app/app.py --query "<question>" --query-id q01

Several other spellings are accepted (--question, --id, a JSON or text question file).
It talks to server.py over a unix socket. Every query writes
/app/output/<query-id>_output.json with both "answer" and "citations", even on failure."""
import argparse
import json
import os
import re
import socket
import sys
import time

SOCK = os.environ.get("RAG_SOCKET", "/tmp/rag.sock")
QUERY_BUDGET_S = 26.0   # per-question limit is 30 s
INDEX_BUDGET_S = 570.0  # startup limit (model load + index) is 10 min
EMPTY = {"answer": "", "citations": [], "confidence": 0.0}


def call(req, deadline):
    while True:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(max(1.0, deadline - time.monotonic()))
                s.connect(SOCK)
                s.sendall((json.dumps(req) + "\n").encode("utf-8"))
                buf = b""
                while not buf.endswith(b"\n"):
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
            return json.loads(buf.decode("utf-8"))
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > deadline - 1.0:  # server still loading the model
                raise
            time.sleep(0.5)


def load_query_file(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        data = json.loads(raw)
    except ValueError:
        return [(stem, raw.strip())]
    items = data if isinstance(data, list) else [data]
    out = []
    for n, it in enumerate(items, 1):
        default_id = stem if len(items) == 1 else f"{stem}_{n}"
        if isinstance(it, str):
            out.append((default_id, it))
            continue
        if not isinstance(it, dict):
            continue
        q = next((it[k] for k in ("question", "query", "text", "q", "prompt")
                  if isinstance(it.get(k), str)), "")
        qid = next((str(it[k]) for k in ("id", "query_id", "qid", "query-id", "question_id")
                    if it.get(k) not in (None, "")), default_id)
        out.append((qid, q))
    return out


def write_output(out_dir, qid, result):
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^\w.\-]+", "_", qid).strip("._") or "query"
    path = os.path.join(out_dir, f"{safe}_output.json")
    res = {"answer": str(result.get("answer") or ""),
           "citations": [str(c) for c in (result.get("citations") or [])],
           "confidence": float(result.get("confidence") or 0.0)}
    if not res["answer"]:
        res["citations"] = []
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    os.replace(tmp, path)
    print(json.dumps(res, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="?", const="/app/corpus")
    ap.add_argument("--query")
    ap.add_argument("--question")
    ap.add_argument("--query-id", "--query_id", "--qid", "--id", "--question-id", dest="qid")
    ap.add_argument("--query-file", "--input", "--input-file", "--questions", dest="qfile")
    ap.add_argument("--output-dir", "--output_dir", default="/app/output")
    args, rest = ap.parse_known_args()

    if args.index is not None:
        try:
            r = call({"cmd": "index", "path": os.path.abspath(args.index)},
                     time.monotonic() + INDEX_BUDGET_S)
            print(json.dumps(r, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001
            print(f"[app] index error: {e}", file=sys.stderr)
        if args.query is None and args.question is None and args.qfile is None:
            return

    if args.qfile:
        items = load_query_file(args.qfile)
    elif args.query and os.path.isfile(args.query):
        items = load_query_file(args.query)
    else:
        q = args.question or args.query or " ".join(r for r in rest if not r.startswith("-"))
        qid = args.qid or (args.query if args.question and args.query else None) or "query"
        items = [(qid, q)]
    if args.qid and len(items) == 1:
        items = [(args.qid, items[0][1])]

    for qid, question in items:
        result = EMPTY
        try:
            result = call({"cmd": "query", "question": question},
                          time.monotonic() + QUERY_BUDGET_S)
        except Exception as e:  # noqa: BLE001
            print(f"[app] query error: {e}", file=sys.stderr)
        write_output(args.output_dir, qid, result)


if __name__ == "__main__":
    main()
