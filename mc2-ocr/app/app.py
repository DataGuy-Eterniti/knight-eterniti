#!/usr/bin/env python3
"""Grader entry point:  python3 /app/app.py --input-image /app/input/image_01.png

Standard library only, so it starts in milliseconds. It sends the image path to the
resident server (server.py) over a unix socket and writes
/app/output/<stem>_output.json. It ALWAYS writes a valid file, even on failure.
"""
import argparse
import json
import os
import socket
import sys
import time

SOCK = os.environ.get("OCR_SOCKET", "/tmp/ocr.sock")
BUDGET_S = 25.0  # stay under the 30 s per-image limit


def ask_server(image_path, deadline):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(max(1.0, deadline - time.monotonic()))
        s.connect(SOCK)
        s.sendall((json.dumps({"image": image_path}) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", required=True)
    ap.add_argument("--output-dir", default="/app/output")
    args = ap.parse_args()

    image_path = os.path.abspath(args.input_image)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{stem}_output.json")

    result = {"text": "", "confidence": 0.0}
    deadline = time.monotonic() + BUDGET_S
    while time.monotonic() < deadline:
        try:
            r = ask_server(image_path, deadline)
            result = {"text": str(r.get("text", "")),
                      "confidence": float(r.get("confidence", 0.0))}
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.5)  # server still loading
        except Exception as e:  # noqa: BLE001
            print(f"[app] error: {e}", file=sys.stderr)
            break

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(tmp, out_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
