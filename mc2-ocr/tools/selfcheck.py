#!/usr/bin/env python3
"""Run app.py over a folder of images and score against expected answers,
using the official normalization.  Works inside the container or in the notebook.

  python3 tools/selfcheck.py samples/ samples/expected.json [--app /app/app.py]
"""
import argparse, json, os, subprocess, sys, time


def norm(s):
    s = s.upper()
    for ch in " \t\r\n-.·_":
        s = s.replace(ch, "")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir")
    ap.add_argument("expected_json")
    ap.add_argument("--app", default="/app/app.py")
    ap.add_argument("--output-dir", default="/app/output")
    a = ap.parse_args()

    expected = json.load(open(a.expected_json, encoding="utf-8"))
    score, slowest = 0, 0.0
    for name, want in expected.items():
        path = os.path.join(a.images_dir, name)
        if not os.path.exists(path):
            print(f"SKIP  {name} (missing)"); continue
        t0 = time.monotonic()
        subprocess.run([sys.executable, a.app, "--input-image", path,
                        "--output-dir", a.output_dir], capture_output=True, timeout=30)
        dt = time.monotonic() - t0
        slowest = max(slowest, dt)
        out = os.path.join(a.output_dir, os.path.splitext(name)[0] + "_output.json")
        got = json.load(open(out, encoding="utf-8"))["text"] if os.path.exists(out) else "<NO FILE>"
        ok = norm(got) == norm(want)
        score += 20 * ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<22} {dt:5.2f}s  got={got!r}  want={want!r}")
    print(f"\nScore: {score}/{20 * len(expected)}   slowest: {slowest:.2f}s (limit 30s)")


if __name__ == "__main__":
    main()
