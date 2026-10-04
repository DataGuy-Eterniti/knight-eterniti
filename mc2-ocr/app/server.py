#!/usr/bin/env python3
"""Resident OCR server. Loads the VLM once, warms the GPU, then answers app.py
requests over a unix socket. The socket only appears AFTER the model is ready."""
import json
import math
import os
import socketserver
import sys
import threading
import time
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from postprocess import clean

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/qwen3-vl-8b")
SOCK = os.environ.get("OCR_SOCKET", "/tmp/ocr.sock")
MAX_SIDE = int(os.environ.get("OCR_MAX_SIDE", "1600"))
MIN_SIDE = int(os.environ.get("OCR_MIN_SIDE", "640"))

PROMPT = """You are a precise OCR engine for road imagery. Transcribe the main text in this image.

Rules:
- Vehicle licence plate: return ONLY the registration number. Do NOT include the state or country name, slogans, websites, county names, registration stickers or dealer frame text.
- Chinese licence plate: the leading province character and letter ARE part of the number (for example 京A·12345). Keep them, in Chinese characters.
- Road sign: return every word and number printed on the sign, exactly as printed. Read top to bottom, left to right, and join lines with single spaces. Never add units or words that are not printed (a plaque showing only 35 is just 35).
- Copy characters exactly. Do not correct spelling or guess hidden characters.
- No explanation.

Answer in exactly this format:
TYPE: <US_PLATE | CN_PLATE | OTHER_PLATE | SIGN>
TEXT: <transcription>"""


def load_image(path):
    img = Image.open(path)
    try:
        img.seek(0)  # multi-page TIFF: first frame
    except Exception:  # noqa: BLE001
        pass
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):  # 16-bit / float TIFF
        a = np.asarray(img).astype("float32")
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo + 1e-6) * 255.0
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
    longest = max(w, h)
    if longest > MAX_SIDE:
        s = MAX_SIDE / longest
    elif longest < MIN_SIDE:
        s = MIN_SIDE / longest  # upscale tiny crops so characters get enough patches
    else:
        s = 1.0
    if s != 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


class Engine:
    def __init__(self, model_dir):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.time()
        print(f"[server] torch {torch.__version__} hip={torch.version.hip}", flush=True)
        print(f"[server] device: {torch.cuda.get_device_name(0)}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        # Load on CPU then move: avoids the caching_allocator_warmup failure some
        # people hit on MI300X VF partitions when using device_map.
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        print(f"[server] model loaded in {time.time() - t0:.1f}s", flush=True)

    @torch.inference_mode()
    def read(self, path):
        img = load_image(path)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[img], return_tensors="pt").to("cuda")
        out = self.model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0, inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(seq, skip_special_tokens=True)

        logps = []
        eos_cfg = self.model.generation_config.eos_token_id
        eos = set(eos_cfg) if isinstance(eos_cfg, (list, tuple)) else (
            {eos_cfg} if eos_cfg is not None else set())
        for step_scores, tok in zip(out.scores, seq):
            tok = int(tok)
            if tok in eos:
                break
            p = torch.softmax(step_scores[0].float(), dim=-1)[tok].item()
            logps.append(math.log(max(p, 1e-9)))
        conf = math.exp(sum(logps) / len(logps)) if logps else 0.0

        text, kind = clean(raw)
        print(f"[server] {os.path.basename(path)} -> {kind} {text!r} "
              f"(conf {conf:.2f}) raw={raw!r}", flush=True)
        return text, round(conf, 4)

    def warmup(self, n=2):
        img = Image.new("RGB", (640, 320), "white")
        ImageDraw.Draw(img).text((200, 140), "STOP 65", fill="black")
        tmp = "/tmp/_warmup.png"
        img.save(tmp)
        for _ in range(n):
            self.read(tmp)


ENGINE = None
LOCK = threading.Lock()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline().decode("utf-8"))
            if ENGINE is None:
                raise RuntimeError("model not loaded")
            with LOCK:  # one GPU, one request at a time
                text, conf = ENGINE.read(req["image"])
            resp = {"text": text, "confidence": conf}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            resp = {"text": "", "confidence": 0.0, "error": str(e)}
        self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))


def main():
    global ENGINE
    try:
        ENGINE = Engine(MODEL_DIR)
        ENGINE.warmup()
    except Exception:  # noqa: BLE001
        # Keep the container alive anyway, so every call still gets a valid JSON file.
        traceback.print_exc()
        ENGINE = None

    if os.path.exists(SOCK):
        os.remove(SOCK)
    server = socketserver.ThreadingUnixStreamServer(SOCK, Handler)
    os.chmod(SOCK, 0o777)
    print(f"[server] READY on {SOCK}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
