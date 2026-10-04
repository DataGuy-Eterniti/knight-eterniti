# Knight Eterniti: OCR + RAG on AMD ROCm

Solo entry for the **Lablab x AMD AI Academy Challenge** (Sep–Dec 2026). Each mini-challenge lives in its own folder
and ships as a Docker image built on the mandated base,
`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`.

| Folder | Mini-challenge | Status |
|---|---|---|
| [`mc2-ocr/`](mc2-ocr/) | 2: Optical Character Recognition (plates and road signs) | Submitted |
| `mc3-rag/` | 3: Retrieval-Augmented Generation | In progress |

## Mini-Challenge 2: OCR

Reads US and Chinese licence plates and road signs from PNG, JPEG and TIFF images and writes
`/app/output/<name>_output.json` with `{"text": ..., "confidence": ...}`.

**How it works**

- **Model:** Qwen3-VL-8B-Instruct in BF16 via Hugging Face Transformers on ROCm. Weights are baked into the image.
- **Resident server** (`app/server.py`): loads the model once at container start, warms up the GPU, then listens on a
  Unix socket. The socket only appears once the model is ready.
- **Thin client** (`app/app.py`): standard library only, so it starts in milliseconds. It always writes a valid JSON
  file, even if the server is down, so one bad image can't end the run.
- **Prompt rules:** registration number only on US plates (no state banners or slogans), province character and letter
  kept on Chinese plates, every printed word kept on signs, multi-line text joined top to bottom, no added units.
- **Post-processing** (`app/postprocess.py`): strips leaked state names, slogans and URLs; normalises full-width
  characters and dot variants; handles 16-bit, transparent, multi-page and EXIF-rotated images.
- **ROCm protection** (`tools/pin_rocm.py`): pip installs run against a constraints file pinned to the base image's
  ROCm torch, and the build fails if torch is ever replaced by a CUDA build.

**Measured on an AMD Instinct MI300X (ROCm 10.0)**

| Check | Result |
|---|---|
| Public samples | 10/10 (200/200) |
| Model load | 8 s |
| Per image | 0.26–0.46 s (limit 30 s) |
| Peak VRAM | 18.8 GB (limit 48 GiB) |
| Image size | 45.7 GiB uncompressed (limit 60 GiB) |

These are results on the published samples, not the hidden graded set.

**Run it yourself:** see [`mc2-ocr/README.md`](mc2-ocr/README.md) for build, test and self-check commands.
The ten public sample images and their expected answers are in `mc2-ocr/samples/`.
