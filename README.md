# Knight Eterniti: OCR + RAG on AMD ROCm

Solo entry for the **Lablab x AMD AI Academy Challenge** (Sep–Dec 2026). Each mini-challenge is a
Docker image built on the mandated base `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`,
with model weights baked in so it runs with no network.

| Folder | Mini-challenge | Public samples |
|---|---|---|
| [`mc2-ocr/`](mc2-ocr/) | 2: OCR of licence plates and road signs | 10/10 (200/200) |
| [`mc3-rag/`](mc3-rag/) | 3: RAG over a mixed corpus, with exact citations | 10/10 (200/200) |

Both use the same design. A **resident server** loads the model once at container start and listens
on a unix socket. `app/app.py` is a **standard-library client** that starts in milliseconds and always
writes a valid JSON file, even if the server fails. The grader starts a new process for every call,
so loading the model inside `app.py` would blow the 30-second budget.

## Mini-Challenge 2: OCR

Qwen3-VL-8B-Instruct (BF16) reads US and Chinese plates and road signs from PNG, JPEG and TIFF.

- The prompt encodes the transcription rules: no state banners or slogans, keep Chinese province
  prefixes, join multi-line signs top to bottom, never add units.
- Deterministic post-processing strips leaked banners and URLs, and normalises full-width characters and dots.
- The build pins torch to the base image's ROCm build so pip cannot replace it with a CUDA wheel.

| Check (MI300X, ROCm 10.0) | Result |
|---|---|
| Public samples | 10/10 |
| Per image | 0.26–0.46 s (limit 30 s) |
| Peak VRAM | 18.8 GB (limit 48 GiB) |
| Image size | 45.7 GiB uncompressed (limit 60 GiB) |

## Mini-Challenge 3: RAG

Answers questions over PDF, DOCX (with tables), XLSX (all sheets), CSV, logs, code and images, and
cites the exact set of files each answer needs.

- **Index:** each file is parsed in an isolated worker process, so unreadable, encrypted or
  unknown files are skipped without stopping the walk. Images and scanned pages are read by
  Qwen3-VL. Withdrawn and superseded revisions are kept out of the model's context.
- **Retrieve:** identifier-aware BM25 fused with bge-small embeddings. Rare identifiers such as
  ticket numbers and error codes are followed across files for multi-hop questions.
- **Cite by necessity:** a file is kept only if it holds the value or supplied the identifier
  that leads to it. An answer found in no document is refused with an empty answer.

| Check (MI300X, ROCm 10.0) | Result |
|---|---|
| Public samples | 10/10, exact citation sets |
| Startup (model load + index) | about 12 s (limit 10 min) |
| Per question | 1.0–1.9 s (limit 30 s) |
| Peak VRAM | 20.0 GB (limit 48 GiB) |
| Image size | 45.8 GiB uncompressed (limit 60 GiB) |

Results are on the published samples, not the hidden graded sets. Build and test steps are in each
folder's README.
