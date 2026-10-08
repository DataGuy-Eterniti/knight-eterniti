# Mini-Challenge 3: RAG on AMD ROCm

Answers questions from a mixed-document corpus and cites the exact files each answer needs.
One container, built on the mandated `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`.

```
app/app.py        grader entry point (stdlib only): --index <dir> | --query ... ; always writes valid JSON
app/server.py     resident server: Qwen3-VL-8B (answers + reads images) + bge-small embedder + index
app/parsers.py    pdf, docx (incl. tables), xlsx (all sheets), csv, txt/log/code; runs in its own process
app/retrieval.py  identifier-aware BM25, chunking, revision families, value matching
```

**How it works**

1. **Index (startup budget).** Each file is parsed in a separate worker process, so a hanging or
   crashing file only costs that file. The run never fails on:
   - unreadable files (including under `--cap-drop DAC_OVERRIDE`)
   - encrypted files
   - unknown file types
   - empty folders

   Images, and PDFs with no text layer, are transcribed by the vision model.
   Withdrawn and superseded revisions are flagged and kept out of the model's context.
   The index is cached to `/tmp`, so a restarted server is ready in under a second.
2. **Retrieve.** BM25 and dense embeddings are fused. For multi-hop questions, a rare
   identifier found in the best chunks (a ticket number or error code, say) pulls in the
   chunks that define it.
3. **Answer.** The model gets only the retrieved documents and returns
   `{answer, sources}`. The answer is cleaned to the value alone: units, quotes and prose
   are stripped, and refusals become `""`.
4. **Cite by necessity.** A cited file is kept only if it contains the value, or if it
   supplied a rare identifier that sits next to the value. An answer found in no document
   is treated as invented and returns empty. If several cited files hold the same value,
   one short follow-up call to the model picks the file the question points at.

## Build and test on the droplet

```bash
unzip amd-mc3-rag.zip && unzip mc3-starter-kit.zip
cd mc3-starter-kit && ./setup.sh && cd ..       # recreates the empty dir + chmod 000 file
grep -rn "add_argument\|--query" mc3-starter-kit --include=*.py | head   # check the query flags

cd amd-mc3-rag
docker build -t mc3-rag:v1 . 2>&1 | tee build.log
docker run --rm mc3-rag:v1 python3 -c "import torch; print(torch.__version__)"   # must say +rocm

docker run -d --name rag --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined --cap-drop DAC_OVERRIDE \
  -v "$(realpath ../mc3-starter-kit/corpus)":/app/corpus:ro mc3-rag:v1
docker logs -f rag                                  # wait for READY and "indexed N files"
time docker exec rag python3 /app/app.py --index /app/corpus
docker exec rag python3 /app/app.py --query "Which firmware version fixed ticket ORR-1847?" --query-id q04
docker exec rag cat /app/output/q04_output.json
```

Then run the starter kit's own self-check against `mc3-rag:v1`, following the kit's README.
Check the uncompressed size and peak VRAM the same way as for MC2:

```bash
docker history --human=false --no-trunc --format '{{.Size}}' mc3-rag:v1 | awk '{s+=$1} END {printf "%.1f GiB\n", s/1024/1024/1024}'
watch -n1 'amd-smi metric --mem | grep USED_VRAM'
```

`docker logs rag` prints, for every question: the answer, the citations, the model's raw
sources and its one-line reasoning. Use that to diagnose any miss.
