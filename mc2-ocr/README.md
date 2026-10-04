# Mini-Challenge 2: OCR on AMD ROCm

The pipeline runs Qwen3-VL-8B-Instruct (BF16, about 18 GB VRAM) in a resident server. `app.py` is a thin
standard-library client that the grader calls once per image.

```
app/app.py          grader entry point, writes /app/output/<stem>_output.json (always)
app/server.py       loads model once, warms GPU, serves over /tmp/ocr.sock
app/postprocess.py  strips state banners and slogans, normalises Chinese plates and dots
tools/pin_rocm.py   keeps pip from replacing ROCm torch with a CUDA build
tools/download_model.py
tools/selfcheck.py  scores a folder of images using the official normalisation
samples/expected.json   rename your sample files to match, or edit the keys
```

## 1. Develop in the hackathon notebook (no Docker there)

Open https://notebooks.amd.com/hackathon in a fresh tab. Do not use the /ai-academy page.
Check the redirected URL. If it contains `jupyter-hack-`, persistent storage is `/persistent`.
If it contains `rgapi-hackathon-`, persistent storage is `/workspace`. The commands below
assume `/persistent`.

```bash
cd /persistent && unzip amd-mc2-ocr.zip && cd amd-mc2-ocr
python3 tools/pin_rocm.py write /tmp/pins.txt
pip install -c /tmp/pins.txt -r requirements.txt     # re-run every new session
python3 tools/pin_rocm.py verify                     # must print a rocm/hip torch

export MODEL_DIR=/persistent/models/qwen3-vl-8b      # downloads once, survives turn-off
python3 tools/download_model.py

python3 app/server.py > server.log 2>&1 &            # wait for "READY" in server.log
python3 tools/selfcheck.py samples/ samples/expected.json \
        --app app/app.py --output-dir /tmp/out
```

`server.log` prints the raw model output for each image, which is where you tune the prompt.
To try a smaller model, set `MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct` and use a different
`MODEL_DIR`. The image budget is 3 h/day, so press "Turn-off Session" when you stop.

## 2. Build and test the container (needs a machine with Docker)

The notebook can't run Docker. Use an AMD Developer Cloud GPU droplet, which is paid
with your $100 credits and has fast bandwidth for a roughly 45 GB image.

```bash
docker build -t <dockerhub-user>/mc2-ocr:v1 .          # never use --squash
docker run --rm <dockerhub-user>/mc2-ocr:v1 python3 -c "import torch;print(torch.__version__)"
docker image inspect <dockerhub-user>/mc2-ocr:v1 --format '{{.Size}}'   # bytes, < 60 GiB

docker run -d --name ocr --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined <dockerhub-user>/mc2-ocr:v1
docker logs -f ocr                                      # wait for READY, note the time
docker cp samples/. ocr:/app/input/
docker exec ocr python3 /app/app.py --input-image /app/input/sample_07.tif
docker cp tools/selfcheck.py ocr:/tmp/ && docker cp samples/expected.json ocr:/tmp/
docker exec ocr python3 /tmp/selfcheck.py /app/input /tmp/expected.json
watch -n1 'amd-smi metric --mem | grep USED_VRAM'       # peak must stay 1–48 GiB
```

## 3. Submit

Push to a public registry. Docker Hub reuses the base image's layers, so you only upload
your own. Submit the image reference on lablab. Don't put the reference in a public repo,
and don't put tokens or `.env` files in the image.

```bash
docker push <dockerhub-user>/mc2-ocr:v1
```

## Where to improve

* The graded set is harder than the samples. Make your own stress set by adding blur, noise,
  glare, low light and perspective to the samples. Pull real plate and sign photos too.
* If a degraded image misreads, try a second pass at higher `OCR_MAX_SIDE` and keep the answer
  with the higher confidence. Each call has about 0.3–1 s of headroom against the 30 s limit.
* Check new raw outputs in `server.log` against `postprocess.py`. The slogan list is not
  exhaustive.
