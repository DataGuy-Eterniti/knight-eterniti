# FINAL stage must be the mandated base (checked by layer identity). Never --squash.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \
    MODEL_DIR=/models/qwen3-vl-8b \
    OCR_SOCKET=/tmp/ocr.sock

WORKDIR /app

# 1) Dependencies. The torch stack is pinned to the ROCm builds already in the
#    base image, so pip cannot silently swap in a CUDA wheel. Build fails if it does.
COPY tools/pin_rocm.py /tmp/pin_rocm.py
COPY requirements.txt /app/requirements.txt
RUN python3 /tmp/pin_rocm.py write /tmp/rocm-pins.txt && \
    pip install --no-cache-dir -c /tmp/rocm-pins.txt -r /app/requirements.txt && \
    python3 /tmp/pin_rocm.py verify

# 2) Model weights baked in (~17 GB, under the 60 GiB cap). Kept in its own layer
#    so code changes don't re-download it.
COPY tools/download_model.py /tmp/download_model.py
RUN python3 /tmp/download_model.py

# 3) Application code (changes most often, so it goes last).
COPY app/ /app/
RUN mkdir -p /app/input /app/output

# Resident server: loads the model once and keeps the container alive.
# The harness then runs `python3 /app/app.py --input-image ...` via exec.
CMD ["python3", "/app/server.py"]
