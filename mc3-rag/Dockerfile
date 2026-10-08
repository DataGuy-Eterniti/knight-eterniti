# FINAL stage must be the mandated base (checked by layer identity). Never --squash.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    LLM_DIR=/models/qwen3-vl-8b \
    EMB_DIR=/models/bge-small \
    RAG_SOCKET=/tmp/rag.sock

WORKDIR /app

# 1) Dependencies, with the base image's ROCm torch stack pinned so pip cannot swap in a
#    CUDA build. The build fails if torch is no longer the ROCm build afterwards.
COPY requirements.txt /app/requirements.txt
RUN python3 -c 'import importlib.metadata as m; w={"torch","torchvision","torchaudio","triton","pytorch-triton-rocm"}; print("\n".join(sorted({d.metadata["Name"]+"=="+d.version for d in m.distributions() if (d.metadata["Name"] or "").lower() in w})))' > /tmp/rocm-pins.txt && \
    cat /tmp/rocm-pins.txt && \
    pip install --no-cache-dir -c /tmp/rocm-pins.txt -r /app/requirements.txt && \
    python3 -c 'import sys, torch, torchvision; print("torch", torch.__version__, "torchvision", torchvision.__version__); sys.exit(0 if "rocm" in torch.__version__ else 1)'

# 2) Weights baked in (~17 GB + 130 MB), in their own layer so code edits rebuild in seconds.
RUN python3 -c 'from huggingface_hub import snapshot_download as d; d("Qwen/Qwen3-VL-8B-Instruct", local_dir="/models/qwen3-vl-8b", allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","*.model","*.tiktoken"]); d("BAAI/bge-small-en-v1.5", local_dir="/models/bge-small", allow_patterns=["*.json","*.safetensors","*.txt"])'

# 3) Code last.
COPY app/ /app/
RUN mkdir -p /app/output

# Resident server keeps the container alive; restarted (index restored from cache) if it dies.
CMD ["sh", "-c", "while true; do python3 /app/server.py; echo '[supervisor] server exited, restarting'; sleep 2; done"]
