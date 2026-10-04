"""Download model weights to MODEL_DIR. Used at build time and in the notebook.

Notebook example (survives session turn-off):
  MODEL_DIR=/persistent/models/qwen3-vl-8b python3 tools/download_model.py
  (use /workspace instead of /persistent if your URL contains "rgapi-hackathon")
"""
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
model_dir = os.environ.get("MODEL_DIR", "/models/qwen3-vl-8b")

snapshot_download(
    model_id,
    local_dir=model_dir,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"],
)
print(f"Downloaded {model_id} -> {model_dir}")
