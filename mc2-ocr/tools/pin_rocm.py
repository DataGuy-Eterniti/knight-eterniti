"""Pin the base image's ROCm torch stack, and verify it survived pip.

  python3 pin_rocm.py write /tmp/rocm-pins.txt   -> constraints file for pip -c
  python3 pin_rocm.py verify                     -> exit 1 unless torch is a ROCm build
"""
import importlib.metadata as md
import sys

PKGS = ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm")


def write(path):
    lines = []
    for p in PKGS:
        try:
            lines.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            pass
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Pinned:", *lines, sep="\n  ")


def verify():
    import torch
    v = torch.__version__
    hip = getattr(torch.version, "hip", None)
    print(f"torch {v}  hip={hip}")
    if "rocm" not in v and not hip:
        print("ERROR: torch is no longer the ROCm build", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "write":
        write(sys.argv[2])
    else:
        verify()
