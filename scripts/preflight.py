from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path


REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors",
    "audiovae.pth",
)


def _fail(message: str) -> int:
    print(f"[ERROR] {message}", file=sys.stderr)
    return 1


def _check_model_files(model_dir: Path) -> int:
    if not model_dir.exists():
        return _fail(f"Model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        return _fail(f"Model path is not a directory: {model_dir}")

    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        print("[ERROR] Missing required model files:", file=sys.stderr)
        for name in missing:
            print(f"  - {model_dir / name}", file=sys.stderr)
        return 1
    return 0


def _check_port(host: str, port: int) -> int:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        result = sock.connect_ex((bind_host, port))
    if result == 0:
        return _fail(f"Port {port} is already in use on {bind_host}. Set VOXCPM_PORT to another port.")
    return 0


def _check_torch() -> int:
    try:
        import torch
    except Exception as exc:
        return _fail(f"PyTorch import failed: {exc}")

    print(f"[OK] PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        print(f"[OK] CUDA: {device_name} ({free_gb:.1f}GB free / {total_gb:.1f}GB total)")
    else:
        print("[WARN] CUDA is not available. The model may run very slowly on CPU.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxCPM2 local startup preflight")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8808, type=int)
    args = parser.parse_args()

    print(f"[OK] Python: {sys.version.split()[0]}")

    checks = (
        _check_model_files(args.model_dir),
        _check_port(args.host, args.port),
        _check_torch(),
    )
    if any(code != 0 for code in checks):
        return 1

    print("[OK] Preflight checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
