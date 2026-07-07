# VoxCPM2 Local Runbook

This file documents the local production-style setup for this repository.

## Quick Start

1. Make sure the local model files exist under:

   ```text
   pretrained_models/VoxCPM2/
   ```

2. Start the WebUI:

   ```bat
   start_server.bat
   ```

3. Open the local URL:

   ```text
   http://127.0.0.1:8808
   ```

## Runtime Requirements

- NVIDIA GPU is recommended. This machine has an RTX 4090 with 24GB VRAM.
- The local runtime currently uses:

  ```text
  E:\ziyuan\VoxCPM-2.0.2-20260505\jian27\python.exe
  ```

- The launcher also supports overriding the Python path:

  ```bat
  set VOXCPM_PYTHON=D:\path\to\python.exe
  start_server.bat
  ```

## Required Model Files

The launcher checks these files before starting:

```text
pretrained_models/VoxCPM2/config.json
pretrained_models/VoxCPM2/tokenizer.json
pretrained_models/VoxCPM2/model.safetensors
pretrained_models/VoxCPM2/audiovae.pth
```

The model directory is intentionally ignored by Git. Do not upload these large model files to GitHub.

## Network Policy

The WebUI binds to `127.0.0.1` by default, so only the local machine can access it.

Use LAN access only when you really need it:

```bat
set VOXCPM_HOST=0.0.0.0
start_server.bat
```

## Common Problems

### Port 8808 is already in use

Set another port:

```bat
set VOXCPM_PORT=8810
start_server.bat
```

### CUDA out of memory

Close other GPU-heavy software and start again. This includes other AI apps, games, video tools, and browser GPU workloads.

### Missing model file

Check `pretrained_models/VoxCPM2/` and restore the missing file shown by the launcher.

## Git Policy

- Source code, docs, configs, and small examples are tracked.
- `pretrained_models/` is ignored.
- Generated caches and local outputs should not be committed.
