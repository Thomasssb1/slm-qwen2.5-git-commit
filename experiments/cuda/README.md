# CUDA experiment container

This is the isolated NVIDIA CUDA environment for the base-model, SFT, and DPO work. It is separate from the rest of the project so that local development never installs CUDA or training dependencies.

## Rental-GPU smoke test

Run this only on a machine with a compatible NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit. The smoke test does not download Qwen or perform model inference. It reads one batch from `train.parquet`, verifies the requested GPU is usable with a CUDA tensor operation, prints diagnostics, and writes `cuda-smoke.json` to the artifact directory.

In PowerShell, from the repository root:

```powershell
$env:QWEN_COMMIT_DATASET_DIR = (Resolve-Path prepared).Path
$env:QWEN_COMMIT_ARTIFACT_DIR = (
    New-Item -ItemType Directory -Force experiments/cuda/artifacts
).FullName
docker compose --file experiments/cuda/compose.yaml build
docker compose --file experiments/cuda/compose.yaml run --rm cuda-experiment
```

`QWEN_COMMIT_ARTIFACT_DIR` must be a private, writable directory. Do not commit it or the generated smoke report. A CUDA-unavailable result is a failure, rather than a CPU fallback: the point of this check is to validate the rental GPU before later work consumes it.
