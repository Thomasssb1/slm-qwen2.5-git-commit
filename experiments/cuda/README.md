# CUDA experiment container

This is the isolated NVIDIA CUDA environment for the base-model, SFT, and DPO work. It is separate from the rest of the project so that local development never installs CUDA or training dependencies.

## One-command Vast smoke test

Requires a funded Vast account with `VAST_API_KEY` set, Terraform < 2.0,
OpenSSH, and `notebooks/prepared/train.parquet`.
Run from the repository root:

```console
uv run python experiments/cuda/run_vast.py
```

Builds the image locally, rents a GPU VM, tests one training row, downloads
`cuda-smoke.json`, and attempts teardown. Reports, logs, and recovery state are
saved privately under `experiments/cuda/artifacts/vast/`.

Defaults to the cheapest matching RTX 3090 or RTX PRO 4000, up to $0.50/hour
including storage, with a 60-minute execution timeout.

- `--dry-run`: preview settings without renting.
- `--gpu NAME`: replace the default GPU allowlist; repeat for multiple models.
- `--max-hourly-price USD`: set the maximum hourly price, including storage.
- `--timeout-minutes N`: set the rental execution timeout.
- `--dataset PATH`: use a different training Parquet file.

Keep the run directory until cleanup succeeds. Retry failed teardown with:

```console
uv run python experiments/cuda/run_vast.py --cleanup PATH_TO_RUN_DIRECTORY
```

If provisioning was interrupted, also check the Vast console for leftover
resources with the run's `qwen-cuda-...` label.

## Rental-GPU smoke test

Run this only on a machine with a compatible NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit. The smoke test does not download Qwen or perform model inference. It reads one batch from `train.parquet`, verifies the requested GPU is usable with a CUDA tensor operation, prints diagnostics, and writes `cuda-smoke.json` to the artifact directory.

In PowerShell, from the repository root:

```powershell
$env:QWEN_COMMIT_DATASET_DIR = (Resolve-Path notebooks/prepared).Path
$env:QWEN_COMMIT_ARTIFACT_DIR = (
    New-Item -ItemType Directory -Force experiments/cuda/artifacts
).FullName
docker compose --file experiments/cuda/compose.yaml build
docker compose --file experiments/cuda/compose.yaml run --rm cuda-experiment
```

`QWEN_COMMIT_ARTIFACT_DIR` must be a private, writable directory. Do not commit it or the generated smoke report. A CUDA-unavailable result is a failure, rather than a CPU fallback: the point of this check is to validate the rental GPU before later work consumes it.
