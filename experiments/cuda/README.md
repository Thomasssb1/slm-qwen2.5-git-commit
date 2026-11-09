# CUDA experiment container

This is the isolated NVIDIA CUDA environment for the base-model, SFT, and DPO work. It is separate from the rest of the project so that local development never installs CUDA or training dependencies.

## One-command Vast smoke test

Requires a funded Vast account, Terraform < 2.0, AWS credentials with S3 access,
and `notebooks/prepared/train.parquet`. Copy `.env.example` to `.env` in the
repository root and set `VAST_API_KEY`. The runner loads this gitignored file
automatically; existing environment variables take precedence.
Log Docker into GHCR using a token with `read:packages` and `write:packages`.
Use that same token in `.env` for `VAST_IMAGE_LOGIN`: `-u USER -p TOKEN ghcr.io`, so Vast
can pull the private image at `ghcr.io/thomasssb1/qwen-commit-cuda`.
Run from the repository root:

```console
uv run python experiments/cuda/run.py
```

Builds and pushes the image, creates a temporary private S3 bucket, and launches
the image directly on Vast. The container tests one training row using expiring
transfer URLs. The runner saves `cuda-smoke.json` and `smoke.log`, then destroys
the rental and bucket, including its objects. The registry image is retained.
The bucket is unversioned; leftover objects expire after seven days.
Artifacts and recovery state stay under `experiments/cuda/artifacts/vast/`.

Defaults to the cheapest matching RTX 3090 or RTX PRO 4000, up to $0.50/hour
including storage, with a 60-minute execution timeout.

- `--gpu NAME`: replace the default GPU allowlist; repeat for multiple models.
- `--max-hourly-price USD`: set the maximum hourly price, including storage.
- `--timeout-minutes N`: set the rental execution timeout.
- `--dataset PATH`: use a different training Parquet file.
- `--image-repository REPO`: change the registry repository; also accepts `VAST_IMAGE_REPOSITORY`.
- `--aws-region REGION`: choose the bucket region; defaults to the AWS configuration or `eu-west-2`.

Keep the run directory until cleanup succeeds. Retry failed teardown with:

```console
uv run python experiments/cuda/run.py --cleanup PATH_TO_RUN_DIRECTORY
```

If provisioning was interrupted, check Vast and S3 for leftovers named
`qwen-cuda-...` from the run. Keep the runner open until teardown finishes.

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
