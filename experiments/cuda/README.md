# CUDA experiments

Run Qwen baseline inference on Vast and evaluate the saved predictions locally. Run commands from the repository root.

## Setup

- Docker with Linux containers, Terraform >=1.6 and <2.0, a funded Vast account, and AWS credentials with S3 access.
- Copy [`.env.example`](../../.env.example) to `.env` in the main checkout. Set `VAST_API_KEY` and `VAST_IMAGE_LOGIN`.
- Log Docker into GHCR with `docker login ghcr.io`. Use the same token in `VAST_IMAGE_LOGIN`, with `read:packages` and `write:packages` access.

## Run a check

```console
uv run python experiments/cuda/run.py --check model
```

| `--check` | Purpose | Required data |
| --- | --- | --- |
| `cuda` (default) | Test GPU access with one training row | `notebooks/prepared/train.parquet` |
| `model` | Load the pinned Qwen model in BF16 | None |
| `inference` | Generate from synthetic diffs and check repeatability | None |
| `validation` | Generate predictions for the full validation split | See below |

The runner builds and pushes the image, rents a GPU, collects results, then removes the rental and temporary S3 bucket. The registry image is retained. Reports and logs are saved under the gitignored `experiments/cuda/artifacts/vast/RUN_ID/` directory.

Defaults: RTX 3090 or RTX PRO 4000, at most **$0.50/hour** including storage, and a **60-minute** execution timeout. Override these with `--gpu`, `--max-hourly-price`, and `--timeout-minutes`. See `uv run python experiments/cuda/run.py --help` for all options.

Model revisions and dependencies are pinned in [model_spec.py](model_spec.py) and [requirements.txt](requirements.txt).

## Run the validation split

```console
uv run python experiments/cuda/run.py --check validation --reviewed-validation notebooks/prepared/reviewed_validation.parquet
```

Reads `notebooks/prepared/validation.parquet`; use `--dataset PATH` to override it. The optional reviewed file marks approved references without filtering the full split. Only IDs and diffs are uploaded; reference subjects stay local. The test split is not read.

Generation uses the [commit-subject prompt](commit_subject_prompt.txt), greedy decoding, the first 4,096 diff tokens, and up to 64 output tokens. Subjects must pass the one-line, 72-character format check; invalid outputs are retained without repair.

Results include `validation-predictions.parquet` with joined predictions, diffs, and references, plus `validation.json` with raw results and generation settings. Keep the run directory intact for evaluation.

## Evaluate the baseline

```console
uv run python experiments/cuda/evaluate.py experiments/cuda/artifacts/vast/RUN_ID
```

Writes `evaluation.json` with formatting results and generation-only latency. Timing excludes tokenization, model loading, provisioning, and transfer. Formatting does not measure correctness.

## Judge correctness locally

Open [judge_predictions.ipynb](../../notebooks/judge_predictions.ipynb) in the **Qwen Commit Review (ROCm)** kernel.

The pinned Qwen3-4B judge assesses every approved validation example. Oversized prompts and invalid responses are deferred for manual review. To review saved proposals without loading the GPU, run the setup blocks through **Calculate correctness and usable output**, then jump to **Review each verdict**.

`correctness-summary.json` keeps model and human scores separate:

- **Correctness:** accurate subjects divided by all approved validation examples.
- **Usable output:** subjects that are both accurate and format-valid, using the same denominator.

Rates remain unavailable while that grader has pending or uncertain examples. Human review is required to establish the correctness baseline.

## Recover interrupted runs

Keep the run directory until cleanup succeeds. Retry teardown with:

```console
uv run python experiments/cuda/run.py --cleanup experiments/cuda/artifacts/vast/RUN_ID
```

Check `run.json` for cleanup status. If provisioning was interrupted, also check Vast and S3 for resources named `qwen-cuda-...` from that run. The runner retries transient polling failures automatically.

<details>
<summary>Run the CUDA smoke test directly on an NVIDIA machine</summary>

Requires Docker, a compatible NVIDIA driver, NVIDIA Container Toolkit, and `notebooks/prepared/train.parquet`. In PowerShell:

```powershell
$env:QWEN_COMMIT_DATASET_DIR = (Resolve-Path notebooks/prepared).Path
$env:QWEN_COMMIT_ARTIFACT_DIR = (
    New-Item -ItemType Directory -Force experiments/cuda/artifacts
).FullName
docker compose --file experiments/cuda/compose.yaml build
docker compose --file experiments/cuda/compose.yaml run --rm cuda-experiment
```

This checks CUDA and writes `cuda-smoke.json`; it does not load Qwen.

</details>
