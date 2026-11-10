# CUDA experiment container

This is the isolated NVIDIA CUDA environment for the base-model, SFT, and DPO work. It is separate from the rest of the project so that local development never installs CUDA or training dependencies.

## Load Qwen on a rental GPU

With the Vast, AWS, and registry prerequisites below configured, run:

```console
uv run python experiments/cuda/run.py --check model
```

This downloads the public `Qwen/Qwen2.5-Coder-1.5B-Instruct` tokenizer and
Safetensors weights at revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`,
using Transformers 5.22.2 in the CUDA image. It loads the full model in BF16
onto the GPU, checks that every parameter is on CUDA, and exercises the
tokenizer. No prepared dataset or Hugging Face token is required. This check
does not generate subjects, evaluate the model, or train adapters.

The runner saves `model-load.json` and `smoke.log` under its private run
directory, then destroys the GPU rental and temporary S3 bucket. The report
records the model revision, library versions, parameter count, GPU placement,
load time, and allocated/peak GPU memory. A failed or partial load fails the
run. The container has up to 30 minutes to download and load the model, within
the runner's overall rental timeout; the same price limit and cleanup recovery
apply. Downloaded weights live in the rental's disposable model cache.

## Generate commit subjects

```console
uv run python experiments/cuda/run.py --check inference
```

This loads Qwen once and generates from three synthetic diffs: a single-file
change, a change with tests in a second file, and a diff longer than the input
budget. It repeats the first case to check token-for-token repeatability on
the same runtime. It does not read the training, validation, or test splits.

`generation.generate_subject(diff, model=model, tokenizer=tokenizer)` is the
reusable inference function. It uses the exact system prompt in
`commit_subject_prompt.txt`, with only the diff in the user message, and
Qwen's pinned chat template. The first 4,096 diff tokens are retained; the
system prompt and chat framing are not truncated. Greedy decoding uses
`do_sample=False`, `num_beams=1`, and `max_new_tokens=64`, with the model's EOS
tokens. These settings are explicit instead of inheriting Qwen's sampling
defaults.

The 64-token generation limit is separate from the 72-character acceptance
limit. Results preserve the decoded response in `raw_output`. An accepted
`subject` has surrounding spaces/tabs removed and must be nonempty, at most
72 Unicode code points, and contain no line breaks, surrounding quotes,
recognized Markdown, or common preambles. Responses that reach the token
limit without EOS also fail. Invalid responses have `subject=null` and
machine-readable `errors`; they are never clipped or rewritten. These checks
do not assess semantic correctness or detect every possible explanation.

The private `inference-smoke.json` report saves every raw response, accepted
subject, failure reason, original/retained token counts, truncation flag,
prompt text/hash, full generation configuration, and generation time alongside
the model-load evidence. A format failure is a recorded model result, while
runtime errors, inconsistent reports, or a repeatability failure fail the run.

## One-command Vast smoke test

Requires a funded Vast account, Terraform < 2.0, AWS credentials with S3 access,
and `notebooks/prepared/train.parquet`. Copy `.env.example` to `.env` in the
repository root and set `VAST_API_KEY`. The runner loads this gitignored file
automatically; existing environment variables take precedence.
Keep `.env` in the main checkout; `.worktreeinclude` lists it for worktree setup.
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

Transient Vast status-request timeouts, disconnected reads, rate limits, and
selected server errors get up to three recovery attempts, with 10/20/30-second
delays within the existing execution or cleanup deadline. After a readiness
poll fails, Terraform's `apply -refresh-only` reads the existing rental and
recovers its outputs. The runner then checks its price and waits for the validated
container report. It never repeats the rental-creation apply. Teardown retries
refresh/destroy and still requires empty Terraform state before confirming cleanup.
Authentication, configuration, host startup, and ambiguous creation failures
remain failures; the private per-operation JSON logs retain every attempt.

The polling regression can also exercise the pinned provider against a local
HTTP server. Set `VASTAI_TEST_PLUGIN_DIR` to an existing Terraform
`.terraform/providers` directory, then run `uv run pytest tests/test_vast_polling.py`.
This test uses no cloud resources or credentials. Without the provider directory,
only this integration test is skipped; the other polling tests still run.

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
