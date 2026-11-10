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

## Run the validation split

```console
uv run python experiments/cuda/run.py --check validation --reviewed-validation notebooks/prepared/reviewed_validation.parquet
```

This reads every row of `notebooks/prepared/validation.parquet` in its saved
order, loads the pinned model once, and calls the same single-diff generation
function once per row. `--dataset PATH` can select another prepared validation
file. The test split is not opened. The optional `--reviewed-validation` file
marks approved references without filtering the full split; its IDs, subjects,
and diffs must match the selected validation data. Without it, review status is
recorded as unknown.

Only IDs and diffs are uploaded. Reference subjects, approval flags, and source
provenance stay local. Each private run directory contains:

- `validation-input.parquet`: the exact uploaded IDs and diffs, with source metadata removed.
- `validation-references.parquet`: local reference subjects and approval flags.
- `validation.json`: model diagnostics and every raw generation result, including the full prompt/configuration.
- `validation-predictions.parquet`: predictions joined to local diffs and references, with raw output, accepted subject or null, format errors, token counts, truncation, and timing.
- `run.json`: source/input/reviewed-file hashes, row counts, image digest, and cleanup status.

Collection verifies the input hash and complete ordered ID list before writing
the joined Parquet file. Format violations remain recorded predictions; runtime
failures or incomplete results fail the run. The existing 2 MiB result-envelope
limit applies. Prediction artifacts remain gitignored, and the rental and
temporary bucket are removed after collection. Compliance summaries and semantic
review are separate from this prediction-generation step.

## Evaluate the baseline

Run the local evaluator against a completed validation run:

```console
uv run python experiments/cuda/evaluate.py experiments/cuda/artifacts/vast/RUN_ID
```

It verifies the saved predictions against the validated generation report and local input/reference join, then writes private `evaluation.json` and `evaluation.md` reports. They include mechanical formatting compliance, overlapping error counts, truncation counts, and generation latency for the full validation split and the approved-reference subset. No model is loaded, no cloud service is contacted, and no test data is read.

Latency is CUDA-synchronized `model.generate` time, including the first call. Median and p95 use linear interpolation. Tokenization, model loading, provisioning, and transfer are excluded; model download/load time is reported separately. A formatting pass does not establish semantic correctness or detect every possible explanation.

The evaluator also writes `semantic-review-template.json` with the first 12 approved-reference rows sorted by ID, including format failures. Copy it to `semantic-review.json`, identify the reviewer, and fill each `rating` and evidence-based `rationale` using the included rubric: `accurate`, `partial`, `incorrect`, or `uncertain`. Inspect the saved diff itself; a historical reference is context, not ground truth. Truncated inputs are flagged because the model did not see the full diff. The sample is descriptive, not an estimate of full-split accuracy.

```console
uv run python experiments/cuda/evaluate.py experiments/cuda/artifacts/vast/RUN_ID --review experiments/cuda/artifacts/vast/RUN_ID/semantic-review.json
```

Reviews must match the prediction artifact's SHA-256 and the complete ordered sample. Existing templates and prediction artifacts are preserved. If no references were approved, subset metrics are unavailable and no semantic sample is selected. Keep all reports, diffs, and review annotations in the ignored run directory; the PR can contain aggregate findings without source examples.

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
- `--dataset PATH`: use a different prepared training or validation Parquet file for the selected check.
- `--reviewed-validation PATH`: mark an approved reference subset for a validation run.
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
