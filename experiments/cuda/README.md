# CUDA experiment container

This is the isolated NVIDIA CUDA environment for the base-model, SFT, and DPO work. It is separate from the rest of the project so that local development never installs CUDA or training dependencies.

## One-command Vast smoke test

From the repository root:

```console
uv run python experiments/cuda/run_vast.py
```

The runner builds the image locally, exports one row from
`notebooks/prepared/train.parquet`, provisions a GPU VM through Terraform, uploads
the image and sample over SSH, runs the existing Compose smoke test, downloads
`cuda-smoke.json`, and destroys the VM and its temporary account SSH key. No
container registry is required. This command rents a paid GPU once local
preparation has succeeded.

One-time prerequisites:

- A funded personal Vast account and a personal API key available in the
  `VAST_API_KEY` environment variable (`VASTAI_API_KEY` is also accepted).
- Terraform 1.6 or newer (below 2.0), Docker with its Linux engine running, and
  OpenSSH's `ssh`, `scp`, and `ssh-keygen` commands on your PATH.
- The prepared training Parquet file. Use `--dataset PATH` if it is elsewhere.

Terraform uses the pinned [`berops/vastai` 0.1.1 provider](https://github.com/berops/terraform-provider-vastai/tree/v0.1.1)
to manage the VM and temporary public-key registration. Provider checksums are
locked for Windows and Linux on AMD64. The Python runner handles the build,
file transfer, job, report validation, and teardown sequence. It passes the API
key to Terraform through `VASTAI_API_KEY`, without writing it to configuration
files or passing it to Docker/SSH subprocesses. No Vast CLI is needed.

Defaults select one verified RTX 3090 or RTX PRO 4000 VM with at least 24 GB VRAM,
CUDA 13.2 support, 98% host reliability, and 80 GB disk. Offers must cost at most $0.50/hour
including the requested storage, with transfer rates no higher than $0.01/GB in
either direction. Bandwidth and taxes are additional to the hourly quote. If no
offer matches, the command fails without increasing the budget.

The GPU names form an allowlist. Terraform searches all allowed models together
and tries the five cheapest matching offers in ascending hourly-price order,
including storage, until one can be rented. List order does not imply preference;
"best" means cheapest while meeting every requirement, not highest performance.

Inspect the intended settings without starting Docker, accessing Vast, or renting:

```console
uv run python experiments/cuda/run_vast.py --dry-run
```

Repeat `--gpu` to replace the default allowlist (one `--gpu` still selects a single
model). Use exact Vast model names; blank names are rejected:

```console
uv run python experiments/cuda/run_vast.py --gpu "RTX 3090" --gpu "RTX PRO 4000" --max-hourly-price 0.40 --timeout-minutes 45 --dataset notebooks/prepared/train.parquet
```

The corresponding Terraform input is `gpu_names = ["RTX 3090", "RTX PRO 4000"]`
in `variables.tf`. Custom models must still meet the CUDA, VRAM, VM, and price
requirements. `run.json` records the model and hourly price actually rented.

The default rental timeout is 60 minutes, starting immediately before Terraform
apply. Local preparation is outside that window. Cleanup has a separate timeout,
so this is an execution deadline, not a guaranteed maximum bill.

Each run has its own gitignored directory under `experiments/cuda/artifacts/vast/`:
the report, operation logs, image archive, single-row sample, temporary private
SSH key, a `terraform/` directory containing the configuration and state, and
`run.json` with the instance details, run label, and cleanup status.
Only `id`, `subject`, and `diff` are exported from the training row;
validation/test files, provenance, and split
metadata are not uploaded. The build context excludes these private artifacts.
Keep this directory private and retain its Terraform state and `run.json` until
cleanup is confirmed. State may contain sensitive provider-generated values.

After apply starts, success, normal errors, timeouts, and caught Ctrl+C all
attempt Terraform destroy. The command fails if cleanup cannot be confirmed and
prints an exact recovery command:

```console
uv run python experiments/cuda/run_vast.py --cleanup PATH_TO_RUN_DIRECTORY
```

Each run has isolated local Terraform state; cleanup reuses that saved directory.
It refuses recovery if the state or configuration files are missing. Terraform
destroy must succeed and leave no tracked resources before cleanup is marked complete.

Power loss, forced termination, or provider errors can prevent cleanup. If apply
did not finish, the runner destroys recorded resources but leaves cleanup marked
failed: a lost creation response can leave a resource missing from Terraform state.
Check the Vast console for the run's `qwen-cuda-...` label and temporary public key
in `id_ed25519.pub`, and delete any leftovers. Repeating `--cleanup` cannot prove
that untracked resources are gone; that interrupted run keeps its failed status.
For a failed destroy after a completed apply, retry `--cleanup` to finish teardown.
Stopping a VM is not equivalent to destroying it: storage can keep incurring charges.

The runner pins the first SSH host key it sees in a per-run `known_hosts` file and
rejects subsequent changes. It uses Vast's
[Ubuntu VM mode](https://docs.vast.ai/guides/instances/virtual-machines), which
supports Docker inside the VM. No model is downloaded or inferred in this step.

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
