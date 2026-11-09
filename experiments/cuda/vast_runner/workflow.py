"""Publish, provision, collect, and clean up a CUDA smoke-test run."""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from . import RunError
from .artifacts import RUNS_DIRECTORY, training_sample, write_json
from .commands import Commands
from .image import publish
from .storage import Transfer
from .terraform import cleanup, terraform


def run_smoke(
    dataset: Path,
    *,
    gpu_names: list[str],
    max_hourly_price: float,
    timeout_minutes: float,
    image_repository: str = "ghcr.io/thomasssb1/qwen-commit-cuda",
    aws_region: str | None = None,
    check: str = "cuda",
    runs_directory: Path = RUNS_DIRECTORY,
    commands_type=Commands,
    transfer_type=Transfer,
) -> Path:
    if check not in {"cuda", "model"}:
        raise ValueError("Choose the cuda or model check.")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:12]
    directory = (runs_directory / run_id).resolve()
    directory.mkdir(parents=True, mode=0o700)
    commands = commands_type(directory)
    state = {
        "backend": "terraform",
        "run_id": run_id,
        "label": "qwen-cuda-" + run_id,
        "outcome": "preparing",
        "cleanup": "not-needed",
        "check": check,
    }
    write_json(directory / "run.json", state)
    print(f"Run artifacts: {directory}", flush=True)

    sample = None
    if check == "cuda":
        sample = directory / "train.parquet"
        training_sample(dataset, sample)
    transfer = transfer_type(aws_region)
    commands.run(["terraform", "version"])
    commands.run(["docker", "info"])
    infrastructure = directory / "terraform"
    infrastructure.mkdir()
    terraform(commands, "init", "-input=false", "-no-color", "-lockfile=readonly", timeout=600)
    terraform(commands, "validate", "-no-color")
    image = publish(commands, image_repository, run_id)
    state["image"] = image
    variables = {
        "run_label": state["label"],
        "gpu_names": gpu_names,
        "max_hourly_price": max_hourly_price,
        "rental_seconds": timeout_minutes * 60,
        "aws_region": transfer.region,
        "image": image,
        "launch": False,
    }
    write_json(infrastructure / "run.auto.tfvars.json", variables)
    state.update(outcome="running", cleanup="pending")
    write_json(directory / "run.json", state)
    deadline = time.monotonic() + timeout_minutes * 60

    def apply():
        # Reset before each apply: a lost creation response can leave untracked resources.
        state["apply_complete"] = False
        write_json(directory / "run.json", state)
        terraform(
            commands,
            "apply",
            "-auto-approve",
            "-input=false",
            "-no-color",
            timeout=deadline - time.monotonic(),
        )
        state["apply_complete"] = True
        write_json(directory / "run.json", state)

    try:
        print("Creating the temporary S3 bucket...", flush=True)
        apply()
        bucket = terraform(
            commands, "output", "-raw", "bucket", timeout=deadline - time.monotonic()
        ).stdout.strip()
        state["bucket"] = bucket
        write_json(directory / "run.json", state)
        variables["job"] = transfer.prepare(
            bucket, sample, run_id, math.ceil(timeout_minutes * 60) + 300
        )
        variables["job"]["check"] = check
        variables["launch"] = True
        write_json(infrastructure / "run.auto.tfvars.json", variables)
        print(
            f"Launching the image on the cheapest matching GPU from {', '.join(gpu_names)}, "
            f"at most ${max_hourly_price:.2f}/hour...",
            flush=True,
        )
        apply()
        instance = json.loads(
            terraform(
                commands, "output", "-json", "instance", timeout=deadline - time.monotonic()
            ).stdout
        )
        if not isinstance(instance, dict):
            raise RunError("Terraform returned an invalid instance output.")
        state["instance"] = instance
        write_json(directory / "run.json", state)
        price = instance.get("hourly_price")
        if (
            not isinstance(price, (int, float))
            or isinstance(price, bool)
            or not 0 <= price <= max_hourly_price
        ):
            raise RunError("The provisioned instance exceeds the requested hourly price limit.")
        print(f"Waiting for the container's {check} result...", flush=True)
        report = transfer.collect(bucket, directory, run_id, deadline, check=check)
        state["outcome"] = "passed"
    except BaseException:
        state["outcome"] = "failed"
        raise
    finally:
        cleanup(commands, state)
    return report
