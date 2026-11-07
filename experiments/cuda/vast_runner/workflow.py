"""Build, rent, execute, and clean up a CUDA smoke-test run."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from . import RunError
from .artifacts import CUDA_DIRECTORY, RUNS_DIRECTORY, training_sample, validate_report, write_json
from .commands import Commands
from .remote import connection, remote_script, ssh_arguments
from .terraform import cleanup, terraform

IMAGE = "qwen-commit-cuda"


def run_smoke(
    dataset: Path,
    *,
    gpu_names: list[str],
    max_hourly_price: float,
    timeout_minutes: float,
    runs_directory: Path = RUNS_DIRECTORY,
    commands_type=Commands,
) -> Path:
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
    }
    write_json(directory / "run.json", state)
    print(f"Run artifacts: {directory}", flush=True)

    # All local preparation precedes renting: none of it consumes rental time.
    training_sample(dataset, directory / "train.parquet")
    commands.run(["terraform", "version"])
    commands.run(["docker", "info"])
    commands.run(["ssh", "-V"])
    commands.run(["scp"], check=False)
    commands.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(directory / "id_ed25519")]
    )
    infrastructure = directory / "terraform"
    infrastructure.mkdir()
    for name in ("main.tf", "variables.tf", ".terraform.lock.hcl"):
        shutil.copy2(CUDA_DIRECTORY / "vast" / name, infrastructure / name)
    write_json(
        infrastructure / "run.auto.tfvars.json",
        {
            "ssh_public_key": (directory / "id_ed25519.pub").read_text().strip(),
            "run_label": state["label"],
            "gpu_names": gpu_names,
            "max_hourly_price": max_hourly_price,
            "rental_seconds": timeout_minutes * 60,
        },
    )
    terraform(commands, "init", "-input=false", "-no-color", "-lockfile=readonly", timeout=600)
    terraform(commands, "validate", "-no-color")
    tag = f"{IMAGE}:{run_id.lower()}"
    print("Building and exporting the CUDA image locally...", flush=True)
    commands.run(
        ["docker", "build", "--platform", "linux/amd64", "-t", tag, str(CUDA_DIRECTORY)],
        timeout=3600,
    )
    image_id = commands.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag]
    ).stdout.strip()
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
        raise RunError("Docker returned an invalid image ID.")
    state["image_id"] = image_id
    commands.run(["docker", "save", "--output", str(directory / "image.tar"), tag], timeout=900)
    shutil.copy2(CUDA_DIRECTORY / "compose.yaml", directory / "compose.yaml")

    state["outcome"] = "running"
    state["cleanup"] = "pending"
    write_json(directory / "run.json", state)
    deadline = time.monotonic() + timeout_minutes * 60
    print(
        f"Renting the cheapest matching GPU from {', '.join(gpu_names)}, "
        f"at most ${max_hourly_price:.2f}/hour...",
        flush=True,
    )
    try:
        # Enter the cleanup scope before apply so partial creation is also torn down.
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
        host, port = connection(instance)
        ssh = ssh_arguments(directory, host, port)
        while time.monotonic() < deadline:
            result = commands.run(
                [*ssh, "true"], timeout=min(15, deadline - time.monotonic()), check=False
            )
            if result.returncode == 0:
                break
            time.sleep(min(5, max(0, deadline - time.monotonic())))
        else:
            raise RunError("SSH did not become ready within the rental time limit.")

        remote_directory = f"/tmp/qwen-cuda-{run_id}"
        commands.run(
            [*ssh, f"umask 077; mkdir -p {shlex.quote(remote_directory)}"],
            timeout=deadline - time.monotonic(),
        )
        copy = ssh_arguments(directory, host, port, copy=True)
        print("Uploading the image and one training row...", flush=True)
        commands.run(
            [
                *copy,
                str(directory / "image.tar"),
                str(directory / "compose.yaml"),
                str(directory / "train.parquet"),
                f"root@{host}:{remote_directory}/",
            ],
            timeout=deadline - time.monotonic(),
        )
        print("Running the CUDA smoke test...", flush=True)
        commands.run(
            [*ssh, "bash -s"],
            stdin=remote_script(remote_directory, tag, image_id),
            timeout=deadline - time.monotonic(),
        )
        commands.run(
            [*copy, f"root@{host}:{remote_directory}/artifacts/cuda-smoke.json", str(directory)],
            timeout=deadline - time.monotonic(),
        )
        validate_report(directory / "cuda-smoke.json")
        state["outcome"] = "passed"
    except BaseException:
        state["outcome"] = "failed"
        raise
    finally:
        cleanup(commands, state)
    return directory / "cuda-smoke.json"
