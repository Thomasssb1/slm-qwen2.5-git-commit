"""Terraform execution, teardown, and recovery of saved runs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from . import RunError
from .artifacts import write_json
from .commands import Commands


def terraform(commands: Commands, operation: str, *arguments: str, **options):
    return commands.run(
        ["terraform", f"-chdir={commands.directory / 'terraform'}", operation, *arguments],
        **options,
    )


def cleanup(commands: Commands, state: dict) -> None:
    """Destroy the resources recorded in this run's isolated Terraform state."""
    state["cleanup"] = "pending"
    write_json(commands.directory / "run.json", state)
    print("Destroying the rental and its temporary SSH key...", flush=True)
    try:
        terraform(commands, "destroy", "-auto-approve", "-input=false", "-no-color", timeout=900)
        if terraform(commands, "state", "list").stdout.strip():
            raise RunError("Terraform still tracks resources after destroy.")
        if not state.get("apply_complete"):
            raise RunError(
                "Terraform removed its tracked resources, but apply did not finish. "
                f"Check Vast for untracked resources with label {state['label']} "
                "and the public key in this run's id_ed25519.pub."
            )
    except BaseException:
        state["cleanup"] = "failed"
        write_json(commands.directory / "run.json", state)
        print(
            "Cleanup was not confirmed; the rental may still be billing. Retry with:\n"
            f'uv run python experiments/cuda/run_vast.py --cleanup "{commands.directory}"',
            file=sys.stderr,
        )
        raise
    state["cleanup"] = "complete"
    write_json(commands.directory / "run.json", state)


def recover(directory: Path, *, commands_type=Commands) -> None:
    """Validate a saved run and retry its Terraform teardown."""
    directory = directory.resolve()
    state = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("run_id"), str)
        or not re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{12}", state["run_id"])
        or state["run_id"] != directory.name
        or state.get("backend") != "terraform"
        or state.get("label") != "qwen-cuda-" + state["run_id"]
    ):
        raise RunError("This directory is not a CUDA runner state directory.")
    infrastructure = directory / "terraform"
    if not all(
        (infrastructure / name).is_file()
        for name in (
            "main.tf",
            "variables.tf",
            ".terraform.lock.hcl",
            "run.auto.tfvars.json",
            "terraform.tfstate",
        )
    ):
        raise RunError("Terraform recovery files are missing; check the run label in Vast.")
    commands = commands_type(directory)
    terraform(commands, "init", "-input=false", "-no-color", "-lockfile=readonly", timeout=600)
    cleanup(commands, state)
