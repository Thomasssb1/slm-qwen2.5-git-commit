"""Terraform execution, teardown, and recovery of saved runs."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from . import RunError
from .artifacts import CUDA_DIRECTORY, write_json
from .commands import CommandError, Commands

TERRAFORM_DIRECTORY = CUDA_DIRECTORY / "vast"
POLL_RETRIES = 3


def _transient_poll_failure(output: str, summaries: set[str]) -> bool:
    """Recover only known read failures, never a failed/ambiguous rent request."""
    errors = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            return False
        if not isinstance(event, dict):
            return False
        if event.get("@level") == "error":
            diagnostic = event.get("diagnostic", {})
            if not isinstance(diagnostic, dict):
                return False
            errors.append(diagnostic)
    if not errors:
        return False
    for diagnostic in errors:
        summary = diagnostic.get("summary")
        if not isinstance(summary, str) or summary not in summaries:
            return False
        detail = diagnostic.get("detail", "")
        if not isinstance(detail, str):
            return False
        if not re.fullmatch(
            r'showing instance (?P<id>\d+): (?:Get "[^"\s]+/api/v0/instances/(?P=id)/?": '
            r"[^\n]*(?:Client\.Timeout[^\n]*|i/o timeout|TLS handshake timeout|"
            r"connection reset[^\n]*|forcibly closed[^\n]*|\bEOF)|"
            r"vast\.ai: [^\n]+ \(status (?:429|500|502|503|504)\))",
            detail,
        ):
            return False
    return True


def terraform(commands: Commands, operation: str, *arguments: str, **options):
    infrastructure = commands.directory / "terraform"
    run_arguments = []
    if operation == "init":
        run_arguments = [
            "-reconfigure",
            f"-backend-config=path={infrastructure / 'terraform.tfstate'}",
        ]
    elif operation in {"apply", "destroy"}:
        run_arguments = [f"-var-file={infrastructure / 'run.auto.tfvars.json'}"]
    return commands.run(
        ["terraform", f"-chdir={TERRAFORM_DIRECTORY}", operation, *run_arguments, *arguments],
        **options,
    )


def apply_or_destroy(commands: Commands, operation: str, *, timeout: float, **options) -> None:
    """Bounded status recovery using Terraform's own refresh and destroy operations."""
    deadline = time.monotonic() + timeout
    refresh_only = False
    for attempt in range(POLL_RETRIES + 1):
        arguments = ["-auto-approve", "-input=false", "-no-color", "-json"]
        if refresh_only:
            arguments.append("-refresh-only")
        try:
            terraform(
                commands,
                operation,
                *arguments,
                timeout=deadline - time.monotonic(),
                **options,
            )
            return
        except CommandError as error:
            if operation == "destroy":
                summaries = {"Error reading instance", "Error waiting for instance to be destroyed"}
            elif refresh_only:
                summaries = {"Error reading instance"}
            else:
                summaries = {"Error waiting for instance to become ready"}
            delay = 10 * (attempt + 1)
            if (
                attempt == POLL_RETRIES
                or deadline - time.monotonic() <= delay
                or not _transient_poll_failure(error.output, summaries)
            ):
                raise
            # A failed Create is tainted. Retrying normal apply would replace the
            # already-running rental. Refresh-only reads it and updates outputs;
            # the signed container report still has to pass before run success.
            refresh_only = operation == "apply"
            action = "refreshing the existing rental" if refresh_only else "retrying teardown"
            print(f"Vast status request failed temporarily; {action} in {delay}s...", flush=True)
            time.sleep(delay)


def _save_cleanup_status(commands: Commands, state: dict, status: str) -> None:
    state["cleanup"] = status
    try:
        write_json(commands.directory / "run.json", state)
    except OSError as error:
        print(f"Could not save cleanup status '{status}': {error}", file=sys.stderr)


def cleanup(commands: Commands, state: dict) -> None:
    """Destroy tracked resources even when cleanup bookkeeping cannot be saved."""
    _save_cleanup_status(commands, state, "pending")
    print("Destroying the rental and its temporary S3 bucket...", flush=True)
    try:
        apply_or_destroy(
            commands,
            "destroy",
            timeout=900,
            best_effort_log=True,
        )
        if terraform(commands, "state", "list", best_effort_log=True).stdout.strip():
            raise RunError("Terraform still tracks resources after destroy.")
        if not state.get("apply_complete"):
            raise RunError(
                "Terraform removed its tracked resources, but apply did not finish. "
                f"Check Vast for label {state['label']} and AWS S3 for bucket "
                f"{state['label'].lower()} to find any untracked resources."
            )
    except BaseException:
        _save_cleanup_status(commands, state, "failed")
        print(
            "Cleanup was not confirmed; the rental may still be billing. Retry with:\n"
            f'uv run python experiments/cuda/run.py --cleanup "{commands.directory}"',
            file=sys.stderr,
        )
        raise
    _save_cleanup_status(commands, state, "complete")


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
        (infrastructure / name).is_file() for name in ("run.auto.tfvars.json", "terraform.tfstate")
    ):
        raise RunError("Terraform recovery files are missing; check the run label in Vast.")
    commands = commands_type(directory)
    terraform(
        commands,
        "init",
        "-input=false",
        "-no-color",
        "-lockfile=readonly",
        timeout=600,
        best_effort_log=True,
    )
    cleanup(commands, state)
