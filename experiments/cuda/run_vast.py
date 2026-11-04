"""Build, rent, smoke-test, collect results, and destroy a Vast GPU VM."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

CUDA_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = CUDA_DIRECTORY.parents[1]
RUNS_DIRECTORY = CUDA_DIRECTORY / "artifacts" / "vast"
IMAGE = "qwen-commit-cuda"


class RunError(RuntimeError):
    """A failed local or remote smoke-test operation."""


def write_json(path: Path, document: dict) -> None:
    """Atomically retain recovery information before starting another operation."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def training_sample(source: Path, destination: Path) -> None:
    """Export one training row, excluding provenance and split-only metadata."""
    parquet = pq.ParquetFile(source)
    columns = ["id", "subject", "diff"]
    if any(column not in parquet.schema_arrow.names for column in columns):
        raise RunError("Training Parquet must contain id, subject, and diff.")
    batch = next(parquet.iter_batches(batch_size=1, columns=columns), None)
    if batch is None or batch.num_rows != 1:
        raise RunError("Training Parquet contains no rows.")
    table = pa.Table.from_batches([batch]).replace_schema_metadata(None)
    for column in columns:
        value = table[column][0].as_py()
        if not isinstance(value, str) or not value.strip():
            raise RunError(f"The first training row has an invalid {column}.")
    pq.write_table(table, destination, compression="zstd")


def validate_report(path: Path) -> dict:
    """Require a real, nonempty CUDA result instead of just a zero exit status."""
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_checksum = sum(number * number for number in range(1024))
    try:
        valid = (
            report["one_batch_rows"] == 1
            and report["cuda_device_count"] >= 1
            and bool(report["cuda_device_name"])
            and report["cuda_device_total_memory_bytes"] > 0
            and str(report["cuda_runtime_version"]).startswith("13.2")
            and str(report["torch_version"]).split("+")[0] == "2.12.1"
            and math.isclose(float(report["cuda_tensor_checksum"]), expected_checksum, rel_tol=1e-6)
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RunError("The downloaded report does not prove the pinned CUDA smoke test passed.")
    return report


def connection(instance: dict) -> tuple[str, int]:
    """Reject unusable connection details before passing them to OpenSSH."""
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    if (
        not isinstance(host, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host)
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
    ):
        raise RunError("Vast returned invalid SSH connection details.")
    return host, port


class Commands:
    """Run tools with bounded waits and private, per-operation logs."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.sequence = max(
            (int(path.name.split("-", 1)[0]) for path in directory.glob("[0-9]*-*.log")),
            default=0,
        )
        self.environment = os.environ.copy()
        self.api_key = os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")
        for name in ("VAST_API_KEY", "VASTAI_API_KEY"):
            self.environment.pop(name, None)

    def run(
        self,
        arguments: list[str],
        *,
        timeout: float = 300,
        stdin: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        self.sequence += 1
        log_path = self.directory / f"{self.sequence:02d}-{Path(arguments[0]).stem}.log"
        if timeout <= 0:
            raise RunError("The rental time limit expired.")
        environment = self.environment.copy()
        if Path(arguments[0]).stem == "terraform":
            environment.update(TF_IN_AUTOMATION="1", TF_INPUT="0")
            if self.api_key:
                environment["VASTAI_API_KEY"] = self.api_key
        with log_path.open("w", encoding="utf-8") as log:
            # Give Terraform a chance to flush state before attempting cleanup.
            process = subprocess.Popen(
                arguments,
                cwd=self.directory,
                env=environment,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            try:
                process.communicate(input=stdin, timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    if os.name == "nt":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        os.killpg(process.pid, signal.SIGINT)
                except OSError:
                    # Windows jobs without a console cannot receive console signals.
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        output = log_path.read_text(encoding="utf-8", errors="replace")
        if check and process.returncode:
            raise RunError(f"{Path(arguments[0]).name} failed; see {log_path}")
        return subprocess.CompletedProcess(arguments, process.returncode, output, "")


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


def ssh_arguments(directory: Path, host: str, port: int, *, copy: bool = False) -> list[str]:
    return [
        "scp" if copy else "ssh",
        "-P" if copy else "-p",
        str(port),
        "-i",
        str(directory / "id_ed25519"),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={directory / 'known_hosts'}",
        *([] if copy else [f"root@{host}"]),
    ]


def remote_script(remote_directory: str, image_tag: str, image_id: str) -> str:
    """Run the existing Compose service with host permissions matching its UID."""
    return f"""set -eu
cd {shlex.quote(remote_directory)}
test "$(id -u)" -eq 0
nvidia-smi
docker compose version
docker load --input image.tar
test "$(docker image inspect --format '{{{{.Id}}}}' {shlex.quote(image_tag)})" = {shlex.quote(image_id)}
docker tag {shlex.quote(image_tag)} qwen-commit-cuda:local
mkdir -p datasets artifacts
mv train.parquet datasets/train.parquet
chmod 755 . datasets
chmod 444 datasets/train.parquet
chown 10001:10001 artifacts
chmod 700 artifacts
export QWEN_COMMIT_DATASET_DIR="$PWD/datasets"
export QWEN_COMMIT_ARTIFACT_DIR="$PWD/artifacts"
timeout --signal=TERM 300 docker compose --file compose.yaml run --rm --no-build cuda-experiment
"""


def run_smoke(
    dataset: Path,
    *,
    gpu_name: str,
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
            "gpu_name": gpu_name,
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
    print(f"Renting one {gpu_name}, at most ${max_hourly_price:.2f}/hour...", flush=True)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_DIRECTORY / "notebooks/prepared/train.parquet"
    )
    parser.add_argument("--gpu", default="RTX 3090")
    parser.add_argument("--max-hourly-price", type=float, default=0.50)
    parser.add_argument("--timeout-minutes", type=float, default=60)
    parser.add_argument("--dry-run", action="store_true", help="Describe the run without renting.")
    parser.add_argument(
        "--cleanup", type=Path, help="Retry teardown for an existing run directory."
    )
    arguments = parser.parse_args(argv)
    if not 0 < arguments.max_hourly_price <= 10 or not 0 < arguments.timeout_minutes <= 120:
        parser.error("Use a positive hourly limit <= $10 and a time limit <= 120 minutes.")
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(arguments.dataset.resolve()),
                    "gpu": arguments.gpu,
                    "max_hourly_price_usd": arguments.max_hourly_price,
                    "rental_timeout_minutes": arguments.timeout_minutes,
                    "steps": [
                        "build locally",
                        "terraform apply",
                        "upload",
                        "smoke",
                        "download",
                        "terraform destroy",
                    ],
                    "uploads": [
                        "Docker image",
                        "Compose configuration",
                        "one training row: id/subject/diff",
                    ],
                    "artifacts": str(RUNS_DIRECTORY),
                },
                indent=2,
            )
        )
        return 0
    api_key = os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")
    if not api_key:
        parser.error("Set VAST_API_KEY to a personal Vast API key before running.")
    try:
        if arguments.cleanup:
            directory = arguments.cleanup.resolve()
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
            commands = Commands(directory)
            terraform(
                commands, "init", "-input=false", "-no-color", "-lockfile=readonly", timeout=600
            )
            cleanup(commands, state)
        else:
            report = run_smoke(
                arguments.dataset.resolve(),
                gpu_name=arguments.gpu,
                max_hourly_price=arguments.max_hourly_price,
                timeout_minutes=arguments.timeout_minutes,
            )
            print(f"CUDA smoke test passed. Report: {report}")
    except KeyboardInterrupt:
        print("Interrupted. Check run.json for cleanup status.", file=sys.stderr)
        return 130
    except (RunError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"CUDA run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
