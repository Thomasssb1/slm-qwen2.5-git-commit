"""Exercise the rental lifecycle without contacting Vast or renting a GPU."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

RUNNER_PATH = Path(__file__).parents[1] / "experiments/cuda/run_vast.py"
SPEC = importlib.util.spec_from_file_location("cuda_run_vast", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def valid_report() -> dict:
    return {
        "one_batch_rows": 1,
        "cuda_device_count": 1,
        "cuda_device_name": "NVIDIA RTX 3090",
        "cuda_device_total_memory_bytes": 24 * 1024**3,
        "cuda_runtime_version": "13.2",
        "torch_version": "2.12.1+cu132",
        "cuda_tensor_checksum": 357389824.0,
    }


@pytest.fixture
def training_file(tmp_path):
    path = tmp_path / "train.parquet"
    table = pa.table(
        {
            "id": ["a", "b"],
            "subject": ["Add fixture", "Update fixture"],
            "diff": ["+first", "+second"],
            "repository_group_id": ["private-group", "private-group"],
            "committed_at_utc": ["private-date", "private-date"],
            "patch_id": ["private-patch", "private-patch"],
        }
    ).replace_schema_metadata({b"private": b"local-metadata"})
    pq.write_table(table, path)
    return path


def test_sample_contains_only_one_training_row_and_no_split_metadata(tmp_path, training_file):
    destination = tmp_path / "sample.parquet"
    runner.training_sample(training_file, destination)
    exported = pq.read_table(destination)
    assert exported.to_pylist() == [{"id": "a", "subject": "Add fixture", "diff": "+first"}]
    assert exported.schema.metadata is None
    assert pq.read_table(training_file).num_rows == 2


@pytest.mark.parametrize(
    "rows, message",
    [
        ({"id": []}, "must contain"),
        ({"id": [], "subject": [], "diff": []}, "no rows"),
        ({"id": ["a"], "subject": [None], "diff": ["patch"]}, "invalid subject"),
        ({"id": ["a"], "subject": ["subject"], "diff": [" "]}, "invalid diff"),
    ],
)
def test_sample_rejects_unusable_training_input(tmp_path, rows, message):
    source = tmp_path / "train.parquet"
    destination = tmp_path / "sample.parquet"
    pq.write_table(pa.table(rows), source)
    with pytest.raises(runner.RunError, match=message):
        runner.training_sample(source, destination)
    assert not destination.exists()


class FakeTools:
    """Stateful Terraform, Docker, and SSH boundaries with real temporary files."""

    def __init__(self, directory, cloud, failure=None):
        self.directory = directory
        self.cloud = cloud
        self.failure = failure

    def run(self, arguments, **options):
        command = arguments[0]
        output = ""
        if command == "ssh-keygen":
            (self.directory / "id_ed25519.pub").write_text("ssh-ed25519 fixture-public-key")
        elif command == "docker":
            if arguments[1] == "build" and self.failure == "build":
                raise runner.RunError("build failed")
            if arguments[1:3] == ["image", "inspect"]:
                output = "sha256:" + "a" * 64
            if arguments[1] == "save":
                Path(arguments[3]).write_bytes(b"image fixture")
        elif command == "terraform" and len(arguments) > 2:
            operation = arguments[2]
            infrastructure = self.directory / "terraform"
            if operation == "init" and self.failure == "init":
                raise runner.RunError("init failed")
            if operation == "apply":
                variables = json.loads((infrastructure / "run.auto.tfvars.json").read_text())
                assert variables["run_label"] == "qwen-cuda-" + self.directory.name
                assert variables["max_hourly_price"] == 0.5
                assert "api_key" not in variables
                self.cloud["resources"].add("ssh-key")
                self.cloud["tracked"].add("ssh-key")
                if self.failure != "no-offers":
                    self.cloud["resources"].add("vm")
                    self.cloud["rentals"] += 1
                    if self.failure != "untracked-create":
                        self.cloud["tracked"].add("vm")
                (infrastructure / "terraform.tfstate").write_text(
                    json.dumps({"resources": sorted(self.cloud["tracked"])})
                )
                if self.failure in {"apply", "no-offers", "untracked-create"}:
                    raise runner.RunError("apply failed")
            elif operation == "output":
                if self.failure == "output":
                    raise runner.RunError("output failed")
                output = json.dumps(
                    {
                        "id": 123,
                        "ssh_host": "gpu.example.test",
                        "ssh_port": 2201,
                        "hourly_price": 0.75 if self.failure == "price" else 0.30,
                    }
                )
            elif operation == "destroy":
                if self.failure == "destroy":
                    raise runner.RunError("destroy failed")
                if self.failure != "retained-state":
                    self.cloud["resources"].difference_update(self.cloud["tracked"])
                    self.cloud["tracked"].clear()
                (infrastructure / "terraform.tfstate").write_text(
                    json.dumps({"resources": sorted(self.cloud["tracked"])})
                )
            elif operation == "state":
                output = "\n".join(sorted(self.cloud["tracked"]))
        elif command == "scp" and len(arguments) > 1:
            downloading = arguments[-2].endswith("/artifacts/cuda-smoke.json")
            if downloading:
                if self.failure == "download":
                    raise runner.RunError("download failed")
                report = valid_report()
                if self.failure == "report":
                    report["cuda_tensor_checksum"] = 0
                (self.directory / "cuda-smoke.json").write_text(json.dumps(report))
            else:
                if self.failure == "upload":
                    raise runner.RunError("upload failed")
                if self.failure == "timeout":
                    raise subprocess.TimeoutExpired(arguments, 1)
                self.cloud["uploaded"] = pq.read_table(self.directory / "train.parquet").to_pylist()
        elif command == "ssh" and arguments[-1] == "bash -s":
            if self.failure == "smoke":
                raise runner.RunError("GPU test failed")
            if self.failure == "interrupt":
                raise KeyboardInterrupt
        return subprocess.CompletedProcess(arguments, 0, output, "")


def invoke_fake(tmp_path, training_file, failure=None):
    cloud = {"resources": set(), "tracked": set(), "rentals": 0, "uploaded": []}

    def invoke():
        return runner.run_smoke(
            training_file,
            gpu_name="RTX 3090",
            max_hourly_price=0.50,
            timeout_minutes=60,
            runs_directory=tmp_path / "runs",
            commands_type=lambda directory: FakeTools(directory, cloud, failure),
        )

    return invoke, cloud


def saved_run(tmp_path):
    return json.loads(next((tmp_path / "runs").glob("*/run.json")).read_text())


def test_success_retains_valid_report_and_destroys_vm_and_temporary_key(tmp_path, training_file):
    invoke, cloud = invoke_fake(tmp_path, training_file)
    report_path = invoke()
    assert json.loads(report_path.read_text()) == valid_report()
    assert cloud["resources"] == set()
    assert cloud["rentals"] == 1
    assert cloud["uploaded"] == [{"id": "a", "subject": "Add fixture", "diff": "+first"}]
    state = saved_run(tmp_path)
    assert state["outcome"] == "passed"
    assert state["cleanup"] == "complete"
    assert state["instance"]["id"] == 123


@pytest.mark.parametrize(
    "failure, error",
    [
        ("output", runner.RunError),
        ("upload", runner.RunError),
        ("smoke", runner.RunError),
        ("download", runner.RunError),
        ("report", runner.RunError),
        ("timeout", subprocess.TimeoutExpired),
        ("interrupt", KeyboardInterrupt),
    ],
)
def test_failure_after_rental_still_destroys_resources(tmp_path, training_file, failure, error):
    invoke, cloud = invoke_fake(tmp_path, training_file, failure)
    with pytest.raises(error):
        invoke()
    assert cloud["rentals"] == 1
    assert cloud["resources"] == set()
    assert saved_run(tmp_path)["cleanup"] == "complete"
    assert saved_run(tmp_path)["outcome"] == "failed"


def test_local_build_failure_does_not_rent_a_vm(tmp_path, training_file):
    invoke, cloud = invoke_fake(tmp_path, training_file, "build")
    with pytest.raises(runner.RunError, match="build failed"):
        invoke()
    assert cloud["rentals"] == 0
    assert cloud["resources"] == set()
    assert saved_run(tmp_path)["cleanup"] == "not-needed"


def test_terraform_initialization_failure_does_not_rent_a_vm(tmp_path, training_file):
    invoke, cloud = invoke_fake(tmp_path, training_file, "init")
    with pytest.raises(runner.RunError, match="init failed"):
        invoke()
    assert cloud["resources"] == set()
    assert cloud["rentals"] == 0
    assert saved_run(tmp_path)["cleanup"] == "not-needed"


@pytest.mark.parametrize(
    "failure, rentals, remaining",
    [
        ("apply", 1, set()),
        ("no-offers", 0, set()),
        ("untracked-create", 1, {"vm"}),
    ],
)
def test_incomplete_apply_destroys_tracked_resources_and_requires_console_check(
    tmp_path, training_file, failure, rentals, remaining
):
    invoke, cloud = invoke_fake(tmp_path, training_file, failure)
    with pytest.raises(runner.RunError, match="apply did not finish"):
        invoke()
    assert cloud["rentals"] == rentals
    assert cloud["resources"] == remaining
    assert cloud["tracked"] == set()
    state = saved_run(tmp_path)
    assert state["cleanup"] == "failed"
    assert state["outcome"] == "failed"
    assert not state.get("apply_complete")


def test_destroy_cannot_succeed_while_terraform_still_tracks_resources(tmp_path, training_file):
    invoke, cloud = invoke_fake(tmp_path, training_file, "retained-state")
    with pytest.raises(runner.RunError, match="still tracks resources"):
        invoke()
    assert cloud["resources"] == {"vm", "ssh-key"}
    assert saved_run(tmp_path)["cleanup"] == "failed"


def test_excessive_actual_price_destroys_vm_before_upload(tmp_path, training_file):
    invoke, cloud = invoke_fake(tmp_path, training_file, "price")
    with pytest.raises(runner.RunError, match="hourly price limit"):
        invoke()
    assert cloud["resources"] == set()
    assert cloud["uploaded"] == []


def test_cleanup_failure_preserves_report_and_reports_recovery_command(
    tmp_path, training_file, capsys
):
    invoke, cloud = invoke_fake(tmp_path, training_file, "destroy")
    with pytest.raises(runner.RunError, match="destroy failed"):
        invoke()
    assert cloud["resources"] == {"vm", "ssh-key"}
    assert saved_run(tmp_path)["cleanup"] == "failed"
    assert saved_run(tmp_path)["outcome"] == "passed"
    assert next((tmp_path / "runs").glob("*/cuda-smoke.json")).exists()
    assert "--cleanup" in capsys.readouterr().err


def test_cleanup_command_retries_failed_destroy_and_keeps_report(
    tmp_path, training_file, monkeypatch
):
    invoke, cloud = invoke_fake(tmp_path, training_file, "destroy")
    with pytest.raises(runner.RunError):
        invoke()
    directory = next((tmp_path / "runs").iterdir())
    monkeypatch.setenv("VASTAI_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(runner, "Commands", lambda directory: FakeTools(directory, cloud))
    assert runner.main(["--cleanup", str(directory)]) == 0
    assert cloud["resources"] == set()
    assert saved_run(tmp_path)["cleanup"] == "complete"
    assert json.loads((directory / "cuda-smoke.json").read_text()) == valid_report()
    # Recovery is idempotent after the resources have already gone.
    assert runner.main(["--cleanup", str(directory)]) == 0
    assert cloud["resources"] == set()


@pytest.mark.parametrize(
    "missing",
    ["terraform.tfstate", "main.tf", "variables.tf", ".terraform.lock.hcl", "run.auto.tfvars.json"],
)
def test_cleanup_refuses_missing_recovery_files(
    tmp_path, training_file, monkeypatch, capsys, missing
):
    invoke, cloud = invoke_fake(tmp_path, training_file, "destroy")
    with pytest.raises(runner.RunError):
        invoke()
    directory = next((tmp_path / "runs").iterdir())
    (directory / "terraform" / missing).unlink()
    monkeypatch.setenv("VAST_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(runner, "Commands", lambda path: FakeTools(path, cloud))
    assert runner.main(["--cleanup", str(directory)]) == 1
    assert "recovery files are missing" in capsys.readouterr().err
    assert cloud["resources"] == {"vm", "ssh-key"}
    assert saved_run(tmp_path)["cleanup"] == "failed"


@pytest.mark.parametrize("document", [[], {"run_id": None}, {"run_id": "unrelated"}])
def test_cleanup_command_rejects_invalid_recovery_directory(
    tmp_path, monkeypatch, capsys, document
):
    monkeypatch.setenv("VASTAI_API_KEY", "test-only-placeholder")
    (tmp_path / "run.json").write_text(json.dumps(document))
    assert runner.main(["--cleanup", str(tmp_path)]) == 1
    assert "not a CUDA runner state directory" in capsys.readouterr().err


@pytest.mark.parametrize("argument", ["nan", "0", "-1", "inf", "11"])
def test_cli_rejects_invalid_price_before_requiring_credentials(argument, monkeypatch, capsys):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        runner.main(["--max-hourly-price", argument])
    assert error.value.code == 2
    assert "positive hourly limit" in capsys.readouterr().err


def test_cli_requires_credentials_before_starting_a_run(monkeypatch, capsys):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        runner.main([])
    assert error.value.code == 2
    assert "Set VAST_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    "change",
    [
        {"cuda_device_count": 0},
        {"one_batch_rows": 0},
        {"cuda_tensor_checksum": float("nan")},
        {"cuda_runtime_version": "12.8"},
        {"torch_version": "2.11.0"},
        {"cuda_device_total_memory_bytes": 0},
        {"cuda_device_name": ""},
        {"cuda_tensor_checksum": None},
    ],
)
def test_report_requires_cuda_batch_and_checksum_evidence(tmp_path, change):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(valid_report() | change))
    with pytest.raises(runner.RunError, match="does not prove"):
        runner.validate_report(path)


@pytest.mark.parametrize("host,port", [("-oProxyCommand=bad", 22), ("host;bad", 22), ("host", 0)])
def test_invalid_connection_details_are_rejected(host, port):
    with pytest.raises(runner.RunError, match="SSH connection"):
        runner.connection({"ssh_host": host, "ssh_port": port})


def test_dry_run_works_without_credentials_and_describes_lifecycle(monkeypatch, capsys):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    assert runner.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["steps"][0] == "build locally"
    assert plan["steps"][-1] == "terraform destroy"
    assert plan["max_hourly_price_usd"] == 0.50


def test_command_logs_survive_a_later_cleanup_session(tmp_path):
    first = runner.Commands(tmp_path)
    first.run([sys.executable, "-c", "print('original log')"])
    second = runner.Commands(tmp_path)
    result = second.run([sys.executable, "-c", "print('cleanup log')"])
    assert result.stdout.strip() == "cleanup log"
    assert {path.read_text().strip() for path in tmp_path.glob("*.log")} == {
        "original log",
        "cleanup log",
    }


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="OpenSSH is not installed")
def test_command_can_generate_an_unencrypted_ephemeral_key(tmp_path):
    commands = runner.Commands(tmp_path)
    key_path = tmp_path / "id_ed25519"
    commands.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)])
    assert key_path.is_file()
    assert key_path.with_suffix(".pub").read_text().startswith("ssh-ed25519 ")


def test_command_failure_retains_diagnostics(tmp_path):
    commands = runner.Commands(tmp_path)
    with pytest.raises(runner.RunError, match="failed; see"):
        commands.run([sys.executable, "-c", "print('failure details'); raise SystemExit(3)"])
    assert "failure details" in next(tmp_path.glob("*.log")).read_text()


def test_command_timeout_returns_control_for_cleanup(tmp_path):
    commands = runner.Commands(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        commands.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.1)


def test_api_credentials_are_not_inherited_by_subprocesses(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "test-primary-key")
    monkeypatch.setenv("VASTAI_API_KEY", "test-legacy-key")
    result = runner.Commands(tmp_path).run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('VAST_API_KEY'), os.environ.get('VASTAI_API_KEY'))",
        ]
    )
    assert result.stdout.strip() == "None None"


def test_terraform_receives_api_key_through_environment_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "test-primary-key")
    monkeypatch.setenv("VASTAI_API_KEY", "test-legacy-key")
    start_process = subprocess.Popen

    def terraform_probe(arguments, **options):
        assert arguments == ["terraform", "version"]
        return start_process(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('VAST_API_KEY'), os.environ.get('VASTAI_API_KEY'), os.environ.get('TF_INPUT'))",
            ],
            **options,
        )

    monkeypatch.setattr(runner.subprocess, "Popen", terraform_probe)
    result = runner.Commands(tmp_path).run(["terraform", "version"])
    assert result.stdout.strip() == "None test-primary-key 0"
