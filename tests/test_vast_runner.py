"""Exercise the rental lifecycle without contacting Vast or renting a GPU."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from botocore.exceptions import ClientError

RUNNER_PATH = Path(__file__).parents[1] / "experiments/cuda/run.py"
SPEC = importlib.util.spec_from_file_location("cuda_run", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(RUNNER_PATH.parent))
    SPEC.loader.exec_module(runner)
    from generation import SYSTEM_PROMPT
    from vast_runner import RunError, artifacts
    from vast_runner import commands as command_tools
    from vast_runner import terraform as terraform_tools
    from vast_runner.image import publish
    from vast_runner.storage import Transfer


@pytest.fixture(autouse=True)
def isolated_project_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_DIRECTORY", tmp_path)
    for name in ("VAST_API_KEY", "VASTAI_API_KEY", "VAST_IMAGE_LOGIN"):
        monkeypatch.delenv(name, raising=False)


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


def valid_model_report() -> dict:
    report = valid_report()
    del report["one_batch_rows"]
    return report | {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "model_revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
        "model_class": "Qwen2ForCausalLM",
        "parameter_count": 1543714304,
        "cuda_parameter_count": 1543714304,
        "parameter_dtype": "bfloat16",
        "tokenizer_vocab_size": 151665,
        "tokenizer_probe_tokens": 4,
        "transformers_version": "5.22.2",
        "load_seconds": 20.5,
        "cuda_memory_allocated_bytes": 3100000000,
        "cuda_peak_memory_allocated_bytes": 3200000000,
    }


def valid_inference_report() -> dict:
    examples = {}
    for name in ("single_file", "multiple_files", "truncated"):
        original = 5000 if name == "truncated" else 100
        examples[name] = {
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "max_diff_tokens": 4096,
            "max_subject_characters": 72,
            "truncation_side": "right",
            "generation_config": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "max_new_tokens": 64,
                "min_new_tokens": 0,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
                "repetition_penalty": 1.0,
                "use_cache": True,
            },
            "diff_tokens": original,
            "diff_tokens_used": min(original, 4096),
            "diff_truncated": original > 4096,
            "prompt_tokens": min(original, 4096) + 60,
            "generated_tokens": 3,
            "generated_token_ids": [123, 456, 151645],
            "raw_output": "Trim whitespace from names",
            "subject": "Trim whitespace from names",
            "valid": True,
            "errors": [],
            "finish_reason": "eos",
            "generation_seconds": 0.5,
        }
    return valid_model_report() | {
        "examples": examples,
        "repeat_matches": True,
        "repeat_token_ids": examples["single_file"]["generated_token_ids"],
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
    artifacts.training_sample(training_file, destination)
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
    with pytest.raises(RunError, match=message):
        artifacts.training_sample(source, destination)
    assert not destination.exists()


class FakeStorage:
    def __init__(self, cloud, failure=None):
        self.cloud = cloud
        self.failure = failure

    def put_object(self, *, Bucket, Key, Body):
        if self.failure == "upload":
            raise RunError("upload failed")
        self.cloud.setdefault("objects", {})[Key] = Body
        self.cloud["uploaded"] = pq.read_table(io.BytesIO(Body)).to_pylist()

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        return f"https://example.invalid/{Params['Key']}?operation={operation}&expires={ExpiresIn}"

    def get_object(self, *, Bucket, Key):
        if self.failure == "download":
            raise RunError("download failed")
        if self.failure == "timeout":
            raise RunError("transfer timed out")
        if self.failure == "interrupt":
            raise KeyboardInterrupt
        try:
            return {"Body": io.BytesIO(self.cloud["objects"][Key])}
        except KeyError:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject") from None


def fake_transfer(cloud, failure=None):
    transfer = Transfer.__new__(Transfer)
    transfer.region = "eu-west-2"
    transfer.client = FakeStorage(cloud, failure)
    return transfer


class FakeTools:
    """Stateful cloud boundaries; production code writes inputs and collects results."""

    def __init__(self, directory, cloud, failure=None):
        self.directory = directory
        self.cloud = cloud
        self.failure = failure

    def run(self, arguments, **options):
        command = arguments[0]
        output = ""
        if command == "docker":
            if arguments[1] in {"build", "push"} and self.failure == arguments[1]:
                raise RunError(f"{self.failure} failed")
            if arguments[1:3] == ["image", "inspect"]:
                repository = arguments[-1].rsplit(":", 1)[0]
                output = json.dumps([repository + "@sha256:" + "a" * 64])
                if self.failure == "digest":
                    output = "[]"
        elif command == "terraform" and len(arguments) > 2:
            operation = arguments[2]
            infrastructure = self.directory / "terraform"
            if operation == "init" and self.failure == "init":
                raise RunError("init failed")
            if operation == "apply":
                variables_path = Path(
                    next(
                        arg.removeprefix("-var-file=")
                        for arg in arguments
                        if arg.startswith("-var-file=")
                    )
                )
                variables = json.loads(variables_path.read_text())
                self.cloud["resources"].add("bucket")
                self.cloud["tracked"].add("bucket")
                if variables["launch"]:
                    if self.failure != "no-offers":
                        self.cloud["resources"].add("container")
                        self.cloud["rentals"] += 1
                        if self.failure != "untracked-create":
                            self.cloud["tracked"].add("container")
                    check = variables["job"]["check"]
                    report = valid_model_report() if check == "model" else valid_report()
                    if check == "inference":
                        report = valid_inference_report()
                    if check == "validation":
                        inputs = self.directory / "validation-input.parquet"
                        rows = pq.read_table(inputs).to_pylist()
                        report = valid_model_report() | {
                            "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
                            "row_count": len(rows),
                            "predictions": [
                                {
                                    "id": row["id"],
                                    **valid_inference_report()["examples"]["single_file"],
                                }
                                for row in rows
                            ],
                        }
                    if self.failure == "report":
                        report["cuda_tensor_checksum"] = 0
                    result = {
                        "run_id": variables["job"]["run_id"],
                        "outcome": "failed" if self.failure == "smoke" else "passed",
                        "report": report,
                        "log": "CUDA subprocess output",
                    }
                    self.cloud.setdefault("objects", {})["result.json"] = json.dumps(
                        result
                    ).encode()
                (infrastructure / "terraform.tfstate").write_text(
                    json.dumps({"resources": sorted(self.cloud["tracked"])})
                )
                if self.failure == "storage-apply" or (
                    variables["launch"]
                    and self.failure in {"apply", "no-offers", "untracked-create"}
                ):
                    raise RunError("apply failed")
            elif operation == "output":
                if arguments[-1] == "bucket":
                    output = ("qwen-cuda-" + self.directory.name).lower()
                else:
                    if self.failure == "output":
                        raise RunError("output failed")
                    output = json.dumps(
                        {
                            "id": 123,
                            "hourly_price": 0.75 if self.failure == "price" else 0.30,
                        }
                    )
            elif operation == "destroy":
                if self.failure == "destroy":
                    raise RunError("destroy failed")
                if self.failure != "retained-state":
                    self.cloud["resources"].difference_update(self.cloud["tracked"])
                    self.cloud["tracked"].clear()
                    self.cloud["objects"] = {}
                (infrastructure / "terraform.tfstate").write_text(
                    json.dumps({"resources": sorted(self.cloud["tracked"])})
                )
            elif operation == "state":
                output = "\n".join(sorted(self.cloud["tracked"]))
        else:
            assert command not in {"ssh", "scp", "ssh-keygen"}
        return subprocess.CompletedProcess(arguments, 0, output, "")


def invoke_fake(tmp_path, training_file, failure=None, *, check="cuda"):
    cloud = {"resources": set(), "tracked": set(), "rentals": 0, "uploaded": []}

    def invoke():
        return runner.run_smoke(
            training_file,
            gpu_names=["RTX 3090"],
            max_hourly_price=0.50,
            timeout_minutes=60,
            runs_directory=tmp_path / "runs",
            commands_type=lambda directory: FakeTools(directory, cloud, failure),
            transfer_type=lambda region: fake_transfer(cloud, failure),
            check=check,
        )

    return invoke, cloud


@pytest.mark.parametrize("failure", [None, "smoke", "report", "timeout"])
def test_model_check_needs_no_dataset_collects_model_evidence_and_cleans_up(tmp_path, failure):
    invoke, cloud = invoke_fake(tmp_path, tmp_path / "nonexistent.parquet", failure, check="model")
    if failure:
        with pytest.raises(RunError):
            invoke()
    else:
        report = invoke()
        assert report.name == "model-load.json"
        assert json.loads(report.read_text()) == valid_model_report()
    assert cloud["uploaded"] == []
    assert cloud["resources"] == set()
    assert cloud["rentals"] == 1
    state = saved_run(tmp_path)
    assert state["check"] == "model"
    assert state["outcome"] == ("failed" if failure else "passed")
    assert state["cleanup"] == "complete"
    directory = next((tmp_path / "runs").iterdir())
    assert not (directory / "train.parquet").exists()
    job = json.loads((directory / "terraform/run.auto.tfvars.json").read_text())["job"]
    assert job["check"] == "model"
    assert "sample_url" not in job
    assert "sample_sha256" not in job


def test_unknown_check_fails_before_creating_run_artifacts(tmp_path):
    invoke, cloud = invoke_fake(tmp_path, tmp_path / "missing", check="unknown")
    with pytest.raises(ValueError, match="cuda, model, inference, or validation"):
        invoke()
    assert not (tmp_path / "runs").exists()
    assert cloud["rentals"] == 0


@pytest.mark.parametrize(
    "check, filename, expected",
    [
        ("model", "model-load.json", valid_model_report),
        ("inference", "inference-smoke.json", valid_inference_report),
    ],
)
def test_cli_check_saves_model_or_inference_report_without_a_dataset(
    tmp_path, monkeypatch, capsys, check, filename, expected
):
    cloud = {"resources": set(), "tracked": set(), "rentals": 0, "uploaded": []}
    monkeypatch.setenv("VAST_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(
        runner,
        "run_smoke",
        partial(
            runner.run_smoke,
            runs_directory=tmp_path / "runs",
            commands_type=lambda directory: FakeTools(directory, cloud),
            transfer_type=lambda region: fake_transfer(cloud),
        ),
    )
    assert runner.main(["--check", check]) == 0
    report = next((tmp_path / "runs").glob("*/" + filename))
    assert json.loads(report.read_text()) == expected()
    assert f"{check.capitalize()} check passed" in capsys.readouterr().out
    assert cloud["resources"] == set()
    assert cloud["uploaded"] == []


@pytest.mark.parametrize(
    "change",
    [
        {"system_prompt": "wrong prompt"},
        {"generation_config": {"do_sample": True, "num_beams": 1, "max_new_tokens": 64}},
        {"diff_truncated": True},
        {"diff_tokens_used": 4097},
        {"raw_output": "x" * 73},
        {"subject": "wrong subject"},
        {"raw_output": None},
        {"generated_token_ids": []},
        {"finish_reason": "unknown"},
    ],
)
def test_inference_report_rejects_wrong_prompt_settings_and_inconsistent_results(tmp_path, change):
    report = valid_inference_report()
    report["examples"]["single_file"].update(change)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(RunError, match="pinned inference check"):
        artifacts.validate_report(path, check="inference")


def test_inference_report_accepts_honestly_reported_format_failures(tmp_path):
    report = valid_inference_report()
    report["examples"]["multiple_files"].update(
        raw_output="x" * 73, subject=None, valid=False, errors=["too_long"]
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert artifacts.validate_report(path, check="inference") == report


@pytest.mark.parametrize(
    "change", [{"repeat_matches": False}, {"repeat_token_ids": [0]}, {"examples": {}}]
)
def test_inference_report_requires_all_cases_and_repeatable_output(tmp_path, change):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(valid_inference_report() | change))
    with pytest.raises(RunError, match="pinned inference check"):
        artifacts.validate_report(path, check="inference")


def saved_run(tmp_path):
    return json.loads(next((tmp_path / "runs").glob("*/run.json")).read_text())


def test_success_retains_report_and_log_and_destroys_container_bucket_and_objects(
    tmp_path, training_file
):
    invoke, cloud = invoke_fake(tmp_path, training_file)
    report_path = invoke()
    assert json.loads(report_path.read_text()) == valid_report()
    assert cloud["resources"] == set()
    assert cloud["rentals"] == 1
    assert cloud["objects"] == {}
    assert (report_path.parent / "smoke.log").read_text() == "CUDA subprocess output"
    assert cloud["uploaded"] == [{"id": "a", "subject": "Add fixture", "diff": "+first"}]
    state = saved_run(tmp_path)
    assert state["outcome"] == "passed"
    assert state["cleanup"] == "complete"
    assert state["instance"]["id"] == 123
    infrastructure = report_path.parent / "terraform"
    assert not list(infrastructure.glob("*.tf"))
    assert not (infrastructure / ".terraform.lock.hcl").exists()
    variables = json.loads((infrastructure / "run.auto.tfvars.json").read_text())
    assert (
        variables["image"]
        == state["image"]
        == ("ghcr.io/thomasssb1/qwen-commit-cuda@sha256:" + "a" * 64)
    )
    assert variables["launch"] is True
    assert not (report_path.parent / "image.tar").exists()
    assert not (report_path.parent / "id_ed25519").exists()


@pytest.mark.parametrize(
    "failure, error",
    [
        ("output", RunError),
        ("smoke", RunError),
        ("download", RunError),
        ("report", RunError),
        ("timeout", RunError),
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


@pytest.mark.parametrize("failure", ["build", "push", "digest"])
def test_image_publication_failure_does_not_create_cloud_resources(
    tmp_path, training_file, failure
):
    invoke, cloud = invoke_fake(tmp_path, training_file, failure)
    with pytest.raises(RunError):
        invoke()
    assert cloud["rentals"] == 0
    assert cloud["resources"] == set()
    assert saved_run(tmp_path)["cleanup"] == "not-needed"


def test_terraform_initialization_failure_does_not_create_cloud_resources(tmp_path, training_file):
    invoke, cloud = invoke_fake(tmp_path, training_file, "init")
    with pytest.raises(RunError, match="init failed"):
        invoke()
    assert cloud["resources"] == set()
    assert cloud["rentals"] == 0
    assert saved_run(tmp_path)["cleanup"] == "not-needed"


@pytest.mark.parametrize(
    "failure, rentals, remaining",
    [
        ("apply", 1, set()),
        ("no-offers", 0, set()),
        ("untracked-create", 1, {"container"}),
        ("storage-apply", 0, set()),
    ],
)
def test_incomplete_apply_destroys_tracked_resources_and_requires_console_check(
    tmp_path, training_file, failure, rentals, remaining
):
    invoke, cloud = invoke_fake(tmp_path, training_file, failure)
    with pytest.raises(RunError, match="apply did not finish"):
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
    with pytest.raises(RunError, match="still tracks resources"):
        invoke()
    assert cloud["resources"] == {"container", "bucket"}
    assert saved_run(tmp_path)["cleanup"] == "failed"


def test_excessive_actual_price_destroys_container_and_bucket(tmp_path, training_file):
    invoke, cloud = invoke_fake(tmp_path, training_file, "price")
    with pytest.raises(RunError, match="hourly price limit"):
        invoke()
    assert cloud["resources"] == set()
    assert cloud["objects"] == {}


def test_cleanup_failure_preserves_report_and_reports_recovery_command(
    tmp_path, training_file, capsys
):
    invoke, cloud = invoke_fake(tmp_path, training_file, "destroy")
    with pytest.raises(RunError, match="destroy failed"):
        invoke()
    assert cloud["resources"] == {"container", "bucket"}
    assert saved_run(tmp_path)["cleanup"] == "failed"
    assert saved_run(tmp_path)["outcome"] == "passed"
    assert next((tmp_path / "runs").glob("*/cuda-smoke.json")).exists()
    assert "--cleanup" in capsys.readouterr().err


def test_cleanup_failure_is_not_masked_by_status_write_failure(
    tmp_path, training_file, monkeypatch, capsys
):
    def cannot_save(*args):
        raise OSError("status file is unwritable")

    monkeypatch.setattr(terraform_tools, "write_json", cannot_save)
    invoke, cloud = invoke_fake(tmp_path, training_file, "destroy")
    with pytest.raises(RunError, match="destroy failed"):
        invoke()

    assert cloud["resources"] == {"container", "bucket"}
    assert next((tmp_path / "runs").glob("*/cuda-smoke.json")).exists()
    error = capsys.readouterr().err
    assert "Could not save cleanup status 'pending'" in error
    assert "Could not save cleanup status 'failed'" in error
    assert '--cleanup "' in error


@pytest.mark.parametrize("status_writable", [True, False])
def test_cleanup_command_retries_failed_destroy_and_keeps_report(
    tmp_path, training_file, monkeypatch, capsys, status_writable
):
    invoke, cloud = invoke_fake(tmp_path, training_file, "destroy")
    with pytest.raises(RunError):
        invoke()
    directory = next((tmp_path / "runs").iterdir())
    if not status_writable:
        (directory / "run.tmp").mkdir()
    monkeypatch.setenv("VASTAI_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(
        runner,
        "recover",
        partial(runner.recover, commands_type=lambda directory: FakeTools(directory, cloud)),
    )
    assert runner.main(["--cleanup", str(directory)]) == 0
    assert cloud["resources"] == set()
    assert saved_run(tmp_path)["cleanup"] == ("complete" if status_writable else "failed")
    assert json.loads((directory / "cuda-smoke.json").read_text()) == valid_report()
    # Recovery is idempotent after the resources have already gone.
    assert runner.main(["--cleanup", str(directory)]) == 0
    assert cloud["resources"] == set()
    if not status_writable:
        assert "Could not save cleanup status 'complete'" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["terraform.tfstate", "run.auto.tfvars.json"])
def test_cleanup_refuses_missing_recovery_files(
    tmp_path, training_file, monkeypatch, capsys, missing
):
    invoke, cloud = invoke_fake(tmp_path, training_file, "destroy")
    with pytest.raises(RunError):
        invoke()
    directory = next((tmp_path / "runs").iterdir())
    (directory / "terraform" / missing).unlink()
    monkeypatch.setenv("VAST_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(
        runner,
        "recover",
        partial(runner.recover, commands_type=lambda path: FakeTools(path, cloud)),
    )
    assert runner.main(["--cleanup", str(directory)]) == 1
    assert "recovery files are missing" in capsys.readouterr().err
    assert cloud["resources"] == {"container", "bucket"}
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


@pytest.mark.parametrize("argument", ["", "  ", "\t"])
def test_cli_rejects_blank_gpu_names_before_requiring_credentials(argument, monkeypatch, capsys):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        runner.main(["--gpu", "RTX 3090", "--gpu", argument])
    assert error.value.code == 2
    assert "nonblank GPU model name" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ([], ["RTX 3090", "RTX PRO 4000"]),
        (["--gpu", "RTX PRO 6000 WS"], ["RTX PRO 6000 WS"]),
        (
            ["--gpu", "RTX PRO 4000", "--gpu", "RTX 3090"],
            ["RTX PRO 4000", "RTX 3090"],
        ),
        (
            ["--gpu", " RTX 3090 ", "--gpu", "RTX PRO 4000", "--gpu", "RTX 3090"],
            ["RTX 3090", "RTX PRO 4000"],
        ),
    ],
)
def test_cli_saves_gpu_allowlist_in_terraform_inputs(
    tmp_path, training_file, monkeypatch, arguments, expected
):
    cloud = {"resources": set(), "tracked": set(), "rentals": 0, "uploaded": []}
    monkeypatch.setenv("VAST_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(
        runner,
        "run_smoke",
        partial(
            runner.run_smoke,
            runs_directory=tmp_path / "runs",
            commands_type=lambda directory: FakeTools(directory, cloud),
            transfer_type=lambda region: fake_transfer(cloud),
        ),
    )
    assert runner.main(["--dataset", str(training_file), *arguments]) == 0
    variables_path = next((tmp_path / "runs").glob("*/terraform/run.auto.tfvars.json"))
    variables = json.loads(variables_path.read_text())
    assert variables["gpu_names"] == expected
    assert "gpu_name" not in variables
    assert saved_run(tmp_path)["outcome"] == "passed"
    assert cloud["resources"] == set()


def test_cli_requires_credentials_before_starting_a_run(monkeypatch, capsys):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        runner.main([])
    assert error.value.code == 2
    assert capsys.readouterr().err.rstrip().endswith("error: Set VAST_API_KEY before running.")


@pytest.mark.parametrize("existing_key", [None, "shell-key"])
def test_cli_loads_project_dotenv_without_overriding_shell_credentials(
    tmp_path, monkeypatch, capsys, existing_key
):
    (tmp_path / ".env").write_text(
        'VAST_API_KEY=file-key\nVAST_IMAGE_LOGIN="-u fixture -p registry-key ghcr.io"\n'
    )
    other_directory = tmp_path / "elsewhere"
    other_directory.mkdir()
    (other_directory / ".env").write_text("VAST_API_KEY=wrong-directory-key\n")
    monkeypatch.chdir(other_directory)
    if existing_key:
        monkeypatch.setenv("VAST_API_KEY", existing_key)

    with pytest.raises(SystemExit) as result:
        runner.main(["--help"])

    assert result.value.code == 0
    assert os.environ["VAST_API_KEY"] == (existing_key or "file-key")
    commands = command_tools.Commands(tmp_path)
    assert commands.api_key == (existing_key or "file-key")
    assert commands.image_login == "-u fixture -p registry-key ghcr.io"
    assert "VAST_API_KEY" not in commands.environment
    assert "VAST_IMAGE_LOGIN" not in commands.environment
    output = capsys.readouterr()
    assert "file-key" not in output.out + output.err
    assert "registry-key" not in output.out + output.err


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
    with pytest.raises(RunError, match="does not prove"):
        artifacts.validate_report(path)


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "another-model"},
        {"model_revision": "main"},
        {"model_class": "AnotherModel"},
        {"parameter_count": 0},
        {"cuda_parameter_count": 100},
        {"parameter_dtype": "float32"},
        {"tokenizer_vocab_size": 0},
        {"tokenizer_probe_tokens": 0},
        {"transformers_version": "0.0.0"},
        {"load_seconds": float("nan")},
        {"load_seconds": float("inf")},
        {"load_seconds": -1},
        {"cuda_memory_allocated_bytes": 0},
        {"cuda_peak_memory_allocated_bytes": 1},
        {"cuda_device_count": 0},
        {"cuda_runtime_version": "12.8"},
    ],
)
def test_model_report_rejects_wrong_checkpoint_cpu_weights_and_missing_evidence(tmp_path, change):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(valid_model_report() | change))
    with pytest.raises(RunError, match="pinned model check"):
        artifacts.validate_report(path, check="model")


def test_cuda_report_cannot_pass_as_a_model_load(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(valid_report()))
    with pytest.raises(RunError, match="pinned model check"):
        artifacts.validate_report(path, check="model")


def test_entry_point_help_works_from_another_directory(tmp_path, monkeypatch):
    for name in ("VAST_API_KEY", "VASTAI_API_KEY", "PYTHONPATH"):
        monkeypatch.delenv(name, raising=False)
    result = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert "--gpu NAME" in result.stdout
    assert "--cleanup" in result.stdout


@pytest.mark.skipif(shutil.which("terraform") is None, reason="Terraform is not installed")
@pytest.mark.parametrize("artifacts_writable", [True, False])
def test_shared_configuration_keeps_run_states_separate_during_cleanup(
    tmp_path, monkeypatch, capsys, artifacts_writable
):
    configuration = tmp_path / "shared configuration"
    configuration.mkdir()
    (configuration / "main.tf").write_text(
        """terraform {
  backend "local" {}
}
variable "label" { type = string }
resource "terraform_data" "run" { input = var.label }
output "label" { value = terraform_data.run.output }
"""
    )
    monkeypatch.setattr(terraform_tools, "TERRAFORM_DIRECTORY", configuration)
    monkeypatch.setenv("TF_DATA_DIR", str(tmp_path / "unrelated-cache"))
    monkeypatch.setenv("TF_WORKSPACE", "unrelated-workspace")
    runs = []
    for label in ("first", "second"):
        directory = tmp_path / label
        infrastructure = directory / "terraform"
        infrastructure.mkdir(parents=True)
        artifacts.write_json(infrastructure / "run.auto.tfvars.json", {"label": label})
        commands = command_tools.Commands(directory)
        terraform_tools.terraform(commands, "init", "-input=false", "-lockfile=readonly")
        terraform_tools.terraform(commands, "apply", "-auto-approve", "-input=false")
        runs.append(commands)
        assert (infrastructure / "terraform.tfstate").is_file()
        assert (infrastructure / ".terraform" / "terraform.tfstate").is_file()
        assert terraform_tools.terraform(commands, "output", "-raw", "label").stdout == label

    first, second = runs
    second_state_path = second.directory / "terraform/terraform.tfstate"
    second_state = second_state_path.read_bytes()
    if not artifacts_writable:
        # Real filesystem errors for both status writes and the two teardown logs.
        (first.directory / "run.tmp").mkdir()
        for sequence in (first.sequence + 1, first.sequence + 2):
            (first.directory / f"{sequence:02d}-terraform.log").mkdir()
    state = {"apply_complete": True}
    terraform_tools.cleanup(first, state)
    assert state["cleanup"] == "complete"
    assert terraform_tools.terraform(first, "state", "list").stdout.strip() == ""
    if not artifacts_writable:
        error = capsys.readouterr().err
        assert "Could not save cleanup status 'pending'" in error
        assert "Could not save cleanup status 'complete'" in error
        assert error.count("Could not save command log") == 2
    assert second_state_path.read_bytes() == second_state
    assert terraform_tools.terraform(second, "output", "-raw", "label").stdout == "second"
    assert not (configuration / "terraform.tfstate").exists()
    assert not (configuration / ".terraform").exists()
    assert not (tmp_path / "unrelated-cache").exists()
    terraform_tools.cleanup(second, {"apply_complete": True})


def test_command_logs_survive_a_later_cleanup_session(tmp_path):
    first = command_tools.Commands(tmp_path)
    first.run([sys.executable, "-c", "print('original log')"])
    second = command_tools.Commands(tmp_path)
    result = second.run([sys.executable, "-c", "print('cleanup log')"])
    assert result.stdout.strip() == "cleanup log"
    assert {path.read_text().strip() for path in tmp_path.glob("*.log")} == {
        "original log",
        "cleanup log",
    }


def test_command_failure_retains_diagnostics(tmp_path):
    commands = command_tools.Commands(tmp_path)
    with pytest.raises(command_tools.CommandError, match="failed; see") as raised:
        commands.run([sys.executable, "-c", "print('failure details'); raise SystemExit(3)"])
    assert raised.value.output.strip() == "failure details"
    assert "failure details" in next(tmp_path.glob("*.log")).read_text()


def test_command_failure_is_reported_when_cleanup_log_cannot_be_saved(tmp_path, capsys):
    commands = command_tools.Commands(tmp_path)
    (tmp_path / f"01-{Path(sys.executable).stem}.log").mkdir()
    with pytest.raises(RunError, match="failed; command log could not be saved"):
        commands.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('ran').touch(); raise SystemExit(3)",
            ],
            best_effort_log=True,
        )
    assert (tmp_path / "ran").exists()
    assert "Could not save command log" in capsys.readouterr().err


def test_command_timeout_returns_control_for_cleanup(tmp_path):
    commands = command_tools.Commands(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        commands.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.1)


def test_api_credentials_are_not_inherited_by_subprocesses(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "test-primary-key")
    monkeypatch.setenv("VASTAI_API_KEY", "test-legacy-key")
    monkeypatch.setenv("VAST_IMAGE_LOGIN", "test-registry-login")
    result = command_tools.Commands(tmp_path).run(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('VAST_API_KEY'), os.environ.get('VASTAI_API_KEY'), os.environ.get('VAST_IMAGE_LOGIN'))",
        ]
    )
    assert result.stdout.strip() == "None None None"


def test_terraform_receives_api_key_through_environment_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "test-primary-key")
    monkeypatch.setenv("VASTAI_API_KEY", "test-legacy-key")
    monkeypatch.setenv("VAST_IMAGE_LOGIN", "test-registry-login")
    start_process = subprocess.Popen

    def terraform_probe(arguments, **options):
        assert arguments == ["terraform", "version"]
        return start_process(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ.get('VAST_API_KEY'), os.environ.get('VASTAI_API_KEY'), os.environ.get('TF_INPUT'), os.environ.get('TF_VAR_image_login'))",
            ],
            **options,
        )

    monkeypatch.setattr(command_tools.subprocess, "Popen", terraform_probe)
    result = command_tools.Commands(tmp_path).run(["terraform", "version"])
    assert result.stdout.strip() == "None test-primary-key 0 test-registry-login"


def test_sample_upload_failure_cleans_bucket_without_renting_gpu(tmp_path, training_file):
    invoke, cloud = invoke_fake(tmp_path, training_file, "upload")
    with pytest.raises(RunError, match="upload failed"):
        invoke()
    assert cloud["rentals"] == 0
    assert cloud["resources"] == set()
    assert saved_run(tmp_path)["cleanup"] == "complete"


@pytest.mark.parametrize(
    "repository",
    ["local-image", "ghcr.io/owner/image:tag", "ghcr.io/owner/image@sha256:abc", "-option"],
)
def test_image_publication_rejects_repositories_with_no_registry_or_with_a_tag(
    tmp_path, repository
):
    with pytest.raises(RunError, match="registry hostname and no tag or digest"):
        publish(FakeTools(tmp_path, {}), repository, "run-id")
