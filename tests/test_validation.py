"""Full-split inference, reference isolation, and complete local prediction artifacts."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_generation import inference as inference
from test_vast_runner import (
    FakeTools,
    artifacts,
    fake_transfer,
    runner,
    saved_run,
    valid_inference_report,
    valid_model_report,
)

with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(Path(__file__).parents[1] / "experiments/cuda"))
    import validation_data as data
    import validation_inference


@pytest.fixture
def split(tmp_path):
    source = tmp_path / "validation.parquet"
    reviewed = tmp_path / "reviewed_validation.parquet"
    rows = [
        {"id": "second-id", "subject": "PRIVATE_REFERENCE_A", "diff": "+first change"},
        {"id": "first-id", "subject": "PRIVATE_REFERENCE_B", "diff": "+second change"},
    ]
    schema = pa.schema(
        [
            pa.field("id", pa.string(), metadata={b"secret": b"private-field-metadata"}),
            pa.field("subject", pa.string()),
            pa.field("diff", pa.string()),
            pa.field("repository_group_id", pa.string()),
        ],
        metadata={b"secret": b"private-schema-metadata"},
    )
    pq.write_table(
        pa.Table.from_pylist(
            [row | {"repository_group_id": "private-repo"} for row in rows], schema=schema
        ),
        source,
    )
    pq.write_table(pa.Table.from_pylist(rows[1:]), reviewed)
    return source, reviewed, rows


def test_prepare_validation_exports_all_diffs_in_order_and_keeps_references_local(tmp_path, split):
    source, reviewed, rows = split
    metadata = data.prepare_validation(source, tmp_path, reviewed)
    inputs = pq.read_table(tmp_path / "validation-input.parquet")
    assert inputs.to_pylist() == [{"id": row["id"], "diff": row["diff"]} for row in rows]
    assert inputs.schema.metadata is None
    assert all(field.metadata is None for field in inputs.schema)
    assert pq.read_table(tmp_path / "validation-references.parquet").to_pylist() == [
        {"id": row["id"], "reference_subject": row["subject"], "reference_reviewed": index == 1}
        for index, row in enumerate(rows)
    ]
    assert metadata == {
        "split": "validation",
        "source_name": source.name,
        "source_sha256": data.file_sha256(source),
        "rows": 2,
        "input_sha256": data.file_sha256(tmp_path / "validation-input.parquet"),
        "reviewed_source_name": reviewed.name,
        "reviewed_sha256": data.file_sha256(reviewed),
        "reviewed_rows": 1,
    }


def test_missing_optional_review_file_records_unknown_review_status(tmp_path, split):
    metadata = data.prepare_validation(split[0], tmp_path)
    references = pq.read_table(tmp_path / "validation-references.parquet")
    assert references["reference_reviewed"].to_pylist() == [None, None]
    assert metadata["reviewed_source_name"] is metadata["reviewed_sha256"] is None
    assert metadata["reviewed_rows"] == 0


@pytest.mark.parametrize("change", [{"id": "absent"}, {"subject": "changed"}, {"diff": "+changed"}])
def test_changed_or_foreign_reviewed_references_fail_before_export(tmp_path, split, change):
    source, reviewed, rows = split
    pq.write_table(pa.Table.from_pylist([rows[1] | change]), reviewed)
    with pytest.raises(ValueError, match="do not match"):
        data.prepare_validation(source, tmp_path, reviewed)
    assert not (tmp_path / "validation-input.parquet").exists()


@pytest.mark.parametrize(
    "records,columns,exact,message",
    [
        ([{"id": "a"}], ["id", "diff"], False, "must contain"),
        ([{"id": "a", "diff": "+x", "subject": "reference"}], ["id", "diff"], True, "only"),
        ([], ["id", "diff"], True, "no rows"),
        ([{"id": "a", "diff": None}], ["id", "diff"], True, "nonempty"),
        ([{"id": " ", "diff": "+x"}], ["id", "diff"], True, "nonempty"),
        ([{"id": "a", "diff": 5}], ["id", "diff"], True, "nonempty"),
        ([{"id": "a", "diff": "+x"}] * 2, ["id", "diff"], True, "unique"),
    ],
)
def test_invalid_inputs_cannot_start_validation(tmp_path, records, columns, exact, message):
    path = tmp_path / "input.parquet"
    table = pa.Table.from_pylist(records) if records else pa.table({"id": [], "diff": []})
    pq.write_table(table, path)
    with pytest.raises(ValueError, match=message):
        data.read_rows(path, columns, exact=exact)


def test_validation_loads_once_generates_every_diff_and_never_supplies_a_reference(
    tmp_path, split, inference, monkeypatch
):
    source, reviewed, rows = split
    data.prepare_validation(source, tmp_path, reviewed)
    model, tokenizer = inference
    loads = []
    prompts = []
    generate = model.generate

    def capture(**kwargs):
        prompts.append(tokenizer.messages)
        model.response = "Add change" if len(prompts) == 1 else "x" * 73
        return generate(**kwargs)

    def load():
        loads.append(True)
        return model, tokenizer, valid_model_report()

    monkeypatch.setattr(model, "generate", capture)
    monkeypatch.setattr(validation_inference, "load_model", load)
    report_path = tmp_path / "validation.json"
    report_path.write_text('{"stale":true}')
    validation_inference.main(tmp_path / "validation-input.parquet", report_path)
    report = json.loads(report_path.read_text())
    assert len(loads) == 1
    assert [prompt[1]["content"] for prompt in prompts] == [row["diff"] for row in rows]
    assert "PRIVATE_REFERENCE" not in json.dumps(prompts)
    assert report["row_count"] == 2
    assert report["input_sha256"] == data.file_sha256(tmp_path / "validation-input.parquet")
    assert [row["id"] for row in report["predictions"]] == [row["id"] for row in rows]
    assert report["predictions"][0]["subject"] == "Add change"
    assert report["predictions"][1]["subject"] is None
    assert report["predictions"][1]["raw_output"] == "x" * 73
    assert report["predictions"][1]["errors"] == ["too_long"]
    assert "stale" not in report


def test_failed_generation_cannot_leave_a_stale_success_report(
    tmp_path, split, inference, monkeypatch
):
    data.prepare_validation(split[0], tmp_path)
    monkeypatch.setattr(
        validation_inference, "load_model", lambda: (*inference, valid_model_report())
    )
    monkeypatch.setattr(
        validation_inference,
        "generate_subject",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("GPU failed")),
    )
    report = tmp_path / "validation.json"
    report.write_text('{"row_count":2}')
    with pytest.raises(RuntimeError, match="GPU failed"):
        validation_inference.main(tmp_path / "validation-input.parquet", report)
    assert not report.exists()


def report_for(directory, rows):
    return valid_model_report() | {
        "row_count": len(rows),
        "input_sha256": data.file_sha256(directory / "validation-input.parquet"),
        "predictions": [
            {"id": row["id"], **valid_inference_report()["examples"]["single_file"]} for row in rows
        ],
    }


def test_saved_predictions_pair_every_id_reference_and_diff_and_retain_invalid_outputs(
    tmp_path, split
):
    source, reviewed, rows = split
    data.prepare_validation(source, tmp_path, reviewed)
    report = report_for(tmp_path, rows)
    for row in report["predictions"]:
        row.update(raw_output="x" * 73, subject=None, valid=False, errors=["too_long"])
    path = data.save_predictions(report, tmp_path, run_id="fixture")
    table = pq.read_table(path)
    assert table["id"].to_pylist() == [row["id"] for row in rows]
    assert table["reference_subject"].to_pylist() == [row["subject"] for row in rows]
    assert table["diff"].to_pylist() == [row["diff"] for row in rows]
    assert table["reference_reviewed"].to_pylist() == [False, True]
    assert table["raw_output"].to_pylist() == ["x" * 73] * 2
    assert table["predicted_subject"].to_pylist() == [None, None]
    assert table.schema.field("predicted_subject").type == pa.string()
    assert table["errors"].to_pylist() == [["too_long"]] * 2
    metadata = json.loads(table.schema.metadata[b"qwen_commit"])
    assert metadata["run_id"] == "fixture"
    assert metadata["model_revision"] == report["model_revision"]
    assert metadata["generation_config"] == report["predictions"][0]["generation_config"]
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize(
    "failure", ["hash", "count", "missing", "duplicate", "reordered", "reference-order"]
)
def test_incomplete_or_mismatched_predictions_cannot_be_joined(tmp_path, split, failure):
    source, _, rows = split
    data.prepare_validation(source, tmp_path)
    report = report_for(tmp_path, rows)
    if failure == "hash":
        report["input_sha256"] = "0" * 64
    elif failure == "count":
        report["row_count"] = 1
    elif failure == "missing":
        report["predictions"].pop()
    elif failure == "duplicate":
        report["predictions"][1]["id"] = report["predictions"][0]["id"]
    elif failure == "reordered":
        report["predictions"].reverse()
    else:
        path = tmp_path / "validation-references.parquet"
        pq.write_table(pq.read_table(path).take([1, 0]), path)
    with pytest.raises(ValueError, match="complete uploaded split"):
        data.save_predictions(report, tmp_path, run_id="fixture")
    assert not (tmp_path / "validation-predictions.parquet").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"predictions": []},
        {"predictions": {}},
        {"row_count": True},
        {"row_count": 5},
        {"input_sha256": "invalid"},
        {"input_sha256": None},
    ],
)
def test_validation_report_rejects_missing_or_inconsistent_batch_evidence(tmp_path, split, change):
    data.prepare_validation(split[0], tmp_path)
    report = report_for(tmp_path, split[2]) | change
    path = tmp_path / "report.json"
    artifacts.write_json(path, report)
    with pytest.raises(runner.RunError, match="pinned validation"):
        artifacts.validate_report(path, check="validation")


@pytest.mark.parametrize(
    "change",
    [
        {"id": ""},
        {"id": None},
        {"id": "second-id"},
        {"system_prompt": "wrong"},
        {"valid": False},
        {
            "generation_config": valid_inference_report()["examples"]["single_file"][
                "generation_config"
            ]
            | {"pad_token_id": 123}
        },
    ],
)
def test_validation_report_rejects_bad_ids_and_generation_evidence(tmp_path, split, change):
    data.prepare_validation(split[0], tmp_path)
    report = report_for(tmp_path, split[2])
    report["predictions"][1].update(change)
    path = tmp_path / "report.json"
    artifacts.write_json(path, report)
    with pytest.raises(runner.RunError, match="pinned validation"):
        artifacts.validate_report(path, check="validation")


@pytest.mark.parametrize("failure", [None, "smoke", "report"])
def test_validation_lifecycle_uploads_only_inputs_saves_complete_predictions_and_cleans_up(
    tmp_path, split, failure
):
    source, reviewed, rows = split
    cloud = {"resources": set(), "tracked": set(), "rentals": 0, "uploaded": []}

    def run():
        return runner.run_smoke(
            source,
            gpu_names=["RTX 3090"],
            max_hourly_price=0.5,
            timeout_minutes=60,
            check="validation",
            reviewed_validation=reviewed,
            runs_directory=tmp_path / "runs",
            commands_type=lambda directory: FakeTools(directory, cloud, failure),
            transfer_type=lambda region: fake_transfer(cloud),
        )

    if failure:
        with pytest.raises(runner.RunError):
            run()
    else:
        output = run()
        assert output.name == "validation-predictions.parquet"
        assert pq.read_table(output)["reference_subject"].to_pylist() == [
            row["subject"] for row in rows
        ]
        assert (
            pq.read_table(output)["predicted_subject"].to_pylist()
            == ["Trim whitespace from names"] * 2
        )
    assert cloud["uploaded"] == [{"id": row["id"], "diff": row["diff"]} for row in rows]
    assert cloud["rentals"] == 1
    assert cloud["resources"] == set()
    state = saved_run(tmp_path)
    assert state["dataset"]["rows"] == 2
    assert state["dataset"]["reviewed_rows"] == 1
    assert state["outcome"] == ("failed" if failure else "passed")
    assert state["cleanup"] == "complete"


def test_cli_validation_defaults_to_validation_split(tmp_path, split, monkeypatch):
    prepared = tmp_path / "notebooks/prepared"
    prepared.mkdir(parents=True)
    (prepared / "validation.parquet").write_bytes(split[0].read_bytes())
    reserved = prepared / "test.parquet"
    reserved.write_bytes(b"Reserved test data must not be opened as Parquet")
    cloud = {"resources": set(), "tracked": set(), "rentals": 0, "uploaded": []}
    monkeypatch.setattr(runner, "PROJECT_DIRECTORY", tmp_path)
    monkeypatch.setenv("VAST_API_KEY", "fixture")
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
    assert runner.main(["--check", "validation", "--reviewed-validation", str(split[1])]) == 0
    assert saved_run(tmp_path)["dataset"]["source_name"] == "validation.parquet"
    assert len(cloud["uploaded"]) == 2
    assert cloud["resources"] == set()
    assert reserved.read_bytes() == b"Reserved test data must not be opened as Parquet"


def test_reviewed_reference_option_rejects_nonvalidation_checks():
    with pytest.raises(SystemExit) as raised:
        runner.main(["--check", "inference", "--reviewed-validation", "unused.parquet"])
    assert raised.value.code == 2
