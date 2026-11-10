"""Local baseline metrics and evidence-bound semantic reviews."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_validation import data, report_for

with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(Path(__file__).parents[1] / "experiments/cuda"))
    import evaluate as evaluation


@pytest.fixture
def run_directory(tmp_path):
    rows = [
        {"id": name, "subject": f"Reference {name}", "diff": f"+Change {name}"}
        for name in ("z", "a", "m")
    ]
    source, approved = tmp_path / "source.parquet", tmp_path / "approved.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    pq.write_table(pa.Table.from_pylist(rows[:2]), approved)
    data.prepare_validation(source, tmp_path, approved)
    report = report_for(tmp_path, rows)
    for row, seconds in zip(report["predictions"], (1.0, 2.0, 9.0), strict=True):
        row["generation_seconds"] = seconds
    report["predictions"][1].update(
        raw_output="`" + "x" * 73 + "`", subject=None, valid=False, errors=["too_long", "markdown"]
    )
    report["predictions"][2].update(
        diff_tokens=5000, diff_tokens_used=4096, diff_truncated=True, prompt_tokens=4200
    )
    (tmp_path / "validation.json").write_text(json.dumps(report))
    data.save_predictions(report, tmp_path, run_id="fixture")
    return tmp_path


def completed_review(directory):
    review = json.loads((directory / "semantic-review-template.json").read_text())
    review["reviewer"] = "Test reviewer"
    for row, rating in zip(review["examples"], ("partial", "incorrect"), strict=True):
        row.update(rating=rating, rationale=f"Evidence for {row['id']}.")
    path = directory / "review.json"
    path.write_text(json.dumps(review))
    return path, review


def test_cli_writes_json_only_with_all_rows_and_approved_subset(run_directory, capsys):
    # A reserved file that would fail if evaluation tried to read it as Parquet.
    (run_directory / "test.parquet").write_bytes(b"Do not open")
    original = (run_directory / data.PREDICTIONS_FILENAME).read_bytes()
    evaluation.main([str(run_directory)])
    result = json.loads((run_directory / "evaluation.json").read_text())
    group = result["cohorts"]["all_validation"]
    assert group["rows"] == 3 and group["format_passes"] == 2
    assert group["format_pass_rate"] == pytest.approx(2 / 3)
    assert group["error_counts"] == {"markdown": 1, "too_long": 1}
    assert group["truncated_inputs"] == 1 and group["finish_reasons"] == {"eos": 3}
    assert group["generation_seconds"] == pytest.approx(
        {"min": 1, "median": 2, "p95": 8.3, "max": 9, "mean": 4, "total": 12}
    )
    approved = result["cohorts"]["approved_references"]
    assert approved["rows"] == 2 and approved["format_pass_rate"] == 0.5
    assert approved["generation_seconds"]["median"] == 1.5
    assert result["first_generation_seconds"] == 1
    assert result["semantic_review"]["status"] == "pending"
    assert result["predictions_sha256"] == data.file_sha256(
        run_directory / data.PREDICTIONS_FILENAME
    )
    assert not (run_directory / "evaluation.md").exists()
    assert capsys.readouterr().out == f"Evaluation saved to {run_directory / 'evaluation.json'}\n"
    assert (run_directory / data.PREDICTIONS_FILENAME).read_bytes() == original


def test_sample_is_capped_sorted_and_includes_format_failures_without_using_their_scores():
    rows = [
        {"id": f"id-{i:02d}", "reference_reviewed": True, "valid": i % 2 == 0}
        for i in reversed(range(14))
    ]
    rows.extend(
        [
            {"id": "0-false", "reference_reviewed": False},
            {"id": "0-null", "reference_reviewed": None},
        ]
    )
    selected = evaluation.review_sample(rows)
    assert [row["id"] for row in selected] == [f"id-{i:02d}" for i in range(12)]
    assert sum(row["valid"] for row in selected) == 6


def test_completed_review_aggregates_ratings_but_preserves_original_evidence(run_directory):
    evaluation.evaluate(run_directory)
    path, review = completed_review(run_directory)
    review["examples"][0]["raw_output"] = "Forged copied evidence"
    path.write_text(json.dumps(review))
    template_before = (run_directory / "semantic-review-template.json").read_bytes()
    evaluation.main([str(run_directory), "--review", str(path)])
    result = json.loads((run_directory / "evaluation.json").read_text())
    semantic = result["semantic_review"]
    assert semantic["status"] == "complete" and semantic["reviewer"] == "Test reviewer"
    assert semantic["ratings"] == {"accurate": 0, "partial": 1, "incorrect": 1, "uncertain": 0}
    assert [row["id"] for row in semantic["examples"]] == ["a", "z"]
    assert semantic["examples"][0]["raw_output"] == "`" + "x" * 73 + "`"
    assert semantic["examples"][0]["format_errors"] == ["too_long", "markdown"]
    assert semantic["examples"][0]["rationale"] == "Evidence for a."
    assert (run_directory / "semantic-review-template.json").read_bytes() == template_before


@pytest.mark.parametrize(
    "failure", ["hash", "missing", "reordered", "duplicate", "reviewer", "rating", "rationale"]
)
def test_unbound_or_incomplete_semantic_review_cannot_replace_evaluation(run_directory, failure):
    evaluation.evaluate(run_directory)
    previous = (run_directory / "evaluation.json").read_bytes()
    path, review = completed_review(run_directory)
    if failure == "hash":
        review["predictions_sha256"] = "0" * 64
    elif failure == "missing":
        review["examples"].pop()
    elif failure == "reordered":
        review["examples"].reverse()
    elif failure == "duplicate":
        review["examples"][1]["id"] = review["examples"][0]["id"]
    elif failure == "reviewer":
        review["reviewer"] = " "
    elif failure == "rating":
        review["examples"][0]["rating"] = "probably good"
    else:
        review["examples"][0]["rationale"] = ""
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match=r"Review must|Every review"):
        evaluation.evaluate(run_directory, path)
    assert (run_directory / "evaluation.json").read_bytes() == previous


def test_unknown_reference_approval_is_unavailable_and_single_row_latency_is_defined(run_directory):
    source = run_directory / "source.parquet"
    rows = pq.read_table(source).to_pylist()[:1]
    pq.write_table(pa.Table.from_pylist(rows), source)
    data.prepare_validation(source, run_directory)
    report = report_for(run_directory, rows)
    (run_directory / "validation.json").write_text(json.dumps(report))
    data.save_predictions(report, run_directory, run_id="single")
    result = evaluation.evaluate(run_directory)
    empty = result["cohorts"]["approved_references"]
    assert empty["rows"] == 0 and empty["format_pass_rate"] is None
    assert empty["generation_seconds"] is None and empty["error_counts"] == {}
    assert result["semantic_review"]["status"] == "unavailable"
    timing = result["cohorts"]["all_validation"]["generation_seconds"]
    assert timing["min"] == timing["median"] == timing["p95"] == timing["max"]


def test_changed_prediction_artifact_cannot_be_evaluated(run_directory):
    path = run_directory / data.PREDICTIONS_FILENAME
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("reference_subject"),
        "reference_subject",
        pa.array(["changed"] * 3),
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="differ from the validated report"):
        evaluation.evaluate(run_directory)
    assert not (run_directory / "evaluation.json").exists()
