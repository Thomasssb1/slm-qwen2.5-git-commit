"""Evaluate a saved validation run locally, without loading a model or reading test data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import pyarrow.compute as pc
import pyarrow.parquet as pq
from validation_data import PREDICTIONS_FILENAME, file_sha256, join_predictions
from vast_runner.artifacts import validate_report, write_json

REVIEW_SIZE = 12
REVIEW_SELECTION = "First 12 approved-reference rows ordered lexicographically by ID."
RUBRIC = {
    "accurate": "Captures the main change without a material unsupported claim.",
    "partial": "Describes a supported aspect but misses the main change or an important qualifier.",
    "incorrect": "Invents a change, reverses its direction, or materially misstates what changed.",
    "uncertain": "The saved diff does not provide enough evidence for a confident assessment.",
}
LATENCY_SCOPE = (
    "CUDA-synchronized model.generate time, including the first call. Excludes tokenization, "
    "model loading, provisioning, and transfer. Median/p95 use linear interpolation."
)
REVIEW_SCOPE = (
    "Assess raw output against the full saved diff, using the reference as context rather than "
    "ground truth. Formatting is separate. Truncated inputs are flagged; do not assume the model "
    "saw the full diff. This descriptive sample is not a full-split accuracy estimate."
)


def metrics(rows: list[dict]) -> dict:
    times = [row["generation_seconds"] for row in rows]
    accepted = sum(row["valid"] for row in rows)
    latency = None
    if times:
        median, p95 = pc.quantile(times, q=[0.5, 0.95], interpolation="linear").to_pylist()
        latency = {
            "min": min(times),
            "median": median,
            "p95": p95,
            "max": max(times),
            "mean": fmean(times),
            "total": sum(times),
        }
    return {
        "rows": len(rows),
        "format_passes": accepted,
        "format_pass_rate": accepted / len(rows) if rows else None,
        "error_counts": dict(sorted(Counter(e for row in rows for e in row["errors"]).items())),
        "truncated_inputs": sum(row["diff_truncated"] for row in rows),
        "finish_reasons": dict(sorted(Counter(row["finish_reason"] for row in rows).items())),
        "generation_seconds": latency,
    }


def review_sample(rows: list[dict]) -> list[dict]:
    return sorted(
        (row for row in rows if row["reference_reviewed"] is True), key=lambda r: r["id"]
    )[:REVIEW_SIZE]


def review_template(rows: list[dict], digest: str) -> dict:
    return {
        "predictions_sha256": digest,
        "reviewer": "",
        "selection": REVIEW_SELECTION,
        "scope": REVIEW_SCOPE,
        "rubric": RUBRIC,
        "examples": [
            {
                "id": row["id"],
                "raw_output": row["raw_output"],
                "reference_subject": row["reference_subject"],
                "diff_truncated": row["diff_truncated"],
                "format_errors": row["errors"],
                "diff": row["diff"],
                "rating": None,
                "rationale": "",
            }
            for row in review_sample(rows)
        ],
    }


def semantic_review(template: dict, path: Path | None) -> dict:
    result = {
        "selection": REVIEW_SELECTION,
        "scope": REVIEW_SCOPE,
        "rubric": RUBRIC,
        "sample_size": len(template["examples"]),
        "status": "pending" if template["examples"] else "unavailable",
    }
    if path is None:
        return result
    review = json.loads(path.read_text(encoding="utf-8"))
    examples = review.get("examples", [])
    if (
        review.get("predictions_sha256") != template["predictions_sha256"]
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not examples
        or [row["id"] for row in examples] != [row["id"] for row in template["examples"]]
    ):
        raise ValueError("Review must identify its reviewer and match this run's complete sample.")
    assessed = []
    for source, annotation in zip(template["examples"], examples, strict=True):
        if (
            annotation.get("rating") not in RUBRIC
            or not isinstance(annotation.get("rationale"), str)
            or not annotation["rationale"].strip()
        ):
            raise ValueError("Every review example needs a rubric rating and evidence rationale.")
        # Display only original evidence; copied text in an annotation is not authoritative.
        assessed.append(
            {key: value for key, value in source.items() if key != "diff"}
            | {"rating": annotation["rating"], "rationale": annotation["rationale"]}
        )
    counts = Counter(row["rating"] for row in assessed)
    result.update(
        status="complete",
        reviewer=review["reviewer"],
        ratings={rating: counts[rating] for rating in RUBRIC},
        examples=assessed,
    )
    return result


def render_markdown(result: dict) -> str:
    lines = [
        "# Validation baseline evaluation",
        "",
        f"Run: `{result['run_id']}`. Predictions SHA-256: `{result['predictions_sha256']}`.",
        "",
        f"Model: `{result['model_id']}` at `{result['model_revision']}`.",
        f"GPU: {result['gpu']}. Model download/load: {result['load_seconds']:.3f} seconds.",
        "",
        "## Formatting and latency",
        "",
        "| Cohort | Rows | Format passes | Pass rate | Truncated | Median (s) | p95 (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in result["cohorts"].items():
        rate = f"{group['format_pass_rate']:.1%}" if group["rows"] else "n/a"
        timing = group["generation_seconds"]
        median = f"{timing['median']:.4f}" if timing else "n/a"
        p95 = f"{timing['p95']:.4f}" if timing else "n/a"
        lines.append(
            f"| {name} | {group['rows']} | {group['format_passes']} | {rate} | "
            f"{group['truncated_inputs']} | {median} | {p95} |"
        )
    lines.extend(["", LATENCY_SCOPE, "", "Error counts can overlap within a row.", ""])
    for name, group in result["cohorts"].items():
        lines.append(f"- {name}: `{json.dumps(group['error_counts'], sort_keys=True)}`")
    lines.extend(
        [
            "",
            "Formatting uses the existing mechanical checker; it does not prove semantic correctness "
            "or detect every possible explanation. No outputs were repaired or regenerated.",
            "",
            "## Correctness sample",
            "",
            REVIEW_SELECTION,
            REVIEW_SCOPE,
        ]
    )
    review = result["semantic_review"]
    lines.extend(["", f"Review status: {review['status']}; sample size: {review['sample_size']}."])
    if review["status"] == "complete":
        lines.extend([f"Reviewer: {review['reviewer']}.", "", json.dumps(review["ratings"]), ""])
        for example in review["examples"]:
            lines.extend(
                [
                    f"### {example['id']}: {example['rating']}",
                    "",
                    "```json",
                    json.dumps(
                        {
                            key: example[key]
                            for key in (
                                "raw_output",
                                "reference_subject",
                                "diff_truncated",
                                "format_errors",
                            )
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    "```",
                    "",
                    example["rationale"],
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def load_predictions(directory: Path) -> tuple[dict, list[dict], dict]:
    """Read predictions only after verifying the saved generation report and local join."""
    report = validate_report(directory / "validation.json", check="validation")
    predictions = directory / PREDICTIONS_FILENAME
    table = pq.read_table(predictions)
    rows = table.to_pylist()
    if rows != join_predictions(report, directory):
        raise ValueError("Saved predictions differ from the validated report and local references.")
    metadata = json.loads(table.schema.metadata[b"qwen_commit"])
    return report, rows, metadata


def evaluate(directory: Path, review_path: Path | None = None) -> dict:
    report, rows, metadata = load_predictions(directory)
    digest = file_sha256(directory / PREDICTIONS_FILENAME)
    template = review_template(rows, digest)
    result = {
        "run_id": metadata["run_id"],
        "predictions_sha256": digest,
        "input_sha256": report["input_sha256"],
        "model_id": report["model_id"],
        "model_revision": report["model_revision"],
        "gpu": report["cuda_device_name"],
        "load_seconds": report["load_seconds"],
        "latency_scope": LATENCY_SCOPE,
        "first_generation_seconds": rows[0]["generation_seconds"],
        "cohorts": {
            "all_validation": metrics(rows),
            "approved_references": metrics([r for r in rows if r["reference_reviewed"] is True]),
        },
        "semantic_review": semantic_review(template, review_path),
    }
    template_path = directory / "semantic-review-template.json"
    if not template_path.exists():
        write_json(template_path, template)
    write_json(directory / "evaluation.json", result)
    (directory / "evaluation.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--review", type=Path, help="Completed semantic review JSON for this run.")
    args = parser.parse_args(argv)
    evaluate(args.run_directory, args.review)
    print(f"Evaluation saved to {args.run_directory / 'evaluation.md'}")


if __name__ == "__main__":
    main()
