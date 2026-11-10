"""Summarize formatting and latency for a saved validation run."""

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

LATENCY_SCOPE = (
    "CUDA-synchronized model.generate time, including the first call. Excludes tokenization, "
    "model loading, provisioning, and transfer. Median/p95 use linear interpolation."
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


def evaluate(directory: Path) -> dict:
    report, rows, metadata = load_predictions(directory)
    digest = file_sha256(directory / PREDICTIONS_FILENAME)
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
    }
    write_json(directory / "evaluation.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    evaluate(args.run_directory)
    print(f"Evaluation saved to {args.run_directory / 'evaluation.json'}")


if __name__ == "__main__":
    main()
