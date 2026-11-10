"""Validation inputs and local reference joins; no model dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_FILENAME = "validation-input.parquet"
REFERENCES_FILENAME = "validation-references.parquet"
PREDICTIONS_FILENAME = "validation-predictions.parquet"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path: Path, columns: list[str], *, exact: bool = False) -> list[dict]:
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    if (exact and names != columns) or any(name not in names for name in columns):
        raise ValueError(f"Validation Parquet must contain {'only ' if exact else ''}{columns}.")
    rows = parquet.read(columns=columns).to_pylist()
    if not rows:
        raise ValueError("Validation Parquet contains no rows.")
    for row in rows:
        if any(not isinstance(row[name], str) or not row[name].strip() for name in columns):
            raise ValueError("Validation rows must contain nonempty strings.")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Validation IDs must be unique.")
    return rows


def prepare_validation(source: Path, directory: Path, reviewed: Path | None = None) -> dict:
    """Snapshot all inputs while retaining reference subjects only on the local host."""
    rows = read_rows(source, ["id", "subject", "diff"])
    approved = read_rows(reviewed, ["id", "subject", "diff"]) if reviewed is not None else []
    by_id = {row["id"]: row for row in rows}
    if any(by_id.get(row["id"]) != row for row in approved):
        raise ValueError("Reviewed references do not match the selected validation split.")
    approved_ids = {row["id"] for row in approved}
    inputs = directory / INPUT_FILENAME
    # Build fresh tables so source schema/field metadata cannot cross the transfer boundary.
    pq.write_table(
        pa.Table.from_pylist([{"id": row["id"], "diff": row["diff"]} for row in rows]),
        inputs,
        compression="zstd",
    )
    references = pa.table(
        {
            "id": [row["id"] for row in rows],
            "reference_subject": [row["subject"] for row in rows],
            "reference_reviewed": pa.array(
                [row["id"] in approved_ids if reviewed is not None else None for row in rows],
                type=pa.bool_(),
            ),
        }
    )
    pq.write_table(references, directory / REFERENCES_FILENAME, compression="zstd")
    return {
        "split": "validation",
        "source_name": source.name,
        "source_sha256": file_sha256(source),
        "rows": len(rows),
        "input_sha256": file_sha256(inputs),
        "reviewed_source_name": reviewed.name if reviewed is not None else None,
        "reviewed_sha256": file_sha256(reviewed) if reviewed is not None else None,
        "reviewed_rows": len(approved),
    }


def save_predictions(report: dict, directory: Path, *, run_id: str) -> Path:
    """Join validated predictions to the exact local inputs and references, in source order."""
    inputs = directory / INPUT_FILENAME
    rows = read_rows(inputs, ["id", "diff"], exact=True)
    references = pq.read_table(directory / REFERENCES_FILENAME).to_pylist()
    predictions = report["predictions"]
    ids = [row["id"] for row in rows]
    if (
        report["input_sha256"] != file_sha256(inputs)
        or report["row_count"] != len(rows)
        or [row["id"] for row in predictions] != ids
        or [row["id"] for row in references] != ids
    ):
        raise ValueError("Validation predictions do not match the complete uploaded split.")
    records = []
    for row, reference, prediction in zip(rows, references, predictions, strict=True):
        record = prediction.copy()
        # The complete configuration stays in validation.json and Parquet metadata.
        record.pop("generation_config")
        record["predicted_subject"] = record.pop("subject")
        record.update(reference, diff=row["diff"])
        records.append(record)
    table = pa.Table.from_pylist(records)
    for name, data_type in (
        ("predicted_subject", pa.string()),
        ("reference_reviewed", pa.bool_()),
        ("errors", pa.list_(pa.string())),
    ):
        table = table.set_column(
            table.schema.get_field_index(name),
            name,
            pa.array([row[name] for row in records], type=data_type),
        )
    metadata = {key: value for key, value in report.items() if key != "predictions"}
    metadata.update(run_id=run_id, generation_config=predictions[0]["generation_config"])
    table = table.replace_schema_metadata({b"qwen_commit": json.dumps(metadata).encode()})
    destination = directory / PREDICTIONS_FILENAME
    temporary = destination.with_suffix(".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(destination)
    return destination
