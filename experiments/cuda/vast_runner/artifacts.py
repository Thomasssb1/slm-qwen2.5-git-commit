"""Training samples, smoke reports, and local run artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import RunError

CUDA_DIRECTORY = Path(__file__).resolve().parents[1]
PROJECT_DIRECTORY = CUDA_DIRECTORY.parents[1]
RUNS_DIRECTORY = CUDA_DIRECTORY / "artifacts" / "vast"


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
