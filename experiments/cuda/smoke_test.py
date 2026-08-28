"""Run a one-batch CUDA smoke test without downloading or loading a model."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as pq
import torch


def main() -> None:
    dataset_path = Path("/datasets/train.parquet")
    artifact_path = Path("/artifacts/cuda-smoke.json")
    batch_rows = _read_one_batch(dataset_path)
    diagnostic = _cuda_diagnostic()
    diagnostic.update(
        {
            "dataset_path": str(dataset_path),
            "one_batch_rows": batch_rows,
            "pyarrow_version": version("pyarrow"),
            "ran_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    artifact_path.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostic, sort_keys=True))


def _read_one_batch(dataset_path: Path) -> int:
    if not dataset_path.is_file():
        raise RuntimeError(f"Prepared training Parquet is missing: {dataset_path}")
    parquet = pq.ParquetFile(dataset_path)
    try:
        batch = next(parquet.iter_batches(batch_size=1))
    except StopIteration as error:
        raise RuntimeError(f"Prepared training Parquet contains no rows: {dataset_path}") from error
    return batch.num_rows


def _cuda_diagnostic() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this image on a NVIDIA GPU host with --gpus all."
        )
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    values = torch.arange(1024, device=device, dtype=torch.float32)
    checksum = torch.dot(values, values).item()
    torch.cuda.synchronize(device)
    return {
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": properties.name,
        "cuda_device_total_memory_bytes": properties.total_memory,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_tensor_checksum": checksum,
        "torch_version": torch.__version__,
    }


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"CUDA smoke test failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
