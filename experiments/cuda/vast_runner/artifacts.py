"""Training samples, smoke reports, and local run artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from generation import (
    GENERATION_SETTINGS,
    MAX_DIFF_TOKENS,
    MAX_NEW_TOKENS,
    MAX_SUBJECT_CHARACTERS,
    SYSTEM_PROMPT,
    output_errors,
)
from model_spec import MODEL_ID, MODEL_REVISION, TRANSFORMERS_VERSION

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


def validate_report(path: Path, *, check: str = "cuda") -> dict:
    """Require a real, nonempty CUDA result instead of just a zero exit status."""
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_checksum = sum(number * number for number in range(1024))
    try:
        valid = (
            report["cuda_device_count"] >= 1
            and bool(report["cuda_device_name"])
            and report["cuda_device_total_memory_bytes"] > 0
            and str(report["cuda_runtime_version"]).startswith("13.2")
            and str(report["torch_version"]).split("+")[0] == "2.12.1"
            and math.isclose(float(report["cuda_tensor_checksum"]), expected_checksum, rel_tol=1e-6)
        )
        if check in {"model", "inference"}:
            valid = (
                valid
                and report["model_id"] == MODEL_ID
                and report["model_revision"] == MODEL_REVISION
                and report["model_class"] == "Qwen2ForCausalLM"
                and report["parameter_count"] > 0
                and report["cuda_parameter_count"] == report["parameter_count"]
                and report["parameter_dtype"] == "bfloat16"
                and report["tokenizer_vocab_size"] > 0
                and report["tokenizer_probe_tokens"] > 0
                and report["transformers_version"] == TRANSFORMERS_VERSION
                and 0 <= report["load_seconds"] < float("inf")
                and 0
                < report["cuda_memory_allocated_bytes"]
                <= report["cuda_peak_memory_allocated_bytes"]
            )
            if check == "inference":
                valid = valid and _valid_inference(report)
        else:
            valid = valid and check == "cuda" and report["one_batch_rows"] == 1
    except (AttributeError, KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RunError(f"The downloaded report does not prove the pinned {check} check passed.")
    return report


def _valid_inference(report: dict) -> bool:
    """A smoke check proves generation ran; invalid model responses remain measurable."""
    import hashlib

    examples = report["examples"]
    if set(examples) != {"single_file", "multiple_files", "truncated"}:
        return False
    for result in examples.values():
        config = result["generation_config"]
        errors = output_errors(
            result["raw_output"], hit_token_limit=result["finish_reason"] == "token_limit"
        )
        if not (
            result["system_prompt"] == SYSTEM_PROMPT
            and result["system_prompt_sha256"] == hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
            and result["max_diff_tokens"] == MAX_DIFF_TOKENS
            and result["max_subject_characters"] == MAX_SUBJECT_CHARACTERS
            and result["truncation_side"] == "right"
            and all(config[key] == value for key, value in GENERATION_SETTINGS.items())
            and 0 < result["diff_tokens_used"] == min(result["diff_tokens"], MAX_DIFF_TOKENS)
            and result["diff_truncated"] == (result["diff_tokens"] > MAX_DIFF_TOKENS)
            and result["prompt_tokens"] > result["diff_tokens_used"]
            and 0
            < result["generated_tokens"]
            == len(result["generated_token_ids"])
            <= MAX_NEW_TOKENS
            and result["finish_reason"] in {"eos", "token_limit"}
            and result["errors"] == errors
            and result["valid"] == (not errors)
            and result["subject"] == (None if errors else result["raw_output"].strip(" \t"))
            and 0 <= result["generation_seconds"] < float("inf")
        ):
            return False
    return (
        examples["truncated"]["diff_truncated"] is True
        and report["repeat_matches"] is True
        and report["repeat_token_ids"] == examples["single_file"]["generated_token_ids"]
    )
