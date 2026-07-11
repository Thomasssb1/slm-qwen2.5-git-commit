"""Parquet schemas and deterministic writers for private candidate data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from qwen_commit.candidates.errors import CandidateBuildError
from qwen_commit.candidates.models import Candidate, Provenance

_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("repository_group_id", pa.string(), nullable=False),
        pa.field("subject", pa.string(), nullable=False),
        pa.field("diff", pa.string(), nullable=False),
        pa.field("committed_at_utc", pa.string(), nullable=False),
        pa.field("patch_id", pa.string(), nullable=False),
    ]
)
_PROVENANCE_SCHEMA = pa.schema(
    [
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("repository_group_id", pa.string(), nullable=False),
        pa.field("repository_path", pa.string(), nullable=False),
        pa.field("remote_urls", pa.list_(pa.string()), nullable=False),
        pa.field("commit_sha", pa.string(), nullable=False),
        pa.field("author_name", pa.string(), nullable=False),
        pa.field("author_email", pa.string(), nullable=False),
    ]
)


def write_parquet(
    candidates: Iterable[Candidate],
    provenance: Iterable[Provenance],
    candidates_path: Path,
    provenance_path: Path,
) -> None:
    """Write matching candidate and provenance tables through temporary files."""
    _validate_paths(candidates_path, provenance_path)
    candidate_table = pa.Table.from_pylist(
        [candidate.__dict__ for candidate in candidates], schema=_CANDIDATE_SCHEMA
    )
    provenance_table = pa.Table.from_pylist(
        [entry.__dict__ for entry in provenance], schema=_PROVENANCE_SCHEMA
    )
    pq.write_table(candidate_table, candidates_path, compression="zstd")
    pq.write_table(provenance_table, provenance_path, compression="zstd")


def _validate_paths(candidates_path: Path, provenance_path: Path) -> None:
    if candidates_path == provenance_path:
        raise CandidateBuildError("Candidate and provenance outputs must have different paths.")
    for path in (candidates_path, provenance_path):
        if not path.parent.is_dir():
            raise CandidateBuildError(f"Output directory does not exist: {path.parent}")
