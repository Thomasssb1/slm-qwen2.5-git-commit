"""Data structures for private commit-message candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from qwen_commit.history.models import HistoryScanReport


@dataclass(frozen=True)
class Candidate:
    """One training-safe commit-message example."""

    example_id: str
    repository_group_id: str
    subject: str
    diff: str
    committed_at_utc: str
    patch_id: str


@dataclass(frozen=True)
class Provenance:
    """Private source mapping for one accepted candidate."""

    example_id: str
    repository_group_id: str
    repository_path: str
    remote_urls: tuple[str, ...]
    commit_sha: str
    author_name: str
    author_email: str


@dataclass(frozen=True)
class CandidateBuildReport:
    """Summary of one deterministic candidate-build run."""

    scan_report: HistoryScanReport
    candidates_path: Path
    provenance_path: Path
    accepted_count: int
    rejection_counts: Counter[str] = field(default_factory=Counter)

    @property
    def rejected_count(self) -> int:
        return sum(self.rejection_counts.values())

    @property
    def inspected_count(self) -> int:
        return self.accepted_count + self.rejected_count
