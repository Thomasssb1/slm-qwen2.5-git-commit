"""Candidate-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for candidate extraction."""

    author_emails: tuple[str, ...] = ()


def parse_candidate_config(
    section: dict[str, object], error_type: type[Exception]
) -> CandidateConfig:
    """Parse the `[candidates]` section of the project document."""
    from qwen_commit.config import optional_string_list

    return CandidateConfig(
        author_emails=optional_string_list(
            section, "author_emails", "candidates.author_emails", error_type
        )
    )


def load_candidate_config(path: Path) -> CandidateConfig:
    """Load and return only the candidate section of project configuration."""
    from qwen_commit.config import load_config

    return load_config(path).candidates
