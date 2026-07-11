"""Candidate-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for candidate extraction."""

    bot_names: tuple[str, ...] = ()


def parse_candidate_config(
    section: dict[str, object], error_type: type[Exception]
) -> CandidateConfig:
    """Parse the `[candidates]` section of the project document."""
    from qwen_commit.config import optional_string_list

    return CandidateConfig(
        bot_names=optional_string_list(section, "bot_names", "candidates.bot_names", error_type)
    )


def load_candidate_config(path: Path) -> CandidateConfig:
    """Load and return only the candidate section of project configuration."""
    from qwen_commit.config import load_config

    return load_config(path).candidates
