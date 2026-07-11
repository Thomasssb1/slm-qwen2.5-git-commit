"""Project TOML configuration and section-specific settings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen_commit.candidates.config import CandidateConfig
    from qwen_commit.history.models import HistoryConfig


@dataclass(frozen=True)
class Config:
    """Complete qwen-commit configuration grouped by feature section."""

    history: HistoryConfig
    candidates: CandidateConfig


def load_config(path: Path) -> Config:
    """Load and validate the complete TOML configuration."""
    from qwen_commit.candidates.config import parse_candidate_config
    from qwen_commit.history.config import parse_history_config
    from qwen_commit.history.errors import HistoryScanError

    document = _load_document(path, HistoryScanError)
    history = document.get("history")
    if not isinstance(history, dict):
        raise HistoryScanError("The configuration file must define a [history] table.")

    candidates_section = document.get("candidates", {})
    if not isinstance(candidates_section, dict):
        raise HistoryScanError("The [candidates] configuration section must be a table.")

    return Config(
        history=parse_history_config(history, path.parent, HistoryScanError),
        candidates=parse_candidate_config(candidates_section, HistoryScanError),
    )


def _load_document(path: Path, error_type: type[Exception]) -> dict[str, object]:
    if not path.exists():
        raise error_type(f"Configuration file does not exist: {path}")
    if not path.is_file():
        raise error_type(f"Configuration is not a file: {path}")

    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise error_type(f"Invalid TOML configuration in {path}: {error}") from error
    return document


def required_string_list(
    section: dict[str, object], key: str, setting_name: str, error_type: type[Exception]
) -> tuple[str, ...]:
    """Read a required array of non-blank strings from a TOML section."""
    values = optional_string_list(section, key, setting_name, error_type)
    if not values:
        raise error_type(f"{setting_name} is required.")
    return values


def optional_string_list(
    section: dict[str, object], key: str, setting_name: str, error_type: type[Exception]
) -> tuple[str, ...]:
    """Read an optional array of non-blank strings from a TOML section."""
    if key not in section:
        return ()
    value = section[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise error_type(f"{setting_name} must be an array of strings.")
    if any(not item.strip() for item in value):
        raise error_type(f"{setting_name} must not contain blank entries.")
    return tuple(value)


def resolve_config_path(value: str, config_directory: Path) -> Path:
    """Resolve a path value relative to the configuration file directory."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    return candidate.resolve()
