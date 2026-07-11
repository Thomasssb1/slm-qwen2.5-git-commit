"""Project TOML configuration and section-specific settings."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen_commit.history.models import HistoryConfig


@dataclass(frozen=True)
class CandidateConfig:
    """Configuration for candidate extraction."""

    bot_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    """Complete qwen-commit configuration grouped by feature section."""

    history: HistoryConfig
    candidates: CandidateConfig


def load_config(path: Path) -> Config:
    """Load and validate the complete TOML configuration."""
    from qwen_commit.history.errors import HistoryScanError
    from qwen_commit.history.models import HistoryConfig

    document = _load_document(path, HistoryScanError)
    history = document.get("history")
    if not isinstance(history, dict):
        raise HistoryScanError("The configuration file must define a [history] table.")

    candidates = document.get("candidates", {})
    if not isinstance(candidates, dict):
        raise HistoryScanError("The [candidates] configuration section must be a table.")

    roots = tuple(
        dict.fromkeys(
            _resolve_config_path(value, path.parent)
            for value in _required_string_list(history, "roots", "history.roots", HistoryScanError)
        )
    )
    if not roots:
        raise HistoryScanError("history.roots must contain at least one path.")

    return Config(
        history=HistoryConfig(
            roots=roots,
            ignore_repositories=_optional_string_list(
                history, "ignore_repositories", "history.ignore_repositories", HistoryScanError
            ),
            ignore_remotes=_optional_string_list(
                history, "ignore_remotes", "history.ignore_remotes", HistoryScanError
            ),
        ),
        candidates=CandidateConfig(
            bot_names=_optional_string_list(
                candidates, "bot_names", "candidates.bot_names", HistoryScanError
            )
        ),
    )


def _load_document(path: Path, error_type: type[Exception]) -> dict[str, object]:
    if not path.exists():
        raise error_type(f"History configuration file does not exist: {path}")
    if not path.is_file():
        raise error_type(f"History configuration is not a file: {path}")

    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise error_type(f"Invalid TOML configuration in {path}: {error}") from error
    return document


def _required_string_list(
    section: dict[str, object], key: str, setting_name: str, error_type: type[Exception]
) -> tuple[str, ...]:
    values = _optional_string_list(section, key, setting_name, error_type)
    if not values:
        raise error_type(f"{setting_name} is required.")
    return values


def _optional_string_list(
    section: dict[str, object], key: str, setting_name: str, error_type: type[Exception]
) -> tuple[str, ...]:
    if key not in section:
        return ()
    value = section[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise error_type(f"{setting_name} must be an array of strings.")
    if any(not item.strip() for item in value):
        raise error_type(f"{setting_name} must not contain blank entries.")
    return tuple(value)


def _resolve_config_path(value: str, config_directory: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    return candidate.resolve()
