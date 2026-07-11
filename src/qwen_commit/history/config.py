"""TOML configuration loading for history discovery."""

from __future__ import annotations

import tomllib
from pathlib import Path

from qwen_commit.history.errors import HistoryScanError
from qwen_commit.history.models import HistoryConfig


def load_history_config(path: Path) -> HistoryConfig:
    """Load and validate a local TOML configuration file."""
    if not path.exists():
        raise HistoryScanError(f"History configuration file does not exist: {path}")
    if not path.is_file():
        raise HistoryScanError(f"History configuration is not a file: {path}")

    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as error:
        raise HistoryScanError(f"Invalid TOML configuration in {path}: {error}") from error

    if "history" not in document:
        raise HistoryScanError("The configuration file must define a [history] table.")

    history = document["history"]
    if not isinstance(history, dict):
        raise HistoryScanError("The [history] configuration section must be a table.")

    roots = tuple(
        dict.fromkeys(
            _resolve_config_path(value, path.parent)
            for value in _required_string_list(history, "roots", "history.roots")
        )
    )
    if not roots:
        raise HistoryScanError("history.roots must contain at least one path.")

    return HistoryConfig(
        roots=roots,
        ignore_repositories=_optional_string_list(
            history, "ignore_repositories", "history.ignore_repositories"
        ),
        ignore_remotes=_optional_string_list(history, "ignore_remotes", "history.ignore_remotes"),
    )


def _required_string_list(
    history: dict[str, object], key: str, setting_name: str
) -> tuple[str, ...]:
    values = _optional_string_list(history, key, setting_name)
    if not values:
        raise HistoryScanError(f"{setting_name} is required.")
    return values


def _optional_string_list(
    history: dict[str, object], key: str, setting_name: str
) -> tuple[str, ...]:
    if key not in history:
        return ()
    return _string_list(history[key], setting_name)


def _string_list(value: object, setting_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HistoryScanError(f"{setting_name} must be an array of strings.")
    for item in value:
        if not item.strip():
            raise HistoryScanError(f"{setting_name} must not contain blank entries.")
    return tuple(value)


def _resolve_config_path(value: str, config_directory: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    return candidate.resolve()
