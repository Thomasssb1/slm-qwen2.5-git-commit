"""Compatibility access to the history section of project configuration."""

from __future__ import annotations

from pathlib import Path

from qwen_commit.config import optional_string_list, required_string_list, resolve_config_path
from qwen_commit.history.models import HistoryConfig


def parse_history_config(
    section: dict[str, object], config_directory: Path, error_type: type[Exception]
) -> HistoryConfig:
    """Parse the `[history]` section of the project document."""
    roots = tuple(
        dict.fromkeys(
            resolve_config_path(value, config_directory)
            for value in required_string_list(section, "roots", "history.roots", error_type)
        )
    )
    return HistoryConfig(
        roots=roots,
        ignore_repositories=optional_string_list(
            section, "ignore_repositories", "history.ignore_repositories", error_type
        ),
        ignore_remotes=optional_string_list(
            section, "ignore_remotes", "history.ignore_remotes", error_type
        ),
    )


def load_history_config(path: Path) -> HistoryConfig:
    """Load and return only the history section of project configuration."""
    from qwen_commit.config import load_config

    return load_config(path).history
