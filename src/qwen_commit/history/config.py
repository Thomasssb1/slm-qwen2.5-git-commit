"""Compatibility access to the history section of project configuration."""

from __future__ import annotations

from pathlib import Path

from qwen_commit.config import load_config
from qwen_commit.history.models import HistoryConfig


def load_history_config(path: Path) -> HistoryConfig:
    """Load and return only the history section of project configuration."""
    return load_config(path).history
