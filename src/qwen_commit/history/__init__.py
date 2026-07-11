"""Read-only discovery of local Git histories."""

from qwen_commit.history.config import load_history_config, parse_history_config
from qwen_commit.history.errors import HistoryScanError
from qwen_commit.history.models import (
    HistoryConfig,
    HistoryScanReport,
    RepositoryScan,
    RepositoryScanStatus,
)
from qwen_commit.history.scanner import discover_repositories, scan_history
from qwen_commit.history.utils import normalise_remote_slug

__all__ = [
    "HistoryConfig",
    "HistoryScanError",
    "HistoryScanReport",
    "RepositoryScan",
    "RepositoryScanStatus",
    "discover_repositories",
    "load_history_config",
    "normalise_remote_slug",
    "parse_history_config",
    "scan_history",
]
