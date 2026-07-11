"""Shared Git, matching, and normalization helpers."""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from qwen_commit.history.errors import HistoryScanError




def normalise_remote_slug(url: str) -> str:
    """Convert common Git remote URLs to a comparable host/path slug."""

    def slug(host: str, path: str) -> str:
        return f"{host.casefold()}/{path.strip('/').casefold()}"

    candidate = url.strip().rstrip("/")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]

    parsed = urlsplit(candidate)
    if parsed.hostname:
        return slug(parsed.hostname, parsed.path)

    host_part, separator, path = candidate.partition(":")
    if separator and path and "/" not in host_part:
        host = host_part.rsplit("@", maxsplit=1)[-1]
        if host:
            return slug(host, path)

    return candidate.casefold()


def matches_any(values: tuple[str, ...], patterns: tuple[str, ...]) -> bool:
    """Match case-insensitive glob patterns against candidate values."""
    return any(
        fnmatch.fnmatchcase(value.casefold(), pattern.casefold())
        for value in values
        for pattern in patterns
    )


def git_output(repository: Path, *arguments: str) -> str:
    """Run a read-only Git command in a repository and return stdout."""
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "Git command failed."
        raise HistoryScanError(f"{repository}: {message}")
    return completed.stdout.strip()
