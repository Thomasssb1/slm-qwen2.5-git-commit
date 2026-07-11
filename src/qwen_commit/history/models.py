"""History-discovery data structures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


@dataclass(frozen=True)
class HistoryConfig:
    """Configuration for discovering and filtering local Git repositories."""

    roots: tuple[Path, ...] = ()
    ignore_repositories: tuple[str, ...] = ()
    ignore_remotes: tuple[str, ...] = ()


class RepositoryScanStatus(StrEnum):
    """Repository state after applying history scan filters."""

    INCLUDED = "included"
    IGNORED_PATH = "ignored_path"
    IGNORED_REMOTE = "ignored_remote"


@dataclass(frozen=True)
class RepositoryScan:
    """Read-only scan result for one repository."""

    path: Path
    status: RepositoryScanStatus
    commit_count: int | None
    author_email_count: int | None
    shallow: bool


@dataclass(frozen=True)
class HistoryScanReport:
    """Summary of a local history discovery run."""

    config: HistoryConfig
    repositories: tuple[RepositoryScan, ...]

    @property
    def discovered_repository_count(self) -> int:
        return len(self.repositories)

    @property
    def included_repository_count(self) -> int:
        return sum(
            repository.status is RepositoryScanStatus.INCLUDED for repository in self.repositories
        )

    @property
    def ignored_repository_count(self) -> int:
        return self.discovered_repository_count - self.included_repository_count

    @property
    def shallow_repository_count(self) -> int:
        return sum(repository.shallow for repository in self.repositories)

    @property
    def commit_count(self) -> int:
        return sum(
            repository.commit_count or 0
            for repository in self.repositories
            if repository.status is RepositoryScanStatus.INCLUDED
        )

    @property
    def author_email_count(self) -> int:
        return sum(
            repository.author_email_count or 0
            for repository in self.repositories
            if repository.status is RepositoryScanStatus.INCLUDED
        )

    def to_json(self) -> str:
        """Render the report as stable, machine-readable JSON."""
        payload = {
            "author_email_count": self.author_email_count,
            "commit_count": self.commit_count,
            "discovered_repository_count": self.discovered_repository_count,
            "ignored_repository_count": self.ignored_repository_count,
            "included_repository_count": self.included_repository_count,
            "repositories": [
                {
                    "author_email_count": repository.author_email_count,
                    "commit_count": repository.commit_count,
                    "path": str(repository.path),
                    "shallow": repository.shallow,
                    "status": repository.status.value,
                }
                for repository in self.repositories
            ],
            "roots": [str(root) for root in self.config.roots],
            "shallow_repository_count": self.shallow_repository_count,
        }
        return json.dumps(payload, indent=2, sort_keys=True)
