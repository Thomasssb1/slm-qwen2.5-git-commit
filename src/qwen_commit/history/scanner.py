"""Read-only local Git repository discovery and scanning."""

from __future__ import annotations

import os
from pathlib import Path

from qwen_commit.history.errors import HistoryScanError
from qwen_commit.history.models import (
    HistoryConfig,
    HistoryScanReport,
    RepositoryScan,
    RepositoryScanStatus,
)
from qwen_commit.history.utils import (
    git_output,
    matches_any,
    normalise_remote_slug,
)


def scan_history(config: HistoryConfig) -> HistoryScanReport:
    """Discover repositories and count commits without storing Git content."""
    if not config.roots:
        raise HistoryScanError("Configure at least one history.roots entry.")

    repositories = discover_repositories(config.roots)
    scans: list[RepositoryScan] = []

    for repository in repositories:
        path_candidates = _repository_path_candidates(repository, config.roots)
        remote_slugs = _remote_slugs(repository)

        if matches_any(path_candidates, config.ignore_repositories):
            scans.append(
                RepositoryScan(repository, RepositoryScanStatus.IGNORED_PATH, 0, frozenset(), False)
            )
            continue
        if matches_any(remote_slugs, config.ignore_remotes):
            scans.append(
                RepositoryScan(
                    repository, RepositoryScanStatus.IGNORED_REMOTE, 0, frozenset(), False
                )
            )
            continue

        repository_author_emails = _unique_author_emails(repository)
        scans.append(
            RepositoryScan(
                path=repository,
                status=RepositoryScanStatus.INCLUDED,
                commit_count=_commit_count(repository),
                author_emails=repository_author_emails,
                shallow=_is_shallow_repository(repository),
            )
        )

    return HistoryScanReport(
        config=config,
        repositories=tuple(scans),
    )


def discover_repositories(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    """Find work-tree repositories beneath configured roots without modifying them."""
    repositories: set[Path] = set()
    for configured_root in roots:
        root = configured_root.expanduser().resolve()
        if not root.is_dir():
            raise HistoryScanError(f"Configured history root does not exist: {root}")

        for current, directories, files in os.walk(root):
            if ".git" not in directories and ".git" not in files:
                continue
            directories[:] = [directory for directory in directories if directory != ".git"]
            repository = Path(current).resolve()
            if _is_work_tree(repository):
                repositories.add(repository)

    return tuple(sorted(repositories, key=lambda repository: repository.as_posix().casefold()))


def _repository_path_candidates(repository: Path, roots: tuple[Path, ...]) -> tuple[str, ...]:
    candidates = [repository.name, repository.as_posix()]
    for root in roots:
        try:
            candidates.append(repository.relative_to(root.expanduser().resolve()).as_posix())
        except ValueError:
            continue
    return tuple(candidates)


def _remote_slugs(repository: Path) -> tuple[str, ...]:
    remotes = git_output(repository, "remote").splitlines()
    slugs: set[str] = set()
    for remote in remotes:
        slugs.update(
            normalise_remote_slug(url) for url in _repository_remote_urls(repository, remote)
        )
    return tuple(sorted(slugs))


def _repository_remote_urls(repository: Path, remote: str) -> list[str]:
    try:
        return git_output(repository, "remote", "get-url", "--all", remote).splitlines()
    except HistoryScanError:
        return []


def _commit_count(repository: Path) -> int:
    """Return the total number of commits reachable from any ref."""
    output = git_output(repository, "rev-list", "--all", "--count")
    return int(output) if output else 0


def _unique_author_emails(repository: Path) -> frozenset[str]:
    return frozenset(
        email.strip().casefold()
        for email in git_output(repository, "log", "--all", "--format=%ae").splitlines()
        if email.strip()
    )


def _is_shallow_repository(repository: Path) -> bool:
    try:
        return git_output(repository, "rev-parse", "--is-shallow-repository") == "true"
    except HistoryScanError:
        return False


def _is_work_tree(repository: Path) -> bool:
    try:
        return git_output(repository, "rev-parse", "--is-inside-work-tree") == "true"
    except HistoryScanError:
        return False
