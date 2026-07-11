"""Shared test helpers for Git repository setup."""

from __future__ import annotations

import subprocess
from pathlib import Path


def create_repository(
    path: Path,
    author_email: str,
    *,
    commit_count: int = 1,
    commit_author_email: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    git(path.parent, "init", "--quiet", str(path))
    git(path, "config", "user.name", "Test User")
    git(path, "config", "user.email", author_email)
    for index in range(commit_count):
        (path / "history.txt").write_text(f"commit {index}\n", encoding="utf-8")
        git(path, "add", "history.txt")
        if commit_author_email:
            git(
                path,
                "-c",
                "user.name=Other User",
                "-c",
                f"user.email={commit_author_email}",
                "commit",
                "--quiet",
                "-m",
                f"Commit {index}",
            )
        else:
            git(path, "commit", "--quiet", "-m", f"Commit {index}")
    return path.resolve()


def git(directory: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(directory), *arguments),
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
