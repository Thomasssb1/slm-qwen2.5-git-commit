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


def commit_file(
    repository: Path,
    relative_path: str,
    contents: str | bytes,
    subject: str,
    *,
    author_name: str | None = None,
    author_email: str | None = None,
) -> None:
    """Write and commit one fixture file, optionally with a distinct author identity."""
    if bool(author_name) != bool(author_email):
        raise ValueError("Both author_name and author_email must be provided together, or neither.")
    path = repository / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    else:
        path.write_text(contents, encoding="utf-8")
    git(repository, "add", relative_path)
    arguments = ["commit", "--quiet", "-m", subject]
    if author_name and author_email:
        arguments = [
            "-c",
            f"user.name={author_name}",
            "-c",
            f"user.email={author_email}",
            *arguments,
        ]
    git(repository, *arguments)
