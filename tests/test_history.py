"""Tests for read-only local Git history discovery."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qwen_commit.cli import app
from qwen_commit.history import (
    HistoryConfig,
    HistoryScanError,
    RepositoryScanStatus,
    load_history_config,
    scan_history,
)

runner = CliRunner()


def test_scan_includes_repositories_by_default_and_counts_commits(tmp_path: Path) -> None:
    root = tmp_path / "repositories"
    first_repository = _create_repository(root / "first", "person@example.com", commit_count=2)
    second_repository = _create_repository(
        root / "second",
        "person@example.com",
        commit_author_email="other@example.com",
    )

    report = scan_history(HistoryConfig(roots=(root,)))

    scans = {scan.path: scan for scan in report.repositories}
    assert report.discovered_repository_count == 2
    assert report.included_repository_count == 2
    assert report.commit_count == 3
    assert scans[first_repository].commit_count == 2
    assert scans[first_repository].author_email_count == 1
    assert scans[second_repository].commit_count == 1
    assert scans[second_repository].author_email_count == 1


def test_scan_applies_path_and_remote_ignore_globs(tmp_path: Path) -> None:
    root = tmp_path / "repositories"
    included_repository = _create_repository(root / "included", "person@example.com")
    path_ignored_repository = _create_repository(root / "archive" / "old", "person@example.com")
    remote_ignored_repository = _create_repository(root / "remote", "person@example.com")
    _git(remote_ignored_repository, "remote", "add", "origin", "git@github.com:acme/ignored.git")

    report = scan_history(
        HistoryConfig(
            roots=(root,),
            ignore_repositories=("archive/*",),
            ignore_remotes=("github.com/acme/ignored",),
        )
    )

    statuses = {scan.path: scan.status for scan in report.repositories}
    assert statuses[included_repository] is RepositoryScanStatus.INCLUDED
    assert statuses[path_ignored_repository] is RepositoryScanStatus.IGNORED_PATH
    assert statuses[remote_ignored_repository] is RepositoryScanStatus.IGNORED_REMOTE
    assert report.commit_count == 1


def test_scan_uses_repository_git_email_without_explicit_alias(tmp_path: Path) -> None:
    root = tmp_path / "repositories"
    repository = _create_repository(root / "personal", "person@example.com")
    config = HistoryConfig(roots=(root,))

    report = scan_history(config)

    assert report.repositories[0].path == repository
    assert report.repositories[0].commit_count == 1
    assert report.config == config


def test_load_config_resolves_relative_roots(tmp_path: Path) -> None:
    config_path = tmp_path / "qwen-commit.toml"
    config_path.write_text(
        """[history]
roots = ["repositories"]
ignore_repositories = ["archive/*"]
ignore_remotes = ["github.com/acme/*"]
""",
        encoding="utf-8",
    )

    config = load_history_config(config_path)

    assert config.roots == ((tmp_path / "repositories").resolve(),)
    assert config.ignore_repositories == ("archive/*",)
    assert config.ignore_remotes == ("github.com/acme/*",)


def test_load_config_requires_file_history_table_and_root(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.toml"
    without_history = tmp_path / "without-history.toml"
    without_roots = tmp_path / "without-roots.toml"
    without_history.write_text("[other]\nvalue = true\n", encoding="utf-8")
    without_roots.write_text("[history]\n", encoding="utf-8")

    with pytest.raises(HistoryScanError, match="does not exist"):
        load_history_config(missing_path)
    with pytest.raises(HistoryScanError, match=r"define a \[history\] table"):
        load_history_config(without_history)
    with pytest.raises(HistoryScanError, match=r"history\.roots"):
        load_history_config(without_roots)


def test_history_scan_cli_writes_requested_json_report(tmp_path: Path) -> None:
    root = tmp_path / "repositories"
    _create_repository(root / "personal", "person@example.com")
    config_path = tmp_path / "qwen-commit.toml"
    json_path = tmp_path / "scan.json"
    config_path.write_text(
        f'''[history]
roots = ["{root.as_posix()}"]
''',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "history",
            "scan",
            "--config",
            str(config_path),
            "--json",
            str(json_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["included_repository_count"] == 1
    assert payload["commit_count"] == 1


def test_history_scan_cli_uses_default_json_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repositories"
    _create_repository(root / "personal", "person@example.com")
    config_path = tmp_path / "qwen-commit.toml"
    config_path.write_text(
        f'''[history]
roots = ["{root.as_posix()}"]
''',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["history", "scan", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "history-scan.json").is_file()


def test_shallow_clone_is_reported(tmp_path: Path) -> None:
    source = _create_repository(tmp_path / "source", "person@example.com", commit_count=2)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "--quiet", "--depth", "1", source.as_uri(), str(clone))

    report = scan_history(HistoryConfig(roots=(clone,)))

    assert report.shallow_repository_count == 1
    assert report.repositories[0].shallow is True


def _create_repository(
    path: Path,
    author_email: str,
    *,
    commit_count: int = 1,
    commit_author_email: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(path.parent, "init", "--quiet", str(path))
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", author_email)
    for index in range(commit_count):
        (path / "history.txt").write_text(f"commit {index}\n", encoding="utf-8")
        _git(path, "add", "history.txt")
        if commit_author_email:
            _git(
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
            _git(path, "commit", "--quiet", "-m", f"Commit {index}")
    return path.resolve()


def _git(directory: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(directory), *arguments),
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
