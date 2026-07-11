"""Tests for read-only local Git history discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import helpers
from qwen_commit.cli import app
from qwen_commit.config import load_config
from qwen_commit.history import (
    HistoryConfig,
    HistoryScanError,
    RepositoryScanStatus,
    discover_repositories,
    load_history_config,
    normalise_remote_slug,
    scan_history,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfig:
    def test_resolves_relative_roots(self, tmp_path: Path) -> None:
        config_path = tmp_path / "qwen-commit.toml"
        config_path.write_text(
            """[history]
roots = ["repositories"]
ignore_repositories = ["archive/*"]
ignore_remotes = ["github.com/acme/*"]

[candidates]
bot_names = ["Copilot", "Codex"]
""",
            encoding="utf-8",
        )

        config = load_history_config(config_path)

        assert config.roots == ((tmp_path / "repositories").resolve(),)
        assert config.ignore_repositories == ("archive/*",)
        assert config.ignore_remotes == ("github.com/acme/*",)
        assert load_config(config_path).candidates.bot_names == ("Copilot", "Codex")

    def test_resolves_absolute_roots(self, tmp_path: Path) -> None:
        config_path = tmp_path / "qwen-commit.toml"
        absolute_root = tmp_path / "my-projects"
        config_path.write_text(
            f'[history]\nroots = ["{absolute_root.as_posix()}"]\n',
            encoding="utf-8",
        )

        config = load_history_config(config_path)

        assert config.roots == (absolute_root.resolve(),)

    def test_rejects_invalid_config_format(self, tmp_path: Path) -> None:
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

    def test_wraps_invalid_toml_syntax(self, tmp_path: Path) -> None:
        config_path = tmp_path / "qwen-commit.toml"
        config_path.write_text("[history\nroots = []\n", encoding="utf-8")

        with pytest.raises(HistoryScanError, match="Invalid TOML configuration"):
            load_history_config(config_path)

    def test_rejects_empty_and_blank_root_entries(self, tmp_path: Path) -> None:
        empty_roots = tmp_path / "empty-roots.toml"
        blank_roots = tmp_path / "blank-roots.toml"
        empty_roots.write_text("[history]\nroots = []\n", encoding="utf-8")
        blank_roots.write_text('[history]\nroots = [""]\n', encoding="utf-8")

        with pytest.raises(HistoryScanError):
            load_history_config(empty_roots)
        with pytest.raises(HistoryScanError, match="blank entries"):
            load_history_config(blank_roots)

    def test_deduplicates_identical_roots(self, tmp_path: Path) -> None:
        config_path = tmp_path / "qwen-commit.toml"
        root = tmp_path / "repositories"
        config_path.write_text(
            f'[history]\nroots = ["{root.as_posix()}", "{root.as_posix()}"]\n',
            encoding="utf-8",
        )

        config = load_history_config(config_path)

        assert config.roots == (root.resolve(),)


# ---------------------------------------------------------------------------
# History scan
# ---------------------------------------------------------------------------


class TestScan:
    def test_raises_for_missing_root(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"

        with pytest.raises(HistoryScanError, match="does not exist"):
            discover_repositories((missing,))

    def test_includes_repositories_and_counts_commits(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        first_repository = helpers.create_repository(
            root / "first", "person@example.com", commit_count=2
        )
        second_repository = helpers.create_repository(
            root / "second",
            "person@example.com",
            commit_author_email="person@example.com",
        )

        report = scan_history(HistoryConfig(roots=(root,)))

        scans = {scan.path: scan for scan in report.repositories}
        assert report.discovered_repository_count == 2
        assert report.included_repository_count == 2
        assert report.commit_count == 3
        assert report.author_email_count == 1
        assert scans[first_repository].commit_count == 2
        assert scans[first_repository].author_email_count == 1
        assert scans[second_repository].commit_count == 1
        assert scans[second_repository].author_email_count == 1

    def test_applies_path_and_remote_ignore_globs(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        included_repository = helpers.create_repository(root / "included", "person@example.com")
        path_ignored_repository = helpers.create_repository(
            root / "archive" / "old", "person@example.com"
        )
        remote_ignored_repository = helpers.create_repository(root / "remote", "person@example.com")
        helpers.git(
            remote_ignored_repository,
            "remote",
            "add",
            "origin",
            "git@github.com:acme/ignored.git",
        )

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
        scans = {scan.path: scan for scan in report.repositories}
        assert scans[path_ignored_repository].commit_count == 0
        assert scans[path_ignored_repository].author_email_count == 0
        assert scans[remote_ignored_repository].commit_count == 0
        assert scans[remote_ignored_repository].author_email_count == 0
        assert report.commit_count == 1

    def test_preserves_config_on_report(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        repository = helpers.create_repository(root / "personal", "person@example.com")
        config = HistoryConfig(roots=(root,))

        report = scan_history(config)

        assert report.repositories[0].path == repository
        assert report.repositories[0].commit_count == 1
        assert report.config == config

    def test_reports_shallow_clone(self, tmp_path: Path) -> None:
        source = helpers.create_repository(
            tmp_path / "source", "person@example.com", commit_count=2
        )
        clone = tmp_path / "clone"
        helpers.git(tmp_path, "clone", "--quiet", "--depth", "1", source.as_uri(), str(clone))

        report = scan_history(HistoryConfig(roots=(clone,)))

        assert report.shallow_repository_count == 1
        assert report.repositories[0].shallow is True

    def test_continues_when_remote_url_is_misconfigured(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        repository = helpers.create_repository(root / "broken-remote", "person@example.com")
        helpers.git(repository, "remote", "add", "origin", "https://github.com/example/repo.git")
        # Simulate a broken remote by removing its URL from the git config.
        helpers.git(repository, "config", "--unset", "remote.origin.url")

        report = scan_history(HistoryConfig(roots=(root,)))

        assert report.included_repository_count == 1
        assert report.repositories[0].status is RepositoryScanStatus.INCLUDED


# ---------------------------------------------------------------------------
# Report model and utilities
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_has_expected_json_format(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        helpers.create_repository(root / "personal", "person@example.com")

        report = scan_history(HistoryConfig(roots=(root,)))
        payload = json.loads(report.to_json())

        assert set(payload.keys()) == {
            "author_email_count",
            "commit_count",
            "discovered_repository_count",
            "ignored_repository_count",
            "included_repository_count",
            "repositories",
            "roots",
            "shallow_repository_count",
        }
        assert len(payload["repositories"]) == 1
        assert set(payload["repositories"][0].keys()) == {
            "author_email_count",
            "commit_count",
            "path",
            "shallow",
            "status",
        }
        assert "author_emails" not in payload["repositories"][0]

    def test_normalise_remote_slug(self) -> None:
        assert normalise_remote_slug("git@github.com:owner/repo.git") == "github.com/owner/repo"
        assert normalise_remote_slug("git@github.com:owner/repo") == "github.com/owner/repo"
        assert (
            normalise_remote_slug("git@bitbucket.org:team/project.git")
            == "bitbucket.org/team/project"
        )
        assert normalise_remote_slug("https://github.com/owner/repo.git") == "github.com/owner/repo"


# ---------------------------------------------------------------------------
# CLI — history scan command
# ---------------------------------------------------------------------------


class TestHistoryScanCLI:
    def test_writes_json_report_to_requested_path(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        helpers.create_repository(root / "personal", "person@example.com")
        config_path = tmp_path / "qwen-commit.toml"
        json_path = tmp_path / "scan.json"
        config_path.write_text(
            f'[history]\nroots = ["{root.as_posix()}"]\n',
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["history", "scan", "--config", str(config_path), "--json", str(json_path)],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["included_repository_count"] == 1
        assert payload["commit_count"] == 1

    def test_uses_default_json_report_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "repositories"
        helpers.create_repository(root / "personal", "person@example.com")
        config_path = tmp_path / "qwen-commit.toml"
        config_path.write_text(
            f'[history]\nroots = ["{root.as_posix()}"]\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["history", "scan", "--config", str(config_path)])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "history-scan.json").is_file()

    def test_reports_error_when_json_path_is_unwritable(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        helpers.create_repository(root / "personal", "person@example.com")
        config_path = tmp_path / "qwen-commit.toml"
        config_path.write_text(
            f'[history]\nroots = ["{root.as_posix()}"]\n',
            encoding="utf-8",
        )
        # Parent directory does not exist, so write_text will raise OSError.
        json_path = tmp_path / "nonexistent_dir" / "scan.json"

        result = runner.invoke(
            app,
            ["history", "scan", "--config", str(config_path), "--json", str(json_path)],
        )

        assert result.exit_code != 0
