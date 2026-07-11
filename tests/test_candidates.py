"""Tests for private Parquet candidate extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

import helpers
from qwen_commit.candidates import CandidateBuildError, CandidateRejectionReason, build_candidates
from qwen_commit.candidates.filters import is_bot
from qwen_commit.cli import app
from qwen_commit.history import HistoryConfig, scan_history

runner = CliRunner()


class TestCandidateFilters:
    @pytest.mark.parametrize(
        "author_name",
        ["Copilot", "Codex", "Claude", "Dependabot", "github-actions"],
    )
    def test_recognizes_known_automation_names(self, author_name: str) -> None:
        assert is_bot(author_name, ("claude", "codex", "copilot", "dependabot", "github-actions"))

    def test_uses_configured_bot_names(self) -> None:
        assert is_bot("Review Automation", ("Review Automation",))
        assert not is_bot("Codex", ("Review Automation",))


class TestCandidateBuild:
    @staticmethod
    def _build(repository_root: Path, output_root: Path, bot_names: tuple[str, ...] = ()):
        output_root.mkdir(exist_ok=True)
        return build_candidates(
            scan_history(HistoryConfig(roots=(repository_root,))),
            output_root / "candidates.parquet",
            output_root / "provenance.parquet",
            bot_names,
        )

    def test_writes_candidate_and_provenance(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        repository = helpers.create_repository(root / "personal", "person@example.com")
        helpers.git(
            repository,
            "remote",
            "add",
            "origin",
            "git@github.com:example/private-history.git",
        )

        report = self._build(root, tmp_path)
        candidates = pq.read_table(report.candidates_path).to_pylist()
        provenance = pq.read_table(report.provenance_path).to_pylist()

        assert report.accepted_count == 1
        assert report.rejected_count == 0
        assert report.scan_report.included_repository_count == 1
        assert report.scan_report.commit_count == 1
        assert set(candidates[0]) == {
            "example_id",
            "repository_group_id",
            "subject",
            "diff",
            "committed_at_utc",
            "patch_id",
        }
        assert candidates[0]["subject"] == "Commit 0"
        assert candidates[0]["diff"].startswith("diff --git")
        candidate_text = json.dumps(candidates)
        assert str(repository) not in candidate_text
        assert "person@example.com" not in candidate_text
        assert set(provenance[0]) == {
            "example_id",
            "repository_group_id",
            "repository_path",
            "remote_urls",
            "commit_sha",
            "author_name",
            "author_email",
        }
        assert provenance[0]["repository_path"] == str(repository)
        assert provenance[0]["author_email"] == "person@example.com"
        assert provenance[0]["commit_sha"] not in candidate_text
        assert "git@github.com:example/private-history.git" not in candidate_text
        assert provenance[0]["remote_urls"] == ["git@github.com:example/private-history.git"]

    def test_rejects_unsafe_or_unusable_commits(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        repository = helpers.create_repository(root / "personal", "person@example.com")
        helpers.commit_file(repository, "fixup.txt", "x\n", "fixup! Commit 0")
        helpers.commit_file(
            repository,
            "bot.txt",
            "x\n",
            "Automated update",
            author_name="automation[bot]",
            author_email="bot@example.com",
        )
        helpers.commit_file(repository, "dist/js/app.js", "var x=1;\n", "Build bundle")
        helpers.commit_file(repository, "image.bin", b"\x00\x01\x02", "Add image")
        helpers.git(repository, "commit", "--quiet", "--allow-empty", "-m", "Empty change")

        helpers.git(repository, "checkout", "--quiet", "-b", "feature")
        helpers.commit_file(repository, "feature.txt", "feature\n", "Add feature")
        helpers.git(repository, "checkout", "--quiet", "-")
        helpers.commit_file(repository, "main.txt", "main\n", "Add main")
        helpers.git(repository, "merge", "--quiet", "--no-ff", "feature", "-m", "Merge feature")

        report = self._build(root, tmp_path, ("automation[bot]",))

        assert report.accepted_count == 3
        assert report.rejection_counts == {
            CandidateRejectionReason.BINARY: 1,
            CandidateRejectionReason.BOT: 1,
            CandidateRejectionReason.EMPTY_CHANGE: 1,
            CandidateRejectionReason.FIXUP: 1,
            CandidateRejectionReason.GENERATED_ONLY: 1,
            CandidateRejectionReason.MERGE: 1,
        }

    def test_output_is_deterministic(self, tmp_path: Path) -> None:
        root = tmp_path / "repositories"
        helpers.create_repository(root / "personal", "person@example.com", commit_count=2)

        self._build(root, tmp_path / "first")
        self._build(root, tmp_path / "second")

        first_candidates = pq.read_table(tmp_path / "first" / "candidates.parquet").to_pylist()
        second_candidates = pq.read_table(tmp_path / "second" / "candidates.parquet").to_pylist()
        first_provenance = pq.read_table(tmp_path / "first" / "provenance.parquet").to_pylist()
        second_provenance = pq.read_table(tmp_path / "second" / "provenance.parquet").to_pylist()
        assert first_candidates == second_candidates
        assert first_provenance == second_provenance
        assert (tmp_path / "first" / "candidates.parquet").read_bytes() == (
            tmp_path / "second" / "candidates.parquet"
        ).read_bytes()
        assert (tmp_path / "first" / "provenance.parquet").read_bytes() == (
            tmp_path / "second" / "provenance.parquet"
        ).read_bytes()

    def test_rejects_shallow_repository(self, tmp_path: Path) -> None:
        source = helpers.create_repository(
            tmp_path / "source", "person@example.com", commit_count=2
        )
        root = tmp_path / "repositories"
        root.mkdir()
        clone = root / "shallow"
        helpers.git(tmp_path, "clone", "--quiet", "--depth", "1", source.as_uri(), str(clone))

        with pytest.raises(CandidateBuildError, match="shallow repository"):
            self._build(root, tmp_path / "output")


class TestCandidateBuildCLI:
    def test_cli_builds_candidate_and_provenance_parquet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "repositories"
        helpers.create_repository(root / "personal", "person@example.com")
        candidates_path = tmp_path / "output" / "candidates.parquet"
        provenance_path = tmp_path / "output" / "provenance.parquet"
        candidates_path.parent.mkdir()
        (tmp_path / "qwen-commit.toml").write_text(
            f'[history]\nroots = ["{root.as_posix()}"]\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            [
                "data",
                "build-candidates",
                "--output",
                str(candidates_path),
                "--provenance",
                str(provenance_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert candidates_path.is_file()
        assert provenance_path.is_file()
