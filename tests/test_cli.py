"""Tests for the foundation CLI."""

from typer.testing import CliRunner

from qwen_commit import __version__
from qwen_commit.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "qwen-commit" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"qwen-commit {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app)

    assert result.exit_code == 0
    assert "Tooling for a local Qwen commit-message model." in result.output
