"""History-discovery commands."""

from pathlib import Path
from typing import Annotated

import typer

from qwen_commit.history import (
    HistoryScanError,
    load_history_config,
    scan_history,
)

history_app = typer.Typer(help="Discover configured local Git histories without storing diffs.")


@history_app.command("scan")
def history_scan(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="TOML configuration file."),
    ] = Path("qwen-commit.toml"),
    json_path: Annotated[
        Path,
        typer.Option("--json", help="JSON report path."),
    ] = Path("history-scan.json"),
) -> None:
    """Discover trusted local repositories and count commits."""
    try:
        report = scan_history(load_history_config(config_path))
        json_path.write_text(f"{report.to_json()}\n", encoding="utf-8")
    except (HistoryScanError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
