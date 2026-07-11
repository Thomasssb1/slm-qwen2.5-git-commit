"""Private dataset-building commands."""

from pathlib import Path
from typing import Annotated

import typer

from qwen_commit.candidates import CandidateBuildError, build_candidates
from qwen_commit.history import HistoryScanError, load_history_config, scan_history

data_app = typer.Typer(help="Build private training data from configured local Git histories.")


@data_app.command("build-candidates")
def build_candidate_data(
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Candidate Parquet output path."),
    ] = Path("candidates.parquet"),
    provenance_path: Annotated[
        Path,
        typer.Option("--provenance", help="Private provenance Parquet output path."),
    ] = Path("provenance.parquet"),
) -> None:
    """Extract filtered historical commits into private Parquet files."""
    try:
        report = build_candidates(
            scan_history(load_history_config(Path("qwen-commit.toml"))),
            output_path,
            provenance_path,
        )
    except (CandidateBuildError, HistoryScanError, OSError) as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(f"Accepted candidates: {report.accepted_count}")
    typer.echo(f"Rejected commits: {report.rejected_count}")
    for reason, count in sorted(report.rejection_counts.items()):
        typer.echo(f"Rejected {reason.value}: {count}")
    typer.echo(f"Candidates: {report.candidates_path}")
    typer.echo(f"Provenance: {report.provenance_path}")
