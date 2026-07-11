"""Root command-line application."""

from typing import Annotated

import typer

from qwen_commit import __version__
from qwen_commit.cli.history import history_app

app = typer.Typer(
    name="qwen-commit",
    help="Tooling for a local Qwen commit-message model.",
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_show_locals=False,
)
app.add_typer(history_app, name="history")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"qwen-commit {__version__}")
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_show_version,
            help="Show the installed version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Show root help until a command group is selected."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
