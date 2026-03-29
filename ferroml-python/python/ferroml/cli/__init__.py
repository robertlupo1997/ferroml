"""FerroML command-line interface."""
import typer

from ferroml import __version__

app = typer.Typer(
    name="ferroml",
    help="FerroML: Statistically rigorous ML from the command line.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"ferroml {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit.", callback=version_callback, is_eager=True,
    ),
):
    """FerroML: Statistically rigorous ML from the command line."""


# __main__ support: python -m ferroml.cli
def cli_main():
    app()
