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


# Register subcommands
from ferroml.cli.train import train  # noqa: E402
from ferroml.cli.predict import predict  # noqa: E402
app.command(name="train")(train)
app.command(name="predict")(predict)

from ferroml.cli.evaluate import evaluate  # noqa: E402
from ferroml.cli.recommend import recommend  # noqa: E402
from ferroml.cli.info import info  # noqa: E402
app.command(name="evaluate")(evaluate)
app.command(name="recommend")(recommend)
app.command(name="info")(info)

from ferroml.cli.compare import compare  # noqa: E402
from ferroml.cli.diagnose import diagnose  # noqa: E402
app.command(name="compare")(compare)
app.command(name="diagnose")(diagnose)

from ferroml.cli.automl_cmd import automl as automl_cmd  # noqa: E402
from ferroml.cli.export import export  # noqa: E402
app.command(name="automl")(automl_cmd)
app.command(name="export")(export)


# __main__ support: python -m ferroml.cli
def cli_main():
    app()
