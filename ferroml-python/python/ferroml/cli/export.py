"""ferroml export — export a model to ONNX format."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_model


def export(
    model: str = typer.Option(..., "--model", "-m", help="Path to fitted model (.pkl)."),
    output_path: str = typer.Option(..., "--output", "-o", help="Path for ONNX output."),
    n_features: int = typer.Option(..., "--n-features", help="Number of input features."),
    model_name: Optional[str] = typer.Option(None, "--name", help="ONNX model name."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Export a fitted model to ONNX format."""
    mdl = load_model(model)

    if not hasattr(mdl, "export_onnx"):
        typer.echo("Error: this model does not support ONNX export.", err=True)
        raise typer.Exit(1)

    mdl.export_onnx(output_path, model_name=model_name)

    result_data = {
        "status": "exported",
        "format": "onnx",
        "output": output_path,
        "n_features": n_features,
    }
    output(result_data, json_mode)
