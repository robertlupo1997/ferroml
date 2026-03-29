"""ferroml predict — generate predictions with a fitted model."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, load_model


def predict(
    model: str = typer.Option(..., "--model", "-m", help="Path to fitted model (.pkl)."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Target column to exclude from features."),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Save predictions to CSV."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Generate predictions with a fitted model."""
    mdl = load_model(model)
    X, _, feature_names = load_data(data, target)

    preds = mdl.predict(X)
    preds_list = [round(float(p), 6) for p in preds]

    if output_path:
        import polars as pl
        pl.DataFrame({"prediction": preds_list}).write_csv(output_path)

    result_data = {
        "n_samples": len(preds_list),
        "predictions": preds_list,
    }
    output(result_data, json_mode)
