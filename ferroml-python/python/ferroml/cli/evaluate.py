"""ferroml evaluate — score a fitted model on labeled data."""
from __future__ import annotations

from typing import Optional

import numpy as np
import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, load_model


_REGRESSION_METRICS = {
    "rmse": "rmse",
    "mse": "mse",
    "mae": "mae",
    "r2": "r2_score",
    "explained_variance": "explained_variance",
    "max_error": "max_error",
    "mape": "mape",
    "median_absolute_error": "median_absolute_error",
}

_CLASSIFICATION_METRICS = {
    "accuracy": "accuracy_score",
    "precision": "precision_score",
    "recall": "recall_score",
    "f1": "f1_score",
    "mcc": "matthews_corrcoef",
    "balanced_accuracy": "balanced_accuracy_score",
}


def _compute_metric(name: str, y_true, y_pred) -> float:
    from ferroml import metrics
    all_metrics = {**_REGRESSION_METRICS, **_CLASSIFICATION_METRICS}
    fn_name = all_metrics.get(name)
    if fn_name is None:
        print(f"Error: unknown metric '{name}'. Available: {', '.join(sorted(all_metrics))}")
        raise SystemExit(1)
    fn = getattr(metrics, fn_name)
    return float(fn(y_true, y_pred))


def evaluate(
    model: str = typer.Option(..., "--model", "-m", help="Path to fitted model (.pkl)."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    metrics_str: Optional[str] = typer.Option(None, "--metrics", help="Comma-separated metric names."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Score a fitted model on labeled data."""
    mdl = load_model(model)
    X, y, _ = load_data(data, target)
    preds = mdl.predict(X)

    if metrics_str:
        metric_names = [m.strip() for m in metrics_str.split(",")]
    else:
        if len(np.unique(y)) <= 20:
            metric_names = ["accuracy", "f1", "precision", "recall"]
        else:
            metric_names = ["rmse", "r2", "mae"]

    scores = {}
    for name in metric_names:
        scores[name] = round(_compute_metric(name, y, preds), 6)

    result_data = {
        "n_samples": X.shape[0],
        "metrics": scores,
    }
    output(result_data, json_mode)
