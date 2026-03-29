"""ferroml train — fit a model on a dataset."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, save_model
from ferroml.cli._registry import construct_model


def train(
    model: str = typer.Option(..., "--model", "-m", help="Model class name (e.g. LinearRegression)."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    output_path: str = typer.Option("model.pkl", "--output", "-o", help="Path to save fitted model."),
    params: Optional[str] = typer.Option(None, "--params", "-p", help="Model params as JSON string."),
    test_size: Optional[float] = typer.Option(None, "--test-size", help="Hold-out fraction for scoring."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Fit a model on a dataset and save to disk."""
    import json as json_mod
    import numpy as np

    parsed_params = json_mod.loads(params) if params else None
    mdl = construct_model(model, parsed_params)

    X, y, feature_names = load_data(data, target)

    result_data: dict = {"model": model, "n_samples": X.shape[0], "n_features": X.shape[1]}

    if test_size:
        split_idx = int(X.shape[0] * (1 - test_size))
        indices = np.random.RandomState(42).permutation(X.shape[0])
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        if len(np.unique(y)) <= 20:
            from ferroml.metrics import accuracy_score
            score = accuracy_score(y_test, preds)
            result_data["test_metric"] = "accuracy"
        else:
            from ferroml.metrics import r2_score
            score = r2_score(y_test, preds)
            result_data["test_metric"] = "r2"
        result_data["test_score"] = round(float(score), 6)
        result_data["test_samples"] = len(test_idx)
    else:
        mdl.fit(X, y)

    save_model(mdl, output_path)
    result_data["status"] = "fitted"
    result_data["output"] = output_path

    output(result_data, json_mode)
