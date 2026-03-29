"""ferroml compare — train and compare multiple models on a dataset."""
from __future__ import annotations

import numpy as np
import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data
from ferroml.cli._registry import construct_model


def compare(
    models: str = typer.Option(..., "--models", help="Comma-separated model names."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    test_size: float = typer.Option(0.2, "--test-size", help="Hold-out fraction for scoring."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Train multiple models and compare their performance."""
    import time
    from ferroml import metrics

    X, y, _ = load_data(data, target)
    is_classification = len(np.unique(y)) <= 20

    split_idx = int(X.shape[0] * (1 - test_size))
    indices = np.random.RandomState(42).permutation(X.shape[0])
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model_names = [m.strip() for m in models.split(",")]
    leaderboard = []

    for name in model_names:
        try:
            mdl = construct_model(name)
            t0 = time.perf_counter()
            mdl.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0
            preds = mdl.predict(X_test)

            if is_classification:
                score = float(metrics.accuracy_score(y_test, preds))
                metric_name = "accuracy"
            else:
                score = float(metrics.r2_score(y_test, preds))
                metric_name = "r2"

            leaderboard.append({
                "model": name,
                "score": round(score, 6),
                "metric": metric_name,
                "fit_time_seconds": round(fit_time, 4),
            })
        except Exception as e:
            leaderboard.append({"model": name, "error": str(e)})

    leaderboard.sort(key=lambda x: x.get("score", float("-inf")), reverse=True)

    result_data = {
        "n_models": len(model_names),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "leaderboard": leaderboard,
    }
    output(result_data, json_mode)
