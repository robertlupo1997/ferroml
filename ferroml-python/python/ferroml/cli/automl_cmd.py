"""ferroml automl — run automated machine learning search."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, save_model


def automl(
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    task: str = typer.Option("classification", "--task", help="Task: classification or regression."),
    time_budget: int = typer.Option(60, "--time-budget", help="Time budget in seconds."),
    metric: Optional[str] = typer.Option(None, "--metric", help="Evaluation metric."),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Save best model to disk."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Run AutoML to find the best model for a dataset."""
    from ferroml.automl import AutoML, AutoMLConfig

    X, y, _ = load_data(data, target)

    if metric is None:
        metric = "roc_auc" if task == "classification" else "rmse"

    config = AutoMLConfig(
        task=task,
        metric=metric,
        time_budget_seconds=time_budget,
    )
    aml = AutoML(config)
    result = aml.fit(X, y)

    leaderboard = []
    for entry in result.leaderboard[:20]:  # Cap at 20 for JSON sanity
        leaderboard.append({
            "rank": entry.rank,
            "algorithm": entry.algorithm,
            "cv_score": round(float(entry.cv_score), 6),
            "cv_std": round(float(entry.cv_std), 6),
            "ci_lower": round(float(entry.ci_lower), 6),
            "ci_upper": round(float(entry.ci_upper), 6),
            "training_time": round(float(entry.training_time_seconds), 4),
        })

    if output_path and result.best_model():
        from ferroml.cli._registry import construct_model
        best = result.best_model()
        mdl = construct_model(best.algorithm, best.params)
        mdl.fit(X, y)
        save_model(mdl, output_path)

    result_data = {
        "task": task,
        "metric": metric,
        "n_trials": result.n_successful_trials,
        "total_time_seconds": round(float(result.total_time_seconds), 2),
        "leaderboard": leaderboard,
        "summary": result.summary(),
    }
    output(result_data, json_mode)
