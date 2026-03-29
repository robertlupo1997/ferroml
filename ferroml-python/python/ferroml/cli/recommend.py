"""ferroml recommend — suggest algorithms for a dataset."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data


def recommend(
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    task: Optional[str] = typer.Option(None, "--task", help="Task type: classification or regression."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Suggest the best algorithms for a dataset."""
    import numpy as np
    from ferroml import recommend as _recommend

    X, y, _ = load_data(data, target)

    if task is None:
        task = "classification" if len(np.unique(y)) <= 20 else "regression"

    recs = _recommend(X, y, task=task)

    result_data = {
        "task": task,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "recommendations": [
            {
                "algorithm": r.algorithm,
                "reason": r.reason,
                "estimated_fit_time": r.estimated_fit_time,
                "params": r.params,
                "score": r.score,
            }
            for r in recs
        ],
    }
    output(result_data, json_mode)
