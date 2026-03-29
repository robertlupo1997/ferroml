"""Compare multiple models on a dataset with statistical significance testing.

Usage: Claude adapts this to compare recommended models.
Output: Leaderboard with scores, timing, and pairwise significance tests.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl


def compare(
    path: str,
    target: str,
    model_names: list[str] | None = None,
    task: str | None = None,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Compare models via cross-validation with statistical tests.

    Parameters
    ----------
    path : str
        Path to CSV or Parquet.
    target : str
        Target column.
    model_names : list or None
        Models to compare. If None, uses ferroml.recommend().
    task : str or None
        "classification" or "regression". Auto-detected if None.
    cv_folds : int
        Number of CV folds.
    seed : int
        Random seed.

    Returns
    -------
    dict with leaderboard, pairwise_tests, recommendation
    """
    import ferroml
    from ferroml import metrics
    from ferroml.cli._registry import construct_model
    from ferroml.preprocessing import StandardScaler

    # Load
    if path.endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, infer_schema_length=10000)

    y = df[target].to_numpy().astype(np.float64)
    X = df.drop(target).select(pl.col("*").cast(pl.Float64, strict=False)).to_numpy().astype(np.float64)

    # Handle NaN
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = np.nanmedian(X[:, j])

    if task is None:
        task = "classification" if len(np.unique(y)) <= 20 else "regression"

    # Auto-select models if not provided
    if model_names is None:
        recs = ferroml.recommend(X, y, task=task)
        model_names = [r.algorithm for r in recs[:5]]

    # Cross-validation for each model
    rng = np.random.RandomState(seed)
    indices = rng.permutation(X.shape[0])
    fold_size = X.shape[0] // cv_folds

    leaderboard = []
    all_fold_scores: dict[str, list[float]] = {}

    for name in model_names:
        fold_scores = []
        t0 = time.perf_counter()

        for fold in range(cv_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv_folds - 1 else X.shape[0]
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            try:
                model = construct_model(name)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if task == "regression":
                    score = float(metrics.r2_score(y_test, preds))
                else:
                    score = float(metrics.accuracy_score(y_test, preds))
                fold_scores.append(score)
            except Exception as e:
                fold_scores.append(float("nan"))

        elapsed = time.perf_counter() - t0
        valid_scores = [s for s in fold_scores if not np.isnan(s)]

        if valid_scores:
            entry = {
                "model": name,
                "mean_score": round(float(np.mean(valid_scores)), 6),
                "std_score": round(float(np.std(valid_scores)), 6),
                "min_score": round(float(np.min(valid_scores)), 6),
                "max_score": round(float(np.max(valid_scores)), 6),
                "metric": "r2" if task == "regression" else "accuracy",
                "total_time_seconds": round(elapsed, 4),
                "folds_succeeded": len(valid_scores),
            }
            leaderboard.append(entry)
            all_fold_scores[name] = valid_scores

    leaderboard.sort(key=lambda x: x["mean_score"], reverse=True)

    # Pairwise significance tests
    pairwise_tests = []
    if len(all_fold_scores) >= 2:
        from ferroml.stats import ttest_ind
        names = list(all_fold_scores.keys())
        best_name = leaderboard[0]["model"]
        for name in names:
            if name == best_name:
                continue
            if len(all_fold_scores[best_name]) >= 3 and len(all_fold_scores[name]) >= 3:
                result = ttest_ind(
                    np.array(all_fold_scores[best_name]),
                    np.array(all_fold_scores[name]),
                )
                pairwise_tests.append({
                    "model_a": best_name,
                    "model_b": name,
                    "t_statistic": round(float(result["statistic"]) if isinstance(result, dict) else float(result[0]), 4),
                    "p_value": round(float(result["p_value"]) if isinstance(result, dict) else float(result[1]), 6),
                })

    return {
        "task": task,
        "cv_folds": cv_folds,
        "n_models": len(model_names),
        "leaderboard": leaderboard,
        "pairwise_tests": pairwise_tests,
        "best_model": leaderboard[0]["model"] if leaderboard else None,
    }
