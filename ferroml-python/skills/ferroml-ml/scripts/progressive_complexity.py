"""Progressive complexity — start simple, add complexity only when justified.

Usage: Claude adapts this to find the simplest model that performs well enough.
Output: Model comparison ladder with statistical significance at each step.
"""
from __future__ import annotations

import time

import numpy as np


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Evaluate models of increasing complexity, stopping when improvement is not significant.

    Level 1: Baseline (LinearRegression or LogisticRegression)
    Level 2: Medium (RandomForest)
    Level 3: Complex (HistGradientBoosting)

    At each level, cross-validates and compares to the previous level with
    a paired t-test. Stops when improvement is not statistically significant
    (p > 0.05).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target values.
    task : str
        "classification" or "regression".
    cv_folds : int
        Number of CV folds.
    seed : int
        Random seed.

    Returns
    -------
    dict with levels, recommended_level, reasoning, metric
    """
    from ferroml.cli._registry import construct_model
    from ferroml.preprocessing import StandardScaler
    from ferroml.stats import ttest_ind
    from ferroml import metrics

    # Define model progression
    if task == "classification":
        model_levels = [
            {"name": "LogisticRegression", "level": 1, "complexity": "simple"},
            {"name": "RandomForestClassifier", "level": 2, "complexity": "medium"},
            {"name": "HistGradientBoostingClassifier", "level": 3, "complexity": "complex"},
        ]
        metric_name = "accuracy"
        metric_fn = metrics.accuracy_score
    else:
        model_levels = [
            {"name": "LinearRegression", "level": 1, "complexity": "simple"},
            {"name": "RandomForestRegressor", "level": 2, "complexity": "medium"},
            {"name": "HistGradientBoostingRegressor", "level": 3, "complexity": "complex"},
        ]
        metric_name = "r2"
        metric_fn = metrics.r2_score

    # Create CV folds
    rng = np.random.RandomState(seed)
    indices = rng.permutation(X.shape[0])
    fold_size = X.shape[0] // cv_folds

    levels = []
    recommended_level = None
    stop_reason = None
    prev_fold_scores = None

    for model_info in model_levels:
        name = model_info["name"]
        t0 = time.perf_counter()
        fold_scores = []

        for fold in range(cv_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv_folds - 1 else X.shape[0]
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            try:
                model = construct_model(name)
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
                score = float(metric_fn(y_test, preds))
                fold_scores.append(score)
            except Exception:
                fold_scores.append(float("nan"))

        elapsed = time.perf_counter() - t0
        valid_scores = [s for s in fold_scores if not np.isnan(s)]

        if not valid_scores:
            levels.append({
                "name": name,
                "level": model_info["level"],
                "complexity": model_info["complexity"],
                "mean_score": None,
                "std_score": None,
                "time_seconds": round(elapsed, 4),
                "p_vs_previous": None,
                "significant_improvement": None,
                "error": "All folds failed",
            })
            continue

        mean_score = float(np.mean(valid_scores))
        std_score = float(np.std(valid_scores))

        # Paired t-test vs previous level
        p_value = None
        significant = None
        if prev_fold_scores is not None and len(valid_scores) >= 3 and len(prev_fold_scores) >= 3:
            # Align fold counts
            min_folds = min(len(valid_scores), len(prev_fold_scores))
            try:
                result = ttest_ind(
                    np.array(valid_scores[:min_folds]),
                    np.array(prev_fold_scores[:min_folds]),
                )
                if isinstance(result, dict):
                    p_value = float(result["p_value"])
                else:
                    p_value = float(result[1])
                significant = p_value < 0.05 and mean_score > float(np.mean(prev_fold_scores[:min_folds]))
            except Exception:
                p_value = None
                significant = None

        level_result = {
            "name": name,
            "level": model_info["level"],
            "complexity": model_info["complexity"],
            "mean_score": round(mean_score, 6),
            "std_score": round(std_score, 6),
            "fold_scores": [round(s, 6) for s in valid_scores],
            "time_seconds": round(elapsed, 4),
            "p_vs_previous": round(p_value, 6) if p_value is not None else None,
            "significant_improvement": significant,
        }
        levels.append(level_result)

        # Determine if we should stop
        if recommended_level is None:
            recommended_level = name

        if significant is True:
            recommended_level = name
        elif significant is False and prev_fold_scores is not None:
            # Improvement not significant — stop here
            stop_reason = (
                f"Stopped at {model_info['complexity']} level: {name} did not significantly "
                f"outperform the previous model (p={round(p_value, 4) if p_value else 'N/A'})."
            )
            break

        prev_fold_scores = valid_scores

    # Build reasoning
    if stop_reason is None:
        if len(levels) > 0:
            best = max(levels, key=lambda x: x["mean_score"] if x["mean_score"] is not None else -999)
            recommended_level = best["name"]
            stop_reason = (
                f"All levels evaluated. {recommended_level} achieved the best score."
            )

    reasoning = f"Recommended: {recommended_level}. {stop_reason}"

    return {
        "task": task,
        "metric": metric_name,
        "cv_folds": cv_folds,
        "levels": levels,
        "recommended_level": recommended_level,
        "reasoning": reasoning,
    }


def print_evaluation(result: dict) -> None:
    """Print a human-readable evaluation ladder."""
    print(f"Task: {result['task']} | Metric: {result['metric']} | CV folds: {result['cv_folds']}")
    print()

    for level in result["levels"]:
        marker = " <-- recommended" if level["name"] == result["recommended_level"] else ""
        score = f"{level['mean_score']:.4f} +/- {level['std_score']:.4f}" if level["mean_score"] is not None else "FAILED"
        print(f"  Level {level['level']} ({level['complexity']}): {level['name']}")
        print(f"    Score: {score}  ({level['time_seconds']:.2f}s){marker}")
        if level["p_vs_previous"] is not None:
            sig = "YES" if level["significant_improvement"] else "NO"
            print(f"    vs previous: p={level['p_vs_previous']:.4f}, significant={sig}")
        print()

    print(result["reasoning"])
