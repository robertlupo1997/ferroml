"""Hyperparameter tuning via random search with cross-validation.

Usage: Claude adapts this to tune a specific model on the user's data.
Output: Best parameters, all trial results, and improvement over defaults.
"""
from __future__ import annotations

import time

import numpy as np


def tune(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    task: str,
    n_trials: int = 50,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Tune hyperparameters via random search with cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target array.
    model_name : str
        Model name (e.g. "RandomForest", "Ridge").
    task : str
        "regression" or "classification".
    n_trials : int
        Number of random parameter configurations to try.
    cv_folds : int
        Number of CV folds.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: best_params, best_score, best_metric, all_trials,
    improvement_over_default, search_space, total_time_seconds
    """
    from ferroml.cli._registry import construct_model

    rng = np.random.RandomState(seed)

    # Get search space
    try:
        base_model = construct_model(model_name)
        space = base_model.search_space()
    except Exception:
        space = {}

    if not space:
        return {
            "best_params": {},
            "best_score": None,
            "best_metric": None,
            "all_trials": [],
            "improvement_over_default": 0.0,
            "search_space": {},
            "total_time_seconds": 0.0,
            "error": f"No search space available for {model_name}.",
        }

    # Evaluate default model first
    default_score = _cv_score(X, y, model_name, None, task, cv_folds, rng)

    # Random search
    t0 = time.perf_counter()
    all_trials: list[dict] = []

    for trial in range(n_trials):
        trial_rng = np.random.RandomState(seed + trial + 1)
        params = _sample_params(space, trial_rng)

        try:
            score = _cv_score(X, y, model_name, params, task, cv_folds, rng)
            all_trials.append({
                "trial": trial + 1,
                "params": params,
                "score": round(score, 6),
                "status": "ok",
            })
        except Exception as e:
            all_trials.append({
                "trial": trial + 1,
                "params": params,
                "score": float("-inf"),
                "status": f"error: {str(e)[:100]}",
            })

    elapsed = time.perf_counter() - t0

    # Sort by score (descending)
    all_trials.sort(key=lambda t: t["score"], reverse=True)

    best_trial = all_trials[0] if all_trials else None
    best_score = best_trial["score"] if best_trial else None
    best_params = best_trial["params"] if best_trial else {}
    metric_name = "r2" if task == "regression" else "accuracy"

    improvement = 0.0
    if best_score is not None and default_score is not None:
        improvement = round(best_score - default_score, 6)

    return {
        "model_name": model_name,
        "best_params": best_params,
        "best_score": best_score,
        "best_metric": metric_name,
        "default_score": round(default_score, 6) if default_score is not None else None,
        "improvement_over_default": improvement,
        "all_trials": all_trials,
        "n_trials": len(all_trials),
        "search_space": {k: str(v) for k, v in space.items()},
        "total_time_seconds": round(elapsed, 2),
    }


def _cv_score(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    params: dict | None,
    task: str,
    cv_folds: int,
    rng: np.random.RandomState,
) -> float:
    """Cross-validate a single configuration and return mean score."""
    from ferroml import metrics
    from ferroml.cli._registry import construct_model
    from ferroml.preprocessing import StandardScaler

    indices = rng.permutation(X.shape[0])
    fold_size = X.shape[0] // cv_folds
    scores = []

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

        model = construct_model(model_name, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if task == "regression":
            scores.append(float(metrics.r2_score(y_test, preds)))
        else:
            scores.append(float(metrics.accuracy_score(y_test, preds)))

    return float(np.mean(scores))


def _sample_params(space: dict, rng: np.random.RandomState) -> dict:
    """Sample a parameter configuration from a search space.

    The search space dict maps param names to tuples like:
    - ("int", low, high)
    - ("float", low, high)
    - ("log_float", low, high)
    - ("categorical", [options])
    """
    params = {}
    for name, spec in space.items():
        if isinstance(spec, (list, tuple)):
            kind = spec[0] if isinstance(spec[0], str) else "categorical"
            if kind == "int":
                params[name] = int(rng.randint(spec[1], spec[2] + 1))
            elif kind == "float":
                params[name] = round(float(rng.uniform(spec[1], spec[2])), 6)
            elif kind == "log_float":
                log_low = np.log(max(spec[1], 1e-12))
                log_high = np.log(spec[2])
                params[name] = round(float(np.exp(rng.uniform(log_low, log_high))), 8)
            elif kind == "categorical":
                options = spec[1] if len(spec) > 1 and isinstance(spec[1], list) else list(spec)
                params[name] = options[rng.randint(len(options))]
            else:
                # Treat as categorical list
                params[name] = spec[rng.randint(len(spec))]
        else:
            params[name] = spec  # Fixed value
    return params


def print_summary(result: dict) -> None:
    """Print a human-readable tuning summary."""
    print(f"Model: {result.get('model_name', 'unknown')}")
    print(f"Metric: {result.get('best_metric', 'n/a')}")
    print(f"Trials: {result.get('n_trials', 0)} in {result.get('total_time_seconds', 0)}s")
    print()

    if result.get("default_score") is not None:
        print(f"Default score: {result['default_score']}")
    if result.get("best_score") is not None:
        print(f"Best score:    {result['best_score']}")
    improvement = result.get("improvement_over_default", 0)
    sign = "+" if improvement >= 0 else ""
    print(f"Improvement:   {sign}{improvement}")
    print()

    if result.get("best_params"):
        print("Best parameters:")
        for k, v in result["best_params"].items():
            print(f"  {k}: {v}")
    print()

    if result.get("all_trials"):
        print("Top 5 trials:")
        for trial in result["all_trials"][:5]:
            print(f"  #{trial['trial']}: score={trial['score']}  params={trial['params']}")
