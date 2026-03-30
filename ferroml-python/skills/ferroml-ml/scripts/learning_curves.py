"""Learning curves for bias vs variance diagnosis.

Usage: Claude adapts this to diagnose model fit on the user's data.
Output: Train/test scores at increasing data sizes with diagnosis.
"""
from __future__ import annotations

import time

import numpy as np


def compute(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    task: str,
    train_sizes: list[float] | None = None,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Compute learning curves and diagnose bias/variance.

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
    train_sizes : list[float] or None
        Fractions of training data to use (e.g. [0.1, 0.2, ..., 1.0]).
        Defaults to [0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9, 1.0].
    cv_folds : int
        Number of CV folds.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: train_sizes, train_scores, test_scores, diagnosis,
    recommendation, model_name, metric
    """
    from ferroml import metrics
    from ferroml.cli._registry import construct_model
    from ferroml.preprocessing import StandardScaler

    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9, 1.0]

    n_samples = X.shape[0]
    metric_name = "r2" if task == "regression" else "accuracy"

    all_train_scores: list[dict] = []
    all_test_scores: list[dict] = []

    t0 = time.perf_counter()

    for frac in train_sizes:
        fold_train_scores = []
        fold_test_scores = []

        for fold in range(cv_folds):
            fold_rng = np.random.RandomState(seed + fold)
            indices = fold_rng.permutation(n_samples)
            fold_size = n_samples // cv_folds
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv_folds - 1 else n_samples
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            # Subsample training data
            n_train = max(2, int(len(train_idx) * frac))
            train_idx = train_idx[:n_train]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            try:
                model = construct_model(model_name)
                model.fit(X_train_s, y_train)

                train_preds = model.predict(X_train_s)
                test_preds = model.predict(X_test_s)

                if task == "regression":
                    fold_train_scores.append(float(metrics.r2_score(y_train, train_preds)))
                    fold_test_scores.append(float(metrics.r2_score(y_test, test_preds)))
                else:
                    fold_train_scores.append(float(metrics.accuracy_score(y_train, train_preds)))
                    fold_test_scores.append(float(metrics.accuracy_score(y_test, test_preds)))
            except Exception:
                pass

        if fold_train_scores:
            all_train_scores.append({
                "fraction": frac,
                "n_samples": int(n_samples * frac * (cv_folds - 1) / cv_folds),
                "mean": round(float(np.mean(fold_train_scores)), 6),
                "std": round(float(np.std(fold_train_scores)), 6),
            })
            all_test_scores.append({
                "fraction": frac,
                "n_samples": int(n_samples * frac * (cv_folds - 1) / cv_folds),
                "mean": round(float(np.mean(fold_test_scores)), 6),
                "std": round(float(np.std(fold_test_scores)), 6),
            })

    elapsed = time.perf_counter() - t0

    # --- Diagnosis ---
    diagnosis, recommendation = _diagnose(all_train_scores, all_test_scores, task)

    return {
        "model_name": model_name,
        "metric": metric_name,
        "train_sizes": [s["fraction"] for s in all_train_scores],
        "train_scores": all_train_scores,
        "test_scores": all_test_scores,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "total_time_seconds": round(elapsed, 2),
    }


def _diagnose(
    train_scores: list[dict],
    test_scores: list[dict],
    task: str,
) -> tuple[str, list[str]]:
    """Diagnose bias/variance from learning curves."""
    if not train_scores or not test_scores:
        return "insufficient_data", ["Not enough data points to diagnose."]

    final_train = train_scores[-1]["mean"]
    final_test = test_scores[-1]["mean"]
    gap = final_train - final_test

    # Thresholds depend on task
    if task == "regression":
        good_threshold = 0.7
        decent_threshold = 0.4
    else:
        good_threshold = 0.85
        decent_threshold = 0.65

    recommendations: list[str] = []

    # Check if test score is still improving (look at last 3 points)
    if len(test_scores) >= 3:
        recent_test = [t["mean"] for t in test_scores[-3:]]
        still_improving = recent_test[-1] > recent_test[0] + 0.01

    else:
        still_improving = False

    # High bias: both curves are low
    if final_train < decent_threshold and final_test < decent_threshold:
        diagnosis = "high_bias"
        recommendations.append("Model is underfitting — both train and test scores are low.")
        recommendations.append("Try a more complex model (e.g., ensemble or neural network).")
        recommendations.append("Add more features or polynomial features.")
        recommendations.append("Reduce regularization if applicable.")

    # High variance: big gap between train and test
    elif gap > 0.15 and final_train > good_threshold:
        diagnosis = "high_variance"
        recommendations.append(f"Model is overfitting — train={final_train:.3f} but test={final_test:.3f} (gap={gap:.3f}).")
        recommendations.append("Get more training data if possible.")
        recommendations.append("Increase regularization.")
        recommendations.append("Reduce model complexity or number of features.")
        recommendations.append("Try dropout, early stopping, or pruning.")

    # Good fit: both curves converge high
    elif final_test >= good_threshold and gap < 0.1:
        diagnosis = "good_fit"
        recommendations.append(f"Model fits well — train={final_train:.3f}, test={final_test:.3f}.")
        recommendations.append("Consider hyperparameter tuning for marginal gains.")
        if still_improving:
            recommendations.append("Test score is still improving — more data may help.")

    # Moderate: decent but not great
    else:
        diagnosis = "moderate_fit"
        recommendations.append(f"Moderate fit — train={final_train:.3f}, test={final_test:.3f}.")
        if gap > 0.1:
            recommendations.append(f"Some overfitting (gap={gap:.3f}) — try regularization.")
        if final_test < good_threshold:
            recommendations.append("Test score could be higher — try a different model or more features.")
        if still_improving:
            recommendations.append("Performance still improving with more data — collect more samples.")

    return diagnosis, recommendations


def print_summary(result: dict) -> None:
    """Print a human-readable learning curve summary."""
    print(f"Model: {result['model_name']}  Metric: {result['metric']}")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Time: {result['total_time_seconds']}s")
    print()

    print(f"{'Fraction':<10} {'N_train':<10} {'Train':<12} {'Test':<12} {'Gap':<10}")
    print("-" * 54)
    for tr, te in zip(result["train_scores"], result["test_scores"]):
        gap = tr["mean"] - te["mean"]
        print(f"{tr['fraction']:<10.2f} {tr['n_samples']:<10} {tr['mean']:<12.4f} {te['mean']:<12.4f} {gap:<10.4f}")
    print()

    print("Recommendations:")
    for rec in result["recommendation"]:
        print(f"  - {rec}")
