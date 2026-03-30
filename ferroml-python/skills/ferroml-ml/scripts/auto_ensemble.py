"""Automated diverse ensemble construction with optimized blending weights.

Usage: Claude adapts this to build ensembles that outperform individual models.
Output: Diverse model subset, blending weights, and significance-tested improvement.
"""
from __future__ import annotations

import time

import numpy as np


def build(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    model_names: list[str] | None = None,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Build a diverse blended ensemble.

    1. Train multiple models, collect CV out-of-fold predictions.
    2. Compute prediction correlation matrix.
    3. Select a diverse subset (low correlation).
    4. Optimize blending weights via grid search.
    5. Compare ensemble vs best single model with significance test.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target values.
    task : str
        "classification" or "regression".
    model_names : list or None
        Models to consider. If None, uses ferroml.recommend() top 5.
    cv_folds : int
        Number of CV folds.
    seed : int
        Random seed.

    Returns
    -------
    dict with models_used, weights, ensemble_score, best_single_score,
    improvement, p_value, is_significant, correlation_matrix
    """
    import ferroml
    from ferroml import metrics
    from ferroml.cli._registry import construct_model
    from ferroml.preprocessing import StandardScaler
    from ferroml.stats import ttest_ind

    if task == "classification":
        metric_fn = metrics.accuracy_score
        metric_name = "accuracy"
    else:
        metric_fn = metrics.r2_score
        metric_name = "r2"

    # Auto-select models if not provided
    if model_names is None:
        recs = ferroml.recommend(X, y, task=task)
        model_names = [r.algorithm for r in recs[:5]]

    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)
    fold_size = n_samples // cv_folds

    # Collect out-of-fold predictions for each model
    oof_preds: dict[str, np.ndarray] = {}
    fold_scores: dict[str, list[float]] = {}
    model_times: dict[str, float] = {}

    for name in model_names:
        oof = np.full(n_samples, np.nan)
        scores = []
        t0 = time.perf_counter()

        for fold in range(cv_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < cv_folds - 1 else n_samples
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            try:
                model = construct_model(name)
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
                oof[test_idx] = preds
                score = float(metric_fn(y_test, preds))
                scores.append(score)
            except Exception:
                scores.append(float("nan"))

        elapsed = time.perf_counter() - t0

        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores and not np.all(np.isnan(oof)):
            oof_preds[name] = oof
            fold_scores[name] = valid_scores
            model_times[name] = round(elapsed, 4)

    if len(oof_preds) < 2:
        # Not enough models succeeded for ensemble
        best_name = max(fold_scores, key=lambda n: float(np.mean(fold_scores[n]))) if fold_scores else model_names[0]
        return {
            "models_used": list(oof_preds.keys()),
            "weights": {best_name: 1.0} if best_name in oof_preds else {},
            "ensemble_score": round(float(np.mean(fold_scores.get(best_name, [0]))), 6),
            "best_single_score": round(float(np.mean(fold_scores.get(best_name, [0]))), 6),
            "best_single_model": best_name,
            "improvement": 0.0,
            "p_value": None,
            "is_significant": False,
            "metric": metric_name,
            "note": "Too few models succeeded for ensemble construction.",
        }

    # Compute prediction correlation matrix
    names = list(oof_preds.keys())
    n_models = len(names)
    pred_matrix = np.column_stack([oof_preds[n] for n in names])

    # Handle NaN by filling with column mean
    for j in range(pred_matrix.shape[1]):
        col = pred_matrix[:, j]
        mask = np.isnan(col)
        if mask.any():
            pred_matrix[mask, j] = np.nanmean(col)

    corr_matrix = np.corrcoef(pred_matrix.T)
    correlation_info = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i < j:
                correlation_info[f"{ni}_vs_{nj}"] = round(float(corr_matrix[i, j]), 4)

    # Select diverse subset: greedily pick models with low avg correlation
    selected = _select_diverse(names, corr_matrix, max_models=min(5, n_models))

    # Optimize blending weights via grid search
    selected_preds = np.column_stack([oof_preds[n] for n in selected])
    # Handle NaN
    for j in range(selected_preds.shape[1]):
        col = selected_preds[:, j]
        mask = np.isnan(col)
        if mask.any():
            selected_preds[mask, j] = np.nanmean(col)

    valid_mask = ~np.isnan(selected_preds).any(axis=1) & ~np.isnan(y)
    y_valid = y[valid_mask]
    preds_valid = selected_preds[valid_mask]

    best_weights, best_score = _optimize_weights(
        preds_valid, y_valid, metric_fn, task, len(selected)
    )

    weights = {name: round(w, 4) for name, w in zip(selected, best_weights)}

    # Best single model score
    single_scores = {n: float(np.mean(fold_scores[n])) for n in names}
    best_single_name = max(single_scores, key=lambda n: single_scores[n])
    best_single_score = single_scores[best_single_name]

    improvement = round(best_score - best_single_score, 6)

    # Significance test: ensemble fold scores vs best single model fold scores
    # Compute ensemble fold scores
    ensemble_fold_scores = []
    for fold in range(cv_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < cv_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]

        blend = np.zeros(len(test_idx))
        for name, w in weights.items():
            preds_i = oof_preds[name][test_idx]
            nan_mask = np.isnan(preds_i)
            if nan_mask.any():
                preds_i = preds_i.copy()
                preds_i[nan_mask] = np.nanmean(oof_preds[name])
            blend += w * preds_i

        if task == "classification":
            blend = np.round(blend)

        fold_y = y[test_idx]
        valid = ~np.isnan(fold_y) & ~np.isnan(blend)
        if valid.any():
            score = float(metric_fn(fold_y[valid], blend[valid]))
            ensemble_fold_scores.append(score)

    p_value = None
    is_significant = False
    if len(ensemble_fold_scores) >= 3 and len(fold_scores[best_single_name]) >= 3:
        try:
            min_folds = min(len(ensemble_fold_scores), len(fold_scores[best_single_name]))
            result = ttest_ind(
                np.array(ensemble_fold_scores[:min_folds]),
                np.array(fold_scores[best_single_name][:min_folds]),
            )
            if isinstance(result, dict):
                p_value = float(result["p_value"])
            else:
                p_value = float(result[1])
            is_significant = p_value < 0.05 and float(np.mean(ensemble_fold_scores)) > best_single_score
        except Exception:
            pass

    return {
        "models_used": selected,
        "weights": weights,
        "ensemble_score": round(best_score, 6),
        "ensemble_fold_scores": [round(s, 6) for s in ensemble_fold_scores],
        "best_single_model": best_single_name,
        "best_single_score": round(best_single_score, 6),
        "all_single_scores": {n: round(s, 6) for n, s in single_scores.items()},
        "improvement": improvement,
        "p_value": round(p_value, 6) if p_value is not None else None,
        "is_significant": is_significant,
        "metric": metric_name,
        "correlation_matrix": correlation_info,
        "model_times": model_times,
    }


def _select_diverse(
    names: list[str], corr_matrix: np.ndarray, max_models: int = 5
) -> list[str]:
    """Greedily select models with low average pairwise correlation."""
    n = len(names)
    if n <= max_models:
        return names

    # Start with the model that has lowest average correlation with others
    avg_corr = np.mean(np.abs(corr_matrix), axis=1)
    selected_idx = [int(np.argmin(avg_corr))]

    while len(selected_idx) < max_models:
        best_candidate = None
        best_diversity = float("inf")

        for i in range(n):
            if i in selected_idx:
                continue
            # Average absolute correlation with already-selected models
            avg = float(np.mean([abs(corr_matrix[i, j]) for j in selected_idx]))
            if avg < best_diversity:
                best_diversity = avg
                best_candidate = i

        if best_candidate is not None:
            selected_idx.append(best_candidate)
        else:
            break

    return [names[i] for i in selected_idx]


def _optimize_weights(
    preds: np.ndarray,
    y: np.ndarray,
    metric_fn: object,
    task: str,
    n_models: int,
) -> tuple[np.ndarray, float]:
    """Grid search over weight combinations that sum to 1."""
    if n_models == 1:
        return np.array([1.0]), float(metric_fn(y, preds[:, 0]))

    # For 2-3 models: fine grid. For 4+: coarser grid.
    if n_models <= 3:
        step = 0.1
    else:
        step = 0.2

    best_weights = np.ones(n_models) / n_models
    best_score = -float("inf")

    # Generate weight combinations
    candidates = _weight_grid(n_models, step)

    for w in candidates:
        blend = preds @ w
        if task == "classification":
            blend = np.round(blend)
        try:
            score = float(metric_fn(y, blend))
            if score > best_score:
                best_score = score
                best_weights = w
        except Exception:
            continue

    return best_weights, best_score


def _weight_grid(n: int, step: float) -> list[np.ndarray]:
    """Generate all weight vectors of length n that sum to 1.0 with given step."""
    if n == 1:
        return [np.array([1.0])]

    vals = np.arange(0.0, 1.0 + step / 2, step)
    results = []
    _weight_grid_helper(n, vals, [], 1.0, results)

    # Always include equal weights
    results.append(np.ones(n) / n)
    return results


def _weight_grid_helper(
    remaining: int,
    vals: np.ndarray,
    current: list[float],
    budget: float,
    results: list[np.ndarray],
) -> None:
    """Recursive helper to build weight combinations."""
    if remaining == 1:
        if budget >= -0.01:
            current.append(max(0.0, budget))
            results.append(np.array(current[:-1] + [max(0.0, budget)]))
        return

    for v in vals:
        if v > budget + 0.01:
            break
        _weight_grid_helper(remaining - 1, vals, current + [v], budget - v, results)


def print_ensemble(result: dict) -> None:
    """Print a human-readable ensemble summary."""
    print(f"Ensemble ({result['metric']}): {result['ensemble_score']:.4f}")
    print(f"Best single model: {result['best_single_model']} ({result['best_single_score']:.4f})")
    print(f"Improvement: {result['improvement']:+.4f}")

    if result["p_value"] is not None:
        sig = "significant" if result["is_significant"] else "not significant"
        print(f"Significance: p={result['p_value']:.4f} ({sig})")
    print()

    print("Models and weights:")
    for name, w in result["weights"].items():
        single = result["all_single_scores"].get(name, 0)
        print(f"  {name}: weight={w:.2f}  (solo={single:.4f})")
    print()

    if result.get("correlation_matrix"):
        print("Prediction correlations:")
        for pair, corr in result["correlation_matrix"].items():
            print(f"  {pair}: {corr:.3f}")
