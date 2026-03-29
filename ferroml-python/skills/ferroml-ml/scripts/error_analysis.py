"""Error analysis — find WHERE and WHY a model fails.

Usage: Claude runs this after evaluation to understand model weaknesses.
Output: Error segments, worst-performing subgroups, and actionable recommendations.
"""
from __future__ import annotations

import numpy as np


def analyze_errors(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None = None,
    task: str = "regression",
    n_bins: int = 5,
) -> dict:
    """Segment predictions by feature values to find failure modes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)
    feature_names : list of str or None
    task : "regression" or "classification"
    n_bins : int, number of bins for numeric segmentation

    Returns
    -------
    dict with error_segments, worst_segments, overall_metrics, recommendations
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Overall metrics
    if task == "regression":
        from ferroml.metrics import rmse, r2_score, mae
        overall = {
            "rmse": round(float(rmse(y_true, y_pred)), 6),
            "r2": round(float(r2_score(y_true, y_pred)), 6),
            "mae": round(float(mae(y_true, y_pred)), 6),
        }
        errors = np.abs(y_true - y_pred)
    else:
        from ferroml.metrics import accuracy_score, f1_score
        overall = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "f1": round(float(f1_score(y_true, y_pred)), 6),
        }
        errors = (y_true != y_pred).astype(float)

    # Segment errors by each feature
    segments = []
    for j, fname in enumerate(feature_names):
        col = X[:, j]
        if np.std(col) == 0:
            continue

        # Bin the feature
        try:
            bin_edges = np.percentile(col, np.linspace(0, 100, n_bins + 1))
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                continue
            bin_indices = np.digitize(col, bin_edges[1:-1])
        except Exception:
            continue

        for b in range(len(bin_edges) - 1):
            mask = bin_indices == b
            if mask.sum() < 5:
                continue

            seg_errors = errors[mask]
            seg_info = {
                "feature": fname,
                "bin": f"[{bin_edges[b]:.2f}, {bin_edges[b+1]:.2f}]",
                "n_samples": int(mask.sum()),
                "mean_error": round(float(np.mean(seg_errors)), 6),
            }

            if task == "regression":
                seg_info["segment_rmse"] = round(float(np.sqrt(np.mean(seg_errors ** 2))), 6)
            else:
                seg_info["error_rate"] = round(float(np.mean(seg_errors)), 6)

            segments.append(seg_info)

    # Find worst segments
    if task == "regression":
        segments.sort(key=lambda x: x.get("segment_rmse", 0), reverse=True)
    else:
        segments.sort(key=lambda x: x.get("error_rate", 0), reverse=True)

    worst = segments[:10]

    # High-error samples
    top_error_idx = np.argsort(errors)[-10:][::-1]
    high_error_samples = []
    for idx in top_error_idx:
        sample = {
            "index": int(idx),
            "true": round(float(y_true[idx]), 4),
            "predicted": round(float(y_pred[idx]), 4),
            "error": round(float(errors[idx]), 4),
        }
        high_error_samples.append(sample)

    # Recommendations
    recommendations = []
    if worst:
        w = worst[0]
        recommendations.append(
            f"Worst performance on {w['feature']} in range {w['bin']} "
            f"({w['n_samples']} samples). Consider collecting more data in this range "
            f"or engineering features that capture this pattern."
        )

    # Check if errors correlate with prediction magnitude
    if task == "regression":
        pred_error_corr = abs(float(np.corrcoef(y_pred, errors)[0, 1]))
        if pred_error_corr > 0.5:
            recommendations.append(
                f"Errors increase with prediction magnitude (r={pred_error_corr:.2f}). "
                f"Consider log-transforming the target or using a model that handles "
                f"heteroscedasticity (RobustRegression, QuantileRegression)."
            )

    return {
        "overall": overall,
        "worst_segments": worst,
        "high_error_samples": high_error_samples,
        "recommendations": recommendations,
        "total_segments_analyzed": len(segments),
    }
