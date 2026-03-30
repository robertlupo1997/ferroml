"""Find the optimal classification threshold for probability-based models.

Usage: Claude adapts this after fitting a probabilistic classifier.
Output: Optimal threshold with metric comparison vs default 0.5.
"""
from __future__ import annotations

import numpy as np


def optimize(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1",
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> dict:
    """Find the optimal classification threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    strategy : str
        One of "f1", "balanced", "cost", "youden".
    cost_fp : float
        Cost of a false positive (only used when strategy="cost").
    cost_fn : float
        Cost of a false negative (only used when strategy="cost").

    Returns
    -------
    dict with keys: optimal_threshold, metric_at_threshold, strategy,
    threshold_curve, comparison_vs_default
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    curve: list[dict] = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(np.float64)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)

        if strategy == "f1":
            metric_val = f1
        elif strategy == "balanced":
            metric_val = 1.0 - abs(prec - rec)  # Maximize when prec == rec
        elif strategy == "cost":
            # Minimize cost: cost_fp * FP + cost_fn * FN
            cost = cost_fp * fp + cost_fn * fn
            metric_val = -cost  # Negate so we can maximize
        elif strategy == "youden":
            # Youden's J = sensitivity + specificity - 1
            metric_val = rec + spec - 1.0
        else:
            metric_val = f1

        curve.append({
            "threshold": round(float(t), 4),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "specificity": round(spec, 6),
            "f1": round(f1, 6),
            "accuracy": round(acc, 6),
            "metric_value": round(float(metric_val), 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    # Find optimal
    best_entry = max(curve, key=lambda c: c["metric_value"])
    optimal_threshold = best_entry["threshold"]

    # Default threshold comparison
    default_entry = min(curve, key=lambda c: abs(c["threshold"] - 0.5))

    comparison = {
        "default_threshold": 0.5,
        "default_f1": default_entry["f1"],
        "default_accuracy": default_entry["accuracy"],
        "default_precision": default_entry["precision"],
        "default_recall": default_entry["recall"],
        "optimal_threshold": optimal_threshold,
        "optimal_f1": best_entry["f1"],
        "optimal_accuracy": best_entry["accuracy"],
        "optimal_precision": best_entry["precision"],
        "optimal_recall": best_entry["recall"],
        "f1_improvement": round(best_entry["f1"] - default_entry["f1"], 6),
        "accuracy_improvement": round(best_entry["accuracy"] - default_entry["accuracy"], 6),
    }

    return {
        "optimal_threshold": optimal_threshold,
        "metric_at_threshold": best_entry["metric_value"],
        "strategy": strategy,
        "threshold_curve": curve,
        "comparison_vs_default": comparison,
        "best_entry": best_entry,
    }


def print_summary(result: dict) -> None:
    """Print a human-readable threshold optimization summary."""
    comp = result["comparison_vs_default"]
    print(f"Strategy: {result['strategy']}")
    print(f"Optimal threshold: {result['optimal_threshold']}")
    print()

    print(f"{'Metric':<12} {'Default (0.5)':<15} {'Optimal':<15} {'Change':<10}")
    print("-" * 52)
    for metric in ["f1", "accuracy", "precision", "recall"]:
        d = comp[f"default_{metric}"]
        o = comp[f"optimal_{metric}"]
        diff = o - d
        sign = "+" if diff >= 0 else ""
        print(f"{metric:<12} {d:<15.4f} {o:<15.4f} {sign}{diff:.4f}")
    print()

    best = result["best_entry"]
    print(f"At optimal threshold ({result['optimal_threshold']}):")
    print(f"  TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, TN={best['tn']}")
