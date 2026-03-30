"""Business cost modeling for classification decisions.

Usage: Claude adapts this to evaluate prediction costs and find optimal thresholds.
Output: Total cost, cost-per-prediction, optimal threshold, and savings analysis.
"""
from __future__ import annotations

import numpy as np


def analyze(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    cost_matrix: dict[str, float] | None = None,
) -> dict:
    """Compute business costs of classification predictions.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels.
    y_proba : np.ndarray or None
        Predicted probabilities for the positive class. If provided,
        optimal threshold is computed.
    cost_matrix : dict or None
        Costs for each outcome: {"tp": 0, "fp": 1, "fn": 10, "tn": 0}.
        Defaults to FN costing 10x FP.

    Returns
    -------
    dict with total_cost, cost_per_prediction, optimal_threshold (if proba),
    savings_vs_baseline, cost_breakdown, threshold_analysis
    """
    from ferroml import metrics

    if cost_matrix is None:
        cost_matrix = {"tp": 0.0, "fp": 1.0, "fn": 10.0, "tn": 0.0}

    n = len(y_true)

    # Compute confusion matrix components
    tp, fp, fn, tn = _confusion_components(y_true, y_pred)

    # Cost at current threshold
    total_cost = (
        tp * cost_matrix["tp"]
        + fp * cost_matrix["fp"]
        + fn * cost_matrix["fn"]
        + tn * cost_matrix["tn"]
    )
    cost_per_pred = round(total_cost / max(n, 1), 4)

    cost_breakdown = {
        "true_positives": {"count": tp, "cost": round(tp * cost_matrix["tp"], 2)},
        "false_positives": {"count": fp, "cost": round(fp * cost_matrix["fp"], 2)},
        "false_negatives": {"count": fn, "cost": round(fn * cost_matrix["fn"], 2)},
        "true_negatives": {"count": tn, "cost": round(tn * cost_matrix["tn"], 2)},
    }

    # Standard metrics for reference
    accuracy = float(metrics.accuracy_score(y_true, y_pred))

    # Baseline comparisons
    # Baseline 1: predict all negative
    all_neg_cost = (
        int(np.sum(y_true == 1)) * cost_matrix["fn"]
        + int(np.sum(y_true == 0)) * cost_matrix["tn"]
    )
    # Baseline 2: predict all positive
    all_pos_cost = (
        int(np.sum(y_true == 1)) * cost_matrix["tp"]
        + int(np.sum(y_true == 0)) * cost_matrix["fp"]
    )

    best_baseline = min(all_neg_cost, all_pos_cost)
    best_baseline_name = "all_negative" if all_neg_cost <= all_pos_cost else "all_positive"
    savings_vs_baseline = round(best_baseline - total_cost, 2)

    result = {
        "total_cost": round(total_cost, 2),
        "cost_per_prediction": cost_per_pred,
        "cost_matrix": cost_matrix,
        "cost_breakdown": cost_breakdown,
        "accuracy": round(accuracy, 4),
        "baseline_costs": {
            "all_negative": round(all_neg_cost, 2),
            "all_positive": round(all_pos_cost, 2),
        },
        "best_baseline": best_baseline_name,
        "savings_vs_baseline": savings_vs_baseline,
        "n_predictions": n,
    }

    # Threshold optimization if probabilities provided
    if y_proba is not None:
        threshold_analysis = _optimize_threshold(y_true, y_proba, cost_matrix)
        result["optimal_threshold"] = threshold_analysis["optimal_threshold"]
        result["optimal_cost"] = threshold_analysis["optimal_cost"]
        result["threshold_analysis"] = threshold_analysis["thresholds"]
        result["savings_with_optimal"] = round(total_cost - threshold_analysis["optimal_cost"], 2)

    return result


def _confusion_components(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    """Compute TP, FP, FN, TN counts."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def _optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_matrix: dict[str, float],
) -> dict:
    """Find the threshold that minimizes total cost."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_cost = float("inf")
    best_threshold = 0.5
    threshold_results = []

    for thresh in thresholds:
        y_pred_t = (y_proba >= thresh).astype(float)
        tp, fp, fn, tn = _confusion_components(y_true, y_pred_t)
        cost = (
            tp * cost_matrix["tp"]
            + fp * cost_matrix["fp"]
            + fn * cost_matrix["fn"]
            + tn * cost_matrix["tn"]
        )
        threshold_results.append({
            "threshold": round(float(thresh), 2),
            "cost": round(cost, 2),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(thresh)

    # Return a subset of thresholds for readability
    step = max(1, len(threshold_results) // 10)
    sampled_thresholds = threshold_results[::step]

    return {
        "optimal_threshold": round(best_threshold, 2),
        "optimal_cost": round(best_cost, 2),
        "thresholds": sampled_thresholds,
    }


def print_analysis(result: dict) -> None:
    """Print a human-readable cost analysis."""
    print(f"Total cost: ${result['total_cost']:.2f} ({result['n_predictions']} predictions)")
    print(f"Cost per prediction: ${result['cost_per_prediction']:.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print()

    print("Cost breakdown:")
    for outcome, info in result["cost_breakdown"].items():
        print(f"  {outcome}: {info['count']} x ${result['cost_matrix'][outcome.split('_')[0][0] + outcome.split('_')[1][0]]:.2f} = ${info['cost']:.2f}")
    print()

    print("Baselines:")
    for name, cost in result["baseline_costs"].items():
        print(f"  {name}: ${cost:.2f}")
    print(f"Savings vs best baseline ({result['best_baseline']}): ${result['savings_vs_baseline']:.2f}")

    if "optimal_threshold" in result:
        print()
        print(f"Optimal threshold: {result['optimal_threshold']}")
        print(f"Optimal cost: ${result['optimal_cost']:.2f}")
        print(f"Savings with optimal threshold: ${result['savings_with_optimal']:.2f}")
