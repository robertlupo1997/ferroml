"""Bias and fairness checking for classification models.

Usage: Claude adapts this to audit a model's predictions for fairness
across a sensitive attribute (gender, race, age group, etc.).
Output: Per-group metrics, demographic parity, equalized odds, 80% rule check.
"""
from __future__ import annotations

import numpy as np


def audit(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature_idx: int,
    feature_names: list[str] | None = None,
) -> dict:
    """Audit model predictions for fairness across a sensitive attribute.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y_true : np.ndarray
        True labels (binary: 0 or 1).
    y_pred : np.ndarray
        Predicted labels (binary: 0 or 1).
    sensitive_feature_idx : int
        Column index in X for the sensitive attribute.
    feature_names : list[str] or None
        Feature names. Used to label the sensitive attribute.

    Returns
    -------
    dict with keys: group_metrics, demographic_parity_ratio,
    equalized_odds_diff, passes_80_percent_rule, sensitive_feature,
    recommendations, summary
    """
    sensitive_vals = X[:, sensitive_feature_idx]
    groups = sorted(set(sensitive_vals.tolist()))
    sensitive_name = (
        feature_names[sensitive_feature_idx]
        if feature_names and sensitive_feature_idx < len(feature_names)
        else f"feature_{sensitive_feature_idx}"
    )

    # Per-group metrics
    group_metrics: list[dict] = []
    positive_rates: list[float] = []
    tprs: list[float] = []
    fprs: list[float] = []

    for g in groups:
        mask = sensitive_vals == g
        n_group = int(np.sum(mask))
        if n_group == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        tp = int(np.sum((y_p == 1) & (y_t == 1)))
        fp = int(np.sum((y_p == 1) & (y_t == 0)))
        fn = int(np.sum((y_p == 0) & (y_t == 1)))
        tn = int(np.sum((y_p == 0) & (y_t == 0)))

        accuracy = (tp + tn) / max(n_group, 1)
        positive_rate = (tp + fp) / max(n_group, 1)
        tpr = tp / max(tp + fn, 1)  # Sensitivity / recall
        fpr = fp / max(fp + tn, 1)  # False positive rate
        precision = tp / max(tp + fp, 1)

        positive_rates.append(positive_rate)
        tprs.append(tpr)
        fprs.append(fpr)

        group_metrics.append({
            "group": g,
            "n_samples": n_group,
            "accuracy": round(accuracy, 6),
            "positive_rate": round(positive_rate, 6),
            "true_positive_rate": round(tpr, 6),
            "false_positive_rate": round(fpr, 6),
            "precision": round(precision, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    # --- Fairness metrics ---

    # Demographic parity ratio: min(positive_rate) / max(positive_rate)
    if positive_rates and max(positive_rates) > 0:
        dp_ratio = round(min(positive_rates) / max(positive_rates), 6)
    else:
        dp_ratio = 1.0

    # Equalized odds difference: max gap in TPR + max gap in FPR
    if len(tprs) >= 2:
        tpr_gap = max(tprs) - min(tprs)
        fpr_gap = max(fprs) - min(fprs)
        eq_odds_diff = round(tpr_gap + fpr_gap, 6)
    else:
        eq_odds_diff = 0.0

    # 80% rule (four-fifths rule)
    passes_80 = dp_ratio >= 0.8

    # --- Recommendations ---
    recommendations: list[str] = []

    if not passes_80:
        recommendations.append(
            f"FAILS 80% rule (demographic parity ratio={dp_ratio:.3f} < 0.80). "
            f"The model's positive prediction rate differs significantly across groups."
        )
        # Find most/least favored groups
        if len(group_metrics) >= 2:
            sorted_by_rate = sorted(group_metrics, key=lambda g: g["positive_rate"])
            least = sorted_by_rate[0]
            most = sorted_by_rate[-1]
            recommendations.append(
                f"Most favored group: {most['group']} (positive rate={most['positive_rate']:.3f}). "
                f"Least favored: {least['group']} (positive rate={least['positive_rate']:.3f})."
            )
    else:
        recommendations.append(f"Passes 80% rule (demographic parity ratio={dp_ratio:.3f} >= 0.80).")

    if eq_odds_diff > 0.1:
        recommendations.append(
            f"Equalized odds difference is high ({eq_odds_diff:.3f}). "
            f"The model makes different types of errors for different groups."
        )
        if len(tprs) >= 2 and max(tprs) - min(tprs) > 0.1:
            recommendations.append(
                f"TPR gap: {max(tprs) - min(tprs):.3f} — some groups get fewer true positives."
            )
        if len(fprs) >= 2 and max(fprs) - min(fprs) > 0.1:
            recommendations.append(
                f"FPR gap: {max(fprs) - min(fprs):.3f} — some groups get more false positives."
            )
    else:
        recommendations.append(f"Equalized odds difference is acceptable ({eq_odds_diff:.3f} <= 0.10).")

    # Mitigation suggestions
    if not passes_80 or eq_odds_diff > 0.1:
        recommendations.append("Mitigation strategies:")
        recommendations.append("  - Pre-processing: resample or reweight training data.")
        recommendations.append("  - In-processing: add fairness constraints during training.")
        recommendations.append("  - Post-processing: adjust thresholds per group.")
        recommendations.append("  - Review: check if the sensitive feature should be excluded.")

    # Overall accuracy disparity
    if len(group_metrics) >= 2:
        accs = [g["accuracy"] for g in group_metrics]
        if max(accs) - min(accs) > 0.1:
            recommendations.append(
                f"Accuracy disparity: {max(accs):.3f} vs {min(accs):.3f} (gap={max(accs) - min(accs):.3f})."
            )

    # Summary string
    summary = (
        f"Fairness audit on '{sensitive_name}' ({len(groups)} groups, {len(y_true)} samples): "
        f"DP ratio={dp_ratio:.3f}, EO diff={eq_odds_diff:.3f}, "
        f"80% rule={'PASS' if passes_80 else 'FAIL'}."
    )

    return {
        "sensitive_feature": sensitive_name,
        "sensitive_feature_idx": sensitive_feature_idx,
        "n_groups": len(groups),
        "group_metrics": group_metrics,
        "demographic_parity_ratio": dp_ratio,
        "equalized_odds_diff": eq_odds_diff,
        "passes_80_percent_rule": passes_80,
        "recommendations": recommendations,
        "summary": summary,
    }


def print_summary(result: dict) -> None:
    """Print a human-readable fairness audit."""
    print(f"Fairness Audit: {result['sensitive_feature']} ({result['n_groups']} groups)")
    print()

    # Group metrics table
    col_w = 12
    header = (
        f"{'Group':<{col_w}} {'N':>{col_w}} {'Accuracy':>{col_w}} {'Pos Rate':>{col_w}} "
        f"{'TPR':>{col_w}} {'FPR':>{col_w}} {'Precision':>{col_w}}"
    )
    print(header)
    print("-" * len(header))
    for g in result["group_metrics"]:
        print(
            f"{str(g['group']):<{col_w}} {g['n_samples']:>{col_w}} {g['accuracy']:>{col_w}.4f} "
            f"{g['positive_rate']:>{col_w}.4f} {g['true_positive_rate']:>{col_w}.4f} "
            f"{g['false_positive_rate']:>{col_w}.4f} {g['precision']:>{col_w}.4f}"
        )
    print()

    print(f"Demographic Parity Ratio: {result['demographic_parity_ratio']:.4f}")
    print(f"Equalized Odds Difference: {result['equalized_odds_diff']:.4f}")
    print(f"80% Rule: {'PASS' if result['passes_80_percent_rule'] else 'FAIL'}")
    print()

    print("Recommendations:")
    for rec in result["recommendations"]:
        print(f"  {rec}")
