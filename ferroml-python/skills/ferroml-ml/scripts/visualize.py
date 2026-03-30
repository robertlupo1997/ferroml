"""Generate text-based visualizations and descriptions for ML results.

Usage: Claude calls these functions to describe model outputs in the terminal.
Output: Structured descriptions with statistics, trends, and ASCII summaries.
Note: Terminal-based — no matplotlib, no image files.
"""
from __future__ import annotations

import numpy as np


def describe_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Analyze residuals and detect patterns.

    Parameters
    ----------
    y_true : np.ndarray
        True values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict with keys: statistics, distribution, pattern_flags, histogram
    """
    residuals = y_true - y_pred
    n = len(residuals)

    stats = {
        "mean": round(float(np.mean(residuals)), 6),
        "median": round(float(np.median(residuals)), 6),
        "std": round(float(np.std(residuals)), 6),
        "min": round(float(np.min(residuals)), 6),
        "max": round(float(np.max(residuals)), 6),
        "mae": round(float(np.mean(np.abs(residuals))), 6),
        "rmse": round(float(np.sqrt(np.mean(residuals ** 2))), 6),
    }

    # Distribution analysis
    skewness = _skewness(residuals)
    kurtosis = _kurtosis(residuals)
    distribution = {
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "approximately_normal": abs(skewness) < 0.5 and abs(kurtosis - 3) < 1,
    }

    # Pattern detection
    flags: list[str] = []
    if abs(stats["mean"]) > 0.1 * stats["std"] and stats["std"] > 0:
        flags.append(f"Non-zero mean residual ({stats['mean']:.4f}) — model has systematic bias.")
    if abs(skewness) > 1.0:
        direction = "positive" if skewness > 0 else "negative"
        flags.append(f"Skewed residuals ({direction}, skew={skewness:.2f}) — errors are asymmetric.")
    if kurtosis > 5:
        flags.append(f"Heavy-tailed residuals (kurtosis={kurtosis:.2f}) — occasional large errors.")

    # Check for heteroscedasticity (residuals correlated with predictions)
    if len(y_pred) > 10:
        corr = np.corrcoef(np.abs(residuals), y_pred)[0, 1]
        if not np.isnan(corr) and abs(corr) > 0.3:
            flags.append(f"Heteroscedasticity detected (|corr(|residual|, pred)|={abs(corr):.3f}) — error variance depends on prediction magnitude.")

    if not flags:
        flags.append("Residuals look healthy — no major patterns detected.")

    # ASCII histogram
    histogram = _ascii_histogram(residuals, bins=20, width=40, label="Residuals")

    return {
        "statistics": stats,
        "distribution": distribution,
        "pattern_flags": flags,
        "histogram": histogram,
        "n_samples": n,
    }


def describe_feature_importance(
    importances: np.ndarray | list[float],
    feature_names: list[str],
) -> dict:
    """Rank features by importance with visual bars.

    Parameters
    ----------
    importances : array-like
        Importance scores (higher = more important).
    feature_names : list[str]
        Feature names.

    Returns
    -------
    dict with keys: ranked_features, bar_chart, top_n, cumulative_importance
    """
    importances = np.array(importances, dtype=np.float64)
    order = np.argsort(-importances)

    total = float(np.sum(np.abs(importances))) if np.sum(np.abs(importances)) > 0 else 1.0
    max_imp = float(np.max(np.abs(importances))) if len(importances) > 0 else 1.0

    ranked: list[dict] = []
    cumulative = 0.0
    bar_lines: list[str] = []

    max_name_len = max(len(n) for n in feature_names) if feature_names else 10
    bar_width = 30

    for i, idx in enumerate(order):
        imp = float(importances[idx])
        name = feature_names[idx]
        pct = abs(imp) / total * 100
        cumulative += pct

        ranked.append({
            "rank": i + 1,
            "feature": name,
            "importance": round(imp, 6),
            "percentage": round(pct, 2),
            "cumulative_percentage": round(cumulative, 2),
        })

        # Bar chart line
        bar_len = int(abs(imp) / max(max_imp, 1e-12) * bar_width)
        bar = "#" * bar_len
        bar_lines.append(f"  {name:<{max_name_len}} | {bar:<{bar_width}} {pct:5.1f}%")

    # Find N features for 90% cumulative importance
    top_90_n = 0
    for r in ranked:
        top_90_n += 1
        if r["cumulative_percentage"] >= 90:
            break

    return {
        "ranked_features": ranked,
        "bar_chart": "\n".join(bar_lines),
        "top_n_for_90_pct": top_90_n,
        "total_features": len(feature_names),
    }


def describe_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Format confusion matrix with per-class metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    dict with keys: matrix, classes, per_class_metrics, formatted_table,
    overall_accuracy
    """
    from ferroml.metrics import accuracy_score

    classes = sorted(set(np.unique(y_true).tolist() + np.unique(y_pred).tolist()))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[class_to_idx[t], class_to_idx[p]] += 1

    # Per-class metrics
    per_class: list[dict] = []
    for i, cls in enumerate(classes):
        tp = matrix[i, i]
        fp = int(np.sum(matrix[:, i]) - tp)
        fn = int(np.sum(matrix[i, :]) - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        support = int(np.sum(matrix[i, :]))

        per_class.append({
            "class": cls,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    # Formatted table
    overall_acc = float(accuracy_score(y_true, y_pred))

    # Build text table
    class_strs = [str(c) for c in classes]
    col_width = max(8, max(len(s) for s in class_strs) + 2)

    header = f"{'Actual/Pred':<{col_width}}" + "".join(f"{s:>{col_width}}" for s in class_strs)
    lines = [header, "-" * len(header)]
    for i, cls in enumerate(class_strs):
        row = f"{cls:<{col_width}}"
        for j in range(n_classes):
            row += f"{matrix[i, j]:>{col_width}}"
        lines.append(row)

    lines.append("")
    lines.append(f"{'Class':<{col_width}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("-" * (col_width + 40))
    for pc in per_class:
        lines.append(
            f"{str(pc['class']):<{col_width}} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
            f"{pc['f1']:>10.4f} {pc['support']:>10}"
        )

    return {
        "matrix": matrix.tolist(),
        "classes": classes,
        "per_class_metrics": per_class,
        "formatted_table": "\n".join(lines),
        "overall_accuracy": round(overall_acc, 6),
        "n_samples": len(y_true),
    }


def describe_learning_curve(
    train_sizes: list[float],
    train_scores: list[float],
    test_scores: list[float],
) -> dict:
    """Describe learning curve trends in text.

    Parameters
    ----------
    train_sizes : list[float]
        Fractions or absolute sizes.
    train_scores : list[float]
        Mean training scores at each size.
    test_scores : list[float]
        Mean test scores at each size.

    Returns
    -------
    dict with keys: trend_description, gap_analysis, ascii_chart, recommendations
    """
    n = len(train_sizes)
    if n == 0:
        return {"trend_description": "No data points.", "gap_analysis": "", "ascii_chart": "", "recommendations": []}

    final_train = train_scores[-1]
    final_test = test_scores[-1]
    gap = final_train - final_test

    # Trend analysis
    trends: list[str] = []
    if n >= 2:
        test_improving = test_scores[-1] > test_scores[0]
        train_stable = abs(train_scores[-1] - train_scores[0]) < 0.05
        if test_improving:
            trends.append("Test score improves with more data.")
        else:
            trends.append("Test score is flat or declining — adding data may not help.")
        if train_stable:
            trends.append("Training score is stable across data sizes.")
        else:
            trends.append(f"Training score changed from {train_scores[0]:.3f} to {train_scores[-1]:.3f}.")

    gap_desc = f"Final gap: train={final_train:.4f}, test={final_test:.4f}, delta={gap:.4f}."
    if gap > 0.15:
        gap_desc += " Large gap indicates overfitting."
    elif gap < 0.05:
        gap_desc += " Small gap indicates good generalization."

    # ASCII chart
    chart_lines: list[str] = []
    chart_height = 10
    all_scores = train_scores + test_scores
    y_min = max(0.0, min(all_scores) - 0.05)
    y_max = min(1.0, max(all_scores) + 0.05)
    y_range = max(y_max - y_min, 0.01)

    for row in range(chart_height, -1, -1):
        level = y_min + (row / chart_height) * y_range
        line = f"  {level:.2f} |"
        for i in range(n):
            tr_pos = int((train_scores[i] - y_min) / y_range * chart_height)
            te_pos = int((test_scores[i] - y_min) / y_range * chart_height)
            if tr_pos == row and te_pos == row:
                line += " *"
            elif tr_pos == row:
                line += " T"
            elif te_pos == row:
                line += " V"
            else:
                line += "  "
        chart_lines.append(line)
    chart_lines.append(f"       +{'--' * n}")
    chart_lines.append("       T=train, V=validation, *=overlap")

    recommendations: list[str] = []
    if gap > 0.15:
        recommendations.append("Reduce model complexity or increase regularization.")
    if n >= 3 and test_scores[-1] > test_scores[-2] + 0.005:
        recommendations.append("Test score still improving — more data may help.")
    if final_test < 0.5:
        recommendations.append("Overall performance is low — try a different model family.")

    return {
        "trend_description": " ".join(trends),
        "gap_analysis": gap_desc,
        "ascii_chart": "\n".join(chart_lines),
        "recommendations": recommendations,
        "final_train_score": round(final_train, 6),
        "final_test_score": round(final_test, 6),
        "final_gap": round(gap, 6),
    }


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3) * n / ((n - 1) * (n - 2)) * n)


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4)) - 3.0


def _ascii_histogram(values: np.ndarray, bins: int = 20, width: int = 40, label: str = "") -> str:
    """Create an ASCII histogram."""
    counts, edges = np.histogram(values, bins=bins)
    max_count = int(np.max(counts)) if len(counts) > 0 else 1

    lines = []
    if label:
        lines.append(f"  {label} distribution:")
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(counts[i] / max(max_count, 1) * width)
        bar = "#" * bar_len
        lines.append(f"  [{lo:+8.3f}, {hi:+8.3f}) | {bar:<{width}} {counts[i]}")
    return "\n".join(lines)


def print_residuals(result: dict) -> None:
    """Print residual analysis."""
    stats = result["statistics"]
    print(f"Residual Analysis (n={result['n_samples']})")
    print(f"  Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, Std: {stats['std']:.4f}")
    print(f"  MAE: {stats['mae']:.4f}, RMSE: {stats['rmse']:.4f}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print()
    print(result["histogram"])
    print()
    print("Flags:")
    for flag in result["pattern_flags"]:
        print(f"  - {flag}")


def print_importance(result: dict) -> None:
    """Print feature importance ranking."""
    print(f"Feature Importance ({result['total_features']} features, top {result['top_n_for_90_pct']} cover 90%)")
    print()
    print(result["bar_chart"])


def print_confusion(result: dict) -> None:
    """Print confusion matrix."""
    print(f"Confusion Matrix (n={result['n_samples']}, accuracy={result['overall_accuracy']:.4f})")
    print()
    print(result["formatted_table"])
