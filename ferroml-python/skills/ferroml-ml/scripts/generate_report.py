"""Generate a plain-language ML report suitable for any audience.

Usage: Claude runs this after training/evaluation to produce a readable summary.
Adapts language based on technical level (non-technical, developer, ML engineer).
"""
from __future__ import annotations


def generate(
    pipeline_result: dict,
    audience: str = "non-technical",
) -> str:
    """Generate a report from pipeline results.

    Parameters
    ----------
    pipeline_result : dict
        Output from full_pipeline.run_pipeline() or similar.
    audience : str
        "non-technical", "developer", or "ml-engineer"

    Returns
    -------
    str : Formatted report text
    """
    task = pipeline_result.get("task", "unknown")
    model_name = pipeline_result.get("model_name", "Unknown Model")
    test_metrics = pipeline_result.get("test_metrics", {})
    train_metrics = pipeline_result.get("train_metrics", {})
    diagnostics = pipeline_result.get("diagnostics", {})
    n_train = pipeline_result.get("n_train", 0)
    n_test = pipeline_result.get("n_test", 0)
    n_features = pipeline_result.get("n_features", 0)
    warning = pipeline_result.get("warning")

    lines = []

    if audience == "non-technical":
        lines.append("# Your Model Results\n")
        lines.append(_non_technical_summary(task, model_name, test_metrics, n_train + n_test))
        if warning:
            lines.append(f"\n**Something to watch out for:** {_simplify_warning(warning)}\n")
        if diagnostics.get("summary"):
            lines.append("\n## Detailed Statistics\n")
            lines.append("*(For your data team — you can skip this)*\n")
            lines.append(f"```\n{diagnostics['summary']}\n```\n")

    elif audience == "developer":
        lines.append(f"# ML Report: {model_name}\n")
        lines.append(f"**Task:** {task} | **Samples:** {n_train} train / {n_test} test | **Features:** {n_features}\n")
        lines.append("## Test Set Performance\n")
        for metric, value in test_metrics.items():
            lines.append(f"- **{metric}:** {value}")
        if warning:
            lines.append(f"\n> Warning: {warning}\n")
        if diagnostics.get("normality"):
            norm = diagnostics["normality"]
            lines.append(f"\n## Residual Analysis")
            lines.append(f"- Durbin-Watson: {diagnostics.get('durbin_watson', 'N/A')}")
            lines.append(f"- Normality: {'PASS' if norm.get('is_normal') else 'FAIL'} (p={norm.get('p_value', 'N/A'):.4f})")

    else:  # ml-engineer
        lines.append(f"# {model_name} — {task}\n")
        lines.append(f"n={n_train+n_test}, features={n_features}, split={n_train}/{n_test}\n")
        lines.append("## Metrics\n")
        lines.append("| Set | " + " | ".join(test_metrics.keys()) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(test_metrics)) + " |")
        lines.append("| Train | " + " | ".join(str(v) for v in train_metrics.values()) + " |")
        lines.append("| Test  | " + " | ".join(str(v) for v in test_metrics.values()) + " |")
        if warning:
            lines.append(f"\n**{warning}**")
        if diagnostics:
            lines.append("\n## Diagnostics")
            for k, v in diagnostics.items():
                if k != "summary":
                    lines.append(f"- {k}: {v}")
            if diagnostics.get("summary"):
                lines.append(f"\n```\n{diagnostics['summary']}\n```")

    return "\n".join(lines)


def _non_technical_summary(task: str, model_name: str, metrics: dict, total_samples: int) -> str:
    """Generate a plain-English summary."""
    lines = []
    if task == "regression":
        r2 = metrics.get("r2", 0)
        rmse = metrics.get("rmse", 0)
        pct = round(r2 * 100, 1)
        lines.append(f"I trained a **{_friendly_name(model_name)}** on your {total_samples} data points.\n")
        if pct >= 90:
            lines.append(f"**Great news!** The model explains **{pct}%** of the variation in your data. That's excellent.")
        elif pct >= 70:
            lines.append(f"The model explains **{pct}%** of the variation. That's good — it captures the main patterns.")
        elif pct >= 50:
            lines.append(f"The model explains **{pct}%** of the variation. It's picking up some patterns, but there's room to improve.")
        else:
            lines.append(f"The model explains **{pct}%** of the variation. This suggests the relationship is complex — we may need a more advanced model or better features.")
        lines.append(f"\nOn average, predictions are off by about **{rmse:.2f}** (the typical error).")
    else:
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)
        pct = round(acc * 100, 1)
        lines.append(f"I trained a **{_friendly_name(model_name)}** on your {total_samples} data points.\n")
        if pct >= 95:
            lines.append(f"**Excellent!** The model correctly classifies **{pct}%** of cases.")
        elif pct >= 85:
            lines.append(f"The model correctly classifies **{pct}%** of cases. That's solid performance.")
        elif pct >= 70:
            lines.append(f"The model correctly classifies **{pct}%** of cases. There's room to improve.")
        else:
            lines.append(f"The model correctly classifies **{pct}%** of cases. We should try different approaches to improve this.")
    return "\n".join(lines)


def _friendly_name(model_name: str) -> str:
    """Convert class name to friendly name."""
    friendly = {
        "LinearRegression": "Linear Regression model",
        "LogisticRegression": "Logistic Regression model",
        "RidgeRegression": "Ridge Regression model",
        "RandomForestClassifier": "Random Forest model",
        "RandomForestRegressor": "Random Forest model",
        "GradientBoostingClassifier": "Gradient Boosting model",
        "GradientBoostingRegressor": "Gradient Boosting model",
        "HistGradientBoostingClassifier": "Histogram Gradient Boosting model",
        "HistGradientBoostingRegressor": "Histogram Gradient Boosting model",
        "GaussianNB": "Naive Bayes model",
        "SVC": "Support Vector Machine",
        "SVR": "Support Vector Machine",
        "KNeighborsClassifier": "K-Nearest Neighbors model",
        "MLPClassifier": "Neural Network",
        "MLPRegressor": "Neural Network",
    }
    return friendly.get(model_name, model_name)


def _simplify_warning(warning: str) -> str:
    """Simplify a technical warning for non-technical users."""
    if "overfitting" in warning.lower():
        return ("The model may be memorizing the training data rather than learning "
                "general patterns. I'll try a simpler model or add regularization to fix this.")
    return warning
