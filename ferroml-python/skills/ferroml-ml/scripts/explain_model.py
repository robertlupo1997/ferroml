"""Model interpretability via permutation importance and effect analysis.

Usage: Claude adapts this to explain any fitted model's predictions.
Output: Feature importances, effect directions, and plain-language explanations.
"""
from __future__ import annotations

import numpy as np


def explain(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 10,
    n_repeats: int = 5,
    seed: int = 42,
) -> dict:
    """Explain a fitted model via permutation importance.

    Parameters
    ----------
    model : fitted model
        Must have a .predict() method.
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        True target values.
    feature_names : list or None
        Human-readable feature names. Defaults to feature_0, feature_1, ...
    top_k : int
        Number of top features to highlight.
    n_repeats : int
        Number of permutation shuffles per feature.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with feature_importances, top_features, plain_language_explanations,
    method_used, baseline_score
    """
    from ferroml import metrics

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    rng = np.random.RandomState(seed)

    # Detect task type from target
    n_unique = len(np.unique(y))
    is_classification = n_unique <= 20

    # Baseline score
    preds_baseline = model.predict(X)
    if is_classification:
        baseline_score = float(metrics.accuracy_score(y, preds_baseline))
        metric_name = "accuracy"
    else:
        baseline_score = float(metrics.r2_score(y, preds_baseline))
        metric_name = "r2"

    # Permutation importance: shuffle each feature, measure performance drop
    importances = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            preds_perm = model.predict(X_perm)
            if is_classification:
                score_perm = float(metrics.accuracy_score(y, preds_perm))
            else:
                score_perm = float(metrics.r2_score(y, preds_perm))
            drops.append(baseline_score - score_perm)
        importances.append({
            "feature": feature_names[j],
            "importance_mean": round(float(np.mean(drops)), 6),
            "importance_std": round(float(np.std(drops)), 6),
        })

    # Sort by importance (largest drop = most important)
    importances.sort(key=lambda x: x["importance_mean"], reverse=True)

    # Effect direction for top features: correlation between feature and target
    top_features = []
    plain_language = []
    for entry in importances[:top_k]:
        fname = entry["feature"]
        j = feature_names.index(fname)
        col = X[:, j]

        # Compute correlation for direction
        if np.std(col) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(col, y)[0, 1])
            direction = "positive" if corr > 0 else "negative"
        else:
            corr = 0.0
            direction = "neutral"

        # Estimate marginal effect (simple linear slope)
        if np.std(col) > 0:
            slope = float(np.cov(col, y)[0, 1] / np.var(col))
        else:
            slope = 0.0

        feature_info = {
            "feature": fname,
            "importance": entry["importance_mean"],
            "importance_std": entry["importance_std"],
            "direction": direction,
            "correlation": round(corr, 4),
            "marginal_effect": round(slope, 6),
        }
        top_features.append(feature_info)

        # Plain-language explanation
        imp_pct = round(entry["importance_mean"] * 100, 1)
        if not is_classification:
            unit_change = abs(slope) * 100  # per 100-unit increase
            if direction == "positive":
                explanation = (
                    f"Feature '{fname}' has high importance ({imp_pct}% {metric_name} drop when shuffled). "
                    f"Each 100-unit increase adds ~{round(unit_change, 1)} to the target."
                )
            elif direction == "negative":
                explanation = (
                    f"Feature '{fname}' has high importance ({imp_pct}% {metric_name} drop when shuffled). "
                    f"Each 100-unit increase decreases the target by ~{round(unit_change, 1)}."
                )
            else:
                explanation = f"Feature '{fname}' has high importance ({imp_pct}% {metric_name} drop) but unclear direction."
        else:
            if direction == "positive":
                explanation = f"Feature '{fname}' is important ({imp_pct}% accuracy drop). Higher values push toward the positive class."
            elif direction == "negative":
                explanation = f"Feature '{fname}' is important ({imp_pct}% accuracy drop). Higher values push toward the negative class."
            else:
                explanation = f"Feature '{fname}' is important ({imp_pct}% accuracy drop) but has no clear directional effect."

        plain_language.append(explanation)

    return {
        "baseline_score": round(baseline_score, 6),
        "metric": metric_name,
        "method_used": "permutation_importance",
        "n_repeats": n_repeats,
        "feature_importances": importances,
        "top_features": top_features,
        "plain_language_explanations": plain_language,
    }


def print_explanation(result: dict) -> None:
    """Print a human-readable explanation."""
    print(f"Baseline {result['metric']}: {result['baseline_score']}")
    print(f"Method: {result['method_used']} ({result['n_repeats']} repeats)")
    print()

    print("Top features:")
    for feat in result["top_features"]:
        print(f"  {feat['feature']}: importance={feat['importance']:.4f} "
              f"(+/-{feat['importance_std']:.4f}), direction={feat['direction']}, "
              f"corr={feat['correlation']:.3f}")
    print()

    print("Plain-language explanations:")
    for explanation in result["plain_language_explanations"]:
        print(f"  - {explanation}")
