"""Counterfactual explanations — "what would need to change?" for predictions.

Usage: Claude adapts this to explain what changes would flip a model's prediction.
Output: Minimum feature changes needed to achieve a desired prediction.
"""
from __future__ import annotations

import numpy as np


def explain(
    model: object,
    X_instance: np.ndarray,
    desired_class: int | float = 1,
    feature_names: list[str] | None = None,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    max_changes: int = 3,
    n_steps: int = 50,
) -> dict:
    """Find minimum feature changes to flip a prediction.

    Uses a greedy approach: first tries single-feature changes, then pairs,
    up to max_changes simultaneous features.

    Parameters
    ----------
    model : fitted model
        Must have a .predict() method.
    X_instance : np.ndarray
        Single instance (1D or 2D with shape (1, n_features)).
    desired_class : int or float
        The desired prediction (class label or target value threshold).
    feature_names : list or None
        Human-readable feature names.
    feature_ranges : dict or None
        Allowed range per feature: {name: (min, max)}. If None, uses +/- 3x from current.
    max_changes : int
        Maximum number of features to change simultaneously.
    n_steps : int
        Number of candidate values to try per feature.

    Returns
    -------
    dict with original_prediction, desired_prediction, changes_needed,
    counterfactual_instance, plain_language_explanation
    """
    # Ensure 2D
    if X_instance.ndim == 1:
        X_instance = X_instance.reshape(1, -1)

    n_features = X_instance.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    original_pred = model.predict(X_instance)
    original_value = float(original_pred[0])

    # Determine if classification or regression-style threshold
    is_classification = isinstance(desired_class, int) or (
        isinstance(desired_class, float) and desired_class == int(desired_class)
    )

    # Build feature ranges
    ranges = {}
    for j in range(n_features):
        name = feature_names[j]
        val = float(X_instance[0, j])
        if feature_ranges and name in feature_ranges:
            ranges[j] = feature_ranges[name]
        else:
            spread = max(abs(val) * 3.0, 1.0)
            ranges[j] = (val - spread, val + spread)

    # Phase 1: Single-feature changes
    single_results = []
    for j in range(n_features):
        lo, hi = ranges[j]
        candidates = np.linspace(lo, hi, n_steps)
        original_val = float(X_instance[0, j])

        for cand in candidates:
            X_mod = X_instance.copy()
            X_mod[0, j] = cand
            pred = float(model.predict(X_mod)[0])

            if _prediction_matches(pred, desired_class, is_classification):
                delta = abs(cand - original_val)
                single_results.append({
                    "features": [j],
                    "changes": [{
                        "feature": feature_names[j],
                        "feature_index": j,
                        "from": round(original_val, 4),
                        "to": round(float(cand), 4),
                        "delta": round(delta, 4),
                    }],
                    "new_prediction": round(pred, 4),
                    "total_delta": round(delta, 4),
                    "counterfactual": X_mod[0].copy(),
                })
                break  # found the smallest change for this feature

    # Sort single-feature results by total delta
    single_results.sort(key=lambda x: x["total_delta"])

    # Phase 2: Pair-wise changes (if no single-feature solution or max_changes > 1)
    pair_results = []
    if (not single_results or max_changes >= 2) and n_features >= 2:
        # Try pairs of features with coarser granularity
        coarse_steps = max(10, n_steps // 5)
        for j1 in range(min(n_features, 10)):
            for j2 in range(j1 + 1, min(n_features, 10)):
                lo1, hi1 = ranges[j1]
                lo2, hi2 = ranges[j2]
                cands1 = np.linspace(lo1, hi1, coarse_steps)
                cands2 = np.linspace(lo2, hi2, coarse_steps)

                best_pair = None
                for c1 in cands1:
                    for c2 in cands2:
                        X_mod = X_instance.copy()
                        X_mod[0, j1] = c1
                        X_mod[0, j2] = c2
                        pred = float(model.predict(X_mod)[0])

                        if _prediction_matches(pred, desired_class, is_classification):
                            d1 = abs(c1 - float(X_instance[0, j1]))
                            d2 = abs(c2 - float(X_instance[0, j2]))
                            total = d1 + d2
                            if best_pair is None or total < best_pair["total_delta"]:
                                best_pair = {
                                    "features": [j1, j2],
                                    "changes": [
                                        {
                                            "feature": feature_names[j1],
                                            "feature_index": j1,
                                            "from": round(float(X_instance[0, j1]), 4),
                                            "to": round(float(c1), 4),
                                            "delta": round(d1, 4),
                                        },
                                        {
                                            "feature": feature_names[j2],
                                            "feature_index": j2,
                                            "from": round(float(X_instance[0, j2]), 4),
                                            "to": round(float(c2), 4),
                                            "delta": round(d2, 4),
                                        },
                                    ],
                                    "new_prediction": round(pred, 4),
                                    "total_delta": round(total, 4),
                                    "counterfactual": X_mod[0].copy(),
                                }

                if best_pair is not None:
                    pair_results.append(best_pair)

        pair_results.sort(key=lambda x: x["total_delta"])

    # Combine and pick best
    all_results = single_results + pair_results
    all_results.sort(key=lambda x: x["total_delta"])

    if not all_results:
        return {
            "original_prediction": round(original_value, 4),
            "desired_prediction": desired_class,
            "changes_needed": [],
            "counterfactual_instance": None,
            "found_counterfactual": False,
            "plain_language_explanation": (
                "No counterfactual found within the searched feature ranges. "
                "Try expanding feature_ranges or increasing n_steps."
            ),
        }

    best = all_results[0]

    # Build plain-language explanation
    change_parts = []
    for ch in best["changes"]:
        change_parts.append(
            f"change '{ch['feature']}' from {ch['from']} to {ch['to']}"
        )
    explanation = (
        f"To get prediction={best['new_prediction']} (from {round(original_value, 4)}): "
        + " AND ".join(change_parts) + "."
    )

    # Also list alternative single-feature options
    alternatives = []
    for r in single_results[1:4]:
        ch = r["changes"][0]
        alternatives.append(
            f"OR change '{ch['feature']}' from {ch['from']} to {ch['to']}"
        )

    if alternatives:
        explanation += " Alternatives: " + "; ".join(alternatives) + "."

    return {
        "original_prediction": round(original_value, 4),
        "desired_prediction": desired_class,
        "changes_needed": best["changes"],
        "counterfactual_instance": best["counterfactual"],
        "new_prediction": best["new_prediction"],
        "found_counterfactual": True,
        "n_features_changed": len(best["changes"]),
        "total_delta": best["total_delta"],
        "all_single_feature_options": single_results[:5],
        "all_pair_options": pair_results[:3],
        "plain_language_explanation": explanation,
    }


def _prediction_matches(
    pred: float, desired: int | float, is_classification: bool
) -> bool:
    """Check if a prediction matches the desired outcome."""
    if is_classification:
        return round(pred) == desired
    else:
        # For regression, treat desired as a threshold to exceed
        return pred >= desired


def print_explanation(result: dict) -> None:
    """Print a human-readable counterfactual explanation."""
    print(f"Original prediction: {result['original_prediction']}")
    print(f"Desired prediction:  {result['desired_prediction']}")
    print()

    if not result["found_counterfactual"]:
        print(result["plain_language_explanation"])
        return

    print("Changes needed:")
    for ch in result["changes_needed"]:
        print(f"  {ch['feature']}: {ch['from']} -> {ch['to']} (delta={ch['delta']})")

    print()
    print(result["plain_language_explanation"])

    if result.get("all_single_feature_options"):
        print()
        print("All single-feature options found:")
        for opt in result["all_single_feature_options"]:
            ch = opt["changes"][0]
            print(f"  {ch['feature']}: {ch['from']} -> {ch['to']} (delta={ch['delta']})")
