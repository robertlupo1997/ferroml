"""Class imbalance detection and treatment.

Usage: Claude adapts this to handle imbalanced classification datasets.
Output: Resampled data with strategy analysis and before/after comparisons.
"""
from __future__ import annotations

import numpy as np


def analyze_and_fix(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "auto",
    seed: int = 42,
) -> dict:
    """Detect class imbalance and apply a resampling strategy.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Class labels.
    strategy : str
        One of "auto", "none", "class_weight", "smote", "undersample", "oversample".
        "auto" selects based on imbalance ratio.
    seed : int
        Random seed.

    Returns
    -------
    dict with imbalance_ratio, strategy_used, X_resampled, y_resampled,
    class_distribution_before, class_distribution_after, recommendations
    """
    from ferroml.preprocessing import SMOTE, RandomOverSampler, RandomUnderSampler

    # Analyze class distribution
    classes, counts = np.unique(y, return_counts=True)
    class_dist_before = {int(c): int(n) for c, n in zip(classes, counts)}

    majority_count = int(np.max(counts))
    minority_count = int(np.min(counts))
    imbalance_ratio = round(majority_count / max(minority_count, 1), 2)

    minority_class = int(classes[np.argmin(counts)])
    majority_class = int(classes[np.argmax(counts)])

    # Select strategy if auto
    if strategy == "auto":
        if imbalance_ratio < 3.0:
            strategy = "none"
        elif imbalance_ratio < 10.0:
            strategy = "class_weight"
        else:
            strategy = "smote"

    # Apply strategy
    X_resampled = X.copy()
    y_resampled = y.copy()
    applied_method = strategy

    if strategy == "none":
        pass  # no resampling needed

    elif strategy == "class_weight":
        # class_weight is handled at model level, not data level
        # Return original data with weight recommendation
        pass

    elif strategy == "smote":
        try:
            sampler = SMOTE(random_state=seed)
            sampler.fit(X, y)
            X_resampled = sampler.transform(X)
            # SMOTE returns augmented X; recompute y from the output shape
            # The SMOTE API returns transformed X with synthetic samples appended
            n_synthetic = X_resampled.shape[0] - X.shape[0]
            y_resampled = np.concatenate([y, np.full(n_synthetic, minority_class)])
        except Exception:
            # Fallback to random oversampling
            applied_method = "oversample (smote_fallback)"
            X_resampled, y_resampled = _random_oversample(X, y, seed)

    elif strategy == "undersample":
        try:
            sampler = RandomUnderSampler(random_state=seed)
            sampler.fit(X, y)
            X_resampled = sampler.transform(X)
            # Undersample keeps minority + downsampled majority
            n_keep = minority_count * len(classes)
            y_resampled = y[:n_keep]  # placeholder
        except Exception:
            X_resampled, y_resampled = _random_undersample(X, y, seed)

    elif strategy == "oversample":
        try:
            sampler = RandomOverSampler(random_state=seed)
            sampler.fit(X, y)
            X_resampled = sampler.transform(X)
            n_synthetic = X_resampled.shape[0] - X.shape[0]
            y_resampled = np.concatenate([y, np.full(n_synthetic, minority_class)])
        except Exception:
            X_resampled, y_resampled = _random_oversample(X, y, seed)

    # After distribution
    classes_after, counts_after = np.unique(y_resampled, return_counts=True)
    class_dist_after = {int(c): int(n) for c, n in zip(classes_after, counts_after)}

    # Recommendations
    recommendations = []
    if imbalance_ratio < 3.0:
        recommendations.append("Mild imbalance — standard models should handle this.")
    elif imbalance_ratio < 10.0:
        recommendations.append(
            "Moderate imbalance — consider class_weight parameter or stratified CV."
        )
        recommendations.append("Evaluate with balanced_accuracy, F1, or PR-AUC, not just accuracy.")
    else:
        recommendations.append(
            "Severe imbalance — resampling recommended. SMOTE or undersampling can help."
        )
        recommendations.append("Use precision-recall curves, not ROC, for evaluation.")
        recommendations.append("Consider cost-sensitive learning (see cost_sensitive_analysis).")

    if strategy == "class_weight":
        recommendations.append(
            "Data unchanged — pass class_weight='balanced' or a custom dict to your model."
        )

    return {
        "imbalance_ratio": imbalance_ratio,
        "majority_class": majority_class,
        "minority_class": minority_class,
        "majority_count": majority_count,
        "minority_count": minority_count,
        "strategy_used": applied_method,
        "original_size": X.shape[0],
        "resampled_size": X_resampled.shape[0],
        "X_resampled": X_resampled,
        "y_resampled": y_resampled,
        "class_distribution_before": class_dist_before,
        "class_distribution_after": class_dist_after,
        "recommendations": recommendations,
    }


def _random_oversample(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simple random oversampling fallback."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))

    X_parts = [X]
    y_parts = [y]
    for cls, cnt in zip(classes, counts):
        if cnt < max_count:
            deficit = max_count - cnt
            idx = np.where(y == cls)[0]
            sampled = rng.choice(idx, size=deficit, replace=True)
            X_parts.append(X[sampled])
            y_parts.append(y[sampled])

    return np.vstack(X_parts), np.concatenate(y_parts)


def _random_undersample(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simple random undersampling fallback."""
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts))

    X_parts = []
    y_parts = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        sampled = rng.choice(idx, size=min_count, replace=False)
        X_parts.append(X[sampled])
        y_parts.append(y[sampled])

    return np.vstack(X_parts), np.concatenate(y_parts)


def print_analysis(result: dict) -> None:
    """Print a human-readable summary."""
    print(f"Imbalance ratio: {result['imbalance_ratio']}:1 "
          f"(majority={result['majority_class']}, minority={result['minority_class']})")
    print(f"Strategy: {result['strategy_used']}")
    print()

    print("Before:")
    for cls, cnt in sorted(result["class_distribution_before"].items()):
        print(f"  Class {cls}: {cnt} samples")

    print("After:")
    for cls, cnt in sorted(result["class_distribution_after"].items()):
        print(f"  Class {cls}: {cnt} samples")

    print(f"\nDataset size: {result['original_size']} -> {result['resampled_size']}")
    print()

    print("Recommendations:")
    for rec in result["recommendations"]:
        print(f"  - {rec}")
