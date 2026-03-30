"""Smart sampling for large datasets with power analysis.

Usage: Claude adapts this when datasets are too large for interactive exploration.
Output: Properly sampled data preserving class/quantile distributions.
"""
from __future__ import annotations

import numpy as np


def smart_sample(
    X: np.ndarray,
    y: np.ndarray,
    target_size: int = 5000,
    strategy: str = "stratified",
    seed: int = 42,
) -> dict:
    """Sample a dataset while preserving distributional properties.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target values.
    target_size : int
        Desired sample size.
    strategy : str
        "stratified" (preserve class/quantile distribution), "random", or "systematic".
    seed : int
        Random seed.

    Returns
    -------
    dict with X_sampled, y_sampled, original_size, sample_size, strategy_used,
    class_distribution_comparison, min_sample_size_estimate
    """
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]

    if n_samples <= target_size:
        return {
            "X_sampled": X,
            "y_sampled": y,
            "original_size": n_samples,
            "sample_size": n_samples,
            "strategy_used": "none (dataset fits target size)",
            "class_distribution_comparison": None,
            "min_sample_size_estimate": None,
        }

    # Detect task type
    n_unique = len(np.unique(y))
    is_classification = n_unique <= 20

    if strategy == "stratified":
        if is_classification:
            X_s, y_s = _stratified_classification(X, y, target_size, rng)
        else:
            X_s, y_s = _stratified_regression(X, y, target_size, rng)
    elif strategy == "systematic":
        X_s, y_s = _systematic_sample(X, y, target_size, rng)
    else:  # random
        idx = rng.choice(n_samples, size=target_size, replace=False)
        X_s, y_s = X[idx], y[idx]

    # Distribution comparison
    if is_classification:
        orig_classes, orig_counts = np.unique(y, return_counts=True)
        orig_dist = {int(c): round(n / len(y), 4) for c, n in zip(orig_classes, orig_counts)}
        samp_classes, samp_counts = np.unique(y_s, return_counts=True)
        samp_dist = {int(c): round(n / len(y_s), 4) for c, n in zip(samp_classes, samp_counts)}
        comparison = {"original": orig_dist, "sampled": samp_dist}
    else:
        comparison = {
            "original": {
                "mean": round(float(np.mean(y)), 4),
                "std": round(float(np.std(y)), 4),
                "min": round(float(np.min(y)), 4),
                "max": round(float(np.max(y)), 4),
            },
            "sampled": {
                "mean": round(float(np.mean(y_s)), 4),
                "std": round(float(np.std(y_s)), 4),
                "min": round(float(np.min(y_s)), 4),
                "max": round(float(np.max(y_s)), 4),
            },
        }

    # Minimum sample size estimate (power analysis, simplified)
    min_size = _minimum_sample_size(y, is_classification)

    return {
        "X_sampled": X_s,
        "y_sampled": y_s,
        "original_size": n_samples,
        "sample_size": X_s.shape[0],
        "strategy_used": strategy,
        "class_distribution_comparison": comparison,
        "min_sample_size_estimate": min_size,
    }


def _stratified_classification(
    X: np.ndarray, y: np.ndarray, target_size: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified sampling preserving class proportions."""
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()

    indices = []
    for cls, prop in zip(classes, proportions):
        cls_idx = np.where(y == cls)[0]
        n_take = max(1, int(round(target_size * prop)))
        n_take = min(n_take, len(cls_idx))
        sampled = rng.choice(cls_idx, size=n_take, replace=False)
        indices.append(sampled)

    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _stratified_regression(
    X: np.ndarray, y: np.ndarray, target_size: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified sampling by target quantiles for regression."""
    n_bins = min(10, max(2, target_size // 50))
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y, percentiles)

    indices = []
    per_bin = max(1, target_size // n_bins)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y >= bin_edges[i]) & (y < bin_edges[i + 1])
        else:
            mask = (y >= bin_edges[i]) & (y <= bin_edges[i + 1])

        bin_idx = np.where(mask)[0]
        if len(bin_idx) == 0:
            continue
        n_take = min(per_bin, len(bin_idx))
        sampled = rng.choice(bin_idx, size=n_take, replace=False)
        indices.append(sampled)

    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _systematic_sample(
    X: np.ndarray, y: np.ndarray, target_size: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Systematic sampling: every nth element with random start."""
    n = X.shape[0]
    step = max(1, n // target_size)
    start = rng.randint(0, step)
    idx = np.arange(start, n, step)[:target_size]
    return X[idx], y[idx]


def _minimum_sample_size(y: np.ndarray, is_classification: bool) -> dict:
    """Estimate minimum sample size via power analysis heuristics.

    Uses Cohen's guidelines: effect size d=0.2 (small), d=0.5 (medium), d=0.8 (large).
    For 80% power at alpha=0.05.
    """
    # Sample sizes needed per group for a two-sample t-test (approximation)
    # n = (z_alpha/2 + z_beta)^2 * 2 * sigma^2 / delta^2
    # For 80% power, alpha=0.05: (1.96 + 0.84)^2 = 7.84
    z_factor = 7.84  # (z_0.025 + z_0.20)^2

    if is_classification:
        classes = np.unique(y)
        n_classes = len(classes)
        # Rule of thumb: 10-30 samples per feature per class minimum
        return {
            "small_effect": int(np.ceil(z_factor * 2 / (0.2 ** 2))) * n_classes,
            "medium_effect": int(np.ceil(z_factor * 2 / (0.5 ** 2))) * n_classes,
            "large_effect": int(np.ceil(z_factor * 2 / (0.8 ** 2))) * n_classes,
            "note": "Per-group sizes for two-sample comparison at 80% power, alpha=0.05.",
        }
    else:
        sigma = float(np.std(y))
        return {
            "small_effect": int(np.ceil(z_factor * 2 * sigma ** 2 / (0.2 * sigma) ** 2)),
            "medium_effect": int(np.ceil(z_factor * 2 * sigma ** 2 / (0.5 * sigma) ** 2)),
            "large_effect": int(np.ceil(z_factor * 2 * sigma ** 2 / (0.8 * sigma) ** 2)),
            "note": "Total sample size for detecting effect at 80% power, alpha=0.05.",
        }


def print_sample_info(result: dict) -> None:
    """Print a human-readable sampling summary."""
    print(f"Sampling: {result['original_size']} -> {result['sample_size']} samples")
    print(f"Strategy: {result['strategy_used']}")
    print()

    comp = result.get("class_distribution_comparison")
    if comp:
        print("Distribution comparison:")
        print(f"  Original: {comp['original']}")
        print(f"  Sampled:  {comp['sampled']}")
        print()

    mss = result.get("min_sample_size_estimate")
    if mss:
        print("Minimum sample size estimates (80% power, alpha=0.05):")
        print(f"  Small effect:  {mss['small_effect']}")
        print(f"  Medium effect: {mss['medium_effect']}")
        print(f"  Large effect:  {mss['large_effect']}")
