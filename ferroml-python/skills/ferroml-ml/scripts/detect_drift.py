"""Detect data drift between training and production distributions.

Usage: Claude runs this to monitor model reliability in production.
Output: Per-feature drift metrics (KS statistic, PSI) and recommendations.
"""
from __future__ import annotations

import numpy as np


def detect(
    X_train: np.ndarray,
    X_prod: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = 0.05,
) -> dict:
    """Compare training vs production data distributions for drift.

    Parameters
    ----------
    X_train : np.ndarray
        Training data (n_train, n_features).
    X_prod : np.ndarray
        Production data (n_prod, n_features).
    feature_names : list or None
        Human-readable feature names.
    threshold : float
        KS p-value threshold for flagging drift (default 0.05).

    Returns
    -------
    dict with feature_drift, overall_drift_score, drifted_features, recommendations
    """
    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    feature_drift: list[dict] = []
    drifted_features: list[str] = []

    for j in range(n_features):
        train_col = X_train[:, j]
        prod_col = X_prod[:, j]

        # Clean NaN/Inf
        train_clean = train_col[np.isfinite(train_col)]
        prod_clean = prod_col[np.isfinite(prod_col)]

        if len(train_clean) < 2 or len(prod_clean) < 2:
            feature_drift.append({
                "feature": feature_names[j],
                "ks_statistic": None,
                "ks_drifted": False,
                "psi": None,
                "psi_drifted": False,
                "drifted": False,
                "note": "Insufficient data",
            })
            continue

        # KS statistic: max |F_train(x) - F_prod(x)|
        ks_stat = _ks_statistic(train_clean, prod_clean)
        # Approximate KS critical value (two-sample)
        n_eff = (len(train_clean) * len(prod_clean)) / (len(train_clean) + len(prod_clean))
        # Common approximation: reject if ks_stat > c(alpha) * sqrt((n1+n2)/(n1*n2))
        # c(0.05) ~ 1.36
        c_alpha = {0.01: 1.63, 0.05: 1.36, 0.10: 1.22}.get(threshold, 1.36)
        ks_critical = c_alpha / np.sqrt(n_eff)
        ks_drifted = bool(ks_stat > ks_critical)

        # PSI (Population Stability Index)
        psi = _psi(train_clean, prod_clean)
        psi_drifted = bool(psi > 0.2)  # PSI > 0.2 = significant shift

        drifted = ks_drifted or psi_drifted
        if drifted:
            drifted_features.append(feature_names[j])

        feature_drift.append({
            "feature": feature_names[j],
            "ks_statistic": round(float(ks_stat), 6),
            "ks_critical": round(float(ks_critical), 6),
            "ks_drifted": ks_drifted,
            "psi": round(float(psi), 6),
            "psi_drifted": psi_drifted,
            "drifted": drifted,
            "train_mean": round(float(np.mean(train_clean)), 6),
            "prod_mean": round(float(np.mean(prod_clean)), 6),
            "train_std": round(float(np.std(train_clean)), 6),
            "prod_std": round(float(np.std(prod_clean)), 6),
        })

    # Overall drift score: fraction of features that drifted
    overall_drift_score = len(drifted_features) / n_features if n_features > 0 else 0.0

    # Recommendations
    recommendations = []
    if overall_drift_score > 0.5:
        recommendations.append(
            "CRITICAL: >50% of features have drifted. Retrain the model immediately."
        )
    elif overall_drift_score > 0.2:
        recommendations.append(
            "WARNING: >20% of features have drifted. Schedule retraining soon."
        )
    elif drifted_features:
        recommendations.append(
            f"MODERATE: {len(drifted_features)} feature(s) drifted. "
            "Monitor closely and retrain if performance degrades."
        )
    else:
        recommendations.append("No significant drift detected. Model inputs look stable.")

    if drifted_features:
        recommendations.append(
            f"Drifted features to investigate: {', '.join(drifted_features[:10])}"
        )
        recommendations.append(
            "Consider:\n"
            "  1. Retrain on recent data including production samples\n"
            "  2. Add feature preprocessing to handle distribution shift\n"
            "  3. Set up automated drift monitoring with periodic checks"
        )

    return {
        "feature_drift": feature_drift,
        "overall_drift_score": round(overall_drift_score, 4),
        "n_features": n_features,
        "n_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "recommendations": recommendations,
    }


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max |F_a(x) - F_b(x)|.

    Computes the exact KS statistic without scipy dependency.
    """
    combined = np.concatenate([a, b])
    combined.sort()

    n_a = len(a)
    n_b = len(b)

    # Compute empirical CDFs at each point
    cdf_a = np.searchsorted(np.sort(a), combined, side="right") / n_a
    cdf_b = np.searchsorted(np.sort(b), combined, side="right") / n_b

    return float(np.max(np.abs(cdf_a - cdf_b)))


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1: no significant shift
    PSI 0.1-0.2: moderate shift
    PSI > 0.2: significant shift
    """
    # Use expected's range for bin edges
    min_val = min(float(np.min(expected)), float(np.min(actual)))
    max_val = max(float(np.max(expected)), float(np.max(actual)))

    if max_val == min_val:
        return 0.0

    edges = np.linspace(min_val, max_val, n_bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts = np.histogram(expected, bins=edges)[0].astype(np.float64)
    actual_counts = np.histogram(actual, bins=edges)[0].astype(np.float64)

    # Avoid division by zero with small epsilon
    eps = 1e-6
    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return max(psi, 0.0)


def print_drift(result: dict) -> None:
    """Print human-readable drift report."""
    print(f"\n{'='*60}")
    print(f"DATA DRIFT REPORT  |  Score: {result['overall_drift_score']:.2%}")
    print(f"{'='*60}")
    print(f"\nFeatures: {result['n_features']}  |  Drifted: {result['n_drifted']}")

    for fd in result["feature_drift"]:
        icon = "[!!]" if fd["drifted"] else "[OK]"
        ks = f"KS={fd['ks_statistic']:.4f}" if fd["ks_statistic"] is not None else "KS=N/A"
        psi = f"PSI={fd['psi']:.4f}" if fd["psi"] is not None else "PSI=N/A"
        print(f"  {icon} {fd['feature']:30s}  {ks}  {psi}")

    print(f"\nRecommendations:")
    for rec in result["recommendations"]:
        print(f"  - {rec}")
    print()
