"""Check and fix probability calibration for classifiers.

Usage: Claude adapts this after fitting a probabilistic classifier.
Output: Calibration curve, ECE, and recommendation for recalibration.
"""
from __future__ import annotations

import numpy as np


def calibrate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    task: str = "classification",
    n_bins: int = 10,
    seed: int = 42,
) -> dict:
    """Assess probability calibration and recommend corrections.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        True binary labels (0 or 1).
    model_name : str
        Model name to construct and fit.
    task : str
        Should be "classification".
    n_bins : int
        Number of bins for calibration curve.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: bins, observed_frequency, predicted_probability, ece,
    recommendation, is_well_calibrated, bin_details, model_name
    """
    from ferroml.cli._registry import construct_model
    from ferroml.preprocessing import StandardScaler

    # Train/test split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(X.shape[0])
    split = int(0.7 * X.shape[0])
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit model and get probabilities
    model = construct_model(model_name)
    model.fit(X_train, y_train)

    try:
        y_proba = model.predict_proba(X_test)
        # If multi-column, take positive class column
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
    except (AttributeError, Exception):
        # Fallback: use raw predictions clipped to [0, 1]
        y_proba = np.clip(model.predict(X_test), 0.0, 1.0)

    # Compute calibration curve
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details: list[dict] = []
    observed_freqs: list[float] = []
    predicted_probs: list[float] = []
    bin_centers: list[float] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi) if i < n_bins - 1 else (y_proba >= lo) & (y_proba <= hi)
        n_in_bin = int(np.sum(mask))

        if n_in_bin == 0:
            continue

        mean_pred = float(np.mean(y_proba[mask]))
        mean_true = float(np.mean(y_test[mask]))

        bin_centers.append(round((lo + hi) / 2, 4))
        predicted_probs.append(round(mean_pred, 6))
        observed_freqs.append(round(mean_true, 6))
        bin_details.append({
            "bin": i + 1,
            "range": f"[{lo:.2f}, {hi:.2f})",
            "n_samples": n_in_bin,
            "mean_predicted": round(mean_pred, 6),
            "mean_observed": round(mean_true, 6),
            "gap": round(abs(mean_pred - mean_true), 6),
        })

    # Expected Calibration Error
    total = sum(b["n_samples"] for b in bin_details)
    ece = sum(b["n_samples"] * b["gap"] for b in bin_details) / max(total, 1)
    ece = round(ece, 6)

    # Assess calibration quality
    is_well_calibrated = ece < 0.05

    # Recommendations
    recommendations: list[str] = []
    if is_well_calibrated:
        recommendations.append(f"Model is well-calibrated (ECE={ece:.4f} < 0.05).")
        recommendations.append("No recalibration needed.")
    elif ece < 0.10:
        recommendations.append(f"Model is moderately calibrated (ECE={ece:.4f}).")
        recommendations.append("Consider Platt scaling (sigmoid) for a slight improvement.")
        recommendations.append("Platt scaling works well when calibration error is monotonic.")
    else:
        recommendations.append(f"Model is poorly calibrated (ECE={ece:.4f}).")
        # Check if over-confident or under-confident
        high_pred_bins = [b for b in bin_details if b["mean_predicted"] > 0.5]
        if high_pred_bins:
            avg_gap = np.mean([b["mean_predicted"] - b["mean_observed"] for b in high_pred_bins])
            if avg_gap > 0.05:
                recommendations.append("Model appears over-confident (predicted > observed).")
            elif avg_gap < -0.05:
                recommendations.append("Model appears under-confident (predicted < observed).")
        recommendations.append("Try isotonic calibration for non-monotonic calibration errors.")
        recommendations.append("Alternatively, use Platt scaling (sigmoid) for a parametric fix.")
        recommendations.append("Ensure calibration is done on held-out data (not training data).")

    return {
        "model_name": model_name,
        "bins": bin_centers,
        "observed_frequency": observed_freqs,
        "predicted_probability": predicted_probs,
        "ece": ece,
        "is_well_calibrated": is_well_calibrated,
        "recommendation": recommendations,
        "bin_details": bin_details,
        "n_test_samples": len(y_test),
    }


def print_summary(result: dict) -> None:
    """Print a human-readable calibration summary."""
    print(f"Model: {result['model_name']}")
    print(f"ECE: {result['ece']:.4f}  ({'well-calibrated' if result['is_well_calibrated'] else 'needs calibration'})")
    print(f"Test samples: {result['n_test_samples']}")
    print()

    print(f"{'Bin':<8} {'Range':<15} {'N':<8} {'Predicted':<12} {'Observed':<12} {'Gap':<8}")
    print("-" * 63)
    for b in result["bin_details"]:
        print(f"{b['bin']:<8} {b['range']:<15} {b['n_samples']:<8} {b['mean_predicted']:<12.4f} {b['mean_observed']:<12.4f} {b['gap']:<8.4f}")
    print()

    # Simple ASCII calibration diagram
    print("Calibration diagram (predicted vs observed):")
    print("  1.0 |")
    for row in range(10, 0, -1):
        level = row / 10.0
        line = f"  {level:.1f} |"
        for b in result["bin_details"]:
            if abs(b["mean_observed"] - level) < 0.05:
                line += " o"
            elif abs(b["mean_predicted"] - level) < 0.05:
                line += " x"
            else:
                line += "  "
        print(line)
    print(f"  0.0 +{'--' * max(len(result['bin_details']), 1)}")
    print("       o=observed, x=predicted, diagonal=perfect")
    print()

    print("Recommendations:")
    for rec in result["recommendation"]:
        print(f"  - {rec}")
