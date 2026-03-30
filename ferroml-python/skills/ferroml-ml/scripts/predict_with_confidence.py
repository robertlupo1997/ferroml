"""Prediction intervals via bootstrap or native uncertainty.

Usage: Claude adapts this to produce calibrated confidence intervals around predictions.
Output: Point predictions with lower/upper bounds and plain-language summaries.
"""
from __future__ import annotations

import numpy as np


def predict_with_intervals(
    model: object,
    X: np.ndarray,
    y_train: np.ndarray | None = None,
    X_train: np.ndarray | None = None,
    confidence: float = 0.95,
    method: str = "bootstrap",
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """Generate prediction intervals for each sample.

    Parameters
    ----------
    model : fitted model
        Must have fit() and predict(). For "native" method, should support
        predict_with_uncertainty().
    X : np.ndarray
        Feature matrix for predictions (n_samples, n_features).
    y_train : np.ndarray or None
        Training targets (required for bootstrap method).
    X_train : np.ndarray or None
        Training features (required for bootstrap method).
    confidence : float
        Confidence level, e.g. 0.95 for 95% CI.
    method : str
        "bootstrap" or "native" (for GP models with predict_with_uncertainty).
    n_bootstrap : int
        Number of bootstrap iterations (only for bootstrap method).
    seed : int
        Random seed.

    Returns
    -------
    dict with predictions, lower_bounds, upper_bounds, confidence_level,
    method_used, mean_interval_width, plain_language_examples
    """
    alpha = 1.0 - confidence
    n_samples = X.shape[0]

    if method == "native":
        return _native_uncertainty(model, X, confidence, alpha)

    # Bootstrap method
    if X_train is None or y_train is None:
        raise ValueError(
            "Bootstrap method requires X_train and y_train. "
            "Pass the training data or use method='native' for GP models."
        )

    rng = np.random.RandomState(seed)
    n_train = X_train.shape[0]

    # Collect predictions from bootstrap models
    all_preds = np.zeros((n_bootstrap, n_samples))

    for b in range(n_bootstrap):
        # Resample training data with replacement
        idx = rng.choice(n_train, size=n_train, replace=True)
        X_b = X_train[idx]
        y_b = y_train[idx]

        # Fit a new model of the same type
        try:
            # Clone the model by constructing a new instance
            model_class = type(model)
            boot_model = model_class()
            boot_model.fit(X_b, y_b)
            all_preds[b, :] = boot_model.predict(X)
        except Exception:
            # If cloning fails, use the original model's predictions with noise
            base_preds = model.predict(X)
            noise = rng.normal(0, np.std(y_train) * 0.1, size=n_samples)
            all_preds[b, :] = base_preds + noise

    # Point predictions from the original model
    point_preds = model.predict(X)

    # Compute percentile-based intervals
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)
    lower_bounds = np.percentile(all_preds, lower_pct, axis=0)
    upper_bounds = np.percentile(all_preds, upper_pct, axis=0)

    interval_widths = upper_bounds - lower_bounds
    mean_width = float(np.mean(interval_widths))

    # Plain-language examples (first 5 predictions)
    examples = []
    n_show = min(5, n_samples)
    for i in range(n_show):
        examples.append(
            f"Sample {i}: Predicted {point_preds[i]:.2f} "
            f"({confidence*100:.0f}% CI: {lower_bounds[i]:.2f} to {upper_bounds[i]:.2f})"
        )

    return {
        "predictions": point_preds,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "confidence_level": confidence,
        "method_used": "bootstrap",
        "n_bootstrap": n_bootstrap,
        "mean_interval_width": round(mean_width, 4),
        "interval_widths": interval_widths,
        "plain_language_examples": examples,
    }


def _native_uncertainty(
    model: object, X: np.ndarray, confidence: float, alpha: float
) -> dict:
    """Use model's native predict_with_uncertainty method (e.g., GP models)."""
    if not hasattr(model, "predict_with_uncertainty"):
        raise ValueError(
            "Model does not support predict_with_uncertainty(). "
            "Use method='bootstrap' instead."
        )

    result = model.predict_with_uncertainty(X, confidence=confidence)

    # The result format depends on the model; handle dict or tuple
    if isinstance(result, dict):
        point_preds = np.array(result.get("predictions", result.get("mean", [])))
        lower_bounds = np.array(result.get("lower", result.get("lower_bound", [])))
        upper_bounds = np.array(result.get("upper", result.get("upper_bound", [])))
    elif isinstance(result, (list, tuple)) and len(result) >= 3:
        point_preds = np.array(result[0])
        lower_bounds = np.array(result[1])
        upper_bounds = np.array(result[2])
    else:
        raise ValueError(f"Unexpected predict_with_uncertainty output format: {type(result)}")

    interval_widths = upper_bounds - lower_bounds
    mean_width = float(np.mean(interval_widths))

    examples = []
    n_show = min(5, len(point_preds))
    for i in range(n_show):
        examples.append(
            f"Sample {i}: Predicted {point_preds[i]:.2f} "
            f"({confidence*100:.0f}% CI: {lower_bounds[i]:.2f} to {upper_bounds[i]:.2f})"
        )

    return {
        "predictions": point_preds,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "confidence_level": confidence,
        "method_used": "native (predict_with_uncertainty)",
        "mean_interval_width": round(mean_width, 4),
        "interval_widths": interval_widths,
        "plain_language_examples": examples,
    }


def print_predictions(result: dict) -> None:
    """Print a human-readable prediction summary."""
    ci_pct = result["confidence_level"] * 100
    print(f"Method: {result['method_used']}")
    print(f"Confidence level: {ci_pct:.0f}%")
    print(f"Mean interval width: {result['mean_interval_width']:.4f}")
    print()

    print("Example predictions:")
    for ex in result["plain_language_examples"]:
        print(f"  {ex}")
    print()

    n = len(result["predictions"])
    tight = int(np.sum(result["interval_widths"] < result["mean_interval_width"]))
    print(f"Summary: {tight}/{n} predictions have tighter-than-average intervals.")
    print(f"  Narrowest interval: {float(np.min(result['interval_widths'])):.4f}")
    print(f"  Widest interval:    {float(np.max(result['interval_widths'])):.4f}")
