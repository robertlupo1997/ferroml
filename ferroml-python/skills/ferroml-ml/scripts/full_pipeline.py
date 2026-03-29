"""End-to-end ML pipeline — from data to trained, evaluated model.

Usage: Claude adapts this to the user's data, target, and task type.
Output: Fitted model with full diagnostics, metrics, and plain-language summary.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def run_pipeline(
    path: str,
    target: str,
    task: str | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Run a complete ML pipeline.

    Parameters
    ----------
    path : str
        Path to CSV or Parquet file.
    target : str
        Target column name.
    task : str or None
        "classification" or "regression". Auto-detected if None.
    test_size : float
        Fraction of data for hold-out test set.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: task, model_name, train_metrics, test_metrics, diagnostics,
    recommendations, model (the fitted model object)
    """
    # 1. Load data
    if path.endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, infer_schema_length=10000)

    y_full = df[target].to_numpy().astype(np.float64)
    X_full = df.drop(target).select(pl.col("*").cast(pl.Float64, strict=False)).to_numpy().astype(np.float64)
    feature_names = [c for c in df.columns if c != target]

    # Handle NaN in features (simple median imputation)
    for j in range(X_full.shape[1]):
        col = X_full[:, j]
        mask = np.isnan(col)
        if mask.any():
            median_val = np.nanmedian(col)
            col[mask] = median_val

    # 2. Auto-detect task
    if task is None:
        task = "classification" if len(np.unique(y_full)) <= 20 else "regression"

    # 3. Recommend models
    import ferroml
    recs = ferroml.recommend(X_full, y_full, task=task)
    top_rec = recs[0] if recs else None

    # 4. Preprocessing
    from ferroml.preprocessing import StandardScaler
    scaler = StandardScaler()

    # 5. Train/test split
    rng = np.random.RandomState(seed)
    n = X_full.shape[0]
    indices = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    # Scale features
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Select and train model
    model_name, model = _select_model(task, top_rec)
    model.fit(X_train_scaled, y_train)

    # 7. Predict and evaluate
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    from ferroml import metrics
    result: dict = {
        "task": task,
        "model_name": model_name,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_features": X_full.shape[1],
        "feature_names": feature_names,
    }

    if task == "regression":
        result["train_metrics"] = {
            "r2": round(float(metrics.r2_score(y_train, train_preds)), 6),
            "rmse": round(float(metrics.rmse(y_train, train_preds)), 6),
            "mae": round(float(metrics.mae(y_train, train_preds)), 6),
        }
        result["test_metrics"] = {
            "r2": round(float(metrics.r2_score(y_test, test_preds)), 6),
            "rmse": round(float(metrics.rmse(y_test, test_preds)), 6),
            "mae": round(float(metrics.mae(y_test, test_preds)), 6),
        }
    else:
        result["train_metrics"] = {
            "accuracy": round(float(metrics.accuracy_score(y_train, train_preds)), 6),
            "f1": round(float(metrics.f1_score(y_train, train_preds)), 6),
        }
        result["test_metrics"] = {
            "accuracy": round(float(metrics.accuracy_score(y_test, test_preds)), 6),
            "f1": round(float(metrics.f1_score(y_test, test_preds)), 6),
        }

    # 8. Diagnostics
    diagnostics = {}
    if hasattr(model, "summary"):
        diagnostics["summary"] = model.summary()
    if hasattr(model, "r_squared"):
        diagnostics["r_squared"] = round(float(model.r_squared()), 6)

    # Residual analysis for regression
    if task == "regression":
        from ferroml.stats import durbin_watson, normality_test
        residuals = y_test - test_preds
        diagnostics["residual_mean"] = round(float(np.mean(residuals)), 6)
        diagnostics["residual_std"] = round(float(np.std(residuals)), 6)
        diagnostics["durbin_watson"] = round(float(durbin_watson(residuals)), 6)
        diagnostics["normality"] = normality_test(residuals)

    result["diagnostics"] = diagnostics

    # 9. Overfitting check
    if task == "regression":
        train_r2 = result["train_metrics"]["r2"]
        test_r2 = result["test_metrics"]["r2"]
        if train_r2 - test_r2 > 0.1:
            result["warning"] = f"Possible overfitting: train R2={train_r2:.4f}, test R2={test_r2:.4f}. Consider regularization or simpler model."
    else:
        train_acc = result["train_metrics"]["accuracy"]
        test_acc = result["test_metrics"]["accuracy"]
        if train_acc - test_acc > 0.05:
            result["warning"] = f"Possible overfitting: train accuracy={train_acc:.4f}, test accuracy={test_acc:.4f}."

    # 10. Model recommendations
    recommendations = []
    if top_rec:
        for r in recs[:3]:
            recommendations.append({"algorithm": r.algorithm, "reason": r.reason, "score": r.score})
    result["recommendations"] = recommendations
    result["model"] = model

    return result


def _select_model(task: str, recommendation):
    """Select a model based on task and recommendation."""
    if recommendation:
        name = recommendation.algorithm
    elif task == "regression":
        name = "LinearRegression"
    else:
        name = "LogisticRegression"

    # Import the recommended model
    from ferroml.cli._registry import construct_model
    model = construct_model(name, recommendation.params if recommendation else None)
    return name, model


def print_results(result: dict) -> None:
    """Print a human-readable results summary."""
    print(f"=== ML Pipeline Results ===")
    print(f"Task: {result['task']}")
    print(f"Model: {result['model_name']}")
    print(f"Data: {result['n_train']} train, {result['n_test']} test, {result['n_features']} features")
    print()
    print("Train metrics:", result["train_metrics"])
    print("Test metrics:", result["test_metrics"])

    if result.get("warning"):
        print(f"\nWARNING: {result['warning']}")

    if result.get("diagnostics", {}).get("summary"):
        print(f"\n{result['diagnostics']['summary']}")
