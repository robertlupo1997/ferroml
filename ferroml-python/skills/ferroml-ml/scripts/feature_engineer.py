"""Automated feature engineering — detect column types and apply transformations.

Usage: Claude adapts this to the user's dataset, selecting appropriate encoders
and transformers based on column characteristics.
Output: Transformed feature matrix with a log of all transformations applied.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def engineer(path: str, target: str, max_features: int = 50) -> dict:
    """Auto-engineer features from a dataset.

    Parameters
    ----------
    path : str
        Path to CSV or Parquet file.
    target : str
        Target column name.
    max_features : int
        Maximum number of features to produce (caps polynomial expansion).

    Returns
    -------
    dict with keys: original_features, engineered_features,
    transformations_applied, X_transformed, y, column_types
    """
    from ferroml.preprocessing import (
        PowerTransformer,
        TargetEncoder,
    )

    # Load
    if path.endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, infer_schema_length=10000)

    y = df[target].to_numpy().astype(np.float64)
    feature_df = df.drop(target)

    original_features = feature_df.columns[:]
    transformations: list[dict] = []
    engineered_parts: list[tuple[str, np.ndarray]] = []

    # Classify columns
    numeric_cols = [
        col for col in feature_df.columns
        if feature_df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
    ]
    string_cols = [
        col for col in feature_df.columns
        if feature_df[col].dtype == pl.Utf8
    ]
    column_types: dict[str, str] = {}

    # --- Numeric features ---
    for col in numeric_cols:
        vals = feature_df[col].drop_nulls().to_numpy().astype(np.float64)
        if len(vals) == 0:
            continue
        column_types[col] = "numeric"

        # Impute NaN with median
        full_vals = feature_df[col].to_numpy().astype(np.float64)
        mask = np.isnan(full_vals)
        if mask.any():
            full_vals[mask] = float(np.nanmedian(full_vals))
            transformations.append({"column": col, "action": "median_imputation", "fill_value": round(float(np.nanmedian(vals)), 6)})

        # Skewness check — PowerTransformer if |skew| > 1
        skew = _skewness(vals)
        if abs(skew) > 1.0 and np.all(vals > 0):
            try:
                pt = PowerTransformer()
                transformed = pt.fit_transform(full_vals.reshape(-1, 1))
                engineered_parts.append((f"{col}_power", transformed.ravel()))
                transformations.append({"column": col, "action": "power_transform", "reason": f"skewness={round(skew, 3)}"})
            except Exception:
                engineered_parts.append((col, full_vals))
        else:
            engineered_parts.append((col, full_vals))

    # --- Categorical features ---
    for col in string_cols:
        n_unique = feature_df[col].n_unique()
        column_types[col] = f"categorical (cardinality={n_unique})"
        vals = feature_df[col].fill_null("__MISSING__").to_list()

        if n_unique < 20:
            # Low cardinality: OneHotEncoder
            try:
                unique_vals = sorted(set(vals))
                for uv in unique_vals:
                    binary = np.array([1.0 if v == uv else 0.0 for v in vals])
                    engineered_parts.append((f"{col}_{uv}", binary))
                transformations.append({"column": col, "action": "one_hot_encode", "n_categories": n_unique})
            except Exception:
                pass

        elif n_unique <= 100:
            # Medium cardinality: TargetEncoder
            try:
                te = TargetEncoder()
                unique_vals = sorted(set(vals))
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                codes = np.array([val_to_idx[v] for v in vals], dtype=np.float64).reshape(-1, 1)
                te.fit(codes, y)
                encoded = te.transform(codes).ravel()
                engineered_parts.append((f"{col}_target_enc", encoded))
                transformations.append({"column": col, "action": "target_encode", "n_categories": n_unique})
            except Exception:
                pass

        else:
            # High cardinality: frequency encoding
            from collections import Counter
            freq = Counter(vals)
            total = len(vals)
            freq_encoded = np.array([freq[v] / total for v in vals], dtype=np.float64)
            engineered_parts.append((f"{col}_freq", freq_encoded))
            transformations.append({"column": col, "action": "frequency_encode", "n_categories": n_unique})

    # --- Polynomial features (if few numeric columns) ---
    if 2 <= len(numeric_cols) <= 20:
        numeric_arrays = [
            part[1] for part in engineered_parts
            if any(part[0].startswith(nc) for nc in numeric_cols)
        ][:10]  # Cap at 10 base features
        if len(numeric_arrays) >= 2:
            n_interactions = 0
            for i in range(len(numeric_arrays)):
                for j in range(i + 1, len(numeric_arrays)):
                    if len(engineered_parts) + n_interactions >= max_features:
                        break
                    interaction = numeric_arrays[i] * numeric_arrays[j]
                    engineered_parts.append((f"interaction_{i}_{j}", interaction))
                    n_interactions += 1
            if n_interactions > 0:
                transformations.append({"action": "polynomial_interactions", "n_interactions": n_interactions})

    # Cap at max_features
    if len(engineered_parts) > max_features:
        engineered_parts = engineered_parts[:max_features]
        transformations.append({"action": "feature_cap", "max_features": max_features})

    # Build final matrix
    engineered_names = [p[0] for p in engineered_parts]
    X_transformed = np.column_stack([p[1] for p in engineered_parts]) if engineered_parts else np.empty((df.height, 0))

    return {
        "original_features": original_features,
        "engineered_features": engineered_names,
        "n_original": len(original_features),
        "n_engineered": len(engineered_names),
        "transformations_applied": transformations,
        "column_types": column_types,
        "X_transformed": X_transformed,
        "y": y,
    }


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of an array."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3) * n / ((n - 1) * (n - 2)) * n)


def print_summary(result: dict) -> None:
    """Print a human-readable summary of engineering results."""
    print(f"Features: {result['n_original']} original -> {result['n_engineered']} engineered")
    print()

    if result.get("column_types"):
        print("Column types:")
        for col, ctype in result["column_types"].items():
            print(f"  {col}: {ctype}")
        print()

    if result.get("transformations_applied"):
        print("Transformations applied:")
        for t in result["transformations_applied"]:
            col = t.get("column", "")
            action = t.get("action", "")
            reason = t.get("reason", "")
            detail = f" ({reason})" if reason else ""
            prefix = f"  {col}: " if col else "  "
            print(f"{prefix}{action}{detail}")
