"""Explore and profile a dataset — the first step in any ML workflow.

Usage: Claude adapts this script to the user's data file and target column.
Output: Dataset summary with shape, types, distributions, correlations, and red flags.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def explore(path: str, target: str | None = None) -> dict:
    """Profile a dataset and return structured findings.

    Parameters
    ----------
    path : str
        Path to CSV or Parquet file.
    target : str or None
        Target column name (if known).

    Returns
    -------
    dict with keys: shape, columns, dtypes, missing, unique_counts, numeric_stats,
    correlations, target_info, red_flags
    """
    # Load
    if path.endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, infer_schema_length=10000)

    result: dict = {}
    result["shape"] = {"rows": df.height, "columns": df.width}
    result["columns"] = df.columns
    result["dtypes"] = {col: str(df[col].dtype) for col in df.columns}

    # Missing values
    missing = {}
    for col in df.columns:
        n_null = df[col].null_count()
        if n_null > 0:
            missing[col] = {"count": n_null, "percent": round(100 * n_null / df.height, 2)}
    result["missing"] = missing

    # Unique counts (for cardinality analysis)
    result["unique_counts"] = {col: df[col].n_unique() for col in df.columns}

    # Numeric column statistics
    numeric_cols = [col for col in df.columns if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)]
    numeric_stats = {}
    for col in numeric_cols:
        vals = df[col].drop_nulls().to_numpy().astype(np.float64)
        if len(vals) == 0:
            continue
        numeric_stats[col] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "skewness": round(float(_skewness(vals)), 4),
        }
    result["numeric_stats"] = numeric_stats

    # Correlations (top pairs)
    if len(numeric_cols) >= 2:
        X_num = df.select(numeric_cols).drop_nulls().to_numpy().astype(np.float64)
        if X_num.shape[0] > 10:
            corr = np.corrcoef(X_num.T)
            top_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    r = corr[i, j]
                    if not np.isnan(r) and abs(r) > 0.5:
                        top_pairs.append({
                            "feature_1": numeric_cols[i],
                            "feature_2": numeric_cols[j],
                            "correlation": round(float(r), 4),
                        })
            top_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            result["high_correlations"] = top_pairs[:20]

    # Target analysis
    if target and target in df.columns:
        t = df[target].drop_nulls()
        t_np = t.to_numpy().astype(np.float64)
        n_unique = t.n_unique()
        target_info = {"column": target, "n_unique": n_unique, "n_null": df[target].null_count()}
        if n_unique <= 20:
            target_info["task"] = "classification"
            target_info["class_distribution"] = dict(df[target].value_counts().sort("count", descending=True).iter_rows())
        else:
            target_info["task"] = "regression"
            target_info["mean"] = round(float(np.mean(t_np)), 4)
            target_info["std"] = round(float(np.std(t_np)), 4)
            target_info["min"] = round(float(np.min(t_np)), 4)
            target_info["max"] = round(float(np.max(t_np)), 4)
            target_info["skewness"] = round(float(_skewness(t_np)), 4)
        result["target"] = target_info

    # Red flags
    red_flags = []
    if df.height < 50:
        red_flags.append(f"Very small dataset ({df.height} rows) — models may overfit.")
    if missing:
        total_missing = sum(m["count"] for m in missing.values())
        if total_missing > 0.1 * df.height * df.width:
            red_flags.append(f"High missing data rate ({total_missing} missing values across dataset).")
    for col, n in result["unique_counts"].items():
        if n == 1:
            red_flags.append(f"Column '{col}' has only 1 unique value — drop it.")
        elif n == df.height and col != target:
            red_flags.append(f"Column '{col}' is unique per row — likely an ID column, drop it.")
    high_cardinality = [col for col, n in result["unique_counts"].items()
                        if n > 100 and col != target and df[col].dtype == pl.Utf8]
    if high_cardinality:
        red_flags.append(f"High-cardinality text columns: {high_cardinality} — need encoding strategy.")
    result["red_flags"] = red_flags

    return result


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of an array."""
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3) * n / ((n - 1) * (n - 2)) * n)  # adjusted


def print_summary(result: dict) -> None:
    """Print a human-readable summary."""
    print(f"Dataset: {result['shape']['rows']} rows x {result['shape']['columns']} columns")
    print(f"Columns: {', '.join(result['columns'])}")
    print()

    if result.get("missing"):
        print("Missing values:")
        for col, info in result["missing"].items():
            print(f"  {col}: {info['count']} ({info['percent']}%)")
        print()

    if result.get("target"):
        t = result["target"]
        print(f"Target: '{t['column']}' — {t['task']}")
        if t["task"] == "classification":
            print(f"  Classes: {t['class_distribution']}")
        else:
            print(f"  Range: [{t['min']}, {t['max']}], Mean: {t['mean']}, Std: {t['std']}")
        print()

    if result.get("red_flags"):
        print("Red flags:")
        for flag in result["red_flags"]:
            print(f"  - {flag}")
