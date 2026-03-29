"""Detect data leakage — features that shouldn't be available at prediction time.

Usage: Claude runs this before training to catch leaky features.
Output: List of suspicious features with reasons and recommendations.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def detect(path: str, target: str) -> dict:
    """Scan features for potential data leakage.

    Checks:
    1. Features with suspiciously high correlation to target
    2. Features that perfectly predict target (1:1 mapping)
    3. Features with names suggesting future information

    Returns dict with suspicious_features list and recommendations.
    """
    if path.endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, infer_schema_length=10000)

    y = df[target].drop_nulls().to_numpy().astype(np.float64)
    suspicious = []

    # 1. Check numeric features for high correlation with target
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            continue

        vals = df[col].drop_nulls().to_numpy().astype(np.float64)
        if len(vals) != len(y) or len(vals) < 10:
            continue

        # Correlation
        if np.std(vals) > 0 and np.std(y) > 0:
            corr = abs(float(np.corrcoef(vals, y[:len(vals)])[0, 1]))
            if corr > 0.95:
                suspicious.append({
                    "column": col,
                    "reason": f"Extremely high correlation with target ({corr:.4f})",
                    "severity": "critical",
                    "action": "remove — likely contains target information",
                })
            elif corr > 0.85:
                suspicious.append({
                    "column": col,
                    "reason": f"Very high correlation with target ({corr:.4f})",
                    "severity": "warning",
                    "action": "investigate — may be legitimate or leaky",
                })

    # 2. Check for perfect predictors (categorical features that map 1:1 to target)
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype == pl.Utf8:
            # Check if each category value maps to exactly one target value
            grouped = df.select([col, target]).drop_nulls().group_by(col).agg(
                pl.col(target).n_unique().alias("n_target_vals")
            )
            if grouped["n_target_vals"].max() == 1:
                suspicious.append({
                    "column": col,
                    "reason": "Each category value maps to exactly one target value",
                    "severity": "critical",
                    "action": "remove — this is almost certainly leakage",
                })

    # 3. Check for suspicious column names (heuristic)
    future_keywords = ["future", "next", "outcome", "result", "label", "answer",
                       "prediction", "forecast", "target", "response", "output"]
    for col in df.columns:
        if col == target:
            continue
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        for kw in future_keywords:
            if kw in col_lower:
                suspicious.append({
                    "column": col,
                    "reason": f"Name contains '{kw}' — may contain future information",
                    "severity": "warning",
                    "action": "verify this feature is available at prediction time",
                })
                break

    # Recommendations
    recommendations = []
    critical = [s for s in suspicious if s["severity"] == "critical"]
    warnings = [s for s in suspicious if s["severity"] == "warning"]

    if critical:
        recommendations.append(f"REMOVE {len(critical)} features with critical leakage risk: "
                              f"{', '.join(s['column'] for s in critical)}")
    if warnings:
        recommendations.append(f"INVESTIGATE {len(warnings)} features with potential leakage: "
                              f"{', '.join(s['column'] for s in warnings)}")
    if not suspicious:
        recommendations.append("No obvious leakage detected. Features look clean.")

    return {
        "suspicious_features": suspicious,
        "n_critical": len(critical),
        "n_warnings": len(warnings),
        "recommendations": recommendations,
    }
