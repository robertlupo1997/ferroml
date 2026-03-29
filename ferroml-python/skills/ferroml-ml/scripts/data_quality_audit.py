"""Automated data quality audit — find and fix data issues before modeling.

Usage: Claude adapts this to the user's dataset.
Output: Structured audit with issues found, severity, and recommended fixes.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def audit(path: str, target: str | None = None) -> dict:
    """Run a comprehensive data quality audit.

    Returns dict with keys: duplicates, missing_analysis, type_issues,
    constant_columns, high_cardinality, outlier_columns, recommendations
    """
    if path.endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(path, infer_schema_length=10000)

    result: dict = {"n_rows": df.height, "n_cols": df.width}

    # 1. Duplicate rows
    n_dupes = df.height - df.unique().height
    result["duplicates"] = {"count": n_dupes, "percent": round(100 * n_dupes / max(df.height, 1), 2)}

    # 2. Missing value analysis per column
    missing = {}
    for col in df.columns:
        n_null = df[col].null_count()
        if n_null > 0:
            pct = round(100 * n_null / df.height, 2)
            # Classify missingness pattern
            if pct > 80:
                severity = "critical"
                action = "drop_column"
            elif pct > 30:
                severity = "high"
                action = "impute_or_drop"
            elif pct > 5:
                severity = "medium"
                action = "impute"
            else:
                severity = "low"
                action = "impute"
            missing[col] = {"count": n_null, "percent": pct, "severity": severity, "action": action}
    result["missing"] = missing

    # 3. Type issues (numeric stored as string, etc.)
    type_issues = []
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            # Try to cast to numeric
            try:
                numeric = df[col].drop_nulls().cast(pl.Float64)
                if len(numeric) > 0:
                    type_issues.append({
                        "column": col,
                        "issue": "numeric_as_string",
                        "fix": f"df = df.with_columns(pl.col('{col}').cast(pl.Float64))"
                    })
            except Exception:
                pass

    result["type_issues"] = type_issues

    # 4. Constant or near-constant columns
    constant = []
    for col in df.columns:
        n_unique = df[col].n_unique()
        if n_unique == 1:
            constant.append({"column": col, "issue": "constant", "action": "drop"})
        elif n_unique == 2 and df[col].null_count() > 0:
            constant.append({"column": col, "issue": "near_constant_with_nulls", "action": "review"})
    result["constant_columns"] = constant

    # 5. High cardinality categorical columns
    high_card = []
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            n_unique = df[col].n_unique()
            if n_unique > 100:
                high_card.append({
                    "column": col,
                    "n_unique": n_unique,
                    "ratio": round(n_unique / max(df.height, 1), 4),
                    "strategy": "target_encoding" if n_unique < 1000 else "hashing"
                })
    result["high_cardinality"] = high_card

    # 6. Outlier detection on numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    outlier_cols = []
    for col in numeric_cols:
        vals = df[col].drop_nulls().to_numpy().astype(np.float64)
        if len(vals) < 10:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_outliers = int(np.sum((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)))
        if n_outliers > 0:
            outlier_cols.append({
                "column": col,
                "n_outliers": n_outliers,
                "percent": round(100 * n_outliers / len(vals), 2),
                "strategy": "investigate" if n_outliers < 5 else "winsorize_or_cap"
            })
    result["outlier_columns"] = outlier_cols

    # 7. ID-like columns (unique per row, not target)
    id_columns = []
    for col in df.columns:
        if col == target:
            continue
        if df[col].n_unique() == df.height:
            id_columns.append(col)
    result["id_columns"] = id_columns

    # 8. Generate prioritized recommendations
    recommendations = []
    if n_dupes > 0:
        recommendations.append(f"Remove {n_dupes} duplicate rows.")
    for col in id_columns:
        recommendations.append(f"Drop ID column '{col}' — unique per row, no predictive value.")
    for col_info in constant:
        if col_info["action"] == "drop":
            recommendations.append(f"Drop constant column '{col_info['column']}'.")
    for col, info in missing.items():
        if info["action"] == "drop_column":
            recommendations.append(f"Drop column '{col}' — {info['percent']}% missing (too many to impute).")
        elif info["action"] == "impute":
            recommendations.append(f"Impute '{col}' — {info['count']} missing values ({info['percent']}%).")
    for ti in type_issues:
        recommendations.append(f"Cast '{ti['column']}' to numeric — stored as string.")
    result["recommendations"] = recommendations

    return result


def print_audit(result: dict) -> None:
    """Print a human-readable audit report."""
    print(f"=== Data Quality Audit ({result['n_rows']} rows, {result['n_cols']} columns) ===\n")

    if result["duplicates"]["count"] > 0:
        print(f"Duplicates: {result['duplicates']['count']} ({result['duplicates']['percent']}%)")

    if result["missing"]:
        print("\nMissing values:")
        for col, info in sorted(result["missing"].items(), key=lambda x: -x[1]["percent"]):
            print(f"  [{info['severity'].upper():8s}] {col}: {info['count']} ({info['percent']}%) → {info['action']}")

    if result["type_issues"]:
        print("\nType issues:")
        for ti in result["type_issues"]:
            print(f"  {ti['column']}: {ti['issue']}")

    if result["outlier_columns"]:
        print("\nOutliers (3x IQR):")
        for oc in result["outlier_columns"]:
            print(f"  {oc['column']}: {oc['n_outliers']} outliers ({oc['percent']}%) → {oc['strategy']}")

    if result["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"  {i}. {rec}")
