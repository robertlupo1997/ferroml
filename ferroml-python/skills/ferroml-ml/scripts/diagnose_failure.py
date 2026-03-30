"""Diagnose why a model is performing badly and suggest fixes.

Usage: Claude runs this when a model's metrics are disappointing.
Output: List of issues found with severity, details, and FerroML code to fix each.
"""
from __future__ import annotations

import numpy as np


def diagnose(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: object,
    task: str = "regression",
    feature_names: list[str] | None = None,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> dict:
    """Diagnose model failure and recommend fixes.

    Parameters
    ----------
    X : np.ndarray
        Test features (n_samples, n_features).
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Model predictions.
    model : object
        Fitted FerroML model.
    task : str
        "regression" or "classification".
    feature_names : list or None
        Human-readable feature names.
    X_train : np.ndarray or None
        Training features, used for overfitting check.
    y_train : np.ndarray or None
        Training targets, used for overfitting check.

    Returns
    -------
    dict with issues_found, severity, overall_diagnosis
    """
    from ferroml import metrics

    issues: list[dict] = []
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # --- Check 1: Overfitting (train vs test gap) ---
    if X_train is not None and y_train is not None:
        try:
            y_train_pred = model.predict(X_train)
            if task == "regression":
                train_score = float(metrics.r2_score(y_train, y_train_pred))
                test_score = float(metrics.r2_score(y_true, y_pred))
                metric_name = "R2"
            else:
                train_score = float(metrics.accuracy_score(y_train, y_train_pred))
                test_score = float(metrics.accuracy_score(y_true, y_pred))
                metric_name = "accuracy"

            gap = train_score - test_score
            if gap > 0.15:
                issues.append({
                    "check": "overfitting",
                    "status": "FAIL",
                    "detail": (
                        f"Train {metric_name}={train_score:.4f} vs "
                        f"Test {metric_name}={test_score:.4f} (gap={gap:.4f})"
                    ),
                    "fix": (
                        "Reduce model complexity or add regularization.\n"
                        "  from ferroml.trees import GradientBoostingRegressor\n"
                        "  model = GradientBoostingRegressor(max_depth=3, n_estimators=100, learning_rate=0.05)\n"
                        "Or use cross-validation to tune hyperparameters."
                    ),
                })
            else:
                issues.append({
                    "check": "overfitting",
                    "status": "OK",
                    "detail": f"Train-test gap is acceptable ({gap:.4f})",
                    "fix": None,
                })
        except Exception as e:
            issues.append({
                "check": "overfitting",
                "status": "SKIP",
                "detail": f"Could not compute train predictions: {e}",
                "fix": None,
            })

    # --- Check 2: Data quality (NaN, inf, outliers) ---
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    n_total = X.size
    nan_pct = nan_count / n_total * 100 if n_total > 0 else 0
    outlier_counts = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col_clean = col[np.isfinite(col)]
        if len(col_clean) < 4:
            continue
        q1, q3 = np.percentile(col_clean, [25, 75])
        iqr = q3 - q1
        n_outliers = int(((col_clean < q1 - 3 * iqr) | (col_clean > q3 + 3 * iqr)).sum())
        if n_outliers > 0:
            outlier_counts.append((feature_names[j], n_outliers))

    data_issues = []
    if nan_count > 0:
        data_issues.append(f"{nan_count} NaN values ({nan_pct:.1f}%)")
    if inf_count > 0:
        data_issues.append(f"{inf_count} Inf values")
    if outlier_counts:
        worst = sorted(outlier_counts, key=lambda x: x[1], reverse=True)[:3]
        data_issues.append(
            f"Outliers in {len(outlier_counts)} features "
            f"(worst: {', '.join(f'{n}={c}' for n, c in worst)})"
        )

    if data_issues:
        issues.append({
            "check": "data_quality",
            "status": "FAIL",
            "detail": "; ".join(data_issues),
            "fix": (
                "Clean data before fitting:\n"
                "  from ferroml.preprocessing import StandardScaler\n"
                "  # Remove NaN/Inf, then scale\n"
                "  X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))\n"
                "  scaler = StandardScaler()\n"
                "  scaler.fit(X)\n"
                "  X = scaler.transform(X)"
            ),
        })
    else:
        issues.append({
            "check": "data_quality",
            "status": "OK",
            "detail": "No NaN, Inf, or extreme outliers detected",
            "fix": None,
        })

    # --- Check 3: Feature issues (low variance, high correlation) ---
    low_var_features = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col_clean = col[np.isfinite(col)]
        if len(col_clean) > 0 and np.std(col_clean) < 1e-10:
            low_var_features.append(feature_names[j])

    high_corr_pairs = []
    if X.shape[1] <= 200:
        clean_X = np.nan_to_num(X)
        stds = np.std(clean_X, axis=0)
        valid_cols = np.where(stds > 1e-10)[0]
        if len(valid_cols) > 1:
            corr_matrix = np.corrcoef(clean_X[:, valid_cols].T)
            for i in range(len(valid_cols)):
                for j2 in range(i + 1, len(valid_cols)):
                    if abs(corr_matrix[i, j2]) > 0.95:
                        high_corr_pairs.append(
                            (feature_names[valid_cols[i]], feature_names[valid_cols[j2]],
                             round(float(abs(corr_matrix[i, j2])), 4))
                        )

    feature_issues = []
    if low_var_features:
        feature_issues.append(f"{len(low_var_features)} near-constant features: {low_var_features[:5]}")
    if high_corr_pairs:
        feature_issues.append(f"{len(high_corr_pairs)} highly correlated pairs (>0.95)")

    if feature_issues:
        issues.append({
            "check": "feature_issues",
            "status": "FAIL",
            "detail": "; ".join(feature_issues),
            "fix": (
                "Remove low-variance and redundant features:\n"
                "  from ferroml.preprocessing import StandardScaler\n"
                "  # Drop constant columns\n"
                "  variances = np.var(X, axis=0)\n"
                "  keep = variances > 1e-10\n"
                "  X = X[:, keep]"
            ),
        })
    else:
        issues.append({
            "check": "feature_issues",
            "status": "OK",
            "detail": "No low-variance or highly correlated features",
            "fix": None,
        })

    # --- Check 4: Model assumptions (linear models) ---
    if task == "regression":
        try:
            from ferroml.stats import durbin_watson, normality_test

            residuals = y_true - y_pred
            dw_stat = float(durbin_watson(residuals))
            norm_result = normality_test(residuals)
            norm_p = float(norm_result["p_value"]) if isinstance(norm_result, dict) else float(norm_result[1])

            assumption_issues = []
            if dw_stat < 1.5 or dw_stat > 2.5:
                assumption_issues.append(
                    f"Durbin-Watson={dw_stat:.4f} (autocorrelation in residuals)"
                )
            if norm_p < 0.05:
                assumption_issues.append(
                    f"Residuals not normal (p={norm_p:.6f})"
                )

            # Check homoscedasticity: correlation between |residuals| and predictions
            abs_resid = np.abs(residuals)
            if np.std(y_pred) > 1e-10 and np.std(abs_resid) > 1e-10:
                hetero_corr = abs(float(np.corrcoef(y_pred, abs_resid)[0, 1]))
                if hetero_corr > 0.3:
                    assumption_issues.append(
                        f"Heteroscedasticity detected (|residual|-prediction corr={hetero_corr:.4f})"
                    )

            if assumption_issues:
                issues.append({
                    "check": "model_assumptions",
                    "status": "FAIL",
                    "detail": "; ".join(assumption_issues),
                    "fix": (
                        "Consider a non-linear model or transform the target:\n"
                        "  from ferroml.trees import GradientBoostingRegressor\n"
                        "  model = GradientBoostingRegressor(n_estimators=200)\n"
                        "  # Or log-transform target: y = np.log1p(y)"
                    ),
                })
            else:
                issues.append({
                    "check": "model_assumptions",
                    "status": "OK",
                    "detail": "Residuals look well-behaved",
                    "fix": None,
                })
        except Exception as e:
            issues.append({
                "check": "model_assumptions",
                "status": "SKIP",
                "detail": f"Could not run assumption tests: {e}",
                "fix": None,
            })

    # --- Check 5: Class imbalance (classification only) ---
    if task == "classification":
        classes, counts = np.unique(y_true, return_counts=True)
        if len(classes) >= 2:
            ratio = float(counts.min()) / float(counts.max())
            if ratio < 0.2:
                issues.append({
                    "check": "class_imbalance",
                    "status": "FAIL",
                    "detail": (
                        f"Severe imbalance: minority/majority ratio={ratio:.4f} "
                        f"(classes: {dict(zip(classes.tolist(), counts.tolist()))})"
                    ),
                    "fix": (
                        "Use class weights or resample:\n"
                        "  # Oversample minority class\n"
                        "  from ferroml.linear import LogisticRegression\n"
                        "  model = LogisticRegression()  # supports class_weight internally\n"
                        "  # Or use SMOTE-like oversampling before fit"
                    ),
                })
            else:
                issues.append({
                    "check": "class_imbalance",
                    "status": "OK",
                    "detail": f"Class ratio={ratio:.4f} (acceptable)",
                    "fix": None,
                })

    # --- Check 6: Prediction distribution (constant predictions?) ---
    unique_preds = np.unique(y_pred)
    if len(unique_preds) == 1:
        issues.append({
            "check": "prediction_distribution",
            "status": "FAIL",
            "detail": f"Model predicts a single constant value: {unique_preds[0]:.6f}",
            "fix": (
                "Model has collapsed. Try a different algorithm:\n"
                "  from ferroml.trees import RandomForestRegressor\n"
                "  model = RandomForestRegressor(n_estimators=100)\n"
                "  model.fit(X_train, y_train)\n"
                "Also check: is the target constant? Are features informative?"
            ),
        })
    elif task == "regression" and np.std(y_pred) < np.std(y_true) * 0.1:
        issues.append({
            "check": "prediction_distribution",
            "status": "FAIL",
            "detail": (
                f"Predictions have very low variance "
                f"(std_pred={np.std(y_pred):.6f} vs std_true={np.std(y_true):.6f})"
            ),
            "fix": (
                "Model is under-fitting. Increase complexity:\n"
                "  # Increase tree depth or number of estimators\n"
                "  # Or use a more expressive model (ensemble, MLP)"
            ),
        })
    else:
        issues.append({
            "check": "prediction_distribution",
            "status": "OK",
            "detail": f"Predictions span {len(unique_preds)} unique values",
            "fix": None,
        })

    # --- Summarize ---
    failures = [i for i in issues if i["status"] == "FAIL"]
    severity = "critical" if len(failures) >= 3 else "moderate" if failures else "healthy"

    if failures:
        diagnosis = (
            f"Found {len(failures)} issue(s). "
            f"Top priority: {failures[0]['check']} — {failures[0]['detail']}"
        )
    else:
        diagnosis = "No obvious issues found. Consider more data or feature engineering."

    return {
        "issues_found": issues,
        "n_issues": len(failures),
        "severity": severity,
        "overall_diagnosis": diagnosis,
    }


def print_diagnosis(result: dict) -> None:
    """Print human-readable diagnosis report."""
    print(f"\n{'='*60}")
    print(f"MODEL DIAGNOSIS  |  Severity: {result['severity'].upper()}")
    print(f"{'='*60}")
    print(f"\n{result['overall_diagnosis']}\n")

    for issue in result["issues_found"]:
        icon = {"OK": "[OK]", "FAIL": "[!!]", "SKIP": "[--]"}[issue["status"]]
        print(f"  {icon} {issue['check']}: {issue['detail']}")
        if issue["fix"]:
            print(f"      Fix: {issue['fix'].split(chr(10))[0]}")
    print()
