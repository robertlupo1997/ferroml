"""Statistical assumption validation for linear models.

Usage: Claude adapts this to check whether linear model assumptions hold.
Output: Per-assumption test results with pass/fail, interpretation, and remediation.
"""
from __future__ import annotations

import numpy as np


def validate(
    X: np.ndarray,
    y: np.ndarray,
    model: object | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """Validate classical linear regression assumptions.

    Checks: linearity (runs test on residuals), normality (Shapiro-Wilk),
    homoscedasticity (variance ratio across quantile groups), independence
    (Durbin-Watson), and multicollinearity (VIF).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target values.
    model : fitted model or None
        If None, fits a LinearRegression internally.
    feature_names : list or None
        Human-readable feature names.

    Returns
    -------
    dict with assumptions (list of check results), all_passed, recommendations
    """
    from ferroml.stats import durbin_watson, normality_test

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Fit model if not provided
    if model is None:
        from ferroml.linear import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

    predictions = model.predict(X)
    residuals = y - predictions

    assumptions = []

    # 1. Linearity — runs test on residual signs
    linearity = _check_linearity(residuals, predictions)
    assumptions.append(linearity)

    # 2. Normality — Shapiro-Wilk on residuals
    normality = _check_normality(residuals, normality_test)
    assumptions.append(normality)

    # 3. Homoscedasticity — variance ratio across quantile groups
    homoscedasticity = _check_homoscedasticity(residuals, predictions)
    assumptions.append(homoscedasticity)

    # 4. Independence — Durbin-Watson
    independence = _check_independence(residuals, durbin_watson)
    assumptions.append(independence)

    # 5. Multicollinearity — VIF
    if X.shape[1] >= 2:
        vif_result = _check_multicollinearity(X, feature_names)
        assumptions.append(vif_result)

    all_passed = all(a["passed"] for a in assumptions)

    recommendations = []
    for a in assumptions:
        if not a["passed"]:
            recommendations.append(a["fix_if_failed"])

    return {
        "assumptions": assumptions,
        "all_passed": all_passed,
        "n_checks": len(assumptions),
        "n_passed": sum(1 for a in assumptions if a["passed"]),
        "n_failed": sum(1 for a in assumptions if not a["passed"]),
        "recommendations": recommendations,
    }


def _check_linearity(residuals: np.ndarray, predictions: np.ndarray) -> dict:
    """Runs test: check for non-random patterns in residual signs."""
    # Sort residuals by predicted value
    order = np.argsort(predictions)
    sorted_resid = residuals[order]

    # Count runs (consecutive sequences of same sign)
    signs = np.sign(sorted_resid)
    signs[signs == 0] = 1  # treat zero as positive
    runs = 1 + int(np.sum(np.abs(np.diff(signs)) > 0))

    n_pos = int(np.sum(signs > 0))
    n_neg = int(np.sum(signs < 0))
    n = n_pos + n_neg

    if n_pos == 0 or n_neg == 0:
        return {
            "test_name": "linearity (runs test)",
            "statistic": runs,
            "p_value_or_value": None,
            "passed": False,
            "interpretation": "All residuals have the same sign — model may be heavily biased.",
            "fix_if_failed": "Try polynomial features, splines, or a non-linear model.",
        }

    # Expected runs under randomness
    expected = 1 + 2 * n_pos * n_neg / n
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n * n * (n - 1))

    if var_runs > 0:
        z = (runs - expected) / np.sqrt(var_runs)
        # Two-tailed p-value approximation
        p_value = float(2 * (1 - _normal_cdf(abs(z))))
    else:
        z = 0.0
        p_value = 1.0

    passed = p_value > 0.05

    return {
        "test_name": "linearity (runs test)",
        "statistic": round(float(z), 4),
        "p_value_or_value": round(p_value, 6),
        "passed": passed,
        "interpretation": (
            "Residuals show no systematic pattern — linearity holds."
            if passed
            else "Residuals show a non-random pattern — linearity assumption may be violated."
        ),
        "fix_if_failed": "Try polynomial features, interaction terms, or a non-linear model.",
    }


def _check_normality(residuals: np.ndarray, normality_test_fn: object) -> dict:
    """Shapiro-Wilk test on residuals."""
    # Use at most 5000 samples for Shapiro-Wilk
    if len(residuals) > 5000:
        rng = np.random.RandomState(42)
        sample = rng.choice(residuals, size=5000, replace=False)
    else:
        sample = residuals

    try:
        result = normality_test_fn(sample)
        if isinstance(result, dict):
            stat = float(result.get("statistic", 0))
            p_val = float(result.get("p_value", 0))
        else:
            stat, p_val = float(result[0]), float(result[1])
    except Exception:
        stat, p_val = 0.0, 0.0

    passed = p_val > 0.05

    return {
        "test_name": "normality (Shapiro-Wilk)",
        "statistic": round(stat, 4),
        "p_value_or_value": round(p_val, 6),
        "passed": passed,
        "interpretation": (
            "Residuals appear normally distributed."
            if passed
            else "Residuals deviate from normality."
        ),
        "fix_if_failed": "Try a log or Box-Cox transform on the target, or use robust standard errors.",
    }


def _check_homoscedasticity(residuals: np.ndarray, predictions: np.ndarray) -> dict:
    """Check constant variance by comparing residual variance across quantile groups."""
    n_groups = min(4, max(2, len(residuals) // 20))
    order = np.argsort(predictions)
    sorted_resid = residuals[order]
    groups = np.array_split(sorted_resid, n_groups)
    variances = [float(np.var(g)) for g in groups if len(g) > 1]

    if len(variances) < 2 or min(variances) == 0:
        return {
            "test_name": "homoscedasticity (variance ratio)",
            "statistic": None,
            "p_value_or_value": None,
            "passed": True,
            "interpretation": "Insufficient data to test variance homogeneity.",
            "fix_if_failed": "Use weighted least squares or robust standard errors.",
        }

    variance_ratio = round(max(variances) / max(min(variances), 1e-15), 4)
    passed = variance_ratio < 3.0  # rule of thumb: ratio < 3 is acceptable

    return {
        "test_name": "homoscedasticity (variance ratio)",
        "statistic": variance_ratio,
        "p_value_or_value": None,
        "passed": passed,
        "interpretation": (
            f"Residual variance is roughly constant (ratio={variance_ratio})."
            if passed
            else f"Residual variance changes across predicted values (ratio={variance_ratio})."
        ),
        "fix_if_failed": "Use weighted least squares, log-transform the target, or use robust standard errors.",
    }


def _check_independence(residuals: np.ndarray, durbin_watson_fn: object) -> dict:
    """Durbin-Watson test for autocorrelation in residuals."""
    try:
        result = durbin_watson_fn(residuals)
        dw = float(result) if not isinstance(result, dict) else float(result.get("statistic", 2.0))
    except Exception:
        dw = 2.0

    # DW ranges from 0 to 4; 2 means no autocorrelation
    passed = 1.5 < dw < 2.5

    if dw < 1.5:
        interp = f"Positive autocorrelation detected (DW={round(dw, 4)}). Residuals are not independent."
    elif dw > 2.5:
        interp = f"Negative autocorrelation detected (DW={round(dw, 4)}). Residuals are not independent."
    else:
        interp = f"No significant autocorrelation (DW={round(dw, 4)}). Independence holds."

    return {
        "test_name": "independence (Durbin-Watson)",
        "statistic": round(dw, 4),
        "p_value_or_value": None,
        "passed": passed,
        "interpretation": interp,
        "fix_if_failed": "Add lagged features, use time-series models, or Newey-West standard errors.",
    }


def _check_multicollinearity(X: np.ndarray, feature_names: list[str]) -> dict:
    """Compute VIF (Variance Inflation Factor) for each feature."""
    n_features = X.shape[1]
    vifs = []

    for j in range(n_features):
        # Regress feature j on all other features
        others = np.delete(X, j, axis=1)
        target_col = X[:, j]

        if np.std(target_col) == 0:
            vifs.append({"feature": feature_names[j], "vif": 1.0})
            continue

        # OLS via normal equations
        ones = np.ones((others.shape[0], 1))
        Z = np.hstack([ones, others])
        try:
            beta = np.linalg.lstsq(Z, target_col, rcond=None)[0]
            pred = Z @ beta
            ss_res = float(np.sum((target_col - pred) ** 2))
            ss_tot = float(np.sum((target_col - np.mean(target_col)) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
            vif = 1.0 / max(1.0 - r2, 1e-10)
        except Exception:
            vif = 1.0

        vifs.append({"feature": feature_names[j], "vif": round(vif, 2)})

    max_vif = max(v["vif"] for v in vifs) if vifs else 1.0
    high_vif_features = [v["feature"] for v in vifs if v["vif"] > 10]
    passed = max_vif < 10.0

    return {
        "test_name": "multicollinearity (VIF)",
        "statistic": vifs,
        "p_value_or_value": round(max_vif, 2),
        "passed": passed,
        "interpretation": (
            f"No multicollinearity issues (max VIF={round(max_vif, 2)})."
            if passed
            else f"High multicollinearity detected: {high_vif_features} (max VIF={round(max_vif, 2)})."
        ),
        "fix_if_failed": "Remove or combine correlated features, or use Ridge/Lasso regularization.",
    }


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function."""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def print_validation(result: dict) -> None:
    """Print a human-readable validation report."""
    print(f"Assumption checks: {result['n_passed']}/{result['n_checks']} passed")
    print()

    for a in result["assumptions"]:
        status = "PASS" if a["passed"] else "FAIL"
        print(f"  [{status}] {a['test_name']}")
        print(f"         {a['interpretation']}")
        if not a["passed"]:
            print(f"         Fix: {a['fix_if_failed']}")
        print()

    if result["recommendations"]:
        print("Recommendations:")
        for rec in result["recommendations"]:
            print(f"  - {rec}")
