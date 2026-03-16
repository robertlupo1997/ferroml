"""
FerroML OLS LinearRegression diagnostics vs statsmodels — high-precision tests.

Verifies R², adjusted R², F-statistic, coefficients, standard errors,
t-statistics, p-values, confidence intervals, residuals, Durbin-Watson,
prediction interval coverage, and edge cases.
"""

import re

import numpy as np
import pytest
from numpy.testing import assert_allclose
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

from ferroml.linear import LinearRegression


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def ols_models():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.5

    X_sm = sm.add_constant(X)
    sm_model = sm.OLS(y, X_sm).fit()

    ferro = LinearRegression()
    ferro.fit(X, y)

    return ferro, sm_model, X, y


# ---------------------------------------------------------------------------
# 1. R-squared
# ---------------------------------------------------------------------------

def test_r_squared_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    assert_allclose(ferro.r_squared(), sm_model.rsquared, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Adjusted R-squared
# ---------------------------------------------------------------------------

def test_adjusted_r_squared_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    assert_allclose(ferro.adjusted_r_squared(), sm_model.rsquared_adj, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. F-statistic
# ---------------------------------------------------------------------------

def test_f_statistic_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    f_val, p_val = ferro.f_statistic()
    assert_allclose(f_val, sm_model.fvalue, rtol=1e-6)
    assert_allclose(p_val, sm_model.f_pvalue, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Coefficients
# ---------------------------------------------------------------------------

def test_coefficients_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    ci_info = ferro.coefficients_with_ci()
    # FerroML: intercept first, then slopes; statsmodels: const first, then slopes
    ferro_coefs = np.array([c["estimate"] for c in ci_info])
    assert_allclose(ferro_coefs, sm_model.params, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Standard errors
# ---------------------------------------------------------------------------

def test_standard_errors_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    ci_info = ferro.coefficients_with_ci()
    ferro_se = np.array([c["std_error"] for c in ci_info])
    assert_allclose(ferro_se, sm_model.bse, atol=1e-8)


# ---------------------------------------------------------------------------
# 6. t-statistics
# ---------------------------------------------------------------------------

def test_t_statistics_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    ci_info = ferro.coefficients_with_ci()
    ferro_t = np.array([c["t_statistic"] for c in ci_info])
    assert_allclose(ferro_t, sm_model.tvalues, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. p-values
# ---------------------------------------------------------------------------

def test_p_values_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    ci_info = ferro.coefficients_with_ci()
    ferro_p = np.array([c["p_value"] for c in ci_info])
    assert_allclose(ferro_p, sm_model.pvalues, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. Confidence intervals
# ---------------------------------------------------------------------------

def test_confidence_intervals_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    ci_info = ferro.coefficients_with_ci()
    ferro_lower = np.array([c["ci_lower"] for c in ci_info])
    ferro_upper = np.array([c["ci_upper"] for c in ci_info])
    sm_ci = sm_model.conf_int()  # shape (k, 2)
    assert_allclose(ferro_lower, sm_ci[:, 0], atol=1e-4)
    assert_allclose(ferro_upper, sm_ci[:, 1], atol=1e-4)


# ---------------------------------------------------------------------------
# 9. Residuals
# ---------------------------------------------------------------------------

def test_residuals_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    assert_allclose(ferro.residuals(), sm_model.resid, atol=1e-10)


# ---------------------------------------------------------------------------
# 10. Durbin-Watson
# ---------------------------------------------------------------------------

def test_durbin_watson_vs_statsmodels(ols_models):
    ferro, sm_model, X, y = ols_models
    diag_str = ferro.diagnostics()
    # Parse Durbin-Watson value from diagnostics string
    match = re.search(r"Durbin[- ]Watson[:\s]+([0-9.]+)", diag_str, re.IGNORECASE)
    assert match is not None, f"Could not find Durbin-Watson in diagnostics:\n{diag_str}"
    ferro_dw = float(match.group(1))
    sm_dw = durbin_watson(sm_model.resid)
    assert_allclose(ferro_dw, sm_dw, atol=1e-3)


# ---------------------------------------------------------------------------
# 11. Prediction interval coverage (Monte Carlo)
# ---------------------------------------------------------------------------

def test_prediction_interval_coverage():
    coverage_rates = []
    for seed in range(500):
        rng = np.random.RandomState(seed)
        n = 50
        X_train = rng.randn(n, 2)
        y_train = 2 * X_train[:, 0] - X_train[:, 1] + rng.randn(n) * 1.0

        X_test = rng.randn(5, 2)
        y_test = 2 * X_test[:, 0] - X_test[:, 1] + rng.randn(5) * 1.0

        m = LinearRegression()
        m.fit(X_train, y_train)
        result = m.predict_interval(X_test, level=0.95)

        # predict_interval returns (predictions, lower, upper)
        _, lowers, uppers = result
        lowers = np.asarray(lowers)
        uppers = np.asarray(uppers)

        covered = np.sum((y_test >= lowers) & (y_test <= uppers))
        coverage_rates.append(covered / len(y_test))

    mean_coverage = np.mean(coverage_rates)
    # 95% PI should cover ~95% of the time; allow [0.90, 1.0]
    assert 0.90 <= mean_coverage <= 1.0, (
        f"Mean prediction interval coverage = {mean_coverage:.4f}, expected ~0.95"
    )


# ---------------------------------------------------------------------------
# 12. Perfect fit: R² = 1.0
# ---------------------------------------------------------------------------

def test_perfect_fit_r_squared_one():
    np.random.seed(99)
    X = np.random.randn(50, 2)
    y = 4.0 * X[:, 0] - 3.0 * X[:, 1] + 7.0  # exact linear, no noise
    m = LinearRegression()
    m.fit(X, y)
    assert_allclose(m.r_squared(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# 13. No relationship: R² near zero
# ---------------------------------------------------------------------------

def test_no_relationship_r_squared_near_zero():
    np.random.seed(123)
    X = np.random.randn(500, 3)
    y = np.random.randn(500)  # completely independent
    m = LinearRegression()
    m.fit(X, y)
    assert abs(m.r_squared()) < 0.05, f"R² = {m.r_squared()}, expected near 0"
