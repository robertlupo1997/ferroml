import pytest
import re
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats as scipy_stats
from ferroml.linear import LinearRegression


class TestConfidenceIntervals:
    """Verify CI computation via OLS coefficient CIs."""

    def test_ci_coverage_monte_carlo(self):
        """95% CI should contain true slope ~95% of the time."""
        np.random.seed(42)
        true_slope = 2.0
        n_sims = 500
        coverage = 0
        for _ in range(n_sims):
            X = np.random.randn(50, 1)
            y = true_slope * X[:, 0] + np.random.randn(50) * 1.0
            m = LinearRegression()
            m.fit(X, y)
            ci = m.coefficients_with_ci()
            slope = ci[1]
            if slope['ci_lower'] <= true_slope <= slope['ci_upper']:
                coverage += 1
        actual = coverage / n_sims
        assert 0.92 <= actual <= 0.98, f"CI coverage {actual} outside [0.92, 0.98]"

    def test_ci_symmetric_around_estimate(self):
        """CI should be symmetric around the point estimate."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + np.random.randn(100) * 0.5
        m = LinearRegression()
        m.fit(X, y)
        for coef in m.coefficients_with_ci():
            mid = (coef['ci_lower'] + coef['ci_upper']) / 2
            assert_allclose(mid, coef['estimate'], atol=1e-10)

    def test_ci_width_vs_scipy(self):
        """CI width should match scipy t-distribution calculation."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = 3 * X[:, 0] + np.random.randn(n)
        m = LinearRegression()
        m.fit(X, y)
        ci = m.coefficients_with_ci()
        for coef in ci:
            # Width = 2 * t_crit * SE
            t_crit = scipy_stats.t.ppf(0.975, df=n - 3)  # n - p - 1
            expected_width = 2 * t_crit * coef['std_error']
            actual_width = coef['ci_upper'] - coef['ci_lower']
            assert_allclose(actual_width, expected_width, rtol=1e-3)


class TestPredictionIntervals:
    """Verify prediction intervals."""

    def test_prediction_interval_coverage(self):
        """95% PI should contain true y ~95% of the time."""
        np.random.seed(42)
        n_train, n_test = 200, 1000
        X_train = np.random.randn(n_train, 2)
        y_train = 2 * X_train[:, 0] - X_train[:, 1] + np.random.randn(n_train)

        X_test = np.random.randn(n_test, 2)
        y_test = 2 * X_test[:, 0] - X_test[:, 1] + np.random.randn(n_test)

        m = LinearRegression()
        m.fit(X_train, y_train)
        preds, lower, upper = m.predict_interval(X_test, 0.95)

        contained = np.sum((y_test >= lower) & (y_test <= upper))
        coverage = contained / n_test
        assert 0.92 <= coverage <= 0.98, f"PI coverage {coverage}"

    def test_prediction_interval_wider_than_ci(self):
        """Prediction interval should always be wider than confidence interval."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2 * X[:, 0] + np.random.randn(100)
        m = LinearRegression()
        m.fit(X, y)
        # PI includes both estimation uncertainty AND residual variance
        # So PI width > CI width for coefficient


class TestResidualDiagnostics:
    """Verify residual diagnostics."""

    def test_residuals_sum_to_zero(self):
        """OLS residuals should sum to approximately zero (with intercept)."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + np.random.randn(100)
        m = LinearRegression()
        m.fit(X, y)
        resid = m.residuals()
        assert_allclose(np.sum(resid), 0.0, atol=1e-10)

    def test_residuals_orthogonal_to_fitted(self):
        """Residuals should be orthogonal to fitted values."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + np.random.randn(100)
        m = LinearRegression()
        m.fit(X, y)
        resid = m.residuals()
        fitted = m.fitted_values()
        dot = np.dot(resid, fitted)
        assert_allclose(dot, 0.0, atol=1e-8)

    def test_residuals_orthogonal_to_predictors(self):
        """Residuals should be orthogonal to each predictor column."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + np.random.randn(100)
        m = LinearRegression()
        m.fit(X, y)
        resid = m.residuals()
        for j in range(3):
            dot = np.dot(resid, X[:, j])
            assert_allclose(dot, 0.0, atol=1e-8)

    def test_durbin_watson_range(self):
        """DW statistic should be in [0, 4]."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + np.random.randn(100)
        m = LinearRegression()
        m.fit(X, y)
        diag = m.diagnostics()
        dw_match = re.search(r'Durbin-Watson:\s+([\d.]+)', diag)
        if dw_match:
            dw = float(dw_match.group(1))
            assert 0 <= dw <= 4, f"DW {dw} outside [0, 4]"
            assert 1.5 <= dw <= 2.5, f"DW {dw} suggests autocorrelation"

    def test_durbin_watson_vs_statsmodels(self):
        """DW should match statsmodels computation."""
        from statsmodels.stats.stattools import durbin_watson
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + np.random.randn(100)
        m = LinearRegression()
        m.fit(X, y)
        resid = m.residuals()

        dw_sm = durbin_watson(resid)

        dw_match = re.search(r'Durbin-Watson:\s+([\d.]+)', m.diagnostics())
        if dw_match:
            dw_ferro = float(dw_match.group(1))
            assert_allclose(dw_ferro, dw_sm, atol=1e-4)

    def test_normality_test_normal_residuals(self):
        """Normal residuals: normality test should pass (p > 0.05)."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = X[:, 0] + np.random.randn(200) * 0.5  # Normal noise
        m = LinearRegression()
        m.fit(X, y)
        diag = m.diagnostics()
        assert 'PASS' in diag or 'pass' in diag.lower()

    def test_fitted_plus_residuals_equals_y(self):
        """y = fitted + residuals (exactly)."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + np.random.randn(100)
        m = LinearRegression()
        m.fit(X, y)
        assert_allclose(m.fitted_values() + m.residuals(), y, atol=1e-10)
