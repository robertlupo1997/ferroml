import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats as scipy_stats
from ferroml.linear import LinearRegression


class TestTDistributionViaOLS:
    """Verify t-distribution functions via OLS coefficient inference."""

    def test_known_slope_significance(self):
        """True slope=2, noise=0.1: t-stat should be huge, p≈0."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = 2.0 * X[:, 0] + np.random.randn(100) * 0.1
        m = LinearRegression()
        m.fit(X, y)
        ci = m.coefficients_with_ci()
        slope_info = ci[1]  # index 1 is first slope
        assert slope_info['t_statistic'] > 50  # Huge t-value
        assert slope_info['p_value'] < 1e-50   # Extremely significant

    def test_zero_slope_not_significant(self):
        """True slope=0: p-value should be > 0.05 (usually)."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.random.randn(100)  # No relationship
        m = LinearRegression()
        m.fit(X, y)
        ci = m.coefficients_with_ci()
        slope_info = ci[1]
        assert slope_info['p_value'] > 0.01  # Should not be significant

    def test_t_statistic_equals_estimate_over_se(self):
        """t = estimate / std_error for all coefficients."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + np.random.randn(100)
        m = LinearRegression()
        m.fit(X, y)
        for coef in m.coefficients_with_ci():
            expected_t = coef['estimate'] / coef['std_error']
            assert_allclose(coef['t_statistic'], expected_t, atol=1e-10)

    def test_ci_contains_true_value(self):
        """95% CI should contain the true coefficient (deterministic check)."""
        np.random.seed(42)
        true_slopes = [3.0, -2.0, 0.5]
        X = np.random.randn(500, 3)
        y = sum(s * X[:, i] for i, s in enumerate(true_slopes)) + np.random.randn(500) * 0.3
        m = LinearRegression()
        m.fit(X, y)
        ci = m.coefficients_with_ci()
        for i, true_val in enumerate(true_slopes):
            coef = ci[i + 1]  # skip intercept
            assert coef['ci_lower'] <= true_val <= coef['ci_upper'], \
                f"CI [{coef['ci_lower']}, {coef['ci_upper']}] doesn't contain {true_val}"

    def test_ci_width_shrinks_with_n(self):
        """Larger n → narrower CI (more precision)."""
        widths = []
        for n in [50, 200, 1000]:
            np.random.seed(42)
            X = np.random.randn(n, 1)
            y = 2.0 * X[:, 0] + np.random.randn(n)
            m = LinearRegression()
            m.fit(X, y)
            ci = m.coefficients_with_ci()
            width = ci[1]['ci_upper'] - ci[1]['ci_lower']
            widths.append(width)
        assert widths[0] > widths[1] > widths[2]


class TestFDistributionViaOLS:
    """Verify F-distribution via OLS F-statistic."""

    def test_f_statistic_vs_scipy(self):
        """F-stat from FerroML vs manual computation via scipy."""
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = 2*X[:, 0] - X[:, 1] + np.random.randn(n) * 0.5

        m = LinearRegression()
        m.fit(X, y)
        f_ferro, p_ferro = m.f_statistic()

        # Manual F-test
        y_pred = m.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_reg = ss_tot - ss_res
        f_manual = (ss_reg / p) / (ss_res / (n - p - 1))
        p_manual = 1 - scipy_stats.f.cdf(f_manual, p, n - p - 1)

        assert_allclose(f_ferro, f_manual, atol=1e-6)
        assert_allclose(p_ferro, p_manual, atol=1e-6)

    def test_f_statistic_perfect_fit(self):
        """Perfect linear fit: F should be enormous."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x
        m = LinearRegression()
        m.fit(X, y)
        f, p = m.f_statistic()
        assert f > 1e10
        assert p < 1e-10


class TestPValueCalibration:
    """Verify p-values are uniformly distributed under H0."""

    def test_uniform_pvalues_under_null(self):
        """Under H0 (no effect), p-values should be ~Uniform(0,1)."""
        np.random.seed(42)
        p_values = []
        for _ in range(500):
            X = np.random.randn(50, 1)
            y = np.random.randn(50)  # No relationship
            m = LinearRegression()
            m.fit(X, y)
            ci = m.coefficients_with_ci()
            p_values.append(ci[1]['p_value'])

        p_arr = np.array(p_values)
        # Under H0, ~5% should be < 0.05
        rejection_rate = np.mean(p_arr < 0.05)
        assert 0.02 <= rejection_rate <= 0.10, \
            f"Type I error rate {rejection_rate} outside [0.02, 0.10]"

        # KS test for uniformity
        ks_stat, ks_p = scipy_stats.kstest(p_arr, 'uniform')
        assert ks_p > 0.01, f"P-values not uniform: KS p={ks_p}"


class TestEffectSizeAnalytical:
    """Verify effect size calculations against known formulas."""

    def test_r_squared_interpretation(self):
        """R²=0.5 means model explains 50% of variance."""
        np.random.seed(42)
        X = np.random.randn(1000, 1)
        noise = np.random.randn(1000)
        y = X[:, 0] + noise  # signal = noise → R² ≈ 0.5
        m = LinearRegression()
        m.fit(X, y)
        assert_allclose(m.r_squared(), 0.5, atol=0.05)

    def test_adjusted_r_squared_penalizes_features(self):
        """Adding noise features: adj R² should decrease or stay flat."""
        np.random.seed(42)
        X_real = np.random.randn(100, 2)
        y = X_real[:, 0] + np.random.randn(100) * 0.5

        # With just real features
        m1 = LinearRegression()
        m1.fit(X_real, y)

        # With 10 noise features added
        X_noise = np.column_stack([X_real, np.random.randn(100, 10)])
        m2 = LinearRegression()
        m2.fit(X_noise, y)

        assert m2.adjusted_r_squared() <= m1.adjusted_r_squared() + 0.02
