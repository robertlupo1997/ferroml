"""
Phase M.1: Side-by-side comparison of FerroML vs sklearn linear models.

Tests all 13 linear models on real and synthetic datasets, comparing
predictions, coefficients, R², and accuracy within documented tolerances.
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from conftest_comparison import (
    get_iris, get_wine, get_breast_cancer, get_diabetes,
    get_classification_data, get_regression_data,
    r2_score, accuracy_score,
)


# ============================================================================
# Helpers
# ============================================================================

def scale_data(X_train, X_test=None):
    """StandardScaler for algorithms that need it."""
    sc = SklearnStandardScaler()
    X_train_s = sc.fit_transform(X_train)
    if X_test is not None:
        return X_train_s, sc.transform(X_test)
    return X_train_s


# ============================================================================
# 1. LinearRegression
# ============================================================================

class TestLinearRegressionComparison:
    """FerroML LinearRegression vs sklearn LinearRegression."""

    def _fit_both(self, X, y):
        from ferroml.linear import LinearRegression as FerroLR
        from sklearn.linear_model import LinearRegression as SkLR

        ferro = FerroLR()
        ferro.fit(X, y)
        sk = SkLR()
        sk.fit(X, y)
        return ferro, sk

    def test_diabetes_predictions(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y)
        fp = ferro.predict(X)
        sp = sk.predict(X)
        np.testing.assert_allclose(fp, sp, atol=1e-6)

    def test_diabetes_coefficients(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y)
        np.testing.assert_allclose(ferro.coef_, sk.coef_, atol=1e-6)
        np.testing.assert_allclose(ferro.intercept_, sk.intercept_, atol=1e-6)

    def test_diabetes_r2(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y)
        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert abs(ferro_r2 - sk_r2) < 1e-8

    def test_synthetic_regression(self):
        X, y = get_regression_data(n=500, p=10)
        ferro, sk = self._fit_both(X, y)
        np.testing.assert_allclose(ferro.predict(X), sk.predict(X), atol=1e-5)
        np.testing.assert_allclose(ferro.coef_, sk.coef_, atol=1e-5)


# ============================================================================
# 2. LogisticRegression
# ============================================================================

class TestLogisticRegressionComparison:
    """FerroML LogisticRegression vs sklearn LogisticRegression."""

    def _fit_both(self, X, y, **ferro_kw):
        from ferroml.linear import LogisticRegression as FerroLR
        from sklearn.linear_model import LogisticRegression as SkLR

        ferro = FerroLR(**ferro_kw)
        ferro.fit(X, y)
        sk = SkLR(max_iter=1000, solver='lbfgs')
        sk.fit(X, y)
        return ferro, sk

    def test_iris_binary_accuracy(self):
        """FerroML LogisticRegression is binary-only; use first 2 iris classes."""
        X, y = get_iris()
        mask = y < 2
        X, y = X[mask], y[mask]
        ferro, sk = self._fit_both(X, y)
        ferro_acc = accuracy_score(y, ferro.predict(X))
        sk_acc = accuracy_score(y, sk.predict(X))
        assert ferro_acc > 0.90, f"FerroML accuracy too low: {ferro_acc}"
        assert abs(ferro_acc - sk_acc) < 0.05

    def test_breast_cancer_accuracy(self):
        X, y = get_breast_cancer()
        X_s = scale_data(X)
        ferro, sk = self._fit_both(X_s, y)
        ferro_acc = accuracy_score(y, ferro.predict(X_s))
        sk_acc = accuracy_score(y, sk.predict(X_s))
        assert ferro_acc > 0.90
        assert abs(ferro_acc - sk_acc) < 0.05

    def test_breast_cancer_proba(self):
        X, y = get_breast_cancer()
        X_s = scale_data(X)
        ferro, sk = self._fit_both(X_s, y)
        fp = ferro.predict_proba(X_s)
        sp = sk.predict_proba(X_s)
        assert fp.shape == sp.shape
        np.testing.assert_allclose(fp.sum(axis=1), 1.0, atol=1e-10)
        # Predicted classes should mostly agree
        ferro_classes = np.argmax(fp, axis=1).astype(float)
        sk_classes = np.argmax(sp, axis=1).astype(float)
        agreement = np.mean(ferro_classes == sk_classes)
        assert agreement > 0.95

    def test_synthetic_binary_accuracy(self):
        X, y = get_classification_data(n=500, p=10, n_classes=2)
        X_s = scale_data(X)
        ferro, sk = self._fit_both(X_s, y)
        ferro_acc = accuracy_score(y, ferro.predict(X_s))
        sk_acc = accuracy_score(y, sk.predict(X_s))
        assert ferro_acc > 0.80
        assert abs(ferro_acc - sk_acc) < 0.05


# ============================================================================
# 3. RidgeRegression
# ============================================================================

class TestRidgeRegressionComparison:
    """FerroML RidgeRegression vs sklearn Ridge."""

    def _fit_both(self, X, y, alpha=1.0):
        from ferroml.linear import RidgeRegression as FerroRidge
        from sklearn.linear_model import Ridge as SkRidge

        ferro = FerroRidge(alpha=alpha)
        ferro.fit(X, y)
        sk = SkRidge(alpha=alpha)
        sk.fit(X, y)
        return ferro, sk

    def test_diabetes_predictions(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y)
        np.testing.assert_allclose(ferro.predict(X), sk.predict(X), atol=1e-5)

    def test_diabetes_coefficients(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y)
        np.testing.assert_allclose(ferro.coef_, sk.coef_, atol=1e-5)

    def test_synthetic_alpha_sweep(self):
        X, y = get_regression_data(n=500, p=10)
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            ferro, sk = self._fit_both(X, y, alpha=alpha)
            ferro_r2 = r2_score(y, ferro.predict(X))
            sk_r2 = sk.score(X, y)
            assert abs(ferro_r2 - sk_r2) < 1e-4, f"R² mismatch at alpha={alpha}"


# ============================================================================
# 4. LassoRegression
# ============================================================================

class TestLassoRegressionComparison:
    """FerroML LassoRegression vs sklearn Lasso."""

    def _fit_both(self, X, y, alpha=0.1):
        from ferroml.linear import LassoRegression as FerroLasso
        from sklearn.linear_model import Lasso as SkLasso

        ferro = FerroLasso(alpha=alpha)
        ferro.fit(X, y)
        sk = SkLasso(alpha=alpha, max_iter=10000)
        sk.fit(X, y)
        return ferro, sk

    def test_diabetes_predictions(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y, alpha=1.0)
        fp = ferro.predict(X)
        sp = sk.predict(X)
        np.testing.assert_allclose(fp, sp, atol=1.0)  # Lasso coordinate descent can differ

    def test_diabetes_r2(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y, alpha=1.0)
        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert abs(ferro_r2 - sk_r2) < 0.05

    def test_sparsity_pattern(self):
        """Both should zero out similar coefficients."""
        X, y = get_regression_data(n=500, p=20)
        ferro, sk = self._fit_both(X, y, alpha=1.0)
        # Check similar number of nonzero coefficients
        ferro_nz = np.sum(np.abs(ferro.coef_) > 1e-6)
        sk_nz = np.sum(np.abs(sk.coef_) > 1e-6)
        assert abs(ferro_nz - sk_nz) <= 3, f"Sparsity mismatch: ferro={ferro_nz}, sk={sk_nz}"


# ============================================================================
# 5. ElasticNet
# ============================================================================

class TestElasticNetComparison:
    """FerroML ElasticNet vs sklearn ElasticNet."""

    def _fit_both(self, X, y, alpha=0.1, l1_ratio=0.5):
        from ferroml.linear import ElasticNet as FerroEN
        from sklearn.linear_model import ElasticNet as SkEN

        ferro = FerroEN(alpha=alpha, l1_ratio=l1_ratio)
        ferro.fit(X, y)
        sk = SkEN(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
        sk.fit(X, y)
        return ferro, sk

    def test_diabetes_predictions(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y, alpha=0.5)
        fp = ferro.predict(X)
        sp = sk.predict(X)
        np.testing.assert_allclose(fp, sp, atol=1.0)

    def test_diabetes_r2(self):
        X, y = get_diabetes()
        ferro, sk = self._fit_both(X, y, alpha=0.5)
        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert abs(ferro_r2 - sk_r2) < 0.05

    def test_l1_ratio_sweep(self):
        X, y = get_regression_data(n=500, p=10)
        for l1_ratio in [0.1, 0.5, 0.9]:
            ferro, sk = self._fit_both(X, y, alpha=0.5, l1_ratio=l1_ratio)
            ferro_r2 = r2_score(y, ferro.predict(X))
            sk_r2 = sk.score(X, y)
            assert abs(ferro_r2 - sk_r2) < 0.05, f"R² mismatch at l1_ratio={l1_ratio}"


# ============================================================================
# 6. RobustRegression
# ============================================================================

class TestRobustRegressionComparison:
    """FerroML RobustRegression vs sklearn HuberRegressor."""

    def test_with_outliers(self):
        from ferroml.linear import RobustRegression as FerroRobust
        from sklearn.linear_model import HuberRegressor as SkHuber

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + rng.randn(100) * 0.5
        # Add outliers
        y[:5] = 100.0
        y[-5:] = -100.0

        ferro = FerroRobust(estimator="huber")
        ferro.fit(X, y)
        sk = SkHuber(max_iter=1000)
        sk.fit(X, y)

        # Both should be robust — predict on clean data
        X_clean = rng.randn(50, 3)
        y_clean = 2 * X_clean[:, 0] + 3 * X_clean[:, 1] - X_clean[:, 2]

        ferro_r2 = r2_score(y_clean, ferro.predict(X_clean))
        sk_r2 = r2_score(y_clean, sk.predict(X_clean))
        assert ferro_r2 > 0.7, f"FerroML R² too low: {ferro_r2}"
        assert abs(ferro_r2 - sk_r2) < 0.15

    def test_no_outliers(self):
        from ferroml.linear import RobustRegression as FerroRobust
        from sklearn.linear_model import HuberRegressor as SkHuber

        X, y = get_regression_data(n=200, p=5)
        ferro = FerroRobust()
        ferro.fit(X, y)
        sk = SkHuber(max_iter=1000)
        sk.fit(X, y)

        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert abs(ferro_r2 - sk_r2) < 0.10


# ============================================================================
# 7. QuantileRegression
# ============================================================================

class TestQuantileRegressionComparison:
    """FerroML QuantileRegression vs sklearn QuantileRegressor."""

    def test_median_regression(self):
        from ferroml.linear import QuantileRegression as FerroQR
        from sklearn.linear_model import QuantileRegressor as SkQR

        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = 2 * X[:, 0] + X[:, 1] + rng.randn(200) * 0.5

        ferro = FerroQR(quantile=0.5)
        ferro.fit(X, y)
        sk = SkQR(quantile=0.5, alpha=0.0, solver='highs')
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        # Both should give reasonable median estimates
        ferro_r2 = r2_score(y, fp)
        sk_r2 = r2_score(y, sp)
        assert ferro_r2 > 0.7
        assert abs(ferro_r2 - sk_r2) < 0.10

    def test_upper_quantile(self):
        from ferroml.linear import QuantileRegression as FerroQR
        from sklearn.linear_model import QuantileRegressor as SkQR

        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = 2 * X[:, 0] + rng.randn(200) * 1.0

        ferro = FerroQR(quantile=0.9)
        ferro.fit(X, y)
        sk = SkQR(quantile=0.9, alpha=0.0, solver='highs')
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        # Both predictions should be above most true values
        ferro_above = np.mean(fp > y)
        sk_above = np.mean(sp > y)
        assert abs(ferro_above - sk_above) < 0.10


# ============================================================================
# 8. Perceptron
# ============================================================================

class TestPerceptronComparison:
    """FerroML Perceptron vs sklearn Perceptron."""

    def test_linearly_separable(self):
        from ferroml.linear import Perceptron as FerroPerceptron
        from sklearn.linear_model import Perceptron as SkPerceptron

        # Create linearly separable data
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)

        ferro = FerroPerceptron()
        ferro.fit(X, y)
        sk = SkPerceptron(random_state=42, max_iter=1000)
        sk.fit(X, y)

        ferro_acc = accuracy_score(y, ferro.predict(X))
        sk_acc = accuracy_score(y, sk.predict(X).astype(np.float64))
        assert ferro_acc > 0.85, f"FerroML accuracy: {ferro_acc}"
        assert abs(ferro_acc - sk_acc) < 0.10

    def test_iris_subset(self):
        from ferroml.linear import Perceptron as FerroPerceptron
        from sklearn.linear_model import Perceptron as SkPerceptron

        # Use first 2 classes of iris (linearly separable)
        X, y = get_iris()
        mask = y < 2
        X, y = X[mask], y[mask]
        X_s = scale_data(X)

        ferro = FerroPerceptron()
        ferro.fit(X_s, y)
        sk = SkPerceptron(random_state=42, max_iter=1000)
        sk.fit(X_s, y)

        ferro_acc = accuracy_score(y, ferro.predict(X_s))
        sk_acc = accuracy_score(y, sk.predict(X_s).astype(np.float64))
        assert ferro_acc > 0.90
        assert abs(ferro_acc - sk_acc) < 0.10


# ============================================================================
# 9. RidgeCV
# ============================================================================

class TestRidgeCVComparison:
    """FerroML RidgeCV vs sklearn RidgeCV."""

    def test_diabetes_alpha_selection(self):
        from ferroml.linear import RidgeCV as FerroRidgeCV
        from sklearn.linear_model import RidgeCV as SkRidgeCV

        X, y = get_diabetes()
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        ferro = FerroRidgeCV(alphas=alphas)
        ferro.fit(X, y)
        sk = SkRidgeCV(alphas=alphas)
        sk.fit(X, y)

        # Both should select a reasonable alpha
        # (May not be identical due to CV strategy differences)
        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert ferro_r2 > 0.4
        assert abs(ferro_r2 - sk_r2) < 0.05

    def test_synthetic_r2(self):
        from ferroml.linear import RidgeCV as FerroRidgeCV
        from sklearn.linear_model import RidgeCV as SkRidgeCV

        X, y = get_regression_data(n=500, p=10)
        alphas = [0.01, 0.1, 1.0, 10.0]

        ferro = FerroRidgeCV(alphas=alphas)
        ferro.fit(X, y)
        sk = SkRidgeCV(alphas=alphas)
        sk.fit(X, y)

        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert abs(ferro_r2 - sk_r2) < 0.05


# ============================================================================
# 10. LassoCV
# ============================================================================

class TestLassoCVComparison:
    """FerroML LassoCV vs sklearn LassoCV."""

    def test_diabetes_r2(self):
        from ferroml.linear import LassoCV as FerroLassoCV
        from sklearn.linear_model import LassoCV as SkLassoCV

        X, y = get_diabetes()

        ferro = FerroLassoCV()
        ferro.fit(X, y)
        sk = SkLassoCV(max_iter=10000, cv=5)
        sk.fit(X, y)

        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert ferro_r2 > 0.4
        assert abs(ferro_r2 - sk_r2) < 0.10

    def test_synthetic_r2(self):
        from ferroml.linear import LassoCV as FerroLassoCV
        from sklearn.linear_model import LassoCV as SkLassoCV

        X, y = get_regression_data(n=500, p=10)

        ferro = FerroLassoCV()
        ferro.fit(X, y)
        sk = SkLassoCV(max_iter=10000, cv=5)
        sk.fit(X, y)

        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert abs(ferro_r2 - sk_r2) < 0.10


# ============================================================================
# 11. ElasticNetCV
# ============================================================================

class TestElasticNetCVComparison:
    """FerroML ElasticNetCV vs sklearn ElasticNetCV."""

    def test_diabetes_r2(self):
        from ferroml.linear import ElasticNetCV as FerroENCV
        from sklearn.linear_model import ElasticNetCV as SkENCV

        X, y = get_diabetes()

        ferro = FerroENCV()
        ferro.fit(X, y)
        sk = SkENCV(max_iter=10000, cv=5)
        sk.fit(X, y)

        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert ferro_r2 > 0.4
        assert abs(ferro_r2 - sk_r2) < 0.10

    def test_synthetic_r2(self):
        from ferroml.linear import ElasticNetCV as FerroENCV
        from sklearn.linear_model import ElasticNetCV as SkENCV

        X, y = get_regression_data(n=500, p=10)

        ferro = FerroENCV()
        ferro.fit(X, y)
        sk = SkENCV(max_iter=10000, cv=5)
        sk.fit(X, y)

        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert abs(ferro_r2 - sk_r2) < 0.10


# ============================================================================
# 12. RidgeClassifier
# ============================================================================

class TestRidgeClassifierComparison:
    """FerroML RidgeClassifier vs sklearn RidgeClassifier."""

    def test_iris_accuracy(self):
        from ferroml.linear import RidgeClassifier as FerroRC
        from sklearn.linear_model import RidgeClassifier as SkRC

        X, y = get_iris()
        ferro = FerroRC()
        ferro.fit(X, y)
        sk = SkRC()
        sk.fit(X, y)

        ferro_acc = accuracy_score(y, ferro.predict(X))
        sk_acc = accuracy_score(y, sk.predict(X).astype(np.float64))
        assert ferro_acc > 0.85
        assert abs(ferro_acc - sk_acc) < 0.05

    def test_wine_accuracy(self):
        from ferroml.linear import RidgeClassifier as FerroRC
        from sklearn.linear_model import RidgeClassifier as SkRC

        X, y = get_wine()
        X_s = scale_data(X)
        ferro = FerroRC()
        ferro.fit(X_s, y)
        sk = SkRC()
        sk.fit(X_s, y)

        ferro_acc = accuracy_score(y, ferro.predict(X_s))
        sk_acc = accuracy_score(y, sk.predict(X_s).astype(np.float64))
        assert ferro_acc > 0.85
        assert abs(ferro_acc - sk_acc) < 0.05


# ============================================================================
# 13. IsotonicRegression
# ============================================================================

class TestIsotonicRegressionComparison:
    """FerroML IsotonicRegression vs sklearn IsotonicRegression."""

    def test_monotonic_data(self):
        from ferroml.linear import IsotonicRegression as FerroIso
        from sklearn.isotonic import IsotonicRegression as SkIso

        rng = np.random.RandomState(42)
        n = 100
        X = np.sort(rng.uniform(0, 10, n))
        y = X + rng.randn(n) * 0.5  # Monotonically increasing with noise

        # FerroML expects 2D X
        X_2d = X.reshape(-1, 1)

        ferro = FerroIso()
        ferro.fit(X_2d, y)
        sk = SkIso()
        sk.fit(X, y)

        fp = ferro.predict(X_2d)
        sp = sk.predict(X)

        # Both should produce monotonically non-decreasing output
        assert np.all(np.diff(fp) >= -1e-10), "FerroML not monotonic"
        assert np.all(np.diff(sp) >= -1e-10), "sklearn not monotonic"

        # Predictions should be close
        np.testing.assert_allclose(fp, sp, atol=1e-4)

    def test_step_function(self):
        from ferroml.linear import IsotonicRegression as FerroIso
        from sklearn.isotonic import IsotonicRegression as SkIso

        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 5.0, 6.0, 7.0])

        X_2d = X.reshape(-1, 1)

        ferro = FerroIso()
        ferro.fit(X_2d, y)
        sk = SkIso()
        sk.fit(X, y)

        fp = ferro.predict(X_2d)
        sp = sk.predict(X)

        # Both should produce monotonic output
        assert np.all(np.diff(fp) >= -1e-10)
        np.testing.assert_allclose(fp, sp, atol=1e-4)

    def test_already_monotonic(self):
        from ferroml.linear import IsotonicRegression as FerroIso
        from sklearn.isotonic import IsotonicRegression as SkIso

        X = np.arange(1.0, 11.0)
        y = X * 2.0  # Already perfectly monotonic

        X_2d = X.reshape(-1, 1)

        ferro = FerroIso()
        ferro.fit(X_2d, y)
        sk = SkIso()
        sk.fit(X, y)

        fp = ferro.predict(X_2d)
        sp = sk.predict(X)
        np.testing.assert_allclose(fp, sp, atol=1e-6)


# ============================================================================
# Cross-model: All regressors on same data
# ============================================================================

class TestAllRegressorsConsistency:
    """Verify all FerroML regressors produce competitive R² on diabetes."""

    @pytest.mark.parametrize("model_name,ferro_cls,ferro_kw,sk_cls,sk_kw", [
        ("LinearRegression",
         "ferroml.linear.LinearRegression", {},
         "sklearn.linear_model.LinearRegression", {}),
        ("RidgeRegression",
         "ferroml.linear.RidgeRegression", {"alpha": 1.0},
         "sklearn.linear_model.Ridge", {"alpha": 1.0}),
        ("LassoRegression",
         "ferroml.linear.LassoRegression", {"alpha": 1.0},
         "sklearn.linear_model.Lasso", {"alpha": 1.0, "max_iter": 10000}),
        ("ElasticNet",
         "ferroml.linear.ElasticNet", {"alpha": 0.5, "l1_ratio": 0.5},
         "sklearn.linear_model.ElasticNet", {"alpha": 0.5, "l1_ratio": 0.5, "max_iter": 10000}),
    ])
    def test_regressor_r2_on_diabetes(self, model_name, ferro_cls, ferro_kw, sk_cls, sk_kw):
        import importlib
        X, y = get_diabetes()

        # Import and instantiate
        ferro_mod, ferro_name = ferro_cls.rsplit(".", 1)
        FerroClass = getattr(importlib.import_module(ferro_mod), ferro_name)
        sk_mod, sk_name = sk_cls.rsplit(".", 1)
        SkClass = getattr(importlib.import_module(sk_mod), sk_name)

        ferro = FerroClass(**ferro_kw)
        ferro.fit(X, y)
        sk = SkClass(**sk_kw)
        sk.fit(X, y)

        ferro_r2 = r2_score(y, ferro.predict(X))
        sk_r2 = sk.score(X, y)
        assert abs(ferro_r2 - sk_r2) < 0.05, (
            f"{model_name}: ferro R²={ferro_r2:.4f}, sklearn R²={sk_r2:.4f}"
        )


class TestAllClassifiersConsistency:
    """Verify all FerroML classifiers produce competitive accuracy on iris."""

    @pytest.mark.parametrize("model_name,ferro_cls,ferro_kw,sk_cls,sk_kw,tol,binary_only", [
        ("LogisticRegression",
         "ferroml.linear.LogisticRegression", {},
         "sklearn.linear_model.LogisticRegression", {"max_iter": 1000},
         0.05, True),
        ("RidgeClassifier",
         "ferroml.linear.RidgeClassifier", {},
         "sklearn.linear_model.RidgeClassifier", {},
         0.05, False),
        ("Perceptron",
         "ferroml.linear.Perceptron", {},
         "sklearn.linear_model.Perceptron", {"random_state": 42, "max_iter": 1000},
         0.15, False),
    ])
    def test_classifier_accuracy_on_iris(self, model_name, ferro_cls, ferro_kw, sk_cls, sk_kw, tol, binary_only):
        import importlib
        X, y = get_iris()
        if binary_only:
            mask = y < 2
            X, y = X[mask], y[mask]
        X_s = scale_data(X)

        ferro_mod, ferro_name = ferro_cls.rsplit(".", 1)
        FerroClass = getattr(importlib.import_module(ferro_mod), ferro_name)
        sk_mod, sk_name = sk_cls.rsplit(".", 1)
        SkClass = getattr(importlib.import_module(sk_mod), sk_name)

        ferro = FerroClass(**ferro_kw)
        ferro.fit(X_s, y)
        sk = SkClass(**sk_kw)
        sk.fit(X_s, y)

        ferro_acc = accuracy_score(y, ferro.predict(X_s))
        sk_acc = accuracy_score(y, sk.predict(X_s).astype(np.float64))
        assert abs(ferro_acc - sk_acc) < tol, (
            f"{model_name}: ferro acc={ferro_acc:.4f}, sklearn acc={sk_acc:.4f}"
        )
