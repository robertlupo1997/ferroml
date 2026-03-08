"""Tests for RidgeCV, LassoCV, ElasticNetCV, and RidgeClassifier Python bindings."""

import numpy as np
import pytest

from ferroml.linear import RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

def make_regression_data():
    """Simple regression dataset with known relationship."""
    rng = np.random.RandomState(42)
    X = np.column_stack([np.linspace(0, 10, 50), rng.randn(50)])
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(50) * 0.1
    return X, y


def make_binary_classification_data():
    """Binary classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


def make_multiclass_data():
    """3-class classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(90, 3)
    y = np.zeros(90)
    y[X[:, 0] > 0.5] = 1.0
    y[X[:, 0] < -0.5] = 2.0
    return X, y


# ===========================================================================
# RidgeCV Tests
# ===========================================================================

class TestRidgeCV:
    def test_fit_predict_defaults(self):
        X, y = make_regression_data()
        model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_fit_returns_self(self):
        X, y = make_regression_data()
        model = RidgeCV(alphas=[0.01, 0.1, 1.0])
        result = model.fit(X, y)
        assert result is model

    def test_alpha_getter_after_fit(self):
        X, y = make_regression_data()
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        model.fit(X, y)
        alpha = model.alpha_
        assert isinstance(alpha, float)
        assert alpha > 0

    def test_alpha_getter_before_fit_raises(self):
        model = RidgeCV()
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.alpha_

    def test_predict_before_fit_raises(self):
        X, _ = make_regression_data()
        model = RidgeCV()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_custom_alphas(self):
        X, y = make_regression_data()
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        model.fit(X, y)
        alpha = model.alpha_
        assert alpha in [0.01, 0.1, 1.0, 10.0]

    def test_custom_cv(self):
        X, y = make_regression_data()
        model = RidgeCV(alphas=[0.01, 0.1, 1.0], cv=3)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_repr(self):
        model = RidgeCV(cv=3)
        r = repr(model)
        assert "RidgeCV" in r
        assert "3" in r

    def test_good_predictions(self):
        """RidgeCV should produce reasonable predictions on linear data."""
        X, y = make_regression_data()
        model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0])
        model.fit(X, y)
        preds = model.predict(X)
        # R^2 should be high for nearly-linear data
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.99


# ===========================================================================
# LassoCV Tests
# ===========================================================================

class TestLassoCV:
    def test_fit_predict(self):
        X, y = make_regression_data()
        model = LassoCV()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_fit_returns_self(self):
        X, y = make_regression_data()
        model = LassoCV()
        result = model.fit(X, y)
        assert result is model

    def test_alpha_getter_after_fit(self):
        X, y = make_regression_data()
        model = LassoCV()
        model.fit(X, y)
        alpha = model.alpha_
        assert isinstance(alpha, float)
        assert alpha > 0

    def test_alpha_getter_before_fit_raises(self):
        model = LassoCV()
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.alpha_

    def test_predict_before_fit_raises(self):
        X, _ = make_regression_data()
        model = LassoCV()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_custom_n_alphas(self):
        X, y = make_regression_data()
        model = LassoCV(n_alphas=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_custom_cv(self):
        X, y = make_regression_data()
        model = LassoCV(n_alphas=50, cv=3)
        model.fit(X, y)
        assert model.alpha_ > 0

    def test_repr(self):
        model = LassoCV(n_alphas=50, cv=3)
        r = repr(model)
        assert "LassoCV" in r
        assert "50" in r
        assert "3" in r


# ===========================================================================
# ElasticNetCV Tests
# ===========================================================================

class TestElasticNetCV:
    def test_fit_predict_defaults(self):
        X, y = make_regression_data()
        model = ElasticNetCV()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_fit_returns_self(self):
        X, y = make_regression_data()
        model = ElasticNetCV()
        result = model.fit(X, y)
        assert result is model

    def test_alpha_getter_after_fit(self):
        X, y = make_regression_data()
        model = ElasticNetCV()
        model.fit(X, y)
        alpha = model.alpha_
        assert isinstance(alpha, float)
        assert alpha > 0

    def test_l1_ratio_getter_after_fit(self):
        X, y = make_regression_data()
        model = ElasticNetCV()
        model.fit(X, y)
        ratio = model.l1_ratio_
        assert isinstance(ratio, float)
        assert 0 <= ratio <= 1

    def test_alpha_getter_before_fit_raises(self):
        model = ElasticNetCV()
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.alpha_

    def test_l1_ratio_getter_before_fit_raises(self):
        model = ElasticNetCV()
        with pytest.raises(ValueError, match="not fitted"):
            _ = model.l1_ratio_

    def test_predict_before_fit_raises(self):
        X, _ = make_regression_data()
        model = ElasticNetCV()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_custom_l1_ratios(self):
        X, y = make_regression_data()
        model = ElasticNetCV(l1_ratios=[0.1, 0.5, 0.9])
        model.fit(X, y)
        ratio = model.l1_ratio_
        assert ratio in [0.1, 0.5, 0.9]

    def test_custom_n_alphas_and_cv(self):
        X, y = make_regression_data()
        model = ElasticNetCV(n_alphas=20, cv=3)
        model.fit(X, y)
        assert model.alpha_ > 0
        assert 0 <= model.l1_ratio_ <= 1

    def test_repr(self):
        model = ElasticNetCV(n_alphas=50, cv=3)
        r = repr(model)
        assert "ElasticNetCV" in r
        assert "50" in r
        assert "3" in r


# ===========================================================================
# RidgeClassifier Tests
# ===========================================================================

class TestRidgeClassifier:
    def test_binary_fit_predict(self):
        X, y = make_binary_classification_data()
        model = RidgeClassifier()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)
        # Predictions should be one of the two classes
        unique_preds = set(preds)
        assert unique_preds.issubset({0.0, 1.0})

    def test_fit_returns_self(self):
        X, y = make_binary_classification_data()
        model = RidgeClassifier()
        result = model.fit(X, y)
        assert result is model

    def test_multiclass_fit_predict(self):
        X, y = make_multiclass_data()
        model = RidgeClassifier()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (90,)
        unique_preds = set(preds)
        assert unique_preds.issubset({0.0, 1.0, 2.0})

    def test_predict_before_fit_raises(self):
        X, _ = make_binary_classification_data()
        model = RidgeClassifier()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_custom_alpha(self):
        X, y = make_binary_classification_data()
        model = RidgeClassifier(alpha=10.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)

    def test_no_intercept(self):
        X, y = make_binary_classification_data()
        model = RidgeClassifier(alpha=1.0, fit_intercept=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60,)

    def test_binary_accuracy(self):
        """RidgeClassifier should achieve decent accuracy on separable data."""
        X, y = make_binary_classification_data()
        model = RidgeClassifier()
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.8

    def test_integer_labels(self):
        """RidgeClassifier should accept integer label arrays."""
        X, y = make_binary_classification_data()
        y_int = y.astype(np.int32)
        model = RidgeClassifier()
        model.fit(X, y_int)
        preds = model.predict(X)
        assert preds.shape == (60,)

    def test_repr(self):
        model = RidgeClassifier(alpha=0.5, fit_intercept=False)
        r = repr(model)
        assert "RidgeClassifier" in r
        assert "0.5" in r
        assert "false" in r.lower() or "False" in r
