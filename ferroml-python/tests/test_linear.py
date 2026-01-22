"""Test FerroML linear models."""

import numpy as np
import pytest

from ferroml.linear import (
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNet,
)


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    # Simple linear decision boundary
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


class TestLinearRegression:
    """Tests for LinearRegression."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        # Should have reasonable predictions (not all zeros or constant)
        assert np.std(predictions) > 0.1

    def test_coefficients(self, regression_data):
        """Test coefficient access."""
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)

        coef = model.coef_
        assert coef.shape == (3,)
        # Check coefficients are roughly correct (true: [2, 3, -1])
        assert abs(coef[0] - 2.0) < 0.5
        assert abs(coef[1] - 3.0) < 0.5
        assert abs(coef[2] - (-1.0)) < 0.5

    def test_intercept(self, regression_data):
        """Test intercept access."""
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)

        intercept = model.intercept_
        # Intercept should be close to 0 since data was generated without one
        assert abs(intercept) < 0.5

    def test_r_squared(self, regression_data):
        """Test R-squared calculation."""
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)

        r2 = model.r_squared()
        # Should have high R-squared for this simple linear problem
        assert 0.9 < r2 <= 1.0

    def test_summary(self, regression_data):
        """Test summary output."""
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)

        summary = model.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key statistical terms
        assert "R-squared" in summary or "R²" in summary or "Coefficient" in summary.lower()


class TestLogisticRegression:
    """Tests for LogisticRegression."""

    def test_fit_predict(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        model = LogisticRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        # Should only predict 0 or 1
        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_predict_proba(self, classification_data):
        """Test probability predictions."""
        X, y = classification_data
        model = LogisticRegression()
        model.fit(X, y)
        proba = model.predict_proba(X)

        # Probabilities should be 2D (n_samples, 2 classes)
        assert proba.ndim == 2
        assert proba.shape[0] == X.shape[0]
        # Each row should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(X.shape[0]))
        # Probabilities should be between 0 and 1
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_coefficients(self, classification_data):
        """Test coefficient access."""
        X, y = classification_data
        model = LogisticRegression()
        model.fit(X, y)

        coef = model.coef_
        assert coef.shape[0] == 3


class TestRidgeRegression:
    """Tests for RidgeRegression."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict with regularization."""
        X, y = regression_data
        model = RidgeRegression(alpha=1.0)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_regularization_effect(self, regression_data):
        """Test that regularization shrinks coefficients."""
        X, y = regression_data

        model_low_reg = RidgeRegression(alpha=0.01)
        model_low_reg.fit(X, y)

        model_high_reg = RidgeRegression(alpha=100.0)
        model_high_reg.fit(X, y)

        # Higher regularization should give smaller coefficients
        coef_low = np.abs(model_low_reg.coef_)
        coef_high = np.abs(model_high_reg.coef_)

        assert np.sum(coef_high) < np.sum(coef_low)


class TestLassoRegression:
    """Tests for LassoRegression."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict with L1 regularization."""
        X, y = regression_data
        model = LassoRegression(alpha=0.1)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_sparsity(self):
        """Test that Lasso can produce sparse solutions."""
        np.random.seed(42)
        n_samples = 100
        # Generate data with only first 2 features being relevant
        X = np.random.randn(n_samples, 10)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.1

        model = LassoRegression(alpha=0.5)
        model.fit(X, y)

        # With high regularization, many coefficients should be near zero
        coef = model.coef_
        n_near_zero = np.sum(np.abs(coef) < 0.1)
        assert n_near_zero >= 3  # At least some coefficients should be shrunk


class TestElasticNet:
    """Tests for ElasticNet."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict with combined L1/L2 regularization."""
        X, y = regression_data
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_l1_ratio_effect(self, regression_data):
        """Test different l1_ratio values."""
        X, y = regression_data

        # More L1-like (l1_ratio close to 1)
        model_l1 = ElasticNet(alpha=0.1, l1_ratio=0.9)
        model_l1.fit(X, y)

        # More L2-like (l1_ratio close to 0)
        model_l2 = ElasticNet(alpha=0.1, l1_ratio=0.1)
        model_l2.fit(X, y)

        # Both should produce valid predictions
        assert model_l1.predict(X).shape == y.shape
        assert model_l2.predict(X).shape == y.shape
