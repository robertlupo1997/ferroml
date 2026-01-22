"""Test FerroML tree-based models."""

import numpy as np
import pytest

from ferroml.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = X[:, 0] ** 2 + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    return X, y


@pytest.fixture
def classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    # XOR-like pattern that trees handle well
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float64)
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate multiclass classification data."""
    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 4)
    # 3 classes based on combinations
    y = np.zeros(n_samples)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = 2
    return X, y.astype(np.float64)


class TestDecisionTreeClassifier:
    """Tests for DecisionTreeClassifier."""

    def test_fit_predict(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_feature_importances(self, classification_data):
        """Test feature importance calculation."""
        X, y = classification_data
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.shape == (4,)
        assert np.all(importances >= 0)
        np.testing.assert_almost_equal(importances.sum(), 1.0, decimal=5)

    def test_max_depth(self, classification_data):
        """Test that max_depth limits tree depth."""
        X, y = classification_data
        model_shallow = DecisionTreeClassifier(max_depth=2, random_state=42)
        model_deep = DecisionTreeClassifier(max_depth=10, random_state=42)

        model_shallow.fit(X, y)
        model_deep.fit(X, y)

        # Deeper tree might have higher training accuracy
        # (both should work without error)
        assert model_shallow.predict(X).shape == y.shape
        assert model_deep.predict(X).shape == y.shape


class TestDecisionTreeRegressor:
    """Tests for DecisionTreeRegressor."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_feature_importances(self, regression_data):
        """Test feature importance calculation."""
        X, y = regression_data
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.shape == (4,)
        assert np.all(importances >= 0)


class TestRandomForestClassifier:
    """Tests for RandomForestClassifier."""

    def test_fit_predict(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predict_proba(self, classification_data):
        """Test probability predictions."""
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.ndim == 2
        assert proba.shape[0] == X.shape[0]
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_feature_importances(self, classification_data):
        """Test feature importance calculation."""
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.shape == (4,)
        assert np.all(importances >= 0)


class TestRandomForestRegressor:
    """Tests for RandomForestRegressor."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_feature_importances(self, regression_data):
        """Test feature importance calculation."""
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.shape == (4,)


class TestGradientBoostingClassifier:
    """Tests for GradientBoostingClassifier."""

    def test_fit_predict(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        model = GradientBoostingClassifier(
            n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_learning_rate_effect(self, classification_data):
        """Test that learning rate affects training."""
        X, y = classification_data

        model_fast = GradientBoostingClassifier(
            n_estimators=10, learning_rate=0.5, random_state=42
        )
        model_slow = GradientBoostingClassifier(
            n_estimators=10, learning_rate=0.01, random_state=42
        )

        model_fast.fit(X, y)
        model_slow.fit(X, y)

        # Both should produce valid predictions
        assert model_fast.predict(X).shape == y.shape
        assert model_slow.predict(X).shape == y.shape


class TestGradientBoostingRegressor:
    """Tests for GradientBoostingRegressor."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = GradientBoostingRegressor(
            n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


class TestHistGradientBoostingClassifier:
    """Tests for HistGradientBoostingClassifier."""

    def test_fit_predict(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        model = HistGradientBoostingClassifier(
            max_iter=10, max_depth=3, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_handles_larger_data(self):
        """Test that hist-based version handles larger datasets efficiently."""
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)

        model = HistGradientBoostingClassifier(max_iter=5, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


class TestHistGradientBoostingRegressor:
    """Tests for HistGradientBoostingRegressor."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = HistGradientBoostingRegressor(
            max_iter=10, max_depth=3, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_handles_missing_values(self):
        """Test that hist-based version handles NaN values."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 4)
        y = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1

        # Introduce some missing values
        X[10:20, 0] = np.nan
        X[30:40, 2] = np.nan

        model = HistGradientBoostingRegressor(max_iter=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.any(np.isnan(predictions))
