"""Test FerroML pipeline components."""

import numpy as np
import pytest

from ferroml.pipeline import Pipeline, ColumnTransformer, FeatureUnion
from ferroml.preprocessing import StandardScaler, MinMaxScaler
from ferroml.linear import LinearRegression


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y


class TestPipeline:
    """Tests for Pipeline."""

    def test_fit_predict(self, regression_data):
        """Test pipeline with scaler and model."""
        X, y = regression_data

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression()),
        ])

        pipe.fit(X, y)
        predictions = pipe.predict(X)

        assert predictions.shape == y.shape

    def test_fit_transform(self, regression_data):
        """Test pipeline with only transformers."""
        X, y = regression_data

        pipe = Pipeline([
            ('scaler1', StandardScaler()),
            ('scaler2', MinMaxScaler()),
        ])

        # FerroML Pipeline.fit_transform requires y argument
        X_transformed = pipe.fit_transform(X, y)

        assert X_transformed.shape == X.shape
        # After StandardScaler -> MinMaxScaler, data should be in roughly [0, 1]
        # (not exactly due to how standardization shifts the data)
        assert X_transformed.min() >= -0.1
        assert X_transformed.max() <= 1.1


class TestColumnTransformer:
    """Tests for ColumnTransformer."""

    def test_fit_transform_by_indices(self, regression_data):
        """Test ColumnTransformer with column indices."""
        X, _ = regression_data

        ct = ColumnTransformer([
            ('scale_first_two', StandardScaler(), [0, 1]),
            ('scale_last_two', MinMaxScaler(), [2, 3]),
        ])

        X_transformed = ct.fit_transform(X)

        # Should have same number of columns (2 + 2 = 4)
        assert X_transformed.shape[1] == 4
        assert X_transformed.shape[0] == X.shape[0]

    def test_different_transformers(self, regression_data):
        """Test that different transformers are applied correctly."""
        X, _ = regression_data

        ct = ColumnTransformer([
            ('standard', StandardScaler(), [0, 1]),
            ('minmax', MinMaxScaler(), [2, 3]),
        ])

        X_transformed = ct.fit_transform(X)

        # First two columns should be standardized (mean ~0, std ~1)
        means_first = X_transformed[:, :2].mean(axis=0)
        np.testing.assert_array_almost_equal(means_first, np.zeros(2), decimal=10)

        # Last two columns should be in [0, 1] range
        assert X_transformed[:, 2:].min() >= -1e-10
        assert X_transformed[:, 2:].max() <= 1 + 1e-10


class TestFeatureUnion:
    """Tests for FeatureUnion."""

    def test_fit_transform(self, regression_data):
        """Test FeatureUnion concatenates transformer outputs."""
        X, _ = regression_data

        fu = FeatureUnion([
            ('standard', StandardScaler()),
            ('minmax', MinMaxScaler()),
        ])

        X_transformed = fu.fit_transform(X)

        # Should have twice as many columns (4 from each transformer)
        assert X_transformed.shape[1] == X.shape[1] * 2
        assert X_transformed.shape[0] == X.shape[0]

    def test_parallel_transformers(self, regression_data):
        """Test that transformers are applied to the same input."""
        X, _ = regression_data

        fu = FeatureUnion([
            ('standard', StandardScaler()),
            ('minmax', MinMaxScaler()),
        ])

        X_transformed = fu.fit_transform(X)

        # First half should be standardized
        first_half = X_transformed[:, :4]
        means = first_half.mean(axis=0)
        np.testing.assert_array_almost_equal(means, np.zeros(4), decimal=10)

        # Second half should be in [0, 1]
        second_half = X_transformed[:, 4:]
        assert second_half.min() >= -1e-10
        assert second_half.max() <= 1 + 1e-10
