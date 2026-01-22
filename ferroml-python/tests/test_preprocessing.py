"""Test FerroML preprocessing transformers."""

import numpy as np
import pytest

from ferroml.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    SimpleImputer,
)


@pytest.fixture
def numeric_data():
    """Generate numeric data for scaling tests."""
    np.random.seed(42)
    return np.random.randn(100, 4) * 10 + 5  # mean ~5, std ~10


@pytest.fixture
def categorical_data():
    """Generate categorical data for encoding tests."""
    np.random.seed(42)
    n_samples = 100
    # Create categorical data as float indices
    return np.random.randint(0, 3, size=(n_samples, 2)).astype(np.float64)


class TestStandardScaler:
    """Tests for StandardScaler."""

    def test_fit_transform(self, numeric_data):
        """Test fit_transform produces standardized data."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        # Mean should be approximately 0
        means = X_scaled.mean(axis=0)
        np.testing.assert_array_almost_equal(means, np.zeros(4), decimal=10)

        # Std should be approximately 1
        stds = X_scaled.std(axis=0, ddof=0)
        np.testing.assert_array_almost_equal(stds, np.ones(4), decimal=10)

    def test_transform_separate(self, numeric_data):
        """Test fit and transform can be called separately."""
        scaler = StandardScaler()
        scaler.fit(numeric_data)
        X_scaled = scaler.transform(numeric_data)

        means = X_scaled.mean(axis=0)
        np.testing.assert_array_almost_equal(means, np.zeros(4), decimal=10)

    def test_inverse_transform(self, numeric_data):
        """Test inverse_transform recovers original data."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_data)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X_recovered, numeric_data, decimal=10)


class TestMinMaxScaler:
    """Tests for MinMaxScaler."""

    def test_fit_transform(self, numeric_data):
        """Test fit_transform produces data in [0, 1] range."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        # Min should be 0, max should be 1
        assert X_scaled.min() >= 0 - 1e-10
        assert X_scaled.max() <= 1 + 1e-10

    def test_custom_range(self, numeric_data):
        """Test custom feature_range parameter."""
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        X_scaled = scaler.fit_transform(numeric_data)

        assert X_scaled.min() >= -1 - 1e-10
        assert X_scaled.max() <= 1 + 1e-10

    def test_inverse_transform(self, numeric_data):
        """Test inverse_transform recovers original data."""
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(numeric_data)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X_recovered, numeric_data, decimal=10)


class TestRobustScaler:
    """Tests for RobustScaler."""

    def test_fit_transform(self, numeric_data):
        """Test fit_transform produces scaled data."""
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        # Median should be approximately 0
        medians = np.median(X_scaled, axis=0)
        np.testing.assert_array_almost_equal(medians, np.zeros(4), decimal=10)

    def test_robust_to_outliers(self):
        """Test that RobustScaler is robust to outliers."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Add extreme outliers
        X[0, 0] = 1000
        X[1, 1] = -1000

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Most values should still be in a reasonable range
        non_outlier_mask = np.abs(X_scaled) < 10
        assert non_outlier_mask.sum() > 180  # Most of the 200 values

    def test_inverse_transform(self, numeric_data):
        """Test inverse_transform recovers original data."""
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(numeric_data)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X_recovered, numeric_data, decimal=10)


class TestMaxAbsScaler:
    """Tests for MaxAbsScaler."""

    def test_fit_transform(self, numeric_data):
        """Test fit_transform produces data in [-1, 1] range."""
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(numeric_data)

        # Values should be in [-1, 1]
        assert X_scaled.min() >= -1 - 1e-10
        assert X_scaled.max() <= 1 + 1e-10
        # At least one value per column should have absolute value ~1
        max_abs_per_col = np.abs(X_scaled).max(axis=0)
        np.testing.assert_array_almost_equal(max_abs_per_col, np.ones(4), decimal=10)

    def test_inverse_transform(self, numeric_data):
        """Test inverse_transform recovers original data."""
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(numeric_data)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X_recovered, numeric_data, decimal=10)


class TestOneHotEncoder:
    """Tests for OneHotEncoder."""

    def test_fit_transform(self, categorical_data):
        """Test fit_transform produces one-hot encoded data."""
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(categorical_data)

        # Should have more columns than original (one per category per feature)
        assert X_encoded.shape[1] > categorical_data.shape[1]
        # Values should be 0 or 1
        assert set(np.unique(X_encoded)).issubset({0.0, 1.0})

    def test_each_row_has_ones(self, categorical_data):
        """Test that each original feature has exactly one 1 per row."""
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(categorical_data)

        # Total number of 1s per row should equal number of original features
        ones_per_row = X_encoded.sum(axis=1)
        np.testing.assert_array_equal(
            ones_per_row, np.full(categorical_data.shape[0], categorical_data.shape[1])
        )


class TestOrdinalEncoder:
    """Tests for OrdinalEncoder."""

    def test_fit_transform(self, categorical_data):
        """Test fit_transform produces ordinal encoded data."""
        encoder = OrdinalEncoder()
        X_encoded = encoder.fit_transform(categorical_data)

        # Should have same shape as input
        assert X_encoded.shape == categorical_data.shape
        # Values should be non-negative integers (as floats)
        assert np.all(X_encoded >= 0)
        assert np.all(X_encoded == X_encoded.astype(int))


class TestLabelEncoder:
    """Tests for LabelEncoder."""

    def test_fit_transform(self):
        """Test fit_transform produces encoded labels."""
        encoder = LabelEncoder()
        y = np.array([2.0, 0.0, 1.0, 2.0, 0.0, 1.0])
        y_encoded = encoder.fit_transform(y)

        # Should have same shape as input
        assert y_encoded.shape == y.shape
        # Values should be 0, 1, 2
        assert set(np.unique(y_encoded)).issubset({0.0, 1.0, 2.0})

    def test_inverse_transform(self):
        """Test inverse_transform recovers original labels."""
        encoder = LabelEncoder()
        y = np.array([2.0, 0.0, 1.0, 2.0, 0.0, 1.0])
        y_encoded = encoder.fit_transform(y)
        y_recovered = encoder.inverse_transform(y_encoded)

        np.testing.assert_array_equal(y_recovered, y)


class TestSimpleImputer:
    """Tests for SimpleImputer."""

    def test_mean_imputation(self):
        """Test mean imputation strategy."""
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [3.0, np.nan],
            [4.0, 5.0],
        ])
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # No NaN values should remain
        assert not np.any(np.isnan(X_imputed))
        # Mean of [1, 3, 4] = 8/3 ≈ 2.667 for first column
        assert abs(X_imputed[1, 0] - (1 + 3 + 4) / 3) < 0.01
        # Mean of [2, 3, 5] = 10/3 ≈ 3.333 for second column
        assert abs(X_imputed[2, 1] - (2 + 3 + 5) / 3) < 0.01

    def test_median_imputation(self):
        """Test median imputation strategy."""
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [3.0, np.nan],
            [4.0, 5.0],
            [5.0, 6.0],
        ])
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)

        # No NaN values should remain
        assert not np.any(np.isnan(X_imputed))
        # Median of [1, 3, 4, 5] = 3.5 for first column
        assert abs(X_imputed[1, 0] - 3.5) < 0.01

    def test_constant_imputation(self):
        """Test constant imputation strategy."""
        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [3.0, np.nan],
        ])
        imputer = SimpleImputer(strategy='constant', fill_value=-999.0)
        X_imputed = imputer.fit_transform(X)

        # No NaN values should remain
        assert not np.any(np.isnan(X_imputed))
        # Missing values should be -999
        assert X_imputed[1, 0] == -999.0
        assert X_imputed[2, 1] == -999.0
