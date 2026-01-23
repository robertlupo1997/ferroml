"""
Shared fixtures for FerroML Python tests.

This module provides:
- Sample datasets (regression, classification, multiclass)
- Model factories for common test patterns
- Utility functions for testing
"""

import numpy as np
import pytest
from typing import Tuple


# ============================================================================
# Dataset Fixtures
# ============================================================================


@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simple regression data.

    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + noise

    Returns:
        Tuple of (X, y) with shape (100, 3) and (100,)
    """
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def classification_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simple binary classification data.

    Linearly separable with decision boundary at X[:, 0] + X[:, 1] = 0

    Returns:
        Tuple of (X, y) with shape (100, 3) and (100,)
    """
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def multiclass_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multiclass classification data (3 classes).

    Returns:
        Tuple of (X, y) with shape (150, 4) and (150,)
    """
    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 4)
    y = np.zeros(n_samples)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = 2
    return X, y.astype(np.float64)


@pytest.fixture
def large_regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate larger regression dataset for performance tests.

    Returns:
        Tuple of (X, y) with shape (10000, 10) and (10000,)
    """
    np.random.seed(42)
    n_samples = 10000
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = X @ np.random.randn(n_features) + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def large_classification_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate larger classification dataset for performance tests.

    Returns:
        Tuple of (X, y) with shape (10000, 10) and (10000,)
    """
    np.random.seed(42)
    n_samples = 10000
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def iris_like_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate iris-like dataset (3 classes, 4 features).

    Returns:
        Tuple of (X, y) with shape (150, 4) and (150,)
    """
    np.random.seed(42)
    n_per_class = 50
    X_list = []
    y_list = []

    # Class 0: centered at [0, 0, 0, 0]
    X_list.append(np.random.randn(n_per_class, 4) * 0.5)
    y_list.append(np.zeros(n_per_class))

    # Class 1: centered at [2, 2, 2, 2]
    X_list.append(np.random.randn(n_per_class, 4) * 0.5 + 2)
    y_list.append(np.ones(n_per_class))

    # Class 2: centered at [4, 4, 4, 4]
    X_list.append(np.random.randn(n_per_class, 4) * 0.5 + 4)
    y_list.append(np.full(n_per_class, 2.0))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


@pytest.fixture
def diabetes_like_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate diabetes-like regression dataset (10 features).

    Returns:
        Tuple of (X, y) with shape (442, 10) and (442,)
    """
    np.random.seed(42)
    n_samples = 442
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    # Create a known relationship
    true_coef = np.array([0.5, -0.3, 0.8, -0.2, 0.4, -0.1, 0.3, -0.4, 0.2, -0.5])
    y = X @ true_coef + 150 + np.random.randn(n_samples) * 20
    return X, y


@pytest.fixture
def data_with_nan() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data with NaN values for imputation tests.

    Returns:
        Tuple of (X, y) with ~10% NaN values in X
    """
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1

    # Introduce ~10% NaN values
    nan_mask = np.random.random((n_samples, 4)) < 0.1
    X[nan_mask] = np.nan

    return X, y


@pytest.fixture
def perfect_linear_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate perfect linear data (no noise) for exact coefficient tests.

    y = 2*X + 1

    Returns:
        Tuple of (X, y) with shape (5, 1) and (5,)
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # y = 2x + 1
    return X, y


# ============================================================================
# Model Factory Fixtures
# ============================================================================


@pytest.fixture
def all_regressors():
    """
    Return a list of all regressor classes with default parameters.
    """
    from ferroml.linear import (
        LinearRegression,
        RidgeRegression,
        LassoRegression,
        ElasticNet,
    )
    from ferroml.trees import (
        DecisionTreeRegressor,
        RandomForestRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
    )

    return [
        ("LinearRegression", LinearRegression()),
        ("RidgeRegression", RidgeRegression(alpha=1.0)),
        ("LassoRegression", LassoRegression(alpha=0.1)),
        ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5)),
        ("DecisionTreeRegressor", DecisionTreeRegressor(max_depth=5, random_state=42)),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)),
        ("HistGradientBoostingRegressor", HistGradientBoostingRegressor(max_iter=10, max_depth=3, random_state=42)),
    ]


@pytest.fixture
def all_classifiers():
    """
    Return a list of all classifier classes with default parameters.
    """
    from ferroml.linear import LogisticRegression
    from ferroml.trees import (
        DecisionTreeClassifier,
        RandomForestClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
    )

    return [
        ("LogisticRegression", LogisticRegression()),
        ("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
        ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)),
        ("HistGradientBoostingClassifier", HistGradientBoostingClassifier(max_iter=10, max_depth=3, random_state=42)),
    ]


@pytest.fixture
def all_scalers():
    """
    Return a list of all scaler classes.
    """
    from ferroml.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        MaxAbsScaler,
    )

    return [
        ("StandardScaler", StandardScaler()),
        ("MinMaxScaler", MinMaxScaler()),
        ("RobustScaler", RobustScaler()),
        ("MaxAbsScaler", MaxAbsScaler()),
    ]


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def assert_array_equal():
    """
    Return numpy's assert_array_equal for convenience.
    """
    return np.testing.assert_array_equal


@pytest.fixture
def assert_allclose():
    """
    Return numpy's assert_allclose for convenience.
    """
    return np.testing.assert_allclose


# ============================================================================
# Pytest Markers
# ============================================================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "sklearn_compat: marks tests that require sklearn"
    )
    config.addinivalue_line(
        "markers", "automl: marks AutoML integration tests"
    )
