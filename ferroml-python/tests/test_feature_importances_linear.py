"""Tests for feature_importances_ on linear models.

Verifies that linear models return normalized absolute coefficient magnitudes
matching sklearn's convention: abs(coef) / sum(abs(coef)).
"""

import numpy as np
import pytest

from ferroml.linear import (
    ElasticNet,
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
)
from ferroml.ensemble import SGDClassifier, SGDRegressor


def test_linear_regression_feature_importances_match_manual():
    """feature_importances_ should equal abs(coef) / sum(abs(coef))."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + rng.randn(100) * 0.1

    model = LinearRegression()
    model.fit(X, y)

    coef = model.coef_
    expected = np.abs(coef) / np.sum(np.abs(coef))
    actual = model.feature_importances_

    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    np.testing.assert_allclose(actual.sum(), 1.0, atol=1e-10)


def test_lasso_sparsity_zeros_in_importances():
    """Lasso with high alpha should zero out features, reflected in importances."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    # Only first 2 features matter
    y = 5.0 * X[:, 0] + 3.0 * X[:, 1] + rng.randn(200) * 0.1

    model = LassoRegression(alpha=0.5)
    model.fit(X, y)

    importances = model.feature_importances_
    coef = model.coef_

    # Check that zero coefficients correspond to zero importances
    zero_mask = np.abs(coef) < 1e-10
    if zero_mask.any():
        np.testing.assert_allclose(importances[zero_mask], 0.0, atol=1e-10)

    # Importances should sum to 1 (or 0 if all coef are zero)
    if np.sum(np.abs(coef)) > 0:
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-10)


def test_ridge_feature_importances_sum_to_one():
    """Ridge importances should sum to 1."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = X[:, 0] + 2 * X[:, 1] + rng.randn(100) * 0.1

    model = RidgeRegression(alpha=1.0)
    model.fit(X, y)

    importances = model.feature_importances_
    np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-10)
    assert (importances >= 0).all()


def test_elastic_net_feature_importances_sum_to_one():
    """ElasticNet importances should sum to 1."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = X[:, 0] + 2 * X[:, 1] + rng.randn(100) * 0.1

    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X, y)

    importances = model.feature_importances_
    np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-10)
    assert (importances >= 0).all()


def test_logistic_regression_feature_importances():
    """LogisticRegression importances should be normalized abs coef."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = (X[:, 0] + 2 * X[:, 1] > 0).astype(float)

    model = LogisticRegression()
    model.fit(X, y)

    coef = model.coef_
    expected = np.abs(coef) / np.sum(np.abs(coef))
    actual = model.feature_importances_

    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    np.testing.assert_allclose(actual.sum(), 1.0, atol=1e-10)


def test_sgd_classifier_feature_importances():
    """SGDClassifier importances should be normalized and sum to 1."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 4)
    y = (X[:, 0] + 2 * X[:, 1] > 0).astype(float)

    model = SGDClassifier(loss="log", random_state=42, max_iter=500)
    model.fit(X, y)

    importances = model.feature_importances_
    np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-10)
    assert (importances >= 0).all()
    assert len(importances) == 4


def test_sgd_regressor_feature_importances():
    """SGDRegressor importances should be normalized and sum to 1."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = 3.0 * X[:, 0] - 1.0 * X[:, 1] + rng.randn(200) * 0.1

    model = SGDRegressor(random_state=42, max_iter=500)
    model.fit(X, y)

    importances = model.feature_importances_
    np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-10)
    assert (importances >= 0).all()
    assert len(importances) == 3


def test_feature_importances_not_fitted_raises():
    """Accessing feature_importances_ before fit should raise."""
    model = LinearRegression()
    with pytest.raises(ValueError, match="not fitted"):
        _ = model.feature_importances_

    model2 = SGDClassifier()
    with pytest.raises(ValueError, match="not fitted"):
        _ = model2.feature_importances_

    model3 = SGDRegressor()
    with pytest.raises(ValueError, match="not fitted"):
        _ = model3.feature_importances_
