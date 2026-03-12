"""Tests for Gaussian Process regression and classification."""

import numpy as np
import pytest


def test_gpr_basic_sin():
    """GPR should fit sin(x) and predict accurately at training points."""
    from ferroml.gaussian_process import GaussianProcessRegressor, RBF

    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = np.sin(X).ravel()

    gpr = GaussianProcessRegressor(kernel=RBF(1.0))
    gpr.fit(X, y)
    preds = gpr.predict(X)

    assert preds.shape == (20,)
    np.testing.assert_allclose(preds, y, atol=1e-3)


def test_gpr_predict_with_std_shape():
    """predict_with_std should return mean and std arrays of correct shape."""
    from ferroml.gaussian_process import GaussianProcessRegressor, RBF

    X = np.linspace(0, 5, 15).reshape(-1, 1)
    y = np.sin(X).ravel()

    gpr = GaussianProcessRegressor(kernel=RBF(1.0))
    gpr.fit(X, y)

    X_new = np.array([[0.5], [2.5], [4.5]])
    mean, std = gpr.predict_with_std(X_new)

    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert np.all(std >= 0)


def test_gpr_predict_with_std_small_near_training():
    """Std should be very small at training points."""
    from ferroml.gaussian_process import GaussianProcessRegressor, RBF

    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = np.sin(X).ravel()

    gpr = GaussianProcessRegressor(kernel=RBF(1.0))
    gpr.fit(X, y)

    mean, std = gpr.predict_with_std(X)
    assert np.all(std < 1e-3), f"Max std at training points: {std.max()}"


def test_gpr_with_matern_kernel():
    """GPR should work with Matern kernel."""
    from ferroml.gaussian_process import GaussianProcessRegressor, Matern

    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = np.sin(X).ravel()

    gpr = GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=2.5))
    gpr.fit(X, y)
    preds = gpr.predict(X)

    assert preds.shape == (20,)
    np.testing.assert_allclose(preds, y, atol=1e-2)


def test_gpr_different_alpha():
    """Different alpha values should produce different predictions."""
    from ferroml.gaussian_process import GaussianProcessRegressor, RBF

    X = np.array([[0], [1], [2], [3], [4]], dtype=np.float64)
    y = np.array([0.0, 10.0, 0.0, 10.0, 0.0])

    gpr_small = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-10)
    gpr_small.fit(X, y)
    preds_small = gpr_small.predict(X)

    gpr_large = GaussianProcessRegressor(kernel=RBF(1.0), alpha=10.0)
    gpr_large.fit(X, y)
    preds_large = gpr_large.predict(X)

    # Larger alpha = smoother = smaller range
    range_small = preds_small.max() - preds_small.min()
    range_large = preds_large.max() - preds_large.min()
    assert range_large < range_small


def test_gpc_binary_classification():
    """GPC should classify linearly separable data correctly."""
    from ferroml.gaussian_process import GaussianProcessClassifier, RBF

    X = np.array([
        [1, 1], [2, 1], [1, 2], [2, 2], [1.5, 1.5],
        [6, 6], [7, 6], [6, 7], [7, 7], [6.5, 6.5],
    ], dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    gpc = GaussianProcessClassifier(kernel=RBF(1.0))
    gpc.fit(X, y)
    preds = gpc.predict(X)

    assert preds.shape == (10,)
    np.testing.assert_array_equal(preds, y)


def test_gpc_predict_proba():
    """predict_proba should return valid probabilities."""
    from ferroml.gaussian_process import GaussianProcessClassifier, RBF

    X = np.array([
        [1, 1], [2, 1], [1, 2],
        [6, 6], [7, 6], [6, 7],
    ], dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    gpc = GaussianProcessClassifier(kernel=RBF(1.0))
    gpc.fit(X, y)
    probas = gpc.predict_proba(X)

    assert probas.shape == (6, 2)
    # Probabilities should sum to 1
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(6), atol=1e-6)
    # All values in [0, 1]
    assert np.all(probas >= 0)
    assert np.all(probas <= 1)


def test_kernel_construction():
    """All kernel types should be constructable."""
    from ferroml.gaussian_process import RBF, Matern, ConstantKernel, WhiteKernel

    rbf = RBF(length_scale=2.0)
    matern = Matern(length_scale=1.5, nu=0.5)
    const_k = ConstantKernel(constant=3.0)
    white_k = WhiteKernel(noise_level=0.1)

    # Should not raise
    assert rbf is not None
    assert matern is not None
    assert const_k is not None
    assert white_k is not None


def test_matern_invalid_nu():
    """Matern with invalid nu should raise ValueError."""
    from ferroml.gaussian_process import Matern

    with pytest.raises(ValueError):
        Matern(length_scale=1.0, nu=3.0)


def test_gpr_not_fitted_raises():
    """Predicting without fitting should raise an error."""
    from ferroml.gaussian_process import GaussianProcessRegressor, RBF

    gpr = GaussianProcessRegressor(kernel=RBF(1.0))
    X = np.array([[0.0, 0.0]])

    with pytest.raises(RuntimeError):
        gpr.predict(X)

    with pytest.raises(RuntimeError):
        gpr.predict_with_std(X)
