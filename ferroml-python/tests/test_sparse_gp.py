"""Tests for Sparse/Approximate Gaussian Process models."""

import numpy as np
import pytest
from ferroml.gaussian_process import (
    SparseGPRegressor,
    SparseGPClassifier,
    SVGPRegressor,
    GaussianProcessRegressor,
    RBF,
    Matern,
    ConstantKernel,
)


def make_sin_data(n=100, seed=42):
    """Generate sin regression data."""
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(0, 6, n)).reshape(-1, 1)
    y = np.sin(x).ravel()
    return x, y


def make_separable_data():
    """Generate linearly separable classification data."""
    x = np.array([
        [1.0, 1.0], [1.5, 2.0], [2.0, 1.0], [1.0, 2.5], [2.5, 1.5],
        [6.0, 6.0], [6.5, 7.0], [7.0, 6.0], [6.0, 7.5], [7.5, 6.5],
    ])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return x, y


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


# =========================================================================
# SparseGPRegressor tests
# =========================================================================


class TestSparseGPRegressor:
    def test_fitc_sin_regression(self):
        x, y = make_sin_data()
        model = SparseGPRegressor(
            kernel=RBF(1.0), alpha=0.01, n_inducing=20,
            inducing_method="random", approximation="fitc"
        )
        model.fit(x, y)
        preds = model.predict(x)
        r2 = r_squared(y, preds)
        assert r2 > 0.8, f"FITC R^2 = {r2}"

    def test_vfe_sin_regression(self):
        x, y = make_sin_data()
        model = SparseGPRegressor(
            kernel=RBF(1.0), alpha=0.01, n_inducing=20,
            inducing_method="random", approximation="vfe"
        )
        model.fit(x, y)
        preds = model.predict(x)
        r2 = r_squared(y, preds)
        assert r2 > 0.8, f"VFE R^2 = {r2}"

    def test_predict_with_std_shapes(self):
        x, y = make_sin_data(50)
        model = SparseGPRegressor(
            kernel=RBF(1.0), alpha=0.01, n_inducing=10,
            inducing_method="random"
        )
        model.fit(x, y)
        mean, std = model.predict_with_std(x)
        assert mean.shape == (50,)
        assert std.shape == (50,)
        assert np.all(std >= 0)

    def test_matches_exact_m_eq_n(self):
        x, y = make_sin_data(20)
        exact = GaussianProcessRegressor(kernel=RBF(1.0), alpha=0.01)
        exact.fit(x, y)
        exact_preds = exact.predict(x)

        sparse = SparseGPRegressor(
            kernel=RBF(1.0), alpha=0.01, n_inducing=20,
            inducing_method="random", approximation="fitc"
        )
        sparse.fit(x, y)
        sparse_preds = sparse.predict(x)

        # Should be reasonably close
        np.testing.assert_allclose(exact_preds, sparse_preds, atol=0.5)

    def test_inducing_points_accessible(self):
        x, y = make_sin_data(50)
        model = SparseGPRegressor(
            kernel=RBF(1.0), alpha=0.01, n_inducing=10,
            inducing_method="random"
        )
        model.fit(x, y)
        z = model.inducing_points_
        assert z.shape == (10, 1)

    def test_normalize_y_shifted_target(self):
        x, y = make_sin_data()
        y_shifted = y + 1000.0
        model = SparseGPRegressor(
            kernel=RBF(1.0), alpha=0.01, n_inducing=20,
            inducing_method="random", normalize_y=True
        )
        model.fit(x, y_shifted)
        preds = model.predict(x)
        r2 = r_squared(y_shifted, preds)
        assert r2 > 0.8, f"normalize_y R^2 = {r2}"

    def test_all_kernels(self):
        x, y = make_sin_data(50)
        for kernel in [RBF(1.0), Matern(1.0, 2.5)]:
            model = SparseGPRegressor(
                kernel=kernel, alpha=0.01, n_inducing=10,
                inducing_method="random"
            )
            model.fit(x, y)
            preds = model.predict(x)
            assert len(preds) == 50

    def test_inducing_methods(self):
        x, y = make_sin_data(50)
        for method in ["random", "kmeans", "greedy_variance"]:
            model = SparseGPRegressor(
                kernel=RBF(1.0), alpha=0.01, n_inducing=10,
                inducing_method=method
            )
            model.fit(x, y)
            preds = model.predict(x)
            assert len(preds) == 50

    def test_log_marginal_likelihood(self):
        x, y = make_sin_data()
        model = SparseGPRegressor(
            kernel=RBF(1.0), alpha=0.01, n_inducing=20,
            inducing_method="random"
        )
        model.fit(x, y)
        lml = model.log_marginal_likelihood_
        assert np.isfinite(lml)


# =========================================================================
# SparseGPClassifier tests
# =========================================================================


class TestSparseGPClassifier:
    def test_linearly_separable(self):
        x, y = make_separable_data()
        model = SparseGPClassifier(
            kernel=RBF(1.0), n_inducing=10,
            inducing_method="random"
        )
        model.fit(x, y)
        preds = model.predict(x)
        accuracy = np.mean(preds == y)
        assert accuracy >= 0.8, f"accuracy = {accuracy}"

    def test_predict_proba_valid(self):
        x, y = make_separable_data()
        model = SparseGPClassifier(
            kernel=RBF(1.0), n_inducing=10,
            inducing_method="random"
        )
        model.fit(x, y)
        probas = model.predict_proba(x)
        assert probas.shape == (10, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)

    def test_inducing_points_accessible(self):
        x, y = make_separable_data()
        model = SparseGPClassifier(
            kernel=RBF(1.0), n_inducing=5,
            inducing_method="random"
        )
        model.fit(x, y)
        z = model.inducing_points_
        assert z.shape == (5, 2)


# =========================================================================
# SVGPRegressor tests
# =========================================================================


class TestSVGPRegressor:
    def test_basic_regression(self):
        x, y = make_sin_data(200)
        model = SVGPRegressor(
            kernel=RBF(1.0), noise_variance=0.01, n_inducing=30,
            inducing_method="random", n_epochs=50, batch_size=200,
            learning_rate=0.1
        )
        model.fit(x, y)
        preds = model.predict(x)
        r2 = r_squared(y, preds)
        assert r2 > 0.3, f"SVGP R^2 = {r2}"

    def test_predict_with_std(self):
        x, y = make_sin_data(100)
        model = SVGPRegressor(
            kernel=RBF(1.0), noise_variance=0.01, n_inducing=15,
            inducing_method="random", n_epochs=20, batch_size=100,
            learning_rate=0.1
        )
        model.fit(x, y)
        mean, std = model.predict_with_std(x)
        assert mean.shape == (100,)
        assert std.shape == (100,)
        assert np.all(std >= 0)

    def test_batch_size_parameter(self):
        x, y = make_sin_data(100)
        for bs in [32, 50, 100]:
            model = SVGPRegressor(
                kernel=RBF(1.0), noise_variance=0.01, n_inducing=15,
                inducing_method="random", n_epochs=10, batch_size=bs,
                learning_rate=0.1
            )
            model.fit(x, y)
            preds = model.predict(x)
            assert len(preds) == 100

    def test_inducing_points_accessible(self):
        x, y = make_sin_data(100)
        model = SVGPRegressor(
            kernel=RBF(1.0), noise_variance=0.01, n_inducing=15,
            inducing_method="random", n_epochs=10, batch_size=100,
            learning_rate=0.1
        )
        model.fit(x, y)
        z = model.inducing_points_
        assert z.shape == (15, 1)

    def test_normalize_y(self):
        x, y = make_sin_data(100)
        y_shifted = y + 500.0
        model = SVGPRegressor(
            kernel=RBF(1.0), noise_variance=0.01, n_inducing=20,
            inducing_method="random", n_epochs=30, batch_size=100,
            learning_rate=0.1, normalize_y=True
        )
        model.fit(x, y_shifted)
        preds = model.predict(x)
        # Predictions should be in the right range
        assert np.mean(preds) > 490, f"mean pred = {np.mean(preds)}"
