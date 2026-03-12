"""Tests for MultiOutput wrappers."""

import numpy as np
import pytest

from ferroml.multioutput import MultiOutputClassifier, MultiOutputRegressor


def make_regression_data(n_samples=50, n_features=3, n_outputs=3):
    """Generate deterministic multi-output regression data."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    coefs = rng.randn(n_features, n_outputs)
    Y = X @ coefs + 0.1 * rng.randn(n_samples, n_outputs)
    return X, Y


def make_classification_data(n_samples=60, n_features=4, n_outputs=3):
    """Generate deterministic multi-label classification data."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    Y = np.zeros((n_samples, n_outputs))
    for k in range(n_outputs):
        coefs = rng.randn(n_features)
        Y[:, k] = (X @ coefs > 0).astype(float)
    return X, Y


class TestMultiOutputRegressor:
    def test_linear_regression_3_outputs(self):
        X, Y = make_regression_data(n_outputs=3)
        mo = MultiOutputRegressor("linear_regression")
        mo.fit(X, Y)
        preds = mo.predict(X)
        assert preds.shape == (50, 3)
        assert mo.is_fitted()

    def test_predict_shape(self):
        X, Y = make_regression_data(n_samples=30, n_outputs=5)
        mo = MultiOutputRegressor("linear_regression")
        mo.fit(X, Y)
        preds = mo.predict(X)
        assert preds.shape == (30, 5)
        assert mo.n_outputs == 5

    def test_ridge_estimator(self):
        X, Y = make_regression_data(n_outputs=2)
        mo = MultiOutputRegressor("ridge")
        mo.fit(X, Y)
        preds = mo.predict(X)
        assert preds.shape == (50, 2)

    def test_decision_tree_estimator(self):
        X, Y = make_regression_data(n_outputs=2)
        mo = MultiOutputRegressor("decision_tree")
        mo.fit(X, Y)
        preds = mo.predict(X)
        assert preds.shape == (50, 2)

    def test_knn_estimator(self):
        X, Y = make_regression_data(n_outputs=2)
        mo = MultiOutputRegressor("knn")
        mo.fit(X, Y)
        preds = mo.predict(X)
        assert preds.shape == (50, 2)

    def test_not_fitted_raises(self):
        mo = MultiOutputRegressor("linear_regression")
        X = np.random.randn(10, 3)
        with pytest.raises(RuntimeError, match="[Nn]ot fitted"):
            mo.predict(X)

    def test_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            MultiOutputRegressor("nonexistent_model")


class TestMultiOutputClassifier:
    def test_logistic_regression(self):
        X, Y = make_classification_data(n_outputs=3)
        mo = MultiOutputClassifier("logistic_regression")
        mo.fit(X, Y)
        preds = mo.predict(X)
        assert preds.shape == (60, 3)
        # All predictions should be 0 or 1
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_predict_shape(self):
        X, Y = make_classification_data(n_outputs=2)
        mo = MultiOutputClassifier("logistic_regression")
        mo.fit(X, Y)
        preds = mo.predict(X)
        assert preds.shape == (60, 2)
        assert mo.n_outputs == 2

    def test_decision_tree_classifier(self):
        X, Y = make_classification_data(n_outputs=2)
        mo = MultiOutputClassifier("decision_tree")
        mo.fit(X, Y)
        preds = mo.predict(X)
        assert preds.shape == (60, 2)

    def test_predict_proba(self):
        X, Y = make_classification_data(n_outputs=2)
        mo = MultiOutputClassifier("logistic_regression")
        mo.fit(X, Y)
        probas = mo.predict_proba(X)
        assert len(probas) == 2
        for proba in probas:
            assert proba.shape[0] == 60
            assert proba.shape[1] == 2  # binary
            # Each row should sum to ~1
            row_sums = proba.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_not_fitted_raises(self):
        mo = MultiOutputClassifier("logistic_regression")
        X = np.random.randn(10, 4)
        with pytest.raises(RuntimeError, match="[Nn]ot fitted"):
            mo.predict(X)

    def test_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown estimator"):
            MultiOutputClassifier("nonexistent_model")
