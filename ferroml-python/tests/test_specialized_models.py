"""
Tests for specialized models: RobustRegression, QuantileRegression, Perceptron, NearestCentroid.

These models were added in Plan G Phase 3.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# RobustRegression
# ---------------------------------------------------------------------------

class TestRobustRegression:
    """Tests for RobustRegression (M-estimator IRLS)."""

    def test_fit_predict_basic(self):
        from ferroml.linear import RobustRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.1, 3.9, 6.1, 7.9, 10.1])
        model = RobustRegression()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (5,)

    def test_coef_and_intercept(self):
        from ferroml.linear import RobustRegression

        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        y = np.array([5.0, 8.0, 11.0, 14.0, 17.0])
        model = RobustRegression()
        model.fit(X, y)
        coef = model.coef_
        assert coef.shape == (2,)
        intercept = model.intercept_
        assert isinstance(intercept, float)

    def test_huber_estimator(self):
        from ferroml.linear import RobustRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        model = RobustRegression(estimator="huber")
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_allclose(preds, y, atol=0.5)

    def test_bisquare_estimator(self):
        from ferroml.linear import RobustRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        model = RobustRegression(estimator="bisquare")
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_allclose(preds, y, atol=0.5)

    def test_tukey_alias(self):
        from ferroml.linear import RobustRegression

        model = RobustRegression(estimator="tukey")
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (3,)

    def test_robust_to_outliers(self):
        from ferroml.linear import RobustRegression

        np.random.seed(42)
        X = np.arange(20).reshape(-1, 1).astype(float)
        y = 2.0 * X.ravel() + 1.0
        # Add outliers
        y[0] = 100.0
        y[19] = -100.0
        model = RobustRegression(estimator="bisquare")
        model.fit(X, y)
        coef = model.coef_
        # The slope should be close to 2.0 despite outliers
        assert abs(coef[0] - 2.0) < 1.0

    def test_unfitted_coef_raises(self):
        from ferroml.linear import RobustRegression

        model = RobustRegression()
        with pytest.raises((ValueError, RuntimeError)):
            _ = model.coef_

    def test_unfitted_intercept_raises(self):
        from ferroml.linear import RobustRegression

        model = RobustRegression()
        with pytest.raises((ValueError, RuntimeError)):
            _ = model.intercept_

    def test_unfitted_predict_raises(self):
        from ferroml.linear import RobustRegression

        model = RobustRegression()
        X = np.array([[1.0], [2.0]])
        with pytest.raises((ValueError, RuntimeError)):
            model.predict(X)

    def test_repr(self):
        from ferroml.linear import RobustRegression

        model = RobustRegression()
        r = repr(model)
        assert "RobustRegression" in r

    def test_invalid_estimator_raises(self):
        from ferroml.linear import RobustRegression

        with pytest.raises((ValueError, RuntimeError)):
            RobustRegression(estimator="invalid")


# ---------------------------------------------------------------------------
# QuantileRegression
# ---------------------------------------------------------------------------

class TestQuantileRegression:
    """Tests for QuantileRegression."""

    def test_fit_predict_basic(self):
        from ferroml.linear import QuantileRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        model = QuantileRegression(quantile=0.5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (5,)

    def test_coef_and_intercept(self):
        from ferroml.linear import QuantileRegression

        rng = np.random.RandomState(42)
        X = rng.randn(20, 2)
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 1.0 + rng.randn(20) * 0.1
        model = QuantileRegression()
        model.fit(X, y)
        coef = model.coef_
        assert coef.shape == (2,)
        intercept = model.intercept_
        assert isinstance(intercept, float)

    def test_median_regression(self):
        from ferroml.linear import QuantileRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        model = QuantileRegression(quantile=0.5)
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_allclose(preds, y, atol=1.0)

    def test_quantile_025(self):
        from ferroml.linear import QuantileRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
        model = QuantileRegression(quantile=0.25)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_quantile_075(self):
        from ferroml.linear import QuantileRegression

        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
        model = QuantileRegression(quantile=0.75)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_unfitted_coef_raises(self):
        from ferroml.linear import QuantileRegression

        model = QuantileRegression()
        with pytest.raises((ValueError, RuntimeError)):
            _ = model.coef_

    def test_unfitted_intercept_raises(self):
        from ferroml.linear import QuantileRegression

        model = QuantileRegression()
        with pytest.raises((ValueError, RuntimeError)):
            _ = model.intercept_

    def test_repr(self):
        from ferroml.linear import QuantileRegression

        model = QuantileRegression(quantile=0.25)
        r = repr(model)
        assert "QuantileRegression" in r


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------

class TestPerceptron:
    """Tests for Perceptron classifier."""

    def test_fit_predict_linearly_separable(self):
        from ferroml.linear import Perceptron

        # Linearly separable data
        X = np.array([
            [0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0],
            [5.0, 5.0], [5.5, 5.5], [6.0, 5.0], [5.0, 6.0],
        ])
        y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        model = Perceptron(max_iter=1000, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)
        # Should get high accuracy on linearly separable data
        accuracy = np.mean(preds == y)
        assert accuracy >= 0.75

    def test_random_state_reproducibility(self):
        from ferroml.linear import Perceptron

        X = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [5.0, 5.0], [6.0, 5.0], [5.0, 6.0],
        ])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        model1 = Perceptron(random_state=123)
        model1.fit(X, y)
        preds1 = model1.predict(X)

        model2 = Perceptron(random_state=123)
        model2.fit(X, y)
        preds2 = model2.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

    def test_unfitted_predict_raises(self):
        from ferroml.linear import Perceptron

        model = Perceptron()
        X = np.array([[1.0, 2.0]])
        with pytest.raises((ValueError, RuntimeError)):
            model.predict(X)

    def test_repr(self):
        from ferroml.linear import Perceptron

        model = Perceptron()
        r = repr(model)
        assert "Perceptron" in r

    def test_integer_labels(self):
        from ferroml.linear import Perceptron

        X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        model = Perceptron(random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)


# ---------------------------------------------------------------------------
# NearestCentroid
# ---------------------------------------------------------------------------

class TestNearestCentroid:
    """Tests for NearestCentroid classifier."""

    def test_fit_predict_basic(self):
        from ferroml.neighbors import NearestCentroid

        X = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [5.0, 5.0], [6.0, 5.0], [5.0, 6.0],
        ])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        model = NearestCentroid()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (6,)
        np.testing.assert_array_equal(preds, y)

    def test_centroids_shape(self):
        from ferroml.neighbors import NearestCentroid

        X = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [5.0, 5.0, 5.0], [6.0, 5.0, 5.0],
        ])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = NearestCentroid()
        model.fit(X, y)
        centroids = model.centroids_
        assert centroids.shape == (2, 3)

    def test_classes_shape(self):
        from ferroml.neighbors import NearestCentroid

        X = np.array([[0.0], [1.0], [5.0], [6.0], [10.0], [11.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        model = NearestCentroid()
        model.fit(X, y)
        classes = model.classes_
        assert classes.shape == (3,)
        np.testing.assert_array_equal(np.sort(classes), np.array([0.0, 1.0, 2.0]))

    def test_manhattan_metric(self):
        from ferroml.neighbors import NearestCentroid

        X = np.array([
            [0.0, 0.0], [1.0, 0.0],
            [10.0, 10.0], [11.0, 10.0],
        ])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = NearestCentroid(metric="manhattan")
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_unfitted_centroids_raises(self):
        from ferroml.neighbors import NearestCentroid

        model = NearestCentroid()
        with pytest.raises((ValueError, RuntimeError)):
            _ = model.centroids_

    def test_unfitted_classes_raises(self):
        from ferroml.neighbors import NearestCentroid

        model = NearestCentroid()
        with pytest.raises((ValueError, RuntimeError)):
            _ = model.classes_

    def test_unfitted_predict_raises(self):
        from ferroml.neighbors import NearestCentroid

        model = NearestCentroid()
        X = np.array([[1.0, 2.0]])
        with pytest.raises((ValueError, RuntimeError)):
            model.predict(X)

    def test_repr(self):
        from ferroml.neighbors import NearestCentroid

        model = NearestCentroid()
        r = repr(model)
        assert "NearestCentroid" in r

    def test_integer_labels(self):
        from ferroml.neighbors import NearestCentroid

        X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        model = NearestCentroid()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)
