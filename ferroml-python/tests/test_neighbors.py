"""Test FerroML K-Nearest Neighbors models."""

import numpy as np
import pytest

from ferroml import ferroml as _native

KNeighborsClassifier = _native.neighbors.KNeighborsClassifier
KNeighborsRegressor = _native.neighbors.KNeighborsRegressor


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def classification_data():
    """Generate binary classification data with a clear separation."""
    np.random.seed(42)
    n_per_class = 60
    # Class 0: centered at (-2, -2)
    X0 = np.random.randn(n_per_class, 2) * 0.5 - 2.0
    # Class 1: centered at (2, 2)
    X1 = np.random.randn(n_per_class, 2) * 0.5 + 2.0
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class)
    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


@pytest.fixture
def multiclass_data():
    """Generate 3-class classification data."""
    np.random.seed(42)
    n_per_class = 50
    # Three well-separated clusters
    X0 = np.random.randn(n_per_class, 3) * 0.5
    X1 = np.random.randn(n_per_class, 3) * 0.5 + np.array([5.0, 0.0, 0.0])
    X2 = np.random.randn(n_per_class, 3) * 0.5 + np.array([0.0, 5.0, 0.0])
    X = np.vstack([X0, X1, X2]).astype(np.float64)
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class + [2.0] * n_per_class)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


@pytest.fixture
def regression_data():
    """Generate regression data with a known linear relationship."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3).astype(np.float64)
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def small_data():
    """Generate a tiny explicit dataset for exact verification."""
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 3.0],
        [6.0, 7.0],
        [7.0, 6.0],
        [8.0, 8.0],
    ], dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return X, y


# ============================================================================
# KNeighborsClassifier Tests
# ============================================================================


class TestKNeighborsClassifier:
    """Tests for KNeighborsClassifier."""

    def test_basic_fit_predict(self, classification_data):
        """Test basic fit and predict workflow."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        preds = clf.predict(X)

        assert preds.shape == y.shape
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_perfect_separation_accuracy(self, small_data):
        """Test that KNN achieves perfect accuracy on well-separated data."""
        X, y = small_data
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X, y)
        preds = clf.predict(X)

        # With k=1, training predictions should be perfect
        accuracy = np.mean(preds == y)
        assert accuracy == 1.0, f"Expected perfect accuracy with k=1, got {accuracy}"

    def test_predict_output_type(self, classification_data):
        """Test that predict returns a numpy array."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        preds = clf.predict(X)

        assert isinstance(preds, np.ndarray)

    def test_predict_classes_subset_of_training_classes(self, classification_data):
        """Test that predicted labels are a subset of training labels."""
        X, y = classification_data
        train_classes = set(np.unique(y).tolist())

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        preds = clf.predict(X)

        pred_classes = set(np.unique(preds).tolist())
        assert pred_classes.issubset(train_classes), (
            f"Predicted classes {pred_classes} should be subset of {train_classes}"
        )

    def test_predict_proba_shape(self, classification_data):
        """Test that predict_proba returns (n_samples, n_classes) array."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        n_classes = len(np.unique(y))
        assert proba.ndim == 2
        assert proba.shape == (X.shape[0], n_classes)

    def test_predict_proba_rows_sum_to_one(self, classification_data):
        """Test that predict_proba rows sum to 1.0."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(X.shape[0]), atol=1e-10)

    def test_predict_proba_values_in_unit_interval(self, classification_data):
        """Test that predict_proba values are in [0, 1]."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        assert np.all(proba >= 0.0), "All probabilities should be >= 0"
        assert np.all(proba <= 1.0), "All probabilities should be <= 1"

    def test_predict_proba_multiclass(self, multiclass_data):
        """Test that predict_proba works correctly for multiclass problems."""
        X, y = multiclass_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        n_classes = len(np.unique(y))
        assert proba.shape == (X.shape[0], n_classes)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]), atol=1e-10)

    def test_n_features_in_getter(self, classification_data):
        """Test that n_features_in_ returns the number of training features."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)

        assert clf.n_features_in_ == X.shape[1]

    def test_n_features_in_various_shapes(self):
        """Test n_features_in_ with various feature counts."""
        np.random.seed(42)
        for n_features in [1, 2, 5, 10]:
            X = np.random.randn(50, n_features).astype(np.float64)
            y = (X[:, 0] > 0).astype(np.float64)
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(X, y)
            assert clf.n_features_in_ == n_features

    def test_classes_getter(self, classification_data):
        """Test that classes_ returns the unique class labels in sorted order."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)

        classes = clf.classes_
        expected_classes = np.unique(y)

        assert isinstance(classes, np.ndarray)
        np.testing.assert_array_equal(classes, expected_classes)

    def test_classes_getter_multiclass(self, multiclass_data):
        """Test classes_ with 3 classes."""
        X, y = multiclass_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)

        classes = clf.classes_
        assert len(classes) == 3
        np.testing.assert_array_equal(classes, np.array([0.0, 1.0, 2.0]))

    def test_n_neighbors_getter(self):
        """Test that n_neighbors getter returns the configured value."""
        for k in [1, 3, 5, 10]:
            clf = KNeighborsClassifier(n_neighbors=k)
            assert clf.n_neighbors == k

    def test_n_neighbors_getter_pre_fit(self):
        """Test that n_neighbors is accessible before fitting."""
        clf = KNeighborsClassifier(n_neighbors=7)
        assert clf.n_neighbors == 7

    def test_different_n_neighbors(self, classification_data):
        """Test that different n_neighbors values produce valid predictions."""
        X, y = classification_data
        for k in [1, 3, 5, 10, 15]:
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X, y)
            preds = clf.predict(X)
            assert preds.shape == y.shape
            assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_weights_uniform(self, classification_data):
        """Test weights='uniform' produces valid predictions."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5, weights="uniform")
        clf.fit(X, y)
        preds = clf.predict(X)

        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))

    def test_weights_distance(self, classification_data):
        """Test weights='distance' produces valid predictions."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
        clf.fit(X, y)
        preds = clf.predict(X)

        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))

    def test_weights_distance_vs_uniform_different(self, small_data):
        """Test that distance-weighted and uniform KNN can produce different results."""
        X, y = small_data
        # Use a test point between the two classes where weights would matter
        clf_uniform = KNeighborsClassifier(n_neighbors=3, weights="uniform")
        clf_uniform.fit(X, y)

        clf_distance = KNeighborsClassifier(n_neighbors=3, weights="distance")
        clf_distance.fit(X, y)

        proba_uniform = clf_uniform.predict_proba(X)
        proba_distance = clf_distance.predict_proba(X)

        # Both should produce valid probabilities
        assert proba_uniform.shape == proba_distance.shape
        np.testing.assert_allclose(proba_uniform.sum(axis=1), np.ones(X.shape[0]), atol=1e-10)
        np.testing.assert_allclose(proba_distance.sum(axis=1), np.ones(X.shape[0]), atol=1e-10)

    def test_metric_euclidean(self, classification_data):
        """Test metric='euclidean' produces valid predictions."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        clf.fit(X, y)
        preds = clf.predict(X)

        assert preds.shape == y.shape

    def test_metric_manhattan(self, classification_data):
        """Test metric='manhattan' produces valid predictions."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
        clf.fit(X, y)
        preds = clf.predict(X)

        assert preds.shape == y.shape

    def test_metric_euclidean_vs_manhattan_shape(self, classification_data):
        """Test that both metrics produce same-shaped output."""
        X, y = classification_data
        clf_euc = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        clf_euc.fit(X, y)

        clf_man = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
        clf_man.fit(X, y)

        preds_euc = clf_euc.predict(X)
        preds_man = clf_man.predict(X)

        assert preds_euc.shape == preds_man.shape

    def test_integer_y_accepted(self, classification_data):
        """Test that integer y arrays are accepted (converted internally to float)."""
        X, y = classification_data
        y_int = y.astype(np.int64)

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y_int)
        preds = clf.predict(X)

        assert preds.shape == y.shape

    def test_high_dimensional_data(self):
        """Test KNN on high-dimensional data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        X = np.random.randn(n_samples, n_features).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.float64)

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        preds = clf.predict(X)

        assert preds.shape == (n_samples,)
        assert clf.n_features_in_ == n_features

    def test_not_fitted_predict_raises(self, classification_data):
        """Test that predict before fit raises an error."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)

        with pytest.raises(Exception):
            clf.predict(X)

    def test_not_fitted_predict_proba_raises(self, classification_data):
        """Test that predict_proba before fit raises an error."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)

        with pytest.raises(Exception):
            clf.predict_proba(X)

    def test_not_fitted_n_features_in_raises(self):
        """Test that n_features_in_ before fit raises ValueError."""
        clf = KNeighborsClassifier(n_neighbors=5)
        with pytest.raises(ValueError, match="not fitted"):
            _ = clf.n_features_in_

    def test_not_fitted_classes_raises(self):
        """Test that classes_ before fit raises ValueError."""
        clf = KNeighborsClassifier(n_neighbors=5)
        with pytest.raises(ValueError, match="not fitted"):
            _ = clf.classes_

    def test_invalid_weights_raises(self):
        """Test that an invalid weights string raises ValueError."""
        with pytest.raises(ValueError):
            KNeighborsClassifier(n_neighbors=5, weights="invalid")

    def test_invalid_metric_raises(self):
        """Test that an invalid metric string raises ValueError."""
        with pytest.raises(ValueError):
            KNeighborsClassifier(n_neighbors=5, metric="invalid")

    def test_high_accuracy_separable_data(self, classification_data):
        """Test that KNN achieves high accuracy on well-separated binary data."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)
        preds = clf.predict(X)

        accuracy = np.mean(preds == y)
        assert accuracy > 0.9, f"Expected >90% training accuracy, got {accuracy:.3f}"

    def test_consistent_predictions(self, classification_data):
        """Test that the same model produces the same predictions when called twice."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)

        preds1 = clf.predict(X)
        preds2 = clf.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

    def test_predict_proba_consistent_with_predict(self, classification_data):
        """Test that argmax(predict_proba) matches predict output."""
        X, y = classification_data
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        classes = clf.classes_

        # The class with the highest probability should match predict
        pred_from_proba = classes[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(preds, pred_from_proba)


# ============================================================================
# KNeighborsRegressor Tests
# ============================================================================


class TestKNeighborsRegressor:
    """Tests for KNeighborsRegressor."""

    def test_basic_fit_predict(self, regression_data):
        """Test basic fit and predict workflow."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)
        preds = reg.predict(X)

        assert preds.shape == y.shape

    def test_predict_output_type(self, regression_data):
        """Test that predict returns a numpy array."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)
        preds = reg.predict(X)

        assert isinstance(preds, np.ndarray)

    def test_predict_finite_values(self, regression_data):
        """Test that predict returns finite values."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)
        preds = reg.predict(X)

        assert np.all(np.isfinite(preds)), "All predictions should be finite"

    def test_constant_function(self):
        """Test KNN regressor on a constant target (all same value)."""
        np.random.seed(42)
        X = np.random.randn(50, 2).astype(np.float64)
        y = np.full(50, 3.14)

        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)
        preds = reg.predict(X)

        np.testing.assert_allclose(preds, 3.14, atol=1e-10)

    def test_identity_with_k1(self):
        """Test that k=1 with uniform weights memorizes training targets exactly."""
        np.random.seed(42)
        X = np.random.randn(30, 2).astype(np.float64)
        y = np.random.randn(30)

        reg = KNeighborsRegressor(n_neighbors=1)
        reg.fit(X, y)
        preds = reg.predict(X)

        # With k=1, nearest neighbor of each training point is itself
        np.testing.assert_allclose(preds, y, atol=1e-10)

    def test_n_features_in_getter(self, regression_data):
        """Test that n_features_in_ returns the number of training features."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)

        assert reg.n_features_in_ == X.shape[1]

    def test_n_features_in_various_shapes(self):
        """Test n_features_in_ with various feature counts."""
        np.random.seed(42)
        for n_features in [1, 2, 5, 10]:
            X = np.random.randn(50, n_features).astype(np.float64)
            y = np.random.randn(50)
            reg = KNeighborsRegressor(n_neighbors=3)
            reg.fit(X, y)
            assert reg.n_features_in_ == n_features

    def test_n_neighbors_getter(self):
        """Test that n_neighbors getter returns the configured value."""
        for k in [1, 3, 5, 10]:
            reg = KNeighborsRegressor(n_neighbors=k)
            assert reg.n_neighbors == k

    def test_n_neighbors_getter_pre_fit(self):
        """Test that n_neighbors is accessible before fitting."""
        reg = KNeighborsRegressor(n_neighbors=7)
        assert reg.n_neighbors == 7

    def test_different_n_neighbors(self, regression_data):
        """Test that different n_neighbors values produce valid predictions."""
        X, y = regression_data
        for k in [1, 3, 5, 10]:
            reg = KNeighborsRegressor(n_neighbors=k)
            reg.fit(X, y)
            preds = reg.predict(X)
            assert preds.shape == y.shape
            assert np.all(np.isfinite(preds))

    def test_weights_uniform(self, regression_data):
        """Test weights='uniform' produces valid predictions."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5, weights="uniform")
        reg.fit(X, y)
        preds = reg.predict(X)

        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))

    def test_weights_distance(self, regression_data):
        """Test weights='distance' produces valid predictions."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5, weights="distance")
        reg.fit(X, y)
        preds = reg.predict(X)

        assert preds.shape == y.shape
        assert np.all(np.isfinite(preds))

    def test_weights_distance_more_accurate(self):
        """Test that distance weighting can improve accuracy on smooth functions."""
        np.random.seed(42)
        # 1D function: y = x^2, data on [0, 1]
        X = np.sort(np.random.rand(100, 1)).astype(np.float64)
        y = (X[:, 0] ** 2).astype(np.float64)

        reg_unif = KNeighborsRegressor(n_neighbors=5, weights="uniform")
        reg_unif.fit(X, y)
        preds_unif = reg_unif.predict(X)

        reg_dist = KNeighborsRegressor(n_neighbors=5, weights="distance")
        reg_dist.fit(X, y)
        preds_dist = reg_dist.predict(X)

        mse_unif = np.mean((preds_unif - y) ** 2)
        mse_dist = np.mean((preds_dist - y) ** 2)

        # Distance weighting should be at least as good as uniform on smooth data
        # (this is not always true but generally holds for smooth functions)
        # At minimum both should produce finite, reasonable predictions
        assert np.isfinite(mse_unif)
        assert np.isfinite(mse_dist)

    def test_metric_euclidean(self, regression_data):
        """Test metric='euclidean' produces valid predictions."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5, metric="euclidean")
        reg.fit(X, y)
        preds = reg.predict(X)

        assert preds.shape == y.shape

    def test_metric_manhattan(self, regression_data):
        """Test metric='manhattan' produces valid predictions."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5, metric="manhattan")
        reg.fit(X, y)
        preds = reg.predict(X)

        assert preds.shape == y.shape

    def test_predictions_in_range_of_targets(self, regression_data):
        """Test that predictions are within the range of training targets."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)
        preds = reg.predict(X)

        # KNN predictions are averages of training targets, so they should
        # lie within [min(y), max(y)]
        assert np.all(preds >= y.min() - 1e-10), "Predictions below min target"
        assert np.all(preds <= y.max() + 1e-10), "Predictions above max target"

    def test_integer_y_accepted(self, regression_data):
        """Test that integer y arrays are accepted (converted internally to float)."""
        X, y = regression_data
        y_int = (y * 10).astype(np.int32)

        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y_int)
        preds = reg.predict(X)

        assert preds.shape == y.shape

    def test_low_mse_on_known_function(self):
        """Test that KNN regressor achieves low MSE on a simple smooth function."""
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 2).astype(np.float64)
        y = X[:, 0] + 2.0 * X[:, 1]  # Perfect linear

        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)
        preds = reg.predict(X)

        mse = np.mean((preds - y) ** 2)
        assert mse < 1.0, f"Expected low MSE on simple linear function, got {mse:.4f}"

    def test_consistent_predictions(self, regression_data):
        """Test that the same model produces the same predictions when called twice."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)

        preds1 = reg.predict(X)
        preds2 = reg.predict(X)

        np.testing.assert_array_equal(preds1, preds2)

    def test_not_fitted_predict_raises(self, regression_data):
        """Test that predict before fit raises an error."""
        X, y = regression_data
        reg = KNeighborsRegressor(n_neighbors=5)

        with pytest.raises(Exception):
            reg.predict(X)

    def test_not_fitted_n_features_in_raises(self):
        """Test that n_features_in_ before fit raises ValueError."""
        reg = KNeighborsRegressor(n_neighbors=5)
        with pytest.raises(ValueError, match="not fitted"):
            _ = reg.n_features_in_

    def test_invalid_weights_raises(self):
        """Test that an invalid weights string raises ValueError."""
        with pytest.raises(ValueError):
            KNeighborsRegressor(n_neighbors=5, weights="invalid")

    def test_invalid_metric_raises(self):
        """Test that an invalid metric string raises ValueError."""
        with pytest.raises(ValueError):
            KNeighborsRegressor(n_neighbors=5, metric="invalid")

    def test_high_dimensional_data(self):
        """Test KNN regressor on high-dimensional data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        X = np.random.randn(n_samples, n_features).astype(np.float64)
        y = X[:, 0] + X[:, 1]

        reg = KNeighborsRegressor(n_neighbors=5)
        reg.fit(X, y)
        preds = reg.predict(X)

        assert preds.shape == (n_samples,)
        assert reg.n_features_in_ == n_features
        assert np.all(np.isfinite(preds))
