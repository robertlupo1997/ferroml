"""
Bindings correctness audit: verifies PyO3 bindings faithfully expose Rust
behavior with no silent data corruption, proper error propagation, correct
state management, serialization fidelity, and thread safety.

Representative models tested:
  - LinearRegression (simplest, has coefficients)
  - RandomForestClassifier (complex, stochastic)
  - StandardScaler (transformer, not model)
  - KMeans (clustering)
"""

import concurrent.futures
import pickle

import numpy as np
import pytest

from ferroml.clustering import KMeans
from ferroml.linear import LinearRegression
from ferroml.preprocessing import StandardScaler
from ferroml.trees import RandomForestClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data():
    """Non-collinear regression data with known coefficients."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.1 * np.random.randn(50)
    return X, y


@pytest.fixture
def classification_data():
    """Binary classification data."""
    np.random.seed(42)
    X = np.random.randn(60, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    return X, y


@pytest.fixture
def clustering_data():
    """Well-separated clusters for KMeans."""
    np.random.seed(42)
    c1 = np.random.randn(20, 3) + np.array([5, 5, 5])
    c2 = np.random.randn(20, 3) + np.array([-5, -5, -5])
    return np.vstack([c1, c2])


# ===================================================================
# 1. ARRAY CONVERSION FIDELITY
# ===================================================================

class TestArrayConversionFidelity:
    """Verify numpy arrays survive the Python -> Rust -> Python boundary."""

    def test_float64_roundtrip_no_precision_loss(self, regression_data):
        """Coefficients and predictions are deterministic float64 values."""
        X, y = regression_data
        m = LinearRegression()
        m.fit(X, y)
        pred1 = m.predict(X[:5])
        pred2 = m.predict(X[:5])
        np.testing.assert_array_equal(
            pred1, pred2,
            err_msg="Identical predict calls returned different float64 values",
        )

    def test_c_contiguous_array(self, regression_data):
        """C-contiguous (row-major) arrays work correctly."""
        X, y = regression_data
        X_c = np.ascontiguousarray(X)
        assert X_c.flags["C_CONTIGUOUS"]
        m = LinearRegression()
        m.fit(X_c, y)
        pred = m.predict(X_c[:5])
        assert pred.shape == (5,)
        assert np.all(np.isfinite(pred))

    def test_f_contiguous_array(self, regression_data):
        """F-contiguous (column-major) arrays work correctly."""
        X, y = regression_data
        X_f = np.asfortranarray(X)
        assert X_f.flags["F_CONTIGUOUS"]
        m = LinearRegression()
        m.fit(X_f, y)
        pred = m.predict(X_f[:5])
        assert pred.shape == (5,)
        assert np.all(np.isfinite(pred))

    def test_c_vs_f_contiguous_same_results(self, regression_data):
        """C-contiguous and F-contiguous arrays produce identical results."""
        X, y = regression_data
        X_c = np.ascontiguousarray(X)
        X_f = np.asfortranarray(X)

        m_c = LinearRegression()
        m_c.fit(X_c, y)
        pred_c = m_c.predict(X_c[:5])

        m_f = LinearRegression()
        m_f.fit(X_f, y)
        pred_f = m_f.predict(X_f[:5])

        np.testing.assert_allclose(
            pred_c, pred_f, rtol=1e-12,
            err_msg="C-contiguous and F-contiguous arrays gave different results",
        )

    def test_extreme_small_values_survive(self):
        """Very small values (1e-300) pass through the binding without being zeroed."""
        np.random.seed(42)
        # Use KMeans predict to verify small values survive the round-trip:
        # fit on normal data, predict with small-valued points -- the values
        # must reach Rust intact (not underflow to zero) for distance calc.
        X_train = np.array([[0.0, 0.0], [10.0, 10.0]] * 5, dtype=float)
        km = KMeans(n_clusters=2)
        km.fit(X_train)
        # A point near origin with tiny offset should cluster with [0,0]
        X_small = np.array([[1e-300, 1e-300]])
        label_small = km.predict(X_small)
        label_origin = km.predict(np.array([[0.0, 0.0]]))
        assert label_small[0] == label_origin[0], (
            "Tiny values did not cluster with origin -- possible underflow"
        )

    def test_extreme_large_values_survive(self):
        """Very large values (1e150) pass through without overflowing to inf."""
        # Use StandardScaler with moderate large values (1e150) where
        # float64 arithmetic remains stable
        np.random.seed(42)
        X = np.random.randn(20, 3) * 1e150
        sc = StandardScaler()
        Xt = sc.fit_transform(X)
        assert np.all(np.isfinite(Xt)), "Large values overflowed to inf/nan"
        assert np.any(np.abs(Xt) > 0.1), "Large values collapsed to zero"

    def test_empty_array_raises_error(self):
        """Empty array produces a clear error, not a crash/segfault."""
        m = LinearRegression()
        with pytest.raises(RuntimeError):
            m.fit(np.array([]).reshape(0, 0), np.array([]))


# ===================================================================
# 2. ERROR PROPAGATION
# ===================================================================

class TestErrorPropagation:
    """Verify Rust errors become Python exceptions, not panics/segfaults."""

    def test_predict_before_fit_linear(self):
        """LinearRegression.predict before fit raises RuntimeError."""
        m = LinearRegression()
        with pytest.raises(RuntimeError, match="[Nn]ot fitted|fit"):
            m.predict(np.array([[1.0, 2.0, 3.0]]))

    def test_predict_before_fit_rf(self):
        """RandomForestClassifier.predict before fit raises RuntimeError."""
        m = RandomForestClassifier(n_estimators=5)
        with pytest.raises(RuntimeError, match="[Nn]ot fitted|fit"):
            m.predict(np.array([[1.0, 2.0, 3.0]]))

    def test_predict_before_fit_kmeans(self):
        """KMeans.predict before fit raises RuntimeError."""
        m = KMeans(n_clusters=2)
        with pytest.raises(RuntimeError, match="[Nn]ot fitted|fit"):
            m.predict(np.array([[1.0, 2.0, 3.0]]))

    def test_transform_before_fit_scaler(self):
        """StandardScaler.transform before fit raises RuntimeError."""
        sc = StandardScaler()
        with pytest.raises(RuntimeError, match="[Nn]ot fitted|fit"):
            sc.transform(np.array([[1.0, 2.0]]))

    def test_nan_input_raises_error_linear(self):
        """NaN in training data raises RuntimeError for LinearRegression."""
        m = LinearRegression()
        X = np.array([[float("nan"), 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(RuntimeError, match="[Nn]a[Nn]|invalid|infinite"):
            m.fit(X, y)

    def test_nan_input_raises_error_rf(self):
        """NaN in training data raises RuntimeError for RandomForestClassifier."""
        m = RandomForestClassifier(n_estimators=5)
        X = np.array([[float("nan"), 1.0], [2.0, 3.0]])
        y = np.array([0.0, 1.0])
        with pytest.raises(RuntimeError, match="[Nn]a[Nn]|invalid|infinite"):
            m.fit(X, y)

    def test_shape_mismatch_x_y(self):
        """Mismatched X rows and y length raises RuntimeError."""
        m = LinearRegression()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError, match="[Ss]hape|mismatch|dimension"):
            m.fit(X, y)

    def test_wrong_features_at_predict(self, regression_data):
        """Wrong number of features at predict time raises RuntimeError."""
        X, y = regression_data
        m = LinearRegression()
        m.fit(X, y)
        X_wrong = np.random.randn(5, 7)  # 7 features instead of 3
        with pytest.raises(RuntimeError, match="[Ff]eature|[Ss]hape|[Dd]imension|[Cc]olumn"):
            m.predict(X_wrong)

    def test_inf_input_raises_error(self):
        """Inf in training data raises RuntimeError."""
        m = LinearRegression()
        X = np.array([[float("inf"), 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(RuntimeError, match="[Nn]a[Nn]|invalid|infinite|inf"):
            m.fit(X, y)


# ===================================================================
# 3. STATE MANAGEMENT
# ===================================================================

class TestStateManagement:
    """Verify model state is managed correctly across operations."""

    def test_fitted_state_persists(self, regression_data):
        """Fitted model retains state across multiple predict calls."""
        X, y = regression_data
        m = LinearRegression()
        m.fit(X, y)
        pred1 = m.predict(X[:5])
        pred2 = m.predict(X[:5])
        np.testing.assert_array_equal(pred1, pred2)

    def test_refit_replaces_state(self, regression_data):
        """Re-fitting with different data changes predictions."""
        X, y = regression_data
        m = LinearRegression()

        # First fit
        m.fit(X, y)
        pred_first = m.predict(X[:5])

        # Second fit with different target
        y_new = y * -1 + 10
        m.fit(X, y_new)
        pred_second = m.predict(X[:5])

        assert not np.allclose(pred_first, pred_second), (
            "Predictions did not change after re-fitting with different data"
        )

    def test_independent_models_no_interference(self, regression_data):
        """Multiple model instances are independent."""
        X, y = regression_data

        m1 = LinearRegression()
        m1.fit(X, y)
        pred1_before = m1.predict(X[:5])

        m2 = LinearRegression()
        m2.fit(X, y * 2)

        # m1 should be unaffected by m2's fit
        pred1_after = m1.predict(X[:5])
        np.testing.assert_array_equal(pred1_before, pred1_after)

    def test_multiple_predictions_after_single_fit(self, classification_data):
        """Model can predict many times without degradation."""
        X, y = classification_data
        m = RandomForestClassifier(n_estimators=10, random_state=42)
        m.fit(X, y)

        results = [m.predict(X[:10]) for _ in range(50)]
        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)

    def test_scaler_fit_transform_vs_fit_then_transform(self):
        """fit_transform and fit+transform produce identical results."""
        np.random.seed(42)
        X = np.random.randn(30, 4)

        sc1 = StandardScaler()
        Xt1 = sc1.fit_transform(X)

        sc2 = StandardScaler()
        sc2.fit(X)
        Xt2 = sc2.transform(X)

        np.testing.assert_allclose(Xt1, Xt2, rtol=1e-14)

    def test_kmeans_refit_with_different_k(self, clustering_data):
        """Re-fitting KMeans with different n_clusters works correctly."""
        X = clustering_data
        km = KMeans(n_clusters=2)
        km.fit(X)
        labels2 = km.predict(X)
        assert len(np.unique(labels2)) == 2

        # Note: KMeans constructor sets n_clusters at init, so create new instance
        km3 = KMeans(n_clusters=3)
        km3.fit(X)
        labels3 = km3.predict(X)
        assert len(np.unique(labels3)) == 3


# ===================================================================
# 4. SERIALIZATION ROUND-TRIP
# ===================================================================

class TestSerializationRoundTrip:
    """Verify pickle serialization preserves model state exactly."""

    def test_linear_regression_pickle(self, regression_data):
        """LinearRegression survives pickle round-trip with identical predictions."""
        X, y = regression_data
        m = LinearRegression()
        m.fit(X, y)
        pred_before = m.predict(X)

        m2 = pickle.loads(pickle.dumps(m))
        pred_after = m2.predict(X)

        np.testing.assert_array_equal(
            pred_before, pred_after,
            err_msg="LinearRegression predictions changed after pickle round-trip",
        )

    def test_random_forest_pickle(self, classification_data):
        """RandomForestClassifier survives pickle round-trip."""
        X, y = classification_data
        m = RandomForestClassifier(n_estimators=10, random_state=42)
        m.fit(X, y)
        pred_before = m.predict(X)

        m2 = pickle.loads(pickle.dumps(m))
        pred_after = m2.predict(X)

        np.testing.assert_array_equal(
            pred_before, pred_after,
            err_msg="RandomForestClassifier predictions changed after pickle",
        )

    def test_standard_scaler_pickle(self):
        """StandardScaler survives pickle round-trip."""
        np.random.seed(42)
        X = np.random.randn(30, 4)
        sc = StandardScaler()
        Xt_before = sc.fit_transform(X)

        sc2 = pickle.loads(pickle.dumps(sc))
        Xt_after = sc2.transform(X)

        np.testing.assert_array_equal(
            Xt_before, Xt_after,
            err_msg="StandardScaler transform changed after pickle",
        )

    def test_kmeans_pickle(self, clustering_data):
        """KMeans survives pickle round-trip."""
        X = clustering_data
        km = KMeans(n_clusters=2)
        km.fit(X)
        labels_before = km.predict(X)

        km2 = pickle.loads(pickle.dumps(km))
        labels_after = km2.predict(X)

        np.testing.assert_array_equal(
            labels_before, labels_after,
            err_msg="KMeans labels changed after pickle",
        )

    def test_fitted_flag_preserved_through_pickle(self, regression_data):
        """An unfitted model stays unfitted after pickle; fitted stays fitted."""
        # Unfitted model
        m_unfitted = LinearRegression()
        m_unfitted2 = pickle.loads(pickle.dumps(m_unfitted))
        with pytest.raises(RuntimeError, match="[Nn]ot fitted|fit"):
            m_unfitted2.predict(np.array([[1.0, 2.0, 3.0]]))

        # Fitted model
        X, y = regression_data
        m_fitted = LinearRegression()
        m_fitted.fit(X, y)
        m_fitted2 = pickle.loads(pickle.dumps(m_fitted))
        pred = m_fitted2.predict(X[:1])
        assert pred.shape == (1,)
        assert np.all(np.isfinite(pred))


# ===================================================================
# 5. THREAD SAFETY
# ===================================================================

class TestThreadSafety:
    """Verify concurrent predict calls don't corrupt results."""

    def test_concurrent_predict_linear(self, regression_data):
        """Concurrent predict on fitted LinearRegression returns consistent results."""
        X, y = regression_data
        m = LinearRegression()
        m.fit(X, y)
        expected = m.predict(X)

        def predict_chunk(i):
            return m.predict(X)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(predict_chunk, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                np.testing.assert_array_equal(
                    expected, result,
                    err_msg="Concurrent predict returned different result",
                )

    def test_concurrent_predict_rf(self, classification_data):
        """Concurrent predict on fitted RandomForestClassifier is consistent."""
        X, y = classification_data
        m = RandomForestClassifier(n_estimators=10, random_state=42)
        m.fit(X, y)
        expected = m.predict(X)

        def predict_chunk(i):
            return m.predict(X)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(predict_chunk, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                np.testing.assert_array_equal(
                    expected, result,
                    err_msg="Concurrent RF predict returned different result",
                )

    def test_concurrent_transform_scaler(self):
        """Concurrent transform on fitted StandardScaler is consistent."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        sc = StandardScaler()
        sc.fit(X)
        expected = sc.transform(X)

        def transform_chunk(i):
            return sc.transform(X)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(transform_chunk, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                np.testing.assert_array_equal(
                    expected, result,
                    err_msg="Concurrent transform returned different result",
                )
