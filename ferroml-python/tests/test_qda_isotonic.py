"""Tests for QuadraticDiscriminantAnalysis and IsotonicRegression Python bindings."""

import numpy as np
import pytest


# =============================================================================
# QDA Tests
# =============================================================================


class TestQuadraticDiscriminantAnalysis:
    """Tests for QDA Python binding."""

    def test_import(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        qda = QuadraticDiscriminantAnalysis()
        assert repr(qda) == "QuadraticDiscriminantAnalysis()"

    def test_basic_fit_predict(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        X = np.array([
            [1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [1.5, 2.5],
            [6.0, 7.0], [7.0, 8.0], [8.0, 7.0], [7.5, 7.5],
        ])
        y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        preds = qda.predict(X)
        assert preds.shape == (8,)
        np.testing.assert_array_equal(preds[:4], 0.0)
        np.testing.assert_array_equal(preds[4:], 1.0)

    def test_predict_proba(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        X = np.array([
            [1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [1.5, 2.5],
            [6.0, 7.0], [7.0, 8.0], [8.0, 7.0], [7.5, 7.5],
        ])
        y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        proba = qda.predict_proba(X)
        assert proba.shape == (8, 2)
        # Rows sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        # All probabilities in [0, 1]
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_decision_function(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        X = np.array([
            [1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0],
        ])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        qda = QuadraticDiscriminantAnalysis(reg_param=0.5)
        qda.fit(X, y)
        scores = qda.decision_function(X)
        assert scores.shape == (4, 2)

    def test_multiclass(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(0, 1, (20, 2)),
            rng.normal(5, 1, (20, 2)),
            rng.normal([0, 5], 1, (20, 2)),
        ])
        y = np.array([0.0] * 20 + [1.0] * 20 + [2.0] * 20)
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        preds = qda.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.85, f"accuracy {accuracy:.2f} too low"

    def test_reg_param(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        X = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 6.0], [5.1, 6.1]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        # Should work with regularization
        qda = QuadraticDiscriminantAnalysis(reg_param=0.5)
        qda.fit(X, y)
        preds = qda.predict(X)
        assert preds[0] == 0.0
        assert preds[3] == 1.0

    def test_custom_priors(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        X = np.array([
            [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
            [4.0, 4.0], [5.0, 5.0], [6.0, 6.0],
        ])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        qda = QuadraticDiscriminantAnalysis(priors=[0.99, 0.01], reg_param=0.5)
        qda.fit(X, y)
        # Should still produce valid predictions
        preds = qda.predict(X)
        assert preds.shape == (6,)

    def test_not_fitted_error(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        qda = QuadraticDiscriminantAnalysis()
        X = np.array([[1.0, 2.0]])
        with pytest.raises(RuntimeError):
            qda.predict(X)

    def test_pickle_roundtrip(self):
        import pickle
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        X = np.array([
            [1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [1.5, 2.5],
            [6.0, 7.0], [7.0, 8.0], [8.0, 7.0], [7.5, 7.5],
        ])
        y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        p1 = qda.predict(X)

        data = pickle.dumps(qda)
        qda2 = pickle.loads(data)
        p2 = qda2.predict(X)
        np.testing.assert_array_equal(p1, p2)

    def test_integer_labels(self):
        from ferroml.decomposition import QuadraticDiscriminantAnalysis
        X = np.array([
            [1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [1.5, 2.5],
            [6.0, 7.0], [7.0, 8.0], [8.0, 7.0], [7.5, 7.5],
        ])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(X, y)
        preds = qda.predict(X)
        assert preds.shape == (8,)


# =============================================================================
# IsotonicRegression Tests
# =============================================================================


class TestIsotonicRegression:
    """Tests for IsotonicRegression Python binding."""

    def test_import(self):
        from ferroml.linear import IsotonicRegression
        iso = IsotonicRegression()
        assert repr(iso) == "IsotonicRegression()"

    def test_monotonically_increasing(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        iso = IsotonicRegression()
        iso.fit(X, y)
        preds = iso.predict(X)
        assert preds.shape == (5,)
        # Check monotonicity
        for i in range(1, len(preds)):
            assert preds[i] >= preds[i - 1] - 1e-10

    def test_monotonically_decreasing(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([5.0, 3.0, 4.0, 1.0, 2.0])
        iso = IsotonicRegression(increasing="false")
        iso.fit(X, y)
        preds = iso.predict(X)
        for i in range(1, len(preds)):
            assert preds[i] <= preds[i - 1] + 1e-10

    def test_auto_increasing(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        iso = IsotonicRegression(increasing="auto")
        iso.fit(X, y)
        preds = iso.predict(X)
        # Should be increasing (positive correlation)
        for i in range(1, len(preds)):
            assert preds[i] >= preds[i - 1] - 1e-10

    def test_y_min_y_max(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0.0, 10.0, 5.0, 20.0, 15.0])
        iso = IsotonicRegression(y_min=2.0, y_max=12.0)
        iso.fit(X, y)
        preds = iso.predict(X)
        assert np.all(preds >= 2.0 - 1e-10)
        assert np.all(preds <= 12.0 + 1e-10)

    def test_out_of_bounds_nan(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0])
        iso = IsotonicRegression(out_of_bounds="nan")
        iso.fit(X, y)
        X_test = np.array([[1.0], [5.0]])
        preds = iso.predict(X_test)
        assert np.isnan(preds[0])
        assert np.isnan(preds[1])

    def test_out_of_bounds_clip(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0])
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X, y)
        X_test = np.array([[1.0], [5.0]])
        preds = iso.predict(X_test)
        np.testing.assert_allclose(preds[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(preds[1], 3.0, atol=1e-10)

    def test_out_of_bounds_raise(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0])
        iso = IsotonicRegression(out_of_bounds="raise")
        iso.fit(X, y)
        X_test = np.array([[1.0]])
        with pytest.raises(ValueError):
            iso.predict(X_test)

    def test_interpolation(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        iso = IsotonicRegression()
        iso.fit(X, y)
        X_test = np.array([[0.5], [1.5], [2.5]])
        preds = iso.predict(X_test)
        np.testing.assert_allclose(preds, [0.5, 1.5, 2.5], atol=1e-10)

    def test_not_fitted_error(self):
        from ferroml.linear import IsotonicRegression
        iso = IsotonicRegression()
        X = np.array([[1.0]])
        with pytest.raises(RuntimeError):
            iso.predict(X)

    def test_multi_column_error(self):
        from ferroml.linear import IsotonicRegression
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        iso = IsotonicRegression()
        with pytest.raises(ValueError):
            iso.fit(X, y)

    def test_pickle_roundtrip(self):
        import pickle
        from ferroml.linear import IsotonicRegression
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        iso = IsotonicRegression()
        iso.fit(X, y)
        p1 = iso.predict(X)

        data = pickle.dumps(iso)
        iso2 = pickle.loads(data)
        p2 = iso2.predict(X)
        np.testing.assert_array_equal(p1, p2)

    def test_invalid_increasing(self):
        from ferroml.linear import IsotonicRegression
        with pytest.raises(ValueError):
            IsotonicRegression(increasing="invalid")

    def test_invalid_out_of_bounds(self):
        from ferroml.linear import IsotonicRegression
        with pytest.raises(ValueError):
            IsotonicRegression(out_of_bounds="invalid")
