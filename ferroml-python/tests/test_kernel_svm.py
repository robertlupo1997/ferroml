"""Tests for kernel SVC and SVR Python bindings."""

import numpy as np
import pytest

from ferroml.svm import SVC, SVR


# =============================================================================
# Test data helpers
# =============================================================================

def make_binary_classification_data(n_samples=100, n_features=4, seed=42):
    """Create well-separated binary classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    w = rng.randn(n_features)
    y = (X @ w > 0).astype(np.float64)
    # Shift classes apart for easier separation
    X[y == 0] -= 1.0
    X[y == 1] += 1.0
    return X, y


def make_multiclass_data(n_samples=120, n_features=4, n_classes=3, seed=42):
    """Create well-separated multiclass classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = np.repeat(np.arange(n_classes, dtype=np.float64), n_samples // n_classes)
    for c in range(n_classes):
        mask = y == c
        X[mask] += c * 3.0
    return X, y


def make_regression_data(n_samples=100, n_features=4, seed=42):
    """Create regression data with clear signal."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    w = np.array([3.0, -2.0, 1.0, 0.5])[:n_features]
    y = X @ w + 0.1 * rng.randn(n_samples)
    return X, y


# =============================================================================
# SVC tests
# =============================================================================

class TestSVC:
    def test_fit_predict_binary(self):
        X, y = make_binary_classification_data()
        model = SVC()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        unique = np.unique(preds)
        assert len(unique) <= 2

    def test_fit_predict_binary_accuracy(self):
        """Well-separated data should achieve decent accuracy."""
        X, y = make_binary_classification_data(n_samples=200, seed=123)
        model = SVC(c=10.0)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.8

    def test_fit_returns_self(self):
        X, y = make_binary_classification_data()
        model = SVC()
        result = model.fit(X, y)
        assert result is model

    def test_multiclass_ovo(self):
        X, y = make_multiclass_data()
        model = SVC(multiclass="ovo")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        unique = np.unique(preds)
        assert len(unique) <= 3

    def test_multiclass_ovr(self):
        X, y = make_multiclass_data()
        model = SVC(multiclass="ovr")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        unique = np.unique(preds)
        assert len(unique) <= 3

    def test_predict_proba_enabled(self):
        X, y = make_binary_classification_data()
        model = SVC(probability=True)
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert probas.shape == (X.shape[0], 2)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        # All probabilities non-negative
        assert np.all(probas >= 0.0)

    def test_predict_proba_disabled_raises(self):
        X, y = make_binary_classification_data()
        model = SVC(probability=False)
        model.fit(X, y)
        with pytest.raises(ValueError, match="[Pp]robability"):
            model.predict_proba(X)

    def test_kernel_rbf(self):
        X, y = make_binary_classification_data()
        model = SVC(kernel="rbf")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_kernel_linear(self):
        X, y = make_binary_classification_data()
        model = SVC(kernel="linear")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_kernel_poly(self):
        X, y = make_binary_classification_data()
        model = SVC(kernel="poly", degree=3, coef0=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_kernel_sigmoid(self):
        X, y = make_binary_classification_data()
        model = SVC(kernel="sigmoid", gamma=0.01, coef0=0.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_gamma_explicit(self):
        X, y = make_binary_classification_data()
        model = SVC(kernel="rbf", gamma=0.5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_n_support_vectors(self):
        X, y = make_binary_classification_data()
        model = SVC()
        model.fit(X, y)
        n_sv = model.n_support_vectors
        assert isinstance(n_sv, list)
        assert len(n_sv) > 0
        assert all(isinstance(v, int) for v in n_sv)

    def test_classes_property(self):
        X, y = make_binary_classification_data()
        model = SVC()
        model.fit(X, y)
        classes = model.classes_
        assert isinstance(classes, np.ndarray)
        np.testing.assert_array_equal(np.sort(classes), [0.0, 1.0])

    def test_classes_unfitted_raises(self):
        model = SVC()
        with pytest.raises(RuntimeError):
            _ = model.classes_

    def test_predict_unfitted_raises(self):
        model = SVC()
        X = np.random.randn(10, 4)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_integer_labels(self):
        X, _ = make_binary_classification_data()
        y = np.array([0, 1] * 50, dtype=np.int32)
        model = SVC()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_class_weight_balanced(self):
        X, y = make_binary_classification_data()
        model = SVC(class_weight="balanced")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_repr(self):
        model = SVC(kernel="rbf", c=2.0, probability=True)
        r = repr(model)
        assert "SVC" in r
        assert "rbf" in r
        assert "C=2" in r
        assert "probability=true" in r

    def test_repr_defaults(self):
        model = SVC()
        r = repr(model)
        assert "SVC" in r
        assert "rbf" in r
        assert "C=1" in r
        assert "probability=false" in r

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            SVC(kernel="invalid")

    def test_invalid_multiclass_raises(self):
        with pytest.raises(ValueError, match="Unknown multiclass"):
            SVC(multiclass="invalid")

    def test_invalid_class_weight_raises(self):
        with pytest.raises(ValueError, match="Unknown class_weight"):
            SVC(class_weight="invalid")


# =============================================================================
# SVR tests
# =============================================================================

class TestSVR:
    def test_fit_predict(self):
        X, y = make_regression_data()
        model = SVR()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_fit_returns_self(self):
        X, y = make_regression_data()
        model = SVR()
        result = model.fit(X, y)
        assert result is model

    def test_prediction_quality(self):
        """On well-structured data, predictions should correlate with targets."""
        X, y = make_regression_data(n_samples=200, seed=99)
        model = SVR(c=10.0, max_iter=2000)
        model.fit(X, y)
        preds = model.predict(X)
        correlation = np.corrcoef(y, preds)[0, 1]
        assert correlation > 0.7

    def test_kernel_rbf(self):
        X, y = make_regression_data()
        model = SVR(kernel="rbf")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_kernel_linear(self):
        X, y = make_regression_data()
        model = SVR(kernel="linear")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_kernel_poly(self):
        X, y = make_regression_data()
        model = SVR(kernel="poly", degree=2, gamma=0.1, coef0=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_epsilon_parameter(self):
        X, y = make_regression_data()
        model_small = SVR(epsilon=0.01)
        model_large = SVR(epsilon=1.0)
        model_small.fit(X, y)
        model_large.fit(X, y)
        # Both should produce valid predictions
        preds_small = model_small.predict(X)
        preds_large = model_large.predict(X)
        assert preds_small.shape == preds_large.shape

    def test_n_support_vectors(self):
        X, y = make_regression_data()
        model = SVR()
        model.fit(X, y)
        n_sv = model.n_support_vectors
        assert isinstance(n_sv, int)
        assert n_sv >= 0

    def test_support_indices(self):
        X, y = make_regression_data()
        model = SVR()
        model.fit(X, y)
        indices = model.support_indices_
        assert isinstance(indices, list)
        # All indices should be valid
        for idx in indices:
            assert 0 <= idx < X.shape[0]

    def test_dual_coef(self):
        X, y = make_regression_data()
        model = SVR()
        model.fit(X, y)
        coef = model.dual_coef_
        assert isinstance(coef, np.ndarray)
        n_sv = model.n_support_vectors
        assert coef.shape == (n_sv,)

    def test_intercept(self):
        X, y = make_regression_data()
        model = SVR()
        model.fit(X, y)
        intercept = model.intercept_
        assert isinstance(intercept, float)

    def test_support_indices_unfitted_raises(self):
        model = SVR()
        with pytest.raises(RuntimeError):
            _ = model.support_indices_

    def test_dual_coef_unfitted_raises(self):
        model = SVR()
        with pytest.raises(RuntimeError):
            _ = model.dual_coef_

    def test_predict_unfitted_raises(self):
        model = SVR()
        X = np.random.randn(10, 4)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_repr(self):
        model = SVR(kernel="rbf", c=2.0, epsilon=0.5)
        r = repr(model)
        assert r == "SVR(kernel='rbf', C=2, epsilon=0.5)"

    def test_repr_defaults(self):
        model = SVR()
        r = repr(model)
        assert "SVR" in r
        assert "rbf" in r
        assert "C=1" in r
        assert "epsilon=0.1" in r

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            SVR(kernel="invalid")

    def test_c_parameter_effect(self):
        """Different C values should produce different models."""
        X, y = make_regression_data()
        model_low = SVR(c=0.01)
        model_high = SVR(c=100.0)
        model_low.fit(X, y)
        model_high.fit(X, y)
        preds_low = model_low.predict(X)
        preds_high = model_high.predict(X)
        # Predictions should differ
        assert not np.allclose(preds_low, preds_high, atol=1e-3)


# =============================================================================
# P.2 tests: SVC custom class weights
# =============================================================================

class TestSVCCustomClassWeight:
    def test_svc_custom_class_weight_dict(self):
        X, y = make_binary_classification_data()
        model = SVC(class_weight={0.0: 1.0, 1.0: 10.0})
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_svc_custom_class_weight_invalid(self):
        with pytest.raises(ValueError):
            SVC(class_weight=42)


# =============================================================================
# P.3 tests: SVC/SVR decision_function
# =============================================================================

class TestSVCDecisionFunction:
    def test_svc_decision_function_binary_shape(self):
        X, y = make_binary_classification_data()
        model = SVC(multiclass="ovo")
        model.fit(X, y)
        decisions = model.decision_function(X)
        # Binary OvO: 1 classifier
        assert decisions.shape == (X.shape[0], 1)

    def test_svc_decision_function_multiclass_ovo_shape(self):
        X, y = make_multiclass_data()
        model = SVC(multiclass="ovo")
        model.fit(X, y)
        decisions = model.decision_function(X)
        n_classes = 3
        n_pairs = n_classes * (n_classes - 1) // 2
        assert decisions.shape == (X.shape[0], n_pairs)

    def test_svc_decision_function_multiclass_ovr_shape(self):
        X, y = make_multiclass_data()
        model = SVC(multiclass="ovr")
        model.fit(X, y)
        decisions = model.decision_function(X)
        assert decisions.shape == (X.shape[0], 3)

    def test_svc_decision_function_unfitted(self):
        model = SVC()
        X = np.random.randn(10, 4)
        with pytest.raises(RuntimeError):
            model.decision_function(X)

    def test_svr_decision_function_shape(self):
        X, y = make_regression_data()
        model = SVR()
        model.fit(X, y)
        decisions = model.decision_function(X)
        assert decisions.shape == (X.shape[0],)
        # Should match predict
        preds = model.predict(X)
        np.testing.assert_allclose(decisions, preds, atol=1e-10)

    def test_svr_decision_function_unfitted(self):
        model = SVR()
        X = np.random.randn(10, 4)
        with pytest.raises(RuntimeError):
            model.decision_function(X)
