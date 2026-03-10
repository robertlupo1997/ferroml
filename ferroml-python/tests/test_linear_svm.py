"""Tests for LinearSVC, LinearSVR, and TemperatureScalingCalibrator Python bindings."""

import numpy as np
import pytest

from ferroml.svm import LinearSVC, LinearSVR
from ferroml.calibration import TemperatureScalingCalibrator


# =============================================================================
# Test data helpers
# =============================================================================

def make_binary_classification_data(n_samples=100, n_features=5, seed=42):
    """Create linearly separable binary classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    w = rng.randn(n_features)
    y = (X @ w > 0).astype(np.float64)
    return X, y


def make_multiclass_data(n_samples=150, n_features=4, n_classes=3, seed=42):
    """Create multiclass classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = np.repeat(np.arange(n_classes, dtype=np.float64), n_samples // n_classes)
    # Shift features by class to make separable
    for c in range(n_classes):
        mask = y == c
        X[mask] += c * 2
    return X, y


def make_regression_data(n_samples=100, n_features=5, seed=42):
    """Create regression data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    w = rng.randn(n_features)
    y = X @ w + 0.1 * rng.randn(n_samples)
    return X, y


# =============================================================================
# LinearSVC tests
# =============================================================================

class TestLinearSVC:
    def test_fit_predict_binary(self):
        X, y = make_binary_classification_data()
        model = LinearSVC()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        # Predictions should be class labels
        unique = np.unique(preds)
        assert len(unique) <= 2

    def test_fit_predict_multiclass(self):
        X, y = make_multiclass_data()
        model = LinearSVC()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        unique = np.unique(preds)
        assert len(unique) <= 3

    def test_fit_returns_self(self):
        X, y = make_binary_classification_data()
        model = LinearSVC()
        result = model.fit(X, y)
        assert result is model

    def test_c_parameter_effect(self):
        X, y = make_binary_classification_data()
        model_low = LinearSVC(c=0.01)
        model_high = LinearSVC(c=100.0)
        model_low.fit(X, y)
        model_high.fit(X, y)
        preds_low = model_low.predict(X)
        preds_high = model_high.predict(X)
        # Both should produce valid predictions
        assert preds_low.shape == preds_high.shape

    def test_constructor_defaults(self):
        model = LinearSVC()
        r = repr(model)
        assert "C=1" in r
        assert "max_iter=1000" in r

    def test_constructor_custom(self):
        model = LinearSVC(c=0.5, max_iter=500, tol=1e-5)
        r = repr(model)
        assert "C=0.5" in r
        assert "max_iter=500" in r

    def test_repr(self):
        model = LinearSVC(c=2.0, max_iter=200)
        r = repr(model)
        assert r == "LinearSVC(C=2, loss='squared_hinge', max_iter=200)"

    def test_predict_unfitted_raises(self):
        model = LinearSVC()
        X = np.random.randn(10, 5)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_integer_labels(self):
        X, _ = make_binary_classification_data()
        y = np.array([0, 1] * 50, dtype=np.int32)
        model = LinearSVC()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_accuracy_reasonable(self):
        """On well-separated data, accuracy should be decent."""
        X, y = make_binary_classification_data(n_samples=200, seed=123)
        model = LinearSVC(c=1.0, max_iter=2000)
        model.fit(X, y)
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7


# =============================================================================
# LinearSVR tests
# =============================================================================

class TestLinearSVR:
    def test_fit_predict(self):
        X, y = make_regression_data()
        model = LinearSVR()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_fit_returns_self(self):
        X, y = make_regression_data()
        model = LinearSVR()
        result = model.fit(X, y)
        assert result is model

    def test_coef_getter(self):
        X, y = make_regression_data(n_features=5)
        model = LinearSVR()
        model.fit(X, y)
        coef = model.coef_
        assert coef.shape == (5,)

    def test_intercept_getter(self):
        X, y = make_regression_data()
        model = LinearSVR()
        model.fit(X, y)
        intercept = model.intercept_
        assert isinstance(intercept, float)

    def test_coef_unfitted_raises(self):
        model = LinearSVR()
        with pytest.raises(RuntimeError):
            _ = model.coef_

    def test_predict_unfitted_raises(self):
        model = LinearSVR()
        X = np.random.randn(10, 5)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_epsilon_parameter(self):
        X, y = make_regression_data()
        model = LinearSVR(epsilon=0.5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_constructor_defaults(self):
        model = LinearSVR()
        r = repr(model)
        assert "C=1" in r
        assert "epsilon=0" in r

    def test_repr(self):
        model = LinearSVR(c=0.5, epsilon=0.1)
        r = repr(model)
        assert r == "LinearSVR(C=0.5, epsilon=0.1, loss='epsilon_insensitive')"

    def test_prediction_quality(self):
        """On linear data, predictions should correlate well."""
        X, y = make_regression_data(n_samples=200, seed=99)
        model = LinearSVR(c=10.0, max_iter=2000)
        model.fit(X, y)
        preds = model.predict(X)
        correlation = np.corrcoef(y, preds)[0, 1]
        assert correlation > 0.8


# =============================================================================
# P.1 tests: LinearSVC/LinearSVR loss parameter
# =============================================================================

class TestLinearSVCLoss:
    def test_linear_svc_loss_hinge(self):
        X, y = make_binary_classification_data()
        model = LinearSVC(loss="hinge")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_svc_loss_squared_hinge(self):
        X, y = make_binary_classification_data()
        model = LinearSVC(loss="squared_hinge")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_svc_loss_invalid(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            LinearSVC(loss="invalid")

    def test_linear_svc_loss_in_repr(self):
        model = LinearSVC(loss="hinge")
        r = repr(model)
        assert "loss='hinge'" in r


class TestLinearSVRLoss:
    def test_linear_svr_loss_epsilon_insensitive(self):
        X, y = make_regression_data()
        model = LinearSVR(loss="epsilon_insensitive")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_svr_loss_squared(self):
        X, y = make_regression_data()
        model = LinearSVR(loss="squared_epsilon_insensitive")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_svr_loss_invalid(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            LinearSVR(loss="invalid")

    def test_linear_svr_loss_in_repr(self):
        model = LinearSVR(loss="squared_epsilon_insensitive")
        r = repr(model)
        assert "loss='squared_epsilon_insensitive'" in r


# =============================================================================
# P.2 tests: LinearSVC class weights
# =============================================================================

class TestLinearSVCClassWeight:
    def test_linear_svc_class_weight_balanced(self):
        X, y = make_binary_classification_data()
        model = LinearSVC(class_weight="balanced")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_svc_class_weight_dict(self):
        X, y = make_binary_classification_data()
        model = LinearSVC(class_weight={0.0: 1.0, 1.0: 10.0})
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_svc_class_weight_invalid(self):
        with pytest.raises(ValueError):
            LinearSVC(class_weight="invalid")


# =============================================================================
# P.3 tests: decision_function
# =============================================================================

class TestLinearSVCDecisionFunction:
    def test_linear_svc_decision_function_shape(self):
        X, y = make_binary_classification_data()
        model = LinearSVC()
        model.fit(X, y)
        decisions = model.decision_function(X)
        # Binary: 1 classifier
        assert decisions.shape == (X.shape[0], 1)

    def test_linear_svc_decision_function_unfitted(self):
        model = LinearSVC()
        X = np.random.randn(10, 5)
        with pytest.raises(RuntimeError):
            model.decision_function(X)


class TestLinearSVRDecisionFunction:
    def test_linear_svr_decision_function_shape(self):
        X, y = make_regression_data()
        model = LinearSVR()
        model.fit(X, y)
        decisions = model.decision_function(X)
        assert decisions.shape == (X.shape[0],)
        # Should match predict output
        preds = model.predict(X)
        np.testing.assert_allclose(decisions, preds, atol=1e-10)

    def test_linear_svr_decision_function_unfitted(self):
        model = LinearSVR()
        X = np.random.randn(10, 5)
        with pytest.raises(RuntimeError):
            model.decision_function(X)


# =============================================================================
# TemperatureScalingCalibrator tests
# =============================================================================

class TestTemperatureScalingCalibrator:
    def test_fit_transform_basic(self):
        np.random.seed(42)
        n_samples, n_classes = 100, 3
        y_prob = np.random.dirichlet([0.1] * n_classes, size=n_samples)
        y_true = np.array([np.argmax(row) for row in y_prob], dtype=np.float64)

        cal = TemperatureScalingCalibrator()
        cal.fit(y_prob, y_true)
        calibrated = cal.transform(y_prob)
        assert calibrated.shape == y_prob.shape

    def test_calibrated_sums_to_one(self):
        np.random.seed(42)
        n_samples, n_classes = 100, 3
        y_prob = np.random.dirichlet([0.1] * n_classes, size=n_samples)
        y_true = np.array([np.argmax(row) for row in y_prob], dtype=np.float64)

        cal = TemperatureScalingCalibrator()
        cal.fit(y_prob, y_true)
        calibrated = cal.transform(y_prob)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)

    def test_temperature_getter(self):
        np.random.seed(42)
        n_samples, n_classes = 100, 3
        y_prob = np.random.dirichlet([0.1] * n_classes, size=n_samples)
        y_true = np.array([np.argmax(row) for row in y_prob], dtype=np.float64)

        cal = TemperatureScalingCalibrator()
        cal.fit(y_prob, y_true)
        assert cal.temperature_ > 0

    def test_temperature_unfitted_raises(self):
        cal = TemperatureScalingCalibrator()
        with pytest.raises(RuntimeError):
            _ = cal.temperature_

    def test_transform_unfitted_raises(self):
        cal = TemperatureScalingCalibrator()
        y_prob = np.random.dirichlet([1.0, 1.0, 1.0], size=10)
        with pytest.raises(RuntimeError):
            cal.transform(y_prob)

    def test_fit_returns_self(self):
        np.random.seed(42)
        n_samples, n_classes = 50, 2
        y_prob = np.random.dirichlet([0.5] * n_classes, size=n_samples)
        y_true = np.array([np.argmax(row) for row in y_prob], dtype=np.float64)

        cal = TemperatureScalingCalibrator()
        result = cal.fit(y_prob, y_true)
        assert result is cal

    def test_custom_parameters(self):
        cal = TemperatureScalingCalibrator(max_iter=200, learning_rate=0.05)
        r = repr(cal)
        assert "unfitted" in r

    def test_repr_fitted(self):
        np.random.seed(42)
        n_samples, n_classes = 50, 3
        y_prob = np.random.dirichlet([0.1] * n_classes, size=n_samples)
        y_true = np.array([np.argmax(row) for row in y_prob], dtype=np.float64)

        cal = TemperatureScalingCalibrator()
        cal.fit(y_prob, y_true)
        r = repr(cal)
        assert "temperature=" in r
        assert "unfitted" not in r

    def test_integer_labels(self):
        np.random.seed(42)
        n_samples, n_classes = 80, 4
        y_prob = np.random.dirichlet([0.5] * n_classes, size=n_samples)
        y_true = np.array([np.argmax(row) for row in y_prob], dtype=np.int32)

        cal = TemperatureScalingCalibrator()
        cal.fit(y_prob, y_true)
        calibrated = cal.transform(y_prob)
        assert calibrated.shape == y_prob.shape
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)

    def test_all_probabilities_positive(self):
        np.random.seed(42)
        n_samples, n_classes = 100, 3
        y_prob = np.random.dirichlet([0.1] * n_classes, size=n_samples)
        y_true = np.array([np.argmax(row) for row in y_prob], dtype=np.float64)

        cal = TemperatureScalingCalibrator()
        cal.fit(y_prob, y_true)
        calibrated = cal.transform(y_prob)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)
