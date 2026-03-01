"""Tests for KernelSHAP (model-agnostic SHAP values) in ferroml.explainability.

KernelSHAP is exposed as 10 typed functions, one per supported model class.
All functions share the same signature and return structure:

    result = kernel_shap_<variant>(model, background, x,
                                   n_samples=None, random_state=None)

Return value is always a dict with keys:
    "base_value"     -- float scalar
    "shap_values"    -- ndarray of shape (n_instances, n_features)
    "feature_values" -- ndarray of shape (n_instances, n_features)
"""

import numpy as np
import pytest

from ferroml.explainability import (
    kernel_shap_dt_clf,
    kernel_shap_dt_reg,
    kernel_shap_et_clf,
    kernel_shap_et_reg,
    kernel_shap_gb_clf,
    kernel_shap_gb_reg,
    kernel_shap_linear,
    kernel_shap_logistic,
    kernel_shap_rf_clf,
    kernel_shap_rf_reg,
)
from ferroml.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from ferroml.linear import LinearRegression, LogisticRegression
from ferroml.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAMPLES = 80
N_FEATURES = 4
N_BACKGROUND = 20
N_EXPLAIN = 5

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reg_data():
    """Regression dataset: y = 2*X0 + 0.5*X1 + noise."""
    np.random.seed(42)
    X = np.random.randn(N_SAMPLES, N_FEATURES)
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + np.random.randn(N_SAMPLES) * 0.1
    return X, y


@pytest.fixture(scope="module")
def clf_data():
    """Binary classification dataset: y = (X0 + X1 > 0)."""
    np.random.seed(42)
    X = np.random.randn(N_SAMPLES, N_FEATURES)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture(scope="module")
def rf_reg(reg_data):
    """Fitted RandomForestRegressor."""
    X, y = reg_data
    m = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def rf_clf(clf_data):
    """Fitted RandomForestClassifier."""
    X, y = clf_data
    m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def dt_reg(reg_data):
    """Fitted DecisionTreeRegressor."""
    X, y = reg_data
    m = DecisionTreeRegressor(max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def dt_clf(clf_data):
    """Fitted DecisionTreeClassifier."""
    X, y = clf_data
    m = DecisionTreeClassifier(max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def gb_reg(reg_data):
    """Fitted GradientBoostingRegressor."""
    X, y = reg_data
    m = GradientBoostingRegressor(
        n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42
    )
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def gb_clf(clf_data):
    """Fitted GradientBoostingClassifier."""
    X, y = clf_data
    m = GradientBoostingClassifier(
        n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42
    )
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def lin_reg(reg_data):
    """Fitted LinearRegression."""
    X, y = reg_data
    m = LinearRegression()
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def log_reg(clf_data):
    """Fitted LogisticRegression."""
    X, y = clf_data
    m = LogisticRegression()
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def et_clf(clf_data):
    """Fitted ExtraTreesClassifier."""
    X, y = clf_data
    m = ExtraTreesClassifier(n_estimators=10, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def et_reg(reg_data):
    """Fitted ExtraTreesRegressor."""
    X, y = reg_data
    m = ExtraTreesRegressor(n_estimators=10, max_depth=3, random_state=42)
    m.fit(X, y)
    return m


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset(["base_value", "shap_values", "feature_values"])


def _assert_output_structure(result, n_explain, n_features):
    """Shared structural assertions for any kernel_shap_* result."""
    assert isinstance(result, dict), "Result must be a dict"
    assert _REQUIRED_KEYS.issubset(result.keys()), (
        f"Missing keys: {_REQUIRED_KEYS - result.keys()}"
    )
    sv = result["shap_values"]
    fv = result["feature_values"]
    assert isinstance(sv, np.ndarray), "shap_values must be ndarray"
    assert isinstance(fv, np.ndarray), "feature_values must be ndarray"
    assert sv.shape == (n_explain, n_features), (
        f"shap_values shape {sv.shape} != ({n_explain}, {n_features})"
    )
    assert fv.shape == (n_explain, n_features), (
        f"feature_values shape {fv.shape} != ({n_explain}, {n_features})"
    )
    assert np.all(np.isfinite(sv)), "shap_values contain non-finite values"
    assert np.isfinite(float(result["base_value"])), "base_value is not finite"


# ---------------------------------------------------------------------------
# kernel_shap_rf_reg
# ---------------------------------------------------------------------------


class TestKernelShapRfReg:
    """Tests for kernel_shap_rf_reg (RandomForestRegressor)."""

    def test_basic_output_structure(self, rf_reg, reg_data):
        """Result dict must have all three required keys."""
        X, _ = reg_data
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, rf_reg, reg_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, rf_reg, reg_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, rf_reg, reg_data):
        """All SHAP values must be finite (no NaN or Inf)."""
        X, _ = reg_data
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_rf_clf
# ---------------------------------------------------------------------------


class TestKernelShapRfClf:
    """Tests for kernel_shap_rf_clf (RandomForestClassifier)."""

    def test_basic_output_structure(self, rf_clf, clf_data):
        """Result dict must have all three required keys."""
        X, _ = clf_data
        result = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, rf_clf, clf_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, rf_clf, clf_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, rf_clf, clf_data):
        """All SHAP values must be finite."""
        X, _ = clf_data
        result = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_dt_reg
# ---------------------------------------------------------------------------


class TestKernelShapDtReg:
    """Tests for kernel_shap_dt_reg (DecisionTreeRegressor)."""

    def test_basic_output_structure(self, dt_reg, reg_data):
        """Result dict must have all three required keys."""
        X, _ = reg_data
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, dt_reg, reg_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, dt_reg, reg_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, dt_reg, reg_data):
        """All SHAP values must be finite."""
        X, _ = reg_data
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_dt_clf
# ---------------------------------------------------------------------------


class TestKernelShapDtClf:
    """Tests for kernel_shap_dt_clf (DecisionTreeClassifier)."""

    def test_basic_output_structure(self, dt_clf, clf_data):
        """Result dict must have all three required keys."""
        X, _ = clf_data
        result = kernel_shap_dt_clf(
            dt_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, dt_clf, clf_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_dt_clf(
            dt_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, dt_clf, clf_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_dt_clf(
            dt_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, dt_clf, clf_data):
        """All SHAP values must be finite."""
        X, _ = clf_data
        result = kernel_shap_dt_clf(
            dt_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_gb_reg
# ---------------------------------------------------------------------------


class TestKernelShapGbReg:
    """Tests for kernel_shap_gb_reg (GradientBoostingRegressor)."""

    def test_basic_output_structure(self, gb_reg, reg_data):
        """Result dict must have all three required keys."""
        X, _ = reg_data
        result = kernel_shap_gb_reg(
            gb_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, gb_reg, reg_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_gb_reg(
            gb_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, gb_reg, reg_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_gb_reg(
            gb_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, gb_reg, reg_data):
        """All SHAP values must be finite."""
        X, _ = reg_data
        result = kernel_shap_gb_reg(
            gb_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_gb_clf
# ---------------------------------------------------------------------------


class TestKernelShapGbClf:
    """Tests for kernel_shap_gb_clf (GradientBoostingClassifier)."""

    def test_basic_output_structure(self, gb_clf, clf_data):
        """Result dict must have all three required keys."""
        X, _ = clf_data
        result = kernel_shap_gb_clf(
            gb_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, gb_clf, clf_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_gb_clf(
            gb_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, gb_clf, clf_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_gb_clf(
            gb_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, gb_clf, clf_data):
        """All SHAP values must be finite."""
        X, _ = clf_data
        result = kernel_shap_gb_clf(
            gb_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_linear
# ---------------------------------------------------------------------------


class TestKernelShapLinear:
    """Tests for kernel_shap_linear (LinearRegression)."""

    def test_basic_output_structure(self, lin_reg, reg_data):
        """Result dict must have all three required keys."""
        X, _ = reg_data
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, lin_reg, reg_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, lin_reg, reg_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, lin_reg, reg_data):
        """All SHAP values must be finite."""
        X, _ = reg_data
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_logistic
# ---------------------------------------------------------------------------


class TestKernelShapLogistic:
    """Tests for kernel_shap_logistic (LogisticRegression)."""

    def test_basic_output_structure(self, log_reg, clf_data):
        """Result dict must have all three required keys."""
        X, _ = clf_data
        result = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, log_reg, clf_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, log_reg, clf_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, log_reg, clf_data):
        """All SHAP values must be finite."""
        X, _ = clf_data
        result = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_et_clf
# ---------------------------------------------------------------------------


class TestKernelShapEtClf:
    """Tests for kernel_shap_et_clf (ExtraTreesClassifier)."""

    def test_basic_output_structure(self, et_clf, clf_data):
        """Result dict must have all three required keys."""
        X, _ = clf_data
        result = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, et_clf, clf_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, et_clf, clf_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = clf_data
        result = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, et_clf, clf_data):
        """All SHAP values must be finite."""
        X, _ = clf_data
        result = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# kernel_shap_et_reg
# ---------------------------------------------------------------------------


class TestKernelShapEtReg:
    """Tests for kernel_shap_et_reg (ExtraTreesRegressor)."""

    def test_basic_output_structure(self, et_reg, reg_data):
        """Result dict must have all three required keys."""
        X, _ = reg_data
        result = kernel_shap_et_reg(
            et_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert _REQUIRED_KEYS.issubset(result.keys())

    def test_shap_values_shape(self, et_reg, reg_data):
        """shap_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_et_reg(
            et_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_feature_values_shape(self, et_reg, reg_data):
        """feature_values must have shape (n_explain, n_features)."""
        X, _ = reg_data
        result = kernel_shap_et_reg(
            et_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert result["feature_values"].shape == (N_EXPLAIN, N_FEATURES)

    def test_shap_values_finite(self, et_reg, reg_data):
        """All SHAP values must be finite."""
        X, _ = reg_data
        result = kernel_shap_et_reg(
            et_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# Cross-cutting tests: base_value is scalar
# ---------------------------------------------------------------------------


class TestBaseValueIsScalar:
    """base_value must be a Python/numpy scalar for every KernelSHAP variant."""

    def test_rf_reg_base_value_scalar(self, rf_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_rf_clf_base_value_scalar(self, rf_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_dt_reg_base_value_scalar(self, dt_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_dt_clf_base_value_scalar(self, dt_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_dt_clf(
            dt_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_gb_reg_base_value_scalar(self, gb_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_gb_reg(
            gb_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_gb_clf_base_value_scalar(self, gb_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_gb_clf(
            gb_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_linear_base_value_scalar(self, lin_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_logistic_base_value_scalar(self, log_reg, clf_data):
        X, _ = clf_data
        result = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_et_clf_base_value_scalar(self, et_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))

    def test_et_reg_base_value_scalar(self, et_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_et_reg(
            et_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        assert np.isfinite(float(result["base_value"]))


# ---------------------------------------------------------------------------
# Cross-cutting: single instance (n_explain = 1)
# ---------------------------------------------------------------------------


class TestSingleInstance:
    """All 10 KernelSHAP variants must work when explaining exactly 1 instance."""

    def test_rf_reg_single_instance(self, rf_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)
        assert result["feature_values"].shape == (1, N_FEATURES)

    def test_rf_clf_single_instance(self, rf_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)

    def test_dt_reg_single_instance(self, dt_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)

    def test_dt_clf_single_instance(self, dt_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_dt_clf(
            dt_clf, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)

    def test_gb_reg_single_instance(self, gb_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_gb_reg(
            gb_reg, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)

    def test_gb_clf_single_instance(self, gb_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_gb_clf(
            gb_clf, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)

    def test_linear_single_instance(self, lin_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)

    def test_logistic_single_instance(self, log_reg, clf_data):
        X, _ = clf_data
        result = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)

    def test_et_clf_single_instance(self, et_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)

    def test_et_reg_single_instance(self, et_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_et_reg(
            et_reg, X[:N_BACKGROUND], X[:1], n_samples=50, random_state=42
        )
        assert result["shap_values"].shape == (1, N_FEATURES)


# ---------------------------------------------------------------------------
# Cross-cutting: explicit n_samples parameter
# ---------------------------------------------------------------------------


class TestNSamplesParameter:
    """Passing an explicit n_samples value must not raise and must produce finite output."""

    def test_rf_reg_n_samples_100(self, rf_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=100, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))

    def test_dt_reg_n_samples_100(self, dt_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=100, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))

    def test_linear_n_samples_100(self, lin_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=100, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))

    def test_logistic_n_samples_100(self, log_reg, clf_data):
        X, _ = clf_data
        result = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=100, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))

    def test_et_clf_n_samples_100(self, et_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=100, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))

    def test_et_reg_n_samples_100(self, et_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_et_reg(
            et_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=100, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))

    def test_rf_clf_n_samples_100(self, rf_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=100, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))

    def test_gb_reg_n_samples_100(self, gb_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_gb_reg(
            gb_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=100, random_state=42
        )
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# Cross-cutting: random_state reproducibility
# ---------------------------------------------------------------------------


class TestRandomStateReproducibility:
    """Same random_state must produce identical SHAP values on repeated calls."""

    def test_rf_reg_reproducible(self, rf_reg, reg_data):
        X, _ = reg_data
        r1 = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        r2 = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        np.testing.assert_array_equal(r1["shap_values"], r2["shap_values"])

    def test_dt_reg_reproducible(self, dt_reg, reg_data):
        X, _ = reg_data
        r1 = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        r2 = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        np.testing.assert_array_equal(r1["shap_values"], r2["shap_values"])

    def test_linear_reproducible(self, lin_reg, reg_data):
        X, _ = reg_data
        r1 = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        r2 = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        np.testing.assert_array_equal(r1["shap_values"], r2["shap_values"])

    def test_logistic_reproducible(self, log_reg, clf_data):
        X, _ = clf_data
        r1 = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        r2 = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        np.testing.assert_array_equal(r1["shap_values"], r2["shap_values"])

    def test_et_clf_reproducible(self, et_clf, clf_data):
        X, _ = clf_data
        r1 = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        r2 = kernel_shap_et_clf(
            et_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        np.testing.assert_array_equal(r1["shap_values"], r2["shap_values"])

    def test_rf_clf_reproducible(self, rf_clf, clf_data):
        X, _ = clf_data
        r1 = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        r2 = kernel_shap_rf_clf(
            rf_clf, X[:N_BACKGROUND], X[:N_EXPLAIN], n_samples=50, random_state=7
        )
        np.testing.assert_array_equal(r1["shap_values"], r2["shap_values"])


# ---------------------------------------------------------------------------
# Cross-cutting: approximate SHAP additivity
# ---------------------------------------------------------------------------


class TestShapAdditivity:
    """base_value + sum(shap_values[i]) should approximately equal predict(x[i]).

    KernelSHAP is a Monte-Carlo method, so we use a generous tolerance (0.5).
    For classifiers the model returns probabilities, and KernelSHAP explains
    the same predict() output, so the additivity property still holds.
    """

    def test_rf_reg_additivity(self, rf_reg, reg_data):
        X, _ = reg_data
        x_explain = X[:N_EXPLAIN]
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], x_explain, n_samples=200, random_state=42
        )
        base = float(result["base_value"])
        sv = result["shap_values"]
        preds = rf_reg.predict(x_explain)
        for i in range(N_EXPLAIN):
            approx = base + float(np.sum(sv[i]))
            np.testing.assert_allclose(
                approx, float(preds[i]), atol=0.5,
                err_msg=f"Additivity violated at sample {i}: approx={approx:.4f}, pred={float(preds[i]):.4f}",
            )

    def test_dt_reg_additivity(self, dt_reg, reg_data):
        X, _ = reg_data
        x_explain = X[:N_EXPLAIN]
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], x_explain, n_samples=200, random_state=42
        )
        base = float(result["base_value"])
        sv = result["shap_values"]
        preds = dt_reg.predict(x_explain)
        for i in range(N_EXPLAIN):
            approx = base + float(np.sum(sv[i]))
            np.testing.assert_allclose(
                approx, float(preds[i]), atol=0.5,
                err_msg=f"Additivity violated at sample {i}",
            )

    def test_linear_additivity(self, lin_reg, reg_data):
        X, _ = reg_data
        x_explain = X[:N_EXPLAIN]
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], x_explain, n_samples=200, random_state=42
        )
        base = float(result["base_value"])
        sv = result["shap_values"]
        preds = lin_reg.predict(x_explain)
        for i in range(N_EXPLAIN):
            approx = base + float(np.sum(sv[i]))
            np.testing.assert_allclose(
                approx, float(preds[i]), atol=0.5,
                err_msg=f"Additivity violated at sample {i}",
            )

    def test_et_reg_additivity(self, et_reg, reg_data):
        X, _ = reg_data
        x_explain = X[:N_EXPLAIN]
        result = kernel_shap_et_reg(
            et_reg, X[:N_BACKGROUND], x_explain, n_samples=200, random_state=42
        )
        base = float(result["base_value"])
        sv = result["shap_values"]
        preds = et_reg.predict(x_explain)
        for i in range(N_EXPLAIN):
            approx = base + float(np.sum(sv[i]))
            np.testing.assert_allclose(
                approx, float(preds[i]), atol=0.5,
                err_msg=f"Additivity violated at sample {i}",
            )


# ---------------------------------------------------------------------------
# Cross-cutting: small background (5 samples)
# ---------------------------------------------------------------------------


class TestBackgroundSizeSmall:
    """KernelSHAP must work with a background of only 5 samples."""

    def test_rf_reg_small_background(self, rf_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_rf_reg(
            rf_reg, X[:5], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        _assert_output_structure(result, N_EXPLAIN, N_FEATURES)

    def test_dt_reg_small_background(self, dt_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_dt_reg(
            dt_reg, X[:5], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        _assert_output_structure(result, N_EXPLAIN, N_FEATURES)

    def test_linear_small_background(self, lin_reg, reg_data):
        X, _ = reg_data
        result = kernel_shap_linear(
            lin_reg, X[:5], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        _assert_output_structure(result, N_EXPLAIN, N_FEATURES)

    def test_rf_clf_small_background(self, rf_clf, clf_data):
        X, _ = clf_data
        result = kernel_shap_rf_clf(
            rf_clf, X[:5], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        _assert_output_structure(result, N_EXPLAIN, N_FEATURES)

    def test_logistic_small_background(self, log_reg, clf_data):
        X, _ = clf_data
        result = kernel_shap_logistic(
            log_reg, X[:5], X[:N_EXPLAIN], n_samples=50, random_state=42
        )
        _assert_output_structure(result, N_EXPLAIN, N_FEATURES)


# ---------------------------------------------------------------------------
# Cross-cutting: many features (10 features)
# ---------------------------------------------------------------------------


class TestManyFeatures:
    """KernelSHAP functions must handle wider datasets (10 features) correctly."""

    @pytest.fixture(scope="class")
    def wide_reg_data(self):
        np.random.seed(0)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = X[:, 0] * 3.0 + X[:, 2] * 1.5 + np.random.randn(n) * 0.1
        return X, y, p

    @pytest.fixture(scope="class")
    def wide_clf_data(self):
        np.random.seed(0)
        n, p = 100, 10
        X = np.random.randn(n, p)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
        return X, y, p

    def test_rf_reg_many_features(self, wide_reg_data):
        X, y, p = wide_reg_data
        m = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        m.fit(X, y)
        result = kernel_shap_rf_reg(m, X[:10], X[:3], n_samples=50, random_state=42)
        assert result["shap_values"].shape == (3, p)
        assert np.all(np.isfinite(result["shap_values"]))

    def test_dt_reg_many_features(self, wide_reg_data):
        X, y, p = wide_reg_data
        m = DecisionTreeRegressor(max_depth=3, random_state=42)
        m.fit(X, y)
        result = kernel_shap_dt_reg(m, X[:10], X[:3], n_samples=50, random_state=42)
        assert result["shap_values"].shape == (3, p)
        assert np.all(np.isfinite(result["shap_values"]))

    def test_linear_many_features(self, wide_reg_data):
        X, y, p = wide_reg_data
        m = LinearRegression()
        m.fit(X, y)
        result = kernel_shap_linear(m, X[:10], X[:3], n_samples=50, random_state=42)
        assert result["shap_values"].shape == (3, p)
        assert np.all(np.isfinite(result["shap_values"]))

    def test_rf_clf_many_features(self, wide_clf_data):
        X, y, p = wide_clf_data
        m = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        m.fit(X, y)
        result = kernel_shap_rf_clf(m, X[:10], X[:3], n_samples=50, random_state=42)
        assert result["shap_values"].shape == (3, p)
        assert np.all(np.isfinite(result["shap_values"]))

    def test_et_clf_many_features(self, wide_clf_data):
        X, y, p = wide_clf_data
        m = ExtraTreesClassifier(n_estimators=10, max_depth=3, random_state=42)
        m.fit(X, y)
        result = kernel_shap_et_clf(m, X[:10], X[:3], n_samples=50, random_state=42)
        assert result["shap_values"].shape == (3, p)
        assert np.all(np.isfinite(result["shap_values"]))

    def test_logistic_many_features(self, wide_clf_data):
        X, y, p = wide_clf_data
        m = LogisticRegression()
        m.fit(X, y)
        result = kernel_shap_logistic(m, X[:10], X[:3], n_samples=50, random_state=42)
        assert result["shap_values"].shape == (3, p)
        assert np.all(np.isfinite(result["shap_values"]))


# ---------------------------------------------------------------------------
# Cross-cutting: feature_values mirrors input x
# ---------------------------------------------------------------------------


class TestFeatureValuesMirrorInput:
    """feature_values in the result should equal the input instances x."""

    def test_rf_reg_feature_values_equal_x(self, rf_reg, reg_data):
        X, _ = reg_data
        x_explain = X[:N_EXPLAIN]
        result = kernel_shap_rf_reg(
            rf_reg, X[:N_BACKGROUND], x_explain, n_samples=50, random_state=42
        )
        np.testing.assert_array_equal(
            result["feature_values"], x_explain,
            err_msg="feature_values should equal the input x array",
        )

    def test_dt_reg_feature_values_equal_x(self, dt_reg, reg_data):
        X, _ = reg_data
        x_explain = X[:N_EXPLAIN]
        result = kernel_shap_dt_reg(
            dt_reg, X[:N_BACKGROUND], x_explain, n_samples=50, random_state=42
        )
        np.testing.assert_array_equal(result["feature_values"], x_explain)

    def test_linear_feature_values_equal_x(self, lin_reg, reg_data):
        X, _ = reg_data
        x_explain = X[:N_EXPLAIN]
        result = kernel_shap_linear(
            lin_reg, X[:N_BACKGROUND], x_explain, n_samples=50, random_state=42
        )
        np.testing.assert_array_equal(result["feature_values"], x_explain)

    def test_logistic_feature_values_equal_x(self, log_reg, clf_data):
        X, _ = clf_data
        x_explain = X[:N_EXPLAIN]
        result = kernel_shap_logistic(
            log_reg, X[:N_BACKGROUND], x_explain, n_samples=50, random_state=42
        )
        np.testing.assert_array_equal(result["feature_values"], x_explain)
