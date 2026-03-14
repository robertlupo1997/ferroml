"""Tests for ONNX export Python bindings.

Tests export_onnx() and to_onnx_bytes() on all 34 ONNX-exportable models.
"""

import os
import tempfile

import numpy as np
import pytest

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def regression_data():
    """Small regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(30, 4)
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.randn(30) * 0.1
    return X, y


@pytest.fixture
def binary_data():
    """Small binary classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(40, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def multiclass_data():
    """Small multiclass classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y = np.array([0.0] * 20 + [1.0] * 20 + [2.0] * 20)
    # Shuffle
    idx = rng.permutation(60)
    return X[idx], y[idx]


@pytest.fixture
def scaler_data():
    """Data for testing preprocessing scalers."""
    rng = np.random.RandomState(42)
    return rng.randn(20, 5) * 10 + 3


# ── Helpers ─────────────────────────────────────────────────────────────────

def assert_onnx_export_works(model, X, y=None, is_transformer=False):
    """Assert that export_onnx and to_onnx_bytes work for a fitted model."""
    # Fit
    if is_transformer:
        model.fit(X)
    else:
        model.fit(X, y)

    # to_onnx_bytes returns bytes
    onnx_bytes = model.to_onnx_bytes()
    assert isinstance(onnx_bytes, bytes)
    assert len(onnx_bytes) > 0

    # Bytes should start with ONNX magic number (protobuf field 1, varint)
    # The first byte of a valid ONNX protobuf is 0x08 (field 1, wire type 0)
    assert onnx_bytes[0:1] == b'\x08', f"Expected ONNX protobuf header, got {onnx_bytes[0]:#x}"

    # to_onnx_bytes with custom names
    onnx_bytes2 = model.to_onnx_bytes(
        model_name="test_model",
        input_name="features",
        output_name="prediction",
    )
    assert isinstance(onnx_bytes2, bytes)
    assert len(onnx_bytes2) > 0

    # export_onnx writes to file
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        model.export_onnx(path)
        assert os.path.exists(path)
        file_size = os.path.getsize(path)
        assert file_size > 0
        # File contents should match to_onnx_bytes
        with open(path, "rb") as f:
            file_bytes = f.read()
        assert file_bytes == onnx_bytes
    finally:
        os.unlink(path)

    # export_onnx with custom model name
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    try:
        model.export_onnx(path, model_name="custom_name")
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


# ── Linear Models ───────────────────────────────────────────────────────────

class TestLinearOnnxExport:
    """ONNX export for linear model family."""

    def test_linear_regression(self, regression_data):
        from ferroml.linear import LinearRegression
        assert_onnx_export_works(LinearRegression(), *regression_data)

    def test_ridge_regression(self, regression_data):
        from ferroml.linear import RidgeRegression
        assert_onnx_export_works(RidgeRegression(), *regression_data)

    def test_lasso_regression(self, regression_data):
        from ferroml.linear import LassoRegression
        assert_onnx_export_works(LassoRegression(), *regression_data)

    def test_elastic_net(self, regression_data):
        from ferroml.linear import ElasticNet
        assert_onnx_export_works(ElasticNet(), *regression_data)

    def test_robust_regression(self, regression_data):
        from ferroml.linear import RobustRegression
        assert_onnx_export_works(RobustRegression(), *regression_data)

    def test_quantile_regression(self, regression_data):
        from ferroml.linear import QuantileRegression
        assert_onnx_export_works(QuantileRegression(), *regression_data)

    def test_logistic_regression(self, binary_data):
        from ferroml.linear import LogisticRegression
        assert_onnx_export_works(LogisticRegression(), *binary_data)

    def test_ridge_classifier(self, binary_data):
        from ferroml.linear import RidgeClassifier
        assert_onnx_export_works(RidgeClassifier(), *binary_data)


# ── Tree Models ─────────────────────────────────────────────────────────────

class TestTreeOnnxExport:
    """ONNX export for tree model family."""

    def test_decision_tree_regressor(self, regression_data):
        from ferroml.trees import DecisionTreeRegressor
        assert_onnx_export_works(DecisionTreeRegressor(), *regression_data)

    def test_decision_tree_classifier(self, binary_data):
        from ferroml.trees import DecisionTreeClassifier
        assert_onnx_export_works(DecisionTreeClassifier(), *binary_data)

    def test_random_forest_regressor(self, regression_data):
        from ferroml.trees import RandomForestRegressor
        assert_onnx_export_works(RandomForestRegressor(n_estimators=5), *regression_data)

    def test_random_forest_classifier(self, binary_data):
        from ferroml.trees import RandomForestClassifier
        assert_onnx_export_works(RandomForestClassifier(n_estimators=5), *binary_data)

    def test_gradient_boosting_regressor(self, regression_data):
        from ferroml.trees import GradientBoostingRegressor
        assert_onnx_export_works(GradientBoostingRegressor(n_estimators=5), *regression_data)

    def test_gradient_boosting_classifier(self, binary_data):
        from ferroml.trees import GradientBoostingClassifier
        assert_onnx_export_works(GradientBoostingClassifier(n_estimators=5), *binary_data)

    def test_hist_gradient_boosting_regressor(self, regression_data):
        from ferroml.trees import HistGradientBoostingRegressor
        assert_onnx_export_works(HistGradientBoostingRegressor(max_iter=5, max_depth=3), *regression_data)

    def test_hist_gradient_boosting_classifier(self, binary_data):
        from ferroml.trees import HistGradientBoostingClassifier
        assert_onnx_export_works(HistGradientBoostingClassifier(max_iter=5, max_depth=3), *binary_data)


# ── Ensemble Models ─────────────────────────────────────────────────────────

class TestEnsembleOnnxExport:
    """ONNX export for ensemble model family."""

    def test_extra_trees_regressor(self, regression_data):
        from ferroml.ensemble import ExtraTreesRegressor
        assert_onnx_export_works(ExtraTreesRegressor(n_estimators=5), *regression_data)

    def test_extra_trees_classifier(self, binary_data):
        from ferroml.ensemble import ExtraTreesClassifier
        assert_onnx_export_works(ExtraTreesClassifier(n_estimators=5), *binary_data)

    def test_adaboost_regressor(self, regression_data):
        from ferroml.ensemble import AdaBoostRegressor
        assert_onnx_export_works(AdaBoostRegressor(n_estimators=5), *regression_data)

    def test_adaboost_classifier(self, binary_data):
        from ferroml.ensemble import AdaBoostClassifier
        assert_onnx_export_works(AdaBoostClassifier(n_estimators=5), *binary_data)

    def test_sgd_regressor(self, regression_data):
        from ferroml.ensemble import SGDRegressor
        assert_onnx_export_works(SGDRegressor(), *regression_data)

    def test_sgd_classifier(self, binary_data):
        from ferroml.ensemble import SGDClassifier
        assert_onnx_export_works(SGDClassifier(), *binary_data)

    def test_passive_aggressive_classifier(self, binary_data):
        from ferroml.ensemble import PassiveAggressiveClassifier
        assert_onnx_export_works(PassiveAggressiveClassifier(), *binary_data)


# ── SVM Models ──────────────────────────────────────────────────────────────

class TestSvmOnnxExport:
    """ONNX export for SVM model family."""

    def test_linear_svc(self, binary_data):
        from ferroml.svm import LinearSVC
        assert_onnx_export_works(LinearSVC(), *binary_data)

    def test_linear_svr(self, regression_data):
        from ferroml.svm import LinearSVR
        assert_onnx_export_works(LinearSVR(), *regression_data)

    def test_svc(self, binary_data):
        from ferroml.svm import SVC
        assert_onnx_export_works(SVC(), *binary_data)

    def test_svr(self, regression_data):
        from ferroml.svm import SVR
        assert_onnx_export_works(SVR(), *regression_data)


# ── Naive Bayes Models ──────────────────────────────────────────────────────

class TestNaiveBayesOnnxExport:
    """ONNX export for Naive Bayes model family."""

    def test_gaussian_nb(self, binary_data):
        from ferroml.naive_bayes import GaussianNB
        assert_onnx_export_works(GaussianNB(), *binary_data)

    def test_multinomial_nb(self):
        from ferroml.naive_bayes import MultinomialNB
        rng = np.random.RandomState(42)
        X = rng.randint(0, 10, size=(40, 5)).astype(np.float64)
        y = (X[:, 0] > 5).astype(np.float64)
        assert_onnx_export_works(MultinomialNB(), X, y)

    def test_bernoulli_nb(self):
        from ferroml.naive_bayes import BernoulliNB
        rng = np.random.RandomState(42)
        X = rng.randint(0, 2, size=(40, 5)).astype(np.float64)
        y = (X[:, 0] > 0).astype(np.float64)
        assert_onnx_export_works(BernoulliNB(), X, y)


# ── Preprocessing Scalers ───────────────────────────────────────────────────

class TestPreprocessingOnnxExport:
    """ONNX export for preprocessing scalers."""

    def test_standard_scaler(self, scaler_data):
        from ferroml.preprocessing import StandardScaler
        assert_onnx_export_works(StandardScaler(), scaler_data, is_transformer=True)

    def test_min_max_scaler(self, scaler_data):
        from ferroml.preprocessing import MinMaxScaler
        assert_onnx_export_works(MinMaxScaler(), scaler_data, is_transformer=True)

    def test_robust_scaler(self, scaler_data):
        from ferroml.preprocessing import RobustScaler
        assert_onnx_export_works(RobustScaler(), scaler_data, is_transformer=True)

    def test_max_abs_scaler(self, scaler_data):
        from ferroml.preprocessing import MaxAbsScaler
        assert_onnx_export_works(MaxAbsScaler(), scaler_data, is_transformer=True)


# ── Error Handling ──────────────────────────────────────────────────────────

class TestOnnxExportErrors:
    """Test error handling for ONNX export."""

    def test_export_unfitted_model_raises(self):
        """Exporting an unfitted model should raise RuntimeError."""
        from ferroml.linear import LinearRegression
        model = LinearRegression()
        with pytest.raises(RuntimeError):
            model.to_onnx_bytes()

    def test_export_unfitted_to_file_raises(self):
        """Exporting an unfitted model to file should raise RuntimeError."""
        from ferroml.linear import LinearRegression
        model = LinearRegression()
        with pytest.raises(RuntimeError):
            model.export_onnx("/tmp/should_not_exist.onnx")

    def test_export_to_invalid_path_raises(self, regression_data):
        """Exporting to an invalid path should raise RuntimeError."""
        from ferroml.linear import LinearRegression
        model = LinearRegression()
        model.fit(*regression_data)
        with pytest.raises(RuntimeError):
            model.export_onnx("/nonexistent/directory/model.onnx")


# ── Multiclass ONNX Export ──────────────────────────────────────────────────

class TestMulticlassOnnxExport:
    """ONNX export for multiclass models."""

    def test_ridge_classifier_multiclass(self, multiclass_data):
        from ferroml.linear import RidgeClassifier
        assert_onnx_export_works(RidgeClassifier(), *multiclass_data)

    def test_decision_tree_multiclass(self, multiclass_data):
        from ferroml.trees import DecisionTreeClassifier
        assert_onnx_export_works(DecisionTreeClassifier(), *multiclass_data)

    def test_random_forest_multiclass(self, multiclass_data):
        from ferroml.trees import RandomForestClassifier
        assert_onnx_export_works(RandomForestClassifier(n_estimators=5), *multiclass_data)

    def test_svc_multiclass(self, multiclass_data):
        from ferroml.svm import SVC
        assert_onnx_export_works(SVC(), *multiclass_data)

    def test_gaussian_nb_multiclass(self, multiclass_data):
        from ferroml.naive_bayes import GaussianNB
        assert_onnx_export_works(GaussianNB(), *multiclass_data)
