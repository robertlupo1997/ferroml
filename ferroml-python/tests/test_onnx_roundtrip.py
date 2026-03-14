"""ONNX round-trip validation tests using onnxruntime.

For each ONNX-exportable model:
1. Fit on known data
2. Get native FerroML predictions
3. Export to ONNX bytes
4. Load in onnxruntime
5. Run inference
6. Assert predictions match within tolerance

Tolerances (f64→f32 precision loss):
- Regressors: atol=1e-5
- Classification labels: exact match
- Classification probabilities: atol=1e-4
- Tree models: atol=1e-5 (float arithmetic in aggregation)
- Preprocessing: atol=1e-6
"""

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def regression_data():
    """Regression dataset with known coefficients."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.randn(50) * 0.1
    return X, y


@pytest.fixture
def binary_data():
    """Binary classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def multiclass_data():
    """Multiclass classification dataset (3 classes)."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y = np.array([0.0] * 20 + [1.0] * 20 + [2.0] * 20)
    idx = rng.permutation(60)
    return X[idx], y[idx]


@pytest.fixture
def positive_data():
    """Non-negative data for MultinomialNB."""
    rng = np.random.RandomState(42)
    X = rng.randint(0, 10, size=(50, 5)).astype(np.float64)
    y = (X[:, 0] > 5).astype(np.float64)
    return X, y


@pytest.fixture
def binary_feature_data():
    """Binary feature data for BernoulliNB."""
    rng = np.random.RandomState(42)
    X = rng.randint(0, 2, size=(50, 5)).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def scaler_data():
    """Data for preprocessing scalers."""
    rng = np.random.RandomState(42)
    return rng.randn(30, 5) * 10 + 3


# ── Helpers ─────────────────────────────────────────────────────────────────


def ort_predict(onnx_bytes, X):
    """Run onnxruntime inference, return output array(s)."""
    sess = ort.InferenceSession(onnx_bytes)
    return sess.run(None, {"input": X.astype(np.float32)})


def assert_regressor_roundtrip(model, X, y, atol=1e-5):
    """Validate regressor: ORT output ≈ predict() within f32 tolerance."""
    model.fit(X, y)
    ferro_pred = model.predict(X)
    ort_result = ort_predict(model.to_onnx_bytes(), X)
    ort_pred = ort_result[0].squeeze()
    np.testing.assert_allclose(
        ort_pred, ferro_pred, atol=atol, rtol=1e-4,
        err_msg=f"{type(model).__name__} regressor round-trip mismatch",
    )


def assert_classifier_label_roundtrip(model, X, y):
    """Validate classifier where ONNX outputs class labels directly."""
    model.fit(X, y)
    ferro_pred = model.predict(X)
    ort_result = ort_predict(model.to_onnx_bytes(), X)
    ort_pred = ort_result[0].squeeze().astype(np.float64)
    np.testing.assert_array_equal(
        ort_pred, ferro_pred,
        err_msg=f"{type(model).__name__} classifier label mismatch",
    )


def assert_classifier_proba_roundtrip(model, X, y, atol=1e-4):
    """Validate classifier where ONNX outputs class-1 probability.

    Checks both probability match and thresholded label match.
    """
    model.fit(X, y)
    ferro_pred = model.predict(X)
    ferro_proba = model.predict_proba(X)[:, 1]  # class-1 probability
    ort_result = ort_predict(model.to_onnx_bytes(), X)
    ort_proba = ort_result[0].squeeze()

    # Probability values should match
    np.testing.assert_allclose(
        ort_proba, ferro_proba, atol=atol, rtol=1e-3,
        err_msg=f"{type(model).__name__} probability round-trip mismatch",
    )

    # Thresholded labels should match
    ort_labels = (ort_proba > 0.5).astype(np.float64)
    np.testing.assert_array_equal(
        ort_labels, ferro_pred,
        err_msg=f"{type(model).__name__} thresholded label mismatch",
    )


def assert_classifier_sigmoid_roundtrip(model, X, y):
    """Validate classifier where ONNX outputs sigmoid/raw scores.

    No predict_proba available — just check thresholded labels match.
    """
    model.fit(X, y)
    ferro_pred = model.predict(X)
    ort_result = ort_predict(model.to_onnx_bytes(), X)
    ort_scores = ort_result[0].squeeze()
    ort_labels = (ort_scores > 0.5).astype(np.float64)
    np.testing.assert_array_equal(
        ort_labels, ferro_pred,
        err_msg=f"{type(model).__name__} sigmoid-thresholded label mismatch",
    )


def assert_transformer_roundtrip(transformer, X, atol=1e-6):
    """Validate preprocessing transformer round-trip."""
    transformer.fit(X)
    ferro_out = transformer.transform(X)
    ort_result = ort_predict(transformer.to_onnx_bytes(), X)
    ort_out = ort_result[0].squeeze()
    np.testing.assert_allclose(
        ort_out, ferro_out, atol=atol, rtol=1e-5,
        err_msg=f"{type(transformer).__name__} transformer round-trip mismatch",
    )


# ── Linear Regressors ──────────────────────────────────────────────────────


class TestLinearRegressorRoundtrip:
    """Round-trip validation for linear regression models."""

    def test_linear_regression(self, regression_data):
        from ferroml.linear import LinearRegression
        assert_regressor_roundtrip(LinearRegression(), *regression_data)

    def test_ridge_regression(self, regression_data):
        from ferroml.linear import RidgeRegression
        assert_regressor_roundtrip(RidgeRegression(), *regression_data)

    def test_lasso_regression(self, regression_data):
        from ferroml.linear import LassoRegression
        assert_regressor_roundtrip(LassoRegression(), *regression_data)

    def test_elastic_net(self, regression_data):
        from ferroml.linear import ElasticNet
        assert_regressor_roundtrip(ElasticNet(), *regression_data)

    def test_robust_regression(self, regression_data):
        from ferroml.linear import RobustRegression
        assert_regressor_roundtrip(RobustRegression(), *regression_data)

    def test_quantile_regression(self, regression_data):
        from ferroml.linear import QuantileRegression
        assert_regressor_roundtrip(QuantileRegression(), *regression_data)


# ── Linear Classifiers ─────────────────────────────────────────────────────


class TestLinearClassifierRoundtrip:
    """Round-trip validation for linear classifiers."""

    def test_logistic_regression(self, binary_data):
        from ferroml.linear import LogisticRegression
        # LogisticRegression ONNX exports sigmoid probabilities, not labels
        assert_classifier_sigmoid_roundtrip(LogisticRegression(), *binary_data)

    def test_ridge_classifier(self, binary_data):
        from ferroml.linear import RidgeClassifier
        assert_classifier_label_roundtrip(RidgeClassifier(), *binary_data)


# ── Tree Regressors ────────────────────────────────────────────────────────


class TestTreeRegressorRoundtrip:
    """Round-trip validation for tree-based regressors."""

    def test_decision_tree_regressor(self, regression_data):
        from ferroml.trees import DecisionTreeRegressor
        assert_regressor_roundtrip(DecisionTreeRegressor(), *regression_data)

    def test_random_forest_regressor(self, regression_data):
        from ferroml.trees import RandomForestRegressor
        assert_regressor_roundtrip(
            RandomForestRegressor(n_estimators=10), *regression_data,
        )

    def test_gradient_boosting_regressor(self, regression_data):
        from ferroml.trees import GradientBoostingRegressor
        assert_regressor_roundtrip(
            GradientBoostingRegressor(n_estimators=10), *regression_data,
        )

    def test_hist_gradient_boosting_regressor(self, regression_data):
        from ferroml.trees import HistGradientBoostingRegressor
        assert_regressor_roundtrip(
            HistGradientBoostingRegressor(max_iter=10, max_depth=3),
            *regression_data,
        )


# ── Tree Classifiers ───────────────────────────────────────────────────────


class TestTreeClassifierRoundtrip:
    """Round-trip validation for tree-based classifiers.

    DecisionTreeClassifier passes round-trip. RandomForestClassifier has a
    prediction accuracy mismatch (separate from the output type fix).
    GradientBoosting classifiers use TreeEnsembleRegressor internally.
    """

    def test_decision_tree_classifier(self, binary_data):
        from ferroml.trees import DecisionTreeClassifier
        assert_classifier_label_roundtrip(DecisionTreeClassifier(), *binary_data)

    @pytest.mark.xfail(
        reason="RandomForest ONNX export has prediction mismatch vs native predict",
        strict=True,
    )
    def test_random_forest_classifier(self, binary_data):
        from ferroml.trees import RandomForestClassifier
        assert_classifier_label_roundtrip(
            RandomForestClassifier(n_estimators=10), *binary_data,
        )

    def test_gradient_boosting_classifier(self, binary_data):
        from ferroml.trees import GradientBoostingClassifier
        assert_classifier_proba_roundtrip(
            GradientBoostingClassifier(n_estimators=10), *binary_data,
        )

    def test_hist_gradient_boosting_classifier(self, binary_data):
        from ferroml.trees import HistGradientBoostingClassifier
        assert_classifier_proba_roundtrip(
            HistGradientBoostingClassifier(max_iter=10, max_depth=3),
            *binary_data,
        )


# ── Ensemble Regressors ────────────────────────────────────────────────────


class TestEnsembleRegressorRoundtrip:
    """Round-trip validation for ensemble regressors."""

    def test_extra_trees_regressor(self, regression_data):
        from ferroml.ensemble import ExtraTreesRegressor
        assert_regressor_roundtrip(
            ExtraTreesRegressor(n_estimators=10), *regression_data,
        )

    def test_adaboost_regressor(self, regression_data):
        """AdaBoost.R2 uses weighted median, which is not expressible in ONNX.
        The ONNX export uses a weighted-sum approximation (r²≈0.995)."""
        from ferroml.ensemble import AdaBoostRegressor
        assert_regressor_roundtrip(
            AdaBoostRegressor(n_estimators=10), *regression_data, atol=2.0,
        )

    def test_sgd_regressor(self, regression_data):
        from ferroml.ensemble import SGDRegressor
        assert_regressor_roundtrip(SGDRegressor(), *regression_data)


# ── Ensemble Classifiers ───────────────────────────────────────────────────


class TestEnsembleClassifierRoundtrip:
    """Round-trip validation for ensemble classifiers."""

    def test_extra_trees_classifier(self, binary_data):
        from ferroml.ensemble import ExtraTreesClassifier
        assert_classifier_label_roundtrip(
            ExtraTreesClassifier(n_estimators=10), *binary_data,
        )

    @pytest.mark.xfail(
        reason="AdaBoost ONNX export has prediction mismatch vs native predict",
        strict=True,
    )
    def test_adaboost_classifier(self, binary_data):
        from ferroml.ensemble import AdaBoostClassifier
        assert_classifier_label_roundtrip(
            AdaBoostClassifier(n_estimators=10), *binary_data,
        )

    def test_sgd_classifier(self, binary_data):
        from ferroml.ensemble import SGDClassifier
        assert_classifier_sigmoid_roundtrip(SGDClassifier(), *binary_data)

    def test_passive_aggressive_classifier(self, binary_data):
        from ferroml.ensemble import PassiveAggressiveClassifier
        assert_classifier_label_roundtrip(
            PassiveAggressiveClassifier(), *binary_data,
        )


# ── SVM Models ─────────────────────────────────────────────────────────────


class TestSvmRoundtrip:
    """Round-trip validation for SVM models."""

    def test_linear_svr(self, regression_data):
        from ferroml.svm import LinearSVR
        assert_regressor_roundtrip(LinearSVR(), *regression_data)

    def test_linear_svc(self, binary_data):
        from ferroml.svm import LinearSVC
        assert_classifier_label_roundtrip(LinearSVC(), *binary_data)

    def test_svr(self, regression_data):
        from ferroml.svm import SVR
        # SVR with kernel ops has slightly larger f32 tolerance
        assert_regressor_roundtrip(SVR(), *regression_data, atol=1e-3)

    @pytest.mark.xfail(
        reason="SVMClassifier OvO coefficient encoding produces incorrect predictions",
        strict=True,
    )
    def test_svc(self, binary_data):
        from ferroml.svm import SVC
        assert_classifier_label_roundtrip(SVC(), *binary_data)


# ── Naive Bayes Models ─────────────────────────────────────────────────────


class TestNaiveBayesRoundtrip:
    """Round-trip validation for Naive Bayes models."""

    def test_gaussian_nb(self, binary_data):
        from ferroml.naive_bayes import GaussianNB
        assert_classifier_label_roundtrip(GaussianNB(), *binary_data)

    def test_multinomial_nb(self, positive_data):
        from ferroml.naive_bayes import MultinomialNB
        assert_classifier_label_roundtrip(MultinomialNB(), *positive_data)

    def test_bernoulli_nb(self, binary_feature_data):
        from ferroml.naive_bayes import BernoulliNB
        assert_classifier_label_roundtrip(BernoulliNB(), *binary_feature_data)


# ── Preprocessing Scalers ──────────────────────────────────────────────────


class TestPreprocessingRoundtrip:
    """Round-trip validation for preprocessing transformers."""

    def test_standard_scaler(self, scaler_data):
        from ferroml.preprocessing import StandardScaler
        assert_transformer_roundtrip(StandardScaler(), scaler_data)

    def test_min_max_scaler(self, scaler_data):
        from ferroml.preprocessing import MinMaxScaler
        assert_transformer_roundtrip(MinMaxScaler(), scaler_data)

    def test_robust_scaler(self, scaler_data):
        from ferroml.preprocessing import RobustScaler
        assert_transformer_roundtrip(RobustScaler(), scaler_data)

    def test_max_abs_scaler(self, scaler_data):
        from ferroml.preprocessing import MaxAbsScaler
        assert_transformer_roundtrip(MaxAbsScaler(), scaler_data)


# ── Edge Cases ─────────────────────────────────────────────────────────────


class TestRoundtripEdgeCases:
    """Edge cases for ONNX round-trip: single sample, large batch."""

    def test_single_sample_regressor(self):
        """Single-sample inference should work."""
        from ferroml.linear import LinearRegression

        rng = np.random.RandomState(42)
        X_train = rng.randn(20, 3)
        y_train = X_train @ [1, -1, 2] + rng.randn(20) * 0.1

        model = LinearRegression()
        model.fit(X_train, y_train)

        X_test = rng.randn(1, 3)
        ferro_pred = model.predict(X_test)
        ort_result = ort_predict(model.to_onnx_bytes(), X_test)
        ort_pred = ort_result[0].squeeze()

        np.testing.assert_allclose(ort_pred, ferro_pred, atol=1e-5)

    def test_single_sample_classifier(self):
        """Single-sample classification should work."""
        from ferroml.linear import LogisticRegression

        rng = np.random.RandomState(42)
        X_train = rng.randn(30, 3)
        y_train = (X_train[:, 0] > 0).astype(np.float64)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        X_test = rng.randn(1, 3)
        ferro_pred = model.predict(X_test)
        ort_result = ort_predict(model.to_onnx_bytes(), X_test)
        # LogReg ONNX exports sigmoid probabilities, threshold at 0.5
        ort_label = (ort_result[0].squeeze() > 0.5).astype(np.float64)

        np.testing.assert_array_equal(ort_label, ferro_pred)

    def test_large_batch_regressor(self):
        """Large batch inference should work."""
        from ferroml.linear import RidgeRegression

        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 5)
        y_train = X_train @ [1, -2, 0.5, 3, -1] + rng.randn(50) * 0.1

        model = RidgeRegression()
        model.fit(X_train, y_train)

        X_test = rng.randn(500, 5)
        ferro_pred = model.predict(X_test)
        ort_result = ort_predict(model.to_onnx_bytes(), X_test)
        ort_pred = ort_result[0].squeeze()

        np.testing.assert_allclose(ort_pred, ferro_pred, atol=1e-5, rtol=1e-4)

    def test_single_sample_transformer(self):
        """Single-sample transform should work."""
        from ferroml.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        X_train = rng.randn(20, 4) * 5 + 3

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_test = rng.randn(1, 4) * 5 + 3
        ferro_out = scaler.transform(X_test)
        ort_result = ort_predict(scaler.to_onnx_bytes(), X_test)
        ort_out = ort_result[0].squeeze()

        np.testing.assert_allclose(ort_out, ferro_out.ravel(), atol=1e-6)

    def test_near_zero_coefficients(self):
        """Model with near-zero coefficients should round-trip correctly."""
        from ferroml.linear import ElasticNet

        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        # Only first 2 features matter; rest should have ~zero coefficients
        y = X[:, 0] * 2 + X[:, 1] * (-1) + rng.randn(50) * 0.01

        model = ElasticNet(alpha=0.5)
        model.fit(X, y)
        ferro_pred = model.predict(X)

        ort_result = ort_predict(model.to_onnx_bytes(), X)
        ort_pred = ort_result[0].squeeze()

        np.testing.assert_allclose(ort_pred, ferro_pred, atol=1e-5, rtol=1e-4)

    def test_tree_single_sample(self):
        """Single-sample tree regressor inference."""
        from ferroml.trees import DecisionTreeRegressor

        rng = np.random.RandomState(42)
        X_train = rng.randn(30, 3)
        y_train = X_train @ [1, -1, 2]

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)

        X_test = rng.randn(1, 3)
        ferro_pred = model.predict(X_test)
        ort_result = ort_predict(model.to_onnx_bytes(), X_test)
        ort_pred = ort_result[0].squeeze()

        np.testing.assert_allclose(ort_pred, ferro_pred, atol=1e-5)

    def test_gradient_boosting_proba_single_sample(self):
        """Single-sample GBC probability output."""
        from ferroml.trees import GradientBoostingClassifier

        rng = np.random.RandomState(42)
        X_train = rng.randn(40, 3)
        y_train = (X_train[:, 0] > 0).astype(np.float64)

        model = GradientBoostingClassifier(n_estimators=10)
        model.fit(X_train, y_train)

        X_test = rng.randn(1, 3)
        ferro_proba = model.predict_proba(X_test)[:, 1]
        ort_result = ort_predict(model.to_onnx_bytes(), X_test)
        ort_proba = ort_result[0].squeeze()

        np.testing.assert_allclose(ort_proba, ferro_proba, atol=1e-4)


# ── Multiclass Round-trip ──────────────────────────────────────────────────


class TestMulticlassRoundtrip:
    """Round-trip validation for multiclass models."""

    def test_ridge_classifier_multiclass(self, multiclass_data):
        from ferroml.linear import RidgeClassifier
        assert_classifier_label_roundtrip(RidgeClassifier(), *multiclass_data)

    def test_gaussian_nb_multiclass(self, multiclass_data):
        from ferroml.naive_bayes import GaussianNB
        assert_classifier_label_roundtrip(GaussianNB(), *multiclass_data)

    def test_gradient_boosting_regressor_multifeature(self):
        """GBR with more features for broader coverage."""
        from ferroml.trees import GradientBoostingRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(60, 8)
        y = X @ rng.randn(8) + rng.randn(60) * 0.1

        assert_regressor_roundtrip(
            GradientBoostingRegressor(n_estimators=10), X, y,
        )
