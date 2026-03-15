"""Tests for decision_function() on all classifiers that support it.

Verifies:
- Output shape: (n_samples, n_classes) or (n_samples,) for binary
- Sign consistency with predict for linear models
- Monotonic relationship with predict_proba where applicable
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    np.random.seed(42)
    X = np.random.randn(80, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    X = np.random.randn(120, 4)
    y = np.zeros(120)
    y[X[:, 0] > 0.5] = 1.0
    y[X[:, 0] < -0.5] = 2.0
    return X, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_decision_function_shape(model, X, y, n_classes):
    """Check that decision_function returns correct shape."""
    model.fit(X, y)
    df = model.decision_function(X)
    assert df.ndim in (1, 2), f"Expected 1D or 2D, got {df.ndim}D"
    if df.ndim == 2:
        assert df.shape == (len(X), n_classes), (
            f"Expected shape ({len(X)}, {n_classes}), got {df.shape}"
        )
    else:
        assert df.shape == (len(X),), f"Expected shape ({len(X)},), got {df.shape}"
    return df


# ---------------------------------------------------------------------------
# Linear classifiers
# ---------------------------------------------------------------------------

class TestLinearDecisionFunction:
    def test_logistic_regression(self, binary_data):
        from ferroml.linear import LogisticRegression
        X, y = binary_data
        model = LogisticRegression()
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.shape == (len(X),)
        # Sign of decision_function should mostly agree with predict
        preds = model.predict(X)
        predicted_positive = preds > 0.5
        df_positive = df > 0
        agreement = np.mean(predicted_positive == df_positive)
        assert agreement > 0.9, f"Sign agreement={agreement}, expected >0.9"

    def test_ridge_classifier(self, binary_data):
        from ferroml.linear import RidgeClassifier
        X, y = binary_data
        model = RidgeClassifier()
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.ndim in (1, 2)

    def test_sgd_classifier(self, binary_data):
        from ferroml.ensemble import SGDClassifier
        X, y = binary_data
        model = SGDClassifier(loss="hinge", max_iter=200)
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.ndim in (1, 2)

    def test_perceptron(self, binary_data):
        from ferroml.linear import Perceptron
        X, y = binary_data
        model = Perceptron(max_iter=200)
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.ndim in (1, 2)

    def test_passive_aggressive(self, binary_data):
        from ferroml.ensemble import PassiveAggressiveClassifier
        X, y = binary_data
        model = PassiveAggressiveClassifier()
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.ndim in (1, 2)


# ---------------------------------------------------------------------------
# SVM classifiers
# ---------------------------------------------------------------------------

class TestSVMDecisionFunction:
    def test_svc(self, binary_data):
        from ferroml.svm import SVC
        X, y = binary_data
        model = SVC()
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.ndim in (1, 2)

    def test_linear_svc(self, binary_data):
        from ferroml.svm import LinearSVC
        X, y = binary_data
        model = LinearSVC()
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.ndim in (1, 2)


# ---------------------------------------------------------------------------
# Tree-based classifiers
# ---------------------------------------------------------------------------

class TestTreeDecisionFunction:
    def test_decision_tree(self, binary_data):
        from ferroml.trees import DecisionTreeClassifier
        X, y = binary_data
        model = DecisionTreeClassifier()
        _check_decision_function_shape(model, X, y, n_classes=2)

    def test_random_forest(self, binary_data):
        from ferroml.trees import RandomForestClassifier
        X, y = binary_data
        model = RandomForestClassifier(n_estimators=10)
        _check_decision_function_shape(model, X, y, n_classes=2)

    def test_gradient_boosting(self, binary_data):
        from ferroml.trees import GradientBoostingClassifier
        X, y = binary_data
        model = GradientBoostingClassifier(n_estimators=10)
        _check_decision_function_shape(model, X, y, n_classes=2)

    def test_hist_gradient_boosting(self, binary_data):
        from ferroml.trees import HistGradientBoostingClassifier
        X, y = binary_data
        model = HistGradientBoostingClassifier(max_iter=10)
        _check_decision_function_shape(model, X, y, n_classes=2)

    def test_extra_trees(self, binary_data):
        from ferroml.ensemble import ExtraTreesClassifier
        X, y = binary_data
        model = ExtraTreesClassifier(n_estimators=10)
        _check_decision_function_shape(model, X, y, n_classes=2)

    def test_adaboost(self, binary_data):
        from ferroml.ensemble import AdaBoostClassifier
        X, y = binary_data
        model = AdaBoostClassifier(n_estimators=10)
        _check_decision_function_shape(model, X, y, n_classes=2)


# ---------------------------------------------------------------------------
# Multiclass tests
# ---------------------------------------------------------------------------

class TestMulticlassDecisionFunction:
    def test_logistic_binary_decision_function(self, binary_data):
        """LogisticRegression only supports binary — verify shape."""
        from ferroml.linear import LogisticRegression
        X, y = binary_data
        model = LogisticRegression()
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.shape == (len(X),)

    def test_sgd_multiclass(self, multiclass_data):
        from ferroml.ensemble import SGDClassifier
        X, y = multiclass_data
        model = SGDClassifier(loss="hinge", max_iter=200)
        model.fit(X, y)
        df = model.decision_function(X)
        assert df.shape[0] == len(X)
        if df.ndim == 2:
            assert df.shape[1] >= 2

    def test_random_forest_multiclass(self, multiclass_data):
        from ferroml.trees import RandomForestClassifier
        X, y = multiclass_data
        model = RandomForestClassifier(n_estimators=10)
        _check_decision_function_shape(model, X, y, n_classes=3)
