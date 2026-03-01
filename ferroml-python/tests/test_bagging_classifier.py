"""Test FerroML BaggingClassifier factory methods and API."""

import numpy as np
import pytest

from ferroml.ensemble import BaggingClassifier


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def classification_data():
    """Generate simple binary classification data (100 samples, 4 features)."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture(scope="module")
def multiclass_data():
    """Generate 3-class classification data (150 samples, 4 features)."""
    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 4)
    y = np.zeros(n_samples)
    y[X[:, 0] > 0.5] = 1.0
    y[X[:, 0] < -0.5] = 2.0
    return X, y.astype(np.float64)


@pytest.fixture(scope="module")
def single_feature_data():
    """Generate binary classification data with 1 feature."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1)
    y = (X[:, 0] > 0).astype(np.float64)
    return X, y


# ============================================================================
# TestBaggingWithDecisionTree
# ============================================================================


class TestBaggingWithDecisionTree:
    """Tests for BaggingClassifier.with_decision_tree factory method."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict shapes."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_proba(self, classification_data):
        """Test predict_proba shape and value constraints."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, classification_data):
        """Test OOB score is a float in [0, 1] when oob_score=True."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, oob_score=True, bootstrap=True, random_state=42
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)
        assert 0.0 <= oob <= 1.0

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ shape, non-negativity, and sum."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_repr(self, classification_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(n_estimators=10, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "DecisionTreeClassifier" in r

    def test_min_samples_split_leaf(self, classification_data):
        """Test that min_samples_split and min_samples_leaf are accepted."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape


# ============================================================================
# TestBaggingWithRandomForest
# ============================================================================


class TestBaggingWithRandomForest:
    """Tests for BaggingClassifier.with_random_forest factory method."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict shapes."""
        X, y = classification_data
        model = BaggingClassifier.with_random_forest(
            n_estimators=5, rf_n_estimators=10, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_proba(self, classification_data):
        """Test predict_proba shape and value constraints."""
        X, y = classification_data
        model = BaggingClassifier.with_random_forest(
            n_estimators=5, rf_n_estimators=10, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, classification_data):
        """Test OOB score is a float in [0, 1] when oob_score=True."""
        X, y = classification_data
        model = BaggingClassifier.with_random_forest(
            n_estimators=5,
            rf_n_estimators=10,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)
        assert 0.0 <= oob <= 1.0

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ shape, non-negativity, and sum."""
        X, y = classification_data
        model = BaggingClassifier.with_random_forest(
            n_estimators=5, rf_n_estimators=10, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_repr(self, classification_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = classification_data
        model = BaggingClassifier.with_random_forest(n_estimators=5, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "RandomForestClassifier" in r

    def test_rf_n_estimators_parameter(self, classification_data):
        """Test that rf_n_estimators controls inner RF trees."""
        X, y = classification_data
        model = BaggingClassifier.with_random_forest(
            n_estimators=5, rf_n_estimators=5, max_depth=3, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape


# ============================================================================
# TestBaggingWithLogisticRegression
# ============================================================================


class TestBaggingWithLogisticRegression:
    """Tests for BaggingClassifier.with_logistic_regression factory method."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict shapes."""
        X, y = classification_data
        model = BaggingClassifier.with_logistic_regression(
            n_estimators=10, max_iter=200, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_proba(self, classification_data):
        """Test predict_proba shape and value constraints."""
        X, y = classification_data
        model = BaggingClassifier.with_logistic_regression(
            n_estimators=10, max_iter=200, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, classification_data):
        """Test OOB score is a float in [0, 1] when oob_score=True."""
        X, y = classification_data
        model = BaggingClassifier.with_logistic_regression(
            n_estimators=10,
            max_iter=200,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)
        assert 0.0 <= oob <= 1.0

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ shape and non-negativity."""
        X, y = classification_data
        model = BaggingClassifier.with_logistic_regression(
            n_estimators=10, max_iter=200, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, classification_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = classification_data
        model = BaggingClassifier.with_logistic_regression(
            n_estimators=10, random_state=42
        )
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "LogisticRegression" in r

    def test_max_iter_parameter(self, classification_data):
        """Test that max_iter is accepted."""
        X, y = classification_data
        model = BaggingClassifier.with_logistic_regression(
            n_estimators=5, max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape


# ============================================================================
# TestBaggingWithGaussianNB
# ============================================================================


class TestBaggingWithGaussianNB:
    """Tests for BaggingClassifier.with_gaussian_nb factory method."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict shapes."""
        X, y = classification_data
        model = BaggingClassifier.with_gaussian_nb(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_proba(self, classification_data):
        """Test predict_proba shape and value constraints."""
        X, y = classification_data
        model = BaggingClassifier.with_gaussian_nb(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, classification_data):
        """Test OOB score is a float in [0, 1] when oob_score=True."""
        X, y = classification_data
        model = BaggingClassifier.with_gaussian_nb(
            n_estimators=10,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)
        assert 0.0 <= oob <= 1.0

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ shape and non-negativity."""
        X, y = classification_data
        model = BaggingClassifier.with_gaussian_nb(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, classification_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = classification_data
        model = BaggingClassifier.with_gaussian_nb(n_estimators=10, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "GaussianNB" in r

    def test_predict_classes_valid(self, classification_data):
        """Test that predicted classes are valid class labels."""
        X, y = classification_data
        model = BaggingClassifier.with_gaussian_nb(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0.0, 1.0})


# ============================================================================
# TestBaggingWithKNN
# ============================================================================


class TestBaggingWithKNN:
    """Tests for BaggingClassifier.with_knn factory method."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict shapes."""
        X, y = classification_data
        model = BaggingClassifier.with_knn(
            n_estimators=10, n_neighbors=5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_proba(self, classification_data):
        """Test predict_proba shape and value constraints."""
        X, y = classification_data
        model = BaggingClassifier.with_knn(
            n_estimators=10, n_neighbors=5, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, classification_data):
        """Test OOB score is a float in [0, 1] when oob_score=True."""
        X, y = classification_data
        model = BaggingClassifier.with_knn(
            n_estimators=10,
            n_neighbors=5,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)
        assert 0.0 <= oob <= 1.0

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ shape and non-negativity."""
        X, y = classification_data
        model = BaggingClassifier.with_knn(
            n_estimators=10, n_neighbors=5, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, classification_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = classification_data
        model = BaggingClassifier.with_knn(n_estimators=10, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "KNeighborsClassifier" in r

    def test_n_neighbors_parameter(self, classification_data):
        """Test that n_neighbors is accepted."""
        X, y = classification_data
        for k in [1, 3, 7]:
            model = BaggingClassifier.with_knn(
                n_estimators=5, n_neighbors=k, random_state=42
            )
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, f"Failed for n_neighbors={k}"


# ============================================================================
# TestBaggingWithSVC
# ============================================================================


class TestBaggingWithSVC:
    """Tests for BaggingClassifier.with_svc factory method."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict shapes."""
        X, y = classification_data
        model = BaggingClassifier.with_svc(
            n_estimators=5, c=1.0, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_proba(self, classification_data):
        """Test predict_proba shape and value constraints."""
        X, y = classification_data
        model = BaggingClassifier.with_svc(
            n_estimators=5, c=1.0, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, classification_data):
        """Test OOB score is a float in [0, 1] when oob_score=True."""
        X, y = classification_data
        model = BaggingClassifier.with_svc(
            n_estimators=5,
            c=1.0,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)
        assert 0.0 <= oob <= 1.0

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ shape and non-negativity."""
        X, y = classification_data
        model = BaggingClassifier.with_svc(
            n_estimators=5, c=1.0, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, classification_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = classification_data
        model = BaggingClassifier.with_svc(n_estimators=5, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "SVC" in r

    def test_c_parameter(self, classification_data):
        """Test that c (regularization) parameter is accepted."""
        X, y = classification_data
        for c_val in [0.1, 1.0, 10.0]:
            model = BaggingClassifier.with_svc(
                n_estimators=5, c=c_val, random_state=42
            )
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, f"Failed for c={c_val}"


# ============================================================================
# TestBaggingWithGradientBoosting
# ============================================================================


class TestBaggingWithGradientBoosting:
    """Tests for BaggingClassifier.with_gradient_boosting factory method."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict shapes."""
        X, y = classification_data
        model = BaggingClassifier.with_gradient_boosting(
            n_estimators=5,
            gb_n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_proba(self, classification_data):
        """Test predict_proba shape and value constraints."""
        X, y = classification_data
        model = BaggingClassifier.with_gradient_boosting(
            n_estimators=5,
            gb_n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, classification_data):
        """Test OOB score is a float in [0, 1] when oob_score=True."""
        X, y = classification_data
        model = BaggingClassifier.with_gradient_boosting(
            n_estimators=5,
            gb_n_estimators=10,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)
        assert 0.0 <= oob <= 1.0

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ shape, non-negativity, and sum."""
        X, y = classification_data
        model = BaggingClassifier.with_gradient_boosting(
            n_estimators=5,
            gb_n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_repr(self, classification_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = classification_data
        model = BaggingClassifier.with_gradient_boosting(
            n_estimators=5, random_state=42
        )
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "GradientBoostingClassifier" in r

    def test_gb_parameters(self, classification_data):
        """Test that gb_n_estimators, learning_rate, and max_depth are accepted."""
        X, y = classification_data
        model = BaggingClassifier.with_gradient_boosting(
            n_estimators=5,
            gb_n_estimators=30,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape


# ============================================================================
# TestBaggingWithHistGradientBoosting
# ============================================================================


class TestBaggingWithHistGradientBoosting:
    """Tests for BaggingClassifier.with_hist_gradient_boosting factory method."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict shapes."""
        X, y = classification_data
        model = BaggingClassifier.with_hist_gradient_boosting(
            n_estimators=5,
            hgb_max_iter=30,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_predict_proba(self, classification_data):
        """Test predict_proba shape and value constraints."""
        X, y = classification_data
        model = BaggingClassifier.with_hist_gradient_boosting(
            n_estimators=5,
            hgb_max_iter=30,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_oob_score(self, classification_data):
        """Test OOB score is a float in [0, 1] when oob_score=True."""
        X, y = classification_data
        model = BaggingClassifier.with_hist_gradient_boosting(
            n_estimators=5,
            hgb_max_iter=20,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)
        assert 0.0 <= oob <= 1.0

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ shape and non-negativity."""
        X, y = classification_data
        model = BaggingClassifier.with_hist_gradient_boosting(
            n_estimators=5,
            hgb_max_iter=30,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, classification_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = classification_data
        model = BaggingClassifier.with_hist_gradient_boosting(
            n_estimators=5, random_state=42
        )
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "HistGradientBoostingClassifier" in r

    def test_hgb_parameters(self, classification_data):
        """Test that hgb_max_iter, learning_rate, and max_depth are accepted."""
        X, y = classification_data
        model = BaggingClassifier.with_hist_gradient_boosting(
            n_estimators=5,
            hgb_max_iter=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape


# ============================================================================
# Cross-Cutting Tests
# ============================================================================


class TestBaggingCrossCutting:
    """Cross-cutting tests for all BaggingClassifier factory methods."""

    def test_n_estimators_property(self, classification_data):
        """Verify n_estimators_ matches configured n_estimators after fit."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)

        assert model.n_estimators_ == 10

    def test_n_estimators_property_custom(self, classification_data):
        """Verify n_estimators_ for a non-default value."""
        X, y = classification_data
        model = BaggingClassifier.with_gaussian_nb(
            n_estimators=7, random_state=42
        )
        model.fit(X, y)

        assert model.n_estimators_ == 7

    def test_custom_n_estimators_20(self, classification_data):
        """Test that n_estimators=20 produces correct output shape."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=20, max_depth=3, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert model.n_estimators_ == 20

    def test_predict_before_fit_raises(self):
        """Predict without fit should raise an error."""
        np.random.seed(42)
        X = np.random.randn(20, 4)
        model = BaggingClassifier.with_decision_tree(n_estimators=5, random_state=42)

        with pytest.raises(Exception):
            model.predict(X)

    def test_predict_proba_before_fit_raises(self):
        """predict_proba without fit should raise an error."""
        np.random.seed(42)
        X = np.random.randn(20, 4)
        model = BaggingClassifier.with_gaussian_nb(n_estimators=5, random_state=42)

        with pytest.raises(Exception):
            model.predict_proba(X)

    def test_feature_importances_before_fit_raises(self):
        """feature_importances_ without fit should raise an error."""
        model = BaggingClassifier.with_decision_tree(n_estimators=5, random_state=42)

        with pytest.raises(Exception):
            _ = model.feature_importances_

    def test_oob_score_none_without_oob_flag(self, classification_data):
        """oob_score_ should be None when oob_score=False (default)."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, oob_score=False, random_state=42
        )
        model.fit(X, y)

        assert model.oob_score_ is None

    def test_max_samples_fraction(self, classification_data):
        """Test max_samples=0.5 (subsample 50% of training samples)."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_samples=0.5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_max_features_fraction(self, classification_data):
        """Test max_features=0.5 (subsample 50% of features)."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_features=0.5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_random_state_reproducibility(self, classification_data):
        """Same random_state gives identical predictions."""
        X, y = classification_data
        model1 = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=0
        )
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=0
        )
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    def test_different_random_states_may_differ(self, classification_data):
        """Different random seeds typically produce different ensemble members."""
        X, y = classification_data
        model1 = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=1
        )
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=99
        )
        model2.fit(X, y)
        pred2 = model2.predict(X)

        # Not guaranteed to differ on every dataset, but expected to for random data
        # We just verify both produce valid output; exact equality is acceptable
        assert pred1.shape == pred2.shape

    def test_multiclass_decision_tree(self, multiclass_data):
        """Test BaggingClassifier.with_decision_tree on 3-class data."""
        X, y = multiclass_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(np.unique(predictions)).issubset({0.0, 1.0, 2.0})

    def test_logistic_regression_binary_only(self, classification_data):
        """Test BaggingClassifier.with_logistic_regression on binary data (LR is binary-only)."""
        X, y = classification_data
        model = BaggingClassifier.with_logistic_regression(
            n_estimators=10, max_iter=300, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_multiclass_predict_proba_columns(self, multiclass_data):
        """Test predict_proba has n_classes columns for multiclass."""
        X, y = multiclass_data
        n_classes = len(np.unique(y))
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (X.shape[0], n_classes)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_single_feature(self, single_feature_data):
        """Test BaggingClassifier with X having a single column."""
        X, y = single_feature_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_many_estimators(self, classification_data):
        """Test with n_estimators=50 (more than default)."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=50, max_depth=3, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert model.n_estimators_ == 50

    def test_bootstrap_false_no_oob(self, classification_data):
        """When bootstrap=False, OOB score cannot be computed."""
        X, y = classification_data
        # With bootstrap=False and oob_score=False, should just work
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, bootstrap=False, oob_score=False, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert model.oob_score_ is None

    def test_warm_start_parameter_accepted(self, classification_data):
        """Test that warm_start=True is accepted without error."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=5, warm_start=True, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_all_factories_produce_valid_predictions(self, classification_data):
        """Smoke test: all 8 factory methods fit and predict without error."""
        X, y = classification_data
        factories = [
            BaggingClassifier.with_decision_tree(
                n_estimators=5, max_depth=3, random_state=42
            ),
            BaggingClassifier.with_random_forest(
                n_estimators=5, rf_n_estimators=5, random_state=42
            ),
            BaggingClassifier.with_logistic_regression(
                n_estimators=5, max_iter=100, random_state=42
            ),
            BaggingClassifier.with_gaussian_nb(n_estimators=5, random_state=42),
            BaggingClassifier.with_knn(n_estimators=5, n_neighbors=3, random_state=42),
            BaggingClassifier.with_svc(n_estimators=5, c=1.0, random_state=42),
            BaggingClassifier.with_gradient_boosting(
                n_estimators=5, gb_n_estimators=10, max_depth=2, random_state=42
            ),
            BaggingClassifier.with_hist_gradient_boosting(
                n_estimators=5, hgb_max_iter=20, random_state=42
            ),
        ]
        for model in factories:
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, (
                f"Wrong shape from {repr(model)}: {predictions.shape}"
            )
            assert np.all(np.isfinite(predictions)), (
                f"Non-finite predictions from {repr(model)}"
            )

    def test_all_factories_predict_proba_valid(self, classification_data):
        """Smoke test: all 8 factory methods produce valid probability outputs."""
        X, y = classification_data
        factories = [
            BaggingClassifier.with_decision_tree(
                n_estimators=5, max_depth=3, random_state=42
            ),
            BaggingClassifier.with_random_forest(
                n_estimators=5, rf_n_estimators=5, random_state=42
            ),
            BaggingClassifier.with_logistic_regression(
                n_estimators=5, max_iter=100, random_state=42
            ),
            BaggingClassifier.with_gaussian_nb(n_estimators=5, random_state=42),
            BaggingClassifier.with_knn(n_estimators=5, n_neighbors=3, random_state=42),
            BaggingClassifier.with_svc(n_estimators=5, c=1.0, random_state=42),
            BaggingClassifier.with_gradient_boosting(
                n_estimators=5, gb_n_estimators=10, max_depth=2, random_state=42
            ),
            BaggingClassifier.with_hist_gradient_boosting(
                n_estimators=5, hgb_max_iter=20, random_state=42
            ),
        ]
        for model in factories:
            model.fit(X, y)
            proba = model.predict_proba(X)
            assert proba.shape[0] == X.shape[0], (
                f"Wrong proba rows from {repr(model)}"
            )
            assert np.all(proba >= 0.0), (
                f"Negative probabilities from {repr(model)}"
            )
            assert np.all(proba <= 1.0), (
                f"Probabilities > 1 from {repr(model)}"
            )
            np.testing.assert_allclose(
                proba.sum(axis=1), 1.0, atol=1e-6,
                err_msg=f"Row probabilities don't sum to 1 for {repr(model)}"
            )

    def test_predict_classes_subset_of_training_labels(self, classification_data):
        """Predictions should only contain class labels seen during training."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)
        training_classes = set(np.unique(y))
        predicted_classes = set(np.unique(predictions))

        assert predicted_classes.issubset(training_classes)

    def test_fit_returns_self(self, classification_data):
        """fit() should return self (enabling method chaining)."""
        X, y = classification_data
        model = BaggingClassifier.with_decision_tree(
            n_estimators=5, max_depth=3, random_state=42
        )
        result = model.fit(X, y)

        # The return value should support predict (i.e., it's a fitted model)
        predictions = result.predict(X)
        assert predictions.shape == y.shape
