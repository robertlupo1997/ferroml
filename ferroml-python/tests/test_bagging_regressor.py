"""Test FerroML BaggingRegressor factory methods and API."""

import numpy as np
import pytest

from ferroml.ensemble import BaggingRegressor


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def regression_data():
    """Generate simple regression data (100 samples, 4 features)."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)
    return X, y.astype(np.float64)


@pytest.fixture(scope="module")
def single_feature_data():
    """Generate regression data with 1 feature."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1)
    y = 2.0 * X[:, 0] + 0.1 * np.random.randn(n_samples)
    return X, y.astype(np.float64)


# ============================================================================
# TestBaggingWithDecisionTree
# ============================================================================


class TestBaggingWithDecisionTree:
    """Tests for BaggingRegressor.with_decision_tree factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, oob_score=True, bootstrap=True, random_state=42
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape, non-negativity, and sum (tree-based)."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(n_estimators=10, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "DecisionTreeRegressor" in r

    def test_min_samples_split_leaf(self, regression_data):
        """Test that min_samples_split and min_samples_leaf are accepted."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
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
    """Tests for BaggingRegressor.with_random_forest factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_random_forest(
            n_estimators=5, rf_n_estimators=10, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_random_forest(
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

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape, non-negativity, and sum (tree-based)."""
        X, y = regression_data
        model = BaggingRegressor.with_random_forest(
            n_estimators=5, rf_n_estimators=10, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_random_forest(n_estimators=5, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "RandomForestRegressor" in r

    def test_rf_n_estimators_parameter(self, regression_data):
        """Test that rf_n_estimators controls inner RF trees."""
        X, y = regression_data
        model = BaggingRegressor.with_random_forest(
            n_estimators=5, rf_n_estimators=5, max_depth=3, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape


# ============================================================================
# TestBaggingWithLinearRegression
# ============================================================================


class TestBaggingWithLinearRegression:
    """Tests for BaggingRegressor.with_linear_regression factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_linear_regression(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_linear_regression(
            n_estimators=10,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape and non-negativity (linear — no sum=1 guarantee)."""
        X, y = regression_data
        model = BaggingRegressor.with_linear_regression(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_linear_regression(
            n_estimators=10, random_state=42
        )
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "LinearRegression" in r

    def test_fit_intercept_parameter(self, regression_data):
        """Test that fit_intercept is accepted."""
        X, y = regression_data
        for fit_intercept in [True, False]:
            model = BaggingRegressor.with_linear_regression(
                n_estimators=5, fit_intercept=fit_intercept, random_state=42
            )
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, (
                f"Failed for fit_intercept={fit_intercept}"
            )


# ============================================================================
# TestBaggingWithRidgeRegression
# ============================================================================


class TestBaggingWithRidgeRegression:
    """Tests for BaggingRegressor.with_ridge_regression factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_ridge_regression(
            n_estimators=10, alpha=1.0, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_ridge_regression(
            n_estimators=10,
            alpha=1.0,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape and non-negativity (linear — no sum=1 guarantee)."""
        X, y = regression_data
        model = BaggingRegressor.with_ridge_regression(
            n_estimators=10, alpha=1.0, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_ridge_regression(
            n_estimators=10, random_state=42
        )
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "RidgeRegression" in r

    def test_alpha_parameter(self, regression_data):
        """Test that alpha (regularization strength) is accepted."""
        X, y = regression_data
        for alpha in [0.01, 1.0, 100.0]:
            model = BaggingRegressor.with_ridge_regression(
                n_estimators=5, alpha=alpha, random_state=42
            )
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, f"Failed for alpha={alpha}"


# ============================================================================
# TestBaggingWithExtraTrees
# ============================================================================


class TestBaggingWithExtraTrees:
    """Tests for BaggingRegressor.with_extra_trees factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_extra_trees(
            n_estimators=5, et_n_estimators=10, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_extra_trees(
            n_estimators=5,
            et_n_estimators=10,
            oob_score=True,
            bootstrap=True,
            random_state=42,
        )
        model.fit(X, y)
        oob = model.oob_score_

        assert oob is not None
        assert isinstance(oob, float)

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape, non-negativity, and sum (tree-based)."""
        X, y = regression_data
        model = BaggingRegressor.with_extra_trees(
            n_estimators=5, et_n_estimators=10, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_extra_trees(n_estimators=5, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "ExtraTreesRegressor" in r

    def test_et_n_estimators_parameter(self, regression_data):
        """Test that et_n_estimators controls inner ET trees."""
        X, y = regression_data
        model = BaggingRegressor.with_extra_trees(
            n_estimators=5, et_n_estimators=5, max_depth=3, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape


# ============================================================================
# TestBaggingWithGradientBoosting
# ============================================================================


class TestBaggingWithGradientBoosting:
    """Tests for BaggingRegressor.with_gradient_boosting factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_gradient_boosting(
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

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_gradient_boosting(
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

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape, non-negativity, and sum (tree-based)."""
        X, y = regression_data
        model = BaggingRegressor.with_gradient_boosting(
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

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_gradient_boosting(
            n_estimators=5, random_state=42
        )
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "GradientBoostingRegressor" in r

    def test_gb_parameters(self, regression_data):
        """Test that gb_n_estimators, learning_rate, and max_depth are accepted."""
        X, y = regression_data
        model = BaggingRegressor.with_gradient_boosting(
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
    """Tests for BaggingRegressor.with_hist_gradient_boosting factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_hist_gradient_boosting(
            n_estimators=5,
            hgb_max_iter=30,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_hist_gradient_boosting(
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

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape, non-negativity, and sum (tree-based)."""
        X, y = regression_data
        model = BaggingRegressor.with_hist_gradient_boosting(
            n_estimators=5,
            hgb_max_iter=30,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_hist_gradient_boosting(
            n_estimators=5, random_state=42
        )
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "HistGradientBoostingRegressor" in r

    def test_hgb_parameters(self, regression_data):
        """Test that hgb_max_iter, learning_rate, and max_depth are accepted."""
        X, y = regression_data
        model = BaggingRegressor.with_hist_gradient_boosting(
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
# TestBaggingWithSVR
# ============================================================================


class TestBaggingWithSVR:
    """Tests for BaggingRegressor.with_svr factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_svr(
            n_estimators=5, c=1.0, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_svr(
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

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape and non-negativity (SVR — no sum=1 guarantee)."""
        X, y = regression_data
        model = BaggingRegressor.with_svr(
            n_estimators=5, c=1.0, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_svr(n_estimators=5, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "SVR" in r

    def test_c_epsilon_parameters(self, regression_data):
        """Test that c and epsilon parameters are accepted."""
        X, y = regression_data
        for c_val, eps in [(0.1, 0.01), (1.0, 0.1), (10.0, 0.5)]:
            model = BaggingRegressor.with_svr(
                n_estimators=5, c=c_val, epsilon=eps, random_state=42
            )
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, (
                f"Failed for c={c_val}, epsilon={eps}"
            )


# ============================================================================
# TestBaggingWithKNN
# ============================================================================


class TestBaggingWithKNN:
    """Tests for BaggingRegressor.with_knn factory method."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict shapes."""
        X, y = regression_data
        model = BaggingRegressor.with_knn(
            n_estimators=10, n_neighbors=5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_oob_score(self, regression_data):
        """Test OOB score is a float when oob_score=True."""
        X, y = regression_data
        model = BaggingRegressor.with_knn(
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

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ shape and non-negativity (KNN — no sum=1 guarantee)."""
        X, y = regression_data
        model = BaggingRegressor.with_knn(
            n_estimators=10, n_neighbors=5, random_state=42
        )
        model.fit(X, y)
        importances = model.feature_importances_

        assert importances.shape == (X.shape[1],)
        assert np.all(importances >= 0.0)

    def test_repr(self, regression_data):
        """Test __repr__ returns a non-empty string containing estimator name."""
        X, y = regression_data
        model = BaggingRegressor.with_knn(n_estimators=10, random_state=42)
        r = repr(model)

        assert isinstance(r, str)
        assert len(r) > 0
        assert "KNeighborsRegressor" in r

    def test_n_neighbors_parameter(self, regression_data):
        """Test that n_neighbors is accepted."""
        X, y = regression_data
        for k in [1, 3, 7]:
            model = BaggingRegressor.with_knn(
                n_estimators=5, n_neighbors=k, random_state=42
            )
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, f"Failed for n_neighbors={k}"


# ============================================================================
# Cross-Cutting Tests
# ============================================================================


class TestBaggingRegressorCrossCutting:
    """Cross-cutting tests for all BaggingRegressor factory methods."""

    def test_n_estimators_property(self, regression_data):
        """Verify n_estimators_ matches configured n_estimators after fit."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)

        assert model.n_estimators_ == 10

    def test_n_estimators_property_custom(self, regression_data):
        """Verify n_estimators_ for a non-default value."""
        X, y = regression_data
        model = BaggingRegressor.with_linear_regression(
            n_estimators=7, random_state=42
        )
        model.fit(X, y)

        assert model.n_estimators_ == 7

    def test_custom_n_estimators_20(self, regression_data):
        """Test that n_estimators=20 produces correct output shape."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
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
        model = BaggingRegressor.with_decision_tree(n_estimators=5, random_state=42)

        with pytest.raises(Exception):
            model.predict(X)

    def test_feature_importances_before_fit_raises(self):
        """feature_importances_ without fit should raise an error."""
        model = BaggingRegressor.with_decision_tree(n_estimators=5, random_state=42)

        with pytest.raises(Exception):
            _ = model.feature_importances_

    def test_oob_score_none_without_oob_flag(self, regression_data):
        """oob_score_ should be None when oob_score=False (default)."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, oob_score=False, random_state=42
        )
        model.fit(X, y)

        assert model.oob_score_ is None

    def test_max_samples_fraction(self, regression_data):
        """Test max_samples=0.5 (subsample 50% of training samples)."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_samples=0.5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_max_features_fraction(self, regression_data):
        """Test max_features=0.5 (subsample 50% of features)."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_features=0.5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_random_state_reproducibility(self, regression_data):
        """Same random_state gives identical predictions."""
        X, y = regression_data
        model1 = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=0
        )
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=0
        )
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    def test_different_random_states_may_differ(self, regression_data):
        """Different random seeds typically produce different ensemble members."""
        X, y = regression_data
        model1 = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=1
        )
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=99
        )
        model2.fit(X, y)
        pred2 = model2.predict(X)

        # Not guaranteed to differ on every dataset, but both must be valid outputs
        assert pred1.shape == pred2.shape

    def test_single_feature(self, single_feature_data):
        """Test BaggingRegressor with X having a single column."""
        X, y = single_feature_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))

    def test_many_estimators(self, regression_data):
        """Test with n_estimators=50 (more than default)."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=50, max_depth=3, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert model.n_estimators_ == 50

    def test_bootstrap_false_no_oob(self, regression_data):
        """When bootstrap=False, OOB score cannot be computed."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, bootstrap=False, oob_score=False, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert model.oob_score_ is None

    def test_warm_start_parameter_accepted(self, regression_data):
        """Test that warm_start=True is accepted without error."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=5, warm_start=True, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_all_factories_produce_valid_predictions(self, regression_data):
        """Smoke test: all 9 factory methods fit and predict without error."""
        X, y = regression_data
        factories = [
            BaggingRegressor.with_decision_tree(
                n_estimators=5, max_depth=3, random_state=42
            ),
            BaggingRegressor.with_random_forest(
                n_estimators=5, rf_n_estimators=5, random_state=42
            ),
            BaggingRegressor.with_linear_regression(
                n_estimators=5, random_state=42
            ),
            BaggingRegressor.with_ridge_regression(
                n_estimators=5, alpha=1.0, random_state=42
            ),
            BaggingRegressor.with_extra_trees(
                n_estimators=5, et_n_estimators=5, random_state=42
            ),
            BaggingRegressor.with_gradient_boosting(
                n_estimators=5, gb_n_estimators=10, max_depth=2, random_state=42
            ),
            BaggingRegressor.with_hist_gradient_boosting(
                n_estimators=5, hgb_max_iter=20, random_state=42
            ),
            BaggingRegressor.with_svr(n_estimators=5, c=1.0, random_state=42),
            BaggingRegressor.with_knn(n_estimators=5, n_neighbors=3, random_state=42),
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

    def test_predictions_are_continuous(self, regression_data):
        """Predictions should be continuous floats, not class labels."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=10, max_depth=5, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        # Continuous predictions: not restricted to {0, 1} or a small integer set
        assert predictions.dtype in (np.float32, np.float64)
        # Should have more than 2 distinct values (unlike classification)
        assert len(np.unique(predictions)) > 2

    def test_fit_returns_self(self, regression_data):
        """fit() should return self (enabling method chaining)."""
        X, y = regression_data
        model = BaggingRegressor.with_decision_tree(
            n_estimators=5, max_depth=3, random_state=42
        )
        result = model.fit(X, y)

        # The return value should support predict (i.e., it's a fitted model)
        predictions = result.predict(X)
        assert predictions.shape == y.shape
