"""
Tests for RecursiveFeatureElimination (RFE) Python bindings.

Covers all 13 factory methods and cross-cutting behavior including
fit/transform, properties after fit, and edge cases.
"""

import numpy as np
import pytest

from ferroml.preprocessing import RecursiveFeatureElimination


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data():
    """100 samples, 8 features; only first 2 are informative."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 8))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.standard_normal(100) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    """100 samples, 8 features; binary target based on first 2 features."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def small_data():
    """50 samples, 4 features for edge-case tests."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.standard_normal(50) * 0.05
    return X, y


# ---------------------------------------------------------------------------
# TestRFEWithLinearRegression
# ---------------------------------------------------------------------------


class TestRFEWithLinearRegression:
    def test_fit_transform_basic(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)

    def test_fit_intercept_false(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(
            n_features_to_select=4, fit_intercept=False
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)


# ---------------------------------------------------------------------------
# TestRFEWithRidge
# ---------------------------------------------------------------------------


class TestRFEWithRidge:
    def test_fit_transform_basic(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_ridge(n_features_to_select=4)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_ridge(n_features_to_select=4)
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_ridge(n_features_to_select=4)
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)

    def test_custom_alpha(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_ridge(n_features_to_select=4, alpha=10.0)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)


# ---------------------------------------------------------------------------
# TestRFEWithLasso
# ---------------------------------------------------------------------------


class TestRFEWithLasso:
    def test_fit_transform_basic(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_lasso(n_features_to_select=4)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_lasso(n_features_to_select=4)
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_lasso(n_features_to_select=4)
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithLogisticRegression
# ---------------------------------------------------------------------------


class TestRFEWithLogisticRegression:
    def test_fit_transform_basic(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_logistic_regression(n_features_to_select=4)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_logistic_regression(n_features_to_select=4)
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_logistic_regression(n_features_to_select=4)
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)

    def test_custom_max_iter(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_logistic_regression(
            n_features_to_select=4, max_iter=200
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)


# ---------------------------------------------------------------------------
# TestRFEWithDecisionTreeClassifier
# ---------------------------------------------------------------------------


class TestRFEWithDecisionTreeClassifier:
    def test_fit_transform_basic(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_decision_tree_classifier(n_features_to_select=4)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_decision_tree_classifier(n_features_to_select=4)
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_decision_tree_classifier(n_features_to_select=4)
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithDecisionTreeRegressor
# ---------------------------------------------------------------------------


class TestRFEWithDecisionTreeRegressor:
    def test_fit_transform_basic(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_decision_tree_regressor(n_features_to_select=4)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_decision_tree_regressor(n_features_to_select=4)
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_decision_tree_regressor(n_features_to_select=4)
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithRandomForestClassifier
# ---------------------------------------------------------------------------


class TestRFEWithRandomForestClassifier:
    def test_fit_transform_basic(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_random_forest_classifier(
            n_features_to_select=4, n_estimators=20
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_random_forest_classifier(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_random_forest_classifier(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithRandomForestRegressor
# ---------------------------------------------------------------------------


class TestRFEWithRandomForestRegressor:
    def test_fit_transform_basic(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_random_forest_regressor(
            n_features_to_select=4, n_estimators=20
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_random_forest_regressor(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_random_forest_regressor(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithGradientBoostingClassifier
# ---------------------------------------------------------------------------


class TestRFEWithGradientBoostingClassifier:
    def test_fit_transform_basic(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_gradient_boosting_classifier(
            n_features_to_select=4, n_estimators=20
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_gradient_boosting_classifier(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_gradient_boosting_classifier(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithGradientBoostingRegressor
# ---------------------------------------------------------------------------


class TestRFEWithGradientBoostingRegressor:
    def test_fit_transform_basic(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_gradient_boosting_regressor(
            n_features_to_select=4, n_estimators=20
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_gradient_boosting_regressor(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_gradient_boosting_regressor(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithExtraTreesClassifier
# ---------------------------------------------------------------------------


class TestRFEWithExtraTreesClassifier:
    def test_fit_transform_basic(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_extra_trees_classifier(
            n_features_to_select=4, n_estimators=20
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_extra_trees_classifier(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, classification_data):
        X, y = classification_data
        rfe = RecursiveFeatureElimination.with_extra_trees_classifier(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithExtraTreesRegressor
# ---------------------------------------------------------------------------


class TestRFEWithExtraTreesRegressor:
    def test_fit_transform_basic(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_extra_trees_regressor(
            n_features_to_select=4, n_estimators=20
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)

    def test_selected_indices_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_extra_trees_regressor(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert len(rfe.selected_indices_) == 4

    def test_ranking_shape(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_extra_trees_regressor(
            n_features_to_select=4, n_estimators=20
        )
        rfe.fit(X, y)
        assert rfe.ranking_.shape == (8,)


# ---------------------------------------------------------------------------
# TestRFEWithSVR
# ---------------------------------------------------------------------------


class TestRFEWithSVR:
    """SVR does not expose feature importances, so RFE raises RuntimeError."""

    def test_fit_raises_no_feature_importances(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_svr(n_features_to_select=4)
        with pytest.raises(ValueError, match="feature importances"):
            rfe.fit_transform(X, y)

    def test_custom_c_and_epsilon_still_raises(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_svr(
            n_features_to_select=4, c=0.5, epsilon=0.05
        )
        with pytest.raises(ValueError, match="feature importances"):
            rfe.fit_transform(X, y)


# ---------------------------------------------------------------------------
# TestRFECrossCutting
# ---------------------------------------------------------------------------


class TestRFECrossCutting:
    def test_transform_before_fit_raises(self, regression_data):
        X, _ = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        with pytest.raises(Exception):
            rfe.transform(X)

    def test_fit_transform_equivalent_to_fit_then_transform(self, regression_data):
        X, y = regression_data
        rfe1 = RecursiveFeatureElimination.with_ridge(n_features_to_select=4)
        rfe2 = RecursiveFeatureElimination.with_ridge(n_features_to_select=4)

        X_ft = rfe1.fit_transform(X, y)
        rfe2.fit(X, y)
        X_t = rfe2.transform(X)

        # Both should produce matrices of the same shape
        assert X_ft.shape == X_t.shape
        # The same RFE instance should produce identical columns
        np.testing.assert_array_equal(X_ft, X_t)

    def test_n_features_to_select_controls_output(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=3)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 3)
        assert len(rfe.selected_indices_) == 3

    def test_step_parameter(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(
            n_features_to_select=4, step=2
        )
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 4)
        assert len(rfe.selected_indices_) == 4

    def test_all_regression_factories_produce_valid_output(self, regression_data):
        X, y = regression_data
        factories = [
            RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4),
            RecursiveFeatureElimination.with_ridge(n_features_to_select=4),
            RecursiveFeatureElimination.with_lasso(n_features_to_select=4),
            RecursiveFeatureElimination.with_decision_tree_regressor(n_features_to_select=4),
            RecursiveFeatureElimination.with_random_forest_regressor(
                n_features_to_select=4, n_estimators=10
            ),
            RecursiveFeatureElimination.with_gradient_boosting_regressor(
                n_features_to_select=4, n_estimators=10
            ),
            RecursiveFeatureElimination.with_extra_trees_regressor(
                n_features_to_select=4, n_estimators=10
            ),
            # SVR excluded: does not expose feature importances
        ]
        for rfe in factories:
            X_out = rfe.fit_transform(X, y)
            assert X_out.shape == (100, 4), f"Expected (100, 4), got {X_out.shape}"

    def test_all_classification_factories_produce_valid_output(self, classification_data):
        X, y = classification_data
        factories = [
            RecursiveFeatureElimination.with_logistic_regression(n_features_to_select=4),
            RecursiveFeatureElimination.with_decision_tree_classifier(n_features_to_select=4),
            RecursiveFeatureElimination.with_random_forest_classifier(
                n_features_to_select=4, n_estimators=10
            ),
            RecursiveFeatureElimination.with_gradient_boosting_classifier(
                n_features_to_select=4, n_estimators=10
            ),
            RecursiveFeatureElimination.with_extra_trees_classifier(
                n_features_to_select=4, n_estimators=10
            ),
        ]
        for rfe in factories:
            X_out = rfe.fit_transform(X, y)
            assert X_out.shape == (100, 4), f"Expected (100, 4), got {X_out.shape}"

    def test_informative_features_selected(self):
        """With a very strong linear signal, RFE with linear regression should
        select the two informative features (indices 0 and 1)."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 8))
        y = 5.0 * X[:, 0] + 4.0 * X[:, 1] + rng.standard_normal(200) * 0.01
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=2)
        rfe.fit(X, y)
        selected = set(rfe.selected_indices_)
        assert 0 in selected, f"Feature 0 not selected; selected={selected}"
        assert 1 in selected, f"Feature 1 not selected; selected={selected}"

    def test_ranking_values_valid(self, regression_data):
        """Ranking: 1 for selected features, strictly > 1 for eliminated ones."""
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        rfe.fit(X, y)
        ranking = rfe.ranking_
        selected = set(rfe.selected_indices_)
        for i in range(8):
            if i in selected:
                assert ranking[i] == 1, f"Selected feature {i} has ranking {ranking[i]}, expected 1"
            else:
                assert ranking[i] > 1, f"Eliminated feature {i} has ranking {ranking[i]}, expected > 1"

    def test_support_matches_selected_indices(self, regression_data):
        """support_[i] is True iff i is in selected_indices_."""
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_ridge(n_features_to_select=4)
        rfe.fit(X, y)
        support = rfe.support_
        selected = set(rfe.selected_indices_)
        assert len(support) == 8
        for i, s in enumerate(support):
            if i in selected:
                assert s is True, f"support_[{i}] should be True (feature is selected)"
            else:
                assert s is False, f"support_[{i}] should be False (feature is eliminated)"

    def test_n_iterations_positive(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        rfe.fit(X, y)
        assert rfe.n_iterations_ >= 1

    def test_repr_contains_info(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        r = repr(rfe)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        result = rfe.fit(X, y)
        assert result is rfe

    def test_single_feature_selected(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=1)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (100, 1)
        assert len(rfe.selected_indices_) == 1
        assert sum(rfe.support_) == 1

    def test_selected_indices_sorted(self, regression_data):
        """selected_indices_ should be in ascending order."""
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        rfe.fit(X, y)
        indices = rfe.selected_indices_
        assert indices == sorted(indices), f"selected_indices_ not sorted: {indices}"

    def test_ranking_dtype_is_integer(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=4)
        rfe.fit(X, y)
        assert np.issubdtype(rfe.ranking_.dtype, np.integer), (
            f"Expected integer dtype, got {rfe.ranking_.dtype}"
        )

    def test_transform_preserves_row_count(self, regression_data):
        X, y = regression_data
        rfe = RecursiveFeatureElimination.with_ridge(n_features_to_select=4)
        rfe.fit(X, y)
        X_new = np.random.default_rng(1).standard_normal((30, 8))
        X_out = rfe.transform(X_new)
        assert X_out.shape == (30, 4)

    def test_small_data_select_two(self, small_data):
        """Smoke test on 4-feature dataset selecting 2 features."""
        X, y = small_data
        rfe = RecursiveFeatureElimination.with_lasso(n_features_to_select=2)
        X_out = rfe.fit_transform(X, y)
        assert X_out.shape == (50, 2)
        assert len(rfe.selected_indices_) == 2
        assert rfe.ranking_.shape == (4,)
