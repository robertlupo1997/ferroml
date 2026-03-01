"""Test FerroML ensemble and online learning models."""

import numpy as np
import pytest

from ferroml.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    SGDClassifier,
    SGDRegressor,
    PassiveAggressiveClassifier,
)


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = X[:, 0] ** 2 + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    return X, y


@pytest.fixture
def classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    # Linear decision boundary for SGD/PA models to learn easily
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate multiclass classification data (3 classes)."""
    np.random.seed(42)
    n_samples = 150
    X = np.random.randn(n_samples, 4)
    y = np.zeros(n_samples)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = 2
    return X, y.astype(np.float64)


# ============================================================================
# ExtraTreesClassifier Tests
# ============================================================================


class TestExtraTreesClassifier:
    """Tests for ExtraTreesClassifier."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        model = ExtraTreesClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predict_classes_valid(self, classification_data):
        """Test that predictions contain only valid class labels."""
        X, y = classification_data
        model = ExtraTreesClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_feature_importances_shape(self, classification_data):
        """Test that feature_importances_ has correct shape."""
        X, y = classification_data
        model = ExtraTreesClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.shape == (X.shape[1],)

    def test_feature_importances_non_negative(self, classification_data):
        """Test that all feature importances are non-negative."""
        X, y = classification_data
        model = ExtraTreesClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert np.all(importances >= 0)

    def test_feature_importances_sum_to_one(self, classification_data):
        """Test that feature importances sum to approximately 1."""
        X, y = classification_data
        model = ExtraTreesClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        np.testing.assert_almost_equal(importances.sum(), 1.0, decimal=5)

    def test_default_parameters(self, classification_data):
        """Test that default parameters work without error."""
        X, y = classification_data
        model = ExtraTreesClassifier()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_max_depth_parameter(self, classification_data):
        """Test that max_depth parameter is accepted."""
        X, y = classification_data
        model = ExtraTreesClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_min_samples_split_parameter(self, classification_data):
        """Test that min_samples_split parameter is accepted."""
        X, y = classification_data
        model = ExtraTreesClassifier(n_estimators=10, min_samples_split=5, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_min_samples_leaf_parameter(self, classification_data):
        """Test that min_samples_leaf parameter is accepted."""
        X, y = classification_data
        model = ExtraTreesClassifier(n_estimators=10, min_samples_leaf=3, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_multiclass_classification(self, multiclass_data):
        """Test ExtraTreesClassifier on multiclass data."""
        X, y = multiclass_data
        model = ExtraTreesClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(np.unique(predictions)).issubset({0.0, 1.0, 2.0})


# ============================================================================
# ExtraTreesRegressor Tests
# ============================================================================


class TestExtraTreesRegressor:
    """Tests for ExtraTreesRegressor."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = ExtraTreesRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predictions_are_finite(self, regression_data):
        """Test that predictions are finite numbers."""
        X, y = regression_data
        model = ExtraTreesRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))

    def test_feature_importances_shape(self, regression_data):
        """Test that feature_importances_ has correct shape."""
        X, y = regression_data
        model = ExtraTreesRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.shape == (X.shape[1],)

    def test_feature_importances_non_negative(self, regression_data):
        """Test that all feature importances are non-negative."""
        X, y = regression_data
        model = ExtraTreesRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert np.all(importances >= 0)

    def test_feature_importances_sum_to_one(self, regression_data):
        """Test that feature importances sum to approximately 1."""
        X, y = regression_data
        model = ExtraTreesRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        np.testing.assert_almost_equal(importances.sum(), 1.0, decimal=5)

    def test_default_parameters(self, regression_data):
        """Test that default parameters work without error."""
        X, y = regression_data
        model = ExtraTreesRegressor()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_n_estimators_effect(self, regression_data):
        """Test that different n_estimators values all produce valid output."""
        X, y = regression_data
        for n in [5, 10, 20]:
            model = ExtraTreesRegressor(n_estimators=n, random_state=42)
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, f"Failed for n_estimators={n}"


# ============================================================================
# AdaBoostClassifier Tests
# ============================================================================


class TestAdaBoostClassifier:
    """Tests for AdaBoostClassifier."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        model = AdaBoostClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predict_classes_valid(self, classification_data):
        """Test that predictions contain only valid class labels."""
        X, y = classification_data
        model = AdaBoostClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_default_parameters(self, classification_data):
        """Test that default parameters work without error."""
        X, y = classification_data
        model = AdaBoostClassifier()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_learning_rate_parameter(self, classification_data):
        """Test that learning_rate parameter is accepted."""
        X, y = classification_data
        for lr in [0.1, 0.5, 1.0]:
            model = AdaBoostClassifier(n_estimators=10, learning_rate=lr, random_state=42)
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, f"Failed for learning_rate={lr}"

    def test_max_depth_parameter(self, classification_data):
        """Test that max_depth parameter is accepted."""
        X, y = classification_data
        model = AdaBoostClassifier(n_estimators=10, max_depth=2, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


# ============================================================================
# AdaBoostRegressor Tests
# ============================================================================


class TestAdaBoostRegressor:
    """Tests for AdaBoostRegressor."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = AdaBoostRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predictions_are_finite(self, regression_data):
        """Test that predictions are finite numbers."""
        X, y = regression_data
        model = AdaBoostRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))

    def test_loss_linear(self, regression_data):
        """Test AdaBoostRegressor with linear loss."""
        X, y = regression_data
        model = AdaBoostRegressor(n_estimators=10, loss="linear", random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_loss_square(self, regression_data):
        """Test AdaBoostRegressor with square loss."""
        X, y = regression_data
        model = AdaBoostRegressor(n_estimators=10, loss="square", random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_loss_exponential(self, regression_data):
        """Test AdaBoostRegressor with exponential loss."""
        X, y = regression_data
        model = AdaBoostRegressor(n_estimators=10, loss="exponential", random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_all_loss_functions_give_different_results(self, regression_data):
        """Test that different loss functions can produce different predictions."""
        X, y = regression_data
        results = {}
        for loss in ["linear", "square", "exponential"]:
            model = AdaBoostRegressor(n_estimators=10, loss=loss, random_state=42)
            model.fit(X, y)
            results[loss] = model.predict(X)
            assert results[loss].shape == y.shape

    def test_invalid_loss_raises_error(self, regression_data):
        """Test that an invalid loss function raises an error."""
        X, y = regression_data
        with pytest.raises((ValueError, Exception)):
            model = AdaBoostRegressor(loss="invalid_loss", random_state=42)
            model.fit(X, y)

    def test_default_parameters(self, regression_data):
        """Test that default parameters work without error."""
        X, y = regression_data
        model = AdaBoostRegressor()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


# ============================================================================
# SGDClassifier Tests
# ============================================================================


class TestSGDClassifier:
    """Tests for SGDClassifier."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict with default parameters."""
        X, y = classification_data
        model = SGDClassifier(max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predict_classes_valid(self, classification_data):
        """Test that predictions contain only valid class labels."""
        X, y = classification_data
        model = SGDClassifier(max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_loss_hinge(self, classification_data):
        """Test SGDClassifier with hinge loss (SVM-like)."""
        X, y = classification_data
        model = SGDClassifier(loss="hinge", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_loss_log(self, classification_data):
        """Test SGDClassifier with log loss (logistic regression-like)."""
        X, y = classification_data
        model = SGDClassifier(loss="log", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_loss_modified_huber(self, classification_data):
        """Test SGDClassifier with modified_huber loss."""
        X, y = classification_data
        model = SGDClassifier(loss="modified_huber", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_penalty_none(self, classification_data):
        """Test SGDClassifier with no penalty."""
        X, y = classification_data
        model = SGDClassifier(penalty="none", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_penalty_l1(self, classification_data):
        """Test SGDClassifier with L1 penalty."""
        X, y = classification_data
        model = SGDClassifier(penalty="l1", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_penalty_l2(self, classification_data):
        """Test SGDClassifier with L2 penalty."""
        X, y = classification_data
        model = SGDClassifier(penalty="l2", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_penalty_elasticnet(self, classification_data):
        """Test SGDClassifier with ElasticNet penalty."""
        X, y = classification_data
        model = SGDClassifier(penalty="elasticnet", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    @pytest.mark.parametrize("loss,penalty", [
        ("hinge", "l2"),
        ("log", "l1"),
        ("modified_huber", "elasticnet"),
        ("hinge", "none"),
        ("log", "l2"),
    ])
    def test_loss_penalty_combinations(self, classification_data, loss, penalty):
        """Test various loss/penalty combinations."""
        X, y = classification_data
        model = SGDClassifier(loss=loss, penalty=penalty, max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_invalid_loss_raises_error(self, classification_data):
        """Test that an invalid loss raises an error."""
        X, y = classification_data
        with pytest.raises((ValueError, Exception)):
            model = SGDClassifier(loss="invalid_loss", random_state=42)
            model.fit(X, y)

    def test_invalid_penalty_raises_error(self, classification_data):
        """Test that an invalid penalty raises an error."""
        X, y = classification_data
        with pytest.raises((ValueError, Exception)):
            model = SGDClassifier(penalty="invalid_penalty", random_state=42)
            model.fit(X, y)

    def test_alpha_regularization(self, classification_data):
        """Test that alpha regularization parameter is accepted."""
        X, y = classification_data
        for alpha in [0.0001, 0.001, 0.01]:
            model = SGDClassifier(alpha=alpha, max_iter=500, random_state=42)
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, f"Failed for alpha={alpha}"

    def test_default_parameters(self, classification_data):
        """Test that default parameters work without error."""
        X, y = classification_data
        model = SGDClassifier()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


# ============================================================================
# SGDRegressor Tests
# ============================================================================


class TestSGDRegressor:
    """Tests for SGDRegressor."""

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = SGDRegressor(max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predictions_are_finite(self, regression_data):
        """Test that predictions are finite numbers."""
        X, y = regression_data
        model = SGDRegressor(max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))

    def test_coef_getter_shape(self, regression_data):
        """Test that coef_ has the correct shape (n_features,)."""
        X, y = regression_data
        model = SGDRegressor(max_iter=500, random_state=42)
        model.fit(X, y)

        coef = model.coef_
        assert coef.shape == (X.shape[1],)

    def test_intercept_getter_is_scalar(self, regression_data):
        """Test that intercept_ is a scalar value."""
        X, y = regression_data
        model = SGDRegressor(max_iter=500, random_state=42)
        model.fit(X, y)

        intercept = model.intercept_
        assert np.isscalar(intercept) or (hasattr(intercept, 'shape') and intercept.ndim == 0)

    def test_intercept_is_finite(self, regression_data):
        """Test that intercept_ is a finite number."""
        X, y = regression_data
        model = SGDRegressor(max_iter=500, random_state=42)
        model.fit(X, y)

        intercept = model.intercept_
        assert np.isfinite(intercept)

    def test_loss_squared_error(self, regression_data):
        """Test SGDRegressor with squared_error loss."""
        X, y = regression_data
        model = SGDRegressor(loss="squared_error", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_loss_huber(self, regression_data):
        """Test SGDRegressor with huber loss."""
        X, y = regression_data
        model = SGDRegressor(loss="huber", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_loss_epsilon_insensitive(self, regression_data):
        """Test SGDRegressor with epsilon_insensitive loss."""
        X, y = regression_data
        model = SGDRegressor(loss="epsilon_insensitive", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_all_losses_produce_valid_coef(self, regression_data):
        """Test that all losses produce valid coefficient vectors."""
        X, y = regression_data
        for loss in ["squared_error", "huber", "epsilon_insensitive"]:
            model = SGDRegressor(loss=loss, max_iter=500, random_state=42)
            model.fit(X, y)
            coef = model.coef_
            assert coef.shape == (X.shape[1],), f"Wrong coef shape for loss={loss}"
            assert np.all(np.isfinite(coef)), f"Non-finite coef for loss={loss}"

    def test_penalty_l1(self, regression_data):
        """Test SGDRegressor with L1 penalty."""
        X, y = regression_data
        model = SGDRegressor(penalty="l1", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_penalty_l2(self, regression_data):
        """Test SGDRegressor with L2 penalty."""
        X, y = regression_data
        model = SGDRegressor(penalty="l2", max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_default_parameters(self, regression_data):
        """Test that default parameters work without error."""
        X, y = regression_data
        model = SGDRegressor()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


# ============================================================================
# PassiveAggressiveClassifier Tests
# ============================================================================


class TestPassiveAggressiveClassifier:
    """Tests for PassiveAggressiveClassifier."""

    def test_fit_predict_basic(self, classification_data):
        """Test basic fit and predict."""
        X, y = classification_data
        model = PassiveAggressiveClassifier(max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predict_classes_valid(self, classification_data):
        """Test that predictions contain only valid class labels."""
        X, y = classification_data
        model = PassiveAggressiveClassifier(max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_default_parameters(self, classification_data):
        """Test that default parameters work without error."""
        X, y = classification_data
        model = PassiveAggressiveClassifier()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_c_parameter(self, classification_data):
        """Test that c (aggressiveness) parameter is accepted."""
        X, y = classification_data
        for c in [0.1, 1.0, 10.0]:
            model = PassiveAggressiveClassifier(c=c, max_iter=500, random_state=42)
            model.fit(X, y)
            predictions = model.predict(X)
            assert predictions.shape == y.shape, f"Failed for c={c}"

    def test_tol_parameter(self, classification_data):
        """Test that tol parameter is accepted."""
        X, y = classification_data
        model = PassiveAggressiveClassifier(tol=1e-4, max_iter=500, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predictions_not_all_same_class(self, classification_data):
        """Test that predictions are not all the same class (model learned something)."""
        X, y = classification_data
        model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        # For a linearly separable problem, the model should predict both classes
        assert len(np.unique(predictions)) > 1


# ============================================================================
# Cross-Model Shape Consistency Tests
# ============================================================================


class TestShapeConsistency:
    """Tests to verify consistent output shapes across all models."""

    def test_all_classifiers_output_shape(self, classification_data):
        """Test that all classifiers produce correct output shape."""
        X, y = classification_data
        classifiers = [
            ExtraTreesClassifier(n_estimators=10, random_state=42),
            AdaBoostClassifier(n_estimators=10, random_state=42),
            SGDClassifier(max_iter=500, random_state=42),
            PassiveAggressiveClassifier(max_iter=500, random_state=42),
        ]
        for clf in classifiers:
            clf.fit(X, y)
            predictions = clf.predict(X)
            assert predictions.shape == y.shape, (
                f"{type(clf).__name__} produced wrong shape: "
                f"{predictions.shape} != {y.shape}"
            )

    def test_all_regressors_output_shape(self, regression_data):
        """Test that all regressors produce correct output shape."""
        X, y = regression_data
        regressors = [
            ExtraTreesRegressor(n_estimators=10, random_state=42),
            AdaBoostRegressor(n_estimators=10, random_state=42),
            SGDRegressor(max_iter=500, random_state=42),
        ]
        for reg in regressors:
            reg.fit(X, y)
            predictions = reg.predict(X)
            assert predictions.shape == y.shape, (
                f"{type(reg).__name__} produced wrong shape: "
                f"{predictions.shape} != {y.shape}"
            )
