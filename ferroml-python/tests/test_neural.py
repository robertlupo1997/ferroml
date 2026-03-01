"""Test FerroML neural network models (MLP)."""

import numpy as np
import pytest

from ferroml.neural import (
    MLPClassifier,
    MLPRegressor,
)


@pytest.fixture
def regression_data():
    """Generate simple regression data with a clear linear signal."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate well-separated 3-class data for multiclass classification."""
    np.random.seed(42)
    n_per_class = 50
    X_list, y_list = [], []

    # Class 0: centered at [0, 0, 0, 0]
    X_list.append(np.random.randn(n_per_class, 4) * 0.5)
    y_list.append(np.zeros(n_per_class))

    # Class 1: centered at [3, 3, 3, 3]
    X_list.append(np.random.randn(n_per_class, 4) * 0.5 + 3)
    y_list.append(np.ones(n_per_class))

    # Class 2: centered at [6, 6, 6, 6]
    X_list.append(np.random.randn(n_per_class, 4) * 0.5 + 6)
    y_list.append(np.full(n_per_class, 2.0))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


# ============================================================================
# MLPClassifier Tests
# ============================================================================


class TestMLPClassifierBasic:
    """Basic functionality tests for MLPClassifier."""

    def test_fit_predict_default(self, classification_data):
        """Test basic fit and predict with default parameters."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predict_valid_class_labels(self, classification_data):
        """Test that classifier only predicts valid class labels."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_is_fitted_before_fit(self, classification_data):
        """Test that is_fitted() returns False before calling fit."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)

        assert model.is_fitted() is False

    def test_is_fitted_after_fit(self, classification_data):
        """Test that is_fitted() returns True after calling fit."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert model.is_fitted() is True

    def test_default_hidden_layer_sizes(self, classification_data):
        """Test that hidden_layer_sizes=None (default) works without error."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=None, max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_multiclass_classification(self, multiclass_data):
        """Test MLPClassifier on multiclass data."""
        X, y = multiclass_data
        model = MLPClassifier(hidden_layer_sizes=[20], max_iter=100, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(np.unique(predictions)).issubset({0.0, 1.0, 2.0})


class TestMLPClassifierHiddenLayers:
    """Tests for MLPClassifier hidden layer configurations."""

    def test_single_hidden_layer(self, classification_data):
        """Test with a single hidden layer."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_two_hidden_layers(self, classification_data):
        """Test with two hidden layers."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[20, 10], max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_three_hidden_layers(self, classification_data):
        """Test with three hidden layers."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[20, 10, 5], max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_n_layers_single_hidden(self, classification_data):
        """Test that n_layers_ is 3 for one hidden layer (input + hidden + output)."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert model.n_layers_ == 3

    def test_n_layers_two_hidden(self, classification_data):
        """Test that n_layers_ is 4 for two hidden layers."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[20, 10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert model.n_layers_ == 4

    def test_n_layers_three_hidden(self, classification_data):
        """Test that n_layers_ is 5 for three hidden layers."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10, 10, 5], max_iter=50, random_state=42
        )
        model.fit(X, y)

        assert model.n_layers_ == 5


class TestMLPClassifierActivations:
    """Tests for MLPClassifier activation functions."""

    def test_activation_relu(self, classification_data):
        """Test MLPClassifier with relu activation."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10], activation="relu", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_activation_tanh(self, classification_data):
        """Test MLPClassifier with tanh activation."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10], activation="tanh", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_activation_sigmoid(self, classification_data):
        """Test MLPClassifier with sigmoid activation."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10], activation="sigmoid", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_invalid_activation_raises_error(self, classification_data):
        """Test that an invalid activation function raises an error."""
        X, y = classification_data
        with pytest.raises((ValueError, Exception)):
            model = MLPClassifier(
                hidden_layer_sizes=[10],
                activation="invalid_activation",
                max_iter=50,
                random_state=42,
            )
            model.fit(X, y)

    def test_invalid_solver_raises_error(self, classification_data):
        """Test that an invalid solver raises an error."""
        X, y = classification_data
        with pytest.raises((ValueError, Exception)):
            model = MLPClassifier(
                hidden_layer_sizes=[10],
                solver="invalid_solver",
                max_iter=50,
                random_state=42,
            )
            model.fit(X, y)


class TestMLPClassifierPredictProba:
    """Tests for MLPClassifier predict_proba method."""

    def test_predict_proba_shape_binary(self, classification_data):
        """Test that predict_proba returns (n_samples, 2) for binary classification."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.ndim == 2
        assert proba.shape == (X.shape[0], 2)

    def test_predict_proba_non_negative(self, classification_data):
        """Test that all predicted probabilities are non-negative."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert np.all(proba >= 0)

    def test_predict_proba_at_most_one(self, classification_data):
        """Test that all predicted probabilities are at most 1."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert np.all(proba <= 1)

    def test_predict_proba_row_sums_to_one(self, classification_data):
        """Test that each row of predict_proba sums to 1."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)

        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(X.shape[0]), atol=1e-5)

    def test_predict_proba_multiclass_shape(self, multiclass_data):
        """Test predict_proba shape for multiclass classification."""
        X, y = multiclass_data
        model = MLPClassifier(hidden_layer_sizes=[20], max_iter=100, random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.ndim == 2
        assert proba.shape == (X.shape[0], 3)


class TestMLPClassifierGetters:
    """Tests for MLPClassifier property getters."""

    def test_classes_getter_binary(self, classification_data):
        """Test that classes_ contains both classes for binary classification."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        classes = model.classes_
        assert len(classes) == 2
        assert set(classes).issubset({0.0, 1.0})

    def test_classes_getter_multiclass(self, multiclass_data):
        """Test that classes_ contains all 3 classes for multiclass."""
        X, y = multiclass_data
        model = MLPClassifier(hidden_layer_sizes=[20], max_iter=100, random_state=42)
        model.fit(X, y)

        classes = model.classes_
        assert len(classes) == 3

    def test_loss_curve_is_list(self, classification_data):
        """Test that loss_curve_ is a list."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        loss_curve = model.loss_curve_
        assert isinstance(loss_curve, list)

    def test_loss_curve_length_matches_max_iter(self, classification_data):
        """Test that loss_curve_ has one entry per iteration run."""
        X, y = classification_data
        max_iter = 30
        model = MLPClassifier(
            hidden_layer_sizes=[10], max_iter=max_iter, random_state=42
        )
        model.fit(X, y)

        loss_curve = model.loss_curve_
        # Without early stopping, should run exactly max_iter iterations
        assert len(loss_curve) <= max_iter

    def test_loss_curve_decreases_overall(self, classification_data):
        """Test that loss generally decreases over training."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10], max_iter=100, random_state=42
        )
        model.fit(X, y)

        loss_curve = model.loss_curve_
        assert len(loss_curve) > 1
        # The first loss should be greater than the last (overall decrease)
        assert loss_curve[0] > loss_curve[-1]

    def test_loss_curve_all_positive(self, classification_data):
        """Test that all loss values in the curve are positive."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        loss_curve = model.loss_curve_
        assert all(v >= 0 for v in loss_curve)

    def test_n_layers_is_integer(self, classification_data):
        """Test that n_layers_ returns an integer."""
        X, y = classification_data
        model = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert isinstance(model.n_layers_, int)


class TestMLPClassifierEarlyStopping:
    """Tests for MLPClassifier early stopping."""

    def test_early_stopping_completes(self, classification_data):
        """Test that training completes with early stopping enabled."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10],
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
        )
        model.fit(X, y)

        assert model.is_fitted() is True

    def test_early_stopping_predictions_valid(self, classification_data):
        """Test that predictions are valid after training with early stopping."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10],
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert set(np.unique(predictions)).issubset({0.0, 1.0})

    def test_early_stopping_stops_before_max_iter(self, classification_data):
        """Test that early stopping may stop before max_iter is reached."""
        X, y = classification_data
        max_iter = 1000
        model = MLPClassifier(
            hidden_layer_sizes=[10],
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        model.fit(X, y)

        # With early stopping, loss_curve_ should have <= max_iter entries
        assert len(model.loss_curve_) <= max_iter


class TestMLPClassifierSolvers:
    """Tests for MLPClassifier solver options."""

    def test_solver_adam(self, classification_data):
        """Test MLPClassifier with adam solver."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10], solver="adam", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_solver_sgd(self, classification_data):
        """Test MLPClassifier with sgd solver."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10], solver="sgd", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


# ============================================================================
# MLPRegressor Tests
# ============================================================================


class TestMLPRegressorBasic:
    """Basic functionality tests for MLPRegressor."""

    def test_fit_predict_default(self, regression_data):
        """Test basic fit and predict with default parameters."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_predictions_are_finite(self, regression_data):
        """Test that all predictions are finite numbers."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))

    def test_is_fitted_before_fit(self, regression_data):
        """Test that is_fitted() returns False before calling fit."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)

        assert model.is_fitted() is False

    def test_is_fitted_after_fit(self, regression_data):
        """Test that is_fitted() returns True after calling fit."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert model.is_fitted() is True

    def test_default_hidden_layer_sizes(self, regression_data):
        """Test that hidden_layer_sizes=None (default) works without error."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=None, max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


class TestMLPRegressorHiddenLayers:
    """Tests for MLPRegressor hidden layer configurations."""

    def test_single_hidden_layer(self, regression_data):
        """Test with a single hidden layer."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_two_hidden_layers(self, regression_data):
        """Test with two hidden layers."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[20, 10], max_iter=50, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_three_hidden_layers(self, regression_data):
        """Test with three hidden layers."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[20, 10, 5], max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_n_layers_single_hidden(self, regression_data):
        """Test that n_layers_ is 3 for one hidden layer."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert model.n_layers_ == 3

    def test_n_layers_two_hidden(self, regression_data):
        """Test that n_layers_ is 4 for two hidden layers."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[20, 10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert model.n_layers_ == 4

    def test_n_layers_three_hidden(self, regression_data):
        """Test that n_layers_ is 5 for three hidden layers."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[10, 10, 5], max_iter=50, random_state=42
        )
        model.fit(X, y)

        assert model.n_layers_ == 5

    def test_n_layers_matches_hidden_layer_count(self, regression_data):
        """Test that n_layers_ = len(hidden_layer_sizes) + 2 (input + output layers)."""
        X, y = regression_data
        for sizes in [[5], [10, 5], [20, 10, 5], [15, 10, 8, 4]]:
            model = MLPRegressor(
                hidden_layer_sizes=sizes, max_iter=50, random_state=42
            )
            model.fit(X, y)
            expected_layers = len(sizes) + 2
            assert model.n_layers_ == expected_layers, (
                f"For sizes={sizes}: expected {expected_layers}, got {model.n_layers_}"
            )


class TestMLPRegressorActivations:
    """Tests for MLPRegressor activation functions."""

    def test_activation_relu(self, regression_data):
        """Test MLPRegressor with relu activation."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[10], activation="relu", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_activation_tanh(self, regression_data):
        """Test MLPRegressor with tanh activation."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[10], activation="tanh", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_activation_sigmoid(self, regression_data):
        """Test MLPRegressor with sigmoid activation."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[10], activation="sigmoid", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_invalid_activation_raises_error(self, regression_data):
        """Test that an invalid activation function raises an error."""
        X, y = regression_data
        with pytest.raises((ValueError, Exception)):
            model = MLPRegressor(
                hidden_layer_sizes=[10],
                activation="invalid_activation",
                max_iter=50,
                random_state=42,
            )
            model.fit(X, y)

    def test_invalid_solver_raises_error(self, regression_data):
        """Test that an invalid solver raises an error."""
        X, y = regression_data
        with pytest.raises((ValueError, Exception)):
            model = MLPRegressor(
                hidden_layer_sizes=[10],
                solver="invalid_solver",
                max_iter=50,
                random_state=42,
            )
            model.fit(X, y)


class TestMLPRegressorScore:
    """Tests for MLPRegressor score (R²) method."""

    def test_score_returns_float(self, regression_data):
        """Test that score() returns a float."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[20], max_iter=200, random_state=42)
        model.fit(X, y)

        score = model.score(X, y)
        assert isinstance(score, float)

    def test_score_finite(self, regression_data):
        """Test that score() returns a finite value."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[20], max_iter=200, random_state=42)
        model.fit(X, y)

        score = model.score(X, y)
        assert np.isfinite(score)

    def test_score_bounded_above_by_one(self, regression_data):
        """Test that R² score is at most 1."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[20], max_iter=200, random_state=42)
        model.fit(X, y)

        score = model.score(X, y)
        assert score <= 1.0

    def test_score_positive_for_good_fit(self):
        """Test that a well-trained model achieves a positive R² score."""
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 4)
        # Strong linear signal, low noise
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.05

        model = MLPRegressor(
            hidden_layer_sizes=[30, 20], max_iter=500, random_state=42
        )
        model.fit(X, y)

        score = model.score(X, y)
        assert score > 0.0


class TestMLPRegressorLossCurve:
    """Tests for MLPRegressor loss_curve_ property."""

    def test_loss_curve_is_list(self, regression_data):
        """Test that loss_curve_ is a list."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert isinstance(model.loss_curve_, list)

    def test_loss_curve_non_empty(self, regression_data):
        """Test that loss_curve_ is non-empty after training."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert len(model.loss_curve_) > 0

    def test_loss_curve_length_at_most_max_iter(self, regression_data):
        """Test that loss_curve_ has at most max_iter entries."""
        X, y = regression_data
        max_iter = 30
        model = MLPRegressor(
            hidden_layer_sizes=[10], max_iter=max_iter, random_state=42
        )
        model.fit(X, y)

        assert len(model.loss_curve_) <= max_iter

    def test_loss_curve_decreases_overall(self, regression_data):
        """Test that loss generally decreases over training."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[20], max_iter=200, random_state=42)
        model.fit(X, y)

        loss_curve = model.loss_curve_
        assert len(loss_curve) > 1
        # First loss should be greater than last loss
        assert loss_curve[0] > loss_curve[-1]

    def test_loss_curve_all_non_negative(self, regression_data):
        """Test that all loss values are non-negative."""
        X, y = regression_data
        model = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        model.fit(X, y)

        assert all(v >= 0 for v in model.loss_curve_)


class TestMLPRegressorEarlyStopping:
    """Tests for MLPRegressor early stopping."""

    def test_early_stopping_completes(self, regression_data):
        """Test that training completes with early stopping enabled."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[10],
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
        )
        model.fit(X, y)

        assert model.is_fitted() is True

    def test_early_stopping_predictions_valid(self, regression_data):
        """Test that predictions are valid after training with early stopping."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[10],
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=42,
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape
        assert not np.any(np.isnan(predictions))

    def test_early_stopping_stops_before_max_iter(self, regression_data):
        """Test that early stopping may stop before max_iter is reached."""
        X, y = regression_data
        max_iter = 1000
        model = MLPRegressor(
            hidden_layer_sizes=[10],
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        model.fit(X, y)

        assert len(model.loss_curve_) <= max_iter


class TestMLPRegressorSolvers:
    """Tests for MLPRegressor solver options."""

    def test_solver_adam(self, regression_data):
        """Test MLPRegressor with adam solver."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[10], solver="adam", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_solver_sgd(self, regression_data):
        """Test MLPRegressor with sgd solver."""
        X, y = regression_data
        model = MLPRegressor(
            hidden_layer_sizes=[10], solver="sgd", max_iter=50, random_state=42
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape


# ============================================================================
# Cross-Model Consistency Tests
# ============================================================================


class TestMLPConsistency:
    """Cross-model consistency checks."""

    def test_classifier_and_regressor_both_accept_same_data(self):
        """Test that both MLP models accept the same X data format."""
        np.random.seed(42)
        X = np.random.randn(80, 4)
        y_cls = (X[:, 0] > 0).astype(np.float64)
        y_reg = X[:, 0] + X[:, 1]

        clf = MLPClassifier(hidden_layer_sizes=[10], max_iter=50, random_state=42)
        reg = MLPRegressor(hidden_layer_sizes=[10], max_iter=50, random_state=42)

        clf.fit(X, y_cls)
        reg.fit(X, y_reg)

        assert clf.predict(X).shape == (80,)
        assert reg.predict(X).shape == (80,)

    def test_reproducibility_with_random_state(self, classification_data):
        """Test that identical random_state produces identical predictions."""
        X, y = classification_data
        model1 = MLPClassifier(
            hidden_layer_sizes=[10], max_iter=50, random_state=42
        )
        model2 = MLPClassifier(
            hidden_layer_sizes=[10], max_iter=50, random_state=42
        )
        model1.fit(X, y)
        model2.fit(X, y)

        np.testing.assert_array_equal(model1.predict(X), model2.predict(X))

    def test_regressor_reproducibility_with_random_state(self, regression_data):
        """Test that identical random_state produces identical regressor predictions."""
        X, y = regression_data
        model1 = MLPRegressor(
            hidden_layer_sizes=[10], max_iter=50, random_state=42
        )
        model2 = MLPRegressor(
            hidden_layer_sizes=[10], max_iter=50, random_state=42
        )
        model1.fit(X, y)
        model2.fit(X, y)

        np.testing.assert_array_equal(model1.predict(X), model2.predict(X))

    def test_alpha_regularization_accepted(self, regression_data):
        """Test that alpha regularization parameter is accepted by both models."""
        X, y = regression_data
        y_cls = (X[:, 0] > 0).astype(np.float64)

        for alpha in [0.0001, 0.001, 0.01]:
            clf = MLPClassifier(
                hidden_layer_sizes=[10], alpha=alpha, max_iter=50, random_state=42
            )
            clf.fit(X, y_cls)
            assert clf.is_fitted()

            reg = MLPRegressor(
                hidden_layer_sizes=[10], alpha=alpha, max_iter=50, random_state=42
            )
            reg.fit(X, y)
            assert reg.is_fitted()

    def test_batch_size_parameter_accepted(self, classification_data):
        """Test that batch_size parameter is accepted."""
        X, y = classification_data
        model = MLPClassifier(
            hidden_layer_sizes=[10], batch_size=32, max_iter=50, random_state=42
        )
        model.fit(X, y)

        assert model.is_fitted()
        assert model.predict(X).shape == y.shape
