"""Tests for FerroML Naive Bayes classifiers."""

import numpy as np
import pytest

from ferroml.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


# ---------------------------------------------------------------------------
# GaussianNB tests
# ---------------------------------------------------------------------------

class TestGaussianNB:
    def test_fit_predict_basic(self):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0],
                       [6.0, 7.0], [7.0, 6.0], [8.0, 8.0]])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (6,)
        assert np.all(np.isin(preds, [0.0, 1.0]))

    def test_fit_predict_perfect_separation(self):
        X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_predict_proba_sums_to_one(self):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [6.0, 7.0], [7.0, 6.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert probas.shape == (4, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_values_in_range(self):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [6.0, 7.0], [7.0, 6.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_classes_getter(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        classes = model.classes_
        np.testing.assert_array_equal(classes, [0.0, 1.0])

    def test_theta_shape(self):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [6.0, 7.0], [7.0, 6.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        theta = model.theta_
        assert theta.shape == (2, 2)  # 2 classes, 2 features

    def test_var_shape(self):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [6.0, 7.0], [7.0, 6.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        var = model.var_
        assert var.shape == (2, 2)  # 2 classes, 2 features
        assert np.all(var >= 0.0)

    def test_var_smoothing_effect(self):
        # Use well-separated data with enough samples
        X = np.array([[1.0], [1.5], [2.0], [2.5], [8.0], [8.5], [9.0], [9.5]])
        y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        model_default = GaussianNB(var_smoothing=1e-9)
        model_default.fit(X, y)
        model_large = GaussianNB(var_smoothing=10.0)
        model_large.fit(X, y)
        # Default should predict correctly with good separation
        preds_default = model_default.predict(X)
        np.testing.assert_array_equal(preds_default, y)
        # With large smoothing, probabilities should be less extreme
        probas_default = model_default.predict_proba(X)
        probas_large = model_large.predict_proba(X)
        # max probability should be closer to 0.5 with more smoothing
        assert np.max(probas_large) <= np.max(probas_default) + 1e-10

    def test_unfitted_predict_raises(self):
        model = GaussianNB()
        X = np.array([[1.0, 2.0]])
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_unfitted_classes_raises(self):
        model = GaussianNB()
        with pytest.raises(ValueError):
            _ = model.classes_

    def test_unfitted_theta_raises(self):
        model = GaussianNB()
        with pytest.raises(ValueError):
            _ = model.theta_

    def test_unfitted_var_raises(self):
        model = GaussianNB()
        with pytest.raises(ValueError):
            _ = model.var_

    def test_repr(self):
        model = GaussianNB(var_smoothing=1e-5)
        r = repr(model)
        assert "GaussianNB" in r
        assert "var_smoothing" in r

    def test_three_classes(self):
        X = np.array([[0.0, 0.0], [0.1, 0.1],
                       [5.0, 5.0], [5.1, 5.1],
                       [10.0, 10.0], [10.1, 10.1]])
        y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        model = GaussianNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (6,)
        assert set(preds.tolist()).issubset({0.0, 1.0, 2.0})
        classes = model.classes_
        np.testing.assert_array_equal(classes, [0.0, 1.0, 2.0])
        probas = model.predict_proba(X)
        assert probas.shape == (6, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)

    def test_single_feature(self):
        X = np.array([[1.0], [2.0], [10.0], [11.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_integer_labels(self):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [6.0, 7.0], [7.0, 6.0]])
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        model = GaussianNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_fit_returns_self(self):
        X = np.array([[1.0, 2.0], [6.0, 7.0]])
        y = np.array([0.0, 1.0])
        model = GaussianNB()
        result = model.fit(X, y)
        assert result is model


# ---------------------------------------------------------------------------
# MultinomialNB tests
# ---------------------------------------------------------------------------

class TestMultinomialNB:
    def test_fit_predict_basic(self):
        X = np.array([[5.0, 1.0, 0.0], [4.0, 2.0, 0.0],
                       [0.0, 1.0, 5.0], [0.0, 0.0, 6.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = MultinomialNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)
        np.testing.assert_array_equal(preds, y)

    def test_predict_proba_sums_to_one(self):
        X = np.array([[5.0, 1.0, 0.0], [4.0, 2.0, 0.0],
                       [0.0, 1.0, 5.0], [0.0, 0.0, 6.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = MultinomialNB()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert probas.shape == (4, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)

    def test_alpha_smoothing(self):
        X = np.array([[5.0, 0.0], [4.0, 0.0],
                       [0.0, 5.0], [0.0, 4.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model_smooth = MultinomialNB(alpha=1.0)
        model_smooth.fit(X, y)
        model_less = MultinomialNB(alpha=0.01)
        model_less.fit(X, y)
        # Both should predict correctly
        preds_smooth = model_smooth.predict(X)
        preds_less = model_less.predict(X)
        np.testing.assert_array_equal(preds_smooth, y)
        np.testing.assert_array_equal(preds_less, y)

    def test_fit_prior_false(self):
        X = np.array([[5.0, 0.0], [4.0, 0.0], [3.0, 0.0],
                       [0.0, 5.0]])
        y = np.array([0.0, 0.0, 0.0, 1.0])
        model = MultinomialNB(fit_prior=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_classes_getter(self):
        X = np.array([[5.0, 0.0], [0.0, 5.0]])
        y = np.array([0.0, 1.0])
        model = MultinomialNB()
        model.fit(X, y)
        classes = model.classes_
        np.testing.assert_array_equal(classes, [0.0, 1.0])

    def test_unfitted_predict_raises(self):
        model = MultinomialNB()
        X = np.array([[1.0, 2.0]])
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_unfitted_classes_raises(self):
        model = MultinomialNB()
        with pytest.raises(ValueError):
            _ = model.classes_

    def test_repr(self):
        model = MultinomialNB(alpha=0.5, fit_prior=False)
        r = repr(model)
        assert "MultinomialNB" in r
        assert "alpha" in r

    def test_three_classes(self):
        X = np.array([[5.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                       [0.0, 5.0, 0.0], [0.0, 4.0, 0.0],
                       [0.0, 0.0, 5.0], [0.0, 0.0, 4.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        model = MultinomialNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds.tolist()).issubset({0.0, 1.0, 2.0})
        probas = model.predict_proba(X)
        assert probas.shape == (6, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)

    def test_fit_returns_self(self):
        X = np.array([[5.0, 0.0], [0.0, 5.0]])
        y = np.array([0.0, 1.0])
        model = MultinomialNB()
        result = model.fit(X, y)
        assert result is model


# ---------------------------------------------------------------------------
# BernoulliNB tests
# ---------------------------------------------------------------------------

class TestBernoulliNB:
    def test_fit_predict_basic(self):
        X = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = BernoulliNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)
        np.testing.assert_array_equal(preds, y)

    def test_predict_proba_sums_to_one(self):
        X = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = BernoulliNB()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert probas.shape == (4, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)

    def test_binarize_threshold(self):
        # Continuous data that should be binarized at threshold 0.5
        X = np.array([[0.8, 0.9, 0.1], [0.7, 0.2, 0.1],
                       [0.1, 0.8, 0.9], [0.2, 0.1, 0.8]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = BernoulliNB(binarize=0.5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)
        np.testing.assert_array_equal(preds, y)

    def test_fit_prior_false(self):
        X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                       [0.0, 1.0]])
        y = np.array([0.0, 0.0, 0.0, 1.0])
        model = BernoulliNB(fit_prior=False)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_classes_getter(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([0.0, 1.0])
        model = BernoulliNB()
        model.fit(X, y)
        classes = model.classes_
        np.testing.assert_array_equal(classes, [0.0, 1.0])

    def test_unfitted_predict_raises(self):
        model = BernoulliNB()
        X = np.array([[1.0, 0.0]])
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_unfitted_classes_raises(self):
        model = BernoulliNB()
        with pytest.raises(ValueError):
            _ = model.classes_

    def test_repr(self):
        model = BernoulliNB(alpha=0.5, binarize=0.5, fit_prior=False)
        r = repr(model)
        assert "BernoulliNB" in r
        assert "alpha" in r

    def test_three_classes(self):
        X = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        model = BernoulliNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds.tolist()).issubset({0.0, 1.0, 2.0})
        probas = model.predict_proba(X)
        assert probas.shape == (6, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)

    def test_single_feature(self):
        X = np.array([[1.0], [1.0], [0.0], [0.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = BernoulliNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_fit_returns_self(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([0.0, 1.0])
        model = BernoulliNB()
        result = model.fit(X, y)
        assert result is model
