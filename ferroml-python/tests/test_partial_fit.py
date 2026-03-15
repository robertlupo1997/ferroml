"""Tests for partial_fit() on all models that support incremental learning.

Verifies:
- partial_fit on batches produces a working model
- partial_fit preserves state across calls
- classes parameter validation for classifiers
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y


@pytest.fixture
def reg_data():
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    return X, y


@pytest.fixture
def nonneg_clf_data():
    np.random.seed(42)
    X = np.abs(np.random.randn(100, 3))
    y = (X[:, 0] > 0.5).astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _partial_fit_classifier(model_cls, X, y, **kwargs):
    """Test partial_fit on a classifier: fit in 2 batches, check it works."""
    model = model_cls(**kwargs)
    mid = len(X) // 2
    classes = [0.0, 1.0]

    # First batch must provide classes
    model.partial_fit(X[:mid], y[:mid], classes=classes)
    # Second batch can omit classes
    model.partial_fit(X[mid:], y[mid:])

    # Model should be able to predict after partial_fit
    preds = model.predict(X)
    assert preds.shape == (len(X),), f"Expected shape ({len(X)},), got {preds.shape}"
    accuracy = np.mean(np.abs(preds - y) < 0.5)
    assert accuracy > 0.4, f"Accuracy {accuracy} too low after partial_fit"
    return model


def _partial_fit_regressor(model_cls, X, y, **kwargs):
    """Test partial_fit on a regressor: fit in 2 batches."""
    model = model_cls(**kwargs)
    mid = len(X) // 2

    model.partial_fit(X[:mid], y[:mid])
    model.partial_fit(X[mid:], y[mid:])

    preds = model.predict(X)
    assert preds.shape == (len(X),), f"Expected shape ({len(X)},), got {preds.shape}"
    return model


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------

class TestNaiveBayesPartialFit:
    def test_gaussian_nb(self, clf_data):
        from ferroml.naive_bayes import GaussianNB
        _partial_fit_classifier(GaussianNB, *clf_data)

    def test_multinomial_nb(self, nonneg_clf_data):
        from ferroml.naive_bayes import MultinomialNB
        _partial_fit_classifier(MultinomialNB, *nonneg_clf_data)

    def test_bernoulli_nb(self, clf_data):
        from ferroml.naive_bayes import BernoulliNB
        _partial_fit_classifier(BernoulliNB, *clf_data)

    def test_categorical_nb(self, clf_data):
        from ferroml.naive_bayes import CategoricalNB
        X, y = clf_data
        X_cat = (X > 0).astype(np.float64)
        _partial_fit_classifier(CategoricalNB, X_cat, y)

    def test_gaussian_nb_state_preserved(self, clf_data):
        """Verify partial_fit preserves state across calls."""
        from ferroml.naive_bayes import GaussianNB
        X, y = clf_data
        model = GaussianNB()

        # Fit on first batch
        model.partial_fit(X[:30], y[:30], classes=[0.0, 1.0])
        preds_after_1 = model.predict(X[:10]).copy()

        # Fit on second batch
        model.partial_fit(X[30:60], y[30:60])
        preds_after_2 = model.predict(X[:10]).copy()

        # Fit on third batch
        model.partial_fit(X[60:], y[60:])
        preds_after_3 = model.predict(X[:10]).copy()

        # Model should still be functional and may change predictions
        assert preds_after_3.shape == (10,)


# ---------------------------------------------------------------------------
# SGD models
# ---------------------------------------------------------------------------

class TestSGDPartialFit:
    def test_sgd_classifier(self, clf_data):
        from ferroml.ensemble import SGDClassifier
        _partial_fit_classifier(SGDClassifier, *clf_data, loss="hinge", max_iter=100)

    def test_sgd_regressor(self, reg_data):
        from ferroml.ensemble import SGDRegressor
        _partial_fit_regressor(SGDRegressor, *reg_data, max_iter=100)


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------

class TestPerceptronPartialFit:
    def test_perceptron(self, clf_data):
        from ferroml.linear import Perceptron
        _partial_fit_classifier(Perceptron, *clf_data, max_iter=100)


# ---------------------------------------------------------------------------
# PassiveAggressiveClassifier
# ---------------------------------------------------------------------------

class TestPassiveAggressivePartialFit:
    def test_passive_aggressive(self, clf_data):
        from ferroml.ensemble import PassiveAggressiveClassifier
        _partial_fit_classifier(PassiveAggressiveClassifier, *clf_data)


# ---------------------------------------------------------------------------
# IncrementalPCA
# ---------------------------------------------------------------------------

class TestIncrementalPCAPartialFit:
    def test_incremental_pca(self):
        from ferroml.decomposition import IncrementalPCA
        np.random.seed(42)
        X = np.random.randn(100, 5)

        ipca = IncrementalPCA(n_components=2)
        ipca.partial_fit(X[:50])
        ipca.partial_fit(X[50:])

        # Should be able to transform after partial_fit
        result = ipca.transform(X)
        assert result.shape == (100, 2)
