"""
Phase M.3 — FerroML vs sklearn comparison tests for neighbors, SVM, naive Bayes, and neural models.

Compares FerroML implementations against sklearn equivalents with appropriate
tolerances: tight for deterministic algorithms (KNN), wider for iterative solvers (SVM, MLP).
"""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from conftest_comparison import (
    get_iris,
    get_wine,
    get_breast_cancer,
    get_diabetes,
    get_classification_data,
    get_regression_data,
    r2_score,
    accuracy_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scale_data(X):
    return SklearnStandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# KNeighborsClassifier
# ---------------------------------------------------------------------------

class TestKNeighborsClassifier:
    def test_iris_k5(self):
        from ferroml.neighbors import KNeighborsClassifier as FerroKNC
        from sklearn.neighbors import KNeighborsClassifier as SkKNC

        X, y = get_iris()
        ferro = FerroKNC(n_neighbors=5, weights="uniform", metric="euclidean")
        sk = SkKNC(n_neighbors=5, weights="uniform", metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        assert accuracy_score(y, fp) == pytest.approx(accuracy_score(y, sp), abs=1e-6)

    def test_iris_k3_distance_weights(self):
        from ferroml.neighbors import KNeighborsClassifier as FerroKNC
        from sklearn.neighbors import KNeighborsClassifier as SkKNC

        X, y = get_iris()
        ferro = FerroKNC(n_neighbors=3, weights="distance", metric="euclidean")
        sk = SkKNC(n_neighbors=3, weights="distance", metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        assert accuracy_score(y, fp) == pytest.approx(accuracy_score(y, sp), abs=1e-6)

    def test_wine_k7(self):
        from ferroml.neighbors import KNeighborsClassifier as FerroKNC
        from sklearn.neighbors import KNeighborsClassifier as SkKNC

        X, y = get_wine()
        ferro = FerroKNC(n_neighbors=7, weights="uniform", metric="euclidean")
        sk = SkKNC(n_neighbors=7, weights="uniform", metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        assert accuracy_score(y, fp) == pytest.approx(accuracy_score(y, sp), abs=1e-6)

    def test_predict_proba_iris(self):
        from ferroml.neighbors import KNeighborsClassifier as FerroKNC
        from sklearn.neighbors import KNeighborsClassifier as SkKNC

        X, y = get_iris()
        ferro = FerroKNC(n_neighbors=5, weights="uniform", metric="euclidean")
        sk = SkKNC(n_neighbors=5, weights="uniform", metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict_proba(X))
        sp = sk.predict_proba(X)
        assert fp.shape == sp.shape
        assert np.allclose(fp, sp, atol=1e-6)


# ---------------------------------------------------------------------------
# KNeighborsRegressor
# ---------------------------------------------------------------------------

class TestKNeighborsRegressor:
    def test_diabetes_k5(self):
        from ferroml.neighbors import KNeighborsRegressor as FerroKNR
        from sklearn.neighbors import KNeighborsRegressor as SkKNR

        X, y = get_diabetes()
        ferro = FerroKNR(n_neighbors=5, weights="uniform", metric="euclidean")
        sk = SkKNR(n_neighbors=5, weights="uniform", metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y, fp)
        sk_r2 = r2_score(y, sp)
        assert ferro_r2 == pytest.approx(sk_r2, abs=1e-6)

    def test_diabetes_k3_distance(self):
        from ferroml.neighbors import KNeighborsRegressor as FerroKNR
        from sklearn.neighbors import KNeighborsRegressor as SkKNR

        X, y = get_diabetes()
        ferro = FerroKNR(n_neighbors=3, weights="distance", metric="euclidean")
        sk = SkKNR(n_neighbors=3, weights="distance", metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y, fp)
        sk_r2 = r2_score(y, sp)
        assert ferro_r2 == pytest.approx(sk_r2, abs=1e-6)

    def test_synthetic_regression(self):
        from ferroml.neighbors import KNeighborsRegressor as FerroKNR
        from sklearn.neighbors import KNeighborsRegressor as SkKNR

        X, y = get_regression_data(n=200, p=5, random_state=42)
        ferro = FerroKNR(n_neighbors=5, weights="uniform", metric="euclidean")
        sk = SkKNR(n_neighbors=5, weights="uniform", metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y, fp)
        sk_r2 = r2_score(y, sp)
        assert ferro_r2 == pytest.approx(sk_r2, abs=1e-6)


# ---------------------------------------------------------------------------
# NearestCentroid
# ---------------------------------------------------------------------------

class TestNearestCentroid:
    def test_iris(self):
        from ferroml.neighbors import NearestCentroid as FerroNC
        from sklearn.neighbors import NearestCentroid as SkNC

        X, y = get_iris()
        ferro = FerroNC(metric="euclidean")
        sk = SkNC(metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        assert accuracy_score(y, fp) == pytest.approx(accuracy_score(y, sp), abs=1e-6)

    def test_wine(self):
        from ferroml.neighbors import NearestCentroid as FerroNC
        from sklearn.neighbors import NearestCentroid as SkNC

        X, y = get_wine()
        ferro = FerroNC(metric="euclidean")
        sk = SkNC(metric="euclidean")
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        assert accuracy_score(y, fp) == pytest.approx(accuracy_score(y, sp), abs=1e-6)


# ---------------------------------------------------------------------------
# LinearSVC
# ---------------------------------------------------------------------------

class TestLinearSVC:
    def test_breast_cancer_scaled(self):
        from ferroml.svm import LinearSVC as FerroLSVC
        from sklearn.svm import LinearSVC as SkLSVC

        X, y = get_breast_cancer()
        X = scale_data(X)
        ferro = FerroLSVC(c=1.0, max_iter=10000, tol=1e-4)
        sk = SkLSVC(C=1.0, max_iter=10000, tol=1e-4, dual=True)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=0.05)
        # Both should achieve high accuracy on this dataset
        assert ferro_acc > 0.90

    def test_synthetic_binary(self):
        from ferroml.svm import LinearSVC as FerroLSVC
        from sklearn.svm import LinearSVC as SkLSVC

        X, y = get_classification_data(n=500, p=10, n_classes=2, random_state=42)
        X = scale_data(X)
        ferro = FerroLSVC(c=1.0, max_iter=10000)
        sk = SkLSVC(C=1.0, max_iter=10000, dual=True)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=0.05)


# ---------------------------------------------------------------------------
# LinearSVR
# ---------------------------------------------------------------------------

class TestLinearSVR:
    def test_diabetes_scaled(self):
        from ferroml.svm import LinearSVR as FerroLSVR
        from sklearn.svm import LinearSVR as SkLSVR

        X, y = get_diabetes()
        X = scale_data(X)
        ferro = FerroLSVR(c=1.0, max_iter=10000, tol=1e-4, epsilon=0.1)
        sk = SkLSVR(C=1.0, max_iter=10000, tol=1e-4, epsilon=0.1, dual=True)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y, fp)
        sk_r2 = r2_score(y, sp)
        # Different solver implementations; both should achieve reasonable R2
        assert ferro_r2 == pytest.approx(sk_r2, abs=0.15)
        assert ferro_r2 > 0.30

    def test_synthetic_regression(self):
        from ferroml.svm import LinearSVR as FerroLSVR
        from sklearn.svm import LinearSVR as SkLSVR

        X, y = get_regression_data(n=500, p=10, random_state=42)
        X = scale_data(X)
        # Normalize y to avoid scale issues
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / y_std
        ferro = FerroLSVR(c=1.0, max_iter=10000, epsilon=0.1)
        sk = SkLSVR(C=1.0, max_iter=10000, epsilon=0.1, dual=True)
        ferro.fit(X, y_norm)
        sk.fit(X, y_norm)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y_norm, fp)
        sk_r2 = r2_score(y_norm, sp)
        assert ferro_r2 == pytest.approx(sk_r2, abs=0.05)


# ---------------------------------------------------------------------------
# SVC (kernel)
# ---------------------------------------------------------------------------

class TestSVC:
    def test_rbf_iris(self):
        from ferroml.svm import SVC as FerroSVC
        from sklearn.svm import SVC as SkSVC

        X, y = get_iris()
        X = scale_data(X)
        ferro = FerroSVC(kernel="rbf", c=1.0, gamma=0.5, tol=1e-3, max_iter=5000)
        sk = SkSVC(kernel="rbf", C=1.0, gamma=0.5, tol=1e-3, max_iter=5000)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=0.05)
        assert ferro_acc > 0.90

    def test_linear_iris(self):
        from ferroml.svm import SVC as FerroSVC
        from sklearn.svm import SVC as SkSVC

        X, y = get_iris()
        X = scale_data(X)
        ferro = FerroSVC(kernel="linear", c=1.0, tol=1e-3, max_iter=5000)
        sk = SkSVC(kernel="linear", C=1.0, tol=1e-3, max_iter=5000)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=0.05)
        assert ferro_acc > 0.90

    def test_poly_iris(self):
        from ferroml.svm import SVC as FerroSVC
        from sklearn.svm import SVC as SkSVC

        X, y = get_iris()
        X = scale_data(X)
        ferro = FerroSVC(kernel="poly", c=1.0, degree=3, coef0=0.0, gamma=0.5, tol=1e-3, max_iter=5000)
        sk = SkSVC(kernel="poly", C=1.0, degree=3, coef0=0.0, gamma=0.5, tol=1e-3, max_iter=5000)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=0.05)
        assert ferro_acc > 0.85

    def test_rbf_predict_proba(self):
        from ferroml.svm import SVC as FerroSVC
        from sklearn.svm import SVC as SkSVC

        X, y = get_iris()
        X = scale_data(X)
        ferro = FerroSVC(kernel="rbf", c=1.0, gamma=0.5, probability=True, max_iter=5000)
        sk = SkSVC(kernel="rbf", C=1.0, gamma=0.5, probability=True, max_iter=5000)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict_proba(X))
        sp = sk.predict_proba(X)
        # Probabilities may differ due to Platt scaling differences; just check shapes and ranges
        assert fp.shape == sp.shape
        assert np.all(fp >= 0) and np.all(fp <= 1)
        # Row sums should be ~1
        assert np.allclose(fp.sum(axis=1), 1.0, atol=1e-4)

    def test_breast_cancer_rbf(self):
        from ferroml.svm import SVC as FerroSVC
        from sklearn.svm import SVC as SkSVC

        X, y = get_breast_cancer()
        X = scale_data(X)
        ferro = FerroSVC(kernel="rbf", c=1.0, gamma=0.1, tol=1e-3, max_iter=5000)
        sk = SkSVC(kernel="rbf", C=1.0, gamma=0.1, tol=1e-3, max_iter=5000)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=0.05)
        assert ferro_acc > 0.90


# ---------------------------------------------------------------------------
# SVR (kernel)
# ---------------------------------------------------------------------------

class TestSVR:
    def test_rbf_diabetes(self):
        from ferroml.svm import SVR as FerroSVR
        from sklearn.svm import SVR as SkSVR

        X, y = get_diabetes()
        X = scale_data(X)
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / y_std
        ferro = FerroSVR(kernel="rbf", c=1.0, gamma=0.1, epsilon=0.1, tol=1e-3, max_iter=5000)
        sk = SkSVR(kernel="rbf", C=1.0, gamma=0.1, epsilon=0.1, tol=1e-3, max_iter=5000)
        ferro.fit(X, y_norm)
        sk.fit(X, y_norm)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y_norm, fp)
        sk_r2 = r2_score(y_norm, sp)
        assert ferro_r2 == pytest.approx(sk_r2, abs=0.05)

    def test_linear_diabetes(self):
        from ferroml.svm import SVR as FerroSVR
        from sklearn.svm import SVR as SkSVR

        X, y = get_diabetes()
        X = scale_data(X)
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / y_std
        ferro = FerroSVR(kernel="linear", c=1.0, epsilon=0.1, tol=1e-3, max_iter=5000)
        sk = SkSVR(kernel="linear", C=1.0, epsilon=0.1, tol=1e-3, max_iter=5000)
        ferro.fit(X, y_norm)
        sk.fit(X, y_norm)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y_norm, fp)
        sk_r2 = r2_score(y_norm, sp)
        assert ferro_r2 == pytest.approx(sk_r2, abs=0.05)


# ---------------------------------------------------------------------------
# GaussianNB
# ---------------------------------------------------------------------------

class TestGaussianNB:
    def test_iris(self):
        from ferroml.naive_bayes import GaussianNB as FerroGNB
        from sklearn.naive_bayes import GaussianNB as SkGNB

        X, y = get_iris()
        ferro = FerroGNB()
        sk = SkGNB()
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=1e-4)

    def test_wine(self):
        from ferroml.naive_bayes import GaussianNB as FerroGNB
        from sklearn.naive_bayes import GaussianNB as SkGNB

        X, y = get_wine()
        ferro = FerroGNB()
        sk = SkGNB()
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=1e-4)

    def test_predict_proba_iris(self):
        from ferroml.naive_bayes import GaussianNB as FerroGNB
        from sklearn.naive_bayes import GaussianNB as SkGNB

        X, y = get_iris()
        ferro = FerroGNB()
        sk = SkGNB()
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict_proba(X))
        sp = sk.predict_proba(X)
        assert fp.shape == sp.shape
        # GaussianNB is a closed-form model; probabilities should be close
        assert np.allclose(fp, sp, atol=1e-4)

    def test_breast_cancer(self):
        from ferroml.naive_bayes import GaussianNB as FerroGNB
        from sklearn.naive_bayes import GaussianNB as SkGNB

        X, y = get_breast_cancer()
        ferro = FerroGNB()
        sk = SkGNB()
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        # Breast cancer has many features; minor var_smoothing differences can shift a few predictions
        assert ferro_acc == pytest.approx(sk_acc, abs=0.01)


# ---------------------------------------------------------------------------
# MultinomialNB
# ---------------------------------------------------------------------------

class TestMultinomialNB:
    @staticmethod
    def _make_count_data(random_state=42):
        """Synthetic non-negative count features for MultinomialNB."""
        rng = np.random.RandomState(random_state)
        n = 300
        X = rng.poisson(lam=5, size=(n, 10)).astype(np.float64)
        # Class depends on sum of first 5 features vs last 5
        y = (X[:, :5].sum(axis=1) > X[:, 5:].sum(axis=1)).astype(np.float64)
        return X, y

    def test_count_data(self):
        from ferroml.naive_bayes import MultinomialNB as FerroMNB
        from sklearn.naive_bayes import MultinomialNB as SkMNB

        X, y = self._make_count_data()
        ferro = FerroMNB(alpha=1.0)
        sk = SkMNB(alpha=1.0)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=1e-4)

    def test_predict_proba_count(self):
        from ferroml.naive_bayes import MultinomialNB as FerroMNB
        from sklearn.naive_bayes import MultinomialNB as SkMNB

        X, y = self._make_count_data()
        ferro = FerroMNB(alpha=1.0)
        sk = SkMNB(alpha=1.0)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict_proba(X))
        sp = sk.predict_proba(X)
        assert fp.shape == sp.shape
        assert np.allclose(fp, sp, atol=1e-4)


# ---------------------------------------------------------------------------
# BernoulliNB
# ---------------------------------------------------------------------------

class TestBernoulliNB:
    @staticmethod
    def _make_binary_data(random_state=42):
        """Synthetic binary features for BernoulliNB."""
        rng = np.random.RandomState(random_state)
        n = 300
        X = rng.binomial(1, 0.5, size=(n, 15)).astype(np.float64)
        # Class depends on sum of features
        y = (X.sum(axis=1) > 7).astype(np.float64)
        return X, y

    def test_binary_data(self):
        from ferroml.naive_bayes import BernoulliNB as FerroBNB
        from sklearn.naive_bayes import BernoulliNB as SkBNB

        X, y = self._make_binary_data()
        ferro = FerroBNB(alpha=1.0, binarize=0.0)
        sk = SkBNB(alpha=1.0, binarize=0.0)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=1e-4)

    def test_predict_proba_binary(self):
        from ferroml.naive_bayes import BernoulliNB as FerroBNB
        from sklearn.naive_bayes import BernoulliNB as SkBNB

        X, y = self._make_binary_data()
        ferro = FerroBNB(alpha=1.0, binarize=0.0)
        sk = SkBNB(alpha=1.0, binarize=0.0)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = np.array(ferro.predict_proba(X))
        sp = sk.predict_proba(X)
        assert fp.shape == sp.shape
        assert np.allclose(fp, sp, atol=1e-4)

    def test_with_binarize_threshold(self):
        from ferroml.naive_bayes import BernoulliNB as FerroBNB
        from sklearn.naive_bayes import BernoulliNB as SkBNB

        # Continuous features binarized at threshold
        rng = np.random.RandomState(123)
        X = rng.rand(200, 10).astype(np.float64)
        y = (X[:, :5].mean(axis=1) > 0.5).astype(np.float64)

        ferro = FerroBNB(alpha=1.0, binarize=0.5)
        sk = SkBNB(alpha=1.0, binarize=0.5)
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=1e-4)


# ---------------------------------------------------------------------------
# MLPClassifier
# ---------------------------------------------------------------------------

class TestMLPClassifier:
    def test_iris_scaled(self):
        from ferroml.neural import MLPClassifier as FerroMLP
        from sklearn.neural_network import MLPClassifier as SkMLP

        X, y = get_iris()
        X = scale_data(X)
        ferro = FerroMLP(
            hidden_layer_sizes=[64, 32],
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            alpha=0.0001,
            tol=1e-4,
        )
        sk = SkMLP(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            alpha=0.0001,
            tol=1e-4,
        )
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        # MLP is stochastic; wider tolerance
        assert ferro_acc == pytest.approx(sk_acc, abs=0.10)
        # Both should achieve reasonable accuracy
        assert ferro_acc > 0.85

    def test_breast_cancer_scaled(self):
        from ferroml.neural import MLPClassifier as FerroMLP
        from sklearn.neural_network import MLPClassifier as SkMLP

        X, y = get_breast_cancer()
        X = scale_data(X)
        ferro = FerroMLP(
            hidden_layer_sizes=[32],
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )
        sk = SkMLP(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )
        ferro.fit(X, y)
        sk.fit(X, y)

        fp = ferro.predict(X)
        sp = sk.predict(X)
        ferro_acc = accuracy_score(y, fp)
        sk_acc = accuracy_score(y, sp)
        assert ferro_acc == pytest.approx(sk_acc, abs=0.10)
        assert ferro_acc > 0.90

    def test_predict_proba_iris(self):
        from ferroml.neural import MLPClassifier as FerroMLP

        X, y = get_iris()
        X = scale_data(X)
        ferro = FerroMLP(
            hidden_layer_sizes=[32],
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )
        ferro.fit(X, y)
        fp = np.array(ferro.predict_proba(X))
        # Check shape and valid probability distributions
        assert fp.shape == (len(X), 3)
        assert np.all(fp >= 0) and np.all(fp <= 1)
        assert np.allclose(fp.sum(axis=1), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# MLPRegressor
# ---------------------------------------------------------------------------

class TestMLPRegressor:
    def test_diabetes_scaled(self):
        from ferroml.neural import MLPRegressor as FerroMLPR
        from sklearn.neural_network import MLPRegressor as SkMLPR

        X, y = get_diabetes()
        X = scale_data(X)
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / y_std

        ferro = FerroMLPR(
            hidden_layer_sizes=[64, 32],
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            alpha=0.0001,
        )
        sk = SkMLPR(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            alpha=0.0001,
        )
        ferro.fit(X, y_norm)
        sk.fit(X, y_norm)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y_norm, fp)
        sk_r2 = r2_score(y_norm, sp)
        # MLP implementations differ in initialization and optimizer details;
        # both should achieve reasonable R2 on this dataset
        assert ferro_r2 > 0.40
        assert sk_r2 > 0.40

    def test_synthetic_regression(self):
        from ferroml.neural import MLPRegressor as FerroMLPR
        from sklearn.neural_network import MLPRegressor as SkMLPR

        X, y = get_regression_data(n=300, p=10, random_state=42)
        X = scale_data(X)
        y_mean, y_std = y.mean(), y.std()
        y_norm = (y - y_mean) / y_std

        ferro = FerroMLPR(
            hidden_layer_sizes=[32],
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )
        sk = SkMLPR(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )
        ferro.fit(X, y_norm)
        sk.fit(X, y_norm)

        fp = np.array(ferro.predict(X))
        sp = sk.predict(X)
        ferro_r2 = r2_score(y_norm, fp)
        sk_r2 = r2_score(y_norm, sp)
        assert ferro_r2 == pytest.approx(sk_r2, abs=0.10)
        assert ferro_r2 > 0.30
