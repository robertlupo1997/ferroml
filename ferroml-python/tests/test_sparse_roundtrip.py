"""Tests for scipy.sparse -> Rust CsrMatrix round-trip (no densification).

Verifies that sparse matrices flow through to Rust SparseModel trait methods
without being densified, and produce correct results matching dense equivalents.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from ferroml.linear import LogisticRegression, RidgeRegression
from ferroml.naive_bayes import BernoulliNB, MultinomialNB
from ferroml.preprocessing import TfidfTransformer
from ferroml.svm import LinearSVC, LinearSVR


# =============================================================================
# Helpers
# =============================================================================


def _make_classification_data(n_samples=50, n_features=20, density=0.3, seed=42):
    """Create sparse classification data."""
    rng = np.random.RandomState(seed)
    X_dense = rng.randn(n_samples, n_features)
    # Sparsify: zero out entries below threshold
    mask = rng.rand(n_samples, n_features) > density
    X_dense[mask] = 0.0
    X_sparse = sp.csr_matrix(X_dense)
    # Simple linear decision boundary
    w = rng.randn(n_features)
    y = (X_dense @ w > 0).astype(np.float64)
    return X_sparse, X_dense, y


def _make_regression_data(n_samples=50, n_features=20, density=0.3, seed=42):
    """Create sparse regression data."""
    rng = np.random.RandomState(seed)
    X_dense = rng.randn(n_samples, n_features)
    mask = rng.rand(n_samples, n_features) > density
    X_dense[mask] = 0.0
    X_sparse = sp.csr_matrix(X_dense)
    w = rng.randn(n_features)
    y = X_dense @ w + rng.randn(n_samples) * 0.1
    return X_sparse, X_dense, y


def _make_count_data(n_samples=50, n_features=20, density=0.3, seed=42):
    """Create sparse count data (non-negative integers) for NB models."""
    rng = np.random.RandomState(seed)
    X_dense = rng.poisson(lam=2, size=(n_samples, n_features)).astype(np.float64)
    mask = rng.rand(n_samples, n_features) > density
    X_dense[mask] = 0.0
    X_sparse = sp.csr_matrix(X_dense)
    y = rng.choice([0.0, 1.0], size=n_samples)
    return X_sparse, X_dense, y


# =============================================================================
# LogisticRegression sparse round-trip
# =============================================================================


class TestLogisticSparse:
    def test_fit_predict_sparse(self):
        """LogisticRegression fit_sparse / predict_sparse basic smoke test."""
        X_sparse, _, y = _make_classification_data()
        model = LogisticRegression(max_iter=200)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (X_sparse.shape[0],)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_sparse_vs_dense_match(self):
        """Sparse and dense fitting produce similar predictions."""
        X_sparse, X_dense, y = _make_classification_data(seed=123)

        model_dense = LogisticRegression(max_iter=200)
        model_dense.fit(X_dense, y)
        preds_dense = model_dense.predict(X_dense)

        model_sparse = LogisticRegression(max_iter=200)
        model_sparse.fit_sparse(X_sparse, y)
        preds_sparse = model_sparse.predict_sparse(X_sparse)

        # Predictions should match (same algorithm, same data)
        np.testing.assert_array_equal(preds_dense, preds_sparse)


# =============================================================================
# RidgeRegression sparse round-trip
# =============================================================================


class TestRidgeSparse:
    def test_fit_predict_sparse(self):
        """RidgeRegression fit_sparse / predict_sparse basic smoke test."""
        X_sparse, _, y = _make_regression_data()
        model = RidgeRegression(alpha=1.0)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (X_sparse.shape[0],)

    def test_sparse_vs_dense_match(self):
        """Sparse and dense Ridge produce same predictions."""
        X_sparse, X_dense, y = _make_regression_data(seed=77)

        model_dense = RidgeRegression(alpha=1.0)
        model_dense.fit(X_dense, y)
        preds_dense = model_dense.predict(X_dense)

        model_sparse = RidgeRegression(alpha=1.0)
        model_sparse.fit_sparse(X_sparse, y)
        preds_sparse = model_sparse.predict_sparse(X_sparse)

        np.testing.assert_allclose(preds_dense, preds_sparse, atol=1e-6)


# =============================================================================
# MultinomialNB sparse round-trip
# =============================================================================


class TestMultinomialNBSparse:
    def test_fit_predict_sparse(self):
        """MultinomialNB fit_sparse / predict_sparse basic smoke test."""
        X_sparse, _, y = _make_count_data()
        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (X_sparse.shape[0],)
        assert set(np.unique(preds)).issubset({0.0, 1.0})

    def test_sparse_vs_dense_match(self):
        """Sparse and dense MultinomialNB produce same predictions."""
        X_sparse, X_dense, y = _make_count_data(seed=99)

        model_dense = MultinomialNB()
        model_dense.fit(X_dense, y)
        preds_dense = model_dense.predict(X_dense)

        model_sparse = MultinomialNB()
        model_sparse.fit_sparse(X_sparse, y)
        preds_sparse = model_sparse.predict_sparse(X_sparse)

        np.testing.assert_array_equal(preds_dense, preds_sparse)


# =============================================================================
# BernoulliNB sparse round-trip
# =============================================================================


class TestBernoulliNBSparse:
    def test_fit_predict_sparse(self):
        """BernoulliNB fit_sparse / predict_sparse basic smoke test."""
        rng = np.random.RandomState(42)
        X_dense = (rng.rand(50, 20) > 0.6).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)
        y = rng.choice([0.0, 1.0], size=50)

        model = BernoulliNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (50,)

    def test_sparse_vs_dense_match(self):
        """Sparse and dense BernoulliNB produce same predictions."""
        rng = np.random.RandomState(77)
        X_dense = (rng.rand(50, 20) > 0.6).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)
        y = rng.choice([0.0, 1.0], size=50)

        model_dense = BernoulliNB()
        model_dense.fit(X_dense, y)
        preds_dense = model_dense.predict(X_dense)

        model_sparse = BernoulliNB()
        model_sparse.fit_sparse(X_sparse, y)
        preds_sparse = model_sparse.predict_sparse(X_sparse)

        np.testing.assert_array_equal(preds_dense, preds_sparse)


# =============================================================================
# LinearSVC sparse round-trip
# =============================================================================


class TestLinearSVCSparse:
    def test_fit_predict_sparse(self):
        """LinearSVC fit_sparse / predict_sparse basic smoke test."""
        X_sparse, _, y = _make_classification_data()
        model = LinearSVC(max_iter=2000)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (X_sparse.shape[0],)

    def test_sparse_vs_dense_match(self):
        """Sparse and dense LinearSVC produce same predictions."""
        X_sparse, X_dense, y = _make_classification_data(seed=55)

        model_dense = LinearSVC(max_iter=2000)
        model_dense.fit(X_dense, y)
        preds_dense = model_dense.predict(X_dense)

        model_sparse = LinearSVC(max_iter=2000)
        model_sparse.fit_sparse(X_sparse, y)
        preds_sparse = model_sparse.predict_sparse(X_sparse)

        np.testing.assert_array_equal(preds_dense, preds_sparse)


# =============================================================================
# LinearSVR sparse round-trip
# =============================================================================


class TestLinearSVRSparse:
    def test_fit_predict_sparse(self):
        """LinearSVR fit_sparse / predict_sparse basic smoke test."""
        X_sparse, _, y = _make_regression_data()
        model = LinearSVR(max_iter=2000)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (X_sparse.shape[0],)

    def test_sparse_vs_dense_match(self):
        """Sparse and dense LinearSVR produce same predictions."""
        X_sparse, X_dense, y = _make_regression_data(seed=55)

        model_dense = LinearSVR(max_iter=2000)
        model_dense.fit(X_dense, y)
        preds_dense = model_dense.predict(X_dense)

        model_sparse = LinearSVR(max_iter=2000)
        model_sparse.fit_sparse(X_sparse, y)
        preds_sparse = model_sparse.predict_sparse(X_sparse)

        np.testing.assert_allclose(preds_dense, preds_sparse, atol=1e-6)


# =============================================================================
# TfidfTransformer sparse round-trip
# =============================================================================


class TestTfidfSparse:
    def test_fit_transform_sparse(self):
        """TfidfTransformer accepts scipy.sparse input."""
        rng = np.random.RandomState(42)
        X_dense = rng.poisson(lam=3, size=(30, 15)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)

        tfidf = TfidfTransformer()
        tfidf.fit_sparse(X_sparse)
        result = tfidf.transform_sparse(X_sparse)
        assert result.shape == (30, 15)
        # TF-IDF values should be non-negative
        assert np.all(result >= 0)

    def test_fit_transform_sparse_single_step(self):
        """TfidfTransformer fit_transform_sparse in one step."""
        rng = np.random.RandomState(42)
        X_dense = rng.poisson(lam=3, size=(30, 15)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)

        tfidf = TfidfTransformer()
        result = tfidf.fit_transform_sparse(X_sparse)
        assert result.shape == (30, 15)

    def test_sparse_vs_dense_match(self):
        """Sparse and dense TfidfTransformer produce same output."""
        rng = np.random.RandomState(42)
        X_dense = rng.poisson(lam=3, size=(30, 15)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)

        tfidf_dense = TfidfTransformer()
        result_dense = tfidf_dense.fit_transform(X_dense)

        tfidf_sparse = TfidfTransformer()
        tfidf_sparse.fit_sparse(X_sparse)
        result_sparse = tfidf_sparse.transform_sparse(X_sparse)

        np.testing.assert_allclose(result_dense, result_sparse, atol=1e-10)


# =============================================================================
# End-to-end sparse pipeline
# =============================================================================


class TestSparsePipeline:
    def test_tfidf_to_multinomial_nb(self):
        """scipy.sparse -> TfidfTransformer -> MultinomialNB -> predict."""
        rng = np.random.RandomState(42)
        X_counts = rng.poisson(lam=2, size=(60, 25)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_counts)
        y = rng.choice([0.0, 1.0], size=60)

        # Fit TF-IDF on sparse
        tfidf = TfidfTransformer()
        tfidf.fit_sparse(X_sparse)
        X_tfidf = tfidf.transform_sparse(X_sparse)

        # Fit MultinomialNB on the dense TF-IDF output
        nb = MultinomialNB()
        nb.fit(X_tfidf, y)
        preds = nb.predict(X_tfidf)
        assert preds.shape == (60,)
        assert set(np.unique(preds)).issubset({0.0, 1.0})


# =============================================================================
# Edge cases
# =============================================================================


class TestSparseEdgeCases:
    def test_csc_auto_conversion(self):
        """CSC input auto-converts to CSR internally."""
        X_sparse, _, y = _make_classification_data()
        X_csc = X_sparse.tocsc()

        model = LogisticRegression(max_iter=200)
        model.fit_sparse(X_csc, y)
        preds = model.predict_sparse(X_csc)
        assert preds.shape == (X_sparse.shape[0],)

    def test_coo_auto_conversion(self):
        """COO input auto-converts to CSR internally."""
        X_sparse, _, y = _make_classification_data()
        X_coo = X_sparse.tocoo()

        model = LogisticRegression(max_iter=200)
        model.fit_sparse(X_coo, y)
        preds = model.predict_sparse(X_coo)
        assert preds.shape == (X_sparse.shape[0],)

    def test_empty_sparse_rows(self):
        """Sparse matrix with all-zero rows works correctly."""
        X_dense = np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],  # all-zero row
                [3.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],  # all-zero row
                [2.0, 1.0, 0.0],
            ]
        )
        X_sparse = sp.csr_matrix(X_dense)
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])

        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (5,)

    def test_single_element_sparse(self):
        """1x1 sparse matrix edge case (for regression)."""
        X_sparse = sp.csr_matrix(np.array([[5.0]]))
        y = np.array([3.0])

        model = RidgeRegression(alpha=0.01)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (1,)

    def test_float32_data_auto_cast(self):
        """float32 sparse data is auto-cast to float64."""
        X_dense = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 1.0], [0.0, 2.0, 0.0]]).astype(
            np.float32
        )
        X_sparse = sp.csr_matrix(X_dense)
        y = np.array([0.0, 1.0, 0.0])

        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (3,)

    def test_int32_indices(self):
        """Sparse matrices with int32 indices work correctly."""
        X_dense = np.array(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
        )
        X_sparse = sp.csr_matrix(X_dense)
        # scipy uses int32 indices by default on most platforms
        assert X_sparse.indices.dtype in (np.int32, np.int64)
        y = np.array([0.0, 1.0, 0.0])

        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (3,)

    def test_highly_sparse_data(self):
        """Very sparse matrix (>99% zeros) works correctly."""
        rng = np.random.RandomState(42)
        X_dense = np.zeros((100, 500))
        # Only 50 non-zero entries out of 50000
        for _ in range(50):
            i = rng.randint(0, 100)
            j = rng.randint(0, 500)
            X_dense[i, j] = rng.poisson(3) + 1.0

        X_sparse = sp.csr_matrix(X_dense)
        y = rng.choice([0.0, 1.0], size=100)

        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        assert preds.shape == (100,)
        assert X_sparse.nnz <= 50  # Verify sparsity
