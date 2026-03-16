"""
Sparse Pipeline End-to-End Tests (Phase X.5)

Tests verifying sparse data flows through pipelines correctly:
1. CountVectorizer -> TfidfTransformer -> sparse models (MultinomialNB, BernoulliNB, LogisticRegression)
2. scipy.sparse.random() -> fit_sparse -> predict_sparse for all sparse-capable models
3. Output shape validation and NaN checks
"""

import numpy as np
import pytest
import scipy.sparse as sp

from ferroml.linear import LogisticRegression, RidgeRegression
from ferroml.naive_bayes import BernoulliNB, MultinomialNB
from ferroml.preprocessing import CountVectorizer, TfidfTransformer
from ferroml.svm import LinearSVC, LinearSVR


# =============================================================================
# Helpers
# =============================================================================


def _make_classification_sparse(n_samples=60, n_features=20, density=0.3, seed=42):
    """Create sparse classification data with scipy.sparse."""
    rng = np.random.RandomState(seed)
    X_dense = rng.randn(n_samples, n_features)
    mask = rng.rand(n_samples, n_features) > density
    X_dense[mask] = 0.0
    X_sparse = sp.csr_matrix(X_dense)
    w = rng.randn(n_features)
    y = (X_dense @ w > 0).astype(np.float64)
    return X_sparse, X_dense, y


def _make_regression_sparse(n_samples=60, n_features=20, density=0.3, seed=42):
    """Create sparse regression data with scipy.sparse."""
    rng = np.random.RandomState(seed)
    X_dense = rng.randn(n_samples, n_features)
    mask = rng.rand(n_samples, n_features) > density
    X_dense[mask] = 0.0
    X_sparse = sp.csr_matrix(X_dense)
    w = rng.randn(n_features)
    y = X_dense @ w + rng.randn(n_samples) * 0.1
    return X_sparse, X_dense, y


def _make_count_sparse(n_samples=60, n_features=20, density=0.3, seed=42):
    """Create sparse non-negative count data for NB models."""
    rng = np.random.RandomState(seed)
    X_dense = rng.poisson(lam=2, size=(n_samples, n_features)).astype(np.float64)
    mask = rng.rand(n_samples, n_features) > density
    X_dense[mask] = 0.0
    X_sparse = sp.csr_matrix(X_dense)
    y = rng.choice([0.0, 1.0], size=n_samples)
    return X_sparse, X_dense, y


def _make_binary_sparse(n_samples=60, n_features=20, density=0.4, seed=42):
    """Create sparse binary data for BernoulliNB."""
    rng = np.random.RandomState(seed)
    X_dense = (rng.rand(n_samples, n_features) > (1.0 - density)).astype(np.float64)
    X_sparse = sp.csr_matrix(X_dense)
    y = rng.choice([0.0, 1.0], size=n_samples)
    return X_sparse, X_dense, y


def _sample_corpus():
    """Text corpus for CountVectorizer tests."""
    return [
        "the cat sat on the mat",
        "the dog chased the cat",
        "a bird flew over the house",
        "the fish swam in the pond",
        "the cat and dog played together",
        "a small bird landed on the mat",
        "the dog ran after the bird",
        "fish and chips for dinner",
    ]


def _assert_valid_predictions(preds, n_samples, name=""):
    """Assert predictions have correct shape and no NaN/Inf values."""
    assert preds.shape == (n_samples,), f"{name}: expected shape ({n_samples},), got {preds.shape}"
    assert not np.any(np.isnan(preds)), f"{name}: predictions contain NaN"
    assert not np.any(np.isinf(preds)), f"{name}: predictions contain Inf"


def _assert_valid_classification(preds, n_samples, name=""):
    """Assert classification predictions are valid labels."""
    _assert_valid_predictions(preds, n_samples, name)
    unique_labels = set(np.unique(preds))
    assert unique_labels.issubset({0.0, 1.0}), (
        f"{name}: unexpected labels {unique_labels}"
    )


# =============================================================================
# 1. CountVectorizer -> TfidfTransformer -> Sparse Model Pipelines
# =============================================================================


class TestTextPipelineMultinomialNB:
    """CountVectorizer -> TfidfTransformer -> MultinomialNB."""

    def test_full_pipeline(self):
        corpus = _sample_corpus()
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

        cv = CountVectorizer()
        X_counts = cv.fit_transform(corpus)

        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X_counts)

        nb = MultinomialNB()
        nb.fit(X_tfidf, y)
        preds = nb.predict(X_tfidf)

        _assert_valid_classification(preds, len(corpus), "MultinomialNB pipeline")

    def test_sparse_tfidf_to_multinomial_nb(self):
        """TfidfTransformer sparse path -> MultinomialNB fit on dense result."""
        rng = np.random.RandomState(42)
        X_counts = rng.poisson(lam=2, size=(50, 25)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_counts)
        y = rng.choice([0.0, 1.0], size=50)

        tfidf = TfidfTransformer()
        tfidf.fit_sparse(X_sparse)
        X_tfidf = tfidf.transform_sparse(X_sparse)

        nb = MultinomialNB()
        nb.fit(X_tfidf, y)
        preds = nb.predict(X_tfidf)

        _assert_valid_classification(preds, 50, "sparse tfidf -> MNB")


class TestTextPipelineBernoulliNB:
    """CountVectorizer -> TfidfTransformer -> BernoulliNB."""

    def test_full_pipeline(self):
        corpus = _sample_corpus()
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

        cv = CountVectorizer()
        X_counts = cv.fit_transform(corpus)

        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X_counts)

        model = BernoulliNB()
        model.fit(X_tfidf, y)
        preds = model.predict(X_tfidf)

        _assert_valid_classification(preds, len(corpus), "BernoulliNB pipeline")


class TestTextPipelineLogisticRegression:
    """CountVectorizer -> TfidfTransformer -> LogisticRegression."""

    def test_full_pipeline(self):
        corpus = _sample_corpus()
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

        cv = CountVectorizer()
        X_counts = cv.fit_transform(corpus)

        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(X_counts)

        model = LogisticRegression(max_iter=200)
        model.fit(X_tfidf, y)
        preds = model.predict(X_tfidf)

        _assert_valid_classification(preds, len(corpus), "LogisticRegression pipeline")

    def test_sparse_fit_predict(self):
        """CountVectorizer -> TfidfTransformer(sparse) -> LogisticRegression(fit_sparse)."""
        corpus = _sample_corpus()
        y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

        cv = CountVectorizer()
        X_counts = cv.fit_transform(corpus)
        X_sparse = sp.csr_matrix(X_counts)

        tfidf = TfidfTransformer()
        tfidf.fit_sparse(X_sparse)
        X_tfidf = tfidf.transform_sparse(X_sparse)
        X_tfidf_sparse = sp.csr_matrix(X_tfidf)

        model = LogisticRegression(max_iter=200)
        model.fit_sparse(X_tfidf_sparse, y)
        preds = model.predict_sparse(X_tfidf_sparse)

        _assert_valid_classification(preds, len(corpus), "LogisticRegression sparse pipeline")


# =============================================================================
# 2. scipy.sparse -> fit_sparse -> predict_sparse for All Sparse Models
# =============================================================================


class TestSparseClassifiers:
    """fit_sparse / predict_sparse for all sparse-capable classifiers."""

    def test_logistic_regression(self):
        X_sparse, _, y = _make_classification_sparse(seed=10)
        model = LogisticRegression(max_iter=200)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_classification(preds, X_sparse.shape[0], "LogisticRegression")

    def test_multinomial_nb(self):
        X_sparse, _, y = _make_count_sparse(seed=20)
        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_classification(preds, X_sparse.shape[0], "MultinomialNB")

    def test_bernoulli_nb(self):
        X_sparse, _, y = _make_binary_sparse(seed=30)
        model = BernoulliNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_classification(preds, X_sparse.shape[0], "BernoulliNB")

    def test_linear_svc(self):
        X_sparse, _, y = _make_classification_sparse(seed=40)
        model = LinearSVC(max_iter=2000)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_classification(preds, X_sparse.shape[0], "LinearSVC")


class TestSparseRegressors:
    """fit_sparse / predict_sparse for all sparse-capable regressors."""

    def test_ridge_regression(self):
        X_sparse, _, y = _make_regression_sparse(seed=50)
        model = RidgeRegression(alpha=1.0)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_predictions(preds, X_sparse.shape[0], "RidgeRegression")

    def test_linear_svr(self):
        X_sparse, _, y = _make_regression_sparse(seed=60)
        model = LinearSVR(max_iter=2000)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_predictions(preds, X_sparse.shape[0], "LinearSVR")


# =============================================================================
# 3. Sparse vs Dense Equivalence
# =============================================================================


class TestSparseVsDenseEquivalence:
    """Verify sparse and dense paths produce matching predictions."""

    def test_logistic_regression_match(self):
        X_sparse, X_dense, y = _make_classification_sparse(seed=100)

        m_dense = LogisticRegression(max_iter=200)
        m_dense.fit(X_dense, y)
        p_dense = m_dense.predict(X_dense)

        m_sparse = LogisticRegression(max_iter=200)
        m_sparse.fit_sparse(X_sparse, y)
        p_sparse = m_sparse.predict_sparse(X_sparse)

        np.testing.assert_array_equal(p_dense, p_sparse)

    def test_multinomial_nb_match(self):
        X_sparse, X_dense, y = _make_count_sparse(seed=101)

        m_dense = MultinomialNB()
        m_dense.fit(X_dense, y)
        p_dense = m_dense.predict(X_dense)

        m_sparse = MultinomialNB()
        m_sparse.fit_sparse(X_sparse, y)
        p_sparse = m_sparse.predict_sparse(X_sparse)

        np.testing.assert_array_equal(p_dense, p_sparse)

    def test_bernoulli_nb_match(self):
        X_sparse, X_dense, y = _make_binary_sparse(seed=102)

        m_dense = BernoulliNB()
        m_dense.fit(X_dense, y)
        p_dense = m_dense.predict(X_dense)

        m_sparse = BernoulliNB()
        m_sparse.fit_sparse(X_sparse, y)
        p_sparse = m_sparse.predict_sparse(X_sparse)

        np.testing.assert_array_equal(p_dense, p_sparse)

    def test_ridge_regression_match(self):
        X_sparse, X_dense, y = _make_regression_sparse(seed=103)

        m_dense = RidgeRegression(alpha=1.0)
        m_dense.fit(X_dense, y)
        p_dense = m_dense.predict(X_dense)

        m_sparse = RidgeRegression(alpha=1.0)
        m_sparse.fit_sparse(X_sparse, y)
        p_sparse = m_sparse.predict_sparse(X_sparse)

        np.testing.assert_allclose(p_dense, p_sparse, atol=1e-6)

    def test_linear_svc_match(self):
        X_sparse, X_dense, y = _make_classification_sparse(seed=104)

        m_dense = LinearSVC(max_iter=2000)
        m_dense.fit(X_dense, y)
        p_dense = m_dense.predict(X_dense)

        m_sparse = LinearSVC(max_iter=2000)
        m_sparse.fit_sparse(X_sparse, y)
        p_sparse = m_sparse.predict_sparse(X_sparse)

        np.testing.assert_array_equal(p_dense, p_sparse)

    def test_linear_svr_match(self):
        X_sparse, X_dense, y = _make_regression_sparse(seed=105)

        m_dense = LinearSVR(max_iter=2000)
        m_dense.fit(X_dense, y)
        p_dense = m_dense.predict(X_dense)

        m_sparse = LinearSVR(max_iter=2000)
        m_sparse.fit_sparse(X_sparse, y)
        p_sparse = m_sparse.predict_sparse(X_sparse)

        np.testing.assert_allclose(p_dense, p_sparse, atol=1e-6)


# =============================================================================
# 4. TfidfTransformer Sparse Path Validation
# =============================================================================


class TestTfidfSparseEquivalence:
    """Verify TfidfTransformer sparse and dense paths produce identical results."""

    def test_fit_transform_sparse_vs_dense(self):
        rng = np.random.RandomState(42)
        X_dense = rng.poisson(lam=3, size=(30, 15)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)

        tfidf_dense = TfidfTransformer()
        result_dense = tfidf_dense.fit_transform(X_dense)

        tfidf_sparse = TfidfTransformer()
        tfidf_sparse.fit_sparse(X_sparse)
        result_sparse = tfidf_sparse.transform_sparse(X_sparse)

        np.testing.assert_allclose(result_dense, result_sparse, atol=1e-10)

    def test_fit_transform_sparse_single_step(self):
        rng = np.random.RandomState(42)
        X_dense = rng.poisson(lam=3, size=(30, 15)).astype(np.float64)
        X_sparse = sp.csr_matrix(X_dense)

        tfidf = TfidfTransformer()
        result = tfidf.fit_transform_sparse(X_sparse)

        assert result.shape == (30, 15)
        assert not np.any(np.isnan(result))
        assert np.all(result >= 0)


# =============================================================================
# 5. Output Shape and NaN Checks for All Models
# =============================================================================


class TestSparseOutputValidation:
    """Validate output shapes and absence of NaN across all sparse models."""

    @pytest.mark.parametrize(
        "model_fn,data_fn",
        [
            (lambda: MultinomialNB(), lambda: _make_count_sparse(seed=200)),
            (lambda: BernoulliNB(), lambda: _make_binary_sparse(seed=201)),
            (lambda: LogisticRegression(max_iter=200), lambda: _make_classification_sparse(seed=202)),
            (lambda: LinearSVC(max_iter=2000), lambda: _make_classification_sparse(seed=203)),
        ],
        ids=["MultinomialNB", "BernoulliNB", "LogisticRegression", "LinearSVC"],
    )
    def test_classifier_output(self, model_fn, data_fn):
        X_sparse, _, y = data_fn()
        model = model_fn()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_classification(preds, X_sparse.shape[0])

    @pytest.mark.parametrize(
        "model_fn,data_fn",
        [
            (lambda: RidgeRegression(alpha=1.0), lambda: _make_regression_sparse(seed=210)),
            (lambda: LinearSVR(max_iter=2000), lambda: _make_regression_sparse(seed=211)),
        ],
        ids=["RidgeRegression", "LinearSVR"],
    )
    def test_regressor_output(self, model_fn, data_fn):
        X_sparse, _, y = data_fn()
        model = model_fn()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_predictions(preds, X_sparse.shape[0])


# =============================================================================
# 6. Edge Cases
# =============================================================================


class TestSparseEdgeCases:
    """Edge cases for sparse pipeline workflows."""

    def test_highly_sparse_data(self):
        """99%+ sparse matrix works correctly."""
        rng = np.random.RandomState(42)
        X_dense = np.zeros((80, 200))
        for _ in range(40):
            i = rng.randint(0, 80)
            j = rng.randint(0, 200)
            X_dense[i, j] = rng.poisson(3) + 1.0
        X_sparse = sp.csr_matrix(X_dense)
        y = rng.choice([0.0, 1.0], size=80)

        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_classification(preds, 80, "highly sparse")

    def test_csc_input_auto_conversion(self):
        """CSC matrices are accepted and converted to CSR internally."""
        X_sparse, _, y = _make_classification_sparse(seed=300)
        X_csc = X_sparse.tocsc()

        model = LogisticRegression(max_iter=200)
        model.fit_sparse(X_csc, y)
        preds = model.predict_sparse(X_csc)
        _assert_valid_classification(preds, X_sparse.shape[0], "CSC input")

    def test_coo_input_auto_conversion(self):
        """COO matrices are accepted and converted to CSR internally."""
        X_sparse, _, y = _make_classification_sparse(seed=301)
        X_coo = X_sparse.tocoo()

        model = LogisticRegression(max_iter=200)
        model.fit_sparse(X_coo, y)
        preds = model.predict_sparse(X_coo)
        _assert_valid_classification(preds, X_sparse.shape[0], "COO input")

    def test_all_zero_rows(self):
        """Sparse matrix with all-zero rows works correctly."""
        X_dense = np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        X_sparse = sp.csr_matrix(X_dense)
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])

        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_predictions(preds, 5, "all-zero rows")

    def test_single_sample(self):
        """1-sample sparse matrix edge case."""
        X_sparse = sp.csr_matrix(np.array([[5.0, 0.0, 3.0]]))
        y = np.array([1.0])

        model = RidgeRegression(alpha=0.01)
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_predictions(preds, 1, "single sample")

    def test_float32_auto_cast(self):
        """float32 sparse data is auto-cast to float64."""
        X_dense = np.array(
            [[1.0, 0.0, 2.0], [3.0, 0.0, 1.0], [0.0, 2.0, 0.0]]
        ).astype(np.float32)
        X_sparse = sp.csr_matrix(X_dense)
        y = np.array([0.0, 1.0, 0.0])

        model = MultinomialNB()
        model.fit_sparse(X_sparse, y)
        preds = model.predict_sparse(X_sparse)
        _assert_valid_predictions(preds, 3, "float32 auto-cast")


# =============================================================================
# 7. Multi-Model Pipeline Comparison
# =============================================================================


class TestMultiModelSparsePipeline:
    """Run the same sparse data through multiple models and validate all outputs."""

    def test_all_classifiers_on_same_data(self):
        """All sparse classifiers produce valid outputs on the same data."""
        X_sparse, X_dense, y = _make_count_sparse(n_samples=80, n_features=15, seed=500)

        models = {
            "MultinomialNB": MultinomialNB(),
            "BernoulliNB": BernoulliNB(),
            "LogisticRegression": LogisticRegression(max_iter=300),
            "LinearSVC": LinearSVC(max_iter=2000),
        }

        for name, model in models.items():
            model.fit_sparse(X_sparse, y)
            preds = model.predict_sparse(X_sparse)
            _assert_valid_classification(preds, 80, name)

    def test_all_regressors_on_same_data(self):
        """All sparse regressors produce valid outputs on the same data."""
        X_sparse, X_dense, y = _make_regression_sparse(n_samples=80, n_features=15, seed=501)

        models = {
            "RidgeRegression": RidgeRegression(alpha=1.0),
            "LinearSVR": LinearSVR(max_iter=2000),
        }

        for name, model in models.items():
            model.fit_sparse(X_sparse, y)
            preds = model.predict_sparse(X_sparse)
            _assert_valid_predictions(preds, 80, name)
