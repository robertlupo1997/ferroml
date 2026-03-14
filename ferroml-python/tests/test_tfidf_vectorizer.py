"""Tests for TfidfVectorizer Python bindings."""

import pytest
import numpy as np
from scipy import sparse

from ferroml.preprocessing import TfidfVectorizer


def sample_corpus():
    return [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
        "a bird flew over the house",
    ]


class TestTfidfVectorizerBasic:
    """Basic fit/transform tests."""

    def test_fit_transform_default(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        X = tv.fit_transform(corpus)
        assert X.shape[0] == 4
        assert X.shape[1] > 0

    def test_output_is_sparse_csr(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        X = tv.fit_transform(corpus)
        assert sparse.issparse(X)
        assert isinstance(X, sparse.csr_matrix)

    def test_transform_dense(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        tv.fit(corpus)
        X = tv.transform_dense(corpus)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 4
        assert X.shape[1] > 0

    def test_fit_returns_self(self):
        """Chaining: fit() returns the vectorizer itself."""
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        result = tv.fit(corpus)
        # Should be able to call transform after fit
        X = result.transform(corpus)
        assert X.shape[0] == 4

    def test_values_are_floats(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        X = tv.fit_transform(corpus)
        assert X.dtype == np.float64

    def test_values_nonnegative(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        X = tv.fit_transform(corpus)
        assert (X.toarray() >= 0).all()


class TestTfidfVectorizerProperties:
    """Tests for vocabulary_, get_feature_names_out, idf_."""

    def test_vocabulary_property(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        tv.fit(corpus)
        vocab = tv.vocabulary_
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        assert "cat" in vocab
        assert "dog" in vocab

    def test_get_feature_names_out(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        tv.fit(corpus)
        names = tv.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) == len(tv.vocabulary_)
        # Feature names should be sorted
        assert names == sorted(names)

    def test_idf_property(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        tv.fit(corpus)
        idf = tv.idf_
        assert isinstance(idf, np.ndarray)
        assert len(idf) == len(tv.vocabulary_)
        # IDF values should be positive
        assert (idf > 0).all()

    def test_idf_none_before_fit(self):
        tv = TfidfVectorizer()
        assert tv.idf_ is None

    def test_vocabulary_columns_match_features(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        X = tv.fit_transform(corpus)
        names = tv.get_feature_names_out()
        assert X.shape[1] == len(names)


class TestTfidfVectorizerParams:
    """Tests for builder parameters."""

    def test_max_features(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer(max_features=3)
        X = tv.fit_transform(corpus)
        assert X.shape[1] == 3

    def test_ngram_range_bigrams(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer(ngram_range=(1, 2))
        X = tv.fit_transform(corpus)
        names = tv.get_feature_names_out()
        # Should have some bigrams
        bigrams = [n for n in names if " " in n]
        assert len(bigrams) > 0

    def test_min_df(self):
        corpus = sample_corpus()
        tv_all = TfidfVectorizer(min_df=1)
        tv_all.fit(corpus)
        tv_min2 = TfidfVectorizer(min_df=2)
        tv_min2.fit(corpus)
        # Higher min_df should give fewer features
        assert len(tv_min2.vocabulary_) <= len(tv_all.vocabulary_)

    def test_max_df(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer(max_df=0.5)
        tv.fit(corpus)
        # "the" appears in all docs, should be filtered out
        assert "the" not in tv.vocabulary_

    def test_stop_words(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer(stop_words=["the", "on", "a"])
        tv.fit(corpus)
        vocab = tv.vocabulary_
        assert "the" not in vocab
        assert "on" not in vocab
        assert "a" not in vocab

    def test_binary(self):
        corpus = ["the the the cat", "dog"]
        tv = TfidfVectorizer(binary=True, use_idf=False, norm="none")
        X = tv.fit_transform(corpus)
        arr = X.toarray()
        # With binary=True and no IDF/norm, counts should be 0 or 1
        unique = np.unique(arr)
        assert set(unique).issubset({0.0, 1.0})

    def test_lowercase_false(self):
        corpus = ["Hello World", "hello world"]
        tv = TfidfVectorizer(lowercase=False)
        tv.fit(corpus)
        vocab = tv.vocabulary_
        # "Hello" and "hello" should be separate features
        assert "Hello" in vocab
        assert "hello" in vocab

    def test_norm_l1(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer(norm="l1")
        X = tv.fit_transform(corpus)
        arr = X.toarray()
        # Each row should sum to ~1.0 (L1 norm)
        for i in range(arr.shape[0]):
            row_sum = np.abs(arr[i]).sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 1e-10

    def test_norm_l2(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer(norm="l2")
        X = tv.fit_transform(corpus)
        arr = X.toarray()
        # Each row should have L2 norm ~1.0
        for i in range(arr.shape[0]):
            row_norm = np.sqrt((arr[i] ** 2).sum())
            if row_norm > 0:
                assert abs(row_norm - 1.0) < 1e-10

    def test_norm_none(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer(norm="none")
        X = tv.fit_transform(corpus)
        arr = X.toarray()
        # Rows should NOT necessarily be unit-norm
        norms = np.sqrt((arr**2).sum(axis=1))
        assert not np.allclose(norms[norms > 0], 1.0)

    def test_use_idf_false(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer(use_idf=False)
        tv.fit(corpus)
        # IDF should be None when use_idf=False
        assert tv.idf_ is None

    def test_sublinear_tf(self):
        corpus = ["word word word word", "other"]
        tv_sub = TfidfVectorizer(sublinear_tf=True, norm="none", use_idf=False)
        X_sub = tv_sub.fit_transform(corpus)
        tv_lin = TfidfVectorizer(sublinear_tf=False, norm="none", use_idf=False)
        X_lin = tv_lin.fit_transform(corpus)
        # With sublinear TF, the value for "word" should be less than linear
        arr_sub = X_sub.toarray()
        arr_lin = X_lin.toarray()
        # Get the column for "word"
        word_idx_sub = tv_sub.vocabulary_["word"]
        word_idx_lin = tv_lin.vocabulary_["word"]
        assert arr_sub[0, word_idx_sub] < arr_lin[0, word_idx_lin]

    def test_invalid_norm_raises(self):
        with pytest.raises(ValueError, match="norm must be"):
            TfidfVectorizer(norm="invalid")


class TestTfidfVectorizerErrors:
    """Tests for error cases."""

    def test_not_fitted_vocabulary(self):
        tv = TfidfVectorizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = tv.vocabulary_

    def test_not_fitted_feature_names(self):
        tv = TfidfVectorizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            tv.get_feature_names_out()

    def test_not_fitted_transform(self):
        tv = TfidfVectorizer()
        with pytest.raises(RuntimeError):
            tv.transform(["hello world"])

    def test_not_fitted_transform_dense(self):
        tv = TfidfVectorizer()
        with pytest.raises(RuntimeError):
            tv.transform_dense(["hello world"])


class TestTfidfVectorizerConsistency:
    """Tests for consistency between fit/transform and fit_transform."""

    def test_fit_transform_equals_fit_then_transform(self):
        corpus = sample_corpus()
        tv1 = TfidfVectorizer()
        X1 = tv1.fit_transform(corpus)

        tv2 = TfidfVectorizer()
        tv2.fit(corpus)
        X2 = tv2.transform(corpus)

        np.testing.assert_allclose(X1.toarray(), X2.toarray(), atol=1e-12)

    def test_dense_matches_sparse(self):
        corpus = sample_corpus()
        tv = TfidfVectorizer()
        tv.fit(corpus)
        X_sparse = tv.transform(corpus)
        X_dense = tv.transform_dense(corpus)
        np.testing.assert_allclose(X_sparse.toarray(), X_dense, atol=1e-12)

    def test_unseen_terms_in_transform(self):
        corpus = ["cat dog"]
        tv = TfidfVectorizer()
        tv.fit(corpus)
        # Transform with unseen terms -- should produce zeros for unseen
        X = tv.transform(["bird fish"])
        arr = X.toarray()
        assert arr.sum() == 0.0

    def test_repr(self):
        tv = TfidfVectorizer()
        assert repr(tv) == "TfidfVectorizer()"
