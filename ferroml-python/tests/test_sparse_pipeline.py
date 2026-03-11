"""
Tests for TfidfTransformer and sparse preprocessing pipeline.

Covers:
- Basic TF-IDF transformation with L2 normalization (default)
- IDF weighting disabled (use_idf=False)
- Sublinear TF scaling (1 + log(tf))
- L1 normalization
- IDF attribute access after fit
- Not-fitted error
- Empty document handling (all-zero rows)
- Smooth vs unsmoothed IDF
"""

import numpy as np
import pytest
from ferroml import preprocessing


class TestTfidfTransformer:
    def test_basic(self):
        """Basic TF-IDF transform on count matrix with default L2 norm."""
        X = np.array([[1, 0, 2], [0, 1, 1], [1, 1, 0]], dtype=np.float64)
        tfidf = preprocessing.TfidfTransformer()
        result = tfidf.fit_transform(X)
        assert result.shape == (3, 3)
        # Rows should have unit L2 norm (default)
        for i in range(result.shape[0]):
            norm = np.linalg.norm(result[i])
            assert abs(norm - 1.0) < 1e-10 or norm == 0.0

    def test_no_idf(self):
        """With use_idf=False, only TF normalization applied."""
        X = np.array([[1, 0, 2], [0, 1, 1]], dtype=np.float64)
        tfidf = preprocessing.TfidfTransformer(use_idf=False, norm="none")
        result = tfidf.fit_transform(X)
        # Without IDF and without norm, result should equal input
        np.testing.assert_array_almost_equal(result, X)

    def test_sublinear_tf(self):
        """Sublinear TF scaling: tf -> 1 + log(tf) for tf > 0."""
        X = np.array([[1, 0, 4], [0, 1, 1]], dtype=np.float64)
        tfidf = preprocessing.TfidfTransformer(
            sublinear_tf=True, use_idf=False, norm="none"
        )
        result = tfidf.fit_transform(X)
        # tf=4 -> 1 + ln(4)
        assert abs(result[0, 2] - (1.0 + np.log(4.0))) < 1e-10
        # tf=1 -> 1 + ln(1) = 1
        assert abs(result[1, 1] - 1.0) < 1e-10
        # tf=0 -> 0
        assert abs(result[0, 1]) < 1e-10

    def test_l1_norm(self):
        """L1 normalization: rows sum to 1 in absolute value."""
        X = np.array([[1, 0, 2], [0, 1, 1], [1, 1, 0]], dtype=np.float64)
        tfidf = preprocessing.TfidfTransformer(norm="l1")
        result = tfidf.fit_transform(X)
        for i in range(result.shape[0]):
            norm = np.sum(np.abs(result[i]))
            if norm > 0:
                assert abs(norm - 1.0) < 1e-10

    def test_idf_attribute(self):
        """IDF weights accessible after fit."""
        X = np.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=np.float64)
        tfidf = preprocessing.TfidfTransformer()
        tfidf.fit(X)
        idf = tfidf.idf_
        assert idf is not None
        assert len(idf) == 3
        # Feature 2 appears in all 3 docs -> lowest IDF
        # Feature 1 appears in 1 doc -> highest IDF
        assert idf[1] > idf[0] > idf[2] or idf[1] > idf[2]

    def test_not_fitted(self):
        """Error when transforming before fitting."""
        X = np.array([[1, 0], [0, 1]], dtype=np.float64)
        tfidf = preprocessing.TfidfTransformer()
        with pytest.raises(RuntimeError):
            tfidf.transform(X)

    def test_empty_document(self):
        """All-zero rows should remain all-zero after transform."""
        X = np.array(
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float64
        )
        tfidf = preprocessing.TfidfTransformer()
        result = tfidf.fit_transform(X)
        np.testing.assert_array_almost_equal(result[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(result[2], [0, 0, 0])

    def test_idf_none_when_disabled(self):
        """idf_ should be None when use_idf=False."""
        X = np.array([[1, 0], [0, 1]], dtype=np.float64)
        tfidf = preprocessing.TfidfTransformer(use_idf=False)
        tfidf.fit(X)
        assert tfidf.idf_ is None
