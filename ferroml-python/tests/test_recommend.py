"""Tests for the ferroml.recommend() API."""

import time

import numpy as np
import pytest

import ferroml


class TestRecommendClassification:
    """Tests for classification recommendations."""

    def test_recommend_classification_basic(self):
        """Basic classification recommendation returns results."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = (X[:, 0] > 0).astype(np.float64)
        recs = ferroml.recommend(X, y, task="classification")
        assert len(recs) > 0
        assert len(recs) <= 5
        # Scores should be descending
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_large_dataset_prefers_scalable(self):
        """Large datasets should recommend scalable algorithms."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100_000, 20))
        y = (X[:, 0] > 0).astype(np.float64)
        recs = ferroml.recommend(X, y, task="classification")
        top_algo = recs[0].algorithm
        scalable = {"HistGradientBoostingClassifier", "RandomForestClassifier", "SGDClassifier"}
        assert top_algo in scalable, f"Expected scalable algorithm for large data, got {top_algo}"

    def test_recommend_sparse_data(self):
        """Sparse data should recommend sparse-friendly algorithms."""
        rng = np.random.default_rng(42)
        X = np.zeros((200, 50))
        for i in range(200):
            X[i, i % 50] = 1.0
        y = (np.arange(200) % 2).astype(np.float64)
        recs = ferroml.recommend(X, y, task="classification")
        algos = [r.algorithm for r in recs]
        sparse_friendly = {"LogisticRegression", "SGDClassifier", "GaussianNB"}
        assert any(a in sparse_friendly for a in algos), (
            f"Expected sparse-friendly algorithm, got {algos}"
        )

    def test_recommend_imbalanced_classes(self):
        """Imbalanced classes should trigger class_weight=balanced."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = np.array([0.0] * 180 + [1.0] * 20)
        recs = ferroml.recommend(X, y, task="classification")
        rf = [r for r in recs if r.algorithm == "RandomForestClassifier"]
        assert len(rf) > 0, "RandomForest should be recommended for imbalanced data"
        assert "class_weight" in rf[0].params, (
            "RandomForest should suggest class_weight=balanced for imbalanced data"
        )

    def test_recommend_many_features(self):
        """When features > samples, L1-regularized models should be recommended."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 200))
        y = (X[:, 0] > 0).astype(np.float64)
        recs = ferroml.recommend(X, y, task="classification")
        algos = [r.algorithm for r in recs]
        assert "LogisticRegression" in algos, (
            f"LogisticRegression should be recommended for many features, got {algos}"
        )
        lr = [r for r in recs if r.algorithm == "LogisticRegression"][0]
        assert lr.params.get("penalty") == "l1", (
            "LogisticRegression should use L1 penalty when features > samples"
        )


class TestRecommendRegression:
    """Tests for regression recommendations."""

    def test_recommend_regression_basic(self):
        """Basic regression recommendation returns results."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.standard_normal(200) * 0.1
        recs = ferroml.recommend(X, y, task="regression")
        assert len(recs) > 0
        assert len(recs) <= 5
        # Should not contain classifiers
        for r in recs:
            assert "Classifier" not in r.algorithm, (
                f"Regression should not recommend classifiers: {r.algorithm}"
            )


class TestRecommendValidation:
    """Tests for input validation and edge cases."""

    def test_recommend_returns_top_5(self):
        """Should return at most 5 recommendations."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 10))
        y = (X[:, 0] > 0).astype(np.float64)
        recs = ferroml.recommend(X, y, task="classification")
        assert len(recs) <= 5

    def test_recommend_invalid_task(self):
        """Invalid task should raise ValueError."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = (X[:, 0] > 0).astype(np.float64)
        with pytest.raises(ValueError, match="Unknown task"):
            ferroml.recommend(X, y, task="clustering")

    def test_recommend_recommendation_attributes(self):
        """All recommendation attributes should be populated correctly."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = (X[:, 0] > 0).astype(np.float64)
        recs = ferroml.recommend(X, y, task="classification")
        r = recs[0]
        # Check all attributes exist and have correct types
        assert isinstance(r.algorithm, str) and len(r.algorithm) > 0
        assert isinstance(r.reason, str) and len(r.reason) > 0
        assert r.estimated_fit_time in ("fast", "moderate", "slow")
        assert isinstance(r.params, dict)
        assert isinstance(r.score, float)
        assert 0.0 < r.score <= 1.0

    def test_recommend_fast_execution(self):
        """Recommendation should complete in < 100ms (no fitting)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10_000, 50))
        y = (X[:, 0] > 0).astype(np.float64)
        start = time.time()
        ferroml.recommend(X, y, task="classification")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 100, f"recommend() took {elapsed_ms:.1f}ms, expected < 100ms"

    def test_recommend_repr(self):
        """Recommendation __repr__ should be informative."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = (X[:, 0] > 0).astype(np.float64)
        recs = ferroml.recommend(X, y, task="classification")
        r = repr(recs[0])
        assert "Recommendation" in r
        assert "algorithm=" in r
        assert "score=" in r
