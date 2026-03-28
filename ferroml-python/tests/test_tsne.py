"""Tests for t-SNE Python bindings."""

import numpy as np
import pytest

from ferroml.decomposition import TSNE


def make_clusters(n_per_cluster=30, n_clusters=3, dim=10, seed=42):
    """Generate well-separated Gaussian clusters."""
    rng = np.random.default_rng(seed)
    data = []
    labels = []
    for c in range(n_clusters):
        center = c * 10.0
        cluster = rng.standard_normal((n_per_cluster, dim)) + center
        data.append(cluster)
        labels.extend([c] * n_per_cluster)
    return np.vstack(data), np.array(labels)


class TestTSNEBasic:
    """Basic t-SNE functionality tests."""

    def test_fit_transform_shape(self):
        X, _ = make_clusters()
        tsne = TSNE(n_components=2, perplexity=10.0, max_iter=200, random_state=42)
        result = tsne.fit_transform(X)
        assert result.shape == (90, 2)

    def test_fit_transform_3d(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2, dim=8)
        tsne = TSNE(n_components=3, perplexity=5.0, max_iter=200, random_state=42)
        result = tsne.fit_transform(X)
        assert result.shape == (30, 3)

    def test_fit_then_transform(self):
        X, _ = make_clusters(n_per_cluster=20, n_clusters=2)
        tsne = TSNE(perplexity=5.0, max_iter=200, random_state=42)
        tsne.fit(X)
        result = tsne.transform(X)
        assert result.shape == (40, 2)

    def test_embedding_attribute(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne = TSNE(perplexity=5.0, max_iter=100, random_state=42)
        tsne.fit(X)
        embedding = tsne.embedding_
        assert embedding.shape == (30, 2)

    def test_kl_divergence_attribute(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne = TSNE(perplexity=5.0, max_iter=100, random_state=42)
        tsne.fit_transform(X)
        kl = tsne.kl_divergence_
        assert isinstance(kl, float)
        assert kl >= 0.0
        assert np.isfinite(kl)

    def test_n_iter_attribute(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne = TSNE(perplexity=5.0, max_iter=100, random_state=42)
        tsne.fit_transform(X)
        n_iter = tsne.n_iter_
        assert isinstance(n_iter, int)
        assert 0 < n_iter <= 100


class TestTSNEClusters:
    """Test that t-SNE separates clusters."""

    def test_separates_well_separated_clusters(self):
        X, labels = make_clusters()
        tsne = TSNE(perplexity=15.0, max_iter=500, random_state=42)
        embedding = tsne.fit_transform(X)

        # Compute intra vs inter cluster distance ratio
        intra_dists = []
        inter_dists = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                d = np.linalg.norm(embedding[i] - embedding[j])
                if labels[i] == labels[j]:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)

        ratio = np.mean(intra_dists) / np.mean(inter_dists)
        assert ratio < 0.5, f"Cluster separation ratio {ratio} should be < 0.5"


class TestTSNEParameters:
    """Test parameter configurations."""

    def test_reproducibility(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne1 = TSNE(perplexity=5.0, max_iter=200, random_state=123)
        result1 = tsne1.fit_transform(X)

        tsne2 = TSNE(perplexity=5.0, max_iter=200, random_state=123)
        result2 = tsne2.fit_transform(X)

        np.testing.assert_allclose(result1, result2, atol=1e-10)

    def test_perplexity_affects_result(self):
        X, _ = make_clusters(n_per_cluster=20, n_clusters=3)
        tsne1 = TSNE(perplexity=5.0, max_iter=300, random_state=42)
        result1 = tsne1.fit_transform(X)

        tsne2 = TSNE(perplexity=25.0, max_iter=300, random_state=42)
        result2 = tsne2.fit_transform(X)

        assert np.max(np.abs(result1 - result2)) > 1e-3

    def test_manhattan_metric(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne = TSNE(metric="manhattan", perplexity=5.0, max_iter=200, random_state=42)
        result = tsne.fit_transform(X)
        assert result.shape == (30, 2)

    def test_cosine_metric(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne = TSNE(metric="cosine", perplexity=5.0, max_iter=200, random_state=42)
        result = tsne.fit_transform(X)
        assert result.shape == (30, 2)

    def test_random_init(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne = TSNE(init="random", perplexity=5.0, max_iter=200, random_state=42)
        result = tsne.fit_transform(X)
        assert result.shape == (30, 2)

    def test_fixed_learning_rate(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne = TSNE(learning_rate=200.0, perplexity=5.0, max_iter=200, random_state=42)
        result = tsne.fit_transform(X)
        assert result.shape == (30, 2)

    def test_early_exaggeration(self):
        X, _ = make_clusters(n_per_cluster=15, n_clusters=2)
        tsne = TSNE(early_exaggeration=4.0, perplexity=5.0, max_iter=200, random_state=42)
        result = tsne.fit_transform(X)
        assert result.shape == (30, 2)


class TestTSNEErrors:
    """Test error handling."""

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            TSNE(metric="invalid")

    def test_invalid_init(self):
        with pytest.raises(ValueError, match="Unknown init"):
            TSNE(init="invalid")

    def test_perplexity_too_large(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tsne = TSNE(perplexity=10.0)
        with pytest.raises(ValueError):
            tsne.fit_transform(X)

    def test_too_few_samples(self):
        X = np.array([[1.0, 2.0]])
        tsne = TSNE()
        with pytest.raises(ValueError):
            tsne.fit_transform(X)


class TestTSNERepr:
    """Test string representation."""

    def test_repr(self):
        tsne = TSNE()
        assert repr(tsne) == "TSNE()"
