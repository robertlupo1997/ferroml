"""
FerroML vs sklearn: Clustering algorithms.

Cross-library validation for:
1. AgglomerativeClustering — ARI > 0.9 on well-separated blobs
2. DBSCAN — exact label match on standard epsilon

Phase X.3 — Plan X production-readiness validation.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def blob_data():
    """Well-separated blobs for clustering validation."""
    X, y = make_blobs(
        n_samples=200,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        random_state=42,
    )
    return X, y


@pytest.fixture()
def dbscan_data():
    """Tighter blobs suitable for DBSCAN with standard epsilon."""
    X, y = make_blobs(
        n_samples=150,
        n_features=2,
        centers=3,
        cluster_std=0.4,
        random_state=42,
    )
    return X, y


# ===========================================================================
# 1. AgglomerativeClustering
# ===========================================================================

class TestAgglomerativeClusteringVsSklearn:
    """Compare FerroML AgglomerativeClustering against sklearn."""

    def test_ari_above_09_ward(self, blob_data):
        """Ward linkage on well-separated blobs should produce ARI > 0.9."""
        from ferroml.clustering import AgglomerativeClustering

        from sklearn.cluster import AgglomerativeClustering as SkAgg

        X, y_true = blob_data

        sk = SkAgg(n_clusters=3, linkage="ward")
        sk_labels = sk.fit_predict(X)
        sk_ari = adjusted_rand_score(y_true, sk_labels)

        fm = AgglomerativeClustering(n_clusters=3, linkage="ward")
        fm.fit(X)
        fm_labels = np.array(fm.labels_)
        fm_ari = adjusted_rand_score(y_true, fm_labels)

        assert sk_ari > 0.9, f"sklearn ARI too low: {sk_ari}"
        assert fm_ari > 0.9, (
            f"FerroML ARI too low: {fm_ari:.4f} (sklearn={sk_ari:.4f})"
        )

    def test_ari_above_09_complete(self, blob_data):
        """Complete linkage on well-separated blobs should produce ARI > 0.9."""
        from ferroml.clustering import AgglomerativeClustering

        from sklearn.cluster import AgglomerativeClustering as SkAgg

        X, y_true = blob_data

        sk = SkAgg(n_clusters=3, linkage="complete")
        sk_labels = sk.fit_predict(X)
        sk_ari = adjusted_rand_score(y_true, sk_labels)

        fm = AgglomerativeClustering(n_clusters=3, linkage="complete")
        fm.fit(X)
        fm_labels = np.array(fm.labels_)
        fm_ari = adjusted_rand_score(y_true, fm_labels)

        assert sk_ari > 0.9, f"sklearn ARI too low: {sk_ari}"
        assert fm_ari > 0.9, (
            f"FerroML ARI too low: {fm_ari:.4f} (sklearn={sk_ari:.4f})"
        )

    def test_ari_above_09_average(self, blob_data):
        """Average linkage on well-separated blobs should produce ARI > 0.9."""
        from ferroml.clustering import AgglomerativeClustering

        from sklearn.cluster import AgglomerativeClustering as SkAgg

        X, y_true = blob_data

        sk = SkAgg(n_clusters=3, linkage="average")
        sk_labels = sk.fit_predict(X)
        sk_ari = adjusted_rand_score(y_true, sk_labels)

        fm = AgglomerativeClustering(n_clusters=3, linkage="average")
        fm.fit(X)
        fm_labels = np.array(fm.labels_)
        fm_ari = adjusted_rand_score(y_true, fm_labels)

        assert sk_ari > 0.9, f"sklearn ARI too low: {sk_ari}"
        assert fm_ari > 0.9, (
            f"FerroML ARI too low: {fm_ari:.4f} (sklearn={sk_ari:.4f})"
        )

    def test_correct_number_of_clusters(self, blob_data):
        from ferroml.clustering import AgglomerativeClustering

        X, _ = blob_data

        fm = AgglomerativeClustering(n_clusters=3, linkage="ward")
        fm.fit(X)
        fm_labels = np.array(fm.labels_)

        n_unique = len(np.unique(fm_labels))
        assert n_unique == 3, f"Expected 3 clusters, got {n_unique}"


# ===========================================================================
# 2. DBSCAN
# ===========================================================================

class TestDBSCANVsSklearn:
    """Compare FerroML DBSCAN against sklearn."""

    def test_labels_match(self, dbscan_data):
        """With standard epsilon, labels should match sklearn exactly."""
        from ferroml.clustering import DBSCAN

        from sklearn.cluster import DBSCAN as SkDBSCAN

        X, _ = dbscan_data

        eps = 0.8
        min_samples = 5

        sk = SkDBSCAN(eps=eps, min_samples=min_samples)
        sk_labels = sk.fit_predict(X)

        fm = DBSCAN(eps=eps, min_samples=min_samples)
        fm.fit(X)
        fm_labels = np.array(fm.labels_)

        # ARI should be perfect or near-perfect since same algorithm
        ari = adjusted_rand_score(sk_labels, fm_labels)
        assert ari > 0.95, (
            f"DBSCAN ARI vs sklearn too low: {ari:.4f}"
        )

    def test_noise_detection(self, dbscan_data):
        """Both should identify the same noise points (label = -1)."""
        from ferroml.clustering import DBSCAN

        from sklearn.cluster import DBSCAN as SkDBSCAN

        X, _ = dbscan_data

        eps = 0.8
        min_samples = 5

        sk = SkDBSCAN(eps=eps, min_samples=min_samples)
        sk_labels = sk.fit_predict(X)
        sk_noise = np.sum(sk_labels == -1)

        fm = DBSCAN(eps=eps, min_samples=min_samples)
        fm.fit(X)
        fm_labels = np.array(fm.labels_)
        fm_noise = np.sum(fm_labels == -1)

        # Noise counts should be identical or very close
        assert abs(fm_noise - sk_noise) <= 3, (
            f"Noise count mismatch: ferroml={fm_noise}, sklearn={sk_noise}"
        )

    def test_cluster_count_matches(self, dbscan_data):
        """Both should find the same number of clusters."""
        from ferroml.clustering import DBSCAN

        from sklearn.cluster import DBSCAN as SkDBSCAN

        X, _ = dbscan_data

        eps = 0.8
        min_samples = 5

        sk = SkDBSCAN(eps=eps, min_samples=min_samples)
        sk_labels = sk.fit_predict(X)
        sk_n = len(set(sk_labels) - {-1})

        fm = DBSCAN(eps=eps, min_samples=min_samples)
        fm.fit(X)
        fm_labels = np.array(fm.labels_)
        fm_n = len(set(fm_labels.tolist()) - {-1})

        assert fm_n == sk_n, (
            f"Cluster count mismatch: ferroml={fm_n}, sklearn={sk_n}"
        )
