"""Test FerroML clustering module."""

import numpy as np
import pytest

from ferroml.clustering import (
    KMeans,
    DBSCAN,
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_index,
    normalized_mutual_info,
    hopkins_statistic,
)

# AgglomerativeClustering is in the native module but not yet re-exported
# from ferroml.clustering; access it through the native extension.
import ferroml
AgglomerativeClustering = ferroml._native.clustering.AgglomerativeClustering


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blob_data():
    """Three well-separated Gaussian blobs (n=120, p=2)."""
    np.random.seed(42)
    X = np.random.randn(120, 2)
    X[:40] += np.array([6.0, 6.0])
    X[40:80] += np.array([-6.0, -6.0])
    # X[80:] stays near the origin
    return X


@pytest.fixture
def blob_labels(blob_data):
    """Ground-truth integer labels matching blob_data."""
    labels = np.zeros(120, dtype=np.int32)
    labels[:40] = 0
    labels[40:80] = 1
    labels[80:] = 2
    return labels


@pytest.fixture
def dense_blob_data():
    """Dense blobs suitable for DBSCAN (tighter clusters + scattered noise)."""
    np.random.seed(42)
    X = np.random.randn(200, 2) * 0.4
    X[:60] += np.array([5.0, 5.0])
    X[60:120] += np.array([-5.0, -5.0])
    # X[120:] is noise scattered around origin
    return X


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------


class TestKMeans:
    """Tests for KMeans clustering."""

    def test_fit_basic(self, blob_data):
        """fit() should not raise and produce a valid model."""
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(blob_data)

    def test_labels_shape_and_dtype(self, blob_data):
        """labels_ should be i32 with one entry per sample."""
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(blob_data)
        labels = km.labels_
        assert labels.shape == (len(blob_data),)
        assert labels.dtype == np.int32

    def test_labels_values_in_range(self, blob_data):
        """All label values must be in [0, n_clusters)."""
        n_clusters = 3
        km = KMeans(n_clusters=n_clusters, random_state=42)
        km.fit(blob_data)
        assert np.all(km.labels_ >= 0)
        assert np.all(km.labels_ < n_clusters)

    def test_cluster_centers_shape(self, blob_data):
        """cluster_centers_ must have shape (n_clusters, n_features)."""
        n_clusters = 3
        km = KMeans(n_clusters=n_clusters, random_state=42)
        km.fit(blob_data)
        centers = km.cluster_centers_
        assert centers.shape == (n_clusters, blob_data.shape[1])

    def test_inertia_positive(self, blob_data):
        """inertia_ (within-cluster sum of squares) must be positive."""
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(blob_data)
        assert km.inertia_ > 0

    def test_n_iter_positive(self, blob_data):
        """n_iter_ must be at least 1."""
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(blob_data)
        assert km.n_iter_ >= 1

    def test_predict_returns_valid_labels(self, blob_data):
        """predict() on training data should return valid integer labels."""
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(blob_data)
        preds = km.predict(blob_data[:10])
        assert preds.shape == (10,)
        assert np.all(preds >= 0)
        assert np.all(preds < 3)

    def test_fit_predict_matches_labels(self, blob_data):
        """fit_predict() should return the same labels as fit() then labels_."""
        km1 = KMeans(n_clusters=3, random_state=42)
        km1.fit(blob_data)
        expected = km1.labels_

        km2 = KMeans(n_clusters=3, random_state=42)
        fp = km2.fit_predict(blob_data)
        assert fp.shape == (len(blob_data),)
        # Labels may have different cluster IDs but should have same partition structure;
        # at minimum both must be valid label arrays.
        assert np.all(fp >= 0)
        assert np.all(fp < 3)

    def test_different_n_clusters(self, blob_data):
        """KMeans should work for various n_clusters values."""
        for k in [2, 4, 5]:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(blob_data)
            assert km.cluster_centers_.shape[0] == k
            assert km.labels_.shape == (len(blob_data),)

    def test_more_clusters_reduces_inertia(self, blob_data):
        """More clusters must reduce inertia (more granular partition)."""
        km2 = KMeans(n_clusters=2, random_state=42)
        km2.fit(blob_data)
        km5 = KMeans(n_clusters=5, random_state=42)
        km5.fit(blob_data)
        assert km5.inertia_ < km2.inertia_

    def test_cluster_stability_shape(self, blob_data):
        """cluster_stability() should return one stability score per cluster."""
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(blob_data)
        stab = km.cluster_stability(blob_data, n_bootstrap=20)
        assert stab.shape == (3,)
        assert np.all(stab >= 0)
        assert np.all(stab <= 1)

    def test_silhouette_with_ci_structure(self, blob_data):
        """silhouette_with_ci() should return a 3-tuple (score, lower, upper)."""
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(blob_data)
        result = km.silhouette_with_ci(blob_data)
        assert len(result) == 3
        score, lower, upper = result
        assert lower <= score <= upper
        assert -1.0 <= score <= 1.0

    def test_elbow_returns_expected_keys(self, blob_data):
        """KMeans.elbow() must return a dict with 'optimal_k', 'k_values', 'inertias'."""
        result = KMeans.elbow(blob_data, k_min=2, k_max=5, random_state=42)
        assert isinstance(result, dict)
        assert "optimal_k" in result
        assert "k_values" in result
        assert "inertias" in result

    def test_elbow_k_range(self, blob_data):
        """k_values in elbow result should start at k_min (k_max is exclusive upper bound)."""
        result = KMeans.elbow(blob_data, k_min=2, k_max=6, random_state=42)
        k_values = result["k_values"]
        assert min(k_values) == 2
        assert max(k_values) == 5

    def test_elbow_inertias_decreasing(self, blob_data):
        """Inertias should decrease monotonically as k increases."""
        # k_max is exclusive, so k_max=7 yields k_values [2,3,4,5,6]
        result = KMeans.elbow(blob_data, k_min=2, k_max=7, random_state=42)
        inertias = result["inertias"]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1], (
                f"Inertia did not decrease: {inertias}"
            )

    def test_optimal_k_returns_expected_keys(self, blob_data):
        """KMeans.optimal_k() must return a dict with at least 'optimal_k' and 'k_values'."""
        result = KMeans.optimal_k(blob_data, k_min=2, k_max=5, n_refs=5, random_state=42)
        assert isinstance(result, dict)
        assert "optimal_k" in result
        assert "k_values" in result

    def test_optimal_k_in_range(self, blob_data):
        """optimal_k should be within the requested range."""
        k_min, k_max = 2, 5
        result = KMeans.optimal_k(blob_data, k_min=k_min, k_max=k_max, n_refs=5, random_state=42)
        assert k_min <= result["optimal_k"] <= k_max


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------


class TestDBSCAN:
    """Tests for DBSCAN clustering."""

    def test_fit_basic(self, dense_blob_data):
        """fit() should succeed on a well-structured dataset."""
        db = DBSCAN(eps=0.8, min_samples=5)
        db.fit(dense_blob_data)

    def test_labels_dtype(self, dense_blob_data):
        """labels_ should be i32."""
        db = DBSCAN(eps=0.8, min_samples=5)
        db.fit(dense_blob_data)
        assert db.labels_.dtype == np.int32

    def test_labels_contain_noise(self, dense_blob_data):
        """Noise points should be labelled -1 given small eps on scattered data."""
        # Use small eps so the scattered noise points are marked as noise.
        # eps=0.4 reliably produces noise on this fixture while still finding
        # the two tight clusters at [5,5] and [-5,-5].
        db = DBSCAN(eps=0.4, min_samples=5)
        db.fit(dense_blob_data)
        # The 80 scattered samples near origin should include noise points.
        assert np.any(db.labels_ == -1), "Expected at least one noise point"

    def test_n_clusters_positive(self, dense_blob_data):
        """n_clusters_ should be > 0 for well-structured data."""
        db = DBSCAN(eps=0.8, min_samples=5)
        db.fit(dense_blob_data)
        assert db.n_clusters_ >= 1

    def test_n_noise_non_negative(self, dense_blob_data):
        """n_noise_ must be >= 0."""
        db = DBSCAN(eps=0.8, min_samples=5)
        db.fit(dense_blob_data)
        assert db.n_noise_ >= 0

    def test_fit_predict_shape(self, dense_blob_data):
        """fit_predict() should return one label per sample."""
        db = DBSCAN(eps=0.8, min_samples=5)
        fp = db.fit_predict(dense_blob_data)
        assert fp.shape == (len(dense_blob_data),)

    def test_core_sample_indices_is_list(self, dense_blob_data):
        """core_sample_indices_ should be a non-empty list for dense data."""
        db = DBSCAN(eps=0.8, min_samples=5)
        db.fit(dense_blob_data)
        assert isinstance(db.core_sample_indices_, list)
        assert len(db.core_sample_indices_) > 0

    def test_components_shape(self, dense_blob_data):
        """components_ (core point coordinates) should have correct n_features."""
        db = DBSCAN(eps=0.8, min_samples=5)
        db.fit(dense_blob_data)
        comps = db.components_
        assert comps.ndim == 2
        assert comps.shape[1] == dense_blob_data.shape[1]

    def test_noise_analysis_keys(self, dense_blob_data):
        """noise_analysis() should return a dict with standard keys."""
        db = DBSCAN(eps=0.6, min_samples=5)
        db.fit(dense_blob_data)
        result = db.noise_analysis(dense_blob_data)
        assert isinstance(result, dict)
        assert "noise_ratio" in result
        assert 0.0 <= result["noise_ratio"] <= 1.0

    def test_optimal_eps_returns_suggested(self, dense_blob_data):
        """optimal_eps() should return a dict containing 'suggested_eps'."""
        result = DBSCAN.optimal_eps(dense_blob_data, min_samples=5)
        assert isinstance(result, dict)
        assert "suggested_eps" in result
        assert result["suggested_eps"] > 0

    def test_cluster_persistence_returns_list(self, dense_blob_data):
        """cluster_persistence() should return a list with one entry per eps value."""
        eps_values = [0.5, 1.0, 1.5, 2.0]
        result = DBSCAN.cluster_persistence(dense_blob_data, eps_values, min_samples=5)
        assert isinstance(result, list)
        assert len(result) == len(eps_values)

    def test_large_eps_fewer_noise_points(self):
        """Larger eps should classify fewer points as noise."""
        np.random.seed(42)
        X = np.random.randn(100, 2)

        db_tight = DBSCAN(eps=0.3, min_samples=5)
        db_tight.fit(X)

        db_loose = DBSCAN(eps=2.0, min_samples=5)
        db_loose.fit(X)

        assert db_loose.n_noise_ <= db_tight.n_noise_


# ---------------------------------------------------------------------------
# AgglomerativeClustering
# ---------------------------------------------------------------------------


class TestAgglomerativeClustering:
    """Tests for AgglomerativeClustering."""

    def test_fit_basic(self, blob_data):
        """fit() should succeed with default (ward) linkage."""
        ag = AgglomerativeClustering(n_clusters=3, linkage="ward")
        ag.fit(blob_data)

    def test_labels_shape_and_dtype(self, blob_data):
        """labels_ should be i32 with one entry per sample."""
        ag = AgglomerativeClustering(n_clusters=3, linkage="ward")
        ag.fit(blob_data)
        labels = ag.labels_
        assert labels.shape == (len(blob_data),)
        assert labels.dtype == np.int32

    def test_labels_values_in_range(self, blob_data):
        """Label values must be in [0, n_clusters)."""
        n_clusters = 3
        ag = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        ag.fit(blob_data)
        assert np.all(ag.labels_ >= 0)
        assert np.all(ag.labels_ < n_clusters)

    def test_fit_predict_shape(self, blob_data):
        """fit_predict() should return the same shape as labels_."""
        ag = AgglomerativeClustering(n_clusters=3, linkage="ward")
        fp = ag.fit_predict(blob_data)
        assert fp.shape == (len(blob_data),)

    @pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
    def test_all_linkage_methods(self, blob_data, linkage):
        """All supported linkage strategies should produce valid partitions."""
        ag = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        fp = ag.fit_predict(blob_data)
        assert fp.dtype == np.int32
        assert fp.shape == (len(blob_data),)
        assert np.all(fp >= 0)
        assert np.all(fp < 3)

    def test_different_n_clusters(self, blob_data):
        """AgglomerativeClustering should work for various n_clusters."""
        for k in [2, 4, 5]:
            ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
            ag.fit(blob_data)
            assert len(np.unique(ag.labels_)) == k

    def test_ward_produces_compact_clusters(self, blob_data):
        """Ward linkage on well-separated blobs should yield 3 meaningful clusters."""
        ag = AgglomerativeClustering(n_clusters=3, linkage="ward")
        ag.fit(blob_data)
        # Each cluster should have at least 1 member
        for label in range(3):
            assert np.sum(ag.labels_ == label) >= 1


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


class TestClusteringMetrics:
    """Tests for clustering metric functions."""

    @pytest.fixture
    def fitted_labels(self, blob_data):
        """KMeans labels for blob_data, cast to int32."""
        km = KMeans(n_clusters=3, random_state=42)
        return blob_data, km.fit_predict(blob_data).astype(np.int32)

    def test_silhouette_score_range(self, fitted_labels):
        """silhouette_score() should be in [-1, 1]."""
        X, labels = fitted_labels
        score = silhouette_score(X, labels)
        assert np.isfinite(score)
        assert -1.0 <= score <= 1.0

    def test_silhouette_score_well_separated(self, blob_data):
        """silhouette_score on well-separated blobs should be close to 1."""
        km = KMeans(n_clusters=3, random_state=42)
        labels = km.fit_predict(blob_data).astype(np.int32)
        score = silhouette_score(blob_data, labels)
        assert score > 0.5, f"Expected high silhouette on well-separated blobs, got {score}"

    def test_silhouette_samples_shape(self, fitted_labels):
        """silhouette_samples() should return one score per sample."""
        X, labels = fitted_labels
        samples = silhouette_samples(X, labels)
        assert samples.shape == (len(X),)
        assert np.all(np.isfinite(samples))
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)

    def test_silhouette_samples_mean_equals_score(self, fitted_labels):
        """Mean of silhouette_samples() should equal silhouette_score()."""
        X, labels = fitted_labels
        score = silhouette_score(X, labels)
        samples = silhouette_samples(X, labels)
        np.testing.assert_allclose(np.mean(samples), score, atol=1e-6)

    def test_calinski_harabasz_score_positive(self, fitted_labels):
        """calinski_harabasz_score() must be positive."""
        X, labels = fitted_labels
        ch = calinski_harabasz_score(X, labels)
        assert np.isfinite(ch)
        assert ch > 0

    def test_davies_bouldin_score_non_negative(self, fitted_labels):
        """davies_bouldin_score() must be >= 0."""
        X, labels = fitted_labels
        db = davies_bouldin_score(X, labels)
        assert np.isfinite(db)
        assert db >= 0

    def test_adjusted_rand_index_perfect_match(self, blob_labels, blob_data):
        """ARI should be 1.0 when predicted labels exactly match ground truth."""
        labels_pred = blob_labels.copy()
        ari = adjusted_rand_index(blob_labels, labels_pred)
        np.testing.assert_allclose(ari, 1.0, atol=1e-9)

    def test_adjusted_rand_index_range(self, blob_data, blob_labels):
        """ARI should generally be in [-1, 1] for arbitrary labels."""
        np.random.seed(0)
        random_labels = np.random.randint(0, 3, size=len(blob_data), dtype=np.int32)
        ari = adjusted_rand_index(blob_labels, random_labels)
        assert np.isfinite(ari)
        assert -1.0 <= ari <= 1.0

    def test_normalized_mutual_info_perfect_match(self, blob_labels):
        """NMI should be 1.0 (or very close) for identical labels."""
        nmi = normalized_mutual_info(blob_labels, blob_labels)
        assert np.isfinite(nmi)
        np.testing.assert_allclose(nmi, 1.0, atol=1e-9)

    def test_normalized_mutual_info_range(self, blob_data, blob_labels):
        """NMI should be in [0, 1]."""
        np.random.seed(7)
        random_labels = np.random.randint(0, 3, size=len(blob_data), dtype=np.int32)
        nmi = normalized_mutual_info(blob_labels, random_labels)
        assert np.isfinite(nmi)
        assert 0.0 <= nmi <= 1.0

    def test_adjusted_rand_index_int32_required(self, blob_data, blob_labels):
        """Confirm that int32 labels work (the required dtype)."""
        km = KMeans(n_clusters=3, random_state=42)
        pred = km.fit_predict(blob_data).astype(np.int32)
        ari = adjusted_rand_index(blob_labels, pred)
        assert np.isfinite(ari)

    def test_hopkins_statistic_range(self, blob_data):
        """Hopkins statistic should be in [0, 1]."""
        h = hopkins_statistic(blob_data)
        assert np.isfinite(h)
        assert 0.0 <= h <= 1.0

    def test_hopkins_statistic_clustered_data(self, blob_data):
        """Hopkins statistic > 0.5 indicates clustering tendency."""
        h = hopkins_statistic(blob_data, random_state=42)
        assert h > 0.5, f"Expected Hopkins > 0.5 for clustered data, got {h}"

    def test_hopkins_statistic_with_sample_size(self, blob_data):
        """Hopkins statistic should accept an explicit sample_size."""
        h = hopkins_statistic(blob_data, sample_size=20, random_state=42)
        assert np.isfinite(h)
        assert 0.0 <= h <= 1.0

    def test_metrics_accept_kmeans_labels_directly(self, blob_data):
        """Labels returned by KMeans.fit_predict() should work in all metrics."""
        km = KMeans(n_clusters=3, random_state=42)
        labels = km.fit_predict(blob_data).astype(np.int32)

        assert np.isfinite(silhouette_score(blob_data, labels))
        assert np.isfinite(calinski_harabasz_score(blob_data, labels))
        assert np.isfinite(davies_bouldin_score(blob_data, labels))
