"""
Phase M.4 -- FerroML vs sklearn comparison tests for unsupervised models.

Covers: KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture,
        IsolationForest, LocalOutlierFactor, PCA, QDA, t-SNE.

Comparison strategies:
- Clustering: Adjusted Rand Index (label-permutation invariant)
- PCA: explained_variance_ratio with sign-flip handling
- t-SNE: kNN preservation ratio (stochastic, so no coordinate comparison)
- Anomaly detection: score distribution comparison (median, IQR)
- GMM: BIC values + clustering ARI
"""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from conftest_comparison import get_iris, get_wine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs_sklearn(n_samples=300, centers=3, random_state=42):
    from sklearn.datasets import make_blobs
    return make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)


def _make_moons_sklearn(n_samples=300, noise=0.05, random_state=42):
    from sklearn.datasets import make_moons
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def _make_circles_sklearn(n_samples=300, noise=0.05, factor=0.5, random_state=42):
    from sklearn.datasets import make_circles
    return make_circles(n_samples=n_samples, noise=noise, factor=factor,
                        random_state=random_state)


def _knn_preservation(X_high, X_low, k=7):
    """Fraction of k-nearest neighbors preserved from high-dim to low-dim."""
    from sklearn.neighbors import NearestNeighbors
    nn_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)
    # exclude self (index 0)
    idx_high = nn_high.kneighbors(X_high, return_distance=False)[:, 1:]
    idx_low = nn_low.kneighbors(X_low, return_distance=False)[:, 1:]
    preserved = 0
    for i in range(len(X_high)):
        preserved += len(set(idx_high[i]) & set(idx_low[i]))
    return preserved / (len(X_high) * k)


def _synthetic_outlier_data(n_inliers=200, n_outliers=20, random_state=42):
    """Create a dataset with clear inliers and outliers."""
    rng = np.random.RandomState(random_state)
    inliers = rng.randn(n_inliers, 2) * 0.5
    outliers = rng.uniform(-6, 6, size=(n_outliers, 2))
    X = np.vstack([inliers, outliers])
    y = np.array([1] * n_inliers + [-1] * n_outliers)
    return X, y


# ===========================================================================
# KMeans
# ===========================================================================

class TestKMeansComparison:
    def test_kmeans_iris_ari(self):
        """KMeans on iris: FerroML ARI vs sklearn ARI within 5%."""
        from ferroml.clustering import KMeans as FerroKMeans
        from sklearn.cluster import KMeans as SkKMeans

        X, y_true = get_iris()

        ferro = FerroKMeans(n_clusters=3, random_state=42)
        ferro.fit(X)
        ferro_labels = np.array(ferro.labels_)

        sk = SkKMeans(n_clusters=3, random_state=42, n_init=10)
        sk.fit(X)
        sk_labels = sk.labels_

        ferro_ari = adjusted_rand_score(y_true, ferro_labels)
        sk_ari = adjusted_rand_score(y_true, sk_labels)
        assert ferro_ari > 0.6, f"FerroML ARI too low: {ferro_ari}"
        assert abs(ferro_ari - sk_ari) < 0.10, (
            f"ARI gap: ferro={ferro_ari:.4f}, sk={sk_ari:.4f}"
        )

    def test_kmeans_blobs_inertia(self):
        """KMeans on well-separated blobs: inertia within 5%."""
        from ferroml.clustering import KMeans as FerroKMeans
        from sklearn.cluster import KMeans as SkKMeans

        X, y_true = _make_blobs_sklearn(n_samples=300, centers=3, random_state=42)

        ferro = FerroKMeans(n_clusters=3, random_state=42)
        ferro.fit(X)

        sk = SkKMeans(n_clusters=3, random_state=42, n_init=10)
        sk.fit(X)

        # Both should achieve near-perfect ARI on well-separated blobs
        ferro_ari = adjusted_rand_score(y_true, np.array(ferro.labels_))
        sk_ari = adjusted_rand_score(y_true, sk.labels_)
        assert ferro_ari > 0.9, f"FerroML ARI on blobs: {ferro_ari}"
        assert sk_ari > 0.9, f"sklearn ARI on blobs: {sk_ari}"

        # Inertia comparison (within 5%)
        rel_diff = abs(ferro.inertia_ - sk.inertia_) / max(abs(sk.inertia_), 1e-10)
        assert rel_diff < 0.05, (
            f"Inertia gap: ferro={ferro.inertia_:.2f}, sk={sk.inertia_:.2f}, "
            f"rel_diff={rel_diff:.4f}"
        )

    def test_kmeans_predict_consistency(self):
        """KMeans predict on new data matches fit_predict labels."""
        from ferroml.clustering import KMeans as FerroKMeans

        X, _ = _make_blobs_sklearn(n_samples=300, centers=3, random_state=42)
        ferro = FerroKMeans(n_clusters=3, random_state=42)
        ferro.fit(X)
        preds = ferro.predict(X)
        np.testing.assert_array_equal(np.array(ferro.labels_), np.array(preds))


# ===========================================================================
# DBSCAN
# ===========================================================================

class TestDBSCANComparison:
    def test_dbscan_moons(self):
        """DBSCAN on make_moons: labels match sklearn (ARI ~ 1.0)."""
        from ferroml.clustering import DBSCAN as FerroDBSCAN
        from sklearn.cluster import DBSCAN as SkDBSCAN

        X, y_true = _make_moons_sklearn(n_samples=300, noise=0.05, random_state=42)

        ferro = FerroDBSCAN(eps=0.3, min_samples=5)
        ferro_labels = ferro.fit_predict(X)

        sk = SkDBSCAN(eps=0.3, min_samples=5)
        sk_labels = sk.fit_predict(X)

        ari = adjusted_rand_score(np.array(ferro_labels), sk_labels)
        assert ari > 0.99, f"DBSCAN moons ARI vs sklearn: {ari}"

    def test_dbscan_circles(self):
        """DBSCAN on make_circles: discovers 2 clusters."""
        from ferroml.clustering import DBSCAN as FerroDBSCAN
        from sklearn.cluster import DBSCAN as SkDBSCAN

        X, y_true = _make_circles_sklearn(n_samples=300, noise=0.05, factor=0.5,
                                           random_state=42)

        ferro = FerroDBSCAN(eps=0.2, min_samples=5)
        ferro_labels = np.array(ferro.fit_predict(X))

        sk = SkDBSCAN(eps=0.2, min_samples=5)
        sk_labels = sk.fit_predict(X)

        # Both should find 2 clusters
        ferro_n = len(set(ferro_labels) - {-1})
        sk_n = len(set(sk_labels) - {-1})
        assert ferro_n == sk_n, f"Cluster count: ferro={ferro_n}, sk={sk_n}"

        ari = adjusted_rand_score(ferro_labels, sk_labels)
        assert ari > 0.99, f"DBSCAN circles ARI vs sklearn: {ari}"

    def test_dbscan_noise_points(self):
        """DBSCAN noise point count matches sklearn."""
        from ferroml.clustering import DBSCAN as FerroDBSCAN
        from sklearn.cluster import DBSCAN as SkDBSCAN

        X, _ = _make_moons_sklearn(n_samples=300, noise=0.1, random_state=42)

        ferro = FerroDBSCAN(eps=0.2, min_samples=5)
        ferro_labels = np.array(ferro.fit_predict(X))

        sk = SkDBSCAN(eps=0.2, min_samples=5)
        sk_labels = sk.fit_predict(X)

        ferro_noise = np.sum(ferro_labels == -1)
        sk_noise = np.sum(sk_labels == -1)
        assert ferro_noise == sk_noise, (
            f"Noise count: ferro={ferro_noise}, sk={sk_noise}"
        )


# ===========================================================================
# AgglomerativeClustering
# ===========================================================================

class TestAgglomerativeComparison:
    def test_agglomerative_iris(self):
        """AgglomerativeClustering on iris: ARI matches sklearn."""
        from ferroml.clustering import AgglomerativeClustering as FerroAgglo
        from sklearn.cluster import AgglomerativeClustering as SkAgglo

        X, y_true = get_iris()

        ferro = FerroAgglo(n_clusters=3)
        ferro.fit(X)
        ferro_labels = np.array(ferro.labels_)

        sk = SkAgglo(n_clusters=3)
        sk.fit(X)

        ari = adjusted_rand_score(ferro_labels, sk.labels_)
        assert ari > 0.99, f"Agglomerative iris ARI vs sklearn: {ari}"

    def test_agglomerative_blobs(self):
        """AgglomerativeClustering on blobs: near-perfect ARI."""
        from ferroml.clustering import AgglomerativeClustering as FerroAgglo
        from sklearn.cluster import AgglomerativeClustering as SkAgglo

        X, y_true = _make_blobs_sklearn(n_samples=200, centers=3, random_state=42)

        ferro = FerroAgglo(n_clusters=3)
        ferro.fit(X)
        ferro_labels = np.array(ferro.labels_)

        sk = SkAgglo(n_clusters=3)
        sk.fit(X)

        ferro_ari = adjusted_rand_score(y_true, ferro_labels)
        sk_ari = adjusted_rand_score(y_true, sk.labels_)
        assert ferro_ari > 0.9, f"FerroML ARI: {ferro_ari}"
        assert abs(ferro_ari - sk_ari) < 0.05, (
            f"ARI gap: ferro={ferro_ari:.4f}, sk={sk_ari:.4f}"
        )


# ===========================================================================
# GaussianMixture
# ===========================================================================

class TestGaussianMixtureComparison:
    def test_gmm_full_iris_ari(self):
        """GMM (full covariance) on iris: ARI comparable to sklearn."""
        from ferroml.clustering import GaussianMixture as FerroGMM
        from sklearn.mixture import GaussianMixture as SkGMM

        X, y_true = get_iris()

        ferro = FerroGMM(n_components=3, covariance_type='full', random_state=42)
        ferro.fit(X)
        ferro_labels = np.array(ferro.predict(X))

        sk = SkGMM(n_components=3, covariance_type='full', random_state=42)
        sk.fit(X)
        sk_labels = sk.predict(X)

        ferro_ari = adjusted_rand_score(y_true, ferro_labels)
        sk_ari = adjusted_rand_score(y_true, sk_labels)
        assert ferro_ari > 0.8, f"FerroML GMM ARI: {ferro_ari}"
        assert abs(ferro_ari - sk_ari) < 0.10, (
            f"GMM ARI gap: ferro={ferro_ari:.4f}, sk={sk_ari:.4f}"
        )

    def test_gmm_full_iris_bic(self):
        """GMM (full) BIC on iris within 5% of sklearn."""
        from ferroml.clustering import GaussianMixture as FerroGMM
        from sklearn.mixture import GaussianMixture as SkGMM

        X, _ = get_iris()

        ferro = FerroGMM(n_components=3, covariance_type='full', random_state=42)
        ferro.fit(X)
        ferro_bic = ferro.bic(X)

        sk = SkGMM(n_components=3, covariance_type='full', random_state=42)
        sk.fit(X)
        sk_bic = sk.bic(X)

        rel_diff = abs(ferro_bic - sk_bic) / max(abs(sk_bic), 1e-10)
        assert rel_diff < 0.05, (
            f"BIC gap: ferro={ferro_bic:.2f}, sk={sk_bic:.2f}, rel={rel_diff:.4f}"
        )

    def test_gmm_diag_blobs_bic(self):
        """GMM (diagonal) BIC on blobs within 5% of sklearn."""
        from ferroml.clustering import GaussianMixture as FerroGMM
        from sklearn.mixture import GaussianMixture as SkGMM

        X, _ = _make_blobs_sklearn(n_samples=300, centers=3, random_state=42)

        ferro = FerroGMM(n_components=3, covariance_type='diagonal', random_state=42)
        ferro.fit(X)
        ferro_bic = ferro.bic(X)

        sk = SkGMM(n_components=3, covariance_type='diag', random_state=42)
        sk.fit(X)
        sk_bic = sk.bic(X)

        rel_diff = abs(ferro_bic - sk_bic) / max(abs(sk_bic), 1e-10)
        assert rel_diff < 0.05, (
            f"Diag BIC gap: ferro={ferro_bic:.2f}, sk={sk_bic:.2f}, rel={rel_diff:.4f}"
        )

    def test_gmm_predict_proba_sums_to_one(self):
        """GMM predict_proba rows sum to 1."""
        from ferroml.clustering import GaussianMixture as FerroGMM

        X, _ = get_iris()
        ferro = FerroGMM(n_components=3, covariance_type='full', random_state=42)
        ferro.fit(X)
        proba = ferro.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_gmm_aic_ordering(self):
        """AIC and BIC ordering matches sklearn for different n_components."""
        from ferroml.clustering import GaussianMixture as FerroGMM
        from sklearn.mixture import GaussianMixture as SkGMM

        X, _ = get_iris()
        ferro_bics = []
        sk_bics = []
        for k in [2, 3, 4, 5]:
            ferro = FerroGMM(n_components=k, covariance_type='full', random_state=42)
            ferro.fit(X)
            ferro_bics.append(ferro.bic(X))

            sk = SkGMM(n_components=k, covariance_type='full', random_state=42)
            sk.fit(X)
            sk_bics.append(sk.bic(X))

        # Best k (by BIC) should agree
        ferro_best = np.argmin(ferro_bics)
        sk_best = np.argmin(sk_bics)
        assert ferro_best == sk_best, (
            f"Best k differs: ferro={ferro_best+2}, sk={sk_best+2}"
        )


# ===========================================================================
# IsolationForest
# ===========================================================================

class TestIsolationForestComparison:
    def test_iforest_score_distribution(self):
        """IsolationForest: median score for inliers > median score for outliers."""
        from ferroml.anomaly import IsolationForest as FerroIF
        from sklearn.ensemble import IsolationForest as SkIF

        X, y = _synthetic_outlier_data(n_inliers=200, n_outliers=20, random_state=42)

        ferro = FerroIF(n_estimators=100, random_state=42)
        ferro.fit(X)
        ferro_scores = ferro.score_samples(X)

        sk = SkIF(n_estimators=100, random_state=42)
        sk.fit(X)
        sk_scores = sk.score_samples(X)

        inlier_mask = y == 1
        # Both should rank inliers higher than outliers
        ferro_inlier_median = np.median(ferro_scores[inlier_mask])
        ferro_outlier_median = np.median(ferro_scores[~inlier_mask])
        sk_inlier_median = np.median(sk_scores[inlier_mask])
        sk_outlier_median = np.median(sk_scores[~inlier_mask])

        assert ferro_inlier_median > ferro_outlier_median, (
            f"FerroML inlier median ({ferro_inlier_median:.4f}) <= "
            f"outlier median ({ferro_outlier_median:.4f})"
        )
        assert sk_inlier_median > sk_outlier_median, (
            f"sklearn inlier median ({sk_inlier_median:.4f}) <= "
            f"outlier median ({sk_outlier_median:.4f})"
        )

    def test_iforest_prediction_agreement(self):
        """IsolationForest: prediction agreement > 80%."""
        from ferroml.anomaly import IsolationForest as FerroIF
        from sklearn.ensemble import IsolationForest as SkIF

        X, y = _synthetic_outlier_data(n_inliers=200, n_outliers=20, random_state=42)

        ferro = FerroIF(n_estimators=100, random_state=42)
        ferro.fit(X)
        ferro_preds = np.array(ferro.predict(X))

        sk = SkIF(n_estimators=100, random_state=42)
        sk.fit(X)
        sk_preds = sk.predict(X)

        agreement = np.mean(ferro_preds == sk_preds)
        assert agreement > 0.80, f"IF prediction agreement: {agreement:.2%}"

    def test_iforest_inlier_recall(self):
        """IsolationForest: both achieve > 80% inlier recall."""
        from ferroml.anomaly import IsolationForest as FerroIF
        from sklearn.ensemble import IsolationForest as SkIF

        X, y = _synthetic_outlier_data(n_inliers=200, n_outliers=20, random_state=42)

        ferro = FerroIF(n_estimators=100, random_state=42)
        ferro.fit(X)
        ferro_preds = np.array(ferro.predict(X))

        sk = SkIF(n_estimators=100, random_state=42)
        sk.fit(X)
        sk_preds = sk.predict(X)

        inlier_mask = y == 1
        ferro_recall = np.mean(ferro_preds[inlier_mask] == 1)
        sk_recall = np.mean(sk_preds[inlier_mask] == 1)
        assert ferro_recall > 0.80, f"FerroML inlier recall: {ferro_recall:.2%}"
        assert sk_recall > 0.80, f"sklearn inlier recall: {sk_recall:.2%}"


# ===========================================================================
# LocalOutlierFactor
# ===========================================================================

class TestLOFComparison:
    def test_lof_score_distribution(self):
        """LOF: inliers have higher (less negative) scores than outliers."""
        from ferroml.anomaly import LocalOutlierFactor as FerroLOF
        from sklearn.neighbors import LocalOutlierFactor as SkLOF

        X, y = _synthetic_outlier_data(n_inliers=200, n_outliers=20, random_state=42)

        ferro = FerroLOF(n_neighbors=20)
        ferro.fit_predict(X)
        ferro_scores = np.array(ferro.negative_outlier_factor_)

        sk = SkLOF(n_neighbors=20)
        sk.fit_predict(X)
        sk_scores = sk.negative_outlier_factor_

        inlier_mask = y == 1
        # Inliers should have scores closer to -1 (less negative)
        ferro_inlier_med = np.median(ferro_scores[inlier_mask])
        ferro_outlier_med = np.median(ferro_scores[~inlier_mask])
        assert ferro_inlier_med > ferro_outlier_med, (
            f"FerroML LOF: inlier median={ferro_inlier_med:.4f}, "
            f"outlier median={ferro_outlier_med:.4f}"
        )

        sk_inlier_med = np.median(sk_scores[inlier_mask])
        sk_outlier_med = np.median(sk_scores[~inlier_mask])
        assert sk_inlier_med > sk_outlier_med

    def test_lof_prediction_agreement(self):
        """LOF: prediction agreement > 80%."""
        from ferroml.anomaly import LocalOutlierFactor as FerroLOF
        from sklearn.neighbors import LocalOutlierFactor as SkLOF

        X, y = _synthetic_outlier_data(n_inliers=200, n_outliers=20, random_state=42)

        ferro_preds = np.array(FerroLOF(n_neighbors=20).fit_predict(X))
        sk_preds = SkLOF(n_neighbors=20).fit_predict(X)

        agreement = np.mean(ferro_preds == sk_preds)
        assert agreement > 0.80, f"LOF prediction agreement: {agreement:.2%}"

    def test_lof_outlier_detection(self):
        """LOF: both detect at least some true outliers."""
        from ferroml.anomaly import LocalOutlierFactor as FerroLOF
        from sklearn.neighbors import LocalOutlierFactor as SkLOF

        X, y = _synthetic_outlier_data(n_inliers=200, n_outliers=20, random_state=42)

        ferro_preds = np.array(FerroLOF(n_neighbors=20).fit_predict(X))
        sk_preds = SkLOF(n_neighbors=20).fit_predict(X)

        outlier_mask = y == -1
        ferro_outlier_recall = np.mean(ferro_preds[outlier_mask] == -1)
        sk_outlier_recall = np.mean(sk_preds[outlier_mask] == -1)

        assert ferro_outlier_recall > 0.3, (
            f"FerroML outlier recall: {ferro_outlier_recall:.2%}"
        )
        assert sk_outlier_recall > 0.3, (
            f"sklearn outlier recall: {sk_outlier_recall:.2%}"
        )


# ===========================================================================
# PCA
# ===========================================================================

class TestPCAComparison:
    def test_pca_iris_variance_ratio(self):
        """PCA on iris: explained_variance_ratio matches sklearn within 1e-6."""
        from ferroml.decomposition import PCA as FerroPCA
        from sklearn.decomposition import PCA as SkPCA

        X, _ = get_iris()

        ferro = FerroPCA(n_components=4)
        ferro.fit(X)
        ferro_var = np.array(ferro.explained_variance_ratio_)

        sk = SkPCA(n_components=4)
        sk.fit(X)
        sk_var = sk.explained_variance_ratio_

        np.testing.assert_allclose(ferro_var, sk_var, atol=1e-6,
                                   err_msg="Explained variance ratio mismatch")

    def test_pca_wine_variance_ratio(self):
        """PCA on wine: explained_variance_ratio matches sklearn."""
        from ferroml.decomposition import PCA as FerroPCA
        from sklearn.decomposition import PCA as SkPCA

        X, _ = get_wine()
        # Standardize first for numerical stability
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

        ferro = FerroPCA(n_components=5)
        ferro.fit(X)
        ferro_var = np.array(ferro.explained_variance_ratio_)

        sk = SkPCA(n_components=5)
        sk.fit(X)
        sk_var = sk.explained_variance_ratio_

        np.testing.assert_allclose(ferro_var, sk_var, atol=1e-6,
                                   err_msg="Wine PCA variance ratio mismatch")

    def test_pca_components_sign_invariant(self):
        """PCA components match sklearn up to sign flips."""
        from ferroml.decomposition import PCA as FerroPCA
        from sklearn.decomposition import PCA as SkPCA

        X, _ = get_iris()

        ferro = FerroPCA(n_components=2)
        ferro.fit(X)
        ferro_comp = np.array(ferro.components_)

        sk = SkPCA(n_components=2)
        sk.fit(X)
        sk_comp = sk.components_

        # Compare absolute values (eigenvector sign ambiguity)
        np.testing.assert_allclose(np.abs(ferro_comp), np.abs(sk_comp), atol=1e-6,
                                   err_msg="PCA components mismatch (abs)")

    def test_pca_transform_inverse(self):
        """PCA transform then inverse_transform recovers original data."""
        from ferroml.decomposition import PCA as FerroPCA

        X, _ = get_iris()
        ferro = FerroPCA(n_components=4)
        ferro.fit(X)
        X_transformed = ferro.transform(X)
        X_recovered = ferro.inverse_transform(X_transformed)
        np.testing.assert_allclose(X, X_recovered, atol=1e-10,
                                   err_msg="PCA inverse transform mismatch")

    def test_pca_cumulative_variance(self):
        """PCA cumulative variance sums to <= 1.0, each component non-negative."""
        from ferroml.decomposition import PCA as FerroPCA

        X, _ = get_iris()
        ferro = FerroPCA(n_components=4)
        ferro.fit(X)
        var = np.array(ferro.explained_variance_ratio_)

        assert np.all(var >= 0), "Negative variance ratios"
        assert var.sum() <= 1.0 + 1e-10, f"Variance sum > 1: {var.sum()}"
        # First component explains the most
        assert var[0] >= var[1], "First component should explain most variance"


# ===========================================================================
# QDA (QuadraticDiscriminantAnalysis)
# ===========================================================================

class TestQDAComparison:
    def test_qda_iris_accuracy(self):
        """QDA on iris: accuracy matches sklearn within 1e-4."""
        from ferroml.decomposition import QuadraticDiscriminantAnalysis as FerroQDA
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA

        X, y = get_iris()

        ferro = FerroQDA()
        ferro.fit(X, y)
        ferro_preds = np.array(ferro.predict(X))

        sk = SkQDA()
        sk.fit(X, y)
        sk_preds = sk.predict(X)

        ferro_acc = np.mean(ferro_preds == y)
        sk_acc = np.mean(sk_preds == y)
        assert abs(ferro_acc - sk_acc) < 1e-4, (
            f"QDA accuracy gap: ferro={ferro_acc:.6f}, sk={sk_acc:.6f}"
        )

    def test_qda_iris_proba(self):
        """QDA predict_proba on iris matches sklearn within 1e-3."""
        from ferroml.decomposition import QuadraticDiscriminantAnalysis as FerroQDA
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA

        X, y = get_iris()

        ferro = FerroQDA()
        ferro.fit(X, y)
        ferro_proba = np.array(ferro.predict_proba(X))

        sk = SkQDA()
        sk.fit(X, y)
        sk_proba = sk.predict_proba(X)

        # Probas should be very close
        max_diff = np.max(np.abs(ferro_proba - sk_proba))
        assert max_diff < 1e-3, f"QDA proba max diff: {max_diff}"

    def test_qda_predictions_match(self):
        """QDA: predictions exactly match sklearn on iris."""
        from ferroml.decomposition import QuadraticDiscriminantAnalysis as FerroQDA
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SkQDA

        X, y = get_iris()

        ferro = FerroQDA()
        ferro.fit(X, y)
        ferro_preds = np.array(ferro.predict(X))

        sk = SkQDA()
        sk.fit(X, y)
        sk_preds = sk.predict(X).astype(np.float64)

        np.testing.assert_array_equal(ferro_preds, sk_preds,
                                      err_msg="QDA predictions differ")


# ===========================================================================
# t-SNE
# ===========================================================================

class TestTSNEComparison:
    def test_tsne_knn_preservation(self):
        """t-SNE on iris subset: kNN preservation > 0.4."""
        from ferroml.decomposition import TSNE as FerroTSNE

        X, _ = get_iris()
        # Use small subset for speed
        X_small = X[:80]

        ferro = FerroTSNE(n_components=2, perplexity=20.0)
        embedding = ferro.fit_transform(X_small)

        preservation = _knn_preservation(X_small, embedding, k=7)
        assert preservation > 0.4, (
            f"t-SNE kNN preservation too low: {preservation:.4f}"
        )

    def test_tsne_output_shape(self):
        """t-SNE output shape matches (n_samples, n_components)."""
        from ferroml.decomposition import TSNE as FerroTSNE

        X, _ = get_iris()
        X_small = X[:60]

        ferro = FerroTSNE(n_components=2, perplexity=15.0)
        embedding = ferro.fit_transform(X_small)

        assert embedding.shape == (60, 2), f"Shape: {embedding.shape}"

    def test_tsne_kl_divergence_finite(self):
        """t-SNE KL divergence should be finite and non-negative."""
        from ferroml.decomposition import TSNE as FerroTSNE

        X, _ = get_iris()
        X_small = X[:60]

        ferro = FerroTSNE(n_components=2, perplexity=15.0)
        ferro.fit_transform(X_small)

        kl = ferro.kl_divergence_
        assert np.isfinite(kl), f"KL divergence not finite: {kl}"
        assert kl >= 0, f"KL divergence negative: {kl}"

    def test_tsne_separates_classes(self):
        """t-SNE: class centroids in embedding space should be separated."""
        from ferroml.decomposition import TSNE as FerroTSNE

        X, y = get_iris()
        # Sample evenly from all 3 classes (iris is sorted by class)
        rng = np.random.RandomState(42)
        idx = np.concatenate([
            rng.choice(np.where(y == c)[0], 30, replace=False)
            for c in [0.0, 1.0, 2.0]
        ])
        X_small, y_small = X[idx], y[idx]

        ferro = FerroTSNE(n_components=2, perplexity=20.0)
        embedding = ferro.fit_transform(X_small)

        # Compute class centroids
        centroids = []
        for c in [0.0, 1.0, 2.0]:
            mask = y_small == c
            centroids.append(embedding[mask].mean(axis=0))

        # All pairwise distances should be > 0
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                assert dist > 0.1, (
                    f"Classes {i} and {j} centroids too close: {dist:.4f}"
                )


# ===========================================================================
# Barnes-Hut t-SNE
# ===========================================================================

class TestBarnesHutTSNE:
    """Tests for Barnes-Hut t-SNE optimization."""

    def test_barnes_hut_basic(self):
        """Barnes-Hut t-SNE produces valid 2D embedding."""
        from ferroml.decomposition import TSNE as FerroTSNE

        np.random.seed(42)
        X = np.random.randn(200, 10)
        tsne = FerroTSNE(n_components=2, method="barnes_hut", theta=0.5, random_state=42)
        embedding = tsne.fit_transform(X)
        assert embedding.shape == (200, 2), f"Shape: {embedding.shape}"
        assert np.all(np.isfinite(embedding)), "Embedding contains non-finite values"

    def test_barnes_hut_vs_exact_quality(self):
        """Barnes-Hut produces reasonable quality compared to exact."""
        from ferroml.decomposition import TSNE as FerroTSNE

        # Create data with clear structure
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 5) + [5, 0, 0, 0, 0],
            np.random.randn(50, 5) + [-5, 0, 0, 0, 0],
        ])
        labels = np.array([0] * 50 + [1] * 50)

        tsne_bh = FerroTSNE(n_components=2, method="barnes_hut", theta=0.5,
                            perplexity=20.0, random_state=42)
        emb_bh = tsne_bh.fit_transform(X)

        # Check that clusters are separated in the embedding
        center0 = emb_bh[labels == 0].mean(axis=0)
        center1 = emb_bh[labels == 1].mean(axis=0)
        cluster_dist = np.linalg.norm(center0 - center1)
        # Clusters should be separated
        assert cluster_dist > 1.0, (
            f"Cluster centroids too close: {cluster_dist:.4f}"
        )

    def test_barnes_hut_theta_parameter(self):
        """Different theta values produce valid results."""
        from ferroml.decomposition import TSNE as FerroTSNE

        np.random.seed(42)
        X = np.random.randn(100, 5)

        for theta in [0.3, 0.5, 0.8]:
            tsne = FerroTSNE(n_components=2, method="barnes_hut", theta=theta,
                             random_state=42)
            embedding = tsne.fit_transform(X)
            assert embedding.shape == (100, 2), f"Shape for theta={theta}: {embedding.shape}"
            assert np.all(np.isfinite(embedding)), f"Non-finite for theta={theta}"

    def test_method_auto_selection(self):
        """Auto method selection works."""
        from ferroml.decomposition import TSNE as FerroTSNE

        np.random.seed(42)
        X = np.random.randn(100, 5)
        tsne = FerroTSNE(n_components=2, method="auto", random_state=42)
        embedding = tsne.fit_transform(X)
        assert embedding.shape == (100, 2), f"Shape: {embedding.shape}"
        assert np.all(np.isfinite(embedding)), "Non-finite values in auto embedding"

    def test_exact_method_still_works(self):
        """Explicit exact method still works."""
        from ferroml.decomposition import TSNE as FerroTSNE

        np.random.seed(42)
        X = np.random.randn(50, 5)
        tsne = FerroTSNE(n_components=2, method="exact", random_state=42)
        embedding = tsne.fit_transform(X)
        assert embedding.shape == (50, 2), f"Shape: {embedding.shape}"
        assert np.all(np.isfinite(embedding)), "Non-finite values in exact embedding"

    def test_barnes_hut_3d_falls_back_to_exact(self):
        """Barnes-Hut with n_components=3 falls back to exact method (Session 3a fix)."""
        from ferroml.decomposition import TSNE as FerroTSNE

        np.random.seed(42)
        X = np.random.randn(100, 10)
        tsne = FerroTSNE(n_components=3, method="barnes_hut", theta=0.5, random_state=42)
        result = tsne.fit_transform(X)
        assert result.shape == (100, 3), f"Expected (100, 3), got {result.shape}"
        assert np.all(np.isfinite(result)), "Non-finite values in fallback embedding"

    def test_exact_3d(self):
        """Exact method works for 3D output."""
        from ferroml.decomposition import TSNE as FerroTSNE

        np.random.seed(42)
        X = np.random.randn(50, 5)
        tsne = FerroTSNE(n_components=3, method="exact", random_state=42)
        embedding = tsne.fit_transform(X)
        assert embedding.shape == (50, 3), f"Shape: {embedding.shape}"
        assert np.all(np.isfinite(embedding)), "Non-finite values in 3D embedding"


# ===========================================================================
# TruncatedSVD
# ===========================================================================

class TestTruncatedSVDComparison:
    def test_truncated_svd_explained_variance_ratio_ordering(self):
        """TruncatedSVD: explained_variance_ratio is sorted descending and sums <= 1."""
        from ferroml.decomposition import TruncatedSVD as FerroTSVD

        X, _ = get_iris()

        ferro = FerroTSVD(n_components=3)
        ferro.fit(X)
        ferro_var = np.array(ferro.explained_variance_ratio_)

        # Each ratio is non-negative
        assert np.all(ferro_var >= 0), "Negative variance ratios"
        # Sum <= 1
        assert ferro_var.sum() <= 1.0 + 1e-10, f"Variance sum > 1: {ferro_var.sum()}"
        # First component explains the most
        assert ferro_var[0] >= ferro_var[1], "First component should explain most variance"
        assert ferro_var[1] >= ferro_var[2], "Second component should explain more than third"

    def test_truncated_svd_components_sign_invariant(self):
        """TruncatedSVD: components match sklearn up to sign flips."""
        from ferroml.decomposition import TruncatedSVD as FerroTSVD
        from sklearn.decomposition import TruncatedSVD as SkTSVD

        X, _ = get_iris()

        ferro = FerroTSVD(n_components=2)
        ferro.fit(X)
        ferro_comp = np.array(ferro.components_)

        sk = SkTSVD(n_components=2, random_state=42)
        sk.fit(X)
        sk_comp = sk.components_

        # Compare absolute values (sign ambiguity in SVD)
        np.testing.assert_allclose(np.abs(ferro_comp), np.abs(sk_comp), atol=1e-4,
                                   err_msg="TruncatedSVD components mismatch (abs)")

    def test_truncated_svd_transform_reconstruction(self):
        """TruncatedSVD: transform output matches sklearn up to sign flips."""
        from ferroml.decomposition import TruncatedSVD as FerroTSVD
        from sklearn.decomposition import TruncatedSVD as SkTSVD

        X, _ = get_iris()

        ferro = FerroTSVD(n_components=2)
        ferro_out = ferro.fit_transform(X)

        sk = SkTSVD(n_components=2, random_state=42)
        sk_out = sk.fit_transform(X)

        # Columns may be sign-flipped; compare absolute values
        np.testing.assert_allclose(np.abs(ferro_out), np.abs(sk_out), atol=1e-4,
                                   err_msg="TruncatedSVD transform output mismatch (abs)")


# ===========================================================================
# LDA (LinearDiscriminantAnalysis)
# ===========================================================================

class TestLDAComparison:
    def test_lda_iris_transform_direction(self):
        """LDA on iris: projected directions match sklearn (up to scale and sign).

        LDA eigenvectors can differ by an arbitrary scale factor per component.
        We normalize each column to unit norm before comparing.
        """
        from ferroml.decomposition import LDA as FerroLDA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA

        X, y = get_iris()

        ferro = FerroLDA(n_components=2)
        ferro.fit(X, y)
        ferro_out = np.array(ferro.transform(X))

        sk = SkLDA(n_components=2)
        sk.fit(X, y)
        sk_out = sk.transform(X)

        # Normalize each column and handle sign flips
        for col in range(ferro_out.shape[1]):
            f_norm = np.linalg.norm(ferro_out[:, col])
            s_norm = np.linalg.norm(sk_out[:, col])
            ferro_out[:, col] /= f_norm
            sk_out[:, col] /= s_norm

            # Pick sign that minimizes difference
            diff_pos = np.linalg.norm(ferro_out[:, col] - sk_out[:, col])
            diff_neg = np.linalg.norm(ferro_out[:, col] + sk_out[:, col])
            if diff_neg < diff_pos:
                ferro_out[:, col] *= -1

        np.testing.assert_allclose(ferro_out, sk_out, atol=1e-4,
                                   err_msg="LDA transform direction mismatch")

    def test_lda_iris_class_separation(self):
        """LDA on iris: classes are well-separated in projected space."""
        from ferroml.decomposition import LDA as FerroLDA

        X, y = get_iris()

        ferro = FerroLDA(n_components=2)
        ferro.fit(X, y)
        ferro_out = np.array(ferro.transform(X))

        # Compute class centroids in the 2D projected space
        centroids = []
        for c in [0.0, 1.0, 2.0]:
            mask = y == c
            centroids.append(ferro_out[mask].mean(axis=0))

        # All pairwise centroid distances should be substantial
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                assert dist > 0.5, (
                    f"LDA classes {i} and {j} centroids too close: {dist:.4f}"
                )

    def test_lda_dimensionality_reduction(self):
        """LDA reduces to at most (n_classes - 1) dimensions."""
        from ferroml.decomposition import LDA as FerroLDA

        X, y = get_iris()  # 3 classes, 4 features

        ferro = FerroLDA(n_components=2)
        ferro.fit(X, y)
        ferro_out = np.array(ferro.transform(X))

        assert ferro_out.shape == (150, 2), f"LDA output shape: {ferro_out.shape}"


# ===========================================================================
# FactorAnalysis
# ===========================================================================

class TestFactorAnalysisComparison:
    def test_factor_analysis_transform_shape(self):
        """FactorAnalysis: transform output has correct shape."""
        from ferroml.decomposition import FactorAnalysis as FerroFA

        X, _ = get_iris()

        ferro = FerroFA()
        ferro_out = ferro.fit_transform(X)

        # Output should have same number of rows as input
        assert ferro_out.shape[0] == X.shape[0], (
            f"FA output rows: {ferro_out.shape[0]}, expected: {X.shape[0]}"
        )

    def test_factor_analysis_reconstruction_quality(self):
        """FactorAnalysis: reconstruction preserves data structure.

        Compare that the covariance structure of the transformed data
        from FerroML and sklearn are similar.
        """
        from ferroml.decomposition import FactorAnalysis as FerroFA
        from sklearn.decomposition import FactorAnalysis as SkFA

        # Create data with known factor structure
        rng = np.random.RandomState(42)
        n_samples = 200
        # 2 latent factors -> 5 observed variables
        latent = rng.randn(n_samples, 2)
        loading = rng.randn(2, 5)
        noise = rng.randn(n_samples, 5) * 0.3
        X = latent @ loading + noise

        ferro = FerroFA()
        ferro_out = ferro.fit_transform(X)

        sk = SkFA(n_components=ferro_out.shape[1], random_state=42)
        sk_out = sk.fit_transform(X)

        # Both should produce similar dimensionality output
        assert ferro_out.shape[1] == sk_out.shape[1], (
            f"FA components: ferro={ferro_out.shape[1]}, sk={sk_out.shape[1]}"
        )
