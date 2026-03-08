"""Tests for GaussianMixture Python bindings."""

import numpy as np
import pytest

from ferroml.clustering import GaussianMixture


def make_blobs(centers, n_per_cluster=50, std=0.5, seed=42):
    """Generate well-separated Gaussian blobs."""
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []
    for i, (cx, cy) in enumerate(centers):
        X_list.append(rng.normal(loc=[cx, cy], scale=std, size=(n_per_cluster, 2)))
        y_list.append(np.full(n_per_cluster, i, dtype=np.int32))
    return np.vstack(X_list), np.concatenate(y_list)


class TestGaussianMixtureBasic:
    """Basic fit/predict tests."""

    def test_fit_predict(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        assert labels.shape == (100,)
        assert set(np.unique(labels)) == {0, 1}

    def test_fit_predict_single_call(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        labels = gmm.fit_predict(X)
        assert labels.shape == (100,)

    def test_predict_proba_shape(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        proba = gmm.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_predict_proba_sums_to_one(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        proba = gmm.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_nonneg(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        proba = gmm.predict_proba(X)
        assert np.all(proba >= 0)

    def test_confident_on_separated_data(self):
        X, _ = make_blobs([(0, 0), (20, 20)], std=0.3)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        proba = gmm.predict_proba(X)
        assert np.all(proba.max(axis=1) > 0.99)


class TestCovarianceTypes:
    """Test all four covariance types."""

    @pytest.mark.parametrize("cov_type", ["full", "tied", "diag", "spherical"])
    def test_covariance_type_fits(self, cov_type):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(
            n_components=2, covariance_type=cov_type, random_state=42
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        assert labels.shape == (100,)

    def test_invalid_covariance_type(self):
        with pytest.raises(ValueError, match="Unknown covariance_type"):
            GaussianMixture(n_components=2, covariance_type="invalid")


class TestModelSelection:
    """BIC/AIC tests."""

    def test_bic_selects_correct_k(self):
        X, _ = make_blobs([(0, 0), (10, 0), (5, 10)], n_per_cluster=40)
        bics = []
        for k in range(1, 6):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X)
            bics.append(gmm.bic(X))
        best_k = np.argmin(bics) + 1
        assert best_k == 3, f"BIC selected k={best_k}, expected k=3"

    def test_aic_k3_better_than_k1(self):
        """AIC with k=3 should be better than k=1 on 3-blob data."""
        X, _ = make_blobs([(0, 0), (10, 0), (5, 10)], n_per_cluster=40)
        gmm1 = GaussianMixture(n_components=1, random_state=42)
        gmm1.fit(X)
        gmm3 = GaussianMixture(n_components=3, random_state=42)
        gmm3.fit(X)
        assert gmm3.aic(X) < gmm1.aic(X)

    def test_bic_k2_better_than_k1(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm1 = GaussianMixture(n_components=1, random_state=42)
        gmm1.fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=42)
        gmm2.fit(X)
        assert gmm2.bic(X) < gmm1.bic(X)


class TestScoring:
    """Score and score_samples tests."""

    def test_score_samples_shape(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        scores = gmm.score_samples(X)
        assert scores.shape == (100,)
        assert np.all(np.isfinite(scores))

    def test_score_finite(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        score = gmm.score(X)
        assert np.isfinite(score)


class TestAttributes:
    """Test fitted attributes."""

    def test_weights_sum_to_one(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        np.testing.assert_allclose(gmm.weights_.sum(), 1.0, atol=1e-10)

    def test_means_shape(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        assert gmm.means_.shape == (2, 2)

    def test_means_recovered(self):
        X, _ = make_blobs([(0, 0), (10, 10)], n_per_cluster=100, std=0.3)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        means_sorted = gmm.means_[gmm.means_[:, 0].argsort()]
        np.testing.assert_allclose(means_sorted[0], [0, 0], atol=0.5)
        np.testing.assert_allclose(means_sorted[1], [10, 10], atol=0.5)

    def test_labels_shape(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        assert gmm.labels_.shape == (100,)

    def test_n_iter(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        assert gmm.n_iter_ >= 1

    def test_converged(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        assert gmm.converged_ is True

    def test_lower_bound(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        assert np.isfinite(gmm.lower_bound_)


class TestSample:
    """Test sample generation."""

    def test_sample_shape(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        samples, labels = gmm.sample(50)
        assert samples.shape == (50, 2)
        assert labels.shape == (50,)

    def test_sample_labels_valid(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        _, labels = gmm.sample(100)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_sample_finite(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)
        samples, _ = gmm.sample(100)
        assert np.all(np.isfinite(samples))


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_single_component(self):
        X, _ = make_blobs([(5, 5)], n_per_cluster=50)
        gmm = GaussianMixture(n_components=1, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        assert np.all(labels == 0)
        np.testing.assert_allclose(gmm.weights_, [1.0], atol=1e-10)

    def test_reproducibility(self):
        X, _ = make_blobs([(0, 0), (10, 10)])
        gmm1 = GaussianMixture(n_components=2, random_state=123)
        gmm1.fit(X)
        labels1 = gmm1.predict(X)

        gmm2 = GaussianMixture(n_components=2, random_state=123)
        gmm2.fit(X)
        labels2 = gmm2.predict(X)

        np.testing.assert_array_equal(labels1, labels2)

    def test_not_fitted_error(self):
        gmm = GaussianMixture(n_components=2)
        X = np.zeros((10, 2))
        with pytest.raises(RuntimeError):
            gmm.predict(X)

    def test_repr(self):
        gmm = GaussianMixture(n_components=3, random_state=42)
        assert "GaussianMixture" in repr(gmm)
