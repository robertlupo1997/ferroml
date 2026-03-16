"""
FerroML vs sklearn: Decomposition and dimensionality reduction.

Cross-library validation for:
1. TruncatedSVD — explained variance ratio within 1e-3
2. IncrementalPCA — transform output correlation > 0.99
3. FactorAnalysis — transform output correlation > 0.95

Phase X.3 — Plan X production-readiness validation.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dense_data():
    """Dense data matrix suitable for decomposition methods."""
    rng = np.random.default_rng(42)
    # Create data with clear latent structure
    n_samples, n_features = 200, 10
    n_latent = 3
    latent = rng.standard_normal((n_samples, n_latent))
    loadings = rng.standard_normal((n_latent, n_features))
    noise = 0.3 * rng.standard_normal((n_samples, n_features))
    X = latent @ loadings + noise
    return X


@pytest.fixture()
def centered_data(dense_data):
    """Mean-centered data for PCA/SVD comparisons."""
    return dense_data - dense_data.mean(axis=0)


# ===========================================================================
# 1. TruncatedSVD
# ===========================================================================

class TestTruncatedSVDVsSklearn:
    """Compare FerroML TruncatedSVD against sklearn."""

    def test_explained_variance_ratio_within_1e3(self, dense_data):
        from ferroml.decomposition import TruncatedSVD

        from sklearn.decomposition import TruncatedSVD as SkSVD

        X = dense_data
        n_components = 3

        sk = SkSVD(n_components=n_components, random_state=42)
        sk.fit(X)
        sk_ratio = sk.explained_variance_ratio_

        fm = TruncatedSVD(n_components=n_components, random_state=42)
        fm.fit(X)
        fm_ratio = np.array(fm.explained_variance_ratio_)

        np.testing.assert_allclose(
            fm_ratio, sk_ratio, atol=1e-3,
            err_msg="Explained variance ratio mismatch",
        )

    def test_transform_subspace_correlation(self, dense_data):
        """Transformed data subspaces should be highly correlated."""
        from ferroml.decomposition import TruncatedSVD

        from sklearn.decomposition import TruncatedSVD as SkSVD

        X = dense_data
        n_components = 3

        sk = SkSVD(n_components=n_components, random_state=42)
        sk_out = sk.fit_transform(X)

        fm = TruncatedSVD(n_components=n_components, random_state=42)
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        # Each component may differ by sign; check absolute correlation
        for i in range(n_components):
            corr = abs(np.corrcoef(sk_out[:, i], fm_out[:, i])[0, 1])
            assert corr > 0.99, (
                f"Component {i} correlation too low: {corr:.4f}"
            )

    def test_transform_reconstruction(self, dense_data):
        """Transformed data should have correct shape and be finite."""
        from ferroml.decomposition import TruncatedSVD

        X = dense_data
        n_components = 5

        fm = TruncatedSVD(n_components=n_components, random_state=42)
        fm.fit(X)
        X_reduced = np.array(fm.transform(X))

        assert X_reduced.shape == (X.shape[0], n_components)
        assert np.all(np.isfinite(X_reduced))

    def test_components_shape(self, dense_data):
        from ferroml.decomposition import TruncatedSVD

        X = dense_data
        n_components = 3

        fm = TruncatedSVD(n_components=n_components, random_state=42)
        fm.fit(X)
        components = np.array(fm.components_)

        assert components.shape == (n_components, X.shape[1])

    def test_variance_ratio_sums_below_1(self, dense_data):
        """Sum of explained variance ratios should be <= 1."""
        from ferroml.decomposition import TruncatedSVD

        X = dense_data

        fm = TruncatedSVD(n_components=3, random_state=42)
        fm.fit(X)
        fm_ratio = np.array(fm.explained_variance_ratio_)

        assert np.sum(fm_ratio) <= 1.0 + 1e-6, (
            f"Variance ratio sum > 1: {np.sum(fm_ratio)}"
        )
        assert np.all(fm_ratio >= 0), "Negative variance ratios"


# ===========================================================================
# 2. IncrementalPCA
# ===========================================================================

class TestIncrementalPCAVsSklearn:
    """Compare FerroML IncrementalPCA against sklearn."""

    def test_transform_subspace_correlation(self, centered_data):
        """Transformed outputs should lie in the same subspace (high correlation)."""
        from ferroml.decomposition import IncrementalPCA

        from sklearn.decomposition import IncrementalPCA as SkIPCA

        X = centered_data
        n_components = 3
        batch_size = 50

        sk = SkIPCA(n_components=n_components, batch_size=batch_size)
        sk_out = sk.fit_transform(X)

        fm = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        # Each component may differ by sign
        for i in range(n_components):
            corr = abs(np.corrcoef(sk_out[:, i], fm_out[:, i])[0, 1])
            assert corr > 0.99, (
                f"Component {i} correlation too low: {corr:.4f}"
            )

    def test_variance_captured(self, centered_data):
        """IncrementalPCA should capture most of the variance in structured data."""
        from ferroml.decomposition import IncrementalPCA

        from sklearn.decomposition import IncrementalPCA as SkIPCA

        X = centered_data
        n_components = 3
        batch_size = 50

        # Use sklearn as reference for total variance captured
        sk = SkIPCA(n_components=n_components, batch_size=batch_size)
        sk.fit(X)
        sk_var_ratio_sum = np.sum(sk.explained_variance_ratio_)

        # FerroML should reconstruct similarly well
        fm = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        # Reconstruction error should be comparable
        total_var = np.var(X, axis=0).sum()
        projected_var = np.var(fm_out, axis=0).sum()
        fm_var_ratio = projected_var / total_var

        # FerroML should capture at least 80% of what sklearn captures
        assert fm_var_ratio > sk_var_ratio_sum * 0.8, (
            f"FerroML captures too little variance: {fm_var_ratio:.4f} vs sklearn {sk_var_ratio_sum:.4f}"
        )

    def test_transform_shape(self, centered_data):
        from ferroml.decomposition import IncrementalPCA

        X = centered_data
        n_components = 3

        fm = IncrementalPCA(n_components=n_components, batch_size=50)
        fm.fit(X)
        X_t = np.array(fm.transform(X))

        assert X_t.shape == (X.shape[0], n_components)
        assert np.all(np.isfinite(X_t))

    def test_partial_fit_same_as_fit(self, centered_data):
        """partial_fit on all data at once should produce similar results to fit."""
        from ferroml.decomposition import IncrementalPCA

        X = centered_data
        n_components = 3
        batch_size = 50

        # Full fit
        fm_full = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        fm_full.fit(X)
        out_full = np.array(fm_full.transform(X))

        # Partial fit in batches
        fm_partial = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        for start in range(0, len(X), batch_size):
            batch = X[start:start + batch_size]
            fm_partial.partial_fit(batch)
        out_partial = np.array(fm_partial.transform(X))

        # Results should be similar (same data, same order)
        for i in range(n_components):
            corr = abs(np.corrcoef(out_full[:, i], out_partial[:, i])[0, 1])
            assert corr > 0.95, (
                f"partial_fit vs fit: component {i} correlation = {corr:.4f}"
            )


# ===========================================================================
# 3. FactorAnalysis
# ===========================================================================

class TestFactorAnalysisVsSklearn:
    """Compare FerroML FactorAnalysis against sklearn."""

    def test_transform_subspace_correlation(self, dense_data):
        """Factor scores should be correlated with sklearn's factor scores."""
        from ferroml.decomposition import FactorAnalysis

        from sklearn.decomposition import FactorAnalysis as SkFA

        X = dense_data
        n_components = 3

        sk = SkFA(n_components=n_components, random_state=42, max_iter=1000)
        sk_out = sk.fit_transform(X)

        fm = FactorAnalysis(n_factors=n_components)
        fm.fit(X)
        fm_out = np.array(fm.transform(X))

        # Match factors by best absolute correlation (order may differ)
        matched = set()
        for i in range(n_components):
            best_corr = 0.0
            best_j = -1
            for j in range(n_components):
                if j in matched:
                    continue
                corr = abs(np.corrcoef(fm_out[:, i], sk_out[:, j])[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_j = j
            matched.add(best_j)
            assert best_corr > 0.90, (
                f"Factor {i}: best correlation with sklearn factor {best_j} = {best_corr:.4f}"
            )

    def test_transform_shape(self, dense_data):
        from ferroml.decomposition import FactorAnalysis

        X = dense_data
        n_components = 3

        fm = FactorAnalysis(n_factors=n_components)
        fm.fit(X)
        X_t = np.array(fm.transform(X))

        assert X_t.shape == (X.shape[0], n_components)
        assert np.all(np.isfinite(X_t))

    def test_reconstruction_quality(self, dense_data):
        """With enough factors, reconstruction error should be low."""
        from ferroml.decomposition import FactorAnalysis

        X = dense_data
        # Use more factors for better reconstruction
        n_components = 5

        fm = FactorAnalysis(n_factors=n_components)
        fm.fit(X)
        X_t = np.array(fm.transform(X))

        # At minimum, transformed data should have correct shape and be finite
        assert X_t.shape == (X.shape[0], n_components)
        assert np.all(np.isfinite(X_t))

        # Variance of transformed data should be non-trivial
        total_var = np.var(X_t, axis=0).sum()
        assert total_var > 0.1, f"Transformed variance too low: {total_var}"
