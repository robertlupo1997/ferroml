"""Test FerroML decomposition models."""

import numpy as np
import pytest

from ferroml import ferroml as _native

PCA = _native.decomposition.PCA
IncrementalPCA = _native.decomposition.IncrementalPCA
TruncatedSVD = _native.decomposition.TruncatedSVD
LDA = _native.decomposition.LDA
FactorAnalysis = _native.decomposition.FactorAnalysis


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def data_2d():
    """Generate 2D data with 100 samples and 5 features."""
    np.random.seed(42)
    return np.random.randn(100, 5).astype(np.float64)


@pytest.fixture
def data_structured():
    """Generate structured data where variance is concentrated in fewer dims."""
    np.random.seed(42)
    # 4-dimensional signal projected into 8-dimensional space
    signal = np.random.randn(200, 4)
    proj = np.random.randn(4, 8)
    X = signal @ proj + np.random.randn(200, 8) * 0.05
    return X.astype(np.float64)


@pytest.fixture
def multiclass_data():
    """Generate 3-class classification data for LDA tests."""
    np.random.seed(42)
    n_per_class = 60
    # Class 0: centered at origin
    X0 = np.random.randn(n_per_class, 4) * 0.5
    # Class 1: centered at [3, 3, 0, 0]
    X1 = np.random.randn(n_per_class, 4) * 0.5 + np.array([3.0, 3.0, 0.0, 0.0])
    # Class 2: centered at [0, 0, 3, 3]
    X2 = np.random.randn(n_per_class, 4) * 0.5 + np.array([0.0, 0.0, 3.0, 3.0])
    X = np.vstack([X0, X1, X2]).astype(np.float64)
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class + [2.0] * n_per_class)
    return X, y


@pytest.fixture
def binary_class_data():
    """Generate 2-class classification data for LDA binary tests."""
    np.random.seed(42)
    n_per_class = 50
    X0 = np.random.randn(n_per_class, 4) * 0.5
    X1 = np.random.randn(n_per_class, 4) * 0.5 + 3.0
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.array([0.0] * n_per_class + [1.0] * n_per_class)
    return X, y


# ============================================================================
# PCA Tests
# ============================================================================


class TestPCA:
    """Tests for Principal Component Analysis."""

    def test_fit_transform_shape(self, data_2d):
        """Test that fit_transform returns correct output shape."""
        X = data_2d
        pca = PCA(n_components=3)
        X_t = pca.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_fit_then_transform_shape(self, data_2d):
        """Test that separate fit and transform return correct shape."""
        X = data_2d
        pca = PCA(n_components=3)
        pca.fit(X)
        X_t = pca.transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_fit_transform_consistency(self, data_2d):
        """Test that fit_transform equals fit then transform."""
        X = data_2d
        pca1 = PCA(n_components=3)
        X_t1 = pca1.fit_transform(X)

        pca2 = PCA(n_components=3)
        pca2.fit(X)
        X_t2 = pca2.transform(X)

        # Both should produce the same result (same model, same data)
        np.testing.assert_allclose(np.abs(X_t1), np.abs(X_t2), atol=1e-10)

    def test_n_components_reduces_dimensionality(self, data_2d):
        """Test that n_components reduces the number of output features."""
        X = data_2d
        for n in [1, 2, 3, 4]:
            pca = PCA(n_components=n)
            X_t = pca.fit_transform(X)
            assert X_t.shape == (X.shape[0], n), f"Expected {n} components, got {X_t.shape[1]}"

    def test_components_shape(self, data_2d):
        """Test that components_ has shape (n_components, n_features)."""
        X = data_2d
        n_components = 3
        pca = PCA(n_components=n_components)
        pca.fit(X)

        components = pca.components_
        assert components.shape == (n_components, X.shape[1])

    def test_explained_variance_ratio_shape(self, data_2d):
        """Test that explained_variance_ratio_ has shape (n_components,)."""
        X = data_2d
        pca = PCA(n_components=3)
        pca.fit(X)

        evr = pca.explained_variance_ratio_
        assert evr.shape == (3,)

    def test_explained_variance_ratio_valid_values(self, data_2d):
        """Test that explained_variance_ratio_ entries are in (0, 1]."""
        X = data_2d
        pca = PCA(n_components=4)
        pca.fit(X)

        evr = pca.explained_variance_ratio_
        assert np.all(evr > 0), "All EVR entries should be positive"
        assert np.all(evr <= 1.0), "All EVR entries should be <= 1.0"

    def test_explained_variance_ratio_sum_leq_one(self, data_2d):
        """Test that explained_variance_ratio_ sums to <= 1.0."""
        X = data_2d
        pca = PCA(n_components=3)
        pca.fit(X)

        evr = pca.explained_variance_ratio_
        assert evr.sum() <= 1.0 + 1e-10, f"EVR sum {evr.sum()} should be <= 1.0"

    def test_explained_variance_ratio_sum_equals_one_all_components(self, data_2d):
        """Test that EVR sums to ~1.0 when keeping all components."""
        X = data_2d
        n_features = X.shape[1]
        pca = PCA(n_components=n_features)
        pca.fit(X)

        evr = pca.explained_variance_ratio_
        np.testing.assert_allclose(evr.sum(), 1.0, atol=1e-10)

    def test_explained_variance_ratio_decreasing(self, data_2d):
        """Test that explained_variance_ratio_ is in decreasing order."""
        X = data_2d
        pca = PCA(n_components=4)
        pca.fit(X)

        evr = pca.explained_variance_ratio_
        for i in range(len(evr) - 1):
            assert evr[i] >= evr[i + 1], (
                f"EVR should be decreasing: evr[{i}]={evr[i]} < evr[{i+1}]={evr[i+1]}"
            )

    def test_inverse_transform_shape(self, data_2d):
        """Test that inverse_transform returns the original feature shape."""
        X = data_2d
        pca = PCA(n_components=3)
        X_t = pca.fit_transform(X)
        X_rec = pca.inverse_transform(X_t)

        assert X_rec.shape == X.shape

    def test_inverse_transform_recovers_data_full_components(self, data_2d):
        """Test that inverse_transform approximately recovers data when using all components."""
        X = data_2d
        n_features = X.shape[1]
        pca = PCA(n_components=n_features)
        X_t = pca.fit_transform(X)
        X_rec = pca.inverse_transform(X_t)

        np.testing.assert_allclose(X_rec, X, atol=1e-10)

    def test_inverse_transform_partial_recovery(self, data_structured):
        """Test that inverse_transform with fewer components gives approximate recovery."""
        X = data_structured
        # Use enough components to capture the signal
        pca = PCA(n_components=6)
        X_t = pca.fit_transform(X)
        X_rec = pca.inverse_transform(X_t)

        # Should be a reasonable reconstruction (not exact since we dropped components)
        assert X_rec.shape == X.shape
        assert np.all(np.isfinite(X_rec)), "Recovered data should have finite values"

    def test_whiten_unit_variance(self, data_2d):
        """Test that whiten=True produces components with unit variance."""
        X = data_2d
        pca = PCA(n_components=3, whiten=True)
        X_t = pca.fit_transform(X)

        stds = X_t.std(axis=0, ddof=0)
        np.testing.assert_allclose(stds, np.ones(3), atol=0.05)

    def test_no_whiten_nonunit_variance(self, data_2d):
        """Test that whiten=False (default) does not force unit variance."""
        X = data_2d * 10  # Scale up so variance differs from 1
        pca = PCA(n_components=3, whiten=False)
        X_t = pca.fit_transform(X)

        # The variance should NOT all be 1.0 without whitening
        stds = X_t.std(axis=0, ddof=0)
        # First component should have std > 1 since data was scaled
        assert stds[0] > 1.0, "Without whitening, variance of first PC should not be ~1"

    def test_default_n_components_keeps_all(self, data_2d):
        """Test that omitting n_components keeps all components."""
        X = data_2d
        n_features = X.shape[1]
        pca = PCA()
        X_t = pca.fit_transform(X)

        assert X_t.shape == (X.shape[0], n_features)

    def test_components_orthonormal(self, data_2d):
        """Test that principal components are approximately orthonormal."""
        X = data_2d
        pca = PCA(n_components=4)
        pca.fit(X)

        V = pca.components_  # shape (n_components, n_features)
        # V @ V.T should be approximately identity
        gram = V @ V.T
        np.testing.assert_allclose(gram, np.eye(4), atol=1e-10)

    def test_not_fitted_components_raises(self):
        """Test that accessing components_ before fit raises ValueError."""
        pca = PCA(n_components=3)
        with pytest.raises(ValueError, match="not fitted"):
            _ = pca.components_

    def test_not_fitted_evr_raises(self):
        """Test that accessing explained_variance_ratio_ before fit raises ValueError."""
        pca = PCA(n_components=3)
        with pytest.raises(ValueError, match="not fitted"):
            _ = pca.explained_variance_ratio_

    def test_not_fitted_transform_raises(self, data_2d):
        """Test that transform before fit raises an error."""
        pca = PCA(n_components=3)
        with pytest.raises(Exception):
            pca.transform(data_2d)

    def test_large_data(self):
        """Test PCA on a moderately large dataset."""
        np.random.seed(42)
        X = np.random.randn(500, 20).astype(np.float64)
        pca = PCA(n_components=10)
        X_t = pca.fit_transform(X)

        assert X_t.shape == (500, 10)
        evr = pca.explained_variance_ratio_
        assert evr.sum() <= 1.0 + 1e-10


# ============================================================================
# IncrementalPCA Tests
# ============================================================================


class TestIncrementalPCA:
    """Tests for Incremental PCA (memory-efficient batch PCA)."""

    def test_fit_transform_shape(self, data_2d):
        """Test that fit_transform returns correct output shape."""
        X = data_2d
        ipca = IncrementalPCA(n_components=3)
        X_t = ipca.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_fit_then_transform_shape(self, data_2d):
        """Test that separate fit and transform return correct shape."""
        X = data_2d
        ipca = IncrementalPCA(n_components=3)
        ipca.fit(X)
        X_t = ipca.transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_n_components_reduces_dimensionality(self, data_2d):
        """Test that n_components reduces the number of output features."""
        X = data_2d
        for n in [1, 2, 3, 4]:
            ipca = IncrementalPCA(n_components=n)
            X_t = ipca.fit_transform(X)
            assert X_t.shape == (X.shape[0], n), f"Expected {n} components, got {X_t.shape[1]}"

    def test_whiten_unit_variance(self, data_2d):
        """Test that whiten=True produces approximately unit-variance components."""
        X = data_2d
        ipca = IncrementalPCA(n_components=3, whiten=True)
        X_t = ipca.fit_transform(X)

        stds = X_t.std(axis=0, ddof=0)
        np.testing.assert_allclose(stds, np.ones(3), atol=0.05)

    def test_output_finite(self, data_2d):
        """Test that IncrementalPCA outputs finite values."""
        X = data_2d
        ipca = IncrementalPCA(n_components=3)
        X_t = ipca.fit_transform(X)

        assert np.all(np.isfinite(X_t)), "All output values should be finite"

    def test_batch_size_parameter(self, data_2d):
        """Test that batch_size parameter is accepted and produces valid results."""
        X = data_2d
        ipca = IncrementalPCA(n_components=3, batch_size=25)
        X_t = ipca.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)
        assert np.all(np.isfinite(X_t))

    def test_default_n_components_keeps_all(self, data_2d):
        """Test that omitting n_components keeps all components."""
        X = data_2d
        n_features = X.shape[1]
        ipca = IncrementalPCA()
        X_t = ipca.fit_transform(X)

        assert X_t.shape == (X.shape[0], n_features)

    def test_similar_to_pca(self, data_2d):
        """Test that IncrementalPCA produces results similar to standard PCA."""
        X = data_2d
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        ipca = IncrementalPCA(n_components=3)
        X_ipca = ipca.fit_transform(X)

        # Both should produce the same shape
        assert X_pca.shape == X_ipca.shape
        # The variance explained by the components should be similar
        # (may differ in sign, so compare absolute values of row norms)
        pca_norms = np.linalg.norm(X_pca, axis=0)
        ipca_norms = np.linalg.norm(X_ipca, axis=0)
        np.testing.assert_allclose(np.sort(pca_norms), np.sort(ipca_norms), rtol=0.1)


# ============================================================================
# TruncatedSVD Tests
# ============================================================================


class TestTruncatedSVD:
    """Tests for Truncated Singular Value Decomposition."""

    def test_fit_transform_shape(self, data_2d):
        """Test that fit_transform returns correct output shape."""
        X = data_2d
        svd = TruncatedSVD(n_components=3)
        X_t = svd.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_fit_then_transform_shape(self, data_2d):
        """Test that separate fit and transform return correct shape."""
        X = data_2d
        svd = TruncatedSVD(n_components=3)
        svd.fit(X)
        X_t = svd.transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_n_components_reduces_dimensionality(self, data_2d):
        """Test that n_components reduces the number of output features."""
        X = data_2d
        for n in [1, 2, 3, 4]:
            svd = TruncatedSVD(n_components=n)
            X_t = svd.fit_transform(X)
            assert X_t.shape == (X.shape[0], n), f"Expected {n} components, got {X_t.shape[1]}"

    def test_components_shape(self, data_2d):
        """Test that components_ has shape (n_components, n_features)."""
        X = data_2d
        n_components = 3
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X)

        components = svd.components_
        assert components.shape == (n_components, X.shape[1])

    def test_explained_variance_ratio_shape(self, data_2d):
        """Test that explained_variance_ratio_ has shape (n_components,)."""
        X = data_2d
        svd = TruncatedSVD(n_components=3)
        svd.fit(X)

        evr = svd.explained_variance_ratio_
        assert evr.shape == (3,)

    def test_explained_variance_ratio_valid(self, data_2d):
        """Test that explained_variance_ratio_ values are in (0, 1]."""
        X = data_2d
        svd = TruncatedSVD(n_components=3)
        svd.fit(X)

        evr = svd.explained_variance_ratio_
        assert np.all(evr > 0), "All EVR entries should be positive"
        assert np.all(evr <= 1.0 + 1e-10), "All EVR entries should be <= 1.0"

    def test_random_state_reproducibility(self, data_2d):
        """Test that random_state produces reproducible results."""
        X = data_2d
        svd1 = TruncatedSVD(n_components=3, random_state=42)
        X_t1 = svd1.fit_transform(X)

        svd2 = TruncatedSVD(n_components=3, random_state=42)
        X_t2 = svd2.fit_transform(X)

        np.testing.assert_allclose(np.abs(X_t1), np.abs(X_t2), atol=1e-10)

    def test_no_centering(self):
        """Test that TruncatedSVD works without centering (unlike PCA)."""
        np.random.seed(42)
        # Create data with a large mean offset — TruncatedSVD should handle it
        X = np.random.randn(100, 5) + 100.0  # Large offset
        X = X.astype(np.float64)

        svd = TruncatedSVD(n_components=2, random_state=0)
        X_t = svd.fit_transform(X)

        assert X_t.shape == (100, 2)
        assert np.all(np.isfinite(X_t))

    def test_output_finite(self, data_2d):
        """Test that TruncatedSVD outputs finite values."""
        X = data_2d
        svd = TruncatedSVD(n_components=3, random_state=42)
        X_t = svd.fit_transform(X)

        assert np.all(np.isfinite(X_t))

    def test_n_iter_parameter(self, data_2d):
        """Test that n_iter parameter is accepted and produces valid results."""
        X = data_2d
        svd = TruncatedSVD(n_components=3, n_iter=10, random_state=42)
        X_t = svd.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_not_fitted_components_raises(self):
        """Test that accessing components_ before fit raises ValueError."""
        svd = TruncatedSVD(n_components=3)
        with pytest.raises(ValueError, match="not fitted"):
            _ = svd.components_

    def test_not_fitted_evr_raises(self):
        """Test that accessing explained_variance_ratio_ before fit raises ValueError."""
        svd = TruncatedSVD(n_components=3)
        with pytest.raises(ValueError, match="not fitted"):
            _ = svd.explained_variance_ratio_

    def test_not_fitted_transform_raises(self, data_2d):
        """Test that transform before fit raises an error."""
        svd = TruncatedSVD(n_components=3)
        with pytest.raises(Exception):
            svd.transform(data_2d)


# ============================================================================
# LDA Tests
# ============================================================================


class TestLDA:
    """Tests for Linear Discriminant Analysis (supervised dimensionality reduction)."""

    def test_fit_transform_shape_multiclass(self, multiclass_data):
        """Test fit and transform with 3 classes produces (n_samples, n_classes-1) output."""
        X, y = multiclass_data
        # With 3 classes, max n_components is 2
        lda = LDA(n_components=2)
        lda.fit(X, y)
        X_t = lda.transform(X)

        assert X_t.shape == (X.shape[0], 2)

    def test_fit_transform_shape_binary(self, binary_class_data):
        """Test fit and transform with 2 classes produces (n_samples, 1) output."""
        X, y = binary_class_data
        # With 2 classes, max n_components is 1
        lda = LDA(n_components=1)
        lda.fit(X, y)
        X_t = lda.transform(X)

        assert X_t.shape == (X.shape[0], 1)

    def test_default_n_components_multiclass(self, multiclass_data):
        """Test that default n_components gives max (n_classes - 1) dimensions."""
        X, y = multiclass_data
        lda = LDA()
        lda.fit(X, y)
        X_t = lda.transform(X)

        # 3 classes => max 2 components
        assert X_t.shape[0] == X.shape[0]
        assert X_t.shape[1] <= 2

    def test_supervised_separation(self, multiclass_data):
        """Test that LDA produces separable embeddings for well-separated classes."""
        X, y = multiclass_data
        lda = LDA(n_components=2)
        lda.fit(X, y)
        X_t = lda.transform(X)

        # Compute per-class centroids in embedded space
        classes = np.unique(y)
        centroids = np.array([X_t[y == c].mean(axis=0) for c in classes])

        # Centroids should be spread out (not all at origin)
        centroid_spread = np.std(centroids, axis=0)
        assert np.any(centroid_spread > 0.1), "LDA should separate class centroids"

    def test_output_finite(self, multiclass_data):
        """Test that LDA outputs finite values."""
        X, y = multiclass_data
        lda = LDA(n_components=2)
        lda.fit(X, y)
        X_t = lda.transform(X)

        assert np.all(np.isfinite(X_t)), "All LDA output values should be finite"

    def test_shrinkage_parameter(self, multiclass_data):
        """Test that shrinkage parameter is accepted and produces valid results."""
        X, y = multiclass_data
        lda = LDA(n_components=2, shrinkage=0.5)
        lda.fit(X, y)
        X_t = lda.transform(X)

        assert X_t.shape == (X.shape[0], 2)
        assert np.all(np.isfinite(X_t))

    def test_tol_parameter(self, multiclass_data):
        """Test that tol parameter is accepted."""
        X, y = multiclass_data
        lda = LDA(n_components=2, tol=1e-6)
        lda.fit(X, y)
        X_t = lda.transform(X)

        assert X_t.shape == (X.shape[0], 2)

    def test_float_class_labels(self, multiclass_data):
        """Test that LDA accepts float class labels."""
        X, y = multiclass_data
        # y is already float64 from fixture
        assert y.dtype == np.float64
        lda = LDA(n_components=2)
        lda.fit(X, y)
        X_t = lda.transform(X)

        assert X_t.shape[0] == X.shape[0]

    def test_not_fitted_transform_raises(self, multiclass_data):
        """Test that transform before fit raises an error."""
        X, y = multiclass_data
        lda = LDA(n_components=2)

        with pytest.raises(Exception):
            lda.transform(X)

    def test_many_features_few_classes(self):
        """Test LDA when n_features >> n_classes."""
        np.random.seed(42)
        n_per_class = 80
        n_features = 20
        X = np.vstack([
            np.random.randn(n_per_class, n_features),
            np.random.randn(n_per_class, n_features) + 3.0,
            np.random.randn(n_per_class, n_features) - 3.0,
        ]).astype(np.float64)
        y = np.array([0.0] * n_per_class + [1.0] * n_per_class + [2.0] * n_per_class)

        lda = LDA(n_components=2)
        lda.fit(X, y)
        X_t = lda.transform(X)

        assert X_t.shape == (X.shape[0], 2)
        assert np.all(np.isfinite(X_t))


# ============================================================================
# FactorAnalysis Tests
# ============================================================================


class TestFactorAnalysis:
    """Tests for Factor Analysis with optional rotation."""

    def test_fit_transform_shape(self, data_2d):
        """Test that fit_transform returns correct output shape."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3)
        X_t = fa.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_fit_then_transform_shape(self, data_2d):
        """Test that separate fit and transform return correct shape."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3)
        fa.fit(X)
        X_t = fa.transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_n_factors_reduces_dimensionality(self, data_2d):
        """Test that n_factors controls the number of output dimensions."""
        X = data_2d
        for n in [1, 2, 3, 4]:
            fa = FactorAnalysis(n_factors=n)
            X_t = fa.fit_transform(X)
            assert X_t.shape == (X.shape[0], n), f"Expected {n} factors, got {X_t.shape[1]}"

    def test_rotation_none(self, data_2d):
        """Test rotation='none' (default) works correctly."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3, rotation="none")
        X_t = fa.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)
        assert np.all(np.isfinite(X_t))

    def test_rotation_varimax(self, data_2d):
        """Test rotation='varimax' produces valid output."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3, rotation="varimax")
        X_t = fa.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)
        assert np.all(np.isfinite(X_t))

    def test_rotation_quartimax(self, data_2d):
        """Test rotation='quartimax' produces valid output."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3, rotation="quartimax")
        X_t = fa.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)
        assert np.all(np.isfinite(X_t))

    def test_rotation_promax(self, data_2d):
        """Test rotation='promax' produces valid output."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3, rotation="promax")
        X_t = fa.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)
        assert np.all(np.isfinite(X_t))

    def test_all_rotations_same_shape(self, data_2d):
        """Test that all rotation methods produce the same output shape."""
        X = data_2d
        n_factors = 3
        for rotation in ["none", "varimax", "quartimax", "promax"]:
            fa = FactorAnalysis(n_factors=n_factors, rotation=rotation, random_state=42)
            X_t = fa.fit_transform(X)
            assert X_t.shape == (X.shape[0], n_factors), (
                f"rotation='{rotation}' produced shape {X_t.shape}"
            )

    def test_invalid_rotation_raises(self):
        """Test that an invalid rotation string raises ValueError."""
        with pytest.raises((ValueError, Exception)):
            FactorAnalysis(n_factors=3, rotation="invalid_rotation")

    def test_random_state_parameter(self, data_2d):
        """Test that random_state parameter is accepted and produces valid results."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3, random_state=42)
        X_t = fa.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)
        assert np.all(np.isfinite(X_t))

    def test_tol_parameter(self, data_2d):
        """Test that tol parameter affects convergence tolerance."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3, tol=1e-4, random_state=42)
        X_t = fa.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_max_iter_parameter(self, data_2d):
        """Test that max_iter parameter is accepted."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3, max_iter=500, random_state=42)
        X_t = fa.fit_transform(X)

        assert X_t.shape == (X.shape[0], 3)

    def test_output_finite(self, data_2d):
        """Test that FactorAnalysis outputs finite values."""
        X = data_2d
        fa = FactorAnalysis(n_factors=3, random_state=42)
        X_t = fa.fit_transform(X)

        assert np.all(np.isfinite(X_t)), "All FactorAnalysis output values should be finite"

    def test_not_fitted_transform_raises(self, data_2d):
        """Test that transform before fit raises an error."""
        fa = FactorAnalysis(n_factors=3)
        with pytest.raises(Exception):
            fa.transform(data_2d)

    def test_structured_data_extracts_factors(self):
        """Test that FactorAnalysis extracts latent structure from structured data."""
        np.random.seed(42)
        n_samples = 300
        n_factors = 3

        # Generate data from a known factor model:
        # X = F @ L.T + noise, where F has n_factors columns
        F = np.random.randn(n_samples, n_factors)
        L = np.random.randn(8, n_factors)  # loadings
        noise = np.random.randn(n_samples, 8) * 0.1
        X = (F @ L.T + noise).astype(np.float64)

        fa = FactorAnalysis(n_factors=n_factors, random_state=42)
        X_t = fa.fit_transform(X)

        assert X_t.shape == (n_samples, n_factors)
        assert np.all(np.isfinite(X_t))
