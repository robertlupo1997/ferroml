//! Truncated Singular Value Decomposition (TruncatedSVD)
//!
//! Dimensionality reduction using truncated SVD, also known as LSA (Latent Semantic Analysis)
//! when applied to term-document matrices.
//!
//! ## Mathematical Background
//!
//! TruncatedSVD computes the SVD of the data matrix X = U Σ V^T and keeps only the first k
//! components. Unlike PCA, TruncatedSVD does NOT center the data before computing the SVD.
//! This is crucial for sparse matrices where centering would destroy sparsity.
//!
//! ## Comparison with PCA
//!
//! | Aspect | PCA | TruncatedSVD |
//! |--------|-----|--------------|
//! | Centering | Yes (subtracts mean) | No |
//! | Sparse data | Not suitable | Suitable |
//! | Use case | Dense data | Sparse data, LSA/LSI |
//!
//! ## Features
//!
//! - Randomized SVD algorithm for efficiency with large datasets
//! - No data centering (preserves sparsity)
//! - Explained variance and variance ratio computation
//! - Component loadings for interpretability
//!
//! ## Example
//!
//! ```
//! use ferroml_core::decomposition::TruncatedSVD;
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! let mut svd = TruncatedSVD::new().with_n_components(2);
//! let x = array![
//!     [1.0, 0.0, 0.0, 1.0],
//!     [0.0, 1.0, 1.0, 0.0],
//!     [1.0, 1.0, 0.0, 1.0],
//!     [0.0, 0.0, 1.0, 1.0]
//! ];
//!
//! let x_transformed = svd.fit_transform(&x).unwrap();
//! assert_eq!(x_transformed.ncols(), 2);
//!
//! // Check explained variance ratio
//! let var_ratio = svd.explained_variance_ratio().unwrap();
//! assert!(var_ratio[0] >= var_ratio[1]);
//! ```

// Allow common patterns in numerical/scientific code
#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::return_self_not_must_use)]

use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::preprocessing::{check_is_fitted, check_non_empty, check_shape, Transformer};
use crate::{FerroError, Result};

/// Algorithm for computing truncated SVD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TruncatedSvdAlgorithm {
    /// Automatically choose between full and randomized based on data size.
    #[default]
    Auto,
    /// Randomized SVD for large datasets.
    /// Uses the algorithm from Halko et al. (2011).
    Randomized,
    /// Exact truncated SVD using ARPACK-style iteration.
    /// More accurate but slower for large datasets.
    Arpack,
}

/// Truncated Singular Value Decomposition (TruncatedSVD).
///
/// Dimensionality reduction using truncated SVD. This transformer performs linear
/// dimensionality reduction by means of truncated SVD. Contrary to PCA, this
/// estimator does not center the data before computing the SVD, which makes it
/// suitable for use with sparse matrices.
///
/// Also known as LSA (Latent Semantic Analysis) when applied to term-document matrices.
///
/// # Algorithm
///
/// 1. Compute SVD of data matrix: X = U Σ V^T
/// 2. Keep only the first k components
/// 3. Transform: X_transformed = X · V_k^T = U_k · Σ_k
///
/// # When to Use
///
/// - **Sparse data**: Unlike PCA, TruncatedSVD doesn't center data, preserving sparsity
/// - **LSA/LSI**: Text analysis with term-document matrices
/// - **Large datasets**: Randomized algorithm is efficient for large matrices
/// - **Feature extraction**: Reduce dimensionality while preserving structure
///
/// # Configuration
///
/// - `n_components`: Number of components to keep
/// - `n_iter`: Number of power iterations for randomized SVD
/// - `random_state`: Seed for reproducibility
/// - `algorithm`: SVD computation method (auto, randomized, arpack)
///
/// # Attributes (after fitting)
///
/// - `components_`: The right singular vectors V^T (n_components × n_features)
/// - `explained_variance_`: Variance explained by each component
/// - `explained_variance_ratio_`: Fraction of variance explained
/// - `singular_values_`: Singular values for each component
///
/// # Example
///
/// ```
/// use ferroml_core::decomposition::TruncatedSVD;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// // Create sparse-like data (many zeros)
/// let x = array![
///     [1.0, 0.0, 2.0, 0.0, 1.0],
///     [0.0, 1.0, 0.0, 3.0, 0.0],
///     [2.0, 0.0, 1.0, 0.0, 2.0],
///     [0.0, 2.0, 0.0, 1.0, 0.0]
/// ];
///
/// let mut svd = TruncatedSVD::new().with_n_components(2);
/// svd.fit(&x).unwrap();
///
/// // Transform new data
/// let x_reduced = svd.transform(&x).unwrap();
/// assert_eq!(x_reduced.ncols(), 2);
///
/// // Singular values are ordered by magnitude
/// let sv = svd.singular_values().unwrap();
/// assert!(sv[0] >= sv[1]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncatedSVD {
    // Configuration
    /// Number of components to keep
    n_components: usize,
    /// Number of iterations for randomized SVD
    n_iter: usize,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    /// Algorithm to use for SVD
    algorithm: TruncatedSvdAlgorithm,
    /// Oversampling factor for randomized SVD
    n_oversamples: usize,

    // Fitted state
    /// Components (right singular vectors V^T), shape (n_components, n_features)
    components: Option<Array2<f64>>,
    /// Explained variance per component
    explained_variance: Option<Array1<f64>>,
    /// Explained variance ratio per component
    explained_variance_ratio: Option<Array1<f64>>,
    /// Singular values
    singular_values: Option<Array1<f64>>,
    /// Number of features in input data
    n_features_in: Option<usize>,
    /// Number of samples seen during fitting
    n_samples: Option<usize>,
}

impl TruncatedSVD {
    /// Create a new TruncatedSVD with default settings.
    ///
    /// Default configuration:
    /// - `n_components`: 2
    /// - `n_iter`: 5 (power iterations for randomized SVD)
    /// - `n_oversamples`: 10
    /// - `algorithm`: Auto
    /// - `random_state`: None (random)
    pub fn new() -> Self {
        Self {
            n_components: 2,
            n_iter: 5,
            random_state: None,
            algorithm: TruncatedSvdAlgorithm::Auto,
            n_oversamples: 10,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
            n_features_in: None,
            n_samples: None,
        }
    }

    /// Set the number of components to keep.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of components (must be positive)
    ///
    /// # Panics
    ///
    /// Panics if n is 0.
    pub fn with_n_components(mut self, n: usize) -> Self {
        assert!(n > 0, "n_components must be positive");
        self.n_components = n;
        self
    }

    /// Set the number of power iterations for randomized SVD.
    ///
    /// More iterations improve accuracy but increase computation time.
    /// Default is 5, which is usually sufficient.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of iterations
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Set the random seed for reproducibility.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set the SVD algorithm to use.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - Algorithm choice
    pub fn with_algorithm(mut self, algorithm: TruncatedSvdAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the oversampling factor for randomized SVD.
    ///
    /// A larger value improves accuracy at the cost of memory.
    /// Default is 10.
    ///
    /// # Arguments
    ///
    /// * `n` - Oversampling factor
    pub fn with_n_oversamples(mut self, n: usize) -> Self {
        self.n_oversamples = n;
        self
    }

    /// Get the components (right singular vectors V^T).
    ///
    /// Returns a matrix of shape (n_components, n_features) where each row
    /// is a component in the original feature space.
    ///
    /// Returns `None` if not fitted.
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Get the explained variance for each component.
    ///
    /// The explained variance equals the variance of the projected data
    /// along each component direction. Equals σ² / (n_samples - 1) where
    /// σ is the singular value.
    ///
    /// Returns `None` if not fitted.
    pub fn explained_variance(&self) -> Option<&Array1<f64>> {
        self.explained_variance.as_ref()
    }

    /// Get the explained variance ratio for each component.
    ///
    /// Represents the percentage of total variance explained by each component.
    ///
    /// **Note**: Since TruncatedSVD doesn't center the data, the explained variance
    /// ratio is computed relative to the total variance of the uncentered data
    /// (sum of squared values). This differs from PCA.
    ///
    /// Returns `None` if not fitted.
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    /// Get the singular values corresponding to each component.
    ///
    /// Singular values are ordered from largest to smallest.
    ///
    /// Returns `None` if not fitted.
    pub fn singular_values(&self) -> Option<&Array1<f64>> {
        self.singular_values.as_ref()
    }

    /// Compute randomized SVD.
    ///
    /// Uses the algorithm from Halko et al. (2011):
    /// "Finding structure with randomness: Probabilistic algorithms for
    /// constructing approximate matrix decompositions"
    #[allow(clippy::many_single_char_names)]
    fn randomized_svd(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};

        let (n_samples, n_features) = x.dim();
        let n_random = (self.n_components + self.n_oversamples)
            .min(n_features)
            .min(n_samples);

        // Create RNG with optional seed
        let mut rng: StdRng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Generate random Gaussian matrix Ω (n_features × n_random)
        let omega =
            Array2::from_shape_fn((n_features, n_random), |_| StandardNormal.sample(&mut rng));

        // Form Y = X · Ω (n_samples × n_random)
        let y = x.dot(&omega);

        // Power iteration to improve approximation
        let mut q = y;
        for _ in 0..self.n_iter {
            // QR decomposition of Y to get orthonormal basis Q
            let (q_new, _) = qr_decomposition(&q)?;
            // Power iteration: Z = X^T · Q, then Y = X · Z
            let z = x.t().dot(&q_new);
            q = x.dot(&z);
        }

        // Final QR to get orthonormal basis
        let (q, _) = qr_decomposition(&q)?;

        // Form B = Q^T · X (small matrix)
        let b = q.t().dot(x);

        // SVD of the small matrix B via faer
        let (u_b, s_arr, vt_arr) = crate::linalg::thin_svd(&b)?;

        // Recover U = Q · U_B
        let u_arr = q.dot(&u_b);

        Ok((u_arr, s_arr, vt_arr))
    }

    /// Compute full SVD via the shared linalg module (faer when available).
    #[allow(clippy::unused_self)]
    fn full_svd(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        crate::linalg::thin_svd(x)
    }

    /// Compute SVD based on the selected algorithm.
    fn compute_svd(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (n_samples, n_features) = x.dim();

        let algorithm = match self.algorithm {
            TruncatedSvdAlgorithm::Auto => {
                // Use randomized for large matrices
                if n_samples > 500 || n_features > 500 {
                    TruncatedSvdAlgorithm::Randomized
                } else {
                    TruncatedSvdAlgorithm::Arpack
                }
            }
            other => other,
        };

        match algorithm {
            TruncatedSvdAlgorithm::Randomized => self.randomized_svd(x),
            TruncatedSvdAlgorithm::Arpack | TruncatedSvdAlgorithm::Auto => self.full_svd(x),
        }
    }
}

impl Default for TruncatedSVD {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for TruncatedSVD {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let (n_samples, n_features) = x.dim();

        // Validate n_components
        let max_components = n_samples.min(n_features);
        if self.n_components > max_components {
            return Err(FerroError::invalid_input(format!(
                "n_components ({}) must be <= min(n_samples, n_features) ({})",
                self.n_components, max_components
            )));
        }

        // Compute SVD (WITHOUT centering - key difference from PCA)
        let (_u, singular_values, vt) = self.compute_svd(x)?;

        // Extract top components
        let n_components = self.n_components.min(singular_values.len());
        let components = vt.slice(s![..n_components, ..]).to_owned();
        let singular_vals = singular_values.slice(s![..n_components]).to_owned();

        // Compute explained variance
        // For uncentered data, variance = σ² / (n_samples - 1)
        let n_minus_1 = (n_samples - 1).max(1) as f64;
        let explained_variance: Array1<f64> = singular_vals.mapv(|s| s * s / n_minus_1);

        // Total variance of the data (sum of all squared singular values / (n-1))
        // This equals the Frobenius norm squared / (n-1)
        let total_variance: f64 = singular_values.iter().map(|&s| s * s / n_minus_1).sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            explained_variance.mapv(|v| v / total_variance)
        } else {
            Array1::zeros(n_components)
        };

        // Store fitted state
        self.components = Some(components);
        self.singular_values = Some(singular_vals);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        self.n_features_in = Some(n_features);
        self.n_samples = Some(n_samples);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let components = self.components.as_ref().unwrap();

        // Project onto components: X_transformed = X · V^T.T = X · V
        // Since components is V^T, we need to transpose it
        let x_transformed = x.dot(&components.t());

        Ok(x_transformed)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;

        let n_components = self.components.as_ref().unwrap().nrows();
        if x.ncols() != n_components {
            return Err(FerroError::shape_mismatch(
                format!("({}, {})", x.nrows(), n_components),
                format!("({}, {})", x.nrows(), x.ncols()),
            ));
        }

        let components = self.components.as_ref().unwrap();

        // Inverse: X_reconstructed = X_transformed · V^T
        let x_reconstructed = x.dot(components);

        Ok(x_reconstructed)
    }

    fn is_fitted(&self) -> bool {
        self.components.is_some()
            && self.singular_values.is_some()
            && self.explained_variance.is_some()
    }

    fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
        if !self.is_fitted() {
            return None;
        }
        let n_components = self.components.as_ref()?.nrows();
        Some((0..n_components).map(|i| format!("svd{i}")).collect())
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.components.as_ref().map(|c| c.nrows())
    }
}

/// Simple QR decomposition using modified Gram-Schmidt.
///
/// Returns (Q, R) where Q is orthonormal and R is upper triangular.
#[allow(clippy::many_single_char_names)]
/// QR decomposition — delegates to shared linalg module.
fn qr_decomposition(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    crate::linalg::qr_decomposition(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_truncated_svd_basic() {
        let mut svd = TruncatedSVD::new().with_n_components(2);
        let x = array![
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 3.0],
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ];

        svd.fit(&x).unwrap();

        assert!(svd.is_fitted());
        assert_eq!(svd.components().unwrap().nrows(), 2);
        assert_eq!(svd.components().unwrap().ncols(), 4);
    }

    #[test]
    fn test_truncated_svd_transform() {
        let mut svd = TruncatedSVD::new().with_n_components(2);
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0]
        ];

        svd.fit(&x).unwrap();
        let x_transformed = svd.transform(&x).unwrap();

        assert_eq!(x_transformed.nrows(), 3);
        assert_eq!(x_transformed.ncols(), 2);
    }

    #[test]
    fn test_truncated_svd_singular_values_ordered() {
        let mut svd = TruncatedSVD::new().with_n_components(3);
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [3.0, 6.0, 9.0, 12.0, 15.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0]
        ];

        svd.fit(&x).unwrap();
        let sv = svd.singular_values().unwrap();

        // Singular values should be in descending order
        for i in 1..sv.len() {
            assert!(
                sv[i - 1] >= sv[i] - TOLERANCE,
                "Singular values should be descending: {} < {}",
                sv[i - 1],
                sv[i]
            );
        }
    }

    #[test]
    fn test_truncated_svd_explained_variance_ratio() {
        let mut svd = TruncatedSVD::new().with_n_components(2);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        svd.fit(&x).unwrap();
        let ratio = svd.explained_variance_ratio().unwrap();

        // First component should explain more variance
        assert!(ratio[0] >= ratio[1]);

        // Ratios should be positive
        for &r in ratio.iter() {
            assert!(r >= 0.0, "Variance ratio should be non-negative: {}", r);
        }
    }

    #[test]
    fn test_truncated_svd_inverse_transform() {
        let mut svd = TruncatedSVD::new().with_n_components(2);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        svd.fit(&x).unwrap();
        let x_transformed = svd.transform(&x).unwrap();
        let x_reconstructed = svd.inverse_transform(&x_transformed).unwrap();

        // With 2 components for 2 features, should reconstruct exactly
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(
                    (x[[i, j]] - x_reconstructed[[i, j]]).abs() < TOLERANCE,
                    "Mismatch at [{}, {}]: {} vs {}",
                    i,
                    j,
                    x[[i, j]],
                    x_reconstructed[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_truncated_svd_no_centering() {
        // TruncatedSVD should NOT center data (unlike PCA)
        let mut svd = TruncatedSVD::new().with_n_components(1);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        svd.fit(&x).unwrap();
        let x_transformed = svd.transform(&x).unwrap();

        // Transformed data should NOT have zero mean (since we didn't center)
        let mean: f64 = x_transformed.iter().sum::<f64>() / x_transformed.len() as f64;
        // The mean should be non-zero (showing data wasn't centered)
        // Note: This is a characteristic test, actual value depends on data
        assert!(
            mean.abs() > TOLERANCE,
            "TruncatedSVD should not center data, but mean is {}",
            mean
        );
    }

    #[test]
    fn test_truncated_svd_components_orthonormal() {
        let mut svd = TruncatedSVD::new().with_n_components(3);
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 5.0, 8.0],
            [1.0, 4.0, 2.0, 6.0]
        ];

        svd.fit(&x).unwrap();
        let components = svd.components().unwrap();

        // Components (rows of V^T) should be orthonormal
        for i in 0..components.nrows() {
            // Unit length
            let norm: f64 = components.row(i).dot(&components.row(i)).sqrt();
            assert!(
                (norm - 1.0).abs() < TOLERANCE,
                "Component {} has norm {} instead of 1",
                i,
                norm
            );

            // Orthogonal to other components
            for j in (i + 1)..components.nrows() {
                let dot: f64 = components.row(i).dot(&components.row(j));
                assert!(
                    dot.abs() < TOLERANCE,
                    "Components {} and {} have dot product {} instead of 0",
                    i,
                    j,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_truncated_svd_sparse_like_data() {
        // Test with sparse-like data (many zeros)
        let mut svd = TruncatedSVD::new().with_n_components(2);
        let x = array![
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0]
        ];

        svd.fit(&x).unwrap();
        let x_transformed = svd.transform(&x).unwrap();

        assert_eq!(x_transformed.ncols(), 2);
        assert!(svd.singular_values().unwrap()[0] > 0.0);
    }

    #[test]
    fn test_truncated_svd_feature_names() {
        let mut svd = TruncatedSVD::new().with_n_components(3);
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0]
        ];

        svd.fit(&x).unwrap();
        let names = svd.get_feature_names_out(None).unwrap();

        assert_eq!(names, vec!["svd0", "svd1", "svd2"]);
    }

    #[test]
    fn test_truncated_svd_not_fitted() {
        let svd = TruncatedSVD::new();
        let x = array![[1.0, 2.0]];

        assert!(!svd.is_fitted());
        assert!(svd.transform(&x).is_err());
    }

    #[test]
    fn test_truncated_svd_shape_mismatch() {
        let mut svd = TruncatedSVD::new().with_n_components(2);
        let x_fit = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        svd.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0]]; // 2 features instead of 3
        assert!(svd.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_truncated_svd_too_many_components() {
        let mut svd = TruncatedSVD::new().with_n_components(10);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // Only 2 features

        let result = svd.fit(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncated_svd_randomized_algorithm() {
        let mut svd = TruncatedSVD::new()
            .with_n_components(2)
            .with_algorithm(TruncatedSvdAlgorithm::Randomized)
            .with_random_state(42);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [10.0, 11.0, 12.0, 13.0]
        ];

        svd.fit(&x).unwrap();
        let x_transformed = svd.transform(&x).unwrap();

        assert_eq!(x_transformed.ncols(), 2);
    }

    #[test]
    fn test_truncated_svd_reproducibility() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let mut svd1 = TruncatedSVD::new()
            .with_n_components(2)
            .with_algorithm(TruncatedSvdAlgorithm::Randomized)
            .with_random_state(42);
        svd1.fit(&x).unwrap();

        let mut svd2 = TruncatedSVD::new()
            .with_n_components(2)
            .with_algorithm(TruncatedSvdAlgorithm::Randomized)
            .with_random_state(42);
        svd2.fit(&x).unwrap();

        // Same seed should give same results
        let sv1 = svd1.singular_values().unwrap();
        let sv2 = svd2.singular_values().unwrap();

        for i in 0..sv1.len() {
            assert!(
                (sv1[i] - sv2[i]).abs() < TOLERANCE,
                "Reproducibility failed: {} vs {}",
                sv1[i],
                sv2[i]
            );
        }
    }

    #[test]
    fn test_truncated_svd_n_iter() {
        // More iterations should improve accuracy for randomized SVD
        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0]
        ];

        let mut svd = TruncatedSVD::new()
            .with_n_components(2)
            .with_algorithm(TruncatedSvdAlgorithm::Randomized)
            .with_n_iter(10)
            .with_random_state(42);

        svd.fit(&x).unwrap();
        assert!(svd.is_fitted());
    }

    #[test]
    fn test_qr_decomposition() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]];

        let (q, r) = qr_decomposition(&a).unwrap();

        // Q should be orthogonal
        for i in 0..q.ncols() {
            let norm: f64 = q.column(i).dot(&q.column(i)).sqrt();
            if norm > TOLERANCE {
                assert!(
                    (norm - 1.0).abs() < TOLERANCE,
                    "Column {} has norm {}",
                    i,
                    norm
                );
            }
        }

        // Q * R should approximately equal A
        let qr = q.dot(&r);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert!(
                    (a[[i, j]] - qr[[i, j]]).abs() < TOLERANCE,
                    "QR decomposition mismatch at [{}, {}]: {} vs {}",
                    i,
                    j,
                    a[[i, j]],
                    qr[[i, j]]
                );
            }
        }
    }
}
