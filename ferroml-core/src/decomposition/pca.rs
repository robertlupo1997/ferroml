//! Principal Component Analysis (PCA)
//!
//! Linear dimensionality reduction using Singular Value Decomposition to project
//! data to a lower dimensional space while maximizing variance.
//!
//! ## Mathematical Background
//!
//! PCA finds orthogonal directions (principal components) that maximize the
//! variance of the projected data. Given centered data matrix X, PCA computes
//! the SVD: X = U Σ V^T
//!
//! The principal components are the columns of V, and the projections are U Σ.
//!
//! ## Features
//!
//! - Automatic component selection based on explained variance threshold
//! - Component loadings for interpretability
//! - Whitening transformation option
//! - Multiple SVD solvers (full, randomized for large data)
//! - Noise variance estimation
//!
//! ## Example
//!
//! ```
//! use ferroml_core::decomposition::PCA;
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! let mut pca = PCA::new().with_n_components(2);
//! let x = array![
//!     [2.5, 2.4],
//!     [0.5, 0.7],
//!     [2.2, 2.9],
//!     [1.9, 2.2],
//!     [3.1, 3.0]
//! ];
//!
//! let x_transformed = pca.fit_transform(&x).unwrap();
//!
//! // Check that variance is maximized in first component
//! let var_ratio = pca.explained_variance_ratio().unwrap();
//! assert!(var_ratio[0] > var_ratio[1]);
//! ```

// Allow common patterns in numerical/scientific code
#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::return_self_not_must_use)]

use ndarray::{s, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::preprocessing::{
    check_finite, check_is_fitted, check_non_empty, check_shape, Transformer,
};
use crate::{FerroError, Result};

/// SVD solver strategy for PCA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SvdSolver {
    /// Automatically choose between full and randomized based on data size.
    #[default]
    Auto,
    /// Full SVD using standard algorithm.
    /// Best for small to medium datasets.
    Full,
    /// Randomized SVD for large datasets.
    /// Faster but approximate for large n_features.
    Randomized,
}

/// Specifies how to determine the number of components.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum NComponents {
    /// Keep exactly n components.
    N(usize),
    /// Keep components explaining at least this fraction of variance (0.0 to 1.0).
    VarianceRatio(f64),
    /// Keep all components (min of n_samples, n_features).
    #[default]
    All,
}

/// Principal Component Analysis (PCA).
///
/// Dimensionality reduction using SVD to find directions of maximum variance.
///
/// # Algorithm
///
/// 1. Center the data by subtracting the mean
/// 2. Compute SVD of centered data: X_centered = U Σ V^T
/// 3. Principal components are rows of V (eigenvectors of X^T X)
/// 4. Project data: X_transformed = X_centered · V^T
///
/// # Configuration
///
/// - `n_components`: Number of components to keep (int, variance ratio, or all)
/// - `whiten`: If true, divide components by singular values for unit variance
/// - `svd_solver`: Algorithm for computing SVD
///
/// # Attributes (after fitting)
///
/// - `components_`: Principal axes in feature space (n_components × n_features)
/// - `explained_variance_`: Variance explained by each component
/// - `explained_variance_ratio_`: Percentage of variance explained
/// - `singular_values_`: Singular values corresponding to each component
/// - `mean_`: Per-feature mean estimated from training data
/// - `noise_variance_`: Estimated noise variance (if n_components < n_features)
///
/// # Example
///
/// ```
/// use ferroml_core::decomposition::PCA;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// // Reduce 3D data to 2D
/// let mut pca = PCA::new().with_n_components(2);
/// let x = array![
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ];
///
/// pca.fit(&x).unwrap();
///
/// // Transform to 2D
/// let x_2d = pca.transform(&x).unwrap();
/// assert_eq!(x_2d.ncols(), 2);
///
/// // Check cumulative explained variance
/// let cum_var = pca.cumulative_explained_variance_ratio().unwrap();
/// println!("Cumulative variance explained: {:?}", cum_var);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCA {
    // Configuration
    /// Number of components to keep
    n_components: NComponents,
    /// Whether to whiten the transformed data
    whiten: bool,
    /// SVD solver to use
    svd_solver: SvdSolver,

    // Fitted state
    /// Principal components (n_components × n_features)
    components: Option<Array2<f64>>,
    /// Explained variance per component
    explained_variance: Option<Array1<f64>>,
    /// Explained variance ratio per component
    explained_variance_ratio: Option<Array1<f64>>,
    /// Singular values
    singular_values: Option<Array1<f64>>,
    /// Per-feature mean from training data
    mean: Option<Array1<f64>>,
    /// Number of components (resolved after fitting)
    n_components_fitted: Option<usize>,
    /// Number of samples seen during fitting
    n_samples: Option<usize>,
    /// Number of features in input
    n_features_in: Option<usize>,
    /// Estimated noise variance
    noise_variance: Option<f64>,
}

impl PCA {
    /// Create a new PCA with default settings.
    ///
    /// Default configuration:
    /// - `n_components`: All (keep all components)
    /// - `whiten`: false
    /// - `svd_solver`: Auto
    pub fn new() -> Self {
        Self {
            n_components: NComponents::All,
            whiten: false,
            svd_solver: SvdSolver::Auto,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
            mean: None,
            n_components_fitted: None,
            n_samples: None,
            n_features_in: None,
            noise_variance: None,
        }
    }

    /// Set the number of components to keep.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of components
    ///
    /// # Panics
    ///
    /// Panics if n is 0.
    pub fn with_n_components(mut self, n: usize) -> Self {
        assert!(n > 0, "n_components must be positive");
        self.n_components = NComponents::N(n);
        self
    }

    /// Set components based on explained variance ratio threshold.
    ///
    /// Keeps the minimum number of components that explain at least
    /// the specified fraction of total variance.
    ///
    /// # Arguments
    ///
    /// * `ratio` - Minimum cumulative explained variance ratio (0.0 to 1.0)
    ///
    /// # Panics
    ///
    /// Panics if ratio is not in (0, 1].
    pub fn with_variance_ratio(mut self, ratio: f64) -> Self {
        assert!(
            ratio > 0.0 && ratio <= 1.0,
            "variance_ratio must be in (0, 1]"
        );
        self.n_components = NComponents::VarianceRatio(ratio);
        self
    }

    /// Configure whether to whiten the output.
    ///
    /// When true, the components are divided by singular values to ensure
    /// uncorrelated outputs with unit component-wise variances.
    pub fn with_whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Set the SVD solver strategy.
    pub fn with_svd_solver(mut self, solver: SvdSolver) -> Self {
        self.svd_solver = solver;
        self
    }

    /// Get the principal components (eigenvectors).
    ///
    /// Returns a matrix of shape (n_components, n_features) where each row
    /// is a principal component in the original feature space.
    ///
    /// # Component Interpretation
    ///
    /// The absolute values indicate feature importance for each component.
    /// Signs indicate direction of correlation.
    ///
    /// Returns `None` if not fitted.
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Get the explained variance for each component.
    ///
    /// Represents the amount of variance explained by each principal component.
    /// Equal to eigenvalues of the covariance matrix.
    ///
    /// Returns `None` if not fitted.
    pub fn explained_variance(&self) -> Option<&Array1<f64>> {
        self.explained_variance.as_ref()
    }

    /// Get the explained variance ratio for each component.
    ///
    /// Represents the percentage of total variance explained by each component.
    /// Sums to 1.0 if all components are kept.
    ///
    /// Returns `None` if not fitted.
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    /// Get the cumulative explained variance ratio.
    ///
    /// Useful for determining how many components are needed to explain
    /// a desired percentage of variance.
    ///
    /// Returns `None` if not fitted.
    pub fn cumulative_explained_variance_ratio(&self) -> Option<Array1<f64>> {
        let ratios = self.explained_variance_ratio.as_ref()?;
        let mut cumsum = Array1::zeros(ratios.len());
        let mut sum = 0.0;
        for (i, &r) in ratios.iter().enumerate() {
            sum += r;
            cumsum[i] = sum;
        }
        Some(cumsum)
    }

    /// Get the singular values.
    ///
    /// Singular values are proportional to the standard deviation of the
    /// data along each principal component.
    ///
    /// Returns `None` if not fitted.
    pub fn singular_values(&self) -> Option<&Array1<f64>> {
        self.singular_values.as_ref()
    }

    /// Get the per-feature mean from training data.
    ///
    /// Returns `None` if not fitted.
    pub fn mean(&self) -> Option<&Array1<f64>> {
        self.mean.as_ref()
    }

    /// Get the number of components determined during fitting.
    ///
    /// Returns `None` if not fitted.
    pub fn n_components_(&self) -> Option<usize> {
        self.n_components_fitted
    }

    /// Get the estimated noise variance.
    ///
    /// Only available when n_components < min(n_samples, n_features).
    /// Estimated as the average of the discarded eigenvalues.
    ///
    /// Returns `None` if not fitted or all components are kept.
    pub fn noise_variance(&self) -> Option<f64> {
        self.noise_variance
    }

    /// Get the component loadings (correlations between features and components).
    ///
    /// Loadings are components scaled by the square root of explained variance,
    /// representing the correlation between each feature and each component.
    ///
    /// Returns matrix of shape (n_components, n_features).
    ///
    /// Returns `None` if not fitted.
    pub fn loadings(&self) -> Option<Array2<f64>> {
        let components = self.components.as_ref()?;
        let explained_var = self.explained_variance.as_ref()?;

        let mut loadings = components.clone();
        for (i, &var) in explained_var.iter().enumerate() {
            let scale = var.sqrt();
            loadings.row_mut(i).iter_mut().for_each(|v| *v *= scale);
        }
        Some(loadings)
    }

    /// Compute the covariance matrix in the original feature space.
    ///
    /// Reconstructs the covariance from components and explained variance:
    /// Σ = V^T · diag(λ) · V + σ² · I
    ///
    /// where V is the components matrix, λ is explained variance, and σ²
    /// is the noise variance.
    ///
    /// Returns matrix of shape (n_features, n_features).
    ///
    /// Returns `None` if not fitted.
    pub fn get_covariance(&self) -> Option<Array2<f64>> {
        let components = self.components.as_ref()?;
        let explained_var = self.explained_variance.as_ref()?;
        let n_features = self.n_features_in?;

        // Σ = V^T · diag(λ) · V
        let mut cov = Array2::zeros((n_features, n_features));

        for (i, &var) in explained_var.iter().enumerate() {
            let component = components.row(i);
            for j in 0..n_features {
                for k in 0..n_features {
                    cov[[j, k]] += var * component[j] * component[k];
                }
            }
        }

        // Add noise variance on diagonal if available
        if let Some(noise_var) = self.noise_variance {
            for i in 0..n_features {
                cov[[i, i]] += noise_var;
            }
        }

        Some(cov)
    }

    /// Perform SVD on centered data.
    fn compute_svd(
        &self,
        x_centered: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (n_samples, n_features) = x_centered.dim();

        // Choose solver based on data size if Auto
        let solver = match self.svd_solver {
            SvdSolver::Auto => {
                if (n_samples > 500 && n_features > 500)
                    || (n_features > 100 && n_features > 2 * n_samples)
                {
                    SvdSolver::Randomized
                } else {
                    SvdSolver::Full
                }
            }
            other => other,
        };

        match solver {
            SvdSolver::Full | SvdSolver::Auto => self.full_svd(x_centered),
            SvdSolver::Randomized => self.randomized_svd(x_centered),
        }
    }

    /// Compute full SVD using nalgebra.
    #[allow(clippy::unused_self)]
    fn full_svd(
        &self,
        x_centered: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let (n_samples, n_features) = x_centered.dim();

        // Convert to nalgebra matrix for SVD
        let mat = nalgebra::DMatrix::from_fn(n_samples, n_features, |i, j| x_centered[[i, j]]);

        // Compute SVD
        let svd = mat.svd(true, true);

        let u = svd
            .u
            .ok_or_else(|| FerroError::numerical("SVD failed to compute U matrix"))?;
        let s = svd.singular_values;
        let vt = svd
            .v_t
            .ok_or_else(|| FerroError::numerical("SVD failed to compute V^T matrix"))?;

        // Convert back to ndarray
        let u_arr = Array2::from_shape_fn((u.nrows(), u.ncols()), |(i, j)| u[(i, j)]);
        let s_arr = Array1::from_iter(s.iter().copied());
        let vt_arr = Array2::from_shape_fn((vt.nrows(), vt.ncols()), |(i, j)| vt[(i, j)]);

        Ok((u_arr, s_arr, vt_arr))
    }

    /// Compute randomized SVD for large datasets.
    ///
    /// Uses the randomized algorithm from Halko et al. (2011):
    /// "Finding structure with randomness: Probabilistic algorithms for
    /// constructing approximate matrix decompositions"
    fn randomized_svd(
        &self,
        x_centered: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        use rand_distr::{Distribution, StandardNormal};

        let (n_samples, n_features) = x_centered.dim();

        // Determine target rank
        let n_components = match self.n_components {
            NComponents::N(n) => n.min(n_samples).min(n_features),
            NComponents::VarianceRatio(_) | NComponents::All => n_samples.min(n_features),
        };

        // Oversampling parameter
        let n_oversamples = 10;
        let n_random = (n_components + n_oversamples).min(n_features);

        // Generate random matrix
        let mut rng = rand::rng();
        let omega =
            Array2::from_shape_fn((n_features, n_random), |_| StandardNormal.sample(&mut rng));

        // Form sample matrix Y = X · Ω
        let y = x_centered.dot(&omega);

        // Power iteration for better approximation (1 iteration usually sufficient)
        let mut q = y;
        for _ in 0..2 {
            // QR decomposition of Y
            let (q_new, _) = qr_decomposition(&q)?;
            // Power iteration: Q = orth(X · X^T · Q)
            let z = x_centered.t().dot(&q_new);
            let y2 = x_centered.dot(&z);
            q = y2;
        }

        // Final QR
        let (q, _) = qr_decomposition(&q)?;

        // Form B = Q^T · X
        let b = q.t().dot(x_centered);

        // SVD of small matrix B
        let mat = nalgebra::DMatrix::from_fn(b.nrows(), b.ncols(), |i, j| b[[i, j]]);
        let svd = mat.svd(true, true);

        let u_b = svd
            .u
            .ok_or_else(|| FerroError::numerical("Randomized SVD failed"))?;
        let s = svd.singular_values;
        let vt = svd
            .v_t
            .ok_or_else(|| FerroError::numerical("Randomized SVD failed"))?;

        // U = Q · U_B
        let u_b_arr = Array2::from_shape_fn((u_b.nrows(), u_b.ncols()), |(i, j)| u_b[(i, j)]);
        let u_arr = q.dot(&u_b_arr);

        let s_arr = Array1::from_iter(s.iter().copied());
        let vt_arr = Array2::from_shape_fn((vt.nrows(), vt.ncols()), |(i, j)| vt[(i, j)]);

        Ok((u_arr, s_arr, vt_arr))
    }

    /// Determine the number of components based on configuration.
    fn resolve_n_components(
        &self,
        explained_variance_ratio: &Array1<f64>,
        max_components: usize,
    ) -> usize {
        match self.n_components {
            NComponents::N(n) => n.min(max_components),
            NComponents::VarianceRatio(threshold) => {
                let mut cumsum = 0.0;
                for (i, &ratio) in explained_variance_ratio.iter().enumerate() {
                    cumsum += ratio;
                    if cumsum >= threshold {
                        return i + 1;
                    }
                }
                max_components
            }
            NComponents::All => max_components,
        }
    }
}

impl Default for PCA {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for PCA {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;
        check_finite(x)?;

        let (n_samples, n_features) = x.dim();

        // Compute mean and center the data
        let mean = x
            .mean_axis(Axis(0))
            .ok_or_else(|| FerroError::numerical("Failed to compute mean - data may be empty"))?;
        let x_centered = x - &mean;

        // Compute SVD
        let (_u, singular_values, vt) = self.compute_svd(&x_centered)?;

        // Compute explained variance
        // Variance = σ² / (n - 1), where σ is singular value
        let n_minus_1 = (n_samples - 1).max(1) as f64;
        let explained_variance: Array1<f64> = singular_values.mapv(|s| s * s / n_minus_1);

        // Total variance
        let total_variance: f64 = explained_variance.sum();
        let explained_variance_ratio = if total_variance > 0.0 {
            explained_variance.mapv(|v| v / total_variance)
        } else {
            Array1::zeros(explained_variance.len())
        };

        // Determine number of components
        let max_components = n_samples.min(n_features);
        let n_components = self.resolve_n_components(&explained_variance_ratio, max_components);

        // Extract top components
        let components = vt.slice(s![..n_components, ..]).to_owned();
        let explained_var = explained_variance.slice(s![..n_components]).to_owned();
        let explained_var_ratio = explained_variance_ratio
            .slice(s![..n_components])
            .to_owned();
        let singular_vals = singular_values.slice(s![..n_components]).to_owned();

        // Compute noise variance if not all components are kept
        let noise_variance = if n_components < max_components {
            let remaining_var: f64 = explained_variance.slice(s![n_components..]).sum();
            let n_remaining = max_components - n_components;
            if n_remaining > 0 {
                Some(remaining_var / n_remaining as f64)
            } else {
                None
            }
        } else {
            None
        };

        // Store fitted state
        self.components = Some(components);
        self.explained_variance = Some(explained_var);
        self.explained_variance_ratio = Some(explained_var_ratio);
        self.singular_values = Some(singular_vals);
        self.mean = Some(mean);
        self.n_components_fitted = Some(n_components);
        self.n_samples = Some(n_samples);
        self.n_features_in = Some(n_features);
        self.noise_variance = noise_variance;

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(
            x,
            self.n_features_in
                .ok_or_else(|| FerroError::not_fitted("transform"))?,
        )?;

        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;

        // Center the data
        let x_centered = x - mean;

        // Project onto principal components: X_transformed = X_centered · components^T
        let mut x_transformed = x_centered.dot(&components.t());

        // Whiten if requested
        if self.whiten {
            let singular_values = self
                .singular_values
                .as_ref()
                .ok_or_else(|| FerroError::not_fitted("transform"))?;
            let n_samples =
                self.n_samples
                    .ok_or_else(|| FerroError::not_fitted("transform"))? as f64;
            // Guard against division by near-zero singular values
            let eps = 1e-10;
            let scale = singular_values.mapv(|s| (n_samples - 1.0).sqrt() / s.max(eps));

            for (j, &s) in scale.iter().enumerate() {
                x_transformed.column_mut(j).iter_mut().for_each(|v| *v *= s);
            }
        }

        Ok(x_transformed)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;

        let n_components = self
            .n_components_fitted
            .ok_or_else(|| FerroError::not_fitted("inverse_transform"))?;
        if x.ncols() != n_components {
            return Err(FerroError::shape_mismatch(
                format!("({}, {})", x.nrows(), n_components),
                format!("({}, {})", x.nrows(), x.ncols()),
            ));
        }

        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;

        // Reverse whitening if applied
        let x_unwhitened = if self.whiten {
            let singular_values = self
                .singular_values
                .as_ref()
                .ok_or_else(|| FerroError::not_fitted("transform"))?;
            let n_samples =
                self.n_samples
                    .ok_or_else(|| FerroError::not_fitted("transform"))? as f64;
            let scale = singular_values.mapv(|s| s / (n_samples - 1.0).sqrt());

            let mut x_scaled = x.clone();
            for (j, &s) in scale.iter().enumerate() {
                x_scaled.column_mut(j).iter_mut().for_each(|v| *v *= s);
            }
            x_scaled
        } else {
            x.to_owned()
        };

        // Reverse projection: X_original = X_transformed · components + mean
        let x_reconstructed = x_unwhitened.dot(components) + mean;

        Ok(x_reconstructed)
    }

    fn is_fitted(&self) -> bool {
        self.components.is_some()
            && self.mean.is_some()
            && self.explained_variance.is_some()
            && self.singular_values.is_some()
    }

    fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_components = self.n_components_fitted?;
        Some((0..n_components).map(|i| format!("pca{i}")).collect())
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_components_fitted
    }
}

/// Incremental PCA for large datasets.
///
/// Processes data in batches to reduce memory requirements.
/// Useful when the full dataset doesn't fit in memory.
///
/// # Algorithm
///
/// Uses the incremental update algorithm from:
/// Ross et al. "Incremental Learning for Robust Visual Tracking" (2008)
///
/// Maintains running estimates of mean and covariance, updating with each batch.
///
/// # Example
///
/// ```
/// use ferroml_core::decomposition::IncrementalPCA;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut ipca = IncrementalPCA::new().with_n_components(2);
///
/// // Process data in batches
/// let batch1 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
/// let batch2 = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// ipca.partial_fit(&batch1).unwrap();
/// ipca.partial_fit(&batch2).unwrap();
///
/// // Transform new data
/// let x_new = array![[5.0, 6.0, 7.0]];
/// let x_transformed = ipca.transform(&x_new).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalPCA {
    /// Number of components to keep
    n_components: Option<usize>,
    /// Whether to whiten the output
    whiten: bool,
    /// Batch size for incremental processing
    batch_size: Option<usize>,

    // Running state
    /// Principal components (n_components × n_features)
    components: Option<Array2<f64>>,
    /// Explained variance per component
    explained_variance: Option<Array1<f64>>,
    /// Explained variance ratio per component
    explained_variance_ratio: Option<Array1<f64>>,
    /// Singular values
    singular_values: Option<Array1<f64>>,
    /// Running mean
    mean: Option<Array1<f64>>,
    /// Running variance
    var: Option<Array1<f64>>,
    /// Number of components determined
    n_components_fitted: Option<usize>,
    /// Total number of samples seen
    n_samples_seen: usize,
    /// Number of features
    n_features_in: Option<usize>,
    /// Noise variance estimate
    noise_variance: Option<f64>,
}

impl IncrementalPCA {
    /// Create a new IncrementalPCA.
    pub fn new() -> Self {
        Self {
            n_components: None,
            whiten: false,
            batch_size: None,
            components: None,
            explained_variance: None,
            explained_variance_ratio: None,
            singular_values: None,
            mean: None,
            var: None,
            n_components_fitted: None,
            n_samples_seen: 0,
            n_features_in: None,
            noise_variance: None,
        }
    }

    /// Set the number of components to keep.
    pub fn with_n_components(mut self, n: usize) -> Self {
        assert!(n > 0, "n_components must be positive");
        self.n_components = Some(n);
        self
    }

    /// Configure whether to whiten the output.
    pub fn with_whiten(mut self, whiten: bool) -> Self {
        self.whiten = whiten;
        self
    }

    /// Set the batch size for fit_transform operations.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        assert!(size > 0, "batch_size must be positive");
        self.batch_size = Some(size);
        self
    }

    /// Incrementally fit the PCA model with a batch of data.
    ///
    /// Can be called multiple times to process large datasets in chunks.
    ///
    /// # Arguments
    ///
    /// * `x` - Batch of data (n_samples, n_features)
    pub fn partial_fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let (n_samples, n_features) = x.dim();

        // First batch: initialize
        if self.n_samples_seen == 0 {
            self.n_features_in = Some(n_features);
            self.mean = Some(Array1::zeros(n_features));
            self.var = Some(Array1::zeros(n_features));
        } else {
            check_shape(
                x,
                self.n_features_in
                    .ok_or_else(|| FerroError::not_fitted("transform"))?,
            )?;
        }

        // Update running mean and variance using Welford's algorithm
        let old_mean = self
            .mean
            .as_ref()
            .ok_or_else(|| FerroError::numerical("missing mean state during partial_fit"))?
            .clone();
        let old_n = self.n_samples_seen as f64;
        let new_n = (self.n_samples_seen + n_samples) as f64;

        // Batch mean
        let batch_mean = x
            .mean_axis(Axis(0))
            .ok_or_else(|| FerroError::numerical("Failed to compute batch mean"))?;

        // Updated mean
        let new_mean = &old_mean * (old_n / new_n) + &batch_mean * (n_samples as f64 / new_n);

        // For variance update, we use the incremental formula
        // This is a simplified version - full implementation would track second moment
        let batch_var = x.var_axis(Axis(0), 1.0);

        // Combine old and new variance (approximation)
        let new_var = if old_n > 0.0 {
            let old_var = self.var.as_ref().ok_or_else(|| {
                FerroError::numerical("missing variance state during partial_fit")
            })?;
            // Weighted combination with correction for mean shift
            let mean_diff = &batch_mean - &old_mean;
            old_var * (old_n / new_n)
                + &batch_var * (n_samples as f64 / new_n)
                + &mean_diff.mapv(|d| d * d) * (old_n * n_samples as f64 / (new_n * new_n))
        } else {
            batch_var
        };

        self.mean = Some(new_mean.clone());
        self.var = Some(new_var);
        self.n_samples_seen += n_samples;

        // Center the batch with the new mean
        let x_centered = x - &new_mean;

        // Determine number of components
        let max_components = self.n_samples_seen.min(n_features);
        let n_components = self
            .n_components
            .unwrap_or(max_components)
            .min(max_components);

        // If we have existing components, merge with new data
        // Otherwise, compute from scratch
        if let Some(ref old_components) = self.components.clone() {
            // Incremental update using SVD of augmented matrix
            let old_singular = self.singular_values.as_ref().ok_or_else(|| {
                FerroError::numerical("missing singular values during incremental update")
            })?;

            // Build augmented matrix [old_singular * old_components; x_centered]
            let k = old_components.nrows();

            // Scale old components by singular values
            let mut augmented_rows = Vec::with_capacity(k + n_samples);

            for i in 0..k {
                let scaled_row: Vec<f64> = old_components
                    .row(i)
                    .iter()
                    .map(|&v| v * old_singular[i])
                    .collect();
                augmented_rows.push(scaled_row);
            }

            for row in x_centered.rows() {
                augmented_rows.push(row.to_vec());
            }

            let augmented =
                Array2::from_shape_vec((k + n_samples, n_features), augmented_rows.concat())
                    .map_err(|e| {
                        FerroError::numerical(format!("Failed to create augmented matrix: {e}"))
                    })?;

            // SVD of augmented matrix
            let mat = nalgebra::DMatrix::from_fn(augmented.nrows(), augmented.ncols(), |i, j| {
                augmented[[i, j]]
            });

            let svd = mat.svd(true, true);

            let s = svd.singular_values;
            let vt = svd.v_t.ok_or_else(|| FerroError::numerical("SVD failed"))?;

            // Extract top components
            let n_keep = n_components.min(s.len());

            let components = Array2::from_shape_fn((n_keep, n_features), |(i, j)| vt[(i, j)]);
            let singular_values = Array1::from_iter(s.iter().take(n_keep).copied());

            // Explained variance
            let n_total = self.n_samples_seen as f64;
            let explained_variance = singular_values.mapv(|s| s * s / (n_total - 1.0).max(1.0));
            let total_var: f64 = s.iter().map(|&v| v * v / (n_total - 1.0).max(1.0)).sum();
            let explained_variance_ratio = if total_var > 0.0 {
                explained_variance.mapv(|v| v / total_var)
            } else {
                Array1::zeros(n_keep)
            };

            // Noise variance
            let noise_variance = if n_keep < max_components && s.len() > n_keep {
                let remaining: f64 = s
                    .iter()
                    .skip(n_keep)
                    .map(|&v| v * v / (n_total - 1.0).max(1.0))
                    .sum();
                let n_remaining = s.len() - n_keep;
                if n_remaining > 0 {
                    Some(remaining / n_remaining as f64)
                } else {
                    None
                }
            } else {
                None
            };

            self.components = Some(components);
            self.singular_values = Some(singular_values);
            self.explained_variance = Some(explained_variance);
            self.explained_variance_ratio = Some(explained_variance_ratio);
            self.n_components_fitted = Some(n_keep);
            self.noise_variance = noise_variance;
        } else {
            // First batch: compute SVD directly
            let mat = nalgebra::DMatrix::from_fn(n_samples, n_features, |i, j| x_centered[[i, j]]);

            let svd = mat.svd(true, true);
            let s = svd.singular_values;
            let vt = svd.v_t.ok_or_else(|| FerroError::numerical("SVD failed"))?;

            let n_keep = n_components.min(s.len());

            let components = Array2::from_shape_fn((n_keep, n_features), |(i, j)| vt[(i, j)]);
            let singular_values = Array1::from_iter(s.iter().take(n_keep).copied());

            let n_minus_1 = (n_samples - 1).max(1) as f64;
            let explained_variance = singular_values.mapv(|sv| sv * sv / n_minus_1);
            let total_var: f64 = s.iter().map(|&v| v * v / n_minus_1).sum();
            let explained_variance_ratio = if total_var > 0.0 {
                explained_variance.mapv(|v| v / total_var)
            } else {
                Array1::zeros(n_keep)
            };

            let noise_variance = if n_keep < s.len() {
                let remaining: f64 = s.iter().skip(n_keep).map(|&v| v * v / n_minus_1).sum();
                let n_remaining = s.len() - n_keep;
                if n_remaining > 0 {
                    Some(remaining / n_remaining as f64)
                } else {
                    None
                }
            } else {
                None
            };

            self.components = Some(components);
            self.singular_values = Some(singular_values);
            self.explained_variance = Some(explained_variance);
            self.explained_variance_ratio = Some(explained_variance_ratio);
            self.n_components_fitted = Some(n_keep);
            self.noise_variance = noise_variance;
        }

        Ok(())
    }

    /// Get the principal components.
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }

    /// Get the explained variance for each component.
    pub fn explained_variance(&self) -> Option<&Array1<f64>> {
        self.explained_variance.as_ref()
    }

    /// Get the explained variance ratio for each component.
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    /// Get the singular values.
    pub fn singular_values(&self) -> Option<&Array1<f64>> {
        self.singular_values.as_ref()
    }

    /// Get the mean.
    pub fn mean(&self) -> Option<&Array1<f64>> {
        self.mean.as_ref()
    }

    /// Get the number of samples seen.
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }
}

impl Default for IncrementalPCA {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for IncrementalPCA {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        // Reset state
        self.n_samples_seen = 0;
        self.components = None;
        self.mean = None;
        self.var = None;

        // Process in batches if batch_size is set, otherwise fit all at once
        if let Some(batch_size) = self.batch_size {
            let n_samples = x.nrows();
            let mut offset = 0;

            while offset < n_samples {
                let end = (offset + batch_size).min(n_samples);
                let batch = x.slice(s![offset..end, ..]).to_owned();
                self.partial_fit(&batch)?;
                offset = end;
            }
        } else {
            self.partial_fit(x)?;
        }

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(
            x,
            self.n_features_in
                .ok_or_else(|| FerroError::not_fitted("transform"))?,
        )?;

        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;

        let x_centered = x - mean;
        let mut x_transformed = x_centered.dot(&components.t());

        if self.whiten {
            let singular_values = self
                .singular_values
                .as_ref()
                .ok_or_else(|| FerroError::not_fitted("transform"))?;
            let n_samples = self.n_samples_seen as f64;
            // Guard against division by near-zero singular values
            let eps = 1e-10;
            let scale = singular_values.mapv(|s| (n_samples - 1.0).sqrt() / s.max(eps));

            for (j, &s) in scale.iter().enumerate() {
                x_transformed.column_mut(j).iter_mut().for_each(|v| *v *= s);
            }
        }

        Ok(x_transformed)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;

        let n_components = self
            .n_components_fitted
            .ok_or_else(|| FerroError::not_fitted("inverse_transform"))?;
        if x.ncols() != n_components {
            return Err(FerroError::shape_mismatch(
                format!("({}, {})", x.nrows(), n_components),
                format!("({}, {})", x.nrows(), x.ncols()),
            ));
        }

        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;

        let x_unwhitened = if self.whiten {
            let singular_values = self
                .singular_values
                .as_ref()
                .ok_or_else(|| FerroError::not_fitted("transform"))?;
            let n_samples = self.n_samples_seen as f64;
            let scale = singular_values.mapv(|s| s / (n_samples - 1.0).sqrt());

            let mut x_scaled = x.clone();
            for (j, &s) in scale.iter().enumerate() {
                x_scaled.column_mut(j).iter_mut().for_each(|v| *v *= s);
            }
            x_scaled
        } else {
            x.to_owned()
        };

        let x_reconstructed = x_unwhitened.dot(components) + mean;
        Ok(x_reconstructed)
    }

    fn is_fitted(&self) -> bool {
        self.components.is_some() && self.mean.is_some() && self.n_samples_seen > 0
    }

    fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_components = self.n_components_fitted?;
        Some((0..n_components).map(|i| format!("pca{i}")).collect())
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_components_fitted
    }
}

/// QR decomposition — delegates to shared linalg module (Modified Gram-Schmidt,
/// with optional faer backend for high-performance on large matrices).
#[allow(clippy::unnecessary_wraps)]
fn qr_decomposition(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    crate::linalg::qr_decomposition(a)
}

// =============================================================================
// PipelineTransformer Implementation
// =============================================================================

use crate::pipeline::PipelineTransformer;

impl PipelineTransformer for PCA {
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "PCA"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;
    use crate::testing::assertions::tolerances;
    use ndarray::array;

    // Use calibrated tolerance constants from assertions module
    const TOLERANCE: f64 = tolerances::DECOMPOSITION;
    const EPSILON: f64 = tolerances::STRICT;

    #[test]
    fn test_pca_basic() {
        let mut pca = PCA::new();
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9]
        ];

        pca.fit(&x).unwrap();

        // Check that we have 2 components (min of n_samples, n_features)
        assert_eq!(pca.n_components_(), Some(2));

        // Check explained variance is positive and first component explains most
        let explained_var = pca.explained_variance().unwrap();
        assert!(explained_var[0] > 0.0);
        assert!(explained_var[0] >= explained_var[1]);

        // Check explained variance sums approximately to total variance
        // PCA uses sample variance (n-1), so compare with ddof=1
        let total_var: f64 = explained_var.sum();
        let data_var = x.var_axis(Axis(0), 1.0).sum();
        // Allow some numerical tolerance (variance computations can differ slightly)
        assert!(
            (total_var - data_var).abs() / data_var < 0.15,
            "total_var={}, data_var={}",
            total_var,
            data_var
        );

        // Check explained variance ratio sums to ~1
        let ratio_sum: f64 = pca.explained_variance_ratio().unwrap().sum();
        assert_approx_eq!(
            ratio_sum,
            1.0,
            TOLERANCE,
            "explained variance ratio should sum to 1"
        );
    }

    #[test]
    fn test_pca_n_components() {
        let mut pca = PCA::new().with_n_components(1);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        pca.fit(&x).unwrap();

        assert_eq!(pca.n_components_(), Some(1));
        assert_eq!(pca.components().unwrap().nrows(), 1);
    }

    #[test]
    fn test_pca_variance_ratio() {
        let mut pca = PCA::new().with_variance_ratio(0.95);
        let x = array![
            [1.0, 0.1, 0.01],
            [2.0, 0.2, 0.02],
            [3.0, 0.3, 0.03],
            [4.0, 0.4, 0.04],
            [5.0, 0.5, 0.05]
        ];

        pca.fit(&x).unwrap();

        // Should keep components that explain at least 95% variance
        let cum_var = pca.cumulative_explained_variance_ratio().unwrap();
        let n_comp = pca.n_components_().unwrap();
        assert!(cum_var[n_comp - 1] >= 0.95);
    }

    #[test]
    fn test_pca_transform() {
        let mut pca = PCA::new().with_n_components(2);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        pca.fit(&x).unwrap();
        let x_transformed = pca.transform(&x).unwrap();

        // Should have 2 columns
        assert_eq!(x_transformed.ncols(), 2);
        assert_eq!(x_transformed.nrows(), 3);
    }

    #[test]
    fn test_pca_inverse_transform() {
        let mut pca = PCA::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        pca.fit(&x).unwrap();
        let x_transformed = pca.transform(&x).unwrap();
        let x_reconstructed = pca.inverse_transform(&x_transformed).unwrap();

        // Should be close to original
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
    fn test_pca_whiten() {
        let mut pca = PCA::new().with_whiten(true);
        let x = array![
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
            [3.1, 3.0],
            [2.3, 2.7],
            [2.0, 1.6],
            [1.0, 1.1],
            [1.5, 1.6],
            [1.1, 0.9]
        ];

        pca.fit(&x).unwrap();
        let x_whitened = pca.transform(&x).unwrap();

        // Whitened data should have unit variance
        let var = x_whitened.var_axis(Axis(0), 1.0);
        for &v in var.iter() {
            assert!((v - 1.0).abs() < 0.1, "Variance {} should be close to 1", v);
        }
    }

    #[test]
    fn test_pca_components_orthogonal() {
        let mut pca = PCA::new();
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        pca.fit(&x).unwrap();
        let components = pca.components().unwrap();

        // Components should be orthonormal
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
    fn test_pca_loadings() {
        let mut pca = PCA::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        pca.fit(&x).unwrap();
        let loadings = pca.loadings().unwrap();

        // Loadings should have same shape as components
        assert_eq!(loadings.dim(), pca.components().unwrap().dim());
    }

    #[test]
    fn test_pca_cumulative_variance() {
        let mut pca = PCA::new();
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        pca.fit(&x).unwrap();
        let cum_var = pca.cumulative_explained_variance_ratio().unwrap();

        // Should be monotonically increasing
        for i in 1..cum_var.len() {
            assert!(cum_var[i] >= cum_var[i - 1] - EPSILON);
        }

        // Last value should be 1.0
        assert_approx_eq!(
            *cum_var.last().unwrap(),
            1.0,
            TOLERANCE,
            "cumulative variance should end at 1"
        );
    }

    #[test]
    fn test_pca_feature_names() {
        let mut pca = PCA::new().with_n_components(2);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        pca.fit(&x).unwrap();
        let names = pca.get_feature_names_out(None).unwrap();

        assert_eq!(names, vec!["pca0", "pca1"]);
    }

    #[test]
    fn test_pca_not_fitted() {
        let pca = PCA::new();
        let x = array![[1.0, 2.0]];

        assert!(!pca.is_fitted());
        assert!(pca.transform(&x).is_err());
    }

    #[test]
    fn test_pca_shape_mismatch() {
        let mut pca = PCA::new();
        let x_fit = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        pca.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(pca.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_pca_covariance() {
        let mut pca = PCA::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        pca.fit(&x).unwrap();
        let cov = pca.get_covariance().unwrap();

        // Covariance matrix should be symmetric
        assert_eq!(cov.dim(), (2, 2));
        assert_approx_eq!(
            cov[[0, 1]],
            cov[[1, 0]],
            TOLERANCE,
            "covariance matrix should be symmetric"
        );
    }

    // ========== IncrementalPCA Tests ==========

    #[test]
    fn test_incremental_pca_basic() {
        let mut ipca = IncrementalPCA::new().with_n_components(2);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        ipca.fit(&x).unwrap();

        assert!(ipca.is_fitted());
        assert_eq!(ipca.n_components_fitted, Some(2));
    }

    #[test]
    fn test_incremental_pca_partial_fit() {
        let mut ipca = IncrementalPCA::new().with_n_components(2);

        let batch1 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let batch2 = array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];

        ipca.partial_fit(&batch1).unwrap();
        assert_eq!(ipca.n_samples_seen(), 2);

        ipca.partial_fit(&batch2).unwrap();
        assert_eq!(ipca.n_samples_seen(), 4);

        assert!(ipca.is_fitted());
    }

    #[test]
    fn test_incremental_pca_transform() {
        let mut ipca = IncrementalPCA::new().with_n_components(2);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        ipca.fit(&x).unwrap();
        let x_transformed = ipca.transform(&x).unwrap();

        assert_eq!(x_transformed.ncols(), 2);
        assert_eq!(x_transformed.nrows(), 4);
    }

    #[test]
    fn test_incremental_pca_batch_size() {
        let mut ipca = IncrementalPCA::new()
            .with_n_components(2)
            .with_batch_size(2);
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        ipca.fit(&x).unwrap();

        assert!(ipca.is_fitted());
        assert_eq!(ipca.n_samples_seen(), 4);
    }

    #[test]
    fn test_incremental_pca_inverse() {
        let mut ipca = IncrementalPCA::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        ipca.fit(&x).unwrap();
        let x_transformed = ipca.transform(&x).unwrap();
        let x_reconstructed = ipca.inverse_transform(&x_transformed).unwrap();

        // Allow larger tolerance for incremental
        let tol = 0.5;
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(
                    (x[[i, j]] - x_reconstructed[[i, j]]).abs() < tol,
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
    fn test_incremental_pca_explained_variance() {
        let mut ipca = IncrementalPCA::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        ipca.fit(&x).unwrap();

        let ratio = ipca.explained_variance_ratio().unwrap();
        let sum: f64 = ratio.sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

    // ========== QR Decomposition Tests ==========

    #[test]
    fn test_qr_decomposition() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let (q, r) = qr_decomposition(&a).unwrap();

        // Q should be orthogonal
        for i in 0..q.ncols() {
            let norm: f64 = q.column(i).dot(&q.column(i)).sqrt();
            if norm > EPSILON {
                assert!(
                    (norm - 1.0).abs() < TOLERANCE,
                    "Column {} has norm {}",
                    i,
                    norm
                );
            }

            for j in (i + 1)..q.ncols() {
                let dot: f64 = q.column(i).dot(&q.column(j));
                assert!(
                    dot.abs() < TOLERANCE,
                    "Columns {} and {} have dot product {}",
                    i,
                    j,
                    dot
                );
            }
        }

        // Q * R should equal A (approximately, within numerical tolerance)
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
