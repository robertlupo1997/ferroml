//! Factor Analysis
//!
//! A statistical model relating observed variables to latent factors.
//!
//! ## Mathematical Background
//!
//! Factor Analysis assumes that observed variables are linear combinations of
//! unobserved latent factors plus noise:
//!
//! X = μ + L·F + ε
//!
//! where:
//! - X: observed data (n_samples × n_features)
//! - μ: mean vector
//! - L: factor loadings matrix (n_features × n_factors)
//! - F: factor scores (n_samples × n_factors)
//! - ε: noise/unique variance (independent for each feature)
//!
//! ## Key Concepts
//!
//! - **Factor Loadings**: Correlations between observed variables and factors
//! - **Communalities**: Proportion of variance explained by common factors
//! - **Specific Variance**: Variance unique to each variable (not explained by factors)
//! - **Rotation**: Transform factors for better interpretability
//!
//! ## Comparison with PCA
//!
//! | Aspect | PCA | Factor Analysis |
//! |--------|-----|-----------------|
//! | Goal | Maximize variance | Model covariance structure |
//! | Noise | None (all variance explained) | Explicit noise model |
//! | Components | Principal components | Latent factors |
//! | Rotation | Not typically rotated | Often rotated for interpretation |
//! | Statistical model | No | Yes |
//!
//! ## Features
//!
//! - Maximum likelihood estimation via EM algorithm
//! - Rotation methods: Varimax, Quartimax, Promax (oblique)
//! - Factor loadings and communalities
//! - Explained variance per factor
//! - Noise variance estimation
//!
//! ## Example
//!
//! ```
//! use ferroml_core::decomposition::FactorAnalysis;
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! let mut fa = FactorAnalysis::new().with_n_factors(2);
//!
//! let x = array![
//!     [1.0, 2.0, 3.0, 1.5],
//!     [4.0, 5.0, 6.0, 4.5],
//!     [7.0, 8.0, 9.0, 7.5],
//!     [10.0, 11.0, 12.0, 10.5],
//!     [2.0, 3.0, 4.0, 2.5]
//! ];
//!
//! fa.fit(&x).unwrap();
//!
//! // Get factor loadings
//! let loadings = fa.loadings().unwrap();
//! println!("Factor loadings: {:?}", loadings);
//!
//! // Get communalities (variance explained by factors)
//! let communalities = fa.communalities().unwrap();
//! println!("Communalities: {:?}", communalities);
//! ```

// Allow common patterns in numerical/scientific code
#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::preprocessing::{check_is_fitted, Transformer};
use crate::{FerroError, Result};

/// Rotation method for factor loadings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Rotation {
    /// No rotation applied.
    #[default]
    None,
    /// Varimax rotation (orthogonal) - maximizes variance of squared loadings.
    /// Most common rotation, produces uncorrelated factors.
    Varimax,
    /// Quartimax rotation (orthogonal) - maximizes fourth power of loadings.
    /// Tends to produce a general factor.
    Quartimax,
    /// Promax rotation (oblique) - allows correlated factors.
    /// Based on varimax solution raised to a power.
    Promax,
}

/// SVD method for initial estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FaSvdMethod {
    /// Use randomized SVD (faster for large datasets).
    #[default]
    Randomized,
    /// Use full SVD via LAPACK.
    Full,
}

/// Factor Analysis.
///
/// A statistical model that explains observed variables as linear combinations
/// of latent factors plus noise.
///
/// # Algorithm
///
/// Uses the EM algorithm for Maximum Likelihood estimation:
///
/// 1. **E-step**: Compute expected factor scores given current parameters
/// 2. **M-step**: Update loadings and noise variance to maximize likelihood
///
/// # Configuration
///
/// - `n_factors`: Number of latent factors to extract
/// - `rotation`: Rotation method for interpretability (varimax, quartimax, promax)
/// - `tol`: Convergence tolerance for EM algorithm
/// - `max_iter`: Maximum number of EM iterations
///
/// # Attributes (after fitting)
///
/// - `loadings_`: Factor loadings matrix (n_features × n_factors)
/// - `noise_variance_`: Per-feature noise variance
/// - `communalities_`: Proportion of variance explained by factors per feature
/// - `explained_variance_`: Variance explained by each factor
/// - `explained_variance_ratio_`: Proportion of total variance per factor
///
/// # Example
///
/// ```
/// use ferroml_core::decomposition::{FactorAnalysis, Rotation};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// // Create factor analysis with varimax rotation
/// let mut fa = FactorAnalysis::new()
///     .with_n_factors(2)
///     .with_rotation(Rotation::Varimax);
///
/// let x = array![
///     [1.0, 2.0, 3.0, 4.0],
///     [4.0, 5.0, 6.0, 7.0],
///     [7.0, 8.0, 9.0, 10.0],
///     [2.0, 3.0, 4.0, 5.0],
///     [5.0, 6.0, 7.0, 8.0]
/// ];
///
/// fa.fit(&x).unwrap();
///
/// // Transform data to factor scores
/// let scores = fa.transform(&x).unwrap();
/// assert_eq!(scores.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAnalysis {
    // Configuration
    /// Number of factors to extract
    n_factors: Option<usize>,
    /// Rotation method
    rotation: Rotation,
    /// SVD method for initial estimation
    svd_method: FaSvdMethod,
    /// Convergence tolerance
    tol: f64,
    /// Maximum iterations for EM algorithm
    max_iter: usize,
    /// Power parameter for promax rotation
    promax_power: f64,
    /// Random state for reproducibility
    random_state: Option<u64>,

    // Fitted state
    /// Factor loadings (n_features × n_factors)
    loadings: Option<Array2<f64>>,
    /// Per-feature noise variance
    noise_variance: Option<Array1<f64>>,
    /// Communalities (proportion of variance explained by factors)
    communalities: Option<Array1<f64>>,
    /// Explained variance per factor
    explained_variance: Option<Array1<f64>>,
    /// Explained variance ratio per factor
    explained_variance_ratio: Option<Array1<f64>>,
    /// Per-feature mean from training data
    mean: Option<Array1<f64>>,
    /// Number of factors determined during fitting
    n_factors_fitted: Option<usize>,
    /// Number of features in input
    n_features_in: Option<usize>,
    /// Number of EM iterations performed
    n_iter: Option<usize>,
    /// Final log-likelihood
    log_likelihood: Option<f64>,
    /// Factor correlation matrix (for oblique rotations)
    factor_correlation: Option<Array2<f64>>,
}

impl FactorAnalysis {
    /// Create a new Factor Analysis with default settings.
    ///
    /// Default configuration:
    /// - `n_factors`: None (must be specified or defaults to n_features)
    /// - `rotation`: None
    /// - `tol`: 1e-3
    /// - `max_iter`: 1000
    pub fn new() -> Self {
        Self {
            n_factors: None,
            rotation: Rotation::None,
            svd_method: FaSvdMethod::Randomized,
            tol: 1e-3,
            max_iter: 1000,
            promax_power: 4.0,
            random_state: None,
            loadings: None,
            noise_variance: None,
            communalities: None,
            explained_variance: None,
            explained_variance_ratio: None,
            mean: None,
            n_factors_fitted: None,
            n_features_in: None,
            n_iter: None,
            log_likelihood: None,
            factor_correlation: None,
        }
    }

    /// Set the number of factors to extract.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of factors
    ///
    /// # Panics
    ///
    /// Panics if n is 0.
    pub fn with_n_factors(mut self, n: usize) -> Self {
        assert!(n > 0, "n_factors must be positive");
        self.n_factors = Some(n);
        self
    }

    /// Set the rotation method.
    ///
    /// # Arguments
    ///
    /// * `rotation` - Rotation method to apply after extraction
    pub fn with_rotation(mut self, rotation: Rotation) -> Self {
        self.rotation = rotation;
        self
    }

    /// Set the SVD method for initial estimation.
    ///
    /// # Arguments
    ///
    /// * `method` - SVD method to use
    pub fn with_svd_method(mut self, method: FaSvdMethod) -> Self {
        self.svd_method = method;
        self
    }

    /// Set the convergence tolerance.
    ///
    /// # Arguments
    ///
    /// * `tol` - Convergence tolerance for EM algorithm
    ///
    /// # Panics
    ///
    /// Panics if tol is not positive.
    pub fn with_tol(mut self, tol: f64) -> Self {
        assert!(tol > 0.0, "tol must be positive");
        self.tol = tol;
        self
    }

    /// Set the maximum number of iterations.
    ///
    /// # Arguments
    ///
    /// * `max_iter` - Maximum iterations for EM algorithm
    ///
    /// # Panics
    ///
    /// Panics if max_iter is 0.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        assert!(max_iter > 0, "max_iter must be positive");
        self.max_iter = max_iter;
        self
    }

    /// Set the power parameter for promax rotation.
    ///
    /// Higher values produce more simple structure but allow more correlation.
    ///
    /// # Arguments
    ///
    /// * `power` - Power for promax rotation (typical: 2-4)
    ///
    /// # Panics
    ///
    /// Panics if power is less than 1.
    pub fn with_promax_power(mut self, power: f64) -> Self {
        assert!(power >= 1.0, "promax_power must be >= 1");
        self.promax_power = power;
        self
    }

    /// Set the random state for reproducibility.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get the factor loadings matrix.
    ///
    /// Returns a matrix of shape (n_features, n_factors) where each column
    /// represents a factor and each row represents a feature.
    ///
    /// # Interpretation
    ///
    /// - Values close to 1 or -1 indicate strong relationship with the factor
    /// - Values close to 0 indicate weak relationship
    /// - Sign indicates direction of relationship
    ///
    /// Returns `None` if not fitted.
    pub fn loadings(&self) -> Option<&Array2<f64>> {
        self.loadings.as_ref()
    }

    /// Get the noise variance for each feature.
    ///
    /// This is the variance not explained by the common factors (unique variance).
    ///
    /// Returns `None` if not fitted.
    pub fn noise_variance(&self) -> Option<&Array1<f64>> {
        self.noise_variance.as_ref()
    }

    /// Get the communalities for each feature.
    ///
    /// Communality is the proportion of each feature's variance explained by
    /// the common factors. High communality means the factors explain the variable well.
    ///
    /// communality_j = 1 - noise_variance_j / total_variance_j
    ///
    /// Returns `None` if not fitted.
    pub fn communalities(&self) -> Option<&Array1<f64>> {
        self.communalities.as_ref()
    }

    /// Get the explained variance for each factor.
    ///
    /// Returns `None` if not fitted.
    pub fn explained_variance(&self) -> Option<&Array1<f64>> {
        self.explained_variance.as_ref()
    }

    /// Get the explained variance ratio for each factor.
    ///
    /// Returns `None` if not fitted.
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    /// Get the per-feature mean from training data.
    ///
    /// Returns `None` if not fitted.
    pub fn mean(&self) -> Option<&Array1<f64>> {
        self.mean.as_ref()
    }

    /// Get the number of factors determined during fitting.
    ///
    /// Returns `None` if not fitted.
    pub fn n_factors_(&self) -> Option<usize> {
        self.n_factors_fitted
    }

    /// Get the number of iterations performed.
    ///
    /// Returns `None` if not fitted.
    pub fn n_iter_(&self) -> Option<usize> {
        self.n_iter
    }

    /// Get the final log-likelihood.
    ///
    /// Returns `None` if not fitted.
    pub fn log_likelihood_(&self) -> Option<f64> {
        self.log_likelihood
    }

    /// Get the factor correlation matrix (for oblique rotations).
    ///
    /// For orthogonal rotations, this is the identity matrix.
    /// For promax rotation, factors may be correlated.
    ///
    /// Returns `None` if not fitted.
    pub fn factor_correlation(&self) -> Option<&Array2<f64>> {
        self.factor_correlation.as_ref()
    }

    /// Get the model covariance matrix.
    ///
    /// The covariance implied by the factor model:
    /// Σ = L·L^T + Ψ
    ///
    /// where L is the loadings matrix and Ψ is the diagonal noise covariance.
    ///
    /// Returns `None` if not fitted.
    pub fn get_covariance(&self) -> Option<Array2<f64>> {
        let loadings = self.loadings.as_ref()?;
        let noise_var = self.noise_variance.as_ref()?;
        let n_features = loadings.nrows();

        // Σ = L·L^T + Ψ
        let mut cov = loadings.dot(&loadings.t());

        // Add noise variance on diagonal
        for i in 0..n_features {
            cov[[i, i]] += noise_var[i];
        }

        Some(cov)
    }

    /// Compute factor scores using the Bartlett method.
    ///
    /// Bartlett scores: F = (L^T Ψ^{-1} L)^{-1} L^T Ψ^{-1} (X - μ)
    ///
    /// These are weighted least squares estimates of the factor scores.
    fn compute_bartlett_scores(&self, x_centered: &Array2<f64>) -> Result<Array2<f64>> {
        let loadings = self
            .loadings
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("compute_bartlett_scores"))?;
        let noise_var = self
            .noise_variance
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("compute_bartlett_scores"))?;

        let n_samples = x_centered.nrows();
        let n_factors = loadings.ncols();

        // Compute Ψ^{-1} (diagonal, so just invert elements)
        let psi_inv: Array1<f64> =
            noise_var.mapv(|v| if v > 1e-10 { 1.0 / v } else { 1.0 / 1e-10 });

        // L^T Ψ^{-1} L (n_factors × n_factors)
        let mut ltpsiinvl = Array2::<f64>::zeros((n_factors, n_factors));
        for i in 0..n_factors {
            for j in 0..n_factors {
                let mut sum = 0.0;
                for k in 0..loadings.nrows() {
                    sum += loadings[[k, i]] * psi_inv[k] * loadings[[k, j]];
                }
                ltpsiinvl[[i, j]] = sum;
            }
        }

        // Invert (L^T Ψ^{-1} L)
        let inv = invert_matrix(&ltpsiinvl)?;

        // L^T Ψ^{-1}
        let mut ltpsiinv = Array2::<f64>::zeros((n_factors, loadings.nrows()));
        for i in 0..n_factors {
            for j in 0..loadings.nrows() {
                ltpsiinv[[i, j]] = loadings[[j, i]] * psi_inv[j];
            }
        }

        // Full transformation matrix: (L^T Ψ^{-1} L)^{-1} L^T Ψ^{-1}
        let transform_matrix = inv.dot(&ltpsiinv);

        // Compute scores: F = X_centered · transform_matrix^T
        let mut scores = Array2::zeros((n_samples, n_factors));
        for i in 0..n_samples {
            for f in 0..n_factors {
                let mut sum = 0.0;
                for j in 0..x_centered.ncols() {
                    sum += x_centered[[i, j]] * transform_matrix[[f, j]];
                }
                scores[[i, f]] = sum;
            }
        }

        Ok(scores)
    }

    /// Fit the factor analysis model using EM algorithm.
    fn fit_em(&mut self, x: &Array2<f64>) -> Result<()> {
        let (n_samples, n_features) = x.dim();

        // Compute mean and center data
        let mean = x
            .mean_axis(Axis(0))
            .ok_or_else(|| FerroError::numerical("Failed to compute mean"))?;
        let x_centered = x - &mean;

        // Determine number of factors
        let n_factors = self.n_factors.unwrap_or(n_features).min(n_features);

        // Compute sample covariance
        let cov = compute_covariance(&x_centered);

        // Initialize with PCA-like solution
        let (mut loadings, mut noise_var) = self.initialize_from_svd(&x_centered, n_factors)?;

        // EM algorithm
        let mut prev_log_likelihood = f64::NEG_INFINITY;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // E-step: Compute expected sufficient statistics
            // E[F|X] and E[F F^T|X]
            let (exp_f, exp_fft) = self.e_step(&x_centered, &loadings, &noise_var)?;

            // M-step: Update loadings and noise variance
            let (new_loadings, new_noise_var) =
                self.m_step(&x_centered, &cov, &exp_f, &exp_fft, n_factors)?;

            // Compute log-likelihood
            let log_likelihood =
                self.compute_log_likelihood(&cov, &new_loadings, &new_noise_var, n_samples)?;

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                loadings = new_loadings;
                noise_var = new_noise_var;
                break;
            }

            loadings = new_loadings;
            noise_var = new_noise_var;
            prev_log_likelihood = log_likelihood;
        }

        // Compute final log-likelihood
        let log_likelihood = self.compute_log_likelihood(&cov, &loadings, &noise_var, n_samples)?;

        // Apply rotation if requested
        let (rotated_loadings, factor_corr) = self.apply_rotation(&loadings)?;

        // Compute communalities and explained variance
        let communalities = self.compute_communalities(&rotated_loadings, &noise_var, &cov);
        let (explained_var, explained_var_ratio) =
            self.compute_explained_variance(&rotated_loadings, &cov);

        // Store fitted state
        self.loadings = Some(rotated_loadings);
        self.noise_variance = Some(noise_var);
        self.communalities = Some(communalities);
        self.explained_variance = Some(explained_var);
        self.explained_variance_ratio = Some(explained_var_ratio);
        self.mean = Some(mean);
        self.n_factors_fitted = Some(n_factors);
        self.n_features_in = Some(n_features);
        self.n_iter = Some(n_iter);
        self.log_likelihood = Some(log_likelihood);
        self.factor_correlation = Some(factor_corr);

        Ok(())
    }

    /// Initialize loadings and noise variance from SVD.
    fn initialize_from_svd(
        &self,
        x_centered: &Array2<f64>,
        n_factors: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let (n_samples, n_features) = x_centered.dim();

        // Compute SVD via faer
        let (_u, s, vt) = crate::linalg::thin_svd(x_centered)?;

        // Initialize loadings from top singular vectors
        let n_comp = n_factors.min(s.len()).min(n_features);
        let mut loadings = Array2::zeros((n_features, n_comp));

        // Loadings = V * S / sqrt(n_samples - 1)
        let scale = ((n_samples - 1).max(1) as f64).sqrt();
        for j in 0..n_features {
            for k in 0..n_comp {
                loadings[[j, k]] = vt[[k, j]] * s[k] / scale;
            }
        }

        // Initialize noise variance as residual variance
        // Noise = diag(Cov) - diag(L L^T)
        let cov_diag: Array1<f64> = x_centered.var_axis(Axis(0), 1.0);
        let mut noise_var = Array1::zeros(n_features);

        for j in 0..n_features {
            let loading_var: f64 = (0..n_comp).map(|k| loadings[[j, k]].powi(2)).sum();
            noise_var[j] = (cov_diag[j] - loading_var).max(1e-10);
        }

        Ok((loadings, noise_var))
    }

    /// E-step: Compute expected factor scores and their outer product.
    #[allow(clippy::type_complexity)]
    fn e_step(
        &self,
        x_centered: &Array2<f64>,
        loadings: &Array2<f64>,
        noise_var: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (n_samples, _n_features) = x_centered.dim();
        let n_factors = loadings.ncols();

        // Compute Ψ^{-1}
        let psi_inv: Array1<f64> =
            noise_var.mapv(|v| if v > 1e-10 { 1.0 / v } else { 1.0 / 1e-10 });

        // Compute L^T Ψ^{-1} L + I (n_factors × n_factors)
        let mut beta_inv = Array2::<f64>::zeros((n_factors, n_factors));
        for i in 0..n_factors {
            for j in 0..n_factors {
                let mut sum = 0.0;
                for k in 0..loadings.nrows() {
                    sum += loadings[[k, i]] * psi_inv[k] * loadings[[k, j]];
                }
                beta_inv[[i, j]] = sum;
                if i == j {
                    beta_inv[[i, j]] += 1.0;
                }
            }
        }

        // Invert to get β = (L^T Ψ^{-1} L + I)^{-1}
        let beta = invert_matrix(&beta_inv)?;

        // β L^T Ψ^{-1}
        let mut bltp = Array2::<f64>::zeros((n_factors, loadings.nrows()));
        for f in 0..n_factors {
            for j in 0..loadings.nrows() {
                let mut sum = 0.0;
                for k in 0..n_factors {
                    sum += beta[[f, k]] * loadings[[j, k]] * psi_inv[j];
                }
                bltp[[f, j]] = sum;
            }
        }

        // E[F|X] = β L^T Ψ^{-1} X^T
        let mut exp_f = Array2::zeros((n_samples, n_factors));
        for i in 0..n_samples {
            for f in 0..n_factors {
                let mut sum = 0.0;
                for j in 0..x_centered.ncols() {
                    sum += bltp[[f, j]] * x_centered[[i, j]];
                }
                exp_f[[i, f]] = sum;
            }
        }

        // E[F F^T|X] = β + E[F|X] E[F|X]^T (averaged)
        // We compute the average over samples: 1/n Σ E[F F^T|X_i]
        let mut exp_fft = beta;
        let eft_outer = exp_f.t().dot(&exp_f) / n_samples as f64;
        for i in 0..n_factors {
            for j in 0..n_factors {
                exp_fft[[i, j]] += eft_outer[[i, j]];
            }
        }

        Ok((exp_f, exp_fft))
    }

    /// M-step: Update loadings and noise variance.
    fn m_step(
        &self,
        x_centered: &Array2<f64>,
        cov: &Array2<f64>,
        exp_f: &Array2<f64>,
        exp_fft: &Array2<f64>,
        _n_factors: usize,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let (n_samples, n_features) = x_centered.dim();
        let _n_factors = exp_f.ncols();

        // Update loadings: L_new = S_XF * (S_FF)^{-1}
        // where S_XF = (1/n) X^T E[F] and S_FF = E[F F^T]

        // S_XF = (1/n) X^T E[F]
        let s_xf = x_centered.t().dot(exp_f) / n_samples as f64;

        // Invert E[F F^T]
        let exp_fft_inv = invert_matrix(exp_fft)?;

        // New loadings
        let new_loadings = s_xf.dot(&exp_fft_inv);

        // Update noise variance: Ψ_new = diag(S - L_new S_XF^T)
        // where S is the sample covariance
        let lsxft = new_loadings.dot(&s_xf.t());
        let mut new_noise_var = Array1::zeros(n_features);

        for j in 0..n_features {
            new_noise_var[j] = (cov[[j, j]] - lsxft[[j, j]]).max(1e-10);
        }

        Ok((new_loadings, new_noise_var))
    }

    /// Compute log-likelihood of the factor model.
    fn compute_log_likelihood(
        &self,
        cov: &Array2<f64>,
        loadings: &Array2<f64>,
        noise_var: &Array1<f64>,
        n_samples: usize,
    ) -> Result<f64> {
        let n_features = cov.nrows();

        // Model covariance: Σ = L L^T + Ψ
        let mut model_cov = loadings.dot(&loadings.t());
        for i in 0..n_features {
            model_cov[[i, i]] += noise_var[i];
        }

        // Log-likelihood = -n/2 * (p*log(2π) + log|Σ| + tr(Σ^{-1} S))
        let log_det = log_determinant(&model_cov)?;
        let model_cov_inv = invert_matrix(&model_cov)?;

        // tr(Σ^{-1} S)
        let mut trace = 0.0;
        for i in 0..n_features {
            for j in 0..n_features {
                trace += model_cov_inv[[i, j]] * cov[[j, i]];
            }
        }

        let log_2pi = (2.0 * std::f64::consts::PI).ln();
        let log_likelihood =
            -0.5 * n_samples as f64 * ((n_features as f64).mul_add(log_2pi, log_det) + trace);

        Ok(log_likelihood)
    }

    /// Apply rotation to loadings.
    fn apply_rotation(&self, loadings: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let n_factors = loadings.ncols();

        match self.rotation {
            Rotation::None => {
                let identity = Array2::from_diag(&Array1::ones(n_factors));
                Ok((loadings.clone(), identity))
            }
            Rotation::Varimax => {
                let (rotated, rotation_matrix) = varimax_rotation(loadings)?;
                let factor_corr = Array2::from_diag(&Array1::ones(n_factors));
                let _ = rotation_matrix; // Orthogonal rotation
                Ok((rotated, factor_corr))
            }
            Rotation::Quartimax => {
                let (rotated, rotation_matrix) = quartimax_rotation(loadings)?;
                let factor_corr = Array2::from_diag(&Array1::ones(n_factors));
                let _ = rotation_matrix;
                Ok((rotated, factor_corr))
            }
            Rotation::Promax => {
                // Promax starts with varimax
                let (varimax_loadings, _) = varimax_rotation(loadings)?;
                let (rotated, factor_corr) = promax_rotation(&varimax_loadings, self.promax_power)?;
                Ok((rotated, factor_corr))
            }
        }
    }

    /// Compute communalities from loadings.
    fn compute_communalities(
        &self,
        loadings: &Array2<f64>,
        noise_var: &Array1<f64>,
        cov: &Array2<f64>,
    ) -> Array1<f64> {
        let n_features = loadings.nrows();
        let mut communalities = Array1::zeros(n_features);

        for j in 0..n_features {
            let loading_var: f64 = loadings.row(j).iter().map(|v| v.powi(2)).sum();
            let total_var = cov[[j, j]];
            if total_var > 1e-10 {
                communalities[j] = loading_var / (loading_var + noise_var[j]);
            } else {
                communalities[j] = 0.0;
            }
        }

        communalities
    }

    /// Compute explained variance from loadings.
    fn compute_explained_variance(
        &self,
        loadings: &Array2<f64>,
        cov: &Array2<f64>,
    ) -> (Array1<f64>, Array1<f64>) {
        let n_factors = loadings.ncols();
        let n_features = loadings.nrows();

        // Variance explained by each factor = sum of squared loadings for that factor
        let mut explained_var = Array1::zeros(n_factors);
        for f in 0..n_factors {
            for j in 0..n_features {
                explained_var[f] += loadings[[j, f]].powi(2);
            }
        }

        // Total variance = trace of covariance matrix
        let total_var: f64 = (0..n_features).map(|i| cov[[i, i]]).sum();

        let explained_var_ratio = if total_var > 0.0 {
            explained_var.mapv(|v| v / total_var)
        } else {
            Array1::zeros(n_factors)
        };

        (explained_var, explained_var_ratio)
    }
}

impl Default for FactorAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for FactorAnalysis {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        crate::validation::validate_unsupervised_input(x)?;
        self.fit_em(x)
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        crate::validation::validate_transform_input(x, self.n_features_in.unwrap())?;

        let mean = self.mean.as_ref().unwrap();
        let x_centered = x - mean;

        // Compute factor scores using Bartlett method
        self.compute_bartlett_scores(&x_centered)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;

        let n_factors = self.n_factors_fitted.unwrap();
        if x.ncols() != n_factors {
            return Err(FerroError::shape_mismatch(
                format!("({}, {})", x.nrows(), n_factors),
                format!("({}, {})", x.nrows(), x.ncols()),
            ));
        }

        let loadings = self.loadings.as_ref().unwrap();
        let mean = self.mean.as_ref().unwrap();

        // X_reconstructed = F · L^T + μ
        let x_reconstructed = x.dot(&loadings.t()) + mean;

        Ok(x_reconstructed)
    }

    fn is_fitted(&self) -> bool {
        self.loadings.is_some() && self.noise_variance.is_some() && self.mean.is_some()
    }

    fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_factors = self.n_factors_fitted?;
        Some((0..n_factors).map(|i| format!("factor{i}")).collect())
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_factors_fitted
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Compute sample covariance matrix.
fn compute_covariance(x_centered: &Array2<f64>) -> Array2<f64> {
    let n_samples = x_centered.nrows();
    let n_features = x_centered.ncols();

    let mut cov = Array2::zeros((n_features, n_features));
    let scale = (n_samples - 1).max(1) as f64;

    for i in 0..n_features {
        for j in i..n_features {
            let mut sum = 0.0;
            for k in 0..n_samples {
                sum += x_centered[[k, i]] * x_centered[[k, j]];
            }
            cov[[i, j]] = sum / scale;
            cov[[j, i]] = cov[[i, j]];
        }
    }

    cov
}

/// Invert a matrix using Cholesky or LU decomposition.
fn invert_matrix(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::invalid_input("Matrix must be square"));
    }

    let mat = nalgebra::DMatrix::from_fn(n, n, |i, j| a[[i, j]]);

    // Try Cholesky first (for positive definite matrices)
    if let Some(chol) = mat.clone().cholesky() {
        let inv = chol.inverse();
        return Ok(Array2::from_shape_fn((n, n), |(i, j)| inv[(i, j)]));
    }

    // Fall back to LU decomposition
    let lu = mat.clone().lu();
    if let Some(inv) = lu.try_inverse() {
        return Ok(Array2::from_shape_fn((n, n), |(i, j)| inv[(i, j)]));
    }

    // Last resort: pseudoinverse via SVD
    let svd = mat.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| FerroError::numerical("SVD failed in matrix inversion"))?;
    let s = svd.singular_values;
    let vt = svd
        .v_t
        .ok_or_else(|| FerroError::numerical("SVD failed in matrix inversion"))?;
    let tol = 1e-10 * s.iter().copied().fold(0.0_f64, f64::max);
    let mut s_inv = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        if s[i] > tol {
            s_inv[(i, i)] = 1.0 / s[i];
        }
    }
    let pseudo_inv = vt.transpose() * &s_inv * u.transpose();
    Ok(Array2::from_shape_fn((n, n), |(i, j)| pseudo_inv[(i, j)]))
}

/// Compute log-determinant of a matrix.
fn log_determinant(a: &Array2<f64>) -> Result<f64> {
    let n = a.nrows();
    let mat = nalgebra::DMatrix::from_fn(n, n, |i, j| a[[i, j]]);

    // Use LU decomposition for determinant
    let lu = mat.lu();
    let det = lu.determinant();

    if det <= 0.0 {
        return Err(FerroError::numerical(
            "Matrix is singular or not positive definite (determinant <= 0)",
        ));
    }
    Ok(det.ln())
}

/// Varimax rotation (orthogonal).
///
/// Maximizes the sum of variances of squared loadings within each factor.
fn varimax_rotation(loadings: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    orthomax_rotation(loadings, 1.0)
}

/// Quartimax rotation (orthogonal).
///
/// Maximizes the fourth power of loadings, tends to produce a general factor.
fn quartimax_rotation(loadings: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    orthomax_rotation(loadings, 0.0)
}

/// General orthomax rotation.
///
/// gamma = 0: quartimax
/// gamma = 1: varimax
/// gamma = p/2: equamax
fn orthomax_rotation(loadings: &Array2<f64>, gamma: f64) -> Result<(Array2<f64>, Array2<f64>)> {
    let (n_features, n_factors) = loadings.dim();

    if n_factors < 2 {
        return Ok((
            loadings.clone(),
            Array2::from_diag(&Array1::ones(n_factors)),
        ));
    }

    let max_iter = 100;
    let tol = 1e-5;

    let mut rotated = loadings.clone();
    let mut rotation = Array2::from_diag(&Array1::ones(n_factors));

    for _ in 0..max_iter {
        let mut max_change = 0.0_f64;

        // Apply pairwise rotations
        for i in 0..n_factors {
            for j in (i + 1)..n_factors {
                // Compute rotation angle
                let (angle, changed) = compute_rotation_angle(&rotated, i, j, gamma, n_features);
                max_change = max_change.max(changed);

                if changed > tol {
                    // Apply rotation
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    for k in 0..n_features {
                        let xi = rotated[[k, i]];
                        let xj = rotated[[k, j]];
                        rotated[[k, i]] = cos_a.mul_add(xi, sin_a * xj);
                        rotated[[k, j]] = (-sin_a).mul_add(xi, cos_a * xj);
                    }

                    // Update rotation matrix
                    for k in 0..n_factors {
                        let ri = rotation[[k, i]];
                        let rj = rotation[[k, j]];
                        rotation[[k, i]] = cos_a.mul_add(ri, sin_a * rj);
                        rotation[[k, j]] = (-sin_a).mul_add(ri, cos_a * rj);
                    }
                }
            }
        }

        if max_change < tol {
            break;
        }
    }

    Ok((rotated, rotation))
}

/// Compute rotation angle for orthomax criterion.
fn compute_rotation_angle(
    loadings: &Array2<f64>,
    i: usize,
    j: usize,
    gamma: f64,
    n_features: usize,
) -> (f64, f64) {
    let mut a = 0.0;
    let mut b = 0.0;
    let mut c = 0.0;
    let mut d = 0.0;

    for k in 0..n_features {
        let li = loadings[[k, i]];
        let lj = loadings[[k, j]];

        let u = li.mul_add(li, -(lj * lj));
        let v = 2.0 * li * lj;

        a += u;
        b += v;
        c += u.mul_add(u, -(v * v));
        d += 2.0 * u * v;
    }

    let x = d - 2.0 * gamma * a * b / n_features as f64;
    let y = c - gamma * (a * a - b * b) / n_features as f64;

    let angle = 0.25 * x.atan2(y);
    let change = (x * x + y * y).sqrt();

    (angle, change)
}

/// Promax rotation (oblique).
///
/// Based on varimax solution raised to a power, allowing correlated factors.
fn promax_rotation(
    varimax_loadings: &Array2<f64>,
    power: f64,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let (n_features, n_factors) = varimax_loadings.dim();

    // Create target matrix: sign(L) * |L|^power
    let mut target = Array2::zeros((n_features, n_factors));
    for i in 0..n_features {
        for j in 0..n_factors {
            let v = varimax_loadings[[i, j]];
            target[[i, j]] = v.signum() * v.abs().powf(power);
        }
    }

    // Compute transformation: T = (L^T L)^{-1} L^T target
    let ltl = varimax_loadings.t().dot(varimax_loadings);

    // Add small regularization for numerical stability
    let mut ltl_reg = ltl;
    for i in 0..n_factors {
        ltl_reg[[i, i]] += 1e-8;
    }

    let ltl_inv = invert_matrix_with_pseudoinverse(&ltl_reg)?;
    let ltt = varimax_loadings.t().dot(&target);
    let transform = ltl_inv.dot(&ltt);

    // Apply transformation
    let rotated = varimax_loadings.dot(&transform);

    // Normalize columns to unit length for standard promax
    let mut rotated_normalized = rotated.clone();
    for j in 0..n_factors {
        let norm: f64 = rotated.column(j).iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            rotated_normalized
                .column_mut(j)
                .iter_mut()
                .for_each(|v| *v /= norm);
        }
    }

    // Rescale to match original variance
    let mut rotated_final = Array2::zeros((n_features, n_factors));
    for j in 0..n_factors {
        let orig_var: f64 = varimax_loadings.column(j).iter().map(|v| v * v).sum();
        let new_var: f64 = rotated_normalized.column(j).iter().map(|v| v * v).sum();
        let scale = if new_var > 1e-10 {
            (orig_var / new_var).sqrt()
        } else {
            1.0
        };

        for i in 0..n_features {
            rotated_final[[i, j]] = rotated_normalized[[i, j]] * scale;
        }
    }

    // Compute factor correlation matrix
    // Φ = D^{-1} T^T T D^{-1} where D = diag(T^T T)^{1/2}
    let ttt = transform.t().dot(&transform);
    let mut d_inv = Array1::zeros(n_factors);
    for i in 0..n_factors {
        d_inv[i] = if ttt[[i, i]] > 1e-10 {
            1.0 / ttt[[i, i]].sqrt()
        } else {
            1.0
        };
    }

    let mut factor_corr = Array2::zeros((n_factors, n_factors));
    for i in 0..n_factors {
        for j in 0..n_factors {
            factor_corr[[i, j]] = ttt[[i, j]] * d_inv[i] * d_inv[j];
        }
    }

    Ok((rotated_final, factor_corr))
}

/// Invert a matrix using pseudoinverse for better numerical stability.
fn invert_matrix_with_pseudoinverse(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(FerroError::invalid_input("Matrix must be square"));
    }

    let mat = nalgebra::DMatrix::from_fn(n, n, |i, j| a[[i, j]]);

    // Try Cholesky first (for positive definite matrices)
    if let Some(chol) = mat.clone().cholesky() {
        let inv = chol.inverse();
        return Ok(Array2::from_shape_fn((n, n), |(i, j)| inv[(i, j)]));
    }

    // Fall back to pseudoinverse via SVD
    let svd = mat.svd(true, true);
    let u = svd.u.ok_or_else(|| FerroError::numerical("SVD failed"))?;
    let s = svd.singular_values;
    let vt = svd.v_t.ok_or_else(|| FerroError::numerical("SVD failed"))?;

    // Compute pseudoinverse: V @ S^{-1} @ U^T
    let tol = 1e-10 * s.iter().copied().fold(0.0_f64, f64::max);
    let mut s_inv = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        if s[i] > tol {
            s_inv[(i, i)] = 1.0 / s[i];
        }
    }

    let v_s_inv = vt.transpose() * &s_inv;
    let pseudo_inv = v_s_inv * u.transpose();

    Ok(Array2::from_shape_fn((n, n), |(i, j)| pseudo_inv[(i, j)]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const TOLERANCE: f64 = 1e-3;

    #[test]
    fn test_factor_analysis_basic() {
        let mut fa = FactorAnalysis::new().with_n_factors(2);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0],
            [8.0, 9.0, 10.0, 11.0]
        ];

        fa.fit(&x).unwrap();

        assert!(fa.is_fitted());
        assert_eq!(fa.n_factors_(), Some(2));
        assert_eq!(fa.n_features_in(), Some(4));
    }

    #[test]
    fn test_factor_analysis_loadings() {
        let mut fa = FactorAnalysis::new().with_n_factors(2);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0],
            [8.0, 9.0, 10.0, 11.0],
            [3.0, 4.0, 5.0, 6.0]
        ];

        fa.fit(&x).unwrap();

        let loadings = fa.loadings().unwrap();
        assert_eq!(loadings.nrows(), 4); // n_features
        assert_eq!(loadings.ncols(), 2); // n_factors
    }

    #[test]
    fn test_factor_analysis_communalities() {
        let mut fa = FactorAnalysis::new().with_n_factors(2);

        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        fa.fit(&x).unwrap();

        let communalities = fa.communalities().unwrap();
        assert_eq!(communalities.len(), 3);

        // Communalities should be between 0 and 1
        for &c in communalities.iter() {
            assert!(
                c >= 0.0 && c <= 1.0 + TOLERANCE,
                "Communality {} out of range",
                c
            );
        }
    }

    #[test]
    fn test_factor_analysis_transform() {
        let mut fa = FactorAnalysis::new().with_n_factors(2);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0]
        ];

        fa.fit(&x).unwrap();
        let scores = fa.transform(&x).unwrap();

        assert_eq!(scores.nrows(), 5);
        assert_eq!(scores.ncols(), 2);
    }

    #[test]
    fn test_factor_analysis_inverse_transform() {
        let mut fa = FactorAnalysis::new().with_n_factors(2);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0]
        ];

        fa.fit(&x).unwrap();
        let scores = fa.transform(&x).unwrap();
        let x_reconstructed = fa.inverse_transform(&scores).unwrap();

        assert_eq!(x_reconstructed.dim(), x.dim());

        // Reconstruction should be reasonably close (not exact due to noise model)
        // The mean should be preserved
        let orig_mean = x.mean_axis(Axis(0)).unwrap();
        let recon_mean = x_reconstructed.mean_axis(Axis(0)).unwrap();

        for i in 0..orig_mean.len() {
            assert!(
                (orig_mean[i] - recon_mean[i]).abs() < 1.0,
                "Mean mismatch at {}: {} vs {}",
                i,
                orig_mean[i],
                recon_mean[i]
            );
        }
    }

    #[test]
    fn test_factor_analysis_varimax() {
        let mut fa = FactorAnalysis::new()
            .with_n_factors(2)
            .with_rotation(Rotation::Varimax);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0],
            [8.0, 9.0, 10.0, 11.0]
        ];

        fa.fit(&x).unwrap();

        assert!(fa.is_fitted());
        let loadings = fa.loadings().unwrap();
        assert_eq!(loadings.dim(), (4, 2));
    }

    #[test]
    fn test_factor_analysis_quartimax() {
        let mut fa = FactorAnalysis::new()
            .with_n_factors(2)
            .with_rotation(Rotation::Quartimax);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0],
            [8.0, 9.0, 10.0, 11.0]
        ];

        fa.fit(&x).unwrap();

        assert!(fa.is_fitted());
    }

    #[test]
    fn test_factor_analysis_promax() {
        let mut fa = FactorAnalysis::new()
            .with_n_factors(2)
            .with_rotation(Rotation::Promax)
            .with_promax_power(4.0);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0],
            [8.0, 9.0, 10.0, 11.0]
        ];

        fa.fit(&x).unwrap();

        assert!(fa.is_fitted());

        // Promax allows correlated factors
        let factor_corr = fa.factor_correlation().unwrap();
        assert_eq!(factor_corr.dim(), (2, 2));
    }

    #[test]
    fn test_factor_analysis_explained_variance() {
        let mut fa = FactorAnalysis::new().with_n_factors(2);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0]
        ];

        fa.fit(&x).unwrap();

        let evr = fa.explained_variance_ratio().unwrap();
        assert_eq!(evr.len(), 2);

        // Ratios should be non-negative
        for &r in evr.iter() {
            assert!(r >= 0.0, "Negative variance ratio: {}", r);
        }
    }

    #[test]
    fn test_factor_analysis_covariance() {
        let mut fa = FactorAnalysis::new().with_n_factors(2);

        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        fa.fit(&x).unwrap();

        let cov = fa.get_covariance().unwrap();
        assert_eq!(cov.dim(), (3, 3));

        // Covariance should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (cov[[i, j]] - cov[[j, i]]).abs() < TOLERANCE,
                    "Covariance not symmetric at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_factor_analysis_feature_names() {
        let mut fa = FactorAnalysis::new().with_n_factors(3);

        let x = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0, 10.0, 11.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0, 8.0, 9.0]
        ];

        fa.fit(&x).unwrap();

        let names = fa.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["factor0", "factor1", "factor2"]);
    }

    #[test]
    fn test_factor_analysis_not_fitted() {
        let fa = FactorAnalysis::new();
        let x = array![[1.0, 2.0]];

        assert!(!fa.is_fitted());
        assert!(fa.transform(&x).is_err());
    }

    #[test]
    fn test_factor_analysis_noise_variance() {
        let mut fa = FactorAnalysis::new().with_n_factors(1);

        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        fa.fit(&x).unwrap();

        let noise_var = fa.noise_variance().unwrap();
        assert_eq!(noise_var.len(), 3);

        // Noise variance should be positive
        for &v in noise_var.iter() {
            assert!(v > 0.0, "Noise variance should be positive: {}", v);
        }
    }

    #[test]
    fn test_factor_analysis_single_factor() {
        let mut fa = FactorAnalysis::new().with_n_factors(1);

        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        fa.fit(&x).unwrap();

        assert_eq!(fa.n_factors_(), Some(1));

        let scores = fa.transform(&x).unwrap();
        assert_eq!(scores.ncols(), 1);
    }

    #[test]
    fn test_factor_analysis_convergence() {
        let mut fa = FactorAnalysis::new()
            .with_n_factors(2)
            .with_max_iter(100)
            .with_tol(1e-4);

        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 5.0, 6.0, 7.0],
            [7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0, 8.0],
            [8.0, 9.0, 10.0, 11.0]
        ];

        fa.fit(&x).unwrap();

        // Should have converged
        let n_iter = fa.n_iter_().unwrap();
        assert!(n_iter <= 100);

        // Log-likelihood should be computed
        let ll = fa.log_likelihood_().unwrap();
        assert!(ll.is_finite());
    }

    #[test]
    fn test_orthomax_single_factor() {
        // Single factor case should return unchanged
        let loadings = array![[0.8], [0.6], [0.7]];
        let (rotated, _) = varimax_rotation(&loadings).unwrap();

        assert_eq!(rotated.dim(), (3, 1));
    }
}
