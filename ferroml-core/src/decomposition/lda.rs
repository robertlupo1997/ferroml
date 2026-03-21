//! Linear Discriminant Analysis (LDA)
//!
//! Supervised dimensionality reduction that maximizes class separability.
//!
//! ## Mathematical Background
//!
//! LDA finds linear combinations of features that maximize the ratio of
//! between-class variance to within-class variance:
//!
//! - **Between-class scatter matrix**: S_b = Σ_c n_c (μ_c - μ)(μ_c - μ)^T
//! - **Within-class scatter matrix**: S_w = Σ_c Σ_{x∈c} (x - μ_c)(x - μ_c)^T
//!
//! LDA solves the generalized eigenvalue problem: S_b · w = λ · S_w · w
//!
//! ## Comparison with PCA
//!
//! | Aspect | PCA | LDA |
//! |--------|-----|-----|
//! | Type | Unsupervised | Supervised |
//! | Goal | Maximize variance | Maximize class separation |
//! | Max components | min(n_samples, n_features) | min(n_features, n_classes - 1) |
//! | Use case | General dim reduction | Classification preprocessing |
//!
//! ## Features
//!
//! - Supervised dimensionality reduction using class labels
//! - Explained variance ratios for component selection
//! - Class priors (learned or specified)
//! - Shrinkage regularization for small sample sizes
//! - SVD solver for numerical stability
//!
//! ## Example
//!
//! ```
//! use ferroml_core::decomposition::LDA;
//! use ndarray::array;
//!
//! let mut lda = LDA::new();
//!
//! // Two-class dataset
//! let x = array![
//!     [1.0, 2.0],
//!     [2.0, 3.0],
//!     [3.0, 3.0],
//!     [6.0, 7.0],
//!     [7.0, 8.0],
//!     [8.0, 7.0]
//! ];
//! let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//!
//! lda.fit(&x, &y).unwrap();
//!
//! // Transform data - projects onto discriminant direction
//! let x_lda = lda.transform(&x).unwrap();
//! assert_eq!(x_lda.ncols(), 1); // n_classes - 1 = 1 component
//!
//! // Classes should be more separated in transformed space
//! let class0_mean = x_lda.slice(ndarray::s![0..3, 0]).mean().unwrap();
//! let class1_mean = x_lda.slice(ndarray::s![3..6, 0]).mean().unwrap();
//! assert!((class0_mean - class1_mean).abs() > 1.0);
//! ```

// Allow common patterns in numerical/scientific code
#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::return_self_not_must_use)]

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::preprocessing::{check_is_fitted, check_shape};
use crate::{FerroError, Result};

/// SVD solver strategy for LDA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LdaSolver {
    /// Singular Value Decomposition - recommended for most cases.
    /// Does not compute the covariance matrix, works well when n_samples < n_features.
    #[default]
    Svd,
    /// Eigenvalue decomposition of covariance matrices.
    /// May be faster when n_samples >> n_features.
    Eigen,
}

/// Linear Discriminant Analysis (LDA).
///
/// Supervised dimensionality reduction that finds directions maximizing
/// class separation. Also usable as a classifier.
///
/// # Algorithm
///
/// 1. Compute class means μ_c and overall mean μ
/// 2. Compute within-class scatter: S_w = Σ_c Σ_{x∈c} (x - μ_c)(x - μ_c)^T
/// 3. Compute between-class scatter: S_b = Σ_c n_c (μ_c - μ)(μ_c - μ)^T
/// 4. Solve generalized eigenvalue problem S_b w = λ S_w w
/// 5. Keep top k eigenvectors (k ≤ n_classes - 1)
///
/// # Configuration
///
/// - `n_components`: Number of discriminant directions (default: min(n_classes-1, n_features))
/// - `solver`: Algorithm for eigendecomposition (SVD or Eigen)
/// - `shrinkage`: Regularization for S_w when n_samples is small
/// - `priors`: Class prior probabilities (default: estimate from data)
///
/// # Attributes (after fitting)
///
/// - `scalings_`: Projection matrix (n_features × n_components)
/// - `explained_variance_ratio_`: Percentage of discriminant info per component
/// - `means_`: Per-class means
/// - `xbar_`: Overall mean
/// - `classes_`: Unique class labels
/// - `priors_`: Class prior probabilities
///
/// # Example
///
/// ```
/// use ferroml_core::decomposition::LDA;
/// use ndarray::array;
///
/// // Three-class Iris-like data
/// let x = array![
///     [5.1, 3.5], [4.9, 3.0], [4.7, 3.2],  // Class 0
///     [7.0, 3.2], [6.4, 3.2], [6.9, 3.1],  // Class 1
///     [6.3, 3.3], [5.8, 2.7], [7.1, 3.0]   // Class 2
/// ];
/// let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
///
/// let mut lda = LDA::new().with_n_components(2);
/// lda.fit(&x, &y).unwrap();
///
/// // With 3 classes, max components = 2
/// let x_lda = lda.transform(&x).unwrap();
/// assert_eq!(x_lda.ncols(), 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LDA {
    // Configuration
    /// Number of components to keep (if None, uses min(n_classes-1, n_features))
    n_components: Option<usize>,
    /// Solver algorithm
    solver: LdaSolver,
    /// Shrinkage parameter for regularization (0 to 1, or None for automatic)
    shrinkage: Option<f64>,
    /// Class prior probabilities (if None, estimated from data)
    priors: Option<Vec<f64>>,
    /// Tolerance for eigenvalue computation
    tol: f64,

    // Fitted state
    /// Projection matrix (n_features × n_components)
    scalings: Option<Array2<f64>>,
    /// Explained variance ratio per component
    explained_variance_ratio: Option<Array1<f64>>,
    /// Per-class means (n_classes × n_features)
    means: Option<Array2<f64>>,
    /// Overall mean
    xbar: Option<Array1<f64>>,
    /// Unique class labels (sorted)
    classes: Option<Vec<f64>>,
    /// Fitted class priors
    priors_fitted: Option<Array1<f64>>,
    /// Eigenvalues of the discriminant functions
    eigenvalues: Option<Array1<f64>>,
    /// Number of components determined during fitting
    n_components_fitted: Option<usize>,
    /// Number of features in input
    n_features_in: Option<usize>,
    /// Coefficients for classification (n_classes × n_features)
    coef: Option<Array2<f64>>,
    /// Intercept for classification
    intercept: Option<Array1<f64>>,
}

impl LDA {
    /// Create a new LDA with default settings.
    ///
    /// Default configuration:
    /// - `n_components`: None (uses min(n_classes-1, n_features))
    /// - `solver`: SVD
    /// - `shrinkage`: None (no regularization)
    /// - `priors`: None (estimate from data)
    pub fn new() -> Self {
        Self {
            n_components: None,
            solver: LdaSolver::Svd,
            shrinkage: None,
            priors: None,
            tol: 1e-4,
            scalings: None,
            explained_variance_ratio: None,
            means: None,
            xbar: None,
            classes: None,
            priors_fitted: None,
            eigenvalues: None,
            n_components_fitted: None,
            n_features_in: None,
            coef: None,
            intercept: None,
        }
    }

    /// Set the number of components to keep.
    ///
    /// Must be <= min(n_classes - 1, n_features).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of components
    pub fn with_n_components(mut self, n: usize) -> Self {
        assert!(n > 0, "n_components must be positive");
        self.n_components = Some(n);
        self
    }

    /// Set the solver algorithm.
    ///
    /// # Arguments
    ///
    /// * `solver` - Solver to use (SVD or Eigen)
    pub fn with_solver(mut self, solver: LdaSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Set shrinkage regularization for the within-class covariance.
    ///
    /// Useful when n_samples is small relative to n_features.
    /// Regularizes S_w as: S_w_reg = (1 - shrinkage) * S_w + shrinkage * trace(S_w)/p * I
    ///
    /// # Arguments
    ///
    /// * `shrinkage` - Shrinkage parameter in [0, 1]
    pub fn with_shrinkage(mut self, shrinkage: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&shrinkage),
            "shrinkage must be in [0, 1]"
        );
        self.shrinkage = Some(shrinkage);
        self
    }

    /// Set class prior probabilities.
    ///
    /// If not set, priors are estimated from the training data.
    ///
    /// # Arguments
    ///
    /// * `priors` - Prior probability for each class (must sum to 1)
    pub fn with_priors(mut self, priors: Vec<f64>) -> Self {
        let sum: f64 = priors.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "priors must sum to 1, got {}",
            sum
        );
        self.priors = Some(priors);
        self
    }

    /// Set tolerance for singular value truncation.
    ///
    /// # Arguments
    ///
    /// * `tol` - Tolerance threshold
    pub fn with_tol(mut self, tol: f64) -> Self {
        assert!(tol > 0.0, "tol must be positive");
        self.tol = tol;
        self
    }

    /// Fit the LDA model to training data with labels.
    ///
    /// # Arguments
    ///
    /// * `x` - Training data of shape (n_samples, n_features)
    /// * `y` - Target labels of shape (n_samples,)
    ///
    /// # Returns
    ///
    /// * `Ok(())` if fitting succeeds
    /// * `Err(FerroError)` if fitting fails
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        crate::validation::validate_unsupervised_input(x)?;

        let (n_samples, n_features) = x.dim();

        if y.len() != n_samples {
            return Err(FerroError::shape_mismatch(
                format!("({},)", n_samples),
                format!("({},)", y.len()),
            ));
        }

        // Find unique classes
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::invalid_input("LDA requires at least 2 classes"));
        }

        // Determine number of components
        let max_components = (n_classes - 1).min(n_features);
        let n_components = self
            .n_components
            .unwrap_or(max_components)
            .min(max_components);

        if n_components == 0 {
            return Err(FerroError::invalid_input("n_components must be at least 1"));
        }

        // Map classes to indices
        let class_to_idx: HashMap<i64, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c.to_bits() as i64, i))
            .collect();

        // Compute class counts and priors
        let mut class_counts = vec![0usize; n_classes];
        for &yi in y.iter() {
            let idx = class_to_idx[&(yi.to_bits() as i64)];
            class_counts[idx] += 1;
        }

        let priors = if let Some(ref p) = self.priors {
            if p.len() != n_classes {
                return Err(FerroError::invalid_input(format!(
                    "priors length ({}) must match number of classes ({})",
                    p.len(),
                    n_classes
                )));
            }
            Array1::from_vec(p.clone())
        } else {
            Array1::from_iter(class_counts.iter().map(|&c| c as f64 / n_samples as f64))
        };

        // Compute overall mean
        let xbar = x
            .mean_axis(Axis(0))
            .ok_or_else(|| FerroError::numerical("Failed to compute overall mean"))?;

        // Compute class means
        let mut means = Array2::zeros((n_classes, n_features));
        for (k, &class) in classes.iter().enumerate() {
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &yi)| if yi == class { Some(i) } else { None })
                .collect();

            let mut class_data = Vec::with_capacity(mask.len() * n_features);
            for &i in &mask {
                for j in 0..n_features {
                    class_data.push(x[[i, j]]);
                }
            }
            let class_matrix = Array2::from_shape_vec((mask.len(), n_features), class_data)
                .map_err(|e| {
                    FerroError::numerical(format!("Failed to create class matrix: {}", e))
                })?;

            let class_mean = class_matrix.mean_axis(Axis(0)).ok_or_else(|| {
                FerroError::numerical(format!("Failed to compute mean for class {}", class))
            })?;
            means.row_mut(k).assign(&class_mean);
        }

        // Fit using selected solver
        match self.solver {
            LdaSolver::Svd => self.fit_svd(
                x,
                y,
                &classes,
                &class_to_idx,
                &means,
                &xbar,
                &priors,
                n_components,
            )?,
            LdaSolver::Eigen => self.fit_eigen(
                x,
                y,
                &classes,
                &class_to_idx,
                &means,
                &xbar,
                &priors,
                n_components,
            )?,
        }

        // Store fitted state
        self.classes = Some(classes);
        self.means = Some(means);
        self.xbar = Some(xbar);
        self.priors_fitted = Some(priors);
        self.n_features_in = Some(n_features);
        self.n_components_fitted = Some(n_components);

        // Compute classification coefficients
        self.compute_coef()?;

        Ok(())
    }

    /// Fit using SVD solver.
    #[allow(clippy::too_many_arguments)]
    fn fit_svd(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: &[f64],
        class_to_idx: &HashMap<i64, usize>,
        means: &Array2<f64>,
        xbar: &Array1<f64>,
        priors: &Array1<f64>,
        n_components: usize,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let n_classes = classes.len();

        // Center the data by class means
        let mut x_centered = x.clone();
        for i in 0..n_samples {
            let yi = y[i];
            let class_idx = class_to_idx[&(yi.to_bits() as i64)];
            let class_mean = means.row(class_idx);
            for j in 0..n_features {
                x_centered[[i, j]] -= class_mean[j];
            }
        }

        // Apply shrinkage if specified
        let std_factor = if self.shrinkage.is_some() {
            // With shrinkage, we need to regularize
            // For SVD solver, we apply it differently
            1.0
        } else {
            1.0
        };

        // Compute SVD of centered data for within-class scatter
        // S_w = X_c^T X_c where X_c is class-centered data
        let x_scaled = &x_centered / (n_samples as f64 - n_classes as f64).sqrt().max(1.0);

        let x_scaled_factored = x_scaled.mapv(|v| v * std_factor);

        let (_u_sw, s_sw, vt_sw) = crate::linalg::thin_svd(&x_scaled_factored)?;

        // Find rank of S_w (non-zero singular values)
        let rank = s_sw.iter().filter(|&&s| s > self.tol).count().max(1);

        // Create scaled class means for between-class scatter
        // S_b = Σ n_c (μ_c - μ)(μ_c - μ)^T
        let mut class_means_centered = Array2::zeros((n_classes, n_features));
        for (k, class_mean) in means.rows().into_iter().enumerate() {
            let weight = (priors[k] * n_samples as f64).sqrt();
            for j in 0..n_features {
                class_means_centered[[k, j]] = (class_mean[j] - xbar[j]) * weight;
            }
        }

        // Whiten the class means using S_w^{-1/2}
        // First compute S_w^{-1/2} = V @ S^{-1} @ V^T  (for the covariance, not data)
        // Actually for LDA with SVD: project means onto S_w^{-1/2} space

        // Simpler approach: compute between-class scatter directly and solve
        // using the generalized eigenvalue approach with regularization

        // Between-class scatter (stored for potential future use)
        let _sb_mat =
            nalgebra::DMatrix::from_fn(n_classes, n_features, |i, j| class_means_centered[[i, j]]);

        // Compute SVD of S_b^{1/2} projected onto S_w^{-1/2} space
        // For numerical stability, use a different approach:
        // Project class means onto the principal subspace of S_w

        // Invert singular values (with regularization)
        let mut s_inv = vec![0.0; rank];
        for (i, s_inv_i) in s_inv.iter_mut().enumerate().take(rank) {
            if s_sw[i] > self.tol {
                *s_inv_i = 1.0 / s_sw[i];
            }
        }

        // Apply shrinkage to inverse singular values
        if let Some(shrink) = self.shrinkage {
            let trace: f64 = s_sw.iter().map(|s| s * s).sum();
            let trace_avg = trace / n_features as f64;
            let shrink_factor = (1.0 - shrink).sqrt();
            let regularizer = (shrink * trace_avg).sqrt();

            for (i, s_inv_i) in s_inv.iter_mut().enumerate().take(rank) {
                if s_sw[i] > self.tol {
                    let reg_s = (s_sw[i] * shrink_factor * s_sw[i])
                        .mul_add(shrink_factor, regularizer * regularizer)
                        .sqrt();
                    *s_inv_i = 1.0 / reg_s;
                }
            }
        }

        // Project class means onto whitened space
        // means_w = means_centered @ V @ diag(s_inv)
        let v_cols = vt_sw.t();
        let mut means_whitened = Array2::zeros((n_classes, rank));

        for k in 0..n_classes {
            for j in 0..rank {
                let mut sum = 0.0;
                for f in 0..n_features {
                    sum += class_means_centered[[k, f]] * v_cols[[f, j]] * s_inv[j];
                }
                means_whitened[[k, j]] = sum;
            }
        }

        // SVD of whitened class means gives the discriminant directions in whitened space
        let (_u_mw, s_mw, vt_mw) = crate::linalg::thin_svd(&means_whitened)?;

        // Number of discriminant directions
        // Use a relative tolerance based on the largest singular value
        let max_sv = s_mw.iter().copied().fold(0.0_f64, f64::max);
        let rel_tol = self.tol * max_sv.max(1.0);
        let n_discriminants = s_mw
            .iter()
            .filter(|&&s| s > rel_tol)
            .count()
            .min(n_components);

        // Explained variance is proportional to squared singular values
        let total_var: f64 = s_mw.iter().map(|s| s * s).sum();
        let explained_variance_ratio = if total_var > 0.0 {
            Array1::from_iter(s_mw.iter().take(n_discriminants).map(|s| s * s / total_var))
        } else {
            Array1::zeros(n_discriminants)
        };

        let eigenvalues = Array1::from_iter(s_mw.iter().take(n_discriminants).map(|s| s * s));

        // Transform back to original space:
        // scalings = V @ diag(s_inv) @ V_mw^T
        let mut scalings = Array2::zeros((n_features, n_discriminants));
        for f in 0..n_features {
            for c in 0..n_discriminants {
                let mut sum = 0.0;
                for j in 0..rank {
                    sum += v_cols[[f, j]] * s_inv[j] * vt_mw[[c, j]];
                }
                scalings[[f, c]] = sum;
            }
        }

        // Normalize scalings
        for c in 0..n_discriminants {
            let norm: f64 = scalings
                .column(c)
                .iter()
                .map(|&v| v * v)
                .sum::<f64>()
                .sqrt();
            if norm > self.tol {
                scalings.column_mut(c).iter_mut().for_each(|v| *v /= norm);
            }
        }

        self.scalings = Some(scalings);
        self.explained_variance_ratio = Some(explained_variance_ratio);
        self.eigenvalues = Some(eigenvalues);

        Ok(())
    }

    /// Fit using eigenvalue decomposition solver.
    #[allow(clippy::too_many_arguments)]
    fn fit_eigen(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: &[f64],
        class_to_idx: &HashMap<i64, usize>,
        means: &Array2<f64>,
        xbar: &Array1<f64>,
        priors: &Array1<f64>,
        n_components: usize,
    ) -> Result<()> {
        let (n_samples, n_features) = x.dim();
        let n_classes = classes.len();

        // Compute within-class scatter matrix S_w
        let mut sw = Array2::<f64>::zeros((n_features, n_features));
        for i in 0..n_samples {
            let yi = y[i];
            let class_idx = class_to_idx[&(yi.to_bits() as i64)];
            let class_mean = means.row(class_idx);

            for j in 0..n_features {
                let diff_j = x[[i, j]] - class_mean[j];
                for k in j..n_features {
                    let diff_k = x[[i, k]] - class_mean[k];
                    let val = diff_j * diff_k;
                    sw[[j, k]] += val;
                    if j != k {
                        sw[[k, j]] += val;
                    }
                }
            }
        }

        // Apply shrinkage regularization
        if let Some(shrink) = self.shrinkage {
            let trace: f64 = (0..n_features).map(|i| sw[[i, i]]).sum();
            let mu = trace / n_features as f64;

            for i in 0..n_features {
                for j in 0..n_features {
                    if i == j {
                        sw[[i, j]] = (1.0 - shrink).mul_add(sw[[i, j]], shrink * mu);
                    } else {
                        sw[[i, j]] = (1.0 - shrink) * sw[[i, j]];
                    }
                }
            }
        }

        // Compute between-class scatter matrix S_b
        let mut sb = Array2::<f64>::zeros((n_features, n_features));
        for (k, class_mean) in means.rows().into_iter().enumerate() {
            let weight = priors[k] * n_samples as f64;
            for j in 0..n_features {
                let diff_j = class_mean[j] - xbar[j];
                for l in j..n_features {
                    let diff_l = class_mean[l] - xbar[l];
                    let val = weight * diff_j * diff_l;
                    sb[[j, l]] += val;
                    if j != l {
                        sb[[l, j]] += val;
                    }
                }
            }
        }

        // Solve generalized eigenvalue problem: S_b w = λ S_w w
        // Use symmetric transformation so symmetric_eigen() is valid.
        let sw_mat: nalgebra::DMatrix<f64> =
            nalgebra::DMatrix::from_fn(n_features, n_features, |i, j| sw[[i, j]]);
        let sb_mat = nalgebra::DMatrix::<f64>::from_fn(n_features, n_features, |i, j| sb[[i, j]]);

        // back_transform: eigenvectors of the original problem = back_transform * v
        let (eigenvalues, eigenvectors) = if let Some(chol) = sw_mat.clone().cholesky() {
            // Cholesky: S_w = L L^T → M = L^{-1} S_b L^{-T} (symmetric)
            let l_inv = chol
                .l()
                .try_inverse()
                .ok_or_else(|| FerroError::numerical("Cholesky L inverse failed"))?;
            let l_inv_t = l_inv.transpose();
            let m = &l_inv * &sb_mat * &l_inv_t;
            let eigen = m.symmetric_eigen();
            // Back-transform: w = L^{-T} v
            let raw_vecs = &l_inv_t * &eigen.eigenvectors;
            (eigen.eigenvalues, raw_vecs)
        } else {
            // SVD fallback: S_w^{-1/2} via SVD
            let svd = sw_mat.svd(true, true);
            let u = svd
                .u
                .ok_or_else(|| FerroError::numerical("SVD failed for S_w"))?;
            let s = svd.singular_values;
            let vt = svd
                .v_t
                .ok_or_else(|| FerroError::numerical("SVD failed for S_w"))?;

            // S_w^{-1/2} = V diag(s^{-1/2}) U^T
            let mut s_inv_sqrt = nalgebra::DMatrix::<f64>::zeros(n_features, n_features);
            for i in 0..n_features {
                if s[i] > self.tol {
                    s_inv_sqrt[(i, i)] = 1.0 / s[i].sqrt();
                }
            }
            let sw_inv_sqrt = vt.transpose() * &s_inv_sqrt * u.transpose();

            // M = S_w^{-1/2} S_b S_w^{-1/2} (symmetric)
            let m = &sw_inv_sqrt * &sb_mat * &sw_inv_sqrt;
            let eigen = m.symmetric_eigen();
            // Back-transform: w = S_w^{-1/2} v
            let raw_vecs = &sw_inv_sqrt * &eigen.eigenvectors;
            (eigen.eigenvalues, raw_vecs)
        };

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top n_components eigenvectors
        let n_comp = n_components.min(n_classes - 1);
        let mut scalings = Array2::<f64>::zeros((n_features, n_comp));
        let mut eig_vals = Array1::<f64>::zeros(n_comp);

        for (new_idx, &old_idx) in indices.iter().take(n_comp).enumerate() {
            for j in 0..n_features {
                scalings[[j, new_idx]] = eigenvectors[(j, old_idx)];
            }
            eig_vals[new_idx] = eigenvalues[old_idx].max(0.0);
        }

        // Compute explained variance ratio
        let total_eig: f64 = eigenvalues.iter().filter(|&&e| e > 0.0).sum();
        let explained_variance_ratio = if total_eig > 0.0 {
            eig_vals.mapv(|e| e / total_eig)
        } else {
            Array1::zeros(n_comp)
        };

        self.scalings = Some(scalings);
        self.eigenvalues = Some(eig_vals);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
    }

    /// Compute classification coefficients.
    fn compute_coef(&mut self) -> Result<()> {
        let scalings = self
            .scalings
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("compute_coef"))?;
        let means = self
            .means
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("compute_coef"))?;
        let priors = self
            .priors_fitted
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("compute_coef"))?;

        let n_classes = means.nrows();
        let n_features = means.ncols();

        // Project class means onto discriminant space
        let means_proj = means.dot(scalings);

        // For classification, we can use the projected means
        // coef[c] = scalings @ means_proj[c].T
        // But we store the projection of means for decision function

        // Simple coefficients: project to discriminant space then use class means
        let mut coef = Array2::zeros((n_classes, n_features));
        let mut intercept = Array1::zeros(n_classes);

        for c in 0..n_classes {
            // coef for class c: direction towards class mean in discriminant space
            // transformed back to original space
            let class_mean_proj = means_proj.row(c);

            // Compute coefficient as: scalings @ class_mean_proj
            for f in 0..n_features {
                let mut sum = 0.0;
                for d in 0..scalings.ncols() {
                    sum += scalings[[f, d]] * class_mean_proj[d];
                }
                coef[[c, f]] = sum;
            }

            // Intercept includes prior probability
            let mean_proj_sq: f64 = class_mean_proj.iter().map(|v| v * v).sum();
            intercept[c] = 0.5f64.mul_add(-mean_proj_sq, priors[c].ln());
        }

        self.coef = Some(coef);
        self.intercept = Some(intercept);

        Ok(())
    }

    /// Transform data by projecting onto discriminant directions.
    ///
    /// # Arguments
    ///
    /// * `x` - Data to transform of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// * Transformed data of shape (n_samples, n_components)
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        crate::validation::validate_transform_input(x, self.n_features_in.unwrap())?;

        let scalings = self.scalings.as_ref().unwrap();
        let xbar = self.xbar.as_ref().unwrap();

        // Center and project: (X - mean) @ scalings
        let x_centered = x - xbar;
        let x_transformed = x_centered.dot(scalings);

        Ok(x_transformed)
    }

    /// Fit and transform in one step.
    ///
    /// # Arguments
    ///
    /// * `x` - Data to fit and transform
    /// * `y` - Target labels
    ///
    /// # Returns
    ///
    /// * Transformed data
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Predict class labels for samples.
    ///
    /// Uses the decision function to find the class with highest score.
    ///
    /// # Arguments
    ///
    /// * `x` - Data to classify
    ///
    /// # Returns
    ///
    /// * Predicted class labels
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(self.is_fitted(), "predict")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let scores = self.decision_function(x)?;
        let classes = self.classes.as_ref().unwrap();

        let mut predictions = Array1::zeros(x.nrows());
        for i in 0..x.nrows() {
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for (c, &_class) in classes.iter().enumerate() {
                if scores[[i, c]] > best_score {
                    best_score = scores[[i, c]];
                    best_class = c;
                }
            }
            predictions[i] = classes[best_class];
        }

        Ok(predictions)
    }

    /// Compute decision function values.
    ///
    /// Returns the log-posterior scores for each class.
    ///
    /// # Arguments
    ///
    /// * `x` - Data to score
    ///
    /// # Returns
    ///
    /// * Decision function values of shape (n_samples, n_classes)
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "decision_function")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let scalings = self.scalings.as_ref().unwrap();
        let means = self.means.as_ref().unwrap();
        let xbar = self.xbar.as_ref().unwrap();
        let priors = self.priors_fitted.as_ref().unwrap();

        let n_samples = x.nrows();
        let n_classes = means.nrows();

        // Project data
        let x_centered = x - xbar;
        let x_proj = x_centered.dot(scalings);

        // Project class means
        let means_centered = means - xbar;
        let means_proj = means_centered.dot(scalings);

        // Compute decision function: log P(class) - 0.5 * ||x_proj - mean_proj||^2
        let mut scores = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            for c in 0..n_classes {
                let mut dist_sq = 0.0;
                for d in 0..x_proj.ncols() {
                    let diff = x_proj[[i, d]] - means_proj[[c, d]];
                    dist_sq += diff * diff;
                }
                scores[[i, c]] = 0.5f64.mul_add(-dist_sq, priors[c].ln());
            }
        }

        Ok(scores)
    }

    /// Predict class probabilities.
    ///
    /// Uses softmax of decision function values.
    ///
    /// # Arguments
    ///
    /// * `x` - Data to classify
    ///
    /// # Returns
    ///
    /// * Class probabilities of shape (n_samples, n_classes)
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let scores = self.decision_function(x)?;

        // Softmax
        let n_samples = scores.nrows();
        let n_classes = scores.ncols();
        let mut proba = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let max_score = scores
                .row(i)
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0;

            for c in 0..n_classes {
                let exp_val = (scores[[i, c]] - max_score).exp();
                proba[[i, c]] = exp_val;
                sum_exp += exp_val;
            }

            for c in 0..n_classes {
                proba[[i, c]] /= sum_exp;
            }
        }

        Ok(proba)
    }

    // Accessor methods

    /// Get the projection matrix (scalings).
    ///
    /// Returns matrix of shape (n_features, n_components) where each column
    /// is a discriminant direction.
    pub fn scalings(&self) -> Option<&Array2<f64>> {
        self.scalings.as_ref()
    }

    /// Get the explained variance ratio for each component.
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    /// Get the eigenvalues of the discriminant functions.
    pub fn eigenvalues(&self) -> Option<&Array1<f64>> {
        self.eigenvalues.as_ref()
    }

    /// Get the class means.
    pub fn means(&self) -> Option<&Array2<f64>> {
        self.means.as_ref()
    }

    /// Get the overall mean.
    pub fn xbar(&self) -> Option<&Array1<f64>> {
        self.xbar.as_ref()
    }

    /// Get the unique class labels.
    pub fn classes(&self) -> Option<&[f64]> {
        self.classes.as_deref()
    }

    /// Get the class prior probabilities.
    pub fn priors(&self) -> Option<&Array1<f64>> {
        self.priors_fitted.as_ref()
    }

    /// Get the number of components.
    pub fn n_components_(&self) -> Option<usize> {
        self.n_components_fitted
    }

    /// Get the classification coefficients.
    pub fn coef(&self) -> Option<&Array2<f64>> {
        self.coef.as_ref()
    }

    /// Get the classification intercept.
    pub fn intercept(&self) -> Option<&Array1<f64>> {
        self.intercept.as_ref()
    }

    /// Check if the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.scalings.is_some()
            && self.means.is_some()
            && self.xbar.is_some()
            && self.classes.is_some()
    }

    /// Get the number of features expected in input.
    pub fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    /// Get the number of output features.
    pub fn n_features_out(&self) -> Option<usize> {
        self.n_components_fitted
    }

    /// Get output feature names.
    pub fn get_feature_names_out(&self, _input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_components = self.n_components_fitted?;
        Some((0..n_components).map(|i| format!("lda{i}")).collect())
    }
}

impl Default for LDA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;
    use crate::testing::assertions::tolerances;
    use ndarray::{array, s};

    // Use calibrated tolerance from assertions module
    const TOLERANCE: f64 = tolerances::DECOMPOSITION;

    #[test]
    fn test_lda_two_classes() {
        let mut lda = LDA::new();

        // Two well-separated classes
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 7.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        lda.fit(&x, &y).unwrap();

        assert!(lda.is_fitted());
        assert_eq!(lda.n_components_(), Some(1)); // n_classes - 1 = 1
        assert_eq!(lda.classes().unwrap(), &[0.0, 1.0]);
    }

    #[test]
    fn test_lda_transform() {
        let mut lda = LDA::new();

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 7.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let x_lda = lda.fit_transform(&x, &y).unwrap();

        assert_eq!(x_lda.ncols(), 1);
        assert_eq!(x_lda.nrows(), 6);

        // Classes should be well separated in transformed space
        let class0_mean: f64 = x_lda.slice(s![0..3, 0]).mean().unwrap();
        let class1_mean: f64 = x_lda.slice(s![3..6, 0]).mean().unwrap();

        // The means should be different
        assert!((class0_mean - class1_mean).abs() > 0.1);
    }

    #[test]
    fn test_lda_three_classes() {
        let mut lda = LDA::new();

        // Three classes with non-collinear points within each class
        let x = array![
            [1.0, 1.0],
            [1.5, 1.8], // Changed to not be collinear
            [2.0, 1.5], // Changed to not be collinear
            [5.0, 1.0],
            [5.5, 1.8], // Changed to not be collinear
            [6.0, 1.2], // Changed to not be collinear
            [3.0, 5.0],
            [3.5, 5.2], // Changed to not be collinear
            [4.0, 5.8]  // Changed to not be collinear
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        lda.fit(&x, &y).unwrap();

        assert_eq!(lda.n_components_(), Some(2)); // min(n_classes-1=2, n_features=2)

        let x_lda = lda.transform(&x).unwrap();
        assert_eq!(x_lda.ncols(), 2);
    }

    #[test]
    fn test_lda_predict() {
        let mut lda = LDA::new();

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 7.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        lda.fit(&x, &y).unwrap();

        // Predict on training data
        let y_pred = lda.predict(&x).unwrap();

        // Should predict training data correctly (well-separated classes)
        for i in 0..6 {
            assert_eq!(
                y_pred[i], y[i],
                "Mismatch at index {}: predicted {} vs actual {}",
                i, y_pred[i], y[i]
            );
        }

        // Predict on new data
        let x_new = array![[2.0, 2.0], [7.0, 7.0]];
        let y_new = lda.predict(&x_new).unwrap();

        assert_eq!(y_new[0], 0.0);
        assert_eq!(y_new[1], 1.0);
    }

    #[test]
    fn test_lda_predict_proba() {
        let mut lda = LDA::new();

        let x = array![[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        lda.fit(&x, &y).unwrap();

        let proba = lda.predict_proba(&x).unwrap();

        // Probabilities should sum to 1
        for i in 0..proba.nrows() {
            let sum: f64 = proba.row(i).sum();
            assert_approx_eq!(
                sum,
                1.0,
                tolerances::PROBABILITY,
                "row {} probabilities should sum to 1",
                i
            );
        }

        // Class 0 samples should have higher prob for class 0
        assert!(proba[[0, 0]] > proba[[0, 1]]);
        assert!(proba[[1, 0]] > proba[[1, 1]]);

        // Class 1 samples should have higher prob for class 1
        assert!(proba[[2, 1]] > proba[[2, 0]]);
        assert!(proba[[3, 1]] > proba[[3, 0]]);
    }

    #[test]
    fn test_lda_explained_variance() {
        let mut lda = LDA::new();

        let x = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [5.0, 1.0],
            [5.5, 1.5],
            [3.0, 5.0],
            [3.5, 5.5]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        lda.fit(&x, &y).unwrap();

        let evr = lda.explained_variance_ratio().unwrap();

        // Should have positive explained variance ratio
        assert!(evr[0] > 0.0);

        // First component should explain more than second
        if evr.len() > 1 {
            assert!(evr[0] >= evr[1] - TOLERANCE);
        }
    }

    #[test]
    fn test_lda_shrinkage() {
        let mut lda = LDA::new().with_shrinkage(0.5);

        // Small sample size case
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0];

        // Should not fail with shrinkage
        lda.fit(&x, &y).unwrap();
        assert!(lda.is_fitted());
    }

    #[test]
    fn test_lda_priors() {
        let mut lda = LDA::new().with_priors(vec![0.7, 0.3]);

        let x = array![[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        lda.fit(&x, &y).unwrap();

        let priors = lda.priors().unwrap();
        assert!((priors[0] - 0.7).abs() < TOLERANCE);
        assert!((priors[1] - 0.3).abs() < TOLERANCE);
    }

    #[test]
    fn test_lda_eigen_solver() {
        let mut lda = LDA::new().with_solver(LdaSolver::Eigen);

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 7.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        lda.fit(&x, &y).unwrap();
        assert!(lda.is_fitted());

        let x_lda = lda.transform(&x).unwrap();
        assert_eq!(x_lda.ncols(), 1);
    }

    #[test]
    fn test_lda_n_components() {
        let mut lda = LDA::new().with_n_components(1);

        // 3 classes -> could have 2 components, but we limit to 1
        let x = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [5.0, 1.0],
            [5.5, 1.5],
            [3.0, 5.0],
            [3.5, 5.5]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        lda.fit(&x, &y).unwrap();
        assert_eq!(lda.n_components_(), Some(1));

        let x_lda = lda.transform(&x).unwrap();
        assert_eq!(x_lda.ncols(), 1);
    }

    #[test]
    fn test_lda_feature_names() {
        let mut lda = LDA::new();

        let x = array![[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        lda.fit(&x, &y).unwrap();

        let names = lda.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["lda0"]);
    }

    #[test]
    fn test_lda_not_fitted() {
        let lda = LDA::new();
        let x = array![[1.0, 2.0]];

        assert!(!lda.is_fitted());
        assert!(lda.transform(&x).is_err());
        assert!(lda.predict(&x).is_err());
    }

    #[test]
    fn test_lda_shape_mismatch() {
        let mut lda = LDA::new();

        let x = array![[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0]];
        let y_wrong = array![0.0, 0.0, 1.0]; // Wrong length

        assert!(lda.fit(&x, &y_wrong).is_err());
    }

    #[test]
    fn test_lda_single_class() {
        let mut lda = LDA::new();

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![0.0, 0.0, 0.0]; // Only one class

        assert!(lda.fit(&x, &y).is_err());
    }

    #[test]
    fn test_lda_class_means() {
        let mut lda = LDA::new();

        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [10.0, 20.0],
            [30.0, 40.0],
            [50.0, 60.0]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        lda.fit(&x, &y).unwrap();

        let means = lda.means().unwrap();

        // Class 0 mean: (1+3+5)/3 = 3, (2+4+6)/3 = 4
        assert!((means[[0, 0]] - 3.0).abs() < TOLERANCE);
        assert!((means[[0, 1]] - 4.0).abs() < TOLERANCE);

        // Class 1 mean: (10+30+50)/3 = 30, (20+40+60)/3 = 40
        assert!((means[[1, 0]] - 30.0).abs() < TOLERANCE);
        assert!((means[[1, 1]] - 40.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_lda_decision_function() {
        let mut lda = LDA::new();

        let x = array![[1.0, 2.0], [2.0, 3.0], [6.0, 7.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];

        lda.fit(&x, &y).unwrap();

        let scores = lda.decision_function(&x).unwrap();

        assert_eq!(scores.nrows(), 4);
        assert_eq!(scores.ncols(), 2);

        // Class 0 samples should have higher score for class 0
        assert!(scores[[0, 0]] > scores[[0, 1]]);
        assert!(scores[[1, 0]] > scores[[1, 1]]);

        // Class 1 samples should have higher score for class 1
        assert!(scores[[2, 1]] > scores[[2, 0]]);
        assert!(scores[[3, 1]] > scores[[3, 0]]);
    }

    #[test]
    fn test_lda_eigen_svd_agreement() {
        // Regression: Eigen solver called symmetric_eigen on non-symmetric matrix.
        // Both solvers should produce equivalent predictions on 3-class data.
        let x = array![
            [1.0, 2.0],
            [1.5, 1.8],
            [1.2, 2.2],
            [5.0, 6.0],
            [5.5, 5.8],
            [5.2, 6.2],
            [9.0, 2.0],
            [9.5, 1.8],
            [9.2, 2.2]
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let mut lda_eigen = LDA::new().with_solver(LdaSolver::Eigen);
        let mut lda_svd = LDA::new().with_solver(LdaSolver::Svd);

        lda_eigen.fit(&x, &y).unwrap();
        lda_svd.fit(&x, &y).unwrap();

        // Both should predict correctly on training data
        let pred_eigen = lda_eigen.predict(&x).unwrap();
        let pred_svd = lda_svd.predict(&x).unwrap();

        for i in 0..9 {
            assert_eq!(pred_eigen[i], y[i], "Eigen solver mispredicted sample {i}");
            assert_eq!(pred_svd[i], y[i], "SVD solver mispredicted sample {i}");
        }
    }
}
