//! Gaussian Process models for regression and classification.
//!
//! This module provides:
//! - `GaussianProcessRegressor` — exact GP regression with uncertainty estimates
//! - `GaussianProcessClassifier` — binary GP classification via Laplace approximation
//! - Kernel functions: `RBF`, `Matern`, `ConstantKernel`, `WhiteKernel`, `SumKernel`, `ProductKernel`

use crate::hpo::SearchSpace;
use crate::models::{sigmoid, Model};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};

// =============================================================================
// Kernel trait
// =============================================================================

/// Kernel function for Gaussian Process models.
pub trait Kernel: Send + Sync + std::fmt::Debug {
    /// Compute the kernel matrix K(X1, X2).
    fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64>;

    /// Compute diagonal of K(X, X) efficiently.
    fn diagonal(&self, x: &Array2<f64>) -> Array1<f64>;

    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Kernel>;
}

impl Clone for Box<dyn Kernel> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// =============================================================================
// RBF Kernel
// =============================================================================

/// Radial Basis Function (squared exponential) kernel.
///
/// K(x, x') = exp(-||x - x'||² / (2 * length_scale²))
#[derive(Debug, Clone)]
pub struct RBF {
    /// Length scale parameter.
    pub length_scale: f64,
}

impl RBF {
    /// Create a new RBF kernel with the given length scale.
    #[must_use]
    pub fn new(length_scale: f64) -> Self {
        Self { length_scale }
    }
}

impl Kernel for RBF {
    fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));
        let ls2 = 2.0 * self.length_scale * self.length_scale;

        for i in 0..n1 {
            for j in 0..n2 {
                let row_i = x1.row(i);
                let row_j = x2.row(j);
                let mut sq_dist = 0.0;
                for d in 0..row_i.len() {
                    let diff = row_i[d] - row_j[d];
                    sq_dist += diff * diff;
                }
                k[[i, j]] = (-sq_dist / ls2).exp();
            }
        }
        k
    }

    fn diagonal(&self, x: &Array2<f64>) -> Array1<f64> {
        // K(x, x) = exp(0) = 1 for all x
        Array1::ones(x.nrows())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// =============================================================================
// Matern Kernel
// =============================================================================

/// Matern kernel with support for nu = 0.5, 1.5, 2.5.
///
/// - nu=0.5: K = exp(-d/l) (exponential)
/// - nu=1.5: K = (1 + sqrt(3)*d/l) * exp(-sqrt(3)*d/l)
/// - nu=2.5: K = (1 + sqrt(5)*d/l + 5*d^2/(3*l^2)) * exp(-sqrt(5)*d/l)
#[derive(Debug, Clone)]
pub struct Matern {
    /// Length scale parameter.
    pub length_scale: f64,
    /// Smoothness parameter (must be 0.5, 1.5, or 2.5).
    pub nu: f64,
}

impl Matern {
    /// Create a new Matern kernel.
    ///
    /// # Panics
    /// Panics if nu is not one of 0.5, 1.5, or 2.5.
    #[must_use]
    pub fn new(length_scale: f64, nu: f64) -> Self {
        assert!(
            (nu - 0.5).abs() < 1e-10 || (nu - 1.5).abs() < 1e-10 || (nu - 2.5).abs() < 1e-10,
            "Matern kernel only supports nu = 0.5, 1.5, or 2.5, got {}",
            nu
        );
        Self { length_scale, nu }
    }

    fn euclidean_distance(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        let mut sq = 0.0;
        for d in 0..a.len() {
            let diff = a[d] - b[d];
            sq += diff * diff;
        }
        sq.sqrt()
    }
}

impl Kernel for Matern {
    fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));
        let l = self.length_scale;

        for i in 0..n1 {
            for j in 0..n2 {
                let d = Self::euclidean_distance(&x1.row(i), &x2.row(j));
                let val = if (self.nu - 0.5).abs() < 1e-10 {
                    (-d / l).exp()
                } else if (self.nu - 1.5).abs() < 1e-10 {
                    let s3 = 3.0_f64.sqrt();
                    let r = s3 * d / l;
                    (1.0 + r) * (-r).exp()
                } else {
                    // nu = 2.5
                    let s5 = 5.0_f64.sqrt();
                    let r = s5 * d / l;
                    (1.0 + r + r * r / 3.0) * (-r).exp()
                };
                k[[i, j]] = val;
            }
        }
        k
    }

    fn diagonal(&self, x: &Array2<f64>) -> Array1<f64> {
        // K(x, x) = 1 for all Matern kernels (distance=0)
        Array1::ones(x.nrows())
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// =============================================================================
// ConstantKernel
// =============================================================================

/// Constant kernel: K(x, x') = constant for all x, x'.
#[derive(Debug, Clone)]
pub struct ConstantKernel {
    /// The constant value.
    pub constant: f64,
}

impl ConstantKernel {
    /// Create a new constant kernel.
    #[must_use]
    pub fn new(constant: f64) -> Self {
        Self { constant }
    }
}

impl Kernel for ConstantKernel {
    fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        Array2::from_elem((x1.nrows(), x2.nrows()), self.constant)
    }

    fn diagonal(&self, x: &Array2<f64>) -> Array1<f64> {
        Array1::from_elem(x.nrows(), self.constant)
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// =============================================================================
// WhiteKernel
// =============================================================================

/// White noise kernel: K(x, x') = noise_level * delta(x, x').
///
/// Only the diagonal is non-zero (when computing K(X, X)).
#[derive(Debug, Clone)]
pub struct WhiteKernel {
    /// Noise level.
    pub noise_level: f64,
}

impl WhiteKernel {
    /// Create a new white noise kernel.
    #[must_use]
    pub fn new(noise_level: f64) -> Self {
        Self { noise_level }
    }
}

impl Kernel for WhiteKernel {
    fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));
        // White kernel is noise_level * I when X1 == X2 (same dataset)
        // We check by pointer equality or row equality
        let min_n = n1.min(n2);
        for i in 0..min_n {
            if x1.row(i) == x2.row(i) {
                k[[i, i]] = self.noise_level;
            }
        }
        k
    }

    fn diagonal(&self, x: &Array2<f64>) -> Array1<f64> {
        Array1::from_elem(x.nrows(), self.noise_level)
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// =============================================================================
// SumKernel
// =============================================================================

/// Sum of two kernels: K = K1 + K2.
#[derive(Debug, Clone)]
pub struct SumKernel {
    /// First kernel.
    pub k1: Box<dyn Kernel>,
    /// Second kernel.
    pub k2: Box<dyn Kernel>,
}

impl SumKernel {
    /// Create a sum kernel.
    pub fn new(k1: Box<dyn Kernel>, k2: Box<dyn Kernel>) -> Self {
        Self { k1, k2 }
    }
}

impl Kernel for SumKernel {
    fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let m1 = self.k1.compute(x1, x2);
        let m2 = self.k2.compute(x1, x2);
        m1 + m2
    }

    fn diagonal(&self, x: &Array2<f64>) -> Array1<f64> {
        let d1 = self.k1.diagonal(x);
        let d2 = self.k2.diagonal(x);
        d1 + d2
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// =============================================================================
// ProductKernel
// =============================================================================

/// Product of two kernels: K = K1 * K2.
#[derive(Debug, Clone)]
pub struct ProductKernel {
    /// First kernel.
    pub k1: Box<dyn Kernel>,
    /// Second kernel.
    pub k2: Box<dyn Kernel>,
}

impl ProductKernel {
    /// Create a product kernel.
    pub fn new(k1: Box<dyn Kernel>, k2: Box<dyn Kernel>) -> Self {
        Self { k1, k2 }
    }
}

impl Kernel for ProductKernel {
    fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let m1 = self.k1.compute(x1, x2);
        let m2 = self.k2.compute(x1, x2);
        m1 * m2
    }

    fn diagonal(&self, x: &Array2<f64>) -> Array1<f64> {
        let d1 = self.k1.diagonal(x);
        let d2 = self.k2.diagonal(x);
        d1 * d2
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

// =============================================================================
// Cholesky and triangular solvers
// =============================================================================

/// Numerically stable GP posterior variance: `prior - reduction`.
/// Catastrophic cancellation can occur when `prior ≈ reduction` (both large).
/// We floor at `prior * 1e-10` rather than zero to preserve relative precision.
#[inline]
fn stable_posterior_variance(prior: f64, reduction: f64) -> f64 {
    let floor = prior.abs() * 1e-10;
    (prior - reduction).max(floor)
}

/// Cholesky decomposition: A = L @ L^T.
/// Returns the lower-triangular factor L.
fn cholesky(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));

    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[[j, k]] * l[[j, k]];
        }
        let diag = a[[j, j]] - sum;
        if diag <= 0.0 {
            return Err(FerroError::NumericalError(format!(
                "Cholesky decomposition failed: matrix is not positive definite (diagonal element {} = {})",
                j, diag
            )));
        }
        l[[j, j]] = diag.sqrt();

        for i in (j + 1)..n {
            let mut sum2 = 0.0;
            for k in 0..j {
                sum2 += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - sum2) / l[[j, j]];
        }
    }

    Ok(l)
}

/// Solve L @ x = b where L is lower-triangular (forward substitution).
fn solve_lower_triangular(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * x[j];
        }
        x[i] = (b[i] - sum) / l[[i, i]];
    }
    x
}

/// Solve L^T @ x = b where L is lower-triangular (backward substitution).
fn solve_upper_triangular_transpose(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j]; // L^T[i,j] = L[j,i]
        }
        x[i] = (b[i] - sum) / l[[i, i]];
    }
    x
}

/// Solve L @ X = B where B is a matrix (column-wise forward substitution).
fn solve_lower_triangular_mat(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = b.nrows();
    let m = b.ncols();
    let mut x = Array2::zeros((n, m));
    for col in 0..m {
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * x[[j, col]];
            }
            x[[i, col]] = (b[[i, col]] - sum) / l[[i, i]];
        }
    }
    x
}

// =============================================================================
// GaussianProcessRegressor
// =============================================================================

/// Gaussian Process Regressor.
///
/// Implements exact Gaussian process regression with Cholesky-based inference.
/// Supports uncertainty estimation via `predict_with_std`.
///
/// # Example
///
/// ```
/// use ferroml_core::models::gaussian_process::{GaussianProcessRegressor, RBF};
/// use ferroml_core::models::Model;
/// use ndarray::{Array1, Array2};
///
/// let kernel = RBF::new(1.0);
/// let mut gpr = GaussianProcessRegressor::new(Box::new(kernel));
/// let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = Array1::from_vec(vec![0.0, 0.84, 0.91, 0.14, -0.76]);
/// gpr.fit(&x, &y).unwrap();
/// let preds = gpr.predict(&x).unwrap();
/// assert_eq!(preds.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct GaussianProcessRegressor {
    kernel: Box<dyn Kernel>,
    alpha: f64,
    normalize_y: bool,
    // Fitted state
    x_train_: Option<Array2<f64>>,
    y_train_mean_: Option<f64>,
    y_train_std_: Option<f64>,
    alpha_: Option<Array1<f64>>,
    l_: Option<Array2<f64>>,
    log_marginal_likelihood_: Option<f64>,
}

impl GaussianProcessRegressor {
    /// Create a new GP regressor with the given kernel.
    ///
    /// Defaults: alpha=1e-10, normalize_y=false.
    #[must_use]
    pub fn new(kernel: Box<dyn Kernel>) -> Self {
        Self {
            kernel,
            alpha: 1e-10,
            normalize_y: false,
            x_train_: None,
            y_train_mean_: None,
            y_train_std_: None,
            alpha_: None,
            l_: None,
            log_marginal_likelihood_: None,
        }
    }

    /// Set the noise/regularization parameter added to the kernel diagonal.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to normalize y before fitting.
    #[must_use]
    pub fn with_normalize_y(mut self, normalize_y: bool) -> Self {
        self.normalize_y = normalize_y;
        self
    }

    /// Predict with uncertainty: returns (mean, std).
    ///
    /// The standard deviation represents the posterior uncertainty at each point.
    /// It is small near training data and large far from it.
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let x_train = self
            .x_train_
            .as_ref()
            .ok_or_else(|| FerroError::NotFitted {
                operation: "predict_with_std".to_string(),
            })?;
        let alpha_ = self.alpha_.as_ref().unwrap();
        let l = self.l_.as_ref().unwrap();

        // K_star: (n_new, n_train)
        let k_star = self.kernel.compute(x, x_train);

        // mean = K_star @ alpha_
        let n_new = x.nrows();
        let n_train = x_train.nrows();
        let mut mean = Array1::zeros(n_new);
        for i in 0..n_new {
            let mut s = 0.0;
            for j in 0..n_train {
                s += k_star[[i, j]] * alpha_[j];
            }
            mean[i] = s;
        }

        // v = L^{-1} @ K_star^T  (forward solve, column by column)
        // K_star^T is (n_train, n_new)
        let k_star_t = k_star.t().to_owned();
        let v = solve_lower_triangular_mat(l, &k_star_t);

        // var = diag(K(X_new, X_new)) - column_norms_squared(v)
        let k_diag = self.kernel.diagonal(x);
        let mut var = Array1::zeros(n_new);
        for j in 0..n_new {
            let mut col_sq_norm = 0.0;
            for i in 0..n_train {
                col_sq_norm += v[[i, j]] * v[[i, j]];
            }
            var[j] = stable_posterior_variance(k_diag[j], col_sq_norm);
        }

        let std = var.mapv(f64::sqrt);

        // Un-normalize
        if self.normalize_y {
            let y_mean = self.y_train_mean_.unwrap_or(0.0);
            let y_std = self.y_train_std_.unwrap_or(1.0);
            let mean_out = mean.mapv(|v| v * y_std + y_mean);
            let std_out = std.mapv(|v| v * y_std);
            Ok((mean_out, std_out))
        } else {
            Ok((mean, std))
        }
    }

    /// Get the log marginal likelihood from the last fit.
    pub fn log_marginal_likelihood(&self) -> Result<f64> {
        self.log_marginal_likelihood_
            .ok_or_else(|| FerroError::NotFitted {
                operation: "log_marginal_likelihood".to_string(),
            })
    }
}

impl Model for GaussianProcessRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        if n != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("({}, _)", n),
                actual: format!("y has length {}", y.len()),
            });
        }
        if n == 0 {
            return Err(FerroError::InvalidInput("empty training data".to_string()));
        }

        // Optionally normalize y
        let (y_fit, y_mean, y_std) = if self.normalize_y {
            let mean = y.mean().unwrap_or(0.0);
            let variance = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            let std = if variance < 1e-16 {
                1.0
            } else {
                variance.sqrt()
            };
            let y_norm = y.mapv(|v| (v - mean) / std);
            (y_norm, mean, std)
        } else {
            (y.clone(), 0.0, 1.0)
        };

        // Compute K + alpha * I
        let mut k = self.kernel.compute(x, x);
        for i in 0..n {
            k[[i, i]] += self.alpha;
        }

        // Cholesky decomposition
        let l = cholesky(&k)?;

        // Solve L @ L^T @ alpha_ = y_fit
        let z = solve_lower_triangular(&l, &y_fit);
        let alpha_ = solve_upper_triangular_transpose(&l, &z);

        // Log marginal likelihood: -0.5 * y^T alpha_ - sum(log(diag(L))) - n/2 * log(2*pi)
        let mut yt_alpha = 0.0;
        for i in 0..n {
            yt_alpha += y_fit[i] * alpha_[i];
        }
        let log_det = (0..n).map(|i| l[[i, i]].ln()).sum::<f64>();
        let lml = -0.5 * yt_alpha - log_det - (n as f64) / 2.0 * (2.0 * std::f64::consts::PI).ln();

        self.x_train_ = Some(x.clone());
        self.y_train_mean_ = Some(y_mean);
        self.y_train_std_ = Some(y_std);
        self.alpha_ = Some(alpha_);
        self.l_ = Some(l);
        self.log_marginal_likelihood_ = Some(lml);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let x_train = self
            .x_train_
            .as_ref()
            .ok_or_else(|| FerroError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let alpha_ = self.alpha_.as_ref().unwrap();

        let k_star = self.kernel.compute(x, x_train);
        let n_new = x.nrows();
        let n_train = x_train.nrows();
        let mut mean = Array1::zeros(n_new);
        for i in 0..n_new {
            let mut s = 0.0;
            for j in 0..n_train {
                s += k_star[[i, j]] * alpha_[j];
            }
            mean[i] = s;
        }

        if self.normalize_y {
            let y_mean = self.y_train_mean_.unwrap_or(0.0);
            let y_std = self.y_train_std_.unwrap_or(1.0);
            Ok(mean.mapv(|v| v * y_std + y_mean))
        } else {
            Ok(mean)
        }
    }

    fn is_fitted(&self) -> bool {
        self.x_train_.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.x_train_.as_ref().map(|x| x.ncols())
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float_log("alpha", 1e-12, 1.0)
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}

// =============================================================================
// GaussianProcessClassifier
// =============================================================================

/// Gaussian Process Classifier using Laplace approximation.
///
/// Implements binary classification. Labels are mapped to {0, 1}.
///
/// # Example
///
/// ```
/// use ferroml_core::models::gaussian_process::{GaussianProcessClassifier, RBF};
/// use ferroml_core::models::Model;
/// use ndarray::{Array1, Array2};
///
/// let kernel = RBF::new(1.0);
/// let mut gpc = GaussianProcessClassifier::new(Box::new(kernel));
/// let x = Array2::from_shape_vec((6, 2), vec![
///     1.0, 1.0, 2.0, 1.0, 1.0, 2.0,
///     5.0, 5.0, 6.0, 5.0, 5.0, 6.0,
/// ]).unwrap();
/// let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
/// gpc.fit(&x, &y).unwrap();
/// let preds = gpc.predict(&x).unwrap();
/// assert_eq!(preds.len(), 6);
/// ```
#[derive(Debug, Clone)]
pub struct GaussianProcessClassifier {
    kernel: Box<dyn Kernel>,
    max_iter: usize,
    tol: f64,
    // Fitted state
    x_train_: Option<Array2<f64>>,
    y_train_: Option<Array1<f64>>,
    f_hat_: Option<Array1<f64>>,
    classes_: Option<Vec<f64>>,
    l_: Option<Array2<f64>>,
    k_: Option<Array2<f64>>,
}

impl GaussianProcessClassifier {
    /// Create a new GP classifier with the given kernel.
    ///
    /// Defaults: max_iter=50, tol=1e-6.
    #[must_use]
    pub fn new(kernel: Box<dyn Kernel>) -> Self {
        Self {
            kernel,
            max_iter: 50,
            tol: 1e-6,
            x_train_: None,
            y_train_: None,
            f_hat_: None,
            classes_: None,
            l_: None,
            k_: None,
        }
    }

    /// Set maximum Newton iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Predict class probabilities. Returns Array2 with columns [P(class=0), P(class=1)].
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let x_train = self
            .x_train_
            .as_ref()
            .ok_or_else(|| FerroError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;
        let f_hat = self.f_hat_.as_ref().unwrap();
        let y_train = self.y_train_.as_ref().unwrap();
        let l = self.l_.as_ref().unwrap();
        let _k_stored = self.k_.as_ref().unwrap();
        let n_train = x_train.nrows();
        let n_new = x.nrows();

        // pi = sigmoid(f_hat)
        let pi: Array1<f64> = f_hat.mapv(sigmoid);

        // W = pi * (1 - pi)
        let w: Array1<f64> = &pi * &(1.0 - &pi);

        // Compute predictive mean: k_star^T @ (y - pi)
        let k_star = self.kernel.compute(x, x_train);
        let grad: Array1<f64> = y_train - &pi;

        let mut f_mean = Array1::zeros(n_new);
        for i in 0..n_new {
            let mut s = 0.0;
            for j in 0..n_train {
                s += k_star[[i, j]] * grad[j];
            }
            f_mean[i] = s;
        }

        // Compute predictive variance for probit approximation
        // v = L^{-1} @ W^{1/2} @ k_star^T
        let mut wk = Array2::zeros((n_train, n_new));
        for j in 0..n_new {
            for i in 0..n_train {
                wk[[i, j]] = w[i].sqrt() * k_star[[j, i]];
            }
        }
        let v = solve_lower_triangular_mat(l, &wk);

        let k_diag = self.kernel.diagonal(x);
        let mut f_var = Array1::zeros(n_new);
        for j in 0..n_new {
            let mut col_sq = 0.0;
            for i in 0..n_train {
                col_sq += v[[i, j]] * v[[i, j]];
            }
            f_var[j] = stable_posterior_variance(k_diag[j], col_sq);
        }

        // Probit approximation: P(y=1|x) = sigmoid(kappa * f_mean)
        // where kappa = 1 / sqrt(1 + pi/8 * f_var)
        let mut probas = Array2::zeros((n_new, 2));
        for i in 0..n_new {
            let kappa = 1.0 / (1.0 + std::f64::consts::FRAC_PI_8 * f_var[i]).sqrt();
            let p1 = sigmoid(kappa * f_mean[i]);
            probas[[i, 0]] = 1.0 - p1;
            probas[[i, 1]] = p1;
        }

        Ok(probas)
    }
}

impl Model for GaussianProcessClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        if n != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("({}, _)", n),
                actual: format!("y has length {}", y.len()),
            });
        }
        if n == 0 {
            return Err(FerroError::InvalidInput("empty training data".to_string()));
        }

        // Extract classes and map to {0, 1}
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        if classes.len() != 2 {
            return Err(FerroError::InvalidInput(format!(
                "GaussianProcessClassifier requires exactly 2 classes, got {}",
                classes.len()
            )));
        }

        let y_binary: Array1<f64> = y.mapv(|v| {
            if (v - classes[0]).abs() < 1e-10 {
                0.0
            } else {
                1.0
            }
        });

        // Compute kernel matrix
        let k = self.kernel.compute(x, x);

        // Newton's method to find f_hat (mode of Laplace approximation)
        let mut f = Array1::zeros(n);

        for _iter in 0..self.max_iter {
            // pi = sigmoid(f)
            let pi: Array1<f64> = f.mapv(sigmoid);

            // W = diag(pi * (1 - pi))
            let w: Array1<f64> = &pi * &(1.0 - &pi);

            // Clamp W to avoid numerical issues
            let w_clamped: Array1<f64> = w.mapv(|v| v.max(1e-12));
            let w_sqrt: Array1<f64> = w_clamped.mapv(f64::sqrt);

            // B = I + W^{1/2} K W^{1/2}
            let mut b = Array2::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    b[[i, j]] = w_sqrt[i] * k[[i, j]] * w_sqrt[j];
                }
                b[[i, i]] += 1.0;
            }

            // L = cholesky(B)
            let l = cholesky(&b)?;

            // b_vec = W f + (y - pi)
            let grad: Array1<f64> = &y_binary - &pi;
            let b_vec: Array1<f64> = &w_clamped * &f + &grad;

            // a = b_vec - W^{1/2} L^{-T} L^{-1} W^{1/2} K b_vec
            // First compute K @ b_vec
            let mut kb = Array1::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += k[[i, j]] * b_vec[j];
                }
                kb[i] = s;
            }
            // W^{1/2} @ K @ b_vec
            let wkb: Array1<f64> = &w_sqrt * &kb;
            // L^{-1} @ wkb
            let z = solve_lower_triangular(&l, &wkb);
            // L^{-T} @ z
            let z2 = solve_upper_triangular_transpose(&l, &z);
            // W^{1/2} @ z2
            let wz2: Array1<f64> = &w_sqrt * &z2;
            let a: Array1<f64> = &b_vec - &wz2;

            // f_new = K @ a
            let mut f_new = Array1::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n {
                    s += k[[i, j]] * a[j];
                }
                f_new[i] = s;
            }

            // Check convergence
            let diff: f64 = (&f_new - &f).mapv(|v| v.abs()).sum();
            f = f_new;

            // Store L for prediction
            self.l_ = Some(l);

            if diff < self.tol {
                break;
            }
        }

        self.x_train_ = Some(x.clone());
        self.y_train_ = Some(y_binary);
        self.f_hat_ = Some(f);
        self.classes_ = Some(classes);
        self.k_ = Some(k);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().unwrap();
        let n = x.nrows();
        let mut preds = Array1::zeros(n);
        for i in 0..n {
            preds[i] = if probas[[i, 1]] >= 0.5 {
                classes[1]
            } else {
                classes[0]
            };
        }
        Ok(preds)
    }

    fn is_fitted(&self) -> bool {
        self.x_train_.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.x_train_.as_ref().map(|x| x.ncols())
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }
}

// =============================================================================
// Inducing Point Selection
// =============================================================================

/// Method for selecting inducing points in sparse GP models.
#[derive(Debug, Clone)]
pub enum InducingPointMethod {
    /// Random subset of training data.
    RandomSubset { seed: Option<u64> },
    /// K-means cluster centers.
    KMeans { max_iter: usize, seed: Option<u64> },
    /// Greedy variance maximization: iteratively pick point with highest posterior variance.
    GreedyVariance { seed: Option<u64> },
}

/// Select m inducing points from the training data.
///
/// Returns an array of shape (m, d) containing the selected inducing points.
pub fn select_inducing_points(
    x: &Array2<f64>,
    m: usize,
    method: &InducingPointMethod,
    kernel: &dyn Kernel,
) -> Result<Array2<f64>> {
    let n = x.nrows();
    if n == 0 {
        return Err(FerroError::InvalidInput(
            "cannot select inducing points from empty data".to_string(),
        ));
    }
    // Clamp m to n
    let m = m.min(n);
    if m == 0 {
        return Err(FerroError::InvalidInput(
            "number of inducing points must be > 0".to_string(),
        ));
    }
    let d = x.ncols();

    match method {
        InducingPointMethod::RandomSubset { seed } => {
            use rand::prelude::*;
            use rand::SeedableRng;
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            indices.truncate(m);
            let mut z = Array2::zeros((m, d));
            for (i, &idx) in indices.iter().enumerate() {
                z.row_mut(i).assign(&x.row(idx));
            }
            Ok(z)
        }
        InducingPointMethod::KMeans { max_iter, seed } => {
            use crate::clustering::kmeans::KMeans as KMeansClustering;
            use crate::clustering::ClusteringModel;
            let mut km = KMeansClustering::new(m);
            km = km.max_iter(*max_iter);
            if let Some(s) = seed {
                km = km.random_state(*s);
            }
            km.fit(x)?;
            Ok(km.cluster_centers().unwrap().clone())
        }
        InducingPointMethod::GreedyVariance { seed } => {
            use rand::prelude::*;
            use rand::SeedableRng;
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));

            let mut selected = Vec::with_capacity(m);
            let first = rng.random_range(0..n);
            selected.push(first);

            // Track posterior variance at each candidate point
            // Initially, var = diag(K(X,X))
            let k_diag = kernel.diagonal(x);
            let mut var = k_diag.to_vec();

            for _ in 1..m {
                // Pick the point with highest remaining variance
                let mut best_idx = 0;
                let mut best_var = -1.0;
                for i in 0..n {
                    if !selected.contains(&i) && var[i] > best_var {
                        best_var = var[i];
                        best_idx = i;
                    }
                }
                selected.push(best_idx);

                // Update variance: subtract the reduction from the new inducing point
                // This is a rank-1 update: var_i -= k(x_i, z_new)^2 / (var[z_new] + jitter)
                let z_row = x
                    .row(best_idx)
                    .to_owned()
                    .into_shape_with_order((1, d))
                    .unwrap();
                let k_col = kernel.compute(x, &z_row); // (n, 1)
                let denom = var[best_idx].max(1e-10);
                for i in 0..n {
                    let reduction = k_col[[i, 0]] * k_col[[i, 0]] / denom;
                    var[i] = stable_posterior_variance(var[i], reduction);
                }
            }

            let mut z = Array2::zeros((m, d));
            for (i, &idx) in selected.iter().enumerate() {
                z.row_mut(i).assign(&x.row(idx));
            }
            Ok(z)
        }
    }
}

// =============================================================================
// Sparse GP helper functions
// =============================================================================

/// Compute column norms squared: for each column j, sum of v[i,j]^2 over i.
fn column_norms_squared(v: &Array2<f64>) -> Array1<f64> {
    let n_cols = v.ncols();
    let n_rows = v.nrows();
    let mut norms = Array1::zeros(n_cols);
    for j in 0..n_cols {
        let mut s = 0.0;
        for i in 0..n_rows {
            s += v[[i, j]] * v[[i, j]];
        }
        norms[j] = s;
    }
    norms
}

/// Solve L @ L^T @ x = b via forward + backward substitution.
fn solve_cholesky_vec(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let z = solve_lower_triangular(l, b);
    solve_upper_triangular_transpose(l, &z)
}

// =============================================================================
// SparseApproximation
// =============================================================================

/// Approximation method for sparse GP models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SparseApproximation {
    /// Fully Independent Training Conditional (Snelson & Ghahramani, 2006).
    FITC,
    /// Variational Free Energy (Titsias, 2009).
    VFE,
}

// =============================================================================
// SparseGPRegressor
// =============================================================================

/// Sparse Gaussian Process Regressor using inducing points.
///
/// Supports FITC and VFE approximations with O(nm^2) complexity instead of O(n^3).
///
/// # Example
///
/// ```
/// use ferroml_core::models::gaussian_process::{SparseGPRegressor, RBF, SparseApproximation, InducingPointMethod};
/// use ferroml_core::models::Model;
/// use ndarray::{Array1, Array2};
///
/// let kernel = RBF::new(1.0);
/// let mut sgpr = SparseGPRegressor::new(Box::new(kernel))
///     .with_n_inducing(10)
///     .with_alpha(0.01);
/// let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64 * 0.3).collect()).unwrap();
/// let y = x.column(0).mapv(f64::sin);
/// sgpr.fit(&x, &y).unwrap();
/// let preds = sgpr.predict(&x).unwrap();
/// assert_eq!(preds.len(), 20);
/// ```
#[derive(Debug, Clone)]
pub struct SparseGPRegressor {
    kernel: Box<dyn Kernel>,
    alpha: f64,
    normalize_y: bool,
    n_inducing: usize,
    inducing_method: InducingPointMethod,
    approximation: SparseApproximation,
    // Fitted state
    inducing_points_: Option<Array2<f64>>,
    l_m_: Option<Array2<f64>>,
    l_b_: Option<Array2<f64>>,
    woodbury_vec_: Option<Array1<f64>>,
    y_train_mean_: Option<f64>,
    y_train_std_: Option<f64>,
    log_marginal_likelihood_: Option<f64>,
    n_features_: Option<usize>,
}

impl SparseGPRegressor {
    /// Create a new Sparse GP Regressor with the given kernel.
    #[must_use]
    pub fn new(kernel: Box<dyn Kernel>) -> Self {
        Self {
            kernel,
            alpha: 1e-2,
            normalize_y: false,
            n_inducing: 100,
            inducing_method: InducingPointMethod::KMeans {
                max_iter: 100,
                seed: Some(42),
            },
            approximation: SparseApproximation::FITC,
            inducing_points_: None,
            l_m_: None,
            l_b_: None,
            woodbury_vec_: None,
            y_train_mean_: None,
            y_train_std_: None,
            log_marginal_likelihood_: None,
            n_features_: None,
        }
    }

    /// Set noise/regularization parameter (sigma^2).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to normalize y before fitting.
    #[must_use]
    pub fn with_normalize_y(mut self, normalize_y: bool) -> Self {
        self.normalize_y = normalize_y;
        self
    }

    /// Set number of inducing points.
    #[must_use]
    pub fn with_n_inducing(mut self, n_inducing: usize) -> Self {
        self.n_inducing = n_inducing;
        self
    }

    /// Set the inducing point selection method.
    #[must_use]
    pub fn with_inducing_method(mut self, method: InducingPointMethod) -> Self {
        self.inducing_method = method;
        self
    }

    /// Set the sparse approximation method (FITC or VFE).
    #[must_use]
    pub fn with_approximation(mut self, approx: SparseApproximation) -> Self {
        self.approximation = approx;
        self
    }

    /// Predict with uncertainty: returns (mean, std).
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let z = self
            .inducing_points_
            .as_ref()
            .ok_or_else(|| FerroError::NotFitted {
                operation: "predict_with_std".to_string(),
            })?;
        let l_m = self.l_m_.as_ref().unwrap();
        let l_b = self.l_b_.as_ref().unwrap();
        let woodbury_vec = self.woodbury_vec_.as_ref().unwrap();

        let n_new = x.nrows();
        let m = z.nrows();

        // K_star_m = K(X_*, Z), shape (n_new, m)
        let k_star_m = self.kernel.compute(x, z);

        // mean = K_star_m @ L_m^{-T} @ woodbury_vec
        // First: v_mean = L_m^{-1} @ K_m_star, then mean = v_mean^T @ woodbury_vec
        // More directly: solve L_m^T @ tmp = woodbury_vec, then mean = K_star_m @ tmp
        let tmp = solve_upper_triangular_transpose(l_m, woodbury_vec);
        let mut mean = Array1::zeros(n_new);
        for i in 0..n_new {
            let mut s = 0.0;
            for j in 0..m {
                s += k_star_m[[i, j]] * tmp[j];
            }
            mean[i] = s;
        }

        // Variance
        // v1 = L_m^{-1} @ K_m_star^T, shape (m, n_new)
        let k_m_star_t = k_star_m.t().to_owned();
        let v1 = solve_lower_triangular_mat(l_m, &k_m_star_t);
        // v2 = L_B^{-1} @ v1, shape (m, n_new)
        let v2 = solve_lower_triangular_mat(l_b, &v1);

        let k_diag = self.kernel.diagonal(x);
        let v1_norms = column_norms_squared(&v1);
        let v2_norms = column_norms_squared(&v2);

        let mut var = Array1::zeros(n_new);
        for i in 0..n_new {
            var[i] = stable_posterior_variance(k_diag[i], v1_norms[i] - v2_norms[i]);
        }
        let std = var.mapv(f64::sqrt);

        if self.normalize_y {
            let y_mean = self.y_train_mean_.unwrap_or(0.0);
            let y_std = self.y_train_std_.unwrap_or(1.0);
            Ok((mean.mapv(|v| v * y_std + y_mean), std.mapv(|v| v * y_std)))
        } else {
            Ok((mean, std))
        }
    }

    /// Get the log marginal likelihood from the last fit.
    pub fn log_marginal_likelihood(&self) -> Result<f64> {
        self.log_marginal_likelihood_
            .ok_or_else(|| FerroError::NotFitted {
                operation: "log_marginal_likelihood".to_string(),
            })
    }

    /// Get the inducing points from the last fit.
    pub fn inducing_points(&self) -> Option<&Array2<f64>> {
        self.inducing_points_.as_ref()
    }
}

impl Model for SparseGPRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        if n != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("({}, _)", n),
                actual: format!("y has length {}", y.len()),
            });
        }
        if n == 0 {
            return Err(FerroError::InvalidInput("empty training data".to_string()));
        }

        let m = self.n_inducing.min(n);

        // Normalize y if requested
        let (y_fit, y_mean, y_std) = if self.normalize_y {
            let mean_val = y.mean().unwrap_or(0.0);
            let variance = y.iter().map(|&v| (v - mean_val).powi(2)).sum::<f64>() / n as f64;
            let std_val = if variance < 1e-16 {
                1.0
            } else {
                variance.sqrt()
            };
            let y_norm = y.mapv(|v| (v - mean_val) / std_val);
            (y_norm, mean_val, std_val)
        } else {
            (y.clone(), 0.0, 1.0)
        };

        // 1. Select inducing points
        let z = select_inducing_points(x, m, &self.inducing_method, self.kernel.as_ref())?;
        let m = z.nrows(); // actual m (may be clamped)

        // 2. K_mm + jitter
        let jitter = 1e-6;
        let mut k_mm = self.kernel.compute(&z, &z);
        for i in 0..m {
            k_mm[[i, i]] += jitter;
        }

        // 3. L_m = cholesky(K_mm)
        let l_m = cholesky(&k_mm)?;

        // 4. K_nm = K(X, Z), shape (n, m)
        let k_nm = self.kernel.compute(x, &z);

        // 5. V = L_m^{-1} @ K_mn = L_m^{-1} @ K_nm^T, shape (m, n)
        let k_mn = k_nm.t().to_owned();
        let v = solve_lower_triangular_mat(&l_m, &k_mn);

        // 6. K_nn_diag
        let k_nn_diag = self.kernel.diagonal(x);

        // 7. Q_nn_diag = column_norms_squared(V) = diag(V^T V)
        let q_nn_diag = column_norms_squared(&v);

        // 8. Lambda_diag
        let sigma2 = self.alpha;
        let lambda_diag: Array1<f64> = match self.approximation {
            SparseApproximation::FITC => {
                let mut ld = Array1::zeros(n);
                for i in 0..n {
                    ld[i] = stable_posterior_variance(k_nn_diag[i], q_nn_diag[i]) + sigma2;
                    ld[i] = ld[i].max(1e-10);
                }
                ld
            }
            SparseApproximation::VFE => Array1::from_elem(n, sigma2.max(1e-10)),
        };

        // 9. Lambda_inv
        let lambda_inv: Array1<f64> = lambda_diag.mapv(|v| 1.0 / v);

        // 10. V_scaled: scale each column j of V by sqrt(lambda_inv[j])
        let mut v_scaled = v.clone();
        for j in 0..n {
            let scale = lambda_inv[j].sqrt();
            for i in 0..m {
                v_scaled[[i, j]] *= scale;
            }
        }

        // 11. B = I_m + V_scaled @ V_scaled^T, shape (m, m)
        let mut b_mat = Array2::eye(m);
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k in 0..n {
                    s += v_scaled[[i, k]] * v_scaled[[j, k]];
                }
                b_mat[[i, j]] += s;
            }
        }

        // 12. L_B = cholesky(B)
        let l_b = cholesky(&b_mat)?;

        // 13. beta = Lambda_inv * y_fit
        let beta: Array1<f64> = &lambda_inv * &y_fit;

        // 14. woodbury_vec = L_B^{-T} @ L_B^{-1} @ (V @ beta)
        // First compute V @ beta, shape (m,)
        let mut v_beta = Array1::zeros(m);
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..n {
                s += v[[i, j]] * beta[j];
            }
            v_beta[i] = s;
        }
        let woodbury_vec = solve_cholesky_vec(&l_b, &v_beta);

        // 15. Log marginal likelihood
        // -0.5 * (y^T Lambda_inv y - beta^T V^T woodbury_vec + log|B| + sum(log Lambda_diag) + n*log(2pi))
        let yt_lambda_inv_y: f64 = y_fit
            .iter()
            .zip(lambda_inv.iter())
            .map(|(&yi, &li)| yi * yi * li)
            .sum();
        // beta^T V^T woodbury_vec = (V @ beta)^T @ woodbury_vec = v_beta . woodbury_vec
        let beta_vt_w: f64 = v_beta
            .iter()
            .zip(woodbury_vec.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let log_det_b: f64 = (0..m).map(|i| l_b[[i, i]].ln()).sum::<f64>() * 2.0;
        let log_lambda_sum: f64 = lambda_diag.iter().map(|v| v.ln()).sum();
        let n_log_2pi = n as f64 * (2.0 * std::f64::consts::PI).ln();

        let mut lml = -0.5 * (yt_lambda_inv_y - beta_vt_w + log_det_b + log_lambda_sum + n_log_2pi);

        // VFE trace penalty
        if self.approximation == SparseApproximation::VFE {
            let trace_term: f64 = (0..n).map(|i| k_nn_diag[i] - q_nn_diag[i]).sum::<f64>();
            lml -= trace_term / (2.0 * sigma2);
        }

        self.inducing_points_ = Some(z);
        self.l_m_ = Some(l_m);
        self.l_b_ = Some(l_b);
        self.woodbury_vec_ = Some(woodbury_vec);
        self.y_train_mean_ = Some(y_mean);
        self.y_train_std_ = Some(y_std);
        self.log_marginal_likelihood_ = Some(lml);
        self.n_features_ = Some(x.ncols());

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let (mean, _) = self.predict_with_std(x)?;
        Ok(mean)
    }

    fn is_fitted(&self) -> bool {
        self.inducing_points_.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features_
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float_log("alpha", 1e-6, 1.0)
            .int("n_inducing", 10, 500)
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}

// =============================================================================
// SparseGPClassifier
// =============================================================================

/// Sparse Gaussian Process Classifier using FITC + Laplace approximation.
///
/// Binary classification with O(nm^2) complexity via inducing points.
#[derive(Debug, Clone)]
pub struct SparseGPClassifier {
    kernel: Box<dyn Kernel>,
    n_inducing: usize,
    inducing_method: InducingPointMethod,
    max_iter: usize,
    tol: f64,
    // Fitted state
    inducing_points_: Option<Array2<f64>>,
    classes_: Option<Vec<f64>>,
    // Store precomputed quantities for prediction
    l_m_: Option<Array2<f64>>,
    alpha_vec_: Option<Array1<f64>>, // K_mm^{-1} @ K_mn @ (y - pi)
    l_approx_: Option<Array2<f64>>,  // chol of approximate B for variance
    n_features_: Option<usize>,
}

impl SparseGPClassifier {
    /// Create a new Sparse GP Classifier with the given kernel.
    #[must_use]
    pub fn new(kernel: Box<dyn Kernel>) -> Self {
        Self {
            kernel,
            n_inducing: 100,
            inducing_method: InducingPointMethod::KMeans {
                max_iter: 100,
                seed: Some(42),
            },
            max_iter: 50,
            tol: 1e-6,
            inducing_points_: None,
            classes_: None,
            l_m_: None,
            alpha_vec_: None,
            l_approx_: None,
            n_features_: None,
        }
    }

    /// Set number of inducing points.
    #[must_use]
    pub fn with_n_inducing(mut self, n_inducing: usize) -> Self {
        self.n_inducing = n_inducing;
        self
    }

    /// Set the inducing point selection method.
    #[must_use]
    pub fn with_inducing_method(mut self, method: InducingPointMethod) -> Self {
        self.inducing_method = method;
        self
    }

    /// Set maximum Newton iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Get the inducing points from the last fit.
    pub fn inducing_points(&self) -> Option<&Array2<f64>> {
        self.inducing_points_.as_ref()
    }

    /// Predict class probabilities. Returns Array2 with columns [P(class=0), P(class=1)].
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let z = self
            .inducing_points_
            .as_ref()
            .ok_or_else(|| FerroError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;
        let l_m = self.l_m_.as_ref().unwrap();
        let alpha_vec = self.alpha_vec_.as_ref().unwrap();
        let l_approx = self.l_approx_.as_ref().unwrap();

        let n_new = x.nrows();
        let m = z.nrows();

        // K_star_m = K(X_*, Z)
        let k_star_m = self.kernel.compute(x, z);

        // Predictive mean: f_mean = K_star_m @ alpha_vec
        let mut f_mean = Array1::zeros(n_new);
        for i in 0..n_new {
            let mut s = 0.0;
            for j in 0..m {
                s += k_star_m[[i, j]] * alpha_vec[j];
            }
            f_mean[i] = s;
        }

        // Predictive variance via low-rank representation
        let k_m_star_t = k_star_m.t().to_owned();
        let v1 = solve_lower_triangular_mat(l_m, &k_m_star_t);
        let v2 = solve_lower_triangular_mat(l_approx, &v1);

        let k_diag = self.kernel.diagonal(x);
        let v1_norms = column_norms_squared(&v1);
        let v2_norms = column_norms_squared(&v2);

        let mut probas = Array2::zeros((n_new, 2));
        for i in 0..n_new {
            let f_var = stable_posterior_variance(k_diag[i], v1_norms[i] - v2_norms[i]);
            let kappa = 1.0 / (1.0 + std::f64::consts::FRAC_PI_8 * f_var).sqrt();
            let p1 = sigmoid(kappa * f_mean[i]);
            probas[[i, 0]] = 1.0 - p1;
            probas[[i, 1]] = p1;
        }

        Ok(probas)
    }
}

impl Model for SparseGPClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        if n != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("({}, _)", n),
                actual: format!("y has length {}", y.len()),
            });
        }
        if n == 0 {
            return Err(FerroError::InvalidInput("empty training data".to_string()));
        }

        // Extract classes
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        if classes.len() != 2 {
            return Err(FerroError::InvalidInput(format!(
                "SparseGPClassifier requires exactly 2 classes, got {}",
                classes.len()
            )));
        }

        let y_binary: Array1<f64> = y.mapv(|v| {
            if (v - classes[0]).abs() < 1e-10 {
                0.0
            } else {
                1.0
            }
        });

        let m = self.n_inducing.min(n);

        // Select inducing points
        let z = select_inducing_points(x, m, &self.inducing_method, self.kernel.as_ref())?;
        let m = z.nrows();

        // K_mm + jitter
        let jitter = 1e-6;
        let mut k_mm = self.kernel.compute(&z, &z);
        for i in 0..m {
            k_mm[[i, i]] += jitter;
        }
        let l_m = cholesky(&k_mm)?;

        // K_nm = K(X, Z)
        let k_nm = self.kernel.compute(x, &z);

        // Q_nn approximation via Nystrom: Q = K_nm @ K_mm^{-1} @ K_mn
        // Compute A = L_m^{-1} @ K_mn, shape (m, n)
        let k_mn = k_nm.t().to_owned();
        let a_mat = solve_lower_triangular_mat(&l_m, &k_mn);

        // Newton iteration for the latent function values
        // f = Q_nn @ alpha_n, parameterized through inducing space
        // We work with f directly at training points (low-rank approximation)
        let mut f = Array1::zeros(n);

        let mut last_l_approx = Array2::eye(m);

        for _iter in 0..self.max_iter {
            let pi: Array1<f64> = f.mapv(sigmoid);
            let w: Array1<f64> = &pi * &(1.0 - &pi);
            let w_clamped: Array1<f64> = w.mapv(|v| v.max(1e-12));
            let w_sqrt: Array1<f64> = w_clamped.mapv(f64::sqrt);

            // B = I_m + A @ W @ A^T where A = L_m^{-1} @ K_mn
            // A_scaled = A @ diag(sqrt(W))
            let mut a_scaled = a_mat.clone();
            for j in 0..n {
                for i in 0..m {
                    a_scaled[[i, j]] *= w_sqrt[j];
                }
            }

            // B = I + A_scaled @ A_scaled^T
            let mut b_mat = Array2::eye(m);
            for i in 0..m {
                for j in 0..m {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += a_scaled[[i, k]] * a_scaled[[j, k]];
                    }
                    b_mat[[i, j]] += s;
                }
            }

            let l_b = cholesky(&b_mat)?;
            last_l_approx = l_b.clone();

            // grad = y - pi
            let grad: Array1<f64> = &y_binary - &pi;

            // b_vec = W * f + grad
            let b_vec: Array1<f64> = &w_clamped * &f + &grad;

            // Using the Woodbury identity to compute the Newton step:
            // f_new = Q_nn @ (I + W Q_nn)^{-1} @ b_vec
            // = A^T @ (I + A diag(W) A^T)^{-1} @ A @ b_vec (approx via L_m^{-1})
            // More precisely: f_new = Q_nn b_vec - Q_nn W^{1/2} L_B^{-T} L_B^{-1} W^{1/2} Q_nn b_vec

            // Compute A^T @ (something) for Q_nn @ b_vec
            // Q_nn @ b_vec = K_nm @ K_mm^{-1} @ K_mn @ b_vec
            //              = A^T @ A @ b_vec (since A = L_m^{-1} K_mn)
            // c1 = A @ b_vec, shape (m,)
            let mut c1 = Array1::zeros(m);
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..n {
                    s += a_mat[[i, j]] * b_vec[j];
                }
                c1[i] = s;
            }

            // c2 = A @ diag(sqrt(W)) @ Q_nn @ b_vec = A_scaled @ (A^T @ c1)
            // Wait, let's just do: W^{1/2} Q_nn b_vec
            // q_b = A^T @ c1 = Q_nn @ b_vec, shape (n,)
            let mut q_b = Array1::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..m {
                    s += a_mat[[j, i]] * c1[j];
                }
                q_b[i] = s;
            }

            // w_sqrt_qb = W^{1/2} @ q_b
            let w_sqrt_qb: Array1<f64> = &w_sqrt * &q_b;

            // c3 = A @ w_sqrt_qb, shape (m,)
            let mut c3 = Array1::zeros(m);
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..n {
                    s += a_mat[[i, j]] * w_sqrt_qb[j];
                }
                c3[i] = s;
            }

            // c4 = L_B^{-T} @ L_B^{-1} @ c3
            let c4 = solve_cholesky_vec(&l_b, &c3);

            // c5 = A^T @ c4, shape (n,)
            let mut c5 = Array1::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..m {
                    s += a_mat[[j, i]] * c4[j];
                }
                c5[i] = s;
            }

            // c6 = W^{1/2} @ c5
            let c6: Array1<f64> = &w_sqrt * &c5;

            // Q_nn c6 = A^T A w_sqrt c5
            let mut c7_m = Array1::zeros(m);
            for i in 0..m {
                let mut s = 0.0;
                for j in 0..n {
                    s += a_mat[[i, j]] * c6[j];
                }
                c7_m[i] = s;
            }
            let mut q_c6 = Array1::zeros(n);
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..m {
                    s += a_mat[[j, i]] * c7_m[j];
                }
                q_c6[i] = s;
            }

            // f_new = q_b - q_c6
            let f_new: Array1<f64> = &q_b - &q_c6;

            let diff: f64 = (&f_new - &f).mapv(|v| v.abs()).sum();
            f = f_new;

            if diff < self.tol {
                break;
            }
        }

        // Store prediction quantities
        // alpha_vec = K_mm^{-1} @ K_mn @ (y - sigmoid(f))
        let pi_final: Array1<f64> = f.mapv(sigmoid);
        let grad_final: Array1<f64> = &y_binary - &pi_final;

        // c = K_mn @ grad = A_mat is L_m^{-1} K_mn so K_mn = L_m @ A_mat
        // Actually we need K_mm^{-1} @ K_mn @ grad
        // K_mm^{-1} @ K_mn = L_m^{-T} @ L_m^{-1} @ K_mn = L_m^{-T} @ A_mat
        // So alpha_vec = L_m^{-T} @ A_mat @ grad
        let mut a_grad = Array1::zeros(m);
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..n {
                s += a_mat[[i, j]] * grad_final[j];
            }
            a_grad[i] = s;
        }
        let alpha_vec = solve_upper_triangular_transpose(&l_m, &a_grad);

        self.inducing_points_ = Some(z);
        self.classes_ = Some(classes);
        self.l_m_ = Some(l_m);
        self.alpha_vec_ = Some(alpha_vec);
        self.l_approx_ = Some(last_l_approx);
        self.n_features_ = Some(x.ncols());

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().unwrap();
        let n = x.nrows();
        let mut preds = Array1::zeros(n);
        for i in 0..n {
            preds[i] = if probas[[i, 1]] >= 0.5 {
                classes[1]
            } else {
                classes[0]
            };
        }
        Ok(preds)
    }

    fn is_fitted(&self) -> bool {
        self.inducing_points_.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features_
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().int("n_inducing", 10, 500)
    }
}

// =============================================================================
// SVGPRegressor
// =============================================================================

/// Sparse Variational Gaussian Process Regressor.
///
/// Uses stochastic variational inference with mini-batches, enabling
/// training on datasets with 100K+ samples. Complexity is O(bm^2) per step
/// where b is the batch size and m is the number of inducing points.
#[derive(Debug, Clone)]
pub struct SVGPRegressor {
    kernel: Box<dyn Kernel>,
    noise_variance: f64,
    n_inducing: usize,
    inducing_method: InducingPointMethod,
    n_epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    normalize_y: bool,
    // Fitted state
    inducing_points_: Option<Array2<f64>>,
    mu_: Option<Array1<f64>>,
    l_s_: Option<Array2<f64>>,
    l_mm_: Option<Array2<f64>>,
    y_train_mean_: Option<f64>,
    y_train_std_: Option<f64>,
    n_features_: Option<usize>,
}

impl SVGPRegressor {
    /// Create a new SVGP Regressor.
    #[must_use]
    pub fn new(kernel: Box<dyn Kernel>) -> Self {
        Self {
            kernel,
            noise_variance: 1.0,
            n_inducing: 100,
            inducing_method: InducingPointMethod::KMeans {
                max_iter: 100,
                seed: Some(42),
            },
            n_epochs: 100,
            batch_size: 256,
            learning_rate: 0.01,
            normalize_y: false,
            inducing_points_: None,
            mu_: None,
            l_s_: None,
            l_mm_: None,
            y_train_mean_: None,
            y_train_std_: None,
            n_features_: None,
        }
    }

    /// Set noise variance (sigma^2).
    #[must_use]
    pub fn with_noise_variance(mut self, noise_variance: f64) -> Self {
        self.noise_variance = noise_variance;
        self
    }

    /// Set number of inducing points.
    #[must_use]
    pub fn with_n_inducing(mut self, n_inducing: usize) -> Self {
        self.n_inducing = n_inducing;
        self
    }

    /// Set the inducing point selection method.
    #[must_use]
    pub fn with_inducing_method(mut self, method: InducingPointMethod) -> Self {
        self.inducing_method = method;
        self
    }

    /// Set number of training epochs.
    #[must_use]
    pub fn with_n_epochs(mut self, n_epochs: usize) -> Self {
        self.n_epochs = n_epochs;
        self
    }

    /// Set mini-batch size.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set learning rate for natural gradient updates.
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set whether to normalize y before fitting.
    #[must_use]
    pub fn with_normalize_y(mut self, normalize_y: bool) -> Self {
        self.normalize_y = normalize_y;
        self
    }

    /// Get the inducing points from the last fit.
    pub fn inducing_points(&self) -> Option<&Array2<f64>> {
        self.inducing_points_.as_ref()
    }

    /// Predict with uncertainty: returns (mean, std).
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let z = self
            .inducing_points_
            .as_ref()
            .ok_or_else(|| FerroError::NotFitted {
                operation: "predict_with_std".to_string(),
            })?;
        let mu = self.mu_.as_ref().unwrap();
        let l_s = self.l_s_.as_ref().unwrap();
        let l_mm = self.l_mm_.as_ref().unwrap();

        let n_new = x.nrows();
        let m = z.nrows();

        // K_star_m = K(X_*, Z)
        let k_star_m = self.kernel.compute(x, z);

        // alpha = K_mm^{-1} @ mu = L_mm^{-T} @ L_mm^{-1} @ mu
        let alpha = solve_cholesky_vec(l_mm, mu);

        // mean = K_star_m @ alpha
        let mut mean = Array1::zeros(n_new);
        for i in 0..n_new {
            let mut s = 0.0;
            for j in 0..m {
                s += k_star_m[[i, j]] * alpha[j];
            }
            mean[i] = s;
        }

        // Variance: K** - K*m K_mm^{-1} K_m* + K*m K_mm^{-1} S K_mm^{-1} K_m*
        let k_m_star_t = k_star_m.t().to_owned();
        // v1 = L_mm^{-1} @ K_m*, shape (m, n_new)
        let v1 = solve_lower_triangular_mat(l_mm, &k_m_star_t);
        // v2 = L_S @ L_mm^{-1} @ K_m* -- we need S = L_S @ L_S^T
        // variance_from_S = K*m K_mm^{-1} S K_mm^{-1} K_m*
        //                 = (L_mm^{-1} K_m*)^T @ L_S @ L_S^T @ (L_mm^{-1} K_m*)
        //                 = |L_S^T @ v1|^2 per column
        // v3 = L_S^T @ v1, but L_S is lower triangular, L_S^T is upper
        let mut v3 = Array2::zeros((m, n_new));
        for j in 0..n_new {
            for i in 0..m {
                let mut s = 0.0;
                for k in 0..=i {
                    // L_S^T[k, i] = L_S[i, k] -- wait, we want row i of L_S^T which is col i of L_S
                    // Actually L_S^T @ v1[:,j] element i = sum_k L_S[k,i] * v1[k,j]
                    s += l_s[[k, i]] * v1[[k, j]];
                }
                // Wait, L_S^T[i,k] = L_S[k,i], so (L_S^T @ v1)[i,j] = sum_k L_S[k,i] * v1[k,j]
                // Since L_S is lower triangular, L_S[k,i] is nonzero only for k >= i
                // Let me redo this properly
                v3[[i, j]] = s; // will recalculate below
            }
        }
        // Redo v3 = L_S^T @ v1 properly
        // (L_S^T)[i,k] = L_S[k,i], nonzero when k >= i (since L_S is lower tri)
        for j in 0..n_new {
            for i in 0..m {
                let mut s = 0.0;
                for k in i..m {
                    s += l_s[[k, i]] * v1[[k, j]];
                }
                v3[[i, j]] = s;
            }
        }

        let k_diag = self.kernel.diagonal(x);
        let v1_norms = column_norms_squared(&v1);
        let v3_norms = column_norms_squared(&v3);

        let mut var = Array1::zeros(n_new);
        for i in 0..n_new {
            var[i] = stable_posterior_variance(k_diag[i], v1_norms[i] - v3_norms[i]);
        }
        let std = var.mapv(f64::sqrt);

        if self.normalize_y {
            let y_mean = self.y_train_mean_.unwrap_or(0.0);
            let y_std = self.y_train_std_.unwrap_or(1.0);
            Ok((mean.mapv(|v| v * y_std + y_mean), std.mapv(|v| v * y_std)))
        } else {
            Ok((mean, std))
        }
    }
}

impl Model for SVGPRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        use rand::prelude::*;
        use rand::SeedableRng;

        let n = x.nrows();
        if n != y.len() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("({}, _)", n),
                actual: format!("y has length {}", y.len()),
            });
        }
        if n == 0 {
            return Err(FerroError::InvalidInput("empty training data".to_string()));
        }

        let m = self.n_inducing.min(n);

        // Normalize y
        let (y_fit, y_mean, y_std) = if self.normalize_y {
            let mean_val = y.mean().unwrap_or(0.0);
            let variance = y.iter().map(|&v| (v - mean_val).powi(2)).sum::<f64>() / n as f64;
            let std_val = if variance < 1e-16 {
                1.0
            } else {
                variance.sqrt()
            };
            let y_norm = y.mapv(|v| (v - mean_val) / std_val);
            (y_norm, mean_val, std_val)
        } else {
            (y.clone(), 0.0, 1.0)
        };

        // Select inducing points
        let z = select_inducing_points(x, m, &self.inducing_method, self.kernel.as_ref())?;
        let m = z.nrows();

        // K_mm + jitter
        let jitter = 1e-6;
        let mut k_mm = self.kernel.compute(&z, &z);
        for i in 0..m {
            k_mm[[i, i]] += jitter;
        }
        let l_mm = cholesky(&k_mm)?;

        // Initialize variational parameters
        let mut mu = Array1::zeros(m);
        // S = I initially, so L_S = I
        // We'll maintain S^{-1} = lambda (precision) for natural gradient updates
        // lambda = S^{-1}, initially I
        let mut lambda = Array2::eye(m);
        let sigma2 = self.noise_variance;

        let batch_size = self.batch_size.min(n);
        let lr = self.learning_rate;

        let mut rng = StdRng::seed_from_u64(42);

        for _epoch in 0..self.n_epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);

            let n_batches = (n + batch_size - 1) / batch_size;

            for batch_idx in 0..n_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(n);
                let b = end - start;

                // Extract mini-batch
                let mut x_b = Array2::zeros((b, x.ncols()));
                let mut y_b = Array1::zeros(b);
                for (i, &idx) in indices[start..end].iter().enumerate() {
                    x_b.row_mut(i).assign(&x.row(idx));
                    y_b[i] = y_fit[idx];
                }

                // K_bm = K(X_b, Z), shape (b, m)
                let k_bm = self.kernel.compute(&x_b, &z);

                // K_mm^{-1} @ K_mb = L_mm^{-T} @ L_mm^{-1} @ K_bm^T
                // alpha_mat = L_mm^{-1} @ K_bm^T, shape (m, b)
                let k_bm_t = k_bm.t().to_owned();
                let alpha_mat = solve_lower_triangular_mat(&l_mm, &k_bm_t);

                // K_mm^{-1} @ K_mb = L_mm^{-T} @ alpha_mat
                // We need K_mm^{-1} @ K_mb columns
                let mut kinv_kmb = Array2::zeros((m, b));
                for col in 0..b {
                    let col_vec = alpha_mat.column(col).to_owned();
                    let solved = solve_upper_triangular_transpose(&l_mm, &col_vec);
                    kinv_kmb.column_mut(col).assign(&solved);
                }

                // Natural gradient for mu:
                // d_mu = S @ (K_mm^{-1} @ K_mb @ (1/sigma^2) @ (y_b - K_bm @ K_mm^{-1} @ mu) - K_mm^{-1} @ mu)
                // = S @ (sum_b ... - K_mm^{-1} @ mu)

                // Predictive mean at batch: K_bm @ K_mm^{-1} @ mu
                let kinv_mu = solve_cholesky_vec(&l_mm, &mu);
                let mut pred_mean = Array1::zeros(b);
                for i in 0..b {
                    let mut s = 0.0;
                    for j in 0..m {
                        s += k_bm[[i, j]] * kinv_mu[j];
                    }
                    pred_mean[i] = s;
                }

                // residuals = y_b - pred_mean
                let residuals: Array1<f64> = &y_b - &pred_mean;

                // grad_mu_nat = K_mm^{-1} @ K_mb @ (1/sigma^2) @ residuals - K_mm^{-1} @ mu
                // scaled by (n/b) for mini-batch
                let scale = n as f64 / b as f64;
                let mut kinv_kmb_resid = Array1::zeros(m);
                for i in 0..m {
                    let mut s = 0.0;
                    for j in 0..b {
                        s += kinv_kmb[[i, j]] * residuals[j];
                    }
                    kinv_kmb_resid[i] = s * scale / sigma2;
                }

                // grad for precision lambda:
                // d_lambda = K_mm^{-1} @ K_mb @ (1/sigma^2) @ K_bm @ K_mm^{-1} * (n/b) - K_mm^{-1}
                // But we update lambda directly

                // Natural gradient update for precision (lambda = S^{-1}):
                // lambda_new = (1 - lr) * lambda + lr * (K_mm^{-1} + K_mm^{-1} K_mb (1/sigma^2)(n/b) K_bm K_mm^{-1})
                // The second term is kinv_kmb @ kinv_kmb^T * scale / sigma2

                // First compute kinv_kmb @ kinv_kmb^T
                let mut outer = Array2::zeros((m, m));
                for i in 0..m {
                    for j in 0..m {
                        let mut s = 0.0;
                        for k in 0..b {
                            s += kinv_kmb[[i, k]] * kinv_kmb[[j, k]];
                        }
                        outer[[i, j]] = s * scale / sigma2;
                    }
                }

                // K_mm^{-1}
                let kinv = {
                    let mut ki = Array2::zeros((m, m));
                    for i in 0..m {
                        let mut e = Array1::zeros(m);
                        e[i] = 1.0;
                        let col = solve_cholesky_vec(&l_mm, &e);
                        ki.column_mut(i).assign(&col);
                    }
                    ki
                };

                // Target lambda = K_mm^{-1} + outer
                let target_lambda = &kinv + &outer;

                // Update lambda with learning rate
                for i in 0..m {
                    for j in 0..m {
                        lambda[[i, j]] = (1.0 - lr) * lambda[[i, j]] + lr * target_lambda[[i, j]];
                    }
                }

                // Update mu: mu_new = lambda^{-1} @ (lambda_old @ mu + lr * nat_grad_for_mu_term)
                // Actually with natural gradients:
                // theta1 = lambda @ mu (natural parameter)
                // theta1_new = (1 - lr) * theta1 + lr * (K_mm^{-1} @ K_mb @ (1/sigma2)(n/b) @ y_b_resid_at_prior)
                // But let's use a simpler approach: just update mu via gradient

                // theta1_old = lambda_old @ mu (before lambda update, use old lambda)
                // Actually use the gradient directly on mu:
                // mu = mu + lr * S @ grad_mu_nat
                // where S = lambda^{-1} (use new lambda)

                // Compute S = lambda^{-1} by Cholesky of lambda
                let l_lambda = match cholesky(&lambda) {
                    Ok(l) => l,
                    Err(_) => {
                        // If lambda is not PD, add jitter
                        let mut lambda_jit = lambda.clone();
                        for i in 0..m {
                            lambda_jit[[i, i]] += 1e-6;
                        }
                        cholesky(&lambda_jit)?
                    }
                };

                // grad for mu in natural param space: kinv_kmb_resid - kinv_mu
                let grad_nat = &kinv_kmb_resid - &solve_cholesky_vec(&l_mm, &mu);

                // S @ grad_nat = lambda^{-1} @ grad_nat
                let s_grad = solve_cholesky_vec(&l_lambda, &grad_nat);

                // Update mu
                for i in 0..m {
                    mu[i] += lr * s_grad[i];
                }
            }
        }

        // Compute L_S from lambda = S^{-1}
        // S = lambda^{-1}
        let l_lambda = match cholesky(&lambda) {
            Ok(l) => l,
            Err(_) => {
                let mut lambda_jit = lambda.clone();
                for i in 0..m {
                    lambda_jit[[i, i]] += 1e-6;
                }
                cholesky(&lambda_jit)?
            }
        };

        // S = lambda^{-1}, L_S such that S = L_S @ L_S^T
        // Compute S explicitly then Cholesky
        let mut s_mat = Array2::zeros((m, m));
        for i in 0..m {
            let mut e = Array1::zeros(m);
            e[i] = 1.0;
            let col = solve_cholesky_vec(&l_lambda, &e);
            s_mat.column_mut(i).assign(&col);
        }
        // Ensure symmetry
        for i in 0..m {
            for j in (i + 1)..m {
                let avg = (s_mat[[i, j]] + s_mat[[j, i]]) / 2.0;
                s_mat[[i, j]] = avg;
                s_mat[[j, i]] = avg;
            }
        }
        // Add small jitter for numerical stability
        for i in 0..m {
            s_mat[[i, i]] = s_mat[[i, i]].max(1e-10);
        }
        let l_s = cholesky(&s_mat)?;

        self.inducing_points_ = Some(z);
        self.mu_ = Some(mu);
        self.l_s_ = Some(l_s);
        self.l_mm_ = Some(l_mm);
        self.y_train_mean_ = Some(y_mean);
        self.y_train_std_ = Some(y_std);
        self.n_features_ = Some(x.ncols());

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let (mean, _) = self.predict_with_std(x)?;
        Ok(mean)
    }

    fn is_fitted(&self) -> bool {
        self.inducing_points_.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features_
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float_log("noise_variance", 1e-4, 10.0)
            .int("n_inducing", 10, 500)
            .int("batch_size", 32, 512)
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}
