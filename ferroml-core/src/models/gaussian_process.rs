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
            var[j] = (k_diag[j] - col_sq_norm).max(0.0);
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
            f_var[j] = (k_diag[j] - col_sq).max(0.0);
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
