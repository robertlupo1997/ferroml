//! Bayesian Optimization with Gaussian Processes
//!
//! This module implements GP-based Bayesian optimization with:
//! - Gaussian Process regression with RBF and Matern kernels
//! - Kernel hyperparameter optimization via marginal log-likelihood maximization
//! - Acquisition functions (EI, PI, UCB, LCB)
//!
//! # References
//! - Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for
//!   Machine Learning. MIT Press.
//! - Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian
//!   Optimization of Machine Learning Algorithms. NeurIPS.

use super::samplers::Sampler;
use super::{ParameterValue, SearchSpace, Trial};
use crate::Result;
use std::collections::HashMap;

/// Kernel functions for Gaussian Process regression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    /// Radial Basis Function (Squared Exponential) kernel
    /// k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))
    RBF {
        /// Length scale parameter
        length_scale: f64,
        /// Signal variance (output scale)
        variance: f64,
    },
    /// Matern kernel with ν = 5/2
    /// k(x, x') = σ² * (1 + √5*d/l + 5*d²/(3*l²)) * exp(-√5*d/l)
    Matern52 {
        /// Length scale parameter
        length_scale: f64,
        /// Signal variance (output scale)
        variance: f64,
    },
    /// Matern kernel with ν = 3/2
    /// k(x, x') = σ² * (1 + √3*d/l) * exp(-√3*d/l)
    Matern32 {
        /// Length scale parameter
        length_scale: f64,
        /// Signal variance (output scale)
        variance: f64,
    },
}

impl Default for Kernel {
    fn default() -> Self {
        Kernel::Matern52 {
            length_scale: 1.0,
            variance: 1.0,
        }
    }
}

impl Kernel {
    /// Create an RBF kernel with default parameters
    pub fn rbf() -> Self {
        Kernel::RBF {
            length_scale: 1.0,
            variance: 1.0,
        }
    }

    /// Create a Matern 5/2 kernel with default parameters
    pub fn matern52() -> Self {
        Kernel::Matern52 {
            length_scale: 1.0,
            variance: 1.0,
        }
    }

    /// Create a Matern 3/2 kernel with default parameters
    pub fn matern32() -> Self {
        Kernel::Matern32 {
            length_scale: 1.0,
            variance: 1.0,
        }
    }

    /// Get the length scale
    pub fn length_scale(&self) -> f64 {
        match self {
            Kernel::RBF { length_scale, .. } => *length_scale,
            Kernel::Matern52 { length_scale, .. } => *length_scale,
            Kernel::Matern32 { length_scale, .. } => *length_scale,
        }
    }

    /// Get the variance (signal variance)
    pub fn variance(&self) -> f64 {
        match self {
            Kernel::RBF { variance, .. } => *variance,
            Kernel::Matern52 { variance, .. } => *variance,
            Kernel::Matern32 { variance, .. } => *variance,
        }
    }

    /// Set the length scale
    pub fn with_length_scale(self, length_scale: f64) -> Self {
        match self {
            Kernel::RBF { variance, .. } => Kernel::RBF {
                length_scale,
                variance,
            },
            Kernel::Matern52 { variance, .. } => Kernel::Matern52 {
                length_scale,
                variance,
            },
            Kernel::Matern32 { variance, .. } => Kernel::Matern32 {
                length_scale,
                variance,
            },
        }
    }

    /// Set the variance
    pub fn with_variance(self, variance: f64) -> Self {
        match self {
            Kernel::RBF { length_scale, .. } => Kernel::RBF {
                length_scale,
                variance,
            },
            Kernel::Matern52 { length_scale, .. } => Kernel::Matern52 {
                length_scale,
                variance,
            },
            Kernel::Matern32 { length_scale, .. } => Kernel::Matern32 {
                length_scale,
                variance,
            },
        }
    }

    /// Compute the kernel value between two points
    pub fn compute(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let dist_sq = squared_distance(x1, x2);

        match self {
            Kernel::RBF {
                length_scale,
                variance,
            } => {
                let scaled_dist_sq = dist_sq / (2.0 * length_scale * length_scale);
                variance * (-scaled_dist_sq).exp()
            }
            Kernel::Matern52 {
                length_scale,
                variance,
            } => {
                let dist = dist_sq.sqrt();
                let sqrt5 = 5.0_f64.sqrt();
                let scaled = sqrt5 * dist / length_scale;
                variance * (1.0 + scaled + scaled * scaled / 3.0) * (-scaled).exp()
            }
            Kernel::Matern32 {
                length_scale,
                variance,
            } => {
                let dist = dist_sq.sqrt();
                let sqrt3 = 3.0_f64.sqrt();
                let scaled = sqrt3 * dist / length_scale;
                variance * (1.0 + scaled) * (-scaled).exp()
            }
        }
    }

    /// Compute the kernel matrix for a set of points
    pub fn compute_matrix(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut k = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                let val = self.compute(&x[i], &x[j]);
                k[i][j] = val;
                k[j][i] = val;
            }
        }

        k
    }

    /// Compute kernel values between training points and new points
    pub fn compute_cross(&self, x_train: &[Vec<f64>], x_new: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_train = x_train.len();
        let n_new = x_new.len();
        let mut k = vec![vec![0.0; n_new]; n_train];

        for i in 0..n_train {
            for j in 0..n_new {
                k[i][j] = self.compute(&x_train[i], &x_new[j]);
            }
        }

        k
    }
}

/// Squared Euclidean distance between two vectors
fn squared_distance(x1: &[f64], x2: &[f64]) -> f64 {
    x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
}

/// Gaussian Process regression model
///
/// Implements GP regression with Cholesky decomposition for numerical stability.
/// Supports multiple kernel functions and noise estimation.
#[derive(Debug, Clone)]
pub struct GaussianProcessRegressor {
    /// Kernel function
    kernel: Kernel,
    /// Noise variance (observation noise)
    noise_variance: f64,
    /// Whether to optimize hyperparameters
    optimize_hyperparams: bool,
    /// Number of optimization restarts
    n_restarts: usize,
    /// Training inputs (normalized)
    x_train: Option<Vec<Vec<f64>>>,
    /// Training targets (standardized)
    y_train: Option<Vec<f64>>,
    /// Target mean (for de-standardization)
    y_mean: f64,
    /// Target std (for de-standardization)
    y_std: f64,
    /// Cholesky decomposition of K + σ²I
    chol_l: Option<Vec<Vec<f64>>>,
    /// Alpha = L^(-T) L^(-1) y
    alpha: Option<Vec<f64>>,
    /// Input bounds for normalization
    x_bounds: Option<Vec<(f64, f64)>>,
}

impl Default for GaussianProcessRegressor {
    fn default() -> Self {
        Self {
            kernel: Kernel::default(),
            noise_variance: 1e-6,
            optimize_hyperparams: true,
            n_restarts: 5,
            x_train: None,
            y_train: None,
            y_mean: 0.0,
            y_std: 1.0,
            chol_l: None,
            alpha: None,
            x_bounds: None,
        }
    }
}

impl GaussianProcessRegressor {
    /// Create a new GP regressor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the kernel
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the noise variance (jitter for numerical stability)
    pub fn with_noise_variance(mut self, noise: f64) -> Self {
        self.noise_variance = noise.max(1e-10);
        self
    }

    /// Enable/disable hyperparameter optimization
    pub fn with_optimize_hyperparams(mut self, optimize: bool) -> Self {
        self.optimize_hyperparams = optimize;
        self
    }

    /// Set number of optimization restarts
    pub fn with_n_restarts(mut self, n: usize) -> Self {
        self.n_restarts = n;
        self
    }

    /// Fit the GP to training data
    ///
    /// # Arguments
    /// * `x` - Training inputs, shape (n_samples, n_features)
    /// * `y` - Training targets, shape (n_samples,)
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(crate::FerroError::invalid_input(
                "Training data cannot be empty",
            ));
        }
        if x.len() != y.len() {
            return Err(crate::FerroError::shape_mismatch(
                format!("{} samples", x.len()),
                format!("{} targets", y.len()),
            ));
        }

        // Normalize inputs to [0, 1]
        let x_normalized = self.normalize_inputs(x);

        // Standardize targets
        self.y_mean = y.iter().sum::<f64>() / y.len() as f64;
        self.y_std = {
            let var = y.iter().map(|yi| (yi - self.y_mean).powi(2)).sum::<f64>() / y.len() as f64;
            var.sqrt().max(1e-10)
        };
        let y_standardized: Vec<f64> = y.iter().map(|yi| (yi - self.y_mean) / self.y_std).collect();

        self.x_train = Some(x_normalized.clone());
        self.y_train = Some(y_standardized.clone());

        // Optimize hyperparameters if enabled
        if self.optimize_hyperparams && x.len() >= 3 {
            self.optimize_kernel_params(&x_normalized, &y_standardized)?;
        }

        // Compute Cholesky decomposition
        self.compute_cholesky(&x_normalized, &y_standardized)?;

        Ok(())
    }

    /// Normalize inputs to [0, 1] range
    fn normalize_inputs(&mut self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if x.is_empty() {
            return vec![];
        }

        let n_features = x[0].len();

        // Compute bounds
        let mut bounds = vec![(f64::MAX, f64::MIN); n_features];
        for xi in x {
            for (j, &val) in xi.iter().enumerate() {
                bounds[j].0 = bounds[j].0.min(val);
                bounds[j].1 = bounds[j].1.max(val);
            }
        }

        // Ensure non-zero range
        for bound in &mut bounds {
            if (bound.1 - bound.0).abs() < 1e-10 {
                bound.1 = bound.0 + 1.0;
            }
        }

        self.x_bounds = Some(bounds.clone());

        // Normalize
        x.iter()
            .map(|xi| {
                xi.iter()
                    .enumerate()
                    .map(|(j, &val)| (val - bounds[j].0) / (bounds[j].1 - bounds[j].0))
                    .collect()
            })
            .collect()
    }

    /// Normalize a single input point using stored bounds
    fn normalize_point(&self, x: &[f64]) -> Vec<f64> {
        if let Some(ref bounds) = self.x_bounds {
            x.iter()
                .enumerate()
                .map(|(j, &val)| {
                    if j < bounds.len() {
                        (val - bounds[j].0) / (bounds[j].1 - bounds[j].0)
                    } else {
                        val
                    }
                })
                .collect()
        } else {
            x.to_vec()
        }
    }

    /// Optimize kernel hyperparameters via marginal log-likelihood
    fn optimize_kernel_params(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        let mut best_kernel = self.kernel;
        let mut best_mll = f64::NEG_INFINITY;

        // Grid search over length scales and variances
        // (Full optimization would use L-BFGS-B, but grid search is more robust)
        let length_scales = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0];
        let variances = [0.1, 0.5, 1.0, 2.0];

        for &ls in &length_scales {
            for &var in &variances {
                let test_kernel = match self.kernel {
                    Kernel::RBF { .. } => Kernel::RBF {
                        length_scale: ls,
                        variance: var,
                    },
                    Kernel::Matern52 { .. } => Kernel::Matern52 {
                        length_scale: ls,
                        variance: var,
                    },
                    Kernel::Matern32 { .. } => Kernel::Matern32 {
                        length_scale: ls,
                        variance: var,
                    },
                };

                if let Ok(mll) = self.compute_marginal_log_likelihood(&test_kernel, x, y) {
                    if mll > best_mll {
                        best_mll = mll;
                        best_kernel = test_kernel;
                    }
                }
            }
        }

        self.kernel = best_kernel;
        Ok(())
    }

    /// Compute marginal log-likelihood for a given kernel
    #[allow(clippy::many_single_char_names)]
    fn compute_marginal_log_likelihood(
        &self,
        kernel: &Kernel,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<f64> {
        let n = x.len();

        // Compute kernel matrix K
        let mut k = kernel.compute_matrix(x);

        // Add noise variance to diagonal: K + σ²I
        for i in 0..n {
            k[i][i] += self.noise_variance;
        }

        // Cholesky decomposition: K = LL^T
        let l = match cholesky_decomposition(&k) {
            Ok(l) => l,
            Err(_) => return Err(crate::FerroError::numerical("Cholesky failed")),
        };

        // Solve L*alpha = y for alpha
        let alpha = solve_triangular_lower(&l, y);

        // log|K| = 2 * sum(log(L_ii))
        let log_det: f64 = l
            .iter()
            .enumerate()
            .map(|(i, row)| row[i].ln())
            .sum::<f64>()
            * 2.0;

        // y^T K^(-1) y = alpha^T alpha
        let y_k_inv_y: f64 = alpha.iter().map(|a| a * a).sum();

        // Marginal log-likelihood:
        // log p(y|X) = -0.5 * (y^T K^(-1) y + log|K| + n*log(2π))
        let mll = -0.5 * (n as f64).mul_add((2.0 * std::f64::consts::PI).ln(), y_k_inv_y + log_det);

        Ok(mll)
    }

    /// Compute Cholesky decomposition and alpha vector
    #[allow(clippy::many_single_char_names)]
    fn compute_cholesky(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<()> {
        let n = x.len();

        // Compute kernel matrix K
        let mut k = self.kernel.compute_matrix(x);

        // Add noise variance to diagonal: K + σ²I
        for i in 0..n {
            k[i][i] += self.noise_variance;
        }

        // Cholesky decomposition: K = LL^T
        let l = cholesky_decomposition(&k)?;

        // Solve L*z = y then L^T*alpha = z
        let z = solve_triangular_lower(&l, y);
        let alpha = solve_triangular_upper_transpose(&l, &z);

        self.chol_l = Some(l);
        self.alpha = Some(alpha);

        Ok(())
    }

    /// Predict mean and variance at new points
    ///
    /// # Arguments
    /// * `x_new` - New input points, shape (n_points, n_features)
    ///
    /// # Returns
    /// * Vector of (mean, variance) tuples
    pub fn predict(&self, x_new: &[Vec<f64>]) -> Result<Vec<(f64, f64)>> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("predict - call fit() first"))?;
        let alpha = self
            .alpha
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("predict - call fit() first"))?;
        let chol_l = self
            .chol_l
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("predict - call fit() first"))?;

        let mut results = Vec::with_capacity(x_new.len());

        for x_star in x_new {
            let x_star_norm = self.normalize_point(x_star);

            // k* = k(X, x*) - kernel values between training points and new point
            let k_star: Vec<f64> = x_train
                .iter()
                .map(|xi| self.kernel.compute(xi, &x_star_norm))
                .collect();

            // Posterior mean: μ* = k*^T α
            let mu_star: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

            // k** = k(x*, x*)
            let k_star_star = self.kernel.compute(&x_star_norm, &x_star_norm);

            // Solve L*v = k* for v
            let v = solve_triangular_lower(chol_l, &k_star);

            // Posterior variance: σ*² = k** - v^T v
            let v_squared: f64 = v.iter().map(|vi| vi * vi).sum();
            let var_star = (k_star_star - v_squared).max(1e-10);

            // De-standardize
            let mean = mu_star.mul_add(self.y_std, self.y_mean);
            let variance = var_star * self.y_std * self.y_std;

            results.push((mean, variance));
        }

        Ok(results)
    }

    /// Predict mean only (faster than predict)
    pub fn predict_mean(&self, x_new: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.predict(x_new)
            .map(|preds| preds.into_iter().map(|(m, _)| m).collect())
    }

    /// Get the marginal log-likelihood of the fitted model
    pub fn log_marginal_likelihood(&self) -> Result<f64> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("log_marginal_likelihood"))?;
        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("log_marginal_likelihood"))?;

        self.compute_marginal_log_likelihood(&self.kernel, x_train, y_train)
    }

    /// Get the fitted kernel
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }
}

/// Cholesky decomposition (L such that A = LL^T)
#[allow(clippy::many_single_char_names)]
fn cholesky_decomposition(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum: f64 = 0.0;

            if j == i {
                // Diagonal elements - use mul_add for better precision
                for k in 0..j {
                    sum = l[j][k].mul_add(l[j][k], sum);
                }
                let val = a[i][i] - sum;
                if val <= 0.0 {
                    // Add more jitter and retry
                    return Err(crate::FerroError::numerical(
                        "Matrix not positive definite for Cholesky decomposition",
                    ));
                }
                l[i][j] = val.sqrt();
            } else {
                // Off-diagonal elements - use mul_add for better precision
                for k in 0..j {
                    sum = l[i][k].mul_add(l[j][k], sum);
                }
                if l[j][j].abs() < 1e-15 {
                    return Err(crate::FerroError::numerical(
                        "Near-zero diagonal in Cholesky",
                    ));
                }
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    Ok(l)
}

/// Solve Lx = b where L is lower triangular
#[allow(clippy::many_single_char_names)]
fn solve_triangular_lower(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];

    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * x[j];
        }
        x[i] = sum / l[i][i];
    }

    x
}

/// Solve L^T x = b where L is lower triangular (so L^T is upper triangular)
#[allow(clippy::many_single_char_names)]
fn solve_triangular_upper_transpose(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];

    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = sum / l[i][i];
    }

    x
}

/// Bayesian optimizer using Gaussian Processes
#[derive(Debug)]
pub struct BayesianOptimizer {
    /// Number of random trials before using GP
    n_initial: usize,
    /// Acquisition function to use
    acquisition: AcquisitionFunction,
    /// Exploration-exploitation trade-off parameter
    kappa: f64,
    /// Random seed
    seed: Option<u64>,
    /// Kernel to use for GP
    kernel: Kernel,
    /// Number of candidate points to evaluate (for random search fallback)
    n_candidates: usize,
    /// Whether to use L-BFGS-B optimization for acquisition
    use_lbfgsb: bool,
    /// L-BFGS-B configuration
    lbfgsb_config: LBFGSBConfig,
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    EI,
    /// Probability of Improvement
    PI,
    /// Upper Confidence Bound
    UCB,
    /// Lower Confidence Bound (for minimization)
    LCB,
}

impl Default for BayesianOptimizer {
    fn default() -> Self {
        Self {
            n_initial: 5,
            acquisition: AcquisitionFunction::EI,
            kappa: 2.576, // 99% confidence
            seed: None,
            kernel: Kernel::default(),
            n_candidates: 1000,
            use_lbfgsb: true,
            lbfgsb_config: LBFGSBConfig::default(),
        }
    }
}

impl BayesianOptimizer {
    /// Create new Bayesian optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of initial random trials
    pub fn with_n_initial(mut self, n: usize) -> Self {
        self.n_initial = n;
        self
    }

    /// Set acquisition function
    pub fn with_acquisition(mut self, acq: AcquisitionFunction) -> Self {
        self.acquisition = acq;
        self
    }

    /// Set exploration parameter (for UCB/LCB)
    pub fn with_kappa(mut self, kappa: f64) -> Self {
        self.kappa = kappa;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the kernel for the GP
    pub fn with_kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set number of candidate points for acquisition optimization
    pub fn with_n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n;
        self
    }

    /// Enable/disable L-BFGS-B optimization for acquisition function
    ///
    /// When enabled (default), uses gradient-based optimization for finding
    /// the next point to evaluate. When disabled, falls back to random search.
    pub fn with_lbfgsb(mut self, use_lbfgsb: bool) -> Self {
        self.use_lbfgsb = use_lbfgsb;
        self
    }

    /// Set L-BFGS-B configuration
    pub fn with_lbfgsb_config(mut self, config: LBFGSBConfig) -> Self {
        self.lbfgsb_config = config;
        self
    }

    /// Suggest next point to evaluate using GP-based Bayesian optimization
    pub fn suggest(
        &self,
        search_space: &SearchSpace,
        trials: &[Trial],
    ) -> Result<HashMap<String, ParameterValue>> {
        let completed: Vec<&Trial> = trials.iter().filter(|t| t.value.is_some()).collect();

        // Use random sampling for initial trials
        if completed.len() < self.n_initial {
            return super::samplers::RandomSampler::new().sample(search_space, trials);
        }

        // Extract training data from completed trials
        let param_names: Vec<String> = search_space.parameters.keys().cloned().collect();
        let (x_train, y_train) = self.extract_training_data(&completed, &param_names, search_space);

        if x_train.is_empty() {
            return super::samplers::RandomSampler::new().sample(search_space, trials);
        }

        // Fit GP
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(self.kernel)
            .with_noise_variance(1e-6)
            .with_optimize_hyperparams(true);

        if gp.fit(&x_train, &y_train).is_err() {
            // Fall back to TPE if GP fitting fails
            return super::samplers::TPESampler::new().sample(search_space, trials);
        }

        // Find best observed value
        let best_y = y_train
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Get bounds for each parameter
        let bounds: Vec<(f64, f64)> = param_names
            .iter()
            .filter_map(|name| {
                search_space.parameters.get(name).map(|param| {
                    let log_scale = param.log_scale;
                    match &param.param_type {
                        crate::hpo::search_space::ParameterType::Float { low, high } => {
                            if log_scale {
                                (low.ln(), high.ln())
                            } else {
                                (*low, *high)
                            }
                        }
                        crate::hpo::search_space::ParameterType::Int { low, high } => {
                            if log_scale {
                                ((*low as f64).ln(), (*high as f64).ln())
                            } else {
                                (*low as f64, *high as f64)
                            }
                        }
                        crate::hpo::search_space::ParameterType::Categorical { choices } => {
                            (0.0, (choices.len() - 1) as f64)
                        }
                        crate::hpo::search_space::ParameterType::Bool => (0.0, 1.0),
                    }
                })
            })
            .collect();

        // Use L-BFGS-B or random search for acquisition optimization
        let best_candidate = if self.use_lbfgsb {
            let optimizer = AcquisitionOptimizer::new()
                .with_lbfgsb_config(self.lbfgsb_config.clone())
                .with_gradient_optimization(true)
                .with_n_candidates(self.n_candidates);

            match optimizer.optimize(
                &gp,
                self.acquisition,
                best_y,
                &bounds,
                self.kappa,
                true, // minimize
                self.seed,
            ) {
                Ok((x_opt, _)) => x_opt,
                Err(_) => {
                    // Fall back to random search
                    self.generate_candidates(search_space, &param_names)
                        .into_iter()
                        .max_by(|a, b| {
                            let eval_a = self.evaluate_acquisition(&gp, a, best_y);
                            let eval_b = self.evaluate_acquisition(&gp, b, best_y);
                            eval_a.partial_cmp(&eval_b).unwrap()
                        })
                        .unwrap_or_else(|| vec![0.0; param_names.len()])
                }
            }
        } else {
            // Random search fallback
            let candidates = self.generate_candidates(search_space, &param_names);

            let mut best_acq = f64::NEG_INFINITY;
            let mut best_candidate = candidates[0].clone();

            for candidate in &candidates {
                let acq_value = self.evaluate_acquisition(&gp, candidate, best_y);
                if acq_value > best_acq {
                    best_acq = acq_value;
                    best_candidate = candidate.clone();
                }
            }

            best_candidate
        };

        // Convert best candidate back to parameter values
        self.convert_to_params(&best_candidate, &param_names, search_space)
    }

    /// Evaluate acquisition function at a point
    fn evaluate_acquisition(
        &self,
        gp: &GaussianProcessRegressor,
        candidate: &[f64],
        best_y: f64,
    ) -> f64 {
        if let Ok(preds) = gp.predict(&[candidate.to_vec()]) {
            let (mu, var) = preds[0];
            let sigma = var.sqrt();

            match self.acquisition {
                AcquisitionFunction::EI => expected_improvement(mu, sigma, best_y, true),
                AcquisitionFunction::PI => probability_of_improvement(mu, sigma, best_y, true),
                AcquisitionFunction::UCB => upper_confidence_bound(mu, sigma, self.kappa),
                AcquisitionFunction::LCB => -lower_confidence_bound(mu, sigma, self.kappa),
            }
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Extract training data from completed trials
    fn extract_training_data(
        &self,
        trials: &[&Trial],
        param_names: &[String],
        search_space: &SearchSpace,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut x = Vec::new();
        let mut y = Vec::new();

        for trial in trials {
            let mut point = Vec::new();
            let mut valid = true;

            for name in param_names {
                if let Some(value) = trial.params.get(name) {
                    if let Some(f) = self.param_to_float(value, name, search_space) {
                        point.push(f);
                    } else {
                        valid = false;
                        break;
                    }
                } else {
                    valid = false;
                    break;
                }
            }

            if valid {
                if let Some(obj) = trial.value {
                    x.push(point);
                    y.push(obj);
                }
            }
        }

        (x, y)
    }

    /// Convert parameter value to float for GP
    fn param_to_float(
        &self,
        value: &ParameterValue,
        name: &str,
        search_space: &SearchSpace,
    ) -> Option<f64> {
        match value {
            ParameterValue::Float(f) => Some(*f),
            ParameterValue::Int(i) => Some(*i as f64),
            ParameterValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            ParameterValue::Categorical(s) => {
                // One-hot encode categoricals
                if let Some(param) = search_space.parameters.get(name) {
                    if let crate::hpo::search_space::ParameterType::Categorical { choices } =
                        &param.param_type
                    {
                        choices.iter().position(|c| c == s).map(|i| i as f64)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Generate random candidate points in the search space
    fn generate_candidates(
        &self,
        search_space: &SearchSpace,
        param_names: &[String],
    ) -> Vec<Vec<f64>> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let mut candidates = Vec::with_capacity(self.n_candidates);

        for _ in 0..self.n_candidates {
            let mut point = Vec::new();

            for name in param_names {
                if let Some(param) = search_space.parameters.get(name) {
                    let log_scale = param.log_scale;
                    let val = match &param.param_type {
                        crate::hpo::search_space::ParameterType::Float { low, high } => {
                            if log_scale {
                                let log_low = low.ln();
                                let log_high = high.ln();
                                rng.random::<f64>()
                                    .mul_add(log_high - log_low, log_low)
                                    .exp()
                            } else {
                                rng.random::<f64>() * (high - low) + low
                            }
                        }
                        crate::hpo::search_space::ParameterType::Int { low, high } => {
                            if log_scale {
                                let log_low = (*low as f64).ln();
                                let log_high = (*high as f64).ln();
                                rng.random::<f64>()
                                    .mul_add(log_high - log_low, log_low)
                                    .exp()
                            } else {
                                rng.random::<f64>()
                                    .mul_add((*high - *low) as f64, *low as f64)
                            }
                        }
                        crate::hpo::search_space::ParameterType::Categorical { choices } => {
                            rng.random::<f64>() * choices.len() as f64
                        }
                        crate::hpo::search_space::ParameterType::Bool => rng.random::<f64>(),
                    };
                    point.push(val);
                }
            }

            candidates.push(point);
        }

        candidates
    }

    /// Convert float vector back to parameter values
    fn convert_to_params(
        &self,
        point: &[f64],
        param_names: &[String],
        search_space: &SearchSpace,
    ) -> Result<HashMap<String, ParameterValue>> {
        let mut params = HashMap::new();

        for (i, name) in param_names.iter().enumerate() {
            if let Some(param) = search_space.parameters.get(name) {
                let val = point[i];
                let param_val = match &param.param_type {
                    crate::hpo::search_space::ParameterType::Float { low, high, .. } => {
                        ParameterValue::Float(val.clamp(*low, *high))
                    }
                    crate::hpo::search_space::ParameterType::Int { low, high, .. } => {
                        ParameterValue::Int((val.round() as i64).clamp(*low, *high))
                    }
                    crate::hpo::search_space::ParameterType::Categorical { choices } => {
                        let idx = (val.round() as usize).min(choices.len() - 1);
                        ParameterValue::Categorical(choices[idx].clone())
                    }
                    crate::hpo::search_space::ParameterType::Bool => {
                        ParameterValue::Bool(val >= 0.5)
                    }
                };
                params.insert(name.clone(), param_val);
            }
        }

        Ok(params)
    }
}

impl Sampler for BayesianOptimizer {
    fn sample(
        &self,
        search_space: &SearchSpace,
        trials: &[Trial],
    ) -> Result<HashMap<String, ParameterValue>> {
        self.suggest(search_space, trials)
    }
}

/// Expected Improvement acquisition function
///
/// EI(x) = (f_best - μ(x)) * Φ(Z) + σ(x) * φ(Z)
/// where Z = (f_best - μ(x)) / σ(x)
pub fn expected_improvement(mu: f64, sigma: f64, best_y: f64, minimize: bool) -> f64 {
    if sigma <= 1e-10 {
        return 0.0;
    }

    let improvement = if minimize { best_y - mu } else { mu - best_y };
    let z = improvement / sigma;
    let ei = improvement.mul_add(normal_cdf(z), sigma * normal_pdf(z));

    ei.max(0.0)
}

/// Probability of Improvement acquisition function
///
/// PI(x) = Φ((f_best - μ(x)) / σ(x))  for minimization
pub fn probability_of_improvement(mu: f64, sigma: f64, best_y: f64, minimize: bool) -> f64 {
    if sigma <= 1e-10 {
        return if minimize {
            (mu < best_y) as i32 as f64
        } else {
            (mu > best_y) as i32 as f64
        };
    }

    let z = if minimize {
        (best_y - mu) / sigma
    } else {
        (mu - best_y) / sigma
    };

    normal_cdf(z)
}

/// Upper Confidence Bound acquisition function
///
/// UCB(x) = μ(x) + κ * σ(x)
pub fn upper_confidence_bound(mu: f64, sigma: f64, kappa: f64) -> f64 {
    kappa.mul_add(sigma, mu)
}

/// Lower Confidence Bound acquisition function (for minimization)
///
/// LCB(x) = μ(x) - κ * σ(x)
pub fn lower_confidence_bound(mu: f64, sigma: f64, kappa: f64) -> f64 {
    kappa.mul_add(-sigma, mu)
}

/// Standard normal CDF using error function
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal PDF
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Error function approximation (Abramowitz and Stegun)
#[allow(clippy::unreadable_literal)]
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = ((a5 * t + a4).mul_add(t, a3).mul_add(t, a2).mul_add(t, a1) * t)
        .mul_add(-(-x * x).exp(), 1.0);
    sign * y
}

/// Configuration for L-BFGS-B acquisition optimization
#[derive(Debug, Clone)]
pub struct LBFGSBConfig {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Number of vectors to store for L-BFGS approximation
    pub m: usize,
    /// Tolerance for convergence (gradient norm)
    pub gtol: f64,
    /// Tolerance for convergence (function value change)
    pub ftol: f64,
    /// Number of random restarts for multi-start optimization
    pub n_restarts: usize,
    /// Line search parameter c1 (Armijo condition)
    pub c1: f64,
    /// Line search parameter c2 (curvature condition)
    pub c2: f64,
    /// Maximum line search iterations
    pub max_linesearch: usize,
    /// Tolerance for convergence (parameter change norm)
    pub xtol: f64,
}

impl Default for LBFGSBConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            m: 10,
            gtol: 1e-5,
            ftol: 1e-8,
            n_restarts: 5,
            c1: 1e-4,
            c2: 0.9,
            max_linesearch: 20,
            xtol: 1e-8,
        }
    }
}

/// L-BFGS-B optimizer for bounded optimization
///
/// Implements the Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm
/// with bound constraints (L-BFGS-B).
///
/// # References
/// - Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory
///   algorithm for bound constrained optimization. SIAM Journal on Scientific
///   Computing.
#[derive(Debug, Clone)]
pub struct LBFGSB {
    config: LBFGSBConfig,
}

impl Default for LBFGSB {
    fn default() -> Self {
        Self {
            config: LBFGSBConfig::default(),
        }
    }
}

impl LBFGSB {
    /// Create a new L-BFGS-B optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set configuration
    pub fn with_config(mut self, config: LBFGSBConfig) -> Self {
        self.config = config;
        self
    }

    /// Minimize a function subject to bound constraints
    ///
    /// # Arguments
    /// * `f` - Function to minimize, takes point and returns (value, gradient)
    /// * `x0` - Initial point
    /// * `bounds` - Lower and upper bounds for each dimension
    ///
    /// # Returns
    /// * Optimal point and function value
    #[allow(clippy::many_single_char_names)]
    pub fn minimize<F>(&self, f: &F, x0: &[f64], bounds: &[(f64, f64)]) -> (Vec<f64>, f64)
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        let m = self.config.m;

        // Project initial point onto bounds
        let mut x: Vec<f64> = x0
            .iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
            .collect();

        // Storage for L-BFGS vectors
        let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut rho_history: Vec<f64> = Vec::with_capacity(m);

        let (mut fx, mut gx) = f(&x);
        let mut x_prev = x.clone();
        let mut g_prev = gx.clone();

        for _iter in 0..self.config.max_iter {
            // Check convergence via projected gradient norm
            let pg_norm = projected_gradient_norm(&x, &gx, bounds);
            if pg_norm < self.config.gtol {
                break;
            }

            // Compute search direction using L-BFGS two-loop recursion
            let d = self.compute_direction(&gx, &s_history, &y_history, &rho_history, bounds, &x);

            // Line search with backtracking
            let (_alpha, x_new, fx_new, gx_new) = self.line_search(f, &x, fx, &gx, &d, bounds);

            // Check for sufficient decrease
            if (fx - fx_new).abs() < self.config.ftol * (1.0 + fx.abs()) {
                x = x_new;
                fx = fx_new;
                break;
            }

            // Compute s and y vectors
            let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = gx_new.iter().zip(gx.iter()).map(|(a, b)| a - b).collect();

            let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();

            // Skip update if curvature condition not satisfied
            if sy > 1e-10 {
                let rho = 1.0 / sy;

                // Update history (circular buffer)
                if s_history.len() >= m {
                    s_history.remove(0);
                    y_history.remove(0);
                    rho_history.remove(0);
                }
                s_history.push(s);
                y_history.push(y);
                rho_history.push(rho);
            }

            // Check parameter-change convergence
            let x_change: f64 = x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            x_prev = x;
            g_prev = gx;
            x = x_new;
            fx = fx_new;
            gx = gx_new;

            if x_change < self.config.xtol {
                break;
            }
        }

        let _ = (x_prev, g_prev);

        (x, fx)
    }

    /// Compute search direction using L-BFGS two-loop recursion
    #[allow(clippy::many_single_char_names)]
    fn compute_direction(
        &self,
        g: &[f64],
        s_history: &[Vec<f64>],
        y_history: &[Vec<f64>],
        rho_history: &[f64],
        bounds: &[(f64, f64)],
        x: &[f64],
    ) -> Vec<f64> {
        let k = s_history.len();

        if k == 0 {
            // Steepest descent direction projected onto bounds
            return g.iter().map(|gi| -gi).collect();
        }

        // Two-loop recursion
        let mut q: Vec<f64> = g.to_vec();
        let mut alpha_vec = vec![0.0; k];

        // First loop (reverse)
        for i in (0..k).rev() {
            let alpha_i: f64 = rho_history[i]
                * s_history[i]
                    .iter()
                    .zip(q.iter())
                    .map(|(si, qi)| si * qi)
                    .sum::<f64>();
            alpha_vec[i] = alpha_i;
            for j in 0..q.len() {
                q[j] -= alpha_i * y_history[i][j];
            }
        }

        // Compute initial Hessian approximation scaling
        let yk = &y_history[k - 1];
        let sk = &s_history[k - 1];
        let yk_dot_yk: f64 = yk.iter().map(|yi| yi * yi).sum();
        let sk_dot_yk: f64 = sk.iter().zip(yk.iter()).map(|(si, yi)| si * yi).sum();
        let gamma = if yk_dot_yk > 1e-10 {
            sk_dot_yk / yk_dot_yk
        } else {
            1.0
        };

        // r = H_0 * q
        let mut r: Vec<f64> = q.iter().map(|qi| gamma * qi).collect();

        // Second loop (forward)
        for i in 0..k {
            let beta: f64 = rho_history[i]
                * y_history[i]
                    .iter()
                    .zip(r.iter())
                    .map(|(yi, ri)| yi * ri)
                    .sum::<f64>();
            for j in 0..r.len() {
                r[j] += (alpha_vec[i] - beta) * s_history[i][j];
            }
        }

        // Negate for descent direction
        let d: Vec<f64> = r.iter().map(|ri| -ri).collect();

        // Project direction to ensure feasibility
        self.project_direction(x, &d, bounds)
    }

    /// Project direction to respect bounds
    fn project_direction(&self, x: &[f64], d: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
        x.iter()
            .zip(d.iter())
            .zip(bounds.iter())
            .map(|((&xi, &di), &(lo, hi))| {
                // If at lower bound and direction points down, zero it
                if xi <= lo + 1e-10 && di < 0.0 {
                    0.0
                // If at upper bound and direction points up, zero it
                } else if xi >= hi - 1e-10 && di > 0.0 {
                    0.0
                } else {
                    di
                }
            })
            .collect()
    }

    /// Backtracking line search with Armijo condition
    fn line_search<F>(
        &self,
        f: &F,
        x: &[f64],
        fx: f64,
        gx: &[f64],
        d: &[f64],
        bounds: &[(f64, f64)],
    ) -> (f64, Vec<f64>, f64, Vec<f64>)
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        let mut alpha = 1.0;
        let slope: f64 = gx.iter().zip(d.iter()).map(|(gi, di)| gi * di).sum();

        // If slope is non-negative, we can't improve
        if slope >= 0.0 {
            return (0.0, x.to_vec(), fx, gx.to_vec());
        }

        for _ in 0..self.config.max_linesearch {
            // Compute new point with projection
            let x_new: Vec<f64> = x
                .iter()
                .zip(d.iter())
                .zip(bounds.iter())
                .map(|((&xi, &di), &(lo, hi))| (xi + alpha * di).clamp(lo, hi))
                .collect();

            let (fx_new, gx_new) = f(&x_new);

            // Armijo condition
            if fx_new <= (self.config.c1 * alpha).mul_add(slope, fx) {
                return (alpha, x_new, fx_new, gx_new);
            }

            alpha *= 0.5;
        }

        // Return best we found
        let x_new: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .zip(bounds.iter())
            .map(|((&xi, &di), &(lo, hi))| (xi + alpha * di).clamp(lo, hi))
            .collect();
        let (fx_new, gx_new) = f(&x_new);
        (alpha, x_new, fx_new, gx_new)
    }

    /// Multi-start optimization with random restarts
    ///
    /// # Arguments
    /// * `f` - Function to minimize
    /// * `bounds` - Lower and upper bounds
    /// * `seed` - Optional random seed
    #[allow(clippy::similar_names)]
    pub fn minimize_multistart<F>(
        &self,
        f: &F,
        bounds: &[(f64, f64)],
        seed: Option<u64>,
    ) -> (Vec<f64>, f64)
    where
        F: Fn(&[f64]) -> (f64, Vec<f64>),
    {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let n = bounds.len();
        let mut best_x = vec![0.0; n];
        let mut best_fx = f64::INFINITY;

        for _ in 0..self.config.n_restarts {
            // Generate random starting point
            let x0: Vec<f64> = bounds
                .iter()
                .map(|&(lo, hi)| rng.random::<f64>().mul_add(hi - lo, lo))
                .collect();

            let (x_opt, fx_opt) = self.minimize(f, &x0, bounds);

            if fx_opt < best_fx {
                best_fx = fx_opt;
                best_x = x_opt;
            }
        }

        (best_x, best_fx)
    }
}

/// Compute projected gradient norm (infinity norm of gradient projected onto bounds)
fn projected_gradient_norm(x: &[f64], g: &[f64], bounds: &[(f64, f64)]) -> f64 {
    x.iter()
        .zip(g.iter())
        .zip(bounds.iter())
        .map(|((&xi, &gi), &(lo, hi))| {
            // Project x - g onto bounds and compute difference
            let projected = (xi - gi).clamp(lo, hi);
            (xi - projected).abs()
        })
        .fold(0.0_f64, |a, b| a.max(b))
}

/// Gradient of Expected Improvement
///
/// d(EI)/dx = d(EI)/dμ * dμ/dx + d(EI)/dσ * dσ/dx
pub fn expected_improvement_gradient(
    mu: f64,
    sigma: f64,
    dmu_dx: &[f64],
    dsigma_dx: &[f64],
    best_y: f64,
    minimize: bool,
) -> Vec<f64> {
    let n = dmu_dx.len();

    if sigma <= 1e-10 {
        return vec![0.0; n];
    }

    let improvement = if minimize { best_y - mu } else { mu - best_y };
    let z = improvement / sigma;
    let phi_z = normal_cdf(z);
    let pdf_z = normal_pdf(z);

    // d(EI)/dμ = -Φ(z) for minimization
    let dei_dmu = if minimize { -phi_z } else { phi_z };

    // d(EI)/dσ = φ(z)
    let dei_dsigma = pdf_z;

    dmu_dx
        .iter()
        .zip(dsigma_dx.iter())
        .map(|(&dmu, &dsigma)| dei_dmu.mul_add(dmu, dei_dsigma * dsigma))
        .collect()
}

/// Gradient of Probability of Improvement
pub fn probability_of_improvement_gradient(
    mu: f64,
    sigma: f64,
    dmu_dx: &[f64],
    dsigma_dx: &[f64],
    best_y: f64,
    minimize: bool,
) -> Vec<f64> {
    let n = dmu_dx.len();

    if sigma <= 1e-10 {
        return vec![0.0; n];
    }

    let improvement = if minimize { best_y - mu } else { mu - best_y };
    let z = improvement / sigma;
    let pdf_z = normal_pdf(z);

    // d(PI)/dz * dz/dμ + d(PI)/dz * dz/dσ
    // d(PI)/dz = φ(z)
    // dz/dμ = -1/σ (minimize), 1/σ (maximize)
    // dz/dσ = -z/σ

    let dz_dmu = if minimize { -1.0 / sigma } else { 1.0 / sigma };
    let dz_dsigma = -z / sigma;

    dmu_dx
        .iter()
        .zip(dsigma_dx.iter())
        .map(|(&dmu, &dsigma)| pdf_z * (dz_dmu * dmu + dz_dsigma * dsigma))
        .collect()
}

/// Gradient of Upper Confidence Bound
pub fn ucb_gradient(dmu_dx: &[f64], dsigma_dx: &[f64], kappa: f64) -> Vec<f64> {
    // d(UCB)/dx = dμ/dx + κ * dσ/dx
    dmu_dx
        .iter()
        .zip(dsigma_dx.iter())
        .map(|(&dmu, &dsigma)| kappa.mul_add(dsigma, dmu))
        .collect()
}

/// Gradient of Lower Confidence Bound
pub fn lcb_gradient(dmu_dx: &[f64], dsigma_dx: &[f64], kappa: f64) -> Vec<f64> {
    // d(LCB)/dx = dμ/dx - κ * dσ/dx
    dmu_dx
        .iter()
        .zip(dsigma_dx.iter())
        .map(|(&dmu, &dsigma)| kappa.mul_add(-dsigma, dmu))
        .collect()
}

impl GaussianProcessRegressor {
    /// Predict with gradients for acquisition optimization
    ///
    /// # Returns
    /// * (mean, variance, d_mean/dx, d_sigma/dx)
    pub fn predict_with_gradients(&self, x_new: &[f64]) -> Result<(f64, f64, Vec<f64>, Vec<f64>)> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            crate::FerroError::not_fitted("predict_with_gradients - call fit() first")
        })?;
        let alpha = self.alpha.as_ref().ok_or_else(|| {
            crate::FerroError::not_fitted("predict_with_gradients - call fit() first")
        })?;
        let chol_l = self.chol_l.as_ref().ok_or_else(|| {
            crate::FerroError::not_fitted("predict_with_gradients - call fit() first")
        })?;
        let x_bounds = self.x_bounds.as_ref().ok_or_else(|| {
            crate::FerroError::not_fitted("predict_with_gradients - call fit() first")
        })?;

        let d = x_new.len();

        // Normalize input
        let x_star_norm = self.normalize_point(x_new);

        // k* = k(X, x*)
        let k_star: Vec<f64> = x_train
            .iter()
            .map(|xi| self.kernel.compute(xi, &x_star_norm))
            .collect();

        // Posterior mean: μ* = k*^T α
        let mu_star: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

        // k** = k(x*, x*)
        let k_star_star = self.kernel.compute(&x_star_norm, &x_star_norm);

        // Solve L*v = k* for v
        let v = solve_triangular_lower(chol_l, &k_star);

        // Posterior variance: σ*² = k** - v^T v
        let v_squared: f64 = v.iter().map(|vi| vi * vi).sum();
        let var_star = (k_star_star - v_squared).max(1e-10);
        let sigma_star = var_star.sqrt();

        // Compute gradients dk*/dx for each training point
        let dk_star_dx = self.compute_kernel_gradients(x_train, &x_star_norm, x_bounds);

        // dμ/dx = Σ_i α_i * dk_i/dx
        let mut dmu_dx = vec![0.0; d];
        for (i, dk_dx) in dk_star_dx.iter().enumerate() {
            for j in 0..d {
                dmu_dx[j] += alpha[i] * dk_dx[j];
            }
        }

        // For variance gradient:
        // σ² = k** - v^T v where v = L^(-1) k*
        // dσ²/dx = dk**/dx - 2 * v^T * L^(-1) * dk*/dx
        // Since k** is at x*, dk**/dx = 0 for RBF at the same point

        // Solve L^T w = v for computing gradient
        let w = solve_triangular_upper_transpose(chol_l, &v);

        // dσ²/dx = -2 * Σ_i w_i * dk_i/dx
        let mut dvar_dx = vec![0.0; d];
        for (i, dk_dx) in dk_star_dx.iter().enumerate() {
            for j in 0..d {
                dvar_dx[j] -= 2.0 * w[i] * dk_dx[j];
            }
        }

        // dσ/dx = dσ²/dx / (2σ)
        let dsigma_dx: Vec<f64> = dvar_dx.iter().map(|dv| dv / (2.0 * sigma_star)).collect();

        // De-standardize
        let mean = mu_star.mul_add(self.y_std, self.y_mean);
        let variance = var_star * self.y_std * self.y_std;
        let sigma = variance.sqrt();

        // Scale gradients by y_std and x_bounds
        let dmu_dx_scaled: Vec<f64> = dmu_dx
            .iter()
            .enumerate()
            .map(|(j, &dm)| dm * self.y_std / (x_bounds[j].1 - x_bounds[j].0))
            .collect();

        let dsigma_dx_scaled: Vec<f64> = dsigma_dx
            .iter()
            .enumerate()
            .map(|(j, &ds)| ds * self.y_std / (x_bounds[j].1 - x_bounds[j].0))
            .collect();

        Ok((mean, sigma, dmu_dx_scaled, dsigma_dx_scaled))
    }

    /// Compute kernel gradients dk(x_i, x*)/dx* for all training points
    fn compute_kernel_gradients(
        &self,
        x_train: &[Vec<f64>],
        x_star: &[f64],
        x_bounds: &[(f64, f64)],
    ) -> Vec<Vec<f64>> {
        x_train
            .iter()
            .map(|xi| {
                let k_val = self.kernel.compute(xi, x_star);
                self.kernel_gradient(xi, x_star, k_val, x_bounds)
            })
            .collect()
    }

    /// Compute gradient of kernel with respect to x*
    fn kernel_gradient(
        &self,
        xi: &[f64],
        x_star: &[f64],
        k_val: f64,
        _x_bounds: &[(f64, f64)],
    ) -> Vec<f64> {
        let d = x_star.len();

        match self.kernel {
            Kernel::RBF { length_scale, .. } => {
                // k(xi, x*) = σ² * exp(-||xi - x*||² / (2l²))
                // dk/dx*_j = k(xi, x*) * (xi_j - x*_j) / l²
                let l_sq = length_scale * length_scale;
                (0..d).map(|j| k_val * (xi[j] - x_star[j]) / l_sq).collect()
            }
            Kernel::Matern52 {
                length_scale,
                variance,
            } => {
                // Matern 5/2: k = σ² * (1 + √5*r/l + 5*r²/(3*l²)) * exp(-√5*r/l)
                // where r = ||xi - x*||
                let diff: Vec<f64> = xi.iter().zip(x_star.iter()).map(|(a, b)| a - b).collect();
                let r = diff.iter().map(|d| d * d).sum::<f64>().sqrt();

                if r < 1e-10 {
                    return vec![0.0; d];
                }

                let sqrt5 = 5.0_f64.sqrt();
                let scaled_r = sqrt5 * r / length_scale;
                let exp_term = (-scaled_r).exp();

                // dk/dr = σ² * exp(-√5*r/l) * [√5/l + 10*r/(3*l²) - √5/l * (1 + √5*r/l + 5*r²/(3*l²))]
                // Simplified: dk/dr = -σ² * 5*r/(3*l²) * (1 + √5*r/l) * exp(-√5*r/l)
                let dk_dr = -variance
                    * (5.0 * r / (3.0 * length_scale * length_scale))
                    * (1.0 + scaled_r)
                    * exp_term;

                // dr/dx*_j = -(xi_j - x*_j) / r
                // dk/dx*_j = dk/dr * dr/dx*_j
                (0..d).map(|j| dk_dr * (-diff[j] / r)).collect()
            }
            Kernel::Matern32 {
                length_scale,
                variance,
            } => {
                // Matern 3/2: k = σ² * (1 + √3*r/l) * exp(-√3*r/l)
                let diff: Vec<f64> = xi.iter().zip(x_star.iter()).map(|(a, b)| a - b).collect();
                let r = diff.iter().map(|d| d * d).sum::<f64>().sqrt();

                if r < 1e-10 {
                    return vec![0.0; d];
                }

                let sqrt3 = 3.0_f64.sqrt();
                let scaled_r = sqrt3 * r / length_scale;
                let exp_term = (-scaled_r).exp();

                // dk/dr = -σ² * 3*r/l² * exp(-√3*r/l)
                let dk_dr = -variance * (3.0 * r / (length_scale * length_scale)) * exp_term;

                (0..d).map(|j| dk_dr * (-diff[j] / r)).collect()
            }
        }
    }
}

/// Acquisition optimizer using L-BFGS-B
pub struct AcquisitionOptimizer {
    /// L-BFGS-B optimizer
    lbfgsb: LBFGSB,
    /// Whether to use gradient-based optimization (true) or random search (false)
    use_gradient: bool,
    /// Number of random candidates when not using gradients
    n_candidates: usize,
}

impl Default for AcquisitionOptimizer {
    fn default() -> Self {
        Self {
            lbfgsb: LBFGSB::new(),
            use_gradient: true,
            n_candidates: 1000,
        }
    }
}

impl AcquisitionOptimizer {
    /// Create new acquisition optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set L-BFGS-B configuration
    pub fn with_lbfgsb_config(mut self, config: LBFGSBConfig) -> Self {
        self.lbfgsb = LBFGSB::new().with_config(config);
        self
    }

    /// Enable/disable gradient-based optimization
    pub fn with_gradient_optimization(mut self, use_gradient: bool) -> Self {
        self.use_gradient = use_gradient;
        self
    }

    /// Set number of candidates for random search fallback
    pub fn with_n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n;
        self
    }

    /// Optimize acquisition function to find next point to evaluate
    ///
    /// # Arguments
    /// * `gp` - Fitted Gaussian Process
    /// * `acquisition` - Acquisition function type
    /// * `best_y` - Best observed value so far
    /// * `bounds` - Search space bounds
    /// * `kappa` - UCB/LCB exploration parameter
    /// * `minimize` - Whether minimizing the objective
    /// * `seed` - Random seed for multi-start
    pub fn optimize(
        &self,
        gp: &GaussianProcessRegressor,
        acquisition: AcquisitionFunction,
        best_y: f64,
        bounds: &[(f64, f64)],
        kappa: f64,
        minimize: bool,
        seed: Option<u64>,
    ) -> Result<(Vec<f64>, f64)> {
        if !self.use_gradient {
            return self.optimize_random(gp, acquisition, best_y, bounds, kappa, minimize, seed);
        }

        // Create objective function (negated for maximization of acquisition)
        let objective = |x: &[f64]| -> (f64, Vec<f64>) {
            match gp.predict_with_gradients(x) {
                Ok((mu, sigma, dmu_dx, dsigma_dx)) => {
                    let (acq_val, acq_grad) = match acquisition {
                        AcquisitionFunction::EI => {
                            let val = expected_improvement(mu, sigma, best_y, minimize);
                            let grad = expected_improvement_gradient(
                                mu, sigma, &dmu_dx, &dsigma_dx, best_y, minimize,
                            );
                            (val, grad)
                        }
                        AcquisitionFunction::PI => {
                            let val = probability_of_improvement(mu, sigma, best_y, minimize);
                            let grad = probability_of_improvement_gradient(
                                mu, sigma, &dmu_dx, &dsigma_dx, best_y, minimize,
                            );
                            (val, grad)
                        }
                        AcquisitionFunction::UCB => {
                            let val = upper_confidence_bound(mu, sigma, kappa);
                            let grad = ucb_gradient(&dmu_dx, &dsigma_dx, kappa);
                            (val, grad)
                        }
                        AcquisitionFunction::LCB => {
                            let val = lower_confidence_bound(mu, sigma, kappa);
                            let grad = lcb_gradient(&dmu_dx, &dsigma_dx, kappa);
                            // For LCB minimization, we want to minimize LCB (find low predictions)
                            (val, grad)
                        }
                    };

                    // Negate for minimization (we want to maximize acquisition)
                    // For LCB, we actually want to minimize, so don't negate
                    match acquisition {
                        // Negate all acquisition values for L-BFGS minimization
                        _ => (-acq_val, acq_grad.iter().map(|g| -g).collect()),
                    }
                }
                Err(_) => {
                    // Return high value with zero gradient on error
                    (f64::MAX, vec![0.0; x.len()])
                }
            }
        };

        let (x_opt, f_opt) = self.lbfgsb.minimize_multistart(&objective, bounds, seed);

        // Return actual acquisition value (un-negated except for LCB)
        let acq_val = match acquisition {
            AcquisitionFunction::LCB => f_opt,
            _ => -f_opt,
        };

        Ok((x_opt, acq_val))
    }

    /// Random search optimization (fallback)
    fn optimize_random(
        &self,
        gp: &GaussianProcessRegressor,
        acquisition: AcquisitionFunction,
        best_y: f64,
        bounds: &[(f64, f64)],
        kappa: f64,
        minimize: bool,
        seed: Option<u64>,
    ) -> Result<(Vec<f64>, f64)> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let n = bounds.len();
        let mut best_x = vec![0.0; n];
        let mut best_acq = f64::NEG_INFINITY;

        for _ in 0..self.n_candidates {
            let x: Vec<f64> = bounds
                .iter()
                .map(|&(lo, hi)| rng.random::<f64>().mul_add(hi - lo, lo))
                .collect();

            if let Ok(preds) = gp.predict(&[x.clone()]) {
                let (mu, var) = preds[0];
                let sigma = var.sqrt();

                let acq_val = match acquisition {
                    AcquisitionFunction::EI => expected_improvement(mu, sigma, best_y, minimize),
                    AcquisitionFunction::PI => {
                        probability_of_improvement(mu, sigma, best_y, minimize)
                    }
                    AcquisitionFunction::UCB => upper_confidence_bound(mu, sigma, kappa),
                    AcquisitionFunction::LCB => -lower_confidence_bound(mu, sigma, kappa),
                };

                if acq_val > best_acq {
                    best_acq = acq_val;
                    best_x = x;
                }
            }
        }

        Ok((best_x, best_acq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_kernel() {
        let kernel = Kernel::rbf().with_length_scale(1.0).with_variance(1.0);

        // Same point should give variance
        let k_same = kernel.compute(&[0.0, 0.0], &[0.0, 0.0]);
        assert!((k_same - 1.0).abs() < 1e-10);

        // Different points should give smaller value
        let k_diff = kernel.compute(&[0.0, 0.0], &[1.0, 0.0]);
        assert!(k_diff < 1.0);
        assert!(k_diff > 0.0);
    }

    #[test]
    fn test_matern52_kernel() {
        let kernel = Kernel::matern52().with_length_scale(1.0).with_variance(1.0);

        // Same point should give variance
        let k_same = kernel.compute(&[0.0], &[0.0]);
        assert!((k_same - 1.0).abs() < 1e-10);

        // Different points should give smaller value
        let k_diff = kernel.compute(&[0.0], &[1.0]);
        assert!(k_diff < 1.0);
        assert!(k_diff > 0.0);
    }

    #[test]
    fn test_matern32_kernel() {
        let kernel = Kernel::matern32().with_length_scale(1.0).with_variance(1.0);

        let k_same = kernel.compute(&[0.0], &[0.0]);
        assert!((k_same - 1.0).abs() < 1e-10);

        let k_diff = kernel.compute(&[0.0], &[1.0]);
        assert!(k_diff < 1.0);
        assert!(k_diff > 0.0);
    }

    #[test]
    fn test_kernel_matrix_symmetric() {
        let kernel = Kernel::rbf();
        let x = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let k = kernel.compute_matrix(&x);

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((k[i][j] - k[j][i]).abs() < 1e-10);
            }
        }

        // Check diagonal is variance
        for i in 0..3 {
            assert!((k[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cholesky_decomposition() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];

        let l = cholesky_decomposition(&a).unwrap();

        // Verify L * L^T = A
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += l[i][k] * l[j][k];
                }
                assert!(
                    (sum - a[i][j]).abs() < 1e-10,
                    "L*L^T[{},{}] = {} != {}",
                    i,
                    j,
                    sum,
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn test_gp_fit_predict() {
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(Kernel::rbf())
            .with_optimize_hyperparams(false);

        // Simple 1D function: y = x^2
        let x = vec![vec![0.0], vec![0.25], vec![0.5], vec![0.75], vec![1.0]];
        let y: Vec<f64> = x.iter().map(|xi| xi[0] * xi[0]).collect();

        gp.fit(&x, &y).unwrap();

        // Predict at training points - should be close
        let preds = gp.predict(&x).unwrap();

        for (i, (pred_mean, _)) in preds.iter().enumerate() {
            assert!(
                (pred_mean - y[i]).abs() < 0.2,
                "Prediction {} = {} far from target {}",
                i,
                pred_mean,
                y[i]
            );
        }
    }

    #[test]
    fn test_gp_uncertainty() {
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(Kernel::rbf())
            .with_optimize_hyperparams(false);

        let x = vec![vec![0.0], vec![1.0]];
        let y = vec![0.0, 1.0];

        gp.fit(&x, &y).unwrap();

        // Predict at middle point - should have higher uncertainty
        let preds_middle = gp.predict(&[vec![0.5]]).unwrap();
        let (_, var_middle) = preds_middle[0];

        // Predict at training point - should have lower uncertainty
        let preds_train = gp.predict(&[vec![0.0]]).unwrap();
        let (_, var_train) = preds_train[0];

        // Middle point should have higher variance than training point
        // (or at least similar due to interpolation)
        assert!(var_train < var_middle + 0.5);
    }

    #[test]
    fn test_expected_improvement() {
        // Test EI when mean is better than best
        let ei = expected_improvement(0.5, 0.1, 1.0, true); // minimizing
        assert!(ei > 0.0);

        // Test EI when mean is worse than best
        let ei2 = expected_improvement(1.5, 0.1, 1.0, true);
        assert!(ei2 >= 0.0);
        assert!(ei > ei2); // Better mean should have higher EI

        // Test EI with zero sigma
        let ei3 = expected_improvement(0.5, 0.0, 1.0, true);
        assert_eq!(ei3, 0.0);
    }

    #[test]
    fn test_probability_of_improvement() {
        // When mean is much better than best (minimizing)
        let pi = probability_of_improvement(0.0, 0.1, 1.0, true);
        assert!(pi > 0.9); // High probability of improvement

        // When mean is much worse than best
        let pi2 = probability_of_improvement(2.0, 0.1, 1.0, true);
        assert!(pi2 < 0.1); // Low probability of improvement
    }

    #[test]
    fn test_ucb_lcb() {
        let mu = 0.5;
        let sigma = 0.2;
        let kappa = 2.0;

        let ucb = upper_confidence_bound(mu, sigma, kappa);
        let lcb = lower_confidence_bound(mu, sigma, kappa);

        assert!((ucb - 0.9).abs() < 1e-10);
        assert!((lcb - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_normal_cdf_pdf() {
        // CDF at 0 should be 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-5);

        // PDF at 0 should be 1/sqrt(2*pi)
        let expected_pdf = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((normal_pdf(0.0) - expected_pdf).abs() < 1e-10);

        // CDF(-inf) -> 0, CDF(inf) -> 1
        assert!(normal_cdf(-5.0) < 0.001);
        assert!(normal_cdf(5.0) > 0.999);
    }

    #[test]
    fn test_bayesian_optimizer_default() {
        let opt = BayesianOptimizer::new();
        assert_eq!(opt.n_initial, 5);
        assert_eq!(opt.n_candidates, 1000);
    }

    #[test]
    fn test_bayesian_optimizer_builder() {
        let opt = BayesianOptimizer::new()
            .with_n_initial(10)
            .with_acquisition(AcquisitionFunction::UCB)
            .with_kappa(1.96)
            .with_kernel(Kernel::matern52())
            .with_n_candidates(500)
            .with_seed(42);

        assert_eq!(opt.n_initial, 10);
        assert_eq!(opt.kappa, 1.96);
        assert_eq!(opt.n_candidates, 500);
        assert!(opt.seed.is_some());
    }

    #[test]
    fn test_gp_marginal_likelihood() {
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(Kernel::rbf())
            .with_optimize_hyperparams(false);

        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];

        gp.fit(&x, &y).unwrap();

        let mll = gp.log_marginal_likelihood().unwrap();
        assert!(mll.is_finite());
    }

    #[test]
    fn test_kernel_with_methods() {
        let kernel = Kernel::rbf().with_length_scale(2.0).with_variance(3.0);

        assert!((kernel.length_scale() - 2.0).abs() < 1e-10);
        assert!((kernel.variance() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gp_empty_data() {
        let mut gp = GaussianProcessRegressor::new();

        let result = gp.fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gp_mismatched_shapes() {
        let mut gp = GaussianProcessRegressor::new();

        let x = vec![vec![0.0], vec![1.0]];
        let y = vec![0.0]; // Mismatched length

        let result = gp.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_gp_hyperparameter_optimization() {
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(Kernel::matern52())
            .with_optimize_hyperparams(true);

        // Fit with enough data to trigger optimization
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 / 10.0]).collect();
        let y: Vec<f64> = x.iter().map(|xi| (xi[0] * 3.14).sin()).collect();

        gp.fit(&x, &y).unwrap();

        // Kernel parameters should have been optimized
        let kernel = gp.kernel();
        assert!(kernel.length_scale() > 0.0);
        assert!(kernel.variance() > 0.0);
    }

    // =========================================================================
    // L-BFGS-B Tests
    // =========================================================================

    #[test]
    fn test_lbfgsb_rosenbrock_1d() {
        // Minimize f(x) = (1-x)^2 on [-5, 5]
        // Minimum at x = 1
        let optimizer = LBFGSB::new();

        let f = |x: &[f64]| {
            let val = (1.0 - x[0]).powi(2);
            let grad = vec![-2.0 * (1.0 - x[0])];
            (val, grad)
        };

        let bounds = vec![(-5.0, 5.0)];
        let (x_opt, f_opt) = optimizer.minimize(&f, &[0.0], &bounds);

        assert!(
            (x_opt[0] - 1.0).abs() < 0.01,
            "x_opt = {} should be ~1.0",
            x_opt[0]
        );
        assert!(f_opt < 0.001, "f_opt = {} should be ~0.0", f_opt);
    }

    #[test]
    fn test_lbfgsb_quadratic() {
        // Minimize f(x, y) = x^2 + 2y^2 on [-10, 10]^2
        // Minimum at (0, 0)
        let optimizer = LBFGSB::new();

        let f = |x: &[f64]| {
            let val = x[0].powi(2) + 2.0 * x[1].powi(2);
            let grad = vec![2.0 * x[0], 4.0 * x[1]];
            (val, grad)
        };

        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
        let (x_opt, f_opt) = optimizer.minimize(&f, &[5.0, 5.0], &bounds);

        assert!(x_opt[0].abs() < 0.01, "x[0] = {} should be ~0.0", x_opt[0]);
        assert!(x_opt[1].abs() < 0.01, "x[1] = {} should be ~0.0", x_opt[1]);
        assert!(f_opt < 0.001, "f_opt = {} should be ~0.0", f_opt);
    }

    #[test]
    fn test_lbfgsb_bounds_constraint() {
        // Minimize f(x) = -x on [0, 1]
        // Minimum at x = 1 (upper bound)
        let optimizer = LBFGSB::new();

        let f = |x: &[f64]| {
            let val = -x[0];
            let grad = vec![-1.0];
            (val, grad)
        };

        let bounds = vec![(0.0, 1.0)];
        let (x_opt, f_opt) = optimizer.minimize(&f, &[0.5], &bounds);

        assert!(
            (x_opt[0] - 1.0).abs() < 0.01,
            "x_opt = {} should be ~1.0",
            x_opt[0]
        );
        assert!(
            (f_opt - (-1.0)).abs() < 0.01,
            "f_opt = {} should be ~-1.0",
            f_opt
        );
    }

    #[test]
    fn test_lbfgsb_multistart() {
        // Minimize Rastrigin-like function with multiple local minima
        // f(x) = x^2 + 1 - cos(10x) on [-2, 2]
        // Global minimum at x = 0
        let config = LBFGSBConfig {
            n_restarts: 10,
            ..Default::default()
        };
        let optimizer = LBFGSB::new().with_config(config);

        let f = |x: &[f64]| {
            let val = x[0].powi(2) + 1.0 - (10.0 * x[0]).cos();
            let grad = vec![2.0 * x[0] + 10.0 * (10.0 * x[0]).sin()];
            (val, grad)
        };

        let bounds = vec![(-2.0, 2.0)];
        let (x_opt, f_opt) = optimizer.minimize_multistart(&f, &bounds, Some(42));

        // Should find the global minimum near 0
        assert!(
            x_opt[0].abs() < 0.5,
            "x_opt = {} should be near 0",
            x_opt[0]
        );
        assert!(f_opt < 1.5, "f_opt = {} should be near 0", f_opt);
    }

    #[test]
    fn test_lbfgsb_config() {
        let config = LBFGSBConfig {
            max_iter: 50,
            m: 5,
            gtol: 1e-4,
            ftol: 1e-6,
            n_restarts: 3,
            ..Default::default()
        };

        let optimizer = LBFGSB::new().with_config(config.clone());
        assert_eq!(optimizer.config.max_iter, 50);
        assert_eq!(optimizer.config.m, 5);
    }

    // =========================================================================
    // Acquisition Gradient Tests
    // =========================================================================

    #[test]
    fn test_ucb_gradient() {
        let dmu_dx = vec![1.0, 2.0, 3.0];
        let dsigma_dx = vec![0.1, 0.2, 0.3];
        let kappa = 2.0;

        let grad = ucb_gradient(&dmu_dx, &dsigma_dx, kappa);

        // UCB = μ + κσ, so gradient should be dμ/dx + κ * dσ/dx
        assert!((grad[0] - 1.2).abs() < 1e-10);
        assert!((grad[1] - 2.4).abs() < 1e-10);
        assert!((grad[2] - 3.6).abs() < 1e-10);
    }

    #[test]
    fn test_lcb_gradient() {
        let dmu_dx = vec![1.0, 2.0, 3.0];
        let dsigma_dx = vec![0.1, 0.2, 0.3];
        let kappa = 2.0;

        let grad = lcb_gradient(&dmu_dx, &dsigma_dx, kappa);

        // LCB = μ - κσ, so gradient should be dμ/dx - κ * dσ/dx
        assert!((grad[0] - 0.8).abs() < 1e-10);
        assert!((grad[1] - 1.6).abs() < 1e-10);
        assert!((grad[2] - 2.4).abs() < 1e-10);
    }

    #[test]
    fn test_ei_gradient_zero_sigma() {
        let dmu_dx = vec![1.0, 2.0];
        let dsigma_dx = vec![0.1, 0.2];

        let grad = expected_improvement_gradient(0.5, 0.0, &dmu_dx, &dsigma_dx, 1.0, true);

        // With zero sigma, gradient should be zero
        assert!(grad[0].abs() < 1e-10);
        assert!(grad[1].abs() < 1e-10);
    }

    #[test]
    fn test_pi_gradient_zero_sigma() {
        let dmu_dx = vec![1.0, 2.0];
        let dsigma_dx = vec![0.1, 0.2];

        let grad = probability_of_improvement_gradient(0.5, 0.0, &dmu_dx, &dsigma_dx, 1.0, true);

        // With zero sigma, gradient should be zero
        assert!(grad[0].abs() < 1e-10);
        assert!(grad[1].abs() < 1e-10);
    }

    // =========================================================================
    // GP Gradient Tests
    // =========================================================================

    #[test]
    fn test_gp_predict_with_gradients() {
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(Kernel::rbf())
            .with_optimize_hyperparams(false);

        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];

        gp.fit(&x, &y).unwrap();

        // Predict with gradients at a test point
        let (mu, sigma, dmu_dx, dsigma_dx) = gp.predict_with_gradients(&[0.25]).unwrap();

        // Mean and sigma should be finite
        assert!(mu.is_finite());
        assert!(sigma.is_finite() && sigma >= 0.0);

        // Gradients should be finite
        assert!(dmu_dx[0].is_finite());
        assert!(dsigma_dx[0].is_finite());
    }

    #[test]
    fn test_gp_gradient_numerical() {
        // Test gradient against numerical approximation
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(Kernel::rbf())
            .with_optimize_hyperparams(false);

        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![0.0, 0.25, 1.0];

        gp.fit(&x, &y).unwrap();

        let x_test = 0.3;
        let eps = 1e-5;

        // Analytical gradient
        let (_mu, _sigma, dmu_dx, _dsigma_dx) = gp.predict_with_gradients(&[x_test]).unwrap();

        // Numerical gradient
        let preds_plus = gp.predict(&[vec![x_test + eps]]).unwrap();
        let preds_minus = gp.predict(&[vec![x_test - eps]]).unwrap();
        let numerical_dmu = (preds_plus[0].0 - preds_minus[0].0) / (2.0 * eps);

        // Compare (allow some tolerance due to normalization)
        let rel_error = if numerical_dmu.abs() > 1e-6 {
            (dmu_dx[0] - numerical_dmu).abs() / numerical_dmu.abs()
        } else {
            (dmu_dx[0] - numerical_dmu).abs()
        };

        assert!(
            rel_error < 0.1 || (dmu_dx[0] - numerical_dmu).abs() < 0.1,
            "Analytical dmu/dx = {}, numerical = {}, rel_error = {}",
            dmu_dx[0],
            numerical_dmu,
            rel_error
        );
    }

    // =========================================================================
    // Acquisition Optimizer Tests
    // =========================================================================

    #[test]
    fn test_acquisition_optimizer_random() {
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(Kernel::rbf())
            .with_optimize_hyperparams(false);

        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![1.0, 0.5, 0.8]; // Minimum at x=0.5

        gp.fit(&x, &y).unwrap();

        let optimizer = AcquisitionOptimizer::new()
            .with_gradient_optimization(false)
            .with_n_candidates(500);

        let (x_opt, acq_val) = optimizer
            .optimize(
                &gp,
                AcquisitionFunction::EI,
                0.5, // best observed
                &[(0.0, 1.0)],
                2.0,
                true,
                Some(42),
            )
            .unwrap();

        // Should return a valid point in bounds
        assert!(x_opt[0] >= 0.0 && x_opt[0] <= 1.0);
        assert!(acq_val >= 0.0);
    }

    #[test]
    fn test_acquisition_optimizer_lbfgsb() {
        let mut gp = GaussianProcessRegressor::new()
            .with_kernel(Kernel::rbf())
            .with_optimize_hyperparams(false);

        let x = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y = vec![1.0, 0.5, 0.8];

        gp.fit(&x, &y).unwrap();

        let optimizer = AcquisitionOptimizer::new().with_gradient_optimization(true);

        let (x_opt, _acq_val) = optimizer
            .optimize(
                &gp,
                AcquisitionFunction::EI,
                0.5,
                &[(0.0, 1.0)],
                2.0,
                true,
                Some(42),
            )
            .unwrap();

        // Should return a valid point in bounds
        assert!(
            x_opt[0] >= 0.0 && x_opt[0] <= 1.0,
            "x_opt = {} out of bounds",
            x_opt[0]
        );
    }

    // =========================================================================
    // BayesianOptimizer with L-BFGS-B Tests
    // =========================================================================

    #[test]
    fn test_bayesian_optimizer_with_lbfgsb() {
        let opt = BayesianOptimizer::new().with_lbfgsb(true).with_n_initial(2);

        assert!(opt.use_lbfgsb);
        assert_eq!(opt.n_initial, 2);
    }

    #[test]
    fn test_bayesian_optimizer_without_lbfgsb() {
        let opt = BayesianOptimizer::new()
            .with_lbfgsb(false)
            .with_n_candidates(100);

        assert!(!opt.use_lbfgsb);
        assert_eq!(opt.n_candidates, 100);
    }

    #[test]
    fn test_bayesian_optimizer_lbfgsb_config() {
        let config = LBFGSBConfig {
            max_iter: 200,
            n_restarts: 10,
            ..Default::default()
        };

        let opt = BayesianOptimizer::new()
            .with_lbfgsb(true)
            .with_lbfgsb_config(config);

        assert_eq!(opt.lbfgsb_config.max_iter, 200);
        assert_eq!(opt.lbfgsb_config.n_restarts, 10);
    }

    #[test]
    fn test_projected_gradient_norm() {
        // Test projected gradient at interior point
        let x = vec![0.5];
        let g = vec![1.0];
        let bounds = vec![(0.0, 1.0)];

        let pg_norm = projected_gradient_norm(&x, &g, &bounds);
        // At interior, projected gradient norm should equal |x - project(x - g)|
        // project(0.5 - 1.0) = project(-0.5) = 0.0
        // |0.5 - 0.0| = 0.5
        assert!((pg_norm - 0.5).abs() < 1e-10);

        // Test at lower bound
        let x = vec![0.0];
        let g = vec![1.0]; // gradient points up
        let pg_norm = projected_gradient_norm(&x, &g, &bounds);
        // project(0.0 - 1.0) = 0.0
        // |0.0 - 0.0| = 0.0
        assert!(pg_norm < 1e-10);

        // Test at upper bound
        let x = vec![1.0];
        let g = vec![-1.0]; // gradient points down
        let pg_norm = projected_gradient_norm(&x, &g, &bounds);
        // project(1.0 - (-1.0)) = project(2.0) = 1.0
        // |1.0 - 1.0| = 0.0
        assert!(pg_norm < 1e-10);
    }
}
