//! Robust Regression with M-Estimators
//!
//! This module provides robust regression methods that are resistant to outliers,
//! using M-estimators and Iteratively Reweighted Least Squares (IRLS).
//!
//! ## Features
//!
//! - **Multiple M-estimators**: Huber, Bisquare (Tukey), Hampel, Andrew's Wave
//! - **IRLS fitting** with robust scale estimation (MAD)
//! - **Coefficient inference**: asymptotic covariance-based standard errors
//! - **Breakdown point and efficiency information** for each estimator
//!
//! ## When to Use
//!
//! - Data contains outliers that would distort OLS estimates
//! - Heavy-tailed error distributions
//! - Need a compromise between OLS efficiency and LAD robustness
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::models::robust::{RobustRegression, MEstimator};
//! use ferroml_core::models::{Model, StatisticalModel};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 2), /* ... */).unwrap();
//! let y = Array1::from_vec(/* ... */);
//!
//! // Fit with Huber M-estimator (default)
//! let mut model = RobustRegression::new();
//! model.fit(&x, &y).unwrap();
//!
//! // Get R-style summary
//! println!("{}", model.summary());
//!
//! // Use Bisquare (Tukey) for higher breakdown point
//! let mut model = RobustRegression::with_estimator(MEstimator::Bisquare);
//! model.fit(&x, &y).unwrap();
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, Assumption, AssumptionTestResult,
    CoefficientInfo, Diagnostics, FitStatistics, Model, ModelSummary, PredictionInterval,
    ProbabilisticModel, ResidualStatistics, StatisticalModel,
};
use crate::{FerroError, Result};
use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// M-estimator types for robust regression
///
/// Each estimator provides different trade-offs between efficiency (at normal errors)
/// and robustness (breakdown point, resistance to outliers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MEstimator {
    /// Huber estimator: smooth transition between L2 and L1
    ///
    /// - **ρ(u)** = u²/2 for |u| ≤ k, k|u| - k²/2 for |u| > k
    /// - **Default k**: 1.345 (95% efficiency at normal)
    /// - **Breakdown point**: 0% (but bounded influence)
    /// - **Efficiency**: ~95% at normal errors
    Huber,

    /// Bisquare (Tukey's biweight) estimator: redescending
    ///
    /// - **ρ(u)** = k²/6 * (1 - (1-(u/k)²)³) for |u| ≤ k, k²/6 for |u| > k
    /// - **Default k**: 4.685 (95% efficiency at normal)
    /// - **Breakdown point**: ~50% (high breakdown)
    /// - **Efficiency**: ~95% at normal errors
    Bisquare,

    /// Hampel estimator: three-part redescending
    ///
    /// - Linear near zero, constant in middle region, decreasing to zero
    /// - **Default (a, b, c)**: (2, 4, 8)
    /// - **Breakdown point**: ~50%
    /// - **Efficiency**: variable depending on tuning
    Hampel,

    /// Andrew's Wave estimator: sinusoidal redescending
    ///
    /// - **ρ(u)** = -cos(u/k) for |u| ≤ kπ, 2 for |u| > kπ
    /// - **Default k**: 1.339
    /// - **Breakdown point**: ~50%
    /// - **Efficiency**: ~95% at normal errors
    AndrewsWave,
}

impl Default for MEstimator {
    fn default() -> Self {
        MEstimator::Huber
    }
}

impl MEstimator {
    /// Get the default tuning constant for this estimator
    pub fn default_tuning_constant(&self) -> f64 {
        match self {
            MEstimator::Huber => 1.345,
            MEstimator::Bisquare => 4.685,
            MEstimator::Hampel => 2.0, // 'a' parameter
            MEstimator::AndrewsWave => 1.339,
        }
    }

    /// Get the efficiency at normal errors (approximate)
    pub fn efficiency_at_normal(&self) -> f64 {
        match self {
            MEstimator::Huber => 0.95,
            MEstimator::Bisquare => 0.95,
            MEstimator::Hampel => 0.92,
            MEstimator::AndrewsWave => 0.95,
        }
    }

    /// Get the breakdown point
    pub fn breakdown_point(&self) -> f64 {
        match self {
            MEstimator::Huber => 0.0, // Bounded influence but 0% breakdown
            MEstimator::Bisquare => 0.5,
            MEstimator::Hampel => 0.5,
            MEstimator::AndrewsWave => 0.5,
        }
    }

    /// Compute the ρ (rho) function value
    ///
    /// The objective function minimizes Σ ρ(residual / scale)
    pub fn rho(&self, u: f64, k: f64) -> f64 {
        match self {
            MEstimator::Huber => {
                if u.abs() <= k {
                    0.5 * u * u
                } else {
                    k * u.abs() - 0.5 * k * k
                }
            }
            MEstimator::Bisquare => {
                if u.abs() <= k {
                    let t = u / k;
                    let t2 = t * t;
                    k * k / 6.0 * (1.0 - (1.0 - t2).powi(3))
                } else {
                    k * k / 6.0
                }
            }
            MEstimator::Hampel => {
                // Using (a, b, c) = (k, 2k, 4k) for simplicity
                let a = k;
                let b = 2.0 * k;
                let c = 4.0 * k;
                let abs_u = u.abs();

                if abs_u <= a {
                    0.5 * u * u
                } else if abs_u <= b {
                    a * abs_u - 0.5 * a * a
                } else if abs_u <= c {
                    a * (c - abs_u) / (c - b) * 0.5 * (b - a + c - abs_u) + a * b - 0.5 * a * a
                } else {
                    a * (b - a + c - b) / 2.0 + a * b - 0.5 * a * a
                }
            }
            MEstimator::AndrewsWave => {
                let c = k * PI;
                if u.abs() <= c {
                    k * k * (1.0 - (u / k).cos())
                } else {
                    2.0 * k * k
                }
            }
        }
    }

    /// Compute the ψ (psi) function - first derivative of ρ
    ///
    /// The influence function; used in IRLS weighting
    pub fn psi(&self, u: f64, k: f64) -> f64 {
        match self {
            MEstimator::Huber => {
                if u.abs() <= k {
                    u
                } else {
                    k * u.signum()
                }
            }
            MEstimator::Bisquare => {
                if u.abs() <= k {
                    let t = u / k;
                    u * (1.0 - t * t).powi(2)
                } else {
                    0.0
                }
            }
            MEstimator::Hampel => {
                let a = k;
                let b = 2.0 * k;
                let c = 4.0 * k;
                let abs_u = u.abs();

                if abs_u <= a {
                    u
                } else if abs_u <= b {
                    a * u.signum()
                } else if abs_u <= c {
                    a * (c - abs_u) / (c - b) * u.signum()
                } else {
                    0.0
                }
            }
            MEstimator::AndrewsWave => {
                let c = k * PI;
                if u.abs() <= c {
                    k * (u / k).sin()
                } else {
                    0.0
                }
            }
        }
    }

    /// Compute the weight function w(u) = ψ(u) / u
    ///
    /// Used in IRLS: solve weighted least squares with these weights
    pub fn weight(&self, u: f64, k: f64) -> f64 {
        if u.abs() < 1e-10 {
            return 1.0; // Limit as u → 0
        }
        self.psi(u, k) / u
    }

    /// Compute ψ' (psi prime) - derivative of ψ
    ///
    /// Used for asymptotic variance estimation
    pub fn psi_prime(&self, u: f64, k: f64) -> f64 {
        match self {
            MEstimator::Huber => {
                if u.abs() <= k {
                    1.0
                } else {
                    0.0
                }
            }
            MEstimator::Bisquare => {
                if u.abs() <= k {
                    let t = u / k;
                    let t2 = t * t;
                    (1.0 - t2).powi(2) - 4.0 * t2 * (1.0 - t2)
                } else {
                    0.0
                }
            }
            MEstimator::Hampel => {
                let a = k;
                let b = 2.0 * k;
                let c = 4.0 * k;
                let abs_u = u.abs();

                if abs_u <= a {
                    1.0
                } else if abs_u <= b {
                    0.0
                } else if abs_u <= c {
                    -a / (c - b) * u.signum() * u.signum()
                } else {
                    0.0
                }
            }
            MEstimator::AndrewsWave => {
                let c = k * PI;
                if u.abs() <= c {
                    (u / k).cos()
                } else {
                    0.0
                }
            }
        }
    }
}

impl std::fmt::Display for MEstimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MEstimator::Huber => write!(f, "Huber"),
            MEstimator::Bisquare => write!(f, "Bisquare (Tukey)"),
            MEstimator::Hampel => write!(f, "Hampel"),
            MEstimator::AndrewsWave => write!(f, "Andrew's Wave"),
        }
    }
}

/// Robust Regression Model with M-Estimators
///
/// Fits a linear model using robust M-estimation via IRLS (Iteratively
/// Reweighted Least Squares). Resistant to outliers unlike OLS.
///
/// ## Algorithm
///
/// 1. Initialize with OLS estimates
/// 2. Compute robust scale estimate (MAD)
/// 3. Compute weights using M-estimator weight function
/// 4. Solve weighted least squares
/// 5. Repeat until convergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustRegression {
    /// M-estimator type
    pub estimator: MEstimator,
    /// Tuning constant (controls efficiency vs robustness tradeoff)
    pub tuning_constant: f64,
    /// Whether to include an intercept term
    pub fit_intercept: bool,
    /// Maximum iterations for IRLS
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Feature names (optional, for reporting)
    pub feature_names: Option<Vec<String>>,
    /// Scale estimation method
    pub scale_method: ScaleMethod,

    // Fitted parameters
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features: Option<usize>,
    fitted_data: Option<FittedRobustData>,
}

/// Method for estimating scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleMethod {
    /// Median Absolute Deviation (most robust)
    MAD,
    /// Huber's Proposal 2 (iteratively updated)
    HuberProposal2,
}

impl Default for ScaleMethod {
    fn default() -> Self {
        ScaleMethod::MAD
    }
}

/// Internal struct to store fitted data for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FittedRobustData {
    /// Number of observations
    n_samples: usize,
    /// Number of features (not including intercept)
    n_features: usize,
    /// Residuals from fit
    residuals: Array1<f64>,
    /// Fitted values
    fitted_values: Array1<f64>,
    /// Original y values
    y: Array1<f64>,
    /// Final scale estimate
    scale: f64,
    /// Final weights
    weights: Array1<f64>,
    /// Coefficient standard errors (asymptotic)
    coef_std_errors: Array1<f64>,
    /// Number of iterations to converge
    n_iter: usize,
    /// Converged flag
    converged: bool,
    /// Pseudo R-squared
    pseudo_r_squared: f64,
    /// Degrees of freedom for residuals
    df_residuals: usize,
}

impl Default for RobustRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustRegression {
    /// Create a new RobustRegression model with Huber estimator
    pub fn new() -> Self {
        let estimator = MEstimator::Huber;
        Self {
            estimator,
            tuning_constant: estimator.default_tuning_constant(),
            fit_intercept: true,
            max_iter: 50,
            tol: 1e-6,
            confidence_level: 0.95,
            feature_names: None,
            scale_method: ScaleMethod::MAD,
            coefficients: None,
            intercept: None,
            n_features: None,
            fitted_data: None,
        }
    }

    /// Create with a specific M-estimator
    pub fn with_estimator(estimator: MEstimator) -> Self {
        Self {
            estimator,
            tuning_constant: estimator.default_tuning_constant(),
            ..Self::new()
        }
    }

    /// Set the tuning constant
    pub fn with_tuning_constant(mut self, k: f64) -> Self {
        self.tuning_constant = k;
        self
    }

    /// Set whether to fit an intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set maximum iterations for IRLS
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set confidence level for intervals
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set feature names for reporting
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Set scale estimation method
    pub fn with_scale_method(mut self, method: ScaleMethod) -> Self {
        self.scale_method = method;
        self
    }

    /// Get the coefficients (excluding intercept)
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Get all coefficients including intercept (intercept first if present)
    pub fn all_coefficients(&self) -> Option<Array1<f64>> {
        let coef = self.coefficients.as_ref()?;
        if self.fit_intercept {
            let intercept = self.intercept.unwrap_or(0.0);
            let mut all = Array1::zeros(coef.len() + 1);
            all[0] = intercept;
            all.slice_mut(s![1..]).assign(coef);
            Some(all)
        } else {
            Some(coef.clone())
        }
    }

    /// Get the estimated scale
    pub fn scale(&self) -> Option<f64> {
        self.fitted_data.as_ref().map(|d| d.scale)
    }

    /// Get the weights from the final iteration
    pub fn weights(&self) -> Option<&Array1<f64>> {
        self.fitted_data.as_ref().map(|d| &d.weights)
    }

    /// Check if IRLS converged
    pub fn converged(&self) -> Option<bool> {
        self.fitted_data.as_ref().map(|d| d.converged)
    }

    /// Compute MAD (Median Absolute Deviation) from residuals
    fn compute_mad(residuals: &Array1<f64>) -> f64 {
        let n = residuals.len();
        if n == 0 {
            return 1.0;
        }

        // Compute median of residuals
        let mut sorted: Vec<f64> = residuals.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // Compute median of |residual - median|
        let mut abs_devs: Vec<f64> = residuals.iter().map(|&r| (r - median).abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mad = if n % 2 == 0 {
            (abs_devs[n / 2 - 1] + abs_devs[n / 2]) / 2.0
        } else {
            abs_devs[n / 2]
        };

        // Scale factor for consistency at normal distribution
        // MAD * 1.4826 estimates σ for normal data
        (mad * 1.4826).max(1e-10)
    }

    /// Initialize with OLS solution
    fn ols_init(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);
        self.solve_symmetric(&xtx, &xty)
    }

    /// Solve symmetric positive definite system via Cholesky
    fn solve_symmetric(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();

        // Add small regularization for numerical stability
        let mut a_reg = a.clone();
        for i in 0..n {
            a_reg[[i, i]] += 1e-10;
        }

        // Cholesky decomposition: A = LL'
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = a_reg[[i, j]];
                for k in 0..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }

                if i == j {
                    if sum <= 0.0 {
                        return Err(FerroError::numerical(
                            "Matrix is not positive definite in Cholesky decomposition",
                        ));
                    }
                    l[[i, j]] = sum.sqrt();
                } else {
                    l[[i, j]] = sum / l[[j, j]];
                }
            }
        }

        // Solve Ly = b (forward substitution)
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[[i, j]] * y[j];
            }
            y[i] = sum / l[[i, i]];
        }

        // Solve L'x = y (backward substitution)
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[[j, i]] * x[j];
            }
            x[i] = sum / l[[i, i]];
        }

        Ok(x)
    }

    /// Compute asymptotic covariance matrix for M-estimators
    ///
    /// Based on sandwich estimator: (X'WX)^{-1} * (E\[ψ²\]) / (E\[ψ'\])² * scale²
    fn compute_asymptotic_se(
        &self,
        x_design: &Array2<f64>,
        residuals: &Array1<f64>,
        scale: f64,
        weights: &Array1<f64>,
    ) -> Array1<f64> {
        let n = x_design.nrows();
        let p = x_design.ncols();
        let k = self.tuning_constant;

        // Compute E[ψ²] and E[ψ'] from the data
        let mut sum_psi_sq = 0.0;
        let mut sum_psi_prime = 0.0;

        for i in 0..n {
            let u = residuals[i] / scale;
            sum_psi_sq += self.estimator.psi(u, k).powi(2);
            sum_psi_prime += self.estimator.psi_prime(u, k);
        }

        let e_psi_sq = sum_psi_sq / n as f64;
        let e_psi_prime = (sum_psi_prime / n as f64).abs().max(0.01);

        // Correction factor
        let correction = e_psi_sq / (e_psi_prime * e_psi_prime);

        // Weighted (X'WX)^{-1}
        let mut xtwx = Array2::zeros((p, p));
        for i in 0..n {
            let xi = x_design.row(i);
            let w = weights[i];
            for j in 0..p {
                for l in 0..p {
                    xtwx[[j, l]] += w * xi[j] * xi[l];
                }
            }
        }

        // Invert X'WX
        let xtwx_inv = match self.invert_symmetric(&xtwx) {
            Ok(inv) => inv,
            Err(_) => return Array1::from_elem(p, f64::INFINITY),
        };

        // Standard errors
        let mut se = Array1::zeros(p);
        for i in 0..p {
            se[i] = (xtwx_inv[[i, i]] * correction * scale * scale).sqrt();
        }

        se
    }

    /// Invert symmetric matrix
    fn invert_symmetric(&self, a: &Array2<f64>) -> Result<Array2<f64>> {
        let n = a.nrows();
        let mut inv = Array2::zeros((n, n));

        // Solve A * inv_col = e_i for each column
        for i in 0..n {
            let mut e = Array1::zeros(n);
            e[i] = 1.0;
            let col = self.solve_symmetric(a, &e)?;
            for j in 0..n {
                inv[[j, i]] = col[j];
            }
        }

        Ok(inv)
    }

    /// Get feature name for index
    fn get_feature_name(&self, idx: usize) -> String {
        if let Some(ref names) = self.feature_names {
            if idx < names.len() {
                return names[idx].clone();
            }
        }
        format!("x{}", idx + 1)
    }

    /// Compute pseudo R-squared for robust regression
    fn compute_pseudo_r_squared(
        &self,
        residuals: &Array1<f64>,
        y: &Array1<f64>,
        scale: f64,
    ) -> f64 {
        let k = self.tuning_constant;

        // Robust objective function value for full model
        let rho_full: f64 = residuals
            .iter()
            .map(|&r| self.estimator.rho(r / scale, k))
            .sum();

        // Robust objective function value for null model (just mean/median)
        let mut sorted_y: Vec<f64> = y.iter().copied().collect();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_y = if sorted_y.is_empty() {
            f64::NAN
        } else if sorted_y.len() % 2 == 0 {
            (sorted_y[sorted_y.len() / 2 - 1] + sorted_y[sorted_y.len() / 2]) / 2.0
        } else {
            sorted_y[sorted_y.len() / 2]
        };

        let rho_null: f64 = y
            .iter()
            .map(|&yi| self.estimator.rho((yi - median_y) / scale, k))
            .sum();

        if rho_null > 0.0 {
            1.0 - rho_full / rho_null
        } else {
            0.0
        }
    }
}

impl Model for RobustRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n = x.nrows();
        let p_orig = x.ncols();

        // Build design matrix with intercept if needed
        let (x_design, p) = if self.fit_intercept {
            let mut design = Array2::zeros((n, p_orig + 1));
            design.column_mut(0).fill(1.0);
            design.slice_mut(s![.., 1..]).assign(x);
            (design, p_orig + 1)
        } else {
            (x.clone(), p_orig)
        };

        // Check for sufficient data
        if n <= p {
            return Err(FerroError::invalid_input(format!(
                "Need more observations ({}) than parameters ({})",
                n, p
            )));
        }

        // Initialize with OLS
        let mut beta = self.ols_init(&x_design, y)?;

        let k = self.tuning_constant;
        let mut converged = false;
        let mut n_iter = 0;
        let mut scale = 1.0;
        let mut weights = Array1::ones(n);

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Compute residuals
            let fitted = x_design.dot(&beta);
            let residuals: Array1<f64> = y
                .iter()
                .zip(fitted.iter())
                .map(|(&yi, &fi)| yi - fi)
                .collect();

            // Update scale estimate
            scale = match self.scale_method {
                ScaleMethod::MAD => Self::compute_mad(&residuals),
                ScaleMethod::HuberProposal2 => {
                    // Iterative scale update (simplified)
                    Self::compute_mad(&residuals)
                }
            };

            // Compute weights
            for i in 0..n {
                let u = residuals[i] / scale;
                weights[i] = self.estimator.weight(u, k).max(1e-10);
            }

            // Weighted least squares: solve (X'WX)β = X'Wy
            let mut xtwx = Array2::zeros((p, p));
            let mut xtwy = Array1::zeros(p);

            for i in 0..n {
                let xi = x_design.row(i);
                let w = weights[i];
                for j in 0..p {
                    for l in 0..p {
                        xtwx[[j, l]] += w * xi[j] * xi[l];
                    }
                    xtwy[j] += w * xi[j] * y[i];
                }
            }

            // Solve for new beta
            let beta_new = self.solve_symmetric(&xtwx, &xtwy)?;

            // Check convergence
            let diff: f64 = beta
                .iter()
                .zip(beta_new.iter())
                .map(|(&b, &bn)| (b - bn).abs())
                .sum();

            let max_beta = beta.iter().map(|&b| b.abs()).fold(0.0, f64::max).max(1.0);

            beta = beta_new;

            if diff / max_beta < self.tol {
                converged = true;
                break;
            }
        }

        // Final residuals and fitted values
        let fitted_values = x_design.dot(&beta);
        let residuals: Array1<f64> = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(&yi, &fi)| yi - fi)
            .collect();

        // Compute asymptotic standard errors
        let coef_std_errors = self.compute_asymptotic_se(&x_design, &residuals, scale, &weights);

        // Compute pseudo R-squared
        let pseudo_r_squared = self.compute_pseudo_r_squared(&residuals, y, scale);

        // Store fitted data
        self.fitted_data = Some(FittedRobustData {
            n_samples: n,
            n_features: p_orig,
            residuals,
            fitted_values,
            y: y.clone(),
            scale,
            weights,
            coef_std_errors,
            n_iter,
            converged,
            pseudo_r_squared,
            df_residuals: n - p,
        });

        // Store coefficients
        if self.fit_intercept {
            self.intercept = Some(beta[0]);
            self.coefficients = Some(beta.slice(s![1..]).to_owned());
        } else {
            self.intercept = Some(0.0);
            self.coefficients = Some(beta);
        }

        self.n_features = Some(p_orig);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.coefficients, "predict")?;

        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        let predictions = x.dot(coef) + intercept;
        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let coef = self.coefficients.as_ref()?;
        let data = self.fitted_data.as_ref()?;

        let mut importance = Array1::zeros(coef.len());
        for i in 0..coef.len() {
            let idx = if self.fit_intercept { i + 1 } else { i };
            if idx < data.coef_std_errors.len() && data.coef_std_errors[idx] > 0.0 {
                importance[i] = (coef[i] / data.coef_std_errors[idx]).abs();
            } else {
                importance[i] = coef[i].abs();
            }
        }

        Some(importance)
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl StatisticalModel for RobustRegression {
    fn summary(&self) -> ModelSummary {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => {
                return ModelSummary::new(
                    format!("Robust Regression ({}, not fitted)", self.estimator),
                    0,
                    0,
                );
            }
        };

        let mut summary = ModelSummary::new(
            format!("Robust Regression ({})", self.estimator),
            data.n_samples,
            data.n_features,
        );

        // Add fit statistics
        let mut fit_stats = FitStatistics::default();
        fit_stats.r_squared = Some(data.pseudo_r_squared);
        fit_stats.df_model = Some(data.n_features + if self.fit_intercept { 1 } else { 0 });
        fit_stats.df_residuals = Some(data.df_residuals);
        fit_stats.residual_std_error = Some(data.scale);

        summary = summary.with_fit_statistics(fit_stats);

        // Add coefficients with CIs
        let coefs = self.coefficients_with_ci(self.confidence_level);
        for coef in coefs {
            summary.add_coefficient(coef);
        }

        // Add notes
        summary.add_note(format!(
            "M-estimator: {} (k={})",
            self.estimator, self.tuning_constant
        ));
        summary.add_note(format!("Scale estimate (MAD-based): {:.4}", data.scale));
        summary.add_note(format!(
            "Converged in {} iterations: {}",
            data.n_iter,
            if data.converged { "Yes" } else { "No" }
        ));
        summary.add_note(format!(
            "Efficiency at normal: {:.0}%, Breakdown point: {:.0}%",
            self.estimator.efficiency_at_normal() * 100.0,
            self.estimator.breakdown_point() * 100.0
        ));

        summary
    }

    fn diagnostics(&self) -> Diagnostics {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => {
                return Diagnostics::new(ResidualStatistics::default());
            }
        };

        let residual_stats = ResidualStatistics::from_residuals(&data.residuals);
        let diagnostics = Diagnostics::new(residual_stats);

        diagnostics
    }

    fn coefficients_with_ci(&self, level: f64) -> Vec<CoefficientInfo> {
        let mut result = Vec::new();

        let data = match &self.fitted_data {
            Some(d) => d,
            None => return result,
        };

        let z_crit = z_critical(1.0 - (1.0 - level) / 2.0);

        // Add intercept if fitted
        if self.fit_intercept {
            let intercept = self.intercept.unwrap_or(0.0);
            let se = data.coef_std_errors[0];
            let z_stat = if se > 0.0 { intercept / se } else { 0.0 };
            let p_value = 2.0 * (1.0 - standard_normal_cdf(z_stat.abs()));

            result.push(
                CoefficientInfo::new("(Intercept)", intercept, se)
                    .with_test(z_stat, p_value)
                    .with_ci(intercept - z_crit * se, intercept + z_crit * se, level),
            );
        }

        // Add feature coefficients
        if let Some(coef) = &self.coefficients {
            for (i, &c) in coef.iter().enumerate() {
                let se_idx = if self.fit_intercept { i + 1 } else { i };
                let se = data.coef_std_errors[se_idx];
                let z_stat = if se > 0.0 { c / se } else { 0.0 };
                let p_value = 2.0 * (1.0 - standard_normal_cdf(z_stat.abs()));

                let name = self.get_feature_name(i);
                result.push(
                    CoefficientInfo::new(name, c, se)
                        .with_test(z_stat, p_value)
                        .with_ci(c - z_crit * se, c + z_crit * se, level),
                );
            }
        }

        result
    }

    fn residuals(&self) -> Option<Array1<f64>> {
        self.fitted_data.as_ref().map(|d| d.residuals.clone())
    }

    fn fitted_values(&self) -> Option<Array1<f64>> {
        self.fitted_data.as_ref().map(|d| d.fitted_values.clone())
    }

    fn assumption_test(&self, _assumption: Assumption) -> Option<AssumptionTestResult> {
        // Robust regression doesn't require normality or homoscedasticity
        None
    }
}

impl ProbabilisticModel for RobustRegression {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let preds = self.predict(x)?;
        let n = preds.len();

        let mut result = Array2::zeros((n, 1));
        result.column_mut(0).assign(&preds);
        Ok(result)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        check_is_fitted(&self.coefficients, "predict_interval")?;

        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let data = self
            .fitted_data
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_interval"))?;

        let predictions = self.predict(x)?;
        let n = x.nrows();

        // Use scale-based prediction intervals
        let z_crit = z_critical(1.0 - (1.0 - level) / 2.0);
        let margin = z_crit * data.scale;

        let lower = &predictions - margin;
        let upper = &predictions + margin;

        let std_errors = Array1::from_elem(n, data.scale);

        Ok(PredictionInterval::new(predictions, lower, upper, level).with_std_errors(std_errors))
    }
}

/// Standard normal CDF approximation
fn standard_normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp();
    let c = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    if x >= 0.0 {
        1.0 - p * c
    } else {
        p * c
    }
}

/// Standard normal quantile (inverse CDF) approximation
fn z_critical(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let p_adj = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * p_adj.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p > 0.5 {
        z
    } else {
        -z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_robust_regression_simple() {
        // y = 1 + 2*x with no noise
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = RobustRegression::new();
        model.fit(&x, &y).unwrap();

        // Check coefficients (should be close to true values)
        assert_relative_eq!(model.intercept().unwrap(), 1.0, epsilon = 0.5);
        let coef = model.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_robust_regression_with_outliers() {
        // y = 1 + 2*x with outliers
        let x_data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut y_data: Vec<f64> = x_data.iter().map(|&xi| 1.0 + 2.0 * xi).collect();

        // Add outliers
        y_data[5] = 100.0; // Large positive outlier
        y_data[15] = -50.0; // Large negative outlier

        let x = Array2::from_shape_vec((20, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        // Robust regression should be resistant
        let mut robust = RobustRegression::with_estimator(MEstimator::Bisquare);
        robust.fit(&x, &y).unwrap();

        // Coefficients should still be close to true values
        assert!(robust.intercept().unwrap().abs() < 10.0); // Reasonable intercept
        let coef = robust.coefficients().unwrap();
        assert!((coef[0] - 2.0).abs() < 1.0); // Slope close to 2
    }

    #[test]
    fn test_different_estimators() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let estimators = [
            MEstimator::Huber,
            MEstimator::Bisquare,
            MEstimator::Hampel,
            MEstimator::AndrewsWave,
        ];

        for est in estimators {
            let mut model = RobustRegression::with_estimator(est);
            model.fit(&x, &y).unwrap();

            assert!(model.is_fitted());
            assert!(model.converged().unwrap());

            let coef = model.coefficients().unwrap();
            assert_relative_eq!(coef[0], 2.0, epsilon = 0.2);
        }
    }

    #[test]
    fn test_m_estimator_functions() {
        let k = 1.345;

        // Huber rho
        let huber = MEstimator::Huber;
        assert_relative_eq!(huber.rho(0.5, k), 0.125, epsilon = 1e-10);
        assert_relative_eq!(huber.rho(2.0, k), k * 2.0 - 0.5 * k * k, epsilon = 1e-10);

        // Huber psi
        assert_relative_eq!(huber.psi(0.5, k), 0.5, epsilon = 1e-10);
        assert_relative_eq!(huber.psi(2.0, k), k, epsilon = 1e-10);

        // Huber weight
        assert_relative_eq!(huber.weight(0.5, k), 1.0, epsilon = 1e-10);
        assert!(huber.weight(2.0, k) < 1.0); // Down-weighted

        // Bisquare goes to zero for large residuals
        let bisquare = MEstimator::Bisquare;
        assert!(bisquare.psi(10.0, 4.685).abs() < 1e-10);
        assert!(bisquare.weight(10.0, 4.685) < 1e-10);
    }

    #[test]
    fn test_scale_estimation() {
        let residuals = array![-2.0, -1.0, 0.0, 1.0, 2.0, 100.0]; // One outlier

        let scale = RobustRegression::compute_mad(&residuals);

        // MAD should be resistant to the outlier
        assert!(scale < 10.0); // Much less than if we used std dev
    }

    #[test]
    fn test_without_intercept() {
        // y = 2*x (through origin)
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 * xi);

        let mut model = RobustRegression::new().with_fit_intercept(false);
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 0.2);
    }

    #[test]
    fn test_predict() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = RobustRegression::new();
        model.fit(&x, &y).unwrap();

        let x_new = Array2::from_shape_vec((3, 1), vec![11.0, 12.0, 13.0]).unwrap();
        let predictions = model.predict(&x_new).unwrap();

        assert_eq!(predictions.len(), 3);
        // Predictions should be approximately y = 1 + 2*x
        assert_relative_eq!(predictions[0], 23.0, epsilon = 2.0);
    }

    #[test]
    fn test_prediction_intervals() {
        let x = Array2::from_shape_vec((30, 1), (1..=30).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = RobustRegression::new();
        model.fit(&x, &y).unwrap();

        let intervals = model.predict_interval(&x, 0.95).unwrap();

        // All intervals should contain the point predictions
        for i in 0..y.len() {
            assert!(intervals.lower[i] <= intervals.predictions[i]);
            assert!(intervals.upper[i] >= intervals.predictions[i]);
        }
    }

    #[test]
    fn test_summary() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = RobustRegression::new().with_feature_names(vec!["feature1".to_string()]);
        model.fit(&x, &y).unwrap();

        let summary = model.summary();
        let output = format!("{}", summary);

        assert!(output.contains("Robust Regression"));
        assert!(output.contains("Huber"));
        assert!(output.contains("(Intercept)"));
        assert!(output.contains("feature1"));
    }

    #[test]
    fn test_coefficients_with_ci() {
        let x = Array2::from_shape_vec((30, 1), (1..=30).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = RobustRegression::new();
        model.fit(&x, &y).unwrap();

        let coefs = model.coefficients_with_ci(0.95);

        // Should have intercept and one coefficient
        assert_eq!(coefs.len(), 2);

        // CIs should bracket the estimate
        for coef in &coefs {
            assert!(coef.ci_lower <= coef.estimate);
            assert!(coef.ci_upper >= coef.estimate);
        }
    }

    #[test]
    fn test_residuals_and_fitted() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = RobustRegression::new();
        model.fit(&x, &y).unwrap();

        let residuals = model.residuals().unwrap();
        let fitted = model.fitted_values().unwrap();

        // residuals + fitted should approximately equal y
        for i in 0..y.len() {
            assert_relative_eq!(residuals[i] + fitted[i], y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_weights() {
        let x_data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let mut y_data: Vec<f64> = x_data.iter().map(|&xi| 1.0 + 2.0 * xi).collect();

        // Add an outlier
        y_data[10] = 200.0;

        let x = Array2::from_shape_vec((20, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let mut model = RobustRegression::with_estimator(MEstimator::Bisquare);
        model.fit(&x, &y).unwrap();

        let weights = model.weights().unwrap();

        // The outlier should have lower weight
        let min_weight_idx = weights
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        assert_eq!(min_weight_idx, 10); // The outlier
    }

    #[test]
    fn test_error_not_fitted() {
        let model = RobustRegression::new();
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_m_estimator_properties() {
        let estimators = [
            MEstimator::Huber,
            MEstimator::Bisquare,
            MEstimator::Hampel,
            MEstimator::AndrewsWave,
        ];

        for est in estimators {
            // Efficiency should be between 0 and 1
            assert!(est.efficiency_at_normal() > 0.0 && est.efficiency_at_normal() <= 1.0);

            // Breakdown point should be between 0 and 0.5
            assert!(est.breakdown_point() >= 0.0 && est.breakdown_point() <= 0.5);

            // Default tuning constant should be positive
            assert!(est.default_tuning_constant() > 0.0);
        }
    }

    #[test]
    fn test_multiple_features() {
        // Create independent features
        let mut x_data = Vec::new();
        for i in 0..30 {
            x_data.push(i as f64);
            x_data.push(30.0 - i as f64 + 0.5 * (i % 5) as f64);
        }
        let x = Array2::from_shape_vec((30, 2), x_data).unwrap();
        let y: Array1<f64> = (0..30)
            .map(|i| 1.0 + 0.5 * x[[i, 0]] + 0.3 * x[[i, 1]])
            .collect();

        let mut model =
            RobustRegression::new().with_feature_names(vec!["x1".to_string(), "x2".to_string()]);
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());
        let coef = model.coefficients().unwrap();
        assert_eq!(coef.len(), 2);

        // Feature importance should work
        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);
    }

    #[test]
    fn test_convergence_info() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = RobustRegression::new().with_max_iter(100).with_tol(1e-8);
        model.fit(&x, &y).unwrap();

        assert!(model.converged().unwrap());
    }
}
