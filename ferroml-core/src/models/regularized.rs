//! Regularized Linear Models
//!
//! This module provides Ridge, Lasso, and ElasticNet regression with full statistical
//! diagnostics and cross-validated hyperparameter selection.
//!
//! ## Models
//!
//! - **RidgeRegression**: L2 regularization (closed-form solution)
//! - **LassoRegression**: L1 regularization (coordinate descent)
//! - **ElasticNet**: L1 + L2 combination (coordinate descent)
//!
//! ## CV Variants
//!
//! - **RidgeCV**: Ridge with cross-validated alpha selection
//! - **LassoCV**: Lasso with cross-validated alpha selection
//! - **ElasticNetCV**: ElasticNet with cross-validated alpha and l1_ratio selection
//!
//! ## Example
//!
//! ```
//! use ferroml_core::models::regularized::{RidgeRegression, LassoRegression, ElasticNet};
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 2), vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]).unwrap();
//! let y = Array1::from_vec(vec![3., 7., 11., 15., 19.]);
//!
//! // Ridge regression with L2 penalty
//! let mut ridge = RidgeRegression::new(1.0);
//! ridge.fit(&x, &y).unwrap();
//!
//! // Lasso with L1 penalty (sparse solutions)
//! let mut lasso = LassoRegression::new(0.1);
//! lasso.fit(&x, &y).unwrap();
//!
//! // ElasticNet combining L1 and L2
//! let mut elastic = ElasticNet::new(0.1, 0.5); // alpha=0.1, l1_ratio=0.5
//! elastic.fit(&x, &y).unwrap();
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, Assumption, AssumptionTestResult,
    CoefficientInfo, Diagnostics, FitStatistics, Model, ModelSummary, PredictionInterval,
    ProbabilisticModel, ResidualStatistics, StatisticalModel,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

// =============================================================================
// Ridge Regression
// =============================================================================

/// Ridge Regression (L2 regularization)
///
/// Minimizes: ||y - Xβ||² + α||β||²
///
/// Ridge regression adds L2 penalty to the coefficients, which:
/// - Shrinks coefficients toward zero (but never exactly zero)
/// - Helps with multicollinearity
/// - Has a closed-form solution: β = (X'X + αI)^(-1)X'y
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::regularized::RidgeRegression;
/// use ferroml_core::models::Model;
///
/// let mut ridge = RidgeRegression::new(1.0);
/// ridge.fit(&x, &y)?;
/// let predictions = ridge.predict(&x_test)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeRegression {
    /// Regularization strength (must be > 0)
    pub alpha: f64,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Whether to normalize features before fitting
    pub normalize: bool,
    /// Maximum iterations for solver (if iterative)
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Feature names for reporting
    pub feature_names: Option<Vec<String>>,

    // Fitted parameters
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features: Option<usize>,
    fitted_data: Option<RidgeFittedData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RidgeFittedData {
    n_samples: usize,
    n_features: usize,
    residuals: Array1<f64>,
    fitted_values: Array1<f64>,
    y_mean: f64,
    tss: f64,
    rss: f64,
    coef_std_errors: Array1<f64>,
    df_residuals: usize,
    effective_df: f64, // Effective degrees of freedom for ridge
}

impl RidgeRegression {
    /// Create a new Ridge regression model
    ///
    /// # Arguments
    /// * `alpha` - Regularization strength. Larger values mean more regularization.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            fit_intercept: true,
            normalize: false,
            max_iter: 1000,
            tol: 1e-6,
            confidence_level: 0.95,
            feature_names: None,
            coefficients: None,
            intercept: None,
            n_features: None,
            fitted_data: None,
        }
    }

    /// Set whether to fit an intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set feature names for reporting
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Get the R² value
    pub fn r_squared(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        Some(1.0 - data.rss / data.tss)
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
}

impl Default for RidgeRegression {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Model for RidgeRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.alpha < 0.0 {
            return Err(FerroError::invalid_input("alpha must be non-negative"));
        }

        let n = x.nrows();
        let p = x.ncols();

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap_or(0.0);
            let x_c = x - &x_mean;
            let y_c = y - y_mean;
            (x_c, y_c, x_mean, y_mean)
        } else {
            (x.clone(), y.clone(), Array1::zeros(p), 0.0)
        };

        // Ridge closed-form: β = (X'X + αI)^(-1)X'y
        let xtx = x_centered.t().dot(&x_centered);
        let xty = x_centered.t().dot(&y_centered);

        // Add regularization: X'X + αI
        let mut xtx_reg = xtx;
        for i in 0..p {
            xtx_reg[[i, i]] += self.alpha;
        }

        // Solve via Cholesky decomposition
        let coefficients = solve_symmetric_positive_definite(&xtx_reg, &xty)?;

        // Compute intercept
        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&coefficients)
        } else {
            0.0
        };

        // Compute fitted values and residuals
        let fitted_values = x.dot(&coefficients) + intercept;
        let residuals = y - &fitted_values;

        // Compute statistics
        let tss: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|r| r.powi(2)).sum();

        // Effective degrees of freedom for ridge regression
        // df_eff = tr(H) where H = X(X'X + αI)^(-1)X'
        let xtx_inv = invert_symmetric(&xtx_reg)?;
        let h_diag = compute_hat_diagonal(x, &xtx_inv);
        let effective_df = h_diag.sum().min(n as f64 - 1.0).max(1.0);

        let df_residuals = n.saturating_sub(effective_df.round() as usize).max(1);

        // Coefficient standard errors (approximate for ridge)
        let mse = rss / df_residuals as f64;
        let coef_var = &xtx_inv * mse;
        let mut coef_std_errors = Array1::zeros(p);
        for i in 0..p {
            coef_std_errors[i] = coef_var[[i, i]].sqrt();
        }

        // Store results
        self.coefficients = Some(coefficients);
        self.intercept = Some(intercept);
        self.n_features = Some(p);
        self.fitted_data = Some(RidgeFittedData {
            n_samples: n,
            n_features: p,
            residuals,
            fitted_values,
            y_mean,
            tss,
            rss,
            coef_std_errors,
            df_residuals,
            effective_df,
        });

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.coefficients, "predict")?;

        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        Ok(x.dot(coef) + intercept)
    }

    fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let coef = self.coefficients.as_ref()?;
        Some(coef.mapv(|c| c.abs()))
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float_log("alpha", 1e-4, 1e4)
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl StatisticalModel for RidgeRegression {
    fn summary(&self) -> ModelSummary {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => return ModelSummary::new("Ridge Regression (not fitted)", 0, 0),
        };

        let mut summary = ModelSummary::new(
            format!("Ridge Regression (alpha={:.4})", self.alpha),
            data.n_samples,
            data.n_features,
        );

        // Fit statistics
        let r2 = 1.0 - data.rss / data.tss;
        let adj_r2 = 1.0
            - (1.0 - r2) * (data.n_samples as f64 - 1.0)
                / (data.n_samples as f64 - data.effective_df - 1.0);

        let fit_stats = FitStatistics::with_r_squared(r2, adj_r2)
            .with_df(data.effective_df.round() as usize, data.df_residuals);
        summary = summary.with_fit_statistics(fit_stats);

        summary.add_note(format!(
            "Effective degrees of freedom: {:.2}",
            data.effective_df
        ));

        // Add coefficients
        for coef in self.coefficients_with_ci(self.confidence_level) {
            summary.add_coefficient(coef);
        }

        summary
    }

    fn diagnostics(&self) -> Diagnostics {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => return Diagnostics::new(ResidualStatistics::default()),
        };

        Diagnostics::new(ResidualStatistics::from_residuals(&data.residuals))
    }

    fn coefficients_with_ci(&self, level: f64) -> Vec<CoefficientInfo> {
        let mut result = Vec::new();
        let data = match &self.fitted_data {
            Some(d) => d,
            None => return result,
        };

        let df = data.df_residuals as f64;
        let t_crit = t_critical(1.0 - (1.0 - level) / 2.0, df);

        // Intercept
        if self.fit_intercept {
            if let Some(intercept) = self.intercept {
                result.push(CoefficientInfo::new("(Intercept)", intercept, 0.0));
            }
        }

        // Feature coefficients
        if let Some(coef) = &self.coefficients {
            for (i, &c) in coef.iter().enumerate() {
                let se = data.coef_std_errors[i];
                let t_stat = if se > 0.0 { c / se } else { 0.0 };
                let p_value = 2.0 * (1.0 - t_cdf_approx(t_stat.abs(), df));

                result.push(
                    CoefficientInfo::new(self.get_feature_name(i), c, se)
                        .with_test(t_stat, p_value)
                        .with_ci(c - t_crit * se, c + t_crit * se, level),
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
        None
    }
}

impl ProbabilisticModel for RidgeRegression {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let preds = self.predict(x)?;
        let n = preds.len();
        let mut result = Array2::zeros((n, 1));
        result.column_mut(0).assign(&preds);
        Ok(result)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        check_is_fitted(&self.coefficients, "predict_interval")?;

        let predictions = self.predict(x)?;
        let data = self.fitted_data.as_ref().unwrap();

        let t_crit = t_critical(1.0 - (1.0 - level) / 2.0, data.df_residuals as f64);
        let mse = data.rss / data.df_residuals as f64;
        let se = mse.sqrt();

        let lower = &predictions - se * t_crit;
        let upper = &predictions + se * t_crit;

        Ok(PredictionInterval::new(predictions, lower, upper, level))
    }
}

// =============================================================================
// Lasso Regression
// =============================================================================

/// Lasso Regression (L1 regularization)
///
/// Minimizes: (1/2n)||y - Xβ||² + α||β||₁
///
/// Lasso regression adds L1 penalty to the coefficients, which:
/// - Encourages sparse solutions (some coefficients become exactly zero)
/// - Performs automatic feature selection
/// - Requires iterative coordinate descent to solve
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::regularized::LassoRegression;
/// use ferroml_core::models::Model;
///
/// let mut lasso = LassoRegression::new(0.1);
/// lasso.fit(&x, &y)?;
///
/// // Get sparse coefficients
/// let coef = lasso.coefficients().unwrap();
/// let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-10).count();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LassoRegression {
    /// Regularization strength
    pub alpha: f64,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Maximum iterations for coordinate descent
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Warm start from previous coefficients
    pub warm_start: bool,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Feature names for reporting
    pub feature_names: Option<Vec<String>>,

    // Fitted parameters
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features: Option<usize>,
    n_iter: Option<usize>,
    fitted_data: Option<LassoFittedData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LassoFittedData {
    n_samples: usize,
    n_features: usize,
    residuals: Array1<f64>,
    fitted_values: Array1<f64>,
    y_mean: f64,
    tss: f64,
    rss: f64,
    n_nonzero: usize,
}

impl LassoRegression {
    /// Create a new Lasso regression model
    ///
    /// # Arguments
    /// * `alpha` - Regularization strength. Larger values mean more regularization.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            warm_start: false,
            confidence_level: 0.95,
            feature_names: None,
            coefficients: None,
            intercept: None,
            n_features: None,
            n_iter: None,
            fitted_data: None,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit an intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set feature names for reporting
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Get the number of iterations used
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter
    }

    /// Get the number of non-zero coefficients
    pub fn n_nonzero(&self) -> Option<usize> {
        self.fitted_data.as_ref().map(|d| d.n_nonzero)
    }

    /// Get the R² value
    pub fn r_squared(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        Some(1.0 - data.rss / data.tss)
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
}

impl Default for LassoRegression {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Model for LassoRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.alpha < 0.0 {
            return Err(FerroError::invalid_input("alpha must be non-negative"));
        }

        let n = x.nrows();
        let p = x.ncols();

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap_or(0.0);
            let x_c = x - &x_mean;
            let y_c = y - y_mean;
            (x_c, y_c, x_mean, y_mean)
        } else {
            (x.clone(), y.clone(), Array1::zeros(p), 0.0)
        };

        // Initialize coefficients
        let mut coef = if self.warm_start && self.coefficients.is_some() {
            self.coefficients.clone().unwrap()
        } else {
            Array1::zeros(p)
        };

        // Precompute X'X diagonal and X'y
        let mut x_col_norms_sq = Array1::zeros(p);
        for j in 0..p {
            x_col_norms_sq[j] = x_centered.column(j).dot(&x_centered.column(j));
        }
        let xty = x_centered.t().dot(&y_centered);

        // Coordinate descent
        let alpha_n = self.alpha * n as f64;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            let coef_old = coef.clone();

            for j in 0..p {
                if x_col_norms_sq[j] < 1e-14 {
                    continue;
                }

                // Compute partial residual
                let mut partial_residual = xty[j];
                for k in 0..p {
                    if k != j {
                        partial_residual -=
                            x_centered.column(j).dot(&x_centered.column(k)) * coef[k];
                    }
                }

                // Soft thresholding
                coef[j] = soft_threshold(partial_residual, alpha_n) / x_col_norms_sq[j];
            }

            n_iter = iter + 1;

            // Check convergence
            let max_change = (&coef - &coef_old)
                .mapv(|c| c.abs())
                .fold(0.0_f64, |a, &b| a.max(b));
            if max_change < self.tol {
                break;
            }
        }

        // Compute intercept
        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&coef)
        } else {
            0.0
        };

        // Compute fitted values and residuals
        let fitted_values = x.dot(&coef) + intercept;
        let residuals = y - &fitted_values;

        // Compute statistics
        let tss: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|r| r.powi(2)).sum();
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-10).count();

        // Store results
        self.coefficients = Some(coef);
        self.intercept = Some(intercept);
        self.n_features = Some(p);
        self.n_iter = Some(n_iter);
        self.fitted_data = Some(LassoFittedData {
            n_samples: n,
            n_features: p,
            residuals,
            fitted_values,
            y_mean,
            tss,
            rss,
            n_nonzero,
        });

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.coefficients, "predict")?;

        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        Ok(x.dot(coef) + intercept)
    }

    fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let coef = self.coefficients.as_ref()?;
        Some(coef.mapv(|c| c.abs()))
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new().float_log("alpha", 1e-4, 1e2)
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl StatisticalModel for LassoRegression {
    fn summary(&self) -> ModelSummary {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => return ModelSummary::new("Lasso Regression (not fitted)", 0, 0),
        };

        let mut summary = ModelSummary::new(
            format!("Lasso Regression (alpha={:.4})", self.alpha),
            data.n_samples,
            data.n_features,
        );

        // Fit statistics
        let r2 = 1.0 - data.rss / data.tss;
        let df_model = data.n_nonzero;
        let df_residuals = data.n_samples.saturating_sub(df_model + 1);
        let adj_r2 = if df_residuals > 0 {
            1.0 - (1.0 - r2) * (data.n_samples as f64 - 1.0) / df_residuals as f64
        } else {
            r2
        };

        let fit_stats = FitStatistics::with_r_squared(r2, adj_r2).with_df(df_model, df_residuals);
        summary = summary.with_fit_statistics(fit_stats);

        summary.add_note(format!("Non-zero coefficients: {}", data.n_nonzero));
        if let Some(n_iter) = self.n_iter {
            summary.add_note(format!("Iterations: {}", n_iter));
        }

        // Add only non-zero coefficients
        for coef in self.coefficients_with_ci(self.confidence_level) {
            if coef.estimate.abs() > 1e-10 || coef.name == "(Intercept)" {
                summary.add_coefficient(coef);
            }
        }

        summary
    }

    fn diagnostics(&self) -> Diagnostics {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => return Diagnostics::new(ResidualStatistics::default()),
        };

        Diagnostics::new(ResidualStatistics::from_residuals(&data.residuals))
    }

    fn coefficients_with_ci(&self, level: f64) -> Vec<CoefficientInfo> {
        let mut result = Vec::new();

        // Intercept
        if self.fit_intercept {
            if let Some(intercept) = self.intercept {
                result.push(
                    CoefficientInfo::new("(Intercept)", intercept, 0.0)
                        .with_ci(intercept, intercept, level),
                );
            }
        }

        // Feature coefficients (CIs not well-defined for Lasso)
        if let Some(coef) = &self.coefficients {
            for (i, &c) in coef.iter().enumerate() {
                result.push(
                    CoefficientInfo::new(self.get_feature_name(i), c, 0.0).with_ci(c, c, level),
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
        None
    }
}

// =============================================================================
// ElasticNet
// =============================================================================

/// ElasticNet Regression (L1 + L2 regularization)
///
/// Minimizes: (1/2n)||y - Xβ||² + α * l1_ratio * ||β||₁ + α * (1 - l1_ratio) * ||β||²/2
///
/// ElasticNet combines L1 and L2 penalties:
/// - `l1_ratio = 1.0`: Pure Lasso
/// - `l1_ratio = 0.0`: Pure Ridge
/// - `0 < l1_ratio < 1`: Mixed penalties
///
/// ElasticNet is useful when:
/// - You want sparse solutions (like Lasso)
/// - But also want to handle correlated features (like Ridge)
/// - When p >> n (more features than samples)
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::regularized::ElasticNet;
/// use ferroml_core::models::Model;
///
/// // alpha=0.1, l1_ratio=0.5 means equal L1 and L2 penalty
/// let mut elastic = ElasticNet::new(0.1, 0.5);
/// elastic.fit(&x, &y)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticNet {
    /// Overall regularization strength
    pub alpha: f64,
    /// L1/L2 ratio (0.0 = pure Ridge, 1.0 = pure Lasso)
    pub l1_ratio: f64,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Maximum iterations for coordinate descent
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Warm start from previous coefficients
    pub warm_start: bool,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Feature names for reporting
    pub feature_names: Option<Vec<String>>,

    // Fitted parameters
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features: Option<usize>,
    n_iter: Option<usize>,
    fitted_data: Option<ElasticNetFittedData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ElasticNetFittedData {
    n_samples: usize,
    n_features: usize,
    residuals: Array1<f64>,
    fitted_values: Array1<f64>,
    y_mean: f64,
    tss: f64,
    rss: f64,
    n_nonzero: usize,
}

impl ElasticNet {
    /// Create a new ElasticNet model
    ///
    /// # Arguments
    /// * `alpha` - Overall regularization strength
    /// * `l1_ratio` - Ratio of L1 to L2 penalty (0.0 to 1.0)
    pub fn new(alpha: f64, l1_ratio: f64) -> Self {
        Self {
            alpha,
            l1_ratio,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            warm_start: false,
            confidence_level: 0.95,
            feature_names: None,
            coefficients: None,
            intercept: None,
            n_features: None,
            n_iter: None,
            fitted_data: None,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit an intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set feature names for reporting
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Get the coefficients
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Get the number of iterations used
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter
    }

    /// Get the number of non-zero coefficients
    pub fn n_nonzero(&self) -> Option<usize> {
        self.fitted_data.as_ref().map(|d| d.n_nonzero)
    }

    /// Get the R² value
    pub fn r_squared(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        Some(1.0 - data.rss / data.tss)
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
}

impl Default for ElasticNet {
    fn default() -> Self {
        Self::new(1.0, 0.5)
    }
}

impl Model for ElasticNet {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.alpha < 0.0 {
            return Err(FerroError::invalid_input("alpha must be non-negative"));
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(FerroError::invalid_input(
                "l1_ratio must be between 0 and 1",
            ));
        }

        let n = x.nrows();
        let p = x.ncols();

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap_or(0.0);
            let x_c = x - &x_mean;
            let y_c = y - y_mean;
            (x_c, y_c, x_mean, y_mean)
        } else {
            (x.clone(), y.clone(), Array1::zeros(p), 0.0)
        };

        // Initialize coefficients
        let mut coef = if self.warm_start && self.coefficients.is_some() {
            self.coefficients.clone().unwrap()
        } else {
            Array1::zeros(p)
        };

        // Precompute X'X diagonal and X'y
        let mut x_col_norms_sq = Array1::zeros(p);
        for j in 0..p {
            x_col_norms_sq[j] = x_centered.column(j).dot(&x_centered.column(j));
        }
        let xty = x_centered.t().dot(&y_centered);

        // ElasticNet coordinate descent
        let alpha_l1 = self.alpha * self.l1_ratio * n as f64;
        let alpha_l2 = self.alpha * (1.0 - self.l1_ratio) * n as f64;
        let mut n_iter = 0;

        for iter in 0..self.max_iter {
            let coef_old = coef.clone();

            for j in 0..p {
                let denom = x_col_norms_sq[j] + alpha_l2;
                if denom < 1e-14 {
                    continue;
                }

                // Compute partial residual
                let mut partial_residual = xty[j];
                for k in 0..p {
                    if k != j {
                        partial_residual -=
                            x_centered.column(j).dot(&x_centered.column(k)) * coef[k];
                    }
                }

                // Soft thresholding with L2 scaling
                coef[j] = soft_threshold(partial_residual, alpha_l1) / denom;
            }

            n_iter = iter + 1;

            // Check convergence
            let max_change = (&coef - &coef_old)
                .mapv(|c| c.abs())
                .fold(0.0_f64, |a, &b| a.max(b));
            if max_change < self.tol {
                break;
            }
        }

        // Compute intercept
        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&coef)
        } else {
            0.0
        };

        // Compute fitted values and residuals
        let fitted_values = x.dot(&coef) + intercept;
        let residuals = y - &fitted_values;

        // Compute statistics
        let tss: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|r| r.powi(2)).sum();
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-10).count();

        // Store results
        self.coefficients = Some(coef);
        self.intercept = Some(intercept);
        self.n_features = Some(p);
        self.n_iter = Some(n_iter);
        self.fitted_data = Some(ElasticNetFittedData {
            n_samples: n,
            n_features: p,
            residuals,
            fitted_values,
            y_mean,
            tss,
            rss,
            n_nonzero,
        });

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.coefficients, "predict")?;

        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        Ok(x.dot(coef) + intercept)
    }

    fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let coef = self.coefficients.as_ref()?;
        Some(coef.mapv(|c| c.abs()))
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .float_log("alpha", 1e-4, 1e2)
            .float("l1_ratio", 0.0, 1.0)
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl StatisticalModel for ElasticNet {
    fn summary(&self) -> ModelSummary {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => return ModelSummary::new("ElasticNet (not fitted)", 0, 0),
        };

        let mut summary = ModelSummary::new(
            format!(
                "ElasticNet (alpha={:.4}, l1_ratio={:.2})",
                self.alpha, self.l1_ratio
            ),
            data.n_samples,
            data.n_features,
        );

        // Fit statistics
        let r2 = 1.0 - data.rss / data.tss;
        let df_model = data.n_nonzero;
        let df_residuals = data.n_samples.saturating_sub(df_model + 1);
        let adj_r2 = if df_residuals > 0 {
            1.0 - (1.0 - r2) * (data.n_samples as f64 - 1.0) / df_residuals as f64
        } else {
            r2
        };

        let fit_stats = FitStatistics::with_r_squared(r2, adj_r2).with_df(df_model, df_residuals);
        summary = summary.with_fit_statistics(fit_stats);

        summary.add_note(format!("Non-zero coefficients: {}", data.n_nonzero));
        if let Some(n_iter) = self.n_iter {
            summary.add_note(format!("Iterations: {}", n_iter));
        }

        // Add only non-zero coefficients
        for coef in self.coefficients_with_ci(self.confidence_level) {
            if coef.estimate.abs() > 1e-10 || coef.name == "(Intercept)" {
                summary.add_coefficient(coef);
            }
        }

        summary
    }

    fn diagnostics(&self) -> Diagnostics {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => return Diagnostics::new(ResidualStatistics::default()),
        };

        Diagnostics::new(ResidualStatistics::from_residuals(&data.residuals))
    }

    fn coefficients_with_ci(&self, level: f64) -> Vec<CoefficientInfo> {
        let mut result = Vec::new();

        // Intercept
        if self.fit_intercept {
            if let Some(intercept) = self.intercept {
                result.push(
                    CoefficientInfo::new("(Intercept)", intercept, 0.0)
                        .with_ci(intercept, intercept, level),
                );
            }
        }

        // Feature coefficients
        if let Some(coef) = &self.coefficients {
            for (i, &c) in coef.iter().enumerate() {
                result.push(
                    CoefficientInfo::new(self.get_feature_name(i), c, 0.0).with_ci(c, c, level),
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
        None
    }
}

// =============================================================================
// Regularization Path
// =============================================================================

/// Result of computing a regularization path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationPath {
    /// Alpha values used
    pub alphas: Vec<f64>,
    /// Coefficients at each alpha (shape: n_alphas x n_features)
    pub coefs: Array2<f64>,
    /// Intercepts at each alpha
    pub intercepts: Vec<f64>,
    /// Number of non-zero coefficients at each alpha
    pub n_nonzeros: Vec<usize>,
    /// R² at each alpha (if computed)
    pub r_squared: Option<Vec<f64>>,
}

impl RegularizationPath {
    /// Get the regularization path as a vector of (alpha, coef) tuples
    pub fn as_tuples(&self) -> Vec<(f64, Array1<f64>)> {
        self.alphas
            .iter()
            .enumerate()
            .map(|(i, &alpha)| (alpha, self.coefs.row(i).to_owned()))
            .collect()
    }

    /// Find the alpha that gives k non-zero coefficients
    pub fn alpha_for_n_nonzero(&self, k: usize) -> Option<f64> {
        for (i, &n) in self.n_nonzeros.iter().enumerate() {
            if n >= k {
                return Some(self.alphas[i]);
            }
        }
        None
    }
}

/// Compute regularization path for Lasso
pub fn lasso_path(
    x: &Array2<f64>,
    y: &Array1<f64>,
    alphas: Option<&[f64]>,
    n_alphas: usize,
) -> Result<RegularizationPath> {
    let n = x.nrows();
    let p = x.ncols();

    // Generate alpha values if not provided
    let alphas = if let Some(a) = alphas {
        a.to_vec()
    } else {
        // Compute alpha_max (smallest alpha that gives all-zero solution)
        let y_mean = y.mean().unwrap_or(0.0);
        let y_centered = y - y_mean;
        let mut alpha_max = 0.0_f64;
        for j in 0..p {
            let correlation = x.column(j).dot(&y_centered).abs() / n as f64;
            alpha_max = alpha_max.max(correlation);
        }

        // Generate log-spaced alphas from alpha_max to alpha_max/1000
        let alpha_min = alpha_max / 1000.0;
        (0..n_alphas)
            .map(|i| {
                let t = i as f64 / (n_alphas - 1) as f64;
                alpha_max.ln().mul_add(1.0 - t, alpha_min.ln() * t).exp()
            })
            .collect()
    };

    let n_paths = alphas.len();
    let mut coefs = Array2::zeros((n_paths, p));
    let mut intercepts = Vec::with_capacity(n_paths);
    let mut n_nonzeros = Vec::with_capacity(n_paths);
    let mut r_squareds = Vec::with_capacity(n_paths);

    // Warm start: use previous solution as initialization
    let mut lasso = LassoRegression::new(alphas[0]);
    lasso.warm_start = true;

    for (i, &alpha) in alphas.iter().enumerate() {
        lasso.alpha = alpha;
        lasso.fit(x, y)?;

        if let Some(coef) = lasso.coefficients() {
            coefs.row_mut(i).assign(coef);
        }
        intercepts.push(lasso.intercept().unwrap_or(0.0));
        n_nonzeros.push(lasso.n_nonzero().unwrap_or(0));
        r_squareds.push(lasso.r_squared().unwrap_or(0.0));
    }

    Ok(RegularizationPath {
        alphas,
        coefs,
        intercepts,
        n_nonzeros,
        r_squared: Some(r_squareds),
    })
}

/// Compute regularization path for ElasticNet
pub fn elastic_net_path(
    x: &Array2<f64>,
    y: &Array1<f64>,
    l1_ratio: f64,
    alphas: Option<&[f64]>,
    n_alphas: usize,
) -> Result<RegularizationPath> {
    let n = x.nrows();
    let p = x.ncols();

    // Generate alpha values if not provided
    let alphas = if let Some(a) = alphas {
        a.to_vec()
    } else {
        let y_mean = y.mean().unwrap_or(0.0);
        let y_centered = y - y_mean;
        let mut alpha_max = 0.0_f64;
        for j in 0..p {
            let correlation = x.column(j).dot(&y_centered).abs() / (n as f64 * l1_ratio.max(0.01));
            alpha_max = alpha_max.max(correlation);
        }

        let alpha_min = alpha_max / 1000.0;
        (0..n_alphas)
            .map(|i| {
                let t = i as f64 / (n_alphas - 1) as f64;
                alpha_max.ln().mul_add(1.0 - t, alpha_min.ln() * t).exp()
            })
            .collect()
    };

    let n_paths = alphas.len();
    let mut coefs = Array2::zeros((n_paths, p));
    let mut intercepts = Vec::with_capacity(n_paths);
    let mut n_nonzeros = Vec::with_capacity(n_paths);
    let mut r_squareds = Vec::with_capacity(n_paths);

    let mut elastic = ElasticNet::new(alphas[0], l1_ratio);
    elastic.warm_start = true;

    for (i, &alpha) in alphas.iter().enumerate() {
        elastic.alpha = alpha;
        elastic.fit(x, y)?;

        if let Some(coef) = elastic.coefficients() {
            coefs.row_mut(i).assign(coef);
        }
        intercepts.push(elastic.intercept().unwrap_or(0.0));
        n_nonzeros.push(elastic.n_nonzero().unwrap_or(0));
        r_squareds.push(elastic.r_squared().unwrap_or(0.0));
    }

    Ok(RegularizationPath {
        alphas,
        coefs,
        intercepts,
        n_nonzeros,
        r_squared: Some(r_squareds),
    })
}

// =============================================================================
// Cross-Validated Model Selection
// =============================================================================

/// Ridge Regression with cross-validated alpha selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RidgeCV {
    /// Candidate alpha values
    pub alphas: Vec<f64>,
    /// Number of CV folds
    pub cv: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Feature names for reporting
    pub feature_names: Option<Vec<String>>,

    // Results
    best_alpha: Option<f64>,
    cv_scores: Option<Vec<f64>>,
    model: Option<RidgeRegression>,
}

impl RidgeCV {
    /// Create a new RidgeCV
    pub fn new(alphas: Vec<f64>, cv: usize) -> Self {
        Self {
            alphas,
            cv,
            fit_intercept: true,
            feature_names: None,
            best_alpha: None,
            cv_scores: None,
            model: None,
        }
    }

    /// Create with default alphas (log-spaced from 1e-4 to 1e4)
    pub fn with_defaults(cv: usize) -> Self {
        let alphas: Vec<f64> = (0..50)
            .map(|i| {
                let t = i as f64 / 49.0;
                (-4.0_f64).ln().mul_add(1.0 - t, 4.0_f64.ln() * t).exp()
            })
            .collect();
        Self::new(alphas, cv)
    }

    /// Get the best alpha found
    pub fn best_alpha(&self) -> Option<f64> {
        self.best_alpha
    }

    /// Get CV scores for each alpha
    pub fn cv_scores(&self) -> Option<&[f64]> {
        self.cv_scores.as_deref()
    }

    /// Get the underlying fitted model
    pub fn model(&self) -> Option<&RidgeRegression> {
        self.model.as_ref()
    }
}

impl Model for RidgeCV {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n = x.nrows();
        if n < self.cv {
            return Err(FerroError::invalid_input(format!(
                "Number of samples ({}) must be at least cv ({})",
                n, self.cv
            )));
        }

        // Cross-validation for each alpha
        let mut cv_scores = Vec::with_capacity(self.alphas.len());
        let fold_size = n / self.cv;

        for &alpha in &self.alphas {
            let mut fold_scores = Vec::with_capacity(self.cv);

            for fold in 0..self.cv {
                // Create train/test split
                let test_start = fold * fold_size;
                let test_end = if fold == self.cv - 1 {
                    n
                } else {
                    (fold + 1) * fold_size
                };

                let train_indices: Vec<usize> = (0..test_start).chain(test_end..n).collect();
                let test_indices: Vec<usize> = (test_start..test_end).collect();

                // Extract train/test data
                let x_train = x.select(Axis(0), &train_indices);
                let y_train = y.select(Axis(0), &train_indices);
                let x_test = x.select(Axis(0), &test_indices);
                let y_test = y.select(Axis(0), &test_indices);

                // Fit and score
                let mut ridge = RidgeRegression::new(alpha).with_fit_intercept(self.fit_intercept);
                ridge.fit(&x_train, &y_train)?;

                let predictions = ridge.predict(&x_test)?;
                let mse: f64 = predictions
                    .iter()
                    .zip(y_test.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / test_indices.len() as f64;

                fold_scores.push(-mse); // Negative MSE (higher is better)
            }

            let mean_score = fold_scores.iter().sum::<f64>() / self.cv as f64;
            cv_scores.push(mean_score);
        }

        // Find best alpha
        let best_idx = cv_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_alpha = self.alphas[best_idx];
        self.best_alpha = Some(best_alpha);
        self.cv_scores = Some(cv_scores);

        // Fit final model with best alpha
        let mut model = RidgeRegression::new(best_alpha).with_fit_intercept(self.fit_intercept);
        if let Some(ref names) = self.feature_names {
            model = model.with_feature_names(names.clone());
        }
        model.fit(x, y)?;
        self.model = Some(model);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.model
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?
            .predict(x)
    }

    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.model.as_ref()?.feature_importance()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.model.as_ref()?.n_features()
    }
}

/// Lasso Regression with cross-validated alpha selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LassoCV {
    /// Number of alphas to try (log-spaced)
    pub n_alphas: usize,
    /// Candidate alpha values (overrides n_alphas if provided)
    pub alphas: Option<Vec<f64>>,
    /// Number of CV folds
    pub cv: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Feature names for reporting
    pub feature_names: Option<Vec<String>>,

    // Results
    best_alpha: Option<f64>,
    cv_scores: Option<Vec<f64>>,
    alphas_used: Option<Vec<f64>>,
    model: Option<LassoRegression>,
}

impl LassoCV {
    /// Create a new LassoCV
    pub fn new(n_alphas: usize, cv: usize) -> Self {
        Self {
            n_alphas,
            alphas: None,
            cv,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            feature_names: None,
            best_alpha: None,
            cv_scores: None,
            alphas_used: None,
            model: None,
        }
    }

    /// Create with specific alpha values
    pub fn with_alphas(alphas: Vec<f64>, cv: usize) -> Self {
        Self {
            n_alphas: alphas.len(),
            alphas: Some(alphas),
            cv,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            feature_names: None,
            best_alpha: None,
            cv_scores: None,
            alphas_used: None,
            model: None,
        }
    }

    /// Get the best alpha found
    pub fn best_alpha(&self) -> Option<f64> {
        self.best_alpha
    }

    /// Get CV scores for each alpha
    pub fn cv_scores(&self) -> Option<&[f64]> {
        self.cv_scores.as_deref()
    }

    /// Get alphas that were used
    pub fn alphas_used(&self) -> Option<&[f64]> {
        self.alphas_used.as_deref()
    }

    /// Get the underlying fitted model
    pub fn model(&self) -> Option<&LassoRegression> {
        self.model.as_ref()
    }
}

impl Model for LassoCV {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n = x.nrows();
        let p = x.ncols();

        if n < self.cv {
            return Err(FerroError::invalid_input(format!(
                "Number of samples ({}) must be at least cv ({})",
                n, self.cv
            )));
        }

        // Generate alpha values if not provided
        let alphas = if let Some(ref a) = self.alphas {
            a.clone()
        } else {
            let y_mean = y.mean().unwrap_or(0.0);
            let y_centered = y - y_mean;
            let mut alpha_max = 0.0_f64;
            for j in 0..p {
                let correlation = x.column(j).dot(&y_centered).abs() / n as f64;
                alpha_max = alpha_max.max(correlation);
            }

            let alpha_min = alpha_max / 1000.0;
            (0..self.n_alphas)
                .map(|i| {
                    let t = i as f64 / (self.n_alphas - 1).max(1) as f64;
                    alpha_max.ln().mul_add(1.0 - t, alpha_min.ln() * t).exp()
                })
                .collect()
        };

        // Cross-validation for each alpha
        let mut cv_scores = Vec::with_capacity(alphas.len());
        let fold_size = n / self.cv;

        for &alpha in &alphas {
            let mut fold_scores = Vec::with_capacity(self.cv);

            for fold in 0..self.cv {
                let test_start = fold * fold_size;
                let test_end = if fold == self.cv - 1 {
                    n
                } else {
                    (fold + 1) * fold_size
                };

                let train_indices: Vec<usize> = (0..test_start).chain(test_end..n).collect();
                let test_indices: Vec<usize> = (test_start..test_end).collect();

                let x_train = x.select(Axis(0), &train_indices);
                let y_train = y.select(Axis(0), &train_indices);
                let x_test = x.select(Axis(0), &test_indices);
                let y_test = y.select(Axis(0), &test_indices);

                let mut lasso = LassoRegression::new(alpha)
                    .with_fit_intercept(self.fit_intercept)
                    .with_max_iter(self.max_iter)
                    .with_tol(self.tol);
                lasso.fit(&x_train, &y_train)?;

                let predictions = lasso.predict(&x_test)?;
                let mse: f64 = predictions
                    .iter()
                    .zip(y_test.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / test_indices.len() as f64;

                fold_scores.push(-mse);
            }

            let mean_score = fold_scores.iter().sum::<f64>() / self.cv as f64;
            cv_scores.push(mean_score);
        }

        // Find best alpha
        let best_idx = cv_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_alpha = alphas[best_idx];
        self.best_alpha = Some(best_alpha);
        self.cv_scores = Some(cv_scores);
        self.alphas_used = Some(alphas);

        // Fit final model
        let mut model = LassoRegression::new(best_alpha)
            .with_fit_intercept(self.fit_intercept)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol);
        if let Some(ref names) = self.feature_names {
            model = model.with_feature_names(names.clone());
        }
        model.fit(x, y)?;
        self.model = Some(model);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.model
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?
            .predict(x)
    }

    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.model.as_ref()?.feature_importance()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.model.as_ref()?.n_features()
    }
}

/// ElasticNet with cross-validated alpha and l1_ratio selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticNetCV {
    /// Number of alphas to try
    pub n_alphas: usize,
    /// L1 ratios to try
    pub l1_ratios: Vec<f64>,
    /// Number of CV folds
    pub cv: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Maximum iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Feature names for reporting
    pub feature_names: Option<Vec<String>>,

    // Results
    best_alpha: Option<f64>,
    best_l1_ratio: Option<f64>,
    cv_scores: Option<Vec<Vec<f64>>>,
    alphas_used: Option<Vec<f64>>,
    model: Option<ElasticNet>,
}

impl ElasticNetCV {
    /// Create a new ElasticNetCV
    pub fn new(n_alphas: usize, l1_ratios: Vec<f64>, cv: usize) -> Self {
        Self {
            n_alphas,
            l1_ratios,
            cv,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
            feature_names: None,
            best_alpha: None,
            best_l1_ratio: None,
            cv_scores: None,
            alphas_used: None,
            model: None,
        }
    }

    /// Create with default l1_ratios
    pub fn with_defaults(n_alphas: usize, cv: usize) -> Self {
        Self::new(n_alphas, vec![0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], cv)
    }

    /// Get the best alpha found
    pub fn best_alpha(&self) -> Option<f64> {
        self.best_alpha
    }

    /// Get the best l1_ratio found
    pub fn best_l1_ratio(&self) -> Option<f64> {
        self.best_l1_ratio
    }

    /// Get the underlying fitted model
    pub fn model(&self) -> Option<&ElasticNet> {
        self.model.as_ref()
    }
}

impl Model for ElasticNetCV {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n = x.nrows();
        let p = x.ncols();

        if n < self.cv {
            return Err(FerroError::invalid_input(format!(
                "Number of samples ({}) must be at least cv ({})",
                n, self.cv
            )));
        }

        let fold_size = n / self.cv;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_alpha = 1.0;
        let mut best_l1_ratio = 0.5;

        let mut all_cv_scores = Vec::new();
        let mut alphas_used = None;

        for &l1_ratio in &self.l1_ratios {
            // Generate alphas for this l1_ratio
            let y_mean = y.mean().unwrap_or(0.0);
            let y_centered = y - y_mean;
            let mut alpha_max = 0.0_f64;
            for j in 0..p {
                let correlation =
                    x.column(j).dot(&y_centered).abs() / (n as f64 * l1_ratio.max(0.01));
                alpha_max = alpha_max.max(correlation);
            }

            let alpha_min = alpha_max / 1000.0;
            let alphas: Vec<f64> = (0..self.n_alphas)
                .map(|i| {
                    let t = i as f64 / (self.n_alphas - 1).max(1) as f64;
                    alpha_max.ln().mul_add(1.0 - t, alpha_min.ln() * t).exp()
                })
                .collect();

            if alphas_used.is_none() {
                alphas_used = Some(alphas.clone());
            }

            let mut cv_scores = Vec::with_capacity(alphas.len());

            for &alpha in &alphas {
                let mut fold_scores = Vec::with_capacity(self.cv);

                for fold in 0..self.cv {
                    let test_start = fold * fold_size;
                    let test_end = if fold == self.cv - 1 {
                        n
                    } else {
                        (fold + 1) * fold_size
                    };

                    let train_indices: Vec<usize> = (0..test_start).chain(test_end..n).collect();
                    let test_indices: Vec<usize> = (test_start..test_end).collect();

                    let x_train = x.select(Axis(0), &train_indices);
                    let y_train = y.select(Axis(0), &train_indices);
                    let x_test = x.select(Axis(0), &test_indices);
                    let y_test = y.select(Axis(0), &test_indices);

                    let mut elastic = ElasticNet::new(alpha, l1_ratio)
                        .with_fit_intercept(self.fit_intercept)
                        .with_max_iter(self.max_iter)
                        .with_tol(self.tol);
                    elastic.fit(&x_train, &y_train)?;

                    let predictions = elastic.predict(&x_test)?;
                    let mse: f64 = predictions
                        .iter()
                        .zip(y_test.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f64>()
                        / test_indices.len() as f64;

                    fold_scores.push(-mse);
                }

                let mean_score = fold_scores.iter().sum::<f64>() / self.cv as f64;
                cv_scores.push(mean_score);

                if mean_score > best_score {
                    best_score = mean_score;
                    best_alpha = alpha;
                    best_l1_ratio = l1_ratio;
                }
            }

            all_cv_scores.push(cv_scores);
        }

        self.best_alpha = Some(best_alpha);
        self.best_l1_ratio = Some(best_l1_ratio);
        self.cv_scores = Some(all_cv_scores);
        self.alphas_used = alphas_used;

        // Fit final model
        let mut model = ElasticNet::new(best_alpha, best_l1_ratio)
            .with_fit_intercept(self.fit_intercept)
            .with_max_iter(self.max_iter)
            .with_tol(self.tol);
        if let Some(ref names) = self.feature_names {
            model = model.with_feature_names(names.clone());
        }
        model.fit(x, y)?;
        self.model = Some(model);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.model
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?
            .predict(x)
    }

    fn is_fitted(&self) -> bool {
        self.model.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.model.as_ref()?.feature_importance()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.model.as_ref()?.n_features()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Soft thresholding operator for L1 penalty
#[inline]
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Solve symmetric positive definite system Ax = b via Cholesky decomposition
fn solve_symmetric_positive_definite(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();

    // Cholesky decomposition: A = LL'
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(FerroError::numerical(
                        "Matrix is not positive definite (Cholesky failed)",
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }

    // Solve L*y = b
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }

    // Solve L'*x = y
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

/// Invert symmetric positive definite matrix
fn invert_symmetric(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut inv = Array2::zeros((n, n));

    // Solve A * inv_col = e_i for each column
    for i in 0..n {
        let mut e_i = Array1::zeros(n);
        e_i[i] = 1.0;
        let col = solve_symmetric_positive_definite(a, &e_i)?;
        inv.column_mut(i).assign(&col);
    }

    Ok(inv)
}

/// Compute diagonal of hat matrix H = X(X'X + αI)^(-1)X'
fn compute_hat_diagonal(x: &Array2<f64>, xtx_inv: &Array2<f64>) -> Array1<f64> {
    let n = x.nrows();
    let mut h_diag = Array1::zeros(n);

    for i in 0..n {
        let xi = x.row(i);
        h_diag[i] = xi.dot(&xtx_inv.dot(&xi.t()));
    }

    h_diag
}

/// Student's t critical value approximation
fn t_critical(p: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }

    if df == 1.0 {
        return (std::f64::consts::PI * (p - 0.5)).tan();
    }

    let a = 1.0 / (df - 0.5);
    let b = 48.0 / (a * a);
    let c = (20700.0 * a / b - 98.0).mul_add(a, -16.0).mul_add(a, 96.36);
    let d = ((94.5 / (b + c) - 3.0) / b + 1.0) * (a * std::f64::consts::PI * 0.5).sqrt() * df;

    let x = d * p;
    let mut y = x.powf(2.0 / df);

    if y > 0.05 + a {
        let x_norm = z_inv_normal(p);
        y = x_norm * x_norm;

        let c = (0.05 * d)
            .mul_add(x_norm, -5.0)
            .mul_add(x_norm, -7.0)
            .mul_add(x_norm, -2.0)
            .mul_add(x_norm, b)
            + c;
        y = ((0.4f64.mul_add(y, 6.3).mul_add(y, 36.0).mul_add(y, 94.5) / c - y - 3.0) / b + 1.0)
            * x_norm;
        y = (a * y * y).exp_m1();
    } else {
        y = (1.0 / ((0.089f64.mul_add(-d, (df + 6.0) / (df * y)) - 0.822) * (df + 2.0) * 3.0)
            + 0.5 / (df + 4.0))
            .mul_add(y, -1.0)
            * (df + 1.0)
            / (df + 2.0)
            + 1.0 / y;
    }

    (df * y).sqrt()
}

/// Inverse normal CDF approximation
fn z_inv_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let p = if p > 0.5 { 1.0 - p } else { p };

    let t = (-2.0 * p.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t
        - (c2 * t).mul_add(t, c0 + c1 * t)
            / (d3 * t * t).mul_add(t, (d2 * t).mul_add(t, 1.0 + d1 * t));

    if p > 0.5 {
        -z
    } else {
        z
    }
}

/// Student's t CDF approximation
fn t_cdf_approx(t: f64, df: f64) -> f64 {
    let x = df / t.mul_add(t, df);
    0.5f64.mul_add(
        (1.0 - incomplete_beta_regularized(df / 2.0, 0.5, x)).copysign(t),
        0.5,
    )
}

/// Regularized incomplete beta function
fn incomplete_beta_regularized(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        b.mul_add(
            (1.0 - x).ln(),
            a.mul_add(x.ln(), ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)),
        )
        .exp()
    };

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for incomplete beta
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-12;
    let fpmin = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;

        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Log gamma function
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];

    let tmp = x + 5.5;
    let tmp = (x + 0.5).mul_add(-tmp.ln(), tmp);

    let mut ser = 1.000_000_000_190_015;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (x + i as f64 + 1.0);
    }

    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_ridge_regression_simple() {
        // y = 1 + 2*x with small ridge penalty
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut ridge = RidgeRegression::new(0.01);
        ridge.fit(&x, &y).unwrap();

        // Should be close to OLS solution
        let coef = ridge.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 0.1);

        let preds = ridge.predict(&x).unwrap();
        for (i, &p) in preds.iter().enumerate() {
            assert_relative_eq!(p, y[i], epsilon = 0.5);
        }
    }

    #[test]
    fn test_ridge_regularization_shrinks_coefficients() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = (0..10)
            .map(|i| (i * 2) as f64 + (i * 2 + 1) as f64)
            .collect();

        let mut ridge_weak = RidgeRegression::new(0.001);
        let mut ridge_strong = RidgeRegression::new(100.0);

        ridge_weak.fit(&x, &y).unwrap();
        ridge_strong.fit(&x, &y).unwrap();

        let coef_weak = ridge_weak.coefficients().unwrap();
        let coef_strong = ridge_strong.coefficients().unwrap();

        // Strong regularization should shrink coefficients more
        let norm_weak: f64 = coef_weak.iter().map(|c| c.powi(2)).sum();
        let norm_strong: f64 = coef_strong.iter().map(|c| c.powi(2)).sum();

        assert!(norm_strong < norm_weak);
    }

    #[test]
    fn test_lasso_regression_simple() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut lasso = LassoRegression::new(0.01);
        lasso.fit(&x, &y).unwrap();

        let coef = lasso.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 0.2);
    }

    #[test]
    fn test_lasso_sparsity() {
        // Create data where only first feature is relevant
        let n = 50;
        let p = 10;
        let mut x_data = Vec::with_capacity(n * p);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            for j in 0..p {
                x_data.push(if j == 0 {
                    i as f64
                } else {
                    (i * j) as f64 * 0.01
                });
            }
            y_data.push(2.0 * (i as f64) + 1.0);
        }

        let x = Array2::from_shape_vec((n, p), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let mut lasso = LassoRegression::new(0.5);
        lasso.fit(&x, &y).unwrap();

        // First coefficient should be significant, others should be near zero
        let coef = lasso.coefficients().unwrap();
        assert!(coef[0].abs() > 0.1);

        // With enough regularization, many coefficients should be zero
        let n_nonzero = lasso.n_nonzero().unwrap();
        assert!(n_nonzero < p);
    }

    #[test]
    fn test_elastic_net_simple() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut elastic = ElasticNet::new(0.01, 0.5);
        elastic.fit(&x, &y).unwrap();

        let coef = elastic.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 0.2);
    }

    #[test]
    fn test_elastic_net_l1_ratio_extremes() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = (0..10).map(|i| 2.0 * (i * 2) as f64 + 1.0).collect();

        // l1_ratio = 0 should behave like ridge
        let mut elastic_ridge = ElasticNet::new(1.0, 0.0);
        elastic_ridge.fit(&x, &y).unwrap();

        // l1_ratio = 1 should behave like lasso
        let mut elastic_lasso = ElasticNet::new(1.0, 1.0);
        elastic_lasso.fit(&x, &y).unwrap();

        // Both should produce valid results
        assert!(elastic_ridge.r_squared().unwrap() > 0.0);
        assert!(elastic_lasso.r_squared().unwrap() > 0.0);
    }

    #[test]
    fn test_lasso_path() {
        let x = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = (0..20).map(|i| 2.0 * (i * 3) as f64 + 1.0).collect();

        let path = lasso_path(&x, &y, None, 10).unwrap();

        assert_eq!(path.alphas.len(), 10);
        assert_eq!(path.coefs.nrows(), 10);
        assert_eq!(path.coefs.ncols(), 3);

        // Alphas should be decreasing
        for i in 1..path.alphas.len() {
            assert!(path.alphas[i] < path.alphas[i - 1]);
        }

        // More coefficients should become non-zero as alpha decreases
        let first_nonzero = path.n_nonzeros[0];
        let last_nonzero = path.n_nonzeros[path.n_nonzeros.len() - 1];
        assert!(last_nonzero >= first_nonzero);
    }

    #[test]
    fn test_ridge_cv() {
        let x = Array2::from_shape_vec((30, 2), (0..60).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = (0..30).map(|i| 2.0 * (i * 2) as f64 + 1.0).collect();

        let alphas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let mut ridge_cv = RidgeCV::new(alphas, 5);
        ridge_cv.fit(&x, &y).unwrap();

        assert!(ridge_cv.best_alpha().is_some());
        assert!(ridge_cv.is_fitted());

        let preds = ridge_cv.predict(&x).unwrap();
        assert_eq!(preds.len(), 30);
    }

    #[test]
    fn test_lasso_cv() {
        let x = Array2::from_shape_vec((30, 2), (0..60).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = (0..30).map(|i| 2.0 * (i * 2) as f64 + 1.0).collect();

        let mut lasso_cv = LassoCV::new(20, 5);
        lasso_cv.fit(&x, &y).unwrap();

        assert!(lasso_cv.best_alpha().is_some());
        assert!(lasso_cv.is_fitted());
    }

    #[test]
    fn test_elastic_net_cv() {
        let x = Array2::from_shape_vec((30, 2), (0..60).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = (0..30).map(|i| 2.0 * (i * 2) as f64 + 1.0).collect();

        let mut elastic_cv = ElasticNetCV::with_defaults(10, 5);
        elastic_cv.fit(&x, &y).unwrap();

        assert!(elastic_cv.best_alpha().is_some());
        assert!(elastic_cv.best_l1_ratio().is_some());
        assert!(elastic_cv.is_fitted());
    }

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_eq!(soft_threshold(1.5, 2.0), 0.0);
        assert_eq!(soft_threshold(-1.5, 2.0), 0.0);
        assert_eq!(soft_threshold(0.0, 2.0), 0.0);
    }

    #[test]
    fn test_ridge_summary() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = (0..10).map(|i| 2.0 * (i * 2) as f64 + 1.0).collect();

        let mut ridge =
            RidgeRegression::new(0.1).with_feature_names(vec!["var1".into(), "var2".into()]);
        ridge.fit(&x, &y).unwrap();

        let summary = ridge.summary();
        let output = format!("{}", summary);

        assert!(output.contains("Ridge Regression"));
        assert!(output.contains("R-squared"));
        assert!(output.contains("var1"));
        assert!(output.contains("var2"));
    }

    #[test]
    fn test_lasso_summary() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = (0..10).map(|i| 2.0 * (i * 2) as f64 + 1.0).collect();

        let mut lasso = LassoRegression::new(0.1);
        lasso.fit(&x, &y).unwrap();

        let summary = lasso.summary();
        let output = format!("{}", summary);

        assert!(output.contains("Lasso Regression"));
        assert!(output.contains("Non-zero coefficients"));
    }

    #[test]
    fn test_not_fitted_error() {
        let ridge = RidgeRegression::new(1.0);
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        assert!(ridge.predict(&x).is_err());
    }

    #[test]
    fn test_invalid_alpha() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut ridge = RidgeRegression::new(-1.0);
        assert!(ridge.fit(&x, &y).is_err());
    }

    #[test]
    fn test_invalid_l1_ratio() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut elastic = ElasticNet::new(1.0, 1.5);
        assert!(elastic.fit(&x, &y).is_err());
    }
}
