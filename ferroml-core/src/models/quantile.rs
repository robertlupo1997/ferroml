//! Quantile Regression with Statistical Inference
//!
//! This module provides quantile regression for estimating conditional quantiles
//! of the response variable - FerroML's extension beyond mean-based regression.
//!
//! ## Features
//!
//! - **IRLS fitting** using iteratively reweighted least squares
//! - **Multiple quantile estimation**: fit 0.25, 0.5, 0.75 simultaneously
//! - **Coefficient inference**: bootstrap-based standard errors and CIs
//! - **Robust to outliers**: less sensitive than OLS to extreme values
//! - **Heteroscedasticity handling**: captures different relationships across the distribution
//!
//! ## When to Use
//!
//! - Data with heteroscedastic errors (variance changes with X)
//! - Interest in tails of the distribution (not just the mean)
//! - Presence of outliers that would distort OLS estimates
//! - Understanding the full conditional distribution
//!
//! ## Example
//!
//! ```
//! use ferroml_core::models::quantile::QuantileRegression;
//! use ferroml_core::models::{Model, StatisticalModel};
//! use ndarray::{Array1, Array2};
//!
//! # fn main() -> ferroml_core::Result<()> {
//! let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| (i as f64 * 0.1).sin()).collect()).unwrap();
//! let y = Array1::from_vec((0..100).map(|i| (i as f64 * 0.3).cos() + 1.0).collect());
//!
//! // Fit median regression (tau = 0.5)
//! let mut model = QuantileRegression::new(0.5);
//! model.fit(&x, &y).unwrap();
//!
//! // Get R-style summary with bootstrap CIs
//! println!("{}", model.summary());
//!
//! // Fit multiple quantiles at once
//! let results = QuantileRegression::fit_quantiles(&x, &y, &[0.25, 0.5, 0.75]).unwrap();
//! # Ok(())
//! # }
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, CoefficientInfo, Diagnostics,
    FitStatistics, Model, ModelSummary, PredictionInterval, ProbabilisticModel, ResidualStatistics,
    StatisticalModel,
};
use crate::{FerroError, Result};
use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};

/// Quantile Regression Model
///
/// Estimates conditional quantiles of the response variable given the predictors.
/// Unlike OLS which minimizes squared error (estimating the conditional mean),
/// quantile regression minimizes the asymmetric "pinball" loss to estimate
/// arbitrary quantiles.
///
/// ## Quantile Loss Function
///
/// For quantile τ ∈ (0, 1), the loss is:
/// ```text
/// ρ_τ(u) = u(τ - I(u < 0)) = { τ|u|      if u ≥ 0
///                            { (1-τ)|u|  if u < 0
/// ```
///
/// This weights positive and negative residuals asymmetrically.
///
/// ## Interpretation
///
/// - τ = 0.5: Median regression (robust to outliers)
/// - τ = 0.1: 10th percentile (lower tail)
/// - τ = 0.9: 90th percentile (upper tail)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileRegression {
    /// Quantile to estimate (τ ∈ (0, 1))
    pub quantile: f64,
    /// Whether to include an intercept term
    pub fit_intercept: bool,
    /// Maximum iterations for IRLS
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Number of bootstrap samples for inference
    pub n_bootstrap: usize,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Feature names (optional, for reporting)
    pub feature_names: Option<Vec<String>>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,

    // Fitted parameters
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features: Option<usize>,
    fitted_data: Option<FittedQuantileData>,
}

/// Internal struct to store fitted data for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FittedQuantileData {
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
    /// Quantile loss value
    quantile_loss: f64,
    /// Bootstrap coefficient samples (for inference)
    bootstrap_coefs: Option<Array2<f64>>,
    /// Coefficient standard errors
    coef_std_errors: Array1<f64>,
    /// Number of iterations to converge
    n_iter: usize,
}

/// Results from fitting multiple quantiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiQuantileResults {
    /// Quantile values fitted
    pub quantiles: Vec<f64>,
    /// Coefficient estimates for each quantile (including intercept if fitted)
    /// Shape: (n_quantiles, n_coefficients)
    pub coefficients: Array2<f64>,
    /// Standard errors for each quantile
    /// Shape: (n_quantiles, n_coefficients)
    pub std_errors: Array2<f64>,
    /// Whether intercept was fitted
    pub fit_intercept: bool,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl Default for QuantileRegression {
    fn default() -> Self {
        Self::new(0.5) // Default to median regression
    }
}

impl QuantileRegression {
    /// Create a new QuantileRegression model
    ///
    /// # Arguments
    /// * `quantile` - The quantile to estimate, must be in (0, 1)
    ///
    /// # Panics
    /// Panics if quantile is not in (0, 1)
    pub fn new(quantile: f64) -> Self {
        assert!(
            quantile > 0.0 && quantile < 1.0,
            "Quantile must be in (0, 1), got {}",
            quantile
        );

        Self {
            quantile,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-6,
            n_bootstrap: 200,
            confidence_level: 0.95,
            feature_names: None,
            random_state: None,
            coefficients: None,
            intercept: None,
            n_features: None,
            fitted_data: None,
        }
    }

    /// Create a median regression model (quantile = 0.5)
    pub fn median() -> Self {
        Self::new(0.5)
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

    /// Set number of bootstrap samples
    pub fn with_n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        self.n_bootstrap = n_bootstrap;
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

    /// Set random seed for reproducibility
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
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

    /// Get the quantile loss (objective function value)
    pub fn quantile_loss(&self) -> Option<f64> {
        self.fitted_data.as_ref().map(|d| d.quantile_loss)
    }

    /// Fit multiple quantiles simultaneously
    ///
    /// This is more efficient than fitting each quantile separately when
    /// you need estimates at multiple quantile levels.
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    /// * `quantiles` - Slice of quantiles to estimate (each in (0, 1))
    ///
    /// # Returns
    /// `MultiQuantileResults` containing coefficients and standard errors for all quantiles
    pub fn fit_quantiles(
        x: &Array2<f64>,
        y: &Array1<f64>,
        quantiles: &[f64],
    ) -> Result<MultiQuantileResults> {
        Self::fit_quantiles_with_options(x, y, quantiles, true, 200, None)
    }

    /// Fit multiple quantiles with custom options
    pub fn fit_quantiles_with_options(
        x: &Array2<f64>,
        y: &Array1<f64>,
        quantiles: &[f64],
        fit_intercept: bool,
        n_bootstrap: usize,
        feature_names: Option<Vec<String>>,
    ) -> Result<MultiQuantileResults> {
        validate_fit_input(x, y)?;

        for &q in quantiles {
            if q <= 0.0 || q >= 1.0 {
                return Err(FerroError::invalid_input(format!(
                    "Quantile must be in (0, 1), got {}",
                    q
                )));
            }
        }

        let n_coef = if fit_intercept {
            x.ncols() + 1
        } else {
            x.ncols()
        };
        let n_quantiles = quantiles.len();

        let mut coefficients = Array2::zeros((n_quantiles, n_coef));
        let mut std_errors = Array2::zeros((n_quantiles, n_coef));

        for (i, &q) in quantiles.iter().enumerate() {
            let mut model = QuantileRegression::new(q)
                .with_fit_intercept(fit_intercept)
                .with_n_bootstrap(n_bootstrap);

            if let Some(ref names) = feature_names {
                model = model.with_feature_names(names.clone());
            }

            model.fit(x, y)?;

            // Extract coefficients
            let all_coef = model.all_coefficients().unwrap();
            coefficients.row_mut(i).assign(&all_coef);

            // Extract standard errors
            if let Some(ref data) = model.fitted_data {
                std_errors.row_mut(i).assign(&data.coef_std_errors);
            }
        }

        Ok(MultiQuantileResults {
            quantiles: quantiles.to_vec(),
            coefficients,
            std_errors,
            fit_intercept,
            feature_names,
        })
    }

    /// Compute the quantile (pinball) loss for given residuals
    fn compute_quantile_loss(&self, residuals: &Array1<f64>) -> f64 {
        let tau = self.quantile;
        residuals
            .iter()
            .map(|&r| if r >= 0.0 { tau * r } else { (tau - 1.0) * r })
            .sum()
    }

    /// IRLS (Iteratively Reweighted Least Squares) fitting
    ///
    /// Uses the MM algorithm with Huber-like smoothing for stability.
    fn fit_irls(&self, x_design: &Array2<f64>, y: &Array1<f64>) -> Result<(Array1<f64>, usize)> {
        let n = x_design.nrows();
        let tau = self.quantile;

        // Initialize with OLS solution
        let mut beta = self.ols_init(x_design, y)?;

        // Small epsilon for numerical stability
        let eps = 1e-6;

        for iter in 0..self.max_iter {
            // Compute residuals
            let fitted = x_design.dot(&beta);
            let residuals: Array1<f64> = y
                .iter()
                .zip(fitted.iter())
                .map(|(&yi, &fi)| yi - fi)
                .collect();

            // Compute weights for IRLS
            // w_i = 1 / max(|r_i|, eps) with asymmetric adjustment
            let mut weights = Array1::zeros(n);
            for i in 0..n {
                let r = residuals[i];
                let abs_r = r.abs().max(eps);
                // Asymmetric weights based on sign of residual
                if r >= 0.0 {
                    weights[i] = tau / abs_r;
                } else {
                    weights[i] = (1.0 - tau) / abs_r;
                }
            }

            // Weighted least squares: solve (X'WX)β = X'Wy
            // X'WX = (W^½X)' @ (W^½X)
            let mut scaled_x = x_design.to_owned();
            for i in 0..n {
                let w_sqrt = weights[i].max(0.0).sqrt();
                scaled_x.row_mut(i).mapv_inplace(|v| v * w_sqrt);
            }
            let xtwx = scaled_x.t().dot(&scaled_x);

            // X'Wy = X' @ (W ⊙ y)
            let wy: Array1<f64> = &weights * y;
            let xtwy = x_design.t().dot(&wy);

            // Solve via Cholesky
            let beta_new = self.solve_symmetric(&xtwx, &xtwy)?;

            // Check convergence
            let diff: f64 = beta
                .iter()
                .zip(beta_new.iter())
                .map(|(&b, &bn)| (b - bn).abs())
                .sum();

            beta = beta_new;

            if diff < self.tol {
                return Ok((beta, iter + 1));
            }
        }

        Ok((beta, self.max_iter))
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

    /// Bootstrap inference for coefficient standard errors
    fn bootstrap_inference(
        &self,
        x_design: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n = x_design.nrows();
        let p = x_design.ncols();

        let mut bootstrap_coefs = Array2::zeros((self.n_bootstrap, p));

        // Simple LCG for reproducibility
        let mut rng_state = self.random_state.unwrap_or(42);

        for b in 0..self.n_bootstrap {
            // Bootstrap sample indices
            let indices: Vec<usize> = (0..n)
                .map(|_| {
                    rng_state = lcg_next(rng_state);
                    (rng_state % n as u64) as usize
                })
                .collect();

            // Create bootstrap sample
            let mut x_boot = Array2::zeros((n, p));
            let mut y_boot = Array1::zeros(n);

            for (i, &idx) in indices.iter().enumerate() {
                x_boot.row_mut(i).assign(&x_design.row(idx));
                y_boot[i] = y[idx];
            }

            // Fit on bootstrap sample
            if let Ok((beta, _)) = self.fit_irls(&x_boot, &y_boot) {
                bootstrap_coefs.row_mut(b).assign(&beta);
            }
        }

        // Compute standard errors as std of bootstrap distribution
        let mut std_errors = Array1::zeros(p);
        for j in 0..p {
            let col = bootstrap_coefs.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let variance: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (self.n_bootstrap - 1) as f64;
            std_errors[j] = variance.sqrt();
        }

        Ok((bootstrap_coefs, std_errors))
    }

    fn get_feature_name(&self, idx: usize) -> String {
        super::get_feature_name(&self.feature_names, idx)
    }
}

/// Simple LCG random number generator
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

impl Model for QuantileRegression {
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

        // Fit via IRLS
        let (beta, n_iter) = self.fit_irls(&x_design, y)?;

        // Compute fitted values and residuals
        let fitted_values = x_design.dot(&beta);
        let residuals: Array1<f64> = y
            .iter()
            .zip(fitted_values.iter())
            .map(|(&yi, &fi)| yi - fi)
            .collect();

        // Compute quantile loss
        let quantile_loss = self.compute_quantile_loss(&residuals);

        // Bootstrap inference
        let (bootstrap_coefs, coef_std_errors) = self.bootstrap_inference(&x_design, y)?;

        // Store fitted data
        self.fitted_data = Some(FittedQuantileData {
            n_samples: n,
            n_features: p_orig,
            residuals,
            fitted_values,
            y: y.clone(),
            quantile_loss,
            bootstrap_coefs: Some(bootstrap_coefs),
            coef_std_errors,
            n_iter,
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
        // For quantile regression, use absolute coefficients normalized by std errors
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
        // Quantile regression has no hyperparameters to tune
        // (quantile itself is a modeling choice, not a hyperparameter)
        SearchSpace::new()
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl StatisticalModel for QuantileRegression {
    fn summary(&self) -> ModelSummary {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => {
                return ModelSummary::new(
                    format!("Quantile Regression (τ={:.2}, not fitted)", self.quantile),
                    0,
                    0,
                );
            }
        };

        let mut summary = ModelSummary::new(
            format!("Quantile Regression (τ={:.2})", self.quantile),
            data.n_samples,
            data.n_features,
        );

        // Add fit statistics
        let mut fit_stats = FitStatistics::default();
        fit_stats.df_model = Some(data.n_features + if self.fit_intercept { 1 } else { 0 });
        fit_stats.df_residuals = Some(data.n_samples - fit_stats.df_model.unwrap());

        // Pseudo R-squared for quantile regression (Koenker & Machado, 1999)
        // R1(τ) = 1 - V(τ̂) / V(τ̃) where V is the quantile loss
        // and τ̃ is the intercept-only model
        let null_loss = self.compute_null_loss(&data.y);
        if null_loss > 0.0 {
            let pseudo_r2 = 1.0 - data.quantile_loss / null_loss;
            fit_stats.r_squared = Some(pseudo_r2);
        }

        summary = summary.with_fit_statistics(fit_stats);

        // Add coefficients with bootstrap CIs
        let coefs = self.coefficients_with_ci(self.confidence_level);
        for coef in coefs {
            summary.add_coefficient(coef);
        }

        // Add note about bootstrap inference
        summary.add_note(format!(
            "Standard errors computed via bootstrap ({} samples)",
            self.n_bootstrap
        ));
        summary.add_note(format!("Converged in {} iterations", data.n_iter));

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

        let alpha = 1.0 - level;
        let bootstrap_coefs = match &data.bootstrap_coefs {
            Some(bc) => bc,
            None => return result,
        };

        // Add intercept if fitted
        if self.fit_intercept {
            let intercept = self.intercept.unwrap_or(0.0);
            let se = data.coef_std_errors[0];

            // Bootstrap percentile CI
            let (ci_lower, ci_upper) =
                bootstrap_percentile_ci(&bootstrap_coefs.column(0).to_owned(), alpha);

            // Z-test for asymptotic inference
            let z_stat = if se > 0.0 { intercept / se } else { 0.0 };
            let p_value = 2.0 * (1.0 - standard_normal_cdf(z_stat.abs()));

            result.push(
                CoefficientInfo::new("(Intercept)", intercept, se)
                    .with_test(z_stat, p_value)
                    .with_ci(ci_lower, ci_upper, level),
            );
        }

        // Add feature coefficients
        if let Some(coef) = &self.coefficients {
            for (i, &c) in coef.iter().enumerate() {
                let se_idx = if self.fit_intercept { i + 1 } else { i };
                let se = data.coef_std_errors[se_idx];

                // Bootstrap percentile CI
                let (ci_lower, ci_upper) =
                    bootstrap_percentile_ci(&bootstrap_coefs.column(se_idx).to_owned(), alpha);

                // Z-test
                let z_stat = if se > 0.0 { c / se } else { 0.0 };
                let p_value = 2.0 * (1.0 - standard_normal_cdf(z_stat.abs()));

                let name = self.get_feature_name(i);
                result.push(
                    CoefficientInfo::new(name, c, se)
                        .with_test(z_stat, p_value)
                        .with_ci(ci_lower, ci_upper, level),
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

    fn assumption_test(
        &self,
        _assumption: crate::models::Assumption,
    ) -> Option<crate::models::AssumptionTestResult> {
        // Quantile regression doesn't have the same assumptions as OLS
        // No normality, homoscedasticity assumptions
        None
    }
}

impl ProbabilisticModel for QuantileRegression {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // For quantile regression, return predictions as (n, 1) matrix
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

        // Use bootstrap for prediction intervals
        let bootstrap_coefs = match &data.bootstrap_coefs {
            Some(bc) => bc,
            None => {
                // Fallback: use residual-based interval
                let residual_iqr = compute_iqr(&data.residuals);
                let margin = residual_iqr * 1.5; // Heuristic
                let lower = &predictions - margin;
                let upper = &predictions + margin;
                return Ok(PredictionInterval::new(predictions, lower, upper, level));
            }
        };

        // Bootstrap prediction intervals
        let p = if self.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        let mut pred_matrix = Array2::zeros((self.n_bootstrap, n));

        for b in 0..self.n_bootstrap {
            let beta = bootstrap_coefs.row(b);

            for i in 0..n {
                let mut pred = if self.fit_intercept { beta[0] } else { 0.0 };
                for j in 0..n_features {
                    let beta_idx = if self.fit_intercept { j + 1 } else { j };
                    if beta_idx < p {
                        pred += beta[beta_idx] * x[[i, j]];
                    }
                }
                pred_matrix[[b, i]] = pred;
            }
        }

        // Compute percentile intervals for each prediction
        let alpha = 1.0 - level;
        let mut lower = Array1::zeros(n);
        let mut upper = Array1::zeros(n);

        for i in 0..n {
            let mut preds_i: Vec<f64> = pred_matrix.column(i).iter().copied().collect();
            preds_i.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let lower_idx = ((alpha / 2.0) * preds_i.len() as f64) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * preds_i.len() as f64) as usize;

            lower[i] = preds_i[lower_idx.min(preds_i.len() - 1)];
            upper[i] = preds_i[upper_idx.min(preds_i.len() - 1)];
        }

        Ok(PredictionInterval::new(predictions, lower, upper, level))
    }
}

impl QuantileRegression {
    /// Compute quantile loss for null (intercept-only) model
    fn compute_null_loss(&self, y: &Array1<f64>) -> f64 {
        // Find the τ-th quantile of y
        let mut sorted: Vec<f64> = y.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (self.quantile * (sorted.len() - 1) as f64) as usize;
        let quantile_val = sorted[idx];

        // Compute loss with just the quantile as prediction
        y.iter()
            .map(|&yi| {
                let r = yi - quantile_val;
                if r >= 0.0 {
                    self.quantile * r
                } else {
                    (self.quantile - 1.0) * r
                }
            })
            .sum()
    }
}

/// Bootstrap percentile confidence interval
fn bootstrap_percentile_ci(samples: &Array1<f64>, alpha: f64) -> (f64, f64) {
    let mut sorted: Vec<f64> = samples.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 0 {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }

    let lower_idx = ((alpha / 2.0) * n as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64) as usize;

    (sorted[lower_idx.min(n - 1)], sorted[upper_idx.min(n - 1)])
}

/// Standard normal CDF approximation
fn standard_normal_cdf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let t = 1.0 / 0.2316419f64.mul_add(x.abs(), 1.0);
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

/// Compute interquartile range
fn compute_iqr(data: &Array1<f64>) -> f64 {
    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n < 4 {
        return sorted.last().copied().unwrap_or(0.0) - sorted.first().copied().unwrap_or(0.0);
    }

    let q1_idx = n / 4;
    let q3_idx = 3 * n / 4;

    sorted[q3_idx] - sorted[q1_idx]
}

impl MultiQuantileResults {
    /// Get coefficient for a specific quantile and feature
    ///
    /// # Arguments
    /// * `quantile_idx` - Index of the quantile
    /// * `feature_idx` - Index of the feature (0 = intercept if fit_intercept)
    pub fn get_coefficient(&self, quantile_idx: usize, feature_idx: usize) -> Option<f64> {
        if quantile_idx < self.quantiles.len() && feature_idx < self.coefficients.ncols() {
            Some(self.coefficients[[quantile_idx, feature_idx]])
        } else {
            None
        }
    }

    /// Get standard error for a specific quantile and feature
    pub fn get_std_error(&self, quantile_idx: usize, feature_idx: usize) -> Option<f64> {
        if quantile_idx < self.quantiles.len() && feature_idx < self.std_errors.ncols() {
            Some(self.std_errors[[quantile_idx, feature_idx]])
        } else {
            None
        }
    }

    /// Get coefficients for a specific feature across all quantiles
    pub fn coefficients_for_feature(&self, feature_idx: usize) -> Option<Array1<f64>> {
        if feature_idx < self.coefficients.ncols() {
            Some(self.coefficients.column(feature_idx).to_owned())
        } else {
            None
        }
    }

    /// Get the feature name for an index
    pub fn feature_name(&self, idx: usize) -> String {
        if self.fit_intercept {
            if idx == 0 {
                return "(Intercept)".to_string();
            }
            let feature_idx = idx - 1;
            if let Some(ref names) = self.feature_names {
                if feature_idx < names.len() {
                    return names[feature_idx].clone();
                }
            }
            format!("x{}", feature_idx + 1)
        } else {
            if let Some(ref names) = self.feature_names {
                if idx < names.len() {
                    return names[idx].clone();
                }
            }
            format!("x{}", idx + 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_quantile_regression_median() {
        // y = 1 + 2*x with no noise
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = QuantileRegression::median().with_n_bootstrap(50);
        model.fit(&x, &y).unwrap();

        // Check coefficients (should be close to true values for exact linear data)
        assert_relative_eq!(model.intercept().unwrap(), 1.0, epsilon = 0.5);
        let coef = model.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_quantile_regression_with_noise() {
        // Create data with heteroscedastic errors
        let n = 100;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        let mut rng = 12345u64;
        for i in 0..n {
            let xi = (i as f64) / 10.0;
            x_data.push(xi);

            // Add noise that increases with x
            rng = lcg_next(rng);
            let noise = ((rng % 1000) as f64 / 1000.0 - 0.5) * xi;
            y_data.push(1.0 + 2.0 * xi + noise);
        }

        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        let mut model = QuantileRegression::new(0.5).with_n_bootstrap(50);
        model.fit(&x, &y).unwrap();

        // Model should fit
        assert!(model.is_fitted());

        // Quantile loss should be computed
        assert!(model.quantile_loss().unwrap() > 0.0);
    }

    #[test]
    fn test_different_quantiles() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        // Lower quantile
        let mut lower = QuantileRegression::new(0.25).with_n_bootstrap(50);
        lower.fit(&x, &y).unwrap();

        // Upper quantile
        let mut upper = QuantileRegression::new(0.75).with_n_bootstrap(50);
        upper.fit(&x, &y).unwrap();

        // For exact linear data, all quantiles should give similar coefficients
        assert_relative_eq!(
            lower.coefficients().unwrap()[0],
            upper.coefficients().unwrap()[0],
            epsilon = 0.5
        );
    }

    #[test]
    fn test_fit_multiple_quantiles() {
        let x = Array2::from_shape_vec((30, 1), (1..=30).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let results = QuantileRegression::fit_quantiles_with_options(
            &x,
            &y,
            &[0.25, 0.5, 0.75],
            true,
            50,
            None,
        )
        .unwrap();

        assert_eq!(results.quantiles.len(), 3);
        assert_eq!(results.coefficients.nrows(), 3);
        assert_eq!(results.coefficients.ncols(), 2); // intercept + 1 feature
    }

    #[test]
    fn test_without_intercept() {
        // y = 2*x (through origin)
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 * xi);

        let mut model = QuantileRegression::median()
            .with_fit_intercept(false)
            .with_n_bootstrap(50);
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 0.2);
    }

    #[test]
    fn test_predict() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = QuantileRegression::median().with_n_bootstrap(50);
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

        let mut model = QuantileRegression::median().with_n_bootstrap(50);
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

        let mut model = QuantileRegression::median()
            .with_n_bootstrap(50)
            .with_feature_names(vec!["feature1".to_string()]);
        model.fit(&x, &y).unwrap();

        let summary = model.summary();
        let output = format!("{}", summary);

        assert!(output.contains("Quantile Regression"));
        assert!(output.contains("(Intercept)"));
        assert!(output.contains("feature1"));
    }

    #[test]
    fn test_coefficients_with_ci() {
        let x = Array2::from_shape_vec((30, 1), (1..=30).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = QuantileRegression::median().with_n_bootstrap(100);
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
    fn test_residuals() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 1.0 + 2.0 * xi);

        let mut model = QuantileRegression::median().with_n_bootstrap(50);
        model.fit(&x, &y).unwrap();

        let residuals = model.residuals().unwrap();
        let fitted = model.fitted_values().unwrap();

        // residuals + fitted should approximately equal y
        for i in 0..y.len() {
            assert_relative_eq!(residuals[i] + fitted[i], y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_error_not_fitted() {
        let model = QuantileRegression::median();
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_error_invalid_quantile() {
        // This should panic
        let result = std::panic::catch_unwind(|| QuantileRegression::new(1.5));
        assert!(result.is_err());
    }

    #[test]
    fn test_quantile_loss_computation() {
        let model = QuantileRegression::new(0.5);

        // For median (tau=0.5), loss is just 0.5 * |r| for all r
        let residuals = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let loss = model.compute_quantile_loss(&residuals);

        // Expected: 0.5 * (2 + 1 + 0 + 1 + 2) = 3.0
        assert_relative_eq!(loss, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_asymmetric_quantile_loss() {
        // For tau=0.25, positive residuals cost more
        let model = QuantileRegression::new(0.25);

        let residuals = array![-1.0, 1.0];
        let loss = model.compute_quantile_loss(&residuals);

        // Expected: (0.25 - 1) * (-1) + 0.25 * 1 = 0.75 + 0.25 = 1.0
        assert_relative_eq!(loss, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_quantile_results() {
        // Create independent features to avoid singularity
        let mut x_data = Vec::new();
        for i in 0..30 {
            x_data.push(i as f64);
            x_data.push(30.0 - i as f64 + 0.5 * (i % 5) as f64); // Different pattern
        }
        let x = Array2::from_shape_vec((30, 2), x_data).unwrap();
        let y: Array1<f64> = (0..30)
            .map(|i| 1.0 + 0.5 * x[[i, 0]] + 0.3 * x[[i, 1]])
            .collect();

        let results = QuantileRegression::fit_quantiles_with_options(
            &x,
            &y,
            &[0.25, 0.5, 0.75],
            true,
            50,
            Some(vec!["x1".to_string(), "x2".to_string()]),
        )
        .unwrap();

        // Test accessor methods
        assert!(results.get_coefficient(0, 0).is_some());
        assert!(results.get_std_error(1, 1).is_some());

        let feat_coefs = results.coefficients_for_feature(0);
        assert!(feat_coefs.is_some());
        assert_eq!(feat_coefs.unwrap().len(), 3);

        assert_eq!(results.feature_name(0), "(Intercept)");
        assert_eq!(results.feature_name(1), "x1");
        assert_eq!(results.feature_name(2), "x2");
    }

    #[test]
    fn test_feature_importance() {
        // Create independent features to avoid singularity
        let mut x_data = Vec::new();
        for i in 0..30 {
            x_data.push(i as f64);
            x_data.push(30.0 - i as f64 + 0.3 * (i % 7) as f64); // Different pattern
        }
        let x = Array2::from_shape_vec((30, 2), x_data).unwrap();
        let y: Array1<f64> = (0..30)
            .map(|i| 1.0 + 2.0 * x[[i, 0]] + 0.1 * x[[i, 1]])
            .collect();

        let mut model = QuantileRegression::median().with_n_bootstrap(50);
        model.fit(&x, &y).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // First feature should be more important (coefficient ~2 vs ~0.1)
        assert!(importance[0] > importance[1]);
    }
}
