//! Logistic Regression with Full Statistical Diagnostics
//!
//! This module provides logistic regression with comprehensive statistical
//! diagnostics - FerroML's key differentiator from sklearn.
//!
//! ## Features
//!
//! - **Maximum likelihood via IRLS** (Iteratively Reweighted Least Squares)
//! - **Coefficient inference**: standard errors, z-statistics, p-values, confidence intervals
//! - **Odds ratios**: with confidence intervals for interpretability
//! - **Model fit statistics**: Log-likelihood, AIC, BIC, pseudo R², deviance
//! - **Wald tests**: for coefficient significance
//! - **Likelihood ratio test**: for overall model significance
//! - **ROC-AUC**: with confidence interval via bootstrap
//!
//! ## Example
//!
//! ```
//! use ferroml_core::models::logistic::LogisticRegression;
//! use ferroml_core::models::{Model, ProbabilisticModel};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
//!
//! let mut model = LogisticRegression::new();
//! model.fit(&x, &y).unwrap();
//!
//! // Get probability predictions
//! let probas = model.predict_proba(&x).unwrap();
//! assert_eq!(probas.nrows(), 6);
//! ```

use crate::hpo::{ParameterValue, SearchSpace};
use crate::metrics::probabilistic::{roc_auc_score, roc_auc_with_ci};
use crate::models::{
    check_is_fitted, compute_sample_weights, get_unique_classes, validate_fit_input,
    validate_predict_input, Assumption, AssumptionTestResult, ClassWeight, CoefficientInfo,
    Diagnostics, FitStatistics, Model, ModelSummary, PredictionInterval, ProbabilisticModel,
    ResidualStatistics, StatisticalModel,
};
use crate::pipeline::PipelineModel;
use crate::{FerroError, Result};
use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};

/// Logistic Regression with full statistical diagnostics
///
/// Fits a logistic model: P(y=1|X) = 1 / (1 + exp(-Xβ))
///
/// Uses Iteratively Reweighted Least Squares (IRLS) for maximum likelihood estimation.
/// This is equivalent to Newton-Raphson optimization for the log-likelihood.
///
/// ## Model Assumptions
///
/// 1. **Binary outcome**: y ∈ {0, 1}
/// 2. **Independence**: Observations are independent
/// 3. **No multicollinearity**: Features are not highly correlated
/// 4. **Linearity in log-odds**: log(p/(1-p)) is linear in X
/// 5. **Large sample size**: For valid inference (asymptotic normality)
///
/// ## Numerical Stability
///
/// The implementation includes an epsilon fallback in Cholesky decomposition to handle
/// near-singular Hessians that can occur with **perfect separation** or **quasi-complete
/// separation** (when a linear combination of features perfectly predicts the outcome).
///
/// For datasets prone to separation issues, consider using `with_l2_penalty()` to add
/// regularization, which stabilizes coefficient estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    /// Whether to include an intercept term
    pub fit_intercept: bool,
    /// Maximum number of IRLS iterations
    pub max_iter: usize,
    /// Convergence tolerance for coefficient change
    pub tol: f64,
    /// L2 regularization strength (0 = no regularization)
    pub l2_penalty: f64,
    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,
    /// Feature names (optional, for reporting)
    pub feature_names: Option<Vec<String>>,
    /// Number of bootstrap samples for ROC-AUC CI
    pub n_bootstrap: usize,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,

    // Fitted parameters (None before fit)
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features: Option<usize>,

    // Fitted data for diagnostics
    fitted_data: Option<FittedLogisticData>,
}

/// Internal struct to store fitted data for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FittedLogisticData {
    /// Number of observations
    n_samples: usize,
    /// Number of features (not including intercept)
    n_features: usize,
    /// Number of IRLS iterations until convergence
    n_iterations: usize,
    /// Whether IRLS converged
    converged: bool,
    /// Fitted probabilities
    fitted_probabilities: Array1<f64>,
    /// Original y values
    y: Array1<f64>,
    /// Deviance residuals
    deviance_residuals: Array1<f64>,
    /// Pearson residuals
    pearson_residuals: Array1<f64>,
    /// Coefficient standard errors (from Fisher information)
    coef_std_errors: Array1<f64>,
    /// Covariance matrix of coefficients (inverse Fisher information)
    coef_covariance: Array2<f64>,
    /// Log-likelihood at convergence
    log_likelihood: f64,
    /// Null log-likelihood (intercept-only model)
    null_log_likelihood: f64,
    /// Deviance (= -2 * log_likelihood)
    deviance: f64,
    /// Null deviance
    null_deviance: f64,
    /// Degrees of freedom for residuals
    df_residuals: usize,
    /// Degrees of freedom for null model
    df_null: usize,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegression {
    /// Create a new LogisticRegression model with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            max_iter: 100,
            tol: 1e-8,
            l2_penalty: 0.0,
            confidence_level: 0.95,
            feature_names: None,
            n_bootstrap: 1000,
            class_weight: ClassWeight::Uniform,
            coefficients: None,
            intercept: None,
            n_features: None,
            fitted_data: None,
        }
    }

    /// Create with no intercept term
    #[must_use]
    pub fn without_intercept() -> Self {
        Self {
            fit_intercept: false,
            ..Self::new()
        }
    }

    /// Set whether to fit an intercept
    #[must_use]
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set maximum iterations for IRLS
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    #[must_use]
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set L2 regularization strength.
    ///
    /// # Arguments
    /// * `penalty` - L2 regularization strength (default: 0.0):
    ///   - `0.0` - No regularization (standard MLE)
    ///   - `> 0` - Ridge regularization, shrinks coefficients toward zero
    ///
    /// # Numerical Stability
    /// Use a small positive value (e.g., 0.01-1.0) to prevent numerical issues
    /// with **near-perfect separation** or **quasi-complete separation**, which
    /// can cause coefficient estimates to diverge during IRLS optimization.
    ///
    /// Signs of separation issues:
    /// - Very large coefficient magnitudes (|β| > 10)
    /// - Model fails to converge
    /// - Standard errors are extremely large
    ///
    /// # Example
    /// ```
    /// use ferroml_core::models::LogisticRegression;
    ///
    /// // For datasets with potential separation issues
    /// let model = LogisticRegression::new()
    ///     .with_l2_penalty(1.0);
    /// ```
    #[must_use]
    pub fn with_l2_penalty(mut self, penalty: f64) -> Self {
        self.l2_penalty = penalty;
        self
    }

    /// Set confidence level for intervals
    #[must_use]
    pub fn with_confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set feature names for reporting
    #[must_use]
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Set class weights for handling imbalanced data
    ///
    /// # Arguments
    /// * `class_weight` - Weight specification: `Uniform`, `Balanced`, or `Custom`
    ///
    /// # Example
    /// ```
    /// use ferroml_core::models::{LogisticRegression, ClassWeight};
    ///
    /// // Automatically balance weights inversely proportional to class frequencies
    /// let model = LogisticRegression::new()
    ///     .with_class_weight(ClassWeight::Balanced);
    ///
    /// // Or use custom weights
    /// let model = LogisticRegression::new()
    ///     .with_class_weight(ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 3.0)]));
    /// ```
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Get the coefficients (excluding intercept)
    #[must_use]
    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    /// Get the intercept
    #[must_use]
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    /// Get all coefficients including intercept (intercept first if present)
    #[must_use]
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

    /// Get odds ratios for each coefficient
    ///
    /// Odds ratio = exp(β). An odds ratio > 1 means the odds of the positive
    /// class increase when the feature increases by 1 unit.
    #[must_use]
    pub fn odds_ratios(&self) -> Option<Array1<f64>> {
        self.coefficients.as_ref().map(|c| c.mapv(f64::exp))
    }

    /// Get odds ratios with confidence intervals
    pub fn odds_ratios_with_ci(&self, level: f64) -> Vec<OddsRatioInfo> {
        let mut result = Vec::new();

        let data = match &self.fitted_data {
            Some(d) => d,
            None => return result,
        };

        let z_crit = z_critical(1.0 - (1.0 - level) / 2.0);

        // Skip intercept, compute for feature coefficients only
        if let Some(coef) = &self.coefficients {
            for (i, &c) in coef.iter().enumerate() {
                let se_idx = if self.fit_intercept { i + 1 } else { i };
                let se = data.coef_std_errors[se_idx];

                let or = c.exp();
                // CI for log(OR) is β ± z * se, then exp to get OR CI
                let ci_lower = (c - z_crit * se).exp();
                let ci_upper = (c + z_crit * se).exp();

                let name = self.get_feature_name(i);
                result.push(OddsRatioInfo {
                    name,
                    odds_ratio: or,
                    ci_lower,
                    ci_upper,
                    confidence_level: level,
                });
            }
        }

        result
    }

    /// Get McFadden's pseudo R² (proportion of deviance explained)
    ///
    /// Values are typically lower than linear regression R².
    /// 0.2-0.4 is considered good fit for logistic regression.
    #[must_use]
    pub fn pseudo_r_squared(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        Some(1.0 - data.log_likelihood / data.null_log_likelihood)
    }

    /// Get the log-likelihood
    #[must_use]
    pub fn log_likelihood(&self) -> Option<f64> {
        self.fitted_data.as_ref().map(|d| d.log_likelihood)
    }

    /// Get AIC (Akaike Information Criterion)
    #[must_use]
    pub fn aic(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        let k = (data.n_features + usize::from(self.fit_intercept)) as f64;
        Some(2.0f64.mul_add(k, -(2.0 * data.log_likelihood)))
    }

    /// Get BIC (Bayesian Information Criterion)
    #[must_use]
    pub fn bic(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        let n = data.n_samples as f64;
        let k = (data.n_features + usize::from(self.fit_intercept)) as f64;
        Some(k.mul_add(n.ln(), -(2.0 * data.log_likelihood)))
    }

    /// Get deviance (= -2 * log_likelihood)
    #[must_use]
    pub fn deviance(&self) -> Option<f64> {
        self.fitted_data.as_ref().map(|d| d.deviance)
    }

    /// Perform likelihood ratio test for overall model significance
    ///
    /// Tests H0: all coefficients (except intercept) are zero
    /// vs H1: at least one coefficient is non-zero
    #[must_use]
    pub fn likelihood_ratio_test(&self) -> Option<(f64, f64)> {
        let data = self.fitted_data.as_ref()?;

        // LR statistic = 2 * (LL_full - LL_null) = null_deviance - deviance
        let lr_stat = data.null_deviance - data.deviance;
        let df = data.n_features as f64;

        // Chi-squared test
        let p_value = 1.0 - chi_squared_cdf(lr_stat, df);

        Some((lr_stat, p_value))
    }

    /// Compute ROC-AUC on training data
    pub fn train_roc_auc(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        roc_auc_score(&data.y, &data.fitted_probabilities).ok()
    }

    /// Compute ROC-AUC with confidence interval on training data
    pub fn train_roc_auc_with_ci(&self) -> Option<crate::metrics::MetricValueWithCI> {
        let data = self.fitted_data.as_ref()?;
        roc_auc_with_ci(
            &data.y,
            &data.fitted_probabilities,
            self.confidence_level,
            self.n_bootstrap,
            None,
        )
        .ok()
    }

    /// Get Pearson residuals
    #[must_use]
    pub fn pearson_residuals(&self) -> Option<Array1<f64>> {
        self.fitted_data
            .as_ref()
            .map(|d| d.pearson_residuals.clone())
    }

    /// Get deviance residuals
    #[must_use]
    pub fn deviance_residuals(&self) -> Option<Array1<f64>> {
        self.fitted_data
            .as_ref()
            .map(|d| d.deviance_residuals.clone())
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

    /// Fit using IRLS (Iteratively Reweighted Least Squares)
    ///
    /// # Arguments
    /// * `x_design` - Design matrix with intercept column if applicable
    /// * `y` - Binary target labels
    /// * `sample_weights` - Per-sample weights (from class_weight)
    fn fit_irls(
        &mut self,
        x_design: &Array2<f64>,
        y: &Array1<f64>,
        sample_weights: &Array1<f64>,
    ) -> Result<()> {
        let n = x_design.nrows();
        let p = x_design.ncols();

        // Initialize coefficients to zero
        let mut beta = Array1::zeros(p);

        // IRLS iterations
        let mut n_iter = 0;
        let mut converged = false;

        for iter in 0..self.max_iter {
            n_iter = iter + 1;

            // Compute linear predictor and probabilities
            let eta = x_design.dot(&beta);
            let mu = eta.mapv(sigmoid);

            // Check for numerical issues
            for &m in mu.iter() {
                if m < 1e-10 || m > 1.0 - 1e-10 {
                    // Probability too extreme, might have perfect separation
                }
            }

            // Weights: w_i = μ_i * (1 - μ_i) * sample_weight_i
            let w = &mu * &(1.0 - &mu) * sample_weights;

            // Clamp weights to avoid numerical issues
            let w_clamped: Array1<f64> = w.mapv(|wi| wi.clamp(1e-10, 0.25));

            // Working response: z = η + (y - μ) / (μ * (1-μ))
            // Note: We divide by the variance part only, not sample weights
            let variance_part: Array1<f64> = (&mu * &(1.0 - &mu)).mapv(|v| v.clamp(1e-10, 0.25));
            let z: Array1<f64> = &eta + &((y - &mu) / &variance_part);

            // Weighted least squares: solve (X'WX)β = X'Wz
            // Compute X'WX
            let mut xtwx = Array2::zeros((p, p));
            for i in 0..n {
                let xi = x_design.row(i);
                for j in 0..p {
                    for k in 0..p {
                        xtwx[[j, k]] += w_clamped[i] * xi[j] * xi[k];
                    }
                }
            }

            // Add L2 regularization if specified (don't regularize intercept)
            if self.l2_penalty > 0.0 {
                let start = usize::from(self.fit_intercept);
                for i in start..p {
                    xtwx[[i, i]] += self.l2_penalty;
                }
            }

            // Compute X'Wz
            let xtwz: Array1<f64> = (0..p)
                .map(|j| (0..n).map(|i| w_clamped[i] * x_design[[i, j]] * z[i]).sum())
                .collect();

            // Solve using Cholesky decomposition
            let beta_new = solve_symmetric_positive_definite(&xtwx, &xtwz)?;

            // Check convergence
            let delta: f64 = (&beta_new - &beta).mapv(|x| x.powi(2)).sum();
            let beta_norm: f64 = beta.mapv(|x| x.powi(2)).sum();

            beta = beta_new;

            if delta < self.tol * (beta_norm + self.tol) {
                converged = true;
                break;
            }
        }

        // Compute final probabilities
        let eta = x_design.dot(&beta);
        let mu = eta.mapv(sigmoid);

        // Compute weighted log-likelihood
        let log_likelihood = compute_weighted_log_likelihood(y, &mu, sample_weights);

        // Compute null log-likelihood (intercept only)
        // Use weighted mean for the null model
        let total_weight: f64 = sample_weights.sum();
        let p_bar = if total_weight > 0.0 {
            y.iter()
                .zip(sample_weights.iter())
                .map(|(&yi, &wi)| yi * wi)
                .sum::<f64>()
                / total_weight
        } else {
            y.mean().unwrap_or(0.5)
        };
        let null_log_likelihood = y
            .iter()
            .zip(sample_weights.iter())
            .map(|(&yi, &wi)| {
                wi * if yi > 0.5 {
                    p_bar.max(1e-15).ln()
                } else {
                    (1.0 - p_bar).max(1e-15).ln()
                }
            })
            .sum::<f64>();

        // Deviance
        let deviance = -2.0 * log_likelihood;
        let null_deviance = -2.0 * null_log_likelihood;

        // Compute Fisher information matrix (= X'WX at convergence)
        let w = &mu * &(1.0 - &mu) * sample_weights;
        let w_clamped: Array1<f64> = w.mapv(|wi| wi.max(1e-10));

        let mut fisher_info = Array2::zeros((p, p));
        for i in 0..n {
            let xi = x_design.row(i);
            for j in 0..p {
                for k in 0..p {
                    fisher_info[[j, k]] += w_clamped[i] * xi[j] * xi[k];
                }
            }
        }

        // Covariance matrix = (Fisher information)^-1
        let coef_covariance = invert_symmetric_matrix(&fisher_info)?;

        // Standard errors
        let mut coef_std_errors = Array1::zeros(p);
        for i in 0..p {
            coef_std_errors[i] = coef_covariance[[i, i]].max(0.0).sqrt();
        }

        // Compute residuals
        let deviance_residuals = compute_deviance_residuals(y, &mu);
        let pearson_residuals = compute_pearson_residuals(y, &mu);

        // Store coefficients
        let n_orig = if self.fit_intercept { p - 1 } else { p };
        if self.fit_intercept {
            self.intercept = Some(beta[0]);
            self.coefficients = Some(beta.slice(s![1..]).to_owned());
        } else {
            self.intercept = Some(0.0);
            self.coefficients = Some(beta);
        }

        self.n_features = Some(n_orig);

        // Store fitted data
        self.fitted_data = Some(FittedLogisticData {
            n_samples: n,
            n_features: n_orig,
            n_iterations: n_iter,
            converged,
            fitted_probabilities: mu,
            y: y.clone(),
            deviance_residuals,
            pearson_residuals,
            coef_std_errors,
            coef_covariance,
            log_likelihood,
            null_log_likelihood,
            deviance,
            null_deviance,
            df_residuals: n.saturating_sub(p),
            df_null: n.saturating_sub(1),
        });

        Ok(())
    }
}

impl Model for LogisticRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Validate binary labels
        for &yi in y {
            if !(yi == 0.0 || yi == 1.0 || (yi - 0.0).abs() < 1e-10 || (yi - 1.0).abs() < 1e-10) {
                return Err(FerroError::invalid_input(
                    "LogisticRegression requires binary labels (0 or 1)",
                ));
            }
        }

        // Check for both classes
        let n_pos = y.iter().filter(|&&yi| yi > 0.5).count();
        let n_neg = y.len() - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return Err(FerroError::invalid_input(
                "LogisticRegression requires both positive and negative samples",
            ));
        }

        let n = x.nrows();
        let p_orig = x.ncols();

        // Build design matrix with intercept if needed
        let x_design = if self.fit_intercept {
            let mut design = Array2::zeros((n, p_orig + 1));
            design.column_mut(0).fill(1.0);
            design.slice_mut(s![.., 1..]).assign(x);
            design
        } else {
            x.clone()
        };

        // Compute sample weights from class weights
        let classes = get_unique_classes(y);
        let sample_weights = compute_sample_weights(y, &classes, &self.class_weight);

        // Fit using IRLS
        self.fit_irls(&x_design, y, &sample_weights)
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        // Predict class labels (0 or 1) based on P(class=1)
        let probas = self.predict_proba(x)?;
        // Column 1 contains P(class=1)
        Ok(probas.column(1).mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 }))
    }

    fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // For logistic regression, use absolute z-statistics
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
        // Add L2 penalty as tunable parameter
        SearchSpace::new().float_log("l2_penalty", 1e-5, 10.0)
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn try_predict_proba(&self, x: &Array2<f64>) -> Option<Result<Array2<f64>>> {
        Some(self.predict_proba(x))
    }
}

impl StatisticalModel for LogisticRegression {
    fn summary(&self) -> ModelSummary {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => {
                return ModelSummary::new("Logistic Regression (not fitted)", 0, 0);
            }
        };

        let mut summary = ModelSummary::new("Logistic Regression", data.n_samples, data.n_features);

        // Add fit statistics
        let pseudo_r2 = self.pseudo_r_squared().unwrap_or(0.0);
        let mut fit_stats = FitStatistics::default();
        fit_stats.r_squared = Some(pseudo_r2); // Using pseudo R² in r_squared field
        fit_stats.df_model = Some(data.n_features);
        fit_stats.df_residuals = Some(data.df_residuals);
        fit_stats.log_likelihood = self.log_likelihood();
        fit_stats.aic = self.aic();
        fit_stats.bic = self.bic();

        // Add likelihood ratio test result
        if let Some((lr_stat, lr_p)) = self.likelihood_ratio_test() {
            fit_stats.f_statistic = Some(lr_stat); // Using f_statistic field for LR stat
            fit_stats.f_p_value = Some(lr_p);
        }

        summary = summary.with_fit_statistics(fit_stats);

        // Add coefficients with Wald tests
        let coefs = self.coefficients_with_ci(self.confidence_level);
        for coef in coefs {
            summary.add_coefficient(coef);
        }

        // Add convergence note
        if !data.converged {
            summary.add_note(format!(
                "Warning: IRLS did not converge in {} iterations",
                data.n_iterations
            ));
        }

        // Add ROC-AUC if available
        if let Some(auc) = self.train_roc_auc() {
            summary.add_note(format!("Training ROC-AUC: {:.4}", auc));
        }

        summary
    }

    fn diagnostics(&self) -> Diagnostics {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => {
                return Diagnostics::new(ResidualStatistics::default());
            }
        };

        // Use deviance residuals for statistics
        let residual_stats = ResidualStatistics::from_residuals(&data.deviance_residuals);
        let mut diagnostics = Diagnostics::new(residual_stats);

        // Add convergence as assumption test
        let convergence_test = AssumptionTestResult::new(
            Assumption::Independence, // Repurposing for convergence
            "IRLS Convergence",
            data.n_iterations as f64,
            if data.converged { 1.0 } else { 0.0 },
            0.5,
        )
        .with_details(format!("{} iterations", data.n_iterations));
        diagnostics.add_assumption_test(convergence_test);

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
            let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));

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
                let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));

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
        self.fitted_data
            .as_ref()
            .map(|d| d.deviance_residuals.clone())
    }

    fn fitted_values(&self) -> Option<Array1<f64>> {
        self.fitted_data
            .as_ref()
            .map(|d| d.fitted_probabilities.clone())
    }

    fn assumption_test(&self, assumption: Assumption) -> Option<AssumptionTestResult> {
        let diag = self.diagnostics();
        diag.assumption_tests
            .into_iter()
            .find(|t| t.assumption == assumption)
    }
}

impl ProbabilisticModel for LogisticRegression {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.coefficients, "predict_proba")?;

        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let coef = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        let linear_pred = x.dot(coef) + intercept;
        let probas_class1 = linear_pred.mapv(sigmoid);

        // Return as (n_samples, 2) matrix with [P(class=0), P(class=1)]
        // This ensures probabilities sum to 1 for each sample
        let n = probas_class1.len();
        let mut result = Array2::zeros((n, 2));
        for i in 0..n {
            let p1 = probas_class1[i];
            result[[i, 0]] = 1.0 - p1; // P(class=0)
            result[[i, 1]] = p1; // P(class=1)
        }
        Ok(result)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        // For logistic regression, return CI for predicted probabilities
        check_is_fitted(&self.coefficients, "predict_interval")?;

        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let data = self
            .fitted_data
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_interval"))?;

        let coef = self.coefficients.as_ref().unwrap();
        let intercept = self.intercept.unwrap_or(0.0);

        // Build design matrix
        let n = x.nrows();
        let x_design = if self.fit_intercept {
            let mut design = Array2::zeros((n, n_features + 1));
            design.column_mut(0).fill(1.0);
            design.slice_mut(s![.., 1..]).assign(x);
            design
        } else {
            x.clone()
        };

        let z_crit = z_critical(1.0 - (1.0 - level) / 2.0);

        let linear_pred = x.dot(coef) + intercept;
        let predictions = linear_pred.mapv(sigmoid);

        // Compute standard errors for linear predictor using delta method
        let mut std_errors = Array1::zeros(n);
        for i in 0..n {
            let xi = x_design.row(i);
            let var_eta = xi.dot(&data.coef_covariance.dot(&xi.t()));
            std_errors[i] = var_eta.max(0.0).sqrt();
        }

        // CI for linear predictor, then transform to probability scale
        let lower = (&linear_pred - &(&std_errors * z_crit)).mapv(sigmoid);
        let upper = (&linear_pred + &(&std_errors * z_crit)).mapv(sigmoid);

        Ok(PredictionInterval::new(predictions, lower, upper, level).with_std_errors(std_errors))
    }
}

/// Information about an odds ratio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OddsRatioInfo {
    /// Name of the variable/feature
    pub name: String,
    /// Odds ratio (= exp(coefficient))
    pub odds_ratio: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Confidence level used for CI
    pub confidence_level: f64,
}

impl OddsRatioInfo {
    /// Check if the odds ratio is significantly different from 1
    ///
    /// True if the CI does not contain 1.
    pub fn is_significant(&self) -> bool {
        self.ci_lower > 1.0 || self.ci_upper < 1.0
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let ex = (-x).exp();
        1.0 / (1.0 + ex)
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Compute log-likelihood for logistic regression (unweighted variant kept for reference)
fn _compute_log_likelihood(y: &Array1<f64>, mu: &Array1<f64>) -> f64 {
    y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mi)| {
            let mi_clamped = mi.clamp(1e-15, 1.0 - 1e-15);
            if yi > 0.5 {
                mi_clamped.ln()
            } else {
                (1.0 - mi_clamped).ln()
            }
        })
        .sum()
}

/// Compute weighted log-likelihood for logistic regression
fn compute_weighted_log_likelihood(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    sample_weights: &Array1<f64>,
) -> f64 {
    y.iter()
        .zip(mu.iter())
        .zip(sample_weights.iter())
        .map(|((&yi, &mi), &wi)| {
            let mi_clamped = mi.clamp(1e-15, 1.0 - 1e-15);
            wi * if yi > 0.5 {
                mi_clamped.ln()
            } else {
                (1.0 - mi_clamped).ln()
            }
        })
        .sum()
}

/// Compute deviance residuals
fn compute_deviance_residuals(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mi)| {
            let mi_clamped = mi.clamp(1e-15, 1.0 - 1e-15);
            let d = if yi > 0.5 {
                2.0 * (yi.ln() - mi_clamped.ln())
            } else {
                2.0 * ((1.0 - yi + 1e-15).ln() - (1.0 - mi_clamped).ln())
            };
            d.max(0.0).sqrt().copysign(yi - mi_clamped)
        })
        .collect()
}

/// Compute Pearson residuals
fn compute_pearson_residuals(y: &Array1<f64>, mu: &Array1<f64>) -> Array1<f64> {
    y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mi)| {
            let var = mi * (1.0 - mi);
            (yi - mi) / var.max(1e-15).sqrt()
        })
        .collect()
}

/// Solve symmetric positive definite system Ax = b using Cholesky decomposition
fn solve_symmetric_positive_definite(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();

    // Cholesky decomposition: A = LL'
    // With small epsilon fallback for numerical stability (handles near-perfect separation)
    let mut l = Array2::zeros((n, n));
    let epsilon = 1e-6;

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }

            if i == j {
                // Add small epsilon for numerical stability if diagonal is non-positive
                if sum <= 0.0 {
                    sum = epsilon;
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

/// Invert symmetric matrix using Cholesky decomposition
fn invert_symmetric_matrix(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut inv = Array2::zeros((n, n));

    // Solve A * inv_col = e_i for each column
    for i in 0..n {
        let mut e = Array1::zeros(n);
        e[i] = 1.0;
        let inv_col = solve_symmetric_positive_definite(a, &e)?;
        inv.column_mut(i).assign(&inv_col);
    }

    Ok(inv)
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / 0.231_641_9f64.mul_add(x.abs(), 1.0);
    let d = 0.398_942_280_401_432_7 * (-x * x / 2.0).exp();
    let p = d
        * t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));

    if x > 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Standard normal critical value (inverse CDF approximation)
fn z_critical(p: f64) -> f64 {
    // Rational approximation for inverse normal CDF
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let p_adj = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * p_adj.ln()).sqrt();

    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;

    let z = t
        - (c2 * t).mul_add(t, c0 + c1 * t)
            / (d3 * t * t).mul_add(t, (d2 * t).mul_add(t, 1.0 + d1 * t));

    if p > 0.5 {
        z
    } else {
        -z
    }
}

/// Chi-squared CDF approximation using normal approximation for large df
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }

    // Wilson-Hilferty transformation for large df
    let z = (x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df));
    let z = z / (2.0 / (9.0 * df)).sqrt();

    normal_cdf(z)
}

// =============================================================================
// PipelineModel Implementation
// =============================================================================

impl PipelineModel for LogisticRegression {
    fn clone_boxed(&self) -> Box<dyn PipelineModel> {
        Box::new(self.clone())
    }

    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> {
        match name {
            "fit_intercept" => {
                if let Some(v) = value.as_bool() {
                    self.fit_intercept = v;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input("fit_intercept must be a boolean"))
                }
            }
            "confidence_level" => {
                if let Some(v) = value.as_f64() {
                    self.confidence_level = v;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input(
                        "confidence_level must be a number",
                    ))
                }
            }
            "l2_penalty" => {
                if let Some(v) = value.as_f64() {
                    self.l2_penalty = v;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input(
                        "l2_penalty (regularization) must be a number",
                    ))
                }
            }
            "max_iter" => {
                if let Some(v) = value.as_i64() {
                    self.max_iter = v as usize;
                    Ok(())
                } else {
                    Err(FerroError::invalid_input("max_iter must be an integer"))
                }
            }
            _ => Err(FerroError::invalid_input(format!(
                "Unknown parameter: {}",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "LogisticRegression"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);

        // Check symmetry
        assert_relative_eq!(sigmoid(2.0), 1.0 - sigmoid(-2.0), epsilon = 1e-10);
    }

    #[test]
    fn test_logistic_regression_simple() {
        // Simple linearly separable data
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());
        assert!(model.coefficients().is_some());

        // Coefficient should be positive (higher x -> more likely y=1)
        let coef = model.coefficients().unwrap();
        assert!(coef[0] > 0.0);

        // Check predictions
        let pred = model.predict(&x).unwrap();
        for (i, &p) in pred.iter().enumerate() {
            if i < 4 {
                assert_eq!(p, 0.0, "Expected class 0 for sample {}", i);
            } else {
                assert_eq!(p, 1.0, "Expected class 1 for sample {}", i);
            }
        }
    }

    #[test]
    fn test_predict_proba() {
        let x = Array2::from_shape_vec((6, 1), vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]).unwrap();
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let probas = model.predict_proba(&x).unwrap();

        // Output has 2 columns: [P(class=0), P(class=1)]
        assert_eq!(probas.ncols(), 2);

        // For sample 0 (class 0): P(class=1) should be low
        assert!(probas[[0, 1]] < 0.5, "Sample 0 should have low P(class=1)");
        // For sample 5 (class 1): P(class=1) should be high
        assert!(probas[[5, 1]] > 0.5, "Sample 5 should have high P(class=1)");

        // All probabilities in [0, 1] and rows sum to 1
        for i in 0..6 {
            assert!(probas[[i, 0]] >= 0.0 && probas[[i, 0]] <= 1.0);
            assert!(probas[[i, 1]] >= 0.0 && probas[[i, 1]] <= 1.0);
            let sum = probas[[i, 0]] + probas[[i, 1]];
            assert!((sum - 1.0).abs() < 1e-10, "Row {} should sum to 1.0", i);
        }
    }

    #[test]
    fn test_odds_ratios() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let odds_ratios = model.odds_ratios().unwrap();

        // Odds ratio > 1 means positive relationship
        assert!(odds_ratios[0] > 1.0);

        // Odds ratio = exp(coef)
        let coef = model.coefficients().unwrap();
        assert_relative_eq!(odds_ratios[0], coef[0].exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_odds_ratios_with_ci() {
        // Use data that's not perfectly separable to get proper CIs
        let x = Array2::from_shape_vec(
            (20, 1),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
                8.5, 9.5, 4.0, 7.0,
            ],
        )
        .unwrap();
        // Mix classes slightly for realistic data
        let y = array![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 1.0
        ];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let or_info = model.odds_ratios_with_ci(0.95);
        assert_eq!(or_info.len(), 1);

        let or = &or_info[0];
        // CI should be properly ordered
        assert!(or.ci_lower < or.odds_ratio);
        assert!(or.odds_ratio < or.ci_upper);
        // Odds ratio should be > 1 (positive relationship)
        assert!(or.odds_ratio > 1.0);
    }

    #[test]
    fn test_summary() {
        // Use independent features and some overlap in classes for stability
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 8.0, 2.0, 6.0, 3.0, 9.0, 4.0, 5.0, 5.0, 7.0, 6.0, 2.0, 7.0, 4.0, 8.0, 1.0,
                9.0, 3.0, 10.0, 5.0, 1.5, 7.5, 2.5, 5.5, 3.5, 8.5, 4.5, 4.5, 5.5, 6.5, 6.5, 2.5,
                7.5, 3.5, 8.5, 1.5, 9.5, 2.5, 10.5, 4.5,
            ],
        )
        .unwrap();
        let y = array![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 1.0
        ];

        let mut model = LogisticRegression::new()
            .with_feature_names(vec!["feature1".into(), "feature2".into()]);
        model.fit(&x, &y).unwrap();

        let summary = model.summary();
        let output = format!("{}", summary);

        assert!(output.contains("Logistic Regression"));
        assert!(output.contains("feature1"));
        assert!(output.contains("feature2"));
    }

    #[test]
    fn test_pseudo_r_squared() {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let pseudo_r2 = model.pseudo_r_squared().unwrap();

        // Pseudo R² should be between 0 and 1
        assert!(pseudo_r2 >= 0.0 && pseudo_r2 <= 1.0);

        // For perfectly separable data, should be high
        assert!(pseudo_r2 > 0.5);
    }

    #[test]
    fn test_likelihood_ratio_test() {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let (lr_stat, p_value) = model.likelihood_ratio_test().unwrap();

        // LR stat should be positive
        assert!(lr_stat > 0.0);

        // p-value in [0, 1]
        assert!(p_value >= 0.0 && p_value <= 1.0);

        // Should be significant
        assert!(p_value < 0.05);
    }

    #[test]
    fn test_coefficients_with_ci() {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let coefs = model.coefficients_with_ci(0.95);

        // Should have intercept + 1 feature
        assert_eq!(coefs.len(), 2);

        // Check CI contains estimate
        for coef in &coefs {
            assert!(coef.ci_lower <= coef.estimate);
            assert!(coef.estimate <= coef.ci_upper);
        }
    }

    #[test]
    fn test_prediction_interval() {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let interval = model.predict_interval(&x, 0.95).unwrap();

        // All predictions and bounds should be in [0, 1]
        for i in 0..10 {
            assert!(interval.predictions[i] >= 0.0 && interval.predictions[i] <= 1.0);
            assert!(interval.lower[i] >= 0.0 && interval.lower[i] <= 1.0);
            assert!(interval.upper[i] >= 0.0 && interval.upper[i] <= 1.0);
            assert!(interval.lower[i] <= interval.predictions[i]);
            assert!(interval.predictions[i] <= interval.upper[i]);
        }
    }

    #[test]
    fn test_without_intercept() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::without_intercept();
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());
        assert_eq!(model.intercept(), Some(0.0));
    }

    #[test]
    fn test_multivariate() {
        // 2D data
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 1.0, 6.0, 6.0, 7.0, 7.0, 8.0, 6.0,
                9.0, 7.0, 10.0, 6.0,
            ],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());
        let coef = model.coefficients().unwrap();
        assert_eq!(coef.len(), 2);
    }

    #[test]
    fn test_error_not_fitted() {
        let model = LogisticRegression::new();
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        assert!(model.predict(&x).is_err());
        assert!(model.predict_proba(&x).is_err());
    }

    #[test]
    fn test_error_non_binary() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![0.0, 1.0, 2.0, 0.0, 1.0]; // Contains 2.0

        let mut model = LogisticRegression::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_error_single_class() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0]; // Only class 0

        let mut model = LogisticRegression::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_l2_regularization() {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut model_no_reg = LogisticRegression::new();
        model_no_reg.fit(&x, &y).unwrap();

        let mut model_reg = LogisticRegression::new().with_l2_penalty(1.0);
        model_reg.fit(&x, &y).unwrap();

        // Regularized coefficients should be smaller in magnitude
        let coef_no_reg = model_no_reg.coefficients().unwrap()[0].abs();
        let coef_reg = model_reg.coefficients().unwrap()[0].abs();
        assert!(coef_reg < coef_no_reg);
    }

    #[test]
    fn test_residuals() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let dev_resid = model.deviance_residuals().unwrap();
        let pearson_resid = model.pearson_residuals().unwrap();

        assert_eq!(dev_resid.len(), 8);
        assert_eq!(pearson_resid.len(), 8);
    }

    #[test]
    fn test_information_criteria() {
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        assert!(model.log_likelihood().is_some());
        assert!(model.aic().is_some());
        assert!(model.bic().is_some());

        // Log-likelihood should be negative
        assert!(model.log_likelihood().unwrap() < 0.0);

        // AIC and BIC should be positive
        assert!(model.aic().unwrap() > 0.0);
        assert!(model.bic().unwrap() > 0.0);
    }

    #[test]
    fn test_feature_importance() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 5.0, 2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 5.0, 7.0, 4.0, 8.0, 3.0,
                9.0, 2.0, 10.0, 1.0,
            ],
        )
        .unwrap();
        let y = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Both should be positive
        assert!(importance[0] >= 0.0);
        assert!(importance[1] >= 0.0);
    }

    #[test]
    fn test_normal_cdf() {
        assert_relative_eq!(normal_cdf(0.0), 0.5, epsilon = 1e-3);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }

    #[test]
    fn test_z_critical() {
        // z_0.975 ≈ 1.96
        let z = z_critical(0.975);
        assert!(z > 1.9 && z < 2.0);

        // z_0.95 ≈ 1.645
        let z = z_critical(0.95);
        assert!(z > 1.6 && z < 1.7);
    }
}
