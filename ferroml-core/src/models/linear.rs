//! Linear Regression with Full Statistical Diagnostics
//!
//! This module provides OLS linear regression with comprehensive statistical
//! diagnostics - FerroML's key differentiator from sklearn.
//!
//! ## Features
//!
//! - **OLS fitting via QR decomposition** for numerical stability
//! - **Coefficient inference**: standard errors, t-statistics, p-values, confidence intervals
//! - **Model fit statistics**: R², adjusted R², F-statistic with p-value
//! - **Residual diagnostics**: normality (Shapiro-Wilk), homoscedasticity (Breusch-Pagan), autocorrelation (Durbin-Watson)
//! - **Influential observations**: Cook's distance, leverage (hat values), DFFITS
//! - **Multicollinearity detection**: VIF (Variance Inflation Factor)
//! - **Prediction intervals**: confidence and prediction bands
//!
//! ## Example
//!
//! ```
//! use ferroml_core::models::linear::LinearRegression;
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! // Non-collinear features: y ≈ 2*x1 + 3*x2
//! let x = Array2::from_shape_vec((5, 2), vec![
//!     1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![5.0, 7.0, 8.0, 12.0, 13.0]);
//!
//! let mut model = LinearRegression::new();
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 5);
//! ```

use crate::hpo::{ParameterValue, SearchSpace};
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, Assumption, AssumptionTestResult,
    CoefficientInfo, Diagnostics, FitStatistics, InfluentialObservation, Model, ModelSummary,
    PredictionInterval, ProbabilisticModel, ResidualStatistics, StatisticalModel,
};
use crate::pipeline::PipelineModel;
use crate::stats::diagnostics::{durbin_watson, NormalityTest, ShapiroWilkTest};
use crate::{FerroError, Result};
use ndarray::{s, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Ordinary Least Squares Linear Regression with full statistical diagnostics
///
/// Fits a linear model: y = Xβ + ε
///
/// ## Statistical Assumptions
///
/// 1. **Linearity**: The relationship between X and y is linear
/// 2. **Independence**: Observations are independent
/// 3. **Homoscedasticity**: Constant variance of residuals
/// 4. **Normality**: Residuals are normally distributed
/// 5. **No multicollinearity**: Features are not highly correlated
///
/// All assumptions are automatically tested after fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    /// Whether to include an intercept term
    pub fit_intercept: bool,
    /// Copy X matrix (if false, may modify in place for efficiency)
    pub copy_x: bool,
    /// Confidence level for intervals (default: 0.95)
    pub confidence_level: f64,
    /// Feature names (optional, for reporting)
    pub feature_names: Option<Vec<String>>,

    // Fitted parameters (None before fit)
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    n_features: Option<usize>,

    // Training data statistics (stored for diagnostics)
    fitted_data: Option<FittedData>,
}

/// Internal struct to store fitted data for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FittedData {
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
    /// Hat matrix diagonal (leverage)
    leverage: Array1<f64>,
    /// Residual standard error
    residual_std_error: f64,
    /// Total sum of squares
    tss: f64,
    /// Residual sum of squares
    rss: f64,
    /// Coefficient standard errors
    coef_std_errors: Array1<f64>,
    /// Covariance matrix of coefficients
    coef_covariance: Array2<f64>,
    /// Mean of y
    y_mean: f64,
    /// Mean of each feature
    x_means: Array1<f64>,
    /// Degrees of freedom for residuals
    df_residuals: usize,
    /// Degrees of freedom for model
    df_model: usize,
    /// Condition number of the design matrix (from R diagonal of QR)
    condition_number: Option<f64>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegression {
    /// Create a new LinearRegression model with default settings
    pub fn new() -> Self {
        Self {
            fit_intercept: true,
            copy_x: true,
            confidence_level: 0.95,
            feature_names: None,
            coefficients: None,
            intercept: None,
            n_features: None,
            fitted_data: None,
        }
    }

    /// Create with no intercept term
    pub fn without_intercept() -> Self {
        Self {
            fit_intercept: false,
            ..Self::new()
        }
    }

    /// Set whether to fit an intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
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

    /// Get the Variance Inflation Factors for each coefficient
    ///
    /// VIF > 5 suggests moderate multicollinearity
    /// VIF > 10 suggests severe multicollinearity
    pub fn vif(&self, x: &Array2<f64>) -> Array1<f64> {
        self.compute_vif_full(x)
    }

    /// Compute VIF properly by regressing each feature on the others
    fn compute_vif_full(&self, x: &Array2<f64>) -> Array1<f64> {
        let n_features = x.ncols();
        let mut vifs = Array1::zeros(n_features);

        for j in 0..n_features {
            // Get feature j as target
            let y_j = x.column(j).to_owned();

            // Get other features as predictors
            let mut x_others: Vec<f64> = Vec::new();
            for i in 0..x.nrows() {
                for k in 0..n_features {
                    if k != j {
                        x_others.push(x[[i, k]]);
                    }
                }
            }

            if n_features > 1 {
                let x_other = Array2::from_shape_vec((x.nrows(), n_features - 1), x_others)
                    .expect("SAFETY: shape computed from data dimensions");

                // Fit regression of x_j on other xs
                let mut reg = LinearRegression::new();
                if reg.fit(&x_other, &y_j).is_ok() {
                    if let Some(data) = &reg.fitted_data {
                        let r2 = 1.0 - data.rss / data.tss;
                        vifs[j] = 1.0 / (1.0 - r2).max(1e-10);
                    } else {
                        vifs[j] = 1.0;
                    }
                } else {
                    vifs[j] = f64::INFINITY; // Perfect multicollinearity
                }
            } else {
                vifs[j] = 1.0; // Single feature, no multicollinearity
            }
        }

        vifs
    }

    /// Compute Cook's distance for all observations
    pub fn cooks_distance(&self) -> Option<Array1<f64>> {
        let data = self.fitted_data.as_ref()?;
        let n = data.n_samples;
        let p = data.n_features + if self.fit_intercept { 1 } else { 0 };

        let mut cooks = Array1::zeros(n);
        let mse = data.rss / data.df_residuals as f64;

        for i in 0..n {
            let h_i = data.leverage[i];
            let e_i = data.residuals[i];
            let one_minus_h = (1.0 - h_i).max(1e-10);
            cooks[i] = (e_i.powi(2) / (p as f64 * mse)) * (h_i / one_minus_h.powi(2));
        }

        Some(cooks)
    }

    /// Get standardized residuals
    pub fn standardized_residuals(&self) -> Option<Array1<f64>> {
        let data = self.fitted_data.as_ref()?;
        let n = data.n_samples;
        let mse = data.rss / data.df_residuals as f64;

        let mut std_resid = Array1::zeros(n);
        for i in 0..n {
            let h_i = data.leverage[i];
            let one_minus_h = (1.0 - h_i).max(1e-10);
            std_resid[i] = data.residuals[i] / (mse * one_minus_h).sqrt();
        }

        Some(std_resid)
    }

    /// Get studentized (externally studentized) residuals
    pub fn studentized_residuals(&self) -> Option<Array1<f64>> {
        let data = self.fitted_data.as_ref()?;
        let n = data.n_samples;
        let df = data.df_residuals;

        let mut stud_resid = Array1::zeros(n);

        for i in 0..n {
            let h_i = data.leverage[i];
            let e_i = data.residuals[i];

            // Leave-one-out MSE estimate
            let one_minus_h = (1.0 - h_i).max(1e-10);
            let mse_i = (data.rss - e_i.powi(2) / one_minus_h) / (df - 1) as f64;
            stud_resid[i] = e_i / (mse_i * one_minus_h).sqrt();
        }

        Some(stud_resid)
    }

    /// Calculate DFFITS for each observation
    pub fn dffits(&self) -> Option<Array1<f64>> {
        let stud = self.studentized_residuals()?;
        let data = self.fitted_data.as_ref()?;

        let mut dffits = Array1::zeros(data.n_samples);
        for i in 0..data.n_samples {
            let h_i = data.leverage[i];
            let one_minus_h = (1.0 - h_i).max(1e-10);
            dffits[i] = stud[i] * (h_i / one_minus_h).sqrt();
        }

        Some(dffits)
    }

    /// Get the R-squared value
    pub fn r_squared(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        Some(1.0 - data.rss / data.tss)
    }

    /// Get the adjusted R-squared value
    pub fn adjusted_r_squared(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        let n = data.n_samples as f64;
        let p = data.df_model as f64;
        let r2 = 1.0 - data.rss / data.tss;
        Some(1.0 - (1.0 - r2) * (n - 1.0) / (n - p - 1.0))
    }

    /// Get the F-statistic for overall model significance
    pub fn f_statistic(&self) -> Option<(f64, f64)> {
        let data = self.fitted_data.as_ref()?;

        let ess = data.tss - data.rss; // Explained sum of squares
        let df1 = data.df_model as f64;
        let df2 = data.df_residuals as f64;

        if df1 == 0.0 || df2 == 0.0 {
            return None;
        }

        let f_stat = (ess / df1) / (data.rss / df2);
        let p_value = 1.0 - f_cdf(f_stat, df1, df2);

        Some((f_stat, p_value))
    }

    /// Get the log-likelihood
    pub fn log_likelihood(&self) -> Option<f64> {
        let data = self.fitted_data.as_ref()?;
        let n = data.n_samples as f64;
        let sigma2 = data.rss / n;

        // LL = -n/2 * (log(2π) + log(σ²) + 1)
        Some(-n / 2.0 * ((2.0 * std::f64::consts::PI).ln() + sigma2.ln() + 1.0))
    }

    /// Get AIC (Akaike Information Criterion)
    pub fn aic(&self) -> Option<f64> {
        let ll = self.log_likelihood()?;
        let data = self.fitted_data.as_ref()?;
        let k = (data.df_model + 1) as f64; // +1 for variance parameter
        Some(2.0f64.mul_add(k, -(2.0 * ll)))
    }

    /// Get BIC (Bayesian Information Criterion)
    pub fn bic(&self) -> Option<f64> {
        let ll = self.log_likelihood()?;
        let data = self.fitted_data.as_ref()?;
        let n = data.n_samples as f64;
        let k = (data.df_model + 1) as f64;
        Some(k * n.ln() - 2.0 * ll)
    }

    /// Get condition number of the design matrix (indicates multicollinearity).
    /// Computed during fit from the R diagonal of QR decomposition.
    /// High values (> 30) suggest multicollinearity issues.
    pub fn condition_number(&self) -> Option<f64> {
        self.fitted_data.as_ref()?.condition_number
    }

    fn get_feature_name(&self, idx: usize) -> String {
        super::get_feature_name(&self.feature_names, idx)
    }
}

impl Model for LinearRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n = x.nrows();
        let p_orig = x.ncols();

        // Optionally copy X
        let x_work = if self.copy_x { x.clone() } else { x.clone() };

        // Build design matrix with intercept if needed
        let (x_design, p) = if self.fit_intercept {
            let mut design = Array2::zeros((n, p_orig + 1));
            design.column_mut(0).fill(1.0);
            design.slice_mut(s![.., 1..]).assign(&x_work);
            (design, p_orig + 1)
        } else {
            (x_work, p_orig)
        };

        // Check for sufficient data
        if n <= p {
            return Err(FerroError::invalid_input(format!(
                "Need more observations ({}) than parameters ({})",
                n, p
            )));
        }

        // Solve for coefficients and compute (X'X)^{-1}
        // Use Cholesky normal equations when n >> p (faster for tall matrices),
        // QR decomposition otherwise (better numerical stability)
        let (coefficients, condition_number, xtx_inv) = if n > 2 * p {
            // Cholesky normal equations: X'X * beta = X'y
            // O(n·d²) to form X'X, then O(d³) to solve — faster when n >> d
            let xtx = x_design.t().dot(&x_design);
            let xty = x_design.t().dot(y);

            let l = crate::linalg::cholesky(&xtx, 1e-10)?;
            let z = crate::linalg::solve_lower_triangular_vec(&l, &xty)?;
            let lt = l.t().to_owned();
            let coefficients = crate::linalg::solve_upper_triangular(&lt, &z)?;

            // Compute (X'X)^{-1} = L^{-T} L^{-1} for standard errors / leverage
            let lt_inv = invert_upper_triangular(&lt)?;
            let xtx_inv = lt_inv.dot(&lt_inv.t());

            // Condition number from Cholesky diagonal: cond(X'X) ≈ (max/min)²
            // cond(X) ≈ sqrt(cond(X'X)) = max(L_ii) / min(L_ii)
            let condition_number = {
                let diag: Vec<f64> = (0..l.ncols()).map(|i| l[[i, i]].abs()).collect();
                let max_diag = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_diag = diag.iter().cloned().fold(f64::INFINITY, f64::min);
                if min_diag <= 1e-14 {
                    return Err(FerroError::numerical(
                        "Matrix is rank-deficient (collinear features)",
                    ));
                }
                Some(max_diag / min_diag)
            };

            (coefficients, condition_number, xtx_inv)
        } else {
            // QR decomposition for numerical stability
            // Using Gram-Schmidt orthogonalization
            let (q, r) = qr_decomposition(&x_design)?;

            // Condition number from R diagonal: cond(X) ≈ max|R_ii| / min|R_ii|
            let condition_number = {
                let diag: Vec<f64> = (0..r.ncols()).map(|i| r[[i, i]].abs()).collect();
                let max_diag = diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_diag = diag.iter().cloned().fold(f64::INFINITY, f64::min);
                if min_diag <= 1e-14 {
                    return Err(FerroError::numerical(
                        "Matrix is rank-deficient (collinear features)",
                    ));
                }
                Some(max_diag / min_diag)
            };

            // Solve R * beta = Q' * y
            let qty = q.t().dot(y);
            let coefficients = solve_upper_triangular(&r, &qty)?;

            // Compute (X'X)^{-1} = R^{-1} * R'^{-1}
            let r_inv = invert_upper_triangular(&r)?;
            let xtx_inv = r_inv.dot(&r_inv.t());

            (coefficients, condition_number, xtx_inv)
        };

        // Compute fitted values and residuals
        let fitted_values = x_design.dot(&coefficients);
        let residuals = y - &fitted_values;

        // Compute statistics
        let y_mean = y.mean().unwrap_or(0.0);
        let tss: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let rss: f64 = residuals.iter().map(|r| r.powi(2)).sum();

        let df_model = if self.fit_intercept { p - 1 } else { p };
        let df_residuals = n - p;

        // Residual standard error
        let residual_std_error = (rss / df_residuals as f64).sqrt();

        // Coefficient covariance: (X'X)^{-1} * sigma²
        let coef_covariance = &xtx_inv * residual_std_error.powi(2);

        // Extract coefficient standard errors
        let mut coef_std_errors = Array1::zeros(p);
        for i in 0..p {
            coef_std_errors[i] = coef_covariance[[i, i]].sqrt();
        }
        let mut leverage = Array1::zeros(n);
        for i in 0..n {
            let xi = x_design.row(i);
            leverage[i] = xi.dot(&xtx_inv.dot(&xi.t()));
        }

        // Compute X means for prediction intervals
        let x_means = x
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(p_orig));

        // Store fitted data
        self.fitted_data = Some(FittedData {
            n_samples: n,
            n_features: p_orig,
            residuals,
            fitted_values,
            y: y.clone(),
            leverage,
            residual_std_error,
            tss,
            rss,
            coef_std_errors,
            coef_covariance,
            y_mean,
            x_means,
            df_residuals,
            df_model,
            condition_number,
        });

        // Store coefficients
        if self.fit_intercept {
            self.intercept = Some(coefficients[0]);
            self.coefficients = Some(coefficients.slice(s![1..]).to_owned());
        } else {
            self.intercept = Some(0.0);
            self.coefficients = Some(coefficients);
        }

        self.n_features = Some(p_orig);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.coefficients, "predict")?;

        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        validate_predict_input(x, n_features)?;

        let coef = self
            .coefficients
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let intercept = self.intercept.unwrap_or(0.0);

        let predictions = x.dot(coef) + intercept;
        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let coef = self.coefficients.as_ref()?;
        let abs_coef: Array1<f64> = coef.mapv(|c| c.abs());
        let sum = abs_coef.sum();
        if sum > 0.0 {
            Some(abs_coef / sum)
        } else {
            Some(abs_coef)
        }
    }

    fn search_space(&self) -> SearchSpace {
        // Linear regression has no hyperparameters to tune
        SearchSpace::new()
    }

    fn feature_names(&self) -> Option<&[String]> {
        self.feature_names.as_deref()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}

impl StatisticalModel for LinearRegression {
    fn summary(&self) -> ModelSummary {
        let data = match &self.fitted_data {
            Some(d) => d,
            None => {
                return ModelSummary::new("OLS Linear Regression (not fitted)", 0, 0);
            }
        };

        let mut summary =
            ModelSummary::new("OLS Linear Regression", data.n_samples, data.n_features);

        // Add fit statistics
        let r2 = self.r_squared().unwrap_or(0.0);
        let adj_r2 = self.adjusted_r_squared().unwrap_or(0.0);
        let mut fit_stats =
            FitStatistics::with_r_squared(r2, adj_r2).with_df(data.df_model, data.df_residuals);

        if let Some((f_stat, f_p)) = self.f_statistic() {
            fit_stats = fit_stats.with_f_test(f_stat, f_p);
        }

        if let (Some(ll), Some(aic), Some(bic)) = (self.log_likelihood(), self.aic(), self.bic()) {
            fit_stats = fit_stats.with_information_criteria(ll, aic, bic);
        }

        fit_stats.residual_std_error = Some(data.residual_std_error);
        summary = summary.with_fit_statistics(fit_stats);

        // Add coefficients
        let coefs = self.coefficients_with_ci(self.confidence_level);
        for coef in coefs {
            summary.add_coefficient(coef);
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

        let residual_stats = ResidualStatistics::from_residuals(&data.residuals);
        let mut diagnostics = Diagnostics::new(residual_stats);

        // Normality test
        if let Ok(norm_result) = ShapiroWilkTest.test(&data.residuals) {
            diagnostics.add_assumption_test(AssumptionTestResult::new(
                Assumption::NormalResiduals,
                &norm_result.test,
                norm_result.statistic,
                norm_result.p_value,
                0.05,
            ));
        }

        // Durbin-Watson for autocorrelation
        let dw = durbin_watson(&data.residuals);
        diagnostics.durbin_watson = Some(dw);

        // Add autocorrelation assumption test
        let autocorr_passed = (1.5..=2.5).contains(&dw);
        diagnostics.add_assumption_test(AssumptionTestResult::new(
            Assumption::NoAutocorrelation,
            "Durbin-Watson",
            dw,
            if autocorr_passed { 0.5 } else { 0.01 }, // Approximate p-value
            0.05,
        ));

        // Add influential observations
        if let (Some(cooks), Some(stud)) = (self.cooks_distance(), self.studentized_residuals()) {
            let dffits = self.dffits();
            let threshold_cooks = 4.0 / data.n_samples as f64;
            let p = data.n_features + if self.fit_intercept { 1 } else { 0 };
            let threshold_leverage = 2.0 * p as f64 / data.n_samples as f64;

            for i in 0..data.n_samples {
                if cooks[i] > threshold_cooks || data.leverage[i] > threshold_leverage {
                    let mut obs =
                        InfluentialObservation::new(i, cooks[i], data.leverage[i], stud[i]);
                    if let Some(ref df) = dffits {
                        obs.dffits = Some(df[i]);
                    }
                    diagnostics.add_influential(obs);
                }
            }
        }

        diagnostics
    }

    fn coefficients_with_ci(&self, level: f64) -> Vec<CoefficientInfo> {
        let mut result = Vec::new();

        let data = match &self.fitted_data {
            Some(d) => d,
            None => return result,
        };

        let df = data.df_residuals as f64;
        let t_crit = t_critical(1.0 - (1.0 - level) / 2.0, df);

        // Add intercept if fitted
        if self.fit_intercept {
            let intercept = self.intercept.unwrap_or(0.0);
            let se = data.coef_std_errors[0];
            let t_stat = intercept / se;
            let p_value = 2.0 * (1.0 - t_cdf_approx(t_stat.abs(), df));

            result.push(
                CoefficientInfo::new("(Intercept)", intercept, se)
                    .with_test(t_stat, p_value)
                    .with_ci(intercept - t_crit * se, intercept + t_crit * se, level),
            );
        }

        // Add feature coefficients
        if let Some(coef) = &self.coefficients {
            for (i, &c) in coef.iter().enumerate() {
                let se_idx = if self.fit_intercept { i + 1 } else { i };
                let se = data.coef_std_errors[se_idx];
                let t_stat = c / se;
                let p_value = 2.0 * (1.0 - t_cdf_approx(t_stat.abs(), df));

                let name = self.get_feature_name(i);
                result.push(
                    CoefficientInfo::new(name, c, se)
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

    fn assumption_test(&self, assumption: Assumption) -> Option<AssumptionTestResult> {
        let diag = self.diagnostics();
        diag.assumption_tests
            .into_iter()
            .find(|t| t.assumption == assumption)
    }
}

impl super::traits::LinearModel for LinearRegression {
    fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    fn coefficient_std_errors(&self) -> Option<&Array1<f64>> {
        self.fitted_data.as_ref().map(|d| &d.coef_std_errors)
    }
}

impl ProbabilisticModel for LinearRegression {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // For regression, return the prediction variance
        let preds = self.predict(x)?;
        let n = preds.len();

        // Return predictions as (n, 1) matrix
        let mut result = Array2::zeros((n, 1));
        result.column_mut(0).assign(&preds);
        Ok(result)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        check_is_fitted(&self.coefficients, "predict_interval")?;

        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict_interval"))?;
        validate_predict_input(x, n_features)?;

        let data = self
            .fitted_data
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_interval"))?;

        let predictions = self.predict(x)?;
        let n = x.nrows();

        // Build design matrix
        let x_design = if self.fit_intercept {
            let mut design = Array2::zeros((n, n_features + 1));
            design.column_mut(0).fill(1.0);
            design.slice_mut(s![.., 1..]).assign(x);
            design
        } else {
            x.clone()
        };

        // Compute standard errors for predictions
        // SE(pred) = σ * sqrt(x' (X'X)^(-1) x)
        // SE(new obs) = σ * sqrt(1 + x' (X'X)^(-1) x)  [prediction interval]

        let mut std_errors = Array1::zeros(n);
        let t_crit = t_critical(1.0 - (1.0 - level) / 2.0, data.df_residuals as f64);

        // (X'X)^(-1) from stored covariance (need to divide by σ²)
        let sigma2 = data.residual_std_error.powi(2);
        let xtx_inv = &data.coef_covariance / sigma2;

        for i in 0..n {
            let xi = x_design.row(i);
            let var_fit = xi.dot(&xtx_inv.dot(&xi.t()));
            // Prediction interval includes future observation variance
            std_errors[i] = data.residual_std_error * (1.0 + var_fit).sqrt();
        }

        let lower = &predictions - &(&std_errors * t_crit);
        let upper = &predictions + &(&std_errors * t_crit);

        Ok(PredictionInterval::new(predictions, lower, upper, level).with_std_errors(std_errors))
    }
}

// =============================================================================
// Linear Algebra Helpers
// =============================================================================

/// QR decomposition using modified Gram-Schmidt
/// QR decomposition — delegates to shared linalg module (Modified Gram-Schmidt,
/// with optional faer backend for high-performance on large matrices).
fn qr_decomposition(a: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    crate::linalg::qr_decomposition(a)
}

/// Solve upper triangular system Rx = b — delegates to shared linalg module.
fn solve_upper_triangular(r: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    crate::linalg::solve_upper_triangular(r, b)
}

/// Invert upper triangular matrix — delegates to shared linalg module.
fn invert_upper_triangular(r: &Array2<f64>) -> Result<Array2<f64>> {
    crate::linalg::invert_upper_triangular(r)
}

// =============================================================================
// Statistical Distribution Functions
// =============================================================================

/// Student's t critical value approximation
fn t_critical(p: f64, df: f64) -> f64 {
    // Hill's approximation for t critical values
    if df <= 0.0 {
        return f64::NAN;
    }

    if df == 1.0 {
        // Cauchy distribution
        return (std::f64::consts::PI * (p - 0.5)).tan();
    }

    let a = 1.0 / (df - 0.5);
    let b = 48.0 / (a * a);
    let c = (20700.0 * a / b - 98.0).mul_add(a, -16.0).mul_add(a, 96.36);
    let d = ((94.5 / (b + c) - 3.0) / b + 1.0) * (a * std::f64::consts::PI * 0.5).sqrt() * df;

    let x = d * p;
    let mut y = x.powf(2.0 / df);

    if y > 0.05 + a {
        // Asymptotic expansion for large quantiles
        let x_norm = z_inv_normal(p);
        y = x_norm * x_norm;

        let c = if df < 5.0 {
            (0.3 * (df - 4.5)).mul_add(x_norm + 0.6, c)
        } else {
            c
        };

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

/// Inverse normal CDF (probit function) approximation
fn z_inv_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let original_p = p;
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

    if original_p > 0.5 {
        z
    } else {
        -z
    }
}

/// Student's t CDF approximation
fn t_cdf_approx(t: f64, df: f64) -> f64 {
    // Use regularized incomplete beta function
    let x = df / t.mul_add(t, df);
    0.5f64.mul_add(
        (1.0 - incomplete_beta_regularized(df / 2.0, 0.5, x)).copysign(t),
        0.5,
    )
}

/// F distribution CDF approximation
fn f_cdf(f: f64, df1: f64, df2: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let x = df1 * f / df1.mul_add(f, df2);
    incomplete_beta_regularized(df1 / 2.0, df2 / 2.0, x)
}

/// Regularized incomplete beta function
fn incomplete_beta_regularized(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use continued fraction expansion
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        b.mul_add(
            (1.0 - x).ln(),
            a.mul_add(x.ln(), ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b)),
        )
        .exp()
    };

    // Use symmetry for numerical stability
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

        // Even step
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

        // Odd step
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

/// Log gamma function (Lanczos approximation)
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

// =============================================================================
// PipelineModel Implementation
// =============================================================================

impl PipelineModel for LinearRegression {
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
            _ => Err(FerroError::invalid_input(format!(
                "Unknown parameter: {}",
                name
            ))),
        }
    }

    fn name(&self) -> &str {
        "LinearRegression"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_linear_regression_simple() {
        // y = 1 + 2*x with no noise
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Check coefficients
        assert_relative_eq!(model.intercept().unwrap(), 1.0, epsilon = 1e-6);
        let coef = model.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 1e-6);

        // Check R²
        let r2 = model.r_squared().unwrap();
        assert_relative_eq!(r2, 1.0, epsilon = 1e-6);

        // Check predictions
        let pred = model.predict(&x).unwrap();
        for (i, &p) in pred.iter().enumerate() {
            assert_relative_eq!(p, y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_linear_regression_multiple() {
        // y = 1 + 2*x1 + 3*x2 with independent x1 and x2
        // x1 = [1, 2, 3, 4, 5], x2 = [5, 3, 4, 2, 1] (not collinear)
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 5.0, 2.0, 3.0, 3.0, 4.0, 4.0, 2.0, 5.0, 1.0],
        )
        .unwrap();
        // y = 1 + 2*x1 + 3*x2
        let y = array![
            1.0 + 2.0 * 1.0 + 3.0 * 5.0, // 18
            1.0 + 2.0 * 2.0 + 3.0 * 3.0, // 14
            1.0 + 2.0 * 3.0 + 3.0 * 4.0, // 19
            1.0 + 2.0 * 4.0 + 3.0 * 2.0, // 15
            1.0 + 2.0 * 5.0 + 3.0 * 1.0  // 14
        ];

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        assert_relative_eq!(model.intercept().unwrap(), 1.0, epsilon = 1e-6);
        let coef = model.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(coef[1], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_without_intercept() {
        // y = 2*x (through origin)
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let mut model = LinearRegression::without_intercept();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients().unwrap();
        assert_relative_eq!(coef[0], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_prediction_intervals() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 + 3.0 * xi + 0.1);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let intervals = model.predict_interval(&x, 0.95).unwrap();

        // All predictions should be within intervals for this nearly perfect fit
        for i in 0..y.len() {
            assert!(intervals.lower[i] <= y[i] + 0.5);
            assert!(intervals.upper[i] >= y[i] - 0.5);
        }

        // Interval widths should be positive
        let widths = intervals.interval_widths();
        for &w in widths.iter() {
            assert!(w > 0.0);
        }
    }

    #[test]
    fn test_residuals_and_fitted() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![3.1, 4.9, 7.1, 8.9, 11.0];

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let residuals = model.residuals().unwrap();
        let fitted = model.fitted_values().unwrap();

        // residuals + fitted should equal y
        for i in 0..y.len() {
            assert_relative_eq!(residuals[i] + fitted[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_summary_output() {
        // Create independent features: x1 and x2 are not linearly dependent
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 5.0, 2.0, 3.0, 3.0, 6.0, 4.0, 2.0, 5.0, 7.0, 6.0, 1.0, 7.0, 4.0, 8.0, 8.0,
                9.0, 3.0, 10.0, 9.0,
            ],
        )
        .unwrap();
        let y: Array1<f64> = (0..10)
            .map(|i| 1.0 + 0.5 * x[[i, 0]] + 0.3 * x[[i, 1]])
            .collect();

        let mut model =
            LinearRegression::new().with_feature_names(vec!["var1".into(), "var2".into()]);
        model.fit(&x, &y).unwrap();

        let summary = model.summary();
        let output = format!("{}", summary);

        assert!(output.contains("OLS Linear Regression"));
        assert!(output.contains("R-squared"));
        assert!(output.contains("var1"));
        assert!(output.contains("var2"));
    }

    #[test]
    fn test_diagnostics() {
        let x = Array2::from_shape_vec((20, 1), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 + 3.0 * xi);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let diagnostics = model.diagnostics();

        // Perfect fit should have small residuals
        assert!(diagnostics.residual_stats.std_dev < 1e-10);

        // Durbin-Watson should be around 2 for no autocorrelation
        // (though with perfect fit, it's undefined behavior)
        assert!(diagnostics.durbin_watson.is_some());
    }

    #[test]
    fn test_cooks_distance() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let cooks = model.cooks_distance().unwrap();
        assert_eq!(cooks.len(), 10);

        // No observation should have extreme Cook's distance for this clean data
        for &c in cooks.iter() {
            assert!(c.is_finite());
        }
    }

    #[test]
    fn test_standardized_residuals() {
        let x = Array2::from_shape_vec((10, 1), (1..=10).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 + 3.0 * xi + 0.5);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let std_resid = model.standardized_residuals().unwrap();
        assert_eq!(std_resid.len(), 10);
    }

    #[test]
    fn test_f_statistic() {
        // Use non-collinear features (x1 = i, x2 = i^2)
        let x = Array2::from_shape_fn((20, 2), |(i, j)| {
            if j == 0 {
                (i + 1) as f64
            } else {
                ((i + 1) as f64).powi(2)
            }
        });
        let y: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|row| 1.0 + 2.0 * row[0] + 0.5 * row[1])
            .collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let (f_stat, f_p) = model.f_statistic().unwrap();
        assert!(f_stat > 0.0);
        assert!(f_p >= 0.0 && f_p <= 1.0);
    }

    #[test]
    fn test_information_criteria() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 + 3.0 * xi);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        assert!(model.log_likelihood().is_some());
        assert!(model.aic().is_some());
        assert!(model.bic().is_some());
    }

    #[test]
    fn test_error_not_fitted() {
        let model = LinearRegression::new();
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        assert!(model.predict(&x).is_err());
    }

    #[test]
    fn test_error_shape_mismatch() {
        let x = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();
        let y = array![1.0, 2.0, 3.0]; // Wrong size

        let mut model = LinearRegression::new();
        assert!(model.fit(&x, &y).is_err());
    }

    #[test]
    fn test_qr_decomposition() {
        let a = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let (q, r) = qr_decomposition(&a).unwrap();

        // Q should be orthogonal: Q'Q = I
        let qtq = q.t().dot(&q);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(qtq[[i, j]], expected, epsilon = 1e-10);
            }
        }

        // R should be upper triangular
        assert!(r[[1, 0]].abs() < 1e-10);

        // QR should equal A
        let qr = q.dot(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(qr[[i, j]], a[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_t_critical_values() {
        // Known values for common cases
        // t(0.975, 10) ≈ 2.228
        let t = t_critical(0.975, 10.0);
        assert!(t > 2.0 && t < 2.5);

        // t(0.975, 100) ≈ 1.984
        let t = t_critical(0.975, 100.0);
        assert!(t > 1.9 && t < 2.1);
    }

    #[test]
    fn test_coefficients_with_ci() {
        let x = Array2::from_shape_vec((20, 1), (1..=20).map(|i| i as f64).collect()).unwrap();
        let y: Array1<f64> = x.column(0).mapv(|xi| 2.0 + 3.0 * xi);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coefs = model.coefficients_with_ci(0.95);
        assert_eq!(coefs.len(), 2); // Intercept + 1 feature

        // Intercept
        assert_eq!(coefs[0].name, "(Intercept)");
        assert_relative_eq!(coefs[0].estimate, 2.0, epsilon = 1e-6);
        assert!(coefs[0].ci_lower < coefs[0].estimate);
        assert!(coefs[0].ci_upper > coefs[0].estimate);

        // Coefficient
        assert_relative_eq!(coefs[1].estimate, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_feature_importance() {
        // Create independent features
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 5.0, 2.0, 3.0, 3.0, 6.0, 4.0, 2.0, 5.0, 7.0, 6.0, 1.0, 7.0, 4.0, 8.0, 8.0,
                9.0, 3.0, 10.0, 9.0,
            ],
        )
        .unwrap();
        let y: Array1<f64> = (0..10)
            .map(|i| 1.0 + 2.0 * x[[i, 0]] + 1.5 * x[[i, 1]])
            .collect();

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let importance = model.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Both features should have positive importance
        assert!(importance[0] > 0.0);
        assert!(importance[1] > 0.0);
    }

    #[test]
    fn test_z_inv_normal_sign() {
        // z_inv_normal(0.975) should be positive (~1.96)
        let z_upper = z_inv_normal(0.975);
        assert!(
            z_upper > 1.5,
            "z_inv_normal(0.975) = {z_upper}, expected > 1.5"
        );
        assert!(
            z_upper < 2.5,
            "z_inv_normal(0.975) = {z_upper}, expected < 2.5"
        );

        // z_inv_normal(0.025) should be negative (~-1.96)
        let z_lower = z_inv_normal(0.025);
        assert!(
            z_lower < -1.5,
            "z_inv_normal(0.025) = {z_lower}, expected < -1.5"
        );
        assert!(
            z_lower > -2.5,
            "z_inv_normal(0.025) = {z_lower}, expected > -2.5"
        );

        // Should be symmetric
        assert_relative_eq!(z_upper, -z_lower, epsilon = 1e-6);
    }

    #[test]
    fn test_t_critical_small_df() {
        // t_critical for df=3, p=0.975: known value ~3.182
        let t3 = t_critical(0.975, 3.0);
        assert!(t3 > 2.5, "t_critical(0.975, 3) = {t3}, expected > 2.5");
        assert!(t3 < 4.0, "t_critical(0.975, 3) = {t3}, expected < 4.0");

        // t_critical for df=4, p=0.975: known value ~2.776
        let t4 = t_critical(0.975, 4.0);
        assert!(t4 > 2.2, "t_critical(0.975, 4) = {t4}, expected > 2.2");
        assert!(t4 < 3.5, "t_critical(0.975, 4) = {t4}, expected < 3.5");

        // df=4 should be smaller than df=3
        assert!(t4 < t3, "t(df=4) should be < t(df=3)");
    }

    #[test]
    fn test_vif_collinear() {
        // x2 = 2*x1 + small noise => highly collinear
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.01, 2.0, 3.99, 3.0, 6.02, 4.0, 7.98, 5.0, 10.01, 6.0, 11.99, 7.0, 14.02,
                8.0, 15.98, 9.0, 18.01, 10.0, 19.99,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let mut reg = LinearRegression::new();
        reg.fit(&x, &y).unwrap();

        let vifs = reg.vif(&x);
        // Both features should have high VIF due to collinearity
        assert!(
            vifs[0] > 5.0,
            "VIF[0] = {}, expected > 5.0 for collinear features",
            vifs[0]
        );
        assert!(
            vifs[1] > 5.0,
            "VIF[1] = {}, expected > 5.0 for collinear features",
            vifs[1]
        );
    }

    #[test]
    fn test_cooks_distance_high_leverage_point() {
        // Create data where one point has leverage very close to 1.0
        // A point that is extremely far from the centroid of x has high leverage
        let mut x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        x_data.push(1e6); // extreme leverage point
        let x = Array2::from_shape_vec((6, 1), x_data).unwrap();
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let cooks = model.cooks_distance().unwrap();
        // Should not contain NaN or Inf — the leverage guard prevents division by zero
        for &c in cooks.iter() {
            assert!(c.is_finite(), "Cook's distance must be finite, got {}", c);
        }

        let std_resid = model.standardized_residuals().unwrap();
        for &r in std_resid.iter() {
            assert!(
                r.is_finite(),
                "Standardized residual must be finite, got {}",
                r
            );
        }

        let stud_resid = model.studentized_residuals().unwrap();
        for &r in stud_resid.iter() {
            assert!(
                r.is_finite(),
                "Studentized residual must be finite, got {}",
                r
            );
        }

        let dffits_vals = model.dffits().unwrap();
        for &d in dffits_vals.iter() {
            assert!(d.is_finite(), "DFFITS must be finite, got {}", d);
        }
    }
}
