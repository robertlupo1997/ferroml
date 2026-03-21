//! Machine Learning Models with Statistical Diagnostics
//!
//! This module provides ML models with full statistical diagnostics - FerroML's
//! key differentiator from traditional ML libraries.
//!
//! ## Design Philosophy
//!
//! Every model in FerroML provides:
//! - **Coefficient uncertainty**: Standard errors, confidence intervals, p-values
//! - **Model diagnostics**: Residual analysis, assumption tests
//! - **Prediction intervals**: Not just point predictions, but uncertainty bands
//!
//! ## Trait Hierarchy
//!
//! ```text
//! Model (base trait)
//!   ├── StatisticalModel (linear models with diagnostics)
//!   └── ProbabilisticModel (classifiers with probabilities)
//! ```
//!
//! ## Planned Models
//!
//! - **Linear**: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LogisticRegression`
//! - **Trees**: `DecisionTree`, `RandomForest`, `GradientBoosting`
//! - **SVM**: `SVC`, `SVR`

use crate::hpo::SearchSpace;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

// Submodules
pub mod adaboost;
pub mod boosting;
pub mod calibration;
pub mod extra_trees;
pub mod forest;
pub mod gaussian_process;
pub mod hist_boosting;
pub mod isolation_forest;
pub mod isotonic;
pub mod knn;
pub mod linear;
pub mod lof;
pub mod logistic;
pub mod multioutput;
pub mod naive_bayes;
pub mod qda;
pub mod quantile;
pub mod regularized;
pub mod robust;
pub mod sgd;
pub mod svm;
pub mod traits;
pub mod tree;

#[cfg(test)]
mod compliance_tests;

// Re-export models for convenience
pub use adaboost::{AdaBoostClassifier, AdaBoostLoss, AdaBoostRegressor};
pub use boosting::{
    ClassificationLoss, EarlyStopping, GradientBoostingClassifier, GradientBoostingRegressor,
    LearningRateSchedule, RegressionLoss, TrainingHistory,
};
pub use calibration::{
    calibration_curve, CalibrableClassifier, CalibratedClassifierCV, CalibrationMethod,
    CalibrationResult, Calibrator, IsotonicCalibrator, SigmoidCalibrator,
};
pub use extra_trees::{ExtraTreesClassifier, ExtraTreesRegressor};
pub use forest::{
    FeatureImportanceWithCI, MaxFeatures, RandomForestClassifier, RandomForestRegressor,
};
pub use gaussian_process::{
    ConstantKernel, GaussianProcessClassifier, GaussianProcessRegressor, Kernel as GPKernel,
    Matern, ProductKernel, SumKernel, WhiteKernel, RBF,
};
pub use hist_boosting::{
    BinMapper, GrowthStrategy, HistEarlyStopping, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, HistLoss, HistRegressionLoss, HistTree, HistTreeNode,
    MonotonicConstraint,
};
pub use isolation_forest::{Contamination, IsolationForest, MaxSamples as IFMaxSamples};
pub use isotonic::{Increasing, IsotonicRegression, OutOfBounds};
pub use knn::{
    BallTree, DistanceMetric, KDTree, KNNAlgorithm, KNNWeights, KNeighborsClassifier,
    KNeighborsRegressor, NearestCentroid,
};
pub use linear::LinearRegression;
pub use lof::LocalOutlierFactor;
pub use logistic::{LogisticRegression, LogisticSolver, OddsRatioInfo};
pub use multioutput::{MultiOutputClassifier, MultiOutputRegressor};
pub use naive_bayes::{BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB};
pub use qda::QuadraticDiscriminantAnalysis;
pub use quantile::{MultiQuantileResults, QuantileRegression};
pub use regularized::{
    elastic_net_path, lasso_path, ElasticNet, ElasticNetCV, LassoCV, LassoRegression,
    RegularizationPath, RidgeCV, RidgeClassifier, RidgeRegression,
};
pub use robust::{MEstimator, RobustRegression, ScaleMethod};
pub use sgd::{
    LearningRateScheduleType as SGDLearningRate, PassiveAggressiveClassifier, Penalty, Perceptron,
    SGDClassifier, SGDClassifierLoss, SGDRegressor, SGDRegressorLoss,
};
pub use svm::{
    ClassWeight, Kernel, LinearSVC, LinearSVCLoss, LinearSVR, LinearSVRLoss, MulticlassStrategy,
    SVC, SVR,
};
pub use traits::*;
pub use tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion, SplitStrategy, TreeNode,
    TreeStructure,
};

/// Core trait for all machine learning models
///
/// This trait provides the fundamental interface for training and prediction.
/// All FerroML models implement this trait.
///
/// # Example
///
/// ```
/// use ferroml_core::models::{Model, LinearRegression};
/// use ndarray::{Array1, Array2};
///
/// // Non-collinear features: y ≈ 2*x1 + 3*x2
/// let x = Array2::from_shape_vec((4, 2), vec![1., 1., 2., 1., 1., 2., 3., 2.]).unwrap();
/// let y = Array1::from_vec(vec![5., 7., 8., 12.]);
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y).unwrap();
/// let predictions = model.predict(&x).unwrap();
/// assert_eq!(predictions.len(), 4);
/// ```
pub trait Model: Send + Sync {
    /// Fit the model to training data
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    ///
    /// # Errors
    /// Returns an error if the model cannot be fitted (e.g., numerical issues,
    /// invalid input shapes)
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;

    /// Predict target values for new samples
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Errors
    /// Returns `NotFitted` error if called before `fit()`, or other errors
    /// for invalid input
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Check if the model has been fitted
    fn is_fitted(&self) -> bool;

    /// Get feature importance scores (if available)
    ///
    /// Returns `None` if the model doesn't support feature importance.
    /// For linear models, this could be absolute coefficient values.
    /// For tree models, this is typically Gini importance.
    fn feature_importance(&self) -> Option<Array1<f64>> {
        None
    }

    /// Get the hyperparameter search space for this model
    ///
    /// Used by HPO to tune model hyperparameters.
    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    /// Get feature names (if set during fitting)
    fn feature_names(&self) -> Option<&[String]> {
        None
    }

    /// Get the number of features expected by the model
    ///
    /// Returns `None` if the model hasn't been fitted yet.
    fn n_features(&self) -> Option<usize>;

    /// Fit with sample weights (optional, returns NotImplemented by default)
    fn fit_weighted(
        &mut self,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
        _sample_weight: &Array1<f64>,
    ) -> Result<()> {
        Err(FerroError::NotImplemented(format!(
            "fit_weighted not implemented for {}",
            self.model_name()
        )))
    }

    /// Predict class probabilities if supported.
    ///
    /// Returns `None` by default. Override for classifiers that support
    /// probability predictions.
    fn try_predict_proba(&self, _x: &Array2<f64>) -> Option<Result<Array2<f64>>> {
        None
    }

    /// Get model name for error messages
    fn model_name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Downcast to concrete type (for trait-specific access)
    fn as_any(&self) -> &dyn std::any::Any
    where
        Self: Sized + 'static,
    {
        self
    }

    /// Downcast to concrete type (mutable)
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any
    where
        Self: Sized + 'static,
    {
        self
    }

    /// Score the model on test data.
    ///
    /// The default implementation computes accuracy (fraction of predictions
    /// matching the true values within a tolerance of 1e-10), which is
    /// appropriate for classifiers. Regression models should override this
    /// to return R² (coefficient of determination).
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - True target values of shape (n_samples,)
    ///
    /// # Errors
    /// Returns an error if prediction fails (e.g., model not fitted).
    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        let n = y.len() as f64;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-10)
            .count() as f64;
        Ok(correct / n)
    }
}

/// Extended trait for models with statistical diagnostics
///
/// This is FerroML's key differentiator - providing R-style statistical output
/// for every model that supports it. Linear models, GLMs, and other parametric
/// models implement this trait.
///
/// # Example
///
/// ```
/// # use ferroml_core::models::{Model, StatisticalModel, LinearRegression};
/// # use ndarray::{Array1, Array2};
/// # fn main() -> ferroml_core::Result<()> {
/// # let x = Array2::from_shape_vec((5, 2), vec![1.0,0.5,2.0,1.5,3.0,0.8,4.0,2.1,5.0,1.0]).unwrap();
/// # let y = Array1::from_vec(vec![1.1, 2.3, 2.9, 4.2, 5.0]);
/// use ferroml_core::models::{Model, StatisticalModel};
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y)?;
///
/// // Get R-style summary
/// let summary = model.summary();
/// println!("{}", summary);
///
/// // Check residual diagnostics
/// let diagnostics = model.diagnostics();
/// if !diagnostics.normality_ok() {
///     println!("Warning: Residuals may not be normally distributed");
/// }
/// # Ok(())
/// # }
/// ```
pub trait StatisticalModel: Model {
    /// Get comprehensive model summary (R-style output)
    ///
    /// Includes coefficients, standard errors, t-statistics, p-values,
    /// R², adjusted R², F-statistic, and other relevant statistics.
    fn summary(&self) -> ModelSummary;

    /// Get model diagnostics
    ///
    /// Includes residual analysis, assumption tests, and influential point detection.
    fn diagnostics(&self) -> Diagnostics;

    /// Get coefficients with confidence intervals
    ///
    /// # Arguments
    /// * `level` - Confidence level (e.g., 0.95 for 95% CI)
    fn coefficients_with_ci(&self, level: f64) -> Vec<CoefficientInfo>;

    /// Get the residuals from the fitted model
    fn residuals(&self) -> Option<Array1<f64>>;

    /// Get the fitted values (predicted values for training data)
    fn fitted_values(&self) -> Option<Array1<f64>>;

    /// Check if a specific assumption test passes
    fn assumption_test(&self, assumption: Assumption) -> Option<AssumptionTestResult>;
}

/// Extended trait for models that output probabilities
///
/// Classification models that can estimate class probabilities implement this trait.
/// Also useful for regression models that can provide prediction intervals.
///
/// # Example
///
/// ```
/// # use ferroml_core::models::{Model, ProbabilisticModel, LogisticRegression};
/// # use ndarray::{Array1, Array2};
/// # fn main() -> ferroml_core::Result<()> {
/// # let x = Array2::from_shape_vec((6, 2), vec![1.0,2.0,2.0,1.0,3.0,3.0,6.0,7.0,7.0,6.0,8.0,8.0]).unwrap();
/// # let y = Array1::from_vec(vec![0.0,0.0,0.0,1.0,1.0,1.0]);
/// # let x_test = x.clone();
/// use ferroml_core::models::{Model, ProbabilisticModel};
///
/// let mut model = LogisticRegression::new();
/// model.fit(&x, &y)?;
///
/// // Get probability predictions
/// let probas = model.predict_proba(&x_test)?;
///
/// // Get prediction interval for regression
/// let interval = model.predict_interval(&x_test, 0.95)?;
/// # Ok(())
/// # }
/// ```
pub trait ProbabilisticModel: Model {
    /// Predict class probabilities
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// Probability matrix of shape (n_samples, n_classes) for classification,
    /// or (n_samples, 1) for binary classification / regression uncertainty.
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Predict with prediction intervals
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `level` - Confidence level (e.g., 0.95 for 95% prediction interval)
    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval>;

    /// Get log-probabilities (for numerical stability)
    fn predict_log_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let proba = self.predict_proba(x)?;
        Ok(proba.mapv(|p| p.max(1e-15).ln()))
    }
}

/// R-style model summary output
///
/// Provides comprehensive statistics about a fitted model in a format
/// similar to R's `summary()` output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Model type name (e.g., "OLS Linear Regression")
    pub model_type: String,
    /// Number of observations
    pub n_observations: usize,
    /// Number of features (predictors)
    pub n_features: usize,
    /// Dependent variable name (if known)
    pub dependent_var: Option<String>,
    /// Coefficient table
    pub coefficients: Vec<CoefficientInfo>,
    /// Overall model fit statistics
    pub fit_statistics: FitStatistics,
    /// Information about assumption tests
    pub assumption_tests: Vec<AssumptionTestResult>,
    /// Warnings or notes about the model
    pub notes: Vec<String>,
}

impl ModelSummary {
    /// Create a new model summary
    pub fn new(model_type: impl Into<String>, n_observations: usize, n_features: usize) -> Self {
        Self {
            model_type: model_type.into(),
            n_observations,
            n_features,
            dependent_var: None,
            coefficients: Vec::new(),
            fit_statistics: FitStatistics::default(),
            assumption_tests: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Set the dependent variable name
    pub fn with_dependent_var(mut self, name: impl Into<String>) -> Self {
        self.dependent_var = Some(name.into());
        self
    }

    /// Add a coefficient to the summary
    pub fn add_coefficient(&mut self, coef: CoefficientInfo) {
        self.coefficients.push(coef);
    }

    /// Set the fit statistics
    pub fn with_fit_statistics(mut self, stats: FitStatistics) -> Self {
        self.fit_statistics = stats;
        self
    }

    /// Add an assumption test result
    pub fn add_assumption_test(&mut self, test: AssumptionTestResult) {
        self.assumption_tests.push(test);
    }

    /// Add a note to the summary
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Check if any coefficients are statistically significant at the given level
    pub fn has_significant_coefficients(&self, alpha: f64) -> bool {
        self.coefficients.iter().any(|c| c.p_value < alpha)
    }

    /// Get the number of significant coefficients at the given level
    pub fn count_significant(&self, alpha: f64) -> usize {
        self.coefficients
            .iter()
            .filter(|c| c.p_value < alpha)
            .count()
    }
}

impl fmt::Display for ModelSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", "=".repeat(72))?;
        writeln!(f, "{:^72}", self.model_type)?;
        writeln!(f, "{}", "=".repeat(72))?;

        if let Some(ref dep_var) = self.dependent_var {
            writeln!(f, "Dep. Variable: {:>20}", dep_var)?;
        }
        writeln!(f, "No. Observations: {:>17}", self.n_observations)?;
        writeln!(f, "No. Features: {:>21}", self.n_features)?;
        writeln!(f)?;

        // Fit statistics
        writeln!(f, "{:-<72}", "")?;
        writeln!(f, "{:^72}", "Model Fit Statistics")?;
        writeln!(f, "{:-<72}", "")?;

        if let Some(r2) = self.fit_statistics.r_squared {
            writeln!(f, "R-squared: {:>24.4}", r2)?;
        }
        if let Some(adj_r2) = self.fit_statistics.adj_r_squared {
            writeln!(f, "Adj. R-squared: {:>19.4}", adj_r2)?;
        }
        if let Some(f_stat) = self.fit_statistics.f_statistic {
            writeln!(f, "F-statistic: {:>22.4}", f_stat)?;
        }
        if let Some(f_p) = self.fit_statistics.f_p_value {
            writeln!(f, "Prob (F-statistic): {:>15.4e}", f_p)?;
        }
        if let Some(ll) = self.fit_statistics.log_likelihood {
            writeln!(f, "Log-Likelihood: {:>19.4}", ll)?;
        }
        if let Some(aic) = self.fit_statistics.aic {
            writeln!(f, "AIC: {:>30.4}", aic)?;
        }
        if let Some(bic) = self.fit_statistics.bic {
            writeln!(f, "BIC: {:>30.4}", bic)?;
        }
        writeln!(f)?;

        // Coefficients table
        writeln!(f, "{:-<72}", "")?;
        writeln!(
            f,
            "{:>15} {:>12} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Coef", "Std.Err", "t-stat", "P>|t|", "Signif"
        )?;
        writeln!(f, "{:-<72}", "")?;

        for coef in &self.coefficients {
            let signif = if coef.p_value < 0.001 {
                "***"
            } else if coef.p_value < 0.01 {
                "**"
            } else if coef.p_value < 0.05 {
                "*"
            } else if coef.p_value < 0.1 {
                "."
            } else {
                ""
            };

            writeln!(
                f,
                "{:>15} {:>12.4} {:>10.4} {:>10.3} {:>10.4} {:>10}",
                truncate_string(&coef.name, 15),
                coef.estimate,
                coef.std_error,
                coef.t_statistic,
                coef.p_value,
                signif
            )?;
        }

        writeln!(f, "{:-<72}", "")?;
        writeln!(
            f,
            "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )?;

        // Assumption tests
        if !self.assumption_tests.is_empty() {
            writeln!(f)?;
            writeln!(f, "{:-<72}", "")?;
            writeln!(f, "{:^72}", "Assumption Tests")?;
            writeln!(f, "{:-<72}", "")?;

            for test in &self.assumption_tests {
                let status = if test.passed { "PASS" } else { "FAIL" };
                writeln!(
                    f,
                    "{}: {} ({}: p={:.4})",
                    test.assumption, status, test.test_name, test.p_value
                )?;
            }
        }

        // Notes
        if !self.notes.is_empty() {
            writeln!(f)?;
            writeln!(f, "Notes:")?;
            for note in &self.notes {
                writeln!(f, "  - {}", note)?;
            }
        }

        writeln!(f, "{}", "=".repeat(72))?;
        Ok(())
    }
}

/// Information about a single coefficient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoefficientInfo {
    /// Name of the variable/feature
    pub name: String,
    /// Point estimate of the coefficient
    pub estimate: f64,
    /// Standard error of the estimate
    pub std_error: f64,
    /// t-statistic (or z-statistic for GLMs)
    pub t_statistic: f64,
    /// p-value for the test that coefficient = 0
    pub p_value: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Confidence level used for CI
    pub confidence_level: f64,
    /// Variance Inflation Factor (multicollinearity indicator)
    pub vif: Option<f64>,
}

impl CoefficientInfo {
    /// Create a new coefficient info
    pub fn new(name: impl Into<String>, estimate: f64, std_error: f64) -> Self {
        Self {
            name: name.into(),
            estimate,
            std_error,
            t_statistic: 0.0,
            p_value: 1.0,
            ci_lower: f64::NEG_INFINITY,
            ci_upper: f64::INFINITY,
            confidence_level: 0.95,
            vif: None,
        }
    }

    /// Set t-statistic and p-value
    pub fn with_test(mut self, t_statistic: f64, p_value: f64) -> Self {
        self.t_statistic = t_statistic;
        self.p_value = p_value;
        self
    }

    /// Set confidence interval
    pub fn with_ci(mut self, lower: f64, upper: f64, level: f64) -> Self {
        self.ci_lower = lower;
        self.ci_upper = upper;
        self.confidence_level = level;
        self
    }

    /// Set VIF
    pub fn with_vif(mut self, vif: f64) -> Self {
        self.vif = Some(vif);
        self
    }

    /// Check if the coefficient is statistically significant
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Check if the CI excludes zero
    pub fn ci_excludes_zero(&self) -> bool {
        self.ci_lower > 0.0 || self.ci_upper < 0.0
    }

    /// Check if there's potential multicollinearity (VIF > 5 or > 10 is common threshold)
    pub fn has_multicollinearity(&self, threshold: f64) -> bool {
        self.vif.map(|v| v > threshold).unwrap_or(false)
    }
}

/// Overall model fit statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FitStatistics {
    /// Coefficient of determination
    pub r_squared: Option<f64>,
    /// Adjusted R-squared
    pub adj_r_squared: Option<f64>,
    /// F-statistic for overall model significance
    pub f_statistic: Option<f64>,
    /// p-value for F-statistic
    pub f_p_value: Option<f64>,
    /// Log-likelihood
    pub log_likelihood: Option<f64>,
    /// Akaike Information Criterion
    pub aic: Option<f64>,
    /// Bayesian Information Criterion
    pub bic: Option<f64>,
    /// Root Mean Squared Error
    pub rmse: Option<f64>,
    /// Mean Absolute Error
    pub mae: Option<f64>,
    /// Residual standard error
    pub residual_std_error: Option<f64>,
    /// Degrees of freedom for residuals
    pub df_residuals: Option<usize>,
    /// Degrees of freedom for model
    pub df_model: Option<usize>,
}

impl FitStatistics {
    /// Create new fit statistics with R-squared
    pub fn with_r_squared(r2: f64, adj_r2: f64) -> Self {
        Self {
            r_squared: Some(r2),
            adj_r_squared: Some(adj_r2),
            ..Default::default()
        }
    }

    /// Set F-statistic and p-value
    pub fn with_f_test(mut self, f_stat: f64, p_value: f64) -> Self {
        self.f_statistic = Some(f_stat);
        self.f_p_value = Some(p_value);
        self
    }

    /// Set information criteria
    pub fn with_information_criteria(mut self, ll: f64, aic: f64, bic: f64) -> Self {
        self.log_likelihood = Some(ll);
        self.aic = Some(aic);
        self.bic = Some(bic);
        self
    }

    /// Set error metrics
    pub fn with_errors(mut self, rmse: f64, mae: f64) -> Self {
        self.rmse = Some(rmse);
        self.mae = Some(mae);
        self
    }

    /// Set degrees of freedom
    pub fn with_df(mut self, df_model: usize, df_residuals: usize) -> Self {
        self.df_model = Some(df_model);
        self.df_residuals = Some(df_residuals);
        self
    }

    /// Check if the overall model is significant
    pub fn is_model_significant(&self, alpha: f64) -> bool {
        self.f_p_value.map(|p| p < alpha).unwrap_or(false)
    }
}

/// Model diagnostics including residual analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    /// Residual statistics
    pub residual_stats: ResidualStatistics,
    /// Assumption test results
    pub assumption_tests: Vec<AssumptionTestResult>,
    /// Influential observations (high Cook's distance)
    pub influential_observations: Vec<InfluentialObservation>,
    /// Condition number (multicollinearity indicator)
    pub condition_number: Option<f64>,
    /// Durbin-Watson statistic (autocorrelation)
    pub durbin_watson: Option<f64>,
}

impl Diagnostics {
    /// Create new diagnostics
    pub fn new(residual_stats: ResidualStatistics) -> Self {
        Self {
            residual_stats,
            assumption_tests: Vec::new(),
            influential_observations: Vec::new(),
            condition_number: None,
            durbin_watson: None,
        }
    }

    /// Add an assumption test result
    pub fn add_assumption_test(&mut self, test: AssumptionTestResult) {
        self.assumption_tests.push(test);
    }

    /// Add an influential observation
    pub fn add_influential(&mut self, obs: InfluentialObservation) {
        self.influential_observations.push(obs);
    }

    /// Check if normality assumption holds
    pub fn normality_ok(&self) -> bool {
        self.assumption_tests
            .iter()
            .find(|t| t.assumption == Assumption::NormalResiduals)
            .map(|t| t.passed)
            .unwrap_or(true)
    }

    /// Check if homoscedasticity assumption holds
    pub fn homoscedasticity_ok(&self) -> bool {
        self.assumption_tests
            .iter()
            .find(|t| t.assumption == Assumption::Homoscedasticity)
            .map(|t| t.passed)
            .unwrap_or(true)
    }

    /// Check if no autocorrelation assumption holds
    pub fn no_autocorrelation_ok(&self) -> bool {
        // Durbin-Watson should be around 2 (1.5-2.5 is often acceptable)
        self.durbin_watson
            .map(|dw| (1.5..=2.5).contains(&dw))
            .unwrap_or(true)
    }

    /// Check if multicollinearity is low
    pub fn multicollinearity_ok(&self, threshold: f64) -> bool {
        self.condition_number
            .map(|cn| cn < threshold)
            .unwrap_or(true)
    }

    /// Get all assumptions that failed
    pub fn failed_assumptions(&self) -> Vec<&AssumptionTestResult> {
        self.assumption_tests.iter().filter(|t| !t.passed).collect()
    }

    /// Check if all core assumptions pass
    pub fn all_assumptions_ok(&self) -> bool {
        self.assumption_tests.iter().all(|t| t.passed)
    }
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model Diagnostics")?;
        writeln!(f, "=================")?;
        writeln!(f)?;

        // Residual statistics
        writeln!(f, "Residual Statistics:")?;
        writeln!(f, "  Min:     {:>12.4}", self.residual_stats.min)?;
        writeln!(f, "  1Q:      {:>12.4}", self.residual_stats.q1)?;
        writeln!(f, "  Median:  {:>12.4}", self.residual_stats.median)?;
        writeln!(f, "  3Q:      {:>12.4}", self.residual_stats.q3)?;
        writeln!(f, "  Max:     {:>12.4}", self.residual_stats.max)?;
        writeln!(f)?;

        // Assumption tests
        if !self.assumption_tests.is_empty() {
            writeln!(f, "Assumption Tests:")?;
            for test in &self.assumption_tests {
                let status = if test.passed { "PASS" } else { "FAIL" };
                writeln!(
                    f,
                    "  {} - {}: {} (p={:.4})",
                    test.assumption, test.test_name, status, test.p_value
                )?;
            }
            writeln!(f)?;
        }

        // Additional diagnostics
        if let Some(dw) = self.durbin_watson {
            writeln!(f, "Durbin-Watson: {:.4}", dw)?;
        }
        if let Some(cn) = self.condition_number {
            writeln!(f, "Condition Number: {:.2}", cn)?;
        }

        // Influential observations
        if !self.influential_observations.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "Influential Observations (Cook's distance > 4/n or leverage > 2p/n):"
            )?;
            for obs in &self.influential_observations {
                writeln!(
                    f,
                    "  Observation {}: Cook's D = {:.4}, Leverage = {:.4}",
                    obs.index, obs.cooks_distance, obs.leverage
                )?;
            }
        }

        Ok(())
    }
}

/// Residual statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualStatistics {
    /// Minimum residual
    pub min: f64,
    /// First quartile
    pub q1: f64,
    /// Median residual
    pub median: f64,
    /// Third quartile
    pub q3: f64,
    /// Maximum residual
    pub max: f64,
    /// Mean residual (should be ~0 for unbiased models)
    pub mean: f64,
    /// Standard deviation of residuals
    pub std_dev: f64,
    /// Skewness of residuals
    pub skewness: f64,
    /// Kurtosis of residuals
    pub kurtosis: f64,
}

impl ResidualStatistics {
    /// Create from residual array
    pub fn from_residuals(residuals: &Array1<f64>) -> Self {
        let n = residuals.len();
        if n == 0 {
            return Self::default();
        }

        let mut sorted: Vec<f64> = residuals.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[n - 1];
        let median = percentile(&sorted, 0.5);
        let q1 = percentile(&sorted, 0.25);
        let q3 = percentile(&sorted, 0.75);

        let mean = residuals.mean().unwrap_or(0.0);
        let variance = residuals.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        let std_dev = variance.sqrt();

        // Skewness and kurtosis
        let (skewness, kurtosis) = if std_dev > 0.0 && n > 2 {
            let m3 = residuals
                .mapv(|x| ((x - mean) / std_dev).powi(3))
                .mean()
                .unwrap_or(0.0);
            let m4 = residuals
                .mapv(|x| ((x - mean) / std_dev).powi(4))
                .mean()
                .unwrap_or(3.0);
            (m3, m4 - 3.0) // excess kurtosis
        } else {
            (0.0, 0.0)
        };

        Self {
            min,
            q1,
            median,
            q3,
            max,
            mean,
            std_dev,
            skewness,
            kurtosis,
        }
    }
}

impl Default for ResidualStatistics {
    fn default() -> Self {
        Self {
            min: 0.0,
            q1: 0.0,
            median: 0.0,
            q3: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}

/// Statistical assumptions that can be tested
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Assumption {
    /// Residuals are normally distributed
    NormalResiduals,
    /// Constant variance of residuals (homoscedasticity)
    Homoscedasticity,
    /// No autocorrelation in residuals
    NoAutocorrelation,
    /// Linear relationship between X and Y
    Linearity,
    /// No multicollinearity among predictors
    NoMulticollinearity,
    /// Independence of observations
    Independence,
}

impl fmt::Display for Assumption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Assumption::NormalResiduals => write!(f, "Normal Residuals"),
            Assumption::Homoscedasticity => write!(f, "Homoscedasticity"),
            Assumption::NoAutocorrelation => write!(f, "No Autocorrelation"),
            Assumption::Linearity => write!(f, "Linearity"),
            Assumption::NoMulticollinearity => write!(f, "No Multicollinearity"),
            Assumption::Independence => write!(f, "Independence"),
        }
    }
}

/// Result of testing a statistical assumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionTestResult {
    /// The assumption being tested
    pub assumption: Assumption,
    /// Name of the statistical test used
    pub test_name: String,
    /// Test statistic value
    pub statistic: f64,
    /// p-value from the test
    pub p_value: f64,
    /// Whether the assumption is considered to hold (p > alpha)
    pub passed: bool,
    /// Alpha level used for the test
    pub alpha: f64,
    /// Optional additional details
    pub details: Option<String>,
}

impl AssumptionTestResult {
    /// Create a new assumption test result
    pub fn new(
        assumption: Assumption,
        test_name: impl Into<String>,
        statistic: f64,
        p_value: f64,
        alpha: f64,
    ) -> Self {
        Self {
            assumption,
            test_name: test_name.into(),
            statistic,
            p_value,
            passed: p_value > alpha,
            alpha,
            details: None,
        }
    }

    /// Add details to the test result
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

/// Information about an influential observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfluentialObservation {
    /// Index of the observation
    pub index: usize,
    /// Cook's distance
    pub cooks_distance: f64,
    /// Leverage (hat value)
    pub leverage: f64,
    /// Standardized residual
    pub std_residual: f64,
    /// Studentized residual (externally studentized)
    pub studentized_residual: Option<f64>,
    /// DFFITS value
    pub dffits: Option<f64>,
}

impl InfluentialObservation {
    /// Create a new influential observation record
    pub fn new(index: usize, cooks_distance: f64, leverage: f64, std_residual: f64) -> Self {
        Self {
            index,
            cooks_distance,
            leverage,
            std_residual,
            studentized_residual: None,
            dffits: None,
        }
    }

    /// Check if this is a high-influence point by Cook's distance
    pub fn is_high_cooks(&self, n: usize) -> bool {
        self.cooks_distance > 4.0 / n as f64
    }

    /// Check if this is a high-leverage point
    pub fn is_high_leverage(&self, n_features: usize, n_samples: usize) -> bool {
        self.leverage > 2.0 * (n_features as f64 + 1.0) / n_samples as f64
    }
}

/// Prediction interval for probabilistic models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInterval {
    /// Point predictions
    pub predictions: Array1<f64>,
    /// Lower bound of prediction interval
    pub lower: Array1<f64>,
    /// Upper bound of prediction interval
    pub upper: Array1<f64>,
    /// Confidence level (e.g., 0.95)
    pub confidence_level: f64,
    /// Standard errors for each prediction (if available)
    pub std_errors: Option<Array1<f64>>,
}

impl PredictionInterval {
    /// Create a new prediction interval
    pub fn new(
        predictions: Array1<f64>,
        lower: Array1<f64>,
        upper: Array1<f64>,
        confidence_level: f64,
    ) -> Self {
        Self {
            predictions,
            lower,
            upper,
            confidence_level,
            std_errors: None,
        }
    }

    /// Add standard errors
    pub fn with_std_errors(mut self, std_errors: Array1<f64>) -> Self {
        self.std_errors = Some(std_errors);
        self
    }

    /// Get the width of prediction intervals
    pub fn interval_widths(&self) -> Array1<f64> {
        &self.upper - &self.lower
    }

    /// Check if actual values fall within the prediction intervals
    pub fn coverage(&self, y_actual: &Array1<f64>) -> f64 {
        let n = y_actual.len();
        if n == 0 {
            return 0.0;
        }

        let covered = y_actual
            .iter()
            .zip(self.lower.iter())
            .zip(self.upper.iter())
            .filter(|((y, lower), upper)| *y >= *lower && *y <= *upper)
            .count();

        covered as f64 / n as f64
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate percentile from sorted array
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = p * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    let frac = idx - lower as f64;

    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower].mul_add(1.0 - frac, sorted[upper] * frac)
    }
}

/// Truncate string for display
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Get a feature name by index, falling back to "x{idx+1}" if no names are provided.
///
/// Shared helper for models that report coefficient/feature names in summaries.
pub fn get_feature_name(feature_names: &Option<Vec<String>>, idx: usize) -> String {
    if let Some(ref names) = feature_names {
        if idx < names.len() {
            return names[idx].clone();
        }
    }
    format!("x{}", idx + 1)
}

/// Numerically stable sigmoid: 1 / (1 + exp(-x)).
///
/// Uses two-branch formulation to avoid overflow for large negative inputs.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Convert raw boosting predictions to probabilities.
///
/// For binary classification (n_classes == 2): applies sigmoid to column 0.
/// For multiclass: applies softmax row-wise with numerical stability.
pub fn raw_to_proba(raw: &Array2<f64>, n_classes: usize) -> Array2<f64> {
    let n_samples = raw.nrows();
    let mut probas = Array2::zeros((n_samples, n_classes));

    if n_classes == 2 {
        for i in 0..n_samples {
            let p = sigmoid(raw[[i, 0]]);
            probas[[i, 0]] = 1.0 - p;
            probas[[i, 1]] = p;
        }
    } else {
        for i in 0..n_samples {
            let row = raw.row(i);
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
            for j in 0..n_classes {
                probas[[i, j]] = (raw[[i, j]] - max_val).exp() / exp_sum;
            }
        }
    }

    probas
}

/// Compute log loss (cross-entropy) given true labels, predicted probabilities, and class labels.
pub fn compute_log_loss(y: &Array1<f64>, probas: &Array2<f64>, classes: &Array1<f64>) -> f64 {
    let n = y.len() as f64;
    let mut loss = 0.0;
    for (i, &yi) in y.iter().enumerate() {
        if let Some(class_idx) = classes.iter().position(|&c| (c - yi).abs() < 1e-10) {
            let p = probas[[i, class_idx]].max(1e-15).min(1.0 - 1e-15);
            loss -= p.ln();
        }
    }
    loss / n
}

/// Compute the median of a pre-sorted slice.
///
/// Returns 0.0 if the slice is empty. The caller must ensure the slice is sorted.
pub fn sorted_median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        0.0
    } else if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Validate that the model is fitted before operations that require it
pub fn check_is_fitted<T>(fitted_data: &Option<T>, operation: &str) -> Result<()> {
    if fitted_data.is_none() {
        return Err(FerroError::not_fitted(operation));
    }
    Ok(())
}

/// Validate input shapes for fit
pub fn validate_fit_input(x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(FerroError::shape_mismatch(
            format!("X has {} rows", x.nrows()),
            format!("y has {} elements", y.len()),
        ));
    }
    if x.is_empty() || y.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input(
            "X contains NaN or infinite values",
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input(
            "y contains NaN or infinite values",
        ));
    }
    Ok(())
}

/// Validate input shapes for fit, allowing NaN in X (for models that support missing values).
/// Still rejects infinity and NaN in y.
pub fn validate_fit_input_allow_nan(x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(FerroError::shape_mismatch(
            format!("X has {} rows", x.nrows()),
            format!("y has {} elements", y.len()),
        ));
    }
    if x.is_empty() || y.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    // Allow NaN (missing values) but reject infinity
    if x.iter().any(|v| v.is_infinite()) {
        return Err(FerroError::invalid_input("X contains infinite values"));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input(
            "y contains NaN or infinite values",
        ));
    }
    Ok(())
}

/// Validate input shapes for predict, allowing NaN in X (for models that support missing values).
pub fn validate_predict_input_allow_nan(x: &Array2<f64>, expected_features: usize) -> Result<()> {
    if x.ncols() != expected_features {
        return Err(FerroError::shape_mismatch(
            format!("{} features", expected_features),
            format!("{} features", x.ncols()),
        ));
    }
    if x.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if x.iter().any(|v| v.is_infinite()) {
        return Err(FerroError::invalid_input("X contains infinite values"));
    }
    Ok(())
}

/// Validate input shapes for predict
pub fn validate_predict_input(x: &Array2<f64>, expected_features: usize) -> Result<()> {
    if x.ncols() != expected_features {
        return Err(FerroError::shape_mismatch(
            format!("{} features", expected_features),
            format!("{} features", x.ncols()),
        ));
    }
    if x.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input(
            "X contains NaN or infinite values",
        ));
    }
    Ok(())
}

// =============================================================================
// Class Weight Utilities
// =============================================================================

/// Compute per-sample weights from class weights and target labels.
///
/// # Arguments
/// * `y` - Target labels (class assignments)
/// * `classes` - Unique class labels
/// * `class_weight` - Class weight specification
///
/// # Returns
/// Array of sample weights, one per sample
pub fn compute_sample_weights(
    y: &Array1<f64>,
    classes: &Array1<f64>,
    class_weight: &ClassWeight,
) -> Array1<f64> {
    let class_weights = compute_class_weight_map(y, classes, class_weight);
    y.iter()
        .map(|&label| {
            class_weights
                .iter()
                .find(|(c, _)| (*c - label).abs() < 1e-10)
                .map(|(_, w)| *w)
                .unwrap_or(1.0)
        })
        .collect()
}

/// Compute class weight mapping from class weight specification.
///
/// # Arguments
/// * `y` - Target labels (for computing balanced weights)
/// * `classes` - Unique class labels
/// * `class_weight` - Class weight specification
///
/// # Returns
/// Vector of (class_label, weight) pairs
pub fn compute_class_weight_map(
    y: &Array1<f64>,
    classes: &Array1<f64>,
    class_weight: &ClassWeight,
) -> Vec<(f64, f64)> {
    match class_weight {
        ClassWeight::Uniform => classes.iter().map(|&c| (c, 1.0)).collect(),
        ClassWeight::Balanced => {
            // Weight inversely proportional to class frequency:
            // weight_k = n_samples / (n_classes * n_samples_k)
            let n_samples = y.len() as f64;
            let n_classes = classes.len() as f64;
            classes
                .iter()
                .map(|&c| {
                    let count = y.iter().filter(|&&v| (v - c).abs() < 1e-10).count() as f64;
                    let weight = if count > 0.0 {
                        n_samples / (n_classes * count)
                    } else {
                        1.0
                    };
                    (c, weight)
                })
                .collect()
        }
        ClassWeight::Custom(weights) => weights.clone(),
    }
}

/// Get unique sorted classes from target array.
pub fn get_unique_classes(y: &Array1<f64>) -> Array1<f64> {
    let mut classes: Vec<f64> = y.iter().copied().collect();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    Array1::from_vec(classes)
}

/// Check model output for NaN/Inf values. Returns error if any found.
///
/// Call this after `predict()` in models susceptible to numerical issues
/// (e.g., LogisticRegression, GaussianNB, GMM).
pub fn validate_output(predictions: &Array1<f64>, model_name: &str) -> Result<()> {
    if let Some(pos) = predictions.iter().position(|v| !v.is_finite()) {
        return Err(FerroError::numerical(format!(
            "{} produced non-finite output at index {} (value: {}). \
             This indicates a numerical issue in the model.",
            model_name, pos, predictions[pos]
        )));
    }
    Ok(())
}

/// Check 2D model output (e.g., predict_proba) for NaN/Inf values.
pub fn validate_output_2d(predictions: &Array2<f64>, model_name: &str) -> Result<()> {
    for (idx, val) in predictions.iter().enumerate() {
        if !val.is_finite() {
            let row = idx / predictions.ncols();
            let col = idx % predictions.ncols();
            return Err(FerroError::numerical(format!(
                "{} produced non-finite output at [{}, {}] (value: {}). \
                 This indicates a numerical issue in the model.",
                model_name, row, col, *val
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coefficient_info() {
        let coef = CoefficientInfo::new("x1", 2.5, 0.5)
            .with_test(5.0, 0.001)
            .with_ci(1.5, 3.5, 0.95);

        assert_eq!(coef.name, "x1");
        assert!((coef.estimate - 2.5).abs() < 1e-10);
        assert!(coef.is_significant(0.05));
        assert!(coef.ci_excludes_zero());
    }

    #[test]
    fn test_fit_statistics() {
        let stats = FitStatistics::with_r_squared(0.85, 0.83).with_f_test(42.5, 0.0001);

        assert!(stats.r_squared.unwrap() > 0.8);
        assert!(stats.is_model_significant(0.05));
    }

    #[test]
    fn test_model_summary_display() {
        let mut summary = ModelSummary::new("OLS Linear Regression", 100, 3);
        summary.add_coefficient(
            CoefficientInfo::new("const", 1.5, 0.3)
                .with_test(5.0, 0.0001)
                .with_ci(0.9, 2.1, 0.95),
        );
        summary.add_coefficient(
            CoefficientInfo::new("x1", 2.0, 0.4)
                .with_test(5.0, 0.0001)
                .with_ci(1.2, 2.8, 0.95),
        );
        summary.fit_statistics = FitStatistics::with_r_squared(0.85, 0.83);

        let output = format!("{}", summary);
        assert!(output.contains("OLS Linear Regression"));
        assert!(output.contains("R-squared"));
        assert!(output.contains("const"));
        assert!(output.contains("x1"));
    }

    #[test]
    fn test_residual_statistics() {
        let residuals = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let stats = ResidualStatistics::from_residuals(&residuals);

        assert!((stats.min - (-2.0)).abs() < 1e-10);
        assert!((stats.max - 2.0).abs() < 1e-10);
        assert!((stats.median - 0.0).abs() < 1e-10);
        assert!((stats.mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_assumption_test_result() {
        let test = AssumptionTestResult::new(
            Assumption::NormalResiduals,
            "Shapiro-Wilk",
            0.98,
            0.15,
            0.05,
        );

        assert!(test.passed); // p > alpha means assumption holds
        assert_eq!(test.assumption, Assumption::NormalResiduals);
    }

    #[test]
    fn test_diagnostics() {
        let residuals = Array1::from_vec(vec![-1.0, 0.0, 1.0, 0.5, -0.5]);
        let mut diagnostics = Diagnostics::new(ResidualStatistics::from_residuals(&residuals));

        diagnostics.add_assumption_test(AssumptionTestResult::new(
            Assumption::NormalResiduals,
            "Shapiro-Wilk",
            0.98,
            0.25,
            0.05,
        ));

        assert!(diagnostics.normality_ok());
        assert!(diagnostics.all_assumptions_ok());
    }

    #[test]
    fn test_prediction_interval() {
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let lower = Array1::from_vec(vec![0.5, 1.5, 2.5]);
        let upper = Array1::from_vec(vec![1.5, 2.5, 3.5]);

        let interval = PredictionInterval::new(predictions, lower, upper, 0.95);

        let widths = interval.interval_widths();
        assert!((widths[0] - 1.0).abs() < 1e-10);

        // All predictions within interval
        let actual = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((interval.coverage(&actual) - 1.0).abs() < 1e-10);

        // One outside interval
        let actual2 = Array1::from_vec(vec![0.0, 2.0, 3.0]);
        assert!((interval.coverage(&actual2) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_influential_observation() {
        let obs = InfluentialObservation::new(5, 0.25, 0.15, 2.5);

        // 4/20 = 0.2, and 0.25 > 0.2 so this is high Cook's
        assert!(obs.is_high_cooks(20));
        // 2*(3+1)/100 = 0.08, and 0.15 > 0.08 so this is high leverage
        assert!(obs.is_high_leverage(3, 100));

        // Not high influence with larger sample
        let obs_low = InfluentialObservation::new(5, 0.01, 0.02, 1.5);
        assert!(!obs_low.is_high_cooks(100)); // threshold = 0.04
        assert!(!obs_low.is_high_leverage(3, 100)); // threshold = 0.08
    }

    #[test]
    fn test_validate_fit_input() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        assert!(validate_fit_input(&x, &y).is_ok());

        let y_wrong = Array1::from_vec(vec![1.0, 2.0]);
        assert!(validate_fit_input(&x, &y_wrong).is_err());
    }

    #[test]
    fn test_validate_predict_input() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert!(validate_predict_input(&x, 2).is_ok());
        assert!(validate_predict_input(&x, 3).is_err());
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 0.5) - 3.0).abs() < 1e-10);
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&sorted, 1.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate_output_all_finite() {
        let preds = Array1::from_vec(vec![1.0, 2.0, 3.0, -4.5, 0.0]);
        assert!(validate_output(&preds, "TestModel").is_ok());
    }

    #[test]
    fn test_validate_output_detects_nan() {
        let preds = Array1::from_vec(vec![1.0, 2.0, f64::NAN, 4.0]);
        let err = validate_output(&preds, "TestModel").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("TestModel"));
        assert!(msg.contains("index 2"));
        assert!(msg.contains("non-finite"));
    }

    #[test]
    fn test_validate_output_detects_inf() {
        let preds = Array1::from_vec(vec![1.0, f64::INFINITY, 3.0]);
        let err = validate_output(&preds, "TestModel").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("index 1"));
    }

    #[test]
    fn test_validate_output_detects_neg_inf() {
        let preds = Array1::from_vec(vec![f64::NEG_INFINITY, 2.0, 3.0]);
        let err = validate_output(&preds, "TestModel").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("index 0"));
    }

    #[test]
    fn test_validate_output_2d_all_finite() {
        let preds = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.7, 0.3, 0.3, 0.4]).unwrap();
        assert!(validate_output_2d(&preds, "TestModel").is_ok());
    }

    #[test]
    fn test_validate_output_2d_detects_nan() {
        let preds = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, f64::NAN, 0.5]).unwrap();
        let err = validate_output_2d(&preds, "TestProba").unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("TestProba"));
        assert!(msg.contains("[1, 0]"));
    }
}
