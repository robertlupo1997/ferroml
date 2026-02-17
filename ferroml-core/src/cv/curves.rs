//! Learning Curve and Validation Curve Data Generation
//!
//! This module provides functions to generate data for diagnosing model behavior:
//!
//! - [`learning_curve`]: Performance as a function of training set size
//! - [`validation_curve`]: Performance as a function of a hyperparameter value
//!
//! Both functions return data with proper statistical handling including
//! confidence intervals for the scores.
//!
//! # Example
//!
//! ```
//! # fn main() -> ferroml_core::Result<()> {
//! # use ferroml_core::traits::{Estimator, Predictor, PredictionWithUncertainty};
//! # use ferroml_core::hpo::SearchSpace;
//! # use ferroml_core::metrics::{Metric, MetricValue, Direction};
//! # use ndarray::{Array1, Array2};
//! # #[derive(Clone)]
//! # struct MyModel;
//! # struct MyFitted(f64);
//! # impl Predictor for MyFitted {
//! #     fn predict(&self, x: &Array2<f64>) -> ferroml_core::Result<Array1<f64>> { Ok(Array1::from_elem(x.nrows(), self.0)) }
//! #     fn predict_with_uncertainty(&self, x: &Array2<f64>, c: f64) -> ferroml_core::Result<PredictionWithUncertainty> { let p = self.predict(x)?; let n = p.len(); Ok(PredictionWithUncertainty { predictions: p.clone(), lower: p.clone(), upper: p, confidence_level: c, std_errors: None }) }
//! # }
//! # impl Estimator for MyModel { type Fitted = MyFitted; fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> ferroml_core::Result<MyFitted> { Ok(MyFitted(y.mean().unwrap_or(0.0))) } fn search_space(&self) -> SearchSpace { SearchSpace::new() } }
//! # struct AccuracyMetric;
//! # impl Metric for AccuracyMetric { fn name(&self) -> &str { "acc" } fn direction(&self) -> Direction { Direction::Maximize } fn compute(&self, a: &Array1<f64>, b: &Array1<f64>) -> ferroml_core::Result<MetricValue> { Ok(MetricValue::new("acc", 0.9, Direction::Maximize)) } }
//! # let x = Array2::from_shape_vec((30, 2), (0..60).map(|i| i as f64).collect()).unwrap();
//! # let y = Array1::from_vec((0..30).map(|i| (i % 2) as f64).collect());
//! # let model = MyModel;
//! use ferroml_core::cv::{learning_curve, KFold, LearningCurveConfig};
//!
//! let cv = KFold::new(5);
//! let train_sizes = vec![0.1, 0.25, 0.5, 0.75, 1.0];
//!
//! let result = learning_curve(
//!     &model,
//!     &x,
//!     &y,
//!     &cv,
//!     &AccuracyMetric,
//!     &train_sizes,
//!     LearningCurveConfig::default(),
//! )?;
//!
//! // Plot train_scores_summary and test_scores_summary to diagnose bias/variance
//! for (size, train_summary, test_summary) in result.iter_summaries() {
//!     println!("n={}: train={:.3}±{:.3}, test={:.3}±{:.3}",
//!              size, train_summary.mean, train_summary.std,
//!              test_summary.mean, test_summary.std);
//! }
//! # Ok(())
//! # }
//! ```

use crate::metrics::Metric;
use crate::traits::{Estimator, Predictor};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{compute_t_confidence_interval, select_elements, select_rows, CVFold, CrossValidator};

// ============================================================================
// Score Summary
// ============================================================================

/// Summary statistics for scores across CV folds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreSummary {
    /// Mean score across folds
    pub mean: f64,
    /// Standard deviation of scores
    pub std: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Confidence level used
    pub confidence_level: f64,
    /// Number of folds
    pub n_folds: usize,
}

impl ScoreSummary {
    /// Create summary from a vector of scores
    pub fn from_scores(scores: &[f64], confidence_level: f64) -> Self {
        let n_folds = scores.len();

        if n_folds == 0 {
            return Self {
                mean: f64::NAN,
                std: f64::NAN,
                ci_lower: f64::NAN,
                ci_upper: f64::NAN,
                confidence_level,
                n_folds: 0,
            };
        }

        let mean = scores.iter().sum::<f64>() / n_folds as f64;

        let std = if n_folds > 1 {
            let variance =
                scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n_folds - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let (ci_lower, ci_upper) =
            compute_t_confidence_interval(mean, std, n_folds, confidence_level);

        Self {
            mean,
            std,
            ci_lower,
            ci_upper,
            confidence_level,
            n_folds,
        }
    }

    /// Get a formatted summary string
    pub fn summary(&self) -> String {
        format!(
            "{:.4} ± {:.4} ({}% CI: [{:.4}, {:.4}])",
            self.mean,
            self.std,
            (self.confidence_level * 100.0) as i32,
            self.ci_lower,
            self.ci_upper
        )
    }
}

// ============================================================================
// Learning Curve
// ============================================================================

/// Configuration for learning curve computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurveConfig {
    /// Confidence level for intervals (default 0.95)
    pub confidence_level: f64,
    /// Number of parallel jobs (-1 for all CPUs)
    pub n_jobs: i32,
    /// Whether to shuffle data before generating subsets
    pub shuffle: bool,
    /// Random seed for shuffling
    pub random_seed: Option<u64>,
}

impl Default for LearningCurveConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            n_jobs: 1,
            shuffle: false,
            random_seed: None,
        }
    }
}

impl LearningCurveConfig {
    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence_level = confidence;
        self
    }

    /// Set number of parallel jobs
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Enable shuffling with a seed
    pub fn with_shuffle(mut self, shuffle: bool, seed: Option<u64>) -> Self {
        self.shuffle = shuffle;
        self.random_seed = seed;
        self
    }
}

/// Result from learning curve analysis
///
/// Contains train and test scores at different training set sizes,
/// allowing diagnosis of bias/variance tradeoff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurveResult {
    /// Absolute training set sizes used (number of samples)
    pub train_sizes: Vec<usize>,
    /// Train scores matrix: shape (n_sizes, n_folds)
    pub train_scores: Array2<f64>,
    /// Test scores matrix: shape (n_sizes, n_folds)
    pub test_scores: Array2<f64>,
    /// Summary statistics for train scores at each size
    pub train_scores_summary: Vec<ScoreSummary>,
    /// Summary statistics for test scores at each size
    pub test_scores_summary: Vec<ScoreSummary>,
    /// Total number of samples in the dataset
    pub n_samples: usize,
}

impl LearningCurveResult {
    /// Iterate over (train_size, train_summary, test_summary) tuples
    pub fn iter_summaries(
        &self,
    ) -> impl Iterator<Item = (usize, &ScoreSummary, &ScoreSummary)> + '_ {
        self.train_sizes
            .iter()
            .copied()
            .zip(self.train_scores_summary.iter())
            .zip(self.test_scores_summary.iter())
            .map(|((size, train), test)| (size, train, test))
    }

    /// Get summary for visualization/printing
    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "Learning Curve Results (n_samples={}, {} sizes)",
            self.n_samples,
            self.train_sizes.len()
        )];

        for (size, train, test) in self.iter_summaries() {
            lines.push(format!(
                "  n={:>6}: train={:.4}±{:.4}, test={:.4}±{:.4}",
                size, train.mean, train.std, test.mean, test.std
            ));
        }

        lines.join("\n")
    }
}

/// Generate data for a learning curve (performance vs training set size).
///
/// This function evaluates model performance at different training set sizes,
/// helping diagnose bias/variance tradeoff:
///
/// - **High bias (underfitting)**: Both train and test scores are low and converge
/// - **High variance (overfitting)**: Large gap between train and test scores
///
/// # Arguments
///
/// * `estimator` - The estimator to evaluate
/// * `x` - Feature matrix of shape (n_samples, n_features)
/// * `y` - Target vector of length n_samples
/// * `cv` - Cross-validation strategy
/// * `metric` - Scoring metric to use
/// * `train_sizes` - Training set sizes as fractions (0.0-1.0) or absolute counts (>1)
/// * `config` - Learning curve configuration
///
/// # Returns
///
/// `LearningCurveResult` containing scores at each training size with CIs.
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// # use ferroml_core::traits::{Estimator, Predictor, PredictionWithUncertainty};
/// # use ferroml_core::hpo::SearchSpace;
/// # use ferroml_core::metrics::{Metric, MetricValue, Direction};
/// # use ndarray::{Array1, Array2};
/// # #[derive(Clone)]
/// # struct MyModel;
/// # struct MyFitted(f64);
/// # impl Predictor for MyFitted {
/// #     fn predict(&self, x: &Array2<f64>) -> ferroml_core::Result<Array1<f64>> { Ok(Array1::from_elem(x.nrows(), self.0)) }
/// #     fn predict_with_uncertainty(&self, x: &Array2<f64>, c: f64) -> ferroml_core::Result<PredictionWithUncertainty> { let p = self.predict(x)?; Ok(PredictionWithUncertainty { predictions: p.clone(), lower: p.clone(), upper: p, confidence_level: c, std_errors: None }) }
/// # }
/// # impl Estimator for MyModel { type Fitted = MyFitted; fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> ferroml_core::Result<MyFitted> { Ok(MyFitted(y.mean().unwrap_or(0.0))) } fn search_space(&self) -> SearchSpace { SearchSpace::new() } }
/// # struct AccuracyMetric;
/// # impl Metric for AccuracyMetric { fn name(&self) -> &str { "acc" } fn direction(&self) -> Direction { Direction::Maximize } fn compute(&self, _a: &Array1<f64>, _b: &Array1<f64>) -> ferroml_core::Result<MetricValue> { Ok(MetricValue::new("acc", 0.9, Direction::Maximize)) } }
/// # let x = Array2::from_shape_vec((30, 2), (0..60).map(|i| i as f64).collect()).unwrap();
/// # let y = Array1::from_vec((0..30).map(|i| (i % 2) as f64).collect());
/// # let model = MyModel;
/// use ferroml_core::cv::{learning_curve, KFold, LearningCurveConfig};
///
/// let train_sizes = vec![0.1, 0.33, 0.55, 0.78, 1.0];
/// let result = learning_curve(
///     &model, &x, &y, &KFold::new(5), &AccuracyMetric,
///     &train_sizes, LearningCurveConfig::default()
/// )?;
///
/// // Diagnose: if test scores are much lower than train, model is overfitting
/// # Ok(())
/// # }
/// ```
pub fn learning_curve<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    metric: &M,
    train_sizes: &[f64],
    config: LearningCurveConfig,
) -> Result<LearningCurveResult>
where
    E: Estimator + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,
{
    let n_samples = x.nrows();

    // Validate inputs
    if y.len() != n_samples {
        return Err(FerroError::shape_mismatch(
            format!("x has {} samples", n_samples),
            format!("y has {} samples", y.len()),
        ));
    }

    if train_sizes.is_empty() {
        return Err(FerroError::invalid_input("train_sizes cannot be empty"));
    }

    // Get CV splits first to determine training set sizes
    let folds = cv.split(n_samples, Some(y), None)?;
    let n_folds = folds.len();

    if n_folds == 0 {
        return Err(FerroError::invalid_input(
            "Cross-validator produced no folds",
        ));
    }

    // The minimum train set size in any fold
    let min_train_size = folds
        .iter()
        .map(|f| f.train_indices.len())
        .min()
        .unwrap_or(0);

    // Convert train_sizes to absolute numbers (as fractions of training set, not total data)
    let absolute_sizes = convert_train_sizes(train_sizes, min_train_size)?;

    // Validate that all train sizes are feasible
    for &size in &absolute_sizes {
        if size > min_train_size {
            return Err(FerroError::invalid_input(format!(
                "Train size {} exceeds minimum available training samples {} in CV folds",
                size, min_train_size
            )));
        }
    }

    let n_sizes = absolute_sizes.len();
    let use_parallel = config.n_jobs != 1;

    // Results storage
    let mut train_scores = Array2::zeros((n_sizes, n_folds));
    let mut test_scores = Array2::zeros((n_sizes, n_folds));

    // For each training size
    for (size_idx, &train_size) in absolute_sizes.iter().enumerate() {
        // Evaluate each fold
        let fold_results: Vec<Result<(f64, f64)>> = if use_parallel {
            folds
                .par_iter()
                .map(|fold| {
                    evaluate_fold_with_subset(
                        estimator,
                        x,
                        y,
                        fold,
                        metric,
                        train_size,
                        config.shuffle,
                        config.random_seed,
                    )
                })
                .collect()
        } else {
            folds
                .iter()
                .map(|fold| {
                    evaluate_fold_with_subset(
                        estimator,
                        x,
                        y,
                        fold,
                        metric,
                        train_size,
                        config.shuffle,
                        config.random_seed,
                    )
                })
                .collect()
        };

        // Collect results
        for (fold_idx, result) in fold_results.into_iter().enumerate() {
            let (train_score, test_score) = result?;
            train_scores[[size_idx, fold_idx]] = train_score;
            test_scores[[size_idx, fold_idx]] = test_score;
        }
    }

    // Compute summaries for each size
    let train_scores_summary: Vec<ScoreSummary> = (0..n_sizes)
        .map(|i| {
            let scores: Vec<f64> = train_scores.row(i).iter().copied().collect();
            ScoreSummary::from_scores(&scores, config.confidence_level)
        })
        .collect();

    let test_scores_summary: Vec<ScoreSummary> = (0..n_sizes)
        .map(|i| {
            let scores: Vec<f64> = test_scores.row(i).iter().copied().collect();
            ScoreSummary::from_scores(&scores, config.confidence_level)
        })
        .collect();

    Ok(LearningCurveResult {
        train_sizes: absolute_sizes,
        train_scores,
        test_scores,
        train_scores_summary,
        test_scores_summary,
        n_samples,
    })
}

/// Evaluate a single fold with a subset of training data
fn evaluate_fold_with_subset<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    fold: &CVFold,
    metric: &M,
    train_size: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<(f64, f64)>
where
    E: Estimator + Clone,
    M: Metric,
{
    // Get training indices, potentially shuffled
    let mut train_indices = fold.train_indices.clone();

    if shuffle {
        let actual_seed = seed.unwrap_or(42) + fold.fold_index as u64;
        super::shuffle_indices(&mut train_indices, actual_seed);
    }

    // Take first train_size samples
    let subset_indices: Vec<usize> = train_indices.into_iter().take(train_size).collect();

    // Extract subsets
    let x_train = select_rows(x, &subset_indices);
    let y_train = select_elements(y, &subset_indices);
    let x_test = select_rows(x, &fold.test_indices);
    let y_test = select_elements(y, &fold.test_indices);

    // Fit model
    let fitted = estimator.clone().fit(&x_train, &y_train)?;

    // Score on train subset
    let y_pred_train = fitted.predict(&x_train)?;
    let train_metric = metric.compute(&y_train, &y_pred_train)?;
    let train_score = train_metric.value;

    // Score on test set
    let y_pred_test = fitted.predict(&x_test)?;
    let test_metric = metric.compute(&y_test, &y_pred_test)?;
    let test_score = test_metric.value;

    Ok((train_score, test_score))
}

/// Convert train sizes (fractions or absolute) to absolute counts
fn convert_train_sizes(train_sizes: &[f64], n_samples: usize) -> Result<Vec<usize>> {
    let mut absolute_sizes = Vec::with_capacity(train_sizes.len());

    for &size in train_sizes {
        let abs_size = if size <= 1.0 {
            // Fraction of samples
            if size <= 0.0 {
                return Err(FerroError::invalid_input(
                    "Train size fractions must be > 0",
                ));
            }
            ((size * n_samples as f64).round() as usize).max(1)
        } else {
            // Absolute count
            size as usize
        };

        if abs_size == 0 {
            return Err(FerroError::invalid_input(
                "Train size must result in at least 1 sample",
            ));
        }

        absolute_sizes.push(abs_size);
    }

    Ok(absolute_sizes)
}

// ============================================================================
// Validation Curve
// ============================================================================

/// Configuration for validation curve computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCurveConfig {
    /// Confidence level for intervals (default 0.95)
    pub confidence_level: f64,
    /// Number of parallel jobs (-1 for all CPUs)
    pub n_jobs: i32,
}

impl Default for ValidationCurveConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            n_jobs: 1,
        }
    }
}

impl ValidationCurveConfig {
    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence_level = confidence;
        self
    }

    /// Set number of parallel jobs
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

/// Result from validation curve analysis
///
/// Contains train and test scores at different hyperparameter values,
/// allowing visualization of overfitting/underfitting as a function
/// of model complexity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCurveResult {
    /// Parameter name that was varied
    pub param_name: String,
    /// Parameter values tested
    pub param_values: Vec<f64>,
    /// Train scores matrix: shape (n_values, n_folds)
    pub train_scores: Array2<f64>,
    /// Test scores matrix: shape (n_values, n_folds)
    pub test_scores: Array2<f64>,
    /// Summary statistics for train scores at each value
    pub train_scores_summary: Vec<ScoreSummary>,
    /// Summary statistics for test scores at each value
    pub test_scores_summary: Vec<ScoreSummary>,
}

impl ValidationCurveResult {
    /// Iterate over (param_value, train_summary, test_summary) tuples
    pub fn iter_summaries(&self) -> impl Iterator<Item = (f64, &ScoreSummary, &ScoreSummary)> + '_ {
        self.param_values
            .iter()
            .copied()
            .zip(self.train_scores_summary.iter())
            .zip(self.test_scores_summary.iter())
            .map(|((value, train), test)| (value, train, test))
    }

    /// Get summary for visualization/printing
    pub fn summary(&self) -> String {
        let mut lines = vec![format!(
            "Validation Curve Results (param={}, {} values)",
            self.param_name,
            self.param_values.len()
        )];

        for (value, train, test) in self.iter_summaries() {
            lines.push(format!(
                "  {}={:>8.4}: train={:.4}±{:.4}, test={:.4}±{:.4}",
                self.param_name, value, train.mean, train.std, test.mean, test.std
            ));
        }

        lines.join("\n")
    }

    /// Find the parameter value with best test score
    ///
    /// # Arguments
    ///
    /// * `maximize` - If true, returns value with highest test score; if false, lowest
    pub fn best_param_value(&self, maximize: bool) -> Option<f64> {
        if self.param_values.is_empty() {
            return None;
        }

        let (best_idx, _) =
            self.test_scores_summary
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    if maximize {
                        a.mean
                            .partial_cmp(&b.mean)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        b.mean
                            .partial_cmp(&a.mean)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }
                })?;

        Some(self.param_values[best_idx])
    }
}

/// A trait for estimators that support setting a parameter by name
///
/// This is required for validation_curve to work with different parameter values.
pub trait ParameterSettable: Estimator {
    /// Set a hyperparameter by name
    fn set_param(&mut self, name: &str, value: f64) -> Result<()>;

    /// Get a hyperparameter value by name
    fn get_param(&self, name: &str) -> Result<f64>;
}

/// Generate data for a validation curve (performance vs hyperparameter value).
///
/// This function evaluates model performance at different hyperparameter values,
/// helping visualize overfitting/underfitting:
///
/// - **Underfitting region**: Both train and test scores are low
/// - **Overfitting region**: High train score but low test score
/// - **Optimal region**: Best test score with reasonable train/test gap
///
/// # Arguments
///
/// * `estimator` - The estimator to evaluate (must implement `ParameterSettable`)
/// * `x` - Feature matrix of shape (n_samples, n_features)
/// * `y` - Target vector of length n_samples
/// * `cv` - Cross-validation strategy
/// * `metric` - Scoring metric to use
/// * `param_name` - Name of the hyperparameter to vary
/// * `param_values` - Values to test for the hyperparameter
/// * `config` - Validation curve configuration
///
/// # Returns
///
/// `ValidationCurveResult` containing scores at each parameter value with CIs.
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// # use ferroml_core::traits::{Estimator, Predictor, PredictionWithUncertainty};
/// # use ferroml_core::hpo::SearchSpace;
/// # use ferroml_core::metrics::{Metric, MetricValue, Direction};
/// # use ferroml_core::cv::ParameterSettable;
/// # use ndarray::{Array1, Array2};
/// # #[derive(Clone)]
/// # struct MyModel { regularization: f64 }
/// # struct MyFitted(f64);
/// # impl Predictor for MyFitted {
/// #     fn predict(&self, x: &Array2<f64>) -> ferroml_core::Result<Array1<f64>> { Ok(Array1::from_elem(x.nrows(), self.0)) }
/// #     fn predict_with_uncertainty(&self, x: &Array2<f64>, c: f64) -> ferroml_core::Result<PredictionWithUncertainty> { let p = self.predict(x)?; Ok(PredictionWithUncertainty { predictions: p.clone(), lower: p.clone(), upper: p, confidence_level: c, std_errors: None }) }
/// # }
/// # impl Estimator for MyModel { type Fitted = MyFitted; fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> ferroml_core::Result<MyFitted> { Ok(MyFitted(y.mean().unwrap_or(0.0))) } fn search_space(&self) -> SearchSpace { SearchSpace::new() } }
/// # impl ParameterSettable for MyModel { fn set_param(&mut self, _n: &str, v: f64) -> ferroml_core::Result<()> { self.regularization = v; Ok(()) } fn get_param(&self, _n: &str) -> ferroml_core::Result<f64> { Ok(self.regularization) } }
/// # struct AccuracyMetric;
/// # impl Metric for AccuracyMetric { fn name(&self) -> &str { "acc" } fn direction(&self) -> Direction { Direction::Maximize } fn compute(&self, _a: &Array1<f64>, _b: &Array1<f64>) -> ferroml_core::Result<MetricValue> { Ok(MetricValue::new("acc", 0.9, Direction::Maximize)) } }
/// # let x = Array2::from_shape_vec((30, 2), (0..60).map(|i| i as f64).collect()).unwrap();
/// # let y = Array1::from_vec((0..30).map(|i| (i % 2) as f64).collect());
/// # let model = MyModel { regularization: 1.0 };
/// use ferroml_core::cv::{validation_curve, KFold, ValidationCurveConfig};
///
/// let param_values = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
/// let result = validation_curve(
///     &model, &x, &y, &KFold::new(5), &AccuracyMetric,
///     "regularization", &param_values, ValidationCurveConfig::default()
/// )?;
///
/// // Find optimal regularization
/// let best_reg = result.best_param_value(true);
/// # Ok(())
/// # }
/// ```
pub fn validation_curve<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    metric: &M,
    param_name: &str,
    param_values: &[f64],
    config: ValidationCurveConfig,
) -> Result<ValidationCurveResult>
where
    E: ParameterSettable + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,
{
    let n_samples = x.nrows();

    // Validate inputs
    if y.len() != n_samples {
        return Err(FerroError::shape_mismatch(
            format!("x has {} samples", n_samples),
            format!("y has {} samples", y.len()),
        ));
    }

    if param_values.is_empty() {
        return Err(FerroError::invalid_input("param_values cannot be empty"));
    }

    // Get CV splits
    let folds = cv.split(n_samples, Some(y), None)?;
    let n_folds = folds.len();

    if n_folds == 0 {
        return Err(FerroError::invalid_input(
            "Cross-validator produced no folds",
        ));
    }

    let n_values = param_values.len();
    let use_parallel = config.n_jobs != 1;

    // Results storage
    let mut train_scores = Array2::zeros((n_values, n_folds));
    let mut test_scores = Array2::zeros((n_values, n_folds));

    // For each parameter value
    for (value_idx, &param_value) in param_values.iter().enumerate() {
        // Create estimator with this parameter value
        let mut estimator_clone = estimator.clone();
        estimator_clone.set_param(param_name, param_value)?;

        // Evaluate each fold
        let fold_results: Vec<Result<(f64, f64)>> = if use_parallel {
            folds
                .par_iter()
                .map(|fold| evaluate_fold_full(&estimator_clone, x, y, fold, metric))
                .collect()
        } else {
            folds
                .iter()
                .map(|fold| evaluate_fold_full(&estimator_clone, x, y, fold, metric))
                .collect()
        };

        // Collect results
        for (fold_idx, result) in fold_results.into_iter().enumerate() {
            let (train_score, test_score) = result?;
            train_scores[[value_idx, fold_idx]] = train_score;
            test_scores[[value_idx, fold_idx]] = test_score;
        }
    }

    // Compute summaries for each parameter value
    let train_scores_summary: Vec<ScoreSummary> = (0..n_values)
        .map(|i| {
            let scores: Vec<f64> = train_scores.row(i).iter().copied().collect();
            ScoreSummary::from_scores(&scores, config.confidence_level)
        })
        .collect();

    let test_scores_summary: Vec<ScoreSummary> = (0..n_values)
        .map(|i| {
            let scores: Vec<f64> = test_scores.row(i).iter().copied().collect();
            ScoreSummary::from_scores(&scores, config.confidence_level)
        })
        .collect();

    Ok(ValidationCurveResult {
        param_name: param_name.to_string(),
        param_values: param_values.to_vec(),
        train_scores,
        test_scores,
        train_scores_summary,
        test_scores_summary,
    })
}

/// Evaluate a single fold with full training data
fn evaluate_fold_full<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    fold: &CVFold,
    metric: &M,
) -> Result<(f64, f64)>
where
    E: Estimator + Clone,
    M: Metric,
{
    // Extract train/test data
    let x_train = select_rows(x, &fold.train_indices);
    let y_train = select_elements(y, &fold.train_indices);
    let x_test = select_rows(x, &fold.test_indices);
    let y_test = select_elements(y, &fold.test_indices);

    // Fit model
    let fitted = estimator.clone().fit(&x_train, &y_train)?;

    // Score on train
    let y_pred_train = fitted.predict(&x_train)?;
    let train_metric = metric.compute(&y_train, &y_pred_train)?;
    let train_score = train_metric.value;

    // Score on test
    let y_pred_test = fitted.predict(&x_test)?;
    let test_metric = metric.compute(&y_test, &y_pred_test)?;
    let test_score = test_metric.value;

    Ok((train_score, test_score))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::KFold;
    use crate::hpo::SearchSpace;
    use crate::metrics::{Direction, MetricValue};

    /// Mock estimator that predicts mean of y
    #[derive(Clone)]
    struct MockMeanEstimator {
        regularization: f64,
    }

    impl MockMeanEstimator {
        fn new() -> Self {
            Self {
                regularization: 1.0,
            }
        }
    }

    struct MockMeanPredictor {
        mean: f64,
    }

    impl crate::traits::Predictor for MockMeanPredictor {
        fn predict(&self, x: &Array2<f64>) -> crate::Result<Array1<f64>> {
            Ok(Array1::from_vec(vec![self.mean; x.nrows()]))
        }

        fn predict_with_uncertainty(
            &self,
            x: &Array2<f64>,
            confidence: f64,
        ) -> crate::Result<crate::traits::PredictionWithUncertainty> {
            let predictions = Array1::from_vec(vec![self.mean; x.nrows()]);
            let lower = Array1::from_vec(vec![self.mean - 0.1; x.nrows()]);
            let upper = Array1::from_vec(vec![self.mean + 0.1; x.nrows()]);
            Ok(crate::traits::PredictionWithUncertainty {
                predictions,
                lower,
                upper,
                confidence_level: confidence,
                std_errors: None,
            })
        }
    }

    impl crate::traits::Estimator for MockMeanEstimator {
        type Fitted = MockMeanPredictor;

        fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> crate::Result<Self::Fitted> {
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            // Simulate regularization effect
            let adjusted_mean = mean * (1.0 / (1.0 + self.regularization));
            Ok(MockMeanPredictor {
                mean: adjusted_mean,
            })
        }

        fn search_space(&self) -> SearchSpace {
            SearchSpace::new()
        }
    }

    impl ParameterSettable for MockMeanEstimator {
        fn set_param(&mut self, name: &str, value: f64) -> crate::Result<()> {
            match name {
                "regularization" => {
                    self.regularization = value;
                    Ok(())
                }
                _ => Err(FerroError::invalid_input(format!(
                    "Unknown parameter: {}",
                    name
                ))),
            }
        }

        fn get_param(&self, name: &str) -> crate::Result<f64> {
            match name {
                "regularization" => Ok(self.regularization),
                _ => Err(FerroError::invalid_input(format!(
                    "Unknown parameter: {}",
                    name
                ))),
            }
        }
    }

    /// Mock MSE metric
    struct MockMSE;

    impl Metric for MockMSE {
        fn name(&self) -> &str {
            "mse"
        }

        fn direction(&self) -> Direction {
            Direction::Minimize
        }

        fn compute(
            &self,
            y_true: &Array1<f64>,
            y_pred: &Array1<f64>,
        ) -> crate::Result<MetricValue> {
            if y_true.len() != y_pred.len() {
                return Err(FerroError::shape_mismatch(
                    format!("y_true len {}", y_true.len()),
                    format!("y_pred len {}", y_pred.len()),
                ));
            }
            let mse = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f64>()
                / y_true.len() as f64;
            Ok(MetricValue::new(self.name(), mse, self.direction()))
        }
    }

    // ========================================================================
    // ScoreSummary tests
    // ========================================================================

    #[test]
    fn test_score_summary_basic() {
        let scores = vec![0.80, 0.82, 0.78, 0.81, 0.79];
        let summary = ScoreSummary::from_scores(&scores, 0.95);

        assert_eq!(summary.n_folds, 5);
        assert!((summary.mean - 0.80).abs() < 0.01);
        assert!(summary.std > 0.0);
        assert!(summary.ci_lower < summary.mean);
        assert!(summary.ci_upper > summary.mean);
    }

    #[test]
    fn test_score_summary_single() {
        let scores = vec![0.85];
        let summary = ScoreSummary::from_scores(&scores, 0.95);

        assert_eq!(summary.n_folds, 1);
        assert!((summary.mean - 0.85).abs() < 1e-10);
        assert_eq!(summary.std, 0.0);
    }

    #[test]
    fn test_score_summary_empty() {
        let scores: Vec<f64> = vec![];
        let summary = ScoreSummary::from_scores(&scores, 0.95);

        assert_eq!(summary.n_folds, 0);
        assert!(summary.mean.is_nan());
    }

    // ========================================================================
    // Learning curve tests
    // ========================================================================

    #[test]
    fn test_convert_train_sizes_fractions() {
        let sizes = vec![0.1, 0.5, 1.0];
        let absolute = convert_train_sizes(&sizes, 100).unwrap();

        assert_eq!(absolute, vec![10, 50, 100]);
    }

    #[test]
    fn test_convert_train_sizes_absolute() {
        let sizes = vec![10.0, 50.0, 100.0];
        let absolute = convert_train_sizes(&sizes, 200).unwrap();

        assert_eq!(absolute, vec![10, 50, 100]);
    }

    #[test]
    fn test_convert_train_sizes_mixed() {
        let sizes = vec![0.1, 25.0, 0.5];
        let absolute = convert_train_sizes(&sizes, 100).unwrap();

        assert_eq!(absolute, vec![10, 25, 50]);
    }

    #[test]
    fn test_convert_train_sizes_invalid() {
        let sizes = vec![0.0];
        assert!(convert_train_sizes(&sizes, 100).is_err());

        let sizes = vec![-0.5];
        assert!(convert_train_sizes(&sizes, 100).is_err());
    }

    #[test]
    fn test_learning_curve_basic() {
        let n_samples = 50;
        let n_features = 2;

        let x_data: Vec<f64> = (0..n_samples * n_features)
            .map(|i| (i as f64) / 10.0)
            .collect();
        let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(5);
        let train_sizes = vec![0.2, 0.5, 0.8];
        let config = LearningCurveConfig::default();

        let result =
            learning_curve(&estimator, &x, &y, &cv, &metric, &train_sizes, config).unwrap();

        // Check dimensions
        assert_eq!(result.train_sizes.len(), 3);
        assert_eq!(result.train_scores.nrows(), 3);
        assert_eq!(result.train_scores.ncols(), 5);
        assert_eq!(result.test_scores.nrows(), 3);
        assert_eq!(result.test_scores.ncols(), 5);
        assert_eq!(result.train_scores_summary.len(), 3);
        assert_eq!(result.test_scores_summary.len(), 3);
        assert_eq!(result.n_samples, n_samples);

        // All scores should be finite
        for score in result.train_scores.iter() {
            assert!(score.is_finite());
        }
        for score in result.test_scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_learning_curve_with_shuffle() {
        let n_samples = 30;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..60).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(3);
        let train_sizes = vec![0.5, 1.0];

        // Without shuffle
        let config_no_shuffle = LearningCurveConfig::default();
        let result1 = learning_curve(
            &estimator,
            &x,
            &y,
            &cv,
            &metric,
            &train_sizes,
            config_no_shuffle,
        )
        .unwrap();

        // With shuffle
        let config_shuffle = LearningCurveConfig::default().with_shuffle(true, Some(42));
        let result2 = learning_curve(
            &estimator,
            &x,
            &y,
            &cv,
            &metric,
            &train_sizes,
            config_shuffle,
        )
        .unwrap();

        // Both should have valid results
        assert_eq!(result1.train_sizes.len(), 2);
        assert_eq!(result2.train_sizes.len(), 2);
    }

    #[test]
    fn test_learning_curve_shape_mismatch() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec(vec![1.0; 5]); // Wrong length

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(2);
        let train_sizes = vec![0.5];
        let config = LearningCurveConfig::default();

        let result = learning_curve(&estimator, &x, &y, &cv, &metric, &train_sizes, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_learning_curve_empty_sizes() {
        let x = Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..20).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(2);
        let train_sizes: Vec<f64> = vec![];
        let config = LearningCurveConfig::default();

        let result = learning_curve(&estimator, &x, &y, &cv, &metric, &train_sizes, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_learning_curve_result_summary() {
        let n_samples = 30;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..60).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(3);
        let train_sizes = vec![0.5, 1.0];
        let config = LearningCurveConfig::default();

        let result =
            learning_curve(&estimator, &x, &y, &cv, &metric, &train_sizes, config).unwrap();

        let summary = result.summary();
        assert!(summary.contains("Learning Curve Results"));
        assert!(summary.contains("n="));
        assert!(summary.contains("train="));
        assert!(summary.contains("test="));
    }

    #[test]
    fn test_learning_curve_iter_summaries() {
        let n_samples = 20;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(2);
        let train_sizes = vec![0.5, 1.0];
        let config = LearningCurveConfig::default();

        let result =
            learning_curve(&estimator, &x, &y, &cv, &metric, &train_sizes, config).unwrap();

        let summaries: Vec<_> = result.iter_summaries().collect();
        assert_eq!(summaries.len(), 2);

        for (size, train, test) in summaries {
            assert!(size > 0);
            assert!(train.mean.is_finite());
            assert!(test.mean.is_finite());
        }
    }

    // ========================================================================
    // Validation curve tests
    // ========================================================================

    #[test]
    fn test_validation_curve_basic() {
        let n_samples = 40;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..80).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(4);
        let param_values = vec![0.01, 0.1, 1.0, 10.0];
        let config = ValidationCurveConfig::default();

        let result = validation_curve(
            &estimator,
            &x,
            &y,
            &cv,
            &metric,
            "regularization",
            &param_values,
            config,
        )
        .unwrap();

        // Check dimensions
        assert_eq!(result.param_name, "regularization");
        assert_eq!(result.param_values.len(), 4);
        assert_eq!(result.train_scores.nrows(), 4);
        assert_eq!(result.train_scores.ncols(), 4);
        assert_eq!(result.test_scores.nrows(), 4);
        assert_eq!(result.test_scores.ncols(), 4);
        assert_eq!(result.train_scores_summary.len(), 4);
        assert_eq!(result.test_scores_summary.len(), 4);

        // All scores should be finite
        for score in result.train_scores.iter() {
            assert!(score.is_finite());
        }
        for score in result.test_scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_validation_curve_unknown_param() {
        let x = Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..20).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(2);
        let param_values = vec![1.0, 2.0];
        let config = ValidationCurveConfig::default();

        let result = validation_curve(
            &estimator,
            &x,
            &y,
            &cv,
            &metric,
            "unknown_param",
            &param_values,
            config,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_curve_empty_values() {
        let x = Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..20).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(2);
        let param_values: Vec<f64> = vec![];
        let config = ValidationCurveConfig::default();

        let result = validation_curve(
            &estimator,
            &x,
            &y,
            &cv,
            &metric,
            "regularization",
            &param_values,
            config,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_curve_best_param() {
        let n_samples = 30;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..60).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(3);
        let param_values = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let config = ValidationCurveConfig::default();

        let result = validation_curve(
            &estimator,
            &x,
            &y,
            &cv,
            &metric,
            "regularization",
            &param_values,
            config,
        )
        .unwrap();

        // For MSE (minimize), best param should exist
        let best = result.best_param_value(false);
        assert!(best.is_some());
        assert!(param_values.contains(&best.unwrap()));
    }

    #[test]
    fn test_validation_curve_result_summary() {
        let n_samples = 20;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(2);
        let param_values = vec![0.1, 1.0];
        let config = ValidationCurveConfig::default();

        let result = validation_curve(
            &estimator,
            &x,
            &y,
            &cv,
            &metric,
            "regularization",
            &param_values,
            config,
        )
        .unwrap();

        let summary = result.summary();
        assert!(summary.contains("Validation Curve Results"));
        assert!(summary.contains("regularization"));
        assert!(summary.contains("train="));
        assert!(summary.contains("test="));
    }

    #[test]
    fn test_validation_curve_iter_summaries() {
        let n_samples = 20;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator::new();
        let metric = MockMSE;
        let cv = KFold::new(2);
        let param_values = vec![0.1, 1.0, 10.0];
        let config = ValidationCurveConfig::default();

        let result = validation_curve(
            &estimator,
            &x,
            &y,
            &cv,
            &metric,
            "regularization",
            &param_values,
            config,
        )
        .unwrap();

        let summaries: Vec<_> = result.iter_summaries().collect();
        assert_eq!(summaries.len(), 3);

        for (value, train, test) in summaries {
            assert!(value > 0.0);
            assert!(train.mean.is_finite());
            assert!(test.mean.is_finite());
        }
    }
}
