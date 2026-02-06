//! Cross-Validation Strategies
//!
//! This module provides cross-validation with proper statistical handling of results.
//! Every CV strategy follows FerroML's philosophy of statistical rigor with support
//! for confidence intervals and reproducibility.
//!
//! ## Design Philosophy
//!
//! - **Reproducibility**: All CV splitters accept optional seeds for deterministic splits
//! - **Flexibility**: Trait-based design allows custom CV strategies
//! - **Statistical rigor**: Results include CIs via bootstrap across folds
//!
//! ## Strategies
//!
//! - [`KFold`](struct.KFold.html) - Standard k-fold cross-validation
//! - [`RepeatedKFold`](struct.RepeatedKFold.html) - Multiple repetitions of k-fold
//! - [`StratifiedKFold`](struct.StratifiedKFold.html) - Preserves class distribution
//! - [`RepeatedStratifiedKFold`](struct.RepeatedStratifiedKFold.html) - Repeated stratified k-fold
//! - [`TimeSeriesSplit`](struct.TimeSeriesSplit.html) - For temporal data
//! - [`GroupKFold`](struct.GroupKFold.html) - Respects group boundaries
//! - [`StratifiedGroupKFold`](struct.StratifiedGroupKFold.html) - Groups + stratification
//! - [`LeaveOneOut`](struct.LeaveOneOut.html) - Each sample as a test set (n folds)
//! - [`LeavePOut`](struct.LeavePOut.html) - Each combination of p samples as test set
//! - [`ShuffleSplit`](struct.ShuffleSplit.html) - Random train/test splits with configurable sizes
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::cv::{CrossValidator, KFold};
//!
//! // Create 5-fold cross-validator
//! let cv = KFold::new(5).with_shuffle(true).with_seed(42);
//!
//! // Generate splits for 100 samples
//! let splits = cv.split(100, None, None)?;
//!
//! for (fold_idx, fold) in splits.iter().enumerate() {
//!     println!("Fold {}: train={}, test={}",
//!              fold_idx, fold.train_indices.len(), fold.test_indices.len());
//! }
//! ```

use crate::metrics::Metric;
use crate::traits::{Estimator, Predictor};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

mod curves;
mod group;
mod kfold;
mod loo;
mod nested;
mod stratified;
mod timeseries;

pub use curves::{
    learning_curve, validation_curve, LearningCurveConfig, LearningCurveResult, ParameterSettable,
    ScoreSummary, ValidationCurveConfig, ValidationCurveResult,
};
pub use group::{GroupKFold, StratifiedGroupKFold};
pub use kfold::{KFold, RepeatedKFold};
pub use loo::{LeaveOneOut, LeavePOut, ShuffleSplit};
pub use nested::{nested_cv_score, NestedCVConfig, NestedCVFoldResult, NestedCVResult};
pub use stratified::{RepeatedStratifiedKFold, StratifiedKFold};
pub use timeseries::TimeSeriesSplit;

/// A single fold in cross-validation, containing train and test indices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVFold {
    /// Indices for training samples
    pub train_indices: Vec<usize>,
    /// Indices for test/validation samples
    pub test_indices: Vec<usize>,
    /// Fold number (0-indexed)
    pub fold_index: usize,
}

impl CVFold {
    /// Create a new CV fold
    pub fn new(train_indices: Vec<usize>, test_indices: Vec<usize>, fold_index: usize) -> Self {
        Self {
            train_indices,
            test_indices,
            fold_index,
        }
    }

    /// Get the number of training samples
    pub fn n_train(&self) -> usize {
        self.train_indices.len()
    }

    /// Get the number of test samples
    pub fn n_test(&self) -> usize {
        self.test_indices.len()
    }
}

/// Result from evaluating a model on one CV fold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVFoldResult {
    /// The fold that was evaluated
    pub fold_index: usize,
    /// Train score (if computed)
    pub train_score: Option<f64>,
    /// Test/validation score
    pub test_score: f64,
    /// Time taken for this fold in seconds
    pub fit_time_secs: f64,
    /// Time taken for scoring in seconds
    pub score_time_secs: f64,
}

/// Aggregated results from cross-validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVResult {
    /// Results from each fold
    pub fold_results: Vec<CVFoldResult>,
    /// Mean test score across folds
    pub mean_test_score: f64,
    /// Standard deviation of test scores
    pub std_test_score: f64,
    /// Mean train score (if computed)
    pub mean_train_score: Option<f64>,
    /// Std of train scores (if computed)
    pub std_train_score: Option<f64>,
    /// 95% confidence interval for test score (lower bound)
    pub ci_lower: f64,
    /// 95% confidence interval for test score (upper bound)
    pub ci_upper: f64,
    /// Confidence level used (default 0.95)
    pub confidence_level: f64,
    /// Total number of folds
    pub n_folds: usize,
    /// Total samples in dataset
    pub n_samples: usize,
    /// Raw test scores from each fold
    scores: Vec<f64>,
}

impl CVResult {
    /// Create a new CVResult from fold results
    pub fn from_fold_results(
        fold_results: Vec<CVFoldResult>,
        n_samples: usize,
        confidence_level: f64,
    ) -> Self {
        let n_folds = fold_results.len();
        let test_scores: Vec<f64> = fold_results.iter().map(|r| r.test_score).collect();

        // Compute mean and std of test scores
        let mean_test_score = test_scores.iter().sum::<f64>() / n_folds as f64;
        let std_test_score = if n_folds > 1 {
            let variance = test_scores
                .iter()
                .map(|s| (s - mean_test_score).powi(2))
                .sum::<f64>()
                / (n_folds - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Compute train score stats if available
        let train_scores: Vec<f64> = fold_results.iter().filter_map(|r| r.train_score).collect();

        let (mean_train_score, std_train_score) = if train_scores.len() == n_folds {
            let mean = train_scores.iter().sum::<f64>() / n_folds as f64;
            let std = if n_folds > 1 {
                let variance = train_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                    / (n_folds - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };
            (Some(mean), Some(std))
        } else {
            (None, None)
        };

        // Compute confidence interval using t-distribution
        // CI = mean ± t * (std / sqrt(n))
        let (ci_lower, ci_upper) = compute_t_confidence_interval(
            mean_test_score,
            std_test_score,
            n_folds,
            confidence_level,
        );

        Self {
            fold_results,
            mean_test_score,
            std_test_score,
            mean_train_score,
            std_train_score,
            ci_lower,
            ci_upper,
            confidence_level,
            n_folds,
            n_samples,
            scores: test_scores,
        }
    }

    /// Get raw test scores from each fold
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "{:.4} ± {:.4} ({}% CI: [{:.4}, {:.4}], n_folds={})",
            self.mean_test_score,
            self.std_test_score,
            (self.confidence_level * 100.0) as i32,
            self.ci_lower,
            self.ci_upper,
            self.n_folds
        )
    }

    /// Check if this result is significantly better than another
    /// Uses non-overlapping confidence intervals as a conservative test
    pub fn significantly_better_than(&self, other: &CVResult, maximize: bool) -> bool {
        if maximize {
            self.ci_lower > other.ci_upper
        } else {
            self.ci_upper < other.ci_lower
        }
    }
}

/// Compute t-distribution confidence interval
pub(crate) fn compute_t_confidence_interval(
    mean: f64,
    std: f64,
    n: usize,
    confidence: f64,
) -> (f64, f64) {
    use statrs::distribution::{ContinuousCDF, StudentsT};

    if n <= 1 || std == 0.0 {
        return (mean, mean);
    }

    let df = (n - 1) as f64;
    let alpha = 1.0 - confidence;

    // Get t critical value
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let t_crit = t_dist.inverse_cdf(1.0 - alpha / 2.0);

    let margin = t_crit * std / (n as f64).sqrt();

    (mean - margin, mean + margin)
}

/// Core trait for cross-validation strategies
///
/// All cross-validators implement this trait, which provides methods for
/// generating train/test splits from data indices.
///
/// # Type Parameters
///
/// The trait is object-safe and supports dynamic dispatch.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::cv::{CrossValidator, KFold};
///
/// let cv = KFold::new(5);
/// let splits = cv.split(100, None, None)?;
///
/// assert_eq!(splits.len(), 5);
/// for fold in &splits {
///     assert_eq!(fold.train_indices.len(), 80);
///     assert_eq!(fold.test_indices.len(), 20);
/// }
/// ```
pub trait CrossValidator: Send + Sync {
    /// Generate train/test splits for the given number of samples
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Total number of samples to split
    /// * `y` - Optional target labels (required for stratified CV)
    /// * `groups` - Optional group labels (required for group-based CV)
    ///
    /// # Returns
    ///
    /// Vector of `CVFold` structs, each containing train and test indices
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if n_samples is too small for the CV strategy
    /// - `InvalidInput` if y or groups are required but not provided
    fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<Vec<CVFold>>;

    /// Get the number of splits/folds this strategy will produce
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Total number of samples (may affect n_splits for LOO)
    /// * `y` - Optional labels (may affect n_splits for stratified)
    /// * `groups` - Optional groups (may affect n_splits for group-based)
    fn get_n_splits(
        &self,
        n_samples: Option<usize>,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> usize;

    /// Get a descriptive name for this cross-validator
    fn name(&self) -> &str;

    /// Check if this cross-validator requires labels (y)
    fn requires_labels(&self) -> bool {
        false
    }

    /// Check if this cross-validator requires group information
    fn requires_groups(&self) -> bool {
        false
    }

    /// Validate inputs before splitting
    ///
    /// Default implementation checks basic requirements; override for specific needs
    fn validate_inputs(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<()> {
        if n_samples == 0 {
            return Err(FerroError::invalid_input("Cannot split empty dataset"));
        }

        if self.requires_labels() && y.is_none() {
            return Err(FerroError::invalid_input(format!(
                "{} requires labels (y) for splitting",
                self.name()
            )));
        }

        if self.requires_groups() && groups.is_none() {
            return Err(FerroError::invalid_input(format!(
                "{} requires group information for splitting",
                self.name()
            )));
        }

        if let Some(labels) = y {
            if labels.len() != n_samples {
                return Err(FerroError::shape_mismatch(
                    format!("n_samples={}", n_samples),
                    format!("len(y)={}", labels.len()),
                ));
            }
        }

        if let Some(group_labels) = groups {
            if group_labels.len() != n_samples {
                return Err(FerroError::shape_mismatch(
                    format!("n_samples={}", n_samples),
                    format!("len(groups)={}", group_labels.len()),
                ));
            }
        }

        Ok(())
    }
}

/// Configuration for cross-validation execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVConfig {
    /// Whether to compute train scores in addition to test scores
    pub return_train_score: bool,
    /// Confidence level for intervals (default 0.95)
    pub confidence_level: f64,
    /// Number of parallel jobs (-1 for all CPUs)
    pub n_jobs: i32,
    /// Verbosity level (0=silent, 1=progress, 2=detailed)
    pub verbose: u8,
    /// Random seed for reproducibility (passed to CV splitter)
    pub random_seed: Option<u64>,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            return_train_score: false,
            confidence_level: 0.95,
            n_jobs: 1,
            verbose: 0,
            random_seed: None,
        }
    }
}

impl CVConfig {
    /// Create with train score computation enabled
    pub fn with_train_score(mut self) -> Self {
        self.return_train_score = true;
        self
    }

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

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

// ============================================================================
// Utility functions for CV implementations
// ============================================================================

/// Shuffle indices using Fisher-Yates algorithm
pub(crate) fn shuffle_indices(indices: &mut [usize], seed: u64) {
    use rand::prelude::*;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for i in (1..indices.len()).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }
}

/// Get unique classes and their counts from labels
pub(crate) fn get_class_distribution(y: &Array1<f64>) -> HashMap<i64, Vec<usize>> {
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();

    for (idx, &label) in y.iter().enumerate() {
        // Convert float labels to integers for grouping
        let class = label.round() as i64;
        class_indices.entry(class).or_default().push(idx);
    }

    class_indices
}

/// Get unique groups and their sample indices
pub(crate) fn get_group_indices(groups: &Array1<i64>) -> HashMap<i64, Vec<usize>> {
    let mut group_indices: HashMap<i64, Vec<usize>> = HashMap::new();

    for (idx, &group) in groups.iter().enumerate() {
        group_indices.entry(group).or_default().push(idx);
    }

    group_indices
}

/// Validate that n_folds is appropriate for n_samples
pub(crate) fn validate_n_folds(n_folds: usize, n_samples: usize) -> Result<()> {
    if n_folds < 2 {
        return Err(FerroError::invalid_input(
            "Cross-validation requires at least 2 folds",
        ));
    }

    if n_folds > n_samples {
        return Err(FerroError::invalid_input(format!(
            "Cannot have more folds ({}) than samples ({})",
            n_folds, n_samples
        )));
    }

    Ok(())
}

// ============================================================================
// Cross-Validation Score Functions
// ============================================================================

/// Evaluate an estimator using cross-validation with parallel fold execution.
///
/// This function fits the estimator on training folds and evaluates on test folds,
/// computing scores with proper statistical handling including confidence intervals.
///
/// # Arguments
///
/// * `estimator` - The estimator to evaluate (must be Clone for parallel execution)
/// * `x` - Feature matrix of shape (n_samples, n_features)
/// * `y` - Target vector of length n_samples
/// * `cv` - Cross-validation strategy
/// * `metric` - Scoring metric to use
/// * `config` - Cross-validation configuration
/// * `groups` - Optional group labels for group-based CV
///
/// # Returns
///
/// `CVResult` containing scores from all folds with confidence intervals.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::cv::{cross_val_score, KFold, CVConfig};
/// use ferroml_core::metrics::accuracy;
///
/// let cv = KFold::new(5).with_shuffle(true).with_seed(42);
/// let config = CVConfig::default().with_train_score();
///
/// let result = cross_val_score(
///     &model,
///     &x,
///     &y,
///     &cv,
///     &AccuracyMetric,
///     &config,
///     None,
/// )?;
///
/// println!("CV Score: {}", result.summary());
/// ```
///
/// # Parallel Execution
///
/// When `config.n_jobs != 1`, folds are evaluated in parallel using rayon.
/// Set `n_jobs = -1` to use all available CPUs.
///
/// # Statistical Rigor
///
/// The returned `CVResult` includes:
/// - Mean and standard deviation of scores
/// - t-distribution confidence interval (appropriate for small n_folds)
/// - Per-fold results for detailed analysis
pub fn cross_val_score<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    metric: &M,
    config: &CVConfig,
    groups: Option<&Array1<i64>>,
) -> Result<CVResult>
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

    // Get CV splits
    let folds = cv.split(n_samples, Some(y), groups)?;
    let n_folds = folds.len();

    if n_folds == 0 {
        return Err(FerroError::invalid_input(
            "Cross-validator produced no folds",
        ));
    }

    // Determine parallelism
    let use_parallel = config.n_jobs != 1 && n_folds > 1;

    // Evaluate folds
    let fold_results: Vec<Result<CVFoldResult>> = if use_parallel {
        folds
            .into_par_iter()
            .map(|fold| evaluate_fold(estimator, x, y, &fold, metric, config.return_train_score))
            .collect()
    } else {
        folds
            .into_iter()
            .map(|fold| evaluate_fold(estimator, x, y, &fold, metric, config.return_train_score))
            .collect()
    };

    // Check for errors and collect successful results
    let mut successful_results = Vec::with_capacity(n_folds);
    let mut errors = Vec::new();

    for (idx, result) in fold_results.into_iter().enumerate() {
        match result {
            Ok(fold_result) => successful_results.push(fold_result),
            Err(e) => errors.push((idx, e)),
        }
    }

    // If all folds failed, return an error
    if successful_results.is_empty() {
        let error_msgs: Vec<String> = errors
            .iter()
            .map(|(idx, e)| format!("Fold {}: {}", idx, e))
            .collect();
        return Err(FerroError::cross_validation(format!(
            "All folds failed: {}",
            error_msgs.join("; ")
        )));
    }

    // Log warnings for partial failures (if we had a logger)
    // For now, we proceed with successful folds

    // Build the CVResult
    Ok(CVResult::from_fold_results(
        successful_results,
        n_samples,
        config.confidence_level,
    ))
}

/// Evaluate a single fold of cross-validation
fn evaluate_fold<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    fold: &CVFold,
    metric: &M,
    return_train_score: bool,
) -> Result<CVFoldResult>
where
    E: Estimator + Clone,
    M: Metric,
{
    // Extract train and test data
    let x_train = select_rows(x, &fold.train_indices);
    let y_train = select_elements(y, &fold.train_indices);
    let x_test = select_rows(x, &fold.test_indices);
    let y_test = select_elements(y, &fold.test_indices);

    // Fit the model
    let fit_start = Instant::now();
    let fitted = estimator.clone().fit(&x_train, &y_train)?;
    let fit_time_secs = fit_start.elapsed().as_secs_f64();

    // Score on test set
    let score_start = Instant::now();
    let y_pred_test = fitted.predict(&x_test)?;
    let test_metric = metric.compute(&y_test, &y_pred_test)?;
    let test_score = test_metric.value;

    // Optionally score on train set
    let train_score = if return_train_score {
        let y_pred_train = fitted.predict(&x_train)?;
        let train_metric = metric.compute(&y_train, &y_pred_train)?;
        Some(train_metric.value)
    } else {
        None
    };

    let score_time_secs = score_start.elapsed().as_secs_f64();

    Ok(CVFoldResult {
        fold_index: fold.fold_index,
        train_score,
        test_score,
        fit_time_secs,
        score_time_secs,
    })
}

/// Select rows from a 2D array by indices
pub(crate) fn select_rows(array: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_cols = array.ncols();
    let n_rows = indices.len();

    let mut data = Vec::with_capacity(n_rows * n_cols);
    for &idx in indices {
        for val in array.row(idx).iter() {
            data.push(*val);
        }
    }

    Array2::from_shape_vec((n_rows, n_cols), data)
        .expect("select_rows: shape mismatch - data length does not match n_rows * n_cols")
}

/// Select elements from a 1D array by indices
pub(crate) fn select_elements(array: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_vec(indices.iter().map(|&idx| array[idx]).collect())
}

/// Convenience function to run cross-validation with default config
///
/// This is a simpler interface when you don't need custom configuration.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::cv::{cross_val_score_simple, KFold};
///
/// let cv = KFold::new(5);
/// let result = cross_val_score_simple(&model, &x, &y, &cv, &AccuracyMetric)?;
/// ```
pub fn cross_val_score_simple<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    metric: &M,
) -> Result<CVResult>
where
    E: Estimator + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,
{
    cross_val_score(estimator, x, y, cv, metric, &CVConfig::default(), None)
}

/// Cross-validation scores returning only the test scores array
///
/// This matches the sklearn-style interface more closely.
///
/// # Returns
///
/// Array of test scores, one per fold.
pub fn cross_val_score_array<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    metric: &M,
    config: &CVConfig,
    groups: Option<&Array1<i64>>,
) -> Result<Array1<f64>>
where
    E: Estimator + Clone + Send + Sync,
    E::Fitted: Send,
    M: Metric + Sync,
{
    let result = cross_val_score(estimator, x, y, cv, metric, config, groups)?;
    let scores: Vec<f64> = result.fold_results.iter().map(|r| r.test_score).collect();
    Ok(Array1::from_vec(scores))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_fold_creation() {
        let fold = CVFold::new(vec![0, 1, 2, 3], vec![4, 5], 0);
        assert_eq!(fold.n_train(), 4);
        assert_eq!(fold.n_test(), 2);
        assert_eq!(fold.fold_index, 0);
    }

    #[test]
    fn test_cv_result_from_folds() {
        let fold_results = vec![
            CVFoldResult {
                fold_index: 0,
                train_score: Some(0.95),
                test_score: 0.80,
                fit_time_secs: 1.0,
                score_time_secs: 0.1,
            },
            CVFoldResult {
                fold_index: 1,
                train_score: Some(0.94),
                test_score: 0.82,
                fit_time_secs: 1.1,
                score_time_secs: 0.1,
            },
            CVFoldResult {
                fold_index: 2,
                train_score: Some(0.96),
                test_score: 0.78,
                fit_time_secs: 0.9,
                score_time_secs: 0.1,
            },
        ];

        let result = CVResult::from_fold_results(fold_results, 100, 0.95);

        assert_eq!(result.n_folds, 3);
        assert!((result.mean_test_score - 0.80).abs() < 0.01);
        assert!(result.std_test_score > 0.0);
        assert!(result.ci_lower < result.mean_test_score);
        assert!(result.ci_upper > result.mean_test_score);
        assert!(result.mean_train_score.is_some());
    }

    #[test]
    fn test_cv_result_summary() {
        let fold_results = vec![
            CVFoldResult {
                fold_index: 0,
                train_score: None,
                test_score: 0.80,
                fit_time_secs: 1.0,
                score_time_secs: 0.1,
            },
            CVFoldResult {
                fold_index: 1,
                train_score: None,
                test_score: 0.80,
                fit_time_secs: 1.0,
                score_time_secs: 0.1,
            },
        ];

        let result = CVResult::from_fold_results(fold_results, 100, 0.95);
        let summary = result.summary();

        assert!(summary.contains("0.80"));
        assert!(summary.contains("95%"));
        assert!(summary.contains("n_folds=2"));
    }

    #[test]
    fn test_shuffle_indices() {
        let mut indices: Vec<usize> = (0..10).collect();
        let original = indices.clone();

        shuffle_indices(&mut indices, 42);

        // Should contain same elements
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, original);

        // Should be shuffled (with very high probability)
        assert_ne!(indices, original);

        // Should be deterministic with same seed
        let mut indices2: Vec<usize> = (0..10).collect();
        shuffle_indices(&mut indices2, 42);
        assert_eq!(indices, indices2);
    }

    #[test]
    fn test_get_class_distribution() {
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 2.0]);
        let dist = get_class_distribution(&y);

        assert_eq!(dist.get(&0).unwrap().len(), 2);
        assert_eq!(dist.get(&1).unwrap().len(), 3);
        assert_eq!(dist.get(&2).unwrap().len(), 1);
    }

    #[test]
    fn test_get_group_indices() {
        let groups = Array1::from_vec(vec![1, 1, 2, 2, 2, 3]);
        let indices = get_group_indices(&groups);

        assert_eq!(indices.get(&1).unwrap().len(), 2);
        assert_eq!(indices.get(&2).unwrap().len(), 3);
        assert_eq!(indices.get(&3).unwrap().len(), 1);
    }

    #[test]
    fn test_validate_n_folds() {
        assert!(validate_n_folds(5, 100).is_ok());
        assert!(validate_n_folds(2, 10).is_ok());

        // Too few folds
        assert!(validate_n_folds(1, 100).is_err());
        assert!(validate_n_folds(0, 100).is_err());

        // More folds than samples
        assert!(validate_n_folds(10, 5).is_err());
    }

    #[test]
    fn test_cv_config_builder() {
        let config = CVConfig::default()
            .with_train_score()
            .with_confidence(0.99)
            .with_n_jobs(-1)
            .with_seed(42);

        assert!(config.return_train_score);
        assert!((config.confidence_level - 0.99).abs() < 1e-10);
        assert_eq!(config.n_jobs, -1);
        assert_eq!(config.random_seed, Some(42));
    }

    #[test]
    fn test_t_confidence_interval() {
        // With 30 samples, should be close to z-interval
        let (lower, upper) = compute_t_confidence_interval(0.8, 0.1, 30, 0.95);

        // Mean ± 1.96 * SE for large n
        // SE = 0.1 / sqrt(30) ≈ 0.0183
        // Margin ≈ 0.036
        assert!(lower > 0.75);
        assert!(lower < 0.78);
        assert!(upper > 0.82);
        assert!(upper < 0.85);
    }

    #[test]
    fn test_cv_result_comparison() {
        // Create two CV results with non-overlapping CIs
        let result1 = CVResult {
            fold_results: vec![],
            mean_test_score: 0.90,
            std_test_score: 0.02,
            mean_train_score: None,
            std_train_score: None,
            ci_lower: 0.88,
            ci_upper: 0.92,
            confidence_level: 0.95,
            n_folds: 5,
            n_samples: 100,
            scores: vec![],
        };

        let result2 = CVResult {
            fold_results: vec![],
            mean_test_score: 0.80,
            std_test_score: 0.02,
            mean_train_score: None,
            std_train_score: None,
            ci_lower: 0.78,
            ci_upper: 0.82,
            confidence_level: 0.95,
            n_folds: 5,
            n_samples: 100,
            scores: vec![],
        };

        // For maximization, result1 is significantly better
        assert!(result1.significantly_better_than(&result2, true));
        assert!(!result2.significantly_better_than(&result1, true));

        // For minimization, result2 is significantly better
        assert!(result2.significantly_better_than(&result1, false));
        assert!(!result1.significantly_better_than(&result2, false));
    }

    // ========================================================================
    // Tests for cross_val_score
    // ========================================================================

    use crate::hpo::SearchSpace;
    use crate::metrics::{Direction, MetricValue};

    /// A simple mock model that predicts the mean of training y values
    #[derive(Clone)]
    struct MockMeanEstimator;

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
            Ok(MockMeanPredictor { mean })
        }

        fn search_space(&self) -> SearchSpace {
            SearchSpace::new()
        }
    }

    /// Mock metric: Mean Squared Error
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

    #[test]
    fn test_select_rows() {
        let array = Array2::from_shape_vec((5, 3), (0..15).map(|x| x as f64).collect()).unwrap();
        let indices = vec![0, 2, 4];
        let selected = select_rows(&array, &indices);

        assert_eq!(selected.nrows(), 3);
        assert_eq!(selected.ncols(), 3);
        assert_eq!(selected[[0, 0]], 0.0); // First row
        assert_eq!(selected[[1, 0]], 6.0); // Third row
        assert_eq!(selected[[2, 0]], 12.0); // Fifth row
    }

    #[test]
    fn test_select_elements() {
        let array = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let indices = vec![1, 3];
        let selected = select_elements(&array, &indices);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], 20.0);
        assert_eq!(selected[1], 40.0);
    }

    #[test]
    fn test_cross_val_score_basic() {
        // Create simple linear data: y = x + noise
        let n_samples = 50;
        let n_features = 2;

        let x_data: Vec<f64> = (0..n_samples * n_features)
            .map(|i| (i as f64) / 10.0)
            .collect();
        let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();

        // y is just the sum of features (simple for mean predictor to approximate)
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator;
        let metric = MockMSE;
        let cv = KFold::new(5);
        let config = CVConfig::default();

        let result = cross_val_score(&estimator, &x, &y, &cv, &metric, &config, None).unwrap();

        // Should have 5 folds
        assert_eq!(result.n_folds, 5);
        assert_eq!(result.fold_results.len(), 5);
        assert_eq!(result.n_samples, n_samples);

        // MSE should be finite and positive
        assert!(result.mean_test_score.is_finite());
        assert!(result.mean_test_score >= 0.0);

        // CI should be valid
        assert!(result.ci_lower <= result.mean_test_score);
        assert!(result.ci_upper >= result.mean_test_score);
    }

    #[test]
    fn test_cross_val_score_with_train_scores() {
        let n_samples = 30;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..60).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator;
        let metric = MockMSE;
        let cv = KFold::new(3);
        let config = CVConfig::default().with_train_score();

        let result = cross_val_score(&estimator, &x, &y, &cv, &metric, &config, None).unwrap();

        // Should have train scores
        assert!(result.mean_train_score.is_some());
        assert!(result.std_train_score.is_some());

        // Train scores should be present for all folds
        for fold_result in &result.fold_results {
            assert!(fold_result.train_score.is_some());
        }

        // Train MSE should be lower than test MSE (model fits better on train)
        let train_mse = result.mean_train_score.unwrap();
        let test_mse = result.mean_test_score;
        assert!(
            train_mse <= test_mse + 1.0,
            "Expected train MSE ({}) <= test MSE ({})",
            train_mse,
            test_mse
        );
    }

    #[test]
    fn test_cross_val_score_parallel() {
        let n_samples = 40;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..80).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator;
        let metric = MockMSE;
        let cv = KFold::new(4);

        // Run with single thread
        let config_single = CVConfig::default().with_n_jobs(1);
        let result_single =
            cross_val_score(&estimator, &x, &y, &cv, &metric, &config_single, None).unwrap();

        // Run with parallel
        let config_parallel = CVConfig::default().with_n_jobs(-1);
        let result_parallel =
            cross_val_score(&estimator, &x, &y, &cv, &metric, &config_parallel, None).unwrap();

        // Results should be the same (deterministic without shuffle)
        assert!(
            (result_single.mean_test_score - result_parallel.mean_test_score).abs() < 1e-10,
            "Single: {}, Parallel: {}",
            result_single.mean_test_score,
            result_parallel.mean_test_score
        );
    }

    #[test]
    fn test_cross_val_score_array() {
        let n_samples = 25;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..50).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator;
        let metric = MockMSE;
        let cv = KFold::new(5);
        let config = CVConfig::default();

        let scores =
            cross_val_score_array(&estimator, &x, &y, &cv, &metric, &config, None).unwrap();

        // Should have one score per fold
        assert_eq!(scores.len(), 5);

        // All scores should be finite
        for score in scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_cross_val_score_simple() {
        let n_samples = 20;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator;
        let metric = MockMSE;
        let cv = KFold::new(4);

        let result = cross_val_score_simple(&estimator, &x, &y, &cv, &metric).unwrap();

        assert_eq!(result.n_folds, 4);
        assert!(result.mean_test_score.is_finite());
    }

    #[test]
    fn test_cross_val_score_shape_mismatch() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec(vec![1.0; 5]); // Wrong length

        let estimator = MockMeanEstimator;
        let metric = MockMSE;
        let cv = KFold::new(2);
        let config = CVConfig::default();

        let result = cross_val_score(&estimator, &x, &y, &cv, &metric, &config, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_cross_val_score_with_stratified() {
        let n_samples = 30;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..60).map(|x| x as f64).collect()).unwrap();
        // Binary classification labels
        let y = Array1::from_vec((0..n_samples).map(|i| (i % 2) as f64).collect());

        let estimator = MockMeanEstimator;
        let metric = MockMSE;
        let cv = StratifiedKFold::new(3);
        let config = CVConfig::default();

        let result = cross_val_score(&estimator, &x, &y, &cv, &metric, &config, None).unwrap();

        assert_eq!(result.n_folds, 3);
        assert!(result.mean_test_score.is_finite());
    }

    #[test]
    fn test_cross_val_score_timing() {
        let n_samples = 20;
        let x =
            Array2::from_shape_vec((n_samples, 2), (0..40).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let estimator = MockMeanEstimator;
        let metric = MockMSE;
        let cv = KFold::new(3);
        let config = CVConfig::default();

        let result = cross_val_score(&estimator, &x, &y, &cv, &metric, &config, None).unwrap();

        // Check that timing information is recorded
        for fold_result in &result.fold_results {
            assert!(fold_result.fit_time_secs >= 0.0);
            assert!(fold_result.score_time_secs >= 0.0);
        }
    }
}
