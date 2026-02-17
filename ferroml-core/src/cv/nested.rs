//! Nested Cross-Validation
//!
//! Nested cross-validation provides unbiased performance estimates when
//! hyperparameter optimization is involved. It prevents information leakage
//! by separating model selection (inner loop) from model evaluation (outer loop).
//!
//! ## Why Nested CV?
//!
//! Standard CV with HPO suffers from optimistic bias: the same data used to select
//! hyperparameters is also used to estimate performance. Nested CV fixes this by:
//!
//! 1. **Outer loop**: Held-out test folds for unbiased performance estimation
//! 2. **Inner loop**: HPO using only the outer training fold
//!
//! ## Example
//!
//! ```
//! # fn main() -> ferroml_core::Result<()> {
//! # use ferroml_core::traits::{Estimator, Predictor, PredictionWithUncertainty};
//! # use ferroml_core::metrics::{Metric, MetricValue};
//! # use ndarray::{Array1, Array2};
//! # use std::collections::HashMap;
//! # use ferroml_core::hpo::ParameterValue;
//! use ferroml_core::hpo::Direction;
//! # #[derive(Clone)]
//! # struct MyEstimator { alpha: f64 }
//! # impl MyEstimator { fn new(alpha: f64) -> Self { Self { alpha } } }
//! # struct MyFitted(f64);
//! # impl Predictor for MyFitted {
//! #     fn predict(&self, x: &Array2<f64>) -> ferroml_core::Result<Array1<f64>> { Ok(Array1::from_elem(x.nrows(), self.0)) }
//! #     fn predict_with_uncertainty(&self, x: &Array2<f64>, c: f64) -> ferroml_core::Result<PredictionWithUncertainty> { let p = self.predict(x)?; Ok(PredictionWithUncertainty { predictions: p.clone(), lower: p.clone(), upper: p, confidence_level: c, std_errors: None }) }
//! # }
//! # impl Estimator for MyEstimator { type Fitted = MyFitted; fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> ferroml_core::Result<MyFitted> { Ok(MyFitted(y.mean().unwrap_or(0.0))) } fn search_space(&self) -> ferroml_core::hpo::SearchSpace { ferroml_core::hpo::SearchSpace::new() } }
//! # struct MseMetric;
//! # impl Metric for MseMetric { fn name(&self) -> &str { "mse" } fn direction(&self) -> ferroml_core::metrics::Direction { ferroml_core::metrics::Direction::Minimize } fn compute(&self, a: &Array1<f64>, b: &Array1<f64>) -> ferroml_core::Result<MetricValue> { let mse = a.iter().zip(b.iter()).map(|(x,y)|(x-y).powi(2)).sum::<f64>()/a.len() as f64; Ok(MetricValue::new("mse", mse, ferroml_core::metrics::Direction::Minimize)) } }
//! # let x = Array2::from_shape_vec((30, 2), (0..60).map(|i| i as f64).collect()).unwrap();
//! # let y = Array1::from_vec((0..30).map(|i| i as f64).collect());
//! # let metric = MseMetric;
//! use ferroml_core::cv::{nested_cv_score, KFold, NestedCVConfig};
//! use ferroml_core::hpo::SearchSpace;
//!
//! // Define search space
//! let search_space = SearchSpace::new()
//!     .float("alpha", 0.001, 10.0);
//!
//! // Factory function to create estimator from hyperparameters
//! let estimator_factory = |params: &HashMap<String, ParameterValue>| {
//!     let alpha = params.get("alpha").and_then(|p| p.as_f64()).unwrap_or(1.0);
//!     MyEstimator::new(alpha)
//! };
//!
//! let outer_cv = KFold::new(3);
//! let inner_cv = KFold::new(2);
//! let config = NestedCVConfig::default().with_n_trials(3).with_random_sampler().with_seed(42);
//!
//! let result = nested_cv_score(
//!     estimator_factory,
//!     &x, &y,
//!     &outer_cv, &inner_cv,
//!     &metric,
//!     &search_space,
//!     Direction::Minimize,
//!     &config,
//!     None,
//! )?;
//!
//! println!("Unbiased score: {}", result.summary());
//! # Ok(())
//! # }
//! ```

use crate::hpo::{Direction, ParameterValue, RandomSampler, Sampler, SearchSpace, TPESampler};
use crate::metrics::Metric;
use crate::traits::{Estimator, Predictor};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::{compute_t_confidence_interval, select_elements, select_rows};
use super::{CVFold, CVFoldResult, CVResult, CrossValidator};

/// Configuration for nested cross-validation
#[derive(Debug, Clone)]
pub struct NestedCVConfig {
    /// Number of HPO trials in the inner loop
    pub n_trials: usize,
    /// Whether to use TPE sampler (true) or random sampler (false)
    pub use_tpe: bool,
    /// Confidence level for intervals (default 0.95)
    pub confidence_level: f64,
    /// Whether to return training scores
    pub return_train_score: bool,
    /// Number of parallel jobs for outer loop (-1 for all CPUs)
    pub n_jobs: i32,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to refit on full outer training data after HPO
    pub refit_on_full_train: bool,
    /// Verbosity level
    pub verbose: u8,
}

impl Default for NestedCVConfig {
    fn default() -> Self {
        Self {
            n_trials: 50,
            use_tpe: true,
            confidence_level: 0.95,
            return_train_score: false,
            n_jobs: 1,
            seed: None,
            refit_on_full_train: true,
            verbose: 0,
        }
    }
}

impl NestedCVConfig {
    /// Set the number of HPO trials
    pub fn with_n_trials(mut self, n_trials: usize) -> Self {
        self.n_trials = n_trials;
        self
    }

    /// Use random sampler instead of TPE
    pub fn with_random_sampler(mut self) -> Self {
        self.use_tpe = false;
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence_level = confidence;
        self
    }

    /// Enable train score computation
    pub fn with_train_score(mut self) -> Self {
        self.return_train_score = true;
        self
    }

    /// Set number of parallel jobs
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Disable refitting on full training data
    pub fn without_refit(mut self) -> Self {
        self.refit_on_full_train = false;
        self
    }
}

/// Result from a single outer fold of nested CV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedCVFoldResult {
    /// Outer fold index
    pub outer_fold_index: usize,
    /// Test score on outer fold (unbiased)
    pub test_score: f64,
    /// Train score on outer fold (if computed)
    pub train_score: Option<f64>,
    /// Best hyperparameters found in inner loop
    pub best_params: HashMap<String, ParameterValue>,
    /// Best inner CV score
    pub best_inner_score: f64,
    /// Number of HPO trials completed
    pub n_trials_completed: usize,
    /// Time for HPO in seconds
    pub hpo_time_secs: f64,
    /// Time for final evaluation in seconds
    pub eval_time_secs: f64,
}

/// Aggregated results from nested cross-validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedCVResult {
    /// Results from each outer fold
    pub fold_results: Vec<NestedCVFoldResult>,
    /// Mean test score across outer folds (unbiased estimate)
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
    /// Confidence level used
    pub confidence_level: f64,
    /// Number of outer folds
    pub n_outer_folds: usize,
    /// Total samples
    pub n_samples: usize,
    /// Mean inner CV score (shows potential optimistic bias if much higher than outer)
    pub mean_inner_score: f64,
    /// Optimism estimate (difference between inner and outer scores)
    pub optimism_estimate: f64,
}

impl NestedCVResult {
    /// Create from fold results
    pub fn from_fold_results(
        fold_results: Vec<NestedCVFoldResult>,
        n_samples: usize,
        confidence_level: f64,
    ) -> Self {
        let n_outer_folds = fold_results.len();
        let test_scores: Vec<f64> = fold_results.iter().map(|r| r.test_score).collect();
        let inner_scores: Vec<f64> = fold_results.iter().map(|r| r.best_inner_score).collect();

        // Compute mean and std of test scores
        let mean_test_score = test_scores.iter().sum::<f64>() / n_outer_folds as f64;
        let std_test_score = if n_outer_folds > 1 {
            let variance = test_scores
                .iter()
                .map(|s| (s - mean_test_score).powi(2))
                .sum::<f64>()
                / (n_outer_folds - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Compute train score stats if available
        let train_scores: Vec<f64> = fold_results.iter().filter_map(|r| r.train_score).collect();

        let (mean_train_score, std_train_score) = if train_scores.len() == n_outer_folds {
            let mean = train_scores.iter().sum::<f64>() / n_outer_folds as f64;
            let std = if n_outer_folds > 1 {
                let variance = train_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                    / (n_outer_folds - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };
            (Some(mean), Some(std))
        } else {
            (None, None)
        };

        // Compute confidence interval using t-distribution
        let (ci_lower, ci_upper) = compute_t_confidence_interval(
            mean_test_score,
            std_test_score,
            n_outer_folds,
            confidence_level,
        );

        // Compute inner scores mean
        let mean_inner_score = inner_scores.iter().sum::<f64>() / n_outer_folds as f64;

        // Optimism = inner score - outer score
        // Positive optimism means inner loop overestimates performance
        let optimism_estimate = mean_inner_score - mean_test_score;

        Self {
            fold_results,
            mean_test_score,
            std_test_score,
            mean_train_score,
            std_train_score,
            ci_lower,
            ci_upper,
            confidence_level,
            n_outer_folds,
            n_samples,
            mean_inner_score,
            optimism_estimate,
        }
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Nested CV: {:.4} +/- {:.4} ({}% CI: [{:.4}, {:.4}], n_folds={})\n\
             Inner CV mean: {:.4}, Optimism: {:.4}",
            self.mean_test_score,
            self.std_test_score,
            (self.confidence_level * 100.0) as i32,
            self.ci_lower,
            self.ci_upper,
            self.n_outer_folds,
            self.mean_inner_score,
            self.optimism_estimate
        )
    }

    /// Check if optimism is concerning (inner much better than outer)
    ///
    /// Returns true if inner scores are significantly higher than outer scores,
    /// which indicates potential overfitting in the HPO process.
    pub fn has_concerning_optimism(&self, threshold: f64) -> bool {
        self.optimism_estimate.abs() > threshold
    }
}

/// Run nested cross-validation for unbiased performance estimation with HPO.
///
/// This function performs hyperparameter optimization in the inner loop and
/// model evaluation in the outer loop, preventing data leakage.
///
/// # Arguments
///
/// * `estimator_factory` - Function that creates an estimator from hyperparameters
/// * `x` - Feature matrix of shape (n_samples, n_features)
/// * `y` - Target vector of length n_samples
/// * `outer_cv` - Cross-validation strategy for outer loop (evaluation)
/// * `inner_cv` - Cross-validation strategy for inner loop (HPO)
/// * `metric` - Scoring metric to use
/// * `search_space` - Hyperparameter search space
/// * `direction` - Optimization direction (minimize or maximize)
/// * `config` - Nested CV configuration
/// * `groups` - Optional group labels for group-based CV
///
/// # Returns
///
/// `NestedCVResult` containing unbiased scores and best params per fold.
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// # use ferroml_core::traits::{Estimator, Predictor, PredictionWithUncertainty};
/// # use ferroml_core::metrics::{Metric, MetricValue};
/// # use ferroml_core::hpo::{SearchSpace, ParameterValue, Direction};
/// # use ferroml_core::cv::{nested_cv_score, KFold, NestedCVConfig};
/// # use ndarray::{Array1, Array2};
/// # use std::collections::HashMap;
/// # #[derive(Clone)]
/// # struct RidgeRegression { alpha: f64 }
/// # impl RidgeRegression { fn new(alpha: f64) -> Self { Self { alpha } } }
/// # struct RidgeFitted(f64);
/// # impl Predictor for RidgeFitted {
/// #     fn predict(&self, x: &Array2<f64>) -> ferroml_core::Result<Array1<f64>> { Ok(Array1::from_elem(x.nrows(), self.0)) }
/// #     fn predict_with_uncertainty(&self, x: &Array2<f64>, c: f64) -> ferroml_core::Result<PredictionWithUncertainty> { let p = self.predict(x)?; Ok(PredictionWithUncertainty { predictions: p.clone(), lower: p.clone(), upper: p, confidence_level: c, std_errors: None }) }
/// # }
/// # impl Estimator for RidgeRegression { type Fitted = RidgeFitted; fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> ferroml_core::Result<RidgeFitted> { Ok(RidgeFitted(y.mean().unwrap_or(0.0))) } fn search_space(&self) -> SearchSpace { SearchSpace::new() } }
/// # struct MseMetric;
/// # impl Metric for MseMetric { fn name(&self) -> &str { "mse" } fn direction(&self) -> ferroml_core::metrics::Direction { ferroml_core::metrics::Direction::Minimize } fn compute(&self, a: &Array1<f64>, b: &Array1<f64>) -> ferroml_core::Result<MetricValue> { let mse = a.iter().zip(b.iter()).map(|(x,y)|(x-y).powi(2)).sum::<f64>()/a.len() as f64; Ok(MetricValue::new("mse", mse, ferroml_core::metrics::Direction::Minimize)) } }
/// # let x = Array2::from_shape_vec((30, 2), (0..60).map(|i| i as f64).collect()).unwrap();
/// # let y = Array1::from_vec((0..30).map(|i| i as f64).collect());
/// # let search_space = SearchSpace::new().float("alpha", 0.01, 1.0);
/// let result = nested_cv_score(
///     |params: &HashMap<String, ParameterValue>| {
///         let alpha = params.get("alpha").and_then(|p| p.as_f64()).unwrap_or(1.0);
///         RidgeRegression::new(alpha)
///     },
///     &x, &y,
///     &KFold::new(3),
///     &KFold::new(2),
///     &MseMetric,
///     &search_space,
///     Direction::Minimize,
///     &NestedCVConfig::default().with_n_trials(3).with_random_sampler().with_seed(42),
///     None,
/// )?;
/// # Ok(())
/// # }
/// ```
pub fn nested_cv_score<F, E, M>(
    estimator_factory: F,
    x: &Array2<f64>,
    y: &Array1<f64>,
    outer_cv: &dyn CrossValidator,
    inner_cv: &dyn CrossValidator,
    metric: &M,
    search_space: &SearchSpace,
    direction: Direction,
    config: &NestedCVConfig,
    groups: Option<&Array1<i64>>,
) -> Result<NestedCVResult>
where
    F: Fn(&HashMap<String, ParameterValue>) -> E + Sync,
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

    if config.n_trials == 0 {
        return Err(FerroError::invalid_input(
            "n_trials must be at least 1 for nested CV",
        ));
    }

    // Get outer CV splits
    let outer_folds = outer_cv.split(n_samples, Some(y), groups)?;
    let n_outer_folds = outer_folds.len();

    if n_outer_folds == 0 {
        return Err(FerroError::invalid_input(
            "Outer cross-validator produced no folds",
        ));
    }

    // Determine parallelism
    let use_parallel = config.n_jobs != 1 && n_outer_folds > 1;

    // Evaluate outer folds
    let fold_results: Vec<Result<NestedCVFoldResult>> = if use_parallel {
        outer_folds
            .into_par_iter()
            .enumerate()
            .map(|(fold_idx, outer_fold)| {
                evaluate_nested_fold(
                    &estimator_factory,
                    x,
                    y,
                    &outer_fold,
                    fold_idx,
                    inner_cv,
                    metric,
                    search_space,
                    direction,
                    config,
                )
            })
            .collect()
    } else {
        outer_folds
            .into_iter()
            .enumerate()
            .map(|(fold_idx, outer_fold)| {
                evaluate_nested_fold(
                    &estimator_factory,
                    x,
                    y,
                    &outer_fold,
                    fold_idx,
                    inner_cv,
                    metric,
                    search_space,
                    direction,
                    config,
                )
            })
            .collect()
    };

    // Check for errors and collect successful results
    let mut successful_results = Vec::with_capacity(n_outer_folds);
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
            .map(|(idx, e)| format!("Outer fold {}: {}", idx, e))
            .collect();
        return Err(FerroError::cross_validation(format!(
            "All outer folds failed: {}",
            error_msgs.join("; ")
        )));
    }

    Ok(NestedCVResult::from_fold_results(
        successful_results,
        n_samples,
        config.confidence_level,
    ))
}

/// Evaluate a single outer fold of nested cross-validation
fn evaluate_nested_fold<F, E, M>(
    estimator_factory: &F,
    x: &Array2<f64>,
    y: &Array1<f64>,
    outer_fold: &CVFold,
    outer_fold_index: usize,
    inner_cv: &dyn CrossValidator,
    metric: &M,
    search_space: &SearchSpace,
    direction: Direction,
    config: &NestedCVConfig,
) -> Result<NestedCVFoldResult>
where
    F: Fn(&HashMap<String, ParameterValue>) -> E,
    E: Estimator + Clone,
    M: Metric,
{
    // Split outer fold data
    let x_outer_train = select_rows(x, &outer_fold.train_indices);
    let y_outer_train = select_elements(y, &outer_fold.train_indices);
    let x_outer_test = select_rows(x, &outer_fold.test_indices);
    let y_outer_test = select_elements(y, &outer_fold.test_indices);

    // Create sampler with seed based on outer fold index for reproducibility
    let seed = config.seed.unwrap_or(42) + outer_fold_index as u64;
    let sampler: Box<dyn Sampler> = if config.use_tpe {
        Box::new(TPESampler::new().with_seed(seed))
    } else {
        Box::new(RandomSampler::with_seed(seed))
    };

    // Inner loop: HPO on outer training data
    let hpo_start = Instant::now();

    let inner_result = run_inner_hpo(
        estimator_factory,
        &x_outer_train,
        &y_outer_train,
        inner_cv,
        metric,
        search_space,
        direction,
        sampler.as_ref(),
        config.n_trials,
    )?;

    let hpo_time_secs = hpo_start.elapsed().as_secs_f64();

    // Create best estimator with optimized hyperparameters
    let best_estimator = estimator_factory(&inner_result.best_params);

    // Evaluate on outer test fold
    let eval_start = Instant::now();

    // Fit on full outer training data (or just use inner best if not refitting)
    let fitted = best_estimator.fit(&x_outer_train, &y_outer_train)?;

    // Score on outer test set
    let y_pred_test = fitted.predict(&x_outer_test)?;
    let test_metric = metric.compute(&y_outer_test, &y_pred_test)?;
    let test_score = test_metric.value;

    // Optionally compute train score
    let train_score = if config.return_train_score {
        let y_pred_train = fitted.predict(&x_outer_train)?;
        let train_metric = metric.compute(&y_outer_train, &y_pred_train)?;
        Some(train_metric.value)
    } else {
        None
    };

    let eval_time_secs = eval_start.elapsed().as_secs_f64();

    Ok(NestedCVFoldResult {
        outer_fold_index,
        test_score,
        train_score,
        best_params: inner_result.best_params,
        best_inner_score: inner_result.best_score,
        n_trials_completed: inner_result.n_trials,
        hpo_time_secs,
        eval_time_secs,
    })
}

/// Result from inner HPO loop
struct InnerHPOResult {
    best_params: HashMap<String, ParameterValue>,
    best_score: f64,
    n_trials: usize,
}

/// Run the inner HPO loop
fn run_inner_hpo<F, E, M>(
    estimator_factory: &F,
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    inner_cv: &dyn CrossValidator,
    metric: &M,
    search_space: &SearchSpace,
    direction: Direction,
    sampler: &dyn Sampler,
    n_trials: usize,
) -> Result<InnerHPOResult>
where
    F: Fn(&HashMap<String, ParameterValue>) -> E,
    E: Estimator + Clone,
    M: Metric,
{
    let mut trials: Vec<(HashMap<String, ParameterValue>, f64)> = Vec::new();
    let mut best_score: Option<f64> = None;
    let mut best_params: Option<HashMap<String, ParameterValue>> = None;

    for _ in 0..n_trials {
        // Sample hyperparameters
        let params = sampler.sample(
            search_space,
            &trials
                .iter()
                .enumerate()
                .map(|(id, (p, v))| crate::hpo::Trial {
                    id,
                    params: p.clone(),
                    value: Some(*v),
                    state: crate::hpo::TrialState::Complete,
                    intermediate_values: Vec::new(),
                    duration: None,
                })
                .collect::<Vec<_>>(),
        )?;

        // Create estimator with these hyperparameters
        let estimator = estimator_factory(&params);

        // Run inner CV
        let cv_result = run_inner_cv(&estimator, x_train, y_train, inner_cv, metric)?;

        let score = cv_result.mean_test_score;
        trials.push((params.clone(), score));

        // Update best
        let is_better = match (best_score, direction) {
            (None, _) => true,
            (Some(best), Direction::Minimize) => score < best,
            (Some(best), Direction::Maximize) => score > best,
        };

        if is_better {
            best_score = Some(score);
            best_params = Some(params);
        }
    }

    // Should always have at least one trial
    let best_params =
        best_params.ok_or_else(|| FerroError::cross_validation("No successful HPO trials"))?;
    let best_score =
        best_score.ok_or_else(|| FerroError::cross_validation("No successful HPO trials"))?;

    Ok(InnerHPOResult {
        best_params,
        best_score,
        n_trials: trials.len(),
    })
}

/// Run inner CV for a single set of hyperparameters
fn run_inner_cv<E, M>(
    estimator: &E,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &dyn CrossValidator,
    metric: &M,
) -> Result<CVResult>
where
    E: Estimator + Clone,
    M: Metric,
{
    let n_samples = x.nrows();
    let folds = cv.split(n_samples, Some(y), None)?;

    if folds.is_empty() {
        return Err(FerroError::cross_validation("Inner CV produced no folds"));
    }

    let mut fold_results = Vec::with_capacity(folds.len());

    for fold in folds {
        let x_train = select_rows(x, &fold.train_indices);
        let y_train = select_elements(y, &fold.train_indices);
        let x_test = select_rows(x, &fold.test_indices);
        let y_test = select_elements(y, &fold.test_indices);

        let fit_start = Instant::now();
        let fitted = estimator.clone().fit(&x_train, &y_train)?;
        let fit_time_secs = fit_start.elapsed().as_secs_f64();

        let score_start = Instant::now();
        let y_pred = fitted.predict(&x_test)?;
        let metric_value = metric.compute(&y_test, &y_pred)?;
        let score_time_secs = score_start.elapsed().as_secs_f64();

        fold_results.push(CVFoldResult {
            fold_index: fold.fold_index,
            train_score: None,
            test_score: metric_value.value,
            fit_time_secs,
            score_time_secs,
        });
    }

    Ok(CVResult::from_fold_results(fold_results, n_samples, 0.95))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpo::SearchSpace;
    use crate::metrics::{Direction as MetricDirection, MetricValue};
    use ndarray::{Array1, Array2};

    /// A simple mock estimator for testing
    #[derive(Clone)]
    struct MockEstimator {
        alpha: f64,
    }

    impl MockEstimator {
        fn new(alpha: f64) -> Self {
            Self { alpha }
        }
    }

    struct MockPredictor {
        mean: f64,
        alpha: f64,
    }

    impl crate::traits::Predictor for MockPredictor {
        fn predict(&self, x: &Array2<f64>) -> crate::Result<Array1<f64>> {
            // Prediction is mean adjusted by alpha (simulating regularization effect)
            Ok(Array1::from_vec(vec![
                self.mean * (1.0 - self.alpha * 0.01);
                x.nrows()
            ]))
        }

        fn predict_with_uncertainty(
            &self,
            x: &Array2<f64>,
            confidence: f64,
        ) -> crate::Result<crate::traits::PredictionWithUncertainty> {
            let predictions = self.predict(x)?;
            let n = predictions.len();
            Ok(crate::traits::PredictionWithUncertainty {
                predictions: predictions.clone(),
                lower: Array1::from_vec(vec![self.mean - 0.1; n]),
                upper: Array1::from_vec(vec![self.mean + 0.1; n]),
                confidence_level: confidence,
                std_errors: None,
            })
        }
    }

    impl crate::traits::Estimator for MockEstimator {
        type Fitted = MockPredictor;

        fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> crate::Result<Self::Fitted> {
            let mean = y.iter().sum::<f64>() / y.len() as f64;
            Ok(MockPredictor {
                mean,
                alpha: self.alpha,
            })
        }

        fn search_space(&self) -> SearchSpace {
            SearchSpace::new()
        }
    }

    /// Mock MSE metric
    struct MockMSE;

    impl Metric for MockMSE {
        fn name(&self) -> &str {
            "mse"
        }

        fn direction(&self) -> MetricDirection {
            MetricDirection::Minimize
        }

        fn compute(
            &self,
            y_true: &Array1<f64>,
            y_pred: &Array1<f64>,
        ) -> crate::Result<MetricValue> {
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
    fn test_nested_cv_config_builder() {
        let config = NestedCVConfig::default()
            .with_n_trials(100)
            .with_random_sampler()
            .with_confidence(0.99)
            .with_train_score()
            .with_n_jobs(-1)
            .with_seed(42);

        assert_eq!(config.n_trials, 100);
        assert!(!config.use_tpe);
        assert!((config.confidence_level - 0.99).abs() < 1e-10);
        assert!(config.return_train_score);
        assert_eq!(config.n_jobs, -1);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_nested_cv_result_summary() {
        let fold_results = vec![
            NestedCVFoldResult {
                outer_fold_index: 0,
                test_score: 0.10,
                train_score: Some(0.05),
                best_params: HashMap::new(),
                best_inner_score: 0.08,
                n_trials_completed: 10,
                hpo_time_secs: 1.0,
                eval_time_secs: 0.1,
            },
            NestedCVFoldResult {
                outer_fold_index: 1,
                test_score: 0.12,
                train_score: Some(0.06),
                best_params: HashMap::new(),
                best_inner_score: 0.09,
                n_trials_completed: 10,
                hpo_time_secs: 1.1,
                eval_time_secs: 0.1,
            },
        ];

        let result = NestedCVResult::from_fold_results(fold_results, 100, 0.95);

        assert_eq!(result.n_outer_folds, 2);
        assert!((result.mean_test_score - 0.11).abs() < 0.01);
        assert!(result.mean_train_score.is_some());

        // Inner score should be lower than outer (better) for MSE
        // This shows optimism - inner loop is optimistic
        assert!(result.optimism_estimate < 0.0); // inner < outer for MSE

        let summary = result.summary();
        assert!(summary.contains("Nested CV"));
        assert!(summary.contains("Optimism"));
    }

    #[test]
    fn test_nested_cv_score_basic() {
        use crate::cv::KFold;

        // Create simple data
        let n_samples = 50;
        let x = Array2::from_shape_vec(
            (n_samples, 2),
            (0..n_samples * 2).map(|i| i as f64 / 10.0).collect(),
        )
        .unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        // Define search space
        let search_space = SearchSpace::new().float("alpha", 0.01, 1.0);

        // Factory function
        let estimator_factory = |params: &HashMap<String, ParameterValue>| {
            let alpha = params.get("alpha").and_then(|p| p.as_f64()).unwrap_or(0.5);
            MockEstimator::new(alpha)
        };

        let outer_cv = KFold::new(3);
        let inner_cv = KFold::new(2);
        let config = NestedCVConfig::default()
            .with_n_trials(5)
            .with_random_sampler()
            .with_seed(42);

        let result = nested_cv_score(
            estimator_factory,
            &x,
            &y,
            &outer_cv,
            &inner_cv,
            &MockMSE,
            &search_space,
            Direction::Minimize,
            &config,
            None,
        )
        .unwrap();

        // Should have results for all outer folds
        assert_eq!(result.n_outer_folds, 3);
        assert_eq!(result.fold_results.len(), 3);

        // Scores should be finite
        assert!(result.mean_test_score.is_finite());
        assert!(result.std_test_score.is_finite());
        assert!(result.mean_inner_score.is_finite());

        // CI should bracket mean
        assert!(result.ci_lower <= result.mean_test_score);
        assert!(result.ci_upper >= result.mean_test_score);

        // Each fold should have best params
        for fold_result in &result.fold_results {
            assert!(fold_result.n_trials_completed >= 1);
        }
    }

    #[test]
    fn test_nested_cv_with_train_scores() {
        use crate::cv::KFold;

        let n_samples = 40;
        let x = Array2::from_shape_vec(
            (n_samples, 2),
            (0..n_samples * 2).map(|i| i as f64).collect(),
        )
        .unwrap();
        let y = Array1::from_vec((0..n_samples).map(|i| i as f64).collect());

        let search_space = SearchSpace::new().float("alpha", 0.01, 1.0);

        let estimator_factory = |params: &HashMap<String, ParameterValue>| {
            let alpha = params.get("alpha").and_then(|p| p.as_f64()).unwrap_or(0.5);
            MockEstimator::new(alpha)
        };

        let config = NestedCVConfig::default()
            .with_n_trials(3)
            .with_train_score()
            .with_seed(42);

        let result = nested_cv_score(
            estimator_factory,
            &x,
            &y,
            &KFold::new(2),
            &KFold::new(2),
            &MockMSE,
            &search_space,
            Direction::Minimize,
            &config,
            None,
        )
        .unwrap();

        // Should have train scores
        assert!(result.mean_train_score.is_some());
        for fold_result in &result.fold_results {
            assert!(fold_result.train_score.is_some());
        }
    }

    #[test]
    fn test_nested_cv_error_handling() {
        use crate::cv::KFold;

        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec(vec![1.0; 5]); // Wrong length

        let search_space = SearchSpace::new();
        let estimator_factory = |_: &HashMap<String, ParameterValue>| MockEstimator::new(0.5);

        let result = nested_cv_score(
            estimator_factory,
            &x,
            &y,
            &KFold::new(2),
            &KFold::new(2),
            &MockMSE,
            &search_space,
            Direction::Minimize,
            &NestedCVConfig::default(),
            None,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_nested_cv_optimism_detection() {
        let fold_results = vec![
            NestedCVFoldResult {
                outer_fold_index: 0,
                test_score: 0.20, // Worse (higher MSE)
                train_score: None,
                best_params: HashMap::new(),
                best_inner_score: 0.10, // Better (lower MSE)
                n_trials_completed: 10,
                hpo_time_secs: 1.0,
                eval_time_secs: 0.1,
            },
            NestedCVFoldResult {
                outer_fold_index: 1,
                test_score: 0.22,
                train_score: None,
                best_params: HashMap::new(),
                best_inner_score: 0.11,
                n_trials_completed: 10,
                hpo_time_secs: 1.0,
                eval_time_secs: 0.1,
            },
        ];

        let result = NestedCVResult::from_fold_results(fold_results, 100, 0.95);

        // For MSE (minimize), inner score being lower (better) means optimism is negative
        // optimism = inner - outer = 0.105 - 0.21 = -0.105
        assert!(result.optimism_estimate < -0.05);

        // This is concerning - inner loop appears to overfit
        assert!(result.has_concerning_optimism(0.05));
    }
}
