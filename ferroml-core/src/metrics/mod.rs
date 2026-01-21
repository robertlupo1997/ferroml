//! Evaluation Metrics for Machine Learning
//!
//! This module provides comprehensive metrics for evaluating ML models with
//! statistical rigor. Every metric supports confidence intervals and can be
//! used in model comparison tests.
//!
//! ## Philosophy
//!
//! - **CI by default**: All metrics can compute bootstrap confidence intervals
//! - **Proper averaging**: Macro, micro, and weighted averaging for multiclass
//! - **Model comparison**: Statistical tests for comparing models (McNemar, Nadeau-Bengio)
//! - **Calibration aware**: Metrics like Brier score and log loss for probabilistic models
//!
//! ## Modules
//!
//! - `classification`: Accuracy, precision, recall, F1, ROC-AUC, etc.
//! - `regression`: MSE, RMSE, MAE, R², etc.
//! - `comparison`: Statistical tests for model comparison

pub mod classification;
pub mod comparison;
pub mod probabilistic;
pub mod regression;

// Re-exports
pub use classification::{
    accuracy, balanced_accuracy, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision, recall, BalancedAccuracyMetric, ClassificationReport, CohenKappaMetric,
    ConfusionMatrix, MatthewsCorrCoefMetric,
};
pub use comparison::{
    corrected_resampled_ttest, mcnemar_test, paired_ttest, ModelComparisonResult,
};
pub use probabilistic::{
    average_precision_score, brier_score, brier_skill_score, log_loss, pr_auc_score, roc_auc_score,
    roc_auc_with_ci, AveragePrecisionMetric, BrierScoreMetric, LogLossMetric, PrAucMetric, PrCurve,
    RocAucMetric, RocCurve,
};
pub use regression::{explained_variance, mae, mse, r2_score, rmse, RegressionMetrics};

use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Averaging strategy for multiclass metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Average {
    /// Calculate metric for each class and average (unweighted)
    Macro,
    /// Calculate metric globally by counting total TP, FP, FN
    #[default]
    Micro,
    /// Calculate metric for each class and average weighted by support
    Weighted,
    /// Return metrics for each class without averaging
    None,
}

/// Direction of metric optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// Higher is better (e.g., accuracy, R²)
    Maximize,
    /// Lower is better (e.g., MSE, log loss)
    Minimize,
}

/// Core trait for all evaluation metrics
///
/// Metrics compute scores from true labels and predictions, optionally
/// with confidence intervals via bootstrapping.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::metrics::{Metric, MetricValue};
///
/// struct Accuracy;
///
/// impl Metric for Accuracy {
///     fn name(&self) -> &str { "accuracy" }
///     fn direction(&self) -> Direction { Direction::Maximize }
///
///     fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
///         // Implementation
///     }
/// }
/// ```
pub trait Metric: Send + Sync {
    /// Name of the metric for display and logging
    fn name(&self) -> &str;

    /// Whether higher or lower values are better
    fn direction(&self) -> Direction;

    /// Compute the metric from true labels and predictions
    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue>;

    /// Compute metric with confidence interval via bootstrap
    ///
    /// # Arguments
    /// * `y_true` - True labels
    /// * `y_pred` - Predicted values
    /// * `confidence` - Confidence level (e.g., 0.95)
    /// * `n_bootstrap` - Number of bootstrap samples
    /// * `seed` - Random seed for reproducibility
    fn compute_with_ci(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        confidence: f64,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> Result<MetricValueWithCI>
    where
        Self: Sized,
    {
        // Default implementation using bootstrap
        let base_value = self.compute(y_true, y_pred)?;

        let bootstrap_scores = bootstrap_metric(self, y_true, y_pred, n_bootstrap, seed)?;

        let ci = compute_percentile_ci(&bootstrap_scores, confidence);
        let std_error = standard_error(&bootstrap_scores);

        Ok(MetricValueWithCI {
            value: base_value.value,
            ci_lower: ci.0,
            ci_upper: ci.1,
            confidence_level: confidence,
            std_error,
            n_bootstrap,
        })
    }

    /// Check if this metric requires probability predictions
    fn requires_probabilities(&self) -> bool {
        false
    }
}

/// Trait for metrics that use probability predictions
pub trait ProbabilisticMetric: Metric {
    /// Compute metric from true labels and predicted probabilities
    fn compute_proba(&self, y_true: &Array1<f64>, y_proba: &Array2<f64>) -> Result<MetricValue>;

    /// Compute with confidence interval for probabilistic metrics
    fn compute_proba_with_ci(
        &self,
        y_true: &Array1<f64>,
        y_proba: &Array2<f64>,
        confidence: f64,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> Result<MetricValueWithCI>;
}

/// Simple metric value without confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    /// The metric name
    pub name: String,
    /// The computed value
    pub value: f64,
    /// Whether higher is better
    pub direction: Direction,
}

impl MetricValue {
    /// Create a new metric value
    pub fn new(name: impl Into<String>, value: f64, direction: Direction) -> Self {
        Self {
            name: name.into(),
            value,
            direction,
        }
    }

    /// Check if this value is better than another
    pub fn is_better_than(&self, other: &MetricValue) -> bool {
        match self.direction {
            Direction::Maximize => self.value > other.value,
            Direction::Minimize => self.value < other.value,
        }
    }
}

/// Metric value with confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValueWithCI {
    /// The computed value
    pub value: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Confidence level used (e.g., 0.95)
    pub confidence_level: f64,
    /// Standard error from bootstrap
    pub std_error: f64,
    /// Number of bootstrap samples used
    pub n_bootstrap: usize,
}

impl MetricValueWithCI {
    /// Format as a human-readable string
    pub fn summary(&self) -> String {
        format!(
            "{:.4} ({}% CI: [{:.4}, {:.4}])",
            self.value,
            (self.confidence_level * 100.0) as i32,
            self.ci_lower,
            self.ci_upper
        )
    }

    /// Check if two metrics are significantly different (non-overlapping CIs)
    /// Note: This is a conservative test; overlapping CIs don't imply no difference
    pub fn significantly_different_from(&self, other: &MetricValueWithCI) -> bool {
        self.ci_upper < other.ci_lower || self.ci_lower > other.ci_upper
    }
}

/// Result of computing multiple metrics at once
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsBundle {
    /// All computed metrics
    pub metrics: Vec<MetricValue>,
    /// Sample size
    pub n_samples: usize,
}

impl MetricsBundle {
    /// Get a metric by name
    pub fn get(&self, name: &str) -> Option<&MetricValue> {
        self.metrics.iter().find(|m| m.name == name)
    }

    /// Get the best metric according to its direction
    pub fn best(&self) -> Option<&MetricValue> {
        self.metrics.iter().max_by(|a, b| {
            if a.is_better_than(b) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        })
    }
}

/// Bootstrap a metric to get distribution of scores
fn bootstrap_metric(
    metric: &dyn Metric,
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> Result<Vec<f64>> {
    use rand::prelude::*;
    use rand::SeedableRng;

    let n = y_true.len();
    if n == 0 {
        return Err(FerroError::invalid_input("Empty arrays"));
    }

    let mut rng: Box<dyn RngCore> = match seed {
        Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
        None => Box::new(rand::rng()),
    };

    let mut scores = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Sample with replacement
        let indices: Vec<usize> = (0..n).map(|_| rng.random_range(0..n)).collect();

        let y_true_sample = Array1::from_vec(indices.iter().map(|&i| y_true[i]).collect());
        let y_pred_sample = Array1::from_vec(indices.iter().map(|&i| y_pred[i]).collect());

        match metric.compute(&y_true_sample, &y_pred_sample) {
            Ok(value) => scores.push(value.value),
            Err(_) => continue, // Skip failed samples
        }
    }

    if scores.is_empty() {
        return Err(FerroError::numerical("All bootstrap samples failed"));
    }

    Ok(scores)
}

/// Compute percentile confidence interval from bootstrap samples
fn compute_percentile_ci(scores: &[f64], confidence: f64) -> (f64, f64) {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let n = sorted.len();

    let lower_idx = ((alpha / 2.0) * n as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;

    let lower_idx = lower_idx.min(n - 1);
    let upper_idx = upper_idx.min(n - 1);

    (sorted[lower_idx], sorted[upper_idx])
}

/// Compute standard error from bootstrap samples
fn standard_error(scores: &[f64]) -> f64 {
    let n = scores.len() as f64;
    if n <= 1.0 {
        return 0.0;
    }

    let mean = scores.iter().sum::<f64>() / n;
    let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// Validate that y_true and y_pred have the same length
pub(crate) fn validate_arrays(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    if y_true.len() != y_pred.len() {
        return Err(FerroError::shape_mismatch(
            format!("y_true length {}", y_true.len()),
            format!("y_pred length {}", y_pred.len()),
        ));
    }
    if y_true.is_empty() {
        return Err(FerroError::invalid_input("Empty arrays"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple accuracy metric for testing
    struct TestAccuracy;

    impl Metric for TestAccuracy {
        fn name(&self) -> &str {
            "test_accuracy"
        }

        fn direction(&self) -> Direction {
            Direction::Maximize
        }

        fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
            validate_arrays(y_true, y_pred)?;
            let correct = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&t, &p)| (t - p).abs() < 1e-10)
                .count();
            let acc = correct as f64 / y_true.len() as f64;
            Ok(MetricValue::new(self.name(), acc, self.direction()))
        }
    }

    #[test]
    fn test_metric_compute() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0]);

        let metric = TestAccuracy;
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "test_accuracy");
        assert!((result.value - 0.8).abs() < 1e-10);
        assert_eq!(result.direction, Direction::Maximize);
    }

    #[test]
    fn test_metric_with_ci() {
        let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
        let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);

        let metric = TestAccuracy;
        let result = metric
            .compute_with_ci(&y_true, &y_pred, 0.95, 100, Some(42))
            .unwrap();

        // Should be 0.8 accuracy (8/10 correct)
        assert!((result.value - 0.8).abs() < 1e-10);
        assert!(result.ci_lower <= result.value);
        assert!(result.ci_upper >= result.value);
        assert_eq!(result.confidence_level, 0.95);
    }

    #[test]
    fn test_metric_value_comparison() {
        let better = MetricValue::new("acc", 0.9, Direction::Maximize);
        let worse = MetricValue::new("acc", 0.8, Direction::Maximize);

        assert!(better.is_better_than(&worse));
        assert!(!worse.is_better_than(&better));

        // For minimize direction
        let better_mse = MetricValue::new("mse", 0.1, Direction::Minimize);
        let worse_mse = MetricValue::new("mse", 0.2, Direction::Minimize);

        assert!(better_mse.is_better_than(&worse_mse));
    }

    #[test]
    fn test_validate_arrays() {
        let y1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(validate_arrays(&y1, &y2).is_ok());

        let y3 = Array1::from_vec(vec![1.0, 2.0]);
        assert!(validate_arrays(&y1, &y3).is_err());

        let empty: Array1<f64> = Array1::from_vec(vec![]);
        assert!(validate_arrays(&empty, &empty).is_err());
    }

    #[test]
    fn test_percentile_ci() {
        let scores: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (lower, upper) = compute_percentile_ci(&scores, 0.95);

        // Should be approximately 2.5 and 97.5 percentiles
        assert!(lower < 5.0);
        assert!(upper > 95.0);
    }
}
