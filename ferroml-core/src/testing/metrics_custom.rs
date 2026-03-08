//! Phase 28: Custom Metrics Tests
//!
//! Tests for the Metric trait, built-in metric correctness, custom metric
//! implementations, metric integration with cross-validation and grid search,
//! multi-metric evaluation, edge cases, and metric direction semantics.

#![allow(unused_imports)]
#![allow(dead_code)]

use crate::cv::{
    cross_val_score_simple, search::GridSearchCV, CVConfig, CrossValidator, KFold,
    ParameterSettable,
};
use crate::hpo::SearchSpace;
use crate::metrics::classification::{accuracy, f1_score, precision, recall, AccuracyMetric};
use crate::metrics::regression::{mae, mse, r2_score, MaeMetric, MseMetric, R2Metric, RmseMetric};
use crate::metrics::{Direction, Metric, MetricValue, MetricValueWithCI, MetricsBundle};
use crate::traits::{Estimator, PredictionWithUncertainty, Predictor};
use crate::Result;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

// ============================================================================
// Custom Metric Implementation
// ============================================================================

/// A custom metric that computes mean absolute percentage error (MAPE).
struct MapeMetric;

impl Metric for MapeMetric {
    fn name(&self) -> &str {
        "mape"
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        crate::metrics::validate_arrays(y_true, y_pred)?;
        let n = y_true.len() as f64;
        let sum: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| {
                if t.abs() < 1e-15 {
                    0.0
                } else {
                    ((t - p) / t).abs()
                }
            })
            .sum();
        Ok(MetricValue::new("mape", sum / n, Direction::Minimize))
    }
}

/// A custom metric that counts exact matches (for classification).
struct ExactMatchMetric;

impl Metric for ExactMatchMetric {
    fn name(&self) -> &str {
        "exact_match"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        crate::metrics::validate_arrays(y_true, y_pred)?;
        let matches = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| (t - p).abs() < 1e-10)
            .count();
        Ok(MetricValue::new(
            "exact_match",
            matches as f64 / y_true.len() as f64,
            Direction::Maximize,
        ))
    }
}

/// Simple mean estimator for CV/search tests — always predicts the training mean.
#[derive(Clone)]
struct MeanEstimator {
    offset: f64,
}

struct FittedMean(f64);

impl Predictor for FittedMean {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(Array1::from_elem(x.nrows(), self.0))
    }

    fn predict_with_uncertainty(
        &self,
        x: &Array2<f64>,
        confidence: f64,
    ) -> Result<PredictionWithUncertainty> {
        let p = self.predict(x)?;
        Ok(PredictionWithUncertainty {
            predictions: p.clone(),
            lower: p.clone(),
            upper: p,
            confidence_level: confidence,
            std_errors: None,
        })
    }
}

impl Estimator for MeanEstimator {
    type Fitted = FittedMean;

    fn fit(&self, _x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        Ok(FittedMean(y.mean().unwrap_or(0.0) + self.offset))
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }
}

impl ParameterSettable for MeanEstimator {
    fn set_param(&mut self, name: &str, value: f64) -> Result<()> {
        match name {
            "offset" => {
                self.offset = value;
                Ok(())
            }
            _ => Err(crate::FerroError::invalid_input(format!(
                "Unknown param: {name}"
            ))),
        }
    }

    fn get_param(&self, name: &str) -> Result<f64> {
        match name {
            "offset" => Ok(self.offset),
            _ => Err(crate::FerroError::invalid_input(format!(
                "Unknown param: {name}"
            ))),
        }
    }
}

// ============================================================================
// Built-in Metric Correctness Tests
// ============================================================================

#[test]
fn test_accuracy_known_values() {
    let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0]);
    let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0]);
    let acc = accuracy(&y_true, &y_pred).unwrap();
    assert!((acc - 0.8).abs() < 1e-10, "Expected 0.8, got {acc}");
}

#[test]
fn test_mse_known_values() {
    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let y_pred = Array1::from_vec(vec![1.5, 2.5, 3.5, 4.5]);
    // Each error = 0.5, squared = 0.25, mean = 0.25
    let val = mse(&y_true, &y_pred).unwrap();
    assert!((val - 0.25).abs() < 1e-10, "Expected 0.25, got {val}");
}

#[test]
fn test_mae_known_values() {
    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let y_pred = Array1::from_vec(vec![1.5, 2.0, 2.0]);
    // Errors: 0.5, 0.0, 1.0 -> mean = 0.5
    let val = mae(&y_true, &y_pred).unwrap();
    assert!((val - 0.5).abs() < 1e-10, "Expected 0.5, got {val}");
}

#[test]
fn test_r2_known_values() {
    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let val = r2_score(&y_true, &y_pred).unwrap();
    assert!(
        (val - 1.0).abs() < 1e-10,
        "Perfect predictions should give R2=1.0"
    );
}

#[test]
fn test_r2_mean_predictor() {
    // Predicting the mean should give R2 = 0
    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mean = y_true.mean().unwrap();
    let y_pred = Array1::from_elem(5, mean);
    let val = r2_score(&y_true, &y_pred).unwrap();
    assert!(
        val.abs() < 1e-10,
        "Mean predictor should give R2~0, got {val}"
    );
}

#[test]
fn test_precision_recall_f1_binary() {
    // y_true: [1,1,1,0,0], y_pred: [1,1,0,0,1]
    // Class 0: TP=1, FP=1, FN=1 -> P=1/2, R=1/2, F1=1/2
    // Class 1: TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3
    // Macro: P=(1/2+2/3)/2=7/12, R=(1/2+2/3)/2=7/12, F1=(1/2+2/3)/2=7/12
    let y_true = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0]);
    let y_pred = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0, 1.0]);

    let p = precision(&y_true, &y_pred, crate::metrics::Average::Macro).unwrap();
    let r = recall(&y_true, &y_pred, crate::metrics::Average::Macro).unwrap();
    let f1 = f1_score(&y_true, &y_pred, crate::metrics::Average::Macro).unwrap();

    let expected = 7.0 / 12.0;
    assert!((p - expected).abs() < 1e-10, "Precision mismatch: {p}");
    assert!((r - expected).abs() < 1e-10, "Recall mismatch: {r}");
    assert!((f1 - expected).abs() < 1e-10, "F1 mismatch: {f1}");
}

// ============================================================================
// Built-in Metric Struct Tests (Metric trait impls)
// ============================================================================

#[test]
fn test_accuracy_metric_struct() {
    let metric = AccuracyMetric;
    assert_eq!(metric.name(), "accuracy");
    assert_eq!(metric.direction(), Direction::Maximize);

    let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
    let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let result = metric.compute(&y_true, &y_pred).unwrap();
    assert!((result.value - 0.75).abs() < 1e-10);
}

#[test]
fn test_mse_metric_struct() {
    let metric = MseMetric;
    assert_eq!(metric.name(), "mse");
    assert_eq!(metric.direction(), Direction::Minimize);

    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let result = metric.compute(&y_true, &y_pred).unwrap();
    assert!(
        result.value.abs() < 1e-10,
        "Perfect predictions should give MSE=0"
    );
}

#[test]
fn test_r2_metric_struct() {
    let metric = R2Metric;
    assert_eq!(metric.name(), "r2");
    assert_eq!(metric.direction(), Direction::Maximize);
}

#[test]
fn test_rmse_metric_struct() {
    let metric = RmseMetric;
    assert_eq!(metric.name(), "rmse");
    assert_eq!(metric.direction(), Direction::Minimize);

    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let y_pred = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
    let result = metric.compute(&y_true, &y_pred).unwrap();
    // Each error = 1.0, MSE = 1.0, RMSE = 1.0
    assert!((result.value - 1.0).abs() < 1e-10);
}

// ============================================================================
// Custom Metric Trait Implementation Tests
// ============================================================================

#[test]
fn test_custom_mape_metric() {
    let metric = MapeMetric;
    assert_eq!(metric.name(), "mape");
    assert_eq!(metric.direction(), Direction::Minimize);

    let y_true = Array1::from_vec(vec![100.0, 200.0, 300.0]);
    let y_pred = Array1::from_vec(vec![110.0, 190.0, 330.0]);
    // MAPE: |10/100| + |10/200| + |30/300| = 0.1 + 0.05 + 0.1 = 0.25 / 3 = 0.0833...
    let result = metric.compute(&y_true, &y_pred).unwrap();
    assert!(
        (result.value - 0.25 / 3.0).abs() < 1e-10,
        "Expected MAPE ~0.0833, got {}",
        result.value
    );
}

#[test]
fn test_custom_exact_match_metric() {
    let metric = ExactMatchMetric;
    assert_eq!(metric.direction(), Direction::Maximize);

    let y_true = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);
    let y_pred = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.0]);
    let result = metric.compute(&y_true, &y_pred).unwrap();
    assert!((result.value - 0.75).abs() < 1e-10);
}

#[test]
fn test_custom_metric_with_bootstrap_ci() {
    let metric = ExactMatchMetric;
    let y_true = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let y_pred = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);

    let result = metric
        .compute_with_ci(&y_true, &y_pred, 0.95, 200, Some(42))
        .unwrap();

    assert!((result.value - 0.8).abs() < 1e-10);
    assert!(result.ci_lower <= result.value);
    assert!(result.ci_upper >= result.value);
    assert!(result.std_error >= 0.0);
    assert_eq!(result.confidence_level, 0.95);
}

// ============================================================================
// MetricValue Direction and Comparison Tests
// ============================================================================

#[test]
fn test_metric_value_maximize_comparison() {
    let better = MetricValue::new("acc", 0.95, Direction::Maximize);
    let worse = MetricValue::new("acc", 0.80, Direction::Maximize);
    assert!(better.is_better_than(&worse));
    assert!(!worse.is_better_than(&better));
}

#[test]
fn test_metric_value_minimize_comparison() {
    let better = MetricValue::new("mse", 0.01, Direction::Minimize);
    let worse = MetricValue::new("mse", 0.50, Direction::Minimize);
    assert!(better.is_better_than(&worse));
    assert!(!worse.is_better_than(&better));
}

#[test]
fn test_metric_value_equal_not_better() {
    let a = MetricValue::new("acc", 0.9, Direction::Maximize);
    let b = MetricValue::new("acc", 0.9, Direction::Maximize);
    assert!(!a.is_better_than(&b));
    assert!(!b.is_better_than(&a));
}

// ============================================================================
// MetricsBundle Tests
// ============================================================================

#[test]
fn test_metrics_bundle_get_and_best() {
    let bundle = MetricsBundle {
        metrics: vec![
            MetricValue::new("accuracy", 0.9, Direction::Maximize),
            MetricValue::new("f1", 0.85, Direction::Maximize),
            MetricValue::new("mse", 0.1, Direction::Minimize),
        ],
        n_samples: 100,
    };

    let acc = bundle.get("accuracy").unwrap();
    assert!((acc.value - 0.9).abs() < 1e-10);

    assert!(bundle.get("nonexistent").is_none());

    // best() returns the metric that is_better_than all others — depends on direction mix
    let best = bundle.best().unwrap();
    assert!(best.value > 0.0);
}

// ============================================================================
// Multi-metric Evaluation on Same Predictions
// ============================================================================

#[test]
fn test_multi_metric_same_predictions() {
    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Array1::from_vec(vec![1.1, 2.2, 2.8, 4.1, 4.9]);

    let mse_val = mse(&y_true, &y_pred).unwrap();
    let mae_val = mae(&y_true, &y_pred).unwrap();
    let r2_val = r2_score(&y_true, &y_pred).unwrap();

    // MSE >= 0
    assert!(mse_val >= 0.0);
    // MAE >= 0
    assert!(mae_val >= 0.0);
    // MAE <= sqrt(MSE) * sqrt(n) by Cauchy-Schwarz, but simpler: MAE^2 <= MSE
    assert!(mae_val.powi(2) <= mse_val + 1e-10);
    // Good predictions => R2 close to 1
    assert!(
        r2_val > 0.9,
        "R2 should be high for close predictions, got {r2_val}"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_perfect_classification() {
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    let acc = accuracy(&y, &y).unwrap();
    assert!((acc - 1.0).abs() < 1e-10);
}

#[test]
fn test_all_wrong_classification() {
    let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let y_pred = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
    let acc = accuracy(&y_true, &y_pred).unwrap();
    assert!(acc.abs() < 1e-10, "All wrong should give accuracy=0");
}

#[test]
fn test_single_sample_metric() {
    let y_true = Array1::from_vec(vec![3.0]);
    let y_pred = Array1::from_vec(vec![3.5]);
    let mse_val = mse(&y_true, &y_pred).unwrap();
    assert!((mse_val - 0.25).abs() < 1e-10);
}

#[test]
fn test_constant_predictions_r2() {
    // If predictions are constant but y_true varies, R2 should be 0
    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mean = y_true.mean().unwrap();
    let y_pred = Array1::from_elem(5, mean);
    let r2 = r2_score(&y_true, &y_pred).unwrap();
    assert!(
        r2.abs() < 1e-10,
        "Constant mean prediction => R2=0, got {r2}"
    );
}

#[test]
fn test_metric_empty_arrays_error() {
    let empty: Array1<f64> = Array1::from_vec(vec![]);
    assert!(mse(&empty, &empty).is_err());
    assert!(accuracy(&empty, &empty).is_err());
}

#[test]
fn test_metric_mismatched_lengths_error() {
    let a = Array1::from_vec(vec![1.0, 2.0]);
    let b = Array1::from_vec(vec![1.0]);
    assert!(mse(&a, &b).is_err());
}

// ============================================================================
// Metrics in cross_val_score
// ============================================================================

#[test]
fn test_builtin_metric_in_cross_val_score() {
    let x = Array2::from_shape_fn((40, 2), |(i, j)| (i * 2 + j) as f64);
    let y = Array1::from_shape_fn(40, |i| i as f64);
    let cv = KFold::new(4);
    let metric = MseMetric;
    let estimator = MeanEstimator { offset: 0.0 };

    let result = cross_val_score_simple(&estimator, &x, &y, &cv, &metric).unwrap();
    // Mean estimator on regression data should have positive MSE
    assert!(result.mean_test_score > 0.0);
    assert_eq!(result.fold_results.len(), 4);
}

#[test]
fn test_custom_metric_in_cross_val_score() {
    let x = Array2::from_shape_fn((40, 2), |(i, j)| (i * 2 + j) as f64);
    let y = Array1::from_shape_fn(40, |i| i as f64 * 10.0);
    let cv = KFold::new(4);
    let metric = MapeMetric;
    let estimator = MeanEstimator { offset: 0.0 };

    let result = cross_val_score_simple(&estimator, &x, &y, &cv, &metric).unwrap();
    // MAPE should be positive for a mean estimator on non-constant data
    assert!(result.mean_test_score > 0.0);
}

// ============================================================================
// Metrics in GridSearchCV — metric direction respected
// ============================================================================

#[test]
fn test_grid_search_respects_minimize_direction() {
    let x = Array2::from_shape_fn((40, 2), |(i, j)| (i * 2 + j) as f64);
    let y = Array1::from_shape_fn(40, |i| i as f64);

    let mut param_grid = HashMap::new();
    param_grid.insert("offset".to_string(), vec![0.0, 5.0, 10.0]);

    let mut search = GridSearchCV::new(param_grid, 3).with_maximize(false);
    let metric = MseMetric;
    let estimator = MeanEstimator { offset: 0.0 };

    search.search(&estimator, &x, &y, &metric).unwrap();

    let best = search.best_params().unwrap();
    // offset=0 gives mean prediction which minimizes MSE vs offset=5 or 10
    assert!(
        (best["offset"] - 0.0).abs() < 1e-10,
        "Minimizing MSE should pick offset=0, got {}",
        best["offset"]
    );
}

#[test]
fn test_grid_search_respects_maximize_direction() {
    // R2 is maximized — offset=0 (mean predictor) gives R2=0; any offset makes it worse (negative R2)
    // So offset=0 should still be best even when maximizing R2
    let x = Array2::from_shape_fn((40, 2), |(i, j)| (i * 2 + j) as f64);
    let y = Array1::from_shape_fn(40, |i| i as f64);

    let mut param_grid = HashMap::new();
    param_grid.insert("offset".to_string(), vec![0.0, 50.0]);

    let mut search = GridSearchCV::new(param_grid, 3).with_maximize(true);
    let metric = R2Metric;
    let estimator = MeanEstimator { offset: 0.0 };

    search.search(&estimator, &x, &y, &metric).unwrap();

    let best = search.best_params().unwrap();
    assert!(
        (best["offset"] - 0.0).abs() < 1e-10,
        "Maximizing R2 should pick offset=0, got {}",
        best["offset"]
    );
}

// ============================================================================
// MetricValueWithCI Tests
// ============================================================================

#[test]
fn test_metric_value_with_ci_significantly_different() {
    let a = MetricValueWithCI {
        value: 0.9,
        ci_lower: 0.85,
        ci_upper: 0.95,
        confidence_level: 0.95,
        std_error: 0.025,
        n_bootstrap: 1000,
    };
    let b = MetricValueWithCI {
        value: 0.5,
        ci_lower: 0.45,
        ci_upper: 0.55,
        confidence_level: 0.95,
        std_error: 0.025,
        n_bootstrap: 1000,
    };

    assert!(a.significantly_different_from(&b));
    assert!(b.significantly_different_from(&a));
}

#[test]
fn test_metric_value_with_ci_overlapping_not_significant() {
    let a = MetricValueWithCI {
        value: 0.8,
        ci_lower: 0.7,
        ci_upper: 0.9,
        confidence_level: 0.95,
        std_error: 0.05,
        n_bootstrap: 1000,
    };
    let b = MetricValueWithCI {
        value: 0.85,
        ci_lower: 0.75,
        ci_upper: 0.95,
        confidence_level: 0.95,
        std_error: 0.05,
        n_bootstrap: 1000,
    };

    assert!(!a.significantly_different_from(&b));
}

#[test]
fn test_metric_value_with_ci_summary_format() {
    let ci = MetricValueWithCI {
        value: 0.85,
        ci_lower: 0.80,
        ci_upper: 0.90,
        confidence_level: 0.95,
        std_error: 0.025,
        n_bootstrap: 500,
    };
    let summary = ci.summary();
    assert!(summary.contains("0.85"));
    assert!(summary.contains("95%"));
}
