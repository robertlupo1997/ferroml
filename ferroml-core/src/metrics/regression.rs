//! Regression Metrics
//!
//! Metrics for evaluating regression models including MSE, RMSE, MAE, R², and more.

use crate::metrics::{validate_arrays, Direction, Metric, MetricValue};
use crate::{FerroError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Bundle of common regression metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionMetrics {
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// R-squared (coefficient of determination)
    pub r2: f64,
    /// Explained variance score
    pub explained_variance: f64,
    /// Maximum error
    pub max_error: f64,
    /// Median absolute error
    pub median_absolute_error: f64,
    /// Sample size
    pub n_samples: usize,
}

impl RegressionMetrics {
    /// Compute all regression metrics at once
    pub fn compute(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<Self> {
        validate_arrays(y_true, y_pred)?;

        let n = y_true.len();
        let mse_val = mse(y_true, y_pred)?;
        let rmse_val = mse_val.sqrt();
        let mae_val = mae(y_true, y_pred)?;
        let r2_val = r2_score(y_true, y_pred)?;
        let ev_val = explained_variance(y_true, y_pred)?;
        let max_err = max_error(y_true, y_pred)?;
        let median_ae = median_absolute_error(y_true, y_pred)?;

        Ok(Self {
            mse: mse_val,
            rmse: rmse_val,
            mae: mae_val,
            r2: r2_val,
            explained_variance: ev_val,
            max_error: max_err,
            median_absolute_error: median_ae,
            n_samples: n,
        })
    }

    /// Format as a summary string
    pub fn summary(&self) -> String {
        format!(
            "Regression Metrics (n={})\n\
             ========================\n\
             MSE:               {:.6}\n\
             RMSE:              {:.6}\n\
             MAE:               {:.6}\n\
             R²:                {:.6}\n\
             Explained Var:     {:.6}\n\
             Max Error:         {:.6}\n\
             Median Abs Error:  {:.6}",
            self.n_samples,
            self.mse,
            self.rmse,
            self.mae,
            self.r2,
            self.explained_variance,
            self.max_error,
            self.median_absolute_error
        )
    }
}

/// Compute Mean Squared Error
pub fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;
    let n = y_true.len() as f64;
    let sum_sq_error: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    Ok(sum_sq_error / n)
}

/// Compute Root Mean Squared Error
pub fn rmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    Ok(mse(y_true, y_pred)?.sqrt())
}

/// Compute Mean Absolute Error
pub fn mae(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;
    let n = y_true.len() as f64;
    let sum_abs_error: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .sum();
    Ok(sum_abs_error / n)
}

/// Compute R-squared (coefficient of determination)
///
/// R² = 1 - SS_res / SS_tot
/// where SS_res = Σ(y_true - y_pred)² and SS_tot = Σ(y_true - mean(y_true))²
pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;

    let mean_true = y_true.mean().unwrap_or(0.0);

    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();

    let ss_tot: f64 = y_true.iter().map(|&t| (t - mean_true).powi(2)).sum();

    if ss_tot == 0.0 {
        // All true values are the same
        if ss_res == 0.0 {
            return Ok(1.0); // Perfect prediction
        } else {
            return Ok(0.0); // Can't do better than mean
        }
    }

    Ok(1.0 - ss_res / ss_tot)
}

/// Compute Explained Variance Score
///
/// EV = 1 - Var(y_true - y_pred) / Var(y_true)
pub fn explained_variance(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;

    let residuals: Array1<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| t - p)
        .collect();

    let var_true = y_true.var(1.0);
    let var_residuals = residuals.var(1.0);

    if var_true == 0.0 {
        if var_residuals == 0.0 {
            return Ok(1.0);
        } else {
            return Ok(0.0);
        }
    }

    Ok(1.0 - var_residuals / var_true)
}

/// Compute Maximum Error
pub fn max_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;
    let max_err = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(max_err)
}

/// Compute Median Absolute Error
pub fn median_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;

    let mut errors: Vec<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .collect();

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = errors.len();
    let median = if n % 2 == 0 {
        (errors[n / 2 - 1] + errors[n / 2]) / 2.0
    } else {
        errors[n / 2]
    };

    Ok(median)
}

/// Compute Mean Absolute Percentage Error (MAPE)
///
/// Note: Undefined when y_true contains zeros. Returns error in such cases.
pub fn mape(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64> {
    validate_arrays(y_true, y_pred)?;

    if y_true.iter().any(|&t| t == 0.0) {
        return Err(FerroError::invalid_input(
            "MAPE is undefined when y_true contains zeros",
        ));
    }

    let n = y_true.len() as f64;
    let sum_ape: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| ((t - p) / t).abs())
        .sum();

    Ok(sum_ape / n * 100.0) // Return as percentage
}

// Metric trait implementations

/// MSE metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct MseMetric;

impl Metric for MseMetric {
    fn name(&self) -> &str {
        "mse"
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = mse(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// RMSE metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct RmseMetric;

impl Metric for RmseMetric {
    fn name(&self) -> &str {
        "rmse"
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = rmse(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// MAE metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct MaeMetric;

impl Metric for MaeMetric {
    fn name(&self) -> &str {
        "mae"
    }

    fn direction(&self) -> Direction {
        Direction::Minimize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = mae(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// R² metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct R2Metric;

impl Metric for R2Metric {
    fn name(&self) -> &str {
        "r2"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = r2_score(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

/// Explained Variance metric implementing the Metric trait
#[derive(Debug, Clone, Default)]
pub struct ExplainedVarianceMetric;

impl Metric for ExplainedVarianceMetric {
    fn name(&self) -> &str {
        "explained_variance"
    }

    fn direction(&self) -> Direction {
        Direction::Maximize
    }

    fn compute(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<MetricValue> {
        let value = explained_variance(y_true, y_pred)?;
        Ok(MetricValue::new(self.name(), value, self.direction()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mse() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let result = mse(&y_true, &y_pred).unwrap();
        // (0.5² + 0.5² + 0² + 1²) / 4 = 1.5 / 4 = 0.375
        assert_relative_eq!(result, 0.375, epsilon = 1e-10);
    }

    #[test]
    fn test_rmse() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let result = rmse(&y_true, &y_pred).unwrap();
        assert_relative_eq!(result, 0.375_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_mae() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let result = mae(&y_true, &y_pred).unwrap();
        // (0.5 + 0.5 + 0 + 1) / 4 = 2 / 4 = 0.5
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_r2_score() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let result = r2_score(&y_true, &y_pred).unwrap();
        // sklearn returns 0.9486081370449679 for this example
        assert_relative_eq!(result, 0.9486081370449679, epsilon = 1e-6);
    }

    #[test]
    fn test_r2_perfect_prediction() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = r2_score(&y_true, &y_pred).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_explained_variance() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let result = explained_variance(&y_true, &y_pred).unwrap();
        assert!(result > 0.0 && result <= 1.0);
    }

    #[test]
    fn test_max_error() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let result = max_error(&y_true, &y_pred).unwrap();
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_median_absolute_error() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let result = median_absolute_error(&y_true, &y_pred).unwrap();
        // Errors: 0.5, 0.5, 0.0, 1.0 -> sorted: 0.0, 0.5, 0.5, 1.0
        // Median of even array: (0.5 + 0.5) / 2 = 0.5
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_regression_metrics_bundle() {
        let y_true = Array1::from_vec(vec![3.0, -0.5, 2.0, 7.0]);
        let y_pred = Array1::from_vec(vec![2.5, 0.0, 2.0, 8.0]);

        let metrics = RegressionMetrics::compute(&y_true, &y_pred).unwrap();

        assert_eq!(metrics.n_samples, 4);
        assert!(metrics.mse > 0.0);
        assert!(metrics.r2 > 0.0 && metrics.r2 <= 1.0);
    }

    #[test]
    fn test_mse_metric_trait() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 4.0]);

        let metric = MseMetric;
        let result = metric.compute(&y_true, &y_pred).unwrap();

        assert_eq!(result.name, "mse");
        assert_eq!(result.direction, Direction::Minimize);
        assert_relative_eq!(result.value, 1.0 / 3.0, epsilon = 1e-10);
    }
}
