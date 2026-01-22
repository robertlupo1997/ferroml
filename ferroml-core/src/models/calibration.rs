//! Probability Calibration for Classifiers
//!
//! This module provides probability calibration for classifiers, ensuring that
//! predicted probabilities accurately reflect the true likelihood of outcomes.
//! This is critical for production systems where calibrated probabilities are
//! needed for decision-making.
//!
//! ## Why Calibration Matters
//!
//! Many classifiers (e.g., SVM, Random Forest, Naive Bayes) produce probability
//! estimates that are not well-calibrated. For example, a model might predict
//! P(y=1) = 0.8 for many samples that are only correct 60% of the time.
//!
//! Calibration ensures that when a model predicts P(y=1) = 0.8, approximately
//! 80% of those predictions are actually positive.
//!
//! ## Calibration Methods
//!
//! - **Platt scaling** (sigmoid): Fits a logistic regression to the classifier's
//!   scores. Works well for models with sigmoid-shaped distortions.
//! - **Isotonic regression**: Non-parametric calibration that learns a monotonic
//!   mapping. More flexible but requires more data.
//! - **Temperature scaling**: Simple scaling for neural networks (multiclass).
//!
//! ## CalibratedClassifierCV
//!
//! Uses cross-validation to calibrate probabilities without data leakage:
//! 1. Split data into K folds
//! 2. For each fold: train classifier on K-1 folds, calibrate on held-out fold
//! 3. At prediction time: average calibrated probabilities across all models
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::models::calibration::{CalibratedClassifierCV, CalibrationMethod};
//! use ferroml_core::models::{SVC, Model};
//! use ndarray::{Array1, Array2};
//!
//! // SVC with RBF kernel (probabilities may be miscalibrated)
//! let svc = SVC::new();
//!
//! // Wrap with calibration
//! let mut calibrated = CalibratedClassifierCV::new(Box::new(svc))
//!     .with_method(CalibrationMethod::Sigmoid)
//!     .with_n_folds(5);
//!
//! calibrated.fit(&x, &y)?;
//! let calibrated_probas = calibrated.predict_proba(&x)?;
//! ```

use crate::cv::{CrossValidator, KFold, StratifiedKFold};
use crate::hpo::SearchSpace;
use crate::models::{check_is_fitted, validate_fit_input, validate_predict_input, Model};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Calibration Method Enum
// =============================================================================

/// Method for probability calibration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Platt scaling (sigmoid/logistic calibration)
    ///
    /// Fits: P(y=1) = 1 / (1 + exp(A * f(x) + B))
    /// where f(x) is the classifier's probability/score.
    ///
    /// Works well for models with sigmoid-shaped probability distortions.
    Sigmoid,

    /// Isotonic regression calibration
    ///
    /// Learns a monotonic mapping from scores to probabilities using
    /// isotonic regression. More flexible than sigmoid but needs more data.
    Isotonic,

    /// Temperature scaling calibration
    ///
    /// Divides logits by a learned temperature T before softmax:
    /// P(y=k) = softmax(z_k / T)
    ///
    /// Simple single-parameter calibration that preserves accuracy while
    /// improving calibration. Particularly effective for neural networks
    /// and other softmax-based multi-class classifiers.
    Temperature,
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        Self::Sigmoid
    }
}

impl fmt::Display for CalibrationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CalibrationMethod::Sigmoid => write!(f, "sigmoid"),
            CalibrationMethod::Isotonic => write!(f, "isotonic"),
            CalibrationMethod::Temperature => write!(f, "temperature"),
        }
    }
}

// =============================================================================
// Calibrator Trait
// =============================================================================

/// Trait for probability calibrators
///
/// Calibrators transform uncalibrated scores/probabilities into calibrated ones.
pub trait Calibrator: Send + Sync {
    /// Fit the calibrator on predictions and true labels
    ///
    /// # Arguments
    /// * `y_prob` - Uncalibrated probability predictions (positive class)
    /// * `y_true` - True binary labels (0 or 1)
    fn fit(&mut self, y_prob: &Array1<f64>, y_true: &Array1<f64>) -> Result<()>;

    /// Transform uncalibrated probabilities to calibrated ones
    ///
    /// # Arguments
    /// * `y_prob` - Uncalibrated probability predictions (positive class)
    ///
    /// # Returns
    /// Calibrated probabilities
    fn transform(&self, y_prob: &Array1<f64>) -> Result<Array1<f64>>;

    /// Check if the calibrator has been fitted
    fn is_fitted(&self) -> bool;

    /// Clone the calibrator into a box
    fn clone_boxed(&self) -> Box<dyn Calibrator>;

    /// Get the calibration method type
    fn method(&self) -> CalibrationMethod;
}

// =============================================================================
// Sigmoid Calibrator (Platt Scaling)
// =============================================================================

/// Platt scaling calibrator using sigmoid function
///
/// Fits: P(y=1) = 1 / (1 + exp(A * f(x) + B))
///
/// Uses maximum likelihood to estimate A and B parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmoidCalibrator {
    /// Slope parameter (A)
    a: Option<f64>,
    /// Intercept parameter (B)
    b: Option<f64>,
    /// Maximum iterations for fitting
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
}

impl Default for SigmoidCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl SigmoidCalibrator {
    /// Create a new sigmoid calibrator
    #[must_use]
    pub fn new() -> Self {
        Self {
            a: None,
            b: None,
            max_iter: 100,
            tol: 1e-8,
        }
    }

    /// Set maximum iterations
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

    /// Get the fitted parameters (A, B)
    #[must_use]
    pub fn parameters(&self) -> Option<(f64, f64)> {
        self.a.zip(self.b)
    }

    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Calibrator for SigmoidCalibrator {
    fn fit(&mut self, y_prob: &Array1<f64>, y_true: &Array1<f64>) -> Result<()> {
        let n = y_prob.len();
        if n != y_true.len() {
            return Err(FerroError::shape_mismatch(
                format!("y_prob length {}", n),
                format!("y_true length {}", y_true.len()),
            ));
        }
        if n == 0 {
            return Err(FerroError::invalid_input("Empty input arrays"));
        }

        // Count positive and negative samples
        let n_pos = y_true.iter().filter(|&&y| y > 0.5).count();
        let n_neg = n - n_pos;

        if n_pos == 0 || n_neg == 0 {
            return Err(FerroError::invalid_input(
                "Calibration requires both positive and negative samples",
            ));
        }

        // Use Platt's scaling targets to avoid overfitting
        // t+ = (N+ + 1) / (N+ + 2), t- = 1 / (N- + 2)
        let t_pos = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let t_neg = 1.0 / (n_neg as f64 + 2.0);

        // Transform y_true to target values
        let targets: Array1<f64> = y_true
            .iter()
            .map(|&y| if y > 0.5 { t_pos } else { t_neg })
            .collect();

        // Fit logistic regression: P = 1 / (1 + exp(A*f + B))
        // Initialize parameters
        let mut a = 0.0_f64;
        let mut b = y_prob.mean().unwrap_or(0.0).ln().max(-5.0).min(5.0);

        // Newton-Raphson optimization
        for _ in 0..self.max_iter {
            // Compute probabilities: p = sigmoid(a * f + b)
            let z: Array1<f64> = y_prob.mapv(|f| a * f + b);
            let p: Array1<f64> = z.mapv(Self::sigmoid);

            // Compute gradient
            let diff = &p - &targets;
            let grad_a: f64 = diff.iter().zip(y_prob.iter()).map(|(d, f)| d * f).sum();
            let grad_b: f64 = diff.sum();

            // Compute Hessian diagonal (simplified)
            let pq: Array1<f64> = p.iter().map(|&pi| pi * (1.0 - pi).max(1e-10)).collect();
            let hess_aa: f64 = pq.iter().zip(y_prob.iter()).map(|(pq, f)| pq * f * f).sum();
            let hess_ab: f64 = pq.iter().zip(y_prob.iter()).map(|(pq, f)| pq * f).sum();
            let hess_bb: f64 = pq.sum();

            // Solve 2x2 system using Cramer's rule with regularization
            let det = hess_aa * hess_bb - hess_ab * hess_ab + 1e-10;
            let delta_a = -(hess_bb * grad_a - hess_ab * grad_b) / det;
            let delta_b = -(-hess_ab * grad_a + hess_aa * grad_b) / det;

            // Update with step size limiting
            let step_size = 1.0_f64.min(1.0 / (delta_a.abs() + delta_b.abs() + 1e-10));
            a += step_size * delta_a;
            b += step_size * delta_b;

            // Check convergence
            if (delta_a.abs() + delta_b.abs()) * step_size < self.tol {
                break;
            }
        }

        self.a = Some(a);
        self.b = Some(b);
        Ok(())
    }

    fn transform(&self, y_prob: &Array1<f64>) -> Result<Array1<f64>> {
        let (a, b) = self
            .parameters()
            .ok_or_else(|| FerroError::not_fitted("SigmoidCalibrator.transform"))?;

        // Apply calibration: P_calibrated = sigmoid(a * f + b)
        let calibrated = y_prob.mapv(|f| Self::sigmoid(a * f + b));
        Ok(calibrated)
    }

    fn is_fitted(&self) -> bool {
        self.a.is_some() && self.b.is_some()
    }

    fn clone_boxed(&self) -> Box<dyn Calibrator> {
        Box::new(self.clone())
    }

    fn method(&self) -> CalibrationMethod {
        CalibrationMethod::Sigmoid
    }
}

// =============================================================================
// Isotonic Calibrator
// =============================================================================

/// Isotonic regression calibrator
///
/// Learns a monotonically increasing mapping from scores to probabilities
/// using the pool adjacent violators algorithm (PAVA).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicCalibrator {
    /// Fitted x values (sorted scores)
    x_fitted: Option<Vec<f64>>,
    /// Fitted y values (calibrated probabilities)
    y_fitted: Option<Vec<f64>>,
    /// Whether the mapping should be increasing (true) or decreasing (false)
    increasing: bool,
    /// Clip output to [0, 1]
    clip: bool,
}

impl Default for IsotonicCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl IsotonicCalibrator {
    /// Create a new isotonic calibrator
    #[must_use]
    pub fn new() -> Self {
        Self {
            x_fitted: None,
            y_fitted: None,
            increasing: true,
            clip: true,
        }
    }

    /// Set whether the mapping should be increasing
    #[must_use]
    pub fn with_increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set whether to clip output to [0, 1]
    #[must_use]
    pub fn with_clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    /// Get fitted values for inspection
    #[must_use]
    pub fn fitted_values(&self) -> Option<(&[f64], &[f64])> {
        match (&self.x_fitted, &self.y_fitted) {
            (Some(x), Some(y)) => Some((x.as_slice(), y.as_slice())),
            _ => None,
        }
    }

    /// Pool Adjacent Violators Algorithm (PAVA)
    ///
    /// Produces a monotonically increasing sequence from weighted observations.
    fn pava(y: &[f64], weights: &[f64]) -> Vec<f64> {
        let n = y.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![y[0]];
        }

        // Initialize blocks: each observation starts as its own block
        let mut result = y.to_vec();
        let mut block_weights = weights.to_vec();

        // Weighted block means
        let mut block_start: Vec<usize> = (0..n).collect();
        let mut block_end: Vec<usize> = (0..n).collect();

        // Iteratively merge blocks that violate monotonicity
        let mut changed = true;
        while changed {
            changed = false;

            let mut i = 0;
            while i < n - 1 {
                let next = block_end[i] + 1;
                if next >= n {
                    break;
                }

                // Check if block i violates monotonicity with next block
                if result[i] > result[next] {
                    // Merge blocks: weighted average
                    let w_i = block_weights[i];
                    let w_next = block_weights[next];
                    let new_value = (result[i] * w_i + result[next] * w_next) / (w_i + w_next);

                    // Update values for merged block
                    result[i] = new_value;
                    result[next] = new_value;
                    block_weights[i] = w_i + w_next;
                    block_end[i] = block_end[next];

                    // Propagate block membership
                    for j in block_start[i]..=block_end[i] {
                        result[j] = new_value;
                        block_start[j] = block_start[i];
                        block_end[j] = block_end[i];
                    }

                    changed = true;
                }

                // Move to next unique block
                i = block_end[i] + 1;
            }
        }

        result
    }

    /// Linear interpolation
    fn interpolate(x_query: f64, x_data: &[f64], y_data: &[f64]) -> f64 {
        if x_data.is_empty() {
            return 0.5; // Default
        }
        if x_data.len() == 1 {
            return y_data[0];
        }

        // Find interval containing x_query
        if x_query <= x_data[0] {
            return y_data[0];
        }
        if x_query >= x_data[x_data.len() - 1] {
            return y_data[y_data.len() - 1];
        }

        // Binary search for interval
        let mut lo = 0;
        let mut hi = x_data.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if x_data[mid] <= x_query {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation
        let x0 = x_data[lo];
        let x1 = x_data[hi];
        let y0 = y_data[lo];
        let y1 = y_data[hi];

        if (x1 - x0).abs() < 1e-15 {
            return y0;
        }

        y0 + (y1 - y0) * (x_query - x0) / (x1 - x0)
    }
}

impl Calibrator for IsotonicCalibrator {
    fn fit(&mut self, y_prob: &Array1<f64>, y_true: &Array1<f64>) -> Result<()> {
        let n = y_prob.len();
        if n != y_true.len() {
            return Err(FerroError::shape_mismatch(
                format!("y_prob length {}", n),
                format!("y_true length {}", y_true.len()),
            ));
        }
        if n == 0 {
            return Err(FerroError::invalid_input("Empty input arrays"));
        }

        // Sort by probability scores
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            y_prob[i]
                .partial_cmp(&y_prob[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let x_sorted: Vec<f64> = indices.iter().map(|&i| y_prob[i]).collect();
        let y_sorted: Vec<f64> = indices.iter().map(|&i| y_true[i]).collect();
        let weights: Vec<f64> = vec![1.0; n]; // Uniform weights

        // Apply PAVA to get monotonic calibration
        let mut y_isotonic = if self.increasing {
            Self::pava(&y_sorted, &weights)
        } else {
            // For decreasing, reverse, apply PAVA, reverse again
            let y_rev: Vec<f64> = y_sorted.iter().rev().copied().collect();
            let w_rev: Vec<f64> = weights.iter().rev().copied().collect();
            let mut iso_rev = Self::pava(&y_rev, &w_rev);
            iso_rev.reverse();
            iso_rev
        };

        // Clip if requested
        if self.clip {
            for y in &mut y_isotonic {
                *y = y.clamp(0.0, 1.0);
            }
        }

        // Remove duplicates by averaging
        let mut x_unique = vec![x_sorted[0]];
        let mut y_unique = vec![y_isotonic[0]];
        let mut count = 1.0;

        for i in 1..n {
            if (x_sorted[i] - x_unique.last().unwrap()).abs() < 1e-10 {
                // Same x value: accumulate for average
                let last_idx = y_unique.len() - 1;
                y_unique[last_idx] = (y_unique[last_idx] * count + y_isotonic[i]) / (count + 1.0);
                count += 1.0;
            } else {
                x_unique.push(x_sorted[i]);
                y_unique.push(y_isotonic[i]);
                count = 1.0;
            }
        }

        self.x_fitted = Some(x_unique);
        self.y_fitted = Some(y_unique);
        Ok(())
    }

    fn transform(&self, y_prob: &Array1<f64>) -> Result<Array1<f64>> {
        let (x_data, y_data) = self
            .fitted_values()
            .ok_or_else(|| FerroError::not_fitted("IsotonicCalibrator.transform"))?;

        let mut calibrated = Array1::zeros(y_prob.len());
        for (i, &p) in y_prob.iter().enumerate() {
            let mut cal_p = Self::interpolate(p, x_data, y_data);
            if self.clip {
                cal_p = cal_p.clamp(0.0, 1.0);
            }
            calibrated[i] = cal_p;
        }

        Ok(calibrated)
    }

    fn is_fitted(&self) -> bool {
        self.x_fitted.is_some() && self.y_fitted.is_some()
    }

    fn clone_boxed(&self) -> Box<dyn Calibrator> {
        Box::new(self.clone())
    }

    fn method(&self) -> CalibrationMethod {
        CalibrationMethod::Isotonic
    }
}

// =============================================================================
// Multi-class Calibrator Trait
// =============================================================================

/// Trait for multi-class probability calibrators
///
/// Unlike the binary `Calibrator` trait, this works with 2D probability arrays
/// where each row is a sample and each column is a class probability.
pub trait MulticlassCalibrator: Send + Sync {
    /// Fit the calibrator on predictions and true labels
    ///
    /// # Arguments
    /// * `y_prob` - Uncalibrated probability predictions (n_samples, n_classes)
    /// * `y_true` - True class labels (0, 1, 2, ... n_classes-1)
    fn fit(&mut self, y_prob: &Array2<f64>, y_true: &Array1<f64>) -> Result<()>;

    /// Transform uncalibrated probabilities to calibrated ones
    ///
    /// # Arguments
    /// * `y_prob` - Uncalibrated probability predictions (n_samples, n_classes)
    ///
    /// # Returns
    /// Calibrated probabilities (n_samples, n_classes)
    fn transform(&self, y_prob: &Array2<f64>) -> Result<Array2<f64>>;

    /// Check if the calibrator has been fitted
    fn is_fitted(&self) -> bool;

    /// Clone the calibrator into a box
    fn clone_boxed(&self) -> Box<dyn MulticlassCalibrator>;

    /// Get the calibration method type
    fn method(&self) -> CalibrationMethod;
}

// =============================================================================
// Temperature Scaling Calibrator
// =============================================================================

/// Temperature scaling calibrator for multi-class classifiers
///
/// Temperature scaling is a simple but effective post-hoc calibration method
/// that learns a single scalar parameter T (temperature) to scale the logits:
///
/// ```text
/// calibrated_proba = softmax(log(proba) / T)
/// ```
///
/// ## Key Properties
///
/// - **Single parameter**: Only learns one temperature value T
/// - **Preserves accuracy**: Does not change the predicted class (argmax)
/// - **Simple optimization**: Minimizes negative log-likelihood on validation set
/// - **Effective for neural networks**: Particularly good for overconfident models
///
/// ## Algorithm
///
/// 1. Convert probabilities to pseudo-logits: z = log(p + ε)
/// 2. Learn temperature T by minimizing NLL: -Σ log(softmax(z/T)\[y\])
/// 3. At prediction time: p_calibrated = softmax(z/T)
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::calibration::TemperatureScalingCalibrator;
///
/// let mut calibrator = TemperatureScalingCalibrator::new();
/// calibrator.fit(&y_prob, &y_true)?;
///
/// let calibrated = calibrator.transform(&y_prob)?;
/// println!("Learned temperature: {}", calibrator.temperature().unwrap());
/// ```
///
/// ## Reference
///
/// Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
/// "On Calibration of Modern Neural Networks". ICML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScalingCalibrator {
    /// Learned temperature parameter (T > 0)
    temperature: Option<f64>,
    /// Maximum iterations for optimization
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Small constant to avoid log(0)
    eps: f64,
}

impl Default for TemperatureScalingCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl TemperatureScalingCalibrator {
    /// Create a new temperature scaling calibrator
    #[must_use]
    pub fn new() -> Self {
        Self {
            temperature: None,
            max_iter: 100,
            tol: 1e-6,
            learning_rate: 0.01,
            eps: 1e-12,
        }
    }

    /// Set maximum iterations for optimization
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

    /// Set learning rate for gradient descent
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Get the fitted temperature parameter
    #[must_use]
    pub fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    /// Softmax function applied to each row
    fn softmax(logits: &Array2<f64>) -> Array2<f64> {
        let n_samples = logits.nrows();
        let n_classes = logits.ncols();
        let mut result = Array2::zeros((n_samples, n_classes));

        for (i, row) in logits.rows().into_iter().enumerate() {
            // Subtract max for numerical stability
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_vals: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f64 = exp_vals.iter().sum();

            for (j, &exp_val) in exp_vals.iter().enumerate() {
                result[[i, j]] = exp_val / sum;
            }
        }

        result
    }

    /// Compute negative log-likelihood
    fn nll(&self, y_prob: &Array2<f64>, y_true: &Array1<f64>, temperature: f64) -> f64 {
        let n_samples = y_prob.nrows();
        let n_classes = y_prob.ncols();

        // Convert to pseudo-logits
        let logits: Array2<f64> = y_prob.mapv(|p| (p + self.eps).ln());

        // Scale by temperature
        let scaled_logits: Array2<f64> = logits.mapv(|z| z / temperature);

        // Apply softmax
        let calibrated = Self::softmax(&scaled_logits);

        // Compute NLL
        let mut nll = 0.0;
        for i in 0..n_samples {
            let true_class = y_true[i] as usize;
            if true_class < n_classes {
                nll -= (calibrated[[i, true_class]] + self.eps).ln();
            }
        }

        nll / n_samples as f64
    }

    /// Compute gradient of NLL with respect to temperature
    fn nll_gradient(&self, y_prob: &Array2<f64>, y_true: &Array1<f64>, temperature: f64) -> f64 {
        let n_samples = y_prob.nrows();
        let n_classes = y_prob.ncols();

        // Convert to pseudo-logits
        let logits: Array2<f64> = y_prob.mapv(|p| (p + self.eps).ln());

        // Scale by temperature
        let scaled_logits: Array2<f64> = logits.mapv(|z| z / temperature);

        // Apply softmax to get calibrated probabilities
        let calibrated = Self::softmax(&scaled_logits);

        // Gradient: d(NLL)/d(T) = (1/T²) * Σᵢ(Σⱼ pⱼ * zⱼ - z_yᵢ)
        // where pⱼ is the calibrated probability and z_yᵢ is the logit for true class
        let mut gradient = 0.0;
        for i in 0..n_samples {
            let true_class = y_true[i] as usize;
            if true_class < n_classes {
                // Expected logit under calibrated distribution
                let expected_logit: f64 = (0..n_classes)
                    .map(|j| calibrated[[i, j]] * logits[[i, j]])
                    .sum();

                // Logit for true class
                let true_logit = logits[[i, true_class]];

                // Contribution to gradient
                gradient += expected_logit - true_logit;
            }
        }

        gradient / (temperature * temperature * n_samples as f64)
    }
}

impl MulticlassCalibrator for TemperatureScalingCalibrator {
    fn fit(&mut self, y_prob: &Array2<f64>, y_true: &Array1<f64>) -> Result<()> {
        let n_samples = y_prob.nrows();
        let n_classes = y_prob.ncols();

        if n_samples != y_true.len() {
            return Err(FerroError::shape_mismatch(
                format!("y_prob rows {}", n_samples),
                format!("y_true length {}", y_true.len()),
            ));
        }
        if n_samples == 0 {
            return Err(FerroError::invalid_input("Empty input arrays"));
        }
        if n_classes < 2 {
            return Err(FerroError::invalid_input(
                "Temperature scaling requires at least 2 classes",
            ));
        }

        // Validate class labels
        for &label in y_true.iter() {
            let class_idx = label as usize;
            if class_idx >= n_classes {
                return Err(FerroError::invalid_input(format!(
                    "Class label {} exceeds number of classes {}",
                    label, n_classes
                )));
            }
        }

        // Initialize temperature to 1.0 (uncalibrated)
        let mut t = 1.0_f64;
        let mut prev_nll = f64::INFINITY;

        // Optimize using gradient descent with line search
        for _ in 0..self.max_iter {
            let current_nll = self.nll(y_prob, y_true, t);

            // Check convergence
            if (prev_nll - current_nll).abs() < self.tol {
                break;
            }
            prev_nll = current_nll;

            // Compute gradient
            let grad = self.nll_gradient(y_prob, y_true, t);

            // Update with gradient descent (ensure T > 0)
            let new_t = (t - self.learning_rate * grad).max(0.01);
            t = new_t;
        }

        self.temperature = Some(t);
        Ok(())
    }

    fn transform(&self, y_prob: &Array2<f64>) -> Result<Array2<f64>> {
        let temperature = self
            .temperature
            .ok_or_else(|| FerroError::not_fitted("TemperatureScalingCalibrator.transform"))?;

        // Convert to pseudo-logits
        let logits: Array2<f64> = y_prob.mapv(|p| (p + self.eps).ln());

        // Scale by temperature
        let scaled_logits: Array2<f64> = logits.mapv(|z| z / temperature);

        // Apply softmax
        Ok(Self::softmax(&scaled_logits))
    }

    fn is_fitted(&self) -> bool {
        self.temperature.is_some()
    }

    fn clone_boxed(&self) -> Box<dyn MulticlassCalibrator> {
        Box::new(self.clone())
    }

    fn method(&self) -> CalibrationMethod {
        CalibrationMethod::Temperature
    }
}

// Also implement binary Calibrator trait for compatibility
// When used with binary classification, treats column 1 as positive class
impl Calibrator for TemperatureScalingCalibrator {
    fn fit(&mut self, y_prob: &Array1<f64>, y_true: &Array1<f64>) -> Result<()> {
        // Convert to 2-class format: [P(neg), P(pos)]
        let n = y_prob.len();
        let mut y_prob_2d = Array2::zeros((n, 2));
        for (i, &p) in y_prob.iter().enumerate() {
            y_prob_2d[[i, 0]] = 1.0 - p;
            y_prob_2d[[i, 1]] = p;
        }

        // Use MulticlassCalibrator implementation
        MulticlassCalibrator::fit(self, &y_prob_2d, y_true)
    }

    fn transform(&self, y_prob: &Array1<f64>) -> Result<Array1<f64>> {
        // Convert to 2-class format
        let n = y_prob.len();
        let mut y_prob_2d = Array2::zeros((n, 2));
        for (i, &p) in y_prob.iter().enumerate() {
            y_prob_2d[[i, 0]] = 1.0 - p;
            y_prob_2d[[i, 1]] = p;
        }

        // Transform using MulticlassCalibrator
        let calibrated_2d = MulticlassCalibrator::transform(self, &y_prob_2d)?;

        // Extract positive class probability
        Ok(calibrated_2d.column(1).to_owned())
    }

    fn is_fitted(&self) -> bool {
        MulticlassCalibrator::is_fitted(self)
    }

    fn clone_boxed(&self) -> Box<dyn Calibrator> {
        Box::new(self.clone())
    }

    fn method(&self) -> CalibrationMethod {
        CalibrationMethod::Temperature
    }
}

// =============================================================================
// Trait for Classifiers that can be Calibrated
// =============================================================================

/// Trait for classifiers that can be wrapped for calibration
///
/// Must support probability predictions and cloning.
pub trait CalibrableClassifier: Model {
    /// Predict class probabilities
    ///
    /// Returns array of shape (n_samples, n_classes)
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Clone the classifier into a box
    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier>;
}

// =============================================================================
// CalibratedClassifierCV
// =============================================================================

/// Cross-validated probability calibration wrapper
///
/// Wraps any classifier and uses cross-validation to calibrate its probability
/// estimates, preventing data leakage while learning the calibration mapping.
///
/// ## Algorithm
///
/// Training (fit):
/// 1. Split training data into K folds
/// 2. For each fold k:
///    - Train classifier on folds {1..K} \ {k}
///    - Get probability predictions on fold k
///    - Fit calibrator on (predictions, true labels) for fold k
/// 3. Retrain classifier on all data (for final predictions)
/// 4. Store all K calibrators
///
/// Prediction (predict_proba):
/// 1. Get probability predictions from trained classifier
/// 2. Apply each calibrator to get K calibrated predictions
/// 3. Return average of calibrated predictions
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::calibration::{CalibratedClassifierCV, CalibrationMethod};
/// use ferroml_core::models::SVC;
///
/// let svc = SVC::new();
/// let mut calibrated = CalibratedClassifierCV::new(Box::new(svc))
///     .with_method(CalibrationMethod::Sigmoid)
///     .with_n_folds(5);
///
/// calibrated.fit(&x, &y)?;
/// let proba = calibrated.predict_proba(&x)?;
/// ```
pub struct CalibratedClassifierCV {
    /// Base classifier to calibrate
    base_estimator: Box<dyn CalibrableClassifier>,
    /// Calibration method to use
    method: CalibrationMethod,
    /// Cross-validation strategy
    cv: Box<dyn CrossValidator>,
    /// Whether to use stratified splits
    stratified: bool,
    /// Random seed for reproducibility
    random_state: Option<u64>,

    // Fitted state
    fitted: bool,
    n_features: Option<usize>,
    classes: Option<Array1<f64>>,
    /// Calibrators for each fold
    calibrators: Vec<Box<dyn Calibrator>>,
    /// Final classifier trained on all data
    fitted_classifier: Option<Box<dyn CalibrableClassifier>>,
}

impl fmt::Debug for CalibratedClassifierCV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CalibratedClassifierCV")
            .field("method", &self.method)
            .field("stratified", &self.stratified)
            .field("random_state", &self.random_state)
            .field("fitted", &self.fitted)
            .field("n_features", &self.n_features)
            .field("classes", &self.classes)
            .field("n_calibrators", &self.calibrators.len())
            .finish()
    }
}

impl CalibratedClassifierCV {
    /// Create a new CalibratedClassifierCV
    ///
    /// # Arguments
    ///
    /// * `base_estimator` - The classifier to calibrate
    pub fn new(base_estimator: Box<dyn CalibrableClassifier>) -> Self {
        Self {
            base_estimator,
            method: CalibrationMethod::Sigmoid,
            cv: Box::new(StratifiedKFold::new(5)),
            stratified: true,
            random_state: None,
            fitted: false,
            n_features: None,
            classes: None,
            calibrators: Vec::new(),
            fitted_classifier: None,
        }
    }

    /// Set the calibration method
    #[must_use]
    pub fn with_method(mut self, method: CalibrationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the number of CV folds
    #[must_use]
    pub fn with_n_folds(mut self, n_folds: usize) -> Self {
        if self.stratified {
            self.cv = Box::new(StratifiedKFold::new(n_folds));
        } else {
            self.cv = Box::new(KFold::new(n_folds));
        }
        self
    }

    /// Set the cross-validation strategy
    #[must_use]
    pub fn with_cv(mut self, cv: Box<dyn CrossValidator>) -> Self {
        self.cv = cv;
        self
    }

    /// Set whether to use stratified splits
    #[must_use]
    pub fn with_stratified(mut self, stratified: bool) -> Self {
        self.stratified = stratified;
        // Update CV if we're using default folds
        let n_folds = self.cv.get_n_splits(None, None, None);
        if stratified {
            self.cv = Box::new(StratifiedKFold::new(n_folds));
        } else {
            self.cv = Box::new(KFold::new(n_folds));
        }
        self
    }

    /// Set the random seed for reproducibility
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get the calibration method
    #[must_use]
    pub fn method(&self) -> CalibrationMethod {
        self.method
    }

    /// Get the classes (after fitting)
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the number of calibrators (CV folds)
    #[must_use]
    pub fn n_calibrators(&self) -> usize {
        self.calibrators.len()
    }

    /// Create a new calibrator based on the method
    fn create_calibrator(&self) -> Box<dyn Calibrator> {
        match self.method {
            CalibrationMethod::Sigmoid => Box::new(SigmoidCalibrator::new()),
            CalibrationMethod::Isotonic => Box::new(IsotonicCalibrator::new()),
            CalibrationMethod::Temperature => Box::new(TemperatureScalingCalibrator::new()),
        }
    }

    /// Extract unique classes from labels
    fn extract_classes(y: &Array1<f64>) -> Array1<f64> {
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        Array1::from_vec(classes)
    }

    /// Select rows from an array based on indices
    fn select_rows(arr: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_cols = arr.ncols();
        let n_rows = indices.len();
        let mut result = Array2::zeros((n_rows, n_cols));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            result.row_mut(new_idx).assign(&arr.row(old_idx));
        }
        result
    }

    /// Select elements from a 1D array based on indices
    fn select_elements(arr: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        Array1::from_iter(indices.iter().map(|&i| arr[i]))
    }

    /// Predict class probabilities (calibrated)
    ///
    /// Returns calibrated probability estimates averaged across all CV folds.
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict_proba")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let n_samples = x.nrows();
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();

        // Get uncalibrated probabilities from the fitted classifier
        let fitted_clf = self.fitted_classifier.as_ref().unwrap();
        let uncalibrated_proba = fitted_clf.predict_proba_for_calibration(x)?;

        // For binary classification, calibrate P(y=1)
        // For multiclass, we'd need one-vs-rest calibration (future enhancement)
        if n_classes == 2 {
            // Get positive class probabilities
            // Handle both (n, 1) format (just P(positive)) and (n, 2) format ([P(neg), P(pos)])
            let p_positive: Array1<f64> = if uncalibrated_proba.ncols() == 1 {
                uncalibrated_proba.column(0).to_owned()
            } else {
                uncalibrated_proba.column(1).to_owned()
            };

            // Apply each calibrator and average
            let mut calibrated_sum: Array1<f64> = Array1::zeros(n_samples);
            for calibrator in &self.calibrators {
                let calibrated = calibrator.transform(&p_positive)?;
                calibrated_sum = calibrated_sum + calibrated;
            }

            let n_calibrators = self.calibrators.len() as f64;
            let calibrated_p1: Array1<f64> = calibrated_sum.mapv(|s: f64| s / n_calibrators);
            let calibrated_p0: Array1<f64> = calibrated_p1.mapv(|p: f64| 1.0 - p);

            // Stack into (n_samples, 2) array
            let mut result = Array2::zeros((n_samples, 2));
            result.column_mut(0).assign(&calibrated_p0);
            result.column_mut(1).assign(&calibrated_p1);
            Ok(result)
        } else {
            // For multiclass, return uncalibrated probabilities with warning
            // Full multiclass calibration would use one-vs-rest
            // This is a limitation to note - multiclass calibration is more complex
            Ok(uncalibrated_proba)
        }
    }
}

impl Model for CalibratedClassifierCV {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Extract unique classes
        let classes = Self::extract_classes(y);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(FerroError::invalid_input(
                "Calibration requires at least 2 classes",
            ));
        }

        // Generate CV folds
        let y_option = Some(y.clone());
        let folds = self.cv.split(n_samples, y_option.as_ref(), None)?;

        // Collect out-of-fold predictions and labels for calibration
        let mut oof_proba = Array1::zeros(n_samples);
        let mut oof_valid = vec![false; n_samples];

        self.calibrators.clear();

        // For each fold: train classifier, get OOF predictions, fit calibrator
        for fold in &folds {
            // Select train/test data for this fold
            let x_train = Self::select_rows(x, &fold.train_indices);
            let y_train = Self::select_elements(y, &fold.train_indices);
            let x_test = Self::select_rows(x, &fold.test_indices);
            let y_test = Self::select_elements(y, &fold.test_indices);

            // Clone and fit base estimator on training fold
            let mut fold_classifier = self.base_estimator.clone_boxed();
            fold_classifier.fit(&x_train, &y_train)?;

            // Get probability predictions on held-out fold
            let test_proba = fold_classifier.predict_proba_for_calibration(&x_test)?;

            // For binary classification, use positive class probability
            // Some classifiers return (n, 1) with P(positive), others return (n, 2)
            let test_p1: Array1<f64> = if test_proba.ncols() == 1 {
                // Single column: probability of positive class
                test_proba.column(0).to_owned()
            } else if n_classes == 2 {
                // Two columns: [P(negative), P(positive)]
                test_proba.column(1).to_owned()
            } else {
                // Multiclass: use class 1 for now
                test_proba.column(1.min(n_classes - 1)).to_owned()
            };

            // Store OOF predictions
            for (local_idx, &global_idx) in fold.test_indices.iter().enumerate() {
                oof_proba[global_idx] = test_p1[local_idx];
                oof_valid[global_idx] = true;
            }

            // Fit calibrator on this fold's OOF predictions
            let mut calibrator = self.create_calibrator();
            calibrator.fit(&test_p1, &y_test)?;
            self.calibrators.push(calibrator);
        }

        // Train final classifier on all data
        let mut final_classifier = self.base_estimator.clone_boxed();
        final_classifier.fit(x, y)?;
        self.fitted_classifier = Some(final_classifier);

        // Update state
        self.fitted = true;
        self.n_features = Some(n_features);
        self.classes = Some(classes);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        // Get calibrated probabilities and predict class with highest probability
        let proba = self.predict_proba(x)?;
        let classes = self.classes.as_ref().unwrap();

        let predictions: Array1<f64> = proba
            .rows()
            .into_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn search_space(&self) -> SearchSpace {
        // Could include calibration-specific parameters here
        SearchSpace::new()
    }
}

// =============================================================================
// Calibration Diagnostics
// =============================================================================

/// Result of calibration analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Bin boundaries for reliability diagram
    pub bin_edges: Vec<f64>,
    /// Mean predicted probability in each bin
    pub mean_predicted_proba: Vec<f64>,
    /// Fraction of positives (true probability) in each bin
    pub fraction_of_positives: Vec<f64>,
    /// Number of samples in each bin
    pub bin_counts: Vec<usize>,
    /// Expected Calibration Error (ECE)
    pub ece: f64,
    /// Maximum Calibration Error (MCE)
    pub mce: f64,
    /// Brier score
    pub brier_score: f64,
}

/// Compute calibration curve data for a reliability diagram
///
/// # Arguments
///
/// * `y_true` - True binary labels (0 or 1)
/// * `y_prob` - Predicted probabilities for positive class
/// * `n_bins` - Number of bins for the calibration curve
///
/// # Returns
///
/// `CalibrationResult` with bin statistics and calibration metrics
pub fn calibration_curve(
    y_true: &Array1<f64>,
    y_prob: &Array1<f64>,
    n_bins: usize,
) -> Result<CalibrationResult> {
    if y_true.len() != y_prob.len() {
        return Err(FerroError::shape_mismatch(
            format!("y_true length {}", y_true.len()),
            format!("y_prob length {}", y_prob.len()),
        ));
    }

    let n = y_true.len();
    if n == 0 {
        return Err(FerroError::invalid_input("Empty input arrays"));
    }

    let n_bins = n_bins.max(1);

    // Create uniform bins from 0 to 1
    let bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();

    // Assign samples to bins and compute statistics
    let mut bin_sums = vec![0.0; n_bins];
    let mut bin_true_sums = vec![0.0; n_bins];
    let mut bin_counts = vec![0usize; n_bins];

    for (&prob, &true_label) in y_prob.iter().zip(y_true.iter()) {
        let prob_clipped = prob.clamp(0.0, 1.0 - 1e-10);
        let bin_idx = (prob_clipped * n_bins as f64).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1);

        bin_sums[bin_idx] += prob;
        bin_true_sums[bin_idx] += true_label;
        bin_counts[bin_idx] += 1;
    }

    // Compute mean predicted probability and fraction of positives per bin
    let mean_predicted_proba: Vec<f64> = bin_sums
        .iter()
        .zip(bin_counts.iter())
        .map(
            |(&sum, &count)| {
                if count > 0 {
                    sum / count as f64
                } else {
                    0.0
                }
            },
        )
        .collect();

    let fraction_of_positives: Vec<f64> = bin_true_sums
        .iter()
        .zip(bin_counts.iter())
        .map(
            |(&sum, &count)| {
                if count > 0 {
                    sum / count as f64
                } else {
                    0.0
                }
            },
        )
        .collect();

    // Compute ECE (Expected Calibration Error)
    // ECE = sum over bins of (|accuracy - confidence| * fraction of samples in bin)
    let total_count = n as f64;
    let ece: f64 = mean_predicted_proba
        .iter()
        .zip(fraction_of_positives.iter())
        .zip(bin_counts.iter())
        .map(|((&pred, &true_frac), &count)| {
            if count > 0 {
                (pred - true_frac).abs() * (count as f64 / total_count)
            } else {
                0.0
            }
        })
        .sum();

    // Compute MCE (Maximum Calibration Error)
    let mce: f64 = mean_predicted_proba
        .iter()
        .zip(fraction_of_positives.iter())
        .zip(bin_counts.iter())
        .filter_map(|((&pred, &true_frac), &count)| {
            if count > 0 {
                Some((pred - true_frac).abs())
            } else {
                None
            }
        })
        .fold(0.0_f64, |a, b| a.max(b));

    // Compute Brier score
    let brier_score: f64 = y_prob
        .iter()
        .zip(y_true.iter())
        .map(|(&prob, &true_label)| (prob - true_label).powi(2))
        .sum::<f64>()
        / n as f64;

    Ok(CalibrationResult {
        bin_edges,
        mean_predicted_proba,
        fraction_of_positives,
        bin_counts,
        ece,
        mce,
        brier_score,
    })
}

// =============================================================================
// CalibrableClassifier Implementations for FerroML Classifiers
// =============================================================================

use super::boosting::GradientBoostingClassifier;
use super::forest::RandomForestClassifier;
use super::hist_boosting::HistGradientBoostingClassifier;
use super::knn::KNeighborsClassifier;
use super::logistic::LogisticRegression;
use super::naive_bayes::{BernoulliNB, GaussianNB, MultinomialNB};
use super::tree::DecisionTreeClassifier;
use super::ProbabilisticModel;

impl CalibrableClassifier for LogisticRegression {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

impl CalibrableClassifier for GaussianNB {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

impl CalibrableClassifier for MultinomialNB {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

impl CalibrableClassifier for BernoulliNB {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

impl CalibrableClassifier for DecisionTreeClassifier {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

impl CalibrableClassifier for RandomForestClassifier {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

impl CalibrableClassifier for GradientBoostingClassifier {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

impl CalibrableClassifier for HistGradientBoostingClassifier {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

impl CalibrableClassifier for KNeighborsClassifier {
    fn predict_proba_for_calibration(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn CalibrableClassifier> {
        Box::new(self.clone())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_calibrator() {
        // Well-separated binary classification
        let y_prob = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9]);
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut calibrator = SigmoidCalibrator::new();
        calibrator.fit(&y_prob, &y_true).unwrap();

        assert!(calibrator.is_fitted());
        assert!(calibrator.parameters().is_some());

        let calibrated = calibrator.transform(&y_prob).unwrap();
        assert_eq!(calibrated.len(), 6);

        // Calibrated probabilities should be valid (0, 1)
        for &p in calibrated.iter() {
            assert!(p > 0.0 && p < 1.0);
        }
    }

    #[test]
    fn test_isotonic_calibrator() {
        let y_prob = Array1::from_vec(vec![0.1, 0.4, 0.3, 0.8, 0.7, 0.9]);
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut calibrator = IsotonicCalibrator::new();
        calibrator.fit(&y_prob, &y_true).unwrap();

        assert!(calibrator.is_fitted());
        assert!(calibrator.fitted_values().is_some());

        let calibrated = calibrator.transform(&y_prob).unwrap();
        assert_eq!(calibrated.len(), 6);

        // Calibrated probabilities should be in [0, 1]
        for &p in calibrated.iter() {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_isotonic_monotonicity() {
        // Test that isotonic calibration produces monotonic output
        let y_prob = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);

        let mut calibrator = IsotonicCalibrator::new();
        calibrator.fit(&y_prob, &y_true).unwrap();

        let calibrated = calibrator.transform(&y_prob).unwrap();

        // Check monotonicity
        for i in 1..calibrated.len() {
            assert!(
                calibrated[i] >= calibrated[i - 1] - 1e-10,
                "Non-monotonic at index {}: {} < {}",
                i,
                calibrated[i],
                calibrated[i - 1]
            );
        }
    }

    #[test]
    fn test_calibration_curve() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y_prob = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9]);

        let result = calibration_curve(&y_true, &y_prob, 5).unwrap();

        assert_eq!(result.bin_edges.len(), 6);
        assert_eq!(result.mean_predicted_proba.len(), 5);
        assert_eq!(result.fraction_of_positives.len(), 5);
        assert_eq!(result.bin_counts.len(), 5);

        // ECE should be non-negative
        assert!(result.ece >= 0.0);

        // Brier score for this well-separated data should be reasonable
        assert!(result.brier_score < 0.5);
    }

    #[test]
    fn test_calibration_result_perfect() {
        // Perfect calibration: predicted probabilities match actual fractions
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let y_prob = Array1::from_vec(vec![
            0.05, 0.15, 0.25, 0.35, 0.55, 0.65, 0.75, 0.85, 0.95, 0.95,
        ]);

        let result = calibration_curve(&y_true, &y_prob, 10).unwrap();

        // ECE should be low for well-calibrated predictions
        assert!(result.ece < 0.3);
    }

    #[test]
    fn test_sigmoid_calibrator_edge_cases() {
        // Test with single positive and single negative
        let y_prob = Array1::from_vec(vec![0.2, 0.8]);
        let y_true = Array1::from_vec(vec![0.0, 1.0]);

        let mut calibrator = SigmoidCalibrator::new();
        calibrator.fit(&y_prob, &y_true).unwrap();

        assert!(calibrator.is_fitted());
    }

    #[test]
    fn test_calibrator_errors() {
        let mut calibrator = SigmoidCalibrator::new();

        // Empty inputs
        let empty_prob = Array1::<f64>::zeros(0);
        let empty_true = Array1::<f64>::zeros(0);
        assert!(calibrator.fit(&empty_prob, &empty_true).is_err());

        // Mismatched lengths
        let y_prob = Array1::from_vec(vec![0.5, 0.6]);
        let y_true = Array1::from_vec(vec![0.0]);
        assert!(calibrator.fit(&y_prob, &y_true).is_err());

        // All same class
        let y_prob = Array1::from_vec(vec![0.5, 0.6, 0.7]);
        let y_true = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        assert!(calibrator.fit(&y_prob, &y_true).is_err());
    }

    #[test]
    fn test_calibration_method_display() {
        assert_eq!(format!("{}", CalibrationMethod::Sigmoid), "sigmoid");
        assert_eq!(format!("{}", CalibrationMethod::Isotonic), "isotonic");
    }

    #[test]
    fn test_calibrated_classifier_cv_with_logistic() {
        // Create a binary classification dataset with some overlap to avoid perfect separation
        // Add noise pattern to avoid perfect separation
        let mut data = Vec::with_capacity(80);
        for i in 0..40 {
            let noise = if i % 5 == 0 { 2.0 } else { 0.0 }; // Add periodic noise
            data.push(i as f64 / 10.0 + noise);
            data.push((i as f64).sin() + (i % 3) as f64); // Non-linear pattern
        }
        let x = Array2::from_shape_vec((40, 2), data).unwrap();
        // Create labels with some misclassification to avoid perfect separation
        let y = Array1::from_iter((0..40).map(|i| {
            if i < 18 || (i >= 22 && i < 38) {
                if i < 20 {
                    0.0
                } else {
                    1.0
                }
            } else {
                // Flip some labels
                if i < 20 {
                    1.0
                } else {
                    0.0
                }
            }
        }));

        // Create and fit calibrated classifier with logistic regression
        let base_clf = LogisticRegression::new();
        let mut calibrated = CalibratedClassifierCV::new(Box::new(base_clf))
            .with_method(CalibrationMethod::Sigmoid)
            .with_n_folds(3);

        calibrated.fit(&x, &y).unwrap();

        assert!(calibrated.is_fitted());
        assert_eq!(calibrated.n_features(), Some(2));
        assert_eq!(calibrated.n_calibrators(), 3);
        assert!(calibrated.classes().is_some());
        assert_eq!(calibrated.classes().unwrap().len(), 2);

        // Test predictions
        let predictions = calibrated.predict(&x).unwrap();
        assert_eq!(predictions.len(), 40);

        // Test probability predictions
        let proba = calibrated.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), 40);
        assert_eq!(proba.ncols(), 2);

        // Probabilities should sum to 1
        for row in proba.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Probabilities don't sum to 1");
        }

        // Probabilities should be in [0, 1]
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "Probability out of bounds: {}", p);
        }
    }

    #[test]
    fn test_calibrated_classifier_cv_with_gaussian_nb() {
        // Create a simple binary classification dataset
        let x =
            Array2::from_shape_vec((60, 3), (0..180).map(|i| i as f64 / 60.0).collect()).unwrap();
        let y = Array1::from_iter((0..60).map(|i| if i < 30 { 0.0 } else { 1.0 }));

        // Create and fit calibrated classifier with Gaussian NB
        let base_clf = GaussianNB::new();
        let mut calibrated = CalibratedClassifierCV::new(Box::new(base_clf))
            .with_method(CalibrationMethod::Isotonic)
            .with_n_folds(5);

        calibrated.fit(&x, &y).unwrap();

        assert!(calibrated.is_fitted());
        assert_eq!(calibrated.method(), CalibrationMethod::Isotonic);
        assert_eq!(calibrated.n_calibrators(), 5);

        // Test probability predictions
        let proba = calibrated.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), 60);
        assert_eq!(proba.ncols(), 2);

        // Probabilities should be in [0, 1]
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "Probability out of bounds: {}", p);
        }
    }

    #[test]
    fn test_calibrated_classifier_cv_with_decision_tree() {
        // Create a simple binary classification dataset
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64 / 25.0).collect()).unwrap();
        let y = Array1::from_iter((0..50).map(|i| if i < 25 { 0.0 } else { 1.0 }));

        // Create and fit calibrated classifier with Decision Tree
        let base_clf = DecisionTreeClassifier::new().with_max_depth(Some(3));
        let mut calibrated = CalibratedClassifierCV::new(Box::new(base_clf))
            .with_method(CalibrationMethod::Sigmoid)
            .with_n_folds(3);

        calibrated.fit(&x, &y).unwrap();

        assert!(calibrated.is_fitted());
        assert_eq!(calibrated.n_calibrators(), 3);

        // Test predictions
        let predictions = calibrated.predict(&x).unwrap();
        assert_eq!(predictions.len(), 50);

        // All predictions should be valid class labels
        for &pred in predictions.iter() {
            assert!(pred == 0.0 || pred == 1.0, "Invalid prediction: {}", pred);
        }
    }

    #[test]
    fn test_calibrated_classifier_builder_pattern() {
        let base_clf = LogisticRegression::new();
        let calibrated = CalibratedClassifierCV::new(Box::new(base_clf))
            .with_method(CalibrationMethod::Isotonic)
            .with_n_folds(10)
            .with_stratified(true)
            .with_random_state(42);

        assert_eq!(calibrated.method(), CalibrationMethod::Isotonic);
        assert!(!calibrated.is_fitted());
    }

    #[test]
    fn test_calibrated_classifier_cv_debug() {
        let base_clf = LogisticRegression::new();
        let calibrated = CalibratedClassifierCV::new(Box::new(base_clf));

        // Debug formatting should work
        let debug_str = format!("{:?}", calibrated);
        assert!(debug_str.contains("CalibratedClassifierCV"));
        assert!(debug_str.contains("method"));
    }

    // =============================================================================
    // Temperature Scaling Tests
    // =============================================================================

    #[test]
    fn test_temperature_scaling_basic() {
        // Create a 3-class probability matrix (overconfident predictions)
        let y_prob = Array2::from_shape_vec(
            (6, 3),
            vec![
                0.95, 0.03, 0.02, // Sample 0: high confidence class 0
                0.02, 0.93, 0.05, // Sample 1: high confidence class 1
                0.04, 0.02, 0.94, // Sample 2: high confidence class 2
                0.90, 0.05, 0.05, // Sample 3: high confidence class 0
                0.03, 0.92, 0.05, // Sample 4: high confidence class 1
                0.04, 0.03, 0.93, // Sample 5: high confidence class 2
            ],
        )
        .unwrap();
        let y_true = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);

        let mut calibrator = TemperatureScalingCalibrator::new();
        MulticlassCalibrator::fit(&mut calibrator, &y_prob, &y_true).unwrap();

        assert!(MulticlassCalibrator::is_fitted(&calibrator));
        assert!(calibrator.temperature().is_some());

        // Temperature should be around 1.0 for well-calibrated predictions
        let t = calibrator.temperature().unwrap();
        assert!(t > 0.0, "Temperature must be positive");

        // Transform and verify
        let calibrated = MulticlassCalibrator::transform(&calibrator, &y_prob).unwrap();
        assert_eq!(calibrated.nrows(), 6);
        assert_eq!(calibrated.ncols(), 3);

        // Each row should sum to 1
        for row in calibrated.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row doesn't sum to 1: {}", sum);
        }

        // All probabilities should be in [0, 1]
        for &p in calibrated.iter() {
            assert!(p >= 0.0 && p <= 1.0, "Probability out of bounds: {}", p);
        }
    }

    #[test]
    fn test_temperature_scaling_overconfident() {
        // Create extremely overconfident predictions
        let y_prob = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.99, 0.01, // true class 0
                0.01, 0.99, // true class 1
                0.99, 0.01, // true class 0
                0.01, 0.99, // true class 1
                0.99, 0.01, // true class 0, but actually class 1 (miscalibrated)
                0.01, 0.99, // true class 1, but actually class 0 (miscalibrated)
                0.99, 0.01, // true class 0
                0.01, 0.99, // true class 1
                0.99, 0.01, // true class 0
                0.01, 0.99, // true class 1
            ],
        )
        .unwrap();
        // Some predictions are wrong - these should cause temperature > 1
        let y_true = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);

        let mut calibrator = TemperatureScalingCalibrator::new()
            .with_max_iter(200)
            .with_learning_rate(0.05);

        MulticlassCalibrator::fit(&mut calibrator, &y_prob, &y_true).unwrap();

        let t = calibrator.temperature().unwrap();
        // Temperature should be positive (the optimization ensures T > 0.01)
        assert!(t > 0.0, "Temperature must be positive: {}", t);

        let calibrated = MulticlassCalibrator::transform(&calibrator, &y_prob).unwrap();

        // Verify calibrated probabilities are valid
        for row in calibrated.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row doesn't sum to 1: {}", sum);
            for &p in row.iter() {
                assert!(p >= 0.0 && p <= 1.0, "Probability out of bounds: {}", p);
            }
        }

        // Predictions should still be valid
        // Note: temperature scaling preserves argmax, so ranking is maintained
    }

    #[test]
    fn test_temperature_scaling_binary_calibrator_interface() {
        // Test the binary Calibrator trait implementation
        let y_prob = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9]);
        let y_true = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut calibrator = TemperatureScalingCalibrator::new();
        Calibrator::fit(&mut calibrator, &y_prob, &y_true).unwrap();

        assert!(Calibrator::is_fitted(&calibrator));
        assert_eq!(
            Calibrator::method(&calibrator),
            CalibrationMethod::Temperature
        );

        let calibrated = Calibrator::transform(&calibrator, &y_prob).unwrap();
        assert_eq!(calibrated.len(), 6);

        // Calibrated probabilities should be valid
        for &p in calibrated.iter() {
            assert!(p > 0.0 && p < 1.0, "Probability out of bounds: {}", p);
        }
    }

    #[test]
    fn test_temperature_scaling_preserves_ranking() {
        // Temperature scaling should preserve the ranking of predictions
        let y_prob = Array2::from_shape_vec(
            (4, 3),
            vec![
                0.6, 0.3, 0.1, // Class 0 most likely
                0.2, 0.7, 0.1, // Class 1 most likely
                0.1, 0.2, 0.7, // Class 2 most likely
                0.5, 0.3, 0.2, // Class 0 most likely
            ],
        )
        .unwrap();
        let y_true = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.0]);

        let mut calibrator = TemperatureScalingCalibrator::new();
        MulticlassCalibrator::fit(&mut calibrator, &y_prob, &y_true).unwrap();

        let calibrated = MulticlassCalibrator::transform(&calibrator, &y_prob).unwrap();

        // Check that argmax is preserved
        for (i, (orig_row, cal_row)) in y_prob.rows().into_iter().zip(calibrated.rows()).enumerate()
        {
            let orig_argmax = orig_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let cal_argmax = cal_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            assert_eq!(
                orig_argmax, cal_argmax,
                "Sample {}: argmax changed from {} to {}",
                i, orig_argmax, cal_argmax
            );
        }
    }

    #[test]
    fn test_temperature_scaling_errors() {
        let mut calibrator = TemperatureScalingCalibrator::new();

        // Empty inputs
        let empty_prob = Array2::<f64>::zeros((0, 3));
        let empty_true = Array1::<f64>::zeros(0);
        assert!(MulticlassCalibrator::fit(&mut calibrator, &empty_prob, &empty_true).is_err());

        // Mismatched lengths
        let y_prob = Array2::from_shape_vec((3, 2), vec![0.5, 0.5, 0.6, 0.4, 0.7, 0.3]).unwrap();
        let y_true = Array1::from_vec(vec![0.0, 1.0]);
        assert!(MulticlassCalibrator::fit(&mut calibrator, &y_prob, &y_true).is_err());

        // Invalid class label
        let y_prob = Array2::from_shape_vec((3, 2), vec![0.5, 0.5, 0.6, 0.4, 0.7, 0.3]).unwrap();
        let y_true = Array1::from_vec(vec![0.0, 1.0, 5.0]); // 5 is invalid
        assert!(MulticlassCalibrator::fit(&mut calibrator, &y_prob, &y_true).is_err());

        // Transform without fitting
        let unfitted = TemperatureScalingCalibrator::new();
        let y_prob = Array2::from_shape_vec((3, 2), vec![0.5, 0.5, 0.6, 0.4, 0.7, 0.3]).unwrap();
        assert!(MulticlassCalibrator::transform(&unfitted, &y_prob).is_err());
    }

    #[test]
    fn test_temperature_scaling_builder_pattern() {
        let calibrator = TemperatureScalingCalibrator::new()
            .with_max_iter(50)
            .with_tol(1e-8)
            .with_learning_rate(0.05);

        assert!(!MulticlassCalibrator::is_fitted(&calibrator));
        assert!(calibrator.temperature().is_none());
    }

    #[test]
    fn test_temperature_scaling_clone() {
        let y_prob =
            Array2::from_shape_vec((4, 2), vec![0.8, 0.2, 0.7, 0.3, 0.3, 0.7, 0.2, 0.8]).unwrap();
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut calibrator = TemperatureScalingCalibrator::new();
        MulticlassCalibrator::fit(&mut calibrator, &y_prob, &y_true).unwrap();

        // Clone via trait method
        let cloned = MulticlassCalibrator::clone_boxed(&calibrator);
        assert!(cloned.is_fitted());
        assert_eq!(cloned.method(), CalibrationMethod::Temperature);

        // Both should produce same results
        let result1 = MulticlassCalibrator::transform(&calibrator, &y_prob).unwrap();
        let result2 = cloned.transform(&y_prob).unwrap();

        for (p1, p2) in result1.iter().zip(result2.iter()) {
            assert!((p1 - p2).abs() < 1e-10, "Clone produced different results");
        }
    }

    #[test]
    fn test_calibration_method_display_temperature() {
        assert_eq!(format!("{}", CalibrationMethod::Temperature), "temperature");
    }

    #[test]
    fn test_calibrated_classifier_cv_with_temperature() {
        // Test using temperature scaling with CalibratedClassifierCV
        let x =
            Array2::from_shape_vec((60, 3), (0..180).map(|i| i as f64 / 60.0).collect()).unwrap();
        let y = Array1::from_iter((0..60).map(|i| if i < 30 { 0.0 } else { 1.0 }));

        let base_clf = GaussianNB::new();
        let mut calibrated = CalibratedClassifierCV::new(Box::new(base_clf))
            .with_method(CalibrationMethod::Temperature)
            .with_n_folds(3);

        calibrated.fit(&x, &y).unwrap();

        assert!(calibrated.is_fitted());
        assert_eq!(calibrated.method(), CalibrationMethod::Temperature);
        assert_eq!(calibrated.n_calibrators(), 3);

        // Test probability predictions
        let proba = calibrated.predict_proba(&x).unwrap();
        assert_eq!(proba.nrows(), 60);
        assert_eq!(proba.ncols(), 2);

        // Probabilities should sum to 1
        for row in proba.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row doesn't sum to 1: {}", sum);
        }

        // Probabilities should be in [0, 1]
        for &p in proba.iter() {
            assert!(p >= 0.0 && p <= 1.0, "Probability out of bounds: {}", p);
        }
    }

    #[test]
    fn test_temperature_scaling_multiclass() {
        // Test with 5 classes
        let n_samples = 50;
        let n_classes = 5;

        // Create random-ish probabilities
        let mut probs = Vec::with_capacity(n_samples * n_classes);
        for i in 0..n_samples {
            let mut row = vec![0.1, 0.15, 0.2, 0.25, 0.3];
            // Make the "correct" class more likely
            row[i % n_classes] += 0.4;
            // Normalize
            let sum: f64 = row.iter().sum();
            for p in &mut row {
                *p /= sum;
                probs.push(*p);
            }
        }

        let y_prob = Array2::from_shape_vec((n_samples, n_classes), probs).unwrap();
        let y_true = Array1::from_iter((0..n_samples).map(|i| (i % n_classes) as f64));

        let mut calibrator = TemperatureScalingCalibrator::new()
            .with_max_iter(100)
            .with_learning_rate(0.02);

        MulticlassCalibrator::fit(&mut calibrator, &y_prob, &y_true).unwrap();

        assert!(MulticlassCalibrator::is_fitted(&calibrator));
        let t = calibrator.temperature().unwrap();
        assert!(t > 0.0);

        let calibrated = MulticlassCalibrator::transform(&calibrator, &y_prob).unwrap();
        assert_eq!(calibrated.nrows(), n_samples);
        assert_eq!(calibrated.ncols(), n_classes);

        // Each row should sum to 1
        for row in calibrated.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row doesn't sum to 1: {}", sum);
        }
    }
}
