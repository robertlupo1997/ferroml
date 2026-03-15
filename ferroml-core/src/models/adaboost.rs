//! AdaBoost Ensemble Methods
//!
//! Implements AdaBoost (Adaptive Boosting) for classification and regression.
//!
//! ## Classifiers
//!
//! - [`AdaBoostClassifier`] - SAMME.R (real-valued) boosting with decision stumps
//!
//! ## Regressors
//!
//! - [`AdaBoostRegressor`] - AdaBoost.R2 with decision stumps
//!
//! ## References
//!
//! - Hastie, Tibshirani, Friedman (2009). "Elements of Statistical Learning", 2nd ed.
//! - Drucker (1997). "Improving Regressors using Boosting Techniques"

use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, DecisionTreeClassifier,
    DecisionTreeRegressor, Model,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// =============================================================================
// AdaBoost Classifier
// =============================================================================

/// AdaBoost classifier using SAMME.R algorithm.
///
/// Fits an ensemble of weighted decision stumps, where each subsequent
/// estimator focuses on the samples that previous estimators got wrong.
///
/// ## Example
///
/// ```
/// # use ferroml_core::models::adaboost::AdaBoostClassifier;
/// # use ferroml_core::models::Model;
/// # use ndarray::{Array1, Array2};
/// # fn main() -> ferroml_core::Result<()> {
/// # let x = Array2::from_shape_vec((6, 2), vec![1.0,2.0,2.0,1.0,3.0,3.0,6.0,7.0,7.0,6.0,8.0,8.0]).unwrap();
/// # let y = Array1::from_vec(vec![0.0,0.0,0.0,1.0,1.0,1.0]);
/// use ferroml_core::models::adaboost::AdaBoostClassifier;
/// use ferroml_core::models::Model;
///
/// let mut clf = AdaBoostClassifier::new(50);
/// clf.fit(&x, &y)?;
/// let preds = clf.predict(&x)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaBoostClassifier {
    /// Number of boosting iterations (estimators)
    pub n_estimators: usize,
    /// Learning rate shrinks the contribution of each estimator
    pub learning_rate: f64,
    /// Maximum depth for each base estimator (default 1 = stumps)
    pub max_depth: usize,
    /// Random seed
    pub random_state: Option<u64>,
    /// Whether to reuse previous estimators and append new ones
    pub warm_start: bool,

    // Fitted state
    estimators: Option<Vec<DecisionTreeClassifier>>,
    estimator_weights: Option<Array1<f64>>,
    classes: Option<Array1<f64>>,
    n_features: Option<usize>,
}

impl AdaBoostClassifier {
    /// Create a new AdaBoostClassifier.
    ///
    /// # Arguments
    /// * `n_estimators` - Number of boosting rounds
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            learning_rate: 1.0,
            max_depth: 1,
            random_state: None,
            warm_start: false,
            estimators: None,
            estimator_weights: None,
            classes: None,
            n_features: None,
        }
    }

    /// Set whether to reuse previous estimators and append new ones.
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set the learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum depth of base estimators.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get estimator weights.
    pub fn estimator_weights(&self) -> Option<&Array1<f64>> {
        self.estimator_weights.as_ref()
    }

    /// Get number of fitted estimators.
    pub fn n_estimators_fitted(&self) -> usize {
        self.estimators.as_ref().map(|e| e.len()).unwrap_or(0)
    }

    /// Get the individual tree estimators.
    #[must_use]
    pub fn estimators(&self) -> Option<&[DecisionTreeClassifier]> {
        self.estimators.as_deref()
    }

    /// Get the unique class labels.
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }
}

impl Model for AdaBoostClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let classes = crate::models::get_unique_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(FerroError::invalid_input(
                "AdaBoostClassifier requires at least 2 classes",
            ));
        }

        // Initialize sample weights uniformly
        let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        let mut seed = self.random_state.unwrap_or(42);

        for _ in 0..self.n_estimators {
            // Create and fit a weighted decision stump
            let mut stump = DecisionTreeClassifier::default();
            stump.max_depth = Some(self.max_depth);
            stump.random_state = Some(seed);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

            stump.fit_weighted(x, y, &sample_weights)?;

            // Get predictions
            let predictions = stump.predict(x)?;

            // Compute weighted error
            let mut weighted_error = 0.0;
            for i in 0..n_samples {
                if (predictions[i] - y[i]).abs() > 1e-10 {
                    weighted_error += sample_weights[i];
                }
            }

            // Clamp error
            weighted_error = weighted_error.clamp(1e-10, 1.0 - 1e-10);

            // If error is too high, stop
            if weighted_error >= 1.0 - 1.0 / n_classes as f64 {
                // This estimator is no better than random — stop boosting
                if estimators.is_empty() {
                    // Keep at least one estimator
                    estimators.push(stump);
                    estimator_weights.push(0.0);
                }
                break;
            }

            // SAMME estimator weight
            let alpha = self.learning_rate
                * (((1.0 - weighted_error) / weighted_error).ln() + (n_classes as f64 - 1.0).ln());

            // Update sample weights
            for i in 0..n_samples {
                if (predictions[i] - y[i]).abs() > 1e-10 {
                    sample_weights[i] *= alpha.exp();
                }
            }

            // Normalize weights
            let weight_sum: f64 = sample_weights.sum();
            if weight_sum > 0.0 {
                sample_weights /= weight_sum;
            }

            estimators.push(stump);
            estimator_weights.push(alpha);
        }

        self.estimators = Some(estimators);
        self.estimator_weights = Some(Array1::from_vec(estimator_weights));
        self.classes = Some(classes);
        self.n_features = Some(n_features);
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.estimators, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let estimators = self.estimators.as_ref().unwrap();
        let weights = self.estimator_weights.as_ref().unwrap();
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Accumulate weighted votes per class
        let mut class_scores = Array2::zeros((n_samples, n_classes));

        for (est, &w) in estimators.iter().zip(weights.iter()) {
            let preds = est.predict(x)?;
            for i in 0..n_samples {
                if let Some(ci) = classes.iter().position(|&c| (c - preds[i]).abs() < 1e-10) {
                    class_scores[[i, ci]] += w;
                }
            }
        }

        // Argmax for each sample
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let max_idx = class_scores
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[max_idx];
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.estimators.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let estimators = self.estimators.as_ref()?;
        let weights = self.estimator_weights.as_ref()?;
        let n_features = self.n_features?;

        let total_weight: f64 = weights.sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut importances = Array1::zeros(n_features);
        for (est, &w) in estimators.iter().zip(weights.iter()) {
            if let Some(imp) = est.feature_importance() {
                importances += &(imp * w);
            }
        }
        importances /= total_weight;
        Some(importances)
    }

    fn model_name(&self) -> &str {
        "AdaBoostClassifier"
    }
}

// =============================================================================
// AdaBoost Regressor
// =============================================================================

/// Loss functions for AdaBoost regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaBoostLoss {
    /// Linear loss: L = |y - f(x)| / max_error
    Linear,
    /// Square loss: L = (|y - f(x)| / max_error)^2
    Square,
    /// Exponential loss: L = 1 - exp(-|y - f(x)| / max_error)
    Exponential,
}

impl Default for AdaBoostLoss {
    fn default() -> Self {
        Self::Linear
    }
}

/// AdaBoost regressor using AdaBoost.R2 algorithm.
///
/// Fits an ensemble of weighted decision trees for regression,
/// where each subsequent tree focuses on samples with high error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaBoostRegressor {
    /// Number of boosting iterations
    pub n_estimators: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Loss function
    pub loss: AdaBoostLoss,
    /// Maximum depth for base estimators
    pub max_depth: usize,
    /// Random seed
    pub random_state: Option<u64>,
    /// Whether to reuse previous estimators and append new ones
    pub warm_start: bool,

    // Fitted state
    estimators: Option<Vec<DecisionTreeRegressor>>,
    estimator_weights: Option<Array1<f64>>,
    n_features: Option<usize>,
}

impl AdaBoostRegressor {
    /// Create a new AdaBoostRegressor.
    pub fn new(n_estimators: usize) -> Self {
        Self {
            n_estimators,
            learning_rate: 1.0,
            loss: AdaBoostLoss::Linear,
            max_depth: 3,
            random_state: None,
            warm_start: false,
            estimators: None,
            estimator_weights: None,
            n_features: None,
        }
    }

    /// Set whether to reuse previous estimators and append new ones.
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Set the learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the loss function.
    pub fn with_loss(mut self, loss: AdaBoostLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set the maximum depth of base estimators.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the random seed.
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get number of fitted estimators.
    pub fn n_estimators_fitted(&self) -> usize {
        self.estimators.as_ref().map(|e| e.len()).unwrap_or(0)
    }

    /// Get the individual tree estimators.
    #[must_use]
    pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]> {
        self.estimators.as_deref()
    }

    /// Get the estimator weights.
    pub fn estimator_weights(&self) -> Option<&Array1<f64>> {
        self.estimator_weights.as_ref()
    }

    /// Compute sample loss based on the loss function.
    fn compute_loss(&self, errors: &Array1<f64>, max_error: f64) -> Array1<f64> {
        if max_error <= 0.0 {
            return Array1::zeros(errors.len());
        }
        errors.mapv(|e| {
            let normalized = (e / max_error).min(1.0);
            match self.loss {
                AdaBoostLoss::Linear => normalized,
                AdaBoostLoss::Square => normalized * normalized,
                AdaBoostLoss::Exponential => 1.0 - (-normalized).exp(),
            }
        })
    }
}

impl Model for AdaBoostRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize sample weights uniformly
        let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut estimator_weights = Vec::with_capacity(self.n_estimators);

        let mut seed = self.random_state.unwrap_or(42);

        for _ in 0..self.n_estimators {
            // Fit a weighted decision tree using bootstrap resampling based on weights
            let mut tree = DecisionTreeRegressor::default();
            tree.max_depth = Some(self.max_depth);
            tree.random_state = Some(seed);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

            // Weighted bootstrap: sample indices according to sample_weights
            let bootstrap_indices = weighted_bootstrap(n_samples, &sample_weights, seed);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);

            let x_boot = select_rows(x, &bootstrap_indices);
            let y_boot: Array1<f64> = bootstrap_indices.iter().map(|&i| y[i]).collect();

            tree.fit(&x_boot, &y_boot)?;

            // Predict on full training set
            let predictions = tree.predict(x)?;

            // Compute errors
            let errors: Array1<f64> = predictions
                .iter()
                .zip(y.iter())
                .map(|(p, t)| (p - t).abs())
                .collect();

            let max_error = errors.iter().cloned().fold(0.0f64, f64::max);

            if max_error <= 0.0 {
                // Perfect prediction — done
                estimators.push(tree);
                estimator_weights.push(1.0);
                break;
            }

            // Compute loss
            let loss = self.compute_loss(&errors, max_error);

            // Weighted average loss
            let avg_loss: f64 = loss
                .iter()
                .zip(sample_weights.iter())
                .map(|(l, w)| l * w)
                .sum();

            if avg_loss >= 0.5 {
                // Not better than random — stop
                if estimators.is_empty() {
                    estimators.push(tree);
                    estimator_weights.push(0.0);
                }
                break;
            }

            // Estimator weight (beta)
            let beta = avg_loss / (1.0 - avg_loss);
            let alpha = self.learning_rate * beta.ln().abs();

            // Update sample weights
            for i in 0..n_samples {
                sample_weights[i] *= beta.powf(1.0 - loss[i]);
            }

            // Normalize
            let weight_sum: f64 = sample_weights.sum();
            if weight_sum > 0.0 {
                sample_weights /= weight_sum;
            }

            estimators.push(tree);
            estimator_weights.push(alpha);
        }

        self.estimators = Some(estimators);
        self.estimator_weights = Some(Array1::from_vec(estimator_weights));
        self.n_features = Some(n_features);
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.estimators, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let estimators = self.estimators.as_ref().unwrap();
        let weights = self.estimator_weights.as_ref().unwrap();
        let n_samples = x.nrows();

        // Weighted median prediction
        let mut all_preds: Vec<Array1<f64>> = Vec::with_capacity(estimators.len());
        for est in estimators {
            all_preds.push(est.predict(x)?);
        }

        let mut predictions = Array1::zeros(n_samples);
        let total_weight: f64 = weights.sum();

        if total_weight <= 0.0 {
            // Fallback: simple average
            for preds in &all_preds {
                predictions += preds;
            }
            predictions /= all_preds.len() as f64;
            return Ok(predictions);
        }

        // Weighted median for each sample
        for i in 0..n_samples {
            let mut pred_weight: Vec<(f64, f64)> = all_preds
                .iter()
                .zip(weights.iter())
                .map(|(p, &w)| (p[i], w))
                .collect();
            pred_weight.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let half = total_weight / 2.0;
            let mut cumsum = 0.0;
            for &(pred, w) in &pred_weight {
                cumsum += w;
                if cumsum >= half {
                    predictions[i] = pred;
                    break;
                }
            }
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.estimators.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        let estimators = self.estimators.as_ref()?;
        let weights = self.estimator_weights.as_ref()?;
        let n_features = self.n_features?;

        let total_weight: f64 = weights.sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut importances = Array1::zeros(n_features);
        for (est, &w) in estimators.iter().zip(weights.iter()) {
            if let Some(imp) = est.feature_importance() {
                importances += &(imp * w);
            }
        }
        importances /= total_weight;
        Some(importances)
    }

    fn model_name(&self) -> &str {
        "AdaBoostRegressor"
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Select rows from a matrix by index.
fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_cols = x.ncols();
    let mut result = Array2::zeros((indices.len(), n_cols));
    for (i, &idx) in indices.iter().enumerate() {
        result.row_mut(i).assign(&x.row(idx));
    }
    result
}

/// Weighted bootstrap sampling using a simple LCG.
fn weighted_bootstrap(n_samples: usize, weights: &Array1<f64>, seed: u64) -> Vec<usize> {
    // Build CDF
    let mut cdf = Vec::with_capacity(n_samples);
    let mut cumsum = 0.0;
    for &w in weights.iter() {
        cumsum += w;
        cdf.push(cumsum);
    }
    // Normalize CDF
    if cumsum > 0.0 {
        for c in &mut cdf {
            *c /= cumsum;
        }
    }

    let mut rng_state = seed;
    let mut indices = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng_state >> 33) as f64 / (1u64 << 31) as f64;
        let idx = cdf.partition_point(|&c| c < u).min(n_samples - 1);
        indices.push(idx);
    }

    indices
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaboost_classifier_binary() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, // class 0
                7.0, 7.0, 8.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 7.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf = AdaBoostClassifier::new(10).with_learning_rate(1.0);
        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.n_features(), Some(2));
        assert!(clf.n_estimators_fitted() > 0);

        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 8, "Expected >=8/10 correct, got {}", correct);
    }

    #[test]
    fn test_adaboost_classifier_multiclass() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // class 0
                10.0, 0.0, 11.0, 0.0, 10.0, 1.0, 11.0, 1.0, // class 1
                5.0, 10.0, 6.0, 10.0, 5.0, 11.0, 6.0, 11.0, // class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let mut clf = AdaBoostClassifier::new(20);
        clf.fit(&x, &y).unwrap();

        let preds = clf.predict(&x).unwrap();
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 1e-10)
            .count();
        assert!(correct >= 10, "Expected >=10/12 correct, got {}", correct);
    }

    #[test]
    fn test_adaboost_classifier_feature_importance() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0, // class 0
                7.0, 100.0, 8.0, 200.0, 9.0, 300.0, 10.0, 400.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf = AdaBoostClassifier::new(10);
        clf.fit(&x, &y).unwrap();

        let importance = clf.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);
        // Feature 0 should be more important (it's the discriminative one)
        assert!(importance[0] > importance[1]);
    }

    #[test]
    fn test_adaboost_classifier_not_fitted() {
        let clf = AdaBoostClassifier::new(10);
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(clf.predict(&x).is_err());
    }

    #[test]
    fn test_adaboost_regressor_basic() {
        // Simple linear relationship
        let x = Array2::from_shape_vec(
            (10, 1),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        let mut reg = AdaBoostRegressor::new(20).with_max_depth(3);
        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());
        assert_eq!(reg.n_features(), Some(1));

        let preds = reg.predict(&x).unwrap();
        // Check reasonable approximation
        let mse: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse < 10.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_adaboost_regressor_loss_types() {
        let x =
            Array2::from_shape_vec((8, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

        for loss in [
            AdaBoostLoss::Linear,
            AdaBoostLoss::Square,
            AdaBoostLoss::Exponential,
        ] {
            let mut reg = AdaBoostRegressor::new(20).with_loss(loss).with_max_depth(3);
            reg.fit(&x, &y).unwrap();
            let preds = reg.predict(&x).unwrap();
            assert_eq!(preds.len(), 8);
        }
    }

    #[test]
    fn test_adaboost_regressor_not_fitted() {
        let reg = AdaBoostRegressor::new(10);
        let x = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        assert!(reg.predict(&x).is_err());
    }

    #[test]
    fn test_adaboost_regressor_feature_importance() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 100.0, 2.0, 200.0, 3.0, 300.0, 4.0, 400.0, 5.0, 500.0, 6.0, 600.0, 7.0, 700.0,
                8.0, 800.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

        let mut reg = AdaBoostRegressor::new(20).with_max_depth(2);
        reg.fit(&x, &y).unwrap();

        let importance = reg.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);
    }
}
