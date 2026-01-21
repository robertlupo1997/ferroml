//! Voting Ensemble Methods
//!
//! This module provides voting-based ensemble methods for classification and regression.
//!
//! ## VotingClassifier
//!
//! Combines multiple classifiers via voting:
//! - **Hard voting**: Majority vote of predicted classes
//! - **Soft voting**: Weighted average of predicted probabilities
//!
//! ## VotingRegressor
//!
//! Combines multiple regressors by averaging their predictions, with optional weights.
//!
//! ## Example - VotingClassifier
//!
//! ```ignore
//! use ferroml_core::ensemble::VotingClassifier;
//! use ferroml_core::models::{LogisticRegression, DecisionTreeClassifier, Model};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }));
//!
//! let mut voter = VotingClassifier::new(vec![
//!     ("logistic", Box::new(LogisticRegression::new())),
//!     ("tree", Box::new(DecisionTreeClassifier::new())),
//! ]).with_soft_voting();
//!
//! voter.fit(&x, &y).unwrap();
//! let predictions = voter.predict(&x).unwrap();
//! ```
//!
//! ## Example - VotingRegressor
//!
//! ```ignore
//! use ferroml_core::ensemble::VotingRegressor;
//! use ferroml_core::models::{LinearRegression, DecisionTreeRegressor, Model};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| i as f64 * 0.5 + 1.0));
//!
//! let mut voter = VotingRegressor::new(vec![
//!     ("linear", Box::new(LinearRegression::new())),
//!     ("tree", Box::new(DecisionTreeRegressor::new())),
//! ]).with_weights(vec![2.0, 1.0]);
//!
//! voter.fit(&x, &y).unwrap();
//! let predictions = voter.predict(&x).unwrap();
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, Model, ProbabilisticModel,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Voting Method Enum
// =============================================================================

/// Voting method for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VotingMethod {
    /// Majority vote of predicted class labels
    Hard,
    /// Weighted average of predicted probabilities
    Soft,
}

impl Default for VotingMethod {
    fn default() -> Self {
        Self::Hard
    }
}

// =============================================================================
// Classifier Wrapper Trait
// =============================================================================

/// Trait for classifiers that can be used in VotingClassifier
///
/// This trait abstracts over different classifier types to allow heterogeneous
/// ensembles. Classifiers must implement Model and provide probability predictions.
pub trait VotingClassifierEstimator: Model {
    /// Predict class probabilities for soft voting
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>>;

    /// Clone the estimator into a box
    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator>;
}

// =============================================================================
// VotingClassifier
// =============================================================================

/// Voting Classifier for combining multiple classifiers
///
/// Combines predictions from multiple classifiers using either hard voting
/// (majority class) or soft voting (averaged probabilities).
///
/// ## Features
///
/// - Hard voting: Each classifier votes for a class, majority wins
/// - Soft voting: Averages class probabilities, highest probability wins
/// - Weighted voting: Assign different weights to different classifiers
/// - Named estimators: Access individual classifiers by name
pub struct VotingClassifier {
    /// Named estimators: (name, estimator) pairs
    estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)>,
    /// Voting method (hard or soft)
    voting: VotingMethod,
    /// Weights for each estimator (None means equal weights)
    weights: Option<Vec<f64>>,

    // Fitted parameters
    fitted: bool,
    n_features: Option<usize>,
    classes: Option<Array1<f64>>,
}

impl fmt::Debug for VotingClassifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VotingClassifier")
            .field("estimator_names", &self.estimator_names())
            .field("voting", &self.voting)
            .field("weights", &self.weights)
            .field("fitted", &self.fitted)
            .field("n_features", &self.n_features)
            .field("classes", &self.classes)
            .finish()
    }
}

impl VotingClassifier {
    /// Create a new VotingClassifier with the given estimators
    ///
    /// # Arguments
    ///
    /// * `estimators` - Vector of (name, estimator) pairs
    ///
    /// # Panics
    ///
    /// Panics if no estimators are provided
    pub fn new(estimators: Vec<(impl Into<String>, Box<dyn VotingClassifierEstimator>)>) -> Self {
        assert!(!estimators.is_empty(), "At least one estimator is required");
        let estimators = estimators
            .into_iter()
            .map(|(name, est)| (name.into(), est))
            .collect();
        Self {
            estimators,
            voting: VotingMethod::Hard,
            weights: None,
            fitted: false,
            n_features: None,
            classes: None,
        }
    }

    /// Set hard voting method (majority vote)
    #[must_use]
    pub fn with_hard_voting(mut self) -> Self {
        self.voting = VotingMethod::Hard;
        self
    }

    /// Set soft voting method (probability averaging)
    #[must_use]
    pub fn with_soft_voting(mut self) -> Self {
        self.voting = VotingMethod::Soft;
        self
    }

    /// Set voting method
    #[must_use]
    pub fn with_voting(mut self, voting: VotingMethod) -> Self {
        self.voting = voting;
        self
    }

    /// Set weights for estimators
    ///
    /// # Panics
    ///
    /// Panics if the number of weights doesn't match the number of estimators
    #[must_use]
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        assert_eq!(
            weights.len(),
            self.estimators.len(),
            "Number of weights must match number of estimators"
        );
        self.weights = Some(weights);
        self
    }

    /// Get the voting method
    #[must_use]
    pub fn voting_method(&self) -> VotingMethod {
        self.voting
    }

    /// Get the estimator names
    #[must_use]
    pub fn estimator_names(&self) -> Vec<&str> {
        self.estimators
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get an estimator by name
    #[must_use]
    pub fn get_estimator(&self, name: &str) -> Option<&dyn VotingClassifierEstimator> {
        self.estimators
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, est)| est.as_ref())
    }

    /// Get the unique classes (after fitting)
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the weights (if set)
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Predict class probabilities (soft voting probabilities)
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict_proba")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        // Get weights (normalize if provided)
        let weights = self.get_normalized_weights();

        // Aggregate probabilities from all estimators
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (idx, (_, estimator)) in self.estimators.iter().enumerate() {
            let est_proba = estimator.predict_proba_for_voting(x)?;
            let weight = weights[idx];
            probas += &(est_proba * weight);
        }

        Ok(probas)
    }

    /// Get normalized weights
    fn get_normalized_weights(&self) -> Vec<f64> {
        match &self.weights {
            Some(w) => {
                let sum: f64 = w.iter().sum();
                w.iter().map(|&wi| wi / sum).collect()
            }
            None => {
                let n = self.estimators.len();
                vec![1.0 / n as f64; n]
            }
        }
    }

    /// Hard voting: count votes for each class
    fn predict_hard(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();
        let weights = self.get_normalized_weights();

        let mut votes = Array2::zeros((n_samples, n_classes));

        for (idx, (_, estimator)) in self.estimators.iter().enumerate() {
            let preds = estimator.predict(x)?;
            let weight = weights[idx];

            for (i, &pred) in preds.iter().enumerate() {
                // Find class index
                if let Some(class_idx) = classes.iter().position(|&c| (c - pred).abs() < 1e-10) {
                    votes[[i, class_idx]] += weight;
                }
            }
        }

        // Select class with most votes
        let predictions = votes
            .rows()
            .into_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| {
                        a.partial_cmp(b).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Soft voting: use averaged probabilities
    fn predict_soft(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes.as_ref().unwrap();

        // Select class with highest probability
        let predictions = probas
            .rows()
            .into_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| {
                        a.partial_cmp(b).unwrap()
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Extract unique classes from target array
    fn extract_classes(y: &Array1<f64>) -> Array1<f64> {
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        Array1::from_vec(classes)
    }
}

impl Model for VotingClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Extract classes
        self.classes = Some(Self::extract_classes(y));
        self.n_features = Some(x.ncols());

        // Fit all estimators
        for (name, estimator) in &mut self.estimators {
            estimator.fit(x, y).map_err(|e| {
                FerroError::invalid_input(format!("Failed to fit estimator '{name}': {e}"))
            })?;
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        match self.voting {
            VotingMethod::Hard => self.predict_hard(x),
            VotingMethod::Soft => self.predict_soft(x),
        }
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        if !self.fitted {
            return None;
        }

        let n_features = self.n_features?;
        let weights = self.get_normalized_weights();

        // Aggregate feature importance from estimators that support it
        let mut importance = Array1::zeros(n_features);
        let mut total_weight = 0.0;

        for (idx, (_, estimator)) in self.estimators.iter().enumerate() {
            if let Some(est_importance) = estimator.feature_importance() {
                let weight = weights[idx];
                importance += &(est_importance * weight);
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            Some(importance / total_weight)
        } else {
            None
        }
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// VotingRegressor
// =============================================================================

/// Voting Regressor for combining multiple regressors
///
/// Combines predictions from multiple regressors by averaging (with optional weights).
///
/// ## Features
///
/// - Simple averaging of predictions
/// - Weighted averaging with custom weights
/// - Named estimators for access
pub struct VotingRegressor {
    /// Named estimators: (name, estimator) pairs
    estimators: Vec<(String, Box<dyn Model>)>,
    /// Weights for each estimator (None means equal weights)
    weights: Option<Vec<f64>>,

    // Fitted parameters
    fitted: bool,
    n_features: Option<usize>,
}

impl fmt::Debug for VotingRegressor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VotingRegressor")
            .field("estimator_names", &self.estimator_names())
            .field("weights", &self.weights)
            .field("fitted", &self.fitted)
            .field("n_features", &self.n_features)
            .finish()
    }
}

impl VotingRegressor {
    /// Create a new VotingRegressor with the given estimators
    ///
    /// # Arguments
    ///
    /// * `estimators` - Vector of (name, estimator) pairs
    ///
    /// # Panics
    ///
    /// Panics if no estimators are provided
    pub fn new(estimators: Vec<(impl Into<String>, Box<dyn Model>)>) -> Self {
        assert!(!estimators.is_empty(), "At least one estimator is required");
        let estimators = estimators
            .into_iter()
            .map(|(name, est)| (name.into(), est))
            .collect();
        Self {
            estimators,
            weights: None,
            fitted: false,
            n_features: None,
        }
    }

    /// Set weights for estimators
    ///
    /// # Panics
    ///
    /// Panics if the number of weights doesn't match the number of estimators
    #[must_use]
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        assert_eq!(
            weights.len(),
            self.estimators.len(),
            "Number of weights must match number of estimators"
        );
        self.weights = Some(weights);
        self
    }

    /// Get the estimator names
    #[must_use]
    pub fn estimator_names(&self) -> Vec<&str> {
        self.estimators
            .iter()
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get an estimator by name
    #[must_use]
    pub fn get_estimator(&self, name: &str) -> Option<&dyn Model> {
        self.estimators
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, est)| est.as_ref())
    }

    /// Get the weights (if set)
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Get normalized weights
    fn get_normalized_weights(&self) -> Vec<f64> {
        match &self.weights {
            Some(w) => {
                let sum: f64 = w.iter().sum();
                w.iter().map(|&wi| wi / sum).collect()
            }
            None => {
                let n = self.estimators.len();
                vec![1.0 / n as f64; n]
            }
        }
    }

    /// Get individual predictions from all estimators
    pub fn individual_predictions(&self, x: &Array2<f64>) -> Result<Vec<Array1<f64>>> {
        check_is_fitted(
            &Some(&self.fitted).filter(|&&f| f),
            "individual_predictions",
        )?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        self.estimators
            .iter()
            .map(|(_, est)| est.predict(x))
            .collect()
    }
}

impl Model for VotingRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        self.n_features = Some(x.ncols());

        // Fit all estimators
        for (name, estimator) in &mut self.estimators {
            estimator.fit(x, y).map_err(|e| {
                FerroError::invalid_input(format!("Failed to fit estimator '{name}': {e}"))
            })?;
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let weights = self.get_normalized_weights();
        let n_samples = x.nrows();

        // Weighted average of predictions
        let mut predictions = Array1::zeros(n_samples);

        for (idx, (_, estimator)) in self.estimators.iter().enumerate() {
            let est_pred = estimator.predict(x)?;
            let weight = weights[idx];
            predictions += &(est_pred * weight);
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        if !self.fitted {
            return None;
        }

        let n_features = self.n_features?;
        let weights = self.get_normalized_weights();

        // Aggregate feature importance from estimators that support it
        let mut importance = Array1::zeros(n_features);
        let mut total_weight = 0.0;

        for (idx, (_, estimator)) in self.estimators.iter().enumerate() {
            if let Some(est_importance) = estimator.feature_importance() {
                let weight = weights[idx];
                importance += &(est_importance * weight);
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            Some(importance / total_weight)
        } else {
            None
        }
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// VotingClassifierEstimator Implementations
// =============================================================================

// Implement VotingClassifierEstimator for common classifiers
// This allows them to be used in VotingClassifier

use crate::models::boosting::GradientBoostingClassifier;
use crate::models::forest::RandomForestClassifier;
use crate::models::hist_boosting::HistGradientBoostingClassifier;
use crate::models::knn::KNeighborsClassifier;
use crate::models::logistic::LogisticRegression;
use crate::models::naive_bayes::{BernoulliNB, GaussianNB, MultinomialNB};
use crate::models::svm::SVC;
use crate::models::tree::DecisionTreeClassifier;

impl VotingClassifierEstimator for LogisticRegression {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for DecisionTreeClassifier {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for RandomForestClassifier {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for GaussianNB {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for MultinomialNB {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for BernoulliNB {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for KNeighborsClassifier {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for SVC {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for GradientBoostingClassifier {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

impl VotingClassifierEstimator for HistGradientBoostingClassifier {
    fn predict_proba_for_voting(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba(x)
    }

    fn clone_boxed(&self) -> Box<dyn VotingClassifierEstimator> {
        Box::new(self.clone())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Data with some overlap to avoid perfect separation issues with logistic regression
        let mut x_data = Vec::with_capacity(400);
        let mut y_data = Vec::with_capacity(100);

        // Class 0: centered around (1, 1, 0.5, 0.3) with noise
        for i in 0..50 {
            let noise = (i as f64 * 0.7).sin() * 0.5;
            x_data.push(1.0 + (i as f64) * 0.05 + noise);
            x_data.push(1.0 + (i as f64) * 0.03 + noise * 0.5);
            x_data.push(0.5 + (i as f64) * 0.02 + noise * 0.3);
            x_data.push(0.3 + (i as f64) * 0.01 + noise * 0.2);
            y_data.push(0.0);
        }
        // Class 1: centered around (3, 3, 2.5, 2.3) with noise - closer to class 0 for overlap
        for i in 0..50 {
            let noise = (i as f64 * 0.7).cos() * 0.5;
            x_data.push(3.0 + (i as f64) * 0.05 + noise);
            x_data.push(3.0 + (i as f64) * 0.03 + noise * 0.5);
            x_data.push(2.5 + (i as f64) * 0.02 + noise * 0.3);
            x_data.push(2.3 + (i as f64) * 0.01 + noise * 0.2);
            y_data.push(1.0);
        }

        let x = Array2::from_shape_vec((100, 4), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        (x, y)
    }

    fn create_regression_data() -> (Array2<f64>, Array1<f64>) {
        let n = 50;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x1 = i as f64 / 10.0;
            let x2 = (i as f64).sin();
            x_data.push(x1);
            x_data.push(x2);
            // y = 2*x1 + 0.5*x2 + noise
            y_data.push(2.0 * x1 + 0.5 * x2 + 0.1 * (i as f64 % 3.0 - 1.0));
        }

        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        (x, y)
    }

    #[test]
    fn test_voting_classifier_hard_voting() {
        let (x, y) = create_classification_data();

        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            ("gaussian_nb".to_string(), Box::new(GaussianNB::new())),
            (
                "tree".to_string(),
                Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
            ),
        ];

        let mut voter = VotingClassifier::new(estimators).with_hard_voting();

        voter.fit(&x, &y).unwrap();
        assert!(voter.is_fitted());

        let predictions = voter.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);

        // Check accuracy (should be high for linearly separable data)
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, a)| (**p - **a).abs() < 0.5)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.8, "Accuracy should be high: {}", accuracy);
    }

    #[test]
    fn test_voting_classifier_soft_voting() {
        let (x, y) = create_classification_data();

        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            (
                "tree".to_string(),
                Box::new(DecisionTreeClassifier::new().with_max_depth(Some(5))),
            ),
            ("gaussian_nb".to_string(), Box::new(GaussianNB::new())),
        ];

        let mut voter = VotingClassifier::new(estimators).with_soft_voting();

        voter.fit(&x, &y).unwrap();

        let predictions = voter.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);

        // Test predict_proba
        let probas = voter.predict_proba(&x).unwrap();
        assert_eq!(probas.nrows(), 100);
        assert_eq!(probas.ncols(), 2); // 2 classes

        // Probabilities should sum to 1
        for row in probas.rows() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities should sum to 1: {}",
                sum
            );
        }
    }

    #[test]
    fn test_voting_classifier_weighted() {
        let (x, y) = create_classification_data();

        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            ("gaussian_nb".to_string(), Box::new(GaussianNB::new())),
            ("tree".to_string(), Box::new(DecisionTreeClassifier::new())),
        ];

        let mut voter = VotingClassifier::new(estimators)
            .with_soft_voting()
            .with_weights(vec![2.0, 1.0]); // Give more weight to gaussian_nb

        voter.fit(&x, &y).unwrap();

        let predictions = voter.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);

        // Check weights are stored correctly
        let weights = voter.weights().unwrap();
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 2.0).abs() < 1e-10);
        assert!((weights[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_voting_classifier_estimator_access() {
        let (x, y) = create_classification_data();

        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            ("gaussian_nb".to_string(), Box::new(GaussianNB::new())),
            ("tree".to_string(), Box::new(DecisionTreeClassifier::new())),
        ];

        let mut voter = VotingClassifier::new(estimators);
        voter.fit(&x, &y).unwrap();

        // Check estimator names
        let names = voter.estimator_names();
        assert_eq!(names, vec!["gaussian_nb", "tree"]);

        // Check get_estimator
        assert!(voter.get_estimator("gaussian_nb").is_some());
        assert!(voter.get_estimator("tree").is_some());
        assert!(voter.get_estimator("unknown").is_none());

        // Check classes
        let classes = voter.classes().unwrap();
        assert_eq!(classes.len(), 2);
    }

    #[test]
    fn test_voting_regressor() {
        let (x, y) = create_regression_data();

        let estimators: Vec<(String, Box<dyn Model>)> = vec![
            ("linear".to_string(), Box::new(LinearRegression::new())),
            (
                "tree".to_string(),
                Box::new(crate::models::tree::DecisionTreeRegressor::new().with_max_depth(Some(5))),
            ),
        ];

        let mut voter = VotingRegressor::new(estimators);

        voter.fit(&x, &y).unwrap();
        assert!(voter.is_fitted());

        let predictions = voter.predict(&x).unwrap();
        assert_eq!(predictions.len(), 50);

        // Check that predictions are reasonable (R² > 0)
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.5, "R² should be positive: {}", r2);
    }

    #[test]
    fn test_voting_regressor_weighted() {
        let (x, y) = create_regression_data();

        let estimators: Vec<(String, Box<dyn Model>)> = vec![
            ("linear".to_string(), Box::new(LinearRegression::new())),
            (
                "tree".to_string(),
                Box::new(crate::models::tree::DecisionTreeRegressor::new()),
            ),
        ];

        let mut voter = VotingRegressor::new(estimators).with_weights(vec![3.0, 1.0]);

        voter.fit(&x, &y).unwrap();

        let predictions = voter.predict(&x).unwrap();
        assert_eq!(predictions.len(), 50);

        // Check normalized weights sum to 1
        let weights = voter.get_normalized_weights();
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_voting_regressor_individual_predictions() {
        let (x, y) = create_regression_data();

        let estimators: Vec<(String, Box<dyn Model>)> = vec![
            ("linear".to_string(), Box::new(LinearRegression::new())),
            (
                "tree".to_string(),
                Box::new(crate::models::tree::DecisionTreeRegressor::new()),
            ),
        ];

        let mut voter = VotingRegressor::new(estimators);
        voter.fit(&x, &y).unwrap();

        let individual = voter.individual_predictions(&x).unwrap();
        assert_eq!(individual.len(), 2);
        assert_eq!(individual[0].len(), 50);
        assert_eq!(individual[1].len(), 50);
    }

    #[test]
    fn test_voting_regressor_estimator_access() {
        let estimators: Vec<(String, Box<dyn Model>)> = vec![
            ("linear".to_string(), Box::new(LinearRegression::new())),
            (
                "tree".to_string(),
                Box::new(crate::models::tree::DecisionTreeRegressor::new()),
            ),
        ];

        let voter = VotingRegressor::new(estimators);

        let names = voter.estimator_names();
        assert_eq!(names, vec!["linear", "tree"]);

        assert!(voter.get_estimator("linear").is_some());
        assert!(voter.get_estimator("tree").is_some());
        assert!(voter.get_estimator("unknown").is_none());
    }

    #[test]
    fn test_voting_classifier_not_fitted_error() {
        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> =
            vec![("gaussian_nb".to_string(), Box::new(GaussianNB::new()))];

        let voter = VotingClassifier::new(estimators);
        let x = Array2::zeros((10, 4));

        assert!(!voter.is_fitted());
        assert!(voter.predict(&x).is_err());
        assert!(voter.predict_proba(&x).is_err());
    }

    #[test]
    fn test_voting_regressor_not_fitted_error() {
        let estimators: Vec<(String, Box<dyn Model>)> =
            vec![("linear".to_string(), Box::new(LinearRegression::new()))];

        let voter = VotingRegressor::new(estimators);
        let x = Array2::zeros((10, 2));

        assert!(!voter.is_fitted());
        assert!(voter.predict(&x).is_err());
    }

    #[test]
    fn test_voting_method_default() {
        assert_eq!(VotingMethod::default(), VotingMethod::Hard);
    }

    #[test]
    fn test_voting_classifier_feature_importance() {
        let (x, y) = create_classification_data();

        // Use models that support feature importance
        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            ("tree".to_string(), Box::new(DecisionTreeClassifier::new())),
            (
                "forest".to_string(),
                Box::new(RandomForestClassifier::new().with_n_estimators(10)),
            ),
        ];

        let mut voter = VotingClassifier::new(estimators);
        voter.fit(&x, &y).unwrap();

        let importance = voter.feature_importance();
        assert!(importance.is_some());
        let imp = importance.unwrap();
        assert_eq!(imp.len(), 4);

        // Importance should sum to approximately 1
        let sum: f64 = imp.iter().sum();
        assert!(sum > 0.0, "Feature importance should be positive");
    }
}
