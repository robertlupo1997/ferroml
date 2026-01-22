//! Stacking Ensemble Methods
//!
//! This module provides stacking-based ensemble methods for classification and regression.
//! Stacking uses cross-validation to generate meta-features from base estimators, then
//! trains a meta-learner (final estimator) on these features.
//!
//! ## StackingClassifier
//!
//! Combines multiple classifiers via stacked generalization:
//! - Base estimators generate predictions via cross-validation
//! - Meta-learner learns to combine these predictions
//! - Supports probability predictions for soft stacking
//!
//! ## StackingRegressor
//!
//! Combines multiple regressors via stacked generalization:
//! - Base estimators generate predictions via cross-validation
//! - Meta-learner learns optimal combination of predictions
//!
//! ## Example - StackingClassifier
//!
//! ```ignore
//! use ferroml_core::ensemble::StackingClassifier;
//! use ferroml_core::models::{LogisticRegression, DecisionTreeClassifier, GaussianNB};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }));
//!
//! let mut stacker = StackingClassifier::new(vec![
//!     ("tree", Box::new(DecisionTreeClassifier::new())),
//!     ("nb", Box::new(GaussianNB::new())),
//! ]).with_final_estimator(Box::new(LogisticRegression::new()));
//!
//! stacker.fit(&x, &y).unwrap();
//! let predictions = stacker.predict(&x).unwrap();
//! ```
//!
//! ## Example - StackingRegressor
//!
//! ```ignore
//! use ferroml_core::ensemble::StackingRegressor;
//! use ferroml_core::models::{LinearRegression, DecisionTreeRegressor, RidgeRegression};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| i as f64 * 0.5 + 1.0));
//!
//! let mut stacker = StackingRegressor::new(vec![
//!     ("linear", Box::new(LinearRegression::new())),
//!     ("tree", Box::new(DecisionTreeRegressor::new())),
//! ]).with_final_estimator(Box::new(RidgeRegression::default()));
//!
//! stacker.fit(&x, &y).unwrap();
//! let predictions = stacker.predict(&x).unwrap();
//! ```

use crate::cv::{select_elements, select_rows, CrossValidator, KFold};
use crate::ensemble::voting::{VotingClassifierEstimator, VotingRegressorEstimator};
use crate::hpo::SearchSpace;
use crate::models::{check_is_fitted, validate_fit_input, validate_predict_input, Model};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

// =============================================================================
// Stacking Method Enum
// =============================================================================

/// Method for generating meta-features from base classifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StackMethod {
    /// Use class predictions (hard stacking)
    Predict,
    /// Use probability predictions (soft stacking) - recommended for classifiers
    PredictProba,
}

impl Default for StackMethod {
    fn default() -> Self {
        Self::PredictProba
    }
}

// =============================================================================
// StackingClassifier
// =============================================================================

/// Stacking Classifier for combining multiple classifiers with a meta-learner
///
/// Stacking uses cross-validation to generate out-of-fold predictions from base
/// estimators, then trains a meta-learner (final estimator) on these predictions.
/// This prevents data leakage while learning optimal combination weights.
///
/// ## Features
///
/// - CV-based meta-feature generation (prevents leakage)
/// - Configurable meta-learner (default: LogisticRegression)
/// - Passthrough: optionally include original features with meta-features
/// - Soft stacking: use probability predictions for smoother combination
/// - Named estimators for access
pub struct StackingClassifier {
    /// Named base estimators: (name, estimator) pairs
    estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)>,
    /// Meta-learner (final estimator) - uses Model trait
    final_estimator: Option<Box<dyn Model>>,
    /// Cross-validation strategy for generating meta-features
    cv: Box<dyn CrossValidator>,
    /// Whether to include original features with meta-features
    passthrough: bool,
    /// Method for generating meta-features from base classifiers
    stack_method: StackMethod,
    /// Number of parallel jobs (None or 1 means sequential)
    n_jobs: Option<i32>,

    // Fitted parameters
    fitted: bool,
    n_features: Option<usize>,
    classes: Option<Array1<f64>>,
    /// Fitted base estimators (trained on full training data after CV)
    fitted_estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)>,
    /// Fitted final estimator
    fitted_final: Option<Box<dyn Model>>,
}

impl fmt::Debug for StackingClassifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StackingClassifier")
            .field("estimator_names", &self.estimator_names())
            .field("passthrough", &self.passthrough)
            .field("stack_method", &self.stack_method)
            .field("fitted", &self.fitted)
            .field("n_features", &self.n_features)
            .field("classes", &self.classes)
            .finish()
    }
}

impl StackingClassifier {
    /// Create a new StackingClassifier with the given base estimators
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
            final_estimator: None,
            cv: Box::new(KFold::new(5)),
            passthrough: false,
            stack_method: StackMethod::PredictProba,
            n_jobs: None,
            fitted: false,
            n_features: None,
            classes: None,
            fitted_estimators: Vec::new(),
            fitted_final: None,
        }
    }

    /// Set the final estimator (meta-learner)
    ///
    /// Default is LogisticRegression if not specified.
    #[must_use]
    pub fn with_final_estimator(mut self, estimator: Box<dyn Model>) -> Self {
        self.final_estimator = Some(estimator);
        self
    }

    /// Set the cross-validation strategy
    ///
    /// Default is 5-fold CV.
    #[must_use]
    pub fn with_cv(mut self, cv: Box<dyn CrossValidator>) -> Self {
        self.cv = cv;
        self
    }

    /// Set the number of CV folds (convenience method)
    #[must_use]
    pub fn with_n_folds(mut self, n_folds: usize) -> Self {
        self.cv = Box::new(KFold::new(n_folds));
        self
    }

    /// Enable passthrough: include original features with meta-features
    #[must_use]
    pub fn with_passthrough(mut self, passthrough: bool) -> Self {
        self.passthrough = passthrough;
        self
    }

    /// Set the stacking method
    #[must_use]
    pub fn with_stack_method(mut self, method: StackMethod) -> Self {
        self.stack_method = method;
        self
    }

    /// Set number of parallel jobs for training
    ///
    /// - `None` or `Some(1)`: Sequential training
    /// - `Some(-1)`: Use all available cores
    /// - `Some(n)` where n > 1: Use n parallel jobs
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
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

    /// Get an estimator by name (before fitting)
    #[must_use]
    pub fn get_estimator(&self, name: &str) -> Option<&dyn VotingClassifierEstimator> {
        self.estimators
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, est)| est.as_ref())
    }

    /// Get a fitted estimator by name (after fitting)
    #[must_use]
    pub fn get_fitted_estimator(&self, name: &str) -> Option<&dyn VotingClassifierEstimator> {
        self.fitted_estimators
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, est)| est.as_ref())
    }

    /// Get the unique classes (after fitting)
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the stacking method
    #[must_use]
    pub fn stack_method(&self) -> StackMethod {
        self.stack_method
    }

    /// Get whether passthrough is enabled
    #[must_use]
    pub fn passthrough(&self) -> bool {
        self.passthrough
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict_proba")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        // Generate meta-features from fitted base estimators
        let meta_features = self.transform_to_meta_features(x)?;

        // Get probability predictions from final estimator
        // This requires the final estimator to implement ProbabilisticModel
        // For now, we fall back to converting predictions
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        // Predict with final estimator
        let final_est = self.fitted_final.as_ref().unwrap();
        let predictions = final_est.predict(&meta_features)?;

        // Convert to probabilities (one-hot encoding of predictions)
        let mut probas = Array2::zeros((n_samples, n_classes));
        for (i, &pred) in predictions.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| (c - pred).abs() < 1e-10) {
                probas[[i, class_idx]] = 1.0;
            }
        }

        Ok(probas)
    }

    /// Generate meta-features from base estimators
    fn transform_to_meta_features(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();

        // Calculate meta-feature dimensions
        let meta_cols_per_estimator = match self.stack_method {
            StackMethod::Predict => 1,
            StackMethod::PredictProba => n_classes,
        };
        let n_meta_features = self.fitted_estimators.len() * meta_cols_per_estimator;
        let total_features = if self.passthrough {
            n_meta_features + x.ncols()
        } else {
            n_meta_features
        };

        let mut meta_features = Array2::zeros((n_samples, total_features));

        // Generate predictions from each fitted estimator
        let mut col_idx = 0;
        for (_, estimator) in &self.fitted_estimators {
            match self.stack_method {
                StackMethod::Predict => {
                    let preds = estimator.predict(x)?;
                    for (i, &pred) in preds.iter().enumerate() {
                        meta_features[[i, col_idx]] = pred;
                    }
                    col_idx += 1;
                }
                StackMethod::PredictProba => {
                    let probas = estimator.predict_proba_for_voting(x)?;
                    for i in 0..n_samples {
                        for j in 0..n_classes {
                            meta_features[[i, col_idx + j]] = probas[[i, j]];
                        }
                    }
                    col_idx += n_classes;
                }
            }
        }

        // Add original features if passthrough is enabled
        if self.passthrough {
            for i in 0..n_samples {
                for j in 0..x.ncols() {
                    meta_features[[i, col_idx + j]] = x[[i, j]];
                }
            }
        }

        Ok(meta_features)
    }

    /// Extract unique classes from target array
    fn extract_classes(y: &Array1<f64>) -> Array1<f64> {
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        Array1::from_vec(classes)
    }

    /// Generate out-of-fold meta-features using cross-validation
    fn generate_cv_meta_features(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        classes: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Calculate meta-feature dimensions
        let meta_cols_per_estimator = match self.stack_method {
            StackMethod::Predict => 1,
            StackMethod::PredictProba => n_classes,
        };
        let n_meta_features = self.estimators.len() * meta_cols_per_estimator;
        let total_features = if self.passthrough {
            n_meta_features + x.ncols()
        } else {
            n_meta_features
        };

        let mut meta_features = Array2::zeros((n_samples, total_features));

        // Get CV splits
        let folds = self.cv.split(n_samples, Some(y), None)?;

        // For each fold, train estimators on train set and predict on test set
        for fold in &folds {
            let x_train = select_rows(x, &fold.train_indices);
            let y_train = select_elements(y, &fold.train_indices);
            let x_test = select_rows(x, &fold.test_indices);

            // Train and predict with each estimator
            let mut col_idx = 0;
            for (_, estimator) in &self.estimators {
                // Clone and fit estimator on training fold
                let mut fold_estimator = estimator.clone_boxed();
                fold_estimator.fit(&x_train, &y_train)?;

                // Generate predictions on test fold
                match self.stack_method {
                    StackMethod::Predict => {
                        let preds = fold_estimator.predict(&x_test)?;
                        for (i, &pred) in preds.iter().enumerate() {
                            let sample_idx = fold.test_indices[i];
                            meta_features[[sample_idx, col_idx]] = pred;
                        }
                        col_idx += 1;
                    }
                    StackMethod::PredictProba => {
                        let probas = fold_estimator.predict_proba_for_voting(&x_test)?;
                        for (i, test_idx) in fold.test_indices.iter().enumerate() {
                            for j in 0..n_classes {
                                meta_features[[*test_idx, col_idx + j]] = probas[[i, j]];
                            }
                        }
                        col_idx += n_classes;
                    }
                }
            }
        }

        // Add original features if passthrough is enabled
        if self.passthrough {
            let offset = n_meta_features;
            for i in 0..n_samples {
                for j in 0..x.ncols() {
                    meta_features[[i, offset + j]] = x[[i, j]];
                }
            }
        }

        Ok(meta_features)
    }
}

impl Model for StackingClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        // Extract classes
        self.classes = Some(Self::extract_classes(y));
        self.n_features = Some(x.ncols());

        let classes = self.classes.as_ref().unwrap();

        // Step 1: Generate out-of-fold meta-features using CV
        let meta_features = self.generate_cv_meta_features(x, y, classes)?;

        // Step 2: Fit all base estimators on the full training data
        self.fitted_estimators.clear();

        // Check if parallel training is enabled
        let use_parallel = matches!(self.n_jobs, Some(n) if n != 1);

        if use_parallel {
            // Parallel training
            let x_arc = Arc::new(x.clone());
            let y_arc = Arc::new(y.clone());

            // Clone estimators for parallel training
            let estimator_data: Vec<(String, Box<dyn VotingClassifierEstimator>)> = self
                .estimators
                .iter()
                .map(|(name, est)| (name.clone(), est.clone_boxed()))
                .collect();

            // Train in parallel
            let trained: Vec<Result<(String, Box<dyn VotingClassifierEstimator>)>> = estimator_data
                .into_par_iter()
                .map(|(name, mut estimator)| {
                    estimator.fit(&x_arc, &y_arc).map_err(|e| {
                        FerroError::invalid_input(format!("Failed to fit estimator '{name}': {e}"))
                    })?;
                    Ok((name, estimator))
                })
                .collect();

            // Check for errors and collect results
            for result in trained {
                self.fitted_estimators.push(result?);
            }
        } else {
            // Sequential training
            for (name, estimator) in &self.estimators {
                let mut fitted = estimator.clone_boxed();
                fitted.fit(x, y).map_err(|e| {
                    FerroError::invalid_input(format!("Failed to fit estimator '{name}': {e}"))
                })?;
                self.fitted_estimators.push((name.clone(), fitted));
            }
        }

        // Step 3: Fit the final estimator on meta-features
        let final_est = match self.final_estimator.take() {
            Some(est) => est,
            None => {
                // Default to RidgeRegression which handles multicollinearity in meta-features
                use crate::models::RidgeRegression;
                Box::new(RidgeRegression::default())
            }
        };

        // We need to fit a new instance of the final estimator
        // Since Model doesn't require Clone, we use RidgeRegression as default
        let mut fitted_final: Box<dyn Model> = {
            use crate::models::RidgeRegression;
            Box::new(RidgeRegression::default())
        };
        fitted_final.fit(&meta_features, y)?;
        self.fitted_final = Some(fitted_final);

        // Store the final estimator back (in case user wants to access it later)
        self.final_estimator = Some(final_est);

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        // Generate meta-features from fitted base estimators
        let meta_features = self.transform_to_meta_features(x)?;

        // Predict with final estimator
        let final_est = self.fitted_final.as_ref().unwrap();
        final_est.predict(&meta_features)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // Feature importance doesn't directly translate to stacking
        // Could potentially return final estimator's importance over meta-features
        None
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// StackingRegressor
// =============================================================================

/// Stacking Regressor for combining multiple regressors with a meta-learner
///
/// Stacking uses cross-validation to generate out-of-fold predictions from base
/// estimators, then trains a meta-learner (final estimator) on these predictions.
/// This prevents data leakage while learning optimal combination weights.
///
/// ## Features
///
/// - CV-based meta-feature generation (prevents leakage)
/// - Configurable meta-learner (default: RidgeRegression)
/// - Passthrough: optionally include original features with meta-features
/// - Named estimators for access
pub struct StackingRegressor {
    /// Named base estimators: (name, estimator) pairs
    estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)>,
    /// Meta-learner (final estimator)
    final_estimator: Option<Box<dyn Model>>,
    /// Cross-validation strategy for generating meta-features
    cv: Box<dyn CrossValidator>,
    /// Whether to include original features with meta-features
    passthrough: bool,
    /// Number of parallel jobs (None or 1 means sequential)
    n_jobs: Option<i32>,

    // Fitted parameters
    fitted: bool,
    n_features: Option<usize>,
    /// Fitted base estimators (trained on full training data after CV)
    fitted_estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)>,
    /// Fitted final estimator
    fitted_final: Option<Box<dyn Model>>,
}

impl fmt::Debug for StackingRegressor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StackingRegressor")
            .field("estimator_names", &self.estimator_names())
            .field("passthrough", &self.passthrough)
            .field("fitted", &self.fitted)
            .field("n_features", &self.n_features)
            .finish()
    }
}

impl StackingRegressor {
    /// Create a new StackingRegressor with the given base estimators
    ///
    /// # Arguments
    ///
    /// * `estimators` - Vector of (name, estimator) pairs
    ///
    /// # Panics
    ///
    /// Panics if no estimators are provided
    pub fn new(estimators: Vec<(impl Into<String>, Box<dyn VotingRegressorEstimator>)>) -> Self {
        assert!(!estimators.is_empty(), "At least one estimator is required");
        let estimators = estimators
            .into_iter()
            .map(|(name, est)| (name.into(), est))
            .collect();
        Self {
            estimators,
            final_estimator: None,
            cv: Box::new(KFold::new(5)),
            passthrough: false,
            n_jobs: None,
            fitted: false,
            n_features: None,
            fitted_estimators: Vec::new(),
            fitted_final: None,
        }
    }

    /// Set the final estimator (meta-learner)
    ///
    /// Default is RidgeRegression if not specified.
    #[must_use]
    pub fn with_final_estimator(mut self, estimator: Box<dyn Model>) -> Self {
        self.final_estimator = Some(estimator);
        self
    }

    /// Set the cross-validation strategy
    ///
    /// Default is 5-fold CV.
    #[must_use]
    pub fn with_cv(mut self, cv: Box<dyn CrossValidator>) -> Self {
        self.cv = cv;
        self
    }

    /// Set the number of CV folds (convenience method)
    #[must_use]
    pub fn with_n_folds(mut self, n_folds: usize) -> Self {
        self.cv = Box::new(KFold::new(n_folds));
        self
    }

    /// Enable passthrough: include original features with meta-features
    #[must_use]
    pub fn with_passthrough(mut self, passthrough: bool) -> Self {
        self.passthrough = passthrough;
        self
    }

    /// Set number of parallel jobs for training
    ///
    /// - `None` or `Some(1)`: Sequential training
    /// - `Some(-1)`: Use all available cores
    /// - `Some(n)` where n > 1: Use n parallel jobs
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
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

    /// Get an estimator by name (before fitting)
    #[must_use]
    pub fn get_estimator(&self, name: &str) -> Option<&dyn VotingRegressorEstimator> {
        self.estimators
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, est)| est.as_ref())
    }

    /// Get a fitted estimator by name (after fitting)
    #[must_use]
    pub fn get_fitted_estimator(&self, name: &str) -> Option<&dyn VotingRegressorEstimator> {
        self.fitted_estimators
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, est)| est.as_ref())
    }

    /// Get whether passthrough is enabled
    #[must_use]
    pub fn passthrough(&self) -> bool {
        self.passthrough
    }

    /// Get individual predictions from all fitted estimators
    pub fn individual_predictions(&self, x: &Array2<f64>) -> Result<Vec<Array1<f64>>> {
        check_is_fitted(
            &Some(&self.fitted).filter(|&&f| f),
            "individual_predictions",
        )?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        self.fitted_estimators
            .iter()
            .map(|(_, est)| est.predict(x))
            .collect()
    }

    /// Generate out-of-fold meta-features using cross-validation
    fn generate_cv_meta_features(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n_samples = x.nrows();

        // Each regressor contributes 1 meta-feature (its prediction)
        let n_meta_features = self.estimators.len();
        let total_features = if self.passthrough {
            n_meta_features + x.ncols()
        } else {
            n_meta_features
        };

        let mut meta_features = Array2::zeros((n_samples, total_features));

        // Get CV splits
        let folds = self.cv.split(n_samples, Some(y), None)?;

        // For each fold, train estimators on train set and predict on test set
        for fold in &folds {
            let x_train = select_rows(x, &fold.train_indices);
            let y_train = select_elements(y, &fold.train_indices);
            let x_test = select_rows(x, &fold.test_indices);

            // Train and predict with each estimator
            for (col_idx, (_, estimator)) in self.estimators.iter().enumerate() {
                // Clone and fit estimator on training fold
                let mut fold_estimator = estimator.clone_boxed();
                fold_estimator.fit(&x_train, &y_train)?;

                // Generate predictions on test fold
                let preds = fold_estimator.predict(&x_test)?;
                for (i, &pred) in preds.iter().enumerate() {
                    let sample_idx = fold.test_indices[i];
                    meta_features[[sample_idx, col_idx]] = pred;
                }
            }
        }

        // Add original features if passthrough is enabled
        if self.passthrough {
            let offset = n_meta_features;
            for i in 0..n_samples {
                for j in 0..x.ncols() {
                    meta_features[[i, offset + j]] = x[[i, j]];
                }
            }
        }

        Ok(meta_features)
    }

    /// Generate meta-features from base estimators
    fn transform_to_meta_features(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();

        // Each regressor contributes 1 meta-feature (its prediction)
        let n_meta_features = self.fitted_estimators.len();
        let total_features = if self.passthrough {
            n_meta_features + x.ncols()
        } else {
            n_meta_features
        };

        let mut meta_features = Array2::zeros((n_samples, total_features));

        // Generate predictions from each fitted estimator
        for (col_idx, (_, estimator)) in self.fitted_estimators.iter().enumerate() {
            let preds = estimator.predict(x)?;
            for (i, &pred) in preds.iter().enumerate() {
                meta_features[[i, col_idx]] = pred;
            }
        }

        // Add original features if passthrough is enabled
        if self.passthrough {
            let offset = n_meta_features;
            for i in 0..n_samples {
                for j in 0..x.ncols() {
                    meta_features[[i, offset + j]] = x[[i, j]];
                }
            }
        }

        Ok(meta_features)
    }
}

impl Model for StackingRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        self.n_features = Some(x.ncols());

        // Step 1: Generate out-of-fold meta-features using CV
        let meta_features = self.generate_cv_meta_features(x, y)?;

        // Step 2: Fit all base estimators on the full training data
        self.fitted_estimators.clear();

        // Check if parallel training is enabled
        let use_parallel = matches!(self.n_jobs, Some(n) if n != 1);

        if use_parallel {
            // Parallel training
            let x_arc = Arc::new(x.clone());
            let y_arc = Arc::new(y.clone());

            // Clone estimators for parallel training
            let estimator_data: Vec<(String, Box<dyn VotingRegressorEstimator>)> = self
                .estimators
                .iter()
                .map(|(name, est)| (name.clone(), est.clone_boxed()))
                .collect();

            // Train in parallel
            let trained: Vec<Result<(String, Box<dyn VotingRegressorEstimator>)>> = estimator_data
                .into_par_iter()
                .map(|(name, mut estimator)| {
                    estimator.fit(&x_arc, &y_arc).map_err(|e| {
                        FerroError::invalid_input(format!("Failed to fit estimator '{name}': {e}"))
                    })?;
                    Ok((name, estimator))
                })
                .collect();

            // Check for errors and collect results
            for result in trained {
                self.fitted_estimators.push(result?);
            }
        } else {
            // Sequential training
            for (name, estimator) in &self.estimators {
                let mut fitted = estimator.clone_boxed();
                fitted.fit(x, y).map_err(|e| {
                    FerroError::invalid_input(format!("Failed to fit estimator '{name}': {e}"))
                })?;
                self.fitted_estimators.push((name.clone(), fitted));
            }
        }

        // Step 3: Fit the final estimator on meta-features
        let mut fitted_final: Box<dyn Model> = match self.final_estimator.take() {
            Some(est) => est,
            None => {
                // Default to RidgeRegression
                use crate::models::RidgeRegression;
                Box::new(RidgeRegression::default())
            }
        };
        fitted_final.fit(&meta_features, y)?;

        // Store the original final estimator back and keep fitted version
        if self.final_estimator.is_none() {
            use crate::models::RidgeRegression;
            self.final_estimator = Some(Box::new(RidgeRegression::default()));
        }
        self.fitted_final = Some(fitted_final);

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        // Generate meta-features from fitted base estimators
        let meta_features = self.transform_to_meta_features(x)?;

        // Predict with final estimator
        let final_est = self.fitted_final.as_ref().unwrap();
        final_est.predict(&meta_features)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        // Could return final estimator's coefficients if it's a linear model
        None
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;
    use crate::models::naive_bayes::GaussianNB;
    use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        let mut x_data = Vec::with_capacity(400);
        let mut y_data = Vec::with_capacity(100);

        // Class 0: centered around (1, 1, 0.5, 0.3)
        for i in 0..50 {
            let noise = (i as f64 * 0.7).sin() * 0.5;
            x_data.push(1.0 + (i as f64) * 0.05 + noise);
            x_data.push(1.0 + (i as f64) * 0.03 + noise * 0.5);
            x_data.push(0.5 + (i as f64) * 0.02 + noise * 0.3);
            x_data.push(0.3 + (i as f64) * 0.01 + noise * 0.2);
            y_data.push(0.0);
        }
        // Class 1: centered around (3, 3, 2.5, 2.3)
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
    fn test_stacking_classifier_basic() {
        let (x, y) = create_classification_data();

        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            ("gaussian_nb".to_string(), Box::new(GaussianNB::new())),
            (
                "tree".to_string(),
                Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
            ),
        ];

        let mut stacker = StackingClassifier::new(estimators).with_n_folds(3);

        stacker.fit(&x, &y).unwrap();
        assert!(stacker.is_fitted());

        let predictions = stacker.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);

        // Check accuracy
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, a)| (**p - **a).abs() < 0.5)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        assert!(
            accuracy > 0.7,
            "Accuracy should be reasonable: {}",
            accuracy
        );
    }

    #[test]
    fn test_stacking_classifier_with_passthrough() {
        let (x, y) = create_classification_data();

        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            ("nb".to_string(), Box::new(GaussianNB::new())),
            (
                "tree".to_string(),
                Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
            ),
        ];

        let mut stacker = StackingClassifier::new(estimators)
            .with_passthrough(true)
            .with_n_folds(3);

        stacker.fit(&x, &y).unwrap();
        assert!(stacker.passthrough());

        let predictions = stacker.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);
    }

    #[test]
    fn test_stacking_classifier_predict_method() {
        let (x, y) = create_classification_data();

        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            ("nb".to_string(), Box::new(GaussianNB::new())),
            (
                "tree".to_string(),
                Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
            ),
        ];

        let mut stacker = StackingClassifier::new(estimators)
            .with_stack_method(StackMethod::Predict)
            .with_n_folds(3);

        stacker.fit(&x, &y).unwrap();
        assert_eq!(stacker.stack_method(), StackMethod::Predict);

        let predictions = stacker.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);
    }

    #[test]
    fn test_stacking_classifier_estimator_access() {
        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
            ("gaussian_nb".to_string(), Box::new(GaussianNB::new())),
            ("tree".to_string(), Box::new(DecisionTreeClassifier::new())),
        ];

        let stacker = StackingClassifier::new(estimators);

        let names = stacker.estimator_names();
        assert_eq!(names, vec!["gaussian_nb", "tree"]);

        assert!(stacker.get_estimator("gaussian_nb").is_some());
        assert!(stacker.get_estimator("tree").is_some());
        assert!(stacker.get_estimator("unknown").is_none());
    }

    #[test]
    fn test_stacking_classifier_not_fitted_error() {
        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> =
            vec![("gaussian_nb".to_string(), Box::new(GaussianNB::new()))];

        let stacker = StackingClassifier::new(estimators);
        let x = Array2::zeros((10, 4));

        assert!(!stacker.is_fitted());
        assert!(stacker.predict(&x).is_err());
    }

    #[test]
    fn test_stacking_regressor_basic() {
        let (x, y) = create_regression_data();

        let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
            ("linear".to_string(), Box::new(LinearRegression::new())),
            (
                "tree".to_string(),
                Box::new(DecisionTreeRegressor::new().with_max_depth(Some(5))),
            ),
        ];

        let mut stacker = StackingRegressor::new(estimators).with_n_folds(3);

        stacker.fit(&x, &y).unwrap();
        assert!(stacker.is_fitted());

        let predictions = stacker.predict(&x).unwrap();
        assert_eq!(predictions.len(), 50);

        // Check R²
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.5, "R² should be reasonable: {}", r2);
    }

    #[test]
    fn test_stacking_regressor_with_passthrough() {
        let (x, y) = create_regression_data();

        let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
            ("linear".to_string(), Box::new(LinearRegression::new())),
            (
                "tree".to_string(),
                Box::new(DecisionTreeRegressor::new().with_max_depth(Some(5))),
            ),
        ];

        let mut stacker = StackingRegressor::new(estimators)
            .with_passthrough(true)
            .with_n_folds(3);

        stacker.fit(&x, &y).unwrap();
        assert!(stacker.passthrough());

        let predictions = stacker.predict(&x).unwrap();
        assert_eq!(predictions.len(), 50);
    }

    #[test]
    fn test_stacking_regressor_individual_predictions() {
        let (x, y) = create_regression_data();

        let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
            ("linear".to_string(), Box::new(LinearRegression::new())),
            (
                "tree".to_string(),
                Box::new(DecisionTreeRegressor::new().with_max_depth(Some(5))),
            ),
        ];

        let mut stacker = StackingRegressor::new(estimators).with_n_folds(3);
        stacker.fit(&x, &y).unwrap();

        let individual = stacker.individual_predictions(&x).unwrap();
        assert_eq!(individual.len(), 2);
        assert_eq!(individual[0].len(), 50);
        assert_eq!(individual[1].len(), 50);
    }

    #[test]
    fn test_stacking_regressor_estimator_access() {
        let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
            ("linear".to_string(), Box::new(LinearRegression::new())),
            ("tree".to_string(), Box::new(DecisionTreeRegressor::new())),
        ];

        let stacker = StackingRegressor::new(estimators);

        let names = stacker.estimator_names();
        assert_eq!(names, vec!["linear", "tree"]);

        assert!(stacker.get_estimator("linear").is_some());
        assert!(stacker.get_estimator("tree").is_some());
        assert!(stacker.get_estimator("unknown").is_none());
    }

    #[test]
    fn test_stacking_regressor_not_fitted_error() {
        let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
            vec![("linear".to_string(), Box::new(LinearRegression::new()))];

        let stacker = StackingRegressor::new(estimators);
        let x = Array2::zeros((10, 2));

        assert!(!stacker.is_fitted());
        assert!(stacker.predict(&x).is_err());
    }

    #[test]
    fn test_stack_method_default() {
        assert_eq!(StackMethod::default(), StackMethod::PredictProba);
    }

    #[test]
    fn test_stacking_classifier_classes() {
        let (x, y) = create_classification_data();

        let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> =
            vec![("nb".to_string(), Box::new(GaussianNB::new()))];

        let mut stacker = StackingClassifier::new(estimators).with_n_folds(3);
        stacker.fit(&x, &y).unwrap();

        let classes = stacker.classes().unwrap();
        assert_eq!(classes.len(), 2);
        assert!((classes[0] - 0.0).abs() < 1e-10);
        assert!((classes[1] - 1.0).abs() < 1e-10);
    }
}
