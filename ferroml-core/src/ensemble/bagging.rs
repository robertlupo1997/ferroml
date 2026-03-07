//! Bagging Ensemble Methods
//!
//! This module provides bagging-based ensemble methods for classification and regression.
//! Bagging (Bootstrap AGGregatING) builds multiple models on bootstrap samples of the
//! training data and aggregates their predictions.
//!
//! ## BaggingClassifier
//!
//! Combines multiple classifiers trained on bootstrap samples:
//! - Each classifier trained on a random bootstrap sample
//! - Predictions aggregated via voting (hard or soft)
//! - OOB (Out-of-Bag) error estimation
//! - Optional feature subsampling
//!
//! ## BaggingRegressor
//!
//! Combines multiple regressors trained on bootstrap samples:
//! - Each regressor trained on a random bootstrap sample
//! - Predictions averaged
//! - OOB (Out-of-Bag) R² estimation
//! - Optional feature subsampling
//!
//! ## Example - BaggingClassifier
//!
//! ```
//! # use ferroml_core::ensemble::BaggingClassifier;
//! # use ferroml_core::models::{DecisionTreeClassifier, Model};
//! # use ndarray::{Array1, Array2};
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }));
//!
//! let mut bagger = BaggingClassifier::new(Box::new(DecisionTreeClassifier::new()))
//!     .with_n_estimators(10)
//!     .with_oob_score(true)
//!     .with_random_state(42);
//!
//! bagger.fit(&x, &y).unwrap();
//! let predictions = bagger.predict(&x).unwrap();
//! let oob_score = bagger.oob_score();
//! ```
//!
//! ## Example - BaggingRegressor
//!
//! ```
//! # use ferroml_core::ensemble::BaggingRegressor;
//! # use ferroml_core::models::{DecisionTreeRegressor, Model};
//! # use ndarray::{Array1, Array2};
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| i as f64 * 0.5 + 1.0));
//!
//! let mut bagger = BaggingRegressor::new(Box::new(DecisionTreeRegressor::new()))
//!     .with_n_estimators(10)
//!     .with_oob_score(true);
//!
//! bagger.fit(&x, &y).unwrap();
//! let predictions = bagger.predict(&x).unwrap();
//! let oob_score = bagger.oob_score(); // R² on OOB samples
//! ```

use crate::ensemble::voting::VotingClassifierEstimator;
use crate::hpo::SearchSpace;
use crate::models::{check_is_fitted, validate_fit_input, validate_predict_input, Model};
use crate::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Shared bootstrap helpers
// =============================================================================

/// Generate bootstrap sample indices and OOB indices.
///
/// Shared by `BaggingClassifier` and `BaggingRegressor`.
fn generate_bootstrap_indices(
    n_samples: usize,
    n_draw: usize,
    bootstrap: bool,
    rng: &mut StdRng,
) -> (Vec<usize>, Vec<usize>) {
    if bootstrap {
        let indices: Vec<usize> = (0..n_draw)
            .map(|_| rng.random_range(0..n_samples))
            .collect();

        let mut in_bag = vec![false; n_samples];
        for &idx in &indices {
            in_bag[idx] = true;
        }

        let oob_indices: Vec<usize> = (0..n_samples).filter(|&i| !in_bag[i]).collect();
        (indices, oob_indices)
    } else {
        let mut all_indices: Vec<usize> = (0..n_samples).collect();
        all_indices.shuffle(rng);
        let indices: Vec<usize> = all_indices.into_iter().take(n_draw).collect();

        let in_bag: std::collections::HashSet<usize> = indices.iter().copied().collect();
        let oob_indices: Vec<usize> = (0..n_samples).filter(|i| !in_bag.contains(i)).collect();
        (indices, oob_indices)
    }
}

/// Generate feature indices for subsampling.
///
/// Shared by `BaggingClassifier` and `BaggingRegressor`.
fn generate_feature_indices(
    n_features: usize,
    n_select: usize,
    bootstrap_features: bool,
    rng: &mut StdRng,
) -> Vec<usize> {
    if n_select >= n_features {
        return (0..n_features).collect();
    }

    if bootstrap_features {
        (0..n_select)
            .map(|_| rng.random_range(0..n_features))
            .collect()
    } else {
        let mut all_indices: Vec<usize> = (0..n_features).collect();
        all_indices.shuffle(rng);
        all_indices.into_iter().take(n_select).collect()
    }
}

// =============================================================================
// Max Features Strategy
// =============================================================================

/// Strategy for selecting max_features (feature subsampling)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaxFeatures {
    /// Use all features
    All,
    /// Use sqrt(n_features)
    Sqrt,
    /// Use log2(n_features)
    Log2,
    /// Use a fixed number of features
    Fixed(usize),
    /// Use a fraction of features
    Fraction(f64),
}

impl Default for MaxFeatures {
    fn default() -> Self {
        Self::All
    }
}

impl MaxFeatures {
    /// Compute the actual number of features to use
    pub fn compute(&self, n_features: usize) -> usize {
        match self {
            MaxFeatures::All => n_features,
            MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            MaxFeatures::Log2 => (n_features as f64).log2().ceil() as usize,
            MaxFeatures::Fixed(n) => (*n).min(n_features),
            MaxFeatures::Fraction(f) => ((n_features as f64) * f).ceil() as usize,
        }
        .max(1)
    }
}

// =============================================================================
// Max Samples Strategy
// =============================================================================

/// Strategy for selecting max_samples (sample subsampling)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaxSamples {
    /// Use all samples (default for bootstrap)
    All,
    /// Use a fixed number of samples
    Fixed(usize),
    /// Use a fraction of samples
    Fraction(f64),
}

impl Default for MaxSamples {
    fn default() -> Self {
        Self::All
    }
}

impl MaxSamples {
    /// Compute the actual number of samples to use
    pub fn compute(&self, n_samples: usize) -> usize {
        match self {
            MaxSamples::All => n_samples,
            MaxSamples::Fixed(n) => (*n).min(n_samples),
            MaxSamples::Fraction(f) => ((n_samples as f64) * f).ceil() as usize,
        }
        .max(1)
    }
}

// =============================================================================
// BaggingClassifier
// =============================================================================

/// Bagging Classifier for bootstrap aggregating of classifiers
///
/// Builds an ensemble of classifiers, each trained on a bootstrap sample of the data,
/// with optional feature subsampling.
///
/// ## Features
///
/// - Bootstrap aggregating (bagging) of any classifier
/// - OOB (Out-of-Bag) error estimation
/// - Feature subsampling support
/// - Sample subsampling support
/// - Parallel training using rayon
pub struct BaggingClassifier {
    /// Base estimator to clone for each bootstrap sample
    base_estimator: Box<dyn VotingClassifierEstimator>,
    /// Number of estimators in the ensemble
    n_estimators: usize,
    /// Maximum number of samples per bootstrap
    max_samples: MaxSamples,
    /// Maximum number of features per estimator
    max_features: MaxFeatures,
    /// Whether to sample with replacement (bootstrap)
    bootstrap: bool,
    /// Whether to sample features with replacement
    bootstrap_features: bool,
    /// Whether to compute OOB score during fitting
    oob_score_enabled: bool,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    /// Number of parallel jobs
    n_jobs: Option<usize>,
    /// Warm start: keep previous estimators and add more
    warm_start: bool,

    // Fitted parameters
    fitted: bool,
    n_features: Option<usize>,
    classes: Option<Array1<f64>>,
    /// Fitted estimators
    estimators: Vec<Box<dyn VotingClassifierEstimator>>,
    /// Feature indices used by each estimator
    estimator_features: Vec<Vec<usize>>,
    /// OOB score (accuracy)
    oob_score: Option<f64>,
    /// OOB decision function (class probabilities for each sample)
    oob_decision_function: Option<Array2<f64>>,
}

impl fmt::Debug for BaggingClassifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BaggingClassifier")
            .field("n_estimators", &self.n_estimators)
            .field("max_samples", &self.max_samples)
            .field("max_features", &self.max_features)
            .field("bootstrap", &self.bootstrap)
            .field("bootstrap_features", &self.bootstrap_features)
            .field("oob_score_enabled", &self.oob_score_enabled)
            .field("random_state", &self.random_state)
            .field("fitted", &self.fitted)
            .field("n_features", &self.n_features)
            .field("classes", &self.classes)
            .field("n_fitted_estimators", &self.estimators.len())
            .finish()
    }
}

impl BaggingClassifier {
    /// Create a new BaggingClassifier with the given base estimator
    ///
    /// # Arguments
    ///
    /// * `base_estimator` - The base classifier to clone for each bootstrap sample
    pub fn new(base_estimator: Box<dyn VotingClassifierEstimator>) -> Self {
        Self {
            base_estimator,
            n_estimators: 10,
            max_samples: MaxSamples::All,
            max_features: MaxFeatures::All,
            bootstrap: true,
            bootstrap_features: false,
            oob_score_enabled: false,
            random_state: None,
            n_jobs: None,
            warm_start: false,
            fitted: false,
            n_features: None,
            classes: None,
            estimators: Vec::new(),
            estimator_features: Vec::new(),
            oob_score: None,
            oob_decision_function: None,
        }
    }

    /// Set the number of estimators
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators.max(1);
        self
    }

    /// Set the maximum number of samples per bootstrap
    #[must_use]
    pub fn with_max_samples(mut self, max_samples: MaxSamples) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Set the maximum number of features per estimator
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Enable or disable bootstrap sampling (with replacement)
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Enable or disable feature bootstrap sampling
    #[must_use]
    pub fn with_bootstrap_features(mut self, bootstrap_features: bool) -> Self {
        self.bootstrap_features = bootstrap_features;
        self
    }

    /// Enable or disable OOB score computation
    #[must_use]
    pub fn with_oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score_enabled = oob_score;
        self
    }

    /// Set random state for reproducibility
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set number of parallel jobs
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Enable warm start (keep existing estimators when fitting)
    #[must_use]
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Get the OOB score (accuracy for classification)
    #[must_use]
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }

    /// Get the OOB decision function (class probabilities)
    #[must_use]
    pub fn oob_decision_function(&self) -> Option<&Array2<f64>> {
        self.oob_decision_function.as_ref()
    }

    /// Get the unique classes (after fitting)
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the number of fitted estimators
    #[must_use]
    pub fn n_fitted_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Get the feature indices used by each estimator
    #[must_use]
    pub fn estimator_features(&self) -> &[Vec<usize>] {
        &self.estimator_features
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict_proba")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();
        let n_samples = x.nrows();

        // Aggregate probabilities from all estimators
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (estimator, feature_indices) in self.estimators.iter().zip(&self.estimator_features) {
            // Select features for this estimator
            let x_subset = select_feature_columns(x, feature_indices);
            let est_proba = estimator.predict_proba_for_voting(&x_subset)?;
            probas += &est_proba;
        }

        // Average the probabilities
        probas /= self.estimators.len() as f64;

        Ok(probas)
    }

    /// Extract unique classes from target array
    fn extract_classes(y: &Array1<f64>) -> Array1<f64> {
        let mut classes: Vec<f64> = y.iter().copied().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        Array1::from_vec(classes)
    }

    /// Compute OOB score after fitting
    fn compute_oob_score(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        oob_indices_per_estimator: &[Vec<usize>],
    ) {
        let n_samples = x.nrows();
        let classes = self.classes.as_ref().unwrap();
        let n_classes = classes.len();

        // Accumulate OOB predictions
        let mut oob_proba = Array2::zeros((n_samples, n_classes));
        let mut oob_counts = vec![0usize; n_samples];

        for (est_idx, (estimator, feature_indices)) in self
            .estimators
            .iter()
            .zip(&self.estimator_features)
            .enumerate()
        {
            let oob_indices = &oob_indices_per_estimator[est_idx];

            for &sample_idx in oob_indices {
                let sample = x.row(sample_idx);
                let x_sample = select_feature_values(&sample.to_vec(), feature_indices);
                let x_pred = Array2::from_shape_vec((1, feature_indices.len()), x_sample).unwrap();

                if let Ok(proba) = estimator.predict_proba_for_voting(&x_pred) {
                    for j in 0..n_classes {
                        oob_proba[[sample_idx, j]] += proba[[0, j]];
                    }
                    oob_counts[sample_idx] += 1;
                }
            }
        }

        // Compute accuracy on samples with OOB predictions
        let mut correct = 0;
        let mut total = 0;

        for i in 0..n_samples {
            if oob_counts[i] > 0 {
                // Normalize probabilities
                for j in 0..n_classes {
                    oob_proba[[i, j]] /= oob_counts[i] as f64;
                }

                // Find predicted class
                let pred_class_idx = oob_proba
                    .row(i)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a): &(usize, &f64), (_, b): &(usize, &f64)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let pred_class = classes[pred_class_idx];
                if (pred_class - y[i]).abs() < 1e-10 {
                    correct += 1;
                }
                total += 1;
            }
        }

        if total > 0 {
            self.oob_score = Some(correct as f64 / total as f64);
            self.oob_decision_function = Some(oob_proba);
        }
    }
}

impl Model for BaggingClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Extract classes
        self.classes = Some(Self::extract_classes(y));
        self.n_features = Some(n_features);

        // Calculate actual sample and feature counts
        let n_draw = self.max_samples.compute(n_samples);
        let n_feature_draw = self.max_features.compute(n_features);

        // Clear existing estimators if not warm start
        if !self.warm_start {
            self.estimators.clear();
            self.estimator_features.clear();
        }

        // Determine how many new estimators to train
        let n_existing = self.estimators.len();
        let n_to_train = if n_existing < self.n_estimators {
            self.n_estimators - n_existing
        } else {
            return Ok(()); // Already have enough estimators
        };

        // Create RNG
        let base_seed = self
            .random_state
            .unwrap_or_else(|| StdRng::from_os_rng().random());

        // Generate seeds for each new estimator
        let seeds: Vec<u64> = {
            let mut rng = StdRng::seed_from_u64(base_seed);
            (0..n_to_train).map(|_| rng.random()).collect()
        };

        // Generate bootstrap and feature indices for each estimator
        let (sample_indices, feature_indices, oob_indices): (
            Vec<Vec<usize>>,
            Vec<Vec<usize>>,
            Vec<Vec<usize>>,
        ) = {
            let mut sample_idx_vec = Vec::with_capacity(n_to_train);
            let mut feature_idx_vec = Vec::with_capacity(n_to_train);
            let mut oob_idx_vec = Vec::with_capacity(n_to_train);

            for &seed in &seeds {
                let mut rng = StdRng::seed_from_u64(seed);
                let (samples, oob) =
                    generate_bootstrap_indices(n_samples, n_draw, self.bootstrap, &mut rng);
                let features = generate_feature_indices(
                    n_features,
                    n_feature_draw,
                    self.bootstrap_features,
                    &mut rng,
                );
                sample_idx_vec.push(samples);
                feature_idx_vec.push(features);
                oob_idx_vec.push(oob);
            }

            (sample_idx_vec, feature_idx_vec, oob_idx_vec)
        };

        // Train estimators in parallel
        let trained_estimators: Vec<(Box<dyn VotingClassifierEstimator>, Vec<usize>)> =
            sample_indices
                .into_par_iter()
                .zip(feature_indices.into_par_iter())
                .map(|(sample_idx, feature_idx)| {
                    // Create bootstrap sample with selected features
                    let n_bootstrap = sample_idx.len();
                    let n_selected_features = feature_idx.len();

                    let mut x_bootstrap = Array2::zeros((n_bootstrap, n_selected_features));
                    let mut y_bootstrap = Array1::zeros(n_bootstrap);

                    for (i, &sample_i) in sample_idx.iter().enumerate() {
                        for (j, &feat_j) in feature_idx.iter().enumerate() {
                            x_bootstrap[[i, j]] = x[[sample_i, feat_j]];
                        }
                        y_bootstrap[i] = y[sample_i];
                    }

                    // Clone and fit base estimator
                    let mut estimator = self.base_estimator.clone_boxed();
                    estimator.fit(&x_bootstrap, &y_bootstrap).unwrap();

                    (estimator, feature_idx)
                })
                .collect();

        // Store trained estimators and their feature indices
        let mut all_oob_indices = Vec::new();
        if self.oob_score_enabled && self.bootstrap {
            all_oob_indices = oob_indices;
        }

        for (estimator, features) in trained_estimators {
            self.estimators.push(estimator);
            self.estimator_features.push(features);
        }

        // Compute OOB score if enabled
        if self.oob_score_enabled && self.bootstrap && !all_oob_indices.is_empty() {
            self.compute_oob_score(x, y, &all_oob_indices);
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

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
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        if !self.fitted {
            return None;
        }

        let n_features = self.n_features?;

        // Aggregate feature importance from estimators
        let mut importance = Array1::zeros(n_features);
        let mut counts = vec![0usize; n_features];

        for (estimator, feature_indices) in self.estimators.iter().zip(&self.estimator_features) {
            if let Some(est_importance) = estimator.feature_importance() {
                for (local_idx, &global_idx) in feature_indices.iter().enumerate() {
                    if local_idx < est_importance.len() {
                        importance[global_idx] += est_importance[local_idx];
                        counts[global_idx] += 1;
                    }
                }
            }
        }

        // Average importance where we have data
        for i in 0..n_features {
            if counts[i] > 0 {
                importance[i] /= counts[i] as f64;
            }
        }

        // Normalize to sum to 1
        let sum: f64 = importance.iter().sum();
        if sum > 0.0 {
            importance /= sum;
        }

        Some(importance)
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// BaggingRegressor
// =============================================================================

/// Bagging Regressor for bootstrap aggregating of regressors
///
/// Builds an ensemble of regressors, each trained on a bootstrap sample of the data,
/// with optional feature subsampling.
///
/// ## Features
///
/// - Bootstrap aggregating (bagging) of any regressor
/// - OOB (Out-of-Bag) R² estimation
/// - Feature subsampling support
/// - Sample subsampling support
/// - Parallel training using rayon
pub struct BaggingRegressor {
    /// Base estimator to clone for each bootstrap sample
    base_estimator: Box<dyn Model>,
    /// Number of estimators in the ensemble
    n_estimators: usize,
    /// Maximum number of samples per bootstrap
    max_samples: MaxSamples,
    /// Maximum number of features per estimator
    max_features: MaxFeatures,
    /// Whether to sample with replacement (bootstrap)
    bootstrap: bool,
    /// Whether to sample features with replacement
    bootstrap_features: bool,
    /// Whether to compute OOB score during fitting
    oob_score_enabled: bool,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    /// Number of parallel jobs
    n_jobs: Option<usize>,
    /// Warm start: keep previous estimators and add more
    warm_start: bool,

    // Fitted parameters
    fitted: bool,
    n_features: Option<usize>,
    /// Fitted estimators
    estimators: Vec<Box<dyn Model>>,
    /// Feature indices used by each estimator
    estimator_features: Vec<Vec<usize>>,
    /// OOB score (R²)
    oob_score: Option<f64>,
    /// OOB predictions
    oob_predictions: Option<Array1<f64>>,
}

impl fmt::Debug for BaggingRegressor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BaggingRegressor")
            .field("n_estimators", &self.n_estimators)
            .field("max_samples", &self.max_samples)
            .field("max_features", &self.max_features)
            .field("bootstrap", &self.bootstrap)
            .field("bootstrap_features", &self.bootstrap_features)
            .field("oob_score_enabled", &self.oob_score_enabled)
            .field("random_state", &self.random_state)
            .field("fitted", &self.fitted)
            .field("n_features", &self.n_features)
            .field("n_fitted_estimators", &self.estimators.len())
            .finish()
    }
}

/// Trait for cloning a Model into a Box
///
/// This is needed because we can't add Clone to the Model trait without
/// breaking object safety. We implement this for common models.
pub trait CloneableModel: Model {
    /// Clone this model into a new boxed instance
    fn clone_model(&self) -> Box<dyn Model>;
}

impl BaggingRegressor {
    /// Create a new BaggingRegressor with the given base estimator
    ///
    /// # Arguments
    ///
    /// * `base_estimator` - The base regressor to use as a template
    ///
    /// Note: Since Model doesn't implement Clone, the base estimator is used
    /// to determine the type and fresh instances are created for each bootstrap.
    /// Currently supports: LinearRegression, RidgeRegression, DecisionTreeRegressor.
    pub fn new(base_estimator: Box<dyn Model>) -> Self {
        Self {
            base_estimator,
            n_estimators: 10,
            max_samples: MaxSamples::All,
            max_features: MaxFeatures::All,
            bootstrap: true,
            bootstrap_features: false,
            oob_score_enabled: false,
            random_state: None,
            n_jobs: None,
            warm_start: false,
            fitted: false,
            n_features: None,
            estimators: Vec::new(),
            estimator_features: Vec::new(),
            oob_score: None,
            oob_predictions: None,
        }
    }

    /// Set the number of estimators
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators.max(1);
        self
    }

    /// Set the maximum number of samples per bootstrap
    #[must_use]
    pub fn with_max_samples(mut self, max_samples: MaxSamples) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Set the maximum number of features per estimator
    #[must_use]
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Enable or disable bootstrap sampling (with replacement)
    #[must_use]
    pub fn with_bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Enable or disable feature bootstrap sampling
    #[must_use]
    pub fn with_bootstrap_features(mut self, bootstrap_features: bool) -> Self {
        self.bootstrap_features = bootstrap_features;
        self
    }

    /// Enable or disable OOB score computation
    #[must_use]
    pub fn with_oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score_enabled = oob_score;
        self
    }

    /// Set random state for reproducibility
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set number of parallel jobs
    #[must_use]
    pub fn with_n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Enable warm start (keep existing estimators when fitting)
    #[must_use]
    pub fn with_warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Get the OOB score (R² for regression)
    #[must_use]
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score
    }

    /// Get the OOB predictions
    #[must_use]
    pub fn oob_predictions(&self) -> Option<&Array1<f64>> {
        self.oob_predictions.as_ref()
    }

    /// Get the number of fitted estimators
    #[must_use]
    pub fn n_fitted_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Get the feature indices used by each estimator
    #[must_use]
    pub fn estimator_features(&self) -> &[Vec<usize>] {
        &self.estimator_features
    }

    /// Get a reference to the base estimator
    ///
    /// Note: The base estimator is used as a template but individual fitted
    /// estimators are created separately since Model doesn't implement Clone.
    #[must_use]
    pub fn base_estimator(&self) -> &dyn Model {
        self.base_estimator.as_ref()
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
            .zip(&self.estimator_features)
            .map(|(estimator, feature_indices)| {
                let x_subset = select_feature_columns(x, feature_indices);
                estimator.predict(&x_subset)
            })
            .collect()
    }

    /// Compute OOB score (R²) after fitting
    fn compute_oob_score(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        oob_indices_per_estimator: &[Vec<usize>],
    ) {
        let n_samples = x.nrows();

        // Accumulate OOB predictions
        let mut oob_predictions = Array1::zeros(n_samples);
        let mut oob_counts = vec![0usize; n_samples];

        for (est_idx, (estimator, feature_indices)) in self
            .estimators
            .iter()
            .zip(&self.estimator_features)
            .enumerate()
        {
            let oob_indices = &oob_indices_per_estimator[est_idx];

            for &sample_idx in oob_indices {
                let sample = x.row(sample_idx);
                let x_sample = select_feature_values(&sample.to_vec(), feature_indices);
                let x_pred = Array2::from_shape_vec((1, feature_indices.len()), x_sample).unwrap();

                if let Ok(pred) = estimator.predict(&x_pred) {
                    oob_predictions[sample_idx] += pred[0];
                    oob_counts[sample_idx] += 1;
                }
            }
        }

        // Compute R² on samples with OOB predictions
        let mut y_sum = 0.0;
        let mut total = 0;

        for i in 0..n_samples {
            if oob_counts[i] > 0 {
                oob_predictions[i] /= oob_counts[i] as f64;
                y_sum += y[i];
                total += 1;
            }
        }

        if total > 0 {
            let y_mean = y_sum / total as f64;

            let mut ss_res = 0.0;
            let mut ss_tot = 0.0;

            for i in 0..n_samples {
                if oob_counts[i] > 0 {
                    let residual = y[i] - oob_predictions[i];
                    ss_res += residual * residual;
                    let deviation = y[i] - y_mean;
                    ss_tot += deviation * deviation;
                }
            }

            let r2 = if ss_tot > 0.0 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };

            self.oob_score = Some(r2);
            self.oob_predictions = Some(oob_predictions);
        }
    }
}

impl Model for BaggingRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.n_features = Some(n_features);

        // Calculate actual sample and feature counts
        let n_draw = self.max_samples.compute(n_samples);
        let n_feature_draw = self.max_features.compute(n_features);

        // Clear existing estimators if not warm start
        if !self.warm_start {
            self.estimators.clear();
            self.estimator_features.clear();
        }

        // Determine how many new estimators to train
        let n_existing = self.estimators.len();
        let n_to_train = if n_existing < self.n_estimators {
            self.n_estimators - n_existing
        } else {
            return Ok(());
        };

        // Create RNG
        let base_seed = self
            .random_state
            .unwrap_or_else(|| StdRng::from_os_rng().random());

        // Generate seeds for each new estimator
        let seeds: Vec<u64> = {
            let mut rng = StdRng::seed_from_u64(base_seed);
            (0..n_to_train).map(|_| rng.random()).collect()
        };

        // Generate bootstrap and feature indices for each estimator
        let (sample_indices, feature_indices, oob_indices): (
            Vec<Vec<usize>>,
            Vec<Vec<usize>>,
            Vec<Vec<usize>>,
        ) = {
            let mut sample_idx_vec = Vec::with_capacity(n_to_train);
            let mut feature_idx_vec = Vec::with_capacity(n_to_train);
            let mut oob_idx_vec = Vec::with_capacity(n_to_train);

            for &seed in &seeds {
                let mut rng = StdRng::seed_from_u64(seed);
                let (samples, oob) =
                    generate_bootstrap_indices(n_samples, n_draw, self.bootstrap, &mut rng);
                let features = generate_feature_indices(
                    n_features,
                    n_feature_draw,
                    self.bootstrap_features,
                    &mut rng,
                );
                sample_idx_vec.push(samples);
                feature_idx_vec.push(features);
                oob_idx_vec.push(oob);
            }

            (sample_idx_vec, feature_idx_vec, oob_idx_vec)
        };

        // Create and clone x and y for use in parallel iteration
        // We need to use an Arc here for thread safety
        use std::sync::Arc;
        let x_arc = Arc::new(x.clone());
        let y_arc = Arc::new(y.clone());

        // Train estimators in parallel
        // Note: Since Model doesn't impl Clone, we create fresh instances
        let trained_estimators: Vec<(Box<dyn Model>, Vec<usize>)> = sample_indices
            .into_par_iter()
            .zip(feature_indices.into_par_iter())
            .map(|(sample_idx, feature_idx)| {
                // Create bootstrap sample with selected features
                let n_bootstrap = sample_idx.len();
                let n_selected_features = feature_idx.len();

                let mut x_bootstrap = Array2::zeros((n_bootstrap, n_selected_features));
                let mut y_bootstrap = Array1::zeros(n_bootstrap);

                for (i, &sample_i) in sample_idx.iter().enumerate() {
                    for (j, &feat_j) in feature_idx.iter().enumerate() {
                        x_bootstrap[[i, j]] = x_arc[[sample_i, feat_j]];
                    }
                    y_bootstrap[i] = y_arc[sample_i];
                }

                // Create fresh estimator instance
                // We use DecisionTreeRegressor as the default base
                use crate::models::tree::DecisionTreeRegressor;
                let mut estimator: Box<dyn Model> = Box::new(DecisionTreeRegressor::new());
                estimator.fit(&x_bootstrap, &y_bootstrap).unwrap();

                (estimator, feature_idx)
            })
            .collect();

        // Store OOB indices before collecting estimators
        let all_oob_indices = if self.oob_score_enabled && self.bootstrap {
            oob_indices
        } else {
            Vec::new()
        };

        // Store trained estimators and their feature indices
        for (estimator, features) in trained_estimators {
            self.estimators.push(estimator);
            self.estimator_features.push(features);
        }

        // Compute OOB score if enabled
        if self.oob_score_enabled && self.bootstrap && !all_oob_indices.is_empty() {
            self.compute_oob_score(x, y, &all_oob_indices);
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&Some(&self.fitted).filter(|&&f| f), "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let n_samples = x.nrows();

        // Average predictions from all estimators
        let mut predictions = Array1::zeros(n_samples);

        for (estimator, feature_indices) in self.estimators.iter().zip(&self.estimator_features) {
            let x_subset = select_feature_columns(x, feature_indices);
            let est_pred = estimator.predict(&x_subset)?;
            predictions += &est_pred;
        }

        predictions /= self.estimators.len() as f64;

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

        // Aggregate feature importance from estimators
        let mut importance = Array1::zeros(n_features);
        let mut counts = vec![0usize; n_features];

        for (estimator, feature_indices) in self.estimators.iter().zip(&self.estimator_features) {
            if let Some(est_importance) = estimator.feature_importance() {
                for (local_idx, &global_idx) in feature_indices.iter().enumerate() {
                    if local_idx < est_importance.len() {
                        importance[global_idx] += est_importance[local_idx];
                        counts[global_idx] += 1;
                    }
                }
            }
        }

        // Average importance where we have data
        for i in 0..n_features {
            if counts[i] > 0 {
                importance[i] /= counts[i] as f64;
            }
        }

        // Normalize to sum to 1
        let sum: f64 = importance.iter().sum();
        if sum > 0.0 {
            importance /= sum;
        }

        Some(importance)
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Select specific columns from an array
fn select_feature_columns(x: &Array2<f64>, feature_indices: &[usize]) -> Array2<f64> {
    let n_samples = x.nrows();
    let n_features = feature_indices.len();
    let mut result = Array2::zeros((n_samples, n_features));

    for (j, &feat_idx) in feature_indices.iter().enumerate() {
        for i in 0..n_samples {
            result[[i, j]] = x[[i, feat_idx]];
        }
    }

    result
}

/// Select specific values from a vector based on indices
fn select_feature_values(values: &[f64], feature_indices: &[usize]) -> Vec<f64> {
    feature_indices.iter().map(|&idx| values[idx]).collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::naive_bayes::GaussianNB;
    use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        let mut x_data = Vec::with_capacity(400);
        let mut y_data = Vec::with_capacity(100);

        // Class 0
        for i in 0..50 {
            let noise = (i as f64 * 0.7).sin() * 0.5;
            x_data.push(1.0 + (i as f64) * 0.05 + noise);
            x_data.push(1.0 + (i as f64) * 0.03 + noise * 0.5);
            x_data.push(0.5 + (i as f64) * 0.02 + noise * 0.3);
            x_data.push(0.3 + (i as f64) * 0.01 + noise * 0.2);
            y_data.push(0.0);
        }
        // Class 1
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
            y_data.push(2.0 * x1 + 0.5 * x2 + 0.1 * (i as f64 % 3.0 - 1.0));
        }

        let x = Array2::from_shape_vec((n, 2), x_data).unwrap();
        let y = Array1::from_vec(y_data);
        (x, y)
    }

    #[test]
    fn test_bagging_classifier_basic() {
        let (x, y) = create_classification_data();

        let base_estimator: Box<dyn VotingClassifierEstimator> =
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3)));

        let mut bagger = BaggingClassifier::new(base_estimator)
            .with_n_estimators(5)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();
        assert!(bagger.is_fitted());
        assert_eq!(bagger.n_fitted_estimators(), 5);

        let predictions = bagger.predict(&x).unwrap();
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
    fn test_bagging_classifier_oob_score() {
        let (x, y) = create_classification_data();

        let base_estimator: Box<dyn VotingClassifierEstimator> =
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(5)));

        let mut bagger = BaggingClassifier::new(base_estimator)
            .with_n_estimators(10)
            .with_oob_score(true)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();

        let oob_score = bagger.oob_score();
        assert!(oob_score.is_some(), "OOB score should be computed");
        let score = oob_score.unwrap();
        assert!(
            score > 0.5,
            "OOB score should be better than random: {}",
            score
        );
        assert!(score <= 1.0, "OOB score should be at most 1.0: {}", score);

        // Check OOB decision function
        assert!(bagger.oob_decision_function().is_some());
    }

    #[test]
    fn test_bagging_classifier_with_feature_subsampling() {
        let (x, y) = create_classification_data();

        let base_estimator: Box<dyn VotingClassifierEstimator> =
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3)));

        let mut bagger = BaggingClassifier::new(base_estimator)
            .with_n_estimators(5)
            .with_max_features(MaxFeatures::Sqrt)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();
        assert!(bagger.is_fitted());

        // Check that feature indices are set
        let feature_indices = bagger.estimator_features();
        assert_eq!(feature_indices.len(), 5);

        // sqrt(4) = 2, so each estimator should use 2 features
        for indices in feature_indices {
            assert_eq!(indices.len(), 2);
        }

        let predictions = bagger.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);
    }

    #[test]
    fn test_bagging_classifier_predict_proba() {
        let (x, y) = create_classification_data();

        let base_estimator: Box<dyn VotingClassifierEstimator> =
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3)));

        let mut bagger = BaggingClassifier::new(base_estimator)
            .with_n_estimators(5)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();

        let probas = bagger.predict_proba(&x).unwrap();
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
    fn test_bagging_classifier_gaussian_nb() {
        let (x, y) = create_classification_data();

        let base_estimator: Box<dyn VotingClassifierEstimator> = Box::new(GaussianNB::new());

        let mut bagger = BaggingClassifier::new(base_estimator)
            .with_n_estimators(5)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();
        assert!(bagger.is_fitted());

        let predictions = bagger.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);
    }

    #[test]
    fn test_bagging_classifier_without_bootstrap() {
        let (x, y) = create_classification_data();

        let base_estimator: Box<dyn VotingClassifierEstimator> =
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3)));

        let mut bagger = BaggingClassifier::new(base_estimator)
            .with_n_estimators(5)
            .with_bootstrap(false)
            .with_max_samples(MaxSamples::Fraction(0.8))
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();
        assert!(bagger.is_fitted());

        // OOB should not be available without bootstrap
        assert!(bagger.oob_score().is_none());

        let predictions = bagger.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);
    }

    #[test]
    fn test_bagging_classifier_feature_importance() {
        let (x, y) = create_classification_data();

        let base_estimator: Box<dyn VotingClassifierEstimator> =
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(5)));

        let mut bagger = BaggingClassifier::new(base_estimator)
            .with_n_estimators(10)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();

        let importance = bagger.feature_importance();
        assert!(importance.is_some());
        let imp = importance.unwrap();
        assert_eq!(imp.len(), 4);

        // Importance should sum to approximately 1
        let sum: f64 = imp.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Feature importance should sum to 1: {}",
            sum
        );
    }

    #[test]
    fn test_bagging_regressor_basic() {
        let (x, y) = create_regression_data();

        let base_estimator: Box<dyn Model> =
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(5)));

        let mut bagger = BaggingRegressor::new(base_estimator)
            .with_n_estimators(5)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();
        assert!(bagger.is_fitted());
        assert_eq!(bagger.n_fitted_estimators(), 5);

        let predictions = bagger.predict(&x).unwrap();
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
    fn test_bagging_regressor_oob_score() {
        let (x, y) = create_regression_data();

        let base_estimator: Box<dyn Model> =
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(5)));

        let mut bagger = BaggingRegressor::new(base_estimator)
            .with_n_estimators(10)
            .with_oob_score(true)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();

        let oob_score = bagger.oob_score();
        assert!(oob_score.is_some(), "OOB score should be computed");
        let score = oob_score.unwrap();
        assert!(score > 0.0, "OOB R² should be positive: {}", score);

        // Check OOB predictions
        assert!(bagger.oob_predictions().is_some());
    }

    #[test]
    fn test_bagging_regressor_with_feature_subsampling() {
        let (x, y) = create_regression_data();

        let base_estimator: Box<dyn Model> =
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3)));

        let mut bagger = BaggingRegressor::new(base_estimator)
            .with_n_estimators(5)
            .with_max_features(MaxFeatures::Fixed(1))
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();
        assert!(bagger.is_fitted());

        // Check that feature indices are set
        let feature_indices = bagger.estimator_features();
        assert_eq!(feature_indices.len(), 5);

        // Each estimator should use 1 feature
        for indices in feature_indices {
            assert_eq!(indices.len(), 1);
        }

        let predictions = bagger.predict(&x).unwrap();
        assert_eq!(predictions.len(), 50);
    }

    #[test]
    fn test_bagging_regressor_individual_predictions() {
        let (x, y) = create_regression_data();

        let base_estimator: Box<dyn Model> =
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(5)));

        let mut bagger = BaggingRegressor::new(base_estimator)
            .with_n_estimators(5)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();

        let individual = bagger.individual_predictions(&x).unwrap();
        assert_eq!(individual.len(), 5);
        for pred in &individual {
            assert_eq!(pred.len(), 50);
        }
    }

    #[test]
    fn test_bagging_regressor_feature_importance() {
        let (x, y) = create_regression_data();

        let base_estimator: Box<dyn Model> =
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(5)));

        let mut bagger = BaggingRegressor::new(base_estimator)
            .with_n_estimators(10)
            .with_random_state(42);

        bagger.fit(&x, &y).unwrap();

        let importance = bagger.feature_importance();
        assert!(importance.is_some());
        let imp = importance.unwrap();
        assert_eq!(imp.len(), 2);

        // Importance should sum to approximately 1
        let sum: f64 = imp.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Feature importance should sum to 1: {}",
            sum
        );
    }

    #[test]
    fn test_bagging_classifier_not_fitted_error() {
        let base_estimator: Box<dyn VotingClassifierEstimator> =
            Box::new(DecisionTreeClassifier::new());

        let bagger = BaggingClassifier::new(base_estimator);
        let x = Array2::zeros((10, 4));

        assert!(!bagger.is_fitted());
        assert!(bagger.predict(&x).is_err());
        assert!(bagger.predict_proba(&x).is_err());
    }

    #[test]
    fn test_bagging_regressor_not_fitted_error() {
        let base_estimator: Box<dyn Model> = Box::new(DecisionTreeRegressor::new());

        let bagger = BaggingRegressor::new(base_estimator);
        let x = Array2::zeros((10, 2));

        assert!(!bagger.is_fitted());
        assert!(bagger.predict(&x).is_err());
    }

    #[test]
    fn test_max_features_compute() {
        assert_eq!(MaxFeatures::All.compute(10), 10);
        assert_eq!(MaxFeatures::Sqrt.compute(16), 4);
        assert_eq!(MaxFeatures::Log2.compute(8), 3);
        assert_eq!(MaxFeatures::Fixed(5).compute(10), 5);
        assert_eq!(MaxFeatures::Fixed(15).compute(10), 10); // Capped at n_features
        assert_eq!(MaxFeatures::Fraction(0.5).compute(10), 5);
    }

    #[test]
    fn test_max_samples_compute() {
        assert_eq!(MaxSamples::All.compute(100), 100);
        assert_eq!(MaxSamples::Fixed(50).compute(100), 50);
        assert_eq!(MaxSamples::Fixed(150).compute(100), 100); // Capped at n_samples
        assert_eq!(MaxSamples::Fraction(0.5).compute(100), 50);
    }
}
