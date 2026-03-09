//! Gradient Boosting Models
//!
//! This module provides Gradient Boosting ensemble implementations that build
//! additive models by sequentially fitting trees to the negative gradient.
//!
//! ## Models
//!
//! - **GradientBoostingClassifier**: For classification tasks
//! - **GradientBoostingRegressor**: For regression tasks
//!
//! ## Features
//!
//! - **Learning rate scheduling**: Fixed or decay-based learning rates
//! - **Early stopping**: Stop training when validation score stops improving
//! - **Feature importance**: Computed from impurity decrease across all trees
//! - **Multiple loss functions**: MSE/LAD for regression, deviance/exponential for classification
//! - **Subsample support**: Stochastic gradient boosting via row subsampling
//!
//! ## Example - Regression
//!
//! ```
//! use ferroml_core::models::boosting::GradientBoostingRegressor;
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64 / 10.0).collect()).unwrap();
//! let y = Array1::from_iter((0..20).map(|i| i as f64 * 0.5 + 1.0));
//!
//! let mut model = GradientBoostingRegressor::new()
//!     .with_n_estimators(10)
//!     .with_learning_rate(0.1)
//!     .with_max_depth(Some(3));
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 20);
//! ```
//!
//! ## Example - Classification
//!
//! ```
//! use ferroml_core::models::boosting::GradientBoostingClassifier;
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64 / 10.0).collect()).unwrap();
//! let y = Array1::from_iter((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }));
//!
//! let mut model = GradientBoostingClassifier::new()
//!     .with_n_estimators(10)
//!     .with_learning_rate(0.1);
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 20);
//! ```

use crate::hpo::SearchSpace;
use crate::models::tree::{DecisionTreeRegressor, SplitCriterion};
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, ClassWeight, Model,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// =============================================================================
// Loss Functions
// =============================================================================

/// Loss function for gradient boosting regression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionLoss {
    /// Squared error loss (L2) - default
    SquaredError,
    /// Absolute error loss (L1) - more robust to outliers
    AbsoluteError,
    /// Huber loss - combines L2 for small errors, L1 for large errors
    Huber,
}

impl Default for RegressionLoss {
    fn default() -> Self {
        Self::SquaredError
    }
}

impl RegressionLoss {
    /// Compute the initial prediction (constant that minimizes loss)
    fn init_prediction(&self, y: &Array1<f64>) -> f64 {
        match self {
            RegressionLoss::SquaredError => y.mean().unwrap_or(0.0),
            RegressionLoss::AbsoluteError | RegressionLoss::Huber => {
                // Median minimizes absolute error
                let mut sorted: Vec<f64> = y.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n == 0 {
                    0.0
                } else if n % 2 == 0 {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                } else {
                    sorted[n / 2]
                }
            }
        }
    }

    /// Compute negative gradient (pseudo-residuals)
    fn negative_gradient(
        &self,
        y: &Array1<f64>,
        predictions: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        match self {
            RegressionLoss::SquaredError => y - predictions,
            RegressionLoss::AbsoluteError => y
                .iter()
                .zip(predictions.iter())
                .map(|(&yi, &pi)| {
                    let diff = yi - pi;
                    if diff > 0.0 {
                        1.0
                    } else if diff < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            RegressionLoss::Huber => {
                // alpha is the quantile for Huber loss threshold
                let residuals: Vec<f64> = y
                    .iter()
                    .zip(predictions.iter())
                    .map(|(&yi, &pi)| yi - pi)
                    .collect();
                let abs_residuals: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();

                // Compute delta as the alpha-quantile of absolute residuals
                let mut sorted_abs = abs_residuals;
                sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let delta_idx =
                    ((alpha * sorted_abs.len() as f64) as usize).min(sorted_abs.len() - 1);
                let delta = sorted_abs[delta_idx];

                residuals
                    .iter()
                    .map(|&r| {
                        if r.abs() <= delta {
                            r
                        } else {
                            delta * r.signum()
                        }
                    })
                    .collect()
            }
        }
    }

    /// Compute the loss value
    fn loss(&self, y: &Array1<f64>, predictions: &Array1<f64>, alpha: f64) -> f64 {
        let n = y.len() as f64;
        match self {
            RegressionLoss::SquaredError => {
                y.iter()
                    .zip(predictions.iter())
                    .map(|(&yi, &pi)| (yi - pi).powi(2))
                    .sum::<f64>()
                    / (2.0 * n)
            }
            RegressionLoss::AbsoluteError => {
                y.iter()
                    .zip(predictions.iter())
                    .map(|(&yi, &pi)| (yi - pi).abs())
                    .sum::<f64>()
                    / n
            }
            RegressionLoss::Huber => {
                let residuals: Vec<f64> = y
                    .iter()
                    .zip(predictions.iter())
                    .map(|(&yi, &pi)| yi - pi)
                    .collect();
                let abs_residuals: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();

                let mut sorted_abs = abs_residuals;
                sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let delta_idx =
                    ((alpha * sorted_abs.len() as f64) as usize).min(sorted_abs.len() - 1);
                let delta = sorted_abs[delta_idx];

                residuals
                    .iter()
                    .map(|&r| {
                        if r.abs() <= delta {
                            0.5 * r.powi(2)
                        } else {
                            delta * 0.5f64.mul_add(-delta, r.abs())
                        }
                    })
                    .sum::<f64>()
                    / n
            }
        }
    }
}

/// Loss function for gradient boosting classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassificationLoss {
    /// Deviance (log-loss) - default
    Deviance,
    /// Exponential loss (AdaBoost-style)
    Exponential,
}

impl Default for ClassificationLoss {
    fn default() -> Self {
        Self::Deviance
    }
}

// =============================================================================
// Learning Rate Schedulers
// =============================================================================

/// Learning rate scheduling strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant(f64),
    /// Linear decay: lr = initial * (1 - iteration / n_estimators)
    LinearDecay {
        /// Initial learning rate
        initial: f64,
        /// Minimum learning rate floor
        min_lr: f64,
    },
    /// Exponential decay: lr = initial * decay^iteration
    ExponentialDecay {
        /// Initial learning rate
        initial: f64,
        /// Decay factor (0 < decay < 1)
        decay: f64,
        /// Minimum learning rate floor
        min_lr: f64,
    },
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self::Constant(0.1)
    }
}

impl LearningRateSchedule {
    /// Get the learning rate at a given iteration
    pub fn get_lr(&self, iteration: usize, n_estimators: usize) -> f64 {
        match self {
            LearningRateSchedule::Constant(lr) => *lr,
            LearningRateSchedule::LinearDecay { initial, min_lr } => {
                let progress = iteration as f64 / n_estimators as f64;
                (*initial * (1.0 - progress)).max(*min_lr)
            }
            LearningRateSchedule::ExponentialDecay {
                initial,
                decay,
                min_lr,
            } => (*initial * decay.powi(iteration as i32)).max(*min_lr),
        }
    }
}

// =============================================================================
// Early Stopping
// =============================================================================

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStopping {
    /// Number of iterations with no improvement to wait before stopping
    pub patience: usize,
    /// Minimum improvement to qualify as an improvement
    pub min_delta: f64,
    /// Fraction of data to use for validation (if validation set not provided)
    pub validation_fraction: f64,
}

impl Default for EarlyStopping {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 1e-4,
            validation_fraction: 0.1,
        }
    }
}

// =============================================================================
// Training History
// =============================================================================

/// Training history for gradient boosting
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training loss at each iteration
    pub train_loss: Vec<f64>,
    /// Validation loss at each iteration (if early stopping enabled)
    pub val_loss: Vec<f64>,
    /// Learning rate at each iteration
    pub learning_rates: Vec<f64>,
    /// Iteration at which training stopped (if early stopping triggered)
    pub stopped_at: Option<usize>,
}

// =============================================================================
// Gradient Boosting Regressor
// =============================================================================

/// Gradient Boosting Regressor
///
/// Builds an additive ensemble of decision trees by sequentially fitting
/// trees to the negative gradient of the loss function.
///
/// ## Features
///
/// - Multiple loss functions: squared error, absolute error, Huber
/// - Learning rate scheduling (constant, linear decay, exponential decay)
/// - Early stopping with validation set
/// - Stochastic gradient boosting via subsample
/// - Feature importance from tree impurity decrease
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingRegressor {
    /// Number of boosting stages (trees)
    pub n_estimators: usize,
    /// Loss function
    pub loss: RegressionLoss,
    /// Learning rate schedule
    pub learning_rate_schedule: LearningRateSchedule,
    /// Maximum depth of individual trees
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required at a leaf
    pub min_samples_leaf: usize,
    /// Maximum features to consider at each split (None for all)
    pub max_features: Option<usize>,
    /// Subsample ratio of the training data (1.0 = no subsampling)
    pub subsample: f64,
    /// Early stopping configuration (None to disable)
    pub early_stopping: Option<EarlyStopping>,
    /// Alpha parameter for Huber loss
    pub alpha: f64,
    /// Random seed
    pub random_state: Option<u64>,
    /// Enable warm start to add estimators incrementally
    pub warm_start: bool,

    // Fitted parameters
    estimators: Option<Vec<DecisionTreeRegressor>>,
    init_prediction: Option<f64>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<f64>>,
    training_history: Option<TrainingHistory>,
}

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientBoostingRegressor {
    /// Create a new Gradient Boosting Regressor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            loss: RegressionLoss::default(),
            learning_rate_schedule: LearningRateSchedule::default(),
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            subsample: 1.0,
            early_stopping: None,
            alpha: 0.9,
            random_state: None,
            warm_start: false,
            estimators: None,
            init_prediction: None,
            n_features: None,
            feature_importances: None,
            training_history: None,
        }
    }

    /// Set the number of estimators
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators.max(1);
        self
    }

    /// Set the loss function
    #[must_use]
    pub fn with_loss(mut self, loss: RegressionLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set a constant learning rate
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate_schedule =
            LearningRateSchedule::Constant(learning_rate.clamp(0.001, 1.0));
        self
    }

    /// Set the learning rate schedule
    #[must_use]
    pub fn with_learning_rate_schedule(mut self, schedule: LearningRateSchedule) -> Self {
        self.learning_rate_schedule = schedule;
        self
    }

    /// Set maximum tree depth
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples to split
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split.max(2);
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf.max(1);
        self
    }

    /// Set maximum features for splits
    #[must_use]
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set subsample ratio
    #[must_use]
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample.clamp(0.1, 1.0);
        self
    }

    /// Enable early stopping
    #[must_use]
    pub fn with_early_stopping(mut self, early_stopping: EarlyStopping) -> Self {
        self.early_stopping = Some(early_stopping);
        self
    }

    /// Set alpha for Huber loss
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set random state
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Get training history
    #[must_use]
    pub fn training_history(&self) -> Option<&TrainingHistory> {
        self.training_history.as_ref()
    }

    /// Get the individual estimators
    #[must_use]
    pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]> {
        self.estimators.as_deref()
    }

    /// Get number of actual estimators (may be less than n_estimators if early stopping)
    #[must_use]
    pub fn n_estimators_actual(&self) -> Option<usize> {
        self.estimators.as_ref().map(|e| e.len())
    }

    /// Get the initial prediction (constant term from training)
    #[must_use]
    pub fn init_prediction(&self) -> Option<f64> {
        self.init_prediction
    }

    /// Staged predictions - yields predictions after each stage
    pub fn staged_predict<'a>(
        &'a self,
        x: &'a Array2<f64>,
    ) -> Result<impl Iterator<Item = Array1<f64>> + 'a> {
        check_is_fitted(&self.estimators, "staged_predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let estimators = self.estimators.as_ref().unwrap();
        let init = self.init_prediction.unwrap();
        let n_samples = x.nrows();

        Ok(StagedPredictIterator {
            x,
            estimators,
            current_predictions: Array1::from_elem(n_samples, init),
            iteration: 0,
            learning_rate_schedule: self.learning_rate_schedule,
            n_estimators: self.n_estimators,
        })
    }

    /// Sample indices for stochastic gradient boosting
    fn sample_indices(&self, n_samples: usize, rng: &mut StdRng) -> Vec<usize> {
        sample_subsample_indices(self.subsample, n_samples, rng)
    }

    /// Compute feature importances from all trees
    fn compute_feature_importances(&mut self) {
        if let Some(ref estimators) = self.estimators {
            let n_features = self.n_features.unwrap();
            let mut importances = Array1::zeros(n_features);

            for tree in estimators {
                if let Some(tree_imp) = tree.feature_importance() {
                    importances = importances + tree_imp;
                }
            }

            // Normalize
            let total: f64 = importances.sum();
            if total > 0.0 {
                importances.mapv_inplace(|v| v / total);
            }

            self.feature_importances = Some(importances);
        }
    }
}

/// Iterator for staged predictions
struct StagedPredictIterator<'a> {
    x: &'a Array2<f64>,
    estimators: &'a [DecisionTreeRegressor],
    current_predictions: Array1<f64>,
    iteration: usize,
    learning_rate_schedule: LearningRateSchedule,
    n_estimators: usize,
}

impl<'a> Iterator for StagedPredictIterator<'a> {
    type Item = Array1<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.estimators.len() {
            return None;
        }

        let tree = &self.estimators[self.iteration];
        let lr = self
            .learning_rate_schedule
            .get_lr(self.iteration, self.n_estimators);

        if let Ok(tree_pred) = tree.predict(self.x) {
            self.current_predictions = &self.current_predictions + &(tree_pred * lr);
        }

        self.iteration += 1;
        Some(self.current_predictions.clone())
    }
}

impl Model for GradientBoostingRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features = Some(n_features);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Split data for early stopping if enabled
        let (x_train, y_train, x_val, y_val) = if let Some(ref early_stopping) = self.early_stopping
        {
            let n_val = (n_samples as f64 * early_stopping.validation_fraction).ceil() as usize;
            let n_train = n_samples - n_val;

            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let train_indices = &indices[..n_train];
            let val_indices = &indices[n_train..];

            let mut x_train = Array2::zeros((n_train, n_features));
            let mut y_train = Array1::zeros(n_train);
            for (i, &idx) in train_indices.iter().enumerate() {
                x_train.row_mut(i).assign(&x.row(idx));
                y_train[i] = y[idx];
            }

            let mut x_val = Array2::zeros((n_val, n_features));
            let mut y_val = Array1::zeros(n_val);
            for (i, &idx) in val_indices.iter().enumerate() {
                x_val.row_mut(i).assign(&x.row(idx));
                y_val[i] = y[idx];
            }

            (x_train, y_train, Some(x_val), Some(y_val))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        let n_train = x_train.nrows();

        // Initialize predictions with constant
        let init = self.loss.init_prediction(&y_train);
        self.init_prediction = Some(init);

        let mut predictions = Array1::from_elem(n_train, init);
        let mut val_predictions = x_val.as_ref().map(|xv| Array1::from_elem(xv.nrows(), init));

        let mut estimators = Vec::with_capacity(self.n_estimators);
        let mut history = TrainingHistory::default();

        // Early stopping state
        let mut best_val_loss = f64::INFINITY;
        let mut no_improvement_count = 0;

        // Pre-allocate subsample buffers (reused across iterations)
        let subsample_size = if self.subsample < 1.0 {
            ((n_train as f64 * self.subsample).ceil() as usize).max(1)
        } else {
            n_train
        };
        let mut x_subsample = Array2::zeros((subsample_size, n_features));
        let mut y_subsample = Array1::zeros(subsample_size);

        for iteration in 0..self.n_estimators {
            let lr = self
                .learning_rate_schedule
                .get_lr(iteration, self.n_estimators);
            history.learning_rates.push(lr);

            // Compute negative gradient
            let neg_gradient = self
                .loss
                .negative_gradient(&y_train, &predictions, self.alpha);

            // Sample indices for stochastic gradient boosting
            let sample_indices = self.sample_indices(n_train, &mut rng);

            // Create training subset (reuse pre-allocated buffers)
            let n_subsample = sample_indices.len();
            // Resize if needed (in case subsample ratio changed)
            if n_subsample != x_subsample.nrows() {
                x_subsample = Array2::zeros((n_subsample, n_features));
                y_subsample = Array1::zeros(n_subsample);
            }
            for (i, &idx) in sample_indices.iter().enumerate() {
                x_subsample.row_mut(i).assign(&x_train.row(idx));
                y_subsample[i] = neg_gradient[idx];
            }

            // Fit tree to negative gradient
            let mut tree = DecisionTreeRegressor::new()
                .with_max_depth(self.max_depth)
                .with_min_samples_split(self.min_samples_split)
                .with_min_samples_leaf(self.min_samples_leaf)
                .with_criterion(SplitCriterion::Mse);

            if let Some(max_f) = self.max_features {
                tree = tree.with_max_features(Some(max_f));
            }

            tree = tree.with_random_state(rng.random());
            tree.fit(&x_subsample, &y_subsample)?;

            // Update predictions in-place (avoids temporary allocations)
            let tree_pred = tree.predict(&x_train)?;
            predictions.zip_mut_with(&tree_pred, |p, &tp| *p += tp * lr);

            // Compute training loss
            let train_loss = self.loss.loss(&y_train, &predictions, self.alpha);
            history.train_loss.push(train_loss);

            // Early stopping check
            if let (Some(ref xv), Some(ref yv), Some(ref mut vp)) =
                (&x_val, &y_val, &mut val_predictions)
            {
                let tree_val_pred = tree.predict(xv)?;
                vp.zip_mut_with(&tree_val_pred, |p, &tp| *p += tp * lr);

                let val_loss = self.loss.loss(yv, vp, self.alpha);
                history.val_loss.push(val_loss);

                let early_stopping = self.early_stopping.as_ref().unwrap();
                if val_loss < best_val_loss - early_stopping.min_delta {
                    best_val_loss = val_loss;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                    if no_improvement_count >= early_stopping.patience {
                        history.stopped_at = Some(iteration + 1);
                        estimators.push(tree);
                        break;
                    }
                }
            }

            estimators.push(tree);
        }

        self.estimators = Some(estimators);
        self.training_history = Some(history);
        self.compute_feature_importances();

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.estimators, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let estimators = self.estimators.as_ref().unwrap();
        let init = self.init_prediction.unwrap();
        let n_samples = x.nrows();

        // Parallel prediction: split samples across threads, each thread traverses
        // all trees for its chunk. Minimum chunk size avoids overhead for small inputs.
        #[cfg(feature = "parallel")]
        {
            const MIN_PARALLEL_SAMPLES: usize = 256;
            if n_samples >= MIN_PARALLEL_SAMPLES {
                // Pre-compute learning rates for each estimator
                let lrs: Vec<f64> = (0..estimators.len())
                    .map(|i| self.learning_rate_schedule.get_lr(i, self.n_estimators))
                    .collect();

                let chunk_size = (n_samples / rayon::current_num_threads()).max(64);
                let chunks: Vec<Array1<f64>> = (0..n_samples)
                    .collect::<Vec<_>>()
                    .par_chunks(chunk_size)
                    .map(|indices| {
                        // Build sub-matrix for this chunk
                        let chunk_len = indices.len();
                        let sub_x = x.select(Axis(0), indices);
                        let mut chunk_preds = Array1::from_elem(chunk_len, init);

                        for (tree_idx, tree) in estimators.iter().enumerate() {
                            // tree.predict is infallible on valid input since model is fitted
                            if let Ok(tree_pred) = tree.predict(&sub_x) {
                                let lr = lrs[tree_idx];
                                chunk_preds.zip_mut_with(&tree_pred, |p, &tp| *p += tp * lr);
                            }
                        }

                        chunk_preds
                    })
                    .collect();

                // Assemble results
                let mut predictions = Array1::zeros(n_samples);
                let mut offset = 0;
                for chunk in chunks {
                    let len = chunk.len();
                    predictions
                        .slice_mut(ndarray::s![offset..offset + len])
                        .assign(&chunk);
                    offset += len;
                }

                return Ok(predictions);
            }
        }

        // Sequential fallback (also used when parallel feature is disabled)
        let mut predictions = Array1::from_elem(n_samples, init);

        for (i, tree) in estimators.iter().enumerate() {
            let lr = self.learning_rate_schedule.get_lr(i, self.n_estimators);
            let tree_pred = tree.predict(x)?;
            // In-place update: predictions += tree_pred * lr (avoids 2 temporary allocations)
            predictions.zip_mut_with(&tree_pred, |p, &tp| *p += tp * lr);
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.estimators.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importances.clone()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("n_estimators", 50, 500)
            .float_log("learning_rate", 0.01, 0.3)
            .int("max_depth", 2, 10)
            .int("min_samples_split", 2, 20)
            .int("min_samples_leaf", 1, 10)
            .float("subsample", 0.5, 1.0)
            .categorical(
                "loss",
                vec![
                    "squared_error".to_string(),
                    "absolute_error".to_string(),
                    "huber".to_string(),
                ],
            )
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// Gradient Boosting Classifier
// =============================================================================

/// Gradient Boosting Classifier
///
/// Builds an additive ensemble of decision trees for classification using
/// gradient boosting. For binary classification, fits trees to the log-odds.
/// For multiclass, uses one-vs-all with K trees per stage.
///
/// ## Features
///
/// - Deviance (log-loss) or exponential loss
/// - Learning rate scheduling
/// - Early stopping with validation set
/// - Stochastic gradient boosting via subsample
/// - Feature importance from tree impurity decrease
/// - Probability predictions via sigmoid/softmax
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingClassifier {
    /// Number of boosting stages
    pub n_estimators: usize,
    /// Loss function
    pub loss: ClassificationLoss,
    /// Learning rate schedule
    pub learning_rate_schedule: LearningRateSchedule,
    /// Maximum depth of individual trees
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required at a leaf
    pub min_samples_leaf: usize,
    /// Maximum features to consider at each split
    pub max_features: Option<usize>,
    /// Subsample ratio of the training data
    pub subsample: f64,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStopping>,
    /// Random seed
    pub random_state: Option<u64>,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,
    /// Enable warm start to add estimators incrementally
    pub warm_start: bool,

    // Fitted parameters
    estimators: Option<Vec<Vec<DecisionTreeRegressor>>>, // [n_estimators][n_classes]
    init_predictions: Option<Array1<f64>>,               // [n_classes] for multiclass
    classes: Option<Array1<f64>>,
    n_classes: Option<usize>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<f64>>,
    training_history: Option<TrainingHistory>,
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientBoostingClassifier {
    /// Create a new Gradient Boosting Classifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            loss: ClassificationLoss::default(),
            learning_rate_schedule: LearningRateSchedule::default(),
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            subsample: 1.0,
            early_stopping: None,
            random_state: None,
            class_weight: ClassWeight::Uniform,
            warm_start: false,
            estimators: None,
            init_predictions: None,
            classes: None,
            n_classes: None,
            n_features: None,
            feature_importances: None,
            training_history: None,
        }
    }

    /// Set the number of estimators
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators.max(1);
        self
    }

    /// Set the loss function
    #[must_use]
    pub fn with_loss(mut self, loss: ClassificationLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set a constant learning rate
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate_schedule =
            LearningRateSchedule::Constant(learning_rate.clamp(0.001, 1.0));
        self
    }

    /// Set the learning rate schedule
    #[must_use]
    pub fn with_learning_rate_schedule(mut self, schedule: LearningRateSchedule) -> Self {
        self.learning_rate_schedule = schedule;
        self
    }

    /// Set maximum tree depth
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples to split
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split.max(2);
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf.max(1);
        self
    }

    /// Set maximum features for splits
    #[must_use]
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set subsample ratio
    #[must_use]
    pub fn with_subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample.clamp(0.1, 1.0);
        self
    }

    /// Enable early stopping
    #[must_use]
    pub fn with_early_stopping(mut self, early_stopping: EarlyStopping) -> Self {
        self.early_stopping = Some(early_stopping);
        self
    }

    /// Set random state
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set class weights for handling imbalanced data
    ///
    /// # Arguments
    /// * `class_weight` - Weight specification: `Uniform`, `Balanced`, or `Custom`
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Get training history
    #[must_use]
    pub fn training_history(&self) -> Option<&TrainingHistory> {
        self.training_history.as_ref()
    }

    /// Get the unique class labels
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get number of actual estimators
    #[must_use]
    pub fn n_estimators_actual(&self) -> Option<usize> {
        self.estimators.as_ref().map(|e| e.len())
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.estimators, "predict_proba")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let n_samples = x.nrows();
        let n_classes = self.n_classes.unwrap();
        let estimators = self.estimators.as_ref().unwrap();
        let init = self.init_predictions.as_ref().unwrap();

        // For binary classification, we only have 1 tree per stage
        let n_trees_per_stage = if n_classes == 2 { 1 } else { n_classes };

        // Parallel prediction: split samples across threads
        #[cfg(feature = "parallel")]
        {
            const MIN_PARALLEL_SAMPLES: usize = 256;
            if n_samples >= MIN_PARALLEL_SAMPLES {
                // Pre-compute learning rates
                let lrs: Vec<f64> = (0..estimators.len())
                    .map(|i| self.learning_rate_schedule.get_lr(i, self.n_estimators))
                    .collect();
                let init_clone = init.clone();

                let chunk_size = (n_samples / rayon::current_num_threads()).max(64);
                let chunks: Vec<Array2<f64>> = (0..n_samples)
                    .collect::<Vec<_>>()
                    .par_chunks(chunk_size)
                    .map(|indices| {
                        let chunk_len = indices.len();
                        let sub_x = x.select(Axis(0), indices);
                        let mut raw = Array2::zeros((chunk_len, n_trees_per_stage));
                        for k in 0..n_trees_per_stage {
                            for i in 0..chunk_len {
                                raw[[i, k]] = init_clone[k];
                            }
                        }
                        for (iter_idx, trees_at_stage) in estimators.iter().enumerate() {
                            let lr = lrs[iter_idx];
                            for (k, tree) in trees_at_stage.iter().enumerate() {
                                if let Ok(tree_pred) = tree.predict(&sub_x) {
                                    for i in 0..chunk_len {
                                        raw[[i, k]] += lr * tree_pred[i];
                                    }
                                }
                            }
                        }
                        // Convert to probabilities
                        let mut probas = Array2::zeros((chunk_len, n_classes));
                        if n_classes == 2 {
                            for i in 0..chunk_len {
                                let p = sigmoid(raw[[i, 0]]);
                                probas[[i, 0]] = 1.0 - p;
                                probas[[i, 1]] = p;
                            }
                        } else {
                            for i in 0..chunk_len {
                                let row = raw.row(i);
                                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
                                for j in 0..n_classes {
                                    probas[[i, j]] = (raw[[i, j]] - max_val).exp() / exp_sum;
                                }
                            }
                        }
                        probas
                    })
                    .collect();

                // Assemble results
                let mut probas = Array2::zeros((n_samples, n_classes));
                let mut offset = 0;
                for chunk in chunks {
                    let len = chunk.nrows();
                    probas
                        .slice_mut(ndarray::s![offset..offset + len, ..])
                        .assign(&chunk);
                    offset += len;
                }

                return Ok(probas);
            }
        }

        // Sequential fallback
        // Initialize raw predictions (log-odds space)
        let mut raw_predictions = Array2::zeros((n_samples, n_trees_per_stage));
        for k in 0..n_trees_per_stage {
            for i in 0..n_samples {
                raw_predictions[[i, k]] = init[k];
            }
        }

        // Add tree predictions
        for (iteration, trees_at_stage) in estimators.iter().enumerate() {
            let lr = self
                .learning_rate_schedule
                .get_lr(iteration, self.n_estimators);
            for (k, tree) in trees_at_stage.iter().enumerate() {
                let tree_pred = tree.predict(x)?;
                for i in 0..n_samples {
                    raw_predictions[[i, k]] += lr * tree_pred[i];
                }
            }
        }

        // Convert to probabilities
        let mut probas = Array2::zeros((n_samples, n_classes));

        if n_classes == 2 {
            // Binary classification: sigmoid
            for i in 0..n_samples {
                let p = sigmoid(raw_predictions[[i, 0]]);
                probas[[i, 0]] = 1.0 - p;
                probas[[i, 1]] = p;
            }
        } else {
            // Multiclass: softmax
            for i in 0..n_samples {
                let row = raw_predictions.row(i);
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
                for j in 0..n_classes {
                    probas[[i, j]] = (raw_predictions[[i, j]] - max_val).exp() / exp_sum;
                }
            }
        }

        Ok(probas)
    }

    /// Sample indices for stochastic gradient boosting
    fn sample_indices(&self, n_samples: usize, rng: &mut StdRng) -> Vec<usize> {
        sample_subsample_indices(self.subsample, n_samples, rng)
    }

    /// Compute feature importances from all trees
    fn compute_feature_importances(&mut self) {
        if let Some(ref estimators) = self.estimators {
            let n_features = self.n_features.unwrap();
            let mut importances = Array1::zeros(n_features);

            for trees_at_stage in estimators {
                for tree in trees_at_stage {
                    if let Some(tree_imp) = tree.feature_importance() {
                        importances = importances + tree_imp;
                    }
                }
            }

            // Normalize
            let total: f64 = importances.sum();
            if total > 0.0 {
                importances.mapv_inplace(|v| v / total);
            }

            self.feature_importances = Some(importances);
        }
    }

    /// Compute log-loss for validation
    fn compute_log_loss(&self, y: &Array1<f64>, probas: &Array2<f64>) -> f64 {
        super::compute_log_loss(y, probas, self.classes.as_ref().unwrap())
    }
}

/// Sigmoid function
use super::sigmoid;

fn sample_subsample_indices(subsample: f64, n_samples: usize, rng: &mut StdRng) -> Vec<usize> {
    if subsample >= 1.0 {
        (0..n_samples).collect()
    } else {
        let n_subsample = (n_samples as f64 * subsample).ceil() as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);
        indices.truncate(n_subsample);
        indices
    }
}

impl Model for GradientBoostingClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features = Some(n_features);

        // Find unique classes
        let classes = super::get_unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "GradientBoostingClassifier requires at least 2 classes",
            ));
        }

        let n_classes = classes.len();
        self.classes = Some(classes.clone());
        self.n_classes = Some(n_classes);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Split data for early stopping if enabled
        let (x_train, y_train, x_val, y_val) = if let Some(ref early_stopping) = self.early_stopping
        {
            let n_val = (n_samples as f64 * early_stopping.validation_fraction).ceil() as usize;
            let n_train = n_samples - n_val;

            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let train_indices = &indices[..n_train];
            let val_indices = &indices[n_train..];

            let mut x_train = Array2::zeros((n_train, n_features));
            let mut y_train = Array1::zeros(n_train);
            for (i, &idx) in train_indices.iter().enumerate() {
                x_train.row_mut(i).assign(&x.row(idx));
                y_train[i] = y[idx];
            }

            let mut x_val = Array2::zeros((n_val, n_features));
            let mut y_val = Array1::zeros(n_val);
            for (i, &idx) in val_indices.iter().enumerate() {
                x_val.row_mut(i).assign(&x.row(idx));
                y_val[i] = y[idx];
            }

            (x_train, y_train, Some(x_val), Some(y_val))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        let n_train = x_train.nrows();

        // For binary classification, we only need one tree per stage
        // For multiclass, we need K trees per stage (one-vs-all)
        let n_trees_per_stage = if n_classes == 2 { 1 } else { n_classes };

        // Initialize predictions
        // For binary: init with log(p/(1-p)) where p is proportion of class 1
        // For multiclass: init with log(class_proportion)
        let mut init_predictions = Array1::zeros(n_trees_per_stage);
        if n_classes == 2 {
            let n_pos: usize = y_train
                .iter()
                .filter(|&&yi| (yi - classes[1]).abs() < 1e-10)
                .count();
            let p = (n_pos as f64 + 1.0) / (n_train as f64 + 2.0); // Laplace smoothing
            init_predictions[0] = (p / (1.0 - p)).ln();
        } else {
            for (k, &class_val) in classes.iter().enumerate() {
                let n_class: usize = y_train
                    .iter()
                    .filter(|&&yi| (yi - class_val).abs() < 1e-10)
                    .count();
                let p = (n_class as f64 + 1.0) / (n_train as f64 + n_classes as f64);
                init_predictions[k] = p.ln();
            }
        }
        self.init_predictions = Some(init_predictions.clone());

        // Initialize raw predictions
        let mut raw_predictions = Array2::zeros((n_train, n_trees_per_stage));
        for k in 0..n_trees_per_stage {
            for i in 0..n_train {
                raw_predictions[[i, k]] = init_predictions[k];
            }
        }

        let mut val_raw_predictions = x_val.as_ref().map(|xv| {
            let mut vp = Array2::zeros((xv.nrows(), n_trees_per_stage));
            for k in 0..n_trees_per_stage {
                for i in 0..xv.nrows() {
                    vp[[i, k]] = init_predictions[k];
                }
            }
            vp
        });

        let mut estimators: Vec<Vec<DecisionTreeRegressor>> = Vec::with_capacity(self.n_estimators);
        let mut history = TrainingHistory::default();

        // Early stopping state
        let mut best_val_loss = f64::INFINITY;
        let mut no_improvement_count = 0;

        // Pre-allocate subsample buffers (reused across iterations and classes)
        let subsample_size_cls = if self.subsample < 1.0 {
            ((n_train as f64 * self.subsample).ceil() as usize).max(1)
        } else {
            n_train
        };
        let mut x_subsample = Array2::zeros((subsample_size_cls, n_features));
        let mut y_subsample = Array1::zeros(subsample_size_cls);

        for iteration in 0..self.n_estimators {
            let lr = self
                .learning_rate_schedule
                .get_lr(iteration, self.n_estimators);
            history.learning_rates.push(lr);

            // Sample indices for stochastic gradient boosting
            let sample_indices = self.sample_indices(n_train, &mut rng);

            let mut trees_at_stage = Vec::with_capacity(n_trees_per_stage);

            for k in 0..n_trees_per_stage {
                // Compute negative gradient (pseudo-residuals)
                let neg_gradient = if n_classes == 2 {
                    // Binary: y_binary - sigmoid(F)
                    Array1::from_iter((0..n_train).map(|i| {
                        let y_binary = if (y_train[i] - classes[1]).abs() < 1e-10 {
                            1.0
                        } else {
                            0.0
                        };
                        let p = sigmoid(raw_predictions[[i, 0]]);
                        y_binary - p
                    }))
                } else {
                    // Multiclass: y_one_hot - softmax(F)_k
                    Array1::from_iter((0..n_train).map(|i| {
                        let y_one_hot = if (y_train[i] - classes[k]).abs() < 1e-10 {
                            1.0
                        } else {
                            0.0
                        };

                        // Compute softmax probability for class k (inline without temporary Vec)
                        let mut max_val = f64::NEG_INFINITY;
                        for j in 0..n_trees_per_stage {
                            let v = raw_predictions[[i, j]];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut exp_sum = 0.0f64;
                        for j in 0..n_trees_per_stage {
                            exp_sum += (raw_predictions[[i, j]] - max_val).exp();
                        }
                        let p_k = (raw_predictions[[i, k]] - max_val).exp() / exp_sum;

                        y_one_hot - p_k
                    }))
                };

                // Create training subset (reuse pre-allocated buffers)
                let n_subsample = sample_indices.len();
                if n_subsample != x_subsample.nrows() {
                    x_subsample = Array2::zeros((n_subsample, n_features));
                    y_subsample = Array1::zeros(n_subsample);
                }
                for (i, &idx) in sample_indices.iter().enumerate() {
                    x_subsample.row_mut(i).assign(&x_train.row(idx));
                    y_subsample[i] = neg_gradient[idx];
                }

                // Fit tree to negative gradient
                let mut tree = DecisionTreeRegressor::new()
                    .with_max_depth(self.max_depth)
                    .with_min_samples_split(self.min_samples_split)
                    .with_min_samples_leaf(self.min_samples_leaf)
                    .with_criterion(SplitCriterion::Mse);

                if let Some(max_f) = self.max_features {
                    tree = tree.with_max_features(Some(max_f));
                }

                tree = tree.with_random_state(rng.random());
                tree.fit(&x_subsample, &y_subsample)?;

                // Update raw predictions
                let tree_pred = tree.predict(&x_train)?;
                for i in 0..n_train {
                    raw_predictions[[i, k]] += lr * tree_pred[i];
                }

                // Update validation predictions
                if let (Some(ref xv), Some(ref mut vrp)) = (&x_val, &mut val_raw_predictions) {
                    let tree_val_pred = tree.predict(xv)?;
                    for i in 0..xv.nrows() {
                        vrp[[i, k]] += lr * tree_val_pred[i];
                    }
                }

                trees_at_stage.push(tree);
            }

            // Compute training loss
            let train_probas = self.raw_to_proba(&raw_predictions);
            let train_loss = self.compute_log_loss(&y_train, &train_probas);
            history.train_loss.push(train_loss);

            // Early stopping check
            if let (Some(ref yv), Some(ref vrp)) = (&y_val, &val_raw_predictions) {
                let val_probas = self.raw_to_proba(vrp);
                let val_loss = self.compute_log_loss(yv, &val_probas);
                history.val_loss.push(val_loss);

                let early_stopping = self.early_stopping.as_ref().unwrap();
                if val_loss < best_val_loss - early_stopping.min_delta {
                    best_val_loss = val_loss;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                    if no_improvement_count >= early_stopping.patience {
                        history.stopped_at = Some(iteration + 1);
                        estimators.push(trees_at_stage);
                        break;
                    }
                }
            }

            estimators.push(trees_at_stage);
        }

        self.estimators = Some(estimators);
        self.training_history = Some(history);
        self.compute_feature_importances();

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let max_idx = probas
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[max_idx];
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.estimators.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importances.clone()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("n_estimators", 50, 500)
            .float_log("learning_rate", 0.01, 0.3)
            .int("max_depth", 2, 10)
            .int("min_samples_split", 2, 20)
            .int("min_samples_leaf", 1, 10)
            .float("subsample", 0.5, 1.0)
            .categorical(
                "loss",
                vec!["deviance".to_string(), "exponential".to_string()],
            )
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl GradientBoostingClassifier {
    fn raw_to_proba(&self, raw: &Array2<f64>) -> Array2<f64> {
        super::raw_to_proba(raw, self.n_classes.unwrap())
    }
}

// =============================================================================
impl super::traits::WarmStartModel for GradientBoostingRegressor {
    fn set_warm_start(&mut self, warm_start: bool) {
        self.warm_start = warm_start;
    }

    fn warm_start(&self) -> bool {
        self.warm_start
    }

    fn n_estimators_fitted(&self) -> usize {
        self.estimators.as_ref().map_or(0, |e| e.len())
    }
}

impl super::traits::WarmStartModel for GradientBoostingClassifier {
    fn set_warm_start(&mut self, warm_start: bool) {
        self.warm_start = warm_start;
    }

    fn warm_start(&self) -> bool {
        self.warm_start
    }

    fn n_estimators_fitted(&self) -> usize {
        self.estimators.as_ref().map_or(0, |e| e.len())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        // y ≈ x1 + 2*x2
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5, 3.0, 1.0, 3.0, 2.0,
                1.0, 3.0, 2.0, 3.0, 3.0, 3.0, 4.0, 1.0, 4.0, 2.0, 4.0, 3.0, 4.0, 4.0, 5.0, 1.0,
                5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0, 5.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            3.0, 5.0, 4.0, 6.0, 4.5, 7.5, 5.0, 7.0, 7.0, 8.0, 9.0, 6.0, 8.0, 10.0, 12.0, 7.0, 9.0,
            11.0, 13.0, 15.0,
        ]);
        (x, y)
    }

    fn make_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Linearly separable
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 1.0, 2.0, 1.5, 2.5, 2.0, 1.0, 2.5, 1.5,
                1.2, 1.8, 1.8, 1.2, // Class 0
                5.0, 5.0, 5.5, 5.5, 6.0, 6.0, 6.5, 6.5, 5.0, 6.0, 5.5, 6.5, 6.0, 5.0, 6.5, 5.5,
                5.2, 5.8, 5.8, 5.2, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);
        (x, y)
    }

    fn make_multiclass_data() -> (Array2<f64>, Array1<f64>) {
        // 3 classes
        let x = Array2::from_shape_vec(
            (15, 2),
            vec![
                1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 0.8, 0.8, 1.3, 1.0, // Class 0
                4.0, 4.0, 4.2, 4.2, 4.5, 4.5, 3.8, 3.8, 4.3, 4.0, // Class 1
                7.0, 7.0, 7.2, 7.2, 7.5, 7.5, 6.8, 6.8, 7.3, 7.0, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ]);
        (x, y)
    }

    #[test]
    fn test_gradient_boosting_regressor_fit_predict() {
        let (x, y) = make_regression_data();

        let mut reg = GradientBoostingRegressor::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3))
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());
        assert_eq!(reg.n_features(), Some(2));
        assert_eq!(reg.n_estimators_actual(), Some(50));

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // Should fit reasonably well
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / 20.0;
        assert!(mse < 10.0, "MSE was {}", mse);
    }

    #[test]
    fn test_gradient_boosting_regressor_feature_importance() {
        let (x, y) = make_regression_data();

        let mut reg = GradientBoostingRegressor::new()
            .with_n_estimators(20)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();

        let importance = reg.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Should sum to ~1
        let total: f64 = importance.sum();
        assert!((total - 1.0).abs() < 0.01, "Total importance was {}", total);
    }

    #[test]
    fn test_gradient_boosting_regressor_different_losses() {
        let (x, y) = make_regression_data();

        // Test squared error
        let mut reg_se = GradientBoostingRegressor::new()
            .with_loss(RegressionLoss::SquaredError)
            .with_n_estimators(20)
            .with_random_state(42);
        reg_se.fit(&x, &y).unwrap();
        let pred_se = reg_se.predict(&x).unwrap();

        // Test absolute error
        let mut reg_ae = GradientBoostingRegressor::new()
            .with_loss(RegressionLoss::AbsoluteError)
            .with_n_estimators(20)
            .with_random_state(42);
        reg_ae.fit(&x, &y).unwrap();
        let pred_ae = reg_ae.predict(&x).unwrap();

        // Both should produce reasonable predictions
        assert_eq!(pred_se.len(), 20);
        assert_eq!(pred_ae.len(), 20);
    }

    #[test]
    fn test_gradient_boosting_regressor_subsample() {
        let (x, y) = make_regression_data();

        let mut reg = GradientBoostingRegressor::new()
            .with_n_estimators(20)
            .with_subsample(0.8)
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
    }

    #[test]
    fn test_gradient_boosting_regressor_learning_rate_decay() {
        let (x, y) = make_regression_data();

        let mut reg = GradientBoostingRegressor::new()
            .with_n_estimators(20)
            .with_learning_rate_schedule(LearningRateSchedule::LinearDecay {
                initial: 0.2,
                min_lr: 0.01,
            })
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        let history = reg.training_history().unwrap();
        // Learning rates should decrease
        assert!(
            history.learning_rates[0] > history.learning_rates[history.learning_rates.len() - 1]
        );
    }

    #[test]
    fn test_gradient_boosting_regressor_early_stopping() {
        let (x, y) = make_regression_data();

        let mut reg = GradientBoostingRegressor::new()
            .with_n_estimators(100)
            .with_early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1e-4,
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        // Should have stopped early or completed
        assert!(reg.is_fitted());
        // Validation loss should be recorded
        let history = reg.training_history().unwrap();
        assert!(!history.val_loss.is_empty());
    }

    #[test]
    fn test_gradient_boosting_classifier_binary() {
        let (x, y) = make_classification_data();

        let mut clf = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3))
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.n_features(), Some(2));

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // Should classify most correctly
        let accuracy: f64 = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count() as f64
            / 20.0;
        assert!(accuracy > 0.8, "Accuracy was {}", accuracy);
    }

    #[test]
    fn test_gradient_boosting_classifier_proba() {
        let (x, y) = make_classification_data();

        let mut clf = GradientBoostingClassifier::new()
            .with_n_estimators(20)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let probas = clf.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[20, 2]);

        // Probabilities should sum to 1
        for i in 0..20 {
            let row_sum: f64 = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-6);
        }

        // Probabilities should be in [0, 1]
        for p in probas.iter() {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }

    #[test]
    fn test_gradient_boosting_classifier_multiclass() {
        let (x, y) = make_multiclass_data();

        let mut clf = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1)
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.classes().unwrap().len(), 3);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 15);

        let probas = clf.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[15, 3]);

        // Probabilities should sum to 1
        for i in 0..15 {
            let row_sum: f64 = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gradient_boosting_classifier_feature_importance() {
        let (x, y) = make_classification_data();

        let mut clf = GradientBoostingClassifier::new()
            .with_n_estimators(20)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let importance = clf.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Non-negative importances
        assert!(importance.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_gradient_boosting_classifier_early_stopping() {
        let (x, y) = make_classification_data();

        let mut clf = GradientBoostingClassifier::new()
            .with_n_estimators(100)
            .with_early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1e-4,
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        let history = clf.training_history().unwrap();
        assert!(!history.val_loss.is_empty());
    }

    #[test]
    fn test_search_spaces() {
        let reg = GradientBoostingRegressor::new();
        let space = reg.search_space();
        assert!(space.n_dims() > 0);

        let clf = GradientBoostingClassifier::new();
        let space = clf.search_space();
        assert!(space.n_dims() > 0);
    }

    #[test]
    fn test_not_fitted_errors() {
        let reg = GradientBoostingRegressor::new();
        let x = Array2::zeros((2, 2));
        assert!(reg.predict(&x).is_err());

        let clf = GradientBoostingClassifier::new();
        assert!(clf.predict(&x).is_err());
        assert!(clf.predict_proba(&x).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

        let mut clf = GradientBoostingClassifier::new();
        let result = clf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_staged_predict() {
        let (x, y) = make_regression_data();

        let mut reg = GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();

        let staged_preds: Vec<Array1<f64>> = reg.staged_predict(&x).unwrap().collect();
        assert_eq!(staged_preds.len(), 10);

        // Final staged prediction should match regular predict
        let final_pred = reg.predict(&x).unwrap();
        for i in 0..20 {
            assert_abs_diff_eq!(staged_preds[9][i], final_pred[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_learning_rate_schedules() {
        // Constant
        let constant = LearningRateSchedule::Constant(0.1);
        assert_abs_diff_eq!(constant.get_lr(0, 100), 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(constant.get_lr(50, 100), 0.1, epsilon = 1e-10);

        // Linear decay
        let linear = LearningRateSchedule::LinearDecay {
            initial: 0.2,
            min_lr: 0.01,
        };
        assert_abs_diff_eq!(linear.get_lr(0, 100), 0.2, epsilon = 1e-10);
        assert!(linear.get_lr(50, 100) < 0.2);
        assert!(linear.get_lr(99, 100) >= 0.01);

        // Exponential decay
        let exp = LearningRateSchedule::ExponentialDecay {
            initial: 0.2,
            decay: 0.99,
            min_lr: 0.01,
        };
        assert_abs_diff_eq!(exp.get_lr(0, 100), 0.2, epsilon = 1e-10);
        assert!(exp.get_lr(50, 100) < 0.2);
    }

    #[test]
    fn test_huber_loss() {
        let (x, y) = make_regression_data();

        let mut reg = GradientBoostingRegressor::new()
            .with_loss(RegressionLoss::Huber)
            .with_n_estimators(20)
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);
    }
}
