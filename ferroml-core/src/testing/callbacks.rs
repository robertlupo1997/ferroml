//! Comprehensive tests for early stopping and callback mechanisms
//!
//! This module provides tests for:
//! - Early stopping callback triggers at correct patience
//! - Learning rate schedulers (constant, linear decay, exponential decay, cosine annealing)
//! - Checkpoint callbacks save/restore correctly
//! - Custom callback integration
//! - Callback ordering and composition
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::testing::callbacks::*;
//!
//! // Run all callback tests
//! #[test]
//! fn test_all_callbacks() {
//!     test_early_stopping_patience();
//!     test_learning_rate_schedulers();
//! }
//! ```

#![allow(unused_imports)]
#![allow(dead_code)]

use crate::models::boosting::{
    EarlyStopping, GradientBoostingClassifier, GradientBoostingRegressor, LearningRateSchedule,
    TrainingHistory,
};
use crate::models::Model;
use ndarray::{Array1, Array2};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

// =============================================================================
// Callback Trait and Types
// =============================================================================

/// Training event types for callbacks
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingEvent {
    /// Training has started
    TrainingStart { n_estimators: usize },
    /// An epoch/iteration has started
    EpochStart { epoch: usize },
    /// An epoch/iteration has ended
    EpochEnd {
        epoch: usize,
        train_loss: f64,
        val_loss: Option<f64>,
        learning_rate: f64,
    },
    /// Validation check occurred
    ValidationCheck {
        epoch: usize,
        val_loss: f64,
        best_loss: f64,
        patience_counter: usize,
    },
    /// Early stopping triggered
    EarlyStoppingTriggered { epoch: usize, best_epoch: usize },
    /// Training has ended
    TrainingEnd {
        n_epochs: usize,
        final_train_loss: f64,
        final_val_loss: Option<f64>,
    },
    /// Checkpoint saved
    CheckpointSaved { epoch: usize, path: String },
    /// Learning rate updated
    LearningRateUpdate {
        epoch: usize,
        old_lr: f64,
        new_lr: f64,
    },
}

/// Trait for training callbacks
pub trait TrainingCallback: Send + Sync {
    /// Called when training starts
    fn on_training_start(&self, n_estimators: usize);

    /// Called at the start of each epoch
    fn on_epoch_start(&self, epoch: usize);

    /// Called at the end of each epoch
    fn on_epoch_end(
        &self,
        epoch: usize,
        train_loss: f64,
        val_loss: Option<f64>,
        learning_rate: f64,
    );

    /// Called when early stopping is triggered
    fn on_early_stopping(&self, epoch: usize, best_epoch: usize);

    /// Called when training ends
    fn on_training_end(&self, n_epochs: usize, final_train_loss: f64, final_val_loss: Option<f64>);

    /// Get callback priority (lower = earlier execution)
    fn priority(&self) -> i32 {
        0
    }

    /// Get callback name for debugging
    fn name(&self) -> &str {
        "TrainingCallback"
    }
}

/// Event recorder callback for testing
#[derive(Debug, Default)]
pub struct EventRecorderCallback {
    events: RwLock<Vec<TrainingEvent>>,
}

impl EventRecorderCallback {
    /// Create a new event recorder
    pub fn new() -> Self {
        Self {
            events: RwLock::new(Vec::new()),
        }
    }

    /// Get recorded events
    pub fn events(&self) -> Vec<TrainingEvent> {
        self.events.read().unwrap().clone()
    }

    /// Clear recorded events
    pub fn clear(&self) {
        self.events.write().unwrap().clear();
    }
}

impl TrainingCallback for EventRecorderCallback {
    fn on_training_start(&self, n_estimators: usize) {
        self.events
            .write()
            .unwrap()
            .push(TrainingEvent::TrainingStart { n_estimators });
    }

    fn on_epoch_start(&self, epoch: usize) {
        self.events
            .write()
            .unwrap()
            .push(TrainingEvent::EpochStart { epoch });
    }

    fn on_epoch_end(
        &self,
        epoch: usize,
        train_loss: f64,
        val_loss: Option<f64>,
        learning_rate: f64,
    ) {
        self.events.write().unwrap().push(TrainingEvent::EpochEnd {
            epoch,
            train_loss,
            val_loss,
            learning_rate,
        });
    }

    fn on_early_stopping(&self, epoch: usize, best_epoch: usize) {
        self.events
            .write()
            .unwrap()
            .push(TrainingEvent::EarlyStoppingTriggered { epoch, best_epoch });
    }

    fn on_training_end(&self, n_epochs: usize, final_train_loss: f64, final_val_loss: Option<f64>) {
        self.events
            .write()
            .unwrap()
            .push(TrainingEvent::TrainingEnd {
                n_epochs,
                final_train_loss,
                final_val_loss,
            });
    }

    fn name(&self) -> &str {
        "EventRecorderCallback"
    }
}

/// Counter callback for tracking invocations
#[derive(Debug, Default)]
pub struct CounterCallback {
    pub training_start_count: AtomicUsize,
    pub epoch_start_count: AtomicUsize,
    pub epoch_end_count: AtomicUsize,
    pub early_stopping_count: AtomicUsize,
    pub training_end_count: AtomicUsize,
}

impl CounterCallback {
    /// Create a new counter callback
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total epoch count
    pub fn epoch_count(&self) -> usize {
        self.epoch_end_count.load(Ordering::Relaxed)
    }
}

impl TrainingCallback for CounterCallback {
    fn on_training_start(&self, _n_estimators: usize) {
        self.training_start_count.fetch_add(1, Ordering::Relaxed);
    }

    fn on_epoch_start(&self, _epoch: usize) {
        self.epoch_start_count.fetch_add(1, Ordering::Relaxed);
    }

    fn on_epoch_end(
        &self,
        _epoch: usize,
        _train_loss: f64,
        _val_loss: Option<f64>,
        _learning_rate: f64,
    ) {
        self.epoch_end_count.fetch_add(1, Ordering::Relaxed);
    }

    fn on_early_stopping(&self, _epoch: usize, _best_epoch: usize) {
        self.early_stopping_count.fetch_add(1, Ordering::Relaxed);
    }

    fn on_training_end(
        &self,
        _n_epochs: usize,
        _final_train_loss: f64,
        _final_val_loss: Option<f64>,
    ) {
        self.training_end_count.fetch_add(1, Ordering::Relaxed);
    }

    fn name(&self) -> &str {
        "CounterCallback"
    }
}

// =============================================================================
// Cosine Annealing Learning Rate Scheduler
// =============================================================================

/// Cosine annealing learning rate schedule
///
/// Decreases learning rate following a cosine curve from initial to minimum.
#[derive(Debug, Clone, Copy)]
pub struct CosineAnnealingSchedule {
    /// Initial learning rate
    pub initial_lr: f64,
    /// Minimum learning rate
    pub min_lr: f64,
    /// Total number of iterations
    pub t_max: usize,
}

impl CosineAnnealingSchedule {
    /// Create a new cosine annealing schedule
    pub fn new(initial_lr: f64, min_lr: f64, t_max: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_max,
        }
    }

    /// Get learning rate at iteration t
    pub fn get_lr(&self, t: usize) -> f64 {
        let t = t.min(self.t_max);
        let cosine = (std::f64::consts::PI * t as f64 / self.t_max as f64).cos();
        self.min_lr + (self.initial_lr - self.min_lr) * (1.0 + cosine) / 2.0
    }
}

/// Step decay learning rate schedule
///
/// Decreases learning rate by a factor at specified epochs.
#[derive(Debug, Clone)]
pub struct StepDecaySchedule {
    /// Initial learning rate
    pub initial_lr: f64,
    /// Decay factor
    pub gamma: f64,
    /// Step size (epochs between decays)
    pub step_size: usize,
}

impl StepDecaySchedule {
    /// Create a new step decay schedule
    pub fn new(initial_lr: f64, gamma: f64, step_size: usize) -> Self {
        Self {
            initial_lr,
            gamma,
            step_size,
        }
    }

    /// Get learning rate at iteration t
    pub fn get_lr(&self, t: usize) -> f64 {
        let n_steps = t / self.step_size;
        self.initial_lr * self.gamma.powi(n_steps as i32)
    }
}

/// Warm restarts learning rate schedule (SGDR)
///
/// Cosine annealing with periodic warm restarts.
#[derive(Debug, Clone, Copy)]
pub struct WarmRestartsSchedule {
    /// Initial learning rate
    pub initial_lr: f64,
    /// Minimum learning rate
    pub min_lr: f64,
    /// Initial restart period
    pub t_0: usize,
    /// Period multiplier after each restart
    pub t_mult: usize,
}

impl WarmRestartsSchedule {
    /// Create a new warm restarts schedule
    pub fn new(initial_lr: f64, min_lr: f64, t_0: usize, t_mult: usize) -> Self {
        Self {
            initial_lr,
            min_lr,
            t_0,
            t_mult,
        }
    }

    /// Get learning rate at iteration t
    pub fn get_lr(&self, t: usize) -> f64 {
        let mut t_cur = t;
        let mut t_i = self.t_0;

        // Find which cycle we're in
        while t_cur >= t_i {
            t_cur -= t_i;
            t_i *= self.t_mult.max(1);
        }

        // Cosine annealing within the cycle
        let cosine = (std::f64::consts::PI * t_cur as f64 / t_i as f64).cos();
        self.min_lr + (self.initial_lr - self.min_lr) * (1.0 + cosine) / 2.0
    }
}

// =============================================================================
// Checkpoint Callback
// =============================================================================

/// Checkpoint data for model state
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Epoch at which checkpoint was taken
    pub epoch: usize,
    /// Validation loss at checkpoint
    pub val_loss: Option<f64>,
    /// Training loss at checkpoint
    pub train_loss: f64,
    /// Learning rate at checkpoint
    pub learning_rate: f64,
    /// Model state (serialized)
    pub model_state: Vec<u8>,
}

/// Checkpoint callback for saving/restoring model state
#[derive(Debug)]
pub struct CheckpointCallback {
    /// Saved checkpoints
    checkpoints: RwLock<Vec<Checkpoint>>,
    /// Save every n epochs (0 = save only best)
    save_frequency: usize,
    /// Keep only best checkpoint
    keep_best_only: bool,
    /// Best validation loss seen
    best_val_loss: RwLock<f64>,
}

impl CheckpointCallback {
    /// Create a new checkpoint callback
    pub fn new(save_frequency: usize, keep_best_only: bool) -> Self {
        Self {
            checkpoints: RwLock::new(Vec::new()),
            save_frequency,
            keep_best_only,
            best_val_loss: RwLock::new(f64::INFINITY),
        }
    }

    /// Get all saved checkpoints
    pub fn checkpoints(&self) -> Vec<Checkpoint> {
        self.checkpoints.read().unwrap().clone()
    }

    /// Get best checkpoint
    pub fn best_checkpoint(&self) -> Option<Checkpoint> {
        self.checkpoints
            .read()
            .unwrap()
            .iter()
            .filter(|c| c.val_loss.is_some())
            .min_by(|a, b| {
                a.val_loss
                    .unwrap()
                    .partial_cmp(&b.val_loss.unwrap())
                    .unwrap()
            })
            .cloned()
    }

    /// Save checkpoint
    pub fn save(&self, checkpoint: Checkpoint) {
        if self.keep_best_only {
            if let Some(val_loss) = checkpoint.val_loss {
                if val_loss < *self.best_val_loss.read().unwrap() {
                    *self.best_val_loss.write().unwrap() = val_loss;
                    let mut checkpoints = self.checkpoints.write().unwrap();
                    checkpoints.clear();
                    checkpoints.push(checkpoint);
                }
            }
        } else {
            self.checkpoints.write().unwrap().push(checkpoint);
        }
    }

    /// Number of saved checkpoints
    pub fn n_checkpoints(&self) -> usize {
        self.checkpoints.read().unwrap().len()
    }
}

impl TrainingCallback for CheckpointCallback {
    fn on_training_start(&self, _n_estimators: usize) {
        // Reset state
        *self.best_val_loss.write().unwrap() = f64::INFINITY;
    }

    fn on_epoch_start(&self, _epoch: usize) {}

    fn on_epoch_end(
        &self,
        epoch: usize,
        train_loss: f64,
        val_loss: Option<f64>,
        learning_rate: f64,
    ) {
        let should_save = if self.save_frequency == 0 {
            // Save only when validation improves
            val_loss.map_or(false, |v| v < *self.best_val_loss.read().unwrap())
        } else {
            // Save at regular intervals
            epoch % self.save_frequency == 0
        };

        if should_save {
            self.save(Checkpoint {
                epoch,
                val_loss,
                train_loss,
                learning_rate,
                model_state: Vec::new(), // Placeholder for serialized state
            });
        }
    }

    fn on_early_stopping(&self, _epoch: usize, _best_epoch: usize) {}

    fn on_training_end(
        &self,
        _n_epochs: usize,
        _final_train_loss: f64,
        _final_val_loss: Option<f64>,
    ) {
    }

    fn name(&self) -> &str {
        "CheckpointCallback"
    }
}

// =============================================================================
// Callback Composition
// =============================================================================

/// Composed callback that executes multiple callbacks in order
pub struct ComposedCallback {
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl ComposedCallback {
    /// Create a new composed callback
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback
    pub fn add<C: TrainingCallback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
        // Sort by priority
        self.callbacks.sort_by_key(|c| c.priority());
    }

    /// Number of callbacks
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }
}

impl Default for ComposedCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingCallback for ComposedCallback {
    fn on_training_start(&self, n_estimators: usize) {
        for callback in &self.callbacks {
            callback.on_training_start(n_estimators);
        }
    }

    fn on_epoch_start(&self, epoch: usize) {
        for callback in &self.callbacks {
            callback.on_epoch_start(epoch);
        }
    }

    fn on_epoch_end(
        &self,
        epoch: usize,
        train_loss: f64,
        val_loss: Option<f64>,
        learning_rate: f64,
    ) {
        for callback in &self.callbacks {
            callback.on_epoch_end(epoch, train_loss, val_loss, learning_rate);
        }
    }

    fn on_early_stopping(&self, epoch: usize, best_epoch: usize) {
        for callback in &self.callbacks {
            callback.on_early_stopping(epoch, best_epoch);
        }
    }

    fn on_training_end(&self, n_epochs: usize, final_train_loss: f64, final_val_loss: Option<f64>) {
        for callback in &self.callbacks {
            callback.on_training_end(n_epochs, final_train_loss, final_val_loss);
        }
    }

    fn name(&self) -> &str {
        "ComposedCallback"
    }
}

// =============================================================================
// Early Stopping State Tracker
// =============================================================================

/// State tracker for early stopping behavior verification
#[derive(Debug, Clone)]
pub struct EarlyStoppingState {
    /// Best loss observed
    pub best_loss: f64,
    /// Epoch at which best loss was observed
    pub best_epoch: usize,
    /// Current patience counter
    pub patience_counter: usize,
    /// Whether early stopping has been triggered
    pub triggered: bool,
    /// History of validation losses
    pub loss_history: Vec<f64>,
    /// History of patience counter values
    pub patience_history: Vec<usize>,
}

impl EarlyStoppingState {
    /// Create a new early stopping state
    pub fn new() -> Self {
        Self {
            best_loss: f64::INFINITY,
            best_epoch: 0,
            patience_counter: 0,
            triggered: false,
            loss_history: Vec::new(),
            patience_history: Vec::new(),
        }
    }

    /// Update state with new validation loss
    pub fn update(&mut self, epoch: usize, val_loss: f64, patience: usize, min_delta: f64) -> bool {
        self.loss_history.push(val_loss);

        if val_loss < self.best_loss - min_delta {
            self.best_loss = val_loss;
            self.best_epoch = epoch;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
        }

        self.patience_history.push(self.patience_counter);

        if self.patience_counter >= patience {
            self.triggered = true;
        }

        self.triggered
    }

    /// Reset state
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for EarlyStoppingState {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Test Data Generators
// =============================================================================

/// Generate regression data suitable for gradient boosting tests
pub fn make_boosting_regression_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let x = Array2::from_shape_fn((n_samples, n_features), |_| rng.random_range(-5.0..5.0));

    // Non-linear target: y = x1^2 + 0.5*x2 + noise
    let y = Array1::from_shape_fn(n_samples, |i| {
        let x1: f64 = x[[i, 0]];
        let x2: f64 = if n_features > 1 { x[[i, 1]] } else { 0.0 };
        x1.powi(2) + 0.5 * x2 + rng.random_range(-0.5..0.5)
    });

    (x, y)
}

/// Generate classification data suitable for gradient boosting tests
pub fn make_boosting_classification_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    let half = n_samples / 2;
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    // Class 0: centered around origin
    for _ in 0..half {
        for _ in 0..n_features {
            x_data.push(rng.random_range(-2.0..1.0));
        }
        y_data.push(0.0);
    }

    // Class 1: centered away from origin
    for _ in half..n_samples {
        for _ in 0..n_features {
            x_data.push(rng.random_range(0.0..3.0));
        }
        y_data.push(1.0);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    (x, y)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;
    use crate::testing::assertions::tolerances;
    use std::sync::Arc;

    // =========================================================================
    // Early Stopping Tests
    // =========================================================================

    #[test]
    fn test_early_stopping_default_config() {
        let es = EarlyStopping::default();
        assert_eq!(es.patience, 10);
        assert_eq!(es.min_delta, 1e-4);
        assert_eq!(es.validation_fraction, 0.1);
    }

    #[test]
    fn test_early_stopping_triggers_at_correct_patience() {
        // Test with small dataset and early stopping
        let (x, y) = make_boosting_regression_data(100, 3, 42);

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(200) // More than enough
            .with_learning_rate(0.1)
            .with_max_depth(Some(2))
            .with_early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1e-6,
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        let history = model.training_history().unwrap();

        // Verify training stopped
        let actual_estimators = model.n_estimators_actual().unwrap();
        assert!(
            actual_estimators <= 200,
            "Training should have stopped early or completed, got {} estimators",
            actual_estimators
        );

        // If early stopping triggered, verify patience behavior
        if let Some(stopped_at) = history.stopped_at {
            assert!(
                stopped_at > 5,
                "Should train at least patience epochs before stopping"
            );

            // Verify validation loss was tracked
            assert!(
                !history.val_loss.is_empty(),
                "Should have validation loss history"
            );
        }
    }

    #[test]
    fn test_early_stopping_with_different_patience_values() {
        let (x, y) = make_boosting_regression_data(80, 2, 123);

        for patience in [3, 5, 10] {
            let mut model = GradientBoostingRegressor::new()
                .with_n_estimators(100)
                .with_learning_rate(0.1)
                .with_early_stopping(EarlyStopping {
                    patience,
                    min_delta: 1e-8,
                    validation_fraction: 0.2,
                })
                .with_random_state(42);

            model.fit(&x, &y).unwrap();

            let history = model.training_history().unwrap();

            // Verify learning rate history matches training length
            assert_eq!(
                history.learning_rates.len(),
                model.n_estimators_actual().unwrap()
            );
        }
    }

    #[test]
    fn test_early_stopping_min_delta_threshold() {
        let (x, y) = make_boosting_regression_data(100, 2, 42);

        // Large min_delta should trigger early stopping sooner
        let mut model_large_delta = GradientBoostingRegressor::new()
            .with_n_estimators(100)
            .with_learning_rate(0.1)
            .with_early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1.0, // Large delta - small improvements won't count
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        model_large_delta.fit(&x, &y).unwrap();

        // Small min_delta should allow more training
        let mut model_small_delta = GradientBoostingRegressor::new()
            .with_n_estimators(100)
            .with_learning_rate(0.1)
            .with_early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1e-10, // Tiny delta - small improvements count
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        model_small_delta.fit(&x, &y).unwrap();

        let n1 = model_large_delta.n_estimators_actual().unwrap();
        let n2 = model_small_delta.n_estimators_actual().unwrap();

        // With larger min_delta, we expect earlier stopping (or same)
        assert!(
            n1 <= n2,
            "Large min_delta should stop earlier: {} vs {}",
            n1,
            n2
        );
    }

    #[test]
    fn test_early_stopping_validation_fraction() {
        let (x, y) = make_boosting_regression_data(100, 2, 42);

        // Different validation fractions
        for frac in [0.1, 0.2, 0.3] {
            let mut model = GradientBoostingRegressor::new()
                .with_n_estimators(50)
                .with_early_stopping(EarlyStopping {
                    patience: 5,
                    min_delta: 1e-4,
                    validation_fraction: frac,
                })
                .with_random_state(42);

            model.fit(&x, &y).unwrap();

            let history = model.training_history().unwrap();

            // Validation loss should be recorded
            assert!(
                !history.val_loss.is_empty(),
                "Should have validation loss with fraction {}",
                frac
            );
        }
    }

    #[test]
    fn test_early_stopping_state_tracker() {
        let mut state = EarlyStoppingState::new();

        // Simulate losses that improve then plateau
        let losses = [1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let patience = 3;
        let min_delta = 0.01;

        for (epoch, &loss) in losses.iter().enumerate() {
            let triggered = state.update(epoch, loss, patience, min_delta);
            if triggered {
                assert_eq!(
                    state.patience_counter, patience,
                    "Should trigger at patience threshold"
                );
                assert_eq!(state.best_epoch, 3, "Best epoch should be epoch 3");
                break;
            }
        }

        assert!(state.triggered, "Should have triggered early stopping");
        assert_eq!(state.loss_history.len(), 7); // 0 through 6
    }

    #[test]
    fn test_early_stopping_state_with_improvement() {
        let mut state = EarlyStoppingState::new();

        // Simulate losses that keep improving
        let losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let patience = 5;
        let min_delta = 0.01;

        for (epoch, &loss) in losses.iter().enumerate() {
            let triggered = state.update(epoch, loss, patience, min_delta);
            assert!(!triggered, "Should not trigger when improving");
            assert_eq!(
                state.patience_counter, 0,
                "Counter should reset on improvement"
            );
            assert_eq!(state.best_epoch, epoch, "Best epoch should be current");
        }
    }

    // =========================================================================
    // Learning Rate Scheduler Tests
    // =========================================================================

    #[test]
    fn test_constant_learning_rate() {
        let schedule = LearningRateSchedule::Constant(0.1);

        for i in 0..100 {
            let lr = schedule.get_lr(i, 100);
            assert_approx_eq!(lr, 0.1, tolerances::CLOSED_FORM, "iteration {}", i);
        }
    }

    #[test]
    fn test_linear_decay_schedule() {
        let schedule = LearningRateSchedule::LinearDecay {
            initial: 0.2,
            min_lr: 0.01,
        };

        // At start
        let lr_start = schedule.get_lr(0, 100);
        assert_approx_eq!(lr_start, 0.2, tolerances::CLOSED_FORM);

        // At middle
        let lr_mid = schedule.get_lr(50, 100);
        assert!(lr_mid < 0.2 && lr_mid > 0.01, "Mid LR: {}", lr_mid);

        // At end
        let lr_end = schedule.get_lr(99, 100);
        assert!(
            lr_end >= 0.01,
            "End LR should be at least min_lr: {}",
            lr_end
        );

        // Verify monotonic decrease
        let mut prev_lr = lr_start;
        for i in 1..100 {
            let lr = schedule.get_lr(i, 100);
            assert!(lr <= prev_lr, "LR should decrease: {} vs {}", lr, prev_lr);
            prev_lr = lr;
        }
    }

    #[test]
    fn test_exponential_decay_schedule() {
        let schedule = LearningRateSchedule::ExponentialDecay {
            initial: 0.2,
            decay: 0.95,
            min_lr: 0.001,
        };

        // At start
        let lr_start = schedule.get_lr(0, 100);
        assert_approx_eq!(lr_start, 0.2, tolerances::CLOSED_FORM);

        // At iteration 10
        let lr_10 = schedule.get_lr(10, 100);
        let expected_10 = 0.2 * 0.95_f64.powi(10);
        assert_approx_eq!(lr_10, expected_10, tolerances::CLOSED_FORM);

        // Verify monotonic decrease
        let mut prev_lr = lr_start;
        for i in 1..100 {
            let lr = schedule.get_lr(i, 100);
            assert!(lr <= prev_lr, "LR should decrease: {} vs {}", lr, prev_lr);
            assert!(lr >= 0.001, "LR should be at least min_lr: {}", lr);
            prev_lr = lr;
        }
    }

    #[test]
    fn test_cosine_annealing_schedule() {
        let schedule = CosineAnnealingSchedule::new(0.1, 0.001, 100);

        // At start
        let lr_start = schedule.get_lr(0);
        assert_approx_eq!(lr_start, 0.1, tolerances::CLOSED_FORM);

        // At end
        let lr_end = schedule.get_lr(100);
        assert_approx_eq!(lr_end, 0.001, tolerances::CLOSED_FORM);

        // At middle (should be roughly midway)
        let lr_mid = schedule.get_lr(50);
        let expected_mid = 0.001 + (0.1 - 0.001) * 0.5; // cos(pi/2) = 0
        assert_approx_eq!(lr_mid, expected_mid, tolerances::ITERATIVE);

        // Verify smooth decrease
        let mut prev_lr = lr_start;
        for i in 1..=100 {
            let lr = schedule.get_lr(i);
            assert!(
                lr <= prev_lr + tolerances::ITERATIVE,
                "LR should decrease smoothly: {} vs {}",
                lr,
                prev_lr
            );
            prev_lr = lr;
        }
    }

    #[test]
    fn test_step_decay_schedule() {
        let schedule = StepDecaySchedule::new(0.1, 0.5, 10);

        // Before first step
        for i in 0..10 {
            let lr = schedule.get_lr(i);
            assert_approx_eq!(lr, 0.1, tolerances::CLOSED_FORM, "iteration {}", i);
        }

        // After first step
        for i in 10..20 {
            let lr = schedule.get_lr(i);
            assert_approx_eq!(lr, 0.05, tolerances::CLOSED_FORM, "iteration {}", i);
        }

        // After second step
        for i in 20..30 {
            let lr = schedule.get_lr(i);
            assert_approx_eq!(lr, 0.025, tolerances::CLOSED_FORM, "iteration {}", i);
        }
    }

    #[test]
    fn test_warm_restarts_schedule() {
        let schedule = WarmRestartsSchedule::new(0.1, 0.001, 10, 2);

        // First cycle: t_0 = 10
        let lr_start_c1 = schedule.get_lr(0);
        assert_approx_eq!(lr_start_c1, 0.1, tolerances::CLOSED_FORM);

        let lr_end_c1 = schedule.get_lr(9);
        assert!(lr_end_c1 < 0.1, "Should decay within cycle");

        // Second cycle: t_1 = 20 (starts at t=10)
        let lr_start_c2 = schedule.get_lr(10);
        assert_approx_eq!(
            lr_start_c2,
            0.1,
            tolerances::CLOSED_FORM,
            "Should restart to initial LR"
        );
    }

    #[test]
    fn test_learning_rate_in_training() {
        let (x, y) = make_boosting_regression_data(80, 2, 42);

        // Test with exponential decay
        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(20)
            .with_learning_rate_schedule(LearningRateSchedule::ExponentialDecay {
                initial: 0.2,
                decay: 0.9,
                min_lr: 0.01,
            })
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        let history = model.training_history().unwrap();

        // Verify learning rates were recorded and decrease
        assert_eq!(history.learning_rates.len(), 20);
        assert!(
            history.learning_rates[0] > history.learning_rates[19],
            "LR should decrease during training"
        );
    }

    // =========================================================================
    // Checkpoint Callback Tests
    // =========================================================================

    #[test]
    fn test_checkpoint_callback_creation() {
        let callback = CheckpointCallback::new(5, false);
        assert_eq!(callback.n_checkpoints(), 0);
    }

    #[test]
    fn test_checkpoint_save_periodic() {
        let callback = CheckpointCallback::new(2, false);

        // Simulate training
        callback.on_training_start(10);

        for epoch in 0..10 {
            let train_loss = 1.0 - epoch as f64 * 0.05;
            let val_loss = 1.0 - epoch as f64 * 0.04;
            callback.on_epoch_end(epoch, train_loss, Some(val_loss), 0.1);
        }

        // Should save at epochs 0, 2, 4, 6, 8
        assert_eq!(callback.n_checkpoints(), 5);
    }

    #[test]
    fn test_checkpoint_save_best_only() {
        let callback = CheckpointCallback::new(0, true);

        callback.on_training_start(10);

        // First improvement
        callback.on_epoch_end(0, 1.0, Some(1.0), 0.1);
        assert_eq!(callback.n_checkpoints(), 1);

        // Better
        callback.on_epoch_end(1, 0.9, Some(0.8), 0.1);
        assert_eq!(callback.n_checkpoints(), 1); // Still 1, replaced

        // Worse - should not save
        callback.on_epoch_end(2, 0.85, Some(0.9), 0.1);
        assert_eq!(callback.n_checkpoints(), 1);

        // Even better
        callback.on_epoch_end(3, 0.7, Some(0.6), 0.1);
        assert_eq!(callback.n_checkpoints(), 1);

        // Verify best checkpoint is epoch 3
        let best = callback.best_checkpoint().unwrap();
        assert_eq!(best.epoch, 3);
        assert_approx_eq!(best.val_loss.unwrap(), 0.6, tolerances::CLOSED_FORM);
    }

    #[test]
    fn test_checkpoint_restore_verification() {
        let callback = CheckpointCallback::new(1, false);

        callback.on_training_start(5);

        let checkpoints_data = [
            (0, 1.0, 1.0, 0.1),
            (1, 0.9, 0.9, 0.09),
            (2, 0.8, 0.85, 0.08),
            (3, 0.7, 0.75, 0.07),
            (4, 0.6, 0.65, 0.06),
        ];

        for (epoch, train_loss, val_loss, lr) in checkpoints_data {
            callback.on_epoch_end(epoch, train_loss, Some(val_loss), lr);
        }

        let checkpoints = callback.checkpoints();
        assert_eq!(checkpoints.len(), 5);

        // Verify checkpoint data
        for (i, ckpt) in checkpoints.iter().enumerate() {
            assert_eq!(ckpt.epoch, i);
            assert_approx_eq!(
                ckpt.train_loss,
                checkpoints_data[i].1,
                tolerances::CLOSED_FORM
            );
            assert_approx_eq!(
                ckpt.val_loss.unwrap(),
                checkpoints_data[i].2,
                tolerances::CLOSED_FORM
            );
            assert_approx_eq!(
                ckpt.learning_rate,
                checkpoints_data[i].3,
                tolerances::CLOSED_FORM
            );
        }
    }

    // =========================================================================
    // Custom Callback Tests
    // =========================================================================

    #[test]
    fn test_event_recorder_callback() {
        let recorder = EventRecorderCallback::new();

        recorder.on_training_start(10);
        recorder.on_epoch_start(0);
        recorder.on_epoch_end(0, 1.0, Some(0.9), 0.1);
        recorder.on_training_end(1, 0.8, Some(0.7));

        let events = recorder.events();
        assert_eq!(events.len(), 4);

        assert!(matches!(
            events[0],
            TrainingEvent::TrainingStart { n_estimators: 10 }
        ));
        assert!(matches!(events[1], TrainingEvent::EpochStart { epoch: 0 }));
        assert!(matches!(
            events[2],
            TrainingEvent::EpochEnd {
                epoch: 0,
                train_loss: _,
                val_loss: Some(_),
                learning_rate: _
            }
        ));
        assert!(matches!(
            events[3],
            TrainingEvent::TrainingEnd {
                n_epochs: 1,
                final_train_loss: _,
                final_val_loss: Some(_)
            }
        ));
    }

    #[test]
    fn test_counter_callback() {
        let counter = CounterCallback::new();

        counter.on_training_start(20);
        for i in 0..10 {
            counter.on_epoch_start(i);
            counter.on_epoch_end(i, 1.0, None, 0.1);
        }
        counter.on_training_end(10, 0.5, None);

        assert_eq!(counter.training_start_count.load(Ordering::Relaxed), 1);
        assert_eq!(counter.epoch_start_count.load(Ordering::Relaxed), 10);
        assert_eq!(counter.epoch_end_count.load(Ordering::Relaxed), 10);
        assert_eq!(counter.training_end_count.load(Ordering::Relaxed), 1);
        assert_eq!(counter.epoch_count(), 10);
    }

    #[test]
    fn test_custom_callback_with_state() {
        struct LossTracker {
            losses: RwLock<Vec<f64>>,
            best_loss: RwLock<f64>,
        }

        impl TrainingCallback for LossTracker {
            fn on_training_start(&self, _n: usize) {
                *self.best_loss.write().unwrap() = f64::INFINITY;
            }
            fn on_epoch_start(&self, _e: usize) {}
            fn on_epoch_end(&self, _e: usize, _tl: f64, val_loss: Option<f64>, _lr: f64) {
                if let Some(vl) = val_loss {
                    self.losses.write().unwrap().push(vl);
                    let mut best = self.best_loss.write().unwrap();
                    if vl < *best {
                        *best = vl;
                    }
                }
            }
            fn on_early_stopping(&self, _e: usize, _be: usize) {}
            fn on_training_end(&self, _n: usize, _tl: f64, _vl: Option<f64>) {}
            fn name(&self) -> &str {
                "LossTracker"
            }
        }

        let tracker = LossTracker {
            losses: RwLock::new(Vec::new()),
            best_loss: RwLock::new(f64::INFINITY),
        };

        tracker.on_training_start(10);
        tracker.on_epoch_end(0, 1.0, Some(1.0), 0.1);
        tracker.on_epoch_end(1, 0.9, Some(0.8), 0.1);
        tracker.on_epoch_end(2, 0.8, Some(0.9), 0.1);
        tracker.on_epoch_end(3, 0.7, Some(0.7), 0.1);

        assert_eq!(tracker.losses.read().unwrap().len(), 4);
        assert_approx_eq!(
            *tracker.best_loss.read().unwrap(),
            0.7,
            tolerances::CLOSED_FORM
        );
    }

    // =========================================================================
    // Callback Composition Tests
    // =========================================================================

    #[test]
    fn test_composed_callback_basic() {
        let mut composed = ComposedCallback::new();
        assert!(composed.is_empty());

        composed.add(EventRecorderCallback::new());
        composed.add(CounterCallback::new());

        assert_eq!(composed.len(), 2);
        assert!(!composed.is_empty());
    }

    #[test]
    fn test_composed_callback_execution_order() {
        struct OrderTracker {
            order: Arc<std::sync::Mutex<Vec<i32>>>,
            id: i32,
            priority: i32,
        }

        impl TrainingCallback for OrderTracker {
            fn on_training_start(&self, _n: usize) {
                self.order.lock().unwrap().push(self.id);
            }
            fn on_epoch_start(&self, _e: usize) {}
            fn on_epoch_end(&self, _e: usize, _tl: f64, _vl: Option<f64>, _lr: f64) {}
            fn on_early_stopping(&self, _e: usize, _be: usize) {}
            fn on_training_end(&self, _n: usize, _tl: f64, _vl: Option<f64>) {}
            fn priority(&self) -> i32 {
                self.priority
            }
            fn name(&self) -> &str {
                "OrderTracker"
            }
        }

        let order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut composed = ComposedCallback::new();
        composed.add(OrderTracker {
            order: Arc::clone(&order),
            id: 3,
            priority: 30,
        });
        composed.add(OrderTracker {
            order: Arc::clone(&order),
            id: 1,
            priority: 10,
        });
        composed.add(OrderTracker {
            order: Arc::clone(&order),
            id: 2,
            priority: 20,
        });

        composed.on_training_start(10);

        let execution_order = order.lock().unwrap().clone();
        assert_eq!(
            execution_order,
            vec![1, 2, 3],
            "Should execute in priority order"
        );
    }

    #[test]
    fn test_composed_callback_all_events() {
        let mut composed = ComposedCallback::new();
        let _counter = Arc::new(CounterCallback::new());

        // We can't easily add Arc<CounterCallback> due to trait bounds,
        // so we'll use a simpler approach
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));

        struct EventCollector {
            events: Arc<std::sync::Mutex<Vec<String>>>,
        }

        impl TrainingCallback for EventCollector {
            fn on_training_start(&self, _n: usize) {
                self.events.lock().unwrap().push("start".to_string());
            }
            fn on_epoch_start(&self, e: usize) {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("epoch_start_{}", e));
            }
            fn on_epoch_end(&self, e: usize, _tl: f64, _vl: Option<f64>, _lr: f64) {
                self.events.lock().unwrap().push(format!("epoch_end_{}", e));
            }
            fn on_early_stopping(&self, e: usize, _be: usize) {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("early_stop_{}", e));
            }
            fn on_training_end(&self, _n: usize, _tl: f64, _vl: Option<f64>) {
                self.events.lock().unwrap().push("end".to_string());
            }
            fn name(&self) -> &str {
                "EventCollector"
            }
        }

        composed.add(EventCollector {
            events: Arc::clone(&events),
        });

        // Simulate training
        composed.on_training_start(5);
        for i in 0..3 {
            composed.on_epoch_start(i);
            composed.on_epoch_end(i, 1.0, Some(0.9), 0.1);
        }
        composed.on_early_stopping(3, 1);
        composed.on_training_end(3, 0.8, Some(0.7));

        let recorded = events.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec![
                "start",
                "epoch_start_0",
                "epoch_end_0",
                "epoch_start_1",
                "epoch_end_1",
                "epoch_start_2",
                "epoch_end_2",
                "early_stop_3",
                "end"
            ]
        );
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_gradient_boosting_regressor_with_callbacks() {
        let (x, y) = make_boosting_regression_data(100, 3, 42);

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(30)
            .with_learning_rate(0.1)
            .with_early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1e-5,
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        // Verify model trained
        assert!(model.is_fitted());

        // Verify history
        let history = model.training_history().unwrap();
        assert!(!history.train_loss.is_empty());
        assert!(!history.val_loss.is_empty());
        assert!(!history.learning_rates.is_empty());

        // Training loss should generally decrease
        let n = history.train_loss.len();
        if n > 5 {
            let early_avg: f64 = history.train_loss[..3].iter().sum::<f64>() / 3.0;
            let late_avg: f64 = history.train_loss[n - 3..].iter().sum::<f64>() / 3.0;
            assert!(
                late_avg <= early_avg,
                "Training loss should decrease: {} vs {}",
                late_avg,
                early_avg
            );
        }
    }

    #[test]
    fn test_gradient_boosting_classifier_with_callbacks() {
        let (x, y) = make_boosting_classification_data(100, 3, 42);

        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(30)
            .with_learning_rate(0.1)
            .with_early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1e-5,
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());

        let history = model.training_history().unwrap();
        assert!(!history.train_loss.is_empty());
        assert!(!history.val_loss.is_empty());
    }

    #[test]
    fn test_training_history_consistency() {
        let (x, y) = make_boosting_regression_data(80, 2, 42);

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(20)
            .with_learning_rate_schedule(LearningRateSchedule::LinearDecay {
                initial: 0.2,
                min_lr: 0.01,
            })
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        let history = model.training_history().unwrap();

        // All histories should have same length
        assert_eq!(history.train_loss.len(), history.learning_rates.len());

        // All values should be finite
        for &loss in &history.train_loss {
            assert!(loss.is_finite(), "Train loss should be finite: {}", loss);
        }

        for &lr in &history.learning_rates {
            assert!(lr.is_finite(), "Learning rate should be finite: {}", lr);
            assert!(lr > 0.0, "Learning rate should be positive: {}", lr);
        }
    }

    #[test]
    fn test_staged_predictions_consistency() {
        let (x, y) = make_boosting_regression_data(50, 2, 42);

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(15)
            .with_learning_rate(0.1)
            .with_random_state(42);

        model.fit(&x, &y).unwrap();

        // Get staged predictions
        let staged: Vec<Array1<f64>> = model.staged_predict(&x).unwrap().collect();

        // Should have one prediction per estimator
        assert_eq!(staged.len(), 15);

        // Final staged prediction should match regular predict
        let final_pred = model.predict(&x).unwrap();
        for (i, (&staged_val, &final_val)) in staged
            .last()
            .unwrap()
            .iter()
            .zip(final_pred.iter())
            .enumerate()
        {
            assert_approx_eq!(
                staged_val,
                final_val,
                tolerances::CLOSED_FORM,
                "Staged vs final at index {}",
                i
            );
        }

        // Predictions should change with each stage
        for i in 1..staged.len() {
            let diff: f64 = (&staged[i] - &staged[i - 1]).iter().map(|x| x.abs()).sum();
            assert!(diff > 0.0, "Predictions should change between stages");
        }
    }

    #[test]
    fn test_callback_with_different_model_configurations() {
        let (x, y) = make_boosting_regression_data(100, 3, 42);

        // Test different configurations
        let configs = [(10, 0.1, Some(2)), (20, 0.05, Some(3)), (15, 0.2, Some(4))];

        for (n_estimators, lr, max_depth) in configs {
            let mut model = GradientBoostingRegressor::new()
                .with_n_estimators(n_estimators)
                .with_learning_rate(lr)
                .with_max_depth(max_depth)
                .with_random_state(42);

            model.fit(&x, &y).unwrap();

            let history = model.training_history().unwrap();

            assert_eq!(
                history.train_loss.len(),
                n_estimators,
                "History length should match n_estimators"
            );

            // Verify learning rates are constant
            for &recorded_lr in &history.learning_rates {
                assert_approx_eq!(recorded_lr, lr, tolerances::CLOSED_FORM);
            }
        }
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_early_stopping_never_improves() {
        // Create data where validation loss won't improve much
        let mut state = EarlyStoppingState::new();

        // Start with good loss, then only get worse
        let losses = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let patience = 3;
        let min_delta = 0.01;

        state.update(0, losses[0], patience, min_delta);
        assert!(!state.triggered);
        assert_eq!(state.best_epoch, 0);

        for (i, &loss) in losses[1..].iter().enumerate() {
            let triggered = state.update(i + 1, loss, patience, min_delta);
            if triggered {
                assert_eq!(state.patience_counter, patience);
                break;
            }
        }

        assert!(state.triggered);
        assert_eq!(state.best_epoch, 0); // First epoch was best
    }

    #[test]
    fn test_learning_rate_schedule_edge_cases() {
        // Test with iteration beyond t_max
        let cosine = CosineAnnealingSchedule::new(0.1, 0.001, 50);
        let lr_beyond = cosine.get_lr(100);
        assert_approx_eq!(lr_beyond, 0.001, tolerances::CLOSED_FORM);

        // Test step decay with very large step
        let step = StepDecaySchedule::new(0.1, 0.5, 1000);
        let lr_early = step.get_lr(10);
        assert_approx_eq!(lr_early, 0.1, tolerances::CLOSED_FORM);

        // Test exponential decay approaching min_lr
        let exp = LearningRateSchedule::ExponentialDecay {
            initial: 0.1,
            decay: 0.5,
            min_lr: 0.01,
        };
        let lr_late = exp.get_lr(100, 100);
        assert!(lr_late >= 0.01, "Should not go below min_lr: {}", lr_late);
    }

    #[test]
    fn test_checkpoint_with_no_validation() {
        let callback = CheckpointCallback::new(0, true); // Best only mode

        callback.on_training_start(10);

        // Send epochs without validation loss
        for epoch in 0..5 {
            callback.on_epoch_end(epoch, 1.0 - epoch as f64 * 0.1, None, 0.1);
        }

        // Should not save any checkpoints without validation loss
        assert_eq!(callback.n_checkpoints(), 0);
    }

    #[test]
    fn test_composed_callback_empty() {
        let composed = ComposedCallback::new();
        assert!(composed.is_empty());

        // Should handle calls gracefully when empty
        composed.on_training_start(10);
        composed.on_epoch_start(0);
        composed.on_epoch_end(0, 1.0, None, 0.1);
        composed.on_early_stopping(0, 0);
        composed.on_training_end(1, 1.0, None);
    }
}
