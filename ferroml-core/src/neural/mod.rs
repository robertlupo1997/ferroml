//! Neural Network Algorithms with Statistical Extensions
//!
//! This module provides neural network implementations (MLPClassifier, MLPRegressor)
//! with FerroML-style statistical extensions that exceed typical implementations.
//!
//! ## Design Philosophy
//!
//! Every neural network in FerroML provides:
//! - **Training diagnostics**: Loss curves with convergence tests
//! - **Weight analysis**: Distribution of weights, dead neuron detection
//! - **Gradient flow**: Vanishing/exploding gradient detection
//! - **Uncertainty quantification**: MC Dropout for prediction intervals
//! - **Feature attribution**: Integrated gradients, saliency (future)
//!
//! ## Models
//!
//! - [`MLPClassifier`] - Multi-layer perceptron for classification
//! - [`MLPRegressor`] - Multi-layer perceptron for regression
//!
//! ## Activations
//!
//! - [`Activation::ReLU`] - Rectified Linear Unit
//! - [`Activation::Sigmoid`] - Logistic sigmoid
//! - [`Activation::Tanh`] - Hyperbolic tangent
//! - [`Activation::Softmax`] - Softmax (output layer for classification)
//! - [`Activation::Linear`] - Identity activation (output layer for regression)
//!
//! ## Optimizers
//!
//! - [`Solver::SGD`] - Stochastic Gradient Descent with momentum
//! - [`Solver::Adam`] - Adaptive Moment Estimation
//!
//! ## Example
//!
//! ```
//! # use ferroml_core::neural::{MLPClassifier, Activation, Solver, NeuralModel, NeuralDiagnostics};
//! # use ndarray::{Array1, Array2};
//! let X = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]); // XOR
//!
//! let mut mlp = MLPClassifier::new()
//!     .hidden_layer_sizes(&[4, 4])
//!     .activation(Activation::ReLU)
//!     .solver(Solver::Adam)
//!     .max_iter(1000);
//!
//! mlp.fit(&X, &y).unwrap();
//!
//! let predictions = mlp.predict(&X).unwrap();
//! let diagnostics = mlp.training_diagnostics();
//! ```

use crate::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// Submodules
pub mod activations;
pub mod analysis;
pub mod classifier;
pub mod diagnostics;
pub mod layers;
pub mod mlp;
pub mod optimizers;
pub mod regressor;
pub mod uncertainty;

// Re-exports
pub use activations::Activation;
pub use analysis::{
    dead_neuron_detection, weight_distribution_tests, weight_statistics, WeightStatistics,
};
pub use classifier::MLPClassifier;
pub use diagnostics::{ConvergenceStatus, GradientStatistics, TrainingDiagnostics};
pub use layers::Layer;
pub use mlp::MLP;
pub use optimizers::{
    adam_step, sgd_step, AdamState, LearningRateSchedule, OptimizerConfig, SGDState, Solver,
};
pub use regressor::MLPRegressor;
pub use uncertainty::{predict_with_uncertainty, PredictionUncertainty};

/// Core trait for neural network models
///
/// This trait provides the fundamental interface for neural networks.
/// All FerroML neural network models implement this trait.
pub trait NeuralModel {
    /// Fit the neural network to training data
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    /// * `y` - Target values of shape (n_samples,)
    ///
    /// # Returns
    /// Result indicating success or error
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;

    /// Predict target values for new data
    ///
    /// # Arguments
    /// * `x` - Feature matrix of shape (n_samples, n_features)
    ///
    /// # Returns
    /// Array of predictions
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Check if the model has been fitted
    fn is_fitted(&self) -> bool;

    /// Get the number of layers (including input and output)
    fn n_layers(&self) -> usize;

    /// Get the sizes of each layer
    fn layer_sizes(&self) -> Vec<usize>;
}

/// Trait for neural networks with statistical diagnostics
pub trait NeuralDiagnostics: NeuralModel {
    /// Get training diagnostics from the last fit
    fn training_diagnostics(&self) -> Option<&TrainingDiagnostics>;

    /// Get weight statistics for all layers
    fn weight_statistics(&self) -> Result<Vec<WeightStatistics>>;

    /// Detect dead neurons (ReLU neurons that never activate)
    ///
    /// # Arguments
    /// * `x` - Sample data to test activation patterns
    ///
    /// # Returns
    /// Vec of (layer_idx, neuron_idx) pairs for dead neurons
    fn dead_neurons(&self, x: &Array2<f64>) -> Result<Vec<(usize, usize)>>;
}

/// Trait for neural networks with uncertainty quantification
pub trait NeuralUncertainty: NeuralModel {
    /// Predict with uncertainty using MC Dropout
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `n_samples` - Number of MC samples
    /// * `confidence` - Confidence level (e.g., 0.95)
    ///
    /// # Returns
    /// Prediction with mean, std, and confidence intervals
    fn predict_with_uncertainty(
        &self,
        x: &Array2<f64>,
        n_samples: usize,
        confidence: f64,
    ) -> Result<PredictionUncertainty>;
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStopping {
    /// Number of iterations with no improvement before stopping
    pub patience: usize,
    /// Minimum improvement to qualify as an improvement
    pub min_delta: f64,
    /// Fraction of training data to use for validation
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

/// Weight initialization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WeightInit {
    /// Xavier/Glorot uniform initialization (good for tanh/sigmoid)
    #[default]
    XavierUniform,
    /// Xavier/Glorot normal initialization
    XavierNormal,
    /// He uniform initialization (good for ReLU)
    HeUniform,
    /// He normal initialization
    HeNormal,
    /// Uniform random in [-scale, scale]
    Uniform,
    /// Normal with zero mean and given std
    Normal,
}

/// Regularization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regularization {
    /// L2 regularization strength (weight decay)
    pub alpha: f64,
    /// Dropout rate (0.0 = no dropout, 0.5 = 50% dropout)
    pub dropout_rate: f64,
}

impl Default for Regularization {
    fn default() -> Self {
        Self {
            alpha: 1e-4,
            dropout_rate: 0.0,
        }
    }
}

/// Training history for a neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Loss at each epoch
    pub loss_curve: Vec<f64>,
    /// Validation loss at each epoch (if early stopping enabled)
    pub val_loss_curve: Option<Vec<f64>>,
    /// Best iteration (if early stopping)
    pub best_iter: Option<usize>,
    /// Final number of iterations
    pub n_iter: usize,
    /// Whether training converged
    pub converged: bool,
}

impl TrainingHistory {
    /// Create a new empty training history
    pub fn new() -> Self {
        Self {
            loss_curve: Vec::new(),
            val_loss_curve: None,
            best_iter: None,
            n_iter: 0,
            converged: false,
        }
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_defaults() {
        let es = EarlyStopping::default();
        assert_eq!(es.patience, 10);
        assert!((es.min_delta - 1e-4).abs() < 1e-10);
        assert!((es.validation_fraction - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_regularization_defaults() {
        let reg = Regularization::default();
        assert!((reg.alpha - 1e-4).abs() < 1e-10);
        assert!((reg.dropout_rate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_training_history_new() {
        let history = TrainingHistory::new();
        assert!(history.loss_curve.is_empty());
        assert!(history.val_loss_curve.is_none());
        assert!(!history.converged);
    }
}
