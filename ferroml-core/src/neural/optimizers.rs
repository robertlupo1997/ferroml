//! Optimizers for Neural Networks
//!
//! This module provides optimization algorithms for training neural networks.
//!
//! ## Optimizers
//!
//! - [`Solver::SGD`] - Stochastic Gradient Descent with optional momentum
//! - [`Solver::Adam`] - Adaptive Moment Estimation
//!
//! ## Learning Rate Schedules
//!
//! - [`LearningRateSchedule::Constant`] - Fixed learning rate
//! - [`LearningRateSchedule::InverseScaling`] - lr = initial_lr / (1 + decay * t)
//! - [`LearningRateSchedule::Adaptive`] - Reduce on plateau

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Solver (optimizer) type for neural networks
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Solver {
    /// Stochastic Gradient Descent with momentum
    SGD,
    /// Adaptive Moment Estimation (recommended)
    #[default]
    Adam,
}

impl Solver {
    /// Get a human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Solver::SGD => "sgd",
            Solver::Adam => "adam",
        }
    }
}

/// Learning rate schedule
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum LearningRateSchedule {
    /// Fixed learning rate
    #[default]
    Constant,
    /// Decrease learning rate over time: lr = initial_lr / (1 + decay * t)
    InverseScaling { decay: f64 },
    /// Reduce learning rate when loss plateaus
    Adaptive { factor: f64, patience: usize },
}

/// State for SGD optimizer
#[derive(Debug, Clone)]
pub struct SGDState {
    /// Velocity for momentum (one per layer)
    pub velocities_w: Vec<Array2<f64>>,
    pub velocities_b: Vec<Array1<f64>>,
}

impl SGDState {
    /// Create new SGD state for given layer sizes
    pub fn new(layer_sizes: &[(usize, usize)]) -> Self {
        let velocities_w = layer_sizes
            .iter()
            .map(|&(n_in, n_out)| Array2::zeros((n_in, n_out)))
            .collect();
        let velocities_b = layer_sizes
            .iter()
            .map(|&(_, n_out)| Array1::zeros(n_out))
            .collect();

        Self {
            velocities_w,
            velocities_b,
        }
    }
}

/// State for Adam optimizer
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment estimate (mean of gradients)
    pub m_w: Vec<Array2<f64>>,
    pub m_b: Vec<Array1<f64>>,
    /// Second moment estimate (variance of gradients)
    pub v_w: Vec<Array2<f64>>,
    pub v_b: Vec<Array1<f64>>,
    /// Timestep (for bias correction)
    pub t: usize,
}

impl AdamState {
    /// Create new Adam state for given layer sizes
    pub fn new(layer_sizes: &[(usize, usize)]) -> Self {
        let m_w = layer_sizes
            .iter()
            .map(|&(n_in, n_out)| Array2::zeros((n_in, n_out)))
            .collect();
        let m_b = layer_sizes
            .iter()
            .map(|&(_, n_out)| Array1::zeros(n_out))
            .collect();
        let v_w = layer_sizes
            .iter()
            .map(|&(n_in, n_out)| Array2::zeros((n_in, n_out)))
            .collect();
        let v_b = layer_sizes
            .iter()
            .map(|&(_, n_out)| Array1::zeros(n_out))
            .collect();

        Self {
            m_w,
            m_b,
            v_w,
            v_b,
            t: 0,
        }
    }
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum for SGD (typically 0.9)
    pub momentum: f64,
    /// Beta1 for Adam (typically 0.9)
    pub beta1: f64,
    /// Beta2 for Adam (typically 0.999)
    pub beta2: f64,
    /// Epsilon for numerical stability (typically 1e-8)
    pub epsilon: f64,
    /// Learning rate schedule
    pub lr_schedule: LearningRateSchedule,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lr_schedule: LearningRateSchedule::Constant,
        }
    }
}

impl OptimizerConfig {
    /// Create config for SGD with momentum
    pub fn sgd(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            ..Default::default()
        }
    }

    /// Create config for Adam
    pub fn adam(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    /// Get effective learning rate at given iteration
    pub fn get_lr(
        &self,
        iteration: usize,
        best_loss: f64,
        current_loss: f64,
        plateau_count: &mut usize,
    ) -> f64 {
        match self.lr_schedule {
            LearningRateSchedule::Constant => self.learning_rate,
            LearningRateSchedule::InverseScaling { decay } => {
                self.learning_rate / (1.0 + decay * iteration as f64)
            }
            LearningRateSchedule::Adaptive { factor, patience } => {
                if current_loss >= best_loss {
                    *plateau_count += 1;
                } else {
                    *plateau_count = 0;
                }

                if *plateau_count >= patience {
                    *plateau_count = 0;
                    self.learning_rate * factor
                } else {
                    self.learning_rate
                }
            }
        }
    }
}

/// SGD optimizer step
///
/// Updates weights using: v = momentum * v - lr * grad; w = w + v
pub fn sgd_step(
    weights: &mut Array2<f64>,
    biases: &mut Array1<f64>,
    grad_w: &Array2<f64>,
    grad_b: &Array1<f64>,
    velocity_w: &mut Array2<f64>,
    velocity_b: &mut Array1<f64>,
    lr: f64,
    momentum: f64,
    l2_reg: f64,
) {
    // L2 regularization gradient
    let reg_grad_w = weights.mapv(|w| w * l2_reg);

    // Update velocity: v = momentum * v - lr * (grad + reg)
    *velocity_w = velocity_w.mapv(|v| v * momentum) - (grad_w + &reg_grad_w).mapv(|g| g * lr);
    *velocity_b = velocity_b.mapv(|v| v * momentum) - grad_b.mapv(|g| g * lr);

    // Update weights: w = w + v
    *weights = weights.clone() + velocity_w.clone();
    *biases = biases.clone() + velocity_b.clone();
}

/// Adam optimizer step
///
/// Updates weights using Adam algorithm with bias correction
pub fn adam_step(
    weights: &mut Array2<f64>,
    biases: &mut Array1<f64>,
    grad_w: &Array2<f64>,
    grad_b: &Array1<f64>,
    m_w: &mut Array2<f64>,
    m_b: &mut Array1<f64>,
    v_w: &mut Array2<f64>,
    v_b: &mut Array1<f64>,
    t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    l2_reg: f64,
) {
    // L2 regularization gradient
    let reg_grad_w = weights.mapv(|w| w * l2_reg);
    let total_grad_w = grad_w + &reg_grad_w;

    // Update biased first moment estimate
    *m_w = m_w.mapv(|m| m * beta1) + total_grad_w.mapv(|g| g * (1.0 - beta1));
    *m_b = m_b.mapv(|m| m * beta1) + grad_b.mapv(|g| g * (1.0 - beta1));

    // Update biased second raw moment estimate
    *v_w = v_w.mapv(|v| v * beta2) + total_grad_w.mapv(|g| g * g * (1.0 - beta2));
    *v_b = v_b.mapv(|v| v * beta2) + grad_b.mapv(|g| g * g * (1.0 - beta2));

    // Bias correction
    let t_f64 = t as f64;
    let bias_correction1 = 1.0 - beta1.powf(t_f64);
    let bias_correction2 = 1.0 - beta2.powf(t_f64);

    let m_w_corrected = m_w.mapv(|m| m / bias_correction1);
    let m_b_corrected = m_b.mapv(|m| m / bias_correction1);
    let v_w_corrected = v_w.mapv(|v| v / bias_correction2);
    let v_b_corrected = v_b.mapv(|v| v / bias_correction2);

    // Update weights
    let update_w = m_w_corrected / (v_w_corrected.mapv(|v| v.sqrt()) + epsilon);
    let update_b = m_b_corrected / (v_b_corrected.mapv(|v| v.sqrt()) + epsilon);

    *weights = weights.clone() - update_w.mapv(|u| u * lr);
    *biases = biases.clone() - update_b.mapv(|u| u * lr);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sgd_state_creation() {
        let layer_sizes = vec![(10, 5), (5, 3)];
        let state = SGDState::new(&layer_sizes);
        assert_eq!(state.velocities_w.len(), 2);
        assert_eq!(state.velocities_w[0].shape(), &[10, 5]);
        assert_eq!(state.velocities_b[0].len(), 5);
    }

    #[test]
    fn test_adam_state_creation() {
        let layer_sizes = vec![(10, 5), (5, 3)];
        let state = AdamState::new(&layer_sizes);
        assert_eq!(state.m_w.len(), 2);
        assert_eq!(state.v_w.len(), 2);
        assert_eq!(state.t, 0);
    }

    #[test]
    fn test_optimizer_config_defaults() {
        let config = OptimizerConfig::default();
        assert_abs_diff_eq!(config.learning_rate, 0.001, epsilon = 1e-10);
        assert_abs_diff_eq!(config.momentum, 0.9, epsilon = 1e-10);
        assert_abs_diff_eq!(config.beta1, 0.9, epsilon = 1e-10);
        assert_abs_diff_eq!(config.beta2, 0.999, epsilon = 1e-10);
    }

    #[test]
    fn test_sgd_step() {
        let mut weights = Array2::ones((2, 2));
        let mut biases = Array1::ones(2);
        let grad_w = Array2::from_elem((2, 2), 0.1);
        let grad_b = Array1::from_elem(2, 0.1);
        let mut velocity_w = Array2::zeros((2, 2));
        let mut velocity_b = Array1::zeros(2);

        sgd_step(
            &mut weights,
            &mut biases,
            &grad_w,
            &grad_b,
            &mut velocity_w,
            &mut velocity_b,
            0.1, // lr
            0.9, // momentum
            0.0, // l2_reg
        );

        // Weights should decrease (gradient descent)
        assert!(weights[[0, 0]] < 1.0);
        assert!(biases[0] < 1.0);
    }

    #[test]
    fn test_adam_step() {
        let mut weights = Array2::ones((2, 2));
        let mut biases = Array1::ones(2);
        let grad_w = Array2::from_elem((2, 2), 0.1);
        let grad_b = Array1::from_elem(2, 0.1);
        let mut m_w = Array2::zeros((2, 2));
        let mut m_b = Array1::zeros(2);
        let mut v_w = Array2::zeros((2, 2));
        let mut v_b = Array1::zeros(2);

        adam_step(
            &mut weights,
            &mut biases,
            &grad_w,
            &grad_b,
            &mut m_w,
            &mut m_b,
            &mut v_w,
            &mut v_b,
            1,     // t
            0.001, // lr
            0.9,   // beta1
            0.999, // beta2
            1e-8,  // epsilon
            0.0,   // l2_reg
        );

        // Weights should decrease
        assert!(weights[[0, 0]] < 1.0);
        // First moment should be non-zero
        assert!(m_w[[0, 0]].abs() > 0.0);
    }

    #[test]
    fn test_l2_regularization() {
        let mut weights = Array2::from_elem((2, 2), 10.0);
        let mut biases = Array1::ones(2);
        let grad_w = Array2::zeros((2, 2)); // No gradient, only regularization
        let grad_b = Array1::zeros(2);
        let mut velocity_w = Array2::zeros((2, 2));
        let mut velocity_b = Array1::zeros(2);

        let initial_weight = weights[[0, 0]];

        sgd_step(
            &mut weights,
            &mut biases,
            &grad_w,
            &grad_b,
            &mut velocity_w,
            &mut velocity_b,
            0.1, // lr
            0.0, // no momentum
            0.1, // l2_reg
        );

        // Weights should decrease due to regularization
        assert!(
            weights[[0, 0]] < initial_weight,
            "Expected weight to decrease due to L2 reg"
        );
    }

    #[test]
    fn test_learning_rate_schedule_constant() {
        let config = OptimizerConfig::default();
        let mut plateau_count = 0;
        let lr = config.get_lr(100, 0.5, 0.6, &mut plateau_count);
        assert_abs_diff_eq!(lr, config.learning_rate, epsilon = 1e-10);
    }

    #[test]
    fn test_learning_rate_schedule_inverse() {
        let config = OptimizerConfig {
            learning_rate: 0.1,
            lr_schedule: LearningRateSchedule::InverseScaling { decay: 0.01 },
            ..Default::default()
        };
        let mut plateau_count = 0;

        let lr_0 = config.get_lr(0, 0.5, 0.5, &mut plateau_count);
        let lr_100 = config.get_lr(100, 0.5, 0.5, &mut plateau_count);

        assert!(lr_100 < lr_0, "Learning rate should decrease over time");
    }

    #[test]
    fn test_solver_names() {
        assert_eq!(Solver::SGD.name(), "sgd");
        assert_eq!(Solver::Adam.name(), "adam");
    }
}
