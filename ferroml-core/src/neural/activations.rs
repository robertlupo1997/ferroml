//! Activation Functions for Neural Networks
//!
//! This module provides various activation functions used in neural networks.
//!
//! ## Activations
//!
//! - [`Activation::ReLU`] - Rectified Linear Unit, most common for hidden layers
//! - [`Activation::Sigmoid`] - Logistic sigmoid, outputs in (0, 1)
//! - [`Activation::Tanh`] - Hyperbolic tangent, outputs in (-1, 1)
//! - [`Activation::Softmax`] - Normalized exponential, for multi-class output
//! - [`Activation::Linear`] - Identity, for regression output
//! - [`Activation::LeakyReLU`] - ReLU with small slope for negative values
//! - [`Activation::ELU`] - Exponential Linear Unit

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    #[default]
    ReLU,
    /// Logistic sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Softmax: exp(x_i) / sum(exp(x_j))
    Softmax,
    /// Identity function: x
    Linear,
    /// Leaky ReLU: max(alpha * x, x), alpha = 0.01
    LeakyReLU,
    /// Exponential Linear Unit: x if x > 0, alpha * (exp(x) - 1) otherwise
    ELU,
}

impl Activation {
    /// Apply the activation function element-wise to a 1D array
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Softmax => {
                // Numerically stable softmax
                let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_x = x.mapv(|v| (v - max_val).exp());
                let sum_exp: f64 = exp_x.sum();
                exp_x / sum_exp
            }
            Activation::Linear => x.clone(),
            Activation::LeakyReLU => x.mapv(|v| if v > 0.0 { v } else { 0.01 * v }),
            Activation::ELU => x.mapv(|v| if v > 0.0 { v } else { v.exp() - 1.0 }),
        }
    }

    /// Apply the activation function to a 2D array (row-wise for softmax)
    pub fn apply_2d(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Softmax => {
                // Apply softmax row-wise (each sample independently)
                let mut result = Array2::zeros(x.raw_dim());
                for (i, row) in x.axis_iter(Axis(0)).enumerate() {
                    let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_row: Array1<f64> = row.mapv(|v| (v - max_val).exp());
                    let sum_exp: f64 = exp_row.sum();
                    result.row_mut(i).assign(&(&exp_row / sum_exp));
                }
                result
            }
            Activation::Linear => x.clone(),
            Activation::LeakyReLU => x.mapv(|v| if v > 0.0 { v } else { 0.01 * v }),
            Activation::ELU => x.mapv(|v| if v > 0.0 { v } else { v.exp() - 1.0 }),
        }
    }

    /// Compute the derivative of the activation function
    ///
    /// # Arguments
    /// * `x` - The pre-activation values (input to the activation)
    /// * `output` - The post-activation values (output of the activation)
    ///
    /// # Note
    /// For Softmax, this returns the diagonal of the Jacobian (simplified for cross-entropy loss)
    pub fn derivative(&self, x: &Array1<f64>, output: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => output * &(1.0 - output),
            Activation::Tanh => 1.0 - output * output,
            Activation::Softmax => {
                // For cross-entropy loss, the combined gradient simplifies
                // This returns s * (1 - s) which is used in some contexts
                output * &(1.0 - output)
            }
            Activation::Linear => Array1::ones(x.len()),
            Activation::LeakyReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.01 }),
            Activation::ELU => {
                // d/dx = 1 if x >= 0, else exp(x) = output + 1
                // Use >= to match the forward pass boundary
                x.iter()
                    .zip(output.iter())
                    .map(|(&xi, &oi)| if xi >= 0.0 { 1.0 } else { oi + 1.0 })
                    .collect::<Array1<f64>>()
            }
        }
    }

    /// Compute the derivative for a 2D array
    pub fn derivative_2d(&self, x: &Array2<f64>, output: &Array2<f64>) -> Array2<f64> {
        match self {
            Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => output * &(1.0 - output),
            Activation::Tanh => 1.0 - output * output,
            Activation::Softmax => output * &(1.0 - output),
            Activation::Linear => Array2::ones(x.raw_dim()),
            Activation::LeakyReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.01 }),
            Activation::ELU => {
                let mut result = Array2::zeros(x.raw_dim());
                for ((xi, oi), ri) in x.iter().zip(output.iter()).zip(result.iter_mut()) {
                    *ri = if *xi >= 0.0 { 1.0 } else { oi + 1.0 };
                }
                result
            }
        }
    }

    /// Get a human-readable name for the activation
    pub fn name(&self) -> &'static str {
        match self {
            Activation::ReLU => "relu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::Softmax => "softmax",
            Activation::Linear => "linear",
            Activation::LeakyReLU => "leaky_relu",
            Activation::ELU => "elu",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_relu() {
        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = Activation::ReLU.apply(&x);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[4], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let result = Activation::Sigmoid.apply(&x);
        assert_abs_diff_eq!(result[1], 0.5, epsilon = 1e-10);
        assert!(result[0] < 0.5);
        assert!(result[2] > 0.5);
    }

    #[test]
    fn test_tanh() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let result = Activation::Tanh.apply(&x);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10);
        assert!(result[0] < 0.0);
        assert!(result[2] > 0.0);
    }

    #[test]
    fn test_softmax() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = Activation::Softmax.apply(&x);
        // Sum should be 1
        assert_abs_diff_eq!(result.sum(), 1.0, epsilon = 1e-10);
        // Should be in ascending order
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not cause overflow
        let x = Array1::from_vec(vec![1000.0, 1001.0, 1002.0]);
        let result = Activation::Softmax.apply(&x);
        assert_abs_diff_eq!(result.sum(), 1.0, epsilon = 1e-10);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_linear() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let result = Activation::Linear.apply(&x);
        assert_eq!(result, x);
    }

    #[test]
    fn test_leaky_relu() {
        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = Activation::LeakyReLU.apply(&x);
        assert_abs_diff_eq!(result[0], -0.02, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], -0.01, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_relu_derivative() {
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let output = Activation::ReLU.apply(&x);
        let deriv = Activation::ReLU.derivative(&x, &output);
        assert_abs_diff_eq!(deriv[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let x = Array1::from_vec(vec![0.0]);
        let output = Activation::Sigmoid.apply(&x);
        let deriv = Activation::Sigmoid.derivative(&x, &output);
        // sigmoid'(0) = 0.5 * 0.5 = 0.25
        assert_abs_diff_eq!(deriv[0], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_activation_names() {
        assert_eq!(Activation::ReLU.name(), "relu");
        assert_eq!(Activation::Sigmoid.name(), "sigmoid");
        assert_eq!(Activation::Softmax.name(), "softmax");
    }

    #[test]
    fn test_2d_softmax() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]).unwrap();
        let result = Activation::Softmax.apply_2d(&x);

        // Each row should sum to 1
        assert_abs_diff_eq!(result.row(0).sum(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.row(1).sum(), 1.0, epsilon = 1e-10);

        // Second row should be uniform (all inputs equal)
        let expected = 1.0 / 3.0;
        assert_abs_diff_eq!(result[[1, 0]], expected, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], expected, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 2]], expected, epsilon = 1e-10);
    }
}
