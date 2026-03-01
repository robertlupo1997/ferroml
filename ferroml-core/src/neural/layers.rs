//! Neural Network Layers
//!
//! This module provides layer implementations for neural networks.
//!
//! ## Layer Types
//!
//! - [`Layer`] - A fully connected (dense) layer with weights, biases, and activation

use crate::neural::{Activation, WeightInit};
use crate::Result;
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// A fully connected (dense) neural network layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Weight matrix of shape (n_inputs, n_outputs)
    pub weights: Array2<f64>,
    /// Bias vector of shape (n_outputs,)
    pub biases: Array1<f64>,
    /// Activation function
    pub activation: Activation,
    /// Number of input features
    pub n_inputs: usize,
    /// Number of output features (neurons)
    pub n_outputs: usize,

    // Cached values for backpropagation
    /// Last input to this layer (set during forward pass)
    #[serde(skip)]
    pub last_input: Option<Array2<f64>>,
    /// Last pre-activation output z = Wx + b (set during forward pass)
    #[serde(skip)]
    pub last_z: Option<Array2<f64>>,
    /// Last post-activation output a = activation(z) (set during forward pass)
    #[serde(skip)]
    pub last_output: Option<Array2<f64>>,
    /// Dropout mask (if applicable)
    #[serde(skip)]
    pub dropout_mask: Option<Array2<f64>>,
}

impl Layer {
    /// Create a new layer with random initialization
    ///
    /// # Arguments
    /// * `n_inputs` - Number of input features
    /// * `n_outputs` - Number of output neurons
    /// * `activation` - Activation function to use
    /// * `weight_init` - Weight initialization strategy
    /// * `seed` - Optional random seed
    pub fn new(
        n_inputs: usize,
        n_outputs: usize,
        activation: Activation,
        weight_init: WeightInit,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };

        let weights = Self::initialize_weights(n_inputs, n_outputs, weight_init, &mut rng);
        let biases = Array1::zeros(n_outputs);

        Self {
            weights,
            biases,
            activation,
            n_inputs,
            n_outputs,
            last_input: None,
            last_z: None,
            last_output: None,
            dropout_mask: None,
        }
    }

    /// Initialize weights using the specified strategy
    fn initialize_weights(
        n_inputs: usize,
        n_outputs: usize,
        init: WeightInit,
        rng: &mut StdRng,
    ) -> Array2<f64> {
        let mut weights = Array2::zeros((n_inputs, n_outputs));

        match init {
            WeightInit::XavierUniform => {
                // Glorot uniform: [-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
                let limit = (6.0 / (n_inputs + n_outputs) as f64).sqrt();
                for w in weights.iter_mut() {
                    *w = rng.random_range(-limit..limit);
                }
            }
            WeightInit::XavierNormal => {
                // Glorot normal: std = sqrt(2 / (fan_in + fan_out))
                let std = (2.0 / (n_inputs + n_outputs) as f64).sqrt();
                let normal = rand_distr::Normal::new(0.0, std).unwrap();
                for w in weights.iter_mut() {
                    *w = rng.sample(normal);
                }
            }
            WeightInit::HeUniform => {
                // He uniform: [-sqrt(6 / fan_in), sqrt(6 / fan_in)]
                let limit = (6.0 / n_inputs as f64).sqrt();
                for w in weights.iter_mut() {
                    *w = rng.random_range(-limit..limit);
                }
            }
            WeightInit::HeNormal => {
                // He normal: std = sqrt(2 / fan_in)
                let std = (2.0 / n_inputs as f64).sqrt();
                let normal = rand_distr::Normal::new(0.0, std).unwrap();
                for w in weights.iter_mut() {
                    *w = rng.sample(normal);
                }
            }
            WeightInit::Uniform => {
                // Simple uniform in [-0.5, 0.5]
                for w in weights.iter_mut() {
                    *w = rng.random_range(-0.5..0.5);
                }
            }
            WeightInit::Normal => {
                // Simple normal with std = 0.1
                let normal = rand_distr::Normal::new(0.0, 0.1).unwrap();
                for w in weights.iter_mut() {
                    *w = rng.sample(normal);
                }
            }
        }

        weights
    }

    /// Forward pass through the layer
    ///
    /// # Arguments
    /// * `input` - Input data of shape (batch_size, n_inputs)
    /// * `training` - Whether in training mode (affects caching)
    ///
    /// # Returns
    /// Output of shape (batch_size, n_outputs)
    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Result<Array2<f64>> {
        if input.ncols() != self.n_inputs {
            return Err(crate::FerroError::InvalidInput(format!(
                "Expected {} input features, got {}",
                self.n_inputs,
                input.ncols()
            )));
        }

        // z = X @ W + b
        let z = input.dot(&self.weights) + &self.biases;

        // a = activation(z)
        let output = self.activation.apply_2d(&z);

        // Cache for backpropagation if training
        if training {
            self.last_input = Some(input.clone());
            self.last_z = Some(z);
            self.last_output = Some(output.clone());
        }

        Ok(output)
    }

    /// Forward pass with dropout
    ///
    /// # Arguments
    /// * `input` - Input data
    /// * `dropout_rate` - Probability of dropping a neuron (0.0 to 1.0)
    /// * `training` - Whether in training mode
    /// * `rng` - Random number generator
    pub fn forward_with_dropout(
        &mut self,
        input: &Array2<f64>,
        dropout_rate: f64,
        training: bool,
        rng: &mut StdRng,
    ) -> Result<Array2<f64>> {
        let output = self.forward(input, training)?;

        if training && dropout_rate > 0.0 {
            // Create dropout mask
            let mut mask = Array2::zeros(output.raw_dim());
            let keep_prob = 1.0 - dropout_rate;
            for m in mask.iter_mut() {
                *m = if rng.random::<f64>() < keep_prob {
                    1.0 / keep_prob // Scale to maintain expected value
                } else {
                    0.0
                };
            }

            self.dropout_mask = Some(mask.clone());
            Ok(&output * &mask)
        } else {
            self.dropout_mask = None;
            Ok(output)
        }
    }

    /// Backward pass through the layer
    ///
    /// # Arguments
    /// * `grad_output` - Gradient from the next layer (shape: batch_size x n_outputs)
    ///
    /// # Returns
    /// (grad_weights, grad_biases, grad_input)
    pub fn backward(
        &self,
        grad_output: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        self.backward_impl(grad_output, false)
    }

    /// Backward pass that skips the activation derivative.
    ///
    /// Used for the output layer when the loss gradient already incorporates
    /// the activation derivative (e.g., softmax + cross-entropy, linear + MSE).
    pub fn backward_skip_activation(
        &self,
        grad_output: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        self.backward_impl(grad_output, true)
    }

    /// Internal backward pass implementation.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient from the next layer
    /// * `skip_activation_deriv` - If true, the activation derivative is not applied
    ///   (used for the output layer when loss gradient already accounts for it)
    fn backward_impl(
        &self,
        grad_output: &Array2<f64>,
        skip_activation_deriv: bool,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let z = self
            .last_z
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("backward"))?;
        let output = self
            .last_output
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("backward"))?;
        let input = self
            .last_input
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("backward"))?;

        // Apply dropout mask to gradient if it was used
        let grad_output = match &self.dropout_mask {
            Some(mask) => grad_output * mask,
            None => grad_output.clone(),
        };

        // For the output layer with combined loss+activation gradient
        // (e.g., softmax+CE, linear+MSE), the loss gradient already
        // accounts for the activation — don't apply derivative again.
        let delta = if skip_activation_deriv {
            grad_output
        } else {
            // Compute activation derivative
            let activation_deriv = self.activation.derivative_2d(z, output);
            // Delta = grad_output * activation'(z)
            &grad_output * &activation_deriv
        };

        // Gradient for weights: X^T @ delta
        let grad_weights = input.t().dot(&delta);

        // Gradient for biases: sum over batch
        let grad_biases = delta.sum_axis(Axis(0));

        // Gradient for input: delta @ W^T
        let grad_input = delta.dot(&self.weights.t());

        Ok((grad_weights, grad_biases, grad_input))
    }

    /// Get the number of parameters in this layer
    pub fn n_params(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    /// Clear cached values (call after training epoch)
    pub fn clear_cache(&mut self) {
        self.last_input = None;
        self.last_z = None;
        self.last_output = None;
        self.dropout_mask = None;
    }

    /// GPU-accelerated forward pass
    ///
    /// Uses the GPU backend for matrix multiplication when available.
    /// Falls back to CPU if GPU matmul fails.
    #[cfg(feature = "gpu")]
    pub fn forward_gpu(
        &mut self,
        input: &Array2<f64>,
        training: bool,
        gpu: &dyn crate::gpu::GpuBackend,
    ) -> Result<Array2<f64>> {
        if input.ncols() != self.n_inputs {
            return Err(crate::FerroError::InvalidInput(format!(
                "Expected {} input features, got {}",
                self.n_inputs,
                input.ncols()
            )));
        }

        // z = X @ W + b (GPU-accelerated matmul)
        let z = match gpu.matmul(input, &self.weights) {
            Ok(result) => result + &self.biases,
            Err(_) => input.dot(&self.weights) + &self.biases, // CPU fallback
        };

        let output = self.activation.apply_2d(&z);

        if training {
            self.last_input = Some(input.clone());
            self.last_z = Some(z);
            self.last_output = Some(output.clone());
        }

        Ok(output)
    }

    /// GPU-accelerated backward pass
    #[cfg(feature = "gpu")]
    pub fn backward_gpu(
        &self,
        grad_output: &Array2<f64>,
        gpu: &dyn crate::gpu::GpuBackend,
    ) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let z = self
            .last_z
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("backward"))?;
        let output = self
            .last_output
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("backward"))?;
        let input = self
            .last_input
            .as_ref()
            .ok_or_else(|| crate::FerroError::not_fitted("backward"))?;

        let grad_output = match &self.dropout_mask {
            Some(mask) => grad_output * mask,
            None => grad_output.clone(),
        };

        let activation_deriv = self.activation.derivative_2d(z, output);
        let delta = &grad_output * &activation_deriv;

        // GPU-accelerated matrix multiplications
        let grad_weights = gpu
            .matmul(&input.t().to_owned(), &delta)
            .unwrap_or_else(|_| input.t().dot(&delta));
        let grad_biases = delta.sum_axis(Axis(0));
        let grad_input = gpu
            .matmul(&delta, &self.weights.t().to_owned())
            .unwrap_or_else(|_| delta.dot(&self.weights.t()));

        Ok((grad_weights, grad_biases, grad_input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(10, 5, Activation::ReLU, WeightInit::XavierUniform, Some(42));
        assert_eq!(layer.n_inputs, 10);
        assert_eq!(layer.n_outputs, 5);
        assert_eq!(layer.weights.shape(), &[10, 5]);
        assert_eq!(layer.biases.len(), 5);
    }

    #[test]
    fn test_forward_pass() {
        let mut layer = Layer::new(
            3,
            2,
            Activation::Linear,
            WeightInit::XavierUniform,
            Some(42),
        );
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let output = layer.forward(&input, true).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_forward_caches_for_training() {
        let mut layer = Layer::new(3, 2, Activation::ReLU, WeightInit::XavierUniform, Some(42));
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        layer.forward(&input, true).unwrap();
        assert!(layer.last_input.is_some());
        assert!(layer.last_z.is_some());
        assert!(layer.last_output.is_some());
    }

    #[test]
    fn test_forward_no_cache_for_inference() {
        let mut layer = Layer::new(3, 2, Activation::ReLU, WeightInit::XavierUniform, Some(42));
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        layer.forward(&input, false).unwrap();
        assert!(layer.last_input.is_none());
    }

    #[test]
    fn test_backward_pass() {
        let mut layer = Layer::new(
            3,
            2,
            Activation::Linear,
            WeightInit::XavierUniform,
            Some(42),
        );
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        layer.forward(&input, true).unwrap();

        let grad_output = Array2::ones((2, 2));
        let (grad_weights, grad_biases, grad_input) = layer.backward(&grad_output).unwrap();

        assert_eq!(grad_weights.shape(), &[3, 2]);
        assert_eq!(grad_biases.len(), 2);
        assert_eq!(grad_input.shape(), &[2, 3]);
    }

    #[test]
    fn test_n_params() {
        let layer = Layer::new(10, 5, Activation::ReLU, WeightInit::XavierUniform, Some(42));
        assert_eq!(layer.n_params(), 10 * 5 + 5); // weights + biases
    }

    #[test]
    fn test_he_initialization_scale() {
        // He init should have std ~= sqrt(2/fan_in)
        let layer = Layer::new(100, 50, Activation::ReLU, WeightInit::HeNormal, Some(42));
        let expected_std = (2.0 / 100.0_f64).sqrt();
        let actual_std = (layer.weights.mapv(|x| x * x).sum() / layer.weights.len() as f64).sqrt();
        // Allow 20% tolerance due to random sampling
        assert!(
            (actual_std - expected_std).abs() / expected_std < 0.2,
            "std {} not close to expected {}",
            actual_std,
            expected_std
        );
    }

    #[test]
    fn test_dropout() {
        let mut layer = Layer::new(
            4,
            4,
            Activation::Linear,
            WeightInit::XavierUniform,
            Some(42),
        );
        let input = Array2::ones((10, 4));
        let mut rng = StdRng::seed_from_u64(42);

        let output = layer
            .forward_with_dropout(&input, 0.5, true, &mut rng)
            .unwrap();

        // Some values should be zero (dropped)
        let n_zeros = output.iter().filter(|&&v| v == 0.0).count();
        assert!(n_zeros > 0, "Expected some dropout");
        assert!(n_zeros < output.len(), "Expected some values to remain");
    }

    #[test]
    fn test_clear_cache() {
        let mut layer = Layer::new(3, 2, Activation::ReLU, WeightInit::XavierUniform, Some(42));
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        layer.forward(&input, true).unwrap();
        assert!(layer.last_input.is_some());

        layer.clear_cache();
        assert!(layer.last_input.is_none());
        assert!(layer.last_z.is_none());
        assert!(layer.last_output.is_none());
    }
}
