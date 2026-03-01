//! Core Multi-Layer Perceptron Implementation
//!
//! This module provides the core MLP architecture used by both
//! MLPClassifier and MLPRegressor.

use crate::neural::{
    Activation, AdamState, EarlyStopping, Layer, OptimizerConfig, Regularization, SGDState, Solver,
    TrainingHistory, WeightInit,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Core Multi-Layer Perceptron structure
///
/// This is the shared implementation used by MLPClassifier and MLPRegressor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLP {
    /// Hidden layer sizes (not including input/output)
    pub hidden_layer_sizes: Vec<usize>,
    /// Activation function for hidden layers
    pub hidden_activation: Activation,
    /// Activation function for output layer
    pub output_activation: Activation,
    /// Solver type
    pub solver: Solver,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Batch size for mini-batch gradient descent
    pub batch_size: Option<usize>,
    /// Random seed
    pub random_state: Option<u64>,
    /// Weight initialization strategy
    pub weight_init: WeightInit,
    /// Optimizer configuration
    pub optimizer_config: OptimizerConfig,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStopping>,
    /// Regularization configuration
    pub regularization: Regularization,
    /// Whether to shuffle data each epoch
    pub shuffle: bool,
    /// Verbosity level (0 = silent, 1 = progress, 2 = debug)
    pub verbose: usize,

    // Fitted state
    /// Network layers
    #[serde(skip)]
    pub layers: Vec<Layer>,
    /// Number of input features
    pub n_features_in: Option<usize>,
    /// Number of output features
    pub n_outputs: Option<usize>,
    /// Training history
    pub history: Option<TrainingHistory>,
    /// SGD optimizer state
    #[serde(skip)]
    pub sgd_state: Option<SGDState>,
    /// Adam optimizer state
    #[serde(skip)]
    pub adam_state: Option<AdamState>,
    /// Persistent RNG for dropout masks and shuffling.
    /// Initialized once during `initialize()` and advances across calls,
    /// ensuring different dropout masks per forward pass and different
    /// shuffle orders per epoch.
    #[serde(skip)]
    pub rng: Option<StdRng>,
    /// Optional GPU backend for accelerated matrix operations
    #[cfg(feature = "gpu")]
    #[serde(skip)]
    pub gpu_backend: Option<std::sync::Arc<dyn crate::gpu::GpuBackend>>,
}

impl Default for MLP {
    fn default() -> Self {
        Self {
            hidden_layer_sizes: vec![100],
            hidden_activation: Activation::ReLU,
            output_activation: Activation::Linear,
            solver: Solver::Adam,
            max_iter: 200,
            tol: 1e-4,
            batch_size: Some(200),
            random_state: None,
            weight_init: WeightInit::HeUniform,
            optimizer_config: OptimizerConfig::adam(0.001),
            early_stopping: None,
            regularization: Regularization::default(),
            shuffle: true,
            verbose: 0,

            layers: Vec::new(),
            n_features_in: None,
            n_outputs: None,
            history: None,
            sgd_state: None,
            adam_state: None,
            rng: None,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
        }
    }
}

impl MLP {
    /// Create a new MLP with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set hidden layer sizes
    pub fn hidden_layer_sizes(mut self, sizes: &[usize]) -> Self {
        self.hidden_layer_sizes = sizes.to_vec();
        self
    }

    /// Set hidden layer activation
    pub fn activation(mut self, activation: Activation) -> Self {
        self.hidden_activation = activation;
        // Use appropriate initialization for activation
        self.weight_init = match activation {
            Activation::ReLU | Activation::LeakyReLU | Activation::ELU => WeightInit::HeUniform,
            _ => WeightInit::XavierUniform,
        };
        self
    }

    /// Set output layer activation
    pub fn output_activation(mut self, activation: Activation) -> Self {
        self.output_activation = activation;
        self
    }

    /// Set solver type
    pub fn solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Set GPU backend for accelerated matrix operations
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self {
        self.gpu_backend = Some(backend);
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.optimizer_config.learning_rate = lr;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable early stopping
    pub fn early_stopping(mut self, config: EarlyStopping) -> Self {
        self.early_stopping = Some(config);
        self
    }

    /// Set L2 regularization strength
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.regularization.alpha = alpha;
        self
    }

    /// Set dropout rate
    pub fn dropout(mut self, rate: f64) -> Self {
        self.regularization.dropout_rate = rate;
        self
    }

    /// Set momentum (for SGD)
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.optimizer_config.momentum = momentum;
        self
    }

    /// Set verbose level
    pub fn verbose(mut self, level: usize) -> Self {
        self.verbose = level;
        self
    }

    /// Check if the model is fitted
    pub fn is_fitted(&self) -> bool {
        !self.layers.is_empty()
    }

    /// Get layer sizes including input and output
    pub fn layer_sizes(&self) -> Vec<usize> {
        if !self.is_fitted() {
            return vec![];
        }

        let mut sizes = vec![self.n_features_in.unwrap_or(0)];
        sizes.extend(&self.hidden_layer_sizes);
        sizes.push(self.n_outputs.unwrap_or(0));
        sizes
    }

    /// Initialize the network layers
    pub fn initialize(&mut self, n_features: usize, n_outputs: usize) -> Result<()> {
        if self.hidden_layer_sizes.is_empty() {
            return Err(FerroError::InvalidInput(
                "hidden_layer_sizes cannot be empty".to_string(),
            ));
        }

        self.n_features_in = Some(n_features);
        self.n_outputs = Some(n_outputs);

        // Build layer sizes: [n_features, hidden..., n_outputs]
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend(&self.hidden_layer_sizes);
        layer_sizes.push(n_outputs);

        // Create layers
        self.layers.clear();
        let seed_base = self.random_state.unwrap_or(0);

        for (i, window) in layer_sizes.windows(2).enumerate() {
            let n_in = window[0];
            let n_out = window[1];
            let is_output = i == layer_sizes.len() - 2;

            let activation = if is_output {
                self.output_activation
            } else {
                self.hidden_activation
            };

            let layer = Layer::new(
                n_in,
                n_out,
                activation,
                self.weight_init,
                Some(seed_base + i as u64),
            );
            self.layers.push(layer);
        }

        // Initialize optimizer state
        let layer_dims: Vec<(usize, usize)> = self
            .layers
            .iter()
            .map(|l| (l.n_inputs, l.n_outputs))
            .collect();

        match self.solver {
            Solver::SGD => {
                self.sgd_state = Some(SGDState::new(&layer_dims));
            }
            Solver::Adam => {
                self.adam_state = Some(AdamState::new(&layer_dims));
            }
        }

        // Initialize persistent RNG for dropout and shuffling.
        // Use a different offset from the weight init seeds to avoid correlation.
        self.rng = Some(match self.random_state {
            Some(s) => StdRng::seed_from_u64(s.wrapping_add(1_000_000)),
            None => StdRng::from_os_rng(),
        });

        Ok(())
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    /// * `x` - Input data of shape (n_samples, n_features)
    /// * `training` - Whether in training mode (affects dropout, caching)
    pub fn forward(&mut self, x: &Array2<f64>, training: bool) -> Result<Array2<f64>> {
        if self.layers.is_empty() {
            return Err(FerroError::not_fitted("forward"));
        }

        let mut output = x.clone();

        // Use the persistent RNG for dropout (advances across calls, giving
        // different masks each forward pass). Falls back to a fresh OS RNG
        // only if initialize() was never called.
        let needs_rng = training && self.regularization.dropout_rate > 0.0;

        let n_layers = self.layers.len();
        for i in 0..n_layers {
            let is_output_layer = i == n_layers - 1;

            // Apply dropout to hidden layers only during training
            if needs_rng && !is_output_layer {
                // We need a mutable reference to both the layer and the rng.
                // Borrow the rng from self first, then the layer.
                let rng = self.rng.as_mut().expect("RNG should be initialized");
                let layer = &mut self.layers[i];
                output = layer.forward_with_dropout(
                    &output,
                    self.regularization.dropout_rate,
                    true,
                    rng,
                )?;
            } else {
                // Use GPU-accelerated forward when available
                #[cfg(feature = "gpu")]
                {
                    if let Some(ref gpu) = self.gpu_backend {
                        output = self.layers[i].forward_gpu(&output, training, gpu.as_ref())?;
                        continue;
                    }
                }
                output = self.layers[i].forward(&output, training)?;
            }
        }

        Ok(output)
    }

    /// Backward pass through the network
    ///
    /// # Arguments
    /// * `loss_grad` - Gradient of loss with respect to output.
    ///   For the output layer, this should be the combined loss+activation gradient
    ///   (e.g., `(p - y) / n` for softmax + cross-entropy, or `2*(p - y) / n` for linear + MSE).
    ///   The output layer's activation derivative is NOT applied again.
    ///
    /// # Returns
    /// Gradients for each layer (weights, biases)
    pub fn backward(&self, loss_grad: &Array2<f64>) -> Result<Vec<(Array2<f64>, Array1<f64>)>> {
        if self.layers.is_empty() {
            return Err(FerroError::not_fitted("forward"));
        }

        let mut gradients = Vec::new();
        let mut grad = loss_grad.clone();

        // Backpropagate through layers in reverse order
        for (rev_idx, layer) in self.layers.iter().rev().enumerate() {
            let is_output_layer = rev_idx == 0;

            #[cfg(feature = "gpu")]
            let (grad_w, grad_b, grad_input) = if let Some(ref gpu) = self.gpu_backend {
                layer.backward_gpu(&grad, gpu.as_ref())?
            } else if is_output_layer {
                // For output layer, the loss gradient already accounts for the
                // activation derivative (softmax+CE or linear+MSE), so skip it.
                layer.backward_skip_activation(&grad)?
            } else {
                layer.backward(&grad)?
            };
            #[cfg(not(feature = "gpu"))]
            let (grad_w, grad_b, grad_input) = if is_output_layer {
                // For output layer, the loss gradient already accounts for the
                // activation derivative (softmax+CE or linear+MSE), so skip it.
                layer.backward_skip_activation(&grad)?
            } else {
                layer.backward(&grad)?
            };

            gradients.push((grad_w, grad_b));
            grad = grad_input;
        }

        // Reverse to match layer order
        gradients.reverse();
        Ok(gradients)
    }

    /// Update weights using the optimizer
    pub fn update_weights(
        &mut self,
        gradients: &[(Array2<f64>, Array1<f64>)],
        lr: f64,
    ) -> Result<()> {
        let l2_reg = self.regularization.alpha;

        match self.solver {
            Solver::SGD => {
                let state = self
                    .sgd_state
                    .as_mut()
                    .ok_or(FerroError::not_fitted("forward"))?;
                let momentum = self.optimizer_config.momentum;

                for (i, layer) in self.layers.iter_mut().enumerate() {
                    let (grad_w, grad_b) = &gradients[i];
                    crate::neural::optimizers::sgd_step(
                        &mut layer.weights,
                        &mut layer.biases,
                        grad_w,
                        grad_b,
                        &mut state.velocities_w[i],
                        &mut state.velocities_b[i],
                        lr,
                        momentum,
                        l2_reg,
                    );
                }
            }
            Solver::Adam => {
                let state = self
                    .adam_state
                    .as_mut()
                    .ok_or(FerroError::not_fitted("forward"))?;
                state.t += 1;
                let beta1 = self.optimizer_config.beta1;
                let beta2 = self.optimizer_config.beta2;
                let epsilon = self.optimizer_config.epsilon;

                for (i, layer) in self.layers.iter_mut().enumerate() {
                    let (grad_w, grad_b) = &gradients[i];
                    crate::neural::optimizers::adam_step(
                        &mut layer.weights,
                        &mut layer.biases,
                        grad_w,
                        grad_b,
                        &mut state.m_w[i],
                        &mut state.m_b[i],
                        &mut state.v_w[i],
                        &mut state.v_b[i],
                        state.t,
                        lr,
                        beta1,
                        beta2,
                        epsilon,
                        l2_reg,
                    );
                }
            }
        }

        Ok(())
    }

    /// Train the network for one epoch
    ///
    /// # Arguments
    /// * `x` - Training features
    /// * `y` - Training targets (for loss computation)
    /// * `loss_fn` - Loss function that returns (loss, gradient)
    ///
    /// # Returns
    /// Average loss for the epoch
    pub fn train_epoch<F>(&mut self, x: &Array2<f64>, y: &Array2<f64>, loss_fn: F) -> Result<f64>
    where
        F: Fn(&Array2<f64>, &Array2<f64>) -> (f64, Array2<f64>),
    {
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err(FerroError::InvalidInput(
                "Training data must have at least one sample".to_string(),
            ));
        }
        let batch_size = self.batch_size.unwrap_or(n_samples).min(n_samples);
        let lr = self.optimizer_config.learning_rate;

        // Create indices for shuffling using the persistent RNG
        // (gives different shuffle order each epoch)
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle {
            if let Some(ref mut rng) = self.rng {
                indices.shuffle(rng);
            }
        }

        let mut total_loss = 0.0;
        let mut n_batches = 0;

        // Process in batches
        for chunk in indices.chunks(batch_size) {
            // Extract batch
            let batch_x = x.select(Axis(0), chunk);
            let batch_y = y.select(Axis(0), chunk);

            // Forward pass
            let output = self.forward(&batch_x, true)?;

            // Compute loss and gradient
            let (loss, loss_grad) = loss_fn(&output, &batch_y);
            total_loss += loss * chunk.len() as f64;
            n_batches += chunk.len();

            // Backward pass
            let gradients = self.backward(&loss_grad)?;

            // Update weights
            self.update_weights(&gradients, lr)?;
        }

        // Clear caches
        for layer in &mut self.layers {
            layer.clear_cache();
        }

        Ok(total_loss / n_batches as f64)
    }

    /// Get the total number of parameters
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// Get weights for all layers
    pub fn get_weights(&self) -> Vec<(&Array2<f64>, &Array1<f64>)> {
        self.layers
            .iter()
            .map(|l| (&l.weights, &l.biases))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_creation() {
        let mlp = MLP::new()
            .hidden_layer_sizes(&[10, 5])
            .activation(Activation::ReLU)
            .solver(Solver::Adam);

        assert_eq!(mlp.hidden_layer_sizes, vec![10, 5]);
        assert_eq!(mlp.hidden_activation, Activation::ReLU);
        assert_eq!(mlp.solver, Solver::Adam);
    }

    #[test]
    fn test_mlp_initialization() {
        let mut mlp = MLP::new().hidden_layer_sizes(&[10, 5]);
        mlp.initialize(4, 2).unwrap();

        assert_eq!(mlp.layers.len(), 3); // input->10, 10->5, 5->output
        assert_eq!(mlp.layers[0].n_inputs, 4);
        assert_eq!(mlp.layers[0].n_outputs, 10);
        assert_eq!(mlp.layers[1].n_inputs, 10);
        assert_eq!(mlp.layers[1].n_outputs, 5);
        assert_eq!(mlp.layers[2].n_inputs, 5);
        assert_eq!(mlp.layers[2].n_outputs, 2);
    }

    #[test]
    fn test_forward_pass() {
        let mut mlp = MLP::new().hidden_layer_sizes(&[4]).random_state(42);
        mlp.initialize(2, 1).unwrap();

        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = mlp.forward(&x, false).unwrap();

        assert_eq!(output.shape(), &[3, 1]);
    }

    #[test]
    fn test_layer_sizes() {
        let mut mlp = MLP::new().hidden_layer_sizes(&[10, 5]);
        mlp.initialize(4, 2).unwrap();

        let sizes = mlp.layer_sizes();
        assert_eq!(sizes, vec![4, 10, 5, 2]);
    }

    #[test]
    fn test_n_params() {
        let mut mlp = MLP::new().hidden_layer_sizes(&[10]);
        mlp.initialize(4, 2).unwrap();

        // Layer 1: 4*10 + 10 = 50
        // Layer 2: 10*2 + 2 = 22
        // Total: 72
        assert_eq!(mlp.n_params(), 72);
    }

    #[test]
    fn test_backward_pass() {
        let mut mlp = MLP::new().hidden_layer_sizes(&[4]).random_state(42);
        mlp.initialize(2, 1).unwrap();

        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let _ = mlp.forward(&x, true).unwrap();

        let loss_grad = Array2::ones((3, 1));
        let gradients = mlp.backward(&loss_grad).unwrap();

        assert_eq!(gradients.len(), 2);
    }
}
