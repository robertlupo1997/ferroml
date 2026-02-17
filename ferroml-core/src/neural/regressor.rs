//! Multi-Layer Perceptron Regressor
//!
//! This module provides `MLPRegressor` for regression tasks.
//!
//! ## Features
//!
//! - sklearn-compatible API: fit(), predict()
//! - MSE loss with linear output
//! - Early stopping with validation
//! - Statistical extensions: training diagnostics, uncertainty quantification

use crate::neural::{
    Activation, EarlyStopping, NeuralDiagnostics, NeuralModel, NeuralUncertainty,
    PredictionUncertainty, Solver, TrainingDiagnostics, WeightStatistics, MLP,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Multi-Layer Perceptron Regressor
///
/// A neural network for regression with sklearn-compatible API.
///
/// # Example
///
/// ```
/// # use ferroml_core::neural::{MLPRegressor, Activation, Solver, NeuralModel};
/// # use ndarray::{Array1, Array2};
/// let X = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]); // y = 2x
///
/// let mut mlp = MLPRegressor::new()
///     .hidden_layer_sizes(&[10])
///     .activation(Activation::ReLU)
///     .solver(Solver::Adam)
///     .max_iter(500);
///
/// mlp.fit(&X, &y).unwrap();
/// let predictions = mlp.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPRegressor {
    /// Core MLP network
    pub mlp: MLP,
    /// Number of output features
    pub n_outputs: Option<usize>,
    /// Target mean (for normalization)
    target_mean: f64,
    /// Target std (for normalization)
    target_std: f64,
    /// Training diagnostics
    #[serde(skip)]
    diagnostics: Option<TrainingDiagnostics>,
}

impl Default for MLPRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl MLPRegressor {
    /// Create a new MLPRegressor with default parameters
    pub fn new() -> Self {
        Self {
            mlp: MLP::new().output_activation(Activation::Linear),
            n_outputs: None,
            target_mean: 0.0,
            target_std: 1.0,
            diagnostics: None,
        }
    }

    /// Set hidden layer sizes
    pub fn hidden_layer_sizes(mut self, sizes: &[usize]) -> Self {
        self.mlp = self.mlp.hidden_layer_sizes(sizes);
        self
    }

    /// Set hidden layer activation
    pub fn activation(mut self, activation: Activation) -> Self {
        self.mlp = self.mlp.activation(activation);
        self
    }

    /// Set solver type
    pub fn solver(mut self, solver: Solver) -> Self {
        self.mlp = self.mlp.solver(solver);
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.mlp = self.mlp.learning_rate(lr);
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.mlp = self.mlp.max_iter(max_iter);
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.mlp = self.mlp.random_state(seed);
        self
    }

    /// Enable early stopping
    pub fn early_stopping(mut self, config: EarlyStopping) -> Self {
        self.mlp = self.mlp.early_stopping(config);
        self
    }

    /// Set L2 regularization strength
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.mlp = self.mlp.alpha(alpha);
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.mlp = self.mlp.batch_size(size);
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.mlp = self.mlp.tol(tol);
        self
    }

    /// Set verbose level
    pub fn verbose(mut self, level: usize) -> Self {
        self.mlp = self.mlp.verbose(level);
        self
    }

    /// Mean Squared Error loss function
    fn mse_loss(predictions: &Array2<f64>, targets: &Array2<f64>) -> (f64, Array2<f64>) {
        let n_samples = predictions.nrows() as f64;

        // MSE = mean((pred - target)^2)
        let diff = predictions - targets;
        let loss = diff.mapv(|d| d.powi(2)).sum() / n_samples;

        // Gradient: 2 * (pred - target) / n
        let grad = diff.mapv(|d| 2.0 * d) / n_samples;

        (loss, grad)
    }

    /// Get R² score (coefficient of determination)
    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;

        let y_mean = y.sum() / y.len() as f64;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, yi)| (yi - p).powi(2))
            .sum();

        if ss_tot < 1e-10 {
            // All targets are the same
            Ok(if ss_res < 1e-10 { 1.0 } else { 0.0 })
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }
}

impl NeuralModel for MLPRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.ncols();
        let n_outputs = 1; // Single output for now

        self.n_outputs = Some(n_outputs);

        // Normalize targets for better convergence
        self.target_mean = y.sum() / y.len() as f64;
        let variance = y.mapv(|yi| (yi - self.target_mean).powi(2)).sum() / y.len() as f64;
        self.target_std = variance.sqrt().max(1e-10);

        let y_normalized = y.mapv(|yi| (yi - self.target_mean) / self.target_std);

        // Reshape y to 2D
        let y_2d = y_normalized
            .clone()
            .into_shape_with_order((y.len(), 1))
            .map_err(|e| FerroError::InvalidInput(e.to_string()))?;

        // Initialize network
        self.mlp.initialize(n_features, n_outputs)?;

        // Training loop
        let mut diagnostics = TrainingDiagnostics::new(self.mlp.optimizer_config.learning_rate);
        let max_iter = self.mlp.max_iter;
        let tol = self.mlp.tol;

        // Handle early stopping
        let (x_train, y_train, x_val, y_val) = if let Some(ref es) = self.mlp.early_stopping {
            let n_samples = x.nrows();
            let n_val = ((n_samples as f64) * es.validation_fraction) as usize;
            let n_train = n_samples - n_val;

            let mut rng = match self.mlp.random_state {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_os_rng(),
            };

            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let train_idx: Vec<usize> = indices[..n_train].to_vec();
            let val_idx: Vec<usize> = indices[n_train..].to_vec();

            (
                x.select(Axis(0), &train_idx),
                y_2d.select(Axis(0), &train_idx),
                Some(x.select(Axis(0), &val_idx)),
                Some(y_2d.select(Axis(0), &val_idx)),
            )
        } else {
            (x.clone(), y_2d, None, None)
        };

        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let patience = self
            .mlp
            .early_stopping
            .as_ref()
            .map(|e| e.patience)
            .unwrap_or(10);

        for epoch in 0..max_iter {
            // Train epoch
            let loss = self.mlp.train_epoch(&x_train, &y_train, Self::mse_loss)?;

            // Validation loss
            let val_loss = if let (Some(ref xv), Some(ref yv)) = (&x_val, &y_val) {
                let mut mlp_clone = self.mlp.clone();
                let preds = mlp_clone.forward(xv, false)?;
                let (vl, _) = Self::mse_loss(&preds, yv);
                Some(vl)
            } else {
                None
            };

            diagnostics.record_epoch(loss, val_loss);

            // Early stopping check
            if let Some(vl) = val_loss {
                if vl < best_val_loss - tol {
                    best_val_loss = vl;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                }

                if patience_counter >= patience {
                    if self.mlp.verbose > 0 {
                        println!("Early stopping at epoch {}", epoch + 1);
                    }
                    break;
                }
            }

            // Convergence check (training loss)
            if epoch > 0 {
                let prev_loss = diagnostics.loss_curve[epoch - 1];
                if (prev_loss - loss).abs() < tol {
                    if self.mlp.verbose > 0 {
                        println!("Converged at epoch {}", epoch + 1);
                    }
                    break;
                }
            }

            if self.mlp.verbose > 0 && (epoch + 1) % 50 == 0 {
                if let Some(vl) = val_loss {
                    println!("Epoch {}: loss={:.6}, val_loss={:.6}", epoch + 1, loss, vl);
                } else {
                    println!("Epoch {}: loss={:.6}", epoch + 1, loss);
                }
            }
        }

        diagnostics.analyze_convergence(tol, patience);
        self.diagnostics = Some(diagnostics);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("predict"));
        }

        let mut mlp = self.mlp.clone();
        let output = mlp.forward(x, false)?;

        // Denormalize predictions
        let predictions: Array1<f64> = output
            .column(0)
            .mapv(|p| p * self.target_std + self.target_mean);

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.mlp.is_fitted()
    }

    fn n_layers(&self) -> usize {
        self.mlp.layers.len() + 1 // +1 for input layer
    }

    fn layer_sizes(&self) -> Vec<usize> {
        self.mlp.layer_sizes()
    }
}

impl NeuralDiagnostics for MLPRegressor {
    fn training_diagnostics(&self) -> Option<&TrainingDiagnostics> {
        self.diagnostics.as_ref()
    }

    fn weight_statistics(&self) -> Result<Vec<WeightStatistics>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("predict"));
        }
        Ok(crate::neural::weight_statistics(&self.mlp.layers))
    }

    fn dead_neurons(&self, x: &Array2<f64>) -> Result<Vec<(usize, usize)>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("predict"));
        }
        let mut layers = self.mlp.layers.clone();
        let dead = crate::neural::dead_neuron_detection(&mut layers, x)?;
        Ok(dead
            .into_iter()
            .map(|d| (d.layer_idx, d.neuron_idx))
            .collect())
    }
}

impl NeuralUncertainty for MLPRegressor {
    fn predict_with_uncertainty(
        &self,
        x: &Array2<f64>,
        n_samples: usize,
        confidence: f64,
    ) -> Result<PredictionUncertainty> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("predict"));
        }

        let dropout_rate = self.mlp.regularization.dropout_rate.max(0.1);
        let mut layers = self.mlp.layers.clone();

        let mut result = crate::neural::predict_with_uncertainty(
            &mut layers,
            x,
            n_samples,
            dropout_rate,
            confidence,
            self.mlp.random_state,
        )?;

        // Denormalize the predictions
        result.mean = result.mean.mapv(|p| p * self.target_std + self.target_mean);
        result.std = result.std.mapv(|s| s * self.target_std);
        result.lower = result
            .lower
            .mapv(|l| l * self.target_std + self.target_mean);
        result.upper = result
            .upper
            .mapv(|u| u * self.target_std + self.target_mean);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_linear_data() -> (Array2<f64>, Array1<f64>) {
        // y = 2x + 1
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| 2.0 * i as f64 + 1.0).collect());
        (x, y)
    }

    #[test]
    fn test_mlp_regressor_creation() {
        let mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10, 5])
            .activation(Activation::ReLU)
            .solver(Solver::Adam);

        assert!(!mlp.is_fitted());
    }

    #[test]
    fn test_mlp_regressor_fit() {
        let (x, y) = create_linear_data();

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .activation(Activation::ReLU)
            .learning_rate(0.01)
            .max_iter(500)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        assert!(mlp.is_fitted());
    }

    #[test]
    fn test_mlp_regressor_predict() {
        let (x, y) = create_linear_data();

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[20, 10])
            .activation(Activation::ReLU)
            .learning_rate(0.01)
            .max_iter(1000)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let predictions = mlp.predict(&x).unwrap();

        assert_eq!(predictions.len(), 10);

        // Predictions should be in reasonable range
        for (pred, target) in predictions.iter().zip(y.iter()) {
            let error = (pred - target).abs();
            // Allow reasonable error (within 5 of target)
            assert!(
                error < 5.0,
                "Prediction {} too far from target {}",
                pred,
                target
            );
        }
    }

    #[test]
    fn test_mlp_regressor_score() {
        let (x, y) = create_linear_data();

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[20, 10])
            .activation(Activation::ReLU)
            .learning_rate(0.01)
            .max_iter(1000)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let r2 = mlp.score(&x, &y).unwrap();

        // Should have reasonable R² on training data
        assert!(r2 > 0.5, "R² = {} should be > 0.5", r2);
    }

    #[test]
    fn test_diagnostics() {
        let (x, y) = create_linear_data();

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[5])
            .max_iter(50)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let diag = mlp.training_diagnostics().unwrap();
        assert!(!diag.loss_curve.is_empty());
    }

    #[test]
    fn test_weight_statistics() {
        let (x, y) = create_linear_data();

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[5])
            .max_iter(10)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let stats = mlp.weight_statistics().unwrap();
        assert_eq!(stats.len(), 2); // 2 layers
    }

    #[test]
    fn test_early_stopping() {
        let (x, y) = create_linear_data();

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .max_iter(1000)
            .early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1e-4,
                validation_fraction: 0.2,
            })
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        // Should have stopped before max_iter
        let diag = mlp.training_diagnostics().unwrap();
        assert!(diag.n_iter < 1000);
    }

    #[test]
    fn test_uncertainty_quantification() {
        let (x, y) = create_linear_data();

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .max_iter(100)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let uncertainty = mlp.predict_with_uncertainty(&x, 20, 0.95).unwrap();

        assert_eq!(uncertainty.mean.len(), 10);
        assert_eq!(uncertainty.std.len(), 10);
        assert!(uncertainty.std.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn test_target_normalization() {
        // Test with large target values
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1000.0, 2000.0, 3000.0, 4000.0, 5000.0]);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[20, 10])
            .learning_rate(0.01)
            .max_iter(500)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let predictions = mlp.predict(&x).unwrap();

        // Predictions should be denormalized to approximately the right scale
        let pred_mean = predictions.sum() / predictions.len() as f64;
        let target_mean = y.sum() / y.len() as f64;

        // Mean should be in same order of magnitude (within 10x)
        assert!(
            pred_mean > target_mean / 10.0 && pred_mean < target_mean * 10.0,
            "Prediction mean {} should be within 10x of target mean {}",
            pred_mean,
            target_mean
        );
    }
}
