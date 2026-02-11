//! Multi-Layer Perceptron Classifier
//!
//! This module provides `MLPClassifier` for classification tasks.
//!
//! ## Features
//!
//! - sklearn-compatible API: fit(), predict(), predict_proba()
//! - Cross-entropy loss with softmax output
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

/// Multi-Layer Perceptron Classifier
///
/// A neural network for classification with sklearn-compatible API.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::neural::{MLPClassifier, Activation, Solver};
/// use ndarray::{Array1, Array2};
///
/// let X = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0
/// ]).unwrap();
/// let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]); // XOR
///
/// let mut mlp = MLPClassifier::new()
///     .hidden_layer_sizes(&[4, 4])
///     .activation(Activation::ReLU)
///     .solver(Solver::Adam)
///     .max_iter(1000);
///
/// mlp.fit(&X, &y).unwrap();
/// let predictions = mlp.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPClassifier {
    /// Core MLP network
    pub mlp: MLP,
    /// Number of classes
    pub n_classes: Option<usize>,
    /// Class labels (for multiclass)
    pub classes_: Option<Vec<f64>>,
    /// Training diagnostics
    #[serde(skip)]
    diagnostics: Option<TrainingDiagnostics>,
}

impl Default for MLPClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl MLPClassifier {
    /// Create a new MLPClassifier with default parameters
    pub fn new() -> Self {
        Self {
            mlp: MLP::new().output_activation(Activation::Softmax),
            n_classes: None,
            classes_: None,
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

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("predict"));
        }

        // Clone to allow mutable access for forward pass
        let mut mlp = self.mlp.clone();
        mlp.forward(x, false)
    }

    /// Convert labels to one-hot encoding
    fn to_one_hot(&self, y: &Array1<f64>) -> Array2<f64> {
        let n_samples = y.len();
        let n_classes = self.n_classes.unwrap_or(2);
        let classes = self.classes_.as_ref().unwrap();

        let mut one_hot = Array2::zeros((n_samples, n_classes));

        for (i, &label) in y.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| (c - label).abs() < 1e-10) {
                one_hot[[i, class_idx]] = 1.0;
            }
        }

        one_hot
    }

    /// Convert one-hot or probabilities to class labels
    fn to_labels(&self, probs: &Array2<f64>) -> Array1<f64> {
        let classes = self.classes_.as_ref().unwrap();

        probs
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                classes[max_idx]
            })
            .collect()
    }

    /// Cross-entropy loss function
    fn cross_entropy_loss(predictions: &Array2<f64>, targets: &Array2<f64>) -> (f64, Array2<f64>) {
        let n_samples = predictions.nrows() as f64;
        let eps = 1e-15;

        // Clip predictions to avoid log(0)
        let clipped = predictions.mapv(|p| p.max(eps).min(1.0 - eps));

        // Loss: -sum(y * log(p)) / n
        let loss = -(&clipped.mapv(|p| p.ln()) * targets).sum() / n_samples;

        // Gradient for softmax + cross-entropy simplifies to: (p - y) / n
        let grad = (&clipped - targets) / n_samples;

        (loss, grad)
    }
}

impl NeuralModel for MLPClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.ncols();

        // Determine classes
        let mut classes: Vec<f64> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        self.n_classes = Some(classes.len());
        self.classes_ = Some(classes);

        let n_outputs = self.n_classes.unwrap();

        // Handle binary classification
        let output_activation = if n_outputs == 2 {
            // Still use softmax with 2 outputs for simplicity
            Activation::Softmax
        } else {
            Activation::Softmax
        };
        self.mlp.output_activation = output_activation;

        // Initialize network
        self.mlp.initialize(n_features, n_outputs)?;

        // Convert labels to one-hot
        let y_one_hot = self.to_one_hot(y);

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
                y_one_hot.select(Axis(0), &train_idx),
                Some(x.select(Axis(0), &val_idx)),
                Some(y_one_hot.select(Axis(0), &val_idx)),
            )
        } else {
            (x.clone(), y_one_hot, None, None)
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
            let loss = self
                .mlp
                .train_epoch(&x_train, &y_train, Self::cross_entropy_loss)?;

            // Validation loss
            let val_loss = if let (Some(ref xv), Some(ref yv)) = (&x_val, &y_val) {
                let mut mlp_clone = self.mlp.clone();
                let preds = mlp_clone.forward(xv, false)?;
                let (vl, _) = Self::cross_entropy_loss(&preds, yv);
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
        let probs = self.predict_proba(x)?;
        Ok(self.to_labels(&probs))
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

impl NeuralDiagnostics for MLPClassifier {
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

impl NeuralUncertainty for MLPClassifier {
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

        crate::neural::predict_with_uncertainty(
            &mut layers,
            x,
            n_samples,
            dropout_rate,
            confidence,
            self.mlp.random_state,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn create_xor_data() -> (Array2<f64>, Array1<f64>) {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
        (x, y)
    }

    #[test]
    fn test_mlp_classifier_creation() {
        let mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[10, 5])
            .activation(Activation::ReLU)
            .solver(Solver::Adam);

        assert!(!mlp.is_fitted());
    }

    #[test]
    fn test_mlp_classifier_fit() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[8, 4])
            .activation(Activation::ReLU)
            .learning_rate(0.1)
            .max_iter(500)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        assert!(mlp.is_fitted());
        assert_eq!(mlp.n_classes, Some(2));
    }

    #[test]
    fn test_mlp_classifier_predict() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[8, 4])
            .activation(Activation::ReLU)
            .learning_rate(0.1)
            .max_iter(1000)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let predictions = mlp.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_mlp_classifier_predict_proba() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[4])
            .max_iter(100)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let probs = mlp.predict_proba(&x).unwrap();

        assert_eq!(probs.shape(), &[4, 2]);
        // Each row should sum to 1 (softmax output)
        for row in probs.axis_iter(Axis(0)) {
            assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_diagnostics() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[4])
            .max_iter(50)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let diag = mlp.training_diagnostics().unwrap();
        assert!(!diag.loss_curve.is_empty());
    }

    #[test]
    fn test_weight_statistics() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[4])
            .max_iter(10)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let stats = mlp.weight_statistics().unwrap();
        assert_eq!(stats.len(), 2); // 2 layers
    }

    #[test]
    fn test_multiclass() {
        // 3-class problem
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5, 2.0, 2.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 0.0, 2.0]);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[4])
            .max_iter(100)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        assert_eq!(mlp.n_classes, Some(3));

        let probs = mlp.predict_proba(&x).unwrap();
        assert_eq!(probs.shape(), &[6, 3]);
    }

    #[test]
    fn test_early_stopping() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[4])
            .max_iter(1000)
            .early_stopping(EarlyStopping {
                patience: 5,
                min_delta: 1e-4,
                validation_fraction: 0.25,
            })
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        // Should have stopped before max_iter
        let diag = mlp.training_diagnostics().unwrap();
        assert!(diag.n_iter < 1000);
    }
}
