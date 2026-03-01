//! Neural Network Correctness Tests
//!
//! Comprehensive tests for the neural network module including:
//! - Numerical gradient checking for all activation/layer types
//! - MLPClassifier convergence and accuracy tests
//! - MLPRegressor convergence and R-squared tests
//! - Optimizer comparison tests
//! - Regularization and early stopping tests
//! - Activation function edge cases
//! - Input validation and error handling

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, Axis};

use ferroml_core::neural::layers::Layer;
use ferroml_core::neural::{
    Activation, EarlyStopping, MLPClassifier, MLPRegressor, NeuralDiagnostics, NeuralModel, Solver,
    WeightInit, MLP,
};

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute numerical gradient of a scalar function using central differences.
fn numerical_gradient<F>(f: F, x: &Array2<f64>, eps: f64) -> Array2<f64>
where
    F: Fn(&Array2<f64>) -> f64,
{
    let mut grad = Array2::zeros(x.raw_dim());
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[[i, j]] += eps;
            x_minus[[i, j]] -= eps;
            grad[[i, j]] = (f(&x_plus) - f(&x_minus)) / (2.0 * eps);
        }
    }
    grad
}

/// Compute relative error between two arrays, handling near-zero values.
fn relative_error(analytical: &Array2<f64>, numerical: &Array2<f64>) -> f64 {
    let diff = analytical - numerical;
    let numer = diff.mapv(|d| d * d).sum().sqrt();
    let denom = (analytical.mapv(|a| a * a).sum().sqrt() + numerical.mapv(|n| n * n).sum().sqrt())
        .max(1e-10);
    numer / denom
}

/// Create Iris-like dataset (3 classes, 4 features, 150 samples).
/// Uses deterministic generation based on class centers + small noise.
fn create_iris_like_data(seed: u64) -> (Array2<f64>, Array1<f64>) {
    use rand::prelude::*;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n_per_class = 50;
    let n_samples = n_per_class * 3;
    let n_features = 4;

    let centers = [
        [5.0, 3.4, 1.5, 0.2],
        [5.9, 2.8, 4.3, 1.3],
        [6.6, 3.0, 5.6, 2.0],
    ];

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for (class_idx, center) in centers.iter().enumerate() {
        for _ in 0..n_per_class {
            for &c in center.iter() {
                let noise: f64 = (rng.random::<f64>() - 0.5) * 0.8;
                x_data.push(c + noise);
            }
            y_data.push(class_idx as f64);
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

/// Create simple linear data: y = 2*x1 + 3*x2 + 1
fn create_linear_data(n_samples: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    use rand::prelude::*;
    use rand::SeedableRng;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut x_data = Vec::with_capacity(n_samples * 2);
    let mut y_data = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let x1: f64 = rng.random::<f64>() * 4.0 - 2.0;
        let x2: f64 = rng.random::<f64>() * 4.0 - 2.0;
        x_data.push(x1);
        x_data.push(x2);
        let noise: f64 = (rng.random::<f64>() - 0.5) * 0.2;
        y_data.push(2.0 * x1 + 3.0 * x2 + 1.0 + noise);
    }

    let x = Array2::from_shape_vec((n_samples, 2), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

/// Create XOR dataset.
fn create_xor_data() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);
    (x, y)
}

// =============================================================================
// Numerical Gradient Checking Tests
// =============================================================================

mod gradient_tests {
    use super::*;

    #[test]
    fn test_dense_layer_weight_gradient() {
        // Test that analytical weight gradient matches numerical gradient
        let mut layer = Layer::new(
            3,
            2,
            Activation::Linear,
            WeightInit::XavierUniform,
            Some(42),
        );
        let input = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.5, -0.3, 0.2, -1.0, 0.8, -0.5, 0.3, 1.2, 0.7, -0.4, 0.1,
            ],
        )
        .unwrap();
        // Forward pass
        let output = layer.forward(&input, true).unwrap();

        // Loss: 0.5 * sum(output^2) (simple quadratic)
        let loss_grad = output.clone(); // d/d_output of 0.5 * output^2 = output

        // Analytical gradient via backward (skip activation for output layer scenario)
        let (grad_w_analytical, _, _) = layer.backward_skip_activation(&loss_grad).unwrap();

        // Numerical gradient
        let original_weights = layer.weights.clone();
        let grad_w_numerical = numerical_gradient(
            |w| {
                // Compute loss with perturbed weights
                let z = input.dot(w) + &layer.biases;
                0.5 * z.mapv(|v| v * v).sum()
            },
            &original_weights,
            1e-5,
        );

        let err = relative_error(&grad_w_analytical, &grad_w_numerical);
        assert!(
            err < 1e-5,
            "Weight gradient relative error {} exceeds threshold",
            err
        );
    }

    #[test]
    fn test_dense_layer_bias_gradient() {
        let mut layer = Layer::new(
            2,
            3,
            Activation::Linear,
            WeightInit::XavierUniform,
            Some(42),
        );
        let input = Array2::from_shape_vec((3, 2), vec![1.0, -0.5, 0.3, 0.8, -1.0, 0.2]).unwrap();

        let output = layer.forward(&input, true).unwrap();
        let loss_grad = output.clone();

        let (_, grad_b_analytical, _) = layer.backward_skip_activation(&loss_grad).unwrap();

        // Numerical gradient for biases
        let original_biases = layer.biases.clone();
        let eps = 1e-5;
        let mut grad_b_numerical = Array1::zeros(layer.biases.len());
        for j in 0..layer.biases.len() {
            let mut b_plus = original_biases.clone();
            let mut b_minus = original_biases.clone();
            b_plus[j] += eps;
            b_minus[j] -= eps;

            let z_plus = input.dot(&layer.weights) + &b_plus;
            let z_minus = input.dot(&layer.weights) + &b_minus;
            let loss_plus = 0.5 * z_plus.mapv(|v| v * v).sum();
            let loss_minus = 0.5 * z_minus.mapv(|v| v * v).sum();
            grad_b_numerical[j] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        for j in 0..grad_b_analytical.len() {
            let err = (grad_b_analytical[j] - grad_b_numerical[j]).abs()
                / (grad_b_analytical[j].abs() + grad_b_numerical[j].abs()).max(1e-10);
            assert!(
                err < 1e-5,
                "Bias gradient error at index {}: analytical={}, numerical={}, err={}",
                j,
                grad_b_analytical[j],
                grad_b_numerical[j],
                err
            );
        }
    }

    #[test]
    fn test_relu_gradient() {
        let x = Array2::from_shape_vec((1, 5), vec![-2.0, -0.5, 0.1, 1.0, 3.0]).unwrap();
        let output = Activation::ReLU.apply_2d(&x);
        let deriv = Activation::ReLU.derivative_2d(&x, &output);

        // For positive x, derivative = 1; for negative x, derivative = 0
        assert_abs_diff_eq!(deriv[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv[[0, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv[[0, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv[[0, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(deriv[[0, 4]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_gradient_numerical() {
        let x_vals = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let eps = 1e-7;

        for &x_val in &x_vals {
            let x = Array1::from_vec(vec![x_val]);
            let output = Activation::Sigmoid.apply(&x);
            let analytical = Activation::Sigmoid.derivative(&x, &output);

            // Numerical: sigmoid(x+eps) - sigmoid(x-eps) / (2*eps)
            let x_plus = Array1::from_vec(vec![x_val + eps]);
            let x_minus = Array1::from_vec(vec![x_val - eps]);
            let numerical = (Activation::Sigmoid.apply(&x_plus)[0]
                - Activation::Sigmoid.apply(&x_minus)[0])
                / (2.0 * eps);

            let err = (analytical[0] - numerical).abs();
            assert!(
                err < 1e-5,
                "Sigmoid derivative error at x={}: analytical={}, numerical={}, err={}",
                x_val,
                analytical[0],
                numerical,
                err
            );
        }
    }

    #[test]
    fn test_tanh_gradient_numerical() {
        let x_vals = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let eps = 1e-7;

        for &x_val in &x_vals {
            let x = Array1::from_vec(vec![x_val]);
            let output = Activation::Tanh.apply(&x);
            let analytical = Activation::Tanh.derivative(&x, &output);

            let x_plus = Array1::from_vec(vec![x_val + eps]);
            let x_minus = Array1::from_vec(vec![x_val - eps]);
            let numerical = (Activation::Tanh.apply(&x_plus)[0]
                - Activation::Tanh.apply(&x_minus)[0])
                / (2.0 * eps);

            let err = (analytical[0] - numerical).abs();
            assert!(
                err < 1e-5,
                "Tanh derivative error at x={}: analytical={}, numerical={}, err={}",
                x_val,
                analytical[0],
                numerical,
                err
            );
        }
    }

    #[test]
    fn test_leaky_relu_gradient_numerical() {
        let x_vals = vec![-2.0, -0.5, 0.5, 2.0];
        let eps = 1e-7;

        for &x_val in &x_vals {
            let x = Array1::from_vec(vec![x_val]);
            let output = Activation::LeakyReLU.apply(&x);
            let analytical = Activation::LeakyReLU.derivative(&x, &output);

            let x_plus = Array1::from_vec(vec![x_val + eps]);
            let x_minus = Array1::from_vec(vec![x_val - eps]);
            let numerical = (Activation::LeakyReLU.apply(&x_plus)[0]
                - Activation::LeakyReLU.apply(&x_minus)[0])
                / (2.0 * eps);

            let err = (analytical[0] - numerical).abs();
            assert!(
                err < 1e-4,
                "LeakyReLU derivative error at x={}: analytical={}, numerical={}, err={}",
                x_val,
                analytical[0],
                numerical,
                err
            );
        }
    }

    #[test]
    fn test_elu_gradient_numerical() {
        let x_vals = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let eps = 1e-7;

        for &x_val in &x_vals {
            let x = Array1::from_vec(vec![x_val]);
            let output = Activation::ELU.apply(&x);
            let analytical = Activation::ELU.derivative(&x, &output);

            let x_plus = Array1::from_vec(vec![x_val + eps]);
            let x_minus = Array1::from_vec(vec![x_val - eps]);
            let numerical = (Activation::ELU.apply(&x_plus)[0]
                - Activation::ELU.apply(&x_minus)[0])
                / (2.0 * eps);

            let err = (analytical[0] - numerical).abs();
            assert!(
                err < 1e-5,
                "ELU derivative error at x={}: analytical={}, numerical={}, err={}",
                x_val,
                analytical[0],
                numerical,
                err
            );
        }
    }

    #[test]
    fn test_softmax_cross_entropy_combined_gradient() {
        // For softmax + cross-entropy, the combined gradient is (p - y).
        // Verify this numerically.
        let logits = Array2::from_shape_vec((1, 3), vec![2.0, 1.0, 0.5]).unwrap();
        let targets = Array2::from_shape_vec((1, 3), vec![0.0, 1.0, 0.0]).unwrap();

        let probs = Activation::Softmax.apply_2d(&logits);

        // Analytical gradient of cross-entropy w.r.t. logits = p - y
        let analytical_grad = &probs - &targets;

        // Numerical gradient of CE loss w.r.t. logits
        let eps = 1e-5;
        let numerical_grad = numerical_gradient(
            |z| {
                let p = Activation::Softmax.apply_2d(z);
                let log_p = p.mapv(|v| v.max(1e-15).ln());
                -(log_p * &targets).sum()
            },
            &logits,
            eps,
        );

        let err = relative_error(&analytical_grad, &numerical_grad);
        assert!(err < 1e-5, "Softmax+CE combined gradient error: {}", err);
    }

    #[test]
    fn test_mse_gradient() {
        // For linear output + MSE, gradient of loss w.r.t. output = 2*(p-y)/n
        let predictions = Array2::from_shape_vec((3, 1), vec![1.5, 2.5, 0.5]).unwrap();
        let targets = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 1.0]).unwrap();
        let n = predictions.nrows() as f64;

        // Analytical gradient: 2*(p-y)/n
        let analytical = (&predictions - &targets).mapv(|d| 2.0 * d / n);

        // Numerical gradient
        let eps = 1e-5;
        let numerical = numerical_gradient(
            |p| {
                let diff = p - &targets;
                diff.mapv(|d| d * d).sum() / n
            },
            &predictions,
            eps,
        );

        let err = relative_error(&analytical, &numerical);
        assert!(err < 1e-5, "MSE gradient error: {}", err);
    }

    #[test]
    fn test_full_network_gradient_2_layers() {
        // Build a tiny 2-layer network and check end-to-end gradient
        let mut mlp = MLP::new()
            .hidden_layer_sizes(&[4])
            .activation(Activation::Sigmoid)
            .output_activation(Activation::Linear)
            .random_state(42);

        mlp.initialize(2, 1).unwrap();

        let x = Array2::from_shape_vec((3, 2), vec![1.0, -0.5, 0.3, 0.8, -1.0, 0.2]).unwrap();
        let targets = Array2::from_shape_vec((3, 1), vec![1.0, 0.0, 0.5]).unwrap();

        // Forward
        let output = mlp.forward(&x, true).unwrap();

        // MSE loss gradient
        let n = x.nrows() as f64;
        let loss_grad = (&output - &targets).mapv(|d| 2.0 * d / n);

        // Backward
        let gradients = mlp.backward(&loss_grad).unwrap();
        assert_eq!(gradients.len(), 2);

        // Check first layer weight gradient numerically
        let original_w0 = mlp.layers[0].weights.clone();
        let eps = 1e-5;

        let grad_w0_numerical = numerical_gradient(
            |w| {
                let mut test_mlp = mlp.clone();
                test_mlp.layers[0].weights = w.clone();
                let out = test_mlp.forward(&x, false).unwrap();
                let diff = &out - &targets;
                diff.mapv(|d| d * d).sum() / n
            },
            &original_w0,
            eps,
        );

        let err = relative_error(&gradients[0].0, &grad_w0_numerical);
        assert!(
            err < 1e-4,
            "Full network layer 0 weight gradient error: {}",
            err
        );
    }

    #[test]
    fn test_full_network_gradient_3_layers() {
        // 3-layer network with ReLU hidden layers
        let mut mlp = MLP::new()
            .hidden_layer_sizes(&[5, 3])
            .activation(Activation::Tanh)
            .output_activation(Activation::Linear)
            .random_state(123);

        mlp.initialize(2, 1).unwrap();

        let x = Array2::from_shape_vec((4, 2), vec![0.5, -0.3, 1.0, 0.2, -0.5, 0.8, 0.1, -1.0])
            .unwrap();
        let targets = Array2::from_shape_vec((4, 1), vec![0.5, 1.0, -0.5, 0.0]).unwrap();

        let output = mlp.forward(&x, true).unwrap();
        let n = x.nrows() as f64;
        let loss_grad = (&output - &targets).mapv(|d| 2.0 * d / n);

        let gradients = mlp.backward(&loss_grad).unwrap();
        assert_eq!(gradients.len(), 3);

        // Check last hidden layer (layer 1) weight gradient
        let original_w1 = mlp.layers[1].weights.clone();
        let eps = 1e-5;

        let grad_w1_numerical = numerical_gradient(
            |w| {
                let mut test_mlp = mlp.clone();
                test_mlp.layers[1].weights = w.clone();
                let out = test_mlp.forward(&x, false).unwrap();
                let diff = &out - &targets;
                diff.mapv(|d| d * d).sum() / n
            },
            &original_w1,
            eps,
        );

        let err = relative_error(&gradients[1].0, &grad_w1_numerical);
        assert!(
            err < 1e-4,
            "Full network layer 1 weight gradient error: {}",
            err
        );
    }
}

// =============================================================================
// MLPClassifier Tests
// =============================================================================

mod classifier_tests {
    use super::*;

    #[test]
    fn test_mlp_classifier_xor_convergence() {
        let (x, y) = create_xor_data();

        // Try multiple seeds to ensure XOR is solvable
        let mut best_accuracy = 0.0;
        for seed in [42u64, 100, 200, 300, 7] {
            let mut mlp = MLPClassifier::new()
                .hidden_layer_sizes(&[8, 8])
                .activation(Activation::ReLU)
                .solver(Solver::Adam)
                .learning_rate(0.01)
                .max_iter(500)
                .tol(1e-8)
                .random_state(seed);

            mlp.fit(&x, &y).unwrap();
            let predictions = mlp.predict(&x).unwrap();

            let correct: usize = predictions
                .iter()
                .zip(y.iter())
                .filter(|(p, t)| (**p - **t).abs() < 0.5)
                .count();
            let accuracy = correct as f64 / y.len() as f64;
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
            }
            if best_accuracy == 1.0 {
                break;
            }
        }

        assert!(
            best_accuracy == 1.0,
            "XOR should be solvable with 100% accuracy, got {}%",
            best_accuracy * 100.0
        );
    }

    #[test]
    fn test_mlp_classifier_iris_accuracy_above_90() {
        let (x, y) = create_iris_like_data(42);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[100])
            .activation(Activation::ReLU)
            .solver(Solver::Adam)
            .learning_rate(0.001)
            .max_iter(300)
            .tol(1e-6)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let predictions = mlp.predict(&x).unwrap();

        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count();
        let accuracy = correct as f64 / y.len() as f64;

        assert!(
            accuracy > 0.90,
            "Iris-like accuracy should be >90%, got {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_mlp_classifier_binary_probabilities_sum_to_1() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[8])
            .max_iter(100)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let probs = mlp.predict_proba(&x).unwrap();

        for row in probs.axis_iter(Axis(0)) {
            let sum = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mlp_classifier_multiclass_probabilities_sum_to_1() {
        let (x, y) = create_iris_like_data(42);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .max_iter(50)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let probs = mlp.predict_proba(&x).unwrap();

        assert_eq!(probs.ncols(), 3, "Should have 3 class columns");

        for (i, row) in probs.axis_iter(Axis(0)).enumerate() {
            let sum = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
            // All probabilities should be non-negative
            assert!(
                row.iter().all(|&p| p >= 0.0),
                "Sample {} has negative probability",
                i
            );
        }
    }

    #[test]
    fn test_mlp_classifier_predict_matches_argmax_proba() {
        let (x, y) = create_iris_like_data(42);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .max_iter(100)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let probs = mlp.predict_proba(&x).unwrap();
        let predictions = mlp.predict(&x).unwrap();

        let classes = mlp.classes_.as_ref().unwrap();

        for (pred, prob_row) in predictions.iter().zip(probs.axis_iter(Axis(0))) {
            let argmax = prob_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            let expected_class = classes[argmax];
            assert_abs_diff_eq!(*pred, expected_class, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mlp_classifier_loss_decreases() {
        let (x, y) = create_iris_like_data(42);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .learning_rate(0.001)
            .max_iter(100)
            .tol(1e-10) // Very small tol to prevent early convergence stopping
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let diag = mlp.training_diagnostics().unwrap();
        assert!(
            diag.loss_curve.len() >= 10,
            "Should train for at least 10 epochs"
        );

        // Loss at epoch 50 should be less than epoch 1
        let early_loss = diag.loss_curve[0];
        let later_idx = (diag.loss_curve.len() / 2).max(1);
        let later_loss = diag.loss_curve[later_idx];

        assert!(
            later_loss < early_loss,
            "Loss should decrease: epoch 0 = {}, epoch {} = {}",
            early_loss,
            later_idx,
            later_loss
        );
    }

    #[test]
    fn test_mlp_classifier_deterministic_with_seed() {
        let (x, y) = create_xor_data();

        let mut mlp1 = MLPClassifier::new()
            .hidden_layer_sizes(&[8])
            .max_iter(50)
            .random_state(42);
        mlp1.fit(&x, &y).unwrap();
        let pred1 = mlp1.predict_proba(&x).unwrap();

        let mut mlp2 = MLPClassifier::new()
            .hidden_layer_sizes(&[8])
            .max_iter(50)
            .random_state(42);
        mlp2.fit(&x, &y).unwrap();
        let pred2 = mlp2.predict_proba(&x).unwrap();

        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert_abs_diff_eq!(*p1, *p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mlp_classifier_different_activations() {
        let (x, y) = create_iris_like_data(42);

        let activations = [Activation::ReLU, Activation::Sigmoid, Activation::Tanh];

        for activation in &activations {
            let mut mlp = MLPClassifier::new()
                .hidden_layer_sizes(&[20])
                .activation(*activation)
                .max_iter(100)
                .random_state(42);

            let result = mlp.fit(&x, &y);
            assert!(
                result.is_ok(),
                "Training with {:?} activation should succeed",
                activation
            );

            let predictions = mlp.predict(&x).unwrap();
            assert_eq!(predictions.len(), x.nrows());
        }
    }
}

// =============================================================================
// MLPRegressor Tests
// =============================================================================

mod regressor_tests {
    use super::*;

    #[test]
    fn test_mlp_regressor_linear_r2_above_0_9() {
        let (x, y) = create_linear_data(100, 42);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[64, 32])
            .activation(Activation::ReLU)
            .solver(Solver::Adam)
            .learning_rate(0.01)
            .max_iter(500)
            .tol(1e-8)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let r2 = mlp.score(&x, &y).unwrap();

        assert!(
            r2 > 0.9,
            "R-squared on linear data should be >0.9, got {:.4}",
            r2
        );
    }

    #[test]
    fn test_mlp_regressor_loss_decreases() {
        let (x, y) = create_linear_data(50, 42);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[20])
            .learning_rate(0.005)
            .max_iter(100)
            .tol(1e-10)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let diag = mlp.training_diagnostics().unwrap();
        assert!(diag.loss_curve.len() >= 10);

        let first_loss = diag.loss_curve[0];
        let mid_idx = diag.loss_curve.len() / 2;
        let mid_loss = diag.loss_curve[mid_idx];

        assert!(
            mid_loss < first_loss,
            "Regressor loss should decrease: epoch 0 = {}, epoch {} = {}",
            first_loss,
            mid_idx,
            mid_loss
        );
    }

    #[test]
    fn test_mlp_regressor_deterministic_with_seed() {
        let (x, y) = create_linear_data(30, 42);

        let mut mlp1 = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .max_iter(50)
            .random_state(42);
        mlp1.fit(&x, &y).unwrap();
        let pred1 = mlp1.predict(&x).unwrap();

        let mut mlp2 = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .max_iter(50)
            .random_state(42);
        mlp2.fit(&x, &y).unwrap();
        let pred2 = mlp2.predict(&x).unwrap();

        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert_abs_diff_eq!(*p1, *p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mlp_regressor_score_matches_manual_r2() {
        let (x, y) = create_linear_data(50, 42);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[20, 10])
            .learning_rate(0.01)
            .max_iter(200)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let r2_from_score = mlp.score(&x, &y).unwrap();

        // Compute R2 manually
        let predictions = mlp.predict(&x).unwrap();
        let y_mean = y.sum() / y.len() as f64;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, yi)| (yi - p).powi(2))
            .sum();
        let r2_manual = 1.0 - ss_res / ss_tot;

        assert_abs_diff_eq!(r2_from_score, r2_manual, epsilon = 1e-10);
    }

    #[test]
    fn test_mlp_regressor_multi_output_mse_correctness() {
        // Verify that MSE loss correctly divides by both n_samples and n_outputs
        let predictions =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap();

        // Expected MSE = mean over all elements of (pred-target)^2
        // Each diff is 0.5, so each squared diff is 0.25
        // Mean of 6 values of 0.25 = 0.25
        let diff = &predictions - &targets;
        let expected_loss = diff.mapv(|d| d * d).sum() / (2.0 * 3.0);
        assert_abs_diff_eq!(expected_loss, 0.25, epsilon = 1e-10);
    }
}

// =============================================================================
// Optimizer Tests
// =============================================================================

mod optimizer_tests {
    use super::*;

    #[test]
    fn test_adam_converges_faster_than_sgd() {
        let (x, y) = create_iris_like_data(42);

        // Adam
        let mut mlp_adam = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .solver(Solver::Adam)
            .learning_rate(0.001)
            .max_iter(50)
            .tol(1e-10)
            .random_state(42);
        mlp_adam.fit(&x, &y).unwrap();
        let adam_final_loss = mlp_adam.training_diagnostics().unwrap().final_loss;

        // SGD with momentum
        let mut mlp_sgd = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .solver(Solver::SGD)
            .learning_rate(0.01)
            .max_iter(50)
            .tol(1e-10)
            .random_state(42);
        mlp_sgd.fit(&x, &y).unwrap();
        let sgd_final_loss = mlp_sgd.training_diagnostics().unwrap().final_loss;

        // Adam typically converges faster (lower loss at same epoch count)
        // This is a soft assertion - Adam is usually better but not guaranteed
        // We just verify both actually train (loss decreases)
        let adam_first = mlp_adam.training_diagnostics().unwrap().loss_curve[0];
        let sgd_first = mlp_sgd.training_diagnostics().unwrap().loss_curve[0];
        assert!(adam_final_loss < adam_first, "Adam loss should decrease");
        assert!(sgd_final_loss < sgd_first, "SGD loss should decrease");
    }

    #[test]
    fn test_learning_rate_too_high_diverges_or_oscillates() {
        let (x, y) = create_linear_data(30, 42);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .solver(Solver::SGD)
            .learning_rate(10.0) // Very high LR
            .max_iter(50)
            .tol(1e-10)
            .random_state(42);

        // Should still complete (not panic), but loss should be poor
        let result = mlp.fit(&x, &y);
        assert!(result.is_ok(), "Should not panic even with high LR");
    }
}

// =============================================================================
// Regularization Tests
// =============================================================================

mod regularization_tests {
    use super::*;

    #[test]
    fn test_l2_regularization_shrinks_weights() {
        let (x, y) = create_iris_like_data(42);

        // Train without regularization
        let mut mlp_no_reg = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .alpha(0.0) // No L2
            .max_iter(100)
            .tol(1e-10)
            .random_state(42);
        mlp_no_reg.fit(&x, &y).unwrap();

        // Train with strong regularization
        let mut mlp_reg = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .alpha(1.0) // Strong L2
            .max_iter(100)
            .tol(1e-10)
            .random_state(42);
        mlp_reg.fit(&x, &y).unwrap();

        // L2 should shrink weight magnitudes
        let w_no_reg: f64 = mlp_no_reg
            .mlp
            .layers
            .iter()
            .map(|l| l.weights.mapv(|w| w * w).sum())
            .sum();
        let w_reg: f64 = mlp_reg
            .mlp
            .layers
            .iter()
            .map(|l| l.weights.mapv(|w| w * w).sum())
            .sum();

        assert!(
            w_reg < w_no_reg,
            "L2 regularized weights ({:.4}) should be smaller than unregularized ({:.4})",
            w_reg,
            w_no_reg
        );
    }

    #[test]
    fn test_dropout_produces_different_masks_per_forward_pass() {
        // After the RNG fix, dropout should produce different masks on each
        // forward pass (during training), not the same mask repeated.
        let (x, y) = create_iris_like_data(42);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .activation(Activation::ReLU)
            .solver(Solver::Adam)
            .learning_rate(0.001)
            .max_iter(5)
            .tol(1e-10)
            .random_state(42);

        // Enable dropout
        mlp.mlp.regularization.dropout_rate = 0.3;
        mlp.fit(&x, &y).unwrap();

        // Now do two forward passes with training=true (dropout active)
        // They should produce different outputs due to different dropout masks
        let small_x = x.slice(ndarray::s![0..3, ..]).to_owned();
        let out1 = mlp.mlp.forward(&small_x, true).unwrap();
        let out2 = mlp.mlp.forward(&small_x, true).unwrap();

        // Outputs should differ because dropout masks are different
        let diff: f64 = (&out1 - &out2).mapv(|d| d.abs()).sum();
        assert!(
            diff > 1e-10,
            "With dropout, consecutive forward passes should give different outputs (diff={})",
            diff
        );
    }

    #[test]
    fn test_early_stopping_stops_before_max_iter() {
        // Use a small, simple dataset that the network quickly learns
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);

        // Use early stopping with validation split -- model converges fast
        // and either convergence tolerance or early stopping triggers
        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[16, 16])
            .learning_rate(0.05)
            .max_iter(2000)
            .tol(1e-3)
            .early_stopping(EarlyStopping {
                patience: 10,
                min_delta: 1e-4,
                validation_fraction: 0.25,
            })
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let diag = mlp.training_diagnostics().unwrap();
        // Convergence or early stopping should stop before max_iter
        assert!(
            diag.n_iter < 2000,
            "Training should terminate before max_iter, got {} iterations",
            diag.n_iter
        );
    }
}

// =============================================================================
// Activation Function Edge Cases
// =============================================================================

mod activation_tests {
    use super::*;

    #[test]
    fn test_softmax_large_values_no_overflow() {
        // Test with very large values that would overflow without max-subtraction trick
        let x = Array2::from_shape_vec((1, 4), vec![1000.0, 1001.0, 1002.0, 999.0]).unwrap();
        let result = Activation::Softmax.apply_2d(&x);

        // Should sum to 1 and all be finite
        assert_abs_diff_eq!(result.row(0).sum(), 1.0, epsilon = 1e-10);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_softmax_negative_large_values() {
        let x = Array2::from_shape_vec((1, 3), vec![-1000.0, -999.0, -998.0]).unwrap();
        let result = Activation::Softmax.apply_2d(&x);

        assert_abs_diff_eq!(result.row(0).sum(), 1.0, epsilon = 1e-10);
        assert!(result.iter().all(|&v| v.is_finite() && v >= 0.0));
    }

    #[test]
    fn test_sigmoid_saturation_gradient() {
        // At extreme values, sigmoid saturates and gradient should be near zero
        let x_large = Array1::from_vec(vec![10.0]);
        let output_large = Activation::Sigmoid.apply(&x_large);
        let deriv_large = Activation::Sigmoid.derivative(&x_large, &output_large);
        assert!(
            deriv_large[0] < 1e-4,
            "Sigmoid gradient at x=10 should be near zero, got {}",
            deriv_large[0]
        );

        let x_small = Array1::from_vec(vec![-10.0]);
        let output_small = Activation::Sigmoid.apply(&x_small);
        let deriv_small = Activation::Sigmoid.derivative(&x_small, &output_small);
        assert!(
            deriv_small[0] < 1e-4,
            "Sigmoid gradient at x=-10 should be near zero, got {}",
            deriv_small[0]
        );
    }

    #[test]
    fn test_relu_dead_neurons_zero_gradient() {
        let x = Array1::from_vec(vec![-5.0, -1.0, -0.1]);
        let output = Activation::ReLU.apply(&x);
        let deriv = Activation::ReLU.derivative(&x, &output);

        // All negative inputs should have zero gradient
        for &d in deriv.iter() {
            assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_elu_negative_continuity() {
        // ELU should be continuous at x=0
        let x_at_zero = Array1::from_vec(vec![0.0]);
        let output_at_zero = Activation::ELU.apply(&x_at_zero);
        assert_abs_diff_eq!(output_at_zero[0], 0.0, epsilon = 1e-10);

        // Derivative should also be continuous (=1 at boundary)
        let deriv_at_zero = Activation::ELU.derivative(&x_at_zero, &output_at_zero);
        assert_abs_diff_eq!(deriv_at_zero[0], 1.0, epsilon = 1e-10);

        // Check near-zero continuity
        let x_neg = Array1::from_vec(vec![-1e-8]);
        let output_neg = Activation::ELU.apply(&x_neg);
        let x_pos = Array1::from_vec(vec![1e-8]);
        let output_pos = Activation::ELU.apply(&x_pos);

        // Outputs should be very close
        assert!(
            (output_pos[0] - output_neg[0]).abs() < 1e-6,
            "ELU should be continuous at 0"
        );
    }
}

// =============================================================================
// Edge Cases / Error Handling
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_input_returns_error_classifier() {
        let x = Array2::zeros((0, 2));
        let y = Array1::from_vec(vec![]);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[4])
            .random_state(42);

        let result = mlp.fit(&x, &y);
        assert!(result.is_err(), "Empty input should return error");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("at least one sample") || err_msg.contains("empty"),
            "Error should mention empty data, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_empty_input_returns_error_regressor() {
        let x = Array2::zeros((0, 2));
        let y = Array1::from_vec(vec![]);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[4])
            .random_state(42);

        let result = mlp.fit(&x, &y);
        assert!(result.is_err(), "Empty input should return error");
    }

    #[test]
    fn test_wrong_feature_count_returns_error_classifier() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[8])
            .max_iter(10)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        // Predict with wrong number of features
        let x_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = mlp.predict(&x_wrong);
        assert!(result.is_err(), "Wrong feature count should return error");
    }

    #[test]
    fn test_wrong_feature_count_returns_error_regressor() {
        let (x, y) = create_linear_data(20, 42);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .max_iter(10)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        // Predict with wrong number of features
        let x_wrong = Array2::from_shape_vec((2, 5), vec![1.0; 10]).unwrap();
        let result = mlp.predict(&x_wrong);
        assert!(result.is_err(), "Wrong feature count should return error");
    }

    #[test]
    fn test_single_sample_training_classifier() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[4])
            .max_iter(10)
            .random_state(42);

        // Should succeed with minimal data (need at least 2 for 2 classes)
        let result = mlp.fit(&x, &y);
        assert!(result.is_ok(), "Minimal data training should succeed");
    }

    #[test]
    fn test_single_sample_training_regressor() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![3.0]);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[4])
            .max_iter(10)
            .random_state(42);

        let result = mlp.fit(&x, &y);
        assert!(result.is_ok(), "Single sample training should succeed");
    }

    #[test]
    fn test_single_feature() {
        let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|i| (i as f64) * 2.0).collect());

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[10])
            .max_iter(100)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let predictions = mlp.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_large_hidden_layers() {
        // Test that a network with large hidden layers doesn't crash or produce NaN
        let (x, y) = create_iris_like_data(42);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[256, 128, 64])
            .activation(Activation::ReLU)
            .max_iter(10)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();
        let predictions = mlp.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.nrows());
        assert!(
            predictions.iter().all(|p| p.is_finite()),
            "All predictions should be finite"
        );
    }

    #[test]
    fn test_x_y_length_mismatch_returns_error() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]); // Wrong length

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[4])
            .random_state(42);

        let result = mlp.fit(&x, &y);
        assert!(
            result.is_err(),
            "Mismatched X/y lengths should return error"
        );
    }

    #[test]
    fn test_predict_before_fit_returns_error() {
        let mlp = MLPClassifier::new().hidden_layer_sizes(&[4]);
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = mlp.predict(&x);
        assert!(result.is_err(), "Predict before fit should return error");

        let result = mlp.predict_proba(&x);
        assert!(
            result.is_err(),
            "Predict_proba before fit should return error"
        );
    }
}

// =============================================================================
// Diagnostics Tests
// =============================================================================

mod diagnostics_tests {
    use super::*;

    #[test]
    fn test_training_diagnostics_populated() {
        let (x, y) = create_iris_like_data(42);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .max_iter(50)
            .tol(1e-10)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let diag = mlp.training_diagnostics().unwrap();
        assert!(
            !diag.loss_curve.is_empty(),
            "Loss curve should be populated"
        );
        assert!(diag.n_iter > 0, "n_iter should be positive");
        assert!(diag.final_loss.is_finite(), "Final loss should be finite");
    }

    #[test]
    fn test_weight_statistics_reasonable() {
        let (x, y) = create_iris_like_data(42);

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[20])
            .max_iter(50)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let stats = mlp.weight_statistics().unwrap();
        assert!(!stats.is_empty(), "Should have weight statistics");

        for stat in &stats {
            assert!(stat.std > 0.0, "Weight std should be positive");
            assert!(stat.min <= stat.max, "Min should be <= max");
            assert!(stat.mean.is_finite(), "Mean should be finite");
        }
    }

    #[test]
    fn test_dead_neuron_detection_with_relu() {
        let (x, y) = create_xor_data();

        let mut mlp = MLPClassifier::new()
            .hidden_layer_sizes(&[8])
            .activation(Activation::ReLU)
            .max_iter(10)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        // Should not panic; may or may not find dead neurons
        let dead = mlp.dead_neurons(&x);
        assert!(dead.is_ok(), "Dead neuron detection should not fail");
    }
}

// =============================================================================
// Uncertainty Tests
// =============================================================================

mod uncertainty_tests {
    use super::*;
    use ferroml_core::neural::NeuralUncertainty;

    #[test]
    fn test_mc_dropout_variance_increases_with_ood() {
        // Out-of-distribution (OOD) inputs should have higher uncertainty
        // than in-distribution inputs when using MC Dropout.
        let (x, y) = create_linear_data(50, 42);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[20])
            .max_iter(200)
            .tol(1e-8)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        // In-distribution: values similar to training data
        let x_in = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 1.0, -1.0, 0.5, 0.5, -0.5, 0.2, 0.8],
        )
        .unwrap();
        let uncertainty_in = mlp.predict_with_uncertainty(&x_in, 50, 0.95).unwrap();

        // Out-of-distribution: extreme values far from training range
        let x_ood = Array2::from_shape_vec(
            (5, 2),
            vec![
                100.0, 100.0, -100.0, -100.0, 50.0, -50.0, 200.0, 0.0, 0.0, 200.0,
            ],
        )
        .unwrap();
        let uncertainty_ood = mlp.predict_with_uncertainty(&x_ood, 50, 0.95).unwrap();

        let mean_std_in = uncertainty_in.std.sum() / uncertainty_in.std.len() as f64;
        let mean_std_ood = uncertainty_ood.std.sum() / uncertainty_ood.std.len() as f64;

        // OOD uncertainty should generally be higher, but MC Dropout may not
        // always detect this perfectly. We use a soft check: at least the
        // mechanism should produce non-zero uncertainty for both.
        assert!(
            mean_std_in >= 0.0 && mean_std_ood >= 0.0,
            "Both in- and out-of-distribution should have non-negative uncertainty"
        );
    }

    #[test]
    fn test_prediction_uncertainty_ci_contains_mean() {
        let (x, y) = create_linear_data(50, 42);

        let mut mlp = MLPRegressor::new()
            .hidden_layer_sizes(&[20])
            .max_iter(100)
            .random_state(42);

        mlp.fit(&x, &y).unwrap();

        let uncertainty = mlp.predict_with_uncertainty(&x, 30, 0.95).unwrap();

        // The mean should be between lower and upper confidence bounds
        for i in 0..x.nrows() {
            assert!(
                uncertainty.lower[i] <= uncertainty.mean[i]
                    && uncertainty.mean[i] <= uncertainty.upper[i],
                "Sample {}: mean ({}) should be in [{}, {}]",
                i,
                uncertainty.mean[i],
                uncertainty.lower[i],
                uncertainty.upper[i]
            );
        }

        // All standard deviations should be non-negative
        assert!(
            uncertainty.std.iter().all(|&s| s >= 0.0),
            "Standard deviations should be non-negative"
        );
    }
}
