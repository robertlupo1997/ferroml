# Plan B: Neural Network Module Hardening

**Date:** 2026-02-25
**Priority:** CRITICAL (3 critical bugs in gradient computation)
**Module:** `ferroml-core/src/neural/` (4,374 lines, 65 inline tests)
**Estimated New Tests:** ~40
**Parallel-Safe:** Yes (no overlap with clustering/benchmark/preprocessing plans)

## Overview

The neural network module (MLPClassifier, MLPRegressor) has **3 critical gradient computation bugs** that are currently masked by implementation shortcuts. The softmax derivative is wrong, the backprop applies it anyway (double-compounding the error), and multi-output MSE is broken. This plan fixes bugs first, adds numerical gradient checking, then sklearn comparison tests.

## Current State

### What Exists
- **MLPClassifier** (`classifier.rs:551`): Binary + multiclass classification with cross-entropy
- **MLPRegressor** (`regressor.rs:571`): Regression with MSE loss
- **MLP Core** (`mlp.rs:573`): Forward/backward pass, batching, training loop
- **Layers** (`layers.rs:473`): Dense layer with dropout, weight init, caching
- **Activations** (`activations.rs:256`): ReLU, Sigmoid, Tanh, Softmax, Linear, LeakyReLU, ELU
- **Optimizers** (`optimizers.rs:425`): SGD with momentum, Adam with bias correction
- **Diagnostics** (`diagnostics.rs:508`): Loss curves, convergence, gradient flow analysis
- **Analysis** (`analysis.rs:390`): Weight stats, dead neuron detection
- **Uncertainty** (`uncertainty.rs:354`): MC Dropout, calibration analysis
- 65 inline tests (all passing) — but none verify gradient correctness

### Critical Bugs

| # | Location | Bug | Severity | Why It's Masked |
|---|----------|-----|----------|-----------------|
| 1 | `activations.rs:94-97` | Softmax derivative returns `s*(1-s)` (sigmoid derivative) instead of softmax Jacobian | **CRITICAL** | Classifier bypasses activation derivative for output layer |
| 2 | `classifier.rs:188-199` + backprop | Classifier computes `grad = (p-y)/n` then backward() applies activation derivative on top — double application | **CRITICAL** | Bug #1's wrong derivative partially cancels, making results "close enough" |
| 3 | `regressor.rs:143-154` | MSE loss divides by `n_samples` but not `n_outputs` — wrong for multi-output | **CRITICAL** | Currently `n_outputs=1` always, so division is correct by accident |

### High/Medium Bugs

| # | Location | Bug | Severity |
|---|----------|-----|----------|
| 4 | `classifier.rs:191` | Clips predictions to `[1e-15, 1-1e-15]` altering actual network outputs | HIGH |
| 5 | `activations.rs:101-107` | ELU derivative hardcodes alpha=1, uses `>` instead of `>=` at boundary | MEDIUM |
| 6 | `layers.rs:194` / `mlp.rs:292` | RNG reseeded from OS every forward pass if `random_state=None` | MEDIUM |
| 7 | `mlp.rs:439` | Missing `n_samples > 0` validation — panics on empty input | MEDIUM |

## Desired End State

- All critical bugs fixed with regression tests
- Numerical gradient checking validates all layer types
- 40+ correctness tests including sklearn comparisons
- XOR convergence guaranteed, Iris accuracy >90%
- Edge cases handled gracefully (empty data, wrong dimensions)

## Implementation Phases

### Phase B.1: Fix Critical Gradient Bugs (3 changes)

#### B.1.1: Fix Softmax Output Layer Backprop
**Files:** `classifier.rs`, `mlp.rs`

The root issue: for softmax + cross-entropy, the combined gradient is `(p - y) / n`. The classifier correctly computes this. But then `backward()` in mlp.rs applies the activation derivative on top, which is wrong for the output layer.

**Two valid approaches:**

**Option A (Recommended): Skip activation derivative for output layer**
In `mlp.rs` backward pass, detect that the loss gradient already includes the activation:
```rust
// For output layer with softmax+CE or linear+MSE,
// loss_grad already accounts for activation — don't apply derivative
let delta = if layer_idx == self.layers.len() - 1 {
    grad_output.clone()  // Loss gradient is final
} else {
    let activation_deriv = self.layers[layer_idx].activation.derivative_2d(z, output);
    &grad_output * &activation_deriv
};
```

**Option B: Fix softmax derivative to return identity for combined gradient**
Less clean — makes activation derivative context-dependent.

**Chosen: Option A** — cleanest separation of concerns.

**Also fix `activations.rs:94-97`** to return the correct element-wise approximation for standalone use (even though it won't be called for output layer after Option A):
```rust
Activation::Softmax => {
    // Note: Full softmax Jacobian is diag(s) - s*s^T
    // Element-wise approximation for non-output-layer use:
    output * &(1.0 - output)
    // For output layer backprop, this is bypassed — see mlp.rs backward()
}
```

#### B.1.2: Fix Regressor MSE for Multi-Output
**File:** `regressor.rs:143-154`

```rust
fn mse_loss(predictions: &Array2<f64>, targets: &Array2<f64>) -> (f64, Array2<f64>) {
    let n_samples = predictions.nrows() as f64;
    let n_outputs = predictions.ncols() as f64;
    let diff = predictions - targets;
    let loss = diff.mapv(|d| d.powi(2)).sum() / (n_samples * n_outputs);
    let grad = diff.mapv(|d| 2.0 * d) / (n_samples * n_outputs);
    (loss, grad)
}
```

#### B.1.3: Fix Cross-Entropy Clipping
**File:** `classifier.rs:188-199`

Only clip the log argument, not the predictions themselves:
```rust
fn cross_entropy_loss(predictions: &Array2<f64>, targets: &Array2<f64>) -> (f64, Array2<f64>) {
    let n_samples = predictions.nrows() as f64;
    let eps = 1e-15;

    // Clip only for log computation, not for gradient
    let log_preds = predictions.mapv(|p| p.max(eps).ln());
    let loss = -(log_preds * targets).sum() / n_samples;

    // Gradient: (p - y) / n (already correct for softmax + CE)
    let grad = (predictions - targets) / n_samples;
    (loss, grad)
}
```

### Phase B.2: Fix Medium Bugs (4 changes)

#### B.2.1: Fix ELU Derivative Boundary
**File:** `activations.rs:101-107`
Change `if xi > 0.0` to `if xi >= 0.0` to match forward pass.

#### B.2.2: Fix Dropout RNG Determinism
**File:** `mlp.rs:292-295`
Store RNG state in MLP struct, don't reseed from OS every forward pass:
```rust
// In MLP struct: rng: Option<StdRng>
// In fit(): initialize once from random_state
// In forward(): use stored rng
```

#### B.2.3: Fix Empty Input Validation
**File:** `mlp.rs:439` and `classifier.rs` / `regressor.rs` `fit()`
Add: `if x.nrows() == 0 { return Err(FerroError::InvalidInput("empty training data")); }`

#### B.2.4: Add Input Dimension Validation to predict()
**Files:** `classifier.rs`, `regressor.rs`
Validate `x.ncols()` matches expected input features.

### Phase B.3: Numerical Gradient Checking

**File:** `ferroml-core/tests/correctness_neural.rs` (NEW — gradient section)

Finite difference gradient checking for all layer types:

```rust
/// Compute numerical gradient using central differences
fn numerical_gradient(
    f: impl Fn(&Array2<f64>) -> f64,
    x: &Array2<f64>,
    eps: f64,
) -> Array2<f64> {
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

mod gradient_tests {
    #[test] fn test_dense_layer_weight_gradient() { ... }
    #[test] fn test_dense_layer_bias_gradient() { ... }
    #[test] fn test_relu_gradient_positive() { ... }
    #[test] fn test_relu_gradient_negative() { ... }
    #[test] fn test_sigmoid_gradient() { ... }
    #[test] fn test_tanh_gradient() { ... }
    #[test] fn test_softmax_cross_entropy_gradient() { ... }
    #[test] fn test_mse_gradient() { ... }
    #[test] fn test_full_network_gradient_2_layers() { ... }
    #[test] fn test_full_network_gradient_3_layers() { ... }
}
```

**Tolerance:** Numerical vs analytical gradient relative error < 1e-5.

### Phase B.4: Python Fixture Generation

**File:** `benchmarks/neural_fixtures.py` (NEW)

```python
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.datasets import load_iris, load_diabetes, make_classification

# 1. MLPClassifier on Iris (3-class)
# 2. MLPClassifier on binary classification
# 3. MLPRegressor on Diabetes
# 4. MLPRegressor on simple linear data
# 5. Training loss curves
# 6. Predictions and probabilities
# 7. Weight shapes and layer sizes

# Note: Exact weight comparison NOT possible (different init, different
# convergence path). Compare:
# - Final accuracy/R² within tolerance
# - Output shape matches
# - Probabilities sum to 1.0
# - Loss decreases during training
```

### Phase B.5: Correctness Test File

**File:** `ferroml-core/tests/correctness_neural.rs` (NEW)

```rust
mod gradient_tests { ... }  // Phase B.3

mod classifier_tests {
    #[test] fn test_mlp_classifier_xor_convergence() { ... }
    #[test] fn test_mlp_classifier_iris_accuracy_above_90() { ... }
    #[test] fn test_mlp_classifier_binary_probabilities_sum_to_1() { ... }
    #[test] fn test_mlp_classifier_multiclass_probabilities_sum_to_1() { ... }
    #[test] fn test_mlp_classifier_predict_matches_argmax_proba() { ... }
    #[test] fn test_mlp_classifier_loss_decreases() { ... }
    #[test] fn test_mlp_classifier_deterministic_with_seed() { ... }
    #[test] fn test_mlp_classifier_different_activations() { ... }
}

mod regressor_tests {
    #[test] fn test_mlp_regressor_linear_r2_above_0_9() { ... }
    #[test] fn test_mlp_regressor_diabetes_r2_above_0_3() { ... }
    #[test] fn test_mlp_regressor_loss_decreases() { ... }
    #[test] fn test_mlp_regressor_deterministic_with_seed() { ... }
    #[test] fn test_mlp_regressor_score_matches_manual_r2() { ... }
}

mod optimizer_tests {
    #[test] fn test_adam_converges_faster_than_sgd() { ... }
    #[test] fn test_sgd_momentum_accelerates_convergence() { ... }
    #[test] fn test_learning_rate_too_high_diverges() { ... }
}

mod regularization_tests {
    #[test] fn test_l2_regularization_shrinks_weights() { ... }
    #[test] fn test_dropout_reduces_overfitting() { ... }
    #[test] fn test_early_stopping_stops_at_patience() { ... }
}

mod activation_tests {
    #[test] fn test_softmax_large_values_no_overflow() { ... }
    #[test] fn test_sigmoid_saturation_gradient() { ... }
    #[test] fn test_relu_dead_neurons_zero_gradient() { ... }
    #[test] fn test_elu_negative_continuity() { ... }
}

mod edge_cases {
    #[test] fn test_empty_input_returns_error() { ... }
    #[test] fn test_wrong_feature_count_returns_error() { ... }
    #[test] fn test_single_sample_training() { ... }
    #[test] fn test_single_feature() { ... }
    #[test] fn test_large_hidden_layers() { ... }
}

mod diagnostics_tests {
    #[test] fn test_training_diagnostics_populated() { ... }
    #[test] fn test_weight_statistics_reasonable() { ... }
    #[test] fn test_dead_neuron_detection_with_relu() { ... }
}

mod uncertainty_tests {
    #[test] fn test_mc_dropout_variance_increases_with_ood() { ... }
    #[test] fn test_prediction_uncertainty_ci_contains_mean() { ... }
}
```

**Total: ~40 tests**

## Success Criteria

- [ ] `cargo test -p ferroml-core --test correctness_neural` — all pass
- [ ] Numerical gradient error < 1e-5 for all layer types
- [ ] XOR convergence in <500 epochs (100% accuracy)
- [ ] Iris accuracy >90% with `[100]` hidden layer
- [ ] Diabetes R² >0.3 with `[64, 32]` hidden layers
- [ ] All probabilities sum to 1.0 within 1e-10
- [ ] Loss strictly decreases for first 50 epochs on simple problems
- [ ] L2 regularization measurably shrinks weights
- [ ] Empty/wrong-dimension inputs return FerroError, not panic

## Dependencies

- Python 3.10+ with sklearn for fixture generation
- No new Rust crate dependencies

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Fixing softmax backprop breaks existing classifier behavior | Run all 65 existing tests before/after; if accuracy improves, that confirms the fix |
| Numerical gradient checking is slow | Use tiny networks (2-3 layers, 5-10 neurons) for gradient tests |
| sklearn MLP uses different initialization | Don't compare weights — compare accuracy, loss decrease, probability validity |
| XOR convergence is non-deterministic | Fixed seed + generous epoch budget (500) + multiple restarts |
| Fixing bugs may cause tests to fail temporarily | Fix bugs in Phase B.1, update inline tests in same commit |
