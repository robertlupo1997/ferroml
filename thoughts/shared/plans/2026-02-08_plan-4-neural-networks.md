# Plan 4: Neural Networks (MLP)

**Date:** 2026-02-08
**Reference:** `thoughts/shared/research/2026-02-08_ferroml-comprehensive-assessment.md`
**Priority:** Medium
**Estimated Tasks:** 10

## Objective

Implement MLPClassifier and MLPRegressor with FerroML-style statistical diagnostics.

## Context

From research:
- No neural networks currently implemented
- Gap compared to sklearn's MLPClassifier/MLPRegressor
- User wants statistical extensions beyond typical implementations

## Statistical Extensions (Beyond sklearn)

1. **Training Diagnostics** - Loss curves with convergence tests
2. **Weight Analysis** - Distribution of weights, dead neuron detection
3. **Gradient Flow** - Vanishing/exploding gradient detection
4. **Uncertainty Quantification** - MC Dropout for prediction intervals
5. **Feature Attribution** - Integrated gradients, saliency

## Tasks

### Task 4.1: Create neural module structure
**File:** `ferroml-core/src/neural/mod.rs`
**Description:** Module structure with Layer trait, activation functions.
**Exports:** MLPClassifier, MLPRegressor, layers, activations

### Task 4.2: Implement core MLP architecture
**File:** `ferroml-core/src/neural/mlp.rs`
**Description:**
- Forward pass with configurable hidden layers
- Backpropagation with autodiff-style gradients
- Activation functions: ReLU, Sigmoid, Tanh, Softmax
**Lines:** ~400

### Task 4.3: Implement optimizers
**File:** `ferroml-core/src/neural/optimizers.rs`
**Description:**
- SGD with momentum
- Adam optimizer
- Learning rate scheduling
**Lines:** ~200

### Task 4.4: Implement MLPClassifier
**File:** `ferroml-core/src/neural/classifier.rs`
**Description:**
- sklearn-compatible API: fit(), predict(), predict_proba()
- Cross-entropy loss
- Early stopping with validation
**Lines:** ~300

### Task 4.5: Implement MLPRegressor
**File:** `ferroml-core/src/neural/regressor.rs`
**Description:**
- sklearn-compatible API: fit(), predict()
- MSE loss
- Early stopping with validation
**Lines:** ~250

### Task 4.6: Add training diagnostics
**File:** `ferroml-core/src/neural/diagnostics.rs`
**Description:**
- `TrainingDiagnostics` struct
- Loss curve analysis with convergence test
- Learning rate analysis (too high/low detection)
- Gradient statistics per layer
**Lines:** ~200

### Task 4.7: Add weight analysis
**File:** `ferroml-core/src/neural/analysis.rs`
**Description:**
- `weight_statistics()` - Mean, std, sparsity per layer
- `dead_neuron_detection()` - ReLU neurons that never activate
- `weight_distribution_tests()` - Normality, initialization quality
**Lines:** ~150

### Task 4.8: Add uncertainty quantification
**File:** `ferroml-core/src/neural/uncertainty.rs`
**Description:**
- MC Dropout for prediction intervals
- `predict_with_uncertainty()` - Returns mean, std, CI
- Calibration analysis for probabilities
**Lines:** ~200

### Task 4.9: Add MLP tests
**File:** `ferroml-core/src/neural/` (tests modules)
**Description:**
- XOR problem (non-linear classification)
- Regression on synthetic data
- Sklearn comparison on Iris/Diabetes
- Convergence tests
**Tests:** ~20

### Task 4.10: Add Python bindings for neural
**File:** `ferroml-python/src/neural.rs`
**Description:** PyO3 bindings for MLPClassifier, MLPRegressor.
**Lines:** ~200

## Success Criteria

- [ ] MLPClassifier achieves >95% on Iris with proper hyperparameters
- [ ] MLPRegressor achieves R² > 0.8 on Diabetes
- [ ] Training diagnostics detect convergence issues
- [ ] MC Dropout provides calibrated uncertainty estimates
- [ ] Sklearn API compatibility verified

## API Design

```rust
// sklearn-compatible
let mlp = MLPClassifier::new()
    .hidden_layer_sizes(&[100, 50])
    .activation(Activation::ReLU)
    .solver(Solver::Adam)
    .max_iter(200)
    .fit(&X, &y)?;

// FerroML statistical extensions
let diagnostics = mlp.training_diagnostics();
println!("Converged: {}", diagnostics.converged);
println!("Dead neurons: {:?}", mlp.dead_neuron_detection());

let (pred, uncertainty) = mlp.predict_with_uncertainty(&X_test, n_samples=100)?;
println!("Prediction intervals: {:?}", uncertainty.confidence_intervals(0.95));
```

## Dependencies

- ndarray (existing)
- rand (existing)
- No new crates required (pure Rust implementation)

## Notes

- This is a from-scratch implementation, not wrapping a DL framework
- Focus on correctness and diagnostics over performance
- GPU support deferred to separate plan
