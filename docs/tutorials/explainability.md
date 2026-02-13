# Explainability Tutorial

FerroML provides a comprehensive suite of model explainability tools. This tutorial covers how to understand *why* your model makes the predictions it does.

## Learning Objectives

- Explain individual predictions with TreeSHAP and KernelSHAP
- Visualize feature effects with partial dependence and ICE curves
- Measure feature importance with permutation importance
- Detect feature interactions with the H-statistic

## TreeSHAP: Explaining Tree-Based Models

TreeSHAP computes exact Shapley values for tree-based models in polynomial time, using Lundberg's Algorithm 2 (2018). It explains each prediction as a sum of feature contributions.

```rust
use ferroml_core::explainability::treeshap::TreeExplainer;
use ferroml_core::models::forest::RandomForestClassifier;
use ferroml_core::models::Model;

// Train a model
let mut model = RandomForestClassifier::new()
    .with_n_trees(100)
    .with_random_state(Some(42));
model.fit(&x_train, &y_train)?;

// Create an explainer
let explainer = TreeExplainer::from_random_forest_classifier(&model)?;

// Explain a single prediction
let result = explainer.explain(&x_test.row(0).to_owned())?;

println!("Base value (expected output): {:.4}", result.base_value);
println!("Prediction: {:.4}", result.prediction());

// Show feature contributions sorted by magnitude
for &idx in result.sorted_indices().iter() {
    println!("  Feature {}: SHAP = {:.4} (value = {:.4})",
        idx, result.shap_values[idx], result.feature_values[idx]);
}
```

### How SHAP Values Work

SHAP values decompose a prediction into additive contributions:

```
prediction = base_value + shap[0] + shap[1] + ... + shap[n]
```

- **base_value**: The average model output across the training data
- **shap[i] > 0**: Feature i pushes the prediction *higher*
- **shap[i] < 0**: Feature i pushes the prediction *lower*
- **|shap[i]|**: The magnitude of feature i's influence

### Top-K Features

Quickly identify the most important features for a prediction:

```rust
// Get the top 5 most influential features
let top_features = result.top_k(5);
for &idx in &top_features {
    println!("Feature {}: SHAP = {:.4}", idx, result.shap_values[idx]);
}
```

### Batch Explanations

Explain multiple predictions at once:

```rust
let batch_results = explainer.explain_batch(&x_test)?;

// Aggregate: mean absolute SHAP values across all samples
// gives global feature importance
let n_features = batch_results.results[0].shap_values.len();
let mut global_importance = vec![0.0; n_features];
for r in &batch_results.results {
    for (i, &v) in r.shap_values.iter().enumerate() {
        global_importance[i] += v.abs();
    }
}
for v in &mut global_importance {
    *v /= batch_results.results.len() as f64;
}
```

### Supported Models

TreeSHAP works with all tree-based models:

```rust
let explainer = TreeExplainer::from_decision_tree(&model)?;
let explainer = TreeExplainer::from_random_forest_classifier(&model)?;
let explainer = TreeExplainer::from_random_forest_regressor(&model)?;
let explainer = TreeExplainer::from_gradient_boosting_classifier(&model)?;
let explainer = TreeExplainer::from_gradient_boosting_regressor(&model)?;
```

## KernelSHAP: Explaining Any Model

KernelSHAP is a model-agnostic method that approximates SHAP values for *any* model. It's slower than TreeSHAP but works universally.

```rust
use ferroml_core::explainability::kernelshap::{KernelExplainer, KernelSHAPConfig};

// Configure KernelSHAP
let config = KernelSHAPConfig {
    n_samples: Some(2048),       // Number of coalition samples
    max_background_samples: 100, // Cap background data size
    random_state: Some(42),
    regularization: 0.01,        // Numerical stability
    paired_sampling: true,       // Reduces variance
};

// Works with ANY model (linear, SVM, neural networks, etc.)
let explainer = KernelExplainer::new(&model, &x_train, config)?;

// Explain a prediction
let result = explainer.explain(&x_test.row(0).to_owned())?;

println!("Base value: {:.4}", result.base_value);
for &idx in result.sorted_indices().iter() {
    println!("  Feature {}: SHAP = {:.4}", idx, result.shap_values[idx]);
}
```

### TreeSHAP vs KernelSHAP

| Property | TreeSHAP | KernelSHAP |
|----------|----------|------------|
| Speed | Fast (polynomial time) | Slow (sampling-based) |
| Accuracy | Exact | Approximate |
| Models | Tree-based only | Any model |
| Background data | Not needed | Required |

Use TreeSHAP when available; fall back to KernelSHAP for non-tree models.

## Partial Dependence Plots (PDP)

PDPs show the marginal effect of a feature on predictions, averaged over all samples.

```rust
use ferroml_core::explainability::partial_dependence::{partial_dependence, GridMethod};

// Compute PDP for feature 0 with 50 grid points
let pdp = partial_dependence(
    &model,
    &x_test,
    0,                       // feature index
    50,                      // number of grid points
    GridMethod::Percentile,  // grid based on data percentiles
    false,                   // don't return ICE curves
)?;

// The PDP curve: grid values -> average predictions
for i in 0..pdp.grid_values.len() {
    println!("Feature value: {:.3} -> Avg prediction: {:.4} (std: {:.4})",
        pdp.grid_values[i],
        pdp.pdp_values[i],
        pdp.pdp_std[i]);
}

// Analyze the effect
println!("Effect range: {:.4}", pdp.effect_range());
println!("Monotonically increasing: {}", pdp.is_monotonic_increasing());
println!("Monotonically decreasing: {}", pdp.is_monotonic_decreasing());
```

### PDP with ICE Curves

Request Individual Conditional Expectation (ICE) curves alongside the PDP to see per-sample variation:

```rust
let pdp = partial_dependence(
    &model,
    &x_test,
    0,                       // feature index
    50,                      // grid points
    GridMethod::Percentile,
    true,                    // return ICE curves
)?;

// ICE curves show per-sample effects
if let Some(ice) = &pdp.ice_curves {
    println!("ICE curves shape: ({}, {})", ice.nrows(), ice.ncols());
    // Each row is one sample's response curve
}
```

## Individual Conditional Expectation (ICE)

ICE curves show how predictions change for *each individual sample* as a feature varies. Unlike PDPs (which average), ICE reveals heterogeneity.

```rust
use ferroml_core::explainability::ice::{individual_conditional_expectation, ICEConfig};

let config = ICEConfig::new()
    .with_n_grid_points(50)
    .with_grid_method(GridMethod::Percentile)
    .with_centering(0)    // Center curves at grid point 0 (c-ICE)
    .with_derivative();   // Also compute d-ICE (derivative)

let result = individual_conditional_expectation(&model, &x_test, 0, config)?;

// Check for interaction effects via heterogeneity
println!("Mean heterogeneity: {:.4}", result.mean_heterogeneity());
if result.has_interactions(0.1) {
    println!("Feature likely interacts with other features");
}

// Centered ICE (c-ICE) reveals interaction effects more clearly
if let Some(centered) = &result.centered_ice {
    println!("Centered ICE available ({} samples)", centered.nrows());
}

// Print a summary of the ICE analysis
println!("{}", result.summary());
```

## Permutation Importance

Permutation importance measures how much model performance drops when a feature is randomly shuffled. Works with any model and any metric.

```rust
use ferroml_core::explainability::permutation_importance;
use ferroml_core::metrics::accuracy;

let result = permutation_importance(
    &model,
    &x_test,
    &y_test,
    |y_true, y_pred| accuracy(y_true, y_pred),  // any metric
    10,          // number of repeats
    Some(42),    // random seed
)?;

println!("Baseline accuracy: {:.4}", result.baseline_score);
println!("\nFeature importances (mean decrease in accuracy):");

for &idx in result.sorted_indices().iter() {
    let sig = if result.is_significant(idx) { "*" } else { " " };
    println!("  {} Feature {:2}: {:.4} +/- {:.4} [{:.4}, {:.4}]",
        sig,
        idx,
        result.importances_mean[idx],
        result.importances_std[idx],
        result.ci_lower[idx],
        result.ci_upper[idx]);
}

// Features whose CI doesn't include zero are statistically significant
let sig_features = result.significant_features();
println!("\nSignificant features: {:?}", sig_features);
```

### Permutation Importance vs SHAP

| Property | Permutation Importance | SHAP |
|----------|----------------------|------|
| Scope | Global | Local (per-prediction) |
| Interactions | Captured implicitly | Explicit attribution |
| Correlated features | Can underestimate | Handles properly |
| Speed | Fast | Slower |
| Interpretability | "How much does performance drop?" | "How much does this feature contribute?" |

## H-Statistic: Detecting Interactions

Friedman's H-statistic measures the strength of interaction between features.

```rust
use ferroml_core::explainability::h_statistic::{h_statistic, HStatisticConfig};

let config = HStatisticConfig::new()
    .with_grid_points(50)
    .with_grid_method(GridMethod::Percentile);

// Pairwise interaction between features 0 and 1
let result = h_statistic(&model, &x_test, 0, 1, &config)?;

println!("H-statistic: {:.4}", result.h_statistic);
println!("Interpretation: {}", result.interpretation());
// H > 0.1: Strong interaction
// H < 0.1: Weak interaction
```

### Interaction Matrix

Compute all pairwise interactions at once:

```rust
use ferroml_core::explainability::h_statistic::h_statistic_matrix;

let matrix = h_statistic_matrix(&model, &x_test, &config)?;

// Find the strongest interactions
let top_interactions = matrix.top_k(5);
for (i, j, h) in &top_interactions {
    println!("Features ({}, {}): H = {:.4}", i, j, h);
}

println!("{}", matrix.summary());
```

## Choosing the Right Tool

| Question | Tool |
|----------|------|
| Why did the model make *this* prediction? | TreeSHAP / KernelSHAP |
| Which features are globally most important? | Permutation importance |
| How does a feature affect predictions on average? | PDP |
| Does the feature effect vary across samples? | ICE |
| Do features interact? | H-statistic |

## Complete Explainability Workflow

```rust
use ferroml_core::models::forest::RandomForestClassifier;
use ferroml_core::models::Model;
use ferroml_core::explainability::treeshap::TreeExplainer;
use ferroml_core::explainability::partial_dependence::{partial_dependence, GridMethod};
use ferroml_core::explainability::permutation_importance;
use ferroml_core::metrics::accuracy;

// 1. Train model
let mut model = RandomForestClassifier::new()
    .with_n_trees(100)
    .with_random_state(Some(42));
model.fit(&x_train, &y_train)?;

// 2. Global importance — which features matter most?
let perm = permutation_importance(
    &model, &x_test, &y_test,
    |yt, yp| accuracy(yt, yp), 10, Some(42),
)?;
println!("Top features: {:?}", perm.top_k(5));

// 3. Feature effects — how do top features affect predictions?
for &feat in perm.top_k(3).iter() {
    let pdp = partial_dependence(&model, &x_test, feat, 50, GridMethod::Percentile, false)?;
    println!("Feature {}: effect range = {:.4}, monotonic = {}",
        feat, pdp.effect_range(), pdp.is_monotonic());
}

// 4. Local explanations — why this specific prediction?
let explainer = TreeExplainer::from_random_forest_classifier(&model)?;
let shap = explainer.explain(&x_test.row(0).to_owned())?;
println!("Prediction = {:.4}", shap.prediction());
for &idx in shap.top_k(3).iter() {
    println!("  Feature {}: SHAP = {:.4}", idx, shap.shap_values[idx]);
}
```

## Next Steps

- [Statistical Features Tutorial](statistical-features.md) — Confidence intervals, hypothesis testing
- [Quick Start Tutorial](quickstart.md) — Basic model usage
- [API Reference](https://docs.rs/ferroml-core) — Full API documentation
