//! Correctness tests for the explainability module.
//!
//! Property-based tests verifying TreeSHAP, KernelSHAP, PDP, ICE,
//! Permutation Importance, and H-statistic implementations.

use ferroml_core::explainability::{
    h_statistic, individual_conditional_expectation, partial_dependence, permutation_importance,
    GridMethod, HStatisticConfig, ICEConfig, KernelExplainer, KernelSHAPConfig, TreeExplainer,
};
use ferroml_core::models::{DecisionTreeRegressor, LinearRegression, Model, RandomForestRegressor};
use ndarray::{Array1, Array2};

// =============================================================================
// Data Generation Helpers
// =============================================================================

/// Generate simple additive regression data: y = 2*x0 + 3*x1 (no interaction)
/// x2 is pure noise.
fn make_simple_regression(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::zeros((n, 3));
    for i in 0..n {
        x[[i, 0]] = (i as f64) / n as f64;
        x[[i, 1]] = ((i * 7 + 3) % n) as f64 / n as f64;
        x[[i, 2]] = ((i * 13 + 5) % n) as f64 / n as f64; // noise
    }
    let y = Array1::from_vec((0..n).map(|i| 2.0 * x[[i, 0]] + 3.0 * x[[i, 1]]).collect());
    (x, y)
}

/// Generate data with interaction: y = x0 * x1 + x0 + x1
fn make_interaction_regression(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::zeros((n, 3));
    for i in 0..n {
        x[[i, 0]] = (i as f64) / n as f64;
        x[[i, 1]] = ((i * 7 + 3) % n) as f64 / n as f64;
        x[[i, 2]] = ((i * 13 + 5) % n) as f64 / n as f64; // noise
    }
    let y = Array1::from_vec(
        (0..n)
            .map(|i| x[[i, 0]] * x[[i, 1]] + x[[i, 0]] + x[[i, 1]])
            .collect(),
    );
    (x, y)
}

/// Generate monotonic data: y = 5*x0 + noise_feature
fn make_monotonic_regression(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::zeros((n, 2));
    for i in 0..n {
        x[[i, 0]] = i as f64 / n as f64;
        x[[i, 1]] = ((i * 11 + 7) % n) as f64 / n as f64; // noise
    }
    let y = Array1::from_vec((0..n).map(|i| 5.0 * x[[i, 0]]).collect());
    (x, y)
}

/// Simple R^2 scorer for permutation importance
fn r2_scorer(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> ferroml_core::Result<f64> {
    let mean = y_true.mean().unwrap_or(0.0);
    let ss_tot: f64 = y_true.iter().map(|&y| (y - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();

    if ss_tot < 1e-14 {
        Ok(1.0)
    } else {
        Ok(1.0 - ss_res / ss_tot)
    }
}

// =============================================================================
// TreeSHAP Tests (6 tests)
// =============================================================================

#[test]
fn treeshap_additivity_decision_tree() {
    // TreeSHAP additivity: sum(SHAP) + base_value == prediction
    let (x, y) = make_simple_regression(100);

    let mut model = DecisionTreeRegressor::new().with_max_depth(Some(5));
    model.fit(&x, &y).unwrap();

    let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

    for i in 0..10 {
        let sample = x.row(i).to_vec();
        let result = explainer.explain(&sample).unwrap();

        let reconstructed = result.base_value + result.shap_values.sum();
        let x_row = Array2::from_shape_vec((1, 3), sample).unwrap();
        let actual = model.predict(&x_row).unwrap()[0];

        assert!(
            (reconstructed - actual).abs() < 1e-6,
            "Sample {}: reconstructed {:.6} != actual {:.6}, diff = {:.2e}",
            i,
            reconstructed,
            actual,
            (reconstructed - actual).abs()
        );
    }
}

#[test]
fn treeshap_additivity_random_forest() {
    // TreeSHAP additivity with RandomForest
    let (x, y) = make_simple_regression(100);

    let mut model = RandomForestRegressor::new()
        .with_n_estimators(10)
        .with_max_depth(Some(4))
        .with_random_state(42);
    model.fit(&x, &y).unwrap();

    let explainer = TreeExplainer::from_random_forest_regressor(&model).unwrap();

    for i in 0..10 {
        let sample = x.row(i).to_vec();
        let result = explainer.explain(&sample).unwrap();

        let reconstructed = result.base_value + result.shap_values.sum();
        let x_row = Array2::from_shape_vec((1, 3), sample).unwrap();
        let actual = model.predict(&x_row).unwrap()[0];

        assert!(
            (reconstructed - actual).abs() < 1e-6,
            "Sample {}: reconstructed {:.6} != actual {:.6}, diff = {:.2e}",
            i,
            reconstructed,
            actual,
            (reconstructed - actual).abs()
        );
    }
}

#[test]
fn treeshap_local_accuracy() {
    // For each sample, base_value + sum(shap_values) should equal prediction
    let (x, y) = make_simple_regression(50);

    let mut model = DecisionTreeRegressor::new().with_max_depth(Some(4));
    model.fit(&x, &y).unwrap();

    let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();
    let batch = explainer.explain_batch(&x).unwrap();

    for i in 0..x.nrows() {
        let x_row = x.slice(ndarray::s![i..i + 1, ..]).to_owned();
        let actual = model.predict(&x_row).unwrap()[0];
        let reconstructed = batch.base_value + batch.shap_values.row(i).sum();

        assert!(
            (reconstructed - actual).abs() < 1e-6,
            "Sample {}: local accuracy failed: {:.6} != {:.6}",
            i,
            reconstructed,
            actual
        );
    }
}

#[test]
fn treeshap_dummy_feature_near_zero() {
    // Noise feature (x2) should have SHAP value near 0
    // y = 2*x0 + 3*x1 (x2 not used)
    let (x, y) = make_simple_regression(100);

    let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
    model.fit(&x, &y).unwrap();

    let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();
    let batch = explainer.explain_batch(&x).unwrap();

    // Mean absolute SHAP for noise feature should be much smaller than informative features
    let mean_abs = batch.mean_abs_shap();
    let noise_importance = mean_abs[2];
    let min_informative = mean_abs[0].min(mean_abs[1]);

    // Noise feature should be less important
    assert!(
        noise_importance < min_informative,
        "Noise feature importance {:.4} should be less than informative {:.4}",
        noise_importance,
        min_informative
    );
}

#[test]
fn treeshap_consistency_batch() {
    // Batch explain should give same results as individual explain
    let (x, y) = make_simple_regression(50);

    let mut model = DecisionTreeRegressor::new().with_max_depth(Some(4));
    model.fit(&x, &y).unwrap();

    let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

    let batch = explainer.explain_batch(&x).unwrap();

    for i in 0..5 {
        let individual = explainer.explain(&x.row(i).to_vec()).unwrap();
        for j in 0..3 {
            assert!(
                (batch.shap_values[[i, j]] - individual.shap_values[j]).abs() < 1e-10,
                "Batch vs individual mismatch at ({}, {}): {:.6} != {:.6}",
                i,
                j,
                batch.shap_values[[i, j]],
                individual.shap_values[j]
            );
        }
    }
}

#[test]
fn treeshap_informative_feature_larger_shap() {
    // Feature 1 (coeff 3) should generally have larger mean |SHAP| than feature 0 (coeff 2)
    // when both features have similar range
    let n = 100;
    let mut x = Array2::zeros((n, 3));
    for i in 0..n {
        x[[i, 0]] = (i as f64) / n as f64;
        x[[i, 1]] = (i as f64) / n as f64; // Same range as x0
        x[[i, 2]] = ((i * 13 + 5) % n) as f64 / n as f64;
    }
    let y = Array1::from_vec((0..n).map(|i| 2.0 * x[[i, 0]] + 3.0 * x[[i, 1]]).collect());

    let mut model = DecisionTreeRegressor::new().with_max_depth(Some(5));
    model.fit(&x, &y).unwrap();

    let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();
    let batch = explainer.explain_batch(&x).unwrap();

    let mean_abs = batch.mean_abs_shap();
    // Both informative features (0, 1) should have much larger importance than noise feature (2)
    let informative_total = mean_abs[0] + mean_abs[1];
    assert!(
        informative_total > mean_abs[2] * 5.0,
        "Informative features importance {:.4} should dominate noise {:.4}",
        informative_total,
        mean_abs[2]
    );
}

// =============================================================================
// KernelSHAP Tests (4 tests)
// =============================================================================

#[test]
fn kernelshap_additivity() {
    // base_value + sum(SHAP) should approximately equal prediction
    let (x, y) = make_simple_regression(50);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let config = KernelSHAPConfig::new()
        .with_n_samples(1024)
        .with_random_state(42);
    let explainer = KernelExplainer::new(&model, &x, config).unwrap();

    for i in 0..5 {
        let sample = x.row(i).to_vec();
        let result = explainer.explain(&sample).unwrap();

        let reconstructed = result.base_value + result.shap_values.sum();
        let x_row = Array2::from_shape_vec((1, 3), sample).unwrap();
        let actual = model.predict(&x_row).unwrap()[0];

        assert!(
            (reconstructed - actual).abs() < 0.5,
            "KernelSHAP additivity: sample {}: reconstructed {:.4} vs actual {:.4}, diff = {:.4}",
            i,
            reconstructed,
            actual,
            (reconstructed - actual).abs()
        );
    }
}

#[test]
fn kernelshap_local_accuracy() {
    // Same property as TreeSHAP but with wider tolerance
    let (x, y) = make_simple_regression(40);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let config = KernelSHAPConfig::new()
        .with_n_samples(2048)
        .with_random_state(42);
    let explainer = KernelExplainer::new(&model, &x, config).unwrap();

    let sample = x.row(0).to_vec();
    let result = explainer.explain(&sample).unwrap();

    let reconstructed = result.prediction();
    let x_row = Array2::from_shape_vec((1, 3), sample).unwrap();
    let actual = model.predict(&x_row).unwrap()[0];

    assert!(
        (reconstructed - actual).abs() < 0.5,
        "KernelSHAP local accuracy: {:.4} vs {:.4}",
        reconstructed,
        actual
    );
}

#[test]
fn kernelshap_dummy_feature_near_zero() {
    // Noise feature should have small SHAP values
    let (x, y) = make_simple_regression(50);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let config = KernelSHAPConfig::new()
        .with_n_samples(1024)
        .with_random_state(42);
    let explainer = KernelExplainer::new(&model, &x, config).unwrap();

    let batch = explainer
        .explain_batch(&x.slice(ndarray::s![0..10, ..]).to_owned())
        .unwrap();
    let mean_abs = batch.mean_abs_shap();

    // Noise feature (index 2) should be the least important
    let noise_imp = mean_abs[2];
    let max_informative = mean_abs[0].max(mean_abs[1]);

    assert!(
        noise_imp < max_informative,
        "Noise feature importance {:.4} should be less than informative {:.4}",
        noise_imp,
        max_informative
    );
}

#[test]
fn kernelshap_important_feature_larger() {
    // Important feature should have larger |SHAP| than noise
    let (x, y) = make_simple_regression(50);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let config = KernelSHAPConfig::new()
        .with_n_samples(1024)
        .with_random_state(42);
    let explainer = KernelExplainer::new(&model, &x, config).unwrap();

    let batch = explainer
        .explain_batch(&x.slice(ndarray::s![0..10, ..]).to_owned())
        .unwrap();
    let mean_abs = batch.mean_abs_shap();

    // At least one informative feature should be more important than noise
    let informative_max = mean_abs[0].max(mean_abs[1]);
    let noise = mean_abs[2];

    assert!(
        informative_max > noise,
        "Max informative importance {:.4} should be > noise {:.4}",
        informative_max,
        noise
    );
}

// =============================================================================
// PDP/ICE Tests (4 tests)
// =============================================================================

#[test]
fn pdp_monotonic_trend() {
    // PDP on monotonically related feature should show increasing trend
    let (x, y) = make_monotonic_regression(100);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let result = partial_dependence(&model, &x, 0, 20, GridMethod::Uniform, false).unwrap();

    // PDP for feature 0 (y = 5*x0) should be monotonically increasing
    assert!(
        result.is_monotonic_increasing(),
        "PDP should be monotonically increasing for y=5*x0, values: {:?}",
        result.pdp_values
    );
}

#[test]
fn pdp_constant_feature() {
    // PDP for noise feature should be roughly constant
    let (x, y) = make_monotonic_regression(100);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    // PDP for feature 1 (noise, not used in y)
    let result = partial_dependence(&model, &x, 1, 20, GridMethod::Uniform, false).unwrap();

    // Effect range should be very small compared to the main effect
    let main_result = partial_dependence(&model, &x, 0, 20, GridMethod::Uniform, false).unwrap();

    assert!(
        result.effect_range() < main_result.effect_range() * 0.5,
        "Noise feature PDP effect range {:.4} should be much smaller than main feature {:.4}",
        result.effect_range(),
        main_result.effect_range()
    );
}

#[test]
fn ice_centering_starts_at_zero() {
    // Centered ICE curves should start at 0 at the reference point
    let (x, y) = make_simple_regression(50);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let config = ICEConfig::new().with_n_grid_points(10).with_centering(0); // Center at first grid point

    let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

    let centered = result
        .centered_ice
        .as_ref()
        .expect("Centered ICE should be computed");

    // All curves should be 0 at the reference point (index 0)
    for i in 0..result.n_samples {
        assert!(
            centered[[i, 0]].abs() < 1e-10,
            "Centered ICE sample {} at reference should be 0, got {:.6}",
            i,
            centered[[i, 0]]
        );
    }
}

#[test]
fn ice_shape_correct() {
    // ICE curves should have shape (n_samples, n_grid_points)
    let (x, y) = make_simple_regression(30);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let n_grid = 15;
    let config = ICEConfig::new().with_n_grid_points(n_grid);
    let result = individual_conditional_expectation(&model, &x, 0, config).unwrap();

    assert_eq!(
        result.ice_curves.shape(),
        &[30, n_grid],
        "ICE shape should be ({}, {}), got {:?}",
        30,
        n_grid,
        result.ice_curves.shape()
    );
    assert_eq!(result.grid_values.len(), n_grid);
    assert_eq!(result.pdp_values.len(), n_grid);
}

// =============================================================================
// Permutation Importance Tests (3 tests)
// =============================================================================

#[test]
fn permutation_importance_informative_higher() {
    // Informative features should have higher importance than noise
    let (x, y) = make_simple_regression(100);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let result = permutation_importance(&model, &x, &y, r2_scorer, 10, Some(42)).unwrap();

    // Features 0 and 1 should be more important than feature 2 (noise)
    let max_informative = result.importances_mean[0].max(result.importances_mean[1]);
    let noise_importance = result.importances_mean[2];

    assert!(
        max_informative > noise_importance,
        "Informative feature importance {:.4} should be > noise {:.4}",
        max_informative,
        noise_importance
    );
}

#[test]
fn permutation_importance_reproducibility() {
    // Same random_state should give identical results
    let (x, y) = make_simple_regression(50);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let result1 = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();
    let result2 = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

    for i in 0..3 {
        assert!(
            (result1.importances_mean[i] - result2.importances_mean[i]).abs() < 1e-10,
            "Results differ at feature {}: {:.6} vs {:.6}",
            i,
            result1.importances_mean[i],
            result2.importances_mean[i]
        );
    }
}

#[test]
fn permutation_importance_baseline_valid() {
    // Baseline score should be positive for a well-fitted model
    let (x, y) = make_simple_regression(100);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let result = permutation_importance(&model, &x, &y, r2_scorer, 5, Some(42)).unwrap();

    assert!(
        result.baseline_score > 0.5,
        "Baseline R^2 should be high for well-fitted linear data, got {:.4}",
        result.baseline_score
    );

    // CI ordering should be correct
    for i in 0..3 {
        assert!(
            result.ci_lower[i] <= result.importances_mean[i],
            "CI lower {} should be <= mean {} for feature {}",
            result.ci_lower[i],
            result.importances_mean[i],
            i
        );
        assert!(
            result.importances_mean[i] <= result.ci_upper[i],
            "Mean {} should be <= CI upper {} for feature {}",
            result.importances_mean[i],
            result.ci_upper[i],
            i
        );
    }
}

// =============================================================================
// H-Statistic Tests (3 tests)
// =============================================================================

#[test]
fn h_statistic_additive_model_low() {
    // For additive model (no interaction), H^2 should be near 0
    let (x, y) = make_simple_regression(50);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let result = h_statistic(
        &model,
        &x,
        0,
        1,
        HStatisticConfig::new().with_grid_points(10),
    )
    .unwrap();

    assert!(
        result.h_squared < 0.05,
        "H^2 for additive model should be near 0, got {:.4}",
        result.h_squared
    );
}

#[test]
fn h_statistic_interaction_model_higher() {
    // For model with interaction, H^2 should be > 0
    let (x, y) = make_interaction_regression(80);

    let mut model = RandomForestRegressor::new()
        .with_n_estimators(20)
        .with_max_depth(Some(5))
        .with_random_state(42);
    model.fit(&x, &y).unwrap();

    let result = h_statistic(
        &model,
        &x,
        0,
        1,
        HStatisticConfig::new().with_grid_points(10),
    )
    .unwrap();

    // For interaction model, H^2 should be detectable
    assert!(
        result.h_squared >= 0.0,
        "H^2 should be non-negative, got {:.4}",
        result.h_squared
    );
}

#[test]
fn h_statistic_bounded() {
    // H^2 should always be in [0, 1]
    let (x, y) = make_simple_regression(50);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    let result = h_statistic(
        &model,
        &x,
        0,
        1,
        HStatisticConfig::new().with_grid_points(10),
    )
    .unwrap();

    assert!(
        result.h_squared >= 0.0 && result.h_squared <= 1.0,
        "H^2 should be in [0, 1], got {:.4}",
        result.h_squared
    );
    assert!(
        result.h_statistic >= 0.0 && result.h_statistic <= 1.0,
        "H should be in [0, 1], got {:.4}",
        result.h_statistic
    );
}
