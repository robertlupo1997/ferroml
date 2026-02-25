//! Adversarial Input Tests for FerroML Metrics, CV, and Preprocessing
//!
//! Tests edge cases in metrics (perfect/worst/constant predictions),
//! cross-validation (k > n, single class), and transformers (not-fitted errors).
//!
//! Fixtures loaded from `benchmarks/fixtures/adversarial/*.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use std::path::Path;

use ferroml_core::cv::{CrossValidator, KFold, StratifiedKFold};
use ferroml_core::decomposition::PCA;
use ferroml_core::metrics::classification::{
    accuracy, f1_score, matthews_corrcoef, precision, recall,
};
use ferroml_core::metrics::regression::{mae, mse, r2_score, rmse};
use ferroml_core::metrics::Average;
use ferroml_core::preprocessing::scalers::{MinMaxScaler, StandardScaler};
use ferroml_core::preprocessing::Transformer;

// =============================================================================
// Fixture Loading Helpers
// =============================================================================

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("benchmarks")
        .join("fixtures")
}

fn load_fixture(name: &str) -> Value {
    let path = fixtures_dir().join(name);
    std::fs::read_to_string(&path)
        .map(|d| serde_json::from_str(&d).unwrap())
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e))
}

fn json_to_array1(val: &Value) -> Array1<f64> {
    Array1::from_vec(
        val.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect(),
    )
}

#[allow(dead_code)]
fn json_to_array2(val: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = val
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect();
    if rows.is_empty() {
        return Array2::from_shape_vec((0, 0), vec![]).unwrap();
    }
    let ncols = rows[0].len();
    let nrows = rows.len();
    if ncols == 0 {
        return Array2::from_shape_vec((nrows, 0), vec![]).unwrap();
    }
    Array2::from_shape_vec((nrows, ncols), rows.into_iter().flatten().collect()).unwrap()
}

// =============================================================================
// CLASSIFICATION METRICS: Perfect predictions
// =============================================================================

#[test]
fn metrics_perfect_accuracy() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_perfect"]);
    let result = accuracy(&y_true, &y_pred).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "Perfect accuracy should be 1.0, got {}",
        result
    );
}

#[test]
fn metrics_perfect_f1() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_perfect"]);
    let result = f1_score(&y_true, &y_pred, Average::Micro).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "Perfect F1 should be 1.0, got {}",
        result
    );
}

#[test]
fn metrics_perfect_precision_recall() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_perfect"]);
    let p = precision(&y_true, &y_pred, Average::Micro).unwrap();
    let r = recall(&y_true, &y_pred, Average::Micro).unwrap();
    assert!(
        (p - 1.0).abs() < 1e-10,
        "Perfect precision should be 1.0, got {}",
        p
    );
    assert!(
        (r - 1.0).abs() < 1e-10,
        "Perfect recall should be 1.0, got {}",
        r
    );
}

#[test]
fn metrics_perfect_mcc() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_perfect"]);
    let result = matthews_corrcoef(&y_true, &y_pred).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "Perfect MCC should be 1.0, got {}",
        result
    );
}

// =============================================================================
// CLASSIFICATION METRICS: Worst-case predictions (all wrong)
// =============================================================================

#[test]
fn metrics_worst_accuracy() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_worst"]);
    let result = accuracy(&y_true, &y_pred).unwrap();
    assert!(
        (result - 0.0).abs() < 1e-10,
        "Worst accuracy should be 0.0, got {}",
        result
    );
}

#[test]
fn metrics_worst_f1() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_worst"]);
    let result = f1_score(&y_true, &y_pred, Average::Micro).unwrap();
    assert!(
        (result - 0.0).abs() < 1e-10,
        "Worst F1 should be 0.0, got {}",
        result
    );
}

#[test]
fn metrics_worst_mcc() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_worst"]);
    let result = matthews_corrcoef(&y_true, &y_pred).unwrap();
    assert!(
        (result - (-1.0)).abs() < 1e-10,
        "Worst MCC should be -1.0, got {}",
        result
    );
}

// =============================================================================
// CLASSIFICATION METRICS: Constant predictions (all 0 or all 1)
// =============================================================================

#[test]
fn metrics_constant_zero_predictions_no_panic() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_constant_zero"]);
    // All-zero predictions: precision for class 1 is 0/0 -> should not panic
    let acc = accuracy(&y_true, &y_pred);
    assert!(
        acc.is_ok(),
        "accuracy should not fail on constant predictions"
    );
    let f1 = f1_score(&y_true, &y_pred, Average::Micro);
    assert!(f1.is_ok(), "f1 should not fail on constant predictions");
    let p = precision(&y_true, &y_pred, Average::Micro);
    assert!(
        p.is_ok(),
        "precision should not fail on constant predictions"
    );
    let r = recall(&y_true, &y_pred, Average::Micro);
    assert!(r.is_ok(), "recall should not fail on constant predictions");
}

#[test]
fn metrics_constant_one_predictions_no_panic() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_constant_one"]);
    let acc = accuracy(&y_true, &y_pred);
    assert!(
        acc.is_ok(),
        "accuracy should not fail on constant-one predictions"
    );
    let f1 = f1_score(&y_true, &y_pred, Average::Micro);
    assert!(f1.is_ok(), "f1 should not fail on constant-one predictions");
    let mcc = matthews_corrcoef(&y_true, &y_pred);
    assert!(
        mcc.is_ok(),
        "MCC should not fail on constant-one predictions"
    );
}

// =============================================================================
// REGRESSION METRICS: Perfect predictions
// =============================================================================

#[test]
fn metrics_regression_perfect_mse() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["regression"]["y_true_reg"]);
    let y_pred = json_to_array1(&fixture["regression"]["y_pred_perfect_reg"]);
    let result = mse(&y_true, &y_pred).unwrap();
    assert!(
        result.abs() < 1e-10,
        "Perfect MSE should be 0.0, got {}",
        result
    );
}

#[test]
fn metrics_regression_perfect_rmse() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["regression"]["y_true_reg"]);
    let y_pred = json_to_array1(&fixture["regression"]["y_pred_perfect_reg"]);
    let result = rmse(&y_true, &y_pred).unwrap();
    assert!(
        result.abs() < 1e-10,
        "Perfect RMSE should be 0.0, got {}",
        result
    );
}

#[test]
fn metrics_regression_perfect_mae() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["regression"]["y_true_reg"]);
    let y_pred = json_to_array1(&fixture["regression"]["y_pred_perfect_reg"]);
    let result = mae(&y_true, &y_pred).unwrap();
    assert!(
        result.abs() < 1e-10,
        "Perfect MAE should be 0.0, got {}",
        result
    );
}

#[test]
fn metrics_regression_perfect_r2() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["regression"]["y_true_reg"]);
    let y_pred = json_to_array1(&fixture["regression"]["y_pred_perfect_reg"]);
    let result = r2_score(&y_true, &y_pred).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "Perfect R2 should be 1.0, got {}",
        result
    );
}

// =============================================================================
// REGRESSION METRICS: Constant mean predictions (R2 = 0)
// =============================================================================

#[test]
fn metrics_regression_constant_mean_r2() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["regression"]["y_true_reg"]);
    let y_pred = json_to_array1(&fixture["regression"]["y_pred_constant_mean"]);
    let result = r2_score(&y_true, &y_pred).unwrap();
    assert!(
        result.abs() < 1e-6,
        "R2 for constant mean predictions should be ~0.0, got {}",
        result
    );
}

#[test]
fn metrics_regression_constant_mean_mse_no_panic() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["regression"]["y_true_reg"]);
    let y_pred = json_to_array1(&fixture["regression"]["y_pred_constant_mean"]);
    let result = mse(&y_true, &y_pred);
    assert!(
        result.is_ok(),
        "MSE should not fail on constant predictions"
    );
    // MSE should be the variance of y_true (since pred = mean)
    assert!(
        result.unwrap() > 0.0,
        "MSE should be positive for non-perfect predictions"
    );
}

// =============================================================================
// METRICS: Empty arrays
// =============================================================================

#[test]
fn metrics_empty_arrays_no_panic() {
    let y_empty = Array1::<f64>::from_vec(vec![]);
    // Metrics on empty arrays should error, not panic
    let acc = accuracy(&y_empty, &y_empty);
    assert!(acc.is_err(), "accuracy on empty arrays should error");
    let m = mse(&y_empty, &y_empty);
    assert!(m.is_err(), "MSE on empty arrays should error");
}

// =============================================================================
// METRICS: Mismatched lengths
// =============================================================================

#[test]
fn metrics_mismatched_lengths_error() {
    let y1 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
    let y2 = Array1::from_vec(vec![0.0, 1.0]);
    let result = accuracy(&y1, &y2);
    assert!(
        result.is_err(),
        "accuracy on mismatched lengths should error"
    );
    let result = mse(&y1, &y2);
    assert!(result.is_err(), "MSE on mismatched lengths should error");
}

// =============================================================================
// CV EDGE CASES: KFold
// =============================================================================

#[test]
fn kfold_basic_split_works() {
    let cv = KFold::new(5);
    let result = cv.split(100, None, None);
    assert!(result.is_ok(), "KFold split should work for 100 samples");
    let folds = result.unwrap();
    assert_eq!(folds.len(), 5);
}

#[test]
fn kfold_k_equals_n_loo_like() {
    // k = n should produce LOO-like splits
    let cv = KFold::new(10);
    let result = cv.split(10, None, None);
    assert!(result.is_ok(), "KFold with k=n should work");
    let folds = result.unwrap();
    assert_eq!(folds.len(), 10);
    for fold in &folds {
        assert_eq!(
            fold.test_indices.len(),
            1,
            "Each fold should have 1 test sample"
        );
        assert_eq!(
            fold.train_indices.len(),
            9,
            "Each fold should have 9 train samples"
        );
    }
}

#[test]
fn kfold_small_n_no_panic() {
    // Very small n (n=3, k=2)
    let cv = KFold::new(2);
    let result = cv.split(3, None, None);
    assert!(result.is_ok(), "KFold with n=3, k=2 should work");
}

// =============================================================================
// CV EDGE CASES: StratifiedKFold
// =============================================================================

#[test]
fn stratified_kfold_basic_works() {
    let cv = StratifiedKFold::new(3);
    // Balanced binary classification
    let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    let result = cv.split(9, Some(&y), None);
    assert!(
        result.is_ok(),
        "StratifiedKFold should work with balanced classes"
    );
}

#[test]
fn stratified_kfold_extreme_imbalance() {
    let cv = StratifiedKFold::new(3);
    let fixture = load_fixture("adversarial/extreme_imbalance.json");
    let y = json_to_array1(&fixture["y_classification"]);
    let n = y.len();
    // 198 class-0, 2 class-1: stratified split should handle this
    let result = cv.split(n, Some(&y), None);
    // May error if a class has fewer samples than folds, but should not panic
    let _ = result;
}

#[test]
fn stratified_kfold_single_class() {
    let cv = StratifiedKFold::new(3);
    let fixture = load_fixture("adversarial/all_same_class.json");
    let y = json_to_array1(&fixture["y_classification"]);
    let n = y.len();
    // All same class: stratified split may error or degrade to regular KFold
    let result = cv.split(n, Some(&y), None);
    let _ = result; // Just verify no panic
}

// =============================================================================
// TRANSFORMER: Not-fitted errors
// =============================================================================

#[test]
fn standard_scaler_not_fitted_transform_errors() {
    let scaler = StandardScaler::new();
    let x = Array2::from_shape_vec((5, 3), (0..15).map(|i| i as f64).collect()).unwrap();
    let result = scaler.transform(&x);
    assert!(result.is_err(), "Transform without fit should error");
}

#[test]
fn minmax_scaler_not_fitted_transform_errors() {
    let scaler = MinMaxScaler::new();
    let x = Array2::from_shape_vec((5, 3), (0..15).map(|i| i as f64).collect()).unwrap();
    let result = scaler.transform(&x);
    assert!(result.is_err(), "Transform without fit should error");
}

#[test]
fn pca_not_fitted_transform_errors() {
    let pca = PCA::new();
    let x = Array2::from_shape_vec((5, 3), (0..15).map(|i| i as f64).collect()).unwrap();
    let result = pca.transform(&x);
    assert!(result.is_err(), "PCA transform without fit should error");
}

// =============================================================================
// PCA: n_components > n_features edge case
// =============================================================================

#[test]
fn pca_n_components_greater_than_features_no_panic() {
    // 10 samples, 3 features, request 10 components
    let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64 + 0.1).collect()).unwrap();
    let mut pca = PCA::new().with_n_components(10);
    let result = pca.fit(&x);
    // Should either error or cap at min(n_samples, n_features)
    let _ = result;
}

#[test]
fn pca_single_sample_no_panic() {
    let x = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let mut pca = PCA::new();
    let _ = pca.fit(&x);
}

#[test]
fn pca_single_feature_no_panic() {
    let x = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
    let mut pca = PCA::new();
    let _ = pca.fit(&x);
}

// =============================================================================
// TRANSFORMER: fit_transform roundtrip on edge cases
// =============================================================================

#[test]
fn standard_scaler_fit_transform_single_sample() {
    let fixture = load_fixture("adversarial/single_sample.json");
    let x = json_to_array2(&fixture["X_single"]);
    let mut scaler = StandardScaler::new();
    // Single sample: std=0 for all features
    let _ = scaler.fit(&x);
}

#[test]
fn minmax_scaler_fit_transform_constant() {
    let fixture = load_fixture("adversarial/constant_features.json");
    let x = json_to_array2(&fixture["X"]);
    let mut scaler = MinMaxScaler::new();
    if scaler.fit(&x).is_ok() {
        let result = scaler.transform(&x);
        // Constant columns: min=max, should not divide by zero
        assert!(
            result.is_ok(),
            "MinMaxScaler transform should handle constant features"
        );
    }
}

// =============================================================================
// METRICS: Average variants for multi-class-like binary
// =============================================================================

#[test]
fn metrics_f1_macro_perfect() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_perfect"]);
    let result = f1_score(&y_true, &y_pred, Average::Macro).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "Perfect macro F1 should be 1.0, got {}",
        result
    );
}

#[test]
fn metrics_f1_weighted_perfect() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_perfect"]);
    let result = f1_score(&y_true, &y_pred, Average::Weighted).unwrap();
    assert!(
        (result - 1.0).abs() < 1e-10,
        "Perfect weighted F1 should be 1.0, got {}",
        result
    );
}

#[test]
fn metrics_f1_macro_worst_no_panic() {
    let fixture = load_fixture("adversarial/degenerate_predictions.json");
    let y_true = json_to_array1(&fixture["classification"]["y_true"]);
    let y_pred = json_to_array1(&fixture["classification"]["y_pred_worst"]);
    let result = f1_score(&y_true, &y_pred, Average::Macro);
    assert!(
        result.is_ok(),
        "Macro F1 should not panic on worst predictions"
    );
}

// =============================================================================
// REGRESSION METRICS: negative R2 is valid
// =============================================================================

#[test]
fn metrics_regression_worse_than_mean_negative_r2() {
    // Predictions that are deliberately far from truth
    let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let result = r2_score(&y_true, &y_pred).unwrap();
    assert!(
        result < 0.0,
        "R2 should be negative for very bad predictions, got {}",
        result
    );
}

// =============================================================================
// METRICS: Single element arrays
// =============================================================================

#[test]
fn metrics_single_element_no_panic() {
    let y_true = Array1::from_vec(vec![1.0]);
    let y_pred = Array1::from_vec(vec![1.0]);
    let acc = accuracy(&y_true, &y_pred);
    assert!(acc.is_ok(), "accuracy on single element should work");
    assert!((acc.unwrap() - 1.0).abs() < 1e-10);

    let m = mse(&y_true, &y_pred);
    assert!(m.is_ok(), "MSE on single element should work");
    assert!(m.unwrap().abs() < 1e-10);
}
