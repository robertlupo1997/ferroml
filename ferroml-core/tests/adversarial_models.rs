//! Adversarial Input Tests for FerroML Models
//!
//! Tests that models handle edge-case and adversarial inputs gracefully:
//! returning errors (not panics) on degenerate data.
//!
//! Fixtures loaded from `benchmarks/fixtures/adversarial/*.json`.

use ndarray::{Array1, Array2};
use serde_json::Value;
use std::path::Path;

use ferroml_core::models::forest::{RandomForestClassifier, RandomForestRegressor};
use ferroml_core::models::knn::{KNeighborsClassifier, KNeighborsRegressor};
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::logistic::LogisticRegression;
use ferroml_core::models::naive_bayes::GaussianNB;
use ferroml_core::models::regularized::RidgeRegression;
use ferroml_core::models::svm::{SVC, SVR};
use ferroml_core::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use ferroml_core::models::Model;
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

/// Parse a JSON array that may contain null values (NaN) into Array2
fn json_to_array2_with_nans(val: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = val
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
        .collect();
    if rows.is_empty() {
        return Array2::from_shape_vec((0, 0), vec![]).unwrap();
    }
    let ncols = rows[0].len();
    let nrows = rows.len();
    Array2::from_shape_vec((nrows, ncols), rows.into_iter().flatten().collect()).unwrap()
}

/// Helper: test that a model fit does not panic (returns Ok or Err, but no panic)
fn fit_does_not_panic(
    model: &mut dyn Model,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> std::result::Result<(), ferroml_core::FerroError> {
    model.fit(x, y)
}

/// Helper: test that a transformer fit does not panic
fn transformer_fit_does_not_panic(
    t: &mut dyn Transformer,
    x: &Array2<f64>,
) -> std::result::Result<(), ferroml_core::FerroError> {
    t.fit(x)
}

// =============================================================================
// EMPTY ARRAYS: X_empty (0,5), y_empty (0,), X_no_features (10,0)
// =============================================================================

#[test]
fn empty_arrays_linear_regression_errors() {
    let fixture = load_fixture("adversarial/empty_arrays.json");
    let x_empty = json_to_array2(&fixture["X_empty"]);
    let y_empty = json_to_array1(&fixture["y_empty"]);
    let mut model = LinearRegression::new();
    assert!(fit_does_not_panic(&mut model, &x_empty, &y_empty).is_err());
}

#[test]
fn empty_arrays_logistic_regression_errors() {
    let fixture = load_fixture("adversarial/empty_arrays.json");
    let x_empty = json_to_array2(&fixture["X_empty"]);
    let y_empty = json_to_array1(&fixture["y_empty"]);
    let mut model = LogisticRegression::new();
    assert!(fit_does_not_panic(&mut model, &x_empty, &y_empty).is_err());
}

#[test]
fn empty_arrays_decision_tree_errors() {
    let fixture = load_fixture("adversarial/empty_arrays.json");
    let x_empty = json_to_array2(&fixture["X_empty"]);
    let y_empty = json_to_array1(&fixture["y_empty"]);
    let mut model = DecisionTreeClassifier::new();
    assert!(fit_does_not_panic(&mut model, &x_empty, &y_empty).is_err());
}

#[test]
fn empty_arrays_standard_scaler_errors() {
    let fixture = load_fixture("adversarial/empty_arrays.json");
    let x_empty = json_to_array2(&fixture["X_empty"]);
    let mut scaler = StandardScaler::new();
    assert!(transformer_fit_does_not_panic(&mut scaler, &x_empty).is_err());
}

#[test]
fn no_features_linear_regression_errors() {
    let fixture = load_fixture("adversarial/empty_arrays.json");
    let x_no_feat = json_to_array2(&fixture["X_no_features"]);
    let y = Array1::from_vec(vec![1.0; 10]);
    let mut model = LinearRegression::new();
    assert!(fit_does_not_panic(&mut model, &x_no_feat, &y).is_err());
}

// =============================================================================
// SINGLE SAMPLE: n=1
// =============================================================================

#[test]
fn single_sample_linear_regression_no_panic() {
    let fixture = load_fixture("adversarial/single_sample.json");
    let x = json_to_array2(&fixture["X_single"]);
    let y = json_to_array1(&fixture["y_single"]);
    let mut model = LinearRegression::new();
    // May succeed or error, but must not panic
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn single_sample_decision_tree_no_panic() {
    let fixture = load_fixture("adversarial/single_sample.json");
    let x = json_to_array2(&fixture["X_single"]);
    let y = json_to_array1(&fixture["y_single"]);
    let mut model = DecisionTreeClassifier::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn single_sample_knn_no_panic() {
    let fixture = load_fixture("adversarial/single_sample.json");
    let x = json_to_array2(&fixture["X_single"]);
    let y = json_to_array1(&fixture["y_single"]);
    let mut model = KNeighborsClassifier::new(1);
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn single_sample_gaussian_nb_no_panic() {
    let fixture = load_fixture("adversarial/single_sample.json");
    let x = json_to_array2(&fixture["X_single"]);
    let y = json_to_array1(&fixture["y_single"]);
    let mut model = GaussianNB::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn single_sample_standard_scaler_no_panic() {
    let fixture = load_fixture("adversarial/single_sample.json");
    let x = json_to_array2(&fixture["X_single"]);
    let mut scaler = StandardScaler::new();
    let _ = transformer_fit_does_not_panic(&mut scaler, &x);
}

// =============================================================================
// SINGLE FEATURE: p=1
// =============================================================================

#[test]
fn single_feature_linear_regression_works() {
    let fixture = load_fixture("adversarial/single_feature.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = LinearRegression::new();
    let result = fit_does_not_panic(&mut model, &x, &y);
    assert!(
        result.is_ok(),
        "LinearRegression should handle single feature: {:?}",
        result.err()
    );
}

#[test]
fn single_feature_decision_tree_classifier_works() {
    let fixture = load_fixture("adversarial/single_feature.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = DecisionTreeClassifier::new();
    let result = fit_does_not_panic(&mut model, &x, &y);
    assert!(
        result.is_ok(),
        "DecisionTreeClassifier should handle single feature: {:?}",
        result.err()
    );
}

#[test]
fn single_feature_logistic_regression_works() {
    let fixture = load_fixture("adversarial/single_feature.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = LogisticRegression::new();
    let result = fit_does_not_panic(&mut model, &x, &y);
    assert!(
        result.is_ok(),
        "LogisticRegression should handle single feature: {:?}",
        result.err()
    );
}

// =============================================================================
// CONSTANT FEATURES: no variance in some columns
// =============================================================================

#[test]
fn constant_features_standard_scaler_no_panic() {
    let fixture = load_fixture("adversarial/constant_features.json");
    let x = json_to_array2(&fixture["X"]);
    let mut scaler = StandardScaler::new();
    // Fit should succeed; constant columns get std=0 but no div-by-zero panic
    let _ = transformer_fit_does_not_panic(&mut scaler, &x);
}

#[test]
fn constant_features_minmax_scaler_no_panic() {
    let fixture = load_fixture("adversarial/constant_features.json");
    let x = json_to_array2(&fixture["X"]);
    let mut scaler = MinMaxScaler::new();
    let _ = transformer_fit_does_not_panic(&mut scaler, &x);
}

#[test]
fn constant_features_linear_regression_no_panic() {
    let fixture = load_fixture("adversarial/constant_features.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = LinearRegression::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn constant_features_decision_tree_no_panic() {
    let fixture = load_fixture("adversarial/constant_features.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = DecisionTreeClassifier::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

// =============================================================================
// COLLINEAR FEATURES: near-perfect collinearity
// =============================================================================

#[test]
fn collinear_features_linear_regression_no_panic() {
    let fixture = load_fixture("adversarial/collinear_features.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = LinearRegression::new();
    // Singular matrix: may return error or succeed with poor coefficients
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn collinear_features_ridge_regression_handles_gracefully() {
    let fixture = load_fixture("adversarial/collinear_features.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = RidgeRegression::new(1.0);
    // Ridge regularization should handle collinearity
    let result = fit_does_not_panic(&mut model, &x, &y);
    assert!(
        result.is_ok(),
        "RidgeRegression with alpha=1.0 should handle collinearity: {:?}",
        result.err()
    );
}

#[test]
fn collinear_features_decision_tree_handles_gracefully() {
    let fixture = load_fixture("adversarial/collinear_features.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = DecisionTreeRegressor::new();
    let result = fit_does_not_panic(&mut model, &x, &y);
    assert!(
        result.is_ok(),
        "DecisionTreeRegressor should handle collinearity: {:?}",
        result.err()
    );
}

// =============================================================================
// NEAR ZERO VARIANCE: variance ~1e-20 in one feature
// =============================================================================

#[test]
fn near_zero_variance_standard_scaler_no_panic() {
    let fixture = load_fixture("adversarial/near_zero_variance.json");
    let x = json_to_array2(&fixture["X"]);
    let mut scaler = StandardScaler::new();
    // Must not panic on near-zero variance
    let _ = transformer_fit_does_not_panic(&mut scaler, &x);
}

#[test]
fn near_zero_variance_linear_regression_no_panic() {
    let fixture = load_fixture("adversarial/near_zero_variance.json");
    let x = json_to_array2(&fixture["X"]);
    // near_zero_variance doesn't have y_regression; use synthetic y
    let n = x.nrows();
    let y = Array1::from_vec((0..n).map(|i| i as f64).collect());
    let mut model = LinearRegression::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn near_zero_variance_random_forest_no_panic() {
    let fixture = load_fixture("adversarial/near_zero_variance.json");
    let x = json_to_array2(&fixture["X"]);
    let n = x.nrows();
    let y = Array1::from_vec((0..n).map(|i| (i % 2) as f64).collect());
    let mut model = RandomForestClassifier::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

// =============================================================================
// EXTREME VALUES: ~1e15 and ~1e-15
// =============================================================================

#[test]
fn extreme_values_large_linear_regression_no_panic() {
    let fixture = load_fixture("adversarial/extreme_values.json");
    let x = json_to_array2(&fixture["X_large"]);
    let y = json_to_array1(&fixture["y_large"]);
    let mut model = LinearRegression::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn extreme_values_small_linear_regression_no_panic() {
    let fixture = load_fixture("adversarial/extreme_values.json");
    let x = json_to_array2(&fixture["X_small"]);
    let y = json_to_array1(&fixture["y_small"]);
    let mut model = LinearRegression::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn extreme_values_mixed_standard_scaler_no_panic() {
    let fixture = load_fixture("adversarial/extreme_values.json");
    let x = json_to_array2(&fixture["X_mixed"]);
    let mut scaler = StandardScaler::new();
    let _ = transformer_fit_does_not_panic(&mut scaler, &x);
}

#[test]
fn extreme_values_mixed_decision_tree_no_panic() {
    let fixture = load_fixture("adversarial/extreme_values.json");
    let x = json_to_array2(&fixture["X_mixed"]);
    let y = json_to_array1(&fixture["y_mixed"]);
    let mut model = DecisionTreeRegressor::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

// =============================================================================
// EXTREME IMBALANCE: 99:1 class ratio (198 class-0, 2 class-1)
// =============================================================================

#[test]
fn extreme_imbalance_logistic_regression_no_panic() {
    let fixture = load_fixture("adversarial/extreme_imbalance.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = LogisticRegression::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn extreme_imbalance_decision_tree_no_panic() {
    let fixture = load_fixture("adversarial/extreme_imbalance.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = DecisionTreeClassifier::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn extreme_imbalance_random_forest_no_panic() {
    let fixture = load_fixture("adversarial/extreme_imbalance.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = RandomForestClassifier::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn extreme_imbalance_gaussian_nb_no_panic() {
    let fixture = load_fixture("adversarial/extreme_imbalance.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = GaussianNB::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

// =============================================================================
// ALL SAME CLASS: single-class target (all 0)
// =============================================================================

#[test]
fn all_same_class_logistic_regression_no_panic() {
    let fixture = load_fixture("adversarial/all_same_class.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = LogisticRegression::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn all_same_class_decision_tree_no_panic() {
    let fixture = load_fixture("adversarial/all_same_class.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = DecisionTreeClassifier::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn all_same_class_gaussian_nb_no_panic() {
    let fixture = load_fixture("adversarial/all_same_class.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = GaussianNB::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

// =============================================================================
// HIGH DIMENSIONAL: p(500) >> n(50)
// =============================================================================

#[test]
fn high_dimensional_linear_regression_no_panic() {
    let fixture = load_fixture("adversarial/high_dimensional.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = LinearRegression::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
#[ignore = "slow in debug mode: 500x500 matrix solve takes ~12min unoptimized"]
fn high_dimensional_ridge_regression_no_panic() {
    let fixture = load_fixture("adversarial/high_dimensional.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = RidgeRegression::new(1.0);
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn high_dimensional_decision_tree_no_panic() {
    let fixture = load_fixture("adversarial/high_dimensional.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = DecisionTreeClassifier::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

// =============================================================================
// MISSING VALUES (NaN): imputer robustness, model NaN propagation
// =============================================================================

#[test]
fn missing_values_simple_imputer_no_panic() {
    let fixture = load_fixture("adversarial/missing_values.json");
    let x_nans = json_to_array2_with_nans(&fixture["X_with_nans"]);
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};
    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
    let result = transformer_fit_does_not_panic(&mut imputer, &x_nans);
    // Should be able to fit on data with NaNs
    let _ = result;
}

#[test]
fn missing_values_all_nan_column_imputer_no_panic() {
    let fixture = load_fixture("adversarial/missing_values.json");
    let x_all_nan_col = json_to_array2_with_nans(&fixture["X_all_nan_col"]);
    use ferroml_core::preprocessing::imputers::{ImputeStrategy, SimpleImputer};
    let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
    // All-NaN column: should handle gracefully (error or impute with 0)
    let _ = transformer_fit_does_not_panic(&mut imputer, &x_all_nan_col);
}

#[test]
fn missing_values_linear_regression_nan_input_no_panic() {
    let fixture = load_fixture("adversarial/missing_values.json");
    let x_nans = json_to_array2_with_nans(&fixture["X_with_nans"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = LinearRegression::new();
    // NaN in features: should error or handle, not panic
    let _ = fit_does_not_panic(&mut model, &x_nans, &y);
}

// =============================================================================
// ADDITIONAL EDGE CASES: SVM, KNN, Random Forest on adversarial data
// =============================================================================

#[test]
fn svc_single_feature_no_panic() {
    let fixture = load_fixture("adversarial/single_feature.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = SVC::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn svr_single_feature_no_panic() {
    let fixture = load_fixture("adversarial/single_feature.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = SVR::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn knn_regressor_constant_features_no_panic() {
    let fixture = load_fixture("adversarial/constant_features.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = KNeighborsRegressor::new(5);
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn random_forest_regressor_collinear_no_panic() {
    let fixture = load_fixture("adversarial/collinear_features.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = RandomForestRegressor::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn knn_classifier_all_same_class_no_panic() {
    let fixture = load_fixture("adversarial/all_same_class.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = KNeighborsClassifier::new(3);
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn svc_extreme_imbalance_no_panic() {
    let fixture = load_fixture("adversarial/extreme_imbalance.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = SVC::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn svr_extreme_values_large_no_panic() {
    let fixture = load_fixture("adversarial/extreme_values.json");
    let x = json_to_array2(&fixture["X_large"]);
    let y = json_to_array1(&fixture["y_large"]);
    let mut model = SVR::new();
    let _ = fit_does_not_panic(&mut model, &x, &y);
}

#[test]
fn minmax_scaler_extreme_values_no_panic() {
    let fixture = load_fixture("adversarial/extreme_values.json");
    let x = json_to_array2(&fixture["X_mixed"]);
    let mut scaler = MinMaxScaler::new();
    let _ = transformer_fit_does_not_panic(&mut scaler, &x);
}

// =============================================================================
// FIT + PREDICT ROUNDTRIP: verify predict also does not panic
// =============================================================================

#[test]
fn decision_tree_fit_predict_single_feature() {
    let fixture = load_fixture("adversarial/single_feature.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = DecisionTreeClassifier::new();
    if model.fit(&x, &y).is_ok() {
        let pred = model.predict(&x);
        assert!(pred.is_ok(), "predict should not fail after successful fit");
    }
}

#[test]
fn random_forest_fit_predict_collinear() {
    let fixture = load_fixture("adversarial/collinear_features.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_regression"]);
    let mut model = RandomForestRegressor::new();
    if model.fit(&x, &y).is_ok() {
        let pred = model.predict(&x);
        assert!(pred.is_ok(), "predict should not fail after successful fit");
    }
}

#[test]
fn knn_fit_predict_extreme_imbalance() {
    let fixture = load_fixture("adversarial/extreme_imbalance.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = KNeighborsClassifier::new(5);
    if model.fit(&x, &y).is_ok() {
        let pred = model.predict(&x);
        assert!(pred.is_ok(), "predict should not fail after successful fit");
    }
}

#[test]
fn gaussian_nb_fit_predict_all_same_class() {
    let fixture = load_fixture("adversarial/all_same_class.json");
    let x = json_to_array2(&fixture["X"]);
    let y = json_to_array1(&fixture["y_classification"]);
    let mut model = GaussianNB::new();
    if model.fit(&x, &y).is_ok() {
        let pred = model.predict(&x);
        assert!(pred.is_ok(), "predict should not fail after successful fit");
        // All predictions should be class 0 (the only class)
        if let Ok(predictions) = pred {
            for &p in predictions.iter() {
                assert!((p - 0.0).abs() < 0.5, "Expected class 0, got {}", p);
            }
        }
    }
}

#[test]
fn standard_scaler_fit_transform_constant_features() {
    let fixture = load_fixture("adversarial/constant_features.json");
    let x = json_to_array2(&fixture["X"]);
    let mut scaler = StandardScaler::new();
    if scaler.fit(&x).is_ok() {
        let result = scaler.transform(&x);
        // Transform should succeed; constant features become 0 (or NaN depending on impl)
        assert!(
            result.is_ok(),
            "transform should not fail after successful fit"
        );
    }
}
