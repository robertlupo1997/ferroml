//! Multi-Output Prediction Tests
//!
//! Phase 23 of FerroML testing plan - comprehensive tests for:
//! - Single-model prediction shape validation
//! - Multi-target workaround pattern (fitting separate models per target)
//! - Per-output metric computation
//! - Shape validation edge cases
//! - Prediction consistency and determinism

#![allow(unused_imports)]
#![allow(dead_code)]

use ndarray::{array, s, Array1, Array2, Axis};

use crate::metrics::{mae, mse, r2_score};
use crate::models::{
    DecisionTreeRegressor, KNeighborsRegressor, LinearRegression, Model, RidgeRegression,
};

// ============================================================================
// HELPER: Generate synthetic multi-output data
// ============================================================================

/// Generate a simple multi-output dataset: Y has `n_outputs` columns.
/// Each output is a linear combination of features plus noise.
fn make_multioutput_data(
    n_samples: usize,
    n_features: usize,
    n_outputs: usize,
) -> (Array2<f64>, Array2<f64>) {
    let mut x = Array2::zeros((n_samples, n_features));
    let mut y = Array2::zeros((n_samples, n_outputs));

    for i in 0..n_samples {
        for j in 0..n_features {
            // Deterministic pseudo-random based on indices
            x[[i, j]] = ((i * 7 + j * 13 + 3) % 100) as f64 / 50.0 - 1.0;
        }
        for k in 0..n_outputs {
            let mut val = 0.0;
            for j in 0..n_features {
                // Different coefficient per output
                val += x[[i, j]] * ((k + 1) as f64) * (1.0 + j as f64 * 0.5);
            }
            // Small deterministic noise
            val += ((i * 3 + k * 7) % 17) as f64 / 170.0;
            y[[i, k]] = val;
        }
    }
    (x, y)
}

/// Fit one model per target column and return predictions for each output.
fn multioutput_predict<F>(
    make_model: F,
    x_train: &Array2<f64>,
    y_train: &Array2<f64>,
    x_test: &Array2<f64>,
) -> Vec<Array1<f64>>
where
    F: Fn() -> Box<dyn Model>,
{
    let n_outputs = y_train.ncols();
    let mut predictions = Vec::with_capacity(n_outputs);

    for k in 0..n_outputs {
        let y_k = y_train.column(k).to_owned();
        let mut model = make_model();
        model.fit(x_train, &y_k).expect("fit should succeed");
        let preds = model.predict(x_test).expect("predict should succeed");
        predictions.push(preds);
    }
    predictions
}

// ============================================================================
// SINGLE-MODEL PREDICTION SHAPE VALIDATION
// ============================================================================

#[test]
fn test_predict_shape_matches_n_samples_linear() {
    let (x, y_multi) = make_multioutput_data(50, 3, 4);
    let y = y_multi.column(0).to_owned();
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    let preds = model.predict(&x).unwrap();
    assert_eq!(
        preds.len(),
        50,
        "predict output should have n_samples elements"
    );
}

#[test]
fn test_predict_shape_matches_n_samples_tree() {
    let (x, y_multi) = make_multioutput_data(40, 5, 3);
    let y = y_multi.column(1).to_owned();
    let mut model = DecisionTreeRegressor::new();
    model.fit(&x, &y).unwrap();
    let preds = model.predict(&x).unwrap();
    assert_eq!(preds.len(), 40);
}

#[test]
fn test_predict_shape_matches_n_samples_knn() {
    let (x, y_multi) = make_multioutput_data(30, 4, 2);
    let y = y_multi.column(0).to_owned();
    let mut model = KNeighborsRegressor::new(5);
    model.fit(&x, &y).unwrap();
    let preds = model.predict(&x).unwrap();
    assert_eq!(preds.len(), 30);
}

#[test]
fn test_predict_shape_single_sample() {
    let (x, y_multi) = make_multioutput_data(20, 3, 2);
    let y = y_multi.column(0).to_owned();
    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();
    let x_single = x.slice(s![0..1, ..]).to_owned();
    let preds = model.predict(&x_single).unwrap();
    assert_eq!(
        preds.len(),
        1,
        "single sample should produce single prediction"
    );
}

#[test]
fn test_predict_shape_many_features() {
    // Use KNN which handles many features without rank-deficiency issues
    let (x, y_multi) = make_multioutput_data(30, 20, 2);
    let y = y_multi.column(0).to_owned();
    let mut model = KNeighborsRegressor::new(3);
    model.fit(&x, &y).unwrap();
    let preds = model.predict(&x).unwrap();
    assert_eq!(preds.len(), 30);
}

// ============================================================================
// MULTI-TARGET WORKAROUND PATTERN
// ============================================================================

#[test]
fn test_multioutput_workaround_linear_2_outputs() {
    let (x, y_multi) = make_multioutput_data(60, 4, 2);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);
    assert_eq!(preds.len(), 2);
    for p in &preds {
        assert_eq!(p.len(), 60);
    }
}

#[test]
fn test_multioutput_workaround_linear_5_outputs() {
    let (x, y_multi) = make_multioutput_data(50, 3, 5);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);
    assert_eq!(preds.len(), 5);
    for (k, p) in preds.iter().enumerate() {
        assert_eq!(p.len(), 50, "output {} should have 50 predictions", k);
    }
}

#[test]
fn test_multioutput_workaround_tree() {
    let (x, y_multi) = make_multioutput_data(40, 3, 3);
    let preds = multioutput_predict(|| Box::new(DecisionTreeRegressor::new()), &x, &y_multi, &x);
    assert_eq!(preds.len(), 3);
    for p in &preds {
        assert_eq!(p.len(), 40);
    }
}

#[test]
fn test_multioutput_workaround_knn() {
    let (x, y_multi) = make_multioutput_data(40, 3, 2);
    let preds = multioutput_predict(|| Box::new(KNeighborsRegressor::new(3)), &x, &y_multi, &x);
    assert_eq!(preds.len(), 2);
    for p in &preds {
        assert_eq!(p.len(), 40);
    }
}

#[test]
fn test_multioutput_train_test_split() {
    let (x, y_multi) = make_multioutput_data(60, 4, 3);
    let x_train = x.slice(s![..40, ..]).to_owned();
    let y_train = y_multi.slice(s![..40, ..]).to_owned();
    let x_test = x.slice(s![40.., ..]).to_owned();

    let preds = multioutput_predict(
        || Box::new(LinearRegression::new()),
        &x_train,
        &y_train,
        &x_test,
    );
    assert_eq!(preds.len(), 3);
    for p in &preds {
        assert_eq!(p.len(), 20, "test set has 20 samples");
    }
}

#[test]
fn test_multioutput_independent_models_differ() {
    // Each output has different coefficients, so models should produce different predictions
    let (x, y_multi) = make_multioutput_data(50, 3, 3);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    // Predictions for different outputs should generally differ
    let diff_01: f64 = (&preds[0] - &preds[1]).mapv(|v| v.abs()).sum();
    let diff_02: f64 = (&preds[0] - &preds[2]).mapv(|v| v.abs()).sum();
    assert!(
        diff_01 > 1e-6,
        "different outputs should produce different predictions"
    );
    assert!(
        diff_02 > 1e-6,
        "different outputs should produce different predictions"
    );
}

// ============================================================================
// PER-OUTPUT METRIC COMPUTATION
// ============================================================================

#[test]
fn test_per_output_mse() {
    let (x, y_multi) = make_multioutput_data(50, 4, 3);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    for k in 0..3 {
        let y_true = y_multi.column(k).to_owned();
        let mse_val = mse(&y_true, &preds[k]).unwrap();
        assert!(
            mse_val >= 0.0,
            "MSE should be non-negative for output {}",
            k
        );
        // Linear model on linear data should fit reasonably well
        assert!(
            mse_val < 1.0,
            "MSE should be small for output {} on linear data, got {}",
            k,
            mse_val
        );
    }
}

#[test]
fn test_per_output_r2_score() {
    let (x, y_multi) = make_multioutput_data(50, 4, 3);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    for k in 0..3 {
        let y_true = y_multi.column(k).to_owned();
        let r2 = r2_score(&y_true, &preds[k]).unwrap();
        assert!(
            r2 > 0.9,
            "R2 should be high for output {} on linear data, got {}",
            k,
            r2
        );
    }
}

#[test]
fn test_per_output_mae() {
    let (x, y_multi) = make_multioutput_data(50, 4, 3);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    for k in 0..3 {
        let y_true = y_multi.column(k).to_owned();
        let mae_val = mae(&y_true, &preds[k]).unwrap();
        assert!(
            mae_val >= 0.0,
            "MAE should be non-negative for output {}",
            k
        );
        assert!(
            mae_val < 1.0,
            "MAE should be small for output {}, got {}",
            k,
            mae_val
        );
    }
}

#[test]
fn test_average_metric_across_outputs() {
    let (x, y_multi) = make_multioutput_data(50, 4, 4);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    let mse_values: Vec<f64> = (0..4)
        .map(|k| {
            let y_true = y_multi.column(k).to_owned();
            mse(&y_true, &preds[k]).unwrap()
        })
        .collect();

    let avg_mse: f64 = mse_values.iter().sum::<f64>() / mse_values.len() as f64;
    assert!(avg_mse >= 0.0);
    assert!(
        avg_mse < 1.0,
        "average MSE should be small, got {}",
        avg_mse
    );

    // Uniform averaging: each output contributes equally
    for &v in &mse_values {
        assert!(v >= 0.0);
    }
}

// ============================================================================
// SHAPE VALIDATION EDGE CASES
// ============================================================================

#[test]
fn test_single_feature_multioutput() {
    let (x, y_multi) = make_multioutput_data(30, 1, 3);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);
    assert_eq!(preds.len(), 3);
    for p in &preds {
        assert_eq!(p.len(), 30);
    }
}

#[test]
fn test_many_outputs() {
    let (x, y_multi) = make_multioutput_data(40, 3, 10);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);
    assert_eq!(preds.len(), 10);
}

#[test]
fn test_single_output_is_same_as_direct() {
    // With 1 output, the workaround should give same result as direct fit
    let (x, y_multi) = make_multioutput_data(40, 3, 1);
    let y = y_multi.column(0).to_owned();

    let mut direct_model = LinearRegression::new();
    direct_model.fit(&x, &y).unwrap();
    let direct_preds = direct_model.predict(&x).unwrap();

    let workaround_preds =
        multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    assert_eq!(workaround_preds.len(), 1);
    for i in 0..direct_preds.len() {
        assert!(
            (direct_preds[i] - workaround_preds[0][i]).abs() < 1e-10,
            "direct and workaround should match at index {}",
            i
        );
    }
}

#[test]
fn test_two_samples_multioutput() {
    // Minimal dataset with 2 samples — use KNN(1) which handles small datasets
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y_multi = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

    let preds = multioutput_predict(|| Box::new(KNeighborsRegressor::new(1)), &x, &y_multi, &x);
    assert_eq!(preds.len(), 2);
    for p in &preds {
        assert_eq!(p.len(), 2);
        assert!(
            p.iter().all(|v| v.is_finite()),
            "all predictions should be finite"
        );
    }
}

// ============================================================================
// CONSISTENCY AND DETERMINISM
// ============================================================================

#[test]
fn test_deterministic_multioutput_linear() {
    let (x, y_multi) = make_multioutput_data(40, 3, 3);

    let preds1 = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);
    let preds2 = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    for k in 0..3 {
        for i in 0..40 {
            assert!(
                (preds1[k][i] - preds2[k][i]).abs() < 1e-12,
                "predictions should be deterministic for output {} sample {}",
                k,
                i
            );
        }
    }
}

#[test]
fn test_deterministic_multioutput_tree() {
    let (x, y_multi) = make_multioutput_data(40, 3, 2);

    let preds1 = multioutput_predict(|| Box::new(DecisionTreeRegressor::new()), &x, &y_multi, &x);
    let preds2 = multioutput_predict(|| Box::new(DecisionTreeRegressor::new()), &x, &y_multi, &x);

    for k in 0..2 {
        for i in 0..40 {
            assert!(
                (preds1[k][i] - preds2[k][i]).abs() < 1e-12,
                "tree predictions should be deterministic"
            );
        }
    }
}

#[test]
fn test_predictions_finite() {
    let (x, y_multi) = make_multioutput_data(30, 4, 3);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    for (k, p) in preds.iter().enumerate() {
        for (i, &v) in p.iter().enumerate() {
            assert!(
                v.is_finite(),
                "prediction should be finite at output {} sample {}",
                k,
                i
            );
        }
    }
}

#[test]
fn test_multioutput_combined_into_matrix() {
    // Demonstrate combining per-output predictions into a 2D array
    let (x, y_multi) = make_multioutput_data(30, 3, 4);
    let preds = multioutput_predict(|| Box::new(LinearRegression::new()), &x, &y_multi, &x);

    let n_samples = 30;
    let n_outputs = 4;
    let mut y_pred_combined = Array2::zeros((n_samples, n_outputs));
    for (k, p) in preds.iter().enumerate() {
        y_pred_combined.column_mut(k).assign(p);
    }

    assert_eq!(y_pred_combined.shape(), &[30, 4]);

    // Verify round-trip: each column matches its source prediction
    for k in 0..n_outputs {
        let col = y_pred_combined.column(k).to_owned();
        for i in 0..n_samples {
            assert!((col[i] - preds[k][i]).abs() < 1e-15);
        }
    }
}

#[test]
fn test_multioutput_different_models_per_output() {
    // Use different model types for different outputs
    let (x, y_multi) = make_multioutput_data(50, 3, 2);

    // Output 0: linear, Output 1: tree
    let y0 = y_multi.column(0).to_owned();
    let y1 = y_multi.column(1).to_owned();

    let mut model0 = LinearRegression::new();
    model0.fit(&x, &y0).unwrap();
    let preds0 = model0.predict(&x).unwrap();

    let mut model1 = DecisionTreeRegressor::new();
    model1.fit(&x, &y1).unwrap();
    let preds1 = model1.predict(&x).unwrap();

    assert_eq!(preds0.len(), 50);
    assert_eq!(preds1.len(), 50);

    // Both should produce reasonable results
    let r2_0 = r2_score(&y0, &preds0).unwrap();
    let r2_1 = r2_score(&y1, &preds1).unwrap();
    assert!(r2_0 > 0.8, "linear model R2 = {}", r2_0);
    assert!(r2_1 > 0.8, "tree model R2 = {}", r2_1);
}

#[test]
fn test_multioutput_ridge_regularized() {
    let (x, y_multi) = make_multioutput_data(50, 4, 3);
    let preds = multioutput_predict(|| Box::new(RidgeRegression::new(1.0)), &x, &y_multi, &x);
    assert_eq!(preds.len(), 3);
    for (k, p) in preds.iter().enumerate() {
        assert_eq!(p.len(), 50);
        let y_true = y_multi.column(k).to_owned();
        let r2 = r2_score(&y_true, p).unwrap();
        assert!(
            r2 > 0.5,
            "Ridge R2 for output {} should be reasonable, got {}",
            k,
            r2
        );
    }
}
