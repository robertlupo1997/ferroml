//! Comprehensive NaN/Inf validation tests
//!
//! This module provides exhaustive testing for handling of NaN (Not a Number) and
//! Inf (Infinity) values in features, targets, and preprocessing pipelines.
//!
//! # Overview
//!
//! ML models and transformers must handle non-finite values consistently:
//! - Either reject them with a clear `FerroError::InvalidInput` or `FerroError::NumericalError`
//! - Or handle them gracefully (producing finite outputs for clean data)
//!
//! # Test Categories
//!
//! 1. **NaN in Features**: Tests for NaN values scattered throughout feature matrices
//! 2. **Inf in Features**: Tests for positive and negative infinity in features
//! 3. **Mixed NaN/Inf**: Combined scenarios with both types of non-finite values
//! 4. **NaN/Inf in Targets**: Tests for non-finite values in target arrays
//! 5. **Transformer Handling**: Tests for preprocessing transformers
//! 6. **Propagation Tests**: Verify models don't silently propagate NaN/Inf
//!
//! # Usage
//!
//! ```ignore
//! use ferroml_core::testing::nan_inf_validation::*;
//! use ferroml_core::models::LinearRegression;
//!
//! let model = LinearRegression::new();
//! let results = validate_nan_inf_handling(model);
//! assert!(results.iter().all(|r| r.passed));
//! ```

use crate::models::Model;
use crate::preprocessing::Transformer;
use ndarray::{Array1, Array2};
use std::time::Duration;

use super::{CheckCategory, CheckResult};

// ============================================================================
// Helper Functions
// ============================================================================

/// Run a check function and catch panics
fn run_nan_inf_check<F>(name: &'static str, check_fn: F) -> CheckResult
where
    F: FnOnce() -> (bool, Option<String>),
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(check_fn)) {
        Ok((passed, message)) => CheckResult {
            name,
            passed,
            message,
            duration: Duration::ZERO,
            category: CheckCategory::InputValidation,
        },
        Err(panic_info) => {
            let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            CheckResult {
                name,
                passed: false,
                message: Some(format!("Check panicked: {}", panic_msg)),
                duration: Duration::ZERO,
                category: CheckCategory::InputValidation,
            }
        }
    }
}

/// Check if predictions are all finite
fn predictions_are_finite(preds: &Array1<f64>) -> bool {
    preds.iter().all(|v| v.is_finite())
}

// ============================================================================
// NaN in Features Tests
// ============================================================================

/// Test NaN at the beginning of feature matrix
pub fn check_nan_at_start<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_nan_at_start", move || {
        let mut x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        x[[0, 0]] = f64::NAN;
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None), // Rejection is acceptable
            Ok(_) => {
                // If fit succeeds, verify clean predictions work
                let x_clean = Array2::from_shape_fn((5, 3), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    Ok(_) => (
                        false,
                        Some("Model fit with NaN but produces non-finite predictions".to_string()),
                    ),
                    Err(_) => (
                        false,
                        Some("Model fit with NaN but predict fails on clean data".to_string()),
                    ),
                }
            }
        }
    })
}

/// Test NaN at the end of feature matrix
pub fn check_nan_at_end<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_nan_at_end", move || {
        let mut x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        x[[9, 2]] = f64::NAN;
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((5, 3), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    Ok(_) => (
                        false,
                        Some("Non-finite predictions from model fit with trailing NaN".to_string()),
                    ),
                    Err(_) => (false, Some("Predict failed after NaN fit".to_string())),
                }
            }
        }
    })
}

/// Test multiple NaN values in the same row
pub fn check_nan_entire_row<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_nan_entire_row", move || {
        let mut x = Array2::from_shape_fn((10, 4), |(i, j)| (i + j + 1) as f64);
        // Fill entire row with NaN
        for j in 0..4 {
            x[[5, j]] = f64::NAN;
        }
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((3, 4), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    _ => (
                        false,
                        Some("Model doesn't handle entire NaN row properly".to_string()),
                    ),
                }
            }
        }
    })
}

/// Test multiple NaN values in the same column
pub fn check_nan_entire_column<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_nan_entire_column", move || {
        let mut x = Array2::from_shape_fn((10, 4), |(i, j)| (i + j + 1) as f64);
        // Fill entire column with NaN
        for i in 0..10 {
            x[[i, 2]] = f64::NAN;
        }
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                // Even if fit succeeds, model should handle this gracefully
                let mut x_test = Array2::from_shape_fn((3, 4), |(i, j)| (i + j + 1) as f64);
                for i in 0..3 {
                    x_test[[i, 2]] = f64::NAN;
                }
                // Just check it doesn't panic
                let _ = model.predict(&x_test);
                (true, None)
            }
        }
    })
}

/// Test scattered NaN values throughout the matrix
pub fn check_nan_scattered<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_nan_scattered", move || {
        let mut x = Array2::from_shape_fn((20, 5), |(i, j)| (i + j + 1) as f64);
        // Scatter NaN values
        x[[3, 1]] = f64::NAN;
        x[[7, 3]] = f64::NAN;
        x[[12, 0]] = f64::NAN;
        x[[15, 4]] = f64::NAN;
        x[[18, 2]] = f64::NAN;
        let y = Array1::from_shape_fn(20, |i| (i + 1) as f64);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((5, 5), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    _ => (
                        false,
                        Some("Model doesn't handle scattered NaN properly".to_string()),
                    ),
                }
            }
        }
    })
}

// ============================================================================
// Inf in Features Tests
// ============================================================================

/// Test positive infinity in features
pub fn check_positive_inf<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_positive_inf", move || {
        let mut x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        x[[5, 1]] = f64::INFINITY;
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((5, 3), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    Ok(_) => (
                        false,
                        Some("Model produces non-finite output after +Inf fit".to_string()),
                    ),
                    Err(_) => (false, Some("Predict failed after +Inf fit".to_string())),
                }
            }
        }
    })
}

/// Test negative infinity in features
pub fn check_negative_inf<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_negative_inf", move || {
        let mut x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        x[[5, 1]] = f64::NEG_INFINITY;
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((5, 3), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    Ok(_) => (
                        false,
                        Some("Model produces non-finite output after -Inf fit".to_string()),
                    ),
                    Err(_) => (false, Some("Predict failed after -Inf fit".to_string())),
                }
            }
        }
    })
}

/// Test both positive and negative infinity together
pub fn check_both_inf<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_both_inf", move || {
        let mut x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        x[[3, 0]] = f64::INFINITY;
        x[[7, 2]] = f64::NEG_INFINITY;
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((5, 3), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    _ => (
                        false,
                        Some("Model doesn't handle ±Inf properly".to_string()),
                    ),
                }
            }
        }
    })
}

/// Test entire column of infinity
pub fn check_inf_entire_column<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_inf_entire_column", move || {
        let mut x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        for i in 0..10 {
            x[[i, 1]] = f64::INFINITY;
        }
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => (true, None), // If accepted, that's a policy decision
        }
    })
}

// ============================================================================
// Mixed NaN/Inf Tests
// ============================================================================

/// Test mixed NaN and Inf in the same matrix
pub fn check_mixed_nan_inf<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_mixed_nan_inf", move || {
        let mut x = Array2::from_shape_fn((15, 4), |(i, j)| (i + j + 1) as f64);
        x[[2, 0]] = f64::NAN;
        x[[5, 1]] = f64::INFINITY;
        x[[8, 2]] = f64::NEG_INFINITY;
        x[[11, 3]] = f64::NAN;
        let y = Array1::from_shape_fn(15, |i| (i + 1) as f64);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((5, 4), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    _ => (
                        false,
                        Some("Model doesn't handle mixed NaN/Inf properly".to_string()),
                    ),
                }
            }
        }
    })
}

/// Test NaN and Inf in the same row
pub fn check_nan_inf_same_row<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_nan_inf_same_row", move || {
        let mut x = Array2::from_shape_fn((10, 4), |(i, j)| (i + j + 1) as f64);
        x[[5, 0]] = f64::NAN;
        x[[5, 2]] = f64::INFINITY;
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((3, 4), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    _ => (
                        false,
                        Some("Model doesn't handle NaN/Inf in same row".to_string()),
                    ),
                }
            }
        }
    })
}

/// Test NaN and Inf in the same column
pub fn check_nan_inf_same_column<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_nan_inf_same_column", move || {
        let mut x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        x[[3, 1]] = f64::NAN;
        x[[7, 1]] = f64::INFINITY;
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => {
                let x_clean = Array2::from_shape_fn((3, 3), |(i, j)| (i + j + 1) as f64);
                match model.predict(&x_clean) {
                    Ok(preds) if predictions_are_finite(&preds) => (true, None),
                    _ => (
                        false,
                        Some("Model doesn't handle NaN/Inf in same column".to_string()),
                    ),
                }
            }
        }
    })
}

// ============================================================================
// NaN/Inf in Targets Tests
// ============================================================================

/// Test NaN in target values
pub fn check_nan_in_target<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_nan_in_target", move || {
        let x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        let mut y = Array1::from_shape_fn(10, |i| (i + 1) as f64);
        y[5] = f64::NAN;

        match model.fit(&x, &y) {
            Err(_) => (true, None), // Should reject NaN in target
            Ok(_) => (
                false,
                Some("Model should reject NaN in target values".to_string()),
            ),
        }
    })
}

/// Test Inf in target values
pub fn check_inf_in_target<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_inf_in_target", move || {
        let x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        let mut y = Array1::from_shape_fn(10, |i| (i + 1) as f64);
        y[5] = f64::INFINITY;

        match model.fit(&x, &y) {
            Err(_) => (true, None), // Should reject Inf in target
            Ok(_) => (
                false,
                Some("Model should reject Inf in target values".to_string()),
            ),
        }
    })
}

/// Test negative Inf in target values
pub fn check_neg_inf_in_target<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_neg_inf_in_target", move || {
        let x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        let mut y = Array1::from_shape_fn(10, |i| (i + 1) as f64);
        y[5] = f64::NEG_INFINITY;

        match model.fit(&x, &y) {
            Err(_) => (true, None), // Should reject -Inf in target
            Ok(_) => (
                false,
                Some("Model should reject -Inf in target values".to_string()),
            ),
        }
    })
}

/// Test all targets are NaN
pub fn check_all_nan_targets<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_all_nan_targets", move || {
        let x = Array2::from_shape_fn((10, 3), |(i, j)| (i + j + 1) as f64);
        let y = Array1::from_elem(10, f64::NAN);

        match model.fit(&x, &y) {
            Err(_) => (true, None),
            Ok(_) => (
                false,
                Some("Model should reject all-NaN targets".to_string()),
            ),
        }
    })
}

// ============================================================================
// Predict with NaN/Inf Tests
// ============================================================================

/// Test prediction on data with NaN (after fitting on clean data)
pub fn check_predict_with_nan<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_predict_with_nan", move || {
        // Fit on clean data
        let x_train = Array2::from_shape_fn((20, 3), |(i, j)| (i + j + 1) as f64);
        let y_train = Array1::from_shape_fn(20, |i| (i + 1) as f64);

        if let Err(e) = model.fit(&x_train, &y_train) {
            return (false, Some(format!("Fit failed on clean data: {:?}", e)));
        }

        // Predict on data with NaN
        let mut x_test = Array2::from_shape_fn((5, 3), |(i, j)| (i + j + 1) as f64);
        x_test[[2, 1]] = f64::NAN;

        match model.predict(&x_test) {
            Err(_) => (true, None), // Rejecting NaN at predict time is acceptable
            Ok(preds) => {
                // If predictions are returned, at least some should be finite
                // The NaN sample may legitimately produce NaN
                let finite_count = preds.iter().filter(|v| v.is_finite()).count();
                if finite_count >= 4 {
                    // At least 4 of 5 samples should be finite (only the NaN one can be NaN)
                    (true, None)
                } else {
                    (
                        false,
                        Some(format!(
                            "Too many non-finite predictions: {} of {} finite",
                            finite_count,
                            preds.len()
                        )),
                    )
                }
            }
        }
    })
}

/// Test prediction on data with Inf (after fitting on clean data)
pub fn check_predict_with_inf<M: Model + Clone>(mut model: M) -> CheckResult {
    run_nan_inf_check("check_predict_with_inf", move || {
        let x_train = Array2::from_shape_fn((20, 3), |(i, j)| (i + j + 1) as f64);
        let y_train = Array1::from_shape_fn(20, |i| (i + 1) as f64);

        if let Err(e) = model.fit(&x_train, &y_train) {
            return (false, Some(format!("Fit failed on clean data: {:?}", e)));
        }

        let mut x_test = Array2::from_shape_fn((5, 3), |(i, j)| (i + j + 1) as f64);
        x_test[[2, 1]] = f64::INFINITY;

        match model.predict(&x_test) {
            Err(_) => (true, None),
            Ok(preds) => {
                let finite_count = preds.iter().filter(|v| v.is_finite()).count();
                if finite_count >= 4 {
                    (true, None)
                } else {
                    (
                        false,
                        Some(format!(
                            "Too many non-finite predictions with Inf input: {} of {} finite",
                            finite_count,
                            preds.len()
                        )),
                    )
                }
            }
        }
    })
}

// ============================================================================
// Transformer NaN/Inf Tests
// ============================================================================

/// Test transformer handling of NaN in fit
pub fn check_transformer_nan_fit<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    run_nan_inf_check("check_transformer_nan_fit", move || {
        let mut x = Array2::from_shape_fn((15, 4), |(i, j)| (i + j + 1) as f64);
        x[[5, 2]] = f64::NAN;

        match transformer.fit(&x) {
            Err(_) => (true, None), // Rejection is acceptable
            Ok(()) => {
                // If fit succeeds, transform on clean data should work
                let x_clean = Array2::from_shape_fn((5, 4), |(i, j)| (i + j + 1) as f64);
                match transformer.transform(&x_clean) {
                    Ok(result) => {
                        if result.iter().all(|v| v.is_finite()) {
                            (true, None)
                        } else {
                            (
                                false,
                                Some(
                                    "Transformer produces non-finite output after NaN fit"
                                        .to_string(),
                                ),
                            )
                        }
                    }
                    Err(_) => (false, Some("Transform failed after NaN fit".to_string())),
                }
            }
        }
    })
}

/// Test transformer handling of Inf in fit
pub fn check_transformer_inf_fit<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    run_nan_inf_check("check_transformer_inf_fit", move || {
        let mut x = Array2::from_shape_fn((15, 4), |(i, j)| (i + j + 1) as f64);
        x[[5, 2]] = f64::INFINITY;

        match transformer.fit(&x) {
            Err(_) => (true, None),
            Ok(()) => {
                let x_clean = Array2::from_shape_fn((5, 4), |(i, j)| (i + j + 1) as f64);
                match transformer.transform(&x_clean) {
                    Ok(result) => {
                        if result.iter().all(|v| v.is_finite()) {
                            (true, None)
                        } else {
                            (
                                false,
                                Some(
                                    "Transformer produces non-finite output after Inf fit"
                                        .to_string(),
                                ),
                            )
                        }
                    }
                    Err(_) => (false, Some("Transform failed after Inf fit".to_string())),
                }
            }
        }
    })
}

/// Test transformer handling of NaN in transform (after clean fit)
pub fn check_transformer_nan_transform<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    run_nan_inf_check("check_transformer_nan_transform", move || {
        // Fit on clean data
        let x_train = Array2::from_shape_fn((20, 4), |(i, j)| (i + j + 1) as f64);
        if let Err(e) = transformer.fit(&x_train) {
            return (false, Some(format!("Fit failed on clean data: {:?}", e)));
        }

        // Transform data with NaN
        let mut x_test = Array2::from_shape_fn((5, 4), |(i, j)| (i + j + 1) as f64);
        x_test[[2, 1]] = f64::NAN;

        match transformer.transform(&x_test) {
            Err(_) => (true, None), // Rejection is acceptable
            Ok(result) => {
                // Check that non-NaN rows produce finite output
                let row0_finite = result.row(0).iter().all(|v| v.is_finite());
                let row1_finite = result.row(1).iter().all(|v| v.is_finite());
                let row3_finite = result.row(3).iter().all(|v| v.is_finite());
                let row4_finite = result.row(4).iter().all(|v| v.is_finite());

                if row0_finite && row1_finite && row3_finite && row4_finite {
                    (true, None)
                } else {
                    (
                        false,
                        Some("Clean rows produce non-finite output when NaN present".to_string()),
                    )
                }
            }
        }
    })
}

/// Test transformer handling of Inf in transform (after clean fit)
pub fn check_transformer_inf_transform<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    run_nan_inf_check("check_transformer_inf_transform", move || {
        let x_train = Array2::from_shape_fn((20, 4), |(i, j)| (i + j + 1) as f64);
        if let Err(e) = transformer.fit(&x_train) {
            return (false, Some(format!("Fit failed on clean data: {:?}", e)));
        }

        let mut x_test = Array2::from_shape_fn((5, 4), |(i, j)| (i + j + 1) as f64);
        x_test[[2, 1]] = f64::INFINITY;

        match transformer.transform(&x_test) {
            Err(_) => (true, None),
            Ok(result) => {
                let row0_finite = result.row(0).iter().all(|v| v.is_finite());
                let row1_finite = result.row(1).iter().all(|v| v.is_finite());
                let row3_finite = result.row(3).iter().all(|v| v.is_finite());
                let row4_finite = result.row(4).iter().all(|v| v.is_finite());

                if row0_finite && row1_finite && row3_finite && row4_finite {
                    (true, None)
                } else {
                    (
                        false,
                        Some("Clean rows produce non-finite output when Inf present".to_string()),
                    )
                }
            }
        }
    })
}

// ============================================================================
// Main Validation Functions
// ============================================================================

/// Run all NaN/Inf validation checks on a model
pub fn validate_nan_inf_handling<M: Model + Clone>(model: M) -> Vec<CheckResult> {
    vec![
        // NaN in features
        check_nan_at_start(model.clone()),
        check_nan_at_end(model.clone()),
        check_nan_entire_row(model.clone()),
        check_nan_entire_column(model.clone()),
        check_nan_scattered(model.clone()),
        // Inf in features
        check_positive_inf(model.clone()),
        check_negative_inf(model.clone()),
        check_both_inf(model.clone()),
        check_inf_entire_column(model.clone()),
        // Mixed NaN/Inf
        check_mixed_nan_inf(model.clone()),
        check_nan_inf_same_row(model.clone()),
        check_nan_inf_same_column(model.clone()),
        // NaN/Inf in targets
        check_nan_in_target(model.clone()),
        check_inf_in_target(model.clone()),
        check_neg_inf_in_target(model.clone()),
        check_all_nan_targets(model.clone()),
        // Predict with NaN/Inf
        check_predict_with_nan(model.clone()),
        check_predict_with_inf(model),
    ]
}

/// Run all NaN/Inf validation checks on a transformer
pub fn validate_transformer_nan_inf_handling<T: Transformer + Clone>(
    transformer: T,
) -> Vec<CheckResult> {
    vec![
        check_transformer_nan_fit(transformer.clone()),
        check_transformer_inf_fit(transformer.clone()),
        check_transformer_nan_transform(transformer.clone()),
        check_transformer_inf_transform(transformer),
    ]
}

/// Assert that all NaN/Inf validation checks pass for a model
pub fn assert_nan_inf_handling_valid<M: Model + Clone>(model: M) {
    let results = validate_nan_inf_handling(model);
    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();

    if !failures.is_empty() {
        let msg = failures
            .iter()
            .map(|r| {
                format!(
                    "  - {}: {}",
                    r.name,
                    r.message.as_deref().unwrap_or("failed")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "Model failed {} NaN/Inf validation checks:\n{}",
            failures.len(),
            msg
        );
    }
}

/// Assert that all NaN/Inf validation checks pass for a transformer
pub fn assert_transformer_nan_inf_handling_valid<T: Transformer + Clone>(transformer: T) {
    let results = validate_transformer_nan_inf_handling(transformer);
    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();

    if !failures.is_empty() {
        let msg = failures
            .iter()
            .map(|r| {
                format!(
                    "  - {}: {}",
                    r.name,
                    r.message.as_deref().unwrap_or("failed")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "Transformer failed {} NaN/Inf validation checks:\n{}",
            failures.len(),
            msg
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;
    use crate::FerroError;
    use crate::Result;

    /// A mock model that rejects NaN/Inf for testing
    #[derive(Clone)]
    struct StrictMockModel {
        fitted: bool,
        n_features: Option<usize>,
    }

    impl StrictMockModel {
        fn new() -> Self {
            Self {
                fitted: false,
                n_features: None,
            }
        }
    }

    impl Model for StrictMockModel {
        fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
            // Check for NaN/Inf in features
            if x.iter().any(|v| !v.is_finite()) {
                return Err(FerroError::invalid_input("Non-finite values in features"));
            }
            // Check for NaN/Inf in target
            if y.iter().any(|v| !v.is_finite()) {
                return Err(FerroError::invalid_input("Non-finite values in target"));
            }
            if x.nrows() != y.len() {
                return Err(FerroError::shape_mismatch(
                    format!("{} rows", x.nrows()),
                    format!("{} elements", y.len()),
                ));
            }
            if x.is_empty() {
                return Err(FerroError::invalid_input("Empty data"));
            }
            self.fitted = true;
            self.n_features = Some(x.ncols());
            Ok(())
        }

        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("predict"));
            }
            if let Some(expected) = self.n_features {
                if x.ncols() != expected {
                    return Err(FerroError::shape_mismatch(
                        format!("{} features", expected),
                        format!("{} features", x.ncols()),
                    ));
                }
            }
            // Check for NaN/Inf in predict input
            if x.iter().any(|v| !v.is_finite()) {
                return Err(FerroError::invalid_input(
                    "Non-finite values in predict input",
                ));
            }
            Ok(Array1::from_elem(x.nrows(), 1.0))
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn n_features(&self) -> Option<usize> {
            self.n_features
        }
    }

    /// A mock model that accepts NaN/Inf for testing
    #[derive(Clone)]
    struct LenientMockModel {
        fitted: bool,
        n_features: Option<usize>,
    }

    impl LenientMockModel {
        fn new() -> Self {
            Self {
                fitted: false,
                n_features: None,
            }
        }
    }

    impl Model for LenientMockModel {
        fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
            if x.nrows() != y.len() {
                return Err(FerroError::shape_mismatch(
                    format!("{} rows", x.nrows()),
                    format!("{} elements", y.len()),
                ));
            }
            if x.is_empty() {
                return Err(FerroError::invalid_input("Empty data"));
            }
            self.fitted = true;
            self.n_features = Some(x.ncols());
            Ok(())
        }

        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("predict"));
            }
            if let Some(expected) = self.n_features {
                if x.ncols() != expected {
                    return Err(FerroError::shape_mismatch(
                        format!("{} features", expected),
                        format!("{} features", x.ncols()),
                    ));
                }
            }
            // Always returns finite predictions
            Ok(Array1::from_elem(x.nrows(), 1.0))
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn n_features(&self) -> Option<usize> {
            self.n_features
        }
    }

    #[test]
    fn test_strict_model_passes_all_checks() {
        let model = StrictMockModel::new();
        let results = validate_nan_inf_handling(model);

        for result in &results {
            assert!(
                result.passed,
                "{} failed: {}",
                result.name,
                result.message.as_deref().unwrap_or("no message")
            );
        }
    }

    #[test]
    fn test_lenient_model_passes_nan_checks() {
        let model = LenientMockModel::new();

        // NaN checks should pass because model handles gracefully
        assert!(check_nan_at_start(model.clone()).passed);
        assert!(check_nan_scattered(model.clone()).passed);
    }

    #[test]
    fn test_lenient_model_fails_target_nan_checks() {
        let model = LenientMockModel::new();

        // Target NaN checks should fail because lenient model accepts NaN in targets
        // (which is considered incorrect behavior)
        let result = check_nan_in_target(model);
        assert!(!result.passed, "Lenient model should fail NaN target check");
    }

    #[test]
    fn test_check_count() {
        let model = StrictMockModel::new();
        let results = validate_nan_inf_handling(model);

        // Should have 18 checks
        assert_eq!(results.len(), 18, "Expected 18 NaN/Inf checks");
    }
}
