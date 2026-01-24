//! Individual estimator check implementations
//!
//! This module provides 30+ individual checks for validating ML estimator implementations.
//! Each check tests a specific aspect of the Model trait contract and returns a CheckResult.

use crate::models::Model;
use crate::FerroError;
use ndarray::{s, Array1, Array2};
use std::time::Duration;

use super::{CheckCategory, CheckResult};

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper to run a closure and catch panics, returning a CheckResult
fn run_check<F>(name: &'static str, category: CheckCategory, check_fn: F) -> CheckResult
where
    F: FnOnce() -> (bool, Option<String>),
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(check_fn)) {
        Ok((passed, message)) => CheckResult {
            name,
            passed,
            message,
            duration: Duration::ZERO,
            category,
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
                category,
            }
        }
    }
}

/// Check if arrays are approximately equal within tolerance
fn arrays_approx_equal(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tol || (x.is_nan() && y.is_nan()))
}

// ============================================================================
// Check 1: check_not_fitted
// ============================================================================

/// Check that an unfitted model returns NotFitted error on predict
///
/// Tests that calling predict() on a model that hasn't been fitted returns
/// the appropriate NotFitted error variant.
pub fn check_not_fitted<M: Model>(model: &M) -> CheckResult {
    run_check("check_not_fitted", CheckCategory::Api, || {
        // Create dummy input for prediction
        let x_test = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();

        // Model should not be fitted initially
        if model.is_fitted() {
            return (
                false,
                Some("Model reports is_fitted=true before fitting".to_string()),
            );
        }

        // Predict should return NotFitted error
        match model.predict(&x_test) {
            Err(FerroError::NotFitted { .. }) => (true, None),
            Err(e) => (
                false,
                Some(format!("Expected NotFitted error, got: {:?}", e)),
            ),
            Ok(_) => (
                false,
                Some("Predict succeeded on unfitted model (should fail)".to_string()),
            ),
        }
    })
}

// ============================================================================
// Check 2: check_n_features_in
// ============================================================================

/// Check that feature count is tracked correctly after fitting
///
/// After fitting, the model should report the correct number of features
/// via n_features().
pub fn check_n_features_in<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check("check_n_features_in", CheckCategory::Api, move || {
        let expected_features = x.ncols();

        // Before fitting, n_features should be None
        if model.n_features().is_some() {
            return (
                false,
                Some("n_features() should be None before fitting".to_string()),
            );
        }

        // Fit the model
        if let Err(e) = model.fit(&x, &y) {
            return (false, Some(format!("Fit failed: {:?}", e)));
        }

        // Check n_features matches
        match model.n_features() {
            Some(n) if n == expected_features => (true, None),
            Some(n) => (
                false,
                Some(format!(
                    "n_features reports {}, expected {}",
                    n, expected_features
                )),
            ),
            None => (
                false,
                Some("n_features returns None after fitting".to_string()),
            ),
        }
    })
}

// ============================================================================
// Check 3: check_nan_handling
// ============================================================================

/// Check that NaN values in features are detected and handled
///
/// Models should either reject NaN inputs with an error or handle them gracefully.
pub fn check_nan_handling<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_nan_handling",
        CheckCategory::InputValidation,
        move || {
            let x_with_nan = Array2::from_shape_vec(
                (5, 2),
                vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            )
            .unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

            match model.fit(&x_with_nan, &y) {
                Err(FerroError::InvalidInput(_)) | Err(FerroError::NumericalError(_)) => {
                    (true, None)
                }
                Err(_) => (true, None), // Any error is acceptable for NaN handling
                Ok(_) => {
                    // If fit succeeds, check if model handles NaN gracefully
                    // by producing finite predictions for non-NaN data
                    let x_clean = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
                    match model.predict(&x_clean) {
                        Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                        Ok(_) => (
                            false,
                            Some(
                                "Model fit with NaN but produces non-finite predictions"
                                    .to_string(),
                            ),
                        ),
                        Err(e) => (
                            false,
                            Some(format!("Model fit with NaN but predict fails: {:?}", e)),
                        ),
                    }
                }
            }
        },
    )
}

// ============================================================================
// Check 4: check_inf_handling
// ============================================================================

/// Check that infinity values in features are detected and handled
///
/// Models should either reject infinite inputs with an error or handle them gracefully.
pub fn check_inf_handling<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_inf_handling",
        CheckCategory::InputValidation,
        move || {
            let x_with_inf = Array2::from_shape_vec(
                (5, 2),
                vec![
                    1.0,
                    2.0,
                    f64::INFINITY,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    f64::NEG_INFINITY,
                    9.0,
                    10.0,
                ],
            )
            .unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

            match model.fit(&x_with_inf, &y) {
                Err(FerroError::InvalidInput(_)) | Err(FerroError::NumericalError(_)) => {
                    (true, None)
                }
                Err(_) => (true, None), // Any error is acceptable
                Ok(_) => {
                    // If fit succeeds, check if model handles inf gracefully
                    let x_clean = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
                    match model.predict(&x_clean) {
                        Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                        Ok(_) => (
                            false,
                            Some(
                                "Model fit with infinity but produces non-finite predictions"
                                    .to_string(),
                            ),
                        ),
                        Err(e) => (
                            false,
                            Some(format!(
                                "Model fit with infinity but predict fails: {:?}",
                                e
                            )),
                        ),
                    }
                }
            }
        },
    )
}

// ============================================================================
// Check 5: check_empty_data
// ============================================================================

/// Check that empty data is rejected
///
/// Models should reject empty feature matrices with an appropriate error.
pub fn check_empty_data<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_empty_data",
        CheckCategory::InputValidation,
        move || {
            let x_empty: Array2<f64> = Array2::from_shape_vec((0, 3), vec![]).unwrap();
            let y_empty: Array1<f64> = Array1::from_vec(vec![]);

            match model.fit(&x_empty, &y_empty) {
                Err(FerroError::InvalidInput(_)) | Err(FerroError::ShapeMismatch { .. }) => {
                    (true, None)
                }
                Err(_) => (true, None), // Any error is acceptable
                Ok(_) => (
                    false,
                    Some("Fit succeeded on empty data (should fail)".to_string()),
                ),
            }
        },
    )
}

// ============================================================================
// Check 6: check_fit_idempotent
// ============================================================================

/// Check that fitting twice with the same data gives the same results
///
/// Fitting the same model twice on identical data should produce identical predictions.
pub fn check_fit_idempotent<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check(
        "check_fit_idempotent",
        CheckCategory::Numerical,
        move || {
            // First fit
            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("First fit failed: {:?}", e)));
            }

            let preds1 = match model.predict(&x) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("First predict failed: {:?}", e))),
            };

            // Second fit on same data
            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("Second fit failed: {:?}", e)));
            }

            let preds2 = match model.predict(&x) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("Second predict failed: {:?}", e))),
            };

            // Compare predictions
            if arrays_approx_equal(&preds1, &preds2, 1e-10) {
                (true, None)
            } else {
                (
                    false,
                    Some("Predictions differ after refitting with same data".to_string()),
                )
            }
        },
    )
}

// ============================================================================
// Check 7: check_single_sample
// ============================================================================

/// Check that a single sample can be handled without panic
///
/// Some models may not support single samples (e.g., need variance estimates),
/// but they should not panic - they should return an appropriate error.
pub fn check_single_sample<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_single_sample",
        CheckCategory::InputValidation,
        move || {
            let x_single = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
            let y_single = Array1::from_vec(vec![1.0]);

            // Fit may succeed or fail, but should not panic
            match model.fit(&x_single, &y_single) {
                Ok(_) => {
                    // If fit succeeds, predict should work too
                    match model.predict(&x_single) {
                        Ok(preds) if preds.len() == 1 => (true, None),
                        Ok(preds) => (
                            false,
                            Some(format!(
                                "Prediction has wrong length: {} (expected 1)",
                                preds.len()
                            )),
                        ),
                        Err(_) => (true, None), // Graceful error is acceptable
                    }
                }
                Err(_) => (true, None), // Graceful rejection is acceptable
            }
        },
    )
}

// ============================================================================
// Check 8: check_single_feature
// ============================================================================

/// Check that a single feature can be handled
///
/// Models should work correctly with datasets that have only one feature.
pub fn check_single_feature<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_single_feature",
        CheckCategory::InputValidation,
        move || {
            let x_single_feature = Array2::from_shape_vec(
                (10, 1),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            )
            .unwrap();
            let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

            match model.fit(&x_single_feature, &y) {
                Ok(_) => match model.predict(&x_single_feature) {
                    Ok(preds) if preds.len() == 10 => (true, None),
                    Ok(preds) => (
                        false,
                        Some(format!(
                            "Prediction has wrong length: {} (expected 10)",
                            preds.len()
                        )),
                    ),
                    Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
                },
                Err(_) => (true, None), // Graceful rejection is acceptable
            }
        },
    )
}

// ============================================================================
// Check 9: check_shape_mismatch
// ============================================================================

/// Check that X/y shape mismatch is detected
///
/// When X has a different number of rows than y has elements, an error should be raised.
pub fn check_shape_mismatch<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_shape_mismatch",
        CheckCategory::InputValidation,
        move || {
            let x = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();
            let y = Array1::from_vec(vec![1.0; 5]); // Mismatch: 10 vs 5

            match model.fit(&x, &y) {
                Err(FerroError::ShapeMismatch { .. }) => (true, None),
                Err(FerroError::InvalidInput(msg))
                    if msg.contains("shape") || msg.contains("length") =>
                {
                    (true, None)
                }
                Err(_) => (true, None), // Any error is acceptable for shape mismatch
                Ok(_) => (
                    false,
                    Some("Fit succeeded despite X/y shape mismatch".to_string()),
                ),
            }
        },
    )
}

// ============================================================================
// Check 10: check_subset_invariance
// ============================================================================

/// Check that batch predictions match individual predictions
///
/// Predicting on a batch of samples should give the same results as predicting
/// on each sample individually.
pub fn check_subset_invariance<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check(
        "check_subset_invariance",
        CheckCategory::Numerical,
        move || {
            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("Fit failed: {:?}", e)));
            }

            // Get batch predictions
            let batch_preds = match model.predict(&x) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("Batch predict failed: {:?}", e))),
            };

            // Get individual predictions
            let mut individual_preds = Vec::with_capacity(x.nrows());
            for i in 0..x.nrows() {
                let x_single = x.slice(s![i..i + 1, ..]).to_owned();
                match model.predict(&x_single) {
                    Ok(p) => individual_preds.push(p[0]),
                    Err(e) => {
                        return (
                            false,
                            Some(format!("Individual predict failed at {}: {:?}", i, e)),
                        )
                    }
                }
            }
            let individual_preds = Array1::from_vec(individual_preds);

            // Compare
            if arrays_approx_equal(&batch_preds, &individual_preds, 1e-10) {
                (true, None)
            } else {
                let max_diff = batch_preds
                    .iter()
                    .zip(individual_preds.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                (
                    false,
                    Some(format!(
                        "Batch/individual predictions differ (max diff: {:.2e})",
                        max_diff
                    )),
                )
            }
        },
    )
}

// ============================================================================
// Check 11: check_clone_equivalence
// ============================================================================

/// Check that a cloned model behaves identically to the original
///
/// After cloning, both models should produce identical predictions.
pub fn check_clone_equivalence<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check("check_clone_equivalence", CheckCategory::Api, move || {
        if let Err(e) = model.fit(&x, &y) {
            return (false, Some(format!("Fit failed: {:?}", e)));
        }

        let cloned_model = model.clone();

        let preds_original = match model.predict(&x) {
            Ok(p) => p,
            Err(e) => return (false, Some(format!("Original predict failed: {:?}", e))),
        };

        let preds_cloned = match cloned_model.predict(&x) {
            Ok(p) => p,
            Err(e) => return (false, Some(format!("Cloned predict failed: {:?}", e))),
        };

        if arrays_approx_equal(&preds_original, &preds_cloned, 1e-12) {
            (true, None)
        } else {
            (
                false,
                Some("Cloned model produces different predictions".to_string()),
            )
        }
    })
}

// ============================================================================
// Check 12: check_fit_does_not_modify_input
// ============================================================================

/// Check that fitting does not modify the input arrays
///
/// The X and y arrays should remain unchanged after fitting.
pub fn check_fit_does_not_modify_input<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check(
        "check_fit_does_not_modify_input",
        CheckCategory::Api,
        move || {
            let x_before = x.clone();
            let y_before = y.clone();

            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("Fit failed: {:?}", e)));
            }

            // Check X unchanged
            let x_unchanged = x
                .iter()
                .zip(x_before.iter())
                .all(|(a, b)| (a - b).abs() < 1e-15 || (a.is_nan() && b.is_nan()));

            // Check y unchanged
            let y_unchanged = y
                .iter()
                .zip(y_before.iter())
                .all(|(a, b)| (a - b).abs() < 1e-15 || (a.is_nan() && b.is_nan()));

            if x_unchanged && y_unchanged {
                (true, None)
            } else if !x_unchanged {
                (false, Some("X array was modified by fit".to_string()))
            } else {
                (false, Some("y array was modified by fit".to_string()))
            }
        },
    )
}

// ============================================================================
// Check 13: check_predict_shape
// ============================================================================

/// Check that prediction output shape matches n_samples
///
/// The predict output should have the same length as the number of input samples.
pub fn check_predict_shape<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check("check_predict_shape", CheckCategory::Api, move || {
        if let Err(e) = model.fit(&x, &y) {
            return (false, Some(format!("Fit failed: {:?}", e)));
        }

        // Test with various sizes
        let test_sizes = [1, 5, 10, 50];

        for &n_samples in &test_sizes {
            if n_samples > x.nrows() {
                continue;
            }
            let x_test = x.slice(s![0..n_samples, ..]).to_owned();
            match model.predict(&x_test) {
                Ok(preds) => {
                    if preds.len() != n_samples {
                        return (
                            false,
                            Some(format!(
                                "Prediction length {} doesn't match n_samples {}",
                                preds.len(),
                                n_samples
                            )),
                        );
                    }
                }
                Err(e) => {
                    return (
                        false,
                        Some(format!("Predict failed for {} samples: {:?}", n_samples, e)),
                    )
                }
            }
        }

        (true, None)
    })
}

// ============================================================================
// Check 14: check_no_side_effects
// ============================================================================

/// Check that multiple predicts give the same result
///
/// Calling predict multiple times should produce identical results (no side effects).
pub fn check_no_side_effects<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check(
        "check_no_side_effects",
        CheckCategory::Numerical,
        move || {
            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("Fit failed: {:?}", e)));
            }

            let preds1 = match model.predict(&x) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("First predict failed: {:?}", e))),
            };

            let preds2 = match model.predict(&x) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("Second predict failed: {:?}", e))),
            };

            let preds3 = match model.predict(&x) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("Third predict failed: {:?}", e))),
            };

            if arrays_approx_equal(&preds1, &preds2, 1e-12)
                && arrays_approx_equal(&preds2, &preds3, 1e-12)
            {
                (true, None)
            } else {
                (
                    false,
                    Some("Predictions differ across multiple calls".to_string()),
                )
            }
        },
    )
}

// ============================================================================
// Check 15: check_multithread_safe
// ============================================================================

/// Check that concurrent predict calls are safe
///
/// Multiple threads calling predict should not cause data races or panics.
pub fn check_multithread_safe<M: Model + Clone + Send + Sync + 'static>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check(
        "check_multithread_safe",
        CheckCategory::Concurrency,
        move || {
            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("Fit failed: {:?}", e)));
            }

            // Get reference predictions
            let reference_preds = match model.predict(&x) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("Reference predict failed: {:?}", e))),
            };

            // Spawn multiple threads to predict concurrently
            let model = std::sync::Arc::new(model);
            let x = std::sync::Arc::new(x);
            let reference = std::sync::Arc::new(reference_preds);

            let n_threads = 4;
            let handles: Vec<_> = (0..n_threads)
                .map(|_| {
                    let model = std::sync::Arc::clone(&model);
                    let x = std::sync::Arc::clone(&x);
                    let reference = std::sync::Arc::clone(&reference);

                    std::thread::spawn(move || {
                        for _ in 0..10 {
                            match model.predict(&x) {
                                Ok(preds) => {
                                    if !arrays_approx_equal(&preds, &reference, 1e-10) {
                                        return Err("Concurrent prediction differs from reference");
                                    }
                                }
                                Err(_) => return Err("Concurrent prediction failed"),
                            }
                        }
                        Ok(())
                    })
                })
                .collect();

            // Wait for all threads
            for handle in handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return (false, Some(e.to_string())),
                    Err(_) => {
                        return (
                            false,
                            Some("Thread panicked during concurrent predict".to_string()),
                        )
                    }
                }
            }

            (true, None)
        },
    )
}

// ============================================================================
// Check 16: check_large_input
// ============================================================================

/// Check that model handles large inputs (10k+ samples)
///
/// Model should be able to fit and predict on larger datasets without issues.
pub fn check_large_input<M: Model + Clone>(mut model: M, seed: u64) -> CheckResult {
    run_check("check_large_input", CheckCategory::Performance, move || {
        let (x, y) = super::utils::make_regression(10000, 10, 0.1, seed);

        match model.fit(&x, &y) {
            Ok(_) => match model.predict(&x) {
                Ok(preds) => {
                    if preds.len() == 10000 && preds.iter().all(|v| v.is_finite()) {
                        (true, None)
                    } else {
                        (
                            false,
                            Some("Predictions invalid for large input".to_string()),
                        )
                    }
                }
                Err(e) => (
                    false,
                    Some(format!("Predict failed on large input: {:?}", e)),
                ),
            },
            Err(e) => (false, Some(format!("Fit failed on large input: {:?}", e))),
        }
    })
}

// ============================================================================
// Check 17: check_negative_values
// ============================================================================

/// Check that negative feature values are handled
///
/// Model should work correctly with features that include negative values.
pub fn check_negative_values<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_negative_values",
        CheckCategory::Numerical,
        move || {
            let x = Array2::from_shape_vec(
                (10, 3),
                vec![
                    -5.0, -3.0, -1.0, -4.0, -2.0, 0.0, -3.0, -1.0, 1.0, -2.0, 0.0, 2.0, -1.0, 1.0,
                    3.0, 0.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 5.0, 7.0, 4.0, 6.0, 8.0,
                ],
            )
            .unwrap();
            let y = Array1::from_vec(vec![-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]);

            match model.fit(&x, &y) {
                Ok(_) => match model.predict(&x) {
                    Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                    Ok(_) => (
                        false,
                        Some("Predictions contain non-finite values".to_string()),
                    ),
                    Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
                },
                Err(_) => (true, None), // Graceful rejection is acceptable
            }
        },
    )
}

// ============================================================================
// Check 18: check_very_small_values
// ============================================================================

/// Check that very small values (near-zero) are handled
///
/// Model should handle features with values like 1e-100 without numerical issues.
pub fn check_very_small_values<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_very_small_values",
        CheckCategory::Numerical,
        move || {
            let x = Array2::from_shape_vec(
                (10, 2),
                vec![
                    1e-100, 1e-99, 1e-98, 1e-97, 1e-96, 1e-95, 1e-94, 1e-93, 1e-92, 1e-91, 1e-90,
                    1e-89, 1e-88, 1e-87, 1e-86, 1e-85, 1e-84, 1e-83, 1e-82, 1e-81,
                ],
            )
            .unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

            match model.fit(&x, &y) {
                Ok(_) => match model.predict(&x) {
                    Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                    Ok(_) => (
                        false,
                        Some("Predictions contain non-finite values for small inputs".to_string()),
                    ),
                    Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
                },
                Err(FerroError::NumericalError(_)) => (true, None), // Acceptable
                Err(_) => (true, None), // Any graceful error is acceptable
            }
        },
    )
}

// ============================================================================
// Check 19: check_very_large_values
// ============================================================================

/// Check that very large values are handled
///
/// Model should handle features with large values like 1e10 without overflow.
pub fn check_very_large_values<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_very_large_values",
        CheckCategory::Numerical,
        move || {
            let x = Array2::from_shape_vec(
                (10, 2),
                vec![
                    1e8, 1e9, 2e8, 2e9, 3e8, 3e9, 4e8, 4e9, 5e8, 5e9, 6e8, 6e9, 7e8, 7e9, 8e8, 8e9,
                    9e8, 9e9, 1e9, 1e10,
                ],
            )
            .unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

            match model.fit(&x, &y) {
                Ok(_) => match model.predict(&x) {
                    Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                    Ok(_) => (
                        false,
                        Some("Predictions contain non-finite values for large inputs".to_string()),
                    ),
                    Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
                },
                Err(FerroError::NumericalError(_)) => (true, None), // Acceptable
                Err(_) => (true, None), // Any graceful error is acceptable
            }
        },
    )
}

// ============================================================================
// Check 20: check_mixed_scale_features
// ============================================================================

/// Check that features with different scales are handled
///
/// Model should handle datasets where features vary from 1e-3 to 1e3.
pub fn check_mixed_scale_features<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_mixed_scale_features",
        CheckCategory::Numerical,
        move || {
            // Features at very different scales: ~1e-3, ~1, ~1e3
            let x = Array2::from_shape_vec(
                (10, 3),
                vec![
                    0.001, 1.0, 1000.0, 0.002, 2.0, 2000.0, 0.003, 3.0, 3000.0, 0.004, 4.0, 4000.0,
                    0.005, 5.0, 5000.0, 0.006, 6.0, 6000.0, 0.007, 7.0, 7000.0, 0.008, 8.0, 8000.0,
                    0.009, 9.0, 9000.0, 0.010, 10.0, 10000.0,
                ],
            )
            .unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

            match model.fit(&x, &y) {
                Ok(_) => match model.predict(&x) {
                    Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                    Ok(_) => (
                        false,
                        Some("Predictions contain non-finite values for mixed scales".to_string()),
                    ),
                    Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
                },
                Err(_) => (true, None), // Graceful handling is acceptable
            }
        },
    )
}

// ============================================================================
// Check 21: check_constant_feature
// ============================================================================

/// Check that constant feature columns are handled
///
/// Model should handle datasets where one or more features are constant.
pub fn check_constant_feature<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_constant_feature",
        CheckCategory::Numerical,
        move || {
            // First feature is constant, others vary
            let x = Array2::from_shape_vec(
                (10, 3),
                vec![
                    5.0, 1.0, 2.0, 5.0, 2.0, 4.0, 5.0, 3.0, 6.0, 5.0, 4.0, 8.0, 5.0, 5.0, 10.0,
                    5.0, 6.0, 12.0, 5.0, 7.0, 14.0, 5.0, 8.0, 16.0, 5.0, 9.0, 18.0, 5.0, 10.0,
                    20.0,
                ],
            )
            .unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

            match model.fit(&x, &y) {
                Ok(_) => match model.predict(&x) {
                    Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                    Ok(_) => (
                        false,
                        Some(
                            "Predictions contain non-finite values with constant feature"
                                .to_string(),
                        ),
                    ),
                    Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
                },
                Err(FerroError::NumericalError(_)) => (true, None), // Acceptable for some models
                Err(_) => (true, None), // Graceful handling is acceptable
            }
        },
    )
}

// ============================================================================
// Check 22: check_fit_twice_different_data
// ============================================================================

/// Check that model can be refitted on different data
///
/// A model should be able to be fitted again on a completely different dataset.
pub fn check_fit_twice_different_data<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_fit_twice_different_data",
        CheckCategory::Api,
        move || {
            // First dataset
            let x1 = Array2::from_shape_vec(
                (10, 2),
                vec![
                    1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                    9.0, 10.0, 10.0, 11.0,
                ],
            )
            .unwrap();
            let y1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

            // Second dataset (different size and values)
            let x2 = Array2::from_shape_vec(
                (5, 3),
                vec![
                    10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 30.0, 40.0, 50.0, 40.0, 50.0, 60.0, 50.0,
                    60.0, 70.0,
                ],
            )
            .unwrap();
            let y2 = Array1::from_vec(vec![100.0, 200.0, 300.0, 400.0, 500.0]);

            // First fit
            if let Err(e) = model.fit(&x1, &y1) {
                return (false, Some(format!("First fit failed: {:?}", e)));
            }

            let _preds1 = match model.predict(&x1) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("First predict failed: {:?}", e))),
            };

            // Second fit on different data
            if let Err(e) = model.fit(&x2, &y2) {
                return (false, Some(format!("Second fit failed: {:?}", e)));
            }

            // Verify model now uses new feature count
            if let Some(n) = model.n_features() {
                if n != 3 {
                    return (
                        false,
                        Some(format!("After refit, n_features is {} (expected 3)", n)),
                    );
                }
            }

            let preds2 = match model.predict(&x2) {
                Ok(p) => p,
                Err(e) => return (false, Some(format!("Second predict failed: {:?}", e))),
            };

            if preds2.len() == 5 && preds2.iter().all(|v| v.is_finite()) {
                (true, None)
            } else {
                (false, Some("Predictions invalid after refit".to_string()))
            }
        },
    )
}

// ============================================================================
// Check 23: check_dtype_consistency
// ============================================================================

/// Check that f64 input produces f64 output
///
/// Input and output should maintain consistent data types.
pub fn check_dtype_consistency<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check("check_dtype_consistency", CheckCategory::Api, move || {
        if let Err(e) = model.fit(&x, &y) {
            return (false, Some(format!("Fit failed: {:?}", e)));
        }

        match model.predict(&x) {
            Ok(preds) => {
                // The type is Array1<f64> by definition, so just verify it's valid
                if preds.len() == x.nrows() {
                    (true, None)
                } else {
                    (
                        false,
                        Some(format!(
                            "Output length {} doesn't match input rows {}",
                            preds.len(),
                            x.nrows()
                        )),
                    )
                }
            }
            Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
        }
    })
}

// ============================================================================
// Check 24: check_zero_samples_predict
// ============================================================================

/// Check that model handles zero samples in predict
///
/// Calling predict with zero samples should either return empty predictions
/// or a graceful error.
pub fn check_zero_samples_predict<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check(
        "check_zero_samples_predict",
        CheckCategory::InputValidation,
        move || {
            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("Fit failed: {:?}", e)));
            }

            let n_features = x.ncols();
            let x_empty: Array2<f64> = Array2::from_shape_vec((0, n_features), vec![]).unwrap();

            match model.predict(&x_empty) {
                Ok(preds) if preds.is_empty() => (true, None),
                Ok(preds) => (
                    false,
                    Some(format!(
                        "Zero samples returned {} predictions (expected 0)",
                        preds.len()
                    )),
                ),
                Err(FerroError::InvalidInput(_)) | Err(FerroError::ShapeMismatch { .. }) => {
                    (true, None) // Graceful rejection is acceptable
                }
                Err(_) => (true, None), // Any graceful error is acceptable
            }
        },
    )
}

// ============================================================================
// Check 25: check_deterministic_predictions
// ============================================================================

/// Check that the same fitted model gives the same predictions
///
/// For deterministic models, predictions should be exactly reproducible.
pub fn check_deterministic_predictions<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check(
        "check_deterministic_predictions",
        CheckCategory::Numerical,
        move || {
            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("Fit failed: {:?}", e)));
            }

            // Run predictions multiple times
            let mut all_predictions = Vec::new();
            for i in 0..5 {
                match model.predict(&x) {
                    Ok(preds) => all_predictions.push(preds),
                    Err(e) => return (false, Some(format!("Predict {} failed: {:?}", i, e))),
                }
            }

            // All predictions should be identical
            let reference = &all_predictions[0];
            for (i, preds) in all_predictions.iter().enumerate().skip(1) {
                if !arrays_approx_equal(reference, preds, 1e-15) {
                    return (
                        false,
                        Some(format!("Prediction {} differs from prediction 0", i)),
                    );
                }
            }

            (true, None)
        },
    )
}

// ============================================================================
// Additional Checks (26-30) for comprehensive coverage
// ============================================================================

/// Check 26: Verify feature count mismatch at predict time is detected
///
/// After fitting, predicting with wrong number of features should error.
pub fn check_predict_feature_mismatch<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check(
        "check_predict_feature_mismatch",
        CheckCategory::InputValidation,
        move || {
            if let Err(e) = model.fit(&x, &y) {
                return (false, Some(format!("Fit failed: {:?}", e)));
            }

            // Create test data with wrong number of features
            let wrong_features = x.ncols() + 2;
            let x_wrong =
                Array2::from_shape_vec((5, wrong_features), vec![1.0; 5 * wrong_features]).unwrap();

            match model.predict(&x_wrong) {
                Err(FerroError::ShapeMismatch { .. }) => (true, None),
                Err(_) => (true, None), // Any error is acceptable
                Ok(_) => (
                    false,
                    Some("Predict succeeded with wrong feature count".to_string()),
                ),
            }
        },
    )
}

/// Check 27: Verify is_fitted state changes correctly
///
/// is_fitted should be false before fit and true after.
pub fn check_is_fitted_state<M: Model + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    let x = x.clone();
    let y = y.clone();
    run_check("check_is_fitted_state", CheckCategory::Api, move || {
        // Should not be fitted initially
        if model.is_fitted() {
            return (false, Some("Model reports fitted before fit()".to_string()));
        }

        // Fit the model
        if let Err(e) = model.fit(&x, &y) {
            return (false, Some(format!("Fit failed: {:?}", e)));
        }

        // Should be fitted now
        if !model.is_fitted() {
            return (
                false,
                Some("Model reports not fitted after fit()".to_string()),
            );
        }

        (true, None)
    })
}

/// Check 28: Verify model handles duplicated rows
///
/// Data with duplicate rows should be handled without issues.
pub fn check_duplicate_rows<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_duplicate_rows",
        CheckCategory::Numerical,
        move || {
            // Data with duplicate rows
            let x = Array2::from_shape_vec(
                (10, 2),
                vec![
                    1.0, 2.0, 1.0, 2.0, // duplicate
                    3.0, 4.0, 3.0, 4.0, // duplicate
                    5.0, 6.0, 5.0, 6.0, // duplicate
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                ],
            )
            .unwrap();
            let y = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

            match model.fit(&x, &y) {
                Ok(_) => {
                    match model.predict(&x) {
                        Ok(preds) if preds.iter().all(|v| v.is_finite()) => {
                            // Check that duplicate rows get same predictions
                            let tol = 1e-10;
                            if (preds[0] - preds[1]).abs() < tol
                                && (preds[2] - preds[3]).abs() < tol
                                && (preds[4] - preds[5]).abs() < tol
                            {
                                (true, None)
                            } else {
                                (
                                    false,
                                    Some("Duplicate rows get different predictions".to_string()),
                                )
                            }
                        }
                        Ok(_) => (
                            false,
                            Some("Predictions contain non-finite values".to_string()),
                        ),
                        Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
                    }
                }
                Err(_) => (true, None), // Graceful handling is acceptable
            }
        },
    )
}

/// Check 29: Verify model handles perfectly collinear features
///
/// Features where one is a perfect linear combination of others should be handled.
pub fn check_collinear_features<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check(
        "check_collinear_features",
        CheckCategory::Numerical,
        move || {
            // x3 = 2*x1 + x2 (perfect collinearity)
            let x = Array2::from_shape_vec(
                (10, 3),
                vec![
                    1.0, 2.0, 4.0, // 2*1 + 2 = 4
                    2.0, 3.0, 7.0, // 2*2 + 3 = 7
                    3.0, 4.0, 10.0, // 2*3 + 4 = 10
                    4.0, 5.0, 13.0, 5.0, 6.0, 16.0, 6.0, 7.0, 19.0, 7.0, 8.0, 22.0, 8.0, 9.0, 25.0,
                    9.0, 10.0, 28.0, 10.0, 11.0, 31.0,
                ],
            )
            .unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

            match model.fit(&x, &y) {
                Ok(_) => match model.predict(&x) {
                    Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                    Ok(_) => (
                        false,
                        Some(
                            "Predictions contain non-finite values with collinear features"
                                .to_string(),
                        ),
                    ),
                    Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
                },
                Err(FerroError::NumericalError(_)) => (true, None), // Acceptable
                Err(_) => (true, None), // Graceful handling is acceptable
            }
        },
    )
}

/// Check 30: Verify model handles all-zero features
///
/// A feature column with all zeros should be handled gracefully.
pub fn check_zero_feature<M: Model + Clone>(mut model: M) -> CheckResult {
    run_check("check_zero_feature", CheckCategory::Numerical, move || {
        // Second feature is all zeros
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 2.0, 2.0, 0.0, 4.0, 3.0, 0.0, 6.0, 4.0, 0.0, 8.0, 5.0, 0.0, 10.0, 6.0,
                0.0, 12.0, 7.0, 0.0, 14.0, 8.0, 0.0, 16.0, 9.0, 0.0, 18.0, 10.0, 0.0, 20.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        match model.fit(&x, &y) {
            Ok(_) => match model.predict(&x) {
                Ok(preds) if preds.iter().all(|v| v.is_finite()) => (true, None),
                Ok(_) => (
                    false,
                    Some("Predictions contain non-finite values with zero feature".to_string()),
                ),
                Err(e) => (false, Some(format!("Predict failed: {:?}", e))),
            },
            Err(FerroError::NumericalError(_)) => (true, None), // Acceptable
            Err(_) => (true, None),                             // Graceful handling is acceptable
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;

    // Mock model for testing the checks themselves
    #[derive(Clone)]
    struct MockModel {
        fitted: bool,
        n_features: Option<usize>,
    }

    impl MockModel {
        fn new() -> Self {
            Self {
                fitted: false,
                n_features: None,
            }
        }
    }

    impl Model for MockModel {
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
            // Check for NaN
            if x.iter().any(|v| v.is_nan()) {
                return Err(FerroError::invalid_input("NaN in features"));
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
            // Simple prediction: sum of features
            Ok(Array1::from_iter(x.rows().into_iter().map(|row| row.sum())))
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn n_features(&self) -> Option<usize> {
            self.n_features
        }
    }

    #[test]
    fn test_check_not_fitted() {
        let model = MockModel::new();
        let result = check_not_fitted(&model);
        assert!(
            result.passed,
            "check_not_fitted should pass: {:?}",
            result.message
        );
    }

    #[test]
    fn test_check_n_features_in() {
        let model = MockModel::new();
        let x = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();
        let y = Array1::from_vec(vec![1.0; 10]);
        let result = check_n_features_in(model, &x, &y);
        assert!(
            result.passed,
            "check_n_features_in should pass: {:?}",
            result.message
        );
    }

    #[test]
    fn test_check_nan_handling() {
        let model = MockModel::new();
        let result = check_nan_handling(model);
        assert!(
            result.passed,
            "check_nan_handling should pass: {:?}",
            result.message
        );
    }

    #[test]
    fn test_check_empty_data() {
        let model = MockModel::new();
        let result = check_empty_data(model);
        assert!(
            result.passed,
            "check_empty_data should pass: {:?}",
            result.message
        );
    }

    #[test]
    fn test_check_shape_mismatch() {
        let model = MockModel::new();
        let result = check_shape_mismatch(model);
        assert!(
            result.passed,
            "check_shape_mismatch should pass: {:?}",
            result.message
        );
    }

    #[test]
    fn test_check_clone_equivalence() {
        let model = MockModel::new();
        let x = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();
        let y = Array1::from_vec(vec![1.0; 10]);
        let result = check_clone_equivalence(model, &x, &y);
        assert!(
            result.passed,
            "check_clone_equivalence should pass: {:?}",
            result.message
        );
    }

    #[test]
    fn test_check_deterministic_predictions() {
        let model = MockModel::new();
        let x = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();
        let y = Array1::from_vec(vec![1.0; 10]);
        let result = check_deterministic_predictions(model, &x, &y);
        assert!(
            result.passed,
            "check_deterministic_predictions should pass: {:?}",
            result.message
        );
    }

    #[test]
    fn test_check_is_fitted_state() {
        let model = MockModel::new();
        let x = Array2::from_shape_vec((10, 3), vec![1.0; 30]).unwrap();
        let y = Array1::from_vec(vec![1.0; 10]);
        let result = check_is_fitted_state(model, &x, &y);
        assert!(
            result.passed,
            "check_is_fitted_state should pass: {:?}",
            result.message
        );
    }
}
