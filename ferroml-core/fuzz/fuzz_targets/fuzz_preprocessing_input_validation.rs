//! Fuzz target for preprocessing input validation
//!
//! This fuzzer specifically tests the input validation logic in preprocessing
//! transformers to ensure they properly handle:
//! - Empty arrays
//! - Mismatched dimensions
//! - Invalid shapes
//! - Calling transform on unfitted transformers
//! - Arrays with special float values

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ndarray::Array2;

use ferroml_core::preprocessing::scalers::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
use ferroml_core::preprocessing::{
    check_is_fitted, check_non_empty, check_shape, column_max, column_mean, column_median,
    column_min, column_quantile, column_std, compute_column_statistics, find_constant_features,
    generate_feature_names, Transformer,
};

/// Arbitrary input for validation testing
#[derive(Debug)]
struct ValidationInput {
    /// Number of samples (rows) for fit
    fit_samples: usize,
    /// Number of features (columns) for fit
    fit_features: usize,
    /// Number of samples (rows) for transform
    transform_samples: usize,
    /// Number of features (columns) for transform
    transform_features: usize,
    /// Fit data
    fit_data: Vec<f64>,
    /// Transform data
    transform_data: Vec<f64>,
    /// Whether to call fit before transform
    call_fit: bool,
    /// Quantile value for column_quantile test
    quantile: f64,
    /// Variance threshold for find_constant_features
    variance_threshold: f64,
}

impl<'a> Arbitrary<'a> for ValidationInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Dimensions can include 0 to test edge cases
        let fit_samples: usize = u.int_in_range(0..=50)?;
        let fit_features: usize = u.int_in_range(0..=20)?;
        let transform_samples: usize = u.int_in_range(0..=50)?;
        let transform_features: usize = u.int_in_range(0..=20)?;

        let fit_total = fit_samples * fit_features;
        let transform_total = transform_samples * transform_features;

        // Generate fit data
        let mut fit_data = Vec::with_capacity(fit_total);
        for _ in 0..fit_total {
            let val: f64 = if u.arbitrary()? {
                u.arbitrary()?
            } else {
                u.int_in_range(-100..=100)? as f64
            };
            fit_data.push(val);
        }

        // Generate transform data
        let mut transform_data = Vec::with_capacity(transform_total);
        for _ in 0..transform_total {
            let val: f64 = if u.arbitrary()? {
                u.arbitrary()?
            } else {
                u.int_in_range(-100..=100)? as f64
            };
            transform_data.push(val);
        }

        let call_fit = u.arbitrary()?;

        // Quantile in [0, 1]
        let quantile: f64 = u.int_in_range(0..=100)? as f64 / 100.0;

        // Variance threshold
        let variance_threshold: f64 = u.int_in_range(0..=1000)? as f64 / 100.0;

        Ok(ValidationInput {
            fit_samples,
            fit_features,
            transform_samples,
            transform_features,
            fit_data,
            transform_data,
            call_fit,
            quantile,
            variance_threshold,
        })
    }
}

/// Test validation functions directly
fn fuzz_validation_functions(input: &ValidationInput) {
    // Test check_is_fitted
    let _ = check_is_fitted(true, "test");
    let _ = check_is_fitted(false, "test");

    // Test check_non_empty with various arrays
    if input.fit_data.len() == input.fit_samples * input.fit_features {
        if let Ok(arr) = Array2::from_shape_vec(
            (input.fit_samples, input.fit_features),
            input.fit_data.clone(),
        ) {
            let _ = check_non_empty(&arr);
        }
    }

    // Test with explicit empty array
    let empty: Array2<f64> = Array2::zeros((0, 0));
    let _ = check_non_empty(&empty);

    let zero_rows: Array2<f64> = Array2::zeros((0, 5));
    let _ = check_non_empty(&zero_rows);

    let zero_cols: Array2<f64> = Array2::zeros((5, 0));
    let _ = check_non_empty(&zero_cols);

    // Test check_shape
    if input.fit_data.len() == input.fit_samples * input.fit_features && input.fit_samples > 0 && input.fit_features > 0 {
        if let Ok(arr) = Array2::from_shape_vec(
            (input.fit_samples, input.fit_features),
            input.fit_data.clone(),
        ) {
            let _ = check_shape(&arr, input.fit_features);
            let _ = check_shape(&arr, input.transform_features);
            let _ = check_shape(&arr, 0);
            let _ = check_shape(&arr, usize::MAX);
        }
    }

    // Test generate_feature_names
    let _ = generate_feature_names(input.fit_features);
    let _ = generate_feature_names(0);
    let _ = generate_feature_names(1000);
}

/// Test statistical helper functions
fn fuzz_statistical_functions(input: &ValidationInput) {
    if input.fit_data.len() != input.fit_samples * input.fit_features {
        return;
    }
    if input.fit_samples == 0 || input.fit_features == 0 {
        return;
    }

    let arr = match Array2::from_shape_vec(
        (input.fit_samples, input.fit_features),
        input.fit_data.clone(),
    ) {
        Ok(a) => a,
        Err(_) => return,
    };

    // Test column statistics functions
    let _ = column_mean(&arr);
    let _ = column_std(&arr, 0); // population std
    let _ = column_std(&arr, 1); // sample std
    let _ = column_min(&arr);
    let _ = column_max(&arr);
    let _ = column_median(&arr);

    // Test column_quantile with various values
    if (0.0..=1.0).contains(&input.quantile) {
        let _ = column_quantile(&arr, input.quantile);
    }
    let _ = column_quantile(&arr, 0.0);
    let _ = column_quantile(&arr, 0.5);
    let _ = column_quantile(&arr, 1.0);

    // Test compute_column_statistics
    let _ = compute_column_statistics(&arr);

    // Test find_constant_features
    let _ = find_constant_features(&arr, input.variance_threshold);
    let _ = find_constant_features(&arr, 0.0);
    let _ = find_constant_features(&arr, f64::INFINITY);
}

/// Test transformer validation (fit before transform requirement)
fn fuzz_transformer_validation(input: &ValidationInput) {
    // Create arrays
    let fit_arr = if input.fit_data.len() == input.fit_samples * input.fit_features
        && input.fit_samples > 0
        && input.fit_features > 0
    {
        Array2::from_shape_vec((input.fit_samples, input.fit_features), input.fit_data.clone()).ok()
    } else {
        None
    };

    let transform_arr = if input.transform_data.len() == input.transform_samples * input.transform_features
        && input.transform_samples > 0
        && input.transform_features > 0
    {
        Array2::from_shape_vec(
            (input.transform_samples, input.transform_features),
            input.transform_data.clone(),
        )
        .ok()
    } else {
        None
    };

    // Test each scaler
    for scaler_type in 0..4 {
        let mut fitted = false;

        // StandardScaler
        if scaler_type == 0 {
            let mut scaler = StandardScaler::new();

            // Maybe fit
            if input.call_fit {
                if let Some(ref arr) = fit_arr {
                    fitted = scaler.fit(arr).is_ok();
                }
            }

            // Try transform without fit (should error)
            if !fitted {
                if let Some(ref arr) = transform_arr {
                    let result = scaler.transform(arr);
                    assert!(result.is_err(), "Transform should fail without fit");
                }
            } else if let Some(ref arr) = transform_arr {
                // Transform after fit (may still fail due to dimension mismatch)
                let _ = scaler.transform(arr);
            }
        }

        // MinMaxScaler
        if scaler_type == 1 {
            let mut scaler = MinMaxScaler::new();

            if input.call_fit {
                if let Some(ref arr) = fit_arr {
                    fitted = scaler.fit(arr).is_ok();
                }
            }

            if !fitted {
                if let Some(ref arr) = transform_arr {
                    let result = scaler.transform(arr);
                    assert!(result.is_err(), "Transform should fail without fit");
                }
            } else if let Some(ref arr) = transform_arr {
                let _ = scaler.transform(arr);
            }
        }

        // RobustScaler
        if scaler_type == 2 {
            let mut scaler = RobustScaler::new();

            if input.call_fit {
                if let Some(ref arr) = fit_arr {
                    fitted = scaler.fit(arr).is_ok();
                }
            }

            if !fitted {
                if let Some(ref arr) = transform_arr {
                    let result = scaler.transform(arr);
                    assert!(result.is_err(), "Transform should fail without fit");
                }
            } else if let Some(ref arr) = transform_arr {
                let _ = scaler.transform(arr);
            }
        }

        // MaxAbsScaler
        if scaler_type == 3 {
            let mut scaler = MaxAbsScaler::new();

            if input.call_fit {
                if let Some(ref arr) = fit_arr {
                    fitted = scaler.fit(arr).is_ok();
                }
            }

            if !fitted {
                if let Some(ref arr) = transform_arr {
                    let result = scaler.transform(arr);
                    assert!(result.is_err(), "Transform should fail without fit");
                }
            } else if let Some(ref arr) = transform_arr {
                let _ = scaler.transform(arr);
            }
        }
    }
}

/// Test dimension mismatch handling
fn fuzz_dimension_mismatch(input: &ValidationInput) {
    if input.fit_data.len() != input.fit_samples * input.fit_features {
        return;
    }
    if input.transform_data.len() != input.transform_samples * input.transform_features {
        return;
    }
    if input.fit_samples == 0 || input.fit_features == 0 {
        return;
    }
    if input.transform_samples == 0 || input.transform_features == 0 {
        return;
    }

    let fit_arr = match Array2::from_shape_vec(
        (input.fit_samples, input.fit_features),
        input.fit_data.clone(),
    ) {
        Ok(a) => a,
        Err(_) => return,
    };

    let transform_arr = match Array2::from_shape_vec(
        (input.transform_samples, input.transform_features),
        input.transform_data.clone(),
    ) {
        Ok(a) => a,
        Err(_) => return,
    };

    // Fit with one dimension, transform with another
    let mut scaler = StandardScaler::new();
    if scaler.fit(&fit_arr).is_ok() {
        let result = scaler.transform(&transform_arr);

        // If dimensions match, transform should succeed
        // If dimensions don't match, transform should fail
        if input.fit_features != input.transform_features {
            assert!(
                result.is_err(),
                "Transform should fail with mismatched features: fit={}, transform={}",
                input.fit_features,
                input.transform_features
            );
        }
    }
}

fuzz_target!(|input: ValidationInput| {
    fuzz_validation_functions(&input);
    fuzz_statistical_functions(&input);
    fuzz_transformer_validation(&input);
    fuzz_dimension_mismatch(&input);
});
