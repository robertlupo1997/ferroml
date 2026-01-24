//! Fuzz target for preprocessing scalers
//!
//! This fuzzer tests the robustness of preprocessing transformers (StandardScaler,
//! MinMaxScaler, RobustScaler, MaxAbsScaler) with arbitrary numerical input data.
//! It tests edge cases like:
//! - NaN and infinity values
//! - Very large and very small numbers
//! - Single-element arrays
//! - Arrays with constant values
//! - Numerical stability issues

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use ndarray::Array2;

use ferroml_core::preprocessing::scalers::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
use ferroml_core::preprocessing::Transformer;

/// Arbitrary input for scaler fuzzing
#[derive(Debug)]
struct ScalerInput {
    /// Number of samples (rows)
    n_samples: usize,
    /// Number of features (columns)
    n_features: usize,
    /// Flat data to reshape into array
    data: Vec<f64>,
    /// Scaler configuration
    with_mean: bool,
    with_std: bool,
    with_centering: bool,
    with_scaling: bool,
    feature_range_min: f64,
    feature_range_max: f64,
    quantile_min: f64,
    quantile_max: f64,
}

impl<'a> Arbitrary<'a> for ScalerInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Keep dimensions small to avoid OOM
        let n_samples: usize = u.int_in_range(1..=100)?;
        let n_features: usize = u.int_in_range(1..=20)?;
        let total_elements = n_samples * n_features;

        // Generate data values
        let mut data = Vec::with_capacity(total_elements);
        for _ in 0..total_elements {
            // Generate a mix of normal values, edge cases, and special values
            let value_type: u8 = u.int_in_range(0..=10)?;
            let value = match value_type {
                0 => f64::NAN,                    // NaN
                1 => f64::INFINITY,               // Positive infinity
                2 => f64::NEG_INFINITY,           // Negative infinity
                3 => f64::MIN,                    // Minimum f64
                4 => f64::MAX,                    // Maximum f64
                5 => f64::MIN_POSITIVE,           // Smallest positive f64
                6 => -f64::MIN_POSITIVE,          // Smallest negative f64
                7 => 0.0,                         // Zero
                8 => -0.0,                        // Negative zero
                9 => u.arbitrary::<f64>()?,       // Arbitrary f64
                _ => u.int_in_range(-1000..=1000)? as f64, // Normal range
            };
            data.push(value);
        }

        // Scaler configurations
        let with_mean = u.arbitrary()?;
        let with_std = u.arbitrary()?;
        let with_centering = u.arbitrary()?;
        let with_scaling = u.arbitrary()?;

        // Feature range for MinMaxScaler
        let feature_range_min: f64 = u.int_in_range(-100..=100)? as f64;
        let feature_range_max: f64 = feature_range_min + u.int_in_range(1..=200)? as f64;

        // Quantile range for RobustScaler (must be in [0, 100])
        let quantile_min: f64 = u.int_in_range(0..=49)? as f64;
        let quantile_max: f64 = u.int_in_range(50..=100)? as f64;

        Ok(ScalerInput {
            n_samples,
            n_features,
            data,
            with_mean,
            with_std,
            with_centering,
            with_scaling,
            feature_range_min,
            feature_range_max,
            quantile_min,
            quantile_max,
        })
    }
}

/// Test StandardScaler with arbitrary input
fn fuzz_standard_scaler(input: &ScalerInput) {
    // Skip if data doesn't match dimensions
    if input.data.len() != input.n_samples * input.n_features {
        return;
    }

    let x = match Array2::from_shape_vec(
        (input.n_samples, input.n_features),
        input.data.clone(),
    ) {
        Ok(arr) => arr,
        Err(_) => return,
    };

    let mut scaler = StandardScaler::new()
        .with_mean(input.with_mean)
        .with_std(input.with_std);

    // Test fit
    if scaler.fit(&x).is_ok() {
        // Test transform
        let _ = scaler.transform(&x);

        // Test inverse_transform if transform succeeded
        if let Ok(transformed) = scaler.transform(&x) {
            let _ = scaler.inverse_transform(&transformed);
        }

        // Test fit_transform
        let mut scaler2 = StandardScaler::new()
            .with_mean(input.with_mean)
            .with_std(input.with_std);
        let _ = scaler2.fit_transform(&x);
    }
}

/// Test MinMaxScaler with arbitrary input
fn fuzz_minmax_scaler(input: &ScalerInput) {
    if input.data.len() != input.n_samples * input.n_features {
        return;
    }

    let x = match Array2::from_shape_vec(
        (input.n_samples, input.n_features),
        input.data.clone(),
    ) {
        Ok(arr) => arr,
        Err(_) => return,
    };

    // Ensure valid range
    if input.feature_range_min >= input.feature_range_max {
        return;
    }

    let mut scaler = MinMaxScaler::new()
        .with_range(input.feature_range_min, input.feature_range_max);

    if scaler.fit(&x).is_ok() {
        let _ = scaler.transform(&x);

        if let Ok(transformed) = scaler.transform(&x) {
            let _ = scaler.inverse_transform(&transformed);
        }

        let mut scaler2 = MinMaxScaler::new()
            .with_range(input.feature_range_min, input.feature_range_max);
        let _ = scaler2.fit_transform(&x);
    }
}

/// Test RobustScaler with arbitrary input
fn fuzz_robust_scaler(input: &ScalerInput) {
    if input.data.len() != input.n_samples * input.n_features {
        return;
    }

    let x = match Array2::from_shape_vec(
        (input.n_samples, input.n_features),
        input.data.clone(),
    ) {
        Ok(arr) => arr,
        Err(_) => return,
    };

    // Ensure valid quantile range
    if input.quantile_min >= input.quantile_max {
        return;
    }

    let mut scaler = RobustScaler::new()
        .with_centering(input.with_centering)
        .with_scaling(input.with_scaling)
        .with_quantile_range(input.quantile_min, input.quantile_max);

    if scaler.fit(&x).is_ok() {
        let _ = scaler.transform(&x);

        if let Ok(transformed) = scaler.transform(&x) {
            let _ = scaler.inverse_transform(&transformed);
        }

        let mut scaler2 = RobustScaler::new()
            .with_centering(input.with_centering)
            .with_scaling(input.with_scaling)
            .with_quantile_range(input.quantile_min, input.quantile_max);
        let _ = scaler2.fit_transform(&x);
    }
}

/// Test MaxAbsScaler with arbitrary input
fn fuzz_maxabs_scaler(input: &ScalerInput) {
    if input.data.len() != input.n_samples * input.n_features {
        return;
    }

    let x = match Array2::from_shape_vec(
        (input.n_samples, input.n_features),
        input.data.clone(),
    ) {
        Ok(arr) => arr,
        Err(_) => return,
    };

    let mut scaler = MaxAbsScaler::new();

    if scaler.fit(&x).is_ok() {
        let _ = scaler.transform(&x);

        if let Ok(transformed) = scaler.transform(&x) {
            let _ = scaler.inverse_transform(&transformed);
        }

        let mut scaler2 = MaxAbsScaler::new();
        let _ = scaler2.fit_transform(&x);
    }
}

fuzz_target!(|input: ScalerInput| {
    fuzz_standard_scaler(&input);
    fuzz_minmax_scaler(&input);
    fuzz_robust_scaler(&input);
    fuzz_maxabs_scaler(&input);
});
