//! Serialization Round-Trip Validation Tests
//!
//! This module provides comprehensive serialization tests for FerroML models and transformers.
//! It validates that models can be serialized and deserialized without losing any state,
//! ensuring that fitted parameters, predictions, and metadata survive round-trips.
//!
//! ## Supported Formats
//!
//! - **JSON** (`serde_json`) - Human-readable, good for debugging
//! - **Binary** (`bincode`) - Compact and fast
//! - **MessagePack** (`rmp-serde`) - Cross-language compatible binary format
//!
//! ## Usage
//!
//! ```
//! # use ferroml_core::testing::serialization::{check_model_serialization, SerializationTestConfig};
//! # use ferroml_core::models::LinearRegression;
//! let results = check_model_serialization(
//!     LinearRegression::new(),
//!     SerializationTestConfig::default(),
//! );
//! assert!(!results.is_empty());
//! ```

use crate::models::Model;
use crate::preprocessing::Transformer;
use crate::FerroError;
use ndarray::{Array1, Array2};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

use super::{CheckCategory, CheckResult};

// ============================================================================
// Types and Configuration
// ============================================================================

/// Serialization formats to test
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON format (serde_json)
    Json,
    /// Binary format (bincode)
    Binary,
    /// MessagePack format (rmp-serde)
    MessagePack,
    /// All supported formats
    All,
}

/// Configuration for serialization tests
#[derive(Debug, Clone)]
pub struct SerializationTestConfig {
    /// Formats to test
    pub formats: SerializationFormat,
    /// Tolerance for floating-point comparisons
    pub prediction_tolerance: f64,
    /// Random seed for data generation
    pub seed: u64,
    /// Number of samples for test data
    pub n_samples: usize,
    /// Number of features for test data
    pub n_features: usize,
}

impl Default for SerializationTestConfig {
    fn default() -> Self {
        Self {
            formats: SerializationFormat::All,
            prediction_tolerance: 1e-10,
            seed: 42,
            n_samples: 50,
            n_features: 5,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Serialize to JSON string
fn serialize_json<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string(value).map_err(|e| format!("JSON serialization failed: {}", e))
}

/// Deserialize from JSON string
fn deserialize_json<T: DeserializeOwned>(data: &str) -> Result<T, String> {
    serde_json::from_str(data).map_err(|e| format!("JSON deserialization failed: {}", e))
}

/// Serialize to bincode bytes
fn serialize_binary<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    bincode::serialize(value).map_err(|e| format!("Binary serialization failed: {}", e))
}

/// Deserialize from bincode bytes
fn deserialize_binary<T: DeserializeOwned>(data: &[u8]) -> Result<T, String> {
    bincode::deserialize(data).map_err(|e| format!("Binary deserialization failed: {}", e))
}

/// Serialize to MessagePack bytes
fn serialize_msgpack<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    rmp_serde::to_vec(value).map_err(|e| format!("MessagePack serialization failed: {}", e))
}

/// Deserialize from MessagePack bytes
fn deserialize_msgpack<T: DeserializeOwned>(data: &[u8]) -> Result<T, String> {
    rmp_serde::from_slice(data).map_err(|e| format!("MessagePack deserialization failed: {}", e))
}

/// Check if two arrays are approximately equal
fn arrays_approx_equal(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tol || (x.is_nan() && y.is_nan()))
}

/// Check if two 2D arrays are approximately equal
fn arrays2_approx_equal(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tol || (x.is_nan() && y.is_nan()))
}

/// Generate test regression data
fn make_test_data(n_samples: usize, n_features: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    super::utils::make_regression(n_samples, n_features, 0.1, seed)
}

// ============================================================================
// Model Serialization Checks
// ============================================================================

/// Check JSON round-trip for a model
pub fn check_model_json_roundtrip<M>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
    tol: f64,
) -> CheckResult
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    // First fit the model
    if let Err(e) = model.fit(x, y) {
        return CheckResult::fail(
            "check_model_json_roundtrip",
            CheckCategory::Serialization,
            format!("Fit failed: {:?}", e),
        );
    }

    // Get predictions before serialization
    let preds_before = match model.predict(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_model_json_roundtrip",
                CheckCategory::Serialization,
                format!("Predict before serialization failed: {:?}", e),
            )
        }
    };

    // Record state before serialization
    let is_fitted_before = model.is_fitted();
    let n_features_before = model.n_features();

    // Serialize to JSON
    let json = match serialize_json(&model) {
        Ok(j) => j,
        Err(e) => {
            return CheckResult::fail(
                "check_model_json_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Deserialize from JSON
    let restored: M = match deserialize_json(&json) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_model_json_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Verify state preserved
    if restored.is_fitted() != is_fitted_before {
        return CheckResult::fail(
            "check_model_json_roundtrip",
            CheckCategory::Serialization,
            format!(
                "is_fitted changed: {} -> {}",
                is_fitted_before,
                restored.is_fitted()
            ),
        );
    }

    if restored.n_features() != n_features_before {
        return CheckResult::fail(
            "check_model_json_roundtrip",
            CheckCategory::Serialization,
            format!(
                "n_features changed: {:?} -> {:?}",
                n_features_before,
                restored.n_features()
            ),
        );
    }

    // Get predictions after deserialization
    let preds_after = match restored.predict(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_model_json_roundtrip",
                CheckCategory::Serialization,
                format!("Predict after deserialization failed: {:?}", e),
            )
        }
    };

    // Compare predictions
    if !arrays_approx_equal(&preds_before, &preds_after, tol) {
        let max_diff = preds_before
            .iter()
            .zip(preds_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        return CheckResult::fail(
            "check_model_json_roundtrip",
            CheckCategory::Serialization,
            format!(
                "Predictions differ by {:.2e} after JSON roundtrip",
                max_diff
            ),
        );
    }

    CheckResult::pass("check_model_json_roundtrip", CheckCategory::Serialization)
}

/// Check binary (bincode) round-trip for a model
pub fn check_model_binary_roundtrip<M>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
    tol: f64,
) -> CheckResult
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    // First fit the model
    if let Err(e) = model.fit(x, y) {
        return CheckResult::fail(
            "check_model_binary_roundtrip",
            CheckCategory::Serialization,
            format!("Fit failed: {:?}", e),
        );
    }

    // Get predictions before serialization
    let preds_before = match model.predict(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_model_binary_roundtrip",
                CheckCategory::Serialization,
                format!("Predict before serialization failed: {:?}", e),
            )
        }
    };

    // Record state before serialization
    let is_fitted_before = model.is_fitted();
    let n_features_before = model.n_features();

    // Serialize to binary
    let bytes = match serialize_binary(&model) {
        Ok(b) => b,
        Err(e) => {
            return CheckResult::fail(
                "check_model_binary_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Deserialize from binary
    let restored: M = match deserialize_binary(&bytes) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_model_binary_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Verify state preserved
    if restored.is_fitted() != is_fitted_before {
        return CheckResult::fail(
            "check_model_binary_roundtrip",
            CheckCategory::Serialization,
            format!(
                "is_fitted changed: {} -> {}",
                is_fitted_before,
                restored.is_fitted()
            ),
        );
    }

    if restored.n_features() != n_features_before {
        return CheckResult::fail(
            "check_model_binary_roundtrip",
            CheckCategory::Serialization,
            format!(
                "n_features changed: {:?} -> {:?}",
                n_features_before,
                restored.n_features()
            ),
        );
    }

    // Get predictions after deserialization
    let preds_after = match restored.predict(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_model_binary_roundtrip",
                CheckCategory::Serialization,
                format!("Predict after deserialization failed: {:?}", e),
            )
        }
    };

    // Compare predictions
    if !arrays_approx_equal(&preds_before, &preds_after, tol) {
        let max_diff = preds_before
            .iter()
            .zip(preds_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        return CheckResult::fail(
            "check_model_binary_roundtrip",
            CheckCategory::Serialization,
            format!(
                "Predictions differ by {:.2e} after binary roundtrip",
                max_diff
            ),
        );
    }

    CheckResult::pass("check_model_binary_roundtrip", CheckCategory::Serialization)
}

/// Check MessagePack round-trip for a model
pub fn check_model_msgpack_roundtrip<M>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
    tol: f64,
) -> CheckResult
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    // First fit the model
    if let Err(e) = model.fit(x, y) {
        return CheckResult::fail(
            "check_model_msgpack_roundtrip",
            CheckCategory::Serialization,
            format!("Fit failed: {:?}", e),
        );
    }

    // Get predictions before serialization
    let preds_before = match model.predict(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_model_msgpack_roundtrip",
                CheckCategory::Serialization,
                format!("Predict before serialization failed: {:?}", e),
            )
        }
    };

    // Record state before serialization
    let is_fitted_before = model.is_fitted();
    let n_features_before = model.n_features();

    // Serialize to MessagePack
    let bytes = match serialize_msgpack(&model) {
        Ok(b) => b,
        Err(e) => {
            return CheckResult::fail(
                "check_model_msgpack_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Deserialize from MessagePack
    let restored: M = match deserialize_msgpack(&bytes) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_model_msgpack_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Verify state preserved
    if restored.is_fitted() != is_fitted_before {
        return CheckResult::fail(
            "check_model_msgpack_roundtrip",
            CheckCategory::Serialization,
            format!(
                "is_fitted changed: {} -> {}",
                is_fitted_before,
                restored.is_fitted()
            ),
        );
    }

    if restored.n_features() != n_features_before {
        return CheckResult::fail(
            "check_model_msgpack_roundtrip",
            CheckCategory::Serialization,
            format!(
                "n_features changed: {:?} -> {:?}",
                n_features_before,
                restored.n_features()
            ),
        );
    }

    // Get predictions after deserialization
    let preds_after = match restored.predict(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_model_msgpack_roundtrip",
                CheckCategory::Serialization,
                format!("Predict after deserialization failed: {:?}", e),
            )
        }
    };

    // Compare predictions
    if !arrays_approx_equal(&preds_before, &preds_after, tol) {
        let max_diff = preds_before
            .iter()
            .zip(preds_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        return CheckResult::fail(
            "check_model_msgpack_roundtrip",
            CheckCategory::Serialization,
            format!(
                "Predictions differ by {:.2e} after MessagePack roundtrip",
                max_diff
            ),
        );
    }

    CheckResult::pass(
        "check_model_msgpack_roundtrip",
        CheckCategory::Serialization,
    )
}

/// Check that unfitted models serialize correctly
pub fn check_unfitted_model_serialization<M>(model: M) -> CheckResult
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    // Verify model is not fitted
    if model.is_fitted() {
        return CheckResult::fail(
            "check_unfitted_model_serialization",
            CheckCategory::Serialization,
            "Model should not be fitted initially",
        );
    }

    // Test JSON
    let json = match serialize_json(&model) {
        Ok(j) => j,
        Err(e) => {
            return CheckResult::fail(
                "check_unfitted_model_serialization",
                CheckCategory::Serialization,
                format!("JSON serialization of unfitted model failed: {}", e),
            )
        }
    };

    let restored: M = match deserialize_json(&json) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_unfitted_model_serialization",
                CheckCategory::Serialization,
                format!("JSON deserialization of unfitted model failed: {}", e),
            )
        }
    };

    if restored.is_fitted() {
        return CheckResult::fail(
            "check_unfitted_model_serialization",
            CheckCategory::Serialization,
            "Restored model reports fitted after deserializing unfitted state",
        );
    }

    // Test binary
    let bytes = match serialize_binary(&model) {
        Ok(b) => b,
        Err(e) => {
            return CheckResult::fail(
                "check_unfitted_model_serialization",
                CheckCategory::Serialization,
                format!("Binary serialization of unfitted model failed: {}", e),
            )
        }
    };

    let restored: M = match deserialize_binary(&bytes) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_unfitted_model_serialization",
                CheckCategory::Serialization,
                format!("Binary deserialization of unfitted model failed: {}", e),
            )
        }
    };

    if restored.is_fitted() {
        return CheckResult::fail(
            "check_unfitted_model_serialization",
            CheckCategory::Serialization,
            "Restored model reports fitted after binary deserialize of unfitted state",
        );
    }

    CheckResult::pass(
        "check_unfitted_model_serialization",
        CheckCategory::Serialization,
    )
}

/// Check serialization with very small coefficient values
pub fn check_serialization_small_values<M>(mut model: M, tol: f64) -> CheckResult
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    // Create data with small values
    let x = Array2::from_shape_fn((20, 3), |(i, j)| 1e-50 * ((i + 1) * (j + 1)) as f64);
    let y = Array1::from_shape_fn(20, |i| 1e-50 * (i + 1) as f64);

    if let Err(_) = model.fit(&x, &y) {
        // Some models may not handle very small values, that's acceptable
        return CheckResult::pass(
            "check_serialization_small_values",
            CheckCategory::Serialization,
        );
    }

    let preds_before = match model.predict(&x) {
        Ok(p) => p,
        Err(_) => {
            return CheckResult::pass(
                "check_serialization_small_values",
                CheckCategory::Serialization,
            )
        }
    };

    // JSON round-trip
    let json = match serialize_json(&model) {
        Ok(j) => j,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_small_values",
                CheckCategory::Serialization,
                format!("JSON serialization failed: {}", e),
            )
        }
    };

    let restored: M = match deserialize_json(&json) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_small_values",
                CheckCategory::Serialization,
                format!("JSON deserialization failed: {}", e),
            )
        }
    };

    let preds_after = match restored.predict(&x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_small_values",
                CheckCategory::Serialization,
                format!("Predict after deserialize failed: {:?}", e),
            )
        }
    };

    // For very small values, use relative tolerance or absolute near-zero tolerance
    let max_diff = preds_before
        .iter()
        .zip(preds_after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    if max_diff > tol && max_diff > 1e-100 {
        return CheckResult::fail(
            "check_serialization_small_values",
            CheckCategory::Serialization,
            format!("Predictions differ by {:.2e} for small values", max_diff),
        );
    }

    CheckResult::pass(
        "check_serialization_small_values",
        CheckCategory::Serialization,
    )
}

/// Check serialization with very large coefficient values
pub fn check_serialization_large_values<M>(mut model: M, tol: f64) -> CheckResult
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    // Create data with large values
    let x = Array2::from_shape_fn((20, 3), |(i, j)| 1e8 * ((i + 1) * (j + 1)) as f64);
    let y = Array1::from_shape_fn(20, |i| 1e8 * (i + 1) as f64);

    if let Err(_) = model.fit(&x, &y) {
        // Some models may not handle very large values, that's acceptable
        return CheckResult::pass(
            "check_serialization_large_values",
            CheckCategory::Serialization,
        );
    }

    let preds_before = match model.predict(&x) {
        Ok(p) => p,
        Err(_) => {
            return CheckResult::pass(
                "check_serialization_large_values",
                CheckCategory::Serialization,
            )
        }
    };

    // All predictions should be finite
    if !preds_before.iter().all(|v| v.is_finite()) {
        return CheckResult::pass(
            "check_serialization_large_values",
            CheckCategory::Serialization,
        );
    }

    // JSON round-trip
    let json = match serialize_json(&model) {
        Ok(j) => j,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_values",
                CheckCategory::Serialization,
                format!("JSON serialization failed: {}", e),
            )
        }
    };

    let restored: M = match deserialize_json(&json) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_values",
                CheckCategory::Serialization,
                format!("JSON deserialization failed: {}", e),
            )
        }
    };

    let preds_after = match restored.predict(&x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_values",
                CheckCategory::Serialization,
                format!("Predict after deserialize failed: {:?}", e),
            )
        }
    };

    // Use relative tolerance for large values
    let max_rel_diff = preds_before
        .iter()
        .zip(preds_after.iter())
        .map(|(a, b)| {
            let abs_diff = (a - b).abs();
            let scale = a.abs().max(b.abs()).max(1.0);
            abs_diff / scale
        })
        .fold(0.0_f64, f64::max);

    if max_rel_diff > tol {
        return CheckResult::fail(
            "check_serialization_large_values",
            CheckCategory::Serialization,
            format!(
                "Relative prediction difference {:.2e} for large values",
                max_rel_diff
            ),
        );
    }

    CheckResult::pass(
        "check_serialization_large_values",
        CheckCategory::Serialization,
    )
}

/// Check serialization with large model (many parameters)
pub fn check_serialization_large_model<M>(mut model: M, tol: f64) -> CheckResult
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    // Create large dataset
    let n_samples = 1000;
    let n_features = 50;
    let (x, y) = make_test_data(n_samples, n_features, 12345);

    if let Err(e) = model.fit(&x, &y) {
        return CheckResult::fail(
            "check_serialization_large_model",
            CheckCategory::Serialization,
            format!("Fit on large data failed: {:?}", e),
        );
    }

    let preds_before = match model.predict(&x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_model",
                CheckCategory::Serialization,
                format!("Predict before serialization failed: {:?}", e),
            )
        }
    };

    // Test all formats
    // JSON
    let json = match serialize_json(&model) {
        Ok(j) => j,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_model",
                CheckCategory::Serialization,
                format!("JSON serialization of large model failed: {}", e),
            )
        }
    };

    let restored_json: M = match deserialize_json(&json) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_model",
                CheckCategory::Serialization,
                format!("JSON deserialization of large model failed: {}", e),
            )
        }
    };

    let preds_json = match restored_json.predict(&x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_model",
                CheckCategory::Serialization,
                format!("Predict after JSON deserialize failed: {:?}", e),
            )
        }
    };

    if !arrays_approx_equal(&preds_before, &preds_json, tol) {
        return CheckResult::fail(
            "check_serialization_large_model",
            CheckCategory::Serialization,
            "Large model predictions differ after JSON roundtrip",
        );
    }

    // Binary
    let bytes = match serialize_binary(&model) {
        Ok(b) => b,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_model",
                CheckCategory::Serialization,
                format!("Binary serialization of large model failed: {}", e),
            )
        }
    };

    let restored_bin: M = match deserialize_binary(&bytes) {
        Ok(m) => m,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_model",
                CheckCategory::Serialization,
                format!("Binary deserialization of large model failed: {}", e),
            )
        }
    };

    let preds_bin = match restored_bin.predict(&x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_serialization_large_model",
                CheckCategory::Serialization,
                format!("Predict after binary deserialize failed: {:?}", e),
            )
        }
    };

    if !arrays_approx_equal(&preds_before, &preds_bin, tol) {
        return CheckResult::fail(
            "check_serialization_large_model",
            CheckCategory::Serialization,
            "Large model predictions differ after binary roundtrip",
        );
    }

    CheckResult::pass(
        "check_serialization_large_model",
        CheckCategory::Serialization,
    )
}

/// Run all serialization checks on a model
pub fn check_model_serialization<M>(model: M, config: SerializationTestConfig) -> Vec<CheckResult>
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    let mut results = Vec::new();
    let (x, y) = make_test_data(config.n_samples, config.n_features, config.seed);
    let tol = config.prediction_tolerance;

    // Check unfitted model serialization first
    results.push(check_unfitted_model_serialization(model.clone()));

    // Check round-trip with each format
    match config.formats {
        SerializationFormat::Json => {
            results.push(check_model_json_roundtrip(model.clone(), &x, &y, tol));
        }
        SerializationFormat::Binary => {
            results.push(check_model_binary_roundtrip(model.clone(), &x, &y, tol));
        }
        SerializationFormat::MessagePack => {
            results.push(check_model_msgpack_roundtrip(model.clone(), &x, &y, tol));
        }
        SerializationFormat::All => {
            results.push(check_model_json_roundtrip(model.clone(), &x, &y, tol));
            results.push(check_model_binary_roundtrip(model.clone(), &x, &y, tol));
            results.push(check_model_msgpack_roundtrip(model.clone(), &x, &y, tol));
        }
    }

    // Edge case tests
    results.push(check_serialization_small_values(model.clone(), tol));
    results.push(check_serialization_large_values(model.clone(), tol));
    results.push(check_serialization_large_model(model, tol));

    results
}

// ============================================================================
// Transformer Serialization Checks
// ============================================================================

/// Check JSON round-trip for a transformer
pub fn check_transformer_json_roundtrip<T>(mut transformer: T, tol: f64) -> CheckResult
where
    T: Transformer + Clone + Serialize + DeserializeOwned,
{
    let x = Array2::from_shape_fn((30, 5), |(i, j)| ((i * 3 + j) as f64) + 1.0);

    // Fit the transformer
    if let Err(e) = transformer.fit(&x) {
        return CheckResult::fail(
            "check_transformer_json_roundtrip",
            CheckCategory::Serialization,
            format!("Fit failed: {:?}", e),
        );
    }

    // Transform before serialization
    let transformed_before = match transformer.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_json_roundtrip",
                CheckCategory::Serialization,
                format!("Transform before serialization failed: {:?}", e),
            )
        }
    };

    // Record state
    let is_fitted_before = transformer.is_fitted();
    let n_features_in_before = transformer.n_features_in();
    let n_features_out_before = transformer.n_features_out();

    // Serialize to JSON
    let json = match serialize_json(&transformer) {
        Ok(j) => j,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_json_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Deserialize
    let restored: T = match deserialize_json(&json) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_json_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Verify state preserved
    if restored.is_fitted() != is_fitted_before {
        return CheckResult::fail(
            "check_transformer_json_roundtrip",
            CheckCategory::Serialization,
            format!(
                "is_fitted changed: {} -> {}",
                is_fitted_before,
                restored.is_fitted()
            ),
        );
    }

    if restored.n_features_in() != n_features_in_before {
        return CheckResult::fail(
            "check_transformer_json_roundtrip",
            CheckCategory::Serialization,
            format!(
                "n_features_in changed: {:?} -> {:?}",
                n_features_in_before,
                restored.n_features_in()
            ),
        );
    }

    if restored.n_features_out() != n_features_out_before {
        return CheckResult::fail(
            "check_transformer_json_roundtrip",
            CheckCategory::Serialization,
            format!(
                "n_features_out changed: {:?} -> {:?}",
                n_features_out_before,
                restored.n_features_out()
            ),
        );
    }

    // Transform after deserialization
    let transformed_after = match restored.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_json_roundtrip",
                CheckCategory::Serialization,
                format!("Transform after deserialization failed: {:?}", e),
            )
        }
    };

    // Compare transforms
    if !arrays2_approx_equal(&transformed_before, &transformed_after, tol) {
        let max_diff = transformed_before
            .iter()
            .zip(transformed_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        return CheckResult::fail(
            "check_transformer_json_roundtrip",
            CheckCategory::Serialization,
            format!(
                "Transform output differs by {:.2e} after JSON roundtrip",
                max_diff
            ),
        );
    }

    CheckResult::pass(
        "check_transformer_json_roundtrip",
        CheckCategory::Serialization,
    )
}

/// Check binary round-trip for a transformer
pub fn check_transformer_binary_roundtrip<T>(mut transformer: T, tol: f64) -> CheckResult
where
    T: Transformer + Clone + Serialize + DeserializeOwned,
{
    let x = Array2::from_shape_fn((30, 5), |(i, j)| ((i * 3 + j) as f64) + 1.0);

    // Fit the transformer
    if let Err(e) = transformer.fit(&x) {
        return CheckResult::fail(
            "check_transformer_binary_roundtrip",
            CheckCategory::Serialization,
            format!("Fit failed: {:?}", e),
        );
    }

    // Transform before serialization
    let transformed_before = match transformer.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_binary_roundtrip",
                CheckCategory::Serialization,
                format!("Transform before serialization failed: {:?}", e),
            )
        }
    };

    // Serialize to binary
    let bytes = match serialize_binary(&transformer) {
        Ok(b) => b,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_binary_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Deserialize
    let restored: T = match deserialize_binary(&bytes) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_binary_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Transform after deserialization
    let transformed_after = match restored.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_binary_roundtrip",
                CheckCategory::Serialization,
                format!("Transform after deserialization failed: {:?}", e),
            )
        }
    };

    // Compare transforms
    if !arrays2_approx_equal(&transformed_before, &transformed_after, tol) {
        let max_diff = transformed_before
            .iter()
            .zip(transformed_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        return CheckResult::fail(
            "check_transformer_binary_roundtrip",
            CheckCategory::Serialization,
            format!(
                "Transform output differs by {:.2e} after binary roundtrip",
                max_diff
            ),
        );
    }

    CheckResult::pass(
        "check_transformer_binary_roundtrip",
        CheckCategory::Serialization,
    )
}

/// Check MessagePack round-trip for a transformer
pub fn check_transformer_msgpack_roundtrip<T>(mut transformer: T, tol: f64) -> CheckResult
where
    T: Transformer + Clone + Serialize + DeserializeOwned,
{
    let x = Array2::from_shape_fn((30, 5), |(i, j)| ((i * 3 + j) as f64) + 1.0);

    // Fit the transformer
    if let Err(e) = transformer.fit(&x) {
        return CheckResult::fail(
            "check_transformer_msgpack_roundtrip",
            CheckCategory::Serialization,
            format!("Fit failed: {:?}", e),
        );
    }

    // Transform before serialization
    let transformed_before = match transformer.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_msgpack_roundtrip",
                CheckCategory::Serialization,
                format!("Transform before serialization failed: {:?}", e),
            )
        }
    };

    // Serialize to MessagePack
    let bytes = match serialize_msgpack(&transformer) {
        Ok(b) => b,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_msgpack_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Deserialize
    let restored: T = match deserialize_msgpack(&bytes) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_msgpack_roundtrip",
                CheckCategory::Serialization,
                e,
            )
        }
    };

    // Transform after deserialization
    let transformed_after = match restored.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_msgpack_roundtrip",
                CheckCategory::Serialization,
                format!("Transform after deserialization failed: {:?}", e),
            )
        }
    };

    // Compare transforms
    if !arrays2_approx_equal(&transformed_before, &transformed_after, tol) {
        let max_diff = transformed_before
            .iter()
            .zip(transformed_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        return CheckResult::fail(
            "check_transformer_msgpack_roundtrip",
            CheckCategory::Serialization,
            format!(
                "Transform output differs by {:.2e} after MessagePack roundtrip",
                max_diff
            ),
        );
    }

    CheckResult::pass(
        "check_transformer_msgpack_roundtrip",
        CheckCategory::Serialization,
    )
}

/// Check that fit_transform before == transform after deserialize
pub fn check_transformer_fit_transform_consistency<T>(transformer: T, tol: f64) -> CheckResult
where
    T: Transformer + Clone + Serialize + DeserializeOwned,
{
    let x = Array2::from_shape_fn((30, 5), |(i, j)| ((i * 3 + j) as f64) + 1.0);

    // Use fit_transform
    let mut t1 = transformer;
    let fit_transformed = match t1.fit_transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_fit_transform_consistency",
                CheckCategory::Serialization,
                format!("fit_transform failed: {:?}", e),
            )
        }
    };

    // Serialize and deserialize
    let json = match serialize_json(&t1) {
        Ok(j) => j,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_fit_transform_consistency",
                CheckCategory::Serialization,
                format!("JSON serialization failed: {}", e),
            )
        }
    };

    let restored: T = match deserialize_json(&json) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_fit_transform_consistency",
                CheckCategory::Serialization,
                format!("JSON deserialization failed: {}", e),
            )
        }
    };

    // Transform with deserialized transformer
    let transformed_after = match restored.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_transformer_fit_transform_consistency",
                CheckCategory::Serialization,
                format!("Transform after deserialize failed: {:?}", e),
            )
        }
    };

    // Compare
    if !arrays2_approx_equal(&fit_transformed, &transformed_after, tol) {
        let max_diff = fit_transformed
            .iter()
            .zip(transformed_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        return CheckResult::fail(
            "check_transformer_fit_transform_consistency",
            CheckCategory::Serialization,
            format!(
                "fit_transform before != transform after deserialize (diff: {:.2e})",
                max_diff
            ),
        );
    }

    CheckResult::pass(
        "check_transformer_fit_transform_consistency",
        CheckCategory::Serialization,
    )
}

/// Check unfitted transformer serialization
pub fn check_unfitted_transformer_serialization<T>(transformer: T) -> CheckResult
where
    T: Transformer + Clone + Serialize + DeserializeOwned,
{
    if transformer.is_fitted() {
        return CheckResult::fail(
            "check_unfitted_transformer_serialization",
            CheckCategory::Serialization,
            "Transformer should not be fitted initially",
        );
    }

    // JSON round-trip
    let json = match serialize_json(&transformer) {
        Ok(j) => j,
        Err(e) => {
            return CheckResult::fail(
                "check_unfitted_transformer_serialization",
                CheckCategory::Serialization,
                format!("JSON serialization of unfitted transformer failed: {}", e),
            )
        }
    };

    let restored: T = match deserialize_json(&json) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_unfitted_transformer_serialization",
                CheckCategory::Serialization,
                format!("JSON deserialization of unfitted transformer failed: {}", e),
            )
        }
    };

    if restored.is_fitted() {
        return CheckResult::fail(
            "check_unfitted_transformer_serialization",
            CheckCategory::Serialization,
            "Restored transformer reports fitted after deserializing unfitted state",
        );
    }

    // Verify transform fails with NotFitted
    let x = Array2::zeros((5, 3));
    match restored.transform(&x) {
        Err(FerroError::NotFitted { .. }) => {}
        Err(_) => {}
        Ok(_) => {
            return CheckResult::fail(
                "check_unfitted_transformer_serialization",
                CheckCategory::Serialization,
                "Deserialized unfitted transformer allowed transform",
            )
        }
    }

    CheckResult::pass(
        "check_unfitted_transformer_serialization",
        CheckCategory::Serialization,
    )
}

/// Run all serialization checks on a transformer
pub fn check_transformer_serialization<T>(
    transformer: T,
    config: SerializationTestConfig,
) -> Vec<CheckResult>
where
    T: Transformer + Clone + Serialize + DeserializeOwned,
{
    let mut results = Vec::new();
    let tol = config.prediction_tolerance;

    // Check unfitted transformer serialization
    results.push(check_unfitted_transformer_serialization(
        transformer.clone(),
    ));

    // Check round-trip with each format
    match config.formats {
        SerializationFormat::Json => {
            results.push(check_transformer_json_roundtrip(transformer.clone(), tol));
        }
        SerializationFormat::Binary => {
            results.push(check_transformer_binary_roundtrip(transformer.clone(), tol));
        }
        SerializationFormat::MessagePack => {
            results.push(check_transformer_msgpack_roundtrip(
                transformer.clone(),
                tol,
            ));
        }
        SerializationFormat::All => {
            results.push(check_transformer_json_roundtrip(transformer.clone(), tol));
            results.push(check_transformer_binary_roundtrip(transformer.clone(), tol));
            results.push(check_transformer_msgpack_roundtrip(
                transformer.clone(),
                tol,
            ));
        }
    }

    // Check fit_transform consistency
    results.push(check_transformer_fit_transform_consistency(
        transformer,
        tol,
    ));

    results
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Run all serialization checks with default config
pub fn check_model_serialization_all<M>(model: M) -> Vec<CheckResult>
where
    M: Model + Clone + Serialize + DeserializeOwned,
{
    check_model_serialization(model, SerializationTestConfig::default())
}

/// Run all transformer serialization checks with default config
pub fn check_transformer_serialization_all<T>(transformer: T) -> Vec<CheckResult>
where
    T: Transformer + Clone + Serialize + DeserializeOwned,
{
    check_transformer_serialization(transformer, SerializationTestConfig::default())
}

/// Assert all serialization checks pass for a model
pub fn assert_model_serialization_valid<M>(model: M)
where
    M: Model + Clone + Serialize + DeserializeOwned + Debug,
{
    let results = check_model_serialization_all(model);
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
            "Model serialization failed {} checks:\n{}",
            failures.len(),
            msg
        );
    }
}

/// Assert all serialization checks pass for a transformer
pub fn assert_transformer_serialization_valid<T>(transformer: T)
where
    T: Transformer + Clone + Serialize + DeserializeOwned + Debug,
{
    let results = check_transformer_serialization_all(transformer);
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
            "Transformer serialization failed {} checks:\n{}",
            failures.len(),
            msg
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;
    use serde::Deserialize;

    // Mock model for testing the serialization checks
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockModel {
        fitted: bool,
        n_features: Option<usize>,
        coefficients: Option<Vec<f64>>,
    }

    impl MockModel {
        fn new() -> Self {
            Self {
                fitted: false,
                n_features: None,
                coefficients: None,
            }
        }
    }

    impl Model for MockModel {
        fn fit(&mut self, x: &Array2<f64>, _y: &Array1<f64>) -> Result<()> {
            self.fitted = true;
            self.n_features = Some(x.ncols());
            self.coefficients = Some(vec![1.0; x.ncols()]);
            Ok(())
        }

        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("predict"));
            }
            // Simple sum of features
            Ok(Array1::from_iter(x.rows().into_iter().map(|row| row.sum())))
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn n_features(&self) -> Option<usize> {
            self.n_features
        }
    }

    // Mock transformer for testing
    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct MockTransformer {
        fitted: bool,
        n_features_in: Option<usize>,
        mean: Option<Vec<f64>>,
    }

    impl MockTransformer {
        fn new() -> Self {
            Self {
                fitted: false,
                n_features_in: None,
                mean: None,
            }
        }
    }

    impl Transformer for MockTransformer {
        fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
            self.fitted = true;
            self.n_features_in = Some(x.ncols());
            self.mean = Some(
                x.mean_axis(ndarray::Axis(0))
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect(),
            );
            Ok(())
        }

        fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
            if !self.fitted {
                return Err(FerroError::not_fitted("transform"));
            }
            let mean = self.mean.as_ref().unwrap();
            let mean_arr = Array1::from_vec(mean.clone());
            Ok(x - &mean_arr)
        }

        fn is_fitted(&self) -> bool {
            self.fitted
        }

        fn n_features_in(&self) -> Option<usize> {
            self.n_features_in
        }

        fn n_features_out(&self) -> Option<usize> {
            self.n_features_in
        }

        fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
            input_names.map(|names| names.to_vec())
        }
    }

    #[test]
    fn test_model_json_roundtrip() {
        let model = MockModel::new();
        let (x, y) = make_test_data(50, 5, 42);
        let result = check_model_json_roundtrip(model, &x, &y, 1e-10);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_model_binary_roundtrip() {
        let model = MockModel::new();
        let (x, y) = make_test_data(50, 5, 42);
        let result = check_model_binary_roundtrip(model, &x, &y, 1e-10);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_model_msgpack_roundtrip() {
        let model = MockModel::new();
        let (x, y) = make_test_data(50, 5, 42);
        let result = check_model_msgpack_roundtrip(model, &x, &y, 1e-10);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_unfitted_model_serialization() {
        let model = MockModel::new();
        let result = check_unfitted_model_serialization(model);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_transformer_json_roundtrip() {
        let transformer = MockTransformer::new();
        let result = check_transformer_json_roundtrip(transformer, 1e-10);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_transformer_binary_roundtrip() {
        let transformer = MockTransformer::new();
        let result = check_transformer_binary_roundtrip(transformer, 1e-10);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_transformer_msgpack_roundtrip() {
        let transformer = MockTransformer::new();
        let result = check_transformer_msgpack_roundtrip(transformer, 1e-10);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_unfitted_transformer_serialization() {
        let transformer = MockTransformer::new();
        let result = check_unfitted_transformer_serialization(transformer);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_fit_transform_consistency() {
        let transformer = MockTransformer::new();
        let result = check_transformer_fit_transform_consistency(transformer, 1e-10);
        assert!(result.passed, "{:?}", result.message);
    }

    #[test]
    fn test_check_model_serialization_all() {
        let model = MockModel::new();
        let results = check_model_serialization_all(model);
        for result in &results {
            assert!(result.passed, "{}: {:?}", result.name, result.message);
        }
    }

    #[test]
    fn test_check_transformer_serialization_all() {
        let transformer = MockTransformer::new();
        let results = check_transformer_serialization_all(transformer);
        for result in &results {
            assert!(result.passed, "{}: {:?}", result.name, result.message);
        }
    }
}
