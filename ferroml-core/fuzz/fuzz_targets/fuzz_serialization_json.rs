//! Fuzz target for JSON deserialization
//!
//! This fuzzer tests the robustness of JSON deserialization for model containers.
//! It attempts to deserialize arbitrary bytes as various model types to find
//! edge cases that might cause panics, hangs, or other issues.

#![no_main]

use libfuzzer_sys::fuzz_target;

use ferroml_core::serialization::{from_bytes, Format, ModelContainer, SerializationMetadata};
use serde::{Deserialize, Serialize};

/// A minimal serializable struct for fuzzing
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FuzzModel {
    coefficients: Vec<f64>,
    intercept: f64,
    fitted: bool,
    metadata: Option<String>,
}

/// Test deserialization of ModelContainer with FuzzModel
fn fuzz_model_container(data: &[u8]) {
    // Try to deserialize as a model container
    let _result: Result<FuzzModel, _> = from_bytes(data, Format::Json);
    // We don't care if it fails, just that it doesn't panic or hang
}

/// Test deserialization of raw JSON structures that might be model-like
fn fuzz_raw_json(data: &[u8]) {
    // Try parsing as generic JSON value first
    let _result: Result<serde_json::Value, _> = serde_json::from_slice(data);

    // Also try as a map
    let _result: Result<std::collections::HashMap<String, serde_json::Value>, _> =
        serde_json::from_slice(data);
}

/// Test deserialization with nested structures
fn fuzz_nested_model(data: &[u8]) {
    #[derive(Debug, Deserialize)]
    struct NestedModel {
        metadata: Option<SerializationMetadata>,
        model: Option<FuzzModel>,
        extra: Option<Vec<f64>>,
    }

    let _result: Result<NestedModel, _> = serde_json::from_slice(data);
}

/// Test deserialization of arrays and vectors commonly used in ML
fn fuzz_arrays(data: &[u8]) {
    // 1D array
    let _result: Result<Vec<f64>, _> = serde_json::from_slice(data);

    // 2D array (as nested vecs)
    let _result: Result<Vec<Vec<f64>>, _> = serde_json::from_slice(data);

    // Array with mixed numeric types
    let _result: Result<Vec<serde_json::Number>, _> = serde_json::from_slice(data);
}

fuzz_target!(|data: &[u8]| {
    // Run all fuzzing functions
    fuzz_model_container(data);
    fuzz_raw_json(data);
    fuzz_nested_model(data);
    fuzz_arrays(data);
});
