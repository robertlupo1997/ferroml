//! Fuzz target for MessagePack deserialization
//!
//! This fuzzer tests the robustness of MessagePack deserialization for model containers.
//! MessagePack is a binary format, so malformed input can have different failure modes
//! compared to JSON.

#![no_main]

use libfuzzer_sys::fuzz_target;

use ferroml_core::serialization::{from_bytes, Format};
use serde::{Deserialize, Serialize};

/// A minimal serializable struct for fuzzing
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FuzzModel {
    coefficients: Vec<f64>,
    intercept: f64,
    fitted: bool,
    metadata: Option<String>,
}

/// Extended model with more fields to test edge cases
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExtendedModel {
    // Numeric fields
    f32_val: f32,
    f64_val: f64,
    i32_val: i32,
    i64_val: i64,
    u32_val: u32,
    u64_val: u64,

    // String fields
    name: String,
    description: Option<String>,

    // Collection fields
    weights: Vec<f64>,
    labels: Vec<String>,
    nested_vecs: Vec<Vec<f64>>,

    // Boolean
    is_fitted: bool,
}

/// Test deserialization of ModelContainer with FuzzModel
fn fuzz_model_container(data: &[u8]) {
    // Try to deserialize as a model container
    let _result: Result<FuzzModel, _> = from_bytes(data, Format::MessagePack);
}

/// Test raw MessagePack deserialization
fn fuzz_raw_msgpack(data: &[u8]) {
    // Try as generic value
    let _result: Result<rmp_serde::Value, _> = rmp_serde::from_slice(data);

    // Try as map
    let _result: Result<std::collections::HashMap<String, rmp_serde::Value>, _> =
        rmp_serde::from_slice(data);
}

/// Test extended model deserialization
fn fuzz_extended_model(data: &[u8]) {
    let _result: Result<ExtendedModel, _> = rmp_serde::from_slice(data);
}

/// Test array deserialization
fn fuzz_arrays(data: &[u8]) {
    // 1D arrays
    let _result: Result<Vec<f64>, _> = rmp_serde::from_slice(data);
    let _result: Result<Vec<i64>, _> = rmp_serde::from_slice(data);
    let _result: Result<Vec<u8>, _> = rmp_serde::from_slice(data);

    // 2D arrays
    let _result: Result<Vec<Vec<f64>>, _> = rmp_serde::from_slice(data);
}

/// Test tuple deserialization (commonly used for model parameters)
fn fuzz_tuples(data: &[u8]) {
    let _result: Result<(f64, f64), _> = rmp_serde::from_slice(data);
    let _result: Result<(Vec<f64>, f64), _> = rmp_serde::from_slice(data);
    let _result: Result<(String, Vec<f64>, bool), _> = rmp_serde::from_slice(data);
}

fuzz_target!(|data: &[u8]| {
    fuzz_model_container(data);
    fuzz_raw_msgpack(data);
    fuzz_extended_model(data);
    fuzz_arrays(data);
    fuzz_tuples(data);
});
