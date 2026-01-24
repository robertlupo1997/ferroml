//! Fuzz target for Bincode deserialization
//!
//! This fuzzer tests the robustness of Bincode deserialization for model containers.
//! Bincode is the fastest binary format used by FerroML, and it includes magic bytes
//! validation that should be tested.

#![no_main]

use libfuzzer_sys::fuzz_target;

use ferroml_core::serialization::{from_bytes, Format};
use serde::{Deserialize, Serialize};

/// Magic bytes used by FerroML bincode format
const FERROML_MAGIC: [u8; 4] = *b"FRML";

/// A minimal serializable struct for fuzzing
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FuzzModel {
    coefficients: Vec<f64>,
    intercept: f64,
    fitted: bool,
    metadata: Option<String>,
}

/// Extended model with various field types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExtendedModel {
    // Numeric fields
    f32_val: f32,
    f64_val: f64,
    i32_val: i32,
    i64_val: i64,
    usize_val: usize,

    // String fields
    name: String,

    // Collection fields
    weights: Vec<f64>,
    nested: Vec<Vec<f64>>,

    // Optional fields
    optional_vec: Option<Vec<f64>>,
    optional_string: Option<String>,

    // Boolean
    is_fitted: bool,
}

/// Test deserialization with magic bytes check
fn fuzz_with_magic_bytes(data: &[u8]) {
    // Try to deserialize through the ferroml API (includes magic byte validation)
    let _result: Result<FuzzModel, _> = from_bytes(data, Format::Bincode);
}

/// Test deserialization without magic bytes (raw bincode)
fn fuzz_raw_bincode(data: &[u8]) {
    // Try raw bincode deserialization without magic bytes
    let _result: Result<FuzzModel, _> = bincode::deserialize(data);
    let _result: Result<ExtendedModel, _> = bincode::deserialize(data);
}

/// Test deserialization with valid magic prefix but corrupted data
fn fuzz_valid_prefix_corrupted_data(data: &[u8]) {
    if data.len() >= 4 {
        // Create data with valid magic prefix
        let mut prefixed = FERROML_MAGIC.to_vec();
        prefixed.extend_from_slice(data);

        let _result: Result<FuzzModel, _> = from_bytes(&prefixed, Format::Bincode);
    }
}

/// Test deserialization of arrays
fn fuzz_arrays(data: &[u8]) {
    // 1D arrays of various types
    let _result: Result<Vec<f64>, _> = bincode::deserialize(data);
    let _result: Result<Vec<f32>, _> = bincode::deserialize(data);
    let _result: Result<Vec<i64>, _> = bincode::deserialize(data);
    let _result: Result<Vec<u8>, _> = bincode::deserialize(data);

    // 2D arrays
    let _result: Result<Vec<Vec<f64>>, _> = bincode::deserialize(data);

    // Fixed-size arrays
    let _result: Result<[f64; 4], _> = bincode::deserialize(data);
    let _result: Result<[f64; 16], _> = bincode::deserialize(data);
}

/// Test deserialization of enum types (like Format)
fn fuzz_enums(data: &[u8]) {
    #[derive(Debug, Deserialize)]
    enum TestEnum {
        VariantA,
        VariantB(i32),
        VariantC { x: f64, y: f64 },
    }

    let _result: Result<TestEnum, _> = bincode::deserialize(data);
}

/// Test deserialization of deeply nested structures
fn fuzz_deep_nesting(data: &[u8]) {
    #[derive(Debug, Deserialize)]
    struct DeepStruct {
        level1: Option<Box<DeepStruct>>,
        data: Vec<f64>,
    }

    // Use bincode config with limited depth to avoid stack overflow
    let config = bincode::config::standard()
        .with_limit::<1024>(); // Limit to 1KB to prevent OOM

    let _result: Result<DeepStruct, _> = bincode::decode_from_slice(data, config);
}

fuzz_target!(|data: &[u8]| {
    fuzz_with_magic_bytes(data);
    fuzz_raw_bincode(data);
    fuzz_valid_prefix_corrupted_data(data);
    fuzz_arrays(data);
    fuzz_enums(data);
    fuzz_deep_nesting(data);
});
