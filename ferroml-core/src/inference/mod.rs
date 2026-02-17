//! Pure-Rust ONNX Inference Module
//!
//! This module provides a pure-Rust runtime for executing ONNX models exported by FerroML,
//! enabling model deployment without the Python runtime.
//!
//! ## Supported Operators
//!
//! ### Standard ONNX Operators
//! - `MatMul` - Matrix multiplication
//! - `Add` - Element-wise addition with broadcasting
//! - `Squeeze` - Remove dimensions of size 1
//! - `Sigmoid` - Sigmoid activation function
//! - `Softmax` - Softmax activation function
//! - `Flatten` - Flatten tensor to specified axis
//! - `Reshape` - Reshape tensor
//!
//! ### ONNX-ML Operators
//! - `TreeEnsembleRegressor` - Tree ensemble for regression
//! - `TreeEnsembleClassifier` - Tree ensemble for classification
//!
//! ## Example
//!
//! ```
//! # use ferroml_core::inference::{InferenceSession, Tensor};
//! # use ferroml_core::models::linear::LinearRegression;
//! # use ferroml_core::models::Model;
//! # use ferroml_core::onnx::{OnnxConfig, OnnxExportable};
//! # use ndarray::{Array1, Array2};
//! # fn main() -> ferroml_core::Result<()> {
//! # let x = Array2::from_shape_vec((4, 2), vec![1.0,5.0,3.0,2.0,5.0,1.0,7.0,4.0]).unwrap();
//! # let y = Array1::from_vec(vec![5.0, 11.0, 17.0, 23.0]);
//! // Train and export a model
//! let mut model = LinearRegression::new();
//! model.fit(&x, &y)?;
//!
//! let config = OnnxConfig::new("my_model");
//! let onnx_bytes = model.to_onnx(&config)?;
//!
//! // Load and run with pure-Rust inference
//! let session = InferenceSession::from_bytes(&onnx_bytes)?;
//!
//! // Create input tensor
//! let input = Tensor::from_vec(vec![1.0f32, 2.0], vec![1, 2]);
//! let outputs = session.run(&[("input", input)])?;
//!
//! // Get prediction
//! let prediction = outputs.get("output").unwrap();
//! # Ok(())
//! # }
//! ```

mod operators;
mod session;
mod tensor;

pub use operators::*;
pub use session::*;
pub use tensor::*;

use crate::FerroError;
use std::collections::HashMap;

/// A value in the inference graph (tensor or other types)
#[derive(Debug, Clone)]
pub enum Value {
    /// Float tensor
    Tensor(Tensor),
    /// Integer tensor (for labels)
    TensorI64(TensorI64),
    /// Sequence of maps (for classifier probabilities)
    SequenceMapI64F32(Vec<HashMap<i64, f32>>),
}

impl Value {
    /// Get as float tensor, if applicable
    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Value::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Get as i64 tensor, if applicable
    pub fn as_tensor_i64(&self) -> Option<&TensorI64> {
        match self {
            Value::TensorI64(t) => Some(t),
            _ => None,
        }
    }

    /// Get as sequence of maps, if applicable
    pub fn as_sequence_map(&self) -> Option<&Vec<HashMap<i64, f32>>> {
        match self {
            Value::SequenceMapI64F32(s) => Some(s),
            _ => None,
        }
    }
}

/// Inference error types
#[derive(Debug)]
pub enum InferenceError {
    /// Invalid ONNX model format
    InvalidModel(String),
    /// Unsupported operator
    UnsupportedOperator(String),
    /// Shape mismatch during computation
    ShapeMismatch(String),
    /// Missing input
    MissingInput(String),
    /// Missing attribute
    MissingAttribute(String),
    /// Type mismatch
    TypeMismatch(String),
    /// General runtime error
    RuntimeError(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::InvalidModel(msg) => write!(f, "Invalid ONNX model: {msg}"),
            InferenceError::UnsupportedOperator(msg) => write!(f, "Unsupported operator: {msg}"),
            InferenceError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {msg}"),
            InferenceError::MissingInput(msg) => write!(f, "Missing input: {msg}"),
            InferenceError::MissingAttribute(msg) => write!(f, "Missing attribute: {msg}"),
            InferenceError::TypeMismatch(msg) => write!(f, "Type mismatch: {msg}"),
            InferenceError::RuntimeError(msg) => write!(f, "Runtime error: {msg}"),
        }
    }
}

impl std::error::Error for InferenceError {}

impl From<InferenceError> for FerroError {
    fn from(err: InferenceError) -> Self {
        FerroError::InferenceError(err.to_string())
    }
}
