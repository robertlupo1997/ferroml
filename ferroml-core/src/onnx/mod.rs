//! ONNX Export Module
//!
//! This module provides functionality to export FerroML models to the ONNX format
//! for deployment without the Python runtime.
//!
//! ## Supported Models
//!
//! ### Linear Models
//! - `LinearRegression` - Exported using MatMul + Add ops
//! - `LogisticRegression` - Exported using MatMul + Add + Sigmoid ops (binary) or Softmax (multiclass)
//! - `RidgeRegression` - Exported using MatMul + Add ops
//! - `LassoRegression` - Exported using MatMul + Add ops
//! - `ElasticNet` - Exported using MatMul + Add ops
//!
//! ### Tree Models
//! - `DecisionTreeClassifier` - Exported using TreeEnsembleClassifier (ML domain)
//! - `DecisionTreeRegressor` - Exported using TreeEnsembleRegressor (ML domain)
//! - `RandomForestClassifier` - Exported using TreeEnsembleClassifier (ML domain)
//! - `RandomForestRegressor` - Exported using TreeEnsembleRegressor (ML domain)
//! - `GradientBoostingClassifier` - Exported using TreeEnsembleClassifier (ML domain)
//! - `GradientBoostingRegressor` - Exported using TreeEnsembleRegressor (ML domain)
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::models::linear::LinearRegression;
//! use ferroml_core::models::Model;
//! use ferroml_core::onnx::{OnnxExportable, OnnxConfig};
//!
//! let mut model = LinearRegression::new();
//! model.fit(&x, &y)?;
//!
//! // Export to ONNX
//! let config = OnnxConfig::new("my_model");
//! model.export_onnx("model.onnx", &config)?;
//!
//! // Or get bytes for in-memory use
//! let bytes = model.to_onnx(&config)?;
//! ```

mod linear;
mod protos;
mod tree;

pub use protos::*;

use crate::{FerroError, Result};
use prost::Message;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// ONNX IR version (IR version 9 for ONNX 1.14+)
pub const ONNX_IR_VERSION: i64 = 9;

/// Default ONNX opset version for standard operators
pub const ONNX_OPSET_VERSION: i64 = 18;

/// ONNX-ML opset version for tree ensemble operators
pub const ONNX_ML_OPSET_VERSION: i64 = 3;

/// Configuration for ONNX export
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// Model name
    pub model_name: String,
    /// Optional description
    pub description: Option<String>,
    /// Producer name (default: "FerroML")
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Input name (default: "input")
    pub input_name: String,
    /// Output name (default: "output")
    pub output_name: String,
    /// Optional input shape override (batch_size, n_features)
    /// If None, uses dynamic batch size
    pub input_shape: Option<(i64, i64)>,
    /// ONNX opset version for standard ops
    pub opset_version: i64,
    /// ONNX-ML opset version for ML ops
    pub ml_opset_version: i64,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            model_name: "ferroml_model".to_string(),
            description: None,
            producer_name: "FerroML".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            input_name: "input".to_string(),
            output_name: "output".to_string(),
            input_shape: None,
            opset_version: ONNX_OPSET_VERSION,
            ml_opset_version: ONNX_ML_OPSET_VERSION,
        }
    }
}

impl OnnxConfig {
    /// Create a new ONNX config with the given model name
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            ..Default::default()
        }
    }

    /// Set the model description
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the input name
    #[must_use]
    pub fn with_input_name(mut self, name: impl Into<String>) -> Self {
        self.input_name = name.into();
        self
    }

    /// Set the output name
    #[must_use]
    pub fn with_output_name(mut self, name: impl Into<String>) -> Self {
        self.output_name = name.into();
        self
    }

    /// Set a fixed input shape (batch_size, n_features)
    #[must_use]
    pub fn with_input_shape(mut self, batch_size: i64, n_features: i64) -> Self {
        self.input_shape = Some((batch_size, n_features));
        self
    }

    /// Set the ONNX opset version
    #[must_use]
    pub fn with_opset_version(mut self, version: i64) -> Self {
        self.opset_version = version;
        self
    }
}

/// Trait for models that can be exported to ONNX format
pub trait OnnxExportable {
    /// Export the model to an ONNX file
    ///
    /// # Arguments
    /// * `path` - Output file path (typically with .onnx extension)
    /// * `config` - ONNX export configuration
    ///
    /// # Errors
    /// Returns an error if the model is not fitted or if file writing fails
    fn export_onnx<P: AsRef<Path>>(&self, path: P, config: &OnnxConfig) -> Result<()> {
        let bytes = self.to_onnx(config)?;
        let file = File::create(path.as_ref()).map_err(FerroError::IoError)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&bytes).map_err(FerroError::IoError)?;
        writer.flush().map_err(FerroError::IoError)?;
        Ok(())
    }

    /// Convert the model to ONNX bytes
    ///
    /// # Arguments
    /// * `config` - ONNX export configuration
    ///
    /// # Returns
    /// Serialized ONNX model as bytes
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>>;

    /// Get the number of input features
    fn onnx_n_features(&self) -> Option<usize>;

    /// Get the number of outputs
    fn onnx_n_outputs(&self) -> usize {
        1
    }

    /// Get the output element type
    fn onnx_output_type(&self) -> TensorProtoDataType {
        TensorProtoDataType::Float
    }
}

/// Create a ModelProto from components
pub fn create_model_proto(
    graph: GraphProto,
    config: &OnnxConfig,
    use_ml_domain: bool,
) -> ModelProto {
    let mut opset_imports = vec![OperatorSetIdProto {
        domain: String::new(), // Default ONNX domain
        version: config.opset_version,
    }];

    if use_ml_domain {
        opset_imports.push(OperatorSetIdProto {
            domain: "ai.onnx.ml".to_string(),
            version: config.ml_opset_version,
        });
    }

    ModelProto {
        ir_version: ONNX_IR_VERSION,
        opset_import: opset_imports,
        producer_name: config.producer_name.clone(),
        producer_version: config.producer_version.clone(),
        model_version: 1,
        doc_string: config.description.clone().unwrap_or_default(),
        graph: Some(graph),
        metadata_props: Vec::new(),
        domain: String::new(),
        training_info: Vec::new(),
        functions: Vec::new(),
    }
}

/// Create input ValueInfoProto for a 2D tensor
pub fn create_tensor_input(
    name: &str,
    n_features: usize,
    batch_size: Option<i64>,
    elem_type: TensorProtoDataType,
) -> ValueInfoProto {
    let batch_dim = match batch_size {
        Some(n) => TensorShapeProtoDimension {
            value: Some(tensor_shape_proto_dimension::Value::DimValue(n)),
        },
        None => TensorShapeProtoDimension {
            value: Some(tensor_shape_proto_dimension::Value::DimParam(
                "batch_size".to_string(),
            )),
        },
    };

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(TypeProtoTensor {
                elem_type: elem_type as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        batch_dim,
                        TensorShapeProtoDimension {
                            value: Some(tensor_shape_proto_dimension::Value::DimValue(
                                n_features as i64,
                            )),
                        },
                    ],
                }),
            })),
            denotation: String::new(),
        }),
        doc_string: String::new(),
    }
}

/// Create output ValueInfoProto for a 2D tensor
pub fn create_tensor_output(
    name: &str,
    n_outputs: usize,
    batch_size: Option<i64>,
    elem_type: TensorProtoDataType,
) -> ValueInfoProto {
    let batch_dim = match batch_size {
        Some(n) => TensorShapeProtoDimension {
            value: Some(tensor_shape_proto_dimension::Value::DimValue(n)),
        },
        None => TensorShapeProtoDimension {
            value: Some(tensor_shape_proto_dimension::Value::DimParam(
                "batch_size".to_string(),
            )),
        },
    };

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(TypeProtoTensor {
                elem_type: elem_type as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        batch_dim,
                        TensorShapeProtoDimension {
                            value: Some(tensor_shape_proto_dimension::Value::DimValue(
                                n_outputs as i64,
                            )),
                        },
                    ],
                }),
            })),
            denotation: String::new(),
        }),
        doc_string: String::new(),
    }
}

/// Create a 1D tensor output (for single-output models)
pub fn create_tensor_output_1d(
    name: &str,
    batch_size: Option<i64>,
    elem_type: TensorProtoDataType,
) -> ValueInfoProto {
    let batch_dim = match batch_size {
        Some(n) => TensorShapeProtoDimension {
            value: Some(tensor_shape_proto_dimension::Value::DimValue(n)),
        },
        None => TensorShapeProtoDimension {
            value: Some(tensor_shape_proto_dimension::Value::DimParam(
                "batch_size".to_string(),
            )),
        },
    };

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(TypeProtoTensor {
                elem_type: elem_type as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![batch_dim],
                }),
            })),
            denotation: String::new(),
        }),
        doc_string: String::new(),
    }
}

/// Create a float tensor initializer
pub fn create_float_tensor(name: &str, dims: &[i64], data: Vec<f32>) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type: TensorProtoDataType::Float as i32,
        float_data: data,
        name: name.to_string(),
        ..Default::default()
    }
}

/// Create a Gemm node (General Matrix Multiply): Y = alpha * A * B + beta * C
pub fn create_gemm_node(
    input_a: &str,
    input_b: &str,
    input_c: &str,
    output: &str,
    alpha: f32,
    beta: f32,
    trans_a: i64,
    trans_b: i64,
) -> NodeProto {
    NodeProto {
        input: vec![
            input_a.to_string(),
            input_b.to_string(),
            input_c.to_string(),
        ],
        output: vec![output.to_string()],
        name: "Gemm_0".to_string(),
        op_type: "Gemm".to_string(),
        domain: String::new(),
        attribute: vec![
            AttributeProto {
                name: "alpha".to_string(),
                f: alpha,
                r#type: AttributeProtoType::Float as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "beta".to_string(),
                f: beta,
                r#type: AttributeProtoType::Float as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "transA".to_string(),
                i: trans_a,
                r#type: AttributeProtoType::Int as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "transB".to_string(),
                i: trans_b,
                r#type: AttributeProtoType::Int as i32,
                ..Default::default()
            },
        ],
        doc_string: String::new(),
    }
}

/// Create a MatMul node: Y = A * B
pub fn create_matmul_node(input_a: &str, input_b: &str, output: &str, name: &str) -> NodeProto {
    NodeProto {
        input: vec![input_a.to_string(), input_b.to_string()],
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: "MatMul".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
        doc_string: String::new(),
    }
}

/// Create an Add node: Y = A + B
pub fn create_add_node(input_a: &str, input_b: &str, output: &str, name: &str) -> NodeProto {
    NodeProto {
        input: vec![input_a.to_string(), input_b.to_string()],
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: "Add".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
        doc_string: String::new(),
    }
}

/// Create a Sigmoid node: Y = 1 / (1 + exp(-X))
pub fn create_sigmoid_node(input: &str, output: &str, name: &str) -> NodeProto {
    NodeProto {
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: "Sigmoid".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
        doc_string: String::new(),
    }
}

/// Create a Softmax node
pub fn create_softmax_node(input: &str, output: &str, name: &str, axis: i64) -> NodeProto {
    NodeProto {
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: "Softmax".to_string(),
        domain: String::new(),
        attribute: vec![AttributeProto {
            name: "axis".to_string(),
            i: axis,
            r#type: AttributeProtoType::Int as i32,
            ..Default::default()
        }],
        doc_string: String::new(),
    }
}

/// Create a Flatten node to reshape to 1D output
pub fn create_flatten_node(input: &str, output: &str, name: &str, axis: i64) -> NodeProto {
    NodeProto {
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: "Flatten".to_string(),
        domain: String::new(),
        attribute: vec![AttributeProto {
            name: "axis".to_string(),
            i: axis,
            r#type: AttributeProtoType::Int as i32,
            ..Default::default()
        }],
        doc_string: String::new(),
    }
}

/// Create a Squeeze node to remove dimensions
pub fn create_squeeze_node(input: &str, axes_input: &str, output: &str, name: &str) -> NodeProto {
    NodeProto {
        input: vec![input.to_string(), axes_input.to_string()],
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: "Squeeze".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
        doc_string: String::new(),
    }
}

/// Create a Reshape node
pub fn create_reshape_node(input: &str, shape_input: &str, output: &str, name: &str) -> NodeProto {
    NodeProto {
        input: vec![input.to_string(), shape_input.to_string()],
        output: vec![output.to_string()],
        name: name.to_string(),
        op_type: "Reshape".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
        doc_string: String::new(),
    }
}

/// Create an int64 tensor initializer
pub fn create_int64_tensor(name: &str, dims: &[i64], data: Vec<i64>) -> TensorProto {
    TensorProto {
        dims: dims.to_vec(),
        data_type: TensorProtoDataType::Int64 as i32,
        int64_data: data,
        name: name.to_string(),
        ..Default::default()
    }
}

/// Serialize a ModelProto to bytes
pub fn serialize_model(model: &ModelProto) -> Vec<u8> {
    model.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_config_default() {
        let config = OnnxConfig::default();
        assert_eq!(config.model_name, "ferroml_model");
        assert_eq!(config.producer_name, "FerroML");
        assert_eq!(config.input_name, "input");
        assert_eq!(config.output_name, "output");
    }

    #[test]
    fn test_onnx_config_builder() {
        let config = OnnxConfig::new("my_model")
            .with_description("A test model")
            .with_input_name("X")
            .with_output_name("y");

        assert_eq!(config.model_name, "my_model");
        assert_eq!(config.description, Some("A test model".to_string()));
        assert_eq!(config.input_name, "X");
        assert_eq!(config.output_name, "y");
    }

    #[test]
    fn test_create_tensor_input() {
        let input = create_tensor_input("input", 5, None, TensorProtoDataType::Float);
        assert_eq!(input.name, "input");
        assert!(input.r#type.is_some());
    }

    #[test]
    fn test_create_float_tensor() {
        let tensor = create_float_tensor("weights", &[3, 1], vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.name, "weights");
        assert_eq!(tensor.dims, vec![3, 1]);
        assert_eq!(tensor.float_data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_create_matmul_node() {
        let node = create_matmul_node("A", "B", "C", "MatMul_0");
        assert_eq!(node.op_type, "MatMul");
        assert_eq!(node.input, vec!["A", "B"]);
        assert_eq!(node.output, vec!["C"]);
    }

    #[test]
    fn test_create_model_proto() {
        let graph = GraphProto {
            name: "test_graph".to_string(),
            ..Default::default()
        };
        let config = OnnxConfig::new("test_model");
        let model = create_model_proto(graph, &config, false);

        assert_eq!(model.ir_version, ONNX_IR_VERSION);
        assert_eq!(model.producer_name, "FerroML");
        assert_eq!(model.opset_import.len(), 1);
    }

    #[test]
    fn test_create_model_proto_with_ml_domain() {
        let graph = GraphProto {
            name: "test_graph".to_string(),
            ..Default::default()
        };
        let config = OnnxConfig::new("test_model");
        let model = create_model_proto(graph, &config, true);

        assert_eq!(model.opset_import.len(), 2);
        assert!(model
            .opset_import
            .iter()
            .any(|op| op.domain == "ai.onnx.ml"));
    }
}
