//! Inference Session for ONNX Model Execution
//!
//! Provides the `InferenceSession` struct that loads and executes ONNX models.

use super::{
    AddOp, FlattenOp, InferenceError, MatMulOp, Operator, ReshapeOp, SigmoidOp, SoftmaxOp,
    SqueezeOp, Tensor, TensorI64, TreeEnsembleClassifierOp, TreeEnsembleRegressorOp, Value,
};
use crate::onnx::{AttributeProtoType, GraphProto, ModelProto, NodeProto, TensorProto};
use crate::Result;
use prost::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

/// An inference session for executing ONNX models
///
/// The session loads an ONNX model and provides methods to run inference
/// without requiring the Python runtime.
pub struct InferenceSession {
    /// Model graph
    graph: GraphProto,
    /// Initialized tensors (weights, biases)
    initializers: HashMap<String, Value>,
    /// Input names
    input_names: Vec<String>,
    /// Output names
    output_names: Vec<String>,
    /// Compiled operators for each node
    operators: Vec<(NodeProto, Arc<dyn Operator>)>,
}

impl InferenceSession {
    /// Create a session from ONNX bytes
    ///
    /// # Arguments
    /// * `bytes` - Serialized ONNX model
    ///
    /// # Example
    ///
    /// ```ignore
    /// let session = InferenceSession::from_bytes(&onnx_bytes)?;
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let model =
            ModelProto::decode(bytes).map_err(|e| InferenceError::InvalidModel(e.to_string()))?;

        Self::from_model(model)
    }

    /// Create a session from an ONNX file
    ///
    /// # Arguments
    /// * `path` - Path to the ONNX file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file =
            File::open(path.as_ref()).map_err(|e| InferenceError::InvalidModel(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .map_err(|e| InferenceError::InvalidModel(e.to_string()))?;

        Self::from_bytes(&bytes)
    }

    /// Create a session from a ModelProto
    fn from_model(model: ModelProto) -> Result<Self> {
        let graph = model
            .graph
            .ok_or_else(|| InferenceError::InvalidModel("Model has no graph".to_string()))?;

        // Extract initializers (weights, biases)
        let mut initializers = HashMap::new();
        for tensor in &graph.initializer {
            let value = tensor_proto_to_value(tensor)?;
            initializers.insert(tensor.name.clone(), value);
        }

        // Extract input/output names
        let input_names: Vec<String> = graph
            .input
            .iter()
            .map(|i| i.name.clone())
            .filter(|name| !initializers.contains_key(name))
            .collect();

        let output_names: Vec<String> = graph.output.iter().map(|o| o.name.clone()).collect();

        // Compile operators
        let mut operators = Vec::new();
        for node in &graph.node {
            let op = compile_operator(node)?;
            operators.push((node.clone(), op));
        }

        Ok(Self {
            graph,
            initializers,
            input_names,
            output_names,
            operators,
        })
    }

    /// Get input names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Run inference with named inputs
    ///
    /// # Arguments
    /// * `inputs` - Slice of (name, tensor) pairs
    ///
    /// # Returns
    /// Map of output name to output value
    ///
    /// # Example
    ///
    /// ```ignore
    /// let input = Tensor::from_vec(vec![1.0f32, 2.0], vec![1, 2]);
    /// let outputs = session.run(&[("input", input)])?;
    /// let prediction = outputs.get("output").unwrap();
    /// ```
    pub fn run(&self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Value>> {
        let mut values: HashMap<String, Value> = self.initializers.clone();

        // Add inputs
        for (name, tensor) in inputs {
            values.insert((*name).to_string(), Value::Tensor(tensor.clone()));
        }

        // Execute nodes in topological order
        for (node, op) in &self.operators {
            // Gather inputs
            let input_values: Vec<&Value> = node
                .input
                .iter()
                .filter_map(|name| values.get(name))
                .collect();

            if input_values.len() < node.input.iter().filter(|n| !n.is_empty()).count() {
                let missing: Vec<_> = node
                    .input
                    .iter()
                    .filter(|n| !n.is_empty() && !values.contains_key(*n))
                    .collect();
                return Err(InferenceError::MissingInput(format!(
                    "Node {} missing inputs: {:?}",
                    node.name, missing
                ))
                .into());
            }

            // Execute operator
            let outputs = op.execute(&input_values)?;

            // Store outputs
            for (i, output_name) in node.output.iter().enumerate() {
                if !output_name.is_empty() && i < outputs.len() {
                    values.insert(output_name.clone(), outputs[i].clone());
                }
            }
        }

        // Collect outputs
        let mut result = HashMap::new();
        for name in &self.output_names {
            if let Some(value) = values.get(name) {
                result.insert(name.clone(), value.clone());
            }
        }

        Ok(result)
    }

    /// Run inference and return output as tensor (convenience method)
    ///
    /// Works for models with a single output or when you want the first output.
    pub fn run_single(&self, inputs: &[(&str, Tensor)]) -> Result<Tensor> {
        let outputs = self.run(inputs)?;

        // Return first output
        let output_name = self
            .output_names
            .first()
            .ok_or_else(|| InferenceError::RuntimeError("Model has no outputs".to_string()))?;

        outputs
            .get(output_name)
            .and_then(|v| v.as_tensor())
            .cloned()
            .ok_or_else(|| {
                InferenceError::TypeMismatch("Output is not a tensor".to_string()).into()
            })
    }

    /// Get model metadata (producer name, version, etc.)
    pub fn metadata(&self) -> SessionMetadata {
        SessionMetadata {
            graph_name: self.graph.name.clone(),
            doc_string: self.graph.doc_string.clone(),
            n_inputs: self.input_names.len(),
            n_outputs: self.output_names.len(),
            n_nodes: self.operators.len(),
        }
    }
}

/// Metadata about the inference session
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    /// Name of the computational graph
    pub graph_name: String,
    /// Documentation string
    pub doc_string: String,
    /// Number of inputs
    pub n_inputs: usize,
    /// Number of outputs
    pub n_outputs: usize,
    /// Number of nodes in the graph
    pub n_nodes: usize,
}

/// Convert TensorProto to Value
fn tensor_proto_to_value(tensor: &TensorProto) -> Result<Value> {
    let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();

    // Check data type and extract data
    if !tensor.float_data.is_empty() {
        let data = tensor.float_data.clone();
        return Ok(Value::Tensor(Tensor::from_vec(data, shape)));
    }

    if !tensor.int64_data.is_empty() {
        let data = tensor.int64_data.clone();
        return Ok(Value::TensorI64(TensorI64::from_vec(data, shape)));
    }

    if !tensor.raw_data.is_empty() {
        // Determine element type and parse raw_data
        let elem_type = tensor.data_type;
        if elem_type == 1 {
            // FLOAT
            let data: Vec<f32> = tensor
                .raw_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            return Ok(Value::Tensor(Tensor::from_vec(data, shape)));
        }
        if elem_type == 7 {
            // INT64
            let data: Vec<i64> = tensor
                .raw_data
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            return Ok(Value::TensorI64(TensorI64::from_vec(data, shape)));
        }
    }

    // Default to empty float tensor
    Ok(Value::Tensor(Tensor::zeros(shape)))
}

/// Compile a node into an operator
fn compile_operator(node: &NodeProto) -> Result<Arc<dyn Operator>> {
    let op: Arc<dyn Operator> = match node.op_type.as_str() {
        "MatMul" => Arc::new(MatMulOp),
        "Add" => Arc::new(AddOp),
        "Sigmoid" => Arc::new(SigmoidOp),
        "Squeeze" => {
            let axes = get_attribute_ints(node, "axes").unwrap_or_default();
            Arc::new(SqueezeOp { axes })
        }
        "Softmax" => {
            let axis = get_attribute_int(node, "axis").unwrap_or(1);
            Arc::new(SoftmaxOp { axis })
        }
        "Flatten" => {
            let axis = get_attribute_int(node, "axis").unwrap_or(1);
            Arc::new(FlattenOp { axis })
        }
        "Reshape" => Arc::new(ReshapeOp),
        "TreeEnsembleRegressor" => {
            let n_targets = get_attribute_int(node, "n_targets").unwrap_or(1);
            let nodes_featureids = get_attribute_ints(node, "nodes_featureids").unwrap_or_default();
            let nodes_values = get_attribute_floats(node, "nodes_values").unwrap_or_default();
            let nodes_modes = get_attribute_strings(node, "nodes_modes").unwrap_or_default();
            let nodes_treeids = get_attribute_ints(node, "nodes_treeids").unwrap_or_default();
            let nodes_nodeids = get_attribute_ints(node, "nodes_nodeids").unwrap_or_default();
            let nodes_truenodeids =
                get_attribute_ints(node, "nodes_truenodeids").unwrap_or_default();
            let nodes_falsenodeids =
                get_attribute_ints(node, "nodes_falsenodeids").unwrap_or_default();
            let nodes_missing =
                get_attribute_ints(node, "nodes_missing_value_tracks_true").unwrap_or_default();
            let target_nodeids = get_attribute_ints(node, "target_nodeids").unwrap_or_default();
            let target_treeids = get_attribute_ints(node, "target_treeids").unwrap_or_default();
            let target_ids = get_attribute_ints(node, "target_ids").unwrap_or_default();
            let target_weights = get_attribute_floats(node, "target_weights").unwrap_or_default();
            let aggregate = get_attribute_string(node, "aggregate_function").unwrap_or_default();
            let base_values = get_attribute_floats(node, "base_values").unwrap_or_default();

            Arc::new(TreeEnsembleRegressorOp::from_attributes(
                n_targets,
                &nodes_featureids,
                &nodes_values,
                &nodes_modes,
                &nodes_treeids,
                &nodes_nodeids,
                &nodes_truenodeids,
                &nodes_falsenodeids,
                &nodes_missing,
                &target_nodeids,
                &target_treeids,
                &target_ids,
                &target_weights,
                &aggregate,
                &base_values,
            )?)
        }
        "TreeEnsembleClassifier" => {
            let class_labels = get_attribute_ints(node, "classlabels_int64s").unwrap_or_default();
            let nodes_featureids = get_attribute_ints(node, "nodes_featureids").unwrap_or_default();
            let nodes_values = get_attribute_floats(node, "nodes_values").unwrap_or_default();
            let nodes_modes = get_attribute_strings(node, "nodes_modes").unwrap_or_default();
            let nodes_treeids = get_attribute_ints(node, "nodes_treeids").unwrap_or_default();
            let nodes_nodeids = get_attribute_ints(node, "nodes_nodeids").unwrap_or_default();
            let nodes_truenodeids =
                get_attribute_ints(node, "nodes_truenodeids").unwrap_or_default();
            let nodes_falsenodeids =
                get_attribute_ints(node, "nodes_falsenodeids").unwrap_or_default();
            let nodes_missing =
                get_attribute_ints(node, "nodes_missing_value_tracks_true").unwrap_or_default();
            let class_nodeids = get_attribute_ints(node, "class_nodeids").unwrap_or_default();
            let class_treeids = get_attribute_ints(node, "class_treeids").unwrap_or_default();
            let class_ids = get_attribute_ints(node, "class_ids").unwrap_or_default();
            let class_weights = get_attribute_floats(node, "class_weights").unwrap_or_default();
            let post_transform = get_attribute_string(node, "post_transform").unwrap_or_default();
            let base_values = get_attribute_floats(node, "base_values").unwrap_or_default();

            Arc::new(TreeEnsembleClassifierOp::from_attributes(
                &class_labels,
                &nodes_featureids,
                &nodes_values,
                &nodes_modes,
                &nodes_treeids,
                &nodes_nodeids,
                &nodes_truenodeids,
                &nodes_falsenodeids,
                &nodes_missing,
                &class_nodeids,
                &class_treeids,
                &class_ids,
                &class_weights,
                &post_transform,
                &base_values,
            )?)
        }
        other => {
            return Err(InferenceError::UnsupportedOperator(other.to_string()).into());
        }
    };

    Ok(op)
}

// =============================================================================
// Attribute extraction helpers
// =============================================================================

fn get_attribute_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|a| a.name == name && a.r#type == AttributeProtoType::Int as i32)
        .map(|a| a.i)
}

fn get_attribute_ints(node: &NodeProto, name: &str) -> Option<Vec<i64>> {
    node.attribute
        .iter()
        .find(|a| a.name == name && a.r#type == AttributeProtoType::Ints as i32)
        .map(|a| a.ints.clone())
}

fn get_attribute_floats(node: &NodeProto, name: &str) -> Option<Vec<f32>> {
    node.attribute
        .iter()
        .find(|a| a.name == name && a.r#type == AttributeProtoType::Floats as i32)
        .map(|a| a.floats.clone())
}

fn get_attribute_string(node: &NodeProto, name: &str) -> Option<String> {
    node.attribute
        .iter()
        .find(|a| a.name == name && a.r#type == AttributeProtoType::String as i32)
        .map(|a| String::from_utf8_lossy(&a.s).to_string())
}

fn get_attribute_strings(node: &NodeProto, name: &str) -> Option<Vec<Vec<u8>>> {
    node.attribute
        .iter()
        .find(|a| a.name == name && a.r#type == AttributeProtoType::Strings as i32)
        .map(|a| a.strings.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;
    use crate::models::Model;
    use crate::onnx::{OnnxConfig, OnnxExportable};
    use ndarray::{Array1, Array2};

    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 1.5, 3.0, 3.5, 4.0, 2.0, 5.0, 4.5, 6.0, 3.0, 7.0, 5.5, 8.0, 4.0,
                9.0, 6.5, 10.0, 5.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            4.0, 5.5, 9.5, 10.0, 14.5, 15.0, 19.5, 20.0, 24.5, 25.0,
        ]);
        (x, y)
    }

    #[test]
    fn test_linear_regression_inference() {
        let (x, y) = create_test_data();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Export to ONNX
        let config = OnnxConfig::new("linear_test");
        let onnx_bytes = model.to_onnx(&config).unwrap();

        // Create inference session
        let session = InferenceSession::from_bytes(&onnx_bytes).unwrap();

        // Check metadata
        let meta = session.metadata();
        assert_eq!(meta.graph_name, "linear_test");

        // Run inference
        let test_input = Tensor::from_vec(vec![6.0f32, 3.0], vec![1, 2]);
        let output = session.run_single(&[("input", test_input)]).unwrap();

        // Should get a reasonable prediction
        assert_eq!(output.shape(), &[1]);
        let pred = output.as_slice()[0];
        // y ≈ 2*x1 + x2 + 1, so for [6, 3] we expect ~16
        assert!(
            pred > 10.0 && pred < 20.0,
            "Prediction {} out of range",
            pred
        );
    }

    #[test]
    fn test_linear_regression_batch_inference() {
        let (x, y) = create_test_data();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("linear_batch_test");
        let onnx_bytes = model.to_onnx(&config).unwrap();
        let session = InferenceSession::from_bytes(&onnx_bytes).unwrap();

        // Batch of 3 samples
        let test_input = Tensor::from_vec(vec![1.0f32, 2.0, 5.0, 4.5, 10.0, 5.0], vec![3, 2]);
        let output = session.run_single(&[("input", test_input)]).unwrap();

        assert_eq!(output.shape(), &[3]);
    }

    #[test]
    fn test_session_metadata() {
        let (x, y) = create_test_data();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("meta_test").with_description("Test model");
        let onnx_bytes = model.to_onnx(&config).unwrap();
        let session = InferenceSession::from_bytes(&onnx_bytes).unwrap();

        let meta = session.metadata();
        assert_eq!(meta.graph_name, "meta_test");
        assert_eq!(meta.n_inputs, 1);
        assert_eq!(meta.n_outputs, 1);
    }

    #[test]
    fn test_input_output_names() {
        let (x, y) = create_test_data();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("names_test")
            .with_input_name("X")
            .with_output_name("y");
        let onnx_bytes = model.to_onnx(&config).unwrap();
        let session = InferenceSession::from_bytes(&onnx_bytes).unwrap();

        assert_eq!(session.input_names(), &["X"]);
        assert_eq!(session.output_names(), &["y"]);
    }
}
