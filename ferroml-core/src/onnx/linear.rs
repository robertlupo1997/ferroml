//! ONNX Export for Linear Models
//!
//! Implements ONNX export for LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet.

use crate::models::linear::LinearRegression;
use crate::models::logistic::LogisticRegression;
use crate::models::regularized::{ElasticNet, LassoRegression, RidgeRegression};
use crate::models::Model;
use crate::onnx::{
    create_add_node, create_float_tensor, create_int64_tensor, create_matmul_node,
    create_model_proto, create_sigmoid_node, create_squeeze_node, create_tensor_input,
    create_tensor_output, create_tensor_output_1d, GraphProto, OnnxConfig, OnnxExportable,
    TensorProtoDataType,
};
use crate::{FerroError, Result};
use prost::Message;

/// Helper to create a linear regression graph (used by multiple linear models)
fn create_linear_graph(
    coefficients: &[f64],
    intercept: f64,
    n_features: usize,
    config: &OnnxConfig,
) -> GraphProto {
    let batch_size = config.input_shape.map(|(b, _)| b);

    // Input: [batch_size, n_features]
    let input = create_tensor_input(
        &config.input_name,
        n_features,
        batch_size,
        TensorProtoDataType::Float,
    );

    // Output: [batch_size]
    let output =
        create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

    // Weights: [n_features, 1]
    let weights: Vec<f32> = coefficients.iter().map(|&c| c as f32).collect();
    let weights_tensor = create_float_tensor("weights", &[n_features as i64, 1], weights);

    // Bias: [1]
    let bias_tensor = create_float_tensor("bias", &[1], vec![intercept as f32]);

    // Shape for squeeze: [-1] to flatten last dim
    let squeeze_axes = create_int64_tensor("squeeze_axes", &[1], vec![1]);

    // Nodes:
    // 1. MatMul: input @ weights -> matmul_out [batch_size, 1]
    let matmul_node = create_matmul_node(&config.input_name, "weights", "matmul_out", "MatMul_0");

    // 2. Add: matmul_out + bias -> add_out [batch_size, 1]
    let add_node = create_add_node("matmul_out", "bias", "add_out", "Add_0");

    // 3. Squeeze: add_out -> output [batch_size]
    let squeeze_node =
        create_squeeze_node("add_out", "squeeze_axes", &config.output_name, "Squeeze_0");

    GraphProto {
        name: config.model_name.clone(),
        node: vec![matmul_node, add_node, squeeze_node],
        input: vec![input],
        output: vec![output],
        initializer: vec![weights_tensor, bias_tensor, squeeze_axes],
        ..Default::default()
    }
}

impl OnnxExportable for LinearRegression {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        // Check if model is fitted
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let coefficients = self
            .coefficients()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let intercept = self.intercept().unwrap_or(0.0);
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let coef_slice: Vec<f64> = coefficients.iter().copied().collect();
        let graph = create_linear_graph(&coef_slice, intercept, n_features, config);

        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for RidgeRegression {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let coefficients = self
            .coefficients()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let intercept = self.intercept().unwrap_or(0.0);
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let coef_slice: Vec<f64> = coefficients.iter().copied().collect();
        let graph = create_linear_graph(&coef_slice, intercept, n_features, config);

        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for LassoRegression {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let coefficients = self
            .coefficients()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let intercept = self.intercept().unwrap_or(0.0);
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let coef_slice: Vec<f64> = coefficients.iter().copied().collect();
        let graph = create_linear_graph(&coef_slice, intercept, n_features, config);

        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for ElasticNet {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let coefficients = self
            .coefficients()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let intercept = self.intercept().unwrap_or(0.0);
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let coef_slice: Vec<f64> = coefficients.iter().copied().collect();
        let graph = create_linear_graph(&coef_slice, intercept, n_features, config);

        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

/// Helper to create a logistic regression graph for binary classification
fn create_binary_logistic_graph(
    coefficients: &[f64],
    intercept: f64,
    n_features: usize,
    config: &OnnxConfig,
) -> GraphProto {
    let batch_size = config.input_shape.map(|(b, _)| b);

    // Input: [batch_size, n_features]
    let input = create_tensor_input(
        &config.input_name,
        n_features,
        batch_size,
        TensorProtoDataType::Float,
    );

    // Output: [batch_size, 1] probabilities
    let output = create_tensor_output(
        &config.output_name,
        1,
        batch_size,
        TensorProtoDataType::Float,
    );

    // Weights: [n_features, 1]
    let weights: Vec<f32> = coefficients.iter().map(|&c| c as f32).collect();
    let weights_tensor = create_float_tensor("weights", &[n_features as i64, 1], weights);

    // Bias: [1]
    let bias_tensor = create_float_tensor("bias", &[1], vec![intercept as f32]);

    // Nodes:
    // 1. MatMul: input @ weights -> matmul_out [batch_size, 1]
    let matmul_node = create_matmul_node(&config.input_name, "weights", "matmul_out", "MatMul_0");

    // 2. Add: matmul_out + bias -> add_out [batch_size, 1]
    let add_node = create_add_node("matmul_out", "bias", "logits", "Add_0");

    // 3. Sigmoid: logits -> output [batch_size, 1]
    let sigmoid_node = create_sigmoid_node("logits", &config.output_name, "Sigmoid_0");

    GraphProto {
        name: config.model_name.clone(),
        node: vec![matmul_node, add_node, sigmoid_node],
        input: vec![input],
        output: vec![output],
        initializer: vec![weights_tensor, bias_tensor],
        ..Default::default()
    }
}

impl OnnxExportable for LogisticRegression {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        // LogisticRegression is binary-only in FerroML
        let coefficients = self
            .coefficients()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let intercept = self.intercept().unwrap_or(0.0);
        let coef_slice: Vec<f64> = coefficients.iter().copied().collect();
        let graph = create_binary_logistic_graph(&coef_slice, intercept, n_features, config);

        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }

    fn onnx_n_outputs(&self) -> usize {
        1 // Binary classification outputs probability for class 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;
    use crate::onnx::ModelProto;
    use ndarray::{Array1, Array2};

    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        // Non-collinear features to avoid rank-deficient matrix
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 1.5, 3.0, 3.5, 4.0, 2.0, 5.0, 4.5, 6.0, 3.0, 7.0, 5.5, 8.0, 4.0,
                9.0, 6.5, 10.0, 5.0,
            ],
        )
        .unwrap();
        // y = 2*x1 + x2 + noise
        let y = Array1::from_vec(vec![
            4.0, 5.5, 9.5, 10.0, 14.5, 15.0, 19.5, 20.0, 24.5, 25.0,
        ]);
        (x, y)
    }

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 4.0, 5.0, 5.0, 6.0, 4.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    #[test]
    fn test_linear_regression_onnx_export() {
        let (x, y) = create_test_data();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("linear_regression_test");
        let bytes = model.to_onnx(&config).unwrap();

        // Verify we can decode the ONNX model
        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert_eq!(onnx_model.producer_name, "FerroML");
        assert!(onnx_model.graph.is_some());

        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.name, "linear_regression_test");
        assert_eq!(graph.input.len(), 1);
        assert_eq!(graph.output.len(), 1);
        assert!(!graph.initializer.is_empty());
    }

    #[test]
    fn test_linear_regression_not_fitted() {
        let model = LinearRegression::new();
        let config = OnnxConfig::new("test");
        let result = model.to_onnx(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_ridge_regression_onnx_export() {
        let (x, y) = create_test_data();
        let mut model = RidgeRegression::new(1.0);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("ridge_regression_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
    }

    #[test]
    fn test_lasso_regression_onnx_export() {
        let (x, y) = create_test_data();
        let mut model = LassoRegression::new(0.1);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("lasso_regression_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
    }

    #[test]
    fn test_elastic_net_onnx_export() {
        let (x, y) = create_test_data();
        let mut model = ElasticNet::new(0.1, 0.5);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("elastic_net_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
    }

    #[test]
    fn test_logistic_regression_onnx_export() {
        let (x, y) = create_classification_data();
        let mut model = LogisticRegression::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("logistic_regression_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());

        let graph = onnx_model.graph.unwrap();
        // Should have Sigmoid for binary classification
        assert!(graph.node.iter().any(|n| n.op_type == "Sigmoid"));
    }

    #[test]
    fn test_onnx_config_applied() {
        let (x, y) = create_test_data();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("custom_model")
            .with_input_name("X_input")
            .with_output_name("y_output")
            .with_description("A custom linear regression model");

        let bytes = model.to_onnx(&config).unwrap();
        let onnx_model = ModelProto::decode(&*bytes).unwrap();

        assert_eq!(onnx_model.doc_string, "A custom linear regression model");

        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.input[0].name, "X_input");
        assert_eq!(graph.output[0].name, "y_output");
    }

    #[test]
    fn test_export_to_file() {
        let (x, y) = create_test_data();
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("file_test");
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_model.onnx");

        // Export to file
        model.export_onnx(&file_path, &config).unwrap();

        // Verify file exists and can be read
        let bytes = std::fs::read(&file_path).unwrap();
        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());

        // Cleanup
        std::fs::remove_file(&file_path).ok();
    }
}
