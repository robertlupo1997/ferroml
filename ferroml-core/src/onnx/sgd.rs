//! ONNX Export for SGDClassifier
//!
//! Handles both binary and multi-class cases.

use crate::models::sgd::SGDClassifier;
use crate::models::Model;
use crate::onnx::{
    create_argmax_node, create_cast_node, create_float_tensor, create_gemm_node,
    create_int64_tensor, create_model_proto, create_sigmoid_node, create_squeeze_node,
    create_tensor_input, create_tensor_output_1d, GraphProto, OnnxConfig, OnnxExportable,
    TensorProtoDataType,
};
use crate::{FerroError, Result};
use prost::Message;

impl OnnxExportable for SGDClassifier {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let coef = self
            .coef()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let intercept = self
            .intercept()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let classes = self
            .classes()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_classes = classes.len();
        let n_outputs = coef.nrows(); // 1 for binary, n_classes for multi

        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        // Weights: [n_features, n_outputs] for Gemm (transB=0)
        let mut weights_flat: Vec<f32> = Vec::with_capacity(n_features * n_outputs);
        for feat in 0..n_features {
            for out in 0..n_outputs {
                weights_flat.push(coef[[out, feat]] as f32);
            }
        }
        let weights_tensor = create_float_tensor(
            "weights",
            &[n_features as i64, n_outputs as i64],
            weights_flat,
        );

        let bias: Vec<f32> = intercept.iter().map(|&v| v as f32).collect();
        let bias_tensor = create_float_tensor("bias", &[n_outputs as i64], bias);

        let gemm_node = create_gemm_node(
            &config.input_name,
            "weights",
            "bias",
            "gemm_out",
            1.0,
            1.0,
            0,
            0,
        );

        let mut nodes = vec![gemm_node];
        let mut initializers = vec![weights_tensor, bias_tensor];

        if n_classes == 2 {
            // Binary: Sigmoid output for probability
            let squeeze_axes = create_int64_tensor("squeeze_axes", &[1], vec![1]);
            initializers.push(squeeze_axes);
            let squeeze = create_squeeze_node("gemm_out", "squeeze_axes", "squeezed", "Squeeze_0");
            nodes.push(squeeze);

            let sigmoid = create_sigmoid_node("squeezed", &config.output_name, "Sigmoid_0");
            nodes.push(sigmoid);

            let output = create_tensor_output_1d(
                &config.output_name,
                batch_size,
                TensorProtoDataType::Float,
            );

            let graph = GraphProto {
                name: config.model_name.clone(),
                node: nodes,
                input: vec![input],
                output: vec![output],
                initializer: initializers,
                ..Default::default()
            };
            let model = create_model_proto(graph, config, false);
            Ok(model.encode_to_vec())
        } else {
            // Multi-class: ArgMax for class labels
            let argmax = create_argmax_node("gemm_out", "argmax_out", "ArgMax_0", 1, 0);
            nodes.push(argmax);

            let cast = create_cast_node(
                "argmax_out",
                &config.output_name,
                "Cast_0",
                TensorProtoDataType::Float,
            );
            nodes.push(cast);

            let output = create_tensor_output_1d(
                &config.output_name,
                batch_size,
                TensorProtoDataType::Float,
            );

            let graph = GraphProto {
                name: config.model_name.clone(),
                node: nodes,
                input: vec![input],
                output: vec![output],
                initializer: initializers,
                ..Default::default()
            };
            let model = create_model_proto(graph, config, false);
            Ok(model.encode_to_vec())
        }
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;
    use crate::onnx::ModelProto;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_sgd_classifier_binary_onnx_export() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut model = SGDClassifier::new()
            .with_max_iter(200)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("sgd_classifier_binary");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "Sigmoid"));
    }

    #[test]
    fn test_sgd_classifier_multiclass_onnx_export() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.5, 2.0, 5.0, 5.0, 6.0, 5.0, 5.5, 6.0, 9.0, 1.0, 10.0, 1.0,
                9.5, 2.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let mut model = SGDClassifier::new()
            .with_max_iter(500)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("sgd_classifier_multiclass");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "ArgMax"));
    }

    #[test]
    fn test_sgd_classifier_not_fitted() {
        let model = SGDClassifier::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }
}
