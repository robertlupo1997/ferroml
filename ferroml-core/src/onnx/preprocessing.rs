//! ONNX Export for Preprocessing Transformers
//!
//! Implements ONNX export for StandardScaler, MinMaxScaler, RobustScaler, and MaxAbsScaler.

use crate::onnx::{
    create_add_node, create_div_node, create_float_tensor, create_model_proto, create_mul_node,
    create_sub_node, create_tensor_input, create_tensor_output, GraphProto, OnnxConfig,
    OnnxExportable, TensorProtoDataType,
};
use crate::preprocessing::scalers::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
use crate::{FerroError, Result};
use prost::Message;

impl OnnxExportable for StandardScaler {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        let mean = self
            .mean()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let std = self
            .std()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_features = mean.len();
        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output = create_tensor_output(
            &config.output_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let mean_data: Vec<f32> = mean.iter().map(|&v| v as f32).collect();
        let std_data: Vec<f32> = std
            .iter()
            .map(|&v| if v.abs() < 1e-10 { 1.0f32 } else { v as f32 })
            .collect();

        let mean_tensor = create_float_tensor("mean", &[1, n_features as i64], mean_data);
        let std_tensor = create_float_tensor("std", &[1, n_features as i64], std_data);

        let sub = create_sub_node(&config.input_name, "mean", "centered", "Sub_0");
        let div = create_div_node("centered", "std", &config.output_name, "Div_0");

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![sub, div],
            input: vec![input],
            output: vec![output],
            initializer: vec![mean_tensor, std_tensor],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.mean().map(|m| m.len())
    }
}

impl OnnxExportable for MinMaxScaler {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        let data_min = self
            .data_min()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let data_range = self
            .data_range()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let (feat_min, feat_max) = self.feature_range();
        let feat_scale = feat_max - feat_min;

        let n_features = data_min.len();
        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output = create_tensor_output(
            &config.output_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let min_data: Vec<f32> = data_min.iter().map(|&v| v as f32).collect();
        let range_data: Vec<f32> = data_range
            .iter()
            .map(|&v| if v.abs() < 1e-10 { 1.0f32 } else { v as f32 })
            .collect();
        let scale_data: Vec<f32> = vec![feat_scale as f32; n_features];
        let offset_data: Vec<f32> = vec![feat_min as f32; n_features];

        let min_tensor = create_float_tensor("data_min", &[1, n_features as i64], min_data);
        let range_tensor = create_float_tensor("data_range", &[1, n_features as i64], range_data);
        let scale_tensor = create_float_tensor("feat_scale", &[1, n_features as i64], scale_data);
        let offset_tensor =
            create_float_tensor("feat_offset", &[1, n_features as i64], offset_data);

        // (X - data_min) / data_range * feat_scale + feat_offset
        let sub = create_sub_node(&config.input_name, "data_min", "shifted", "Sub_0");
        let div = create_div_node("shifted", "data_range", "normalized", "Div_0");
        let mul = create_mul_node("normalized", "feat_scale", "scaled", "Mul_0");
        let add = create_add_node("scaled", "feat_offset", &config.output_name, "Add_0");

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![sub, div, mul, add],
            input: vec![input],
            output: vec![output],
            initializer: vec![min_tensor, range_tensor, scale_tensor, offset_tensor],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.data_min().map(|m| m.len())
    }
}

impl OnnxExportable for RobustScaler {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        let center = self
            .center()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let scale = self
            .scale()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_features = center.len();
        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output = create_tensor_output(
            &config.output_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let center_data: Vec<f32> = center.iter().map(|&v| v as f32).collect();
        let scale_data: Vec<f32> = scale
            .iter()
            .map(|&v| if v.abs() < 1e-10 { 1.0f32 } else { v as f32 })
            .collect();

        let center_tensor = create_float_tensor("center", &[1, n_features as i64], center_data);
        let scale_tensor = create_float_tensor("scale", &[1, n_features as i64], scale_data);

        let sub = create_sub_node(&config.input_name, "center", "centered", "Sub_0");
        let div = create_div_node("centered", "scale", &config.output_name, "Div_0");

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![sub, div],
            input: vec![input],
            output: vec![output],
            initializer: vec![center_tensor, scale_tensor],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.center().map(|c| c.len())
    }
}

impl OnnxExportable for MaxAbsScaler {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        let max_abs = self
            .max_abs()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_features = max_abs.len();
        let batch_size = config.input_shape.map(|(b, _)| b);

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output = create_tensor_output(
            &config.output_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let max_abs_data: Vec<f32> = max_abs
            .iter()
            .map(|&v| if v.abs() < 1e-10 { 1.0f32 } else { v as f32 })
            .collect();

        let max_abs_tensor = create_float_tensor("max_abs", &[1, n_features as i64], max_abs_data);

        let div = create_div_node(&config.input_name, "max_abs", &config.output_name, "Div_0");

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![div],
            input: vec![input],
            output: vec![output],
            initializer: vec![max_abs_tensor],
            ..Default::default()
        };
        let model = create_model_proto(graph, config, false);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.max_abs().map(|m| m.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::ModelProto;
    use crate::preprocessing::Transformer;
    use ndarray::Array2;

    fn create_test_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0, 4.0, 40.0, 400.0, 5.0, 50.0,
                500.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_standard_scaler_onnx_export() {
        let x = create_test_data();
        let mut scaler = StandardScaler::new();
        scaler.fit(&x).unwrap();

        let config = OnnxConfig::new("standard_scaler");
        let bytes = scaler.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "Sub"));
        assert!(graph.node.iter().any(|n| n.op_type == "Div"));
    }

    #[test]
    fn test_min_max_scaler_onnx_export() {
        let x = create_test_data();
        let mut scaler = MinMaxScaler::new();
        scaler.fit(&x).unwrap();

        let config = OnnxConfig::new("min_max_scaler");
        let bytes = scaler.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "Sub"));
        assert!(graph.node.iter().any(|n| n.op_type == "Div"));
        assert!(graph.node.iter().any(|n| n.op_type == "Mul"));
        assert!(graph.node.iter().any(|n| n.op_type == "Add"));
    }

    #[test]
    fn test_robust_scaler_onnx_export() {
        let x = create_test_data();
        let mut scaler = RobustScaler::new();
        scaler.fit(&x).unwrap();

        let config = OnnxConfig::new("robust_scaler");
        let bytes = scaler.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "Sub"));
        assert!(graph.node.iter().any(|n| n.op_type == "Div"));
    }

    #[test]
    fn test_max_abs_scaler_onnx_export() {
        let x = create_test_data();
        let mut scaler = MaxAbsScaler::new();
        scaler.fit(&x).unwrap();

        let config = OnnxConfig::new("max_abs_scaler");
        let bytes = scaler.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph.node.iter().any(|n| n.op_type == "Div"));
    }

    #[test]
    fn test_standard_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let config = OnnxConfig::new("test");
        assert!(scaler.to_onnx(&config).is_err());
    }

    #[test]
    fn test_min_max_scaler_not_fitted() {
        let scaler = MinMaxScaler::new();
        let config = OnnxConfig::new("test");
        assert!(scaler.to_onnx(&config).is_err());
    }

    #[test]
    fn test_robust_scaler_not_fitted() {
        let scaler = RobustScaler::new();
        let config = OnnxConfig::new("test");
        assert!(scaler.to_onnx(&config).is_err());
    }

    #[test]
    fn test_max_abs_scaler_not_fitted() {
        let scaler = MaxAbsScaler::new();
        let config = OnnxConfig::new("test");
        assert!(scaler.to_onnx(&config).is_err());
    }
}
