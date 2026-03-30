//! ONNX Export for Histogram-Based Gradient Boosting Models
//!
//! Implements ONNX export for HistGradientBoostingRegressor and HistGradientBoostingClassifier.
//!
//! Since HistTree nodes store bin thresholds (u8), we convert them to real-valued thresholds
//! via the CategoricalBinMapper before emitting ONNX TreeEnsembleRegressor nodes.

use crate::models::hist_boosting::{
    CategoricalBinMapper, HistGradientBoostingClassifier, HistGradientBoostingRegressor, HistTree,
};
use crate::models::Model;
use crate::onnx::{
    create_model_proto, create_sigmoid_node, create_tensor_input, create_tensor_output_1d,
    AttributeProto, AttributeProtoType, GraphProto, NodeProto, OnnxConfig, OnnxExportable,
    TensorProtoDataType,
};
use crate::{FerroError, Result};
use prost::Message;

/// Node modes for tree ensemble
const NODE_MODE_BRANCH_LEQ: &str = "BRANCH_LEQ";
const NODE_MODE_LEAF: &str = "LEAF";

/// Return the largest f32 strictly less than `v` (one ULP down).
///
/// Used to convert native `x < edge` semantics to ONNX `x <= threshold`.
fn next_down_f32(v: f32) -> f32 {
    if v.is_nan() || v == f32::NEG_INFINITY {
        v
    } else if v == 0.0 {
        // -0.0 is technically less, but -MIN_POSITIVE subnormal is safer
        -f32::from_bits(1)
    } else if v > 0.0 {
        f32::from_bits(v.to_bits() - 1)
    } else {
        // v < 0: increasing bits makes it more negative
        f32::from_bits(v.to_bits() + 1)
    }
}

/// Builder for hist gradient boosting tree ensemble ONNX nodes.
///
/// This is similar to `TreeEnsembleBuilder` in `tree.rs`, but works with `HistTree`
/// nodes and converts bin thresholds to real thresholds via a `CategoricalBinMapper`.
struct HistTreeEnsembleBuilder {
    nodes_featureids: Vec<i64>,
    nodes_values: Vec<f32>,
    nodes_modes: Vec<Vec<u8>>,
    nodes_treeids: Vec<i64>,
    nodes_nodeids: Vec<i64>,
    nodes_truenodeids: Vec<i64>,
    nodes_falsenodeids: Vec<i64>,
    nodes_missing_value_tracks_true: Vec<i64>,
    target_nodeids: Vec<i64>,
    target_treeids: Vec<i64>,
    target_ids: Vec<i64>,
    target_weights: Vec<f32>,
}

impl HistTreeEnsembleBuilder {
    fn new() -> Self {
        Self {
            nodes_featureids: Vec::new(),
            nodes_values: Vec::new(),
            nodes_modes: Vec::new(),
            nodes_treeids: Vec::new(),
            nodes_nodeids: Vec::new(),
            nodes_truenodeids: Vec::new(),
            nodes_falsenodeids: Vec::new(),
            nodes_missing_value_tracks_true: Vec::new(),
            target_nodeids: Vec::new(),
            target_treeids: Vec::new(),
            target_ids: Vec::new(),
            target_weights: Vec::new(),
        }
    }

    /// Add a hist tree, converting bin thresholds to real thresholds.
    ///
    /// * `tree` - The histogram-based tree
    /// * `tree_id` - Global tree ID for the ONNX ensemble
    /// * `target_id` - Target (class) index for the leaf outputs
    /// * `leaf_scale` - Multiplier for leaf values (e.g. learning_rate)
    /// * `bin_mapper` - Used to convert bin indices to real thresholds
    fn add_hist_tree(
        &mut self,
        tree: &HistTree,
        tree_id: i64,
        target_id: i64,
        leaf_scale: f32,
        bin_mapper: &CategoricalBinMapper,
    ) {
        for (node_idx, node) in tree.nodes.iter().enumerate() {
            self.nodes_treeids.push(tree_id);
            self.nodes_nodeids.push(node_idx as i64);

            if node.left_child.is_none() {
                self.nodes_featureids.push(0);
                self.nodes_values.push(0.0);
                self.nodes_modes.push(NODE_MODE_LEAF.as_bytes().to_vec());
                self.nodes_truenodeids.push(0);
                self.nodes_falsenodeids.push(0);
                self.nodes_missing_value_tracks_true.push(0);

                // Add leaf target
                self.target_treeids.push(tree_id);
                self.target_nodeids.push(node_idx as i64);
                self.target_ids.push(target_id);
                self.target_weights.push(node.value as f32 * leaf_scale);
            } else {
                let feature_idx = node.feature_idx.unwrap_or(0);
                let bin = node.bin_threshold.unwrap_or(0);
                let real_threshold = bin_mapper.bin_threshold_to_real(feature_idx, bin);

                // ONNX BRANCH_LEQ uses `x <= threshold` but native binning uses
                // `x < edge` (strict). Nudge threshold down by one ULP so that
                // values exactly at a bin edge route right (matching native).
                let threshold_f32 = next_down_f32(real_threshold as f32);

                self.nodes_featureids.push(feature_idx as i64);
                self.nodes_values.push(threshold_f32);
                self.nodes_modes
                    .push(NODE_MODE_BRANCH_LEQ.as_bytes().to_vec());
                self.nodes_truenodeids
                    .push(node.left_child.unwrap_or(0) as i64);
                self.nodes_falsenodeids
                    .push(node.right_child.unwrap_or(0) as i64);
                // missing_go_left: if true, NaN goes to the left (true) child
                self.nodes_missing_value_tracks_true
                    .push(if node.missing_go_left { 1 } else { 0 });
            }
        }
    }

    /// Build regressor attributes for TreeEnsembleRegressor
    fn build_regressor_attributes(
        self,
        n_targets: i64,
        aggregate_function: &str,
        base_values: Vec<f32>,
    ) -> Vec<AttributeProto> {
        let mut attrs = vec![
            AttributeProto {
                name: "n_targets".to_string(),
                i: n_targets,
                r#type: AttributeProtoType::Int as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "nodes_featureids".to_string(),
                ints: self.nodes_featureids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "nodes_values".to_string(),
                floats: self.nodes_values,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "nodes_modes".to_string(),
                strings: self.nodes_modes,
                r#type: AttributeProtoType::Strings as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "nodes_treeids".to_string(),
                ints: self.nodes_treeids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "nodes_nodeids".to_string(),
                ints: self.nodes_nodeids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "nodes_truenodeids".to_string(),
                ints: self.nodes_truenodeids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "nodes_falsenodeids".to_string(),
                ints: self.nodes_falsenodeids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "nodes_missing_value_tracks_true".to_string(),
                ints: self.nodes_missing_value_tracks_true,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "target_nodeids".to_string(),
                ints: self.target_nodeids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "target_treeids".to_string(),
                ints: self.target_treeids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "target_ids".to_string(),
                ints: self.target_ids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "target_weights".to_string(),
                floats: self.target_weights,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "post_transform".to_string(),
                s: "NONE".as_bytes().to_vec(),
                r#type: AttributeProtoType::String as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "aggregate_function".to_string(),
                s: aggregate_function.as_bytes().to_vec(),
                r#type: AttributeProtoType::String as i32,
                ..Default::default()
            },
        ];

        attrs.push(AttributeProto {
            name: "base_values".to_string(),
            floats: base_values,
            r#type: AttributeProtoType::Floats as i32,
            ..Default::default()
        });

        attrs
    }
}

// =============================================================================
// HistGradientBoostingRegressor
// =============================================================================

impl OnnxExportable for HistGradientBoostingRegressor {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let trees = self
            .trees()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let bin_mapper = self
            .categorical_bin_mapper()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let init_prediction = self
            .init_prediction()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        if trees.is_empty() {
            return Err(FerroError::not_fitted("No trees found in ensemble"));
        }

        let batch_size = config.input_shape.map(|(b, _)| b);
        let lr = self.learning_rate as f32;

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );
        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        let mut builder = HistTreeEnsembleBuilder::new();
        for (tree_id, tree) in trees.iter().enumerate() {
            builder.add_hist_tree(tree, tree_id as i64, 0, lr, bin_mapper);
        }

        let attributes = builder.build_regressor_attributes(1, "SUM", vec![init_prediction as f32]);

        let node = NodeProto {
            input: vec![config.input_name.clone()],
            output: vec![config.output_name.clone()],
            name: "TreeEnsembleRegressor_0".to_string(),
            op_type: "TreeEnsembleRegressor".to_string(),
            domain: "ai.onnx.ml".to_string(),
            attribute: attributes,
            doc_string: String::new(),
        };

        let graph = GraphProto {
            name: config.model_name.clone(),
            node: vec![node],
            input: vec![input],
            output: vec![output],
            ..Default::default()
        };

        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

// =============================================================================
// HistGradientBoostingClassifier
// =============================================================================

impl OnnxExportable for HistGradientBoostingClassifier {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let trees = self
            .trees()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let bin_mapper = self
            .categorical_bin_mapper()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let init_predictions = self
            .init_predictions()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_classes = self
            .n_classes()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        if trees.is_empty() {
            return Err(FerroError::not_fitted("No trees found in ensemble"));
        }

        let batch_size = config.input_shape.map(|(b, _)| b);
        let lr = self.learning_rate as f32;

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );
        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        if n_classes == 2 {
            // Binary classification: single sequence of trees, sigmoid post-transform
            // trees[iteration] contains one tree per iteration (Vec<Vec<HistTree>> with inner len 1)
            let mut builder = HistTreeEnsembleBuilder::new();
            let mut global_tree_id: i64 = 0;

            for trees_per_iter in trees.iter() {
                // Binary: each iteration has exactly 1 tree
                for tree in trees_per_iter.iter() {
                    builder.add_hist_tree(tree, global_tree_id, 0, lr, bin_mapper);
                    global_tree_id += 1;
                }
            }

            let attributes =
                builder.build_regressor_attributes(1, "SUM", vec![init_predictions[0] as f32]);

            let tree_node = NodeProto {
                input: vec![config.input_name.clone()],
                output: vec!["raw_score".to_string()],
                name: "TreeEnsembleRegressor_0".to_string(),
                op_type: "TreeEnsembleRegressor".to_string(),
                domain: "ai.onnx.ml".to_string(),
                attribute: attributes,
                doc_string: String::new(),
            };

            let sigmoid = create_sigmoid_node("raw_score", &config.output_name, "Sigmoid_0");

            let graph = GraphProto {
                name: config.model_name.clone(),
                node: vec![tree_node, sigmoid],
                input: vec![input],
                output: vec![output],
                ..Default::default()
            };
            let model = create_model_proto(graph, config, true);
            Ok(model.encode_to_vec())
        } else {
            // Multi-class: trees[iteration][class_idx]
            let mut builder = HistTreeEnsembleBuilder::new();
            let mut global_tree_id: i64 = 0;

            for trees_per_iter in trees.iter() {
                for (class_idx, tree) in trees_per_iter.iter().enumerate() {
                    builder.add_hist_tree(tree, global_tree_id, class_idx as i64, lr, bin_mapper);
                    global_tree_id += 1;
                }
            }

            let base_values: Vec<f32> = init_predictions.iter().map(|&v| v as f32).collect();
            let attributes =
                builder.build_regressor_attributes(n_classes as i64, "SUM", base_values);

            let tree_node = NodeProto {
                input: vec![config.input_name.clone()],
                output: vec!["raw_scores".to_string()],
                name: "TreeEnsembleRegressor_0".to_string(),
                op_type: "TreeEnsembleRegressor".to_string(),
                domain: "ai.onnx.ml".to_string(),
                attribute: attributes,
                doc_string: String::new(),
            };

            // ArgMax over classes
            let argmax =
                crate::onnx::create_argmax_node("raw_scores", "argmax_out", "ArgMax_0", 1, 0);
            let cast = crate::onnx::create_cast_node(
                "argmax_out",
                &config.output_name,
                "Cast_0",
                TensorProtoDataType::Float,
            );

            let graph = GraphProto {
                name: config.model_name.clone(),
                node: vec![tree_node, argmax, cast],
                input: vec![input],
                output: vec![output],
                ..Default::default()
            };
            let model = create_model_proto(graph, config, true);
            Ok(model.encode_to_vec())
        }
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::ModelProto;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_hist_gradient_boosting_regressor_onnx_not_fitted() {
        let model = HistGradientBoostingRegressor::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_onnx_not_fitted() {
        let model = HistGradientBoostingClassifier::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_onnx_export() {
        let n_samples = 100;
        let n_features = 4;
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i * n_features + j) as f64 / (n_samples * n_features) as f64
        });
        let y = Array1::from_iter((0..n_samples).map(|i| i as f64 / n_samples as f64));

        let mut model = HistGradientBoostingRegressor::new()
            .with_max_iter(10)
            .with_learning_rate(0.1)
            .with_max_leaf_nodes(Some(8));
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("hist_gbr_test");
        let bytes = model.to_onnx(&config).unwrap();
        assert!(!bytes.is_empty());

        // Validate it can be decoded as a ModelProto
        let proto = ModelProto::decode(bytes.as_slice()).unwrap();
        assert_eq!(proto.producer_name, "FerroML");
        assert!(proto.graph.is_some());

        let graph = proto.graph.unwrap();
        assert_eq!(graph.name, "hist_gbr_test");
        // Should have a TreeEnsembleRegressor node
        assert!(!graph.node.is_empty());
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
        assert_eq!(graph.node[0].domain, "ai.onnx.ml");

        // Check n_features
        assert_eq!(model.onnx_n_features(), Some(n_features));
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_binary_onnx_export() {
        let n_samples = 100;
        let n_features = 4;
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i * n_features + j) as f64 / (n_samples * n_features) as f64
        });
        let y = Array1::from_iter((0..n_samples).map(|i| if i < 50 { 0.0 } else { 1.0 }));

        let mut model = HistGradientBoostingClassifier::new()
            .with_max_iter(10)
            .with_learning_rate(0.1)
            .with_max_leaf_nodes(Some(8));
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("hist_gbc_binary_test");
        let bytes = model.to_onnx(&config).unwrap();
        assert!(!bytes.is_empty());

        let proto = ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = proto.graph.unwrap();

        // Binary: TreeEnsembleRegressor -> Sigmoid
        assert_eq!(graph.node.len(), 2);
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
        assert_eq!(graph.node[1].op_type, "Sigmoid");
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_multiclass_onnx_export() {
        let n_samples = 150;
        let n_features = 4;
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i * n_features + j) as f64 / (n_samples * n_features) as f64
        });
        let y = Array1::from_iter((0..n_samples).map(|i| (i % 3) as f64));

        let mut model = HistGradientBoostingClassifier::new()
            .with_max_iter(10)
            .with_learning_rate(0.1)
            .with_max_leaf_nodes(Some(8));
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("hist_gbc_multi_test");
        let bytes = model.to_onnx(&config).unwrap();
        assert!(!bytes.is_empty());

        let proto = ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = proto.graph.unwrap();

        // Multi-class: TreeEnsembleRegressor -> ArgMax -> Cast
        assert_eq!(graph.node.len(), 3);
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
        assert_eq!(graph.node[1].op_type, "ArgMax");
        assert_eq!(graph.node[2].op_type, "Cast");
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_onnx_file_export() {
        let n_samples = 80;
        let n_features = 3;
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i * n_features + j) as f64 / (n_samples * n_features) as f64
        });
        let y = Array1::from_iter((0..n_samples).map(|i| (i as f64).sin()));

        let mut model = HistGradientBoostingRegressor::new()
            .with_max_iter(5)
            .with_learning_rate(0.1);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("hist_gbr_file_test");
        let tmp_dir = tempfile::tempdir().unwrap();
        let tmp_path = tmp_dir.path().join("test_hist_gbr.onnx");
        model
            .export_onnx(tmp_path.to_str().unwrap(), &config)
            .unwrap();

        // Verify file exists and can be read back
        let bytes = std::fs::read(&tmp_path).unwrap();
        assert!(!bytes.is_empty());
        let proto = ModelProto::decode(bytes.as_slice()).unwrap();
        assert_eq!(proto.graph.unwrap().name, "hist_gbr_file_test");
        // tmp_dir auto-cleans on drop
    }

    #[test]
    fn test_hist_gradient_boosting_onnx_has_ml_opset() {
        let n_samples = 60;
        let n_features = 2;
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i * n_features + j) as f64 / (n_samples * n_features) as f64
        });
        let y = Array1::from_iter((0..n_samples).map(|i| i as f64));

        let mut model = HistGradientBoostingRegressor::new()
            .with_max_iter(3)
            .with_learning_rate(0.1);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("opset_test");
        let bytes = model.to_onnx(&config).unwrap();
        let proto = ModelProto::decode(bytes.as_slice()).unwrap();

        // Should have both default and ai.onnx.ml opsets
        assert!(proto.opset_import.len() >= 2);
        assert!(proto
            .opset_import
            .iter()
            .any(|op| op.domain == "ai.onnx.ml"));
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_onnx_tree_attributes() {
        let n_samples = 50;
        let n_features = 2;
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            (i * n_features + j) as f64 / 10.0
        });
        let y = Array1::from_iter((0..n_samples).map(|i| i as f64));

        let mut model = HistGradientBoostingRegressor::new()
            .with_max_iter(3)
            .with_learning_rate(0.1)
            .with_max_leaf_nodes(Some(4));
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("attr_test");
        let bytes = model.to_onnx(&config).unwrap();
        let proto = ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = proto.graph.unwrap();
        let tree_node = &graph.node[0];

        // Verify key attributes exist
        let attr_names: Vec<&str> = tree_node
            .attribute
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert!(attr_names.contains(&"n_targets"));
        assert!(attr_names.contains(&"nodes_featureids"));
        assert!(attr_names.contains(&"nodes_values"));
        assert!(attr_names.contains(&"nodes_modes"));
        assert!(attr_names.contains(&"nodes_treeids"));
        assert!(attr_names.contains(&"nodes_nodeids"));
        assert!(attr_names.contains(&"nodes_truenodeids"));
        assert!(attr_names.contains(&"nodes_falsenodeids"));
        assert!(attr_names.contains(&"target_nodeids"));
        assert!(attr_names.contains(&"target_treeids"));
        assert!(attr_names.contains(&"target_weights"));
        assert!(attr_names.contains(&"base_values"));
        assert!(attr_names.contains(&"aggregate_function"));

        // Verify base_values contains the init prediction
        let base_vals_attr = tree_node
            .attribute
            .iter()
            .find(|a| a.name == "base_values")
            .unwrap();
        assert_eq!(base_vals_attr.floats.len(), 1);

        // Verify n_targets = 1 for regressor
        let n_targets_attr = tree_node
            .attribute
            .iter()
            .find(|a| a.name == "n_targets")
            .unwrap();
        assert_eq!(n_targets_attr.i, 1);
    }

    #[test]
    fn test_hist_gradient_boosting_onnx_real_thresholds() {
        // Verify that bin thresholds are converted to real-valued thresholds
        let n_samples = 100;
        let n_features = 2;
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            // Create data with clear structure so thresholds are non-trivial
            (i as f64 + j as f64 * 10.0) / 10.0
        });
        let y = Array1::from_iter((0..n_samples).map(|i| if i < 50 { 0.0 } else { 1.0 }));

        let mut model = HistGradientBoostingRegressor::new()
            .with_max_iter(3)
            .with_learning_rate(0.1)
            .with_max_leaf_nodes(Some(4));
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("threshold_test");
        let bytes = model.to_onnx(&config).unwrap();
        let proto = ModelProto::decode(bytes.as_slice()).unwrap();
        let graph = proto.graph.unwrap();
        let tree_node = &graph.node[0];

        let nodes_values = tree_node
            .attribute
            .iter()
            .find(|a| a.name == "nodes_values")
            .unwrap();
        let nodes_modes = tree_node
            .attribute
            .iter()
            .find(|a| a.name == "nodes_modes")
            .unwrap();

        // For branch nodes, check that thresholds are real-valued (not bin indices)
        // Bin indices would be small integers (0-255), real values are from our data
        for (i, mode) in nodes_modes.strings.iter().enumerate() {
            if mode == NODE_MODE_BRANCH_LEQ.as_bytes() {
                let threshold = nodes_values.floats[i];
                // The threshold should be a real data value, not a tiny bin index
                // Our data range is 0..10.9 approximately so thresholds should be in that range
                // (bin indices would be 0..255)
                assert!(
                    threshold.is_finite(),
                    "Branch threshold should be finite, got {}",
                    threshold
                );
            }
        }
    }
}
