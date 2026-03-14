//! ONNX Export for Tree Models
//!
//! Implements ONNX export for DecisionTreeClassifier, DecisionTreeRegressor,
//! RandomForestClassifier, RandomForestRegressor.
//!
//! Uses TreeEnsembleClassifier and TreeEnsembleRegressor operators from ONNX-ML domain.

use crate::models::adaboost::{AdaBoostClassifier, AdaBoostRegressor};
use crate::models::boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
use crate::models::extra_trees::{ExtraTreesClassifier, ExtraTreesRegressor};
use crate::models::forest::{RandomForestClassifier, RandomForestRegressor};
use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor, TreeStructure};
use crate::models::Model;
use crate::onnx::{
    create_model_proto, create_sigmoid_node, create_tensor_input, create_tensor_output_1d,
    AttributeProto, AttributeProtoType, GraphProto, NodeProto, OnnxConfig, OnnxExportable,
    TensorProtoDataType,
};
use crate::{FerroError, Result};
use prost::Message;

/// Maximum number of nodes allowed in a tree ensemble ONNX export.
/// Exceeding this limit would produce extremely large ONNX files (100MB+).
const MAX_ONNX_TREE_NODES: usize = 10_000_000;

/// Node modes for tree ensemble
const NODE_MODE_BRANCH_LEQ: &str = "BRANCH_LEQ";
const NODE_MODE_LEAF: &str = "LEAF";

/// Check that the total number of nodes across all trees does not exceed the limit.
fn check_tree_node_limit(trees: &[&TreeStructure]) -> Result<()> {
    let total_nodes: usize = trees.iter().map(|t| t.nodes.len()).sum();
    if total_nodes > MAX_ONNX_TREE_NODES {
        return Err(FerroError::invalid_input(format!(
            "ONNX export: tree ensemble has {} nodes, exceeding 10M limit. Reduce n_estimators or max_depth.",
            total_nodes
        )));
    }
    Ok(())
}

/// Helper struct to build tree ensemble ONNX representation
struct TreeEnsembleBuilder {
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
    n_targets: usize,
}

impl TreeEnsembleBuilder {
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
            n_targets: 0,
        }
    }

    /// Add a tree to the ensemble (raw leaf values for regressors)
    fn add_tree(&mut self, tree: &TreeStructure, tree_id: i64) {
        self.add_tree_inner(tree, tree_id, false);
    }

    /// Add a tree with leaf values normalized to probabilities (for classifiers)
    fn add_tree_normalized(&mut self, tree: &TreeStructure, tree_id: i64) {
        self.add_tree_inner(tree, tree_id, true);
    }

    fn add_tree_inner(&mut self, tree: &TreeStructure, tree_id: i64, normalize_leaves: bool) {
        for node in &tree.nodes {
            self.nodes_treeids.push(tree_id);
            self.nodes_nodeids.push(node.id as i64);

            if node.is_leaf {
                self.nodes_featureids.push(0);
                self.nodes_values.push(0.0);
                self.nodes_modes.push(NODE_MODE_LEAF.as_bytes().to_vec());
                self.nodes_truenodeids.push(0);
                self.nodes_falsenodeids.push(0);
                self.nodes_missing_value_tracks_true.push(0);

                // Compute normalization factor for probability conversion
                let total: f64 = if normalize_leaves {
                    let s: f64 = node.value.iter().sum();
                    if s > 0.0 {
                        s
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                // Add target values (leaf node outputs)
                for (target_id, &value) in node.value.iter().enumerate() {
                    self.target_treeids.push(tree_id);
                    self.target_nodeids.push(node.id as i64);
                    self.target_ids.push(target_id as i64);
                    self.target_weights.push((value / total) as f32);
                }
                self.n_targets = self.n_targets.max(node.value.len());
            } else {
                self.nodes_featureids
                    .push(node.feature_index.unwrap_or(0) as i64);
                self.nodes_values.push(node.threshold.unwrap_or(0.0) as f32);
                self.nodes_modes
                    .push(NODE_MODE_BRANCH_LEQ.as_bytes().to_vec());
                self.nodes_truenodeids
                    .push(node.left_child.unwrap_or(0) as i64);
                self.nodes_falsenodeids
                    .push(node.right_child.unwrap_or(0) as i64);
                self.nodes_missing_value_tracks_true.push(1); // NaN goes left
            }
        }
    }

    /// Build regressor attributes
    fn build_regressor_attributes(
        self,
        n_targets: i64,
        aggregate_function: &str,
    ) -> Vec<AttributeProto> {
        vec![
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
        ]
    }

    /// Build classifier attributes (kept for potential future use)
    #[allow(dead_code)]
    fn build_classifier_attributes(
        self,
        class_labels: Vec<i64>,
        post_transform: &str,
    ) -> Vec<AttributeProto> {
        vec![
            AttributeProto {
                name: "classlabels_int64s".to_string(),
                ints: class_labels,
                r#type: AttributeProtoType::Ints as i32,
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
                name: "class_nodeids".to_string(),
                ints: self.target_nodeids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "class_treeids".to_string(),
                ints: self.target_treeids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "class_ids".to_string(),
                ints: self.target_ids,
                r#type: AttributeProtoType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "class_weights".to_string(),
                floats: self.target_weights,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "post_transform".to_string(),
                s: post_transform.as_bytes().to_vec(),
                r#type: AttributeProtoType::String as i32,
                ..Default::default()
            },
        ]
    }
}

/// Create a tree ensemble regressor graph
fn create_tree_regressor_graph(
    trees: &[&TreeStructure],
    n_features: usize,
    config: &OnnxConfig,
    aggregate_function: &str,
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

    // Build tree ensemble attributes
    let mut builder = TreeEnsembleBuilder::new();
    for (tree_id, tree) in trees.iter().enumerate() {
        builder.add_tree(tree, tree_id as i64);
    }

    let attributes = builder.build_regressor_attributes(1, aggregate_function);

    // TreeEnsembleRegressor node
    let node = NodeProto {
        input: vec![config.input_name.clone()],
        output: vec![config.output_name.clone()],
        name: "TreeEnsembleRegressor_0".to_string(),
        op_type: "TreeEnsembleRegressor".to_string(),
        domain: "ai.onnx.ml".to_string(),
        attribute: attributes,
        doc_string: String::new(),
    };

    GraphProto {
        name: config.model_name.clone(),
        node: vec![node],
        input: vec![input],
        output: vec![output],
        ..Default::default()
    }
}

/// Create a tree ensemble classifier graph using TreeEnsembleRegressor + ArgMax.
///
/// Using TreeEnsembleRegressor (not TreeEnsembleClassifier) avoids label-vs-proba
/// inconsistencies in ORT. Each tree's leaf values are normalized to probabilities,
/// the regressor sums them across trees (one target per class), and ArgMax + Cast
/// selects the predicted class label.
fn create_tree_classifier_graph(
    trees: &[&TreeStructure],
    n_features: usize,
    n_classes: usize,
    config: &OnnxConfig,
) -> GraphProto {
    let batch_size = config.input_shape.map(|(b, _)| b);

    let input = create_tensor_input(
        &config.input_name,
        n_features,
        batch_size,
        TensorProtoDataType::Float,
    );

    let output =
        create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

    // Build tree ensemble with normalized leaf probabilities as multi-target regressor.
    // Each class becomes a target: target_id = class_index.
    let mut builder = TreeEnsembleBuilder::new();
    for (tree_id, tree) in trees.iter().enumerate() {
        builder.add_tree_normalized(tree, tree_id as i64);
    }
    builder.n_targets = n_classes;

    let attributes = builder.build_regressor_attributes(n_classes as i64, "SUM");

    let tree_node = NodeProto {
        input: vec![config.input_name.clone()],
        output: vec!["raw_scores".to_string()],
        name: "TreeEnsembleRegressor_0".to_string(),
        op_type: "TreeEnsembleRegressor".to_string(),
        domain: "ai.onnx.ml".to_string(),
        attribute: attributes,
        doc_string: String::new(),
    };

    // ArgMax over classes → predicted class index
    let argmax = crate::onnx::create_argmax_node("raw_scores", "argmax_out", "ArgMax_0", 1, 0);
    let cast = crate::onnx::create_cast_node(
        "argmax_out",
        &config.output_name,
        "Cast_0",
        TensorProtoDataType::Float,
    );

    GraphProto {
        name: config.model_name.clone(),
        node: vec![tree_node, argmax, cast],
        input: vec![input],
        output: vec![output],
        ..Default::default()
    }
}

impl OnnxExportable for DecisionTreeRegressor {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let tree = self
            .tree()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        check_tree_node_limit(&[tree])?;
        let graph = create_tree_regressor_graph(&[tree], n_features, config, "SUM");
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for DecisionTreeClassifier {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let tree = self
            .tree()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_classes = tree.n_classes;

        check_tree_node_limit(&[tree])?;
        let graph = create_tree_classifier_graph(&[tree], n_features, n_classes, config);
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for RandomForestRegressor {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let estimators = self
            .estimators()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        // Collect tree structures from each estimator
        let tree_refs: Vec<&TreeStructure> =
            estimators.iter().filter_map(|est| est.tree()).collect();

        if tree_refs.is_empty() {
            return Err(FerroError::not_fitted("No trees found in forest"));
        }

        check_tree_node_limit(&tree_refs)?;
        let graph = create_tree_regressor_graph(&tree_refs, n_features, config, "AVERAGE");
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for RandomForestClassifier {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let estimators = self
            .estimators()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        // Get number of classes from the class labels
        let classes = self
            .classes()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_classes = classes.len();

        // Collect tree structures from each estimator
        let tree_refs: Vec<&TreeStructure> =
            estimators.iter().filter_map(|est| est.tree()).collect();

        if tree_refs.is_empty() {
            return Err(FerroError::not_fitted("No trees found in forest"));
        }

        check_tree_node_limit(&tree_refs)?;
        let graph = create_tree_classifier_graph(&tree_refs, n_features, n_classes, config);
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

// =============================================================================
// ExtraTrees (same as RandomForest)
// =============================================================================

impl OnnxExportable for ExtraTreesRegressor {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let estimators = self
            .estimators()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let tree_refs: Vec<&TreeStructure> =
            estimators.iter().filter_map(|est| est.tree()).collect();

        if tree_refs.is_empty() {
            return Err(FerroError::not_fitted("No trees found in ensemble"));
        }

        check_tree_node_limit(&tree_refs)?;
        let graph = create_tree_regressor_graph(&tree_refs, n_features, config, "AVERAGE");
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for ExtraTreesClassifier {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let estimators = self
            .estimators()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let classes = self
            .classes()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_classes = classes.len();

        let tree_refs: Vec<&TreeStructure> =
            estimators.iter().filter_map(|est| est.tree()).collect();

        if tree_refs.is_empty() {
            return Err(FerroError::not_fitted("No trees found in ensemble"));
        }

        check_tree_node_limit(&tree_refs)?;
        let graph = create_tree_classifier_graph(&tree_refs, n_features, n_classes, config);
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

// =============================================================================
// GradientBoosting
// =============================================================================

/// Create a tree regressor graph with scaled leaf values and base value offset.
fn create_tree_regressor_graph_with_scaling(
    trees: &[&TreeStructure],
    leaf_scales: &[f32],
    base_value: f32,
    n_features: usize,
    config: &OnnxConfig,
) -> GraphProto {
    let batch_size = config.input_shape.map(|(b, _)| b);

    let input = create_tensor_input(
        &config.input_name,
        n_features,
        batch_size,
        TensorProtoDataType::Float,
    );

    let output =
        create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

    let mut builder = TreeEnsembleBuilder::new();
    for (tree_id, tree) in trees.iter().enumerate() {
        builder.add_tree(tree, tree_id as i64);
    }

    // Scale leaf values
    for (i, weight) in builder.target_weights.iter_mut().enumerate() {
        // Determine which tree this target belongs to
        let tree_id = builder.target_treeids[i] as usize;
        if tree_id < leaf_scales.len() {
            *weight *= leaf_scales[tree_id];
        }
    }

    let mut attributes = builder.build_regressor_attributes(1, "SUM");

    // Add base_values attribute
    attributes.push(AttributeProto {
        name: "base_values".to_string(),
        floats: vec![base_value],
        r#type: AttributeProtoType::Floats as i32,
        ..Default::default()
    });

    let node = NodeProto {
        input: vec![config.input_name.clone()],
        output: vec![config.output_name.clone()],
        name: "TreeEnsembleRegressor_0".to_string(),
        op_type: "TreeEnsembleRegressor".to_string(),
        domain: "ai.onnx.ml".to_string(),
        attribute: attributes,
        doc_string: String::new(),
    };

    GraphProto {
        name: config.model_name.clone(),
        node: vec![node],
        input: vec![input],
        output: vec![output],
        ..Default::default()
    }
}

impl OnnxExportable for GradientBoostingRegressor {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let estimators = self
            .estimators()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let init_prediction = self
            .init_prediction()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let tree_refs: Vec<&TreeStructure> =
            estimators.iter().filter_map(|est| est.tree()).collect();

        if tree_refs.is_empty() {
            return Err(FerroError::not_fitted("No trees found in ensemble"));
        }

        check_tree_node_limit(&tree_refs)?;

        // Compute learning rate per tree
        let n_trees = tree_refs.len();
        let leaf_scales: Vec<f32> = (0..n_trees)
            .map(|i| self.learning_rate_at(i) as f32)
            .collect();

        let graph = create_tree_regressor_graph_with_scaling(
            &tree_refs,
            &leaf_scales,
            init_prediction as f32,
            n_features,
            config,
        );
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for GradientBoostingClassifier {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let estimators = self
            .estimators()
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

        let batch_size = config.input_shape.map(|(b, _)| b);

        if n_classes == 2 {
            // Binary case: single sequence of trees, sigmoid post-transform
            // estimators is Vec<Vec<DTR>> where inner vec has 1 element per iteration
            let tree_refs: Vec<&TreeStructure> = estimators
                .iter()
                .flat_map(|trees_per_iter| trees_per_iter.iter().filter_map(|t| t.tree()))
                .collect();

            if tree_refs.is_empty() {
                return Err(FerroError::not_fitted("No trees found"));
            }

            check_tree_node_limit(&tree_refs)?;

            let n_trees = tree_refs.len();
            let leaf_scales: Vec<f32> = (0..n_trees)
                .map(|i| self.learning_rate_at(i) as f32)
                .collect();

            let input = create_tensor_input(
                &config.input_name,
                n_features,
                batch_size,
                TensorProtoDataType::Float,
            );
            let output = create_tensor_output_1d(
                &config.output_name,
                batch_size,
                TensorProtoDataType::Float,
            );

            let mut builder = TreeEnsembleBuilder::new();
            for (tree_id, tree) in tree_refs.iter().enumerate() {
                builder.add_tree(tree, tree_id as i64);
            }

            // Scale leaf values
            for (i, weight) in builder.target_weights.iter_mut().enumerate() {
                let tree_id = builder.target_treeids[i] as usize;
                if tree_id < leaf_scales.len() {
                    *weight *= leaf_scales[tree_id];
                }
            }

            let mut attributes = builder.build_regressor_attributes(1, "SUM");
            attributes.push(AttributeProto {
                name: "base_values".to_string(),
                floats: vec![init_predictions[0] as f32],
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            });

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
            // Multi-class: n_classes tree sequences, use TreeEnsembleRegressor with n_targets
            // estimators[iteration][class_idx] = DecisionTreeRegressor
            let input = create_tensor_input(
                &config.input_name,
                n_features,
                batch_size,
                TensorProtoDataType::Float,
            );
            let output = create_tensor_output_1d(
                &config.output_name,
                batch_size,
                TensorProtoDataType::Float,
            );

            // Check total node count across all trees
            let all_tree_refs: Vec<&TreeStructure> = estimators
                .iter()
                .flat_map(|trees_per_iter| trees_per_iter.iter().filter_map(|t| t.tree()))
                .collect();
            check_tree_node_limit(&all_tree_refs)?;

            let mut builder = TreeEnsembleBuilder::new();
            let mut global_tree_id: i64 = 0;

            for (iter_idx, trees_per_iter) in estimators.iter().enumerate() {
                let lr = self.learning_rate_at(iter_idx) as f32;
                for (class_idx, tree) in trees_per_iter.iter().enumerate() {
                    if let Some(tree_struct) = tree.tree() {
                        // Add tree nodes
                        for node in &tree_struct.nodes {
                            builder.nodes_treeids.push(global_tree_id);
                            builder.nodes_nodeids.push(node.id as i64);

                            if node.is_leaf {
                                builder.nodes_featureids.push(0);
                                builder.nodes_values.push(0.0);
                                builder.nodes_modes.push(NODE_MODE_LEAF.as_bytes().to_vec());
                                builder.nodes_truenodeids.push(0);
                                builder.nodes_falsenodeids.push(0);
                                builder.nodes_missing_value_tracks_true.push(0);

                                // Target: class_idx is the target_id
                                builder.target_treeids.push(global_tree_id);
                                builder.target_nodeids.push(node.id as i64);
                                builder.target_ids.push(class_idx as i64);
                                // Scale by learning rate
                                let val = if !node.value.is_empty() {
                                    node.value[0] as f32 * lr
                                } else {
                                    0.0
                                };
                                builder.target_weights.push(val);
                            } else {
                                builder
                                    .nodes_featureids
                                    .push(node.feature_index.unwrap_or(0) as i64);
                                builder
                                    .nodes_values
                                    .push(node.threshold.unwrap_or(0.0) as f32);
                                builder
                                    .nodes_modes
                                    .push(NODE_MODE_BRANCH_LEQ.as_bytes().to_vec());
                                builder
                                    .nodes_truenodeids
                                    .push(node.left_child.unwrap_or(0) as i64);
                                builder
                                    .nodes_falsenodeids
                                    .push(node.right_child.unwrap_or(0) as i64);
                                builder.nodes_missing_value_tracks_true.push(1);
                            }
                        }
                        global_tree_id += 1;
                    }
                }
            }

            builder.n_targets = n_classes;

            let mut attributes = builder.build_regressor_attributes(n_classes as i64, "SUM");

            // Base values per class
            let base_values: Vec<f32> = init_predictions.iter().map(|&v| v as f32).collect();
            attributes.push(AttributeProto {
                name: "base_values".to_string(),
                floats: base_values,
                r#type: AttributeProtoType::Floats as i32,
                ..Default::default()
            });

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
// AdaBoost
// =============================================================================

impl OnnxExportable for AdaBoostClassifier {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let estimators = self
            .estimators()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let estimator_weights = self
            .estimator_weights()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let classes = self
            .classes()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let n_classes = classes.len();

        let tree_refs: Vec<&TreeStructure> =
            estimators.iter().filter_map(|est| est.tree()).collect();

        if tree_refs.is_empty() {
            return Err(FerroError::not_fitted("No trees found"));
        }

        check_tree_node_limit(&tree_refs)?;

        let batch_size = config.input_shape.map(|(b, _)| b);

        // Build ensemble where each tree votes for its predicted class.
        // Native AdaBoost: each tree predicts one class (argmax of leaf values),
        // that class gets +weight, all others get 0. We replicate this by
        // converting leaf values to one-hot (based on argmax) * weight.
        let mut builder = TreeEnsembleBuilder::new();
        for (tree_id, tree) in tree_refs.iter().enumerate() {
            let weight = if tree_id < estimator_weights.len() {
                estimator_weights[tree_id] as f32
            } else {
                1.0
            };

            for node in &tree.nodes {
                builder.nodes_treeids.push(tree_id as i64);
                builder.nodes_nodeids.push(node.id as i64);

                if node.is_leaf {
                    builder.nodes_featureids.push(0);
                    builder.nodes_values.push(0.0);
                    builder.nodes_modes.push(NODE_MODE_LEAF.as_bytes().to_vec());
                    builder.nodes_truenodeids.push(0);
                    builder.nodes_falsenodeids.push(0);
                    builder.nodes_missing_value_tracks_true.push(0);

                    // Find the argmax class for this leaf (matches native predict)
                    let argmax_idx = node
                        .value
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    // One-hot: only the winning class gets the weight
                    for (target_id, _) in node.value.iter().enumerate() {
                        builder.target_treeids.push(tree_id as i64);
                        builder.target_nodeids.push(node.id as i64);
                        builder.target_ids.push(target_id as i64);
                        let w = if target_id == argmax_idx { weight } else { 0.0 };
                        builder.target_weights.push(w);
                    }
                    builder.n_targets = builder.n_targets.max(node.value.len());
                } else {
                    builder
                        .nodes_featureids
                        .push(node.feature_index.unwrap_or(0) as i64);
                    builder
                        .nodes_values
                        .push(node.threshold.unwrap_or(0.0) as f32);
                    builder
                        .nodes_modes
                        .push(NODE_MODE_BRANCH_LEQ.as_bytes().to_vec());
                    builder
                        .nodes_truenodeids
                        .push(node.left_child.unwrap_or(0) as i64);
                    builder
                        .nodes_falsenodeids
                        .push(node.right_child.unwrap_or(0) as i64);
                    builder.nodes_missing_value_tracks_true.push(1);
                }
            }
        }

        // Use TreeEnsembleRegressor + ArgMax (same pattern as other classifiers)
        builder.n_targets = n_classes;
        let attributes = builder.build_regressor_attributes(n_classes as i64, "SUM");

        let input = create_tensor_input(
            &config.input_name,
            n_features,
            batch_size,
            TensorProtoDataType::Float,
        );

        let output =
            create_tensor_output_1d(&config.output_name, batch_size, TensorProtoDataType::Float);

        let tree_node = NodeProto {
            input: vec![config.input_name.clone()],
            output: vec!["raw_scores".to_string()],
            name: "TreeEnsembleRegressor_0".to_string(),
            op_type: "TreeEnsembleRegressor".to_string(),
            domain: "ai.onnx.ml".to_string(),
            attribute: attributes,
            doc_string: String::new(),
        };

        let argmax = crate::onnx::create_argmax_node("raw_scores", "argmax_out", "ArgMax_0", 1, 0);
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

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }
}

impl OnnxExportable for AdaBoostRegressor {
    fn to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("ONNX export"));
        }

        let estimators = self
            .estimators()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let estimator_weights = self
            .estimator_weights()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;
        let n_features = self
            .n_features()
            .ok_or_else(|| FerroError::not_fitted("ONNX export"))?;

        let tree_refs: Vec<&TreeStructure> =
            estimators.iter().filter_map(|est| est.tree()).collect();

        if tree_refs.is_empty() {
            return Err(FerroError::not_fitted("No trees found"));
        }

        check_tree_node_limit(&tree_refs)?;

        // Use weighted SUM approximation (note: exact AdaBoost.R2 uses weighted median)
        let leaf_scales: Vec<f32> = (0..tree_refs.len())
            .map(|i| {
                if i < estimator_weights.len() {
                    estimator_weights[i] as f32
                } else {
                    1.0
                }
            })
            .collect();

        // Normalize weights so they sum to 1 (weighted average)
        let total_weight: f32 = leaf_scales.iter().sum();
        let normalized_scales: Vec<f32> = if total_weight > 0.0 {
            leaf_scales.iter().map(|&w| w / total_weight).collect()
        } else {
            leaf_scales
        };

        let graph = create_tree_regressor_graph_with_scaling(
            &tree_refs,
            &normalized_scales,
            0.0,
            n_features,
            config,
        );
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
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

    fn create_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0,
                9.0, 9.0, 10.0, 10.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
        (x, y)
    }

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    #[test]
    fn test_decision_tree_regressor_onnx_export() {
        let (x, y) = create_regression_data();
        let mut model = DecisionTreeRegressor::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("decision_tree_regressor_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert_eq!(onnx_model.producer_name, "FerroML");
        assert!(onnx_model.graph.is_some());

        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
        assert_eq!(graph.node[0].domain, "ai.onnx.ml");
    }

    #[test]
    fn test_decision_tree_classifier_onnx_export() {
        let (x, y) = create_classification_data();
        let mut model = DecisionTreeClassifier::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("decision_tree_classifier_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());

        let graph = onnx_model.graph.unwrap();
        // Uses TreeEnsembleRegressor + ArgMax + Cast
        assert_eq!(graph.node.len(), 3);
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
        assert_eq!(graph.node[0].domain, "ai.onnx.ml");
        assert_eq!(graph.node[1].op_type, "ArgMax");
        assert_eq!(graph.node[2].op_type, "Cast");
    }

    #[test]
    fn test_random_forest_regressor_onnx_export() {
        let (x, y) = create_regression_data();
        let mut model = RandomForestRegressor::new()
            .with_n_estimators(5)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("random_forest_regressor_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());

        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
    }

    #[test]
    fn test_random_forest_classifier_onnx_export() {
        let (x, y) = create_classification_data();
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(5)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("random_forest_classifier_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());

        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.node.len(), 3);
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
    }

    #[test]
    fn test_tree_onnx_opset_imports() {
        let (x, y) = create_regression_data();
        let mut model = DecisionTreeRegressor::new();
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();

        // Should have both default ONNX domain and ai.onnx.ml domain
        assert_eq!(onnx_model.opset_import.len(), 2);
        assert!(onnx_model
            .opset_import
            .iter()
            .any(|op| op.domain.is_empty())); // Default domain
        assert!(onnx_model
            .opset_import
            .iter()
            .any(|op| op.domain == "ai.onnx.ml"));
    }

    #[test]
    fn test_tree_not_fitted() {
        let model = DecisionTreeRegressor::new();
        let config = OnnxConfig::new("test");
        let result = model.to_onnx(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_trees_regressor_onnx_export() {
        let (x, y) = create_regression_data();
        let mut model = ExtraTreesRegressor::new()
            .with_n_estimators(5)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("extra_trees_regressor_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
    }

    #[test]
    fn test_extra_trees_classifier_onnx_export() {
        let (x, y) = create_classification_data();
        let mut model = ExtraTreesClassifier::new()
            .with_n_estimators(5)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("extra_trees_classifier_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
    }

    #[test]
    fn test_gradient_boosting_regressor_onnx_export() {
        let (x, y) = create_regression_data();
        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("gbr_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
    }

    #[test]
    fn test_gradient_boosting_classifier_binary_onnx_export() {
        let (x, y) = create_classification_data();
        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("gbc_binary_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph
            .node
            .iter()
            .any(|n| n.op_type == "TreeEnsembleRegressor"));
        assert!(graph.node.iter().any(|n| n.op_type == "Sigmoid"));
    }

    #[test]
    fn test_gradient_boosting_classifier_multiclass_onnx_export() {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 4.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 5.0, 8.0, 6.0,
                9.0, 1.0, 10.0, 2.0, 11.0, 1.0, 12.0, 2.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("gbc_multi_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert!(graph
            .node
            .iter()
            .any(|n| n.op_type == "TreeEnsembleRegressor"));
        assert!(graph.node.iter().any(|n| n.op_type == "ArgMax"));
    }

    #[test]
    fn test_adaboost_classifier_onnx_export() {
        let (x, y) = create_classification_data();
        let mut model = AdaBoostClassifier::new(10).with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("adaboost_clf_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
    }

    #[test]
    fn test_adaboost_regressor_onnx_export() {
        let (x, y) = create_regression_data();
        let mut model = AdaBoostRegressor::new(10).with_random_state(42);
        model.fit(&x, &y).unwrap();

        let config = OnnxConfig::new("adaboost_reg_test");
        let bytes = model.to_onnx(&config).unwrap();

        let onnx_model = ModelProto::decode(&*bytes).unwrap();
        assert!(onnx_model.graph.is_some());
        let graph = onnx_model.graph.unwrap();
        assert_eq!(graph.node[0].op_type, "TreeEnsembleRegressor");
    }

    #[test]
    fn test_extra_trees_not_fitted() {
        let model = ExtraTreesRegressor::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    #[test]
    fn test_gradient_boosting_not_fitted() {
        let model = GradientBoostingRegressor::new();
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }

    #[test]
    fn test_adaboost_not_fitted() {
        let model = AdaBoostClassifier::new(10);
        let config = OnnxConfig::new("test");
        assert!(model.to_onnx(&config).is_err());
    }
}
