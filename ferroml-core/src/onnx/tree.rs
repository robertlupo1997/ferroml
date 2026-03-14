//! ONNX Export for Tree Models
//!
//! Implements ONNX export for DecisionTreeClassifier, DecisionTreeRegressor,
//! RandomForestClassifier, RandomForestRegressor.
//!
//! Uses TreeEnsembleClassifier and TreeEnsembleRegressor operators from ONNX-ML domain.

use crate::models::forest::{RandomForestClassifier, RandomForestRegressor};
use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor, TreeStructure};
use crate::models::Model;
use crate::onnx::{
    create_model_proto, create_tensor_input, create_tensor_output, create_tensor_output_1d,
    AttributeProto, AttributeProtoType, GraphProto, NodeProto, OnnxConfig, OnnxExportable,
    TensorProtoDataType,
};
use crate::{FerroError, Result};
use prost::Message;

/// Node modes for tree ensemble
const NODE_MODE_BRANCH_LEQ: &str = "BRANCH_LEQ";
const NODE_MODE_LEAF: &str = "LEAF";

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

    /// Add a tree to the ensemble
    fn add_tree(&mut self, tree: &TreeStructure, tree_id: i64) {
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

                // Add target values (leaf node outputs)
                for (target_id, &value) in node.value.iter().enumerate() {
                    self.target_treeids.push(tree_id);
                    self.target_nodeids.push(node.id as i64);
                    self.target_ids.push(target_id as i64);
                    self.target_weights.push(value as f32);
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

    /// Build classifier attributes
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

/// Create a tree ensemble classifier graph
fn create_tree_classifier_graph(
    trees: &[&TreeStructure],
    n_features: usize,
    n_classes: usize,
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

    // Outputs: predicted labels and probabilities
    let label_output = create_tensor_output_1d(
        &format!("{}_labels", config.output_name),
        batch_size,
        TensorProtoDataType::Int64,
    );

    // Probabilities output: tensor(float) with shape [N, n_classes]
    let proba_output = create_tensor_output(
        &config.output_name,
        n_classes,
        batch_size,
        TensorProtoDataType::Float,
    );

    // Build tree ensemble attributes
    let mut builder = TreeEnsembleBuilder::new();
    for (tree_id, tree) in trees.iter().enumerate() {
        builder.add_tree(tree, tree_id as i64);
    }

    let class_labels: Vec<i64> = (0..n_classes as i64).collect();
    // Use SOFTMAX for normalized probabilities
    let attributes = builder.build_classifier_attributes(class_labels, "SOFTMAX");

    // TreeEnsembleClassifier node
    let node = NodeProto {
        input: vec![config.input_name.clone()],
        output: vec![
            format!("{}_labels", config.output_name),
            config.output_name.clone(),
        ],
        name: "TreeEnsembleClassifier_0".to_string(),
        op_type: "TreeEnsembleClassifier".to_string(),
        domain: "ai.onnx.ml".to_string(),
        attribute: attributes,
        doc_string: String::new(),
    };

    GraphProto {
        name: config.model_name.clone(),
        node: vec![node],
        input: vec![input],
        output: vec![label_output, proba_output],
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

        let graph = create_tree_classifier_graph(&[tree], n_features, n_classes, config);
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }

    fn onnx_n_outputs(&self) -> usize {
        2 // Labels and probabilities
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

        let graph = create_tree_classifier_graph(&tree_refs, n_features, n_classes, config);
        let model = create_model_proto(graph, config, true);
        Ok(model.encode_to_vec())
    }

    fn onnx_n_features(&self) -> Option<usize> {
        self.n_features()
    }

    fn onnx_n_outputs(&self) -> usize {
        2 // Labels and probabilities
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
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "TreeEnsembleClassifier");
        assert_eq!(graph.node[0].domain, "ai.onnx.ml");
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
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "TreeEnsembleClassifier");
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
}
