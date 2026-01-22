//! ONNX Operator Implementations
//!
//! This module implements the ONNX operators needed to run FerroML-exported models.

use super::{InferenceError, Tensor, TensorI64, Value};
use std::collections::HashMap;

/// Trait for ONNX operators
pub trait Operator: Send + Sync {
    /// Execute the operator
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError>;

    /// Get operator name
    fn name(&self) -> &str;
}

// =============================================================================
// Standard ONNX Operators
// =============================================================================

/// MatMul operator: Y = A @ B
pub struct MatMulOp;

impl Operator for MatMulOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        if inputs.len() != 2 {
            return Err(InferenceError::RuntimeError(
                "MatMul requires exactly 2 inputs".to_string(),
            ));
        }

        let a = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("MatMul input A must be tensor".into()))?;
        let b = inputs[1]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("MatMul input B must be tensor".into()))?;

        let result = a.matmul(b)?;
        Ok(vec![Value::Tensor(result)])
    }

    fn name(&self) -> &str {
        "MatMul"
    }
}

/// Add operator: Y = A + B (with broadcasting)
pub struct AddOp;

impl Operator for AddOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        if inputs.len() != 2 {
            return Err(InferenceError::RuntimeError(
                "Add requires exactly 2 inputs".to_string(),
            ));
        }

        let a = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Add input A must be tensor".into()))?;
        let b = inputs[1]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Add input B must be tensor".into()))?;

        let result = a.add(b)?;
        Ok(vec![Value::Tensor(result)])
    }

    fn name(&self) -> &str {
        "Add"
    }
}

/// Squeeze operator: Remove dimensions of size 1
pub struct SqueezeOp {
    /// Axes to squeeze (if empty, squeeze all dims of size 1)
    pub axes: Vec<i64>,
}

impl Operator for SqueezeOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        // ONNX opset 13+ has axes as second input, earlier versions have it as attribute
        let input = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Squeeze input must be tensor".into()))?;

        let axes = if inputs.len() > 1 {
            // Axes from second input (opset 13+)
            if let Some(axes_tensor) = inputs[1].as_tensor() {
                axes_tensor.as_slice().iter().map(|&x| x as i64).collect()
            } else if let Some(axes_tensor) = inputs[1].as_tensor_i64() {
                axes_tensor.as_slice().to_vec()
            } else {
                self.axes.clone()
            }
        } else {
            self.axes.clone()
        };

        let result = input.clone().squeeze(&axes)?;
        Ok(vec![Value::Tensor(result)])
    }

    fn name(&self) -> &str {
        "Squeeze"
    }
}

/// Sigmoid operator: Y = 1 / (1 + exp(-X))
pub struct SigmoidOp;

impl Operator for SigmoidOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        if inputs.is_empty() {
            return Err(InferenceError::RuntimeError(
                "Sigmoid requires 1 input".to_string(),
            ));
        }

        let input = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Sigmoid input must be tensor".into()))?;

        let result = input.sigmoid();
        Ok(vec![Value::Tensor(result)])
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// Softmax operator
pub struct SoftmaxOp {
    /// Axis along which to compute softmax
    pub axis: i64,
}

impl Operator for SoftmaxOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        if inputs.is_empty() {
            return Err(InferenceError::RuntimeError(
                "Softmax requires 1 input".to_string(),
            ));
        }

        let input = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Softmax input must be tensor".into()))?;

        let result = input.softmax(self.axis)?;
        Ok(vec![Value::Tensor(result)])
    }

    fn name(&self) -> &str {
        "Softmax"
    }
}

/// Flatten operator
pub struct FlattenOp {
    /// Axis to flatten from
    pub axis: i64,
}

impl Operator for FlattenOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        if inputs.is_empty() {
            return Err(InferenceError::RuntimeError(
                "Flatten requires 1 input".to_string(),
            ));
        }

        let input = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Flatten input must be tensor".into()))?;

        let result = input.clone().flatten(self.axis)?;
        Ok(vec![Value::Tensor(result)])
    }

    fn name(&self) -> &str {
        "Flatten"
    }
}

/// Reshape operator
pub struct ReshapeOp;

impl Operator for ReshapeOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        if inputs.len() < 2 {
            return Err(InferenceError::RuntimeError(
                "Reshape requires 2 inputs (data, shape)".to_string(),
            ));
        }

        let input = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Reshape input must be tensor".into()))?;

        // Shape can be int64 tensor
        let new_shape: Vec<usize> = if let Some(shape_tensor) = inputs[1].as_tensor_i64() {
            shape_tensor.as_slice().iter().map(|&x| x as usize).collect()
        } else if let Some(shape_tensor) = inputs[1].as_tensor() {
            shape_tensor
                .as_slice()
                .iter()
                .map(|&x| x as usize)
                .collect()
        } else {
            return Err(InferenceError::TypeMismatch(
                "Reshape shape must be tensor".into(),
            ));
        };

        let result = input.clone().reshape(new_shape)?;
        Ok(vec![Value::Tensor(result)])
    }

    fn name(&self) -> &str {
        "Reshape"
    }
}

// =============================================================================
// ONNX-ML Tree Ensemble Operators
// =============================================================================

/// Node mode for tree traversal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeMode {
    /// Branch left if feature <= threshold
    BranchLeq,
    /// Branch left if feature < threshold
    BranchLt,
    /// Branch left if feature >= threshold
    BranchGte,
    /// Branch left if feature > threshold
    BranchGt,
    /// Branch left if feature == threshold
    BranchEq,
    /// Branch left if feature != threshold
    BranchNeq,
    /// Leaf node
    Leaf,
}

impl NodeMode {
    fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "BRANCH_LEQ" => NodeMode::BranchLeq,
            "BRANCH_LT" => NodeMode::BranchLt,
            "BRANCH_GTE" => NodeMode::BranchGte,
            "BRANCH_GT" => NodeMode::BranchGt,
            "BRANCH_EQ" => NodeMode::BranchEq,
            "BRANCH_NEQ" => NodeMode::BranchNeq,
            "LEAF" => NodeMode::Leaf,
            _ => NodeMode::Leaf,
        }
    }

    fn evaluate(&self, feature_value: f32, threshold: f32) -> bool {
        match self {
            NodeMode::BranchLeq => feature_value <= threshold,
            NodeMode::BranchLt => feature_value < threshold,
            NodeMode::BranchGte => feature_value >= threshold,
            NodeMode::BranchGt => feature_value > threshold,
            NodeMode::BranchEq => (feature_value - threshold).abs() < f32::EPSILON,
            NodeMode::BranchNeq => (feature_value - threshold).abs() >= f32::EPSILON,
            NodeMode::Leaf => false,
        }
    }
}

/// Tree node for inference
#[derive(Debug, Clone)]
struct TreeNode {
    feature_id: usize,
    threshold: f32,
    mode: NodeMode,
    true_node_id: usize,
    false_node_id: usize,
    missing_tracks_true: bool,
}

/// Aggregate function for tree ensemble
#[derive(Debug, Clone, Copy)]
pub enum AggregateFunction {
    /// Sum of all tree outputs
    Sum,
    /// Average of all tree outputs
    Average,
    /// Minimum of all tree outputs
    Min,
    /// Maximum of all tree outputs
    Max,
}

impl AggregateFunction {
    fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "SUM" => AggregateFunction::Sum,
            "AVERAGE" => AggregateFunction::Average,
            "MIN" => AggregateFunction::Min,
            "MAX" => AggregateFunction::Max,
            _ => AggregateFunction::Sum,
        }
    }
}

/// Post-transform function
#[derive(Debug, Clone, Copy)]
pub enum PostTransform {
    /// No transformation
    None,
    /// Softmax normalization
    Softmax,
    /// Logistic function (sigmoid)
    Logistic,
    /// Normalize (sum to 1)
    SoftmaxZero,
    /// Probit (not implemented)
    Probit,
}

impl PostTransform {
    fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "NONE" => PostTransform::None,
            "SOFTMAX" => PostTransform::Softmax,
            "LOGISTIC" => PostTransform::Logistic,
            "SOFTMAX_ZERO" => PostTransform::SoftmaxZero,
            "PROBIT" => PostTransform::Probit,
            _ => PostTransform::None,
        }
    }
}

/// TreeEnsembleRegressor operator
pub struct TreeEnsembleRegressorOp {
    /// Number of targets
    pub n_targets: usize,
    /// Number of trees
    pub n_trees: usize,
    /// Tree structures: trees\[tree_id\]\[node_id\] = TreeNode
    trees: Vec<Vec<TreeNode>>,
    /// Target values: (tree_id, node_id, target_id) -> weight
    target_weights: HashMap<(usize, usize, usize), f32>,
    /// Aggregate function
    aggregate: AggregateFunction,
    /// Base values for each target
    base_values: Vec<f32>,
}

impl TreeEnsembleRegressorOp {
    /// Create from ONNX attributes
    #[allow(clippy::too_many_arguments)]
    pub fn from_attributes(
        n_targets: i64,
        nodes_featureids: &[i64],
        nodes_values: &[f32],
        nodes_modes: &[Vec<u8>],
        nodes_treeids: &[i64],
        nodes_nodeids: &[i64],
        nodes_truenodeids: &[i64],
        nodes_falsenodeids: &[i64],
        nodes_missing_tracks_true: &[i64],
        target_nodeids: &[i64],
        target_treeids: &[i64],
        target_ids: &[i64],
        target_weights: &[f32],
        aggregate: &str,
        base_values: &[f32],
    ) -> Result<Self, InferenceError> {
        let n_targets = n_targets as usize;

        // Build tree structures
        let n_trees = nodes_treeids
            .iter()
            .max()
            .map(|&x| x as usize + 1)
            .unwrap_or(0);

        let mut trees: Vec<Vec<TreeNode>> = vec![Vec::new(); n_trees];

        // Group nodes by tree
        let mut tree_node_counts: Vec<usize> = vec![0; n_trees];
        for &tree_id in nodes_treeids {
            tree_node_counts[tree_id as usize] += 1;
        }

        // Initialize tree vectors
        for (tree_id, &count) in tree_node_counts.iter().enumerate() {
            trees[tree_id] = vec![
                TreeNode {
                    feature_id: 0,
                    threshold: 0.0,
                    mode: NodeMode::Leaf,
                    true_node_id: 0,
                    false_node_id: 0,
                    missing_tracks_true: false,
                };
                count
            ];
        }

        // Fill in node data
        for i in 0..nodes_treeids.len() {
            let tree_id = nodes_treeids[i] as usize;
            let node_id = nodes_nodeids[i] as usize;

            let mode_str = String::from_utf8_lossy(&nodes_modes[i]);
            let mode = NodeMode::from_str(&mode_str);

            if node_id < trees[tree_id].len() {
                trees[tree_id][node_id] = TreeNode {
                    feature_id: nodes_featureids[i] as usize,
                    threshold: nodes_values[i],
                    mode,
                    true_node_id: nodes_truenodeids[i] as usize,
                    false_node_id: nodes_falsenodeids[i] as usize,
                    missing_tracks_true: nodes_missing_tracks_true
                        .get(i)
                        .map(|&x| x != 0)
                        .unwrap_or(false),
                };
            }
        }

        // Build target weights map
        let mut weights_map = HashMap::new();
        for i in 0..target_nodeids.len() {
            let tree_id = target_treeids[i] as usize;
            let node_id = target_nodeids[i] as usize;
            let target_id = target_ids[i] as usize;
            weights_map.insert((tree_id, node_id, target_id), target_weights[i]);
        }

        let base_values = if base_values.is_empty() {
            vec![0.0; n_targets]
        } else {
            base_values.to_vec()
        };

        Ok(Self {
            n_targets,
            n_trees,
            trees,
            target_weights: weights_map,
            aggregate: AggregateFunction::from_str(aggregate),
            base_values,
        })
    }

    /// Traverse a single tree for a sample
    fn traverse_tree(&self, tree_id: usize, features: &[f32]) -> usize {
        let tree = &self.trees[tree_id];
        let mut node_id = 0;

        while node_id < tree.len() {
            let node = &tree[node_id];

            if node.mode == NodeMode::Leaf {
                return node_id;
            }

            let feature_value = features
                .get(node.feature_id)
                .copied()
                .unwrap_or(f32::NAN);

            // Handle missing values
            let go_left = if feature_value.is_nan() {
                node.missing_tracks_true
            } else {
                node.mode.evaluate(feature_value, node.threshold)
            };

            node_id = if go_left {
                node.true_node_id
            } else {
                node.false_node_id
            };
        }

        0 // Fallback
    }
}

impl Operator for TreeEnsembleRegressorOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        let input = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Input must be tensor".into()))?;

        let shape = input.shape();
        let n_samples = shape[0];
        let n_features = if shape.len() > 1 { shape[1] } else { shape[0] };

        let mut outputs = vec![0.0f32; n_samples * self.n_targets];

        for sample_idx in 0..n_samples {
            // Get features for this sample
            let features: Vec<f32> = (0..n_features)
                .map(|f| input.as_slice()[sample_idx * n_features + f])
                .collect();

            // Initialize with base values
            for t in 0..self.n_targets {
                outputs[sample_idx * self.n_targets + t] = self.base_values[t];
            }

            // Collect predictions from all trees
            let mut tree_preds: Vec<Vec<f32>> = vec![vec![0.0; self.n_targets]; self.n_trees];

            for tree_id in 0..self.n_trees {
                let leaf_id = self.traverse_tree(tree_id, &features);

                for target_id in 0..self.n_targets {
                    if let Some(&weight) = self.target_weights.get(&(tree_id, leaf_id, target_id)) {
                        tree_preds[tree_id][target_id] = weight;
                    }
                }
            }

            // Aggregate across trees
            for target_id in 0..self.n_targets {
                let values: Vec<f32> = tree_preds.iter().map(|p| p[target_id]).collect();
                let agg_value = match self.aggregate {
                    AggregateFunction::Sum => values.iter().sum(),
                    AggregateFunction::Average => {
                        values.iter().sum::<f32>() / values.len() as f32
                    }
                    AggregateFunction::Min => {
                        values.iter().cloned().fold(f32::INFINITY, f32::min)
                    }
                    AggregateFunction::Max => {
                        values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    }
                };
                outputs[sample_idx * self.n_targets + target_id] += agg_value;
            }
        }

        // Squeeze to 1D if single target
        let output_shape = if self.n_targets == 1 {
            vec![n_samples]
        } else {
            vec![n_samples, self.n_targets]
        };

        Ok(vec![Value::Tensor(Tensor::from_vec(outputs, output_shape))])
    }

    fn name(&self) -> &str {
        "TreeEnsembleRegressor"
    }
}

/// TreeEnsembleClassifier operator
pub struct TreeEnsembleClassifierOp {
    /// Number of classes
    pub n_classes: usize,
    /// Class labels
    pub class_labels: Vec<i64>,
    /// Number of trees
    pub n_trees: usize,
    /// Tree structures
    trees: Vec<Vec<TreeNode>>,
    /// Class weights: (tree_id, node_id, class_id) -> weight
    class_weights: HashMap<(usize, usize, usize), f32>,
    /// Post transform
    post_transform: PostTransform,
    /// Base values
    base_values: Vec<f32>,
}

impl TreeEnsembleClassifierOp {
    /// Create from ONNX attributes
    #[allow(clippy::too_many_arguments)]
    pub fn from_attributes(
        class_labels: &[i64],
        nodes_featureids: &[i64],
        nodes_values: &[f32],
        nodes_modes: &[Vec<u8>],
        nodes_treeids: &[i64],
        nodes_nodeids: &[i64],
        nodes_truenodeids: &[i64],
        nodes_falsenodeids: &[i64],
        nodes_missing_tracks_true: &[i64],
        class_nodeids: &[i64],
        class_treeids: &[i64],
        class_ids: &[i64],
        class_weights: &[f32],
        post_transform: &str,
        base_values: &[f32],
    ) -> Result<Self, InferenceError> {
        let n_classes = class_labels.len();

        // Build tree structures (same as regressor)
        let n_trees = nodes_treeids
            .iter()
            .max()
            .map(|&x| x as usize + 1)
            .unwrap_or(0);

        let mut trees: Vec<Vec<TreeNode>> = vec![Vec::new(); n_trees];

        let mut tree_node_counts: Vec<usize> = vec![0; n_trees];
        for &tree_id in nodes_treeids {
            tree_node_counts[tree_id as usize] += 1;
        }

        for (tree_id, &count) in tree_node_counts.iter().enumerate() {
            trees[tree_id] = vec![
                TreeNode {
                    feature_id: 0,
                    threshold: 0.0,
                    mode: NodeMode::Leaf,
                    true_node_id: 0,
                    false_node_id: 0,
                    missing_tracks_true: false,
                };
                count
            ];
        }

        for i in 0..nodes_treeids.len() {
            let tree_id = nodes_treeids[i] as usize;
            let node_id = nodes_nodeids[i] as usize;

            let mode_str = String::from_utf8_lossy(&nodes_modes[i]);
            let mode = NodeMode::from_str(&mode_str);

            if node_id < trees[tree_id].len() {
                trees[tree_id][node_id] = TreeNode {
                    feature_id: nodes_featureids[i] as usize,
                    threshold: nodes_values[i],
                    mode,
                    true_node_id: nodes_truenodeids[i] as usize,
                    false_node_id: nodes_falsenodeids[i] as usize,
                    missing_tracks_true: nodes_missing_tracks_true
                        .get(i)
                        .map(|&x| x != 0)
                        .unwrap_or(false),
                };
            }
        }

        // Build class weights map
        let mut weights_map = HashMap::new();
        for i in 0..class_nodeids.len() {
            let tree_id = class_treeids[i] as usize;
            let node_id = class_nodeids[i] as usize;
            let class_id = class_ids[i] as usize;
            weights_map.insert((tree_id, node_id, class_id), class_weights[i]);
        }

        let base_values = if base_values.is_empty() {
            vec![0.0; n_classes]
        } else {
            base_values.to_vec()
        };

        Ok(Self {
            n_classes,
            class_labels: class_labels.to_vec(),
            n_trees,
            trees,
            class_weights: weights_map,
            post_transform: PostTransform::from_str(post_transform),
            base_values,
        })
    }

    /// Traverse a single tree
    fn traverse_tree(&self, tree_id: usize, features: &[f32]) -> usize {
        let tree = &self.trees[tree_id];
        let mut node_id = 0;

        while node_id < tree.len() {
            let node = &tree[node_id];

            if node.mode == NodeMode::Leaf {
                return node_id;
            }

            let feature_value = features
                .get(node.feature_id)
                .copied()
                .unwrap_or(f32::NAN);

            let go_left = if feature_value.is_nan() {
                node.missing_tracks_true
            } else {
                node.mode.evaluate(feature_value, node.threshold)
            };

            node_id = if go_left {
                node.true_node_id
            } else {
                node.false_node_id
            };
        }

        0
    }

    /// Apply post transform to class scores
    fn apply_post_transform(&self, scores: &mut [f32]) {
        match self.post_transform {
            PostTransform::None => {}
            PostTransform::Softmax | PostTransform::SoftmaxZero => {
                let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|&x| (x - max_val).exp()).sum();
                for score in scores.iter_mut() {
                    *score = (*score - max_val).exp() / exp_sum;
                }
            }
            PostTransform::Logistic => {
                for score in scores.iter_mut() {
                    *score = 1.0 / (1.0 + (-*score).exp());
                }
            }
            PostTransform::Probit => {
                // Not implemented - leave as is
            }
        }
    }
}

impl Operator for TreeEnsembleClassifierOp {
    fn execute(&self, inputs: &[&Value]) -> Result<Vec<Value>, InferenceError> {
        let input = inputs[0]
            .as_tensor()
            .ok_or_else(|| InferenceError::TypeMismatch("Input must be tensor".into()))?;

        let shape = input.shape();
        let n_samples = shape[0];
        let n_features = if shape.len() > 1 { shape[1] } else { shape[0] };

        let mut labels = Vec::with_capacity(n_samples);
        let mut probas: Vec<HashMap<i64, f32>> = Vec::with_capacity(n_samples);

        for sample_idx in 0..n_samples {
            let features: Vec<f32> = (0..n_features)
                .map(|f| input.as_slice()[sample_idx * n_features + f])
                .collect();

            // Initialize scores with base values
            let mut scores = self.base_values.clone();

            // Collect predictions from all trees
            for tree_id in 0..self.n_trees {
                let leaf_id = self.traverse_tree(tree_id, &features);

                for class_id in 0..self.n_classes {
                    if let Some(&weight) = self.class_weights.get(&(tree_id, leaf_id, class_id)) {
                        scores[class_id] += weight;
                    }
                }
            }

            // Apply post transform
            self.apply_post_transform(&mut scores);

            // Find predicted class
            let (best_class_idx, _) = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));

            labels.push(self.class_labels[best_class_idx]);

            // Build probability map
            let mut proba_map = HashMap::new();
            for (class_idx, &score) in scores.iter().enumerate() {
                proba_map.insert(self.class_labels[class_idx], score);
            }
            probas.push(proba_map);
        }

        // Output 1: Labels [batch_size]
        let labels_tensor = TensorI64::from_vec(labels, vec![n_samples]);

        // Output 2: Probabilities as sequence of maps
        Ok(vec![
            Value::TensorI64(labels_tensor),
            Value::SequenceMapI64F32(probas),
        ])
    }

    fn name(&self) -> &str {
        "TreeEnsembleClassifier"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_op() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let op = MatMulOp;
        let result = op
            .execute(&[&Value::Tensor(a), &Value::Tensor(b)])
            .unwrap();

        let out = result[0].as_tensor().unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.as_slice(), &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_add_op() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);

        let op = AddOp;
        let result = op
            .execute(&[&Value::Tensor(a), &Value::Tensor(b)])
            .unwrap();

        let out = result[0].as_tensor().unwrap();
        assert_eq!(out.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_sigmoid_op() {
        let input = Tensor::from_vec(vec![0.0], vec![1]);

        let op = SigmoidOp;
        let result = op.execute(&[&Value::Tensor(input)]).unwrap();

        let out = result[0].as_tensor().unwrap();
        assert!((out.as_slice()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_op() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);

        let op = SoftmaxOp { axis: 1 };
        let result = op.execute(&[&Value::Tensor(input)]).unwrap();

        let out = result[0].as_tensor().unwrap();
        let sum: f32 = out.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_node_mode_evaluate() {
        assert!(NodeMode::BranchLeq.evaluate(1.0, 2.0));
        assert!(NodeMode::BranchLeq.evaluate(2.0, 2.0));
        assert!(!NodeMode::BranchLeq.evaluate(3.0, 2.0));

        assert!(NodeMode::BranchLt.evaluate(1.0, 2.0));
        assert!(!NodeMode::BranchLt.evaluate(2.0, 2.0));
    }
}
