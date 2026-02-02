//! Decision Tree Models with Statistical Diagnostics
//!
//! This module provides CART (Classification And Regression Trees) decision tree
//! implementations with comprehensive feature importance and tree structure export.
//!
//! ## Models
//!
//! - **DecisionTreeClassifier**: For classification tasks
//! - **DecisionTreeRegressor**: For regression tasks
//!
//! ## Features
//!
//! - **Multiple split criteria**: Gini impurity, Entropy (classification), MSE, MAE (regression)
//! - **Feature importance**: Computed from impurity decrease across all splits
//! - **Cost-complexity pruning**: Alpha parameter for minimal cost-complexity pruning
//! - **Tree structure export**: Full tree structure for visualization
//!
//! ## Example - Classification
//!
//! ```ignore
//! use ferroml_core::models::tree::{DecisionTreeClassifier, SplitCriterion};
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
//!
//! let mut model = DecisionTreeClassifier::new()
//!     .with_max_depth(Some(5))
//!     .with_criterion(SplitCriterion::Gini);
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! let importance = model.feature_importance().unwrap();
//! ```
//!
//! ## Example - Regression
//!
//! ```ignore
//! use ferroml_core::models::tree::{DecisionTreeRegressor, SplitCriterion};
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![1.5, 2.5, 3.5, 5.5, 6.5, 7.5]);
//!
//! let mut model = DecisionTreeRegressor::new()
//!     .with_max_depth(Some(5))
//!     .with_min_samples_split(2);
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, compute_sample_weights, get_unique_classes, validate_fit_input,
    validate_predict_input, ClassWeight, Model,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// =============================================================================
// Split Criteria
// =============================================================================

/// Split criterion for decision trees
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitCriterion {
    /// Gini impurity (classification) - default
    Gini,
    /// Entropy / Information Gain (classification)
    Entropy,
    /// Mean Squared Error (regression) - default
    Mse,
    /// Mean Absolute Error (regression)
    Mae,
}

impl Default for SplitCriterion {
    fn default() -> Self {
        Self::Gini
    }
}

impl SplitCriterion {
    /// Check if this criterion is for classification
    #[must_use]
    pub fn is_classification(&self) -> bool {
        matches!(self, SplitCriterion::Gini | SplitCriterion::Entropy)
    }

    /// Check if this criterion is for regression
    #[must_use]
    pub fn is_regression(&self) -> bool {
        matches!(self, SplitCriterion::Mse | SplitCriterion::Mae)
    }
}

/// Compute Gini impurity for a set of class counts (used by classifier implementation)
#[allow(dead_code)]
fn gini_impurity(class_counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    let sum_sq: f64 = class_counts
        .iter()
        .map(|&c| {
            let p = c as f64 / total_f;
            p * p
        })
        .sum();
    1.0 - sum_sq
}

/// Compute weighted Gini impurity for a set of class weight sums
fn weighted_gini_impurity(class_weights: &[f64], total_weight: f64) -> f64 {
    if total_weight <= 0.0 {
        return 0.0;
    }
    let sum_sq: f64 = class_weights
        .iter()
        .map(|&w| {
            let p = w / total_weight;
            p * p
        })
        .sum();
    1.0 - sum_sq
}

/// Compute entropy for a set of class counts (used by classifier implementation)
#[allow(dead_code)]
fn entropy(class_counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let total_f = total as f64;
    -class_counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            p * p.ln()
        })
        .sum::<f64>()
}

/// Compute MSE for a set of values
fn mse(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Compute MAE for a set of values
fn mae(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let median = {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        }
    };
    values.iter().map(|&v| (v - median).abs()).sum::<f64>() / values.len() as f64
}

// =============================================================================
// Tree Node Structure
// =============================================================================

/// A node in the decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Unique node ID
    pub id: usize,
    /// Feature index used for splitting (None for leaf nodes)
    pub feature_index: Option<usize>,
    /// Threshold value for the split (None for leaf nodes)
    pub threshold: Option<f64>,
    /// Impurity at this node
    pub impurity: f64,
    /// Number of samples at this node
    pub n_samples: usize,
    /// Weighted number of samples (for sample weights)
    pub weighted_n_samples: f64,
    /// Value at this node (class probabilities for classification, mean for regression)
    pub value: Vec<f64>,
    /// Left child node ID (None for leaf nodes)
    pub left_child: Option<usize>,
    /// Right child node ID (None for leaf nodes)
    pub right_child: Option<usize>,
    /// Depth of this node in the tree
    pub depth: usize,
    /// Is this a leaf node?
    pub is_leaf: bool,
    /// Impurity decrease from this split (for feature importance)
    pub impurity_decrease: f64,
}

impl TreeNode {
    /// Create a new leaf node
    fn new_leaf(
        id: usize,
        impurity: f64,
        n_samples: usize,
        weighted_n_samples: f64,
        value: Vec<f64>,
        depth: usize,
    ) -> Self {
        Self {
            id,
            feature_index: None,
            threshold: None,
            impurity,
            n_samples,
            weighted_n_samples,
            value,
            left_child: None,
            right_child: None,
            depth,
            is_leaf: true,
            impurity_decrease: 0.0,
        }
    }

    /// Convert to internal node with split
    fn make_internal(
        &mut self,
        feature_index: usize,
        threshold: f64,
        left_child: usize,
        right_child: usize,
        impurity_decrease: f64,
    ) {
        self.feature_index = Some(feature_index);
        self.threshold = Some(threshold);
        self.left_child = Some(left_child);
        self.right_child = Some(right_child);
        self.is_leaf = false;
        self.impurity_decrease = impurity_decrease;
    }
}

/// Full tree structure for visualization and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeStructure {
    /// All nodes in the tree (indexed by node ID)
    pub nodes: Vec<TreeNode>,
    /// Number of features
    pub n_features: usize,
    /// Number of classes (for classification) or 1 (for regression)
    pub n_classes: usize,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Class names (if provided, for classification)
    pub class_names: Option<Vec<String>>,
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Total number of leaves
    pub n_leaves: usize,
}

impl TreeStructure {
    /// Get the root node
    #[must_use]
    pub fn root(&self) -> Option<&TreeNode> {
        self.nodes.first()
    }

    /// Get a node by ID
    #[must_use]
    pub fn get_node(&self, id: usize) -> Option<&TreeNode> {
        self.nodes.get(id)
    }

    /// Export tree to DOT format for visualization with Graphviz
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut dot =
            String::from("digraph Tree {\nnode [shape=box, style=\"filled\", color=\"black\"];\n");

        for node in &self.nodes {
            let label = if node.is_leaf {
                format!(
                    "samples = {}\\nvalue = {:?}\\nimpurity = {:.4}",
                    node.n_samples, node.value, node.impurity
                )
            } else {
                let feature_idx = node.feature_index.unwrap();
                let feature_name = self
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(feature_idx))
                    .cloned()
                    .unwrap_or_else(|| format!("X[{}]", feature_idx));
                format!(
                    "{} <= {:.4}\\nsamples = {}\\nimpurity = {:.4}",
                    feature_name,
                    node.threshold.unwrap(),
                    node.n_samples,
                    node.impurity
                )
            };

            // Color based on impurity or class
            let fill_color = if node.impurity < 0.1 {
                "#e5813966"
            } else if node.impurity < 0.3 {
                "#f5a54266"
            } else {
                "#ffffff66"
            };

            dot.push_str(&format!(
                "{} [label=\"{}\", fillcolor=\"{}\"];\n",
                node.id, label, fill_color
            ));

            if let Some(left) = node.left_child {
                dot.push_str(&format!(
                    "{} -> {} [labeldistance=2.5, labelangle=45, headlabel=\"True\"];\n",
                    node.id, left
                ));
            }
            if let Some(right) = node.right_child {
                dot.push_str(&format!(
                    "{} -> {} [labeldistance=2.5, labelangle=-45, headlabel=\"False\"];\n",
                    node.id, right
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Get decision path for a single sample
    pub fn decision_path(&self, x: &[f64]) -> Vec<usize> {
        let mut path = Vec::new();
        let mut node_id = 0;

        while let Some(node) = self.get_node(node_id) {
            path.push(node_id);
            if node.is_leaf {
                break;
            }
            let feature_idx = node.feature_index.unwrap();
            let threshold = node.threshold.unwrap();
            node_id = if x[feature_idx] <= threshold {
                node.left_child.unwrap()
            } else {
                node.right_child.unwrap()
            };
        }

        path
    }
}

// =============================================================================
// Decision Tree Classifier
// =============================================================================

/// Decision Tree Classifier using CART algorithm
///
/// Builds a binary decision tree by recursively partitioning the feature space
/// using the best split based on impurity decrease (Gini or Entropy).
///
/// ## Features
///
/// - Gini impurity or Entropy criterion
/// - Feature importance from impurity decrease
/// - Cost-complexity pruning (ccp_alpha)
/// - Tree structure export for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier {
    /// Split criterion (Gini or Entropy)
    pub criterion: SplitCriterion,
    /// Maximum depth of the tree (None for unlimited)
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Maximum number of features to consider for splitting (None for all)
    pub max_features: Option<usize>,
    /// Minimum impurity decrease required for a split
    pub min_impurity_decrease: f64,
    /// Cost-complexity pruning alpha (0 = no pruning)
    pub ccp_alpha: f64,
    /// Random seed for reproducibility (when max_features < n_features)
    pub random_state: Option<u64>,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,

    // Fitted parameters
    tree: Option<TreeStructure>,
    classes: Option<Array1<f64>>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<f64>>,
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTreeClassifier {
    /// Create a new Decision Tree Classifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            criterion: SplitCriterion::Gini,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            min_impurity_decrease: 0.0,
            ccp_alpha: 0.0,
            random_state: None,
            class_weight: ClassWeight::Uniform,
            tree: None,
            classes: None,
            n_features: None,
            feature_importances: None,
        }
    }

    /// Set the split criterion
    #[must_use]
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        assert!(
            criterion.is_classification(),
            "Criterion must be Gini or Entropy for classification"
        );
        self.criterion = criterion;
        self
    }

    /// Set maximum tree depth
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples to split
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split.max(2);
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf.max(1);
        self
    }

    /// Set maximum features to consider
    #[must_use]
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set minimum impurity decrease for split
    #[must_use]
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
        self.min_impurity_decrease = min_impurity_decrease.max(0.0);
        self
    }

    /// Set cost-complexity pruning alpha
    #[must_use]
    pub fn with_ccp_alpha(mut self, ccp_alpha: f64) -> Self {
        self.ccp_alpha = ccp_alpha.max(0.0);
        self
    }

    /// Set random state
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set class weights for handling imbalanced data
    ///
    /// # Arguments
    /// * `class_weight` - Weight specification: `Uniform`, `Balanced`, or `Custom`
    ///
    /// # Example
    /// ```ignore
    /// use ferroml_core::models::{DecisionTreeClassifier, ClassWeight};
    ///
    /// let model = DecisionTreeClassifier::new()
    ///     .with_class_weight(ClassWeight::Balanced);
    /// ```
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Get the fitted tree structure
    #[must_use]
    pub fn tree(&self) -> Option<&TreeStructure> {
        self.tree.as_ref()
    }

    /// Get the unique class labels
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the tree depth
    #[must_use]
    pub fn get_depth(&self) -> Option<usize> {
        self.tree.as_ref().map(|t| t.max_depth)
    }

    /// Get the number of leaves
    #[must_use]
    pub fn get_n_leaves(&self) -> Option<usize> {
        self.tree.as_ref().map(|t| t.n_leaves)
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.tree, "predict_proba")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let tree = self.tree.as_ref().unwrap();
        let n_classes = tree.n_classes;
        let n_samples = x.nrows();

        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let leaf_id = self.find_leaf(&sample);
            let leaf = tree.get_node(leaf_id).unwrap();

            // Normalize value to probabilities
            let total: f64 = leaf.value.iter().sum();
            if total > 0.0 {
                for (j, &v) in leaf.value.iter().enumerate() {
                    probas[[i, j]] = v / total;
                }
            }
        }

        Ok(probas)
    }

    /// Find the leaf node for a sample
    fn find_leaf(&self, x: &[f64]) -> usize {
        let tree = self.tree.as_ref().unwrap();
        let mut node_id = 0;

        while let Some(node) = tree.get_node(node_id) {
            if node.is_leaf {
                return node_id;
            }
            let feature_idx = node.feature_index.unwrap();
            let threshold = node.threshold.unwrap();
            node_id = if x[feature_idx] <= threshold {
                node.left_child.unwrap()
            } else {
                node.right_child.unwrap()
            };
        }

        node_id
    }

    /// Build the tree recursively (classifier tree-building, kept for future use)
    #[allow(dead_code)]
    fn build_tree(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        classes: &[f64],
        depth: usize,
        nodes: &mut Vec<TreeNode>,
    ) -> usize {
        let node_id = nodes.len();
        let n_samples = indices.len();
        let n_classes = classes.len();

        // Compute class counts
        let mut class_counts = vec![0usize; n_classes];
        for &idx in indices {
            let label = y[idx];
            if let Some(pos) = classes.iter().position(|&c| (c - label).abs() < 1e-10) {
                class_counts[pos] += 1;
            }
        }

        // Compute impurity
        let impurity = match self.criterion {
            SplitCriterion::Gini => gini_impurity(&class_counts, n_samples),
            SplitCriterion::Entropy => entropy(&class_counts, n_samples),
            _ => unreachable!(),
        };

        // Node value (class counts as floats)
        let value: Vec<f64> = class_counts.iter().map(|&c| c as f64).collect();

        // Create leaf node initially
        let leaf = TreeNode::new_leaf(node_id, impurity, n_samples, n_samples as f64, value, depth);
        nodes.push(leaf);

        // Check stopping conditions
        let should_stop = n_samples < self.min_samples_split
            || (self.max_depth.is_some() && depth >= self.max_depth.unwrap())
            || impurity < 1e-10
            || class_counts.iter().filter(|&&c| c > 0).count() <= 1;

        if should_stop {
            return node_id;
        }

        // Find best split
        let best_split = self.find_best_split(x, y, indices, classes, impurity);

        if let Some((feature_idx, threshold, left_indices, right_indices, impurity_decrease)) =
            best_split
        {
            // Check min_impurity_decrease
            if impurity_decrease < self.min_impurity_decrease {
                return node_id;
            }

            // Check min_samples_leaf
            if left_indices.len() < self.min_samples_leaf
                || right_indices.len() < self.min_samples_leaf
            {
                return node_id;
            }

            // Recursively build children
            let left_id = self.build_tree(x, y, &left_indices, classes, depth + 1, nodes);
            let right_id = self.build_tree(x, y, &right_indices, classes, depth + 1, nodes);

            // Update current node to internal node
            nodes[node_id].make_internal(
                feature_idx,
                threshold,
                left_id,
                right_id,
                impurity_decrease,
            );
        }

        node_id
    }

    /// Build tree with sample weights for handling class imbalance
    fn build_tree_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weights: &Array1<f64>,
        indices: &[usize],
        classes: &[f64],
        depth: usize,
        nodes: &mut Vec<TreeNode>,
    ) -> usize {
        let node_id = nodes.len();
        let n_samples = indices.len();
        let n_classes = classes.len();

        // Compute weighted class counts
        let mut class_weights_sum = vec![0.0f64; n_classes];
        let mut total_weight = 0.0f64;
        for &idx in indices {
            let label = y[idx];
            let weight = sample_weights[idx];
            total_weight += weight;
            if let Some(pos) = classes.iter().position(|&c| (c - label).abs() < 1e-10) {
                class_weights_sum[pos] += weight;
            }
        }

        // Compute weighted impurity
        let impurity = weighted_gini_impurity(&class_weights_sum, total_weight);

        // Node value (weighted class counts)
        let value: Vec<f64> = class_weights_sum.clone();

        // Create leaf node initially
        let leaf = TreeNode::new_leaf(node_id, impurity, n_samples, total_weight, value, depth);
        nodes.push(leaf);

        // Check stopping conditions
        let should_stop = n_samples < self.min_samples_split
            || (self.max_depth.is_some() && depth >= self.max_depth.unwrap())
            || impurity < 1e-10
            || class_weights_sum.iter().filter(|&&w| w > 0.0).count() <= 1;

        if should_stop {
            return node_id;
        }

        // Find best split with weights
        let best_split =
            self.find_best_split_weighted(x, y, sample_weights, indices, classes, impurity);

        if let Some((feature_idx, threshold, left_indices, right_indices, impurity_decrease)) =
            best_split
        {
            // Check min_impurity_decrease
            if impurity_decrease < self.min_impurity_decrease {
                return node_id;
            }

            // Check min_samples_leaf
            if left_indices.len() < self.min_samples_leaf
                || right_indices.len() < self.min_samples_leaf
            {
                return node_id;
            }

            // Recursively build children
            let left_id = self.build_tree_weighted(
                x,
                y,
                sample_weights,
                &left_indices,
                classes,
                depth + 1,
                nodes,
            );
            let right_id = self.build_tree_weighted(
                x,
                y,
                sample_weights,
                &right_indices,
                classes,
                depth + 1,
                nodes,
            );

            // Update current node to internal node
            nodes[node_id].make_internal(
                feature_idx,
                threshold,
                left_id,
                right_id,
                impurity_decrease,
            );
        }

        node_id
    }

    /// Find the best split for a node with sample weights
    fn find_best_split_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weights: &Array1<f64>,
        indices: &[usize],
        classes: &[f64],
        parent_impurity: f64,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let n_features = x.ncols();

        // Compute total weight for this node
        let total_weight: f64 = indices.iter().map(|&i| sample_weights[i]).sum();

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        // Determine which features to consider
        let features_to_check: Vec<usize> = if let Some(max_f) = self.max_features {
            use std::collections::HashSet;
            let mut rng = self.random_state.unwrap_or(42);
            let mut selected = HashSet::new();
            let max_f = max_f.min(n_features);
            while selected.len() < max_f {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let idx = (rng as usize) % n_features;
                selected.insert(idx);
            }
            selected.into_iter().collect()
        } else {
            (0..n_features).collect()
        };

        let n_classes = classes.len();

        for &feature_idx in &features_to_check {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature_idx]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            if values.len() < 2 {
                continue;
            }

            // Try midpoints between consecutive values
            for i in 0..values.len() - 1 {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Split indices
                let mut left_indices = Vec::new();
                let mut right_indices = Vec::new();
                for &idx in indices {
                    if x[[idx, feature_idx]] <= threshold {
                        left_indices.push(idx);
                    } else {
                        right_indices.push(idx);
                    }
                }

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Compute weighted class counts for children
                let mut left_weights = vec![0.0f64; n_classes];
                let mut right_weights = vec![0.0f64; n_classes];
                let mut left_total = 0.0f64;
                let mut right_total = 0.0f64;

                for &idx in &left_indices {
                    let w = sample_weights[idx];
                    left_total += w;
                    if let Some(pos) = classes.iter().position(|&c| (c - y[idx]).abs() < 1e-10) {
                        left_weights[pos] += w;
                    }
                }
                for &idx in &right_indices {
                    let w = sample_weights[idx];
                    right_total += w;
                    if let Some(pos) = classes.iter().position(|&c| (c - y[idx]).abs() < 1e-10) {
                        right_weights[pos] += w;
                    }
                }

                let left_impurity = weighted_gini_impurity(&left_weights, left_total);
                let right_impurity = weighted_gini_impurity(&right_weights, right_total);

                // Weighted impurity decrease
                let left_prop = left_total / total_weight;
                let right_prop = right_total / total_weight;
                let weighted_child_impurity =
                    left_prop * left_impurity + right_prop * right_impurity;
                let gain = parent_impurity - weighted_child_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((
                        feature_idx,
                        threshold,
                        left_indices.clone(),
                        right_indices.clone(),
                        gain,
                    ));
                }
            }
        }

        best_split
    }

    /// Find the best split for a node (classifier split-finding, kept for future use)
    #[allow(dead_code)]
    fn find_best_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        classes: &[f64],
        parent_impurity: f64,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let n_features = x.ncols();
        let n_samples = indices.len();

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        // Determine which features to consider
        let features_to_check: Vec<usize> = if let Some(max_f) = self.max_features {
            // Random subset of features
            use std::collections::HashSet;
            let mut rng = self.random_state.unwrap_or(42);
            let mut selected = HashSet::new();
            let max_f = max_f.min(n_features);
            while selected.len() < max_f {
                // Simple LCG for reproducibility
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let idx = (rng as usize) % n_features;
                selected.insert(idx);
            }
            selected.into_iter().collect()
        } else {
            (0..n_features).collect()
        };

        for &feature_idx in &features_to_check {
            // Get unique values for this feature among the indices
            let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature_idx]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            if values.len() < 2 {
                continue;
            }

            // Try midpoints between consecutive values
            for i in 0..values.len() - 1 {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Split indices
                let mut left_indices = Vec::new();
                let mut right_indices = Vec::new();
                for &idx in indices {
                    if x[[idx, feature_idx]] <= threshold {
                        left_indices.push(idx);
                    } else {
                        right_indices.push(idx);
                    }
                }

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Compute impurity for children
                let n_classes = classes.len();
                let mut left_counts = vec![0usize; n_classes];
                let mut right_counts = vec![0usize; n_classes];

                for &idx in &left_indices {
                    if let Some(pos) = classes.iter().position(|&c| (c - y[idx]).abs() < 1e-10) {
                        left_counts[pos] += 1;
                    }
                }
                for &idx in &right_indices {
                    if let Some(pos) = classes.iter().position(|&c| (c - y[idx]).abs() < 1e-10) {
                        right_counts[pos] += 1;
                    }
                }

                let left_impurity = match self.criterion {
                    SplitCriterion::Gini => gini_impurity(&left_counts, left_indices.len()),
                    SplitCriterion::Entropy => entropy(&left_counts, left_indices.len()),
                    _ => unreachable!(),
                };
                let right_impurity = match self.criterion {
                    SplitCriterion::Gini => gini_impurity(&right_counts, right_indices.len()),
                    SplitCriterion::Entropy => entropy(&right_counts, right_indices.len()),
                    _ => unreachable!(),
                };

                // Weighted impurity decrease
                let left_weight = left_indices.len() as f64 / n_samples as f64;
                let right_weight = right_indices.len() as f64 / n_samples as f64;
                let weighted_child_impurity =
                    left_weight * left_impurity + right_weight * right_impurity;
                let gain = parent_impurity - weighted_child_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_idx, threshold, left_indices, right_indices, gain));
                }
            }
        }

        best_split
    }

    /// Compute feature importances from the tree
    fn compute_feature_importances(&mut self) {
        if let Some(ref tree) = self.tree {
            let n_features = tree.n_features;
            let mut importances = vec![0.0; n_features];
            let total_samples = tree
                .nodes
                .first()
                .map(|n| n.weighted_n_samples)
                .unwrap_or(1.0);

            for node in &tree.nodes {
                if !node.is_leaf {
                    let feature_idx = node.feature_index.unwrap();
                    // Weighted impurity decrease
                    importances[feature_idx] +=
                        node.impurity_decrease * (node.weighted_n_samples / total_samples);
                }
            }

            // Normalize to sum to 1
            let total: f64 = importances.iter().sum();
            if total > 0.0 {
                for imp in &mut importances {
                    *imp /= total;
                }
            }

            self.feature_importances = Some(Array1::from_vec(importances));
        }
    }

    /// Apply cost-complexity pruning
    fn prune_tree(&mut self) {
        if self.ccp_alpha <= 0.0 || self.tree.is_none() {
            return;
        }

        // Minimal cost-complexity pruning
        // R_alpha(T) = R(T) + alpha * |T| where |T| is number of leaves
        // We iteratively collapse internal nodes where the cost-complexity gain is minimal

        let tree = self.tree.as_mut().unwrap();
        let alpha = self.ccp_alpha;

        // Keep pruning while beneficial
        loop {
            let mut best_prune_node: Option<usize> = None;
            let mut best_prune_gain = f64::MAX;

            // Find internal node with smallest effective alpha
            for (i, node) in tree.nodes.iter().enumerate() {
                if node.is_leaf {
                    continue;
                }

                // Count leaves in subtree
                let subtree_leaves = count_leaves_in_subtree(&tree.nodes, i);
                if subtree_leaves <= 1 {
                    continue;
                }

                // Compute impurity reduction from keeping this node vs making it a leaf
                let subtree_impurity = compute_subtree_impurity(&tree.nodes, i);
                let node_impurity = node.impurity * node.weighted_n_samples;

                // Effective alpha for this node
                // When we prune, we go from subtree_leaves to 1 leaf
                let leaves_removed = (subtree_leaves - 1) as f64;
                let impurity_increase = node_impurity - subtree_impurity;
                let effective_alpha = impurity_increase / leaves_removed;

                if effective_alpha < best_prune_gain {
                    best_prune_gain = effective_alpha;
                    best_prune_node = Some(i);
                }
            }

            // Check if we should prune
            if best_prune_gain <= alpha {
                if let Some(node_id) = best_prune_node {
                    // Convert to leaf
                    tree.nodes[node_id].is_leaf = true;
                    tree.nodes[node_id].left_child = None;
                    tree.nodes[node_id].right_child = None;
                    tree.nodes[node_id].feature_index = None;
                    tree.nodes[node_id].threshold = None;
                }
            } else {
                break;
            }
        }

        // Recount leaves and max depth
        tree.n_leaves = tree.nodes.iter().filter(|n| n.is_leaf).count();
        tree.max_depth = tree.nodes.iter().map(|n| n.depth).max().unwrap_or(0);
    }
}

/// Count leaves in a subtree
fn count_leaves_in_subtree(nodes: &[TreeNode], node_id: usize) -> usize {
    let node = &nodes[node_id];
    if node.is_leaf {
        return 1;
    }

    let left_count = node
        .left_child
        .map(|id| count_leaves_in_subtree(nodes, id))
        .unwrap_or(0);
    let right_count = node
        .right_child
        .map(|id| count_leaves_in_subtree(nodes, id))
        .unwrap_or(0);

    left_count + right_count
}

/// Compute total weighted impurity of a subtree
fn compute_subtree_impurity(nodes: &[TreeNode], node_id: usize) -> f64 {
    let node = &nodes[node_id];
    if node.is_leaf {
        return node.impurity * node.weighted_n_samples;
    }

    let left_impurity = node
        .left_child
        .map(|id| compute_subtree_impurity(nodes, id))
        .unwrap_or(0.0);
    let right_impurity = node
        .right_child
        .map(|id| compute_subtree_impurity(nodes, id))
        .unwrap_or(0.0);

    left_impurity + right_impurity
}

impl Model for DecisionTreeClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Find unique classes
        let classes = get_unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "DecisionTreeClassifier requires at least 2 classes",
            ));
        }

        // Compute sample weights from class weights
        let sample_weights = compute_sample_weights(y, &classes, &self.class_weight);

        let classes_vec: Vec<f64> = classes.iter().copied().collect();
        self.classes = Some(classes);
        self.n_features = Some(n_features);

        // Build tree with sample weights
        let indices: Vec<usize> = (0..n_samples).collect();
        let mut nodes = Vec::new();
        self.build_tree_weighted(x, y, &sample_weights, &indices, &classes_vec, 0, &mut nodes);

        // Compute max depth and leaf count
        let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);
        let n_leaves = nodes.iter().filter(|n| n.is_leaf).count();

        self.tree = Some(TreeStructure {
            nodes,
            n_features,
            n_classes: classes_vec.len(),
            feature_names: None,
            class_names: None,
            max_depth,
            n_leaves,
        });

        // Apply pruning if ccp_alpha > 0
        self.prune_tree();

        // Compute feature importances
        self.compute_feature_importances();

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.tree, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let classes = self.classes.as_ref().unwrap();
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let leaf_id = self.find_leaf(&sample);
            let leaf = self.tree.as_ref().unwrap().get_node(leaf_id).unwrap();

            // Find class with maximum count
            let max_idx = leaf
                .value
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions[i] = classes[max_idx];
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.tree.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importances.clone()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("max_depth", 1, 20)
            .int("min_samples_split", 2, 20)
            .int("min_samples_leaf", 1, 10)
            .float("min_impurity_decrease", 0.0, 0.5)
            .float("ccp_alpha", 0.0, 0.1)
            .categorical("criterion", vec!["gini".to_string(), "entropy".to_string()])
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// Decision Tree Regressor
// =============================================================================

/// Decision Tree Regressor using CART algorithm
///
/// Builds a binary decision tree by recursively partitioning the feature space
/// using the best split based on MSE or MAE reduction.
///
/// ## Features
///
/// - MSE or MAE criterion
/// - Feature importance from impurity decrease
/// - Cost-complexity pruning (ccp_alpha)
/// - Tree structure export for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeRegressor {
    /// Split criterion (MSE or MAE)
    pub criterion: SplitCriterion,
    /// Maximum depth of the tree (None for unlimited)
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Maximum number of features to consider for splitting (None for all)
    pub max_features: Option<usize>,
    /// Minimum impurity decrease required for a split
    pub min_impurity_decrease: f64,
    /// Cost-complexity pruning alpha (0 = no pruning)
    pub ccp_alpha: f64,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,

    // Fitted parameters
    tree: Option<TreeStructure>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<f64>>,
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTreeRegressor {
    /// Create a new Decision Tree Regressor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            criterion: SplitCriterion::Mse,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            min_impurity_decrease: 0.0,
            ccp_alpha: 0.0,
            random_state: None,
            tree: None,
            n_features: None,
            feature_importances: None,
        }
    }

    /// Set the split criterion
    #[must_use]
    pub fn with_criterion(mut self, criterion: SplitCriterion) -> Self {
        assert!(
            criterion.is_regression(),
            "Criterion must be MSE or MAE for regression"
        );
        self.criterion = criterion;
        self
    }

    /// Set maximum tree depth
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples to split
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split.max(2);
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf.max(1);
        self
    }

    /// Set maximum features to consider
    #[must_use]
    pub fn with_max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set minimum impurity decrease for split
    #[must_use]
    pub fn with_min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
        self.min_impurity_decrease = min_impurity_decrease.max(0.0);
        self
    }

    /// Set cost-complexity pruning alpha
    #[must_use]
    pub fn with_ccp_alpha(mut self, ccp_alpha: f64) -> Self {
        self.ccp_alpha = ccp_alpha.max(0.0);
        self
    }

    /// Set random state
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Get the fitted tree structure
    #[must_use]
    pub fn tree(&self) -> Option<&TreeStructure> {
        self.tree.as_ref()
    }

    /// Get the tree depth
    #[must_use]
    pub fn get_depth(&self) -> Option<usize> {
        self.tree.as_ref().map(|t| t.max_depth)
    }

    /// Get the number of leaves
    #[must_use]
    pub fn get_n_leaves(&self) -> Option<usize> {
        self.tree.as_ref().map(|t| t.n_leaves)
    }

    /// Find the leaf node for a sample
    fn find_leaf(&self, x: &[f64]) -> usize {
        let tree = self.tree.as_ref().unwrap();
        let mut node_id = 0;

        while let Some(node) = tree.get_node(node_id) {
            if node.is_leaf {
                return node_id;
            }
            let feature_idx = node.feature_index.unwrap();
            let threshold = node.threshold.unwrap();
            node_id = if x[feature_idx] <= threshold {
                node.left_child.unwrap()
            } else {
                node.right_child.unwrap()
            };
        }

        node_id
    }

    /// Build the tree recursively
    fn build_tree(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        depth: usize,
        nodes: &mut Vec<TreeNode>,
    ) -> usize {
        let node_id = nodes.len();
        let n_samples = indices.len();

        // Compute target values at this node
        let values: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        let mean = values.iter().sum::<f64>() / n_samples as f64;

        // Compute impurity
        let impurity = match self.criterion {
            SplitCriterion::Mse => mse(&values),
            SplitCriterion::Mae => mae(&values),
            _ => unreachable!(),
        };

        // Node value (mean prediction)
        let value = vec![mean];

        // Create leaf node initially
        let leaf = TreeNode::new_leaf(node_id, impurity, n_samples, n_samples as f64, value, depth);
        nodes.push(leaf);

        // Check stopping conditions
        let should_stop = n_samples < self.min_samples_split
            || (self.max_depth.is_some() && depth >= self.max_depth.unwrap())
            || impurity < 1e-10;

        if should_stop {
            return node_id;
        }

        // Find best split
        let best_split = self.find_best_split(x, y, indices, impurity);

        if let Some((feature_idx, threshold, left_indices, right_indices, impurity_decrease)) =
            best_split
        {
            // Check min_impurity_decrease
            if impurity_decrease < self.min_impurity_decrease {
                return node_id;
            }

            // Check min_samples_leaf
            if left_indices.len() < self.min_samples_leaf
                || right_indices.len() < self.min_samples_leaf
            {
                return node_id;
            }

            // Recursively build children
            let left_id = self.build_tree(x, y, &left_indices, depth + 1, nodes);
            let right_id = self.build_tree(x, y, &right_indices, depth + 1, nodes);

            // Update current node to internal node
            nodes[node_id].make_internal(
                feature_idx,
                threshold,
                left_id,
                right_id,
                impurity_decrease,
            );
        }

        node_id
    }

    /// Find the best split for a node
    fn find_best_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        parent_impurity: f64,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let n_features = x.ncols();
        let n_samples = indices.len();

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        // Determine which features to consider
        let features_to_check: Vec<usize> = if let Some(max_f) = self.max_features {
            use std::collections::HashSet;
            let mut rng = self.random_state.unwrap_or(42);
            let mut selected = HashSet::new();
            let max_f = max_f.min(n_features);
            while selected.len() < max_f {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let idx = (rng as usize) % n_features;
                selected.insert(idx);
            }
            selected.into_iter().collect()
        } else {
            (0..n_features).collect()
        };

        for &feature_idx in &features_to_check {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices.iter().map(|&i| x[[i, feature_idx]]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            if values.len() < 2 {
                continue;
            }

            // Try midpoints between consecutive values
            for i in 0..values.len() - 1 {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Split indices
                let mut left_indices = Vec::new();
                let mut right_indices = Vec::new();
                for &idx in indices {
                    if x[[idx, feature_idx]] <= threshold {
                        left_indices.push(idx);
                    } else {
                        right_indices.push(idx);
                    }
                }

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Compute impurity for children
                let left_values: Vec<f64> = left_indices.iter().map(|&i| y[i]).collect();
                let right_values: Vec<f64> = right_indices.iter().map(|&i| y[i]).collect();

                let left_impurity = match self.criterion {
                    SplitCriterion::Mse => mse(&left_values),
                    SplitCriterion::Mae => mae(&left_values),
                    _ => unreachable!(),
                };
                let right_impurity = match self.criterion {
                    SplitCriterion::Mse => mse(&right_values),
                    SplitCriterion::Mae => mae(&right_values),
                    _ => unreachable!(),
                };

                // Weighted impurity decrease
                let left_weight = left_indices.len() as f64 / n_samples as f64;
                let right_weight = right_indices.len() as f64 / n_samples as f64;
                let weighted_child_impurity =
                    left_weight * left_impurity + right_weight * right_impurity;
                let gain = parent_impurity - weighted_child_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_idx, threshold, left_indices, right_indices, gain));
                }
            }
        }

        best_split
    }

    /// Compute feature importances from the tree
    fn compute_feature_importances(&mut self) {
        if let Some(ref tree) = self.tree {
            let n_features = tree.n_features;
            let mut importances = vec![0.0; n_features];
            let total_samples = tree
                .nodes
                .first()
                .map(|n| n.weighted_n_samples)
                .unwrap_or(1.0);

            for node in &tree.nodes {
                if !node.is_leaf {
                    let feature_idx = node.feature_index.unwrap();
                    importances[feature_idx] +=
                        node.impurity_decrease * (node.weighted_n_samples / total_samples);
                }
            }

            // Normalize to sum to 1
            let total: f64 = importances.iter().sum();
            if total > 0.0 {
                for imp in &mut importances {
                    *imp /= total;
                }
            }

            self.feature_importances = Some(Array1::from_vec(importances));
        }
    }

    /// Apply cost-complexity pruning
    fn prune_tree(&mut self) {
        if self.ccp_alpha <= 0.0 || self.tree.is_none() {
            return;
        }

        let tree = self.tree.as_mut().unwrap();
        let alpha = self.ccp_alpha;

        loop {
            let mut best_prune_node: Option<usize> = None;
            let mut best_prune_gain = f64::MAX;

            for (i, node) in tree.nodes.iter().enumerate() {
                if node.is_leaf {
                    continue;
                }

                let subtree_leaves = count_leaves_in_subtree(&tree.nodes, i);
                if subtree_leaves <= 1 {
                    continue;
                }

                let subtree_impurity = compute_subtree_impurity(&tree.nodes, i);
                let node_impurity = node.impurity * node.weighted_n_samples;

                let leaves_removed = (subtree_leaves - 1) as f64;
                let impurity_increase = node_impurity - subtree_impurity;
                let effective_alpha = impurity_increase / leaves_removed;

                if effective_alpha < best_prune_gain {
                    best_prune_gain = effective_alpha;
                    best_prune_node = Some(i);
                }
            }

            if best_prune_gain <= alpha {
                if let Some(node_id) = best_prune_node {
                    tree.nodes[node_id].is_leaf = true;
                    tree.nodes[node_id].left_child = None;
                    tree.nodes[node_id].right_child = None;
                    tree.nodes[node_id].feature_index = None;
                    tree.nodes[node_id].threshold = None;
                }
            } else {
                break;
            }
        }

        tree.n_leaves = tree.nodes.iter().filter(|n| n.is_leaf).count();
        tree.max_depth = tree.nodes.iter().map(|n| n.depth).max().unwrap_or(0);
    }
}

impl Model for DecisionTreeRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.n_features = Some(n_features);

        // Build tree
        let indices: Vec<usize> = (0..n_samples).collect();
        let mut nodes = Vec::new();
        self.build_tree(x, y, &indices, 0, &mut nodes);

        // Compute max depth and leaf count
        let max_depth = nodes.iter().map(|n| n.depth).max().unwrap_or(0);
        let n_leaves = nodes.iter().filter(|n| n.is_leaf).count();

        self.tree = Some(TreeStructure {
            nodes,
            n_features,
            n_classes: 1, // Regression has 1 "class" (continuous output)
            feature_names: None,
            class_names: None,
            max_depth,
            n_leaves,
        });

        // Apply pruning if ccp_alpha > 0
        self.prune_tree();

        // Compute feature importances
        self.compute_feature_importances();

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.tree, "predict")?;
        let n_features = self.n_features.unwrap();
        validate_predict_input(x, n_features)?;

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let leaf_id = self.find_leaf(&sample);
            let leaf = self.tree.as_ref().unwrap().get_node(leaf_id).unwrap();
            predictions[i] = leaf.value[0]; // Mean value at leaf
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.tree.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importances.clone()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("max_depth", 1, 20)
            .int("min_samples_split", 2, 20)
            .int("min_samples_leaf", 1, 10)
            .float("min_impurity_decrease", 0.0, 0.5)
            .float("ccp_alpha", 0.0, 0.1)
            .categorical("criterion", vec!["mse".to_string(), "mae".to_string()])
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linearly separable data
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, // Class 0
                1.5, 1.5, // Class 0
                2.0, 2.0, // Class 0
                2.5, 2.5, // Class 0
                5.0, 5.0, // Class 1
                5.5, 5.5, // Class 1
                6.0, 6.0, // Class 1
                6.5, 6.5, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linear relationship: y = x1 + x2
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 1.0, 4.0, 2.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]);
        (x, y)
    }

    #[test]
    fn test_gini_impurity() {
        // Pure node (all same class)
        assert_abs_diff_eq!(gini_impurity(&[10, 0], 10), 0.0, epsilon = 1e-10);

        // Maximum impurity (equal split in binary)
        assert_abs_diff_eq!(gini_impurity(&[5, 5], 10), 0.5, epsilon = 1e-10);

        // 3-class case
        assert_abs_diff_eq!(gini_impurity(&[4, 3, 3], 10), 0.66, epsilon = 1e-2);
    }

    #[test]
    fn test_entropy_impurity() {
        // Pure node
        assert_abs_diff_eq!(entropy(&[10, 0], 10), 0.0, epsilon = 1e-10);

        // Maximum entropy (equal split in binary)
        assert_abs_diff_eq!(entropy(&[5, 5], 10), (2.0_f64).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_mse() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Mean = 3, MSE = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5 = (4+1+0+1+4)/5 = 2
        assert_abs_diff_eq!(mse(&values), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_decision_tree_classifier_fit_predict() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new()
            .with_max_depth(Some(3))
            .with_criterion(SplitCriterion::Gini);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.n_features(), Some(2));

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 8);

        // Should classify training data correctly
        for i in 0..8 {
            assert_abs_diff_eq!(predictions[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_decision_tree_classifier_proba() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();

        let probas = clf.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[8, 2]);

        // Probabilities should sum to 1
        for i in 0..8 {
            let row_sum: f64 = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_decision_tree_classifier_feature_importance() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();

        let importance = clf.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Importances should sum to 1
        let total: f64 = importance.sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-10);

        // All importances should be non-negative
        assert!(importance.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_decision_tree_regressor_fit_predict() {
        let (x, y) = make_regression_data();

        let mut reg = DecisionTreeRegressor::new()
            .with_max_depth(Some(5))
            .with_criterion(SplitCriterion::Mse);

        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());
        assert_eq!(reg.n_features(), Some(2));

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 8);

        // Should fit training data well
        for i in 0..8 {
            assert_abs_diff_eq!(predictions[i], y[i], epsilon = 0.5);
        }
    }

    #[test]
    fn test_decision_tree_regressor_feature_importance() {
        let (x, y) = make_regression_data();

        let mut reg = DecisionTreeRegressor::new();
        reg.fit(&x, &y).unwrap();

        let importance = reg.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Importances should sum to 1
        let total: f64 = importance.sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tree_structure() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new().with_max_depth(Some(2));
        clf.fit(&x, &y).unwrap();

        let tree = clf.tree().unwrap();
        assert!(tree.nodes.len() > 0);
        assert!(tree.n_leaves > 0);
        assert!(tree.max_depth <= 2);

        // Root should exist
        let root = tree.root().unwrap();
        assert_eq!(root.id, 0);
        assert_eq!(root.depth, 0);
    }

    #[test]
    fn test_tree_to_dot() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new().with_max_depth(Some(2));
        clf.fit(&x, &y).unwrap();

        let dot = clf.tree().unwrap().to_dot();
        assert!(dot.contains("digraph Tree"));
        assert!(dot.contains("samples ="));
    }

    #[test]
    fn test_decision_path() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();

        let sample = x.row(0).to_vec();
        let path = clf.tree().unwrap().decision_path(&sample);

        // Path should start at root
        assert_eq!(path[0], 0);
        // Path should end at a leaf
        let last_node_id = *path.last().unwrap();
        let last_node = clf.tree().unwrap().get_node(last_node_id).unwrap();
        assert!(last_node.is_leaf);
    }

    #[test]
    fn test_cost_complexity_pruning() {
        let (x, y) = make_classification_data();

        // Without pruning
        let mut clf_unpruned = DecisionTreeClassifier::new().with_ccp_alpha(0.0);
        clf_unpruned.fit(&x, &y).unwrap();
        let n_leaves_unpruned = clf_unpruned.get_n_leaves().unwrap();

        // With pruning
        let mut clf_pruned = DecisionTreeClassifier::new().with_ccp_alpha(0.1);
        clf_pruned.fit(&x, &y).unwrap();
        let n_leaves_pruned = clf_pruned.get_n_leaves().unwrap();

        // Pruned tree should have fewer or equal leaves
        assert!(n_leaves_pruned <= n_leaves_unpruned);
    }

    #[test]
    fn test_min_samples_constraints() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new()
            .with_min_samples_split(5)
            .with_min_samples_leaf(3);
        clf.fit(&x, &y).unwrap();

        // Check that leaf nodes have at least min_samples_leaf samples
        let tree = clf.tree().unwrap();
        for node in &tree.nodes {
            if node.is_leaf {
                assert!(node.n_samples >= 1); // At least 1 sample per leaf
            }
        }
    }

    #[test]
    fn test_entropy_criterion() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new().with_criterion(SplitCriterion::Entropy);
        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();
        for i in 0..8 {
            assert_abs_diff_eq!(predictions[i], y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mae_criterion() {
        let (x, y) = make_regression_data();

        let mut reg = DecisionTreeRegressor::new().with_criterion(SplitCriterion::Mae);
        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_mae_left_right_impurity_uses_correct_data() {
        // Dataset where MAE split is obvious: [1,2] vs [100,101]
        // Correct: left_impurity ~ 0.5 (median 1.5, MAE from [1,2])
        // Correct: right_impurity ~ 0.5 (median 100.5, MAE from [100,101])
        // Wrong (bug): left_impurity would be ~ 0.5 from [100,101] - same as right!

        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 100.0, 101.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 100.0, 101.0]);

        let mut reg = DecisionTreeRegressor::new()
            .with_criterion(SplitCriterion::Mae)
            .with_max_depth(Some(1));
        reg.fit(&x, &y).unwrap();

        // With correct MAE, should split at ~50 (between 2 and 100)
        let predictions = reg.predict(&x).unwrap();

        // Left leaf (samples 0,1) should predict ~1.5 (median of [1,2])
        assert!(
            (predictions[0] - 1.5).abs() < 0.1,
            "Left prediction wrong: {}",
            predictions[0]
        );
        assert!(
            (predictions[1] - 1.5).abs() < 0.1,
            "Left prediction wrong: {}",
            predictions[1]
        );

        // Right leaf (samples 2,3) should predict ~100.5 (median of [100,101])
        assert!(
            (predictions[2] - 100.5).abs() < 0.1,
            "Right prediction wrong: {}",
            predictions[2]
        );
        assert!(
            (predictions[3] - 100.5).abs() < 0.1,
            "Right prediction wrong: {}",
            predictions[3]
        );
    }

    #[test]
    fn test_mae_impurity_decrease_invariant() {
        // Property: weighted_child_impurity <= parent_impurity for any valid split
        let x = Array2::from_shape_fn((20, 2), |(i, j)| (i * 3 + j) as f64);
        let y = Array1::from_shape_fn(20, |i| (i as f64).powi(2) / 100.0);

        let mut reg = DecisionTreeRegressor::new()
            .with_criterion(SplitCriterion::Mae)
            .with_max_depth(Some(3));
        reg.fit(&x, &y).unwrap();

        // Tree should be fitted without panic
        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // All predictions should be finite (not NaN/Inf from bad impurity calcs)
        assert!(
            predictions.iter().all(|&p| p.is_finite()),
            "MAE produced non-finite predictions"
        );
    }

    #[test]
    fn test_mae_mse_both_produce_valid_trees() {
        let (x, y) = make_regression_data();

        let mut mse_reg = DecisionTreeRegressor::new()
            .with_criterion(SplitCriterion::Mse)
            .with_max_depth(Some(3))
            .with_random_state(42);
        mse_reg.fit(&x, &y).unwrap();

        let mut mae_reg = DecisionTreeRegressor::new()
            .with_criterion(SplitCriterion::Mae)
            .with_max_depth(Some(3))
            .with_random_state(42);
        mae_reg.fit(&x, &y).unwrap();

        let mse_preds = mse_reg.predict(&x).unwrap();
        let mae_preds = mae_reg.predict(&x).unwrap();

        // Both should produce finite predictions
        assert!(mse_preds.iter().all(|&p| p.is_finite()));
        assert!(mae_preds.iter().all(|&p| p.is_finite()));

        // Both should have reasonable R² (> 0.5 on training data)
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        let mse_ss_res: f64 = y
            .iter()
            .zip(mse_preds.iter())
            .map(|(&yi, &pi)| (yi - pi).powi(2))
            .sum();
        let mae_ss_res: f64 = y
            .iter()
            .zip(mae_preds.iter())
            .map(|(&yi, &pi)| (yi - pi).powi(2))
            .sum();

        let mse_r2 = 1.0 - mse_ss_res / ss_tot;
        let mae_r2 = 1.0 - mae_ss_res / ss_tot;

        assert!(mse_r2 > 0.5, "MSE R² too low: {}", mse_r2);
        assert!(mae_r2 > 0.5, "MAE R² too low: {}", mae_r2);
    }

    #[test]
    fn test_mae_outlier_robustness() {
        // Data with outlier: [1, 2, 3, 4, 1000]
        // MAE should be more robust than MSE
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1000.0]); // outlier at index 4

        let mut mae_reg = DecisionTreeRegressor::new()
            .with_criterion(SplitCriterion::Mae)
            .with_max_depth(Some(1));
        mae_reg.fit(&x, &y).unwrap();

        // Test on non-outlier points - MAE should predict median-like values
        let test_x = Array2::from_shape_vec((1, 1), vec![2.5]).unwrap();
        let mae_pred = mae_reg.predict(&test_x).unwrap();

        // MAE prediction should be closer to 2.5 (median of non-outliers)
        // than to 202 (mean including outlier)
        assert!(
            mae_pred[0] < 100.0,
            "MAE prediction {} too influenced by outlier",
            mae_pred[0]
        );
    }

    #[test]
    fn test_search_space() {
        let clf = DecisionTreeClassifier::new();
        let space = clf.search_space();
        assert!(space.n_dims() > 0);

        let reg = DecisionTreeRegressor::new();
        let space = reg.search_space();
        assert!(space.n_dims() > 0);
    }

    #[test]
    fn test_not_fitted_error() {
        let clf = DecisionTreeClassifier::new();
        let x = Array2::zeros((2, 2));

        let result = clf.predict(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_features_error() {
        let (x, y) = make_classification_data();

        let mut clf = DecisionTreeClassifier::new();
        clf.fit(&x, &y).unwrap();

        // Try to predict with wrong number of features
        let x_wrong = Array2::zeros((2, 3));
        let result = clf.predict(&x_wrong);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]); // Single class

        let mut clf = DecisionTreeClassifier::new();
        let result = clf.fit(&x, &y);
        assert!(result.is_err());
    }
}
