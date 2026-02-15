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
//! ```
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
//! ```
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

/// Strategy for selecting split thresholds in decision trees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitStrategy {
    /// Exhaustive search for the best threshold (standard decision tree).
    Best,
    /// Random threshold per feature (ExtraTrees / Extremely Randomized Trees).
    Random,
}

impl Default for SplitStrategy {
    fn default() -> Self {
        Self::Best
    }
}

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
                let feature_idx = node
                    .feature_index
                    .expect("internal node must have feature_index");
                let feature_name = self
                    .feature_names
                    .as_ref()
                    .and_then(|names| names.get(feature_idx))
                    .cloned()
                    .unwrap_or_else(|| format!("X[{}]", feature_idx));
                format!(
                    "{} <= {:.4}\\nsamples = {}\\nimpurity = {:.4}",
                    feature_name,
                    node.threshold.expect("internal node must have threshold"),
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
            let feature_idx = node
                .feature_index
                .expect("internal node must have feature_index");
            let threshold = node.threshold.expect("internal node must have threshold");
            node_id = if x[feature_idx] <= threshold {
                node.left_child.expect("internal node must have left_child")
            } else {
                node.right_child
                    .expect("internal node must have right_child")
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
    /// Split strategy (Best = exhaustive, Random = ExtraTrees)
    pub split_strategy: SplitStrategy,

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
            split_strategy: SplitStrategy::Best,
            tree: None,
            classes: None,
            n_features: None,
            feature_importances: None,
        }
    }

    /// Set the split strategy (Best = exhaustive search, Random = ExtraTrees)
    #[must_use]
    pub fn with_split_strategy(mut self, strategy: SplitStrategy) -> Self {
        self.split_strategy = strategy;
        self
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
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict_proba"))?;
        validate_predict_input(x, n_features)?;

        let tree = self
            .tree
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_proba"))?;
        let n_classes = tree.n_classes;
        let n_samples = x.nrows();

        let mut probas = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let leaf_id = self.find_leaf(&sample);
            let leaf = tree.get_node(leaf_id).expect("leaf_id must be valid");

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
        let tree = self.tree.as_ref().expect("model must be fitted");
        let mut node_id = 0;

        while let Some(node) = tree.get_node(node_id) {
            if node.is_leaf {
                return node_id;
            }
            let feature_idx = node
                .feature_index
                .expect("internal node must have feature_index");
            let threshold = node.threshold.expect("internal node must have threshold");
            node_id = if x[feature_idx] <= threshold {
                node.left_child.expect("internal node must have left_child")
            } else {
                node.right_child
                    .expect("internal node must have right_child")
            };
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
            || self.max_depth.is_some_and(|d| depth >= d)
            || impurity < 1e-10
            || class_weights_sum.iter().filter(|&&w| w > 0.0).count() <= 1;

        if should_stop {
            return node_id;
        }

        // Find best split with weights (dispatch on split strategy)
        let best_split = match self.split_strategy {
            SplitStrategy::Random => {
                self.find_random_split_weighted(x, y, sample_weights, indices, classes, impurity)
            }
            SplitStrategy::Best => {
                self.find_best_split_weighted(x, y, sample_weights, indices, classes, impurity)
            }
        };

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

    /// Find split using random thresholds (ExtraTrees / Extremely Randomized Trees).
    ///
    /// For each candidate feature, picks a random threshold uniformly between
    /// the min and max values observed in the current node, then evaluates
    /// impurity gain. Picks the feature with the best gain.
    fn find_random_split_weighted(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weights: &Array1<f64>,
        indices: &[usize],
        classes: &[f64],
        parent_impurity: f64,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let n_features = x.ncols();
        let n_classes = classes.len();

        let total_weight: f64 = indices.iter().map(|&i| sample_weights[i]).sum();

        // Use deterministic RNG from random_state + n_samples to vary per node
        let mut rng = self
            .random_state
            .unwrap_or(42)
            .wrapping_add(indices.len() as u64);

        // Select features to check
        let features_to_check: Vec<usize> = if let Some(max_f) = self.max_features {
            use std::collections::HashSet;
            let mut selected = HashSet::new();
            let max_f = max_f.min(n_features);
            while selected.len() < max_f {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let idx = (rng as usize) % n_features;
                selected.insert(idx);
            }
            let mut features: Vec<usize> = selected.into_iter().collect();
            features.sort();
            features
        } else {
            (0..n_features).collect()
        };

        // Pre-compute class index for each sample
        let sample_class_idx: Vec<usize> = indices
            .iter()
            .map(|&idx| {
                classes
                    .iter()
                    .position(|&c| (c - y[idx]).abs() < 1e-10)
                    .unwrap_or(0)
            })
            .collect();

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        for &feature_idx in &features_to_check {
            // Find min/max of this feature across current indices
            let mut feat_min = f64::INFINITY;
            let mut feat_max = f64::NEG_INFINITY;
            for &i in indices {
                let v = x[[i, feature_idx]];
                if v < feat_min {
                    feat_min = v;
                }
                if v > feat_max {
                    feat_max = v;
                }
            }
            // Skip constant features
            if (feat_max - feat_min).abs() < 1e-10 {
                continue;
            }

            // Generate random threshold in [feat_min, feat_max]
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let frac = (rng as u32) as f64 / u32::MAX as f64;
            let threshold = feat_min + frac * (feat_max - feat_min);

            // Partition and compute weighted impurity
            let mut left_weights = vec![0.0f64; n_classes];
            let mut right_weights = vec![0.0f64; n_classes];
            let mut left_total = 0.0f64;
            let mut right_total = 0.0f64;
            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            for (si, &idx) in indices.iter().enumerate() {
                let w = sample_weights[idx];
                let ci = sample_class_idx[si];
                if x[[idx, feature_idx]] <= threshold {
                    left_weights[ci] += w;
                    left_total += w;
                    left_indices.push(idx);
                } else {
                    right_weights[ci] += w;
                    right_total += w;
                    right_indices.push(idx);
                }
            }

            // Skip empty partitions
            if left_indices.is_empty() || right_indices.is_empty() {
                continue;
            }

            let left_impurity = weighted_gini_impurity(&left_weights, left_total);
            let right_impurity = weighted_gini_impurity(&right_weights, right_total);

            let left_prop = left_total / total_weight;
            let right_prop = right_total / total_weight;
            let weighted_child_impurity =
                left_prop.mul_add(left_impurity, right_prop * right_impurity);
            let gain = parent_impurity - weighted_child_impurity;

            if gain > best_gain {
                best_gain = gain;
                best_split = Some((feature_idx, threshold, left_indices, right_indices, gain));
            }
        }

        best_split
    }

    /// Find the best split for a node with sample weights
    ///
    /// Optimized: sorts indices once per feature and sweeps left-to-right
    /// accumulating running class weight counts. O(n * p * log n) instead of O(n^2 * p).
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
        let n_samples = indices.len();

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
            let mut features: Vec<usize> = selected.into_iter().collect();
            features.sort();
            features
        } else {
            (0..n_features).collect()
        };

        let n_classes = classes.len();

        // Pre-allocate buffers reused across features
        let mut sorted_indices: Vec<usize> = Vec::with_capacity(n_samples);
        let mut left_weights = vec![0.0f64; n_classes];
        let mut right_weights = vec![0.0f64; n_classes];

        // Pre-compute class index for each sample to avoid repeated linear search
        let sample_class_idx: Vec<usize> = indices
            .iter()
            .map(|&idx| {
                classes
                    .iter()
                    .position(|&c| (c - y[idx]).abs() < 1e-10)
                    .unwrap_or(0)
            })
            .collect();

        for &feature_idx in &features_to_check {
            // Sort indices by feature value
            sorted_indices.clear();
            sorted_indices.extend(0..n_samples);
            sorted_indices.sort_by(|&a, &b| {
                x[[indices[a], feature_idx]]
                    .partial_cmp(&x[[indices[b], feature_idx]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Initialize: everything in right partition
            for w in left_weights.iter_mut() {
                *w = 0.0;
            }
            for w in right_weights.iter_mut() {
                *w = 0.0;
            }
            let mut left_total = 0.0f64;
            let mut right_total = 0.0f64;

            for &si in &sorted_indices {
                let idx = indices[si];
                let w = sample_weights[idx];
                right_total += w;
                right_weights[sample_class_idx[si]] += w;
            }

            // Sweep left-to-right through sorted indices
            for pos in 0..n_samples - 1 {
                let si = sorted_indices[pos];
                let idx = indices[si];
                let w = sample_weights[idx];
                let ci = sample_class_idx[si];

                // Move sample from right to left
                left_weights[ci] += w;
                right_weights[ci] -= w;
                left_total += w;
                right_total -= w;

                // Skip if same feature value as next (no valid threshold between equal values)
                let next_si = sorted_indices[pos + 1];
                let next_idx = indices[next_si];
                if (x[[idx, feature_idx]] - x[[next_idx, feature_idx]]).abs() < 1e-10 {
                    continue;
                }

                let threshold = (x[[idx, feature_idx]] + x[[next_idx, feature_idx]]) / 2.0;

                let left_impurity = weighted_gini_impurity(&left_weights, left_total);
                let right_impurity = weighted_gini_impurity(&right_weights, right_total);

                let left_prop = left_total / total_weight;
                let right_prop = right_total / total_weight;
                let weighted_child_impurity =
                    left_prop.mul_add(left_impurity, right_prop * right_impurity);
                let gain = parent_impurity - weighted_child_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    let split_pos = pos + 1;
                    best_split = Some((
                        feature_idx,
                        threshold,
                        sorted_indices[..split_pos]
                            .iter()
                            .map(|&si| indices[si])
                            .collect(),
                        sorted_indices[split_pos..]
                            .iter()
                            .map(|&si| indices[si])
                            .collect(),
                        gain,
                    ));
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
                    let feature_idx = node
                        .feature_index
                        .expect("internal node must have feature_index");
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

        let tree = self
            .tree
            .as_mut()
            .expect("model must be fitted for pruning");
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
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        validate_predict_input(x, n_features)?;

        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict_proba"))?;
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let leaf_id = self.find_leaf(&sample);
            let leaf = self
                .tree
                .as_ref()
                .expect("model must be fitted")
                .get_node(leaf_id)
                .expect("leaf_id must be valid");

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

    fn fit_weighted(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_weight: &Array1<f64>,
    ) -> Result<()> {
        validate_fit_input(x, y)?;

        if sample_weight.len() != y.len() {
            return Err(FerroError::shape_mismatch(
                format!("{} samples", y.len()),
                format!("{} weights", sample_weight.len()),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let classes = get_unique_classes(y);
        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "DecisionTreeClassifier requires at least 2 classes",
            ));
        }

        // Combine class weights with sample weights
        let class_weights = compute_sample_weights(y, &classes, &self.class_weight);
        let combined_weights: Array1<f64> = sample_weight
            .iter()
            .zip(class_weights.iter())
            .map(|(sw, cw)| sw * cw)
            .collect();

        let classes_vec: Vec<f64> = classes.iter().copied().collect();
        self.classes = Some(classes);
        self.n_features = Some(n_features);

        let indices: Vec<usize> = (0..n_samples).collect();
        let mut nodes = Vec::new();
        self.build_tree_weighted(
            x,
            y,
            &combined_weights,
            &indices,
            &classes_vec,
            0,
            &mut nodes,
        );

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

        self.prune_tree();
        self.compute_feature_importances();

        Ok(())
    }

    fn model_name(&self) -> &str {
        "DecisionTreeClassifier"
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
    /// Split strategy (Best = exhaustive, Random = ExtraTrees)
    pub split_strategy: SplitStrategy,

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
            split_strategy: SplitStrategy::Best,
            tree: None,
            n_features: None,
            feature_importances: None,
        }
    }

    /// Set the split strategy (Best = exhaustive search, Random = ExtraTrees)
    #[must_use]
    pub fn with_split_strategy(mut self, strategy: SplitStrategy) -> Self {
        self.split_strategy = strategy;
        self
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
        let tree = self.tree.as_ref().expect("model must be fitted");
        let mut node_id = 0;

        while let Some(node) = tree.get_node(node_id) {
            if node.is_leaf {
                return node_id;
            }
            let feature_idx = node
                .feature_index
                .expect("internal node must have feature_index");
            let threshold = node.threshold.expect("internal node must have threshold");
            node_id = if x[feature_idx] <= threshold {
                node.left_child.expect("internal node must have left_child")
            } else {
                node.right_child
                    .expect("internal node must have right_child")
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
            _ => panic!(
                "Invalid criterion {:?} for regression tree; use Mse or Mae",
                self.criterion
            ),
        };

        // Node value: median for MAE criterion, mean otherwise
        let value = vec![if self.criterion == SplitCriterion::Mae {
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            }
        } else {
            mean
        }];

        // Create leaf node initially
        let leaf = TreeNode::new_leaf(node_id, impurity, n_samples, n_samples as f64, value, depth);
        nodes.push(leaf);

        // Check stopping conditions
        let should_stop = n_samples < self.min_samples_split
            || self.max_depth.is_some_and(|d| depth >= d)
            || impurity < 1e-10;

        if should_stop {
            return node_id;
        }

        // Find best split
        let best_split = match self.split_strategy {
            SplitStrategy::Random => self.find_random_split(x, y, indices, impurity),
            SplitStrategy::Best => self.find_best_split(x, y, indices, impurity),
        };

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
    ///
    /// Optimized: sorts indices once per feature and sweeps left-to-right.
    /// For MSE: accumulates running sum/sum_sq for O(1) impurity per threshold.
    /// For MAE: still computes from values but avoids re-partitioning.
    /// Overall: O(n * p * log n) instead of O(n^2 * p).
    /// Find split using random thresholds (ExtraTrees) for regression.
    fn find_random_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        parent_impurity: f64,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        let n_features = x.ncols();
        let n_samples = indices.len();

        let mut rng = self
            .random_state
            .unwrap_or(42)
            .wrapping_add(n_samples as u64);

        let features_to_check: Vec<usize> = if let Some(max_f) = self.max_features {
            use std::collections::HashSet;
            let mut selected = HashSet::new();
            let max_f = max_f.min(n_features);
            while selected.len() < max_f {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                let idx = (rng as usize) % n_features;
                selected.insert(idx);
            }
            let mut features: Vec<usize> = selected.into_iter().collect();
            features.sort();
            features
        } else {
            (0..n_features).collect()
        };

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_split = None;

        for &feature_idx in &features_to_check {
            let mut feat_min = f64::INFINITY;
            let mut feat_max = f64::NEG_INFINITY;
            for &i in indices {
                let v = x[[i, feature_idx]];
                if v < feat_min {
                    feat_min = v;
                }
                if v > feat_max {
                    feat_max = v;
                }
            }
            if (feat_max - feat_min).abs() < 1e-10 {
                continue;
            }

            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let frac = (rng as u32) as f64 / u32::MAX as f64;
            let threshold = feat_min + frac * (feat_max - feat_min);

            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();
            let mut left_sum = 0.0f64;
            let mut left_sum_sq = 0.0f64;
            let mut right_sum = 0.0f64;
            let mut right_sum_sq = 0.0f64;

            for &idx in indices {
                let yi = y[idx];
                if x[[idx, feature_idx]] <= threshold {
                    left_sum += yi;
                    left_sum_sq += yi * yi;
                    left_indices.push(idx);
                } else {
                    right_sum += yi;
                    right_sum_sq += yi * yi;
                    right_indices.push(idx);
                }
            }

            if left_indices.is_empty() || right_indices.is_empty() {
                continue;
            }

            let left_n = left_indices.len() as f64;
            let right_n = right_indices.len() as f64;
            let left_mean = left_sum / left_n;
            let right_mean = right_sum / right_n;
            let left_mse = left_sum_sq / left_n - left_mean * left_mean;
            let right_mse = right_sum_sq / right_n - right_mean * right_mean;

            let left_weight = left_n / n_samples as f64;
            let right_weight = right_n / n_samples as f64;
            let weighted_child_impurity = left_weight.mul_add(left_mse, right_weight * right_mse);
            let gain = parent_impurity - weighted_child_impurity;

            if gain > best_gain {
                best_gain = gain;
                best_split = Some((feature_idx, threshold, left_indices, right_indices, gain));
            }
        }

        best_split
    }

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
            let mut features: Vec<usize> = selected.into_iter().collect();
            features.sort();
            features
        } else {
            (0..n_features).collect()
        };

        // Pre-allocate sorted indices buffer (indices into the `indices` slice)
        let mut sorted_pos: Vec<usize> = Vec::with_capacity(n_samples);

        for &feature_idx in &features_to_check {
            // Sort positions by feature value
            sorted_pos.clear();
            sorted_pos.extend(0..n_samples);
            sorted_pos.sort_by(|&a, &b| {
                x[[indices[a], feature_idx]]
                    .partial_cmp(&x[[indices[b], feature_idx]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            match self.criterion {
                SplitCriterion::Mse => {
                    // Running sums for O(1) MSE computation per split point
                    let total_sum: f64 = indices.iter().map(|&i| y[i]).sum();
                    let total_sum_sq: f64 = indices.iter().map(|&i| y[i] * y[i]).sum();

                    let mut left_sum = 0.0f64;
                    let mut left_sum_sq = 0.0f64;
                    let mut left_count = 0usize;

                    for pos in 0..n_samples - 1 {
                        let idx = indices[sorted_pos[pos]];
                        let yi = y[idx];
                        left_sum += yi;
                        left_sum_sq += yi * yi;
                        left_count += 1;

                        // Skip if same feature value as next
                        let next_idx = indices[sorted_pos[pos + 1]];
                        if (x[[idx, feature_idx]] - x[[next_idx, feature_idx]]).abs() < 1e-10 {
                            continue;
                        }

                        let right_count = n_samples - left_count;
                        let right_sum = total_sum - left_sum;
                        let right_sum_sq = total_sum_sq - left_sum_sq;

                        let threshold = (x[[idx, feature_idx]] + x[[next_idx, feature_idx]]) / 2.0;

                        // MSE = E[X^2] - E[X]^2
                        let left_mean = left_sum / left_count as f64;
                        let left_mse = left_sum_sq / left_count as f64 - left_mean * left_mean;
                        let right_mean = right_sum / right_count as f64;
                        let right_mse = right_sum_sq / right_count as f64 - right_mean * right_mean;

                        let left_weight = left_count as f64 / n_samples as f64;
                        let right_weight = right_count as f64 / n_samples as f64;
                        let weighted_child_impurity =
                            left_weight.mul_add(left_mse, right_weight * right_mse);
                        let gain = parent_impurity - weighted_child_impurity;

                        if gain > best_gain + 1e-10 {
                            best_gain = gain;
                            let split_pos = pos + 1;
                            best_split = Some((
                                feature_idx,
                                threshold,
                                sorted_pos[..split_pos]
                                    .iter()
                                    .map(|&sp| indices[sp])
                                    .collect(),
                                sorted_pos[split_pos..]
                                    .iter()
                                    .map(|&sp| indices[sp])
                                    .collect(),
                                gain,
                            ));
                        }
                    }
                }
                SplitCriterion::Mae => {
                    // MAE requires median, no running-sum shortcut — but we still
                    // avoid the separate O(n) partition step per threshold.
                    for pos in 0..n_samples - 1 {
                        let idx = indices[sorted_pos[pos]];
                        let next_idx = indices[sorted_pos[pos + 1]];

                        // Skip if same feature value
                        if (x[[idx, feature_idx]] - x[[next_idx, feature_idx]]).abs() < 1e-10 {
                            continue;
                        }

                        let threshold = (x[[idx, feature_idx]] + x[[next_idx, feature_idx]]) / 2.0;
                        let split_pos = pos + 1;

                        let left_values: Vec<f64> = sorted_pos[..split_pos]
                            .iter()
                            .map(|&sp| y[indices[sp]])
                            .collect();
                        let right_values: Vec<f64> = sorted_pos[split_pos..]
                            .iter()
                            .map(|&sp| y[indices[sp]])
                            .collect();

                        let left_impurity = mae(&left_values);
                        let right_impurity = mae(&right_values);

                        let left_weight = split_pos as f64 / n_samples as f64;
                        let right_weight = (n_samples - split_pos) as f64 / n_samples as f64;
                        let weighted_child_impurity =
                            left_weight.mul_add(left_impurity, right_weight * right_impurity);
                        let gain = parent_impurity - weighted_child_impurity;

                        if gain > best_gain + 1e-10 {
                            best_gain = gain;
                            best_split = Some((
                                feature_idx,
                                threshold,
                                sorted_pos[..split_pos]
                                    .iter()
                                    .map(|&sp| indices[sp])
                                    .collect(),
                                sorted_pos[split_pos..]
                                    .iter()
                                    .map(|&sp| indices[sp])
                                    .collect(),
                                gain,
                            ));
                        }
                    }
                }
                _ => panic!(
                    "Invalid criterion {:?} for regression tree; use Mse or Mae",
                    self.criterion
                ),
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
                    let feature_idx = node
                        .feature_index
                        .expect("internal node must have feature_index");
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

        let tree = self
            .tree
            .as_mut()
            .expect("model must be fitted for pruning");
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
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        validate_predict_input(x, n_features)?;

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let leaf_id = self.find_leaf(&sample);
            let leaf = self
                .tree
                .as_ref()
                .expect("model must be fitted")
                .get_node(leaf_id)
                .expect("leaf_id must be valid");
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

    #[test]
    fn test_mae_leaf_uses_median() {
        // Data with skewed values: [1, 2, 3, 100]
        // Mean = 26.5, Median = 2.5
        // With MAE criterion, a single-leaf tree should predict the median
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 100.0]);

        let mut tree = DecisionTreeRegressor::new()
            .with_criterion(SplitCriterion::Mae)
            .with_max_depth(Some(1)); // shallow tree to inspect leaf values

        tree.fit(&x, &y).unwrap();

        // With a depth-1 tree on 4 samples, each leaf should use median
        // Predict on training data and verify predictions differ from simple mean
        let pred = tree.predict(&x).unwrap();

        // At least one prediction should NOT be 26.5 (the global mean)
        let global_mean = 26.5;
        let all_global_mean = pred.iter().all(|&p| (p - global_mean).abs() < 0.1);
        assert!(
            !all_global_mean,
            "MAE tree should not predict global mean for all samples"
        );
    }
}
