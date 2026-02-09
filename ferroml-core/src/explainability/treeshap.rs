//! TreeSHAP Implementation for Tree-Based Models
//!
//! This module implements the TreeSHAP algorithm for computing exact Shapley values
//! for tree-based models. TreeSHAP provides consistent and locally accurate
//! feature attributions by computing the exact Shapley values.
//!
//! ## Algorithm
//!
//! TreeSHAP computes exact Shapley values in O(TLD²) time where:
//! - T = number of trees
//! - L = maximum number of leaves
//! - D = maximum depth
//!
//! ## Supported Models
//!
//! - `DecisionTreeClassifier`, `DecisionTreeRegressor`
//! - `RandomForestClassifier`, `RandomForestRegressor`
//! - `GradientBoostingClassifier`, `GradientBoostingRegressor`
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::explainability::{TreeExplainer, SHAPResult};
//! use ferroml_core::models::RandomForestClassifier;
//!
//! let mut model = RandomForestClassifier::new();
//! model.fit(&x_train, &y_train)?;
//!
//! // Create TreeSHAP explainer
//! let explainer = TreeExplainer::from_random_forest_classifier(&model)?;
//!
//! // Explain a single prediction
//! let result = explainer.explain(&x_test.row(0).to_vec())?;
//! println!("Base value: {}", result.base_value);
//! println!("SHAP values: {:?}", result.shap_values);
//!
//! // Explain multiple predictions
//! let results = explainer.explain_batch(&x_test)?;
//! ```
//!
//! ## References
//!
//! - Lundberg, S. M., Erion, G. G., & Lee, S. I. (2018). Consistent Individualized
//!   Feature Attribution for Tree Ensembles.
//! - Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting
//!   Model Predictions. NeurIPS 2017.

use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor, TreeStructure};
use crate::models::Model;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// =============================================================================
// Result Types
// =============================================================================

/// SHAP values for a single prediction
///
/// Contains the base value (expected prediction) and SHAP values for each feature.
/// The prediction can be reconstructed as: base_value + sum(shap_values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHAPResult {
    /// Base value (expected model output over the training set)
    pub base_value: f64,
    /// SHAP values for each feature
    pub shap_values: Array1<f64>,
    /// Feature values for this sample (for reference)
    pub feature_values: Array1<f64>,
    /// Number of features
    pub n_features: usize,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
}

impl SHAPResult {
    /// Get features sorted by absolute SHAP value (highest impact first)
    #[must_use]
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.shap_values.len()).collect();
        indices.sort_by(|&a, &b| {
            self.shap_values[b]
                .abs()
                .partial_cmp(&self.shap_values[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Get the top k features by absolute SHAP value
    #[must_use]
    pub fn top_k(&self, k: usize) -> Vec<usize> {
        self.sorted_indices().into_iter().take(k).collect()
    }

    /// Reconstruct the prediction from base value and SHAP values
    #[must_use]
    pub fn prediction(&self) -> f64 {
        self.base_value + self.shap_values.sum()
    }

    /// Format a single feature's contribution
    #[must_use]
    pub fn format_feature(&self, feature_idx: usize) -> String {
        if feature_idx >= self.n_features {
            return String::from("Invalid feature index");
        }

        let name = self
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_idx).cloned())
            .unwrap_or_else(|| format!("feature_{}", feature_idx));

        let shap = self.shap_values[feature_idx];
        let value = self.feature_values[feature_idx];
        let sign = if shap >= 0.0 { "+" } else { "" };

        format!("{} = {:.4} -> {}{:.4}", name, value, sign, shap)
    }

    /// Create a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = vec![
            String::from("SHAP Explanation"),
            String::from("================"),
            format!("Base value: {:.4}", self.base_value),
            format!("Prediction: {:.4}", self.prediction()),
            String::new(),
            String::from("Feature contributions (sorted by impact):"),
            String::from("-----------------------------------------"),
        ];

        for idx in self.sorted_indices() {
            lines.push(self.format_feature(idx));
        }

        lines.join("\n")
    }
}

impl std::fmt::Display for SHAPResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// SHAP values for multiple predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHAPBatchResult {
    /// Base value (same for all samples)
    pub base_value: f64,
    /// SHAP values matrix (n_samples, n_features)
    pub shap_values: Array2<f64>,
    /// Feature values matrix (n_samples, n_features)
    pub feature_values: Array2<f64>,
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
}

impl SHAPBatchResult {
    /// Get SHAP result for a specific sample
    #[must_use]
    pub fn get_sample(&self, idx: usize) -> Option<SHAPResult> {
        if idx >= self.n_samples {
            return None;
        }

        Some(SHAPResult {
            base_value: self.base_value,
            shap_values: self.shap_values.row(idx).to_owned(),
            feature_values: self.feature_values.row(idx).to_owned(),
            n_features: self.n_features,
            feature_names: self.feature_names.clone(),
        })
    }

    /// Compute mean absolute SHAP value per feature (global importance)
    #[must_use]
    pub fn mean_abs_shap(&self) -> Array1<f64> {
        let mut result = Array1::zeros(self.n_features);
        for j in 0..self.n_features {
            let sum: f64 = (0..self.n_samples)
                .map(|i| self.shap_values[[i, j]].abs())
                .sum();
            result[j] = sum / self.n_samples as f64;
        }
        result
    }

    /// Get feature indices sorted by mean absolute SHAP value (global importance)
    #[must_use]
    pub fn global_importance_sorted(&self) -> Vec<usize> {
        let mean_abs = self.mean_abs_shap();
        let mut indices: Vec<usize> = (0..self.n_features).collect();
        indices.sort_by(|&a, &b| {
            mean_abs[b]
                .partial_cmp(&mean_abs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Summary statistics
    #[must_use]
    pub fn summary(&self) -> String {
        let mean_abs = self.mean_abs_shap();
        let sorted = self.global_importance_sorted();

        let mut lines = vec![
            String::from("SHAP Batch Summary"),
            String::from("=================="),
            format!("Base value: {:.4}", self.base_value),
            format!("Samples: {}", self.n_samples),
            format!("Features: {}", self.n_features),
            String::new(),
            String::from("Global feature importance (mean |SHAP|):"),
            String::from("----------------------------------------"),
        ];

        for idx in sorted {
            let name = self
                .feature_names
                .as_ref()
                .and_then(|names| names.get(idx).cloned())
                .unwrap_or_else(|| format!("feature_{}", idx));
            lines.push(format!("{}: {:.4}", name, mean_abs[idx]));
        }

        lines.join("\n")
    }
}

impl std::fmt::Display for SHAPBatchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// =============================================================================
// Internal Tree Representation
// =============================================================================

/// Internal node representation for TreeSHAP algorithm
#[derive(Debug, Clone)]
struct InternalNode {
    /// Feature index (-1 for leaf)
    feature: i32,
    /// Threshold for split
    threshold: f64,
    /// Left child index (-1 for none)
    left_child: i32,
    /// Right child index (-1 for none)
    right_child: i32,
    /// Node value (prediction at this node)
    value: f64,
    /// Number of samples reaching this node (for computing weights)
    cover: f64,
}

/// Internal tree representation optimized for TreeSHAP
#[derive(Debug, Clone)]
struct InternalTree {
    nodes: Vec<InternalNode>,
    #[allow(dead_code)]
    n_features: usize,
}

impl InternalTree {
    /// Convert from TreeStructure
    fn from_tree_structure(tree: &TreeStructure, is_classifier: bool) -> Self {
        let nodes: Vec<InternalNode> = tree
            .nodes
            .iter()
            .map(|node| {
                let value = if is_classifier && !node.value.is_empty() {
                    // For classifier, use the proportion of the majority class
                    // or the raw value if it's already normalized
                    let total: f64 = node.value.iter().sum();
                    if total > 0.0 && node.value.len() > 1 {
                        node.value.iter().cloned().fold(0.0, f64::max) / total
                    } else if !node.value.is_empty() {
                        node.value[0]
                    } else {
                        0.0
                    }
                } else if !node.value.is_empty() {
                    node.value[0]
                } else {
                    0.0
                };

                InternalNode {
                    feature: node.feature_index.map(|f| f as i32).unwrap_or(-1),
                    threshold: node.threshold.unwrap_or(0.0),
                    left_child: node.left_child.map(|c| c as i32).unwrap_or(-1),
                    right_child: node.right_child.map(|c| c as i32).unwrap_or(-1),
                    value,
                    cover: node.weighted_n_samples,
                }
            })
            .collect();

        Self {
            nodes,
            n_features: tree.n_features,
        }
    }

    /// Check if a node is a leaf
    fn is_leaf(&self, node_idx: usize) -> bool {
        self.nodes[node_idx].left_child < 0
    }
}

// =============================================================================
// TreeExplainer
// =============================================================================

/// TreeSHAP explainer for tree-based models
///
/// Provides methods to compute exact SHAP values for predictions made by
/// tree-based models including decision trees, random forests, and gradient
/// boosting models.
#[derive(Debug, Clone)]
pub struct TreeExplainer {
    /// Internal tree representations
    trees: Vec<InternalTree>,
    /// Number of features
    n_features: usize,
    /// Base value (expected prediction)
    base_value: f64,
    /// Feature names (if available)
    feature_names: Option<Vec<String>>,
    /// Model type for context
    #[allow(dead_code)]
    model_type: String,
    /// Whether this is a classifier
    #[allow(dead_code)]
    is_classifier: bool,
    /// Scaling factor for ensemble (e.g., 1/n_trees for average)
    scale_factor: f64,
}

impl TreeExplainer {
    /// Create a TreeExplainer from a DecisionTreeRegressor
    pub fn from_decision_tree_regressor(model: &DecisionTreeRegressor) -> Result<Self> {
        let tree = model
            .tree()
            .ok_or_else(|| FerroError::not_fitted("TreeExplainer::from_decision_tree_regressor"))?;

        let internal_tree = InternalTree::from_tree_structure(tree, false);
        let n_features = tree.n_features;

        // Base value is the root node's value (expected prediction)
        let base_value = internal_tree.nodes[0].value;

        Ok(Self {
            trees: vec![internal_tree],
            n_features,
            base_value,
            feature_names: tree.feature_names.clone(),
            model_type: "DecisionTreeRegressor".to_string(),
            is_classifier: false,
            scale_factor: 1.0,
        })
    }

    /// Create a TreeExplainer from a DecisionTreeClassifier
    pub fn from_decision_tree_classifier(model: &DecisionTreeClassifier) -> Result<Self> {
        let tree = model.tree().ok_or_else(|| {
            FerroError::not_fitted("TreeExplainer::from_decision_tree_classifier")
        })?;

        let internal_tree = InternalTree::from_tree_structure(tree, true);
        let n_features = tree.n_features;

        // Base value is the root node's value
        let base_value = internal_tree.nodes[0].value;

        Ok(Self {
            trees: vec![internal_tree],
            n_features,
            base_value,
            feature_names: tree.feature_names.clone(),
            model_type: "DecisionTreeClassifier".to_string(),
            is_classifier: true,
            scale_factor: 1.0,
        })
    }

    /// Create a TreeExplainer from a RandomForestRegressor
    pub fn from_random_forest_regressor(
        model: &crate::models::forest::RandomForestRegressor,
    ) -> Result<Self> {
        let estimators = model
            .estimators()
            .ok_or_else(|| FerroError::not_fitted("TreeExplainer::from_random_forest_regressor"))?;

        if estimators.is_empty() {
            return Err(FerroError::invalid_input("RandomForest has no estimators"));
        }

        let n_features = model.n_features().unwrap_or(0);
        let n_trees = estimators.len();

        let mut trees = Vec::with_capacity(n_trees);
        let mut base_value_sum = 0.0;

        for tree_model in estimators {
            if let Some(tree) = tree_model.tree() {
                let internal_tree = InternalTree::from_tree_structure(tree, false);
                base_value_sum += internal_tree.nodes[0].value;
                trees.push(internal_tree);
            }
        }

        let base_value = base_value_sum / n_trees as f64;

        Ok(Self {
            trees,
            n_features,
            base_value,
            feature_names: None,
            model_type: "RandomForestRegressor".to_string(),
            is_classifier: false,
            scale_factor: 1.0 / n_trees as f64,
        })
    }

    /// Create a TreeExplainer from a RandomForestClassifier
    pub fn from_random_forest_classifier(
        model: &crate::models::forest::RandomForestClassifier,
    ) -> Result<Self> {
        let estimators = model.estimators().ok_or_else(|| {
            FerroError::not_fitted("TreeExplainer::from_random_forest_classifier")
        })?;

        if estimators.is_empty() {
            return Err(FerroError::invalid_input("RandomForest has no estimators"));
        }

        let n_features = model.n_features().unwrap_or(0);
        let n_trees = estimators.len();

        let mut trees = Vec::with_capacity(n_trees);
        let mut base_value_sum = 0.0;

        for tree_model in estimators {
            if let Some(tree) = tree_model.tree() {
                let internal_tree = InternalTree::from_tree_structure(tree, true);
                base_value_sum += internal_tree.nodes[0].value;
                trees.push(internal_tree);
            }
        }

        let base_value = base_value_sum / n_trees as f64;

        Ok(Self {
            trees,
            n_features,
            base_value,
            feature_names: None,
            model_type: "RandomForestClassifier".to_string(),
            is_classifier: true,
            scale_factor: 1.0 / n_trees as f64,
        })
    }

    /// Create a TreeExplainer from a GradientBoostingRegressor
    pub fn from_gradient_boosting_regressor(
        model: &crate::models::boosting::GradientBoostingRegressor,
    ) -> Result<Self> {
        let estimators = model.estimators().ok_or_else(|| {
            FerroError::not_fitted("TreeExplainer::from_gradient_boosting_regressor")
        })?;

        if estimators.is_empty() {
            return Err(FerroError::invalid_input(
                "GradientBoosting has no estimators",
            ));
        }

        let n_features = model.n_features().unwrap_or(0);
        let init_prediction = model.init_prediction().unwrap_or(0.0);

        let mut trees = Vec::with_capacity(estimators.len());

        for tree_model in estimators {
            if let Some(tree) = tree_model.tree() {
                let internal_tree = InternalTree::from_tree_structure(tree, false);
                trees.push(internal_tree);
            }
        }

        // For gradient boosting, base value is the init_prediction
        // SHAP values are additive with learning rate already applied in tree values

        Ok(Self {
            trees,
            n_features,
            base_value: init_prediction,
            feature_names: None,
            model_type: "GradientBoostingRegressor".to_string(),
            is_classifier: false,
            scale_factor: 1.0, // Learning rate is already in tree values
        })
    }

    /// Get the base value (expected prediction)
    #[must_use]
    pub fn base_value(&self) -> f64 {
        self.base_value
    }

    /// Get the number of trees
    #[must_use]
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Get the number of features
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Explain a single prediction
    ///
    /// # Arguments
    /// * `x` - Feature values for a single sample
    ///
    /// # Returns
    /// SHAP values for the prediction
    pub fn explain(&self, x: &[f64]) -> Result<SHAPResult> {
        if x.len() != self.n_features {
            return Err(FerroError::shape_mismatch(
                format!("Expected {} features", self.n_features),
                format!("Got {} features", x.len()),
            ));
        }

        let mut shap_values = Array1::zeros(self.n_features);

        // Compute SHAP values for each tree and aggregate
        for tree in &self.trees {
            let tree_shap = self.tree_shap_values(tree, x);
            for i in 0..self.n_features {
                shap_values[i] += tree_shap[i] * self.scale_factor;
            }
        }

        Ok(SHAPResult {
            base_value: self.base_value,
            shap_values,
            feature_values: Array1::from_vec(x.to_vec()),
            n_features: self.n_features,
            feature_names: self.feature_names.clone(),
        })
    }

    /// Explain multiple predictions in batch
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples, n_features)
    ///
    /// # Returns
    /// SHAP values for all predictions
    pub fn explain_batch(&self, x: &Array2<f64>) -> Result<SHAPBatchResult> {
        if x.ncols() != self.n_features {
            return Err(FerroError::shape_mismatch(
                format!("Expected {} features", self.n_features),
                format!("Got {} features", x.ncols()),
            ));
        }

        let n_samples = x.nrows();
        let mut shap_values = Array2::zeros((n_samples, self.n_features));

        for i in 0..n_samples {
            let sample: Vec<f64> = x.row(i).to_vec();
            let result = self.explain(&sample)?;
            for j in 0..self.n_features {
                shap_values[[i, j]] = result.shap_values[j];
            }
        }

        Ok(SHAPBatchResult {
            base_value: self.base_value,
            shap_values,
            feature_values: x.clone(),
            n_samples,
            n_features: self.n_features,
            feature_names: self.feature_names.clone(),
        })
    }

    /// Explain multiple predictions in parallel
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples, n_features)
    ///
    /// # Returns
    /// SHAP values for all predictions
    pub fn explain_batch_parallel(&self, x: &Array2<f64>) -> Result<SHAPBatchResult> {
        if x.ncols() != self.n_features {
            return Err(FerroError::shape_mismatch(
                format!("Expected {} features", self.n_features),
                format!("Got {} features", x.ncols()),
            ));
        }

        let n_samples = x.nrows();

        // Compute SHAP values in parallel
        let sample_results: Vec<Array1<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let sample: Vec<f64> = x.row(i).to_vec();
                let mut shap_values = Array1::zeros(self.n_features);
                for tree in &self.trees {
                    let tree_shap = self.tree_shap_values(tree, &sample);
                    for j in 0..self.n_features {
                        shap_values[j] += tree_shap[j] * self.scale_factor;
                    }
                }
                shap_values
            })
            .collect();

        // Collect results into matrix
        let mut shap_values = Array2::zeros((n_samples, self.n_features));
        for (i, result) in sample_results.into_iter().enumerate() {
            for j in 0..self.n_features {
                shap_values[[i, j]] = result[j];
            }
        }

        Ok(SHAPBatchResult {
            base_value: self.base_value,
            shap_values,
            feature_values: x.clone(),
            n_samples,
            n_features: self.n_features,
            feature_names: self.feature_names.clone(),
        })
    }

    /// Compute SHAP values for a single tree using Lundberg's TreeSHAP Algorithm 2
    ///
    /// This implements the exact TreeSHAP algorithm from "Consistent Individualized
    /// Feature Attribution for Tree Ensembles" (Lundberg et al., 2018). The algorithm
    /// visits both hot and cold branches, tracking path elements with polynomial
    /// weights that encode Shapley coefficients.
    fn tree_shap_values(&self, tree: &InternalTree, x: &[f64]) -> Vec<f64> {
        let mut shap_values = vec![0.0; self.n_features];

        // Maximum path depth is bounded by tree depth + 1 for sentinel
        let max_depth = self.estimate_tree_depth(tree);
        let mut path: Vec<PathElement> = Vec::with_capacity(max_depth + 2);

        // Start recursive traversal from root with sentinel feature (usize::MAX)
        // The sentinel ensures pweight calculations match XGBoost exactly
        self.tree_shap_recursive(
            tree,
            x,
            0,
            &mut path,
            1.0,
            1.0,
            usize::MAX,
            &mut shap_values,
        );

        shap_values
    }

    /// Estimate tree depth for path allocation
    fn estimate_tree_depth(&self, tree: &InternalTree) -> usize {
        fn depth_recursive(tree: &InternalTree, node_idx: usize) -> usize {
            if tree.is_leaf(node_idx) {
                return 0;
            }
            let node = &tree.nodes[node_idx];
            let left_depth = depth_recursive(tree, node.left_child as usize);
            let right_depth = depth_recursive(tree, node.right_child as usize);
            1 + left_depth.max(right_depth)
        }
        depth_recursive(tree, 0)
    }

    /// Sentinel feature index (indicates non-feature path element)
    const SENTINEL_FEATURE: usize = usize::MAX;

    /// Recursive TreeSHAP traversal visiting both hot and cold branches
    fn tree_shap_recursive(
        &self,
        tree: &InternalTree,
        x: &[f64],
        node_idx: usize,
        path: &mut Vec<PathElement>,
        zero_fraction: f64,
        one_fraction: f64,
        parent_feature: usize,
        shap_values: &mut [f64],
    ) {
        // Always extend path (including sentinel at root)
        Self::extend_path(path, zero_fraction, one_fraction, parent_feature);

        if tree.is_leaf(node_idx) {
            // At leaf: accumulate contributions for all features in path
            // Skip index 0 (sentinel) - matches XGBoost's loop starting at i=1
            let leaf_value = tree.nodes[node_idx].value;

            for i in 1..path.len() {
                let el = &path[i];
                // Skip sentinel feature (usize::MAX)
                if el.feature_index == Self::SENTINEL_FEATURE {
                    continue;
                }
                let w = Self::unwound_path_sum(path, i);
                let contrib = w * (el.one_fraction - el.zero_fraction) * leaf_value;
                shap_values[el.feature_index] += contrib;
            }
        } else {
            // Internal node: determine hot vs cold child
            let node = &tree.nodes[node_idx];
            let split_feature = node.feature as usize;
            let threshold = node.threshold;

            let left_idx = node.left_child as usize;
            let right_idx = node.right_child as usize;

            let left_cover = tree.nodes[left_idx].cover;
            let right_cover = tree.nodes[right_idx].cover;
            let node_cover = left_cover + right_cover;

            let go_left = x[split_feature] <= threshold;
            let (hot_idx, cold_idx) = if go_left {
                (left_idx, right_idx)
            } else {
                (right_idx, left_idx)
            };

            let hot_cover = if go_left { left_cover } else { right_cover };
            let cold_cover = if go_left { right_cover } else { left_cover };

            let hot_zero_fraction = if node_cover > 0.0 {
                hot_cover / node_cover
            } else {
                0.5
            };
            let cold_zero_fraction = if node_cover > 0.0 {
                cold_cover / node_cover
            } else {
                0.5
            };

            // Check if this feature was already in path (handle repeated features)
            let mut incoming_zero_fraction = 1.0;
            let mut incoming_one_fraction = 1.0;

            let existing_idx = path.iter().position(|el| el.feature_index == split_feature);
            if let Some(idx) = existing_idx {
                incoming_zero_fraction = path[idx].zero_fraction;
                incoming_one_fraction = path[idx].one_fraction;
                Self::unwind_path(path, idx);
            }

            // Recurse into hot branch (sample goes this way, one_fraction preserved)
            // Clone path to preserve state - extend_path modifies all pweights
            let mut hot_path = path.clone();
            self.tree_shap_recursive(
                tree,
                x,
                hot_idx,
                &mut hot_path,
                hot_zero_fraction * incoming_zero_fraction,
                incoming_one_fraction,
                split_feature,
                shap_values,
            );

            // Recurse into cold branch (sample doesn't go this way, one_fraction = 0)
            // Clone path again for cold branch
            let mut cold_path = path.clone();
            self.tree_shap_recursive(
                tree,
                x,
                cold_idx,
                &mut cold_path,
                cold_zero_fraction * incoming_zero_fraction,
                0.0, // Cold branch: one_fraction is always 0
                split_feature,
                shap_values,
            );

            // Re-extend path if we unwound a repeated feature (for potential further use)
            if existing_idx.is_some() {
                Self::extend_path(
                    path,
                    incoming_zero_fraction,
                    incoming_one_fraction,
                    split_feature,
                );
            }
        }
    }

    /// Extend the path with a new feature, updating polynomial weights
    fn extend_path(
        path: &mut Vec<PathElement>,
        zero_fraction: f64,
        one_fraction: f64,
        feature_index: usize,
    ) {
        let unique_depth = path.len();

        // Add new element
        path.push(PathElement {
            feature_index,
            zero_fraction,
            one_fraction,
            pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
        });

        // Update pweights using recurrence relation
        // pweight[i+1] += one_fraction * pweight[i] * (i+1) / (d+1)
        // pweight[i] = zero_fraction * pweight[i] * (d-i) / (d+1)
        let d_plus_1 = (unique_depth + 1) as f64;

        for i in (0..unique_depth).rev() {
            let old_pweight = path[i].pweight;
            path[i + 1].pweight += one_fraction * old_pweight * ((i + 1) as f64) / d_plus_1;
            path[i].pweight = zero_fraction * old_pweight * ((unique_depth - i) as f64) / d_plus_1;
        }
    }

    /// Unwind a feature from the path (for handling repeated features)
    ///
    /// This matches XGBoost's unwind_path: pweights are adjusted in place at their
    /// POSITION indices (not shifted with elements), then only feature/fraction
    /// data is shifted down.
    fn unwind_path(path: &mut Vec<PathElement>, path_index: usize) {
        let unique_depth = path.len() - 1;
        if unique_depth == 0 {
            path.remove(path_index);
            return;
        }

        let one_fraction = path[path_index].one_fraction;
        let zero_fraction = path[path_index].zero_fraction;

        let mut next_one_portion = path[unique_depth].pweight;
        let d_plus_1 = (unique_depth + 1) as f64;

        // Adjust pweights - these stay at their position indices
        for i in (0..unique_depth).rev() {
            if one_fraction != 0.0 {
                let tmp = path[i].pweight;
                path[i].pweight = next_one_portion * d_plus_1 / (((i + 1) as f64) * one_fraction);
                next_one_portion =
                    tmp - path[i].pweight * zero_fraction * ((unique_depth - i) as f64) / d_plus_1;
            } else if zero_fraction != 0.0 {
                path[i].pweight =
                    path[i].pweight * d_plus_1 / (zero_fraction * ((unique_depth - i) as f64));
            }
        }

        // Shift feature/fraction data down (but NOT pweights - they stay at position)
        // This matches XGBoost's behavior where pweights are tied to path positions
        for i in path_index..unique_depth {
            path[i].feature_index = path[i + 1].feature_index;
            path[i].zero_fraction = path[i + 1].zero_fraction;
            path[i].one_fraction = path[i + 1].one_fraction;
            // Note: pweight is NOT copied, it stays at position i
        }

        // Remove the last element (now redundant after shift)
        path.pop();
    }

    /// Compute the sum of Shapley weights for unwinding a feature from the path
    ///
    /// This implements the formula from XGBoost's treeshap.h. The multiplication by
    /// (unique_depth + 1) happens at the very end, not inside the loop.
    fn unwound_path_sum(path: &[PathElement], path_index: usize) -> f64 {
        let unique_depth = path.len() - 1;
        if unique_depth == 0 {
            return 1.0;
        }

        let one_fraction = path[path_index].one_fraction;
        let zero_fraction = path[path_index].zero_fraction;

        let mut total = 0.0;

        if one_fraction != 0.0 {
            // Hot branch case: iterative formula (matching XGBoost exactly)
            let mut next_one_portion = path[unique_depth].pweight;

            for i in (0..unique_depth).rev() {
                let tmp = next_one_portion / (((i + 1) as f64) * one_fraction);
                total += tmp;
                next_one_portion =
                    path[i].pweight - tmp * zero_fraction * ((unique_depth - i) as f64);
            }
        } else if zero_fraction != 0.0 {
            // Cold branch case: direct summation formula
            for i in 0..unique_depth {
                total += path[i].pweight / (zero_fraction * ((unique_depth - i) as f64));
            }
        }

        // Final multiplication by (unique_depth + 1) as per XGBoost
        total * ((unique_depth + 1) as f64)
    }
}

/// Path element for TreeSHAP algorithm
#[derive(Clone, Debug)]
struct PathElement {
    /// Index of the feature at this path position
    feature_index: usize,
    /// Fraction of samples going this way when feature is NOT in coalition
    zero_fraction: f64,
    /// Fraction when feature IS in coalition (1 for hot, 0 for cold)
    one_fraction: f64,
    /// Polynomial weight encoding Shapley coefficients
    pweight: f64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;
    use ndarray::Array1;

    fn make_simple_regression_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linear relationship: y = x0 + 0.5*x1
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 2.0, 5.0, 1.0, 6.0, 4.0, 7.0, 3.0, 8.0, 5.0,
                9.0, 2.0, 10.0, 4.0, 1.5, 3.0, 2.5, 2.0, 3.5, 4.0, 4.5, 1.0, 5.5, 3.0, 6.5, 2.0,
                7.5, 4.0, 8.5, 1.0, 9.5, 3.0, 10.5, 2.0,
            ],
        )
        .unwrap();
        let y: Array1<f64> = x.column(0).to_owned() + x.column(1).to_owned() * 0.5;
        (x, y)
    }

    fn make_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 1.5, 1.0, 2.5, 2.0, 1.0, 6.0, 6.0, 6.5, 6.5,
                7.0, 7.0, 7.5, 6.5, 6.0, 7.5, 7.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        (x, y)
    }

    #[test]
    fn test_tree_explainer_decision_tree_regressor() {
        let (x, y) = make_simple_regression_data();

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        assert_eq!(explainer.n_features(), 2);
        assert_eq!(explainer.n_trees(), 1);

        // Explain a single sample
        let sample = vec![5.0, 3.0];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 2);
        assert_eq!(result.shap_values.len(), 2);

        // Prediction should approximately equal base_value + sum(shap_values)
        let reconstructed = result.prediction();
        let actual = model
            .predict(&Array2::from_shape_vec((1, 2), sample.clone()).unwrap())
            .unwrap()[0];

        // Allow some tolerance for numerical precision
        assert!(
            (reconstructed - actual).abs() < 1.0,
            "Reconstructed {} vs actual {}",
            reconstructed,
            actual
        );
    }

    #[test]
    fn test_tree_explainer_decision_tree_classifier() {
        let (x, y) = make_classification_data();

        let mut model = DecisionTreeClassifier::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_classifier(&model).unwrap();

        assert_eq!(explainer.n_features(), 2);
        assert_eq!(explainer.n_trees(), 1);

        // Explain a single sample
        let sample = vec![1.5, 1.5];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 2);
        assert_eq!(result.shap_values.len(), 2);
    }

    #[test]
    fn test_tree_explainer_batch() {
        let (x, y) = make_simple_regression_data();

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        // Explain batch
        let test_x = x.slice(ndarray::s![0..5, ..]).to_owned();
        let result = explainer.explain_batch(&test_x).unwrap();

        assert_eq!(result.n_samples, 5);
        assert_eq!(result.n_features, 2);
        assert_eq!(result.shap_values.shape(), &[5, 2]);
    }

    #[test]
    fn test_tree_explainer_batch_parallel() {
        let (x, y) = make_simple_regression_data();

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        // Explain batch in parallel
        let test_x = x.slice(ndarray::s![0..5, ..]).to_owned();
        let result = explainer.explain_batch_parallel(&test_x).unwrap();

        assert_eq!(result.n_samples, 5);
        assert_eq!(result.n_features, 2);
    }

    #[test]
    fn test_shap_result_methods() {
        let result = SHAPResult {
            base_value: 0.5,
            shap_values: Array1::from_vec(vec![0.3, -0.1, 0.2]),
            feature_values: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            n_features: 3,
            feature_names: Some(vec![
                "feature_a".to_string(),
                "feature_b".to_string(),
                "feature_c".to_string(),
            ]),
        };

        // Test prediction reconstruction
        let prediction = result.prediction();
        assert!((prediction - 0.9).abs() < 1e-10); // 0.5 + 0.3 - 0.1 + 0.2 = 0.9

        // Test sorted indices (by absolute value)
        let sorted = result.sorted_indices();
        assert_eq!(sorted, vec![0, 2, 1]); // |0.3| > |0.2| > |-0.1|

        // Test top_k
        let top2 = result.top_k(2);
        assert_eq!(top2, vec![0, 2]);

        // Test format_feature
        let formatted = result.format_feature(0);
        assert!(formatted.contains("feature_a"));
        assert!(formatted.contains("1.0000"));
        assert!(formatted.contains("+0.3000"));
    }

    #[test]
    fn test_shap_batch_result_methods() {
        let shap_values =
            Array2::from_shape_vec((3, 2), vec![0.3, -0.1, 0.2, -0.2, 0.1, 0.3]).unwrap();

        let result = SHAPBatchResult {
            base_value: 0.5,
            shap_values,
            feature_values: Array2::zeros((3, 2)),
            n_samples: 3,
            n_features: 2,
            feature_names: None,
        };

        // Test mean absolute SHAP
        let mean_abs = result.mean_abs_shap();
        assert!((mean_abs[0] - 0.2).abs() < 1e-10); // (|0.3| + |0.2| + |0.1|) / 3
        assert!((mean_abs[1] - 0.2).abs() < 1e-10); // (|-0.1| + |-0.2| + |0.3|) / 3

        // Test get_sample
        let sample_result = result.get_sample(1).unwrap();
        assert!((sample_result.shap_values[0] - 0.2).abs() < 1e-10);
        assert!((sample_result.shap_values[1] - (-0.2)).abs() < 1e-10);

        // Test invalid index
        assert!(result.get_sample(10).is_none());
    }

    #[test]
    fn test_tree_explainer_not_fitted_error() {
        let model = DecisionTreeRegressor::new();
        let result = TreeExplainer::from_decision_tree_regressor(&model);
        assert!(result.is_err());
    }

    #[test]
    fn test_tree_explainer_wrong_features_error() {
        let (x, y) = make_simple_regression_data();

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model).unwrap();

        // Wrong number of features
        let sample = vec![1.0, 2.0, 3.0];
        let result = explainer.explain(&sample);
        assert!(result.is_err());
    }

    #[test]
    fn test_summary_display() {
        let result = SHAPResult {
            base_value: 0.5,
            shap_values: Array1::from_vec(vec![0.3, -0.1]),
            feature_values: Array1::from_vec(vec![1.0, 2.0]),
            n_features: 2,
            feature_names: Some(vec!["x1".to_string(), "x2".to_string()]),
        };

        let summary = result.summary();
        assert!(summary.contains("SHAP Explanation"));
        assert!(summary.contains("Base value: 0.5000"));
        assert!(summary.contains("x1"));
        assert!(summary.contains("x2"));
    }

    #[test]
    fn test_random_forest_regressor_explainer() {
        use crate::models::forest::RandomForestRegressor;

        let (x, y) = make_simple_regression_data();

        let mut model = RandomForestRegressor::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_random_forest_regressor(&model).unwrap();

        assert_eq!(explainer.n_features(), 2);
        assert_eq!(explainer.n_trees(), 5);

        let sample = vec![5.0, 3.0];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 2);
        assert_eq!(result.shap_values.len(), 2);
    }

    #[test]
    fn test_random_forest_classifier_explainer() {
        use crate::models::forest::RandomForestClassifier;

        let (x, y) = make_classification_data();

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(5)
            .with_max_depth(Some(3))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_random_forest_classifier(&model).unwrap();

        assert_eq!(explainer.n_features(), 2);
        assert_eq!(explainer.n_trees(), 5);

        let sample = vec![1.5, 1.5];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 2);
    }

    #[test]
    fn test_gradient_boosting_regressor_explainer() {
        use crate::models::boosting::GradientBoostingRegressor;

        let (x, y) = make_simple_regression_data();

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(2))
            .with_learning_rate(0.1);
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_gradient_boosting_regressor(&model).unwrap();

        assert_eq!(explainer.n_features(), 2);
        assert_eq!(explainer.n_trees(), 10);

        let sample = vec![5.0, 3.0];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(result.n_features, 2);
    }

    #[test]
    fn test_feature_names_propagation() {
        let (x, y) = make_simple_regression_data();

        let mut model = DecisionTreeRegressor::new().with_max_depth(Some(3));
        model.fit(&x, &y).unwrap();

        let explainer = TreeExplainer::from_decision_tree_regressor(&model)
            .unwrap()
            .with_feature_names(vec!["height".to_string(), "weight".to_string()]);

        let sample = vec![5.0, 3.0];
        let result = explainer.explain(&sample).unwrap();

        assert_eq!(
            result.feature_names,
            Some(vec!["height".to_string(), "weight".to_string()])
        );
    }

    /// Test TreeSHAP on manually constructed depth-1 tree with known Shapley values
    #[test]
    fn test_treeshap_depth1_manual_verification() {
        // Tree structure:
        // Node 0: split on f0 at threshold 5, left->1, right->2
        // Node 1: leaf, value=2, cover=3
        // Node 2: leaf, value=8, cover=2
        //
        // Sample x=[3,0] goes left (f0=3 <= 5)
        // Base = (3*2 + 2*8) / 5 = 22/5 = 4.4
        // Prediction = 2.0
        // Expected: SHAP[0] = 2.0 - 4.4 = -2.4, SHAP[1] = 0.0

        let tree = super::InternalTree {
            nodes: vec![
                super::InternalNode {
                    feature: 0,
                    threshold: 5.0,
                    left_child: 1,
                    right_child: 2,
                    value: 4.4, // root value = expected value
                    cover: 5.0,
                },
                super::InternalNode {
                    feature: -1,
                    threshold: 0.0,
                    left_child: -1,
                    right_child: -1,
                    value: 2.0,
                    cover: 3.0,
                },
                super::InternalNode {
                    feature: -1,
                    threshold: 0.0,
                    left_child: -1,
                    right_child: -1,
                    value: 8.0,
                    cover: 2.0,
                },
            ],
            n_features: 2,
        };

        let explainer = TreeExplainer {
            trees: vec![tree],
            base_value: 4.4,
            n_features: 2,
            is_classifier: false,
            scale_factor: 1.0,
            feature_names: None,
            model_type: "test".to_string(),
        };

        let sample = vec![3.0, 0.0];
        let shap = explainer.tree_shap_values(&explainer.trees[0], &sample);

        // SHAP[0] should be prediction - base = 2.0 - 4.4 = -2.4
        assert!(
            (shap[0] - (-2.4)).abs() < 1e-10,
            "SHAP[0] = {}, expected -2.4",
            shap[0]
        );

        // SHAP[1] should be 0 (f1 is not on the decision path)
        assert!(shap[1].abs() < 1e-10, "SHAP[1] = {}, expected 0.0", shap[1]);

        // Sum should equal prediction - base
        let shap_sum: f64 = shap.iter().sum();
        assert!(
            (shap_sum - (-2.4)).abs() < 1e-10,
            "SHAP sum = {}, expected -2.4",
            shap_sum
        );
    }

    /// Test TreeSHAP on manually constructed depth-2 tree with known Shapley values
    #[test]
    fn test_treeshap_depth2_manual_verification() {
        // Tree structure:
        // Node 0: split on f0 at 5, left->1, right->4
        // Node 1: split on f1 at 2, left->2, right->3
        // Node 2: leaf, value=1, cover=2
        // Node 3: leaf, value=4, cover=1
        // Node 4: leaf, value=10, cover=2
        //
        // Sample x=[3,1] goes left-left to leaf 2 (f0=3<=5, f1=1<=2)
        // Base = (2*1 + 1*4 + 2*10) / 5 = 26/5 = 5.2
        // Prediction = 1.0
        //
        // Hand-computed Shapley values:
        // Let f(S) = E[prediction | features in S are known]
        // f({}) = 5.2 (base)
        // f({0}) = E[pred | f0=3] = (2*1 + 1*4) / 3 = 2.0 (goes left subtree)
        // f({1}) = E[pred | f1=1] = (2*1 + 2*10) / 5 * (2/3) + (1*4) / 5 * (1/3)
        //        = Actually this is more complex...
        //
        // Using Shapley formula directly:
        // φ_0 = 0.5 * [f({0}) - f({})] + 0.5 * [f({0,1}) - f({1})]
        // φ_1 = 0.5 * [f({1}) - f({})] + 0.5 * [f({0,1}) - f({0})]
        //
        // f({}) = 5.2
        // f({0,1}) = 1.0 (prediction)
        // f({0}) = weighted avg of left subtree = (2*1 + 1*4) / 3 = 6/3 = 2.0
        // f({1}) = need to compute E[pred | only f1 known]
        //   - With f1=1 (<=2), we know we'd go left at node 1 IF we're in left subtree
        //   - But from root, without f0, we might go left or right
        //   - P(left at root) = 3/5, then we're at node 1, where f1=1 goes left -> value 1
        //   - P(right at root) = 2/5, we're at leaf 4 -> value 10
        //   - f({1}) = (3/5)*1 + (2/5)*10 = 3/5 + 4 = 0.6 + 4 = 4.6
        //
        // φ_0 = 0.5 * (2.0 - 5.2) + 0.5 * (1.0 - 4.6)
        //     = 0.5 * (-3.2) + 0.5 * (-3.6)
        //     = -1.6 + (-1.8) = -3.4
        //
        // φ_1 = 0.5 * (4.6 - 5.2) + 0.5 * (1.0 - 2.0)
        //     = 0.5 * (-0.6) + 0.5 * (-1.0)
        //     = -0.3 + (-0.5) = -0.8
        //
        // Sum = -3.4 + (-0.8) = -4.2 = 1.0 - 5.2 ✓

        let tree = super::InternalTree {
            nodes: vec![
                super::InternalNode {
                    feature: 0,
                    threshold: 5.0,
                    left_child: 1,
                    right_child: 4,
                    value: 5.2,
                    cover: 5.0,
                },
                super::InternalNode {
                    feature: 1,
                    threshold: 2.0,
                    left_child: 2,
                    right_child: 3,
                    value: 2.0, // (2*1 + 1*4) / 3
                    cover: 3.0,
                },
                super::InternalNode {
                    feature: -1,
                    threshold: 0.0,
                    left_child: -1,
                    right_child: -1,
                    value: 1.0,
                    cover: 2.0,
                },
                super::InternalNode {
                    feature: -1,
                    threshold: 0.0,
                    left_child: -1,
                    right_child: -1,
                    value: 4.0,
                    cover: 1.0,
                },
                super::InternalNode {
                    feature: -1,
                    threshold: 0.0,
                    left_child: -1,
                    right_child: -1,
                    value: 10.0,
                    cover: 2.0,
                },
            ],
            n_features: 2,
        };

        let explainer = TreeExplainer {
            trees: vec![tree],
            base_value: 5.2,
            n_features: 2,
            is_classifier: false,
            scale_factor: 1.0,
            feature_names: None,
            model_type: "test".to_string(),
        };

        let sample = vec![3.0, 1.0];
        let shap = explainer.tree_shap_values(&explainer.trees[0], &sample);

        // Sum should equal prediction - base = 1.0 - 5.2 = -4.2
        let shap_sum: f64 = shap.iter().sum();
        let expected_sum = 1.0 - 5.2;
        assert!(
            (shap_sum - expected_sum).abs() < 1e-10,
            "SHAP sum = {}, expected {}",
            shap_sum,
            expected_sum
        );

        // Verify individual Shapley values match hand computation
        // φ_0 = -3.4, φ_1 = -0.8
        assert!(
            (shap[0] - (-3.4)).abs() < 1e-10,
            "SHAP[0] = {}, expected -3.4",
            shap[0]
        );
        assert!(
            (shap[1] - (-0.8)).abs() < 1e-10,
            "SHAP[1] = {}, expected -0.8",
            shap[1]
        );
    }
}
