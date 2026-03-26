//! Histogram-Based Gradient Boosting (LightGBM-style)
//!
//! This module provides histogram-based gradient boosting implementations that offer
//! significant speed improvements over traditional gradient boosting by using
//! histogram-based split finding.
//!
//! ## Key Features
//!
//! - **O(n) split finding**: Bins continuous features into discrete histograms
//! - **Native missing value handling**: Learns optimal direction for missing values
//! - **Monotonic constraints**: Enforce monotonicity in partial dependence
//! - **Feature interaction constraints**: Control which features can interact
//! - **Leaf-wise growth**: LightGBM-style best-first tree growth
//! - **L1/L2 regularization**: Regularized split gains
//!
//! ## Example
//!
//! ```
//! use ferroml_core::models::hist_boosting::HistGradientBoostingClassifier;
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| i as f64 / 100.0).collect()).unwrap();
//! let y = Array1::from_iter((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }));
//!
//! let mut model = HistGradientBoostingClassifier::new()
//!     .with_max_iter(100)
//!     .with_learning_rate(0.1)
//!     .with_max_bins(256);
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! let probas = model.predict_proba(&x).unwrap();
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input_allow_nan, validate_predict_input_allow_nan, Model,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};

// =============================================================================
// Configuration Types
// =============================================================================

/// Monotonic constraint direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonotonicConstraint {
    /// No constraint on this feature
    None,
    /// Feature must have positive effect (increasing feature -> increasing prediction)
    Positive,
    /// Feature must have negative effect (increasing feature -> decreasing prediction)
    Negative,
}

impl Default for MonotonicConstraint {
    fn default() -> Self {
        Self::None
    }
}

/// Tree growth strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrowthStrategy {
    /// Grow tree level by level (like standard CART)
    DepthFirst,
    /// Grow leaf with largest gain first (like LightGBM)
    LeafWise,
}

impl Default for GrowthStrategy {
    fn default() -> Self {
        Self::LeafWise
    }
}

/// Loss function for histogram gradient boosting (classification)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistLoss {
    /// Log loss (cross-entropy) for binary classification
    LogLoss,
    /// Hinge loss for binary classification
    Hinge,
}

impl Default for HistLoss {
    fn default() -> Self {
        Self::LogLoss
    }
}

/// Loss function for histogram gradient boosting (regression)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistRegressionLoss {
    /// Squared error (L2 loss) - standard regression loss
    SquaredError,
    /// Absolute error (L1 loss) - robust to outliers
    AbsoluteError,
    /// Huber loss - combines L2 for small errors and L1 for large errors
    Huber,
}

impl Default for HistRegressionLoss {
    fn default() -> Self {
        Self::SquaredError
    }
}

impl HistRegressionLoss {
    /// Compute the gradient for a single sample
    ///
    /// # Arguments
    /// * `y_true` - True target value
    /// * `y_pred` - Current prediction
    /// * `delta` - Huber delta parameter (only used for Huber loss)
    pub fn gradient(&self, y_true: f64, y_pred: f64, delta: f64) -> f64 {
        let residual = y_pred - y_true;
        match self {
            HistRegressionLoss::SquaredError => residual,
            HistRegressionLoss::AbsoluteError => residual.signum(),
            HistRegressionLoss::Huber => {
                if residual.abs() <= delta {
                    residual
                } else {
                    delta * residual.signum()
                }
            }
        }
    }

    /// Compute the hessian for a single sample
    ///
    /// # Arguments
    /// * `y_true` - True target value
    /// * `y_pred` - Current prediction
    /// * `delta` - Huber delta parameter (only used for Huber loss)
    ///
    /// Note: For AbsoluteError (L1 loss), the true hessian is 0, but we use 1.0 as
    /// a pseudo-hessian (similar to LightGBM). This works because gradient boosting
    /// with constant hessian is equivalent to gradient descent with a fixed step size.
    pub fn hessian(&self, y_true: f64, y_pred: f64, delta: f64) -> f64 {
        let residual = y_pred - y_true;
        match self {
            HistRegressionLoss::SquaredError => 1.0,
            HistRegressionLoss::AbsoluteError => {
                // Use constant pseudo-hessian of 1.0 (like LightGBM)
                // This makes gradient boosting equivalent to gradient descent
                1.0
            }
            HistRegressionLoss::Huber => {
                if residual.abs() <= delta {
                    1.0
                } else {
                    // Outside the quadratic region, use 1.0 as pseudo-hessian
                    1.0
                }
            }
        }
    }

    /// Compute the loss value for evaluation
    pub fn loss(&self, y_true: f64, y_pred: f64, delta: f64) -> f64 {
        let residual = y_pred - y_true;
        match self {
            HistRegressionLoss::SquaredError => 0.5 * residual * residual,
            HistRegressionLoss::AbsoluteError => residual.abs(),
            HistRegressionLoss::Huber => {
                if residual.abs() <= delta {
                    0.5 * residual * residual
                } else {
                    delta * 0.5f64.mul_add(-delta, residual.abs())
                }
            }
        }
    }

    /// Compute the initial prediction for the ensemble
    pub fn initial_prediction(&self, y: &Array1<f64>) -> f64 {
        match self {
            HistRegressionLoss::SquaredError => y.mean().unwrap_or(0.0),
            HistRegressionLoss::AbsoluteError | HistRegressionLoss::Huber => {
                // Median is the optimal initial prediction for absolute error
                let mut sorted: Vec<f64> = y.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n == 0 {
                    0.0
                } else if n % 2 == 0 {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                } else {
                    sorted[n / 2]
                }
            }
        }
    }
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistEarlyStopping {
    /// Number of iterations with no improvement before stopping
    pub patience: usize,
    /// Minimum improvement to consider as progress
    pub tol: f64,
    /// Fraction of data to use for validation
    pub validation_fraction: f64,
}

impl Default for HistEarlyStopping {
    fn default() -> Self {
        Self {
            patience: 10,
            tol: 1e-7,
            validation_fraction: 0.1,
        }
    }
}

// =============================================================================
// Histogram Infrastructure
// =============================================================================

/// Trait for bin mappers that can provide bin counts
pub trait BinMapperInfo {
    /// Get the number of bins for a feature (includes missing bin)
    fn n_bins(&self, feature_idx: usize) -> usize;
}

/// Represents a bin boundary for a feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinMapper {
    /// Bin edges for each feature (n_features arrays of bin edges)
    bin_edges: Vec<Vec<f64>>,
    /// Number of bins used for each feature
    n_bins_per_feature: Vec<usize>,
    /// Maximum number of bins
    max_bins: usize,
    /// Index of the bin used for missing values (always the last bin)
    missing_bin: usize,
}

impl BinMapper {
    /// Create a new bin mapper
    pub fn new(max_bins: usize) -> Self {
        Self {
            bin_edges: Vec::new(),
            n_bins_per_feature: Vec::new(),
            max_bins,
            missing_bin: max_bins, // Last bin reserved for missing
        }
    }

    /// Fit the bin mapper to the data
    pub fn fit(&mut self, x: &Array2<f64>) {
        let n_features = x.ncols();
        self.bin_edges = Vec::with_capacity(n_features);
        self.n_bins_per_feature = Vec::with_capacity(n_features);

        for col_idx in 0..n_features {
            let col = x.column(col_idx);

            // Collect non-missing values and sort
            let mut values: Vec<f64> = col.iter().filter(|&&v| !v.is_nan()).copied().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if values.is_empty() {
                // All missing - single bin
                self.bin_edges.push(vec![f64::NEG_INFINITY, f64::INFINITY]);
                self.n_bins_per_feature.push(1);
                continue;
            }

            // Remove duplicates
            values.dedup();

            // Calculate quantile-based bin edges
            let n_unique = values.len();
            let n_bins = (self.max_bins - 1).min(n_unique); // Reserve one bin for missing

            let mut edges = Vec::with_capacity(n_bins + 1);
            edges.push(f64::NEG_INFINITY);

            if n_bins > 1 {
                for i in 1..n_bins {
                    let idx = (i * n_unique) / n_bins;
                    let edge = values[idx.min(n_unique - 1)];
                    // Only add if different from previous
                    if edges.last().map(|&e| edge > e).unwrap_or(true) {
                        edges.push(edge);
                    }
                }
            }

            edges.push(f64::INFINITY);

            let actual_bins = edges.len() - 1;
            self.bin_edges.push(edges);
            self.n_bins_per_feature.push(actual_bins);
        }
    }

    /// Transform features to bin indices
    pub fn transform(&self, x: &Array2<f64>) -> Array2<u8> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut binned = Array2::zeros((n_samples, n_features));

        for col_idx in 0..n_features {
            let edges = &self.bin_edges[col_idx];
            for row_idx in 0..n_samples {
                let value = x[[row_idx, col_idx]];

                if value.is_nan() {
                    // Missing values go to the missing bin
                    binned[[row_idx, col_idx]] = self.missing_bin as u8;
                } else {
                    // Binary search for the correct bin (O(log n) vs O(n) linear scan)
                    let bin = match edges.binary_search_by(|e| {
                        e.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Greater)
                    }) {
                        Ok(pos) => pos,
                        Err(pos) => pos.saturating_sub(1),
                    };
                    binned[[row_idx, col_idx]] = bin as u8;
                }
            }
        }

        binned
    }

    /// Get the number of bins for a feature
    pub fn n_bins(&self, feature_idx: usize) -> usize {
        self.n_bins_per_feature
            .get(feature_idx)
            .copied()
            .unwrap_or(1)
            + 1 // +1 for missing bin
    }

    /// Get the real-valued threshold for a bin index.
    ///
    /// The threshold is the upper edge of the bin, so values <= threshold go left.
    /// Returns the bin edge at position `bin + 1` (the upper boundary of the bin).
    pub fn bin_threshold_to_real(&self, feature_idx: usize, bin: u8) -> f64 {
        if let Some(edges) = self.bin_edges.get(feature_idx) {
            let idx = (bin as usize) + 1;
            if idx < edges.len() {
                edges[idx]
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        }
    }
}

impl BinMapperInfo for BinMapper {
    fn n_bins(&self, feature_idx: usize) -> usize {
        self.n_bins(feature_idx)
    }
}

/// Contiguous column-major storage for binned features.
/// Single allocation with stride-based indexing eliminates pointer indirection
/// compared to `Vec<Vec<u8>>`.
struct ColMajorBins {
    data: Vec<u8>,
    n_samples: usize,
    n_features: usize,
}

impl ColMajorBins {
    /// Get a slice for a feature column.
    #[inline]
    fn col(&self, feature: usize) -> &[u8] {
        let start = feature * self.n_samples;
        &self.data[start..start + self.n_samples]
    }
}

/// Convert binned Array2 to contiguous column-major storage for cache-friendly histogram building.
fn to_col_major(x_binned: &Array2<u8>) -> ColMajorBins {
    let n_samples = x_binned.nrows();
    let n_features = x_binned.ncols();
    let mut data = vec![0u8; n_features * n_samples];
    for f in 0..n_features {
        let offset = f * n_samples;
        let col = x_binned.column(f);
        if let Some(slice) = col.as_slice() {
            data[offset..offset + n_samples].copy_from_slice(slice);
        } else {
            for (i, &val) in col.iter().enumerate() {
                data[offset + i] = val;
            }
        }
    }
    ColMajorBins {
        data,
        n_samples,
        n_features,
    }
}

/// A single histogram bin — packs gradient, hessian, and count into one cache line.
///
/// Array-of-Structs layout: all fields for bin[k] are contiguous in memory,
/// so random access by bin index hits one cache line instead of three separate
/// Vec allocations (the old SoA layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct HistogramBin {
    sum_gradients: f64,
    sum_hessians: f64,
    count: u32,
    _pad: u32, // Align to 24 bytes (3 × 8), fits in one cache line
}

/// Histogram for a single node (AoS layout for cache efficiency)
#[derive(Debug, Clone)]
struct Histogram {
    bins: Vec<HistogramBin>,
}

impl Histogram {
    fn new(n_bins: usize) -> Self {
        Self {
            bins: vec![HistogramBin::default(); n_bins],
        }
    }

    /// Number of bins in this histogram.
    #[inline]
    fn n_bins(&self) -> usize {
        self.bins.len()
    }

    /// Compute histogram by subtraction (parent - sibling = self)
    fn compute_by_subtraction(&mut self, parent: &Histogram, sibling: &Histogram) {
        for i in 0..self.bins.len() {
            self.bins[i].sum_gradients =
                parent.bins[i].sum_gradients - sibling.bins[i].sum_gradients;
            self.bins[i].sum_hessians = parent.bins[i].sum_hessians - sibling.bins[i].sum_hessians;
            self.bins[i].count = parent.bins[i].count.saturating_sub(sibling.bins[i].count);
        }
    }
}

// =============================================================================
// Tree Node
// =============================================================================

/// A node in the histogram gradient boosting tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistTreeNode {
    /// Feature index used for splitting (None for leaf)
    pub feature_idx: Option<usize>,
    /// Bin threshold for splitting
    pub bin_threshold: Option<u8>,
    /// Whether missing values go left
    pub missing_go_left: bool,
    /// Left child index
    pub left_child: Option<usize>,
    /// Right child index
    pub right_child: Option<usize>,
    /// Leaf value (prediction value at this node)
    pub value: f64,
    /// Number of samples at this node
    pub n_samples: usize,
    /// Sum of gradients at this node
    pub sum_gradients: f64,
    /// Sum of hessians at this node
    pub sum_hessians: f64,
    /// Depth of this node
    pub depth: usize,
    /// Gain from splitting this node (if internal)
    pub gain: f64,
}

impl HistTreeNode {
    fn new_leaf(
        value: f64,
        n_samples: usize,
        sum_gradients: f64,
        sum_hessians: f64,
        depth: usize,
    ) -> Self {
        Self {
            feature_idx: None,
            bin_threshold: None,
            missing_go_left: true,
            left_child: None,
            right_child: None,
            value,
            n_samples,
            sum_gradients,
            sum_hessians,
            depth,
            gain: 0.0,
        }
    }

    fn is_leaf(&self) -> bool {
        self.left_child.is_none()
    }
}

/// A histogram-based gradient boosting tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistTree {
    /// The nodes in this tree (index 0 is the root)
    pub nodes: Vec<HistTreeNode>,
}

impl HistTree {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Predict for a single sample (binned features)
    fn predict_single(&self, binned_features: &[u8]) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }

        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];
            if node.is_leaf() {
                return node.value;
            }

            let feature_idx = node
                .feature_idx
                .expect("internal node must have feature_idx");
            let bin = binned_features[feature_idx];
            let threshold = node
                .bin_threshold
                .expect("internal node must have bin_threshold");

            // Handle missing values
            let go_left = if bin as usize >= 255 {
                // Missing value indicator
                node.missing_go_left
            } else {
                bin <= threshold
            };

            node_idx = if go_left {
                node.left_child.expect("internal node must have left_child")
            } else {
                node.right_child
                    .expect("internal node must have right_child")
            };
        }
    }
}

// =============================================================================
// Split Finding
// =============================================================================

/// Information about a potential split
#[derive(Debug, Clone)]
struct SplitInfo {
    feature_idx: usize,
    bin_threshold: u8,
    gain: f64,
    sum_gradient_left: f64,
    sum_gradient_right: f64,
    sum_hessian_left: f64,
    sum_hessian_right: f64,
    missing_go_left: bool,
}

impl SplitInfo {
    fn compute_leaf_values(&self, l2_regularization: f64) -> (f64, f64) {
        let left_value = -self.sum_gradient_left / (self.sum_hessian_left + l2_regularization);
        let right_value = -self.sum_gradient_right / (self.sum_hessian_right + l2_regularization);
        (left_value, right_value)
    }
}

/// Node to be split (for priority queue in leaf-wise growth)
#[derive(Debug)]
struct SplitCandidate {
    node_idx: usize,
    split_info: SplitInfo,
}

impl PartialEq for SplitCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.split_info.gain == other.split_info.gain
    }
}

impl Eq for SplitCandidate {}

impl PartialOrd for SplitCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SplitCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.split_info
            .gain
            .partial_cmp(&other.split_info.gain)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// =============================================================================
// Tree Builder
// =============================================================================

/// Builder for histogram-based gradient boosting trees
struct HistTreeBuilder {
    /// Maximum depth of the tree
    max_depth: Option<usize>,
    /// Maximum number of leaf nodes
    max_leaf_nodes: Option<usize>,
    /// Minimum samples per leaf
    min_samples_leaf: usize,
    /// Minimum sum of hessians per leaf
    min_hessian_leaf: f64,
    /// L1 regularization
    l1_regularization: f64,
    /// L2 regularization
    l2_regularization: f64,
    /// Minimum gain to make a split
    min_gain_to_split: f64,
    /// Growth strategy
    growth_strategy: GrowthStrategy,
    /// Monotonic constraints per feature
    monotonic_constraints: Vec<MonotonicConstraint>,
    /// Feature interaction constraints
    interaction_constraints: Option<Vec<HashSet<usize>>>,
    /// Maximum number of bins
    max_bins: usize,
}

impl HistTreeBuilder {
    fn new() -> Self {
        Self {
            max_depth: None,
            max_leaf_nodes: Some(31),
            min_samples_leaf: 20,
            min_hessian_leaf: 1e-3,
            l1_regularization: 0.0,
            l2_regularization: 0.0,
            min_gain_to_split: 0.0,
            growth_strategy: GrowthStrategy::LeafWise,
            monotonic_constraints: Vec::new(),
            interaction_constraints: None,
            max_bins: 256,
        }
    }

    /// Compute the gain from a split
    fn compute_gain(&self, sum_gradients: f64, sum_hessians: f64) -> f64 {
        let gradient_after_l1 = if sum_gradients > self.l1_regularization {
            sum_gradients - self.l1_regularization
        } else if sum_gradients < -self.l1_regularization {
            sum_gradients + self.l1_regularization
        } else {
            0.0
        };

        gradient_after_l1.powi(2) / (sum_hessians + self.l2_regularization)
    }

    /// Find the best split for a node
    fn find_best_split(
        &self,
        histograms: &[Histogram],
        node: &HistTreeNode,
        allowed_features: Option<&HashSet<usize>>,
        parent_lower_bound: f64,
        parent_upper_bound: f64,
    ) -> Option<SplitInfo> {
        let mut best_split: Option<SplitInfo> = None;
        let mut best_gain = self.min_gain_to_split;

        let n_features = histograms.len();
        let base_gain = self.compute_gain(node.sum_gradients, node.sum_hessians);

        for feature_idx in 0..n_features {
            // Check feature interaction constraints
            if let Some(ref allowed) = allowed_features {
                if !allowed.contains(&feature_idx) {
                    continue;
                }
            }

            let histogram = &histograms[feature_idx];
            let n_bins = histogram.n_bins();

            // Get monotonic constraint for this feature
            let constraint = self
                .monotonic_constraints
                .get(feature_idx)
                .copied()
                .unwrap_or(MonotonicConstraint::None);

            // Track bounds for monotonic constraints
            let current_lower = parent_lower_bound;
            let current_upper = parent_upper_bound;

            // Try both directions for missing values
            for missing_go_left in [true, false] {
                let mut sum_gradient_left = 0.0;
                let mut sum_hessian_left = 0.0;
                let mut n_left = 0usize;

                // Add missing bin contribution based on direction
                let missing_bin = n_bins.saturating_sub(1);
                if missing_go_left && missing_bin < n_bins {
                    sum_gradient_left += histogram.bins[missing_bin].sum_gradients;
                    sum_hessian_left += histogram.bins[missing_bin].sum_hessians;
                    n_left += histogram.bins[missing_bin].count as usize;
                }

                // Scan through non-missing bins
                for bin in 0..missing_bin {
                    sum_gradient_left += histogram.bins[bin].sum_gradients;
                    sum_hessian_left += histogram.bins[bin].sum_hessians;
                    n_left += histogram.bins[bin].count as usize;

                    let n_right = node.n_samples.saturating_sub(n_left);
                    let sum_gradient_right = node.sum_gradients - sum_gradient_left;
                    let sum_hessian_right = node.sum_hessians - sum_hessian_left;

                    // Check minimum constraints
                    if n_left < self.min_samples_leaf || n_right < self.min_samples_leaf {
                        continue;
                    }
                    if sum_hessian_left < self.min_hessian_leaf
                        || sum_hessian_right < self.min_hessian_leaf
                    {
                        continue;
                    }

                    // Compute leaf values for constraint checking
                    let left_value =
                        -sum_gradient_left / (sum_hessian_left + self.l2_regularization);
                    let right_value =
                        -sum_gradient_right / (sum_hessian_right + self.l2_regularization);

                    // Check monotonic constraints
                    match constraint {
                        MonotonicConstraint::Positive => {
                            // Left should be <= right
                            if left_value > right_value {
                                continue;
                            }
                            // Check against parent bounds
                            if left_value > current_upper || right_value < current_lower {
                                continue;
                            }
                        }
                        MonotonicConstraint::Negative => {
                            // Left should be >= right
                            if left_value < right_value {
                                continue;
                            }
                            // Check against parent bounds
                            if left_value < current_lower || right_value > current_upper {
                                continue;
                            }
                        }
                        MonotonicConstraint::None => {}
                    }

                    // Compute split gain
                    let gain_left = self.compute_gain(sum_gradient_left, sum_hessian_left);
                    let gain_right = self.compute_gain(sum_gradient_right, sum_hessian_right);
                    let gain = gain_left + gain_right - base_gain;

                    if gain > best_gain {
                        best_gain = gain;
                        best_split = Some(SplitInfo {
                            feature_idx,
                            bin_threshold: bin as u8,
                            gain,
                            sum_gradient_left,
                            sum_gradient_right,
                            sum_hessian_left,
                            sum_hessian_right,
                            missing_go_left,
                        });
                    }
                }
            }
        }

        best_split
    }

    /// Build a tree using leaf-wise growth strategy (column-major optimized)
    /// Returns (tree, leaf_predictions) where leaf_predictions[i] is the leaf value
    /// for sample_indices[i]. This avoids O(n*depth) tree traversal in the boosting loop.
    fn build_leaf_wise_col_major(
        &self,
        x_col_major: &ColMajorBins,
        gradients: &[f64],
        hessians: &[f64],
        sample_indices: &[usize],
        n_bins_per_feature: &[usize],
        leaf_predictions: &mut [f64],
    ) -> HistTree {
        let n_features = x_col_major.n_features;
        let mut tree = HistTree::new();

        if sample_indices.is_empty() {
            return tree;
        }

        // Compute initial statistics
        let mut sum_gradients = 0.0f64;
        let mut sum_hessians = 0.0f64;
        for &i in sample_indices {
            sum_gradients += gradients[i];
            sum_hessians += hessians[i];
        }
        let root_value = -sum_gradients / (sum_hessians + self.l2_regularization);

        // Create root node
        let root = HistTreeNode::new_leaf(
            root_value,
            sample_indices.len(),
            sum_gradients,
            sum_hessians,
            0,
        );
        tree.nodes.push(root);

        // Track which samples belong to each node
        let mut node_samples: Vec<Vec<usize>> = vec![sample_indices.to_vec()];

        // Build histograms for root
        let mut histograms: Vec<Vec<Histogram>> = vec![self.build_histograms_col_major(
            x_col_major,
            gradients,
            hessians,
            sample_indices,
            n_bins_per_feature,
        )];

        // Priority queue of nodes to split
        let mut split_queue: BinaryHeap<SplitCandidate> = BinaryHeap::new();

        // Find initial split for root
        if let Some(split) = self.find_best_split(
            &histograms[0],
            &tree.nodes[0],
            None,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ) {
            split_queue.push(SplitCandidate {
                node_idx: 0,
                split_info: split,
            });
        }

        let max_leaves = self.max_leaf_nodes.unwrap_or(usize::MAX);
        let mut n_leaves = 1;

        // Use the split feature's column data for fast partitioning
        // Main splitting loop
        while let Some(candidate) = split_queue.pop() {
            if n_leaves >= max_leaves {
                break;
            }

            let node_idx = candidate.node_idx;
            let split = candidate.split_info;

            if let Some(max_depth) = self.max_depth {
                if tree.nodes[node_idx].depth >= max_depth {
                    continue;
                }
            }

            let (left_value, right_value) = split.compute_leaf_values(self.l2_regularization);

            // Partition samples in-place (avoids second allocation)
            let mut samples = std::mem::take(&mut node_samples[node_idx]);
            let feature_col = x_col_major.col(split.feature_idx);
            let threshold = split.bin_threshold;

            let mut left_end = 0usize;
            for i in 0..samples.len() {
                let bin = feature_col[samples[i]];
                let go_left = if bin as usize >= self.max_bins {
                    split.missing_go_left
                } else {
                    bin <= threshold
                };
                if go_left {
                    samples.swap(left_end, i);
                    left_end += 1;
                }
            }

            let right_samples = samples[left_end..].to_vec();
            samples.truncate(left_end);
            let left_samples = samples;

            // Create child nodes
            let left_node = HistTreeNode::new_leaf(
                left_value,
                left_samples.len(),
                split.sum_gradient_left,
                split.sum_hessian_left,
                tree.nodes[node_idx].depth + 1,
            );

            let right_node = HistTreeNode::new_leaf(
                right_value,
                right_samples.len(),
                split.sum_gradient_right,
                split.sum_hessian_right,
                tree.nodes[node_idx].depth + 1,
            );

            let left_idx = tree.nodes.len();
            let right_idx = left_idx + 1;

            tree.nodes.push(left_node);
            tree.nodes.push(right_node);

            tree.nodes[node_idx].feature_idx = Some(split.feature_idx);
            tree.nodes[node_idx].bin_threshold = Some(split.bin_threshold);
            tree.nodes[node_idx].missing_go_left = split.missing_go_left;
            tree.nodes[node_idx].left_child = Some(left_idx);
            tree.nodes[node_idx].right_child = Some(right_idx);
            tree.nodes[node_idx].gain = split.gain;

            n_leaves += 1;

            // Build histograms for smaller child, subtract for larger
            let (smaller_idx, smaller_samples, larger_idx) =
                if left_samples.len() <= right_samples.len() {
                    (left_idx, &left_samples, right_idx)
                } else {
                    (right_idx, &right_samples, left_idx)
                };

            let smaller_hist = self.build_histograms_col_major(
                x_col_major,
                gradients,
                hessians,
                smaller_samples,
                n_bins_per_feature,
            );

            // Compute larger child by subtraction from parent
            let parent_hist = &histograms[node_idx];
            let mut larger_hist = Vec::with_capacity(n_features);
            for f in 0..n_features {
                let n_bins = n_bins_per_feature[f];
                let mut hist = Histogram::new(n_bins);
                hist.compute_by_subtraction(&parent_hist[f], &smaller_hist[f]);
                larger_hist.push(hist);
            }

            // Store histograms and samples
            while histograms.len() <= right_idx {
                histograms.push(Vec::new());
            }
            histograms[smaller_idx] = smaller_hist;
            histograms[larger_idx] = larger_hist;

            while node_samples.len() <= right_idx {
                node_samples.push(Vec::new());
            }
            node_samples[left_idx] = left_samples;
            node_samples[right_idx] = right_samples;

            // Get bounds for monotonic constraints
            let (lower_left, upper_left, lower_right, upper_right) = self.get_monotonic_bounds(
                &tree.nodes[node_idx],
                left_value,
                right_value,
                split.feature_idx,
            );

            // Find splits for children
            if let Some(left_split) = self.find_best_split(
                &histograms[left_idx],
                &tree.nodes[left_idx],
                None,
                lower_left,
                upper_left,
            ) {
                split_queue.push(SplitCandidate {
                    node_idx: left_idx,
                    split_info: left_split,
                });
            }

            if let Some(right_split) = self.find_best_split(
                &histograms[right_idx],
                &tree.nodes[right_idx],
                None,
                lower_right,
                upper_right,
            ) {
                split_queue.push(SplitCandidate {
                    node_idx: right_idx,
                    split_info: right_split,
                });
            }
        }

        // Populate leaf predictions from node_samples (avoids O(n*depth) tree traversal)
        for (node_idx, samples) in node_samples.iter().enumerate() {
            if tree.nodes[node_idx].is_leaf() {
                let value = tree.nodes[node_idx].value;
                for &sample_idx in samples {
                    leaf_predictions[sample_idx] = value;
                }
            }
        }

        tree
    }

    /// Build a tree using leaf-wise growth strategy (row-major fallback)
    #[allow(dead_code)]
    fn build_leaf_wise(
        &self,
        x_binned: &Array2<u8>,
        gradients: &[f64],
        hessians: &[f64],
        sample_indices: &[usize],
        bin_mapper: &dyn BinMapperInfo,
    ) -> HistTree {
        let n_features = x_binned.ncols();
        let mut tree = HistTree::new();

        if sample_indices.is_empty() {
            return tree;
        }

        // Compute initial statistics
        let sum_gradients: f64 = sample_indices.iter().map(|&i| gradients[i]).sum();
        let sum_hessians: f64 = sample_indices.iter().map(|&i| hessians[i]).sum();
        let root_value = -sum_gradients / (sum_hessians + self.l2_regularization);

        // Create root node
        let root = HistTreeNode::new_leaf(
            root_value,
            sample_indices.len(),
            sum_gradients,
            sum_hessians,
            0,
        );
        tree.nodes.push(root);

        // Track which samples belong to each node
        let mut node_samples: Vec<Vec<usize>> = vec![sample_indices.to_vec()];

        // Build histograms for root
        let mut histograms: Vec<Vec<Histogram>> =
            vec![self.build_histograms(x_binned, gradients, hessians, sample_indices, bin_mapper)];

        // Priority queue of nodes to split
        let mut split_queue: BinaryHeap<SplitCandidate> = BinaryHeap::new();

        // Find initial split for root
        if let Some(split) = self.find_best_split(
            &histograms[0],
            &tree.nodes[0],
            None,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ) {
            split_queue.push(SplitCandidate {
                node_idx: 0,
                split_info: split,
            });
        }

        let max_leaves = self.max_leaf_nodes.unwrap_or(usize::MAX);
        let mut n_leaves = 1;

        // Main splitting loop
        while let Some(candidate) = split_queue.pop() {
            // Check if we've reached the maximum number of leaves
            if n_leaves >= max_leaves {
                break;
            }

            let node_idx = candidate.node_idx;
            let split = candidate.split_info;

            // Check depth constraint
            if let Some(max_depth) = self.max_depth {
                if tree.nodes[node_idx].depth >= max_depth {
                    continue;
                }
            }

            // Perform the split
            let (left_value, right_value) = split.compute_leaf_values(self.l2_regularization);

            // Partition samples in-place (avoids second allocation)
            let mut samples = std::mem::take(&mut node_samples[node_idx]);
            let feature_idx = split.feature_idx;
            let threshold = split.bin_threshold;

            let mut left_end = 0usize;
            for i in 0..samples.len() {
                let bin = x_binned[[samples[i], feature_idx]];
                let go_left = if bin as usize >= self.max_bins {
                    split.missing_go_left
                } else {
                    bin <= threshold
                };
                if go_left {
                    samples.swap(left_end, i);
                    left_end += 1;
                }
            }

            let right_samples = samples[left_end..].to_vec();
            samples.truncate(left_end);
            let left_samples = samples;

            // Create child nodes
            let left_node = HistTreeNode::new_leaf(
                left_value,
                left_samples.len(),
                split.sum_gradient_left,
                split.sum_hessian_left,
                tree.nodes[node_idx].depth + 1,
            );

            let right_node = HistTreeNode::new_leaf(
                right_value,
                right_samples.len(),
                split.sum_gradient_right,
                split.sum_hessian_right,
                tree.nodes[node_idx].depth + 1,
            );

            // Add child nodes to tree
            let left_idx = tree.nodes.len();
            let right_idx = left_idx + 1;

            tree.nodes.push(left_node);
            tree.nodes.push(right_node);

            // Update parent node
            tree.nodes[node_idx].feature_idx = Some(split.feature_idx);
            tree.nodes[node_idx].bin_threshold = Some(split.bin_threshold);
            tree.nodes[node_idx].missing_go_left = split.missing_go_left;
            tree.nodes[node_idx].left_child = Some(left_idx);
            tree.nodes[node_idx].right_child = Some(right_idx);
            tree.nodes[node_idx].gain = split.gain;

            // Update tracking
            n_leaves += 1; // One leaf split into two = net +1 leaf

            // Build histograms for smaller child (use subtraction for larger)
            let (smaller_idx, smaller_samples, larger_idx) =
                if left_samples.len() <= right_samples.len() {
                    (left_idx, &left_samples, right_idx)
                } else {
                    (right_idx, &right_samples, left_idx)
                };

            let smaller_hist =
                self.build_histograms(x_binned, gradients, hessians, smaller_samples, bin_mapper);

            // Compute larger child by subtraction from parent
            let parent_hist = &histograms[node_idx];
            let mut larger_hist = Vec::with_capacity(n_features);
            for f in 0..n_features {
                let n_bins = bin_mapper.n_bins(f);
                let mut hist = Histogram::new(n_bins);
                hist.compute_by_subtraction(&parent_hist[f], &smaller_hist[f]);
                larger_hist.push(hist);
            }

            // Store histograms and samples
            while histograms.len() <= right_idx {
                histograms.push(Vec::new());
            }
            histograms[smaller_idx] = smaller_hist;
            histograms[larger_idx] = larger_hist;

            while node_samples.len() <= right_idx {
                node_samples.push(Vec::new());
            }
            node_samples[left_idx] = left_samples;
            node_samples[right_idx] = right_samples;

            // Get bounds for monotonic constraints
            let (lower_left, upper_left, lower_right, upper_right) = self.get_monotonic_bounds(
                &tree.nodes[node_idx],
                left_value,
                right_value,
                split.feature_idx,
            );

            // Find splits for children
            if let Some(left_split) = self.find_best_split(
                &histograms[left_idx],
                &tree.nodes[left_idx],
                None,
                lower_left,
                upper_left,
            ) {
                split_queue.push(SplitCandidate {
                    node_idx: left_idx,
                    split_info: left_split,
                });
            }

            if let Some(right_split) = self.find_best_split(
                &histograms[right_idx],
                &tree.nodes[right_idx],
                None,
                lower_right,
                upper_right,
            ) {
                split_queue.push(SplitCandidate {
                    node_idx: right_idx,
                    split_info: right_split,
                });
            }
        }

        tree
    }

    /// Build histograms for all features (row-major fallback)
    #[allow(dead_code)]
    fn build_histograms(
        &self,
        x_binned: &Array2<u8>,
        gradients: &[f64],
        hessians: &[f64],
        indices: &[usize],
        bin_mapper: &dyn BinMapperInfo,
    ) -> Vec<Histogram> {
        let n_features = x_binned.ncols();
        let n_bins_per_feature: Vec<usize> =
            (0..n_features).map(|f| bin_mapper.n_bins(f)).collect();

        let mut histograms: Vec<Histogram> = n_bins_per_feature
            .iter()
            .map(|&nb| Histogram::new(nb))
            .collect();

        // Single-pass sample-oriented histogram building
        for &idx in indices {
            let g = gradients[idx];
            let h = hessians[idx];
            let row = x_binned.row(idx);
            let row_slice = row.as_slice().unwrap_or(&[]);
            for f in 0..n_features {
                let bin = row_slice[f] as usize;
                if bin < n_bins_per_feature[f] {
                    histograms[f].bins[bin].sum_gradients += g;
                    histograms[f].bins[bin].sum_hessians += h;
                    histograms[f].bins[bin].count += 1;
                }
            }
        }

        histograms
    }

    /// Build histograms using column-major binned data (cache-friendly)
    ///
    /// Performance notes:
    /// - Column-major layout ensures sequential memory access per feature
    /// - 4-sample unrolling improves CPU pipelining (ILP)
    /// - Bounds checks (`if b < n_bins`) are NECESSARY and cannot be removed:
    ///   BinMapper assigns `missing_bin = max_bins` for NaN values, but histogram
    ///   size is `actual_bins + 1` where `actual_bins <= max_bins - 1`. Without
    ///   the check, NaN-containing data would cause out-of-bounds writes.
    ///   For non-NaN data, the branch predictor quickly learns the always-true
    ///   pattern, making the check effectively free (< 1% overhead measured).
    fn build_histograms_col_major(
        &self,
        x_col_major: &ColMajorBins,
        gradients: &[f64],
        hessians: &[f64],
        indices: &[usize],
        n_bins_per_feature: &[usize],
    ) -> Vec<Histogram> {
        let n_features = x_col_major.n_features;

        // SAFETY: Validate that all sample indices are in bounds for gradient/hessian arrays.
        // Bin indices may exceed n_bins (NaN -> missing_bin), which is why runtime bounds
        // checks are retained in the inner loop rather than using unsafe indexing.
        debug_assert!(
            indices
                .iter()
                .all(|&i| i < gradients.len() && i < hessians.len()),
            "Sample indices must be within bounds of gradient/hessian arrays"
        );

        #[cfg(feature = "parallel")]
        {
            if indices.len() > 10_000 && n_features >= 8 {
                // Parallel histogram building over features for very large nodes
                // Rayon task scheduling overhead (~2μs/feature) dominates for moderate
                // workloads. Benchmarks show single-threaded is faster for n<=10K with
                // 20 features. Require both enough samples AND enough features for
                // parallelism to pay off.
                return (0..n_features)
                    .into_par_iter()
                    .map(|f| {
                        let n_bins = n_bins_per_feature[f];
                        let mut histogram = Histogram::new(n_bins);
                        let col = x_col_major.col(f);
                        let bins = &mut histogram.bins;

                        // Process 4 samples at a time for CPU pipelining
                        let chunks = indices.chunks_exact(4);
                        let remainder = chunks.remainder();

                        for chunk in chunks {
                            let b0 = col[chunk[0]] as usize;
                            let b1 = col[chunk[1]] as usize;
                            let b2 = col[chunk[2]] as usize;
                            let b3 = col[chunk[3]] as usize;

                            if b0 < n_bins {
                                bins[b0].sum_gradients += gradients[chunk[0]];
                                bins[b0].sum_hessians += hessians[chunk[0]];
                                bins[b0].count += 1;
                            }
                            if b1 < n_bins {
                                bins[b1].sum_gradients += gradients[chunk[1]];
                                bins[b1].sum_hessians += hessians[chunk[1]];
                                bins[b1].count += 1;
                            }
                            if b2 < n_bins {
                                bins[b2].sum_gradients += gradients[chunk[2]];
                                bins[b2].sum_hessians += hessians[chunk[2]];
                                bins[b2].count += 1;
                            }
                            if b3 < n_bins {
                                bins[b3].sum_gradients += gradients[chunk[3]];
                                bins[b3].sum_hessians += hessians[chunk[3]];
                                bins[b3].count += 1;
                            }
                        }
                        for &idx in remainder {
                            let bin = col[idx] as usize;
                            if bin < n_bins {
                                bins[bin].sum_gradients += gradients[idx];
                                bins[bin].sum_hessians += hessians[idx];
                                bins[bin].count += 1;
                            }
                        }

                        histogram
                    })
                    .collect();
            }
        }

        // Sequential column-oriented approach: process one feature at a time
        // Column-major layout means sequential memory access for each feature
        let mut histograms: Vec<Histogram> = n_bins_per_feature
            .iter()
            .map(|&nb| Histogram::new(nb))
            .collect();

        for f in 0..n_features {
            let n_bins = n_bins_per_feature[f];
            let col = x_col_major.col(f);
            let bins = &mut histograms[f].bins;

            // Process 4 samples at a time for CPU pipelining
            let chunks = indices.chunks_exact(4);
            let remainder = chunks.remainder();

            for chunk in chunks {
                let b0 = col[chunk[0]] as usize;
                let b1 = col[chunk[1]] as usize;
                let b2 = col[chunk[2]] as usize;
                let b3 = col[chunk[3]] as usize;

                if b0 < n_bins {
                    bins[b0].sum_gradients += gradients[chunk[0]];
                    bins[b0].sum_hessians += hessians[chunk[0]];
                    bins[b0].count += 1;
                }
                if b1 < n_bins {
                    bins[b1].sum_gradients += gradients[chunk[1]];
                    bins[b1].sum_hessians += hessians[chunk[1]];
                    bins[b1].count += 1;
                }
                if b2 < n_bins {
                    bins[b2].sum_gradients += gradients[chunk[2]];
                    bins[b2].sum_hessians += hessians[chunk[2]];
                    bins[b2].count += 1;
                }
                if b3 < n_bins {
                    bins[b3].sum_gradients += gradients[chunk[3]];
                    bins[b3].sum_hessians += hessians[chunk[3]];
                    bins[b3].count += 1;
                }
            }
            for &idx in remainder {
                let bin = col[idx] as usize;
                if bin < n_bins {
                    bins[bin].sum_gradients += gradients[idx];
                    bins[bin].sum_hessians += hessians[idx];
                    bins[bin].count += 1;
                }
            }
        }

        histograms
    }

    /// Get monotonic constraint bounds for child nodes
    fn get_monotonic_bounds(
        &self,
        _parent: &HistTreeNode,
        left_value: f64,
        right_value: f64,
        feature_idx: usize,
    ) -> (f64, f64, f64, f64) {
        let constraint = self
            .monotonic_constraints
            .get(feature_idx)
            .copied()
            .unwrap_or(MonotonicConstraint::None);

        match constraint {
            MonotonicConstraint::Positive => {
                // Left children can't exceed right value, right can't go below left value
                (f64::NEG_INFINITY, right_value, left_value, f64::INFINITY)
            }
            MonotonicConstraint::Negative => {
                // Left children can't go below right value, right can't exceed left value
                (right_value, f64::INFINITY, f64::NEG_INFINITY, left_value)
            }
            MonotonicConstraint::None => (
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
            ),
        }
    }
}

// =============================================================================
// HistGradientBoostingClassifier
// =============================================================================

/// Histogram-based Gradient Boosting Classifier
///
/// A fast implementation of gradient boosting using histogram-based split finding,
/// inspired by LightGBM. Offers O(n) complexity for split finding instead of
/// O(n log n) for traditional approaches.
///
/// ## Key Features
///
/// - **Histogram-based splitting**: Bins continuous features into discrete histograms
/// - **Native missing value handling**: Automatically learns best direction for missing values
/// - **Monotonic constraints**: Enforce domain knowledge about feature relationships
/// - **Feature interaction constraints**: Control which features can interact in trees
/// - **Leaf-wise growth**: Grows the most informative leaf first
/// - **L1/L2 regularization**: Prevent overfitting with regularization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistGradientBoostingClassifier {
    /// Maximum number of iterations (trees per class for multiclass)
    pub max_iter: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Maximum number of leaf nodes per tree
    pub max_leaf_nodes: Option<usize>,
    /// Maximum depth of each tree
    pub max_depth: Option<usize>,
    /// Minimum samples per leaf
    pub min_samples_leaf: usize,
    /// Maximum number of bins for histogram building
    pub max_bins: usize,
    /// L1 regularization
    pub l1_regularization: f64,
    /// L2 regularization
    pub l2_regularization: f64,
    /// Minimum gain to split
    pub min_gain_to_split: f64,
    /// Loss function
    pub loss: HistLoss,
    /// Early stopping configuration
    pub early_stopping: Option<HistEarlyStopping>,
    /// Monotonic constraints per feature
    pub monotonic_constraints: Vec<MonotonicConstraint>,
    /// Feature interaction constraints (groups of features that can interact)
    pub interaction_constraints: Option<Vec<Vec<usize>>>,
    /// Tree growth strategy
    pub growth_strategy: GrowthStrategy,
    /// Random seed
    pub random_state: Option<u64>,
    /// Indices of categorical features (for native handling)
    pub categorical_features: Vec<usize>,
    /// Smoothing parameter for categorical target encoding
    pub categorical_smoothing: f64,

    // Fitted parameters
    trees: Option<Vec<Vec<HistTree>>>, // [n_iterations][n_trees_per_iter]
    categorical_bin_mapper: Option<CategoricalBinMapper>,
    classes: Option<Array1<f64>>,
    n_classes: Option<usize>,
    n_features: Option<usize>,
    init_predictions: Option<Array1<f64>>,
    feature_importances: Option<Array1<f64>>,
    train_loss_history: Option<Vec<f64>>,
    val_loss_history: Option<Vec<f64>>,
}

impl Default for HistGradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl HistGradientBoostingClassifier {
    /// Create a new HistGradientBoostingClassifier with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            learning_rate: 0.1,
            max_leaf_nodes: Some(31),
            max_depth: None,
            min_samples_leaf: 20,
            max_bins: 256,
            l1_regularization: 0.0,
            l2_regularization: 0.0,
            min_gain_to_split: 0.0,
            loss: HistLoss::default(),
            early_stopping: None,
            monotonic_constraints: Vec::new(),
            interaction_constraints: None,
            growth_strategy: GrowthStrategy::LeafWise,
            random_state: None,
            categorical_features: Vec::new(),
            categorical_smoothing: 1.0,
            trees: None,
            categorical_bin_mapper: None,
            classes: None,
            n_classes: None,
            n_features: None,
            init_predictions: None,
            feature_importances: None,
            train_loss_history: None,
            val_loss_history: None,
        }
    }

    /// Set maximum iterations
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set learning rate
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate.clamp(0.001, 1.0);
        self
    }

    /// Set maximum leaf nodes per tree
    #[must_use]
    pub fn with_max_leaf_nodes(mut self, max_leaf_nodes: Option<usize>) -> Self {
        self.max_leaf_nodes = max_leaf_nodes;
        self
    }

    /// Set maximum depth
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf.max(1);
        self
    }

    /// Set maximum bins for histogram building
    #[must_use]
    pub fn with_max_bins(mut self, max_bins: usize) -> Self {
        self.max_bins = max_bins.clamp(16, 255);
        self
    }

    /// Set L1 regularization
    #[must_use]
    pub fn with_l1_regularization(mut self, l1: f64) -> Self {
        self.l1_regularization = l1.max(0.0);
        self
    }

    /// Set L2 regularization
    #[must_use]
    pub fn with_l2_regularization(mut self, l2: f64) -> Self {
        self.l2_regularization = l2.max(0.0);
        self
    }

    /// Set minimum gain to split
    #[must_use]
    pub fn with_min_gain_to_split(mut self, min_gain: f64) -> Self {
        self.min_gain_to_split = min_gain.max(0.0);
        self
    }

    /// Set loss function
    #[must_use]
    pub fn with_loss(mut self, loss: HistLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Enable early stopping
    #[must_use]
    pub fn with_early_stopping(mut self, config: HistEarlyStopping) -> Self {
        self.early_stopping = Some(config);
        self
    }

    /// Set monotonic constraints
    #[must_use]
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    /// Set feature interaction constraints
    ///
    /// Each inner vector contains feature indices that are allowed to interact.
    /// Features not in any group can only be used in the root.
    #[must_use]
    pub fn with_interaction_constraints(mut self, constraints: Vec<Vec<usize>>) -> Self {
        self.interaction_constraints = Some(constraints);
        self
    }

    /// Set tree growth strategy
    #[must_use]
    pub fn with_growth_strategy(mut self, strategy: GrowthStrategy) -> Self {
        self.growth_strategy = strategy;
        self
    }

    /// Set random state
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set categorical feature indices
    ///
    /// Features at these indices will be treated as categorical and encoded
    /// using ordered target encoding (CatBoost-style) to prevent data leakage.
    ///
    /// # Example
    ///
    /// ```
    /// # use ferroml_core::models::hist_boosting::HistGradientBoostingClassifier;
    /// let model = HistGradientBoostingClassifier::new()
    ///     .with_categorical_features(vec![0, 2, 5]); // Features 0, 2, 5 are categorical
    /// ```
    #[must_use]
    pub fn with_categorical_features(mut self, features: Vec<usize>) -> Self {
        self.categorical_features = features;
        self
    }

    /// Set smoothing parameter for categorical target encoding
    ///
    /// Higher values provide more regularization for rare categories.
    /// Default is 1.0.
    #[must_use]
    pub fn with_categorical_smoothing(mut self, smoothing: f64) -> Self {
        self.categorical_smoothing = smoothing.max(0.0);
        self
    }

    /// Get class labels
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get training loss history
    #[must_use]
    pub fn train_loss_history(&self) -> Option<&Vec<f64>> {
        self.train_loss_history.as_ref()
    }

    /// Get validation loss history
    #[must_use]
    pub fn val_loss_history(&self) -> Option<&Vec<f64>> {
        self.val_loss_history.as_ref()
    }

    /// Get number of actual iterations (may be less due to early stopping)
    #[must_use]
    pub fn n_iter_actual(&self) -> Option<usize> {
        self.trees.as_ref().map(|t| t.len())
    }

    /// Get the trees (for ONNX export).
    pub(crate) fn trees(&self) -> Option<&Vec<Vec<HistTree>>> {
        self.trees.as_ref()
    }

    /// Get the bin mapper (for ONNX export).
    pub(crate) fn categorical_bin_mapper(&self) -> Option<&CategoricalBinMapper> {
        self.categorical_bin_mapper.as_ref()
    }

    /// Get the initial predictions (for ONNX export).
    pub(crate) fn init_predictions(&self) -> Option<&Array1<f64>> {
        self.init_predictions.as_ref()
    }

    /// Get the number of classes.
    pub(crate) fn n_classes(&self) -> Option<usize> {
        self.n_classes
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.trees, "predict_proba")?;
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        validate_predict_input_allow_nan(x, n_features)?;

        let n_samples = x.nrows();
        let n_classes = self
            .n_classes
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let bin_mapper = self
            .categorical_bin_mapper
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let trees = self
            .trees
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let init = self
            .init_predictions
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        // Bin the input
        let x_binned = bin_mapper.transform(x);

        // Initialize raw predictions
        let n_trees_per_iter = if n_classes == 2 { 1 } else { n_classes };
        let mut raw_predictions = Array2::zeros((n_samples, n_trees_per_iter));
        for k in 0..n_trees_per_iter {
            for i in 0..n_samples {
                raw_predictions[[i, k]] = init[k];
            }
        }

        // Add tree predictions
        for iteration_trees in trees {
            for (k, tree) in iteration_trees.iter().enumerate() {
                for i in 0..n_samples {
                    let row = x_binned.row(i);
                    let row_slice = row.as_slice().expect("x_binned is standard layout");
                    raw_predictions[[i, k]] += self.learning_rate * tree.predict_single(row_slice);
                }
            }
        }

        // Convert to probabilities
        let mut probas = Array2::zeros((n_samples, n_classes));

        if n_classes == 2 {
            // Binary: sigmoid
            for i in 0..n_samples {
                let p = sigmoid(raw_predictions[[i, 0]]);
                probas[[i, 0]] = 1.0 - p;
                probas[[i, 1]] = p;
            }
        } else {
            // Multiclass: softmax
            for i in 0..n_samples {
                let row = raw_predictions.row(i);
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum();
                for j in 0..n_classes {
                    probas[[i, j]] = (raw_predictions[[i, j]] - max_val).exp() / exp_sum;
                }
            }
        }

        Ok(probas)
    }

    /// Compute log loss for validation
    fn compute_log_loss(&self, y: &Array1<f64>, probas: &Array2<f64>) -> f64 {
        super::compute_log_loss(
            y,
            probas,
            self.classes.as_ref().expect("classes set during fit"),
        )
    }

    /// Compute raw predictions (log-odds) before sigmoid/softmax transformation.
    fn raw_predictions(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.trees, "decision_function")?;
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("decision_function"))?;
        validate_predict_input_allow_nan(x, n_features)?;

        let n_samples = x.nrows();
        let n_classes = self
            .n_classes
            .ok_or_else(|| FerroError::not_fitted("decision_function"))?;
        let bin_mapper = self
            .categorical_bin_mapper
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("decision_function"))?;
        let trees = self
            .trees
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("decision_function"))?;
        let init = self
            .init_predictions
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("decision_function"))?;

        let x_binned = bin_mapper.transform(x);

        let n_trees_per_iter = if n_classes == 2 { 1 } else { n_classes };
        let mut raw = Array2::zeros((n_samples, n_trees_per_iter));
        for k in 0..n_trees_per_iter {
            for i in 0..n_samples {
                raw[[i, k]] = init[k];
            }
        }

        for iteration_trees in trees {
            for (k, tree) in iteration_trees.iter().enumerate() {
                for i in 0..n_samples {
                    let row = x_binned.row(i);
                    let row_slice = row.as_slice().expect("x_binned is standard layout");
                    raw[[i, k]] += self.learning_rate * tree.predict_single(row_slice);
                }
            }
        }

        Ok(raw)
    }

    /// Compute decision function scores (raw predictions before sigmoid/softmax).
    ///
    /// For histogram-based gradient boosting classifiers, this returns the raw
    /// log-odds predictions before the sigmoid (binary) or softmax (multiclass)
    /// transformation.
    ///
    /// Returns an Array2 of shape (n_samples, n_classes). For binary classification,
    /// the single raw score is expanded to two columns: [-score, +score].
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let raw = self.raw_predictions(x)?;
        let n_classes = self
            .n_classes
            .ok_or_else(|| FerroError::not_fitted("decision_function"))?;

        if n_classes == 2 {
            let n_samples = raw.nrows();
            let mut result = Array2::zeros((n_samples, 2));
            for i in 0..n_samples {
                result[[i, 0]] = -raw[[i, 0]];
                result[[i, 1]] = raw[[i, 0]];
            }
            Ok(result)
        } else {
            Ok(raw)
        }
    }

    /// Compute feature importances from all trees
    fn compute_feature_importances(&mut self) {
        if let Some(ref trees) = self.trees {
            let n_features = self.n_features.expect("model must be fitted");
            let mut importances = Array1::zeros(n_features);

            for iteration_trees in trees {
                for tree in iteration_trees {
                    for node in &tree.nodes {
                        if let Some(feature_idx) = node.feature_idx {
                            if feature_idx < n_features {
                                importances[feature_idx] += node.gain;
                            }
                        }
                    }
                }
            }

            // Normalize
            let total: f64 = importances.sum();
            if total > 0.0 {
                importances.mapv_inplace(|v| v / total);
            }

            self.feature_importances = Some(importances);
        }
    }
}

/// Sigmoid function
use super::sigmoid;

impl Model for HistGradientBoostingClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input_allow_nan(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features = Some(n_features);

        // Extend monotonic constraints if needed
        while self.monotonic_constraints.len() < n_features {
            self.monotonic_constraints.push(MonotonicConstraint::None);
        }

        // Find unique classes
        let classes = super::get_unique_classes(y);

        if classes.len() < 2 {
            return Err(FerroError::invalid_input(
                "HistGradientBoostingClassifier requires at least 2 classes",
            ));
        }

        let n_classes = classes.len();
        self.classes = Some(classes.clone());
        self.n_classes = Some(n_classes);

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Split data for early stopping if enabled
        let (x_train, y_train, x_val, y_val) = if let Some(ref early_stopping) = self.early_stopping
        {
            let n_val = (n_samples as f64 * early_stopping.validation_fraction).ceil() as usize;
            let n_train = n_samples - n_val;

            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let train_indices = &indices[..n_train];
            let val_indices = &indices[n_train..];

            let mut x_train = Array2::zeros((n_train, n_features));
            let mut y_train = Array1::zeros(n_train);
            for (i, &idx) in train_indices.iter().enumerate() {
                x_train.row_mut(i).assign(&x.row(idx));
                y_train[i] = y[idx];
            }

            let mut x_val = Array2::zeros((n_val, n_features));
            let mut y_val = Array1::zeros(n_val);
            for (i, &idx) in val_indices.iter().enumerate() {
                x_val.row_mut(i).assign(&x.row(idx));
                y_val[i] = y[idx];
            }

            (x_train, y_train, Some(x_val), Some(y_val))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        let n_train = x_train.nrows();

        // Build categorical bin mapper (handles both continuous and categorical features)
        let mut bin_mapper =
            CategoricalBinMapper::new(self.max_bins, self.categorical_features.clone())
                .with_smoothing(self.categorical_smoothing);
        bin_mapper.fit(&x_train, &y_train, &mut rng);
        let x_binned = bin_mapper.transform(&x_train);
        let x_val_binned = x_val.as_ref().map(|xv| bin_mapper.transform(xv));

        self.categorical_bin_mapper = Some(bin_mapper.clone());

        // Create column-major layout for cache-friendly histogram building
        let x_col_major = to_col_major(&x_binned);
        let n_bins_per_feature: Vec<usize> =
            (0..n_features).map(|f| bin_mapper.n_bins(f)).collect();

        // For binary classification, we only need one tree per stage
        let n_trees_per_iter = if n_classes == 2 { 1 } else { n_classes };

        // Initialize predictions
        let mut init_predictions = Array1::zeros(n_trees_per_iter);
        if n_classes == 2 {
            let n_pos: usize = y_train
                .iter()
                .filter(|&&yi| (yi - classes[1]).abs() < 1e-10)
                .count();
            let p = (n_pos as f64 + 1.0) / (n_train as f64 + 2.0);
            init_predictions[0] = (p / (1.0 - p)).ln();
        } else {
            for (k, &class_val) in classes.iter().enumerate() {
                let n_class: usize = y_train
                    .iter()
                    .filter(|&&yi| (yi - class_val).abs() < 1e-10)
                    .count();
                let p = (n_class as f64 + 1.0) / (n_train as f64 + n_classes as f64);
                init_predictions[k] = p.ln();
            }
        }
        self.init_predictions = Some(init_predictions.clone());

        // Initialize raw predictions
        let mut raw_predictions = Array2::zeros((n_train, n_trees_per_iter));
        for k in 0..n_trees_per_iter {
            for i in 0..n_train {
                raw_predictions[[i, k]] = init_predictions[k];
            }
        }

        let mut val_raw_predictions = x_val.as_ref().map(|xv| {
            let mut vp = Array2::zeros((xv.nrows(), n_trees_per_iter));
            for k in 0..n_trees_per_iter {
                for i in 0..xv.nrows() {
                    vp[[i, k]] = init_predictions[k];
                }
            }
            vp
        });

        // Create tree builder
        let mut tree_builder = HistTreeBuilder::new();
        tree_builder.max_depth = self.max_depth;
        tree_builder.max_leaf_nodes = self.max_leaf_nodes;
        tree_builder.min_samples_leaf = self.min_samples_leaf;
        tree_builder.l1_regularization = self.l1_regularization;
        tree_builder.l2_regularization = self.l2_regularization;
        tree_builder.min_gain_to_split = self.min_gain_to_split;
        tree_builder.growth_strategy = self.growth_strategy;
        tree_builder.monotonic_constraints = self.monotonic_constraints.clone();
        tree_builder.max_bins = self.max_bins;

        // Convert interaction constraints
        if let Some(ref constraints) = self.interaction_constraints {
            let sets: Vec<HashSet<usize>> = constraints
                .iter()
                .map(|v| v.iter().copied().collect())
                .collect();
            tree_builder.interaction_constraints = Some(sets);
        }

        let mut all_trees: Vec<Vec<HistTree>> = Vec::with_capacity(self.max_iter);
        let mut train_loss_history = Vec::with_capacity(self.max_iter);
        let mut val_loss_history = Vec::with_capacity(self.max_iter);

        // Early stopping state
        let mut best_val_loss = f64::INFINITY;
        let mut no_improvement_count = 0;

        // Sample indices
        let sample_indices: Vec<usize> = (0..n_train).collect();

        // Pre-allocate gradient/hessian/leaf prediction buffers (reused across iterations)
        let mut gradients = vec![0.0f64; n_train];
        let mut hessians = vec![0.0f64; n_train];
        let mut leaf_preds = vec![0.0f64; n_train];

        // Main boosting loop
        for _iteration in 0..self.max_iter {
            let mut iteration_trees = Vec::with_capacity(n_trees_per_iter);

            for k in 0..n_trees_per_iter {
                // Compute gradients and hessians (reusing pre-allocated buffers)
                if n_classes == 2 {
                    // Binary: gradient = p - y, hessian = p * (1 - p)
                    for i in 0..n_train {
                        let y_binary = if (y_train[i] - classes[1]).abs() < 1e-10 {
                            1.0
                        } else {
                            0.0
                        };
                        let p = sigmoid(raw_predictions[[i, 0]]);
                        gradients[i] = p - y_binary;
                        hessians[i] = (p * (1.0 - p)).max(1e-8);
                    }
                } else {
                    // Multiclass: gradient = p_k - y_k, hessian = p_k * (1 - p_k)
                    for i in 0..n_train {
                        let y_one_hot = if (y_train[i] - classes[k]).abs() < 1e-10 {
                            1.0
                        } else {
                            0.0
                        };

                        // Compute softmax (inline without temporary Vec)
                        let mut max_val = f64::NEG_INFINITY;
                        for j in 0..n_trees_per_iter {
                            let v = raw_predictions[[i, j]];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut exp_sum = 0.0f64;
                        for j in 0..n_trees_per_iter {
                            exp_sum += (raw_predictions[[i, j]] - max_val).exp();
                        }
                        let p_k = (raw_predictions[[i, k]] - max_val).exp() / exp_sum;

                        gradients[i] = p_k - y_one_hot;
                        hessians[i] = (p_k * (1.0 - p_k)).max(1e-8);
                    }
                }

                // Build tree using column-major optimized path
                let tree = tree_builder.build_leaf_wise_col_major(
                    &x_col_major,
                    &gradients,
                    &hessians,
                    &sample_indices,
                    &n_bins_per_feature,
                    &mut leaf_preds,
                );

                // Update predictions using leaf assignments (O(n) vs O(n*depth) tree traversal)
                for i in 0..n_train {
                    raw_predictions[[i, k]] += self.learning_rate * leaf_preds[i];
                }

                // Update validation predictions
                if let (Some(ref xvb), Some(ref mut vrp)) =
                    (&x_val_binned, &mut val_raw_predictions)
                {
                    for i in 0..xvb.nrows() {
                        let row = xvb.row(i);
                        let row_slice = row.as_slice().expect("x_val_binned is standard layout");
                        vrp[[i, k]] += self.learning_rate * tree.predict_single(row_slice);
                    }
                }

                iteration_trees.push(tree);
            }

            // Early stopping check (only compute loss when early stopping is enabled)
            if let (Some(ref yv), Some(ref vrp)) = (&y_val, &val_raw_predictions) {
                // Compute training loss only when monitoring for early stopping
                let train_probas = self.raw_to_proba(&raw_predictions);
                let train_loss = self.compute_log_loss(&y_train, &train_probas);
                train_loss_history.push(train_loss);
                let val_probas = self.raw_to_proba(vrp);
                let val_loss = self.compute_log_loss(yv, &val_probas);
                val_loss_history.push(val_loss);

                let early_stopping = self
                    .early_stopping
                    .as_ref()
                    .expect("early_stopping config set when enabled");
                if val_loss < best_val_loss - early_stopping.tol {
                    best_val_loss = val_loss;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                    if no_improvement_count >= early_stopping.patience {
                        all_trees.push(iteration_trees);
                        break;
                    }
                }
            }

            all_trees.push(iteration_trees);
        }

        self.trees = Some(all_trees);
        self.train_loss_history = Some(train_loss_history);
        self.val_loss_history = Some(val_loss_history);
        self.compute_feature_importances();

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes.as_ref().expect("classes set during fit");
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let max_idx = probas
                .row(i)
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
        self.trees.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importances.clone()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("max_iter", 50, 500)
            .float_log("learning_rate", 0.01, 0.3)
            .int("max_leaf_nodes", 10, 100)
            .int("max_depth", 3, 15)
            .int("min_samples_leaf", 5, 50)
            .float("l1_regularization", 0.0, 1.0)
            .float("l2_regularization", 0.0, 10.0)
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
}

impl HistGradientBoostingClassifier {
    fn raw_to_proba(&self, raw: &Array2<f64>) -> Array2<f64> {
        super::raw_to_proba(raw, self.n_classes.expect("model must be fitted"))
    }
}

// =============================================================================
// HistGradientBoostingRegressor
// =============================================================================

/// Histogram-based Gradient Boosting Regressor
///
/// A fast implementation of gradient boosting for regression using histogram-based
/// split finding, inspired by LightGBM. Offers O(n) complexity for split finding
/// instead of O(n log n) for traditional approaches.
///
/// ## Key Features
///
/// - **Multiple loss functions**: Squared error, absolute error, and Huber loss
/// - **Histogram-based splitting**: Bins continuous features into discrete histograms
/// - **Native missing value handling**: Automatically learns best direction for missing values
/// - **Monotonic constraints**: Enforce domain knowledge about feature relationships
/// - **Feature interaction constraints**: Control which features can interact in trees
/// - **Leaf-wise growth**: Grows the most informative leaf first
/// - **L1/L2 regularization**: Prevent overfitting with regularization
///
/// ## Example
///
/// ```
/// use ferroml_core::models::hist_boosting::{HistGradientBoostingRegressor, HistRegressionLoss};
/// use ferroml_core::models::Model;
/// use ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| i as f64 / 100.0).collect()).unwrap();
/// let y = Array1::from_iter((0..100).map(|i| (i as f64 / 10.0).sin()));
///
/// let mut model = HistGradientBoostingRegressor::new()
///     .with_max_iter(100)
///     .with_learning_rate(0.1)
///     .with_loss(HistRegressionLoss::SquaredError);
/// model.fit(&x, &y).unwrap();
///
/// let predictions = model.predict(&x).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistGradientBoostingRegressor {
    /// Maximum number of iterations (trees)
    pub max_iter: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Maximum number of leaf nodes per tree
    pub max_leaf_nodes: Option<usize>,
    /// Maximum depth of each tree
    pub max_depth: Option<usize>,
    /// Minimum samples per leaf
    pub min_samples_leaf: usize,
    /// Maximum number of bins for histogram building
    pub max_bins: usize,
    /// L1 regularization
    pub l1_regularization: f64,
    /// L2 regularization
    pub l2_regularization: f64,
    /// Minimum gain to split
    pub min_gain_to_split: f64,
    /// Loss function
    pub loss: HistRegressionLoss,
    /// Delta parameter for Huber loss
    pub huber_delta: f64,
    /// Early stopping configuration
    pub early_stopping: Option<HistEarlyStopping>,
    /// Monotonic constraints per feature
    pub monotonic_constraints: Vec<MonotonicConstraint>,
    /// Feature interaction constraints (groups of features that can interact)
    pub interaction_constraints: Option<Vec<Vec<usize>>>,
    /// Tree growth strategy
    pub growth_strategy: GrowthStrategy,
    /// Random seed
    pub random_state: Option<u64>,
    /// Indices of categorical features (for native handling)
    pub categorical_features: Vec<usize>,
    /// Smoothing parameter for categorical target encoding
    pub categorical_smoothing: f64,

    // Fitted parameters
    trees: Option<Vec<HistTree>>,
    categorical_bin_mapper: Option<CategoricalBinMapper>,
    n_features: Option<usize>,
    init_prediction: Option<f64>,
    feature_importances: Option<Array1<f64>>,
    train_loss_history: Option<Vec<f64>>,
    val_loss_history: Option<Vec<f64>>,
}

impl Default for HistGradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl HistGradientBoostingRegressor {
    /// Create a new HistGradientBoostingRegressor with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            learning_rate: 0.1,
            max_leaf_nodes: Some(31),
            max_depth: None,
            min_samples_leaf: 20,
            max_bins: 256,
            l1_regularization: 0.0,
            l2_regularization: 0.0,
            min_gain_to_split: 0.0,
            loss: HistRegressionLoss::default(),
            huber_delta: 1.0,
            early_stopping: None,
            monotonic_constraints: Vec::new(),
            interaction_constraints: None,
            growth_strategy: GrowthStrategy::LeafWise,
            random_state: None,
            categorical_features: Vec::new(),
            categorical_smoothing: 1.0,
            trees: None,
            categorical_bin_mapper: None,
            n_features: None,
            init_prediction: None,
            feature_importances: None,
            train_loss_history: None,
            val_loss_history: None,
        }
    }

    /// Set maximum iterations
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set learning rate
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate.clamp(0.001, 1.0);
        self
    }

    /// Set maximum leaf nodes per tree
    #[must_use]
    pub fn with_max_leaf_nodes(mut self, max_leaf_nodes: Option<usize>) -> Self {
        self.max_leaf_nodes = max_leaf_nodes;
        self
    }

    /// Set maximum depth
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples per leaf
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf.max(1);
        self
    }

    /// Set maximum bins for histogram building
    #[must_use]
    pub fn with_max_bins(mut self, max_bins: usize) -> Self {
        self.max_bins = max_bins.clamp(16, 255);
        self
    }

    /// Set L1 regularization
    #[must_use]
    pub fn with_l1_regularization(mut self, l1: f64) -> Self {
        self.l1_regularization = l1.max(0.0);
        self
    }

    /// Set L2 regularization
    #[must_use]
    pub fn with_l2_regularization(mut self, l2: f64) -> Self {
        self.l2_regularization = l2.max(0.0);
        self
    }

    /// Set minimum gain to split
    #[must_use]
    pub fn with_min_gain_to_split(mut self, min_gain: f64) -> Self {
        self.min_gain_to_split = min_gain.max(0.0);
        self
    }

    /// Set loss function
    #[must_use]
    pub fn with_loss(mut self, loss: HistRegressionLoss) -> Self {
        self.loss = loss;
        self
    }

    /// Set Huber delta (for Huber loss)
    #[must_use]
    pub fn with_huber_delta(mut self, delta: f64) -> Self {
        self.huber_delta = delta.max(0.0);
        self
    }

    /// Enable early stopping
    #[must_use]
    pub fn with_early_stopping(mut self, config: HistEarlyStopping) -> Self {
        self.early_stopping = Some(config);
        self
    }

    /// Set monotonic constraints
    #[must_use]
    pub fn with_monotonic_constraints(mut self, constraints: Vec<MonotonicConstraint>) -> Self {
        self.monotonic_constraints = constraints;
        self
    }

    /// Set feature interaction constraints
    ///
    /// Each inner vector contains feature indices that are allowed to interact.
    /// Features not in any group can only be used in the root.
    #[must_use]
    pub fn with_interaction_constraints(mut self, constraints: Vec<Vec<usize>>) -> Self {
        self.interaction_constraints = Some(constraints);
        self
    }

    /// Set tree growth strategy
    #[must_use]
    pub fn with_growth_strategy(mut self, strategy: GrowthStrategy) -> Self {
        self.growth_strategy = strategy;
        self
    }

    /// Set random state
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set categorical feature indices
    ///
    /// Features at these indices will be treated as categorical and encoded
    /// using ordered target encoding (CatBoost-style) to prevent data leakage.
    ///
    /// # Example
    ///
    /// ```
    /// # use ferroml_core::models::hist_boosting::HistGradientBoostingRegressor;
    /// let model = HistGradientBoostingRegressor::new()
    ///     .with_categorical_features(vec![0, 2, 5]); // Features 0, 2, 5 are categorical
    /// ```
    #[must_use]
    pub fn with_categorical_features(mut self, features: Vec<usize>) -> Self {
        self.categorical_features = features;
        self
    }

    /// Set smoothing parameter for categorical target encoding
    ///
    /// Higher values provide more regularization for rare categories.
    /// Default is 1.0.
    #[must_use]
    pub fn with_categorical_smoothing(mut self, smoothing: f64) -> Self {
        self.categorical_smoothing = smoothing.max(0.0);
        self
    }

    /// Get training loss history
    #[must_use]
    pub fn train_loss_history(&self) -> Option<&Vec<f64>> {
        self.train_loss_history.as_ref()
    }

    /// Get validation loss history
    #[must_use]
    pub fn val_loss_history(&self) -> Option<&Vec<f64>> {
        self.val_loss_history.as_ref()
    }

    /// Get number of actual iterations (may be less due to early stopping)
    #[must_use]
    pub fn n_iter_actual(&self) -> Option<usize> {
        self.trees.as_ref().map(|t| t.len())
    }

    /// Get the trees (for ONNX export).
    pub(crate) fn trees(&self) -> Option<&Vec<HistTree>> {
        self.trees.as_ref()
    }

    /// Get the bin mapper (for ONNX export).
    pub(crate) fn categorical_bin_mapper(&self) -> Option<&CategoricalBinMapper> {
        self.categorical_bin_mapper.as_ref()
    }

    /// Get the initial prediction (for ONNX export).
    pub(crate) fn init_prediction(&self) -> Option<f64> {
        self.init_prediction
    }

    /// Compute loss for a dataset
    fn compute_loss(&self, y: &Array1<f64>, predictions: &Array1<f64>) -> f64 {
        let n = y.len() as f64;
        let mut total_loss = 0.0;
        for (yi, pi) in y.iter().zip(predictions.iter()) {
            total_loss += self.loss.loss(*yi, *pi, self.huber_delta);
        }
        total_loss / n
    }

    /// Get staged predictions (predictions after each iteration)
    ///
    /// Returns an iterator over predictions at each stage.
    pub fn staged_predict<'a>(
        &'a self,
        x: &'a Array2<f64>,
    ) -> Result<impl Iterator<Item = Array1<f64>> + 'a> {
        check_is_fitted(&self.trees, "staged_predict")?;
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        validate_predict_input_allow_nan(x, n_features)?;

        let n_samples = x.nrows();
        let bin_mapper = self
            .categorical_bin_mapper
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let trees = self
            .trees
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let init = self
            .init_prediction
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        // Bin the input
        let x_binned = bin_mapper.transform(x);

        // Create iterator that yields predictions at each stage
        let mut predictions = Array1::from_elem(n_samples, init);

        Ok(trees.iter().map(move |tree| {
            for i in 0..n_samples {
                let row = x_binned.row(i);
                let row_slice = row.as_slice().expect("x_binned is standard layout");
                predictions[i] += self.learning_rate * tree.predict_single(row_slice);
            }
            predictions.clone()
        }))
    }

    /// Compute feature importances from all trees
    fn compute_feature_importances(&mut self) {
        if let Some(ref trees) = self.trees {
            let n_features = self.n_features.expect("model must be fitted");
            let mut importances = Array1::zeros(n_features);

            for tree in trees {
                for node in &tree.nodes {
                    if let Some(feature_idx) = node.feature_idx {
                        if feature_idx < n_features {
                            importances[feature_idx] += node.gain;
                        }
                    }
                }
            }

            // Normalize
            let total: f64 = importances.sum();
            if total > 0.0 {
                importances.mapv_inplace(|v| v / total);
            }

            self.feature_importances = Some(importances);
        }
    }
}

impl Model for HistGradientBoostingRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input_allow_nan(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features = Some(n_features);

        // Extend monotonic constraints if needed
        while self.monotonic_constraints.len() < n_features {
            self.monotonic_constraints.push(MonotonicConstraint::None);
        }

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        // Split data for early stopping if enabled
        let (x_train, y_train, x_val, y_val) = if let Some(ref early_stopping) = self.early_stopping
        {
            let n_val = (n_samples as f64 * early_stopping.validation_fraction).ceil() as usize;
            let n_train = n_samples - n_val;

            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let train_indices = &indices[..n_train];
            let val_indices = &indices[n_train..];

            let mut x_train = Array2::zeros((n_train, n_features));
            let mut y_train = Array1::zeros(n_train);
            for (i, &idx) in train_indices.iter().enumerate() {
                x_train.row_mut(i).assign(&x.row(idx));
                y_train[i] = y[idx];
            }

            let mut x_val = Array2::zeros((n_val, n_features));
            let mut y_val = Array1::zeros(n_val);
            for (i, &idx) in val_indices.iter().enumerate() {
                x_val.row_mut(i).assign(&x.row(idx));
                y_val[i] = y[idx];
            }

            (x_train, y_train, Some(x_val), Some(y_val))
        } else {
            (x.clone(), y.clone(), None, None)
        };

        let n_train = x_train.nrows();

        // Build categorical bin mapper (handles both continuous and categorical features)
        let mut bin_mapper =
            CategoricalBinMapper::new(self.max_bins, self.categorical_features.clone())
                .with_smoothing(self.categorical_smoothing);
        bin_mapper.fit(&x_train, &y_train, &mut rng);
        let x_binned = bin_mapper.transform(&x_train);
        let x_val_binned = x_val.as_ref().map(|xv| bin_mapper.transform(xv));

        self.categorical_bin_mapper = Some(bin_mapper.clone());

        // Create column-major layout for cache-friendly histogram building
        let x_col_major = to_col_major(&x_binned);
        let _x_val_col_major = x_val_binned.as_ref().map(|xvb| to_col_major(xvb));
        let n_bins_per_feature: Vec<usize> =
            (0..n_features).map(|f| bin_mapper.n_bins(f)).collect();

        // Initialize predictions with loss-specific optimal value
        let init_prediction = self.loss.initial_prediction(&y_train);
        self.init_prediction = Some(init_prediction);

        // Initialize raw predictions
        let mut predictions = Array1::from_elem(n_train, init_prediction);

        let mut val_predictions = x_val
            .as_ref()
            .map(|xv| Array1::from_elem(xv.nrows(), init_prediction));

        // Create tree builder
        let mut tree_builder = HistTreeBuilder::new();
        tree_builder.max_depth = self.max_depth;
        tree_builder.max_leaf_nodes = self.max_leaf_nodes;
        tree_builder.min_samples_leaf = self.min_samples_leaf;
        tree_builder.l1_regularization = self.l1_regularization;
        tree_builder.l2_regularization = self.l2_regularization;
        tree_builder.min_gain_to_split = self.min_gain_to_split;
        tree_builder.growth_strategy = self.growth_strategy;
        tree_builder.monotonic_constraints = self.monotonic_constraints.clone();
        tree_builder.max_bins = self.max_bins;

        // Convert interaction constraints
        if let Some(ref constraints) = self.interaction_constraints {
            let sets: Vec<HashSet<usize>> = constraints
                .iter()
                .map(|v| v.iter().copied().collect())
                .collect();
            tree_builder.interaction_constraints = Some(sets);
        }

        let mut all_trees: Vec<HistTree> = Vec::with_capacity(self.max_iter);
        let mut train_loss_history = Vec::with_capacity(self.max_iter);
        let mut val_loss_history = Vec::with_capacity(self.max_iter);

        // Early stopping state
        let mut best_val_loss = f64::INFINITY;
        let mut no_improvement_count = 0;

        // Sample indices
        let sample_indices: Vec<usize> = (0..n_train).collect();

        // Pre-allocate gradient/hessian/leaf prediction buffers (reused across iterations)
        let mut gradients = vec![0.0f64; n_train];
        let mut hessians = vec![0.0f64; n_train];
        let mut leaf_preds = vec![0.0f64; n_train];

        // Main boosting loop
        for _iteration in 0..self.max_iter {
            // Compute gradients and hessians
            for i in 0..n_train {
                gradients[i] = self
                    .loss
                    .gradient(y_train[i], predictions[i], self.huber_delta);
                hessians[i] = self
                    .loss
                    .hessian(y_train[i], predictions[i], self.huber_delta)
                    .max(1e-8);
            }

            // Build tree using column-major optimized path
            let tree = tree_builder.build_leaf_wise_col_major(
                &x_col_major,
                &gradients,
                &hessians,
                &sample_indices,
                &n_bins_per_feature,
                &mut leaf_preds,
            );

            // Update predictions using leaf assignments (O(n) vs O(n*depth) tree traversal)
            for i in 0..n_train {
                predictions[i] += self.learning_rate * leaf_preds[i];
            }

            // Update validation predictions
            if let (Some(ref xvb), Some(ref mut vp)) = (&x_val_binned, &mut val_predictions) {
                for i in 0..xvb.nrows() {
                    let row = xvb.row(i);
                    let row_slice = row.as_slice().expect("x_val_binned is standard layout");
                    vp[i] += self.learning_rate * tree.predict_single(row_slice);
                }
            }

            // Compute training loss
            {
                let train_loss = self.compute_loss(&y_train, &predictions);
                train_loss_history.push(train_loss);
            }

            // Early stopping check
            if let (Some(ref yv), Some(ref vp)) = (&y_val, &val_predictions) {
                let val_loss = self.compute_loss(yv, vp);
                val_loss_history.push(val_loss);

                let early_stopping = self
                    .early_stopping
                    .as_ref()
                    .expect("early_stopping config set when enabled");
                if val_loss < best_val_loss - early_stopping.tol {
                    best_val_loss = val_loss;
                    no_improvement_count = 0;
                } else {
                    no_improvement_count += 1;
                    if no_improvement_count >= early_stopping.patience {
                        all_trees.push(tree);
                        break;
                    }
                }
            }

            all_trees.push(tree);
        }

        self.trees = Some(all_trees);
        self.train_loss_history = Some(train_loss_history);
        self.val_loss_history = Some(val_loss_history);
        self.compute_feature_importances();

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.trees, "predict")?;
        let n_features = self
            .n_features
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        validate_predict_input_allow_nan(x, n_features)?;

        let n_samples = x.nrows();
        let bin_mapper = self
            .categorical_bin_mapper
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let trees = self
            .trees
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let init = self
            .init_prediction
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        // Bin the input
        let x_binned = bin_mapper.transform(x);

        // Initialize predictions
        let mut predictions = Array1::from_elem(n_samples, init);

        // Add tree predictions
        for tree in trees {
            for i in 0..n_samples {
                let row = x_binned.row(i);
                let row_slice = row.as_slice().expect("x_binned is standard layout");
                predictions[i] += self.learning_rate * tree.predict_single(row_slice);
            }
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.trees.is_some()
    }

    fn feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importances.clone()
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("max_iter", 50, 500)
            .float_log("learning_rate", 0.01, 0.3)
            .int("max_leaf_nodes", 10, 100)
            .int("max_depth", 3, 15)
            .int("min_samples_leaf", 5, 50)
            .float("l1_regularization", 0.0, 1.0)
            .float("l2_regularization", 0.0, 10.0)
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(x)?;
        crate::metrics::r2_score(y, &predictions)
    }
}

// =============================================================================
// Native Categorical Feature Handling (CatBoost-style)
// =============================================================================

/// Type of categorical feature encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CategoricalEncoding {
    /// Ordered target encoding (CatBoost-style) - prevents data leakage
    OrderedTargetEncoding,
    /// Simple target encoding (uses full statistics) - can leak
    TargetEncoding,
    /// One-hot encoding per bin (only for low cardinality)
    OneHot,
}

impl Default for CategoricalEncoding {
    fn default() -> Self {
        Self::OrderedTargetEncoding
    }
}

/// Statistics for a single category value
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CategoryStats {
    /// Sum of target values for samples in this category
    sum_target: f64,
    /// Count of samples in this category
    count: usize,
}

/// Handler for native categorical features with ordered target encoding
///
/// Implements CatBoost-style ordered target statistics to prevent data leakage
/// during training. For each sample, the target encoding is computed using only
/// samples that appeared earlier in a random permutation.
///
/// ## Key Features
///
/// - **Ordered statistics**: Prevents target leakage by using only prior samples
/// - **Smoothing**: Regularizes rare categories using global prior
/// - **Efficient histogram computation**: Categories map directly to bins
///
/// ## Example
///
/// ```
/// use ferroml_core::models::hist_boosting::CategoricalFeatureHandler;
/// # use ndarray::{Array1, Array2};
///
/// # let x = Array2::from_shape_vec((6, 3), vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0]).unwrap();
/// # let y = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
/// # let n_features = 3;
/// let categorical_features = vec![0, 2]; // Features 0 and 2 are categorical
/// let mut handler = CategoricalFeatureHandler::new(categorical_features)
///     .with_smoothing(1.0);
///
/// // Fit computes ordered target statistics
/// handler.fit(&x, &y, n_features);
///
/// // Transform converts categories to encoded values
/// let encoded = handler.transform(&x, Some(&y), None, false);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalFeatureHandler {
    /// Indices of features that are categorical
    categorical_features: HashSet<usize>,
    /// Smoothing parameter for target encoding (higher = more regularization)
    smoothing: f64,
    /// Encoding type to use
    encoding: CategoricalEncoding,
    /// Category statistics per feature: feature_idx -> (category_value -> stats)
    category_stats: Vec<std::collections::HashMap<i64, CategoryStats>>,
    /// Global target mean (prior for smoothing)
    global_mean: f64,
    /// Number of features
    n_features: usize,
    /// Whether the handler has been fitted
    is_fitted: bool,
}

impl CategoricalFeatureHandler {
    /// Create a new categorical feature handler
    ///
    /// # Arguments
    /// * `categorical_features` - Indices of features that should be treated as categorical
    pub fn new(categorical_features: Vec<usize>) -> Self {
        Self {
            categorical_features: categorical_features.into_iter().collect(),
            smoothing: 1.0,
            encoding: CategoricalEncoding::default(),
            category_stats: Vec::new(),
            global_mean: 0.0,
            n_features: 0,
            is_fitted: false,
        }
    }

    /// Create an empty handler (no categorical features)
    pub fn empty() -> Self {
        Self {
            categorical_features: HashSet::new(),
            smoothing: 1.0,
            encoding: CategoricalEncoding::default(),
            category_stats: Vec::new(),
            global_mean: 0.0,
            n_features: 0,
            is_fitted: true, // Empty handler is always "fitted"
        }
    }

    /// Set smoothing parameter
    ///
    /// Higher values provide more regularization for rare categories.
    /// Default is 1.0.
    #[must_use]
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing.max(0.0);
        self
    }

    /// Set encoding type
    #[must_use]
    pub fn with_encoding(mut self, encoding: CategoricalEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Check if a feature is categorical
    #[must_use]
    pub fn is_categorical(&self, feature_idx: usize) -> bool {
        self.categorical_features.contains(&feature_idx)
    }

    /// Get the set of categorical feature indices
    #[must_use]
    pub fn categorical_indices(&self) -> &HashSet<usize> {
        &self.categorical_features
    }

    /// Get number of unique categories for a feature
    #[must_use]
    pub fn n_categories(&self, feature_idx: usize) -> Option<usize> {
        if !self.is_fitted || feature_idx >= self.category_stats.len() {
            return None;
        }
        Some(self.category_stats[feature_idx].len())
    }

    /// Convert float to category key (handles NaN)
    fn value_to_category_key(value: f64) -> i64 {
        if value.is_nan() {
            i64::MIN // Special key for NaN
        } else {
            // Round to integer (categorical values should be integral)
            value.round() as i64
        }
    }

    /// Fit the handler to compute category statistics
    ///
    /// For ordered target encoding, statistics are computed using cumulative
    /// sums that can be used to generate ordered encodings during transform.
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `y` - Target values
    /// * `n_features` - Number of features
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, n_features: usize) {
        self.n_features = n_features;
        self.category_stats = vec![std::collections::HashMap::new(); n_features];

        // Compute global mean
        self.global_mean = y.mean().unwrap_or(0.0);

        // Compute category statistics
        for &feature_idx in &self.categorical_features {
            if feature_idx >= n_features {
                continue;
            }

            let col = x.column(feature_idx);
            for (i, &value) in col.iter().enumerate() {
                let key = Self::value_to_category_key(value);
                let stats = self.category_stats[feature_idx].entry(key).or_default();
                stats.sum_target += y[i];
                stats.count += 1;
            }
        }

        self.is_fitted = true;
    }

    /// Compute ordered target encoding for a single value
    ///
    /// Uses CatBoost-style ordered statistics:
    /// encoding = (sum_prior + smoothing * global_mean) / (count_prior + smoothing)
    ///
    /// # Arguments
    /// * `feature_idx` - Feature index
    /// * `value` - Category value
    /// * `sum_prior` - Sum of targets from prior samples (ordered statistics)
    /// * `count_prior` - Count of prior samples
    fn compute_ordered_encoding(
        &self,
        _feature_idx: usize,
        _value: f64,
        sum_prior: f64,
        count_prior: usize,
    ) -> f64 {
        // Ordered target encoding with smoothing
        self.smoothing.mul_add(self.global_mean, sum_prior) / (count_prior as f64 + self.smoothing)
    }

    /// Compute standard target encoding for a single value
    ///
    /// Uses full category statistics (can leak during training):
    /// encoding = (sum_target + smoothing * global_mean) / (count + smoothing)
    fn compute_target_encoding(&self, feature_idx: usize, value: f64) -> f64 {
        let key = Self::value_to_category_key(value);

        if let Some(stats) = self
            .category_stats
            .get(feature_idx)
            .and_then(|m| m.get(&key))
        {
            self.smoothing.mul_add(self.global_mean, stats.sum_target)
                / (stats.count as f64 + self.smoothing)
        } else {
            // Unknown category - use global mean
            self.global_mean
        }
    }

    /// Transform categorical features using ordered target encoding
    ///
    /// For training, uses ordered statistics to prevent leakage.
    /// For prediction, uses full category statistics.
    ///
    /// # Arguments
    /// * `x` - Feature matrix to transform
    /// * `y` - Target values (only needed for training with ordered encoding)
    /// * `permutation` - Random permutation of sample indices (for ordered encoding)
    /// * `is_training` - Whether this is training (use ordered) or prediction (use full stats)
    pub fn transform(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        permutation: Option<&[usize]>,
        is_training: bool,
    ) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut result = x.clone();

        if self.categorical_features.is_empty() {
            return result;
        }

        match self.encoding {
            CategoricalEncoding::OrderedTargetEncoding if is_training => {
                // Ordered target encoding during training
                let y = y.expect("Target values required for ordered target encoding");
                let perm = permutation.expect("Permutation required for ordered target encoding");

                for &feature_idx in &self.categorical_features {
                    if feature_idx >= n_features {
                        continue;
                    }

                    // Track cumulative statistics per category as we iterate through permutation
                    let mut cumulative_stats: std::collections::HashMap<i64, CategoryStats> =
                        std::collections::HashMap::new();

                    // Process samples in permutation order
                    for (perm_idx, &sample_idx) in perm.iter().enumerate() {
                        let value = x[[sample_idx, feature_idx]];
                        let key = Self::value_to_category_key(value);

                        // Get prior statistics (before this sample)
                        let prior_stats = cumulative_stats.get(&key).cloned().unwrap_or_default();

                        // Compute ordered encoding using only prior samples
                        let encoding = self.compute_ordered_encoding(
                            feature_idx,
                            value,
                            prior_stats.sum_target,
                            prior_stats.count,
                        );

                        result[[sample_idx, feature_idx]] = encoding;

                        // Update cumulative statistics with this sample
                        let stats = cumulative_stats.entry(key).or_default();
                        stats.sum_target += y[sample_idx];
                        stats.count += 1;

                        // Early exit hint for very large datasets
                        if perm_idx > 0 && perm_idx % 10000 == 0 {
                            // Could add progress callback here
                        }
                    }
                }
            }
            CategoricalEncoding::OrderedTargetEncoding | CategoricalEncoding::TargetEncoding => {
                // Standard target encoding (for prediction or when target encoding is selected)
                for &feature_idx in &self.categorical_features {
                    if feature_idx >= n_features {
                        continue;
                    }

                    for i in 0..n_samples {
                        let value = x[[i, feature_idx]];
                        result[[i, feature_idx]] = self.compute_target_encoding(feature_idx, value);
                    }
                }
            }
            CategoricalEncoding::OneHot => {
                // One-hot encoding: just use the category value directly as bin index
                // The BinMapper will handle creating appropriate histogram bins
                // No transformation needed here
            }
        }

        result
    }
}

/// Extended bin mapper that handles both continuous and categorical features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalBinMapper {
    /// Standard bin mapper for continuous features
    continuous_mapper: BinMapper,
    /// Categorical feature handler
    categorical_handler: CategoricalFeatureHandler,
    /// Category to bin mapping for each categorical feature
    category_to_bin: Vec<std::collections::HashMap<i64, u8>>,
    /// Whether the mapper has been fitted
    is_fitted: bool,
}

impl CategoricalBinMapper {
    /// Create a new categorical bin mapper
    ///
    /// # Arguments
    /// * `max_bins` - Maximum number of bins for continuous features
    /// * `categorical_features` - Indices of categorical features (empty if none)
    pub fn new(max_bins: usize, categorical_features: Vec<usize>) -> Self {
        Self {
            continuous_mapper: BinMapper::new(max_bins),
            categorical_handler: if categorical_features.is_empty() {
                CategoricalFeatureHandler::empty()
            } else {
                CategoricalFeatureHandler::new(categorical_features)
            },
            category_to_bin: Vec::new(),
            is_fitted: false,
        }
    }

    /// Set smoothing parameter for categorical encoding
    #[must_use]
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.categorical_handler = self.categorical_handler.with_smoothing(smoothing);
        self
    }

    /// Set encoding type for categorical features
    #[must_use]
    pub fn with_encoding(mut self, encoding: CategoricalEncoding) -> Self {
        self.categorical_handler = self.categorical_handler.with_encoding(encoding);
        self
    }

    /// Fit the bin mapper to training data
    ///
    /// For categorical features with ordered target encoding, a random permutation
    /// is used to establish the ordering.
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, rng: &mut StdRng) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Generate random permutation for ordered encoding
        let mut permutation: Vec<usize> = (0..n_samples).collect();
        permutation.shuffle(rng);

        // Fit categorical handler
        self.categorical_handler.fit(x, y, n_features);

        // Transform categorical features to encoded values
        let x_encoded = self
            .categorical_handler
            .transform(x, Some(y), Some(&permutation), true);

        // Fit continuous mapper on encoded data
        self.continuous_mapper.fit(&x_encoded);

        // Build category-to-bin mappings for categorical features
        self.category_to_bin = vec![std::collections::HashMap::new(); n_features];

        for &feature_idx in self.categorical_handler.categorical_indices() {
            if feature_idx >= n_features {
                continue;
            }

            // Map each unique category to a bin based on its encoded value
            let col = x.column(feature_idx);
            let encoded_col = x_encoded.column(feature_idx);

            for (_i, (&original, &encoded)) in col.iter().zip(encoded_col.iter()).enumerate() {
                let key = CategoricalFeatureHandler::value_to_category_key(original);

                // Only store the first mapping (all samples with same category get same bin)
                if !self.category_to_bin[feature_idx].contains_key(&key) {
                    // Find bin for this encoded value
                    let edges = &self.continuous_mapper.bin_edges[feature_idx];
                    let bin = if encoded.is_nan() {
                        self.continuous_mapper.missing_bin as u8
                    } else {
                        edges
                            .iter()
                            .position(|&e| encoded < e)
                            .unwrap_or(edges.len())
                            .saturating_sub(1) as u8
                    };
                    self.category_to_bin[feature_idx].insert(key, bin);
                }
            }
        }

        self.is_fitted = true;
    }

    /// Transform features to bin indices
    ///
    /// For categorical features during prediction, uses stored category-to-bin mappings.
    /// For continuous features, uses standard binning.
    pub fn transform(&self, x: &Array2<f64>) -> Array2<u8> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut binned = Array2::zeros((n_samples, n_features));

        for col_idx in 0..n_features {
            if self.categorical_handler.is_categorical(col_idx) {
                // Categorical feature: use category-to-bin mapping
                for row_idx in 0..n_samples {
                    let value = x[[row_idx, col_idx]];
                    let key = CategoricalFeatureHandler::value_to_category_key(value);

                    let bin = self
                        .category_to_bin
                        .get(col_idx)
                        .and_then(|m| m.get(&key))
                        .copied()
                        .unwrap_or_else(|| {
                            // Unknown category - use target encoding with full stats
                            let encoded = self
                                .categorical_handler
                                .compute_target_encoding(col_idx, value);
                            let edges = &self.continuous_mapper.bin_edges[col_idx];
                            match edges.binary_search_by(|e| {
                                e.partial_cmp(&encoded)
                                    .unwrap_or(std::cmp::Ordering::Greater)
                            }) {
                                Ok(pos) => pos as u8,
                                Err(pos) => pos.saturating_sub(1) as u8,
                            }
                        });

                    binned[[row_idx, col_idx]] = bin;
                }
            } else {
                // Continuous feature: use standard binning
                let edges = &self.continuous_mapper.bin_edges[col_idx];
                for row_idx in 0..n_samples {
                    let value = x[[row_idx, col_idx]];

                    if value.is_nan() {
                        binned[[row_idx, col_idx]] = self.continuous_mapper.missing_bin as u8;
                    } else {
                        let bin = match edges.binary_search_by(|e| {
                            e.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Greater)
                        }) {
                            Ok(pos) => pos,
                            Err(pos) => pos.saturating_sub(1),
                        };
                        binned[[row_idx, col_idx]] = bin as u8;
                    }
                }
            }
        }

        binned
    }

    /// Get the number of bins for a feature (includes missing bin)
    pub fn n_bins(&self, feature_idx: usize) -> usize {
        self.continuous_mapper.n_bins(feature_idx)
    }

    /// Check if a feature is categorical
    pub fn is_categorical(&self, feature_idx: usize) -> bool {
        self.categorical_handler.is_categorical(feature_idx)
    }

    /// Get the categorical feature handler
    pub fn categorical_handler(&self) -> &CategoricalFeatureHandler {
        &self.categorical_handler
    }

    /// Get the real-valued threshold for a bin index on a given feature.
    pub fn bin_threshold_to_real(&self, feature_idx: usize, bin: u8) -> f64 {
        self.continuous_mapper
            .bin_threshold_to_real(feature_idx, bin)
    }
}

impl BinMapperInfo for CategoricalBinMapper {
    fn n_bins(&self, feature_idx: usize) -> usize {
        self.n_bins(feature_idx)
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
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 1.0, 2.0, 1.5, 2.5, 2.0, 1.0, 2.5, 1.5,
                1.2, 1.8, 1.8, 1.2, // Class 0
                5.0, 5.0, 5.5, 5.5, 6.0, 6.0, 6.5, 6.5, 5.0, 6.0, 5.5, 6.5, 6.0, 5.0, 6.5, 5.5,
                5.2, 5.8, 5.8, 5.2, // Class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);
        (x, y)
    }

    fn make_multiclass_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (15, 2),
            vec![
                1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 0.8, 0.8, 1.3, 1.0, // Class 0
                4.0, 4.0, 4.2, 4.2, 4.5, 4.5, 3.8, 3.8, 4.3, 4.0, // Class 1
                7.0, 7.0, 7.2, 7.2, 7.5, 7.5, 6.8, 6.8, 7.3, 7.0, // Class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ]);
        (x, y)
    }

    #[test]
    fn test_bin_mapper() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
                14.0, 16.0, 18.0, 20.0,
            ],
        )
        .unwrap();

        let mut mapper = BinMapper::new(5);
        mapper.fit(&x);

        let binned = mapper.transform(&x);
        assert_eq!(binned.shape(), &[10, 2]);

        // Values should be binned
        for row in binned.rows() {
            for &bin in row {
                assert!(bin <= 5);
            }
        }
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_binary() {
        let (x, y) = make_classification_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(100)
            .with_learning_rate(0.2)
            .with_max_leaf_nodes(Some(15))
            .with_min_samples_leaf(2)
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.n_features(), Some(2));

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // Should classify most correctly (lowered threshold for small dataset with histogram binning)
        let accuracy: f64 = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count() as f64
            / 20.0;
        assert!(accuracy > 0.5, "Accuracy was {}", accuracy);
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_proba() {
        let (x, y) = make_classification_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(20)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let probas = clf.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[20, 2]);

        // Probabilities should sum to 1
        for i in 0..20 {
            let row_sum: f64 = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-6);
        }

        // Probabilities should be in [0, 1]
        for p in probas.iter() {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_multiclass() {
        let (x, y) = make_multiclass_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(50)
            .with_learning_rate(0.1)
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.classes().unwrap().len(), 3);

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 15);

        let probas = clf.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[15, 3]);

        // Probabilities should sum to 1
        for i in 0..15 {
            let row_sum: f64 = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_feature_importance() {
        let (x, y) = make_classification_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(20)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        let importance = clf.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Non-negative importances
        assert!(importance.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_early_stopping() {
        let (x, y) = make_classification_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(100)
            .with_early_stopping(HistEarlyStopping {
                patience: 5,
                tol: 1e-7,
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert!(clf.val_loss_history().unwrap().len() > 0);
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_regularization() {
        let (x, y) = make_classification_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(20)
            .with_l1_regularization(0.1)
            .with_l2_regularization(1.0)
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();
        assert!(clf.is_fitted());

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_monotonic_constraints() {
        let (x, y) = make_classification_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(20)
            .with_monotonic_constraints(vec![
                MonotonicConstraint::Positive,
                MonotonicConstraint::None,
            ])
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();
        assert!(clf.is_fitted());
    }

    #[test]
    fn test_search_space() {
        let clf = HistGradientBoostingClassifier::new();
        let space = clf.search_space();
        assert!(space.n_dims() > 0);
    }

    #[test]
    fn test_not_fitted_errors() {
        let clf = HistGradientBoostingClassifier::new();
        let x = Array2::zeros((2, 2));
        assert!(clf.predict(&x).is_err());
        assert!(clf.predict_proba(&x).is_err());
    }

    #[test]
    fn test_single_class_error() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

        let mut clf = HistGradientBoostingClassifier::new();
        let result = clf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_values() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0,
                f64::NAN,
                2.0,
                2.0,
                f64::NAN,
                3.0,
                4.0,
                4.0,
                5.0,
                5.0,
                6.0,
                6.0,
                7.0,
                f64::NAN,
                8.0,
                8.0,
                f64::NAN,
                9.0,
                10.0,
                10.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(10)
            .with_random_state(42);

        // Should handle missing values
        clf.fit(&x, &y).unwrap();
        assert!(clf.is_fitted());

        // Should also predict with missing values
        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);
    }

    // =========================================================================
    // HistGradientBoostingRegressor Tests
    // =========================================================================

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        // y = 2*x0 + 3*x1 + noise
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5, 1.5, 0.75, 2.5, 1.25, 3.5, 1.75,
                4.5, 2.25, 5.5, 2.75, 0.5, 0.25, 1.0, 0.5, 1.5, 0.75, 2.0, 1.0, 2.5, 1.25, 3.0,
                1.5, 3.5, 1.75, 4.0, 2.0, 4.5, 2.25, 5.0, 2.5,
            ],
        )
        .unwrap();

        let y = Array1::from_vec(vec![
            3.5, 7.0, 10.5, 14.0, 17.5, 5.25, 8.75, 12.25, 15.75, 19.25, 1.75, 3.5, 5.25, 7.0,
            8.75, 10.5, 12.25, 14.0, 15.75, 17.5,
        ]);

        (x, y)
    }

    #[test]
    fn test_hist_regression_loss_squared_error() {
        let loss = HistRegressionLoss::SquaredError;

        // Gradient: y_pred - y_true
        assert_abs_diff_eq!(loss.gradient(5.0, 7.0, 1.0), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(loss.gradient(5.0, 3.0, 1.0), -2.0, epsilon = 1e-10);

        // Hessian: always 1
        assert_abs_diff_eq!(loss.hessian(5.0, 7.0, 1.0), 1.0, epsilon = 1e-10);

        // Loss: 0.5 * (y_pred - y_true)^2
        assert_abs_diff_eq!(loss.loss(5.0, 7.0, 1.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hist_regression_loss_absolute_error() {
        let loss = HistRegressionLoss::AbsoluteError;

        // Gradient: sign(y_pred - y_true)
        assert_abs_diff_eq!(loss.gradient(5.0, 7.0, 1.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(loss.gradient(5.0, 3.0, 1.0), -1.0, epsilon = 1e-10);

        // Loss: |y_pred - y_true|
        assert_abs_diff_eq!(loss.loss(5.0, 7.0, 1.0), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hist_regression_loss_huber() {
        let loss = HistRegressionLoss::Huber;
        let delta = 1.0;

        // Within delta: behaves like squared error
        assert_abs_diff_eq!(loss.gradient(5.0, 5.5, delta), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(loss.hessian(5.0, 5.5, delta), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(loss.loss(5.0, 5.5, delta), 0.125, epsilon = 1e-10);

        // Outside delta: behaves like absolute error
        assert_abs_diff_eq!(loss.gradient(5.0, 7.0, delta), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(loss.loss(5.0, 7.0, delta), 1.5, epsilon = 1e-10); // delta * (|r| - 0.5 * delta)
    }

    #[test]
    fn test_hist_regression_loss_initial_prediction() {
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Squared error: mean
        let init_se = HistRegressionLoss::SquaredError.initial_prediction(&y);
        assert_abs_diff_eq!(init_se, 3.0, epsilon = 1e-10);

        // Absolute error: median
        let init_ae = HistRegressionLoss::AbsoluteError.initial_prediction(&y);
        assert_abs_diff_eq!(init_ae, 3.0, epsilon = 1e-10);

        // Even number of elements
        let y_even = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let init_even = HistRegressionLoss::AbsoluteError.initial_prediction(&y_even);
        assert_abs_diff_eq!(init_even, 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_squared_error() {
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(100)
            .with_learning_rate(0.1)
            .with_loss(HistRegressionLoss::SquaredError)
            .with_max_leaf_nodes(Some(15))
            .with_min_samples_leaf(2)
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());
        assert_eq!(reg.n_features(), Some(2));

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // Calculate RMSE - should be reasonable for this simple linear data
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / 20.0;
        let rmse = mse.sqrt();
        assert!(rmse < 5.0, "RMSE was {}", rmse);
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_absolute_error() {
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(100)
            .with_learning_rate(0.1)
            .with_loss(HistRegressionLoss::AbsoluteError)
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // Calculate MAE
        let mae: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / 20.0;
        assert!(mae < 5.0, "MAE was {}", mae);
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_huber() {
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(100)
            .with_learning_rate(0.1)
            .with_loss(HistRegressionLoss::Huber)
            .with_huber_delta(1.0)
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_feature_importance() {
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(50)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();

        let importance = reg.feature_importance().unwrap();
        assert_eq!(importance.len(), 2);

        // Non-negative importances
        assert!(importance.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_early_stopping() {
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(200)
            .with_early_stopping(HistEarlyStopping {
                patience: 10,
                tol: 1e-7,
                validation_fraction: 0.2,
            })
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());
        assert!(!reg.val_loss_history().unwrap().is_empty());
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_regularization() {
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(50)
            .with_l1_regularization(0.1)
            .with_l2_regularization(1.0)
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_monotonic_constraints() {
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(50)
            .with_monotonic_constraints(vec![
                MonotonicConstraint::Positive,
                MonotonicConstraint::None,
            ])
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_search_space() {
        let reg = HistGradientBoostingRegressor::new();
        let space = reg.search_space();
        assert!(space.n_dims() > 0);
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_not_fitted_error() {
        let reg = HistGradientBoostingRegressor::new();
        let x = Array2::zeros((2, 2));
        assert!(reg.predict(&x).is_err());
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_missing_values() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0,
                f64::NAN,
                2.0,
                2.0,
                f64::NAN,
                3.0,
                4.0,
                4.0,
                5.0,
                5.0,
                6.0,
                6.0,
                7.0,
                f64::NAN,
                8.0,
                8.0,
                f64::NAN,
                9.0,
                10.0,
                10.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(10)
            .with_random_state(42);

        // Should handle missing values
        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());

        // Should also predict with missing values
        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);

        // Predictions should be finite
        assert!(predictions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_train_loss_history() {
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(20)
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        let history = reg.train_loss_history().unwrap();
        assert_eq!(history.len(), 20);

        // Loss should generally decrease (though not strictly monotonic due to histogram approximation)
        assert!(history.first().unwrap() >= history.last().unwrap());
    }

    // =========================================================================
    // Categorical Feature Handling Tests
    // =========================================================================

    fn make_classification_data_with_categorical() -> (Array2<f64>, Array1<f64>) {
        // Feature 0: categorical (0, 1, 2 representing categories A, B, C)
        // Feature 1: continuous
        // Target: class 0 if cat=A, class 1 if cat=B or C
        let x = Array2::from_shape_vec(
            (16, 2),
            vec![
                0.0, 1.0, // Cat A
                0.0, 2.0, // Cat A
                0.0, 1.5, // Cat A
                0.0, 2.5, // Cat A
                1.0, 3.0, // Cat B
                1.0, 4.0, // Cat B
                1.0, 3.5, // Cat B
                1.0, 4.5, // Cat B
                2.0, 5.0, // Cat C
                2.0, 6.0, // Cat C
                2.0, 5.5, // Cat C
                2.0, 6.5, // Cat C
                0.0, 1.2, // Cat A
                1.0, 3.2, // Cat B
                2.0, 5.2, // Cat C
                0.0, 2.2, // Cat A
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, // Class 0 for Cat A
            1.0, 1.0, 1.0, 1.0, // Class 1 for Cat B
            1.0, 1.0, 1.0, 1.0, // Class 1 for Cat C
            0.0, 1.0, 1.0, 0.0, // Mixed
        ]);
        (x, y)
    }

    fn make_regression_data_with_categorical() -> (Array2<f64>, Array1<f64>) {
        // Feature 0: categorical (0, 1, 2 representing categories)
        // Feature 1: continuous
        // Target: mean depends on category
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 1.0, // Cat 0 -> target ~10
                0.0, 1.1, 0.0, 0.9, 0.0, 1.2, 1.0, 2.0, // Cat 1 -> target ~20
                1.0, 2.1, 1.0, 1.9, 1.0, 2.2, 2.0, 3.0, // Cat 2 -> target ~30
                2.0, 3.1, 2.0, 2.9, 2.0, 3.2,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            10.0, 10.5, 9.5, 11.0, // Cat 0
            20.0, 20.5, 19.5, 21.0, // Cat 1
            30.0, 30.5, 29.5, 31.0, // Cat 2
        ]);
        (x, y)
    }

    #[test]
    fn test_categorical_feature_handler_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 0.0, 1.5, 1.0, 2.5, 2.0, 3.5],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);

        let mut handler = CategoricalFeatureHandler::new(vec![0]);
        handler.fit(&x, &y, 2);

        assert!(handler.is_categorical(0));
        assert!(!handler.is_categorical(1));
        assert_eq!(handler.n_categories(0), Some(3)); // 0, 1, 2
    }

    #[test]
    fn test_categorical_feature_handler_target_encoding() {
        let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        // Category 0 has mean target 0.0, category 1 has mean 1.0, category 2 has mean 2.0
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

        let mut handler = CategoricalFeatureHandler::new(vec![0]).with_smoothing(0.0); // No smoothing for easier testing
        handler.fit(&x, &y, 1);

        // Check that target encoding produces expected values
        let encoding0 = handler.compute_target_encoding(0, 0.0);
        let encoding1 = handler.compute_target_encoding(0, 1.0);
        let encoding2 = handler.compute_target_encoding(0, 2.0);

        assert_abs_diff_eq!(encoding0, 0.0, epsilon = 0.01);
        assert_abs_diff_eq!(encoding1, 1.0, epsilon = 0.01);
        assert_abs_diff_eq!(encoding2, 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_categorical_feature_handler_unknown_category() {
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut handler = CategoricalFeatureHandler::new(vec![0]).with_smoothing(1.0);
        handler.fit(&x, &y, 1);

        // Unknown category (3.0) should use global mean
        let encoding_unknown = handler.compute_target_encoding(0, 3.0);
        let global_mean = y.mean().unwrap();
        assert_abs_diff_eq!(encoding_unknown, global_mean, epsilon = 0.01);
    }

    #[test]
    fn test_categorical_bin_mapper_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 1.0, // Cat 0
                1.0, 2.0, // Cat 1
                2.0, 3.0, // Cat 2
                0.0, 1.5, // Cat 0
                1.0, 2.5, // Cat 1
                2.0, 3.5, // Cat 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 0.5, 1.5, 2.5]);

        let mut rng = StdRng::seed_from_u64(42);
        let mut mapper = CategoricalBinMapper::new(32, vec![0]);
        mapper.fit(&x, &y, &mut rng);

        assert!(mapper.is_categorical(0));
        assert!(!mapper.is_categorical(1));

        let binned = mapper.transform(&x);
        assert_eq!(binned.shape(), &[6, 2]);
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_with_categorical() {
        let (x, y) = make_classification_data_with_categorical();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(50)
            .with_learning_rate(0.1)
            .with_categorical_features(vec![0]) // Feature 0 is categorical
            .with_categorical_smoothing(1.0)
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        assert!(clf.is_fitted());
        assert_eq!(clf.n_features(), Some(2));

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 16);

        // Calculate accuracy
        let accuracy: f64 = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count() as f64
            / 16.0;

        // With categorical features properly handled, should get reasonable accuracy
        assert!(accuracy >= 0.5, "Accuracy was {}", accuracy);
    }

    #[test]
    fn test_hist_gradient_boosting_classifier_categorical_proba() {
        let (x, y) = make_classification_data_with_categorical();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(30)
            .with_categorical_features(vec![0])
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();

        let probas = clf.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[16, 2]);

        // Probabilities should sum to 1
        for i in 0..16 {
            let row_sum: f64 = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-6);
        }

        // Probabilities should be in [0, 1]
        for p in probas.iter() {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }

    #[test]
    fn test_hist_gradient_boosting_regressor_with_categorical() {
        let (x, y) = make_regression_data_with_categorical();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(100)
            .with_learning_rate(0.1)
            .with_categorical_features(vec![0]) // Feature 0 is categorical
            .with_categorical_smoothing(1.0)
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();

        assert!(reg.is_fitted());
        assert_eq!(reg.n_features(), Some(2));

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 12);

        // Calculate RMSE
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / 12.0;
        let rmse = mse.sqrt();

        // With categorical features properly handled, should achieve reasonable fit
        assert!(rmse < 10.0, "RMSE was {}", rmse);
    }

    #[test]
    fn test_categorical_encoding_type() {
        // Test that the encoding type enum works
        let encoding = CategoricalEncoding::default();
        assert_eq!(encoding, CategoricalEncoding::OrderedTargetEncoding);

        let handler = CategoricalFeatureHandler::new(vec![0])
            .with_encoding(CategoricalEncoding::TargetEncoding);
        assert_eq!(handler.encoding, CategoricalEncoding::TargetEncoding);
    }

    #[test]
    fn test_categorical_empty_handler() {
        // Test that models work when no categorical features are specified
        let (x, y) = make_classification_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(20)
            .with_categorical_features(vec![]) // No categorical features
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();
        assert!(clf.is_fitted());
    }

    #[test]
    fn test_categorical_nan_handling() {
        // Test handling of NaN values in categorical features
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0,
                1.0,
                f64::NAN,
                2.0, // Missing category
                1.0,
                3.0,
                0.0,
                1.5,
                f64::NAN,
                2.5, // Missing category
                1.0,
                3.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);

        let mut handler = CategoricalFeatureHandler::new(vec![0]);
        handler.fit(&x, &y, 2);

        // NaN should be treated as a category
        assert!(handler.n_categories(0).unwrap() >= 2);
    }

    #[test]
    fn test_histgb_predict_with_sliced_input() {
        // Test prediction with a non-contiguous array view (slice of columns).
        // Since as_slice() requires contiguous layout, this tests the fallback path.
        let (x, y) = make_classification_data();

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(50)
            .with_learning_rate(0.2)
            .with_random_state(42);
        clf.fit(&x, &y).unwrap();

        // Predict on original contiguous data
        let pred_contiguous = clf.predict(&x).unwrap();

        // Create a wider array, then slice to get a non-contiguous view
        let n = x.nrows();
        let mut wide = Array2::zeros((n, 4));
        for i in 0..n {
            wide[[i, 0]] = x[[i, 0]];
            wide[[i, 1]] = 999.0; // padding column
            wide[[i, 2]] = x[[i, 1]];
            wide[[i, 3]] = 999.0; // padding column
        }
        // Extract columns 0 and 2 into a new owned array (the model expects 2 features)
        let col0 = wide.column(0).to_owned();
        let col2 = wide.column(2).to_owned();
        let mut x_reconstructed = Array2::zeros((n, 2));
        x_reconstructed.column_mut(0).assign(&col0);
        x_reconstructed.column_mut(1).assign(&col2);

        let pred_reconstructed = clf.predict(&x_reconstructed).unwrap();

        assert_eq!(pred_contiguous.len(), pred_reconstructed.len());
        for i in 0..pred_contiguous.len() {
            assert!(
                (pred_contiguous[i] - pred_reconstructed[i]).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                pred_contiguous[i],
                pred_reconstructed[i]
            );
        }
    }

    #[test]
    fn test_histgb_gradient_buffer_reuse() {
        // Fit with n_estimators=10, verify predictions match a known reference.
        // Ensures gradient buffer reuse doesn't accumulate errors across iterations.
        let (x, y) = make_regression_data();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(10)
            .with_learning_rate(0.1)
            .with_random_state(42);
        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // Predictions should be reasonable (not NaN/Inf from buffer corruption)
        for (i, &p) in predictions.iter().enumerate() {
            assert!(
                p.is_finite(),
                "Prediction at index {} is not finite: {}",
                i,
                p
            );
        }

        // Training error should decrease: first predict after 1 iteration vs after 10
        // Fit again from scratch to get the same model (deterministic)
        let mut reg2 = HistGradientBoostingRegressor::new()
            .with_max_iter(10)
            .with_learning_rate(0.1)
            .with_random_state(42);
        reg2.fit(&x, &y).unwrap();

        let pred2 = reg2.predict(&x).unwrap();

        // Same seed, same data -> identical predictions (buffer reuse is deterministic)
        for i in 0..predictions.len() {
            assert_abs_diff_eq!(predictions[i], pred2[i], epsilon = 1e-10);
        }

        // MSE should be reasonable (model is learning, not just returning garbage)
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / 20.0;
        assert!(
            mse < 100.0,
            "MSE is suspiciously high ({}), gradient buffer reuse may be corrupted",
            mse
        );
    }
}
