//! K-Nearest Neighbors for Classification and Regression
//!
//! This module provides K-Nearest Neighbors (KNN) algorithms with efficient
//! spatial data structures for fast neighbor queries.
//!
//! ## Classifiers
//!
//! - [`KNeighborsClassifier`] - Classification based on majority voting among nearest neighbors
//!
//! ## Regressors
//!
//! - [`KNeighborsRegressor`] - Regression based on mean/weighted mean of nearest neighbors
//!
//! ## Features
//!
//! - **Distance metrics**: Euclidean, Manhattan, Minkowski (configurable p)
//! - **Weights**: Uniform (all neighbors equal) or Distance-weighted
//! - **KD-Tree**: Efficient O(log n) nearest neighbor search for low-dimensional data
//! - **Ball Tree**: Better for high-dimensional data (planned)
//! - **Brute force**: Exact O(n) search, useful for small datasets or comparison
//!
//! ## Example - Classification
//!
//! ```
//! use ferroml_core::models::knn::KNeighborsClassifier;
//! use ferroml_core::models::{Model, ProbabilisticModel};
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
//!
//! let mut model = KNeighborsClassifier::new(3);
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 6);
//! ```
//!
//! ## Example - Regression
//!
//! ```
//! use ferroml_core::models::knn::KNeighborsRegressor;
//! use ferroml_core::models::Model;
//! use ndarray::{Array1, Array2};
//!
//! let x = Array2::from_shape_vec((5, 2), vec![
//!     1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0
//! ]).unwrap();
//! let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//!
//! let mut model = KNeighborsRegressor::new(2);
//! model.fit(&x, &y).unwrap();
//!
//! let predictions = model.predict(&x).unwrap();
//! assert_eq!(predictions.len(), 5);
//! ```

use crate::hpo::SearchSpace;
use crate::models::{
    check_is_fitted, validate_fit_input, validate_predict_input, ClassWeight, Model,
    PredictionInterval, ProbabilisticModel,
};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Distance metric for K-Nearest Neighbors.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm): sqrt(sum((x_i - y_i)^2))
    Euclidean,
    /// Manhattan distance (L1 norm): sum(|x_i - y_i|)
    Manhattan,
    /// Minkowski distance: (sum(|x_i - y_i|^p))^(1/p)
    /// When p=1, this is Manhattan; when p=2, this is Euclidean.
    Minkowski(f64),
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Euclidean
    }
}

impl DistanceMetric {
    /// Compute distance between two points.
    ///
    /// When the `simd` feature is enabled, this uses SIMD-accelerated implementations
    /// for Euclidean and Manhattan distances.
    #[inline]
    pub fn compute(&self, a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len(), "Points must have same dimension");

        match self {
            #[cfg(feature = "simd")]
            DistanceMetric::Euclidean => crate::simd::euclidean_distance(a, b),

            #[cfg(not(feature = "simd"))]
            DistanceMetric::Euclidean => {
                let sum_sq: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(ai, bi)| (ai - bi).powi(2))
                    .sum();
                sum_sq.sqrt()
            }

            #[cfg(feature = "simd")]
            DistanceMetric::Manhattan => crate::simd::manhattan_distance(a, b),

            #[cfg(not(feature = "simd"))]
            DistanceMetric::Manhattan => {
                a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).abs()).sum()
            }

            #[cfg(feature = "simd")]
            DistanceMetric::Minkowski(p) => crate::simd::minkowski_distance(a, b, *p),

            #[cfg(not(feature = "simd"))]
            DistanceMetric::Minkowski(p) => {
                let sum: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(ai, bi)| (ai - bi).abs().powf(*p))
                    .sum();
                sum.powf(1.0 / p)
            }
        }
    }

    /// Compute squared distance (for Euclidean, avoids sqrt for efficiency).
    ///
    /// When the `simd` feature is enabled, this uses SIMD-accelerated implementations.
    #[inline]
    pub fn compute_squared(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            #[cfg(feature = "simd")]
            DistanceMetric::Euclidean => crate::simd::squared_euclidean_distance(a, b),

            #[cfg(not(feature = "simd"))]
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).powi(2))
                .sum(),

            _ => {
                let d = self.compute(a, b);
                d * d
            }
        }
    }
}

/// Weighting scheme for nearest neighbor contributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KNNWeights {
    /// All neighbors contribute equally (majority vote / simple average).
    Uniform,
    /// Closer neighbors have more influence (weight = 1 / distance).
    /// Uses 1 / (distance + eps) to avoid division by zero.
    Distance,
}

impl Default for KNNWeights {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Algorithm for nearest neighbor search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KNNAlgorithm {
    /// Build and use a KD-Tree for efficient O(log n) queries.
    /// Best for low-dimensional data (d < 20).
    KDTree,
    /// Build and use a Ball Tree for efficient queries.
    /// Better for high-dimensional data.
    BallTree,
    /// Brute force O(n) search.
    /// Guarantees exact results, useful for small datasets.
    BruteForce,
    /// Automatically select based on data characteristics.
    Auto,
}

impl Default for KNNAlgorithm {
    fn default() -> Self {
        Self::Auto
    }
}

// =============================================================================
// KD-Tree Implementation
// =============================================================================

/// A node in the KD-Tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KDNode {
    /// Index of the point in the original data (for internal nodes).
    point_idx: usize,
    /// Indices of all points in this node (for leaf nodes).
    point_indices: Vec<usize>,
    /// The split dimension.
    split_dim: usize,
    /// The split value.
    split_val: f64,
    /// Left child (points with x\[split_dim\] < split_val).
    left: Option<Box<KDNode>>,
    /// Right child (points with x\[split_dim\] >= split_val).
    right: Option<Box<KDNode>>,
    /// Whether this is a leaf node.
    is_leaf: bool,
}

/// KD-Tree for efficient nearest neighbor queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KDTree {
    /// Root of the tree.
    root: Option<Box<KDNode>>,
    /// Reference to the data points.
    data: Array2<f64>,
    /// Number of dimensions.
    n_dims: usize,
    /// Leaf size (stop splitting when fewer points remain).
    leaf_size: usize,
}

impl KDTree {
    /// Build a KD-Tree from data points.
    pub fn build(data: Array2<f64>, leaf_size: usize) -> Self {
        let n_samples = data.nrows();
        let n_dims = data.ncols();

        if n_samples == 0 {
            return Self {
                root: None,
                data,
                n_dims,
                leaf_size,
            };
        }

        // Indices of all points
        let indices: Vec<usize> = (0..n_samples).collect();

        let root = Self::build_node(&data, indices, 0, n_dims, leaf_size);

        Self {
            root: Some(Box::new(root)),
            data,
            n_dims,
            leaf_size,
        }
    }

    /// Recursively build a KD-Tree node.
    fn build_node(
        data: &Array2<f64>,
        mut indices: Vec<usize>,
        depth: usize,
        n_dims: usize,
        leaf_size: usize,
    ) -> KDNode {
        let split_dim = depth % n_dims;

        // Sort indices by the split dimension
        indices.sort_by(|&a, &b| {
            data[[a, split_dim]]
                .partial_cmp(&data[[b, split_dim]])
                .unwrap_or(Ordering::Equal)
        });

        let median_idx = indices.len() / 2;
        let point_idx = indices[median_idx];
        let split_val = data[[point_idx, split_dim]];

        // Base case: small enough to be a leaf - store all points
        if indices.len() <= leaf_size {
            return KDNode {
                point_idx,
                point_indices: indices, // Store all indices for leaf
                split_dim,
                split_val,
                left: None,
                right: None,
                is_leaf: true,
            };
        }

        // Recursively build children
        let left_indices: Vec<usize> = indices[..median_idx].to_vec();
        let right_indices: Vec<usize> = indices[median_idx + 1..].to_vec();

        let left = if left_indices.is_empty() {
            None
        } else {
            Some(Box::new(Self::build_node(
                data,
                left_indices,
                depth + 1,
                n_dims,
                leaf_size,
            )))
        };

        let right = if right_indices.is_empty() {
            None
        } else {
            Some(Box::new(Self::build_node(
                data,
                right_indices,
                depth + 1,
                n_dims,
                leaf_size,
            )))
        };

        KDNode {
            point_idx,
            point_indices: vec![], // Internal nodes don't need to store all indices
            split_dim,
            split_val,
            left,
            right,
            is_leaf: false,
        }
    }

    /// Find k nearest neighbors of a query point.
    pub fn query(&self, query: &[f64], k: usize, metric: &DistanceMetric) -> Vec<(usize, f64)> {
        let mut heap = BinaryHeap::new();

        if let Some(ref root) = self.root {
            self.query_node(root, query, k, metric, &mut heap);
        }

        // Extract results in sorted order (closest first)
        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|n| (n.idx, n.dist)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Recursively search for nearest neighbors.
    fn query_node(
        &self,
        node: &KDNode,
        query: &[f64],
        k: usize,
        metric: &DistanceMetric,
        heap: &mut BinaryHeap<NeighborCandidate>,
    ) {
        // If this is a leaf node, check all points in the leaf
        if node.is_leaf {
            for &idx in &node.point_indices {
                let point: Vec<f64> = self.data.row(idx).to_vec();
                let dist = metric.compute(query, &point);

                if heap.len() < k {
                    heap.push(NeighborCandidate { idx, dist });
                } else if let Some(top) = heap.peek() {
                    if dist < top.dist {
                        heap.pop();
                        heap.push(NeighborCandidate { idx, dist });
                    }
                }
            }
            return;
        }

        // For internal nodes, check the node's point
        let point: Vec<f64> = self.data.row(node.point_idx).to_vec();
        let dist = metric.compute(query, &point);

        if heap.len() < k {
            heap.push(NeighborCandidate {
                idx: node.point_idx,
                dist,
            });
        } else if let Some(top) = heap.peek() {
            if dist < top.dist {
                heap.pop();
                heap.push(NeighborCandidate {
                    idx: node.point_idx,
                    dist,
                });
            }
        }

        // Determine which side to search first
        let query_val = query[node.split_dim];
        let (first, second) = if query_val < node.split_val {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search the closer side
        if let Some(ref child) = first {
            self.query_node(child, query, k, metric, heap);
        }

        // Check if we need to search the other side
        let split_dist = (query_val - node.split_val).abs();
        let should_search_other =
            heap.len() < k || heap.peek().map(|top| split_dist < top.dist).unwrap_or(true);

        if should_search_other {
            if let Some(ref child) = second {
                self.query_node(child, query, k, metric, heap);
            }
        }
    }
}

/// Candidate neighbor for the priority queue.
#[derive(Debug, Clone)]
struct NeighborCandidate {
    idx: usize,
    dist: f64,
}

impl PartialEq for NeighborCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for NeighborCandidate {}

impl PartialOrd for NeighborCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NeighborCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for max-heap (we want to pop the farthest)
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

// =============================================================================
// Ball Tree Implementation
// =============================================================================

/// A node in the Ball Tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BallNode {
    /// Center of the bounding ball.
    center: Vec<f64>,
    /// Radius of the bounding ball.
    radius: f64,
    /// Indices of points in this node (for leaf nodes).
    point_indices: Vec<usize>,
    /// Left child.
    left: Option<Box<BallNode>>,
    /// Right child.
    right: Option<Box<BallNode>>,
    /// Whether this is a leaf node.
    is_leaf: bool,
}

/// Ball Tree for efficient nearest neighbor queries in high dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BallTree {
    /// Root of the tree.
    root: Option<Box<BallNode>>,
    /// Reference to the data points.
    data: Array2<f64>,
    /// Leaf size.
    leaf_size: usize,
}

impl BallTree {
    /// Build a Ball Tree from data points.
    pub fn build(data: Array2<f64>, leaf_size: usize) -> Self {
        let n_samples = data.nrows();

        if n_samples == 0 {
            return Self {
                root: None,
                data,
                leaf_size,
            };
        }

        let indices: Vec<usize> = (0..n_samples).collect();
        let root = Self::build_node(&data, indices, leaf_size);

        Self {
            root: Some(Box::new(root)),
            data,
            leaf_size,
        }
    }

    /// Recursively build a Ball Tree node.
    fn build_node(data: &Array2<f64>, indices: Vec<usize>, leaf_size: usize) -> BallNode {
        let n_dims = data.ncols();

        // Compute center (mean of all points)
        let center = Self::compute_center(data, &indices);

        // Compute radius (max distance from center)
        let radius = Self::compute_radius(data, &indices, &center);

        // Base case: leaf node
        if indices.len() <= leaf_size {
            return BallNode {
                center,
                radius,
                point_indices: indices,
                left: None,
                right: None,
                is_leaf: true,
            };
        }

        // Find the dimension with maximum spread
        let split_dim = Self::find_split_dimension(data, &indices, n_dims);

        // Split by median along this dimension
        let mut sorted_indices = indices;
        sorted_indices.sort_by(|&a, &b| {
            data[[a, split_dim]]
                .partial_cmp(&data[[b, split_dim]])
                .unwrap_or(Ordering::Equal)
        });

        let mid = sorted_indices.len() / 2;
        let left_indices = sorted_indices[..mid].to_vec();
        let right_indices = sorted_indices[mid..].to_vec();

        let left = if left_indices.is_empty() {
            None
        } else {
            Some(Box::new(Self::build_node(data, left_indices, leaf_size)))
        };

        let right = if right_indices.is_empty() {
            None
        } else {
            Some(Box::new(Self::build_node(data, right_indices, leaf_size)))
        };

        BallNode {
            center,
            radius,
            point_indices: vec![],
            left,
            right,
            is_leaf: false,
        }
    }

    /// Compute center of a set of points.
    fn compute_center(data: &Array2<f64>, indices: &[usize]) -> Vec<f64> {
        let n_dims = data.ncols();
        let mut center = vec![0.0; n_dims];

        for &idx in indices {
            for d in 0..n_dims {
                center[d] += data[[idx, d]];
            }
        }

        let n = indices.len() as f64;
        for c in &mut center {
            *c /= n;
        }

        center
    }

    /// Compute radius (max distance from center to any point).
    fn compute_radius(data: &Array2<f64>, indices: &[usize], center: &[f64]) -> f64 {
        let mut max_dist_sq = 0.0_f64;

        for &idx in indices {
            let point: Vec<f64> = data.row(idx).to_vec();
            let dist_sq: f64 = center
                .iter()
                .zip(point.iter())
                .map(|(c, p)| (c - p).powi(2))
                .sum();
            max_dist_sq = max_dist_sq.max(dist_sq);
        }

        max_dist_sq.sqrt()
    }

    /// Find dimension with maximum spread.
    fn find_split_dimension(data: &Array2<f64>, indices: &[usize], n_dims: usize) -> usize {
        let mut max_spread = 0.0_f64;
        let mut best_dim = 0;

        for d in 0..n_dims {
            let values: Vec<f64> = indices.iter().map(|&idx| data[[idx, d]]).collect();
            let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let spread = max_val - min_val;

            if spread > max_spread {
                max_spread = spread;
                best_dim = d;
            }
        }

        best_dim
    }

    /// Find k nearest neighbors of a query point.
    pub fn query(&self, query: &[f64], k: usize, metric: &DistanceMetric) -> Vec<(usize, f64)> {
        let mut heap = BinaryHeap::new();

        if let Some(ref root) = self.root {
            self.query_node(root, query, k, metric, &mut heap);
        }

        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|n| (n.idx, n.dist)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Recursively search for nearest neighbors.
    fn query_node(
        &self,
        node: &BallNode,
        query: &[f64],
        k: usize,
        metric: &DistanceMetric,
        heap: &mut BinaryHeap<NeighborCandidate>,
    ) {
        // Compute distance from query to ball center
        let dist_to_center = metric.compute(query, &node.center);

        // Prune if ball is too far away
        let current_worst = heap.peek().map(|n| n.dist).unwrap_or(f64::INFINITY);
        if heap.len() >= k && dist_to_center - node.radius > current_worst {
            return;
        }

        if node.is_leaf {
            // Check all points in leaf
            for &idx in &node.point_indices {
                let point: Vec<f64> = self.data.row(idx).to_vec();
                let dist = metric.compute(query, &point);

                if heap.len() < k {
                    heap.push(NeighborCandidate { idx, dist });
                } else if let Some(top) = heap.peek() {
                    if dist < top.dist {
                        heap.pop();
                        heap.push(NeighborCandidate { idx, dist });
                    }
                }
            }
        } else {
            // Recursively search children
            // Search the closer child first
            let mut children = vec![];
            if let Some(ref left) = node.left {
                let dist = metric.compute(query, &left.center);
                children.push((left, dist));
            }
            if let Some(ref right) = node.right {
                let dist = metric.compute(query, &right.center);
                children.push((right, dist));
            }

            children.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            for (child, _) in children {
                self.query_node(child, query, k, metric, heap);
            }
        }
    }
}

// =============================================================================
// Brute Force Search
// =============================================================================

/// Find k nearest neighbors using brute force O(n) search.
fn brute_force_search(
    data: &Array2<f64>,
    query: &[f64],
    k: usize,
    metric: &DistanceMetric,
) -> Vec<(usize, f64)> {
    let mut distances: Vec<(usize, f64)> = data
        .rows()
        .into_iter()
        .enumerate()
        .map(|(idx, row)| {
            let point: Vec<f64> = row.to_vec();
            (idx, metric.compute(query, &point))
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    distances.truncate(k);
    distances
}

// =============================================================================
// KNeighborsClassifier
// =============================================================================

/// K-Nearest Neighbors classifier.
///
/// Classification is done by majority vote among the k nearest neighbors.
/// Supports weighted voting where closer neighbors have more influence.
///
/// ## Algorithm Selection
///
/// - **KD-Tree**: Efficient O(log n) queries, best for low dimensions (d < 20)
/// - **Ball Tree**: Better for high-dimensional data
/// - **Brute Force**: O(n) queries, exact results
/// - **Auto**: Selects based on data characteristics
///
/// ## Parameters
///
/// - `n_neighbors`: Number of neighbors to use (default: 5)
/// - `weights`: How to weight neighbor contributions (default: Uniform)
/// - `metric`: Distance metric (default: Euclidean)
/// - `algorithm`: Search algorithm (default: Auto)
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::knn::KNeighborsClassifier;
/// use ferroml_core::models::Model;
///
/// let mut clf = KNeighborsClassifier::new(5)
///     .with_weights(KNNWeights::Distance)
///     .with_metric(DistanceMetric::Manhattan);
///
/// clf.fit(&x_train, &y_train)?;
/// let predictions = clf.predict(&x_test)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNeighborsClassifier {
    /// Number of neighbors to use for prediction.
    pub n_neighbors: usize,
    /// Weighting scheme for neighbors.
    pub weights: KNNWeights,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Algorithm for nearest neighbor search.
    pub algorithm: KNNAlgorithm,
    /// Leaf size for tree-based algorithms.
    pub leaf_size: usize,
    /// Class weights for handling imbalanced datasets
    pub class_weight: ClassWeight,

    // Fitted state
    /// Training data.
    x_train: Option<Array2<f64>>,
    /// Training labels.
    y_train: Option<Array1<f64>>,
    /// Unique class labels.
    classes: Option<Array1<f64>>,
    /// Number of features.
    n_features: Option<usize>,
    /// KD-Tree (if built).
    kdtree: Option<KDTree>,
    /// Ball Tree (if built).
    balltree: Option<BallTree>,
    /// Actual algorithm used (after Auto selection).
    effective_algorithm: Option<KNNAlgorithm>,
}

impl Default for KNeighborsClassifier {
    fn default() -> Self {
        Self::new(5)
    }
}

impl KNeighborsClassifier {
    /// Create a new K-Nearest Neighbors classifier.
    ///
    /// # Arguments
    ///
    /// * `n_neighbors` - Number of neighbors to use (must be >= 1)
    #[must_use]
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(1),
            weights: KNNWeights::Uniform,
            metric: DistanceMetric::Euclidean,
            algorithm: KNNAlgorithm::Auto,
            leaf_size: 30,
            class_weight: ClassWeight::Uniform,
            x_train: None,
            y_train: None,
            classes: None,
            n_features: None,
            kdtree: None,
            balltree: None,
            effective_algorithm: None,
        }
    }

    /// Set the weighting scheme.
    #[must_use]
    pub fn with_weights(mut self, weights: KNNWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the distance metric.
    #[must_use]
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the algorithm for nearest neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the leaf size for tree-based algorithms.
    #[must_use]
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size.max(1);
        self
    }

    /// Set class weights for handling imbalanced datasets.
    ///
    /// # Arguments
    ///
    /// * `class_weight` - The weighting scheme for classes
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Get the unique class labels.
    #[must_use]
    pub fn classes(&self) -> Option<&Array1<f64>> {
        self.classes.as_ref()
    }

    /// Get the effective algorithm used after fitting.
    #[must_use]
    pub fn effective_algorithm(&self) -> Option<KNNAlgorithm> {
        self.effective_algorithm
    }

    /// Find k nearest neighbors for a query point.
    fn find_neighbors(&self, query: &[f64]) -> Vec<(usize, f64)> {
        let k = self
            .n_neighbors
            .min(self.x_train.as_ref().expect("model must be fitted").nrows());

        match self.effective_algorithm.expect("model must be fitted") {
            KNNAlgorithm::KDTree => {
                self.kdtree
                    .as_ref()
                    .expect("kdtree must be built")
                    .query(query, k, &self.metric)
            }
            KNNAlgorithm::BallTree => self
                .balltree
                .as_ref()
                .expect("balltree must be built")
                .query(query, k, &self.metric),
            KNNAlgorithm::BruteForce | KNNAlgorithm::Auto => brute_force_search(
                self.x_train.as_ref().expect("model must be fitted"),
                query,
                k,
                &self.metric,
            ),
        }
    }

    /// Compute weights for neighbors based on distances.
    fn compute_weights(&self, distances: &[(usize, f64)]) -> Vec<f64> {
        match self.weights {
            KNNWeights::Uniform => vec![1.0; distances.len()],
            KNNWeights::Distance => {
                const EPS: f64 = 1e-10;
                distances.iter().map(|(_, d)| 1.0 / (d + EPS)).collect()
            }
        }
    }

    /// Select the best algorithm based on data characteristics.
    fn select_algorithm(&self, n_samples: usize, n_features: usize) -> KNNAlgorithm {
        match self.algorithm {
            KNNAlgorithm::Auto => {
                // Heuristics for algorithm selection
                if n_samples < 30 {
                    // Very small dataset: brute force is fine
                    KNNAlgorithm::BruteForce
                } else if n_features > 20 {
                    // High dimensional: Ball Tree is better
                    KNNAlgorithm::BallTree
                } else {
                    // Default: KD-Tree for low dimensions
                    KNNAlgorithm::KDTree
                }
            }
            other => other,
        }
    }
}

impl Model for KNeighborsClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.n_neighbors > x.nrows() {
            return Err(FerroError::invalid_input(format!(
                "n_neighbors ({}) cannot be greater than n_samples ({})",
                self.n_neighbors,
                x.nrows()
            )));
        }

        // Extract unique classes
        let mut classes_vec: Vec<f64> = y.iter().copied().collect();
        classes_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        classes_vec.dedup();

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Select and build appropriate spatial index
        let effective_algo = self.select_algorithm(n_samples, n_features);
        self.effective_algorithm = Some(effective_algo);

        match effective_algo {
            KNNAlgorithm::KDTree => {
                self.kdtree = Some(KDTree::build(x.clone(), self.leaf_size));
                self.balltree = None;
            }
            KNNAlgorithm::BallTree => {
                self.balltree = Some(BallTree::build(x.clone(), self.leaf_size));
                self.kdtree = None;
            }
            KNNAlgorithm::BruteForce | KNNAlgorithm::Auto => {
                self.kdtree = None;
                self.balltree = None;
            }
        }

        // Store training data
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        self.classes = Some(Array1::from_vec(classes_vec));
        self.n_features = Some(n_features);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.x_train, "predict")?;
        validate_predict_input(
            x,
            self.n_features
                .ok_or_else(|| FerroError::not_fitted("predict"))?,
        )?;

        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let query: Vec<f64> = x.row(i).to_vec();
            let neighbors = self.find_neighbors(&query);
            let weights = self.compute_weights(&neighbors);

            // Weighted vote
            let mut class_weights = vec![0.0; classes.len()];
            for ((idx, _), w) in neighbors.iter().zip(weights.iter()) {
                let label = y_train[*idx];
                // Find class index
                for (ci, &c) in classes.iter().enumerate() {
                    if (label - c).abs() < 1e-10 {
                        class_weights[ci] += w;
                        break;
                    }
                }
            }

            // Find class with highest weight
            let best_class_idx = class_weights
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| {
                    a.partial_cmp(b).unwrap_or(Ordering::Equal).then(ib.cmp(ia))
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions[i] = classes[best_class_idx];
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.x_train.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("n_neighbors", 1, 20)
            .categorical(
                "weights",
                vec!["uniform".to_string(), "distance".to_string()],
            )
            .categorical(
                "metric",
                vec!["euclidean".to_string(), "manhattan".to_string()],
            )
    }
}

impl ProbabilisticModel for KNeighborsClassifier {
    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(&self.x_train, "predict_proba")?;
        validate_predict_input(
            x,
            self.n_features
                .ok_or_else(|| FerroError::not_fitted("predict"))?,
        )?;

        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let n_samples = x.nrows();
        let n_classes = classes.len();

        let mut proba = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let query: Vec<f64> = x.row(i).to_vec();
            let neighbors = self.find_neighbors(&query);
            let weights = self.compute_weights(&neighbors);

            // Compute weighted probabilities
            let mut class_weights = vec![0.0; n_classes];
            let mut total_weight = 0.0;

            for ((idx, _), w) in neighbors.iter().zip(weights.iter()) {
                let label = y_train[*idx];
                for (ci, &c) in classes.iter().enumerate() {
                    if (label - c).abs() < 1e-10 {
                        class_weights[ci] += w;
                        break;
                    }
                }
                total_weight += w;
            }

            // Normalize to probabilities
            if total_weight > 0.0 {
                for ci in 0..n_classes {
                    proba[[i, ci]] = class_weights[ci] / total_weight;
                }
            } else {
                // Uniform if no neighbors (shouldn't happen)
                for ci in 0..n_classes {
                    proba[[i, ci]] = 1.0 / n_classes as f64;
                }
            }
        }

        Ok(proba)
    }

    fn predict_interval(&self, x: &Array2<f64>, level: f64) -> Result<PredictionInterval> {
        // For classification, prediction intervals aren't directly applicable
        // We return dummy intervals based on probability uncertainty
        let probas = self.predict_proba(x)?;
        let predictions = self.predict(x)?;

        let n_samples = x.nrows();
        let mut lower = Array1::zeros(n_samples);
        let mut upper = Array1::zeros(n_samples);

        // Use max probability as a measure of certainty
        for i in 0..n_samples {
            let max_prob = probas.row(i).iter().copied().fold(0.0_f64, f64::max);
            let uncertainty = 1.0 - max_prob;

            // Scale uncertainty by confidence level
            let half_width = uncertainty * (1.0 - level);
            lower[i] = predictions[i] - half_width;
            upper[i] = predictions[i] + half_width;
        }

        Ok(PredictionInterval::new(predictions, lower, upper, level))
    }
}

// =============================================================================
// KNeighborsRegressor
// =============================================================================

/// K-Nearest Neighbors regressor.
///
/// Regression is done by averaging the target values of the k nearest neighbors.
/// Supports weighted averaging where closer neighbors have more influence.
///
/// ## Parameters
///
/// - `n_neighbors`: Number of neighbors to use (default: 5)
/// - `weights`: How to weight neighbor contributions (default: Uniform)
/// - `metric`: Distance metric (default: Euclidean)
/// - `algorithm`: Search algorithm (default: Auto)
///
/// ## Example
///
/// ```ignore
/// use ferroml_core::models::knn::KNeighborsRegressor;
/// use ferroml_core::models::Model;
///
/// let mut reg = KNeighborsRegressor::new(5)
///     .with_weights(KNNWeights::Distance);
///
/// reg.fit(&x_train, &y_train)?;
/// let predictions = reg.predict(&x_test)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNeighborsRegressor {
    /// Number of neighbors to use for prediction.
    pub n_neighbors: usize,
    /// Weighting scheme for neighbors.
    pub weights: KNNWeights,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Algorithm for nearest neighbor search.
    pub algorithm: KNNAlgorithm,
    /// Leaf size for tree-based algorithms.
    pub leaf_size: usize,

    // Fitted state
    /// Training data.
    x_train: Option<Array2<f64>>,
    /// Training targets.
    y_train: Option<Array1<f64>>,
    /// Number of features.
    n_features: Option<usize>,
    /// KD-Tree (if built).
    kdtree: Option<KDTree>,
    /// Ball Tree (if built).
    balltree: Option<BallTree>,
    /// Actual algorithm used.
    effective_algorithm: Option<KNNAlgorithm>,
}

impl Default for KNeighborsRegressor {
    fn default() -> Self {
        Self::new(5)
    }
}

impl KNeighborsRegressor {
    /// Create a new K-Nearest Neighbors regressor.
    ///
    /// # Arguments
    ///
    /// * `n_neighbors` - Number of neighbors to use (must be >= 1)
    #[must_use]
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(1),
            weights: KNNWeights::Uniform,
            metric: DistanceMetric::Euclidean,
            algorithm: KNNAlgorithm::Auto,
            leaf_size: 30,
            x_train: None,
            y_train: None,
            n_features: None,
            kdtree: None,
            balltree: None,
            effective_algorithm: None,
        }
    }

    /// Set the weighting scheme.
    #[must_use]
    pub fn with_weights(mut self, weights: KNNWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the distance metric.
    #[must_use]
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the algorithm for nearest neighbor search.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: KNNAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the leaf size for tree-based algorithms.
    #[must_use]
    pub fn with_leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size.max(1);
        self
    }

    /// Get the effective algorithm used after fitting.
    #[must_use]
    pub fn effective_algorithm(&self) -> Option<KNNAlgorithm> {
        self.effective_algorithm
    }

    /// Find k nearest neighbors for a query point.
    fn find_neighbors(&self, query: &[f64]) -> Vec<(usize, f64)> {
        let k = self
            .n_neighbors
            .min(self.x_train.as_ref().expect("model must be fitted").nrows());

        match self.effective_algorithm.expect("model must be fitted") {
            KNNAlgorithm::KDTree => {
                self.kdtree
                    .as_ref()
                    .expect("kdtree must be built")
                    .query(query, k, &self.metric)
            }
            KNNAlgorithm::BallTree => self
                .balltree
                .as_ref()
                .expect("balltree must be built")
                .query(query, k, &self.metric),
            KNNAlgorithm::BruteForce | KNNAlgorithm::Auto => brute_force_search(
                self.x_train.as_ref().expect("model must be fitted"),
                query,
                k,
                &self.metric,
            ),
        }
    }

    /// Compute weights for neighbors based on distances.
    fn compute_weights(&self, distances: &[(usize, f64)]) -> Vec<f64> {
        match self.weights {
            KNNWeights::Uniform => vec![1.0; distances.len()],
            KNNWeights::Distance => {
                const EPS: f64 = 1e-10;
                distances.iter().map(|(_, d)| 1.0 / (d + EPS)).collect()
            }
        }
    }

    /// Select the best algorithm based on data characteristics.
    fn select_algorithm(&self, n_samples: usize, n_features: usize) -> KNNAlgorithm {
        match self.algorithm {
            KNNAlgorithm::Auto => {
                if n_samples < 30 {
                    KNNAlgorithm::BruteForce
                } else if n_features > 20 {
                    KNNAlgorithm::BallTree
                } else {
                    KNNAlgorithm::KDTree
                }
            }
            other => other,
        }
    }
}

impl Model for KNeighborsRegressor {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        validate_fit_input(x, y)?;

        if self.n_neighbors > x.nrows() {
            return Err(FerroError::invalid_input(format!(
                "n_neighbors ({}) cannot be greater than n_samples ({})",
                self.n_neighbors,
                x.nrows()
            )));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Select and build appropriate spatial index
        let effective_algo = self.select_algorithm(n_samples, n_features);
        self.effective_algorithm = Some(effective_algo);

        match effective_algo {
            KNNAlgorithm::KDTree => {
                self.kdtree = Some(KDTree::build(x.clone(), self.leaf_size));
                self.balltree = None;
            }
            KNNAlgorithm::BallTree => {
                self.balltree = Some(BallTree::build(x.clone(), self.leaf_size));
                self.kdtree = None;
            }
            KNNAlgorithm::BruteForce | KNNAlgorithm::Auto => {
                self.kdtree = None;
                self.balltree = None;
            }
        }

        // Store training data
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());
        self.n_features = Some(n_features);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        check_is_fitted(&self.x_train, "predict")?;
        validate_predict_input(
            x,
            self.n_features
                .ok_or_else(|| FerroError::not_fitted("predict"))?,
        )?;

        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let query: Vec<f64> = x.row(i).to_vec();
            let neighbors = self.find_neighbors(&query);
            let weights = self.compute_weights(&neighbors);

            // Weighted average
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for ((idx, _), w) in neighbors.iter().zip(weights.iter()) {
                weighted_sum += y_train[*idx] * w;
                weight_sum += w;
            }

            predictions[i] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                // Fallback to mean of all training targets
                y_train.mean().unwrap_or(0.0)
            };
        }

        Ok(predictions)
    }

    fn is_fitted(&self) -> bool {
        self.x_train.is_some()
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn search_space(&self) -> SearchSpace {
        SearchSpace::new()
            .int("n_neighbors", 1, 20)
            .categorical(
                "weights",
                vec!["uniform".to_string(), "distance".to_string()],
            )
            .categorical(
                "metric",
                vec!["euclidean".to_string(), "manhattan".to_string()],
            )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Distance Metric Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_euclidean_distance() {
        let metric = DistanceMetric::Euclidean;
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];

        assert!((metric.compute(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let metric = DistanceMetric::Manhattan;
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];

        assert!((metric.compute(&a, &b) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_minkowski_distance() {
        // p=2 should match Euclidean
        let minkowski = DistanceMetric::Minkowski(2.0);
        let euclidean = DistanceMetric::Euclidean;
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];

        assert!((minkowski.compute(&a, &b) - euclidean.compute(&a, &b)).abs() < 1e-10);

        // p=1 should match Manhattan
        let minkowski_1 = DistanceMetric::Minkowski(1.0);
        let manhattan = DistanceMetric::Manhattan;

        assert!((minkowski_1.compute(&a, &b) - manhattan.compute(&a, &b)).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // KD-Tree Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_kdtree_build() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let tree = KDTree::build(data, 2);

        assert!(tree.root.is_some());
    }

    #[test]
    fn test_kdtree_query() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0],
        )
        .unwrap();
        let tree = KDTree::build(data, 2);
        let metric = DistanceMetric::Euclidean;

        // Query near (0, 0)
        let neighbors = tree.query(&[0.0, 0.0], 2, &metric);
        assert_eq!(neighbors.len(), 2);

        // Closest should be (0, 0) at index 0
        assert_eq!(neighbors[0].0, 0);
        assert!(neighbors[0].1 < 0.1);
    }

    // -------------------------------------------------------------------------
    // Ball Tree Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_balltree_build() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let tree = BallTree::build(data, 2);

        assert!(tree.root.is_some());
    }

    #[test]
    fn test_balltree_query() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0],
        )
        .unwrap();
        let tree = BallTree::build(data, 2);
        let metric = DistanceMetric::Euclidean;

        let neighbors = tree.query(&[0.0, 0.0], 2, &metric);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, 0);
    }

    // -------------------------------------------------------------------------
    // KNeighborsClassifier Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_knn_classifier_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = KNeighborsClassifier::new(3);
        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();

        // Points should be classified correctly (training data)
        for i in 0..3 {
            assert!(
                (predictions[i] - 0.0).abs() < 1e-10,
                "Class 0 point {} misclassified",
                i
            );
        }
        for i in 3..6 {
            assert!(
                (predictions[i] - 1.0).abs() < 1e-10,
                "Class 1 point {} misclassified",
                i
            );
        }
    }

    #[test]
    fn test_knn_classifier_predict_proba() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = KNeighborsClassifier::new(3);
        clf.fit(&x, &y).unwrap();

        let probas = clf.predict_proba(&x).unwrap();

        // Check shape
        assert_eq!(probas.nrows(), 6);
        assert_eq!(probas.ncols(), 2);

        // Check probabilities sum to 1
        for i in 0..6 {
            let row_sum: f64 = probas.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} doesn't sum to 1", i);
        }

        // Class 0 points should have high probability for class 0
        for i in 0..3 {
            assert!(
                probas[[i, 0]] > 0.5,
                "Class 0 point {} has low class 0 probability",
                i
            );
        }
    }

    #[test]
    fn test_knn_classifier_distance_weights() {
        let x =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.0, 10.0, 0.0, 11.0, 0.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut clf = KNeighborsClassifier::new(4).with_weights(KNNWeights::Distance);
        clf.fit(&x, &y).unwrap();

        // Query point very close to class 0 cluster
        let query = Array2::from_shape_vec((1, 2), vec![0.05, 0.0]).unwrap();
        let pred = clf.predict(&query).unwrap();

        // Should predict class 0 due to distance weighting
        assert!((pred[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_classifier_kdtree() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::KDTree);
        clf.fit(&x, &y).unwrap();

        assert_eq!(clf.effective_algorithm(), Some(KNNAlgorithm::KDTree));

        let predictions = clf.predict(&x).unwrap();
        for i in 0..3 {
            assert!((predictions[i] - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_knn_classifier_balltree() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::BallTree);
        clf.fit(&x, &y).unwrap();

        assert_eq!(clf.effective_algorithm(), Some(KNNAlgorithm::BallTree));

        let predictions = clf.predict(&x).unwrap();
        for i in 0..3 {
            assert!((predictions[i] - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_knn_classifier_manhattan() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut clf = KNeighborsClassifier::new(3).with_metric(DistanceMetric::Manhattan);
        clf.fit(&x, &y).unwrap();

        let predictions = clf.predict(&x).unwrap();
        for i in 0..3 {
            assert!((predictions[i] - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_knn_classifier_not_fitted_error() {
        let clf = KNeighborsClassifier::new(3);
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(clf.predict(&x).is_err());
    }

    #[test]
    fn test_knn_classifier_n_neighbors_too_large() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let mut clf = KNeighborsClassifier::new(10); // More than n_samples

        assert!(clf.fit(&x, &y).is_err());
    }

    // -------------------------------------------------------------------------
    // KNeighborsRegressor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_knn_regressor_basic() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut reg = KNeighborsRegressor::new(3);
        reg.fit(&x, &y).unwrap();

        let predictions = reg.predict(&x).unwrap();

        // Predictions should be close to actual values (average of neighbors)
        for i in 0..5 {
            assert!(
                (predictions[i] - y[i]).abs() < 2.0,
                "Prediction {} too far from actual",
                i
            );
        }
    }

    #[test]
    fn test_knn_regressor_uniform_weights() {
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 10.0, 20.0, 30.0]);

        let mut reg = KNeighborsRegressor::new(2).with_weights(KNNWeights::Uniform);
        reg.fit(&x, &y).unwrap();

        // Query at 1.5 should average y[1] and y[2] = (10 + 20) / 2 = 15
        let query = Array2::from_shape_vec((1, 1), vec![1.5]).unwrap();
        let pred = reg.predict(&query).unwrap();

        assert!((pred[0] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn_regressor_distance_weights() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 10.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 100.0, 1000.0]);

        let mut reg = KNeighborsRegressor::new(3).with_weights(KNNWeights::Distance);
        reg.fit(&x, &y).unwrap();

        // Query at 0.9 should be dominated by y[1]=100 due to proximity
        let query = Array2::from_shape_vec((1, 1), vec![0.9]).unwrap();
        let pred = reg.predict(&query).unwrap();

        // Should be closer to 100 than to uniform average (366.67)
        assert!(pred[0] > 50.0 && pred[0] < 200.0);
    }

    #[test]
    fn test_knn_regressor_kdtree() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut reg = KNeighborsRegressor::new(2).with_algorithm(KNNAlgorithm::KDTree);
        reg.fit(&x, &y).unwrap();

        assert_eq!(reg.effective_algorithm(), Some(KNNAlgorithm::KDTree));

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_knn_regressor_balltree() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut reg = KNeighborsRegressor::new(2).with_algorithm(KNNAlgorithm::BallTree);
        reg.fit(&x, &y).unwrap();

        assert_eq!(reg.effective_algorithm(), Some(KNNAlgorithm::BallTree));

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_knn_regressor_not_fitted_error() {
        let reg = KNeighborsRegressor::new(3);
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        assert!(reg.predict(&x).is_err());
    }

    #[test]
    fn test_knn_regressor_feature_mismatch() {
        let x_train = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y_train = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let mut reg = KNeighborsRegressor::new(2);
        reg.fit(&x_train, &y_train).unwrap();

        // Wrong number of features
        let x_test = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(reg.predict(&x_test).is_err());
    }

    // -------------------------------------------------------------------------
    // Algorithm Selection Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_auto_algorithm_selection_small_dataset() {
        let x = Array2::from_shape_vec((10, 2), vec![0.0; 20]).unwrap();
        let y = Array1::from_vec(vec![0.0; 10]);

        let mut clf = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::Auto);
        clf.fit(&x, &y).unwrap();

        // Small dataset should use brute force
        assert_eq!(clf.effective_algorithm(), Some(KNNAlgorithm::BruteForce));
    }

    #[test]
    fn test_auto_algorithm_selection_kdtree() {
        let x = Array2::from_shape_vec((100, 5), vec![0.0; 500]).unwrap();
        let y = Array1::from_vec(vec![0.0; 100]);

        let mut clf = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::Auto);
        clf.fit(&x, &y).unwrap();

        // Medium dataset, low dimensions should use KD-Tree
        assert_eq!(clf.effective_algorithm(), Some(KNNAlgorithm::KDTree));
    }

    #[test]
    fn test_auto_algorithm_selection_balltree() {
        let x = Array2::from_shape_vec((100, 30), vec![0.0; 3000]).unwrap();
        let y = Array1::from_vec(vec![0.0; 100]);

        let mut clf = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::Auto);
        clf.fit(&x, &y).unwrap();

        // High dimensions should use Ball Tree
        assert_eq!(clf.effective_algorithm(), Some(KNNAlgorithm::BallTree));
    }

    // -------------------------------------------------------------------------
    // Search Space Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_knn_classifier_search_space() {
        let clf = KNeighborsClassifier::new(5);
        let space = clf.search_space();

        assert!(space.parameters.contains_key("n_neighbors"));
        assert!(space.parameters.contains_key("weights"));
        assert!(space.parameters.contains_key("metric"));
    }

    #[test]
    fn test_knn_regressor_search_space() {
        let reg = KNeighborsRegressor::new(5);
        let space = reg.search_space();

        assert!(space.parameters.contains_key("n_neighbors"));
        assert!(space.parameters.contains_key("weights"));
        assert!(space.parameters.contains_key("metric"));
    }

    // -------------------------------------------------------------------------
    // Consistency Tests (KDTree vs BruteForce)
    // -------------------------------------------------------------------------

    #[test]
    fn test_kdtree_bruteforce_consistency() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0,
                8.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf_brute = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::BruteForce);
        let mut clf_kd = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::KDTree);

        clf_brute.fit(&x, &y).unwrap();
        clf_kd.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((3, 2), vec![2.5, 2.5, 5.0, 5.0, 7.5, 7.5]).unwrap();

        let pred_brute = clf_brute.predict(&query).unwrap();
        let pred_kd = clf_kd.predict(&query).unwrap();

        for i in 0..3 {
            assert!(
                (pred_brute[i] - pred_kd[i]).abs() < 1e-10,
                "Mismatch at index {}: brute={}, kdtree={}",
                i,
                pred_brute[i],
                pred_kd[i]
            );
        }
    }

    #[test]
    fn test_balltree_bruteforce_consistency() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0,
                8.0, 8.0, 9.0, 9.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf_brute = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::BruteForce);
        let mut clf_ball = KNeighborsClassifier::new(3).with_algorithm(KNNAlgorithm::BallTree);

        clf_brute.fit(&x, &y).unwrap();
        clf_ball.fit(&x, &y).unwrap();

        let query = Array2::from_shape_vec((3, 2), vec![2.5, 2.5, 5.0, 5.0, 7.5, 7.5]).unwrap();

        let pred_brute = clf_brute.predict(&query).unwrap();
        let pred_ball = clf_ball.predict(&query).unwrap();

        for i in 0..3 {
            assert!(
                (pred_brute[i] - pred_ball[i]).abs() < 1e-10,
                "Mismatch at index {}: brute={}, balltree={}",
                i,
                pred_brute[i],
                pred_ball[i]
            );
        }
    }

    #[test]
    fn test_knn_tiebreak_first_class() {
        // Create data where classes 0.0 and 1.0 have equal k-nearest neighbors
        // With k=2: one neighbor of each class, should return lowest index (class 0.0)
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 2.0, 10.0, 12.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);

        let mut knn = KNeighborsClassifier::new(2);
        knn.fit(&x, &y).unwrap();

        // Query point at 1.0: equidistant from 0.0 (class 0) and 2.0 (class 1)
        let query = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let pred = knn.predict(&query).unwrap();

        // On tie, should return lowest class index (0.0), matching sklearn behavior
        assert!(
            (pred[0] - 0.0).abs() < 1e-10,
            "Expected class 0.0 on tie, got {}",
            pred[0]
        );
    }
}
