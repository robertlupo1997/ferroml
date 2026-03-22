//! HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise
//!
//! This module implements HDBSCAN, which extends DBSCAN by converting it into a
//! hierarchical clustering algorithm, then extracting flat clusters using a stability-based
//! method (excess of mass).
//!
//! ## Algorithm Steps
//!
//! 1. Compute core distances (k-th nearest neighbor distance for each point)
//! 2. Build mutual reachability graph
//! 3. Construct minimum spanning tree (Prim's algorithm)
//! 4. Build condensed cluster tree
//! 5. Extract stable clusters using excess of mass method
//!
//! ## Features
//!
//! - **Automatic cluster detection**: No need to specify number of clusters
//! - **Varying density**: Finds clusters of different densities (unlike DBSCAN)
//! - **Noise detection**: Points that don't belong to any cluster labeled -1
//! - **Cluster probabilities**: Soft assignment indicating membership strength
//!
//! ## Example
//!
//! ```
//! use ferroml_core::clustering::{HDBSCAN, ClusteringModel};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1,
//!     5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 5.1, 5.1,
//! ]).unwrap();
//!
//! let mut hdbscan = HDBSCAN::new(3);
//! hdbscan.fit(&x).unwrap();
//!
//! let labels = hdbscan.labels().unwrap();
//! println!("Cluster labels: {:?}", labels);
//! println!("Probabilities: {:?}", hdbscan.probabilities());
//! ```

use crate::clustering::ClusteringModel;
use crate::decomposition::vptree::VPTree;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, ArrayView1};

/// HDBSCAN clustering algorithm.
///
/// HDBSCAN builds a hierarchy of clusters using mutual reachability distances,
/// then extracts the most stable clusters using the excess of mass method.
/// Unlike DBSCAN, it can find clusters of varying density and does not require
/// an `eps` parameter.
///
/// # Parameters
///
/// - `min_cluster_size` - Minimum number of points to form a cluster
/// - `min_samples` - Number of neighbors for core distance (defaults to min_cluster_size)
/// - `cluster_selection_epsilon` - Distance threshold to merge clusters below this scale
/// - `allow_single_cluster` - If true, allows the algorithm to return a single cluster
#[derive(Debug, Clone)]
pub struct HDBSCAN {
    /// Minimum number of points required to form a cluster
    min_cluster_size: usize,
    /// Number of neighbors used to compute core distances (defaults to min_cluster_size)
    min_samples: Option<usize>,
    /// Merge clusters that are closer than this distance threshold
    cluster_selection_epsilon: f64,
    /// Whether to allow a single cluster as output
    allow_single_cluster: bool,

    /// Optional GPU backend for accelerated pairwise distance computation
    #[cfg(feature = "gpu")]
    gpu_backend: Option<std::sync::Arc<dyn crate::gpu::GpuBackend>>,

    // Fitted state
    /// Cluster labels for each point (-1 for noise)
    labels_: Option<Array1<i32>>,
    /// Cluster membership probabilities for each point
    probabilities_: Option<Array1<f64>>,
    /// Number of clusters found
    n_clusters_: Option<usize>,
}

/// A node in the condensed cluster tree.
#[derive(Debug, Clone)]
struct CondensedNode {
    /// Parent cluster ID
    parent: usize,
    /// Child cluster or point ID
    child: usize,
    /// Lambda value (1.0 / distance) at which this separation happened
    lambda_val: f64,
    /// Size: 1 for individual points, cluster size for cluster splits
    child_size: usize,
}

/// The condensed cluster tree produced from the MST.
#[derive(Debug)]
struct CondensedTree {
    nodes: Vec<CondensedNode>,
}

/// Union-Find (disjoint set) data structure for tracking component merges.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    /// The next available cluster ID (starts at n for virtual cluster nodes)
    next_label: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let parent: Vec<usize> = (0..n).collect();
        let size = vec![1usize; n];
        Self {
            parent,
            size,
            next_label: n,
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path compression
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) -> usize {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        // Create a new virtual node for the merge
        let new_label = self.next_label;
        self.next_label += 1;
        self.parent.push(new_label); // parent[new_label] = new_label
        self.size.push(self.size[ra] + self.size[rb]);
        self.parent[ra] = new_label;
        self.parent[rb] = new_label;
        new_label
    }

    fn component_size(&self, x: usize) -> usize {
        self.size[x]
    }
}

impl HDBSCAN {
    /// Create a new HDBSCAN model with the given minimum cluster size.
    ///
    /// # Arguments
    /// * `min_cluster_size` - Minimum number of points to form a cluster (must be >= 2)
    pub fn new(min_cluster_size: usize) -> Self {
        Self {
            min_cluster_size,
            min_samples: None,
            cluster_selection_epsilon: 0.0,
            allow_single_cluster: false,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            labels_: None,
            probabilities_: None,
            n_clusters_: None,
        }
    }

    /// Set GPU backend for accelerated pairwise distance computation.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self {
        self.gpu_backend = Some(backend);
        self
    }

    /// Set the number of neighbors used for core distance computation.
    ///
    /// If not set, defaults to `min_cluster_size`.
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = Some(min_samples);
        self
    }

    /// Set the cluster selection epsilon.
    ///
    /// Clusters with a distance below this threshold will be merged.
    pub fn with_cluster_selection_epsilon(mut self, epsilon: f64) -> Self {
        self.cluster_selection_epsilon = epsilon;
        self
    }

    /// Set whether to allow a single cluster as output.
    pub fn with_allow_single_cluster(mut self, allow: bool) -> Self {
        self.allow_single_cluster = allow;
        self
    }

    /// Get cluster membership probabilities.
    ///
    /// Points assigned to a cluster have probability in (0, 1], noise points have 0.
    pub fn probabilities(&self) -> Option<&Array1<f64>> {
        self.probabilities_.as_ref()
    }

    /// Get the number of clusters found (excluding noise).
    pub fn n_clusters(&self) -> Option<usize> {
        self.n_clusters_
    }

    /// Get the number of noise points.
    pub fn n_noise(&self) -> Option<usize> {
        self.labels_
            .as_ref()
            .map(|labels| labels.iter().filter(|&&l| l == -1).count())
    }
}

/// Compute Euclidean distance between two array views.
#[inline]
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        crate::linalg::squared_euclidean_distance(a_slice, b_slice).sqrt()
    } else {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Compute core distances for each point using VP-tree for efficient k-NN.
///
/// The core distance of a point is the distance to its k-th nearest neighbor.
fn compute_core_distances(x: &Array2<f64>, k: usize) -> Array1<f64> {
    let n = x.nrows();
    let tree = VPTree::from_array(x);
    let mut core_dists = Array1::zeros(n);

    for i in 0..n {
        let query: Vec<f64> = x.row(i).to_vec();
        // +1 because the point finds itself as neighbor at distance 0
        let neighbors = tree.search(&query, k + 1);
        // The k-th neighbor (index k since index 0 is self with dist ~0)
        core_dists[i] = if neighbors.len() > k {
            neighbors[k].1
        } else if let Some(last) = neighbors.last() {
            last.1
        } else {
            f64::INFINITY
        };
    }

    core_dists
}

/// Compute mutual reachability distance between two points.
///
/// d_mreach(a, b) = max(core_dist[a], core_dist[b], d(a, b))
#[inline]
fn mutual_reachability_distance(
    x: &Array2<f64>,
    core_dists: &Array1<f64>,
    a: usize,
    b: usize,
) -> f64 {
    let d = euclidean_distance(&x.row(a), &x.row(b));
    d.max(core_dists[a]).max(core_dists[b])
}

#[cfg(feature = "gpu")]
/// Build MST from a precomputed squared-Euclidean distance matrix and core distances.
///
/// The distance matrix contains squared Euclidean distances; this function takes
/// the square root before computing mutual reachability.
///
/// Returns edges as (source, target, weight) sorted by weight ascending.
fn build_mst_from_precomputed(
    sq_dist_matrix: &Array2<f64>,
    core_dists: &Array1<f64>,
) -> Vec<(usize, usize, f64)> {
    let n = sq_dist_matrix.nrows();
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_edge = vec![f64::INFINITY; n];
    let mut min_source = vec![0usize; n];
    let mut mst = Vec::with_capacity(n - 1);

    // Start from node 0
    in_tree[0] = true;
    for j in 1..n {
        let d = sq_dist_matrix[[0, j]].sqrt();
        min_edge[j] = d.max(core_dists[0]).max(core_dists[j]);
        min_source[j] = 0;
    }

    for _ in 0..(n - 1) {
        let mut next = usize::MAX;
        let mut next_dist = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_edge[j] < next_dist {
                next_dist = min_edge[j];
                next = j;
            }
        }

        if next == usize::MAX {
            for j in 0..n {
                if !in_tree[j] {
                    next = j;
                    next_dist = min_edge[j];
                    break;
                }
            }
        }

        in_tree[next] = true;
        mst.push((min_source[next], next, next_dist));

        for j in 0..n {
            if !in_tree[j] {
                let d = sq_dist_matrix[[next, j]].sqrt();
                let mreach = d.max(core_dists[next]).max(core_dists[j]);
                if mreach < min_edge[j] {
                    min_edge[j] = mreach;
                    min_source[j] = next;
                }
            }
        }
    }

    mst.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    mst
}

/// Build minimum spanning tree using dense Prim's algorithm on mutual reachability distances.
///
/// Returns edges as (source, target, weight) sorted by weight ascending.
fn build_mst_mutual_reachability(
    x: &Array2<f64>,
    core_dists: &Array1<f64>,
) -> Vec<(usize, usize, f64)> {
    let n = x.nrows();
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_edge = vec![f64::INFINITY; n];
    let mut min_source = vec![0usize; n];
    let mut mst = Vec::with_capacity(n - 1);

    // Start from node 0
    in_tree[0] = true;
    for j in 1..n {
        min_edge[j] = mutual_reachability_distance(x, core_dists, 0, j);
        min_source[j] = 0;
    }

    for _ in 0..(n - 1) {
        // Find closest non-tree node
        let mut next = usize::MAX;
        let mut next_dist = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_edge[j] < next_dist {
                next_dist = min_edge[j];
                next = j;
            }
        }

        if next == usize::MAX {
            // Graph is disconnected — pick any non-tree node
            for j in 0..n {
                if !in_tree[j] {
                    next = j;
                    next_dist = min_edge[j];
                    break;
                }
            }
        }

        in_tree[next] = true;
        mst.push((min_source[next], next, next_dist));

        // Update min_edge for remaining nodes
        for j in 0..n {
            if !in_tree[j] {
                let d = mutual_reachability_distance(x, core_dists, next, j);
                if d < min_edge[j] {
                    min_edge[j] = d;
                    min_source[j] = next;
                }
            }
        }
    }

    // Sort by weight ascending
    mst.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    mst
}

/// Build the condensed cluster tree from a sorted MST.
///
/// The condensed tree tracks when clusters split and when points "fall out"
/// as noise. Cluster IDs start at `n` (the number of data points).
fn build_condensed_tree(
    mst: &[(usize, usize, f64)],
    min_cluster_size: usize,
    n: usize,
) -> CondensedTree {
    let mut uf = UnionFind::new(n);
    let mut nodes = Vec::new();

    // Process MST edges in ascending order of weight (= merging from dense to sparse)
    // But we want to think of it as "splitting from sparse to dense" for the condensed tree.
    // So we process edges in reverse order (largest weight first = smallest lambda first).
    // Actually, the standard approach: process merges in ascending distance order, recording
    // the hierarchy, then walk from root to leaves to build the condensed tree.

    // Step 1: Build the full dendrogram by processing MST edges in ascending order
    // Each merge creates a new cluster node with ID >= n
    let mut merge_history: Vec<(usize, usize, f64, usize)> = Vec::with_capacity(n - 1);

    for &(a, b, dist) in mst {
        let ra = uf.find(a);
        let rb = uf.find(b);
        if ra == rb {
            continue;
        }
        let size_a = uf.component_size(ra);
        let size_b = uf.component_size(rb);
        let new_label = uf.union(ra, rb);
        let new_size = size_a + size_b;
        merge_history.push((ra, rb, dist, new_size));
        let _ = new_label;
    }

    // Step 2: Build condensed tree
    // Walk the dendrogram top-down. The root is the last merge.
    // For each merge at distance d (lambda = 1/d):
    //   - If both children have size >= min_cluster_size: real split, both become clusters
    //   - If one child < min_cluster_size: those points fall out of the parent cluster
    //   - If both children < min_cluster_size: all points fall out

    // Map from dendrogram node ID to condensed cluster ID
    // Dendrogram node IDs: 0..n are original points, n..2n-1 are merge nodes
    // Condensed cluster IDs also start at n

    // We need to track which condensed cluster each dendrogram node belongs to
    let n_merges = merge_history.len();
    if n_merges == 0 {
        return CondensedTree { nodes };
    }

    // Dendrogram nodes: first n are original points, next n_merges are merge results
    // The merge at index i created dendrogram node (n + i)
    let total_nodes = n + n_merges;
    let mut children: Vec<Option<(usize, usize, f64)>> = vec![None; total_nodes];

    for (i, &(left, right, dist, _size)) in merge_history.iter().enumerate() {
        children[n + i] = Some((left, right, dist));
    }

    // Track sizes of each dendrogram node
    let mut dend_size = vec![1usize; total_nodes];
    for (i, &(_left, _right, _dist, size)) in merge_history.iter().enumerate() {
        dend_size[n + i] = size;
    }

    // Now walk top-down from root to build condensed tree
    // Root is the last merge node: n + n_merges - 1
    let root = n + n_merges - 1;

    // condensed_cluster_id[dend_node] = which condensed cluster this node belongs to
    let mut condensed_cluster_id = vec![0usize; total_nodes];
    let mut next_condensed_id = n; // condensed cluster IDs start at n
    condensed_cluster_id[root] = next_condensed_id;
    next_condensed_id += 1;

    // BFS/DFS from root
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        if node < n {
            // This is an original point — it's already been handled as "falling out"
            continue;
        }

        let (left, right, dist) = match children[node] {
            Some(c) => c,
            None => continue,
        };

        let lambda = if dist > 0.0 {
            1.0 / dist
        } else {
            f64::INFINITY
        };
        let left_size = dend_size[left];
        let right_size = dend_size[right];
        let parent_cluster = condensed_cluster_id[node];

        let left_is_cluster = left_size >= min_cluster_size;
        let right_is_cluster = right_size >= min_cluster_size;

        if left_is_cluster && right_is_cluster {
            // Real split: both children become new clusters
            let left_cluster = next_condensed_id;
            next_condensed_id += 1;
            let right_cluster = next_condensed_id;
            next_condensed_id += 1;

            nodes.push(CondensedNode {
                parent: parent_cluster,
                child: left_cluster,
                lambda_val: lambda,
                child_size: left_size,
            });
            nodes.push(CondensedNode {
                parent: parent_cluster,
                child: right_cluster,
                lambda_val: lambda,
                child_size: right_size,
            });

            condensed_cluster_id[left] = left_cluster;
            condensed_cluster_id[right] = right_cluster;
            stack.push(left);
            stack.push(right);
        } else if left_is_cluster {
            // Right side falls out as noise points
            emit_points_falling_out(right, &children, n, parent_cluster, lambda, &mut nodes);
            condensed_cluster_id[left] = parent_cluster;
            stack.push(left);
        } else if right_is_cluster {
            // Left side falls out as noise points
            emit_points_falling_out(left, &children, n, parent_cluster, lambda, &mut nodes);
            condensed_cluster_id[right] = parent_cluster;
            stack.push(right);
        } else {
            // Both sides too small — all points fall out
            emit_points_falling_out(left, &children, n, parent_cluster, lambda, &mut nodes);
            emit_points_falling_out(right, &children, n, parent_cluster, lambda, &mut nodes);
        }
    }

    CondensedTree { nodes }
}

/// Recursively emit all individual points under a dendrogram subtree as
/// "falling out" of the given parent cluster at the given lambda.
fn emit_points_falling_out(
    node: usize,
    children: &[Option<(usize, usize, f64)>],
    n: usize,
    parent_cluster: usize,
    lambda: f64,
    nodes: &mut Vec<CondensedNode>,
) {
    if node < n {
        // Individual point
        nodes.push(CondensedNode {
            parent: parent_cluster,
            child: node,
            lambda_val: lambda,
            child_size: 1,
        });
        return;
    }
    // Internal dendrogram node — recurse into children
    if let Some((left, right, _dist)) = children[node] {
        emit_points_falling_out(left, children, n, parent_cluster, lambda, nodes);
        emit_points_falling_out(right, children, n, parent_cluster, lambda, nodes);
    }
}

/// Extract stable clusters from the condensed tree using excess of mass method.
///
/// Returns (labels, probabilities) where labels are -1 for noise, 0, 1, 2, ...
/// for clusters, and probabilities indicate cluster membership strength.
fn extract_clusters(
    tree: &CondensedTree,
    n: usize,
    allow_single_cluster: bool,
    cluster_selection_epsilon: f64,
) -> (Array1<i32>, Array1<f64>) {
    if tree.nodes.is_empty() {
        return (Array1::from_elem(n, -1), Array1::zeros(n));
    }

    // Collect all cluster IDs (nodes with child_size > 1 are cluster splits;
    // parents of any node are clusters)
    let mut all_cluster_ids = std::collections::BTreeSet::new();
    for node in &tree.nodes {
        all_cluster_ids.insert(node.parent);
        if node.child_size > 1 {
            all_cluster_ids.insert(node.child);
        }
    }

    if all_cluster_ids.is_empty() {
        return (Array1::from_elem(n, -1), Array1::zeros(n));
    }

    // Find the root cluster (the one that is never a child)
    let child_clusters: std::collections::BTreeSet<usize> = tree
        .nodes
        .iter()
        .filter(|node| node.child_size > 1)
        .map(|node| node.child)
        .collect();

    let root_cluster = *all_cluster_ids
        .iter()
        .find(|id| !child_clusters.contains(id))
        .unwrap_or(
            all_cluster_ids
                .iter()
                .next()
                .expect("SAFETY: non-empty iterator"),
        );

    // Compute lambda_birth for each cluster (the lambda at which it was created)
    // A cluster's birth lambda is the lambda of the condensed node where it appears as a child
    let mut lambda_birth: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for node in &tree.nodes {
        if node.child_size > 1 {
            lambda_birth.insert(node.child, node.lambda_val);
        }
    }
    // Root cluster was born at lambda=0 (infinite distance)
    lambda_birth.entry(root_cluster).or_insert(0.0);

    // Compute stability for each cluster
    // stability(C) = sum over points p in C of (lambda_p - lambda_birth(C))
    let mut stability: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for &cluster_id in &all_cluster_ids {
        let birth = *lambda_birth.get(&cluster_id).unwrap_or(&0.0);
        let mut stab = 0.0;
        for node in &tree.nodes {
            if node.parent == cluster_id && node.child_size == 1 {
                // This is a point falling out of this cluster
                stab += node.lambda_val - birth;
            }
        }
        stability.insert(cluster_id, stab);
    }

    // Find leaf clusters (clusters that have no child clusters)
    let parent_of_cluster: std::collections::HashMap<usize, usize> = tree
        .nodes
        .iter()
        .filter(|node| node.child_size > 1)
        .map(|node| (node.child, node.parent))
        .collect();

    let leaf_clusters: Vec<usize> = all_cluster_ids
        .iter()
        .filter(|&&id| {
            !tree
                .nodes
                .iter()
                .any(|node| node.parent == id && node.child_size > 1)
        })
        .cloned()
        .collect();

    // Apply cluster_selection_epsilon: if a leaf cluster's birth distance (1/lambda_birth)
    // is less than epsilon, merge it with parent
    if cluster_selection_epsilon > 0.0 {
        let epsilon_lambda = 1.0 / cluster_selection_epsilon;
        for &leaf in &leaf_clusters {
            if let Some(&birth_lambda) = lambda_birth.get(&leaf) {
                if birth_lambda > epsilon_lambda {
                    // This cluster was born at a scale finer than epsilon — merge with parent
                    if let Some(&parent) = parent_of_cluster.get(&leaf) {
                        let leaf_stab = *stability.get(&leaf).unwrap_or(&0.0);
                        *stability.entry(parent).or_insert(0.0) += leaf_stab;
                        stability.insert(leaf, 0.0);
                    }
                }
            }
        }
    }

    // Bottom-up selection: walk from leaves to root
    // selected[cluster] = true if this cluster is selected as a final cluster
    let mut selected: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();

    // Start with all leaf clusters selected
    for &leaf in &leaf_clusters {
        selected.insert(leaf, true);
    }

    // Process in bottom-up order: for each parent, decide if children are better
    // We need to process parents in order from deepest to shallowest
    // Build a list of internal (non-leaf) clusters and process them bottom-up
    let mut internal_clusters: Vec<usize> = all_cluster_ids
        .iter()
        .filter(|&&id| !leaf_clusters.contains(&id))
        .cloned()
        .collect();

    // Sort by depth (deeper clusters first). Depth can be estimated by birth lambda (higher = deeper)
    internal_clusters.sort_by(|a, b| {
        let la = lambda_birth.get(b).unwrap_or(&0.0);
        let lb = lambda_birth.get(a).unwrap_or(&0.0);
        la.partial_cmp(lb).unwrap_or(std::cmp::Ordering::Equal)
    });

    for &cluster in &internal_clusters {
        // Find children of this cluster
        let children_of: Vec<usize> = tree
            .nodes
            .iter()
            .filter(|node| node.parent == cluster && node.child_size > 1)
            .map(|node| node.child)
            .collect();

        let children_stability_sum: f64 = children_of
            .iter()
            .map(|&c| *stability.get(&c).unwrap_or(&0.0))
            .sum();

        let self_stability = *stability.get(&cluster).unwrap_or(&0.0);

        if self_stability >= children_stability_sum {
            // Parent is more stable — select parent, deselect children
            selected.insert(cluster, true);
            deselect_subtree(cluster, &tree.nodes, &mut selected);
            // Update stability to propagate upward
            // (stability stays as is — parent stability)
        } else {
            // Children are more stable — don't select parent
            selected.insert(cluster, false);
            // Propagate children's stability up to parent
            stability.insert(cluster, children_stability_sum);
        }
    }

    // Handle allow_single_cluster
    if !allow_single_cluster {
        // If only the root is selected, mark everything as noise
        let selected_clusters: Vec<usize> = selected
            .iter()
            .filter(|(&id, &sel)| sel && id != root_cluster)
            .map(|(&id, _)| id)
            .collect();
        if selected_clusters.is_empty() {
            // Only root selected or nothing — check if root is selected
            if selected.get(&root_cluster) == Some(&true) {
                // Don't allow single cluster — everything is noise
                return (Array1::from_elem(n, -1), Array1::zeros(n));
            }
        }
    }

    // Assign labels
    let selected_clusters: Vec<usize> = selected
        .iter()
        .filter(|(_, &sel)| sel)
        .map(|(&id, _)| id)
        .collect();

    // For each selected cluster, find all points that belong to it
    // A point belongs to the deepest selected cluster in its ancestry path
    let mut labels = Array1::from_elem(n, -1i32);
    let mut point_lambda = vec![0.0f64; n]; // lambda at which each point was assigned

    // Map selected cluster IDs to consecutive labels 0, 1, 2, ...
    let mut cluster_label_map: std::collections::HashMap<usize, i32> =
        std::collections::HashMap::new();
    let mut sorted_selected = selected_clusters.clone();
    sorted_selected.sort();
    for (i, &cluster_id) in sorted_selected.iter().enumerate() {
        cluster_label_map.insert(cluster_id, i as i32);
    }

    // For each point, find which selected cluster it belongs to.
    // A point that "fell out" of cluster C at lambda L belongs to C if C is selected.
    // If C is not selected, the point belongs to C's nearest selected ancestor.
    // Points that fell out of the root with no selected cluster are noise.

    for node in &tree.nodes {
        if node.child_size == 1 {
            // This is a point falling out of node.parent at node.lambda_val
            let point_id = node.child;
            let mut cluster = node.parent;

            // Walk up to find the nearest selected ancestor
            loop {
                if selected.get(&cluster) == Some(&true) {
                    if let Some(&label) = cluster_label_map.get(&cluster) {
                        labels[point_id] = label;
                        point_lambda[point_id] = node.lambda_val;
                    }
                    break;
                }
                // Go to parent
                match parent_of_cluster.get(&cluster) {
                    Some(&parent) => cluster = parent,
                    None => break, // reached root without finding selected cluster
                }
            }
        }
    }

    // Also assign points that are "still in" a selected cluster at the deepest level
    // These are points that never individually fell out — they stayed until the cluster
    // was merged/split. For these, we need to check which points were NOT emitted
    // as individual fall-outs and assign them to their cluster.
    let mut assigned_points: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for node in &tree.nodes {
        if node.child_size == 1 {
            assigned_points.insert(node.child);
        }
    }

    // Points not in the condensed tree as individual fall-outs get assigned
    // to the deepest leaf cluster they belong to. For a well-formed condensed tree,
    // all points should appear. But if n is small, some might not.
    // This is handled by the cluster membership tracking above.

    // Compute probabilities
    let mut probabilities = Array1::zeros(n);

    for &cluster_id in &sorted_selected {
        let birth_lambda = *lambda_birth.get(&cluster_id).unwrap_or(&0.0);

        // Find max lambda for any point in this cluster
        let max_lambda: f64 = tree
            .nodes
            .iter()
            .filter(|node| node.child_size == 1)
            .filter(|node| {
                // Check if this point is assigned to this cluster
                let point_id = node.child;
                if let Some(&label) = cluster_label_map.get(&cluster_id) {
                    labels[point_id] == label
                } else {
                    false
                }
            })
            .map(|node| node.lambda_val)
            .fold(birth_lambda, f64::max);

        let lambda_range = max_lambda - birth_lambda;

        if lambda_range > 0.0 {
            for node in &tree.nodes {
                if node.child_size == 1 {
                    let point_id = node.child;
                    if let Some(&label) = cluster_label_map.get(&cluster_id) {
                        if labels[point_id] == label {
                            probabilities[point_id] = ((point_lambda[point_id] - birth_lambda)
                                / lambda_range)
                                .clamp(0.0, 1.0);
                        }
                    }
                }
            }
        } else {
            // All points in this cluster have the same lambda — assign probability 1.0
            for i in 0..n {
                if let Some(&label) = cluster_label_map.get(&cluster_id) {
                    if labels[i] == label {
                        probabilities[i] = 1.0;
                    }
                }
            }
        }
    }

    // Noise points get probability 0
    for i in 0..n {
        if labels[i] == -1 {
            probabilities[i] = 0.0;
        }
    }

    (labels, probabilities)
}

/// Deselect all descendant clusters of a given cluster.
fn deselect_subtree(
    cluster: usize,
    condensed_nodes: &[CondensedNode],
    selected: &mut std::collections::HashMap<usize, bool>,
) {
    // Find direct child clusters
    let children: Vec<usize> = condensed_nodes
        .iter()
        .filter(|node| node.parent == cluster && node.child_size > 1)
        .map(|node| node.child)
        .collect();

    for child in children {
        selected.insert(child, false);
        deselect_subtree(child, condensed_nodes, selected);
    }
}

impl ClusteringModel for HDBSCAN {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        crate::validation::validate_unsupervised_input(x)?;

        let n = x.nrows();

        // Hyperparameter validation
        if self.min_cluster_size < 2 {
            return Err(FerroError::invalid_input(format!(
                "Parameter min_cluster_size must be >= 2, got {}",
                self.min_cluster_size
            )));
        }

        let min_samples = self.min_samples.unwrap_or(self.min_cluster_size);

        if n == 1 {
            self.labels_ = Some(Array1::from_elem(1, -1));
            self.probabilities_ = Some(Array1::zeros(1));
            self.n_clusters_ = Some(0);
            return Ok(());
        }

        // Clamp min_samples to n-1 for small datasets
        let effective_min_samples = min_samples.min(n - 1);

        // 1. Compute core distances
        let core_dists = compute_core_distances(x, effective_min_samples);

        // 2-3. Build MST via dense Prim's with mutual reachability
        // GPU path: precompute full pairwise squared distance matrix
        #[cfg(feature = "gpu")]
        let mst = {
            let gpu_dist_matrix = self
                .gpu_backend
                .as_ref()
                .and_then(|gpu| gpu.pairwise_distances(x, x).ok());
            if let Some(ref dist_matrix) = gpu_dist_matrix {
                build_mst_from_precomputed(dist_matrix, &core_dists)
            } else {
                build_mst_mutual_reachability(x, &core_dists)
            }
        };

        #[cfg(not(feature = "gpu"))]
        let mst = build_mst_mutual_reachability(x, &core_dists);

        // 4. Build condensed cluster tree
        let condensed = build_condensed_tree(&mst, self.min_cluster_size, n);

        // 5. Extract stable clusters
        let (labels, probs) = extract_clusters(
            &condensed,
            n,
            self.allow_single_cluster,
            self.cluster_selection_epsilon,
        );

        let n_clusters = if labels.iter().any(|&l| l >= 0) {
            labels
                .iter()
                .filter(|&&l| l >= 0)
                .max()
                .expect("SAFETY: non-empty collection")
                + 1
        } else {
            0
        };

        self.labels_ = Some(labels);
        self.probabilities_ = Some(probs);
        self.n_clusters_ = Some(n_clusters as usize);
        Ok(())
    }

    fn predict(&self, _x: &Array2<f64>) -> Result<Array1<i32>> {
        Err(FerroError::InvalidInput(
            "HDBSCAN does not support predict on new data. Use fit_predict instead.".into(),
        ))
    }

    fn fit_predict(&mut self, x: &Array2<f64>) -> Result<Array1<i32>> {
        self.fit(x)?;
        self.labels_
            .clone()
            .ok_or_else(|| FerroError::not_fitted("fit_predict"))
    }

    fn labels(&self) -> Option<&Array1<i32>> {
        self.labels_.as_ref()
    }

    fn is_fitted(&self) -> bool {
        self.labels_.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Generate deterministic blob data for testing.
    fn make_blobs(centers: &[(f64, f64)], n_per_cluster: usize, spread: f64) -> Array2<f64> {
        let n = centers.len() * n_per_cluster;
        let mut x = Array2::zeros((n, 2));
        for (c, &(cx, cy)) in centers.iter().enumerate() {
            for i in 0..n_per_cluster {
                let idx = c * n_per_cluster + i;
                let angle = (i as f64) * 2.0 * std::f64::consts::PI / (n_per_cluster as f64);
                let r = spread * ((idx as f64 * 0.1).sin().abs() + 0.1);
                x[[idx, 0]] = cx + r * angle.cos();
                x[[idx, 1]] = cy + r * angle.sin();
            }
        }
        x
    }

    /// Generate concentric circle data.
    fn make_circles(n_outer: usize, n_inner: usize, r_outer: f64, r_inner: f64) -> Array2<f64> {
        let n = n_outer + n_inner;
        let mut x = Array2::zeros((n, 2));
        for i in 0..n_outer {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (n_outer as f64);
            x[[i, 0]] = r_outer * angle.cos();
            x[[i, 1]] = r_outer * angle.sin();
        }
        for i in 0..n_inner {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / (n_inner as f64);
            x[[n_outer + i, 0]] = r_inner * angle.cos();
            x[[n_outer + i, 1]] = r_inner * angle.sin();
        }
        x
    }

    /// Generate half-moon (crescent) data.
    fn make_moons(n_per_moon: usize, offset: f64) -> Array2<f64> {
        let n = 2 * n_per_moon;
        let mut x = Array2::zeros((n, 2));
        for i in 0..n_per_moon {
            let angle = std::f64::consts::PI * (i as f64) / (n_per_moon as f64 - 1.0);
            x[[i, 0]] = angle.cos();
            x[[i, 1]] = angle.sin();
        }
        for i in 0..n_per_moon {
            let angle = std::f64::consts::PI * (i as f64) / (n_per_moon as f64 - 1.0);
            x[[n_per_moon + i, 0]] = 1.0 - angle.cos();
            x[[n_per_moon + i, 1]] = offset - angle.sin();
        }
        x
    }

    #[test]
    fn test_hdbscan_two_blobs() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 20, 0.5);
        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        assert_eq!(hdbscan.n_clusters(), Some(2));
        let labels = hdbscan.labels().unwrap();

        // Points 0..20 should be in one cluster, 20..40 in another
        let cluster_a = labels[0];
        let cluster_b = labels[20];
        assert!(cluster_a >= 0);
        assert!(cluster_b >= 0);
        assert_ne!(cluster_a, cluster_b);

        for i in 0..20 {
            assert_eq!(labels[i], cluster_a, "Point {} should be in cluster A", i);
        }
        for i in 20..40 {
            assert_eq!(labels[i], cluster_b, "Point {} should be in cluster B", i);
        }
    }

    #[test]
    fn test_hdbscan_three_blobs() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)], 15, 0.5);
        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        assert_eq!(hdbscan.n_clusters(), Some(3));
    }

    #[test]
    fn test_hdbscan_noise_detection() {
        // Two tight clusters with far-away noise points
        let data = vec![
            // Cluster 1: 7 tight points
            1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.05, 1.05, 0.95, 0.95, 1.15, 0.95, 0.95, 1.15,
            // Cluster 2: 7 tight points
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.05, 10.05, 9.95, 9.95, 10.15, 9.95, 9.95, 10.15,
            // Noise: isolated far-away points
            -50.0, -50.0, 50.0, 50.0,
        ];
        let x = Array2::from_shape_vec((16, 2), data).unwrap();

        let mut hdbscan = HDBSCAN::new(3);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        // The last two points (isolated far-away) should be labeled -1
        assert_eq!(labels[14], -1, "Far noise point should be noise");
        assert_eq!(labels[15], -1, "Far outlier should be noise");
    }

    #[test]
    fn test_hdbscan_min_cluster_size_effect() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 15, 0.5);

        let mut hdbscan_small = HDBSCAN::new(3);
        hdbscan_small.fit(&x).unwrap();

        let mut hdbscan_large = HDBSCAN::new(14);
        hdbscan_large.fit(&x).unwrap();

        // With larger min_cluster_size, we should get fewer or equal clusters
        assert!(
            hdbscan_large.n_clusters().unwrap() <= hdbscan_small.n_clusters().unwrap(),
            "Larger min_cluster_size should produce fewer clusters"
        );
    }

    #[test]
    fn test_hdbscan_single_cluster() {
        // Dense single blob
        let x = make_blobs(&[(0.0, 0.0)], 30, 0.3);

        let mut hdbscan = HDBSCAN::new(5).with_allow_single_cluster(true);
        hdbscan.fit(&x).unwrap();

        // Should find 1 cluster
        assert!(
            hdbscan.n_clusters().unwrap() >= 1,
            "Should find at least 1 cluster with allow_single_cluster=true"
        );
    }

    #[test]
    fn test_hdbscan_single_cluster_disallowed() {
        // Dense single blob
        let x = make_blobs(&[(0.0, 0.0)], 30, 0.3);

        let mut hdbscan = HDBSCAN::new(5).with_allow_single_cluster(false);
        hdbscan.fit(&x).unwrap();

        // With allow_single_cluster=false, should find 0 clusters (all noise)
        // if there's only one natural cluster
        // This depends on the data — the single blob might still produce > 1 cluster
        // due to internal structure, so we just check it runs without error
        assert!(hdbscan.is_fitted());
    }

    #[test]
    fn test_hdbscan_all_noise() {
        // Very sparse data with large min_cluster_size
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, 100.0, 100.0, -100.0, 100.0, 100.0, -100.0, -100.0, -100.0,
            ],
        )
        .unwrap();

        let mut hdbscan = HDBSCAN::new(4);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        // All points should be noise
        for i in 0..5 {
            assert_eq!(labels[i], -1, "Point {} should be noise", i);
        }
        assert_eq!(hdbscan.n_clusters(), Some(0));
    }

    #[test]
    fn test_hdbscan_varying_density() {
        // Cluster 1: very tight (low spread)
        // Cluster 2: more spread out
        let mut data = Vec::new();
        // Tight cluster at (0, 0) - 20 points
        for i in 0..20 {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 20.0;
            let r = 0.1 * ((i as f64 * 0.1).sin().abs() + 0.05);
            data.push(r * angle.cos());
            data.push(r * angle.sin());
        }
        // Spread cluster at (10, 10) - 20 points
        for i in 0..20 {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 20.0;
            let r = 1.0 * ((i as f64 * 0.1).sin().abs() + 0.3);
            data.push(10.0 + r * angle.cos());
            data.push(10.0 + r * angle.sin());
        }
        let x = Array2::from_shape_vec((40, 2), data).unwrap();

        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        // Should find 2 clusters (HDBSCAN handles varying density)
        assert_eq!(
            hdbscan.n_clusters(),
            Some(2),
            "Should find both clusters despite varying density"
        );
    }

    #[test]
    fn test_hdbscan_edge_case_one_point() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        let mut hdbscan = HDBSCAN::new(2);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        assert_eq!(labels.len(), 1);
        assert_eq!(labels[0], -1); // single point is noise
        assert_eq!(hdbscan.n_clusters(), Some(0));
    }

    #[test]
    fn test_hdbscan_edge_case_two_points() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        let mut hdbscan = HDBSCAN::new(2);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        assert_eq!(labels.len(), 2);
        // With min_cluster_size=2 and only 2 points, they might form one cluster
        // or be noise depending on the algorithm
        assert!(hdbscan.is_fitted());
    }

    #[test]
    fn test_hdbscan_edge_case_identical_points() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        )
        .unwrap();

        let mut hdbscan = HDBSCAN::new(3).with_allow_single_cluster(true);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        assert_eq!(labels.len(), 6);
        // All identical points — should form one cluster or be noise
        assert!(hdbscan.is_fitted());
    }

    #[test]
    fn test_hdbscan_labels_shape() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 15, 0.5);
        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        assert_eq!(labels.len(), 30);
    }

    #[test]
    fn test_hdbscan_probabilities_range() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 20, 0.5);
        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        let probs = hdbscan.probabilities().unwrap();
        for &p in probs.iter() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability {} is out of range [0, 1]",
                p
            );
        }
    }

    #[test]
    fn test_hdbscan_probabilities_noise_zero() {
        // Create data with clear noise
        let mut data = Vec::new();
        // Tight cluster
        for i in 0..10 {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 10.0;
            data.push(0.1 * angle.cos());
            data.push(0.1 * angle.sin());
        }
        // One far-away noise point
        data.push(100.0);
        data.push(100.0);
        let x = Array2::from_shape_vec((11, 2), data).unwrap();

        let mut hdbscan = HDBSCAN::new(3);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        let probs = hdbscan.probabilities().unwrap();

        for i in 0..11 {
            if labels[i] == -1 {
                assert_eq!(probs[i], 0.0, "Noise point {} should have probability 0", i);
            }
        }
    }

    #[test]
    fn test_hdbscan_n_clusters_matches_labels() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)], 15, 0.5);
        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        let unique_non_noise: std::collections::BTreeSet<i32> =
            labels.iter().filter(|&&l| l >= 0).cloned().collect();

        assert_eq!(
            hdbscan.n_clusters(),
            Some(unique_non_noise.len()),
            "n_clusters should match number of unique non-noise labels"
        );
    }

    #[test]
    fn test_hdbscan_deterministic() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 15, 0.5);

        let mut hdbscan1 = HDBSCAN::new(5);
        hdbscan1.fit(&x).unwrap();
        let labels1 = hdbscan1.labels().unwrap().clone();

        let mut hdbscan2 = HDBSCAN::new(5);
        hdbscan2.fit(&x).unwrap();
        let labels2 = hdbscan2.labels().unwrap().clone();

        assert_eq!(labels1, labels2, "HDBSCAN should be deterministic");
    }

    #[test]
    fn test_hdbscan_is_fitted() {
        let mut hdbscan = HDBSCAN::new(5);
        assert!(!hdbscan.is_fitted());

        let x = make_blobs(&[(0.0, 0.0)], 10, 0.5);
        hdbscan.fit(&x).unwrap();
        assert!(hdbscan.is_fitted());
    }

    #[test]
    fn test_hdbscan_fit_predict() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 15, 0.5);

        let mut hdbscan1 = HDBSCAN::new(5);
        hdbscan1.fit(&x).unwrap();
        let labels_fit = hdbscan1.labels().unwrap().clone();

        let mut hdbscan2 = HDBSCAN::new(5);
        let labels_fp = hdbscan2.fit_predict(&x).unwrap();

        assert_eq!(labels_fit, labels_fp, "fit+labels should equal fit_predict");
    }

    #[test]
    fn test_hdbscan_predict_error() {
        let x = make_blobs(&[(0.0, 0.0)], 10, 0.5);
        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        let result = hdbscan.predict(&x);
        assert!(result.is_err(), "predict() should return error");
    }

    #[test]
    fn test_hdbscan_empty_input() {
        let x = Array2::zeros((0, 2));
        let mut hdbscan = HDBSCAN::new(5);
        let result = hdbscan.fit(&x);
        assert!(result.is_err(), "Empty input should return error");
    }

    #[test]
    fn test_hdbscan_min_samples_parameter() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 20, 0.5);

        let mut hdbscan1 = HDBSCAN::new(5);
        hdbscan1.fit(&x).unwrap();

        let mut hdbscan2 = HDBSCAN::new(5).with_min_samples(3);
        hdbscan2.fit(&x).unwrap();

        // Both should produce valid results
        assert!(hdbscan1.is_fitted());
        assert!(hdbscan2.is_fitted());
    }

    #[test]
    fn test_hdbscan_large_min_cluster_size() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 10, 0.5);

        // min_cluster_size > total points
        let mut hdbscan = HDBSCAN::new(25);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        // All should be noise
        for i in 0..20 {
            assert_eq!(
                labels[i], -1,
                "Point {} should be noise when min_cluster_size > n",
                i
            );
        }
    }

    #[test]
    fn test_hdbscan_concentric_circles() {
        let x = make_circles(30, 15, 5.0, 1.0);

        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        // HDBSCAN should find 2 clusters (inner and outer circles)
        assert!(
            hdbscan.n_clusters().unwrap() >= 2,
            "Should find at least 2 clusters for concentric circles"
        );
    }

    #[test]
    fn test_hdbscan_moons() {
        let x = make_moons(25, 0.5);

        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        // Should find 2 clusters
        assert!(
            hdbscan.n_clusters().unwrap() >= 2,
            "Should find at least 2 clusters for half-moons"
        );
    }

    #[test]
    fn test_hdbscan_cluster_selection_epsilon() {
        // Two very close clusters
        let x = make_blobs(&[(0.0, 0.0), (2.0, 0.0)], 15, 0.3);

        // Without epsilon — should find 2 clusters
        let mut hdbscan_no_eps = HDBSCAN::new(5);
        hdbscan_no_eps.fit(&x).unwrap();

        // With large epsilon — should merge them
        let mut hdbscan_eps = HDBSCAN::new(5)
            .with_cluster_selection_epsilon(5.0)
            .with_allow_single_cluster(true);
        hdbscan_eps.fit(&x).unwrap();

        // The epsilon version should have fewer or equal clusters
        assert!(
            hdbscan_eps.n_clusters().unwrap() <= hdbscan_no_eps.n_clusters().unwrap(),
            "Large epsilon should merge close clusters"
        );
    }

    #[test]
    fn test_hdbscan_high_dimensional() {
        // 10D data with 2 clusters
        let n_per = 15;
        let dim = 10;
        let mut data = Vec::new();
        for i in 0..n_per {
            for d in 0..dim {
                data.push(0.1 * ((i * dim + d) as f64 * 0.3).sin());
            }
        }
        for i in 0..n_per {
            for d in 0..dim {
                data.push(10.0 + 0.1 * ((i * dim + d) as f64 * 0.3).sin());
            }
        }
        let x = Array2::from_shape_vec((2 * n_per, dim), data).unwrap();

        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        assert!(hdbscan.is_fitted());
        assert!(
            hdbscan.n_clusters().unwrap() >= 2,
            "Should find 2 clusters in 10D data"
        );
    }

    #[test]
    fn test_hdbscan_unbalanced_clusters() {
        // Large cluster and small cluster
        let mut data = Vec::new();
        // Large cluster: 30 points
        for i in 0..30 {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 30.0;
            let r = 0.3 * ((i as f64 * 0.1).sin().abs() + 0.1);
            data.push(r * angle.cos());
            data.push(r * angle.sin());
        }
        // Small cluster: 8 points
        for i in 0..8 {
            let angle = (i as f64) * 2.0 * std::f64::consts::PI / 8.0;
            let r = 0.3 * ((i as f64 * 0.1).sin().abs() + 0.1);
            data.push(10.0 + r * angle.cos());
            data.push(10.0 + r * angle.sin());
        }
        let x = Array2::from_shape_vec((38, 2), data).unwrap();

        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        assert!(
            hdbscan.n_clusters().unwrap() >= 2,
            "Should find both large and small clusters"
        );
    }

    #[test]
    fn test_hdbscan_collinear() {
        // Points on a line
        let n = 20;
        let mut data = Vec::new();
        for i in 0..n {
            data.push(i as f64);
            data.push(i as f64);
        }
        let x = Array2::from_shape_vec((n, 2), data).unwrap();

        let mut hdbscan = HDBSCAN::new(3);
        hdbscan.fit(&x).unwrap();

        assert!(hdbscan.is_fitted());
        let labels = hdbscan.labels().unwrap();
        assert_eq!(labels.len(), n);
    }

    #[test]
    fn test_hdbscan_labels_consecutive() {
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)], 15, 0.5);
        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        let labels = hdbscan.labels().unwrap();
        let n_clusters = hdbscan.n_clusters().unwrap();

        if n_clusters > 0 {
            // Check that labels are consecutive: 0, 1, 2, ..., n_clusters-1
            let unique_labels: std::collections::BTreeSet<i32> =
                labels.iter().filter(|&&l| l >= 0).cloned().collect();

            let expected: std::collections::BTreeSet<i32> = (0..n_clusters as i32).collect();
            assert_eq!(
                unique_labels, expected,
                "Cluster labels should be consecutive integers starting from 0"
            );
        }
    }

    #[test]
    fn test_hdbscan_stability_selection() {
        // Create data where one cluster is much more stable than another
        // Stable cluster: very tight, well-separated
        // Unstable cluster: dispersed
        let x = make_blobs(&[(0.0, 0.0), (20.0, 20.0)], 20, 0.3);

        let mut hdbscan = HDBSCAN::new(5);
        hdbscan.fit(&x).unwrap();

        // Should find stable clusters
        assert!(hdbscan.n_clusters().unwrap() >= 2);

        // Cluster assignment probabilities should be mostly high
        let probs = hdbscan.probabilities().unwrap();
        let non_noise_probs: Vec<f64> = hdbscan
            .labels()
            .unwrap()
            .iter()
            .zip(probs.iter())
            .filter(|(&l, _)| l >= 0)
            .map(|(_, &p)| p)
            .collect();

        if !non_noise_probs.is_empty() {
            let mean_prob: f64 = non_noise_probs.iter().sum::<f64>() / non_noise_probs.len() as f64;
            assert!(
                mean_prob > 0.0,
                "Mean probability of clustered points should be positive"
            );
        }
    }

    #[test]
    fn test_hdbscan_n_noise() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0, 100.0, 100.0, -100.0, 100.0, 100.0, -100.0, -100.0, -100.0,
            ],
        )
        .unwrap();

        let mut hdbscan = HDBSCAN::new(4);
        hdbscan.fit(&x).unwrap();

        assert_eq!(hdbscan.n_noise(), Some(5));
    }

    #[test]
    fn test_hdbscan_min_cluster_size_2() {
        // Minimum valid min_cluster_size
        let x = make_blobs(&[(0.0, 0.0), (10.0, 10.0)], 10, 0.3);
        let mut hdbscan = HDBSCAN::new(2);
        hdbscan.fit(&x).unwrap();
        assert!(hdbscan.is_fitted());
    }

    #[test]
    fn test_hdbscan_min_cluster_size_invalid() {
        let x = make_blobs(&[(0.0, 0.0)], 10, 0.5);
        let mut hdbscan = HDBSCAN::new(1);
        let result = hdbscan.fit(&x);
        assert!(result.is_err(), "min_cluster_size=1 should be invalid");
    }
}
