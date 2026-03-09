//! Vantage-Point Tree for efficient nearest neighbor search.
//!
//! Used by Barnes-Hut t-SNE to compute sparse P matrices with only
//! k nearest neighbors per point, reducing the complexity from O(N^2) to O(N log N).

#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A point with its index in the original dataset.
#[derive(Debug, Clone)]
struct VPPoint {
    index: usize,
    coords: Vec<f64>,
}

/// A node in the vantage-point tree.
#[derive(Debug)]
enum VPNode {
    Leaf {
        points: Vec<VPPoint>,
    },
    Internal {
        vantage: VPPoint,
        mu: f64,            // median distance from vantage point
        left: Box<VPNode>,  // points with dist <= mu
        right: Box<VPNode>, // points with dist > mu
    },
}

/// Neighbor candidate for the priority queue (max-heap by distance).
#[derive(Debug, Clone)]
struct Neighbor {
    index: usize,
    distance: f64,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority (so we can pop the farthest)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Vantage-Point Tree for O(N log N) nearest neighbor search.
///
/// The VP-tree partitions space by selecting a vantage point and splitting
/// remaining points into those closer than the median distance (left subtree)
/// and those farther (right subtree).
#[derive(Debug)]
pub struct VPTree {
    root: VPNode,
    n_points: usize,
}

/// Compute squared Euclidean distance between two points.
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

impl VPTree {
    /// Build a VP-tree from a dataset.
    ///
    /// Each row in `data` is a point. The tree stores indices into the original data.
    pub fn new(data: &[Vec<f64>]) -> Self {
        let n = data.len();
        let mut points: Vec<VPPoint> = data
            .iter()
            .enumerate()
            .map(|(i, coords)| VPPoint {
                index: i,
                coords: coords.clone(),
            })
            .collect();

        let root = Self::build(&mut points);
        Self { root, n_points: n }
    }

    /// Build a VP-tree from an ndarray.
    pub fn from_array(data: &ndarray::Array2<f64>) -> Self {
        let rows: Vec<Vec<f64>> = data.rows().into_iter().map(|r| r.to_vec()).collect();
        Self::new(&rows)
    }

    /// Recursively build the tree.
    fn build(points: &mut [VPPoint]) -> VPNode {
        if points.len() <= 8 {
            return VPNode::Leaf {
                points: points.to_vec(),
            };
        }

        // Use the last point as the vantage point
        let vantage = points.last().unwrap().clone();
        let n = points.len();
        let rest = &mut points[..n - 1];

        // Compute distances from vantage point to all others
        let mut distances: Vec<(usize, f64)> = rest
            .iter()
            .enumerate()
            .map(|(i, p)| (i, sq_dist(&vantage.coords, &p.coords).sqrt()))
            .collect();

        // Find median distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let median_idx = distances.len() / 2;
        let mu = if distances.is_empty() {
            0.0
        } else {
            distances[median_idx].1
        };

        // Partition into left (dist <= mu) and right (dist > mu)
        let mut left_points: Vec<VPPoint> = Vec::new();
        let mut right_points: Vec<VPPoint> = Vec::new();

        for &(idx, dist) in &distances {
            if dist <= mu {
                left_points.push(rest[idx].clone());
            } else {
                right_points.push(rest[idx].clone());
            }
        }

        // Handle degenerate case where all points end up on one side
        if left_points.is_empty() || right_points.is_empty() {
            let half = rest.len() / 2;
            left_points.clear();
            right_points.clear();
            for (i, &(idx, _)) in distances.iter().enumerate() {
                if i < half {
                    left_points.push(rest[idx].clone());
                } else {
                    right_points.push(rest[idx].clone());
                }
            }
        }

        let left = Box::new(Self::build(&mut left_points));
        let right = Box::new(Self::build(&mut right_points));

        VPNode::Internal {
            vantage,
            mu,
            left,
            right,
        }
    }

    /// Search for the k nearest neighbors of a query point.
    ///
    /// Returns a vector of `(index, distance)` pairs sorted by distance (ascending).
    /// The index refers to the position in the original data passed to `new()`.
    pub fn search(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        let k = k.min(self.n_points);
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut tau = f64::INFINITY; // current kth-nearest distance

        Self::search_node(&self.root, query, k, &mut heap, &mut tau);

        let mut results: Vec<(usize, f64)> =
            heap.into_iter().map(|n| (n.index, n.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Recursively search a node.
    fn search_node(
        node: &VPNode,
        query: &[f64],
        k: usize,
        heap: &mut BinaryHeap<Neighbor>,
        tau: &mut f64,
    ) {
        match node {
            VPNode::Leaf { points } => {
                for p in points {
                    let dist = sq_dist(query, &p.coords).sqrt();
                    if heap.len() < k {
                        heap.push(Neighbor {
                            index: p.index,
                            distance: dist,
                        });
                        if heap.len() == k {
                            *tau = heap.peek().unwrap().distance;
                        }
                    } else if dist < *tau {
                        heap.pop();
                        heap.push(Neighbor {
                            index: p.index,
                            distance: dist,
                        });
                        *tau = heap.peek().unwrap().distance;
                    }
                }
            }
            VPNode::Internal {
                vantage,
                mu,
                left,
                right,
            } => {
                let dist = sq_dist(query, &vantage.coords).sqrt();

                // Consider the vantage point itself
                if heap.len() < k {
                    heap.push(Neighbor {
                        index: vantage.index,
                        distance: dist,
                    });
                    if heap.len() == k {
                        *tau = heap.peek().unwrap().distance;
                    }
                } else if dist < *tau {
                    heap.pop();
                    heap.push(Neighbor {
                        index: vantage.index,
                        distance: dist,
                    });
                    *tau = heap.peek().unwrap().distance;
                }

                // Decide which subtree to search first
                if dist <= *mu {
                    // Query is closer to vantage => search left first
                    if dist - *tau <= *mu {
                        Self::search_node(left, query, k, heap, tau);
                    }
                    if dist + *tau > *mu {
                        Self::search_node(right, query, k, heap, tau);
                    }
                } else {
                    // Query is farther from vantage => search right first
                    if dist + *tau > *mu {
                        Self::search_node(right, query, k, heap, tau);
                    }
                    if dist - *tau <= *mu {
                        Self::search_node(left, query, k, heap, tau);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    #[test]
    fn test_vptree_correct_nearest_neighbors() {
        // Small dataset — verify VP-tree finds the same neighbors as brute force
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
            vec![5.5, 5.0],
            vec![5.0, 5.5],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        let tree = VPTree::new(&data);

        // Query for 3 nearest neighbors of point [0.0, 0.0]
        let query = vec![0.0, 0.0];
        let neighbors = tree.search(&query, 3);
        assert_eq!(neighbors.len(), 3);

        // The 3 nearest should be indices 0, 1, 2 (the first cluster)
        let indices: Vec<usize> = neighbors.iter().map(|n| n.0).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));

        // Distances should be sorted ascending
        for i in 1..neighbors.len() {
            assert!(neighbors[i].1 >= neighbors[i - 1].1);
        }
    }

    #[test]
    fn test_vptree_handles_duplicates() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![5.0, 6.0],
            vec![5.0, 6.0],
        ];
        let tree = VPTree::new(&data);

        let neighbors = tree.search(&[1.0, 2.0], 3);
        assert_eq!(neighbors.len(), 3);

        // The 3 nearest should all have distance 0.0 (exact duplicates)
        for n in &neighbors {
            assert!(
                n.1 < 1e-10,
                "Expected near-zero distance for duplicates, got {}",
                n.1
            );
        }
    }

    #[test]
    fn test_vptree_all_points() {
        // Request more neighbors than exist
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let tree = VPTree::new(&data);
        let neighbors = tree.search(&[0.0, 0.0], 10);
        assert_eq!(neighbors.len(), 3);
    }

    #[test]
    fn test_vptree_matches_brute_force() {
        // Generate random data and verify VP-tree matches brute force
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 100;
        let dim = 10;
        let k = 5;

        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect())
            .collect();

        let tree = VPTree::new(&data);

        // Check nearest neighbors for several query points
        for qi in [0, 10, 50, 99] {
            let query = &data[qi];
            let vp_neighbors = tree.search(query, k);

            // Brute force
            let mut bf_dists: Vec<(usize, f64)> = data
                .iter()
                .enumerate()
                .map(|(i, p)| (i, sq_dist(query, p).sqrt()))
                .collect();
            bf_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let bf_neighbors: Vec<(usize, f64)> = bf_dists.into_iter().take(k).collect();

            // Same indices
            let vp_indices: Vec<usize> = vp_neighbors.iter().map(|n| n.0).collect();
            let bf_indices: Vec<usize> = bf_neighbors.iter().map(|n| n.0).collect();
            assert_eq!(
                vp_indices, bf_indices,
                "VP-tree and brute force disagree for query {}",
                qi
            );
        }
    }

    #[test]
    fn test_vptree_single_point() {
        let data = vec![vec![3.0, 4.0]];
        let tree = VPTree::new(&data);
        let neighbors = tree.search(&[0.0, 0.0], 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, 0);
        assert!((neighbors[0].1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vptree_empty_search() {
        let data = vec![vec![1.0], vec![2.0]];
        let tree = VPTree::new(&data);
        let neighbors = tree.search(&[0.0], 0);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_vptree_high_dimensional() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let n = 50;
        let dim = 50;

        let data: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect())
            .collect();

        let tree = VPTree::new(&data);
        let neighbors = tree.search(&data[0], 5);
        assert_eq!(neighbors.len(), 5);
        // First neighbor should be the query point itself (distance 0)
        assert_eq!(neighbors[0].0, 0);
        assert!(neighbors[0].1 < 1e-10);
    }
}
