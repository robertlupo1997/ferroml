//! Agglomerative (Hierarchical) Clustering
//!
//! Bottom-up hierarchical clustering that merges the closest pair of clusters
//! until the desired number of clusters is reached.
//!
//! ## Linkage Methods
//!
//! - **Single**: minimum distance between any two points in different clusters
//! - **Complete**: maximum distance between any two points in different clusters
//! - **Average**: average distance between all pairs of points in different clusters
//! - **Ward**: minimizes total within-cluster variance (Euclidean only)

use crate::clustering::ClusteringModel;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Linkage method for agglomerative clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Linkage {
    /// Single linkage: min distance between clusters
    Single,
    /// Complete linkage: max distance between clusters
    Complete,
    /// Average linkage: mean distance between clusters
    Average,
    /// Ward linkage: minimize within-cluster variance
    Ward,
}

impl Default for Linkage {
    fn default() -> Self {
        Self::Ward
    }
}

/// Agglomerative (hierarchical) clustering.
///
/// Recursively merges the pair of clusters that minimizes a given linkage
/// criterion, producing a hierarchy (dendrogram). The process stops when
/// `n_clusters` clusters remain.
///
/// ## Example
///
/// ```
/// # use ferroml_core::clustering::agglomerative::AgglomerativeClustering;
/// # use ferroml_core::clustering::ClusteringModel;
/// # use ndarray::Array2;
/// # let x = Array2::from_shape_vec((6, 2), vec![
/// #     1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 8.0, 8.0, 8.2, 7.8, 7.8, 8.2
/// # ]).unwrap();
/// let mut model = AgglomerativeClustering::new(3);
/// let labels = model.fit_predict(&x).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgglomerativeClustering {
    /// Number of clusters to produce
    pub n_clusters: usize,
    /// Linkage criterion
    pub linkage: Linkage,

    // Fitted state
    labels: Option<Array1<i32>>,
    n_features: Option<usize>,
    /// Merge history: (cluster_a, cluster_b, distance, new_cluster_size)
    children: Option<Vec<(usize, usize, f64, usize)>>,
}

impl AgglomerativeClustering {
    /// Create a new AgglomerativeClustering with the given number of clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
            labels: None,
            n_features: None,
            children: None,
        }
    }

    /// Set the linkage criterion.
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Get the merge history (dendrogram data).
    pub fn children(&self) -> Option<&Vec<(usize, usize, f64, usize)>> {
        self.children.as_ref()
    }
}

impl ClusteringModel for AgglomerativeClustering {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(FerroError::invalid_input("Empty input data"));
        }
        if self.n_clusters == 0 || self.n_clusters > n_samples {
            return Err(FerroError::invalid_input(format!(
                "n_clusters ({}) must be between 1 and n_samples ({})",
                self.n_clusters, n_samples
            )));
        }

        // Each sample starts as its own cluster
        let mut cluster_id: Vec<usize> = (0..n_samples).collect();
        let mut active_clusters: Vec<bool> = vec![true; n_samples];
        let mut cluster_members: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        let mut n_active = n_samples;
        let mut children = Vec::new();

        // Precompute pairwise distance matrix (condensed form — upper triangle)
        // For Ward: also track cluster centroids and sizes
        let mut centroids: Vec<Array1<f64>> = (0..n_samples).map(|i| x.row(i).to_owned()).collect();
        let mut sizes: Vec<usize> = vec![1; n_samples];

        // Distance matrix: dist[i][j] for i < j
        let mut dist = Array2::from_elem((n_samples, n_samples), f64::INFINITY);
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let d =
                    squared_distance(x.row(i).as_slice().unwrap(), x.row(j).as_slice().unwrap());
                let d = match self.linkage {
                    Linkage::Ward => d, // Ward uses squared distances internally
                    _ => d.sqrt(),
                };
                dist[[i, j]] = d;
                dist[[j, i]] = d;
            }
        }

        let mut next_id = n_samples;

        // Merge until n_clusters remain
        while n_active > self.n_clusters {
            // Find the pair of active clusters with minimum distance
            let mut min_dist = f64::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 0;

            for i in 0..cluster_id.len() {
                if !active_clusters[i] {
                    continue;
                }
                for j in (i + 1)..cluster_id.len() {
                    if !active_clusters[j] {
                        continue;
                    }
                    if dist[[i, j]] < min_dist {
                        min_dist = dist[[i, j]];
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            let new_size = sizes[merge_i] + sizes[merge_j];

            // Record merge
            let display_dist = match self.linkage {
                Linkage::Ward => min_dist.sqrt(),
                _ => min_dist,
            };
            children.push((
                cluster_id[merge_i],
                cluster_id[merge_j],
                display_dist,
                new_size,
            ));

            // Merge: assign merge_j members to merge_i
            let members_j = cluster_members[merge_j].clone();
            cluster_members[merge_i].extend_from_slice(&members_j);
            cluster_members[merge_j].clear();

            // Update centroid for merge_i (needed for Ward)
            let ni = sizes[merge_i] as f64;
            let nj = sizes[merge_j] as f64;
            let new_centroid = (&centroids[merge_i] * ni + &centroids[merge_j] * nj) / (ni + nj);
            centroids[merge_i] = new_centroid;
            sizes[merge_i] = new_size;

            // Update cluster ID
            cluster_id[merge_i] = next_id;
            next_id += 1;

            // Deactivate merge_j
            active_clusters[merge_j] = false;
            n_active -= 1;

            // Update distances from the new merged cluster to all other active clusters
            for k in 0..cluster_id.len() {
                if !active_clusters[k] || k == merge_i {
                    continue;
                }

                let new_dist = match self.linkage {
                    Linkage::Single => dist[[merge_i, k]].min(dist[[merge_j, k]]),
                    Linkage::Complete => dist[[merge_i, k]].max(dist[[merge_j, k]]),
                    Linkage::Average => {
                        // (d_ik * n_old_i + d_jk * n_j) / (n_old_i + n_j)
                        // sizes[merge_i] already includes j, so old_i = total - nj
                        (dist[[merge_i, k]] * (sizes[merge_i] as f64 - nj)
                            + dist[[merge_j, k]] * nj)
                            / (sizes[merge_i] as f64)
                    }
                    Linkage::Ward => {
                        // Lance-Williams formula for Ward:
                        // d(i∪j, k) = sqrt(((n_i+n_k)*d(i,k)^2 + (n_j+n_k)*d(j,k)^2 - n_k*d(i,j)^2) / (n_i+n_j+n_k))
                        let nk = sizes[k] as f64;
                        let di = dist[[merge_i, k]]; // squared
                        let dj = dist[[merge_j, k]]; // squared
                        let dij = min_dist; // squared (the distance we just merged on)
                        let total = ni + nj + nk;
                        (((ni + nk) * di * di + (nj + nk) * dj * dj - nk * dij * dij) / total)
                            .max(0.0)
                            .sqrt()
                    }
                };

                dist[[merge_i, k]] = new_dist;
                dist[[k, merge_i]] = new_dist;
            }
        }

        // Assign final labels
        let mut labels = Array1::from_elem(n_samples, -1i32);
        let mut label_idx = 0i32;
        for i in 0..cluster_id.len() {
            if !active_clusters[i] {
                continue;
            }
            for &member in &cluster_members[i] {
                labels[member] = label_idx;
            }
            label_idx += 1;
        }

        self.labels = Some(labels);
        self.n_features = Some(n_features);
        self.children = Some(children);
        Ok(())
    }

    fn predict(&self, _x: &Array2<f64>) -> Result<Array1<i32>> {
        // Agglomerative clustering doesn't support predicting on new data
        // (unlike KMeans). Return labels from fit.
        self.labels
            .clone()
            .ok_or_else(|| FerroError::not_fitted("predict"))
    }

    fn labels(&self) -> Option<&Array1<i32>> {
        self.labels.as_ref()
    }

    fn is_fitted(&self) -> bool {
        self.labels.is_some()
    }
}

/// Squared Euclidean distance between two slices.
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_three_clusters() -> Array2<f64> {
        Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // cluster A
                10.0, 10.0, 11.0, 10.0, 10.0, 11.0, // cluster B
                20.0, 0.0, 21.0, 0.0, 20.0, 1.0, // cluster C
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_agglomerative_ward() {
        let x = make_three_clusters();
        let mut model = AgglomerativeClustering::new(3);
        let labels = model.fit_predict(&x).unwrap();

        assert_eq!(labels.len(), 9);
        // Points in the same cluster should have the same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[6], labels[7]);
        assert_eq!(labels[7], labels[8]);
        // Different clusters should have different labels
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[6]);
        assert_ne!(labels[3], labels[6]);
    }

    #[test]
    fn test_agglomerative_single() {
        let x = make_three_clusters();
        let mut model = AgglomerativeClustering::new(3).with_linkage(Linkage::Single);
        let labels = model.fit_predict(&x).unwrap();

        assert_eq!(labels.len(), 9);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[6], labels[7]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_agglomerative_complete() {
        let x = make_three_clusters();
        let mut model = AgglomerativeClustering::new(3).with_linkage(Linkage::Complete);
        let labels = model.fit_predict(&x).unwrap();

        assert_eq!(labels.len(), 9);
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_agglomerative_average() {
        let x = make_three_clusters();
        let mut model = AgglomerativeClustering::new(3).with_linkage(Linkage::Average);
        let labels = model.fit_predict(&x).unwrap();

        assert_eq!(labels.len(), 9);
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_agglomerative_two_clusters() {
        let x = make_three_clusters();
        let mut model = AgglomerativeClustering::new(2).with_linkage(Linkage::Ward);
        let labels = model.fit_predict(&x).unwrap();

        // Should merge 2 of the 3 clusters
        let unique: std::collections::HashSet<i32> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_agglomerative_one_cluster() {
        let x = make_three_clusters();
        let mut model = AgglomerativeClustering::new(1);
        let labels = model.fit_predict(&x).unwrap();

        // All should be in cluster 0
        for &l in labels.iter() {
            assert_eq!(l, 0);
        }
    }

    #[test]
    fn test_agglomerative_children() {
        let x = make_three_clusters();
        let mut model = AgglomerativeClustering::new(3);
        model.fit(&x).unwrap();

        let children = model.children().unwrap();
        // 9 samples, 3 clusters => 6 merges
        assert_eq!(children.len(), 6);
    }

    #[test]
    fn test_agglomerative_invalid_n_clusters() {
        let x = make_three_clusters();
        let mut model = AgglomerativeClustering::new(0);
        assert!(model.fit(&x).is_err());

        let mut model2 = AgglomerativeClustering::new(100);
        assert!(model2.fit(&x).is_err());
    }

    #[test]
    fn test_agglomerative_not_fitted() {
        let model = AgglomerativeClustering::new(3);
        assert!(!model.is_fitted());
        assert!(model.labels().is_none());
    }
}
