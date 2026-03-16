//! DBSCAN: Density-Based Spatial Clustering
//!
//! This module implements DBSCAN (Density-Based Spatial Clustering of Applications
//! with Noise) with FerroML-style statistical extensions.
//!
//! ## Features
//!
//! - **Noise detection**: Points that don't belong to any cluster are labeled -1
//! - **No predefined k**: Number of clusters determined automatically
//! - **Cluster persistence**: Stability across eps values
//! - **Optimal eps**: k-distance graph analysis
//!
//! ## Example
//!
//! ```
//! use ferroml_core::clustering::{DBSCAN, ClusteringModel};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0
//! ]).unwrap();
//!
//! let mut dbscan = DBSCAN::new(0.5, 2);
//! dbscan.fit(&x).unwrap();
//!
//! let labels = dbscan.labels().unwrap();
//! println!("Cluster labels: {:?}", labels);
//! println!("Core samples: {:?}", dbscan.core_sample_indices());
//! ```

use crate::clustering::ClusteringModel;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};

/// DBSCAN clustering algorithm
///
/// DBSCAN groups together points that are closely packed (density-based),
/// marking outliers as noise that lie alone in low-density regions.
///
/// # Parameters
///
/// - `eps` - Maximum distance between two samples to be considered neighbors
/// - `min_samples` - Minimum number of samples in a neighborhood to form a core point
///
/// # Statistical Extensions
///
/// Beyond sklearn's basic implementation, FerroML's DBSCAN provides:
/// - `cluster_persistence()` - Stability across eps values
/// - `optimal_eps()` - k-distance graph analysis for eps selection
/// - `noise_analysis()` - Statistical profile of noise points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBSCAN {
    /// Maximum distance between two samples for neighborhood
    eps: f64,
    /// Minimum samples in neighborhood to form core point
    min_samples: usize,

    /// Optional GPU backend for accelerated pairwise distance computation
    #[cfg(feature = "gpu")]
    #[serde(skip)]
    gpu_backend: Option<std::sync::Arc<dyn crate::gpu::GpuBackend>>,

    // Fitted state
    /// Labels for each point (-1 for noise)
    labels_: Option<Array1<i32>>,
    /// Indices of core samples
    core_sample_indices_: Option<Vec<usize>>,
    /// Core sample coordinates
    components_: Option<Array2<f64>>,
}

impl Default for DBSCAN {
    fn default() -> Self {
        Self::new(0.5, 5)
    }
}

impl DBSCAN {
    /// Create a new DBSCAN model
    ///
    /// # Arguments
    /// * `eps` - Maximum distance between two samples for neighborhood
    /// * `min_samples` - Minimum number of samples to form a core point
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
            labels_: None,
            core_sample_indices_: None,
            components_: None,
        }
    }

    /// Set GPU backend for accelerated pairwise distance computation.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, backend: std::sync::Arc<dyn crate::gpu::GpuBackend>) -> Self {
        self.gpu_backend = Some(backend);
        self
    }

    /// Set eps (maximum neighborhood distance)
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set min_samples
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Get indices of core samples
    pub fn core_sample_indices(&self) -> Option<&Vec<usize>> {
        self.core_sample_indices_.as_ref()
    }

    /// Get core sample coordinates
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components_.as_ref()
    }

    /// Get number of clusters found (excluding noise)
    pub fn n_clusters(&self) -> Option<usize> {
        self.labels_.as_ref().map(|labels| {
            labels
                .iter()
                .filter(|&&l| l >= 0)
                .max()
                .map_or(0, |&m| m as usize + 1)
        })
    }

    /// Get number of noise points
    pub fn n_noise(&self) -> Option<usize> {
        self.labels_
            .as_ref()
            .map(|labels| labels.iter().filter(|&&l| l == -1).count())
    }

    /// Find neighbors within eps distance.
    ///
    /// Uses squared distances internally to avoid sqrt overhead.
    fn region_query(&self, x: &Array2<f64>, point_idx: usize) -> Vec<usize> {
        let point = x.row(point_idx);
        let eps_sq = self.eps * self.eps;
        let mut neighbors = Vec::new();

        for i in 0..x.nrows() {
            let dist_sq = squared_euclidean_distance(&point, &x.row(i));
            if dist_sq <= eps_sq {
                neighbors.push(i);
            }
        }

        neighbors
    }

    /// Compute optimal eps using k-distance graph
    ///
    /// Analyzes the k-distance graph where k = min_samples to find
    /// an optimal eps value (the "elbow" in the sorted distances).
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    ///
    /// # Returns
    /// (suggested_eps, k_distances sorted)
    pub fn optimal_eps(x: &Array2<f64>, min_samples: usize) -> Result<(f64, Vec<f64>)> {
        let n_samples = x.nrows();

        if n_samples < min_samples {
            return Err(FerroError::InvalidInput(format!(
                "n_samples={} must be >= min_samples={}",
                n_samples, min_samples
            )));
        }

        // Compute k-th nearest neighbor distance for each point
        let k = min_samples;
        let mut k_distances = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut distances: Vec<f64> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| euclidean_distance(&x.row(i), &x.row(j)))
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // k-th nearest neighbor (0-indexed, so k-1)
            k_distances.push(distances[k.min(distances.len()) - 1]);
        }

        // Sort k-distances in descending order
        k_distances.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Find elbow using simple method: maximum curvature
        let n = k_distances.len();
        if n < 3 {
            return Ok((k_distances[0], k_distances));
        }

        // Normalize to [0,1] range
        let x_range = (n - 1) as f64;
        let y_min = k_distances[n - 1];
        let y_max = k_distances[0];
        let y_range = if y_max > y_min { y_max - y_min } else { 1.0 };

        // Find point farthest from line connecting first and last point
        let mut max_dist = 0.0;
        let mut elbow_idx = 0;

        for i in 1..n - 1 {
            let x_norm = i as f64 / x_range;
            let y_norm = (k_distances[i] - y_min) / y_range;

            // Distance from point (x_norm, y_norm) to line from (0, 1) to (1, 0)
            // Line: x + y = 1, or x + y - 1 = 0
            // Distance = |x + y - 1| / sqrt(2)
            let dist = (x_norm + y_norm - 1.0).abs();

            if dist > max_dist {
                max_dist = dist;
                elbow_idx = i;
            }
        }

        Ok((k_distances[elbow_idx], k_distances))
    }

    /// Analyze stability of clusters across eps values
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `eps_values` - Range of eps values to test
    ///
    /// # Returns
    /// Vector of (eps, n_clusters, n_noise) tuples
    pub fn cluster_persistence(
        x: &Array2<f64>,
        eps_values: &[f64],
        min_samples: usize,
    ) -> Result<Vec<(f64, usize, usize)>> {
        let mut results = Vec::with_capacity(eps_values.len());

        for &eps in eps_values {
            let mut dbscan = DBSCAN::new(eps, min_samples);
            dbscan.fit(x)?;

            let n_clusters = dbscan.n_clusters().unwrap_or(0);
            let n_noise = dbscan.n_noise().unwrap_or(0);
            results.push((eps, n_clusters, n_noise));
        }

        Ok(results)
    }

    /// Analyze noise points statistically
    ///
    /// # Returns
    /// (noise_ratio, noise_centroid, noise_std)
    pub fn noise_analysis(&self, x: &Array2<f64>) -> Result<(f64, Array1<f64>, Array1<f64>)> {
        let labels = self
            .labels_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("noise_analysis"))?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Get noise points
        let noise_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &l)| if l == -1 { Some(i) } else { None })
            .collect();

        let noise_ratio = noise_indices.len() as f64 / n_samples as f64;

        if noise_indices.is_empty() {
            return Ok((0.0, Array1::zeros(n_features), Array1::zeros(n_features)));
        }

        // Compute centroid of noise points
        let mut centroid = Array1::zeros(n_features);
        for &i in &noise_indices {
            for j in 0..n_features {
                centroid[j] += x[[i, j]];
            }
        }
        centroid.mapv_inplace(|v| v / noise_indices.len() as f64);

        // Compute standard deviation
        let mut variance = Array1::zeros(n_features);
        for &i in &noise_indices {
            for j in 0..n_features {
                let diff: f64 = x[[i, j]] - centroid[j];
                variance[j] += diff * diff;
            }
        }
        let n_noise_f64 = noise_indices.len() as f64;
        let std = variance.mapv(|v: f64| (v / n_noise_f64).sqrt());

        Ok((noise_ratio, centroid, std))
    }

    /// Fit DBSCAN on sparse CSR matrix data.
    ///
    /// This uses native sparse distance computations, which are efficient
    /// for high-dimensional sparse data (e.g., text/NLP features).
    ///
    /// # Arguments
    /// * `x` - Sparse CSR feature matrix
    ///
    /// # Returns
    /// Cluster labels for each point (-1 for noise)
    #[cfg(feature = "sparse")]
    pub fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix) -> Result<Array1<i32>> {
        let n_samples = x.nrows();

        if self.eps <= 0.0 {
            return Err(FerroError::InvalidInput("eps must be positive".to_string()));
        }

        if self.min_samples < 1 {
            return Err(FerroError::InvalidInput(
                "min_samples must be at least 1".to_string(),
            ));
        }

        // Precompute all neighbors using sparse distances
        let eps_sq = self.eps * self.eps;
        let neighbors: Vec<Vec<usize>> = (0..n_samples)
            .map(|i| {
                let row_i = x.row(i);
                let dists = crate::sparse::sparse_pairwise_distances(
                    &row_i,
                    x,
                    crate::sparse::SparseDistanceMetric::SquaredEuclidean,
                );
                dists
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d <= eps_sq)
                    .map(|(j, _)| j)
                    .collect()
            })
            .collect();

        // Identify core points
        let is_core: Vec<bool> = neighbors
            .iter()
            .map(|n| n.len() >= self.min_samples)
            .collect();

        // Initialize labels as unvisited (-2)
        let mut labels = vec![-2i32; n_samples];
        let mut cluster_id = 0i32;
        let mut core_sample_indices = Vec::new();

        for i in 0..n_samples {
            if labels[i] != -2 {
                continue;
            }

            let point_neighbors = &neighbors[i];

            if !is_core[i] {
                labels[i] = -1;
                continue;
            }

            // Start new cluster
            labels[i] = cluster_id;
            core_sample_indices.push(i);

            // Expand cluster using BFS
            let mut seed_set: Vec<usize> = point_neighbors
                .iter()
                .filter(|&&j| j != i && labels[j] == -2)
                .cloned()
                .collect();

            let mut idx = 0;
            while idx < seed_set.len() {
                let q = seed_set[idx];
                idx += 1;

                if labels[q] == -1 {
                    labels[q] = cluster_id;
                } else if labels[q] != -2 {
                    continue;
                } else {
                    labels[q] = cluster_id;
                }

                if is_core[q] {
                    core_sample_indices.push(q);
                    for &neighbor in &neighbors[q] {
                        if labels[neighbor] == -2 || labels[neighbor] == -1 {
                            if labels[neighbor] == -2 {
                                seed_set.push(neighbor);
                            }
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        // Remove duplicates from core_sample_indices and sort
        core_sample_indices.sort_unstable();
        core_sample_indices.dedup();

        // Extract core sample components as dense (needed for predict)
        let n_core = core_sample_indices.len();
        let n_features = x.ncols();
        let mut components = Array2::zeros((n_core, n_features));
        for (i, &ci) in core_sample_indices.iter().enumerate() {
            let row = x.row(ci);
            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                components[[i, col]] = val;
            }
        }

        let result = Array1::from_vec(labels);
        self.labels_ = Some(result.clone());
        self.core_sample_indices_ = Some(core_sample_indices);
        self.components_ = Some(components);

        Ok(result)
    }
}

impl ClusteringModel for DBSCAN {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.is_empty() || x.nrows() == 0 {
            return Err(FerroError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FerroError::InvalidInput(
                "Input contains NaN or infinite values".to_string(),
            ));
        }

        let n_samples = x.nrows();

        if self.eps <= 0.0 {
            return Err(FerroError::InvalidInput("eps must be positive".to_string()));
        }

        if self.min_samples < 1 {
            return Err(FerroError::InvalidInput(
                "min_samples must be at least 1".to_string(),
            ));
        }

        // Initialize labels as unvisited (-2)
        let mut labels = vec![-2i32; n_samples];
        let mut cluster_id = 0i32;
        let mut core_sample_indices = Vec::new();

        // Find all neighbors for each point (precompute for efficiency)
        // GPU-accelerated path: compute full pairwise distance matrix on GPU
        #[cfg(feature = "gpu")]
        let gpu_dist_matrix = self
            .gpu_backend
            .as_ref()
            .and_then(|gpu| gpu.pairwise_distances(x, x).ok());

        #[cfg(feature = "gpu")]
        let neighbors: Vec<Vec<usize>> = if let Some(ref dist_matrix) = gpu_dist_matrix {
            let eps_sq = self.eps * self.eps;
            (0..n_samples)
                .map(|i| {
                    (0..n_samples)
                        .filter(|&j| dist_matrix[[i, j]] <= eps_sq)
                        .collect()
                })
                .collect()
        } else {
            (0..n_samples).map(|i| self.region_query(x, i)).collect()
        };

        #[cfg(not(feature = "gpu"))]
        let neighbors: Vec<Vec<usize>> = (0..n_samples).map(|i| self.region_query(x, i)).collect();

        // Identify core points
        let is_core: Vec<bool> = neighbors
            .iter()
            .map(|n| n.len() >= self.min_samples)
            .collect();

        for i in 0..n_samples {
            // Skip if already processed
            if labels[i] != -2 {
                continue;
            }

            // Get neighbors
            let point_neighbors = &neighbors[i];

            // Check if core point
            if !is_core[i] {
                // Mark as noise (might be changed to border point later)
                labels[i] = -1;
                continue;
            }

            // Start new cluster
            labels[i] = cluster_id;
            core_sample_indices.push(i);

            // Expand cluster using BFS
            let mut seed_set: Vec<usize> = point_neighbors
                .iter()
                .filter(|&&j| j != i && labels[j] == -2)
                .cloned()
                .collect();

            let mut idx = 0;
            while idx < seed_set.len() {
                let q = seed_set[idx];
                idx += 1;

                if labels[q] == -1 {
                    // Change noise to border point
                    labels[q] = cluster_id;
                } else if labels[q] != -2 {
                    // Already processed
                    continue;
                } else {
                    labels[q] = cluster_id;
                }

                // If q is a core point, add its neighbors to seed set
                if is_core[q] {
                    core_sample_indices.push(q);
                    for &neighbor in &neighbors[q] {
                        if labels[neighbor] == -2 || labels[neighbor] == -1 {
                            if labels[neighbor] == -2 {
                                seed_set.push(neighbor);
                            }
                            // Border points will be assigned in the next iteration
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        // Remove duplicates from core_sample_indices and sort
        core_sample_indices.sort_unstable();
        core_sample_indices.dedup();

        // Extract core sample components
        let n_core = core_sample_indices.len();
        let n_features = x.ncols();
        let mut components = Array2::zeros((n_core, n_features));
        for (i, &idx) in core_sample_indices.iter().enumerate() {
            components.row_mut(i).assign(&x.row(idx));
        }

        self.labels_ = Some(Array1::from_vec(labels));
        self.core_sample_indices_ = Some(core_sample_indices);
        self.components_ = Some(components);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        // DBSCAN doesn't truly support predict for new data
        // We assign each point to the nearest core sample's cluster, or -1 if too far
        let components = self
            .components_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        let core_indices = self.core_sample_indices_.as_ref().unwrap();
        let train_labels = self.labels_.as_ref().unwrap();

        let n_samples = x.nrows();
        let mut labels = Array1::from_elem(n_samples, -1i32);

        let eps_sq = self.eps * self.eps;
        for i in 0..n_samples {
            let mut min_dist_sq = f64::MAX;
            let mut nearest_core = None;

            for (j, &core_idx) in core_indices.iter().enumerate() {
                let dist_sq = squared_euclidean_distance(&x.row(i), &components.row(j));
                if dist_sq <= eps_sq && dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    nearest_core = Some(core_idx);
                }
            }

            if let Some(core_idx) = nearest_core {
                labels[i] = train_labels[core_idx];
            }
        }

        Ok(labels)
    }

    fn labels(&self) -> Option<&Array1<i32>> {
        self.labels_.as_ref()
    }

    fn is_fitted(&self) -> bool {
        self.labels_.is_some()
    }
}

/// Compute Euclidean distance between two vectors.
///
/// Uses SIMD-accelerated computation when the `simd` feature is enabled (default).
#[inline]
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    squared_euclidean_distance(a, b).sqrt()
}

/// Compute squared Euclidean distance between two vectors (avoids sqrt).
///
/// Uses SIMD-accelerated computation when the `simd` feature is enabled (default).
#[inline]
fn squared_euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        crate::linalg::squared_euclidean_distance(a_slice, b_slice)
    } else {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_basic() {
        // Two clear clusters
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 5.1, 5.1,
            ],
        )
        .unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&x).unwrap();

        assert!(dbscan.is_fitted());
        let labels = dbscan.labels().unwrap();
        assert_eq!(labels.len(), 8);

        // Should find 2 clusters
        assert_eq!(dbscan.n_clusters(), Some(2));

        // First 4 points in one cluster, last 4 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[6], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_dbscan_with_noise() {
        // Two clusters with one outlier
        let x = Array2::from_shape_vec(
            (7, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 100.0,
                100.0, // outlier
            ],
        )
        .unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&x).unwrap();

        let labels = dbscan.labels().unwrap();

        // Last point should be noise
        assert_eq!(labels[6], -1);
        assert_eq!(dbscan.n_noise(), Some(1));
    }

    #[test]
    fn test_dbscan_fit_predict() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        let labels = dbscan.fit_predict(&x).unwrap();

        assert_eq!(labels.len(), 6);
        assert!(dbscan.is_fitted());
    }

    #[test]
    fn test_optimal_eps() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 5.1, 5.1,
            ],
        )
        .unwrap();

        let (suggested_eps, k_distances) = DBSCAN::optimal_eps(&x, 2).unwrap();

        // Should suggest a reasonable eps
        assert!(suggested_eps > 0.0);
        assert!(suggested_eps < 5.0);
        assert_eq!(k_distances.len(), 8);
    }

    #[test]
    fn test_cluster_persistence() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
        )
        .unwrap();

        let eps_values = vec![0.1, 0.2, 0.5, 1.0, 2.0];
        let results = DBSCAN::cluster_persistence(&x, &eps_values, 2).unwrap();

        assert_eq!(results.len(), 5);
        // With very small eps, should have more noise
        // With larger eps, clusters merge
    }

    #[test]
    fn test_noise_analysis() {
        let x = Array2::from_shape_vec(
            (7, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 50.0, 50.0,
            ],
        )
        .unwrap();

        let mut dbscan = DBSCAN::new(0.5, 2);
        dbscan.fit(&x).unwrap();

        let (noise_ratio, centroid, std) = dbscan.noise_analysis(&x).unwrap();

        assert!(noise_ratio > 0.0);
        assert_eq!(centroid.len(), 2);
        assert_eq!(std.len(), 2);
    }

    #[test]
    fn test_squared_distance_equivalence() {
        // Verify squared_euclidean_distance(a,b) == euclidean_distance(a,b).powi(2)
        let pairs: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![0.0, 0.0], vec![3.0, 4.0]),
            (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
            (vec![0.0], vec![0.0]),
            (vec![-1.5, 2.7, -0.3], vec![3.1, -4.2, 1.8]),
            (vec![1e6, -1e6], vec![-1e6, 1e6]),
        ];

        for (a_vec, b_vec) in &pairs {
            let a = Array1::from_vec(a_vec.clone());
            let b = Array1::from_vec(b_vec.clone());
            let a_view = a.view();
            let b_view = b.view();

            let sq_dist = squared_euclidean_distance(&a_view, &b_view);
            let euc_dist = euclidean_distance(&a_view, &b_view);
            let euc_sq = euc_dist.powi(2);

            // Use relative tolerance for large values
            let tol = 1e-10 * sq_dist.abs().max(1.0);
            assert!(
                (sq_dist - euc_sq).abs() < tol,
                "Mismatch for {:?} vs {:?}: squared={}, euclidean^2={}, diff={}",
                a_vec,
                b_vec,
                sq_dist,
                euc_sq,
                (sq_dist - euc_sq).abs()
            );
        }
    }

    #[test]
    fn test_dbscan_with_squared_distances_matches_expected() {
        // Small dataset with known cluster structure.
        // Three tight clusters at (0,0), (10,10), (20,20) with points within eps=1.5.
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.5, 0.5, 0.0, 0.5, // cluster A
                10.0, 10.0, 10.5, 10.5, 10.0, 10.5, // cluster B
                20.0, 20.0, 20.5, 20.5, 20.0, 20.5, // cluster C
            ],
        )
        .unwrap();

        let mut dbscan = DBSCAN::new(1.5, 2);
        dbscan.fit(&x).unwrap();

        let labels = dbscan.labels().unwrap();
        assert_eq!(labels.len(), 9);

        // Should find exactly 3 clusters, no noise
        assert_eq!(dbscan.n_clusters(), Some(3));
        assert_eq!(dbscan.n_noise(), Some(0));

        // Points within same cluster should share labels
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
}
