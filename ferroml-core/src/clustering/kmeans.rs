//! K-Means Clustering with Statistical Extensions
//!
//! This module implements K-Means clustering with kmeans++ initialization
//! and FerroML-style statistical extensions.
//!
//! ## Features
//!
//! - **kmeans++ initialization**: Smart centroid initialization for faster convergence
//! - **Cluster stability**: Bootstrap-based stability assessment
//! - **Gap statistic**: Optimal k selection with standard error
//! - **Silhouette with CI**: Confidence intervals on silhouette scores
//!
//! ## Example
//!
//! ```
//! use ferroml_core::clustering::{KMeans, ClusteringModel};
//! use ndarray::Array2;
//!
//! let x = Array2::from_shape_vec((6, 2), vec![
//!     1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0
//! ]).unwrap();
//!
//! let mut kmeans = KMeans::new(2);
//! kmeans.fit(&x).unwrap();
//!
//! let labels = kmeans.labels().unwrap();
//! println!("Cluster centers: {:?}", kmeans.cluster_centers());
//! ```

use crate::clustering::{ClusteringModel, ClusteringStatistics, ElbowResult, GapStatisticResult};
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, ArrayView1};
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// K-Means clustering algorithm with kmeans++ initialization
///
/// K-Means partitions data into k clusters by minimizing within-cluster
/// sum of squares (inertia). This implementation uses the kmeans++
/// initialization strategy for better convergence.
///
/// # Statistical Extensions
///
/// Beyond sklearn's basic implementation, FerroML's KMeans provides:
/// - `cluster_stability()` - Bootstrap-based cluster stability scores
/// - `optimal_k()` - Gap statistic for optimal k selection
/// - `silhouette_with_ci()` - Silhouette scores with confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans {
    /// Number of clusters
    n_clusters: usize,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Random seed for initialization
    random_state: Option<u64>,
    /// Number of times to run with different centroid seeds
    n_init: usize,

    // Fitted state
    /// Cluster centers after fitting
    cluster_centers_: Option<Array2<f64>>,
    /// Labels of each point
    labels_: Option<Array1<i32>>,
    /// Sum of squared distances to closest cluster center (inertia)
    inertia_: Option<f64>,
    /// Number of iterations run
    n_iter_: Option<usize>,
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(8)
    }
}

impl KMeans {
    /// Create a new KMeans model
    ///
    /// # Arguments
    /// * `n_clusters` - Number of clusters to form
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            random_state: None,
            n_init: 10,
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
            n_iter_: None,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Set number of initialization runs
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Get cluster centers
    pub fn cluster_centers(&self) -> Option<&Array2<f64>> {
        self.cluster_centers_.as_ref()
    }

    /// Get inertia (within-cluster sum of squares)
    pub fn inertia(&self) -> Option<f64> {
        self.inertia_
    }

    /// Get number of iterations run
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter_
    }

    /// kmeans++ initialization
    ///
    /// Selects initial centroids to be far apart from each other
    fn kmeans_plus_plus_init(&self, x: &Array2<f64>, rng: &mut StdRng) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut centers = Array2::zeros((self.n_clusters, n_features));

        // First center: random point
        let first_idx = rng.random_range(0..n_samples);
        centers.row_mut(0).assign(&x.row(first_idx));

        // Remaining centers: weighted probability by squared distance
        for k in 1..self.n_clusters {
            // Compute squared distances to nearest center
            let mut min_sq_dists = Array1::from_elem(n_samples, f64::MAX);
            for i in 0..n_samples {
                for j in 0..k {
                    let dist_sq = squared_euclidean(&x.row(i), &centers.row(j));
                    if dist_sq < min_sq_dists[i] {
                        min_sq_dists[i] = dist_sq;
                    }
                }
            }

            // Choose next center with probability proportional to D^2
            let total: f64 = min_sq_dists.sum();
            if total <= 0.0 {
                // All points are at centers, pick random
                let idx = rng.random_range(0..n_samples);
                centers.row_mut(k).assign(&x.row(idx));
            } else {
                let threshold = rng.random::<f64>() * total;
                let mut cumsum = 0.0;
                for i in 0..n_samples {
                    cumsum += min_sq_dists[i];
                    if cumsum >= threshold {
                        centers.row_mut(k).assign(&x.row(i));
                        break;
                    }
                }
            }
        }

        centers
    }

    /// Run a single kmeans iteration
    fn run_kmeans(
        &self,
        x: &Array2<f64>,
        initial_centers: Array2<f64>,
    ) -> (Array2<f64>, Array1<i32>, f64, usize) {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut centers = initial_centers;
        let mut labels = Array1::zeros(n_samples);
        let mut prev_inertia = f64::MAX;

        for iter in 0..self.max_iter {
            // Assignment step: assign each point to nearest center
            let mut inertia = 0.0;
            for i in 0..n_samples {
                let mut min_dist = f64::MAX;
                let mut min_idx = 0;
                for k in 0..self.n_clusters {
                    let dist = squared_euclidean(&x.row(i), &centers.row(k));
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = k;
                    }
                }
                labels[i] = min_idx as i32;
                inertia += min_dist;
            }

            // Check convergence
            if (prev_inertia - inertia).abs() < self.tol {
                return (centers, labels, inertia, iter + 1);
            }
            prev_inertia = inertia;

            // Update step: move centers to mean of assigned points
            let mut new_centers = Array2::zeros((self.n_clusters, n_features));
            let mut counts = vec![0usize; self.n_clusters];

            for i in 0..n_samples {
                let k = labels[i] as usize;
                for j in 0..n_features {
                    new_centers[[k, j]] += x[[i, j]];
                }
                counts[k] += 1;
            }

            for k in 0..self.n_clusters {
                if counts[k] > 0 {
                    for j in 0..n_features {
                        new_centers[[k, j]] /= counts[k] as f64;
                    }
                } else {
                    // Empty cluster: keep old center
                    new_centers.row_mut(k).assign(&centers.row(k));
                }
            }

            centers = new_centers;
        }

        // Compute final inertia
        let mut inertia = 0.0;
        for i in 0..n_samples {
            let k = labels[i] as usize;
            inertia += squared_euclidean(&x.row(i), &centers.row(k));
        }

        (centers, labels, inertia, self.max_iter)
    }

    /// Find optimal k using the gap statistic
    ///
    /// The gap statistic compares the change in within-cluster dispersion
    /// with that expected under an appropriate null reference distribution.
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `k_range` - Range of k values to test
    /// * `n_refs` - Number of reference datasets to generate
    pub fn optimal_k(
        x: &Array2<f64>,
        k_range: std::ops::Range<usize>,
        n_refs: usize,
        random_state: Option<u64>,
    ) -> Result<GapStatisticResult> {
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Compute min/max for each feature (for uniform reference generation)
        let mins: Vec<f64> = (0..n_features)
            .map(|j| x.column(j).iter().cloned().fold(f64::INFINITY, f64::min))
            .collect();
        let maxs: Vec<f64> = (0..n_features)
            .map(|j| {
                x.column(j)
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect();

        let k_values: Vec<usize> = k_range.collect();
        let mut gap_values = Vec::with_capacity(k_values.len());
        let mut gap_se = Vec::with_capacity(k_values.len());

        for &k in &k_values {
            // Fit kmeans on actual data
            let mut kmeans = KMeans::new(k).random_state(rng.random());
            kmeans.fit(x)?;
            let w_k = kmeans.inertia().unwrap().ln();

            // Generate reference datasets and compute their dispersions
            let mut ref_dispersions = Vec::with_capacity(n_refs);
            for _ in 0..n_refs {
                // Generate uniform random data in bounding box
                let mut ref_data = Array2::zeros((n_samples, n_features));
                for i in 0..n_samples {
                    for j in 0..n_features {
                        ref_data[[i, j]] = rng.random::<f64>() * (maxs[j] - mins[j]) + mins[j];
                    }
                }

                let mut ref_kmeans = KMeans::new(k).n_init(1).random_state(rng.random());
                ref_kmeans.fit(&ref_data)?;
                ref_dispersions.push(ref_kmeans.inertia().unwrap().ln());
            }

            // Compute gap statistic: E[log(W_k^*)] - log(W_k)
            let mean_ref: f64 = ref_dispersions.iter().sum::<f64>() / n_refs as f64;
            let gap = mean_ref - w_k;

            // Compute standard error
            let variance: f64 = ref_dispersions
                .iter()
                .map(|&x| (x - mean_ref).powi(2))
                .sum::<f64>()
                / n_refs as f64;
            let se = (variance * (1.0 + 1.0 / n_refs as f64)).sqrt();

            gap_values.push(gap);
            gap_se.push(se);
        }

        // Find optimal k: smallest k such that gap(k) >= gap(k+1) - se(k+1)
        let mut optimal_k = k_values[0];
        for i in 0..k_values.len() - 1 {
            if gap_values[i] >= gap_values[i + 1] - gap_se[i + 1] {
                optimal_k = k_values[i];
                break;
            }
        }

        Ok(GapStatisticResult {
            k_values,
            gap_values,
            gap_se,
            optimal_k,
        })
    }

    /// Find optimal k using the elbow method
    ///
    /// The elbow method looks for a "kink" in the inertia curve.
    ///
    /// # Arguments
    /// * `x` - Feature matrix
    /// * `k_range` - Range of k values to test
    pub fn elbow(
        x: &Array2<f64>,
        k_range: std::ops::Range<usize>,
        random_state: Option<u64>,
    ) -> Result<ElbowResult> {
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let k_values: Vec<usize> = k_range.collect();
        let mut inertias = Vec::with_capacity(k_values.len());

        for &k in &k_values {
            let mut kmeans = KMeans::new(k).random_state(rng.random());
            kmeans.fit(x)?;
            inertias.push(kmeans.inertia().unwrap());
        }

        // Find elbow using the kneedle algorithm (simplified)
        // Normalize data and find point farthest from line connecting first and last
        let n = k_values.len();
        if n < 3 {
            return Ok(ElbowResult {
                k_values: k_values.clone(),
                inertias,
                optimal_k: k_values[0],
            });
        }

        let x1 = k_values[0] as f64;
        let y1 = inertias[0];
        let x2 = k_values[n - 1] as f64;
        let y2 = inertias[n - 1];

        // Distance from point to line
        let _line_len = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();
        let mut max_dist = 0.0;
        let mut optimal_idx = 0;

        for (i, (&k, &inertia)) in k_values.iter().zip(inertias.iter()).enumerate() {
            let px = k as f64;
            let py = inertia;
            // Normalize to similar scales
            let norm_px = (px - x1) / (x2 - x1);
            let norm_py = (py - y2) / (y1 - y2);
            // Distance to diagonal in normalized space
            let dist = (norm_px - (1.0 - norm_py)).abs();
            if dist > max_dist {
                max_dist = dist;
                optimal_idx = i;
            }
        }

        Ok(ElbowResult {
            k_values: k_values.clone(),
            inertias,
            optimal_k: k_values[optimal_idx],
        })
    }
}

impl ClusteringModel for KMeans {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_samples = x.nrows();

        if n_samples < self.n_clusters {
            return Err(FerroError::InvalidInput(format!(
                "n_samples={} should be >= n_clusters={}",
                n_samples, self.n_clusters
            )));
        }

        let base_seed = self.random_state.unwrap_or_else(rand::random);
        let mut best_inertia = f64::MAX;
        let mut best_centers = None;
        let mut best_labels = None;
        let mut best_n_iter = 0;

        // Run n_init times and keep best result
        for init in 0..self.n_init {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(init as u64));
            let initial_centers = self.kmeans_plus_plus_init(x, &mut rng);
            let (centers, labels, inertia, n_iter) = self.run_kmeans(x, initial_centers);

            if inertia < best_inertia {
                best_inertia = inertia;
                best_centers = Some(centers);
                best_labels = Some(labels);
                best_n_iter = n_iter;
            }
        }

        self.cluster_centers_ = best_centers;
        self.labels_ = best_labels;
        self.inertia_ = Some(best_inertia);
        self.n_iter_ = Some(best_n_iter);

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let centers = self
            .cluster_centers_
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("predict"))?;

        let n_samples = x.nrows();
        let mut labels = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut min_dist = f64::MAX;
            let mut min_idx = 0;
            for k in 0..self.n_clusters {
                let dist = squared_euclidean(&x.row(i), &centers.row(k));
                if dist < min_dist {
                    min_dist = dist;
                    min_idx = k;
                }
            }
            labels[i] = min_idx as i32;
        }

        Ok(labels)
    }

    fn labels(&self) -> Option<&Array1<i32>> {
        self.labels_.as_ref()
    }

    fn is_fitted(&self) -> bool {
        self.cluster_centers_.is_some()
    }
}

impl ClusteringStatistics for KMeans {
    fn cluster_stability(&self, x: &Array2<f64>, n_bootstrap: usize) -> Result<Array1<f64>> {
        if !self.is_fitted() {
            return Err(FerroError::not_fitted("cluster_stability"));
        }

        let n_samples = x.nrows();
        let original_labels = self.labels_.as_ref().unwrap();
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(42));

        // For each bootstrap iteration, compute adjusted rand index with original
        let mut stability_scores = vec![0.0; self.n_clusters];
        let mut stability_counts = vec![0usize; self.n_clusters];

        for _ in 0..n_bootstrap {
            // Bootstrap sample indices
            let indices: Vec<usize> = (0..n_samples)
                .map(|_| rng.random_range(0..n_samples))
                .collect();
            let mut boot_x = Array2::zeros((n_samples, x.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                boot_x.row_mut(i).assign(&x.row(idx));
            }

            // Fit new kmeans on bootstrap sample
            let mut boot_kmeans = KMeans::new(self.n_clusters)
                .max_iter(self.max_iter)
                .tol(self.tol)
                .n_init(1)
                .random_state(rng.random());
            boot_kmeans.fit(&boot_x)?;

            // Predict labels for original data
            let boot_labels = boot_kmeans.predict(x)?;

            // Compute per-cluster stability (Jaccard similarity)
            for k in 0..self.n_clusters {
                let orig_mask: Vec<bool> = original_labels.iter().map(|&l| l == k as i32).collect();

                // Find best matching cluster in bootstrap
                let mut best_jaccard = 0.0;
                for bk in 0..self.n_clusters {
                    let boot_mask: Vec<bool> =
                        boot_labels.iter().map(|&l| l == bk as i32).collect();

                    let intersection: usize = orig_mask
                        .iter()
                        .zip(boot_mask.iter())
                        .filter(|(&a, &b)| a && b)
                        .count();
                    let union: usize = orig_mask
                        .iter()
                        .zip(boot_mask.iter())
                        .filter(|(&a, &b)| a || b)
                        .count();

                    if union > 0 {
                        let jaccard = intersection as f64 / union as f64;
                        if jaccard > best_jaccard {
                            best_jaccard = jaccard;
                        }
                    }
                }

                stability_scores[k] += best_jaccard;
                stability_counts[k] += 1;
            }
        }

        // Average stability scores
        let result: Vec<f64> = stability_scores
            .iter()
            .zip(stability_counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
            .collect();

        Ok(Array1::from_vec(result))
    }

    fn silhouette_with_ci(&self, x: &Array2<f64>, confidence: f64) -> Result<(f64, f64, f64)> {
        use crate::clustering::metrics::silhouette_score;

        if !self.is_fitted() {
            return Err(FerroError::not_fitted("silhouette_with_ci"));
        }

        let labels = self.labels_.as_ref().unwrap();
        let n_bootstrap = 1000;
        let mut rng = StdRng::seed_from_u64(self.random_state.unwrap_or(42));

        let n_samples = x.nrows();
        let mut scores = Vec::with_capacity(n_bootstrap);

        for _ in 0..n_bootstrap {
            // Bootstrap sample
            let indices: Vec<usize> = (0..n_samples)
                .map(|_| rng.random_range(0..n_samples))
                .collect();
            let mut boot_x = Array2::zeros((n_samples, x.ncols()));
            let mut boot_labels = Array1::zeros(n_samples);
            for (i, &idx) in indices.iter().enumerate() {
                boot_x.row_mut(i).assign(&x.row(idx));
                boot_labels[i] = labels[idx];
            }

            if let Ok(score) = silhouette_score(&boot_x, &boot_labels) {
                scores.push(score);
            }
        }

        if scores.is_empty() {
            return Err(FerroError::numerical("Could not compute silhouette scores"));
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let alpha = 1.0 - confidence;
        let lower_idx = ((alpha / 2.0) * scores.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * scores.len() as f64) as usize;

        let lower = scores[lower_idx.min(scores.len() - 1)];
        let upper = scores[upper_idx.min(scores.len() - 1)];

        Ok((mean, lower, upper))
    }
}

/// Compute squared Euclidean distance between two vectors
#[inline]
fn squared_euclidean(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.0, 1.0, 8.0, 8.0, 8.5, 8.0, 8.0, 7.5],
        )
        .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        kmeans.fit(&x).unwrap();

        assert!(kmeans.is_fitted());
        assert!(kmeans.cluster_centers().is_some());
        assert!(kmeans.inertia().unwrap() < 10.0);

        let labels = kmeans.labels().unwrap();
        assert_eq!(labels.len(), 6);

        // First 3 points should be in same cluster, last 3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_kmeans_predict() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.5, 1.5, 10.0, 10.0, 10.5, 10.5])
            .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        kmeans.fit(&x).unwrap();

        let new_x = Array2::from_shape_vec((2, 2), vec![1.2, 1.2, 10.2, 10.2]).unwrap();
        let new_labels = kmeans.predict(&new_x).unwrap();

        assert_eq!(new_labels.len(), 2);
        // Points should be assigned to different clusters
        assert_ne!(new_labels[0], new_labels[1]);
    }

    #[test]
    fn test_kmeans_fit_predict() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.5, 1.5, 10.0, 10.0, 10.5, 10.5])
            .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        let labels = kmeans.fit_predict(&x).unwrap();

        assert_eq!(labels.len(), 4);
        assert!(kmeans.is_fitted());
    }

    #[test]
    fn test_kmeans_elbow() {
        let x = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 1.0, 1.5, 5.0, 5.0, 5.5, 5.5, 5.0, 5.5, 9.0, 9.0, 9.5, 9.5,
                9.0, 9.5,
            ],
        )
        .unwrap();

        let result = KMeans::elbow(&x, 1..6, Some(42)).unwrap();
        assert_eq!(result.k_values.len(), 5);
        assert_eq!(result.inertias.len(), 5);
        // With 3 clear clusters, optimal k should be around 3
        assert!(result.optimal_k >= 2 && result.optimal_k <= 4);
    }

    #[test]
    fn test_cluster_stability() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 1.0, 1.1, 1.1, 1.0, 10.0, 10.0, 10.1, 10.1, 10.0, 10.1, 10.1,
                10.0,
            ],
        )
        .unwrap();

        let mut kmeans = KMeans::new(2).random_state(42);
        kmeans.fit(&x).unwrap();

        let stability = kmeans.cluster_stability(&x, 50).unwrap();
        assert_eq!(stability.len(), 2);
        // Well-separated clusters should have high stability
        for s in stability.iter() {
            assert!(*s > 0.7);
        }
    }
}
