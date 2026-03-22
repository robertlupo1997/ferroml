//! Cluster Diagnostics
//!
//! This module provides diagnostic tools for analyzing clustering results.
//!
//! ## Features
//!
//! - Within-cluster variance analysis
//! - Cluster separation tests
//! - Outlier detection within clusters
//! - Cluster quality summary

use crate::clustering::silhouette_samples;
use crate::{FerroError, Result};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};

/// Comprehensive diagnostics for clustering results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterDiagnostics {
    /// Number of clusters
    pub n_clusters: usize,
    /// Number of samples per cluster
    pub cluster_sizes: Vec<usize>,
    /// Within-cluster sum of squares for each cluster
    pub within_ss: Vec<f64>,
    /// Total within-cluster sum of squares
    pub total_within_ss: f64,
    /// Between-cluster sum of squares
    pub between_ss: f64,
    /// Cluster centroids
    pub centroids: Array2<f64>,
    /// Mean silhouette score per cluster
    pub silhouette_per_cluster: Vec<f64>,
    /// Overall silhouette score
    pub silhouette_overall: f64,
    /// Potential outliers (indices) per cluster
    pub outliers_per_cluster: Vec<Vec<usize>>,
    /// Cluster compactness (mean distance to centroid)
    pub compactness: Vec<f64>,
    /// Cluster separation (min distance to other centroids)
    pub separation: Vec<f64>,
}

impl ClusterDiagnostics {
    /// Compute diagnostics for a fitted clustering model
    ///
    /// # Arguments
    /// * `x` - Feature matrix used for clustering
    /// * `labels` - Cluster labels from the model
    /// * `outlier_threshold` - Silhouette threshold below which points are outliers (default: 0.0)
    pub fn from_labels(
        x: &Array2<f64>,
        labels: &Array1<i32>,
        outlier_threshold: Option<f64>,
    ) -> Result<Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != labels.len() {
            return Err(FerroError::InvalidInput(
                "x and labels must have same number of samples".to_string(),
            ));
        }

        let threshold = outlier_threshold.unwrap_or(0.0);

        // Get unique labels (excluding noise)
        let mut unique_labels: Vec<i32> = labels.iter().cloned().filter(|&l| l >= 0).collect();
        unique_labels.sort_unstable();
        unique_labels.dedup();
        let n_clusters = unique_labels.len();

        if n_clusters == 0 {
            return Err(FerroError::InvalidInput(
                "No valid clusters found".to_string(),
            ));
        }

        // Compute cluster sizes and centroids
        let mut cluster_sizes = vec![0usize; n_clusters];
        let mut centroids = Array2::zeros((n_clusters, n_features));

        for (k, &label) in unique_labels.iter().enumerate() {
            for i in 0..n_samples {
                if labels[i] == label {
                    for j in 0..n_features {
                        centroids[[k, j]] += x[[i, j]];
                    }
                    cluster_sizes[k] += 1;
                }
            }
            if cluster_sizes[k] > 0 {
                for j in 0..n_features {
                    centroids[[k, j]] /= cluster_sizes[k] as f64;
                }
            }
        }

        // Compute within-cluster sum of squares
        let mut within_ss = vec![0.0; n_clusters];
        for (k, &label) in unique_labels.iter().enumerate() {
            for i in 0..n_samples {
                if labels[i] == label {
                    for j in 0..n_features {
                        let diff: f64 = x[[i, j]] - centroids[[k, j]];
                        within_ss[k] += diff * diff;
                    }
                }
            }
        }
        let total_within_ss: f64 = within_ss.iter().sum();

        // Compute overall centroid and between-cluster SS
        let overall_centroid = x.mean_axis(Axis(0)).expect("SAFETY: non-empty axis");
        let mut between_ss = 0.0;
        for (k, _) in unique_labels.iter().enumerate() {
            let n_k = cluster_sizes[k] as f64;
            for j in 0..n_features {
                let diff: f64 = centroids[[k, j]] - overall_centroid[j];
                between_ss += n_k * diff * diff;
            }
        }

        // Compute silhouette scores
        let silhouette_samples = silhouette_samples(x, labels)?;
        let silhouette_overall = silhouette_samples.mean().unwrap_or(0.0);

        // Compute per-cluster silhouette and identify outliers
        let mut silhouette_per_cluster = vec![0.0; n_clusters];
        let mut outliers_per_cluster = vec![Vec::new(); n_clusters];

        for (k, &label) in unique_labels.iter().enumerate() {
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..n_samples {
                if labels[i] == label {
                    sum += silhouette_samples[i];
                    count += 1;
                    if silhouette_samples[i] < threshold {
                        outliers_per_cluster[k].push(i);
                    }
                }
            }
            silhouette_per_cluster[k] = if count > 0 { sum / count as f64 } else { 0.0 };
        }

        // Compute compactness (mean distance to centroid)
        let mut compactness = vec![0.0; n_clusters];
        for (k, &label) in unique_labels.iter().enumerate() {
            let mut sum = 0.0;
            for i in 0..n_samples {
                if labels[i] == label {
                    sum += euclidean_distance(&x.row(i), &centroids.row(k));
                }
            }
            compactness[k] = if cluster_sizes[k] > 0 {
                sum / cluster_sizes[k] as f64
            } else {
                0.0
            };
        }

        // Compute separation (min distance to other centroids)
        let mut separation = vec![f64::MAX; n_clusters];
        for i in 0..n_clusters {
            for j in 0..n_clusters {
                if i != j {
                    let dist = euclidean_distance(&centroids.row(i), &centroids.row(j));
                    if dist < separation[i] {
                        separation[i] = dist;
                    }
                }
            }
            // Handle single cluster case
            if separation[i] == f64::MAX {
                separation[i] = 0.0;
            }
        }

        Ok(Self {
            n_clusters,
            cluster_sizes,
            within_ss,
            total_within_ss,
            between_ss,
            centroids,
            silhouette_per_cluster,
            silhouette_overall,
            outliers_per_cluster,
            compactness,
            separation,
        })
    }

    /// Get the ratio of between-cluster to total variance
    ///
    /// Higher values indicate better cluster separation
    pub fn variance_ratio(&self) -> f64 {
        let total_ss = self.total_within_ss + self.between_ss;
        if total_ss > 0.0 {
            self.between_ss / total_ss
        } else {
            0.0
        }
    }

    /// Get the Dunn index (ratio of min inter-cluster to max intra-cluster distance)
    ///
    /// Higher values indicate better clustering
    pub fn dunn_index(&self) -> f64 {
        let min_separation = self
            .separation
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_compactness = self
            .compactness
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if max_compactness > 0.0 {
            min_separation / (2.0 * max_compactness)
        } else {
            0.0
        }
    }

    /// Generate a text summary of the diagnostics
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Clustering Diagnostics ({} clusters)\n",
            self.n_clusters
        ));
        s.push_str(&format!("{}\n", "=".repeat(40)));

        s.push_str("\nCluster Sizes:\n");
        for (k, &size) in self.cluster_sizes.iter().enumerate() {
            s.push_str(&format!("  Cluster {}: {} samples\n", k, size));
        }

        s.push_str(&format!(
            "\nVariance Decomposition:\n  Within-cluster SS:  {:.4}\n  Between-cluster SS: {:.4}\n  Variance ratio:     {:.4}\n",
            self.total_within_ss, self.between_ss, self.variance_ratio()
        ));

        s.push_str(&format!(
            "\nSilhouette Scores:\n  Overall: {:.4}\n",
            self.silhouette_overall
        ));
        for (k, &score) in self.silhouette_per_cluster.iter().enumerate() {
            s.push_str(&format!("  Cluster {}: {:.4}\n", k, score));
        }

        s.push_str(&format!("\nDunn Index: {:.4}\n", self.dunn_index()));

        let total_outliers: usize = self.outliers_per_cluster.iter().map(|v| v.len()).sum();
        if total_outliers > 0 {
            s.push_str(&format!("\nPotential Outliers: {} total\n", total_outliers));
            for (k, outliers) in self.outliers_per_cluster.iter().enumerate() {
                if !outliers.is_empty() {
                    s.push_str(&format!("  Cluster {}: {} outliers\n", k, outliers.len()));
                }
            }
        }

        s
    }
}

/// Compute Euclidean distance
#[inline]
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cluster_diagnostics() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let labels = array![0, 0, 0, 1, 1, 1];

        let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();

        assert_eq!(diag.n_clusters, 2);
        assert_eq!(diag.cluster_sizes, vec![3, 3]);
        assert!(diag.silhouette_overall > 0.9);
        assert!(diag.variance_ratio() > 0.9);
    }

    #[test]
    fn test_dunn_index() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let labels = array![0, 0, 0, 1, 1, 1];

        let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();
        let dunn = diag.dunn_index();

        // Well-separated clusters should have high Dunn index
        assert!(dunn > 10.0);
    }

    #[test]
    fn test_summary() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.0, 10.0, 10.0, 10.1, 10.0])
            .unwrap();
        let labels = array![0, 0, 1, 1];

        let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();
        let summary = diag.summary();

        assert!(summary.contains("2 clusters"));
        assert!(summary.contains("Silhouette"));
    }
}
