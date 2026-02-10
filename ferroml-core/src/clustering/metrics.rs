//! Clustering Metrics
//!
//! This module provides metrics for evaluating clustering quality.
//!
//! ## Internal Metrics (no ground truth needed)
//!
//! - [`silhouette_score`] - Mean silhouette coefficient
//! - [`silhouette_samples`] - Per-sample silhouette coefficients
//! - [`calinski_harabasz_score`] - Variance ratio criterion
//! - [`davies_bouldin_score`] - Average similarity between clusters
//! - [`hopkins_statistic`] - Clustering tendency assessment
//!
//! ## External Metrics (ground truth required)
//!
//! - [`adjusted_rand_index`] - Adjusted for chance agreement
//! - [`normalized_mutual_info`] - Normalized mutual information

use crate::{FerroError, Result};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::prelude::*;

/// Compute silhouette score for clustering
///
/// The silhouette score measures how similar a sample is to its own cluster
/// compared to other clusters. Ranges from -1 to 1, where 1 is best.
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `labels` - Cluster labels for each sample
///
/// # Returns
/// Mean silhouette coefficient
pub fn silhouette_score(x: &Array2<f64>, labels: &Array1<i32>) -> Result<f64> {
    let samples = silhouette_samples(x, labels)?;
    Ok(samples.mean().unwrap_or(0.0))
}

/// Compute silhouette coefficient for each sample
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `labels` - Cluster labels for each sample
///
/// # Returns
/// Silhouette coefficient for each sample
pub fn silhouette_samples(x: &Array2<f64>, labels: &Array1<i32>) -> Result<Array1<f64>> {
    let n_samples = x.nrows();

    if n_samples != labels.len() {
        return Err(FerroError::InvalidInput(
            "x and labels must have same number of samples".to_string(),
        ));
    }

    // Get unique labels (excluding -1 for noise)
    let mut unique_labels: Vec<i32> = labels.iter().cloned().filter(|&l| l >= 0).collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();

    if unique_labels.len() < 2 {
        return Err(FerroError::InvalidInput(
            "Need at least 2 clusters to compute silhouette".to_string(),
        ));
    }

    let mut silhouettes = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let label_i = labels[i];

        // Skip noise points
        if label_i < 0 {
            silhouettes[i] = 0.0;
            continue;
        }

        // Compute mean distance to points in same cluster (a)
        let mut a_sum = 0.0;
        let mut a_count = 0;
        for j in 0..n_samples {
            if i != j && labels[j] == label_i {
                a_sum += euclidean_distance(&x.row(i), &x.row(j));
                a_count += 1;
            }
        }
        let a = if a_count > 0 {
            a_sum / a_count as f64
        } else {
            0.0
        };

        // Compute mean distance to points in nearest other cluster (b)
        let mut b = f64::MAX;
        for &other_label in &unique_labels {
            if other_label == label_i {
                continue;
            }

            let mut dist_sum = 0.0;
            let mut dist_count = 0;
            for j in 0..n_samples {
                if labels[j] == other_label {
                    dist_sum += euclidean_distance(&x.row(i), &x.row(j));
                    dist_count += 1;
                }
            }

            if dist_count > 0 {
                let mean_dist = dist_sum / dist_count as f64;
                if mean_dist < b {
                    b = mean_dist;
                }
            }
        }

        // Compute silhouette
        if a.max(b) > 0.0 {
            silhouettes[i] = (b - a) / a.max(b);
        } else {
            silhouettes[i] = 0.0;
        }
    }

    Ok(silhouettes)
}

/// Compute Calinski-Harabasz score (Variance Ratio Criterion)
///
/// Higher values indicate better-defined clusters. Also known as the
/// Variance Ratio Criterion.
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `labels` - Cluster labels for each sample
pub fn calinski_harabasz_score(x: &Array2<f64>, labels: &Array1<i32>) -> Result<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples != labels.len() {
        return Err(FerroError::InvalidInput(
            "x and labels must have same number of samples".to_string(),
        ));
    }

    // Get unique labels (excluding noise)
    let mut unique_labels: Vec<i32> = labels.iter().cloned().filter(|&l| l >= 0).collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return Err(FerroError::InvalidInput(
            "Need at least 2 clusters".to_string(),
        ));
    }

    // Compute overall centroid
    let overall_centroid = x.mean_axis(Axis(0)).unwrap();

    // Compute cluster centroids and sizes
    let mut cluster_centroids: Array2<f64> = Array2::zeros((n_clusters, n_features));
    let mut cluster_sizes = vec![0usize; n_clusters];

    for (k, &label) in unique_labels.iter().enumerate() {
        for i in 0..n_samples {
            if labels[i] == label {
                for j in 0..n_features {
                    cluster_centroids[[k, j]] += x[[i, j]];
                }
                cluster_sizes[k] += 1;
            }
        }
        for j in 0..n_features {
            if cluster_sizes[k] > 0 {
                cluster_centroids[[k, j]] /= cluster_sizes[k] as f64;
            }
        }
    }

    // Between-cluster dispersion (BGss)
    let mut bgss = 0.0;
    for (k, _) in unique_labels.iter().enumerate() {
        let n_k = cluster_sizes[k] as f64;
        for j in 0..n_features {
            let diff: f64 = cluster_centroids[[k, j]] - overall_centroid[j];
            bgss += n_k * diff * diff;
        }
    }

    // Within-cluster dispersion (WGss)
    let mut wgss = 0.0;
    for (k, &label) in unique_labels.iter().enumerate() {
        for i in 0..n_samples {
            if labels[i] == label {
                for j in 0..n_features {
                    let diff: f64 = x[[i, j]] - cluster_centroids[[k, j]];
                    wgss += diff * diff;
                }
            }
        }
    }

    // CH = (BGss / (k-1)) / (WGss / (n-k))
    let n_valid = labels.iter().filter(|&&l| l >= 0).count();
    if wgss == 0.0 || n_valid <= n_clusters {
        return Ok(0.0);
    }

    let score = (bgss / (n_clusters - 1) as f64) / (wgss / (n_valid - n_clusters) as f64);
    Ok(score)
}

/// Compute Davies-Bouldin score
///
/// Lower values indicate better clustering. Computes the average similarity
/// between each cluster and its most similar cluster.
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `labels` - Cluster labels for each sample
pub fn davies_bouldin_score(x: &Array2<f64>, labels: &Array1<i32>) -> Result<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    if n_samples != labels.len() {
        return Err(FerroError::InvalidInput(
            "x and labels must have same number of samples".to_string(),
        ));
    }

    // Get unique labels
    let mut unique_labels: Vec<i32> = labels.iter().cloned().filter(|&l| l >= 0).collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return Err(FerroError::InvalidInput(
            "Need at least 2 clusters".to_string(),
        ));
    }

    // Compute cluster centroids
    let mut cluster_centroids: Array2<f64> = Array2::zeros((n_clusters, n_features));
    let mut cluster_sizes = vec![0usize; n_clusters];

    for (k, &label) in unique_labels.iter().enumerate() {
        for i in 0..n_samples {
            if labels[i] == label {
                for j in 0..n_features {
                    cluster_centroids[[k, j]] += x[[i, j]];
                }
                cluster_sizes[k] += 1;
            }
        }
        for j in 0..n_features {
            if cluster_sizes[k] > 0 {
                cluster_centroids[[k, j]] /= cluster_sizes[k] as f64;
            }
        }
    }

    // Compute intra-cluster distances (average distance to centroid)
    let mut s = vec![0.0; n_clusters];
    for (k, &label) in unique_labels.iter().enumerate() {
        let mut sum = 0.0;
        for i in 0..n_samples {
            if labels[i] == label {
                sum += euclidean_distance(&x.row(i), &cluster_centroids.row(k));
            }
        }
        s[k] = if cluster_sizes[k] > 0 {
            sum / cluster_sizes[k] as f64
        } else {
            0.0
        };
    }

    // Compute Davies-Bouldin index
    let mut db_sum = 0.0;
    for i in 0..n_clusters {
        let mut max_ratio = 0.0;
        for j in 0..n_clusters {
            if i != j {
                let centroid_dist =
                    euclidean_distance(&cluster_centroids.row(i), &cluster_centroids.row(j));
                if centroid_dist > 0.0 {
                    let ratio = (s[i] + s[j]) / centroid_dist;
                    if ratio > max_ratio {
                        max_ratio = ratio;
                    }
                }
            }
        }
        db_sum += max_ratio;
    }

    Ok(db_sum / n_clusters as f64)
}

/// Compute Adjusted Rand Index
///
/// Measures similarity between two clusterings, adjusted for chance.
/// Ranges from -1 to 1, where 1 is perfect agreement.
///
/// # Arguments
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
pub fn adjusted_rand_index(labels_true: &Array1<i32>, labels_pred: &Array1<i32>) -> Result<f64> {
    let n = labels_true.len();
    if n != labels_pred.len() {
        return Err(FerroError::InvalidInput(
            "Labels must have same length".to_string(),
        ));
    }

    // Build contingency table
    let mut true_labels: Vec<i32> = labels_true.iter().cloned().collect();
    let mut pred_labels: Vec<i32> = labels_pred.iter().cloned().collect();

    true_labels.sort_unstable();
    true_labels.dedup();
    pred_labels.sort_unstable();
    pred_labels.dedup();

    let n_true = true_labels.len();
    let n_pred = pred_labels.len();

    // Map labels to indices
    let true_map: std::collections::HashMap<i32, usize> = true_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();
    let pred_map: std::collections::HashMap<i32, usize> = pred_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    // Build contingency table
    let mut contingency = vec![vec![0i64; n_pred]; n_true];
    for i in 0..n {
        let ti = true_map[&labels_true[i]];
        let pi = pred_map[&labels_pred[i]];
        contingency[ti][pi] += 1;
    }

    // Compute row and column sums
    let mut row_sums = vec![0i64; n_true];
    let mut col_sums = vec![0i64; n_pred];
    for i in 0..n_true {
        for j in 0..n_pred {
            row_sums[i] += contingency[i][j];
            col_sums[j] += contingency[i][j];
        }
    }

    // Compute ARI components
    let mut sum_comb_c = 0i64; // sum of C(n_ij, 2)
    for i in 0..n_true {
        for j in 0..n_pred {
            let nij = contingency[i][j];
            sum_comb_c += nij * (nij - 1) / 2;
        }
    }

    let mut sum_comb_a = 0i64; // sum of C(a_i, 2)
    for &a in &row_sums {
        sum_comb_a += a * (a - 1) / 2;
    }

    let mut sum_comb_b = 0i64; // sum of C(b_j, 2)
    for &b in &col_sums {
        sum_comb_b += b * (b - 1) / 2;
    }

    let n_total = n as i64;
    let comb_n = n_total * (n_total - 1) / 2;

    let expected = (sum_comb_a as f64 * sum_comb_b as f64) / comb_n as f64;
    let max_index = (sum_comb_a + sum_comb_b) as f64 / 2.0;

    if max_index == expected {
        return Ok(1.0);
    }

    let ari = (sum_comb_c as f64 - expected) / (max_index - expected);
    Ok(ari)
}

/// Compute Normalized Mutual Information
///
/// Measures mutual information between two clusterings, normalized to [0, 1].
///
/// # Arguments
/// * `labels_true` - Ground truth cluster labels
/// * `labels_pred` - Predicted cluster labels
pub fn normalized_mutual_info(labels_true: &Array1<i32>, labels_pred: &Array1<i32>) -> Result<f64> {
    let n = labels_true.len();
    if n != labels_pred.len() {
        return Err(FerroError::InvalidInput(
            "Labels must have same length".to_string(),
        ));
    }

    if n == 0 {
        return Ok(1.0);
    }

    // Build contingency table (same as ARI)
    let mut true_labels: Vec<i32> = labels_true.iter().cloned().collect();
    let mut pred_labels: Vec<i32> = labels_pred.iter().cloned().collect();

    true_labels.sort_unstable();
    true_labels.dedup();
    pred_labels.sort_unstable();
    pred_labels.dedup();

    let n_true = true_labels.len();
    let n_pred = pred_labels.len();

    let true_map: std::collections::HashMap<i32, usize> = true_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();
    let pred_map: std::collections::HashMap<i32, usize> = pred_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    let mut contingency = vec![vec![0usize; n_pred]; n_true];
    for i in 0..n {
        let ti = true_map[&labels_true[i]];
        let pi = pred_map[&labels_pred[i]];
        contingency[ti][pi] += 1;
    }

    // Compute row and column sums
    let row_sums: Vec<usize> = contingency.iter().map(|row| row.iter().sum()).collect();
    let col_sums: Vec<usize> = (0..n_pred)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // Compute mutual information
    let n_f = n as f64;
    let mut mi = 0.0;
    for i in 0..n_true {
        for j in 0..n_pred {
            if contingency[i][j] > 0 {
                let p_ij = contingency[i][j] as f64 / n_f;
                let p_i = row_sums[i] as f64 / n_f;
                let p_j = col_sums[j] as f64 / n_f;
                mi += p_ij * (p_ij / (p_i * p_j)).ln();
            }
        }
    }

    // Compute entropies
    let h_true: f64 = row_sums
        .iter()
        .filter(|&&x| x > 0)
        .map(|&x| {
            let p = x as f64 / n_f;
            -p * p.ln()
        })
        .sum();

    let h_pred: f64 = col_sums
        .iter()
        .filter(|&&x| x > 0)
        .map(|&x| {
            let p = x as f64 / n_f;
            -p * p.ln()
        })
        .sum();

    // Normalized using arithmetic mean of entropies
    let norm = (h_true + h_pred) / 2.0;
    if norm == 0.0 {
        return Ok(1.0);
    }

    Ok(mi / norm)
}

/// Compute Hopkins statistic for clustering tendency
///
/// Tests whether the data has a uniform distribution or contains clusters.
/// Values close to 0.5 indicate uniform distribution, close to 1 indicates
/// clustered data.
///
/// # Arguments
/// * `x` - Feature matrix (n_samples, n_features)
/// * `sample_size` - Number of points to sample (default: 10% of n_samples)
/// * `random_state` - Random seed for reproducibility
pub fn hopkins_statistic(
    x: &Array2<f64>,
    sample_size: Option<usize>,
    random_state: Option<u64>,
) -> Result<f64> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let m = sample_size
        .unwrap_or((n_samples as f64 * 0.1).ceil() as usize)
        .max(1);
    let m = m.min(n_samples);

    let mut rng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    // Compute feature ranges for uniform sampling
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

    // Sample m random points from data
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);
    let sample_indices: Vec<usize> = indices.into_iter().take(m).collect();

    // For each sampled point, compute distance to nearest neighbor in X
    let mut u_distances = Vec::with_capacity(m);
    for &idx in &sample_indices {
        let mut min_dist = f64::MAX;
        for i in 0..n_samples {
            if i != idx {
                let dist = euclidean_distance(&x.row(idx), &x.row(i));
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        u_distances.push(min_dist);
    }

    // Generate m random points uniformly in data range
    // For each, compute distance to nearest neighbor in X
    let mut w_distances = Vec::with_capacity(m);
    for _ in 0..m {
        // Generate random point
        let random_point: Vec<f64> = (0..n_features)
            .map(|j| rng.random::<f64>() * (maxs[j] - mins[j]) + mins[j])
            .collect();
        let random_arr = Array1::from_vec(random_point);

        // Find nearest neighbor in X
        let mut min_dist = f64::MAX;
        for i in 0..n_samples {
            let dist = euclidean_distance(&random_arr.view(), &x.row(i));
            if dist < min_dist {
                min_dist = dist;
            }
        }
        w_distances.push(min_dist);
    }

    // Hopkins statistic: sum(w) / (sum(w) + sum(u))
    let sum_u: f64 = u_distances.iter().sum();
    let sum_w: f64 = w_distances.iter().sum();

    if sum_u + sum_w == 0.0 {
        return Ok(0.5);
    }

    Ok(sum_w / (sum_w + sum_u))
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
    fn test_silhouette_score() {
        // Two well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let labels = array![0, 0, 0, 1, 1, 1];

        let score = silhouette_score(&x, &labels).unwrap();
        assert!(score > 0.9); // Should be very high for well-separated clusters
    }

    #[test]
    fn test_calinski_harabasz() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let labels = array![0, 0, 0, 1, 1, 1];

        let score = calinski_harabasz_score(&x, &labels).unwrap();
        assert!(score > 0.0); // Should be positive for well-defined clusters
    }

    #[test]
    fn test_davies_bouldin() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
            ],
        )
        .unwrap();
        let labels = array![0, 0, 0, 1, 1, 1];

        let score = davies_bouldin_score(&x, &labels).unwrap();
        assert!(score < 1.0); // Should be low for well-separated clusters
    }

    #[test]
    fn test_adjusted_rand_index() {
        // Perfect agreement
        let true_labels = array![0, 0, 1, 1, 2, 2];
        let pred_labels = array![0, 0, 1, 1, 2, 2];

        let ari = adjusted_rand_index(&true_labels, &pred_labels).unwrap();
        assert!((ari - 1.0).abs() < 0.001);

        // Different labeling of same clustering
        let pred_labels2 = array![1, 1, 2, 2, 0, 0];
        let ari2 = adjusted_rand_index(&true_labels, &pred_labels2).unwrap();
        assert!((ari2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalized_mutual_info() {
        // Perfect agreement
        let true_labels = array![0, 0, 1, 1];
        let pred_labels = array![0, 0, 1, 1];

        let nmi = normalized_mutual_info(&true_labels, &pred_labels).unwrap();
        assert!((nmi - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hopkins_statistic() {
        // Clustered data should have Hopkins > 0.5
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1,
                10.1, 20.0, 20.0, 20.1, 20.0, 20.0, 20.1, 20.1, 20.1,
            ],
        )
        .unwrap();

        let h = hopkins_statistic(&x, Some(4), Some(42)).unwrap();
        // Clustered data should have Hopkins significantly > 0.5
        assert!(h > 0.5);
    }
}
