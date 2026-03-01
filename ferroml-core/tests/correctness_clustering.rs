//! Correctness tests for the clustering module.
//!
//! Tests cover KMeans, DBSCAN, AgglomerativeClustering, clustering metrics,
//! diagnostics, and statistical extensions. Tests use property-based validation,
//! analytical results, and cross-checks against known mathematical properties.

use ferroml_core::clustering::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score, hopkins_statistic,
    normalized_mutual_info, silhouette_samples, silhouette_score, AgglomerativeClustering,
    ClusterDiagnostics, ClusteringModel, ClusteringStatistics, KMeans, Linkage, DBSCAN,
};
use ndarray::{array, Array1, Array2};
use std::collections::HashSet;

// =============================================================================
// Helper Functions
// =============================================================================

/// Generate well-separated 2D blob data with known cluster structure.
/// Returns (X, true_labels) where clusters are at:
///   cluster 0: center (0, 0)
///   cluster 1: center (10, 10)
///   cluster 2: center (10, 0)
/// with very small spread (0.1) so clusters are trivially separable.
fn make_blobs_3c() -> (Array2<f64>, Array1<i32>) {
    // 5 points per cluster, 15 total
    let x = Array2::from_shape_vec(
        (15, 2),
        vec![
            // Cluster 0 near (0, 0)
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, // Cluster 1 near (10, 10)
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, 9.9, 10.0,
            // Cluster 2 near (10, 0)
            10.0, 0.0, 10.1, 0.0, 10.0, 0.1, 10.1, 0.1, 9.9, 0.0,
        ],
    )
    .unwrap();
    let labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
    (x, labels)
}

/// Generate two well-separated clusters in 2D.
fn make_blobs_2c() -> (Array2<f64>, Array1<i32>) {
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            // Cluster 0 near (0, 0)
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0, // Cluster 1 near (10, 10)
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, 9.9, 10.0,
        ],
    )
    .unwrap();
    let labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    (x, labels)
}

/// Generate half-moon shaped data (simplified version).
fn make_moons() -> Array2<f64> {
    // Upper moon
    let mut data = Vec::new();
    for i in 0..10 {
        let angle = std::f64::consts::PI * i as f64 / 9.0;
        data.push(angle.cos());
        data.push(angle.sin());
    }
    // Lower moon (shifted right and down)
    for i in 0..10 {
        let angle = std::f64::consts::PI * i as f64 / 9.0;
        data.push(1.0 - angle.cos());
        data.push(0.5 - angle.sin());
    }
    Array2::from_shape_vec((20, 2), data).unwrap()
}

/// Generate high-dimensional blob data.
fn make_blobs_high_dim(n_features: usize) -> (Array2<f64>, Array1<i32>) {
    let n_per_cluster = 10;
    let n_clusters = 3;
    let n_total = n_per_cluster * n_clusters;
    let mut data = vec![0.0; n_total * n_features];

    for c in 0..n_clusters {
        for i in 0..n_per_cluster {
            let row = c * n_per_cluster + i;
            for j in 0..n_features {
                // Center of cluster c at (c*10, c*10, ...) with small perturbation
                data[row * n_features + j] =
                    (c as f64) * 10.0 + ((i * 7 + j * 3) % 10) as f64 * 0.01;
            }
        }
    }

    let labels = Array1::from_vec(
        (0..n_clusters)
            .flat_map(|c| vec![c as i32; n_per_cluster])
            .collect(),
    );
    (
        Array2::from_shape_vec((n_total, n_features), data).unwrap(),
        labels,
    )
}

/// Count unique non-negative labels.
fn count_clusters(labels: &Array1<i32>) -> usize {
    let unique: HashSet<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
    unique.len()
}

/// Check that two clusterings agree up to label permutation using ARI.
fn clusterings_agree(a: &Array1<i32>, b: &Array1<i32>) -> bool {
    let ari = adjusted_rand_index(a, b).unwrap();
    ari > 0.99
}

// =============================================================================
// KMeans Tests
// =============================================================================

#[test]
fn test_kmeans_blobs_finds_correct_clusters() {
    let (x, true_labels) = make_blobs_3c();
    let mut kmeans = KMeans::new(3).random_state(42).n_init(5);
    kmeans.fit(&x).unwrap();

    let pred_labels = kmeans.labels().unwrap();
    // Clusters should match ground truth (up to permutation)
    assert!(
        clusterings_agree(&true_labels, pred_labels),
        "KMeans labels do not match ground truth (ARI={:.4})",
        adjusted_rand_index(&true_labels, pred_labels).unwrap()
    );
}

#[test]
fn test_kmeans_centers_near_true_centers() {
    let (x, _) = make_blobs_3c();
    let mut kmeans = KMeans::new(3).random_state(42).n_init(5);
    kmeans.fit(&x).unwrap();

    let centers = kmeans.cluster_centers().unwrap();
    // Each center should be near one of (0,0), (10,10), (10,0)
    let expected_centers = vec![(0.02, 0.04), (10.02, 10.04), (10.02, 0.04)];

    for expected in &expected_centers {
        let mut found = false;
        for row in 0..3 {
            let dx = centers[[row, 0]] - expected.0;
            let dy = centers[[row, 1]] - expected.1;
            if (dx * dx + dy * dy).sqrt() < 0.5 {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "No center found near ({}, {})",
            expected.0, expected.1
        );
    }
}

#[test]
fn test_kmeans_inertia_positive() {
    let (x, _) = make_blobs_3c();
    let mut kmeans = KMeans::new(3).random_state(42);
    kmeans.fit(&x).unwrap();

    let inertia = kmeans.inertia().unwrap();
    assert!(inertia >= 0.0, "Inertia should be non-negative");
    // With tight clusters, inertia should be very small
    assert!(
        inertia < 1.0,
        "Inertia too large for tight clusters: {}",
        inertia
    );
}

#[test]
fn test_kmeans_inertia_decreases_with_k() {
    let (x, _) = make_blobs_3c();
    let mut inertias = Vec::new();

    for k in 1..=5 {
        let mut kmeans = KMeans::new(k).random_state(42).n_init(3);
        kmeans.fit(&x).unwrap();
        inertias.push(kmeans.inertia().unwrap());
    }

    // Inertia should monotonically decrease (or stay same) as k increases
    for i in 1..inertias.len() {
        assert!(
            inertias[i] <= inertias[i - 1] + 1e-6,
            "Inertia should decrease with k: k={} inertia={:.6} > k={} inertia={:.6}",
            i + 1,
            inertias[i],
            i,
            inertias[i - 1]
        );
    }
}

#[test]
fn test_kmeans_high_dimensional() {
    let (x, true_labels) = make_blobs_high_dim(50);
    let mut kmeans = KMeans::new(3).random_state(42).n_init(5);
    kmeans.fit(&x).unwrap();

    let pred_labels = kmeans.labels().unwrap();
    assert!(
        clusterings_agree(&true_labels, pred_labels),
        "KMeans failed on 50-dim data"
    );
}

#[test]
fn test_kmeans_many_clusters() {
    // Create 8 well-separated clusters in 2D
    let n_clusters = 8;
    let n_per = 5;
    let mut data = Vec::new();
    for c in 0..n_clusters {
        let cx = (c % 4) as f64 * 20.0;
        let cy = (c / 4) as f64 * 20.0;
        for i in 0..n_per {
            data.push(cx + (i as f64) * 0.01);
            data.push(cy + (i as f64) * 0.01);
        }
    }
    let x = Array2::from_shape_vec((n_clusters * n_per, 2), data).unwrap();

    let mut kmeans = KMeans::new(n_clusters).random_state(42).n_init(5);
    kmeans.fit(&x).unwrap();

    let labels = kmeans.labels().unwrap();
    assert_eq!(count_clusters(labels), n_clusters);
}

#[test]
fn test_kmeans_single_cluster() {
    let (x, _) = make_blobs_3c();
    let mut kmeans = KMeans::new(1).random_state(42);
    kmeans.fit(&x).unwrap();

    let labels = kmeans.labels().unwrap();
    // All points should be in cluster 0
    assert!(labels.iter().all(|&l| l == 0));
    assert_eq!(count_clusters(labels), 1);
}

#[test]
fn test_kmeans_predict_feature_mismatch_error() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![1.0, 2.0, 1.5, 1.8, 1.0, 1.0, 8.0, 8.0, 8.5, 8.0, 8.0, 7.5],
    )
    .unwrap();

    let mut kmeans = KMeans::new(2).random_state(42);
    kmeans.fit(&x).unwrap();

    // Try to predict with wrong number of features
    let wrong_x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = kmeans.predict(&wrong_x);
    assert!(
        result.is_err(),
        "predict() should error on feature count mismatch"
    );

    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("Shape mismatch") || err_msg.contains("shape"),
        "Error should mention shape mismatch, got: {}",
        err_msg
    );
}

#[test]
fn test_kmeans_predict_correct_features_works() {
    let x =
        Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.5, 1.5, 10.0, 10.0, 10.5, 10.5]).unwrap();

    let mut kmeans = KMeans::new(2).random_state(42);
    kmeans.fit(&x).unwrap();

    // Correct number of features should work
    let new_x = Array2::from_shape_vec((2, 2), vec![1.2, 1.2, 10.2, 10.2]).unwrap();
    let labels = kmeans.predict(&new_x).unwrap();
    assert_eq!(labels.len(), 2);
    assert_ne!(labels[0], labels[1]);
}

#[test]
fn test_kmeans_empty_cluster_reinitialization() {
    // Use more clusters than natural groupings to force empty clusters
    // with tight data that may cause some clusters to be empty initially
    let x = Array2::from_shape_vec(
        (10, 2),
        vec![
            0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01, 0.02, 0.0, 10.0, 10.0, 10.01, 10.0, 10.0,
            10.01, 10.01, 10.01, 10.02, 10.0,
        ],
    )
    .unwrap();

    // k=5 with only 2 natural clusters -- some clusters may initially be empty
    let mut kmeans = KMeans::new(5).random_state(42).max_iter(100);
    let result = kmeans.fit(&x);
    // Should not panic or error
    assert!(
        result.is_ok(),
        "KMeans with k > natural clusters should not fail"
    );

    let labels = kmeans.labels().unwrap();
    assert_eq!(labels.len(), 10);
    // Every point should have a valid label
    assert!(labels.iter().all(|&l| l >= 0));
}

#[test]
fn test_kmeans_deterministic_with_seed() {
    let (x, _) = make_blobs_3c();

    let mut km1 = KMeans::new(3).random_state(42).n_init(1);
    km1.fit(&x).unwrap();
    let labels1 = km1.labels().unwrap().clone();
    let inertia1 = km1.inertia().unwrap();

    let mut km2 = KMeans::new(3).random_state(42).n_init(1);
    km2.fit(&x).unwrap();
    let labels2 = km2.labels().unwrap().clone();
    let inertia2 = km2.inertia().unwrap();

    assert_eq!(labels1, labels2, "Same seed should produce same labels");
    assert!(
        (inertia1 - inertia2).abs() < 1e-10,
        "Same seed should produce same inertia"
    );
}

#[test]
fn test_kmeans_k_equals_n() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

    let mut kmeans = KMeans::new(4).random_state(42);
    kmeans.fit(&x).unwrap();

    let labels = kmeans.labels().unwrap();
    // Each point should be in its own cluster
    let unique: HashSet<i32> = labels.iter().copied().collect();
    assert_eq!(unique.len(), 4);
    // Inertia should be 0 (each point is its own center)
    assert!(
        kmeans.inertia().unwrap() < 1e-10,
        "Inertia should be ~0 when k=n"
    );
}

// =============================================================================
// DBSCAN Tests
// =============================================================================

#[test]
fn test_dbscan_finds_two_clusters() {
    let (x, true_labels) = make_blobs_2c();
    let mut dbscan = DBSCAN::new(1.0, 2);
    dbscan.fit(&x).unwrap();

    let labels = dbscan.labels().unwrap();
    assert_eq!(dbscan.n_clusters(), Some(2));

    // Should match ground truth
    assert!(
        clusterings_agree(&true_labels, labels),
        "DBSCAN did not find the correct clusters"
    );
}

#[test]
fn test_dbscan_all_noise() {
    // Use very small eps so no points are core points
    let (x, _) = make_blobs_3c();
    let mut dbscan = DBSCAN::new(0.001, 5);
    dbscan.fit(&x).unwrap();

    let labels = dbscan.labels().unwrap();
    // All points should be noise (-1)
    assert!(
        labels.iter().all(|&l| l == -1),
        "All points should be noise with tiny eps"
    );
    assert_eq!(dbscan.n_noise(), Some(15));
    assert_eq!(dbscan.n_clusters(), Some(0));
}

#[test]
fn test_dbscan_single_cluster() {
    // Use very large eps so all points are in one cluster
    let (x, _) = make_blobs_3c();
    let mut dbscan = DBSCAN::new(100.0, 2);
    dbscan.fit(&x).unwrap();

    let labels = dbscan.labels().unwrap();
    assert_eq!(dbscan.n_clusters(), Some(1));
    assert_eq!(dbscan.n_noise(), Some(0));
    assert!(labels.iter().all(|&l| l == 0));
}

#[test]
fn test_dbscan_noise_count() {
    // Two tight clusters plus one outlier far away
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1,
            50.0, 50.0, // outlier
        ],
    )
    .unwrap();

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&x).unwrap();

    assert_eq!(dbscan.n_clusters(), Some(2));
    assert_eq!(dbscan.n_noise(), Some(1));

    let labels = dbscan.labels().unwrap();
    assert_eq!(labels[8], -1, "The outlier should be noise");
}

#[test]
fn test_dbscan_core_indices() {
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&x).unwrap();

    let core_indices = dbscan.core_sample_indices().unwrap();
    // All 8 points should be core samples (each has >= 2 neighbors within eps=0.5)
    assert_eq!(core_indices.len(), 8);
}

#[test]
fn test_dbscan_moons_data() {
    let x = make_moons();
    // With appropriate eps, DBSCAN should find 2 clusters
    let mut dbscan = DBSCAN::new(0.5, 3);
    dbscan.fit(&x).unwrap();

    let _labels = dbscan.labels().unwrap();
    let n_clusters = dbscan.n_clusters().unwrap_or(0);
    // Should find at least 1 cluster (exact count depends on eps/min_samples)
    assert!(n_clusters >= 1, "DBSCAN should find clusters in moon data");
}

#[test]
fn test_dbscan_min_samples_one() {
    // With min_samples=1, every point is a core point -- no noise
    let (x, _) = make_blobs_3c();
    let mut dbscan = DBSCAN::new(1.0, 1);
    dbscan.fit(&x).unwrap();

    assert_eq!(
        dbscan.n_noise(),
        Some(0),
        "With min_samples=1, no noise should exist"
    );
}

#[test]
fn test_dbscan_predict_new_data() {
    let (x, _) = make_blobs_2c();
    let mut dbscan = DBSCAN::new(1.0, 2);
    dbscan.fit(&x).unwrap();

    // Predict a point near cluster 0
    let new_x = Array2::from_shape_vec((1, 2), vec![0.05, 0.05]).unwrap();
    let pred = dbscan.predict(&new_x).unwrap();
    assert!(
        pred[0] >= 0,
        "Point near cluster should be assigned to a cluster"
    );
}

// =============================================================================
// AgglomerativeClustering Tests
// =============================================================================

#[test]
fn test_ward_linkage_correct_clustering() {
    let (x, true_labels) = make_blobs_3c();
    let mut model = AgglomerativeClustering::new(3).with_linkage(Linkage::Ward);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    assert!(
        clusterings_agree(&true_labels, labels),
        "Ward linkage should find correct clusters"
    );
}

#[test]
fn test_ward_linkage_dendrogram_distances_nondecreasing() {
    // In Ward linkage, merge distances should be non-decreasing
    let (x, _) = make_blobs_3c();
    let mut model = AgglomerativeClustering::new(1).with_linkage(Linkage::Ward);
    model.fit(&x).unwrap();

    let children = model.children().unwrap();
    // Check merge distances are non-decreasing
    for i in 1..children.len() {
        assert!(
            children[i].2 >= children[i - 1].2 - 1e-10,
            "Ward merge distances should be non-decreasing: {} < {} at step {}",
            children[i].2,
            children[i - 1].2,
            i
        );
    }
}

#[test]
fn test_single_linkage_correct_clustering() {
    let (x, true_labels) = make_blobs_3c();
    let mut model = AgglomerativeClustering::new(3).with_linkage(Linkage::Single);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    assert!(
        clusterings_agree(&true_labels, labels),
        "Single linkage should find correct clusters"
    );
}

#[test]
fn test_complete_linkage_correct_clustering() {
    let (x, true_labels) = make_blobs_3c();
    let mut model = AgglomerativeClustering::new(3).with_linkage(Linkage::Complete);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    assert!(
        clusterings_agree(&true_labels, labels),
        "Complete linkage should find correct clusters"
    );
}

#[test]
fn test_average_linkage_correct_clustering() {
    let (x, true_labels) = make_blobs_3c();
    let mut model = AgglomerativeClustering::new(3).with_linkage(Linkage::Average);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    assert!(
        clusterings_agree(&true_labels, labels),
        "Average linkage should find correct clusters"
    );
}

#[test]
fn test_agglomerative_two_points() {
    let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 10.0, 10.0]).unwrap();

    // With 2 clusters, each point should be its own cluster
    let mut model = AgglomerativeClustering::new(2);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    assert_ne!(labels[0], labels[1]);

    // With 1 cluster, both should be in same cluster
    let mut model1 = AgglomerativeClustering::new(1);
    model1.fit(&x).unwrap();

    let labels1 = model1.labels().unwrap();
    assert_eq!(labels1[0], labels1[1]);
}

#[test]
fn test_agglomerative_identical_points() {
    // All points are identical -- should still work
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    )
    .unwrap();

    let mut model = AgglomerativeClustering::new(2);
    let result = model.fit(&x);
    // Should not panic; may produce degenerate clusters but should complete
    assert!(result.is_ok(), "Identical points should not cause a crash");

    let labels = model.labels().unwrap();
    assert_eq!(labels.len(), 6);
}

#[test]
fn test_agglomerative_merge_count() {
    // n samples merged into k clusters requires (n - k) merges
    let (x, _) = make_blobs_3c();
    let n = x.nrows();

    for k in 1..=5 {
        let mut model = AgglomerativeClustering::new(k);
        model.fit(&x).unwrap();
        let children = model.children().unwrap();
        assert_eq!(
            children.len(),
            n - k,
            "n={} samples, k={} clusters should produce {} merges, got {}",
            n,
            k,
            n - k,
            children.len()
        );
    }
}

#[test]
fn test_ward_linkage_four_equidistant_points() {
    // Four points forming a square: (0,0), (1,0), (0,1), (1,1)
    // Ward should first merge closest pairs, then merge the two pairs
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    let mut model = AgglomerativeClustering::new(2).with_linkage(Linkage::Ward);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    let unique: HashSet<i32> = labels.iter().copied().collect();
    assert_eq!(unique.len(), 2, "Should produce exactly 2 clusters");

    // Each cluster should have 2 points
    let c0_count = labels.iter().filter(|&&l| l == 0).count();
    let c1_count = labels.iter().filter(|&&l| l == 1).count();
    assert_eq!(c0_count, 2);
    assert_eq!(c1_count, 2);
}

// =============================================================================
// Metrics Tests
// =============================================================================

#[test]
fn test_silhouette_score_well_separated() {
    let (x, labels) = make_blobs_2c();
    let score = silhouette_score(&x, &labels).unwrap();

    // Well-separated clusters should have silhouette close to 1
    assert!(
        score > 0.95,
        "Silhouette score should be >0.95 for well-separated clusters, got {}",
        score
    );
}

#[test]
fn test_silhouette_score_range() {
    let (x, labels) = make_blobs_3c();
    let score = silhouette_score(&x, &labels).unwrap();

    assert!(
        (-1.0..=1.0).contains(&score),
        "Silhouette must be in [-1, 1], got {}",
        score
    );
}

#[test]
fn test_silhouette_samples_count() {
    let (x, labels) = make_blobs_3c();
    let samples = silhouette_samples(&x, &labels).unwrap();

    assert_eq!(samples.len(), x.nrows());
    // All samples from well-separated clusters should have positive silhouette
    for s in samples.iter() {
        assert!(
            *s > 0.0,
            "Well-separated sample should have positive silhouette"
        );
    }
}

#[test]
fn test_silhouette_samples_vs_score() {
    // silhouette_score should equal mean of silhouette_samples
    let (x, labels) = make_blobs_3c();
    let score = silhouette_score(&x, &labels).unwrap();
    let samples = silhouette_samples(&x, &labels).unwrap();
    let mean_samples = samples.mean().unwrap();

    assert!(
        (score - mean_samples).abs() < 1e-10,
        "silhouette_score ({}) should equal mean of silhouette_samples ({})",
        score,
        mean_samples
    );
}

#[test]
fn test_calinski_harabasz_well_separated() {
    let (x, labels) = make_blobs_2c();
    let score = calinski_harabasz_score(&x, &labels).unwrap();

    // Well-separated clusters should have high CH score
    assert!(
        score > 100.0,
        "CH score should be high for well-separated clusters, got {}",
        score
    );
}

#[test]
fn test_calinski_harabasz_positive() {
    let (x, labels) = make_blobs_3c();
    let score = calinski_harabasz_score(&x, &labels).unwrap();

    assert!(
        score > 0.0,
        "CH score should be positive for valid clusters"
    );
}

#[test]
fn test_davies_bouldin_well_separated() {
    let (x, labels) = make_blobs_2c();
    let score = davies_bouldin_score(&x, &labels).unwrap();

    // Well-separated clusters should have low DB score (close to 0)
    assert!(
        score < 0.1,
        "DB score should be near 0 for well-separated clusters, got {}",
        score
    );
}

#[test]
fn test_davies_bouldin_nonnegative() {
    let (x, labels) = make_blobs_3c();
    let score = davies_bouldin_score(&x, &labels).unwrap();
    assert!(score >= 0.0, "Davies-Bouldin score should be non-negative");
}

#[test]
fn test_ari_perfect_agreement() {
    let a = array![0, 0, 1, 1, 2, 2];
    let b = array![0, 0, 1, 1, 2, 2];
    let ari = adjusted_rand_index(&a, &b).unwrap();
    assert!(
        (ari - 1.0).abs() < 1e-10,
        "Perfect agreement should give ARI=1.0, got {}",
        ari
    );
}

#[test]
fn test_ari_permuted_labels() {
    let a = array![0, 0, 1, 1, 2, 2];
    let b = array![2, 2, 0, 0, 1, 1]; // Same clustering, different label names
    let ari = adjusted_rand_index(&a, &b).unwrap();
    assert!(
        (ari - 1.0).abs() < 1e-10,
        "Permuted labels should give ARI=1.0, got {}",
        ari
    );
}

#[test]
fn test_ari_random_labels() {
    // Random labels vs structured labels should give ARI near 0
    let true_labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
    let random_labels = array![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let ari = adjusted_rand_index(&true_labels, &random_labels).unwrap();
    assert!(
        ari.abs() < 0.3,
        "ARI of random vs structured should be near 0, got {}",
        ari
    );
}

#[test]
fn test_ari_symmetric() {
    let a = array![0, 0, 1, 1, 2, 2];
    let b = array![0, 1, 1, 2, 2, 0];
    let ari_ab = adjusted_rand_index(&a, &b).unwrap();
    let ari_ba = adjusted_rand_index(&b, &a).unwrap();
    assert!(
        (ari_ab - ari_ba).abs() < 1e-10,
        "ARI should be symmetric: {} vs {}",
        ari_ab,
        ari_ba
    );
}

#[test]
fn test_nmi_perfect_agreement() {
    let a = array![0, 0, 1, 1];
    let b = array![0, 0, 1, 1];
    let nmi = normalized_mutual_info(&a, &b).unwrap();
    assert!(
        (nmi - 1.0).abs() < 1e-10,
        "Perfect agreement should give NMI=1.0, got {}",
        nmi
    );
}

#[test]
fn test_nmi_permuted_labels() {
    let a = array![0, 0, 1, 1, 2, 2];
    let b = array![2, 2, 0, 0, 1, 1];
    let nmi = normalized_mutual_info(&a, &b).unwrap();
    assert!(
        (nmi - 1.0).abs() < 1e-10,
        "Permuted labels should give NMI=1.0, got {}",
        nmi
    );
}

#[test]
fn test_nmi_range() {
    let a = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let b = array![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let nmi = normalized_mutual_info(&a, &b).unwrap();
    assert!(
        (-1e-10..=1.0 + 1e-10).contains(&nmi),
        "NMI should be in [0, 1], got {}",
        nmi
    );
}

#[test]
fn test_nmi_symmetric() {
    let a = array![0, 0, 1, 1, 2, 2];
    let b = array![0, 1, 1, 2, 2, 0];
    let nmi_ab = normalized_mutual_info(&a, &b).unwrap();
    let nmi_ba = normalized_mutual_info(&b, &a).unwrap();
    assert!(
        (nmi_ab - nmi_ba).abs() < 1e-10,
        "NMI should be symmetric: {} vs {}",
        nmi_ab,
        nmi_ba
    );
}

#[test]
fn test_hopkins_uniform_data() {
    // Generate approximately uniform data
    let n = 100;
    let mut data = Vec::new();
    for i in 0..n {
        let x = (i as f64) / (n as f64);
        let y = ((i * 67 + 13) % n) as f64 / n as f64;
        data.push(x);
        data.push(y);
    }
    let x = Array2::from_shape_vec((n, 2), data).unwrap();

    let h = hopkins_statistic(&x, Some(20), Some(42)).unwrap();
    // Uniform data should have Hopkins near 0.5
    assert!(
        h > 0.2 && h < 0.8,
        "Hopkins for uniform data should be ~0.5, got {}",
        h
    );
}

#[test]
fn test_hopkins_clustered_data() {
    // Tightly clustered data
    let mut data = Vec::new();
    for c in 0..3 {
        let cx = c as f64 * 100.0;
        let cy = c as f64 * 100.0;
        for i in 0..20 {
            data.push(cx + (i as f64) * 0.001);
            data.push(cy + (i as f64) * 0.001);
        }
    }
    let x = Array2::from_shape_vec((60, 2), data).unwrap();

    let h = hopkins_statistic(&x, Some(10), Some(42)).unwrap();
    // Clustered data should have Hopkins > 0.5 (closer to 1.0)
    assert!(
        h > 0.5,
        "Hopkins for clustered data should be >0.5, got {}",
        h
    );
}

#[test]
fn test_hopkins_range() {
    let (x, _) = make_blobs_3c();
    let h = hopkins_statistic(&x, Some(5), Some(42)).unwrap();
    assert!(
        (0.0..=1.0).contains(&h),
        "Hopkins must be in [0, 1], got {}",
        h
    );
}

#[test]
fn test_metrics_require_two_clusters() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
    let labels = array![0, 0, 0, 0]; // Only one cluster

    assert!(silhouette_score(&x, &labels).is_err());
    assert!(calinski_harabasz_score(&x, &labels).is_err());
    assert!(davies_bouldin_score(&x, &labels).is_err());
}

// =============================================================================
// Diagnostics Tests
// =============================================================================

#[test]
fn test_variance_decomposition_sums() {
    let (x, labels) = make_blobs_3c();
    let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();

    // TSS = BGSS + WGSS (total = between + within)
    let total_ss = diag.total_within_ss + diag.between_ss;

    // Compute TSS directly
    let mean_x = x.mean_axis(ndarray::Axis(0)).unwrap();
    let mut direct_tss = 0.0;
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            let diff = x[[i, j]] - mean_x[j];
            direct_tss += diff * diff;
        }
    }

    assert!(
        (total_ss - direct_tss).abs() < 1e-8,
        "WGSS + BGSS ({}) should equal TSS ({})",
        total_ss,
        direct_tss
    );
}

#[test]
fn test_variance_ratio_high_for_separated() {
    let (x, labels) = make_blobs_2c();
    let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();

    // For well-separated clusters, most variance should be between clusters
    let ratio = diag.variance_ratio();
    assert!(
        ratio > 0.99,
        "Variance ratio should be >0.99 for well-separated clusters, got {}",
        ratio
    );
}

#[test]
fn test_dunn_index_well_separated() {
    let (x, labels) = make_blobs_2c();
    let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();

    let dunn = diag.dunn_index();
    assert!(
        dunn > 1.0,
        "Dunn index should be >1 for well-separated clusters, got {}",
        dunn
    );
}

#[test]
fn test_silhouette_per_cluster_consistency() {
    let (x, labels) = make_blobs_3c();
    let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();

    // Per-cluster silhouette should be consistent with overall
    assert_eq!(diag.silhouette_per_cluster.len(), 3);

    // Overall silhouette should be weighted average of per-cluster
    let total_points: usize = diag.cluster_sizes.iter().sum();
    let weighted_mean: f64 = diag
        .silhouette_per_cluster
        .iter()
        .zip(diag.cluster_sizes.iter())
        .map(|(s, &n)| s * n as f64)
        .sum::<f64>()
        / total_points as f64;

    assert!(
        (diag.silhouette_overall - weighted_mean).abs() < 1e-10,
        "Overall silhouette ({}) should equal weighted average of per-cluster ({})",
        diag.silhouette_overall,
        weighted_mean
    );
}

#[test]
fn test_outlier_detection() {
    // Create data where one point in cluster 0 is much closer to cluster 1
    // than to cluster 0 -- should get negative silhouette
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 8.0,
            8.0, // This point is assigned to cluster 0 but much closer to cluster 1
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();
    let labels = array![0, 0, 0, 0, 1, 1, 1, 1];

    let diag = ClusterDiagnostics::from_labels(&x, &labels, Some(0.0)).unwrap();

    // Point 3 (at 8.0, 8.0) should have negative silhouette because it's
    // much closer to cluster 1 centroid than cluster 0 centroid
    let total_outliers: usize = diag.outliers_per_cluster.iter().map(|v| v.len()).sum();
    assert!(total_outliers >= 1, "Should detect at least one outlier");
}

// =============================================================================
// Statistical Extensions Tests
// =============================================================================

#[test]
fn test_gap_statistic_finds_correct_k() {
    let (x, _) = make_blobs_3c();
    let result = KMeans::optimal_k(&x, 1..6, 5, Some(42)).unwrap();

    // Gap statistic should suggest k=3 for 3-cluster data
    // Allow some tolerance since it's stochastic
    assert!(
        result.optimal_k >= 2 && result.optimal_k <= 4,
        "Gap statistic optimal k should be 2-4, got {}",
        result.optimal_k
    );
    assert_eq!(result.k_values.len(), 5);
    assert_eq!(result.gap_values.len(), 5);
}

#[test]
fn test_elbow_method_finds_correct_k() {
    let (x, _) = make_blobs_3c();
    let result = KMeans::elbow(&x, 1..8, Some(42)).unwrap();

    // Elbow should be at k=3 for 3-cluster data (allow tolerance)
    assert!(
        result.optimal_k >= 2 && result.optimal_k <= 5,
        "Elbow optimal k should be 2-5, got {}",
        result.optimal_k
    );
    // Inertia should decrease with k
    for i in 1..result.inertias.len() {
        assert!(
            result.inertias[i] <= result.inertias[i - 1] + 1e-6,
            "Inertia should decrease with k"
        );
    }
}

#[test]
fn test_cluster_stability_stable_clusters() {
    let (x, _) = make_blobs_2c();
    let mut kmeans = KMeans::new(2).random_state(42);
    kmeans.fit(&x).unwrap();

    let stability = kmeans.cluster_stability(&x, 30).unwrap();
    assert_eq!(stability.len(), 2);

    // Well-separated clusters should be highly stable
    for (i, &s) in stability.iter().enumerate() {
        assert!(s > 0.7, "Cluster {} stability should be >0.7, got {}", i, s);
    }
}

#[test]
fn test_silhouette_ci_contains_point_estimate() {
    let (x, _) = make_blobs_2c();
    let mut kmeans = KMeans::new(2).random_state(42);
    kmeans.fit(&x).unwrap();

    let (mean, lower, upper) = kmeans.silhouette_with_ci(&x, 0.95).unwrap();

    // CI should contain the point estimate
    assert!(
        lower <= mean,
        "Lower CI ({}) should be <= mean ({})",
        lower,
        mean
    );
    assert!(
        mean <= upper,
        "Mean ({}) should be <= upper CI ({})",
        mean,
        upper
    );
    // Silhouette should be in valid range
    assert!(
        (-1.0..=1.0).contains(&mean),
        "Mean silhouette should be in [-1, 1]"
    );
}

#[test]
fn test_optimal_eps_reasonable_value() {
    let (x, _) = make_blobs_2c();
    let (suggested_eps, k_distances) = DBSCAN::optimal_eps(&x, 3).unwrap();

    assert!(suggested_eps > 0.0, "Suggested eps should be positive");
    assert_eq!(k_distances.len(), x.nrows());
    // k-distances should be sorted in descending order
    for i in 1..k_distances.len() {
        assert!(k_distances[i] <= k_distances[i - 1] + 1e-10);
    }
}

#[test]
fn test_cluster_persistence_monotonic() {
    let (x, _) = make_blobs_2c();
    let eps_values: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();
    let results = DBSCAN::cluster_persistence(&x, &eps_values, 2).unwrap();

    assert_eq!(results.len(), 20);

    // As eps increases, number of clusters should generally decrease or stay same
    // and noise should generally decrease or stay same
    // (not strictly true for all eps values, but on average)
    let first_n_noise = results[0].2;
    let last_n_noise = results.last().unwrap().2;
    assert!(
        last_n_noise <= first_n_noise || first_n_noise == 0,
        "Noise should decrease with larger eps"
    );
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_all_identical_points_kmeans() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    )
    .unwrap();

    let mut kmeans = KMeans::new(2).random_state(42).max_iter(10);
    let result = kmeans.fit(&x);
    assert!(
        result.is_ok(),
        "KMeans on identical points should not crash"
    );
    // Inertia should be 0 (all points at same location)
    assert!(kmeans.inertia().unwrap() < 1e-10);
}

#[test]
fn test_single_feature_kmeans() {
    // 1D clustering
    let x = Array2::from_shape_vec((8, 1), vec![0.0, 0.1, 0.05, 0.15, 10.0, 10.1, 10.05, 10.15])
        .unwrap();

    let mut kmeans = KMeans::new(2).random_state(42);
    kmeans.fit(&x).unwrap();

    let labels = kmeans.labels().unwrap();
    // First 4 should be in same cluster, last 4 in another
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_eq!(labels[4], labels[5]);
    assert_eq!(labels[6], labels[7]);
    assert_ne!(labels[0], labels[4]);
}

#[test]
fn test_single_feature_dbscan() {
    let x = Array2::from_shape_vec((6, 1), vec![0.0, 0.1, 0.05, 10.0, 10.1, 10.05]).unwrap();

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&x).unwrap();

    assert_eq!(dbscan.n_clusters(), Some(2));
}

#[test]
fn test_high_dimensional_100_features() {
    let (x, true_labels) = make_blobs_high_dim(100);
    let mut kmeans = KMeans::new(3).random_state(42).n_init(5);
    kmeans.fit(&x).unwrap();

    let labels = kmeans.labels().unwrap();
    assert!(
        clusterings_agree(&true_labels, labels),
        "KMeans should work on 100-dim data"
    );
}

#[test]
fn test_dbscan_eps_zero_returns_error() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();

    let mut dbscan = DBSCAN::new(0.0, 2);
    let result = dbscan.fit(&x);
    assert!(result.is_err(), "DBSCAN with eps=0 should return error");
}

#[test]
fn test_kmeans_not_fitted() {
    let kmeans = KMeans::new(3);
    assert!(!kmeans.is_fitted());

    let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
    assert!(kmeans.predict(&x).is_err());
}

#[test]
fn test_dbscan_not_fitted() {
    let dbscan = DBSCAN::new(0.5, 2);
    assert!(!dbscan.is_fitted());

    let x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 2.0, 2.0]).unwrap();
    assert!(dbscan.predict(&x).is_err());
}

#[test]
fn test_agglomerative_not_fitted() {
    let model = AgglomerativeClustering::new(2);
    assert!(!model.is_fitted());
    assert!(model.labels().is_none());
}

// =============================================================================
// Cross-Algorithm Consistency Tests
// =============================================================================

#[test]
fn test_kmeans_and_agglomerative_agree_on_easy_data() {
    let (x, true_labels) = make_blobs_3c();

    let mut kmeans = KMeans::new(3).random_state(42).n_init(5);
    kmeans.fit(&x).unwrap();
    let km_labels = kmeans.labels().unwrap();

    let mut agg = AgglomerativeClustering::new(3).with_linkage(Linkage::Ward);
    agg.fit(&x).unwrap();
    let agg_labels = agg.labels().unwrap();

    // Both should agree with ground truth
    assert!(clusterings_agree(&true_labels, km_labels));
    assert!(clusterings_agree(&true_labels, agg_labels));
    // And therefore agree with each other
    assert!(clusterings_agree(km_labels, agg_labels));
}

#[test]
fn test_metrics_consistent_across_good_clustering() {
    let (x, labels) = make_blobs_3c();

    let silhouette = silhouette_score(&x, &labels).unwrap();
    let ch = calinski_harabasz_score(&x, &labels).unwrap();
    let db = davies_bouldin_score(&x, &labels).unwrap();

    // For well-separated clusters:
    // - Silhouette should be high (near 1)
    // - CH should be high (> 100 for well-separated)
    // - DB should be low (near 0)
    assert!(silhouette > 0.9, "Silhouette should be high");
    assert!(ch > 50.0, "CH should be high");
    assert!(db < 0.5, "DB should be low");
}

#[test]
fn test_metrics_with_noise_labels() {
    // Data with noise label (-1) should be handled
    let x = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 50.0,
            50.0, // noise
        ],
    )
    .unwrap();
    let labels = array![0, 0, 0, 1, 1, 1, -1];

    // Should handle noise labels gracefully
    let score = silhouette_score(&x, &labels).unwrap();
    assert!(
        score > 0.0,
        "Should compute valid silhouette with noise labels"
    );

    let ch = calinski_harabasz_score(&x, &labels).unwrap();
    assert!(ch > 0.0);

    let db = davies_bouldin_score(&x, &labels).unwrap();
    assert!(db >= 0.0);
}

// =============================================================================
// Analytical Correctness Tests (hand-computed expected values)
// =============================================================================

#[test]
fn test_silhouette_score_analytical_four_points() {
    // Four points in 1D: [0, 1] in cluster 0, [10, 11] in cluster 1
    // For point 0 (at x=0):
    //   a(0) = mean dist to same cluster = |0-1| = 1.0
    //   b(0) = mean dist to nearest other cluster = (|0-10| + |0-11|)/2 = 10.5
    //   s(0) = (10.5 - 1.0) / max(10.5, 1.0) = 9.5/10.5 = 0.904762...
    // For point 1 (at x=1):
    //   a(1) = |1-0| = 1.0
    //   b(1) = (|1-10| + |1-11|)/2 = 9.5
    //   s(1) = (9.5 - 1.0) / 9.5 = 8.5/9.5 = 0.894737...
    // For point 2 (at x=10):
    //   a(2) = |10-11| = 1.0
    //   b(2) = (|10-0| + |10-1|)/2 = 9.5
    //   s(2) = (9.5 - 1.0) / 9.5 = 0.894737...
    // For point 3 (at x=11):
    //   a(3) = |11-10| = 1.0
    //   b(3) = (|11-0| + |11-1|)/2 = 10.5
    //   s(3) = (10.5 - 1.0) / 10.5 = 0.904762...
    // Mean = (0.904762 + 0.894737 + 0.894737 + 0.904762) / 4 = 0.899750

    let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 10.0, 11.0]).unwrap();
    let labels = array![0, 0, 1, 1];

    let samples = silhouette_samples(&x, &labels).unwrap();
    let score = silhouette_score(&x, &labels).unwrap();

    let expected_s0 = 9.5 / 10.5;
    let expected_s1 = 8.5 / 9.5;
    let expected_s2 = 8.5 / 9.5;
    let expected_s3 = 9.5 / 10.5;
    let expected_mean = (expected_s0 + expected_s1 + expected_s2 + expected_s3) / 4.0;

    assert!(
        (samples[0] - expected_s0).abs() < 1e-10,
        "s(0) expected {}, got {}",
        expected_s0,
        samples[0]
    );
    assert!(
        (samples[1] - expected_s1).abs() < 1e-10,
        "s(1) expected {}, got {}",
        expected_s1,
        samples[1]
    );
    assert!(
        (samples[2] - expected_s2).abs() < 1e-10,
        "s(2) expected {}, got {}",
        expected_s2,
        samples[2]
    );
    assert!(
        (samples[3] - expected_s3).abs() < 1e-10,
        "s(3) expected {}, got {}",
        expected_s3,
        samples[3]
    );
    assert!(
        (score - expected_mean).abs() < 1e-10,
        "Mean silhouette expected {}, got {}",
        expected_mean,
        score
    );
}

#[test]
fn test_calinski_harabasz_analytical() {
    // 2 clusters in 1D: cluster 0 = {0, 2}, cluster 1 = {10, 12}
    // Overall centroid = (0+2+10+12)/4 = 6
    // Cluster 0 centroid = 1, Cluster 1 centroid = 11
    // BGSS = 2*(1-6)^2 + 2*(11-6)^2 = 2*25 + 2*25 = 100
    // WGSS = (0-1)^2 + (2-1)^2 + (10-11)^2 + (12-11)^2 = 1+1+1+1 = 4
    // CH = (BGSS/(k-1)) / (WGSS/(n-k)) = (100/1) / (4/2) = 100/2 = 50

    let x = Array2::from_shape_vec((4, 1), vec![0.0, 2.0, 10.0, 12.0]).unwrap();
    let labels = array![0, 0, 1, 1];

    let ch = calinski_harabasz_score(&x, &labels).unwrap();
    assert!(
        (ch - 50.0).abs() < 1e-10,
        "CH score expected 50.0, got {}",
        ch
    );
}

#[test]
fn test_davies_bouldin_analytical() {
    // 2 clusters in 1D: cluster 0 = {0, 2}, cluster 1 = {10, 12}
    // Centroids: c0=1, c1=11
    // S0 = mean dist to centroid = (|0-1| + |2-1|) / 2 = 1.0
    // S1 = mean dist to centroid = (|10-11| + |12-11|) / 2 = 1.0
    // d(c0, c1) = |1-11| = 10
    // R01 = (S0 + S1) / d(c0,c1) = 2/10 = 0.2
    // DB = max(R_ij) for each cluster, averaged
    // For cluster 0: max_j(R0j) = R01 = 0.2
    // For cluster 1: max_j(R1j) = R10 = 0.2
    // DB = (0.2 + 0.2) / 2 = 0.2

    let x = Array2::from_shape_vec((4, 1), vec![0.0, 2.0, 10.0, 12.0]).unwrap();
    let labels = array![0, 0, 1, 1];

    let db = davies_bouldin_score(&x, &labels).unwrap();
    assert!(
        (db - 0.2).abs() < 1e-10,
        "DB score expected 0.2, got {}",
        db
    );
}

#[test]
fn test_ari_analytical_known_value() {
    // ARI for specific known case:
    // true = [0, 0, 0, 1, 1, 1], pred = [0, 0, 1, 1, 1, 1]
    // Contingency:
    //        pred0  pred1
    // true0:   2      1
    // true1:   0      3
    //
    // Row sums: 3, 3; Col sums: 2, 4
    // C(2,2)=1, C(1,2)=0, C(0,2)=0, C(3,2)=3  =>  sum_comb_c = 1+0+0+3 = 4
    // C(3,2)=3, C(3,2)=3 => sum_comb_a = 6
    // C(2,2)=1, C(4,2)=6 => sum_comb_b = 7
    // C(6,2) = 15
    // expected = 6*7/15 = 2.8
    // max_index = (6+7)/2 = 6.5
    // ARI = (4 - 2.8) / (6.5 - 2.8) = 1.2 / 3.7 = 0.324324324...

    let true_l = array![0, 0, 0, 1, 1, 1];
    let pred_l = array![0, 0, 1, 1, 1, 1];
    let ari = adjusted_rand_index(&true_l, &pred_l).unwrap();

    let expected = 1.2 / 3.7;
    assert!(
        (ari - expected).abs() < 1e-10,
        "ARI expected {}, got {}",
        expected,
        ari
    );
}

#[test]
fn test_nmi_analytical_known_value() {
    // Perfect agreement: NMI should be exactly 1.0
    // Complete disagreement: two independent label sets should give low NMI
    let a = array![0, 0, 1, 1];
    let b = array![0, 1, 0, 1];

    let nmi = normalized_mutual_info(&a, &b).unwrap();
    // These labelings share zero information -- NMI should be 0
    // Contingency:
    //       pred0 pred1
    // true0:  1     1
    // true1:  1     1
    // MI = 4 * (1/4) * ln((1/4) / (1/2 * 1/2)) = 4 * (1/4) * ln(1) = 0
    assert!(
        nmi.abs() < 1e-10,
        "NMI for independent labels should be 0, got {}",
        nmi
    );
}

#[test]
fn test_ward_linkage_two_point_merge_distance() {
    // Two points: (0,0) and (3,4)
    // Euclidean distance = 5.0
    // Ward merge distance (display) = sqrt(squared_distance) = 5.0
    let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 3.0, 4.0]).unwrap();

    let mut model = AgglomerativeClustering::new(1).with_linkage(Linkage::Ward);
    model.fit(&x).unwrap();

    let children = model.children().unwrap();
    assert_eq!(children.len(), 1);
    // Display distance should be the Euclidean distance
    assert!(
        (children[0].2 - 5.0).abs() < 1e-10,
        "Ward merge distance for 2 points should be 5.0, got {}",
        children[0].2
    );
}

#[test]
fn test_single_linkage_chaining_property() {
    // Single linkage should exhibit the "chaining" effect:
    // Points in a chain: (0,0), (1,0), (2,0), ..., (5,0), and (20,0)
    // With n_clusters=2, single linkage should put 0-5 in one cluster and 20 alone
    let x = Array2::from_shape_vec(
        (7, 2),
        vec![
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 20.0, 0.0,
        ],
    )
    .unwrap();

    let mut model = AgglomerativeClustering::new(2).with_linkage(Linkage::Single);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    // Points 0-5 should be in one cluster, point 6 in another
    for i in 1..6 {
        assert_eq!(
            labels[0], labels[i],
            "Point {} should be in same cluster as point 0",
            i
        );
    }
    assert_ne!(
        labels[0], labels[6],
        "Point 6 should be in different cluster"
    );
}

#[test]
fn test_single_linkage_merge_distances() {
    // Single linkage: merge distances are minimum pairwise distances
    // Points: (0,0), (1,0), (5,0)
    // Distances: d01=1, d02=5, d12=4
    // Merge 1: (0,1) at distance 1
    // Merge 2: ({0,1}, 2) at distance min(d02,d12) = 4
    let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 5.0, 0.0]).unwrap();

    let mut model = AgglomerativeClustering::new(1).with_linkage(Linkage::Single);
    model.fit(&x).unwrap();

    let children = model.children().unwrap();
    assert_eq!(children.len(), 2);

    // First merge at distance 1.0
    assert!(
        (children[0].2 - 1.0).abs() < 1e-10,
        "First single linkage merge should be at distance 1.0, got {}",
        children[0].2
    );
    // Second merge at distance 4.0
    assert!(
        (children[1].2 - 4.0).abs() < 1e-10,
        "Second single linkage merge should be at distance 4.0, got {}",
        children[1].2
    );
}

#[test]
fn test_complete_linkage_merge_distances() {
    // Complete linkage: merge distances are maximum pairwise distances
    // Points: (0,0), (1,0), (5,0)
    // Distances: d01=1, d02=5, d12=4
    // Merge 1: (0,1) at distance 1
    // Merge 2: ({0,1}, 2) at distance max(d02,d12) = 5
    let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 5.0, 0.0]).unwrap();

    let mut model = AgglomerativeClustering::new(1).with_linkage(Linkage::Complete);
    model.fit(&x).unwrap();

    let children = model.children().unwrap();
    assert_eq!(children.len(), 2);

    assert!(
        (children[0].2 - 1.0).abs() < 1e-10,
        "First complete linkage merge at distance 1.0, got {}",
        children[0].2
    );
    assert!(
        (children[1].2 - 5.0).abs() < 1e-10,
        "Second complete linkage merge at distance 5.0, got {}",
        children[1].2
    );
}

#[test]
fn test_average_linkage_merge_distances() {
    // Average linkage: merge distances are mean pairwise distances
    // Points: (0,0), (1,0), (5,0)
    // Distances: d01=1, d02=5, d12=4
    // Merge 1: (0,1) at distance 1
    // Merge 2: ({0,1}, 2) at distance (d02+d12)/2 = (5+4)/2 = 4.5
    let x = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 5.0, 0.0]).unwrap();

    let mut model = AgglomerativeClustering::new(1).with_linkage(Linkage::Average);
    model.fit(&x).unwrap();

    let children = model.children().unwrap();
    assert_eq!(children.len(), 2);

    assert!(
        (children[0].2 - 1.0).abs() < 1e-10,
        "First average linkage merge at distance 1.0, got {}",
        children[0].2
    );
    assert!(
        (children[1].2 - 4.5).abs() < 1e-10,
        "Second average linkage merge at distance 4.5, got {}",
        children[1].2
    );
}

#[test]
fn test_kmeans_unbalanced_cluster_sizes() {
    // Create clusters of unequal size: 20 points vs 5 points
    let mut data = Vec::new();
    // Cluster 0: 20 points near (0, 0)
    for i in 0..20 {
        data.push((i as f64) * 0.01);
        data.push((i as f64) * 0.01);
    }
    // Cluster 1: 5 points near (10, 10)
    for i in 0..5 {
        data.push(10.0 + (i as f64) * 0.01);
        data.push(10.0 + (i as f64) * 0.01);
    }
    let x = Array2::from_shape_vec((25, 2), data).unwrap();

    let mut kmeans = KMeans::new(2).random_state(42).n_init(5);
    kmeans.fit(&x).unwrap();

    let labels = kmeans.labels().unwrap();
    // First 20 should be in one cluster, last 5 in another
    for i in 1..20 {
        assert_eq!(labels[0], labels[i]);
    }
    for i in 21..25 {
        assert_eq!(labels[20], labels[i]);
    }
    assert_ne!(labels[0], labels[20]);
}

#[test]
fn test_kmeans_n_init_improves_result() {
    // With more n_init, we should get equal or better inertia
    let (x, _) = make_blobs_3c();

    let mut km1 = KMeans::new(3).random_state(42).n_init(1);
    km1.fit(&x).unwrap();
    let inertia_1 = km1.inertia().unwrap();

    let mut km10 = KMeans::new(3).random_state(42).n_init(10);
    km10.fit(&x).unwrap();
    let inertia_10 = km10.inertia().unwrap();

    assert!(
        inertia_10 <= inertia_1 + 1e-6,
        "n_init=10 should have <= inertia than n_init=1: {} > {}",
        inertia_10,
        inertia_1
    );
}

#[test]
fn test_dbscan_circles_data() {
    // Inner ring: points around radius 1
    // Outer ring: points around radius 5
    // The key is that the gap between rings (distance ~4) is much larger than
    // the spacing between adjacent points on each ring.
    let n_per_ring = 20;
    let mut data = Vec::new();
    for i in 0..n_per_ring {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_per_ring as f64;
        data.push(angle.cos());
        data.push(angle.sin());
    }
    for i in 0..n_per_ring {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_per_ring as f64;
        data.push(5.0 * angle.cos());
        data.push(5.0 * angle.sin());
    }
    let x = Array2::from_shape_vec((2 * n_per_ring, 2), data).unwrap();

    // Adjacent points on inner ring (radius 1) are spaced ~2*sin(pi/20) ~ 0.31 apart
    // Adjacent points on outer ring (radius 5) are spaced ~2*5*sin(pi/20) ~ 1.56 apart
    // Gap between rings is ~4.0
    // Use eps=2.0 to connect outer ring points but not bridge the gap
    let mut dbscan = DBSCAN::new(2.0, 3);
    dbscan.fit(&x).unwrap();

    let n_clusters = dbscan.n_clusters().unwrap_or(0);
    assert!(
        n_clusters >= 2,
        "DBSCAN should find at least 2 clusters in concentric circles, got {}",
        n_clusters
    );
}

#[test]
fn test_dbscan_deterministic() {
    // DBSCAN is deterministic (no random initialization)
    let (x, _) = make_blobs_2c();

    let mut d1 = DBSCAN::new(1.0, 2);
    d1.fit(&x).unwrap();
    let labels1 = d1.labels().unwrap().clone();

    let mut d2 = DBSCAN::new(1.0, 2);
    d2.fit(&x).unwrap();
    let labels2 = d2.labels().unwrap().clone();

    assert_eq!(labels1, labels2, "DBSCAN should be deterministic");
}

#[test]
fn test_dbscan_border_point_assignment() {
    // Create a scenario where a border point connects to a core point
    // Core points need min_samples=3 neighbors within eps=1.5
    let x = Array2::from_shape_vec(
        (5, 2),
        vec![
            0.0, 0.0, // core
            0.5, 0.0, // core
            0.0, 0.5, // core
            1.2, 0.0, // border point -- within eps of cluster but not enough neighbors
            10.0, 10.0, // isolated noise
        ],
    )
    .unwrap();

    let mut dbscan = DBSCAN::new(1.5, 3);
    dbscan.fit(&x).unwrap();

    let labels = dbscan.labels().unwrap();
    // First 3 are core, point 3 should be a border point (assigned to cluster 0)
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(
        labels[0], labels[3],
        "Border point should be assigned to nearest cluster"
    );
    assert_eq!(labels[4], -1, "Isolated point should be noise");
}

#[test]
fn test_diagnostics_cluster_sizes_sum_to_n() {
    let (x, labels) = make_blobs_3c();
    let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();

    let total: usize = diag.cluster_sizes.iter().sum();
    assert_eq!(total, x.nrows(), "Cluster sizes should sum to n_samples");
}

#[test]
fn test_diagnostics_compactness_nonnegative() {
    let (x, labels) = make_blobs_3c();
    let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();

    for (i, &c) in diag.compactness.iter().enumerate() {
        assert!(
            c >= 0.0,
            "Compactness of cluster {} should be non-negative, got {}",
            i,
            c
        );
    }
}

#[test]
fn test_diagnostics_separation_nonnegative() {
    let (x, labels) = make_blobs_3c();
    let diag = ClusterDiagnostics::from_labels(&x, &labels, None).unwrap();

    for (i, &s) in diag.separation.iter().enumerate() {
        assert!(
            s >= 0.0,
            "Separation of cluster {} should be non-negative, got {}",
            i,
            s
        );
    }
}

#[test]
fn test_diagnostics_within_ss_equals_inertia() {
    // For KMeans, total_within_ss should equal inertia
    let (x, _) = make_blobs_3c();
    let mut kmeans = KMeans::new(3).random_state(42).n_init(5);
    kmeans.fit(&x).unwrap();

    let labels = kmeans.labels().unwrap();
    let diag = ClusterDiagnostics::from_labels(&x, labels, None).unwrap();

    let inertia = kmeans.inertia().unwrap();
    assert!(
        (diag.total_within_ss - inertia).abs() < 1e-6,
        "Diagnostics total_within_ss ({}) should equal KMeans inertia ({})",
        diag.total_within_ss,
        inertia
    );
}

#[test]
fn test_ward_four_point_analytical_merge_sequence() {
    // Points: A=(0,0), B=(1,0), C=(10,0), D=(11,0)
    // Initial squared distances:
    //   d(A,B)=1, d(A,C)=100, d(A,D)=121, d(B,C)=81, d(B,D)=100, d(C,D)=1
    // Step 1: merge A and B (or C and D) at squared dist 1
    // Step 2: merge C and D at squared dist 1
    // Step 3: merge {A,B} and {C,D}
    let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 11.0, 0.0]).unwrap();

    let mut model = AgglomerativeClustering::new(2).with_linkage(Linkage::Ward);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    // A and B should be together, C and D should be together
    assert_eq!(labels[0], labels[1], "A and B should be in same cluster");
    assert_eq!(labels[2], labels[3], "C and D should be in same cluster");
    assert_ne!(labels[0], labels[2], "Clusters should be different");

    // Check dendrogram distances are non-decreasing
    let mut model_full = AgglomerativeClustering::new(1).with_linkage(Linkage::Ward);
    model_full.fit(&x).unwrap();
    let children = model_full.children().unwrap();

    for i in 1..children.len() {
        assert!(
            children[i].2 >= children[i - 1].2 - 1e-10,
            "Ward distances should be non-decreasing"
        );
    }
}

#[test]
fn test_kmeans_two_samples_two_clusters() {
    // Simplest case: 2 points, 2 clusters
    let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 10.0, 10.0]).unwrap();
    let mut kmeans = KMeans::new(2).random_state(42);
    kmeans.fit(&x).unwrap();

    let labels = kmeans.labels().unwrap();
    assert_ne!(labels[0], labels[1]);

    let centers = kmeans.cluster_centers().unwrap();
    // Each center should be exactly at one of the data points
    let c0 = (centers[[0, 0]], centers[[0, 1]]);
    let c1 = (centers[[1, 0]], centers[[1, 1]]);

    let at_origin = |c: (f64, f64)| c.0.abs() < 1e-10 && c.1.abs() < 1e-10;
    let at_ten = |c: (f64, f64)| (c.0 - 10.0).abs() < 1e-10 && (c.1 - 10.0).abs() < 1e-10;

    assert!(
        (at_origin(c0) && at_ten(c1)) || (at_origin(c1) && at_ten(c0)),
        "Centers should be at the data points"
    );
}

#[test]
fn test_ari_all_same_cluster() {
    // When all are in one cluster vs any other grouping
    // ARI should handle this gracefully
    let a = array![0, 0, 0, 0];
    let b = array![0, 0, 1, 1];

    // This should compute without error
    let ari = adjusted_rand_index(&a, &b).unwrap();
    assert!(
        ari.is_finite(),
        "ARI with one cluster should be finite, got {}",
        ari
    );
}

#[test]
fn test_noise_analysis_no_noise() {
    // When there's no noise, noise_ratio should be 0
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 5.0, 5.0, 5.1, 5.0, 5.0, 5.1],
    )
    .unwrap();

    let mut dbscan = DBSCAN::new(0.5, 2);
    dbscan.fit(&x).unwrap();

    let (noise_ratio, _centroid, _std) = dbscan.noise_analysis(&x).unwrap();
    assert!(
        noise_ratio.abs() < 1e-10,
        "Noise ratio should be 0 when no noise points, got {}",
        noise_ratio
    );
}

#[test]
fn test_noise_analysis_all_noise() {
    // When all points are noise
    let x =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0]).unwrap();

    let mut dbscan = DBSCAN::new(0.001, 3);
    dbscan.fit(&x).unwrap();

    let (noise_ratio, _centroid, _std) = dbscan.noise_analysis(&x).unwrap();
    assert!(
        (noise_ratio - 1.0).abs() < 1e-10,
        "Noise ratio should be 1.0 when all points are noise, got {}",
        noise_ratio
    );
}

#[test]
fn test_kmeans_fit_predict_equals_fit_then_predict() {
    let (x, _) = make_blobs_3c();

    // fit then predict
    let mut km1 = KMeans::new(3).random_state(42).n_init(1);
    km1.fit(&x).unwrap();
    let labels_fp = km1.predict(&x).unwrap();

    // fit_predict
    let mut km2 = KMeans::new(3).random_state(42).n_init(1);
    let labels_combined = km2.fit_predict(&x).unwrap();

    // Should produce same results
    assert_eq!(labels_fp, labels_combined);
}

#[test]
fn test_silhouette_negative_for_bad_clustering() {
    // Deliberately bad clustering: assign nearby points to different clusters
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 10.0, 10.0, 10.1, 10.0, 10.0, 10.1,
        ],
    )
    .unwrap();
    // Swap cluster assignments: points near (0,0) get label 1, points near (10,10) get label 0
    let bad_labels = array![1, 1, 1, 0, 0, 0];

    let score = silhouette_score(&x, &bad_labels).unwrap();
    // This is actually still a valid clustering (just relabeled), silhouette doesn't care about label names
    // It should still be high because the clusters are well-separated
    assert!(
        score > 0.9,
        "Silhouette is label-invariant, should still be high"
    );
}

#[test]
fn test_silhouette_score_poor_clustering() {
    // Create a truly poor clustering by interleaving assignments
    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.1, 0.0, 10.0, 10.0, 10.1, 10.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.1, 10.1, 10.1,
        ],
    )
    .unwrap();
    // Interleave: put one point from each natural cluster into each assigned cluster
    let bad_labels = array![0, 1, 0, 1, 0, 1, 0, 1];

    let score = silhouette_score(&x, &bad_labels).unwrap();
    // Interleaved clustering should have negative or near-zero silhouette
    assert!(
        score < 0.5,
        "Interleaved clustering should have low silhouette, got {}",
        score
    );
}

#[test]
fn test_ward_symmetric_clusters() {
    // Symmetric data should produce symmetric clustering
    // 3 equidistant clusters forming equilateral triangle at scale 100
    let x = Array2::from_shape_vec(
        (9, 2),
        vec![
            // Cluster at (0, 0)
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // Cluster at (100, 0)
            100.0, 0.0, 100.1, 0.0, 100.0, 0.1, // Cluster at (50, 86.6)
            50.0, 86.6, 50.1, 86.6, 50.0, 86.7,
        ],
    )
    .unwrap();

    let mut model = AgglomerativeClustering::new(3).with_linkage(Linkage::Ward);
    model.fit(&x).unwrap();

    let labels = model.labels().unwrap();
    // Each group of 3 should be in its own cluster
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[1], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[4], labels[5]);
    assert_eq!(labels[6], labels[7]);
    assert_eq!(labels[7], labels[8]);
    // All three clusters should be different
    assert_ne!(labels[0], labels[3]);
    assert_ne!(labels[0], labels[6]);
    assert_ne!(labels[3], labels[6]);
}

#[test]
fn test_dbscan_components_match_core_indices() {
    let (x, _) = make_blobs_2c();
    let mut dbscan = DBSCAN::new(1.0, 2);
    dbscan.fit(&x).unwrap();

    let core_indices = dbscan.core_sample_indices().unwrap();
    let components = dbscan.components().unwrap();

    assert_eq!(
        core_indices.len(),
        components.nrows(),
        "Number of core indices should match number of component rows"
    );

    // Each component row should match the corresponding data point
    for (i, &idx) in core_indices.iter().enumerate() {
        for j in 0..x.ncols() {
            assert!(
                (components[[i, j]] - x[[idx, j]]).abs() < 1e-10,
                "Component {} feature {} doesn't match data point {}",
                i,
                j,
                idx
            );
        }
    }
}

#[test]
fn test_all_linkages_produce_valid_labels() {
    let (x, _) = make_blobs_3c();
    let linkages = [
        Linkage::Ward,
        Linkage::Single,
        Linkage::Complete,
        Linkage::Average,
    ];

    for linkage in &linkages {
        let mut model = AgglomerativeClustering::new(3).with_linkage(*linkage);
        model.fit(&x).unwrap();

        let labels = model.labels().unwrap();
        assert_eq!(labels.len(), x.nrows());

        let unique: HashSet<i32> = labels.iter().copied().collect();
        assert_eq!(
            unique.len(),
            3,
            "Linkage {:?} should produce exactly 3 clusters",
            linkage
        );

        // All labels should be 0, 1, or 2
        for &l in labels.iter() {
            assert!(
                (0..3).contains(&l),
                "Invalid label {} for {:?} linkage",
                l,
                linkage
            );
        }
    }
}

#[test]
fn test_calinski_harabasz_improves_with_correct_k() {
    // CH score should be higher for the correct k (3) than for wrong k (2 or 5)
    let (x, _) = make_blobs_3c();

    let mut km3 = KMeans::new(3).random_state(42).n_init(5);
    km3.fit(&x).unwrap();
    let ch3 = calinski_harabasz_score(&x, km3.labels().unwrap()).unwrap();

    let mut km2 = KMeans::new(2).random_state(42).n_init(5);
    km2.fit(&x).unwrap();
    let ch2 = calinski_harabasz_score(&x, km2.labels().unwrap()).unwrap();

    // For well-separated 3 clusters, k=3 should give better CH than k=2
    assert!(
        ch3 > ch2,
        "CH with k=3 ({}) should be higher than k=2 ({}) for 3-cluster data",
        ch3,
        ch2
    );
}
