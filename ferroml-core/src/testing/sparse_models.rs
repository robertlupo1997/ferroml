//! Sparse Model Integration Tests
//!
//! Tests for SparseModel implementations on KNN, NearestCentroid, DBSCAN,
//! and Naive Bayes (all 4 variants).
//! All tests require the `sparse` feature flag.

#![allow(unused_imports)]
#![allow(dead_code)]

#[cfg(feature = "sparse")]
#[cfg(test)]
mod sparse_model_tests {
    use crate::clustering::{ClusteringModel, DBSCAN};
    use crate::models::knn::{
        DistanceMetric, KNNWeights, KNeighborsClassifier, KNeighborsRegressor, NearestCentroid,
    };
    use crate::models::traits::SparseModel;
    use crate::models::Model;
    use crate::sparse::CsrMatrix;
    use ndarray::{Array1, Array2};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const EPSILON: f64 = 1e-10;
    const APPROX_EPSILON: f64 = 1e-6;

    /// Generate random sparse data with controlled sparsity
    fn generate_sparse_data(n_rows: usize, n_cols: usize, sparsity: f64, seed: u64) -> Array2<f64> {
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for i in 0..n_rows {
            for j in 0..n_cols {
                let mut hasher = DefaultHasher::new();
                (seed, i, j).hash(&mut hasher);
                let hash = hasher.finish();
                let frac = (hash % 10000) as f64 / 10000.0;
                if frac < sparsity {
                    data.push(0.0);
                } else {
                    // Generate a value in [0.1, 10.0]
                    let mut hasher2 = DefaultHasher::new();
                    (seed + 1, i, j).hash(&mut hasher2);
                    let val_hash = hasher2.finish();
                    let val = 0.1 + (val_hash % 10000) as f64 / 1000.0;
                    data.push(val);
                }
            }
        }
        Array2::from_shape_vec((n_rows, n_cols), data).unwrap()
    }

    // =========================================================================
    // KNeighborsClassifier Sparse Tests
    // =========================================================================

    #[test]
    fn test_knn_classifier_sparse_basic() {
        // Two well-separated clusters
        let dense = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = KNeighborsClassifier::new(3);
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        for i in 0..3 {
            assert!(
                (predictions[i] - 0.0).abs() < EPSILON,
                "Class 0 point {} misclassified",
                i
            );
        }
        for i in 3..6 {
            assert!(
                (predictions[i] - 1.0).abs() < EPSILON,
                "Class 1 point {} misclassified",
                i
            );
        }
    }

    #[test]
    fn test_knn_classifier_sparse_multiclass() {
        let dense = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // class 0
                5.0, 5.0, 5.1, 5.0, 5.0, 5.1, // class 1
                10.0, 0.0, 10.1, 0.0, 10.0, 0.1, // class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = KNeighborsClassifier::new(3);
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        for i in 0..3 {
            assert!((predictions[i] - 0.0).abs() < EPSILON);
        }
        for i in 3..6 {
            assert!((predictions[i] - 1.0).abs() < EPSILON);
        }
        for i in 6..9 {
            assert!((predictions[i] - 2.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_knn_classifier_sparse_k1_exact() {
        let dense =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = KNeighborsClassifier::new(1);
        clf.fit_sparse(&sparse, &y).unwrap();

        // With k=1, predicting on training data should return exact labels
        let predictions = clf.predict_sparse(&sparse).unwrap();
        for i in 0..4 {
            assert!(
                (predictions[i] - y[i]).abs() < EPSILON,
                "k=1 prediction mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_knn_classifier_sparse_dense_equivalence() {
        let dense = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0, 7.0, 0.0, 8.0, 0.0,
                9.0, 0.0, 8.0, 7.0, 0.0, 7.0, 0.0, 9.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        // Dense prediction
        let mut clf_dense = KNeighborsClassifier::new(3)
            .with_algorithm(crate::models::knn::KNNAlgorithm::BruteForce);
        clf_dense.fit(&dense, &y).unwrap();
        let preds_dense = clf_dense.predict(&dense).unwrap();

        // Sparse prediction
        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf_sparse = KNeighborsClassifier::new(3);
        clf_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = clf_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..dense.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "Dense/sparse mismatch at index {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_knn_classifier_sparse_high_dim() {
        // High-dimensional sparse data (99% zeros)
        let dense = generate_sparse_data(20, 100, 0.99, 42);
        let y = Array1::from_vec((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect());

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = KNeighborsClassifier::new(3);
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 20);
        // Predictions should be valid class labels
        for &p in predictions.iter() {
            assert!(p == 0.0 || p == 1.0, "Invalid prediction: {}", p);
        }
    }

    #[test]
    fn test_knn_classifier_sparse_empty_row() {
        // Matrix where one row is all zeros
        let dense = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 0.0, // point near class 0
                0.0, 1.0, 0.0, // point near class 0
                0.0, 0.0, 0.0, // all-zero row
                5.0, 0.0, 0.0, // point near class 1
                0.0, 5.0, 0.0, // point near class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = KNeighborsClassifier::new(2);
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 5);
        // All-zero row should still get a valid prediction
        assert!(
            predictions[2] == 0.0 || predictions[2] == 1.0,
            "All-zero row should get valid prediction"
        );
    }

    // =========================================================================
    // KNeighborsRegressor Sparse Tests
    // =========================================================================

    #[test]
    fn test_knn_regressor_sparse_basic() {
        let dense = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut reg = KNeighborsRegressor::new(2);
        reg.fit_sparse(&sparse, &y).unwrap();

        let predictions = reg.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 5);

        // Middle point should predict close to 3.0 (average of 2 nearest)
        assert!(
            (predictions[2] - 3.0).abs() < 1.0,
            "Middle point prediction {} too far from expected ~3.0",
            predictions[2]
        );
    }

    #[test]
    fn test_knn_regressor_sparse_dense_equivalence() {
        let dense = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0]);

        let mut reg_dense = KNeighborsRegressor::new(2)
            .with_algorithm(crate::models::knn::KNNAlgorithm::BruteForce);
        reg_dense.fit(&dense, &y).unwrap();
        let preds_dense = reg_dense.predict(&dense).unwrap();

        let sparse = CsrMatrix::from_dense(&dense);
        let mut reg_sparse = KNeighborsRegressor::new(2);
        reg_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = reg_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..dense.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < APPROX_EPSILON,
                "Dense/sparse mismatch at index {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_knn_regressor_sparse_weighted() {
        let dense =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut reg = KNeighborsRegressor::new(4).with_weights(KNNWeights::Distance);
        reg.fit_sparse(&sparse, &y).unwrap();

        let predictions = reg.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 4);

        // With distance weighting and k=4, each point's own label dominates
        // because distance to self is ~0 (weight is very high)
        for i in 0..4 {
            assert!(
                (predictions[i] - y[i]).abs() < 0.5,
                "Distance-weighted prediction {} too far from label {} at index {}",
                predictions[i],
                y[i],
                i
            );
        }
    }

    #[test]
    fn test_knn_regressor_sparse_high_dim() {
        let dense = generate_sparse_data(15, 50, 0.95, 99);
        let y = Array1::from_vec((0..15).map(|i| i as f64).collect());

        let sparse = CsrMatrix::from_dense(&dense);
        let mut reg = KNeighborsRegressor::new(3);
        reg.fit_sparse(&sparse, &y).unwrap();

        let predictions = reg.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 15);

        // Predictions should be within range of target values
        for &p in predictions.iter() {
            assert!(p >= -1.0 && p <= 15.0, "Prediction {} out of range", p);
        }
    }

    // =========================================================================
    // NearestCentroid Sparse Tests
    // =========================================================================

    #[test]
    fn test_nearest_centroid_sparse_basic() {
        let dense = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0, 6.0, 5.0, 5.0, 6.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut nc = NearestCentroid::new();
        nc.fit_sparse(&sparse, &y).unwrap();

        let predictions = nc.predict_sparse(&sparse).unwrap();
        for i in 0..3 {
            assert!((predictions[i] - 0.0).abs() < EPSILON);
        }
        for i in 3..6 {
            assert!((predictions[i] - 1.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_nearest_centroid_sparse_dense_equivalence() {
        let dense = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0, 7.0, 0.0, 8.0, 0.0,
                9.0, 0.0, 8.0, 7.0, 0.0, 7.0, 0.0, 9.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut nc_dense = NearestCentroid::new();
        nc_dense.fit(&dense, &y).unwrap();
        let preds_dense = nc_dense.predict(&dense).unwrap();

        let sparse = CsrMatrix::from_dense(&dense);
        let mut nc_sparse = NearestCentroid::new();
        nc_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = nc_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..dense.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "Dense/sparse NC mismatch at index {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_nearest_centroid_sparse_multiclass() {
        let dense = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, // class 0
                5.0, 5.0, 5.5, 5.0, 5.0, 5.5, 5.5, 5.5, // class 1
                10.0, 0.0, 10.5, 0.0, 10.0, 0.5, 10.5, 0.5, // class 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut nc = NearestCentroid::new();
        nc.fit_sparse(&sparse, &y).unwrap();

        let predictions = nc.predict_sparse(&sparse).unwrap();
        for i in 0..4 {
            assert!((predictions[i] - 0.0).abs() < EPSILON);
        }
        for i in 4..8 {
            assert!((predictions[i] - 1.0).abs() < EPSILON);
        }
        for i in 8..12 {
            assert!((predictions[i] - 2.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_nearest_centroid_sparse_features() {
        // Test with actually sparse features (many zeros)
        let dense = Array2::from_shape_vec(
            (6, 6),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // class 0 - feature 0
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // class 0 - feature 1
                1.0, 1.0, 0.0, 0.0, 0.0, 0.0, // class 0 - features 0,1
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // class 1 - feature 4
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // class 1 - feature 5
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0, // class 1 - features 4,5
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(
            sparse.sparsity() > 0.5,
            "Data should be sparse, got sparsity={}",
            sparse.sparsity()
        );

        let mut nc = NearestCentroid::new();
        nc.fit_sparse(&sparse, &y).unwrap();

        let predictions = nc.predict_sparse(&sparse).unwrap();
        for i in 0..3 {
            assert!((predictions[i] - 0.0).abs() < EPSILON);
        }
        for i in 3..6 {
            assert!((predictions[i] - 1.0).abs() < EPSILON);
        }
    }

    // =========================================================================
    // DBSCAN Sparse Tests
    // =========================================================================

    #[test]
    fn test_dbscan_sparse_basic() {
        let dense = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, // cluster 0
                5.0, 5.0, 5.1, 5.0, 5.0, 5.1, 5.1, 5.1, // cluster 1
            ],
        )
        .unwrap();

        let sparse = CsrMatrix::from_dense(&dense);
        let mut dbscan = DBSCAN::new(0.5, 2);
        let labels = dbscan.fit_sparse(&sparse).unwrap();

        assert_eq!(labels.len(), 8);
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
    fn test_dbscan_sparse_dense_equivalence() {
        let dense = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, // cluster 0
                5.0, 5.0, 5.1, 5.0, 5.0, 5.1, // cluster 1
                20.0, 20.0, // noise
                20.5, 20.5, // noise
                30.0, 30.0, // noise
            ],
        )
        .unwrap();

        // Dense DBSCAN
        let mut dbscan_dense = DBSCAN::new(0.5, 2);
        dbscan_dense.fit(&dense).unwrap();
        let labels_dense = dbscan_dense.labels().unwrap().clone();

        // Sparse DBSCAN
        let sparse = CsrMatrix::from_dense(&dense);
        let mut dbscan_sparse = DBSCAN::new(0.5, 2);
        let labels_sparse = dbscan_sparse.fit_sparse(&sparse).unwrap();

        assert_eq!(labels_dense.len(), labels_sparse.len());
        for i in 0..labels_dense.len() {
            assert_eq!(
                labels_dense[i], labels_sparse[i],
                "Label mismatch at index {}: dense={} vs sparse={}",
                i, labels_dense[i], labels_sparse[i]
            );
        }
    }

    #[test]
    fn test_dbscan_sparse_noise() {
        // Points spread far apart — all noise with small eps
        let dense =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0])
                .unwrap();

        let sparse = CsrMatrix::from_dense(&dense);
        let mut dbscan = DBSCAN::new(1.0, 2);
        let labels = dbscan.fit_sparse(&sparse).unwrap();

        // All points should be noise
        for &l in labels.iter() {
            assert_eq!(l, -1, "Expected noise label -1, got {}", l);
        }
        assert_eq!(dbscan.n_clusters(), Some(0));
    }

    #[test]
    fn test_dbscan_sparse_high_dim() {
        // High-dimensional sparse data
        let mut dense = Array2::zeros((12, 50));
        // Cluster 0: first 4 points close in feature 0-1
        for i in 0..4 {
            dense[[i, 0]] = 1.0 + (i as f64) * 0.05;
            dense[[i, 1]] = 1.0 + (i as f64) * 0.05;
        }
        // Cluster 1: next 4 points close in feature 10-11
        for i in 4..8 {
            dense[[i, 10]] = 1.0 + ((i - 4) as f64) * 0.05;
            dense[[i, 11]] = 1.0 + ((i - 4) as f64) * 0.05;
        }
        // Noise: remaining 4 points scattered
        dense[[8, 25]] = 50.0;
        dense[[9, 30]] = 50.0;
        dense[[10, 35]] = 50.0;
        dense[[11, 40]] = 50.0;

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(sparse.sparsity() > 0.9);

        let mut dbscan = DBSCAN::new(0.5, 2);
        let labels = dbscan.fit_sparse(&sparse).unwrap();

        assert_eq!(labels.len(), 12);

        // Cluster 0 points should share a label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[2], labels[3]);
        assert!(labels[0] >= 0, "Cluster 0 points should not be noise");

        // Cluster 1 points should share a different label
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[6], labels[7]);
        assert!(labels[4] >= 0, "Cluster 1 points should not be noise");

        assert_ne!(labels[0], labels[4], "Clusters should be distinct");

        // Noise points should be labeled -1
        for i in 8..12 {
            assert_eq!(labels[i], -1, "Scattered point {} should be noise", i);
        }
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_sparse_all_zero_query_row() {
        // Training data with non-zero values
        let train = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0]);

        let train_sparse = CsrMatrix::from_dense(&train);
        let mut clf = KNeighborsClassifier::new(1);
        clf.fit_sparse(&train_sparse, &y).unwrap();

        // Query with all-zero row
        let query = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let query_sparse = CsrMatrix::from_dense(&query);
        let pred = clf.predict_sparse(&query_sparse).unwrap();

        assert_eq!(pred.len(), 1);
        // Should predict something valid (closest to origin)
        assert!(
            pred[0] == 0.0 || pred[0] == 1.0 || pred[0] == 2.0 || pred[0] == 3.0,
            "Should predict valid class, got {}",
            pred[0]
        );
    }

    #[test]
    fn test_sparse_single_feature_matrix() {
        // 1D features
        let dense = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);

        // Test KNN
        let mut clf = KNeighborsClassifier::new(3);
        clf.fit_sparse(&sparse, &y).unwrap();
        let predictions = clf.predict_sparse(&sparse).unwrap();

        for i in 0..3 {
            assert!((predictions[i] - 0.0).abs() < EPSILON);
        }
        for i in 3..6 {
            assert!((predictions[i] - 1.0).abs() < EPSILON);
        }

        // Test NearestCentroid
        let mut nc = NearestCentroid::new();
        nc.fit_sparse(&sparse, &y).unwrap();
        let nc_preds = nc.predict_sparse(&sparse).unwrap();

        for i in 0..3 {
            assert!((nc_preds[i] - 0.0).abs() < EPSILON);
        }
        for i in 3..6 {
            assert!((nc_preds[i] - 1.0).abs() < EPSILON);
        }
    }

    // =========================================================================
    // MultinomialNB Sparse Tests
    // =========================================================================

    #[test]
    fn test_multinomial_nb_sparse_basic() {
        use crate::models::naive_bayes::MultinomialNB;

        // Word count features (non-negative)
        let dense = Array2::from_shape_vec(
            (6, 4),
            vec![
                5.0, 1.0, 0.0, 0.0, // class 0
                4.0, 2.0, 0.0, 0.0, // class 0
                3.0, 3.0, 0.0, 0.0, // class 0
                0.0, 0.0, 5.0, 1.0, // class 1
                0.0, 0.0, 4.0, 2.0, // class 1
                0.0, 0.0, 3.0, 3.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = MultinomialNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 6);
        for i in 0..3 {
            assert!(
                (predictions[i] - 0.0).abs() < EPSILON,
                "MultinomialNB class 0 misclassified at {}",
                i
            );
        }
        for i in 3..6 {
            assert!(
                (predictions[i] - 1.0).abs() < EPSILON,
                "MultinomialNB class 1 misclassified at {}",
                i
            );
        }
    }

    #[test]
    fn test_multinomial_nb_sparse_text_data() {
        use crate::models::naive_bayes::MultinomialNB;

        // High-dimensional sparse count data (simulating bag-of-words)
        let n_samples = 200;
        let n_features = 500;
        let mut dense = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class = if i < n_samples / 2 { 0.0 } else { 1.0 };
            y[i] = class;

            // Class 0: features 0-49 have counts, rest sparse
            // Class 1: features 50-99 have counts, rest sparse
            let offset = if class == 0.0 { 0 } else { 50 };
            for j in 0..10 {
                let mut hasher = DefaultHasher::new();
                (42u64, i, j).hash(&mut hasher);
                let idx = offset + (hasher.finish() as usize % 50);
                let mut hasher2 = DefaultHasher::new();
                (43u64, i, j).hash(&mut hasher2);
                let count = 1.0 + (hasher2.finish() % 5) as f64;
                dense[[i, idx]] += count;
            }
        }

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(
            sparse.sparsity() > 0.9,
            "Text data should be >90% sparse, got {:.1}%",
            sparse.sparsity() * 100.0
        );

        let mut clf = MultinomialNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        // Count accuracy
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < EPSILON)
            .count();
        let accuracy = correct as f64 / n_samples as f64;
        assert!(
            accuracy > 0.7,
            "MultinomialNB text accuracy should be >70%, got {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_multinomial_nb_sparse_dense_equivalence() {
        use crate::models::naive_bayes::MultinomialNB;

        let dense = Array2::from_shape_vec(
            (8, 5),
            vec![
                3.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 1.0,
                2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0,
                0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        // Dense
        let mut clf_dense = MultinomialNB::new();
        clf_dense.fit(&dense, &y).unwrap();
        let preds_dense = clf_dense.predict(&dense).unwrap();

        // Sparse
        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf_sparse = MultinomialNB::new();
        clf_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = clf_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..dense.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "MultinomialNB dense/sparse mismatch at {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_multinomial_nb_sparse_imbalanced() {
        use crate::models::naive_bayes::MultinomialNB;

        // Heavily imbalanced: 5 samples class 0, 1 sample class 1
        let dense = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
                0.0, 2.0, // class 1 has high feature 2
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = MultinomialNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 6);
        for &p in predictions.iter() {
            assert!(p == 0.0 || p == 1.0, "Should be valid class, got {}", p);
        }
    }

    // =========================================================================
    // BernoulliNB Sparse Tests
    // =========================================================================

    #[test]
    fn test_bernoulli_nb_sparse_basic() {
        use crate::models::naive_bayes::BernoulliNB;

        // Binary features
        let dense = Array2::from_shape_vec(
            (6, 4),
            vec![
                1.0, 1.0, 0.0, 0.0, // class 0
                1.0, 0.0, 0.0, 0.0, // class 0
                0.0, 1.0, 0.0, 0.0, // class 0
                0.0, 0.0, 1.0, 1.0, // class 1
                0.0, 0.0, 1.0, 0.0, // class 1
                0.0, 0.0, 0.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = BernoulliNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 6);
        for i in 0..3 {
            assert!(
                (predictions[i] - 0.0).abs() < EPSILON,
                "BernoulliNB class 0 misclassified at {}",
                i
            );
        }
        for i in 3..6 {
            assert!(
                (predictions[i] - 1.0).abs() < EPSILON,
                "BernoulliNB class 1 misclassified at {}",
                i
            );
        }
    }

    #[test]
    fn test_bernoulli_nb_sparse_text_data() {
        use crate::models::naive_bayes::BernoulliNB;

        // High-dimensional binary features (simulating term presence)
        let n_samples = 200;
        let n_features = 500;
        let mut dense = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class = if i < n_samples / 2 { 0.0 } else { 1.0 };
            y[i] = class;

            // Class 0: features 0-49 present, rest absent
            // Class 1: features 50-99 present, rest absent
            let offset = if class == 0.0 { 0 } else { 50 };
            for j in 0..8 {
                let mut hasher = DefaultHasher::new();
                (44u64, i, j).hash(&mut hasher);
                let idx = offset + (hasher.finish() as usize % 50);
                dense[[i, idx]] = 1.0;
            }
        }

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(
            sparse.sparsity() > 0.9,
            "Binary text data should be >90% sparse"
        );

        let mut clf = BernoulliNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < EPSILON)
            .count();
        let accuracy = correct as f64 / n_samples as f64;
        assert!(
            accuracy > 0.7,
            "BernoulliNB text accuracy should be >70%, got {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_bernoulli_nb_sparse_dense_equivalence() {
        use crate::models::naive_bayes::BernoulliNB;

        let dense = Array2::from_shape_vec(
            (8, 5),
            vec![
                1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf_dense = BernoulliNB::new();
        clf_dense.fit(&dense, &y).unwrap();
        let preds_dense = clf_dense.predict(&dense).unwrap();

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf_sparse = BernoulliNB::new();
        clf_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = clf_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..dense.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "BernoulliNB dense/sparse mismatch at {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_bernoulli_nb_sparse_very_sparse() {
        use crate::models::naive_bayes::BernoulliNB;

        // Very sparse: only 1 feature per sample in 20 features
        let dense = Array2::from_shape_vec(
            (6, 20),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(
            sparse.sparsity() > 0.9,
            "Should be >90% sparse, got {:.1}%",
            sparse.sparsity() * 100.0
        );

        let mut clf = BernoulliNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 6);
        for &p in predictions.iter() {
            assert!(p == 0.0 || p == 1.0, "Invalid prediction: {}", p);
        }
    }

    // =========================================================================
    // GaussianNB Sparse Tests
    // =========================================================================

    #[test]
    fn test_gaussian_nb_sparse_basic() {
        use crate::models::naive_bayes::GaussianNB;

        let dense = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 2.0, 1.0, 1.5, 1.5, // class 0
                7.0, 8.0, 8.0, 7.0, 7.5, 7.5, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = GaussianNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 6);
        for i in 0..3 {
            assert!(
                (predictions[i] - 0.0).abs() < EPSILON,
                "GaussianNB class 0 misclassified at {}",
                i
            );
        }
        for i in 3..6 {
            assert!(
                (predictions[i] - 1.0).abs() < EPSILON,
                "GaussianNB class 1 misclassified at {}",
                i
            );
        }
    }

    #[test]
    fn test_gaussian_nb_sparse_real_valued() {
        use crate::models::naive_bayes::GaussianNB;

        // Real-valued features including negatives
        let dense = Array2::from_shape_vec(
            (8, 3),
            vec![
                -2.0, 0.0, 1.0, -1.5, 0.0, 0.5, -1.0, 0.0, 1.5, -2.5, 0.0, 0.8, 3.0, 0.0, -1.0,
                2.5, 0.0, -0.5, 3.5, 0.0, -1.5, 2.0, 0.0, -0.8,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = GaussianNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        for i in 0..4 {
            assert!(
                (predictions[i] - 0.0).abs() < EPSILON,
                "GaussianNB class 0 misclassified at {}",
                i
            );
        }
        for i in 4..8 {
            assert!(
                (predictions[i] - 1.0).abs() < EPSILON,
                "GaussianNB class 1 misclassified at {}",
                i
            );
        }
    }

    #[test]
    fn test_gaussian_nb_sparse_dense_equivalence() {
        use crate::models::naive_bayes::GaussianNB;

        let dense = Array2::from_shape_vec(
            (8, 4),
            vec![
                1.0, 0.0, 0.5, 0.0, 0.0, 2.0, 0.0, 1.0, 1.5, 0.0, 0.3, 0.0, 0.0, 1.8, 0.0, 0.9,
                5.0, 0.0, 4.5, 0.0, 0.0, 6.0, 0.0, 5.0, 5.5, 0.0, 4.3, 0.0, 0.0, 5.8, 0.0, 4.9,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf_dense = GaussianNB::new();
        clf_dense.fit(&dense, &y).unwrap();
        let preds_dense = clf_dense.predict(&dense).unwrap();

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf_sparse = GaussianNB::new();
        clf_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = clf_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..dense.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "GaussianNB dense/sparse mismatch at {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_gaussian_nb_sparse_imbalanced() {
        use crate::models::naive_bayes::GaussianNB;

        // Heavily imbalanced: 5 class 0 vs 1 class 1
        let dense = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 10.0,
                10.0, 10.0, // class 1 outlier
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = GaussianNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 6);
        for &p in predictions.iter() {
            assert!(p == 0.0 || p == 1.0, "Should be valid class, got {}", p);
        }
        // The majority class samples should mostly predict 0.0
        let class0_correct: usize = (0..5)
            .filter(|&i| (predictions[i] - 0.0).abs() < EPSILON)
            .count();
        assert!(
            class0_correct >= 3,
            "At least 3/5 class 0 should be correct, got {}",
            class0_correct
        );
    }

    // =========================================================================
    // CategoricalNB Sparse Tests
    // =========================================================================

    #[test]
    fn test_categorical_nb_sparse_basic() {
        use crate::models::naive_bayes::CategoricalNB;

        // Categorical features (non-negative integers as floats)
        let dense = Array2::from_shape_vec(
            (6, 3),
            vec![
                0.0, 0.0, 1.0, // class 0
                0.0, 1.0, 1.0, // class 0
                1.0, 0.0, 1.0, // class 0
                2.0, 2.0, 0.0, // class 1
                2.0, 1.0, 0.0, // class 1
                1.0, 2.0, 0.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = CategoricalNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 6);
        for i in 0..3 {
            assert!(
                (predictions[i] - 0.0).abs() < EPSILON,
                "CategoricalNB class 0 misclassified at {}",
                i
            );
        }
        for i in 3..6 {
            assert!(
                (predictions[i] - 1.0).abs() < EPSILON,
                "CategoricalNB class 1 misclassified at {}",
                i
            );
        }
    }

    #[test]
    fn test_categorical_nb_sparse_high_dim() {
        use crate::models::naive_bayes::CategoricalNB;

        // Higher dimensional with sparse structure (many 0 categories)
        let n_samples = 40;
        let n_features = 20;
        let mut dense = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let class = if i < n_samples / 2 { 0.0 } else { 1.0 };
            y[i] = class;

            // Class 0: first few features have category 1 or 2
            // Class 1: last few features have category 1 or 2
            let offset = if class == 0.0 { 0 } else { 10 };
            for j in 0..3 {
                let mut hasher = DefaultHasher::new();
                (45u64, i, j).hash(&mut hasher);
                let feat_idx = offset + (hasher.finish() as usize % 10);
                let mut hasher2 = DefaultHasher::new();
                (46u64, i, j).hash(&mut hasher2);
                let cat = 1.0 + (hasher2.finish() % 2) as f64;
                dense[[i, feat_idx]] = cat;
            }
        }

        let sparse = CsrMatrix::from_dense(&dense);

        let mut clf = CategoricalNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), n_samples);

        let correct: usize = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&p, &t)| (p - t).abs() < EPSILON)
            .count();
        let accuracy = correct as f64 / n_samples as f64;
        assert!(
            accuracy > 0.6,
            "CategoricalNB accuracy should be >60%, got {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_categorical_nb_sparse_dense_equivalence() {
        use crate::models::naive_bayes::CategoricalNB;

        let dense = Array2::from_shape_vec(
            (8, 4),
            vec![
                0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 2.0,
                2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 0.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut clf_dense = CategoricalNB::new();
        clf_dense.fit(&dense, &y).unwrap();
        let preds_dense = clf_dense.predict(&dense).unwrap();

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf_sparse = CategoricalNB::new();
        clf_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = clf_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..dense.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "CategoricalNB dense/sparse mismatch at {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_categorical_nb_sparse_imbalanced() {
        use crate::models::naive_bayes::CategoricalNB;

        // Heavily imbalanced: 4 class 0 vs 2 class 1
        let dense = Array2::from_shape_vec(
            (6, 3),
            vec![
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0,
                2.0, // class 1
                2.0, 2.0, 0.0, // class 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);
        let mut clf = CategoricalNB::new();
        clf.fit_sparse(&sparse, &y).unwrap();

        let predictions = clf.predict_sparse(&sparse).unwrap();
        assert_eq!(predictions.len(), 6);
        for &p in predictions.iter() {
            assert!(p == 0.0 || p == 1.0, "Should be valid class, got {}", p);
        }
    }
}

// =============================================================================
// Sparse Linear Model Tests (Q.7)
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod sparse_linear_model_tests {
    use crate::models::logistic::LogisticRegression;
    use crate::models::regularized::RidgeRegression;
    use crate::models::svm::{LinearSVC, LinearSVR};
    use crate::models::traits::SparseModel;
    use crate::models::Model;
    use crate::sparse::CsrMatrix;
    use ndarray::{Array1, Array2};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const EPSILON: f64 = 1e-10;

    /// Generate random sparse data with controlled sparsity
    fn generate_sparse_data(n_rows: usize, n_cols: usize, sparsity: f64, seed: u64) -> Array2<f64> {
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for i in 0..n_rows {
            for j in 0..n_cols {
                let mut hasher = DefaultHasher::new();
                (seed, i, j).hash(&mut hasher);
                let hash = hasher.finish();
                let frac = (hash % 10000) as f64 / 10000.0;
                if frac < sparsity {
                    data.push(0.0);
                } else {
                    let mut hasher2 = DefaultHasher::new();
                    (seed + 1, i, j).hash(&mut hasher2);
                    let val_hash = hasher2.finish();
                    let val = 0.1 + (val_hash % 10000) as f64 / 1000.0;
                    data.push(val);
                }
            }
        }
        Array2::from_shape_vec((n_rows, n_cols), data).unwrap()
    }

    /// Generate linearly separable classification data
    fn make_classification_data(
        n_samples: usize,
        n_features: usize,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let half = n_samples / 2;
        let mut data = Vec::with_capacity(n_samples * n_features);
        let mut labels = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let class = if i < half { 0.0 } else { 1.0 };
            labels.push(class);
            for j in 0..n_features {
                let mut hasher = DefaultHasher::new();
                (seed, i, j).hash(&mut hasher);
                let hash = hasher.finish();
                let base = if i < half { 0.0 } else { 3.0 };
                let val = base + (hash % 1000) as f64 / 1000.0;
                data.push(val);
            }
        }
        (
            Array2::from_shape_vec((n_samples, n_features), data).unwrap(),
            Array1::from_vec(labels),
        )
    }

    /// Generate regression data: y = sum(x) + noise
    fn make_regression_data(
        n_samples: usize,
        n_features: usize,
        seed: u64,
    ) -> (Array2<f64>, Array1<f64>) {
        let mut data = Vec::with_capacity(n_samples * n_features);
        let mut targets = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut row_sum = 0.0;
            for j in 0..n_features {
                let mut hasher = DefaultHasher::new();
                (seed, i, j).hash(&mut hasher);
                let hash = hasher.finish();
                let val = (hash % 10000) as f64 / 1000.0;
                data.push(val);
                row_sum += val;
            }
            let mut hasher = DefaultHasher::new();
            (seed + 999, i).hash(&mut hasher);
            let noise = (hasher.finish() % 1000) as f64 / 10000.0;
            targets.push(row_sum + noise);
        }
        (
            Array2::from_shape_vec((n_samples, n_features), data).unwrap(),
            Array1::from_vec(targets),
        )
    }

    // =========================================================================
    // LinearSVC Sparse Tests
    // =========================================================================

    #[test]
    fn test_linear_svc_sparse_basic() {
        let (x, y) = make_classification_data(40, 3, 100);
        let sparse = CsrMatrix::from_dense(&x);

        let mut clf = LinearSVC::new();
        clf.fit_sparse(&sparse, &y).unwrap();
        let preds = clf.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 40);
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &a)| (p - a).abs() < EPSILON)
            .count();
        assert!(
            correct >= 30,
            "LinearSVC sparse should classify >= 30/40, got {}",
            correct
        );
    }

    #[test]
    fn test_linear_svc_sparse_dense_equivalence() {
        let (x, y) = make_classification_data(30, 4, 200);
        let sparse = CsrMatrix::from_dense(&x);

        let mut clf_dense = LinearSVC::new();
        clf_dense.fit(&x, &y).unwrap();
        let preds_dense = clf_dense.predict(&x).unwrap();

        let mut clf_sparse = LinearSVC::new();
        clf_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = clf_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "LinearSVC dense/sparse mismatch at {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_linear_svc_sparse_high_dim() {
        let mut dense = generate_sparse_data(50, 10000, 0.99, 300);
        let y: Array1<f64> =
            Array1::from_vec((0..50).map(|i| if i < 25 { 0.0 } else { 1.0 }).collect());
        // Make classes separable by adding signal to first feature
        for i in 0..50 {
            dense[[i, 0]] = if i < 25 { -5.0 } else { 5.0 };
        }

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(sparse.sparsity() > 0.9);

        let mut clf = LinearSVC::new().with_c(0.1);
        clf.fit_sparse(&sparse, &y).unwrap();
        let preds = clf.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 50);
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0, "Invalid prediction: {}", p);
        }
    }

    #[test]
    fn test_linear_svc_sparse_edge_case_very_sparse() {
        let mut dense = Array2::zeros((20, 50));
        let y: Array1<f64> =
            Array1::from_vec((0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect());

        // Class 0 has signal in feature 0
        for i in 0..10 {
            dense[[i, 0]] = 1.0 + i as f64 * 0.1;
        }
        // Class 1 has signal in feature 1
        for i in 10..20 {
            dense[[i, 1]] = 1.0 + (i - 10) as f64 * 0.1;
        }

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(sparse.sparsity() > 0.95);

        let mut clf = LinearSVC::new().with_c(10.0);
        clf.fit_sparse(&sparse, &y).unwrap();
        let preds = clf.predict_sparse(&sparse).unwrap();

        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &a)| (p - a).abs() < EPSILON)
            .count();
        assert!(
            correct >= 15,
            "Very sparse LinearSVC should classify >= 15/20, got {}",
            correct
        );
    }

    // =========================================================================
    // LinearSVR Sparse Tests
    // =========================================================================

    #[test]
    fn test_linear_svr_sparse_basic() {
        let (x, y) = make_regression_data(30, 3, 400);
        let sparse = CsrMatrix::from_dense(&x);

        let mut reg = LinearSVR::new();
        reg.fit_sparse(&sparse, &y).unwrap();
        let preds = reg.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 30);
        let y_mean = y.mean().unwrap();
        for &p in preds.iter() {
            assert!(
                (p - y_mean).abs() < y_mean * 10.0,
                "SVR prediction {} way off from mean {}",
                p,
                y_mean
            );
        }
    }

    #[test]
    fn test_linear_svr_sparse_dense_equivalence() {
        let (x, y) = make_regression_data(25, 4, 500);
        let sparse = CsrMatrix::from_dense(&x);

        let mut reg_dense = LinearSVR::new();
        reg_dense.fit(&x, &y).unwrap();
        let preds_dense = reg_dense.predict(&x).unwrap();

        let mut reg_sparse = LinearSVR::new();
        reg_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = reg_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "LinearSVR dense/sparse mismatch at {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_linear_svr_sparse_high_dim() {
        let dense = generate_sparse_data(40, 10000, 0.99, 600);
        let mut y_vals = Vec::with_capacity(40);
        for i in 0..40 {
            let signal = dense[[i, 0]] + dense[[i, 1]] * 2.0;
            y_vals.push(signal);
        }
        let y = Array1::from_vec(y_vals);

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(sparse.sparsity() > 0.9);

        let mut reg = LinearSVR::new();
        reg.fit_sparse(&sparse, &y).unwrap();
        let preds = reg.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 40);
        for &p in preds.iter() {
            assert!(p.is_finite(), "SVR prediction should be finite, got {}", p);
        }
    }

    #[test]
    fn test_linear_svr_sparse_edge_case_very_sparse() {
        let mut dense = Array2::zeros((20, 100));
        let mut y_vals = Vec::with_capacity(20);

        for i in 0..20 {
            dense[[i, i % 5]] = (i as f64 + 1.0) * 0.5;
            y_vals.push(dense[[i, i % 5]] * 2.0 + 1.0);
        }
        let y = Array1::from_vec(y_vals);

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(sparse.sparsity() > 0.95);

        let mut reg = LinearSVR::new();
        reg.fit_sparse(&sparse, &y).unwrap();
        let preds = reg.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 20);
        for &p in preds.iter() {
            assert!(p.is_finite(), "Prediction should be finite");
        }
    }

    // =========================================================================
    // LogisticRegression Sparse Tests
    // =========================================================================

    #[test]
    fn test_logistic_regression_sparse_basic() {
        let (x, y) = make_classification_data(40, 3, 700);
        let sparse = CsrMatrix::from_dense(&x);

        let mut clf = LogisticRegression::default();
        clf.fit_sparse(&sparse, &y).unwrap();
        let preds = clf.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 40);
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &a)| (p - a).abs() < EPSILON)
            .count();
        assert!(
            correct >= 30,
            "LogReg sparse should classify >= 30/40, got {}",
            correct
        );
    }

    #[test]
    fn test_logistic_regression_sparse_dense_equivalence() {
        let (x, y) = make_classification_data(30, 4, 800);
        let sparse = CsrMatrix::from_dense(&x);

        let mut clf_dense = LogisticRegression::default();
        clf_dense.fit(&x, &y).unwrap();
        let preds_dense = clf_dense.predict(&x).unwrap();

        let mut clf_sparse = LogisticRegression::default();
        clf_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = clf_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "LogReg dense/sparse mismatch at {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_logistic_regression_sparse_high_dim() {
        // Use 200 features because LogReg IRLS involves n_features x n_features matrix ops
        let mut dense = generate_sparse_data(60, 200, 0.90, 900);
        let y: Array1<f64> =
            Array1::from_vec((0..60).map(|i| if i < 30 { 0.0 } else { 1.0 }).collect());
        // Add separability signal
        for i in 0..60 {
            dense[[i, 0]] = if i < 30 { -3.0 } else { 3.0 };
        }

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(sparse.sparsity() > 0.9);

        let mut clf = LogisticRegression::default();
        clf.fit_sparse(&sparse, &y).unwrap();
        let preds = clf.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 60);
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0, "Invalid prediction: {}", p);
        }
    }

    #[test]
    fn test_logistic_regression_sparse_edge_case() {
        let mut dense = Array2::zeros((30, 40));
        let y: Array1<f64> =
            Array1::from_vec((0..30).map(|i| if i < 15 { 0.0 } else { 1.0 }).collect());

        for i in 0..15 {
            dense[[i, 0]] = 2.0;
            dense[[i, 1]] = 1.0;
        }
        for i in 15..30 {
            dense[[i, 0]] = -2.0;
            dense[[i, 1]] = -1.0;
        }

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(sparse.sparsity() > 0.9);

        let mut clf = LogisticRegression::default();
        clf.fit_sparse(&sparse, &y).unwrap();
        let preds = clf.predict_sparse(&sparse).unwrap();

        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(&p, &a)| (p - a).abs() < EPSILON)
            .count();
        assert!(
            correct >= 25,
            "LogReg edge case should classify >= 25/30, got {}",
            correct
        );
    }

    // =========================================================================
    // RidgeRegression Sparse Tests
    // =========================================================================

    #[test]
    fn test_ridge_sparse_basic() {
        let (x, y) = make_regression_data(30, 3, 1000);
        let sparse = CsrMatrix::from_dense(&x);

        let mut reg = RidgeRegression::new(1.0);
        reg.fit_sparse(&sparse, &y).unwrap();
        let preds = reg.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 30);
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = preds
            .iter()
            .zip(y.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        assert!(r2 > 0.0, "Ridge R^2 should be > 0, got {}", r2);
    }

    #[test]
    fn test_ridge_sparse_dense_equivalence() {
        let (x, y) = make_regression_data(25, 4, 1100);
        let sparse = CsrMatrix::from_dense(&x);

        let mut reg_dense = RidgeRegression::new(1.0);
        reg_dense.fit(&x, &y).unwrap();
        let preds_dense = reg_dense.predict(&x).unwrap();

        let mut reg_sparse = RidgeRegression::new(1.0);
        reg_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = reg_sparse.predict_sparse(&sparse).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (preds_dense[i] - preds_sparse[i]).abs() < EPSILON,
                "Ridge dense/sparse mismatch at {}: {} vs {}",
                i,
                preds_dense[i],
                preds_sparse[i]
            );
        }
    }

    #[test]
    fn test_ridge_sparse_high_dim() {
        // Use 200 features because Ridge involves n_features x n_features matrix solve
        let dense = generate_sparse_data(40, 200, 0.90, 1200);
        let mut y_vals = Vec::with_capacity(40);
        for i in 0..40 {
            y_vals.push(dense[[i, 0]] * 3.0 + dense[[i, 1]] + 1.0);
        }
        let y = Array1::from_vec(y_vals);

        let sparse = CsrMatrix::from_dense(&dense);
        assert!(sparse.sparsity() > 0.8);

        let mut reg = RidgeRegression::new(10.0);
        reg.fit_sparse(&sparse, &y).unwrap();
        let preds = reg.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 40);
        for &p in preds.iter() {
            assert!(
                p.is_finite(),
                "Ridge prediction should be finite, got {}",
                p
            );
        }
    }

    #[test]
    fn test_ridge_sparse_edge_case_strong_regularization() {
        let mut dense = Array2::zeros((20, 50));
        let mut y_vals = Vec::with_capacity(20);

        for i in 0..20 {
            dense[[i, 0]] = (i as f64) * 0.5;
            y_vals.push(dense[[i, 0]] * 2.0 + 3.0);
        }
        let y = Array1::from_vec(y_vals);

        let sparse = CsrMatrix::from_dense(&dense);

        // Strong regularization
        let mut reg_strong = RidgeRegression::new(1000.0);
        reg_strong.fit_sparse(&sparse, &y).unwrap();
        let preds_strong = reg_strong.predict_sparse(&sparse).unwrap();

        // Weak regularization
        let mut reg_weak = RidgeRegression::new(0.001);
        reg_weak.fit_sparse(&sparse, &y).unwrap();
        let preds_weak = reg_weak.predict_sparse(&sparse).unwrap();

        // Weak regularization should fit better
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res_strong: f64 = preds_strong
            .iter()
            .zip(y.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum();
        let ss_res_weak: f64 = preds_weak
            .iter()
            .zip(y.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum();

        let r2_strong = 1.0 - ss_res_strong / ss_tot;
        let r2_weak = 1.0 - ss_res_weak / ss_tot;

        assert!(
            r2_weak > r2_strong,
            "Weak regularization should fit better: R2_weak={} vs R2_strong={}",
            r2_weak,
            r2_strong
        );
    }

    // =========================================================================
    // CsrMatrix utility method tests
    // =========================================================================

    #[test]
    fn test_csr_transpose_dot() {
        let dense = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = sparse.transpose_dot(&y);
        // X^T @ y: [1*1+3*2+5*3, 2*1+4*2+6*3] = [22, 28]
        assert!((result[0] - 22.0).abs() < EPSILON);
        assert!((result[1] - 28.0).abs() < EPSILON);
    }

    #[test]
    fn test_csr_gram_matrix() {
        let dense = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let gram = sparse.gram_matrix();
        let expected = dense.t().dot(&dense);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (gram[[i, j]] - expected[[i, j]]).abs() < EPSILON,
                    "Gram mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    gram[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_csr_weighted_gram() {
        let dense = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);
        let w = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let wgram = sparse.weighted_gram(&w);
        let weighted = &dense * &w.clone().insert_axis(ndarray::Axis(1));
        let expected = dense.t().dot(&weighted);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (wgram[[i, j]] - expected[[i, j]]).abs() < EPSILON,
                    "Weighted Gram mismatch at [{},{}]: {} vs {}",
                    i,
                    j,
                    wgram[[i, j]],
                    expected[[i, j]]
                );
            }
        }
    }
}
