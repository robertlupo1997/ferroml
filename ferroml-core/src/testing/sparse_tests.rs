//! Sparse Data Support Tests
//!
//! This module provides comprehensive tests for sparse matrix functionality:
//! - CSR/CSC matrix creation and manipulation
//! - Sparse/dense equivalence tests
//! - Sparse distance calculations
//! - Memory efficiency validation
//! - Edge case handling for sparse data
//!
//! Enable sparse functionality with the `sparse` feature flag.

use ndarray::{Array1, Array2};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// =============================================================================
// Test Utilities
// =============================================================================

/// Generate random sparse data with controlled sparsity
fn generate_sparse_data(n_rows: usize, n_cols: usize, sparsity: f64, seed: u64) -> Array2<f64> {
    let mut data = Vec::with_capacity(n_rows * n_cols);

    for i in 0..n_rows {
        for j in 0..n_cols {
            let mut hasher = DefaultHasher::new();
            (seed, i, j, "sparse").hash(&mut hasher);
            let h = hasher.finish();
            let random = (h as f64 / u64::MAX as f64);

            // Only create non-zero values if random > sparsity
            if random > sparsity {
                let mut hasher2 = DefaultHasher::new();
                (seed, i, j, "value").hash(&mut hasher2);
                let h2 = hasher2.finish();
                let val = (h2 as f64 / u64::MAX as f64) * 10.0 - 5.0;
                data.push(val);
            } else {
                data.push(0.0);
            }
        }
    }

    Array2::from_shape_vec((n_rows, n_cols), data).expect("Failed to create sparse data")
}

/// Generate highly sparse text-like data (typical for NLP)
fn generate_text_like_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    // Text data is typically >99% sparse
    generate_sparse_data(n_samples, n_features, 0.99, seed)
}

/// Generate moderately sparse data (typical for recommendation systems)
fn generate_moderate_sparse_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    // Recommendation data is typically 90-95% sparse
    generate_sparse_data(n_samples, n_features, 0.90, seed)
}

/// Generate dense data for comparison
fn generate_dense_data(n_samples: usize, n_features: usize, seed: u64) -> Array2<f64> {
    let mut data = Vec::with_capacity(n_samples * n_features);

    for i in 0..n_samples {
        for j in 0..n_features {
            let mut hasher = DefaultHasher::new();
            (seed, i, j, "dense").hash(&mut hasher);
            let h = hasher.finish();
            let val = (h as f64 / u64::MAX as f64) * 10.0 - 5.0;
            data.push(val);
        }
    }

    Array2::from_shape_vec((n_samples, n_features), data).expect("Failed to create dense data")
}

/// Compute dense Euclidean distance for comparison
fn dense_euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute dense Manhattan distance for comparison
fn dense_manhattan_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Compute dense cosine similarity for comparison
fn dense_cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute dense dot product for comparison
fn dense_dot_product(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

const EPSILON: f64 = 1e-10;

// =============================================================================
// CSR Matrix Creation Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod csr_creation_tests {
    use super::*;
    use crate::sparse::{CsrMatrix, SparseMatrixInfo};

    /// Test basic CSR matrix creation from dense data
    #[test]
    fn test_csr_from_dense_basic() {
        let dense = generate_dense_data(10, 5, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.shape(), (10, 5));
        assert_eq!(sparse.nrows(), 10);
        assert_eq!(sparse.ncols(), 5);
    }

    /// Test CSR matrix creation from highly sparse data
    #[test]
    fn test_csr_from_highly_sparse() {
        let dense = generate_text_like_data(100, 1000, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.shape(), (100, 1000));
        assert!(sparse.sparsity() > 0.95, "Sparsity should be >95%");
    }

    /// Test CSR matrix creation from moderately sparse data
    #[test]
    fn test_csr_from_moderate_sparse() {
        let dense = generate_moderate_sparse_data(50, 100, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.shape(), (50, 100));
        assert!(sparse.sparsity() > 0.80, "Sparsity should be >80%");
    }

    /// Test CSR matrix creation from triplets
    #[test]
    fn test_csr_from_triplets() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let sparse = CsrMatrix::from_triplets((3, 3), &rows, &cols, &values)
            .expect("Failed to create from triplets");

        assert_eq!(sparse.shape(), (3, 3));
        assert_eq!(sparse.nnz(), 5);
    }

    /// Test CSR matrix creation with threshold
    #[test]
    fn test_csr_from_dense_with_threshold() {
        let mut dense = Array2::zeros((5, 5));
        dense[[0, 0]] = 0.01; // Very small
        dense[[1, 1]] = 0.1; // Small but above default threshold
        dense[[2, 2]] = 1.0; // Normal value

        // With threshold = 0.05, should ignore 0.01 but keep 0.1
        let sparse = CsrMatrix::from_dense_with_threshold(&dense, 0.05);

        assert_eq!(sparse.nnz(), 2); // Only 0.1 and 1.0
    }

    /// Test CSR matrix creation from empty array
    #[test]
    fn test_csr_from_empty_cols() {
        let dense = Array2::<f64>::zeros((5, 0));
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.shape(), (5, 0));
        assert_eq!(sparse.nnz(), 0);
    }

    /// Test CSR matrix creation with all zeros
    #[test]
    fn test_csr_all_zeros() {
        let dense = Array2::<f64>::zeros((10, 10));
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.shape(), (10, 10));
        assert_eq!(sparse.nnz(), 0);
        assert!((sparse.sparsity() - 1.0).abs() < EPSILON);
    }

    /// Test CSR matrix with single non-zero element
    #[test]
    fn test_csr_single_nonzero() {
        let mut dense = Array2::<f64>::zeros((10, 10));
        dense[[5, 5]] = 42.0;

        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.nnz(), 1);
        let recovered = sparse.to_dense();
        assert!((recovered[[5, 5]] - 42.0).abs() < EPSILON);
    }

    /// Test invalid triplet creation (index out of bounds)
    #[test]
    fn test_csr_triplets_invalid_index() {
        let rows = vec![0, 5]; // 5 is out of bounds for shape (3, 3)
        let cols = vec![0, 0];
        let values = vec![1.0, 2.0];

        let result = CsrMatrix::from_triplets((3, 3), &rows, &cols, &values);
        assert!(result.is_err(), "Should fail with out of bounds index");
    }

    /// Test triplet creation with mismatched lengths
    #[test]
    fn test_csr_triplets_mismatched_lengths() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1]; // Different length
        let values = vec![1.0, 2.0, 3.0];

        let result = CsrMatrix::from_triplets((3, 3), &rows, &cols, &values);
        assert!(result.is_err(), "Should fail with mismatched lengths");
    }
}

// =============================================================================
// CSC Matrix Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod csc_tests {
    use super::*;
    use crate::sparse::{CscMatrix, CsrMatrix};

    /// Test CSC matrix creation from CSR transpose
    #[test]
    fn test_csc_from_csr_transpose() {
        let dense = generate_moderate_sparse_data(20, 30, 42);
        let csr = CsrMatrix::from_dense(&dense);
        let csc = csr.transpose();

        // CSC of original should be CSR of transpose
        assert_eq!(csc.shape(), (30, 20)); // Transposed shape
    }

    /// Test CSC to CSR conversion
    #[test]
    fn test_csc_to_csr_roundtrip() {
        let dense = generate_moderate_sparse_data(15, 25, 42);
        let csr_original = CsrMatrix::from_dense(&dense);
        let csc = csr_original.transpose();
        let csr_recovered = csc.to_csr();

        // After transpose and convert back, should have transposed data
        assert_eq!(csr_recovered.shape(), (25, 15));
    }

    /// Test CSC to dense conversion
    #[test]
    fn test_csc_to_dense() {
        let dense = generate_moderate_sparse_data(10, 10, 42);
        let csr = CsrMatrix::from_dense(&dense);
        let csc = csr.transpose();
        let recovered = csc.to_dense();

        // Recovered dense should be transpose of original
        for i in 0..10 {
            for j in 0..10 {
                assert!(
                    (dense[[i, j]] - recovered[[j, i]]).abs() < EPSILON,
                    "Transpose mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}

// =============================================================================
// Sparse Vector Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod sparse_vector_tests {
    use super::*;
    use crate::sparse::SparseVector;

    /// Test sparse vector creation from dense
    #[test]
    fn test_sparse_vector_from_dense() {
        let dense = Array1::from_vec(vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0]);
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.dim(), 6);
        assert_eq!(sparse.nnz(), 3);
    }

    /// Test sparse vector with threshold
    #[test]
    fn test_sparse_vector_with_threshold() {
        let dense = Array1::from_vec(vec![0.001, 0.0, 0.1, 1.0, 0.05]);
        let sparse = SparseVector::from_dense_with_threshold(&dense, 0.05);

        // Should only keep 0.1 and 1.0
        assert_eq!(sparse.nnz(), 2);
    }

    /// Test sparse vector to dense roundtrip
    #[test]
    fn test_sparse_vector_roundtrip() {
        let dense = Array1::from_vec(vec![1.0, 0.0, 2.0, 0.0, 3.0]);
        let sparse = SparseVector::from_dense(&dense);
        let recovered = sparse.to_dense();

        for i in 0..5 {
            assert!((dense[i] - recovered[i]).abs() < EPSILON);
        }
    }

    /// Test sparse vector creation with explicit indices
    #[test]
    fn test_sparse_vector_explicit() {
        let indices = vec![0, 2, 4];
        let data = vec![1.0, 2.0, 3.0];

        let sparse = SparseVector::new(5, indices, data).expect("Failed to create vector");

        assert_eq!(sparse.dim(), 5);
        assert_eq!(sparse.nnz(), 3);
    }

    /// Test sparse vector with out of bounds index
    #[test]
    fn test_sparse_vector_invalid_index() {
        let indices = vec![0, 10]; // 10 is out of bounds for dim=5
        let data = vec![1.0, 2.0];

        let result = SparseVector::new(5, indices, data);
        assert!(result.is_err());
    }

    /// Test empty sparse vector
    #[test]
    fn test_sparse_vector_empty() {
        let dense = Array1::<f64>::zeros(10);
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.dim(), 10);
    }
}

// =============================================================================
// Sparse/Dense Equivalence Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod sparse_dense_equivalence_tests {
    use super::*;
    use crate::sparse::{
        sparse_cosine_distance, sparse_cosine_similarity, sparse_dot_product,
        sparse_euclidean_distance, sparse_manhattan_distance, CsrMatrix,
    };

    /// Test that sparse and dense Euclidean distances match
    #[test]
    fn test_euclidean_distance_equivalence() {
        let dense = generate_moderate_sparse_data(10, 20, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        // Compare distances for several row pairs
        for i in 0..5 {
            for j in (i + 1)..5 {
                let dense_dist =
                    dense_euclidean_distance(&dense.row(i).to_owned(), &dense.row(j).to_owned());
                let sparse_dist = sparse_euclidean_distance(&sparse.row(i), &sparse.row(j));

                assert!(
                    (dense_dist - sparse_dist).abs() < 1e-8,
                    "Euclidean distance mismatch for rows ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    dense_dist,
                    sparse_dist
                );
            }
        }
    }

    /// Test that sparse and dense Manhattan distances match
    #[test]
    fn test_manhattan_distance_equivalence() {
        let dense = generate_moderate_sparse_data(10, 20, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        for i in 0..5 {
            for j in (i + 1)..5 {
                let dense_dist =
                    dense_manhattan_distance(&dense.row(i).to_owned(), &dense.row(j).to_owned());
                let sparse_dist = sparse_manhattan_distance(&sparse.row(i), &sparse.row(j));

                assert!(
                    (dense_dist - sparse_dist).abs() < 1e-8,
                    "Manhattan distance mismatch for rows ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    dense_dist,
                    sparse_dist
                );
            }
        }
    }

    /// Test that sparse and dense cosine similarities match
    #[test]
    fn test_cosine_similarity_equivalence() {
        let dense = generate_moderate_sparse_data(10, 20, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        for i in 0..5 {
            for j in (i + 1)..5 {
                let dense_sim =
                    dense_cosine_similarity(&dense.row(i).to_owned(), &dense.row(j).to_owned());
                let sparse_sim = sparse_cosine_similarity(&sparse.row(i), &sparse.row(j));

                assert!(
                    (dense_sim - sparse_sim).abs() < 1e-8,
                    "Cosine similarity mismatch for rows ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    dense_sim,
                    sparse_sim
                );
            }
        }
    }

    /// Test that sparse and dense dot products match
    #[test]
    fn test_dot_product_equivalence() {
        let dense = generate_moderate_sparse_data(10, 20, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        for i in 0..5 {
            for j in 0..5 {
                let dense_dot =
                    dense_dot_product(&dense.row(i).to_owned(), &dense.row(j).to_owned());
                let sparse_dot = sparse_dot_product(&sparse.row(i), &sparse.row(j));

                assert!(
                    (dense_dot - sparse_dot).abs() < 1e-8,
                    "Dot product mismatch for rows ({}, {}): dense={}, sparse={}",
                    i,
                    j,
                    dense_dot,
                    sparse_dot
                );
            }
        }
    }

    /// Test dense/sparse roundtrip preserves values
    #[test]
    fn test_dense_sparse_roundtrip() {
        let original = generate_dense_data(20, 15, 42);
        let sparse = CsrMatrix::from_dense(&original);
        let recovered = sparse.to_dense();

        for i in 0..20 {
            for j in 0..15 {
                assert!(
                    (original[[i, j]] - recovered[[i, j]]).abs() < EPSILON,
                    "Roundtrip mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    /// Test dense/sparse roundtrip with highly sparse data
    #[test]
    fn test_dense_sparse_roundtrip_highly_sparse() {
        let original = generate_text_like_data(50, 100, 42);
        let sparse = CsrMatrix::from_dense(&original);
        let recovered = sparse.to_dense();

        for i in 0..50 {
            for j in 0..100 {
                assert!(
                    (original[[i, j]] - recovered[[i, j]]).abs() < EPSILON,
                    "Roundtrip mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}

// =============================================================================
// Sparse Distance Batch Operations Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod sparse_distance_batch_tests {
    use super::*;
    use crate::sparse::{sparse_pairwise_distances, CsrMatrix, SparseDistanceMetric};

    /// Test pairwise Euclidean distances
    #[test]
    fn test_pairwise_euclidean() {
        let dense = generate_moderate_sparse_data(20, 30, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        let query_row = sparse.row(0);
        let distances =
            sparse_pairwise_distances(&query_row, &sparse, SparseDistanceMetric::Euclidean);

        assert_eq!(distances.len(), 20);

        // Distance to self should be 0
        assert!(distances[0].abs() < EPSILON, "Distance to self should be 0");

        // All distances should be non-negative
        for d in distances.iter() {
            assert!(*d >= 0.0, "Distance should be non-negative");
        }
    }

    /// Test pairwise squared Euclidean distances
    #[test]
    fn test_pairwise_squared_euclidean() {
        let dense = generate_moderate_sparse_data(15, 25, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        let query_row = sparse.row(5);
        let distances =
            sparse_pairwise_distances(&query_row, &sparse, SparseDistanceMetric::SquaredEuclidean);

        assert_eq!(distances.len(), 15);

        // Distance to self should be 0
        assert!(distances[5].abs() < EPSILON);

        // All squared distances should be non-negative
        for d in distances.iter() {
            assert!(*d >= 0.0, "Squared distance should be non-negative");
        }
    }

    /// Test pairwise Manhattan distances
    #[test]
    fn test_pairwise_manhattan() {
        let dense = generate_moderate_sparse_data(15, 25, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        let query_row = sparse.row(3);
        let distances =
            sparse_pairwise_distances(&query_row, &sparse, SparseDistanceMetric::Manhattan);

        assert_eq!(distances.len(), 15);
        assert!(distances[3].abs() < EPSILON, "Distance to self should be 0");
    }

    /// Test pairwise Cosine distances
    #[test]
    fn test_pairwise_cosine() {
        let dense = generate_moderate_sparse_data(15, 25, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        let query_row = sparse.row(7);
        let distances =
            sparse_pairwise_distances(&query_row, &sparse, SparseDistanceMetric::Cosine);

        assert_eq!(distances.len(), 15);

        // Cosine distance to self should be 0 (similarity = 1)
        assert!(
            distances[7].abs() < EPSILON,
            "Cosine distance to self should be 0"
        );

        // Cosine distances should be in [0, 2]
        for d in distances.iter() {
            assert!(
                *d >= 0.0 - EPSILON && *d <= 2.0 + EPSILON,
                "Cosine distance should be in [0, 2], got {}",
                d
            );
        }
    }
}

// =============================================================================
// Sparse Matrix Operations Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod sparse_matrix_operations_tests {
    use super::*;
    use crate::sparse::{
        sparse_column_means, sparse_column_nnz, sparse_column_sums, sparse_diag, sparse_eye,
        sparse_hstack, sparse_normalize_rows_l2, sparse_vstack, CsrMatrix,
    };

    /// Test sparse identity matrix
    #[test]
    fn test_sparse_eye() {
        let eye = sparse_eye(5);

        assert_eq!(eye.shape(), (5, 5));
        assert_eq!(eye.nnz(), 5);

        let dense = eye.to_dense();
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dense[[i, j]] - expected).abs() < EPSILON);
            }
        }
    }

    /// Test sparse diagonal matrix
    #[test]
    fn test_sparse_diag() {
        let diag_values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let diag = sparse_diag(&diag_values);

        assert_eq!(diag.shape(), (4, 4));
        assert_eq!(diag.nnz(), 4);

        let dense = diag.to_dense();
        for i in 0..4 {
            assert!((dense[[i, i]] - diag_values[i]).abs() < EPSILON);
        }
    }

    /// Test sparse diagonal with zeros
    #[test]
    fn test_sparse_diag_with_zeros() {
        let diag_values = Array1::from_vec(vec![1.0, 0.0, 3.0, 0.0]);
        let diag = sparse_diag(&diag_values);

        assert_eq!(diag.shape(), (4, 4));
        assert_eq!(diag.nnz(), 2); // Only 2 non-zeros
    }

    /// Test vertical stacking of sparse matrices
    #[test]
    fn test_sparse_vstack() {
        let a = CsrMatrix::from_dense(
            &Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap(),
        );
        let b = CsrMatrix::from_dense(
            &Array2::from_shape_vec((2, 3), vec![4.0, 0.0, 5.0, 0.0, 6.0, 0.0]).unwrap(),
        );

        let stacked = sparse_vstack(&[&a, &b]).expect("Failed to vstack");

        assert_eq!(stacked.shape(), (4, 3));
        assert_eq!(stacked.nnz(), 6);
    }

    /// Test vertical stacking with mismatched columns fails
    #[test]
    fn test_sparse_vstack_mismatch() {
        let a = CsrMatrix::from_dense(&Array2::<f64>::zeros((2, 3)));
        let b = CsrMatrix::from_dense(&Array2::<f64>::zeros((2, 4)));

        let result = sparse_vstack(&[&a, &b]);
        assert!(result.is_err(), "Should fail with mismatched columns");
    }

    /// Test horizontal stacking of sparse matrices
    #[test]
    fn test_sparse_hstack() {
        let a = CsrMatrix::from_dense(
            &Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0]).unwrap(),
        );
        let b = CsrMatrix::from_dense(
            &Array2::from_shape_vec((3, 2), vec![4.0, 0.0, 0.0, 5.0, 6.0, 0.0]).unwrap(),
        );

        let stacked = sparse_hstack(&[&a, &b]).expect("Failed to hstack");

        assert_eq!(stacked.shape(), (3, 4));
    }

    /// Test horizontal stacking with mismatched rows fails
    #[test]
    fn test_sparse_hstack_mismatch() {
        let a = CsrMatrix::from_dense(&Array2::<f64>::zeros((3, 2)));
        let b = CsrMatrix::from_dense(&Array2::<f64>::zeros((4, 2)));

        let result = sparse_hstack(&[&a, &b]);
        assert!(result.is_err(), "Should fail with mismatched rows");
    }

    /// Test column sums
    #[test]
    fn test_sparse_column_sums() {
        let dense =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0])
                .unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let sums = sparse_column_sums(&sparse);

        assert!((sums[0] - 9.0).abs() < EPSILON); // 1 + 3 + 5
        assert!((sums[1] - 8.0).abs() < EPSILON); // 2 + 0 + 6
        assert!((sums[2] - 4.0).abs() < EPSILON); // 0 + 4 + 0
    }

    /// Test column means
    #[test]
    fn test_sparse_column_means() {
        let dense =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let means = sparse_column_means(&sparse);

        assert!((means[0] - 4.0).abs() < EPSILON); // (1+3+5+7)/4
        assert!((means[1] - 5.0).abs() < EPSILON); // (2+4+6+8)/4
    }

    /// Test column non-zero counts
    #[test]
    fn test_sparse_column_nnz() {
        let dense = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
        )
        .unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let nnz_per_col = sparse_column_nnz(&sparse);

        assert_eq!(nnz_per_col[0], 3); // col 0: 1, 2, 6
        assert_eq!(nnz_per_col[1], 2); // col 1: 3, 4
        assert_eq!(nnz_per_col[2], 1); // col 2: 5
    }

    /// Test row L2 normalization
    #[test]
    fn test_sparse_normalize_rows_l2() {
        let dense = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 5.0, 12.0]).unwrap();
        let mut sparse = CsrMatrix::from_dense(&dense);

        sparse_normalize_rows_l2(&mut sparse);

        let normalized = sparse.to_dense();

        // Row 0: [3, 4] with norm 5 -> [0.6, 0.8]
        assert!((normalized[[0, 0]] - 0.6).abs() < EPSILON);
        assert!((normalized[[0, 1]] - 0.8).abs() < EPSILON);

        // Row 1: [5, 12] with norm 13 -> [5/13, 12/13]
        assert!((normalized[[1, 0]] - 5.0 / 13.0).abs() < EPSILON);
        assert!((normalized[[1, 1]] - 12.0 / 13.0).abs() < EPSILON);
    }

    /// Test row norms
    #[test]
    fn test_row_norms() {
        let dense = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 5.0, 12.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let norms = sparse.row_norms();

        assert!((norms[0] - 5.0).abs() < EPSILON);
        assert!((norms[1] - 13.0).abs() < EPSILON);
    }

    /// Test matrix-vector dot product
    #[test]
    fn test_matrix_vector_dot() {
        let dense = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = sparse.dot(&x).expect("Failed to compute dot product");

        // Row 0: 1*1 + 2*2 + 3*3 = 14
        // Row 1: 4*1 + 5*2 + 6*3 = 32
        assert!((result[0] - 14.0).abs() < EPSILON);
        assert!((result[1] - 32.0).abs() < EPSILON);
    }

    /// Test matrix-vector dot product with wrong dimensions
    #[test]
    fn test_matrix_vector_dot_mismatch() {
        let sparse = CsrMatrix::from_dense(&Array2::<f64>::zeros((3, 4)));
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Wrong size

        let result = sparse.dot(&x);
        assert!(result.is_err(), "Should fail with mismatched dimensions");
    }
}

// =============================================================================
// Memory Efficiency Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod memory_efficiency_tests {
    use super::*;
    use crate::sparse::{sparse_eye, CsrMatrix, SparseMatrixInfo};

    /// Test that highly sparse data uses less memory in sparse format
    #[test]
    fn test_memory_efficiency_highly_sparse() {
        let dense = generate_text_like_data(100, 1000, 42);
        let sparse = CsrMatrix::from_dense(&dense);
        let info = SparseMatrixInfo::from_matrix(&sparse);

        assert!(
            info.sparse_memory < info.dense_memory,
            "Sparse format should use less memory for highly sparse data. \
             Sparse: {} bytes, Dense: {} bytes",
            info.sparse_memory,
            info.dense_memory
        );

        assert!(
            !info.recommend_dense,
            "Should not recommend dense for highly sparse data"
        );
    }

    /// Test that dense data is recommended for low sparsity
    #[test]
    fn test_memory_efficiency_low_sparsity() {
        // Create dense data with only 30% zeros
        let dense = generate_sparse_data(50, 50, 0.3, 42);
        let sparse = CsrMatrix::from_dense(&dense);
        let info = SparseMatrixInfo::from_matrix(&sparse);

        assert!(
            info.recommend_dense,
            "Should recommend dense for low sparsity data (sparsity: {:.2})",
            info.sparsity
        );
    }

    /// Test SparseMatrixInfo correctness
    #[test]
    fn test_sparse_matrix_info() {
        let dense = generate_moderate_sparse_data(100, 200, 42);
        let sparse = CsrMatrix::from_dense(&dense);
        let info = SparseMatrixInfo::from_matrix(&sparse);

        assert_eq!(info.nrows, 100);
        assert_eq!(info.ncols, 200);
        assert_eq!(info.nnz, sparse.nnz());
        assert!((info.sparsity + info.density - 1.0).abs() < EPSILON);
    }

    /// Test memory calculation formula
    #[test]
    fn test_memory_calculation() {
        // Create a known sparse matrix
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let values = vec![1.0, 2.0, 3.0];
        let sparse = CsrMatrix::from_triplets((3, 3), &rows, &cols, &values).unwrap();
        let info = SparseMatrixInfo::from_matrix(&sparse);

        // Expected sparse memory: 3 nnz * 16 bytes + 4 indptr * 8 bytes = 48 + 32 = 80 bytes
        let expected_sparse_memory = 3 * 16 + 4 * 8;
        assert_eq!(info.sparse_memory, expected_sparse_memory);

        // Expected dense memory: 9 elements * 8 bytes = 72 bytes
        let expected_dense_memory = 9 * 8;
        assert_eq!(info.dense_memory, expected_dense_memory);
    }

    /// Test sparsity summary output
    #[test]
    fn test_sparse_matrix_info_summary() {
        let sparse = sparse_eye(10);
        let info = SparseMatrixInfo::from_matrix(&sparse);
        let summary = info.summary();

        assert!(summary.contains("10x10"));
        assert!(summary.contains("10 nnz"));
        assert!(summary.contains("90.0% sparse"));
    }
}

// =============================================================================
// Sparse Row View Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod sparse_row_view_tests {
    use super::*;
    use crate::sparse::CsrMatrix;

    /// Test sparse row view basic properties
    #[test]
    fn test_row_view_properties() {
        let dense = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
        )
        .unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let row0 = sparse.row(0);
        assert_eq!(row0.nnz(), 2);
        assert_eq!(row0.dim(), 4);
        assert!(!row0.is_empty());
    }

    /// Test empty row view
    #[test]
    fn test_empty_row_view() {
        let mut dense = Array2::<f64>::zeros((3, 4));
        dense[[1, 0]] = 1.0; // Only row 1 has values

        let sparse = CsrMatrix::from_dense(&dense);

        let row0 = sparse.row(0);
        assert!(row0.is_empty());
        assert_eq!(row0.nnz(), 0);
        assert_eq!(row0.norm(), 0.0);
    }

    /// Test row view norm calculations
    #[test]
    fn test_row_view_norms() {
        let dense = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let row = sparse.row(0);

        assert!((row.norm() - 5.0).abs() < EPSILON);
        assert!((row.norm_squared() - 25.0).abs() < EPSILON);
        assert!((row.l1_norm() - 7.0).abs() < EPSILON);
    }

    /// Test row view to dense conversion
    #[test]
    fn test_row_view_to_dense() {
        let dense =
            Array2::from_shape_vec((2, 4), vec![1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let row0_view = sparse.row(0);
        let row0_dense = row0_view.to_dense();

        assert_eq!(row0_dense.len(), 4);
        assert!((row0_dense[0] - 1.0).abs() < EPSILON);
        assert!((row0_dense[1] - 0.0).abs() < EPSILON);
        assert!((row0_dense[2] - 2.0).abs() < EPSILON);
        assert!((row0_dense[3] - 0.0).abs() < EPSILON);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod edge_case_tests {
    use super::*;
    use crate::sparse::{
        sparse_cosine_similarity, sparse_euclidean_distance, sparse_manhattan_distance, CsrMatrix,
    };

    /// Test distance between identical rows
    #[test]
    fn test_distance_identical_rows() {
        let dense = generate_moderate_sparse_data(5, 10, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        let row0 = sparse.row(0);
        let euclidean = sparse_euclidean_distance(&row0, &row0);
        let manhattan = sparse_manhattan_distance(&row0, &row0);
        let cosine_sim = sparse_cosine_similarity(&row0, &row0);

        assert!(euclidean.abs() < EPSILON, "Euclidean to self should be 0");
        assert!(manhattan.abs() < EPSILON, "Manhattan to self should be 0");
        assert!(
            (cosine_sim - 1.0).abs() < EPSILON,
            "Cosine similarity to self should be 1"
        );
    }

    /// Test distance between zero rows
    #[test]
    fn test_distance_zero_rows() {
        let dense = Array2::<f64>::zeros((2, 5));
        let sparse = CsrMatrix::from_dense(&dense);

        let row0 = sparse.row(0);
        let row1 = sparse.row(1);

        let euclidean = sparse_euclidean_distance(&row0, &row1);
        let manhattan = sparse_manhattan_distance(&row0, &row1);
        let cosine_sim = sparse_cosine_similarity(&row0, &row1);

        assert!(
            euclidean.abs() < EPSILON,
            "Zero rows have 0 Euclidean distance"
        );
        assert!(
            manhattan.abs() < EPSILON,
            "Zero rows have 0 Manhattan distance"
        );
        assert!(
            cosine_sim.abs() < EPSILON,
            "Zero rows have 0 cosine similarity"
        );
    }

    /// Test distance between orthogonal rows
    #[test]
    fn test_distance_orthogonal_rows() {
        // Two orthogonal unit vectors
        let dense = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        let row0 = sparse.row(0);
        let row1 = sparse.row(1);

        let cosine_sim = sparse_cosine_similarity(&row0, &row1);
        assert!(
            cosine_sim.abs() < EPSILON,
            "Orthogonal vectors have 0 cosine similarity"
        );
    }

    /// Test with very large values
    #[test]
    fn test_large_values() {
        let mut dense = Array2::<f64>::zeros((2, 3));
        dense[[0, 0]] = 1e10;
        dense[[1, 1]] = 1e10;

        let sparse = CsrMatrix::from_dense(&dense);

        let row0 = sparse.row(0);
        let row1 = sparse.row(1);

        let dist = sparse_euclidean_distance(&row0, &row1);
        assert!(
            dist.is_finite(),
            "Distance with large values should be finite"
        );
    }

    /// Test with very small values
    #[test]
    fn test_small_values() {
        let mut dense = Array2::<f64>::zeros((2, 3));
        dense[[0, 0]] = 1e-15;
        dense[[1, 1]] = 1e-15;

        let sparse = CsrMatrix::from_dense(&dense);
        assert_eq!(sparse.nnz(), 2); // Should still capture tiny values > 0
    }

    /// Test with negative values
    #[test]
    fn test_negative_values() {
        let dense = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, -2.0, 3.0, -4.0, 0.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.nnz(), 4);

        let row0 = sparse.row(0);
        let row1 = sparse.row(1);

        let dist = sparse_euclidean_distance(&row0, &row1);
        assert!(dist.is_finite() && dist >= 0.0);
    }

    /// Test single element matrix
    #[test]
    fn test_single_element() {
        let dense = Array2::from_shape_vec((1, 1), vec![42.0]).unwrap();
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.shape(), (1, 1));
        assert_eq!(sparse.nnz(), 1);

        let recovered = sparse.to_dense();
        assert!((recovered[[0, 0]] - 42.0).abs() < EPSILON);
    }

    /// Test very wide matrix (many columns)
    #[test]
    fn test_wide_matrix() {
        let dense = generate_text_like_data(10, 10000, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.nrows(), 10);
        assert_eq!(sparse.ncols(), 10000);
        assert!(sparse.sparsity() > 0.95);
    }

    /// Test very tall matrix (many rows)
    #[test]
    fn test_tall_matrix() {
        let dense = generate_text_like_data(10000, 10, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.nrows(), 10000);
        assert_eq!(sparse.ncols(), 10);
    }

    /// Test matrix with all same values
    #[test]
    fn test_all_same_nonzero_values() {
        let dense = Array2::from_elem((5, 5), 3.14159);
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.nnz(), 25);

        let recovered = sparse.to_dense();
        for i in 0..5 {
            for j in 0..5 {
                assert!((recovered[[i, j]] - 3.14159).abs() < EPSILON);
            }
        }
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(feature = "sparse")]
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::sparse::{sparse_pairwise_distances, CsrMatrix, SparseDistanceMetric};

    /// Test complete workflow: create -> manipulate -> query
    #[test]
    fn test_complete_workflow() {
        // 1. Create sparse matrix from text-like data
        let dense = generate_text_like_data(50, 500, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        // 2. Verify properties
        assert_eq!(sparse.shape(), (50, 500));
        assert!(sparse.sparsity() > 0.95);

        // 3. Compute distances
        let query = sparse.row(0);
        let distances = sparse_pairwise_distances(&query, &sparse, SparseDistanceMetric::Euclidean);

        // 4. Find nearest neighbor (should be self at index 0)
        let mut min_idx = 1;
        let mut min_dist = distances[1];
        for i in 2..distances.len() {
            if distances[i] < min_dist {
                min_dist = distances[i];
                min_idx = i;
            }
        }

        assert!(distances[0] < min_dist, "Self should be closest");
        println!(
            "Nearest neighbor to row 0: row {} with distance {}",
            min_idx, min_dist
        );
    }

    /// Test sparse matrix for k-NN style operation
    #[test]
    fn test_knn_style_operation() {
        let dense = generate_moderate_sparse_data(100, 50, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        // For each row, find k=3 nearest neighbors
        let k = 3;
        for query_idx in 0..5 {
            let query = sparse.row(query_idx);
            let distances =
                sparse_pairwise_distances(&query, &sparse, SparseDistanceMetric::SquaredEuclidean);

            // Get k nearest (excluding self)
            let mut indexed_distances: Vec<(usize, f64)> =
                distances.iter().cloned().enumerate().collect();
            indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // First should be self with distance 0
            assert_eq!(indexed_distances[0].0, query_idx);
            assert!(indexed_distances[0].1 < EPSILON);

            // Next k should be neighbors
            let neighbors: Vec<usize> = indexed_distances[1..=k].iter().map(|(i, _)| *i).collect();
            assert_eq!(neighbors.len(), k);
        }
    }

    /// Test recommendation-style workflow
    #[test]
    fn test_recommendation_workflow() {
        // User-item interaction matrix (sparse)
        let dense = generate_moderate_sparse_data(20, 100, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        // Find similar users based on cosine similarity
        let user_0 = sparse.row(0);
        let distances = sparse_pairwise_distances(&user_0, &sparse, SparseDistanceMetric::Cosine);

        // Convert distances to similarities
        let similarities: Vec<f64> = distances.iter().map(|d| 1.0 - d).collect();

        // Verify self-similarity is highest
        assert!((similarities[0] - 1.0).abs() < EPSILON);
        for &sim in &similarities[1..] {
            assert!(sim <= 1.0 + EPSILON);
        }
    }
}

// Note: All tests in this module require the `sparse` feature flag.
// Run with: cargo test -p ferroml-core --features sparse testing::sparse_tests
