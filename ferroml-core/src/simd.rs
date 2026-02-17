//! SIMD-Accelerated Distance Calculations
//!
//! This module provides SIMD-optimized implementations of common distance metrics
//! used in machine learning algorithms like K-Nearest Neighbors.
//!
//! ## Features
//!
//! Enable this module by adding the `simd` feature to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! ferroml-core = { version = "0.1", features = ["simd"] }
//! ```
//!
//! ## Supported Distance Metrics
//!
//! - **Euclidean distance**: L2 norm, sqrt(sum((x_i - y_i)^2))
//! - **Squared Euclidean distance**: L2 squared, sum((x_i - y_i)^2)
//! - **Manhattan distance**: L1 norm, sum(|x_i - y_i|)
//!
//! ## Performance
//!
//! SIMD operations process multiple elements in parallel using vector instructions:
//! - AVX2 (x86_64): 4 f64 elements per instruction
//! - SSE2 (x86_64 fallback): 2 f64 elements per instruction
//! - NEON (ARM): 2 f64 elements per instruction
//!
//! ## Example
//!
//! ```
//! use ferroml_core::simd::{euclidean_distance, manhattan_distance};
//!
//! let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
//!
//! let euclidean = euclidean_distance(&a, &b);
//! let manhattan = manhattan_distance(&a, &b);
//! ```

use wide::f64x4;

/// Compute the squared Euclidean distance between two vectors using SIMD.
///
/// This is the sum of squared differences: sum((a_i - b_i)^2)
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The squared Euclidean distance.
///
/// # Panics
///
/// Panics in debug mode if the vectors have different lengths.
#[inline]
pub fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4 elements at a time using SIMD
    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);

        let diff = a_vec - b_vec;
        sum_vec = sum_vec + diff * diff;
    }

    // Horizontal sum of the SIMD vector
    let sum_array: [f64; 4] = sum_vec.into();
    let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle remainder elements with mul_add for better precision
    let remainder_start = chunks * 4;
    for i in 0..remainder {
        let diff = a[remainder_start + i] - b[remainder_start + i];
        sum = diff.mul_add(diff, sum);
    }

    sum
}

/// Compute the Euclidean distance between two vectors using SIMD.
///
/// This is the L2 norm: sqrt(sum((a_i - b_i)^2))
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The Euclidean distance.
///
/// # Example
///
/// ```
/// use ferroml_core::simd::euclidean_distance;
///
/// let a = [0.0, 0.0];
/// let b = [3.0, 4.0];
/// assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
/// ```
#[inline]
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_distance(a, b).sqrt()
}

/// Compute the Manhattan (L1) distance between two vectors using SIMD.
///
/// This is the sum of absolute differences: sum(|a_i - b_i|)
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The Manhattan distance.
///
/// # Example
///
/// ```
/// use ferroml_core::simd::manhattan_distance;
///
/// let a = [0.0, 0.0];
/// let b = [3.0, 4.0];
/// assert!((manhattan_distance(&a, &b) - 7.0).abs() < 1e-10);
/// ```
#[inline]
pub fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4 elements at a time using SIMD
    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);

        let diff = a_vec - b_vec;
        sum_vec = sum_vec + diff.abs();
    }

    // Horizontal sum of the SIMD vector
    let sum_array: [f64; 4] = sum_vec.into();
    let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle remainder elements
    let remainder_start = chunks * 4;
    for i in 0..remainder {
        sum += (a[remainder_start + i] - b[remainder_start + i]).abs();
    }

    sum
}

/// Compute the squared Euclidean distance for a batch of query-reference pairs.
///
/// This is useful for K-NN where we compute distances from one query to many references.
///
/// # Arguments
///
/// * `query` - Query vector
/// * `references` - Matrix of reference vectors (row-major, each row is a reference)
///
/// # Returns
///
/// Vector of squared Euclidean distances from query to each reference.
#[inline]
pub fn batch_squared_euclidean(
    query: &[f64],
    references: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Vec<f64> {
    debug_assert_eq!(
        query.len(),
        n_features,
        "Query must have n_features dimensions"
    );
    debug_assert_eq!(
        references.len(),
        n_samples * n_features,
        "References must have n_samples * n_features elements"
    );

    let mut distances = Vec::with_capacity(n_samples);

    for sample_idx in 0..n_samples {
        let ref_start = sample_idx * n_features;
        let ref_slice = &references[ref_start..ref_start + n_features];
        distances.push(squared_euclidean_distance(query, ref_slice));
    }

    distances
}

/// Compute the dot product of two vectors using SIMD.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The dot product sum(a_i * b_i).
#[inline]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4 elements at a time using SIMD
    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);

        sum_vec = sum_vec + a_vec * b_vec;
    }

    // Horizontal sum of the SIMD vector
    let sum_array: [f64; 4] = sum_vec.into();
    let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle remainder elements with mul_add for better precision
    let remainder_start = chunks * 4;
    for i in 0..remainder {
        sum = a[remainder_start + i].mul_add(b[remainder_start + i], sum);
    }

    sum
}

/// Compute the sum of a vector using SIMD.
///
/// # Arguments
///
/// * `a` - Input vector
///
/// # Returns
///
/// The sum of all elements.
#[inline]
pub fn sum(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4 elements at a time using SIMD
    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        sum_vec = sum_vec + a_vec;
    }

    // Horizontal sum of the SIMD vector
    let sum_array: [f64; 4] = sum_vec.into();
    let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle remainder elements
    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result += a[remainder_start + i];
    }

    result
}

/// Compute the sum of squares of a vector using SIMD.
///
/// # Arguments
///
/// * `a` - Input vector
///
/// # Returns
///
/// The sum of squared elements: sum(a_i^2).
#[inline]
pub fn sum_of_squares(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    // Process 4 elements at a time using SIMD
    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        sum_vec = sum_vec + a_vec * a_vec;
    }

    // Horizontal sum of the SIMD vector
    let sum_array: [f64; 4] = sum_vec.into();
    let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    // Handle remainder elements
    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result += a[remainder_start + i] * a[remainder_start + i];
    }

    result
}

/// Compute element-wise difference and square using SIMD.
///
/// Returns a new vector where each element is (a_i - b_i)^2.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// Vector of squared differences.
#[inline]
pub fn squared_differences(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = vec![0.0; n];

    // Process 4 elements at a time using SIMD
    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);

        let diff = a_vec - b_vec;
        let sq = diff * diff;
        let sq_array: [f64; 4] = sq.into();

        result[offset] = sq_array[0];
        result[offset + 1] = sq_array[1];
        result[offset + 2] = sq_array[2];
        result[offset + 3] = sq_array[3];
    }

    // Handle remainder elements
    let remainder_start = chunks * 4;
    for i in 0..remainder {
        let diff = a[remainder_start + i] - b[remainder_start + i];
        result[remainder_start + i] = diff * diff;
    }

    result
}

/// Compute the Minkowski distance using SIMD for the inner loop.
///
/// d(a, b) = (sum(|a_i - b_i|^p))^(1/p)
///
/// Note: For p=1 (Manhattan) and p=2 (Euclidean), use the specialized
/// functions `manhattan_distance` and `euclidean_distance` for better performance.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
/// * `p` - The Minkowski exponent (must be >= 1)
///
/// # Returns
///
/// The Minkowski distance.
#[inline]
pub fn minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    debug_assert!(p >= 1.0, "Minkowski p must be >= 1");

    // Special cases for common p values
    if (p - 1.0).abs() < 1e-10 {
        return manhattan_distance(a, b);
    }
    if (p - 2.0).abs() < 1e-10 {
        return euclidean_distance(a, b);
    }

    // General case: |a_i - b_i|^p
    // SIMD doesn't have a direct pow operation, so we compute abs differences
    // with SIMD and then apply powf in the scalar reduction.
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = 0.0;

    // For non-integer p, we need to compute |diff|^p which requires scalar powf
    // We still benefit from SIMD for the abs(diff) computation
    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);

        let diff = a_vec - b_vec;
        let abs_diff = diff.abs();
        let abs_array: [f64; 4] = abs_diff.into();

        // Apply powf to each element (scalar operation)
        sum += abs_array[0].powf(p);
        sum += abs_array[1].powf(p);
        sum += abs_array[2].powf(p);
        sum += abs_array[3].powf(p);
    }

    // Handle remainder elements
    let remainder_start = chunks * 4;
    for i in 0..remainder {
        sum += (a[remainder_start + i] - b[remainder_start + i])
            .abs()
            .powf(p);
    }

    sum.powf(1.0 / p)
}

/// Cosine similarity between two vectors using SIMD.
///
/// cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||)
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The cosine similarity (in range [-1, 1]).
#[inline]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let dot = dot_product(a, b);
    let norm_a = sum_of_squares(a).sqrt();
    let norm_b = sum_of_squares(b).sqrt();

    let denominator = norm_a * norm_b;
    if denominator < 1e-10 {
        0.0
    } else {
        dot / denominator
    }
}

/// Cosine distance between two vectors using SIMD.
///
/// cosine_distance(a, b) = 1 - cosine_similarity(a, b)
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// The cosine distance (in range [0, 2]).
#[inline]
pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    1.0 - cosine_similarity(a, b)
}

// =============================================================================
// Matrix Operations
// =============================================================================

/// Matrix-vector multiplication using SIMD.
///
/// Computes y = A * x where A is an m×n matrix and x is an n-vector.
/// The matrix is stored in row-major order.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `vector` - Input vector (n elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Result vector (m elements)
///
/// # Example
///
/// ```
/// use ferroml_core::simd::matrix_vector_mul;
///
/// // 2×3 matrix [[1, 2, 3], [4, 5, 6]]
/// let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let vector = [1.0, 2.0, 3.0];
/// let result = matrix_vector_mul(&matrix, &vector, 2, 3);
/// // result = [14.0, 32.0]
/// ```
#[inline]
pub fn matrix_vector_mul(matrix: &[f64], vector: &[f64], m: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");
    debug_assert_eq!(vector.len(), n, "Vector must have n elements");

    let mut result = Vec::with_capacity(m);

    for row in 0..m {
        let row_start = row * n;
        let row_slice = &matrix[row_start..row_start + n];
        result.push(dot_product(row_slice, vector));
    }

    result
}

/// Matrix-vector multiplication storing result in pre-allocated buffer.
///
/// Computes y = A * x where A is an m×n matrix and x is an n-vector.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `vector` - Input vector (n elements)
/// * `result` - Output buffer (m elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
#[inline]
pub fn matrix_vector_mul_into(
    matrix: &[f64],
    vector: &[f64],
    result: &mut [f64],
    m: usize,
    n: usize,
) {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");
    debug_assert_eq!(vector.len(), n, "Vector must have n elements");
    debug_assert_eq!(result.len(), m, "Result must have m elements");

    for row in 0..m {
        let row_start = row * n;
        let row_slice = &matrix[row_start..row_start + n];
        result[row] = dot_product(row_slice, vector);
    }
}

/// Vector-matrix multiplication using SIMD.
///
/// Computes y = x^T * A where x is an m-vector and A is an m×n matrix.
/// The matrix is stored in row-major order.
///
/// # Arguments
///
/// * `vector` - Input vector (m elements)
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Result vector (n elements)
#[inline]
pub fn vector_matrix_mul(vector: &[f64], matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");
    debug_assert_eq!(vector.len(), m, "Vector must have m elements");

    let mut result = vec![0.0; n];
    let chunks = n / 4;
    let remainder = n % 4;

    // For each row, add vector[row] * row_data to result
    for row in 0..m {
        let row_start = row * n;
        let scale = vector[row];
        let scale_vec = f64x4::splat(scale);

        // SIMD portion
        for c in 0..chunks {
            let col = c * 4;
            let mat_vec = f64x4::new([
                matrix[row_start + col],
                matrix[row_start + col + 1],
                matrix[row_start + col + 2],
                matrix[row_start + col + 3],
            ]);
            let res_vec = f64x4::new([
                result[col],
                result[col + 1],
                result[col + 2],
                result[col + 3],
            ]);
            let sum_vec = res_vec + scale_vec * mat_vec;
            let sum_array: [f64; 4] = sum_vec.into();
            result[col] = sum_array[0];
            result[col + 1] = sum_array[1];
            result[col + 2] = sum_array[2];
            result[col + 3] = sum_array[3];
        }

        // Remainder
        let rem_start = chunks * 4;
        for c in 0..remainder {
            result[rem_start + c] += scale * matrix[row_start + rem_start + c];
        }
    }

    result
}

// =============================================================================
// Element-wise Vector Operations
// =============================================================================

/// Add a scalar to every element of a vector using SIMD.
///
/// # Arguments
///
/// * `a` - Input vector
/// * `scalar` - Scalar to add
///
/// # Returns
///
/// New vector with scalar added to each element.
#[inline]
pub fn vector_add_scalar(a: &[f64], scalar: f64) -> Vec<f64> {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = vec![0.0; n];
    let scalar_vec = f64x4::splat(scalar);

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let sum_vec = a_vec + scalar_vec;
        let sum_array: [f64; 4] = sum_vec.into();
        result[offset] = sum_array[0];
        result[offset + 1] = sum_array[1];
        result[offset + 2] = sum_array[2];
        result[offset + 3] = sum_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result[remainder_start + i] = a[remainder_start + i] + scalar;
    }

    result
}

/// Multiply every element of a vector by a scalar using SIMD.
///
/// # Arguments
///
/// * `a` - Input vector
/// * `scalar` - Scalar to multiply by
///
/// # Returns
///
/// New vector with each element multiplied by scalar.
#[inline]
pub fn vector_mul_scalar(a: &[f64], scalar: f64) -> Vec<f64> {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = vec![0.0; n];
    let scalar_vec = f64x4::splat(scalar);

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let prod_vec = a_vec * scalar_vec;
        let prod_array: [f64; 4] = prod_vec.into();
        result[offset] = prod_array[0];
        result[offset + 1] = prod_array[1];
        result[offset + 2] = prod_array[2];
        result[offset + 3] = prod_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result[remainder_start + i] = a[remainder_start + i] * scalar;
    }

    result
}

/// Element-wise addition of two vectors using SIMD.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// New vector where `result[i] = a[i] + b[i]`.
#[inline]
pub fn vector_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = vec![0.0; n];

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let sum_vec = a_vec + b_vec;
        let sum_array: [f64; 4] = sum_vec.into();
        result[offset] = sum_array[0];
        result[offset + 1] = sum_array[1];
        result[offset + 2] = sum_array[2];
        result[offset + 3] = sum_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result[remainder_start + i] = a[remainder_start + i] + b[remainder_start + i];
    }

    result
}

/// Element-wise subtraction of two vectors using SIMD.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// New vector where `result[i] = a[i] - b[i]`.
#[inline]
pub fn vector_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = vec![0.0; n];

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let diff_vec = a_vec - b_vec;
        let diff_array: [f64; 4] = diff_vec.into();
        result[offset] = diff_array[0];
        result[offset + 1] = diff_array[1];
        result[offset + 2] = diff_array[2];
        result[offset + 3] = diff_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result[remainder_start + i] = a[remainder_start + i] - b[remainder_start + i];
    }

    result
}

/// Element-wise subtraction with destination using SIMD.
///
/// Computes `dst[i] = a[i] - b[i]` in-place, avoiding allocation.
///
/// # Arguments
///
/// * `a` - First vector (minuend)
/// * `b` - Second vector (subtrahend, must have same length as `a`)
/// * `dst` - Destination slice (must have same length as `a`)
///
/// # Panics
///
/// Debug panics if vectors have different lengths.
#[inline]
pub fn vector_sub_into(a: &[f64], b: &[f64], dst: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    debug_assert_eq!(a.len(), dst.len(), "Destination must have same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let diff_vec = a_vec - b_vec;
        let diff_array: [f64; 4] = diff_vec.into();
        dst[offset] = diff_array[0];
        dst[offset + 1] = diff_array[1];
        dst[offset + 2] = diff_array[2];
        dst[offset + 3] = diff_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        dst[remainder_start + i] = a[remainder_start + i] - b[remainder_start + i];
    }
}

/// Element-wise multiplication (Hadamard product) of two vectors using SIMD.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
///
/// New vector where `result[i] = a[i] * b[i]`.
#[inline]
pub fn vector_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = vec![0.0; n];

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let prod_vec = a_vec * b_vec;
        let prod_array: [f64; 4] = prod_vec.into();
        result[offset] = prod_array[0];
        result[offset + 1] = prod_array[1];
        result[offset + 2] = prod_array[2];
        result[offset + 3] = prod_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result[remainder_start + i] = a[remainder_start + i] * b[remainder_start + i];
    }

    result
}

/// Element-wise division of two vectors using SIMD.
///
/// # Arguments
///
/// * `a` - Numerator vector
/// * `b` - Denominator vector (must have same length as `a`)
///
/// # Returns
///
/// New vector where `result[i] = a[i] / b[i]`.
///
/// # Note
///
/// Division by zero will produce infinity or NaN as per IEEE 754.
#[inline]
pub fn vector_div(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = vec![0.0; n];

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        let b_vec = f64x4::new([b[offset], b[offset + 1], b[offset + 2], b[offset + 3]]);
        let div_vec = a_vec / b_vec;
        let div_array: [f64; 4] = div_vec.into();
        result[offset] = div_array[0];
        result[offset + 1] = div_array[1];
        result[offset + 2] = div_array[2];
        result[offset + 3] = div_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result[remainder_start + i] = a[remainder_start + i] / b[remainder_start + i];
    }

    result
}

/// AXPY operation: y = a * x + y (BLAS-like) using SIMD.
///
/// This is a common operation in linear algebra that modifies `y` in place.
///
/// # Arguments
///
/// * `a` - Scalar multiplier
/// * `x` - Input vector
/// * `y` - Vector to be modified in place (must have same length as `x`)
#[inline]
pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len(), "Vectors must have the same length");

    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let a_vec = f64x4::splat(a);

    for i in 0..chunks {
        let offset = i * 4;
        let x_vec = f64x4::new([x[offset], x[offset + 1], x[offset + 2], x[offset + 3]]);
        let y_vec = f64x4::new([y[offset], y[offset + 1], y[offset + 2], y[offset + 3]]);
        let result_vec = a_vec * x_vec + y_vec;
        let result_array: [f64; 4] = result_vec.into();
        y[offset] = result_array[0];
        y[offset + 1] = result_array[1];
        y[offset + 2] = result_array[2];
        y[offset + 3] = result_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        y[remainder_start + i] += a * x[remainder_start + i];
    }
}

/// AXPBY operation: y = a * x + b * y using SIMD.
///
/// A generalized version of AXPY where both vectors are scaled.
///
/// # Arguments
///
/// * `a` - Scalar multiplier for x
/// * `x` - First input vector
/// * `b` - Scalar multiplier for y
/// * `y` - Vector to be modified in place (must have same length as `x`)
#[inline]
pub fn axpby(a: f64, x: &[f64], b: f64, y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len(), "Vectors must have the same length");

    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let a_vec = f64x4::splat(a);
    let b_vec = f64x4::splat(b);

    for i in 0..chunks {
        let offset = i * 4;
        let x_vec = f64x4::new([x[offset], x[offset + 1], x[offset + 2], x[offset + 3]]);
        let y_vec = f64x4::new([y[offset], y[offset + 1], y[offset + 2], y[offset + 3]]);
        let result_vec = a_vec * x_vec + b_vec * y_vec;
        let result_array: [f64; 4] = result_vec.into();
        y[offset] = result_array[0];
        y[offset + 1] = result_array[1];
        y[offset + 2] = result_array[2];
        y[offset + 3] = result_array[3];
    }

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        y[remainder_start + i] = a * x[remainder_start + i] + b * y[remainder_start + i];
    }
}

// =============================================================================
// Matrix Row/Column Operations
// =============================================================================

/// Sum each row of a matrix using SIMD.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Vector of row sums (m elements)
#[inline]
pub fn sum_rows(matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");

    let mut result = Vec::with_capacity(m);

    for row in 0..m {
        let row_start = row * n;
        let row_slice = &matrix[row_start..row_start + n];
        result.push(sum(row_slice));
    }

    result
}

/// Sum each column of a matrix using SIMD.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Vector of column sums (n elements)
#[inline]
pub fn sum_cols(matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");

    let mut result = vec![0.0; n];
    let chunks = n / 4;
    let remainder = n % 4;

    for row in 0..m {
        let row_start = row * n;

        // SIMD portion
        for c in 0..chunks {
            let col = c * 4;
            let mat_vec = f64x4::new([
                matrix[row_start + col],
                matrix[row_start + col + 1],
                matrix[row_start + col + 2],
                matrix[row_start + col + 3],
            ]);
            let res_vec = f64x4::new([
                result[col],
                result[col + 1],
                result[col + 2],
                result[col + 3],
            ]);
            let sum_vec = res_vec + mat_vec;
            let sum_array: [f64; 4] = sum_vec.into();
            result[col] = sum_array[0];
            result[col + 1] = sum_array[1];
            result[col + 2] = sum_array[2];
            result[col + 3] = sum_array[3];
        }

        // Remainder
        let rem_start = chunks * 4;
        for c in 0..remainder {
            result[rem_start + c] += matrix[row_start + rem_start + c];
        }
    }

    result
}

/// Compute the L2 norm of each row of a matrix using SIMD.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Vector of row L2 norms (m elements)
#[inline]
pub fn row_norms_l2(matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");

    let mut result = Vec::with_capacity(m);

    for row in 0..m {
        let row_start = row * n;
        let row_slice = &matrix[row_start..row_start + n];
        result.push(sum_of_squares(row_slice).sqrt());
    }

    result
}

/// Compute the L1 norm (sum of absolute values) of each row of a matrix using SIMD.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Vector of row L1 norms (m elements)
#[inline]
pub fn row_norms_l1(matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");

    let mut result = Vec::with_capacity(m);

    for row in 0..m {
        let row_start = row * n;
        let row_slice = &matrix[row_start..row_start + n];
        result.push(sum_abs(row_slice));
    }

    result
}

/// Compute the sum of absolute values in a vector using SIMD.
///
/// # Arguments
///
/// * `a` - Input vector
///
/// # Returns
///
/// Sum of `|a[i]|` for all `i`.
#[inline]
pub fn sum_abs(a: &[f64]) -> f64 {
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum_vec = f64x4::ZERO;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = f64x4::new([a[offset], a[offset + 1], a[offset + 2], a[offset + 3]]);
        sum_vec = sum_vec + a_vec.abs();
    }

    let sum_array: [f64; 4] = sum_vec.into();
    let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

    let remainder_start = chunks * 4;
    for i in 0..remainder {
        result += a[remainder_start + i].abs();
    }

    result
}

/// Normalize each row of a matrix to unit L2 norm using SIMD.
///
/// Rows with zero norm are left unchanged.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements), modified in place
/// * `m` - Number of rows
/// * `n` - Number of columns
#[inline]
pub fn normalize_rows_l2(matrix: &mut [f64], m: usize, n: usize) {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");

    let chunks = n / 4;
    let remainder = n % 4;

    for row in 0..m {
        let row_start = row * n;
        let row_slice = &matrix[row_start..row_start + n];
        let norm = sum_of_squares(row_slice).sqrt();

        if norm > 1e-10 {
            let inv_norm = 1.0 / norm;
            let inv_norm_vec = f64x4::splat(inv_norm);

            // SIMD portion
            for c in 0..chunks {
                let col = c * 4;
                let idx = row_start + col;
                let mat_vec = f64x4::new([
                    matrix[idx],
                    matrix[idx + 1],
                    matrix[idx + 2],
                    matrix[idx + 3],
                ]);
                let norm_vec = mat_vec * inv_norm_vec;
                let norm_array: [f64; 4] = norm_vec.into();
                matrix[idx] = norm_array[0];
                matrix[idx + 1] = norm_array[1];
                matrix[idx + 2] = norm_array[2];
                matrix[idx + 3] = norm_array[3];
            }

            // Remainder
            let rem_start = row_start + chunks * 4;
            for c in 0..remainder {
                matrix[rem_start + c] *= inv_norm;
            }
        }
    }
}

/// Compute the mean of each row of a matrix using SIMD.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Vector of row means (m elements)
#[inline]
pub fn row_means(matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");

    let mut result = Vec::with_capacity(m);
    let n_f64 = n as f64;

    for row in 0..m {
        let row_start = row * n;
        let row_slice = &matrix[row_start..row_start + n];
        result.push(sum(row_slice) / n_f64);
    }

    result
}

/// Compute the mean of each column of a matrix using SIMD.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Returns
///
/// Vector of column means (n elements)
#[inline]
pub fn col_means(matrix: &[f64], m: usize, n: usize) -> Vec<f64> {
    let sums = sum_cols(matrix, m, n);
    let m_f64 = m as f64;
    sums.into_iter().map(|s| s / m_f64).collect()
}

/// Center each row of a matrix (subtract row mean) using SIMD.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements), modified in place
/// * `m` - Number of rows
/// * `n` - Number of columns
#[inline]
pub fn center_rows(matrix: &mut [f64], m: usize, n: usize) {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");

    let chunks = n / 4;
    let remainder = n % 4;

    for row in 0..m {
        let row_start = row * n;
        let row_slice = &matrix[row_start..row_start + n];
        let mean = sum(row_slice) / n as f64;
        let mean_vec = f64x4::splat(mean);

        // SIMD portion
        for c in 0..chunks {
            let col = c * 4;
            let idx = row_start + col;
            let mat_vec = f64x4::new([
                matrix[idx],
                matrix[idx + 1],
                matrix[idx + 2],
                matrix[idx + 3],
            ]);
            let centered = mat_vec - mean_vec;
            let centered_array: [f64; 4] = centered.into();
            matrix[idx] = centered_array[0];
            matrix[idx + 1] = centered_array[1];
            matrix[idx + 2] = centered_array[2];
            matrix[idx + 3] = centered_array[3];
        }

        // Remainder
        let rem_start = row_start + chunks * 4;
        for c in 0..remainder {
            matrix[rem_start + c] -= mean;
        }
    }
}

/// Scale each row by its standard deviation using SIMD.
///
/// Computes row variance, then divides each element by sqrt(variance).
/// Rows with zero variance are left unchanged.
///
/// # Arguments
///
/// * `matrix` - Matrix in row-major order (m × n elements), modified in place
/// * `m` - Number of rows
/// * `n` - Number of columns
#[inline]
pub fn scale_rows_by_std(matrix: &mut [f64], m: usize, n: usize) {
    debug_assert_eq!(matrix.len(), m * n, "Matrix must have m * n elements");

    let chunks = n / 4;
    let remainder = n % 4;
    let n_f64 = n as f64;

    for row in 0..m {
        let row_start = row * n;

        // Compute mean
        let row_slice = &matrix[row_start..row_start + n];
        let mean = sum(row_slice) / n_f64;

        // Compute variance using SIMD
        let mean_vec = f64x4::splat(mean);
        let mut var_sum = f64x4::ZERO;

        for c in 0..chunks {
            let col = c * 4;
            let idx = row_start + col;
            let mat_vec = f64x4::new([
                matrix[idx],
                matrix[idx + 1],
                matrix[idx + 2],
                matrix[idx + 3],
            ]);
            let diff = mat_vec - mean_vec;
            var_sum = var_sum + diff * diff;
        }

        let var_array: [f64; 4] = var_sum.into();
        let mut variance = var_array[0] + var_array[1] + var_array[2] + var_array[3];

        // Remainder for variance
        let rem_start = row_start + chunks * 4;
        for c in 0..remainder {
            let diff = matrix[rem_start + c] - mean;
            variance += diff * diff;
        }

        variance /= n_f64;
        let std_dev = variance.sqrt();

        if std_dev > 1e-10 {
            let inv_std = 1.0 / std_dev;
            let inv_std_vec = f64x4::splat(inv_std);

            // Scale using SIMD
            for c in 0..chunks {
                let col = c * 4;
                let idx = row_start + col;
                let mat_vec = f64x4::new([
                    matrix[idx],
                    matrix[idx + 1],
                    matrix[idx + 2],
                    matrix[idx + 3],
                ]);
                let scaled = mat_vec * inv_std_vec;
                let scaled_array: [f64; 4] = scaled.into();
                matrix[idx] = scaled_array[0];
                matrix[idx + 1] = scaled_array[1];
                matrix[idx + 2] = scaled_array[2];
                matrix[idx + 3] = scaled_array[3];
            }

            // Remainder for scaling
            for c in 0..remainder {
                matrix[rem_start + c] *= inv_std;
            }
        }
    }
}

/// Outer product of two vectors using SIMD: result = a * b^T
///
/// Creates an m×n matrix where `result[i,j] = a[i] * b[j]`.
///
/// # Arguments
///
/// * `a` - First vector (m elements)
/// * `b` - Second vector (n elements)
///
/// # Returns
///
/// Matrix in row-major order (m × n elements)
#[inline]
pub fn outer_product(a: &[f64], b: &[f64]) -> Vec<f64> {
    let m = a.len();
    let n = b.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut result = vec![0.0; m * n];

    for row in 0..m {
        let scale = a[row];
        let scale_vec = f64x4::splat(scale);
        let row_start = row * n;

        // SIMD portion
        for c in 0..chunks {
            let col = c * 4;
            let b_vec = f64x4::new([b[col], b[col + 1], b[col + 2], b[col + 3]]);
            let prod = scale_vec * b_vec;
            let prod_array: [f64; 4] = prod.into();
            result[row_start + col] = prod_array[0];
            result[row_start + col + 1] = prod_array[1];
            result[row_start + col + 2] = prod_array[2];
            result[row_start + col + 3] = prod_array[3];
        }

        // Remainder
        let rem_start = chunks * 4;
        for c in 0..remainder {
            result[row_start + rem_start + c] = scale * b[rem_start + c];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_squared_euclidean_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((squared_euclidean_distance(&a, &b) - 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_euclidean_distance_8d() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        // Sum of squared differences = 8 * 1 = 8
        let expected = 8.0_f64.sqrt();
        assert!((euclidean_distance(&a, &b) - expected).abs() < EPSILON);
    }

    #[test]
    fn test_euclidean_distance_remainder() {
        // 5 elements to test remainder handling
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = 5.0_f64.sqrt();
        assert!((euclidean_distance(&a, &b) - expected).abs() < EPSILON);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((manhattan_distance(&a, &b) - 7.0).abs() < EPSILON);
    }

    #[test]
    fn test_manhattan_distance_8d() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert!((manhattan_distance(&a, &b) - 8.0).abs() < EPSILON);
    }

    #[test]
    fn test_manhattan_distance_remainder() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert!((manhattan_distance(&a, &b) - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((dot_product(&a, &b) - 70.0).abs() < EPSILON);
    }

    #[test]
    fn test_dot_product_remainder() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        assert!((dot_product(&a, &b) - 70.0).abs() < EPSILON);
    }

    #[test]
    fn test_sum() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!((sum(&a) - 36.0).abs() < EPSILON);
    }

    #[test]
    fn test_sum_remainder() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sum(&a) - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_sum_of_squares() {
        let a = [1.0, 2.0, 3.0, 4.0];
        // 1 + 4 + 9 + 16 = 30
        assert!((sum_of_squares(&a) - 30.0).abs() < EPSILON);
    }

    #[test]
    fn test_squared_differences() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 3.0, 4.0, 5.0];
        let result = squared_differences(&a, &b);
        assert_eq!(result.len(), 4);
        for val in result {
            assert!((val - 1.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_minkowski_p1() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        // p=1 should equal Manhattan
        assert!((minkowski_distance(&a, &b, 1.0) - 7.0).abs() < EPSILON);
    }

    #[test]
    fn test_minkowski_p2() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        // p=2 should equal Euclidean
        assert!((minkowski_distance(&a, &b, 2.0) - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_minkowski_p3() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        // (3^3 + 4^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3) ≈ 4.4979
        let expected = 91.0_f64.powf(1.0 / 3.0);
        assert!((minkowski_distance(&a, &b, 3.0) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [-1.0, -2.0, -3.0, -4.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!(cosine_distance(&a, &a).abs() < EPSILON);
    }

    #[test]
    fn test_batch_squared_euclidean() {
        let query = [0.0, 0.0];
        let references = [
            1.0, 0.0, // distance^2 = 1
            0.0, 2.0, // distance^2 = 4
            3.0, 4.0, // distance^2 = 25
        ];

        let distances = batch_squared_euclidean(&query, &references, 3, 2);
        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 1.0).abs() < EPSILON);
        assert!((distances[1] - 4.0).abs() < EPSILON);
        assert!((distances[2] - 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_empty_vectors() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];

        assert!(euclidean_distance(&a, &b).abs() < EPSILON);
        assert!(manhattan_distance(&a, &b).abs() < EPSILON);
        assert!(dot_product(&a, &b).abs() < EPSILON);
        assert!(sum(&a).abs() < EPSILON);
    }

    #[test]
    fn test_zero_vectors() {
        let a = [0.0, 0.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0, 0.0];

        assert!(euclidean_distance(&a, &b).abs() < EPSILON);
        assert!(manhattan_distance(&a, &b).abs() < EPSILON);
    }

    #[test]
    fn test_large_vector() {
        let n = 1000;
        let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        // Each difference is 1, so squared_euclidean = n, euclidean = sqrt(n)
        let expected_sq = n as f64;
        let expected = expected_sq.sqrt();

        assert!((squared_euclidean_distance(&a, &b) - expected_sq).abs() < EPSILON);
        assert!((euclidean_distance(&a, &b) - expected).abs() < 1e-6);
        assert!((manhattan_distance(&a, &b) - n as f64).abs() < EPSILON);
    }

    // Consistency tests comparing SIMD with scalar implementations
    #[test]
    fn test_simd_scalar_consistency_euclidean() {
        let a: [f64; 9] = [1.5, 2.7, 3.1, 4.9, 5.2, 6.8, 7.3, 8.1, 9.0];
        let b: [f64; 9] = [2.1, 3.4, 4.0, 5.5, 6.1, 7.2, 8.0, 9.1, 10.5];

        // Scalar computation
        let scalar_result: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi): (&f64, &f64)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt();

        let simd_result = euclidean_distance(&a, &b);

        assert!((simd_result - scalar_result).abs() < 1e-10);
    }

    #[test]
    fn test_simd_scalar_consistency_manhattan() {
        let a: [f64; 9] = [1.5, 2.7, 3.1, 4.9, 5.2, 6.8, 7.3, 8.1, 9.0];
        let b: [f64; 9] = [2.1, 3.4, 4.0, 5.5, 6.1, 7.2, 8.0, 9.1, 10.5];

        // Scalar computation
        let scalar_result: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi): (&f64, &f64)| (ai - bi).abs())
            .sum();

        let simd_result = manhattan_distance(&a, &b);

        assert!((simd_result - scalar_result).abs() < 1e-10);
    }

    // ==========================================================================
    // Matrix Operations Tests
    // ==========================================================================

    #[test]
    fn test_matrix_vector_mul_2x3() {
        // [[1, 2, 3], [4, 5, 6]] * [1, 2, 3] = [14, 32]
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = [1.0, 2.0, 3.0];
        let result = matrix_vector_mul(&matrix, &vector, 2, 3);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < EPSILON);
        assert!((result[1] - 32.0).abs() < EPSILON);
    }

    #[test]
    fn test_matrix_vector_mul_identity() {
        // Identity matrix times any vector equals that vector
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let vector = [5.0, 7.0, 9.0];
        let result = matrix_vector_mul(&identity, &vector, 3, 3);

        assert_eq!(result.len(), 3);
        for i in 0..3 {
            assert!((result[i] - vector[i]).abs() < EPSILON);
        }
    }

    #[test]
    fn test_matrix_vector_mul_into() {
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vector = [1.0, 2.0, 3.0];
        let mut result = [0.0; 2];
        matrix_vector_mul_into(&matrix, &vector, &mut result, 2, 3);

        assert!((result[0] - 14.0).abs() < EPSILON);
        assert!((result[1] - 32.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_matrix_mul() {
        // [1, 2] * [[1, 2, 3], [4, 5, 6]] = [9, 12, 15]
        let vector = [1.0, 2.0];
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = vector_matrix_mul(&vector, &matrix, 2, 3);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 9.0).abs() < EPSILON);
        assert!((result[1] - 12.0).abs() < EPSILON);
        assert!((result[2] - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_matrix_mul_large() {
        // 1x8 * 1x8 matrix (single row)
        let vector = [2.0];
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = vector_matrix_mul(&vector, &matrix, 1, 8);

        assert_eq!(result.len(), 8);
        for i in 0..8 {
            assert!((result[i] - 2.0 * (i as f64 + 1.0)).abs() < EPSILON);
        }
    }

    // ==========================================================================
    // Element-wise Vector Operations Tests
    // ==========================================================================

    #[test]
    fn test_vector_add_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = vector_add_scalar(&a, 10.0);

        assert_eq!(result.len(), 5);
        for i in 0..5 {
            assert!((result[i] - (a[i] + 10.0)).abs() < EPSILON);
        }
    }

    #[test]
    fn test_vector_mul_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = vector_mul_scalar(&a, 3.0);

        assert_eq!(result.len(), 8);
        for i in 0..8 {
            assert!((result[i] - (a[i] * 3.0)).abs() < EPSILON);
        }
    }

    #[test]
    fn test_vector_add() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        let result = vector_add(&a, &b);

        assert_eq!(result.len(), 5);
        for i in 0..5 {
            assert!((result[i] - 6.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_vector_sub() {
        let a = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = vector_sub(&a, &b);

        assert_eq!(result.len(), 8);
        for i in 0..8 {
            assert!((result[i] - 4.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_vector_mul() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 3.0, 4.0, 5.0];
        let result = vector_mul(&a, &b);

        assert_eq!(result.len(), 4);
        assert!((result[0] - 2.0).abs() < EPSILON);
        assert!((result[1] - 6.0).abs() < EPSILON);
        assert!((result[2] - 12.0).abs() < EPSILON);
        assert!((result[3] - 20.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_div() {
        let a = [10.0, 20.0, 30.0, 40.0, 50.0];
        let b = [2.0, 4.0, 5.0, 8.0, 10.0];
        let result = vector_div(&a, &b);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 5.0).abs() < EPSILON);
        assert!((result[1] - 5.0).abs() < EPSILON);
        assert!((result[2] - 6.0).abs() < EPSILON);
        assert!((result[3] - 5.0).abs() < EPSILON);
        assert!((result[4] - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_axpy() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = [10.0, 20.0, 30.0, 40.0, 50.0];
        axpy(2.0, &x, &mut y);

        // y = 2 * x + y = [12, 24, 36, 48, 60]
        assert!((y[0] - 12.0).abs() < EPSILON);
        assert!((y[1] - 24.0).abs() < EPSILON);
        assert!((y[2] - 36.0).abs() < EPSILON);
        assert!((y[3] - 48.0).abs() < EPSILON);
        assert!((y[4] - 60.0).abs() < EPSILON);
    }

    #[test]
    fn test_axpy_large() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut y: Vec<f64> = (0..100).map(|i| (i * 2) as f64).collect();
        let y_original: Vec<f64> = y.clone();

        axpy(3.0, &x, &mut y);

        for i in 0..100 {
            let expected = 3.0 * x[i] + y_original[i];
            assert!((y[i] - expected).abs() < EPSILON);
        }
    }

    #[test]
    fn test_axpby() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut y = [10.0, 20.0, 30.0, 40.0];

        axpby(2.0, &x, 0.5, &mut y);

        // y = 2 * x + 0.5 * y = [7, 14, 21, 28]
        assert!((y[0] - 7.0).abs() < EPSILON);
        assert!((y[1] - 14.0).abs() < EPSILON);
        assert!((y[2] - 21.0).abs() < EPSILON);
        assert!((y[3] - 28.0).abs() < EPSILON);
    }

    // ==========================================================================
    // Matrix Row/Column Operations Tests
    // ==========================================================================

    #[test]
    fn test_sum_rows() {
        // [[1, 2, 3], [4, 5, 6]] -> [6, 15]
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = sum_rows(&matrix, 2, 3);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < EPSILON);
        assert!((result[1] - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_sum_cols() {
        // [[1, 2, 3], [4, 5, 6]] -> [5, 7, 9]
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = sum_cols(&matrix, 2, 3);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < EPSILON);
        assert!((result[1] - 7.0).abs() < EPSILON);
        assert!((result[2] - 9.0).abs() < EPSILON);
    }

    #[test]
    fn test_sum_cols_large() {
        // 3x8 matrix
        let matrix: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let result = sum_cols(&matrix, 3, 8);

        assert_eq!(result.len(), 8);
        // Column 0: 1 + 9 + 17 = 27
        assert!((result[0] - 27.0).abs() < EPSILON);
        // Column 7: 8 + 16 + 24 = 48
        assert!((result[7] - 48.0).abs() < EPSILON);
    }

    #[test]
    fn test_row_norms_l2() {
        // [[3, 4], [5, 12]] -> [5, 13]
        let matrix = [3.0, 4.0, 5.0, 12.0];
        let result = row_norms_l2(&matrix, 2, 2);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 5.0).abs() < EPSILON);
        assert!((result[1] - 13.0).abs() < EPSILON);
    }

    #[test]
    fn test_row_norms_l1() {
        // [[1, -2, 3], [-4, 5, -6]] -> [6, 15]
        let matrix = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let result = row_norms_l1(&matrix, 2, 3);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < EPSILON);
        assert!((result[1] - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_sum_abs() {
        let a = [1.0, -2.0, 3.0, -4.0, 5.0];
        assert!((sum_abs(&a) - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_rows_l2() {
        // [[3, 4], [0, 0], [5, 12]] -> [[0.6, 0.8], [0, 0], [5/13, 12/13]]
        let mut matrix = [3.0, 4.0, 0.0, 0.0, 5.0, 12.0];
        normalize_rows_l2(&mut matrix, 3, 2);

        assert!((matrix[0] - 0.6).abs() < 1e-6);
        assert!((matrix[1] - 0.8).abs() < 1e-6);
        assert!((matrix[2] - 0.0).abs() < EPSILON); // zero row unchanged
        assert!((matrix[3] - 0.0).abs() < EPSILON);
        assert!((matrix[4] - 5.0 / 13.0).abs() < 1e-6);
        assert!((matrix[5] - 12.0 / 13.0).abs() < 1e-6);
    }

    #[test]
    fn test_row_means() {
        // [[1, 2, 3], [4, 5, 6]] -> [2, 5]
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = row_means(&matrix, 2, 3);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 2.0).abs() < EPSILON);
        assert!((result[1] - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_col_means() {
        // [[1, 2, 3], [4, 5, 6]] -> [2.5, 3.5, 4.5]
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = col_means(&matrix, 2, 3);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.5).abs() < EPSILON);
        assert!((result[1] - 3.5).abs() < EPSILON);
        assert!((result[2] - 4.5).abs() < EPSILON);
    }

    #[test]
    fn test_center_rows() {
        // [[1, 2, 3], [4, 5, 6]] with means [2, 5]
        // -> [[-1, 0, 1], [-1, 0, 1]]
        let mut matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        center_rows(&mut matrix, 2, 3);

        assert!((matrix[0] - (-1.0)).abs() < EPSILON);
        assert!((matrix[1] - 0.0).abs() < EPSILON);
        assert!((matrix[2] - 1.0).abs() < EPSILON);
        assert!((matrix[3] - (-1.0)).abs() < EPSILON);
        assert!((matrix[4] - 0.0).abs() < EPSILON);
        assert!((matrix[5] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_scale_rows_by_std() {
        // Row [1, 3] has mean=2, variance=1, std=1, so scaled = [1, 3]
        // Row [2, 4, 6] has mean=4, variance=8/3, std~=1.633
        let mut matrix = [1.0, 3.0, 2.0, 4.0, 6.0, 0.0]; // 2x3 with padding
        scale_rows_by_std(&mut matrix[0..2], 1, 2);

        // std of [1, 3] is 1, so values unchanged
        assert!((matrix[0] - 1.0).abs() < EPSILON);
        assert!((matrix[1] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_outer_product() {
        // [1, 2] outer [3, 4, 5] = [[3, 4, 5], [6, 8, 10]]
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        let result = outer_product(&a, &b);

        assert_eq!(result.len(), 6);
        assert!((result[0] - 3.0).abs() < EPSILON);
        assert!((result[1] - 4.0).abs() < EPSILON);
        assert!((result[2] - 5.0).abs() < EPSILON);
        assert!((result[3] - 6.0).abs() < EPSILON);
        assert!((result[4] - 8.0).abs() < EPSILON);
        assert!((result[5] - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_outer_product_large() {
        let a: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let result = outer_product(&a, &b);

        assert_eq!(result.len(), 80);
        // result[i*8 + j] = a[i] * b[j]
        for i in 0..10 {
            for j in 0..8 {
                let expected = (i + 1) as f64 * (j + 1) as f64;
                assert!((result[i * 8 + j] - expected).abs() < EPSILON);
            }
        }
    }

    // ==========================================================================
    // Consistency Tests (SIMD vs Scalar)
    // ==========================================================================

    #[test]
    fn test_matrix_vector_mul_consistency() {
        let matrix: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let vector: Vec<f64> = (1..=6).map(|x| x as f64).collect();

        let simd_result = matrix_vector_mul(&matrix, &vector, 4, 6);

        // Scalar computation
        let mut scalar_result = vec![0.0; 4];
        for row in 0..4 {
            for col in 0..6 {
                scalar_result[row] += matrix[row * 6 + col] * vector[col];
            }
        }

        for i in 0..4 {
            assert!((simd_result[i] - scalar_result[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_vector_operations_consistency() {
        let a: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let b: Vec<f64> = (101..=200).map(|x| x as f64).collect();

        // Test vector_add
        let add_result = vector_add(&a, &b);
        for i in 0..100 {
            assert!((add_result[i] - (a[i] + b[i])).abs() < EPSILON);
        }

        // Test vector_sub
        let sub_result = vector_sub(&b, &a);
        for i in 0..100 {
            assert!((sub_result[i] - (b[i] - a[i])).abs() < EPSILON);
        }

        // Test vector_sub_into
        let mut dst = vec![0.0; 100];
        vector_sub_into(&b, &a, &mut dst);
        for i in 0..100 {
            assert!((dst[i] - (b[i] - a[i])).abs() < EPSILON);
        }

        // Test vector_mul
        let mul_result = vector_mul(&a, &b);
        for i in 0..100 {
            assert!((mul_result[i] - (a[i] * b[i])).abs() < EPSILON);
        }

        // Test vector_div
        let div_result = vector_div(&b, &a);
        for i in 0..100 {
            assert!((div_result[i] - (b[i] / a[i])).abs() < 1e-10);
        }
    }
}
