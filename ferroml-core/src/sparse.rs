//! Native Sparse Matrix Support
//!
//! This module provides native sparse matrix operations for FerroML, enabling
//! efficient handling of high-dimensional sparse data (e.g., text/NLP features,
//! one-hot encoded categorical features, etc.).
//!
//! ## Features
//!
//! Enable this module by adding the `sparse` feature to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! ferroml-core = { version = "0.1", features = ["sparse"] }
//! ```
//!
//! ## Supported Formats
//!
//! - **CSR (Compressed Sparse Row)**: Row-oriented, efficient for row slicing
//!   and matrix-vector products. Best for most ML algorithms.
//! - **CSC (Compressed Sparse Column)**: Column-oriented, efficient for column
//!   operations. Useful for feature selection.
//!
//! ## Performance Benefits
//!
//! For sparse data (e.g., >90% zeros), native sparse operations provide:
//! - **Memory**: O(nnz) instead of O(n×m) for dense
//! - **Speed**: Skip zero elements in distance calculations
//! - **KNN**: 10-100x faster on text/NLP data
//!
//! ## Example
//!
//! ```
//! use ferroml_core::sparse::{CsrMatrix, sparse_euclidean_distance};
//! use ndarray::array;
//!
//! // Create sparse matrix from dense data
//! let dense = array![[1.0, 0.0, 0.0, 2.0], [0.0, 3.0, 0.0, 0.0]];
//! let sparse = CsrMatrix::from_dense(&dense);
//!
//! // Compute sparse distance
//! let row0 = sparse.row(0);
//! let row1 = sparse.row(1);
//! let dist = sparse_euclidean_distance(&row0, &row1);
//! ```

use ndarray::{Array1, Array2};
use sprs::{CsMat, CsMatView, CsVec, CsVecView, TriMat};

use crate::{FerroError, Result};

// =============================================================================
// CSR Matrix Wrapper
// =============================================================================

/// A Compressed Sparse Row (CSR) matrix optimized for machine learning operations.
///
/// CSR format stores:
/// - `data`: Non-zero values in row-major order
/// - `indices`: Column indices for each non-zero value
/// - `indptr`: Row pointers indicating where each row starts in data/indices
///
/// This format is efficient for:
/// - Row slicing: O(1) to get a row
/// - Matrix-vector products: O(nnz)
/// - Distance calculations between rows
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    inner: CsMat<f64>,
}

impl CsrMatrix {
    /// Create a new CSR matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `shape` - (rows, cols) tuple
    /// * `indptr` - Row pointers (length = rows + 1)
    /// * `indices` - Column indices for non-zero values
    /// * `data` - Non-zero values
    ///
    /// # Returns
    ///
    /// Result containing the CSR matrix or an error if components are invalid.
    pub fn new(
        shape: (usize, usize),
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Result<Self> {
        let mat = CsMat::new(shape, indptr, indices, data);
        Ok(Self { inner: mat })
    }

    /// Create a CSR matrix from a dense ndarray.
    ///
    /// Values below `threshold` (default: 0.0) are considered zeros.
    pub fn from_dense(dense: &Array2<f64>) -> Self {
        Self::from_dense_with_threshold(dense, 0.0)
    }

    /// Create a CSR matrix from a dense ndarray with custom zero threshold.
    pub fn from_dense_with_threshold(dense: &Array2<f64>, threshold: f64) -> Self {
        let (nrows, ncols) = dense.dim();
        let mut tri = TriMat::new((nrows, ncols));

        for i in 0..nrows {
            for j in 0..ncols {
                let val = dense[[i, j]];
                if val.abs() > threshold {
                    tri.add_triplet(i, j, val);
                }
            }
        }

        Self {
            inner: tri.to_csr(),
        }
    }

    /// Create a CSR matrix from coordinate (COO) format triplets.
    ///
    /// # Arguments
    ///
    /// * `shape` - (rows, cols) tuple
    /// * `rows` - Row indices
    /// * `cols` - Column indices
    /// * `values` - Non-zero values
    pub fn from_triplets(
        shape: (usize, usize),
        rows: &[usize],
        cols: &[usize],
        values: &[f64],
    ) -> Result<Self> {
        if rows.len() != cols.len() || rows.len() != values.len() {
            return Err(FerroError::InvalidInput(
                "triplet arrays must have the same length".to_string(),
            ));
        }

        let mut tri = TriMat::new(shape);
        for ((&r, &c), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            if r >= shape.0 || c >= shape.1 {
                return Err(FerroError::InvalidInput(format!(
                    "index ({}, {}) out of bounds for shape {:?}",
                    r, c, shape
                )));
            }
            tri.add_triplet(r, c, v);
        }

        Ok(Self {
            inner: tri.to_csr(),
        })
    }

    /// Get the shape of the matrix as (rows, cols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Get the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.inner.rows()
    }

    /// Get the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.inner.cols()
    }

    /// Get the number of non-zero elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Get the sparsity ratio (fraction of zeros).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        let total = self.nrows() * self.ncols();
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Get a view of a specific row as a sparse vector.
    #[inline]
    pub fn row(&self, i: usize) -> SparseRowView<'_> {
        let outer_view = self.inner.outer_view(i);
        SparseRowView {
            inner: outer_view,
            ncols: self.ncols(),
        }
    }

    /// Get the raw data array.
    #[inline]
    pub fn data(&self) -> &[f64] {
        self.inner.data()
    }

    /// Get the column indices array.
    #[inline]
    pub fn indices(&self) -> &[usize] {
        self.inner.indices()
    }

    /// Get the row pointer array.
    #[inline]
    pub fn indptr(&self) -> Vec<usize> {
        self.inner.indptr().raw_storage().to_vec()
    }

    /// Convert to dense ndarray.
    pub fn to_dense(&self) -> Array2<f64> {
        let (nrows, ncols) = self.shape();
        let mut dense = Array2::zeros((nrows, ncols));

        for (row_idx, row) in self.inner.outer_iterator().enumerate() {
            for (&col_idx, &val) in row.indices().iter().zip(row.data().iter()) {
                dense[[row_idx, col_idx]] = val;
            }
        }

        dense
    }

    /// Get a view of the underlying sprs matrix.
    #[inline]
    pub fn view(&self) -> CsMatView<'_, f64> {
        self.inner.view()
    }

    /// Check if the matrix should probably be stored as dense instead.
    ///
    /// Returns true if sparsity < 50% (more than half non-zeros).
    #[inline]
    pub fn recommend_dense(&self) -> bool {
        self.sparsity() < 0.5
    }

    /// Compute X^T @ y efficiently (sparse transpose-vector product).
    ///
    /// For each row i, for each nnz (col, val) in row i: result[col] += val * y[i]
    pub fn transpose_dot(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut result = Array1::zeros(self.ncols());
        for (i, row) in self.inner.outer_iterator().enumerate() {
            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                result[col] += val * y[i];
            }
        }
        result
    }

    /// Compute X^T @ X (Gram matrix) — returns dense matrix.
    ///
    /// Native O(nnz * avg_nnz_per_row) implementation that avoids densifying the matrix.
    /// For each row, computes the outer product of its sparse entries and accumulates.
    pub fn gram_matrix(&self) -> Array2<f64> {
        let p = self.ncols();
        let mut result = Array2::zeros((p, p));
        for row in self.inner.outer_iterator() {
            let indices = row.indices();
            let data = row.data();
            for (a, &col_a) in indices.iter().enumerate() {
                for (b, &col_b) in indices.iter().enumerate() {
                    result[[col_a, col_b]] += data[a] * data[b];
                }
            }
        }
        result
    }

    /// Compute X^T @ diag(w) @ X (weighted Gram matrix).
    ///
    /// Native O(nnz * avg_nnz_per_row) implementation that avoids densifying the matrix.
    /// For each row i, multiplies the outer product of its sparse entries by w[i].
    pub fn weighted_gram(&self, w: &Array1<f64>) -> Array2<f64> {
        let p = self.ncols();
        let mut result = Array2::zeros((p, p));
        for (i, row) in self.inner.outer_iterator().enumerate() {
            let wi = w[i];
            let indices = row.indices();
            let data = row.data();
            for (a, &col_a) in indices.iter().enumerate() {
                for (b, &col_b) in indices.iter().enumerate() {
                    result[[col_a, col_b]] += data[a] * data[b] * wi;
                }
            }
        }
        result
    }

    /// Transpose the matrix (returns CSC, which is CSR of transpose).
    pub fn transpose(&self) -> CscMatrix {
        CscMatrix {
            inner: self.inner.clone().transpose_into(),
        }
    }

    /// Matrix-vector multiplication: y = A * x
    pub fn dot(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        if x.len() != self.ncols() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("vector of length {}", self.ncols()),
                actual: format!("vector of length {}", x.len()),
            });
        }

        let mut result = Array1::zeros(self.nrows());
        for (row_idx, row) in self.inner.outer_iterator().enumerate() {
            let mut sum = 0.0;
            for (&col_idx, &val) in row.indices().iter().zip(row.data().iter()) {
                sum += val * x[col_idx];
            }
            result[row_idx] = sum;
        }

        Ok(result)
    }

    /// Compute row-wise L2 norms (squared).
    pub fn row_norms_squared(&self) -> Array1<f64> {
        let mut norms = Array1::zeros(self.nrows());
        for (row_idx, row) in self.inner.outer_iterator().enumerate() {
            let norm_sq: f64 = row.data().iter().map(|&v| v * v).sum();
            norms[row_idx] = norm_sq;
        }
        norms
    }

    /// Compute row-wise L2 norms.
    pub fn row_norms(&self) -> Array1<f64> {
        self.row_norms_squared().mapv(f64::sqrt)
    }

    /// Scale rows by given factors (in-place).
    pub fn scale_rows(&mut self, factors: &Array1<f64>) -> Result<()> {
        if factors.len() != self.nrows() {
            return Err(FerroError::ShapeMismatch {
                expected: format!("{} factors for {} rows", self.nrows(), self.nrows()),
                actual: format!("{} factors", factors.len()),
            });
        }

        // sprs doesn't allow direct modification, so we rebuild
        let (nrows, ncols) = self.shape();
        let mut tri = TriMat::new((nrows, ncols));

        for (row_idx, row) in self.inner.outer_iterator().enumerate() {
            let scale = factors[row_idx];
            for (&col_idx, &val) in row.indices().iter().zip(row.data().iter()) {
                tri.add_triplet(row_idx, col_idx, val * scale);
            }
        }

        self.inner = tri.to_csr();
        Ok(())
    }
}

// =============================================================================
// Sparse Row View
// =============================================================================

/// A view of a single row from a CSR matrix.
#[derive(Debug, Clone)]
pub struct SparseRowView<'a> {
    inner: Option<CsVecView<'a, f64>>,
    ncols: usize,
}

impl<'a> SparseRowView<'a> {
    /// Get the number of non-zero elements in this row.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.inner.as_ref().map_or(0, |v| v.nnz())
    }

    /// Get the full dimension of the row.
    #[inline]
    pub fn dim(&self) -> usize {
        self.ncols
    }

    /// Get the column indices of non-zero elements.
    #[inline]
    pub fn indices(&self) -> &[usize] {
        self.inner.as_ref().map_or(&[], |v| v.indices())
    }

    /// Get the non-zero values.
    #[inline]
    pub fn data(&self) -> &[f64] {
        self.inner.as_ref().map_or(&[], |v| v.data())
    }

    /// Check if this row is empty (all zeros).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nnz() == 0
    }

    /// Get the L2 norm squared of this row.
    #[inline]
    pub fn norm_squared(&self) -> f64 {
        self.data().iter().map(|&v| v * v).sum()
    }

    /// Get the L2 norm of this row.
    #[inline]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Get the L1 norm of this row.
    #[inline]
    pub fn l1_norm(&self) -> f64 {
        self.data().iter().map(|&v| v.abs()).sum()
    }

    /// Convert to a dense array.
    pub fn to_dense(&self) -> Array1<f64> {
        let mut dense = Array1::zeros(self.ncols);
        for (&col, &val) in self.indices().iter().zip(self.data().iter()) {
            dense[col] = val;
        }
        dense
    }
}

// =============================================================================
// CSC Matrix Wrapper
// =============================================================================

/// A Compressed Sparse Column (CSC) matrix.
///
/// CSC format is efficient for:
/// - Column slicing: O(1) to get a column
/// - Feature selection operations
/// - Transpose of CSR operations
#[derive(Debug, Clone)]
pub struct CscMatrix {
    inner: CsMat<f64>,
}

impl CscMatrix {
    /// Create from raw CSC components.
    pub fn new(
        shape: (usize, usize),
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
    ) -> Result<Self> {
        let mat = CsMat::new_csc(shape, indptr, indices, data);
        Ok(Self { inner: mat })
    }

    /// Get the shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Get number of non-zeros.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> CsrMatrix {
        CsrMatrix {
            inner: self.inner.to_csr(),
        }
    }

    /// Convert to dense.
    pub fn to_dense(&self) -> Array2<f64> {
        self.to_csr().to_dense()
    }
}

// =============================================================================
// Sparse Vector
// =============================================================================

/// An owned sparse vector.
#[derive(Debug, Clone)]
pub struct SparseVector {
    inner: CsVec<f64>,
}

impl SparseVector {
    /// Create from indices and values.
    pub fn new(dim: usize, indices: Vec<usize>, data: Vec<f64>) -> Result<Self> {
        if indices.len() != data.len() {
            return Err(FerroError::InvalidInput(
                "indices and data must have the same length".to_string(),
            ));
        }
        for &idx in &indices {
            if idx >= dim {
                return Err(FerroError::InvalidInput(format!(
                    "index {} out of bounds for dimension {}",
                    idx, dim
                )));
            }
        }
        Ok(Self {
            inner: CsVec::new(dim, indices, data),
        })
    }

    /// Create from a dense array.
    pub fn from_dense(dense: &Array1<f64>) -> Self {
        Self::from_dense_with_threshold(dense, 0.0)
    }

    /// Create from a dense array with threshold.
    pub fn from_dense_with_threshold(dense: &Array1<f64>, threshold: f64) -> Self {
        let dim = dense.len();
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for (i, &val) in dense.iter().enumerate() {
            if val.abs() > threshold {
                indices.push(i);
                data.push(val);
            }
        }

        Self {
            inner: CsVec::new(dim, indices, data),
        }
    }

    /// Get dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Get number of non-zeros.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Get indices.
    #[inline]
    pub fn indices(&self) -> &[usize] {
        self.inner.indices()
    }

    /// Get data.
    #[inline]
    pub fn data(&self) -> &[f64] {
        self.inner.data()
    }

    /// Convert to dense.
    pub fn to_dense(&self) -> Array1<f64> {
        let mut dense = Array1::zeros(self.dim());
        for (&idx, &val) in self.indices().iter().zip(self.data().iter()) {
            dense[idx] = val;
        }
        dense
    }
}

// =============================================================================
// Sparse Distance Calculations
// =============================================================================

/// Compute squared Euclidean distance between two sparse rows.
///
/// This is O(nnz_a + nnz_b) instead of O(n) for dense.
///
/// # Algorithm
///
/// ||a - b||² = ||a||² + ||b||² - 2 * a·b
///
/// where a·b is computed by iterating over the union of non-zero indices.
#[inline]
pub fn sparse_squared_euclidean_distance(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64 {
    let dot = sparse_dot_product(a, b);
    let a_norm_sq = a.norm_squared();
    let b_norm_sq = b.norm_squared();

    (a_norm_sq + b_norm_sq - 2.0 * dot).max(0.0)
}

/// Compute Euclidean distance between two sparse rows.
#[inline]
pub fn sparse_euclidean_distance(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64 {
    sparse_squared_euclidean_distance(a, b).sqrt()
}

/// Compute Manhattan (L1) distance between two sparse rows.
///
/// This iterates over the union of non-zero indices.
pub fn sparse_manhattan_distance(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64 {
    let a_indices = a.indices();
    let a_data = a.data();
    let b_indices = b.indices();
    let b_data = b.data();

    let mut sum = 0.0;
    let mut i = 0;
    let mut j = 0;

    // Merge-style iteration over sorted indices
    while i < a_indices.len() && j < b_indices.len() {
        match a_indices[i].cmp(&b_indices[j]) {
            std::cmp::Ordering::Less => {
                sum += a_data[i].abs();
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                sum += b_data[j].abs();
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                sum += (a_data[i] - b_data[j]).abs();
                i += 1;
                j += 1;
            }
        }
    }

    // Handle remaining elements
    while i < a_indices.len() {
        sum += a_data[i].abs();
        i += 1;
    }
    while j < b_indices.len() {
        sum += b_data[j].abs();
        j += 1;
    }

    sum
}

/// Compute dot product of two sparse rows.
///
/// This is O(min(nnz_a, nnz_b)) in the best case.
pub fn sparse_dot_product(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64 {
    let a_indices = a.indices();
    let a_data = a.data();
    let b_indices = b.indices();
    let b_data = b.data();

    let mut sum = 0.0;
    let mut i = 0;
    let mut j = 0;

    // Merge-style iteration (indices are sorted in sprs)
    while i < a_indices.len() && j < b_indices.len() {
        match a_indices[i].cmp(&b_indices[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                sum += a_data[i] * b_data[j];
                i += 1;
                j += 1;
            }
        }
    }

    sum
}

/// Compute cosine similarity between two sparse rows.
pub fn sparse_cosine_similarity(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64 {
    let dot = sparse_dot_product(a, b);
    let norm_a = a.norm();
    let norm_b = b.norm();

    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute cosine distance between two sparse rows.
#[inline]
pub fn sparse_cosine_distance(a: &SparseRowView<'_>, b: &SparseRowView<'_>) -> f64 {
    1.0 - sparse_cosine_similarity(a, b)
}

// =============================================================================
// Batch Distance Calculations
// =============================================================================

/// Compute distances from a query row to all rows in a sparse matrix.
///
/// # Arguments
///
/// * `query` - The query row
/// * `matrix` - The sparse matrix to compute distances against
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// Array of distances from query to each row in matrix.
pub fn sparse_pairwise_distances(
    query: &SparseRowView<'_>,
    matrix: &CsrMatrix,
    metric: SparseDistanceMetric,
) -> Array1<f64> {
    let n = matrix.nrows();
    let mut distances = Array1::zeros(n);

    match metric {
        SparseDistanceMetric::Euclidean => {
            // Pre-compute query norm for efficiency
            let query_norm_sq = query.norm_squared();

            for i in 0..n {
                let row = matrix.row(i);
                let dot = sparse_dot_product(query, &row);
                let row_norm_sq = row.norm_squared();
                distances[i] = (query_norm_sq + row_norm_sq - 2.0 * dot).max(0.0).sqrt();
            }
        }
        SparseDistanceMetric::SquaredEuclidean => {
            let query_norm_sq = query.norm_squared();

            for i in 0..n {
                let row = matrix.row(i);
                let dot = sparse_dot_product(query, &row);
                let row_norm_sq = row.norm_squared();
                distances[i] = (query_norm_sq + row_norm_sq - 2.0 * dot).max(0.0);
            }
        }
        SparseDistanceMetric::Manhattan => {
            for i in 0..n {
                let row = matrix.row(i);
                distances[i] = sparse_manhattan_distance(query, &row);
            }
        }
        SparseDistanceMetric::Cosine => {
            let query_norm = query.norm();

            for i in 0..n {
                let row = matrix.row(i);
                let dot = sparse_dot_product(query, &row);
                let row_norm = row.norm();
                let denom = query_norm * row_norm;
                distances[i] = if denom < 1e-10 {
                    1.0
                } else {
                    1.0 - dot / denom
                };
            }
        }
    }

    distances
}

/// Distance metrics available for sparse matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseDistanceMetric {
    /// Euclidean (L2) distance
    Euclidean,
    /// Squared Euclidean distance (avoids sqrt for efficiency)
    SquaredEuclidean,
    /// Manhattan (L1) distance
    Manhattan,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

// =============================================================================
// Sparse Matrix Operations
// =============================================================================

/// Normalize rows of a sparse matrix to unit L2 norm.
///
/// Rows with zero norm are left unchanged.
pub fn sparse_normalize_rows_l2(matrix: &mut CsrMatrix) {
    let norms = matrix.row_norms();
    let inv_norms: Array1<f64> = norms.mapv(|n| if n > 1e-10 { 1.0 / n } else { 1.0 });
    let _ = matrix.scale_rows(&inv_norms);
}

/// Compute column sums of a sparse matrix.
pub fn sparse_column_sums(matrix: &CsrMatrix) -> Array1<f64> {
    let mut sums = Array1::zeros(matrix.ncols());

    for row in matrix.inner.outer_iterator() {
        for (&col_idx, &val) in row.indices().iter().zip(row.data().iter()) {
            sums[col_idx] += val;
        }
    }

    sums
}

/// Compute column means of a sparse matrix.
pub fn sparse_column_means(matrix: &CsrMatrix) -> Array1<f64> {
    let sums = sparse_column_sums(matrix);
    let n = matrix.nrows() as f64;
    sums / n
}

/// Count non-zeros per column.
pub fn sparse_column_nnz(matrix: &CsrMatrix) -> Array1<usize> {
    let mut counts = Array1::zeros(matrix.ncols());

    for row in matrix.inner.outer_iterator() {
        for &col_idx in row.indices() {
            counts[col_idx] += 1;
        }
    }

    counts
}

// =============================================================================
// Sparse-Dense Hybrid Operations
// =============================================================================

/// Information about a sparse matrix useful for deciding operations.
#[derive(Debug, Clone)]
pub struct SparseMatrixInfo {
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Sparsity ratio (fraction of zeros)
    pub sparsity: f64,
    /// Density ratio (fraction of non-zeros)
    pub density: f64,
    /// Estimated memory for sparse format (bytes)
    pub sparse_memory: usize,
    /// Estimated memory for dense format (bytes)
    pub dense_memory: usize,
    /// Whether dense format is recommended
    pub recommend_dense: bool,
}

impl SparseMatrixInfo {
    /// Compute info for a sparse matrix.
    pub fn from_matrix(matrix: &CsrMatrix) -> Self {
        let (nrows, ncols) = matrix.shape();
        let nnz = matrix.nnz();
        let total = nrows * ncols;

        let sparsity = if total == 0 {
            1.0
        } else {
            1.0 - (nnz as f64 / total as f64)
        };

        // Sparse memory: data (8 bytes) + indices (8 bytes) + indptr (8 bytes per row)
        let sparse_memory = nnz * 16 + (nrows + 1) * 8;
        // Dense memory: 8 bytes per element
        let dense_memory = total * 8;

        Self {
            nrows,
            ncols,
            nnz,
            sparsity,
            density: 1.0 - sparsity,
            sparse_memory,
            dense_memory,
            recommend_dense: sparsity < 0.5,
        }
    }

    /// Print a summary of the matrix info.
    pub fn summary(&self) -> String {
        format!(
            "SparseMatrix: {}x{} ({} nnz, {:.1}% sparse)\n\
             Memory: sparse={:.1}KB, dense={:.1}KB\n\
             Recommendation: {}",
            self.nrows,
            self.ncols,
            self.nnz,
            self.sparsity * 100.0,
            self.sparse_memory as f64 / 1024.0,
            self.dense_memory as f64 / 1024.0,
            if self.recommend_dense {
                "use dense"
            } else {
                "use sparse"
            }
        )
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Create a sparse identity matrix.
pub fn sparse_eye(n: usize) -> CsrMatrix {
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n);
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        indptr.push(i);
        indices.push(i);
        data.push(1.0);
    }
    indptr.push(n);

    CsrMatrix::new((n, n), indptr, indices, data).unwrap()
}

/// Create a sparse diagonal matrix from a vector.
pub fn sparse_diag(diag: &Array1<f64>) -> CsrMatrix {
    let n = diag.len();
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();

    for (i, &val) in diag.iter().enumerate() {
        indptr.push(indices.len());
        if val.abs() > 1e-15 {
            indices.push(i);
            data.push(val);
        }
    }
    indptr.push(indices.len());

    CsrMatrix::new((n, n), indptr, indices, data).unwrap()
}

/// Vertically stack sparse matrices.
pub fn sparse_vstack(matrices: &[&CsrMatrix]) -> Result<CsrMatrix> {
    if matrices.is_empty() {
        return Err(FerroError::InvalidInput(
            "cannot vstack empty list".to_string(),
        ));
    }

    let ncols = matrices[0].ncols();
    for mat in matrices.iter().skip(1) {
        if mat.ncols() != ncols {
            return Err(FerroError::ShapeMismatch {
                expected: format!("{} columns", ncols),
                actual: format!("{} columns", mat.ncols()),
            });
        }
    }

    let total_rows: usize = matrices.iter().map(|m| m.nrows()).sum();
    let total_nnz: usize = matrices.iter().map(|m| m.nnz()).sum();

    let mut indptr = Vec::with_capacity(total_rows + 1);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut data = Vec::with_capacity(total_nnz);

    indptr.push(0);

    for mat in matrices {
        for row in mat.inner.outer_iterator() {
            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                indices.push(col);
                data.push(val);
            }
            indptr.push(indices.len());
        }
    }

    CsrMatrix::new((total_rows, ncols), indptr, indices, data)
}

/// Horizontally stack sparse matrices.
pub fn sparse_hstack(matrices: &[&CsrMatrix]) -> Result<CsrMatrix> {
    if matrices.is_empty() {
        return Err(FerroError::InvalidInput(
            "cannot hstack empty list".to_string(),
        ));
    }

    let nrows = matrices[0].nrows();
    for mat in matrices.iter().skip(1) {
        if mat.nrows() != nrows {
            return Err(FerroError::ShapeMismatch {
                expected: format!("{} rows", nrows),
                actual: format!("{} rows", mat.nrows()),
            });
        }
    }

    let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();
    let total_nnz: usize = matrices.iter().map(|m| m.nnz()).sum();

    let mut tri = TriMat::with_capacity((nrows, total_cols), total_nnz);

    let mut col_offset = 0;
    for mat in matrices {
        for (row_idx, row) in mat.inner.outer_iterator().enumerate() {
            for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
                tri.add_triplet(row_idx, col + col_offset, val);
            }
        }
        col_offset += mat.ncols();
    }

    Ok(CsrMatrix {
        inner: tri.to_csr(),
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_csr_from_dense() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let sparse = CsrMatrix::from_dense(&dense);

        assert_eq!(sparse.shape(), (3, 3));
        assert_eq!(sparse.nnz(), 5);
        assert!((sparse.sparsity() - 4.0 / 9.0).abs() < EPSILON);
    }

    #[test]
    fn test_csr_to_dense_roundtrip() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]];
        let sparse = CsrMatrix::from_dense(&dense);
        let recovered = sparse.to_dense();

        for i in 0..3 {
            for j in 0..3 {
                assert!((dense[[i, j]] - recovered[[i, j]]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_sparse_row_view() {
        let dense = array![[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]];
        let sparse = CsrMatrix::from_dense(&dense);

        let row0 = sparse.row(0);
        assert_eq!(row0.nnz(), 2);
        assert_eq!(row0.dim(), 3);

        let row1 = sparse.row(1);
        assert_eq!(row1.nnz(), 1);
    }

    #[test]
    fn test_sparse_dot_product() {
        let dense_a = array![[1.0, 0.0, 2.0]];
        let dense_b = array![[2.0, 3.0, 4.0]];

        let sparse_a = CsrMatrix::from_dense(&dense_a);
        let sparse_b = CsrMatrix::from_dense(&dense_b);

        let row_a = sparse_a.row(0);
        let row_b = sparse_b.row(0);

        let dot = sparse_dot_product(&row_a, &row_b);
        // 1*2 + 0*3 + 2*4 = 2 + 8 = 10
        assert!((dot - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_euclidean_distance() {
        let dense_a = array![[0.0, 0.0]];
        let dense_b = array![[3.0, 4.0]];

        let sparse_a = CsrMatrix::from_dense(&dense_a);
        let sparse_b = CsrMatrix::from_dense(&dense_b);

        let row_a = sparse_a.row(0);
        let row_b = sparse_b.row(0);

        let dist = sparse_euclidean_distance(&row_a, &row_b);
        assert!((dist - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_manhattan_distance() {
        let dense_a = array![[1.0, 0.0, 2.0]];
        let dense_b = array![[0.0, 3.0, 5.0]];

        let sparse_a = CsrMatrix::from_dense(&dense_a);
        let sparse_b = CsrMatrix::from_dense(&dense_b);

        let row_a = sparse_a.row(0);
        let row_b = sparse_b.row(0);

        let dist = sparse_manhattan_distance(&row_a, &row_b);
        // |1-0| + |0-3| + |2-5| = 1 + 3 + 3 = 7
        assert!((dist - 7.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_cosine_similarity() {
        let dense_a = array![[1.0, 0.0]];
        let dense_b = array![[1.0, 0.0]];

        let sparse_a = CsrMatrix::from_dense(&dense_a);
        let sparse_b = CsrMatrix::from_dense(&dense_b);

        let row_a = sparse_a.row(0);
        let row_b = sparse_b.row(0);

        let sim = sparse_cosine_similarity(&row_a, &row_b);
        assert!((sim - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_cosine_orthogonal() {
        let dense_a = array![[1.0, 0.0]];
        let dense_b = array![[0.0, 1.0]];

        let sparse_a = CsrMatrix::from_dense(&dense_a);
        let sparse_b = CsrMatrix::from_dense(&dense_b);

        let row_a = sparse_a.row(0);
        let row_b = sparse_b.row(0);

        let sim = sparse_cosine_similarity(&row_a, &row_b);
        assert!(sim.abs() < EPSILON);
    }

    #[test]
    fn test_csr_matrix_vector_dot() {
        let dense = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let sparse = CsrMatrix::from_dense(&dense);
        let x = array![1.0, 2.0, 3.0];

        let result = sparse.dot(&x).unwrap();
        // Row 0: 1*1 + 2*2 + 3*3 = 14
        // Row 1: 4*1 + 5*2 + 6*3 = 32
        assert!((result[0] - 14.0).abs() < EPSILON);
        assert!((result[1] - 32.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_pairwise_distances() {
        let query_dense = array![[0.0, 0.0]];
        let matrix_dense = array![[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]];

        let query_sparse = CsrMatrix::from_dense(&query_dense);
        let matrix_sparse = CsrMatrix::from_dense(&matrix_dense);

        let query_row = query_sparse.row(0);
        let distances =
            sparse_pairwise_distances(&query_row, &matrix_sparse, SparseDistanceMetric::Euclidean);

        assert!((distances[0] - 1.0).abs() < EPSILON);
        assert!((distances[1] - 2.0).abs() < EPSILON);
        assert!((distances[2] - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_column_sums() {
        let dense = array![[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]];
        let sparse = CsrMatrix::from_dense(&dense);

        let sums = sparse_column_sums(&sparse);
        assert!((sums[0] - 4.0).abs() < EPSILON);
        assert!((sums[1] - 2.0).abs() < EPSILON);
        assert!((sums[2] - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_eye() {
        let eye = sparse_eye(3);

        assert_eq!(eye.shape(), (3, 3));
        assert_eq!(eye.nnz(), 3);

        let dense = eye.to_dense();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dense[[i, j]] - expected).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_sparse_vstack() {
        let a = CsrMatrix::from_dense(&array![[1.0, 2.0]]);
        let b = CsrMatrix::from_dense(&array![[3.0, 4.0]]);

        let stacked = sparse_vstack(&[&a, &b]).unwrap();
        assert_eq!(stacked.shape(), (2, 2));

        let dense = stacked.to_dense();
        assert!((dense[[0, 0]] - 1.0).abs() < EPSILON);
        assert!((dense[[0, 1]] - 2.0).abs() < EPSILON);
        assert!((dense[[1, 0]] - 3.0).abs() < EPSILON);
        assert!((dense[[1, 1]] - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_hstack() {
        let a = CsrMatrix::from_dense(&array![[1.0], [2.0]]);
        let b = CsrMatrix::from_dense(&array![[3.0], [4.0]]);

        let stacked = sparse_hstack(&[&a, &b]).unwrap();
        assert_eq!(stacked.shape(), (2, 2));

        let dense = stacked.to_dense();
        assert!((dense[[0, 0]] - 1.0).abs() < EPSILON);
        assert!((dense[[0, 1]] - 3.0).abs() < EPSILON);
        assert!((dense[[1, 0]] - 2.0).abs() < EPSILON);
        assert!((dense[[1, 1]] - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_matrix_info() {
        let dense = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0]
        ];
        let sparse = CsrMatrix::from_dense(&dense);
        let info = SparseMatrixInfo::from_matrix(&sparse);

        assert_eq!(info.nrows, 4);
        assert_eq!(info.ncols, 4);
        assert_eq!(info.nnz, 4);
        assert!((info.sparsity - 0.75).abs() < EPSILON);
        assert!(!info.recommend_dense);
    }

    #[test]
    fn test_sparse_from_triplets() {
        let rows = [0, 1, 2];
        let cols = [0, 1, 2];
        let values = [1.0, 2.0, 3.0];

        let sparse = CsrMatrix::from_triplets((3, 3), &rows, &cols, &values).unwrap();
        assert_eq!(sparse.nnz(), 3);

        let dense = sparse.to_dense();
        assert!((dense[[0, 0]] - 1.0).abs() < EPSILON);
        assert!((dense[[1, 1]] - 2.0).abs() < EPSILON);
        assert!((dense[[2, 2]] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_vector() {
        let dense = array![1.0, 0.0, 2.0, 0.0, 3.0];
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.dim(), 5);
        assert_eq!(sparse.nnz(), 3);

        let recovered = sparse.to_dense();
        for i in 0..5 {
            assert!((dense[i] - recovered[i]).abs() < EPSILON);
        }
    }

    #[test]
    fn test_row_norms() {
        let dense = array![[3.0, 4.0], [5.0, 12.0]];
        let sparse = CsrMatrix::from_dense(&dense);

        let norms = sparse.row_norms();
        assert!((norms[0] - 5.0).abs() < EPSILON);
        assert!((norms[1] - 13.0).abs() < EPSILON);
    }

    #[test]
    fn test_sparse_diag() {
        let diag = array![1.0, 2.0, 3.0];
        let sparse = sparse_diag(&diag);

        assert_eq!(sparse.shape(), (3, 3));
        assert_eq!(sparse.nnz(), 3);

        let dense = sparse.to_dense();
        for i in 0..3 {
            assert!((dense[[i, i]] - diag[i]).abs() < EPSILON);
        }
    }

    #[test]
    fn test_sparse_high_sparsity() {
        // Test with very sparse data (typical for text/NLP)
        let mut dense = Array2::zeros((100, 1000));
        // Only 0.1% density
        dense[[0, 0]] = 1.0;
        dense[[50, 500]] = 2.0;
        dense[[99, 999]] = 3.0;

        let sparse = CsrMatrix::from_dense(&dense);
        let info = SparseMatrixInfo::from_matrix(&sparse);

        assert!(!info.recommend_dense);
        assert!(info.sparsity > 0.99);
        assert!(info.sparse_memory < info.dense_memory);
    }

    #[test]
    fn test_empty_sparse_row() {
        let dense = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let sparse = CsrMatrix::from_dense(&dense);

        let row0 = sparse.row(0);
        assert!(row0.is_empty());
        assert_eq!(row0.nnz(), 0);
        assert!((row0.norm() - 0.0).abs() < EPSILON);

        let row1 = sparse.row(1);
        assert!(!row1.is_empty());
    }
}
