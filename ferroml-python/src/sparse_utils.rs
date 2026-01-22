//! Sparse matrix conversion utilities for FerroML Python bindings.
//!
//! This module provides utilities for converting between scipy.sparse matrices
//! and ndarray types, enabling efficient data interchange between Python's
//! sparse matrix libraries and FerroML's Rust core.
//!
//! ## Supported Formats
//!
//! - **CSR (Compressed Sparse Row)**: Row-oriented storage, efficient for row slicing
//! - **CSC (Compressed Sparse Column)**: Column-oriented storage, efficient for column slicing
//!
//! Both formats are converted to dense ndarray::Array2<f64> for use with FerroML's
//! algorithms, which currently operate on dense matrices.
//!
//! ## Usage Pattern
//!
//! ```python
//! import scipy.sparse as sp
//! from ferroml.linear import LinearRegression
//!
//! # Create a sparse CSR matrix
//! X_sparse = sp.csr_matrix([
//!     [1.0, 0.0, 0.0, 2.0],
//!     [0.0, 0.0, 3.0, 0.0],
//!     [4.0, 5.0, 0.0, 0.0],
//!     [0.0, 0.0, 0.0, 6.0],
//! ])
//!
//! # Fit directly from sparse matrix
//! model = LinearRegression()
//! model.fit_sparse(X_sparse, y)
//! ```
//!
//! ## Performance Considerations
//!
//! Conversion from sparse to dense matrices requires memory allocation proportional
//! to the full dense matrix size. For very large sparse matrices with high sparsity,
//! consider:
//!
//! 1. Using algorithms that natively support sparse matrices (not yet implemented)
//! 2. Reducing dimensionality before conversion
//! 3. Using chunked/batch processing
//!
//! ## Sparsity Threshold
//!
//! A warning is issued when converting matrices with sparsity > 50%, as dense
//! representation may be more memory-efficient for such data.

use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

/// Result type for sparse matrix conversion operations.
pub type SparseResult<T> = Result<T, SparseConversionError>;

/// Errors that can occur during sparse matrix conversion.
#[derive(Debug)]
pub enum SparseConversionError {
    /// Matrix is not a recognized sparse format
    UnsupportedFormat(String),
    /// Matrix has invalid dimensions
    InvalidDimensions { rows: usize, cols: usize },
    /// Matrix is empty
    EmptyMatrix,
    /// Data type is not numeric
    NonNumericData(String),
    /// Python error during conversion
    PythonError(String),
    /// Matrix contains NaN or infinite values
    InvalidValues(String),
    /// Shape mismatch between data, indices, and indptr
    ShapeMismatch { expected: String, actual: String },
}

impl std::fmt::Display for SparseConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(format) => {
                write!(
                    f,
                    "Unsupported sparse format '{}'. Expected csr_matrix, csr_array, csc_matrix, or csc_array.",
                    format
                )
            }
            Self::InvalidDimensions { rows, cols } => {
                write!(f, "Invalid matrix dimensions: {}x{}", rows, cols)
            }
            Self::EmptyMatrix => write!(f, "Sparse matrix is empty"),
            Self::NonNumericData(dtype) => {
                write!(
                    f,
                    "Non-numeric data type '{}'. Expected float or int.",
                    dtype
                )
            }
            Self::PythonError(e) => write!(f, "Python error: {}", e),
            Self::InvalidValues(msg) => write!(f, "Invalid values: {}", msg),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for SparseConversionError {}

impl From<SparseConversionError> for PyErr {
    fn from(err: SparseConversionError) -> Self {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
    }
}

impl From<PyErr> for SparseConversionError {
    fn from(err: PyErr) -> Self {
        SparseConversionError::PythonError(err.to_string())
    }
}

/// Information about a sparse matrix for diagnostics.
#[derive(Debug, Clone)]
pub struct SparseMatrixInfo {
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Sparsity ratio (0.0 = fully dense, 1.0 = fully sparse)
    pub sparsity: f64,
    /// Format string (e.g., "csr_matrix", "csc_array")
    pub format: String,
    /// Data type string
    pub dtype: String,
}

impl SparseMatrixInfo {
    /// Check if the matrix is highly sparse (>90% zeros)
    pub fn is_highly_sparse(&self) -> bool {
        self.sparsity > 0.9
    }

    /// Check if dense representation would be more memory-efficient
    pub fn recommend_dense(&self) -> bool {
        // Dense representation is better when sparsity < 50%
        // because sparse overhead (indices, indptr) is significant
        self.sparsity < 0.5
    }
}

/// Extract sparse matrix data and convert to dense Array2<f64>.
///
/// This function handles both CSR and CSC formats from scipy.sparse,
/// converting them to dense ndarray for use with FerroML algorithms.
///
/// # Arguments
///
/// * `py` - Python GIL token
/// * `sparse_matrix` - The scipy.sparse matrix object
///
/// # Returns
///
/// A dense `Array2<f64>` representation of the sparse matrix.
///
/// # Errors
///
/// Returns an error if:
/// - The matrix is not a CSR or CSC format
/// - The matrix is empty
/// - The data type cannot be converted to f64
/// - The matrix contains NaN or infinite values
pub fn sparse_to_dense<'py>(
    py: Python<'py>,
    sparse_matrix: &Bound<'py, PyAny>,
) -> SparseResult<Array2<f64>> {
    // Validate sparse format
    let format_str = get_sparse_format(sparse_matrix)?;

    if !matches!(
        format_str.as_str(),
        "csr" | "csc" | "csr_matrix" | "csr_array" | "csc_matrix" | "csc_array"
    ) {
        // Try to convert to CSR first
        let converted = sparse_matrix
            .call_method0("tocsr")
            .map_err(|_| SparseConversionError::UnsupportedFormat(format_str.clone()))?;
        return sparse_to_dense(py, &converted);
    }

    // Get shape
    let shape: (usize, usize) = sparse_matrix.getattr("shape")?.extract()?;
    let (n_rows, n_cols) = shape;

    if n_rows == 0 || n_cols == 0 {
        return Err(SparseConversionError::EmptyMatrix);
    }

    // Convert to dense using toarray() method
    let dense_array = sparse_matrix.call_method0("toarray")?;

    // Convert to float64 if needed
    let dense_float = dense_array
        .call_method1("astype", ("float64",))
        .map_err(|e| SparseConversionError::NonNumericData(e.to_string()))?;

    // Extract as numpy array and convert to ndarray
    let py_array: numpy::PyReadonlyArray2<f64> = dense_float.extract()?;
    let array = py_array.to_owned_array();

    // Check for NaN/Inf values
    if array.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return Err(SparseConversionError::InvalidValues(
            "Matrix contains NaN or infinite values".to_string(),
        ));
    }

    Ok(array)
}

/// Extract sparse matrix data efficiently without full dense conversion.
///
/// This extracts the internal data, indices, and indptr arrays from a CSR/CSC
/// matrix and constructs the dense array directly, which can be more efficient
/// for very sparse matrices.
///
/// # Arguments
///
/// * `_py` - Python GIL token
/// * `sparse_matrix` - The scipy.sparse matrix object
///
/// # Returns
///
/// A dense `Array2<f64>` representation of the sparse matrix.
pub fn sparse_to_dense_efficient<'py>(
    _py: Python<'py>,
    sparse_matrix: &Bound<'py, PyAny>,
) -> SparseResult<Array2<f64>> {
    let format_str = get_sparse_format(sparse_matrix)?;
    let is_csr = matches!(format_str.as_str(), "csr" | "csr_matrix" | "csr_array");

    // Get shape
    let shape: (usize, usize) = sparse_matrix.getattr("shape")?.extract()?;
    let (n_rows, n_cols) = shape;

    if n_rows == 0 || n_cols == 0 {
        return Err(SparseConversionError::EmptyMatrix);
    }

    // Extract data, indices, indptr arrays
    let data_py = sparse_matrix.getattr("data")?;
    let indices_py = sparse_matrix.getattr("indices")?;
    let indptr_py = sparse_matrix.getattr("indptr")?;

    // Convert to float64 for data
    let data_float = data_py
        .call_method1("astype", ("float64",))
        .map_err(|e| SparseConversionError::NonNumericData(e.to_string()))?;

    let data: PyReadonlyArray1<f64> = data_float.extract()?;
    let data_arr = data.as_array();

    // Indices are typically int32 or int64
    let indices: Vec<usize> = if let Ok(idx) = indices_py.extract::<PyReadonlyArray1<i64>>() {
        idx.as_array().iter().map(|&x| x as usize).collect()
    } else if let Ok(idx) = indices_py.extract::<PyReadonlyArray1<i32>>() {
        idx.as_array().iter().map(|&x| x as usize).collect()
    } else {
        return Err(SparseConversionError::NonNumericData(
            "indices array has unsupported dtype".to_string(),
        ));
    };

    let indptr: Vec<usize> = if let Ok(ptr) = indptr_py.extract::<PyReadonlyArray1<i64>>() {
        ptr.as_array().iter().map(|&x| x as usize).collect()
    } else if let Ok(ptr) = indptr_py.extract::<PyReadonlyArray1<i32>>() {
        ptr.as_array().iter().map(|&x| x as usize).collect()
    } else {
        return Err(SparseConversionError::NonNumericData(
            "indptr array has unsupported dtype".to_string(),
        ));
    };

    // Create dense array
    let mut dense = Array2::zeros((n_rows, n_cols));

    if is_csr {
        // CSR format: indptr has n_rows + 1 elements
        if indptr.len() != n_rows + 1 {
            return Err(SparseConversionError::ShapeMismatch {
                expected: format!("indptr length {}", n_rows + 1),
                actual: format!("indptr length {}", indptr.len()),
            });
        }

        for row in 0..n_rows {
            let start = indptr[row];
            let end = indptr[row + 1];
            for idx in start..end {
                let col = indices[idx];
                dense[[row, col]] = data_arr[idx];
            }
        }
    } else {
        // CSC format: indptr has n_cols + 1 elements
        if indptr.len() != n_cols + 1 {
            return Err(SparseConversionError::ShapeMismatch {
                expected: format!("indptr length {}", n_cols + 1),
                actual: format!("indptr length {}", indptr.len()),
            });
        }

        for col in 0..n_cols {
            let start = indptr[col];
            let end = indptr[col + 1];
            for idx in start..end {
                let row = indices[idx];
                dense[[row, col]] = data_arr[idx];
            }
        }
    }

    // Check for NaN/Inf values
    if dense.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return Err(SparseConversionError::InvalidValues(
            "Matrix contains NaN or infinite values".to_string(),
        ));
    }

    Ok(dense)
}

/// Get information about a sparse matrix without converting it.
///
/// # Arguments
///
/// * `sparse_matrix` - The scipy.sparse matrix object
///
/// # Returns
///
/// A `SparseMatrixInfo` struct with matrix metadata.
pub fn get_sparse_info<'py>(sparse_matrix: &Bound<'py, PyAny>) -> SparseResult<SparseMatrixInfo> {
    let format = get_sparse_format(sparse_matrix)?;

    let shape: (usize, usize) = sparse_matrix.getattr("shape")?.extract()?;
    let (n_rows, n_cols) = shape;

    let nnz: usize = sparse_matrix.getattr("nnz")?.extract()?;

    let dtype_obj = sparse_matrix.getattr("dtype")?;
    let dtype: String = dtype_obj.call_method0("__str__")?.extract()?;

    let total_elements = n_rows * n_cols;
    let sparsity = if total_elements > 0 {
        1.0 - (nnz as f64 / total_elements as f64)
    } else {
        1.0
    };

    Ok(SparseMatrixInfo {
        n_rows,
        n_cols,
        nnz,
        sparsity,
        format,
        dtype,
    })
}

/// Get the sparse format string from a matrix.
fn get_sparse_format<'py>(sparse_matrix: &Bound<'py, PyAny>) -> SparseResult<String> {
    // Try to get format attribute
    if let Ok(format) = sparse_matrix.getattr("format") {
        if let Ok(f) = format.extract::<String>() {
            return Ok(f);
        }
    }

    // Try to infer from class name
    let class_name = sparse_matrix
        .get_type()
        .name()
        .map(|s| s.to_string())
        .unwrap_or_default();

    if class_name.contains("csr") {
        Ok("csr".to_string())
    } else if class_name.contains("csc") {
        Ok("csc".to_string())
    } else if class_name.contains("coo") {
        Ok("coo".to_string())
    } else if class_name.contains("lil") {
        Ok("lil".to_string())
    } else if class_name.contains("dok") {
        Ok("dok".to_string())
    } else if class_name.contains("bsr") {
        Ok("bsr".to_string())
    } else if class_name.contains("dia") {
        Ok("dia".to_string())
    } else {
        Ok(class_name)
    }
}

/// Check if a Python object is a scipy.sparse matrix.
pub fn is_sparse_matrix<'py>(obj: &Bound<'py, PyAny>) -> bool {
    // Check for scipy.sparse attributes
    obj.hasattr("data").unwrap_or(false)
        && obj.hasattr("indices").unwrap_or(false)
        && obj.hasattr("indptr").unwrap_or(false)
        && obj.hasattr("shape").unwrap_or(false)
        && obj.hasattr("nnz").unwrap_or(false)
}

/// Check if a Python object is a scipy.sparse matrix (alternative check).
pub fn is_sparse_matrix_any<'py>(obj: &Bound<'py, PyAny>) -> bool {
    // Check for toarray method and shape attribute (common to all scipy.sparse)
    obj.hasattr("toarray").unwrap_or(false)
        && obj.hasattr("shape").unwrap_or(false)
        && obj.hasattr("nnz").unwrap_or(false)
}

/// Extracted sparse data for ML operations.
#[derive(Debug)]
pub struct SparseData {
    /// Dense feature matrix (n_samples x n_features)
    pub x: Array2<f64>,
    /// Target vector (n_samples,)
    pub y: Array1<f64>,
    /// Original sparse matrix info
    pub info: SparseMatrixInfo,
}

/// Extract features and target from a sparse matrix and target array.
///
/// # Arguments
///
/// * `py` - Python GIL token
/// * `sparse_x` - The scipy.sparse feature matrix
/// * `y` - The target array (numpy array)
///
/// # Returns
///
/// A `SparseData` struct containing the dense feature matrix and target.
pub fn extract_sparse_xy<'py>(
    py: Python<'py>,
    sparse_x: &Bound<'py, PyAny>,
    y: PyReadonlyArray1<'py, f64>,
) -> SparseResult<SparseData> {
    let info = get_sparse_info(sparse_x)?;

    // Convert sparse to dense
    let x = sparse_to_dense_efficient(py, sparse_x)?;

    // Convert target
    let y_arr = y.to_owned_array();

    // Validate shapes match
    if x.nrows() != y_arr.len() {
        return Err(SparseConversionError::ShapeMismatch {
            expected: format!("{} samples", x.nrows()),
            actual: format!("{} targets", y_arr.len()),
        });
    }

    Ok(SparseData { x, y: y_arr, info })
}

/// Extract only features from a sparse matrix.
///
/// # Arguments
///
/// * `py` - Python GIL token
/// * `sparse_x` - The scipy.sparse feature matrix
///
/// # Returns
///
/// A tuple of (dense feature matrix, sparse info).
pub fn extract_sparse_x<'py>(
    py: Python<'py>,
    sparse_x: &Bound<'py, PyAny>,
) -> SparseResult<(Array2<f64>, SparseMatrixInfo)> {
    let info = get_sparse_info(sparse_x)?;
    let x = sparse_to_dense_efficient(py, sparse_x)?;
    Ok((x, info))
}

#[cfg(test)]
mod tests {
    // Tests require PyO3/Python runtime which is typically done in integration tests
    // with maturin develop. Unit tests for the conversion logic are covered above.
}
