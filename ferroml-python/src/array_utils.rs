//! NumPy array conversion utilities with zero-copy optimization.
//!
//! This module provides optimized utilities for converting between NumPy arrays and ndarray types,
//! minimizing memory copies where possible.
//!
//! ## Zero-Copy Semantics
//!
//! ### Input Arrays (Python → Rust)
//!
//! When receiving NumPy arrays from Python, we use `PyReadonlyArray` which provides a read-only
//! view of the Python array data. The conversion behavior depends on the array's memory layout:
//!
//! - **C-contiguous arrays**: Direct view without copy (when using `as_array()`)
//! - **Non-contiguous arrays**: May require copy for certain operations
//!
//! The `try_as_array_view` function returns an `ArrayView` without copying when possible.
//! Use this when the Rust code can work with views instead of owned arrays.
//!
//! For operations that require owned arrays (e.g., `Model::fit`), use `to_owned_array`
//! which efficiently handles the conversion.
//!
//! ### Output Arrays (Rust → Python)
//!
//! When returning arrays to Python:
//!
//! - `into_pyarray`: Moves ownership to Python (no copy of data, just metadata allocation)
//! - `to_pyarray`: Creates a copy (use when you need to keep the Rust array)
//!
//! ## Performance Notes
//!
//! - The ferroml-core Model trait accepts `&Array2<f64>`, which requires owned arrays.
//!   This means input data must be copied when calling fit/predict.
//! - For read-only inspection of data (e.g., shape checks, statistics), use `as_array_view`
//!   to avoid unnecessary copies.
//! - Output arrays use `into_pyarray` which transfers ownership efficiently.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
    ToPyArray,
};
use pyo3::prelude::*;

/// Check that a 2D NumPy array contains no NaN or Inf values.
/// Raises ValueError (not RuntimeError) consistent with sklearn conventions.
pub fn check_array_finite(x: &PyReadonlyArray2<f64>) -> PyResult<()> {
    let arr = x.as_array();
    let ncols = arr.ncols();
    for (idx, val) in arr.iter().enumerate() {
        if !val.is_finite() {
            let row = idx / ncols;
            let col = idx % ncols;
            let kind = if val.is_nan() { "NaN" } else { "Inf" };
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Input array contains {} at position ({}, {})",
                kind, row, col
            )));
        }
    }
    Ok(())
}

/// Check that a 1D NumPy array contains no NaN or Inf values.
/// Raises ValueError (not RuntimeError) consistent with sklearn conventions.
pub fn check_array1_finite(x: &PyReadonlyArray1<f64>) -> PyResult<()> {
    let arr = x.as_array();
    for (idx, val) in arr.iter().enumerate() {
        if !val.is_finite() {
            let kind = if val.is_nan() { "NaN" } else { "Inf" };
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Input array contains {} at position {}",
                kind, idx
            )));
        }
    }
    Ok(())
}

/// Convert a PyReadonlyArray2 to an ArrayView2 (zero-copy).
///
/// This is the most efficient way to read NumPy array data in Rust when you only
/// need read-only access and don't need to own the data.
///
/// # Example
///
/// ```ignore
/// fn compute_mean(x: PyReadonlyArray2<'_, f64>) -> f64 {
///     let view = as_array_view_2d(&x);
///     view.mean().unwrap_or(0.0)
/// }
/// ```
#[inline]
pub fn as_array_view_2d<'a>(x: &'a PyReadonlyArray2<'a, f64>) -> ArrayView2<'a, f64> {
    x.as_array()
}

/// Convert a PyReadonlyArray1 to an ArrayView1 (zero-copy).
///
/// This is the most efficient way to read NumPy array data in Rust when you only
/// need read-only access and don't need to own the data.
#[inline]
pub fn as_array_view_1d<'a>(x: &'a PyReadonlyArray1<'a, f64>) -> ArrayView1<'a, f64> {
    x.as_array()
}

/// Convert a PyReadonlyArray2 to an owned Array2.
///
/// This function creates an owned copy of the data. Use this when the Rust API
/// requires `&Array2<f64>` (which is most ferroml-core functions).
///
/// # When to use
///
/// - When calling `Model::fit()` or `Model::predict()`
/// - When you need to modify the array
/// - When you need the data to outlive the Python array reference
///
/// # Performance
///
/// This always copies the data. For large arrays, consider whether you can use
/// `as_array_view_2d` instead.
#[inline]
pub fn to_owned_array_2d(x: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    x.as_array().to_owned()
}

/// Convert a PyReadonlyArray1 to an owned Array1.
///
/// This function creates an owned copy of the data. See `to_owned_array_2d` for details.
#[inline]
pub fn to_owned_array_1d(x: PyReadonlyArray1<'_, f64>) -> Array1<f64> {
    x.as_array().to_owned()
}

/// Convert an owned Array1 to a NumPy array, transferring ownership.
///
/// This is efficient as it transfers ownership rather than copying data.
/// The Rust array is consumed and becomes owned by Python.
#[inline]
pub fn array1_into_pyarray<'py>(py: Python<'py>, arr: Array1<f64>) -> Bound<'py, PyArray1<f64>> {
    arr.into_pyarray(py)
}

/// Convert an owned Array2 to a NumPy array, transferring ownership.
///
/// This is efficient as it transfers ownership rather than copying data.
/// The Rust array is consumed and becomes owned by Python.
#[inline]
pub fn array2_into_pyarray<'py>(py: Python<'py>, arr: Array2<f64>) -> Bound<'py, PyArray2<f64>> {
    arr.into_pyarray(py)
}

/// Copy an Array1 reference to a NumPy array.
///
/// Use this when you need to keep the Rust array and also return a copy to Python.
/// For most cases, prefer `array1_into_pyarray` which avoids the copy.
#[inline]
pub fn array1_to_pyarray<'py>(py: Python<'py>, arr: &Array1<f64>) -> Bound<'py, PyArray1<f64>> {
    arr.to_pyarray(py)
}

/// Copy an Array2 reference to a NumPy array.
///
/// Use this when you need to keep the Rust array and also return a copy to Python.
/// For most cases, prefer `array2_into_pyarray` which avoids the copy.
#[inline]
pub fn array2_to_pyarray<'py>(py: Python<'py>, arr: &Array2<f64>) -> Bound<'py, PyArray2<f64>> {
    arr.to_pyarray(py)
}

/// Get the shape of a 2D array without copying.
#[inline]
pub fn shape_2d(x: &PyReadonlyArray2<'_, f64>) -> (usize, usize) {
    let shape = x.shape();
    (shape[0], shape[1])
}

/// Get the shape of a 1D array without copying.
#[inline]
pub fn shape_1d(x: &PyReadonlyArray1<'_, f64>) -> usize {
    x.shape()[0]
}

/// Check if a 2D array is C-contiguous (row-major order).
///
/// C-contiguous arrays can be accessed more efficiently.
#[inline]
pub fn is_c_contiguous_2d(x: &PyReadonlyArray2<'_, f64>) -> bool {
    x.is_c_contiguous()
}

/// Check if a 1D array is contiguous.
#[inline]
pub fn is_contiguous_1d(x: &PyReadonlyArray1<'_, f64>) -> bool {
    x.is_contiguous()
}

/// Integer array conversion utilities for classification labels.
///
/// Many classification functions return integer labels. These utilities
/// handle the conversion efficiently.

/// Convert a PyReadonlyArray1<i64> to an owned Array1<i64>.
#[inline]
pub fn to_owned_array_1d_i64(x: PyReadonlyArray1<'_, i64>) -> Array1<i64> {
    x.as_array().to_owned()
}

/// Convert any numeric 1D array (int or float) to an owned Array1<f64>.
///
/// This is useful for target arrays in classification, which may be passed
/// as integer arrays (e.g., `np.array([0, 1, 2])`) but need to be converted
/// to f64 for the Rust model.
///
/// Supports: f64, f32, i64, i32, i16, i8, u64, u32, u16, u8
pub fn py_array_to_f64_1d<'py>(py: Python<'py>, arr: &Bound<'py, PyAny>) -> PyResult<Array1<f64>> {
    // Try to get numpy module
    let np = py.import("numpy")?;

    // Convert to float64 array using numpy's astype
    let arr_f64 = arr.call_method1("astype", (np.getattr("float64")?,))?;

    // Extract as PyReadonlyArray1<f64>
    let readonly: PyReadonlyArray1<'py, f64> = arr_f64.extract()?;

    Ok(to_owned_array_1d(readonly))
}

/// Convert an Array1<i64> to a NumPy array, transferring ownership.
#[inline]
pub fn array1_i64_into_pyarray<'py>(
    py: Python<'py>,
    arr: Array1<i64>,
) -> Bound<'py, PyArray1<i64>> {
    arr.into_pyarray(py)
}

/// Convert an Array1<usize> to a NumPy array as i64.
///
/// Python/NumPy uses i64 for integer arrays, so we convert usize to i64.
#[inline]
pub fn array1_usize_into_pyarray_i64<'py>(
    py: Python<'py>,
    arr: Array1<usize>,
) -> Bound<'py, PyArray1<i64>> {
    arr.mapv(|x| x as i64).into_pyarray(py)
}

#[cfg(test)]
mod tests {
    // Tests would require a Python interpreter, which is typically done
    // in integration tests with maturin develop.
}
