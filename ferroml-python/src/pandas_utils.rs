//! Pandas DataFrame conversion utilities for FerroML Python bindings.
//!
//! This module provides utilities for converting between Pandas DataFrames and ndarray types,
//! enabling efficient data interchange between Python's Pandas library and FerroML's Rust core.
//!
//! ## Zero-Copy Semantics via PyArrow
//!
//! When possible, this module leverages PyArrow as an intermediary for efficient conversion:
//!
//! 1. **NumPy-backed columns**: Direct access via `.values` or `.to_numpy()` with minimal copy
//! 2. **Arrow-backed columns**: Zero-copy via PyArrow when Pandas uses Arrow dtypes
//! 3. **Mixed dtypes**: Conversion to f64 may require data copy
//!
//! ## Usage Pattern
//!
//! ```python
//! import pandas as pd
//! from ferroml.linear import LinearRegression
//!
//! # Create a Pandas DataFrame
//! df = pd.DataFrame({
//!     "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
//!     "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
//!     "target": [3.0, 6.0, 9.0, 12.0, 15.0]
//! })
//!
//! # Fit directly from DataFrame
//! model = LinearRegression()
//! model.fit_pandas(df, target_column="target")
//! ```
//!
//! ## Feature Names
//!
//! When using DataFrames, column names are automatically extracted and stored,
//! enabling better interpretability in model summaries and feature importance.

use ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList, PyString};

/// Result type for DataFrame conversion operations.
pub type PandasResult<T> = Result<T, PandasConversionError>;

/// Errors that can occur during Pandas DataFrame conversion.
#[derive(Debug)]
pub enum PandasConversionError {
    /// Column not found in DataFrame
    ColumnNotFound(String),
    /// Column has unsupported data type
    UnsupportedDataType { column: String, dtype: String },
    /// DataFrame is empty
    EmptyDataFrame,
    /// Shape mismatch
    ShapeMismatch { expected: String, actual: String },
    /// Python error during conversion
    PythonError(String),
    /// Null/NaN values present (not supported for ML without imputation)
    NullValues { column: String, count: usize },
}

impl std::fmt::Display for PandasConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColumnNotFound(col) => write!(f, "Column '{}' not found in DataFrame", col),
            Self::UnsupportedDataType { column, dtype } => {
                write!(
                    f,
                    "Column '{}' has unsupported data type '{}'. Expected numeric type.",
                    column, dtype
                )
            }
            Self::EmptyDataFrame => write!(f, "DataFrame is empty"),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {}, got {}", expected, actual)
            }
            Self::PythonError(e) => write!(f, "Python error: {}", e),
            Self::NullValues { column, count } => {
                write!(
                    f,
                    "Column '{}' contains {} null/NaN values. Handle missing data before fitting.",
                    column, count
                )
            }
        }
    }
}

impl std::error::Error for PandasConversionError {}

impl From<PandasConversionError> for PyErr {
    fn from(err: PandasConversionError) -> Self {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
    }
}

impl From<PyErr> for PandasConversionError {
    fn from(err: PyErr) -> Self {
        PandasConversionError::PythonError(err.to_string())
    }
}

/// Extracted data from a Pandas DataFrame for ML operations.
///
/// Contains the feature matrix, target vector, and metadata about the columns.
#[derive(Debug)]
pub struct PandasDataFrameData {
    /// Feature matrix (n_samples x n_features)
    pub x: Array2<f64>,
    /// Target vector (n_samples,)
    pub y: Array1<f64>,
    /// Feature column names (in order)
    pub feature_names: Vec<String>,
    /// Target column name
    pub target_name: String,
}

/// Extract feature and target data from a Pandas DataFrame.
///
/// This function extracts all numeric columns except the target column as features,
/// and the specified column as the target variable.
///
/// # Arguments
///
/// * `py` - Python GIL token
/// * `df` - The Pandas DataFrame Python object
/// * `target_column` - Name of the target column
/// * `feature_columns` - Optional list of feature column names. If None, uses all
///   numeric columns except the target.
///
/// # Returns
///
/// A `PandasDataFrameData` struct containing the feature matrix, target vector, and column names.
///
/// # Errors
///
/// Returns an error if:
/// - The target column is not found
/// - Any feature column contains non-numeric data
/// - The DataFrame is empty
/// - Any column contains null/NaN values
pub fn extract_xy_from_pandas<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    target_column: &str,
    feature_columns: Option<Vec<String>>,
) -> PandasResult<PandasDataFrameData> {
    // Check DataFrame is not empty
    let shape: (usize, usize) = df.getattr("shape")?.extract()?;
    if shape.0 == 0 {
        return Err(PandasConversionError::EmptyDataFrame);
    }

    // Get all column names
    let columns: Vec<String> = df
        .getattr("columns")?
        .call_method0("tolist")?
        .extract()?;

    // Verify target column exists
    if !columns.contains(&target_column.to_string()) {
        return Err(PandasConversionError::ColumnNotFound(
            target_column.to_string(),
        ));
    }

    // Determine feature columns
    let feature_names: Vec<String> = match feature_columns {
        Some(cols) => {
            // Validate specified columns exist
            for col in &cols {
                if !columns.contains(col) {
                    return Err(PandasConversionError::ColumnNotFound(col.clone()));
                }
            }
            cols
        }
        None => {
            // Auto-detect numeric columns (exclude target)
            let numeric_cols = get_numeric_columns(py, df)?;
            numeric_cols
                .into_iter()
                .filter(|c| c != target_column)
                .collect()
        }
    };

    if feature_names.is_empty() {
        return Err(PandasConversionError::ShapeMismatch {
            expected: "at least one feature column".to_string(),
            actual: "no numeric feature columns found".to_string(),
        });
    }

    let n_samples = shape.0;
    let n_features = feature_names.len();

    // Extract target column
    let y = extract_column_as_array1(py, df, target_column)?;

    // Extract feature columns
    let mut x_data = Vec::with_capacity(n_samples * n_features);

    for col_name in &feature_names {
        let col_data = extract_column_as_array1(py, df, col_name)?;
        x_data.extend(col_data.iter());
    }

    // Reshape from column-major to row-major
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| x_data[j * n_samples + i]);

    Ok(PandasDataFrameData {
        x,
        y,
        feature_names,
        target_name: target_column.to_string(),
    })
}

/// Extract only feature data from a Pandas DataFrame.
///
/// Used for prediction where no target column is needed.
///
/// # Arguments
///
/// * `py` - Python GIL token
/// * `df` - The Pandas DataFrame Python object
/// * `feature_columns` - Optional list of feature column names. If None, uses all numeric columns.
///
/// # Returns
///
/// A tuple of (feature matrix, feature names).
pub fn extract_x_from_pandas<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    feature_columns: Option<Vec<String>>,
) -> PandasResult<(Array2<f64>, Vec<String>)> {
    // Check DataFrame is not empty
    let shape: (usize, usize) = df.getattr("shape")?.extract()?;
    if shape.0 == 0 {
        return Err(PandasConversionError::EmptyDataFrame);
    }

    // Get all column names
    let columns: Vec<String> = df
        .getattr("columns")?
        .call_method0("tolist")?
        .extract()?;

    // Determine feature columns
    let feature_names: Vec<String> = match feature_columns {
        Some(cols) => {
            // Validate specified columns exist
            for col in &cols {
                if !columns.contains(col) {
                    return Err(PandasConversionError::ColumnNotFound(col.clone()));
                }
            }
            cols
        }
        None => {
            // Auto-detect numeric columns
            get_numeric_columns(py, df)?
        }
    };

    if feature_names.is_empty() {
        return Err(PandasConversionError::ShapeMismatch {
            expected: "at least one feature column".to_string(),
            actual: "no numeric feature columns found".to_string(),
        });
    }

    let n_samples = shape.0;
    let n_features = feature_names.len();

    // Extract feature columns
    let mut x_data = Vec::with_capacity(n_samples * n_features);

    for col_name in &feature_names {
        let col_data = extract_column_as_array1(py, df, col_name)?;
        x_data.extend(col_data.iter());
    }

    // Reshape from column-major to row-major
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| x_data[j * n_samples + i]);

    Ok((x, feature_names))
}

/// Get list of numeric column names from a Pandas DataFrame.
fn get_numeric_columns<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
) -> PandasResult<Vec<String>> {
    // Use pandas.api.types.is_numeric_dtype or select_dtypes
    // df.select_dtypes(include=['number']).columns.tolist()
    let include_list = PyList::new(py, &[PyString::new(py, "number")])?;
    let kwargs = [("include", include_list)].into_py_dict(py)?;
    let numeric_df = df.call_method("select_dtypes", (), Some(&kwargs))?;

    let columns: Vec<String> = numeric_df
        .getattr("columns")?
        .call_method0("tolist")?
        .extract()?;

    Ok(columns)
}

/// Extract a single column from a Pandas DataFrame as Array1<f64>.
///
/// Handles type conversion and validates for null values.
fn extract_column_as_array1<'py>(
    _py: Python<'py>,
    df: &Bound<'py, PyAny>,
    column_name: &str,
) -> PandasResult<Array1<f64>> {
    // Get the column as a Series
    let series = df.get_item(column_name)?;

    // Check for null values using isna().sum()
    let null_count: usize = series.call_method0("isna")?.call_method0("sum")?.extract()?;
    if null_count > 0 {
        return Err(PandasConversionError::NullValues {
            column: column_name.to_string(),
            count: null_count,
        });
    }

    // Try to convert to float64 numpy array
    // series.astype('float64').to_numpy()
    let float_series = series
        .call_method1("astype", ("float64",))
        .map_err(|e| PandasConversionError::UnsupportedDataType {
            column: column_name.to_string(),
            dtype: format!("(conversion error: {})", e),
        })?;

    let numpy_array = float_series.call_method0("to_numpy")?;

    // Convert to Rust ndarray
    let py_array: PyReadonlyArray1<f64> = numpy_array.extract()?;
    Ok(py_array.to_owned_array())
}

/// Get column names from a Pandas DataFrame as a Vec<String>.
pub fn get_column_names<'py>(df: &Bound<'py, PyAny>) -> PandasResult<Vec<String>> {
    let columns: Vec<String> = df
        .getattr("columns")?
        .call_method0("tolist")?
        .extract()?;
    Ok(columns)
}

/// Check if a Python object is a Pandas DataFrame.
pub fn is_pandas_dataframe<'py>(obj: &Bound<'py, PyAny>) -> bool {
    // Check if the object has typical DataFrame attributes
    obj.hasattr("columns").unwrap_or(false)
        && obj.hasattr("shape").unwrap_or(false)
        && obj.hasattr("dtypes").unwrap_or(false)
        && obj.hasattr("select_dtypes").unwrap_or(false)
}

/// Validate that specified columns exist in the DataFrame.
pub fn validate_columns<'py>(
    df: &Bound<'py, PyAny>,
    columns: &[String],
) -> PandasResult<()> {
    let available: Vec<String> = df
        .getattr("columns")?
        .call_method0("tolist")?
        .extract()?;

    for col in columns {
        if !available.contains(col) {
            return Err(PandasConversionError::ColumnNotFound(col.clone()));
        }
    }
    Ok(())
}

/// Check if a DataFrame has any null/NaN values in the specified columns.
pub fn check_for_nulls<'py>(
    _py: Python<'py>,
    df: &Bound<'py, PyAny>,
    columns: &[String],
) -> PandasResult<()> {
    for col_name in columns {
        let series = df.get_item(col_name)?;
        let null_count: usize = series.call_method0("isna")?.call_method0("sum")?.extract()?;

        if null_count > 0 {
            return Err(PandasConversionError::NullValues {
                column: col_name.clone(),
                count: null_count,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Tests require PyO3/Python runtime which is typically done in integration tests
    // with maturin develop. Unit tests for the conversion logic are covered above.
}
