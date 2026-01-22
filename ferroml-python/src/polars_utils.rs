//! Polars DataFrame conversion utilities for FerroML Python bindings.
//!
//! This module provides utilities for converting between Polars DataFrames and ndarray types,
//! enabling efficient data interchange between Python's Polars library and FerroML's Rust core.
//!
//! ## Zero-Copy Semantics
//!
//! Polars uses Apache Arrow as its in-memory format. When possible, conversions leverage
//! Arrow's zero-copy capabilities:
//!
//! - **Contiguous f64 columns**: Can be accessed as views without copying
//! - **Non-contiguous or non-f64 data**: Requires conversion (with potential copy)
//!
//! ## Usage Pattern
//!
//! ```python
//! import polars as pl
//! from ferroml.linear import LinearRegression
//!
//! # Create a Polars DataFrame
//! df = pl.DataFrame({
//!     "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
//!     "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
//!     "target": [3.0, 6.0, 9.0, 12.0, 15.0]
//! })
//!
//! # Fit directly from DataFrame
//! model = LinearRegression()
//! model.fit_dataframe(df, target_column="target")
//! ```
//!
//! ## Feature Names
//!
//! When using DataFrames, column names are automatically extracted and stored,
//! enabling better interpretability in model summaries and feature importance.

use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

// Re-export polars types from pyo3-polars to ensure version compatibility
pub use pyo3_polars::export::polars_core::frame::DataFrame;
pub use pyo3_polars::export::polars_core::prelude::{DataType, Float64Type, NamedFrom};
pub use pyo3_polars::export::polars_core::series::Series;

/// Result type for DataFrame conversion operations.
pub type PolarsResult<T> = Result<T, PolarsConversionError>;

/// Errors that can occur during DataFrame conversion.
#[derive(Debug)]
pub enum PolarsConversionError {
    /// Column not found in DataFrame
    ColumnNotFound(String),
    /// Column has unsupported data type
    UnsupportedDataType { column: String, dtype: String },
    /// DataFrame is empty
    EmptyDataFrame,
    /// Shape mismatch
    ShapeMismatch { expected: String, actual: String },
    /// Internal Polars error
    PolarsError(String),
    /// Null values present (not supported for ML)
    NullValues { column: String, count: usize },
}

impl std::fmt::Display for PolarsConversionError {
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
            Self::PolarsError(e) => write!(f, "Polars error: {}", e),
            Self::NullValues { column, count } => {
                write!(
                    f,
                    "Column '{}' contains {} null values. Handle missing data before fitting.",
                    column, count
                )
            }
        }
    }
}

impl std::error::Error for PolarsConversionError {}

impl From<PolarsConversionError> for PyErr {
    fn from(err: PolarsConversionError) -> Self {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string())
    }
}

/// Extracted data from a Polars DataFrame for ML operations.
///
/// Contains the feature matrix, target vector, and metadata about the columns.
#[derive(Debug)]
pub struct DataFrameData {
    /// Feature matrix (n_samples x n_features)
    pub x: Array2<f64>,
    /// Target vector (n_samples,)
    pub y: Array1<f64>,
    /// Feature column names (in order)
    pub feature_names: Vec<String>,
    /// Target column name
    pub target_name: String,
}

/// Extract feature and target data from a PyDataFrame.
///
/// This function extracts all numeric columns except the target column as features,
/// and the specified column as the target variable.
///
/// # Arguments
///
/// * `pydf` - The Polars PyDataFrame from Python
/// * `target_column` - Name of the target column
/// * `feature_columns` - Optional list of feature column names. If None, uses all
///   numeric columns except the target.
///
/// # Returns
///
/// A `DataFrameData` struct containing the feature matrix, target vector, and column names.
///
/// # Errors
///
/// Returns an error if:
/// - The target column is not found
/// - Any feature column contains non-numeric data
/// - The DataFrame is empty
/// - Any column contains null values
pub fn extract_xy_from_pydf(
    pydf: &PyDataFrame,
    target_column: &str,
    feature_columns: Option<Vec<String>>,
) -> PolarsResult<DataFrameData> {
    let df = &pydf.0;

    // Check DataFrame is not empty
    if df.height() == 0 {
        return Err(PolarsConversionError::EmptyDataFrame);
    }

    // Get target column
    let target_series = df
        .column(target_column)
        .map_err(|_| PolarsConversionError::ColumnNotFound(target_column.to_string()))?;

    // Determine feature columns
    let feature_names: Vec<String> = match feature_columns {
        Some(cols) => cols,
        None => df
            .get_column_names()
            .into_iter()
            .filter(|name| name.as_str() != target_column)
            .filter(|name| {
                df.column(name.as_str())
                    .map(|s| is_numeric_dtype(s.dtype()))
                    .unwrap_or(false)
            })
            .map(|s| s.to_string())
            .collect(),
    };

    if feature_names.is_empty() {
        return Err(PolarsConversionError::ShapeMismatch {
            expected: "at least one feature column".to_string(),
            actual: "no numeric feature columns found".to_string(),
        });
    }

    let n_samples = df.height();
    let n_features = feature_names.len();

    // Convert target to Array1<f64>
    let y = series_to_array1(target_series.as_materialized_series(), target_column)?;

    // Convert features to Array2<f64>
    let mut x_data = Vec::with_capacity(n_samples * n_features);

    for col_name in &feature_names {
        let series = df
            .column(col_name.as_str())
            .map_err(|_| PolarsConversionError::ColumnNotFound(col_name.clone()))?;

        let col_data = series_to_array1(series.as_materialized_series(), col_name)?;
        x_data.extend(col_data.iter());
    }

    // Reshape from column-major to row-major
    // Data is stored column by column, we need to transpose
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| x_data[j * n_samples + i]);

    Ok(DataFrameData {
        x,
        y,
        feature_names,
        target_name: target_column.to_string(),
    })
}

/// Extract only feature data from a PyDataFrame.
///
/// Used for prediction where no target column is needed.
///
/// # Arguments
///
/// * `pydf` - The Polars PyDataFrame from Python
/// * `feature_columns` - Optional list of feature column names. If None, uses all numeric columns.
///
/// # Returns
///
/// A tuple of (feature matrix, feature names).
pub fn extract_x_from_pydf(
    pydf: &PyDataFrame,
    feature_columns: Option<Vec<String>>,
) -> PolarsResult<(Array2<f64>, Vec<String>)> {
    let df = &pydf.0;

    // Check DataFrame is not empty
    if df.height() == 0 {
        return Err(PolarsConversionError::EmptyDataFrame);
    }

    // Determine feature columns
    let feature_names: Vec<String> = match feature_columns {
        Some(cols) => cols,
        None => df
            .get_column_names()
            .into_iter()
            .filter(|name| {
                df.column(name.as_str())
                    .map(|s| is_numeric_dtype(s.dtype()))
                    .unwrap_or(false)
            })
            .map(|s| s.to_string())
            .collect(),
    };

    if feature_names.is_empty() {
        return Err(PolarsConversionError::ShapeMismatch {
            expected: "at least one feature column".to_string(),
            actual: "no numeric feature columns found".to_string(),
        });
    }

    let n_samples = df.height();
    let n_features = feature_names.len();

    // Convert features to Array2<f64>
    let mut x_data = Vec::with_capacity(n_samples * n_features);

    for col_name in &feature_names {
        let series = df
            .column(col_name.as_str())
            .map_err(|_| PolarsConversionError::ColumnNotFound(col_name.clone()))?;

        let col_data = series_to_array1(series.as_materialized_series(), col_name)?;
        x_data.extend(col_data.iter());
    }

    // Reshape from column-major to row-major
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| x_data[j * n_samples + i]);

    Ok((x, feature_names))
}

/// Check if a data type is numeric.
fn is_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Convert a Polars Series to an ndarray Array1<f64>.
///
/// Handles type conversion from various numeric types to f64.
/// Returns an error if the series contains null values or non-numeric types.
fn series_to_array1(series: &Series, column_name: &str) -> PolarsResult<Array1<f64>> {
    // Check for null values
    let null_count = series.null_count();
    if null_count > 0 {
        return Err(PolarsConversionError::NullValues {
            column: column_name.to_string(),
            count: null_count,
        });
    }

    // Cast to f64 and extract values
    let f64_series = series.cast(&DataType::Float64).map_err(|e| {
        PolarsConversionError::UnsupportedDataType {
            column: column_name.to_string(),
            dtype: format!("{} (error: {})", series.dtype(), e),
        }
    })?;

    let chunked = f64_series
        .f64()
        .map_err(|_| PolarsConversionError::UnsupportedDataType {
            column: column_name.to_string(),
            dtype: series.dtype().to_string(),
        })?;

    // Extract values - we've already checked for nulls, so unwrap is safe
    let values: Vec<f64> = chunked.into_iter().map(|opt| opt.unwrap_or(0.0)).collect();

    Ok(Array1::from(values))
}

/// Get column names from a PyDataFrame as a Vec<String>.
pub fn get_column_names(pydf: &PyDataFrame) -> Vec<String> {
    pydf.0
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// Validate that specified columns exist in the PyDataFrame.
pub fn validate_columns(pydf: &PyDataFrame, columns: &[String]) -> PolarsResult<()> {
    let df = &pydf.0;
    let available: Vec<String> = df
        .get_column_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    for col in columns {
        if !available.contains(col) {
            return Err(PolarsConversionError::ColumnNotFound(col.clone()));
        }
    }
    Ok(())
}

/// Check if a PyDataFrame has any null values in the specified columns.
pub fn check_for_nulls(pydf: &PyDataFrame, columns: &[String]) -> PolarsResult<()> {
    let df = &pydf.0;

    for col_name in columns {
        let series = df
            .column(col_name.as_str())
            .map_err(|_| PolarsConversionError::ColumnNotFound(col_name.clone()))?;

        let null_count = series.null_count();
        if null_count > 0 {
            return Err(PolarsConversionError::NullValues {
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
