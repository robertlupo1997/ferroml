//! Data loading utilities for CSV and Parquet files.
//!
//! This module provides utilities for loading datasets from common file formats.
//! It uses Polars for efficient file parsing and type inference.
//!
//! # Supported Formats
//!
//! - **CSV**: Comma-separated values with automatic delimiter detection
//! - **Parquet**: Apache Parquet columnar format for efficient storage
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::datasets::{load_csv, load_parquet, CsvOptions};
//!
//! // Load CSV with automatic type inference
//! let (dataset, info) = load_csv("data.csv", "target_column")?;
//!
//! // Load with custom options
//! let opts = CsvOptions::new()
//!     .with_delimiter(b';')
//!     .with_has_header(true);
//! let (dataset, info) = load_csv_with_options("data.csv", "target", opts)?;
//!
//! // Load Parquet file
//! let (dataset, info) = load_parquet("data.parquet", "target")?;
//! ```

use crate::{FerroError, Result};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::path::Path;
use std::sync::Arc;

use super::{Dataset, DatasetInfo};

/// Options for loading CSV files.
#[derive(Debug, Clone)]
pub struct CsvOptions {
    /// Delimiter character (default: comma)
    pub delimiter: u8,
    /// Whether the file has a header row (default: true)
    pub has_header: bool,
    /// Number of rows to skip at the beginning (default: 0)
    pub skip_rows: usize,
    /// Maximum number of rows to read (default: None, read all)
    pub n_rows: Option<usize>,
    /// Columns to select (by name). If None, all numeric columns are used.
    pub columns: Option<Vec<String>>,
    /// How to handle missing values (default: keep as NaN)
    pub null_values: Option<Vec<String>>,
    /// Whether to infer schema from the first n rows (default: 100)
    pub infer_schema_length: Option<usize>,
    /// Whether to try parsing dates (default: false)
    pub try_parse_dates: bool,
    /// Encoding (default: UTF-8)
    pub encoding: CsvEncoding,
}

/// CSV encoding options.
#[derive(Debug, Clone, Copy, Default)]
pub enum CsvEncoding {
    /// UTF-8 encoding
    #[default]
    Utf8,
    /// UTF-8 with lossy replacement for invalid bytes
    Utf8Lossy,
}

impl Default for CsvOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: true,
            skip_rows: 0,
            n_rows: None,
            columns: None,
            null_values: None,
            infer_schema_length: Some(100),
            try_parse_dates: false,
            encoding: CsvEncoding::Utf8,
        }
    }
}

impl CsvOptions {
    /// Create new CSV options with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the delimiter character.
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// Set whether the file has a header row.
    pub fn with_has_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Set the number of rows to skip at the beginning.
    pub fn with_skip_rows(mut self, skip_rows: usize) -> Self {
        self.skip_rows = skip_rows;
        self
    }

    /// Set the maximum number of rows to read.
    pub fn with_n_rows(mut self, n_rows: usize) -> Self {
        self.n_rows = Some(n_rows);
        self
    }

    /// Set columns to select.
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Set null value representations.
    pub fn with_null_values(mut self, null_values: Vec<String>) -> Self {
        self.null_values = Some(null_values);
        self
    }

    /// Set schema inference length.
    pub fn with_infer_schema_length(mut self, length: Option<usize>) -> Self {
        self.infer_schema_length = length;
        self
    }

    /// Set whether to try parsing dates.
    pub fn with_try_parse_dates(mut self, try_parse_dates: bool) -> Self {
        self.try_parse_dates = try_parse_dates;
        self
    }

    /// Set the encoding.
    pub fn with_encoding(mut self, encoding: CsvEncoding) -> Self {
        self.encoding = encoding;
        self
    }
}

/// Options for loading Parquet files.
#[derive(Debug, Clone, Default)]
pub struct ParquetOptions {
    /// Columns to select (by name). If None, all numeric columns are used.
    pub columns: Option<Vec<String>>,
    /// Whether to use parallel reading (default: true)
    pub parallel: bool,
}

impl ParquetOptions {
    /// Create new Parquet options with defaults.
    pub fn new() -> Self {
        Self {
            parallel: true,
            ..Default::default()
        }
    }

    /// Set columns to select.
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Set whether to use parallel reading.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// Load a CSV file as a Dataset.
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `target_column` - Name of the target column (y). If None, uses the last numeric column.
///
/// # Returns
///
/// A tuple of (Dataset, DatasetInfo)
///
/// # Example
///
/// ```ignore
/// use ferroml_core::datasets::load_csv;
///
/// let (dataset, info) = load_csv("data.csv", Some("target"))?;
/// println!("Loaded {} samples with {} features", dataset.n_samples(), dataset.n_features());
/// ```
pub fn load_csv<P: AsRef<Path>>(
    path: P,
    target_column: Option<&str>,
) -> Result<(Dataset, DatasetInfo)> {
    load_csv_with_options(path, target_column, CsvOptions::default())
}

/// Load a CSV file with custom options.
///
/// # Arguments
///
/// * `path` - Path to the CSV file
/// * `target_column` - Name of the target column (y). If None, uses the last numeric column.
/// * `options` - CSV loading options
///
/// # Returns
///
/// A tuple of (Dataset, DatasetInfo)
pub fn load_csv_with_options<P: AsRef<Path>>(
    path: P,
    target_column: Option<&str>,
    options: CsvOptions,
) -> Result<(Dataset, DatasetInfo)> {
    let path = path.as_ref();

    // Build the CSV reader
    let mut reader = CsvReadOptions::default()
        .with_has_header(options.has_header)
        .with_skip_rows(options.skip_rows)
        .with_n_rows(options.n_rows)
        .with_infer_schema_length(options.infer_schema_length);

    // Set columns to select if specified
    if let Some(cols) = &options.columns {
        let pl_cols: Vec<PlSmallStr> = cols.iter().map(|s| PlSmallStr::from(s.as_str())).collect();
        reader = reader.with_columns(Some(Arc::from(pl_cols.into_boxed_slice())));
    }

    // Build parse options
    let mut parse_options = CsvParseOptions::default()
        .with_separator(options.delimiter)
        .with_try_parse_dates(options.try_parse_dates);

    // Set null values
    if let Some(nulls) = &options.null_values {
        let pl_nulls: Vec<PlSmallStr> = nulls.iter().map(|s| PlSmallStr::from(s.as_str())).collect();
        parse_options = parse_options.with_null_values(Some(NullValues::AllColumns(pl_nulls)));
    }

    // Set encoding
    match options.encoding {
        CsvEncoding::Utf8 => {}
        CsvEncoding::Utf8Lossy => {
            parse_options = parse_options.with_encoding(polars::prelude::CsvEncoding::LossyUtf8);
        }
    }

    reader = reader.with_parse_options(parse_options);

    // Read the file
    let df = reader
        .try_into_reader_with_file_path(Some(path.into()))
        .map_err(|e| FerroError::invalid_input(format!("Failed to create CSV reader: {}", e)))?
        .finish()
        .map_err(|e| FerroError::invalid_input(format!("Failed to read CSV file: {}", e)))?;

    dataframe_to_dataset(df, target_column, path)
}

/// Load a Parquet file as a Dataset.
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `target_column` - Name of the target column (y). If None, uses the last numeric column.
///
/// # Returns
///
/// A tuple of (Dataset, DatasetInfo)
///
/// # Example
///
/// ```ignore
/// use ferroml_core::datasets::load_parquet;
///
/// let (dataset, info) = load_parquet("data.parquet", Some("target"))?;
/// ```
pub fn load_parquet<P: AsRef<Path>>(
    path: P,
    target_column: Option<&str>,
) -> Result<(Dataset, DatasetInfo)> {
    load_parquet_with_options(path, target_column, ParquetOptions::new())
}

/// Load a Parquet file with custom options.
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `target_column` - Name of the target column (y). If None, uses the last numeric column.
/// * `options` - Parquet loading options
///
/// # Returns
///
/// A tuple of (Dataset, DatasetInfo)
pub fn load_parquet_with_options<P: AsRef<Path>>(
    path: P,
    target_column: Option<&str>,
    options: ParquetOptions,
) -> Result<(Dataset, DatasetInfo)> {
    let path = path.as_ref();

    // Build the Parquet reader
    let file = std::fs::File::open(path)?;

    let mut reader_builder = ParquetReader::new(file);

    // Set columns to select if specified
    if let Some(cols) = &options.columns {
        reader_builder = reader_builder.with_columns(Some(cols.clone()));
    }

    // Set parallel mode
    if options.parallel {
        reader_builder = reader_builder.read_parallel(ParallelStrategy::Auto);
    } else {
        reader_builder = reader_builder.read_parallel(ParallelStrategy::None);
    }

    // Read the file
    let df = reader_builder
        .finish()
        .map_err(|e| FerroError::invalid_input(format!("Failed to read Parquet file: {}", e)))?;

    dataframe_to_dataset(df, target_column, path)
}

/// Convert a Polars DataFrame to a Dataset.
fn dataframe_to_dataset<P: AsRef<Path>>(
    df: DataFrame,
    target_column: Option<&str>,
    source_path: P,
) -> Result<(Dataset, DatasetInfo)> {
    let path = source_path.as_ref();

    // Get all numeric columns
    let numeric_cols: Vec<String> = df
        .schema()
        .iter()
        .filter_map(|(name, dtype)| {
            if is_numeric_dtype(dtype) {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect();

    if numeric_cols.is_empty() {
        return Err(FerroError::invalid_input(
            "No numeric columns found in the dataset",
        ));
    }

    // Determine target column
    let target_col = if let Some(tc) = target_column {
        if !numeric_cols.contains(&tc.to_string()) {
            return Err(FerroError::invalid_input(format!(
                "Target column '{}' not found or not numeric. Available numeric columns: {:?}",
                tc, numeric_cols
            )));
        }
        tc.to_string()
    } else {
        // Use last numeric column as target
        numeric_cols.last().unwrap().clone()
    };

    // Get feature columns (all numeric except target)
    let feature_cols: Vec<String> = numeric_cols
        .iter()
        .filter(|&c| c != &target_col)
        .cloned()
        .collect();

    if feature_cols.is_empty() {
        return Err(FerroError::invalid_input(
            "No feature columns found after excluding target",
        ));
    }

    let n_samples = df.height();
    let n_features = feature_cols.len();

    // Reshape from column-major to row-major
    let mut x = Array2::zeros((n_samples, n_features));
    for (j, col_name) in feature_cols.iter().enumerate() {
        let series = df
            .column(col_name)
            .map_err(|e| FerroError::invalid_input(format!("Failed to get column '{}': {}", col_name, e)))?;
        let values = series_to_f64_vec(series)?;
        for (i, &val) in values.iter().enumerate() {
            x[[i, j]] = val;
        }
    }

    // Extract target vector
    let target_series = df
        .column(&target_col)
        .map_err(|e| FerroError::invalid_input(format!("Failed to get target column '{}': {}", target_col, e)))?;
    let y_vec = series_to_f64_vec(target_series)?;
    let y = Array1::from_vec(y_vec);

    // Create Dataset
    let dataset = Dataset::new(x, y).with_feature_names(feature_cols.clone());

    // Infer task type
    let task = dataset.infer_task();

    // Create DatasetInfo
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");

    let info = DatasetInfo::new(file_name, task, n_samples, n_features)
        .with_description(format!(
            "Dataset loaded from {}. Target column: '{}'",
            path.display(),
            target_col
        ))
        .with_feature_names(feature_cols)
        .with_source(path.display().to_string());

    Ok((dataset, info))
}

/// Check if a Polars DataType is numeric.
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

/// Convert a Polars Series to a `Vec<f64>`.
fn series_to_f64_vec(series: &Column) -> Result<Vec<f64>> {
    let series = series.as_materialized_series();
    match series.dtype() {
        DataType::Float64 => {
            let chunked = series.f64().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to f64: {}", e))
            })?;
            Ok(chunked.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        }
        DataType::Float32 => {
            let chunked = series.f32().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to f32: {}", e))
            })?;
            Ok(chunked
                .into_iter()
                .map(|v: Option<f32>| v.map(|f| f64::from(f)).unwrap_or(f64::NAN))
                .collect())
        }
        DataType::Int64 => {
            let chunked = series.i64().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to i64: {}", e))
            })?;
            #[allow(clippy::cast_precision_loss)]
            let result = chunked
                .into_iter()
                .map(|v: Option<i64>| v.map(|i| i as f64).unwrap_or(f64::NAN))
                .collect();
            Ok(result)
        }
        DataType::Int32 => {
            let chunked = series.i32().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to i32: {}", e))
            })?;
            Ok(chunked
                .into_iter()
                .map(|v: Option<i32>| v.map(|i| f64::from(i)).unwrap_or(f64::NAN))
                .collect())
        }
        DataType::Int16 => {
            let chunked = series.i16().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to i16: {}", e))
            })?;
            Ok(chunked
                .into_iter()
                .map(|v: Option<i16>| v.map(|i| f64::from(i)).unwrap_or(f64::NAN))
                .collect())
        }
        DataType::Int8 => {
            let chunked = series.i8().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to i8: {}", e))
            })?;
            Ok(chunked
                .into_iter()
                .map(|v: Option<i8>| v.map(|i| f64::from(i)).unwrap_or(f64::NAN))
                .collect())
        }
        DataType::UInt64 => {
            let chunked = series.u64().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to u64: {}", e))
            })?;
            #[allow(clippy::cast_precision_loss)]
            let result = chunked
                .into_iter()
                .map(|v: Option<u64>| v.map(|i| i as f64).unwrap_or(f64::NAN))
                .collect();
            Ok(result)
        }
        DataType::UInt32 => {
            let chunked = series.u32().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to u32: {}", e))
            })?;
            Ok(chunked
                .into_iter()
                .map(|v: Option<u32>| v.map(|i| f64::from(i)).unwrap_or(f64::NAN))
                .collect())
        }
        DataType::UInt16 => {
            let chunked = series.u16().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to u16: {}", e))
            })?;
            Ok(chunked
                .into_iter()
                .map(|v: Option<u16>| v.map(|i| f64::from(i)).unwrap_or(f64::NAN))
                .collect())
        }
        DataType::UInt8 => {
            let chunked = series.u8().map_err(|e| {
                FerroError::invalid_input(format!("Failed to cast to u8: {}", e))
            })?;
            Ok(chunked
                .into_iter()
                .map(|v: Option<u8>| v.map(|i| f64::from(i)).unwrap_or(f64::NAN))
                .collect())
        }
        dt => Err(FerroError::invalid_input(format!(
            "Cannot convert {:?} to f64",
            dt
        ))),
    }
}

/// Load a dataset from a file, automatically detecting the format.
///
/// Supports CSV and Parquet files based on file extension.
///
/// # Arguments
///
/// * `path` - Path to the data file
/// * `target_column` - Name of the target column
///
/// # Example
///
/// ```ignore
/// use ferroml_core::datasets::load_file;
///
/// // Automatically detects format from extension
/// let (dataset, info) = load_file("data.csv", Some("target"))?;
/// let (dataset, info) = load_file("data.parquet", Some("target"))?;
/// ```
pub fn load_file<P: AsRef<Path>>(
    path: P,
    target_column: Option<&str>,
) -> Result<(Dataset, DatasetInfo)> {
    let path = path.as_ref();

    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match extension.as_deref() {
        Some("csv") | Some("tsv") | Some("txt") => load_csv(path, target_column),
        Some("parquet") | Some("pq") => load_parquet(path, target_column),
        Some(ext) => Err(FerroError::invalid_input(format!(
            "Unsupported file extension: '{}'. Supported: csv, tsv, txt, parquet, pq",
            ext
        ))),
        None => Err(FerroError::invalid_input(
            "Cannot determine file format: no extension",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Task;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_options_builder() {
        let opts = CsvOptions::new()
            .with_delimiter(b';')
            .with_has_header(false)
            .with_skip_rows(1)
            .with_n_rows(100);

        assert_eq!(opts.delimiter, b';');
        assert!(!opts.has_header);
        assert_eq!(opts.skip_rows, 1);
        assert_eq!(opts.n_rows, Some(100));
    }

    #[test]
    fn test_parquet_options_builder() {
        let opts = ParquetOptions::new()
            .with_columns(vec!["a".to_string(), "b".to_string()])
            .with_parallel(false);

        assert_eq!(
            opts.columns,
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert!(!opts.parallel);
    }

    #[test]
    fn test_load_csv_simple() {
        // Create a temporary CSV file
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a,b,c,target").unwrap();
        writeln!(file, "1.0,2.0,3.0,0").unwrap();
        writeln!(file, "4.0,5.0,6.0,1").unwrap();
        writeln!(file, "7.0,8.0,9.0,0").unwrap();
        file.flush().unwrap();

        let (dataset, info) = load_csv(file.path(), Some("target")).unwrap();

        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 3);
        assert_eq!(info.feature_names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_load_csv_auto_target() {
        // Create a temporary CSV file without specifying target
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "x1,x2,y").unwrap();
        writeln!(file, "1.0,2.0,10.0").unwrap();
        writeln!(file, "3.0,4.0,20.0").unwrap();
        file.flush().unwrap();

        // Should use last numeric column (y) as target
        let (dataset, _info) = load_csv(file.path(), None).unwrap();

        assert_eq!(dataset.n_samples(), 2);
        assert_eq!(dataset.n_features(), 2);
        // y values should be [10.0, 20.0]
        assert!((dataset.y()[0] - 10.0).abs() < 1e-10);
        assert!((dataset.y()[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_csv_with_options() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a;b;target").unwrap();
        writeln!(file, "1;2;0").unwrap();
        writeln!(file, "3;4;1").unwrap();
        file.flush().unwrap();

        let opts = CsvOptions::new().with_delimiter(b';');
        let (dataset, _info) =
            load_csv_with_options(file.path(), Some("target"), opts).unwrap();

        assert_eq!(dataset.n_samples(), 2);
        assert_eq!(dataset.n_features(), 2);
    }

    #[test]
    fn test_load_csv_missing_values() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a,b,target").unwrap();
        writeln!(file, "1.0,2.0,0").unwrap();
        writeln!(file, ",5.0,1").unwrap(); // Missing value in first column
        writeln!(file, "7.0,8.0,0").unwrap();
        file.flush().unwrap();

        let (dataset, _info) = load_csv(file.path(), Some("target")).unwrap();

        assert_eq!(dataset.n_samples(), 3);
        // Missing value should be NaN
        assert!(dataset.x()[[1, 0]].is_nan());
    }

    #[test]
    fn test_load_csv_integer_columns() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a,b,target").unwrap();
        writeln!(file, "1,2,0").unwrap();
        writeln!(file, "3,4,1").unwrap();
        writeln!(file, "5,6,0").unwrap();
        file.flush().unwrap();

        let (dataset, _info) = load_csv(file.path(), Some("target")).unwrap();

        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
        // Should convert integers to f64
        assert!((dataset.x()[[0, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_csv_no_numeric_columns() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "name,category").unwrap();
        writeln!(file, "alice,A").unwrap();
        writeln!(file, "bob,B").unwrap();
        file.flush().unwrap();

        let result = load_csv(file.path(), Some("category"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_csv_invalid_target() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a,b,c").unwrap();
        writeln!(file, "1,2,3").unwrap();
        file.flush().unwrap();

        let result = load_csv(file.path(), Some("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_file_auto_detect_csv() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a,b,target").unwrap();
        writeln!(file, "1,2,0").unwrap();
        file.flush().unwrap();

        let (dataset, _info) = load_file(file.path(), Some("target")).unwrap();
        assert_eq!(dataset.n_samples(), 1);
    }

    #[test]
    fn test_load_file_unsupported_extension() {
        let result = load_file("data.xlsx", Some("target"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_parquet_simple() {
        use polars::prelude::*;

        // Create a temporary Parquet file
        let file = NamedTempFile::with_suffix(".parquet").unwrap();
        let path = file.path().to_path_buf();

        // Create a DataFrame and write to Parquet
        let df = df! {
            "a" => &[1.0, 2.0, 3.0],
            "b" => &[4.0, 5.0, 6.0],
            "target" => &[0.0, 1.0, 0.0],
        }
        .unwrap();

        let mut output_file = std::fs::File::create(&path).unwrap();
        ParquetWriter::new(&mut output_file)
            .finish(&mut df.clone())
            .unwrap();
        drop(output_file);

        let (dataset, info) = load_parquet(&path, Some("target")).unwrap();

        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(info.feature_names.len(), 2);
    }

    #[test]
    fn test_load_csv_skip_rows() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "# This is a comment").unwrap();
        writeln!(file, "a,b,target").unwrap();
        writeln!(file, "1,2,0").unwrap();
        writeln!(file, "3,4,1").unwrap();
        file.flush().unwrap();

        let opts = CsvOptions::new().with_skip_rows(1);
        let (dataset, _info) =
            load_csv_with_options(file.path(), Some("target"), opts).unwrap();

        assert_eq!(dataset.n_samples(), 2);
    }

    #[test]
    fn test_load_csv_n_rows() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a,b,target").unwrap();
        writeln!(file, "1,2,0").unwrap();
        writeln!(file, "3,4,1").unwrap();
        writeln!(file, "5,6,0").unwrap();
        writeln!(file, "7,8,1").unwrap();
        file.flush().unwrap();

        let opts = CsvOptions::new().with_n_rows(2);
        let (dataset, _info) =
            load_csv_with_options(file.path(), Some("target"), opts).unwrap();

        assert_eq!(dataset.n_samples(), 2);
    }

    #[test]
    fn test_infer_task_classification() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a,b,target").unwrap();
        for i in 0..100 {
            writeln!(file, "{},{},{}", i, i * 2, i % 3).unwrap();
        }
        file.flush().unwrap();

        let (_dataset, info) = load_csv(file.path(), Some("target")).unwrap();

        // Should infer classification (few integer values)
        assert_eq!(info.task, Task::Classification);
    }

    #[test]
    fn test_infer_task_regression() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "a,b,target").unwrap();
        for i in 0..100 {
            writeln!(file, "{},{},{:.3}", i, i * 2, (i as f64) * 0.123).unwrap();
        }
        file.flush().unwrap();

        let (_dataset, info) = load_csv(file.path(), Some("target")).unwrap();

        // Should infer regression (continuous values)
        assert_eq!(info.task, Task::Regression);
    }

    #[test]
    fn test_load_csv_mixed_types() {
        // CSV with mixed numeric and string columns
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "name,value,category,target").unwrap();
        writeln!(file, "alice,1.5,A,0").unwrap();
        writeln!(file, "bob,2.5,B,1").unwrap();
        file.flush().unwrap();

        let (dataset, info) = load_csv(file.path(), Some("target")).unwrap();

        // Should only include numeric columns as features
        assert_eq!(dataset.n_samples(), 2);
        assert_eq!(dataset.n_features(), 1); // Only 'value' column
        assert_eq!(info.feature_names, vec!["value"]);
    }

    #[test]
    fn test_csv_encoding_default() {
        let opts = CsvOptions::new();
        assert!(matches!(opts.encoding, CsvEncoding::Utf8));
    }
}
