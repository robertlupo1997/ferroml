//! Python bindings for FerroML datasets module.
//!
//! This module provides:
//! - HuggingFace Hub dataset loading via Python's `datasets` library
//! - Python wrappers for built-in toy datasets (iris, wine, diabetes, linnerud)
//! - Synthetic data generators (make_classification, make_regression, etc.)
//! - Dataset and DatasetInfo wrapper types for ML workflows
//!
//! ## HuggingFace Hub Integration
//!
//! Load datasets directly from HuggingFace Hub:
//!
//! ```python
//! from ferroml.datasets import load_huggingface
//!
//! # Load a classification dataset
//! dataset, info = load_huggingface(
//!     "scikit-learn/iris",
//!     target_column="species"
//! )
//!
//! # Access the data
//! X, y = dataset.x, dataset.y
//! ```
//!
//! ## Built-in Datasets
//!
//! ```python
//! from ferroml.datasets import load_iris, load_wine
//!
//! dataset, info = load_iris()
//! print(f"Samples: {dataset.n_samples}, Features: {dataset.n_features}")
//! ```

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use ferroml_core::datasets::{self, Dataset as RustDataset, DatasetInfo as RustDatasetInfo};
use ferroml_core::Task;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Error type for dataset loading operations.
#[derive(Debug)]
pub enum DatasetError {
    /// HuggingFace datasets library not installed
    DatasetsNotInstalled,
    /// Dataset not found on Hub
    DatasetNotFound(String),
    /// Column not found in dataset
    ColumnNotFound(String),
    /// Unsupported data type
    UnsupportedDataType { column: String, dtype: String },
    /// Empty dataset
    EmptyDataset,
    /// Python error
    PythonError(String),
    /// Conversion error
    ConversionError(String),
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DatasetsNotInstalled => write!(
                f,
                "HuggingFace datasets library not installed. Install with: pip install datasets"
            ),
            Self::DatasetNotFound(name) => {
                write!(f, "Dataset '{}' not found on HuggingFace Hub", name)
            }
            Self::ColumnNotFound(col) => write!(f, "Column '{}' not found in dataset", col),
            Self::UnsupportedDataType { column, dtype } => write!(
                f,
                "Column '{}' has unsupported type '{}'. Expected numeric type.",
                column, dtype
            ),
            Self::EmptyDataset => write!(f, "Dataset is empty"),
            Self::PythonError(e) => write!(f, "Python error: {}", e),
            Self::ConversionError(e) => write!(f, "Conversion error: {}", e),
        }
    }
}

impl std::error::Error for DatasetError {}

impl From<DatasetError> for PyErr {
    fn from(err: DatasetError) -> Self {
        match err {
            DatasetError::DatasetsNotInstalled => {
                PyErr::new::<pyo3::exceptions::PyModuleNotFoundError, _>(err.to_string())
            }
            DatasetError::DatasetNotFound(_) => {
                PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(err.to_string())
            }
            _ => PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()),
        }
    }
}

impl From<PyErr> for DatasetError {
    fn from(err: PyErr) -> Self {
        DatasetError::PythonError(err.to_string())
    }
}

// =============================================================================
// Dataset Python Wrapper
// =============================================================================

/// A dataset containing features and targets for supervised learning.
///
/// This class holds the feature matrix X and target vector y, along with
/// optional metadata like feature names and target names.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Feature matrix of shape (n_samples, n_features)
/// y : numpy.ndarray
///     Target vector of shape (n_samples,)
///
/// Attributes
/// ----------
/// n_samples : int
///     Number of samples in the dataset
/// n_features : int
///     Number of features in the dataset
/// feature_names : list[str] or None
///     Names of the features (if available)
/// target_names : list[str] or None
///     Names of the target classes (for classification)
///
/// Examples
/// --------
/// >>> from ferroml.datasets import Dataset
/// >>> import numpy as np
/// >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
/// >>> y = np.array([0.0, 1.0, 0.0])
/// >>> dataset = Dataset(X, y)
/// >>> dataset.n_samples
/// 3
/// >>> dataset.n_features
/// 2
#[pyclass(name = "Dataset", module = "ferroml.datasets")]
pub struct PyDataset {
    inner: RustDataset,
}

#[pymethods]
impl PyDataset {
    /// Create a new Dataset from feature matrix and target vector.
    #[new]
    #[pyo3(signature = (x, y))]
    fn new(x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<Self> {
        let x_array = to_owned_array_2d(x);
        let y_array = to_owned_array_1d(y);

        let dataset = RustDataset::try_new(x_array, y_array)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self { inner: dataset })
    }

    /// Get the feature matrix as a numpy array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Feature matrix of shape (n_samples, n_features)
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.x().clone().into_pyarray(py)
    }

    /// Get the target vector as a numpy array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Target vector of shape (n_samples,)
    #[getter]
    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.y().clone().into_pyarray(py)
    }

    /// Get the number of samples.
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples()
    }

    /// Get the number of features.
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    /// Get the shape as (n_samples, n_features).
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Get feature names (if available).
    #[getter]
    fn feature_names(&self) -> Option<Vec<String>> {
        self.inner.feature_names().map(|v| v.to_vec())
    }

    /// Get target names (if available).
    #[getter]
    fn target_names(&self) -> Option<Vec<String>> {
        self.inner.target_names().map(|v| v.to_vec())
    }

    /// Set feature names.
    fn set_feature_names(&mut self, names: Vec<String>) -> PyResult<()> {
        // We need to reconstruct the dataset with new metadata
        let (x, y) = (self.inner.x().clone(), self.inner.y().clone());
        let mut new_dataset = RustDataset::new(x, y).with_feature_names(names);
        if let Some(tn) = self.inner.target_names() {
            new_dataset = new_dataset.with_target_names(tn.to_vec());
        }
        self.inner = new_dataset;
        Ok(())
    }

    /// Set target names.
    fn set_target_names(&mut self, names: Vec<String>) -> PyResult<()> {
        let (x, y) = (self.inner.x().clone(), self.inner.y().clone());
        let mut new_dataset = RustDataset::new(x, y);
        if let Some(fn_) = self.inner.feature_names() {
            new_dataset = new_dataset.with_feature_names(fn_.to_vec());
        }
        new_dataset = new_dataset.with_target_names(names);
        self.inner = new_dataset;
        Ok(())
    }

    /// Get unique class values in the target.
    fn unique_classes(&self) -> Vec<f64> {
        self.inner.unique_classes()
    }

    /// Get class counts as a dictionary.
    fn class_counts(&self) -> std::collections::HashMap<i64, usize> {
        self.inner.class_counts()
    }

    /// Check if this is a binary classification problem.
    fn is_binary(&self) -> bool {
        self.inner.is_binary()
    }

    /// Check if this is a multiclass classification problem.
    fn is_multiclass(&self) -> bool {
        self.inner.is_multiclass()
    }

    /// Infer the task type (classification or regression).
    ///
    /// Returns
    /// -------
    /// str
    ///     Either "classification", "regression", "timeseries", or "survival"
    fn infer_task(&self) -> String {
        match self.inner.infer_task() {
            Task::Classification => "classification".to_string(),
            Task::Regression => "regression".to_string(),
            Task::TimeSeries => "timeseries".to_string(),
            Task::Survival => "survival".to_string(),
        }
    }

    /// Get basic statistics about the dataset.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary containing n_samples, n_features, and feature_stats
    fn describe(&self, py: Python<'_>) -> PyResult<PyObject> {
        let stats = self.inner.describe();
        let dict = PyDict::new(py);
        dict.set_item("n_samples", stats.n_samples)?;
        dict.set_item("n_features", stats.n_features)?;

        // Convert feature stats to list of dicts
        let feature_stats_list = PyList::empty(py);
        for fs in &stats.feature_stats {
            let fs_dict = PyDict::new(py);
            if let Some(ref name) = fs.name {
                fs_dict.set_item("name", name)?;
            } else {
                fs_dict.set_item("name", py.None())?;
            }
            fs_dict.set_item("mean", fs.mean)?;
            fs_dict.set_item("std", fs.std)?;
            fs_dict.set_item("min", fs.min)?;
            fs_dict.set_item("max", fs.max)?;
            fs_dict.set_item("median", fs.median)?;
            fs_dict.set_item("n_missing", fs.n_missing)?;
            feature_stats_list.append(fs_dict)?;
        }
        dict.set_item("feature_stats", feature_stats_list)?;

        Ok(dict.into())
    }

    /// Split into train and test sets.
    ///
    /// Parameters
    /// ----------
    /// test_size : float
    ///     Fraction of data to use for testing (0.0 to 1.0)
    /// shuffle : bool, optional (default=True)
    ///     Whether to shuffle before splitting
    /// random_state : int or None, optional
    ///     Random seed for reproducibility
    ///
    /// Returns
    /// -------
    /// tuple[Dataset, Dataset]
    ///     (train_dataset, test_dataset)
    #[pyo3(signature = (test_size, shuffle=true, random_state=None))]
    fn train_test_split(
        &self,
        test_size: f64,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> PyResult<(PyDataset, PyDataset)> {
        let (train, test) = self
            .inner
            .train_test_split(test_size, shuffle, random_state)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok((PyDataset { inner: train }, PyDataset { inner: test }))
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(n_samples={}, n_features={}, task={})",
            self.n_samples(),
            self.n_features(),
            self.infer_task()
        )
    }
}

// =============================================================================
// DatasetInfo Python Wrapper
// =============================================================================

/// Metadata about a dataset.
///
/// Contains descriptive information including name, description, task type,
/// feature names, and source attribution.
///
/// Attributes
/// ----------
/// name : str
///     Name of the dataset
/// description : str
///     Description of the dataset
/// task : str
///     Task type ("classification" or "regression")
/// n_samples : int
///     Number of samples
/// n_features : int
///     Number of features
/// n_classes : int or None
///     Number of classes (for classification)
/// feature_names : list[str]
///     Names of the features
/// target_names : list[str] or None
///     Names of the target classes
/// source : str or None
///     Original source/citation
/// url : str or None
///     URL for more information
/// license : str or None
///     License information
#[pyclass(name = "DatasetInfo", module = "ferroml.datasets")]
pub struct PyDatasetInfo {
    inner: RustDatasetInfo,
}

#[pymethods]
impl PyDatasetInfo {
    /// Create new DatasetInfo with required fields.
    #[new]
    #[pyo3(signature = (name, task, n_samples, n_features))]
    fn new(name: String, task: String, n_samples: usize, n_features: usize) -> PyResult<Self> {
        let task_enum = match task.to_lowercase().as_str() {
            "classification" => Task::Classification,
            "regression" => Task::Regression,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "task must be 'classification' or 'regression'",
                ))
            }
        };

        Ok(Self {
            inner: RustDatasetInfo::new(name, task_enum, n_samples, n_features),
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    #[getter]
    fn task(&self) -> String {
        match self.inner.task {
            Task::Classification => "classification".to_string(),
            Task::Regression => "regression".to_string(),
            Task::TimeSeries => "timeseries".to_string(),
            Task::Survival => "survival".to_string(),
        }
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.n_samples
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features
    }

    #[getter]
    fn n_classes(&self) -> Option<usize> {
        self.inner.n_classes
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names.clone()
    }

    #[getter]
    fn target_names(&self) -> Option<Vec<String>> {
        self.inner.target_names.clone()
    }

    #[getter]
    fn source(&self) -> Option<String> {
        self.inner.source.clone()
    }

    #[getter]
    fn url(&self) -> Option<String> {
        self.inner.url.clone()
    }

    #[getter]
    fn license(&self) -> Option<String> {
        self.inner.license.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetInfo(name='{}', task='{}', samples={}, features={})",
            self.name(),
            self.task(),
            self.n_samples(),
            self.n_features()
        )
    }
}

// =============================================================================
// HuggingFace Hub Integration
// =============================================================================

/// Load a dataset from HuggingFace Hub.
///
/// This function downloads and loads datasets from HuggingFace Hub using
/// the `datasets` Python library. The data is converted to FerroML's
/// Dataset format for use with ML models.
///
/// Parameters
/// ----------
/// dataset_name : str
///     The HuggingFace Hub dataset identifier (e.g., "scikit-learn/iris")
/// target_column : str
///     Name of the column to use as the target variable
/// split : str, optional (default="train")
///     Which split to load (e.g., "train", "test", "validation")
/// feature_columns : list[str] or None, optional
///     Specific feature columns to include. If None, uses all numeric columns.
/// config_name : str or None, optional
///     Configuration name for datasets with multiple configs
/// cache_dir : str or None, optional
///     Directory to cache downloaded datasets
/// trust_remote_code : bool, optional (default=False)
///     Whether to trust and execute remote code from the dataset
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The loaded dataset and its metadata
///
/// Raises
/// ------
/// ModuleNotFoundError
///     If the `datasets` library is not installed
/// FileNotFoundError
///     If the dataset is not found on HuggingFace Hub
/// ValueError
///     If the target column is not found or data conversion fails
///
/// Examples
/// --------
/// >>> from ferroml.datasets import load_huggingface
/// >>> dataset, info = load_huggingface(
/// ...     "scikit-learn/iris",
/// ...     target_column="species"
/// ... )
/// >>> print(f"Samples: {dataset.n_samples}, Features: {dataset.n_features}")
/// Samples: 150, Features: 4
#[pyfunction]
#[pyo3(signature = (
    dataset_name,
    target_column,
    split="train",
    feature_columns=None,
    config_name=None,
    cache_dir=None,
    trust_remote_code=false
))]
fn load_huggingface(
    py: Python<'_>,
    dataset_name: &str,
    target_column: &str,
    split: &str,
    feature_columns: Option<Vec<String>>,
    config_name: Option<&str>,
    cache_dir: Option<&str>,
    trust_remote_code: bool,
) -> PyResult<(PyDataset, PyDatasetInfo)> {
    // Import the datasets library
    let datasets_module = py.import("datasets").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyModuleNotFoundError, _>(
            "HuggingFace datasets library not installed. Install with: pip install datasets",
        )
    })?;

    // Build the load_dataset arguments
    let kwargs = PyDict::new(py);
    kwargs.set_item("split", split)?;
    if let Some(config) = config_name {
        kwargs.set_item("name", config)?;
    }
    if let Some(cache) = cache_dir {
        kwargs.set_item("cache_dir", cache)?;
    }
    kwargs.set_item("trust_remote_code", trust_remote_code)?;

    // Load the dataset from Hub
    let load_dataset = datasets_module.getattr("load_dataset")?;
    let hf_dataset = load_dataset
        .call((dataset_name,), Some(&kwargs))
        .map_err(|e| {
            // Check if it's a dataset not found error
            let err_str = e.to_string();
            if err_str.contains("DatasetNotFoundError") || err_str.contains("FileNotFoundError") {
                DatasetError::DatasetNotFound(dataset_name.to_string()).into()
            } else {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to load dataset '{}': {}",
                    dataset_name, e
                ))
            }
        })?;

    // Get the number of rows
    let n_samples: usize = hf_dataset
        .call_method0("num_rows")?
        .extract()
        .unwrap_or_else(|_| {
            // Try alternative: len(dataset)
            hf_dataset.len().unwrap_or(0)
        });

    if n_samples == 0 {
        return Err(DatasetError::EmptyDataset.into());
    }

    // Get column names
    let column_names: Vec<String> = hf_dataset.getattr("column_names")?.extract()?;

    // Verify target column exists
    if !column_names.contains(&target_column.to_string()) {
        return Err(DatasetError::ColumnNotFound(target_column.to_string()).into());
    }

    // Determine feature columns
    let feature_cols: Vec<String> = if let Some(cols) = feature_columns {
        // Validate specified columns exist
        for col in &cols {
            if !column_names.contains(col) {
                return Err(DatasetError::ColumnNotFound(col.clone()).into());
            }
        }
        cols
    } else {
        // Auto-detect: use all columns except target
        column_names
            .iter()
            .filter(|c| *c != target_column)
            .cloned()
            .collect()
    };

    let n_features = feature_cols.len();
    if n_features == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No feature columns found in dataset",
        ));
    }

    // Extract target column
    let target_data = hf_dataset.get_item(target_column)?;
    let y = extract_column_to_f64(py, &target_data, target_column)?;

    // Extract feature columns
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    for col_name in &feature_cols {
        let col_data = hf_dataset.get_item(col_name.as_str())?;
        let col_array = extract_column_to_f64(py, &col_data, col_name)?;
        x_data.extend(col_array.iter());
    }

    // Reshape from column-major to row-major
    let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| x_data[j * n_samples + i]);

    // Create Dataset
    let dataset = RustDataset::new(x, y).with_feature_names(feature_cols.clone());

    // Try to infer if classification (target has few unique integer values)
    let task = dataset.infer_task();
    let n_classes = if task == Task::Classification {
        Some(dataset.unique_classes().len())
    } else {
        None
    };

    // Create DatasetInfo
    let mut info = RustDatasetInfo::new(dataset_name, task, n_samples, n_features)
        .with_description(format!(
            "Dataset loaded from HuggingFace Hub: {}",
            dataset_name
        ))
        .with_feature_names(feature_cols)
        .with_source(format!("HuggingFace Hub: {}", dataset_name))
        .with_url(format!("https://huggingface.co/datasets/{}", dataset_name));

    if let Some(n) = n_classes {
        info = info.with_n_classes(n);
        // Try to get target names from unique values
        let unique_classes = dataset.unique_classes();
        let target_names: Vec<String> = unique_classes.iter().map(|v| format!("{}", v)).collect();
        info = info.with_target_names(target_names);
    }

    let py_dataset = PyDataset { inner: dataset };
    let py_info = PyDatasetInfo { inner: info };

    Ok((py_dataset, py_info))
}

/// Extract a HuggingFace dataset column to Array1<f64>.
fn extract_column_to_f64(
    _py: Python<'_>,
    column: &Bound<'_, PyAny>,
    col_name: &str,
) -> PyResult<Array1<f64>> {
    // Try to convert to numpy first
    let numpy_result = column.call_method0("to_numpy");

    if let Ok(np_array) = numpy_result {
        // Try to extract as f64
        if let Ok(arr) = np_array.extract::<PyReadonlyArray1<f64>>() {
            return Ok(arr.to_owned_array());
        }
        // Try i64 and convert
        if let Ok(arr) = np_array.extract::<PyReadonlyArray1<i64>>() {
            let owned = arr.to_owned_array();
            return Ok(owned.mapv(|v| v as f64));
        }
        // Try i32 and convert
        if let Ok(arr) = np_array.extract::<PyReadonlyArray1<i32>>() {
            let owned = arr.to_owned_array();
            return Ok(owned.mapv(|v| v as f64));
        }
    }

    // Fallback: iterate through Python list/sequence
    let len = column.len()?;
    let mut values = Vec::with_capacity(len);

    for i in 0..len {
        let item = column.get_item(i)?;
        // Try to extract as float
        if let Ok(v) = item.extract::<f64>() {
            values.push(v);
        } else if let Ok(v) = item.extract::<i64>() {
            values.push(v as f64);
        } else if let Ok(v) = item.extract::<i32>() {
            values.push(v as f64);
        } else if let Ok(v) = item.extract::<bool>() {
            values.push(if v { 1.0 } else { 0.0 });
        } else {
            // Try string-to-int encoding for categorical
            // Get unique values and encode
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Column '{}' contains non-numeric values that cannot be converted. \
                 Consider using LabelEncoder or specifying feature_columns to exclude this column.",
                col_name
            )));
        }
    }

    Ok(Array1::from(values))
}

// =============================================================================
// Built-in Toy Datasets
// =============================================================================

/// Load the classic Iris flower dataset.
///
/// The iris dataset contains 150 samples of 3 species of iris flowers,
/// with 4 features each (sepal/petal length and width).
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The dataset and its metadata
///
/// Examples
/// --------
/// >>> from ferroml.datasets import load_iris
/// >>> dataset, info = load_iris()
/// >>> print(info.description)
#[pyfunction]
fn load_iris() -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) = datasets::load_iris();
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

/// Load the Wine recognition dataset.
///
/// The wine dataset contains 178 samples of wines from 3 cultivars,
/// with 13 chemical analysis features.
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The dataset and its metadata
#[pyfunction]
fn load_wine() -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) = datasets::load_wine();
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

/// Load the Diabetes regression dataset.
///
/// Contains 442 samples with 10 baseline variables (age, sex, bmi, blood pressure,
/// and 6 blood serum measurements) and a target measuring disease progression.
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The dataset and its metadata
#[pyfunction]
fn load_diabetes() -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) = datasets::load_diabetes();
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

/// Load the Linnerud multi-output regression dataset.
///
/// Contains 20 samples with 3 exercise features and 3 physiological measurements.
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The dataset and its metadata
#[pyfunction]
fn load_linnerud() -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) = datasets::load_linnerud();
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

// =============================================================================
// Synthetic Data Generators
// =============================================================================

/// Generate a synthetic classification dataset.
///
/// Creates a random n-class classification problem with configurable
/// numbers of informative features.
///
/// Parameters
/// ----------
/// n_samples : int, optional (default=100)
///     Number of samples to generate
/// n_features : int, optional (default=10)
///     Total number of features
/// n_informative : int, optional (default=5)
///     Number of informative features
/// n_classes : int, optional (default=2)
///     Number of classes
/// random_state : int or None, optional
///     Random seed for reproducibility
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The generated dataset and its metadata
///
/// Examples
/// --------
/// >>> from ferroml.datasets import make_classification
/// >>> dataset, info = make_classification(n_samples=1000, n_classes=3, random_state=42)
#[pyfunction]
#[pyo3(signature = (n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=None))]
fn make_classification(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    n_classes: usize,
    random_state: Option<u64>,
) -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) = datasets::make_classification(
        n_samples,
        n_features,
        n_informative,
        n_classes,
        random_state,
    );
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

/// Generate a synthetic regression dataset.
///
/// Creates a random regression problem with linear relationships
/// and configurable noise.
///
/// Parameters
/// ----------
/// n_samples : int, optional (default=100)
///     Number of samples to generate
/// n_features : int, optional (default=10)
///     Total number of features
/// n_informative : int, optional (default=5)
///     Number of informative features
/// noise : float, optional (default=0.1)
///     Standard deviation of Gaussian noise
/// random_state : int or None, optional
///     Random seed for reproducibility
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The generated dataset and its metadata
#[pyfunction]
#[pyo3(signature = (n_samples=100, n_features=10, n_informative=5, noise=0.1, random_state=None))]
fn make_regression(
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    random_state: Option<u64>,
) -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) =
        datasets::make_regression(n_samples, n_features, n_informative, noise, random_state);
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

/// Generate synthetic Gaussian blobs for clustering.
///
/// Creates isotropic Gaussian blobs centered at random positions.
///
/// Parameters
/// ----------
/// n_samples : int, optional (default=100)
///     Number of samples to generate
/// n_features : int, optional (default=2)
///     Number of features
/// centers : int, optional (default=3)
///     Number of cluster centers
/// cluster_std : float, optional (default=1.0)
///     Standard deviation of clusters
/// random_state : int or None, optional
///     Random seed for reproducibility
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The generated dataset and its metadata
#[pyfunction]
#[pyo3(signature = (n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=None))]
fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    random_state: Option<u64>,
) -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) =
        datasets::make_blobs(n_samples, n_features, centers, cluster_std, random_state);
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

/// Generate two interleaving half circles (moons).
///
/// A simple toy dataset for binary classification that is not linearly separable.
///
/// Parameters
/// ----------
/// n_samples : int, optional (default=100)
///     Number of samples to generate
/// noise : float, optional (default=0.1)
///     Standard deviation of Gaussian noise
/// random_state : int or None, optional
///     Random seed for reproducibility
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The generated dataset and its metadata
#[pyfunction]
#[pyo3(signature = (n_samples=100, noise=0.1, random_state=None))]
fn make_moons(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) = datasets::make_moons(n_samples, noise, random_state);
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

/// Generate a large circle containing a smaller circle.
///
/// A simple toy dataset for binary classification that is not linearly separable.
///
/// Parameters
/// ----------
/// n_samples : int, optional (default=100)
///     Number of samples to generate
/// noise : float, optional (default=0.1)
///     Standard deviation of Gaussian noise
/// factor : float, optional (default=0.5)
///     Scale factor between inner and outer circle (0 to 1)
/// random_state : int or None, optional
///     Random seed for reproducibility
///
/// Returns
/// -------
/// tuple[Dataset, DatasetInfo]
///     The generated dataset and its metadata
#[pyfunction]
#[pyo3(signature = (n_samples=100, noise=0.1, factor=0.5, random_state=None))]
fn make_circles(
    n_samples: usize,
    noise: f64,
    factor: f64,
    random_state: Option<u64>,
) -> (PyDataset, PyDatasetInfo) {
    let (dataset, info) = datasets::make_circles(n_samples, noise, factor, random_state);
    (PyDataset { inner: dataset }, PyDatasetInfo { inner: info })
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register the datasets submodule.
pub fn register_datasets_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let datasets_module = PyModule::new(parent.py(), "datasets")?;

    // Add Dataset and DatasetInfo classes
    datasets_module.add_class::<PyDataset>()?;
    datasets_module.add_class::<PyDatasetInfo>()?;

    // Add HuggingFace Hub loader
    datasets_module.add_function(wrap_pyfunction!(load_huggingface, &datasets_module)?)?;

    // Add built-in toy datasets
    datasets_module.add_function(wrap_pyfunction!(load_iris, &datasets_module)?)?;
    datasets_module.add_function(wrap_pyfunction!(load_wine, &datasets_module)?)?;
    datasets_module.add_function(wrap_pyfunction!(load_diabetes, &datasets_module)?)?;
    datasets_module.add_function(wrap_pyfunction!(load_linnerud, &datasets_module)?)?;

    // Add synthetic data generators
    datasets_module.add_function(wrap_pyfunction!(make_classification, &datasets_module)?)?;
    datasets_module.add_function(wrap_pyfunction!(make_regression, &datasets_module)?)?;
    datasets_module.add_function(wrap_pyfunction!(make_blobs, &datasets_module)?)?;
    datasets_module.add_function(wrap_pyfunction!(make_moons, &datasets_module)?)?;
    datasets_module.add_function(wrap_pyfunction!(make_circles, &datasets_module)?)?;

    // Register the submodule
    parent.add_submodule(&datasets_module)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    // Integration tests with Python runtime are typically done via maturin develop.
    // These would test the bindings end-to-end.
}
