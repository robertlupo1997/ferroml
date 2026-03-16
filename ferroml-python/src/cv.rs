//! Python bindings for FerroML Cross-Validation
//!
//! This module exposes cross-validation splitters and utilities:
//!
//! - Splitters: KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut
//! - Utility: cross_val_score (fit/predict loop with scoring)

use ferroml_core::cv::{
    CrossValidator, GroupKFold, KFold, LeaveOneOut, LeavePOut, RepeatedKFold, ShuffleSplit,
    StratifiedKFold, TimeSeriesSplit,
};
use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

// =============================================================================
// KFold
// =============================================================================

/// K-Fold cross-validation splitter.
///
/// Parameters
/// ----------
/// n_folds : int, optional (default=5)
///     Number of folds.
/// shuffle : bool, optional (default=False)
///     Whether to shuffle before splitting.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
#[pyclass(name = "KFold")]
#[derive(Clone)]
struct PyKFold {
    n_folds: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

#[pymethods]
impl PyKFold {
    #[new]
    #[pyo3(signature = (n_folds=5, shuffle=false, random_state=None))]
    fn new(n_folds: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        Self {
            n_folds,
            shuffle,
            random_state,
        }
    }

    /// Generate train/test index splits.
    ///
    /// Parameters
    /// ----------
    /// X : numpy.ndarray
    ///     Feature matrix (used only for n_samples).
    /// y : numpy.ndarray or None, optional
    ///     Target array (unused for KFold, accepted for API compatibility).
    ///
    /// Returns
    /// -------
    /// list of (numpy.ndarray, numpy.ndarray)
    ///     List of (train_indices, test_indices) tuples.
    #[pyo3(signature = (x, y=None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        let n_samples = x.shape()[0];
        let mut cv = KFold::new(self.n_folds).with_shuffle(self.shuffle);
        if let Some(seed) = self.random_state {
            cv = cv.with_seed(seed);
        }

        let folds = cv
            .split(n_samples, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let _ = y; // consumed for API compatibility
        folds_to_py_list(py, &folds)
    }

    /// Number of folds.
    #[getter]
    fn get_n_splits(&self) -> usize {
        self.n_folds
    }

    fn __repr__(&self) -> String {
        format!(
            "KFold(n_folds={}, shuffle={}, random_state={:?})",
            self.n_folds, self.shuffle, self.random_state
        )
    }
}

// =============================================================================
// StratifiedKFold
// =============================================================================

/// Stratified K-Fold cross-validation splitter.
///
/// Preserves the percentage of samples for each class across folds.
///
/// Parameters
/// ----------
/// n_folds : int, optional (default=5)
///     Number of folds.
/// shuffle : bool, optional (default=False)
///     Whether to shuffle before splitting.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
#[pyclass(name = "StratifiedKFold")]
#[derive(Clone)]
struct PyStratifiedKFold {
    n_folds: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

#[pymethods]
impl PyStratifiedKFold {
    #[new]
    #[pyo3(signature = (n_folds=5, shuffle=false, random_state=None))]
    fn new(n_folds: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        Self {
            n_folds,
            shuffle,
            random_state,
        }
    }

    /// Generate stratified train/test index splits.
    ///
    /// Parameters
    /// ----------
    /// X : numpy.ndarray
    ///     Feature matrix (used only for n_samples).
    /// y : numpy.ndarray
    ///     Target array (required for stratification).
    ///
    /// Returns
    /// -------
    /// list of (numpy.ndarray, numpy.ndarray)
    ///     List of (train_indices, test_indices) tuples.
    #[pyo3(signature = (x, y=None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        let n_samples = x.shape()[0];
        let mut cv = StratifiedKFold::new(self.n_folds).with_shuffle(self.shuffle);
        if let Some(seed) = self.random_state {
            cv = cv.with_seed(seed);
        }

        let y_arr = y.map(|arr| arr.as_array().to_owned()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "StratifiedKFold requires y (target array) for stratification",
            )
        })?;

        let folds = cv
            .split(n_samples, Some(&y_arr), None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        folds_to_py_list(py, &folds)
    }

    /// Number of folds.
    #[getter]
    fn get_n_splits(&self) -> usize {
        self.n_folds
    }

    fn __repr__(&self) -> String {
        format!(
            "StratifiedKFold(n_folds={}, shuffle={}, random_state={:?})",
            self.n_folds, self.shuffle, self.random_state
        )
    }
}

// =============================================================================
// TimeSeriesSplit
// =============================================================================

/// Time Series cross-validation splitter.
///
/// Provides train/test splits that respect temporal ordering:
/// each test set is always after the training set.
///
/// Parameters
/// ----------
/// n_splits : int, optional (default=5)
///     Number of splits.
#[pyclass(name = "TimeSeriesSplit")]
#[derive(Clone)]
struct PyTimeSeriesSplit {
    n_splits: usize,
}

#[pymethods]
impl PyTimeSeriesSplit {
    #[new]
    #[pyo3(signature = (n_splits=5))]
    fn new(n_splits: usize) -> Self {
        Self { n_splits }
    }

    /// Generate time-series train/test index splits.
    ///
    /// Parameters
    /// ----------
    /// X : numpy.ndarray
    ///     Feature matrix (used only for n_samples).
    /// y : numpy.ndarray or None, optional
    ///     Target array (unused, accepted for API compatibility).
    ///
    /// Returns
    /// -------
    /// list of (numpy.ndarray, numpy.ndarray)
    ///     List of (train_indices, test_indices) tuples.
    #[pyo3(signature = (x, y=None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        let n_samples = x.shape()[0];
        let cv = TimeSeriesSplit::new(self.n_splits);

        let _ = y; // consumed for API compatibility

        let folds = cv
            .split(n_samples, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        folds_to_py_list(py, &folds)
    }

    /// Number of splits.
    #[getter]
    fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    fn __repr__(&self) -> String {
        format!("TimeSeriesSplit(n_splits={})", self.n_splits)
    }
}

// =============================================================================
// LeaveOneOut
// =============================================================================

/// Leave-One-Out cross-validation splitter.
///
/// Each sample is used once as the test set while the remaining
/// samples form the training set.
#[pyclass(name = "LeaveOneOut")]
#[derive(Clone)]
struct PyLeaveOneOut;

#[pymethods]
impl PyLeaveOneOut {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Generate LOO train/test index splits.
    ///
    /// Parameters
    /// ----------
    /// X : numpy.ndarray
    ///     Feature matrix (used only for n_samples).
    /// y : numpy.ndarray or None, optional
    ///     Target array (unused, accepted for API compatibility).
    ///
    /// Returns
    /// -------
    /// list of (numpy.ndarray, numpy.ndarray)
    ///     List of (train_indices, test_indices) tuples.
    #[pyo3(signature = (x, y=None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        let n_samples = x.shape()[0];
        let cv = LeaveOneOut::new();

        let _ = y; // consumed for API compatibility

        let folds = cv
            .split(n_samples, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        folds_to_py_list(py, &folds)
    }

    /// Number of splits (equal to n_samples, determined at split time).
    fn get_n_splits(&self, x: PyReadonlyArray2<f64>) -> usize {
        x.shape()[0]
    }

    fn __repr__(&self) -> String {
        "LeaveOneOut()".to_string()
    }
}

// =============================================================================
// RepeatedKFold
// =============================================================================

/// Repeated K-Fold cross-validation splitter.
///
/// Repeats K-Fold n_repeats times with different randomization in each repetition.
///
/// Parameters
/// ----------
/// n_folds : int, optional (default=5)
///     Number of folds.
/// n_repeats : int, optional (default=10)
///     Number of times to repeat the cross-validation.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
#[pyclass(name = "RepeatedKFold")]
#[derive(Clone)]
struct PyRepeatedKFold {
    n_folds: usize,
    n_repeats: usize,
    random_state: Option<u64>,
}

#[pymethods]
impl PyRepeatedKFold {
    #[new]
    #[pyo3(signature = (n_folds=5, n_repeats=10, random_state=None))]
    fn new(n_folds: usize, n_repeats: usize, random_state: Option<u64>) -> Self {
        Self {
            n_folds,
            n_repeats,
            random_state,
        }
    }

    #[pyo3(signature = (x, y=None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        let n_samples = x.shape()[0];
        let mut cv = RepeatedKFold::new(self.n_folds, self.n_repeats);
        if let Some(seed) = self.random_state {
            cv = cv.with_seed(seed);
        }
        let _ = y;
        let folds = cv
            .split(n_samples, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        folds_to_py_list(py, &folds)
    }

    #[getter]
    fn get_n_splits(&self) -> usize {
        self.n_folds * self.n_repeats
    }

    fn __repr__(&self) -> String {
        format!(
            "RepeatedKFold(n_folds={}, n_repeats={}, random_state={:?})",
            self.n_folds, self.n_repeats, self.random_state
        )
    }
}

// =============================================================================
// ShuffleSplit
// =============================================================================

/// Random permutation cross-validation splitter.
///
/// Yields random train/test splits of the data. Unlike K-Fold, test sets
/// can overlap between iterations.
///
/// Parameters
/// ----------
/// n_splits : int, optional (default=10)
///     Number of re-shuffling & splitting iterations.
/// test_size : float, optional (default=0.1)
///     Fraction of the dataset to include in the test split.
/// train_size : float or None, optional (default=None)
///     Fraction for training. Default is complement of test_size.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
#[pyclass(name = "ShuffleSplit")]
#[derive(Clone)]
struct PyShuffleSplit {
    n_splits: usize,
    test_size: f64,
    train_size: Option<f64>,
    random_state: Option<u64>,
}

#[pymethods]
impl PyShuffleSplit {
    #[new]
    #[pyo3(signature = (n_splits=10, test_size=0.1, train_size=None, random_state=None))]
    fn new(
        n_splits: usize,
        test_size: f64,
        train_size: Option<f64>,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_splits,
            test_size,
            train_size,
            random_state,
        }
    }

    #[pyo3(signature = (x, y=None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        let n_samples = x.shape()[0];
        let mut cv = ShuffleSplit::new(self.n_splits).with_test_size(self.test_size);
        if let Some(train) = self.train_size {
            cv = cv.with_train_size(train);
        }
        if let Some(seed) = self.random_state {
            cv = cv.with_seed(seed);
        }
        let _ = y;
        let folds = cv
            .split(n_samples, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        folds_to_py_list(py, &folds)
    }

    #[getter]
    fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    fn __repr__(&self) -> String {
        format!(
            "ShuffleSplit(n_splits={}, test_size={}, random_state={:?})",
            self.n_splits, self.test_size, self.random_state
        )
    }
}

// =============================================================================
// GroupKFold
// =============================================================================

/// K-fold cross-validation with non-overlapping groups.
///
/// Each group appears in exactly one test fold. Groups are assigned to
/// folds using greedy bin-packing.
///
/// Parameters
/// ----------
/// n_folds : int, optional (default=5)
///     Number of folds.
#[pyclass(name = "GroupKFold")]
#[derive(Clone)]
struct PyGroupKFold {
    n_folds: usize,
}

#[pymethods]
impl PyGroupKFold {
    #[new]
    #[pyo3(signature = (n_folds=5))]
    fn new(n_folds: usize) -> Self {
        Self { n_folds }
    }

    /// Generate group-aware train/test index splits.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray
    ///     Feature matrix (used only for n_samples).
    /// y : numpy.ndarray or None, optional
    ///     Target array (unused).
    /// groups : numpy.ndarray
    ///     Group labels for each sample (integer array).
    #[pyo3(signature = (x, y=None, groups=None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray1<f64>>,
        groups: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<PyObject> {
        let n_samples = x.shape()[0];
        let cv = GroupKFold::new(self.n_folds);
        let _ = y;

        let groups_arr = groups
            .map(|g| ndarray::Array1::from_vec(g.as_array().to_vec()))
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("GroupKFold requires groups parameter")
            })?;

        let folds = cv
            .split(n_samples, None, Some(&groups_arr))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        folds_to_py_list(py, &folds)
    }

    #[getter]
    fn get_n_splits(&self) -> usize {
        self.n_folds
    }

    fn __repr__(&self) -> String {
        format!("GroupKFold(n_folds={})", self.n_folds)
    }
}

// =============================================================================
// LeavePOut
// =============================================================================

/// Leave-P-Out cross-validation splitter.
///
/// Each test set contains exactly p samples, chosen from all C(n, p) combinations.
///
/// Parameters
/// ----------
/// p : int, optional (default=2)
///     Size of the test sets.
#[pyclass(name = "LeavePOut")]
#[derive(Clone)]
struct PyLeavePOut {
    p: usize,
}

#[pymethods]
impl PyLeavePOut {
    #[new]
    #[pyo3(signature = (p=2))]
    fn new(p: usize) -> Self {
        Self { p }
    }

    #[pyo3(signature = (x, y=None))]
    fn split<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        let n_samples = x.shape()[0];
        let cv = LeavePOut::new(self.p);
        let _ = y;
        let folds = cv
            .split(n_samples, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        folds_to_py_list(py, &folds)
    }

    #[getter]
    fn get_p(&self) -> usize {
        self.p
    }

    fn __repr__(&self) -> String {
        format!("LeavePOut(p={})", self.p)
    }
}

// =============================================================================
// cross_val_score
// =============================================================================

/// Evaluate a model using cross-validation and return per-fold scores.
///
/// Parameters
/// ----------
/// model : object
///     A FerroML model with `fit(X, y)` and `predict(X)` methods.
/// X : numpy.ndarray
///     Feature matrix of shape (n_samples, n_features).
/// y : numpy.ndarray
///     Target array of shape (n_samples,).
/// cv : int, optional (default=5)
///     Number of cross-validation folds (uses KFold).
/// scoring : str, optional (default="accuracy")
///     Scoring metric: "accuracy", "mse", "mae", or "r2".
///
/// Returns
/// -------
/// numpy.ndarray
///     Array of scores, one per fold.
#[pyfunction]
#[pyo3(signature = (model, x, y, cv=5, scoring="accuracy"))]
fn cross_val_score<'py>(
    py: Python<'py>,
    model: &Bound<'py, PyAny>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    cv: usize,
    scoring: &str,
) -> PyResult<PyObject> {
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let n_samples = x_arr.nrows();

    // Validate scoring metric up front
    match scoring {
        "accuracy" | "mse" | "mae" | "r2" => {}
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown scoring '{}'. Use 'accuracy', 'mse', 'mae', or 'r2'.",
                scoring
            )));
        }
    }

    // Create KFold splitter
    let kfold = KFold::new(cv);
    let folds = kfold
        .split(n_samples, None, None)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut scores = Vec::with_capacity(folds.len());

    for fold in &folds {
        // Build train/test arrays by selecting rows
        let train_x = select_rows(&x_arr, &fold.train_indices);
        let train_y = select_elements(&y_arr, &fold.train_indices);
        let test_x = select_rows(&x_arr, &fold.test_indices);
        let test_y = select_elements(&y_arr, &fold.test_indices);

        // Convert to numpy for Python calls
        let train_x_py = train_x.into_pyarray(py);
        let train_y_py = train_y.into_pyarray(py);
        let test_x_py = test_x.into_pyarray(py);

        // Clone the model for this fold (call Python's clone-like pattern)
        // We use the same model reference — fit mutates in place per sklearn convention
        // To avoid contamination, we create a fresh instance via __class__
        let model_class = model.getattr("__class__")?;
        let fold_model = model_class.call0()?;

        // fit
        fold_model.call_method1("fit", (train_x_py, train_y_py))?;

        // predict
        let predictions_obj = fold_model.call_method1("predict", (test_x_py,))?;

        // Convert predictions back to Rust array
        let predictions: PyReadonlyArray1<f64> = predictions_obj.extract()?;
        let pred_arr = predictions.as_array().to_owned();

        // Compute score
        let score = compute_score(&test_y, &pred_arr, scoring)?;
        scores.push(score);
    }

    let scores_arr = Array1::from_vec(scores);
    Ok(scores_arr.into_pyarray(py).into())
}

// =============================================================================
// Helpers
// =============================================================================

/// Convert CVFold vector to Python list of (train_indices, test_indices) tuples
fn folds_to_py_list(py: Python<'_>, folds: &[ferroml_core::cv::CVFold]) -> PyResult<PyObject> {
    let list = pyo3::types::PyList::empty(py);
    for fold in folds {
        let train = Array1::from_vec(fold.train_indices.iter().map(|&i| i as i64).collect())
            .into_pyarray(py);
        let test = Array1::from_vec(fold.test_indices.iter().map(|&i| i as i64).collect())
            .into_pyarray(py);
        let tuple = PyTuple::new(py, [train.as_any(), test.as_any()])?;
        list.append(tuple)?;
    }
    Ok(list.into())
}

/// Select rows from a 2D array by index
fn select_rows(arr: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    arr.select(Axis(0), indices)
}

/// Select elements from a 1D array by index
fn select_elements(arr: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_vec(indices.iter().map(|&i| arr[i]).collect())
}

/// Compute a scoring metric between true and predicted values
fn compute_score(y_true: &Array1<f64>, y_pred: &Array1<f64>, scoring: &str) -> PyResult<f64> {
    let n = y_true.len() as f64;
    match scoring {
        "accuracy" => {
            let correct = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(a, b)| (*a - *b).abs() < 1e-10)
                .count();
            Ok(correct as f64 / n)
        }
        "mse" => {
            let mse: f64 = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                / n;
            // Return negative MSE (sklearn convention: higher is better)
            Ok(-mse)
        }
        "mae" => {
            let mae: f64 = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / n;
            Ok(-mae)
        }
        "r2" => {
            let mean_y = y_true.iter().sum::<f64>() / n;
            let ss_res: f64 = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            let ss_tot: f64 = y_true.iter().map(|a| (a - mean_y).powi(2)).sum();
            if ss_tot == 0.0 {
                Ok(0.0)
            } else {
                Ok(1.0 - ss_res / ss_tot)
            }
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown scoring '{}'",
            scoring
        ))),
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register the cv submodule
pub fn register_cv_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "cv")?;
    m.add_class::<PyKFold>()?;
    m.add_class::<PyStratifiedKFold>()?;
    m.add_class::<PyTimeSeriesSplit>()?;
    m.add_class::<PyLeaveOneOut>()?;
    m.add_class::<PyRepeatedKFold>()?;
    m.add_class::<PyShuffleSplit>()?;
    m.add_class::<PyGroupKFold>()?;
    m.add_class::<PyLeavePOut>()?;
    m.add_function(wrap_pyfunction!(cross_val_score, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
