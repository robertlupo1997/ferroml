//! Python bindings for FerroML Model Selection
//!
//! This module provides the `train_test_split` function matching sklearn's API.

use ferroml_core::cv::{CrossValidator, ShuffleSplit, StratifiedKFold};
use ndarray::{Array1, Axis};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Split arrays into random train and test subsets.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Feature matrix of shape (n_samples, n_features).
/// y : numpy.ndarray
///     Target array of shape (n_samples,).
/// test_size : float, optional (default=0.25)
///     Fraction of samples to include in the test split.
/// shuffle : bool, optional (default=True)
///     Whether to shuffle the data before splitting.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
/// stratify : numpy.ndarray or None, optional (default=None)
///     If not None, data is split in a stratified fashion using this as the class labels.
///
/// Returns
/// -------
/// tuple of (X_train, X_test, y_train, y_test)
///     Numpy arrays with the train/test split.
#[pyfunction]
#[pyo3(signature = (x, y, test_size=0.25, shuffle=true, random_state=None, stratify=None))]
fn train_test_split(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    test_size: f64,
    shuffle: bool,
    random_state: Option<u64>,
    stratify: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
    let x_arr = x.as_array().to_owned();
    let y_arr = y.as_array().to_owned();
    let n_samples = x_arr.nrows();

    if n_samples != y_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "X and y must have the same number of samples, got {} and {}",
            n_samples,
            y_arr.len()
        )));
    }

    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "test_size must be between 0 and 1 (exclusive)",
        ));
    }

    let fold = if let Some(strat) = stratify {
        // Stratified split: use StratifiedKFold with n_folds chosen to approximate test_size
        let strat_arr = strat.as_array().to_owned();
        let n_folds = (1.0 / test_size).round().max(2.0) as usize;
        let mut cv = StratifiedKFold::new(n_folds).with_shuffle(shuffle);
        if let Some(seed) = random_state {
            cv = cv.with_seed(seed);
        }
        let folds = cv
            .split(n_samples, Some(&strat_arr), None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // Use the first fold (test set is approximately test_size)
        folds.into_iter().next().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Failed to generate stratified split")
        })?
    } else if shuffle {
        // ShuffleSplit for a single random split
        let mut cv = ShuffleSplit::new(1).with_test_size(test_size);
        if let Some(seed) = random_state {
            cv = cv.with_seed(seed);
        }
        let folds = cv
            .split(n_samples, None, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        folds
            .into_iter()
            .next()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Failed to generate split"))?
    } else {
        // No shuffle: sequential split
        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = n_samples - n_test;
        ferroml_core::cv::CVFold {
            train_indices: (0..n_train).collect(),
            test_indices: (n_train..n_samples).collect(),
            fold_index: 0,
        }
    };

    let x_train = x_arr.select(Axis(0), &fold.train_indices);
    let x_test = x_arr.select(Axis(0), &fold.test_indices);
    let y_train = Array1::from_vec(fold.train_indices.iter().map(|&i| y_arr[i]).collect());
    let y_test = Array1::from_vec(fold.test_indices.iter().map(|&i| y_arr[i]).collect());

    Ok((
        x_train.into_pyarray(py).into(),
        x_test.into_pyarray(py).into(),
        y_train.into_pyarray(py).into(),
        y_test.into_pyarray(py).into(),
    ))
}

/// Register the model_selection submodule
pub fn register_model_selection_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "model_selection")?;
    m.add_function(wrap_pyfunction!(train_test_split, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
