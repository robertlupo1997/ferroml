//! Python bindings for MultiOutput wrappers.
//!
//! Since Rust generics cannot cross the PyO3 boundary, we use an internal
//! enum to type-erase supported base estimators and dispatch at runtime.

use crate::array_utils::{check_array_finite, to_owned_array_2d};
use ferroml_core::models::multioutput::{MultiOutputClassifier, MultiOutputRegressor};
use ferroml_core::models::{
    DecisionTreeClassifier, DecisionTreeRegressor, KNeighborsRegressor, LinearRegression,
    LogisticRegression, RidgeRegression,
};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// MultiOutputRegressor
// ============================================================================

/// Internal enum for supported regressor base estimators.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize)]
enum RegressorInner {
    LinearReg(MultiOutputRegressor<LinearRegression>),
    Ridge(MultiOutputRegressor<RidgeRegression>),
    DecisionTree(MultiOutputRegressor<DecisionTreeRegressor>),
    Knn(MultiOutputRegressor<KNeighborsRegressor>),
}

/// Multi-output regressor: fits one regressor per target column.
///
/// Parameters
/// ----------
/// estimator : str, optional (default="linear_regression")
///     Name of the base estimator. Supported values:
///     "linear_regression", "ridge", "decision_tree", "knn".
///
/// Examples
/// --------
/// >>> from ferroml.multioutput import MultiOutputRegressor
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 3)
/// >>> Y = np.column_stack([X @ [1, 2, 3], X @ [4, 5, 6]])
/// >>> mo = MultiOutputRegressor("linear_regression")
/// >>> mo.fit(X, Y)
/// >>> preds = mo.predict(X)
/// >>> preds.shape
/// (100, 2)
#[pyclass(name = "MultiOutputRegressor", module = "ferroml.multioutput")]
pub struct PyMultiOutputRegressor {
    inner: Option<RegressorInner>,
    estimator_name: String,
}

#[pymethods]
impl PyMultiOutputRegressor {
    #[new]
    #[pyo3(signature = (estimator="linear_regression"))]
    fn new(estimator: &str) -> PyResult<Self> {
        // Validate estimator name upfront
        match estimator {
            "linear_regression" | "ridge" | "decision_tree" | "knn" => {}
            _ => {
                return Err(PyValueError::new_err(format!(
                "Unknown estimator: '{}'. Supported: linear_regression, ridge, decision_tree, knn",
                estimator
            )))
            }
        }
        Ok(Self {
            inner: None,
            estimator_name: estimator.to_string(),
        })
    }

    /// Fit one estimator per target column.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Training features.
    /// y : numpy.ndarray of shape (n_samples, n_outputs)
    ///     Training targets (one column per output).
    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = y.as_array().to_owned();

        match self.estimator_name.as_str() {
            "linear_regression" => {
                let mut mo = MultiOutputRegressor::new(LinearRegression::new());
                mo.fit_multi(&x_arr, &y_arr)
                    .map_err(crate::errors::ferro_to_pyerr)?;
                self.inner = Some(RegressorInner::LinearReg(mo));
            }
            "ridge" => {
                let mut mo = MultiOutputRegressor::new(RidgeRegression::new(1.0));
                mo.fit_multi(&x_arr, &y_arr)
                    .map_err(crate::errors::ferro_to_pyerr)?;
                self.inner = Some(RegressorInner::Ridge(mo));
            }
            "decision_tree" => {
                let mut mo = MultiOutputRegressor::new(DecisionTreeRegressor::new());
                mo.fit_multi(&x_arr, &y_arr)
                    .map_err(crate::errors::ferro_to_pyerr)?;
                self.inner = Some(RegressorInner::DecisionTree(mo));
            }
            "knn" => {
                let mut mo = MultiOutputRegressor::new(KNeighborsRegressor::new(5));
                mo.fit_multi(&x_arr, &y_arr)
                    .map_err(crate::errors::ferro_to_pyerr)?;
                self.inner = Some(RegressorInner::Knn(mo));
            }
            _ => unreachable!("Validated in __init__"),
        }
        Ok(())
    }

    /// Predict all outputs.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Input features.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples, n_outputs)
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = match self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Not fitted"))?
        {
            RegressorInner::LinearReg(mo) => mo.predict_multi(&x_arr),
            RegressorInner::Ridge(mo) => mo.predict_multi(&x_arr),
            RegressorInner::DecisionTree(mo) => mo.predict_multi(&x_arr),
            RegressorInner::Knn(mo) => mo.predict_multi(&x_arr),
        }
        .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Number of output columns (available after fitting).
    #[getter]
    fn n_outputs(&self) -> PyResult<Option<usize>> {
        Ok(match &self.inner {
            Some(RegressorInner::LinearReg(mo)) => mo.n_outputs(),
            Some(RegressorInner::Ridge(mo)) => mo.n_outputs(),
            Some(RegressorInner::DecisionTree(mo)) => mo.n_outputs(),
            Some(RegressorInner::Knn(mo)) => mo.n_outputs(),
            None => None,
        })
    }

    /// Whether the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.inner.is_some()
    }

    /// Serialize for pickle.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Py<pyo3::types::PyBytes>> {
        crate::pickle::getstate(py, &(&self.inner, &self.estimator_name))
    }
    /// Deserialize for pickle.
    pub fn __setstate__(&mut self, state: &pyo3::Bound<'_, pyo3::types::PyBytes>) -> PyResult<()> {
        let (inner, name): (Option<RegressorInner>, String) =
            crate::pickle::setstate(state.as_bytes())?;
        self.inner = inner;
        self.estimator_name = name;
        Ok(())
    }
}

// ============================================================================
// MultiOutputClassifier
// ============================================================================

/// Internal enum for supported classifier base estimators.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize)]
enum ClassifierInner {
    LogisticReg(MultiOutputClassifier<LogisticRegression>),
    DecisionTree(MultiOutputClassifier<DecisionTreeClassifier>),
}

/// Multi-output classifier: fits one classifier per target column.
///
/// Parameters
/// ----------
/// estimator : str, optional (default="logistic_regression")
///     Name of the base estimator. Supported values:
///     "logistic_regression", "decision_tree".
///
/// Examples
/// --------
/// >>> from ferroml.multioutput import MultiOutputClassifier
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 3)
/// >>> Y = np.column_stack([(X[:, 0] > 0).astype(float), (X[:, 1] > 0).astype(float)])
/// >>> mo = MultiOutputClassifier("logistic_regression")
/// >>> mo.fit(X, Y)
/// >>> preds = mo.predict(X)
/// >>> preds.shape
/// (100, 2)
#[pyclass(name = "MultiOutputClassifier", module = "ferroml.multioutput")]
pub struct PyMultiOutputClassifier {
    inner: Option<ClassifierInner>,
    estimator_name: String,
}

#[pymethods]
impl PyMultiOutputClassifier {
    #[new]
    #[pyo3(signature = (estimator="logistic_regression"))]
    fn new(estimator: &str) -> PyResult<Self> {
        match estimator {
            "logistic_regression" | "decision_tree" => {}
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown estimator: '{}'. Supported: logistic_regression, decision_tree",
                    estimator
                )))
            }
        }
        Ok(Self {
            inner: None,
            estimator_name: estimator.to_string(),
        })
    }

    /// Fit one estimator per target column.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Training features.
    /// y : numpy.ndarray of shape (n_samples, n_outputs)
    ///     Training targets (one column per output, binary or multiclass labels).
    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = y.as_array().to_owned();

        match self.estimator_name.as_str() {
            "logistic_regression" => {
                let mut mo = MultiOutputClassifier::new(LogisticRegression::new());
                mo.fit_multi(&x_arr, &y_arr)
                    .map_err(crate::errors::ferro_to_pyerr)?;
                self.inner = Some(ClassifierInner::LogisticReg(mo));
            }
            "decision_tree" => {
                let mut mo = MultiOutputClassifier::new(DecisionTreeClassifier::new());
                mo.fit_multi(&x_arr, &y_arr)
                    .map_err(crate::errors::ferro_to_pyerr)?;
                self.inner = Some(ClassifierInner::DecisionTree(mo));
            }
            _ => unreachable!("Validated in __init__"),
        }
        Ok(())
    }

    /// Predict all outputs.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Input features.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples, n_outputs)
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = match self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Not fitted"))?
        {
            ClassifierInner::LogisticReg(mo) => mo.predict_multi(&x_arr),
            ClassifierInner::DecisionTree(mo) => mo.predict_multi(&x_arr),
        }
        .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Predict class probabilities for each output.
    ///
    /// Returns a list of 2D arrays, one per output column.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Input features.
    ///
    /// Returns
    /// -------
    /// list of numpy.ndarray
    ///     Each element has shape (n_samples, n_classes_for_that_output).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let results = match self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Not fitted"))?
        {
            ClassifierInner::LogisticReg(mo) => mo.predict_proba_multi(&x_arr),
            ClassifierInner::DecisionTree(mo) => mo.predict_proba_multi(&x_arr),
        }
        .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(results
            .into_iter()
            .map(|arr| arr.into_pyarray(py))
            .collect())
    }

    /// Number of output columns (available after fitting).
    #[getter]
    fn n_outputs(&self) -> PyResult<Option<usize>> {
        Ok(match &self.inner {
            Some(ClassifierInner::LogisticReg(mo)) => mo.n_outputs(),
            Some(ClassifierInner::DecisionTree(mo)) => mo.n_outputs(),
            None => None,
        })
    }

    /// Whether the model has been fitted.
    fn is_fitted(&self) -> bool {
        self.inner.is_some()
    }

    /// Serialize for pickle.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Py<pyo3::types::PyBytes>> {
        crate::pickle::getstate(py, &(&self.inner, &self.estimator_name))
    }
    /// Deserialize for pickle.
    pub fn __setstate__(&mut self, state: &pyo3::Bound<'_, pyo3::types::PyBytes>) -> PyResult<()> {
        let (inner, name): (Option<ClassifierInner>, String) =
            crate::pickle::setstate(state.as_bytes())?;
        self.inner = inner;
        self.estimator_name = name;
        Ok(())
    }
}

// ============================================================================
// Module registration
// ============================================================================

pub fn register_multioutput_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "multioutput")?;
    m.add_class::<PyMultiOutputRegressor>()?;
    m.add_class::<PyMultiOutputClassifier>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
