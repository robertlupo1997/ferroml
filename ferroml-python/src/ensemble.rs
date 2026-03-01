//! Python bindings for FerroML ensemble and SGD models
//!
//! This module provides Python wrappers for:
//! - ExtraTreesClassifier, ExtraTreesRegressor
//! - AdaBoostClassifier, AdaBoostRegressor
//! - SGDClassifier, SGDRegressor
//! - PassiveAggressiveClassifier
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays, a copy is made.
//! Output arrays use `into_pyarray` to transfer ownership to Python without copying data.

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use crate::pickle::{getstate, setstate};
use ferroml_core::models::{
    adaboost::{AdaBoostClassifier, AdaBoostRegressor},
    extra_trees::{ExtraTreesClassifier, ExtraTreesRegressor},
    sgd::{PassiveAggressiveClassifier, SGDClassifier, SGDRegressor},
    Model,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// =============================================================================
// ExtraTreesClassifier
// =============================================================================

/// Extremely Randomized Trees classifier.
///
/// Like RandomForestClassifier but uses random thresholds at each split
/// and does not use bootstrap sampling by default.
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=100)
///     Number of trees in the ensemble.
/// max_depth : int, optional
///     Maximum depth of each tree.
/// min_samples_split : int, optional (default=2)
///     Minimum samples to split an internal node.
/// min_samples_leaf : int, optional (default=1)
///     Minimum samples at a leaf node.
/// random_state : int, optional
///     Random seed for reproducibility.
#[pyclass(name = "ExtraTreesClassifier", module = "ferroml.ensemble")]
pub struct PyExtraTreesClassifier {
    inner: ExtraTreesClassifier,
}

impl PyExtraTreesClassifier {
    pub fn inner_ref(&self) -> &ExtraTreesClassifier {
        &self.inner
    }
}

#[pymethods]
impl PyExtraTreesClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: Option<u64>,
    ) -> Self {
        let mut clf = ExtraTreesClassifier::new()
            .with_n_estimators(n_estimators)
            .with_max_depth(max_depth)
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf);
        if let Some(seed) = random_state {
            clf = clf.with_random_state(seed);
        }
        Self { inner: clf }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self
            .inner
            .feature_importances()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(imp.clone().into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtraTreesClassifier(n_estimators={})",
            self.inner.n_estimators
        )
    }
}

// =============================================================================
// ExtraTreesRegressor
// =============================================================================

/// Extremely Randomized Trees regressor.
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=100)
///     Number of trees in the ensemble.
/// max_depth : int, optional
///     Maximum depth of each tree.
/// min_samples_split : int, optional (default=2)
///     Minimum samples to split an internal node.
/// min_samples_leaf : int, optional (default=1)
///     Minimum samples at a leaf node.
/// random_state : int, optional
///     Random seed for reproducibility.
#[pyclass(name = "ExtraTreesRegressor", module = "ferroml.ensemble")]
pub struct PyExtraTreesRegressor {
    inner: ExtraTreesRegressor,
}

impl PyExtraTreesRegressor {
    pub fn inner_ref(&self) -> &ExtraTreesRegressor {
        &self.inner
    }
}

#[pymethods]
impl PyExtraTreesRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        random_state: Option<u64>,
    ) -> Self {
        let mut reg = ExtraTreesRegressor::new()
            .with_n_estimators(n_estimators)
            .with_max_depth(max_depth)
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf);
        if let Some(seed) = random_state {
            reg = reg.with_random_state(seed);
        }
        Self { inner: reg }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self
            .inner
            .feature_importances()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(imp.clone().into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtraTreesRegressor(n_estimators={})",
            self.inner.n_estimators
        )
    }
}

// =============================================================================
// AdaBoostClassifier
// =============================================================================

/// AdaBoost classifier using SAMME.R algorithm.
///
/// Fits an ensemble of weighted decision stumps.
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=50)
///     Number of boosting rounds.
/// learning_rate : float, optional (default=1.0)
///     Weight applied to each classifier at each boosting round.
/// max_depth : int, optional (default=1)
///     Maximum depth of each base estimator.
/// random_state : int, optional
///     Random seed.
#[pyclass(name = "AdaBoostClassifier", module = "ferroml.ensemble")]
pub struct PyAdaBoostClassifier {
    inner: AdaBoostClassifier,
}

impl PyAdaBoostClassifier {
    pub fn inner_ref(&self) -> &AdaBoostClassifier {
        &self.inner
    }
}

#[pymethods]
impl PyAdaBoostClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=50, learning_rate=1.0, max_depth=1, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        random_state: Option<u64>,
    ) -> Self {
        let mut clf = AdaBoostClassifier::new(n_estimators)
            .with_learning_rate(learning_rate)
            .with_max_depth(max_depth);
        if let Some(seed) = random_state {
            clf = clf.with_random_state(seed);
        }
        Self { inner: clf }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "AdaBoostClassifier(n_estimators={})",
            self.inner.n_estimators
        )
    }
}

// =============================================================================
// AdaBoostRegressor
// =============================================================================

/// AdaBoost regressor using AdaBoost.R2.
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=50)
///     Number of boosting rounds.
/// learning_rate : float, optional (default=1.0)
///     Weight applied to each estimator.
/// loss : str, optional (default="linear")
///     Loss function: "linear", "square", "exponential".
/// max_depth : int, optional (default=3)
///     Maximum depth of each base estimator.
/// random_state : int, optional
///     Random seed.
#[pyclass(name = "AdaBoostRegressor", module = "ferroml.ensemble")]
pub struct PyAdaBoostRegressor {
    inner: AdaBoostRegressor,
}

impl PyAdaBoostRegressor {
    pub fn inner_ref(&self) -> &AdaBoostRegressor {
        &self.inner
    }
}

#[pymethods]
impl PyAdaBoostRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=50, learning_rate=1.0, loss="linear", max_depth=3, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        loss: &str,
        max_depth: usize,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        use ferroml_core::models::adaboost::AdaBoostLoss;

        let loss_enum = match loss {
            "linear" => AdaBoostLoss::Linear,
            "square" => AdaBoostLoss::Square,
            "exponential" => AdaBoostLoss::Exponential,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown loss: '{}'. Use 'linear', 'square', or 'exponential'.",
                    loss
                )));
            }
        };

        let mut reg = AdaBoostRegressor::new(n_estimators)
            .with_learning_rate(learning_rate)
            .with_loss(loss_enum)
            .with_max_depth(max_depth);
        if let Some(seed) = random_state {
            reg = reg.with_random_state(seed);
        }
        Ok(Self { inner: reg })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "AdaBoostRegressor(n_estimators={})",
            self.inner.n_estimators
        )
    }
}

// =============================================================================
// SGDClassifier
// =============================================================================

/// Linear classifier fitted via Stochastic Gradient Descent.
///
/// Supports multiple loss functions and regularization penalties.
///
/// Parameters
/// ----------
/// loss : str, optional (default="hinge")
///     Loss function: "hinge", "log", "modified_huber".
/// penalty : str, optional (default="l2")
///     Regularization: "none", "l1", "l2", "elasticnet".
/// alpha : float, optional (default=0.0001)
///     Regularization strength.
/// max_iter : int, optional (default=1000)
///     Maximum number of epochs.
/// tol : float, optional (default=1e-3)
///     Convergence tolerance.
/// random_state : int, optional
///     Random seed.
#[pyclass(name = "SGDClassifier", module = "ferroml.ensemble")]
pub struct PySGDClassifier {
    inner: SGDClassifier,
}

impl PySGDClassifier {
    pub fn inner_ref(&self) -> &SGDClassifier {
        &self.inner
    }
}

#[pymethods]
impl PySGDClassifier {
    #[new]
    #[pyo3(signature = (loss="hinge", penalty="l2", alpha=0.0001, max_iter=1000, tol=1e-3, random_state=None))]
    fn new(
        loss: &str,
        penalty: &str,
        alpha: f64,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        use ferroml_core::models::sgd::{Penalty, SGDClassifierLoss};

        let loss_enum = match loss {
            "hinge" => SGDClassifierLoss::Hinge,
            "log" => SGDClassifierLoss::Log,
            "modified_huber" => SGDClassifierLoss::ModifiedHuber,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown loss: '{}'. Use 'hinge', 'log', or 'modified_huber'.",
                    loss
                )));
            }
        };

        let penalty_enum = match penalty {
            "none" => Penalty::None,
            "l1" => Penalty::L1,
            "l2" => Penalty::L2,
            "elasticnet" => Penalty::ElasticNet,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown penalty: '{}'. Use 'none', 'l1', 'l2', or 'elasticnet'.",
                    penalty
                )));
            }
        };

        let mut clf = SGDClassifier::new();
        clf.loss = loss_enum;
        clf.penalty = penalty_enum;
        clf.alpha = alpha;
        clf.max_iter = max_iter;
        clf.tol = tol;
        if let Some(seed) = random_state {
            clf.random_state = Some(seed);
        }
        Ok(Self { inner: clf })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "SGDClassifier()".to_string()
    }
}

// =============================================================================
// SGDRegressor
// =============================================================================

/// Linear regressor fitted via Stochastic Gradient Descent.
///
/// Parameters
/// ----------
/// loss : str, optional (default="squared_error")
///     Loss function: "squared_error", "huber", "epsilon_insensitive".
/// penalty : str, optional (default="l2")
///     Regularization: "none", "l1", "l2", "elasticnet".
/// alpha : float, optional (default=0.0001)
///     Regularization strength.
/// max_iter : int, optional (default=1000)
///     Maximum number of epochs.
/// tol : float, optional (default=1e-3)
///     Convergence tolerance.
/// random_state : int, optional
///     Random seed.
#[pyclass(name = "SGDRegressor", module = "ferroml.ensemble")]
pub struct PySGDRegressor {
    inner: SGDRegressor,
}

impl PySGDRegressor {
    pub fn inner_ref(&self) -> &SGDRegressor {
        &self.inner
    }
}

#[pymethods]
impl PySGDRegressor {
    #[new]
    #[pyo3(signature = (loss="squared_error", penalty="l2", alpha=0.0001, max_iter=1000, tol=1e-3, random_state=None))]
    fn new(
        loss: &str,
        penalty: &str,
        alpha: f64,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        use ferroml_core::models::sgd::{Penalty, SGDRegressorLoss};

        let loss_enum = match loss {
            "squared_error" => SGDRegressorLoss::SquaredError,
            "huber" => SGDRegressorLoss::Huber,
            "epsilon_insensitive" => SGDRegressorLoss::EpsilonInsensitive,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown loss: '{}'. Use 'squared_error', 'huber', or 'epsilon_insensitive'.",
                    loss
                )));
            }
        };

        let penalty_enum = match penalty {
            "none" => Penalty::None,
            "l1" => Penalty::L1,
            "l2" => Penalty::L2,
            "elasticnet" => Penalty::ElasticNet,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown penalty: '{}'. Use 'none', 'l1', 'l2', or 'elasticnet'.",
                    penalty
                )));
            }
        };

        let mut reg = SGDRegressor::new()
            .with_loss(loss_enum)
            .with_penalty(penalty_enum)
            .with_alpha(alpha)
            .with_max_iter(max_iter);
        reg.tol = tol;
        if let Some(seed) = random_state {
            reg = reg.with_random_state(seed);
        }
        Ok(Self { inner: reg })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .inner
            .coef()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(coef.clone().into_pyarray(py))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner
            .intercept()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "SGDRegressor()".to_string()
    }
}

// =============================================================================
// PassiveAggressiveClassifier
// =============================================================================

/// Passive Aggressive classifier.
///
/// Online learning classifier that makes aggressive updates when a
/// prediction violates the margin.
///
/// Parameters
/// ----------
/// c : float, optional (default=1.0)
///     Regularization parameter (aggressiveness).
/// max_iter : int, optional (default=1000)
///     Maximum epochs.
/// tol : float, optional (default=1e-3)
///     Convergence tolerance.
/// random_state : int, optional
///     Random seed.
#[pyclass(name = "PassiveAggressiveClassifier", module = "ferroml.ensemble")]
pub struct PyPassiveAggressiveClassifier {
    inner: PassiveAggressiveClassifier,
}

impl PyPassiveAggressiveClassifier {
    pub fn inner_ref(&self) -> &PassiveAggressiveClassifier {
        &self.inner
    }
}

#[pymethods]
impl PyPassiveAggressiveClassifier {
    #[new]
    #[pyo3(signature = (c=1.0, max_iter=1000, tol=1e-3, random_state=None))]
    fn new(c: f64, max_iter: usize, tol: f64, random_state: Option<u64>) -> Self {
        let mut clf = PassiveAggressiveClassifier::new(c);
        clf.max_iter = max_iter;
        clf.tol = tol;
        if let Some(seed) = random_state {
            clf.random_state = Some(seed);
        }
        Self { inner: clf }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("PassiveAggressiveClassifier(c={})", self.inner.c)
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the ensemble submodule.
pub fn register_ensemble_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "ensemble")?;

    m.add_class::<PyExtraTreesClassifier>()?;
    m.add_class::<PyExtraTreesRegressor>()?;
    m.add_class::<PyAdaBoostClassifier>()?;
    m.add_class::<PyAdaBoostRegressor>()?;
    m.add_class::<PySGDClassifier>()?;
    m.add_class::<PySGDRegressor>()?;
    m.add_class::<PyPassiveAggressiveClassifier>()?;

    parent_module.add_submodule(&m)?;

    Ok(())
}
