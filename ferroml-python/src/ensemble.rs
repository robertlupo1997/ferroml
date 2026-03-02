//! Python bindings for FerroML ensemble and SGD models
//!
//! This module provides Python wrappers for:
//! - ExtraTreesClassifier, ExtraTreesRegressor
//! - AdaBoostClassifier, AdaBoostRegressor
//! - SGDClassifier, SGDRegressor
//! - PassiveAggressiveClassifier
//! - BaggingClassifier (via factory pattern)
//! - BaggingRegressor (via factory pattern)
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays, a copy is made.
//! Output arrays use `into_pyarray` to transfer ownership to Python without copying data.

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use crate::pickle::{getstate, setstate};
use ferroml_core::ensemble::bagging::MaxFeatures as BaggingMaxFeatures;
use ferroml_core::ensemble::voting::VotingClassifierEstimator;
use ferroml_core::ensemble::{BaggingClassifier, BaggingRegressor, MaxSamples};
use ferroml_core::models::{
    AdaBoostClassifier, AdaBoostRegressor, DecisionTreeClassifier, DecisionTreeRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor, GaussianNB, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    KNeighborsClassifier, KNeighborsRegressor, LinearRegression, LogisticRegression, Model,
    PassiveAggressiveClassifier, RandomForestClassifier, RandomForestRegressor, RidgeRegression,
    SGDClassifier, SGDRegressor, SVC, SVR,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
// Shared penalty parser for SGD models
// =============================================================================

fn parse_penalty(penalty: &str) -> PyResult<ferroml_core::models::sgd::Penalty> {
    use ferroml_core::models::sgd::Penalty;
    match penalty {
        "none" => Ok(Penalty::None),
        "l1" => Ok(Penalty::L1),
        "l2" => Ok(Penalty::L2),
        "elasticnet" => Ok(Penalty::ElasticNet),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown penalty: '{}'. Use 'none', 'l1', 'l2', or 'elasticnet'.",
            penalty
        ))),
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
        use ferroml_core::models::sgd::SGDClassifierLoss;

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

        let penalty_enum = parse_penalty(penalty)?;

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
        use ferroml_core::models::sgd::SGDRegressorLoss;

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

        let penalty_enum = parse_penalty(penalty)?;

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
// BaggingClassifier
// =============================================================================

/// Bagging (Bootstrap Aggregating) classifier.
///
/// Builds an ensemble of classifiers, each trained on a bootstrap sample
/// of the training data, with optional feature subsampling.
///
/// Since BaggingClassifier requires a base estimator (trait object), use
/// the factory static methods to create instances with specific base types:
///
/// - ``BaggingClassifier.with_decision_tree(...)``
/// - ``BaggingClassifier.with_random_forest(...)``
/// - ``BaggingClassifier.with_logistic_regression(...)``
/// - ``BaggingClassifier.with_gaussian_nb(...)``
/// - ``BaggingClassifier.with_knn(...)``
/// - ``BaggingClassifier.with_svc(...)``
/// - ``BaggingClassifier.with_gradient_boosting(...)``
/// - ``BaggingClassifier.with_hist_gradient_boosting(...)``
///
/// Parameters (common to all factory methods)
/// -------------------------------------------
/// n_estimators : int, optional (default=10)
///     Number of base estimators in the ensemble.
/// max_samples : float, optional (default=1.0)
///     Fraction of samples to draw for each base estimator.
/// max_features : float, optional (default=1.0)
///     Fraction of features to draw for each base estimator.
/// bootstrap : bool, optional (default=True)
///     Whether to sample with replacement.
/// oob_score : bool, optional (default=False)
///     Whether to compute out-of-bag score after fitting.
/// random_state : int, optional
///     Random seed for reproducibility.
/// warm_start : bool, optional (default=False)
///     Whether to keep existing estimators when fitting again.
///
/// Example
/// -------
/// >>> from ferroml.ensemble import BaggingClassifier
/// >>> import numpy as np
/// >>>
/// >>> model = BaggingClassifier.with_decision_tree(
/// ...     n_estimators=10, max_depth=5, random_state=42
/// ... )
/// >>> model.fit(X_train, y_train)
/// >>> predictions = model.predict(X_test)
/// >>> print(f"OOB score: {model.oob_score_}")
#[pyclass(name = "BaggingClassifier", module = "ferroml.ensemble")]
pub struct PyBaggingClassifier {
    inner: BaggingClassifier,
    /// Store n_estimators for __repr__ (field is private in core)
    n_estimators_cfg: usize,
    /// Description of the base estimator type for __repr__
    base_estimator_name: &'static str,
}

#[pymethods]
impl PyBaggingClassifier {
    /// Create a BaggingClassifier with a DecisionTreeClassifier base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// max_depth : int, optional
    ///     Maximum depth of each decision tree.
    /// min_samples_split : int, optional (default=2)
    ///     Minimum samples to split a node in each tree.
    /// min_samples_leaf : int, optional (default=1)
    ///     Minimum samples at a leaf node in each tree.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_decision_tree(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        let base = DecisionTreeClassifier::new()
            .with_max_depth(max_depth)
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf);
        let inner = build_bagging_classifier(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "DecisionTreeClassifier",
        }
    }

    /// Create a BaggingClassifier with a RandomForestClassifier base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of bagging estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// rf_n_estimators : int, optional (default=100)
    ///     Number of trees in each random forest base estimator.
    /// max_depth : int, optional
    ///     Maximum depth of trees in each random forest.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        rf_n_estimators=100,
        max_depth=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_random_forest(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        rf_n_estimators: usize,
        max_depth: Option<usize>,
    ) -> Self {
        let base = RandomForestClassifier::new()
            .with_n_estimators(rf_n_estimators)
            .with_max_depth(max_depth);
        let inner = build_bagging_classifier(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "RandomForestClassifier",
        }
    }

    /// Create a BaggingClassifier with a LogisticRegression base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// max_iter : int, optional (default=100)
    ///     Maximum iterations for logistic regression convergence.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        max_iter=100,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_logistic_regression(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        max_iter: usize,
    ) -> Self {
        let base = LogisticRegression::new().with_max_iter(max_iter);
        let inner = build_bagging_classifier(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "LogisticRegression",
        }
    }

    /// Create a BaggingClassifier with a GaussianNB base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
    ))]
    fn with_gaussian_nb(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
    ) -> Self {
        let base = GaussianNB::new();
        let inner = build_bagging_classifier(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "GaussianNB",
        }
    }

    /// Create a BaggingClassifier with a KNeighborsClassifier base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// n_neighbors : int, optional (default=5)
    ///     Number of neighbors for KNN.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        n_neighbors=5,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_knn(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        n_neighbors: usize,
    ) -> Self {
        let base = KNeighborsClassifier::new(n_neighbors);
        let inner = build_bagging_classifier(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "KNeighborsClassifier",
        }
    }

    /// Create a BaggingClassifier with an SVC base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// c : float, optional (default=1.0)
    ///     Regularization parameter for SVC.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        c=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_svc(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        c: f64,
    ) -> Self {
        let base = SVC::new().with_c(c).with_probability(true);
        let inner = build_bagging_classifier(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "SVC",
        }
    }

    /// Create a BaggingClassifier with a GradientBoostingClassifier base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of bagging estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// gb_n_estimators : int, optional (default=100)
    ///     Number of boosting stages in each gradient boosting base estimator.
    /// learning_rate : float, optional (default=0.1)
    ///     Learning rate for gradient boosting.
    /// max_depth : int, optional (default=3)
    ///     Maximum depth of trees in gradient boosting.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        gb_n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_gradient_boosting(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        gb_n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
    ) -> Self {
        let base = GradientBoostingClassifier::new()
            .with_n_estimators(gb_n_estimators)
            .with_learning_rate(learning_rate)
            .with_max_depth(Some(max_depth));
        let inner = build_bagging_classifier(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "GradientBoostingClassifier",
        }
    }

    /// Create a BaggingClassifier with a HistGradientBoostingClassifier base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of bagging estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// hgb_max_iter : int, optional (default=100)
    ///     Maximum number of boosting iterations in each hist gradient boosting base estimator.
    /// learning_rate : float, optional (default=0.1)
    ///     Learning rate for hist gradient boosting.
    /// max_depth : int, optional
    ///     Maximum depth of trees in hist gradient boosting.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        hgb_max_iter=100,
        learning_rate=0.1,
        max_depth=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_hist_gradient_boosting(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        hgb_max_iter: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
    ) -> Self {
        let base = HistGradientBoostingClassifier::new()
            .with_max_iter(hgb_max_iter)
            .with_learning_rate(learning_rate)
            .with_max_depth(max_depth);
        let inner = build_bagging_classifier(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "HistGradientBoostingClassifier",
        }
    }

    /// Fit the bagging classifier on training data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Training feature matrix.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Target class labels.
    ///
    /// Returns
    /// -------
    /// self
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

    /// Predict class labels for samples.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Predicted class labels.
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

    /// Predict class probabilities for samples.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples, n_classes)
    ///     Predicted class probabilities.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(probas.into_pyarray(py))
    }

    /// Out-of-bag accuracy score (only available if ``oob_score=True``).
    ///
    /// Returns
    /// -------
    /// float or None
    ///     OOB accuracy score, or None if not computed.
    #[getter]
    fn oob_score_(&self) -> Option<f64> {
        self.inner.oob_score()
    }

    /// Number of fitted estimators.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of estimators that have been fitted.
    #[getter]
    fn n_estimators_(&self) -> usize {
        self.inner.n_fitted_estimators()
    }

    /// Feature importances (averaged across estimators).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_features,) or None
    ///     Normalized feature importances, or raises if not fitted.
    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(imp.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "BaggingClassifier(base_estimator={}, n_estimators={})",
            self.base_estimator_name, self.n_estimators_cfg
        )
    }
}

// =============================================================================
// BaggingRegressor
// =============================================================================

/// Bootstrap Aggregating (Bagging) regressor.
///
/// Uses the factory pattern to create instances with specific base regressors.
/// Since the base estimator uses a trait object (`Box<dyn Model>`), instances
/// cannot be constructed with `__init__`. Instead, use one of the static
/// factory methods:
///
/// - ``BaggingRegressor.with_decision_tree(...)``
/// - ``BaggingRegressor.with_random_forest(...)``
/// - ``BaggingRegressor.with_linear_regression(...)``
/// - ``BaggingRegressor.with_ridge_regression(...)``
/// - ``BaggingRegressor.with_extra_trees(...)``
/// - ``BaggingRegressor.with_gradient_boosting(...)``
/// - ``BaggingRegressor.with_hist_gradient_boosting(...)``
/// - ``BaggingRegressor.with_svr(...)``
/// - ``BaggingRegressor.with_knn(...)``
///
/// Example
/// -------
/// >>> from ferroml.ensemble import BaggingRegressor
/// >>> import numpy as np
/// >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
/// >>> y_train = np.array([1.0, 2.0, 3.0, 4.0])
/// >>> model = BaggingRegressor.with_decision_tree(n_estimators=10, max_depth=5)
/// >>> model.fit(X_train, y_train)
/// >>> predictions = model.predict(X_train)
/// >>> print(f"OOB R² score: {model.oob_score_}")
#[pyclass(name = "BaggingRegressor", module = "ferroml.ensemble")]
pub struct PyBaggingRegressor {
    inner: BaggingRegressor,
    /// Store n_estimators for __repr__ (field is private in core)
    n_estimators_cfg: usize,
    /// Description of the base estimator type for __repr__
    base_estimator_name: &'static str,
}

#[pymethods]
impl PyBaggingRegressor {
    /// Create a BaggingRegressor with a DecisionTreeRegressor base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// max_depth : int, optional
    ///     Maximum depth of each decision tree.
    /// min_samples_split : int, optional (default=2)
    ///     Minimum samples to split a node in each tree.
    /// min_samples_leaf : int, optional (default=1)
    ///     Minimum samples at a leaf node in each tree.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_decision_tree(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        let base = DecisionTreeRegressor::new()
            .with_max_depth(max_depth)
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf);
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "DecisionTreeRegressor",
        }
    }

    /// Create a BaggingRegressor with a RandomForestRegressor base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of bagging estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// rf_n_estimators : int, optional (default=100)
    ///     Number of trees in each random forest base estimator.
    /// max_depth : int, optional
    ///     Maximum depth of trees in each random forest.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        rf_n_estimators=100,
        max_depth=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_random_forest(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        rf_n_estimators: usize,
        max_depth: Option<usize>,
    ) -> Self {
        let base = RandomForestRegressor::new()
            .with_n_estimators(rf_n_estimators)
            .with_max_depth(max_depth);
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "RandomForestRegressor",
        }
    }

    /// Create a BaggingRegressor with a LinearRegression base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// fit_intercept : bool, optional (default=True)
    ///     Whether to fit an intercept term.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        fit_intercept=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_linear_regression(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        fit_intercept: bool,
    ) -> Self {
        let base = LinearRegression::new().with_fit_intercept(fit_intercept);
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "LinearRegression",
        }
    }

    /// Create a BaggingRegressor with a RidgeRegression base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// alpha : float, optional (default=1.0)
    ///     Regularization strength for Ridge.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        alpha=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_ridge_regression(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        alpha: f64,
    ) -> Self {
        let base = RidgeRegression::new(alpha);
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "RidgeRegression",
        }
    }

    /// Create a BaggingRegressor with an ExtraTreesRegressor base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of bagging estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// et_n_estimators : int, optional (default=100)
    ///     Number of trees in each extra trees base estimator.
    /// max_depth : int, optional
    ///     Maximum depth of trees in each extra trees estimator.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        et_n_estimators=100,
        max_depth=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_extra_trees(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        et_n_estimators: usize,
        max_depth: Option<usize>,
    ) -> Self {
        let base = ExtraTreesRegressor::new()
            .with_n_estimators(et_n_estimators)
            .with_max_depth(max_depth);
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "ExtraTreesRegressor",
        }
    }

    /// Create a BaggingRegressor with a GradientBoostingRegressor base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of bagging estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// gb_n_estimators : int, optional (default=100)
    ///     Number of boosting stages in each gradient boosting base estimator.
    /// learning_rate : float, optional (default=0.1)
    ///     Learning rate for gradient boosting.
    /// max_depth : int, optional (default=3)
    ///     Maximum depth of trees in gradient boosting.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        gb_n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_gradient_boosting(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        gb_n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
    ) -> Self {
        let base = GradientBoostingRegressor::new()
            .with_n_estimators(gb_n_estimators)
            .with_learning_rate(learning_rate)
            .with_max_depth(Some(max_depth));
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "GradientBoostingRegressor",
        }
    }

    /// Create a BaggingRegressor with a HistGradientBoostingRegressor base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of bagging estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// hgb_max_iter : int, optional (default=100)
    ///     Maximum number of boosting iterations in each hist gradient boosting base estimator.
    /// learning_rate : float, optional (default=0.1)
    ///     Learning rate for hist gradient boosting.
    /// max_depth : int, optional
    ///     Maximum depth of trees in hist gradient boosting.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        hgb_max_iter=100,
        learning_rate=0.1,
        max_depth=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_hist_gradient_boosting(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        hgb_max_iter: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
    ) -> Self {
        let base = HistGradientBoostingRegressor::new()
            .with_max_iter(hgb_max_iter)
            .with_learning_rate(learning_rate)
            .with_max_depth(max_depth);
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "HistGradientBoostingRegressor",
        }
    }

    /// Create a BaggingRegressor with an SVR base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// c : float, optional (default=1.0)
    ///     Regularization parameter for SVR.
    /// epsilon : float, optional (default=0.1)
    ///     Epsilon parameter (tube width) for SVR.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        c=1.0,
        epsilon=0.1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_svr(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        c: f64,
        epsilon: f64,
    ) -> Self {
        let base = SVR::new().with_c(c).with_epsilon(epsilon);
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "SVR",
        }
    }

    /// Create a BaggingRegressor with a KNeighborsRegressor base estimator.
    ///
    /// Parameters
    /// ----------
    /// n_estimators : int, optional (default=10)
    ///     Number of base estimators.
    /// max_samples : float, optional (default=1.0)
    ///     Fraction of samples per estimator.
    /// max_features : float, optional (default=1.0)
    ///     Fraction of features per estimator.
    /// bootstrap : bool, optional (default=True)
    ///     Sample with replacement.
    /// oob_score : bool, optional (default=False)
    ///     Compute out-of-bag R² score.
    /// random_state : int, optional
    ///     Random seed.
    /// warm_start : bool, optional (default=False)
    ///     Keep existing estimators on refit.
    /// n_neighbors : int, optional (default=5)
    ///     Number of neighbors for KNN.
    #[staticmethod]
    #[pyo3(signature = (
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=true,
        oob_score=false,
        random_state=None,
        warm_start=false,
        n_neighbors=5,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn with_knn(
        n_estimators: usize,
        max_samples: f64,
        max_features: f64,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
        warm_start: bool,
        n_neighbors: usize,
    ) -> Self {
        let base = KNeighborsRegressor::new(n_neighbors);
        let inner = build_bagging_regressor(
            Box::new(base),
            n_estimators,
            max_samples,
            max_features,
            bootstrap,
            oob_score,
            random_state,
            warm_start,
        );
        Self {
            inner,
            n_estimators_cfg: n_estimators,
            base_estimator_name: "KNeighborsRegressor",
        }
    }

    /// Fit the bagging regressor on training data.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Training feature matrix.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self
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

    /// Predict target values for samples.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Feature matrix.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_samples,)
    ///     Predicted target values.
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

    /// Out-of-bag R² score (only available if ``oob_score=True``).
    ///
    /// Returns
    /// -------
    /// float or None
    ///     OOB R² score, or None if not computed.
    #[getter]
    fn oob_score_(&self) -> Option<f64> {
        self.inner.oob_score()
    }

    /// Number of fitted estimators.
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of estimators that have been fitted.
    #[getter]
    fn n_estimators_(&self) -> usize {
        self.inner.n_fitted_estimators()
    }

    /// Feature importances (averaged across estimators).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of shape (n_features,) or None
    ///     Normalized feature importances, or raises if not fitted.
    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(imp.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "BaggingRegressor(base_estimator={}, n_estimators={})",
            self.base_estimator_name, self.n_estimators_cfg
        )
    }
}

/// Helper to build a BaggingRegressor with common parameters.
fn build_bagging_regressor(
    base_estimator: Box<dyn Model>,
    n_estimators: usize,
    max_samples: f64,
    max_features: f64,
    bootstrap: bool,
    oob_score: bool,
    random_state: Option<u64>,
    warm_start: bool,
) -> BaggingRegressor {
    let mut reg = BaggingRegressor::new(base_estimator)
        .with_n_estimators(n_estimators)
        .with_max_samples(MaxSamples::Fraction(max_samples))
        .with_max_features(BaggingMaxFeatures::Fraction(max_features))
        .with_bootstrap(bootstrap)
        .with_oob_score(oob_score)
        .with_warm_start(warm_start);
    if let Some(seed) = random_state {
        reg = reg.with_random_state(seed);
    }
    reg
}

/// Helper to build a BaggingClassifier with common parameters.
fn build_bagging_classifier(
    base_estimator: Box<dyn VotingClassifierEstimator>,
    n_estimators: usize,
    max_samples: f64,
    max_features: f64,
    bootstrap: bool,
    oob_score: bool,
    random_state: Option<u64>,
    warm_start: bool,
) -> BaggingClassifier {
    let mut clf = BaggingClassifier::new(base_estimator)
        .with_n_estimators(n_estimators)
        .with_max_samples(MaxSamples::Fraction(max_samples))
        .with_max_features(BaggingMaxFeatures::Fraction(max_features))
        .with_bootstrap(bootstrap)
        .with_oob_score(oob_score)
        .with_warm_start(warm_start);
    if let Some(seed) = random_state {
        clf = clf.with_random_state(seed);
    }
    clf
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
    m.add_class::<PyBaggingClassifier>()?;
    m.add_class::<PyBaggingRegressor>()?;

    parent_module.add_submodule(&m)?;

    Ok(())
}
