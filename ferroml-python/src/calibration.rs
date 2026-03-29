//! Python bindings for FerroML calibration models
//!
//! This module provides Python wrappers for:
//! - TemperatureScalingCalibrator (post-hoc multi-class probability calibration)

use crate::array_utils::{py_array_to_f64_1d, to_owned_array_2d};
use ferroml_core::model_card::HasModelCard;
use ferroml_core::models::calibration::{MulticlassCalibrator, TemperatureScalingCalibrator};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::model_card::PyModelCard;

// =============================================================================
// TemperatureScalingCalibrator
// =============================================================================

/// Temperature scaling calibrator for multi-class classifiers.
///
/// Temperature scaling learns a single scalar parameter T to recalibrate
/// predicted probabilities: calibrated = softmax(log(proba) / T).
///
/// This is particularly effective for overconfident neural network predictions.
///
/// Parameters
/// ----------
/// max_iter : int, optional (default=100)
///     Maximum number of optimization iterations.
/// learning_rate : float, optional (default=0.01)
///     Learning rate for gradient descent optimization. Valid range: (0, inf).
///
/// Attributes
/// ----------
/// temperature_ : float
///     The learned temperature parameter. Values > 1 soften probabilities
///     (reduce confidence), values < 1 sharpen them.
///
/// Examples
/// --------
/// >>> from ferroml.calibration import TemperatureScalingCalibrator
/// >>> import numpy as np
/// >>> # Uncalibrated probabilities from a classifier
/// >>> y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
/// >>> y_true = np.array([0.0, 0.0, 1.0, 1.0])
/// >>> cal = TemperatureScalingCalibrator(max_iter=100)
/// >>> cal.fit(y_prob, y_true)
/// >>> calibrated = cal.transform(y_prob)
/// >>> print(f"Temperature: {cal.temperature_}")
#[pyclass(name = "TemperatureScalingCalibrator", module = "ferroml.calibration")]
pub struct PyTemperatureScalingCalibrator {
    inner: TemperatureScalingCalibrator,
}

#[pymethods]
impl PyTemperatureScalingCalibrator {
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(<ferroml_core::models::calibration::TemperatureScalingCalibrator as HasModelCard>::model_card())
    }

    /// Create a new TemperatureScalingCalibrator.
    ///
    /// Parameters
    /// ----------
    /// max_iter : int, optional (default=100)
    ///     Maximum number of optimization iterations.
    /// learning_rate : float, optional (default=0.01)
    ///     Learning rate for gradient descent optimization.
    ///
    /// Returns
    /// -------
    /// TemperatureScalingCalibrator
    ///     A new model instance.
    #[new]
    #[pyo3(signature = (max_iter=100, learning_rate=0.01))]
    fn new(max_iter: usize, learning_rate: f64) -> Self {
        Self {
            inner: TemperatureScalingCalibrator::new()
                .with_max_iter(max_iter)
                .with_learning_rate(learning_rate),
        }
    }

    /// Fit the calibrator on predicted probabilities and true labels.
    ///
    /// Parameters
    /// ----------
    /// y_prob : array-like of shape (n_samples, n_classes)
    ///     Uncalibrated probability predictions.
    /// y_true : array-like of shape (n_samples,)
    ///     True class labels (0, 1, 2, ..., n_classes-1).
    ///
    /// Returns
    /// -------
    /// self : TemperatureScalingCalibrator
    ///     Fitted calibrator.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        y_prob: PyReadonlyArray2<'py, f64>,
        y_true: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let y_prob_arr = to_owned_array_2d(y_prob);
        let y_true_arr = py_array_to_f64_1d(py, y_true)?;

        MulticlassCalibrator::fit(&mut slf.inner, &y_prob_arr, &y_true_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Transform uncalibrated probabilities to calibrated ones.
    ///
    /// Parameters
    /// ----------
    /// y_prob : array-like of shape (n_samples, n_classes)
    ///     Uncalibrated probability predictions.
    ///
    /// Returns
    /// -------
    /// calibrated : ndarray of shape (n_samples, n_classes)
    ///     Calibrated probabilities.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        y_prob: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let y_prob_arr = to_owned_array_2d(y_prob);

        let calibrated = self
            .inner
            .transform(&y_prob_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(calibrated.into_pyarray(py))
    }

    /// The learned temperature parameter.
    #[getter]
    fn temperature_(&self) -> PyResult<f64> {
        self.inner.temperature().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Calibrator is not fitted yet. Call fit() first.",
            )
        })
    }

    /// Serialize for pickle.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Py<pyo3::types::PyBytes>> {
        crate::pickle::getstate(py, &self.inner)
    }
    /// Deserialize for pickle.
    pub fn __setstate__(&mut self, state: &pyo3::Bound<'_, pyo3::types::PyBytes>) -> PyResult<()> {
        self.inner = crate::pickle::setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        match self.inner.temperature() {
            Some(t) => format!("TemperatureScalingCalibrator(temperature={:.4})", t),
            None => "TemperatureScalingCalibrator(unfitted)".to_string(),
        }
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the calibration submodule.
pub fn register_calibration_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let cal_module = PyModule::new(parent_module.py(), "calibration")?;

    cal_module.add_class::<PyTemperatureScalingCalibrator>()?;

    parent_module.add_submodule(&cal_module)?;

    Ok(())
}
