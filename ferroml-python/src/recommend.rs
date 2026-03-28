//! Python bindings for the FerroML recommend API.
//!
//! Exposes the `recommend()` function and `Recommendation` class to Python.

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use ferroml_core::automl::ParamValue;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A single algorithm recommendation returned by `recommend()`.
#[pyclass(name = "Recommendation", module = "ferroml")]
pub struct PyRecommendation {
    inner: ferroml_core::recommend::Recommendation,
}

#[pymethods]
impl PyRecommendation {
    /// The recommended algorithm name (e.g., "RandomForestClassifier").
    #[getter]
    fn algorithm(&self) -> &str {
        &self.inner.algorithm
    }

    /// Human-readable explanation of why this algorithm is recommended.
    #[getter]
    fn reason(&self) -> &str {
        &self.inner.reason
    }

    /// Estimated fit time: "fast", "moderate", or "slow".
    #[getter]
    fn estimated_fit_time(&self) -> &str {
        &self.inner.estimated_fit_time
    }

    /// Suggested hyperparameters as a dictionary.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in &self.inner.params {
            match value {
                ParamValue::Int(v) => dict.set_item(key, v)?,
                ParamValue::Float(v) => dict.set_item(key, v)?,
                ParamValue::String(v) => dict.set_item(key, v)?,
                ParamValue::Bool(v) => dict.set_item(key, v)?,
            }
        }
        Ok(dict)
    }

    /// Internal ranking score (0.0 to 1.0, higher is better).
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    fn __repr__(&self) -> String {
        format!(
            "Recommendation(algorithm='{}', score={:.3}, estimated_fit_time='{}', reason='{}')",
            self.inner.algorithm,
            self.inner.score,
            self.inner.estimated_fit_time,
            self.inner.reason
        )
    }
}

/// Recommend ML algorithms for a dataset without fitting any models.
///
/// Analyzes dataset properties (size, sparsity, class balance, dimensionality)
/// and returns up to 5 ranked algorithm recommendations.
///
/// Parameters
/// ----------
/// x : numpy.ndarray
///     Feature matrix of shape (n_samples, n_features).
/// y : numpy.ndarray
///     Target vector of shape (n_samples,).
/// task : str, default="classification"
///     Task type: "classification" or "regression".
///
/// Returns
/// -------
/// list[Recommendation]
///     Up to 5 recommendations sorted by score descending.
///
/// Raises
/// ------
/// ValueError
///     If task is not "classification" or "regression", or shapes are invalid.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> import ferroml
/// >>> X = np.random.randn(200, 5)
/// >>> y = (X[:, 0] > 0).astype(float)
/// >>> recs = ferroml.recommend(X, y, task="classification")
/// >>> print(recs[0].algorithm)
/// RandomForestClassifier
#[pyfunction]
#[pyo3(signature = (x, y, task="classification"))]
pub fn recommend<'py>(
    _py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    task: &str,
) -> PyResult<Vec<PyRecommendation>> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);

    let recs = ferroml_core::recommend::recommend(&x_arr, &y_arr, task)
        .map_err(crate::errors::ferro_to_pyerr)?;

    Ok(recs
        .into_iter()
        .map(|r| PyRecommendation { inner: r })
        .collect())
}
