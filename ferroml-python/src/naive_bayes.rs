//! Python bindings for FerroML Naive Bayes classifiers
//!
//! This module provides Python wrappers for:
//! - GaussianNB
//! - MultinomialNB
//! - BernoulliNB

use crate::array_utils::{py_array_to_f64_1d, to_owned_array_2d};
use ferroml_core::models::naive_bayes::{BernoulliNB, GaussianNB, MultinomialNB};
use ferroml_core::models::{Model, ProbabilisticModel};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// GaussianNB
// ---------------------------------------------------------------------------

/// Gaussian Naive Bayes classifier.
///
/// Implements the Gaussian Naive Bayes algorithm for classification with
/// continuous features assumed to follow a Gaussian distribution per class.
///
/// Parameters
/// ----------
/// var_smoothing : float, optional (default=1e-9)
///     Portion of the largest variance of all features that is added to
///     variances for calculation stability.
///
/// Example
/// -------
/// >>> from ferroml.naive_bayes import GaussianNB
/// >>> import numpy as np
/// >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
/// >>> y = np.array([0.0, 0.0, 1.0, 1.0])
/// >>> model = GaussianNB()
/// >>> model.fit(X, y)
/// >>> model.predict(X)
#[pyclass(name = "GaussianNB", module = "ferroml.naive_bayes")]
pub struct PyGaussianNB {
    inner: GaussianNB,
}

#[pymethods]
impl PyGaussianNB {
    /// Create a new Gaussian Naive Bayes classifier.
    #[new]
    #[pyo3(signature = (var_smoothing=1e-9))]
    fn new(var_smoothing: f64) -> Self {
        let inner = GaussianNB::new().with_var_smoothing(var_smoothing);
        Self { inner }
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target class labels. Can be integer or float array.
    ///
    /// Returns
    /// -------
    /// self : GaussianNB
    ///     Fitted estimator.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;

        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict class labels.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted class labels.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Predict class probabilities.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// probas : ndarray of shape (n_samples, n_classes)
    ///     Probability of each class.
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

    /// Get the unique class labels.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let classes = self
            .inner
            .classes()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(classes.into_pyarray(py))
    }

    /// Get the per-class mean for each feature (shape: n_classes x n_features).
    #[getter]
    fn theta_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let theta = self
            .inner
            .theta()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(theta.into_pyarray(py))
    }

    /// Get the per-class variance for each feature (shape: n_classes x n_features).
    #[getter]
    fn var_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let var = self
            .inner
            .var()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(var.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("GaussianNB(var_smoothing={})", self.inner.var_smoothing)
    }
}

// ---------------------------------------------------------------------------
// MultinomialNB
// ---------------------------------------------------------------------------

/// Multinomial Naive Bayes classifier.
///
/// Suitable for classification with discrete features (e.g., word counts for
/// text classification). The multinomial distribution normally requires integer
/// feature counts, but fractional counts (such as tf-idf) also work in practice.
///
/// Parameters
/// ----------
/// alpha : float, optional (default=1.0)
///     Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
/// fit_prior : bool, optional (default=True)
///     Whether to learn class prior probabilities. If False, a uniform prior
///     will be used.
///
/// Example
/// -------
/// >>> from ferroml.naive_bayes import MultinomialNB
/// >>> import numpy as np
/// >>> X = np.array([[5.0, 1.0, 0.0], [4.0, 2.0, 0.0], [0.0, 1.0, 5.0], [0.0, 0.0, 6.0]])
/// >>> y = np.array([0.0, 0.0, 1.0, 1.0])
/// >>> model = MultinomialNB()
/// >>> model.fit(X, y)
/// >>> model.predict(X)
#[pyclass(name = "MultinomialNB", module = "ferroml.naive_bayes")]
pub struct PyMultinomialNB {
    inner: MultinomialNB,
}

#[pymethods]
impl PyMultinomialNB {
    /// Create a new Multinomial Naive Bayes classifier.
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_prior=true))]
    fn new(alpha: f64, fit_prior: bool) -> Self {
        let inner = MultinomialNB::new()
            .with_alpha(alpha)
            .with_fit_prior(fit_prior);
        Self { inner }
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data (non-negative counts or frequencies).
    /// y : array-like of shape (n_samples,)
    ///     Target class labels. Can be integer or float array.
    ///
    /// Returns
    /// -------
    /// self : MultinomialNB
    ///     Fitted estimator.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;

        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict class labels.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted class labels.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Predict class probabilities.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// probas : ndarray of shape (n_samples, n_classes)
    ///     Probability of each class.
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

    /// Get the unique class labels.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let classes = self
            .inner
            .classes()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(classes.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "MultinomialNB(alpha={}, fit_prior={})",
            self.inner.alpha, self.inner.fit_prior
        )
    }
}

// ---------------------------------------------------------------------------
// BernoulliNB
// ---------------------------------------------------------------------------

/// Bernoulli Naive Bayes classifier.
///
/// Like MultinomialNB, this classifier is suitable for discrete data. The
/// difference is that while MultinomialNB works with occurrence counts,
/// BernoulliNB is designed for binary/boolean features.
///
/// Parameters
/// ----------
/// alpha : float, optional (default=1.0)
///     Additive (Laplace/Lidstone) smoothing parameter.
/// binarize : float, optional (default=0.0)
///     Threshold for binarizing features. Features with values above this
///     threshold are mapped to 1, others to 0. Set to None to assume input
///     is already binary.
/// fit_prior : bool, optional (default=True)
///     Whether to learn class prior probabilities. If False, a uniform prior
///     will be used.
///
/// Example
/// -------
/// >>> from ferroml.naive_bayes import BernoulliNB
/// >>> import numpy as np
/// >>> X = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
/// >>> y = np.array([0.0, 0.0, 1.0, 1.0])
/// >>> model = BernoulliNB()
/// >>> model.fit(X, y)
/// >>> model.predict(X)
#[pyclass(name = "BernoulliNB", module = "ferroml.naive_bayes")]
pub struct PyBernoulliNB {
    inner: BernoulliNB,
}

#[pymethods]
impl PyBernoulliNB {
    /// Create a new Bernoulli Naive Bayes classifier.
    #[new]
    #[pyo3(signature = (alpha=1.0, binarize=0.0, fit_prior=true))]
    fn new(alpha: f64, binarize: f64, fit_prior: bool) -> Self {
        let inner = BernoulliNB::new()
            .with_alpha(alpha)
            .with_binarize(Some(binarize))
            .with_fit_prior(fit_prior);
        Self { inner }
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data (binary or will be binarized).
    /// y : array-like of shape (n_samples,)
    ///     Target class labels. Can be integer or float array.
    ///
    /// Returns
    /// -------
    /// self : BernoulliNB
    ///     Fitted estimator.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;

        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict class labels.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted class labels.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Predict class probabilities.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// probas : ndarray of shape (n_samples, n_classes)
    ///     Probability of each class.
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

    /// Get the unique class labels.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let classes = self
            .inner
            .classes()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(classes.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "BernoulliNB(alpha={}, binarize={:?}, fit_prior={})",
            self.inner.alpha, self.inner.binarize, self.inner.fit_prior
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register_naive_bayes_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let nb_module = PyModule::new(parent_module.py(), "naive_bayes")?;
    nb_module.add_class::<PyGaussianNB>()?;
    nb_module.add_class::<PyMultinomialNB>()?;
    nb_module.add_class::<PyBernoulliNB>()?;
    parent_module.add_submodule(&nb_module)?;
    Ok(())
}
