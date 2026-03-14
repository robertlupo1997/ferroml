//! Python bindings for FerroML SVM models
//!
//! This module provides Python wrappers for:
//! - LinearSVC (Linear Support Vector Classification)
//! - LinearSVR (Linear Support Vector Regression)
//! - SVC (Support Vector Classification with kernel methods)
//! - SVR (Support Vector Regression with kernel methods)

use crate::array_utils::{py_array_to_f64_1d, to_owned_array_1d, to_owned_array_2d};
use ferroml_core::models::svm::{
    ClassWeight, Kernel, LinearSVC, LinearSVCLoss, LinearSVR, LinearSVRLoss, MulticlassStrategy,
    SVC, SVR,
};
use ferroml_core::models::{Model, ProbabilisticModel};
use ferroml_core::onnx::{OnnxConfig, OnnxExportable};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[cfg(feature = "sparse")]
use crate::sparse_utils::py_csr_to_ferro;
#[cfg(feature = "sparse")]
use ferroml_core::models::traits::SparseModel;

// =============================================================================
// LinearSVC
// =============================================================================

/// Linear Support Vector Classification.
///
/// Similar to SVC with a linear kernel, but implemented using coordinate descent
/// which is more efficient for large datasets.
///
/// Parameters
/// ----------
/// c : float, optional (default=1.0)
///     Regularization parameter. Larger values specify stronger regularization.
/// max_iter : int, optional (default=1000)
///     Maximum number of iterations.
/// tol : float, optional (default=1e-4)
///     Tolerance for stopping criterion.
#[pyclass(name = "LinearSVC", module = "ferroml.svm")]
pub struct PyLinearSVC {
    inner: LinearSVC,
    loss: String,
}

#[pymethods]
impl PyLinearSVC {
    #[new]
    #[pyo3(signature = (c=1.0, loss="squared_hinge", max_iter=1000, tol=1e-4, class_weight=None))]
    fn new(
        c: f64,
        loss: &str,
        max_iter: usize,
        tol: f64,
        class_weight: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let parsed_loss = match loss {
            "hinge" => LinearSVCLoss::Hinge,
            "squared_hinge" => LinearSVCLoss::SquaredHinge,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown loss '{}'. Supported: 'hinge', 'squared_hinge'",
                    other
                )));
            }
        };
        let cw = parse_class_weight(class_weight)?;

        Ok(Self {
            inner: LinearSVC::new()
                .with_c(c)
                .with_loss(parsed_loss)
                .with_max_iter(max_iter)
                .with_tol(tol)
                .with_class_weight(cw),
            loss: loss.to_string(),
        })
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training vectors.
    /// y : array-like of shape (n_samples,)
    ///     Target class labels.
    ///
    /// Returns
    /// -------
    /// self : LinearSVC
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

    /// Predict class labels for samples.
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

    /// Compute decision function values.
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Fit the model from a scipy.sparse matrix (native sparse, no densification).
    #[cfg(feature = "sparse")]
    fn fit_sparse<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let csr = py_csr_to_ferro(x)?;
        let y_arr = py_array_to_f64_1d(py, y)?;

        slf.inner
            .fit_sparse(&csr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict class labels using a scipy.sparse matrix (native sparse, no densification).
    #[cfg(feature = "sparse")]
    fn predict_sparse<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let csr = py_csr_to_ferro(x)?;

        let predictions = self
            .inner
            .predict_sparse(&csr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Export the fitted model to ONNX format.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Output file path (typically with .onnx extension).
    /// model_name : str, optional
    ///     Name for the model in the ONNX graph (default: "ferroml_model").
    /// input_name : str, optional
    ///     Name for the input tensor (default: "input").
    /// output_name : str, optional
    ///     Name for the output tensor (default: "output").
    #[pyo3(signature = (path, model_name=None, input_name=None, output_name=None))]
    fn export_onnx(
        &self,
        path: &str,
        model_name: Option<String>,
        input_name: Option<String>,
        output_name: Option<String>,
    ) -> PyResult<()> {
        let mut config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
        if let Some(name) = input_name {
            config = config.with_input_name(name);
        }
        if let Some(name) = output_name {
            config = config.with_output_name(name);
        }
        self.inner
            .export_onnx(path, &config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Export the fitted model to ONNX format as bytes.
    ///
    /// Parameters
    /// ----------
    /// model_name : str, optional
    ///     Name for the model in the ONNX graph (default: "ferroml_model").
    /// input_name : str, optional
    ///     Name for the input tensor (default: "input").
    /// output_name : str, optional
    ///     Name for the output tensor (default: "output").
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The ONNX model as bytes.
    #[pyo3(signature = (model_name=None, input_name=None, output_name=None))]
    fn to_onnx_bytes(
        &self,
        py: Python<'_>,
        model_name: Option<String>,
        input_name: Option<String>,
        output_name: Option<String>,
    ) -> PyResult<Py<PyBytes>> {
        let mut config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
        if let Some(name) = input_name {
            config = config.with_input_name(name);
        }
        if let Some(name) = output_name {
            config = config.with_output_name(name);
        }
        let bytes = self
            .inner
            .to_onnx(&config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearSVC(C={}, loss='{}', max_iter={})",
            self.inner.c, self.loss, self.inner.max_iter
        )
    }
}

// =============================================================================
// LinearSVR
// =============================================================================

/// Linear Support Vector Regression.
///
/// Similar to SVR with a linear kernel, but implemented using coordinate descent
/// which is more efficient for large datasets.
///
/// Parameters
/// ----------
/// c : float, optional (default=1.0)
///     Regularization parameter.
/// epsilon : float, optional (default=0.0)
///     Epsilon in the epsilon-SVR model. Specifies the epsilon-tube within which
///     no penalty is associated in the training loss function.
/// max_iter : int, optional (default=1000)
///     Maximum number of iterations.
/// tol : float, optional (default=1e-4)
///     Tolerance for stopping criterion.
#[pyclass(name = "LinearSVR", module = "ferroml.svm")]
pub struct PyLinearSVR {
    inner: LinearSVR,
    loss: String,
}

#[pymethods]
impl PyLinearSVR {
    #[new]
    #[pyo3(signature = (c=1.0, epsilon=0.0, loss="epsilon_insensitive", max_iter=1000, tol=1e-4))]
    fn new(c: f64, epsilon: f64, loss: &str, max_iter: usize, tol: f64) -> PyResult<Self> {
        let parsed_loss = match loss {
            "epsilon_insensitive" => LinearSVRLoss::EpsilonInsensitive,
            "squared_epsilon_insensitive" => LinearSVRLoss::SquaredEpsilonInsensitive,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown loss '{}'. Supported: 'epsilon_insensitive', 'squared_epsilon_insensitive'",
                    other
                )));
            }
        };

        Ok(Self {
            inner: LinearSVR::new()
                .with_c(c)
                .with_epsilon(epsilon)
                .with_loss(parsed_loss)
                .with_max_iter(max_iter)
                .with_tol(tol),
            loss: loss.to_string(),
        })
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training vectors.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : LinearSVR
    ///     Fitted estimator.
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
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted target values.
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

    /// Compute decision function values (same as predict for regression).
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Coefficient vector.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let weights = self.inner.weights().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model is not fitted yet. Call fit() first.",
            )
        })?;

        Ok(weights.clone().into_pyarray(py))
    }

    /// Intercept (bias) term.
    #[getter]
    fn intercept_(&self) -> f64 {
        self.inner.intercept()
    }

    /// Fit the model from a scipy.sparse matrix (native sparse, no densification).
    #[cfg(feature = "sparse")]
    fn fit_sparse<'py>(
        mut slf: PyRefMut<'py, Self>,
        _py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let csr = py_csr_to_ferro(x)?;
        let y_arr = to_owned_array_1d(y);

        slf.inner
            .fit_sparse(&csr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict using a scipy.sparse matrix (native sparse, no densification).
    #[cfg(feature = "sparse")]
    fn predict_sparse<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let csr = py_csr_to_ferro(x)?;

        let predictions = self
            .inner
            .predict_sparse(&csr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Export the fitted model to ONNX format.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Output file path (typically with .onnx extension).
    /// model_name : str, optional
    ///     Name for the model in the ONNX graph (default: "ferroml_model").
    /// input_name : str, optional
    ///     Name for the input tensor (default: "input").
    /// output_name : str, optional
    ///     Name for the output tensor (default: "output").
    #[pyo3(signature = (path, model_name=None, input_name=None, output_name=None))]
    fn export_onnx(
        &self,
        path: &str,
        model_name: Option<String>,
        input_name: Option<String>,
        output_name: Option<String>,
    ) -> PyResult<()> {
        let mut config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
        if let Some(name) = input_name {
            config = config.with_input_name(name);
        }
        if let Some(name) = output_name {
            config = config.with_output_name(name);
        }
        self.inner
            .export_onnx(path, &config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Export the fitted model to ONNX format as bytes.
    ///
    /// Parameters
    /// ----------
    /// model_name : str, optional
    ///     Name for the model in the ONNX graph (default: "ferroml_model").
    /// input_name : str, optional
    ///     Name for the input tensor (default: "input").
    /// output_name : str, optional
    ///     Name for the output tensor (default: "output").
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The ONNX model as bytes.
    #[pyo3(signature = (model_name=None, input_name=None, output_name=None))]
    fn to_onnx_bytes(
        &self,
        py: Python<'_>,
        model_name: Option<String>,
        input_name: Option<String>,
        output_name: Option<String>,
    ) -> PyResult<Py<PyBytes>> {
        let mut config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
        if let Some(name) = input_name {
            config = config.with_input_name(name);
        }
        if let Some(name) = output_name {
            config = config.with_output_name(name);
        }
        let bytes = self
            .inner
            .to_onnx(&config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearSVR(C={}, epsilon={}, loss='{}')",
            self.inner.c, self.inner.epsilon, self.loss
        )
    }
}

// =============================================================================
// Kernel parsing helper
// =============================================================================

/// Parse a kernel string and gamma (which can be "auto" or a float) into a Kernel enum.
fn parse_kernel(kernel: &str, gamma: f64, degree: u32, coef0: f64) -> PyResult<Kernel> {
    match kernel {
        "linear" => Ok(Kernel::Linear),
        "rbf" => {
            if gamma <= 0.0 {
                Ok(Kernel::rbf_auto())
            } else {
                Ok(Kernel::rbf(gamma))
            }
        }
        "poly" | "polynomial" => {
            if gamma <= 0.0 {
                Ok(Kernel::Polynomial {
                    gamma: 0.0,
                    coef0,
                    degree,
                })
            } else {
                Ok(Kernel::poly(degree, gamma, coef0))
            }
        }
        "sigmoid" => {
            if gamma <= 0.0 {
                Ok(Kernel::Sigmoid { gamma: 0.0, coef0 })
            } else {
                Ok(Kernel::sigmoid(gamma, coef0))
            }
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown kernel '{}'. Supported: 'linear', 'rbf', 'poly', 'sigmoid'",
            kernel
        ))),
    }
}

/// Parse a multiclass strategy string.
fn parse_multiclass_strategy(strategy: &str) -> PyResult<MulticlassStrategy> {
    match strategy {
        "ovo" | "one_vs_one" => Ok(MulticlassStrategy::OneVsOne),
        "ovr" | "one_vs_rest" => Ok(MulticlassStrategy::OneVsRest),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown multiclass strategy '{}'. Supported: 'ovo', 'ovr'",
            strategy
        ))),
    }
}

// =============================================================================
// Class weight parsing helper
// =============================================================================

/// Parse a class_weight argument: None, "balanced", or {class: weight} dict.
fn parse_class_weight(class_weight: Option<&Bound<'_, PyAny>>) -> PyResult<ClassWeight> {
    match class_weight {
        None => Ok(ClassWeight::Uniform),
        Some(ob) => {
            if let Ok(s) = ob.extract::<String>() {
                match s.as_str() {
                    "balanced" => Ok(ClassWeight::Balanced),
                    other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown class_weight '{}'. Supported: None, 'balanced', or dict",
                        other
                    ))),
                }
            } else if let Ok(dict) = ob.downcast::<pyo3::types::PyDict>() {
                let mut weights = Vec::new();
                for (key, value) in dict.iter() {
                    let k: f64 = key.extract().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "class_weight dict keys must be floats",
                        )
                    })?;
                    let v: f64 = value.extract().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "class_weight dict values must be floats",
                        )
                    })?;
                    weights.push((k, v));
                }
                Ok(ClassWeight::Custom(weights))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "class_weight must be None, 'balanced', or a dict {class: weight}",
                ))
            }
        }
    }
}

// =============================================================================
// SVC (Kernel Support Vector Classification)
// =============================================================================

/// Support Vector Classification with kernel methods.
///
/// Uses SMO (Sequential Minimal Optimization) to solve the dual problem
/// with support for RBF, polynomial, sigmoid, and linear kernels.
///
/// Parameters
/// ----------
/// kernel : str, optional (default="rbf")
///     Kernel type: "linear", "rbf", "poly", "sigmoid".
/// c : float, optional (default=1.0)
///     Regularization parameter.
/// gamma : float, optional (default=0.0)
///     Kernel coefficient. 0.0 means "auto" (1/n_features).
/// degree : int, optional (default=3)
///     Degree for polynomial kernel.
/// coef0 : float, optional (default=0.0)
///     Independent term in polynomial/sigmoid kernels.
/// tol : float, optional (default=1e-3)
///     Tolerance for stopping criterion.
/// max_iter : int, optional (default=1000)
///     Maximum number of SMO iterations.
/// probability : bool, optional (default=False)
///     Whether to enable probability estimates.
/// multiclass : str, optional (default="ovo")
///     Multiclass strategy: "ovo" (one-vs-one) or "ovr" (one-vs-rest).
/// class_weight : str, dict, or None, optional (default=None)
///     Class weight strategy: None (uniform), "balanced", or dict {class: weight}.
#[pyclass(name = "SVC", module = "ferroml.svm")]
pub struct PySVC {
    inner: SVC,
    kernel_str: String,
}

#[pymethods]
impl PySVC {
    #[new]
    #[pyo3(signature = (kernel="rbf", c=1.0, gamma=0.0, degree=3, coef0=0.0, tol=1e-3, max_iter=1000, probability=false, multiclass="ovo", class_weight=None))]
    fn new(
        kernel: &str,
        c: f64,
        gamma: f64,
        degree: u32,
        coef0: f64,
        tol: f64,
        max_iter: usize,
        probability: bool,
        multiclass: &str,
        class_weight: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let kernel_enum = parse_kernel(kernel, gamma, degree, coef0)?;
        let strategy = parse_multiclass_strategy(multiclass)?;
        let cw = parse_class_weight(class_weight)?;

        Ok(Self {
            inner: SVC::new()
                .with_c(c)
                .with_kernel(kernel_enum)
                .with_tol(tol)
                .with_max_iter(max_iter)
                .with_probability(probability)
                .with_multiclass_strategy(strategy)
                .with_class_weight(cw),
            kernel_str: kernel.to_string(),
        })
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training vectors.
    /// y : array-like of shape (n_samples,)
    ///     Target class labels.
    ///
    /// Returns
    /// -------
    /// self : SVC
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

    /// Predict class labels for samples.
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

    /// Predict class probabilities for samples.
    ///
    /// Only available if `probability=True` was set at construction time.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// probas : ndarray of shape (n_samples, n_classes)
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

    /// Predict log-probabilities for each class.
    ///
    /// Parameters
    /// ----------
    /// x : ndarray of shape (n_samples, n_features)
    ///     Input features.
    ///
    /// Returns
    /// -------
    /// log_probas : ndarray of shape (n_samples, n_classes)
    ///     Log-probability of each class.
    fn predict_log_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);

        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(probas.mapv(|p| p.max(1e-15).ln()).into_pyarray(py))
    }

    /// Compute decision function values.
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Number of support vectors per binary classifier.
    #[getter]
    fn n_support_vectors(&self) -> Vec<usize> {
        self.inner.n_support_vectors()
    }

    /// Unique class labels learned during fitting.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let classes = self.inner.classes().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model is not fitted yet. Call fit() first.",
            )
        })?;

        Ok(classes.clone().into_pyarray(py))
    }

    /// Export the fitted model to ONNX format.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Output file path (typically with .onnx extension).
    /// model_name : str, optional
    ///     Name for the model in the ONNX graph (default: "ferroml_model").
    /// input_name : str, optional
    ///     Name for the input tensor (default: "input").
    /// output_name : str, optional
    ///     Name for the output tensor (default: "output").
    #[pyo3(signature = (path, model_name=None, input_name=None, output_name=None))]
    fn export_onnx(
        &self,
        path: &str,
        model_name: Option<String>,
        input_name: Option<String>,
        output_name: Option<String>,
    ) -> PyResult<()> {
        let mut config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
        if let Some(name) = input_name {
            config = config.with_input_name(name);
        }
        if let Some(name) = output_name {
            config = config.with_output_name(name);
        }
        self.inner
            .export_onnx(path, &config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Export the fitted model to ONNX format as bytes.
    ///
    /// Parameters
    /// ----------
    /// model_name : str, optional
    ///     Name for the model in the ONNX graph (default: "ferroml_model").
    /// input_name : str, optional
    ///     Name for the input tensor (default: "input").
    /// output_name : str, optional
    ///     Name for the output tensor (default: "output").
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The ONNX model as bytes.
    #[pyo3(signature = (model_name=None, input_name=None, output_name=None))]
    fn to_onnx_bytes(
        &self,
        py: Python<'_>,
        model_name: Option<String>,
        input_name: Option<String>,
        output_name: Option<String>,
    ) -> PyResult<Py<PyBytes>> {
        let mut config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
        if let Some(name) = input_name {
            config = config.with_input_name(name);
        }
        if let Some(name) = output_name {
            config = config.with_output_name(name);
        }
        let bytes = self
            .inner
            .to_onnx(&config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "SVC(kernel='{}', C={}, probability={})",
            self.kernel_str, self.inner.c, self.inner.probability
        )
    }
}

// =============================================================================
// SVR (Kernel Support Vector Regression)
// =============================================================================

/// Support Vector Regression with kernel methods.
///
/// Uses SMO to find a function that deviates from targets by at most epsilon
/// while being as flat as possible.
///
/// Parameters
/// ----------
/// kernel : str, optional (default="rbf")
///     Kernel type: "linear", "rbf", "poly", "sigmoid".
/// c : float, optional (default=1.0)
///     Regularization parameter.
/// epsilon : float, optional (default=0.1)
///     Width of the epsilon-insensitive tube.
/// gamma : float, optional (default=0.0)
///     Kernel coefficient. 0.0 means "auto" (1/n_features).
/// degree : int, optional (default=3)
///     Degree for polynomial kernel.
/// coef0 : float, optional (default=0.0)
///     Independent term in polynomial/sigmoid kernels.
/// tol : float, optional (default=1e-3)
///     Tolerance for stopping criterion.
/// max_iter : int, optional (default=1000)
///     Maximum number of SMO iterations.
#[pyclass(name = "SVR", module = "ferroml.svm")]
pub struct PySVR {
    inner: SVR,
    kernel_str: String,
}

#[pymethods]
impl PySVR {
    #[new]
    #[pyo3(signature = (kernel="rbf", c=1.0, epsilon=0.1, gamma=0.0, degree=3, coef0=0.0, tol=1e-3, max_iter=1000))]
    fn new(
        kernel: &str,
        c: f64,
        epsilon: f64,
        gamma: f64,
        degree: u32,
        coef0: f64,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        let kernel_enum = parse_kernel(kernel, gamma, degree, coef0)?;

        Ok(Self {
            inner: SVR::new()
                .with_c(c)
                .with_epsilon(epsilon)
                .with_kernel(kernel_enum)
                .with_tol(tol)
                .with_max_iter(max_iter),
            kernel_str: kernel.to_string(),
        })
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training vectors.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : SVR
    ///     Fitted estimator.
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
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted target values.
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

    /// Compute decision function values (same as predict for regression).
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Number of support vectors.
    #[getter]
    fn n_support_vectors(&self) -> usize {
        self.inner.n_support_vectors()
    }

    /// Indices of support vectors in the training data.
    #[getter]
    fn support_indices_(&self) -> PyResult<Vec<usize>> {
        let indices = self.inner.support_indices().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model is not fitted yet. Call fit() first.",
            )
        })?;

        Ok(indices.to_vec())
    }

    /// Dual coefficients of support vectors.
    #[getter]
    fn dual_coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self.inner.dual_coef().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model is not fitted yet. Call fit() first.",
            )
        })?;

        Ok(coef.clone().into_pyarray(py))
    }

    /// Intercept (bias) term.
    #[getter]
    fn intercept_(&self) -> f64 {
        self.inner.intercept()
    }

    /// Export the fitted model to ONNX format.
    ///
    /// Parameters
    /// ----------
    /// path : str
    ///     Output file path (typically with .onnx extension).
    /// model_name : str, optional
    ///     Name for the model in the ONNX graph (default: "ferroml_model").
    /// input_name : str, optional
    ///     Name for the input tensor (default: "input").
    /// output_name : str, optional
    ///     Name for the output tensor (default: "output").
    #[pyo3(signature = (path, model_name=None, input_name=None, output_name=None))]
    fn export_onnx(
        &self,
        path: &str,
        model_name: Option<String>,
        input_name: Option<String>,
        output_name: Option<String>,
    ) -> PyResult<()> {
        let mut config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
        if let Some(name) = input_name {
            config = config.with_input_name(name);
        }
        if let Some(name) = output_name {
            config = config.with_output_name(name);
        }
        self.inner
            .export_onnx(path, &config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Export the fitted model to ONNX format as bytes.
    ///
    /// Parameters
    /// ----------
    /// model_name : str, optional
    ///     Name for the model in the ONNX graph (default: "ferroml_model").
    /// input_name : str, optional
    ///     Name for the input tensor (default: "input").
    /// output_name : str, optional
    ///     Name for the output tensor (default: "output").
    ///
    /// Returns
    /// -------
    /// bytes
    ///     The ONNX model as bytes.
    #[pyo3(signature = (model_name=None, input_name=None, output_name=None))]
    fn to_onnx_bytes(
        &self,
        py: Python<'_>,
        model_name: Option<String>,
        input_name: Option<String>,
        output_name: Option<String>,
    ) -> PyResult<Py<PyBytes>> {
        let mut config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
        if let Some(name) = input_name {
            config = config.with_input_name(name);
        }
        if let Some(name) = output_name {
            config = config.with_output_name(name);
        }
        let bytes = self
            .inner
            .to_onnx(&config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "SVR(kernel='{}', C={}, epsilon={})",
            self.kernel_str, self.inner.c, self.inner.epsilon
        )
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the svm submodule.
pub fn register_svm_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let svm_module = PyModule::new(parent_module.py(), "svm")?;

    svm_module.add_class::<PyLinearSVC>()?;
    svm_module.add_class::<PyLinearSVR>()?;
    svm_module.add_class::<PySVC>()?;
    svm_module.add_class::<PySVR>()?;

    parent_module.add_submodule(&svm_module)?;

    Ok(())
}
