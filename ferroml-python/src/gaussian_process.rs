//! Python bindings for Gaussian Process models.
//!
//! Provides:
//! - GaussianProcessRegressor
//! - GaussianProcessClassifier
//! - Kernel classes: RBF, Matern, ConstantKernel, WhiteKernel

use crate::array_utils::{
    check_array1_finite, check_array_finite, py_array_to_f64_1d, to_owned_array_1d,
    to_owned_array_2d,
};
use ferroml_core::model_card::HasModelCard;
use ferroml_core::models::gaussian_process::{
    self, GaussianProcessClassifier, GaussianProcessRegressor, InducingPointMethod, Kernel,
    SVGPRegressor, SparseApproximation, SparseGPClassifier, SparseGPRegressor,
};
use ferroml_core::models::Model;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::model_card::PyModelCard;

// =============================================================================
// Kernel wrappers
// =============================================================================

/// RBF (Radial Basis Function) kernel.
///
/// K(x, x') = exp(-||x - x'||^2 / (2 * length_scale^2))
///
/// The most commonly used kernel for Gaussian processes. Produces
/// infinitely differentiable (smooth) functions.
///
/// Parameters
/// ----------
/// length_scale : float, optional (default=1.0)
///     Length scale of the kernel. Valid range: (0, inf).
///     Larger values produce smoother functions.
#[pyclass(name = "RBF", module = "ferroml.gaussian_process")]
#[derive(Clone)]
pub struct PyRBF {
    length_scale: f64,
}

#[pymethods]
impl PyRBF {
    #[new]
    #[pyo3(signature = (length_scale=1.0))]
    fn new(length_scale: f64) -> Self {
        Self { length_scale }
    }
}

/// Matern kernel.
///
/// Supports nu = 0.5, 1.5, 2.5 only.
///
/// Parameters
/// ----------
/// length_scale : float, optional (default=1.0)
///     Length scale of the kernel.
/// nu : float, optional (default=1.5)
///     Smoothness parameter. Must be 0.5, 1.5, or 2.5.
#[pyclass(name = "Matern", module = "ferroml.gaussian_process")]
#[derive(Clone)]
pub struct PyMatern {
    length_scale: f64,
    nu: f64,
}

#[pymethods]
impl PyMatern {
    #[new]
    #[pyo3(signature = (length_scale=1.0, nu=1.5))]
    fn new(length_scale: f64, nu: f64) -> PyResult<Self> {
        if !((nu - 0.5).abs() < 1e-10 || (nu - 1.5).abs() < 1e-10 || (nu - 2.5).abs() < 1e-10) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Matern kernel only supports nu = 0.5, 1.5, or 2.5",
            ));
        }
        Ok(Self { length_scale, nu })
    }
}

/// Constant kernel: K(x, x') = constant.
///
/// Parameters
/// ----------
/// constant : float, optional (default=1.0)
///     The constant value of the kernel.
#[pyclass(name = "ConstantKernel", module = "ferroml.gaussian_process")]
#[derive(Clone)]
pub struct PyConstantKernel {
    constant: f64,
}

#[pymethods]
impl PyConstantKernel {
    #[new]
    #[pyo3(signature = (constant=1.0))]
    fn new(constant: f64) -> Self {
        Self { constant }
    }
}

/// White noise kernel: K(x, x') = noise_level * delta(x, x').
///
/// Parameters
/// ----------
/// noise_level : float, optional (default=1.0)
///     The noise level.
#[pyclass(name = "WhiteKernel", module = "ferroml.gaussian_process")]
#[derive(Clone)]
pub struct PyWhiteKernel {
    noise_level: f64,
}

#[pymethods]
impl PyWhiteKernel {
    #[new]
    #[pyo3(signature = (noise_level=1.0))]
    fn new(noise_level: f64) -> Self {
        Self { noise_level }
    }
}

// =============================================================================
// Helper to convert Python kernel to Rust kernel
// =============================================================================

fn parse_kernel(kernel: Option<&Bound<'_, PyAny>>) -> PyResult<Box<dyn Kernel>> {
    match kernel {
        None => Ok(Box::new(gaussian_process::RBF::new(1.0))),
        Some(obj) => {
            if let Ok(rbf) = obj.extract::<PyRef<PyRBF>>() {
                Ok(Box::new(gaussian_process::RBF::new(rbf.length_scale)))
            } else if let Ok(m) = obj.extract::<PyRef<PyMatern>>() {
                Ok(Box::new(gaussian_process::Matern::new(
                    m.length_scale,
                    m.nu,
                )))
            } else if let Ok(c) = obj.extract::<PyRef<PyConstantKernel>>() {
                Ok(Box::new(gaussian_process::ConstantKernel::new(c.constant)))
            } else if let Ok(w) = obj.extract::<PyRef<PyWhiteKernel>>() {
                Ok(Box::new(gaussian_process::WhiteKernel::new(w.noise_level)))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Unknown kernel type. Supported: RBF, Matern, ConstantKernel, WhiteKernel",
                ))
            }
        }
    }
}

// =============================================================================
// GaussianProcessRegressor
// =============================================================================

/// Gaussian Process Regressor.
///
/// Provides exact GP regression with uncertainty estimation.
///
/// Parameters
/// ----------
/// kernel : kernel object, optional (default=RBF(1.0))
///     The kernel specifying the covariance function.
/// alpha : float, optional (default=1e-10)
///     Noise / regularization added to the diagonal of the kernel matrix.
/// normalize_y : bool, optional (default=False)
///     Whether to normalize the target values.
///
/// Attributes
/// ----------
/// log_marginal_likelihood_ : float
///     Log marginal likelihood of the fitted model.
///
/// Examples
/// --------
/// >>> from ferroml.gaussian_process import GaussianProcessRegressor, RBF
/// >>> import numpy as np
/// >>> X = np.linspace(0, 5, 20).reshape(-1, 1)
/// >>> y = np.sin(X).ravel()
/// >>> gpr = GaussianProcessRegressor(kernel=RBF(1.0))
/// >>> gpr.fit(X, y)
/// >>> mean, std = gpr.predict_with_std(X)
///
/// Notes
/// -----
/// - GP models do not support pickle serialization due to kernel trait objects.
/// - Computational complexity is O(n^3) for exact inference. For large datasets,
///   use SparseGPRegressor or SVGPRegressor instead.
#[pyclass(name = "GaussianProcessRegressor", module = "ferroml.gaussian_process")]
pub struct PyGaussianProcessRegressor {
    inner: GaussianProcessRegressor,
}

#[pymethods]
impl PyGaussianProcessRegressor {
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(<ferroml_core::models::gaussian_process::GaussianProcessRegressor as HasModelCard>::model_card())
    }

    #[new]
    #[pyo3(signature = (kernel=None, alpha=1e-10, normalize_y=false))]
    fn new(kernel: Option<&Bound<'_, PyAny>>, alpha: f64, normalize_y: bool) -> PyResult<Self> {
        let k = parse_kernel(kernel)?;
        let inner = GaussianProcessRegressor::new(k)
            .with_alpha(alpha)
            .with_normalize_y(normalize_y);
        Ok(Self { inner })
    }

    /// Fit the model to training data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Predict target values.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    /// Predict with uncertainty estimates.
    ///
    /// Returns
    /// -------
    /// mean : ndarray of shape (n_samples,)
    ///     Predicted mean values.
    /// std : ndarray of shape (n_samples,)
    ///     Predicted standard deviations.
    #[allow(clippy::type_complexity)]
    fn predict_with_std<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let (mean, std) = self
            .inner
            .predict_with_std(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok((mean.into_pyarray(py), std.into_pyarray(py)))
    }

    /// Evaluate the model on test data.
    ///
    /// Returns R² for regressors.
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        self.inner
            .score(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    /// Log marginal likelihood of the fitted model.
    #[getter]
    fn log_marginal_likelihood_(&self) -> PyResult<f64> {
        self.inner
            .log_marginal_likelihood()
            .map_err(crate::errors::ferro_to_pyerr)
    }
}

// =============================================================================
// GaussianProcessClassifier
// =============================================================================

/// Gaussian Process Classifier (binary, Laplace approximation).
///
/// Uses Laplace approximation for posterior inference with support for
/// probability estimates via predict_proba().
///
/// Parameters
/// ----------
/// kernel : kernel object, optional (default=RBF(1.0))
///     The kernel specifying the covariance function.
/// max_iter : int, optional (default=50)
///     Maximum number of Newton iterations for Laplace approximation.
///
/// Examples
/// --------
/// >>> from ferroml.gaussian_process import GaussianProcessClassifier, RBF
/// >>> import numpy as np
/// >>> X = np.array([[1, 1], [2, 1], [5, 5], [6, 5]])
/// >>> y = np.array([0.0, 0.0, 1.0, 1.0])
/// >>> gpc = GaussianProcessClassifier(kernel=RBF(1.0))
/// >>> gpc.fit(X, y)
/// >>> preds = gpc.predict(X)
///
/// Notes
/// -----
/// - GP models do not support pickle serialization due to kernel trait objects.
/// - Binary classification only. For multiclass, use one-vs-rest.
#[pyclass(
    name = "GaussianProcessClassifier",
    module = "ferroml.gaussian_process"
)]
pub struct PyGaussianProcessClassifier {
    inner: GaussianProcessClassifier,
}

#[pymethods]
impl PyGaussianProcessClassifier {
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(<ferroml_core::models::gaussian_process::GaussianProcessClassifier as HasModelCard>::model_card())
    }

    #[new]
    #[pyo3(signature = (kernel=None, max_iter=50))]
    fn new(kernel: Option<&Bound<'_, PyAny>>, max_iter: usize) -> PyResult<Self> {
        let k = parse_kernel(kernel)?;
        let inner = GaussianProcessClassifier::new(k).with_max_iter(max_iter);
        Ok(Self { inner })
    }

    /// Fit the model to training data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Predict class labels.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    /// Predict class probabilities.
    ///
    /// Returns
    /// -------
    /// probas : ndarray of shape (n_samples, 2)
    ///     Class probabilities [P(class=0), P(class=1)].
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
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
    /// log_probas : ndarray of shape (n_samples, 2)
    ///     Log-probability of each class.
    fn predict_log_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(probas.mapv(|p| p.max(1e-15).ln()).into_pyarray(py))
    }

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers.
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        self.inner
            .score(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }
}

// =============================================================================
// Helper to parse inducing point method from string
// =============================================================================

fn parse_inducing_method(method: &str, seed: Option<u64>) -> PyResult<InducingPointMethod> {
    match method {
        "random" => Ok(InducingPointMethod::RandomSubset { seed }),
        "kmeans" => Ok(InducingPointMethod::KMeans {
            max_iter: 100,
            seed,
        }),
        "greedy_variance" => Ok(InducingPointMethod::GreedyVariance { seed }),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown inducing method '{}'. Supported: 'random', 'kmeans', 'greedy_variance'",
            method
        ))),
    }
}

fn parse_approximation(approx: &str) -> PyResult<SparseApproximation> {
    match approx {
        "fitc" => Ok(SparseApproximation::FITC),
        "vfe" => Ok(SparseApproximation::VFE),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown approximation '{}'. Supported: 'fitc', 'vfe'",
            approx
        ))),
    }
}

// =============================================================================
// SparseGPRegressor
// =============================================================================

/// Sparse Gaussian Process Regressor using inducing points.
///
/// Scales GP regression to large datasets by using a set of inducing
/// points to approximate the full GP posterior. Supports FITC and VFE
/// approximations.
///
/// Parameters
/// ----------
/// kernel : kernel object, optional (default=RBF(1.0))
///     The kernel specifying the covariance function.
/// alpha : float, optional (default=0.01)
///     Noise / regularization added to the diagonal.
/// n_inducing : int, optional (default=100)
///     Number of inducing points.
/// inducing_method : str, optional (default="kmeans")
///     How to select inducing points: "random", "kmeans", "greedy_variance".
/// approximation : str, optional (default="fitc")
///     Sparse approximation method: "fitc" or "vfe".
/// normalize_y : bool, optional (default=False)
///     Whether to normalize the target values.
///
/// Attributes
/// ----------
/// log_marginal_likelihood_ : float
///     Log marginal likelihood of the fitted model.
/// inducing_points_ : ndarray of shape (n_inducing, n_features)
///     The selected inducing point locations.
///
/// Examples
/// --------
/// >>> from ferroml.gaussian_process import SparseGPRegressor, RBF
/// >>> import numpy as np
/// >>> X = np.random.randn(500, 2)
/// >>> y = np.sin(X[:, 0]) + 0.1 * np.random.randn(500)
/// >>> sgpr = SparseGPRegressor(kernel=RBF(1.0), n_inducing=50)
/// >>> sgpr.fit(X, y)
/// >>> mean, std = sgpr.predict_with_std(X[:5])
///
/// Notes
/// -----
/// - GP models do not support pickle serialization due to kernel trait objects.
/// - FITC is faster but less accurate; VFE gives a tighter lower bound.
#[pyclass(name = "SparseGPRegressor", module = "ferroml.gaussian_process")]
pub struct PySparseGPRegressor {
    inner: SparseGPRegressor,
}

#[pymethods]
impl PySparseGPRegressor {
    #[new]
    #[pyo3(signature = (kernel=None, alpha=0.01, n_inducing=100, inducing_method="kmeans", approximation="fitc", normalize_y=false))]
    fn new(
        kernel: Option<&Bound<'_, PyAny>>,
        alpha: f64,
        n_inducing: usize,
        inducing_method: &str,
        approximation: &str,
        normalize_y: bool,
    ) -> PyResult<Self> {
        let k = parse_kernel(kernel)?;
        let method = parse_inducing_method(inducing_method, Some(42))?;
        let approx = parse_approximation(approximation)?;
        let inner = SparseGPRegressor::new(k)
            .with_alpha(alpha)
            .with_n_inducing(n_inducing)
            .with_inducing_method(method)
            .with_approximation(approx)
            .with_normalize_y(normalize_y);
        Ok(Self { inner })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    #[allow(clippy::type_complexity)]
    fn predict_with_std<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let (mean, std) = self
            .inner
            .predict_with_std(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok((mean.into_pyarray(py), std.into_pyarray(py)))
    }

    /// Evaluate the model on test data.
    ///
    /// Returns R² for regressors.
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        self.inner
            .score(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    #[getter]
    fn log_marginal_likelihood_(&self) -> PyResult<f64> {
        self.inner
            .log_marginal_likelihood()
            .map_err(crate::errors::ferro_to_pyerr)
    }

    #[getter]
    fn inducing_points_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let z = self
            .inner
            .inducing_points()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(z.clone().into_pyarray(py))
    }
}

// =============================================================================
// SparseGPClassifier
// =============================================================================

/// Sparse Gaussian Process Classifier (binary, FITC + Laplace).
///
/// Scales GP classification to larger datasets using inducing points
/// with Laplace approximation for posterior inference.
///
/// Parameters
/// ----------
/// kernel : kernel object, optional (default=RBF(1.0))
///     The kernel specifying the covariance function.
/// n_inducing : int, optional (default=100)
///     Number of inducing points.
/// inducing_method : str, optional (default="kmeans")
///     How to select inducing points: "random", "kmeans", "greedy_variance".
/// max_iter : int, optional (default=50)
///     Maximum Newton iterations for Laplace approximation.
///
/// Attributes
/// ----------
/// inducing_points_ : ndarray of shape (n_inducing, n_features)
///     The selected inducing point locations.
///
/// Examples
/// --------
/// >>> from ferroml.gaussian_process import SparseGPClassifier, RBF
/// >>> import numpy as np
/// >>> X = np.random.randn(200, 2)
/// >>> y = (X[:, 0] + X[:, 1] > 0).astype(float)
/// >>> sgpc = SparseGPClassifier(kernel=RBF(1.0), n_inducing=30)
/// >>> sgpc.fit(X, y)
/// >>> preds = sgpc.predict(X[:5])
///
/// Notes
/// -----
/// - GP models do not support pickle serialization due to kernel trait objects.
/// - Binary classification only.
#[pyclass(name = "SparseGPClassifier", module = "ferroml.gaussian_process")]
pub struct PySparseGPClassifier {
    inner: SparseGPClassifier,
}

#[pymethods]
impl PySparseGPClassifier {
    #[new]
    #[pyo3(signature = (kernel=None, n_inducing=100, inducing_method="kmeans", max_iter=50))]
    fn new(
        kernel: Option<&Bound<'_, PyAny>>,
        n_inducing: usize,
        inducing_method: &str,
        max_iter: usize,
    ) -> PyResult<Self> {
        let k = parse_kernel(kernel)?;
        let method = parse_inducing_method(inducing_method, Some(42))?;
        let inner = SparseGPClassifier::new(k)
            .with_n_inducing(n_inducing)
            .with_inducing_method(method)
            .with_max_iter(max_iter);
        Ok(Self { inner })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
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
    /// log_probas : ndarray of shape (n_samples, 2)
    ///     Log-probability of each class.
    fn predict_log_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(probas.mapv(|p| p.max(1e-15).ln()).into_pyarray(py))
    }

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers.
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        self.inner
            .score(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    #[getter]
    fn inducing_points_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let z = self
            .inner
            .inducing_points()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(z.clone().into_pyarray(py))
    }
}

// =============================================================================
// SVGPRegressor
// =============================================================================

/// Sparse Variational Gaussian Process Regressor for large datasets.
///
/// Uses stochastic variational inference with mini-batches for scalable
/// GP regression. Best suited for datasets with thousands or more samples.
///
/// Parameters
/// ----------
/// kernel : kernel object, optional (default=RBF(1.0))
///     The kernel specifying the covariance function.
/// noise_variance : float, optional (default=1.0)
///     Observation noise variance.
/// n_inducing : int, optional (default=100)
///     Number of inducing points.
/// inducing_method : str, optional (default="kmeans")
///     How to select inducing points: "random", "kmeans", "greedy_variance".
/// n_epochs : int, optional (default=100)
///     Number of training epochs.
/// batch_size : int, optional (default=256)
///     Mini-batch size for stochastic optimization.
/// learning_rate : float, optional (default=0.01)
///     Learning rate for variational parameter updates.
/// normalize_y : bool, optional (default=False)
///     Whether to normalize the target values.
///
/// Attributes
/// ----------
/// inducing_points_ : ndarray of shape (n_inducing, n_features)
///     The selected inducing point locations.
///
/// Examples
/// --------
/// >>> from ferroml.gaussian_process import SVGPRegressor, RBF
/// >>> import numpy as np
/// >>> X = np.random.randn(1000, 3)
/// >>> y = np.sin(X[:, 0]) + 0.1 * np.random.randn(1000)
/// >>> svgp = SVGPRegressor(kernel=RBF(1.0), n_inducing=50, n_epochs=50)
/// >>> svgp.fit(X, y)
/// >>> mean, std = svgp.predict_with_std(X[:5])
///
/// Notes
/// -----
/// - GP models do not support pickle serialization due to kernel trait objects.
/// - For best results, normalize your features before fitting.
#[pyclass(name = "SVGPRegressor", module = "ferroml.gaussian_process")]
pub struct PySVGPRegressor {
    inner: SVGPRegressor,
}

#[pymethods]
impl PySVGPRegressor {
    #[new]
    #[pyo3(signature = (kernel=None, noise_variance=1.0, n_inducing=100, inducing_method="kmeans", n_epochs=100, batch_size=256, learning_rate=0.01, normalize_y=false))]
    fn new(
        kernel: Option<&Bound<'_, PyAny>>,
        noise_variance: f64,
        n_inducing: usize,
        inducing_method: &str,
        n_epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        normalize_y: bool,
    ) -> PyResult<Self> {
        let k = parse_kernel(kernel)?;
        let method = parse_inducing_method(inducing_method, Some(42))?;
        let inner = SVGPRegressor::new(k)
            .with_noise_variance(noise_variance)
            .with_n_inducing(n_inducing)
            .with_inducing_method(method)
            .with_n_epochs(n_epochs)
            .with_batch_size(batch_size)
            .with_learning_rate(learning_rate)
            .with_normalize_y(normalize_y);
        Ok(Self { inner })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    #[allow(clippy::type_complexity)]
    fn predict_with_std<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let (mean, std) = self
            .inner
            .predict_with_std(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok((mean.into_pyarray(py), std.into_pyarray(py)))
    }

    /// Evaluate the model on test data.
    ///
    /// Returns R² for regressors.
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        self.inner
            .score(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    #[getter]
    fn inducing_points_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let z = self
            .inner
            .inducing_points()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(z.clone().into_pyarray(py))
    }
}

// =============================================================================
// Module registration
// =============================================================================

pub fn register_gaussian_process_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "gaussian_process")?;
    m.add_class::<PyRBF>()?;
    m.add_class::<PyMatern>()?;
    m.add_class::<PyConstantKernel>()?;
    m.add_class::<PyWhiteKernel>()?;
    m.add_class::<PyGaussianProcessRegressor>()?;
    m.add_class::<PyGaussianProcessClassifier>()?;
    m.add_class::<PySparseGPRegressor>()?;
    m.add_class::<PySparseGPClassifier>()?;
    m.add_class::<PySVGPRegressor>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
