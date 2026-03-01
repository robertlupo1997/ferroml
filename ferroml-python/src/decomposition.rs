//! Python bindings for FerroML decomposition models
//!
//! This module provides Python wrappers for:
//! - PCA (Principal Component Analysis)
//! - IncrementalPCA (Memory-efficient PCA for large datasets)
//! - TruncatedSVD (for sparse data / LSA)
//! - LDA (Linear Discriminant Analysis)
//! - FactorAnalysis (latent factor model)
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays, a copy is made.
//! Output arrays use `into_pyarray` to transfer ownership to Python without copying data.

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use crate::pickle::{getstate, setstate};
use ferroml_core::decomposition::{FactorAnalysis, IncrementalPCA, TruncatedSVD, LDA, PCA};
use ferroml_core::preprocessing::Transformer;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// =============================================================================
// PCA
// =============================================================================

/// Principal Component Analysis (PCA).
///
/// Linear dimensionality reduction using SVD to project data to a lower
/// dimensional space while maximizing variance.
///
/// Parameters
/// ----------
/// n_components : int, optional
///     Number of components to keep. If not set, all components are kept.
/// whiten : bool, optional (default=False)
///     When True, the components are divided by singular values to ensure
///     uncorrelated outputs with unit component-wise variances.
///
/// Attributes
/// ----------
/// components_ : ndarray of shape (n_components, n_features)
///     Principal axes in feature space.
/// explained_variance_ratio_ : ndarray of shape (n_components,)
///     Percentage of variance explained by each component.
///
/// Examples
/// --------
/// >>> from ferroml.decomposition import PCA
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 5)
/// >>> pca = PCA(n_components=3)
/// >>> X_t = pca.fit_transform(X)
/// >>> X_t.shape
/// (100, 3)
#[pyclass(name = "PCA", module = "ferroml.decomposition")]
pub struct PyPCA {
    inner: PCA,
}

#[pymethods]
impl PyPCA {
    #[new]
    #[pyo3(signature = (n_components=None, whiten=false))]
    fn new(n_components: Option<usize>, whiten: bool) -> Self {
        let mut pca = PCA::new().with_whiten(whiten);
        if let Some(n) = n_components {
            pca = pca.with_n_components(n);
        }
        Self { inner: pca }
    }

    /// Fit the PCA model to the data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Transform data using the fitted PCA model.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Apply the inverse transformation.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .inverse_transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Get the principal components.
    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let components = self
            .inner
            .components()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(components.clone().into_pyarray(py))
    }

    /// Get the explained variance ratio.
    #[getter]
    fn explained_variance_ratio_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let ratio = self
            .inner
            .explained_variance_ratio()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(ratio.clone().into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "PCA()".to_string()
    }
}

// =============================================================================
// IncrementalPCA
// =============================================================================

/// Incremental Principal Component Analysis.
///
/// Memory-efficient PCA for large datasets using batch processing.
///
/// Parameters
/// ----------
/// n_components : int, optional
///     Number of components to keep.
/// whiten : bool, optional (default=False)
///     When True, components are scaled to unit variance.
/// batch_size : int, optional
///     Size of batches for partial_fit.
#[pyclass(name = "IncrementalPCA", module = "ferroml.decomposition")]
pub struct PyIncrementalPCA {
    inner: IncrementalPCA,
}

#[pymethods]
impl PyIncrementalPCA {
    #[new]
    #[pyo3(signature = (n_components=None, whiten=false, batch_size=None))]
    fn new(n_components: Option<usize>, whiten: bool, batch_size: Option<usize>) -> Self {
        let mut ipca = IncrementalPCA::new().with_whiten(whiten);
        if let Some(n) = n_components {
            ipca = ipca.with_n_components(n);
        }
        if let Some(bs) = batch_size {
            ipca = ipca.with_batch_size(bs);
        }
        Self { inner: ipca }
    }

    /// Fit the model to the data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Transform data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
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
        "IncrementalPCA()".to_string()
    }
}

// =============================================================================
// TruncatedSVD
// =============================================================================

/// Truncated Singular Value Decomposition (TruncatedSVD).
///
/// Dimensionality reduction using truncated SVD. Unlike PCA, does not center
/// the data, making it suitable for sparse matrices. Also known as LSA.
///
/// Parameters
/// ----------
/// n_components : int, optional (default=2)
///     Number of components to keep.
/// n_iter : int, optional (default=5)
///     Number of iterations for randomized SVD.
/// random_state : int, optional
///     Random seed for reproducibility.
#[pyclass(name = "TruncatedSVD", module = "ferroml.decomposition")]
pub struct PyTruncatedSVD {
    inner: TruncatedSVD,
}

#[pymethods]
impl PyTruncatedSVD {
    #[new]
    #[pyo3(signature = (n_components=2, n_iter=5, random_state=None))]
    fn new(n_components: usize, n_iter: usize, random_state: Option<u64>) -> Self {
        let mut svd = TruncatedSVD::new()
            .with_n_components(n_components)
            .with_n_iter(n_iter);
        if let Some(seed) = random_state {
            svd = svd.with_random_state(seed);
        }
        Self { inner: svd }
    }

    /// Fit the model to the data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Transform data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Get the explained variance ratio.
    #[getter]
    fn explained_variance_ratio_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let ratio = self
            .inner
            .explained_variance_ratio()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(ratio.clone().into_pyarray(py))
    }

    /// Get the components.
    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let components = self
            .inner
            .components()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(components.clone().into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "TruncatedSVD()".to_string()
    }
}

// =============================================================================
// LDA (Linear Discriminant Analysis)
// =============================================================================

/// Linear Discriminant Analysis (LDA).
///
/// Supervised dimensionality reduction that maximizes class separability.
/// Finds linear combinations of features that maximize the ratio of
/// between-class to within-class variance.
///
/// Parameters
/// ----------
/// n_components : int, optional
///     Number of components. Max is n_classes - 1.
/// shrinkage : float, optional
///     Shrinkage parameter for regularization (0 to 1).
/// tol : float, optional (default=1e-4)
///     Convergence tolerance.
#[pyclass(name = "LDA", module = "ferroml.decomposition")]
pub struct PyLDA {
    inner: LDA,
}

#[pymethods]
impl PyLDA {
    #[new]
    #[pyo3(signature = (n_components=None, shrinkage=None, tol=1e-4))]
    fn new(n_components: Option<usize>, shrinkage: Option<f64>, tol: f64) -> Self {
        let mut lda = LDA::new().with_tol(tol);
        if let Some(n) = n_components {
            lda = lda.with_n_components(n);
        }
        if let Some(s) = shrinkage {
            lda = lda.with_shrinkage(s);
        }
        Self { inner: lda }
    }

    /// Fit the LDA model with class labels.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target class labels.
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

    /// Transform data using the fitted LDA model.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
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
        "LDA()".to_string()
    }
}

// =============================================================================
// FactorAnalysis
// =============================================================================

/// Factor Analysis.
///
/// A statistical model relating observed variables to latent factors.
/// Separates common variance from unique variance and supports rotation
/// methods for improved interpretability.
///
/// Parameters
/// ----------
/// n_factors : int, optional
///     Number of latent factors.
/// rotation : str, optional (default="none")
///     Rotation method: "none", "varimax", "quartimax", "promax".
/// tol : float, optional (default=1e-3)
///     Convergence tolerance.
/// max_iter : int, optional (default=1000)
///     Maximum number of EM iterations.
/// random_state : int, optional
///     Random seed.
#[pyclass(name = "FactorAnalysis", module = "ferroml.decomposition")]
pub struct PyFactorAnalysis {
    inner: FactorAnalysis,
}

#[pymethods]
impl PyFactorAnalysis {
    #[new]
    #[pyo3(signature = (n_factors=None, rotation="none", tol=1e-3, max_iter=1000, random_state=None))]
    fn new(
        n_factors: Option<usize>,
        rotation: &str,
        tol: f64,
        max_iter: usize,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        use ferroml_core::decomposition::Rotation;

        let rot = match rotation {
            "none" => Rotation::None,
            "varimax" => Rotation::Varimax,
            "quartimax" => Rotation::Quartimax,
            "promax" => Rotation::Promax,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown rotation: '{}'. Use 'none', 'varimax', 'quartimax', or 'promax'.",
                    rotation
                )));
            }
        };

        let mut fa = FactorAnalysis::new()
            .with_rotation(rot)
            .with_tol(tol)
            .with_max_iter(max_iter);
        if let Some(n) = n_factors {
            fa = fa.with_n_factors(n);
        }
        if let Some(seed) = random_state {
            fa = fa.with_random_state(seed);
        }
        Self { inner: fa }.into_ok()
    }

    /// Fit the model to the data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Transform data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
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
        "FactorAnalysis()".to_string()
    }
}

// Helper trait to convert Self into PyResult<Self>
trait IntoOk: Sized {
    fn into_ok(self) -> PyResult<Self> {
        Ok(self)
    }
}
impl IntoOk for PyFactorAnalysis {}

// =============================================================================
// Module registration
// =============================================================================

/// Register the decomposition submodule.
pub fn register_decomposition_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "decomposition")?;

    m.add_class::<PyPCA>()?;
    m.add_class::<PyIncrementalPCA>()?;
    m.add_class::<PyTruncatedSVD>()?;
    m.add_class::<PyLDA>()?;
    m.add_class::<PyFactorAnalysis>()?;

    parent_module.add_submodule(&m)?;

    Ok(())
}
