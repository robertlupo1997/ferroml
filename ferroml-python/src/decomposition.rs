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

use crate::array_utils::{
    check_array1_finite, check_array_finite, to_owned_array_1d, to_owned_array_2d,
};
use crate::pickle::{getstate, setstate};
use ferroml_core::decomposition::{
    FactorAnalysis, IncrementalPCA, LearningRate, TruncatedSVD, TsneInit, TsneMethod, TsneMetric,
    LDA, PCA, TSNE,
};
use ferroml_core::model_card::HasModelCard;
use ferroml_core::preprocessing::Transformer;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::model_card::PyModelCard;

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
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(<ferroml_core::decomposition::PCA as HasModelCard>::model_card())
    }

    /// Create a new PCA.
    ///
    /// Parameters
    /// ----------
    /// n_components : int, optional
    ///     Number of components to keep. If not set, all components are kept.
    /// whiten : bool, optional (default=False)
    ///     When True, components are divided by singular values for uncorrelated outputs.
    ///
    /// Returns
    /// -------
    /// PCA
    ///     A new model instance.
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
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform data using the fitted PCA model.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Apply the inverse transformation.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .inverse_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
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
/// n_components : int or None, optional (default=None)
///     Number of components to keep. If None, keeps min(n_samples, n_features).
/// whiten : bool, optional (default=False)
///     When True, components are scaled to unit variance.
/// batch_size : int or None, optional (default=None)
///     Size of batches for partial_fit. If None, the entire dataset is used.
///
/// Attributes
/// ----------
/// components_ : ndarray of shape (n_components, n_features)
///     Principal axes in feature space.
/// explained_variance_ratio_ : ndarray of shape (n_components,)
///     Percentage of variance explained by each component.
/// n_samples_seen_ : int
///     Total number of samples seen during fitting.
///
/// Examples
/// --------
/// >>> from ferroml.decomposition import IncrementalPCA
/// >>> import numpy as np
/// >>> X = np.random.randn(200, 10)
/// >>> ipca = IncrementalPCA(n_components=3)
/// >>> # Fit on full data
/// >>> X_t = ipca.fit_transform(X)
/// >>> X_t.shape
/// (200, 3)
/// >>> # Or fit incrementally with partial_fit
/// >>> ipca2 = IncrementalPCA(n_components=3)
/// >>> for i in range(0, 200, 50):
/// ...     ipca2.partial_fit(X[i:i+50])
///
/// Notes
/// -----
/// IncrementalPCA is useful when the dataset is too large to fit in memory.
/// Use ``partial_fit()`` to process data in batches. Results may differ
/// slightly from PCA due to incremental mean/variance estimation.
#[pyclass(name = "IncrementalPCA", module = "ferroml.decomposition")]
pub struct PyIncrementalPCA {
    inner: IncrementalPCA,
}

#[pymethods]
impl PyIncrementalPCA {
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(
            <ferroml_core::decomposition::IncrementalPCA as HasModelCard>::model_card(),
        )
    }

    /// Create a new IncrementalPCA.
    ///
    /// Parameters
    /// ----------
    /// n_components : int, optional
    ///     Number of components to keep. If not set, all components are kept.
    /// whiten : bool, optional (default=False)
    ///     When True, components are scaled to unit variance.
    /// batch_size : int, optional
    ///     Size of batches for partial_fit. If None, the entire dataset is used.
    ///
    /// Returns
    /// -------
    /// IncrementalPCA
    ///     A new model instance.
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
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Incremental fit on a batch of samples.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Training data batch.
    ///
    /// Returns
    /// -------
    /// self : IncrementalPCA
    ///     Updated estimator.
    fn partial_fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .partial_fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
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
///     Number of components to keep. Must be < n_features.
/// n_iter : int, optional (default=5)
///     Number of iterations for randomized SVD solver.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Attributes
/// ----------
/// components_ : ndarray of shape (n_components, n_features)
///     Right singular vectors (loadings).
/// explained_variance_ratio_ : ndarray of shape (n_components,)
///     Percentage of variance explained by each component.
///
/// Examples
/// --------
/// >>> from ferroml.decomposition import TruncatedSVD
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 20)
/// >>> svd = TruncatedSVD(n_components=5, random_state=42)
/// >>> X_t = svd.fit_transform(X)
/// >>> X_t.shape
/// (100, 5)
///
/// Notes
/// -----
/// Unlike PCA, TruncatedSVD does not center the data before computing SVD.
/// This makes it suitable for sparse matrices (e.g., term-document matrices
/// in LSA/LSI). For dense data, PCA is generally preferred.
#[pyclass(name = "TruncatedSVD", module = "ferroml.decomposition")]
pub struct PyTruncatedSVD {
    inner: TruncatedSVD,
}

#[pymethods]
impl PyTruncatedSVD {
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(<ferroml_core::decomposition::TruncatedSVD as HasModelCard>::model_card())
    }

    /// Create a new TruncatedSVD.
    ///
    /// Parameters
    /// ----------
    /// n_components : int, optional (default=2)
    ///     Number of components to compute.
    /// n_iter : int, optional (default=5)
    ///     Number of iterations for randomized SVD solver.
    /// random_state : int, optional
    ///     Random seed for reproducibility.
    ///
    /// Returns
    /// -------
    /// TruncatedSVD
    ///     A new model instance.
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
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
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
/// n_components : int or None, optional (default=None)
///     Number of components for dimensionality reduction. Maximum is
///     min(n_classes - 1, n_features). If None, uses n_classes - 1.
/// shrinkage : float or None, optional (default=None)
///     Shrinkage parameter for regularization (0.0 to 1.0). Useful when
///     n_samples < n_features. If None, no shrinkage is applied.
/// tol : float, optional (default=1e-4)
///     Convergence tolerance for eigenvalue truncation.
///
/// Attributes
/// ----------
/// scalings_ : ndarray of shape (n_features, n_components)
///     Projection vectors (discriminant directions).
/// classes_ : ndarray
///     Unique class labels seen during fit.
///
/// Examples
/// --------
/// >>> from ferroml.decomposition import LDA
/// >>> import numpy as np
/// >>> X = np.vstack([np.random.randn(30, 4) + [0, 0, 0, 0],
/// ...               np.random.randn(30, 4) + [3, 3, 3, 3]])
/// >>> y = np.array([0]*30 + [1]*30, dtype=np.float64)
/// >>> lda = LDA(n_components=1)
/// >>> X_proj = lda.fit(X, y).transform(X)
/// >>> X_proj.shape
/// (60, 1)
///
/// Notes
/// -----
/// LDA is both a classifier and a dimensionality reduction method. It
/// assumes equal covariance matrices across classes. Use ``shrinkage``
/// when n_samples is small relative to n_features to regularize the
/// within-class scatter matrix.
#[pyclass(name = "LDA", module = "ferroml.decomposition")]
pub struct PyLDA {
    inner: LDA,
}

#[pymethods]
impl PyLDA {
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(<ferroml_core::decomposition::LDA as HasModelCard>::model_card())
    }

    /// Create a new LDA.
    ///
    /// Parameters
    /// ----------
    /// n_components : int, optional
    ///     Number of components. Maximum is min(n_classes - 1, n_features).
    /// shrinkage : float, optional
    ///     Shrinkage parameter for regularization (0.0 to 1.0).
    /// tol : float, optional (default=1e-4)
    ///     Convergence tolerance for eigenvalue truncation.
    ///
    /// Returns
    /// -------
    /// LDA
    ///     A new model instance.
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
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform data using the fitted LDA model.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
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
/// n_factors : int or None, optional (default=None)
///     Number of latent factors. If None, uses n_features.
/// rotation : str, optional (default="none")
///     Rotation method: "none", "varimax", "quartimax", "promax".
///     Rotation improves interpretability without changing model fit.
/// tol : float, optional (default=1e-3)
///     Convergence tolerance for the EM algorithm.
/// max_iter : int, optional (default=1000)
///     Maximum number of EM iterations.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Attributes
/// ----------
/// components_ : ndarray of shape (n_factors, n_features)
///     Factor loading matrix (optionally rotated).
/// noise_variance_ : ndarray of shape (n_features,)
///     Estimated noise variance for each feature (uniquenesses).
///
/// Examples
/// --------
/// >>> from ferroml.decomposition import FactorAnalysis
/// >>> import numpy as np
/// >>> X = np.random.randn(200, 6)
/// >>> fa = FactorAnalysis(n_factors=2, rotation="varimax", random_state=42)
/// >>> X_scores = fa.fit_transform(X)
/// >>> X_scores.shape
/// (200, 2)
///
/// Notes
/// -----
/// Factor Analysis separates common variance (shared across features)
/// from unique variance (noise specific to each feature). Use varimax
/// rotation when you want orthogonal, interpretable factors.
#[pyclass(name = "FactorAnalysis", module = "ferroml.decomposition")]
pub struct PyFactorAnalysis {
    inner: FactorAnalysis,
}

#[pymethods]
impl PyFactorAnalysis {
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(
            <ferroml_core::decomposition::FactorAnalysis as HasModelCard>::model_card(),
        )
    }

    /// Create a new FactorAnalysis.
    ///
    /// Parameters
    /// ----------
    /// n_factors : int, optional
    ///     Number of latent factors. If None, uses n_features.
    /// rotation : str, optional (default="none")
    ///     Rotation method: "none", "varimax", "quartimax", "promax".
    /// tol : float, optional (default=1e-3)
    ///     Convergence tolerance for the EM algorithm.
    /// max_iter : int, optional (default=1000)
    ///     Maximum number of EM iterations.
    /// random_state : int, optional
    ///     Random seed for reproducibility.
    ///
    /// Returns
    /// -------
    /// FactorAnalysis
    ///     A new model instance.
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
        Ok(Self { inner: fa })
    }

    /// Fit the model to the data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform data.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
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

// =============================================================================
// TSNE
// =============================================================================

/// t-distributed Stochastic Neighbor Embedding (t-SNE).
///
/// Nonlinear dimensionality reduction for visualization. Maps high-dimensional
/// data to 2D or 3D while preserving local neighborhood structure.
///
/// Parameters
/// ----------
/// n_components : int, optional (default=2)
///     Number of output dimensions.
/// perplexity : float, optional (default=30.0)
///     Effective number of neighbors. Typical range: 5-50.
/// learning_rate : float or str, optional (default="auto")
///     Step size for gradient descent. "auto" = max(N/early_exaggeration/4, 50).
/// max_iter : int, optional (default=1000)
///     Maximum number of gradient descent iterations.
/// early_exaggeration : float, optional (default=12.0)
///     Factor to multiply P by for first 250 iterations.
/// min_grad_norm : float, optional (default=1e-7)
///     Convergence threshold on gradient norm.
/// metric : str, optional (default="euclidean")
///     Distance metric: "euclidean", "manhattan", or "cosine".
/// init : str, optional (default="pca")
///     Initialization: "pca" or "random".
/// method : str, optional (default="auto")
///     Algorithm: "exact" for O(N^2), "barnes_hut" for O(N log N) approximation,
///     or "auto" to choose based on dataset size (Barnes-Hut for n > 1000).
/// theta : float, optional (default=0.5)
///     Barnes-Hut trade-off parameter (0.0 = exact, higher = faster but less accurate).
///     Only used when method is "barnes_hut" or "auto" selects Barnes-Hut.
/// random_state : int, optional
///     Seed for reproducibility.
///
/// Attributes
/// ----------
/// embedding_ : ndarray of shape (n_samples, n_components)
///     The low-dimensional embedding after fitting.
/// kl_divergence_ : float
///     Final KL divergence between P and Q.
/// n_iter_ : int
///     Number of iterations actually run.
///
/// Examples
/// --------
/// >>> from ferroml.decomposition import TSNE
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 50)
/// >>> tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
/// >>> X_embedded = tsne.fit_transform(X)
/// >>> X_embedded.shape
/// (100, 2)
#[pyclass(name = "TSNE", module = "ferroml.decomposition")]
pub struct PyTSNE {
    inner: TSNE,
}

#[pymethods]
impl PyTSNE {
    /// Return structured metadata about this model.
    ///
    /// Returns
    /// -------
    /// ModelCard
    ///     Metadata including task type, complexity, interpretability, and more.
    #[staticmethod]
    fn model_card() -> PyModelCard {
        PyModelCard::new(<ferroml_core::decomposition::TSNE as HasModelCard>::model_card())
    }

    /// Create a new TSNE.
    ///
    /// Parameters
    /// ----------
    /// n_components : int, optional (default=2)
    ///     Dimension of the embedded space.
    /// perplexity : float, optional (default=30.0)
    ///     Related to the number of nearest neighbors; larger values consider more neighbors.
    /// learning_rate : float, optional
    ///     Learning rate for optimization. If None, uses "auto" (n_samples / early_exaggeration / 4).
    /// max_iter : int, optional (default=1000)
    ///     Maximum number of optimization iterations.
    /// early_exaggeration : float, optional (default=12.0)
    ///     Factor by which P is multiplied in early iterations.
    /// min_grad_norm : float, optional (default=1e-7)
    ///     Minimum gradient norm for early stopping.
    /// metric : str, optional (default="euclidean")
    ///     Distance metric: "euclidean", "manhattan", or "cosine".
    /// init : str, optional (default="pca")
    ///     Initialization: "pca" or "random".
    /// method : str, optional (default="auto")
    ///     Algorithm: "exact", "barnes_hut", or "auto" to choose based on dataset size.
    /// theta : float, optional (default=0.5)
    ///     Barnes-Hut trade-off parameter (0.0 = exact, higher = faster).
    /// random_state : int, optional
    ///     Seed for reproducibility.
    ///
    /// Returns
    /// -------
    /// TSNE
    ///     A new model instance.
    #[new]
    #[pyo3(signature = (
        n_components=2,
        perplexity=30.0,
        learning_rate=None,
        max_iter=1000,
        early_exaggeration=12.0,
        min_grad_norm=1e-7,
        metric="euclidean",
        init="pca",
        method="auto",
        theta=0.5,
        random_state=None,
    ))]
    fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: Option<f64>,
        max_iter: usize,
        early_exaggeration: f64,
        min_grad_norm: f64,
        metric: &str,
        init: &str,
        method: &str,
        theta: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let tsne_metric = match metric {
            "euclidean" => TsneMetric::Euclidean,
            "manhattan" => TsneMetric::Manhattan,
            "cosine" => TsneMetric::Cosine,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown metric '{}'. Expected 'euclidean', 'manhattan', or 'cosine'.",
                    metric
                )));
            }
        };

        let tsne_init = match init {
            "pca" => TsneInit::Pca,
            "random" => TsneInit::Random,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown init '{}'. Expected 'pca' or 'random'.",
                    init
                )));
            }
        };

        let lr = match learning_rate {
            Some(val) => LearningRate::Fixed(val),
            None => LearningRate::Auto,
        };

        let tsne_method = match method {
            "exact" => Some(TsneMethod::Exact),
            "barnes_hut" => Some(TsneMethod::BarnesHut),
            "auto" => None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown method '{}'. Expected 'exact', 'barnes_hut', or 'auto'.",
                    method
                )));
            }
        };

        let mut tsne = TSNE::new()
            .with_n_components(n_components)
            .with_perplexity(perplexity)
            .with_learning_rate(lr)
            .with_max_iter(max_iter)
            .with_early_exaggeration(early_exaggeration)
            .with_min_grad_norm(min_grad_norm)
            .with_metric(tsne_metric)
            .with_init(tsne_init)
            .with_theta(theta);

        if let Some(m) = tsne_method {
            tsne = tsne.with_method(m);
        }

        if let Some(seed) = random_state {
            tsne = tsne.with_random_state(seed);
        }

        Ok(Self { inner: tsne })
    }

    /// Fit the t-SNE model to the data.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Return the stored embedding (t-SNE is transductive).
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step (main entry point for t-SNE).
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Get the embedding after fitting.
    #[getter]
    fn embedding_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let embedding = self
            .inner
            .embedding()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(embedding.clone().into_pyarray(py))
    }

    /// Get the final KL divergence.
    #[getter]
    fn kl_divergence_(&self) -> PyResult<f64> {
        self.inner
            .kl_divergence()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))
    }

    /// Get the number of iterations run.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        self.inner
            .n_iter_final()
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
        "TSNE()".to_string()
    }
}

// =============================================================================
// Module registration
// =============================================================================

// Register the decomposition submodule — see register_decomposition_module below.

// =============================================================================
// QuadraticDiscriminantAnalysis (QDA)
// =============================================================================

/// Quadratic Discriminant Analysis.
///
/// Fits per-class covariance matrices for quadratic decision boundaries.
/// Unlike LDA which assumes shared covariance, QDA allows each class its own
/// covariance structure.
///
/// Parameters
/// ----------
/// reg_param : float, optional (default=0.0)
///     Regularization: Sigma_k = (1 - reg_param) * Sigma_k + reg_param * I.
///     Range: [0.0, 1.0].
/// priors : list of float or None, optional (default=None)
///     Class priors (must sum to 1). If None, estimated from data.
/// store_covariance : bool, optional (default=False)
///     Whether to store full covariance matrices.
/// tol : float, optional (default=1e-4)
///     Tolerance for eigenvalue truncation.
///
/// Attributes
/// ----------
/// classes_ : ndarray
///     Unique class labels.
/// priors_ : ndarray
///     Class prior probabilities.
///
/// Examples
/// --------
/// >>> from ferroml.decomposition import QuadraticDiscriminantAnalysis
/// >>> import numpy as np
/// >>> X = np.vstack([np.random.randn(30, 3), np.random.randn(30, 3) + 2])
/// >>> y = np.array([0]*30 + [1]*30, dtype=np.float64)
/// >>> model = QuadraticDiscriminantAnalysis(reg_param=0.1)
/// >>> model.fit(X, y)
/// >>> model.predict(np.array([[1.0, 1.0, 1.0]]))
///
/// Notes
/// -----
/// QDA fits separate covariance matrices per class, unlike LDA which
/// assumes shared covariance. Use ``reg_param`` to regularize when
/// individual class covariance matrices are poorly conditioned.
#[pyclass(
    name = "QuadraticDiscriminantAnalysis",
    module = "ferroml.decomposition"
)]
pub struct PyQDA {
    inner: ferroml_core::models::QuadraticDiscriminantAnalysis,
}

#[pymethods]
impl PyQDA {
    /// Create a new QuadraticDiscriminantAnalysis.
    ///
    /// Parameters
    /// ----------
    /// reg_param : float, optional (default=0.0)
    ///     Regularization parameter for covariance estimation. Range: [0.0, 1.0].
    /// priors : list of float, optional
    ///     Class priors (must sum to 1). If None, estimated from data.
    /// store_covariance : bool, optional (default=False)
    ///     Whether to store full covariance matrices.
    /// tol : float, optional (default=1e-4)
    ///     Tolerance for eigenvalue truncation.
    ///
    /// Returns
    /// -------
    /// QuadraticDiscriminantAnalysis
    ///     A new model instance.
    #[new]
    #[pyo3(signature = (reg_param=0.0, priors=None, store_covariance=false, tol=1e-4))]
    fn new(reg_param: f64, priors: Option<Vec<f64>>, store_covariance: bool, tol: f64) -> Self {
        let mut qda = ferroml_core::models::QuadraticDiscriminantAnalysis::new()
            .with_reg_param(reg_param)
            .with_store_covariance(store_covariance)
            .with_tol(tol);
        if let Some(p) = priors {
            qda = qda.with_priors(p);
        }
        Self { inner: qda }
    }

    /// Fit the QDA model.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = crate::array_utils::py_array_to_f64_1d(py, y)?;
        use ferroml_core::models::Model;
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
        use ferroml_core::models::Model;
        let preds = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    /// Predict class probabilities.
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let proba = self
            .inner
            .predict_proba(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(proba.into_pyarray(py))
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
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let proba = self
            .inner
            .predict_proba(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(proba.mapv(|p| p.max(1e-15).ln()).into_pyarray(py))
    }

    /// Compute decision function values.
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let scores = self
            .inner
            .decision_function(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(scores.into_pyarray(py))
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
        use ferroml_core::models::Model;
        self.inner
            .score(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "QuadraticDiscriminantAnalysis()".to_string()
    }
}

pub fn register_decomposition_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "decomposition")?;

    m.add_class::<PyPCA>()?;
    m.add_class::<PyIncrementalPCA>()?;
    m.add_class::<PyTruncatedSVD>()?;
    m.add_class::<PyLDA>()?;
    m.add_class::<PyQDA>()?;
    m.add_class::<PyFactorAnalysis>()?;
    m.add_class::<PyTSNE>()?;

    parent_module.add_submodule(&m)?;

    Ok(())
}
