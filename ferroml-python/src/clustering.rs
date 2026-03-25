//! Python bindings for FerroML Clustering algorithms
//!
//! This module provides Python wrappers for:
//! - KMeans clustering with kmeans++ initialization
//! - DBSCAN density-based clustering
//! - Clustering metrics (silhouette, Calinski-Harabasz, Davies-Bouldin, etc.)
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays, a copy is made.
//! Output arrays use `into_pyarray` to transfer ownership to Python without copying data.
//!
//! See `crate::array_utils` for detailed documentation.

use crate::array_utils::{check_array_finite, to_owned_array_2d};
use crate::pickle::{getstate, setstate};
use ferroml_core::clustering::metrics;
use ferroml_core::clustering::{
    AgglomerativeClustering, ClusteringModel, ClusteringStatistics, CovarianceType,
    GaussianMixture, GmmInit, KMeans, KMeansAlgorithm, DBSCAN,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// =============================================================================
// KMeans
// =============================================================================

/// K-Means Clustering with kmeans++ initialization.
///
/// K-Means partitions data into k clusters by minimizing within-cluster
/// sum of squares (inertia). This implementation uses the kmeans++
/// initialization strategy for better convergence.
///
/// Parameters
/// ----------
/// n_clusters : int, optional (default=8)
///     Number of clusters to form.
/// max_iter : int, optional (default=300)
///     Maximum number of iterations of the k-means algorithm.
/// tol : float, optional (default=1e-4)
///     Relative tolerance for convergence.
/// random_state : int, optional
///     Random seed for centroid initialization.
/// n_init : int, optional (default=10)
///     Number of times to run with different centroid seeds.
/// algorithm : str, optional (default="auto")
///     Algorithm variant: "auto", "lloyd", "elkan", "hamerly".
///     Auto selects Hamerly for k<=20, Elkan for k<=256, Lloyd otherwise.
///
/// Attributes
/// ----------
/// cluster_centers_ : ndarray of shape (n_clusters, n_features)
///     Coordinates of cluster centers.
/// labels_ : ndarray of shape (n_samples,)
///     Labels of each point.
/// inertia_ : float
///     Sum of squared distances to closest cluster center.
/// n_iter_ : int
///     Number of iterations run.
///
/// Examples
/// --------
/// >>> from ferroml.clustering import KMeans
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
/// >>> kmeans = KMeans(n_clusters=2, random_state=42)
/// >>> kmeans.fit(X)
/// >>> kmeans.labels_
/// >>> kmeans.cluster_centers_
#[pyclass(name = "KMeans", module = "ferroml.clustering")]
pub struct PyKMeans {
    inner: KMeans,
}

#[pymethods]
impl PyKMeans {
    /// Create a new KMeans model.
    #[new]
    #[pyo3(signature = (n_clusters=8, max_iter=300, tol=1e-4, random_state=None, n_init=10, algorithm="auto"))]
    fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
        n_init: usize,
        algorithm: &str,
    ) -> PyResult<Self> {
        let algo = match algorithm {
            "auto" => KMeansAlgorithm::Auto,
            "lloyd" => KMeansAlgorithm::Lloyd,
            "elkan" => KMeansAlgorithm::Elkan,
            "hamerly" => KMeansAlgorithm::Hamerly,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown algorithm '{}'. Expected one of: 'auto', 'lloyd', 'elkan', 'hamerly'",
                    other
                )));
            }
        };

        let mut inner = KMeans::new(n_clusters)
            .max_iter(max_iter)
            .tol(tol)
            .n_init(n_init)
            .algorithm(algo);

        if let Some(seed) = random_state {
            inner = inner.random_state(seed);
        }

        Ok(Self { inner })
    }

    /// Fit the KMeans model to the data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// self : KMeans
    ///     Fitted estimator.
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

    /// Predict cluster labels for samples.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     New data to predict.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Index of the cluster each sample belongs to.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let labels = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(labels.into_pyarray(py))
    }

    /// Fit the model and predict cluster labels in one step.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Cluster labels.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let labels = self
            .inner
            .fit_predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(labels.into_pyarray(py))
    }

    /// Compute cluster stability using bootstrap resampling.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data used for stability assessment.
    /// n_bootstrap : int, optional (default=100)
    ///     Number of bootstrap iterations.
    ///
    /// Returns
    /// -------
    /// stability : ndarray of shape (n_clusters,)
    ///     Stability score for each cluster (0-1, higher is more stable).
    #[pyo3(signature = (x, n_bootstrap=100))]
    fn cluster_stability<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        n_bootstrap: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let stability = self
            .inner
            .cluster_stability(&x_arr, n_bootstrap)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(stability.into_pyarray(py))
    }

    /// Compute silhouette scores with bootstrap confidence intervals.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data to compute silhouette on.
    /// confidence : float, optional (default=0.95)
    ///     Confidence level for the interval.
    ///
    /// Returns
    /// -------
    /// tuple of (mean, lower_ci, upper_ci)
    #[pyo3(signature = (x, confidence=0.95))]
    fn silhouette_with_ci(
        &self,
        x: PyReadonlyArray2<'_, f64>,
        confidence: f64,
    ) -> PyResult<(f64, f64, f64)> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        self.inner
            .silhouette_with_ci(&x_arr, confidence)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    /// Find optimal k using the gap statistic.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data for analysis.
    /// k_min : int, optional (default=1)
    ///     Minimum k to test.
    /// k_max : int, optional (default=10)
    ///     Maximum k to test.
    /// n_refs : int, optional (default=10)
    ///     Number of reference datasets.
    /// random_state : int, optional
    ///     Random seed.
    ///
    /// Returns
    /// -------
    /// dict with keys: 'optimal_k', 'k_values', 'gap_values', 'gap_se'
    #[staticmethod]
    #[pyo3(signature = (x, k_min=1, k_max=10, n_refs=10, random_state=None))]
    fn optimal_k(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        k_min: usize,
        k_max: usize,
        n_refs: usize,
        random_state: Option<u64>,
    ) -> PyResult<PyObject> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let result = KMeans::optimal_k(&x_arr, k_min..k_max, n_refs, random_state)
            .map_err(crate::errors::ferro_to_pyerr)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("optimal_k", result.optimal_k)?;
        dict.set_item("k_values", result.k_values)?;
        dict.set_item("gap_values", result.gap_values)?;
        dict.set_item("gap_se", result.gap_se)?;

        Ok(dict.into())
    }

    /// Find optimal k using the elbow method.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data for analysis.
    /// k_min : int, optional (default=1)
    ///     Minimum k to test.
    /// k_max : int, optional (default=10)
    ///     Maximum k to test.
    /// random_state : int, optional
    ///     Random seed.
    ///
    /// Returns
    /// -------
    /// dict with keys: 'optimal_k', 'k_values', 'inertias'
    #[staticmethod]
    #[pyo3(signature = (x, k_min=1, k_max=10, random_state=None))]
    fn elbow(
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        k_min: usize,
        k_max: usize,
        random_state: Option<u64>,
    ) -> PyResult<PyObject> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let result = KMeans::elbow(&x_arr, k_min..k_max, random_state)
            .map_err(crate::errors::ferro_to_pyerr)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("optimal_k", result.optimal_k)?;
        dict.set_item("k_values", result.k_values)?;
        dict.set_item("inertias", result.inertias)?;

        Ok(dict.into())
    }

    /// Get cluster centers.
    #[getter]
    fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let centers = self.inner.cluster_centers().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(centers.clone().into_pyarray(py))
    }

    /// Get cluster labels.
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let labels = self.inner.labels().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(labels.clone().into_pyarray(py))
    }

    /// Opposite of the value of X on the K-means objective (negative inertia).
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     New data.
    ///
    /// Returns
    /// -------
    /// float
    ///     Negative inertia on the given data.
    fn score(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        // Compute inertia on new data: sum of squared distances to nearest center
        let labels = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        let centers = self.inner.cluster_centers().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        let mut inertia = 0.0;
        for (i, &label) in labels.iter().enumerate() {
            let center = centers.row(label as usize);
            let diff = &x_arr.row(i) - &center;
            inertia += diff.dot(&diff);
        }
        Ok(-inertia)
    }

    /// Get inertia (within-cluster sum of squares).
    #[getter]
    fn inertia_(&self) -> PyResult<f64> {
        self.inner.inertia().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get number of iterations run.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        self.inner.n_iter().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Return the state of the model for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the model state from pickled bytes.
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "KMeans(n_clusters={})",
            self.inner.cluster_centers().map_or(0, |c| c.nrows())
        )
    }
}

// =============================================================================
// DBSCAN
// =============================================================================

/// DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
///
/// DBSCAN groups together points that are closely packed (density-based),
/// marking outliers as noise that lie alone in low-density regions.
///
/// Parameters
/// ----------
/// eps : float, optional (default=0.5)
///     Maximum distance between two samples to be considered neighbors.
/// min_samples : int, optional (default=5)
///     Minimum number of samples in a neighborhood to form a core point.
///
/// Attributes
/// ----------
/// labels_ : ndarray of shape (n_samples,)
///     Cluster labels for each point. Noisy samples are given label -1.
/// core_sample_indices_ : ndarray
///     Indices of core samples.
/// components_ : ndarray
///     Copy of each core sample.
///
/// Examples
/// --------
/// >>> from ferroml.clustering import DBSCAN
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
/// >>> dbscan = DBSCAN(eps=3, min_samples=2)
/// >>> dbscan.fit(X)
/// >>> dbscan.labels_
#[pyclass(name = "DBSCAN", module = "ferroml.clustering")]
pub struct PyDBSCAN {
    inner: DBSCAN,
}

#[pymethods]
impl PyDBSCAN {
    /// Create a new DBSCAN model.
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5))]
    fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            inner: DBSCAN::new(eps, min_samples),
        }
    }

    /// Fit the DBSCAN model to the data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// self : DBSCAN
    ///     Fitted estimator.
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

    /// Predict cluster labels for new samples.
    ///
    /// Note: DBSCAN assigns new points to the nearest core sample's cluster
    /// if within eps distance, otherwise marks as noise (-1).
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     New data to predict.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Cluster labels (-1 for noise).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let labels = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(labels.into_pyarray(py))
    }

    /// Fit the model and predict cluster labels in one step.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Cluster labels (-1 for noise).
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let labels = self
            .inner
            .fit_predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(labels.into_pyarray(py))
    }

    /// Compute optimal eps using k-distance graph analysis.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data for analysis.
    /// min_samples : int, optional (default=5)
    ///     min_samples parameter to use.
    ///
    /// Returns
    /// -------
    /// dict with keys: 'suggested_eps', 'k_distances'
    #[staticmethod]
    #[pyo3(signature = (x, min_samples=5))]
    fn optimal_eps<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        min_samples: usize,
    ) -> PyResult<PyObject> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let (suggested_eps, k_distances) =
            DBSCAN::optimal_eps(&x_arr, min_samples).map_err(crate::errors::ferro_to_pyerr)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("suggested_eps", suggested_eps)?;
        dict.set_item("k_distances", k_distances)?;

        Ok(dict.into())
    }

    /// Analyze stability of clusters across eps values.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data for analysis.
    /// eps_values : list of float
    ///     Eps values to test.
    /// min_samples : int, optional (default=5)
    ///     min_samples parameter to use.
    ///
    /// Returns
    /// -------
    /// list of (eps, n_clusters, n_noise) tuples
    #[staticmethod]
    #[pyo3(signature = (x, eps_values, min_samples=5))]
    fn cluster_persistence(
        x: PyReadonlyArray2<'_, f64>,
        eps_values: Vec<f64>,
        min_samples: usize,
    ) -> PyResult<Vec<(f64, usize, usize)>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        DBSCAN::cluster_persistence(&x_arr, &eps_values, min_samples)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    /// Analyze noise points statistically.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data used for analysis.
    ///
    /// Returns
    /// -------
    /// dict with keys: 'noise_ratio', 'centroid', 'std'
    fn noise_analysis<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyObject> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let (noise_ratio, centroid, std) = self
            .inner
            .noise_analysis(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("noise_ratio", noise_ratio)?;
        dict.set_item("centroid", centroid.into_pyarray(py))?;
        dict.set_item("std", std.into_pyarray(py))?;

        Ok(dict.into())
    }

    /// Get cluster labels.
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let labels = self.inner.labels().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(labels.clone().into_pyarray(py))
    }

    /// Get indices of core samples.
    #[getter]
    fn core_sample_indices_(&self) -> PyResult<Vec<usize>> {
        self.inner.core_sample_indices().cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get core sample coordinates.
    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let components = self.inner.components().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(components.clone().into_pyarray(py))
    }

    /// Get number of clusters found (excluding noise).
    #[getter]
    fn n_clusters_(&self) -> PyResult<usize> {
        self.inner.n_clusters().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get number of noise points.
    #[getter]
    fn n_noise_(&self) -> PyResult<usize> {
        self.inner.n_noise().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Return the state of the model for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the model state from pickled bytes.
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let n_clusters = self.inner.n_clusters().unwrap_or(0);
        let n_noise = self.inner.n_noise().unwrap_or(0);
        format!("DBSCAN(n_clusters={}, n_noise={})", n_clusters, n_noise)
    }
}

// =============================================================================
// AgglomerativeClustering
// =============================================================================

/// Agglomerative (hierarchical) Clustering.
///
/// Recursively merges the pair of clusters that minimally increases a
/// given linkage distance.
///
/// Parameters
/// ----------
/// n_clusters : int, optional (default=2)
///     Number of clusters to find.
/// linkage : str, optional (default="ward")
///     Linkage criterion: "ward", "complete", "average", "single".
///
/// Attributes
/// ----------
/// labels_ : ndarray of shape (n_samples,)
///     Cluster labels for each sample.
///
/// Examples
/// --------
/// >>> from ferroml.clustering import AgglomerativeClustering
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
/// >>> model = AgglomerativeClustering(n_clusters=2, linkage="ward")
/// >>> labels = model.fit_predict(X)
#[pyclass(name = "AgglomerativeClustering", module = "ferroml.clustering")]
pub struct PyAgglomerativeClustering {
    inner: AgglomerativeClustering,
}

#[pymethods]
impl PyAgglomerativeClustering {
    #[new]
    #[pyo3(signature = (n_clusters=2, linkage="ward"))]
    fn new(n_clusters: usize, linkage: &str) -> PyResult<Self> {
        use ferroml_core::clustering::Linkage;

        let link = match linkage {
            "ward" => Linkage::Ward,
            "complete" => Linkage::Complete,
            "average" => Linkage::Average,
            "single" => Linkage::Single,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown linkage: '{}'. Use 'ward', 'complete', 'average', or 'single'.",
                    linkage
                )));
            }
        };

        Ok(Self {
            inner: AgglomerativeClustering::new(n_clusters).with_linkage(link),
        })
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

    /// Fit the model and return cluster labels.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        self.inner
            .fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        let labels = self
            .inner
            .labels()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(labels.clone().into_pyarray(py))
    }

    /// Get cluster labels.
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let labels = self.inner.labels().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(labels.clone().into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "AgglomerativeClustering()".to_string()
    }
}

// =============================================================================
// GaussianMixture
// =============================================================================

/// Gaussian Mixture Model fitted via Expectation-Maximization.
///
/// Parameters
/// ----------
/// n_components : int, optional (default=1)
///     Number of mixture components.
/// covariance_type : str, optional (default="full")
///     Covariance type: "full", "tied", "diagonal", "spherical".
/// max_iter : int, optional (default=100)
///     Maximum number of EM iterations.
/// tol : float, optional (default=1e-3)
///     Convergence threshold.
/// n_init : int, optional (default=1)
///     Number of initializations to perform.
/// init_params : str, optional (default="kmeans")
///     Initialization method: "kmeans" or "random".
/// reg_covar : float, optional (default=1e-6)
///     Regularization added to diagonal of covariance matrices.
/// random_state : int, optional
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.clustering import GaussianMixture
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
/// >>> gmm = GaussianMixture(n_components=2, random_state=42)
/// >>> gmm.fit(X)
/// >>> gmm.predict(X)
/// >>> gmm.predict_proba(X)
#[pyclass(name = "GaussianMixture", module = "ferroml.clustering")]
pub struct PyGaussianMixture {
    inner: GaussianMixture,
}

#[pymethods]
impl PyGaussianMixture {
    #[new]
    #[pyo3(signature = (
        n_components=1,
        covariance_type="full",
        max_iter=100,
        tol=1e-3,
        n_init=1,
        init_params="kmeans",
        reg_covar=1e-6,
        random_state=None
    ))]
    fn new(
        n_components: usize,
        covariance_type: &str,
        max_iter: usize,
        tol: f64,
        n_init: usize,
        init_params: &str,
        reg_covar: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let cov_type = match covariance_type {
            "full" => CovarianceType::Full,
            "tied" => CovarianceType::Tied,
            "diag" | "diagonal" => CovarianceType::Diagonal,
            "spherical" => CovarianceType::Spherical,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown covariance_type: '{}'. Use 'full', 'tied', 'diag', or 'spherical'.",
                    covariance_type
                )));
            }
        };

        let init = match init_params {
            "kmeans" | "k-means" => GmmInit::KMeans,
            "random" => GmmInit::Random,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown init_params: '{}'. Use 'kmeans' or 'random'.",
                    init_params
                )));
            }
        };

        let mut inner = GaussianMixture::new(n_components)
            .covariance_type(cov_type)
            .max_iter(max_iter)
            .tol(tol)
            .n_init(n_init)
            .init_params(init)
            .reg_covar(reg_covar);

        if let Some(seed) = random_state {
            inner = inner.random_state(seed);
        }

        Ok(Self { inner })
    }

    /// Fit the Gaussian mixture model.
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

    /// Predict cluster labels (hard assignment).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let labels = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(labels.into_pyarray(py))
    }

    /// Fit and predict in one step.
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let labels = self
            .inner
            .fit_predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(labels.into_pyarray(py))
    }

    /// Predict posterior probability of each component given data.
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

    /// Predict log-probabilities of each component given data.
    ///
    /// Parameters
    /// ----------
    /// x : ndarray of shape (n_samples, n_features)
    ///     Input features.
    ///
    /// Returns
    /// -------
    /// log_probas : ndarray of shape (n_samples, n_components)
    ///     Log posterior probability of each component.
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

    /// Compute per-sample log-likelihood.
    fn score_samples<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let scores = self
            .inner
            .score_samples(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(scores.into_pyarray(py))
    }

    /// Compute mean per-sample log-likelihood.
    fn score(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        self.inner
            .score(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    /// Compute Bayesian Information Criterion.
    fn bic(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        self.inner
            .bic(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    /// Compute Akaike Information Criterion.
    fn aic(&self, x: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        self.inner
            .aic(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)
    }

    /// Generate random samples from the fitted model.
    ///
    /// Returns
    /// -------
    /// tuple of (X, labels) where X has shape (n_samples, n_features)
    #[pyo3(signature = (n_samples=1))]
    #[allow(clippy::type_complexity)]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n_samples: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i32>>)> {
        let (samples, labels) = self
            .inner
            .sample(n_samples)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok((samples.into_pyarray(py), labels.into_pyarray(py)))
    }

    /// Get mixture weights.
    #[getter]
    fn weights_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let weights = self.inner.weights().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(weights.clone().into_pyarray(py))
    }

    /// Get component means.
    #[getter]
    fn means_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let means = self.inner.means().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(means.clone().into_pyarray(py))
    }

    /// Get cluster labels from last fit.
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let labels = self.inner.labels().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(labels.clone().into_pyarray(py))
    }

    /// Get number of iterations run.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        self.inner.n_iter().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Whether the EM algorithm converged.
    #[getter]
    fn converged_(&self) -> PyResult<bool> {
        self.inner.converged().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the lower bound value on the log-likelihood.
    #[getter]
    fn lower_bound_(&self) -> PyResult<f64> {
        self.inner.lower_bound().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Return the state of the model for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the model state from pickled bytes.
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GaussianMixture(n_components={})",
            self.inner.weights().map_or(0, |w| w.len())
        )
    }
}

// =============================================================================
// Clustering Metrics Functions
// =============================================================================

/// Compute silhouette score for clustering.
///
/// The silhouette score measures how similar a sample is to its own cluster
/// compared to other clusters. Ranges from -1 to 1, where 1 is best.
///
/// Parameters
/// ----------
/// X : array-like of shape (n_samples, n_features)
///     Feature matrix.
/// labels : array-like of shape (n_samples,)
///     Cluster labels for each sample.
///
/// Returns
/// -------
/// score : float
///     Mean silhouette coefficient.
///
/// Examples
/// --------
/// >>> from ferroml.clustering import silhouette_score
/// >>> import numpy as np
/// >>> X = np.array([[1, 1], [1.5, 1.5], [10, 10], [10.5, 10.5]])
/// >>> labels = np.array([0, 0, 1, 1])
/// >>> silhouette_score(X, labels)
#[pyfunction]
#[pyo3(name = "silhouette_score")]
fn py_silhouette_score(
    x: PyReadonlyArray2<'_, f64>,
    labels: PyReadonlyArray1<'_, i32>,
) -> PyResult<f64> {
    check_array_finite(&x)?;
    let x_arr = to_owned_array_2d(x);
    let labels_arr = labels.as_array().to_owned();

    metrics::silhouette_score(&x_arr, &labels_arr).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute silhouette coefficient for each sample.
///
/// Parameters
/// ----------
/// X : array-like of shape (n_samples, n_features)
///     Feature matrix.
/// labels : array-like of shape (n_samples,)
///     Cluster labels for each sample.
///
/// Returns
/// -------
/// silhouette : ndarray of shape (n_samples,)
///     Silhouette coefficient for each sample.
#[pyfunction]
#[pyo3(name = "silhouette_samples")]
fn py_silhouette_samples<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    labels: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_array_finite(&x)?;
    let x_arr = to_owned_array_2d(x);
    let labels_arr = labels.as_array().to_owned();

    let samples =
        metrics::silhouette_samples(&x_arr, &labels_arr).map_err(crate::errors::ferro_to_pyerr)?;

    Ok(samples.into_pyarray(py))
}

/// Compute Calinski-Harabasz score (Variance Ratio Criterion).
///
/// Higher values indicate better-defined clusters.
///
/// Parameters
/// ----------
/// X : array-like of shape (n_samples, n_features)
///     Feature matrix.
/// labels : array-like of shape (n_samples,)
///     Cluster labels for each sample.
///
/// Returns
/// -------
/// score : float
///     Calinski-Harabasz index.
#[pyfunction]
#[pyo3(name = "calinski_harabasz_score")]
fn py_calinski_harabasz_score(
    x: PyReadonlyArray2<'_, f64>,
    labels: PyReadonlyArray1<'_, i32>,
) -> PyResult<f64> {
    check_array_finite(&x)?;
    let x_arr = to_owned_array_2d(x);
    let labels_arr = labels.as_array().to_owned();

    metrics::calinski_harabasz_score(&x_arr, &labels_arr).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Davies-Bouldin score.
///
/// Lower values indicate better clustering.
///
/// Parameters
/// ----------
/// X : array-like of shape (n_samples, n_features)
///     Feature matrix.
/// labels : array-like of shape (n_samples,)
///     Cluster labels for each sample.
///
/// Returns
/// -------
/// score : float
///     Davies-Bouldin index.
#[pyfunction]
#[pyo3(name = "davies_bouldin_score")]
fn py_davies_bouldin_score(
    x: PyReadonlyArray2<'_, f64>,
    labels: PyReadonlyArray1<'_, i32>,
) -> PyResult<f64> {
    check_array_finite(&x)?;
    let x_arr = to_owned_array_2d(x);
    let labels_arr = labels.as_array().to_owned();

    metrics::davies_bouldin_score(&x_arr, &labels_arr).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Adjusted Rand Index.
///
/// Measures similarity between two clusterings, adjusted for chance.
/// Ranges from -1 to 1, where 1 is perfect agreement.
///
/// Parameters
/// ----------
/// labels_true : array-like of shape (n_samples,)
///     Ground truth cluster labels.
/// labels_pred : array-like of shape (n_samples,)
///     Predicted cluster labels.
///
/// Returns
/// -------
/// score : float
///     Adjusted Rand Index.
#[pyfunction]
#[pyo3(name = "adjusted_rand_index")]
fn py_adjusted_rand_index(
    labels_true: PyReadonlyArray1<'_, i32>,
    labels_pred: PyReadonlyArray1<'_, i32>,
) -> PyResult<f64> {
    let true_arr = labels_true.as_array().to_owned();
    let pred_arr = labels_pred.as_array().to_owned();

    metrics::adjusted_rand_index(&true_arr, &pred_arr).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Normalized Mutual Information.
///
/// Measures mutual information between two clusterings, normalized to [0, 1].
///
/// Parameters
/// ----------
/// labels_true : array-like of shape (n_samples,)
///     Ground truth cluster labels.
/// labels_pred : array-like of shape (n_samples,)
///     Predicted cluster labels.
///
/// Returns
/// -------
/// score : float
///     Normalized Mutual Information.
#[pyfunction]
#[pyo3(name = "normalized_mutual_info")]
fn py_normalized_mutual_info(
    labels_true: PyReadonlyArray1<'_, i32>,
    labels_pred: PyReadonlyArray1<'_, i32>,
) -> PyResult<f64> {
    let true_arr = labels_true.as_array().to_owned();
    let pred_arr = labels_pred.as_array().to_owned();

    metrics::normalized_mutual_info(&true_arr, &pred_arr).map_err(crate::errors::ferro_to_pyerr)
}

/// Compute Hopkins statistic for clustering tendency.
///
/// Tests whether the data has a uniform distribution or contains clusters.
/// Values close to 0.5 indicate uniform distribution, close to 1 indicates
/// clustered data.
///
/// Parameters
/// ----------
/// X : array-like of shape (n_samples, n_features)
///     Feature matrix.
/// sample_size : int, optional
///     Number of points to sample. Default is 10% of n_samples.
/// random_state : int, optional
///     Random seed for reproducibility.
///
/// Returns
/// -------
/// score : float
///     Hopkins statistic (0-1, >0.5 suggests clustering tendency).
#[pyfunction]
#[pyo3(name = "hopkins_statistic", signature = (x, sample_size=None, random_state=None))]
fn py_hopkins_statistic(
    x: PyReadonlyArray2<'_, f64>,
    sample_size: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<f64> {
    check_array_finite(&x)?;
    let x_arr = to_owned_array_2d(x);

    metrics::hopkins_statistic(&x_arr, sample_size, random_state)
        .map_err(crate::errors::ferro_to_pyerr)
}

// =============================================================================
// HDBSCAN
// =============================================================================

/// HDBSCAN: Hierarchical Density-Based Spatial Clustering.
///
/// HDBSCAN extends DBSCAN by building a hierarchy of clusters and extracting
/// the most stable flat clusters using the excess of mass method. Unlike DBSCAN,
/// it can find clusters of varying density without requiring an `eps` parameter.
///
/// Parameters
/// ----------
/// min_cluster_size : int, optional (default=5)
///     Minimum number of points to form a cluster.
/// min_samples : int, optional
///     Number of neighbors for core distance computation. Defaults to
///     min_cluster_size if not specified.
/// cluster_selection_epsilon : float, optional (default=0.0)
///     Clusters closer than this distance threshold will be merged.
/// allow_single_cluster : bool, optional (default=False)
///     Whether to allow a single cluster as output.
///
/// Attributes
/// ----------
/// labels_ : ndarray of shape (n_samples,)
///     Cluster labels (-1 for noise).
/// probabilities_ : ndarray of shape (n_samples,)
///     Cluster membership probabilities.
/// n_clusters_ : int
///     Number of clusters found.
///
/// Examples
/// --------
/// >>> from ferroml.clustering import HDBSCAN
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [1.5, 1.8], [1.2, 2.1], [5, 8], [5.5, 7.8], [5.2, 8.1]])
/// >>> hdbscan = HDBSCAN(min_cluster_size=2)
/// >>> hdbscan.fit(X)
/// >>> hdbscan.labels_
#[pyclass(name = "HDBSCAN", module = "ferroml.clustering")]
pub struct PyHDBSCAN {
    inner: ferroml_core::clustering::HDBSCAN,
}

#[pymethods]
impl PyHDBSCAN {
    /// Create a new HDBSCAN model.
    #[new]
    #[pyo3(signature = (min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, allow_single_cluster=false))]
    fn new(
        min_cluster_size: usize,
        min_samples: Option<usize>,
        cluster_selection_epsilon: f64,
        allow_single_cluster: bool,
    ) -> Self {
        let mut inner = ferroml_core::clustering::HDBSCAN::new(min_cluster_size);
        if let Some(ms) = min_samples {
            inner = inner.with_min_samples(ms);
        }
        if cluster_selection_epsilon > 0.0 {
            inner = inner.with_cluster_selection_epsilon(cluster_selection_epsilon);
        }
        if allow_single_cluster {
            inner = inner.with_allow_single_cluster(allow_single_cluster);
        }
        Self { inner }
    }

    /// Fit the HDBSCAN model to the data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// self : HDBSCAN
    ///     Fitted estimator.
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

    /// Fit the model and predict cluster labels in one step.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// labels : ndarray of shape (n_samples,)
    ///     Cluster labels (-1 for noise).
    fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let labels = self
            .inner
            .fit_predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(labels.into_pyarray(py))
    }

    /// Cluster labels for each point (-1 for noise).
    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let labels = self
            .inner
            .labels()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(labels.into_pyarray(py))
    }

    /// Cluster membership probabilities for each point.
    #[getter]
    fn probabilities_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let probs = self
            .inner
            .probabilities()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(probs.into_pyarray(py))
    }

    /// Number of clusters found (excluding noise).
    #[getter]
    fn n_clusters_(&self) -> PyResult<usize> {
        self.inner.n_clusters().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted. Call fit() first.")
        })
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the clustering submodule.
pub fn register_clustering_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let clustering_module = PyModule::new(parent_module.py(), "clustering")?;

    // Add classes
    clustering_module.add_class::<PyKMeans>()?;
    clustering_module.add_class::<PyDBSCAN>()?;
    clustering_module.add_class::<PyAgglomerativeClustering>()?;
    clustering_module.add_class::<PyGaussianMixture>()?;
    clustering_module.add_class::<PyHDBSCAN>()?;

    // Add metric functions
    clustering_module.add_function(wrap_pyfunction!(py_silhouette_score, &clustering_module)?)?;
    clustering_module.add_function(wrap_pyfunction!(py_silhouette_samples, &clustering_module)?)?;
    clustering_module.add_function(wrap_pyfunction!(
        py_calinski_harabasz_score,
        &clustering_module
    )?)?;
    clustering_module.add_function(wrap_pyfunction!(
        py_davies_bouldin_score,
        &clustering_module
    )?)?;
    clustering_module.add_function(wrap_pyfunction!(
        py_adjusted_rand_index,
        &clustering_module
    )?)?;
    clustering_module.add_function(wrap_pyfunction!(
        py_normalized_mutual_info,
        &clustering_module
    )?)?;
    clustering_module.add_function(wrap_pyfunction!(py_hopkins_statistic, &clustering_module)?)?;

    parent_module.add_submodule(&clustering_module)?;

    Ok(())
}
