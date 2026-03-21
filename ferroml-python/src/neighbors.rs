//! Python bindings for FerroML K-Nearest Neighbors models
//!
//! This module provides Python wrappers for:
//! - KNeighborsClassifier
//! - KNeighborsRegressor
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays (e.g., `Model::fit`),
//! a copy is made. Output arrays use `into_pyarray` to transfer ownership to Python
//! without copying data.
//!
//! See `crate::array_utils` for detailed documentation.

use crate::array_utils::{
    check_array1_finite, check_array_finite, py_array_to_f64_1d, to_owned_array_1d,
    to_owned_array_2d,
};
use crate::pickle::{getstate, setstate};
use ferroml_core::models::knn::{
    DistanceMetric, KNNAlgorithm, KNNWeights, KNeighborsClassifier, KNeighborsRegressor,
    NearestCentroid,
};
use ferroml_core::models::{Model, ProbabilisticModel};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn parse_knn_weights(weights: &str) -> PyResult<KNNWeights> {
    match weights.to_lowercase().as_str() {
        "uniform" => Ok(KNNWeights::Uniform),
        "distance" => Ok(KNNWeights::Distance),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must be 'uniform' or 'distance'",
        )),
    }
}

fn parse_distance_metric(metric: &str, p: f64) -> PyResult<DistanceMetric> {
    match metric.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "manhattan" | "l1" | "cityblock" => Ok(DistanceMetric::Manhattan),
        "minkowski" => Ok(DistanceMetric::Minkowski(p)),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "metric must be 'euclidean', 'manhattan', or 'minkowski'",
        )),
    }
}

fn parse_knn_algorithm(algorithm: &str) -> PyResult<KNNAlgorithm> {
    match algorithm.to_lowercase().as_str() {
        "auto" => Ok(KNNAlgorithm::Auto),
        "kd_tree" | "kdtree" => Ok(KNNAlgorithm::KDTree),
        "ball_tree" | "balltree" => Ok(KNNAlgorithm::BallTree),
        "brute" | "brute_force" => Ok(KNNAlgorithm::BruteForce),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "algorithm must be 'auto', 'kd_tree', 'ball_tree', or 'brute'",
        )),
    }
}

// =============================================================================
// KNeighborsClassifier
// =============================================================================

/// K-Nearest Neighbors Classifier.
///
/// Classification based on majority voting among the k nearest neighbors.
/// Supports weighted voting where closer neighbors have more influence.
///
/// Parameters
/// ----------
/// n_neighbors : int, optional (default=5)
///     Number of neighbors to use for prediction.
/// weights : str, optional (default="uniform")
///     Weight function used in prediction. "uniform" for equal weights,
///     "distance" for weight by inverse of distance.
/// metric : str, optional (default="euclidean")
///     Distance metric. Options: "euclidean", "manhattan", "minkowski".
/// algorithm : str, optional (default="auto")
///     Algorithm for neighbor search. Options: "auto", "kd_tree", "ball_tree", "brute".
/// p : float, optional (default=2.0)
///     Power parameter for Minkowski metric. p=1 is Manhattan, p=2 is Euclidean.
/// leaf_size : int, optional (default=30)
///     Leaf size for tree-based algorithms.
///
/// Attributes
/// ----------
/// n_features_in_ : int
///     Number of features seen during fit.
/// classes_ : ndarray
///     Unique class labels.
///
/// Examples
/// --------
/// >>> from ferroml.neighbors import KNeighborsClassifier
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [2, 1], [3, 3], [6, 7], [7, 6], [8, 8]])
/// >>> y = np.array([0, 0, 0, 1, 1, 1])
/// >>> model = KNeighborsClassifier(n_neighbors=3)
/// >>> model.fit(X, y)
/// >>> model.predict(X)
#[pyclass(name = "KNeighborsClassifier", module = "ferroml.neighbors")]
pub struct PyKNeighborsClassifier {
    inner: KNeighborsClassifier,
}

#[pymethods]
impl PyKNeighborsClassifier {
    /// Create a new KNeighborsClassifier.
    #[new]
    #[pyo3(signature = (n_neighbors=5, weights="uniform", metric="euclidean", algorithm="auto", p=2.0, leaf_size=30))]
    fn new(
        n_neighbors: usize,
        weights: &str,
        metric: &str,
        algorithm: &str,
        p: f64,
        leaf_size: usize,
    ) -> PyResult<Self> {
        let weights_enum = parse_knn_weights(weights)?;
        let metric_enum = parse_distance_metric(metric, p)?;
        let algorithm_enum = parse_knn_algorithm(algorithm)?;

        let inner = KNeighborsClassifier::new(n_neighbors)
            .with_weights(weights_enum)
            .with_metric(metric_enum)
            .with_algorithm(algorithm_enum)
            .with_leaf_size(leaf_size);

        Ok(Self { inner })
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values. Can be integer or float array.
    ///
    /// Returns
    /// -------
    /// self : KNeighborsClassifier
    ///     Fitted estimator.
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
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Predict class probabilities for samples.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// probas : ndarray of shape (n_samples, n_classes)
    ///     Probability of each class for each sample.
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
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(probas.mapv(|p| p.max(1e-15).ln()).into_pyarray(py))
    }

    /// Get the number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the unique class labels.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let classes = self.inner.classes().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(classes.clone().into_pyarray(py))
    }

    /// Get the number of neighbors.
    #[getter]
    fn n_neighbors(&self) -> usize {
        self.inner.n_neighbors
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
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
            "KNeighborsClassifier(n_neighbors={}, weights={:?}, metric={:?})",
            self.inner.n_neighbors, self.inner.weights, self.inner.metric
        )
    }
}

// =============================================================================
// KNeighborsRegressor
// =============================================================================

/// K-Nearest Neighbors Regressor.
///
/// Regression based on averaging the target values of the k nearest neighbors.
/// Supports weighted averaging where closer neighbors have more influence.
///
/// Parameters
/// ----------
/// n_neighbors : int, optional (default=5)
///     Number of neighbors to use for prediction.
/// weights : str, optional (default="uniform")
///     Weight function used in prediction. "uniform" for equal weights,
///     "distance" for weight by inverse of distance.
/// metric : str, optional (default="euclidean")
///     Distance metric. Options: "euclidean", "manhattan", "minkowski".
/// algorithm : str, optional (default="auto")
///     Algorithm for neighbor search. Options: "auto", "kd_tree", "ball_tree", "brute".
/// p : float, optional (default=2.0)
///     Power parameter for Minkowski metric.
/// leaf_size : int, optional (default=30)
///     Leaf size for tree-based algorithms.
///
/// Attributes
/// ----------
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from ferroml.neighbors import KNeighborsRegressor
/// >>> import numpy as np
/// >>> X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
/// >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
/// >>> model = KNeighborsRegressor(n_neighbors=2)
/// >>> model.fit(X, y)
/// >>> model.predict(X)
#[pyclass(name = "KNeighborsRegressor", module = "ferroml.neighbors")]
pub struct PyKNeighborsRegressor {
    inner: KNeighborsRegressor,
}

#[pymethods]
impl PyKNeighborsRegressor {
    /// Create a new KNeighborsRegressor.
    #[new]
    #[pyo3(signature = (n_neighbors=5, weights="uniform", metric="euclidean", algorithm="auto", p=2.0, leaf_size=30))]
    fn new(
        n_neighbors: usize,
        weights: &str,
        metric: &str,
        algorithm: &str,
        p: f64,
        leaf_size: usize,
    ) -> PyResult<Self> {
        let weights_enum = parse_knn_weights(weights)?;
        let metric_enum = parse_distance_metric(metric, p)?;
        let algorithm_enum = parse_knn_algorithm(algorithm)?;

        let inner = KNeighborsRegressor::new(n_neighbors)
            .with_weights(weights_enum)
            .with_metric(metric_enum)
            .with_algorithm(algorithm_enum)
            .with_leaf_size(leaf_size);

        Ok(Self { inner })
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values. Can be integer or float array.
    ///
    /// Returns
    /// -------
    /// self : KNeighborsRegressor
    ///     Fitted estimator.
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
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Get the number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the number of neighbors.
    #[getter]
    fn n_neighbors(&self) -> usize {
        self.inner.n_neighbors
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
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
            "KNeighborsRegressor(n_neighbors={}, weights={:?}, metric={:?})",
            self.inner.n_neighbors, self.inner.weights, self.inner.metric
        )
    }
}

// =============================================================================
// NearestCentroid
// =============================================================================

/// Nearest Centroid classifier.
///
/// Classification based on the nearest class centroid. Each class is
/// represented by its centroid, and samples are assigned to the class
/// with the nearest centroid.
///
/// Parameters
/// ----------
/// metric : str, optional (default="euclidean")
///     Distance metric. Options: "euclidean", "manhattan", "minkowski".
/// shrink_threshold : float or None, optional (default=None)
///     Threshold for shrinking centroids. If provided, feature values
///     are shrunk towards the overall centroid.
///
/// Attributes
/// ----------
/// centroids_ : ndarray of shape (n_classes, n_features)
///     Centroid of each class.
/// classes_ : ndarray of shape (n_classes,)
///     Unique class labels.
#[pyclass(name = "NearestCentroid", module = "ferroml.neighbors")]
pub struct PyNearestCentroid {
    inner: NearestCentroid,
}

#[pymethods]
impl PyNearestCentroid {
    #[new]
    #[pyo3(signature = (metric="euclidean", shrink_threshold=None))]
    fn new(metric: &str, shrink_threshold: Option<f64>) -> PyResult<Self> {
        let metric_enum = parse_distance_metric(metric, 2.0)?;
        let mut nc = NearestCentroid::new().with_metric(metric_enum);
        if let Some(t) = shrink_threshold {
            nc = nc.with_shrink_threshold(t);
        }
        Ok(Self { inner: nc })
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Get the class centroids.
    #[getter]
    fn centroids_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let centroids = self.inner.centroids().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(centroids.clone().into_pyarray(py))
    }

    /// Get the unique class labels.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let classes = self.inner.classes().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(classes.clone().into_pyarray(py))
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("NearestCentroid(metric={:?})", self.inner.metric)
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the neighbors submodule.
pub fn register_neighbors_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let neighbors_module = PyModule::new(parent_module.py(), "neighbors")?;

    neighbors_module.add_class::<PyKNeighborsClassifier>()?;
    neighbors_module.add_class::<PyKNeighborsRegressor>()?;
    neighbors_module.add_class::<PyNearestCentroid>()?;

    parent_module.add_submodule(&neighbors_module)?;

    Ok(())
}
