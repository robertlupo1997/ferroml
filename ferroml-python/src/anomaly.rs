//! Python bindings for FerroML anomaly detection models
//!
//! This module provides Python wrappers for:
//! - IsolationForest
//! - LocalOutlierFactor

use crate::array_utils::{check_array_finite, to_owned_array_2d};
use ferroml_core::models::isolation_forest::{Contamination, IsolationForest, MaxSamples};
use ferroml_core::models::knn::{DistanceMetric, KNNAlgorithm};
use ferroml_core::models::lof::LocalOutlierFactor;
use ferroml_core::models::OutlierDetector;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// IsolationForest
// ---------------------------------------------------------------------------

/// Isolation Forest anomaly detector.
///
/// Detects anomalies using random recursive partitioning. Anomalies are
/// isolated in fewer splits (shorter average path length).
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=100)
///     Number of isolation trees.
/// max_samples : str or int or float, optional (default="auto")
///     Samples per tree: "auto" = min(256, n_samples), int = fixed count,
///     float in (0,1] = fraction.
/// contamination : str or float, optional (default="auto")
///     "auto" sets offset=-0.5. Float in (0, 0.5] fits threshold on training data.
/// max_features : float, optional (default=1.0)
///     Fraction of features per tree.
/// bootstrap : bool, optional (default=False)
///     Sample with replacement.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Attributes
/// ----------
/// offset_ : float
///     Offset threshold for decision_function. Negative values indicate outliers.
///
/// Examples
/// --------
/// >>> from ferroml.anomaly import IsolationForest
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 2)
/// >>> X = np.vstack([X, [[10, 10], [12, 12]]])  # add outliers
/// >>> model = IsolationForest(random_state=42)
/// >>> model.fit(X)
/// >>> preds = model.predict(X)  # +1 inlier, -1 outlier
#[pyclass(name = "IsolationForest", module = "ferroml.anomaly")]
pub struct PyIsolationForest {
    inner: IsolationForest,
}

#[pymethods]
impl PyIsolationForest {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=false,
        random_state=None,
    ))]
    fn new(
        n_estimators: usize,
        max_samples: &str,
        contamination: &str,
        max_features: f64,
        bootstrap: bool,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let ms = match max_samples {
            "auto" => MaxSamples::Auto,
            _ => {
                // Try parsing as int first, then float
                if let Ok(n) = max_samples.parse::<usize>() {
                    MaxSamples::Count(n)
                } else if let Ok(f) = max_samples.parse::<f64>() {
                    MaxSamples::Fraction(f)
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid max_samples: '{max_samples}'. Use 'auto', int, or float."
                    )));
                }
            }
        };

        let cont = match contamination {
            "auto" => Contamination::Auto,
            _ => {
                let c: f64 = contamination.parse().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("Invalid contamination: '{contamination}'. Use 'auto' or float in (0, 0.5]."),
                    )
                })?;
                Contamination::Proportion(c)
            }
        };

        let mut model = IsolationForest::new(n_estimators)
            .with_max_samples(ms)
            .with_contamination(cont)
            .with_max_features(max_features)
            .with_bootstrap(bootstrap);

        if let Some(seed) = random_state {
            model = model.with_random_state(seed);
        }

        Ok(Self { inner: model })
    }

    /// Fit the model on training data (unsupervised).
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit_unsupervised(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Predict inlier (+1) or outlier (-1) labels.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = self
            .inner
            .predict_outliers(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    /// Fit and predict in one step.
    fn fit_predict<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = slf
            .inner
            .fit_predict_outliers(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    /// Anomaly scores. Lower = more anomalous.
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

    /// Decision function: score_samples - offset. Negative = outlier.
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let decision = self
            .inner
            .decision_function(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(decision.into_pyarray(py))
    }

    /// Get the offset threshold.
    #[getter]
    fn offset_(&self) -> f64 {
        self.inner.offset()
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
        "IsolationForest()".to_string()
    }
}

// ---------------------------------------------------------------------------
// LocalOutlierFactor
// ---------------------------------------------------------------------------

/// Local Outlier Factor for anomaly / novelty detection.
///
/// Measures local density deviation via k-nearest neighbors. Points with
/// substantially lower density than neighbors are outliers.
///
/// Parameters
/// ----------
/// n_neighbors : int, optional (default=20)
///     Number of neighbors for density estimation.
/// contamination : str or float, optional (default="auto")
///     "auto" sets offset=-1.5. Float in (0, 0.5] fits threshold.
/// metric : str, optional (default="euclidean")
///     Distance metric: "euclidean", "manhattan", or "minkowski".
/// algorithm : str, optional (default="auto")
///     KNN algorithm: "auto", "kd_tree", "ball_tree", or "brute".
/// novelty : bool, optional (default=False)
///     If True, predict/score_samples work on new data. If False, use fit_predict.
///
/// Attributes
/// ----------
/// negative_outlier_factor_ : ndarray of shape (n_samples,) or None
///     Negative outlier factor for training samples. Only available after fit().
/// offset_ : float
///     Offset threshold for decision_function.
///
/// Examples
/// --------
/// >>> from ferroml.anomaly import LocalOutlierFactor
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 2)
/// >>> model = LocalOutlierFactor(n_neighbors=20)
/// >>> preds = model.fit_predict(X)  # +1 inlier, -1 outlier
#[pyclass(name = "LocalOutlierFactor", module = "ferroml.anomaly")]
pub struct PyLocalOutlierFactor {
    inner: LocalOutlierFactor,
}

#[pymethods]
impl PyLocalOutlierFactor {
    #[new]
    #[pyo3(signature = (
        n_neighbors=20,
        contamination="auto",
        metric="euclidean",
        algorithm="auto",
        novelty=false,
    ))]
    fn new(
        n_neighbors: usize,
        contamination: &str,
        metric: &str,
        algorithm: &str,
        novelty: bool,
    ) -> PyResult<Self> {
        let cont = match contamination {
            "auto" => Contamination::Auto,
            _ => {
                let c: f64 = contamination.parse().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid contamination: '{contamination}'."
                    ))
                })?;
                Contamination::Proportion(c)
            }
        };

        let dist_metric = match metric {
            "euclidean" => DistanceMetric::Euclidean,
            "manhattan" => DistanceMetric::Manhattan,
            "minkowski" => DistanceMetric::Minkowski(2.0),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown metric: '{metric}'. Use 'euclidean', 'manhattan', or 'minkowski'."
                )));
            }
        };

        let knn_alg = match algorithm {
            "auto" => KNNAlgorithm::Auto,
            "kd_tree" => KNNAlgorithm::KDTree,
            "ball_tree" => KNNAlgorithm::BallTree,
            "brute" => KNNAlgorithm::BruteForce,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown algorithm: '{algorithm}'."
                )));
            }
        };

        let model = LocalOutlierFactor::new(n_neighbors)
            .with_contamination(cont)
            .with_metric(dist_metric)
            .with_algorithm(knn_alg)
            .with_novelty(novelty);

        Ok(Self { inner: model })
    }

    /// Fit the model on training data (unsupervised).
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit_unsupervised(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Fit and predict in one step.
    fn fit_predict<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = slf
            .inner
            .fit_predict_outliers(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    /// Predict inlier (+1) or outlier (-1) labels. Only available when novelty=True.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let preds = self
            .inner
            .predict_outliers(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(preds.into_pyarray(py))
    }

    /// Anomaly scores. Only available when novelty=True.
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

    /// Decision function: score_samples - offset. Only available when novelty=True.
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let decision = self
            .inner
            .decision_function(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(decision.into_pyarray(py))
    }

    /// Get the negative outlier factor for training samples.
    #[getter]
    fn negative_outlier_factor_<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .inner
            .negative_outlier_factor()
            .map(|nof| nof.clone().into_pyarray(py)))
    }

    /// Get the offset threshold.
    #[getter]
    fn offset_(&self) -> f64 {
        self.inner.offset()
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
        "LocalOutlierFactor()".to_string()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register_anomaly_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "anomaly")?;
    m.add_class::<PyIsolationForest>()?;
    m.add_class::<PyLocalOutlierFactor>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
