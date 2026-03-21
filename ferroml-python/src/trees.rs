//! Python bindings for FerroML tree-based models
//!
//! This module provides Python wrappers for:
//! - DecisionTreeClassifier, DecisionTreeRegressor
//! - RandomForestClassifier, RandomForestRegressor
//! - GradientBoostingClassifier, GradientBoostingRegressor
//! - HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
    check_array1_finite, check_array_finite, to_owned_array_1d, to_owned_array_2d,
};
use crate::pickle::{getstate, setstate};
use ferroml_core::models::{
    boosting::{GradientBoostingClassifier, GradientBoostingRegressor, RegressionLoss},
    forest::{MaxFeatures, RandomForestClassifier, RandomForestRegressor},
    hist_boosting::{
        HistGradientBoostingClassifier, HistGradientBoostingRegressor, HistRegressionLoss,
    },
    tree::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion},
    Model,
};
use ferroml_core::onnx::{OnnxConfig, OnnxExportable};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// =============================================================================
// DecisionTreeClassifier
// =============================================================================

/// Decision Tree Classifier using CART algorithm.
///
/// Builds a binary decision tree by recursively partitioning the feature space
/// using the best split based on impurity decrease (Gini or Entropy).
///
/// Parameters
/// ----------
/// criterion : str, optional (default="gini")
///     The function to measure split quality ("gini" or "entropy").
/// max_depth : int or None, optional (default=None)
///     Maximum depth of the tree. None for unlimited depth.
/// min_samples_split : int, optional (default=2)
///     Minimum samples required to split an internal node.
/// min_samples_leaf : int, optional (default=1)
///     Minimum samples required at a leaf node.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Attributes
/// ----------
/// feature_importances_ : ndarray of shape (n_features,)
///     Feature importances computed as impurity decrease.
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from ferroml.trees import DecisionTreeClassifier
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [2, 1], [3, 3], [6, 7], [7, 6], [8, 8]])
/// >>> y = np.array([0, 0, 0, 1, 1, 1])
/// >>> model = DecisionTreeClassifier(max_depth=3)
/// >>> model.fit(X, y)
/// >>> model.predict(X)
#[pyclass(name = "DecisionTreeClassifier", module = "ferroml.trees")]
pub struct PyDecisionTreeClassifier {
    inner: DecisionTreeClassifier,
}

impl PyDecisionTreeClassifier {
    /// Get a reference to the inner model (for use by other Python binding modules).
    pub fn inner_ref(&self) -> &DecisionTreeClassifier {
        &self.inner
    }
}

#[pymethods]
impl PyDecisionTreeClassifier {
    /// Create a new DecisionTreeClassifier.
    #[new]
    #[pyo3(signature = (criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0, random_state=None))]
    fn new(
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        ccp_alpha: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let crit = match criterion.to_lowercase().as_str() {
            "gini" => SplitCriterion::Gini,
            "entropy" => SplitCriterion::Entropy,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "criterion must be 'gini' or 'entropy'",
                ))
            }
        };

        let mut inner = DecisionTreeClassifier::new()
            .with_criterion(crit)
            .with_max_depth(max_depth)
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
            .with_ccp_alpha(ccp_alpha);

        if let Some(seed) = random_state {
            inner = inner.with_random_state(seed);
        }

        Ok(Self { inner })
    }

    /// Fit the model to training data.
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

    /// Predict class probabilities.
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

    /// Get feature importances.
    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    /// Get the number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the tree depth.
    fn get_depth(&self) -> PyResult<usize> {
        self.inner.get_depth().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the number of leaves.
    fn get_n_leaves(&self) -> PyResult<usize> {
        self.inner.get_n_leaves().ok_or_else(|| {
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

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers, R² for regressors.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Compute the decision function (raw scores).
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "DecisionTreeClassifier(criterion={:?}, max_depth={:?})",
            self.inner.criterion, self.inner.max_depth
        )
    }
}

// =============================================================================
// DecisionTreeRegressor
// =============================================================================

/// Decision Tree Regressor using CART algorithm.
///
/// Builds a binary decision tree by recursively partitioning the feature space
/// using the best split based on MSE or MAE reduction.
///
/// Parameters
/// ----------
/// criterion : str, optional (default="mse")
///     The function to measure split quality ("mse" or "mae").
/// max_depth : int or None, optional (default=None)
///     Maximum depth of the tree.
/// min_samples_split : int, optional (default=2)
///     Minimum samples required to split an internal node.
/// min_samples_leaf : int, optional (default=1)
///     Minimum samples required at a leaf node.
///
/// Examples
/// --------
/// >>> from ferroml.trees import DecisionTreeRegressor
/// >>> model = DecisionTreeRegressor(max_depth=5)
/// >>> model.fit(X, y)
/// >>> model.predict(X)
#[pyclass(name = "DecisionTreeRegressor", module = "ferroml.trees")]
pub struct PyDecisionTreeRegressor {
    inner: DecisionTreeRegressor,
}

impl PyDecisionTreeRegressor {
    pub fn inner_ref(&self) -> &DecisionTreeRegressor {
        &self.inner
    }
}

#[pymethods]
impl PyDecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.0, random_state=None))]
    fn new(
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        ccp_alpha: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let crit = match criterion.to_lowercase().as_str() {
            "mse" | "squared_error" => SplitCriterion::Mse,
            "mae" | "absolute_error" => SplitCriterion::Mae,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "criterion must be 'mse' or 'mae'",
                ))
            }
        };

        let mut inner = DecisionTreeRegressor::new()
            .with_criterion(crit)
            .with_max_depth(max_depth)
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
            .with_ccp_alpha(ccp_alpha);

        if let Some(seed) = random_state {
            inner = inner.with_random_state(seed);
        }

        Ok(Self { inner })
    }

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

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

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    fn get_depth(&self) -> PyResult<usize> {
        self.inner.get_depth().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    fn get_n_leaves(&self) -> PyResult<usize> {
        self.inner.get_n_leaves().ok_or_else(|| {
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

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers, R² for regressors.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "DecisionTreeRegressor(criterion={:?}, max_depth={:?})",
            self.inner.criterion, self.inner.max_depth
        )
    }
}

// =============================================================================
// RandomForestClassifier
// =============================================================================

/// Random Forest Classifier using bootstrap aggregating.
///
/// Builds an ensemble of decision trees, each trained on a bootstrap sample
/// of the data, with random feature subsampling at each split.
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=100)
///     Number of trees in the forest.
/// criterion : str, optional (default="gini")
///     Split criterion ("gini" or "entropy").
/// max_depth : int or None, optional (default=None)
///     Maximum depth of trees.
/// min_samples_split : int, optional (default=2)
///     Minimum samples to split a node.
/// min_samples_leaf : int, optional (default=1)
///     Minimum samples at a leaf.
/// max_features : str or None, optional (default="sqrt")
///     Number of features to consider at each split.
///     - "sqrt": sqrt(n_features)
///     - "log2": log2(n_features)
///     - None: all features
/// bootstrap : bool, optional (default=True)
///     Whether to use bootstrap sampling.
/// oob_score : bool, optional (default=True)
///     Whether to compute out-of-bag score.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Attributes
/// ----------
/// feature_importances_ : ndarray
///     Feature importances from impurity decrease.
/// oob_score_ : float
///     Out-of-bag accuracy (if oob_score=True).
///
/// Examples
/// --------
/// >>> from ferroml.trees import RandomForestClassifier
/// >>> model = RandomForestClassifier(n_estimators=100, random_state=42)
/// >>> model.fit(X, y)
/// >>> model.predict(X)
#[pyclass(name = "RandomForestClassifier", module = "ferroml.trees")]
pub struct PyRandomForestClassifier {
    inner: RandomForestClassifier,
}

impl PyRandomForestClassifier {
    pub fn inner_ref(&self) -> &RandomForestClassifier {
        &self.inner
    }
}

#[pymethods]
impl PyRandomForestClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=true, oob_score=true, random_state=None))]
    fn new(
        n_estimators: usize,
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: &str,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let crit = match criterion.to_lowercase().as_str() {
            "gini" => SplitCriterion::Gini,
            "entropy" => SplitCriterion::Entropy,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "criterion must be 'gini' or 'entropy'",
                ))
            }
        };

        let max_feat = match max_features.to_lowercase().as_str() {
            "sqrt" | "auto" => Some(MaxFeatures::Sqrt),
            "log2" => Some(MaxFeatures::Log2),
            "none" | "all" => None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "max_features must be 'sqrt', 'log2', or 'none'",
                ))
            }
        };

        let mut inner = RandomForestClassifier::new()
            .with_n_estimators(n_estimators)
            .with_criterion(crit)
            .with_max_depth(max_depth)
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
            .with_max_features(max_feat)
            .with_bootstrap(bootstrap)
            .with_oob_score(oob_score);

        if let Some(seed) = random_state {
            inner = inner.with_random_state(seed);
        }

        Ok(Self { inner })
    }

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

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

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get out-of-bag score (accuracy).
    #[getter]
    fn oob_score_(&self) -> PyResult<f64> {
        self.inner.oob_score().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "OOB score not available. Ensure oob_score=True and model is fitted.",
            )
        })
    }

    /// Get the number of estimators.
    #[getter]
    fn n_estimators(&self) -> usize {
        self.inner.n_estimators
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

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers, R² for regressors.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Compute the decision function (raw scores).
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "RandomForestClassifier(n_estimators={}, max_depth={:?})",
            self.inner.n_estimators, self.inner.max_depth
        )
    }
}

// =============================================================================
// RandomForestRegressor
// =============================================================================

/// Random Forest Regressor using bootstrap aggregating.
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=100)
///     Number of trees in the forest.
/// max_depth : int or None, optional (default=None)
///     Maximum depth of trees.
/// min_samples_split : int, optional (default=2)
///     Minimum samples to split a node.
/// min_samples_leaf : int, optional (default=1)
///     Minimum samples at a leaf.
/// max_features : str or None, optional (default="sqrt")
///     Features to consider at each split.
/// bootstrap : bool, optional (default=True)
///     Whether to use bootstrap sampling.
/// oob_score : bool, optional (default=True)
///     Whether to compute out-of-bag score.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.trees import RandomForestRegressor
/// >>> model = RandomForestRegressor(n_estimators=100)
/// >>> model.fit(X, y)
#[pyclass(name = "RandomForestRegressor", module = "ferroml.trees")]
pub struct PyRandomForestRegressor {
    inner: RandomForestRegressor,
}

impl PyRandomForestRegressor {
    pub fn inner_ref(&self) -> &RandomForestRegressor {
        &self.inner
    }
}

#[pymethods]
impl PyRandomForestRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", bootstrap=true, oob_score=true, random_state=None))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: &str,
        bootstrap: bool,
        oob_score: bool,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let max_feat = match max_features.to_lowercase().as_str() {
            "sqrt" | "auto" => Some(MaxFeatures::Sqrt),
            "log2" => Some(MaxFeatures::Log2),
            "none" | "all" => None,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "max_features must be 'sqrt', 'log2', or 'none'",
                ))
            }
        };

        let mut inner = RandomForestRegressor::new()
            .with_n_estimators(n_estimators)
            .with_max_depth(max_depth)
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
            .with_max_features(max_feat)
            .with_bootstrap(bootstrap)
            .with_oob_score(oob_score);

        if let Some(seed) = random_state {
            inner = inner.with_random_state(seed);
        }

        Ok(Self { inner })
    }

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

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

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    #[getter]
    fn oob_score_(&self) -> PyResult<f64> {
        self.inner.oob_score().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "OOB score not available. Ensure oob_score=True and model is fitted.",
            )
        })
    }

    #[getter]
    fn n_estimators(&self) -> usize {
        self.inner.n_estimators
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

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers, R² for regressors.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "RandomForestRegressor(n_estimators={}, max_depth={:?})",
            self.inner.n_estimators, self.inner.max_depth
        )
    }
}

// =============================================================================
// GradientBoostingClassifier
// =============================================================================

/// Gradient Boosting Classifier.
///
/// Builds an additive ensemble of decision trees by sequentially fitting
/// trees to the negative gradient of the loss function.
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=100)
///     Number of boosting stages.
/// learning_rate : float, optional (default=0.1)
///     Learning rate shrinks the contribution of each tree.
/// max_depth : int or None, optional (default=3)
///     Maximum depth of individual trees.
/// min_samples_split : int, optional (default=2)
///     Minimum samples to split a node.
/// min_samples_leaf : int, optional (default=1)
///     Minimum samples at a leaf.
/// subsample : float, optional (default=1.0)
///     Fraction of samples used for fitting trees.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Attributes
/// ----------
/// feature_importances_ : ndarray
///     Feature importances from impurity decrease.
/// n_estimators_ : int
///     Actual number of estimators (may differ with early stopping).
///
/// Examples
/// --------
/// >>> from ferroml.trees import GradientBoostingClassifier
/// >>> model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
/// >>> model.fit(X, y)
/// >>> model.predict_proba(X)
#[pyclass(name = "GradientBoostingClassifier", module = "ferroml.trees")]
pub struct PyGradientBoostingClassifier {
    inner: GradientBoostingClassifier,
}

impl PyGradientBoostingClassifier {
    pub fn inner_ref(&self) -> &GradientBoostingClassifier {
        &self.inner
    }
}

#[pymethods]
impl PyGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1.0, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        subsample: f64,
        random_state: Option<u64>,
    ) -> Self {
        let mut inner = GradientBoostingClassifier::new()
            .with_n_estimators(n_estimators)
            .with_learning_rate(learning_rate)
            .with_max_depth(Some(max_depth))
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
            .with_subsample(subsample);

        if let Some(seed) = random_state {
            inner = inner.with_random_state(seed);
        }

        Self { inner }
    }

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

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

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the actual number of estimators fitted.
    #[getter]
    fn n_estimators_(&self) -> PyResult<usize> {
        self.inner.n_estimators_actual().ok_or_else(|| {
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

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers, R² for regressors.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Compute the decision function (raw scores).
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "GradientBoostingClassifier(n_estimators={}, max_depth={:?})",
            self.inner.n_estimators, self.inner.max_depth
        )
    }
}

// =============================================================================
// GradientBoostingRegressor
// =============================================================================

/// Gradient Boosting Regressor.
///
/// Parameters
/// ----------
/// n_estimators : int, optional (default=100)
///     Number of boosting stages.
/// learning_rate : float, optional (default=0.1)
///     Learning rate shrinks the contribution of each tree.
/// loss : str, optional (default="squared_error")
///     Loss function ("squared_error", "absolute_error", or "huber").
/// max_depth : int or None, optional (default=3)
///     Maximum depth of individual trees.
/// min_samples_split : int, optional (default=2)
///     Minimum samples to split a node.
/// min_samples_leaf : int, optional (default=1)
///     Minimum samples at a leaf.
/// subsample : float, optional (default=1.0)
///     Fraction of samples used for fitting trees.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.trees import GradientBoostingRegressor
/// >>> model = GradientBoostingRegressor(n_estimators=100)
/// >>> model.fit(X, y)
#[pyclass(name = "GradientBoostingRegressor", module = "ferroml.trees")]
pub struct PyGradientBoostingRegressor {
    inner: GradientBoostingRegressor,
}

impl PyGradientBoostingRegressor {
    pub fn inner_ref(&self) -> &GradientBoostingRegressor {
        &self.inner
    }
}

#[pymethods]
impl PyGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, loss="squared_error", max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1.0, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        loss: &str,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        subsample: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let loss_fn = match loss.to_lowercase().as_str() {
            "squared_error" | "ls" | "mse" => RegressionLoss::SquaredError,
            "absolute_error" | "lad" | "mae" => RegressionLoss::AbsoluteError,
            "huber" => RegressionLoss::Huber,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "loss must be 'squared_error', 'absolute_error', or 'huber'",
                ))
            }
        };

        let mut inner = GradientBoostingRegressor::new()
            .with_n_estimators(n_estimators)
            .with_learning_rate(learning_rate)
            .with_loss(loss_fn)
            .with_max_depth(Some(max_depth))
            .with_min_samples_split(min_samples_split)
            .with_min_samples_leaf(min_samples_leaf)
            .with_subsample(subsample);

        if let Some(seed) = random_state {
            inner = inner.with_random_state(seed);
        }

        Ok(Self { inner })
    }

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

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

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    #[getter]
    fn n_estimators_(&self) -> PyResult<usize> {
        self.inner.n_estimators_actual().ok_or_else(|| {
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

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers, R² for regressors.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "GradientBoostingRegressor(n_estimators={}, loss={:?})",
            self.inner.n_estimators, self.inner.loss
        )
    }
}

// =============================================================================
// HistGradientBoostingClassifier
// =============================================================================

/// Histogram-based Gradient Boosting Classifier (LightGBM-style).
///
/// Uses histogram-based split finding for O(n) complexity. Supports native
/// missing value handling and monotonic constraints.
///
/// Parameters
/// ----------
/// max_iter : int, optional (default=100)
///     Number of boosting iterations.
/// learning_rate : float, optional (default=0.1)
///     Learning rate.
/// max_depth : int or None, optional (default=None)
///     Maximum depth. None for leaf-wise growth.
/// max_leaf_nodes : int or None, optional (default=31)
///     Maximum number of leaves per tree.
/// max_bins : int, optional (default=255)
///     Maximum number of bins for histogram.
/// min_samples_leaf : int, optional (default=20)
///     Minimum samples at a leaf.
/// l2_regularization : float, optional (default=0.0)
///     L2 regularization.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.trees import HistGradientBoostingClassifier
/// >>> model = HistGradientBoostingClassifier(max_iter=100)
/// >>> model.fit(X, y)
/// >>> model.predict_proba(X)
#[pyclass(name = "HistGradientBoostingClassifier", module = "ferroml.trees")]
pub struct PyHistGradientBoostingClassifier {
    inner: HistGradientBoostingClassifier,
}

#[pymethods]
impl PyHistGradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (max_iter=100, learning_rate=0.1, max_depth=None, max_leaf_nodes=31, max_bins=255, min_samples_leaf=20, l2_regularization=0.0, random_state=None))]
    fn new(
        max_iter: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        max_leaf_nodes: usize,
        max_bins: usize,
        min_samples_leaf: usize,
        l2_regularization: f64,
        random_state: Option<u64>,
    ) -> Self {
        let mut inner = HistGradientBoostingClassifier::new()
            .with_max_iter(max_iter)
            .with_learning_rate(learning_rate)
            .with_max_leaf_nodes(Some(max_leaf_nodes))
            .with_max_bins(max_bins)
            .with_min_samples_leaf(min_samples_leaf)
            .with_l2_regularization(l2_regularization)
            .with_max_depth(max_depth);

        if let Some(seed) = random_state {
            inner = inner.with_random_state(seed);
        }

        Self { inner }
    }

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

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

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
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

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers, R² for regressors.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Compute the decision function (raw scores).
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "HistGradientBoostingClassifier(max_iter={}, learning_rate={})",
            self.inner.max_iter, self.inner.learning_rate
        )
    }
}

// =============================================================================
// HistGradientBoostingRegressor
// =============================================================================

/// Histogram-based Gradient Boosting Regressor (LightGBM-style).
///
/// Parameters
/// ----------
/// max_iter : int, optional (default=100)
///     Number of boosting iterations.
/// learning_rate : float, optional (default=0.1)
///     Learning rate.
/// loss : str, optional (default="squared_error")
///     Loss function ("squared_error", "absolute_error", "huber").
/// max_depth : int or None, optional (default=None)
///     Maximum depth.
/// max_leaf_nodes : int or None, optional (default=31)
///     Maximum number of leaves per tree.
/// max_bins : int, optional (default=255)
///     Maximum number of bins for histogram.
/// min_samples_leaf : int, optional (default=20)
///     Minimum samples at a leaf.
/// l2_regularization : float, optional (default=0.0)
///     L2 regularization.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.trees import HistGradientBoostingRegressor
/// >>> model = HistGradientBoostingRegressor(max_iter=100)
/// >>> model.fit(X, y)
#[pyclass(name = "HistGradientBoostingRegressor", module = "ferroml.trees")]
pub struct PyHistGradientBoostingRegressor {
    inner: HistGradientBoostingRegressor,
}

#[pymethods]
impl PyHistGradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (max_iter=100, learning_rate=0.1, loss="squared_error", max_depth=None, max_leaf_nodes=31, max_bins=255, min_samples_leaf=20, l2_regularization=0.0, random_state=None))]
    fn new(
        max_iter: usize,
        learning_rate: f64,
        loss: &str,
        max_depth: Option<usize>,
        max_leaf_nodes: usize,
        max_bins: usize,
        min_samples_leaf: usize,
        l2_regularization: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let loss_fn = match loss.to_lowercase().as_str() {
            "squared_error" | "mse" => HistRegressionLoss::SquaredError,
            "absolute_error" | "mae" => HistRegressionLoss::AbsoluteError,
            "huber" => HistRegressionLoss::Huber,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "loss must be 'squared_error', 'absolute_error', or 'huber'",
                ))
            }
        };

        let mut inner = HistGradientBoostingRegressor::new()
            .with_max_iter(max_iter)
            .with_learning_rate(learning_rate)
            .with_loss(loss_fn)
            .with_max_leaf_nodes(Some(max_leaf_nodes))
            .with_max_bins(max_bins)
            .with_min_samples_leaf(min_samples_leaf)
            .with_l2_regularization(l2_regularization)
            .with_max_depth(max_depth);

        if let Some(seed) = random_state {
            inner = inner.with_random_state(seed);
        }

        Ok(Self { inner })
    }

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

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

    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
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

    /// Evaluate the model on test data.
    ///
    /// Returns accuracy for classifiers, R² for regressors.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "HistGradientBoostingRegressor(max_iter={}, loss={:?})",
            self.inner.max_iter, self.inner.loss
        )
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the trees submodule.
pub fn register_trees_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let trees_module = PyModule::new(parent_module.py(), "trees")?;

    // Decision Trees
    trees_module.add_class::<PyDecisionTreeClassifier>()?;
    trees_module.add_class::<PyDecisionTreeRegressor>()?;

    // Random Forests
    trees_module.add_class::<PyRandomForestClassifier>()?;
    trees_module.add_class::<PyRandomForestRegressor>()?;

    // Gradient Boosting
    trees_module.add_class::<PyGradientBoostingClassifier>()?;
    trees_module.add_class::<PyGradientBoostingRegressor>()?;

    // Histogram-based Gradient Boosting
    trees_module.add_class::<PyHistGradientBoostingClassifier>()?;
    trees_module.add_class::<PyHistGradientBoostingRegressor>()?;

    parent_module.add_submodule(&trees_module)?;

    Ok(())
}
