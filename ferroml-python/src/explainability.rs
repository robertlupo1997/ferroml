//! Python bindings for FerroML explainability module
//!
//! This module provides Python wrappers for:
//! - **TreeSHAP**: Exact SHAP values via `TreeExplainer` (DT, RF, GB models)
//! - **KernelSHAP**: Model-agnostic approximate SHAP values via `KernelExplainer`
//!   (RF, DT, GB, Linear, Logistic, ExtraTrees -- 10 model types)
//! - **Permutation Importance**: Model-agnostic feature importance with CI
//!   (RF, DT, GB, Linear, Logistic, ExtraTrees -- 10 model types)
//! - **Partial Dependence (PDP)**: 1D and 2D marginal effect visualization
//!   (RF, DT, GB, Linear -- 7+ model types)
//! - **ICE Curves**: Individual Conditional Expectation with centering and derivatives
//!   (RF, DT, GB, Linear -- 4 model types)
//! - **H-Statistic**: Friedman's feature interaction detection
//!   (RF, GB + pairwise interaction matrix)
//!
//! ## Design
//!
//! Explainability functions in ferroml-core are generic over `Model`. Since PyO3
//! cannot directly express Rust trait objects, we provide concrete model wrappers
//! and typed function variants for each supported model type.

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use ferroml_core::explainability::{
    h_statistic, h_statistic_matrix, individual_conditional_expectation, partial_dependence,
    partial_dependence_2d, permutation_importance, GridMethod, HStatisticConfig, ICEConfig,
    KernelExplainer, KernelSHAPConfig, PDPResult, PermutationImportanceResult, SHAPBatchResult,
    TreeExplainer,
};
use ferroml_core::metrics::{accuracy, mae, mse, r2_score};
use ferroml_core::models::Model;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// =============================================================================
// TreeExplainer (TreeSHAP)
// =============================================================================

/// TreeExplainer computes exact SHAP values for tree-based models.
///
/// Uses the TreeSHAP algorithm (O(TLD^2)) for exact Shapley value computation.
/// Supports DecisionTree, RandomForest, and GradientBoosting models.
///
/// Parameters
/// ----------
/// model : fitted tree-based model
///     The model to explain. Must be one of: DecisionTreeClassifier,
///     DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor,
///     GradientBoostingRegressor.
///
/// Examples
/// --------
/// >>> from ferroml.trees import RandomForestRegressor
/// >>> from ferroml.explainability import TreeExplainer
/// >>> model = RandomForestRegressor(n_estimators=10, random_state=42)
/// >>> model.fit(X_train, y_train)
/// >>> explainer = TreeExplainer.from_random_forest_regressor(model)
/// >>> result = explainer.explain(X_test[0])
/// >>> result['shap_values']
#[pyclass(name = "TreeExplainer", module = "ferroml.explainability")]
pub struct PyTreeExplainer {
    inner: TreeExplainer,
}

#[pymethods]
impl PyTreeExplainer {
    /// Create a TreeExplainer from a fitted DecisionTreeRegressor.
    #[staticmethod]
    fn from_decision_tree_regressor(
        model: &crate::trees::PyDecisionTreeRegressor,
    ) -> PyResult<Self> {
        let inner = TreeExplainer::from_decision_tree_regressor(model.inner_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a TreeExplainer from a fitted DecisionTreeClassifier.
    #[staticmethod]
    fn from_decision_tree_classifier(
        model: &crate::trees::PyDecisionTreeClassifier,
    ) -> PyResult<Self> {
        let inner = TreeExplainer::from_decision_tree_classifier(model.inner_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a TreeExplainer from a fitted RandomForestRegressor.
    #[staticmethod]
    fn from_random_forest_regressor(
        model: &crate::trees::PyRandomForestRegressor,
    ) -> PyResult<Self> {
        let inner = TreeExplainer::from_random_forest_regressor(model.inner_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a TreeExplainer from a fitted RandomForestClassifier.
    #[staticmethod]
    fn from_random_forest_classifier(
        model: &crate::trees::PyRandomForestClassifier,
    ) -> PyResult<Self> {
        let inner = TreeExplainer::from_random_forest_classifier(model.inner_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a TreeExplainer from a fitted GradientBoostingRegressor.
    #[staticmethod]
    fn from_gradient_boosting_regressor(
        model: &crate::trees::PyGradientBoostingRegressor,
    ) -> PyResult<Self> {
        let inner = TreeExplainer::from_gradient_boosting_regressor(model.inner_ref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Explain a single prediction.
    ///
    /// Parameters
    /// ----------
    /// x : array-like of shape (n_features,)
    ///     Feature values for a single sample.
    ///
    /// Returns
    /// -------
    /// dict with keys:
    ///     'base_value' : float - expected model output
    ///     'shap_values' : ndarray - SHAP value for each feature
    ///     'feature_values' : ndarray - input feature values
    fn explain<'py>(&self, py: Python<'py>, x: PyReadonlyArray1<'py, f64>) -> PyResult<PyObject> {
        let x_vec: Vec<f64> = x.as_array().to_vec();

        let result = self
            .inner
            .explain(&x_vec)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("base_value", result.base_value)?;
        dict.set_item("shap_values", result.shap_values.into_pyarray(py))?;
        dict.set_item("feature_values", result.feature_values.into_pyarray(py))?;

        Ok(dict.into())
    }

    /// Explain multiple predictions.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Feature matrix.
    ///
    /// Returns
    /// -------
    /// dict with keys:
    ///     'base_value' : float - expected model output
    ///     'shap_values' : ndarray of shape (n_samples, n_features)
    ///     'feature_values' : ndarray of shape (n_samples, n_features)
    fn explain_batch<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyObject> {
        let x_arr = to_owned_array_2d(x);

        let result = self
            .inner
            .explain_batch(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("base_value", result.base_value)?;
        dict.set_item("shap_values", result.shap_values.into_pyarray(py))?;
        dict.set_item("feature_values", result.feature_values.into_pyarray(py))?;

        Ok(dict.into())
    }

    /// Get the number of trees in the explainer.
    #[getter]
    fn n_trees(&self) -> usize {
        self.inner.n_trees()
    }

    /// Get the number of features.
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    fn __repr__(&self) -> String {
        format!(
            "TreeExplainer(n_trees={}, n_features={})",
            self.inner.n_trees(),
            self.inner.n_features()
        )
    }
}

// =============================================================================
// Permutation importance helpers
// =============================================================================

/// Internal helper to run permutation importance on any model
fn run_permutation_importance<M: Model>(
    model: &M,
    x: &ndarray::Array2<f64>,
    y: &ndarray::Array1<f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> Result<PermutationImportanceResult, ferroml_core::FerroError> {
    match metric {
        "accuracy" => permutation_importance(model, x, y, accuracy, n_repeats, random_state),
        "r2" => permutation_importance(model, x, y, r2_score, n_repeats, random_state),
        "mse" | "neg_mse" => permutation_importance(
            model,
            x,
            y,
            |y_true, y_pred| mse(y_true, y_pred).map(|v| -v),
            n_repeats,
            random_state,
        ),
        "mae" | "neg_mae" => permutation_importance(
            model,
            x,
            y,
            |y_true, y_pred| mae(y_true, y_pred).map(|v| -v),
            n_repeats,
            random_state,
        ),
        _ => Err(ferroml_core::FerroError::invalid_input(format!(
            "Unknown metric: '{}'. Use 'accuracy', 'r2', 'mse', or 'mae'.",
            metric
        ))),
    }
}

fn perm_result_to_dict(py: Python<'_>, result: PermutationImportanceResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("importances_mean", result.importances_mean.into_pyarray(py))?;
    dict.set_item("importances_std", result.importances_std.into_pyarray(py))?;
    dict.set_item("ci_lower", result.ci_lower.into_pyarray(py))?;
    dict.set_item("ci_upper", result.ci_upper.into_pyarray(py))?;
    dict.set_item("baseline_score", result.baseline_score)?;
    dict.set_item("n_repeats", result.n_repeats)?;
    Ok(dict.into())
}

// =============================================================================
// Permutation importance for all model types
// =============================================================================

/// Compute permutation importance for a RandomForestRegressor.
///
/// Parameters
/// ----------
/// model : RandomForestRegressor
///     Fitted model.
/// X : ndarray of shape (n_samples, n_features)
///     Test data.
/// y : ndarray of shape (n_samples,)
///     True labels/values.
/// metric : str, optional (default="r2")
///     Scoring metric: "accuracy", "r2", "mse", "mae".
/// n_repeats : int, optional (default=10)
///     Number of times to permute each feature.
/// random_state : int, optional
///     Random seed.
///
/// Returns
/// -------
/// dict with keys: 'importances_mean', 'importances_std', 'ci_lower', 'ci_upper', 'baseline_score', 'n_repeats'
#[pyfunction]
#[pyo3(name = "permutation_importance_rf_reg", signature = (model, x, y, metric="r2", n_repeats=10, random_state=None))]
fn py_permutation_importance_rf_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestRegressor,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for a RandomForestClassifier.
#[pyfunction]
#[pyo3(name = "permutation_importance_rf_clf", signature = (model, x, y, metric="accuracy", n_repeats=10, random_state=None))]
fn py_permutation_importance_rf_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestClassifier,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for a DecisionTreeRegressor.
#[pyfunction]
#[pyo3(name = "permutation_importance_dt_reg", signature = (model, x, y, metric="r2", n_repeats=10, random_state=None))]
fn py_permutation_importance_dt_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyDecisionTreeRegressor,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for a DecisionTreeClassifier.
#[pyfunction]
#[pyo3(name = "permutation_importance_dt_clf", signature = (model, x, y, metric="accuracy", n_repeats=10, random_state=None))]
fn py_permutation_importance_dt_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyDecisionTreeClassifier,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for a GradientBoostingRegressor.
#[pyfunction]
#[pyo3(name = "permutation_importance_gb_reg", signature = (model, x, y, metric="r2", n_repeats=10, random_state=None))]
fn py_permutation_importance_gb_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingRegressor,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for a GradientBoostingClassifier.
#[pyfunction]
#[pyo3(name = "permutation_importance_gb_clf", signature = (model, x, y, metric="accuracy", n_repeats=10, random_state=None))]
fn py_permutation_importance_gb_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingClassifier,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for a LinearRegression.
#[pyfunction]
#[pyo3(name = "permutation_importance_linear", signature = (model, x, y, metric="r2", n_repeats=10, random_state=None))]
fn py_permutation_importance_linear<'py>(
    py: Python<'py>,
    model: &crate::linear::PyLinearRegression,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for a LogisticRegression.
#[pyfunction]
#[pyo3(name = "permutation_importance_logistic", signature = (model, x, y, metric="accuracy", n_repeats=10, random_state=None))]
fn py_permutation_importance_logistic<'py>(
    py: Python<'py>,
    model: &crate::linear::PyLogisticRegression,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for an ExtraTreesClassifier.
#[pyfunction]
#[pyo3(name = "permutation_importance_et_clf", signature = (model, x, y, metric="accuracy", n_repeats=10, random_state=None))]
fn py_permutation_importance_et_clf<'py>(
    py: Python<'py>,
    model: &crate::ensemble::PyExtraTreesClassifier,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

/// Compute permutation importance for an ExtraTreesRegressor.
#[pyfunction]
#[pyo3(name = "permutation_importance_et_reg", signature = (model, x, y, metric="r2", n_repeats=10, random_state=None))]
fn py_permutation_importance_et_reg<'py>(
    py: Python<'py>,
    model: &crate::ensemble::PyExtraTreesRegressor,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    metric: &str,
    n_repeats: usize,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    let result = run_permutation_importance(
        model.inner_ref(),
        &x_arr,
        &y_arr,
        metric,
        n_repeats,
        random_state,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    perm_result_to_dict(py, result)
}

// =============================================================================
// Partial Dependence (using concrete model types)
// =============================================================================

fn run_pdp<M: Model>(
    model: &M,
    x: &ndarray::Array2<f64>,
    feature_idx: usize,
    grid_resolution: usize,
) -> Result<PDPResult, ferroml_core::FerroError> {
    partial_dependence(
        model,
        x,
        feature_idx,
        grid_resolution,
        GridMethod::Percentile,
        false,
    )
}

fn pdp_result_to_dict(py: Python<'_>, result: PDPResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("grid_values", result.grid_values.into_pyarray(py))?;
    dict.set_item("pdp_values", result.pdp_values.into_pyarray(py))?;
    dict.set_item("pdp_std", result.pdp_std.into_pyarray(py))?;
    dict.set_item("feature_idx", result.feature_idx)?;
    Ok(dict.into())
}

/// Compute partial dependence for a RandomForestRegressor.
///
/// Parameters
/// ----------
/// model : RandomForestRegressor
///     Fitted model.
/// X : ndarray of shape (n_samples, n_features)
///     Data to compute PDP on.
/// feature_idx : int
///     Index of the feature.
/// grid_resolution : int, optional (default=50)
///     Number of grid points.
///
/// Returns
/// -------
/// dict with keys: 'grid_values', 'pdp_values', 'pdp_std', 'feature_idx'
#[pyfunction]
#[pyo3(name = "partial_dependence_rf_reg", signature = (model, x, feature_idx, grid_resolution=50))]
fn py_partial_dependence_rf_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp(model.inner_ref(), &x_arr, feature_idx, grid_resolution)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_result_to_dict(py, result)
}

/// Compute partial dependence for a RandomForestClassifier.
#[pyfunction]
#[pyo3(name = "partial_dependence_rf_clf", signature = (model, x, feature_idx, grid_resolution=50))]
fn py_partial_dependence_rf_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestClassifier,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp(model.inner_ref(), &x_arr, feature_idx, grid_resolution)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_result_to_dict(py, result)
}

/// Compute partial dependence for a GradientBoostingRegressor.
#[pyfunction]
#[pyo3(name = "partial_dependence_gb_reg", signature = (model, x, feature_idx, grid_resolution=50))]
fn py_partial_dependence_gb_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp(model.inner_ref(), &x_arr, feature_idx, grid_resolution)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_result_to_dict(py, result)
}

/// Compute partial dependence for a GradientBoostingClassifier.
#[pyfunction]
#[pyo3(name = "partial_dependence_gb_clf", signature = (model, x, feature_idx, grid_resolution=50))]
fn py_partial_dependence_gb_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingClassifier,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp(model.inner_ref(), &x_arr, feature_idx, grid_resolution)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_result_to_dict(py, result)
}

/// Compute partial dependence for a DecisionTreeRegressor.
#[pyfunction]
#[pyo3(name = "partial_dependence_dt_reg", signature = (model, x, feature_idx, grid_resolution=50))]
fn py_partial_dependence_dt_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyDecisionTreeRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp(model.inner_ref(), &x_arr, feature_idx, grid_resolution)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_result_to_dict(py, result)
}

/// Compute partial dependence for a DecisionTreeClassifier.
#[pyfunction]
#[pyo3(name = "partial_dependence_dt_clf", signature = (model, x, feature_idx, grid_resolution=50))]
fn py_partial_dependence_dt_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyDecisionTreeClassifier,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp(model.inner_ref(), &x_arr, feature_idx, grid_resolution)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_result_to_dict(py, result)
}

/// Compute partial dependence for a LinearRegression.
#[pyfunction]
#[pyo3(name = "partial_dependence_linear", signature = (model, x, feature_idx, grid_resolution=50))]
fn py_partial_dependence_linear<'py>(
    py: Python<'py>,
    model: &crate::linear::PyLinearRegression,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp(model.inner_ref(), &x_arr, feature_idx, grid_resolution)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_result_to_dict(py, result)
}

// =============================================================================
// 2D Partial Dependence (feature interaction PDP)
// =============================================================================

fn run_pdp_2d<M: Model>(
    model: &M,
    x: &ndarray::Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    grid_resolution: usize,
) -> Result<ferroml_core::explainability::PDP2DResult, ferroml_core::FerroError> {
    partial_dependence_2d(
        model,
        x,
        feature_idx_1,
        feature_idx_2,
        grid_resolution,
        GridMethod::Percentile,
    )
}

fn pdp_2d_result_to_dict(
    py: Python<'_>,
    result: ferroml_core::explainability::PDP2DResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("grid_values_1", result.grid_values_1.into_pyarray(py))?;
    dict.set_item("grid_values_2", result.grid_values_2.into_pyarray(py))?;
    dict.set_item("pdp_values", result.pdp_values.into_pyarray(py))?;
    dict.set_item("feature_idx_1", result.feature_idx_1)?;
    dict.set_item("feature_idx_2", result.feature_idx_2)?;
    Ok(dict.into())
}

/// Compute 2D partial dependence for a RandomForestRegressor.
///
/// Computes the joint effect of two features on the model prediction.
///
/// Parameters
/// ----------
/// model : RandomForestRegressor
///     Fitted model.
/// X : ndarray of shape (n_samples, n_features)
///     Data to compute PDP on.
/// feature_idx_1 : int
///     Index of the first feature.
/// feature_idx_2 : int
///     Index of the second feature.
/// grid_resolution : int, optional (default=20)
///     Number of grid points per dimension.
///
/// Returns
/// -------
/// dict with keys: 'grid_values_1', 'grid_values_2', 'pdp_values', 'feature_idx_1', 'feature_idx_2'
#[pyfunction]
#[pyo3(name = "partial_dependence_2d_rf_reg", signature = (model, x, feature_idx_1, feature_idx_2, grid_resolution=20))]
fn py_partial_dependence_2d_rf_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp_2d(
        model.inner_ref(),
        &x_arr,
        feature_idx_1,
        feature_idx_2,
        grid_resolution,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_2d_result_to_dict(py, result)
}

/// Compute 2D partial dependence for a GradientBoostingRegressor.
#[pyfunction]
#[pyo3(name = "partial_dependence_2d_gb_reg", signature = (model, x, feature_idx_1, feature_idx_2, grid_resolution=20))]
fn py_partial_dependence_2d_gb_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    grid_resolution: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_pdp_2d(
        model.inner_ref(),
        &x_arr,
        feature_idx_1,
        feature_idx_2,
        grid_resolution,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    pdp_2d_result_to_dict(py, result)
}

// =============================================================================
// Individual Conditional Expectation (ICE)
// =============================================================================

fn run_ice<M: Model>(
    model: &M,
    x: &ndarray::Array2<f64>,
    feature_idx: usize,
    n_grid_points: usize,
    center: bool,
    compute_derivative: bool,
) -> Result<ferroml_core::explainability::ICEResult, ferroml_core::FerroError> {
    let mut config = ICEConfig::new().with_n_grid_points(n_grid_points);
    if center {
        config = config.with_centering(0);
    }
    if compute_derivative {
        config = config.with_derivative();
    }
    individual_conditional_expectation(model, x, feature_idx, config)
}

fn ice_result_to_dict(
    py: Python<'_>,
    result: ferroml_core::explainability::ICEResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("grid_values", result.grid_values.into_pyarray(py))?;
    dict.set_item("ice_curves", result.ice_curves.into_pyarray(py))?;
    dict.set_item("pdp_values", result.pdp_values.into_pyarray(py))?;
    if let Some(centered) = result.centered_ice {
        dict.set_item("centered_ice", centered.into_pyarray(py))?;
    }
    if let Some(centered_pdp) = result.centered_pdp {
        dict.set_item("centered_pdp", centered_pdp.into_pyarray(py))?;
    }
    if let Some(derivative) = result.derivative_ice {
        dict.set_item("derivative_ice", derivative.into_pyarray(py))?;
    }
    dict.set_item("feature_idx", result.feature_idx)?;
    Ok(dict.into())
}

/// Compute ICE curves for a RandomForestRegressor.
///
/// Individual Conditional Expectation shows how each sample's prediction
/// changes as a feature value varies, unlike PDP which shows the average.
///
/// Parameters
/// ----------
/// model : RandomForestRegressor
///     Fitted model.
/// X : ndarray of shape (n_samples, n_features)
///     Data to compute ICE on.
/// feature_idx : int
///     Index of the feature.
/// n_grid_points : int, optional (default=50)
///     Number of grid points.
/// center : bool, optional (default=False)
///     Whether to compute centered ICE (c-ICE).
/// compute_derivative : bool, optional (default=False)
///     Whether to compute derivative ICE (d-ICE).
///
/// Returns
/// -------
/// dict with keys: 'grid_values', 'ice_curves', 'pdp_values', 'feature_idx',
///     optionally 'centered_ice', 'centered_pdp', 'derivative_ice'
#[pyfunction]
#[pyo3(name = "ice_rf_reg", signature = (model, x, feature_idx, n_grid_points=50, center=false, compute_derivative=false))]
fn py_ice_rf_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    n_grid_points: usize,
    center: bool,
    compute_derivative: bool,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_ice(
        model.inner_ref(),
        &x_arr,
        feature_idx,
        n_grid_points,
        center,
        compute_derivative,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    ice_result_to_dict(py, result)
}

/// Compute ICE curves for a GradientBoostingRegressor.
#[pyfunction]
#[pyo3(name = "ice_gb_reg", signature = (model, x, feature_idx, n_grid_points=50, center=false, compute_derivative=false))]
fn py_ice_gb_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    n_grid_points: usize,
    center: bool,
    compute_derivative: bool,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_ice(
        model.inner_ref(),
        &x_arr,
        feature_idx,
        n_grid_points,
        center,
        compute_derivative,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    ice_result_to_dict(py, result)
}

/// Compute ICE curves for a DecisionTreeRegressor.
#[pyfunction]
#[pyo3(name = "ice_dt_reg", signature = (model, x, feature_idx, n_grid_points=50, center=false, compute_derivative=false))]
fn py_ice_dt_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyDecisionTreeRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    n_grid_points: usize,
    center: bool,
    compute_derivative: bool,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_ice(
        model.inner_ref(),
        &x_arr,
        feature_idx,
        n_grid_points,
        center,
        compute_derivative,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    ice_result_to_dict(py, result)
}

/// Compute ICE curves for a LinearRegression.
#[pyfunction]
#[pyo3(name = "ice_linear", signature = (model, x, feature_idx, n_grid_points=50, center=false, compute_derivative=false))]
fn py_ice_linear<'py>(
    py: Python<'py>,
    model: &crate::linear::PyLinearRegression,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx: usize,
    n_grid_points: usize,
    center: bool,
    compute_derivative: bool,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_ice(
        model.inner_ref(),
        &x_arr,
        feature_idx,
        n_grid_points,
        center,
        compute_derivative,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    ice_result_to_dict(py, result)
}

// =============================================================================
// H-Statistic (Feature Interaction Detection)
// =============================================================================

fn run_h_statistic<M: Model>(
    model: &M,
    x: &ndarray::Array2<f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
) -> Result<ferroml_core::explainability::HStatisticResult, ferroml_core::FerroError> {
    let config = HStatisticConfig::new().with_grid_points(n_grid_points);
    h_statistic(model, x, feature_idx_1, feature_idx_2, config)
}

fn h_stat_result_to_dict(
    py: Python<'_>,
    result: ferroml_core::explainability::HStatisticResult,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("h_squared", result.h_squared)?;
    dict.set_item("h_statistic", result.h_statistic)?;
    dict.set_item("feature_idx_1", result.feature_idx_1)?;
    dict.set_item("feature_idx_2", result.feature_idx_2)?;
    if let Some(ci) = result.ci {
        dict.set_item("ci_lower", ci.0)?;
        dict.set_item("ci_upper", ci.1)?;
    }
    Ok(dict.into())
}

/// Compute H-statistic for feature interaction between two features in a RandomForestRegressor.
///
/// Friedman's H-statistic quantifies how much of the joint effect of two features
/// is due to their interaction (as opposed to independent additive effects).
///
/// Parameters
/// ----------
/// model : RandomForestRegressor
///     Fitted model.
/// X : ndarray of shape (n_samples, n_features)
///     Data for computing the H-statistic.
/// feature_idx_1 : int
///     Index of the first feature.
/// feature_idx_2 : int
///     Index of the second feature.
/// n_grid_points : int, optional (default=20)
///     Number of grid points for PDP computation.
///
/// Returns
/// -------
/// dict with keys: 'h_squared', 'h_statistic', 'feature_idx_1', 'feature_idx_2'
///     Optionally 'ci_lower', 'ci_upper' if bootstrap was enabled.
#[pyfunction]
#[pyo3(name = "h_statistic_rf_reg", signature = (model, x, feature_idx_1, feature_idx_2, n_grid_points=20))]
fn py_h_statistic_rf_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_h_statistic(
        model.inner_ref(),
        &x_arr,
        feature_idx_1,
        feature_idx_2,
        n_grid_points,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    h_stat_result_to_dict(py, result)
}

/// Compute H-statistic for a GradientBoostingRegressor.
#[pyfunction]
#[pyo3(name = "h_statistic_gb_reg", signature = (model, x, feature_idx_1, feature_idx_2, n_grid_points=20))]
fn py_h_statistic_gb_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_idx_1: usize,
    feature_idx_2: usize,
    n_grid_points: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = run_h_statistic(
        model.inner_ref(),
        &x_arr,
        feature_idx_1,
        feature_idx_2,
        n_grid_points,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    h_stat_result_to_dict(py, result)
}

/// Compute the pairwise H-statistic interaction matrix for a RandomForestRegressor.
///
/// Computes H-statistics for all pairs of features (or a specified subset),
/// returning a matrix of interaction strengths.
///
/// Parameters
/// ----------
/// model : RandomForestRegressor
///     Fitted model.
/// X : ndarray of shape (n_samples, n_features)
///     Data for computing the H-statistic.
/// feature_indices : list of int, optional
///     Subset of feature indices to compute. None for all features.
/// n_grid_points : int, optional (default=20)
///     Number of grid points per dimension.
///
/// Returns
/// -------
/// dict with keys: 'h_squared_matrix', 'feature_indices', 'n_samples'
#[pyfunction]
#[pyo3(name = "h_statistic_matrix_rf_reg", signature = (model, x, feature_indices=None, n_grid_points=20))]
fn py_h_statistic_matrix_rf_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestRegressor,
    x: PyReadonlyArray2<'py, f64>,
    feature_indices: Option<Vec<usize>>,
    n_grid_points: usize,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let config = HStatisticConfig::new().with_grid_points(n_grid_points);
    let result = h_statistic_matrix(
        model.inner_ref(),
        &x_arr,
        feature_indices.as_deref(),
        config,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("h_squared_matrix", result.h_squared_matrix.into_pyarray(py))?;
    dict.set_item("feature_indices", result.feature_indices)?;
    dict.set_item("n_samples", result.n_samples)?;
    Ok(dict.into())
}

// =============================================================================
// KernelSHAP (model-agnostic SHAP values)
// =============================================================================

/// Internal helper to run KernelSHAP on any model and return a batch result.
///
/// Creates a KernelExplainer within the function scope so the lifetime of the
/// model borrow is bounded by the call. Always uses explain_batch for uniformity.
fn run_kernel_shap<M: Model>(
    model: &M,
    background: &ndarray::Array2<f64>,
    x: &ndarray::Array2<f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> Result<SHAPBatchResult, ferroml_core::FerroError> {
    let mut config = KernelSHAPConfig::new();
    if let Some(n) = n_samples {
        config = config.with_n_samples(n);
    }
    if let Some(seed) = random_state {
        config = config.with_random_state(seed);
    }
    let explainer = KernelExplainer::new(model, background, config)?;
    explainer.explain_batch(x)
}

fn kernel_shap_batch_result_to_dict(py: Python<'_>, result: SHAPBatchResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("base_value", result.base_value)?;
    dict.set_item("shap_values", result.shap_values.into_pyarray(py))?;
    dict.set_item("feature_values", result.feature_values.into_pyarray(py))?;
    Ok(dict.into())
}

/// Compute KernelSHAP values for a RandomForestRegressor.
///
/// KernelSHAP is a model-agnostic method for computing approximate Shapley values.
/// It uses weighted linear regression over feature coalitions.
///
/// Parameters
/// ----------
/// model : RandomForestRegressor
///     Fitted model.
/// background : ndarray of shape (n_background, n_features)
///     Background dataset for computing expected values.
/// x : ndarray of shape (n_samples, n_features)
///     Samples to explain.
/// n_samples : int, optional
///     Number of coalition samples for approximation. Default: 2*n_features + 2048.
/// random_state : int, optional
///     Random seed for reproducibility.
///
/// Returns
/// -------
/// dict with keys: 'base_value', 'shap_values', 'feature_values'
#[pyfunction]
#[pyo3(name = "kernel_shap_rf_reg", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_rf_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestRegressor,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for a RandomForestClassifier.
#[pyfunction]
#[pyo3(name = "kernel_shap_rf_clf", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_rf_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyRandomForestClassifier,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for a DecisionTreeRegressor.
#[pyfunction]
#[pyo3(name = "kernel_shap_dt_reg", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_dt_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyDecisionTreeRegressor,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for a DecisionTreeClassifier.
#[pyfunction]
#[pyo3(name = "kernel_shap_dt_clf", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_dt_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyDecisionTreeClassifier,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for a GradientBoostingRegressor.
#[pyfunction]
#[pyo3(name = "kernel_shap_gb_reg", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_gb_reg<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingRegressor,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for a GradientBoostingClassifier.
#[pyfunction]
#[pyo3(name = "kernel_shap_gb_clf", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_gb_clf<'py>(
    py: Python<'py>,
    model: &crate::trees::PyGradientBoostingClassifier,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for a LinearRegression.
#[pyfunction]
#[pyo3(name = "kernel_shap_linear", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_linear<'py>(
    py: Python<'py>,
    model: &crate::linear::PyLinearRegression,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for a LogisticRegression.
#[pyfunction]
#[pyo3(name = "kernel_shap_logistic", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_logistic<'py>(
    py: Python<'py>,
    model: &crate::linear::PyLogisticRegression,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for an ExtraTreesClassifier.
#[pyfunction]
#[pyo3(name = "kernel_shap_et_clf", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_et_clf<'py>(
    py: Python<'py>,
    model: &crate::ensemble::PyExtraTreesClassifier,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

/// Compute KernelSHAP values for an ExtraTreesRegressor.
#[pyfunction]
#[pyo3(name = "kernel_shap_et_reg", signature = (model, background, x, n_samples=None, random_state=None))]
fn py_kernel_shap_et_reg<'py>(
    py: Python<'py>,
    model: &crate::ensemble::PyExtraTreesRegressor,
    background: PyReadonlyArray2<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    n_samples: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> {
    let bg_arr = to_owned_array_2d(background);
    let x_arr = to_owned_array_2d(x);
    let result = run_kernel_shap(model.inner_ref(), &bg_arr, &x_arr, n_samples, random_state)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    kernel_shap_batch_result_to_dict(py, result)
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the explainability submodule.
pub fn register_explainability_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "explainability")?;

    // TreeSHAP
    m.add_class::<PyTreeExplainer>()?;

    // Permutation importance functions (all model types)
    m.add_function(wrap_pyfunction!(py_permutation_importance_rf_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_rf_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_dt_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_dt_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_gb_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_gb_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_linear, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_logistic, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_et_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_importance_et_reg, &m)?)?;

    // Partial dependence functions (multiple model types)
    m.add_function(wrap_pyfunction!(py_partial_dependence_rf_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_dependence_rf_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_dependence_gb_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_dependence_gb_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_dependence_dt_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_dependence_dt_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_dependence_linear, &m)?)?;

    // 2D Partial dependence
    m.add_function(wrap_pyfunction!(py_partial_dependence_2d_rf_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_dependence_2d_gb_reg, &m)?)?;

    // ICE (Individual Conditional Expectation)
    m.add_function(wrap_pyfunction!(py_ice_rf_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_ice_gb_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_ice_dt_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_ice_linear, &m)?)?;

    // H-statistic (feature interaction detection)
    m.add_function(wrap_pyfunction!(py_h_statistic_rf_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_h_statistic_gb_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_h_statistic_matrix_rf_reg, &m)?)?;

    // KernelSHAP (model-agnostic SHAP values)
    m.add_function(wrap_pyfunction!(py_kernel_shap_rf_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_rf_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_dt_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_dt_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_gb_reg, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_gb_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_linear, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_logistic, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_et_clf, &m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_shap_et_reg, &m)?)?;

    parent_module.add_submodule(&m)?;

    Ok(())
}
