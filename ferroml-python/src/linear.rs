//! Python bindings for FerroML linear models
//!
//! This module provides Python wrappers for:
//! - LinearRegression
//! - LogisticRegression
//! - RidgeRegression
//! - LassoRegression
//! - ElasticNet
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays (e.g., `Model::fit`),
//! a copy is made. Output arrays use `into_pyarray` to transfer ownership to Python
//! without copying data.
//!
//! ## DataFrame Support
//!
//! ### Pandas
//! When the `pandas` feature is enabled, models support fitting and predicting
//! directly from Pandas DataFrames via `fit_pandas()` and `predict_pandas()`.
//!
//! See `crate::array_utils` for detailed documentation.

use crate::array_utils::{
    check_array1_finite, check_array_finite, py_array_to_f64_1d, to_owned_array_1d,
    to_owned_array_2d,
};
use crate::pickle::{getstate, setstate};
use ferroml_core::models::quantile::QuantileRegression;
use ferroml_core::models::regularized::{ElasticNetCV, LassoCV, RidgeCV, RidgeClassifier};
use ferroml_core::models::robust::{MEstimator, RobustRegression};
use ferroml_core::models::sgd::Perceptron;
use ferroml_core::models::traits::IncrementalModel;
use ferroml_core::models::{
    ElasticNet, LassoRegression, LinearRegression, LogisticRegression, LogisticSolver, Model,
    ProbabilisticModel, RidgeRegression, StatisticalModel,
};
use ferroml_core::onnx::{OnnxConfig, OnnxExportable};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[cfg(feature = "pandas")]
use crate::pandas_utils::{extract_x_from_pandas, extract_xy_from_pandas};
#[cfg(feature = "sparse")]
use crate::sparse_utils::{extract_sparse_x, extract_sparse_xy, py_csr_to_ferro};
#[cfg(feature = "sparse")]
use ferroml_core::models::traits::SparseModel;

// =============================================================================
// LinearRegression
// =============================================================================

/// Linear Regression with full statistical diagnostics.
///
/// Fits a linear model y = Xβ + ε using Ordinary Least Squares (OLS).
///
/// FerroML's key differentiator: provides R-style statistical output including
/// coefficient standard errors, t-statistics, p-values, confidence intervals,
/// R², adjusted R², F-statistic, and residual diagnostics.
///
/// Parameters
/// ----------
/// fit_intercept : bool, optional (default=True)
///     Whether to calculate the intercept for this model.
/// confidence_level : float, optional (default=0.95)
///     Confidence level for coefficient intervals.
///
/// Attributes
/// ----------
/// coef_ : ndarray of shape (n_features,)
///     Estimated coefficients for the linear regression problem.
/// intercept_ : float
///     Independent term in the linear model.
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from ferroml.linear import LinearRegression
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
/// >>> y = np.array([3, 5, 7, 9, 11])
/// >>> model = LinearRegression()
/// >>> model.fit(X, y)
/// >>> model.predict(X)
/// array([ 3.,  5.,  7.,  9., 11.])
/// >>> print(model.summary())  # R-style statistical output
#[pyclass(name = "LinearRegression", module = "ferroml.linear")]
pub struct PyLinearRegression {
    inner: LinearRegression,
}

impl PyLinearRegression {
    /// Get a reference to the inner model (for use by other Python binding modules).
    pub fn inner_ref(&self) -> &LinearRegression {
        &self.inner
    }
}

#[pymethods]
impl PyLinearRegression {
    /// Create a new LinearRegression model.
    ///
    /// Parameters
    /// ----------
    /// fit_intercept : bool, optional (default=True)
    ///     Whether to calculate the intercept for this model.
    /// confidence_level : float, optional (default=0.95)
    ///     Confidence level for confidence intervals.
    #[new]
    #[pyo3(signature = (fit_intercept=true, confidence_level=0.95))]
    fn new(fit_intercept: bool, confidence_level: f64) -> Self {
        let inner = LinearRegression::new()
            .with_fit_intercept(fit_intercept)
            .with_confidence_level(confidence_level);
        Self { inner }
    }

    /// Fit the model to training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : LinearRegression
    ///     Fitted estimator.
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

    /// Predict using the linear model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted values.
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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Get the coefficient estimates.
    ///
    /// Returns
    /// -------
    /// coef : ndarray of shape (n_features,)
    ///     Estimated coefficients.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .inner
            .coefficients()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(coef.into_pyarray(py))
    }

    /// Get the intercept.
    ///
    /// Returns
    /// -------
    /// intercept : float
    ///     The intercept term.
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.intercept().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get feature importances (normalized absolute coefficients).
    ///
    /// Returns
    /// -------
    /// feature_importances : ndarray of shape (n_features,)
    ///     Normalized absolute coefficient magnitudes summing to 1.
    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(imp.into_pyarray(py))
    }

    /// Get the number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the R-squared value.
    ///
    /// Returns
    /// -------
    /// r2 : float
    ///     Coefficient of determination.
    fn r_squared(&self) -> PyResult<f64> {
        self.inner.r_squared().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the adjusted R-squared value.
    ///
    /// Returns
    /// -------
    /// adj_r2 : float
    ///     Adjusted coefficient of determination.
    fn adjusted_r_squared(&self) -> PyResult<f64> {
        self.inner.adjusted_r_squared().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get F-statistic and p-value.
    ///
    /// Returns
    /// -------
    /// f_stat : float
    ///     F-statistic for overall model significance.
    /// p_value : float
    ///     p-value for the F-test.
    fn f_statistic(&self) -> PyResult<(f64, f64)> {
        self.inner.f_statistic().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get R-style summary output.
    ///
    /// Returns a comprehensive model summary similar to R's summary() function,
    /// including coefficient estimates, standard errors, t-statistics, p-values,
    /// R², adjusted R², F-statistic, and more.
    ///
    /// Returns
    /// -------
    /// summary : str
    ///     Formatted model summary.
    fn summary(&self) -> PyResult<String> {
        if !self.inner.is_fitted() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model not fitted. Call fit() first.",
            ));
        }
        Ok(format!("{}", self.inner.summary()))
    }

    /// Get model diagnostics.
    ///
    /// Returns
    /// -------
    /// diagnostics : str
    ///     Formatted diagnostics output.
    fn diagnostics(&self) -> PyResult<String> {
        if !self.inner.is_fitted() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model not fitted. Call fit() first.",
            ));
        }
        Ok(format!("{}", self.inner.diagnostics()))
    }

    /// Get residuals from the fitted model.
    ///
    /// Returns
    /// -------
    /// residuals : ndarray of shape (n_samples,)
    ///     The residuals (y - y_pred).
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let residuals = self.inner.residuals().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(residuals.into_pyarray(py))
    }

    /// Get fitted values (predictions on training data).
    ///
    /// Returns
    /// -------
    /// fitted : ndarray of shape (n_samples,)
    ///     The fitted values.
    fn fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self.inner.fitted_values().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(fitted.into_pyarray(py))
    }

    /// Get coefficients with confidence intervals.
    ///
    /// Parameters
    /// ----------
    /// level : float, optional (default=0.95)
    ///     Confidence level.
    ///
    /// Returns
    /// -------
    /// coef_info : list of dict
    ///     List of coefficient information dictionaries.
    #[pyo3(signature = (level=0.95))]
    fn coefficients_with_ci<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        if !self.inner.is_fitted() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model not fitted. Call fit() first.",
            ));
        }

        let coefs = self.inner.coefficients_with_ci(level);
        let mut result = Vec::new();

        for coef in coefs {
            let dict = PyDict::new(py);
            dict.set_item("name", coef.name)?;
            dict.set_item("estimate", coef.estimate)?;
            dict.set_item("std_error", coef.std_error)?;
            dict.set_item("t_statistic", coef.t_statistic)?;
            dict.set_item("p_value", coef.p_value)?;
            dict.set_item("ci_lower", coef.ci_lower)?;
            dict.set_item("ci_upper", coef.ci_upper)?;
            dict.set_item("confidence_level", coef.confidence_level)?;
            result.push(dict);
        }

        Ok(result)
    }

    /// Get feature importance (absolute standardized coefficients).
    ///
    /// Returns
    /// -------
    /// importance : ndarray of shape (n_features,)
    ///     Feature importance scores.
    fn feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    /// Predict with prediction intervals.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Samples.
    /// level : float, optional (default=0.95)
    ///     Confidence level for prediction intervals.
    ///
    /// Returns
    /// -------
    /// predictions : ndarray of shape (n_samples,)
    ///     Point predictions.
    /// lower : ndarray of shape (n_samples,)
    ///     Lower bound of prediction interval.
    /// upper : ndarray of shape (n_samples,)
    ///     Upper bound of prediction interval.
    #[pyo3(signature = (x, level=0.95))]
    #[allow(clippy::type_complexity)]
    fn predict_interval<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        level: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);

        let interval = self
            .inner
            .predict_interval(&x_arr, level)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok((
            interval.predictions.into_pyarray(py),
            interval.lower.into_pyarray(py),
            interval.upper.into_pyarray(py),
        ))
    }

    /// Fit the model from a Pandas DataFrame.
    ///
    /// Parameters
    /// ----------
    /// df : pandas.DataFrame
    ///     DataFrame containing features and target.
    /// target_column : str
    ///     Name of the target column.
    /// feature_columns : list of str, optional
    ///     Column names to use as features. If None, uses all numeric columns
    ///     except the target.
    ///
    /// Returns
    /// -------
    /// self : LinearRegression
    ///     Fitted estimator.
    ///
    /// Examples
    /// --------
    /// >>> import pandas as pd
    /// >>> from ferroml.linear import LinearRegression
    /// >>> df = pd.DataFrame({
    /// ...     "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    /// ...     "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
    /// ...     "y": [3.0, 6.0, 9.0, 12.0, 15.0]
    /// ... })
    /// >>> model = LinearRegression()
    /// >>> model.fit_pandas(df, target_column="y")
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_pandas<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pandas(py, df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using a Pandas DataFrame.
    ///
    /// Parameters
    /// ----------
    /// df : pandas.DataFrame
    ///     DataFrame containing features.
    /// feature_columns : list of str, optional
    ///     Column names to use as features. If None, uses all numeric columns.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted values.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_pandas<'py>(
        &self,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pandas(py, df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Fit the model from a scipy.sparse matrix.
    ///
    /// Parameters
    /// ----------
    /// X : scipy.sparse matrix (CSR or CSC)
    ///     Sparse feature matrix.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : LinearRegression
    ///     Fitted estimator.
    ///
    /// Notes
    /// -----
    /// The sparse matrix is converted to dense format internally, as FerroML's
    /// algorithms currently operate on dense matrices. For very large sparse
    /// matrices, consider dimensionality reduction first.
    ///
    /// Examples
    /// --------
    /// >>> import scipy.sparse as sp
    /// >>> from ferroml.linear import LinearRegression
    /// >>> X_sparse = sp.csr_matrix([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    /// >>> y = np.array([1.0, 2.0, 3.0])
    /// >>> model = LinearRegression()
    /// >>> model.fit_sparse(X_sparse, y)
    #[cfg(feature = "sparse")]
    fn fit_sparse<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_sparse_xy(py, x, y)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using a scipy.sparse matrix.
    ///
    /// Parameters
    /// ----------
    /// X : scipy.sparse matrix (CSR or CSC)
    ///     Sparse feature matrix.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted values.
    #[cfg(feature = "sparse")]
    fn predict_sparse<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_sparse_x(py, x)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Return the state of the model for pickling.
    ///
    /// Serializes the model to MessagePack bytes for use with pickle or joblib.
    ///
    /// Examples
    /// --------
    /// >>> import pickle
    /// >>> model = LinearRegression()
    /// >>> model.fit(X, y)
    /// >>> state = model.__getstate__()  # bytes
    /// >>> pickle.dumps(model)  # or use pickle directly
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the model state from pickled bytes.
    ///
    /// Deserializes model from MessagePack bytes.
    ///
    /// Examples
    /// --------
    /// >>> import pickle
    /// >>> loaded_model = pickle.load(open('model.pkl', 'rb'))
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "LinearRegression(fit_intercept={}, confidence_level={})",
            self.inner.fit_intercept, self.inner.confidence_level
        )
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

// =============================================================================
// LogisticRegression
// =============================================================================

/// Logistic Regression with full statistical diagnostics.
///
/// Fits a logistic model P(y=1|X) = 1 / (1 + exp(-Xβ)) using Iteratively
/// Reweighted Least Squares (IRLS) for maximum likelihood estimation.
///
/// FerroML's key differentiator: provides odds ratios with confidence intervals,
/// pseudo R², likelihood ratio tests, AIC/BIC, and deviance diagnostics.
///
/// Parameters
/// ----------
/// fit_intercept : bool, optional (default=True)
///     Whether to calculate the intercept.
/// max_iter : int, optional (default=100)
///     Maximum number of IRLS iterations.
/// tol : float, optional (default=1e-8)
///     Convergence tolerance.
/// l2_penalty : float, optional (default=0.0)
///     L2 regularization strength.
/// confidence_level : float, optional (default=0.95)
///     Confidence level for intervals.
///
/// Examples
/// --------
/// >>> from ferroml.linear import LogisticRegression
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [2, 1], [3, 3], [4, 4], [5, 5], [6, 6]])
/// >>> y = np.array([0, 0, 0, 1, 1, 1])
/// >>> model = LogisticRegression()
/// >>> model.fit(X, y)
/// >>> model.predict_proba(X)
/// >>> print(model.odds_ratios())  # Interpretable coefficients
#[pyclass(name = "LogisticRegression", module = "ferroml.linear")]
pub struct PyLogisticRegression {
    inner: LogisticRegression,
}

impl PyLogisticRegression {
    pub fn inner_ref(&self) -> &LogisticRegression {
        &self.inner
    }
}

#[pymethods]
impl PyLogisticRegression {
    /// Create a new LogisticRegression model.
    ///
    /// Parameters
    /// ----------
    /// fit_intercept : bool, optional (default=True)
    ///     Whether to add an intercept term.
    /// max_iter : int, optional (default=100)
    ///     Maximum number of solver iterations.
    /// tol : float, optional (default=1e-8)
    ///     Convergence tolerance.
    /// l2_penalty : float, optional (default=0.0)
    ///     L2 regularization strength.
    /// confidence_level : float, optional (default=0.95)
    ///     Confidence level for intervals.
    /// solver : str, optional (default="auto")
    ///     Optimization algorithm: "irls", "lbfgs", "sag", "saga", or "auto".
    ///     "auto" selects IRLS for < 50 features, L-BFGS otherwise.
    ///     "sag"/"saga" use Stochastic Average Gradient — O(d) per iteration,
    ///     best for large datasets (n > 10K).
    #[new]
    #[pyo3(signature = (fit_intercept=true, max_iter=100, tol=1e-8, l2_penalty=0.0, confidence_level=0.95, solver="auto"))]
    fn new(
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        l2_penalty: f64,
        confidence_level: f64,
        solver: &str,
    ) -> Self {
        let inner = LogisticRegression::new()
            .with_fit_intercept(fit_intercept)
            .with_max_iter(max_iter)
            .with_tol(tol)
            .with_l2_penalty(l2_penalty)
            .with_confidence_level(confidence_level)
            .with_solver(LogisticSolver::from_str_lossy(solver));
        Self { inner }
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
    /// self : LogisticRegression
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
            .map_err(crate::errors::ferro_to_pyerr)?;

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
    ///     Predicted class labels (0 or 1).
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
            .map_err(crate::errors::ferro_to_pyerr)?;

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
    /// probas : ndarray of shape (n_samples, 2)
    ///     Probability of each class (column 0 = P(y=0), column 1 = P(y=1)).
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

    /// Get the coefficient estimates.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .inner
            .coefficients()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(coef.into_pyarray(py))
    }

    /// Get the intercept.
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.intercept().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get feature importances (normalized absolute coefficients).
    ///
    /// Returns
    /// -------
    /// feature_importances : ndarray of shape (n_features,)
    ///     Normalized absolute coefficient magnitudes summing to 1.
    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(imp.into_pyarray(py))
    }

    /// Get odds ratios for each coefficient.
    ///
    /// Odds ratio = exp(β). An odds ratio > 1 means the odds of the positive
    /// class increase when the feature increases by 1 unit.
    ///
    /// Returns
    /// -------
    /// odds_ratios : ndarray of shape (n_features,)
    ///     Odds ratios for each feature coefficient.
    fn odds_ratios<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let or = self.inner.odds_ratios().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(or.into_pyarray(py))
    }

    /// Get odds ratios with confidence intervals.
    ///
    /// Parameters
    /// ----------
    /// level : float, optional (default=0.95)
    ///     Confidence level.
    ///
    /// Returns
    /// -------
    /// or_info : list of dict
    ///     List of odds ratio information dictionaries.
    #[pyo3(signature = (level=0.95))]
    fn odds_ratios_with_ci<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        if !self.inner.is_fitted() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model not fitted. Call fit() first.",
            ));
        }

        let ors = self.inner.odds_ratios_with_ci(level);
        let mut result = Vec::new();

        for or in ors {
            let dict = PyDict::new(py);
            dict.set_item("name", or.name)?;
            dict.set_item("odds_ratio", or.odds_ratio)?;
            dict.set_item("ci_lower", or.ci_lower)?;
            dict.set_item("ci_upper", or.ci_upper)?;
            dict.set_item("confidence_level", or.confidence_level)?;
            result.push(dict);
        }

        Ok(result)
    }

    /// Get McFadden's pseudo R² value.
    fn pseudo_r_squared(&self) -> PyResult<f64> {
        self.inner.pseudo_r_squared().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the log-likelihood.
    fn log_likelihood(&self) -> PyResult<f64> {
        self.inner.log_likelihood().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get AIC (Akaike Information Criterion).
    fn aic(&self) -> PyResult<f64> {
        self.inner.aic().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get BIC (Bayesian Information Criterion).
    fn bic(&self) -> PyResult<f64> {
        self.inner.bic().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Perform likelihood ratio test for overall model significance.
    ///
    /// Returns
    /// -------
    /// lr_stat : float
    ///     Likelihood ratio statistic.
    /// p_value : float
    ///     p-value for the test.
    fn likelihood_ratio_test(&self) -> PyResult<(f64, f64)> {
        self.inner.likelihood_ratio_test().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get R-style summary output.
    fn summary(&self) -> PyResult<String> {
        if !self.inner.is_fitted() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model not fitted. Call fit() first.",
            ));
        }
        Ok(format!("{}", self.inner.summary()))
    }

    /// Fit the model from a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_pandas<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pandas(py, df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict class labels using a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_pandas<'py>(
        &self,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pandas(py, df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Predict class probabilities using a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_proba_pandas<'py>(
        &self,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (x_arr, _) = extract_x_from_pandas(py, df, feature_columns)?;

        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(probas.into_pyarray(py))
    }

    /// Fit the model from a scipy.sparse matrix (native sparse, no densification).
    ///
    /// Uses the SparseModel trait for O(nnz) fitting on sparse data.
    #[cfg(feature = "sparse")]
    fn fit_sparse<'py>(
        mut slf: PyRefMut<'py, Self>,
        _py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let csr = py_csr_to_ferro(x)?;
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);

        slf.inner
            .fit_sparse(&csr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Predict class probabilities using a scipy.sparse matrix.
    #[cfg(feature = "sparse")]
    fn predict_proba_sparse<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (x_arr, _) = extract_sparse_x(py, x)?;

        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(probas.into_pyarray(py))
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    /// Compute the decision function (raw scores).
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .decision_function(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "LogisticRegression(fit_intercept={}, max_iter={}, l2_penalty={}, solver='{}')",
            self.inner.fit_intercept, self.inner.max_iter, self.inner.l2_penalty, self.inner.solver
        )
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

// =============================================================================
// RidgeRegression
// =============================================================================

/// Ridge Regression (L2 regularization).
///
/// Minimizes: ||y - Xβ||² + α||β||²
///
/// Ridge regression adds L2 penalty which shrinks coefficients toward zero
/// (but never exactly zero), helps with multicollinearity, and has a closed-form
/// solution.
///
/// Parameters
/// ----------
/// alpha : float, optional (default=1.0)
///     Regularization strength. Must be positive.
/// fit_intercept : bool, optional (default=True)
///     Whether to calculate the intercept.
///
/// Examples
/// --------
/// >>> from ferroml.linear import RidgeRegression
/// >>> model = RidgeRegression(alpha=1.0)
/// >>> model.fit(X, y)
/// >>> model.predict(X_test)
#[pyclass(name = "RidgeRegression", module = "ferroml.linear")]
pub struct PyRidgeRegression {
    inner: RidgeRegression,
}

impl PyRidgeRegression {
    pub fn inner_ref(&self) -> &RidgeRegression {
        &self.inner
    }
}

#[pymethods]
impl PyRidgeRegression {
    /// Create a new RidgeRegression model.
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> Self {
        let inner = RidgeRegression::new(alpha).with_fit_intercept(fit_intercept);
        Self { inner }
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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using the model.
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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Get the coefficient estimates.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .inner
            .coefficients()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(coef.into_pyarray(py))
    }

    /// Get the intercept.
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.intercept().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get feature importances (normalized absolute coefficients).
    ///
    /// Returns
    /// -------
    /// feature_importances : ndarray of shape (n_features,)
    ///     Normalized absolute coefficient magnitudes summing to 1.
    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(imp.into_pyarray(py))
    }

    /// Get the R² value.
    fn r_squared(&self) -> PyResult<f64> {
        self.inner.r_squared().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get R-style summary output.
    fn summary(&self) -> PyResult<String> {
        if !self.inner.is_fitted() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model not fitted. Call fit() first.",
            ));
        }
        Ok(format!("{}", self.inner.summary()))
    }

    /// Get residuals from the fitted model.
    fn residuals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let residuals = self.inner.residuals().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(residuals.into_pyarray(py))
    }

    /// Get feature importance (absolute coefficients).
    fn feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    /// Fit the model from a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_pandas<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pandas(py, df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_pandas<'py>(
        &self,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pandas(py, df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
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
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);

        slf.inner
            .fit_sparse(&csr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "RidgeRegression(alpha={}, fit_intercept={})",
            self.inner.alpha, self.inner.fit_intercept
        )
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

// =============================================================================
// LassoRegression
// =============================================================================

/// Lasso Regression (L1 regularization).
///
/// Minimizes: (1/2n)||y - Xβ||² + α||β||₁
///
/// Lasso regression adds L1 penalty which encourages sparse solutions
/// (some coefficients become exactly zero), performing automatic feature selection.
///
/// Parameters
/// ----------
/// alpha : float, optional (default=1.0)
///     Regularization strength. Must be positive.
/// fit_intercept : bool, optional (default=True)
///     Whether to calculate the intercept.
/// max_iter : int, optional (default=1000)
///     Maximum number of coordinate descent iterations.
/// tol : float, optional (default=1e-4)
///     Convergence tolerance.
///
/// Examples
/// --------
/// >>> from ferroml.linear import LassoRegression
/// >>> model = LassoRegression(alpha=0.1)
/// >>> model.fit(X, y)
/// >>> # Get sparse coefficients
/// >>> coef = model.coef_
/// >>> n_nonzero = (abs(coef) > 1e-10).sum()
#[pyclass(name = "LassoRegression", module = "ferroml.linear")]
pub struct PyLassoRegression {
    inner: LassoRegression,
}

impl PyLassoRegression {
    pub fn inner_ref(&self) -> &LassoRegression {
        &self.inner
    }
}

#[pymethods]
impl PyLassoRegression {
    /// Create a new LassoRegression model.
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true, max_iter=1000, tol=1e-4))]
    fn new(alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        let inner = LassoRegression::new(alpha)
            .with_fit_intercept(fit_intercept)
            .with_max_iter(max_iter)
            .with_tol(tol);
        Self { inner }
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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using the model.
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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Get the coefficient estimates.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .inner
            .coefficients()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(coef.into_pyarray(py))
    }

    /// Get the intercept.
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.intercept().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the R² value.
    fn r_squared(&self) -> PyResult<f64> {
        self.inner.r_squared().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get number of non-zero coefficients.
    fn n_nonzero(&self) -> PyResult<usize> {
        let coef = self.inner.coefficients().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(coef.iter().filter(|&&c| c.abs() > 1e-10).count())
    }

    /// Get R-style summary output.
    fn summary(&self) -> PyResult<String> {
        if !self.inner.is_fitted() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model not fitted. Call fit() first.",
            ));
        }
        Ok(format!("{}", self.inner.summary()))
    }

    /// Get feature importances (normalized absolute coefficients).
    ///
    /// Returns
    /// -------
    /// feature_importances : ndarray of shape (n_features,)
    ///     Normalized absolute coefficient magnitudes summing to 1.
    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(imp.into_pyarray(py))
    }

    /// Get feature importance (absolute coefficients).
    fn feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    /// Fit the model from a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_pandas<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pandas(py, df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_pandas<'py>(
        &self,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pandas(py, df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Fit the model from a scipy.sparse matrix.
    #[cfg(feature = "sparse")]
    fn fit_sparse<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_sparse_xy(py, x, y)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using a scipy.sparse matrix.
    #[cfg(feature = "sparse")]
    fn predict_sparse<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_sparse_x(py, x)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "LassoRegression(alpha={}, fit_intercept={}, max_iter={})",
            self.inner.alpha, self.inner.fit_intercept, self.inner.max_iter
        )
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

// =============================================================================
// ElasticNet
// =============================================================================

/// ElasticNet Regression (L1 + L2 regularization).
///
/// Minimizes: (1/2n)||y - Xβ||² + α * ρ * ||β||₁ + α * (1-ρ)/2 * ||β||²
///
/// ElasticNet combines L1 and L2 penalties, providing a balance between
/// feature selection (Lasso) and coefficient shrinkage (Ridge).
///
/// Parameters
/// ----------
/// alpha : float, optional (default=1.0)
///     Total regularization strength.
/// l1_ratio : float, optional (default=0.5)
///     The mixing parameter (0 = Ridge, 1 = Lasso).
/// fit_intercept : bool, optional (default=True)
///     Whether to calculate the intercept.
/// max_iter : int, optional (default=1000)
///     Maximum number of coordinate descent iterations.
/// tol : float, optional (default=1e-4)
///     Convergence tolerance.
///
/// Examples
/// --------
/// >>> from ferroml.linear import ElasticNet
/// >>> # 50% L1, 50% L2
/// >>> model = ElasticNet(alpha=0.1, l1_ratio=0.5)
/// >>> model.fit(X, y)
#[pyclass(name = "ElasticNet", module = "ferroml.linear")]
pub struct PyElasticNet {
    inner: ElasticNet,
}

impl PyElasticNet {
    pub fn inner_ref(&self) -> &ElasticNet {
        &self.inner
    }
}

#[pymethods]
impl PyElasticNet {
    /// Create a new ElasticNet model.
    #[new]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5, fit_intercept=true, max_iter=1000, tol=1e-4))]
    fn new(alpha: f64, l1_ratio: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        let inner = ElasticNet::new(alpha, l1_ratio)
            .with_fit_intercept(fit_intercept)
            .with_max_iter(max_iter)
            .with_tol(tol);
        Self { inner }
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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using the model.
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
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Get the coefficient estimates.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .inner
            .coefficients()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Model not fitted. Call fit() first.",
                )
            })?
            .clone();
        Ok(coef.into_pyarray(py))
    }

    /// Get the intercept.
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.intercept().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get the R² value.
    fn r_squared(&self) -> PyResult<f64> {
        self.inner.r_squared().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    /// Get number of non-zero coefficients.
    fn n_nonzero(&self) -> PyResult<usize> {
        let coef = self.inner.coefficients().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(coef.iter().filter(|&&c| c.abs() > 1e-10).count())
    }

    /// Get R-style summary output.
    fn summary(&self) -> PyResult<String> {
        if !self.inner.is_fitted() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Model not fitted. Call fit() first.",
            ));
        }
        Ok(format!("{}", self.inner.summary()))
    }

    /// Get feature importances (normalized absolute coefficients).
    ///
    /// Returns
    /// -------
    /// feature_importances : ndarray of shape (n_features,)
    ///     Normalized absolute coefficient magnitudes summing to 1.
    #[getter]
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let imp = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(imp.into_pyarray(py))
    }

    /// Get feature importance (absolute coefficients).
    fn feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    /// Fit the model from a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_pandas<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pandas(py, df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using a Pandas DataFrame.
    #[cfg(feature = "pandas")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_pandas<'py>(
        &self,
        py: Python<'py>,
        df: &Bound<'py, PyAny>,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pandas(py, df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Fit the model from a scipy.sparse matrix.
    #[cfg(feature = "sparse")]
    fn fit_sparse<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_sparse_xy(py, x, y)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(slf)
    }

    /// Predict using a scipy.sparse matrix.
    #[cfg(feature = "sparse")]
    fn predict_sparse<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_sparse_x(py, x)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "ElasticNet(alpha={}, l1_ratio={}, fit_intercept={})",
            self.inner.alpha, self.inner.l1_ratio, self.inner.fit_intercept
        )
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

// =============================================================================
// RobustRegression
// =============================================================================

/// Robust Regression using M-estimators.
///
/// Iteratively Reweighted Least Squares (IRLS) regression that is resistant
/// to outliers by using robust loss functions.
///
/// Parameters
/// ----------
/// estimator : str, optional (default="huber")
///     M-estimator type: "huber", "bisquare"/"tukey", "hampel", "andrews".
/// max_iter : int, optional (default=50)
///     Maximum number of IRLS iterations.
/// tol : float, optional (default=1e-6)
///     Convergence tolerance.
///
/// Attributes
/// ----------
/// coef_ : ndarray of shape (n_features,)
///     Estimated coefficients.
/// intercept_ : float
///     Estimated intercept.
#[pyclass(name = "RobustRegression", module = "ferroml.linear")]
pub struct PyRobustRegression {
    inner: RobustRegression,
}

#[pymethods]
impl PyRobustRegression {
    #[new]
    #[pyo3(signature = (estimator="huber", max_iter=50, tol=1e-6))]
    fn new(estimator: &str, max_iter: usize, tol: f64) -> PyResult<Self> {
        let est = match estimator.to_lowercase().as_str() {
            "huber" => MEstimator::Huber,
            "bisquare" | "tukey" => MEstimator::Bisquare,
            "hampel" => MEstimator::Hampel,
            "andrews" | "andrews_wave" => MEstimator::AndrewsWave,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "estimator must be 'huber', 'bisquare'/'tukey', 'hampel', or 'andrews'",
                ))
            }
        };
        let inner = RobustRegression::with_estimator(est)
            .with_max_iter(max_iter)
            .with_tol(tol);
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

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Get the fitted coefficients.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self.inner.coefficients().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(coef.clone().into_pyarray(py))
    }

    /// Get the fitted intercept.
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.intercept().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "RobustRegression(estimator={:?}, max_iter={}, tol={})",
            self.inner.estimator, self.inner.max_iter, self.inner.tol
        )
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

// =============================================================================
// QuantileRegression
// =============================================================================

/// Quantile Regression.
///
/// Estimates conditional quantiles of the response variable, rather than
/// the conditional mean. Useful for understanding the full distribution
/// of the response.
///
/// Parameters
/// ----------
/// quantile : float, optional (default=0.5)
///     The quantile to estimate (0 < quantile < 1). 0.5 gives the median.
/// max_iter : int, optional (default=1000)
///     Maximum number of iterations.
/// tol : float, optional (default=1e-6)
///     Convergence tolerance.
///
/// Attributes
/// ----------
/// coef_ : ndarray of shape (n_features,)
///     Estimated coefficients.
/// intercept_ : float
///     Estimated intercept.
#[pyclass(name = "QuantileRegression", module = "ferroml.linear")]
pub struct PyQuantileRegression {
    inner: QuantileRegression,
}

#[pymethods]
impl PyQuantileRegression {
    #[new]
    #[pyo3(signature = (quantile=0.5, max_iter=1000, tol=1e-6))]
    fn new(quantile: f64, max_iter: usize, tol: f64) -> PyResult<Self> {
        let inner = QuantileRegression::new(quantile)
            .with_max_iter(max_iter)
            .with_tol(tol);
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

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    /// Get the fitted coefficients.
    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self.inner.coefficients().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(coef.clone().into_pyarray(py))
    }

    /// Get the fitted intercept.
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.intercept().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantileRegression(quantile={}, max_iter={}, tol={})",
            self.inner.quantile, self.inner.max_iter, self.inner.tol
        )
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

// =============================================================================
// Perceptron
// =============================================================================

/// Perceptron classifier.
///
/// A simple online learning algorithm for binary and multiclass classification.
/// Equivalent to SGDClassifier with hinge loss and no regularization.
///
/// Parameters
/// ----------
/// max_iter : int, optional (default=1000)
///     Maximum number of passes over the training data.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
#[pyclass(name = "Perceptron", module = "ferroml.linear")]
pub struct PyPerceptron {
    inner: Perceptron,
}

#[pymethods]
impl PyPerceptron {
    #[new]
    #[pyo3(signature = (max_iter=1000, random_state=None))]
    fn new(max_iter: usize, random_state: Option<u64>) -> PyResult<Self> {
        let mut p = Perceptron::new().with_max_iter(max_iter);
        if let Some(seed) = random_state {
            p = p.with_random_state(seed);
        }
        Ok(Self { inner: p })
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

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    /// Incremental fit on a batch of samples.
    ///
    /// Parameters
    /// ----------
    /// x : numpy.ndarray of shape (n_samples, n_features)
    ///     Training data.
    /// y : numpy.ndarray of shape (n_samples,)
    ///     Target values.
    /// classes : list of float, optional
    ///     List of all classes that can possibly appear in y.
    ///
    /// Returns
    /// -------
    /// self : Perceptron
    ///     Updated estimator.
    #[pyo3(signature = (x, y, classes=None))]
    fn partial_fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: &Bound<'py, PyAny>,
        classes: Option<Vec<f64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;
        slf.inner
            .partial_fit_with_classes(&x_arr, &y_arr, classes.as_deref())
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Compute the decision function (raw scores before thresholding).
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        "Perceptron()".to_string()
    }
}

// =============================================================================
// RidgeCV
// =============================================================================

/// Ridge Regression with built-in cross-validated alpha selection.
///
/// Automatically selects the best regularization strength (alpha) using
/// cross-validation over a range of candidate values.
///
/// Parameters
/// ----------
/// alphas : list of float, optional
///     List of alpha values to try. If None, uses log-spaced defaults.
/// cv : int, optional (default=5)
///     Number of cross-validation folds.
///
/// Attributes
/// ----------
/// alpha_ : float
///     The best alpha found by cross-validation. Only available after fit.
#[pyclass(name = "RidgeCV", module = "ferroml.linear")]
pub struct PyRidgeCV {
    inner: RidgeCV,
}

#[pymethods]
impl PyRidgeCV {
    #[new]
    #[pyo3(signature = (alphas=None, cv=5))]
    fn new(alphas: Option<Vec<f64>>, cv: usize) -> Self {
        let inner = match alphas {
            Some(a) => RidgeCV::new(a, cv),
            None => RidgeCV::with_defaults(cv),
        };
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

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    #[getter]
    fn alpha_(&self) -> PyResult<f64> {
        self.inner.best_alpha().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!("RidgeCV(cv={})", self.inner.cv)
    }
}

// =============================================================================
// LassoCV
// =============================================================================

/// Lasso Regression with built-in cross-validated alpha selection.
///
/// Automatically generates log-spaced alpha candidates and selects the best
/// one using cross-validation.
///
/// Parameters
/// ----------
/// n_alphas : int, optional (default=100)
///     Number of alpha values to try (log-spaced).
/// cv : int, optional (default=5)
///     Number of cross-validation folds.
///
/// Attributes
/// ----------
/// alpha_ : float
///     The best alpha found by cross-validation. Only available after fit.
#[pyclass(name = "LassoCV", module = "ferroml.linear")]
pub struct PyLassoCV {
    inner: LassoCV,
}

#[pymethods]
impl PyLassoCV {
    #[new]
    #[pyo3(signature = (n_alphas=100, cv=5))]
    fn new(n_alphas: usize, cv: usize) -> Self {
        Self {
            inner: LassoCV::new(n_alphas, cv),
        }
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

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    #[getter]
    fn alpha_(&self) -> PyResult<f64> {
        self.inner.best_alpha().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "LassoCV(n_alphas={}, cv={})",
            self.inner.n_alphas, self.inner.cv
        )
    }
}

// =============================================================================
// ElasticNetCV
// =============================================================================

/// ElasticNet with built-in cross-validated alpha and l1_ratio selection.
///
/// Searches over a grid of alpha and l1_ratio values using cross-validation
/// to find the best combination.
///
/// Parameters
/// ----------
/// n_alphas : int, optional (default=100)
///     Number of alpha values to try (log-spaced).
/// l1_ratios : list of float, optional
///     List of l1_ratio values to try. If None, uses defaults
///     [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0].
/// cv : int, optional (default=5)
///     Number of cross-validation folds.
///
/// Attributes
/// ----------
/// alpha_ : float
///     The best alpha found by cross-validation. Only available after fit.
/// l1_ratio_ : float
///     The best l1_ratio found by cross-validation. Only available after fit.
#[pyclass(name = "ElasticNetCV", module = "ferroml.linear")]
pub struct PyElasticNetCV {
    inner: ElasticNetCV,
}

#[pymethods]
impl PyElasticNetCV {
    #[new]
    #[pyo3(signature = (n_alphas=100, l1_ratios=None, cv=5))]
    fn new(n_alphas: usize, l1_ratios: Option<Vec<f64>>, cv: usize) -> Self {
        let inner = match l1_ratios {
            Some(ratios) => ElasticNetCV::new(n_alphas, ratios, cv),
            None => ElasticNetCV::with_defaults(n_alphas, cv),
        };
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

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
    }

    #[getter]
    fn alpha_(&self) -> PyResult<f64> {
        self.inner.best_alpha().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
    }

    #[getter]
    fn l1_ratio_(&self) -> PyResult<f64> {
        self.inner.best_l1_ratio().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "ElasticNetCV(n_alphas={}, cv={})",
            self.inner.n_alphas, self.inner.cv
        )
    }
}

// =============================================================================
// RidgeClassifier
// =============================================================================

/// Ridge Classifier.
///
/// Classification using Ridge regression on {-1, +1} encoded targets.
/// For binary classification, fits a single Ridge and thresholds at 0.
/// For multiclass, fits one Ridge per class (One-vs-Rest) and predicts argmax.
///
/// Parameters
/// ----------
/// alpha : float, optional (default=1.0)
///     Regularization strength. Larger values specify stronger regularization.
/// fit_intercept : bool, optional (default=True)
///     Whether to calculate the intercept for this model.
#[pyclass(name = "RidgeClassifier", module = "ferroml.linear")]
pub struct PyRidgeClassifier {
    inner: RidgeClassifier,
}

#[pymethods]
impl PyRidgeClassifier {
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> Self {
        Self {
            inner: RidgeClassifier::new(alpha).with_fit_intercept(fit_intercept),
        }
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

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "RidgeClassifier(alpha={}, fit_intercept={})",
            self.inner.alpha, self.inner.fit_intercept
        )
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
            .map_err(crate::errors::ferro_to_pyerr)
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
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(PyBytes::new(py, &bytes).unbind())
    }
}

// =============================================================================
// Module registration
// =============================================================================

// Register the linear models submodule — see register_linear_module below.

// =============================================================================
// IsotonicRegression
// =============================================================================

/// Isotonic Regression.
///
/// Fits a monotonically non-decreasing (or non-increasing) piecewise-linear function
/// using the Pool Adjacent Violators Algorithm (PAVA).
///
/// Parameters
/// ----------
/// increasing : str, optional (default="true")
///     "true" for non-decreasing, "false" for non-increasing, "auto" for auto-detection.
/// y_min : float, optional
///     Minimum output value.
/// y_max : float, optional
///     Maximum output value.
/// out_of_bounds : str, optional (default="nan")
///     How to handle out-of-range predictions: "nan", "clip", or "raise".
#[pyclass(name = "IsotonicRegression", module = "ferroml.linear")]
pub struct PyIsotonicRegression {
    inner: ferroml_core::models::IsotonicRegression,
}

#[pymethods]
impl PyIsotonicRegression {
    #[new]
    #[pyo3(signature = (increasing="true", y_min=None, y_max=None, out_of_bounds="nan"))]
    fn new(
        increasing: &str,
        y_min: Option<f64>,
        y_max: Option<f64>,
        out_of_bounds: &str,
    ) -> PyResult<Self> {
        use ferroml_core::models::isotonic::{Increasing, OutOfBounds};

        let inc = match increasing {
            "true" => Increasing::True,
            "false" => Increasing::False,
            "auto" => Increasing::Auto,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "increasing must be 'true', 'false', or 'auto', got '{}'",
                    other
                )))
            }
        };

        let oob = match out_of_bounds {
            "nan" => OutOfBounds::Nan,
            "clip" => OutOfBounds::Clip,
            "raise" => OutOfBounds::Raise,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "out_of_bounds must be 'nan', 'clip', or 'raise', got '{}'",
                    other
                )))
            }
        };

        let mut iso = ferroml_core::models::IsotonicRegression::new()
            .with_increasing(inc)
            .with_out_of_bounds(oob);
        if let Some(ymin) = y_min {
            iso = iso.with_y_min(ymin);
        }
        if let Some(ymax) = y_max {
            iso = iso.with_y_max(ymax);
        }

        Ok(Self { inner: iso })
    }

    /// Fit the isotonic regression model.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, 1)
    ///     Training data (must be single-column).
    /// y : array-like of shape (n_samples,)
    ///     Target values.
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
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, 1)
    ///     Samples (must be single-column).
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

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
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
            .map_err(crate::errors::ferro_to_pyerr)
    }

    fn __repr__(&self) -> String {
        "IsotonicRegression()".to_string()
    }
}

pub fn register_linear_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let linear_module = PyModule::new(parent_module.py(), "linear")?;

    linear_module.add_class::<PyLinearRegression>()?;
    linear_module.add_class::<PyLogisticRegression>()?;
    linear_module.add_class::<PyRidgeRegression>()?;
    linear_module.add_class::<PyLassoRegression>()?;
    linear_module.add_class::<PyElasticNet>()?;
    linear_module.add_class::<PyRobustRegression>()?;
    linear_module.add_class::<PyQuantileRegression>()?;
    linear_module.add_class::<PyPerceptron>()?;
    linear_module.add_class::<PyRidgeCV>()?;
    linear_module.add_class::<PyLassoCV>()?;
    linear_module.add_class::<PyElasticNetCV>()?;
    linear_module.add_class::<PyRidgeClassifier>()?;
    linear_module.add_class::<PyIsotonicRegression>()?;

    parent_module.add_submodule(&linear_module)?;

    Ok(())
}
