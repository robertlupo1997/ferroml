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
//! ### Polars
//! When the `polars` feature is enabled, models support fitting and predicting
//! directly from Polars DataFrames via `fit_dataframe()` and `predict_dataframe()`.
//!
//! ### Pandas
//! When the `pandas` feature is enabled, models support fitting and predicting
//! directly from Pandas DataFrames via `fit_pandas()` and `predict_pandas()`.
//!
//! See `crate::array_utils` for detailed documentation.

use crate::array_utils::{py_array_to_f64_1d, to_owned_array_1d, to_owned_array_2d};
use crate::pickle::{getstate, setstate};
use ferroml_core::models::{
    ElasticNet, LassoRegression, LinearRegression, LogisticRegression, Model, ProbabilisticModel,
    RidgeRegression, StatisticalModel,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[cfg(feature = "pandas")]
use crate::pandas_utils::{extract_x_from_pandas, extract_xy_from_pandas};
#[cfg(feature = "polars")]
use crate::polars_utils::{extract_x_from_pydf, extract_xy_from_pydf};
#[cfg(feature = "sparse")]
use crate::sparse_utils::{extract_sparse_x, extract_sparse_xy};
#[cfg(feature = "polars")]
use pyo3_polars::PyDataFrame;

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
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);

        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
        let x_arr = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
        let x_arr = to_owned_array_2d(x);

        let interval = self
            .inner
            .predict_interval(&x_arr, level)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok((
            interval.predictions.into_pyarray(py),
            interval.lower.into_pyarray(py),
            interval.upper.into_pyarray(py),
        ))
    }

    /// Fit the model from a Polars DataFrame.
    ///
    /// Parameters
    /// ----------
    /// df : polars.DataFrame
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
    /// >>> import polars as pl
    /// >>> from ferroml.linear import LinearRegression
    /// >>> df = pl.DataFrame({
    /// ...     "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
    /// ...     "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
    /// ...     "y": [3.0, 6.0, 9.0, 12.0, 15.0]
    /// ... })
    /// >>> model = LinearRegression()
    /// >>> model.fit_dataframe(df, target_column="y")
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_dataframe<'py>(
        mut slf: PyRefMut<'py, Self>,
        df: PyDataFrame,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pydf(&df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict using a Polars DataFrame.
    ///
    /// Parameters
    /// ----------
    /// df : polars.DataFrame
    ///     DataFrame containing features.
    /// feature_columns : list of str, optional
    ///     Column names to use as features. If None, uses all numeric columns.
    ///
    /// Returns
    /// -------
    /// y_pred : ndarray of shape (n_samples,)
    ///     Predicted values.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_dataframe<'py>(
        &self,
        py: Python<'py>,
        df: PyDataFrame,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pydf(&df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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

    fn __repr__(&self) -> String {
        format!(
            "LinearRegression(fit_intercept={}, confidence_level={})",
            self.inner.fit_intercept, self.inner.confidence_level
        )
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

#[pymethods]
impl PyLogisticRegression {
    /// Create a new LogisticRegression model.
    #[new]
    #[pyo3(signature = (fit_intercept=true, max_iter=100, tol=1e-8, l2_penalty=0.0, confidence_level=0.95))]
    fn new(
        fit_intercept: bool,
        max_iter: usize,
        tol: f64,
        l2_penalty: f64,
        confidence_level: f64,
    ) -> Self {
        let inner = LogisticRegression::new()
            .with_fit_intercept(fit_intercept)
            .with_max_iter(max_iter)
            .with_tol(tol)
            .with_l2_penalty(l2_penalty)
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
        let x_arr = to_owned_array_2d(x);
        let y_arr = py_array_to_f64_1d(py, y)?;

        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
        let x_arr = to_owned_array_2d(x);

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
        let x_arr = to_owned_array_2d(x);

        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(probas.into_pyarray(py))
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

    /// Fit the model from a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_dataframe<'py>(
        mut slf: PyRefMut<'py, Self>,
        df: PyDataFrame,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pydf(&df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict class labels using a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_dataframe<'py>(
        &self,
        py: Python<'py>,
        df: PyDataFrame,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pydf(&df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
    }

    /// Predict class probabilities using a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_proba_dataframe<'py>(
        &self,
        py: Python<'py>,
        df: PyDataFrame,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (x_arr, _) = extract_x_from_pydf(&df, feature_columns)?;

        let probas = self
            .inner
            .predict_proba(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(probas.into_pyarray(py))
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(probas.into_pyarray(py))
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict class labels using a scipy.sparse matrix.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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

    fn __repr__(&self) -> String {
        format!(
            "LogisticRegression(fit_intercept={}, max_iter={}, l2_penalty={})",
            self.inner.fit_intercept, self.inner.max_iter, self.inner.l2_penalty
        )
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
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);

        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict using the model.
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

    /// Fit the model from a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_dataframe<'py>(
        mut slf: PyRefMut<'py, Self>,
        df: PyDataFrame,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pydf(&df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict using a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_dataframe<'py>(
        &self,
        py: Python<'py>,
        df: PyDataFrame,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pydf(&df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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

    fn __repr__(&self) -> String {
        format!(
            "RidgeRegression(alpha={}, fit_intercept={})",
            self.inner.alpha, self.inner.fit_intercept
        )
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
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);

        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict using the model.
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

    /// Get feature importance (absolute coefficients).
    fn feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    /// Fit the model from a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_dataframe<'py>(
        mut slf: PyRefMut<'py, Self>,
        df: PyDataFrame,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pydf(&df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict using a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_dataframe<'py>(
        &self,
        py: Python<'py>,
        df: PyDataFrame,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pydf(&df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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

    fn __repr__(&self) -> String {
        format!(
            "LassoRegression(alpha={}, fit_intercept={}, max_iter={})",
            self.inner.alpha, self.inner.fit_intercept, self.inner.max_iter
        )
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
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);

        slf.inner
            .fit(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict using the model.
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

    /// Get feature importance (absolute coefficients).
    fn feature_importance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let importance = self.inner.feature_importance().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted. Call fit() first.")
        })?;
        Ok(importance.into_pyarray(py))
    }

    /// Fit the model from a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, target_column, feature_columns=None))]
    fn fit_dataframe<'py>(
        mut slf: PyRefMut<'py, Self>,
        df: PyDataFrame,
        target_column: &str,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data = extract_xy_from_pydf(&df, target_column, feature_columns)?;

        slf.inner
            .fit(&data.x, &data.y)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(slf)
    }

    /// Predict using a Polars DataFrame.
    #[cfg(feature = "polars")]
    #[pyo3(signature = (df, feature_columns=None))]
    fn predict_dataframe<'py>(
        &self,
        py: Python<'py>,
        df: PyDataFrame,
        feature_columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (x_arr, _) = extract_x_from_pydf(&df, feature_columns)?;

        let predictions = self
            .inner
            .predict(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(predictions.into_pyarray(py))
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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

    fn __repr__(&self) -> String {
        format!(
            "ElasticNet(alpha={}, l1_ratio={}, fit_intercept={})",
            self.inner.alpha, self.inner.l1_ratio, self.inner.fit_intercept
        )
    }
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the linear models submodule.
pub fn register_linear_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let linear_module = PyModule::new(parent_module.py(), "linear")?;

    linear_module.add_class::<PyLinearRegression>()?;
    linear_module.add_class::<PyLogisticRegression>()?;
    linear_module.add_class::<PyRidgeRegression>()?;
    linear_module.add_class::<PyLassoRegression>()?;
    linear_module.add_class::<PyElasticNet>()?;

    parent_module.add_submodule(&linear_module)?;

    Ok(())
}
