//! Python bindings for FerroML preprocessing transformers
//!
//! This module provides Python wrappers for:
//! - Scalers: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
//! - Encoders: OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder
//! - Imputers: SimpleImputer, KNNImputer
//! - Transformers: PowerTransformer, QuantileTransformer, PolynomialFeatures, KBinsDiscretizer
//! - Feature Selection: VarianceThreshold, SelectKBest, SelectFromModel
//! - Resampling: SMOTE, ADASYN, RandomUnderSampler, RandomOverSampler
//!
//! ## Zero-Copy Semantics
//!
//! Input arrays are passed as `PyReadonlyArray` which provides read-only access.
//! When calling core Rust functions that require owned arrays (e.g., `Transformer::fit`),
//! a copy is made. Output arrays use `into_pyarray` to transfer ownership to Python
//! without copying data.
//!
//! See `crate::array_utils` for detailed documentation.

use crate::array_utils::{
    array1_usize_into_pyarray_i64, check_array1_finite, check_array_finite, to_owned_array_1d,
    to_owned_array_2d,
};
use crate::pickle::{getstate, setstate};
use ferroml_core::models::regularized::{LassoRegression, RidgeRegression};
use ferroml_core::models::{
    DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, LinearRegression, LogisticRegression,
    Model, RandomForestClassifier, RandomForestRegressor, SVR,
};
use ferroml_core::onnx::{OnnxConfig, OnnxExportable};
use ferroml_core::preprocessing::{
    discretizers::KBinsDiscretizer,
    encoders::{LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder},
    imputers::{ImputeStrategy, KNNImputer, SimpleImputer},
    polynomial::PolynomialFeatures,
    power::PowerTransformer,
    quantile::QuantileTransformer,
    scalers::{MaxAbsScaler, MinMaxScaler, NormType, Normalizer, RobustScaler, StandardScaler},
    selection::{ClosureEstimator, RecursiveFeatureElimination, SelectKBest, VarianceThreshold},
    Transformer, UnknownCategoryHandling,
};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Py;

#[cfg(feature = "sparse")]
use crate::sparse_utils::py_csr_to_ferro;

// =============================================================================
// StandardScaler
// =============================================================================

/// Standardize features by removing the mean and scaling to unit variance.
///
/// The standard score of a sample x is: z = (x - mean) / std
///
/// Parameters
/// ----------
/// with_mean : bool, optional (default=True)
///     If True, center the data before scaling.
/// with_std : bool, optional (default=True)
///     If True, scale the data to unit variance.
///
/// Attributes
/// ----------
/// mean_ : ndarray of shape (n_features,)
///     The mean value for each feature in the training set.
/// scale_ : ndarray of shape (n_features,)
///     The standard deviation for each feature.
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import StandardScaler
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [3, 4], [5, 6]])
/// >>> scaler = StandardScaler()
/// >>> scaler.fit(X)
/// >>> X_scaled = scaler.transform(X)
/// >>> X_scaled.mean(axis=0)  # approximately [0, 0]
#[pyclass(name = "StandardScaler", module = "ferroml.preprocessing")]
pub struct PyStandardScaler {
    inner: StandardScaler,
}

#[pymethods]
impl PyStandardScaler {
    /// Create a new StandardScaler.
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        let inner = StandardScaler::new()
            .with_mean(with_mean)
            .with_std(with_std);
        Self { inner }
    }

    /// Compute the mean and std to be used for later scaling.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    ///
    /// Returns
    /// -------
    /// self : StandardScaler
    ///     Fitted scaler.
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

    /// Standardize features by removing the mean and scaling to unit variance.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data to transform.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_features)
    ///     Transformed data.
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

    /// Fit to data, then transform it.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Input data.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_features)
    ///     Transformed data.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Scale back the data to the original representation.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data to inverse transform.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_features)
    ///     Inverse transformed data.
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

    /// Get the mean for each feature.
    #[getter]
    fn mean_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mean = self.inner.mean().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })?;
        Ok(mean.clone().into_pyarray(py))
    }

    /// Get the standard deviation for each feature.
    #[getter]
    fn scale_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let std = self.inner.std().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })?;
        Ok(std.clone().into_pyarray(py))
    }

    /// Get the number of features seen during fit.
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features_in().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })
    }

    /// Return the state of the scaler for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the scaler state from pickled bytes.
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

    fn __repr__(&self) -> String {
        "StandardScaler()".to_string()
    }
}

// =============================================================================
// MinMaxScaler
// =============================================================================

/// Scale features to a given range (default [0, 1]).
///
/// The transformation is: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
///
/// Parameters
/// ----------
/// feature_range : tuple (min, max), optional (default=(0, 1))
///     Desired range of transformed data.
///
/// Attributes
/// ----------
/// data_min_ : ndarray of shape (n_features,)
///     Per feature minimum seen in the data.
/// data_max_ : ndarray of shape (n_features,)
///     Per feature maximum seen in the data.
/// data_range_ : ndarray of shape (n_features,)
///     Per feature range (data_max_ - data_min_).
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import MinMaxScaler
/// >>> import numpy as np
/// >>> X = np.array([[1], [2], [3], [4], [5]])
/// >>> scaler = MinMaxScaler()
/// >>> scaler.fit_transform(X)  # Values now in [0, 1]
#[pyclass(name = "MinMaxScaler", module = "ferroml.preprocessing")]
pub struct PyMinMaxScaler {
    inner: MinMaxScaler,
}

#[pymethods]
impl PyMinMaxScaler {
    /// Create a new MinMaxScaler.
    #[new]
    #[pyo3(signature = (feature_range=(0.0, 1.0)))]
    fn new(feature_range: (f64, f64)) -> PyResult<Self> {
        if feature_range.0 >= feature_range.1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "feature_range[0] must be less than feature_range[1]",
            ));
        }
        let inner = MinMaxScaler::new().with_range(feature_range.0, feature_range.1);
        Ok(Self { inner })
    }

    /// Compute the minimum and maximum to be used for later scaling.
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

    /// Transform features by scaling to the feature range.
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

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Undo the scaling transformation.
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

    /// Get the minimum value for each feature.
    #[getter]
    fn data_min_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_min = self.inner.data_min().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })?;
        Ok(data_min.clone().into_pyarray(py))
    }

    /// Get the maximum value for each feature.
    #[getter]
    fn data_max_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_max = self.inner.data_max().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })?;
        Ok(data_max.clone().into_pyarray(py))
    }

    /// Get the range (data_max - data_min) for each feature.
    #[getter]
    fn data_range_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_range = self.inner.data_range().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })?;
        Ok(data_range.clone().into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features_in().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })
    }

    /// Return the state of the scaler for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the scaler state from pickled bytes.
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

    fn __repr__(&self) -> String {
        let (min, max) = self.inner.feature_range();
        format!("MinMaxScaler(feature_range=({}, {}))", min, max)
    }
}

// =============================================================================
// RobustScaler
// =============================================================================

/// Scale features using statistics that are robust to outliers.
///
/// This scaler removes the median and scales using the interquartile range (IQR).
///
/// Parameters
/// ----------
/// with_centering : bool, optional (default=True)
///     If True, center the data before scaling.
/// with_scaling : bool, optional (default=True)
///     If True, scale the data to the IQR.
/// quantile_range : tuple (q_min, q_max), optional (default=(25.0, 75.0))
///     Quantile range used to calculate scale.
///
/// Attributes
/// ----------
/// center_ : ndarray of shape (n_features,)
///     The median value for each feature.
/// scale_ : ndarray of shape (n_features,)
///     The IQR for each feature.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import RobustScaler
/// >>> import numpy as np
/// >>> X = np.array([[1], [2], [3], [4], [100]])  # Note outlier
/// >>> scaler = RobustScaler()
/// >>> X_scaled = scaler.fit_transform(X)
#[pyclass(name = "RobustScaler", module = "ferroml.preprocessing")]
pub struct PyRobustScaler {
    inner: RobustScaler,
}

#[pymethods]
impl PyRobustScaler {
    /// Create a new RobustScaler.
    #[new]
    #[pyo3(signature = (with_centering=true, with_scaling=true, quantile_range=(25.0, 75.0)))]
    fn new(with_centering: bool, with_scaling: bool, quantile_range: (f64, f64)) -> PyResult<Self> {
        if quantile_range.0 >= quantile_range.1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "quantile_range[0] must be less than quantile_range[1]",
            ));
        }
        let inner = RobustScaler::new()
            .with_centering(with_centering)
            .with_scaling(with_scaling)
            .with_quantile_range(quantile_range.0, quantile_range.1);
        Ok(Self { inner })
    }

    /// Compute the median and quantiles to be used for scaling.
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

    /// Center and scale the data.
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

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Undo the centering and scaling.
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

    /// Get the median (center) for each feature.
    #[getter]
    fn center_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let center = self.inner.center().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })?;
        Ok(center.clone().into_pyarray(py))
    }

    /// Get the IQR (scale) for each feature.
    #[getter]
    fn scale_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let scale = self.inner.scale().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })?;
        Ok(scale.clone().into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features_in().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })
    }

    /// Return the state of the scaler for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the scaler state from pickled bytes.
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

    fn __repr__(&self) -> String {
        "RobustScaler()".to_string()
    }
}

// =============================================================================
// MaxAbsScaler
// =============================================================================

/// Scale features by their maximum absolute value.
///
/// This scaler divides by the maximum absolute value, resulting in features in [-1, 1].
/// Does not shift/center the data, preserving sparsity.
///
/// Attributes
/// ----------
/// max_abs_ : ndarray of shape (n_features,)
///     Per feature maximum absolute value.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import MaxAbsScaler
/// >>> import numpy as np
/// >>> X = np.array([[-5], [0], [3]])
/// >>> scaler = MaxAbsScaler()
/// >>> scaler.fit_transform(X)  # Values now in [-1, 1]
#[pyclass(name = "MaxAbsScaler", module = "ferroml.preprocessing")]
pub struct PyMaxAbsScaler {
    inner: MaxAbsScaler,
}

#[pymethods]
impl PyMaxAbsScaler {
    /// Create a new MaxAbsScaler.
    #[new]
    fn new() -> Self {
        Self {
            inner: MaxAbsScaler::new(),
        }
    }

    /// Compute the maximum absolute value to be used for scaling.
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

    /// Scale features by dividing by max absolute value.
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

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Undo the scaling.
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

    /// Get the maximum absolute value for each feature.
    #[getter]
    fn max_abs_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let max_abs = self.inner.max_abs().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })?;
        Ok(max_abs.clone().into_pyarray(py))
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features_in().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Scaler not fitted. Call fit() first.")
        })
    }

    /// Return the state of the scaler for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the scaler state from pickled bytes.
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

    fn __repr__(&self) -> String {
        "MaxAbsScaler()".to_string()
    }
}

// =============================================================================
// OneHotEncoder
// =============================================================================

/// Encode categorical features as a one-hot numeric array.
///
/// Each unique value becomes a binary column.
///
/// Parameters
/// ----------
/// handle_unknown : str, optional (default='error')
///     How to handle unknown categories during transform.
///     'error': raise an error
///     'ignore': output all zeros for the feature
/// drop : str, optional (default=None)
///     Strategy for dropping categories to avoid collinearity.
///     None: don't drop any category
///     'first': drop the first category in each feature
///     'if_binary': drop first category only for binary features
///
/// Attributes
/// ----------
/// categories_ : list of arrays
///     The categories for each feature.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import OneHotEncoder
/// >>> import numpy as np
/// >>> X = np.array([[0], [1], [2], [1]])
/// >>> encoder = OneHotEncoder()
/// >>> X_encoded = encoder.fit_transform(X)
/// >>> X_encoded.shape  # (4, 3) - 3 categories
#[pyclass(name = "OneHotEncoder", module = "ferroml.preprocessing")]
pub struct PyOneHotEncoder {
    inner: OneHotEncoder,
}

#[pymethods]
impl PyOneHotEncoder {
    /// Create a new OneHotEncoder.
    #[new]
    #[pyo3(signature = (handle_unknown="error", drop=None))]
    fn new(handle_unknown: &str, drop: Option<&str>) -> PyResult<Self> {
        let unknown_handling = parse_handle_unknown(handle_unknown)?;

        let drop_strategy = match drop {
            None => ferroml_core::preprocessing::encoders::DropStrategy::None,
            Some("first") => ferroml_core::preprocessing::encoders::DropStrategy::First,
            Some("if_binary") => ferroml_core::preprocessing::encoders::DropStrategy::IfBinary,
            Some(other) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown drop value: {}. Use None, 'first', or 'if_binary'",
                    other
                )));
            }
        };

        let inner = OneHotEncoder::new()
            .with_handle_unknown(unknown_handling)
            .with_drop(drop_strategy);
        Ok(Self { inner })
    }

    /// Fit the encoder to the data.
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

    /// Transform X to one-hot encoding.
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

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Convert back to original categorical representation.
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

    /// Get the categories for each input feature.
    #[getter]
    fn categories_(&self) -> PyResult<Vec<Vec<f64>>> {
        let cats = self.inner.categories().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Encoder not fitted. Call fit() first.")
        })?;
        Ok(cats.to_vec())
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features_in().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Encoder not fitted. Call fit() first.")
        })
    }

    /// Return the state of the encoder for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the encoder state from pickled bytes.
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "OneHotEncoder()".to_string()
    }
}

// =============================================================================
// OrdinalEncoder
// =============================================================================

/// Encode categorical features as integers.
///
/// Each unique value is mapped to an integer (0, 1, 2, ...).
///
/// Parameters
/// ----------
/// handle_unknown : str, optional (default='error')
///     How to handle unknown categories during transform.
///     'error': raise an error
///     'ignore': map to -1
///
/// Attributes
/// ----------
/// categories_ : list of arrays
///     The categories for each feature.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import OrdinalEncoder
/// >>> import numpy as np
/// >>> X = np.array([[1.0], [3.0], [2.0], [1.0]])
/// >>> encoder = OrdinalEncoder()
/// >>> encoder.fit_transform(X)  # Returns [[0], [1], [2], [0]]
#[pyclass(name = "OrdinalEncoder", module = "ferroml.preprocessing")]
pub struct PyOrdinalEncoder {
    inner: OrdinalEncoder,
}

#[pymethods]
impl PyOrdinalEncoder {
    /// Create a new OrdinalEncoder.
    #[new]
    #[pyo3(signature = (handle_unknown="error"))]
    fn new(handle_unknown: &str) -> PyResult<Self> {
        let unknown_handling = parse_handle_unknown(handle_unknown)?;

        let inner = OrdinalEncoder::new().with_handle_unknown(unknown_handling);
        Ok(Self { inner })
    }

    /// Fit the encoder to the data.
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

    /// Transform X to ordinal codes.
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

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Convert back to original categorical representation.
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

    /// Get the categories for each input feature.
    #[getter]
    fn categories_(&self) -> PyResult<Vec<Vec<f64>>> {
        let cats = self.inner.categories().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Encoder not fitted. Call fit() first.")
        })?;
        Ok(cats.to_vec())
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features_in().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Encoder not fitted. Call fit() first.")
        })
    }

    /// Return the state of the encoder for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the encoder state from pickled bytes.
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "OrdinalEncoder()".to_string()
    }
}

// =============================================================================
// LabelEncoder
// =============================================================================

/// Encode target labels as integers.
///
/// Transforms labels to normalized encoding (0 to n_classes-1).
///
/// Attributes
/// ----------
/// classes_ : array of shape (n_classes,)
///     Holds the label for each class.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import LabelEncoder
/// >>> import numpy as np
/// >>> y = np.array([2.0, 0.0, 1.0, 2.0, 1.0])
/// >>> encoder = LabelEncoder()
/// >>> encoder.fit(y)
/// >>> encoder.transform(y)  # Returns [0, 1, 2, 0, 2] based on first appearance
#[pyclass(name = "LabelEncoder", module = "ferroml.preprocessing")]
pub struct PyLabelEncoder {
    inner: LabelEncoder,
}

#[pymethods]
impl PyLabelEncoder {
    /// Create a new LabelEncoder.
    #[new]
    fn new() -> Self {
        Self {
            inner: LabelEncoder::new(),
        }
    }

    /// Fit label encoder.
    ///
    /// Parameters
    /// ----------
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit_1d(&y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform labels to normalized encoding.
    ///
    /// Parameters
    /// ----------
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// y : ndarray of shape (n_samples,)
    ///     Encoded labels.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        let result = self
            .inner
            .transform_1d(&y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit label encoder and return encoded labels.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        let result = slf
            .inner
            .fit_transform_1d(&y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Transform labels back to original encoding.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        let result = self
            .inner
            .inverse_transform_1d(&y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Get the unique classes/labels.
    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let classes = self.inner.classes().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Encoder not fitted. Call fit() first.")
        })?;
        Ok(Array1::from_vec(classes.to_vec()).into_pyarray(py))
    }

    /// Return the state of the encoder for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the encoder state from pickled bytes.
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "LabelEncoder()".to_string()
    }
}

// =============================================================================
// SimpleImputer
// =============================================================================

/// Imputation transformer for completing missing values.
///
/// Parameters
/// ----------
/// strategy : str, optional (default='mean')
///     The imputation strategy.
///     'mean': replace missing values using the mean
///     'median': replace using the median
///     'most_frequent': replace using the most frequent value (mode)
///     'constant': replace with fill_value
/// fill_value : float, optional (default=0.0)
///     Value used for Constant strategy or fallback for features with all missing.
/// missing_values : float, optional (default=NaN)
///     The placeholder for missing values (default is np.nan).
/// add_indicator : bool, optional (default=False)
///     If True, add indicator columns for missing values.
///
/// Attributes
/// ----------
/// statistics_ : ndarray of shape (n_features,)
///     The imputation fill value for each feature.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import SimpleImputer
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [np.nan, 4], [3, np.nan]])
/// >>> imputer = SimpleImputer(strategy='mean')
/// >>> imputer.fit_transform(X)  # Missing values filled with column means
#[pyclass(name = "SimpleImputer", module = "ferroml.preprocessing")]
pub struct PySimpleImputer {
    inner: SimpleImputer,
}

#[pymethods]
impl PySimpleImputer {
    /// Create a new SimpleImputer.
    #[new]
    #[pyo3(signature = (strategy="mean", fill_value=0.0, missing_values=None, add_indicator=false))]
    fn new(
        strategy: &str,
        fill_value: f64,
        missing_values: Option<f64>,
        add_indicator: bool,
    ) -> PyResult<Self> {
        let impute_strategy = match strategy {
            "mean" => ImputeStrategy::Mean,
            "median" => ImputeStrategy::Median,
            "most_frequent" => ImputeStrategy::MostFrequent,
            "constant" => ImputeStrategy::Constant,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown strategy: {}. Use 'mean', 'median', 'most_frequent', or 'constant'",
                    strategy
                )));
            }
        };

        let mut inner = SimpleImputer::new(impute_strategy)
            .with_fill_value(fill_value)
            .with_indicator(add_indicator);

        if let Some(mv) = missing_values {
            inner = inner.with_missing_value(mv);
        }

        Ok(Self { inner })
    }

    /// Fit the imputer on X.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Impute all missing values in X.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit to data, then transform it.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Get the computed statistics (fill values) for each feature.
    #[getter]
    fn statistics_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let stats = self.inner.statistics().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Imputer not fitted. Call fit() first.")
        })?;
        Ok(stats.clone().into_pyarray(py))
    }

    /// Get the number of missing values per feature found during fit.
    fn get_missing_counts(&self) -> PyResult<Vec<usize>> {
        let counts = self.inner.missing_counts().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Imputer not fitted. Call fit() first.")
        })?;
        Ok(counts.to_vec())
    }

    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        self.inner.n_features_in().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Imputer not fitted. Call fit() first.")
        })
    }

    /// Return the state of the imputer for pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    /// Restore the imputer state from pickled bytes.
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let strategy_str = match self.inner.strategy() {
            ImputeStrategy::Mean => "mean",
            ImputeStrategy::Median => "median",
            ImputeStrategy::MostFrequent => "most_frequent",
            ImputeStrategy::Constant => "constant",
        };
        format!("SimpleImputer(strategy='{}')", strategy_str)
    }
}

// =============================================================================
// PowerTransformer
// =============================================================================

/// Apply a power transformation to make data more Gaussian-like.
///
/// Supports Box-Cox (positive data only) and Yeo-Johnson (any data) methods.
///
/// Parameters
/// ----------
/// method : str, optional (default="yeo-johnson")
///     Power transform method: "yeo-johnson" (works with any data) or
///     "box-cox" (requires strictly positive data).
/// standardize : bool, optional (default=True)
///     Whether to standardize the transformed output to zero mean and unit
///     variance.
///
/// Attributes
/// ----------
/// lambdas_ : ndarray of shape (n_features,)
///     The fitted lambda parameters for each feature.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import PowerTransformer
/// >>> import numpy as np
/// >>> X = np.random.exponential(2.0, size=(100, 3))
/// >>> pt = PowerTransformer(method="yeo-johnson")
/// >>> X_t = pt.fit_transform(X)
/// >>> # X_t is now approximately Gaussian
///
/// Notes
/// -----
/// Use "yeo-johnson" for data that may contain zero or negative values.
/// Use "box-cox" only when all values are strictly positive. The transformer
/// supports ``inverse_transform()`` to recover original scale.
#[pyclass(name = "PowerTransformer", module = "ferroml.preprocessing")]
pub struct PyPowerTransformer {
    inner: PowerTransformer,
}

#[pymethods]
impl PyPowerTransformer {
    #[new]
    #[pyo3(signature = (method="yeo-johnson"))]
    fn new(method: &str) -> PyResult<Self> {
        use ferroml_core::preprocessing::power::PowerMethod;

        let pm = match method {
            "yeo-johnson" => PowerMethod::YeoJohnson,
            "box-cox" => PowerMethod::BoxCox,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown method: '{}'. Use 'yeo-johnson' or 'box-cox'.",
                    method
                )));
            }
        };
        Ok(Self {
            inner: PowerTransformer::new(pm),
        })
    }

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

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "PowerTransformer()".to_string()
    }
}

// =============================================================================
// QuantileTransformer
// =============================================================================

/// Transform features using quantile information.
///
/// Maps the data to a uniform or normal distribution.
///
/// Parameters
/// ----------
/// output_distribution : str, optional (default="uniform")
///     Target distribution: "uniform" maps to U(0,1), "normal" maps to N(0,1).
/// n_quantiles : int, optional (default=1000)
///     Number of quantiles to compute. Should be <= n_samples.
///
/// Attributes
/// ----------
/// quantiles_ : ndarray
///     The computed quantile boundaries per feature.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import QuantileTransformer
/// >>> import numpy as np
/// >>> X = np.random.exponential(2.0, size=(200, 3))
/// >>> qt = QuantileTransformer(output_distribution="normal", n_quantiles=100)
/// >>> X_t = qt.fit_transform(X)
/// >>> # X_t columns are now approximately standard normal
///
/// Notes
/// -----
/// This transform is non-parametric and robust to outliers. It maps each
/// feature to a uniform or normal distribution using estimated quantiles.
/// Supports ``inverse_transform()`` to recover the original scale.
#[pyclass(name = "QuantileTransformer", module = "ferroml.preprocessing")]
pub struct PyQuantileTransformer {
    inner: QuantileTransformer,
}

#[pymethods]
impl PyQuantileTransformer {
    #[new]
    #[pyo3(signature = (output_distribution="uniform", n_quantiles=1000))]
    fn new(output_distribution: &str, n_quantiles: usize) -> PyResult<Self> {
        use ferroml_core::preprocessing::quantile::OutputDistribution;

        let dist = match output_distribution {
            "uniform" => OutputDistribution::Uniform,
            "normal" => OutputDistribution::Normal,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown distribution: '{}'. Use 'uniform' or 'normal'.",
                    output_distribution
                )));
            }
        };
        Ok(Self {
            inner: QuantileTransformer::new(dist).with_n_quantiles(n_quantiles),
        })
    }

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

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "QuantileTransformer()".to_string()
    }
}

// =============================================================================
// PolynomialFeatures
// =============================================================================

/// Generate polynomial and interaction features.
///
/// Parameters
/// ----------
/// degree : int, optional (default=2)
///     Maximum degree of polynomial features. Must be >= 1.
/// interaction_only : bool, optional (default=False)
///     If True, only interaction features (not powers) are produced.
/// include_bias : bool, optional (default=True)
///     If True, include a bias column of ones.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import PolynomialFeatures
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [3, 4]], dtype=np.float64)
/// >>> poly = PolynomialFeatures(degree=2)
/// >>> X_poly = poly.fit_transform(X)
/// >>> # X_poly includes [1, a, b, a^2, ab, b^2] for each row
///
/// Notes
/// -----
/// The number of output features grows combinatorially with degree and
/// input features. For n features and degree d, the output has
/// C(n+d, d) columns. Use with caution on high-dimensional data.
#[pyclass(name = "PolynomialFeatures", module = "ferroml.preprocessing")]
pub struct PyPolynomialFeatures {
    inner: PolynomialFeatures,
}

#[pymethods]
impl PyPolynomialFeatures {
    #[new]
    #[pyo3(signature = (degree=2))]
    fn new(degree: usize) -> Self {
        let pf = PolynomialFeatures::new(degree);
        Self { inner: pf }
    }

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
        format!("PolynomialFeatures(degree={})", self.inner.degree())
    }
}

// =============================================================================
// KBinsDiscretizer
// =============================================================================

/// Bin continuous data into intervals.
///
/// Parameters
/// ----------
/// n_bins : int, optional (default=5)
///     Number of bins per feature. Must be >= 2.
/// strategy : str, optional (default="quantile")
///     Binning strategy: "uniform" (equal width), "quantile" (equal frequency),
///     or "kmeans" (k-means clustering).
/// encode : str, optional (default="ordinal")
///     Output encoding: "ordinal" (integer bin labels) or "onehot"
///     (one-hot encoded).
///
/// Attributes
/// ----------
/// bin_edges_ : list of ndarray
///     Bin edges for each feature.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import KBinsDiscretizer
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 3)
/// >>> kbd = KBinsDiscretizer(n_bins=4, strategy="quantile")
/// >>> X_binned = kbd.fit_transform(X)
/// >>> # Each feature is now discretized into 4 bins (0, 1, 2, 3)
#[pyclass(name = "KBinsDiscretizer", module = "ferroml.preprocessing")]
pub struct PyKBinsDiscretizer {
    inner: KBinsDiscretizer,
}

#[pymethods]
impl PyKBinsDiscretizer {
    #[new]
    #[pyo3(signature = (n_bins=5, strategy="quantile", encode="ordinal"))]
    fn new(n_bins: usize, strategy: &str, encode: &str) -> PyResult<Self> {
        use ferroml_core::preprocessing::discretizers::{BinEncoding, BinningStrategy};

        let strat = match strategy {
            "uniform" => BinningStrategy::Uniform,
            "quantile" => BinningStrategy::Quantile,
            "kmeans" => BinningStrategy::KMeans,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown strategy: '{}'. Use 'uniform', 'quantile', or 'kmeans'.",
                    strategy
                )));
            }
        };

        let enc = match encode {
            "ordinal" => BinEncoding::Ordinal,
            "onehot" => BinEncoding::OneHot,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown encoding: '{}'. Use 'ordinal' or 'onehot'.",
                    encode
                )));
            }
        };

        Ok(Self {
            inner: KBinsDiscretizer::new()
                .with_n_bins(n_bins)
                .with_strategy(strat)
                .with_encode(enc),
        })
    }

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
        format!("KBinsDiscretizer(n_bins={})", self.inner.n_bins())
    }
}

// =============================================================================
// VarianceThreshold
// =============================================================================

/// Feature selector that removes low-variance features.
///
/// Parameters
/// ----------
/// threshold : float, optional (default=0.0)
///     Features with variance below this value are removed. Must be >= 0.
///     A threshold of 0.0 removes only constant features.
///
/// Attributes
/// ----------
/// variances_ : ndarray of shape (n_features,)
///     Variance of each feature in the training data.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import VarianceThreshold
/// >>> import numpy as np
/// >>> X = np.array([[0, 2, 0.1], [0, 1, 0.2], [0, 3, 0.3]], dtype=np.float64)
/// >>> # First column is constant (variance=0), will be removed
/// >>> vt = VarianceThreshold(threshold=0.0)
/// >>> X_sel = vt.fit_transform(X)
/// >>> X_sel.shape
/// (3, 2)
#[pyclass(name = "VarianceThreshold", module = "ferroml.preprocessing")]
pub struct PyVarianceThreshold {
    inner: VarianceThreshold,
}

#[pymethods]
impl PyVarianceThreshold {
    #[new]
    #[pyo3(signature = (threshold=0.0))]
    fn new(threshold: f64) -> Self {
        Self {
            inner: VarianceThreshold::new(threshold),
        }
    }

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
        "VarianceThreshold()".to_string()
    }
}

// =============================================================================
// SelectKBest
// =============================================================================

/// Select the K highest scoring features.
///
/// Parameters
/// ----------
/// score_func : str, optional (default="f_classif")
///     Scoring function: "f_classif" (ANOVA F for classification),
///     "f_regression" (F-statistic for regression), "chi2" (chi-squared
///     for non-negative features in classification).
/// k : int, optional (default=10)
///     Number of top features to select. Must be <= n_features.
///
/// Attributes
/// ----------
/// scores_ : ndarray of shape (n_features,)
///     Scores of each feature.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import SelectKBest
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 10)
/// >>> y = (X[:, 0] + X[:, 3] > 0).astype(np.float64)
/// >>> sel = SelectKBest(score_func="f_classif", k=3)
/// >>> sel.fit(X, y)
/// >>> X_sel = sel.transform(X)
/// >>> X_sel.shape
/// (100, 3)
/// >>> sel.scores_  # feature importance scores
#[pyclass(name = "SelectKBest", module = "ferroml.preprocessing")]
pub struct PySelectKBest {
    inner: SelectKBest,
}

#[pymethods]
impl PySelectKBest {
    #[new]
    #[pyo3(signature = (score_func="f_classif", k=10))]
    fn new(score_func: &str, k: usize) -> PyResult<Self> {
        use ferroml_core::preprocessing::selection::ScoreFunction;

        let sf = match score_func {
            "f_classif" => ScoreFunction::FClassif,
            "f_regression" => ScoreFunction::FRegression,
            "chi2" => ScoreFunction::Chi2,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown score_func: '{}'. Use 'f_classif', 'f_regression', or 'chi2'.",
                    score_func
                )));
            }
        };
        Ok(Self {
            inner: SelectKBest::new(sf, k),
        })
    }

    /// Fit SelectKBest with target variable (required).
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
            .fit_with_target(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

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

    #[getter]
    fn scores_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let feature_scores = self
            .inner
            .scores()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(feature_scores.scores.clone().into_pyarray(py))
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
        getstate(py, &self.inner)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.inner = setstate(state.as_bytes())?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        "SelectKBest()".to_string()
    }
}

// =============================================================================
// KNNImputer
// =============================================================================

/// Impute missing values using K-Nearest Neighbors.
///
/// Each missing feature is imputed using values from the k nearest
/// neighbors that have non-missing values for that feature.
///
/// Parameters
/// ----------
/// n_neighbors : int, optional (default=5)
///     Number of nearest neighbors to use for imputation.
/// weights : str, optional (default="uniform")
///     Weight function: "uniform" (equal weight) or "distance" (inverse
///     distance weighting).
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import KNNImputer
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [np.nan, 3], [7, 6], [4, np.nan]])
/// >>> imputer = KNNImputer(n_neighbors=2)
/// >>> X_imputed = imputer.fit_transform(X)
/// >>> # NaN values are replaced with weighted means from nearest neighbors
///
/// Notes
/// -----
/// KNNImputer handles NaN values natively (exempt from finite check).
/// Distance computation ignores features with missing values. This is
/// more accurate than SimpleImputer for data with complex correlations
/// but slower for large datasets.
#[pyclass(name = "KNNImputer", module = "ferroml.preprocessing")]
pub struct PyKNNImputer {
    inner: KNNImputer,
}

#[pymethods]
impl PyKNNImputer {
    #[new]
    #[pyo3(signature = (n_neighbors=5, weights="uniform"))]
    fn new(n_neighbors: usize, weights: &str) -> PyResult<Self> {
        use ferroml_core::preprocessing::imputers::KNNWeights;

        let w = match weights {
            "uniform" => KNNWeights::Uniform,
            "distance" => KNNWeights::Distance,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown weights: '{}'. Use 'uniform' or 'distance'.",
                    weights
                )));
            }
        };
        Ok(Self {
            inner: KNNImputer::new(n_neighbors).with_weights(w),
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
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
        "KNNImputer()".to_string()
    }
}

// =============================================================================
// TargetEncoder
// =============================================================================

/// Encode categorical features using target statistics.
///
/// Each category is encoded by the mean of the target variable for that
/// category, with smoothing to reduce overfitting.
///
/// Parameters
/// ----------
/// smooth : float, optional (default=1.0)
///     Smoothing parameter. Higher values pull rare category encodings
///     toward the global mean. Must be >= 0.
/// cv : int, optional (default=5)
///     Number of CV folds for internal cross-validation. Must be >= 2.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import TargetEncoder
/// >>> import numpy as np
/// >>> X = np.array([[0], [1], [0], [1], [2], [2]], dtype=np.float64)
/// >>> y = np.array([10.0, 20.0, 12.0, 22.0, 30.0, 28.0])
/// >>> enc = TargetEncoder(smooth=1.0, cv=3)
/// >>> enc.fit(X, y)
/// >>> X_enc = enc.transform(X)
/// >>> # Each category is replaced by smoothed target mean
///
/// Notes
/// -----
/// TargetEncoder uses internal cross-validation to prevent target leakage.
/// Requires a target variable y during fit. Works best with categorical
/// features encoded as integers.
#[pyclass(name = "TargetEncoder", module = "ferroml.preprocessing")]
pub struct PyTargetEncoder {
    inner: TargetEncoder,
}

#[pymethods]
impl PyTargetEncoder {
    #[new]
    #[pyo3(signature = (smooth=1.0, cv=5))]
    fn new(smooth: f64, cv: usize) -> Self {
        Self {
            inner: TargetEncoder::new().with_smooth(smooth).with_cv(cv),
        }
    }

    /// Fit TargetEncoder with target variable (required).
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
            .fit_with_target(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

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
        "TargetEncoder()".to_string()
    }
}

// =============================================================================
// SelectFromModel
// =============================================================================

/// Select features based on importance weights from a fitted model.
///
/// Selects features whose importance is above a threshold, determined
/// by a fitted model's feature_importances_ or coef_ attribute.
///
/// Parameters
/// ----------
/// importances : ndarray of shape (n_features,)
///     Feature importances from a fitted model.
/// threshold : str or float, optional (default="mean")
///     The threshold value to use: "mean", "median", or a float value.
/// max_features : int, optional
///     Maximum number of features to select.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import SelectFromModel
/// >>> import numpy as np
/// >>> importances = np.array([0.1, 0.5, 0.3, 0.05, 0.8])
/// >>> selector = SelectFromModel(importances, threshold="mean")
/// >>> selector.fit(X)
/// >>> X_selected = selector.transform(X)
#[pyclass(name = "SelectFromModel", module = "ferroml.preprocessing")]
pub struct PySelectFromModel {
    inner: ferroml_core::preprocessing::selection::SelectFromModel,
}

#[pymethods]
impl PySelectFromModel {
    #[new]
    #[pyo3(signature = (importances, threshold="mean", max_features=None))]
    fn new(
        importances: PyReadonlyArray1<'_, f64>,
        threshold: &str,
        max_features: Option<usize>,
    ) -> PyResult<Self> {
        use ferroml_core::preprocessing::selection::ImportanceThreshold;

        let imp_arr = importances.as_array().to_owned();
        let thresh = match threshold {
            "mean" => ImportanceThreshold::Mean,
            "median" => ImportanceThreshold::Median,
            other => {
                let val: f64 = other.parse().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown threshold: '{}'. Use 'mean', 'median', or a float value.",
                        other
                    ))
                })?;
                ImportanceThreshold::Value(val)
            }
        };

        let mut sfm = ferroml_core::preprocessing::selection::SelectFromModel::new(imp_arr, thresh);
        if let Some(max) = max_features {
            sfm = sfm.with_max_features(max);
        }
        Ok(Self { inner: sfm })
    }

    /// Fit the selector to the data.
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

    /// Transform data by selecting features.
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

    /// Get the boolean support mask indicating which features are selected.
    fn get_support(&self) -> PyResult<Vec<bool>> {
        let support = self
            .inner
            .get_support()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not fitted."))?;
        Ok(support.clone())
    }

    fn __repr__(&self) -> String {
        "SelectFromModel()".to_string()
    }
}

// =============================================================================
// SMOTE (Synthetic Minority Oversampling Technique)
// =============================================================================

/// SMOTE: Synthetic Minority Over-sampling Technique.
///
/// Generates synthetic samples for minority classes by interpolating between
/// existing minority samples and their k-nearest neighbors.
///
/// Parameters
/// ----------
/// k_neighbors : int, optional (default=5)
///     Number of nearest neighbors to use for generating synthetic samples.
/// sampling_strategy : str, optional (default="auto")
///     Strategy: "auto" balances all classes to majority, or a float ratio.
/// random_state : int, optional
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import SMOTE
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 5)
/// >>> y = np.array([0]*90 + [1]*10)  # imbalanced
/// >>> smote = SMOTE(k_neighbors=5, random_state=42)
/// >>> X_res, y_res = smote.fit_resample(X, y)
#[pyclass(name = "SMOTE", module = "ferroml.preprocessing")]
pub struct PySMOTE {
    inner: ferroml_core::preprocessing::sampling::SMOTE,
}

#[pymethods]
impl PySMOTE {
    #[new]
    #[pyo3(signature = (k_neighbors=5, sampling_strategy="auto", random_state=None))]
    fn new(
        k_neighbors: usize,
        sampling_strategy: &str,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let mut smote =
            ferroml_core::preprocessing::sampling::SMOTE::new().with_k_neighbors(k_neighbors);

        smote = smote.with_sampling_strategy(parse_sampling_strategy(sampling_strategy)?);

        if let Some(seed) = random_state {
            smote = smote.with_random_state(seed);
        }
        Ok(Self { inner: smote })
    }

    /// Fit to the data and generate synthetic samples.
    ///
    /// Parameters
    /// ----------
    /// X : ndarray of shape (n_samples, n_features)
    ///     Feature matrix.
    /// y : ndarray of shape (n_samples,)
    ///     Target class labels.
    ///
    /// Returns
    /// -------
    /// tuple of (X_resampled, y_resampled)
    ///     X_resampled : ndarray of shape (n_new_samples, n_features)
    ///     y_resampled : ndarray of shape (n_new_samples,)
    #[allow(clippy::type_complexity)]
    fn fit_resample<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        use ferroml_core::preprocessing::sampling::Resampler;

        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        let (x_res, y_res) = self
            .inner
            .fit_resample(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok((x_res.into_pyarray(py), y_res.into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        "SMOTE()".to_string()
    }
}

// =============================================================================
// ADASYN (Adaptive Synthetic Sampling)
// =============================================================================

/// ADASYN: Adaptive Synthetic Sampling.
///
/// Similar to SMOTE but generates more synthetic samples near the decision
/// boundary where the classifier has difficulty, focusing on harder-to-learn
/// minority samples.
///
/// Parameters
/// ----------
/// k_neighbors : int, optional (default=5)
///     Number of nearest neighbors for density estimation.
/// n_neighbors : int, optional (default=5)
///     Number of nearest neighbors for synthetic sample generation.
/// sampling_strategy : str, optional (default="auto")
///     Strategy: "auto" balances all classes to majority.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import ADASYN
/// >>> import numpy as np
/// >>> X = np.vstack([np.random.randn(90, 2), np.random.randn(10, 2) + 3])
/// >>> y = np.array([0]*90 + [1]*10, dtype=np.float64)
/// >>> ada = ADASYN(random_state=42)
/// >>> X_res, y_res = ada.fit_resample(X, y)
/// >>> # y_res is now approximately balanced
///
/// Notes
/// -----
/// ADASYN generates more synthetic samples for minority instances that
/// are harder to learn (near the decision boundary), unlike SMOTE which
/// generates uniformly. This can improve classifier performance in
/// difficult regions.
#[pyclass(name = "ADASYN", module = "ferroml.preprocessing")]
pub struct PyADASYN {
    inner: ferroml_core::preprocessing::sampling::ADASYN,
}

#[pymethods]
impl PyADASYN {
    #[new]
    #[pyo3(signature = (k_neighbors=5, n_neighbors=5, sampling_strategy="auto", random_state=None))]
    fn new(
        k_neighbors: usize,
        n_neighbors: usize,
        sampling_strategy: &str,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let mut adasyn = ferroml_core::preprocessing::sampling::ADASYN::new()
            .with_k_neighbors(k_neighbors)
            .with_n_neighbors(n_neighbors);

        adasyn = adasyn.with_sampling_strategy(parse_sampling_strategy(sampling_strategy)?);

        if let Some(seed) = random_state {
            adasyn = adasyn.with_random_state(seed);
        }
        Ok(Self { inner: adasyn })
    }

    /// Fit to the data and generate synthetic samples.
    ///
    /// Parameters
    /// ----------
    /// X : ndarray of shape (n_samples, n_features)
    ///     Feature matrix.
    /// y : ndarray of shape (n_samples,)
    ///     Target class labels.
    ///
    /// Returns
    /// -------
    /// tuple of (X_resampled, y_resampled)
    #[allow(clippy::type_complexity)]
    fn fit_resample<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        use ferroml_core::preprocessing::sampling::Resampler;

        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        let (x_res, y_res) = self
            .inner
            .fit_resample(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok((x_res.into_pyarray(py), y_res.into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        "ADASYN()".to_string()
    }
}

// =============================================================================
// RandomUnderSampler
// =============================================================================

/// Random Under-Sampling.
///
/// Reduces the majority class by randomly removing samples.
///
/// Parameters
/// ----------
/// sampling_strategy : str, optional (default="auto")
///     Strategy: "auto" balances all classes to the minority count.
/// replacement : bool, optional (default=False)
///     Whether to sample with replacement.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import RandomUnderSampler
/// >>> import numpy as np
/// >>> X = np.vstack([np.random.randn(90, 2), np.random.randn(10, 2) + 3])
/// >>> y = np.array([0]*90 + [1]*10, dtype=np.float64)
/// >>> rus = RandomUnderSampler(random_state=42)
/// >>> X_res, y_res = rus.fit_resample(X, y)
/// >>> # Majority class reduced to match minority class count
#[pyclass(name = "RandomUnderSampler", module = "ferroml.preprocessing")]
pub struct PyRandomUnderSampler {
    inner: ferroml_core::preprocessing::sampling::RandomUnderSampler,
}

#[pymethods]
impl PyRandomUnderSampler {
    #[new]
    #[pyo3(signature = (sampling_strategy="auto", replacement=false, random_state=None))]
    fn new(
        sampling_strategy: &str,
        replacement: bool,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let mut sampler = ferroml_core::preprocessing::sampling::RandomUnderSampler::new()
            .with_replacement(replacement);

        sampler = sampler.with_sampling_strategy(parse_sampling_strategy(sampling_strategy)?);

        if let Some(seed) = random_state {
            sampler = sampler.with_random_state(seed);
        }
        Ok(Self { inner: sampler })
    }

    /// Fit to the data and undersample.
    ///
    /// Parameters
    /// ----------
    /// X : ndarray of shape (n_samples, n_features)
    ///     Feature matrix.
    /// y : ndarray of shape (n_samples,)
    ///     Target class labels.
    ///
    /// Returns
    /// -------
    /// tuple of (X_resampled, y_resampled)
    #[allow(clippy::type_complexity)]
    fn fit_resample<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        use ferroml_core::preprocessing::sampling::Resampler;

        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        let (x_res, y_res) = self
            .inner
            .fit_resample(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok((x_res.into_pyarray(py), y_res.into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        "RandomUnderSampler()".to_string()
    }
}

// =============================================================================
// RandomOverSampler
// =============================================================================

/// Random Over-Sampling.
///
/// Increases the minority class by randomly duplicating samples.
///
/// Parameters
/// ----------
/// sampling_strategy : str, optional (default="auto")
///     Strategy: "auto" balances all classes to the majority count.
/// shrinkage : float or None, optional (default=None)
///     If provided, adds slight noise to duplicated samples using a smoothed
///     bootstrap approach. Higher values add more noise.
/// random_state : int or None, optional (default=None)
///     Random seed for reproducibility.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import RandomOverSampler
/// >>> import numpy as np
/// >>> X = np.vstack([np.random.randn(90, 2), np.random.randn(10, 2) + 3])
/// >>> y = np.array([0]*90 + [1]*10, dtype=np.float64)
/// >>> ros = RandomOverSampler(random_state=42)
/// >>> X_res, y_res = ros.fit_resample(X, y)
/// >>> # Minority class duplicated to match majority class count
#[pyclass(name = "RandomOverSampler", module = "ferroml.preprocessing")]
pub struct PyRandomOverSampler {
    inner: ferroml_core::preprocessing::sampling::RandomOverSampler,
}

#[pymethods]
impl PyRandomOverSampler {
    #[new]
    #[pyo3(signature = (sampling_strategy="auto", shrinkage=None, random_state=None))]
    fn new(
        sampling_strategy: &str,
        shrinkage: Option<f64>,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        let mut sampler = ferroml_core::preprocessing::sampling::RandomOverSampler::new();

        sampler = sampler.with_sampling_strategy(parse_sampling_strategy(sampling_strategy)?);

        if let Some(s) = shrinkage {
            sampler = sampler.with_shrinkage(s);
        }
        if let Some(seed) = random_state {
            sampler = sampler.with_random_state(seed);
        }
        Ok(Self { inner: sampler })
    }

    /// Fit to the data and oversample.
    ///
    /// Parameters
    /// ----------
    /// X : ndarray of shape (n_samples, n_features)
    ///     Feature matrix.
    /// y : ndarray of shape (n_samples,)
    ///     Target class labels.
    ///
    /// Returns
    /// -------
    /// tuple of (X_resampled, y_resampled)
    #[allow(clippy::type_complexity)]
    fn fit_resample<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        use ferroml_core::preprocessing::sampling::Resampler;

        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        let (x_res, y_res) = self
            .inner
            .fit_resample(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok((x_res.into_pyarray(py), y_res.into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        "RandomOverSampler()".to_string()
    }
}

// =============================================================================
// Shared parsing helpers
// =============================================================================

fn parse_handle_unknown(handle_unknown: &str) -> PyResult<UnknownCategoryHandling> {
    match handle_unknown {
        "error" => Ok(UnknownCategoryHandling::Error),
        "ignore" => Ok(UnknownCategoryHandling::Ignore),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown handle_unknown value: '{}'. Use 'error' or 'ignore'",
            handle_unknown
        ))),
    }
}

fn parse_sampling_strategy(
    strategy: &str,
) -> PyResult<ferroml_core::preprocessing::sampling::SamplingStrategy> {
    use ferroml_core::preprocessing::sampling::SamplingStrategy;
    match strategy {
        "auto" => Ok(SamplingStrategy::Auto),
        other => other
            .parse::<f64>()
            .map(SamplingStrategy::Ratio)
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown sampling_strategy: '{}'. Use 'auto' or a float ratio.",
                    other
                ))
            }),
    }
}

// =============================================================================
// RecursiveFeatureElimination (RFE) — factory pattern
// =============================================================================

/// Recursive Feature Elimination (RFE).
///
/// Selects features by recursively fitting an estimator, ranking features
/// by importance, and eliminating the least important features at each step.
///
/// RFE cannot be constructed directly because the underlying Rust trait object
/// (`dyn FeatureImportanceEstimator`) cannot cross the PyO3 boundary. Instead,
/// use one of the factory staticmethods to specify the estimator type:
///
/// - ``RecursiveFeatureElimination.with_linear_regression(...)``
/// - ``RecursiveFeatureElimination.with_ridge(...)``
/// - ``RecursiveFeatureElimination.with_lasso(...)``
/// - ``RecursiveFeatureElimination.with_logistic_regression(...)``
/// - ``RecursiveFeatureElimination.with_decision_tree_classifier(...)``
/// - ``RecursiveFeatureElimination.with_decision_tree_regressor(...)``
/// - ``RecursiveFeatureElimination.with_random_forest_classifier(...)``
/// - ``RecursiveFeatureElimination.with_random_forest_regressor(...)``
/// - ``RecursiveFeatureElimination.with_gradient_boosting_classifier(...)``
/// - ``RecursiveFeatureElimination.with_gradient_boosting_regressor(...)``
/// - ``RecursiveFeatureElimination.with_extra_trees_classifier(...)``
/// - ``RecursiveFeatureElimination.with_extra_trees_regressor(...)``
/// - ``RecursiveFeatureElimination.with_svr(...)``
///
/// Parameters (common to all factories)
/// -------------------------------------
/// n_features_to_select : int, optional
///     Number of features to select. If None, selects half of features.
/// step : int, optional (default=1)
///     Number of features to remove at each iteration.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import RecursiveFeatureElimination
/// >>> import numpy as np
/// >>> X = np.random.randn(100, 10)
/// >>> y = X[:, 0] * 2 + X[:, 3] * 5 + np.random.randn(100) * 0.1
/// >>> rfe = RecursiveFeatureElimination.with_linear_regression(n_features_to_select=3)
/// >>> rfe.fit(X, y)
/// >>> print(rfe.selected_indices_)
/// >>> X_selected = rfe.transform(X)
#[pyclass(name = "RecursiveFeatureElimination", module = "ferroml.preprocessing")]
pub struct PyRFE {
    inner: RecursiveFeatureElimination,
    estimator_name: &'static str,
    n_features_to_select_cfg: Option<usize>,
    step_cfg: usize,
}

/// Build an RFE instance from a boxed FeatureImportanceEstimator.
fn build_rfe(
    estimator: Box<dyn ferroml_core::preprocessing::selection::FeatureImportanceEstimator>,
    estimator_name: &'static str,
    n_features_to_select: Option<usize>,
    step: usize,
) -> PyRFE {
    let mut rfe = RecursiveFeatureElimination::new(estimator);
    if let Some(n) = n_features_to_select {
        rfe = rfe.with_n_features_to_select(n);
    }
    rfe = rfe.with_step(step);
    PyRFE {
        inner: rfe,
        estimator_name,
        n_features_to_select_cfg: n_features_to_select,
        step_cfg: step,
    }
}

#[pymethods]
impl PyRFE {
    // =========================================================================
    // Factory staticmethods — one per concrete estimator type
    // =========================================================================

    /// Create RFE with a LinearRegression estimator.
    ///
    /// Uses absolute standardized coefficients as feature importances.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// fit_intercept : bool, optional (default=True)
    ///     Whether to fit an intercept.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, fit_intercept=true))]
    fn with_linear_regression(
        n_features_to_select: Option<usize>,
        step: usize,
        fit_intercept: bool,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = LinearRegression::new().with_fit_intercept(fit_intercept);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "LinearRegression failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "LinearRegression",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a RidgeRegression estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// alpha : float, optional (default=1.0)
    ///     Regularization strength.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, alpha=1.0))]
    fn with_ridge(n_features_to_select: Option<usize>, step: usize, alpha: f64) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = RidgeRegression::new(alpha);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "RidgeRegression failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "RidgeRegression",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a LassoRegression estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// alpha : float, optional (default=1.0)
    ///     Regularization strength.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, alpha=1.0))]
    fn with_lasso(n_features_to_select: Option<usize>, step: usize, alpha: f64) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = LassoRegression::new(alpha);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "LassoRegression failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "LassoRegression",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a LogisticRegression estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// max_iter : int, optional (default=100)
    ///     Maximum iterations for convergence.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, max_iter=100))]
    fn with_logistic_regression(
        n_features_to_select: Option<usize>,
        step: usize,
        max_iter: usize,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = LogisticRegression::new().with_max_iter(max_iter);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "LogisticRegression failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "LogisticRegression",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a DecisionTreeClassifier estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// max_depth : int, optional
    ///     Maximum tree depth.
    /// min_samples_split : int, optional (default=2)
    ///     Minimum samples to split an internal node.
    /// min_samples_leaf : int, optional (default=1)
    ///     Minimum samples at a leaf node.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, max_depth=None, min_samples_split=2, min_samples_leaf=1))]
    fn with_decision_tree_classifier(
        n_features_to_select: Option<usize>,
        step: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = DecisionTreeClassifier::new()
                    .with_max_depth(max_depth)
                    .with_min_samples_split(min_samples_split)
                    .with_min_samples_leaf(min_samples_leaf);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "DecisionTreeClassifier failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "DecisionTreeClassifier",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a DecisionTreeRegressor estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// max_depth : int, optional
    ///     Maximum tree depth.
    /// min_samples_split : int, optional (default=2)
    ///     Minimum samples to split an internal node.
    /// min_samples_leaf : int, optional (default=1)
    ///     Minimum samples at a leaf node.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, max_depth=None, min_samples_split=2, min_samples_leaf=1))]
    fn with_decision_tree_regressor(
        n_features_to_select: Option<usize>,
        step: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = DecisionTreeRegressor::new()
                    .with_max_depth(max_depth)
                    .with_min_samples_split(min_samples_split)
                    .with_min_samples_leaf(min_samples_leaf);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "DecisionTreeRegressor failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "DecisionTreeRegressor",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a RandomForestClassifier estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// n_estimators : int, optional (default=100)
    ///     Number of trees in the forest.
    /// max_depth : int, optional
    ///     Maximum tree depth.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, n_estimators=100, max_depth=None))]
    fn with_random_forest_classifier(
        n_features_to_select: Option<usize>,
        step: usize,
        n_estimators: usize,
        max_depth: Option<usize>,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = RandomForestClassifier::new()
                    .with_n_estimators(n_estimators)
                    .with_max_depth(max_depth);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "RandomForestClassifier failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "RandomForestClassifier",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a RandomForestRegressor estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// n_estimators : int, optional (default=100)
    ///     Number of trees in the forest.
    /// max_depth : int, optional
    ///     Maximum tree depth.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, n_estimators=100, max_depth=None))]
    fn with_random_forest_regressor(
        n_features_to_select: Option<usize>,
        step: usize,
        n_estimators: usize,
        max_depth: Option<usize>,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = RandomForestRegressor::new()
                    .with_n_estimators(n_estimators)
                    .with_max_depth(max_depth);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "RandomForestRegressor failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "RandomForestRegressor",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a GradientBoostingClassifier estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// n_estimators : int, optional (default=100)
    ///     Number of boosting stages.
    /// max_depth : int, optional (default=3)
    ///     Maximum depth of individual trees.
    /// learning_rate : float, optional (default=0.1)
    ///     Shrinkage factor applied to each tree.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, n_estimators=100, max_depth=Some(3), learning_rate=0.1))]
    fn with_gradient_boosting_classifier(
        n_features_to_select: Option<usize>,
        step: usize,
        n_estimators: usize,
        max_depth: Option<usize>,
        learning_rate: f64,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = GradientBoostingClassifier::new()
                    .with_n_estimators(n_estimators)
                    .with_max_depth(max_depth)
                    .with_learning_rate(learning_rate);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "GradientBoostingClassifier failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "GradientBoostingClassifier",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with a GradientBoostingRegressor estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// n_estimators : int, optional (default=100)
    ///     Number of boosting stages.
    /// max_depth : int, optional (default=3)
    ///     Maximum depth of individual trees.
    /// learning_rate : float, optional (default=0.1)
    ///     Shrinkage factor applied to each tree.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, n_estimators=100, max_depth=Some(3), learning_rate=0.1))]
    fn with_gradient_boosting_regressor(
        n_features_to_select: Option<usize>,
        step: usize,
        n_estimators: usize,
        max_depth: Option<usize>,
        learning_rate: f64,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = GradientBoostingRegressor::new()
                    .with_n_estimators(n_estimators)
                    .with_max_depth(max_depth)
                    .with_learning_rate(learning_rate);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "GradientBoostingRegressor failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "GradientBoostingRegressor",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with an ExtraTreesClassifier estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// n_estimators : int, optional (default=100)
    ///     Number of trees in the ensemble.
    /// max_depth : int, optional
    ///     Maximum tree depth.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, n_estimators=100, max_depth=None))]
    fn with_extra_trees_classifier(
        n_features_to_select: Option<usize>,
        step: usize,
        n_estimators: usize,
        max_depth: Option<usize>,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = ExtraTreesClassifier::new()
                    .with_n_estimators(n_estimators)
                    .with_max_depth(max_depth);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "ExtraTreesClassifier failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "ExtraTreesClassifier",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with an ExtraTreesRegressor estimator.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// n_estimators : int, optional (default=100)
    ///     Number of trees in the ensemble.
    /// max_depth : int, optional
    ///     Maximum tree depth.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, n_estimators=100, max_depth=None))]
    fn with_extra_trees_regressor(
        n_features_to_select: Option<usize>,
        step: usize,
        n_estimators: usize,
        max_depth: Option<usize>,
    ) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = ExtraTreesRegressor::new()
                    .with_n_estimators(n_estimators)
                    .with_max_depth(max_depth);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "ExtraTreesRegressor failed to produce feature importances",
                    )
                })
            });
        build_rfe(
            Box::new(estimator),
            "ExtraTreesRegressor",
            n_features_to_select,
            step,
        )
    }

    /// Create RFE with an SVR (Support Vector Regressor) estimator.
    ///
    /// Uses linear kernel by default; absolute weights used as importances.
    ///
    /// Parameters
    /// ----------
    /// n_features_to_select : int, optional
    ///     Number of features to select. If None, selects half.
    /// step : int, optional (default=1)
    ///     Number of features to remove per iteration.
    /// c : float, optional (default=1.0)
    ///     Regularization parameter.
    /// epsilon : float, optional (default=0.1)
    ///     Epsilon in the epsilon-SVR model.
    #[staticmethod]
    #[pyo3(signature = (n_features_to_select=None, step=1, c=1.0, epsilon=0.1))]
    fn with_svr(n_features_to_select: Option<usize>, step: usize, c: f64, epsilon: f64) -> Self {
        let estimator =
            ClosureEstimator::new(move |x: &ndarray::Array2<f64>, y: &ndarray::Array1<f64>| {
                let mut model = SVR::new().with_c(c).with_epsilon(epsilon);
                model.fit(x, y)?;
                model.feature_importance().ok_or_else(|| {
                    ferroml_core::FerroError::invalid_input(
                        "SVR failed to produce feature importances",
                    )
                })
            });
        build_rfe(Box::new(estimator), "SVR", n_features_to_select, step)
    }

    // =========================================================================
    // Fitting and transforming
    // =========================================================================

    /// Fit the RFE selector on training data.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training feature matrix.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self : RecursiveFeatureElimination
    ///     Fitted selector.
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
            .fit_with_target(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform data by selecting only the chosen features.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Data to transform.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_selected_features)
    ///     Data with only the selected features.
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

    /// Fit the selector and transform the data in one step.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training feature matrix.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_selected_features)
    ///     Data with only the selected features.
    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        check_array1_finite(&y)?;
        let y_arr = to_owned_array_1d(y);
        self.inner
            .fit_with_target(&x_arr, &y_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    // =========================================================================
    // Accessor properties
    // =========================================================================

    /// Feature rankings. 1 = selected, higher values = eliminated earlier.
    ///
    /// Returns
    /// -------
    /// ranking : ndarray of shape (n_features,)
    ///     Integer ranking for each original feature.
    #[getter]
    fn ranking_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ranking = self.inner.ranking().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("RFE not fitted. Call fit() first.")
        })?;
        Ok(array1_usize_into_pyarray_i64(py, ranking.clone()))
    }

    /// Boolean mask of selected features.
    ///
    /// Returns
    /// -------
    /// support : list of bool
    ///     True for features that are selected, False otherwise.
    #[getter]
    fn support_(&self) -> PyResult<Vec<bool>> {
        self.inner.get_support().map(|s| s.to_vec()).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("RFE not fitted. Call fit() first.")
        })
    }

    /// Indices of selected features (sorted).
    ///
    /// Returns
    /// -------
    /// indices : list of int
    ///     Sorted indices of selected features.
    #[getter]
    fn selected_indices_(&self) -> PyResult<Vec<usize>> {
        self.inner
            .selected_indices()
            .map(|s| s.to_vec())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("RFE not fitted. Call fit() first.")
            })
    }

    /// Number of elimination iterations performed.
    ///
    /// Returns
    /// -------
    /// n_iterations : int
    ///     Number of RFE iterations.
    #[getter]
    fn n_iterations_(&self) -> PyResult<usize> {
        self.inner.n_iterations().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("RFE not fitted. Call fit() first.")
        })
    }

    /// Configured number of features to select.
    ///
    /// Returns
    /// -------
    /// n_features_to_select : int or None
    ///     The configured value, or None if using the default (half).
    #[getter]
    fn n_features_to_select_(&self) -> Option<usize> {
        self.n_features_to_select_cfg
    }

    fn __repr__(&self) -> String {
        let nf = match self.n_features_to_select_cfg {
            Some(n) => format!("{}", n),
            None => "None".to_string(),
        };
        format!(
            "RecursiveFeatureElimination(estimator={}, n_features_to_select={}, step={})",
            self.estimator_name, nf, self.step_cfg
        )
    }
}

// =============================================================================
// TfidfTransformer
// =============================================================================

/// Transform a count matrix to a TF-IDF representation.
///
/// TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure
/// used to evaluate the importance of a word in a document relative to a
/// collection of documents (corpus).
///
/// Parameters
/// ----------
/// norm : str, optional (default='l2')
///     Normalization to apply: 'l1', 'l2', or 'none'.
/// use_idf : bool, optional (default=True)
///     Enable inverse-document-frequency reweighting.
/// smooth_idf : bool, optional (default=True)
///     Smooth IDF weights by adding one to document frequencies.
/// sublinear_tf : bool, optional (default=False)
///     Apply sublinear TF scaling: replace tf with 1 + log(tf).
///
/// Attributes
/// ----------
/// idf_ : ndarray of shape (n_features,)
///     The inverse document frequency vector (only if use_idf=True).
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import TfidfTransformer
/// >>> import numpy as np
/// >>> X = np.array([[1, 0, 2], [0, 1, 1], [1, 1, 0]], dtype=np.float64)
/// >>> tfidf = TfidfTransformer()
/// >>> X_tfidf = tfidf.fit_transform(X)
#[pyclass(name = "TfidfTransformer", module = "ferroml.preprocessing")]
pub struct PyTfidfTransformer {
    inner: ferroml_core::preprocessing::tfidf::TfidfTransformer,
}

#[pymethods]
impl PyTfidfTransformer {
    /// Create a new TfidfTransformer.
    #[new]
    #[pyo3(signature = (norm="l2", use_idf=true, smooth_idf=true, sublinear_tf=false))]
    fn new(norm: &str, use_idf: bool, smooth_idf: bool, sublinear_tf: bool) -> PyResult<Self> {
        use ferroml_core::preprocessing::tfidf::TfidfNorm;

        let norm_enum = match norm {
            "l1" => TfidfNorm::L1,
            "l2" => TfidfNorm::L2,
            "none" | "" => TfidfNorm::None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "norm must be 'l1', 'l2', or 'none'",
                ))
            }
        };

        Ok(Self {
            inner: ferroml_core::preprocessing::tfidf::TfidfTransformer::new()
                .with_norm(norm_enum)
                .with_use_idf(use_idf)
                .with_smooth_idf(smooth_idf)
                .with_sublinear_tf(sublinear_tf),
        })
    }

    /// Fit the transformer to a count matrix.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Term-count matrix.
    ///
    /// Returns
    /// -------
    /// self : TfidfTransformer
    ///     Fitted transformer.
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

    /// Transform a count matrix to TF-IDF representation.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Term-count matrix.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_features)
    ///     TF-IDF weighted matrix.
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
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Term-count matrix.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_features)
    ///     TF-IDF weighted matrix.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// The inverse document frequency vector.
    ///
    /// Returns
    /// -------
    /// idf : ndarray of shape (n_features,) or None
    ///     IDF weights, or None if use_idf=False or not fitted.
    #[getter]
    fn idf_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self.inner.idf().map(|idf| idf.clone().into_pyarray(py)))
    }

    /// Fit the transformer from a scipy.sparse matrix (native sparse, no densification).
    ///
    /// Parameters
    /// ----------
    /// X : scipy.sparse matrix (CSR or CSC)
    ///     Sparse term-count matrix.
    ///
    /// Returns
    /// -------
    /// self : TfidfTransformer
    ///     Fitted transformer.
    #[cfg(feature = "sparse")]
    fn fit_sparse<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let csr = py_csr_to_ferro(x)?;
        slf.inner
            .fit_sparse(&csr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform a scipy.sparse count matrix to TF-IDF (returns dense ndarray).
    ///
    /// Parameters
    /// ----------
    /// X : scipy.sparse matrix (CSR or CSC)
    ///     Sparse term-count matrix.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_features)
    ///     TF-IDF weighted matrix.
    #[cfg(feature = "sparse")]
    fn transform_sparse<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let csr = py_csr_to_ferro(x)?;
        let result = self
            .inner
            .transform_sparse(&csr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform a scipy.sparse count matrix in one step.
    ///
    /// Parameters
    /// ----------
    /// X : scipy.sparse matrix (CSR or CSC)
    ///     Sparse term-count matrix.
    ///
    /// Returns
    /// -------
    /// X_new : ndarray of shape (n_samples, n_features)
    ///     TF-IDF weighted matrix.
    #[cfg(feature = "sparse")]
    fn fit_transform_sparse<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let csr = py_csr_to_ferro(x)?;
        slf.inner
            .fit_sparse(&csr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        let result = slf
            .inner
            .transform_sparse(&csr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        "TfidfTransformer()".to_string()
    }
}

// =============================================================================
// CountVectorizer
// =============================================================================

/// Convert text documents to a matrix of token counts.
///
/// CountVectorizer tokenizes text, builds a vocabulary from training data,
/// and transforms documents into term-count matrices.
///
/// Parameters
/// ----------
/// max_features : int, optional (default=None)
///     Build a vocabulary that only considers the top max_features ordered
///     by term frequency across the corpus.
/// min_df : int, optional (default=1)
///     Ignore terms that appear in fewer than min_df documents (absolute count).
/// max_df : float, optional (default=1.0)
///     Ignore terms that appear in more than max_df fraction of documents.
/// ngram_range : tuple (min_n, max_n), optional (default=(1, 1))
///     The lower and upper boundary of the range of n-values for different
///     n-grams to be extracted.
/// binary : bool, optional (default=False)
///     If True, all non-zero counts are set to 1.
/// lowercase : bool, optional (default=True)
///     Convert all characters to lowercase before tokenizing.
/// stop_words : list of str, optional (default=None)
///     List of stop words to remove during tokenization.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import CountVectorizer
/// >>> corpus = ["the cat sat", "the dog sat"]
/// >>> cv = CountVectorizer()
/// >>> X = cv.fit_transform(corpus)
#[pyclass(name = "CountVectorizer", module = "ferroml.preprocessing")]
#[derive(Clone)]
pub struct PyCountVectorizer {
    inner: ferroml_core::preprocessing::count_vectorizer::CountVectorizer,
}

#[pymethods]
impl PyCountVectorizer {
    /// Create a new CountVectorizer.
    #[new]
    #[pyo3(signature = (max_features=None, min_df=1, max_df=1.0, ngram_range=(1,1), binary=false, lowercase=true, stop_words=None))]
    fn new(
        max_features: Option<usize>,
        min_df: usize,
        max_df: f64,
        ngram_range: (usize, usize),
        binary: bool,
        lowercase: bool,
        stop_words: Option<Vec<String>>,
    ) -> Self {
        use ferroml_core::preprocessing::count_vectorizer::{CountVectorizer, DocFrequency};

        let mut cv = CountVectorizer::new()
            .with_min_df(DocFrequency::Count(min_df))
            .with_max_df(DocFrequency::Fraction(max_df))
            .with_ngram_range(ngram_range)
            .with_binary(binary)
            .with_lowercase(lowercase);

        if let Some(mf) = max_features {
            cv = cv.with_max_features(mf);
        }

        if let Some(sw) = stop_words {
            cv = cv.with_stop_words(sw);
        }

        Self { inner: cv }
    }

    /// Fit the vectorizer on a corpus of documents.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     The text documents to learn vocabulary from.
    ///
    /// Returns
    /// -------
    /// self : CountVectorizer
    ///     Fitted vectorizer.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        documents: Vec<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        use ferroml_core::preprocessing::count_vectorizer::TextTransformer;
        slf.inner
            .fit_text(&documents)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform documents into a term-count matrix.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     The text documents to transform.
    ///
    /// Returns
    /// -------
    /// X : ndarray of shape (n_documents, n_features)
    ///     Document-term matrix.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        documents: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let result = self
            .inner
            .transform_text_dense(&documents)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// Fit and transform in one step.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     The text documents to fit on and transform.
    ///
    /// Returns
    /// -------
    /// X : ndarray of shape (n_documents, n_features)
    ///     Document-term matrix.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        documents: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let result = slf
            .inner
            .fit_transform_text_dense(&documents)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// The learned vocabulary mapping (term -> index).
    ///
    /// Returns
    /// -------
    /// vocabulary : dict
    ///     Mapping from term to column index.
    #[getter]
    fn vocabulary_(&self) -> PyResult<std::collections::HashMap<String, usize>> {
        self.inner.vocabulary().cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("CountVectorizer is not fitted yet")
        })
    }

    /// Get the feature names (sorted vocabulary terms).
    ///
    /// Returns
    /// -------
    /// feature_names : list of str
    ///     Sorted list of vocabulary terms.
    fn get_feature_names_out(&self) -> PyResult<Vec<String>> {
        self.inner
            .get_feature_names()
            .map(|names| names.to_vec())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "CountVectorizer is not fitted yet",
                )
            })
    }

    fn __repr__(&self) -> String {
        "CountVectorizer()".to_string()
    }
}

// =============================================================================
// TfidfVectorizer
// =============================================================================

/// Convert text documents to a TF-IDF weighted sparse matrix.
///
/// TfidfVectorizer combines CountVectorizer and TfidfTransformer into a single step.
/// It tokenizes text, builds a vocabulary, computes term frequencies, and applies
/// IDF weighting.
///
/// Parameters
/// ----------
/// max_features : int, optional (default=None)
///     Build a vocabulary that only considers the top max_features.
/// min_df : int, optional (default=1)
///     Ignore terms appearing in fewer documents.
/// max_df : float, optional (default=1.0)
///     Ignore terms appearing in more than this fraction of documents.
/// ngram_range : tuple (min_n, max_n), optional (default=(1, 1))
///     Range of n-gram sizes.
/// binary : bool, optional (default=False)
///     If True, all non-zero counts are set to 1 before TF-IDF.
/// lowercase : bool, optional (default=True)
///     Convert text to lowercase before tokenizing.
/// stop_words : list of str, optional (default=None)
///     Stop words to remove.
/// norm : str, optional (default='l2')
///     Normalization: 'l1', 'l2', or 'none'.
/// use_idf : bool, optional (default=True)
///     Enable IDF weighting.
/// smooth_idf : bool, optional (default=True)
///     Smooth IDF weights.
/// sublinear_tf : bool, optional (default=False)
///     Apply sublinear TF scaling (1 + log(tf)).
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import TfidfVectorizer
/// >>> corpus = ["the cat sat", "the dog sat"]
/// >>> tv = TfidfVectorizer()
/// >>> X = tv.fit_transform(corpus)  # returns scipy.sparse.csr_matrix
#[pyclass(name = "TfidfVectorizer", module = "ferroml.preprocessing")]
#[derive(Clone)]
pub struct PyTfidfVectorizer {
    inner: ferroml_core::preprocessing::tfidf_vectorizer::TfidfVectorizer,
}

#[pymethods]
impl PyTfidfVectorizer {
    #[new]
    #[pyo3(signature = (max_features=None, min_df=1, max_df=1.0, ngram_range=(1,1), binary=false, lowercase=true, stop_words=None, norm="l2", use_idf=true, smooth_idf=true, sublinear_tf=false))]
    fn new(
        max_features: Option<usize>,
        min_df: usize,
        max_df: f64,
        ngram_range: (usize, usize),
        binary: bool,
        lowercase: bool,
        stop_words: Option<Vec<String>>,
        norm: &str,
        use_idf: bool,
        smooth_idf: bool,
        sublinear_tf: bool,
    ) -> PyResult<Self> {
        use ferroml_core::preprocessing::count_vectorizer::DocFrequency;
        use ferroml_core::preprocessing::tfidf::TfidfNorm;
        use ferroml_core::preprocessing::tfidf_vectorizer::TfidfVectorizer;

        let norm_enum = match norm {
            "l1" => TfidfNorm::L1,
            "l2" => TfidfNorm::L2,
            "none" | "" => TfidfNorm::None,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "norm must be 'l1', 'l2', or 'none'",
                ))
            }
        };

        let mut tv = TfidfVectorizer::new()
            .with_min_df(DocFrequency::Count(min_df))
            .with_max_df(DocFrequency::Fraction(max_df))
            .with_ngram_range(ngram_range)
            .with_binary(binary)
            .with_lowercase(lowercase)
            .with_norm(norm_enum)
            .with_use_idf(use_idf)
            .with_smooth_idf(smooth_idf)
            .with_sublinear_tf(sublinear_tf);

        if let Some(mf) = max_features {
            tv = tv.with_max_features(mf);
        }
        if let Some(sw) = stop_words {
            tv = tv.with_stop_words(sw);
        }

        Ok(Self { inner: tv })
    }

    /// Fit the vectorizer on a corpus of documents.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     The text documents to learn vocabulary from.
    ///
    /// Returns
    /// -------
    /// self : TfidfVectorizer
    ///     Fitted vectorizer.
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        documents: Vec<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        use ferroml_core::preprocessing::count_vectorizer::TextTransformer;
        slf.inner
            .fit_text(&documents)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(slf)
    }

    /// Transform documents to TF-IDF sparse matrix.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     The text documents to transform.
    ///
    /// Returns
    /// -------
    /// X : scipy.sparse.csr_matrix
    ///     TF-IDF weighted sparse matrix.
    #[cfg(feature = "sparse")]
    fn transform(&self, py: Python<'_>, documents: Vec<String>) -> PyResult<PyObject> {
        use ferroml_core::preprocessing::count_vectorizer::TextTransformer;
        let csr = self
            .inner
            .transform_text(&documents)
            .map_err(crate::errors::ferro_to_pyerr)?;
        crate::sparse_utils::ferro_csr_to_py(&csr, py)
    }

    /// Fit and transform in one step. Returns scipy.sparse.csr_matrix.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     The text documents to fit on and transform.
    ///
    /// Returns
    /// -------
    /// X : scipy.sparse.csr_matrix
    ///     TF-IDF weighted sparse matrix.
    #[cfg(feature = "sparse")]
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        documents: Vec<String>,
    ) -> PyResult<PyObject> {
        use ferroml_core::preprocessing::count_vectorizer::TextTransformer;
        slf.inner
            .fit_text(&documents)
            .map_err(crate::errors::ferro_to_pyerr)?;
        let csr = slf
            .inner
            .transform_text(&documents)
            .map_err(crate::errors::ferro_to_pyerr)?;
        crate::sparse_utils::ferro_csr_to_py(&csr, py)
    }

    /// Transform documents to a dense numpy array.
    ///
    /// Parameters
    /// ----------
    /// documents : list of str
    ///     The text documents to transform.
    ///
    /// Returns
    /// -------
    /// X : ndarray of shape (n_documents, n_features)
    ///     TF-IDF weighted dense matrix.
    fn transform_dense<'py>(
        &self,
        py: Python<'py>,
        documents: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let result = self
            .inner
            .transform_text_dense(&documents)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    /// The learned vocabulary mapping (term -> index).
    ///
    /// Returns
    /// -------
    /// vocabulary : dict
    ///     Mapping from term to column index.
    #[getter]
    fn vocabulary_(&self) -> PyResult<std::collections::HashMap<String, usize>> {
        self.inner.vocabulary().cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("TfidfVectorizer is not fitted yet")
        })
    }

    /// Get the feature names (sorted vocabulary terms).
    ///
    /// Returns
    /// -------
    /// feature_names : list of str
    ///     Sorted list of vocabulary terms.
    fn get_feature_names_out(&self) -> PyResult<Vec<String>> {
        self.inner
            .get_feature_names()
            .map(|names| names.to_vec())
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "TfidfVectorizer is not fitted yet",
                )
            })
    }

    /// The IDF weight vector.
    #[getter]
    fn idf_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        Ok(self.inner.idf().map(|idf| idf.clone().into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        "TfidfVectorizer()".to_string()
    }
}

// =============================================================================
// Module registration
// =============================================================================

// =============================================================================
// Normalizer
// =============================================================================

/// Normalize samples individually to unit norm.
///
/// Each sample (row) is scaled independently to have unit norm.
///
/// Parameters
/// ----------
/// norm : str, optional (default="l2")
///     The norm to use: "l1" (sum of absolute values), "l2" (Euclidean),
///     or "max" (maximum absolute value).
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import Normalizer
/// >>> import numpy as np
/// >>> X = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
/// >>> norm = Normalizer(norm="l2")
/// >>> X_norm = norm.fit_transform(X)
/// >>> # First row: [0.6, 0.8] (unit L2 norm)
///
/// Notes
/// -----
/// Normalizer operates on each sample (row) independently, not on
/// features (columns). This is useful for text classification (TF-IDF
/// vectors) or when feature magnitudes vary across samples.
#[pyclass(name = "Normalizer")]
pub struct PyNormalizer {
    inner: Normalizer,
}

#[pymethods]
impl PyNormalizer {
    #[new]
    #[pyo3(signature = (norm="l2"))]
    fn new(norm: &str) -> PyResult<Self> {
        let norm_type = match norm {
            "l1" => NormType::L1,
            "l2" => NormType::L2,
            "max" => NormType::Max,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown norm '{}'. Use 'l1', 'l2', or 'max'.",
                    norm
                )));
            }
        };
        Ok(Self {
            inner: Normalizer::new().with_norm(norm_type),
        })
    }

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

    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_array_finite(&x)?;
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(crate::errors::ferro_to_pyerr)?;
        Ok(result.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        let norm_str = match self.inner.norm {
            NormType::L1 => "l1",
            NormType::L2 => "l2",
            NormType::Max => "max",
        };
        format!("Normalizer(norm='{}')", norm_str)
    }
}

pub fn register_preprocessing_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let preprocessing_module = PyModule::new(parent_module.py(), "preprocessing")?;

    // Scalers
    preprocessing_module.add_class::<PyStandardScaler>()?;
    preprocessing_module.add_class::<PyMinMaxScaler>()?;
    preprocessing_module.add_class::<PyRobustScaler>()?;
    preprocessing_module.add_class::<PyMaxAbsScaler>()?;

    // Encoders
    preprocessing_module.add_class::<PyOneHotEncoder>()?;
    preprocessing_module.add_class::<PyOrdinalEncoder>()?;
    preprocessing_module.add_class::<PyLabelEncoder>()?;

    // Imputers
    preprocessing_module.add_class::<PySimpleImputer>()?;
    preprocessing_module.add_class::<PyKNNImputer>()?;

    // Transformers
    preprocessing_module.add_class::<PyPowerTransformer>()?;
    preprocessing_module.add_class::<PyQuantileTransformer>()?;
    preprocessing_module.add_class::<PyPolynomialFeatures>()?;
    preprocessing_module.add_class::<PyKBinsDiscretizer>()?;

    // Feature selectors
    preprocessing_module.add_class::<PyVarianceThreshold>()?;
    preprocessing_module.add_class::<PySelectKBest>()?;

    // Encoders (additional)
    preprocessing_module.add_class::<PyTargetEncoder>()?;

    // Feature selector (model-based)
    preprocessing_module.add_class::<PySelectFromModel>()?;

    // Resampling (imbalanced data)
    preprocessing_module.add_class::<PySMOTE>()?;
    preprocessing_module.add_class::<PyADASYN>()?;
    preprocessing_module.add_class::<PyRandomUnderSampler>()?;
    preprocessing_module.add_class::<PyRandomOverSampler>()?;

    // Recursive Feature Elimination (factory pattern)
    preprocessing_module.add_class::<PyRFE>()?;

    // TF-IDF
    preprocessing_module.add_class::<PyTfidfTransformer>()?;

    // CountVectorizer
    preprocessing_module.add_class::<PyCountVectorizer>()?;

    // TfidfVectorizer
    preprocessing_module.add_class::<PyTfidfVectorizer>()?;

    // Normalizer
    preprocessing_module.add_class::<PyNormalizer>()?;

    parent_module.add_submodule(&preprocessing_module)?;

    Ok(())
}
