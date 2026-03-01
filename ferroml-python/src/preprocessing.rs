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

use crate::array_utils::{to_owned_array_1d, to_owned_array_2d};
use crate::pickle::{getstate, setstate};
use ferroml_core::preprocessing::{
    discretizers::KBinsDiscretizer,
    encoders::{LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder},
    imputers::{ImputeStrategy, KNNImputer, SimpleImputer},
    polynomial::PolynomialFeatures,
    power::PowerTransformer,
    quantile::QuantileTransformer,
    scalers::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler},
    selection::{SelectKBest, VarianceThreshold},
    Transformer, UnknownCategoryHandling,
};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::Py;

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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        let x_arr = to_owned_array_2d(x);
        let result = slf
            .inner
            .fit_transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        let x_arr = to_owned_array_2d(x);
        let result = self
            .inner
            .inverse_transform(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Transform features by scaling to the feature range.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Undo the scaling transformation.
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Center and scale the data.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Undo the centering and scaling.
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Scale features by dividing by max absolute value.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Undo the scaling.
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
        let unknown_handling = match handle_unknown {
            "error" => UnknownCategoryHandling::Error,
            "ignore" => UnknownCategoryHandling::Ignore,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown handle_unknown value: {}. Use 'error' or 'ignore'",
                    handle_unknown
                )));
            }
        };

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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Transform X to one-hot encoding.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Convert back to original categorical representation.
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
        let unknown_handling = match handle_unknown {
            "error" => UnknownCategoryHandling::Error,
            "ignore" => UnknownCategoryHandling::Ignore,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown handle_unknown value: {}. Use 'error' or 'ignore'",
                    handle_unknown
                )));
            }
        };

        let inner = OrdinalEncoder::new().with_handle_unknown(unknown_handling);
        Ok(Self { inner })
    }

    /// Fit the encoder to the data.
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

    /// Transform X to ordinal codes.
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Convert back to original categorical representation.
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
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit_1d(&y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        let y_arr = to_owned_array_1d(y);
        let result = self
            .inner
            .transform_1d(&y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Fit label encoder and return encoded labels.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y_arr = to_owned_array_1d(y);
        let result = slf
            .inner
            .fit_transform_1d(&y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

    /// Transform labels back to original encoding.
    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y_arr = to_owned_array_1d(y);
        let result = self
            .inner
            .inverse_transform_1d(&y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
///     Power transform method: "yeo-johnson" or "box-cox".
/// standardize : bool, optional (default=True)
///     Whether to standardize the transformed output.
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

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
///     Target distribution: "uniform" or "normal".
/// n_quantiles : int, optional (default=1000)
///     Number of quantiles to compute.
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result.into_pyarray(py))
    }

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
///     Maximum degree of polynomial features.
/// interaction_only : bool, optional (default=False)
///     If True, only interaction features (not powers) are produced.
/// include_bias : bool, optional (default=True)
///     If True, include a bias column of ones.
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
///     Number of bins.
/// strategy : str, optional (default="quantile")
///     Strategy: "uniform", "quantile", or "kmeans".
/// encode : str, optional (default="ordinal")
///     Output encoding: "ordinal" or "onehot".
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
///     Features with variance below this value are removed.
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
///     Scoring function: "f_classif", "f_regression", "chi2".
/// k : int, optional (default=10)
///     Number of features to select.
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
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit_with_target(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
///     Number of nearest neighbors to use.
/// weights : str, optional (default="uniform")
///     Weight function: "uniform" or "distance".
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
///     toward the global mean.
/// cv : int, optional (default=5)
///     Number of CV folds for internal cross-validation.
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
        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        slf.inner
            .fit_with_target(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        let x_arr = to_owned_array_2d(x);
        slf.inner
            .fit(&x_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf)
    }

    /// Transform data by selecting features.
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

        smote = apply_sampling_strategy(smote, sampling_strategy)?;

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

        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        let (x_res, y_res) = self
            .inner
            .fit_resample(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
/// random_state : int, optional
///     Random seed.
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

        adasyn = apply_sampling_strategy_adasyn(adasyn, sampling_strategy)?;

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

        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        let (x_res, y_res) = self
            .inner
            .fit_resample(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
///     Strategy: "auto" balances all classes.
/// replacement : bool, optional (default=False)
///     Whether to sample with replacement.
/// random_state : int, optional
///     Random seed.
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

        sampler = apply_sampling_strategy_undersampler(sampler, sampling_strategy)?;

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

        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        let (x_res, y_res) = self
            .inner
            .fit_resample(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
///     Strategy: "auto" balances all classes to majority.
/// shrinkage : float, optional
///     If provided, adds slight noise to duplicated samples using a smoothed
///     bootstrap approach.
/// random_state : int, optional
///     Random seed.
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

        sampler = apply_sampling_strategy_oversampler(sampler, sampling_strategy)?;

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

        let x_arr = to_owned_array_2d(x);
        let y_arr = to_owned_array_1d(y);
        let (x_res, y_res) = self
            .inner
            .fit_resample(&x_arr, &y_arr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok((x_res.into_pyarray(py), y_res.into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        "RandomOverSampler()".to_string()
    }
}

// =============================================================================
// Sampling strategy helpers
// =============================================================================

fn apply_sampling_strategy(
    mut smote: ferroml_core::preprocessing::sampling::SMOTE,
    strategy: &str,
) -> PyResult<ferroml_core::preprocessing::sampling::SMOTE> {
    use ferroml_core::preprocessing::sampling::SamplingStrategy;
    match strategy {
        "auto" => {
            smote = smote.with_sampling_strategy(SamplingStrategy::Auto);
        }
        other => {
            if let Ok(ratio) = other.parse::<f64>() {
                smote = smote.with_sampling_strategy(SamplingStrategy::Ratio(ratio));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown sampling_strategy: '{}'. Use 'auto' or a float ratio.",
                    other
                )));
            }
        }
    }
    Ok(smote)
}

fn apply_sampling_strategy_adasyn(
    mut adasyn: ferroml_core::preprocessing::sampling::ADASYN,
    strategy: &str,
) -> PyResult<ferroml_core::preprocessing::sampling::ADASYN> {
    use ferroml_core::preprocessing::sampling::SamplingStrategy;
    match strategy {
        "auto" => {
            adasyn = adasyn.with_sampling_strategy(SamplingStrategy::Auto);
        }
        other => {
            if let Ok(ratio) = other.parse::<f64>() {
                adasyn = adasyn.with_sampling_strategy(SamplingStrategy::Ratio(ratio));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown sampling_strategy: '{}'. Use 'auto' or a float ratio.",
                    other
                )));
            }
        }
    }
    Ok(adasyn)
}

fn apply_sampling_strategy_undersampler(
    mut sampler: ferroml_core::preprocessing::sampling::RandomUnderSampler,
    strategy: &str,
) -> PyResult<ferroml_core::preprocessing::sampling::RandomUnderSampler> {
    use ferroml_core::preprocessing::sampling::SamplingStrategy;
    match strategy {
        "auto" => {
            sampler = sampler.with_sampling_strategy(SamplingStrategy::Auto);
        }
        other => {
            if let Ok(ratio) = other.parse::<f64>() {
                sampler = sampler.with_sampling_strategy(SamplingStrategy::Ratio(ratio));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown sampling_strategy: '{}'. Use 'auto' or a float ratio.",
                    other
                )));
            }
        }
    }
    Ok(sampler)
}

fn apply_sampling_strategy_oversampler(
    mut sampler: ferroml_core::preprocessing::sampling::RandomOverSampler,
    strategy: &str,
) -> PyResult<ferroml_core::preprocessing::sampling::RandomOverSampler> {
    use ferroml_core::preprocessing::sampling::SamplingStrategy;
    match strategy {
        "auto" => {
            sampler = sampler.with_sampling_strategy(SamplingStrategy::Auto);
        }
        other => {
            if let Ok(ratio) = other.parse::<f64>() {
                sampler = sampler.with_sampling_strategy(SamplingStrategy::Ratio(ratio));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown sampling_strategy: '{}'. Use 'auto' or a float ratio.",
                    other
                )));
            }
        }
    }
    Ok(sampler)
}

// =============================================================================
// Module registration
// =============================================================================

/// Register the preprocessing submodule.
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

    parent_module.add_submodule(&preprocessing_module)?;

    Ok(())
}
