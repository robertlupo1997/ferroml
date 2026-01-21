//! Python pickle support for FerroML models
//!
//! This module provides utilities for Python pickle compatibility via `__getstate__`
//! and `__setstate__` methods. Models are serialized using MessagePack format from
//! the `ferroml_core::serialization` module.
//!
//! ## Usage
//!
//! Python users can use pickle or joblib to save and load FerroML models:
//!
//! ```python
//! import pickle
//! from ferroml.linear import LinearRegression
//!
//! # Train a model
//! model = LinearRegression()
//! model.fit(X_train, y_train)
//!
//! # Save with pickle
//! with open('model.pkl', 'wb') as f:
//!     pickle.dump(model, f)
//!
//! # Load with pickle
//! with open('model.pkl', 'rb') as f:
//!     loaded_model = pickle.load(f)
//!
//! # Also works with joblib (sklearn standard)
//! import joblib
//! joblib.dump(model, 'model.joblib')
//! loaded_model = joblib.load('model.joblib')
//! ```
//!
//! ## Implementation
//!
//! The `__getstate__` method serializes the inner Rust model to MessagePack bytes.
//! The `__setstate__` method deserializes the bytes back to the inner model.
//!
//! MessagePack is chosen because:
//! - Compact binary format (small file size)
//! - Fast serialization/deserialization
//! - Cross-platform compatible

use ferroml_core::serialization::{from_bytes, to_bytes, Format};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{de::DeserializeOwned, Serialize};

/// Serialize a model to Python bytes for pickle `__getstate__`.
///
/// Uses MessagePack format for compact binary serialization.
///
/// # Errors
/// Returns a PyErr if serialization fails.
pub fn getstate<T: Serialize>(py: Python<'_>, model: &T) -> PyResult<Py<PyBytes>> {
    let bytes = to_bytes(model, Format::MessagePack).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to serialize model: {}",
            e
        ))
    })?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// Deserialize a model from Python bytes for pickle `__setstate__`.
///
/// # Errors
/// Returns a PyErr if deserialization fails.
pub fn setstate<T: DeserializeOwned>(bytes: &[u8]) -> PyResult<T> {
    from_bytes(bytes, Format::MessagePack).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to deserialize model: {}",
            e
        ))
    })
}

/// Macro to implement pickle support (`__getstate__` and `__setstate__`) for a PyO3 class.
///
/// This macro generates the required `__getstate__` and `__setstate__` methods for
/// Python pickle compatibility. It requires the inner model type to implement
/// `Serialize` and `DeserializeOwned`.
///
/// # Usage
///
/// ```ignore
/// impl_pickle!(PyLinearRegression, LinearRegression, inner);
/// ```
///
/// Where:
/// - `PyLinearRegression` is the PyO3 wrapper struct
/// - `LinearRegression` is the inner Rust model type
/// - `inner` is the field name containing the inner model
#[macro_export]
macro_rules! impl_pickle {
    ($py_type:ty, $rust_type:ty, $field:ident) => {
        #[pymethods]
        impl $py_type {
            /// Return the state of the model for pickling.
            ///
            /// Returns serialized model bytes using MessagePack format.
            pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyBytes>> {
                $crate::pickle::getstate(py, &self.$field)
            }

            /// Restore the model state from pickled bytes.
            ///
            /// # Arguments
            /// * `state` - Bytes containing the serialized model
            pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
                self.$field = $crate::pickle::setstate(state.as_bytes())?;
                Ok(())
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferroml_core::models::linear::LinearRegression;

    #[test]
    fn test_roundtrip() {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let model = LinearRegression::new().with_fit_intercept(true);

            // Serialize
            let bytes_obj = getstate(py, &model).unwrap();
            let bytes = bytes_obj.bind(py).as_bytes();

            // Deserialize
            let loaded: LinearRegression = setstate(bytes).unwrap();

            // Verify configuration preserved
            assert_eq!(model.fit_intercept, loaded.fit_intercept);
        });
    }
}
