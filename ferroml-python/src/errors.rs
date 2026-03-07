//! Shared error conversion helpers for PyO3 bindings.
//!
//! These helpers reduce boilerplate when converting `FerroError` (and other
//! `Display`-implementing errors) into `PyErr` for return to Python.
//!
//! ## Background
//!
//! The canonical solution would be `impl From<FerroError> for PyErr`, but that
//! requires either adding a PyO3 dependency to `ferroml-core` or using a
//! newtype wrapper. Both introduce undesirable coupling. Instead, we provide
//! two small helpers that replace the most common error-conversion patterns:
//!
//! - `to_py_runtime_err` -- replaces 237 occurrences of
//!   `.map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))`
//! - `not_fitted_err` -- replaces ~100 occurrences of
//!   `PyErr::new::<PyValueError, _>("... not fitted. Call fit() first.")`

use pyo3::prelude::*;

/// Convert any `Display`-implementing error into a `PyRuntimeError`.
///
/// Use with `.map_err(to_py_runtime_err)` on `Result` chains. This replaces
/// the repeated pattern:
///
/// ```ignore
/// .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
/// ```
#[inline]
pub fn to_py_runtime_err(e: impl std::fmt::Display) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
}

/// Create a "not fitted" `PyValueError` with a consistent message.
///
/// The `entity` parameter is the name of the model or transformer, e.g.
/// `"Model"`, `"Scaler"`, `"Encoder"`.
///
/// ```ignore
/// self.inner.mean().ok_or_else(|| not_fitted_err("Scaler"))?;
/// ```
#[inline]
pub fn not_fitted_err(entity: &str) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "{} not fitted. Call fit() first.",
        entity
    ))
}
