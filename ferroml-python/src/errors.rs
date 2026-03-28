//! Shared error conversion helpers for PyO3 bindings.
//!
//! These helpers reduce boilerplate when converting `FerroError` (and other
//! `Display`-implementing errors) into `PyErr` for return to Python.
//!
//! ## Error Mapping
//!
//! FerroError variants are mapped to semantically correct Python exceptions:
//!
//! | FerroError variant       | Python exception       |
//! |--------------------------|------------------------|
//! | InvalidInput             | ValueError             |
//! | ShapeMismatch            | ValueError             |
//! | ConfigError              | ValueError             |
//! | NotFitted                | RuntimeError           |
//! | NumericalError           | RuntimeError           |
//! | ConvergenceFailure       | RuntimeError           |
//! | AssumptionViolation      | RuntimeError           |
//! | NotImplemented           | NotImplementedError    |
//! | NotImplementedFor        | NotImplementedError    |
//! | SerializationError       | RuntimeError           |
//! | CrossValidation          | RuntimeError           |
//! | InferenceError           | RuntimeError           |
//! | IoError                  | OSError                |
//! | Timeout                  | TimeoutError           |
//! | ResourceExhausted        | RuntimeError           |

use ferroml_core::FerroError;
use pyo3::prelude::*;

/// Convert a `FerroError` into a semantically correct Python exception.
///
/// This maps each FerroError variant to the most appropriate Python
/// exception type, enabling users to catch specific exception types:
///
/// ```python
/// try:
///     model.predict(X_wrong_shape)
/// except ValueError:
///     print("Bad input!")
/// ```
#[inline]
pub fn ferro_to_pyerr(e: FerroError) -> PyErr {
    let hint = e.hint();
    let message = if hint.is_empty() {
        e.to_string()
    } else {
        format!("{}\n{}", e, hint)
    };

    match &e {
        FerroError::InvalidInput(_)
        | FerroError::ShapeMismatch { .. }
        | FerroError::ConfigError(_) => PyErr::new::<pyo3::exceptions::PyValueError, _>(message),
        FerroError::NotFitted { .. } => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(message),
        FerroError::NumericalError(_)
        | FerroError::ConvergenceFailure { .. }
        | FerroError::AssumptionViolation { .. }
        | FerroError::SerializationError(_)
        | FerroError::CrossValidation(_)
        | FerroError::InferenceError(_)
        | FerroError::ResourceExhausted { .. } => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(message)
        }
        FerroError::NotImplemented(_) | FerroError::NotImplementedFor { .. } => {
            PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(message)
        }
        FerroError::IoError(_) => PyErr::new::<pyo3::exceptions::PyOSError, _>(message),
        FerroError::Timeout { .. } => PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(message),
    }
}

/// Convert any `Display`-implementing error into a `PyRuntimeError`.
///
/// **Deprecated:** Prefer `ferro_to_pyerr` for `FerroError` values, which maps
/// to semantically correct Python exceptions. This function is retained for
/// non-FerroError error types (e.g., serde errors, string errors).
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
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
        "{} not fitted. Call fit() first.",
        entity
    ))
}
