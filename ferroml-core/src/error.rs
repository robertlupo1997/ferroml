//! Error types for FerroML
//!
//! Provides structured error handling with detailed context for debugging.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type alias for FerroML operations
pub type Result<T> = std::result::Result<T, FerroError>;

/// Main error type for FerroML
#[derive(Error, Debug)]
pub enum FerroError {
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Shape mismatch in arrays
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch {
        /// Expected shape description
        expected: String,
        /// Actual shape description
        actual: String,
    },

    /// Statistical assumption violated
    #[error("Statistical assumption violated: {assumption} (test: {test}, p-value: {p_value:.4})")]
    AssumptionViolation {
        /// The violated assumption
        assumption: String,
        /// The statistical test used
        test: String,
        /// The p-value from the test
        p_value: f64,
    },

    /// Numerical computation failed
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Convergence failure in optimization
    #[error("Convergence failed after {iterations} iterations: {reason}")]
    ConvergenceFailure {
        /// Number of iterations attempted
        iterations: usize,
        /// Reason for failure
        reason: String,
    },

    /// Feature not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Feature not implemented for a specific model
    #[error("Feature '{feature}' is not implemented for {model}")]
    NotImplementedFor {
        /// The feature that is not implemented
        feature: String,
        /// The model type
        model: String,
    },

    /// Model not fitted
    #[error("Model not fitted: call fit() before {operation}")]
    NotFitted {
        /// The operation that required a fitted model
        operation: String,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Timeout exceeded
    #[error(
        "Timeout exceeded: {operation} took {elapsed_seconds:.1}s (budget: {budget_seconds}s)"
    )]
    Timeout {
        /// The operation that timed out
        operation: String,
        /// Elapsed time in seconds
        elapsed_seconds: f64,
        /// Time budget in seconds
        budget_seconds: u64,
    },

    /// Resource exhausted
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        /// The exhausted resource
        resource: String,
    },

    /// Cross-validation error
    #[error("Cross-validation error: {0}")]
    CrossValidation(String),

    /// Inference error (ONNX runtime)
    #[error("Inference error: {0}")]
    InferenceError(String),
}

/// Status of an iterative model's convergence.
///
/// Stored by iterative models (KMeans, GMM, LogisticRegression, SVM) after fitting.
/// When a model does not converge, it emits a `tracing::warn!` and returns the
/// best partial result rather than erroring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    /// Model converged within tolerance.
    Converged {
        /// Number of iterations used.
        iterations: usize,
    },
    /// Model did not converge within max_iter.
    NotConverged {
        /// Number of iterations run (== max_iter).
        iterations: usize,
        /// Final change metric (e.g., coefficient delta, inertia change).
        final_change: f64,
    },
}

impl FerroError {
    /// Create an invalid input error
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create an assumption violation error
    pub fn assumption_violation(
        assumption: impl Into<String>,
        test: impl Into<String>,
        p_value: f64,
    ) -> Self {
        Self::AssumptionViolation {
            assumption: assumption.into(),
            test: test.into(),
            p_value,
        }
    }

    /// Create a numerical error
    pub fn numerical(msg: impl Into<String>) -> Self {
        Self::NumericalError(msg.into())
    }

    /// Create a convergence failure error
    pub fn convergence_failure(iterations: usize, reason: impl Into<String>) -> Self {
        Self::ConvergenceFailure {
            iterations,
            reason: reason.into(),
        }
    }

    /// Create a not fitted error
    pub fn not_fitted(operation: impl Into<String>) -> Self {
        Self::NotFitted {
            operation: operation.into(),
        }
    }

    /// Create a not implemented for specific model error
    pub fn not_implemented_for(feature: impl Into<String>, model: impl Into<String>) -> Self {
        Self::NotImplementedFor {
            feature: feature.into(),
            model: model.into(),
        }
    }

    /// Create a cross-validation error
    pub fn cross_validation(msg: impl Into<String>) -> Self {
        Self::CrossValidation(msg.into())
    }

    /// Returns a remediation hint for this error, if available.
    ///
    /// Hints are static strings (no allocation) that provide actionable guidance
    /// for resolving the error. Context-sensitive hints are returned for
    /// `InvalidInput` and `NumericalError` based on the error message content.
    pub fn hint(&self) -> &str {
        match self {
            FerroError::ShapeMismatch { .. } => {
                "Hint: Ensure X.shape[0] == y.shape[0]. Check for missing values or filtering steps that may have changed array lengths."
            }
            FerroError::NotFitted { .. } => {
                "Hint: Call model.fit(X, y) before calling predict, transform, or other methods that require a fitted model."
            }
            FerroError::InvalidInput(msg) => {
                if msg.contains("negative") || msg.contains("positive") {
                    "Hint: Check parameter bounds. Many parameters require positive values (e.g., n_clusters, C, alpha)."
                } else if msg.contains("NaN") || msg.contains("nan") {
                    "Hint: Check input data for NaN values. Use np.nan_to_num() or imputation to handle missing data."
                } else if msg.contains("empty") {
                    "Hint: Ensure input arrays are non-empty. Check data loading and filtering steps."
                } else {
                    "Hint: Check the parameter documentation for valid ranges and types."
                }
            }
            FerroError::ConvergenceFailure { .. } => {
                "Hint: Try increasing max_iter, adjusting the learning rate, or scaling features with StandardScaler."
            }
            FerroError::NumericalError(msg) => {
                if msg.contains("singular") || msg.contains("Singular") {
                    "Hint: The matrix is singular (not invertible). Try adding regularization (use Ridge instead of LinearRegression) or removing collinear features."
                } else if msg.contains("NaN") || msg.contains("overflow") {
                    "Hint: Numerical instability detected. Try scaling features with StandardScaler or reducing the learning rate."
                } else {
                    "Hint: Check for extreme values in input data. Feature scaling or regularization may help."
                }
            }
            FerroError::ConfigError(_) => {
                "Hint: Check model documentation for valid parameter combinations and ranges."
            }
            FerroError::AssumptionViolation { .. } => {
                "Hint: Consider using a model that does not assume this property, or transform your data to meet the assumption."
            }
            FerroError::SerializationError(_) => {
                "Hint: Ensure the model was saved with a compatible version of FerroML. Some models (Pipeline, Voting, Stacking) do not yet support serialization."
            }
            FerroError::Timeout { .. } => {
                "Hint: Reduce dataset size, decrease max_iter, or increase the time budget."
            }
            FerroError::ResourceExhausted { .. } => {
                "Hint: Reduce dataset size or model complexity. Consider using incremental learning (partial_fit) for large datasets."
            }
            FerroError::CrossValidation(_) => {
                "Hint: Check that the number of CV folds does not exceed the number of samples per class."
            }
            FerroError::InferenceError(_) => {
                "Hint: Verify input features match the training data format (same number of features, same encoding)."
            }
            // NotImplemented, NotImplementedFor, IoError don't need hints
            _ => "",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err =
            FerroError::assumption_violation("Normality of residuals", "Shapiro-Wilk", 0.0023);
        let msg = format!("{}", err);
        assert!(msg.contains("Normality"));
        assert!(msg.contains("Shapiro-Wilk"));
        assert!(msg.contains("0.0023"));
    }

    #[test]
    fn test_shape_mismatch() {
        let err = FerroError::shape_mismatch("(100, 10)", "(100, 5)");
        let msg = format!("{}", err);
        assert!(msg.contains("(100, 10)"));
        assert!(msg.contains("(100, 5)"));
    }

    #[test]
    fn test_shape_mismatch_hint() {
        let err = FerroError::ShapeMismatch {
            expected: "100 samples".to_string(),
            actual: "50 samples".to_string(),
        };
        assert!(!err.hint().is_empty());
        assert!(err.hint().starts_with("Hint:"));
        assert!(err.hint().contains("shape"));
    }

    #[test]
    fn test_not_fitted_hint() {
        let err = FerroError::not_fitted("predict");
        assert!(err.hint().starts_with("Hint:"));
        assert!(err.hint().contains("fit"));
    }

    #[test]
    fn test_convergence_failure_hint() {
        let err = FerroError::convergence_failure(100, "loss did not decrease");
        assert!(err.hint().starts_with("Hint:"));
        assert!(err.hint().contains("max_iter"));
    }

    #[test]
    fn test_invalid_input_hint_negative() {
        let err = FerroError::invalid_input("alpha must be positive");
        assert!(err.hint().contains("positive"));
    }

    #[test]
    fn test_invalid_input_hint_nan() {
        let err = FerroError::invalid_input("Input contains NaN");
        assert!(err.hint().contains("NaN"));
    }

    #[test]
    fn test_invalid_input_hint_empty() {
        let err = FerroError::invalid_input("Input array is empty");
        assert!(err.hint().contains("non-empty"));
    }

    #[test]
    fn test_invalid_input_hint_generic() {
        let err = FerroError::invalid_input("bad parameter");
        assert!(err.hint().starts_with("Hint:"));
    }

    #[test]
    fn test_numerical_error_hint_singular() {
        let err = FerroError::numerical("singular matrix encountered");
        assert!(err.hint().contains("singular"));
        assert!(err.hint().contains("Ridge"));
    }

    #[test]
    fn test_numerical_error_hint_nan() {
        let err = FerroError::numerical("NaN in gradient computation");
        assert!(err.hint().contains("scaling"));
    }

    #[test]
    fn test_numerical_error_hint_generic() {
        let err = FerroError::numerical("computation failed");
        assert!(err.hint().starts_with("Hint:"));
    }

    #[test]
    fn test_no_hint_for_not_implemented() {
        let err = FerroError::NotImplemented("some feature".to_string());
        assert!(err.hint().is_empty());
    }

    #[test]
    fn test_no_hint_for_not_implemented_for() {
        let err = FerroError::not_implemented_for("feature", "model");
        assert!(err.hint().is_empty());
    }

    #[test]
    fn test_all_hinted_variants_start_with_hint() {
        let errors: Vec<FerroError> = vec![
            FerroError::shape_mismatch("a", "b"),
            FerroError::not_fitted("predict"),
            FerroError::invalid_input("test"),
            FerroError::convergence_failure(10, "reason"),
            FerroError::numerical("error"),
            FerroError::ConfigError("bad config".to_string()),
            FerroError::assumption_violation("norm", "test", 0.01),
            FerroError::SerializationError("err".to_string()),
            FerroError::Timeout {
                operation: "fit".to_string(),
                elapsed_seconds: 10.0,
                budget_seconds: 5,
            },
            FerroError::ResourceExhausted {
                resource: "memory".to_string(),
            },
            FerroError::cross_validation("bad folds"),
            FerroError::InferenceError("err".to_string()),
        ];
        for err in &errors {
            let hint = err.hint();
            assert!(
                hint.starts_with("Hint:"),
                "Expected hint for {:?}, got: '{}'",
                err,
                hint
            );
        }
    }
}
