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
}
