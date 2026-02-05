//! Comprehensive estimator validation framework
//!
//! This module provides sklearn-compatible estimator checks for `FerroML` models.
//! It ensures all models conform to the expected API and behave correctly.
//!
//! # Usage
//!
//! ```ignore
//! use ferroml_core::testing::check_estimator;
//! use ferroml_core::models::LinearRegression;
//!
//! let results = check_estimator(LinearRegression::new());
//! for result in &results {
//!     assert!(result.passed, "{}: {}", result.name, result.message.as_deref().unwrap_or(""));
//! }
//! ```
//!
//! # Assertions
//!
//! For writing tests with proper floating-point tolerance:
//!
//! ```ignore
//! use ferroml_core::testing::assertions::{tolerances, assert_approx_eq, assert_array_approx_eq};
//!
//! // Use calibrated tolerances for different algorithm types
//! assert_approx_eq!(actual, expected, tolerances::CLOSED_FORM);
//! assert_array_approx_eq!(predictions, expected, tolerances::ITERATIVE);
//! ```
//!
//! # Serialization Testing
//!
//! For testing model and transformer serialization across multiple formats:
//!
//! ```ignore
//! use ferroml_core::testing::serialization::{check_model_serialization, SerializationFormat};
//! use ferroml_core::models::LinearRegression;
//!
//! let results = check_model_serialization(
//!     LinearRegression::new(),
//!     SerializationTestConfig::default(),
//! );
//! for result in &results {
//!     assert!(result.passed, "{}: {}", result.name, result.message.as_deref().unwrap_or(""));
//! }
//! ```

pub mod assertions;
pub mod automl;
pub mod callbacks;
pub mod categorical;
pub mod checks;
pub mod cv_advanced;
pub mod drift;
pub mod ensemble_advanced;
pub mod explainability;
pub mod fairness;
pub mod hpo;
pub mod incremental;
pub mod metrics;
pub mod nan_inf_validation;
pub mod onnx;
pub mod probabilistic;
pub mod properties;
pub mod regression;
pub mod serialization;
#[cfg(feature = "sparse")]
pub mod sparse_tests;
pub mod transformer;
pub mod utils;
pub mod weights;

pub use assertions::*;
pub use checks::*;
pub use nan_inf_validation::*;
pub use probabilistic::*;
pub use utils::*;

use crate::models::Model;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result of a single estimator check
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Name of the check
    pub name: &'static str,
    /// Whether the check passed
    pub passed: bool,
    /// Optional message (especially for failures)
    pub message: Option<String>,
    /// How long the check took
    pub duration: Duration,
    /// Check category for filtering
    pub category: CheckCategory,
}

impl CheckResult {
    /// Create a passing check result
    #[must_use]
    pub fn pass(name: &'static str, category: CheckCategory) -> Self {
        Self {
            name,
            passed: true,
            message: None,
            duration: Duration::ZERO,
            category,
        }
    }

    /// Create a failing check result
    pub fn fail(name: &'static str, category: CheckCategory, message: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            message: Some(message.into()),
            duration: Duration::ZERO,
            category,
        }
    }
}

/// Categories of checks for filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckCategory {
    /// Basic API conformance
    Api,
    /// Input validation
    InputValidation,
    /// Numerical correctness
    Numerical,
    /// Performance/scalability
    Performance,
    /// Thread safety
    Concurrency,
    /// Serialization
    Serialization,
}

/// Configuration for `check_estimator`
#[derive(Debug, Clone)]
pub struct CheckConfig {
    /// Skip checks in these categories
    pub skip_categories: Vec<CheckCategory>,
    /// Skip checks by name
    pub skip_checks: Vec<&'static str>,
    /// Timeout per check
    pub timeout: Duration,
    /// Generate regression data with this many samples
    pub n_samples_regression: usize,
    /// Generate classification data with this many samples
    pub n_samples_classification: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for CheckConfig {
    fn default() -> Self {
        Self {
            skip_categories: vec![],
            skip_checks: vec![],
            timeout: Duration::from_secs(60),
            n_samples_regression: 100,
            n_samples_classification: 100,
            seed: 42,
        }
    }
}

/// Run all standard checks on a model
pub fn check_estimator<M: Model + Clone + Send + Sync + 'static>(model: M) -> Vec<CheckResult> {
    check_estimator_with_config(model, CheckConfig::default())
}

/// Run checks with custom configuration
pub fn check_estimator_with_config<M: Model + Clone + Send + Sync + 'static>(
    model: M,
    config: CheckConfig,
) -> Vec<CheckResult> {
    let mut results = Vec::new();

    // Generate test data
    let (x_reg, y_reg) = utils::make_regression(config.n_samples_regression, 5, 0.1, config.seed);
    let (_x_clf, _y_clf) =
        utils::make_binary_classification(config.n_samples_classification, 5, config.seed);

    // Clone references for closures
    let x_reg_ref = &x_reg;
    let y_reg_ref = &y_reg;

    // Define all checks as (name, category, check_fn)
    let checks: Vec<(&str, CheckCategory, Box<dyn Fn(&M) -> CheckResult + '_>)> = vec![
        (
            "check_not_fitted",
            CheckCategory::Api,
            Box::new(|m: &M| checks::check_not_fitted(m)),
        ),
        (
            "check_n_features_in",
            CheckCategory::Api,
            Box::new(move |m: &M| checks::check_n_features_in(m.clone(), x_reg_ref, y_reg_ref)),
        ),
        (
            "check_nan_handling",
            CheckCategory::InputValidation,
            Box::new(|m: &M| checks::check_nan_handling(m.clone())),
        ),
        (
            "check_inf_handling",
            CheckCategory::InputValidation,
            Box::new(|m: &M| checks::check_inf_handling(m.clone())),
        ),
        (
            "check_empty_data",
            CheckCategory::InputValidation,
            Box::new(|m: &M| checks::check_empty_data(m.clone())),
        ),
        (
            "check_fit_idempotent",
            CheckCategory::Numerical,
            Box::new(move |m: &M| checks::check_fit_idempotent(m.clone(), x_reg_ref, y_reg_ref)),
        ),
        (
            "check_single_sample",
            CheckCategory::InputValidation,
            Box::new(|m: &M| checks::check_single_sample(m.clone())),
        ),
        (
            "check_single_feature",
            CheckCategory::InputValidation,
            Box::new(|m: &M| checks::check_single_feature(m.clone())),
        ),
        (
            "check_shape_mismatch",
            CheckCategory::InputValidation,
            Box::new(|m: &M| checks::check_shape_mismatch(m.clone())),
        ),
        (
            "check_subset_invariance",
            CheckCategory::Numerical,
            Box::new(move |m: &M| checks::check_subset_invariance(m.clone(), x_reg_ref, y_reg_ref)),
        ),
        (
            "check_clone_equivalence",
            CheckCategory::Api,
            Box::new(move |m: &M| checks::check_clone_equivalence(m.clone(), x_reg_ref, y_reg_ref)),
        ),
        (
            "check_fit_does_not_modify_input",
            CheckCategory::Api,
            Box::new(move |m: &M| {
                checks::check_fit_does_not_modify_input(m.clone(), x_reg_ref, y_reg_ref)
            }),
        ),
        (
            "check_predict_shape",
            CheckCategory::Api,
            Box::new(move |m: &M| checks::check_predict_shape(m.clone(), x_reg_ref, y_reg_ref)),
        ),
        (
            "check_no_side_effects",
            CheckCategory::Numerical,
            Box::new(move |m: &M| checks::check_no_side_effects(m.clone(), x_reg_ref, y_reg_ref)),
        ),
        (
            "check_multithread_safe",
            CheckCategory::Concurrency,
            Box::new(move |m: &M| checks::check_multithread_safe(m.clone(), x_reg_ref, y_reg_ref)),
        ),
        (
            "check_large_input",
            CheckCategory::Performance,
            Box::new(move |m: &M| checks::check_large_input(m.clone(), config.seed)),
        ),
        (
            "check_negative_values",
            CheckCategory::Numerical,
            Box::new(|m: &M| checks::check_negative_values(m.clone())),
        ),
        (
            "check_very_small_values",
            CheckCategory::Numerical,
            Box::new(|m: &M| checks::check_very_small_values(m.clone())),
        ),
        (
            "check_very_large_values",
            CheckCategory::Numerical,
            Box::new(|m: &M| checks::check_very_large_values(m.clone())),
        ),
        (
            "check_mixed_scale_features",
            CheckCategory::Numerical,
            Box::new(|m: &M| checks::check_mixed_scale_features(m.clone())),
        ),
        (
            "check_constant_feature",
            CheckCategory::Numerical,
            Box::new(|m: &M| checks::check_constant_feature(m.clone())),
        ),
        (
            "check_fit_twice_different_data",
            CheckCategory::Api,
            Box::new(|m: &M| checks::check_fit_twice_different_data(m.clone())),
        ),
        (
            "check_dtype_consistency",
            CheckCategory::Api,
            Box::new(move |m: &M| checks::check_dtype_consistency(m.clone(), x_reg_ref, y_reg_ref)),
        ),
        (
            "check_zero_samples_predict",
            CheckCategory::InputValidation,
            Box::new(move |m: &M| {
                checks::check_zero_samples_predict(m.clone(), x_reg_ref, y_reg_ref)
            }),
        ),
        (
            "check_deterministic_predictions",
            CheckCategory::Numerical,
            Box::new(move |m: &M| {
                checks::check_deterministic_predictions(m.clone(), x_reg_ref, y_reg_ref)
            }),
        ),
    ];

    // Run checks with filtering
    for (name, category, check_fn) in checks {
        if config.skip_categories.contains(&category) {
            continue;
        }
        if config.skip_checks.contains(&name) {
            continue;
        }

        let start = Instant::now();
        let mut result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check_fn(&model)))
                .unwrap_or_else(|e| {
                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                        format!("Check panicked: {s}")
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        format!("Check panicked: {s}")
                    } else {
                        "Check panicked!".into()
                    };
                    CheckResult {
                        name,
                        passed: false,
                        message: Some(msg),
                        duration: Duration::ZERO,
                        category,
                    }
                });

        result.duration = start.elapsed();
        result.category = category;
        results.push(result);
    }

    results
}

/// Run checks and assert all pass (convenience for tests)
pub fn assert_estimator_valid<M: Model + Clone + Send + Sync + 'static>(model: M) {
    let results = check_estimator(model);
    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();

    if !failures.is_empty() {
        let msg = failures
            .iter()
            .map(|r| {
                format!(
                    "  - {}: {}",
                    r.name,
                    r.message.as_deref().unwrap_or("failed")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        panic!("Estimator failed {} checks:\n{}", failures.len(), msg);
    }
}

/// Summary statistics for check results
#[must_use]
pub fn summarize_results(results: &[CheckResult]) -> CheckSummary {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;
    let total_duration: Duration = results.iter().map(|r| r.duration).sum();

    let mut by_category: HashMap<CheckCategory, (usize, usize)> = HashMap::new();
    for result in results {
        let entry = by_category.entry(result.category).or_insert((0, 0));
        if result.passed {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    CheckSummary {
        total,
        passed,
        failed,
        total_duration,
        by_category,
    }
}

/// Summary of check results
#[derive(Debug)]
pub struct CheckSummary {
    /// Total number of checks run
    pub total: usize,
    /// Number of passed checks
    pub passed: usize,
    /// Number of failed checks
    pub failed: usize,
    /// Total time to run all checks
    pub total_duration: Duration,
    /// (passed, failed) counts by category
    pub by_category: HashMap<CheckCategory, (usize, usize)>,
}

impl std::fmt::Display for CheckSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Check Results: {}/{} passed ({:.1}%)",
            self.passed,
            self.total,
            if self.total > 0 {
                100.0 * self.passed as f64 / self.total as f64
            } else {
                0.0
            }
        )?;
        writeln!(f, "Total time: {:?}", self.total_duration)?;
        writeln!(f, "\nBy category:")?;
        for (cat, (p, fail)) in &self.by_category {
            writeln!(f, "  {:?}: {}/{} passed", cat, p, p + fail)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_result_pass() {
        let result = CheckResult::pass("test_check", CheckCategory::Api);
        assert!(result.passed);
        assert!(result.message.is_none());
    }

    #[test]
    fn test_check_result_fail() {
        let result = CheckResult::fail("test_check", CheckCategory::Api, "something went wrong");
        assert!(!result.passed);
        assert_eq!(result.message.as_deref(), Some("something went wrong"));
    }

    #[test]
    fn test_check_config_default() {
        let config = CheckConfig::default();
        assert!(config.skip_categories.is_empty());
        assert!(config.skip_checks.is_empty());
        assert_eq!(config.n_samples_regression, 100);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_summarize_results() {
        let results = vec![
            CheckResult::pass("check1", CheckCategory::Api),
            CheckResult::pass("check2", CheckCategory::Api),
            CheckResult::fail("check3", CheckCategory::Numerical, "failed"),
        ];

        let summary = summarize_results(&results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 1);
    }
}
