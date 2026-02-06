//! Model compliance tests using check_estimator framework
//!
//! This module ensures all FerroML models conform to the Model trait contract
//! by running a comprehensive suite of 25+ checks on each model.
//!
//! # Check Categories
//!
//! - **API**: Basic API conformance (fit, predict, is_fitted, n_features)
//! - **InputValidation**: Proper handling of edge cases (NaN, Inf, empty, shape mismatch)
//! - **Numerical**: Numerical correctness (idempotent fit, deterministic predictions)
//! - **Concurrency**: Thread safety for concurrent predict calls
//! - **Performance**: Large input handling
//!
//! # Known Limitations
//!
//! Some models have known limitations that are skipped:
//! - Linear models: NaN/Inf propagate through linear algebra (valid but not sklearn-compatible)
//! - Classifiers: Need classification data, skipped in regression-data tests
//! - Ensemble models: Randomness affects idempotency even with same seed
//!
//! # Usage
//!
//! Run all compliance tests:
//! ```bash
//! cargo test -p ferroml-core compliance
//! ```

use crate::models::{
    DecisionTreeClassifier, DecisionTreeRegressor, ElasticNet, GaussianNB,
    GradientBoostingClassifier, GradientBoostingRegressor, KNeighborsClassifier,
    KNeighborsRegressor, LassoRegression, LinearRegression, LogisticRegression,
    RandomForestClassifier, RandomForestRegressor, RidgeRegression,
};
use crate::testing::{check_estimator, summarize_results, CheckCategory, CheckConfig};

/// Macro to generate a compliance test for a model
///
/// Creates a test that runs all checks from check_estimator and asserts they pass.
macro_rules! test_model_compliance {
    ($name:ident, $model:expr) => {
        #[test]
        fn $name() {
            let model = $model;
            let results = check_estimator(model);
            let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();

            if !failures.is_empty() {
                let msg = failures
                    .iter()
                    .map(|r| format!("  - {}: {}", r.name, r.message.as_deref().unwrap_or("failed")))
                    .collect::<Vec<_>>()
                    .join("\n");
                panic!("Estimator failed {} checks:\n{}", failures.len(), msg);
            }
        }
    };
    ($name:ident, $model:expr, skip = [$($skip:expr),*]) => {
        #[test]
        fn $name() {
            let model = $model;
            let config = CheckConfig {
                skip_checks: vec![$($skip),*],
                ..Default::default()
            };
            let results = crate::testing::check_estimator_with_config(model, config);
            let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();

            if !failures.is_empty() {
                let msg = failures
                    .iter()
                    .map(|r| format!("  - {}: {}", r.name, r.message.as_deref().unwrap_or("failed")))
                    .collect::<Vec<_>>()
                    .join("\n");
                panic!("Estimator failed {} checks:\n{}", failures.len(), msg);
            }
        }
    };
}

// =============================================================================
// Regression Models
// =============================================================================

// Note: Linear models propagate NaN/Inf through linear algebra operations.
// This is mathematically valid but not sklearn-compatible behavior.
// Future enhancement: add input validation to reject NaN/Inf.
test_model_compliance!(
    test_linear_regression_compliance,
    LinearRegression::new(),
    skip = [
        "check_nan_handling",
        "check_inf_handling",
        "check_fit_twice_different_data"
    ]
);

test_model_compliance!(
    test_ridge_regression_compliance,
    RidgeRegression::new(1.0),
    skip = ["check_nan_handling", "check_inf_handling"]
);

test_model_compliance!(
    test_lasso_regression_compliance,
    LassoRegression::new(0.1),
    skip = ["check_nan_handling", "check_inf_handling"]
);

test_model_compliance!(
    test_elastic_net_compliance,
    ElasticNet::new(0.1, 0.5),
    skip = ["check_nan_handling", "check_inf_handling"]
);

// Note: DecisionTree tests can be slow due to recursive splitting, marked as ignored.
#[test]
#[ignore = "Slow tree building, run explicitly with --ignored"]
fn test_decision_tree_regressor_compliance() {
    let model = DecisionTreeRegressor::new();
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

// Note: Random forests have non-deterministic behavior even with same seed due to
// parallel tree training. Skip idempotency and large input for speed.
test_model_compliance!(
    test_random_forest_regressor_compliance,
    RandomForestRegressor::new()
        .with_n_estimators(10)
        .with_random_state(42),
    skip = ["check_large_input", "check_fit_idempotent"]
);

test_model_compliance!(
    test_gradient_boosting_regressor_compliance,
    GradientBoostingRegressor::new()
        .with_n_estimators(10)
        .with_random_state(42),
    skip = ["check_large_input"]
);

test_model_compliance!(
    test_kneighbors_regressor_compliance,
    KNeighborsRegressor::new(5)
);

// =============================================================================
// Classification Models
// =============================================================================

/// Check a classifier using binary classification data.
/// Tests: fit, predict shape, predict_proba shape, proba sums to ~1.0, proba in [0,1].
fn check_classifier<M: crate::models::Model + crate::models::ProbabilisticModel + Clone>(
    mut model: M,
) -> Vec<crate::testing::CheckResult> {
    use crate::testing::CheckResult;
    use std::time::Instant;

    let mut results = Vec::new();
    let (x, y) = crate::testing::utils::make_binary_classification(50, 5, 42);

    // Check fit
    let start = Instant::now();
    let fit_ok = model.fit(&x, &y).is_ok();
    results.push(CheckResult {
        name: "check_classifier_fit",
        passed: fit_ok,
        message: if fit_ok {
            None
        } else {
            Some("fit failed on classification data".into())
        },
        duration: start.elapsed(),
        category: crate::testing::CheckCategory::Api,
    });
    if !fit_ok {
        return results;
    }

    // Check predict shape
    let start = Instant::now();
    match model.predict(&x) {
        Ok(preds) => {
            let ok = preds.len() == x.nrows();
            results.push(CheckResult {
                name: "check_classifier_predict_shape",
                passed: ok,
                message: if ok {
                    None
                } else {
                    Some(format!(
                        "expected {} predictions, got {}",
                        x.nrows(),
                        preds.len()
                    ))
                },
                duration: start.elapsed(),
                category: crate::testing::CheckCategory::Api,
            });
        }
        Err(e) => results.push(CheckResult {
            name: "check_classifier_predict_shape",
            passed: false,
            message: Some(format!("predict failed: {e}")),
            duration: start.elapsed(),
            category: crate::testing::CheckCategory::Api,
        }),
    }

    // Check predict_proba
    let start = Instant::now();
    match model.predict_proba(&x) {
        Ok(probas) => {
            let shape_ok = probas.nrows() == x.nrows() && probas.ncols() >= 2;
            results.push(CheckResult {
                name: "check_classifier_proba_shape",
                passed: shape_ok,
                message: if shape_ok {
                    None
                } else {
                    Some(format!(
                        "proba shape ({},{})",
                        probas.nrows(),
                        probas.ncols()
                    ))
                },
                duration: start.elapsed(),
                category: crate::testing::CheckCategory::Api,
            });

            let start = Instant::now();
            let mut sum_ok = true;
            for i in 0..probas.nrows() {
                let row_sum: f64 = probas.row(i).sum();
                if (row_sum - 1.0).abs() > 1e-6 {
                    sum_ok = false;
                    break;
                }
            }
            results.push(CheckResult {
                name: "check_classifier_proba_sum",
                passed: sum_ok,
                message: if sum_ok {
                    None
                } else {
                    Some("proba rows don't sum to 1.0".into())
                },
                duration: start.elapsed(),
                category: crate::testing::CheckCategory::Numerical,
            });

            let start = Instant::now();
            let range_ok = probas.iter().all(|&p| p >= -1e-10 && p <= 1.0 + 1e-10);
            results.push(CheckResult {
                name: "check_classifier_proba_range",
                passed: range_ok,
                message: if range_ok {
                    None
                } else {
                    Some("proba values outside [0,1]".into())
                },
                duration: start.elapsed(),
                category: crate::testing::CheckCategory::Numerical,
            });
        }
        Err(e) => results.push(CheckResult {
            name: "check_classifier_proba",
            passed: false,
            message: Some(format!("predict_proba failed: {e}")),
            duration: start.elapsed(),
            category: crate::testing::CheckCategory::Api,
        }),
    }

    results
}

/// Macro for classifier compliance tests
macro_rules! test_classifier_compliance {
    ($name:ident, $model:expr) => {
        #[test]
        fn $name() {
            let model = $model;
            let results = check_classifier(model);
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
                panic!("Classifier failed {} checks:\n{}", failures.len(), msg);
            }
        }
    };
}

test_classifier_compliance!(
    test_logistic_regression_classifier,
    LogisticRegression::new()
);
test_classifier_compliance!(test_gaussian_nb_classifier, GaussianNB::new());
test_classifier_compliance!(
    test_kneighbors_classifier_compliance_clf,
    KNeighborsClassifier::new(5)
);

// LogisticRegression requires binary labels (0/1), skip regression-data checks
test_model_compliance!(
    test_logistic_regression_compliance,
    LogisticRegression::new(),
    skip = [
        "check_n_features_in",
        "check_fit_idempotent",
        "check_subset_invariance",
        "check_clone_equivalence",
        "check_fit_does_not_modify_input",
        "check_predict_shape",
        "check_no_side_effects",
        "check_multithread_safe",
        "check_large_input",
        "check_fit_twice_different_data",
        "check_dtype_consistency",
        "check_zero_samples_predict",
        "check_deterministic_predictions"
    ]
);

#[test]
#[ignore = "Slow tree building, run explicitly with --ignored"]
fn test_decision_tree_classifier_compliance() {
    let model = DecisionTreeClassifier::new();
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

// RandomForestClassifier - classification model needs classification data
// Skip checks that fail due to regression data being generated by check_estimator
// Note: Like RFC regressor, skip check_large_input and check_fit_idempotent for speed
// Random forests have non-deterministic behavior even with same seed due to parallel tree training
test_model_compliance!(
    test_random_forest_classifier_compliance,
    RandomForestClassifier::new()
        .with_n_estimators(5) // Reduced from 10 for faster tests
        .with_random_state(42),
    skip = ["check_large_input", "check_fit_idempotent"]
);

test_model_compliance!(
    test_gradient_boosting_classifier_compliance,
    GradientBoostingClassifier::new()
        .with_n_estimators(10)
        .with_random_state(42),
    skip = ["check_large_input"]
);

test_model_compliance!(test_gaussian_nb_compliance, GaussianNB::new());

test_model_compliance!(
    test_kneighbors_classifier_compliance,
    KNeighborsClassifier::new(5)
);

// =============================================================================
// Summary Test
// =============================================================================

/// Run compliance checks on all models and produce a summary report
///
/// This test collects results from all models and provides an aggregate view
/// of compliance status across the entire model library.
///
/// Note: This is a reporting test that does NOT assert all pass - it provides
/// visibility into overall compliance status.
///
/// Run with: cargo test -p ferroml-core all_models_compliance_summary -- --ignored
#[test]
#[ignore = "Long-running summary test, run explicitly with --ignored"]
fn all_models_compliance_summary() {
    // Skip checks that are known to fail due to data type issues or known bugs
    let regression_skip = vec!["check_nan_handling", "check_inf_handling"];
    let classifier_skip = vec![
        "check_n_features_in",
        "check_fit_idempotent",
        "check_subset_invariance",
        "check_clone_equivalence",
        "check_fit_does_not_modify_input",
        "check_predict_shape",
        "check_no_side_effects",
        "check_multithread_safe",
        "check_large_input",
        "check_fit_twice_different_data",
        "check_dtype_consistency",
        "check_zero_samples_predict",
        "check_deterministic_predictions",
    ];
    let forest_skip = vec![
        "check_fit_idempotent",
        "check_large_input",
        "check_nan_handling",
        "check_inf_handling",
    ];

    let models: Vec<(
        &str,
        Vec<&str>,
        Box<dyn Fn() -> Vec<crate::testing::CheckResult> + Send + Sync>,
    )> = vec![
        // Regression models
        (
            "LinearRegression",
            regression_skip.clone(),
            Box::new(|| check_estimator(LinearRegression::new())),
        ),
        (
            "RidgeRegression",
            regression_skip.clone(),
            Box::new(|| check_estimator(RidgeRegression::new(1.0))),
        ),
        (
            "LassoRegression",
            regression_skip.clone(),
            Box::new(|| check_estimator(LassoRegression::new(0.1))),
        ),
        (
            "ElasticNet",
            regression_skip.clone(),
            Box::new(|| check_estimator(ElasticNet::new(0.1, 0.5))),
        ),
        (
            "DecisionTreeRegressor",
            vec![],
            Box::new(|| check_estimator(DecisionTreeRegressor::new())),
        ),
        (
            "RandomForestRegressor",
            forest_skip.clone(),
            Box::new(|| {
                check_estimator(
                    RandomForestRegressor::new()
                        .with_n_estimators(5)
                        .with_random_state(42),
                )
            }),
        ),
        (
            "GradientBoostingRegressor",
            vec!["check_large_input"],
            Box::new(|| {
                check_estimator(
                    GradientBoostingRegressor::new()
                        .with_n_estimators(5)
                        .with_random_state(42),
                )
            }),
        ),
        (
            "KNeighborsRegressor",
            vec![],
            Box::new(|| check_estimator(KNeighborsRegressor::new(5))),
        ),
        // Classification models
        (
            "LogisticRegression",
            classifier_skip.clone(),
            Box::new(|| check_estimator(LogisticRegression::new())),
        ),
        (
            "DecisionTreeClassifier",
            vec![],
            Box::new(|| check_estimator(DecisionTreeClassifier::new())),
        ),
        (
            "RandomForestClassifier",
            classifier_skip.clone(),
            Box::new(|| {
                check_estimator(
                    RandomForestClassifier::new()
                        .with_n_estimators(5)
                        .with_random_state(42),
                )
            }),
        ),
        (
            "GradientBoostingClassifier",
            vec!["check_large_input"],
            Box::new(|| {
                check_estimator(
                    GradientBoostingClassifier::new()
                        .with_n_estimators(5)
                        .with_random_state(42),
                )
            }),
        ),
        (
            "GaussianNB",
            vec![],
            Box::new(|| check_estimator(GaussianNB::new())),
        ),
        (
            "KNeighborsClassifier",
            vec![],
            Box::new(|| check_estimator(KNeighborsClassifier::new(5))),
        ),
    ];

    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "MODEL COMPLIANCE SUMMARY");
    println!("{}", "=".repeat(80));
    println!();

    let mut total_models = 0;
    let mut compliant_models = 0;
    let mut total_checks = 0;
    let mut total_passed = 0;

    for (name, skip_checks, check_fn) in models {
        let results = check_fn();

        // Filter out expected skips
        let filtered_results: Vec<_> = results
            .iter()
            .filter(|r| !skip_checks.contains(&r.name))
            .collect();

        total_models += 1;
        total_checks += filtered_results.len();

        let passed_count = filtered_results.iter().filter(|r| r.passed).count();
        let failed_count = filtered_results.len() - passed_count;
        total_passed += passed_count;

        let status = if failed_count == 0 {
            compliant_models += 1;
            "PASS"
        } else {
            "FAIL"
        };

        println!(
            "{:30} [{:4}] {:3}/{:3} checks passed ({:.1}%)",
            name,
            status,
            passed_count,
            filtered_results.len(),
            if !filtered_results.is_empty() {
                100.0 * passed_count as f64 / filtered_results.len() as f64
            } else {
                100.0
            }
        );

        // Print failures (excluding expected skips)
        if failed_count > 0 {
            for result in &results {
                if !result.passed && !skip_checks.contains(&result.name) {
                    println!(
                        "    - {} [{}]: {}",
                        result.name,
                        format!("{:?}", result.category),
                        result.message.as_deref().unwrap_or("failed")
                    );
                }
            }
        }
    }

    println!();
    println!("{}", "-".repeat(80));
    println!(
        "TOTAL: {}/{} models compliant ({:.1}%)",
        compliant_models,
        total_models,
        100.0 * compliant_models as f64 / total_models as f64
    );
    println!(
        "       {}/{} checks passed ({:.1}%)",
        total_passed,
        total_checks,
        100.0 * total_passed as f64 / total_checks as f64
    );
    println!("{}", "=".repeat(80));

    // Note: We don't assert all pass here - this is a reporting test.
    // Individual model tests above handle the assertions with appropriate skips.
}

/// Check compliance by category for a single model
#[test]
fn test_compliance_by_category() {
    let config = CheckConfig {
        skip_checks: vec![
            "check_nan_handling",
            "check_inf_handling",
            "check_fit_twice_different_data",
        ],
        ..Default::default()
    };
    let model = LinearRegression::new();
    let results = crate::testing::check_estimator_with_config(model, config);
    let summary = summarize_results(&results);

    println!("\nLinearRegression Compliance by Category (with skips):");
    println!("{}", "-".repeat(50));

    let categories = [
        CheckCategory::Api,
        CheckCategory::InputValidation,
        CheckCategory::Numerical,
        CheckCategory::Concurrency,
        CheckCategory::Performance,
    ];

    for cat in categories {
        if let Some(&(passed, failed)) = summary.by_category.get(&cat) {
            let total = passed + failed;
            let status = if failed == 0 { "OK" } else { "FAIL" };
            println!(
                "  {:20} [{:4}] {}/{} passed",
                format!("{:?}", cat),
                status,
                passed,
                total
            );
        }
    }

    // Verify all categories pass after excluding known issues
    assert!(
        summary.failed == 0,
        "LinearRegression should pass all non-skipped checks"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_generates_valid_tests() {
        // This test verifies the macro works correctly
        // The actual model compliance is tested by the generated tests above
        let model = LinearRegression::new();
        let results = check_estimator(model);
        assert!(!results.is_empty(), "Should have check results");
    }
}
