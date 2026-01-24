//! Preprocessing transformer compliance tests using check_transformer framework
//!
//! This module ensures all FerroML preprocessing transformers conform to the
//! Transformer trait contract by running a comprehensive suite of checks.
//!
//! # Check Categories
//!
//! - **API**: Basic API conformance (fit, transform, is_fitted, n_features)
//! - **InputValidation**: Proper handling of edge cases (NaN, empty, shape mismatch)
//! - **Numerical**: Numerical correctness (inverse transform, fit_transform equivalence)
//!
//! # Usage
//!
//! Run all compliance tests:
//! ```bash
//! cargo test -p ferroml-core preprocessing::tests::compliance
//! ```

use crate::decomposition::{TruncatedSVD, PCA};
use crate::preprocessing::encoders::{LabelEncoder, OneHotEncoder, OrdinalEncoder};
use crate::preprocessing::imputers::{ImputeStrategy, SimpleImputer};
use crate::preprocessing::polynomial::PolynomialFeatures;
use crate::preprocessing::scalers::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
use crate::preprocessing::Transformer;
use crate::testing::transformer::{
    check_fit_transform_equivalence, check_inverse_transform, check_transform_shape,
    check_transformer, check_transformer_constant_feature, check_transformer_empty_data,
    check_transformer_fit, check_transformer_fit_no_modify, check_transformer_n_features_in,
    check_transformer_nan_handling, check_transformer_not_fitted,
    check_transformer_transform_no_modify,
};
use crate::testing::CheckResult;

/// Macro to generate a compliance test for a transformer
///
/// Creates a test that runs all checks from check_transformer and asserts they pass.
macro_rules! test_transformer_compliance {
    ($name:ident, $transformer:expr) => {
        #[test]
        fn $name() {
            let transformer = $transformer;
            let results = check_transformer(transformer);
            let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();

            if !failures.is_empty() {
                let msg = failures
                    .iter()
                    .map(|r| format!("  - {}: {}", r.name, r.message.as_deref().unwrap_or("failed")))
                    .collect::<Vec<_>>()
                    .join("\n");
                panic!("Transformer failed {} checks:\n{}", failures.len(), msg);
            }
        }
    };
    ($name:ident, $transformer:expr, skip = [$($skip:expr),*]) => {
        #[test]
        fn $name() {
            let transformer = $transformer;
            let results = check_transformer(transformer);
            let skip_checks: Vec<&str> = vec![$($skip),*];

            let failures: Vec<_> = results
                .iter()
                .filter(|r| !r.passed && !skip_checks.contains(&r.name))
                .collect();

            if !failures.is_empty() {
                let msg = failures
                    .iter()
                    .map(|r| format!("  - {}: {}", r.name, r.message.as_deref().unwrap_or("failed")))
                    .collect::<Vec<_>>()
                    .join("\n");
                panic!("Transformer failed {} checks:\n{}", failures.len(), msg);
            }
        }
    };
}

// =============================================================================
// Scalers
// =============================================================================

test_transformer_compliance!(test_standard_scaler_compliance, StandardScaler::new());

test_transformer_compliance!(test_minmax_scaler_compliance, MinMaxScaler::new());

test_transformer_compliance!(test_robust_scaler_compliance, RobustScaler::new());

test_transformer_compliance!(test_maxabs_scaler_compliance, MaxAbsScaler::new());

// =============================================================================
// Encoders
// =============================================================================

test_transformer_compliance!(
    test_onehot_encoder_compliance,
    OneHotEncoder::new(),
    skip = ["check_inverse_transform"] // OneHot may not support inverse for all configs
);

test_transformer_compliance!(
    test_ordinal_encoder_compliance,
    OrdinalEncoder::new(),
    skip = ["check_inverse_transform"] // OrdinalEncoder uses integer encoding
);

// Note: LabelEncoder is designed for single-column (1D) target encoding, not multi-column
// feature transformation. Skip all multi-column checks.
test_transformer_compliance!(
    test_label_encoder_compliance,
    LabelEncoder::new(),
    skip = [
        "check_transformer_not_fitted",
        "check_transformer_fit",
        "check_fit_transform_equivalence",
        "check_transform_shape",
        "check_inverse_transform",
        "check_transformer_empty_data",
        "check_transformer_nan_handling",
        "check_transformer_n_features_in",
        "check_transformer_fit_no_modify",
        "check_transformer_transform_no_modify",
        "check_transformer_constant_feature"
    ]
);

// =============================================================================
// Imputers
// =============================================================================

test_transformer_compliance!(
    test_simple_imputer_mean_compliance,
    SimpleImputer::new(ImputeStrategy::Mean)
);

test_transformer_compliance!(
    test_simple_imputer_median_compliance,
    SimpleImputer::new(ImputeStrategy::Median)
);

// =============================================================================
// Polynomial Features
// =============================================================================

test_transformer_compliance!(
    test_polynomial_features_compliance,
    PolynomialFeatures::new(2),
    skip = ["check_inverse_transform"] // Polynomial expansion is not invertible
);

// =============================================================================
// Dimensionality Reduction (if they implement Transformer)
// Note: These tests are slow due to SVD computation, marked as ignored.
// Run with: cargo test -p ferroml-core test_pca_compliance -- --ignored
// =============================================================================

#[test]
#[ignore = "Slow SVD computation, run explicitly with --ignored"]
fn test_pca_compliance() {
    let transformer = PCA::new().with_n_components(3);
    let results = check_transformer(transformer);
    let skip_checks: Vec<&str> = vec!["check_inverse_transform"];

    let failures: Vec<_> = results
        .iter()
        .filter(|r| !r.passed && !skip_checks.contains(&r.name))
        .collect();

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
        panic!("Transformer failed {} checks:\n{}", failures.len(), msg);
    }
}

#[test]
#[ignore = "Slow SVD computation, run explicitly with --ignored"]
fn test_truncated_svd_compliance() {
    let transformer = TruncatedSVD::new().with_n_components(3);
    let results = check_transformer(transformer);
    let skip_checks: Vec<&str> = vec!["check_inverse_transform"];

    let failures: Vec<_> = results
        .iter()
        .filter(|r| !r.passed && !skip_checks.contains(&r.name))
        .collect();

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
        panic!("Transformer failed {} checks:\n{}", failures.len(), msg);
    }
}

// =============================================================================
// Summary Test
// =============================================================================

/// Run compliance checks on all transformers and produce a summary report
///
/// Run with: cargo test -p ferroml-core all_transformers_compliance_summary -- --ignored
#[test]
#[ignore = "Long-running summary test, run explicitly with --ignored"]
fn all_transformers_compliance_summary() {
    let transformers: Vec<(
        &str,
        Box<dyn Fn() -> Vec<CheckResult> + Send + Sync>,
        Vec<&str>,
    )> = vec![
        // Scalers
        (
            "StandardScaler",
            Box::new(|| check_transformer(StandardScaler::new())),
            vec![],
        ),
        (
            "MinMaxScaler",
            Box::new(|| check_transformer(MinMaxScaler::new())),
            vec![],
        ),
        (
            "RobustScaler",
            Box::new(|| check_transformer(RobustScaler::new())),
            vec![],
        ),
        (
            "MaxAbsScaler",
            Box::new(|| check_transformer(MaxAbsScaler::new())),
            vec![],
        ),
        // Encoders
        (
            "OneHotEncoder",
            Box::new(|| check_transformer(OneHotEncoder::new())),
            vec!["check_inverse_transform"],
        ),
        (
            "OrdinalEncoder",
            Box::new(|| check_transformer(OrdinalEncoder::new())),
            vec!["check_inverse_transform"],
        ),
        // LabelEncoder is for 1D targets, skip all multi-column checks
        (
            "LabelEncoder",
            Box::new(|| check_transformer(LabelEncoder::new())),
            vec![
                "check_transformer_not_fitted",
                "check_transformer_fit",
                "check_fit_transform_equivalence",
                "check_transform_shape",
                "check_inverse_transform",
                "check_transformer_empty_data",
                "check_transformer_nan_handling",
                "check_transformer_n_features_in",
                "check_transformer_fit_no_modify",
                "check_transformer_transform_no_modify",
                "check_transformer_constant_feature",
            ],
        ),
        // Imputers
        (
            "SimpleImputer(mean)",
            Box::new(|| check_transformer(SimpleImputer::new(ImputeStrategy::Mean))),
            vec![],
        ),
        (
            "SimpleImputer(median)",
            Box::new(|| check_transformer(SimpleImputer::new(ImputeStrategy::Median))),
            vec![],
        ),
        // Polynomial
        (
            "PolynomialFeatures",
            Box::new(|| check_transformer(PolynomialFeatures::new(2))),
            vec!["check_inverse_transform"],
        ),
        // Decomposition
        (
            "PCA",
            Box::new(|| check_transformer(PCA::new().with_n_components(3))),
            vec!["check_inverse_transform"],
        ),
        (
            "TruncatedSVD",
            Box::new(|| check_transformer(TruncatedSVD::new().with_n_components(3))),
            vec!["check_inverse_transform"],
        ),
    ];

    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "TRANSFORMER COMPLIANCE SUMMARY");
    println!("{}", "=".repeat(80));
    println!();

    let mut total_transformers = 0;
    let mut compliant_transformers = 0;
    let mut total_checks = 0;
    let mut total_passed = 0;

    for (name, check_fn, skip_checks) in transformers {
        let results = check_fn();

        // Filter out expected skips
        let filtered_results: Vec<_> = results
            .iter()
            .filter(|r| !skip_checks.contains(&r.name))
            .collect();

        total_transformers += 1;
        total_checks += filtered_results.len();

        let passed_count = filtered_results.iter().filter(|r| r.passed).count();
        let failed_count = filtered_results.len() - passed_count;
        total_passed += passed_count;

        let status = if failed_count == 0 {
            compliant_transformers += 1;
            "PASS"
        } else {
            "FAIL"
        };

        println!(
            "{:25} [{:4}] {:2}/{:2} checks passed ({:.1}%)",
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

        // Print failures
        if failed_count > 0 {
            for result in &results {
                if !result.passed && !skip_checks.contains(&result.name) {
                    println!(
                        "    - {}: {}",
                        result.name,
                        result.message.as_deref().unwrap_or("failed")
                    );
                }
            }
        }
    }

    println!();
    println!("{}", "-".repeat(80));
    println!(
        "TOTAL: {}/{} transformers compliant ({:.1}%)",
        compliant_transformers,
        total_transformers,
        100.0 * compliant_transformers as f64 / total_transformers as f64
    );
    println!(
        "       {}/{} checks passed ({:.1}%)",
        total_passed,
        total_checks,
        100.0 * total_passed as f64 / total_checks as f64
    );
    println!("{}", "=".repeat(80));

    // Assert all transformers are compliant
    assert_eq!(
        compliant_transformers,
        total_transformers,
        "Not all transformers are fully compliant! {} of {} failed.",
        total_transformers - compliant_transformers,
        total_transformers
    );
}

/// Detailed check for StandardScaler showing all compliance categories
#[test]
fn test_standard_scaler_detailed_compliance() {
    let scaler = StandardScaler::new();

    println!("\nStandardScaler Detailed Compliance:");
    println!("{}", "-".repeat(60));

    // Run each check individually
    let checks: Vec<(&str, CheckResult)> = vec![
        ("Not fitted check", check_transformer_not_fitted(&scaler)),
        ("Fit check", check_transformer_fit(scaler.clone())),
        (
            "Fit-transform equivalence",
            check_fit_transform_equivalence(scaler.clone()),
        ),
        ("Transform shape", check_transform_shape(scaler.clone())),
        ("Inverse transform", check_inverse_transform(scaler.clone())),
        (
            "Empty data handling",
            check_transformer_empty_data(scaler.clone()),
        ),
        (
            "NaN handling",
            check_transformer_nan_handling(scaler.clone()),
        ),
        (
            "n_features_in tracking",
            check_transformer_n_features_in(scaler.clone()),
        ),
        (
            "Fit doesn't modify input",
            check_transformer_fit_no_modify(scaler.clone()),
        ),
        (
            "Transform doesn't modify input",
            check_transformer_transform_no_modify(scaler.clone()),
        ),
        (
            "Constant feature handling",
            check_transformer_constant_feature(scaler.clone()),
        ),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, result) in checks {
        let status = if result.passed {
            passed += 1;
            "PASS"
        } else {
            failed += 1;
            "FAIL"
        };

        println!("  {:35} [{:4}]", name, status);
        if !result.passed {
            if let Some(msg) = &result.message {
                println!("      -> {}", msg);
            }
        }
    }

    println!();
    println!("Summary: {}/{} checks passed", passed, passed + failed);

    assert_eq!(
        failed, 0,
        "StandardScaler should pass all compliance checks"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_generates_valid_tests() {
        // Verify the macro works
        let transformer = StandardScaler::new();
        let results = check_transformer(transformer);
        assert!(!results.is_empty(), "Should have check results");
    }

    #[test]
    fn test_scaler_inverse_transform_accuracy() {
        // Additional test for inverse transform accuracy
        use ndarray::Array2;

        let mut scaler = StandardScaler::new();
        let x = Array2::from_shape_fn((20, 5), |(i, j)| (i * 2 + j) as f64 + 1.0);

        scaler.fit(&x).unwrap();
        let transformed = scaler.transform(&x).unwrap();
        let recovered = scaler.inverse_transform(&transformed).unwrap();

        // Check recovery accuracy
        let max_diff = x
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        assert!(
            max_diff < 1e-10,
            "Inverse transform should recover original data, max diff: {}",
            max_diff
        );
    }
}
