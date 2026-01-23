//! Checks specific to Transformer trait
//!
//! These checks verify that preprocessing transformers conform to the
//! expected API and behave correctly.

use crate::preprocessing::Transformer;
use crate::FerroError;
use ndarray::Array2;

use super::{CheckCategory, CheckResult};

/// Check that unfitted transformer returns NotFitted error
pub fn check_transformer_not_fitted<T: Transformer>(transformer: &T) -> CheckResult {
    let x = Array2::zeros((5, 3));
    let result = transformer.transform(&x);

    match &result {
        Err(FerroError::NotFitted { .. }) => {
            CheckResult::pass("check_transformer_not_fitted", CheckCategory::Api)
        }
        Err(e) => CheckResult::fail(
            "check_transformer_not_fitted",
            CheckCategory::Api,
            format!("Expected NotFitted error, got: {:?}", e),
        ),
        Ok(_) => CheckResult::fail(
            "check_transformer_not_fitted",
            CheckCategory::Api,
            "Transform succeeded on unfitted transformer",
        ),
    }
}

/// Check that fit returns Ok on valid data
pub fn check_transformer_fit<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    let x = Array2::from_shape_fn((20, 5), |(i, j)| (i + j) as f64);

    match transformer.fit(&x) {
        Ok(()) => {
            if !transformer.is_fitted() {
                return CheckResult::fail(
                    "check_transformer_fit",
                    CheckCategory::Api,
                    "Fit succeeded but is_fitted() returns false",
                );
            }
            CheckResult::pass("check_transformer_fit", CheckCategory::Api)
        }
        Err(e) => CheckResult::fail(
            "check_transformer_fit",
            CheckCategory::Api,
            format!("Fit failed on valid data: {:?}", e),
        ),
    }
}

/// Check that fit_transform equals fit then transform
pub fn check_fit_transform_equivalence<T: Transformer + Clone>(transformer: T) -> CheckResult {
    let x = Array2::from_shape_fn((20, 5), |(i, j)| (i * 3 + j) as f64);

    // Method 1: fit then transform
    let mut t1 = transformer.clone();
    let result1 = match t1.fit(&x) {
        Ok(()) => t1.transform(&x),
        Err(e) => {
            return CheckResult::fail(
                "check_fit_transform_equivalence",
                CheckCategory::Api,
                format!("Fit failed: {:?}", e),
            )
        }
    };

    // Method 2: fit_transform
    let mut t2 = transformer;
    let result2 = t2.fit_transform(&x);

    match (result1, result2) {
        (Ok(r1), Ok(r2)) => {
            if r1.shape() != r2.shape() {
                return CheckResult::fail(
                    "check_fit_transform_equivalence",
                    CheckCategory::Api,
                    format!(
                        "Shapes differ: {:?} vs {:?}",
                        r1.shape(),
                        r2.shape()
                    ),
                );
            }

            let max_diff = r1
                .iter()
                .zip(r2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            if max_diff > 1e-10 {
                return CheckResult::fail(
                    "check_fit_transform_equivalence",
                    CheckCategory::Api,
                    format!("Results differ by {:.2e}", max_diff),
                );
            }

            CheckResult::pass("check_fit_transform_equivalence", CheckCategory::Api)
        }
        (Err(_), Err(_)) => {
            // Both failed, that's consistent
            CheckResult::pass("check_fit_transform_equivalence", CheckCategory::Api)
        }
        (Ok(_), Err(e)) => CheckResult::fail(
            "check_fit_transform_equivalence",
            CheckCategory::Api,
            format!("fit+transform succeeded but fit_transform failed: {:?}", e),
        ),
        (Err(e), Ok(_)) => CheckResult::fail(
            "check_fit_transform_equivalence",
            CheckCategory::Api,
            format!("fit+transform failed but fit_transform succeeded: {:?}", e),
        ),
    }
}

/// Check that transform output has correct shape
pub fn check_transform_shape<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    let x = Array2::from_shape_fn((20, 5), |(i, j)| (i + j) as f64);

    if let Err(e) = transformer.fit(&x) {
        return CheckResult::fail(
            "check_transform_shape",
            CheckCategory::Api,
            format!("Fit failed: {:?}", e),
        );
    }

    match transformer.transform(&x) {
        Ok(result) => {
            // Number of rows should be preserved
            if result.nrows() != x.nrows() {
                return CheckResult::fail(
                    "check_transform_shape",
                    CheckCategory::Api,
                    format!(
                        "Row count changed from {} to {}",
                        x.nrows(),
                        result.nrows()
                    ),
                );
            }

            // n_features_out should match if available
            if let Some(n_out) = transformer.n_features_out() {
                if result.ncols() != n_out {
                    return CheckResult::fail(
                        "check_transform_shape",
                        CheckCategory::Api,
                        format!(
                            "n_features_out()={} but transform has {} cols",
                            n_out,
                            result.ncols()
                        ),
                    );
                }
            }

            CheckResult::pass("check_transform_shape", CheckCategory::Api)
        }
        Err(e) => CheckResult::fail(
            "check_transform_shape",
            CheckCategory::Api,
            format!("Transform failed: {:?}", e),
        ),
    }
}

/// Check that inverse_transform recovers original (when supported)
pub fn check_inverse_transform<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    let x = Array2::from_shape_fn((20, 5), |(i, j)| (i * 2 + j) as f64 + 1.0);

    if let Err(e) = transformer.fit(&x) {
        return CheckResult::fail(
            "check_inverse_transform",
            CheckCategory::Numerical,
            format!("Fit failed: {:?}", e),
        );
    }

    let transformed = match transformer.transform(&x) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::fail(
                "check_inverse_transform",
                CheckCategory::Numerical,
                format!("Transform failed: {:?}", e),
            )
        }
    };

    match transformer.inverse_transform(&transformed) {
        Ok(recovered) => {
            if recovered.shape() != x.shape() {
                return CheckResult::fail(
                    "check_inverse_transform",
                    CheckCategory::Numerical,
                    format!(
                        "Shape mismatch: original {:?}, recovered {:?}",
                        x.shape(),
                        recovered.shape()
                    ),
                );
            }

            let max_diff = x
                .iter()
                .zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            if max_diff > 1e-6 {
                return CheckResult::fail(
                    "check_inverse_transform",
                    CheckCategory::Numerical,
                    format!("Inverse transform differs by {:.2e}", max_diff),
                );
            }

            CheckResult::pass("check_inverse_transform", CheckCategory::Numerical)
        }
        Err(FerroError::NotImplemented(_)) => {
            // inverse_transform not supported, that's fine
            CheckResult::pass("check_inverse_transform", CheckCategory::Numerical)
        }
        Err(e) => CheckResult::fail(
            "check_inverse_transform",
            CheckCategory::Numerical,
            format!("Inverse transform failed: {:?}", e),
        ),
    }
}

/// Check that transformer handles empty data correctly
pub fn check_transformer_empty_data<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    let x = Array2::zeros((0, 5));

    match transformer.fit(&x) {
        Err(_) => CheckResult::pass("check_transformer_empty_data", CheckCategory::InputValidation),
        Ok(()) => {
            // Some transformers might accept empty data, check transform too
            match transformer.transform(&x) {
                Ok(result) if result.nrows() == 0 => {
                    CheckResult::pass("check_transformer_empty_data", CheckCategory::InputValidation)
                }
                Ok(_) => CheckResult::fail(
                    "check_transformer_empty_data",
                    CheckCategory::InputValidation,
                    "Transform of empty data produced non-empty result",
                ),
                Err(_) => CheckResult::pass(
                    "check_transformer_empty_data",
                    CheckCategory::InputValidation,
                ),
            }
        }
    }
}

/// Check that transformer handles NaN correctly
pub fn check_transformer_nan_handling<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    let mut x = Array2::from_shape_fn((10, 5), |(i, j)| (i + j) as f64);
    x[[3, 2]] = f64::NAN;

    match transformer.fit(&x) {
        Err(_) => {
            // Rejecting NaN is acceptable
            CheckResult::pass("check_transformer_nan_handling", CheckCategory::InputValidation)
        }
        Ok(()) => {
            // Accepting NaN means it should handle it consistently
            match transformer.transform(&x) {
                Ok(result) => {
                    // Check for infinite values or other issues
                    for &v in result.iter() {
                        if v.is_infinite() {
                            return CheckResult::fail(
                                "check_transformer_nan_handling",
                                CheckCategory::InputValidation,
                                "NaN input produced infinite output",
                            );
                        }
                    }
                    CheckResult::pass(
                        "check_transformer_nan_handling",
                        CheckCategory::InputValidation,
                    )
                }
                Err(_) => CheckResult::pass(
                    "check_transformer_nan_handling",
                    CheckCategory::InputValidation,
                ),
            }
        }
    }
}

/// Check that n_features_in is tracked correctly
pub fn check_transformer_n_features_in<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    let x = Array2::from_shape_fn((20, 7), |(i, j)| (i + j) as f64);

    if let Err(e) = transformer.fit(&x) {
        return CheckResult::fail(
            "check_transformer_n_features_in",
            CheckCategory::Api,
            format!("Fit failed: {:?}", e),
        );
    }

    if let Some(n_in) = transformer.n_features_in() {
        if n_in != x.ncols() {
            return CheckResult::fail(
                "check_transformer_n_features_in",
                CheckCategory::Api,
                format!("n_features_in()={} but fitted on {} features", n_in, x.ncols()),
            );
        }
    }

    // Test that wrong number of features is rejected
    let x_wrong = Array2::zeros((5, 4)); // Different number of features
    match transformer.transform(&x_wrong) {
        Err(FerroError::ShapeMismatch { .. }) => {
            CheckResult::pass("check_transformer_n_features_in", CheckCategory::Api)
        }
        Err(_) => {
            // Some other error is acceptable
            CheckResult::pass("check_transformer_n_features_in", CheckCategory::Api)
        }
        Ok(_) => CheckResult::fail(
            "check_transformer_n_features_in",
            CheckCategory::Api,
            "Transformer accepted wrong number of features",
        ),
    }
}

/// Check that fit does not modify input
pub fn check_transformer_fit_no_modify<T: Transformer + Clone>(mut transformer: T) -> CheckResult {
    let x = Array2::from_shape_fn((20, 5), |(i, j)| (i * 2 + j) as f64);
    let x_copy = x.clone();

    let _ = transformer.fit(&x);

    let unchanged = x.iter().zip(x_copy.iter()).all(|(a, b)| {
        (a == b) || (a.is_nan() && b.is_nan())
    });

    if unchanged {
        CheckResult::pass("check_transformer_fit_no_modify", CheckCategory::Api)
    } else {
        CheckResult::fail(
            "check_transformer_fit_no_modify",
            CheckCategory::Api,
            "Fit modified input array",
        )
    }
}

/// Check that transform does not modify input
pub fn check_transformer_transform_no_modify<T: Transformer + Clone>(
    mut transformer: T,
) -> CheckResult {
    let x = Array2::from_shape_fn((20, 5), |(i, j)| (i * 2 + j) as f64);

    if let Err(e) = transformer.fit(&x) {
        return CheckResult::fail(
            "check_transformer_transform_no_modify",
            CheckCategory::Api,
            format!("Fit failed: {:?}", e),
        );
    }

    let x_copy = x.clone();
    let _ = transformer.transform(&x);

    let unchanged = x.iter().zip(x_copy.iter()).all(|(a, b)| {
        (a == b) || (a.is_nan() && b.is_nan())
    });

    if unchanged {
        CheckResult::pass("check_transformer_transform_no_modify", CheckCategory::Api)
    } else {
        CheckResult::fail(
            "check_transformer_transform_no_modify",
            CheckCategory::Api,
            "Transform modified input array",
        )
    }
}

/// Check transformer handles constant features
pub fn check_transformer_constant_feature<T: Transformer + Clone>(
    mut transformer: T,
) -> CheckResult {
    let mut x = Array2::from_shape_fn((20, 5), |(i, j)| (i + j) as f64);
    // Make first column constant
    x.column_mut(0).fill(5.0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        transformer.fit(&x)
    }));

    match result {
        Ok(Ok(())) => {
            // Fit succeeded, try transform
            match transformer.transform(&x) {
                Ok(result) => {
                    // Check for NaN/Inf in output
                    for &v in result.iter() {
                        if v.is_nan() || v.is_infinite() {
                            return CheckResult::fail(
                                "check_transformer_constant_feature",
                                CheckCategory::Numerical,
                                "Constant feature produced NaN/Inf",
                            );
                        }
                    }
                    CheckResult::pass("check_transformer_constant_feature", CheckCategory::Numerical)
                }
                Err(_) => {
                    // Transform failed on constant feature, acceptable
                    CheckResult::pass("check_transformer_constant_feature", CheckCategory::Numerical)
                }
            }
        }
        Ok(Err(_)) => {
            // Fit rejected constant feature, acceptable
            CheckResult::pass("check_transformer_constant_feature", CheckCategory::Numerical)
        }
        Err(_) => CheckResult::fail(
            "check_transformer_constant_feature",
            CheckCategory::Numerical,
            "Panicked on constant feature",
        ),
    }
}

/// Run all transformer checks
pub fn check_transformer<T: Transformer + Clone>(transformer: T) -> Vec<CheckResult> {
    let mut results = Vec::new();

    results.push(check_transformer_not_fitted(&transformer));
    results.push(check_transformer_fit(transformer.clone()));
    results.push(check_fit_transform_equivalence(transformer.clone()));
    results.push(check_transform_shape(transformer.clone()));
    results.push(check_inverse_transform(transformer.clone()));
    results.push(check_transformer_empty_data(transformer.clone()));
    results.push(check_transformer_nan_handling(transformer.clone()));
    results.push(check_transformer_n_features_in(transformer.clone()));
    results.push(check_transformer_fit_no_modify(transformer.clone()));
    results.push(check_transformer_transform_no_modify(transformer.clone()));
    results.push(check_transformer_constant_feature(transformer.clone()));

    results
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    // Tests would use actual transformers when available
}
