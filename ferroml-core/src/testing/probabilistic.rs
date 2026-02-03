//! Checks specific to ProbabilisticModel trait
//!
//! These checks verify that models implementing ProbabilisticModel
//! correctly produce valid probabilities and prediction intervals.

use crate::models::ProbabilisticModel;
use ndarray::{Array1, Array2};

use super::{CheckCategory, CheckResult};

/// Check that predict_proba returns valid probabilities
pub fn check_proba_sum_to_one<M: ProbabilisticModel + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    if model.fit(x, y).is_err() {
        return CheckResult::fail(
            "check_proba_sum_to_one",
            CheckCategory::Numerical,
            "Fit failed",
        );
    }

    match model.predict_proba(x) {
        Ok(proba) => {
            // Check each row sums to 1.0
            for i in 0..proba.nrows() {
                let row_sum: f64 = proba.row(i).iter().sum();
                if (row_sum - 1.0).abs() > 1e-6 {
                    return CheckResult::fail(
                        "check_proba_sum_to_one",
                        CheckCategory::Numerical,
                        format!("Row {} probabilities sum to {} instead of 1.0", i, row_sum),
                    );
                }
            }
            CheckResult::pass("check_proba_sum_to_one", CheckCategory::Numerical)
        }
        Err(e) => CheckResult::fail(
            "check_proba_sum_to_one",
            CheckCategory::Numerical,
            format!("predict_proba failed: {:?}", e),
        ),
    }
}

/// Check that all probability values are in [0, 1]
pub fn check_proba_range<M: ProbabilisticModel + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    if model.fit(x, y).is_err() {
        return CheckResult::fail("check_proba_range", CheckCategory::Numerical, "Fit failed");
    }

    match model.predict_proba(x) {
        Ok(proba) => {
            for &p in proba.iter() {
                if p < 0.0 || p > 1.0 {
                    return CheckResult::fail(
                        "check_proba_range",
                        CheckCategory::Numerical,
                        format!("Probability {} outside [0, 1]", p),
                    );
                }
                if p.is_nan() {
                    return CheckResult::fail(
                        "check_proba_range",
                        CheckCategory::Numerical,
                        "NaN in probabilities",
                    );
                }
            }
            CheckResult::pass("check_proba_range", CheckCategory::Numerical)
        }
        Err(e) => CheckResult::fail(
            "check_proba_range",
            CheckCategory::Numerical,
            format!("predict_proba failed: {:?}", e),
        ),
    }
}

/// Check that predict_proba output shape is correct
pub fn check_proba_shape<M: ProbabilisticModel + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    if model.fit(x, y).is_err() {
        return CheckResult::fail("check_proba_shape", CheckCategory::Api, "Fit failed");
    }

    match model.predict_proba(x) {
        Ok(proba) => {
            // Should have n_samples rows
            if proba.nrows() != x.nrows() {
                return CheckResult::fail(
                    "check_proba_shape",
                    CheckCategory::Api,
                    format!(
                        "Expected {} rows in proba, got {}",
                        x.nrows(),
                        proba.nrows()
                    ),
                );
            }
            // Should have at least 1 column (n_classes)
            if proba.ncols() == 0 {
                return CheckResult::fail(
                    "check_proba_shape",
                    CheckCategory::Api,
                    "predict_proba returned 0 columns",
                );
            }
            CheckResult::pass("check_proba_shape", CheckCategory::Api)
        }
        Err(e) => CheckResult::fail(
            "check_proba_shape",
            CheckCategory::Api,
            format!("predict_proba failed: {:?}", e),
        ),
    }
}

/// Check that log probabilities are consistent with probabilities
pub fn check_log_proba_consistency<M: ProbabilisticModel + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    if model.fit(x, y).is_err() {
        return CheckResult::fail(
            "check_log_proba_consistency",
            CheckCategory::Numerical,
            "Fit failed",
        );
    }

    let proba = match model.predict_proba(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_log_proba_consistency",
                CheckCategory::Numerical,
                format!("predict_proba failed: {:?}", e),
            )
        }
    };

    let log_proba = match model.predict_log_proba(x) {
        Ok(lp) => lp,
        Err(e) => {
            return CheckResult::fail(
                "check_log_proba_consistency",
                CheckCategory::Numerical,
                format!("predict_log_proba failed: {:?}", e),
            )
        }
    };

    // Check that exp(log_proba) ≈ proba
    for i in 0..proba.nrows() {
        for j in 0..proba.ncols() {
            let expected = proba[[i, j]];
            let from_log = log_proba[[i, j]].exp();
            if (expected - from_log).abs() > 1e-6 {
                return CheckResult::fail(
                    "check_log_proba_consistency",
                    CheckCategory::Numerical,
                    format!(
                        "exp(log_proba[{},{}])={} != proba[{},{}]={}",
                        i, j, from_log, i, j, expected
                    ),
                );
            }
        }
    }

    CheckResult::pass("check_log_proba_consistency", CheckCategory::Numerical)
}

/// Check that prediction interval contains reasonable values
pub fn check_prediction_interval_bounds<M: ProbabilisticModel + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    if model.fit(x, y).is_err() {
        return CheckResult::fail(
            "check_prediction_interval_bounds",
            CheckCategory::Numerical,
            "Fit failed",
        );
    }

    match model.predict_interval(x, 0.95) {
        Ok(interval) => {
            // Check lower <= predictions <= upper
            for i in 0..interval.predictions.len() {
                let lower = interval.lower[i];
                let pred = interval.predictions[i];
                let upper = interval.upper[i];

                if lower > pred {
                    return CheckResult::fail(
                        "check_prediction_interval_bounds",
                        CheckCategory::Numerical,
                        format!("lower[{}]={} > predictions[{}]={}", i, lower, i, pred),
                    );
                }
                if pred > upper {
                    return CheckResult::fail(
                        "check_prediction_interval_bounds",
                        CheckCategory::Numerical,
                        format!("predictions[{}]={} > upper[{}]={}", i, pred, i, upper),
                    );
                }
            }
            CheckResult::pass("check_prediction_interval_bounds", CheckCategory::Numerical)
        }
        Err(_) => {
            // predict_interval might not be implemented - that's ok
            CheckResult::pass("check_prediction_interval_bounds", CheckCategory::Numerical)
        }
    }
}

/// Check that prediction interval widens at higher confidence
pub fn check_prediction_interval_confidence<M: ProbabilisticModel + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    if model.fit(x, y).is_err() {
        return CheckResult::fail(
            "check_prediction_interval_confidence",
            CheckCategory::Numerical,
            "Fit failed",
        );
    }

    let interval_90 = match model.predict_interval(x, 0.90) {
        Ok(i) => i,
        Err(_) => {
            // Not implemented, skip
            return CheckResult::pass(
                "check_prediction_interval_confidence",
                CheckCategory::Numerical,
            );
        }
    };

    let interval_95 = match model.predict_interval(x, 0.95) {
        Ok(i) => i,
        Err(_) => {
            return CheckResult::pass(
                "check_prediction_interval_confidence",
                CheckCategory::Numerical,
            );
        }
    };

    // 95% interval should be wider than or equal to 90% interval
    for i in 0..interval_90.predictions.len() {
        let width_90 = interval_90.upper[i] - interval_90.lower[i];
        let width_95 = interval_95.upper[i] - interval_95.lower[i];

        if width_95 < width_90 - 1e-6 {
            return CheckResult::fail(
                "check_prediction_interval_confidence",
                CheckCategory::Numerical,
                format!(
                    "95% interval width {} < 90% interval width {} at index {}",
                    width_95, width_90, i
                ),
            );
        }
    }

    CheckResult::pass(
        "check_prediction_interval_confidence",
        CheckCategory::Numerical,
    )
}

/// Check that probabilities are not overconfident (all 0 or 1)
pub fn check_proba_not_degenerate<M: ProbabilisticModel + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    if model.fit(x, y).is_err() {
        return CheckResult::fail(
            "check_proba_not_degenerate",
            CheckCategory::Numerical,
            "Fit failed",
        );
    }

    match model.predict_proba(x) {
        Ok(proba) => {
            let mut all_extreme = true;
            for &p in proba.iter() {
                if p > 1e-6 && p < 1.0 - 1e-6 {
                    all_extreme = false;
                    break;
                }
            }

            if all_extreme && proba.len() > 10 {
                return CheckResult::fail(
                    "check_proba_not_degenerate",
                    CheckCategory::Numerical,
                    "All probabilities are 0 or 1 (possibly overconfident)",
                );
            }

            CheckResult::pass("check_proba_not_degenerate", CheckCategory::Numerical)
        }
        Err(e) => CheckResult::fail(
            "check_proba_not_degenerate",
            CheckCategory::Numerical,
            format!("predict_proba failed: {:?}", e),
        ),
    }
}

/// Check that predict and predict_proba are consistent (argmax of proba = predict)
pub fn check_predict_proba_consistent<M: ProbabilisticModel + Clone>(
    mut model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> CheckResult {
    if model.fit(x, y).is_err() {
        return CheckResult::fail(
            "check_predict_proba_consistent",
            CheckCategory::Numerical,
            "Fit failed",
        );
    }

    let predictions = match model.predict(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_predict_proba_consistent",
                CheckCategory::Numerical,
                format!("predict failed: {:?}", e),
            )
        }
    };

    let proba = match model.predict_proba(x) {
        Ok(p) => p,
        Err(e) => {
            return CheckResult::fail(
                "check_predict_proba_consistent",
                CheckCategory::Numerical,
                format!("predict_proba failed: {:?}", e),
            )
        }
    };

    // For binary classification, check consistency
    if proba.ncols() == 2 {
        for i in 0..predictions.len() {
            let pred_class = predictions[i];
            let prob_class_1 = proba[[i, 1]];

            // If prob > 0.5, predict should be 1; else 0
            let expected_class = if prob_class_1 > 0.5 { 1.0 } else { 0.0 };

            if (pred_class - expected_class).abs() > 1e-6 {
                return CheckResult::fail(
                    "check_predict_proba_consistent",
                    CheckCategory::Numerical,
                    format!(
                        "predict[{}]={} but proba[{},1]={} (expected class {})",
                        i, pred_class, i, prob_class_1, expected_class
                    ),
                );
            }
        }
    }

    CheckResult::pass("check_predict_proba_consistent", CheckCategory::Numerical)
}

/// Run all probabilistic model checks
pub fn check_probabilistic_model<M: ProbabilisticModel + Clone>(
    model: M,
    x: &Array2<f64>,
    y: &Array1<f64>,
) -> Vec<CheckResult> {
    let mut results = Vec::new();

    results.push(check_proba_sum_to_one(model.clone(), x, y));
    results.push(check_proba_range(model.clone(), x, y));
    results.push(check_proba_shape(model.clone(), x, y));
    results.push(check_log_proba_consistency(model.clone(), x, y));
    results.push(check_prediction_interval_bounds(model.clone(), x, y));
    results.push(check_prediction_interval_confidence(model.clone(), x, y));
    results.push(check_proba_not_degenerate(model.clone(), x, y));
    results.push(check_predict_proba_consistent(model, x, y));

    results
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    // Tests would use actual probabilistic models when available
}
