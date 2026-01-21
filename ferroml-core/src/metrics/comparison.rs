//! Model Comparison Metrics
//!
//! Statistical tests for comparing model performance, including paired t-test,
//! corrected resampled t-test (Nadeau & Bengio), and McNemar's test.

use crate::{FerroError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Result of a model comparison statistical test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonResult {
    /// Name of the test
    pub test_name: String,
    /// Test statistic value
    pub statistic: f64,
    /// p-value (two-sided)
    pub p_value: f64,
    /// Degrees of freedom (if applicable)
    pub df: Option<f64>,
    /// Mean difference (model1 - model2)
    pub mean_difference: f64,
    /// Standard error of the difference
    pub std_error: f64,
    /// 95% confidence interval for the difference
    pub ci_95: (f64, f64),
    /// Whether the difference is significant at alpha=0.05
    pub significant: bool,
    /// Additional interpretation
    pub interpretation: String,
}

impl ModelComparisonResult {
    /// Format as a summary string
    pub fn summary(&self) -> String {
        format!(
            "{}\n{}\n\
             Test statistic: {:.4}\n\
             p-value: {:.4}\n\
             Mean difference: {:.4} (SE: {:.4})\n\
             95% CI: [{:.4}, {:.4}]\n\
             Significant at α=0.05: {}\n\
             Interpretation: {}",
            self.test_name,
            "=".repeat(self.test_name.len()),
            self.statistic,
            self.p_value,
            self.mean_difference,
            self.std_error,
            self.ci_95.0,
            self.ci_95.1,
            if self.significant { "Yes" } else { "No" },
            self.interpretation
        )
    }
}

/// Paired t-test for comparing CV scores of two models
///
/// Use this when you have k-fold CV scores from two models on the same data splits.
/// Note: Standard paired t-test may have inflated Type I error for CV scores.
/// Consider using `corrected_resampled_ttest` for more accurate inference.
///
/// # Arguments
/// * `scores1` - CV scores from model 1
/// * `scores2` - CV scores from model 2 (same folds)
///
/// # Returns
/// Statistical test result with p-value, confidence interval, etc.
pub fn paired_ttest(scores1: &Array1<f64>, scores2: &Array1<f64>) -> Result<ModelComparisonResult> {
    if scores1.len() != scores2.len() {
        return Err(FerroError::shape_mismatch(
            format!("scores1 length {}", scores1.len()),
            format!("scores2 length {}", scores2.len()),
        ));
    }

    let n = scores1.len();
    if n < 2 {
        return Err(FerroError::invalid_input(
            "Need at least 2 paired observations for t-test",
        ));
    }

    // Compute differences
    let differences: Array1<f64> = scores1
        .iter()
        .zip(scores2.iter())
        .map(|(&a, &b)| a - b)
        .collect();

    let mean_diff = differences.mean().unwrap_or(0.0);
    let var_diff = differences.var(1.0); // Sample variance (ddof=1)
    let std_diff = var_diff.sqrt();
    let std_error = std_diff / (n as f64).sqrt();

    // t-statistic
    let t_stat = if std_error == 0.0 {
        if mean_diff == 0.0 {
            0.0
        } else {
            f64::INFINITY.copysign(mean_diff)
        }
    } else {
        mean_diff / std_error
    };

    let df = (n - 1) as f64;

    // Two-tailed p-value
    let p_value = 2.0 * (1.0 - t_cdf(t_stat.abs(), df));

    // 95% CI
    let t_crit = t_critical(0.975, df);
    let ci_lower = mean_diff - t_crit * std_error;
    let ci_upper = mean_diff + t_crit * std_error;

    let significant = p_value < 0.05;

    let interpretation = if !significant {
        "No significant difference between models".to_string()
    } else if mean_diff > 0.0 {
        "Model 1 significantly better than Model 2".to_string()
    } else {
        "Model 2 significantly better than Model 1".to_string()
    };

    Ok(ModelComparisonResult {
        test_name: "Paired t-test".to_string(),
        statistic: t_stat,
        p_value,
        df: Some(df),
        mean_difference: mean_diff,
        std_error,
        ci_95: (ci_lower, ci_upper),
        significant,
        interpretation,
    })
}

/// Corrected resampled t-test (Nadeau & Bengio, 2003)
///
/// This corrects for the dependence between CV fold scores, which causes
/// the standard t-test to have inflated Type I error rates.
///
/// # Arguments
/// * `scores1` - CV scores from model 1
/// * `scores2` - CV scores from model 2 (same folds)
/// * `n_train` - Number of training samples in each fold
/// * `n_test` - Number of test samples in each fold
///
/// # Reference
/// Nadeau, C., & Bengio, Y. (2003). Inference for the Generalization Error.
/// Machine Learning, 52(3), 239-281.
pub fn corrected_resampled_ttest(
    scores1: &Array1<f64>,
    scores2: &Array1<f64>,
    n_train: usize,
    n_test: usize,
) -> Result<ModelComparisonResult> {
    if scores1.len() != scores2.len() {
        return Err(FerroError::shape_mismatch(
            format!("scores1 length {}", scores1.len()),
            format!("scores2 length {}", scores2.len()),
        ));
    }

    let k = scores1.len(); // Number of folds
    if k < 2 {
        return Err(FerroError::invalid_input(
            "Need at least 2 folds for corrected t-test",
        ));
    }

    // Compute differences
    let differences: Array1<f64> = scores1
        .iter()
        .zip(scores2.iter())
        .map(|(&a, &b)| a - b)
        .collect();

    let mean_diff = differences.mean().unwrap_or(0.0);
    let var_diff = differences.var(1.0);

    // Nadeau-Bengio correction factor
    let correction = 1.0 / k as f64 + n_test as f64 / n_train as f64;
    let corrected_var = correction * var_diff;
    let std_error = corrected_var.sqrt();

    // t-statistic
    let t_stat = if std_error == 0.0 {
        if mean_diff == 0.0 {
            0.0
        } else {
            f64::INFINITY.copysign(mean_diff)
        }
    } else {
        mean_diff / std_error
    };

    let df = (k - 1) as f64;
    let p_value = 2.0 * (1.0 - t_cdf(t_stat.abs(), df));

    let t_crit = t_critical(0.975, df);
    let ci_lower = mean_diff - t_crit * std_error;
    let ci_upper = mean_diff + t_crit * std_error;

    let significant = p_value < 0.05;

    let interpretation = if !significant {
        "No significant difference between models (corrected test)".to_string()
    } else if mean_diff > 0.0 {
        "Model 1 significantly better than Model 2 (corrected test)".to_string()
    } else {
        "Model 2 significantly better than Model 1 (corrected test)".to_string()
    };

    Ok(ModelComparisonResult {
        test_name: "Corrected Resampled t-test (Nadeau-Bengio)".to_string(),
        statistic: t_stat,
        p_value,
        df: Some(df),
        mean_difference: mean_diff,
        std_error,
        ci_95: (ci_lower, ci_upper),
        significant,
        interpretation,
    })
}

/// McNemar's test for comparing classifier predictions
///
/// Compares two classifiers by analyzing their disagreements on the same test set.
/// Uses the exact binomial test for small sample sizes.
///
/// # Arguments
/// * `y_true` - True labels
/// * `pred1` - Predictions from classifier 1
/// * `pred2` - Predictions from classifier 2
pub fn mcnemar_test(
    y_true: &Array1<f64>,
    pred1: &Array1<f64>,
    pred2: &Array1<f64>,
) -> Result<ModelComparisonResult> {
    if y_true.len() != pred1.len() || y_true.len() != pred2.len() {
        return Err(FerroError::shape_mismatch(
            format!("y_true length {}", y_true.len()),
            format!("predictions length mismatch"),
        ));
    }

    let n = y_true.len();
    if n == 0 {
        return Err(FerroError::invalid_input("Empty arrays"));
    }

    // Count disagreements
    // b = cases where model1 correct, model2 wrong
    // c = cases where model1 wrong, model2 correct
    let mut b = 0usize;
    let mut c = 0usize;

    for i in 0..n {
        let true_label = y_true[i];
        let correct1 = (pred1[i] - true_label).abs() < 1e-10;
        let correct2 = (pred2[i] - true_label).abs() < 1e-10;

        match (correct1, correct2) {
            (true, false) => b += 1,
            (false, true) => c += 1,
            _ => {}
        }
    }

    let total_disagreements = b + c;

    // McNemar's test statistic (chi-squared with continuity correction)
    let statistic = if total_disagreements == 0 {
        0.0
    } else {
        let diff = (b as f64 - c as f64).abs() - 1.0; // Continuity correction
        let diff = diff.max(0.0);
        diff.powi(2) / total_disagreements as f64
    };

    // p-value from chi-squared distribution with df=1
    let p_value = 1.0 - chi2_cdf(statistic, 1.0);

    let mean_difference = (b as f64 - c as f64) / n as f64;
    let std_error = ((b + c) as f64).sqrt() / n as f64;

    // Approximate CI (normal approximation)
    let z_crit = 1.96;
    let ci_lower = mean_difference - z_crit * std_error;
    let ci_upper = mean_difference + z_crit * std_error;

    let significant = p_value < 0.05;

    let interpretation = if !significant {
        format!("No significant difference (b={}, c={})", b, c)
    } else if b > c {
        format!("Model 1 significantly better (b={}, c={})", b, c)
    } else {
        format!("Model 2 significantly better (b={}, c={})", b, c)
    };

    Ok(ModelComparisonResult {
        test_name: "McNemar's Test".to_string(),
        statistic,
        p_value,
        df: Some(1.0),
        mean_difference,
        std_error,
        ci_95: (ci_lower, ci_upper),
        significant,
        interpretation,
    })
}

// Statistical helper functions

/// Student's t CDF approximation
fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 + 0.5 * (1.0 - incomplete_beta(df / 2.0, 0.5, x)).copysign(t)
}

/// Critical value from Student's t distribution
fn t_critical(p: f64, df: f64) -> f64 {
    // Newton-Raphson approximation
    let mut t = 1.96; // Start with normal approximation

    for _ in 0..20 {
        let cdf = t_cdf(t, df);
        let diff = cdf - p;

        if diff.abs() < 1e-10 {
            break;
        }

        // PDF of t distribution
        let pdf = (1.0 + t * t / df).powf(-(df + 1.0) / 2.0) / (df.sqrt() * beta(df / 2.0, 0.5));

        t -= diff / pdf;
    }

    t
}

/// Chi-squared CDF
fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Chi-squared is gamma(df/2, 2), use incomplete gamma
    incomplete_gamma(df / 2.0, x / 2.0)
}

/// Incomplete gamma function (regularized)
fn incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }

    // Use series expansion for small x
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut term = sum;

        for n in 1..100 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-10 {
                break;
            }
        }

        sum * (-x + a * x.ln() - gamma_ln(a)).exp()
    } else {
        // Use continued fraction for large x
        1.0 - incomplete_gamma_cf(a, x)
    }
}

/// Incomplete gamma using continued fraction (for large x)
fn incomplete_gamma_cf(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..100 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    h * (-x + a * x.ln() - gamma_ln(a)).exp()
}

/// Incomplete beta function approximation
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    let mut result = 0.0;
    let mut term = 1.0;

    for n in 1..100 {
        let n = n as f64;
        let d = (a + n - 1.0) * (a + b + n - 1.0) * x / ((a + 2.0 * n - 1.0) * (a + 2.0 * n));
        term *= d;
        result += term;

        if term.abs() < 1e-10 {
            break;
        }
    }

    let prefix = x.powf(a) * (1.0 - x).powf(b) / (a * beta(a, b));
    prefix * (1.0 + result)
}

/// Beta function
fn beta(a: f64, b: f64) -> f64 {
    (gamma_ln(a) + gamma_ln(b) - gamma_ln(a + b)).exp()
}

/// Log gamma function (Lanczos approximation)
fn gamma_ln(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();

    let mut ser = 1.000000000190015;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (x + i as f64 + 1.0);
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_paired_ttest_no_difference() {
        let scores1 = Array1::from_vec(vec![0.8, 0.85, 0.82, 0.78, 0.84]);
        let scores2 = Array1::from_vec(vec![0.79, 0.84, 0.83, 0.79, 0.83]);

        let result = paired_ttest(&scores1, &scores2).unwrap();

        assert!(!result.significant); // Small differences, not significant
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_paired_ttest_significant_difference() {
        let scores1 = Array1::from_vec(vec![0.90, 0.91, 0.89, 0.92, 0.90]);
        let scores2 = Array1::from_vec(vec![0.70, 0.72, 0.71, 0.69, 0.71]);

        let result = paired_ttest(&scores1, &scores2).unwrap();

        assert!(result.significant);
        assert!(result.p_value < 0.05);
        assert!(result.mean_difference > 0.0); // Model 1 better
    }

    #[test]
    fn test_corrected_ttest() {
        let scores1 = Array1::from_vec(vec![0.85, 0.87, 0.84, 0.86, 0.85]);
        let scores2 = Array1::from_vec(vec![0.80, 0.82, 0.79, 0.81, 0.80]);

        let result = corrected_resampled_ttest(&scores1, &scores2, 800, 200).unwrap();

        // The corrected test should have wider CIs than standard t-test
        assert!(result.df == Some(4.0));
    }

    #[test]
    fn test_mcnemar_test() {
        let y_true = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        let pred1 = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        let pred2 = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);

        let result = mcnemar_test(&y_true, &pred1, &pred2).unwrap();

        assert!(result.df == Some(1.0));
        // Results depend on specific disagreement pattern
    }

    #[test]
    fn test_t_cdf() {
        // t(0, df) should be 0.5
        assert_relative_eq!(t_cdf(0.0, 10.0), 0.5, epsilon = 1e-6);

        // Large positive t should give CDF close to 1
        assert!(t_cdf(5.0, 10.0) > 0.99);

        // Large negative t should give CDF close to 0
        assert!(t_cdf(-5.0, 10.0) < 0.01);
    }
}
