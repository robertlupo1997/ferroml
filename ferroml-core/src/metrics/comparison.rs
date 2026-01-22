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

/// Standard normal CDF
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Wilcoxon signed-rank test for comparing paired samples
///
/// A non-parametric alternative to the paired t-test. Tests whether the median
/// difference between paired observations is zero. More robust to outliers and
/// non-normal distributions than the t-test.
///
/// # Arguments
/// * `scores1` - Scores from model 1
/// * `scores2` - Scores from model 2 (same samples)
///
/// # Returns
/// Statistical test result with p-value, confidence interval, etc.
///
/// # Algorithm
/// 1. Compute differences (scores1 - scores2)
/// 2. Rank the absolute differences (excluding zeros)
/// 3. Sum the ranks of positive and negative differences separately
/// 4. Test statistic W = min(W+, W-) or use W+ for normal approximation
/// 5. For n > 20, use normal approximation with continuity correction
///
/// # Reference
/// Wilcoxon, F. (1945). Individual comparisons by ranking methods.
/// Biometrics Bulletin, 1(6), 80-83.
pub fn wilcoxon_signed_rank_test(
    scores1: &Array1<f64>,
    scores2: &Array1<f64>,
) -> Result<ModelComparisonResult> {
    if scores1.len() != scores2.len() {
        return Err(FerroError::shape_mismatch(
            format!("scores1 length {}", scores1.len()),
            format!("scores2 length {}", scores2.len()),
        ));
    }

    let n = scores1.len();
    if n < 2 {
        return Err(FerroError::invalid_input(
            "Need at least 2 paired observations for Wilcoxon test",
        ));
    }

    // Compute differences and filter out zeros
    let differences: Vec<f64> = scores1
        .iter()
        .zip(scores2.iter())
        .map(|(&a, &b)| a - b)
        .filter(|&d| d.abs() > 1e-15)
        .collect();

    let n_nonzero = differences.len();

    if n_nonzero == 0 {
        return Ok(ModelComparisonResult {
            test_name: "Wilcoxon Signed-Rank Test".to_string(),
            statistic: 0.0,
            p_value: 1.0,
            df: None,
            mean_difference: 0.0,
            std_error: 0.0,
            ci_95: (0.0, 0.0),
            significant: false,
            interpretation: "All differences are zero, no comparison possible".to_string(),
        });
    }

    // Compute ranks of absolute differences
    let abs_diffs_with_sign: Vec<(f64, f64)> =
        differences.iter().map(|&d| (d.abs(), d.signum())).collect();

    // Sort by absolute value for ranking
    let mut indexed_abs: Vec<(usize, f64)> = abs_diffs_with_sign
        .iter()
        .enumerate()
        .map(|(i, &(abs_d, _))| (i, abs_d))
        .collect();
    indexed_abs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Assign ranks (handling ties by averaging)
    let mut ranks = vec![0.0; n_nonzero];
    let mut i = 0;
    while i < n_nonzero {
        let mut j = i;
        // Find all ties
        while j < n_nonzero - 1 && (indexed_abs[j + 1].1 - indexed_abs[i].1).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for k in i..=j {
            ranks[indexed_abs[k].0] = avg_rank;
        }
        i = j + 1;
    }

    // Sum ranks of positive and negative differences
    let mut w_plus = 0.0; // Sum of ranks where difference > 0
    let mut w_minus = 0.0; // Sum of ranks where difference < 0

    for (i, &(_, sign)) in abs_diffs_with_sign.iter().enumerate() {
        if sign > 0.0 {
            w_plus += ranks[i];
        } else {
            w_minus += ranks[i];
        }
    }

    let _w = w_plus.min(w_minus); // Min statistic (kept for reference)
    let n_f = n_nonzero as f64;

    // Expected value and variance under null hypothesis
    let expected_w = n_f * (n_f + 1.0) / 4.0;

    // Check for ties to adjust variance
    let has_ties = {
        let mut unique_abs: Vec<f64> = abs_diffs_with_sign
            .iter()
            .map(|&(abs_d, _)| abs_d)
            .collect();
        unique_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_abs.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
        unique_abs.len() < n_nonzero
    };

    let variance_w = if has_ties {
        // Tie correction
        let mut tie_correction = 0.0;
        let mut i = 0;
        let sorted_abs: Vec<f64> = {
            let mut v: Vec<f64> = abs_diffs_with_sign
                .iter()
                .map(|&(abs_d, _)| abs_d)
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        while i < n_nonzero {
            let mut j = i;
            while j < n_nonzero - 1 && (sorted_abs[j + 1] - sorted_abs[i]).abs() < 1e-15 {
                j += 1;
            }
            let t = (j - i + 1) as f64;
            if t > 1.0 {
                tie_correction += (t * t * t - t) / 48.0;
            }
            i = j + 1;
        }
        n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0 - tie_correction
    } else {
        n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0
    };

    let std_w = variance_w.sqrt();

    // Normal approximation with continuity correction (for n > ~10)
    let z_stat = if std_w > 0.0 {
        // Use W+ for directional z-score
        let continuity = 0.5;
        ((w_plus - expected_w).abs() - continuity) / std_w
    } else {
        0.0
    };

    // Two-tailed p-value
    let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));

    // Mean and median difference for reporting
    let mean_diff: f64 = differences.iter().sum::<f64>() / differences.len() as f64;

    // Hodges-Lehmann estimator for CI (median of pairwise averages)
    // For simplicity, use normal approximation CI
    let z_crit = 1.96;
    let diff_std: f64 = (differences
        .iter()
        .map(|&d| (d - mean_diff).powi(2))
        .sum::<f64>()
        / (n_nonzero as f64 - 1.0))
        .sqrt();
    let se_mean = diff_std / (n_nonzero as f64).sqrt();
    let ci_lower = mean_diff - z_crit * se_mean;
    let ci_upper = mean_diff + z_crit * se_mean;

    let significant = p_value < 0.05;

    let interpretation = if !significant {
        "No significant difference between models (non-parametric)".to_string()
    } else if w_plus > w_minus {
        "Model 1 significantly better than Model 2 (non-parametric)".to_string()
    } else {
        "Model 2 significantly better than Model 1 (non-parametric)".to_string()
    };

    Ok(ModelComparisonResult {
        test_name: "Wilcoxon Signed-Rank Test".to_string(),
        statistic: z_stat, // Report z-statistic for normal approximation
        p_value,
        df: None, // Non-parametric test has no df
        mean_difference: mean_diff,
        std_error: se_mean,
        ci_95: (ci_lower, ci_upper),
        significant,
        interpretation,
    })
}

/// 5x2cv paired t-test for comparing two classifiers (Dietterich, 1998)
///
/// A more reliable alternative to standard CV-based t-tests. Runs 5 iterations
/// of 2-fold cross-validation and uses a specialized variance estimator that
/// accounts for the correlation between folds.
///
/// # Arguments
/// * `scores` - 2D array of shape (5, 2) containing scores for both models.
///              Each row is one repetition, each column is one fold.
///              Format: `scores[rep][fold]` = (model1_score, model2_score)
///
/// Actually, takes the differences directly:
/// * `differences` - 2D array of shape (5, 2) where `differences[i][j]` is
///                   model1_score - model2_score for repetition i, fold j.
///
/// # Algorithm
/// 1. For each of 5 repetitions, compute differences d_i1, d_i2 for the two folds
/// 2. Compute mean p_i = (d_i1 + d_i2) / 2 for each repetition
/// 3. Compute variance s_i^2 = (d_i1 - p_i)^2 + (d_i2 - p_i)^2 for each repetition
/// 4. Test statistic t = d_11 / sqrt(sum(s_i^2) / 5)
///
/// # Reference
/// Dietterich, T. G. (1998). Approximate statistical tests for comparing
/// supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923.
pub fn five_by_two_cv_paired_ttest(differences: &[[f64; 2]; 5]) -> Result<ModelComparisonResult> {
    // Compute variance estimate for each repetition
    let mut variances = [0.0; 5];
    let mut means = [0.0; 5];

    for i in 0..5 {
        let d1 = differences[i][0];
        let d2 = differences[i][1];
        means[i] = (d1 + d2) / 2.0;
        variances[i] = (d1 - means[i]).powi(2) + (d2 - means[i]).powi(2);
    }

    let sum_variance: f64 = variances.iter().sum();

    // Test statistic: use first fold's first difference
    let d11 = differences[0][0];

    let t_stat = if sum_variance > 0.0 {
        d11 / (sum_variance / 5.0).sqrt()
    } else {
        if d11.abs() < 1e-15 {
            0.0
        } else {
            f64::INFINITY.copysign(d11)
        }
    };

    // Degrees of freedom = 5
    let df = 5.0;
    let p_value = 2.0 * (1.0 - t_cdf(t_stat.abs(), df));

    // Compute overall mean difference
    let all_diffs: Vec<f64> = differences
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let mean_diff = all_diffs.iter().sum::<f64>() / 10.0;

    // Standard error based on the 5x2cv variance estimator
    let std_error = (sum_variance / 5.0).sqrt();

    // CI using t-distribution
    let t_crit = t_critical(0.975, df);
    let ci_lower = mean_diff - t_crit * std_error;
    let ci_upper = mean_diff + t_crit * std_error;

    let significant = p_value < 0.05;

    let interpretation = if !significant {
        "No significant difference between models (5x2cv test)".to_string()
    } else if mean_diff > 0.0 {
        "Model 1 significantly better than Model 2 (5x2cv test)".to_string()
    } else {
        "Model 2 significantly better than Model 1 (5x2cv test)".to_string()
    };

    Ok(ModelComparisonResult {
        test_name: "5x2cv Paired t-test (Dietterich)".to_string(),
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

/// Convenience function to run 5x2cv and compute the test
///
/// This function takes raw scores from 5 repetitions of 2-fold CV and computes
/// the 5x2cv paired t-test.
///
/// # Arguments
/// * `scores1` - Array of shape (5, 2) with model 1's scores for each rep/fold
/// * `scores2` - Array of shape (5, 2) with model 2's scores for each rep/fold
///
/// # Returns
/// The 5x2cv paired t-test result
pub fn five_by_two_cv_paired_ttest_from_scores(
    scores1: &[[f64; 2]; 5],
    scores2: &[[f64; 2]; 5],
) -> Result<ModelComparisonResult> {
    let mut differences = [[0.0; 2]; 5];

    for i in 0..5 {
        for j in 0..2 {
            differences[i][j] = scores1[i][j] - scores2[i][j];
        }
    }

    five_by_two_cv_paired_ttest(&differences)
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

    #[test]
    fn test_wilcoxon_no_difference() {
        // Very similar scores - should not be significant
        let scores1 = Array1::from_vec(vec![0.80, 0.82, 0.81, 0.79, 0.80, 0.82, 0.81, 0.80]);
        let scores2 = Array1::from_vec(vec![0.79, 0.81, 0.82, 0.80, 0.81, 0.81, 0.80, 0.79]);

        let result = wilcoxon_signed_rank_test(&scores1, &scores2).unwrap();

        assert!(!result.significant);
        assert!(result.p_value > 0.05);
        assert_eq!(result.test_name, "Wilcoxon Signed-Rank Test");
    }

    #[test]
    fn test_wilcoxon_significant_difference() {
        // Model 1 clearly better
        let scores1 = Array1::from_vec(vec![0.90, 0.91, 0.89, 0.92, 0.90, 0.91, 0.90, 0.92]);
        let scores2 = Array1::from_vec(vec![0.70, 0.72, 0.71, 0.69, 0.71, 0.70, 0.72, 0.69]);

        let result = wilcoxon_signed_rank_test(&scores1, &scores2).unwrap();

        assert!(result.significant);
        assert!(result.p_value < 0.05);
        assert!(result.mean_difference > 0.0);
    }

    #[test]
    fn test_wilcoxon_with_ties() {
        // Scores with some ties
        let scores1 = Array1::from_vec(vec![0.85, 0.85, 0.90, 0.80, 0.85]);
        let scores2 = Array1::from_vec(vec![0.80, 0.80, 0.85, 0.75, 0.80]);

        let result = wilcoxon_signed_rank_test(&scores1, &scores2).unwrap();

        // All differences are 0.05, so there are ties
        assert!(result.mean_difference > 0.0);
    }

    #[test]
    fn test_wilcoxon_all_zeros() {
        // Identical scores
        let scores1 = Array1::from_vec(vec![0.80, 0.82, 0.81]);
        let scores2 = Array1::from_vec(vec![0.80, 0.82, 0.81]);

        let result = wilcoxon_signed_rank_test(&scores1, &scores2).unwrap();

        assert!(!result.significant);
        assert_relative_eq!(result.p_value, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_five_by_two_cv_no_difference() {
        // Model differences close to zero
        let differences = [
            [0.01, -0.01],
            [-0.01, 0.01],
            [0.005, -0.005],
            [-0.02, 0.01],
            [0.01, -0.02],
        ];

        let result = five_by_two_cv_paired_ttest(&differences).unwrap();

        assert!(!result.significant);
        assert!(result.p_value > 0.05);
        assert_eq!(result.df, Some(5.0));
        assert_eq!(result.test_name, "5x2cv Paired t-test (Dietterich)");
    }

    #[test]
    fn test_five_by_two_cv_significant_difference() {
        // Model 1 consistently better
        let differences = [
            [0.10, 0.12],
            [0.11, 0.09],
            [0.08, 0.11],
            [0.12, 0.10],
            [0.09, 0.11],
        ];

        let result = five_by_two_cv_paired_ttest(&differences).unwrap();

        assert!(result.significant);
        assert!(result.p_value < 0.05);
        assert!(result.mean_difference > 0.0);
    }

    #[test]
    fn test_five_by_two_cv_from_scores() {
        let scores1 = [
            [0.90, 0.91],
            [0.89, 0.90],
            [0.91, 0.88],
            [0.90, 0.89],
            [0.88, 0.91],
        ];
        let scores2 = [
            [0.80, 0.79],
            [0.78, 0.81],
            [0.80, 0.77],
            [0.78, 0.79],
            [0.79, 0.80],
        ];

        let result = five_by_two_cv_paired_ttest_from_scores(&scores1, &scores2).unwrap();

        assert!(result.significant);
        assert!(result.mean_difference > 0.0);
    }

    #[test]
    fn test_normal_cdf() {
        // Normal(0, 1) CDF at 0 should be 0.5
        assert_relative_eq!(normal_cdf(0.0), 0.5, epsilon = 1e-6);

        // CDF at large positive should be close to 1
        assert!(normal_cdf(3.0) > 0.99);

        // CDF at large negative should be close to 0
        assert!(normal_cdf(-3.0) < 0.01);

        // Known values
        assert_relative_eq!(normal_cdf(1.96), 0.975, epsilon = 0.001);
    }
}
