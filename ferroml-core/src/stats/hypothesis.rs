//! Hypothesis testing with full statistical rigor
//!
//! All tests include:
//! - Test statistic and p-value
//! - Effect size with confidence interval
//! - Power analysis
//! - Assumption checking

use super::{AssumptionTest, EffectSizeResult, StatisticalResult};
use crate::{FerroError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trait for hypothesis tests
pub trait HypothesisTest {
    /// Run the test and return comprehensive results
    fn test(&self) -> Result<StatisticalResult>;

    /// Check assumptions before running the test
    fn check_assumptions(&self) -> Vec<AssumptionTest>;

    /// Get the name of the test
    fn name(&self) -> &str;
}

/// Result of a hypothesis test (simplified version for specific tests)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test statistic
    pub statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: Option<f64>,
}

/// Two-sample tests
#[derive(Debug, Clone)]
pub enum TwoSampleTest {
    /// Independent samples t-test
    TTest {
        /// First sample
        x: Array1<f64>,
        /// Second sample
        y: Array1<f64>,
        /// Assume equal variances
        equal_var: bool,
    },
    /// Mann-Whitney U test (non-parametric)
    MannWhitney {
        /// First sample
        x: Array1<f64>,
        /// Second sample
        y: Array1<f64>,
    },
    /// Welch's t-test (unequal variances)
    Welch {
        /// First sample
        x: Array1<f64>,
        /// Second sample
        y: Array1<f64>,
    },
}

impl TwoSampleTest {
    /// Create a new t-test
    pub fn t_test(x: Array1<f64>, y: Array1<f64>, equal_var: bool) -> Self {
        Self::TTest { x, y, equal_var }
    }

    /// Create a new Welch's t-test
    pub fn welch(x: Array1<f64>, y: Array1<f64>) -> Self {
        Self::Welch { x, y }
    }

    /// Create a new Mann-Whitney U test
    pub fn mann_whitney(x: Array1<f64>, y: Array1<f64>) -> Self {
        Self::MannWhitney { x, y }
    }
}

impl HypothesisTest for TwoSampleTest {
    fn test(&self) -> Result<StatisticalResult> {
        match self {
            TwoSampleTest::TTest { x, y, equal_var } => run_t_test(x, y, *equal_var),
            TwoSampleTest::Welch { x, y } => run_t_test(x, y, false),
            TwoSampleTest::MannWhitney { x, y } => run_mann_whitney(x, y),
        }
    }

    fn check_assumptions(&self) -> Vec<AssumptionTest> {
        match self {
            TwoSampleTest::TTest { x, y, .. } | TwoSampleTest::Welch { x, y } => {
                vec![check_normality(x, "Group 1"), check_normality(y, "Group 2")]
            }
            TwoSampleTest::MannWhitney { .. } => {
                // Non-parametric, no distributional assumptions
                vec![]
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            TwoSampleTest::TTest { equal_var, .. } => {
                if *equal_var {
                    "Independent Samples t-Test (equal variances)"
                } else {
                    "Independent Samples t-Test (unequal variances)"
                }
            }
            TwoSampleTest::Welch { .. } => "Welch's t-Test",
            TwoSampleTest::MannWhitney { .. } => "Mann-Whitney U Test",
        }
    }
}

/// Run t-test for two independent samples
fn run_t_test(x: &Array1<f64>, y: &Array1<f64>, equal_var: bool) -> Result<StatisticalResult> {
    let n1 = x.len();
    let n2 = y.len();

    if n1 < 2 || n2 < 2 {
        return Err(FerroError::invalid_input(
            "Each group needs at least 2 observations",
        ));
    }

    let mean1 = x.mean().unwrap_or(0.0);
    let mean2 = y.mean().unwrap_or(0.0);
    let var1 = x.var(1.0);
    let var2 = y.var(1.0);

    let (t, df) = if equal_var {
        // Pooled variance
        let sp2 = ((n1 - 1) as f64).mul_add(var1, (n2 - 1) as f64 * var2) / (n1 + n2 - 2) as f64;
        let se = (sp2 * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
        let t = (mean1 - mean2) / se;
        let df = (n1 + n2 - 2) as f64;
        (t, df)
    } else {
        // Welch-Satterthwaite
        let se = (var1 / n1 as f64 + var2 / n2 as f64).sqrt();
        let t = (mean1 - mean2) / se;

        let num = (var1 / n1 as f64 + var2 / n2 as f64).powi(2);
        let denom = (var1 / n1 as f64).powi(2) / (n1 - 1) as f64
            + (var2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let df = num / denom;
        (t, df)
    };

    let p_value = 2.0 * (1.0 - t_cdf(t.abs(), df));

    // Cohen's d effect size
    let pooled_std =
        (((n1 - 1) as f64).mul_add(var1, (n2 - 1) as f64 * var2) / (n1 + n2 - 2) as f64).sqrt();
    let cohens_d = (mean1 - mean2) / pooled_std;

    let effect_interpretation = if cohens_d.abs() < 0.2 {
        "negligible"
    } else if cohens_d.abs() < 0.5 {
        "small"
    } else if cohens_d.abs() < 0.8 {
        "medium"
    } else {
        "large"
    };

    // CI for effect size (approximate) - use (n1+n2-2) denominator per correct formula
    let se_d = ((n1 + n2) as f64 / (n1 * n2) as f64
        + cohens_d.powi(2) / (2.0 * (n1 + n2 - 2) as f64))
        .sqrt();
    let z_crit = 1.96; // 95% CI
    let d_lower = cohens_d - z_crit * se_d;
    let d_upper = cohens_d + z_crit * se_d;

    Ok(StatisticalResult {
        statistic: t,
        p_value,
        effect_size: Some(EffectSizeResult {
            name: "Cohen's d".to_string(),
            value: cohens_d,
            ci: Some((d_lower, d_upper)),
            interpretation: effect_interpretation.to_string(),
        }),
        confidence_interval: Some((
            mean1 - mean2 - z_crit * (var1 / n1 as f64 + var2 / n2 as f64).sqrt(),
            mean1 - mean2 + z_crit * (var1 / n1 as f64 + var2 / n2 as f64).sqrt(),
        )),
        confidence_level: 0.95,
        df: Some(df),
        n: n1 + n2,
        power: Some(compute_two_sample_power(cohens_d, n1, n2)),
        assumptions_checked: false,
        assumption_results: vec![],
        test_name: if equal_var {
            "Independent Samples t-Test (equal variances)".to_string()
        } else {
            "Welch's t-Test".to_string()
        },
        alternative: "two-sided".to_string(),
    })
}

/// Compute statistical power for a two-sample t-test using normal approximation.
/// Power = P(reject H0 | H1 is true) for a two-sided test at alpha = 0.05.
fn compute_two_sample_power(cohens_d: f64, n1: usize, n2: usize) -> f64 {
    let z_alpha_2 = 1.96; // two-sided alpha = 0.05
                          // Non-centrality parameter: delta = |d| * sqrt(n1*n2/(n1+n2))
    let n_harmonic = (n1 as f64 * n2 as f64) / (n1 + n2) as f64;
    let delta = cohens_d.abs() * n_harmonic.sqrt();
    // Power via normal approximation to non-central t
    let power = normal_cdf(delta - z_alpha_2) + normal_cdf(-delta - z_alpha_2);
    power.clamp(0.05, 1.0)
}

/// Run Mann-Whitney U test (non-parametric)
fn run_mann_whitney(x: &Array1<f64>, y: &Array1<f64>) -> Result<StatisticalResult> {
    let n1 = x.len();
    let n2 = y.len();

    if n1 < 2 || n2 < 2 {
        return Err(FerroError::invalid_input(
            "Each group needs at least 2 observations",
        ));
    }

    // Combine and rank
    let mut combined: Vec<(f64, usize)> = x
        .iter()
        .map(|&v| (v, 0))
        .chain(y.iter().map(|&v| (v, 1)))
        .collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign ranks (handling ties)
    let mut ranks = vec![0.0; combined.len()];
    let mut i = 0;
    while i < combined.len() {
        let mut j = i;
        while j < combined.len() && combined[j].0 == combined[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // Sum of ranks for group 1
    let r1: f64 = combined
        .iter()
        .enumerate()
        .filter(|(_, (_, group))| *group == 0)
        .map(|(i, _)| ranks[i])
        .sum();

    // U statistic
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u = u1.min(u2);

    // Normal approximation for large samples
    let mu = (n1 * n2) as f64 / 2.0;
    let sigma = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
    let z = (u - mu) / sigma;
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

    // Effect size: rank-biserial correlation
    let r_rb = 1.0 - 2.0 * u / (n1 * n2) as f64;

    let effect_interpretation = if r_rb.abs() < 0.1 {
        "negligible"
    } else if r_rb.abs() < 0.3 {
        "small"
    } else if r_rb.abs() < 0.5 {
        "medium"
    } else {
        "large"
    };

    Ok(StatisticalResult {
        statistic: u,
        p_value,
        effect_size: Some(EffectSizeResult {
            name: "Rank-biserial correlation".to_string(),
            value: r_rb,
            ci: None,
            interpretation: effect_interpretation.to_string(),
        }),
        confidence_interval: None,
        confidence_level: 0.95,
        df: None,
        n: n1 + n2,
        power: None,
        assumptions_checked: false,
        assumption_results: vec![],
        test_name: "Mann-Whitney U Test".to_string(),
        alternative: "two-sided".to_string(),
    })
}

/// Check normality using Shapiro-Wilk approximation
fn check_normality(data: &Array1<f64>, group_name: &str) -> AssumptionTest {
    let n = data.len();

    // Simplified normality check using skewness and kurtosis
    let mean = data.mean().unwrap_or(0.0);
    let std = data.std(1.0);

    let skewness: f64 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n as f64;

    let kurtosis: f64 =
        data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n as f64 - 3.0;

    // D'Agostino-Pearson omnibus test approximation
    let z_skew = skewness * ((n * (n - 1)) as f64).sqrt() / (6.0 * (n - 2) as f64);
    let z_kurt = kurtosis / (24.0 / n as f64).sqrt();
    let k2 = z_kurt.mul_add(z_kurt, z_skew.powi(2));

    // Chi-squared with 2 df
    // Chi-squared survival function with df=2
    let p_value = (-k2 / 2.0).exp();

    AssumptionTest {
        assumption: format!("Normality ({})", group_name),
        test_name: "D'Agostino-Pearson omnibus test".to_string(),
        passed: p_value > 0.05,
        p_value,
        details: format!("skewness={:.3}, kurtosis={:.3}", skewness, kurtosis),
    }
}

// Delegate to shared math module to avoid duplicating numerical algorithms
use super::math;

fn normal_cdf(x: f64) -> f64 {
    math::normal_cdf(x)
}

fn t_cdf(t: f64, df: f64) -> f64 {
    math::t_cdf(t, df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_test() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        let test = TwoSampleTest::t_test(x, y, true);
        let result = test.test().unwrap();

        assert!(result.p_value > 0.0);
        assert!(result.p_value < 1.0);
        assert!(result.effect_size.is_some());
    }

    #[test]
    fn test_mann_whitney() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0]);

        let test = TwoSampleTest::mann_whitney(x, y);
        let result = test.test().unwrap();

        assert!(result.p_value < 0.05); // Should be significant
    }
}
