//! Statistical foundations for FerroML
//!
//! This module provides rigorous statistical tools that go beyond what's typically
//! available in AutoML libraries. Every test includes effect sizes, confidence
//! intervals, and power analysis.
//!
//! ## Philosophy
//!
//! - **p-values are not enough**: We always report effect sizes and confidence intervals
//! - **Assumptions matter**: Tests check their own assumptions before running
//! - **Multiple testing**: Built-in correction methods to avoid false discoveries
//! - **Reproducibility**: All randomness is controllable via seeds

pub mod bootstrap;
pub mod confidence;
pub mod diagnostics;
pub mod distributions;
pub mod effect_size;
pub mod hypothesis;
pub mod multiple_testing;
pub mod power;

// Re-exports
pub use bootstrap::{Bootstrap, BootstrapResult};
pub use confidence::{confidence_interval, ConfidenceInterval};
pub use diagnostics::{NormalityTest, ResidualDiagnostics};
pub use effect_size::{CohensD, EffectSize, GlasssDelta, HedgesG};
pub use hypothesis::{HypothesisTest, TestResult, TwoSampleTest};
pub use multiple_testing::{adjust_pvalues, MultipleTestingCorrection};

use crate::{FerroError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Comprehensive statistical result that includes everything needed for rigorous inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalResult {
    /// The test statistic value
    pub statistic: f64,
    /// p-value (two-sided unless otherwise noted)
    pub p_value: f64,
    /// Effect size with interpretation
    pub effect_size: Option<EffectSizeResult>,
    /// Confidence interval for the parameter of interest
    pub confidence_interval: Option<(f64, f64)>,
    /// Confidence level used
    pub confidence_level: f64,
    /// Degrees of freedom (if applicable)
    pub df: Option<f64>,
    /// Sample size
    pub n: usize,
    /// Statistical power (if calculable)
    pub power: Option<f64>,
    /// Whether assumptions were tested
    pub assumptions_checked: bool,
    /// Results of assumption tests
    pub assumption_results: Vec<AssumptionTest>,
    /// Name of the test
    pub test_name: String,
    /// Alternative hypothesis description
    pub alternative: String,
}

impl StatisticalResult {
    /// Check if result is statistically significant at given alpha level
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Get practical significance based on effect size
    pub fn practical_significance(&self) -> Option<&str> {
        self.effect_size
            .as_ref()
            .map(|es| es.interpretation.as_str())
    }

    /// Format as a human-readable summary
    pub fn summary(&self) -> String {
        let mut s = format!("{}\n{}\n", self.test_name, "=".repeat(self.test_name.len()));
        s.push_str(&format!("Test statistic: {:.4}\n", self.statistic));
        s.push_str(&format!("p-value: {:.4}\n", self.p_value));

        if let Some(df) = self.df {
            s.push_str(&format!("Degrees of freedom: {:.1}\n", df));
        }

        if let Some(ref es) = self.effect_size {
            s.push_str(&format!(
                "Effect size ({}): {:.4} ({})\n",
                es.name, es.value, es.interpretation
            ));
        }

        if let Some((lower, upper)) = self.confidence_interval {
            s.push_str(&format!(
                "{}% CI: [{:.4}, {:.4}]\n",
                (self.confidence_level * 100.0) as i32,
                lower,
                upper
            ));
        }

        if let Some(power) = self.power {
            s.push_str(&format!("Statistical power: {:.2}%\n", power * 100.0));
        }

        s
    }
}

/// Result of an effect size calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeResult {
    /// Name of the effect size measure
    pub name: String,
    /// The effect size value
    pub value: f64,
    /// Confidence interval for effect size
    pub ci: Option<(f64, f64)>,
    /// Interpretation (e.g., "small", "medium", "large")
    pub interpretation: String,
}

/// Result of an assumption test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionTest {
    /// Name of the assumption
    pub assumption: String,
    /// Name of the test used
    pub test_name: String,
    /// Whether the assumption is met
    pub passed: bool,
    /// p-value of the assumption test
    pub p_value: f64,
    /// Additional details
    pub details: String,
}

/// Descriptive statistics for a sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    /// Sample size
    pub n: usize,
    /// Mean
    pub mean: f64,
    /// Standard deviation (sample)
    pub std: f64,
    /// Standard error of the mean
    pub sem: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Interquartile range
    pub iqr: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis (excess)
    pub kurtosis: f64,
}

impl DescriptiveStats {
    /// Compute descriptive statistics for a sample
    pub fn compute(data: &Array1<f64>) -> Result<Self> {
        let n = data.len();
        if n == 0 {
            return Err(FerroError::invalid_input("Empty data array"));
        }

        let mut sorted: Vec<f64> = data.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = data.mean().unwrap_or(0.0);
        let variance = data.var(1.0); // ddof=1 for sample variance
        let std = variance.sqrt();
        let sem = std / (n as f64).sqrt();

        let min = sorted[0];
        let max = sorted[n - 1];
        let median = percentile(&sorted, 0.5);
        let q1 = percentile(&sorted, 0.25);
        let q3 = percentile(&sorted, 0.75);
        let iqr = q3 - q1;

        // Skewness and kurtosis
        let m3: f64 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n as f64;
        let m4: f64 = data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n as f64;
        let skewness = m3;
        let kurtosis = m4 - 3.0; // Excess kurtosis

        Ok(Self {
            n,
            mean,
            std,
            sem,
            min,
            max,
            median,
            q1,
            q3,
            iqr,
            skewness,
            kurtosis,
        })
    }
}

/// Compute percentile from sorted data
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }

    let index = p * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let weight = index - lower as f64;

    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// Correlation coefficient with confidence interval and significance test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    /// Pearson correlation coefficient
    pub r: f64,
    /// p-value for H0: r = 0
    pub p_value: f64,
    /// Confidence interval for r
    pub ci: (f64, f64),
    /// Confidence level
    pub confidence_level: f64,
    /// Sample size
    pub n: usize,
    /// R-squared (coefficient of determination)
    pub r_squared: f64,
}

/// Compute Pearson correlation with full statistical analysis
pub fn correlation(x: &Array1<f64>, y: &Array1<f64>, confidence: f64) -> Result<CorrelationResult> {
    if x.len() != y.len() {
        return Err(FerroError::shape_mismatch(
            format!("x length {}", x.len()),
            format!("y length {}", y.len()),
        ));
    }

    let n = x.len();
    if n < 3 {
        return Err(FerroError::invalid_input(
            "Need at least 3 observations for correlation",
        ));
    }

    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let r = sum_xy / (sum_x2.sqrt() * sum_y2.sqrt());

    // Fisher's z transformation for CI
    let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
    let se_z = 1.0 / ((n - 3) as f64).sqrt();

    // Get z critical value for confidence level
    let alpha = 1.0 - confidence;
    let z_crit = z_critical(1.0 - alpha / 2.0);

    let z_lower = z - z_crit * se_z;
    let z_upper = z + z_crit * se_z;

    // Transform back to r
    let r_lower = (z_lower.exp() - (-z_lower).exp()) / (z_lower.exp() + (-z_lower).exp());
    let r_upper = (z_upper.exp() - (-z_upper).exp()) / (z_upper.exp() + (-z_upper).exp());

    // t-test for significance
    let t = r * ((n - 2) as f64).sqrt() / (1.0 - r * r).sqrt();
    let df = (n - 2) as f64;
    let p_value = 2.0 * (1.0 - t_cdf(t.abs(), df));

    Ok(CorrelationResult {
        r,
        p_value,
        ci: (r_lower, r_upper),
        confidence_level: confidence,
        n,
        r_squared: r * r,
    })
}

/// Critical value from standard normal distribution
fn z_critical(p: f64) -> f64 {
    // Approximation using rational function
    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }

    let p = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * p.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p > 0.5 {
        -z
    } else {
        z
    }
}

/// Student's t CDF approximation
fn t_cdf(t: f64, df: f64) -> f64 {
    // Use regularized incomplete beta function
    let x = df / (df + t * t);
    0.5 + 0.5 * (1.0 - incomplete_beta(df / 2.0, 0.5, x)).copysign(t)
}

/// Incomplete beta function approximation
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    // Continued fraction approximation
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
    fn test_descriptive_stats() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = DescriptiveStats::compute(&data).unwrap();

        assert_eq!(stats.n, 5);
        assert_relative_eq!(stats.mean, 3.0, epsilon = 1e-10);
        assert_relative_eq!(stats.median, 3.0, epsilon = 1e-10);
        assert_relative_eq!(stats.min, 1.0, epsilon = 1e-10);
        assert_relative_eq!(stats.max, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let result = correlation(&x, &y, 0.95).unwrap();
        assert_relative_eq!(result.r, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.r_squared, 1.0, epsilon = 1e-10);
    }
}
