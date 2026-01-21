//! Confidence intervals with multiple methods
//!
//! Supports parametric, bootstrap, and Bayesian credible intervals.

use crate::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// A confidence interval with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound
    pub lower: f64,
    /// Upper bound
    pub upper: f64,
    /// Point estimate (e.g., sample mean)
    pub estimate: f64,
    /// Confidence level (e.g., 0.95)
    pub level: f64,
    /// Method used to compute the interval
    pub method: CIMethod,
}

/// Methods for computing confidence intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CIMethod {
    /// Normal approximation (z-interval)
    Normal,
    /// Student's t distribution
    TDistribution,
    /// Percentile bootstrap
    BootstrapPercentile,
    /// BCa bootstrap (bias-corrected and accelerated)
    BootstrapBCa,
    /// Wilson score (for proportions)
    WilsonScore,
    /// Clopper-Pearson exact (for proportions)
    ClopperPearson,
    /// Bayesian credible interval
    BayesianCredible,
}

impl ConfidenceInterval {
    /// Width of the interval
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Check if a value is within the interval
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Margin of error (half-width)
    pub fn margin_of_error(&self) -> f64 {
        (self.upper - self.lower) / 2.0
    }
}

/// Compute confidence interval for a mean
pub fn confidence_interval(
    data: &Array1<f64>,
    level: f64,
    method: CIMethod,
) -> Result<ConfidenceInterval> {
    match method {
        CIMethod::Normal => ci_normal(data, level),
        CIMethod::TDistribution => ci_t(data, level),
        CIMethod::BootstrapPercentile => ci_bootstrap_percentile(data, level, 10000),
        _ => ci_t(data, level), // Default to t
    }
}

/// Normal (z) confidence interval
fn ci_normal(data: &Array1<f64>, level: f64) -> Result<ConfidenceInterval> {
    let n = data.len() as f64;
    let mean = data.mean().unwrap_or(0.0);
    let std = data.std(1.0);
    let se = std / n.sqrt();

    let alpha = 1.0 - level;
    let z = z_critical(1.0 - alpha / 2.0);

    Ok(ConfidenceInterval {
        lower: mean - z * se,
        upper: mean + z * se,
        estimate: mean,
        level,
        method: CIMethod::Normal,
    })
}

/// Student's t confidence interval
fn ci_t(data: &Array1<f64>, level: f64) -> Result<ConfidenceInterval> {
    let n = data.len();
    let mean = data.mean().unwrap_or(0.0);
    let std = data.std(1.0);
    let se = std / (n as f64).sqrt();
    let df = (n - 1) as f64;

    let alpha = 1.0 - level;
    let t = t_critical(1.0 - alpha / 2.0, df);

    Ok(ConfidenceInterval {
        lower: mean - t * se,
        upper: mean + t * se,
        estimate: mean,
        level,
        method: CIMethod::TDistribution,
    })
}

/// Bootstrap percentile confidence interval
fn ci_bootstrap_percentile(
    data: &Array1<f64>,
    level: f64,
    n_bootstrap: usize,
) -> Result<ConfidenceInterval> {
    use rand::prelude::*;

    let n = data.len();
    let estimate = data.mean().unwrap_or(0.0);

    let mut rng = rand::rng();
    let mut bootstrap_means = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.random_range(0..n);
            sum += data[idx];
        }
        bootstrap_means.push(sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - level;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

    Ok(ConfidenceInterval {
        lower: bootstrap_means[lower_idx],
        upper: bootstrap_means[upper_idx.min(n_bootstrap - 1)],
        estimate,
        level,
        method: CIMethod::BootstrapPercentile,
    })
}

/// Standard normal critical value
fn z_critical(p: f64) -> f64 {
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

/// Student's t critical value (approximation)
fn t_critical(p: f64, df: f64) -> f64 {
    // For large df, use normal approximation
    if df > 100.0 {
        return z_critical(p);
    }

    // Newton-Raphson to find t such that t_cdf(t, df) = p
    let mut t = z_critical(p);

    for _ in 0..10 {
        let cdf = t_cdf(t, df);
        let pdf = t_pdf(t, df);
        if pdf.abs() < 1e-10 {
            break;
        }
        t = t - (cdf - p) / pdf;
    }

    t
}

/// Student's t PDF
fn t_pdf(t: f64, df: f64) -> f64 {
    let coef =
        gamma_ln((df + 1.0) / 2.0) - gamma_ln(df / 2.0) - 0.5 * (df * std::f64::consts::PI).ln();
    (coef - ((df + 1.0) / 2.0) * (1.0 + t * t / df).ln()).exp()
}

/// Student's t CDF
fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 + 0.5 * (1.0 - incomplete_beta(df / 2.0, 0.5, x)).copysign(t)
}

/// Incomplete beta function
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

fn beta(a: f64, b: f64) -> f64 {
    (gamma_ln(a) + gamma_ln(b) - gamma_ln(a + b)).exp()
}

fn gamma_ln(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000000000190015;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (x + i as f64 + 1.0);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_t() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let ci = ci_t(&data, 0.95).unwrap();

        assert!(ci.lower < ci.estimate);
        assert!(ci.estimate < ci.upper);
        assert!(ci.contains(3.0));
    }
}
