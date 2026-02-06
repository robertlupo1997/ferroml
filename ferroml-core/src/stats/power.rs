//! Statistical power analysis
//! Placeholder for power calculations

use serde::{Deserialize, Serialize};

/// Power analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    /// Statistical power (1 - beta)
    pub power: f64,
    /// Sample size
    pub n: usize,
    /// Effect size
    pub effect_size: f64,
    /// Alpha level
    pub alpha: f64,
}

/// Calculate required sample size for desired power
pub fn sample_size_for_power(effect_size: f64, alpha: f64, power: f64, _test_type: &str) -> usize {
    // Simplified calculation for two-sample t-test
    let z_alpha = z_critical(1.0 - alpha / 2.0);
    let z_beta = z_critical(power);

    let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
    n.ceil() as usize
}

/// Calculate power for given sample size
pub fn power_for_sample_size(n: usize, effect_size: f64, alpha: f64) -> f64 {
    let z_alpha = z_critical(1.0 - alpha / 2.0);
    let z = effect_size.mul_add((n as f64 / 2.0).sqrt(), -z_alpha);
    normal_cdf(z)
}

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
    let z = t
        - (c2 * t).mul_add(t, c0 + c1 * t)
            / (d3 * t * t).mul_add(t, (d2 * t).mul_add(t, 1.0 + d1 * t));
    if p > 0.5 {
        -z
    } else {
        z
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

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
    let y = ((a5 * t + a4).mul_add(t, a3).mul_add(t, a2).mul_add(t, a1) * t)
        .mul_add(-(-x * x).exp(), 1.0);
    sign * y
}
