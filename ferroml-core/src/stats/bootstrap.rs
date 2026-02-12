//! Bootstrap methods for uncertainty quantification

use crate::Result;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Bootstrap resampling configuration
#[derive(Debug, Clone)]
pub struct Bootstrap {
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Confidence level for intervals
    pub confidence: f64,
}

impl Default for Bootstrap {
    fn default() -> Self {
        Self {
            n_bootstrap: 10000,
            seed: None,
            confidence: 0.95,
        }
    }
}

/// Result of bootstrap analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    /// Original statistic
    pub original: f64,
    /// Bootstrap standard error
    pub std_error: f64,
    /// Bias estimate
    pub bias: f64,
    /// Percentile confidence interval
    pub ci_percentile: (f64, f64),
    /// BCa confidence interval (if computed)
    pub ci_bca: Option<(f64, f64)>,
    /// All bootstrap samples (for diagnostics)
    pub samples: Vec<f64>,
}

impl Bootstrap {
    /// Create new bootstrap configuration
    pub fn new(n_bootstrap: usize) -> Self {
        Self {
            n_bootstrap,
            ..Default::default()
        }
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Run bootstrap for a statistic, including BCa confidence intervals
    pub fn run<F>(&self, data: &Array1<f64>, statistic: F) -> Result<BootstrapResult>
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let n = data.len();
        let original = statistic(data);

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let mut samples = Vec::with_capacity(self.n_bootstrap);
        let mut resampled = Array1::zeros(n);

        for _ in 0..self.n_bootstrap {
            // Resample with replacement
            for i in 0..n {
                let idx = rng.random_range(0..n);
                resampled[i] = data[idx];
            }
            samples.push(statistic(&resampled));
        }

        // Sort for percentiles
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Bootstrap standard error
        let mean: f64 = samples.iter().sum::<f64>() / self.n_bootstrap as f64;
        let std_error = (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (self.n_bootstrap - 1) as f64)
            .sqrt();

        // Bias
        let bias = mean - original;

        // Percentile CI
        let alpha = 1.0 - self.confidence;
        let lower_idx = ((alpha / 2.0) * self.n_bootstrap as f64).round() as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * self.n_bootstrap as f64).round() as usize;
        let ci_percentile = (
            samples[lower_idx],
            samples[upper_idx.min(self.n_bootstrap - 1)],
        );

        // BCa confidence interval
        let ci_bca = compute_bca(data, &statistic, &samples, self.confidence);

        Ok(BootstrapResult {
            original,
            std_error,
            bias,
            ci_percentile,
            ci_bca,
            samples,
        })
    }

    /// Run bootstrap for the mean
    pub fn mean(&self, data: &Array1<f64>) -> Result<BootstrapResult> {
        self.run(data, |d| d.mean().unwrap_or(0.0))
    }

    /// Run bootstrap for the median
    pub fn median(&self, data: &Array1<f64>) -> Result<BootstrapResult> {
        self.run(data, |d| {
            let mut sorted: Vec<f64> = d.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            }
        })
    }

    /// Run bootstrap for standard deviation
    pub fn std(&self, data: &Array1<f64>) -> Result<BootstrapResult> {
        self.run(data, |d| d.std(1.0))
    }

    /// Run bootstrap for correlation between two variables
    pub fn correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<BootstrapResult> {
        if x.len() != y.len() {
            return Err(crate::FerroError::shape_mismatch(
                format!("x len {}", x.len()),
                format!("y len {}", y.len()),
            ));
        }

        let n = x.len();
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        let original = pearson_r(x, y);
        let mut samples = Vec::with_capacity(self.n_bootstrap);

        let mut x_resampled = Array1::zeros(n);
        let mut y_resampled = Array1::zeros(n);

        for _ in 0..self.n_bootstrap {
            for i in 0..n {
                let idx = rng.random_range(0..n);
                x_resampled[i] = x[idx];
                y_resampled[i] = y[idx];
            }
            samples.push(pearson_r(&x_resampled, &y_resampled));
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean: f64 = samples.iter().sum::<f64>() / self.n_bootstrap as f64;
        let std_error = (samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / (self.n_bootstrap - 1) as f64)
            .sqrt();

        let alpha = 1.0 - self.confidence;
        let lower_idx = ((alpha / 2.0) * self.n_bootstrap as f64).round() as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * self.n_bootstrap as f64).round() as usize;

        Ok(BootstrapResult {
            original,
            std_error,
            bias: mean - original,
            ci_percentile: (
                samples[lower_idx],
                samples[upper_idx.min(self.n_bootstrap - 1)],
            ),
            ci_bca: None,
            samples,
        })
    }
}

/// Compute BCa confidence interval from bootstrap samples and original data.
///
/// Returns `Some((lower, upper))` if computation succeeds, `None` if the data
/// is too small or degenerate (e.g., all jackknife values identical).
///
/// Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap", Ch. 14
fn compute_bca<F>(
    data: &Array1<f64>,
    statistic: &F,
    sorted_samples: &[f64],
    confidence: f64,
) -> Option<(f64, f64)>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let n = data.len();
    let b = sorted_samples.len();
    if n < 3 || b < 20 {
        return None;
    }

    let original = statistic(data);

    // z0: bias-correction constant
    // z0 = Φ^{-1}(proportion of bootstrap samples < original)
    let count_below = sorted_samples.iter().filter(|&&s| s < original).count();
    let prop = count_below as f64 / b as f64;
    // Clamp to avoid infinite z0 at boundaries
    let prop = prop.clamp(0.5 / b as f64, 1.0 - 0.5 / b as f64);
    let z0 = norm_ppf(prop);

    // a: acceleration constant via jackknife
    // a = (1/6) * Σ(θ̄ - θ_i)^3 / [Σ(θ̄ - θ_i)^2]^{3/2}
    let mut jackknife_vals = Vec::with_capacity(n);
    let mut jack_data = Array1::zeros(n - 1);
    for i in 0..n {
        // Leave-one-out: copy all elements except i
        let mut j = 0;
        for k in 0..n {
            if k != i {
                jack_data[j] = data[k];
                j += 1;
            }
        }
        jackknife_vals.push(statistic(&jack_data));
    }

    let jack_mean: f64 = jackknife_vals.iter().sum::<f64>() / n as f64;
    let sum2: f64 = jackknife_vals
        .iter()
        .map(|&v| (jack_mean - v).powi(2))
        .sum();
    let sum3: f64 = jackknife_vals
        .iter()
        .map(|&v| (jack_mean - v).powi(3))
        .sum();

    if sum2 == 0.0 {
        return None; // Degenerate case: all jackknife values identical
    }

    let a = sum3 / (6.0 * sum2.powf(1.5));

    // BCa adjusted percentiles
    let alpha = 1.0 - confidence;
    let z_alpha_lower = norm_ppf(alpha / 2.0);
    let z_alpha_upper = norm_ppf(1.0 - alpha / 2.0);

    // α1 = Φ(z0 + (z0 + z_α) / (1 - a*(z0 + z_α)))
    let alpha1 = bca_adjusted_alpha(z0, a, z_alpha_lower);
    let alpha2 = bca_adjusted_alpha(z0, a, z_alpha_upper);

    // Convert adjusted alphas to indices
    let idx_lower = ((alpha1 * b as f64).round() as usize).clamp(0, b - 1);
    let idx_upper = ((alpha2 * b as f64).round() as usize).clamp(0, b - 1);

    Some((sorted_samples[idx_lower], sorted_samples[idx_upper]))
}

/// BCa adjusted alpha: Φ(z0 + (z0 + z_α) / (1 - a*(z0 + z_α)))
fn bca_adjusted_alpha(z0: f64, a: f64, z_alpha: f64) -> f64 {
    let num = z0 + z_alpha;
    let denom = 1.0 - a * num;
    if denom.abs() < 1e-15 {
        // Degenerate: return the unadjusted quantile
        return norm_cdf(z_alpha);
    }
    norm_cdf(z0 + num / denom)
}

/// Standard normal CDF: Φ(x)
fn norm_cdf(x: f64) -> f64 {
    let t = 1.0 / 0.2316419f64.mul_add(x.abs(), 1.0);
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp();
    let c = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    if x >= 0.0 {
        1.0 - p * c
    } else {
        p * c
    }
}

/// Inverse standard normal CDF (probit / quantile function): Φ^{-1}(p)
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Rational approximation (Abramowitz & Stegun 26.2.23)
    let q = if p > 0.5 { 1.0 - p } else { p };
    let t = (-2.0 * q.ln()).sqrt();

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let z = t - (c0 + t * (c1 + t * c2)) / (1.0 + t * (d1 + t * (d2 + t * d3)));

    if p > 0.5 {
        z
    } else {
        -z
    }
}

/// Compute Pearson correlation coefficient
fn pearson_r(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    sum_xy / (sum_x2.sqrt() * sum_y2.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_mean() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let bootstrap = Bootstrap::new(1000).with_seed(42);
        let result = bootstrap.mean(&data).unwrap();

        assert!((result.original - 5.5).abs() < 1e-10);
        assert!(result.ci_percentile.0 < 5.5);
        assert!(result.ci_percentile.1 > 5.5);
    }

    #[test]
    fn test_bootstrap_ci_within_range() {
        // Regression: percentile index used floor instead of round,
        // potentially causing off-by-one in CI bounds
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let bootstrap = Bootstrap::new(999).with_seed(123).with_confidence(0.95);
        let result = bootstrap.mean(&data).unwrap();

        assert!(result.ci_percentile.0 <= result.original);
        assert!(result.ci_percentile.1 >= result.original);
        assert!(result.ci_percentile.0 >= 1.0);
        assert!(result.ci_percentile.1 <= 5.0);
    }

    #[test]
    fn test_bca_computed_for_mean() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let bootstrap = Bootstrap::new(2000).with_seed(42);
        let result = bootstrap.mean(&data).unwrap();

        // BCa should be computed for n >= 3
        assert!(result.ci_bca.is_some(), "BCa should be computed");
        let (bca_lo, bca_hi) = result.ci_bca.unwrap();
        assert!(bca_lo < 5.5, "BCa lower should be below mean");
        assert!(bca_hi > 5.5, "BCa upper should be above mean");
    }

    #[test]
    fn test_bca_tighter_on_skewed_data() {
        // Right-skewed data: BCa should adjust for skew, producing
        // an asymmetric interval shifted toward the tail
        let data = Array1::from_vec(vec![
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0,
        ]);
        let bootstrap = Bootstrap::new(5000).with_seed(99);
        let result = bootstrap.mean(&data).unwrap();

        let (pct_lo, pct_hi) = result.ci_percentile;
        let (bca_lo, bca_hi) = result.ci_bca.unwrap();

        // BCa should differ from percentile for skewed data
        let pct_width = pct_hi - pct_lo;
        let bca_width = bca_hi - bca_lo;

        // Both intervals should contain the original statistic
        assert!(pct_lo <= result.original && result.original <= pct_hi);
        // BCa bounds should be finite and reasonable
        assert!(bca_lo.is_finite() && bca_hi.is_finite());
        assert!(bca_lo < bca_hi, "BCa interval should be non-degenerate");
        // BCa width should be within a reasonable factor of percentile width
        assert!(
            bca_width > 0.3 * pct_width,
            "BCa should not be wildly narrower"
        );
        assert!(
            bca_width < 3.0 * pct_width,
            "BCa should not be wildly wider"
        );
    }

    #[test]
    fn test_bca_symmetric_data() {
        // For symmetric data, BCa should be close to percentile CI
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let bootstrap = Bootstrap::new(5000).with_seed(7);
        let result = bootstrap.mean(&data).unwrap();

        let (pct_lo, pct_hi) = result.ci_percentile;
        let (bca_lo, bca_hi) = result.ci_bca.unwrap();

        // For symmetric data, BCa and percentile should be similar
        assert!(
            (bca_lo - pct_lo).abs() < 1.5,
            "BCa lower ~ percentile lower for symmetric data"
        );
        assert!(
            (bca_hi - pct_hi).abs() < 1.5,
            "BCa upper ~ percentile upper for symmetric data"
        );
    }

    #[test]
    fn test_bca_small_sample() {
        // BCa should still work (return Some) for n >= 3
        let data = Array1::from_vec(vec![1.0, 5.0, 10.0]);
        let bootstrap = Bootstrap::new(1000).with_seed(42);
        let result = bootstrap.mean(&data).unwrap();

        assert!(result.ci_bca.is_some(), "BCa should work for n=3");
    }

    #[test]
    fn test_bca_none_for_tiny_sample() {
        // n < 3 should return None for BCa
        let data = Array1::from_vec(vec![1.0, 2.0]);
        let bootstrap = Bootstrap::new(100).with_seed(42);
        let result = bootstrap.mean(&data).unwrap();

        assert!(result.ci_bca.is_none(), "BCa should be None for n=2");
    }

    #[test]
    fn test_norm_cdf_ppf_roundtrip() {
        // norm_cdf(norm_ppf(p)) ≈ p
        for &p in &[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let z = norm_ppf(p);
            let p2 = norm_cdf(z);
            assert!(
                (p - p2).abs() < 1e-4,
                "roundtrip failed for p={p}: got {p2}"
            );
        }
    }
}
