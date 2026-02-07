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

    /// Run bootstrap for a statistic
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

        Ok(BootstrapResult {
            original,
            std_error,
            bias,
            ci_percentile,
            ci_bca: None, // BCa requires jackknife, implement separately
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
}
