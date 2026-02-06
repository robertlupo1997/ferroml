//! Quantile Transformations
//!
//! This module provides transformers that map data to specific probability distributions
//! using quantile information.
//!
//! ## Available Transformers
//!
//! - [`QuantileTransformer`] - Transform features to uniform or normal distribution
//!
//! ## When to Use Quantile Transformations
//!
//! Quantile transformations are useful when:
//! - You need to transform features to a known distribution (uniform or normal)
//! - You want robust handling of outliers (quantile-based approach)
//! - Linear models require normally distributed features but data is heavily skewed
//!
//! ## Comparison with Power Transformations
//!
//! | Method | Approach | Guarantees | Best For |
//! |--------|----------|------------|----------|
//! | QuantileTransformer | Non-parametric (empirical CDF) | Exact target distribution | Any data, strong guarantees |
//! | PowerTransformer | Parametric (Box-Cox/Yeo-Johnson) | Approximate normality | Moderately skewed data |
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::preprocessing::quantile::{QuantileTransformer, OutputDistribution};
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! // Transform to uniform distribution [0, 1]
//! let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
//! let x = array![[1.0], [2.0], [3.0], [10.0], [100.0]];
//!
//! let x_transformed = transformer.fit_transform(&x).unwrap();
//! // Data is now uniformly distributed in [0, 1]
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use super::{check_is_fitted, check_non_empty, check_shape, generate_feature_names, Transformer};
use crate::{FerroError, Result};

/// Output distribution type for QuantileTransformer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputDistribution {
    /// Transform to uniform distribution in [0, 1].
    Uniform,
    /// Transform to standard normal distribution (mean=0, std=1).
    Normal,
}

impl Default for OutputDistribution {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Transform features using quantile information to map data to a specified distribution.
///
/// This transformer uses the empirical cumulative distribution function (CDF) to map
/// each feature to a uniform or normal distribution. The transformation is non-parametric
/// and works well for any continuous distribution.
///
/// # Algorithm
///
/// 1. **Fit**: For each feature, compute `n_quantiles` evenly spaced quantiles from the
///    training data. This creates a mapping from data values to their CDF values.
///
/// 2. **Transform**: For new data, use linear interpolation to find the CDF value
///    (position in the training distribution). For uniform output, this is the final
///    value. For normal output, apply the inverse normal CDF (probit function).
///
/// 3. **Inverse Transform**: Reverse the process - for normal output, first apply the
///    normal CDF, then use linear interpolation to map back to original scale.
///
/// # Configuration
///
/// - `output_distribution`: Target distribution (Uniform or Normal)
/// - `n_quantiles`: Number of quantiles to compute (default: 1000 or n_samples)
/// - `subsample`: Maximum samples to use for fitting (for memory efficiency)
///
/// # Notes
///
/// - Robust to outliers due to quantile-based approach
/// - For uniform output, values are clipped to [0, 1]
/// - For normal output, values are clipped to avoid infinite values at 0 and 1
/// - Values outside the training range are mapped to 0 (min) or 1 (max) for uniform
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::quantile::{QuantileTransformer, OutputDistribution};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
/// let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
///
/// transformer.fit(&x).unwrap();
/// let x_transformed = transformer.transform(&x).unwrap();
///
/// // Values are now in [0, 1], uniformly distributed
/// assert!(x_transformed[[0, 0]] >= 0.0 && x_transformed[[0, 0]] <= 1.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileTransformer {
    /// Target output distribution
    output_distribution: OutputDistribution,
    /// Number of quantiles to compute
    n_quantiles: usize,
    /// Maximum number of samples to use for fitting (None = use all)
    subsample: Option<usize>,
    /// Learned quantiles for each feature: shape (n_quantiles, n_features)
    quantiles: Option<Vec<Array1<f64>>>,
    /// References (percentile values) corresponding to quantiles: [0, 1/(n-1), 2/(n-1), ..., 1]
    references: Option<Array1<f64>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Number of samples seen during fit
    n_samples_seen: Option<usize>,
    /// Small epsilon for clipping to avoid numerical issues at boundaries
    clip_epsilon: f64,
}

impl QuantileTransformer {
    /// Create a new QuantileTransformer with the specified output distribution.
    ///
    /// Default configuration:
    /// - `n_quantiles`: 1000 (or n_samples if smaller)
    /// - `subsample`: None (use all samples)
    ///
    /// # Arguments
    ///
    /// * `output_distribution` - Target distribution (Uniform or Normal)
    ///
    /// # Example
    ///
    /// ```
    /// use ferroml_core::preprocessing::quantile::{QuantileTransformer, OutputDistribution};
    ///
    /// // Transform to uniform distribution
    /// let uniform_transformer = QuantileTransformer::new(OutputDistribution::Uniform);
    ///
    /// // Transform to normal distribution
    /// let normal_transformer = QuantileTransformer::new(OutputDistribution::Normal);
    /// ```
    pub fn new(output_distribution: OutputDistribution) -> Self {
        Self {
            output_distribution,
            n_quantiles: 1000,
            subsample: None,
            quantiles: None,
            references: None,
            n_features_in: None,
            n_samples_seen: None,
            clip_epsilon: 1e-7,
        }
    }

    /// Set the number of quantiles to compute.
    ///
    /// More quantiles provide finer-grained mapping but require more memory.
    /// If `n_quantiles` exceeds the number of training samples, it will be
    /// automatically reduced to `n_samples`.
    ///
    /// # Arguments
    ///
    /// * `n_quantiles` - Number of quantiles (default: 1000)
    ///
    /// # Panics
    ///
    /// Panics if `n_quantiles < 2`.
    pub fn with_n_quantiles(mut self, n_quantiles: usize) -> Self {
        assert!(n_quantiles >= 2, "n_quantiles must be at least 2");
        self.n_quantiles = n_quantiles;
        self
    }

    /// Set the maximum number of samples to use for fitting.
    ///
    /// This is useful for large datasets to reduce memory usage and computation time.
    /// Samples are selected randomly if subsampling is needed.
    ///
    /// # Arguments
    ///
    /// * `subsample` - Maximum samples to use (None = use all)
    pub fn with_subsample(mut self, subsample: Option<usize>) -> Self {
        self.subsample = subsample;
        self
    }

    /// Get the output distribution type.
    pub fn output_distribution(&self) -> OutputDistribution {
        self.output_distribution
    }

    /// Get the number of quantiles.
    pub fn n_quantiles(&self) -> usize {
        self.n_quantiles
    }

    /// Get the learned quantiles for each feature.
    ///
    /// Returns `None` if not fitted.
    pub fn quantiles(&self) -> Option<&Vec<Array1<f64>>> {
        self.quantiles.as_ref()
    }

    /// Get the reference values (percentile positions).
    ///
    /// Returns `None` if not fitted.
    pub fn references(&self) -> Option<&Array1<f64>> {
        self.references.as_ref()
    }

    /// Compute quantiles for a single column.
    fn compute_column_quantiles(&self, col: &[f64], n_quantiles: usize) -> Array1<f64> {
        let mut sorted: Vec<f64> = col.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let mut quantiles = Array1::zeros(n_quantiles);

        for (i, q) in quantiles.iter_mut().enumerate() {
            let p = i as f64 / (n_quantiles - 1).max(1) as f64;
            *q = self.interpolate_quantile(&sorted, p, n);
        }

        quantiles
    }

    /// Interpolate a quantile value using linear interpolation.
    fn interpolate_quantile(&self, sorted: &[f64], p: f64, n: usize) -> f64 {
        if n == 0 {
            return f64::NAN;
        }
        if n == 1 {
            return sorted[0];
        }

        // Position in the sorted array
        let pos = p * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        let frac = pos - lower as f64;

        if lower == upper || upper >= n {
            sorted[lower.min(n - 1)]
        } else {
            sorted[lower].mul_add(1.0 - frac, sorted[upper] * frac)
        }
    }

    /// Transform a single value using the empirical CDF and learned quantiles.
    fn transform_value(
        &self,
        value: f64,
        quantiles: &Array1<f64>,
        references: &Array1<f64>,
    ) -> f64 {
        let n = quantiles.len();
        if n < 2 {
            return 0.5;
        }

        // Handle values outside the training range
        if value <= quantiles[0] {
            return self.clip_epsilon;
        }
        if value >= quantiles[n - 1] {
            return 1.0 - self.clip_epsilon;
        }

        // Binary search to find the position in quantiles
        let mut lo = 0;
        let mut hi = n - 1;

        while lo < hi {
            let mid = (lo + hi) / 2;
            if quantiles[mid] < value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Interpolate between quantiles[lo-1] and quantiles[lo]
        if lo == 0 {
            references[0]
        } else {
            let q_low = quantiles[lo - 1];
            let q_high = quantiles[lo];
            let r_low = references[lo - 1];
            let r_high = references[lo];

            if (q_high - q_low).abs() < 1e-10 {
                // Constant region - return midpoint
                (r_low + r_high) / 2.0
            } else {
                // Linear interpolation
                let frac = (value - q_low) / (q_high - q_low);
                frac.mul_add(r_high - r_low, r_low)
            }
        }
    }

    /// Inverse transform a single value from CDF space back to original scale.
    fn inverse_transform_value(
        &self,
        value: f64,
        quantiles: &Array1<f64>,
        references: &Array1<f64>,
    ) -> f64 {
        let n = references.len();
        if n < 2 {
            return quantiles.get(0).copied().unwrap_or(0.0);
        }

        // Clip to valid range
        let value = value.clamp(self.clip_epsilon, 1.0 - self.clip_epsilon);

        // Binary search in references to find the position
        let mut lo = 0;
        let mut hi = n - 1;

        while lo < hi {
            let mid = (lo + hi) / 2;
            if references[mid] < value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Interpolate between quantiles[lo-1] and quantiles[lo]
        if lo == 0 {
            quantiles[0]
        } else {
            let r_low = references[lo - 1];
            let r_high = references[lo];
            let q_low = quantiles[lo - 1];
            let q_high = quantiles[lo];

            if (r_high - r_low).abs() < 1e-10 {
                (q_low + q_high) / 2.0
            } else {
                let frac = (value - r_low) / (r_high - r_low);
                q_low + frac * (q_high - q_low)
            }
        }
    }

    /// Apply the output distribution transformation.
    fn apply_output_distribution(&self, value: f64) -> f64 {
        match self.output_distribution {
            OutputDistribution::Uniform => value,
            OutputDistribution::Normal => {
                // Clip to avoid infinite values at 0 and 1
                let clipped = value.clamp(self.clip_epsilon, 1.0 - self.clip_epsilon);
                // Apply inverse normal CDF (probit function)
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.inverse_cdf(clipped)
            }
        }
    }

    /// Inverse the output distribution transformation.
    fn inverse_output_distribution(&self, value: f64) -> f64 {
        match self.output_distribution {
            OutputDistribution::Uniform => value,
            OutputDistribution::Normal => {
                // Apply normal CDF (standard normal cumulative distribution)
                let normal = Normal::new(0.0, 1.0).unwrap();
                normal.cdf(value)
            }
        }
    }
}

impl Default for QuantileTransformer {
    fn default() -> Self {
        Self::new(OutputDistribution::Uniform)
    }
}

impl Transformer for QuantileTransformer {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Determine actual number of quantiles to use
        let actual_n_quantiles = self.n_quantiles.min(n_samples);

        if actual_n_quantiles < 2 {
            return Err(FerroError::invalid_input(
                "Need at least 2 samples to compute quantiles",
            ));
        }

        // Subsample if needed
        let data = if let Some(max_samples) = self.subsample {
            if max_samples < n_samples {
                // Simple deterministic subsampling (take evenly spaced samples)
                let step = n_samples / max_samples;
                let indices: Vec<usize> = (0..max_samples).map(|i| i * step).collect();
                let mut subsampled = Array2::zeros((indices.len(), n_features));
                for (new_i, &old_i) in indices.iter().enumerate() {
                    subsampled.row_mut(new_i).assign(&x.row(old_i));
                }
                subsampled
            } else {
                x.to_owned()
            }
        } else {
            x.to_owned()
        };

        // Compute quantiles for each feature
        let mut quantiles = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col: Vec<f64> = data.column(j).to_vec();
            quantiles.push(self.compute_column_quantiles(&col, actual_n_quantiles));
        }

        // Create reference values (evenly spaced in [0, 1])
        let references: Array1<f64> = (0..actual_n_quantiles)
            .map(|i| i as f64 / (actual_n_quantiles - 1).max(1) as f64)
            .collect();

        self.quantiles = Some(quantiles);
        self.references = Some(references);
        self.n_features_in = Some(n_features);
        self.n_samples_seen = Some(n_samples);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let quantiles = self.quantiles.as_ref().unwrap();
        let references = self.references.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut result = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let feature_quantiles = &quantiles[j];
            for i in 0..n_samples {
                // Map to CDF value [0, 1]
                let cdf_value = self.transform_value(x[[i, j]], feature_quantiles, references);
                // Apply output distribution transformation
                result[[i, j]] = self.apply_output_distribution(cdf_value);
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let quantiles = self.quantiles.as_ref().unwrap();
        let references = self.references.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut result = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let feature_quantiles = &quantiles[j];
            for i in 0..n_samples {
                // Inverse output distribution transformation to get CDF value
                let cdf_value = self.inverse_output_distribution(x[[i, j]]);
                // Map from CDF value back to original scale
                result[[i, j]] =
                    self.inverse_transform_value(cdf_value, feature_quantiles, references);
            }
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.quantiles.is_some() && self.references.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        Some(
            input_names
                .map(|names| names.to_vec())
                .unwrap_or_else(|| generate_feature_names(n_features)),
        )
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        self.n_features_in
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-5;

    // ========== OutputDistribution Tests ==========

    #[test]
    fn test_output_distribution_default() {
        assert_eq!(OutputDistribution::default(), OutputDistribution::Uniform);
    }

    // ========== Uniform Distribution Tests ==========

    #[test]
    fn test_uniform_basic() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let x_transformed = transformer.fit_transform(&x).unwrap();

        // All values should be in [0, 1]
        for val in x_transformed.iter() {
            assert!(*val >= 0.0 && *val <= 1.0, "Value {} not in [0, 1]", val);
        }

        // Min should map to near 0, max to near 1
        assert!(x_transformed[[0, 0]] < 0.1);
        assert!(x_transformed[[4, 0]] > 0.9);
    }

    #[test]
    fn test_uniform_monotonicity() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[1.0], [3.0], [5.0], [7.0], [9.0]];

        let x_transformed = transformer.fit_transform(&x).unwrap();

        // Transformed values should preserve order
        for i in 0..x_transformed.nrows() - 1 {
            assert!(
                x_transformed[[i, 0]] < x_transformed[[i + 1, 0]],
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_uniform_distribution_shape() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        // Create data with known distribution (uniform input -> uniform output)
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0],
            [10.0]
        ];

        let x_transformed = transformer.fit_transform(&x).unwrap();

        // Check that values are roughly evenly spaced (uniform distribution)
        let mean: f64 = x_transformed.iter().sum::<f64>() / x_transformed.len() as f64;
        // Mean of uniform [0,1] should be approximately 0.5
        assert!((mean - 0.5).abs() < 0.1, "Mean {} not near 0.5", mean);
    }

    // ========== Normal Distribution Tests ==========

    #[test]
    fn test_normal_basic() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Normal);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let x_transformed = transformer.fit_transform(&x).unwrap();

        // Median should map to near 0 (center of normal distribution)
        assert!(
            x_transformed[[2, 0]].abs() < 0.3,
            "Median {} not near 0",
            x_transformed[[2, 0]]
        );

        // Values should be ordered
        for i in 0..x_transformed.nrows() - 1 {
            assert!(
                x_transformed[[i, 0]] < x_transformed[[i + 1, 0]],
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_normal_distribution_properties() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Normal);
        // More samples for better statistical properties
        let x: Array2<f64> =
            Array2::from_shape_vec((100, 1), (1..=100).map(|i| i as f64).collect()).unwrap();

        let x_transformed = transformer.fit_transform(&x).unwrap();

        // Mean should be approximately 0
        let mean: f64 = x_transformed.iter().sum::<f64>() / x_transformed.len() as f64;
        assert!(mean.abs() < 0.1, "Mean {} not near 0", mean);

        // Std should be approximately 1 (allowing larger tolerance for finite sample effects)
        let var: f64 = x_transformed
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / x_transformed.len() as f64;
        let std = var.sqrt();
        assert!((std - 1.0).abs() < 0.3, "Std {} not near 1", std);
    }

    // ========== Inverse Transform Tests ==========

    #[test]
    fn test_uniform_roundtrip() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();
        let x_recovered = transformer.inverse_transform(&x_transformed).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (x[[i, 0]] - x_recovered[[i, 0]]).abs() < 0.1,
                "Roundtrip failed at index {}: expected {}, got {}",
                i,
                x[[i, 0]],
                x_recovered[[i, 0]]
            );
        }
    }

    #[test]
    fn test_normal_roundtrip() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Normal);
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();
        let x_recovered = transformer.inverse_transform(&x_transformed).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (x[[i, 0]] - x_recovered[[i, 0]]).abs() < 0.1,
                "Roundtrip failed at index {}: expected {}, got {}",
                i,
                x[[i, 0]],
                x_recovered[[i, 0]]
            );
        }
    }

    // ========== Multiple Features Tests ==========

    #[test]
    fn test_multiple_features() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
            [5.0, 50.0, 500.0]
        ];

        transformer.fit(&x).unwrap();

        // Check that we have quantiles for each feature
        let quantiles = transformer.quantiles().unwrap();
        assert_eq!(quantiles.len(), 3);

        // Transform
        let x_transformed = transformer.transform(&x).unwrap();
        assert_eq!(x_transformed.shape(), &[5, 3]);

        // All values should be in [0, 1]
        for val in x_transformed.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_outliers() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        // Data with outlier
        let x = array![[1.0], [2.0], [3.0], [4.0], [100.0]];

        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();

        // All values still in [0, 1]
        for val in x_transformed.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }

        // Most values should still be spread across the range
        // (not all bunched near 0 due to the outlier)
        let median_idx = 2;
        assert!(
            x_transformed[[median_idx, 0]] > 0.3 && x_transformed[[median_idx, 0]] < 0.7,
            "Median value {} not in reasonable range",
            x_transformed[[median_idx, 0]]
        );
    }

    #[test]
    fn test_values_outside_training_range() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x_train = array![[2.0], [3.0], [4.0], [5.0], [6.0]];
        let x_test = array![[0.0], [1.0], [4.0], [7.0], [10.0]]; // Values outside training range

        transformer.fit(&x_train).unwrap();
        let x_transformed = transformer.transform(&x_test).unwrap();

        // Values below training min should map to near 0
        assert!(x_transformed[[0, 0]] < 0.1);
        assert!(x_transformed[[1, 0]] < 0.1);

        // Values above training max should map to near 1
        assert!(x_transformed[[3, 0]] > 0.9);
        assert!(x_transformed[[4, 0]] > 0.9);
    }

    #[test]
    fn test_constant_feature() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0], [5.0, 5.0]];

        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();

        // Constant feature should have consistent output
        // (likely all mapping to the same value)
        let first_val = x_transformed[[0, 0]];
        for i in 1..x.nrows() {
            assert!(
                (x_transformed[[i, 0]] - first_val).abs() < EPSILON,
                "Constant feature values differ"
            );
        }
    }

    #[test]
    fn test_single_sample() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[5.0]];

        // Should fail - need at least 2 samples for quantiles
        let result = transformer.fit(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_two_samples() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[1.0], [5.0]];

        transformer.fit(&x).unwrap();
        let x_transformed = transformer.transform(&x).unwrap();

        // Min maps to near 0, max maps to near 1
        assert!(x_transformed[[0, 0]] < 0.1);
        assert!(x_transformed[[1, 0]] > 0.9);
    }

    #[test]
    fn test_empty_input() {
        let empty: Array2<f64> = Array2::zeros((0, 0));
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        assert!(transformer.fit(&empty).is_err());
    }

    // ========== Configuration Tests ==========

    #[test]
    fn test_n_quantiles_config() {
        let mut transformer =
            QuantileTransformer::new(OutputDistribution::Uniform).with_n_quantiles(10);

        assert_eq!(transformer.n_quantiles(), 10);

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        transformer.fit(&x).unwrap();

        // With 5 samples and n_quantiles=10, should use 5 quantiles
        let quantiles = transformer.quantiles().unwrap();
        assert_eq!(quantiles[0].len(), 5);
    }

    #[test]
    fn test_subsample_config() {
        let mut transformer =
            QuantileTransformer::new(OutputDistribution::Uniform).with_subsample(Some(5));

        let x: Array2<f64> =
            Array2::from_shape_vec((100, 1), (1..=100).map(|i| i as f64).collect()).unwrap();

        transformer.fit(&x).unwrap();

        // Should still work, just using subsampled data
        assert!(transformer.is_fitted());
    }

    // ========== Fitted State Tests ==========

    #[test]
    fn test_not_fitted() {
        let transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[1.0, 2.0]];

        assert!(transformer.transform(&x).is_err());
        assert!(!transformer.is_fitted());
    }

    #[test]
    fn test_shape_mismatch() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x_fit = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        transformer.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(transformer.transform(&x_wrong).is_err());
    }

    // ========== Feature Metadata Tests ==========

    #[test]
    fn test_feature_names() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        transformer.fit(&x).unwrap();

        // Default names
        let names = transformer.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["x0", "x1", "x2"]);

        // Custom names
        let custom = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let names = transformer.get_feature_names_out(Some(&custom)).unwrap();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_n_features() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        // Before fit
        assert!(transformer.n_features_in().is_none());
        assert!(transformer.n_features_out().is_none());

        // After fit
        transformer.fit(&x).unwrap();
        assert_eq!(transformer.n_features_in(), Some(3));
        assert_eq!(transformer.n_features_out(), Some(3));
    }

    // ========== Serialization Tests ==========

    #[test]
    fn test_serialization() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Normal);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        transformer.fit(&x).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&transformer).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let restored: QuantileTransformer = serde_json::from_str(&json).unwrap();
        assert!(restored.is_fitted());
        assert_eq!(restored.quantiles().unwrap().len(), 2);
        assert_eq!(restored.output_distribution(), OutputDistribution::Normal);
    }

    // ========== Comparison with Expected Behavior ==========

    #[test]
    fn test_uniform_maps_to_percentiles() {
        let mut transformer = QuantileTransformer::new(OutputDistribution::Uniform);
        // Simple data: [1, 2, 3, 4, 5] -> should map to [0, 0.25, 0.5, 0.75, 1]
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let x_transformed = transformer.fit_transform(&x).unwrap();

        // Check approximate mapping to percentiles
        assert!(x_transformed[[0, 0]] < 0.1); // ~0
        assert!((x_transformed[[2, 0]] - 0.5).abs() < 0.1); // ~0.5 (median)
        assert!(x_transformed[[4, 0]] > 0.9); // ~1
    }
}
