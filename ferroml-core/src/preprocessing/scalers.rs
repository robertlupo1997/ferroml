//! Feature Scaling Transformers
//!
//! This module provides transformers for scaling numerical features.
//!
//! ## Available Scalers
//!
//! - [`StandardScaler`] - Standardize features by removing mean and scaling to unit variance
//! - [`MinMaxScaler`] - Scale features to a given range (default [0, 1])
//! - [`RobustScaler`] - Scale using statistics robust to outliers (median, IQR)
//! - [`MaxAbsScaler`] - Scale by the maximum absolute value
//!
//! ## When to Use Which Scaler
//!
//! | Scaler | Best For | Notes |
//! |--------|----------|-------|
//! | `StandardScaler` | Data with Gaussian distribution | Sensitive to outliers |
//! | `MinMaxScaler` | Bounded data, neural networks | Sensitive to outliers |
//! | `RobustScaler` | Data with outliers | Uses median/IQR |
//! | `MaxAbsScaler` | Sparse data | Preserves sparsity |
//!
//! ## Example
//!
//! ```
//! use ferroml_core::preprocessing::scalers::StandardScaler;
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! let mut scaler = StandardScaler::new();
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//!
//! // Fit and transform
//! let x_scaled = scaler.fit_transform(&x).unwrap();
//!
//! // Each feature now has mean ≈ 0 and std ≈ 1
//! assert!((x_scaled.column(0).mean().unwrap()).abs() < 1e-10);
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::{
    check_is_fitted, check_non_empty, check_shape, column_max, column_median, column_min,
    column_quantile, compute_column_statistics, generate_feature_names, Transformer,
};
use crate::pipeline::PipelineTransformer;
use crate::Result;

/// Standardize features by removing the mean and scaling to unit variance.
///
/// The standard score of a sample `x` is calculated as:
/// ```text
/// z = (x - mean) / std
/// ```
///
/// where `mean` is the mean of the training samples and `std` is the standard
/// deviation of the training samples.
///
/// # Configuration
///
/// - `with_mean`: Whether to center the data (default: true)
/// - `with_std`: Whether to scale to unit variance (default: true)
///
/// # Notes
///
/// - Features with zero variance are left unchanged (to avoid division by zero)
/// - Sensitive to outliers; consider [`RobustScaler`] for data with outliers
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::scalers::StandardScaler;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut scaler = StandardScaler::new();
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// scaler.fit(&x).unwrap();
/// let x_scaled = scaler.transform(&x).unwrap();
///
/// // Recover original scale
/// let x_recovered = scaler.inverse_transform(&x_scaled).unwrap();
/// assert!((x_recovered[[0, 0]] - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    /// Whether to center the data before scaling
    with_mean: bool,
    /// Whether to scale the data to unit variance
    with_std: bool,
    /// Mean of each feature (learned during fit)
    mean: Option<Array1<f64>>,
    /// Standard deviation of each feature (learned during fit)
    std: Option<Array1<f64>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Number of samples seen during fit
    n_samples_seen: Option<usize>,
    /// Indices of features with zero variance (these are not scaled)
    constant_features: Vec<usize>,
}

impl StandardScaler {
    /// Create a new StandardScaler with default settings.
    ///
    /// Default configuration:
    /// - `with_mean`: true (center data)
    /// - `with_std`: true (scale to unit variance)
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            mean: None,
            std: None,
            n_features_in: None,
            n_samples_seen: None,
            constant_features: Vec::new(),
        }
    }

    /// Configure whether to center data by subtracting mean.
    ///
    /// Setting this to false keeps the data centered at its original mean.
    pub fn with_mean(mut self, center: bool) -> Self {
        self.with_mean = center;
        self
    }

    /// Configure whether to scale data to unit variance.
    ///
    /// Setting this to false only centers the data without scaling.
    pub fn with_std(mut self, scale: bool) -> Self {
        self.with_std = scale;
        self
    }

    /// Get the learned mean for each feature.
    ///
    /// Returns `None` if not fitted.
    pub fn mean(&self) -> Option<&Array1<f64>> {
        self.mean.as_ref()
    }

    /// Get the learned standard deviation for each feature.
    ///
    /// Returns `None` if not fitted.
    pub fn std(&self) -> Option<&Array1<f64>> {
        self.std.as_ref()
    }

    /// Get the indices of constant features (zero variance).
    ///
    /// These features are not scaled to avoid division by zero.
    pub fn constant_features(&self) -> &[usize] {
        &self.constant_features
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for StandardScaler {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let (mean, variance, n_samples) = compute_column_statistics(x);
        let std = variance.mapv(f64::sqrt);

        // Track constant features (zero variance)
        self.constant_features = std
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s < 1e-10 { Some(i) } else { None })
            .collect();

        self.mean = Some(mean);
        self.std = Some(std);
        self.n_features_in = Some(x.ncols());
        self.n_samples_seen = Some(n_samples);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        let mut result = x.clone();

        // Center the data
        if self.with_mean {
            result = result - mean;
        }

        // Scale by standard deviation
        if self.with_std {
            for (j, &s) in std.iter().enumerate() {
                if s > 1e-10 {
                    result.column_mut(j).iter_mut().for_each(|v| *v /= s);
                }
                // Constant features are left unchanged (division by near-zero would be unstable)
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        let mut result = x.clone();

        // Inverse scale by standard deviation
        if self.with_std {
            for (j, &s) in std.iter().enumerate() {
                if s > 1e-10 {
                    result.column_mut(j).iter_mut().for_each(|v| *v *= s);
                }
            }
        }

        // Add back the mean
        if self.with_mean {
            result = result + mean;
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.mean.is_some() && self.std.is_some()
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

/// Scale features to a given range (default [0, 1]).
///
/// The transformation is given by:
/// ```text
/// X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
/// ```
///
/// where `X_min` and `X_max` are the minimum and maximum values from the
/// training data, and `min` and `max` define the target range.
///
/// # Configuration
///
/// - `feature_range`: Target range as `(min, max)` (default: `(0.0, 1.0)`)
///
/// # Notes
///
/// - Features with zero range (constant) are scaled to `min` of the target range
/// - Sensitive to outliers; consider [`RobustScaler`] for data with outliers
/// - Good for neural networks that require bounded inputs
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::scalers::MinMaxScaler;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut scaler = MinMaxScaler::new();
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let x_scaled = scaler.fit_transform(&x).unwrap();
///
/// // All values now in [0, 1]
/// assert!((x_scaled[[0, 0]] - 0.0).abs() < 1e-10); // min -> 0
/// assert!((x_scaled[[2, 0]] - 1.0).abs() < 1e-10); // max -> 1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    /// Target range for scaling (min, max)
    feature_range: (f64, f64),
    /// Minimum value of each feature (learned during fit)
    data_min: Option<Array1<f64>>,
    /// Maximum value of each feature (learned during fit)
    data_max: Option<Array1<f64>>,
    /// Range of each feature: max - min (learned during fit)
    data_range: Option<Array1<f64>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Number of samples seen during fit
    n_samples_seen: Option<usize>,
    /// Indices of constant features (zero range)
    constant_features: Vec<usize>,
}

impl MinMaxScaler {
    /// Create a new MinMaxScaler with default range [0, 1].
    pub fn new() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            data_min: None,
            data_max: None,
            data_range: None,
            n_features_in: None,
            n_samples_seen: None,
            constant_features: Vec::new(),
        }
    }

    /// Set the target range for scaling.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum value in the target range
    /// * `max` - Maximum value in the target range
    ///
    /// # Panics
    ///
    /// Panics if `min >= max`.
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        assert!(min < max, "min must be less than max");
        self.feature_range = (min, max);
        self
    }

    /// Get the learned data minimum for each feature.
    pub fn data_min(&self) -> Option<&Array1<f64>> {
        self.data_min.as_ref()
    }

    /// Get the learned data maximum for each feature.
    pub fn data_max(&self) -> Option<&Array1<f64>> {
        self.data_max.as_ref()
    }

    /// Get the learned data range for each feature.
    pub fn data_range(&self) -> Option<&Array1<f64>> {
        self.data_range.as_ref()
    }

    /// Get the target feature range.
    pub fn feature_range(&self) -> (f64, f64) {
        self.feature_range
    }
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for MinMaxScaler {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let data_min = column_min(x);
        let data_max = column_max(x);
        let data_range = &data_max - &data_min;

        // Track constant features
        self.constant_features = data_range
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| if r.abs() < 1e-10 { Some(i) } else { None })
            .collect();

        self.data_min = Some(data_min);
        self.data_max = Some(data_max);
        self.data_range = Some(data_range);
        self.n_features_in = Some(x.ncols());
        self.n_samples_seen = Some(x.nrows());

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let data_min = self.data_min.as_ref().unwrap();
        let data_range = self.data_range.as_ref().unwrap();
        let (target_min, target_max) = self.feature_range;
        let target_range = target_max - target_min;

        let mut result = Array2::zeros(x.raw_dim());

        for j in 0..x.ncols() {
            let range = data_range[j];
            let min_val = data_min[j];

            if range.abs() < 1e-10 {
                // Constant feature: map to target_min
                result.column_mut(j).fill(target_min);
            } else {
                for (i, &val) in x.column(j).iter().enumerate() {
                    result[[i, j]] = ((val - min_val) / range).mul_add(target_range, target_min);
                }
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let data_min = self.data_min.as_ref().unwrap();
        let data_range = self.data_range.as_ref().unwrap();
        let (target_min, target_max) = self.feature_range;
        let target_range = target_max - target_min;

        let mut result = Array2::zeros(x.raw_dim());

        for j in 0..x.ncols() {
            let range = data_range[j];
            let min_val = data_min[j];

            if range.abs() < 1e-10 {
                // Constant feature: map back to original value
                result.column_mut(j).fill(min_val);
            } else {
                for (i, &val) in x.column(j).iter().enumerate() {
                    result[[i, j]] = ((val - target_min) / target_range).mul_add(range, min_val);
                }
            }
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.data_min.is_some() && self.data_max.is_some() && self.data_range.is_some()
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

/// Scale features using statistics that are robust to outliers.
///
/// This scaler removes the median and scales the data according to the
/// Interquartile Range (IQR). The IQR is the range between the 1st quartile
/// (25th percentile) and the 3rd quartile (75th percentile).
///
/// The transformation is given by:
/// ```text
/// X_scaled = (X - median) / IQR
/// ```
///
/// # Configuration
///
/// - `with_centering`: Whether to center by median (default: true)
/// - `with_scaling`: Whether to scale by IQR (default: true)
/// - `quantile_range`: Percentiles for IQR (default: (25.0, 75.0))
///
/// # Notes
///
/// - Robust to outliers due to use of median and IQR
/// - Useful when data contains outliers that would distort StandardScaler
/// - Features with zero IQR are left unchanged
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::scalers::RobustScaler;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut scaler = RobustScaler::new();
/// let x = array![[1.0], [2.0], [3.0], [4.0], [100.0]]; // Note outlier
///
/// let x_scaled = scaler.fit_transform(&x).unwrap();
///
/// // Scaling is based on IQR, so outlier has less influence
/// assert!(x_scaled[[0, 0]].abs() < 2.0); // Values near median are small
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustScaler {
    /// Whether to center by subtracting median
    with_centering: bool,
    /// Whether to scale by IQR
    with_scaling: bool,
    /// Quantile range for computing IQR (percentiles)
    quantile_range: (f64, f64),
    /// Median of each feature (learned during fit)
    center: Option<Array1<f64>>,
    /// IQR of each feature (learned during fit)
    scale: Option<Array1<f64>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Number of samples seen during fit
    n_samples_seen: Option<usize>,
    /// Indices of features with zero IQR
    constant_features: Vec<usize>,
}

impl RobustScaler {
    /// Create a new RobustScaler with default settings.
    ///
    /// Default configuration:
    /// - `with_centering`: true (subtract median)
    /// - `with_scaling`: true (divide by IQR)
    /// - `quantile_range`: (25.0, 75.0)
    pub fn new() -> Self {
        Self {
            with_centering: true,
            with_scaling: true,
            quantile_range: (25.0, 75.0),
            center: None,
            scale: None,
            n_features_in: None,
            n_samples_seen: None,
            constant_features: Vec::new(),
        }
    }

    /// Configure whether to center data by subtracting median.
    pub fn with_centering(mut self, center: bool) -> Self {
        self.with_centering = center;
        self
    }

    /// Configure whether to scale data by IQR.
    pub fn with_scaling(mut self, scale: bool) -> Self {
        self.with_scaling = scale;
        self
    }

    /// Set the quantile range used for computing IQR.
    ///
    /// # Arguments
    ///
    /// * `q_min` - Lower percentile (default: 25.0)
    /// * `q_max` - Upper percentile (default: 75.0)
    ///
    /// # Panics
    ///
    /// Panics if values are not in [0, 100] or if `q_min >= q_max`.
    pub fn with_quantile_range(mut self, q_min: f64, q_max: f64) -> Self {
        assert!(
            (0.0..=100.0).contains(&q_min) && (0.0..=100.0).contains(&q_max),
            "Quantiles must be in [0, 100]"
        );
        assert!(q_min < q_max, "q_min must be less than q_max");
        self.quantile_range = (q_min, q_max);
        self
    }

    /// Get the learned center (median) for each feature.
    pub fn center(&self) -> Option<&Array1<f64>> {
        self.center.as_ref()
    }

    /// Get the learned scale (IQR) for each feature.
    pub fn scale(&self) -> Option<&Array1<f64>> {
        self.scale.as_ref()
    }
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for RobustScaler {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        // Compute median
        let center = column_median(x);

        // Compute IQR
        let (q_min, q_max) = self.quantile_range;
        let q_low = column_quantile(x, q_min / 100.0);
        let q_high = column_quantile(x, q_max / 100.0);
        let scale = &q_high - &q_low;

        // Track constant features (zero IQR)
        self.constant_features = scale
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if s.abs() < 1e-10 { Some(i) } else { None })
            .collect();

        self.center = Some(center);
        self.scale = Some(scale);
        self.n_features_in = Some(x.ncols());
        self.n_samples_seen = Some(x.nrows());

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let center = self.center.as_ref().unwrap();
        let scale = self.scale.as_ref().unwrap();

        let mut result = x.clone();

        // Center by subtracting median
        if self.with_centering {
            result = result - center;
        }

        // Scale by IQR
        if self.with_scaling {
            for (j, &s) in scale.iter().enumerate() {
                if s.abs() > 1e-10 {
                    result.column_mut(j).iter_mut().for_each(|v| *v /= s);
                }
                // Features with zero IQR are left unchanged
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let center = self.center.as_ref().unwrap();
        let scale = self.scale.as_ref().unwrap();

        let mut result = x.clone();

        // Inverse scale by IQR
        if self.with_scaling {
            for (j, &s) in scale.iter().enumerate() {
                if s.abs() > 1e-10 {
                    result.column_mut(j).iter_mut().for_each(|v| *v *= s);
                }
            }
        }

        // Add back the median
        if self.with_centering {
            result = result + center;
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.center.is_some() && self.scale.is_some()
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

/// Scale features by their maximum absolute value.
///
/// This scaler divides each feature by its maximum absolute value, resulting
/// in features in the range [-1, 1].
///
/// The transformation is given by:
/// ```text
/// X_scaled = X / max(|X|)
/// ```
///
/// # Notes
///
/// - Does not shift/center data, which preserves sparsity
/// - Useful for sparse data where centering would destroy sparsity
/// - Features with zero maximum absolute value are left unchanged
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::scalers::MaxAbsScaler;
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut scaler = MaxAbsScaler::new();
/// let x = array![[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]];
///
/// let x_scaled = scaler.fit_transform(&x).unwrap();
///
/// // All values now in [-1, 1]
/// assert!((x_scaled[[2, 0]] - (-1.0)).abs() < 1e-10); // -5/5 = -1
/// assert!((x_scaled[[2, 1]] - 1.0).abs() < 1e-10);     // 6/6 = 1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxAbsScaler {
    /// Maximum absolute value of each feature (learned during fit)
    max_abs: Option<Array1<f64>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Number of samples seen during fit
    n_samples_seen: Option<usize>,
    /// Indices of features with zero max_abs
    constant_features: Vec<usize>,
}

impl MaxAbsScaler {
    /// Create a new MaxAbsScaler.
    pub fn new() -> Self {
        Self {
            max_abs: None,
            n_features_in: None,
            n_samples_seen: None,
            constant_features: Vec::new(),
        }
    }

    /// Get the learned maximum absolute value for each feature.
    pub fn max_abs(&self) -> Option<&Array1<f64>> {
        self.max_abs.as_ref()
    }
}

impl Default for MaxAbsScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for MaxAbsScaler {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_features = x.ncols();
        let mut max_abs = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            max_abs[j] = col.iter().map(|v| v.abs()).fold(0.0, f64::max);
        }

        // Track constant features (zero max_abs)
        self.constant_features = max_abs
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m.abs() < 1e-10 { Some(i) } else { None })
            .collect();

        self.max_abs = Some(max_abs);
        self.n_features_in = Some(n_features);
        self.n_samples_seen = Some(x.nrows());

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let max_abs = self.max_abs.as_ref().unwrap();

        let mut result = x.clone();

        for (j, &m) in max_abs.iter().enumerate() {
            if m.abs() > 1e-10 {
                result.column_mut(j).iter_mut().for_each(|v| *v /= m);
            }
            // Features with zero max_abs are left unchanged
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let max_abs = self.max_abs.as_ref().unwrap();

        let mut result = x.clone();

        for (j, &m) in max_abs.iter().enumerate() {
            if m.abs() > 1e-10 {
                result.column_mut(j).iter_mut().for_each(|v| *v *= m);
            }
        }

        Ok(result)
    }

    fn is_fitted(&self) -> bool {
        self.max_abs.is_some()
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

// =============================================================================
// PipelineTransformer Implementations
// =============================================================================

impl PipelineTransformer for StandardScaler {
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "StandardScaler"
    }
}

impl PipelineTransformer for MinMaxScaler {
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "MinMaxScaler"
    }
}

impl PipelineTransformer for RobustScaler {
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "RobustScaler"
    }
}

impl PipelineTransformer for MaxAbsScaler {
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "MaxAbsScaler"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Axis};

    const EPSILON: f64 = 1e-10;

    // ========== StandardScaler Tests ==========

    #[test]
    fn test_standard_scaler_basic() {
        let mut scaler = StandardScaler::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Check mean is approximately 0
        let mean_col0 = x_scaled.column(0).mean().unwrap();
        let mean_col1 = x_scaled.column(1).mean().unwrap();
        assert!(mean_col0.abs() < EPSILON);
        assert!(mean_col1.abs() < EPSILON);

        // StandardScaler uses population variance (n) to match sklearn,
        // so check std using population std (ddof=0)
        let std_col0 = x_scaled.std_axis(Axis(0), 0.0)[0];
        let std_col1 = x_scaled.std_axis(Axis(0), 0.0)[1];
        assert!((std_col0 - 1.0).abs() < EPSILON);
        assert!((std_col1 - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_standard_scaler_inverse() {
        let mut scaler = StandardScaler::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();
        let x_recovered = scaler.inverse_transform(&x_scaled).unwrap();

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x[[i, j]] - x_recovered[[i, j]]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_standard_scaler_constant_feature() {
        let mut scaler = StandardScaler::new();
        let x = array![[1.0, 5.0], [1.0, 10.0], [1.0, 15.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Constant feature should remain constant (zero after centering)
        assert!(x_scaled[[0, 0]].abs() < EPSILON);
        assert!(x_scaled[[1, 0]].abs() < EPSILON);
        assert!(x_scaled[[2, 0]].abs() < EPSILON);

        // Check constant_features tracking
        assert_eq!(scaler.constant_features(), &[0]);
    }

    #[test]
    fn test_standard_scaler_no_mean() {
        let mut scaler = StandardScaler::new().with_mean(false);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Mean should not be zero (data not centered)
        let mean_col0 = x_scaled.column(0).mean().unwrap();
        assert!(mean_col0.abs() > 0.1);
    }

    #[test]
    fn test_standard_scaler_no_std() {
        let mut scaler = StandardScaler::new().with_std(false);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Only centered, not scaled
        let mean_col0 = x_scaled.column(0).mean().unwrap();
        assert!(mean_col0.abs() < EPSILON);

        // Std should not be 1
        let std_col0 = x_scaled.std_axis(Axis(0), 1.0)[0];
        assert!((std_col0 - 1.0).abs() > 0.1);
    }

    #[test]
    fn test_standard_scaler_not_fitted() {
        let scaler = StandardScaler::new();
        let x = array![[1.0, 2.0]];

        assert!(scaler.transform(&x).is_err());
        assert!(!scaler.is_fitted());
    }

    #[test]
    fn test_standard_scaler_shape_mismatch() {
        let mut scaler = StandardScaler::new();
        let x_fit = array![[1.0, 2.0], [3.0, 4.0]];
        scaler.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(scaler.transform(&x_wrong).is_err());
    }

    // ========== MinMaxScaler Tests ==========

    #[test]
    fn test_minmax_scaler_basic() {
        let mut scaler = MinMaxScaler::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Min should be 0, max should be 1
        assert!((x_scaled[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[2, 0]] - 1.0).abs() < EPSILON);
        assert!((x_scaled[[0, 1]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[2, 1]] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_minmax_scaler_custom_range() {
        let mut scaler = MinMaxScaler::new().with_range(-1.0, 1.0);
        let x = array![[1.0], [3.0], [5.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        assert!((x_scaled[[0, 0]] - (-1.0)).abs() < EPSILON);
        assert!((x_scaled[[1, 0]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[2, 0]] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_minmax_scaler_inverse() {
        let mut scaler = MinMaxScaler::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();
        let x_recovered = scaler.inverse_transform(&x_scaled).unwrap();

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x[[i, j]] - x_recovered[[i, j]]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_minmax_scaler_constant_feature() {
        let mut scaler = MinMaxScaler::new();
        let x = array![[5.0, 1.0], [5.0, 3.0], [5.0, 5.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Constant feature should map to 0 (target min)
        assert!((x_scaled[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[1, 0]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[2, 0]] - 0.0).abs() < EPSILON);
    }

    // ========== RobustScaler Tests ==========

    #[test]
    fn test_robust_scaler_basic() {
        let mut scaler = RobustScaler::new();
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Median (3.0) should become 0
        assert!((x_scaled[[2, 0]] - 0.0).abs() < EPSILON);

        // Values should be scaled by IQR
        let center = scaler.center().unwrap()[0];
        let scale = scaler.scale().unwrap()[0];
        assert!((center - 3.0).abs() < EPSILON);
        // IQR = Q3 - Q1 = 4.0 - 2.0 = 2.0
        assert!((scale - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_robust_scaler_outlier_robustness() {
        let mut standard = StandardScaler::new();
        let mut robust = RobustScaler::new();

        // Data with outlier
        let x = array![[1.0], [2.0], [3.0], [4.0], [100.0]];

        standard.fit(&x).unwrap();
        robust.fit(&x).unwrap();

        // StandardScaler's std is heavily influenced by outlier
        let std_std = standard.std().unwrap()[0];

        // RobustScaler's IQR is not influenced by outlier
        let robust_scale = robust.scale().unwrap()[0];

        // RobustScaler's scale should be much smaller
        assert!(robust_scale < std_std);
    }

    #[test]
    fn test_robust_scaler_inverse() {
        let mut scaler = RobustScaler::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();
        let x_recovered = scaler.inverse_transform(&x_scaled).unwrap();

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x[[i, j]] - x_recovered[[i, j]]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_robust_scaler_custom_quantiles() {
        let mut scaler = RobustScaler::new().with_quantile_range(10.0, 90.0);
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

        scaler.fit(&x).unwrap();

        // 10th percentile = 1.9, 90th percentile = 9.1
        // Scale should be 9.1 - 1.9 = 7.2
        let scale = scaler.scale().unwrap()[0];
        assert!((scale - 7.2).abs() < 0.1);
    }

    // ========== MaxAbsScaler Tests ==========

    #[test]
    fn test_maxabs_scaler_basic() {
        let mut scaler = MaxAbsScaler::new();
        let x = array![[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Max abs for col 0 is 5, for col 1 is 6
        assert!((x_scaled[[0, 0]] - 0.2).abs() < EPSILON); // 1/5
        assert!((x_scaled[[0, 1]] - (-2.0 / 6.0)).abs() < EPSILON);
        assert!((x_scaled[[2, 0]] - (-1.0)).abs() < EPSILON); // -5/5
        assert!((x_scaled[[2, 1]] - 1.0).abs() < EPSILON); // 6/6
    }

    #[test]
    fn test_maxabs_scaler_inverse() {
        let mut scaler = MaxAbsScaler::new();
        let x = array![[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();
        let x_recovered = scaler.inverse_transform(&x_scaled).unwrap();

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x[[i, j]] - x_recovered[[i, j]]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_maxabs_scaler_preserves_sparsity() {
        let mut scaler = MaxAbsScaler::new();
        let x = array![[0.0, 0.0], [3.0, 0.0], [0.0, 6.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // Zeros should remain zeros (sparsity preserved)
        assert!((x_scaled[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[0, 1]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[1, 1]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[2, 0]] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_maxabs_scaler_constant_zero_feature() {
        let mut scaler = MaxAbsScaler::new();
        let x = array![[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]];

        let x_scaled = scaler.fit_transform(&x).unwrap();

        // All-zero feature should remain zero
        assert!((x_scaled[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[1, 0]] - 0.0).abs() < EPSILON);
        assert!((x_scaled[[2, 0]] - 0.0).abs() < EPSILON);
    }

    // ========== Common Tests ==========

    #[test]
    fn test_empty_input() {
        let empty: Array2<f64> = Array2::zeros((0, 0));

        let mut standard = StandardScaler::new();
        let mut minmax = MinMaxScaler::new();
        let mut robust = RobustScaler::new();
        let mut maxabs = MaxAbsScaler::new();

        assert!(standard.fit(&empty).is_err());
        assert!(minmax.fit(&empty).is_err());
        assert!(robust.fit(&empty).is_err());
        assert!(maxabs.fit(&empty).is_err());
    }

    #[test]
    fn test_feature_names() {
        let mut scaler = StandardScaler::new();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        scaler.fit(&x).unwrap();

        // Default names
        let names = scaler.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["x0", "x1", "x2"]);

        // Custom names
        let custom = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let names = scaler.get_feature_names_out(Some(&custom)).unwrap();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_n_features() {
        let mut scaler = StandardScaler::new();
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Before fit
        assert!(scaler.n_features_in().is_none());
        assert!(scaler.n_features_out().is_none());

        // After fit
        scaler.fit(&x).unwrap();
        assert_eq!(scaler.n_features_in(), Some(3));
        assert_eq!(scaler.n_features_out(), Some(3));
    }

    #[test]
    fn test_single_sample() {
        let mut standard = StandardScaler::new();
        let mut minmax = MinMaxScaler::new();

        let x = array![[1.0, 2.0, 3.0]]; // Single sample

        // StandardScaler with single sample: variance is 0
        standard.fit(&x).unwrap();
        let x_scaled = standard.transform(&x).unwrap();
        // With zero variance, values are just centered (mean = value, so centered = 0)
        assert!((x_scaled[[0, 0]] - 0.0).abs() < EPSILON);

        // MinMaxScaler with single sample: range is 0, maps to target min
        minmax.fit(&x).unwrap();
        let x_scaled = minmax.transform(&x).unwrap();
        // Constant features map to 0 (target min)
        assert!((x_scaled[[0, 0]] - 0.0).abs() < EPSILON);
    }
}
