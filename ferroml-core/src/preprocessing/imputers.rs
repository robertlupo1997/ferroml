//! Missing Value Imputation
//!
//! This module provides transformers for handling missing values.
//!
//! ## Available Imputers
//!
//! - [`SimpleImputer`] - Impute using simple strategies (mean, median, mode, constant)
//! - [`KNNImputer`] - Impute using k-nearest neighbors
//!
//! ## Imputation Strategies
//!
//! | Strategy | Best For | Notes |
//! |----------|----------|-------|
//! | Mean | Normally distributed features | Sensitive to outliers |
//! | Median | Skewed distributions, outliers | More robust than mean |
//! | Mode (MostFrequent) | Categorical or discrete features | Most frequent value |
//! | Constant | When domain-specific fill makes sense | User-specified value |
//! | KNN | Complex missing patterns | Computationally expensive |
//!
//! ## Example
//!
//! ```
//! use ferroml_core::preprocessing::imputers::{SimpleImputer, ImputeStrategy};
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
//! let x = array![[1.0, 2.0], [f64::NAN, 4.0], [3.0, f64::NAN]];
//!
//! let x_imputed = imputer.fit_transform(&x).unwrap();
//!
//! // Missing values are now filled with column means
//! assert!(!x_imputed[[1, 0]].is_nan()); // Was NaN, now filled
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{check_is_fitted, check_non_empty, check_shape, generate_feature_names, Transformer};
use crate::pipeline::PipelineTransformer;
use crate::{FerroError, Result};

/// Strategy for imputing missing values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImputeStrategy {
    /// Replace missing values with the mean of each column.
    ///
    /// Best for normally distributed data. Sensitive to outliers.
    Mean,

    /// Replace missing values with the median of each column.
    ///
    /// More robust to outliers than mean. Good for skewed distributions.
    Median,

    /// Replace missing values with the most frequent value in each column.
    ///
    /// Best for discrete or categorical data (represented as floats).
    MostFrequent,

    /// Replace missing values with a constant value.
    ///
    /// Use when you have domain knowledge about what missing means.
    Constant,
}

impl Default for ImputeStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

/// Imputation transformer for completing missing values.
///
/// Simple imputation fills missing values (NaN) using statistics computed
/// from the non-missing values in each column.
///
/// # Strategies
///
/// - [`ImputeStrategy::Mean`]: Use column mean (default)
/// - [`ImputeStrategy::Median`]: Use column median
/// - [`ImputeStrategy::MostFrequent`]: Use most frequent value (mode)
/// - [`ImputeStrategy::Constant`]: Use a user-specified constant
///
/// # Missing Value Detection
///
/// By default, `f64::NAN` is treated as missing. You can configure a
/// different "missing indicator" value using [`SimpleImputer::with_missing_value`].
///
/// # Diagnostics
///
/// The imputer tracks statistics useful for data quality analysis:
/// - Number of missing values per feature
/// - The fill value computed for each feature
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::imputers::{SimpleImputer, ImputeStrategy};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// // Create imputer with median strategy
/// let mut imputer = SimpleImputer::new(ImputeStrategy::Median);
///
/// let x = array![[1.0, 2.0], [f64::NAN, 4.0], [3.0, f64::NAN], [5.0, 6.0]];
/// let x_imputed = imputer.fit_transform(&x).unwrap();
///
/// // Check diagnostics
/// let missing_counts = imputer.missing_counts().unwrap();
/// assert_eq!(missing_counts[0], 1); // 1 missing in first column
/// assert_eq!(missing_counts[1], 1); // 1 missing in second column
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleImputer {
    /// Strategy for imputation
    strategy: ImputeStrategy,

    /// Value to use when strategy is Constant
    fill_value: f64,

    /// Value to treat as missing (if Some, otherwise NaN is used)
    missing_value: Option<f64>,

    /// Computed fill values for each feature (learned during fit)
    statistics: Option<Array1<f64>>,

    /// Number of features seen during fit
    n_features_in: Option<usize>,

    /// Number of missing values per feature during fit
    missing_counts: Option<Vec<usize>>,

    /// Indices of features where all values were missing (filled with fill_value or 0)
    all_missing_features: Vec<usize>,

    /// Whether to add an indicator column for missing values
    add_indicator: bool,
}

impl SimpleImputer {
    /// Create a new SimpleImputer with the specified strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The imputation strategy to use
    ///
    /// # Example
    ///
    /// ```
    /// use ferroml_core::preprocessing::imputers::{SimpleImputer, ImputeStrategy};
    ///
    /// let imputer = SimpleImputer::new(ImputeStrategy::Mean);
    /// ```
    pub fn new(strategy: ImputeStrategy) -> Self {
        Self {
            strategy,
            fill_value: 0.0,
            missing_value: None,
            statistics: None,
            n_features_in: None,
            missing_counts: None,
            all_missing_features: Vec::new(),
            add_indicator: false,
        }
    }

    /// Set the fill value for Constant strategy.
    ///
    /// This value is also used as a fallback for features where all values
    /// are missing (regardless of strategy).
    ///
    /// # Arguments
    ///
    /// * `value` - The constant value to use for imputation
    ///
    /// # Example
    ///
    /// ```
    /// use ferroml_core::preprocessing::imputers::{SimpleImputer, ImputeStrategy};
    ///
    /// let imputer = SimpleImputer::new(ImputeStrategy::Constant)
    ///     .with_fill_value(-999.0);
    /// ```
    pub fn with_fill_value(mut self, value: f64) -> Self {
        self.fill_value = value;
        self
    }

    /// Set a specific value to treat as missing instead of NaN.
    ///
    /// Some datasets use sentinel values like -1, -999, or 0 to indicate
    /// missing values instead of NaN.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to treat as missing
    ///
    /// # Example
    ///
    /// ```
    /// use ferroml_core::preprocessing::imputers::{SimpleImputer, ImputeStrategy};
    ///
    /// // Treat -1 as missing
    /// let imputer = SimpleImputer::new(ImputeStrategy::Mean)
    ///     .with_missing_value(-1.0);
    /// ```
    pub fn with_missing_value(mut self, value: f64) -> Self {
        self.missing_value = Some(value);
        self
    }

    /// Configure whether to add indicator columns for missing values.
    ///
    /// When enabled, the transform output includes additional binary columns
    /// indicating which values were originally missing.
    ///
    /// # Arguments
    ///
    /// * `add` - Whether to add indicator columns
    pub fn with_indicator(mut self, add: bool) -> Self {
        self.add_indicator = add;
        self
    }

    /// Get the computed statistics (fill values) for each feature.
    ///
    /// Returns `None` if not fitted.
    pub fn statistics(&self) -> Option<&Array1<f64>> {
        self.statistics.as_ref()
    }

    /// Get the number of missing values found per feature during fit.
    ///
    /// Returns `None` if not fitted.
    pub fn missing_counts(&self) -> Option<&[usize]> {
        self.missing_counts.as_deref()
    }

    /// Get the indices of features where all values were missing.
    ///
    /// These features are filled with the fallback value (fill_value or 0).
    pub fn all_missing_features(&self) -> &[usize] {
        &self.all_missing_features
    }

    /// Get the imputation strategy.
    pub fn strategy(&self) -> ImputeStrategy {
        self.strategy
    }

    /// Check if a value should be treated as missing.
    #[inline]
    fn is_missing(&self, value: f64) -> bool {
        match self.missing_value {
            Some(mv) => {
                // Handle the case where missing_value itself is NaN
                if mv.is_nan() {
                    value.is_nan()
                } else {
                    // Use approximate equality for the sentinel value
                    (value - mv).abs() < f64::EPSILON || value.is_nan()
                }
            }
            None => value.is_nan(),
        }
    }

    /// Compute mean for a column, ignoring missing values.
    fn compute_mean(&self, col: &[f64]) -> Option<f64> {
        let valid: Vec<f64> = col
            .iter()
            .filter(|&&v| !self.is_missing(v))
            .copied()
            .collect();
        if valid.is_empty() {
            None
        } else {
            Some(valid.iter().sum::<f64>() / valid.len() as f64)
        }
    }

    /// Compute median for a column, ignoring missing values.
    fn compute_median(&self, col: &[f64]) -> Option<f64> {
        let mut valid: Vec<f64> = col
            .iter()
            .filter(|&&v| !self.is_missing(v))
            .copied()
            .collect();
        if valid.is_empty() {
            return None;
        }

        valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = valid.len();

        Some(if n % 2 == 0 {
            (valid[n / 2 - 1] + valid[n / 2]) / 2.0
        } else {
            valid[n / 2]
        })
    }

    /// Compute mode (most frequent value) for a column, ignoring missing values.
    fn compute_mode(&self, col: &[f64]) -> Option<f64> {
        let valid: Vec<f64> = col
            .iter()
            .filter(|&&v| !self.is_missing(v))
            .copied()
            .collect();
        if valid.is_empty() {
            return None;
        }

        // Count frequencies using a HashMap
        // We use a string key to handle float comparison issues
        let mut counts: HashMap<String, (f64, usize)> = HashMap::new();
        for &v in &valid {
            let key = format!("{:.10}", v);
            counts
                .entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((v, 1));
        }

        // Find the most frequent value; break ties by smallest value (matches scikit-learn)
        counts
            .into_values()
            .max_by(|a, b| {
                a.1.cmp(&b.1).then_with(|| {
                    // When counts are equal, prefer the smaller value
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .map(|(val, _)| val)
    }
}

impl Default for SimpleImputer {
    fn default() -> Self {
        Self::new(ImputeStrategy::Mean)
    }
}

impl Transformer for SimpleImputer {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_features = x.ncols();
        let mut statistics = Array1::zeros(n_features);
        let mut missing_counts = vec![0usize; n_features];
        let mut all_missing_features = Vec::new();

        for j in 0..n_features {
            let col: Vec<f64> = x.column(j).to_vec();

            // Count missing values
            let missing_count = col.iter().filter(|&&v| self.is_missing(v)).count();
            missing_counts[j] = missing_count;

            // Compute the fill value based on strategy
            let fill = match self.strategy {
                ImputeStrategy::Mean => self.compute_mean(&col),
                ImputeStrategy::Median => self.compute_median(&col),
                ImputeStrategy::MostFrequent => self.compute_mode(&col),
                ImputeStrategy::Constant => Some(self.fill_value),
            };

            match fill {
                Some(v) => {
                    statistics[j] = v;
                }
                None => {
                    // All values are missing - use fill_value as fallback
                    all_missing_features.push(j);
                    statistics[j] = self.fill_value;
                }
            }
        }

        self.statistics = Some(statistics);
        self.n_features_in = Some(n_features);
        self.missing_counts = Some(missing_counts);
        self.all_missing_features = all_missing_features;

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(
            x,
            self.n_features_in
                .ok_or_else(|| FerroError::not_fitted("transform"))?,
        )?;

        let statistics = self
            .statistics
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;
        let n_features = x.ncols();

        // If add_indicator is true, we'll have extra columns
        let n_cols_out = if self.add_indicator {
            n_features * 2
        } else {
            n_features
        };

        let mut result = if self.add_indicator {
            Array2::zeros((x.nrows(), n_cols_out))
        } else {
            x.clone()
        };

        // Fill missing values
        for j in 0..n_features {
            let fill_val = statistics[j];

            for i in 0..x.nrows() {
                let val = x[[i, j]];
                let is_missing = self.is_missing(val);

                if self.add_indicator {
                    result[[i, j]] = if is_missing { fill_val } else { val };
                    result[[i, n_features + j]] = if is_missing { 1.0 } else { 0.0 };
                } else if is_missing {
                    result[[i, j]] = fill_val;
                }
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, _x: &Array2<f64>) -> Result<Array2<f64>> {
        // Imputation is not reversible in general - we've lost information
        // about which values were originally missing
        Err(FerroError::NotImplemented(
            "inverse_transform is not supported for SimpleImputer - imputation is not reversible"
                .to_string(),
        ))
    }

    fn is_fitted(&self) -> bool {
        self.statistics.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features = self.n_features_in?;
        let base_names = input_names
            .map(|names| names.to_vec())
            .unwrap_or_else(|| generate_feature_names(n_features));

        if self.add_indicator {
            let mut names = base_names.clone();
            for name in &base_names {
                names.push(format!("{}_missing", name));
            }
            Some(names)
        } else {
            Some(base_names)
        }
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        if self.add_indicator {
            self.n_features_in.map(|n| n * 2)
        } else {
            self.n_features_in
        }
    }
}

/// Weighting function for KNN imputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KNNWeights {
    /// All neighbors contribute equally.
    Uniform,
    /// Closer neighbors have more influence (weight = 1 / distance).
    Distance,
}

impl Default for KNNWeights {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Distance metric for KNN computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KNNMetric {
    /// Euclidean distance (L2 norm).
    Euclidean,
    /// Manhattan distance (L1 norm).
    Manhattan,
}

impl Default for KNNMetric {
    fn default() -> Self {
        Self::Euclidean
    }
}

/// Imputation using k-nearest neighbors.
///
/// Missing values are imputed using the weighted mean of the k-nearest neighbors
/// found in the training set. A sample's missing values are imputed using the mean
/// (or weighted mean for `Distance` weighting) of the values from the nearest
/// neighbors that have non-missing values for that feature.
///
/// # Algorithm
///
/// For each sample with missing values:
/// 1. Compute distances to all training samples using only features that are
///    present in both samples
/// 2. Select the k nearest neighbors that have non-missing values for the
///    feature being imputed
/// 3. Compute the (weighted) mean of those neighbors' values
///
/// # Configuration
///
/// - `n_neighbors`: Number of neighbors to use (default: 5)
/// - `weights`: Weighting function (`Uniform` or `Distance`)
/// - `metric`: Distance metric (`Euclidean` or `Manhattan`)
///
/// # Edge Cases
///
/// - If a sample has all features missing, column means are used as fallback
/// - If fewer than k neighbors are available, uses all available neighbors
/// - If no neighbors are available, uses column mean
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::imputers::{KNNImputer, KNNWeights};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// let mut imputer = KNNImputer::new(3).with_weights(KNNWeights::Distance);
/// let x = array![
///     [1.0, 2.0, 3.0],
///     [4.0, f64::NAN, 6.0],
///     [7.0, 8.0, 9.0],
///     [10.0, 11.0, 12.0]
/// ];
///
/// let x_imputed = imputer.fit_transform(&x).unwrap();
/// assert!(!x_imputed[[1, 1]].is_nan()); // Was NaN, now imputed
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNImputer {
    /// Number of neighbors to use for imputation
    n_neighbors: usize,

    /// Weighting function for neighbors
    weights: KNNWeights,

    /// Distance metric
    metric: KNNMetric,

    /// Training data (stored after fit for neighbor lookup)
    fit_data: Option<Array2<f64>>,

    /// Column means from training data (fallback for edge cases)
    column_means: Option<Array1<f64>>,

    /// Number of features seen during fit
    n_features_in: Option<usize>,

    /// Number of missing values per feature during fit
    missing_counts: Option<Vec<usize>>,
}

impl KNNImputer {
    /// Create a new KNNImputer with the specified number of neighbors.
    ///
    /// # Arguments
    ///
    /// * `n_neighbors` - Number of nearest neighbors to use (must be >= 1)
    ///
    /// # Panics
    ///
    /// Panics if `n_neighbors` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use ferroml_core::preprocessing::imputers::KNNImputer;
    ///
    /// let imputer = KNNImputer::new(5); // Use 5 neighbors
    /// ```
    pub fn new(n_neighbors: usize) -> Self {
        assert!(n_neighbors >= 1, "n_neighbors must be at least 1");

        Self {
            n_neighbors,
            weights: KNNWeights::default(),
            metric: KNNMetric::default(),
            fit_data: None,
            column_means: None,
            n_features_in: None,
            missing_counts: None,
        }
    }

    /// Set the weighting function for neighbors.
    ///
    /// - `Uniform`: All neighbors contribute equally
    /// - `Distance`: Closer neighbors have more influence (weight = 1/distance)
    ///
    /// # Example
    ///
    /// ```
    /// use ferroml_core::preprocessing::imputers::{KNNImputer, KNNWeights};
    ///
    /// let imputer = KNNImputer::new(5).with_weights(KNNWeights::Distance);
    /// ```
    pub fn with_weights(mut self, weights: KNNWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set the distance metric.
    ///
    /// - `Euclidean`: L2 norm (sqrt of sum of squared differences)
    /// - `Manhattan`: L1 norm (sum of absolute differences)
    ///
    /// # Example
    ///
    /// ```
    /// use ferroml_core::preprocessing::imputers::{KNNImputer, KNNMetric};
    ///
    /// let imputer = KNNImputer::new(5).with_metric(KNNMetric::Manhattan);
    /// ```
    pub fn with_metric(mut self, metric: KNNMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Get the number of neighbors used.
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Get the weighting function.
    pub fn weights(&self) -> KNNWeights {
        self.weights
    }

    /// Get the distance metric.
    pub fn metric(&self) -> KNNMetric {
        self.metric
    }

    /// Get the column means computed during fit.
    pub fn column_means(&self) -> Option<&Array1<f64>> {
        self.column_means.as_ref()
    }

    /// Get the number of missing values per feature during fit.
    pub fn missing_counts(&self) -> Option<&[usize]> {
        self.missing_counts.as_deref()
    }

    /// Check if a value is missing (NaN).
    #[inline]
    fn is_missing(value: f64) -> bool {
        value.is_nan()
    }

    /// Compute distance between two samples, using only shared valid features.
    ///
    /// For Euclidean distance, this implements sklearn's `nan_euclidean_distance`:
    /// the sum of squared differences over valid pairs is scaled by
    /// `n_total_features / n_valid_pairs` before taking the square root.
    /// This accounts for missing features by extrapolating the per-feature
    /// contribution to the full dimensionality.
    ///
    /// Returns `None` if there are no shared valid features.
    fn compute_distance(&self, a: &[f64], b: &[f64]) -> Option<f64> {
        let n_total = a.len();
        let mut sum = 0.0;
        let mut valid_count = 0;

        for (av, bv) in a.iter().zip(b.iter()) {
            if !Self::is_missing(*av) && !Self::is_missing(*bv) {
                let diff = av - bv;
                match self.metric {
                    KNNMetric::Euclidean => sum += diff * diff,
                    KNNMetric::Manhattan => sum += diff.abs(),
                }
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            return None;
        }

        // Scale by n_features / n_valid to match sklearn's nan_euclidean_distance
        let scale = n_total as f64 / valid_count as f64;
        Some(match self.metric {
            KNNMetric::Euclidean => (sum * scale).sqrt(),
            KNNMetric::Manhattan => sum * scale,
        })
    }

    /// Find k nearest neighbors for a sample that have non-missing values for a specific feature.
    ///
    /// Returns a vector of (index, distance) pairs.
    fn find_neighbors(
        &self,
        sample: &[f64],
        feature_idx: usize,
        fit_data: &Array2<f64>,
    ) -> Vec<(usize, f64)> {
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for (i, row) in fit_data.rows().into_iter().enumerate() {
            // Skip samples that are missing the feature we want to impute
            let row_slice: Vec<f64> = row.to_vec();
            if Self::is_missing(row_slice[feature_idx]) {
                continue;
            }

            // Compute distance
            if let Some(dist) = self.compute_distance(sample, &row_slice) {
                candidates.push((i, dist));
            }
        }

        // Sort by distance and take k nearest
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.n_neighbors);

        candidates
    }

    /// Impute a missing value using nearest neighbors.
    fn impute_value(
        &self,
        neighbors: &[(usize, f64)],
        feature_idx: usize,
        fit_data: &Array2<f64>,
        fallback: f64,
    ) -> f64 {
        if neighbors.is_empty() {
            return fallback;
        }

        match self.weights {
            KNNWeights::Uniform => {
                // Simple mean
                let sum: f64 = neighbors
                    .iter()
                    .map(|(idx, _)| fit_data[[*idx, feature_idx]])
                    .sum();
                sum / neighbors.len() as f64
            }
            KNNWeights::Distance => {
                // Distance-weighted mean: weight = 1/distance
                // Handle zero distance (exact match) by using a very small value
                let eps = 1e-10;

                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for (idx, dist) in neighbors {
                    let weight = 1.0 / (dist + eps);
                    weighted_sum += weight * fit_data[[*idx, feature_idx]];
                    weight_sum += weight;
                }

                if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    fallback
                }
            }
        }
    }

    /// Compute column means, ignoring NaN values.
    fn compute_column_means(x: &Array2<f64>) -> Array1<f64> {
        let n_features = x.ncols();
        let mut means = Array1::zeros(n_features);

        for j in 0..n_features {
            let valid: Vec<f64> = x
                .column(j)
                .iter()
                .filter(|&&v| !Self::is_missing(v))
                .copied()
                .collect();

            means[j] = if valid.is_empty() {
                0.0 // Fallback if entire column is missing
            } else {
                valid.iter().sum::<f64>() / valid.len() as f64
            };
        }

        means
    }
}

impl Default for KNNImputer {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Transformer for KNNImputer {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_features = x.ncols();

        // Count missing values per feature
        let mut missing_counts = vec![0usize; n_features];
        for j in 0..n_features {
            missing_counts[j] = x.column(j).iter().filter(|&&v| Self::is_missing(v)).count();
        }

        // Compute column means as fallback
        let column_means = Self::compute_column_means(x);

        // Store the training data for neighbor lookup during transform
        self.fit_data = Some(x.clone());
        self.column_means = Some(column_means);
        self.n_features_in = Some(n_features);
        self.missing_counts = Some(missing_counts);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(
            x,
            self.n_features_in
                .ok_or_else(|| FerroError::not_fitted("transform"))?,
        )?;

        let fit_data = self
            .fit_data
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;
        let column_means = self
            .column_means
            .as_ref()
            .ok_or_else(|| FerroError::not_fitted("transform"))?;
        let n_features = x.ncols();
        let n_samples = x.nrows();

        let mut result = x.clone();

        for i in 0..n_samples {
            let row: Vec<f64> = x.row(i).to_vec();

            // Check which features are missing
            let missing_features: Vec<usize> = row
                .iter()
                .enumerate()
                .filter_map(|(j, &v)| if Self::is_missing(v) { Some(j) } else { None })
                .collect();

            if missing_features.is_empty() {
                continue; // No imputation needed
            }

            // Check if all features are missing - use column means as fallback
            if missing_features.len() == n_features {
                for j in 0..n_features {
                    result[[i, j]] = column_means[j];
                }
                continue;
            }

            // Impute each missing feature
            for &j in &missing_features {
                let neighbors = self.find_neighbors(&row, j, fit_data);
                result[[i, j]] = self.impute_value(&neighbors, j, fit_data, column_means[j]);
            }
        }

        Ok(result)
    }

    fn inverse_transform(&self, _x: &Array2<f64>) -> Result<Array2<f64>> {
        Err(FerroError::NotImplemented(
            "inverse_transform is not supported for KNNImputer - imputation is not reversible"
                .to_string(),
        ))
    }

    fn is_fitted(&self) -> bool {
        self.fit_data.is_some()
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

impl PipelineTransformer for SimpleImputer {
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "SimpleImputer"
    }
}

impl PipelineTransformer for KNNImputer {
    fn clone_boxed(&self) -> Box<dyn PipelineTransformer> {
        Box::new(self.clone())
    }

    fn name(&self) -> &str {
        "KNNImputer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-10;

    // ========== Basic Functionality Tests ==========

    #[test]
    fn test_mean_imputation_basic() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0], [f64::NAN, 4.0], [3.0, f64::NAN]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Column 0: mean of [1.0, 3.0] = 2.0
        // Column 1: mean of [2.0, 4.0] = 3.0
        assert!((x_imputed[[0, 0]] - 1.0).abs() < EPSILON);
        assert!((x_imputed[[1, 0]] - 2.0).abs() < EPSILON); // Was NaN
        assert!((x_imputed[[2, 0]] - 3.0).abs() < EPSILON);

        assert!((x_imputed[[0, 1]] - 2.0).abs() < EPSILON);
        assert!((x_imputed[[1, 1]] - 4.0).abs() < EPSILON);
        assert!((x_imputed[[2, 1]] - 3.0).abs() < EPSILON); // Was NaN
    }

    #[test]
    fn test_median_imputation_basic() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Median);
        let x = array![
            [1.0, 2.0],
            [f64::NAN, 4.0],
            [3.0, f64::NAN],
            [5.0, 6.0],
            [7.0, 8.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Column 0: median of [1.0, 3.0, 5.0, 7.0] = 4.0
        // Column 1: median of [2.0, 4.0, 6.0, 8.0] = 5.0
        assert!((x_imputed[[1, 0]] - 4.0).abs() < EPSILON); // Was NaN
        assert!((x_imputed[[2, 1]] - 5.0).abs() < EPSILON); // Was NaN
    }

    #[test]
    fn test_most_frequent_imputation_basic() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::MostFrequent);
        let x = array![
            [1.0, 2.0],
            [1.0, 4.0],
            [f64::NAN, 4.0],
            [3.0, f64::NAN],
            [1.0, 4.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Column 0: mode of [1.0, 1.0, 3.0, 1.0] = 1.0 (appears 3 times)
        // Column 1: mode of [2.0, 4.0, 4.0, 4.0] = 4.0 (appears 3 times)
        assert!((x_imputed[[2, 0]] - 1.0).abs() < EPSILON); // Was NaN
        assert!((x_imputed[[3, 1]] - 4.0).abs() < EPSILON); // Was NaN
    }

    #[test]
    fn test_constant_imputation_basic() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Constant).with_fill_value(-999.0);
        let x = array![[1.0, 2.0], [f64::NAN, 4.0], [3.0, f64::NAN]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        assert!((x_imputed[[1, 0]] - (-999.0)).abs() < EPSILON); // Was NaN
        assert!((x_imputed[[2, 1]] - (-999.0)).abs() < EPSILON); // Was NaN
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_no_missing_values() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Should be unchanged
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x[[i, j]] - x_imputed[[i, j]]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_all_missing_in_column() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean).with_fill_value(-1.0);
        let x = array![[f64::NAN, 2.0], [f64::NAN, 4.0], [f64::NAN, 6.0]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Column 0: all missing, should use fill_value
        assert!((x_imputed[[0, 0]] - (-1.0)).abs() < EPSILON);
        assert!((x_imputed[[1, 0]] - (-1.0)).abs() < EPSILON);
        assert!((x_imputed[[2, 0]] - (-1.0)).abs() < EPSILON);

        // Column 1: mean = 4.0, no missing values
        assert!((x_imputed[[0, 1]] - 2.0).abs() < EPSILON);

        // Check all_missing_features tracking
        assert_eq!(imputer.all_missing_features(), &[0]);
    }

    #[test]
    fn test_single_valid_value() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![[1.0, f64::NAN], [f64::NAN, f64::NAN], [f64::NAN, f64::NAN]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Column 0: single value 1.0, mean = 1.0
        assert!((x_imputed[[0, 0]] - 1.0).abs() < EPSILON);
        assert!((x_imputed[[1, 0]] - 1.0).abs() < EPSILON);
        assert!((x_imputed[[2, 0]] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_custom_missing_value() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean).with_missing_value(-1.0);
        let x = array![[1.0, 2.0], [-1.0, 4.0], [3.0, -1.0]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // -1.0 is treated as missing
        // Column 0: mean of [1.0, 3.0] = 2.0
        // Column 1: mean of [2.0, 4.0] = 3.0
        assert!((x_imputed[[1, 0]] - 2.0).abs() < EPSILON); // Was -1.0
        assert!((x_imputed[[2, 1]] - 3.0).abs() < EPSILON); // Was -1.0
    }

    // ========== Diagnostics Tests ==========

    #[test]
    fn test_missing_counts() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![
            [f64::NAN, 2.0],
            [f64::NAN, 4.0],
            [3.0, f64::NAN],
            [4.0, 6.0]
        ];

        imputer.fit(&x).unwrap();

        let counts = imputer.missing_counts().unwrap();
        assert_eq!(counts[0], 2); // 2 missing in column 0
        assert_eq!(counts[1], 1); // 1 missing in column 1
    }

    #[test]
    fn test_statistics_accessor() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0], [f64::NAN, 4.0], [3.0, f64::NAN]];

        imputer.fit(&x).unwrap();

        let stats = imputer.statistics().unwrap();
        assert!((stats[0] - 2.0).abs() < EPSILON); // mean of [1.0, 3.0]
        assert!((stats[1] - 3.0).abs() < EPSILON); // mean of [2.0, 4.0]
    }

    // ========== Indicator Feature Tests ==========

    #[test]
    fn test_indicator_columns() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean).with_indicator(true);
        let x = array![[1.0, 2.0], [f64::NAN, 4.0], [3.0, f64::NAN]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Should have 4 columns now (2 original + 2 indicators)
        assert_eq!(x_imputed.ncols(), 4);

        // Check indicator values
        assert!((x_imputed[[0, 2]] - 0.0).abs() < EPSILON); // Not missing
        assert!((x_imputed[[1, 2]] - 1.0).abs() < EPSILON); // Was missing
        assert!((x_imputed[[2, 2]] - 0.0).abs() < EPSILON); // Not missing

        assert!((x_imputed[[0, 3]] - 0.0).abs() < EPSILON); // Not missing
        assert!((x_imputed[[1, 3]] - 0.0).abs() < EPSILON); // Not missing
        assert!((x_imputed[[2, 3]] - 1.0).abs() < EPSILON); // Was missing
    }

    // ========== Transformer Trait Tests ==========

    #[test]
    fn test_not_fitted_error() {
        let imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0]];

        assert!(imputer.transform(&x).is_err());
        assert!(!imputer.is_fitted());
    }

    #[test]
    fn test_shape_mismatch() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x_fit = array![[1.0, 2.0], [3.0, 4.0]];
        imputer.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(imputer.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_inverse_transform_not_supported() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0], [f64::NAN, 4.0]];

        imputer.fit(&x).unwrap();
        let x_imputed = imputer.transform(&x).unwrap();

        // Inverse transform should fail
        assert!(imputer.inverse_transform(&x_imputed).is_err());
    }

    #[test]
    fn test_feature_names_basic() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        imputer.fit(&x).unwrap();

        // Default names
        let names = imputer.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["x0", "x1", "x2"]);

        // Custom names
        let custom = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let names = imputer.get_feature_names_out(Some(&custom)).unwrap();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_feature_names_with_indicator() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean).with_indicator(true);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        imputer.fit(&x).unwrap();

        let names = imputer.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["x0", "x1", "x0_missing", "x1_missing"]);
    }

    #[test]
    fn test_n_features() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        // Before fit
        assert!(imputer.n_features_in().is_none());
        assert!(imputer.n_features_out().is_none());

        // After fit
        imputer.fit(&x).unwrap();
        assert_eq!(imputer.n_features_in(), Some(3));
        assert_eq!(imputer.n_features_out(), Some(3));
    }

    #[test]
    fn test_n_features_with_indicator() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean).with_indicator(true);
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        imputer.fit(&x).unwrap();

        assert_eq!(imputer.n_features_in(), Some(2));
        assert_eq!(imputer.n_features_out(), Some(4)); // 2 original + 2 indicators
    }

    #[test]
    fn test_empty_input() {
        let empty: Array2<f64> = Array2::zeros((0, 0));
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);

        assert!(imputer.fit(&empty).is_err());
    }

    // ========== Median Edge Cases ==========

    #[test]
    fn test_median_even_number_of_values() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Median);
        let x = array![[1.0], [f64::NAN], [3.0], [5.0]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Median of [1.0, 3.0, 5.0] = 3.0 (odd count)
        assert!((x_imputed[[1, 0]] - 3.0).abs() < EPSILON);

        let x2 = array![[1.0], [f64::NAN], [3.0], [5.0], [7.0]];
        let mut imputer2 = SimpleImputer::new(ImputeStrategy::Median);
        let x2_imputed = imputer2.fit_transform(&x2).unwrap();

        // Median of [1.0, 3.0, 5.0, 7.0] = (3.0 + 5.0) / 2 = 4.0 (even count)
        assert!((x2_imputed[[1, 0]] - 4.0).abs() < EPSILON);
    }

    // ========== Mode Edge Cases ==========

    #[test]
    fn test_mode_tie_breaking() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::MostFrequent);
        // Two values appear equally often
        let x = array![[1.0], [1.0], [2.0], [2.0], [f64::NAN]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // When there's a tie, any of the modes is acceptable
        let imputed_val = x_imputed[[4, 0]];
        assert!(imputed_val == 1.0 || imputed_val == 2.0);
    }

    // ========== Fit then transform separately ==========

    #[test]
    fn test_fit_then_transform_new_data() {
        let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
        let x_train = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let x_test = array![[f64::NAN, f64::NAN], [10.0, 20.0]];

        imputer.fit(&x_train).unwrap();
        let x_test_imputed = imputer.transform(&x_test).unwrap();

        // Should use training statistics (mean of [1,3,5]=3, mean of [2,4,6]=4)
        assert!((x_test_imputed[[0, 0]] - 3.0).abs() < EPSILON);
        assert!((x_test_imputed[[0, 1]] - 4.0).abs() < EPSILON);
        assert!((x_test_imputed[[1, 0]] - 10.0).abs() < EPSILON);
        assert!((x_test_imputed[[1, 1]] - 20.0).abs() < EPSILON);
    }

    #[test]
    fn test_default_imputer() {
        let imputer = SimpleImputer::default();
        assert_eq!(imputer.strategy(), ImputeStrategy::Mean);
    }

    // ========== KNNImputer Tests ==========

    #[test]
    fn test_knn_imputer_basic() {
        // Simple case: 4 samples, 1 missing value
        let mut imputer = KNNImputer::new(2);
        let x = array![
            [1.0, 2.0],
            [2.0, f64::NAN], // Missing second feature
            [3.0, 4.0],
            [4.0, 5.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Row 1 missing feature 1
        // Nearest neighbors by feature 0: row 0 (dist 1), row 2 (dist 1)
        // Their values for feature 1: 2.0, 4.0 -> mean = 3.0
        assert!(!x_imputed[[1, 1]].is_nan());
        // The imputed value should be close to 3.0 (average of nearest neighbors)
        assert!((x_imputed[[1, 1]] - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_knn_imputer_uniform_weights() {
        let mut imputer = KNNImputer::new(3).with_weights(KNNWeights::Uniform);
        let x = array![
            [0.0, 10.0],
            [1.0, f64::NAN], // Target row
            [2.0, 20.0],
            [3.0, 30.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // With uniform weights, imputed value is mean of 3 neighbors
        // Neighbors: row 0 (10), row 2 (20), row 3 (30) -> mean = 20
        assert!(!x_imputed[[1, 1]].is_nan());
        assert!((x_imputed[[1, 1]] - 20.0).abs() < EPSILON);
    }

    #[test]
    fn test_knn_imputer_distance_weights() {
        let mut imputer = KNNImputer::new(2).with_weights(KNNWeights::Distance);
        let x = array![
            [0.0, 10.0],     // distance 1 from target
            [1.0, f64::NAN], // Target row
            [100.0, 50.0]    // distance 99 from target
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // With distance weights, closer neighbors have more influence
        // Row 0 is much closer than row 2, so result should be closer to 10 than 50
        assert!(!x_imputed[[1, 1]].is_nan());
        // Value should be heavily weighted toward 10.0
        assert!(x_imputed[[1, 1]] < 15.0);
    }

    #[test]
    fn test_knn_imputer_manhattan_metric() {
        let mut imputer = KNNImputer::new(2).with_metric(KNNMetric::Manhattan);
        let x = array![
            [1.0, 1.0, 10.0],
            [2.0, 2.0, f64::NAN], // Target
            [3.0, 3.0, 30.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Should work with Manhattan distance
        assert!(!x_imputed[[1, 2]].is_nan());
        // With uniform weights (default), mean of 10 and 30 = 20
        assert!((x_imputed[[1, 2]] - 20.0).abs() < EPSILON);
    }

    #[test]
    fn test_knn_imputer_no_missing() {
        let mut imputer = KNNImputer::new(3);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Should be unchanged
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((x[[i, j]] - x_imputed[[i, j]]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_knn_imputer_multiple_missing() {
        let mut imputer = KNNImputer::new(2);
        let x = array![
            [1.0, 2.0, 3.0],
            [f64::NAN, f64::NAN, 6.0], // Two missing features
            [7.0, 8.0, 9.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Both missing values should be filled
        assert!(!x_imputed[[1, 0]].is_nan());
        assert!(!x_imputed[[1, 1]].is_nan());
        // Feature 2 (value 6) unchanged
        assert!((x_imputed[[1, 2]] - 6.0).abs() < EPSILON);
    }

    #[test]
    fn test_knn_imputer_all_row_missing() {
        let mut imputer = KNNImputer::new(2);
        let x = array![
            [1.0, 2.0],
            [f64::NAN, f64::NAN], // All features missing
            [3.0, 4.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Should use column means as fallback
        // Mean of col 0: (1 + 3) / 2 = 2.0
        // Mean of col 1: (2 + 4) / 2 = 3.0
        assert!((x_imputed[[1, 0]] - 2.0).abs() < EPSILON);
        assert!((x_imputed[[1, 1]] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_knn_imputer_not_fitted_error() {
        let imputer = KNNImputer::new(3);
        let x = array![[1.0, 2.0]];

        assert!(imputer.transform(&x).is_err());
        assert!(!imputer.is_fitted());
    }

    #[test]
    fn test_knn_imputer_shape_mismatch() {
        let mut imputer = KNNImputer::new(2);
        let x_fit = array![[1.0, 2.0], [3.0, 4.0]];
        imputer.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]]; // Wrong number of features
        assert!(imputer.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_knn_imputer_inverse_transform_not_supported() {
        let mut imputer = KNNImputer::new(2);
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        imputer.fit(&x).unwrap();
        assert!(imputer.inverse_transform(&x).is_err());
    }

    #[test]
    fn test_knn_imputer_feature_names() {
        let mut imputer = KNNImputer::new(2);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        imputer.fit(&x).unwrap();

        // Default names
        let names = imputer.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["x0", "x1", "x2"]);

        // Custom names
        let custom = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let names = imputer.get_feature_names_out(Some(&custom)).unwrap();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_knn_imputer_n_features() {
        let mut imputer = KNNImputer::new(2);
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        assert!(imputer.n_features_in().is_none());
        assert!(imputer.n_features_out().is_none());

        imputer.fit(&x).unwrap();

        assert_eq!(imputer.n_features_in(), Some(2));
        assert_eq!(imputer.n_features_out(), Some(2));
    }

    #[test]
    fn test_knn_imputer_column_means() {
        let mut imputer = KNNImputer::new(2);
        let x = array![[1.0, 10.0], [f64::NAN, 20.0], [3.0, f64::NAN]];

        imputer.fit(&x).unwrap();

        let means = imputer.column_means().unwrap();
        // Col 0: mean of [1, 3] = 2.0
        // Col 1: mean of [10, 20] = 15.0
        assert!((means[0] - 2.0).abs() < EPSILON);
        assert!((means[1] - 15.0).abs() < EPSILON);
    }

    #[test]
    fn test_knn_imputer_missing_counts() {
        let mut imputer = KNNImputer::new(2);
        let x = array![[f64::NAN, 2.0], [f64::NAN, f64::NAN], [3.0, 4.0]];

        imputer.fit(&x).unwrap();

        let counts = imputer.missing_counts().unwrap();
        assert_eq!(counts[0], 2); // 2 missing in col 0
        assert_eq!(counts[1], 1); // 1 missing in col 1
    }

    #[test]
    fn test_knn_imputer_default() {
        let imputer = KNNImputer::default();
        assert_eq!(imputer.n_neighbors(), 5);
        assert_eq!(imputer.weights(), KNNWeights::Uniform);
        assert_eq!(imputer.metric(), KNNMetric::Euclidean);
    }

    #[test]
    fn test_knn_imputer_accessors() {
        let imputer = KNNImputer::new(7)
            .with_weights(KNNWeights::Distance)
            .with_metric(KNNMetric::Manhattan);

        assert_eq!(imputer.n_neighbors(), 7);
        assert_eq!(imputer.weights(), KNNWeights::Distance);
        assert_eq!(imputer.metric(), KNNMetric::Manhattan);
    }

    #[test]
    fn test_knn_imputer_empty_input() {
        let empty: Array2<f64> = Array2::zeros((0, 0));
        let mut imputer = KNNImputer::new(2);

        assert!(imputer.fit(&empty).is_err());
    }

    #[test]
    fn test_knn_imputer_single_neighbor() {
        // When k=1, should return the value from the single nearest neighbor
        let mut imputer = KNNImputer::new(1);
        let x = array![
            [0.0, 10.0],
            [1.0, f64::NAN], // Target
            [100.0, 50.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Nearest neighbor to row 1 is row 0 (distance 1)
        // So imputed value should be exactly 10.0
        assert!((x_imputed[[1, 1]] - 10.0).abs() < EPSILON);
    }

    #[test]
    fn test_knn_imputer_fit_then_transform_new_data() {
        let mut imputer = KNNImputer::new(2);
        let x_train = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

        imputer.fit(&x_train).unwrap();

        // Transform new data with missing values
        let x_test = array![
            [1.5, f64::NAN], // Between row 0 and 1 in training
            [2.5, f64::NAN]  // Between row 1 and 2 in training
        ];

        let x_test_imputed = imputer.transform(&x_test).unwrap();

        // Row 0: nearest neighbors are rows 0 and 1 (10, 20) -> mean 15
        assert!(!x_test_imputed[[0, 1]].is_nan());
        assert!((x_test_imputed[[0, 1]] - 15.0).abs() < EPSILON);

        // Row 1: nearest neighbors are rows 1 and 2 (20, 30) -> mean 25
        assert!(!x_test_imputed[[1, 1]].is_nan());
        assert!((x_test_imputed[[1, 1]] - 25.0).abs() < EPSILON);
    }

    #[test]
    fn test_knn_imputer_larger_k_than_samples() {
        // When k is larger than available neighbors, use all available
        let mut imputer = KNNImputer::new(10); // k=10, but only 2 complete neighbors
        let x = array![
            [1.0, 10.0],
            [2.0, f64::NAN], // Target
            [3.0, 30.0]
        ];

        let x_imputed = imputer.fit_transform(&x).unwrap();

        // Only 2 neighbors available, so use both
        // Mean of 10 and 30 = 20
        assert!((x_imputed[[1, 1]] - 20.0).abs() < EPSILON);
    }

    #[test]
    #[should_panic(expected = "n_neighbors must be at least 1")]
    fn test_knn_imputer_zero_neighbors_panic() {
        let _ = KNNImputer::new(0);
    }

    #[test]
    fn test_knn_imputer_all_column_missing_in_training() {
        // Edge case: a column is entirely missing during training
        let mut imputer = KNNImputer::new(2);
        let x = array![[1.0, f64::NAN], [2.0, f64::NAN], [3.0, f64::NAN]];

        imputer.fit(&x).unwrap();

        // Column mean fallback when all values missing
        let means = imputer.column_means().unwrap();
        assert!((means[0] - 2.0).abs() < EPSILON); // Mean of [1,2,3]
        assert!((means[1] - 0.0).abs() < EPSILON); // Fallback to 0

        // Transform should use fallback
        let x_test = array![[1.5, f64::NAN]];
        let x_imputed = imputer.transform(&x_test).unwrap();
        assert!((x_imputed[[0, 1]] - 0.0).abs() < EPSILON); // Falls back to 0
    }
}
