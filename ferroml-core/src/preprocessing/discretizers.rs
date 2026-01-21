//! Feature Discretization Transformers
//!
//! This module provides transformers for discretizing continuous features into bins.
//!
//! ## Available Discretizers
//!
//! - [`KBinsDiscretizer`] - Bin continuous features into discrete intervals
//!
//! ## Binning Strategies
//!
//! | Strategy | Description | Best For |
//! |----------|-------------|----------|
//! | `Uniform` | Equal-width bins | Uniformly distributed data |
//! | `Quantile` | Equal-frequency bins | Arbitrary distributions |
//! | `KMeans` | Bins based on k-means clustering | Data with natural clusters |
//!
//! ## Output Encodings
//!
//! | Encoding | Output | Notes |
//! |----------|--------|-------|
//! | `Ordinal` | Integer bin indices (0, 1, 2, ...) | Preserves order, invertible |
//! | `OneHot` | Sparse-like one-hot (dense array) | For models requiring categorical input |
//!
//! ## Example
//!
//! ```
//! use ferroml_core::preprocessing::discretizers::{KBinsDiscretizer, BinningStrategy, BinEncoding};
//! use ferroml_core::preprocessing::Transformer;
//! use ndarray::array;
//!
//! let mut discretizer = KBinsDiscretizer::new()
//!     .with_n_bins(3)
//!     .with_strategy(BinningStrategy::Quantile);
//!
//! let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
//! let x_binned = discretizer.fit_transform(&x).unwrap();
//!
//! // Each value is now assigned to a bin: 0, 1, or 2
//! assert!(x_binned[[0, 0]] >= 0.0 && x_binned[[0, 0]] < 3.0);
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use super::{check_is_fitted, check_non_empty, check_shape, generate_feature_names, Transformer};
use crate::{FerroError, Result};

/// Strategy for computing bin edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinningStrategy {
    /// Equal-width bins: bins have identical widths.
    /// Bin edges are computed as: min + i * (max - min) / n_bins for i in 0..=n_bins
    Uniform,
    /// Equal-frequency bins: bins have (approximately) the same number of samples.
    /// Bin edges are computed using quantiles.
    Quantile,
    /// K-means based bins: bins are determined by k-means clustering.
    /// Each bin corresponds to a cluster center.
    KMeans,
}

impl Default for BinningStrategy {
    fn default() -> Self {
        Self::Quantile
    }
}

/// Output encoding for discretized features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinEncoding {
    /// Ordinal encoding: each bin is represented by an integer (0, 1, 2, ...).
    /// Output shape: (n_samples, n_features)
    Ordinal,
    /// One-hot encoding: each bin is represented by a binary column.
    /// Output shape: (n_samples, sum(n_bins_per_feature))
    OneHot,
}

impl Default for BinEncoding {
    fn default() -> Self {
        Self::Ordinal
    }
}

/// Discretize continuous features into bins.
///
/// KBinsDiscretizer transforms continuous features into categorical features by
/// dividing them into intervals (bins). This is useful for:
///
/// - Reducing the effect of outliers
/// - Converting continuous data to categorical for certain models
/// - Feature engineering for non-linear relationships
///
/// # Algorithm
///
/// 1. **Fit**: Compute bin edges for each feature based on the strategy:
///    - `Uniform`: Divide range into equal-width bins
///    - `Quantile`: Compute quantiles to get equal-frequency bins
///    - `KMeans`: Run k-means clustering to determine bin boundaries
///
/// 2. **Transform**: Assign each value to its bin using the learned edges.
///    Apply the specified encoding (ordinal or one-hot).
///
/// # Configuration
///
/// - `n_bins`: Number of bins per feature (default: 5)
/// - `strategy`: How to compute bin edges (default: Quantile)
/// - `encode`: Output encoding format (default: Ordinal)
///
/// # Notes
///
/// - Values outside the training range are clipped to the edge bins
/// - Constant features (zero variance) will have only one bin
/// - Quantile strategy may produce fewer bins if there are many duplicate values
///
/// # Example
///
/// ```
/// use ferroml_core::preprocessing::discretizers::{KBinsDiscretizer, BinningStrategy, BinEncoding};
/// use ferroml_core::preprocessing::Transformer;
/// use ndarray::array;
///
/// // Create discretizer with 4 uniform bins
/// let mut discretizer = KBinsDiscretizer::new()
///     .with_n_bins(4)
///     .with_strategy(BinningStrategy::Uniform);
///
/// let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
/// let x_binned = discretizer.fit_transform(&x).unwrap();
///
/// // Values are now bin indices: 0, 1, 2, or 3
/// assert_eq!(x_binned[[0, 0]], 0.0);
/// assert_eq!(x_binned[[4, 0]], 3.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KBinsDiscretizer {
    /// Number of bins per feature
    n_bins: usize,
    /// Strategy for computing bin edges
    strategy: BinningStrategy,
    /// Output encoding format
    encode: BinEncoding,
    /// Bin edges for each feature: bin_edges[j] contains n_bins_[j] + 1 edges
    bin_edges: Option<Vec<Array1<f64>>>,
    /// Actual number of bins per feature (may be less than n_bins for constant features)
    n_bins_per_feature: Option<Vec<usize>>,
    /// Number of features seen during fit
    n_features_in: Option<usize>,
    /// Number of samples seen during fit
    n_samples_seen: Option<usize>,
}

impl KBinsDiscretizer {
    /// Create a new KBinsDiscretizer with default settings.
    ///
    /// Default configuration:
    /// - `n_bins`: 5
    /// - `strategy`: Quantile
    /// - `encode`: Ordinal
    pub fn new() -> Self {
        Self {
            n_bins: 5,
            strategy: BinningStrategy::Quantile,
            encode: BinEncoding::Ordinal,
            bin_edges: None,
            n_bins_per_feature: None,
            n_features_in: None,
            n_samples_seen: None,
        }
    }

    /// Set the number of bins.
    ///
    /// # Arguments
    ///
    /// * `n_bins` - Number of bins per feature (must be >= 2)
    ///
    /// # Panics
    ///
    /// Panics if `n_bins < 2`.
    pub fn with_n_bins(mut self, n_bins: usize) -> Self {
        assert!(n_bins >= 2, "n_bins must be at least 2");
        self.n_bins = n_bins;
        self
    }

    /// Set the binning strategy.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Strategy for computing bin edges
    pub fn with_strategy(mut self, strategy: BinningStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the output encoding.
    ///
    /// # Arguments
    ///
    /// * `encode` - Output encoding format (Ordinal or OneHot)
    pub fn with_encode(mut self, encode: BinEncoding) -> Self {
        self.encode = encode;
        self
    }

    /// Get the number of bins.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Get the binning strategy.
    pub fn strategy(&self) -> BinningStrategy {
        self.strategy
    }

    /// Get the output encoding.
    pub fn encode(&self) -> BinEncoding {
        self.encode
    }

    /// Get the learned bin edges for each feature.
    ///
    /// Returns `None` if not fitted.
    pub fn bin_edges(&self) -> Option<&Vec<Array1<f64>>> {
        self.bin_edges.as_ref()
    }

    /// Get the actual number of bins per feature.
    ///
    /// This may be less than the configured `n_bins` for features with many
    /// duplicate values or constant features.
    ///
    /// Returns `None` if not fitted.
    pub fn n_bins_per_feature(&self) -> Option<&Vec<usize>> {
        self.n_bins_per_feature.as_ref()
    }

    /// Compute uniform bin edges for a column.
    fn compute_uniform_edges(&self, col: &[f64], n_bins: usize) -> Array1<f64> {
        let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10 {
            // Constant feature: return edges that create a single bin
            return Array1::from_vec(vec![min_val - 1e-10, max_val + 1e-10]);
        }

        // Create n_bins + 1 edges
        let edges: Vec<f64> = (0..=n_bins)
            .map(|i| min_val + (max_val - min_val) * i as f64 / n_bins as f64)
            .collect();

        Array1::from_vec(edges)
    }

    /// Compute quantile-based bin edges for a column.
    fn compute_quantile_edges(&self, col: &[f64], n_bins: usize) -> Array1<f64> {
        let mut sorted: Vec<f64> = col.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        if n == 0 {
            return Array1::from_vec(vec![0.0, 1.0]);
        }

        // Compute quantile edges
        let mut edges = Vec::with_capacity(n_bins + 1);
        for i in 0..=n_bins {
            let q = i as f64 / n_bins as f64;
            let pos = q * (n - 1) as f64;
            let lower = pos.floor() as usize;
            let upper = pos.ceil() as usize;
            let frac = pos - lower as f64;

            let edge = if lower == upper || upper >= n {
                sorted[lower.min(n - 1)]
            } else {
                sorted[lower] * (1.0 - frac) + sorted[upper] * frac
            };
            edges.push(edge);
        }

        // Remove duplicate edges (can happen with many identical values)
        let mut unique_edges: Vec<f64> = Vec::with_capacity(edges.len());
        for edge in edges {
            if unique_edges.is_empty() || (edge - *unique_edges.last().unwrap()).abs() > 1e-10 {
                unique_edges.push(edge);
            }
        }

        // Ensure at least 2 edges
        if unique_edges.len() < 2 {
            let min_val = sorted[0];
            let max_val = sorted[n - 1];
            unique_edges = vec![min_val - 1e-10, max_val + 1e-10];
        }

        Array1::from_vec(unique_edges)
    }

    /// Compute k-means based bin edges for a column.
    fn compute_kmeans_edges(&self, col: &[f64], n_bins: usize) -> Array1<f64> {
        let mut data: Vec<f64> = col.to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = data.len();
        if n < n_bins {
            // Not enough unique points for k-means, fall back to quantile
            return self.compute_quantile_edges(col, n_bins);
        }

        let min_val = data[0];
        let max_val = data[n - 1];

        if (max_val - min_val).abs() < 1e-10 {
            // Constant feature
            return Array1::from_vec(vec![min_val - 1e-10, max_val + 1e-10]);
        }

        // Initialize centroids uniformly
        let mut centroids: Vec<f64> = (0..n_bins)
            .map(|i| min_val + (max_val - min_val) * (i as f64 + 0.5) / n_bins as f64)
            .collect();

        // Run k-means for a fixed number of iterations
        let max_iter = 100;
        for _ in 0..max_iter {
            // Assign points to nearest centroid
            let mut cluster_sums = vec![0.0; n_bins];
            let mut cluster_counts = vec![0usize; n_bins];

            for &val in &data {
                let mut best_cluster = 0;
                let mut best_dist = f64::INFINITY;
                for (k, &centroid) in centroids.iter().enumerate() {
                    let dist = (val - centroid).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = k;
                    }
                }
                cluster_sums[best_cluster] += val;
                cluster_counts[best_cluster] += 1;
            }

            // Update centroids
            let mut converged = true;
            for k in 0..n_bins {
                if cluster_counts[k] > 0 {
                    let new_centroid = cluster_sums[k] / cluster_counts[k] as f64;
                    if (new_centroid - centroids[k]).abs() > 1e-8 {
                        converged = false;
                    }
                    centroids[k] = new_centroid;
                }
            }

            if converged {
                break;
            }
        }

        // Sort centroids
        centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Compute bin edges as midpoints between consecutive centroids
        let mut edges = Vec::with_capacity(n_bins + 1);
        edges.push(min_val);

        for i in 0..centroids.len() - 1 {
            edges.push((centroids[i] + centroids[i + 1]) / 2.0);
        }

        edges.push(max_val);

        // Remove duplicate edges
        let mut unique_edges: Vec<f64> = Vec::with_capacity(edges.len());
        for edge in edges {
            if unique_edges.is_empty() || (edge - *unique_edges.last().unwrap()).abs() > 1e-10 {
                unique_edges.push(edge);
            }
        }

        // Ensure at least 2 edges
        if unique_edges.len() < 2 {
            unique_edges = vec![min_val - 1e-10, max_val + 1e-10];
        }

        Array1::from_vec(unique_edges)
    }

    /// Find the bin index for a value given bin edges.
    fn find_bin(&self, value: f64, edges: &Array1<f64>) -> usize {
        let n_edges = edges.len();
        if n_edges < 2 {
            return 0;
        }

        // Clip to valid range
        let n_bins = n_edges - 1;
        if value <= edges[0] {
            return 0;
        }
        if value >= edges[n_edges - 1] {
            return n_bins - 1;
        }

        // Binary search for the correct bin
        let mut lo = 0;
        let mut hi = n_edges - 1;

        while lo < hi {
            let mid = (lo + hi) / 2;
            if edges[mid] <= value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // lo now points to the first edge greater than value
        // The bin index is lo - 1 (but at least 0)
        lo.saturating_sub(1).min(n_bins - 1)
    }

    /// Get the midpoint of a bin (for inverse_transform).
    fn bin_midpoint(&self, bin_idx: usize, edges: &Array1<f64>) -> f64 {
        let n_bins = edges.len() - 1;
        let bin_idx = bin_idx.min(n_bins - 1);
        (edges[bin_idx] + edges[bin_idx + 1]) / 2.0
    }
}

impl Default for KBinsDiscretizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for KBinsDiscretizer {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        check_non_empty(x)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 2 {
            return Err(FerroError::invalid_input(
                "Need at least 2 samples to compute bin edges",
            ));
        }

        let mut bin_edges = Vec::with_capacity(n_features);
        let mut n_bins_per_feature = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col: Vec<f64> = x.column(j).to_vec();

            let edges = match self.strategy {
                BinningStrategy::Uniform => self.compute_uniform_edges(&col, self.n_bins),
                BinningStrategy::Quantile => self.compute_quantile_edges(&col, self.n_bins),
                BinningStrategy::KMeans => self.compute_kmeans_edges(&col, self.n_bins),
            };

            // Number of bins is number of edges minus 1
            let actual_n_bins = edges.len().saturating_sub(1).max(1);
            n_bins_per_feature.push(actual_n_bins);
            bin_edges.push(edges);
        }

        self.bin_edges = Some(bin_edges);
        self.n_bins_per_feature = Some(n_bins_per_feature);
        self.n_features_in = Some(n_features);
        self.n_samples_seen = Some(n_samples);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "transform")?;
        check_shape(x, self.n_features_in.unwrap())?;

        let bin_edges = self.bin_edges.as_ref().unwrap();
        let n_bins_per_feature = self.n_bins_per_feature.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        match self.encode {
            BinEncoding::Ordinal => {
                // Output shape: (n_samples, n_features)
                let mut result = Array2::zeros((n_samples, n_features));

                for j in 0..n_features {
                    let edges = &bin_edges[j];
                    for i in 0..n_samples {
                        let bin_idx = self.find_bin(x[[i, j]], edges);
                        result[[i, j]] = bin_idx as f64;
                    }
                }

                Ok(result)
            }
            BinEncoding::OneHot => {
                // Output shape: (n_samples, sum of n_bins per feature)
                let total_cols: usize = n_bins_per_feature.iter().sum();
                let mut result = Array2::zeros((n_samples, total_cols));

                let mut col_offset = 0;
                for j in 0..n_features {
                    let edges = &bin_edges[j];
                    let n_bins = n_bins_per_feature[j];

                    for i in 0..n_samples {
                        let bin_idx = self.find_bin(x[[i, j]], edges);
                        result[[i, col_offset + bin_idx]] = 1.0;
                    }

                    col_offset += n_bins;
                }

                Ok(result)
            }
        }
    }

    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        check_is_fitted(self.is_fitted(), "inverse_transform")?;

        let bin_edges = self.bin_edges.as_ref().unwrap();
        let n_bins_per_feature = self.n_bins_per_feature.as_ref().unwrap();
        let n_features_in = self.n_features_in.unwrap();

        match self.encode {
            BinEncoding::Ordinal => {
                check_shape(x, n_features_in)?;

                let n_samples = x.nrows();
                let mut result = Array2::zeros((n_samples, n_features_in));

                for j in 0..n_features_in {
                    let edges = &bin_edges[j];
                    for i in 0..n_samples {
                        let bin_idx = x[[i, j]].round() as usize;
                        result[[i, j]] = self.bin_midpoint(bin_idx, edges);
                    }
                }

                Ok(result)
            }
            BinEncoding::OneHot => {
                // For one-hot encoding, we need to decode each block of columns
                let total_cols: usize = n_bins_per_feature.iter().sum();
                if x.ncols() != total_cols {
                    return Err(FerroError::shape_mismatch(
                        format!("({}, {})", x.nrows(), total_cols),
                        format!("({}, {})", x.nrows(), x.ncols()),
                    ));
                }

                let n_samples = x.nrows();
                let mut result = Array2::zeros((n_samples, n_features_in));

                let mut col_offset = 0;
                for j in 0..n_features_in {
                    let edges = &bin_edges[j];
                    let n_bins = n_bins_per_feature[j];

                    for i in 0..n_samples {
                        // Find which bin is "hot" (value == 1)
                        let mut bin_idx = 0;
                        let mut max_val = x[[i, col_offset]];
                        for k in 1..n_bins {
                            if x[[i, col_offset + k]] > max_val {
                                max_val = x[[i, col_offset + k]];
                                bin_idx = k;
                            }
                        }
                        result[[i, j]] = self.bin_midpoint(bin_idx, edges);
                    }

                    col_offset += n_bins;
                }

                Ok(result)
            }
        }
    }

    fn is_fitted(&self) -> bool {
        self.bin_edges.is_some() && self.n_bins_per_feature.is_some()
    }

    fn get_feature_names_out(&self, input_names: Option<&[String]>) -> Option<Vec<String>> {
        let n_features_in = self.n_features_in?;
        let n_bins_per_feature = self.n_bins_per_feature.as_ref()?;

        let base_names = input_names
            .map(|names| names.to_vec())
            .unwrap_or_else(|| generate_feature_names(n_features_in));

        match self.encode {
            BinEncoding::Ordinal => Some(base_names),
            BinEncoding::OneHot => {
                let mut names = Vec::new();
                for (j, base_name) in base_names.iter().enumerate() {
                    for k in 0..n_bins_per_feature[j] {
                        names.push(format!("{}_bin{}", base_name, k));
                    }
                }
                Some(names)
            }
        }
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }

    fn n_features_out(&self) -> Option<usize> {
        match self.encode {
            BinEncoding::Ordinal => self.n_features_in,
            BinEncoding::OneHot => self
                .n_bins_per_feature
                .as_ref()
                .map(|bins| bins.iter().sum()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f64 = 1e-10;

    // ========== Basic Functionality Tests ==========

    #[test]
    fn test_uniform_binning_basic() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(4)
            .with_strategy(BinningStrategy::Uniform);

        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let x_binned = discretizer.fit_transform(&x).unwrap();

        // With 4 uniform bins on [0, 4], bins are [0,1), [1,2), [2,3), [3,4]
        assert_eq!(x_binned[[0, 0]], 0.0); // 0.0 -> bin 0
        assert_eq!(x_binned[[4, 0]], 3.0); // 4.0 -> bin 3 (last bin)
    }

    #[test]
    fn test_quantile_binning_basic() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(4)
            .with_strategy(BinningStrategy::Quantile);

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let x_binned = discretizer.fit_transform(&x).unwrap();

        // Quantile binning should distribute samples roughly equally across bins
        // With 8 samples and 4 bins, expect ~2 samples per bin
        let mut bin_counts = vec![0; 4];
        for i in 0..8 {
            let bin = x_binned[[i, 0]] as usize;
            bin_counts[bin] += 1;
        }

        // Each bin should have 2 samples
        for count in bin_counts {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_kmeans_binning_basic() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_strategy(BinningStrategy::KMeans);

        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [10.0],
            [11.0],
            [12.0],
            [20.0],
            [21.0],
            [22.0]
        ];
        let x_binned = discretizer.fit_transform(&x).unwrap();

        // K-means should find 3 clusters around 2, 11, and 21
        // Values 1,2,3 should be in one bin, 10,11,12 in another, 20,21,22 in another
        let bin_0 = x_binned[[0, 0]] as usize;
        let bin_1 = x_binned[[1, 0]] as usize;
        let bin_2 = x_binned[[2, 0]] as usize;
        let bin_3 = x_binned[[3, 0]] as usize;
        let bin_6 = x_binned[[6, 0]] as usize;

        // First three should be in same bin
        assert_eq!(bin_0, bin_1);
        assert_eq!(bin_1, bin_2);

        // Different clusters should be in different bins
        assert_ne!(bin_0, bin_3);
        assert_ne!(bin_3, bin_6);
    }

    // ========== Encoding Tests ==========

    #[test]
    fn test_ordinal_encoding() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_encode(BinEncoding::Ordinal);

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let x_binned = discretizer.fit_transform(&x).unwrap();

        // Should have same shape as input
        assert_eq!(x_binned.shape(), &[6, 1]);

        // All values should be in [0, n_bins-1]
        for val in x_binned.iter() {
            assert!(*val >= 0.0 && *val < 3.0);
        }
    }

    #[test]
    fn test_onehot_encoding() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_encode(BinEncoding::OneHot);

        let x = array![[1.0], [3.0], [5.0]];
        let x_binned = discretizer.fit_transform(&x).unwrap();

        // Should have shape (n_samples, n_bins)
        assert_eq!(x_binned.shape(), &[3, 3]);

        // Each row should have exactly one 1.0
        for i in 0..3 {
            let row_sum: f64 = x_binned.row(i).sum();
            assert!((row_sum - 1.0).abs() < EPSILON);
        }

        // Values should be 0 or 1
        for val in x_binned.iter() {
            assert!(*val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_onehot_multiple_features() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_encode(BinEncoding::OneHot);

        let x = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let x_binned = discretizer.fit_transform(&x).unwrap();

        // Should have shape (n_samples, n_bins * n_features) = (3, 6)
        assert_eq!(x_binned.shape(), &[3, 6]);

        // Each row should have exactly 2 ones (one for each feature)
        for i in 0..3 {
            let row_sum: f64 = x_binned.row(i).sum();
            assert!((row_sum - 2.0).abs() < EPSILON);
        }
    }

    // ========== Inverse Transform Tests ==========

    #[test]
    fn test_inverse_transform_ordinal() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(5)
            .with_strategy(BinningStrategy::Uniform)
            .with_encode(BinEncoding::Ordinal);

        let x = array![[0.0], [2.5], [5.0], [7.5], [10.0]];
        discretizer.fit(&x).unwrap();

        let x_binned = discretizer.transform(&x).unwrap();
        let x_recovered = discretizer.inverse_transform(&x_binned).unwrap();

        // Inverse transform returns bin midpoints, so not exact but close
        // Check that recovered values are in the right bins
        let x_rebinned = discretizer.transform(&x_recovered).unwrap();

        for i in 0..5 {
            assert_eq!(x_binned[[i, 0]], x_rebinned[[i, 0]]);
        }
    }

    #[test]
    fn test_inverse_transform_onehot() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_strategy(BinningStrategy::Uniform)
            .with_encode(BinEncoding::OneHot);

        let x = array![[1.0], [3.0], [5.0]];
        discretizer.fit(&x).unwrap();

        let x_binned = discretizer.transform(&x).unwrap();
        let x_recovered = discretizer.inverse_transform(&x_binned).unwrap();

        // Recovered values should map back to same bins
        let x_rebinned = discretizer.transform(&x_recovered).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!((x_binned[[i, j]] - x_rebinned[[i, j]]).abs() < EPSILON);
            }
        }
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_constant_feature() {
        let mut discretizer = KBinsDiscretizer::new().with_n_bins(5);

        let x = array![[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0], [5.0, 5.0]];
        discretizer.fit(&x).unwrap();

        let n_bins_per_feature = discretizer.n_bins_per_feature().unwrap();

        // Constant feature should have only 1 bin
        assert_eq!(n_bins_per_feature[0], 1);

        // Non-constant feature should have requested bins
        assert!(n_bins_per_feature[1] >= 1);

        // Transform should work
        let x_binned = discretizer.transform(&x).unwrap();

        // All values of constant feature should be in bin 0
        for i in 0..5 {
            assert_eq!(x_binned[[i, 0]], 0.0);
        }
    }

    #[test]
    fn test_values_outside_training_range() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_strategy(BinningStrategy::Uniform);

        let x_train = array![[2.0], [3.0], [4.0], [5.0], [6.0]];
        discretizer.fit(&x_train).unwrap();

        // Test values outside training range
        let x_test = array![[0.0], [1.0], [4.0], [8.0], [10.0]];
        let x_binned = discretizer.transform(&x_test).unwrap();

        // Values below min should be in bin 0
        assert_eq!(x_binned[[0, 0]], 0.0);
        assert_eq!(x_binned[[1, 0]], 0.0);

        // Values above max should be in last bin
        let n_bins = discretizer.n_bins_per_feature().unwrap()[0];
        assert_eq!(x_binned[[3, 0]], (n_bins - 1) as f64);
        assert_eq!(x_binned[[4, 0]], (n_bins - 1) as f64);
    }

    #[test]
    fn test_duplicate_values_quantile() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(5)
            .with_strategy(BinningStrategy::Quantile);

        // Many duplicate values
        let x = array![[1.0], [1.0], [1.0], [1.0], [5.0], [5.0], [5.0], [5.0]];
        discretizer.fit(&x).unwrap();

        // Should still work, may have fewer bins due to duplicates
        let x_binned = discretizer.transform(&x).unwrap();

        // First half should be in same bin
        let bin_0 = x_binned[[0, 0]];
        for i in 1..4 {
            assert_eq!(x_binned[[i, 0]], bin_0);
        }

        // Second half should be in same bin (different from first)
        let bin_4 = x_binned[[4, 0]];
        for i in 5..8 {
            assert_eq!(x_binned[[i, 0]], bin_4);
        }
    }

    #[test]
    fn test_empty_input() {
        let empty: Array2<f64> = Array2::zeros((0, 0));
        let mut discretizer = KBinsDiscretizer::new();
        assert!(discretizer.fit(&empty).is_err());
    }

    #[test]
    fn test_single_sample() {
        let mut discretizer = KBinsDiscretizer::new();
        let x = array![[1.0, 2.0]];
        assert!(discretizer.fit(&x).is_err());
    }

    // ========== Configuration Tests ==========

    #[test]
    fn test_default_config() {
        let discretizer = KBinsDiscretizer::new();
        assert_eq!(discretizer.n_bins(), 5);
        assert_eq!(discretizer.strategy(), BinningStrategy::Quantile);
        assert_eq!(discretizer.encode(), BinEncoding::Ordinal);
    }

    #[test]
    fn test_builder_pattern() {
        let discretizer = KBinsDiscretizer::new()
            .with_n_bins(10)
            .with_strategy(BinningStrategy::Uniform)
            .with_encode(BinEncoding::OneHot);

        assert_eq!(discretizer.n_bins(), 10);
        assert_eq!(discretizer.strategy(), BinningStrategy::Uniform);
        assert_eq!(discretizer.encode(), BinEncoding::OneHot);
    }

    #[test]
    #[should_panic(expected = "n_bins must be at least 2")]
    fn test_invalid_n_bins() {
        KBinsDiscretizer::new().with_n_bins(1);
    }

    // ========== Feature Names Tests ==========

    #[test]
    fn test_feature_names_ordinal() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_encode(BinEncoding::Ordinal);

        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        discretizer.fit(&x).unwrap();

        // Default names
        let names = discretizer.get_feature_names_out(None).unwrap();
        assert_eq!(names, vec!["x0", "x1", "x2"]);

        // Custom names
        let custom = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let names = discretizer.get_feature_names_out(Some(&custom)).unwrap();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_feature_names_onehot() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_encode(BinEncoding::OneHot);

        let x = array![[1.0, 10.0], [5.0, 50.0], [9.0, 90.0]];
        discretizer.fit(&x).unwrap();

        // Default names
        let names = discretizer.get_feature_names_out(None).unwrap();
        assert!(names.len() == 6); // 3 bins * 2 features
        assert!(names[0].starts_with("x0_bin"));
        assert!(names[3].starts_with("x1_bin"));

        // Custom names
        let custom = vec!["age".to_string(), "income".to_string()];
        let names = discretizer.get_feature_names_out(Some(&custom)).unwrap();
        assert!(names[0].starts_with("age_bin"));
        assert!(names[3].starts_with("income_bin"));
    }

    // ========== Fitted State Tests ==========

    #[test]
    fn test_not_fitted() {
        let discretizer = KBinsDiscretizer::new();
        let x = array![[1.0, 2.0]];

        assert!(discretizer.transform(&x).is_err());
        assert!(!discretizer.is_fitted());
    }

    #[test]
    fn test_shape_mismatch() {
        let mut discretizer = KBinsDiscretizer::new();
        let x_fit = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        discretizer.fit(&x_fit).unwrap();

        let x_wrong = array![[1.0, 2.0, 3.0]]; // 3 features instead of 2
        assert!(discretizer.transform(&x_wrong).is_err());
    }

    #[test]
    fn test_n_features() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_encode(BinEncoding::Ordinal);

        // Before fit
        assert!(discretizer.n_features_in().is_none());
        assert!(discretizer.n_features_out().is_none());

        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        discretizer.fit(&x).unwrap();

        // After fit - ordinal keeps same number of features
        assert_eq!(discretizer.n_features_in(), Some(3));
        assert_eq!(discretizer.n_features_out(), Some(3));
    }

    #[test]
    fn test_n_features_onehot() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(4)
            .with_encode(BinEncoding::OneHot);

        let x = array![[1.0, 10.0], [5.0, 50.0], [9.0, 90.0]];
        discretizer.fit(&x).unwrap();

        // One-hot expands features
        assert_eq!(discretizer.n_features_in(), Some(2));
        assert_eq!(discretizer.n_features_out(), Some(8)); // 4 bins * 2 features
    }

    // ========== Serialization Tests ==========

    #[test]
    fn test_serialization() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(4)
            .with_strategy(BinningStrategy::Uniform)
            .with_encode(BinEncoding::OneHot);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        discretizer.fit(&x).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&discretizer).unwrap();
        assert!(!json.is_empty());

        // Deserialize
        let restored: KBinsDiscretizer = serde_json::from_str(&json).unwrap();
        assert!(restored.is_fitted());
        assert_eq!(restored.n_bins(), 4);
        assert_eq!(restored.strategy(), BinningStrategy::Uniform);
        assert_eq!(restored.encode(), BinEncoding::OneHot);
    }

    // ========== Multiple Features Tests ==========

    #[test]
    fn test_multiple_features_uniform() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_strategy(BinningStrategy::Uniform);

        let x = array![
            [0.0, 0.0, 0.0],
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0]
        ];

        let x_binned = discretizer.fit_transform(&x).unwrap();
        assert_eq!(x_binned.shape(), &[4, 3]);

        // All features should have consistent binning behavior
        // First row (all minimums) should be in bin 0
        assert_eq!(x_binned[[0, 0]], 0.0);
        assert_eq!(x_binned[[0, 1]], 0.0);
        assert_eq!(x_binned[[0, 2]], 0.0);
    }

    #[test]
    fn test_monotonic_bins() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(5)
            .with_strategy(BinningStrategy::Uniform);

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
        let x_binned = discretizer.fit_transform(&x).unwrap();

        // Bins should be non-decreasing for increasing input
        for i in 0..x.nrows() - 1 {
            assert!(
                x_binned[[i + 1, 0]] >= x_binned[[i, 0]],
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    // ========== Regression Tests ==========

    #[test]
    fn test_bins_are_integers() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(5)
            .with_encode(BinEncoding::Ordinal);

        let x = array![[1.5], [2.7], [3.2], [4.9], [5.1]];
        let x_binned = discretizer.fit_transform(&x).unwrap();

        // All bin indices should be whole numbers
        for val in x_binned.iter() {
            assert!(
                (val - val.round()).abs() < EPSILON,
                "Bin value {} is not an integer",
                val
            );
        }
    }

    #[test]
    fn test_bin_edges_stored() {
        let mut discretizer = KBinsDiscretizer::new()
            .with_n_bins(3)
            .with_strategy(BinningStrategy::Uniform);

        let x = array![[0.0], [3.0], [6.0]];
        discretizer.fit(&x).unwrap();

        let edges = discretizer.bin_edges().unwrap();
        assert_eq!(edges.len(), 1); // One feature

        let feature_edges = &edges[0];
        assert_eq!(feature_edges.len(), 4); // n_bins + 1 = 4 edges

        // Edges should be 0, 2, 4, 6 for uniform binning
        assert!((feature_edges[0] - 0.0).abs() < EPSILON);
        assert!((feature_edges[3] - 6.0).abs() < EPSILON);
    }
}
