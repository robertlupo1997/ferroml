//! Time Series Cross-Validation
//!
//! This module provides cross-validation strategies for time series data that
//! respect temporal ordering and prevent future data leakage.
//!
//! ## Key Properties
//!
//! - Training data always precedes test data in time
//! - No shuffling (would violate temporal ordering)
//! - Expanding or sliding window training sets
//!
//! ## Strategy
//!
//! - [`TimeSeriesSplit`] - K-fold cross-validation for time series with expanding/sliding windows

use super::{CVFold, CrossValidator};
use crate::{FerroError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Time Series Split cross-validator
///
/// Provides train/test indices to split time series data samples that are
/// observed at fixed time intervals, in train/test sets. In each split,
/// test indices must be higher than before, and thus shuffling in cross
/// validator is inappropriate.
///
/// This cross-validation object is a variation of k-fold which returns first
/// `k` folds as train set and the `(k+1)`th fold as test set. Note that unlike
/// standard cross-validation methods, successive training sets are supersets of
/// those that come before them (unless `max_train_size` is specified).
///
/// # Parameters
///
/// - `n_splits`: Number of splits. Must be at least 2.
/// - `max_train_size`: Maximum size for a single training set. If `None`, the training
///   set grows with each split (expanding window). If set, training sets are capped
///   at this size (sliding window).
/// - `test_size`: Number of samples in each test set. If `None`, defaults to
///   `n_samples / (n_splits + 1)`.
/// - `gap`: Number of samples to exclude from the end of each training set before
///   the test set. Useful when observations close in time are not independent.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::cv::{CrossValidator, TimeSeriesSplit};
///
/// // 5 splits for 100 time-ordered samples
/// let cv = TimeSeriesSplit::new(5);
/// let folds = cv.split(100, None, None)?;
///
/// // Training set grows, test sets are consecutive
/// // Fold 0: train=[0..16], test=[16..32]
/// // Fold 1: train=[0..32], test=[32..48]
/// // etc.
///
/// for (i, fold) in folds.iter().enumerate() {
///     println!("Fold {}: train={:?}, test={:?}",
///              i, fold.train_indices.len(), fold.test_indices.len());
/// }
/// ```
///
/// # Example with Gap
///
/// ```ignore
/// use ferroml_core::cv::{CrossValidator, TimeSeriesSplit};
///
/// // With a gap of 5 samples between train and test
/// let cv = TimeSeriesSplit::new(3).with_gap(5);
/// let folds = cv.split(100, None, None)?;
///
/// // The last 5 samples of training are excluded to prevent leakage
/// // from autocorrelated observations
/// ```
///
/// # Notes
///
/// - Unlike KFold, TimeSeriesSplit NEVER shuffles data
/// - Training sets grow over time by default (expanding window)
/// - Use `max_train_size` for sliding window behavior
/// - Use `gap` when neighboring observations are autocorrelated
/// - Suitable for forecasting tasks where past data predicts future
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesSplit {
    /// Number of splits
    n_splits: usize,
    /// Maximum training set size (None = expanding window)
    max_train_size: Option<usize>,
    /// Fixed test set size (None = automatically determined)
    test_size: Option<usize>,
    /// Gap between training and test sets
    gap: usize,
}

impl TimeSeriesSplit {
    /// Create a new TimeSeriesSplit cross-validator
    ///
    /// # Arguments
    ///
    /// * `n_splits` - Number of splits (must be >= 2)
    ///
    /// # Panics
    ///
    /// Panics if n_splits < 2
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ferroml_core::cv::TimeSeriesSplit;
    ///
    /// let cv = TimeSeriesSplit::new(5);
    /// assert_eq!(cv.n_splits(), 5);
    /// ```
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "TimeSeriesSplit requires at least 2 splits");
        Self {
            n_splits,
            max_train_size: None,
            test_size: None,
            gap: 0,
        }
    }

    /// Set the maximum training set size (sliding window)
    ///
    /// When set, the training set will not exceed this size. Earlier samples
    /// will be dropped as newer samples are added, creating a sliding window
    /// effect.
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum number of samples in training set
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ferroml_core::cv::TimeSeriesSplit;
    ///
    /// // Sliding window with max 50 training samples
    /// let cv = TimeSeriesSplit::new(5).with_max_train_size(50);
    /// ```
    pub fn with_max_train_size(mut self, size: usize) -> Self {
        self.max_train_size = Some(size);
        self
    }

    /// Set a fixed test set size
    ///
    /// When set, all test sets will have exactly this many samples.
    /// If `None`, the test size is determined automatically as
    /// `n_samples / (n_splits + 1)`.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of samples in each test set
    pub fn with_test_size(mut self, size: usize) -> Self {
        self.test_size = Some(size);
        self
    }

    /// Set the gap between training and test sets
    ///
    /// The gap excludes a number of samples from the end of the training
    /// set, before the test set begins. This is useful when consecutive
    /// observations are autocorrelated and using them together would
    /// constitute data leakage.
    ///
    /// # Arguments
    ///
    /// * `gap` - Number of samples to exclude between train and test
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ferroml_core::cv::TimeSeriesSplit;
    ///
    /// // Gap of 5 samples to avoid autocorrelation leakage
    /// let cv = TimeSeriesSplit::new(3).with_gap(5);
    /// ```
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Get the number of splits
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Get the maximum training set size
    pub fn max_train_size(&self) -> Option<usize> {
        self.max_train_size
    }

    /// Get the test set size
    pub fn test_size(&self) -> Option<usize> {
        self.test_size
    }

    /// Get the gap size
    pub fn gap(&self) -> usize {
        self.gap
    }
}

impl Default for TimeSeriesSplit {
    fn default() -> Self {
        Self::new(5)
    }
}

impl CrossValidator for TimeSeriesSplit {
    fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<Vec<CVFold>> {
        // Validate inputs
        self.validate_inputs(n_samples, y, groups)?;

        // Determine test size
        let test_size = match self.test_size {
            Some(size) => size,
            None => n_samples / (self.n_splits + 1),
        };

        if test_size == 0 {
            return Err(FerroError::invalid_input(
                "Test size would be zero. Either provide more samples or fewer splits.",
            ));
        }

        // Calculate minimum number of samples needed
        // We need at least: 1 training sample + gap + test_size * n_splits
        let min_samples_needed = 1 + self.gap + test_size * self.n_splits;
        if n_samples < min_samples_needed {
            return Err(FerroError::invalid_input(format!(
                "Not enough samples ({}) for {} splits with test_size={} and gap={}. \
                 Need at least {} samples.",
                n_samples, self.n_splits, test_size, self.gap, min_samples_needed
            )));
        }

        let mut folds = Vec::with_capacity(self.n_splits);

        // Work backwards from the end of the data
        // Last fold ends at n_samples, and we work back from there
        for split_idx in 0..self.n_splits {
            // Calculate test set boundaries for this fold
            // Test sets are consecutive at the end of available data for each fold
            let test_end = n_samples - (self.n_splits - split_idx - 1) * test_size;
            let test_start = test_end - test_size;

            // Training set ends at test_start - gap
            let train_end = if self.gap < test_start {
                test_start - self.gap
            } else {
                return Err(FerroError::invalid_input(format!(
                    "Gap ({}) is too large for split {} where test starts at index {}",
                    self.gap, split_idx, test_start
                )));
            };

            // Training set start depends on max_train_size
            let train_start = match self.max_train_size {
                Some(max_size) => {
                    if train_end > max_size {
                        train_end - max_size
                    } else {
                        0
                    }
                }
                None => 0, // Expanding window: always start from 0
            };

            // Validate we have at least 1 training sample
            if train_start >= train_end {
                return Err(FerroError::invalid_input(format!(
                    "No training samples available for split {}. \
                     Train range would be [{}, {})",
                    split_idx, train_start, train_end
                )));
            }

            let train_indices: Vec<usize> = (train_start..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            folds.push(CVFold::new(train_indices, test_indices, split_idx));
        }

        Ok(folds)
    }

    fn get_n_splits(
        &self,
        _n_samples: Option<usize>,
        _y: Option<&Array1<f64>>,
        _groups: Option<&Array1<i64>>,
    ) -> usize {
        self.n_splits
    }

    fn name(&self) -> &str {
        "TimeSeriesSplit"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_timeseries_split_basic() {
        let cv = TimeSeriesSplit::new(3);
        let folds = cv.split(12, None, None).unwrap();

        assert_eq!(folds.len(), 3);

        // With 12 samples and 3 splits, test_size = 12 / (3+1) = 3
        // Fold 0: test=[3..6], train=[0..3]
        // Fold 1: test=[6..9], train=[0..6]
        // Fold 2: test=[9..12], train=[0..9]

        assert_eq!(folds[0].test_indices, vec![3, 4, 5]);
        assert_eq!(folds[0].train_indices, vec![0, 1, 2]);

        assert_eq!(folds[1].test_indices, vec![6, 7, 8]);
        assert_eq!(folds[1].train_indices, vec![0, 1, 2, 3, 4, 5]);

        assert_eq!(folds[2].test_indices, vec![9, 10, 11]);
        assert_eq!(folds[2].train_indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_timeseries_split_expanding_window() {
        // Default is expanding window (no max_train_size)
        let cv = TimeSeriesSplit::new(4);
        let folds = cv.split(20, None, None).unwrap();

        // test_size = 20 / 5 = 4
        // Training should grow with each fold
        assert!(folds[1].train_indices.len() > folds[0].train_indices.len());
        assert!(folds[2].train_indices.len() > folds[1].train_indices.len());
        assert!(folds[3].train_indices.len() > folds[2].train_indices.len());

        // First training set should start at 0
        assert_eq!(folds[0].train_indices[0], 0);
        assert_eq!(folds[3].train_indices[0], 0);
    }

    #[test]
    fn test_timeseries_split_sliding_window() {
        let cv = TimeSeriesSplit::new(3).with_max_train_size(4);
        let folds = cv.split(15, None, None).unwrap();

        // test_size = 15 / 4 ≈ 3 (integer division)
        // With max_train_size=4, training sets should be capped

        for fold in &folds {
            assert!(
                fold.train_indices.len() <= 4,
                "Training set should not exceed max_train_size"
            );
        }
    }

    #[test]
    fn test_timeseries_split_with_gap() {
        let cv = TimeSeriesSplit::new(3).with_gap(2);
        let folds = cv.split(20, None, None).unwrap();

        // Verify gap between train end and test start
        for fold in &folds {
            let train_max = fold.train_indices.iter().max().unwrap();
            let test_min = fold.test_indices.iter().min().unwrap();
            assert_eq!(
                test_min - train_max - 1,
                2,
                "Gap should be 2 between train end and test start"
            );
        }
    }

    #[test]
    fn test_timeseries_split_fixed_test_size() {
        let cv = TimeSeriesSplit::new(3).with_test_size(5);
        let folds = cv.split(30, None, None).unwrap();

        for fold in &folds {
            assert_eq!(fold.test_indices.len(), 5, "Test size should be exactly 5");
        }
    }

    #[test]
    fn test_timeseries_split_temporal_ordering() {
        let cv = TimeSeriesSplit::new(5);
        let folds = cv.split(60, None, None).unwrap();

        for fold in &folds {
            let train_max = fold.train_indices.iter().max().unwrap();
            let test_min = fold.test_indices.iter().min().unwrap();

            // Training data must come before test data
            assert!(
                train_max < test_min,
                "Training indices must all be less than test indices (no future leakage)"
            );
        }

        // Test sets should be in increasing order across folds
        for i in 1..folds.len() {
            let prev_test_start = folds[i - 1].test_indices[0];
            let curr_test_start = folds[i].test_indices[0];
            assert!(curr_test_start > prev_test_start);
        }
    }

    #[test]
    fn test_timeseries_split_no_overlap() {
        let cv = TimeSeriesSplit::new(4);
        let folds = cv.split(40, None, None).unwrap();

        for fold in &folds {
            let train_set: HashSet<usize> = fold.train_indices.iter().copied().collect();
            let test_set: HashSet<usize> = fold.test_indices.iter().copied().collect();

            assert!(
                train_set.is_disjoint(&test_set),
                "Train and test sets must be disjoint"
            );
        }
    }

    #[test]
    fn test_timeseries_split_test_sets_consecutive() {
        let cv = TimeSeriesSplit::new(4);
        let folds = cv.split(40, None, None).unwrap();

        // Test sets should not overlap (consecutive regions)
        for i in 1..folds.len() {
            let prev_test_max = folds[i - 1].test_indices.iter().max().unwrap();
            let curr_test_min = folds[i].test_indices.iter().min().unwrap();
            assert!(
                curr_test_min > prev_test_max,
                "Test sets across folds should not overlap"
            );
        }
    }

    #[test]
    fn test_timeseries_split_deterministic() {
        let cv = TimeSeriesSplit::new(5);

        let folds1 = cv.split(100, None, None).unwrap();
        let folds2 = cv.split(100, None, None).unwrap();

        // TimeSeriesSplit is deterministic (no shuffling)
        for (f1, f2) in folds1.iter().zip(folds2.iter()) {
            assert_eq!(f1.train_indices, f2.train_indices);
            assert_eq!(f1.test_indices, f2.test_indices);
        }
    }

    #[test]
    fn test_timeseries_split_get_n_splits() {
        let cv = TimeSeriesSplit::new(5);
        assert_eq!(cv.get_n_splits(None, None, None), 5);
        assert_eq!(cv.get_n_splits(Some(100), None, None), 5);
    }

    #[test]
    fn test_timeseries_split_name() {
        let cv = TimeSeriesSplit::new(5);
        assert_eq!(cv.name(), "TimeSeriesSplit");
    }

    #[test]
    fn test_timeseries_split_default() {
        let cv = TimeSeriesSplit::default();
        assert_eq!(cv.n_splits(), 5);
        assert!(cv.max_train_size().is_none());
        assert!(cv.test_size().is_none());
        assert_eq!(cv.gap(), 0);
    }

    #[test]
    fn test_timeseries_split_builder() {
        let cv = TimeSeriesSplit::new(10)
            .with_max_train_size(50)
            .with_test_size(10)
            .with_gap(5);

        assert_eq!(cv.n_splits(), 10);
        assert_eq!(cv.max_train_size(), Some(50));
        assert_eq!(cv.test_size(), Some(10));
        assert_eq!(cv.gap(), 5);
    }

    #[test]
    fn test_timeseries_split_error_too_few_samples() {
        let cv = TimeSeriesSplit::new(5);
        // Need at least 1 + 0 + test_size * 5 samples
        // test_size would be 5 / 6 = 0, which is invalid
        let result = cv.split(5, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_timeseries_split_error_empty_dataset() {
        let cv = TimeSeriesSplit::new(3);
        let result = cv.split(0, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_timeseries_split_error_gap_too_large() {
        let cv = TimeSeriesSplit::new(3).with_gap(100);
        let result = cv.split(20, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_timeseries_split_sklearn_compatibility() {
        // Verify behavior matches sklearn's TimeSeriesSplit
        // sklearn example: n_samples=6, n_splits=3
        // Output:
        //   Fold 0: train=[0], test=[1]
        //   Fold 1: train=[0, 1], test=[2]
        //   Fold 2: train=[0, 1, 2], test=[3]
        //   etc. for larger datasets

        let cv = TimeSeriesSplit::new(3);
        let folds = cv.split(12, None, None).unwrap();

        // test_size = 12 / 4 = 3
        // Fold 0: train up to index 3, test [3,4,5]
        // Fold 1: train up to index 6, test [6,7,8]
        // Fold 2: train up to index 9, test [9,10,11]

        // Check expanding train sets
        assert_eq!(folds[0].train_indices.len(), 3);
        assert_eq!(folds[1].train_indices.len(), 6);
        assert_eq!(folds[2].train_indices.len(), 9);

        // Check consecutive test sets
        assert_eq!(folds[0].test_indices, vec![3, 4, 5]);
        assert_eq!(folds[1].test_indices, vec![6, 7, 8]);
        assert_eq!(folds[2].test_indices, vec![9, 10, 11]);
    }

    #[test]
    fn test_timeseries_split_large_dataset() {
        let cv = TimeSeriesSplit::new(10);
        let folds = cv.split(1000, None, None).unwrap();

        assert_eq!(folds.len(), 10);

        // test_size = 1000 / 11 = 90 (integer division)
        for fold in &folds {
            assert_eq!(fold.test_indices.len(), 90);
        }

        // Verify temporal ordering is maintained
        for fold in &folds {
            let train_max = fold.train_indices.iter().max().unwrap();
            let test_min = fold.test_indices.iter().min().unwrap();
            assert!(train_max < test_min);
        }
    }

    #[test]
    fn test_timeseries_split_sliding_window_behavior() {
        // With sliding window, training size should stabilize
        let cv = TimeSeriesSplit::new(5).with_max_train_size(20);
        let folds = cv.split(100, None, None).unwrap();

        // test_size = 100 / 6 = 16
        // After first few folds, training size should be capped at 20
        for fold in &folds[1..] {
            // Later folds should have training size capped
            assert!(fold.train_indices.len() <= 20);
        }

        // Training sets should "slide" forward
        let train_starts: Vec<usize> = folds.iter().map(|f| f.train_indices[0]).collect();
        for i in 1..train_starts.len() {
            // With sliding window and enough data, train start increases
            assert!(train_starts[i] >= train_starts[i - 1]);
        }
    }

    #[test]
    fn test_timeseries_split_serialization() {
        let cv = TimeSeriesSplit::new(5)
            .with_max_train_size(100)
            .with_test_size(20)
            .with_gap(5);

        let json = serde_json::to_string(&cv).unwrap();
        let cv2: TimeSeriesSplit = serde_json::from_str(&json).unwrap();

        assert_eq!(cv.n_splits(), cv2.n_splits());
        assert_eq!(cv.max_train_size(), cv2.max_train_size());
        assert_eq!(cv.test_size(), cv2.test_size());
        assert_eq!(cv.gap(), cv2.gap());
    }

    #[test]
    fn test_timeseries_combined_options() {
        // Test with all options combined
        let cv = TimeSeriesSplit::new(3)
            .with_max_train_size(10)
            .with_test_size(5)
            .with_gap(2);

        let folds = cv.split(50, None, None).unwrap();

        for fold in &folds {
            // Test size should be exactly 5
            assert_eq!(fold.test_indices.len(), 5);

            // Train size should not exceed 10
            assert!(fold.train_indices.len() <= 10);

            // Gap should be 2
            let train_max = fold.train_indices.iter().max().unwrap();
            let test_min = fold.test_indices.iter().min().unwrap();
            assert_eq!(test_min - train_max - 1, 2);
        }
    }

    #[test]
    fn test_timeseries_minimum_viable_split() {
        // Minimum case that should work: 2 splits, need 3+ samples
        let cv = TimeSeriesSplit::new(2);
        // test_size = 6 / 3 = 2
        // min_samples = 1 + 0 + 2*2 = 5
        let folds = cv.split(6, None, None).unwrap();

        assert_eq!(folds.len(), 2);
        assert!(folds[0].train_indices.len() >= 1);
        assert!(folds[0].test_indices.len() >= 1);
    }
}
