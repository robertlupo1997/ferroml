//! K-Fold Cross-Validation Strategies
//!
//! This module provides standard k-fold cross-validation and its variants.
//!
//! ## Strategies
//!
//! - [`KFold`] - Standard k-fold cross-validation
//! - [`RepeatedKFold`] - Multiple repetitions of k-fold with different random shuffles

use super::{shuffle_indices, validate_n_folds, CVFold, CrossValidator};
use crate::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// K-Fold cross-validator
///
/// Divides all the samples into `n_folds` groups (folds) of approximately equal size.
/// Each fold is used once as validation while the remaining folds form the training set.
///
/// # Parameters
///
/// - `n_folds`: Number of folds. Must be at least 2.
/// - `shuffle`: Whether to shuffle data before splitting. Default: false.
/// - `random_seed`: Seed for the random number generator (only used if shuffle=true).
///
/// # Example
///
/// ```
/// use ferroml_core::cv::{CrossValidator, KFold};
///
/// // 5-fold CV with shuffling
/// let cv = KFold::new(5).with_shuffle(true).with_seed(42);
/// let folds = cv.split(100, None, None).unwrap();
///
/// assert_eq!(folds.len(), 5);
/// for fold in &folds {
///     // Each fold has ~20% of samples for testing
///     assert_eq!(fold.test_indices.len(), 20);
/// }
/// ```
///
/// # Notes
///
/// - The first `n_samples % n_folds` folds have size `n_samples // n_folds + 1`,
///   other folds have size `n_samples // n_folds`.
/// - Shuffling is recommended to ensure random sampling and reduce variance.
/// - For stratified splitting (preserving class proportions), use `StratifiedKFold`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KFold {
    /// Number of folds
    n_folds: usize,
    /// Whether to shuffle the data before splitting
    shuffle: bool,
    /// Random seed for shuffling
    random_seed: Option<u64>,
}

impl KFold {
    /// Create a new KFold cross-validator
    ///
    /// # Arguments
    ///
    /// * `n_folds` - Number of folds (must be >= 2)
    ///
    /// # Panics
    ///
    /// Panics if n_folds < 2
    pub fn new(n_folds: usize) -> Self {
        assert!(n_folds >= 2, "KFold requires at least 2 folds");
        Self {
            n_folds,
            shuffle: false,
            random_seed: None,
        }
    }

    /// Enable shuffling before splitting
    ///
    /// When shuffle is enabled, data is randomly permuted before splitting
    /// into folds. This helps ensure that folds are representative of the
    /// overall distribution.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the random seed for shuffling
    ///
    /// Setting a seed ensures reproducible splits across runs.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Get the number of folds
    pub fn n_folds(&self) -> usize {
        self.n_folds
    }

    /// Check if shuffling is enabled
    pub fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Get the random seed (if set)
    pub fn random_seed(&self) -> Option<u64> {
        self.random_seed
    }
}

impl Default for KFold {
    fn default() -> Self {
        Self::new(5)
    }
}

impl CrossValidator for KFold {
    fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<Vec<CVFold>> {
        // Validate inputs
        self.validate_inputs(n_samples, y, groups)?;
        validate_n_folds(self.n_folds, n_samples)?;

        // Create indices array
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if requested
        if self.shuffle {
            let seed = self.random_seed.unwrap_or(0);
            shuffle_indices(&mut indices, seed);
        }

        // Calculate fold sizes
        // First n_samples % n_folds folds have size (n_samples / n_folds) + 1
        // Remaining folds have size n_samples / n_folds
        let fold_sizes: Vec<usize> = (0..self.n_folds)
            .map(|i| {
                let base_size = n_samples / self.n_folds;
                if i < n_samples % self.n_folds {
                    base_size + 1
                } else {
                    base_size
                }
            })
            .collect();

        // Build folds
        let mut folds = Vec::with_capacity(self.n_folds);
        let mut current = 0;

        for (fold_idx, &fold_size) in fold_sizes.iter().enumerate() {
            // Test indices for this fold
            let test_indices: Vec<usize> = indices[current..current + fold_size].to_vec();

            // Train indices are everything except test
            let train_indices: Vec<usize> = indices[..current]
                .iter()
                .chain(indices[current + fold_size..].iter())
                .copied()
                .collect();

            folds.push(CVFold::new(train_indices, test_indices, fold_idx));
            current += fold_size;
        }

        Ok(folds)
    }

    fn get_n_splits(
        &self,
        _n_samples: Option<usize>,
        _y: Option<&Array1<f64>>,
        _groups: Option<&Array1<i64>>,
    ) -> usize {
        self.n_folds
    }

    fn name(&self) -> &str {
        "KFold"
    }
}

/// Repeated K-Fold cross-validator
///
/// Repeats K-Fold `n_repeats` times with different random seeds.
/// This provides more robust estimates of model performance by averaging
/// over multiple random splits.
///
/// # Parameters
///
/// - `n_folds`: Number of folds for each repetition. Must be at least 2.
/// - `n_repeats`: Number of times to repeat the k-fold. Default: 10.
/// - `random_seed`: Base seed for the random number generator.
///
/// # Example
///
/// ```
/// use ferroml_core::cv::{CrossValidator, RepeatedKFold};
///
/// // 5-fold CV repeated 10 times = 50 total folds
/// let cv = RepeatedKFold::new(5, 10).with_seed(42);
/// let folds = cv.split(100, None, None).unwrap();
///
/// assert_eq!(folds.len(), 50);
/// ```
///
/// # Statistical Notes
///
/// - Repeated k-fold provides more stable performance estimates
/// - The variance of the estimate decreases with more repeats
/// - However, folds across repeats are not independent, so CIs should
///   account for this (FerroML's CVResult uses corrected formulas)
/// - Typical values: 5-fold repeated 10 times or 10-fold repeated 5 times
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatedKFold {
    /// Number of folds per repetition
    n_folds: usize,
    /// Number of repetitions
    n_repeats: usize,
    /// Base random seed
    random_seed: Option<u64>,
}

impl RepeatedKFold {
    /// Create a new RepeatedKFold cross-validator
    ///
    /// # Arguments
    ///
    /// * `n_folds` - Number of folds per repetition (must be >= 2)
    /// * `n_repeats` - Number of times to repeat the k-fold (must be >= 1)
    ///
    /// # Panics
    ///
    /// Panics if n_folds < 2 or n_repeats < 1
    pub fn new(n_folds: usize, n_repeats: usize) -> Self {
        assert!(n_folds >= 2, "RepeatedKFold requires at least 2 folds");
        assert!(n_repeats >= 1, "RepeatedKFold requires at least 1 repeat");
        Self {
            n_folds,
            n_repeats,
            random_seed: None,
        }
    }

    /// Set the random seed
    ///
    /// The seed is used as a base; each repetition uses `seed + repeat_idx`
    /// to ensure different shuffles while maintaining reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Get the number of folds per repetition
    pub fn n_folds(&self) -> usize {
        self.n_folds
    }

    /// Get the number of repetitions
    pub fn n_repeats(&self) -> usize {
        self.n_repeats
    }

    /// Get the random seed (if set)
    pub fn random_seed(&self) -> Option<u64> {
        self.random_seed
    }
}

impl Default for RepeatedKFold {
    fn default() -> Self {
        Self::new(5, 10)
    }
}

impl CrossValidator for RepeatedKFold {
    fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<Vec<CVFold>> {
        // Validate inputs
        self.validate_inputs(n_samples, y, groups)?;
        validate_n_folds(self.n_folds, n_samples)?;

        let base_seed = self.random_seed.unwrap_or(0);
        let mut all_folds = Vec::with_capacity(self.n_folds * self.n_repeats);

        for repeat_idx in 0..self.n_repeats {
            // Create a KFold for this repetition with a different seed
            let kfold = KFold::new(self.n_folds)
                .with_shuffle(true)
                .with_seed(base_seed.wrapping_add(repeat_idx as u64));

            let folds = kfold.split(n_samples, y, groups)?;

            // Add folds with updated indices to track repetition
            for (fold_in_repeat, fold) in folds.into_iter().enumerate() {
                let global_fold_idx = repeat_idx * self.n_folds + fold_in_repeat;
                all_folds.push(CVFold::new(
                    fold.train_indices,
                    fold.test_indices,
                    global_fold_idx,
                ));
            }
        }

        Ok(all_folds)
    }

    fn get_n_splits(
        &self,
        _n_samples: Option<usize>,
        _y: Option<&Array1<f64>>,
        _groups: Option<&Array1<i64>>,
    ) -> usize {
        self.n_folds * self.n_repeats
    }

    fn name(&self) -> &str {
        "RepeatedKFold"
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
    fn test_kfold_basic() {
        let cv = KFold::new(5);
        let folds = cv.split(100, None, None).unwrap();

        assert_eq!(folds.len(), 5);

        // Each sample should appear in exactly one test set
        let mut all_test: HashSet<usize> = HashSet::new();
        for fold in &folds {
            for &idx in &fold.test_indices {
                assert!(
                    all_test.insert(idx),
                    "Index {} appears in multiple folds",
                    idx
                );
            }
        }
        assert_eq!(all_test.len(), 100);

        // Each fold should have ~20 test samples
        for fold in &folds {
            assert_eq!(fold.test_indices.len(), 20);
            assert_eq!(fold.train_indices.len(), 80);
        }
    }

    #[test]
    fn test_kfold_uneven_split() {
        // 103 samples, 5 folds
        // First 3 folds get 21 samples, last 2 get 20
        let cv = KFold::new(5);
        let folds = cv.split(103, None, None).unwrap();

        let sizes: Vec<usize> = folds.iter().map(|f| f.test_indices.len()).collect();
        assert_eq!(sizes, vec![21, 21, 21, 20, 20]);

        // Total should still be 103
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 103);
    }

    #[test]
    fn test_kfold_shuffle_deterministic() {
        let cv = KFold::new(5).with_shuffle(true).with_seed(42);

        let folds1 = cv.split(100, None, None).unwrap();
        let folds2 = cv.split(100, None, None).unwrap();

        // Same seed should produce same splits
        for (f1, f2) in folds1.iter().zip(folds2.iter()) {
            assert_eq!(f1.test_indices, f2.test_indices);
            assert_eq!(f1.train_indices, f2.train_indices);
        }
    }

    #[test]
    fn test_kfold_shuffle_different_seeds() {
        let cv1 = KFold::new(5).with_shuffle(true).with_seed(42);
        let cv2 = KFold::new(5).with_shuffle(true).with_seed(123);

        let folds1 = cv1.split(100, None, None).unwrap();
        let folds2 = cv2.split(100, None, None).unwrap();

        // Different seeds should (very likely) produce different splits
        let same = folds1
            .iter()
            .zip(folds2.iter())
            .all(|(f1, f2)| f1.test_indices == f2.test_indices);
        assert!(!same);
    }

    #[test]
    fn test_kfold_no_shuffle() {
        let cv = KFold::new(5);
        let folds = cv.split(100, None, None).unwrap();

        // Without shuffle, indices should be sequential
        assert_eq!(folds[0].test_indices, (0..20).collect::<Vec<_>>());
        assert_eq!(folds[1].test_indices, (20..40).collect::<Vec<_>>());
        assert_eq!(folds[2].test_indices, (40..60).collect::<Vec<_>>());
        assert_eq!(folds[3].test_indices, (60..80).collect::<Vec<_>>());
        assert_eq!(folds[4].test_indices, (80..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_kfold_minimum_samples() {
        // Minimum case: 2 samples, 2 folds
        let cv = KFold::new(2);
        let folds = cv.split(2, None, None).unwrap();

        assert_eq!(folds.len(), 2);
        assert_eq!(folds[0].test_indices.len(), 1);
        assert_eq!(folds[0].train_indices.len(), 1);
    }

    #[test]
    fn test_kfold_error_too_many_folds() {
        let cv = KFold::new(10);
        let result = cv.split(5, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_kfold_error_empty_dataset() {
        let cv = KFold::new(5);
        let result = cv.split(0, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_kfold_get_n_splits() {
        let cv = KFold::new(5);
        assert_eq!(cv.get_n_splits(None, None, None), 5);
        assert_eq!(cv.get_n_splits(Some(100), None, None), 5);
    }

    #[test]
    fn test_kfold_name() {
        let cv = KFold::new(5);
        assert_eq!(cv.name(), "KFold");
    }

    #[test]
    fn test_kfold_default() {
        let cv = KFold::default();
        assert_eq!(cv.n_folds(), 5);
        assert!(!cv.shuffle());
        assert!(cv.random_seed().is_none());
    }

    #[test]
    fn test_kfold_builder() {
        let cv = KFold::new(10).with_shuffle(true).with_seed(42);
        assert_eq!(cv.n_folds(), 10);
        assert!(cv.shuffle());
        assert_eq!(cv.random_seed(), Some(42));
    }

    // RepeatedKFold tests

    #[test]
    fn test_repeated_kfold_basic() {
        let cv = RepeatedKFold::new(5, 3);
        let folds = cv.split(100, None, None).unwrap();

        // 5 folds × 3 repeats = 15 total
        assert_eq!(folds.len(), 15);

        // Check fold indices are sequential
        for (i, fold) in folds.iter().enumerate() {
            assert_eq!(fold.fold_index, i);
        }
    }

    #[test]
    fn test_repeated_kfold_coverage_per_repeat() {
        let cv = RepeatedKFold::new(5, 2).with_seed(42);
        let folds = cv.split(100, None, None).unwrap();

        // Check each repetition covers all samples
        for repeat_idx in 0..2 {
            let start = repeat_idx * 5;
            let end = start + 5;
            let repeat_folds = &folds[start..end];

            let mut covered: HashSet<usize> = HashSet::new();
            for fold in repeat_folds {
                for &idx in &fold.test_indices {
                    assert!(covered.insert(idx));
                }
            }
            assert_eq!(covered.len(), 100);
        }
    }

    #[test]
    fn test_repeated_kfold_different_shuffles() {
        let cv = RepeatedKFold::new(5, 2).with_seed(42);
        let folds = cv.split(100, None, None).unwrap();

        // First repeat and second repeat should have different orderings
        // They might have some overlap but shouldn't be identical
        // (with very high probability given the shuffle)
        assert_ne!(folds[0].test_indices, folds[5].test_indices);
    }

    #[test]
    fn test_repeated_kfold_deterministic() {
        let cv = RepeatedKFold::new(5, 3).with_seed(42);

        let folds1 = cv.split(100, None, None).unwrap();
        let folds2 = cv.split(100, None, None).unwrap();

        for (f1, f2) in folds1.iter().zip(folds2.iter()) {
            assert_eq!(f1.test_indices, f2.test_indices);
        }
    }

    #[test]
    fn test_repeated_kfold_get_n_splits() {
        let cv = RepeatedKFold::new(5, 10);
        assert_eq!(cv.get_n_splits(None, None, None), 50);
    }

    #[test]
    fn test_repeated_kfold_name() {
        let cv = RepeatedKFold::new(5, 10);
        assert_eq!(cv.name(), "RepeatedKFold");
    }

    #[test]
    fn test_repeated_kfold_default() {
        let cv = RepeatedKFold::default();
        assert_eq!(cv.n_folds(), 5);
        assert_eq!(cv.n_repeats(), 10);
        assert!(cv.random_seed().is_none());
    }

    #[test]
    fn test_repeated_kfold_error_too_many_folds() {
        let cv = RepeatedKFold::new(10, 3);
        let result = cv.split(5, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_test_disjoint() {
        // Verify train and test indices are always disjoint
        let cv = KFold::new(5).with_shuffle(true).with_seed(42);
        let folds = cv.split(100, None, None).unwrap();

        for fold in &folds {
            let train_set: HashSet<usize> = fold.train_indices.iter().copied().collect();
            let test_set: HashSet<usize> = fold.test_indices.iter().copied().collect();

            // No overlap
            assert!(train_set.is_disjoint(&test_set));

            // Together they cover all samples
            assert_eq!(train_set.len() + test_set.len(), 100);
        }
    }

    #[test]
    fn test_serialization() {
        let cv = KFold::new(5).with_shuffle(true).with_seed(42);
        let json = serde_json::to_string(&cv).unwrap();
        let cv2: KFold = serde_json::from_str(&json).unwrap();

        assert_eq!(cv.n_folds(), cv2.n_folds());
        assert_eq!(cv.shuffle(), cv2.shuffle());
        assert_eq!(cv.random_seed(), cv2.random_seed());
    }

    #[test]
    fn test_repeated_kfold_serialization() {
        let cv = RepeatedKFold::new(5, 10).with_seed(42);
        let json = serde_json::to_string(&cv).unwrap();
        let cv2: RepeatedKFold = serde_json::from_str(&json).unwrap();

        assert_eq!(cv.n_folds(), cv2.n_folds());
        assert_eq!(cv.n_repeats(), cv2.n_repeats());
        assert_eq!(cv.random_seed(), cv2.random_seed());
    }
}
