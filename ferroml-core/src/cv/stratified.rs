//! Stratified K-Fold Cross-Validation Strategies
//!
//! This module provides stratified k-fold cross-validation that preserves
//! the class distribution in each fold.
//!
//! ## Strategies
//!
//! - [`StratifiedKFold`] - K-fold that preserves class proportions
//! - [`RepeatedStratifiedKFold`] - Multiple repetitions with different shuffles

use super::{get_class_distribution, shuffle_indices, validate_n_folds, CVFold, CrossValidator};
use crate::{FerroError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Stratified K-Fold cross-validator
///
/// Provides train/test indices to split data while preserving the percentage
/// of samples for each class. This is especially important for imbalanced datasets
/// where random splitting might result in folds with very different class distributions.
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
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::cv::{CrossValidator, StratifiedKFold};
/// use ndarray::Array1;
///
/// // Labels with imbalanced classes: 90% class 0, 10% class 1
/// let y = Array1::from_vec(vec![
///     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
/// ]);
///
/// let cv = StratifiedKFold::new(5).with_shuffle(true).with_seed(42);
/// let folds = cv.split(10, Some(&y), None)?;
///
/// // Each fold preserves ~90%/10% class distribution
/// # Ok(())
/// # }
/// ```
///
/// # Algorithm
///
/// 1. Group samples by class
/// 2. For each class, distribute samples across folds as evenly as possible
/// 3. Each fold receives approximately n_samples_in_class / n_folds samples
///
/// # Notes
///
/// - Requires labels (y parameter) for splitting
/// - Labels are rounded to integers for class identification
/// - Works best when each class has at least n_folds samples
/// - For classes with fewer samples than n_folds, some folds may not have that class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedKFold {
    /// Number of folds
    n_folds: usize,
    /// Whether to shuffle within each class before splitting
    shuffle: bool,
    /// Random seed for shuffling
    random_seed: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new StratifiedKFold cross-validator
    ///
    /// # Arguments
    ///
    /// * `n_folds` - Number of folds (must be >= 2)
    ///
    /// # Panics
    ///
    /// Panics if n_folds < 2
    pub fn new(n_folds: usize) -> Self {
        assert!(n_folds >= 2, "StratifiedKFold requires at least 2 folds");
        Self {
            n_folds,
            shuffle: false,
            random_seed: None,
        }
    }

    /// Enable shuffling before splitting
    ///
    /// When shuffle is enabled, samples within each class are randomly permuted
    /// before being distributed across folds.
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

impl Default for StratifiedKFold {
    fn default() -> Self {
        Self::new(5)
    }
}

impl CrossValidator for StratifiedKFold {
    fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<Vec<CVFold>> {
        // Validate inputs
        self.validate_inputs(n_samples, y, groups)?;
        validate_n_folds(self.n_folds, n_samples)?;

        let labels = y.ok_or_else(|| {
            FerroError::invalid_input("StratifiedKFold requires labels (y) for splitting")
        })?;

        // Get class distribution
        let mut class_indices = get_class_distribution(labels);

        // Sort class keys for deterministic ordering (needed for consistent shuffling)
        let mut class_keys: Vec<i64> = class_indices.keys().copied().collect();
        class_keys.sort();

        // Shuffle indices within each class if requested
        // Must iterate in sorted order for determinism
        if self.shuffle {
            let base_seed = self.random_seed.unwrap_or(0);
            for (_class_idx, &class_key) in class_keys.iter().enumerate() {
                if let Some(indices) = class_indices.get_mut(&class_key) {
                    // Use different seed for each class to avoid correlations
                    // Use class value (not index) for seed derivation for stability
                    shuffle_indices(indices, base_seed.wrapping_add(class_key as u64));
                }
            }
        }

        // Build fold assignments
        // For each class, distribute samples across folds
        let mut fold_test_indices: Vec<Vec<usize>> = vec![Vec::new(); self.n_folds];

        for &class_key in &class_keys {
            let indices = class_indices.get(&class_key).unwrap();

            // Distribute this class's samples across folds
            for (sample_idx, &idx) in indices.iter().enumerate() {
                // Assign to fold: sample_idx mod n_folds
                let fold_idx = sample_idx % self.n_folds;
                fold_test_indices[fold_idx].push(idx);
            }

            // Note: The above distribution ensures that:
            // - First n_class % n_folds folds get one extra sample
            // - Samples are distributed as evenly as possible
            // This matches sklearn's behavior
        }

        // Sort indices within each fold for deterministic output
        for indices in &mut fold_test_indices {
            indices.sort();
        }

        // Build CVFold structs
        let all_indices: Vec<usize> = (0..n_samples).collect();
        let mut folds = Vec::with_capacity(self.n_folds);

        for (fold_idx, test_indices) in fold_test_indices.into_iter().enumerate() {
            // Train indices are all indices not in test
            let test_set: std::collections::HashSet<usize> = test_indices.iter().copied().collect();
            let train_indices: Vec<usize> = all_indices
                .iter()
                .copied()
                .filter(|idx| !test_set.contains(idx))
                .collect();

            folds.push(CVFold::new(train_indices, test_indices, fold_idx));
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
        "StratifiedKFold"
    }

    fn requires_labels(&self) -> bool {
        true
    }
}

/// Repeated Stratified K-Fold cross-validator
///
/// Repeats Stratified K-Fold `n_repeats` times with different random shuffles.
/// This provides more robust estimates of model performance while maintaining
/// class balance in each fold.
///
/// # Parameters
///
/// - `n_folds`: Number of folds for each repetition. Must be at least 2.
/// - `n_repeats`: Number of times to repeat. Default: 10.
/// - `random_seed`: Base seed for the random number generator.
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::cv::{CrossValidator, RepeatedStratifiedKFold};
/// use ndarray::Array1;
///
/// let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
///
/// // 3-fold CV repeated 5 times = 15 total folds
/// let cv = RepeatedStratifiedKFold::new(3, 5).with_seed(42);
/// let folds = cv.split(6, Some(&y), None)?;
///
/// assert_eq!(folds.len(), 15);
/// # Ok(())
/// # }
/// ```
///
/// # Statistical Notes
///
/// - Each repetition uses a different random shuffle
/// - Provides more stable estimates than single stratified k-fold
/// - The class balance is maintained in each fold across all repetitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatedStratifiedKFold {
    /// Number of folds per repetition
    n_folds: usize,
    /// Number of repetitions
    n_repeats: usize,
    /// Base random seed
    random_seed: Option<u64>,
}

impl RepeatedStratifiedKFold {
    /// Create a new RepeatedStratifiedKFold cross-validator
    ///
    /// # Arguments
    ///
    /// * `n_folds` - Number of folds per repetition (must be >= 2)
    /// * `n_repeats` - Number of times to repeat (must be >= 1)
    ///
    /// # Panics
    ///
    /// Panics if n_folds < 2 or n_repeats < 1
    pub fn new(n_folds: usize, n_repeats: usize) -> Self {
        assert!(
            n_folds >= 2,
            "RepeatedStratifiedKFold requires at least 2 folds"
        );
        assert!(
            n_repeats >= 1,
            "RepeatedStratifiedKFold requires at least 1 repeat"
        );
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

impl Default for RepeatedStratifiedKFold {
    fn default() -> Self {
        Self::new(5, 10)
    }
}

impl CrossValidator for RepeatedStratifiedKFold {
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
            // Create a StratifiedKFold for this repetition with a different seed
            // Use a large offset between repeats to avoid seed correlations
            let repeat_seed = base_seed.wrapping_add((repeat_idx as u64) * 1000);
            let skfold = StratifiedKFold::new(self.n_folds)
                .with_shuffle(true)
                .with_seed(repeat_seed);

            let folds = skfold.split(n_samples, y, groups)?;

            // Add folds with updated indices
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
        "RepeatedStratifiedKFold"
    }

    fn requires_labels(&self) -> bool {
        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn check_class_balance(y: &Array1<f64>, folds: &[CVFold], tolerance: f64) {
        // Get overall class distribution
        let class_dist = get_class_distribution(y);
        let n_samples = y.len();

        for fold in folds {
            // Check class distribution in test set
            let test_labels: Vec<f64> = fold.test_indices.iter().map(|&i| y[i]).collect();
            let test_dist = get_class_distribution(&Array1::from_vec(test_labels.clone()));

            for (class, overall_indices) in &class_dist {
                let overall_ratio = overall_indices.len() as f64 / n_samples as f64;
                let test_ratio = test_dist
                    .get(class)
                    .map(|v| v.len() as f64 / test_labels.len() as f64)
                    .unwrap_or(0.0);

                // Check that fold ratio is close to overall ratio
                let diff = (test_ratio - overall_ratio).abs();
                assert!(
                    diff <= tolerance,
                    "Class {} has ratio {:.3} in fold but {:.3} overall (diff={:.3} > {:.3})",
                    class,
                    test_ratio,
                    overall_ratio,
                    diff,
                    tolerance
                );
            }
        }
    }

    #[test]
    fn test_stratified_kfold_basic() {
        // 60% class 0, 40% class 1 with enough samples for good stratification
        let mut labels = vec![0.0; 60];
        labels.extend(vec![1.0; 40]);
        let y = Array1::from_vec(labels);

        let cv = StratifiedKFold::new(5);
        let folds = cv.split(100, Some(&y), None).unwrap();

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

        // Check class balance with some tolerance
        check_class_balance(&y, &folds, 0.05);
    }

    #[test]
    fn test_stratified_kfold_perfect_balance() {
        // Perfectly balanced: 50 class 0, 50 class 1
        let mut labels = vec![0.0; 50];
        labels.extend(vec![1.0; 50]);
        let y = Array1::from_vec(labels);

        let cv = StratifiedKFold::new(5);
        let folds = cv.split(100, Some(&y), None).unwrap();

        // Each fold should have exactly 10 from class 0 and 10 from class 1
        for fold in &folds {
            let class_0_count = fold.test_indices.iter().filter(|&&i| y[i] == 0.0).count();
            let class_1_count = fold.test_indices.iter().filter(|&&i| y[i] == 1.0).count();

            assert_eq!(class_0_count, 10, "Expected 10 class 0 samples");
            assert_eq!(class_1_count, 10, "Expected 10 class 1 samples");
        }
    }

    #[test]
    fn test_stratified_kfold_imbalanced() {
        // Highly imbalanced: 90% class 0, 10% class 1
        let mut labels = vec![0.0; 90];
        labels.extend(vec![1.0; 10]);
        let y = Array1::from_vec(labels);

        let cv = StratifiedKFold::new(5);
        let folds = cv.split(100, Some(&y), None).unwrap();

        // Each fold should have 18 from class 0 and 2 from class 1
        for fold in &folds {
            let class_0_count = fold.test_indices.iter().filter(|&&i| y[i] == 0.0).count();
            let class_1_count = fold.test_indices.iter().filter(|&&i| y[i] == 1.0).count();

            assert_eq!(class_0_count, 18, "Expected 18 class 0 samples");
            assert_eq!(class_1_count, 2, "Expected 2 class 1 samples");
        }
    }

    #[test]
    fn test_stratified_kfold_multiclass() {
        // 3 classes: 40% class 0, 40% class 1, 20% class 2
        let mut labels = vec![0.0; 40];
        labels.extend(vec![1.0; 40]);
        labels.extend(vec![2.0; 20]);
        let y = Array1::from_vec(labels);

        let cv = StratifiedKFold::new(5);
        let folds = cv.split(100, Some(&y), None).unwrap();

        for fold in &folds {
            let class_0_count = fold.test_indices.iter().filter(|&&i| y[i] == 0.0).count();
            let class_1_count = fold.test_indices.iter().filter(|&&i| y[i] == 1.0).count();
            let class_2_count = fold.test_indices.iter().filter(|&&i| y[i] == 2.0).count();

            assert_eq!(class_0_count, 8);
            assert_eq!(class_1_count, 8);
            assert_eq!(class_2_count, 4);
        }
    }

    #[test]
    fn test_stratified_kfold_shuffle_deterministic() {
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let cv = StratifiedKFold::new(5).with_shuffle(true).with_seed(42);

        let folds1 = cv.split(10, Some(&y), None).unwrap();
        let folds2 = cv.split(10, Some(&y), None).unwrap();

        for (f1, f2) in folds1.iter().zip(folds2.iter()) {
            assert_eq!(f1.test_indices, f2.test_indices);
        }
    }

    #[test]
    fn test_stratified_kfold_shuffle_different_seeds() {
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);

        let cv1 = StratifiedKFold::new(5).with_shuffle(true).with_seed(42);
        let cv2 = StratifiedKFold::new(5).with_shuffle(true).with_seed(123);

        let folds1 = cv1.split(20, Some(&y), None).unwrap();
        let folds2 = cv2.split(20, Some(&y), None).unwrap();

        // At least one fold should be different
        let all_same = folds1
            .iter()
            .zip(folds2.iter())
            .all(|(f1, f2)| f1.test_indices == f2.test_indices);
        assert!(!all_same);
    }

    #[test]
    fn test_stratified_kfold_requires_labels() {
        let cv = StratifiedKFold::new(5);
        assert!(cv.requires_labels());

        // Should error without labels
        let result = cv.split(100, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_stratified_kfold_allows_sparse_class() {
        // Class 1 has only 3 samples for 5 folds - this is allowed (like sklearn)
        // Some folds will have 0 samples from class 1
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let cv = StratifiedKFold::new(5);
        let result = cv.split(10, Some(&y), None);
        assert!(result.is_ok());

        let folds = result.unwrap();
        assert_eq!(folds.len(), 5);

        // Verify all samples are covered
        let mut all_test: HashSet<usize> = HashSet::new();
        for fold in &folds {
            for &idx in &fold.test_indices {
                all_test.insert(idx);
            }
        }
        assert_eq!(all_test.len(), 10);
    }

    #[test]
    fn test_stratified_kfold_name() {
        let cv = StratifiedKFold::new(5);
        assert_eq!(cv.name(), "StratifiedKFold");
    }

    #[test]
    fn test_stratified_kfold_default() {
        let cv = StratifiedKFold::default();
        assert_eq!(cv.n_folds(), 5);
        assert!(!cv.shuffle());
        assert!(cv.random_seed().is_none());
    }

    #[test]
    fn test_stratified_kfold_builder() {
        let cv = StratifiedKFold::new(10).with_shuffle(true).with_seed(42);
        assert_eq!(cv.n_folds(), 10);
        assert!(cv.shuffle());
        assert_eq!(cv.random_seed(), Some(42));
    }

    #[test]
    fn test_train_test_disjoint() {
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let cv = StratifiedKFold::new(5).with_shuffle(true).with_seed(42);
        let folds = cv.split(10, Some(&y), None).unwrap();

        for fold in &folds {
            let train_set: HashSet<usize> = fold.train_indices.iter().copied().collect();
            let test_set: HashSet<usize> = fold.test_indices.iter().copied().collect();

            assert!(train_set.is_disjoint(&test_set));
            assert_eq!(train_set.len() + test_set.len(), 10);
        }
    }

    // RepeatedStratifiedKFold tests

    #[test]
    fn test_repeated_stratified_kfold_basic() {
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let cv = RepeatedStratifiedKFold::new(5, 3);
        let folds = cv.split(10, Some(&y), None).unwrap();

        // 5 folds × 3 repeats = 15 total
        assert_eq!(folds.len(), 15);

        // Check fold indices are sequential
        for (i, fold) in folds.iter().enumerate() {
            assert_eq!(fold.fold_index, i);
        }
    }

    #[test]
    fn test_repeated_stratified_kfold_class_balance() {
        // 60% class 0, 40% class 1
        let mut labels = vec![0.0; 60];
        labels.extend(vec![1.0; 40]);
        let y = Array1::from_vec(labels);

        let cv = RepeatedStratifiedKFold::new(5, 3).with_seed(42);
        let folds = cv.split(100, Some(&y), None).unwrap();

        // Check class balance in each fold
        check_class_balance(&y, &folds, 0.1);
    }

    #[test]
    fn test_repeated_stratified_kfold_coverage_per_repeat() {
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let cv = RepeatedStratifiedKFold::new(5, 2).with_seed(42);
        let folds = cv.split(10, Some(&y), None).unwrap();

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
            assert_eq!(covered.len(), 10);
        }
    }

    #[test]
    fn test_repeated_stratified_kfold_different_shuffles() {
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]);

        let cv = RepeatedStratifiedKFold::new(5, 2).with_seed(42);
        let folds = cv.split(20, Some(&y), None).unwrap();

        // First repeat and second repeat should have different orderings
        // (comparing corresponding folds)
        assert_ne!(folds[0].test_indices, folds[5].test_indices);
    }

    #[test]
    fn test_repeated_stratified_kfold_deterministic() {
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let cv = RepeatedStratifiedKFold::new(5, 3).with_seed(42);

        let folds1 = cv.split(10, Some(&y), None).unwrap();
        let folds2 = cv.split(10, Some(&y), None).unwrap();

        for (f1, f2) in folds1.iter().zip(folds2.iter()) {
            assert_eq!(f1.test_indices, f2.test_indices);
        }
    }

    #[test]
    fn test_repeated_stratified_kfold_requires_labels() {
        let cv = RepeatedStratifiedKFold::new(5, 3);
        assert!(cv.requires_labels());

        // Should error without labels
        let result = cv.split(100, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_repeated_stratified_kfold_get_n_splits() {
        let cv = RepeatedStratifiedKFold::new(5, 10);
        assert_eq!(cv.get_n_splits(None, None, None), 50);
    }

    #[test]
    fn test_repeated_stratified_kfold_name() {
        let cv = RepeatedStratifiedKFold::new(5, 10);
        assert_eq!(cv.name(), "RepeatedStratifiedKFold");
    }

    #[test]
    fn test_repeated_stratified_kfold_default() {
        let cv = RepeatedStratifiedKFold::default();
        assert_eq!(cv.n_folds(), 5);
        assert_eq!(cv.n_repeats(), 10);
        assert!(cv.random_seed().is_none());
    }

    #[test]
    fn test_serialization() {
        let cv = StratifiedKFold::new(5).with_shuffle(true).with_seed(42);
        let json = serde_json::to_string(&cv).unwrap();
        let cv2: StratifiedKFold = serde_json::from_str(&json).unwrap();

        assert_eq!(cv.n_folds(), cv2.n_folds());
        assert_eq!(cv.shuffle(), cv2.shuffle());
        assert_eq!(cv.random_seed(), cv2.random_seed());
    }

    #[test]
    fn test_repeated_stratified_kfold_serialization() {
        let cv = RepeatedStratifiedKFold::new(5, 10).with_seed(42);
        let json = serde_json::to_string(&cv).unwrap();
        let cv2: RepeatedStratifiedKFold = serde_json::from_str(&json).unwrap();

        assert_eq!(cv.n_folds(), cv2.n_folds());
        assert_eq!(cv.n_repeats(), cv2.n_repeats());
        assert_eq!(cv.random_seed(), cv2.random_seed());
    }
}
