//! Leave-One-Out and Leave-P-Out Cross-Validation
//!
//! This module provides leave-out based cross-validation strategies:
//!
//! - [`LeaveOneOut`] - Each sample is used once as the test set
//! - [`LeavePOut`] - Each combination of p samples is used as the test set
//! - [`ShuffleSplit`] - Random train/test splits with configurable sizes

use super::{shuffle_indices, CVFold, CrossValidator};
use crate::{FerroError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Leave-One-Out cross-validator
///
/// Each sample is used once as a test set (singleton) while the remaining
/// samples form the training set. This is equivalent to `KFold(n)` where
/// `n` is the number of samples.
///
/// # Characteristics
///
/// - Produces `n` folds for `n` samples
/// - Each test fold contains exactly 1 sample
/// - Training set size is always `n - 1`
/// - Maximum utilization of data for training
/// - High variance in estimates (each fold has only 1 test sample)
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::cv::{CrossValidator, LeaveOneOut};
///
/// let cv = LeaveOneOut::new();
/// let folds = cv.split(100, None, None)?;
///
/// assert_eq!(folds.len(), 100);
/// for fold in &folds {
///     assert_eq!(fold.test_indices.len(), 1);
///     assert_eq!(fold.train_indices.len(), 99);
/// }
/// # Ok(())
/// # }
/// ```
///
/// # When to Use
///
/// - Small datasets where maximizing training data is critical
/// - When bias is a bigger concern than variance
/// - For precise estimation of generalization error (though high variance)
///
/// # Caveats
///
/// - Computationally expensive for large datasets (n model fits)
/// - High variance due to small test sets (each sample has high influence)
/// - Not suitable for large datasets; use KFold instead
/// - No shuffling needed (all permutations are equivalent)
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LeaveOneOut;

impl LeaveOneOut {
    /// Create a new Leave-One-Out cross-validator
    pub fn new() -> Self {
        Self
    }
}

impl CrossValidator for LeaveOneOut {
    fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<Vec<CVFold>> {
        // Validate inputs
        self.validate_inputs(n_samples, y, groups)?;

        if n_samples < 2 {
            return Err(FerroError::invalid_input(
                "LeaveOneOut requires at least 2 samples",
            ));
        }

        // Create one fold per sample
        let folds: Vec<CVFold> = (0..n_samples)
            .map(|test_idx| {
                // Training is all indices except test_idx
                let train_indices: Vec<usize> = (0..n_samples).filter(|&i| i != test_idx).collect();
                let test_indices = vec![test_idx];
                CVFold::new(train_indices, test_indices, test_idx)
            })
            .collect();

        Ok(folds)
    }

    fn get_n_splits(
        &self,
        n_samples: Option<usize>,
        _y: Option<&Array1<f64>>,
        _groups: Option<&Array1<i64>>,
    ) -> usize {
        // LOO produces n folds for n samples
        n_samples.unwrap_or(0)
    }

    fn name(&self) -> &str {
        "LeaveOneOut"
    }
}

/// Leave-P-Out cross-validator
///
/// Each combination of p samples is used once as the test set while the
/// remaining samples form the training set. This produces C(n, p) = n! / (p! * (n-p)!)
/// folds.
///
/// # Parameters
///
/// - `p`: Number of samples in each test set. Must be at least 1 and less than n.
///
/// # Characteristics
///
/// - Produces C(n, p) folds
/// - Each test fold contains exactly p samples
/// - Training set size is always n - p
/// - Exhaustive enumeration of all possible test sets of size p
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::cv::{CrossValidator, LeavePOut};
///
/// // Leave-2-out: C(10, 2) = 45 folds
/// let cv = LeavePOut::new(2);
/// let folds = cv.split(10, None, None)?;
///
/// assert_eq!(folds.len(), 45);
/// for fold in &folds {
///     assert_eq!(fold.test_indices.len(), 2);
///     assert_eq!(fold.train_indices.len(), 8);
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Computational Complexity
///
/// **Warning**: The number of folds grows combinatorially!
///
/// - p=1: n folds (same as LOO)
/// - p=2: n*(n-1)/2 folds (100 samples → 4950 folds)
/// - p=3: n*(n-1)*(n-2)/6 folds (100 samples → 161,700 folds)
///
/// Use with caution for large datasets or large p.
///
/// # When to Use
///
/// - Small datasets where exhaustive evaluation is feasible
/// - When you need unbiased estimates of test performance
/// - Research settings comparing different CV strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeavePOut {
    /// Number of samples to leave out for testing
    p: usize,
}

impl LeavePOut {
    /// Create a new Leave-P-Out cross-validator
    ///
    /// # Arguments
    ///
    /// * `p` - Number of samples in each test set (must be >= 1)
    ///
    /// # Panics
    ///
    /// Panics if p < 1
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "LeavePOut requires p >= 1");
        Self { p }
    }

    /// Get the value of p
    pub fn p(&self) -> usize {
        self.p
    }

    /// Calculate the number of combinations C(n, k) = n! / (k! * (n-k)!)
    fn n_choose_k(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        // Use the symmetry property: C(n, k) = C(n, n-k)
        let k = k.min(n - k);

        // Calculate incrementally to avoid overflow
        let mut result = 1usize;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Generate all combinations of k elements from n elements
    /// Uses iterative approach for memory efficiency
    fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
        let n_combinations = Self::n_choose_k(n, k);
        let mut result = Vec::with_capacity(n_combinations);

        if k == 0 {
            result.push(vec![]);
            return result;
        }

        if k > n {
            return result;
        }

        // Initialize first combination: [0, 1, 2, ..., k-1]
        let mut combo: Vec<usize> = (0..k).collect();
        result.push(combo.clone());

        loop {
            // Find rightmost element that can be incremented
            let mut i = k;
            while i > 0 {
                i -= 1;
                if combo[i] < n - k + i {
                    break;
                }
            }

            // Check if we've exhausted all combinations
            if combo[i] >= n - k + i {
                break;
            }

            // Increment this element
            combo[i] += 1;

            // Reset all elements to the right
            for j in (i + 1)..k {
                combo[j] = combo[j - 1] + 1;
            }

            result.push(combo.clone());
        }

        result
    }
}

impl Default for LeavePOut {
    fn default() -> Self {
        Self::new(2) // Leave-2-out is common
    }
}

impl CrossValidator for LeavePOut {
    fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<Vec<CVFold>> {
        // Validate inputs
        self.validate_inputs(n_samples, y, groups)?;

        if n_samples < self.p + 1 {
            return Err(FerroError::invalid_input(format!(
                "LeavePOut with p={} requires at least {} samples, got {}",
                self.p,
                self.p + 1,
                n_samples
            )));
        }

        // Check for combinatorial explosion
        let n_folds = Self::n_choose_k(n_samples, self.p);
        if n_folds > 1_000_000 {
            return Err(FerroError::invalid_input(format!(
                "LeavePOut would produce {} folds (C({}, {})), which is too many. \
                 Consider using KFold or ShuffleSplit instead.",
                n_folds, n_samples, self.p
            )));
        }

        // Generate all combinations of p indices
        let test_combinations = Self::combinations(n_samples, self.p);

        let folds: Vec<CVFold> = test_combinations
            .into_iter()
            .enumerate()
            .map(|(fold_idx, test_indices)| {
                // Training is all indices not in test
                let test_set: std::collections::HashSet<usize> =
                    test_indices.iter().copied().collect();
                let train_indices: Vec<usize> =
                    (0..n_samples).filter(|i| !test_set.contains(i)).collect();
                CVFold::new(train_indices, test_indices, fold_idx)
            })
            .collect();

        Ok(folds)
    }

    fn get_n_splits(
        &self,
        n_samples: Option<usize>,
        _y: Option<&Array1<f64>>,
        _groups: Option<&Array1<i64>>,
    ) -> usize {
        match n_samples {
            Some(n) => Self::n_choose_k(n, self.p),
            None => 0,
        }
    }

    fn name(&self) -> &str {
        "LeavePOut"
    }
}

/// Shuffle-Split cross-validator
///
/// Random permutation cross-validator. Yields train/test indices by randomly
/// sampling the data. Unlike KFold, ShuffleSplit allows:
///
/// - Configurable test size (not tied to 1/n_folds)
/// - Overlapping test sets across iterations
/// - Independent random splits
///
/// # Parameters
///
/// - `n_splits`: Number of train/test splits to generate
/// - `test_size`: Fraction (0-1) or absolute number of test samples
/// - `train_size`: Optional fraction/number of train samples (defaults to complement of test)
/// - `random_seed`: Seed for reproducibility
///
/// # Example
///
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::cv::{CrossValidator, ShuffleSplit};
///
/// // 10 random splits with 20% test size
/// let cv = ShuffleSplit::new(10)
///     .with_test_size(0.2)
///     .with_seed(42);
///
/// let folds = cv.split(100, None, None)?;
///
/// assert_eq!(folds.len(), 10);
/// for fold in &folds {
///     assert_eq!(fold.test_indices.len(), 20);
///     assert_eq!(fold.train_indices.len(), 80);
/// }
/// # Ok(())
/// # }
/// ```
///
/// # When to Use
///
/// - When you need specific train/test proportions
/// - Large datasets where exhaustive CV is expensive
/// - When you want to control the number of iterations independently of test size
/// - Quick preliminary evaluation before thorough CV
///
/// # Comparison with KFold
///
/// - KFold: Each sample appears in exactly one test set
/// - ShuffleSplit: Samples may appear in multiple or zero test sets
/// - KFold: Deterministic number of splits based on n_folds
/// - ShuffleSplit: User specifies n_splits independently
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShuffleSplit {
    /// Number of train/test splits to generate
    n_splits: usize,
    /// Fraction (0-1) of samples for test set
    test_fraction: f64,
    /// Optional fraction (0-1) of samples for train set
    train_fraction: Option<f64>,
    /// Random seed for reproducibility
    random_seed: Option<u64>,
}

impl ShuffleSplit {
    /// Create a new ShuffleSplit cross-validator
    ///
    /// # Arguments
    ///
    /// * `n_splits` - Number of train/test splits (must be >= 1)
    ///
    /// # Panics
    ///
    /// Panics if n_splits < 1
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 1, "ShuffleSplit requires at least 1 split");
        Self {
            n_splits,
            test_fraction: 0.1, // Default 10% test
            train_fraction: None,
            random_seed: None,
        }
    }

    /// Set the test size as a fraction (0-1) of the dataset
    ///
    /// # Panics
    ///
    /// Panics if fraction is not in (0, 1)
    pub fn with_test_size(mut self, fraction: f64) -> Self {
        assert!(
            fraction > 0.0 && fraction < 1.0,
            "Test fraction must be in (0, 1)"
        );
        self.test_fraction = fraction;
        self
    }

    /// Set the train size as a fraction (0-1) of the dataset
    ///
    /// If not set, train size is the complement of test size.
    /// If set, train_size + test_size may be < 1 (some samples unused).
    ///
    /// # Panics
    ///
    /// Panics if fraction is not in (0, 1)
    pub fn with_train_size(mut self, fraction: f64) -> Self {
        assert!(
            fraction > 0.0 && fraction < 1.0,
            "Train fraction must be in (0, 1)"
        );
        self.train_fraction = Some(fraction);
        self
    }

    /// Set the random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Get the number of splits
    pub fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// Get the test fraction
    pub fn test_fraction(&self) -> f64 {
        self.test_fraction
    }

    /// Get the train fraction (if explicitly set)
    pub fn train_fraction(&self) -> Option<f64> {
        self.train_fraction
    }

    /// Get the random seed (if set)
    pub fn random_seed(&self) -> Option<u64> {
        self.random_seed
    }

    /// Calculate actual train and test sizes from fractions
    fn calculate_sizes(&self, n_samples: usize) -> (usize, usize) {
        // Use floor for consistent behavior (matches sklearn)
        let n_test = (n_samples as f64 * self.test_fraction).floor() as usize;
        let n_test = n_test.max(1).min(n_samples - 1);

        let n_train = match self.train_fraction {
            Some(frac) => {
                let train = (n_samples as f64 * frac).round() as usize;
                train.max(1).min(n_samples - n_test)
            }
            None => n_samples - n_test,
        };

        (n_train, n_test)
    }
}

impl Default for ShuffleSplit {
    fn default() -> Self {
        Self::new(10).with_test_size(0.1)
    }
}

impl CrossValidator for ShuffleSplit {
    fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<f64>>,
        groups: Option<&Array1<i64>>,
    ) -> Result<Vec<CVFold>> {
        // Validate inputs
        self.validate_inputs(n_samples, y, groups)?;

        if n_samples < 2 {
            return Err(FerroError::invalid_input(
                "ShuffleSplit requires at least 2 samples",
            ));
        }

        let (n_train, n_test) = self.calculate_sizes(n_samples);

        // Validate sizes
        if n_train + n_test > n_samples {
            return Err(FerroError::invalid_input(format!(
                "Train size ({}) + test size ({}) exceeds n_samples ({})",
                n_train, n_test, n_samples
            )));
        }

        let base_seed = self.random_seed.unwrap_or(0);
        let mut folds = Vec::with_capacity(self.n_splits);

        for split_idx in 0..self.n_splits {
            // Create and shuffle indices for this split
            let mut indices: Vec<usize> = (0..n_samples).collect();
            shuffle_indices(&mut indices, base_seed.wrapping_add(split_idx as u64));

            // First n_test indices are test, next n_train are train
            let test_indices: Vec<usize> = indices[..n_test].to_vec();
            let train_indices: Vec<usize> = indices[n_test..n_test + n_train].to_vec();

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
        "ShuffleSplit"
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // LeaveOneOut tests

    #[test]
    fn test_loo_basic() {
        let cv = LeaveOneOut::new();
        let folds = cv.split(10, None, None).unwrap();

        assert_eq!(folds.len(), 10);

        for (i, fold) in folds.iter().enumerate() {
            // Each fold has exactly 1 test sample
            assert_eq!(fold.test_indices.len(), 1);
            assert_eq!(fold.test_indices[0], i);

            // Train is all others
            assert_eq!(fold.train_indices.len(), 9);
            assert!(!fold.train_indices.contains(&i));
        }
    }

    #[test]
    fn test_loo_coverage() {
        let cv = LeaveOneOut::new();
        let folds = cv.split(50, None, None).unwrap();

        // Each sample appears exactly once as test
        let mut test_count = vec![0usize; 50];
        for fold in &folds {
            for &idx in &fold.test_indices {
                test_count[idx] += 1;
            }
        }
        assert!(test_count.iter().all(|&c| c == 1));

        // Each sample appears n-1 times as train
        let mut train_count = vec![0usize; 50];
        for fold in &folds {
            for &idx in &fold.train_indices {
                train_count[idx] += 1;
            }
        }
        assert!(train_count.iter().all(|&c| c == 49));
    }

    #[test]
    fn test_loo_minimum_samples() {
        let cv = LeaveOneOut::new();
        let folds = cv.split(2, None, None).unwrap();

        assert_eq!(folds.len(), 2);
        assert_eq!(folds[0].test_indices, vec![0]);
        assert_eq!(folds[0].train_indices, vec![1]);
        assert_eq!(folds[1].test_indices, vec![1]);
        assert_eq!(folds[1].train_indices, vec![0]);
    }

    #[test]
    fn test_loo_error_single_sample() {
        let cv = LeaveOneOut::new();
        let result = cv.split(1, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_loo_error_empty() {
        let cv = LeaveOneOut::new();
        let result = cv.split(0, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_loo_get_n_splits() {
        let cv = LeaveOneOut::new();
        assert_eq!(cv.get_n_splits(Some(100), None, None), 100);
        assert_eq!(cv.get_n_splits(Some(5), None, None), 5);
        assert_eq!(cv.get_n_splits(None, None, None), 0);
    }

    #[test]
    fn test_loo_name() {
        let cv = LeaveOneOut::new();
        assert_eq!(cv.name(), "LeaveOneOut");
    }

    #[test]
    fn test_loo_serialization() {
        let cv = LeaveOneOut::new();
        let json = serde_json::to_string(&cv).unwrap();
        let cv2: LeaveOneOut = serde_json::from_str(&json).unwrap();
        assert_eq!(cv, cv2);
    }

    // LeavePOut tests

    #[test]
    fn test_lpo_basic() {
        let cv = LeavePOut::new(2);
        let folds = cv.split(5, None, None).unwrap();

        // C(5, 2) = 10 combinations
        assert_eq!(folds.len(), 10);

        for fold in &folds {
            assert_eq!(fold.test_indices.len(), 2);
            assert_eq!(fold.train_indices.len(), 3);
        }
    }

    #[test]
    fn test_lpo_is_loo_for_p1() {
        let lpo = LeavePOut::new(1);
        let loo = LeaveOneOut::new();

        let lpo_folds = lpo.split(10, None, None).unwrap();
        let loo_folds = loo.split(10, None, None).unwrap();

        assert_eq!(lpo_folds.len(), loo_folds.len());

        for (lpo_fold, loo_fold) in lpo_folds.iter().zip(loo_folds.iter()) {
            assert_eq!(lpo_fold.test_indices, loo_fold.test_indices);
            assert_eq!(lpo_fold.train_indices, loo_fold.train_indices);
        }
    }

    #[test]
    fn test_lpo_combinations_count() {
        // C(n, k) tests
        assert_eq!(LeavePOut::n_choose_k(5, 2), 10);
        assert_eq!(LeavePOut::n_choose_k(10, 3), 120);
        assert_eq!(LeavePOut::n_choose_k(6, 3), 20);
        assert_eq!(LeavePOut::n_choose_k(4, 4), 1);
        assert_eq!(LeavePOut::n_choose_k(4, 0), 1);
        assert_eq!(LeavePOut::n_choose_k(4, 5), 0);
    }

    #[test]
    fn test_lpo_all_combinations_unique() {
        let cv = LeavePOut::new(3);
        let folds = cv.split(6, None, None).unwrap();

        // C(6, 3) = 20
        assert_eq!(folds.len(), 20);

        // All test sets should be unique
        let test_sets: HashSet<Vec<usize>> = folds.iter().map(|f| f.test_indices.clone()).collect();
        assert_eq!(test_sets.len(), 20);
    }

    #[test]
    fn test_lpo_coverage() {
        let cv = LeavePOut::new(2);
        let folds = cv.split(5, None, None).unwrap();

        // Each sample should appear in C(n-1, p-1) = C(4, 1) = 4 test sets
        let mut test_count = vec![0usize; 5];
        for fold in &folds {
            for &idx in &fold.test_indices {
                test_count[idx] += 1;
            }
        }
        assert!(test_count.iter().all(|&c| c == 4));
    }

    #[test]
    fn test_lpo_error_p_too_large() {
        let cv = LeavePOut::new(5);
        let result = cv.split(5, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_lpo_error_combinatorial_explosion() {
        let cv = LeavePOut::new(10);
        // C(100, 10) is huge
        let result = cv.split(100, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_lpo_get_n_splits() {
        let cv = LeavePOut::new(2);
        assert_eq!(cv.get_n_splits(Some(5), None, None), 10); // C(5, 2)
        assert_eq!(cv.get_n_splits(Some(10), None, None), 45); // C(10, 2)
        assert_eq!(cv.get_n_splits(None, None, None), 0);
    }

    #[test]
    fn test_lpo_name() {
        let cv = LeavePOut::new(2);
        assert_eq!(cv.name(), "LeavePOut");
    }

    #[test]
    fn test_lpo_p_accessor() {
        let cv = LeavePOut::new(3);
        assert_eq!(cv.p(), 3);
    }

    #[test]
    fn test_lpo_default() {
        let cv = LeavePOut::default();
        assert_eq!(cv.p(), 2);
    }

    #[test]
    fn test_lpo_serialization() {
        let cv = LeavePOut::new(3);
        let json = serde_json::to_string(&cv).unwrap();
        let cv2: LeavePOut = serde_json::from_str(&json).unwrap();
        assert_eq!(cv.p(), cv2.p());
    }

    // ShuffleSplit tests

    #[test]
    fn test_shuffle_split_basic() {
        let cv = ShuffleSplit::new(10).with_test_size(0.2).with_seed(42);
        let folds = cv.split(100, None, None).unwrap();

        assert_eq!(folds.len(), 10);

        for fold in &folds {
            assert_eq!(fold.test_indices.len(), 20);
            assert_eq!(fold.train_indices.len(), 80);
        }
    }

    #[test]
    fn test_shuffle_split_disjoint() {
        let cv = ShuffleSplit::new(5).with_test_size(0.3).with_seed(42);
        let folds = cv.split(100, None, None).unwrap();

        for fold in &folds {
            let train_set: HashSet<usize> = fold.train_indices.iter().copied().collect();
            let test_set: HashSet<usize> = fold.test_indices.iter().copied().collect();

            // Train and test should be disjoint
            assert!(train_set.is_disjoint(&test_set));
        }
    }

    #[test]
    fn test_shuffle_split_deterministic() {
        let cv = ShuffleSplit::new(5).with_test_size(0.2).with_seed(42);

        let folds1 = cv.split(100, None, None).unwrap();
        let folds2 = cv.split(100, None, None).unwrap();

        for (f1, f2) in folds1.iter().zip(folds2.iter()) {
            assert_eq!(f1.test_indices, f2.test_indices);
            assert_eq!(f1.train_indices, f2.train_indices);
        }
    }

    #[test]
    fn test_shuffle_split_different_seeds() {
        let cv1 = ShuffleSplit::new(1).with_test_size(0.2).with_seed(42);
        let cv2 = ShuffleSplit::new(1).with_test_size(0.2).with_seed(123);

        let folds1 = cv1.split(100, None, None).unwrap();
        let folds2 = cv2.split(100, None, None).unwrap();

        // Different seeds should produce different splits
        assert_ne!(folds1[0].test_indices, folds2[0].test_indices);
    }

    #[test]
    fn test_shuffle_split_different_splits_have_different_indices() {
        let cv = ShuffleSplit::new(3).with_test_size(0.2).with_seed(42);
        let folds = cv.split(100, None, None).unwrap();

        // Different splits should have different test indices
        assert_ne!(folds[0].test_indices, folds[1].test_indices);
        assert_ne!(folds[1].test_indices, folds[2].test_indices);
    }

    #[test]
    fn test_shuffle_split_with_train_size() {
        // Use only 50% train, 20% test, 30% unused
        let cv = ShuffleSplit::new(5)
            .with_test_size(0.2)
            .with_train_size(0.5)
            .with_seed(42);
        let folds = cv.split(100, None, None).unwrap();

        for fold in &folds {
            assert_eq!(fold.test_indices.len(), 20);
            assert_eq!(fold.train_indices.len(), 50);
        }
    }

    #[test]
    fn test_shuffle_split_small_dataset() {
        let cv = ShuffleSplit::new(3).with_test_size(0.5).with_seed(42);
        let folds = cv.split(4, None, None).unwrap();

        for fold in &folds {
            assert_eq!(fold.test_indices.len(), 2);
            assert_eq!(fold.train_indices.len(), 2);
        }
    }

    #[test]
    fn test_shuffle_split_error_single_sample() {
        let cv = ShuffleSplit::new(5);
        let result = cv.split(1, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_shuffle_split_get_n_splits() {
        let cv = ShuffleSplit::new(10);
        assert_eq!(cv.get_n_splits(None, None, None), 10);
        assert_eq!(cv.get_n_splits(Some(100), None, None), 10);
    }

    #[test]
    fn test_shuffle_split_name() {
        let cv = ShuffleSplit::new(5);
        assert_eq!(cv.name(), "ShuffleSplit");
    }

    #[test]
    fn test_shuffle_split_accessors() {
        let cv = ShuffleSplit::new(10)
            .with_test_size(0.25)
            .with_train_size(0.5)
            .with_seed(42);

        assert_eq!(cv.n_splits(), 10);
        assert!((cv.test_fraction() - 0.25).abs() < 1e-10);
        assert!((cv.train_fraction().unwrap() - 0.5).abs() < 1e-10);
        assert_eq!(cv.random_seed(), Some(42));
    }

    #[test]
    fn test_shuffle_split_default() {
        let cv = ShuffleSplit::default();
        assert_eq!(cv.n_splits(), 10);
        assert!((cv.test_fraction() - 0.1).abs() < 1e-10);
        assert!(cv.train_fraction().is_none());
        assert!(cv.random_seed().is_none());
    }

    #[test]
    fn test_shuffle_split_serialization() {
        let cv = ShuffleSplit::new(5)
            .with_test_size(0.2)
            .with_train_size(0.6)
            .with_seed(42);

        let json = serde_json::to_string(&cv).unwrap();
        let cv2: ShuffleSplit = serde_json::from_str(&json).unwrap();

        assert_eq!(cv.n_splits(), cv2.n_splits());
        assert!((cv.test_fraction() - cv2.test_fraction()).abs() < 1e-10);
        assert_eq!(cv.train_fraction(), cv2.train_fraction());
        assert_eq!(cv.random_seed(), cv2.random_seed());
    }

    // Cross-validator trait tests

    #[test]
    fn test_train_test_disjoint_all_cvs() {
        // Test disjoint property for all three CV types
        let n_samples = 20;

        let loo = LeaveOneOut::new();
        for fold in loo.split(n_samples, None, None).unwrap() {
            let train: HashSet<_> = fold.train_indices.iter().collect();
            let test: HashSet<_> = fold.test_indices.iter().collect();
            assert!(train.is_disjoint(&test));
        }

        let lpo = LeavePOut::new(3);
        for fold in lpo.split(n_samples, None, None).unwrap() {
            let train: HashSet<_> = fold.train_indices.iter().collect();
            let test: HashSet<_> = fold.test_indices.iter().collect();
            assert!(train.is_disjoint(&test));
        }

        let ss = ShuffleSplit::new(5).with_test_size(0.2).with_seed(42);
        for fold in ss.split(n_samples, None, None).unwrap() {
            let train: HashSet<_> = fold.train_indices.iter().collect();
            let test: HashSet<_> = fold.test_indices.iter().collect();
            assert!(train.is_disjoint(&test));
        }
    }

    #[test]
    fn test_fold_indices_valid() {
        let n_samples = 50;

        // All fold indices should be in [0, n_samples)
        let loo = LeaveOneOut::new();
        for fold in loo.split(n_samples, None, None).unwrap() {
            assert!(fold.train_indices.iter().all(|&i| i < n_samples));
            assert!(fold.test_indices.iter().all(|&i| i < n_samples));
        }

        let lpo = LeavePOut::new(2);
        for fold in lpo.split(n_samples, None, None).unwrap() {
            assert!(fold.train_indices.iter().all(|&i| i < n_samples));
            assert!(fold.test_indices.iter().all(|&i| i < n_samples));
        }

        let ss = ShuffleSplit::new(5).with_test_size(0.2).with_seed(42);
        for fold in ss.split(n_samples, None, None).unwrap() {
            assert!(fold.train_indices.iter().all(|&i| i < n_samples));
            assert!(fold.test_indices.iter().all(|&i| i < n_samples));
        }
    }
}
