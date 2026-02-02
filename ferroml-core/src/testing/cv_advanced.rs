//! Advanced Cross-Validation Tests
//!
//! Phase 24 of FerroML testing plan - comprehensive tests for:
//! - NestedCV data leakage prevention
//! - GroupKFold group integrity
//! - TimeSeriesSplit temporal ordering
//! - Integration with actual ML models

#![allow(unused_imports)]
#![allow(dead_code)]

use ndarray::Array1;
use std::collections::HashSet;

use crate::cv::{
    CrossValidator, GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit,
};

// ============================================================================
// NESTED CV TESTS - Focus on CV mechanics
// ============================================================================

#[test]
fn test_nested_cv_outer_folds_disjoint() {
    let n_samples = 30;
    let outer_cv = KFold::new(3);

    let all_test_indices: Vec<Vec<usize>> = outer_cv
        .split(n_samples, None, None)
        .unwrap()
        .iter()
        .map(|fold| fold.test_indices.clone())
        .collect();

    // Check no overlap between outer test sets
    for i in 0..all_test_indices.len() {
        for j in (i + 1)..all_test_indices.len() {
            let set_i: HashSet<_> = all_test_indices[i].iter().collect();
            let set_j: HashSet<_> = all_test_indices[j].iter().collect();
            let overlap: Vec<_> = set_i.intersection(&set_j).collect();
            assert!(
                overlap.is_empty(),
                "Outer folds {} and {} overlap: {:?}",
                i,
                j,
                overlap
            );
        }
    }
}

#[test]
fn test_nested_cv_inner_never_sees_outer_test() {
    // This test verifies the critical invariant:
    // inner_train ∪ inner_test ⊆ outer_train (never outer_test)

    let n_samples = 50;
    let outer_cv = KFold::new(5);
    let inner_cv = KFold::new(3);

    for outer_fold in outer_cv.split(n_samples, None, None).unwrap() {
        let outer_train = &outer_fold.train_indices;
        let outer_test: HashSet<_> = outer_fold.test_indices.iter().copied().collect();

        // Inner CV should only operate on outer_train
        let inner_folds = inner_cv.split(outer_train.len(), None, None).unwrap();

        for inner_fold in inner_folds {
            // Map inner indices to original indices
            let inner_train_original: Vec<usize> = inner_fold
                .train_indices
                .iter()
                .map(|&i| outer_train[i])
                .collect();
            let inner_test_original: Vec<usize> = inner_fold
                .test_indices
                .iter()
                .map(|&i| outer_train[i])
                .collect();

            // Verify no inner indices are in outer_test
            for idx in inner_train_original
                .iter()
                .chain(inner_test_original.iter())
            {
                assert!(
                    !outer_test.contains(idx),
                    "Inner fold contains outer test index {}",
                    idx
                );
            }
        }
    }
}

#[test]
fn test_nested_cv_fold_counts() {
    let n_samples = 40;
    let n_outer = 4;
    let n_inner = 3;

    let outer_cv = KFold::new(n_outer);
    let inner_cv = KFold::new(n_inner);

    let outer_folds = outer_cv.split(n_samples, None, None).unwrap();
    assert_eq!(outer_folds.len(), n_outer, "Should have {} outer folds", n_outer);

    for outer_fold in outer_folds {
        let inner_folds = inner_cv
            .split(outer_fold.train_indices.len(), None, None)
            .unwrap();
        assert_eq!(
            inner_folds.len(),
            n_inner,
            "Should have {} inner folds",
            n_inner
        );
    }
}

#[test]
fn test_nested_cv_train_test_coverage() {
    let n_samples = 50;
    let outer_cv = KFold::new(5);

    let folds = outer_cv.split(n_samples, None, None).unwrap();

    // All samples should appear in test exactly once across all folds
    let mut all_test_indices: Vec<usize> = Vec::new();
    for fold in &folds {
        all_test_indices.extend(&fold.test_indices);
    }
    all_test_indices.sort();

    let expected: Vec<usize> = (0..n_samples).collect();
    assert_eq!(
        all_test_indices, expected,
        "All samples should be tested exactly once"
    );
}

// ============================================================================
// GROUP KFOLD TESTS
// ============================================================================

#[test]
fn test_group_kfold_no_group_split_property() {
    // Property: for all folds, no group appears in both train and test
    let n_samples = 100;
    let groups = Array1::from_shape_fn(n_samples, |i| (i / 10) as i64); // 10 groups

    let cv = GroupKFold::new(5);
    let folds = cv.split(n_samples, None, Some(&groups)).unwrap();

    for (fold_idx, fold) in folds.iter().enumerate() {
        let train_groups: HashSet<i64> = fold.train_indices.iter().map(|&i| groups[i]).collect();
        let test_groups: HashSet<i64> = fold.test_indices.iter().map(|&i| groups[i]).collect();

        let overlap: Vec<_> = train_groups.intersection(&test_groups).collect();
        assert!(
            overlap.is_empty(),
            "Fold {} has group overlap: {:?}",
            fold_idx,
            overlap
        );
    }
}

#[test]
fn test_group_kfold_all_groups_tested() {
    let n_samples = 50;
    let n_groups = 10;
    let groups = Array1::from_shape_fn(n_samples, |i| (i % n_groups) as i64);

    let cv = GroupKFold::new(5);
    let folds = cv.split(n_samples, None, Some(&groups)).unwrap();

    let mut tested_groups: HashSet<i64> = HashSet::new();
    for fold in &folds {
        for &idx in &fold.test_indices {
            tested_groups.insert(groups[idx]);
        }
    }

    assert_eq!(tested_groups.len(), n_groups, "Not all groups were tested");
}

#[test]
fn test_group_kfold_single_sample_groups() {
    // Edge case: each sample is its own group
    let n_samples = 20;
    let groups = Array1::from_shape_fn(n_samples, |i| i as i64);

    let cv = GroupKFold::new(4);
    let result = cv.split(n_samples, None, Some(&groups));

    assert!(result.is_ok());
    let folds = result.unwrap();
    assert_eq!(folds.len(), 4);
}

#[test]
fn test_group_kfold_highly_imbalanced_groups() {
    // One giant group + many small groups
    let mut groups_vec = vec![0i64; 50]; // 50 samples in group 0
    groups_vec.extend(vec![1i64; 5]); // 5 samples in group 1
    groups_vec.extend(vec![2i64; 5]); // 5 samples in group 2
    let groups = Array1::from_vec(groups_vec);

    let cv = GroupKFold::new(3);
    let folds = cv.split(groups.len(), None, Some(&groups)).unwrap();

    // Should still produce valid folds
    assert_eq!(folds.len(), 3);

    // Verify group integrity
    for fold in &folds {
        let train_groups: HashSet<i64> = fold.train_indices.iter().map(|&i| groups[i]).collect();
        let test_groups: HashSet<i64> = fold.test_indices.iter().map(|&i| groups[i]).collect();

        assert!(train_groups.is_disjoint(&test_groups));
    }
}

#[test]
fn test_group_kfold_negative_group_ids() {
    let groups = Array1::from_vec(vec![-5i64, -5, -3, -3, 0, 0, 10, 10, 100, 100]);

    let cv = GroupKFold::new(2);
    let folds = cv.split(groups.len(), None, Some(&groups)).unwrap();

    assert_eq!(folds.len(), 2);

    // Verify group integrity with negative IDs
    for fold in &folds {
        let train_groups: HashSet<i64> = fold.train_indices.iter().map(|&i| groups[i]).collect();
        let test_groups: HashSet<i64> = fold.test_indices.iter().map(|&i| groups[i]).collect();

        assert!(train_groups.is_disjoint(&test_groups));
    }
}

#[test]
fn test_stratified_group_kfold_class_balance() {
    // Verify class proportions are maintained
    let n_samples = 60;
    let y = Array1::from_shape_fn(n_samples, |i| if i < 40 { 0.0 } else { 1.0 }); // 40:20 split
    let groups = Array1::from_shape_fn(n_samples, |i| (i / 6) as i64); // 10 groups

    let cv = StratifiedGroupKFold::new(3);
    let folds = cv.split(n_samples, Some(&y), Some(&groups)).unwrap();

    for fold in &folds {
        let test_class_0: usize = fold.test_indices.iter().filter(|&&i| y[i] == 0.0).count();
        let test_class_1: usize = fold.test_indices.iter().filter(|&&i| y[i] == 1.0).count();

        // Ratio should be roughly 2:1 (40:20 = 2:1)
        let ratio = test_class_0 as f64 / (test_class_1.max(1) as f64);
        assert!(
            ratio > 1.0 && ratio < 4.0,
            "Class balance violated: ratio = {}",
            ratio
        );
    }
}

#[test]
fn test_stratified_group_kfold_multiclass() {
    let n_samples = 90;
    let y = Array1::from_shape_fn(n_samples, |i| (i % 3) as f64); // 3 classes
    let groups = Array1::from_shape_fn(n_samples, |i| (i / 9) as i64); // 10 groups

    let cv = StratifiedGroupKFold::new(3);
    let folds = cv.split(n_samples, Some(&y), Some(&groups)).unwrap();

    // All 3 classes should appear in each test set
    for fold in &folds {
        let test_classes: HashSet<i64> = fold.test_indices.iter().map(|&i| y[i] as i64).collect();

        assert_eq!(test_classes.len(), 3, "Not all classes in test set");
    }
}

#[test]
fn test_group_kfold_requires_groups() {
    let cv = GroupKFold::new(3);
    assert!(cv.requires_groups());
}

// ============================================================================
// TIME SERIES SPLIT TESTS
// ============================================================================

#[test]
fn test_timeseries_temporal_ordering_invariant() {
    // Property: max(train_indices) < min(test_indices) for all folds
    let n_samples = 100;
    let cv = TimeSeriesSplit::new(5);
    let folds = cv.split(n_samples, None, None).unwrap();

    for (fold_idx, fold) in folds.iter().enumerate() {
        let max_train = *fold.train_indices.iter().max().unwrap();
        let min_test = *fold.test_indices.iter().min().unwrap();

        assert!(
            max_train < min_test,
            "Fold {}: train max {} >= test min {}",
            fold_idx,
            max_train,
            min_test
        );
    }
}

#[test]
fn test_timeseries_gap_respected() {
    let n_samples = 50;
    let gap = 3;
    let cv = TimeSeriesSplit::new(4).with_gap(gap);
    let folds = cv.split(n_samples, None, None).unwrap();

    for (fold_idx, fold) in folds.iter().enumerate() {
        let max_train = *fold.train_indices.iter().max().unwrap();
        let min_test = *fold.test_indices.iter().min().unwrap();

        let actual_gap = min_test - max_train - 1;
        assert!(
            actual_gap >= gap,
            "Fold {}: gap {} < required {}",
            fold_idx,
            actual_gap,
            gap
        );
    }
}

#[test]
fn test_timeseries_expanding_window() {
    let n_samples = 60;
    let cv = TimeSeriesSplit::new(5); // No max_train_size = expanding
    let folds = cv.split(n_samples, None, None).unwrap();

    let mut prev_train_size = 0;
    for fold in &folds {
        assert!(
            fold.train_indices.len() > prev_train_size,
            "Training set should expand"
        );
        prev_train_size = fold.train_indices.len();
    }
}

#[test]
fn test_timeseries_sliding_window() {
    let n_samples = 60;
    let max_train = 20;
    let cv = TimeSeriesSplit::new(5).with_max_train_size(max_train);
    let folds = cv.split(n_samples, None, None).unwrap();

    for fold in &folds {
        assert!(
            fold.train_indices.len() <= max_train,
            "Training set {} exceeds max {}",
            fold.train_indices.len(),
            max_train
        );
    }
}

#[test]
fn test_timeseries_test_sets_non_overlapping() {
    let n_samples = 50;
    let cv = TimeSeriesSplit::new(5);
    let folds = cv.split(n_samples, None, None).unwrap();

    let mut all_test_indices: Vec<usize> = Vec::new();
    for fold in &folds {
        for &idx in &fold.test_indices {
            assert!(
                !all_test_indices.contains(&idx),
                "Index {} appears in multiple test sets",
                idx
            );
            all_test_indices.push(idx);
        }
    }
}

#[test]
fn test_timeseries_fixed_test_size() {
    let n_samples = 60;
    let test_size = 8;
    let cv = TimeSeriesSplit::new(4).with_test_size(test_size);
    let folds = cv.split(n_samples, None, None).unwrap();

    for fold in &folds {
        assert_eq!(
            fold.test_indices.len(),
            test_size,
            "Test size should be exactly {}",
            test_size
        );
    }
}

#[test]
fn test_timeseries_train_indices_sorted() {
    let n_samples = 50;
    let cv = TimeSeriesSplit::new(4);
    let folds = cv.split(n_samples, None, None).unwrap();

    for (fold_idx, fold) in folds.iter().enumerate() {
        let mut sorted_train = fold.train_indices.clone();
        sorted_train.sort();
        assert_eq!(
            fold.train_indices, sorted_train,
            "Fold {} train indices not sorted",
            fold_idx
        );

        let mut sorted_test = fold.test_indices.clone();
        sorted_test.sort();
        assert_eq!(
            fold.test_indices, sorted_test,
            "Fold {} test indices not sorted",
            fold_idx
        );
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[test]
fn test_all_cv_strategies_produce_valid_splits() {
    let n_samples = 50;
    let y = Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);
    let groups = Array1::from_shape_fn(n_samples, |i| (i / 5) as i64);

    // Test KFold
    let kfold = KFold::new(5);
    let kfold_folds = kfold.split(n_samples, None, None).unwrap();
    assert_eq!(kfold_folds.len(), 5);

    // Test StratifiedKFold
    let stratified = StratifiedKFold::new(5);
    let stratified_folds = stratified.split(n_samples, Some(&y), None).unwrap();
    assert_eq!(stratified_folds.len(), 5);

    // Test GroupKFold
    let group_cv = GroupKFold::new(5);
    let group_folds = group_cv.split(n_samples, None, Some(&groups)).unwrap();
    assert_eq!(group_folds.len(), 5);

    // Test TimeSeriesSplit
    let ts_cv = TimeSeriesSplit::new(5);
    let ts_folds = ts_cv.split(n_samples, None, None).unwrap();
    assert_eq!(ts_folds.len(), 5);
}

#[test]
fn test_cv_fold_indices_are_valid() {
    let n_samples = 30;
    let cv = KFold::new(5);
    let folds = cv.split(n_samples, None, None).unwrap();

    for fold in &folds {
        // All indices should be in valid range
        assert!(fold.train_indices.iter().all(|&i| i < n_samples));
        assert!(fold.test_indices.iter().all(|&i| i < n_samples));

        // Train and test should be disjoint
        for &test_idx in &fold.test_indices {
            assert!(!fold.train_indices.contains(&test_idx));
        }

        // Union should cover all indices (for KFold)
        let all: HashSet<usize> = fold
            .train_indices
            .iter()
            .chain(fold.test_indices.iter())
            .copied()
            .collect();
        assert_eq!(all.len(), n_samples);
    }
}

#[test]
fn test_reproducibility_with_seed() {
    let n_samples = 50;
    let y = Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);

    let cv1 = StratifiedKFold::new(5)
        .with_shuffle(true)
        .with_seed(42);
    let cv2 = StratifiedKFold::new(5)
        .with_shuffle(true)
        .with_seed(42);
    let cv3 = StratifiedKFold::new(5)
        .with_shuffle(true)
        .with_seed(99);

    let folds1 = cv1.split(n_samples, Some(&y), None).unwrap();
    let folds2 = cv2.split(n_samples, Some(&y), None).unwrap();
    let folds3 = cv3.split(n_samples, Some(&y), None).unwrap();

    // Same seed should produce same splits
    assert_eq!(folds1[0].test_indices, folds2[0].test_indices);

    // Different seed should produce different splits
    assert_ne!(folds1[0].test_indices, folds3[0].test_indices);
}

#[test]
fn test_large_dataset_performance() {
    // Verify CV doesn't have O(n²) or worse complexity issues
    use std::time::Instant;

    let n_samples = 10_000;
    let groups = Array1::from_shape_fn(n_samples, |i| (i / 100) as i64);

    let start = Instant::now();
    let cv = GroupKFold::new(5);
    let folds = cv.split(n_samples, None, Some(&groups)).unwrap();
    let duration = start.elapsed();

    assert_eq!(folds.len(), 5);
    // Should complete in reasonable time (< 1 second for 10k samples)
    assert!(
        duration.as_secs() < 1,
        "GroupKFold too slow: {:?}",
        duration
    );
}

#[test]
fn test_cv_with_minimum_samples() {
    // Test with minimum valid samples
    let n_samples = 5;
    let cv = KFold::new(5);
    let result = cv.split(n_samples, None, None);

    assert!(result.is_ok());
    let folds = result.unwrap();
    assert_eq!(folds.len(), 5);

    // Each test fold should have exactly 1 sample
    for fold in &folds {
        assert_eq!(fold.test_indices.len(), 1);
    }
}

#[test]
fn test_stratified_maintains_class_ratios() {
    let n_samples = 100;
    // 70% class 0, 30% class 1
    let y = Array1::from_shape_fn(n_samples, |i| if i < 70 { 0.0 } else { 1.0 });

    let cv = StratifiedKFold::new(5);
    let folds = cv.split(n_samples, Some(&y), None).unwrap();

    for fold in &folds {
        let test_class_0: usize = fold.test_indices.iter().filter(|&&i| y[i] == 0.0).count();
        let test_class_1: usize = fold.test_indices.iter().filter(|&&i| y[i] == 1.0).count();

        let test_ratio = test_class_0 as f64 / test_class_1.max(1) as f64;
        // Original ratio is 70/30 ≈ 2.33, should be close
        assert!(
            test_ratio > 1.5 && test_ratio < 3.5,
            "Class ratio in test set ({}) deviates too much from original (2.33)",
            test_ratio
        );
    }
}

#[test]
fn test_kfold_without_shuffle_is_deterministic() {
    let n_samples = 30;

    let cv1 = KFold::new(5);
    let cv2 = KFold::new(5);

    let folds1 = cv1.split(n_samples, None, None).unwrap();
    let folds2 = cv2.split(n_samples, None, None).unwrap();

    // Without shuffle, should always produce identical results
    for i in 0..folds1.len() {
        assert_eq!(folds1[i].train_indices, folds2[i].train_indices);
        assert_eq!(folds1[i].test_indices, folds2[i].test_indices);
    }
}
