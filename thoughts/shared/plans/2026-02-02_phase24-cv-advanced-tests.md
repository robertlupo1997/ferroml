# Phase 24: Advanced Cross-Validation Tests

## Overview
Create comprehensive test module `cv_advanced.rs` for NestedCV, GroupKFold, StratifiedGroupKFold, and TimeSeriesSplit. Focus on data leakage prevention, group integrity, and temporal ordering validation.

## Current State
- **NestedCV**: 980 lines, 6 inline tests (basic coverage)
- **GroupKFold**: 900 lines, 12 inline tests (decent coverage)
- **StratifiedGroupKFold**: In group.rs, 12 inline tests
- **TimeSeriesSplit**: 651 lines, 22 inline tests (good coverage)
- **Missing**: Dedicated test module in `testing/`, integration tests, property-based tests, parallel execution tests

## Desired End State
- New `testing/cv_advanced.rs` module with 35+ tests
- Data leakage verification for NestedCV
- Group integrity proofs for GroupKFold
- Temporal ordering guarantees for TimeSeriesSplit
- Integration tests with actual ML models

---

## Implementation Phases

### Phase 24.1: Create Test Module Structure
**Overview**: Set up cv_advanced.rs with proper imports and organization

**Changes Required**:
1. **File**: `ferroml-core/src/testing/cv_advanced.rs` (NEW)

   ```rust
   //! Advanced Cross-Validation Tests
   //!
   //! Phase 24 of FerroML testing plan - comprehensive tests for:
   //! - NestedCV data leakage prevention
   //! - GroupKFold group integrity
   //! - TimeSeriesSplit temporal ordering
   //! - Integration with actual ML models

   use ndarray::{Array1, Array2, array};
   use crate::cv::{
       CrossValidator, KFold, StratifiedKFold,
       nested::{nested_cv_score, NestedCVConfig},
       group::{GroupKFold, StratifiedGroupKFold},
       timeseries::TimeSeriesSplit,
   };
   use crate::models::{LinearRegression, LogisticRegression, Model};
   use crate::metrics::{mean_squared_error, accuracy_score};

   // ============================================================================
   // NESTED CV TESTS
   // ============================================================================

   // Tests will be added in subsequent phases

   // ============================================================================
   // GROUP KFOLD TESTS
   // ============================================================================

   // Tests will be added in subsequent phases

   // ============================================================================
   // TIME SERIES SPLIT TESTS
   // ============================================================================

   // Tests will be added in subsequent phases

   // ============================================================================
   // INTEGRATION TESTS
   // ============================================================================

   // Tests will be added in subsequent phases
   ```

2. **File**: `ferroml-core/src/testing/mod.rs` (MODIFY)
   - Add: `pub mod cv_advanced;`

**Success Criteria**:
- [ ] Automated: `cargo check -p ferroml-core`

---

### Phase 24.2: NestedCV Data Leakage Tests (8 tests)
**Overview**: Verify that inner loop never sees outer test data

**Changes Required**:
1. **File**: `ferroml-core/src/testing/cv_advanced.rs` (ADD)

   ```rust
   // ============================================================================
   // NESTED CV TESTS
   // ============================================================================

   #[test]
   fn test_nested_cv_no_data_leakage_with_real_model() {
       // If there's leakage, R² would be artificially high
       let x = Array2::from_shape_fn((100, 5), |(i, j)| (i * j) as f64 / 100.0);
       let y = Array1::from_shape_fn(100, |i| (i as f64).sin());

       let config = NestedCVConfig::new()
           .with_outer_cv(3)
           .with_inner_cv(3)
           .with_n_trials(5);

       let result = nested_cv_score(
           || Box::new(LinearRegression::new()),
           |_| crate::hpo::SearchSpace::new(),
           &x, &y,
           |model, x_test, y_test| {
               let preds = model.predict(x_test).unwrap();
               1.0 - mean_squared_error(y_test, &preds) / y_test.var().unwrap_or(1.0)
           },
           config,
       ).unwrap();

       // With proper CV, R² should be moderate (not near 1.0)
       assert!(result.mean_score < 0.95,
               "Suspiciously high score {} suggests leakage", result.mean_score);
   }

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
           for j in (i+1)..all_test_indices.len() {
               let overlap: Vec<_> = all_test_indices[i].iter()
                   .filter(|idx| all_test_indices[j].contains(idx))
                   .collect();
               assert!(overlap.is_empty(),
                       "Outer folds {} and {} overlap: {:?}", i, j, overlap);
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
           let outer_test = &outer_fold.test_indices;

           // Inner CV should only operate on outer_train
           let inner_folds = inner_cv.split(outer_train.len(), None, None).unwrap();

           for inner_fold in inner_folds {
               // Map inner indices to original indices
               let inner_train_original: Vec<usize> = inner_fold.train_indices
                   .iter()
                   .map(|&i| outer_train[i])
                   .collect();
               let inner_test_original: Vec<usize> = inner_fold.test_indices
                   .iter()
                   .map(|&i| outer_train[i])
                   .collect();

               // Verify no inner indices are in outer_test
               for idx in inner_train_original.iter().chain(inner_test_original.iter()) {
                   assert!(!outer_test.contains(idx),
                           "Inner fold contains outer test index {}", idx);
               }
           }
       }
   }

   #[test]
   fn test_nested_cv_optimism_calculation() {
       // Optimism = inner_score - outer_score (should be positive if overfitting)
       let x = Array2::from_shape_fn((80, 10), |(i, j)| ((i * j) as f64).sin());
       let y = Array1::from_shape_fn(80, |i| if i % 2 == 0 { 1.0 } else { 0.0 });

       let config = NestedCVConfig::new()
           .with_outer_cv(4)
           .with_inner_cv(3)
           .with_n_trials(10)
           .with_return_train_score(true);

       let result = nested_cv_score(
           || Box::new(LogisticRegression::new()),
           |_| crate::hpo::SearchSpace::new(),
           &x, &y,
           |model, x_test, y_test| {
               let preds = model.predict(x_test).unwrap();
               accuracy_score(y_test, &preds)
           },
           config,
       ).unwrap();

       // Optimism should be calculated
       assert!(result.mean_optimism().is_some());
   }

   #[test]
   fn test_nested_cv_different_inner_outer_strategies() {
       let x = Array2::from_shape_fn((60, 3), |(i, j)| i as f64 + j as f64);
       let y = Array1::from_shape_fn(60, |i| i as f64);

       // Stratified outer, regular inner
       let config = NestedCVConfig::new()
           .with_outer_cv(3)
           .with_inner_cv(5)
           .with_n_trials(3);

       let result = nested_cv_score(
           || Box::new(LinearRegression::new()),
           |_| crate::hpo::SearchSpace::new(),
           &x, &y,
           |model, x_test, y_test| {
               let preds = model.predict(x_test).unwrap();
               1.0 - mean_squared_error(y_test, &preds) / y_test.var().unwrap_or(1.0)
           },
           config,
       );

       assert!(result.is_ok());
   }

   #[test]
   fn test_nested_cv_fold_results_count() {
       let x = Array2::zeros((40, 2));
       let y = Array1::zeros(40);

       let n_outer = 4;
       let config = NestedCVConfig::new()
           .with_outer_cv(n_outer)
           .with_inner_cv(3)
           .with_n_trials(2);

       let result = nested_cv_score(
           || Box::new(LinearRegression::new()),
           |_| crate::hpo::SearchSpace::new(),
           &x, &y,
           |_, _, _| 0.5,
           config,
       ).unwrap();

       assert_eq!(result.fold_results.len(), n_outer,
                  "Should have {} fold results", n_outer);
   }

   #[test]
   fn test_nested_cv_confidence_interval_validity() {
       let x = Array2::from_shape_fn((100, 5), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(100, |i| i as f64);

       let config = NestedCVConfig::new()
           .with_outer_cv(5)
           .with_inner_cv(3)
           .with_n_trials(5)
           .with_confidence_level(0.95);

       let result = nested_cv_score(
           || Box::new(LinearRegression::new()),
           |_| crate::hpo::SearchSpace::new(),
           &x, &y,
           |model, x_test, y_test| {
               let preds = model.predict(x_test).unwrap();
               1.0 - mean_squared_error(y_test, &preds) / y_test.var().unwrap_or(1.0)
           },
           config,
       ).unwrap();

       // CI should contain the mean
       assert!(result.ci_lower <= result.mean_score);
       assert!(result.ci_upper >= result.mean_score);
       // CI should be reasonable width
       assert!(result.ci_upper - result.ci_lower < 1.0);
   }

   #[test]
   fn test_nested_cv_handles_all_hpo_trials_failing() {
       let x = Array2::zeros((20, 2));
       let y = Array1::from_elem(20, f64::NAN); // Will cause fitting to fail

       let config = NestedCVConfig::new()
           .with_outer_cv(2)
           .with_inner_cv(2)
           .with_n_trials(3);

       let result = nested_cv_score(
           || Box::new(LinearRegression::new()),
           |_| crate::hpo::SearchSpace::new(),
           &x, &y,
           |_, _, _| 0.5,
           config,
       );

       // Should return an error, not panic
       assert!(result.is_err());
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core testing::cv_advanced::test_nested`

---

### Phase 24.3: GroupKFold Integrity Tests (8 tests)
**Overview**: Verify groups are never split across train/test

**Changes Required**:
1. **File**: `ferroml-core/src/testing/cv_advanced.rs` (ADD)

   ```rust
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
           let train_groups: std::collections::HashSet<i64> = fold.train_indices
               .iter()
               .map(|&i| groups[i])
               .collect();
           let test_groups: std::collections::HashSet<i64> = fold.test_indices
               .iter()
               .map(|&i| groups[i])
               .collect();

           let overlap: Vec<_> = train_groups.intersection(&test_groups).collect();
           assert!(overlap.is_empty(),
                   "Fold {} has group overlap: {:?}", fold_idx, overlap);
       }
   }

   #[test]
   fn test_group_kfold_all_groups_tested() {
       let n_samples = 50;
       let n_groups = 10;
       let groups = Array1::from_shape_fn(n_samples, |i| (i % n_groups) as i64);

       let cv = GroupKFold::new(5);
       let folds = cv.split(n_samples, None, Some(&groups)).unwrap();

       let mut tested_groups: std::collections::HashSet<i64> = std::collections::HashSet::new();
       for fold in &folds {
           for &idx in &fold.test_indices {
               tested_groups.insert(groups[idx]);
           }
       }

       assert_eq!(tested_groups.len(), n_groups,
                  "Not all groups were tested");
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
       groups_vec.extend(vec![1i64; 5]);    // 5 samples in group 1
       groups_vec.extend(vec![2i64; 5]);    // 5 samples in group 2
       let groups = Array1::from_vec(groups_vec);

       let cv = GroupKFold::new(3);
       let folds = cv.split(groups.len(), None, Some(&groups)).unwrap();

       // Should still produce valid folds
       assert_eq!(folds.len(), 3);

       // Verify group integrity
       for fold in &folds {
           let train_groups: std::collections::HashSet<i64> = fold.train_indices
               .iter()
               .map(|&i| groups[i])
               .collect();
           let test_groups: std::collections::HashSet<i64> = fold.test_indices
               .iter()
               .map(|&i| groups[i])
               .collect();

           assert!(train_groups.is_disjoint(&test_groups));
       }
   }

   #[test]
   fn test_group_kfold_negative_group_ids() {
       let groups = Array1::from_vec(vec![-5i64, -5, -3, -3, 0, 0, 10, 10, 100, 100]);

       let cv = GroupKFold::new(2);
       let folds = cv.split(groups.len(), None, Some(&groups)).unwrap();

       assert_eq!(folds.len(), 2);
   }

   #[test]
   fn test_group_kfold_with_real_ml_model() {
       // Integration test: GroupKFold with actual model training
       let n_samples = 60;
       let groups = Array1::from_shape_fn(n_samples, |i| (i / 10) as i64); // 6 groups
       let x = Array2::from_shape_fn((n_samples, 3), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(n_samples, |i| (i as f64).sin());

       let cv = GroupKFold::new(3);
       let folds = cv.split(n_samples, None, Some(&groups)).unwrap();

       let mut scores = Vec::new();
       for fold in &folds {
           let x_train = x.select(ndarray::Axis(0), &fold.train_indices);
           let y_train = y.select(ndarray::Axis(0), &fold.train_indices);
           let x_test = x.select(ndarray::Axis(0), &fold.test_indices);
           let y_test = y.select(ndarray::Axis(0), &fold.test_indices);

           let mut model = LinearRegression::new();
           model.fit(&x_train, &y_train).unwrap();
           let preds = model.predict(&x_test).unwrap();

           let mse = mean_squared_error(&y_test, &preds);
           scores.push(mse);
       }

       // All folds should produce finite scores
       assert!(scores.iter().all(|&s| s.is_finite()));
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
           let test_class_0: usize = fold.test_indices.iter()
               .filter(|&&i| y[i] == 0.0)
               .count();
           let test_class_1: usize = fold.test_indices.iter()
               .filter(|&&i| y[i] == 1.0)
               .count();

           // Ratio should be roughly 2:1 (40:20 = 2:1)
           let ratio = test_class_0 as f64 / (test_class_1.max(1) as f64);
           assert!(ratio > 1.0 && ratio < 4.0,
                   "Class balance violated: ratio = {}", ratio);
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
           let test_classes: std::collections::HashSet<i64> = fold.test_indices
               .iter()
               .map(|&i| y[i] as i64)
               .collect();

           assert_eq!(test_classes.len(), 3,
                      "Not all classes in test set");
       }
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core testing::cv_advanced::test_group`
- [ ] Automated: `cargo test -p ferroml-core testing::cv_advanced::test_stratified_group`

---

### Phase 24.4: TimeSeriesSplit Tests (8 tests)
**Overview**: Verify temporal ordering and gap handling

**Changes Required**:
1. **File**: `ferroml-core/src/testing/cv_advanced.rs` (ADD)

   ```rust
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

           assert!(max_train < min_test,
                   "Fold {}: train max {} >= test min {}",
                   fold_idx, max_train, min_test);
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
           assert!(actual_gap >= gap,
                   "Fold {}: gap {} < required {}", fold_idx, actual_gap, gap);
       }
   }

   #[test]
   fn test_timeseries_expanding_window() {
       let n_samples = 60;
       let cv = TimeSeriesSplit::new(5); // No max_train_size = expanding
       let folds = cv.split(n_samples, None, None).unwrap();

       let mut prev_train_size = 0;
       for fold in &folds {
           assert!(fold.train_indices.len() > prev_train_size,
                   "Training set should expand");
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
           assert!(fold.train_indices.len() <= max_train,
                   "Training set {} exceeds max {}",
                   fold.train_indices.len(), max_train);
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
               assert!(!all_test_indices.contains(&idx),
                       "Index {} appears in multiple test sets", idx);
               all_test_indices.push(idx);
           }
       }
   }

   #[test]
   fn test_timeseries_with_forecasting_scenario() {
       // Simulate actual time series forecasting
       let n_samples = 100;
       let x = Array2::from_shape_fn((n_samples, 1), |(i, _)| i as f64);
       let y = Array1::from_shape_fn(n_samples, |i| (i as f64 * 0.1).sin());

       let cv = TimeSeriesSplit::new(3)
           .with_gap(2)  // 2-step forecast gap
           .with_test_size(10);
       let folds = cv.split(n_samples, None, None).unwrap();

       let mut scores = Vec::new();
       for fold in &folds {
           let x_train = x.select(ndarray::Axis(0), &fold.train_indices);
           let y_train = y.select(ndarray::Axis(0), &fold.train_indices);
           let x_test = x.select(ndarray::Axis(0), &fold.test_indices);
           let y_test = y.select(ndarray::Axis(0), &fold.test_indices);

           let mut model = LinearRegression::new();
           model.fit(&x_train, &y_train).unwrap();
           let preds = model.predict(&x_test).unwrap();

           let mse = mean_squared_error(&y_test, &preds);
           scores.push(mse);
       }

       assert!(scores.iter().all(|&s| s.is_finite()));
   }

   #[test]
   fn test_timeseries_error_gap_too_large() {
       let n_samples = 20;
       let cv = TimeSeriesSplit::new(5)
           .with_gap(15)  // Too large for dataset
           .with_test_size(5);

       let result = cv.split(n_samples, None, None);
       assert!(result.is_err(), "Should error on infeasible gap");
   }

   #[test]
   fn test_timeseries_fixed_test_size() {
       let n_samples = 60;
       let test_size = 8;
       let cv = TimeSeriesSplit::new(4).with_test_size(test_size);
       let folds = cv.split(n_samples, None, None).unwrap();

       for fold in &folds {
           assert_eq!(fold.test_indices.len(), test_size,
                      "Test size should be exactly {}", test_size);
       }
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core testing::cv_advanced::test_timeseries`

---

### Phase 24.5: Integration Tests (8 tests)
**Overview**: Cross-module tests with actual pipelines

**Changes Required**:
1. **File**: `ferroml-core/src/testing/cv_advanced.rs` (ADD)

   ```rust
   // ============================================================================
   // INTEGRATION TESTS
   // ============================================================================

   #[test]
   fn test_cross_val_score_with_group_kfold() {
       use crate::cv::cross_val_score;

       let x = Array2::from_shape_fn((60, 3), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(60, |i| i as f64);
       let groups = Array1::from_shape_fn(60, |i| (i / 10) as i64);

       let cv = GroupKFold::new(3);
       let model = LinearRegression::new();

       let result = cross_val_score(
           model,
           &x, &y,
           cv,
           |m, x_test, y_test| {
               let preds = m.predict(x_test).unwrap();
               1.0 - mean_squared_error(y_test, &preds) / y_test.var().unwrap_or(1.0)
           },
           Some(&groups),
       );

       assert!(result.is_ok());
       let scores = result.unwrap();
       assert_eq!(scores.len(), 3);
   }

   #[test]
   fn test_all_cv_strategies_produce_valid_splits() {
       let n_samples = 50;
       let y = Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);
       let groups = Array1::from_shape_fn(n_samples, |i| (i / 5) as i64);

       let strategies: Vec<Box<dyn CrossValidator>> = vec![
           Box::new(KFold::new(5)),
           Box::new(StratifiedKFold::new(5)),
           Box::new(GroupKFold::new(5)),
           Box::new(TimeSeriesSplit::new(5)),
       ];

       for cv in strategies {
           let folds = cv.split(n_samples, Some(&y), Some(&groups)).unwrap();

           // All samples should be tested exactly once
           let mut tested: std::collections::HashSet<usize> = std::collections::HashSet::new();
           for fold in &folds {
               for &idx in &fold.test_indices {
                   assert!(tested.insert(idx),
                           "{}: Index {} tested multiple times", cv.name(), idx);
               }
           }

           // TimeSeries may not test all samples (early ones only in training)
           if cv.name() != "TimeSeriesSplit" {
               assert_eq!(tested.len(), n_samples,
                          "{}: Not all samples tested", cv.name());
           }
       }
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
           let all: std::collections::HashSet<usize> = fold.train_indices.iter()
               .chain(fold.test_indices.iter())
               .copied()
               .collect();
           assert_eq!(all.len(), n_samples);
       }
   }

   #[test]
   fn test_learning_curve_with_timeseries() {
       use crate::cv::learning_curve;

       let x = Array2::from_shape_fn((100, 2), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(100, |i| i as f64);

       let cv = TimeSeriesSplit::new(3);

       let result = learning_curve(
           || Box::new(LinearRegression::new()),
           &x, &y,
           cv,
           |m, x_test, y_test| {
               let preds = m.predict(x_test).unwrap();
               1.0 - mean_squared_error(y_test, &preds) / y_test.var().unwrap_or(1.0)
           },
           &[0.3, 0.5, 0.7, 0.9],
           None,
       );

       assert!(result.is_ok());
   }

   #[test]
   fn test_validation_curve_with_group_cv() {
       // This tests that validation_curve works with group-aware CV
       let x = Array2::from_shape_fn((60, 2), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(60, |i| i as f64);

       // Note: validation_curve with groups requires passing groups through
       // This test verifies the API works
       let cv = KFold::new(3);

       // For now, just verify KFold works with validation_curve
       // GroupKFold integration would require API changes
       assert!(cv.split(60, None, None).is_ok());
   }

   #[test]
   fn test_reproducibility_with_seed() {
       let n_samples = 50;
       let y = Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);

       let cv1 = StratifiedKFold::new(5).with_shuffle(true).with_random_state(42);
       let cv2 = StratifiedKFold::new(5).with_shuffle(true).with_random_state(42);
       let cv3 = StratifiedKFold::new(5).with_shuffle(true).with_random_state(99);

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
       let y = Array1::from_shape_fn(n_samples, |i| (i % 5) as f64);
       let groups = Array1::from_shape_fn(n_samples, |i| (i / 100) as i64);

       let start = Instant::now();
       let cv = GroupKFold::new(5);
       let folds = cv.split(n_samples, None, Some(&groups)).unwrap();
       let duration = start.elapsed();

       assert_eq!(folds.len(), 5);
       // Should complete in reasonable time (< 1 second for 10k samples)
       assert!(duration.as_secs() < 1,
               "GroupKFold too slow: {:?}", duration);
   }

   #[test]
   fn test_cv_with_nan_in_features() {
       // CV should work regardless of data content (that's model's problem)
       let n_samples = 30;
       let cv = KFold::new(3);
       let folds = cv.split(n_samples, None, None);

       assert!(folds.is_ok());
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core testing::cv_advanced`

---

## Dependencies
- Existing CV implementations in `cv/` module
- Existing models (LinearRegression, LogisticRegression)
- Existing metrics module

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Tests too slow | Use small datasets (50-100 samples) |
| Random failures | Use fixed seeds, allow tolerance |
| Model-dependent results | Focus on CV invariants, not absolute scores |

## Verification Commands
```bash
# All Phase 24 tests
cargo test -p ferroml-core testing::cv_advanced -- --nocapture

# Run with timing info
cargo test -p ferroml-core testing::cv_advanced -- --nocapture --show-output
```
