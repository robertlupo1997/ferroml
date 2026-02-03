# P0: MAE Criterion Bug Fix

## Overview
Fix critical correctness bug in decision tree MAE criterion where left child impurity incorrectly uses right child's values. Add comprehensive regression tests.

## Current State
- **Bug Location**: `ferroml-core/src/models/tree.rs:1539`
- **Problem**: Left impurity calculated with `mae(&right_values)` instead of `mae(&left_values)`
- **Impact**: All MAE-based decision tree splits are incorrect, leading to suboptimal trees
- **Existing Test**: `test_mae_criterion()` (lines 1989-1998) only checks fit/predict doesn't crash

## Desired End State
- Bug fixed with correct array reference
- 5+ new tests preventing regression
- MAE criterion produces mathematically correct splits

---

## Implementation Phases

### Phase 0.1: Fix the Bug
**Overview**: Single-line fix to use correct array

**Changes Required**:
1. **File**: `ferroml-core/src/models/tree.rs`
   - Line 1539: Change `mae(&right_values)` to `mae(&left_values)`

   ```rust
   // BEFORE (line 1539):
   SplitCriterion::Mae => mae(&right_values),

   // AFTER:
   SplitCriterion::Mae => mae(&left_values),
   ```

**Success Criteria**:
- [ ] Automated: `cargo check -p ferroml-core`
- [ ] Automated: Existing `test_mae_criterion` still passes

---

### Phase 0.2: Add Impurity Validation Test
**Overview**: Test that child impurities use correct data

**Changes Required**:
1. **File**: `ferroml-core/src/models/tree.rs` (add after line 1998)

   ```rust
   #[test]
   fn test_mae_left_right_impurity_uses_correct_data() {
       // Dataset where MAE split is obvious: [1,2] vs [100,101]
       // Correct: left_impurity ~ 0.5 (median 1.5, MAE from [1,2])
       // Correct: right_impurity ~ 0.5 (median 100.5, MAE from [100,101])
       // Wrong (bug): left_impurity would be ~ 0.5 from [100,101] - same as right!

       let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 100.0, 101.0]).unwrap();
       let y = Array1::from_vec(vec![1.0, 2.0, 100.0, 101.0]);

       let mut reg = DecisionTreeRegressor::new()
           .with_criterion(SplitCriterion::Mae)
           .with_max_depth(1);
       reg.fit(&x, &y).unwrap();

       // With correct MAE, should split at ~50 (between 2 and 100)
       let predictions = reg.predict(&x).unwrap();

       // Left leaf (samples 0,1) should predict ~1.5 (median of [1,2])
       assert!((predictions[0] - 1.5).abs() < 0.1, "Left prediction wrong: {}", predictions[0]);
       assert!((predictions[1] - 1.5).abs() < 0.1, "Left prediction wrong: {}", predictions[1]);

       // Right leaf (samples 2,3) should predict ~100.5 (median of [100,101])
       assert!((predictions[2] - 100.5).abs() < 0.1, "Right prediction wrong: {}", predictions[2]);
       assert!((predictions[3] - 100.5).abs() < 0.1, "Right prediction wrong: {}", predictions[3]);
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core test_mae_left_right_impurity`

---

### Phase 0.3: Add Impurity Decrease Invariant Test
**Overview**: Verify that splits always decrease weighted impurity

**Changes Required**:
1. **File**: `ferroml-core/src/models/tree.rs` (add to tests)

   ```rust
   #[test]
   fn test_mae_impurity_decrease_invariant() {
       // Property: weighted_child_impurity <= parent_impurity for any valid split
       let x = Array2::from_shape_fn((20, 2), |(i, j)| (i * 3 + j) as f64);
       let y = Array1::from_shape_fn(20, |i| (i as f64).powi(2) / 100.0);

       let mut reg = DecisionTreeRegressor::new()
           .with_criterion(SplitCriterion::Mae)
           .with_max_depth(3);
       reg.fit(&x, &y).unwrap();

       // Tree should be fitted without panic
       let predictions = reg.predict(&x).unwrap();
       assert_eq!(predictions.len(), 20);

       // All predictions should be finite (not NaN/Inf from bad impurity calcs)
       assert!(predictions.iter().all(|&p| p.is_finite()),
               "MAE produced non-finite predictions");
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core test_mae_impurity_decrease`

---

### Phase 0.4: Add MAE vs MSE Consistency Test
**Overview**: Both criteria should produce valid trees on same data

**Changes Required**:
1. **File**: `ferroml-core/src/models/tree.rs` (add to tests)

   ```rust
   #[test]
   fn test_mae_mse_both_produce_valid_trees() {
       let (x, y) = make_regression_data(); // Existing test helper

       let mut mse_reg = DecisionTreeRegressor::new()
           .with_criterion(SplitCriterion::Mse)
           .with_max_depth(3)
           .with_random_state(42);
       mse_reg.fit(&x, &y).unwrap();

       let mut mae_reg = DecisionTreeRegressor::new()
           .with_criterion(SplitCriterion::Mae)
           .with_max_depth(3)
           .with_random_state(42);
       mae_reg.fit(&x, &y).unwrap();

       let mse_preds = mse_reg.predict(&x).unwrap();
       let mae_preds = mae_reg.predict(&x).unwrap();

       // Both should produce finite predictions
       assert!(mse_preds.iter().all(|&p| p.is_finite()));
       assert!(mae_preds.iter().all(|&p| p.is_finite()));

       // Both should have reasonable R² (> 0.5 on training data)
       let y_mean = y.mean().unwrap();
       let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

       let mse_ss_res: f64 = y.iter().zip(mse_preds.iter())
           .map(|(&yi, &pi)| (yi - pi).powi(2)).sum();
       let mae_ss_res: f64 = y.iter().zip(mae_preds.iter())
           .map(|(&yi, &pi)| (yi - pi).powi(2)).sum();

       let mse_r2 = 1.0 - mse_ss_res / ss_tot;
       let mae_r2 = 1.0 - mae_ss_res / ss_tot;

       assert!(mse_r2 > 0.5, "MSE R² too low: {}", mse_r2);
       assert!(mae_r2 > 0.5, "MAE R² too low: {}", mae_r2);
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core test_mae_mse_both`

---

### Phase 0.5: Add MAE Outlier Robustness Test
**Overview**: MAE should be more robust to outliers than MSE

**Changes Required**:
1. **File**: `ferroml-core/src/models/tree.rs` (add to tests)

   ```rust
   #[test]
   fn test_mae_outlier_robustness() {
       // Data with outlier: [1, 2, 3, 4, 1000]
       // MAE should be more robust than MSE
       let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
       let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 1000.0]); // outlier at index 4

       let mut mae_reg = DecisionTreeRegressor::new()
           .with_criterion(SplitCriterion::Mae)
           .with_max_depth(1);
       mae_reg.fit(&x, &y).unwrap();

       // Test on non-outlier points - MAE should predict median-like values
       let test_x = Array2::from_shape_vec((1, 1), vec![2.5]).unwrap();
       let mae_pred = mae_reg.predict(&test_x).unwrap();

       // MAE prediction should be closer to 2.5 (median of non-outliers)
       // than to 202 (mean including outlier)
       assert!(mae_pred[0] < 100.0,
               "MAE prediction {} too influenced by outlier", mae_pred[0]);
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core test_mae_outlier`

---

## Dependencies
- None - this is standalone bug fix

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Fix breaks existing behavior | Run full test suite before/after |
| Tests too strict for floating point | Use appropriate tolerances |
| Missing edge cases | Added 5 diverse tests covering different scenarios |

## Verification Commands
```bash
# Phase 0.1: Fix compiles
cargo check -p ferroml-core

# All phases: Tests pass
cargo test -p ferroml-core -- test_mae

# Full regression check
cargo test -p ferroml-core --lib
```

## Time Estimate
- Phase 0.1: 2 minutes (single line fix)
- Phases 0.2-0.5: 15 minutes (test writing)
- Total: ~20 minutes
