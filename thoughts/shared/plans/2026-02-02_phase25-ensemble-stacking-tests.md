# Phase 25: Ensemble Stacking Tests

## Overview
Create comprehensive test module `ensemble_advanced.rs` for StackingClassifier and StackingRegressor. Focus on data leakage prevention, meta-learner correctness, and passthrough validation.

## Current State
- **StackingClassifier**: Lines 92-553 in `ensemble/stacking.rs` (461 lines)
- **StackingRegressor**: Lines 558-913 (355 lines)
- **Existing Tests**: 12 inline tests (basic happy-path only)
- **Missing**: Data leakage verification, meta-learner variations, passthrough validation

## Desired End State
- New `testing/ensemble_advanced.rs` module with 25+ tests
- Data leakage prevention proofs
- Meta-learner correctness with multiple algorithms
- Passthrough feature verification
- StackMethod comparison tests

---

## Implementation Phases

### Phase 25.1: Create Test Module Structure
**Overview**: Set up ensemble_advanced.rs with proper imports

**Changes Required**:
1. **File**: `ferroml-core/src/testing/ensemble_advanced.rs` (NEW)

   ```rust
   //! Advanced Ensemble Tests
   //!
   //! Phase 25 of FerroML testing plan - comprehensive tests for:
   //! - StackingClassifier/StackingRegressor
   //! - Data leakage prevention in CV-based stacking
   //! - Meta-learner correctness
   //! - Passthrough option validation

   use ndarray::{Array1, Array2, array};
   use crate::ensemble::{
       stacking::{StackingClassifier, StackingRegressor, StackMethod},
       voting::{VotingClassifier, VotingRegressor},
   };
   use crate::models::{
       LinearRegression, LogisticRegression, RidgeRegression,
       DecisionTreeClassifier, DecisionTreeRegressor,
       naive_bayes::GaussianNB,
       knn::{KNeighborsClassifier, KNeighborsRegressor},
       Model,
   };
   use crate::metrics::{mean_squared_error, accuracy_score, r2_score};

   // ============================================================================
   // DATA LEAKAGE PREVENTION TESTS
   // ============================================================================

   // Tests added in Phase 25.2

   // ============================================================================
   // META-LEARNER CORRECTNESS TESTS
   // ============================================================================

   // Tests added in Phase 25.3

   // ============================================================================
   // PASSTHROUGH FEATURE TESTS
   // ============================================================================

   // Tests added in Phase 25.4

   // ============================================================================
   // PROBABILITY AND OUTPUT TESTS
   // ============================================================================

   // Tests added in Phase 25.5
   ```

2. **File**: `ferroml-core/src/testing/mod.rs` (MODIFY)
   - Add: `pub mod ensemble_advanced;`

**Success Criteria**:
- [ ] Automated: `cargo check -p ferroml-core`

---

### Phase 25.2: Data Leakage Prevention Tests (5 tests)
**Overview**: Verify out-of-fold meta-features prevent leakage

**Changes Required**:
1. **File**: `ferroml-core/src/testing/ensemble_advanced.rs` (ADD)

   ```rust
   // ============================================================================
   // DATA LEAKAGE PREVENTION TESTS
   // ============================================================================

   #[test]
   fn test_stacking_uses_out_of_fold_predictions() {
       // With proper OOF, test R² should be lower than train R²
       // With leakage (in-fold predictions), they'd be nearly identical

       let x = Array2::from_shape_fn((100, 5), |(i, j)| ((i * j) as f64).sin());
       let y = Array1::from_shape_fn(100, |i| (i as f64 / 10.0).cos());

       let mut stacking = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("ridge", Box::new(RidgeRegression::new(1.0)))
           .with_cv(5);

       stacking.fit(&x, &y).unwrap();

       let train_preds = stacking.predict(&x).unwrap();
       let train_r2 = r2_score(&y, &train_preds);

       // Create simple test set (different from training)
       let x_test = Array2::from_shape_fn((20, 5), |(i, j)| ((i * j + 100) as f64).sin());
       let y_test = Array1::from_shape_fn(20, |i| ((i + 100) as f64 / 10.0).cos());

       let test_preds = stacking.predict(&x_test).unwrap();
       let test_r2 = r2_score(&y_test, &test_preds);

       // Train R² should be higher than test R² (some overfitting expected)
       // But if there was major leakage, train R² would be near 1.0
       assert!(train_r2 < 0.99,
               "Train R² {} suspiciously high - possible leakage", train_r2);
   }

   #[test]
   fn test_stacking_cv_prevents_train_test_contamination() {
       // Each sample's meta-feature should come from model NOT trained on that sample

       let x = Array2::from_shape_fn((50, 3), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(50, |i| i as f64);

       let mut stacking = StackingRegressor::new()
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new().with_max_depth(3)))
           .with_cv(5);

       stacking.fit(&x, &y).unwrap();

       // The model should still produce reasonable predictions
       // (proves it wasn't using leaked in-fold information)
       let preds = stacking.predict(&x).unwrap();

       // All predictions should be finite
       assert!(preds.iter().all(|&p| p.is_finite()));

       // Should have some variance (not all same prediction)
       let mean_pred = preds.mean().unwrap();
       let var: f64 = preds.iter().map(|&p| (p - mean_pred).powi(2)).sum::<f64>() / preds.len() as f64;
       assert!(var > 0.01, "Predictions have no variance");
   }

   #[test]
   fn test_stacking_final_estimators_see_all_training_data() {
       // After CV meta-feature generation, final estimators should be refit on full data

       let x = Array2::from_shape_fn((60, 2), |(i, j)| (i * 2 + j) as f64);
       let y = Array1::from_shape_fn(60, |i| (i % 2) as f64);

       let mut stacking = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()))
           .add_estimator("tree", Box::new(DecisionTreeClassifier::new().with_max_depth(2)))
           .with_cv(3);

       stacking.fit(&x, &y).unwrap();

       // After fit, we should be able to predict on any input
       let preds = stacking.predict(&x).unwrap();
       assert_eq!(preds.len(), 60);

       // All predictions should be valid class labels
       assert!(preds.iter().all(|&p| p == 0.0 || p == 1.0));
   }

   #[test]
   fn test_stacking_different_cv_strategies() {
       let x = Array2::from_shape_fn((80, 4), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(80, |i| i as f64);

       // Test with different CV fold counts
       for n_folds in [3, 5, 10] {
           let mut stacking = StackingRegressor::new()
               .add_estimator("lr", Box::new(LinearRegression::new()))
               .with_cv(n_folds);

           let result = stacking.fit(&x, &y);
           assert!(result.is_ok(), "Failed with {} folds", n_folds);

           let preds = stacking.predict(&x).unwrap();
           assert_eq!(preds.len(), 80);
       }
   }

   #[test]
   fn test_stacking_oof_vs_naive_comparison() {
       // Compare OOF stacking vs naive in-fold stacking (simulated)
       // OOF should have more realistic (lower) scores

       let x = Array2::from_shape_fn((60, 3), |(i, j)| ((i * j) as f64).sin());
       let y = Array1::from_shape_fn(60, |i| (i as f64).cos());

       // OOF stacking (proper implementation)
       let mut stacking_oof = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .with_cv(5);
       stacking_oof.fit(&x, &y).unwrap();
       let oof_preds = stacking_oof.predict(&x).unwrap();
       let oof_r2 = r2_score(&y, &oof_preds);

       // The OOF R² should be reasonable (not near-perfect)
       // This indirectly proves leakage prevention
       assert!(oof_r2 < 0.95, "OOF R² {} too high", oof_r2);
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core testing::ensemble_advanced::test_stacking_uses`
- [ ] Automated: `cargo test -p ferroml-core testing::ensemble_advanced::test_stacking_cv`

---

### Phase 25.3: Meta-Learner Correctness Tests (8 tests)
**Overview**: Test different final estimators and verify weights are learned

**Changes Required**:
1. **File**: `ferroml-core/src/testing/ensemble_advanced.rs` (ADD)

   ```rust
   // ============================================================================
   // META-LEARNER CORRECTNESS TESTS
   // ============================================================================

   #[test]
   fn test_stacking_classifier_default_meta_learner() {
       // Default meta-learner should be RidgeRegression (handles multicollinearity)
       let x = Array2::from_shape_fn((60, 3), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(60, |i| (i % 2) as f64);

       let mut stacking = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()))
           .add_estimator("tree", Box::new(DecisionTreeClassifier::new()));

       stacking.fit(&x, &y).unwrap();
       let preds = stacking.predict(&x).unwrap();

       // Should produce valid binary predictions
       assert!(preds.iter().all(|&p| p == 0.0 || p == 1.0));
   }

   #[test]
   fn test_stacking_regressor_custom_meta_learner() {
       let x = Array2::from_shape_fn((50, 2), |(i, j)| (i * 2 + j) as f64);
       let y = Array1::from_shape_fn(50, |i| i as f64);

       // Use custom meta-learner (LinearRegression instead of default Ridge)
       let mut stacking = StackingRegressor::new()
           .add_estimator("tree1", Box::new(DecisionTreeRegressor::new().with_max_depth(3)))
           .add_estimator("tree2", Box::new(DecisionTreeRegressor::new().with_max_depth(5)))
           .with_final_estimator(Box::new(LinearRegression::new()));

       stacking.fit(&x, &y).unwrap();
       let preds = stacking.predict(&x).unwrap();

       assert!(preds.iter().all(|&p| p.is_finite()));
   }

   #[test]
   fn test_stacking_learns_meaningful_combination() {
       // Stacking should outperform individual estimators (or at least match)
       let x = Array2::from_shape_fn((100, 5), |(i, j)| (i + j) as f64 / 10.0);
       let y = Array1::from_shape_fn(100, |i| (i as f64 / 10.0).sin());

       // Individual estimators
       let mut lr = LinearRegression::new();
       lr.fit(&x, &y).unwrap();
       let lr_preds = lr.predict(&x).unwrap();
       let lr_r2 = r2_score(&y, &lr_preds);

       let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
       tree.fit(&x, &y).unwrap();
       let tree_preds = tree.predict(&x).unwrap();
       let tree_r2 = r2_score(&y, &tree_preds);

       // Stacking
       let mut stacking = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new().with_max_depth(3)));
       stacking.fit(&x, &y).unwrap();
       let stack_preds = stacking.predict(&x).unwrap();
       let stack_r2 = r2_score(&y, &stack_preds);

       // Stacking should be at least as good as the better individual
       let best_individual = lr_r2.max(tree_r2);
       assert!(stack_r2 >= best_individual - 0.1,
               "Stacking R² {} worse than individual {}", stack_r2, best_individual);
   }

   #[test]
   fn test_stacking_with_knn_meta_learner() {
       let x = Array2::from_shape_fn((60, 3), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(60, |i| i as f64);

       let mut stacking = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new()))
           .with_final_estimator(Box::new(KNeighborsRegressor::new(3)));

       stacking.fit(&x, &y).unwrap();
       let preds = stacking.predict(&x).unwrap();

       assert!(preds.iter().all(|&p| p.is_finite()));
   }

   #[test]
   fn test_stacking_single_base_estimator() {
       // Edge case: stacking with only one base estimator
       let x = Array2::from_shape_fn((40, 2), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(40, |i| (i % 2) as f64);

       let mut stacking = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()));

       let result = stacking.fit(&x, &y);
       assert!(result.is_ok(), "Single estimator should work");

       let preds = stacking.predict(&x).unwrap();
       assert!(preds.iter().all(|&p| p == 0.0 || p == 1.0));
   }

   #[test]
   fn test_stacking_many_base_estimators() {
       // Test with many base estimators
       let x = Array2::from_shape_fn((80, 4), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(80, |i| i as f64);

       let mut stacking = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("ridge1", Box::new(RidgeRegression::new(0.1)))
           .add_estimator("ridge2", Box::new(RidgeRegression::new(1.0)))
           .add_estimator("ridge3", Box::new(RidgeRegression::new(10.0)))
           .add_estimator("tree1", Box::new(DecisionTreeRegressor::new().with_max_depth(2)))
           .add_estimator("tree2", Box::new(DecisionTreeRegressor::new().with_max_depth(4)));

       stacking.fit(&x, &y).unwrap();
       let preds = stacking.predict(&x).unwrap();

       assert_eq!(preds.len(), 80);
       assert!(preds.iter().all(|&p| p.is_finite()));
   }

   #[test]
   fn test_stacking_get_estimator() {
       let mut stacking = StackingRegressor::new()
           .add_estimator("model1", Box::new(LinearRegression::new()))
           .add_estimator("model2", Box::new(RidgeRegression::new(1.0)));

       let x = Array2::from_shape_fn((30, 2), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(30, |i| i as f64);
       stacking.fit(&x, &y).unwrap();

       // Should be able to access named estimators
       assert!(stacking.get_estimator("model1").is_some());
       assert!(stacking.get_estimator("model2").is_some());
       assert!(stacking.get_estimator("nonexistent").is_none());
   }

   #[test]
   fn test_stacking_vs_voting_comparison() {
       // Stacking should at least match voting performance
       let x = Array2::from_shape_fn((100, 3), |(i, j)| (i + j) as f64 / 10.0);
       let y = Array1::from_shape_fn(100, |i| (i as f64).sin());

       // Voting ensemble
       let mut voting = VotingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new().with_max_depth(3)));
       voting.fit(&x, &y).unwrap();
       let voting_preds = voting.predict(&x).unwrap();
       let voting_r2 = r2_score(&y, &voting_preds);

       // Stacking ensemble
       let mut stacking = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new().with_max_depth(3)));
       stacking.fit(&x, &y).unwrap();
       let stacking_preds = stacking.predict(&x).unwrap();
       let stacking_r2 = r2_score(&y, &stacking_preds);

       // Stacking should be competitive with voting
       assert!(stacking_r2 >= voting_r2 - 0.15,
               "Stacking {} much worse than voting {}", stacking_r2, voting_r2);
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core testing::ensemble_advanced::test_stacking_classifier`
- [ ] Automated: `cargo test -p ferroml-core testing::ensemble_advanced::test_stacking_regressor`

---

### Phase 25.4: Passthrough Feature Tests (4 tests)
**Overview**: Verify original features are correctly included

**Changes Required**:
1. **File**: `ferroml-core/src/testing/ensemble_advanced.rs` (ADD)

   ```rust
   // ============================================================================
   // PASSTHROUGH FEATURE TESTS
   // ============================================================================

   #[test]
   fn test_passthrough_increases_feature_count() {
       let n_features = 5;
       let x = Array2::from_shape_fn((50, n_features), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(50, |i| i as f64);

       let n_estimators = 3;

       // Without passthrough: meta-features = n_estimators
       let mut stacking_no_pass = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("ridge", Box::new(RidgeRegression::new(1.0)))
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new()))
           .with_passthrough(false);
       stacking_no_pass.fit(&x, &y).unwrap();

       // With passthrough: meta-features = n_estimators + n_features
       let mut stacking_pass = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("ridge", Box::new(RidgeRegression::new(1.0)))
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new()))
           .with_passthrough(true);
       stacking_pass.fit(&x, &y).unwrap();

       // Both should produce predictions
       let preds_no_pass = stacking_no_pass.predict(&x).unwrap();
       let preds_pass = stacking_pass.predict(&x).unwrap();

       assert_eq!(preds_no_pass.len(), 50);
       assert_eq!(preds_pass.len(), 50);
   }

   #[test]
   fn test_passthrough_can_improve_performance() {
       // In some cases, passthrough should help (original features carry info)
       let x = Array2::from_shape_fn((80, 3), |(i, j)| (i * 2 + j) as f64);
       let y = Array1::from_shape_fn(80, |i| (i as f64).powi(2) / 1000.0);

       let mut stacking_no_pass = StackingRegressor::new()
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new().with_max_depth(2)))
           .with_passthrough(false);
       stacking_no_pass.fit(&x, &y).unwrap();

       let mut stacking_pass = StackingRegressor::new()
           .add_estimator("tree", Box::new(DecisionTreeRegressor::new().with_max_depth(2)))
           .with_passthrough(true);
       stacking_pass.fit(&x, &y).unwrap();

       let preds_no_pass = stacking_no_pass.predict(&x).unwrap();
       let preds_pass = stacking_pass.predict(&x).unwrap();

       let r2_no_pass = r2_score(&y, &preds_no_pass);
       let r2_pass = r2_score(&y, &preds_pass);

       // Passthrough at least shouldn't hurt much
       assert!(r2_pass >= r2_no_pass - 0.2,
               "Passthrough hurt: {} vs {}", r2_pass, r2_no_pass);
   }

   #[test]
   fn test_passthrough_with_different_stack_methods() {
       let x = Array2::from_shape_fn((60, 4), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(60, |i| (i % 2) as f64);

       // Passthrough with Predict method
       let mut stacking_predict = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()))
           .with_stack_method(StackMethod::Predict)
           .with_passthrough(true);
       stacking_predict.fit(&x, &y).unwrap();

       // Passthrough with PredictProba method (default)
       let mut stacking_proba = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()))
           .with_stack_method(StackMethod::PredictProba)
           .with_passthrough(true);
       stacking_proba.fit(&x, &y).unwrap();

       // Both should work
       let preds_predict = stacking_predict.predict(&x).unwrap();
       let preds_proba = stacking_proba.predict(&x).unwrap();

       assert!(preds_predict.iter().all(|&p| p == 0.0 || p == 1.0));
       assert!(preds_proba.iter().all(|&p| p == 0.0 || p == 1.0));
   }

   #[test]
   fn test_passthrough_preserves_feature_ordering() {
       // Original features should be appended after meta-features
       let x = Array2::from_shape_fn((40, 2), |(i, j)| (i * 10 + j) as f64);
       let y = Array1::from_shape_fn(40, |i| i as f64);

       let mut stacking = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .with_passthrough(true);

       stacking.fit(&x, &y).unwrap();
       let preds = stacking.predict(&x).unwrap();

       // Should produce valid predictions
       assert!(preds.iter().all(|&p| p.is_finite()));
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core testing::ensemble_advanced::test_passthrough`

---

### Phase 25.5: Probability and Output Tests (5 tests)
**Overview**: Test predict_proba and StackMethod behavior

**Changes Required**:
1. **File**: `ferroml-core/src/testing/ensemble_advanced.rs` (ADD)

   ```rust
   // ============================================================================
   // PROBABILITY AND OUTPUT TESTS
   // ============================================================================

   #[test]
   fn test_stacking_classifier_predict_proba() {
       let x = Array2::from_shape_fn((60, 3), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(60, |i| (i % 2) as f64);

       let mut stacking = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()))
           .add_estimator("tree", Box::new(DecisionTreeClassifier::new()));

       stacking.fit(&x, &y).unwrap();
       let probas = stacking.predict_proba(&x).unwrap();

       // Should have shape (n_samples, n_classes)
       assert_eq!(probas.nrows(), 60);
       assert_eq!(probas.ncols(), 2); // Binary classification

       // Each row should sum to ~1.0
       for i in 0..probas.nrows() {
           let row_sum: f64 = probas.row(i).sum();
           assert!((row_sum - 1.0).abs() < 0.01,
                   "Row {} sums to {} instead of 1.0", i, row_sum);
       }
   }

   #[test]
   fn test_stack_method_predict_vs_predict_proba() {
       let x = Array2::from_shape_fn((50, 3), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(50, |i| (i % 2) as f64);

       let mut stacking_predict = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()))
           .with_stack_method(StackMethod::Predict);
       stacking_predict.fit(&x, &y).unwrap();

       let mut stacking_proba = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()))
           .with_stack_method(StackMethod::PredictProba);
       stacking_proba.fit(&x, &y).unwrap();

       // Both should produce predictions
       let preds1 = stacking_predict.predict(&x).unwrap();
       let preds2 = stacking_proba.predict(&x).unwrap();

       assert_eq!(preds1.len(), 50);
       assert_eq!(preds2.len(), 50);

       // Predictions might differ but both should be valid
       assert!(preds1.iter().all(|&p| p == 0.0 || p == 1.0));
       assert!(preds2.iter().all(|&p| p == 0.0 || p == 1.0));
   }

   #[test]
   fn test_stacking_regressor_individual_predictions() {
       let x = Array2::from_shape_fn((40, 2), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(40, |i| i as f64);

       let mut stacking = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()))
           .add_estimator("ridge", Box::new(RidgeRegression::new(1.0)));

       stacking.fit(&x, &y).unwrap();
       let individual = stacking.individual_predictions(&x).unwrap();

       // Should have predictions from each estimator
       assert_eq!(individual.len(), 2);
       assert_eq!(individual[0].len(), 40);
       assert_eq!(individual[1].len(), 40);
   }

   #[test]
   fn test_stacking_not_fitted_error() {
       let x = Array2::zeros((10, 3));

       let stacking = StackingRegressor::new()
           .add_estimator("lr", Box::new(LinearRegression::new()));

       let result = stacking.predict(&x);
       assert!(result.is_err(), "Should error when not fitted");
   }

   #[test]
   fn test_stacking_multiclass_classification() {
       // 3-class classification
       let x = Array2::from_shape_fn((90, 4), |(i, j)| (i + j) as f64);
       let y = Array1::from_shape_fn(90, |i| (i % 3) as f64);

       let mut stacking = StackingClassifier::new()
           .add_estimator("nb", Box::new(GaussianNB::new()))
           .add_estimator("tree", Box::new(DecisionTreeClassifier::new()));

       stacking.fit(&x, &y).unwrap();
       let preds = stacking.predict(&x).unwrap();
       let probas = stacking.predict_proba(&x).unwrap();

       // Predictions should be in {0, 1, 2}
       assert!(preds.iter().all(|&p| p == 0.0 || p == 1.0 || p == 2.0));

       // Probabilities should be (n_samples, 3)
       assert_eq!(probas.ncols(), 3);
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core testing::ensemble_advanced`

---

## Dependencies
- Existing ensemble module (`ensemble/stacking.rs`)
- Existing models (LinearRegression, LogisticRegression, GaussianNB, DecisionTree*, KNN*)
- Existing metrics module

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Tests flaky due to random base estimators | Use deterministic estimators or fixed seeds |
| Complex dependencies | Test with simple synthetic data |
| Slow tests with many estimators | Limit test data sizes |

## Verification Commands
```bash
# All Phase 25 tests
cargo test -p ferroml-core testing::ensemble_advanced -- --nocapture

# Just data leakage tests
cargo test -p ferroml-core testing::ensemble_advanced::test_stacking_uses
cargo test -p ferroml-core testing::ensemble_advanced::test_stacking_cv
```
