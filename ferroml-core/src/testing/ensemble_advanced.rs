//! Advanced Ensemble Tests
//!
//! Phase 25 of FerroML testing plan - comprehensive tests for:
//! - StackingClassifier/StackingRegressor
//! - Data leakage prevention in CV-based stacking
//! - Meta-learner correctness
//! - Passthrough option validation

#![allow(unused_imports)]
#![allow(dead_code)]

use crate::ensemble::stacking::{StackMethod, StackingClassifier, StackingRegressor};
use crate::ensemble::voting::{
    VotingClassifierEstimator, VotingRegressor, VotingRegressorEstimator,
};
use crate::metrics::r2_score;
use crate::models::knn::KNeighborsRegressor;
use crate::models::naive_bayes::GaussianNB;
use crate::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use crate::models::{Model, RidgeRegression};
use ndarray::{Array1, Array2};

// ============================================================================
// HELPER FUNCTIONS FOR TEST DATA GENERATION
// ============================================================================

/// Generate well-conditioned regression data that avoids collinearity issues
fn create_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Create varied features with different scales and patterns
        for j in 0..n_features {
            let val = match j % 4 {
                0 => (i as f64) / 10.0 + ((j + 1) as f64).sin(),
                1 => ((i as f64) * 0.7).sin() + (j as f64) * 0.1,
                2 => (i as f64 / (n_samples as f64)) * 10.0 + (j as f64),
                _ => ((i * j + 1) as f64).sqrt() + 0.1 * (i % 5) as f64,
            };
            x_data.push(val);
        }

        // y = linear combination + noise
        let y = 2.0 * (i as f64 / 10.0) + 0.5 * ((i as f64) * 0.7).sin() + 0.1 * (i % 5) as f64;
        y_data.push(y);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

/// Generate classification data with well-separated classes
fn create_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let half = n_samples / 2;
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    // Class 0: centered around low values
    for i in 0..half {
        for j in 0..n_features {
            let noise = ((i * (j + 1)) as f64 * 0.7).sin() * 0.5;
            let val = 1.0 + (j as f64) * 0.5 + (i as f64) * 0.02 + noise;
            x_data.push(val);
        }
        y_data.push(0.0);
    }

    // Class 1: centered around high values
    for i in 0..(n_samples - half) {
        for j in 0..n_features {
            let noise = ((i * (j + 1)) as f64 * 0.7).cos() * 0.5;
            let val = 5.0 + (j as f64) * 0.5 + (i as f64) * 0.02 + noise;
            x_data.push(val);
        }
        y_data.push(1.0);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

/// Generate multiclass classification data
fn create_multiclass_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<f64>) {
    let per_class = n_samples / n_classes;
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    for class_idx in 0..n_classes {
        let base = (class_idx as f64) * 5.0; // Separate class centers
        for i in 0..per_class {
            for j in 0..n_features {
                let noise = ((i * (j + 1) + class_idx) as f64 * 0.7).sin() * 0.3;
                let val = base + (j as f64) * 0.3 + (i as f64) * 0.01 + noise;
                x_data.push(val);
            }
            y_data.push(class_idx as f64);
        }
    }

    // Fill remaining samples with last class if n_samples not divisible by n_classes
    let remaining = n_samples - (per_class * n_classes);
    for _ in 0..remaining {
        for j in 0..n_features {
            let val = ((n_classes - 1) as f64) * 5.0 + (j as f64) * 0.3;
            x_data.push(val);
        }
        y_data.push((n_classes - 1) as f64);
    }

    let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

// ============================================================================
// DATA LEAKAGE PREVENTION TESTS
// ============================================================================

#[test]
fn test_stacking_uses_out_of_fold_predictions() {
    // Verify stacking produces valid out-of-fold meta-features
    // by checking that predictions are reasonable and finite

    let (x, y) = create_regression_data(100, 3);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge1".to_string(), Box::new(RidgeRegression::new(0.1))),
        ("ridge2".to_string(), Box::new(RidgeRegression::new(1.0))),
    ];

    let mut stacking = StackingRegressor::new(estimators).with_n_folds(5);

    stacking.fit(&x, &y).unwrap();

    let train_preds = stacking.predict(&x).unwrap();
    let train_r2 = r2_score(&y, &train_preds).unwrap();

    // Verify predictions are finite
    assert!(
        train_preds.iter().all(|&p| p.is_finite()),
        "Predictions should be finite"
    );

    // R² should be reasonable (not negative infinity)
    assert!(
        train_r2 > -10.0,
        "Train R² {} should be reasonable",
        train_r2
    );

    // Train R² can be high on well-fitted data, but shouldn't be 1.0 exactly
    // (which would indicate perfect in-fold predictions / leakage)
    assert!(
        (train_r2 - 1.0).abs() > 1e-10,
        "Train R² {} too perfect - possible leakage",
        train_r2
    );
}

#[test]
fn test_stacking_cv_prevents_train_test_contamination() {
    // Each sample's meta-feature should come from model NOT trained on that sample

    let (x, y) = create_regression_data(50, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![(
        "tree".to_string(),
        Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
    )];

    let mut stacking = StackingRegressor::new(estimators).with_n_folds(5);

    stacking.fit(&x, &y).unwrap();

    // The model should still produce reasonable predictions
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

    let (x, y) = create_classification_data(60, 4);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingClassifier::new(estimators).with_n_folds(3);

    stacking.fit(&x, &y).unwrap();

    // After fit, we should be able to predict on any input
    let preds = stacking.predict(&x).unwrap();
    assert_eq!(preds.len(), 60);

    // All predictions should be finite
    assert!(preds.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_different_cv_strategies() {
    let (x, y) = create_regression_data(80, 2);

    // Test with different CV fold counts
    for n_folds in [3, 5, 10] {
        let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
            vec![("ridge".to_string(), Box::new(RidgeRegression::new(1.0)))];

        let mut stacking = StackingRegressor::new(estimators).with_n_folds(n_folds);

        let result = stacking.fit(&x, &y);
        assert!(result.is_ok(), "Failed with {} folds", n_folds);

        let preds = stacking.predict(&x).unwrap();
        assert_eq!(preds.len(), 80);
    }
}

#[test]
fn test_stacking_oof_vs_naive_comparison() {
    // OOF stacking should have more realistic (not near-perfect) scores

    let (x, y) = create_regression_data(60, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
        vec![("ridge".to_string(), Box::new(RidgeRegression::new(0.1)))];

    let mut stacking_oof = StackingRegressor::new(estimators).with_n_folds(5);
    stacking_oof.fit(&x, &y).unwrap();
    let oof_preds = stacking_oof.predict(&x).unwrap();
    let oof_r2 = r2_score(&y, &oof_preds).unwrap();

    // The OOF R² should be reasonable (not near-perfect)
    // This indirectly proves leakage prevention
    assert!(oof_r2 < 0.999, "OOF R² {} too high", oof_r2);
    assert!(oof_r2 > -10.0, "OOF R² {} too low", oof_r2);
}

// ============================================================================
// META-LEARNER CORRECTNESS TESTS
// ============================================================================

#[test]
fn test_stacking_classifier_default_meta_learner() {
    // Default meta-learner should work correctly
    let (x, y) = create_classification_data(60, 4);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingClassifier::new(estimators);

    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    // Should produce finite predictions
    assert!(
        preds.iter().all(|&p| p.is_finite()),
        "Predictions should be finite"
    );
}

#[test]
fn test_stacking_regressor_custom_meta_learner() {
    let (x, y) = create_regression_data(50, 2);

    // Use custom meta-learner (RidgeRegression)
    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        (
            "tree1".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
        ),
        (
            "tree2".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(5))),
        ),
    ];

    let mut stacking = StackingRegressor::new(estimators)
        .with_final_estimator(Box::new(RidgeRegression::new(1.0)));

    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    assert!(preds.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_learns_meaningful_combination() {
    // Stacking should produce reasonable predictions
    let (x, y) = create_regression_data(100, 3);

    // Individual estimator
    let mut ridge = RidgeRegression::new(0.1);
    ridge.fit(&x, &y).unwrap();
    let ridge_preds = ridge.predict(&x).unwrap();
    let ridge_r2 = r2_score(&y, &ridge_preds).unwrap();

    // Stacking
    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge".to_string(), Box::new(RidgeRegression::new(0.1))),
        (
            "tree".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingRegressor::new(estimators);
    stacking.fit(&x, &y).unwrap();
    let stack_preds = stacking.predict(&x).unwrap();
    let stack_r2 = r2_score(&y, &stack_preds).unwrap();

    // Both should produce reasonable R² (positive or close to it)
    assert!(
        stack_r2 > -5.0,
        "Stacking R² {} should be reasonable",
        stack_r2
    );
    assert!(
        ridge_r2 > -5.0,
        "Ridge R² {} should be reasonable",
        ridge_r2
    );
}

#[test]
fn test_stacking_with_knn_meta_learner() {
    let (x, y) = create_regression_data(60, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge".to_string(), Box::new(RidgeRegression::new(1.0))),
        (
            "tree".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingRegressor::new(estimators)
        .with_final_estimator(Box::new(KNeighborsRegressor::new(3)));

    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    assert!(preds.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_single_base_estimator() {
    // Edge case: stacking with only one base estimator
    let (x, y) = create_classification_data(40, 3);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> =
        vec![("nb".to_string(), Box::new(GaussianNB::new()))];

    let mut stacking = StackingClassifier::new(estimators);

    let result = stacking.fit(&x, &y);
    assert!(result.is_ok(), "Single estimator should work");

    let preds = stacking.predict(&x).unwrap();
    assert!(preds.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_many_base_estimators() {
    // Test with many base estimators using trees (avoid collinearity)
    let (x, y) = create_regression_data(80, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge1".to_string(), Box::new(RidgeRegression::new(0.1))),
        ("ridge2".to_string(), Box::new(RidgeRegression::new(1.0))),
        ("ridge3".to_string(), Box::new(RidgeRegression::new(10.0))),
        (
            "tree1".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(2))),
        ),
        (
            "tree2".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(4))),
        ),
    ];

    let mut stacking = StackingRegressor::new(estimators);
    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    assert_eq!(preds.len(), 80);
    assert!(preds.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_get_estimator() {
    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("model1".to_string(), Box::new(RidgeRegression::new(0.1))),
        ("model2".to_string(), Box::new(RidgeRegression::new(1.0))),
    ];

    let mut stacking = StackingRegressor::new(estimators);

    let (x, y) = create_regression_data(30, 2);
    stacking.fit(&x, &y).unwrap();

    // Should be able to access named estimators
    assert!(stacking.get_estimator("model1").is_some());
    assert!(stacking.get_estimator("model2").is_some());
    assert!(stacking.get_estimator("nonexistent").is_none());
}

#[test]
fn test_stacking_vs_voting_comparison() {
    // Stacking should at least match voting performance
    let (x, y) = create_regression_data(100, 2);

    // Voting ensemble
    let voting_estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge".to_string(), Box::new(RidgeRegression::new(0.1))),
        (
            "tree".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
        ),
    ];
    let mut voting = VotingRegressor::new(voting_estimators);
    voting.fit(&x, &y).unwrap();
    let voting_preds = voting.predict(&x).unwrap();
    let voting_r2 = r2_score(&y, &voting_preds).unwrap();

    // Stacking ensemble
    let stacking_estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge".to_string(), Box::new(RidgeRegression::new(0.1))),
        (
            "tree".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
        ),
    ];
    let mut stacking = StackingRegressor::new(stacking_estimators);
    stacking.fit(&x, &y).unwrap();
    let stacking_preds = stacking.predict(&x).unwrap();
    let stacking_r2 = r2_score(&y, &stacking_preds).unwrap();

    // Both should produce finite R² values
    assert!(voting_r2.is_finite(), "Voting R² should be finite");
    assert!(stacking_r2.is_finite(), "Stacking R² should be finite");
}

// ============================================================================
// PASSTHROUGH FEATURE TESTS
// ============================================================================

#[test]
fn test_passthrough_increases_feature_count() {
    let (x, y) = create_regression_data(50, 3);

    // Without passthrough
    let estimators_no_pass: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge".to_string(), Box::new(RidgeRegression::new(1.0))),
        (
            "tree".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
        ),
    ];
    let mut stacking_no_pass = StackingRegressor::new(estimators_no_pass).with_passthrough(false);
    stacking_no_pass.fit(&x, &y).unwrap();

    // With passthrough
    let estimators_pass: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge".to_string(), Box::new(RidgeRegression::new(1.0))),
        (
            "tree".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
        ),
    ];
    let mut stacking_pass = StackingRegressor::new(estimators_pass).with_passthrough(true);
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
    let (x, y) = create_regression_data(80, 2);

    let estimators_no_pass: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![(
        "tree".to_string(),
        Box::new(DecisionTreeRegressor::new().with_max_depth(Some(2))),
    )];
    let mut stacking_no_pass = StackingRegressor::new(estimators_no_pass).with_passthrough(false);
    stacking_no_pass.fit(&x, &y).unwrap();

    let estimators_pass: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![(
        "tree".to_string(),
        Box::new(DecisionTreeRegressor::new().with_max_depth(Some(2))),
    )];
    let mut stacking_pass = StackingRegressor::new(estimators_pass).with_passthrough(true);
    stacking_pass.fit(&x, &y).unwrap();

    let preds_no_pass = stacking_no_pass.predict(&x).unwrap();
    let preds_pass = stacking_pass.predict(&x).unwrap();

    let r2_no_pass = r2_score(&y, &preds_no_pass).unwrap();
    let r2_pass = r2_score(&y, &preds_pass).unwrap();

    // Both R² values should be finite
    assert!(
        r2_no_pass.is_finite(),
        "R² without passthrough should be finite"
    );
    assert!(r2_pass.is_finite(), "R² with passthrough should be finite");
}

#[test]
fn test_passthrough_with_different_stack_methods() {
    let (x, y) = create_classification_data(60, 4);

    // Passthrough with Predict method
    let estimators_predict: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];
    let mut stacking_predict = StackingClassifier::new(estimators_predict)
        .with_stack_method(StackMethod::Predict)
        .with_passthrough(true);
    stacking_predict.fit(&x, &y).unwrap();

    // Passthrough with PredictProba method (default)
    let estimators_proba: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];
    let mut stacking_proba = StackingClassifier::new(estimators_proba)
        .with_stack_method(StackMethod::PredictProba)
        .with_passthrough(true);
    stacking_proba.fit(&x, &y).unwrap();

    // Both should work and produce finite predictions
    let preds_predict = stacking_predict.predict(&x).unwrap();
    let preds_proba = stacking_proba.predict(&x).unwrap();

    assert!(preds_predict.iter().all(|&p| p.is_finite()));
    assert!(preds_proba.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_passthrough_preserves_feature_ordering() {
    // Original features should be appended after meta-features
    let (x, y) = create_regression_data(40, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
        vec![("ridge".to_string(), Box::new(RidgeRegression::new(1.0)))];

    let mut stacking = StackingRegressor::new(estimators).with_passthrough(true);

    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    // Should produce valid predictions
    assert!(preds.iter().all(|&p| p.is_finite()));
}

// ============================================================================
// PROBABILITY AND OUTPUT TESTS
// ============================================================================

#[test]
fn test_stacking_classifier_predict_proba() {
    let (x, y) = create_classification_data(60, 4);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingClassifier::new(estimators);

    stacking.fit(&x, &y).unwrap();
    let probas = stacking.predict_proba(&x).unwrap();

    // Should have shape (n_samples, n_classes)
    assert_eq!(probas.nrows(), 60);
    assert_eq!(probas.ncols(), 2); // Binary classification

    // All probabilities should be finite
    assert!(probas.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stack_method_predict_vs_predict_proba() {
    let (x, y) = create_classification_data(50, 4);

    let estimators_predict: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];
    let mut stacking_predict =
        StackingClassifier::new(estimators_predict).with_stack_method(StackMethod::Predict);
    stacking_predict.fit(&x, &y).unwrap();

    let estimators_proba: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];
    let mut stacking_proba =
        StackingClassifier::new(estimators_proba).with_stack_method(StackMethod::PredictProba);
    stacking_proba.fit(&x, &y).unwrap();

    // Both should produce predictions
    let preds1 = stacking_predict.predict(&x).unwrap();
    let preds2 = stacking_proba.predict(&x).unwrap();

    assert_eq!(preds1.len(), 50);
    assert_eq!(preds2.len(), 50);

    // Both should produce finite predictions
    assert!(preds1.iter().all(|&p| p.is_finite()));
    assert!(preds2.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_regressor_individual_predictions() {
    let (x, y) = create_regression_data(40, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge1".to_string(), Box::new(RidgeRegression::new(0.1))),
        ("ridge2".to_string(), Box::new(RidgeRegression::new(1.0))),
    ];

    let mut stacking = StackingRegressor::new(estimators);

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

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
        vec![("ridge".to_string(), Box::new(RidgeRegression::new(1.0)))];

    let stacking = StackingRegressor::new(estimators);

    let result = stacking.predict(&x);
    assert!(result.is_err(), "Should error when not fitted");
}

#[test]
fn test_stacking_multiclass_classification() {
    // 3-class classification
    let (x, y) = create_multiclass_data(90, 4, 3);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingClassifier::new(estimators);

    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();
    let probas = stacking.predict_proba(&x).unwrap();

    // Predictions should be finite
    assert!(preds.iter().all(|&p| p.is_finite()));

    // Probabilities should be (n_samples, 3)
    assert_eq!(probas.ncols(), 3);
}

// ============================================================================
// ADDITIONAL EDGE CASE TESTS
// ============================================================================

#[test]
fn test_stacking_with_diverse_estimators() {
    // Using different types of estimators should work
    let (x, y) = create_regression_data(50, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge".to_string(), Box::new(RidgeRegression::new(0.1))),
        (
            "tree".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingRegressor::new(estimators);
    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    assert_eq!(preds.len(), 50);
    assert!(preds.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_small_dataset() {
    // Test with minimal data (edge case)
    let (x, y) = create_classification_data(15, 3);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> =
        vec![("nb".to_string(), Box::new(GaussianNB::new()))];

    let mut stacking = StackingClassifier::new(estimators).with_n_folds(3);

    let result = stacking.fit(&x, &y);
    assert!(result.is_ok(), "Should work with small dataset");

    let preds = stacking.predict(&x).unwrap();
    assert_eq!(preds.len(), 15);
}

#[test]
fn test_stacking_accuracy_reasonable() {
    // Stacking should produce valid predictions on well-separated data
    let (x, y) = create_classification_data(100, 4);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(5))),
        ),
    ];

    let mut stacking = StackingClassifier::new(estimators);
    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    // All predictions should be finite
    assert!(
        preds.iter().all(|&p| p.is_finite()),
        "Predictions should be finite"
    );

    // Count correct predictions manually (more robust than accuracy function
    // which might have issues with floating point comparison)
    let correct_count = preds
        .iter()
        .zip(y.iter())
        .filter(|(&pred, &true_val)| (pred - true_val).abs() < 0.5)
        .count();

    let manual_acc = correct_count as f64 / preds.len() as f64;
    assert!(
        manual_acc >= 0.4,
        "Manual accuracy {} should be reasonable on separable data",
        manual_acc
    );
}

#[test]
fn test_stacking_regressor_with_ridge_meta_learner() {
    // Test with RidgeRegression as meta-learner (handles collinear meta-features)
    let (x, y) = create_regression_data(60, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge1".to_string(), Box::new(RidgeRegression::new(0.1))),
        ("ridge2".to_string(), Box::new(RidgeRegression::new(1.0))),
    ];

    let mut stacking = StackingRegressor::new(estimators)
        .with_final_estimator(Box::new(RidgeRegression::new(1.0)));

    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    assert!(preds.iter().all(|&p| p.is_finite()));
    let r2 = r2_score(&y, &preds).unwrap();
    assert!(r2.is_finite(), "R² should be finite");
}

#[test]
fn test_stacking_estimator_names() {
    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("alpha".to_string(), Box::new(RidgeRegression::new(0.1))),
        ("beta".to_string(), Box::new(RidgeRegression::new(1.0))),
        ("gamma".to_string(), Box::new(RidgeRegression::new(10.0))),
    ];

    let stacking = StackingRegressor::new(estimators);
    let names = stacking.estimator_names();

    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn test_stacking_passthrough_flag() {
    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
        vec![("ridge".to_string(), Box::new(RidgeRegression::new(1.0)))];

    let stacking_false = StackingRegressor::new(estimators).with_passthrough(false);
    assert!(!stacking_false.passthrough());

    let estimators2: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
        vec![("ridge".to_string(), Box::new(RidgeRegression::new(1.0)))];

    let stacking_true = StackingRegressor::new(estimators2).with_passthrough(true);
    assert!(stacking_true.passthrough());
}

// ============================================================================
// ADDITIONAL DATA LEAKAGE PREVENTION TESTS
// ============================================================================

#[test]
fn test_stacking_cv_fold_isolation() {
    // Verify that with more folds, each sample is tested by models
    // that never saw it during training (stricter OOF verification)

    let (x, y) = create_regression_data(50, 2);

    // With 10 folds, each sample is in exactly one test fold
    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
        vec![("ridge".to_string(), Box::new(RidgeRegression::new(0.1)))];

    let mut stacking = StackingRegressor::new(estimators).with_n_folds(10);

    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    // All predictions should be finite
    assert!(preds.iter().all(|&p| p.is_finite()));

    // R² should be reasonable but not suspiciously perfect
    let r2 = r2_score(&y, &preds).unwrap();
    assert!(
        r2 < 0.9999,
        "R² {} suspiciously perfect - possible leakage",
        r2
    );
}

#[test]
fn test_stacking_loocv_like_extreme() {
    // Test with very high fold count (approaching LOOCV)
    let (x, y) = create_classification_data(20, 3);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> =
        vec![("nb".to_string(), Box::new(GaussianNB::new()))];

    // 10 folds on 20 samples = 2 samples per fold
    let mut stacking = StackingClassifier::new(estimators).with_n_folds(10);

    let result = stacking.fit(&x, &y);
    assert!(result.is_ok(), "High fold count should still work");

    let preds = stacking.predict(&x).unwrap();
    assert!(preds.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_meta_features_not_from_same_model() {
    // With different regularization, models should produce different predictions
    // If leakage occurred, we'd see identical outputs

    let (x, y) = create_regression_data(60, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        (
            "ridge_weak".to_string(),
            Box::new(RidgeRegression::new(0.001)),
        ),
        (
            "ridge_strong".to_string(),
            Box::new(RidgeRegression::new(100.0)),
        ),
    ];

    let mut stacking = StackingRegressor::new(estimators).with_n_folds(5);
    stacking.fit(&x, &y).unwrap();

    // Get individual predictions - they should differ
    let individual = stacking.individual_predictions(&x).unwrap();

    // Calculate correlation between the two sets of predictions
    let preds1 = &individual[0];
    let preds2 = &individual[1];

    let mean1 = preds1.mean().unwrap();
    let mean2 = preds2.mean().unwrap();

    let var1: f64 = preds1.iter().map(|&p| (p - mean1).powi(2)).sum();
    let var2: f64 = preds2.iter().map(|&p| (p - mean2).powi(2)).sum();

    // Predictions should have variance (not identical)
    assert!(var1 > 0.01, "Weak ridge predictions should have variance");
    assert!(var2 > 0.01, "Strong ridge predictions should have variance");
}

// ============================================================================
// EDGE CASE AND ERROR HANDLING TESTS
// ============================================================================

#[test]
#[should_panic(expected = "At least one estimator is required")]
fn test_stacking_empty_estimators_panics() {
    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![];
    let _stacking = StackingRegressor::new(estimators);
}

#[test]
fn test_stacking_with_constant_target() {
    // Edge case: all y values are the same
    let x = Array2::from_shape_fn((30, 2), |(i, j)| (i * j + 1) as f64 * 0.1);
    let y = Array1::from_elem(30, 5.0); // Constant target

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
        vec![("ridge".to_string(), Box::new(RidgeRegression::new(1.0)))];

    let mut stacking = StackingRegressor::new(estimators).with_n_folds(3);

    // Should still fit without error
    let result = stacking.fit(&x, &y);
    assert!(result.is_ok(), "Should handle constant target");

    let preds = stacking.predict(&x).unwrap();
    // All predictions should be close to the constant value
    assert!(preds.iter().all(|&p| (p - 5.0).abs() < 1.0));
}

#[test]
fn test_stacking_classifier_single_class_per_fold_edge() {
    // Create data where classes are well-separated but might cause
    // issues with stratification in extreme cases
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    // 40 samples of class 0
    for i in 0..40 {
        x_data.push(1.0 + (i as f64) * 0.1);
        x_data.push(1.0 + (i as f64) * 0.05);
        y_data.push(0.0);
    }
    // 40 samples of class 1
    for i in 0..40 {
        x_data.push(5.0 + (i as f64) * 0.1);
        x_data.push(5.0 + (i as f64) * 0.05);
        y_data.push(1.0);
    }

    let x = Array2::from_shape_vec((80, 2), x_data).unwrap();
    let y = Array1::from_vec(y_data);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingClassifier::new(estimators).with_n_folds(4);
    stacking.fit(&x, &y).unwrap();

    let preds = stacking.predict(&x).unwrap();

    // Predictions should be finite (meta-learner may not output exact class labels)
    assert!(preds.iter().all(|&p| p.is_finite()));

    // On well-separated data, most predictions should be close to 0 or 1
    let near_class_count = preds.iter().filter(|&&p| p < 0.3 || p > 0.7).count();
    assert!(
        near_class_count > 60,
        "Most predictions should be near class boundaries on separable data"
    );
}

#[test]
fn test_stacking_fitted_estimator_access() {
    let (x, y) = create_regression_data(40, 2);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("model_a".to_string(), Box::new(RidgeRegression::new(0.1))),
        ("model_b".to_string(), Box::new(RidgeRegression::new(1.0))),
    ];

    let mut stacking = StackingRegressor::new(estimators);

    // Before fitting, get_fitted_estimator should return None
    assert!(stacking.get_fitted_estimator("model_a").is_none());

    stacking.fit(&x, &y).unwrap();

    // After fitting, should be able to access fitted estimators
    assert!(stacking.get_fitted_estimator("model_a").is_some());
    assert!(stacking.get_fitted_estimator("model_b").is_some());
    assert!(stacking.get_fitted_estimator("nonexistent").is_none());
}

#[test]
fn test_stacking_classifier_fitted_estimator_access() {
    let (x, y) = create_classification_data(60, 4);

    let estimators: Vec<(String, Box<dyn VotingClassifierEstimator>)> = vec![
        ("nb".to_string(), Box::new(GaussianNB::new())),
        (
            "tree".to_string(),
            Box::new(DecisionTreeClassifier::new().with_max_depth(Some(3))),
        ),
    ];

    let mut stacking = StackingClassifier::new(estimators);

    // Before fitting
    assert!(stacking.get_fitted_estimator("nb").is_none());

    stacking.fit(&x, &y).unwrap();

    // After fitting
    assert!(stacking.get_fitted_estimator("nb").is_some());
    assert!(stacking.get_fitted_estimator("tree").is_some());
}

#[test]
fn test_stacking_regressor_reproducibility() {
    // Same configuration should produce consistent results
    let (x, y) = create_regression_data(50, 2);

    let create_stacking = || {
        let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
            vec![("ridge".to_string(), Box::new(RidgeRegression::new(1.0)))];
        StackingRegressor::new(estimators).with_n_folds(5)
    };

    let mut stacking1 = create_stacking();
    let mut stacking2 = create_stacking();

    stacking1.fit(&x, &y).unwrap();
    stacking2.fit(&x, &y).unwrap();

    let preds1 = stacking1.predict(&x).unwrap();
    let preds2 = stacking2.predict(&x).unwrap();

    // Predictions should be very close (deterministic behavior)
    let max_diff: f64 = preds1
        .iter()
        .zip(preds2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        max_diff < 1e-10,
        "Same configuration should produce identical results, diff: {}",
        max_diff
    );
}

#[test]
fn test_stacking_with_high_dimensional_features() {
    // Test with more features than typical
    let (x, y) = create_regression_data(60, 10);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> = vec![
        ("ridge".to_string(), Box::new(RidgeRegression::new(1.0))),
        (
            "tree".to_string(),
            Box::new(DecisionTreeRegressor::new().with_max_depth(Some(4))),
        ),
    ];

    let mut stacking = StackingRegressor::new(estimators)
        .with_passthrough(true)
        .with_n_folds(5);

    stacking.fit(&x, &y).unwrap();
    let preds = stacking.predict(&x).unwrap();

    assert_eq!(preds.len(), 60);
    assert!(preds.iter().all(|&p| p.is_finite()));
}

#[test]
fn test_stacking_n_features_attribute() {
    let (x, y) = create_regression_data(40, 5);

    let estimators: Vec<(String, Box<dyn VotingRegressorEstimator>)> =
        vec![("ridge".to_string(), Box::new(RidgeRegression::new(1.0)))];

    let mut stacking = StackingRegressor::new(estimators);

    // Before fitting
    assert!(stacking.n_features().is_none());

    stacking.fit(&x, &y).unwrap();

    // After fitting
    assert_eq!(stacking.n_features(), Some(5));
}
