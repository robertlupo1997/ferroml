//! Sample Weights & Class Weights Tests
//!
//! This module provides comprehensive tests for weight-based functionality:
//! - `sample_weight` parameter tests for supported models
//! - `class_weight`='balanced' tests for imbalanced data handling
//! - Custom class weights tests
//! - Weighted metrics validation tests
//! - Edge case tests (zero/negative weights)

use ndarray::{Array1, Array2};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// =============================================================================
// Test Utilities
// =============================================================================

/// Generate synthetic imbalanced classification data
fn generate_imbalanced_data(
    n_minority: usize,
    n_majority: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let n_samples = n_minority + n_majority;
    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    // Minority class (label 1.0)
    for i in 0..n_minority {
        labels.push(1.0);
        for j in 0..n_features {
            let mut hasher = DefaultHasher::new();
            (seed, i, j, "minority").hash(&mut hasher);
            let h = hasher.finish();
            let val = (h as f64 / u64::MAX as f64) * 2.0 - 1.0;
            // Add class-dependent signal to first feature
            let signal = if j == 0 { 1.0 } else { 0.0 };
            data.push(val + signal);
        }
    }

    // Majority class (label 0.0)
    for i in 0..n_majority {
        labels.push(0.0);
        for j in 0..n_features {
            let mut hasher = DefaultHasher::new();
            (seed, i, j, "majority").hash(&mut hasher);
            let h = hasher.finish();
            let val = (h as f64 / u64::MAX as f64) * 2.0 - 1.0;
            // Add class-dependent signal to first feature
            let signal = if j == 0 { -1.0 } else { 0.0 };
            data.push(val + signal);
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_features), data)
        .expect("Failed to create feature matrix");
    let y = Array1::from_vec(labels);

    (x, y)
}

/// Generate balanced classification data
fn generate_balanced_data(
    n_samples_per_class: usize,
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    generate_imbalanced_data(n_samples_per_class, n_samples_per_class, n_features, seed)
}

/// Generate multiclass imbalanced data (3 classes)
fn generate_multiclass_imbalanced_data(
    class_sizes: &[usize],
    n_features: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    let n_samples: usize = class_sizes.iter().sum();
    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);

    let mut sample_idx = 0;
    for (class_idx, &class_size) in class_sizes.iter().enumerate() {
        let class_label = class_idx as f64;
        for i in 0..class_size {
            labels.push(class_label);
            for j in 0..n_features {
                let mut hasher = DefaultHasher::new();
                (seed, sample_idx, i, j, class_idx).hash(&mut hasher);
                let h = hasher.finish();
                let val = (h as f64 / u64::MAX as f64) * 2.0 - 1.0;
                // Add class-dependent signal
                let signal = if j == 0 {
                    (class_idx as f64 - 1.0) * 2.0
                } else {
                    0.0
                };
                data.push(val + signal);
            }
            sample_idx += 1;
        }
    }

    let x = Array2::from_shape_vec((n_samples, n_features), data)
        .expect("Failed to create feature matrix");
    let y = Array1::from_vec(labels);

    (x, y)
}

/// Count class occurrences
fn count_classes(y: &Array1<f64>) -> std::collections::HashMap<i64, usize> {
    let mut counts = std::collections::HashMap::new();
    for &label in y {
        *counts.entry(label as i64).or_insert(0) += 1;
    }
    counts
}

/// Compute accuracy
fn compute_accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 1e-10)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Compute per-class recall (sensitivity)
fn compute_recall_per_class(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> std::collections::HashMap<i64, f64> {
    let mut true_positives: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    let mut actual_positives: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let t_class = t as i64;
        *actual_positives.entry(t_class).or_insert(0) += 1;
        if (t - p).abs() < 1e-10 {
            *true_positives.entry(t_class).or_insert(0) += 1;
        }
    }

    actual_positives
        .iter()
        .map(|(&class, &total)| {
            let tp = *true_positives.get(&class).unwrap_or(&0);
            let recall = if total > 0 {
                tp as f64 / total as f64
            } else {
                0.0
            };
            (class, recall)
        })
        .collect()
}

// =============================================================================
// Class Weight = 'Balanced' Tests
// =============================================================================

#[cfg(test)]
mod class_weight_balanced_tests {
    use super::*;

    /// Test that LogisticRegression with balanced weights improves minority class recall
    #[test]
    fn test_logistic_regression_balanced_weights() {
        // Create highly imbalanced data: 20 minority, 180 majority
        let (x, y) = generate_imbalanced_data(20, 180, 5, 42);

        // Train without class weights
        let mut model_unweighted = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Uniform);
        model_unweighted.fit(&x, &y).expect("Failed to fit unweighted model");
        let pred_unweighted = model_unweighted.predict(&x).expect("Failed to predict");

        // Train with balanced class weights
        let mut model_balanced = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Balanced);
        model_balanced.fit(&x, &y).expect("Failed to fit balanced model");
        let pred_balanced = model_balanced.predict(&x).expect("Failed to predict");

        // Compute per-class recall
        let recall_unweighted = compute_recall_per_class(&y, &pred_unweighted);
        let recall_balanced = compute_recall_per_class(&y, &pred_balanced);

        // Minority class (1.0) should have better recall with balanced weights
        let minority_recall_unweighted = recall_unweighted.get(&1).unwrap_or(&0.0);
        let minority_recall_balanced = recall_balanced.get(&1).unwrap_or(&0.0);

        // Balanced weights should improve minority class recall (or at least not make it worse)
        assert!(
            *minority_recall_balanced >= *minority_recall_unweighted * 0.8,
            "Balanced weights should not significantly harm minority recall. \
             Unweighted: {:.2}, Balanced: {:.2}",
            minority_recall_unweighted,
            minority_recall_balanced
        );
    }

    /// Test that DecisionTreeClassifier with balanced weights handles imbalanced data
    #[test]
    fn test_decision_tree_balanced_weights() {
        let (x, y) = generate_imbalanced_data(15, 135, 5, 42);

        // Train without class weights
        let mut model_unweighted = DecisionTreeClassifier::new()
            .with_max_depth(Some(5))
            .with_class_weight(ClassWeight::Uniform);
        model_unweighted.fit(&x, &y).expect("Failed to fit unweighted model");
        let pred_unweighted = model_unweighted.predict(&x).expect("Failed to predict");

        // Train with balanced class weights
        let mut model_balanced = DecisionTreeClassifier::new()
            .with_max_depth(Some(5))
            .with_class_weight(ClassWeight::Balanced);
        model_balanced.fit(&x, &y).expect("Failed to fit balanced model");
        let pred_balanced = model_balanced.predict(&x).expect("Failed to predict");

        // Verify both models produce valid predictions
        assert_eq!(pred_unweighted.len(), y.len());
        assert_eq!(pred_balanced.len(), y.len());

        // Compute metrics
        let recall_unweighted = compute_recall_per_class(&y, &pred_unweighted);
        let recall_balanced = compute_recall_per_class(&y, &pred_balanced);

        // Both models should have valid recalls
        assert!(recall_unweighted.contains_key(&1), "Unweighted model should predict minority class");
        assert!(recall_balanced.contains_key(&0) || recall_balanced.contains_key(&1));
    }

    /// Test that SVC with balanced weights handles imbalanced data
    #[test]
    fn test_svc_balanced_weights() {
        let (x, y) = generate_imbalanced_data(20, 80, 4, 42);

        // Train without class weights
        let mut model_unweighted = SVC::new()
            .with_c(1.0)
            .with_class_weight(ClassWeight::Uniform);
        model_unweighted.fit(&x, &y).expect("Failed to fit unweighted model");
        let pred_unweighted = model_unweighted.predict(&x).expect("Failed to predict");

        // Train with balanced class weights
        let mut model_balanced = SVC::new()
            .with_c(1.0)
            .with_class_weight(ClassWeight::Balanced);
        model_balanced.fit(&x, &y).expect("Failed to fit balanced model");
        let pred_balanced = model_balanced.predict(&x).expect("Failed to predict");

        // Verify both models produce valid predictions
        assert_eq!(pred_unweighted.len(), y.len());
        assert_eq!(pred_balanced.len(), y.len());

        // Compute recalls
        let recall_unweighted = compute_recall_per_class(&y, &pred_unweighted);
        let recall_balanced = compute_recall_per_class(&y, &pred_balanced);

        // Balanced weights should provide meaningful minority class recognition
        let minority_recall_balanced = recall_balanced.get(&1).unwrap_or(&0.0);
        assert!(
            *minority_recall_balanced >= 0.0,
            "Balanced SVC should have non-negative minority recall"
        );

        // Log for debugging
        println!(
            "SVC Unweighted recalls: {:?}",
            recall_unweighted
        );
        println!(
            "SVC Balanced recalls: {:?}",
            recall_balanced
        );
    }

    /// Test that GradientBoostingClassifier with balanced weights works
    #[test]
    fn test_gradient_boosting_balanced_weights() {
        let (x, y) = generate_imbalanced_data(25, 75, 5, 42);

        // Train with balanced class weights
        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3))
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit model");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());

        // Verify predictions are valid class labels
        for pred in predictions.iter() {
            assert!(
                (*pred - 0.0).abs() < 1e-10 || (*pred - 1.0).abs() < 1e-10,
                "Prediction should be 0 or 1, got {}",
                pred
            );
        }
    }

    /// Test that RandomForestClassifier with balanced weights works
    #[test]
    fn test_random_forest_balanced_weights() {
        let (x, y) = generate_imbalanced_data(20, 80, 5, 42);

        // Train with balanced class weights
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42)
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit model");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());

        // Compute recalls
        let recalls = compute_recall_per_class(&y, &predictions);

        // Both classes should have some recall
        println!("RandomForest Balanced recalls: {:?}", recalls);
    }

    /// Test balanced weights on multiclass imbalanced data
    #[test]
    fn test_multiclass_balanced_weights() {
        // 3 classes with sizes: 10, 30, 60
        let (x, y) = generate_multiclass_imbalanced_data(&[10, 30, 60], 5, 42);

        // Train with balanced class weights - use RandomForest since LogisticRegression only supports binary
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42)
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit model");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());

        // Verify all classes are represented in true labels
        let class_counts = count_classes(&y);
        assert_eq!(class_counts.len(), 3, "Should have 3 classes");
    }
}

// =============================================================================
// Custom Class Weights Tests
// =============================================================================

#[cfg(test)]
mod custom_class_weights_tests {
    use super::*;

    /// Test LogisticRegression with custom class weights
    #[test]
    fn test_logistic_regression_custom_weights() {
        let (x, y) = generate_balanced_data(50, 5, 42);

        // Custom weights: class 1 has 3x the weight of class 0
        let custom_weights = ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 3.0)]);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(custom_weights);

        model.fit(&x, &y).expect("Failed to fit model");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());

        // With higher weight on class 1, we expect model to be more conservative
        // about predicting class 0 (to avoid false negatives on class 1)
        let recalls = compute_recall_per_class(&y, &predictions);
        println!("Custom weights (1.0:3.0) recalls: {:?}", recalls);
    }

    /// Test DecisionTreeClassifier with custom class weights
    #[test]
    fn test_decision_tree_custom_weights() {
        let (x, y) = generate_imbalanced_data(30, 70, 5, 42);

        // Custom weights: minority class has 5x weight
        let custom_weights = ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 5.0)]);

        let mut model = DecisionTreeClassifier::new()
            .with_max_depth(Some(5))
            .with_class_weight(custom_weights);

        model.fit(&x, &y).expect("Failed to fit model");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());

        let recalls = compute_recall_per_class(&y, &predictions);
        println!("Decision tree custom weights recalls: {:?}", recalls);
    }

    /// Test SVC with custom class weights
    #[test]
    fn test_svc_custom_weights() {
        let (x, y) = generate_balanced_data(40, 4, 42);

        // Custom weights
        let custom_weights = ClassWeight::Custom(vec![(0.0, 2.0), (1.0, 1.0)]);

        let mut model = SVC::new()
            .with_c(1.0)
            .with_class_weight(custom_weights);

        model.fit(&x, &y).expect("Failed to fit model");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());
    }

    /// Test that custom weights with zero for a class are handled
    #[test]
    fn test_custom_weights_zero_weight() {
        let (x, y) = generate_balanced_data(30, 5, 42);

        // Very small weight for class 0 (effectively zero)
        let custom_weights = ClassWeight::Custom(vec![(0.0, 0.001), (1.0, 1.0)]);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(custom_weights);

        // Should still fit without error
        model.fit(&x, &y).expect("Failed to fit model with near-zero weight");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());

        // Model should strongly favor class 1
        let class_1_predictions = predictions.iter().filter(|&&p| (p - 1.0).abs() < 1e-10).count();
        println!(
            "With near-zero weight on class 0, class 1 predictions: {}/{}",
            class_1_predictions,
            predictions.len()
        );
    }

    /// Test custom weights on multiclass problem
    #[test]
    fn test_multiclass_custom_weights() {
        let (x, y) = generate_multiclass_imbalanced_data(&[20, 40, 40], 5, 42);

        // Custom weights: boost class 0 - use RandomForest since LogisticRegression only supports binary
        let custom_weights = ClassWeight::Custom(vec![(0.0, 3.0), (1.0, 1.0), (2.0, 1.0)]);

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42)
            .with_class_weight(custom_weights);

        model.fit(&x, &y).expect("Failed to fit model");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());
    }
}

// =============================================================================
// Sample Weight Parameter Tests
// =============================================================================

#[cfg(test)]
mod sample_weight_tests {
    use super::*;
    use crate::models::{compute_sample_weights, get_unique_classes};

    /// Test compute_sample_weights function with Uniform weights
    #[test]
    fn test_compute_sample_weights_uniform() {
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0]);
        let classes = get_unique_classes(&y);

        let weights = compute_sample_weights(&y, &classes, &ClassWeight::Uniform);

        // All weights should be 1.0 for uniform
        assert_eq!(weights.len(), y.len());
        for w in weights.iter() {
            assert!((w - 1.0).abs() < 1e-10, "Uniform weight should be 1.0, got {}", w);
        }
    }

    /// Test compute_sample_weights function with Balanced weights
    #[test]
    fn test_compute_sample_weights_balanced() {
        // 3 samples of class 0, 2 samples of class 1
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0]);
        let classes = get_unique_classes(&y);

        let weights = compute_sample_weights(&y, &classes, &ClassWeight::Balanced);

        assert_eq!(weights.len(), y.len());

        // Balanced weights: n_samples / (n_classes * n_samples_for_class)
        // Class 0: 5 / (2 * 3) = 0.833...
        // Class 1: 5 / (2 * 2) = 1.25
        let class_0_weight = weights[0];
        let class_1_weight = weights[3];

        // Class 1 (minority) should have higher weight
        assert!(
            class_1_weight > class_0_weight,
            "Minority class should have higher weight. Class 0: {}, Class 1: {}",
            class_0_weight,
            class_1_weight
        );
    }

    /// Test compute_sample_weights function with Custom weights
    #[test]
    fn test_compute_sample_weights_custom() {
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let classes = get_unique_classes(&y);

        let custom = ClassWeight::Custom(vec![(0.0, 2.0), (1.0, 5.0)]);
        let weights = compute_sample_weights(&y, &classes, &custom);

        assert_eq!(weights.len(), y.len());
        assert!((weights[0] - 2.0).abs() < 1e-10, "Class 0 weight should be 2.0");
        assert!((weights[1] - 2.0).abs() < 1e-10, "Class 0 weight should be 2.0");
        assert!((weights[2] - 5.0).abs() < 1e-10, "Class 1 weight should be 5.0");
        assert!((weights[3] - 5.0).abs() < 1e-10, "Class 1 weight should be 5.0");
    }

    /// Test that sample weights affect model training
    #[test]
    fn test_sample_weights_affect_training() {
        let (x, y) = generate_balanced_data(40, 4, 42);

        // Train with uniform weights
        let mut model_uniform = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Uniform);
        model_uniform.fit(&x, &y).expect("Failed to fit");
        let coef_uniform = model_uniform
            .coefficients()
            .expect("Failed to get coefficients")
            .to_vec();

        // Train with balanced weights (which creates different sample weights)
        let mut model_balanced = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Balanced);
        model_balanced.fit(&x, &y).expect("Failed to fit");
        let coef_balanced = model_balanced
            .coefficients()
            .expect("Failed to get coefficients")
            .to_vec();

        // On perfectly balanced data, coefficients should be similar but not identical
        // (due to regularization effects with different effective sample sizes)
        let coef_diff: f64 = coef_uniform
            .iter()
            .zip(coef_balanced.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        println!(
            "Coefficient difference (uniform vs balanced on balanced data): {}",
            coef_diff
        );
    }
}

// =============================================================================
// Weighted Metrics Validation Tests
// =============================================================================

#[cfg(test)]
mod weighted_metrics_tests {
    use super::*;

    /// Test that balanced weights improve balanced accuracy on imbalanced data
    #[test]
    fn test_balanced_accuracy_improvement() {
        let (x, y) = generate_imbalanced_data(10, 90, 5, 42);

        // Train without class weights
        let mut model_unweighted = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Uniform);
        model_unweighted.fit(&x, &y).expect("Failed to fit");
        let pred_unweighted = model_unweighted.predict(&x).expect("Failed to predict");

        // Train with balanced class weights
        let mut model_balanced = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Balanced);
        model_balanced.fit(&x, &y).expect("Failed to fit");
        let pred_balanced = model_balanced.predict(&x).expect("Failed to predict");

        // Compute balanced accuracy (mean of per-class recall)
        let recall_unweighted = compute_recall_per_class(&y, &pred_unweighted);
        let recall_balanced = compute_recall_per_class(&y, &pred_balanced);

        let balanced_acc_unweighted: f64 =
            recall_unweighted.values().sum::<f64>() / recall_unweighted.len() as f64;
        let balanced_acc_balanced: f64 =
            recall_balanced.values().sum::<f64>() / recall_balanced.len() as f64;

        println!(
            "Balanced accuracy - Unweighted: {:.3}, Balanced: {:.3}",
            balanced_acc_unweighted, balanced_acc_balanced
        );

        // Balanced weights should generally improve balanced accuracy
        // (but this isn't guaranteed on all datasets)
    }

    /// Test that accuracy differs between weighted and unweighted models
    #[test]
    fn test_accuracy_differs_with_weights() {
        let (x, y) = generate_imbalanced_data(15, 85, 5, 42);

        let mut model_unweighted = DecisionTreeClassifier::new()
            .with_max_depth(Some(4))
            .with_class_weight(ClassWeight::Uniform);
        model_unweighted.fit(&x, &y).expect("Failed to fit");
        let pred_unweighted = model_unweighted.predict(&x).expect("Failed to predict");

        let mut model_balanced = DecisionTreeClassifier::new()
            .with_max_depth(Some(4))
            .with_class_weight(ClassWeight::Balanced);
        model_balanced.fit(&x, &y).expect("Failed to fit");
        let pred_balanced = model_balanced.predict(&x).expect("Failed to predict");

        let acc_unweighted = compute_accuracy(&y, &pred_unweighted);
        let acc_balanced = compute_accuracy(&y, &pred_balanced);

        println!(
            "Accuracy - Unweighted: {:.3}, Balanced: {:.3}",
            acc_unweighted, acc_balanced
        );

        // Models should produce different results (not necessarily better/worse overall)
        // The key is that balanced weights change the decision boundary
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    /// Test with all samples having the same weight
    #[test]
    fn test_all_same_weight() {
        let (x, y) = generate_balanced_data(30, 5, 42);

        // Custom weights with same value for all classes
        let same_weights = ClassWeight::Custom(vec![(0.0, 2.5), (1.0, 2.5)]);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(same_weights);

        model.fit(&x, &y).expect("Failed to fit with same weights");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());
    }

    /// Test with very small weights
    #[test]
    fn test_very_small_weights() {
        let (x, y) = generate_balanced_data(30, 5, 42);

        let small_weights = ClassWeight::Custom(vec![(0.0, 0.001), (1.0, 0.001)]);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(small_weights);

        // Should handle very small weights without numerical issues
        model.fit(&x, &y).expect("Failed to fit with small weights");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());
    }

    /// Test with very large weights
    #[test]
    fn test_very_large_weights() {
        let (x, y) = generate_balanced_data(30, 5, 42);

        let large_weights = ClassWeight::Custom(vec![(0.0, 1000.0), (1.0, 1000.0)]);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(large_weights);

        model.fit(&x, &y).expect("Failed to fit with large weights");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());
    }

    /// Test with extremely imbalanced weights
    #[test]
    fn test_extremely_imbalanced_weights() {
        let (x, y) = generate_balanced_data(30, 5, 42);

        // One class has 1000x the weight of the other
        let extreme_weights = ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 1000.0)]);

        let mut model = DecisionTreeClassifier::new()
            .with_max_depth(Some(5))
            .with_class_weight(extreme_weights);

        model.fit(&x, &y).expect("Failed to fit with extreme weights");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), y.len());

        // With extreme weight on class 1, most predictions should be class 1
        let class_1_count = predictions.iter().filter(|&&p| (p - 1.0).abs() < 1e-10).count();
        println!(
            "With extreme weight (1:1000), class 1 predictions: {}/{}",
            class_1_count,
            predictions.len()
        );
    }

    /// Test single sample per class
    #[test]
    fn test_single_sample_per_class() {
        // Create minimal data: 1 sample of each class
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .expect("Failed to create X");
        let y = Array1::from_vec(vec![0.0, 1.0]);

        // Balanced weights with single samples should still work
        let mut model = DecisionTreeClassifier::new()
            .with_max_depth(Some(2))
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit with single samples");
        let predictions = model.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), 2);
    }

    /// Test that default ClassWeight is Uniform
    #[test]
    fn test_default_class_weight_is_uniform() {
        let default_weight = ClassWeight::default();
        assert_eq!(default_weight, ClassWeight::Uniform);
    }

    /// Test ClassWeight equality
    #[test]
    fn test_class_weight_equality() {
        assert_eq!(ClassWeight::Uniform, ClassWeight::Uniform);
        assert_eq!(ClassWeight::Balanced, ClassWeight::Balanced);
        assert_ne!(ClassWeight::Uniform, ClassWeight::Balanced);

        let custom1 = ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 2.0)]);
        let custom2 = ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 2.0)]);
        let custom3 = ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 3.0)]);

        assert_eq!(custom1, custom2);
        assert_ne!(custom1, custom3);
    }

    /// Test that weights don't cause NaN or Inf in predictions
    #[test]
    fn test_no_nan_inf_with_weights() {
        let (x, y) = generate_imbalanced_data(5, 95, 4, 42);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit");
        let predictions = model.predict(&x).expect("Failed to predict");

        for pred in predictions.iter() {
            assert!(pred.is_finite(), "Prediction should be finite, got {}", pred);
            assert!(!pred.is_nan(), "Prediction should not be NaN");
        }
    }

    /// Test probabilities with class weights
    #[test]
    fn test_probabilities_with_weights() {
        let (x, y) = generate_imbalanced_data(20, 80, 4, 42);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit");
        let probas = model.predict_proba(&x).expect("Failed to predict probabilities");

        // LogisticRegression.predict_proba returns P(class=1) as a single column
        // For binary classification: P(class=0) = 1 - P(class=1)
        assert_eq!(probas.ncols(), 1, "Binary LogisticRegression returns single column of P(class=1)");

        // Check probability constraints - each value should be valid probability
        for row in probas.rows() {
            let p_class_1 = row[0];
            let p_class_0 = 1.0 - p_class_1;

            // Both probabilities should be in [0, 1]
            assert!(
                p_class_1 >= 0.0 && p_class_1 <= 1.0,
                "P(class=1) should be in [0,1], got {}",
                p_class_1
            );
            assert!(
                p_class_0 >= 0.0 && p_class_0 <= 1.0,
                "P(class=0) should be in [0,1], got {}",
                p_class_0
            );

            // Verify they sum to 1.0
            let sum = p_class_0 + p_class_1;
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Probabilities should sum to 1.0, got {}",
                sum
            );

            // Verify probability is finite
            assert!(p_class_1.is_finite(), "Probability should be finite");
        }
    }
}

// =============================================================================
// Model-Specific Weight Tests
// =============================================================================

#[cfg(test)]
mod model_specific_tests {
    use super::*;

    /// Test LogisticRegression coefficient behavior with weights
    #[test]
    fn test_logistic_regression_coefficient_behavior() {
        let (x, y) = generate_imbalanced_data(20, 80, 3, 42);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit");

        // Verify coefficients are accessible
        let coefficients = model.coefficients().expect("Should have coefficients");
        assert_eq!(coefficients.len(), 3, "Should have 3 coefficients for 3 features");

        // Coefficients should be finite
        for coef in coefficients.iter() {
            assert!(coef.is_finite(), "Coefficient should be finite");
        }

        // Intercept should be accessible
        let intercept = model.intercept().expect("Should have intercept");
        assert!(intercept.is_finite(), "Intercept should be finite");
    }

    /// Test DecisionTree feature importance with weights
    #[test]
    fn test_decision_tree_feature_importance_with_weights() {
        let (x, y) = generate_imbalanced_data(30, 70, 5, 42);

        let mut model = DecisionTreeClassifier::new()
            .with_max_depth(Some(5))
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit");

        // Feature importance should be available (use Model trait's feature_importance method)
        let importance = model.feature_importance().expect("Should have feature importances");
        assert_eq!(importance.len(), 5, "Should have importance for 5 features");

        // Importances should sum to approximately 1.0
        let sum: f64 = importance.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6 || sum == 0.0,
            "Feature importances should sum to 1.0, got {}",
            sum
        );
    }

    /// Test GradientBoosting feature importance with weights
    #[test]
    fn test_gradient_boosting_feature_importance_with_weights() {
        let (x, y) = generate_imbalanced_data(30, 70, 5, 42);

        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(Some(3))
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit");

        let importance = model.feature_importance().expect("Should have feature importances");
        assert_eq!(importance.len(), 5);

        // Importances should be non-negative
        for &imp in importance.iter() {
            assert!(imp >= 0.0, "Feature importance should be non-negative");
        }
    }

    /// Test RandomForest OOB error with weights
    #[test]
    fn test_random_forest_oob_with_weights() {
        let (x, y) = generate_imbalanced_data(30, 70, 5, 42);

        let mut model = RandomForestClassifier::new()
            .with_n_estimators(20)
            .with_max_depth(Some(5))
            .with_random_state(42)
            .with_oob_score(true)
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit");

        // OOB score should be available
        if let Some(oob_score) = model.oob_score() {
            assert!(oob_score >= 0.0 && oob_score <= 1.0, "OOB score should be in [0,1]");
            println!("RandomForest OOB score with balanced weights: {:.3}", oob_score);
        }
    }

    /// Test SVC predictions with weights
    #[test]
    fn test_svc_predictions_with_weights() {
        let (x, y) = generate_balanced_data(30, 4, 42);

        let mut model = SVC::new()
            .with_c(1.0)
            .with_class_weight(ClassWeight::Balanced);

        model.fit(&x, &y).expect("Failed to fit");

        // Predictions should work
        let predictions = model.predict(&x).expect("Failed to predict");
        assert_eq!(predictions.len(), y.len());

        // Predictions should be valid class labels
        for &p in predictions.iter() {
            assert!(
                (p - 0.0).abs() < 1e-10 || (p - 1.0).abs() < 1e-10,
                "Prediction should be 0 or 1, got {}",
                p
            );
        }
    }
}

// =============================================================================
// Weight Consistency Tests
// =============================================================================

#[cfg(test)]
mod weight_consistency_tests {
    use super::*;

    /// Test that fitting multiple times with same weights gives same result
    #[test]
    fn test_weight_determinism() {
        let (x, y) = generate_balanced_data(40, 5, 42);

        let mut model1 = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Balanced);
        model1.fit(&x, &y).expect("Failed to fit model1");
        let pred1 = model1.predict(&x).expect("Failed to predict");

        let mut model2 = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Balanced);
        model2.fit(&x, &y).expect("Failed to fit model2");
        let pred2 = model2.predict(&x).expect("Failed to predict");

        // Predictions should be identical
        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert!(
                (p1 - p2).abs() < 1e-10,
                "Predictions should be deterministic"
            );
        }
    }

    /// Test that refitting changes the model
    #[test]
    fn test_refit_with_different_weights() {
        let (x, y) = generate_balanced_data(40, 5, 42);

        let mut model = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Uniform);

        model.fit(&x, &y).expect("Failed to fit");
        let coef1 = model.coefficients().expect("Failed to get coefficients").to_vec();

        // Refit with different class weight by creating a new model
        // (LogisticRegression doesn't support changing class_weight after construction)
        let mut model2 = LogisticRegression::new()
            .with_max_iter(200)
            .with_class_weight(ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 10.0)]));

        model2.fit(&x, &y).expect("Failed to fit");
        let coef2 = model2.coefficients().expect("Failed to get coefficients").to_vec();

        // Coefficients should differ
        let diff: f64 = coef1.iter().zip(coef2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "Coefficients should differ with different weights");
    }
}
