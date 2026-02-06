//! Incremental Learning Tests
//!
//! Phase 27 of FerroML testing plan - comprehensive tests for:
//! - partial_fit on Naive Bayes classifiers (GaussianNB, MultinomialNB, BernoulliNB)
//! - Equivalence between fit() and partial_fit() on full data
//! - Incremental updates accumulate correctly
//! - Class and feature validation across batches
//! - Numerical stability (Welford's algorithm)
//! - Streaming data simulation
//!
//! Note: WarmStartModel trait is defined but not yet implemented on ensemble models.
//! Warm start tests will be added when implementations are available.

#![allow(clippy::float_cmp)]

#[cfg(test)]
use ndarray::{Array1, Array2};

#[cfg(test)]
use crate::models::naive_bayes::{BernoulliNB, GaussianNB, MultinomialNB};
#[cfg(test)]
use crate::models::{Model, ProbabilisticModel};
#[cfg(test)]
use crate::testing::assertions::tolerances;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

#[cfg(test)]
/// Generate synthetic classification data with known class structure
fn make_classification_data(
    n_samples_per_class: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut state = hasher.finish();

    let mut next_rand = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (u32::MAX as f64)
    };

    let total_samples = n_samples_per_class * n_classes;
    let mut x_data = Vec::with_capacity(total_samples * n_features);
    let mut y_data = Vec::with_capacity(total_samples);

    for class in 0..n_classes {
        let class_center = (class + 1) as f64 * 3.0;
        for _ in 0..n_samples_per_class {
            for _ in 0..n_features {
                x_data.push(class_center + (next_rand() - 0.5) * 2.0);
            }
            y_data.push(class as f64);
        }
    }

    let x = Array2::from_shape_vec((total_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

#[cfg(test)]
/// Generate count data for MultinomialNB (non-negative integers)
fn make_count_data(
    n_samples_per_class: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut state = hasher.finish();

    let mut next_rand = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (u32::MAX as f64)
    };

    let total_samples = n_samples_per_class * n_classes;
    let mut x_data = Vec::with_capacity(total_samples * n_features);
    let mut y_data = Vec::with_capacity(total_samples);

    for class in 0..n_classes {
        let base_count = (class + 1) as f64 * 5.0;
        for _ in 0..n_samples_per_class {
            for feat in 0..n_features {
                // Different features have different distributions per class
                let count = if feat % n_classes == class {
                    (base_count + next_rand() * 10.0).floor()
                } else {
                    (next_rand() * 3.0).floor()
                };
                x_data.push(count);
            }
            y_data.push(class as f64);
        }
    }

    let x = Array2::from_shape_vec((total_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

#[cfg(test)]
/// Generate binary data for BernoulliNB
fn make_binary_data(
    n_samples_per_class: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Array2<f64>, Array1<f64>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut state = hasher.finish();

    let mut next_rand = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f64) / (u32::MAX as f64)
    };

    let total_samples = n_samples_per_class * n_classes;
    let mut x_data = Vec::with_capacity(total_samples * n_features);
    let mut y_data = Vec::with_capacity(total_samples);

    for class in 0..n_classes {
        for _ in 0..n_samples_per_class {
            for feat in 0..n_features {
                // Higher probability of 1 for features matching class
                let prob = if feat % n_classes == class { 0.8 } else { 0.2 };
                x_data.push(if next_rand() < prob { 1.0 } else { 0.0 });
            }
            y_data.push(class as f64);
        }
    }

    let x = Array2::from_shape_vec((total_samples, n_features), x_data).unwrap();
    let y = Array1::from_vec(y_data);
    (x, y)
}

#[cfg(test)]
/// Assert arrays are approximately equal within tolerance
fn assert_arrays_close(a: &Array1<f64>, b: &Array1<f64>, tol: f64, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (ai - bi).abs() < tol,
            "{}: index {} differs: {} vs {} (diff={})",
            msg,
            i,
            ai,
            bi,
            (ai - bi).abs()
        );
    }
}

#[cfg(test)]
fn assert_arrays2_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64, msg: &str) {
    assert_eq!(a.shape(), b.shape(), "{}: shape mismatch", msg);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let ai = a[[i, j]];
            let bi = b[[i, j]];
            assert!(
                (ai - bi).abs() < tol,
                "{}: index [{},{}] differs: {} vs {} (diff={})",
                msg,
                i,
                j,
                ai,
                bi,
                (ai - bi).abs()
            );
        }
    }
}

// ============================================================================
// GAUSSIAN NB PARTIAL FIT TESTS
// ============================================================================

#[cfg(test)]
mod gaussian_nb_tests {
    use super::*;

    #[test]
    fn test_partial_fit_requires_classes_on_first_call() {
        let (x, y) = make_classification_data(10, 3, 2, 42);
        let mut model = GaussianNB::new();

        // Should fail without classes on first call
        let result = model.partial_fit(&x, &y, None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Classes must be provided"));
    }

    #[test]
    fn test_partial_fit_classes_optional_after_first() {
        let (x, y) = make_classification_data(10, 3, 2, 42);
        let mut model = GaussianNB::new();

        // First call with classes
        let x1 = x.slice(ndarray::s![0..5, ..]).to_owned();
        let y1 = y.slice(ndarray::s![0..5]).to_owned();
        model
            .partial_fit(&x1, &y1, Some(vec![0.0, 1.0]))
            .expect("First partial_fit should succeed");

        // Second call without classes should work
        let x2 = x.slice(ndarray::s![5..10, ..]).to_owned();
        let y2 = y.slice(ndarray::s![5..10]).to_owned();
        model
            .partial_fit(&x2, &y2, None)
            .expect("Second partial_fit should succeed without classes");
    }

    #[test]
    fn test_partial_fit_equivalence_to_fit() {
        let (x, y) = make_classification_data(50, 4, 3, 123);

        // Model trained with fit()
        let mut fit_model = GaussianNB::new();
        fit_model.fit(&x, &y).unwrap();

        // Model trained with single partial_fit()
        let mut partial_model = GaussianNB::new();
        partial_model
            .partial_fit(&x, &y, Some(vec![0.0, 1.0, 2.0]))
            .unwrap();

        // Should have same parameters
        assert_arrays2_close(
            fit_model.theta().unwrap(),
            partial_model.theta().unwrap(),
            tolerances::ITERATIVE,
            "theta mismatch",
        );
        assert_arrays2_close(
            fit_model.var().unwrap(),
            partial_model.var().unwrap(),
            tolerances::ITERATIVE,
            "var mismatch",
        );
        assert_arrays_close(
            fit_model.class_prior().unwrap(),
            partial_model.class_prior().unwrap(),
            tolerances::ITERATIVE,
            "class_prior mismatch",
        );
    }

    #[test]
    fn test_partial_fit_incremental_updates() {
        // 30 samples per class * 2 classes = 60 total samples
        let (x, y) = make_classification_data(30, 3, 2, 456);
        let total_samples = x.nrows(); // 60

        // Split into 3 batches of 20
        let batch_size = total_samples / 3;
        let mut incremental_model = GaussianNB::new();

        for i in 0..3 {
            let start = i * batch_size;
            let end = start + batch_size;
            let x_batch = x.slice(ndarray::s![start..end, ..]).to_owned();
            let y_batch = y.slice(ndarray::s![start..end]).to_owned();

            let classes = if i == 0 { Some(vec![0.0, 1.0]) } else { None };
            incremental_model
                .partial_fit(&x_batch, &y_batch, classes)
                .unwrap();
        }

        // Compare to fit on full data
        let mut full_model = GaussianNB::new();
        full_model.fit(&x, &y).unwrap();

        // Should have approximately same parameters
        assert_arrays2_close(
            full_model.theta().unwrap(),
            incremental_model.theta().unwrap(),
            tolerances::ITERATIVE,
            "incremental theta mismatch",
        );
        assert_arrays2_close(
            full_model.var().unwrap(),
            incremental_model.var().unwrap(),
            tolerances::ITERATIVE,
            "incremental var mismatch",
        );
    }

    #[test]
    fn test_partial_fit_class_count_accumulates() {
        let (x, y) = make_classification_data(30, 2, 2, 789);

        let mut model = GaussianNB::new();

        // First batch: 15 samples per class = 30 total
        model.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();
        let counts_after_1 = model.class_count().unwrap().clone();

        // Second batch: same data again
        model.partial_fit(&x, &y, None).unwrap();
        let counts_after_2 = model.class_count().unwrap();

        // Counts should double
        for (c1, c2) in counts_after_1.iter().zip(counts_after_2.iter()) {
            assert!(
                (*c2 - 2.0 * c1).abs() < tolerances::CLOSED_FORM,
                "Class counts should double: {} vs {}",
                c1,
                c2
            );
        }
    }

    #[test]
    fn test_partial_fit_feature_mismatch_error() {
        let mut model = GaussianNB::new();

        // First batch with 3 features
        let x1 = Array2::from_shape_vec((5, 3), vec![1.0; 15]).unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0]);
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        // Second batch with 4 features should fail
        let x2 = Array2::from_shape_vec((5, 4), vec![1.0; 20]).unwrap();
        let y2 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0]);
        let result = model.partial_fit(&x2, &y2, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_fit_empty_class_in_batch() {
        let mut model = GaussianNB::new();

        // First batch has both classes
        let x1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 5.0, 6.0, 6.0, 7.0]).unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        let counts_1 = model.class_count().unwrap().clone();

        // Second batch has only class 0
        let x2 = Array2::from_shape_vec((2, 2), vec![1.5, 2.5, 2.5, 3.5]).unwrap();
        let y2 = Array1::from_vec(vec![0.0, 0.0]);
        model.partial_fit(&x2, &y2, None).unwrap();

        let counts_2 = model.class_count().unwrap();

        // Class 0 count should increase, class 1 unchanged
        assert!(counts_2[0] > counts_1[0], "Class 0 count should increase");
        assert!(
            (counts_2[1] - counts_1[1]).abs() < 1e-10,
            "Class 1 count should be unchanged"
        );
    }

    #[test]
    fn test_partial_fit_single_sample_batches() {
        let (x, y) = make_classification_data(20, 2, 2, 111);

        let mut single_batch_model = GaussianNB::new();

        // Train one sample at a time
        for i in 0..x.nrows() {
            let x_i = x.slice(ndarray::s![i..i + 1, ..]).to_owned();
            let y_i = y.slice(ndarray::s![i..i + 1]).to_owned();

            let classes = if i == 0 { Some(vec![0.0, 1.0]) } else { None };
            single_batch_model.partial_fit(&x_i, &y_i, classes).unwrap();
        }

        // Should be fitted and make predictions
        assert!(single_batch_model.is_fitted());
        let preds = single_batch_model.predict(&x).unwrap();
        assert_eq!(preds.len(), x.nrows());
    }

    #[test]
    fn test_partial_fit_predictions_valid() {
        let (x, y) = make_classification_data(40, 3, 2, 222);

        let mut model = GaussianNB::new();
        model.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();

        // Predictions should be valid class labels
        let preds = model.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(
                p == 0.0 || p == 1.0,
                "Prediction should be 0 or 1, got {}",
                p
            );
        }

        // Probabilities should sum to 1
        let probas = model.predict_proba(&x).unwrap();
        for i in 0..probas.nrows() {
            let row_sum: f64 = probas.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < tolerances::ITERATIVE,
                "Probabilities should sum to 1, got {}",
                row_sum
            );
        }
    }

    #[test]
    fn test_partial_fit_welford_numerical_stability() {
        // Test that Welford's algorithm handles large value differences
        let mut model = GaussianNB::new();

        // First batch with small values
        let x1 = Array2::from_shape_vec((10, 2), vec![1.0; 20]).unwrap();
        let y1 = Array1::from_vec(vec![0.0; 10]);
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        // Second batch with large values
        let x2 = Array2::from_shape_vec((10, 2), vec![1e6; 20]).unwrap();
        let y2 = Array1::from_vec(vec![0.0; 10]);
        model.partial_fit(&x2, &y2, None).unwrap();

        // Variance should be computed (not NaN or Inf)
        let var = model.var().unwrap();
        for &v in var.iter() {
            assert!(v.is_finite(), "Variance should be finite, got {}", v);
            assert!(v >= 0.0, "Variance should be non-negative, got {}", v);
        }

        // Mean should be between small and large values
        let theta = model.theta().unwrap();
        let mean_class0 = theta[[0, 0]];
        assert!(
            mean_class0 > 1.0 && mean_class0 < 1e6,
            "Mean should be between batches: {}",
            mean_class0
        );
    }

    #[test]
    fn test_partial_fit_priors_update_correctly() {
        let mut model = GaussianNB::new();

        // First batch: 3 class 0, 1 class 1
        let x1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 5.0, 6.0]).unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        let priors_1 = model.class_prior().unwrap().clone();
        assert!(
            (priors_1[0] - 0.75).abs() < tolerances::CLOSED_FORM,
            "Prior for class 0 should be 0.75"
        );
        assert!(
            (priors_1[1] - 0.25).abs() < tolerances::CLOSED_FORM,
            "Prior for class 1 should be 0.25"
        );

        // Second batch: 1 class 0, 3 class 1 (now balanced overall)
        let x2 =
            Array2::from_shape_vec((4, 2), vec![2.5, 3.5, 5.5, 6.5, 6.0, 7.0, 6.5, 7.5]).unwrap();
        let y2 = Array1::from_vec(vec![0.0, 1.0, 1.0, 1.0]);
        model.partial_fit(&x2, &y2, None).unwrap();

        let priors_2 = model.class_prior().unwrap();
        // Total: 4 class 0, 4 class 1 = 0.5 each
        assert!(
            (priors_2[0] - 0.5).abs() < tolerances::CLOSED_FORM,
            "Prior for class 0 should be 0.5, got {}",
            priors_2[0]
        );
        assert!(
            (priors_2[1] - 0.5).abs() < tolerances::CLOSED_FORM,
            "Prior for class 1 should be 0.5, got {}",
            priors_2[1]
        );
    }

    #[test]
    fn test_partial_fit_with_three_classes() {
        let (x, y) = make_classification_data(30, 4, 3, 333);

        let mut model = GaussianNB::new();
        model
            .partial_fit(&x, &y, Some(vec![0.0, 1.0, 2.0]))
            .unwrap();

        assert!(model.is_fitted());
        assert_eq!(model.classes().unwrap().len(), 3);

        let preds = model.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(
                p == 0.0 || p == 1.0 || p == 2.0,
                "Prediction should be 0, 1, or 2, got {}",
                p
            );
        }
    }
}

// ============================================================================
// MULTINOMIAL NB PARTIAL FIT TESTS
// ============================================================================

#[cfg(test)]
mod multinomial_nb_tests {
    use super::*;

    #[test]
    fn test_partial_fit_requires_classes_on_first_call() {
        let (x, y) = make_count_data(10, 3, 2, 42);
        let mut model = MultinomialNB::new();

        let result = model.partial_fit(&x, &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_fit_rejects_negative_values() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, -1.0, 3.0, 2.0, 1.0, 3.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = MultinomialNB::new();
        let result = model.partial_fit(&x, &y, Some(vec![0.0, 1.0]));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("non-negative"));
    }

    #[test]
    fn test_partial_fit_equivalence_to_fit() {
        let (x, y) = make_count_data(40, 5, 2, 123);

        let mut fit_model = MultinomialNB::new();
        fit_model.fit(&x, &y).unwrap();

        let mut partial_model = MultinomialNB::new();
        partial_model
            .partial_fit(&x, &y, Some(vec![0.0, 1.0]))
            .unwrap();

        // Predictions should match
        let fit_preds = fit_model.predict(&x).unwrap();
        let partial_preds = partial_model.predict(&x).unwrap();
        assert_arrays_close(&fit_preds, &partial_preds, 1e-10, "predictions mismatch");
    }

    #[test]
    fn test_partial_fit_incremental_counts() {
        let (x, y) = make_count_data(40, 4, 2, 456);

        // Split into batches
        let mut incremental = MultinomialNB::new();
        let half = x.nrows() / 2;

        let x1 = x.slice(ndarray::s![..half, ..]).to_owned();
        let y1 = y.slice(ndarray::s![..half]).to_owned();
        incremental
            .partial_fit(&x1, &y1, Some(vec![0.0, 1.0]))
            .unwrap();

        let x2 = x.slice(ndarray::s![half.., ..]).to_owned();
        let y2 = y.slice(ndarray::s![half..]).to_owned();
        incremental.partial_fit(&x2, &y2, None).unwrap();

        // Compare to full fit
        let mut full = MultinomialNB::new();
        full.fit(&x, &y).unwrap();

        // Probabilities should be close
        let inc_proba = incremental.predict_proba(&x).unwrap();
        let full_proba = full.predict_proba(&x).unwrap();
        assert_arrays2_close(
            &inc_proba,
            &full_proba,
            tolerances::ITERATIVE,
            "proba mismatch",
        );
    }

    #[test]
    fn test_partial_fit_feature_count_mismatch() {
        let mut model = MultinomialNB::new();

        let x1 = Array2::from_shape_vec((3, 4), vec![1.0; 12]).unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0]);
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        let x2 = Array2::from_shape_vec((3, 5), vec![1.0; 15]).unwrap();
        let y2 = Array1::from_vec(vec![0.0, 1.0, 1.0]);
        let result = model.partial_fit(&x2, &y2, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_fit_probabilities_sum_to_one() {
        let (x, y) = make_count_data(30, 3, 2, 789);

        let mut model = MultinomialNB::new();
        model.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();

        let probas = model.predict_proba(&x).unwrap();
        for i in 0..probas.nrows() {
            let row_sum: f64 = probas.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < tolerances::ITERATIVE,
                "Row {} probabilities should sum to 1, got {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_partial_fit_with_smoothing() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![5.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0, 4.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = MultinomialNB::new().with_alpha(1.0);
        model.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();

        // With smoothing, should handle zero counts gracefully
        let probas = model.predict_proba(&x).unwrap();
        for &p in probas.iter() {
            assert!(p.is_finite(), "Probability should be finite");
            assert!(p > 0.0, "With smoothing, probability should be > 0");
        }
    }

    #[test]
    fn test_partial_fit_streaming_simulation() {
        // Simulate streaming data arrival
        let (x, y) = make_count_data(100, 5, 3, 999);

        let mut model = MultinomialNB::new();
        let batch_size = 10;

        for batch_idx in 0..(x.nrows() / batch_size) {
            let start = batch_idx * batch_size;
            let end = start + batch_size;
            let x_batch = x.slice(ndarray::s![start..end, ..]).to_owned();
            let y_batch = y.slice(ndarray::s![start..end]).to_owned();

            let classes = if batch_idx == 0 {
                Some(vec![0.0, 1.0, 2.0])
            } else {
                None
            };
            model.partial_fit(&x_batch, &y_batch, classes).unwrap();

            // Model should be usable after each batch
            assert!(model.is_fitted());
            let _preds = model.predict(&x_batch).unwrap();
        }
    }
}

// ============================================================================
// BERNOULLI NB PARTIAL FIT TESTS
// ============================================================================

#[cfg(test)]
mod bernoulli_nb_tests {
    use super::*;

    #[test]
    fn test_partial_fit_requires_classes_on_first_call() {
        let (x, y) = make_binary_data(10, 3, 2, 42);
        let mut model = BernoulliNB::new();

        let result = model.partial_fit(&x, &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_fit_equivalence_to_fit() {
        let (x, y) = make_binary_data(50, 4, 2, 123);

        let mut fit_model = BernoulliNB::new();
        fit_model.fit(&x, &y).unwrap();

        let mut partial_model = BernoulliNB::new();
        partial_model
            .partial_fit(&x, &y, Some(vec![0.0, 1.0]))
            .unwrap();

        let fit_preds = fit_model.predict(&x).unwrap();
        let partial_preds = partial_model.predict(&x).unwrap();
        assert_arrays_close(&fit_preds, &partial_preds, 1e-10, "predictions mismatch");
    }

    #[test]
    fn test_partial_fit_incremental_updates() {
        // 30 samples per class * 2 classes = 60 total
        let (x, y) = make_binary_data(30, 4, 2, 456);
        let total_samples = x.nrows();

        let batch_size = total_samples / 3;
        let mut incremental = BernoulliNB::new();

        for i in 0..3 {
            let start = i * batch_size;
            let end = start + batch_size;
            let x_batch = x.slice(ndarray::s![start..end, ..]).to_owned();
            let y_batch = y.slice(ndarray::s![start..end]).to_owned();

            let classes = if i == 0 { Some(vec![0.0, 1.0]) } else { None };
            incremental
                .partial_fit(&x_batch, &y_batch, classes)
                .unwrap();
        }

        let mut full = BernoulliNB::new();
        full.fit(&x, &y).unwrap();

        let inc_proba = incremental.predict_proba(&x).unwrap();
        let full_proba = full.predict_proba(&x).unwrap();
        assert_arrays2_close(
            &inc_proba,
            &full_proba,
            tolerances::ITERATIVE,
            "proba mismatch",
        );
    }

    #[test]
    fn test_partial_fit_with_binarization() {
        // Test that binarization works consistently across batches
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.3, 0.7, 0.2, // Will be binarized to 0, 1, 0
                0.8, 0.1, 0.9, // Will be binarized to 1, 0, 1
                0.6, 0.6, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.4, 0.7, 0.3, 0.6, 0.2, 0.8, 0.5, 0.4,
                0.6, 0.7,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut model = BernoulliNB::new().with_binarize(Some(0.5));
        model.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();

        // Should produce valid predictions
        let preds = model.predict(&x).unwrap();
        for &p in preds.iter() {
            assert!(p == 0.0 || p == 1.0);
        }
    }

    #[test]
    fn test_partial_fit_feature_mismatch() {
        let mut model = BernoulliNB::new();

        let x1 = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let y1 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();

        let x2 = Array2::from_shape_vec((4, 4), vec![1.0; 16]).unwrap();
        let y2 = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let result = model.partial_fit(&x2, &y2, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_fit_probabilities_valid() {
        let (x, y) = make_binary_data(40, 5, 2, 789);

        let mut model = BernoulliNB::new();
        model.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();

        let probas = model.predict_proba(&x).unwrap();

        // All probabilities in [0, 1]
        for &p in probas.iter() {
            assert!(
                p >= 0.0 && p <= 1.0,
                "Probability should be in [0, 1], got {}",
                p
            );
        }

        // Rows sum to 1
        for i in 0..probas.nrows() {
            let row_sum: f64 = probas.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < tolerances::ITERATIVE,
                "Row {} should sum to 1, got {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_partial_fit_single_sample_per_batch() {
        let (x, y) = make_binary_data(10, 3, 2, 111);

        let mut model = BernoulliNB::new();

        for i in 0..x.nrows() {
            let x_i = x.slice(ndarray::s![i..i + 1, ..]).to_owned();
            let y_i = y.slice(ndarray::s![i..i + 1]).to_owned();

            let classes = if i == 0 { Some(vec![0.0, 1.0]) } else { None };
            model.partial_fit(&x_i, &y_i, classes).unwrap();
        }

        assert!(model.is_fitted());
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), x.nrows());
    }

    #[test]
    fn test_partial_fit_with_smoothing_alpha() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = BernoulliNB::new().with_alpha(1.0);
        model.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();

        let probas = model.predict_proba(&x).unwrap();
        for &p in probas.iter() {
            assert!(p.is_finite() && p > 0.0);
        }
    }
}

// ============================================================================
// CROSS-MODEL COMPARISON TESTS
// ============================================================================

#[cfg(test)]
mod cross_model_tests {
    use super::*;

    #[test]
    fn test_all_nb_models_support_partial_fit() {
        // Verify all three NB models work with partial_fit
        let classes = Some(vec![0.0, 1.0]);

        // GaussianNB
        let (x_cont, y) = make_classification_data(20, 3, 2, 42);
        let mut gnb = GaussianNB::new();
        gnb.partial_fit(&x_cont, &y, classes.clone()).unwrap();
        assert!(gnb.is_fitted());

        // MultinomialNB
        let (x_count, y) = make_count_data(20, 3, 2, 42);
        let mut mnb = MultinomialNB::new();
        mnb.partial_fit(&x_count, &y, classes.clone()).unwrap();
        assert!(mnb.is_fitted());

        // BernoulliNB
        let (x_bin, y) = make_binary_data(20, 3, 2, 42);
        let mut bnb = BernoulliNB::new();
        bnb.partial_fit(&x_bin, &y, classes).unwrap();
        assert!(bnb.is_fitted());
    }

    #[test]
    fn test_partial_fit_error_messages_consistent() {
        // All models should give similar error for missing classes
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut gnb = GaussianNB::new();
        let gnb_err = gnb.partial_fit(&x, &y, None).unwrap_err();
        assert!(gnb_err.to_string().contains("Classes"));

        let mut mnb = MultinomialNB::new();
        let mnb_err = mnb.partial_fit(&x, &y, None).unwrap_err();
        assert!(mnb_err.to_string().contains("Classes"));

        let mut bnb = BernoulliNB::new();
        let bnb_err = bnb.partial_fit(&x, &y, None).unwrap_err();
        assert!(bnb_err.to_string().contains("Classes"));
    }
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_classes_rejected() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = GaussianNB::new();
        let result = model.partial_fit(&x, &y, Some(vec![]));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_partial_fit_many_batches() {
        // Stress test with many small batches
        // 50 samples per class * 2 classes = 100 total samples
        let (x, y) = make_classification_data(50, 3, 2, 555);
        let total_samples = x.nrows(); // 100

        let mut model = GaussianNB::new();
        let batch_size = 2;

        for i in 0..(total_samples / batch_size) {
            let start = i * batch_size;
            let end = start + batch_size;
            let x_batch = x.slice(ndarray::s![start..end, ..]).to_owned();
            let y_batch = y.slice(ndarray::s![start..end]).to_owned();

            let classes = if i == 0 { Some(vec![0.0, 1.0]) } else { None };
            model.partial_fit(&x_batch, &y_batch, classes).unwrap();
        }

        // Should still work after 50 batches
        assert!(model.is_fitted());
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), total_samples);
    }

    #[test]
    fn test_partial_fit_reproducibility() {
        let (x, y) = make_classification_data(30, 3, 2, 777);

        // Train two models the same way
        let mut model1 = GaussianNB::new();
        let mut model2 = GaussianNB::new();

        model1.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();
        model2.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();

        // Should have identical parameters
        assert_arrays2_close(
            model1.theta().unwrap(),
            model2.theta().unwrap(),
            1e-15,
            "theta should be identical",
        );
        assert_arrays2_close(
            model1.var().unwrap(),
            model2.var().unwrap(),
            1e-15,
            "var should be identical",
        );
    }

    #[test]
    fn test_partial_fit_after_fit_resets() {
        // Verify that fit() resets the model even after partial_fit
        let (x1, y1) = make_classification_data(20, 3, 2, 111);
        let (x2, y2) = make_classification_data(20, 3, 2, 222);

        let mut model = GaussianNB::new();

        // First: partial_fit
        model.partial_fit(&x1, &y1, Some(vec![0.0, 1.0])).unwrap();
        let theta_after_partial = model.theta().unwrap().clone();

        // Then: fit on different data
        model.fit(&x2, &y2).unwrap();
        let theta_after_fit = model.theta().unwrap();

        // Parameters should be different (fit resets, doesn't accumulate)
        let mut different = false;
        for (&a, &b) in theta_after_partial.iter().zip(theta_after_fit.iter()) {
            if (a - b).abs() > tolerances::ITERATIVE {
                different = true;
                break;
            }
        }
        assert!(different, "fit() should reset model, not accumulate");
    }

    #[test]
    fn test_multinomial_zero_counts_handled() {
        // Test that MultinomialNB handles batches where some features have zero counts
        let x = Array2::from_shape_vec(
            (4, 4),
            vec![
                5.0, 0.0, 0.0, 0.0, // All count in first feature
                4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, // All count in last feature
                0.0, 0.0, 1.0, 4.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = MultinomialNB::new().with_alpha(1.0);
        model.partial_fit(&x, &y, Some(vec![0.0, 1.0])).unwrap();

        // Should produce valid finite probabilities
        let probas = model.predict_proba(&x).unwrap();
        for &p in probas.iter() {
            assert!(p.is_finite(), "Probability should be finite");
        }
    }
}

// ============================================================================
// WARM START PLACEHOLDER TESTS
// ============================================================================

/// WarmStartModel tests using a mock ensemble to validate the trait contract.
#[cfg(test)]
mod warm_start_tests {
    use super::*;
    use crate::models::traits::WarmStartModel;
    use crate::models::Model;
    use crate::{FerroError, Result};

    #[derive(Clone)]
    struct MockEnsemble {
        warm_start: bool,
        n_estimators_target: usize,
        n_estimators_fitted: usize,
        is_fitted: bool,
        n_features: Option<usize>,
    }

    impl MockEnsemble {
        fn new(n_estimators: usize) -> Self {
            Self {
                warm_start: false,
                n_estimators_target: n_estimators,
                n_estimators_fitted: 0,
                is_fitted: false,
                n_features: None,
            }
        }
    }

    impl Model for MockEnsemble {
        fn fit(&mut self, x: &Array2<f64>, _y: &Array1<f64>) -> Result<()> {
            self.n_features = Some(x.ncols());
            if self.warm_start && self.is_fitted {
                let new = self
                    .n_estimators_target
                    .saturating_sub(self.n_estimators_fitted);
                self.n_estimators_fitted += new;
            } else {
                self.n_estimators_fitted = self.n_estimators_target;
            }
            self.is_fitted = true;
            Ok(())
        }

        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            if !self.is_fitted {
                return Err(FerroError::not_fitted("MockEnsemble"));
            }
            Ok(Array1::zeros(x.nrows()))
        }

        fn is_fitted(&self) -> bool {
            self.is_fitted
        }

        fn feature_importance(&self) -> Option<Array1<f64>> {
            None
        }

        fn search_space(&self) -> crate::hpo::SearchSpace {
            crate::hpo::SearchSpace::new()
        }

        fn feature_names(&self) -> Option<&[String]> {
            None
        }

        fn n_features(&self) -> Option<usize> {
            self.n_features
        }
    }

    impl WarmStartModel for MockEnsemble {
        fn set_warm_start(&mut self, warm_start: bool) {
            self.warm_start = warm_start;
        }
        fn warm_start(&self) -> bool {
            self.warm_start
        }
        fn n_estimators_fitted(&self) -> usize {
            self.n_estimators_fitted
        }
    }

    #[test]
    fn test_warm_start_trait_exists() {
        fn _accepts_warm_start<T: WarmStartModel>(_: &T) {}
    }

    #[test]
    fn test_warm_start_adds_estimators() {
        let (x, y) = make_classification_data(20, 3, 2, 42);
        let mut model = MockEnsemble::new(5);
        model.warm_start = true;
        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), 5);

        model.n_estimators_target = 10;
        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), 10);
    }

    #[test]
    fn test_warm_start_preserves_existing_estimators() {
        let (x, y) = make_classification_data(20, 3, 2, 42);
        let mut model = MockEnsemble::new(5);
        model.warm_start = true;
        model.fit(&x, &y).unwrap();
        let count = model.n_estimators_fitted();

        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), count);
    }

    #[test]
    fn test_warm_start_disabled_resets_model() {
        let (x, y) = make_classification_data(20, 3, 2, 42);
        let mut model = MockEnsemble::new(5);
        model.warm_start = false;
        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), 5);

        model.n_estimators_target = 3;
        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), 3);
    }

    #[test]
    fn test_n_estimators_fitted_increases_with_warm_start() {
        let (x, y) = make_classification_data(20, 3, 2, 42);
        let mut model = MockEnsemble::new(3);
        model.warm_start = true;
        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), 3);

        model.n_estimators_target = 6;
        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), 6);

        model.n_estimators_target = 10;
        model.fit(&x, &y).unwrap();
        assert_eq!(model.n_estimators_fitted(), 10);
    }
}
