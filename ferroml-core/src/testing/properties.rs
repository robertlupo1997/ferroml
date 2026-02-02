//! Property-Based Tests with proptest
//!
//! This module provides property-based testing using proptest to validate
//! invariants that must hold across all valid inputs. These tests complement
//! the deterministic check_estimator framework by exploring the input space
//! more thoroughly.
//!
//! ## Key Properties Tested
//!
//! ### Model Invariants
//! - `fit` then `predict` doesn't panic for valid random data
//! - Predictions have same length as input samples
//! - `n_features()` matches input after fit
//! - Clone produces equivalent model
//! - Multiple predicts give identical results
//! - Fitted model rejects wrong feature count
//!
//! ### Transformer Invariants
//! - Output rows == input rows
//! - `fit_transform` == `fit` then `transform`
//!
//! ### Serialization Invariants
//! - Round-trip preserves predictions
//!
//! ## Usage
//!
//! Property tests are only available during test compilation.
//! Use `cargo test -p ferroml-core properties` to run them.
//!
//! ```ignore
//! // Property tests are run via cargo test
//! cargo test -p ferroml-core properties
//! cargo test -p ferroml-core properties -- --ignored  // For longer-running tests
//! ```

// ============================================================================
// Helper Functions (available in both test and non-test builds)
// ============================================================================

/// Convert a Vec<Vec<f64>> to ndarray Array2.
pub fn vecs_to_array2(vecs: Vec<Vec<f64>>) -> ndarray::Array2<f64> {
    if vecs.is_empty() {
        return ndarray::Array2::zeros((0, 0));
    }
    let n_rows = vecs.len();
    let n_cols = vecs[0].len();
    let flat: Vec<f64> = vecs.into_iter().flatten().collect();
    ndarray::Array2::from_shape_vec((n_rows, n_cols), flat).unwrap()
}

/// Convert a Vec<f64> to ndarray Array1.
pub fn vec_to_array1(vec: Vec<f64>) -> ndarray::Array1<f64> {
    ndarray::Array1::from_vec(vec)
}

/// Check if all values in an array are finite (no NaN or Inf).
pub fn all_finite(arr: &ndarray::Array1<f64>) -> bool {
    arr.iter().all(|&v| v.is_finite())
}

/// Check if two arrays are approximately equal.
pub fn arrays_approx_equal(a: &ndarray::Array1<f64>, b: &ndarray::Array1<f64>, tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| (x - y).abs() < tol || (x.is_nan() && y.is_nan()))
}

// ============================================================================
// Model-Family Specific Property Test Functions
// (These can be called from any test, with or without proptest)
// ============================================================================

/// Property tests specific to linear models.
pub mod linear_model_properties {
    use crate::models::traits::LinearModel;
    use crate::models::Model;

    /// Test that coefficients have correct dimension after fitting.
    pub fn prop_coefficients_dimension<M: Model + LinearModel + Clone>(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| i as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_ok() {
            if let Some(coefs) = model.coefficients() {
                return coefs.len() == n_features;
            }
        }
        true // If fit fails or no coefficients, we can't test this
    }

    /// Property: linear model predictions are linear in features.
    /// For data x, predictions should be close to X @ coef + intercept.
    pub fn prop_predictions_linear<M: Model + LinearModel + Clone>(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| i as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        let coefs = match model.coefficients() {
            Some(c) => c.clone(),
            None => return true,
        };
        let intercept = model.intercept().unwrap_or(0.0);

        if let Ok(predictions) = model.predict(&x) {
            // Compute expected: X @ coef + intercept
            for i in 0..n_samples {
                let expected: f64 = x
                    .row(i)
                    .iter()
                    .zip(coefs.iter())
                    .map(|(x, c)| x * c)
                    .sum::<f64>()
                    + intercept;
                let actual = predictions[i];
                if (expected - actual).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }
}

/// Property tests specific to tree models.
pub mod tree_model_properties {
    use crate::models::Model;

    /// Test that tree predictions are bounded by training target range.
    pub fn prop_predictions_bounded<M: Model + Clone>(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| i as f64);

        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        if let Ok(predictions) = model.predict(&x) {
            // All predictions should be within [y_min, y_max] for regression trees
            for &pred in predictions.iter() {
                if pred < y_min - 1e-10 || pred > y_max + 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Test that feature importances sum to 1 (or close to it).
    pub fn prop_feature_importances_sum_to_one<M: Model + Clone>(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| i as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        if let Some(importances) = model.feature_importance() {
            let sum: f64 = importances.iter().sum();
            // Feature importances should sum to approximately 1
            return (sum - 1.0).abs() < 0.01 || sum < 1e-10; // Allow zero if no splits
        }
        true
    }
}

/// Property tests specific to probabilistic models.
pub mod probabilistic_model_properties {
    use crate::models::{Model, ProbabilisticModel};

    /// Test that predict_proba outputs sum to 1.0 for each sample.
    pub fn prop_probabilities_sum_to_one<M: Model + ProbabilisticModel + Clone>(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        // Binary classification targets
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        if let Ok(proba) = model.predict_proba(&x) {
            for i in 0..proba.nrows() {
                let row_sum: f64 = proba.row(i).iter().sum();
                if (row_sum - 1.0).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    /// Test that all probabilities are in [0, 1].
    pub fn prop_probabilities_in_range<M: Model + ProbabilisticModel + Clone>(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        if let Ok(proba) = model.predict_proba(&x) {
            for &p in proba.iter() {
                if p < -1e-10 || p > 1.0 + 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Test that predicted class matches highest probability.
    pub fn prop_predict_matches_argmax_proba<M: Model + ProbabilisticModel + Clone>(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| (i % 2) as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        let predictions = match model.predict(&x) {
            Ok(p) => p,
            Err(_) => return true,
        };
        let probabilities = match model.predict_proba(&x) {
            Ok(p) => p,
            Err(_) => return true,
        };

        for i in 0..n_samples {
            let proba_row = probabilities.row(i);
            let mut max_idx = 0;
            let mut max_val = proba_row[0];
            for (j, &p) in proba_row.iter().enumerate() {
                if p > max_val {
                    max_val = p;
                    max_idx = j;
                }
            }
            let predicted_class = predictions[i] as usize;
            if predicted_class != max_idx {
                return false;
            }
        }
        true
    }
}

/// Property tests for serialization round-trips.
pub mod serialization_properties {
    use super::arrays_approx_equal;
    use crate::models::Model;
    use serde::{de::DeserializeOwned, Serialize};

    /// Test that JSON serialization round-trip preserves predictions.
    pub fn prop_json_roundtrip_preserves_predictions<
        M: Model + Clone + Serialize + DeserializeOwned,
    >(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| i as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        let pred_before = match model.predict(&x) {
            Ok(p) => p,
            Err(_) => return true,
        };

        // Serialize to JSON
        let json = match serde_json::to_string(&model) {
            Ok(s) => s,
            Err(_) => return false,
        };

        // Deserialize
        let restored: M = match serde_json::from_str(&json) {
            Ok(m) => m,
            Err(_) => return false,
        };

        let pred_after = match restored.predict(&x) {
            Ok(p) => p,
            Err(_) => return false,
        };

        arrays_approx_equal(&pred_before, &pred_after, 1e-10)
    }

    /// Test that bincode serialization round-trip preserves predictions.
    pub fn prop_bincode_roundtrip_preserves_predictions<
        M: Model + Clone + Serialize + DeserializeOwned,
    >(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| i as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        let pred_before = match model.predict(&x) {
            Ok(p) => p,
            Err(_) => return true,
        };

        // Serialize to bincode
        let bytes = match bincode::serialize(&model) {
            Ok(b) => b,
            Err(_) => return false,
        };

        // Deserialize
        let restored: M = match bincode::deserialize(&bytes) {
            Ok(m) => m,
            Err(_) => return false,
        };

        let pred_after = match restored.predict(&x) {
            Ok(p) => p,
            Err(_) => return false,
        };

        arrays_approx_equal(&pred_before, &pred_after, 1e-10)
    }

    /// Test that MessagePack serialization round-trip preserves predictions.
    pub fn prop_msgpack_roundtrip_preserves_predictions<
        M: Model + Clone + Serialize + DeserializeOwned,
    >(
        model_constructor: impl Fn() -> M,
        n_samples: usize,
        n_features: usize,
    ) -> bool {
        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i + j) as f64);
        let y = ndarray::Array1::from_shape_fn(n_samples, |i| i as f64);

        let mut model = model_constructor();
        if model.fit(&x, &y).is_err() {
            return true;
        }

        let pred_before = match model.predict(&x) {
            Ok(p) => p,
            Err(_) => return true,
        };

        // Serialize to MessagePack
        let bytes = match rmp_serde::to_vec(&model) {
            Ok(b) => b,
            Err(_) => return false,
        };

        // Deserialize
        let restored: M = match rmp_serde::from_slice(&bytes) {
            Ok(m) => m,
            Err(_) => return false,
        };

        let pred_after = match restored.predict(&x) {
            Ok(p) => p,
            Err(_) => return false,
        };

        arrays_approx_equal(&pred_before, &pred_after, 1e-10)
    }
}

// ============================================================================
// Proptest-based tests (only available during test compilation)
// ============================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::strategy::Strategy;

    // ========================================================================
    // Proptest Strategies for ML Data Generation
    // ========================================================================

    /// Strategy for generating random feature matrices.
    pub fn feature_matrix_strategy(
        min_samples: usize,
        max_samples: usize,
        min_features: usize,
        max_features: usize,
    ) -> impl Strategy<Value = Vec<Vec<f64>>> {
        (min_samples..=max_samples, min_features..=max_features).prop_flat_map(
            move |(n_samples, n_features)| {
                proptest::collection::vec(
                    proptest::collection::vec(reasonable_float_strategy(), n_features),
                    n_samples,
                )
            },
        )
    }

    /// Strategy for generating reasonable float values (avoiding extreme values).
    pub fn reasonable_float_strategy() -> impl Strategy<Value = f64> {
        prop_oneof![
            // Most values are in a reasonable range
            8 => -100.0..100.0f64,
            // Some edge cases
            1 => Just(0.0),
            // Near-zero values
            1 => -1e-6..1e-6f64,
        ]
    }

    /// Strategy for edge-case matrices.
    #[allow(dead_code)]
    pub fn edge_case_matrix_strategy(
        n_samples: usize,
        n_features: usize,
    ) -> impl Strategy<Value = Vec<Vec<f64>>> {
        prop_oneof![
            // Normal random data
            5 => proptest::collection::vec(
                proptest::collection::vec(reasonable_float_strategy(), n_features),
                n_samples
            ),
            // Near-zero values
            1 => proptest::collection::vec(
                proptest::collection::vec(-1e-8..1e-8f64, n_features),
                n_samples
            ),
            // Large values (but not extreme)
            1 => proptest::collection::vec(
                proptest::collection::vec(1e6..1e8f64, n_features),
                n_samples
            ),
            // Negative values
            1 => proptest::collection::vec(
                proptest::collection::vec(-100.0..-1.0f64, n_features),
                n_samples
            ),
            // Mixed scales
            1 => proptest::collection::vec(
                proptest::collection::vec(
                    prop_oneof![
                        Just(0.0),
                        -1e-6..1e-6f64,
                        -100.0..100.0f64,
                        1e4..1e6f64,
                    ],
                    n_features
                ),
                n_samples
            ),
        ]
    }

    /// Strategy for random seeds for reproducibility testing.
    pub fn random_seed_strategy() -> impl Strategy<Value = u64> {
        0u64..=u64::MAX
    }

    // ========================================================================
    // Property Test Macros
    // ========================================================================

    /// Generate property tests for a model type.
    macro_rules! proptest_model {
        (
            model_name: $name:ident,
            model_type: $model_type:ty,
            model_constructor: $constructor:expr,
            is_classifier: $is_classifier:expr,
        ) => {
            mod $name {
                use super::*;
                use crate::models::Model;

                proptest! {
                    #![proptest_config(ProptestConfig::with_cases(50))]

                    #[test]
                    fn prop_fit_predict_no_panic(
                        x_data in feature_matrix_strategy(10, 100, 2, 10),
                        _seed in random_seed_strategy()
                    ) {
                        let x = vecs_to_array2(x_data.clone());
                        if x.nrows() < 2 || x.ncols() < 1 {
                            return Ok(());
                        }

                        // Generate matching target
                        let y = if $is_classifier {
                            vec_to_array1((0..x.nrows()).map(|i| (i % 2) as f64).collect())
                        } else {
                            vec_to_array1((0..x.nrows()).map(|i| i as f64).collect())
                        };

                        let mut model: $model_type = $constructor;
                        let _ = model.fit(&x, &y);
                        // If fit succeeds, predict should not panic
                        if model.is_fitted() {
                            let _ = model.predict(&x);
                        }
                    }

                    #[test]
                    fn prop_predict_shape_matches_input(
                        n_samples in 10usize..100,
                        n_features in 2usize..10,
                    ) {
                        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                            (i + j) as f64
                        });
                        let y = if $is_classifier {
                            vec_to_array1((0..n_samples).map(|i| (i % 2) as f64).collect())
                        } else {
                            vec_to_array1((0..n_samples).map(|i| i as f64).collect())
                        };

                        let mut model: $model_type = $constructor;
                        if model.fit(&x, &y).is_ok() {
                            if let Ok(pred) = model.predict(&x) {
                                prop_assert_eq!(pred.len(), n_samples);
                            }
                        }
                    }

                    #[test]
                    fn prop_n_features_tracked(
                        n_samples in 10usize..50,
                        n_features in 2usize..10,
                    ) {
                        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                            (i + j) as f64
                        });
                        let y = if $is_classifier {
                            vec_to_array1((0..n_samples).map(|i| (i % 2) as f64).collect())
                        } else {
                            vec_to_array1((0..n_samples).map(|i| i as f64).collect())
                        };

                        let mut model: $model_type = $constructor;
                        if model.fit(&x, &y).is_ok() {
                            if let Some(n) = model.n_features() {
                                prop_assert_eq!(n, n_features);
                            }
                        }
                    }

                    #[test]
                    fn prop_clone_equivalence(
                        n_samples in 20usize..50,
                        n_features in 2usize..5,
                    ) {
                        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                            (i + j) as f64
                        });
                        let y = if $is_classifier {
                            vec_to_array1((0..n_samples).map(|i| (i % 2) as f64).collect())
                        } else {
                            vec_to_array1((0..n_samples).map(|i| i as f64).collect())
                        };

                        let mut model: $model_type = $constructor;
                        if model.fit(&x, &y).is_ok() {
                            let cloned = model.clone();
                            if let (Ok(pred1), Ok(pred2)) = (model.predict(&x), cloned.predict(&x)) {
                                prop_assert!(arrays_approx_equal(&pred1, &pred2, 1e-10));
                            }
                        }
                    }

                    #[test]
                    fn prop_multiple_predict_identical(
                        n_samples in 20usize..50,
                        n_features in 2usize..5,
                    ) {
                        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                            (i + j) as f64
                        });
                        let y = if $is_classifier {
                            vec_to_array1((0..n_samples).map(|i| (i % 2) as f64).collect())
                        } else {
                            vec_to_array1((0..n_samples).map(|i| i as f64).collect())
                        };

                        let mut model: $model_type = $constructor;
                        if model.fit(&x, &y).is_ok() {
                            let pred1 = model.predict(&x);
                            let pred2 = model.predict(&x);
                            let pred3 = model.predict(&x);

                            if let (Ok(p1), Ok(p2), Ok(p3)) = (pred1, pred2, pred3) {
                                prop_assert!(arrays_approx_equal(&p1, &p2, 1e-12));
                                prop_assert!(arrays_approx_equal(&p2, &p3, 1e-12));
                            }
                        }
                    }

                    #[test]
                    fn prop_rejects_wrong_feature_count(
                        n_samples in 20usize..50,
                        n_features in 3usize..8,
                    ) {
                        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                            (i + j) as f64
                        });
                        let y = if $is_classifier {
                            vec_to_array1((0..n_samples).map(|i| (i % 2) as f64).collect())
                        } else {
                            vec_to_array1((0..n_samples).map(|i| i as f64).collect())
                        };

                        let mut model: $model_type = $constructor;
                        if model.fit(&x, &y).is_ok() {
                            // Try predicting with wrong number of features
                            let x_wrong = ndarray::Array2::zeros((10, n_features + 1));
                            let result = model.predict(&x_wrong);
                            prop_assert!(result.is_err(), "Should reject wrong feature count");
                        }
                    }
                }
            }
        };
    }

    /// Generate property tests for a transformer type.
    macro_rules! proptest_transformer {
        (
            transformer_name: $name:ident,
            transformer_type: $transformer_type:ty,
            transformer_constructor: $constructor:expr,
        ) => {
            mod $name {
                use super::*;
                use crate::preprocessing::Transformer;

                proptest! {
                    #![proptest_config(ProptestConfig::with_cases(50))]

                    #[test]
                    fn prop_transform_preserves_row_count(
                        n_samples in 10usize..100,
                        n_features in 2usize..10,
                    ) {
                        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                            (i * 3 + j) as f64 + 0.1
                        });

                        let mut transformer: $transformer_type = $constructor;
                        if transformer.fit(&x).is_ok() {
                            if let Ok(result) = transformer.transform(&x) {
                                prop_assert_eq!(result.nrows(), n_samples);
                            }
                        }
                    }

                    #[test]
                    fn prop_fit_transform_equals_fit_then_transform(
                        n_samples in 20usize..50,
                        n_features in 2usize..5,
                    ) {
                        let x = ndarray::Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                            (i * 2 + j) as f64 + 1.0
                        });

                        // Method 1: fit then transform
                        let mut t1: $transformer_type = $constructor;
                        let result1 = t1.fit(&x).and_then(|_| t1.transform(&x));

                        // Method 2: fit_transform
                        let mut t2: $transformer_type = $constructor;
                        let result2 = t2.fit_transform(&x);

                        match (result1, result2) {
                            (Ok(r1), Ok(r2)) => {
                                prop_assert_eq!(r1.shape(), r2.shape());
                                for (&a, &b) in r1.iter().zip(r2.iter()) {
                                    if a.is_finite() && b.is_finite() {
                                        prop_assert!((a - b).abs() < 1e-10,
                                            "fit+transform and fit_transform differ: {} vs {}", a, b);
                                    }
                                }
                            }
                            (Err(_), Err(_)) => {
                                // Both failed consistently
                            }
                            (Ok(_), Err(_)) | (Err(_), Ok(_)) => {
                                prop_assert!(false, "Inconsistent behavior between fit+transform and fit_transform");
                            }
                        }
                    }
                }
            }
        };
    }

    // ========================================================================
    // Tests for Helpers
    // ========================================================================

    #[test]
    fn test_vecs_to_array2() {
        let vecs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let arr = vecs_to_array2(vecs);
        assert_eq!(arr.shape(), &[2, 3]);
        assert!((arr[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((arr[[1, 2]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec_to_array1() {
        let vec = vec![1.0, 2.0, 3.0];
        let arr = vec_to_array1(vec);
        assert_eq!(arr.len(), 3);
        assert!((arr[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_finite() {
        let arr = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(all_finite(&arr));

        let arr_nan = ndarray::Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        assert!(!all_finite(&arr_nan));

        let arr_inf = ndarray::Array1::from_vec(vec![1.0, f64::INFINITY, 3.0]);
        assert!(!all_finite(&arr_inf));
    }

    #[test]
    fn test_arrays_approx_equal() {
        let a = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert!(arrays_approx_equal(&a, &b, 1e-10));

        let c = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0 + 1e-11]);
        assert!(arrays_approx_equal(&a, &c, 1e-10));

        let d = ndarray::Array1::from_vec(vec![1.0, 2.0, 4.0]);
        assert!(!arrays_approx_equal(&a, &d, 1e-10));
    }

    // Property tests for the strategies themselves
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn test_feature_matrix_strategy_valid_shape(
            matrix in feature_matrix_strategy(5, 20, 2, 5)
        ) {
            prop_assert!(matrix.len() >= 5);
            prop_assert!(matrix.len() <= 20);
            if !matrix.is_empty() {
                prop_assert!(matrix[0].len() >= 2);
                prop_assert!(matrix[0].len() <= 5);
            }
        }

        #[test]
        fn test_reasonable_float_strategy_finite(value in reasonable_float_strategy()) {
            prop_assert!(value.is_finite());
        }
    }

    // ========================================================================
    // Integration Tests with Actual Models
    // ========================================================================

    use crate::models::LinearRegression;

    // Generate property tests for LinearRegression
    proptest_model! {
        model_name: linear_regression_props,
        model_type: LinearRegression,
        model_constructor: LinearRegression::new(),
        is_classifier: false,
    }

    // Note: Linear model specific property tests are commented out because
    // LinearRegression doesn't implement LinearModel trait yet.
    // Uncomment when the trait implementation is added.
    //
    // #[test]
    // fn test_linear_regression_coefficients_dimension() {
    //     assert!(linear_model_properties::prop_coefficients_dimension(
    //         LinearRegression::new,
    //         50,
    //         5
    //     ));
    // }
    //
    // #[test]
    // fn test_linear_regression_predictions_linear() {
    //     assert!(linear_model_properties::prop_predictions_linear(
    //         LinearRegression::new,
    //         50,
    //         5
    //     ));
    // }

    // Test serialization properties
    #[test]
    fn test_linear_regression_json_roundtrip() {
        assert!(
            serialization_properties::prop_json_roundtrip_preserves_predictions(
                LinearRegression::new,
                50,
                5
            )
        );
    }

    #[test]
    fn test_linear_regression_bincode_roundtrip() {
        assert!(
            serialization_properties::prop_bincode_roundtrip_preserves_predictions(
                LinearRegression::new,
                50,
                5
            )
        );
    }

    #[test]
    fn test_linear_regression_msgpack_roundtrip() {
        assert!(
            serialization_properties::prop_msgpack_roundtrip_preserves_predictions(
                LinearRegression::new,
                50,
                5
            )
        );
    }

    // ========================================================================
    // Extended Property Tests (marked #[ignore] for longer runs)
    // ========================================================================

    use crate::models::{
        DecisionTreeRegressor, GaussianNB, LogisticRegression, RandomForestRegressor,
        RidgeRegression,
    };

    // Ridge Regression
    proptest_model! {
        model_name: ridge_regression_props,
        model_type: RidgeRegression,
        model_constructor: RidgeRegression::new(1.0),
        is_classifier: false,
    }

    // Logistic Regression
    proptest_model! {
        model_name: logistic_regression_props,
        model_type: LogisticRegression,
        model_constructor: LogisticRegression::new(),
        is_classifier: true,
    }

    // Decision Tree Regressor
    proptest_model! {
        model_name: decision_tree_regressor_props,
        model_type: DecisionTreeRegressor,
        model_constructor: DecisionTreeRegressor::new(),
        is_classifier: false,
    }

    // Random Forest Regressor (with small n_estimators for speed)
    proptest_model! {
        model_name: random_forest_regressor_props,
        model_type: RandomForestRegressor,
        model_constructor: {
            let mut rf = RandomForestRegressor::new();
            rf.n_estimators = 5;
            rf
        },
        is_classifier: false,
    }

    // Gaussian Naive Bayes
    proptest_model! {
        model_name: gaussian_nb_props,
        model_type: GaussianNB,
        model_constructor: GaussianNB::new(),
        is_classifier: true,
    }

    // Transformer tests
    use crate::preprocessing::scalers::{MinMaxScaler, StandardScaler};

    proptest_transformer! {
        transformer_name: standard_scaler_props,
        transformer_type: StandardScaler,
        transformer_constructor: StandardScaler::new(),
    }

    proptest_transformer! {
        transformer_name: minmax_scaler_props,
        transformer_type: MinMaxScaler,
        transformer_constructor: MinMaxScaler::new(),
    }

    // Test tree-specific properties
    #[test]
    #[ignore] // Long running
    fn test_tree_predictions_bounded() {
        assert!(tree_model_properties::prop_predictions_bounded(
            DecisionTreeRegressor::new,
            100,
            5
        ));
    }

    #[test]
    #[ignore] // Long running
    fn test_forest_predictions_bounded() {
        assert!(tree_model_properties::prop_predictions_bounded(
            || {
                let mut rf = RandomForestRegressor::new();
                rf.n_estimators = 5;
                rf
            },
            100,
            5
        ));
    }

    // Test probabilistic model properties
    #[test]
    #[ignore] // Long running
    fn test_logistic_regression_probabilities_sum_to_one() {
        assert!(
            probabilistic_model_properties::prop_probabilities_sum_to_one(
                LogisticRegression::new,
                100,
                5
            )
        );
    }

    #[test]
    #[ignore] // Long running
    fn test_logistic_regression_probabilities_in_range() {
        assert!(probabilistic_model_properties::prop_probabilities_in_range(
            LogisticRegression::new,
            100,
            5
        ));
    }

    #[test]
    #[ignore] // Long running
    fn test_gaussian_nb_probabilities_sum_to_one() {
        assert!(
            probabilistic_model_properties::prop_probabilities_sum_to_one(GaussianNB::new, 100, 5)
        );
    }
}
