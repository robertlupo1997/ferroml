//! Advanced Categorical Encoding Tests
//!
//! Phase 26 testing: comprehensive tests for categorical encoders that go beyond
//! basic unit tests. Focuses on cross-encoder consistency, pipeline integration,
//! edge cases, serialization, and target leakage prevention.

#[cfg(test)]
mod tests {
    use crate::metrics::r2_score;
    use crate::preprocessing::encoders::{
        DropStrategy, LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder,
    };
    use crate::preprocessing::{Transformer, UnknownCategoryHandling};
    use ndarray::{array, Array1, Array2};

    const EPSILON: f64 = 1e-10;

    // ========== Cross-Encoder Consistency Tests ==========

    #[test]
    fn test_onehot_ordinal_category_discovery_consistency() {
        // Both encoders should discover the same categories
        let x = array![[3.0], [1.0], [4.0], [1.0], [5.0]];

        let mut ohe = OneHotEncoder::new();
        let mut oe = OrdinalEncoder::new();

        ohe.fit(&x).unwrap();
        oe.fit(&x).unwrap();

        let ohe_cats = ohe.categories().unwrap();
        let oe_cats = oe.categories().unwrap();

        // Both should have same number of categories
        assert_eq!(ohe_cats[0].len(), oe_cats[0].len());

        // OneHotEncoder sorts categories, OrdinalEncoder preserves order
        // But they should contain the same values
        let ohe_set: std::collections::HashSet<i64> =
            ohe_cats[0].iter().map(|&v| v as i64).collect();
        let oe_set: std::collections::HashSet<i64> = oe_cats[0].iter().map(|&v| v as i64).collect();
        assert_eq!(ohe_set, oe_set);
    }

    #[test]
    fn test_label_encoder_ordinal_encoder_equivalence() {
        // LabelEncoder on 1D should match OrdinalEncoder on reshaped 2D
        let y = array![2.0, 0.0, 1.0, 2.0, 1.0];
        let x = y.clone().insert_axis(ndarray::Axis(1));

        let mut le = LabelEncoder::new();
        let mut oe = OrdinalEncoder::new();

        let le_encoded = le.fit_transform_1d(&y).unwrap();
        let oe_encoded = oe.fit_transform(&x).unwrap();

        // Both should produce same encoding pattern
        // Note: ordering may differ (first appearance vs sorted), so check consistency
        for i in 0..y.len() {
            for j in (i + 1)..y.len() {
                // If original values are equal, encoded should be equal
                if (y[i] - y[j]).abs() < EPSILON {
                    assert!(
                        (le_encoded[i] - le_encoded[j]).abs() < EPSILON,
                        "LabelEncoder inconsistent"
                    );
                    assert!(
                        (oe_encoded[[i, 0]] - oe_encoded[[j, 0]]).abs() < EPSILON,
                        "OrdinalEncoder inconsistent"
                    );
                }
            }
        }
    }

    #[test]
    fn test_onehot_inverse_transform_roundtrip() {
        // Multiple features should roundtrip correctly
        let x = array![[0.0, 10.0], [1.0, 20.0], [2.0, 10.0], [0.0, 30.0]];

        let mut encoder = OneHotEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();
        let recovered = encoder.inverse_transform(&encoded).unwrap();

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(
                    (x[[i, j]] - recovered[[i, j]]).abs() < EPSILON,
                    "Roundtrip failed at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_ordinal_inverse_transform_roundtrip() {
        let x = array![[5.0, 100.0], [3.0, 200.0], [5.0, 300.0], [7.0, 100.0]];

        let mut encoder = OrdinalEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();
        let recovered = encoder.inverse_transform(&encoded).unwrap();

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!(
                    (x[[i, j]] - recovered[[i, j]]).abs() < EPSILON,
                    "Roundtrip failed at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    // ========== High-Cardinality Stress Tests ==========

    #[test]
    fn test_onehot_high_cardinality() {
        // 100 unique categories
        let n_categories = 100;
        let n_samples = 200;
        let mut data = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            data.push(vec![(i % n_categories) as f64]);
        }

        let x =
            Array2::from_shape_vec((n_samples, 1), data.into_iter().flatten().collect()).unwrap();

        let mut encoder = OneHotEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();

        assert_eq!(encoded.ncols(), n_categories);
        assert_eq!(encoded.nrows(), n_samples);

        // Each row should have exactly one 1.0
        for i in 0..n_samples {
            let row_sum: f64 = encoded.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < EPSILON,
                "Row {} sum is {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_ordinal_high_cardinality() {
        // 500 unique categories
        let n_categories = 500;
        let n_samples = 1000;
        let mut data = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            data.push((i % n_categories) as f64);
        }

        let x = Array2::from_shape_vec((n_samples, 1), data).unwrap();

        let mut encoder = OrdinalEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();

        // All codes should be in [0, n_categories)
        for i in 0..n_samples {
            let code = encoded[[i, 0]];
            assert!(code >= 0.0 && code < n_categories as f64);
        }
    }

    #[test]
    fn test_target_encoder_high_cardinality_smoothing() {
        // High cardinality with varying category frequencies
        // Rare categories should be pulled toward global mean
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        // Common category (0.0) appears 100 times with mean 10.0
        for _ in 0..100 {
            x_data.push(0.0);
            y_data.push(10.0);
        }
        // Rare categories (1-99) appear once each with extreme value 100.0
        for i in 1..100 {
            x_data.push(i as f64);
            y_data.push(100.0);
        }

        let n_samples = x_data.len();
        let x = Array2::from_shape_vec((n_samples, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        // High smoothing should pull rare categories toward global mean
        let mut encoder = TargetEncoder::new().with_smooth(10.0);
        encoder.fit_with_target(&x, &y).unwrap();

        let global_mean = encoder.global_mean().unwrap();

        // Encode a rare category
        let rare = array![[50.0]];
        let encoded = encoder.transform(&rare).unwrap();

        // With high smoothing, rare category encoding should be closer to global mean
        // than to its raw value of 100.0
        let dist_to_global = (encoded[[0, 0]] - global_mean).abs();
        let dist_to_raw = (encoded[[0, 0]] - 100.0).abs();
        assert!(
            dist_to_global < dist_to_raw,
            "Rare category not smoothed toward global mean"
        );
    }

    // ========== NaN Edge Cases ==========

    #[test]
    fn test_onehot_nan_in_multiple_columns() {
        // NaN should be treated as distinct category in each column
        let x = array![
            [1.0, f64::NAN],
            [f64::NAN, 2.0],
            [1.0, 2.0],
            [f64::NAN, f64::NAN]
        ];

        let mut encoder = OneHotEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();

        // Feature 0: 1.0 and NaN = 2 categories
        // Feature 1: 2.0 and NaN = 2 categories
        // Total: 4 columns
        assert_eq!(encoded.ncols(), 4);

        // Rows with same values should have same encodings
        // Row 0 and Row 2 share feature 0 value (1.0)
        assert_eq!(encoded[[0, 0]], encoded[[2, 0]]);
        assert_eq!(encoded[[0, 1]], encoded[[2, 1]]);
    }

    #[test]
    fn test_ordinal_nan_consistency() {
        // NaN values should consistently map to same code
        let x = array![[f64::NAN], [1.0], [f64::NAN], [2.0], [f64::NAN]];

        let mut encoder = OrdinalEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();

        // All NaN positions should have same code
        assert!((encoded[[0, 0]] - encoded[[2, 0]]).abs() < EPSILON);
        assert!((encoded[[2, 0]] - encoded[[4, 0]]).abs() < EPSILON);
    }

    #[test]
    fn test_label_encoder_nan_as_class() {
        // NaN should be valid as a class label
        let y = array![1.0, f64::NAN, 2.0, f64::NAN, 1.0];

        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform_1d(&y).unwrap();

        // NaN positions should have same code
        assert!((encoded[1] - encoded[3]).abs() < EPSILON);

        // Should have 3 classes: 1.0, NaN, 2.0
        assert_eq!(encoder.n_classes(), Some(3));
    }

    // ========== Drop Strategy Edge Cases ==========

    #[test]
    fn test_onehot_drop_first_with_unknown_ignore() {
        // When first category is dropped and unknown is encountered
        let mut encoder = OneHotEncoder::new()
            .with_drop(DropStrategy::First)
            .with_handle_unknown(UnknownCategoryHandling::Ignore);

        let x_train = array![[0.0], [1.0], [2.0]];
        encoder.fit(&x_train).unwrap();

        // Unknown category
        let x_test = array![[5.0]];
        let encoded = encoder.transform(&x_test).unwrap();

        // Should produce all zeros (2 columns since first dropped)
        assert_eq!(encoded.ncols(), 2);
        assert!((encoded[[0, 0]]).abs() < EPSILON);
        assert!((encoded[[0, 1]]).abs() < EPSILON);
    }

    #[test]
    fn test_onehot_drop_first_inverse_transform() {
        // Inverse transform should recover dropped category correctly
        let mut encoder = OneHotEncoder::new().with_drop(DropStrategy::First);

        let x = array![[0.0], [1.0], [2.0], [0.0]];
        let encoded = encoder.fit_transform(&x).unwrap();

        // Dropped category (0.0) results in all-zeros encoding
        // But inverse should recover it
        let recovered = encoder.inverse_transform(&encoded).unwrap();

        for i in 0..x.nrows() {
            assert!(
                (x[[i, 0]] - recovered[[i, 0]]).abs() < EPSILON,
                "Failed to recover row {}",
                i
            );
        }
    }

    #[test]
    fn test_onehot_drop_if_binary_multifeature() {
        // Mix of binary and non-binary features
        let mut encoder = OneHotEncoder::new().with_drop(DropStrategy::IfBinary);

        // Feature 0: binary (0, 1)
        // Feature 1: ternary (0, 1, 2)
        let x = array![[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]];

        let encoded = encoder.fit_transform(&x).unwrap();

        // Feature 0: binary, drop first -> 1 column
        // Feature 1: ternary, no drop -> 3 columns
        // Total: 4 columns
        assert_eq!(encoded.ncols(), 4);
    }

    // ========== TargetEncoder Leakage Prevention ==========

    #[test]
    fn test_target_encoder_cv_prevents_leakage() {
        // Create data where naive target encoding would cause severe overfitting
        let n = 100;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);

        // Each sample has unique category
        for i in 0..n {
            x_data.push(i as f64);
            y_data.push(i as f64); // Target equals category
        }

        let x = Array2::from_shape_vec((n, 1), x_data).unwrap();
        let y = Array1::from_vec(y_data);

        // Naive encoding (no CV) would give perfect prediction
        let mut encoder_naive = TargetEncoder::new().with_smooth(0.0);
        encoder_naive.fit_with_target(&x, &y).unwrap();
        let encoded_naive = encoder_naive.transform(&x).unwrap();

        // CV-based encoding should prevent this
        let mut encoder_cv = TargetEncoder::new().with_smooth(0.0).with_cv(5);
        let encoded_cv = encoder_cv.fit_transform_with_target(&x, &y).unwrap();

        // Naive encoding: encoded values should exactly match y
        let naive_r2 = r2_score(&y, &encoded_naive.column(0).to_owned()).unwrap();
        assert!(naive_r2 > 0.999, "Naive encoding should be near-perfect");

        // CV encoding: encoded values should not perfectly match y
        let cv_r2 = r2_score(&y, &encoded_cv.column(0).to_owned()).unwrap();
        assert!(
            cv_r2 < naive_r2,
            "CV encoding should be less perfect than naive"
        );
    }

    #[test]
    fn test_target_encoder_oof_different_from_fitted() {
        // The OOF values during fit_transform should differ from post-fit transform
        let x = array![[0.0], [0.0], [1.0], [1.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let mut encoder = TargetEncoder::new().with_cv(2);

        // fit_transform uses CV-based OOF encoding
        let encoded_oof = encoder.fit_transform_with_target(&x, &y).unwrap();

        // Regular transform uses full-data statistics
        let encoded_full = encoder.transform(&x).unwrap();

        // They should not be identical (CV prevents using same-fold data)
        let mut any_different = false;
        for i in 0..4 {
            if (encoded_oof[[i, 0]] - encoded_full[[i, 0]]).abs() > EPSILON {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "OOF encoding should differ from full encoding"
        );
    }

    // ========== Feature Name Propagation ==========

    #[test]
    fn test_onehot_feature_names_with_input_names() {
        let mut encoder = OneHotEncoder::new();
        let x = array![[0.0, 0.0], [1.0, 1.0]];

        encoder.fit(&x).unwrap();

        let input_names = vec!["color".to_string(), "size".to_string()];
        let output_names = encoder.get_feature_names_out(Some(&input_names));

        // Should have names for each output column
        assert!(output_names.is_some());
        let names = output_names.unwrap();
        assert_eq!(names.len(), encoder.n_features_out().unwrap());
    }

    #[test]
    fn test_target_encoder_feature_names_suffix() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0], [1.0]];
        let y = array![1.0, 2.0];

        encoder.fit_with_target(&x, &y).unwrap();

        let input_names = vec!["category".to_string()];
        let output_names = encoder.get_feature_names_out(Some(&input_names)).unwrap();

        assert_eq!(output_names.len(), 1);
        assert!(
            output_names[0].contains("target_enc"),
            "Name should indicate target encoding"
        );
    }

    // ========== Serialization Tests ==========
    // Note: Encoders use HashMap with tuple keys, which serde_json doesn't support.
    // We use bincode for serialization tests instead.

    #[test]
    fn test_onehot_serialization_roundtrip() {
        let mut encoder = OneHotEncoder::new().with_drop(DropStrategy::First);
        let x = array![[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]];

        encoder.fit(&x).unwrap();
        let original_encoded = encoder.transform(&x).unwrap();

        // Serialize and deserialize using bincode (supports tuple keys)
        let bytes = bincode::serialize(&encoder).unwrap();
        let restored: OneHotEncoder = bincode::deserialize(&bytes).unwrap();

        // Should produce same output
        let restored_encoded = restored.transform(&x).unwrap();

        assert_eq!(original_encoded.shape(), restored_encoded.shape());
        for i in 0..original_encoded.nrows() {
            for j in 0..original_encoded.ncols() {
                assert!(
                    (original_encoded[[i, j]] - restored_encoded[[i, j]]).abs() < EPSILON,
                    "Serialization mismatch at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_ordinal_serialization_roundtrip() {
        let mut encoder =
            OrdinalEncoder::new().with_handle_unknown(UnknownCategoryHandling::Ignore);
        let x = array![[5.0], [3.0], [1.0], [3.0]];

        encoder.fit(&x).unwrap();
        let original_encoded = encoder.transform(&x).unwrap();

        let bytes = bincode::serialize(&encoder).unwrap();
        let restored: OrdinalEncoder = bincode::deserialize(&bytes).unwrap();

        let restored_encoded = restored.transform(&x).unwrap();

        for i in 0..original_encoded.nrows() {
            assert!(
                (original_encoded[[i, 0]] - restored_encoded[[i, 0]]).abs() < EPSILON,
                "Serialization mismatch at row {}",
                i
            );
        }
    }

    #[test]
    fn test_label_encoder_serialization_roundtrip() {
        let mut encoder = LabelEncoder::new();
        let y = array![2.0, 0.0, 1.0, 2.0];

        encoder.fit_1d(&y).unwrap();
        let original_encoded = encoder.transform_1d(&y).unwrap();

        // LabelEncoder also has HashMap with tuple-like keys, use bincode
        let bytes = bincode::serialize(&encoder).unwrap();
        let restored: LabelEncoder = bincode::deserialize(&bytes).unwrap();

        let restored_encoded = restored.transform_1d(&y).unwrap();

        for i in 0..original_encoded.len() {
            assert!(
                (original_encoded[i] - restored_encoded[i]).abs() < EPSILON,
                "Serialization mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_target_encoder_serialization_roundtrip() {
        let mut encoder = TargetEncoder::new()
            .with_smooth(5.0)
            .with_handle_unknown(UnknownCategoryHandling::Ignore);

        let x = array![[0.0], [0.0], [1.0], [1.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        encoder.fit_with_target(&x, &y).unwrap();
        let original_encoded = encoder.transform(&x).unwrap();

        let bytes = bincode::serialize(&encoder).unwrap();
        let restored: TargetEncoder = bincode::deserialize(&bytes).unwrap();

        let restored_encoded = restored.transform(&x).unwrap();

        for i in 0..original_encoded.nrows() {
            assert!(
                (original_encoded[[i, 0]] - restored_encoded[[i, 0]]).abs() < EPSILON,
                "Serialization mismatch at row {}",
                i
            );
        }

        // Also verify configuration was preserved
        assert!((restored.smooth() - 5.0).abs() < EPSILON);
    }

    // ========== Edge Cases ==========

    #[test]
    fn test_onehot_single_category() {
        // Single category should produce single column of all 1s
        let x = array![[5.0], [5.0], [5.0]];

        let mut encoder = OneHotEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();

        assert_eq!(encoded.ncols(), 1);
        for i in 0..encoded.nrows() {
            assert!((encoded[[i, 0]] - 1.0).abs() < EPSILON);
        }
    }

    #[test]
    fn test_ordinal_single_sample() {
        // Single sample should work
        let x = array![[42.0]];

        let mut encoder = OrdinalEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();

        assert_eq!(encoded.nrows(), 1);
        assert!((encoded[[0, 0]] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_target_encoder_constant_target() {
        // Constant target should produce constant encoding (global mean)
        let x = array![[0.0], [1.0], [2.0]];
        let y = array![5.0, 5.0, 5.0];

        let mut encoder = TargetEncoder::new();
        encoder.fit_with_target(&x, &y).unwrap();
        let encoded = encoder.transform(&x).unwrap();

        // All encodings should be 5.0 (or close, with smoothing)
        for i in 0..encoded.nrows() {
            assert!(
                (encoded[[i, 0]] - 5.0).abs() < 0.1,
                "Constant target should yield constant encoding"
            );
        }
    }

    #[test]
    fn test_onehot_negative_values() {
        // Negative values should work as categories
        let x = array![[-1.0], [0.0], [1.0], [-1.0]];

        let mut encoder = OneHotEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();

        assert_eq!(encoded.ncols(), 3);

        // -1.0 samples should have same encoding
        assert_eq!(encoded.row(0), encoded.row(3));
    }

    #[test]
    fn test_ordinal_floating_point_categories() {
        // Floating point categories should work (be careful with precision)
        let x = array![[0.1], [0.2], [0.3], [0.1]];

        let mut encoder = OrdinalEncoder::new();
        let encoded = encoder.fit_transform(&x).unwrap();

        // 0.1 appears first -> code 0
        // 0.2 appears second -> code 1
        // 0.3 appears third -> code 2
        assert!((encoded[[0, 0]] - 0.0).abs() < EPSILON);
        assert!((encoded[[1, 0]] - 1.0).abs() < EPSILON);
        assert!((encoded[[2, 0]] - 2.0).abs() < EPSILON);
        assert!((encoded[[3, 0]] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_target_encoder_binary_classification_target() {
        // Binary classification target (0/1) should work
        let x = array![[0.0], [0.0], [0.0], [1.0], [1.0]];
        let y = array![0.0, 0.0, 1.0, 1.0, 1.0]; // category 0: 1/3 positive, category 1: 2/2 positive

        let mut encoder = TargetEncoder::new().with_smooth(0.0);
        encoder.fit_with_target(&x, &y).unwrap();
        let encoded = encoder.transform(&x).unwrap();

        // Category 0: mean = (0+0+1)/3 = 0.333...
        // Category 1: mean = (1+1)/2 = 1.0
        let cat0_mean = 1.0 / 3.0;
        let cat1_mean = 1.0;

        assert!(
            (encoded[[0, 0]] - cat0_mean).abs() < EPSILON,
            "Category 0 encoding incorrect"
        );
        assert!(
            (encoded[[3, 0]] - cat1_mean).abs() < EPSILON,
            "Category 1 encoding incorrect"
        );
    }

    // ========== n_features Attribute Tests ==========

    #[test]
    fn test_onehot_n_features_before_after_fit() {
        let encoder = OneHotEncoder::new();
        assert!(encoder.n_features_in().is_none());
        assert!(encoder.n_features_out().is_none());

        let mut encoder = OneHotEncoder::new();
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        encoder.fit(&x).unwrap();

        assert_eq!(encoder.n_features_in(), Some(2));
        // 3 categories per feature = 6 output features
        assert_eq!(encoder.n_features_out(), Some(6));
    }

    #[test]
    fn test_ordinal_n_features_preserves_count() {
        let mut encoder = OrdinalEncoder::new();
        let x = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        encoder.fit(&x).unwrap();

        // OrdinalEncoder preserves feature count
        assert_eq!(encoder.n_features_in(), Some(3));
        assert_eq!(encoder.n_features_out(), Some(3));
    }

    #[test]
    fn test_target_encoder_n_features_preserves_count() {
        let mut encoder = TargetEncoder::new();
        let x = array![[0.0, 0.0], [1.0, 1.0]];
        let y = array![1.0, 2.0];
        encoder.fit_with_target(&x, &y).unwrap();

        // TargetEncoder preserves feature count
        assert_eq!(encoder.n_features_in(), Some(2));
        assert_eq!(encoder.n_features_out(), Some(2));
    }

    // ========== HistGradientBoosting Native Categorical Tests ==========

    #[test]
    fn test_hist_boosting_classifier_categorical_fits_and_predicts() {
        use crate::models::hist_boosting::HistGradientBoostingClassifier;
        use crate::models::Model;

        // Feature 0: categorical (3 categories), Feature 1: continuous
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                0.0, 1.0, 0.0, 1.5, 0.0, 0.8, 0.0, 1.2, 0.0, 1.1, 1.0, 3.0, 1.0, 3.5, 1.0, 2.8,
                1.0, 3.2, 1.0, 3.1, 2.0, 5.0, 2.0, 5.5, 2.0, 4.8, 2.0, 5.2, 2.0, 5.1, 0.0, 1.3,
                1.0, 2.9, 2.0, 5.3, 0.0, 0.9, 1.0, 3.3,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            1.0, 0.0, 1.0,
        ]);

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(50)
            .with_learning_rate(0.1)
            .with_categorical_features(vec![0])
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();
        assert!(clf.is_fitted());

        let predictions = clf.predict(&x).unwrap();
        assert_eq!(predictions.len(), 20);

        // All predictions should be valid class labels
        for p in predictions.iter() {
            assert!(*p == 0.0 || *p == 1.0, "Invalid prediction: {}", p);
        }
    }

    #[test]
    fn test_hist_boosting_regressor_categorical_fits_and_predicts() {
        use crate::models::hist_boosting::HistGradientBoostingRegressor;
        use crate::models::Model;

        // Feature 0: categorical (3 categories), Feature 1: continuous
        // Target depends on category: cat 0 -> ~10, cat 1 -> ~20, cat 2 -> ~30
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 1.0, 0.0, 1.1, 0.0, 0.9, 0.0, 1.2, 1.0, 2.0, 1.0, 2.1, 1.0, 1.9, 1.0, 2.2,
                2.0, 3.0, 2.0, 3.1, 2.0, 2.9, 2.0, 3.2,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            10.0, 10.5, 9.5, 11.0, 20.0, 20.5, 19.5, 21.0, 30.0, 30.5, 29.5, 31.0,
        ]);

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(100)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(2)
            .with_categorical_features(vec![0])
            .with_random_state(42);

        reg.fit(&x, &y).unwrap();
        assert!(reg.is_fitted());

        let predictions = reg.predict(&x).unwrap();
        assert_eq!(predictions.len(), 12);

        // RMSE should be reasonable (< 10.0 given clear category-target relationship)
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / 12.0;
        let rmse = mse.sqrt();
        assert!(rmse < 10.0, "RMSE was {}", rmse);
    }

    #[test]
    fn test_hist_boosting_classifier_categorical_reasonable_accuracy() {
        use crate::models::hist_boosting::HistGradientBoostingClassifier;
        use crate::models::Model;

        // Data where the categorical feature is the primary signal
        // Category 0 -> class 0, Category 1 -> class 1
        let x = Array2::from_shape_vec(
            (24, 2),
            vec![
                0.0, 0.5, 0.0, 0.6, 0.0, 0.4, 0.0, 0.7, 0.0, 0.3, 0.0, 0.55, 0.0, 0.45, 0.0, 0.65,
                0.0, 0.35, 0.0, 0.52, 0.0, 0.48, 0.0, 0.58, 1.0, 0.5, 1.0, 0.6, 1.0, 0.4, 1.0, 0.7,
                1.0, 0.3, 1.0, 0.55, 1.0, 0.45, 1.0, 0.65, 1.0, 0.35, 1.0, 0.52, 1.0, 0.48, 1.0,
                0.58,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);

        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(50)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(2)
            .with_categorical_features(vec![0])
            .with_random_state(42);

        clf.fit(&x, &y).unwrap();
        let predictions = clf.predict(&x).unwrap();

        let accuracy: f64 = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count() as f64
            / y.len() as f64;

        // With a perfectly separable categorical feature, accuracy should be high
        assert!(
            accuracy >= 0.75,
            "Expected high accuracy with categorical signal, got {}",
            accuracy
        );
    }

    #[test]
    fn test_hist_boosting_with_vs_without_categorical_declaration() {
        use crate::models::hist_boosting::HistGradientBoostingRegressor;
        use crate::models::Model;

        // Data where category strongly determines target
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 1.0, 0.0, 1.1, 0.0, 0.9, 0.0, 1.2, 1.0, 2.0, 1.0, 2.1, 1.0, 1.9, 1.0, 2.2,
                2.0, 3.0, 2.0, 3.1, 2.0, 2.9, 2.0, 3.2,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            10.0, 10.5, 9.5, 11.0, 20.0, 20.5, 19.5, 21.0, 30.0, 30.5, 29.5, 31.0,
        ]);

        // With categorical declaration
        let mut reg_cat = HistGradientBoostingRegressor::new()
            .with_max_iter(100)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(2)
            .with_categorical_features(vec![0])
            .with_random_state(42);
        reg_cat.fit(&x, &y).unwrap();
        let pred_cat = reg_cat.predict(&x).unwrap();

        // Without categorical declaration
        let mut reg_no_cat = HistGradientBoostingRegressor::new()
            .with_max_iter(100)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(2)
            .with_categorical_features(vec![])
            .with_random_state(42);
        reg_no_cat.fit(&x, &y).unwrap();
        let pred_no_cat = reg_no_cat.predict(&x).unwrap();

        // Both should produce reasonable predictions (they learn the same data)
        // but they may differ in approach
        let mse_cat: f64 = pred_cat
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        let mse_no_cat: f64 = pred_no_cat
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;

        // Both should fit reasonably
        assert!(
            mse_cat.sqrt() < 10.0,
            "Categorical RMSE too high: {}",
            mse_cat.sqrt()
        );
        assert!(
            mse_no_cat.sqrt() < 10.0,
            "Non-categorical RMSE too high: {}",
            mse_no_cat.sqrt()
        );
    }

    // ========== ColumnTransformer with Different Encoders ==========

    #[test]
    fn test_column_transformer_onehot_and_standard_scaler() {
        use crate::pipeline::{ColumnSelector, ColumnTransformer, RemainderHandling};
        use crate::preprocessing::scalers::StandardScaler;

        // 4 features: col 0 = categorical, cols 1-3 = numeric
        let x = array![
            [0.0, 10.0, 100.0, 1000.0],
            [1.0, 20.0, 200.0, 2000.0],
            [2.0, 30.0, 300.0, 3000.0],
            [0.0, 40.0, 400.0, 4000.0],
            [1.0, 50.0, 500.0, 5000.0],
        ];

        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "encoder",
                OneHotEncoder::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "scaler",
                StandardScaler::new(),
                ColumnSelector::indices([1, 2, 3]),
            )
            .with_remainder(RemainderHandling::Drop);

        let transformed = ct.fit_transform(&x).unwrap();

        // OneHot on col 0 with 3 categories -> 3 cols, StandardScaler on 3 cols -> 3 cols
        // Total: 6 columns
        assert_eq!(transformed.ncols(), 6);
        assert_eq!(transformed.nrows(), 5);

        // First 3 columns are one-hot: each row should have exactly one 1.0
        for i in 0..5 {
            let ohe_sum: f64 = (0..3).map(|j| transformed[[i, j]]).sum();
            assert!(
                (ohe_sum - 1.0).abs() < EPSILON,
                "Row {} one-hot sum is {}",
                i,
                ohe_sum
            );
        }

        // Last 3 columns are standard-scaled: mean ~0, std ~1
        for col in 3..6 {
            let col_mean: f64 = (0..5).map(|i| transformed[[i, col]]).sum::<f64>() / 5.0;
            assert!(
                col_mean.abs() < 1e-6,
                "Scaled column {} has mean {}",
                col,
                col_mean
            );
        }
    }

    #[test]
    fn test_column_transformer_onehot_and_passthrough() {
        use crate::pipeline::{ColumnSelector, ColumnTransformer, RemainderHandling};

        // col 0 = categorical, cols 1-2 = numeric (passthrough via remainder)
        let x = array![[0.0, 10.0, 100.0], [1.0, 20.0, 200.0], [2.0, 30.0, 300.0],];

        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "encoder",
                OneHotEncoder::new(),
                ColumnSelector::indices([0]),
            )
            .with_remainder(RemainderHandling::Passthrough);

        let transformed = ct.fit_transform(&x).unwrap();

        // OneHot on col 0: 3 categories -> 3 cols
        // Passthrough cols 1,2 -> 2 cols
        // Total: 5 columns
        assert_eq!(transformed.ncols(), 5);
        assert_eq!(transformed.nrows(), 3);

        // Passthrough columns should be unchanged (last 2 cols)
        assert!((transformed[[0, 3]] - 10.0).abs() < EPSILON);
        assert!((transformed[[0, 4]] - 100.0).abs() < EPSILON);
        assert!((transformed[[1, 3]] - 20.0).abs() < EPSILON);
        assert!((transformed[[1, 4]] - 200.0).abs() < EPSILON);
    }

    #[test]
    fn test_column_transformer_output_shape_correctness() {
        use crate::pipeline::{ColumnSelector, ColumnTransformer, RemainderHandling};
        use crate::preprocessing::scalers::MinMaxScaler;

        // 5 features: cols 0,1 = categorical (binary), cols 2,3,4 = numeric
        let x = array![
            [0.0, 0.0, 1.0, 2.0, 3.0],
            [1.0, 1.0, 4.0, 5.0, 6.0],
            [0.0, 1.0, 7.0, 8.0, 9.0],
            [1.0, 0.0, 10.0, 11.0, 12.0],
        ];

        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "onehot",
                OneHotEncoder::new(),
                ColumnSelector::indices([0, 1]),
            )
            .add_transformer(
                "minmax",
                MinMaxScaler::new(),
                ColumnSelector::indices([2, 3, 4]),
            )
            .with_remainder(RemainderHandling::Drop);

        let transformed = ct.fit_transform(&x).unwrap();

        // Col 0: 2 categories -> 2 OHE cols
        // Col 1: 2 categories -> 2 OHE cols
        // Cols 2,3,4: 3 MinMax cols
        // Total: 7 columns
        assert_eq!(transformed.ncols(), 7);
        assert_eq!(transformed.nrows(), 4);

        // MinMax scaled values should be in [0, 1]
        for i in 0..4 {
            for j in 4..7 {
                assert!(
                    transformed[[i, j]] >= -EPSILON && transformed[[i, j]] <= 1.0 + EPSILON,
                    "MinMax value out of range at [{}, {}]: {}",
                    i,
                    j,
                    transformed[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_column_transformer_transform_after_fit() {
        use crate::pipeline::{ColumnSelector, ColumnTransformer, RemainderHandling};
        use crate::preprocessing::scalers::StandardScaler;

        // Fit on training data, transform test data
        let x_train = array![[0.0, 10.0], [1.0, 20.0], [2.0, 30.0], [0.0, 40.0],];
        let x_test = array![[1.0, 25.0], [2.0, 35.0],];

        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "encoder",
                OneHotEncoder::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "scaler",
                StandardScaler::new(),
                ColumnSelector::indices([1]),
            )
            .with_remainder(RemainderHandling::Drop);

        ct.fit(&x_train).unwrap();
        let transformed = ct.transform(&x_test).unwrap();

        // 3 OHE cols + 1 scaled col = 4 columns
        assert_eq!(transformed.ncols(), 4);
        assert_eq!(transformed.nrows(), 2);

        // First row: category 1 -> [0, 1, 0]
        assert!((transformed[[0, 1]] - 1.0).abs() < EPSILON);
        assert!((transformed[[0, 0]]).abs() < EPSILON);
        assert!((transformed[[0, 2]]).abs() < EPSILON);
    }

    // ========== Mixed Categorical/Numeric Pipeline Tests ==========

    #[test]
    fn test_mixed_pipeline_column_transformer_to_model() {
        use crate::models::hist_boosting::HistGradientBoostingClassifier;
        use crate::models::Model;
        use crate::pipeline::{ColumnSelector, ColumnTransformer, RemainderHandling};
        use crate::preprocessing::scalers::StandardScaler;

        // Simulate categorical + numeric features
        // Col 0: categorical (0, 1, 2), Cols 1-2: numeric
        let x = array![
            [0.0, 1.0, 0.5],
            [0.0, 1.5, 0.6],
            [0.0, 0.8, 0.4],
            [0.0, 1.2, 0.55],
            [1.0, 3.0, 1.5],
            [1.0, 3.5, 1.6],
            [1.0, 2.8, 1.4],
            [1.0, 3.2, 1.55],
            [2.0, 5.0, 2.5],
            [2.0, 5.5, 2.6],
            [2.0, 4.8, 2.4],
            [2.0, 5.2, 2.55],
        ];
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);

        // Step 1: encode categoricals + scale numerics
        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "encoder",
                OneHotEncoder::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "scaler",
                StandardScaler::new(),
                ColumnSelector::indices([1, 2]),
            )
            .with_remainder(RemainderHandling::Drop);

        let x_transformed = ct.fit_transform(&x).unwrap();

        // 3 OHE cols + 2 scaled cols = 5 features
        assert_eq!(x_transformed.ncols(), 5);

        // Step 2: fit model on transformed data
        let mut clf = HistGradientBoostingClassifier::new()
            .with_max_iter(50)
            .with_learning_rate(0.1)
            .with_min_samples_leaf(2)
            .with_random_state(42);

        clf.fit(&x_transformed, &y).unwrap();
        let predictions = clf.predict(&x_transformed).unwrap();
        assert_eq!(predictions.len(), 12);

        let accuracy: f64 = predictions
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count() as f64
            / y.len() as f64;

        // Should do better than random (>50%) on training data
        assert!(accuracy >= 0.6, "Pipeline accuracy too low: {}", accuracy);
    }

    #[test]
    fn test_mixed_pipeline_produces_valid_predictions() {
        use crate::models::hist_boosting::HistGradientBoostingRegressor;
        use crate::models::Model;
        use crate::pipeline::{ColumnSelector, ColumnTransformer, RemainderHandling};
        use crate::preprocessing::scalers::MinMaxScaler;

        // Col 0: categorical, Cols 1-2: numeric
        let x = array![
            [0.0, 10.0, 100.0],
            [0.0, 11.0, 110.0],
            [0.0, 9.0, 90.0],
            [1.0, 20.0, 200.0],
            [1.0, 21.0, 210.0],
            [1.0, 19.0, 190.0],
            [2.0, 30.0, 300.0],
            [2.0, 31.0, 310.0],
            [2.0, 29.0, 290.0],
        ];
        let y = Array1::from_vec(vec![5.0, 5.5, 4.5, 15.0, 15.5, 14.5, 25.0, 25.5, 24.5]);

        // Encode + scale
        let mut ct = ColumnTransformer::new()
            .add_transformer(
                "encoder",
                OneHotEncoder::new(),
                ColumnSelector::indices([0]),
            )
            .add_transformer(
                "minmax",
                MinMaxScaler::new(),
                ColumnSelector::indices([1, 2]),
            )
            .with_remainder(RemainderHandling::Drop);

        let x_transformed = ct.fit_transform(&x).unwrap();

        let mut reg = HistGradientBoostingRegressor::new()
            .with_max_iter(100)
            .with_learning_rate(0.1)
            .with_random_state(42);

        reg.fit(&x_transformed, &y).unwrap();
        let predictions = reg.predict(&x_transformed).unwrap();

        // All predictions should be finite
        for p in predictions.iter() {
            assert!(p.is_finite(), "Non-finite prediction: {}", p);
        }

        // Predictions should be in a reasonable range (not wildly off)
        for p in predictions.iter() {
            assert!(*p > -50.0 && *p < 80.0, "Prediction out of range: {}", p);
        }

        // RMSE should be reasonable
        let mse: f64 = predictions
            .iter()
            .zip(y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / y.len() as f64;
        assert!(mse.sqrt() < 10.0, "RMSE too high: {}", mse.sqrt());
    }

    #[test]
    fn test_mixed_pipeline_multiple_encoder_types() {
        use crate::pipeline::{ColumnSelector, ColumnTransformer, RemainderHandling};
        use crate::preprocessing::scalers::StandardScaler;

        // Col 0: to be one-hot encoded
        // Col 1: numeric, to be scaled
        // Col 2: numeric, passthrough
        let x = array![
            [0.0, 10.0, 1.0],
            [1.0, 20.0, 2.0],
            [2.0, 30.0, 3.0],
            [0.0, 40.0, 4.0],
            [1.0, 50.0, 5.0],
        ];

        let mut ct = ColumnTransformer::new()
            .add_transformer("onehot", OneHotEncoder::new(), ColumnSelector::indices([0]))
            .add_transformer(
                "scaler",
                StandardScaler::new(),
                ColumnSelector::indices([1]),
            )
            .with_remainder(RemainderHandling::Passthrough);

        let transformed = ct.fit_transform(&x).unwrap();

        // 3 OHE cols + 1 scaled col + 1 passthrough col = 5
        assert_eq!(transformed.ncols(), 5);
        assert_eq!(transformed.nrows(), 5);

        // One-hot columns sum to 1 per row
        for i in 0..5 {
            let ohe_sum: f64 = (0..3).map(|j| transformed[[i, j]]).sum();
            assert!(
                (ohe_sum - 1.0).abs() < EPSILON,
                "Row {} one-hot sum: {}",
                i,
                ohe_sum
            );
        }

        // Scaled column has mean ~0
        let scaled_mean: f64 = (0..5).map(|i| transformed[[i, 3]]).sum::<f64>() / 5.0;
        assert!(scaled_mean.abs() < 1e-6, "Scaled mean: {}", scaled_mean);

        // Passthrough column preserves original values
        for i in 0..5 {
            assert!(
                (transformed[[i, 4]] - x[[i, 2]]).abs() < EPSILON,
                "Passthrough mismatch at row {}",
                i
            );
        }
    }
}
