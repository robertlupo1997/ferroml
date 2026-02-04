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
}
