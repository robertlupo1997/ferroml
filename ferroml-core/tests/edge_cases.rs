//! Edge case and sparse pipeline tests

mod edge_case_matrix {
    //! Edge Case Matrix — systematic edge case tests for every model.
    //!
    //! Phase X.4 of Plan X: 13 scenarios per model, covering single-sample,
    //! high-dimensional, degenerate inputs, NaN/Inf rejection, extreme values,
    //! class imbalance, and multicollinearity.

    use ferroml_core::clustering::{ClusteringModel, KMeans, DBSCAN};
    use ferroml_core::decomposition::PCA;
    use ferroml_core::models::{
        BernoulliNB, DecisionTreeClassifier, DecisionTreeRegressor, ElasticNet, GaussianNB,
        GradientBoostingClassifier, GradientBoostingRegressor, KNeighborsClassifier,
        KNeighborsRegressor, LassoRegression, LinearRegression, LinearSVC, LinearSVR,
        LogisticRegression, Model, MultinomialNB, RandomForestClassifier, RandomForestRegressor,
        RidgeRegression, SGDClassifier, SGDRegressor, SVC, SVR,
    };
    use ferroml_core::preprocessing::scalers::{MinMaxScaler, RobustScaler, StandardScaler};
    use ferroml_core::preprocessing::Transformer;
    use ndarray::{Array1, Array2};

    // =============================================================================
    // Helper data generators
    // =============================================================================

    /// Single sample: n=1, p=2
    fn gen_single_sample() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let y_reg = Array1::from_vec(vec![3.0]);
        let y_cls = Array1::from_vec(vec![1.0]);
        (x, y_reg, y_cls)
    }

    /// Single feature: n=20, p=1
    fn gen_single_feature() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n = 20;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y_reg = Array1::from_iter((0..n).map(|i| i as f64 * 2.0 + 1.0));
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_reg, y_cls)
    }

    /// High dimensional: p > n (p=50, n=10)
    fn gen_high_dimensional() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n = 10;
        let p = 50;
        let x = Array2::from_shape_fn((n, p), |(i, j)| (i * p + j) as f64 * 0.01);
        let y_reg = Array1::from_iter((0..n).map(|i| i as f64));
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_reg, y_cls)
    }

    /// Zero variance: all features constant
    fn gen_zero_variance() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n = 20;
        let x = Array2::from_elem((n, 2), 5.0);
        let y_reg = Array1::from_iter((0..n).map(|i| i as f64));
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_reg, y_cls)
    }

    /// Identical targets: y all same value
    fn gen_identical_targets() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n = 20;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| (i + j) as f64);
        let y_reg = Array1::from_elem(n, 42.0);
        let y_cls = Array1::from_elem(n, 1.0);
        (x, y_reg, y_cls)
    }

    /// NaN in features
    fn gen_nan_features() -> Array2<f64> {
        Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap()
    }

    /// Inf in features
    fn gen_inf_features() -> Array2<f64> {
        Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::INFINITY, 4.0, 5.0, 6.0]).unwrap()
    }

    /// NaN in targets
    fn gen_nan_targets() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((3, 2), |(i, j)| (i + j) as f64);
        let y = Array1::from_vec(vec![1.0, f64::NAN, 3.0]);
        (x, y)
    }

    /// Empty input (n=0)
    fn gen_empty_input() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((0, 2), |_| 0.0);
        let y = Array1::from_vec(vec![]);
        (x, y)
    }

    /// Extreme large values
    fn gen_extreme_large() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n = 20;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i + j) as f64) * 1e15);
        let y_reg = Array1::from_iter((0..n).map(|i| i as f64 * 1e15));
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_reg, y_cls)
    }

    /// Extreme small values
    fn gen_extreme_small() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n = 20;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i + j) as f64 + 1.0) * 1e-15);
        let y_reg = Array1::from_iter((0..n).map(|i| (i as f64 + 1.0) * 1e-15));
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_reg, y_cls)
    }

    /// Class imbalance: 99:1 ratio
    fn gen_class_imbalance() -> (Array2<f64>, Array1<f64>) {
        let n = 100;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| (i * 2 + j) as f64 * 0.1);
        let y = Array1::from_iter((0..n).map(|i| if i < 99 { 0.0 } else { 1.0 }));
        (x, y)
    }

    /// Multicollinear features (duplicate column)
    fn gen_multicollinear() -> (Array2<f64>, Array1<f64>) {
        let n = 30;
        let col: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut data = Vec::with_capacity(n * 3);
        for &v in &col {
            data.push(v);
            data.push(v); // duplicate
            data.push(v * 2.0 + 1.0); // linearly dependent
        }
        let x = Array2::from_shape_vec((n, 3), data).unwrap();
        let y = Array1::from_iter((0..n).map(|i| i as f64 * 0.5));
        (x, y)
    }

    /// Normal training data for fitting models before predict-time edge cases
    fn gen_normal_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n = 50;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64 * 0.1);
        let y_reg = Array1::from_iter((0..n).map(|i| i as f64 * 0.5 + 1.0));
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_reg, y_cls)
    }

    /// Normal training data with non-negative values (for MultinomialNB)
    fn gen_normal_data_nonneg() -> (Array2<f64>, Array1<f64>) {
        let n = 50;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64 * 0.1 + 1.0);
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_cls)
    }

    /// Non-negative single feature data (for MultinomialNB)
    fn gen_single_feature_nonneg() -> (Array2<f64>, Array1<f64>) {
        let n = 20;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 + 1.0);
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_cls)
    }

    /// Binary data for BernoulliNB
    fn gen_binary_data() -> (Array2<f64>, Array1<f64>) {
        let n = 40;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| if (i + j) % 2 == 0 { 1.0 } else { 0.0 });
        let y_cls = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
        (x, y_cls)
    }

    // =============================================================================
    // Macro for supervised regressor edge cases
    // =============================================================================

    macro_rules! regressor_edge_cases {
        ($mod_name:ident, $model_expr:expr) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn single_sample_fit() {
                    let (x, y, _) = gen_single_sample();
                    let mut model = $model_expr;
                    // Single sample: may succeed or return a clean error
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok(), "predict after single-sample fit should work");
                        assert_eq!(pred.unwrap().len(), 1);
                    }
                    // Either way, should not panic
                }

                #[test]
                fn single_feature() {
                    let (x, y, _) = gen_single_feature();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x).expect("predict should work");
                        assert_eq!(pred.len(), x.nrows());
                        // Predictions should be finite
                        for &v in pred.iter() {
                            assert!(v.is_finite(), "prediction should be finite, got {v}");
                        }
                    }
                }

                #[test]
                fn high_dimensional() {
                    let (x, y, _) = gen_high_dimensional();
                    let mut model = $model_expr;
                    // p > n: may succeed (tree/forest) or error (linear)
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }

                #[test]
                fn zero_variance_features() {
                    let (x, y, _) = gen_zero_variance();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    // May fail for some models (e.g., linear), should not panic
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }

                #[test]
                fn identical_targets() {
                    let (x, y, _) = gen_identical_targets();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x).expect("predict should work");
                        // All predictions should be close to the constant target
                        for &v in pred.iter() {
                            assert!(v.is_finite(), "prediction should be finite, got {v}");
                        }
                    }
                }

                #[test]
                fn nan_input_rejected() {
                    let x_nan = gen_nan_features();
                    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
                    let mut model = $model_expr;
                    let result = model.fit(&x_nan, &y);
                    assert!(result.is_err(), "NaN features should be rejected");
                }

                #[test]
                fn inf_input_rejected() {
                    let x_inf = gen_inf_features();
                    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
                    let mut model = $model_expr;
                    let result = model.fit(&x_inf, &y);
                    assert!(result.is_err(), "Inf features should be rejected");
                }

                #[test]
                fn nan_target_rejected() {
                    let (x, y) = gen_nan_targets();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    assert!(result.is_err(), "NaN targets should be rejected");
                }

                #[test]
                fn empty_input_rejected() {
                    let (x, y) = gen_empty_input();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    assert!(result.is_err(), "empty input should be rejected");
                }

                #[test]
                fn extreme_large_values() {
                    let (x, y, _) = gen_extreme_large();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    // Should not panic; may succeed or error cleanly
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        if let Ok(pred) = pred {
                            for &v in pred.iter() {
                                assert!(!v.is_nan(), "prediction should not be NaN");
                            }
                        }
                    }
                }

                #[test]
                fn extreme_small_values() {
                    let (x, y, _) = gen_extreme_small();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        if let Ok(pred) = pred {
                            for &v in pred.iter() {
                                assert!(!v.is_nan(), "prediction should not be NaN");
                            }
                        }
                    }
                }

                #[test]
                fn multicollinear_features() {
                    let (x, y) = gen_multicollinear();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    // Multicollinear: should not panic, may degrade gracefully
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }
            }
        };
    }

    // =============================================================================
    // Macro for supervised classifier edge cases
    // =============================================================================

    macro_rules! classifier_edge_cases {
        ($mod_name:ident, $model_expr:expr) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn single_sample_fit() {
                    let (x, _, y) = gen_single_sample();
                    let mut model = $model_expr;
                    // Single sample with single class: may error
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                        assert_eq!(pred.unwrap().len(), 1);
                    }
                }

                #[test]
                fn single_feature() {
                    let (x, _, y) = gen_single_feature();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x).expect("predict should work");
                        assert_eq!(pred.len(), x.nrows());
                        // All predictions should be valid class labels
                        for &v in pred.iter() {
                            assert!(v == 0.0 || v == 1.0, "expected 0 or 1, got {v}");
                        }
                    }
                }

                #[test]
                fn high_dimensional() {
                    let (x, _, y) = gen_high_dimensional();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }

                #[test]
                fn zero_variance_features() {
                    let (x, _, y) = gen_zero_variance();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }

                #[test]
                fn identical_targets() {
                    let (x, _, y) = gen_identical_targets();
                    let mut model = $model_expr;
                    // All same class: should work, predicting the single class
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x).expect("predict should work");
                        for &v in pred.iter() {
                            assert!(
                                (v - 1.0).abs() < 1e-10,
                                "with single class=1.0, expected 1.0 got {v}"
                            );
                        }
                    }
                }

                #[test]
                fn nan_input_rejected() {
                    let x_nan = gen_nan_features();
                    let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);
                    let mut model = $model_expr;
                    let result = model.fit(&x_nan, &y);
                    assert!(result.is_err(), "NaN features should be rejected");
                }

                #[test]
                fn inf_input_rejected() {
                    let x_inf = gen_inf_features();
                    let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);
                    let mut model = $model_expr;
                    let result = model.fit(&x_inf, &y);
                    assert!(result.is_err(), "Inf features should be rejected");
                }

                #[test]
                fn nan_target_rejected() {
                    let (x, y) = gen_nan_targets();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    assert!(result.is_err(), "NaN targets should be rejected");
                }

                #[test]
                fn empty_input_rejected() {
                    let (x, y) = gen_empty_input();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    assert!(result.is_err(), "empty input should be rejected");
                }

                #[test]
                fn extreme_large_values() {
                    let (x, _, y) = gen_extreme_large();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        if let Ok(pred) = pred {
                            for &v in pred.iter() {
                                assert!(v.is_finite(), "prediction should be finite, got {v}");
                            }
                        }
                    }
                }

                #[test]
                fn extreme_small_values() {
                    let (x, _, y) = gen_extreme_small();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        if let Ok(pred) = pred {
                            for &v in pred.iter() {
                                assert!(v.is_finite(), "prediction should be finite, got {v}");
                            }
                        }
                    }
                }

                #[test]
                fn class_imbalance_99_1() {
                    let (x, y) = gen_class_imbalance();
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    // Should handle imbalanced data without panic
                    if result.is_ok() {
                        let pred = model.predict(&x).expect("predict should work");
                        assert_eq!(pred.len(), x.nrows());
                        for &v in pred.iter() {
                            assert!(v == 0.0 || v == 1.0, "expected valid class label, got {v}");
                        }
                    }
                }

                #[test]
                fn multicollinear_features() {
                    let (x, _) = gen_multicollinear();
                    let y = Array1::from_iter((0..x.nrows()).map(|i| {
                        if i < x.nrows() / 2 {
                            0.0
                        } else {
                            1.0
                        }
                    }));
                    let mut model = $model_expr;
                    let result = model.fit(&x, &y);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }
            }
        };
    }

    // =============================================================================
    // Macro for clustering model edge cases
    // =============================================================================

    macro_rules! clustering_edge_cases {
        ($mod_name:ident, $model_expr:expr) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn single_sample_fit() {
                    let (x, _, _) = gen_single_sample();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    // Single sample: may fail for k>1 or succeed
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        if let Ok(pred) = pred {
                            assert_eq!(pred.len(), 1);
                        }
                    }
                }

                #[test]
                fn single_feature() {
                    let (x, _, _) = gen_single_feature();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    if result.is_ok() {
                        let pred = model.predict(&x).expect("predict should work");
                        assert_eq!(pred.len(), x.nrows());
                    }
                }

                #[test]
                fn high_dimensional() {
                    let (x, _, _) = gen_high_dimensional();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }

                #[test]
                fn zero_variance_features() {
                    let (x, _, _) = gen_zero_variance();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    // All identical points: should not panic
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }

                #[test]
                fn nan_input_rejected() {
                    let x_nan = gen_nan_features();
                    let mut model = $model_expr;
                    let result = model.fit(&x_nan);
                    assert!(result.is_err(), "NaN features should be rejected");
                }

                #[test]
                fn inf_input_rejected() {
                    let x_inf = gen_inf_features();
                    let mut model = $model_expr;
                    let result = model.fit(&x_inf);
                    assert!(result.is_err(), "Inf features should be rejected");
                }

                #[test]
                fn empty_input_rejected() {
                    let x = Array2::from_shape_fn((0, 2), |_| 0.0);
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    assert!(result.is_err(), "empty input should be rejected");
                }

                #[test]
                fn extreme_large_values() {
                    let (x, _, _) = gen_extreme_large();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    // Should not panic
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        if let Ok(pred) = pred {
                            for &v in pred.iter() {
                                assert!(v >= -1, "cluster label should be >= -1, got {v}");
                            }
                        }
                    }
                }

                #[test]
                fn extreme_small_values() {
                    let (x, _, _) = gen_extreme_small();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    if result.is_ok() {
                        let pred = model.predict(&x);
                        assert!(pred.is_ok());
                    }
                }
            }
        };
    }

    // =============================================================================
    // Macro for transformer edge cases
    // =============================================================================

    macro_rules! transformer_edge_cases {
        ($mod_name:ident, $model_expr:expr) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn single_sample_fit() {
                    let (x, _, _) = gen_single_sample();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    if result.is_ok() {
                        let transformed = model.transform(&x);
                        assert!(transformed.is_ok());
                        assert_eq!(transformed.unwrap().nrows(), 1);
                    }
                }

                #[test]
                fn single_feature() {
                    let (x, _, _) = gen_single_feature();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    if result.is_ok() {
                        let transformed = model.transform(&x).expect("transform should work");
                        assert_eq!(transformed.nrows(), x.nrows());
                    }
                }

                #[test]
                fn high_dimensional() {
                    let (x, _, _) = gen_high_dimensional();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    if result.is_ok() {
                        let transformed = model.transform(&x);
                        assert!(transformed.is_ok());
                    }
                }

                #[test]
                fn zero_variance_features() {
                    let (x, _, _) = gen_zero_variance();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    // Zero variance: scalers may produce NaN/zeros, should not panic
                    if result.is_ok() {
                        let transformed = model.transform(&x);
                        assert!(transformed.is_ok());
                    }
                }

                #[test]
                fn nan_input_rejected() {
                    let x_nan = gen_nan_features();
                    let mut model = $model_expr;
                    let result = model.fit(&x_nan);
                    assert!(result.is_err(), "NaN features should be rejected");
                }

                #[test]
                fn inf_input_rejected() {
                    let x_inf = gen_inf_features();
                    let mut model = $model_expr;
                    let result = model.fit(&x_inf);
                    assert!(result.is_err(), "Inf features should be rejected");
                }

                #[test]
                fn empty_input_rejected() {
                    let x = Array2::from_shape_fn((0, 2), |_| 0.0);
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    assert!(result.is_err(), "empty input should be rejected");
                }

                #[test]
                fn extreme_large_values() {
                    let (x, _, _) = gen_extreme_large();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    if result.is_ok() {
                        let transformed = model.transform(&x);
                        if let Ok(t) = transformed {
                            for &v in t.iter() {
                                assert!(!v.is_nan(), "transformed value should not be NaN");
                            }
                        }
                    }
                }

                #[test]
                fn extreme_small_values() {
                    let (x, _, _) = gen_extreme_small();
                    let mut model = $model_expr;
                    let result = model.fit(&x);
                    if result.is_ok() {
                        let transformed = model.transform(&x);
                        if let Ok(t) = transformed {
                            for &v in t.iter() {
                                assert!(!v.is_nan(), "transformed value should not be NaN");
                            }
                        }
                    }
                }

                #[test]
                fn fit_transform_consistent() {
                    let n = 30;
                    let x = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64 * 0.5);
                    let mut model1 = $model_expr;
                    let mut model2 = $model_expr;

                    model1.fit(&x).expect("fit should work");
                    let t1 = model1.transform(&x).expect("transform should work");
                    let t2 = model2.fit_transform(&x).expect("fit_transform should work");

                    // fit + transform should equal fit_transform
                    for (a, b) in t1.iter().zip(t2.iter()) {
                        assert!(
                            (a - b).abs() < 1e-10,
                            "fit+transform vs fit_transform mismatch: {a} vs {b}"
                        );
                    }
                }
            }
        };
    }

    // =============================================================================
    // Regressor edge case instantiations
    // =============================================================================

    regressor_edge_cases!(linear_regression_edge, LinearRegression::new());
    regressor_edge_cases!(ridge_edge, RidgeRegression::new(1.0));
    regressor_edge_cases!(lasso_edge, LassoRegression::new(0.1));
    regressor_edge_cases!(elastic_net_edge, ElasticNet::new(0.1, 0.5));
    regressor_edge_cases!(
        decision_tree_regressor_edge,
        DecisionTreeRegressor::new().with_max_depth(Some(10))
    );
    regressor_edge_cases!(
        random_forest_regressor_edge,
        RandomForestRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42)
    );
    regressor_edge_cases!(
        gradient_boosting_regressor_edge,
        GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(3))
    );
    regressor_edge_cases!(kneighbors_regressor_edge, KNeighborsRegressor::new(3));
    regressor_edge_cases!(svr_edge, SVR::new());
    regressor_edge_cases!(linear_svr_edge, LinearSVR::new());
    regressor_edge_cases!(sgd_regressor_edge, SGDRegressor::new());

    // =============================================================================
    // Classifier edge case instantiations
    // =============================================================================

    classifier_edge_cases!(logistic_regression_edge, LogisticRegression::new());
    classifier_edge_cases!(
        decision_tree_classifier_edge,
        DecisionTreeClassifier::new().with_max_depth(Some(10))
    );
    classifier_edge_cases!(
        random_forest_classifier_edge,
        RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(5))
            .with_random_state(42)
    );
    classifier_edge_cases!(
        gradient_boosting_classifier_edge,
        GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(3))
    );
    classifier_edge_cases!(gaussian_nb_edge, GaussianNB::new());
    classifier_edge_cases!(kneighbors_classifier_edge, KNeighborsClassifier::new(3));
    classifier_edge_cases!(svc_edge, SVC::new());
    classifier_edge_cases!(linear_svc_edge, LinearSVC::new());
    classifier_edge_cases!(sgd_classifier_edge, SGDClassifier::new());

    // =============================================================================
    // Clustering edge case instantiations
    // =============================================================================

    clustering_edge_cases!(kmeans_edge, KMeans::new(2));
    clustering_edge_cases!(dbscan_edge, DBSCAN::new(0.5, 3));

    // =============================================================================
    // Transformer edge case instantiations
    // =============================================================================

    transformer_edge_cases!(standard_scaler_edge, StandardScaler::new());
    transformer_edge_cases!(min_max_scaler_edge, MinMaxScaler::new());
    transformer_edge_cases!(robust_scaler_edge, RobustScaler::new());
    transformer_edge_cases!(pca_edge, PCA::new());

    // =============================================================================
    // Special cases: MultinomialNB (requires non-negative features)
    // =============================================================================

    mod multinomial_nb_edge {
        use super::*;

        #[test]
        fn single_sample_fit() {
            let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
            let y = Array1::from_vec(vec![1.0]);
            let mut model = MultinomialNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x);
                assert!(pred.is_ok());
            }
        }

        #[test]
        fn single_feature() {
            let (x, y) = gen_single_feature_nonneg();
            let mut model = MultinomialNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x).expect("predict should work");
                assert_eq!(pred.len(), x.nrows());
            }
        }

        #[test]
        fn nan_input_rejected() {
            let x_nan = gen_nan_features();
            let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);
            let mut model = MultinomialNB::new();
            let result = model.fit(&x_nan, &y);
            assert!(result.is_err(), "NaN features should be rejected");
        }

        #[test]
        fn inf_input_rejected() {
            let x_inf = gen_inf_features();
            let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);
            let mut model = MultinomialNB::new();
            let result = model.fit(&x_inf, &y);
            assert!(result.is_err(), "Inf features should be rejected");
        }

        #[test]
        fn nan_target_rejected() {
            let (x, y) = gen_nan_targets();
            let mut model = MultinomialNB::new();
            let result = model.fit(&x, &y);
            assert!(result.is_err(), "NaN targets should be rejected");
        }

        #[test]
        fn empty_input_rejected() {
            let (x, y) = gen_empty_input();
            let mut model = MultinomialNB::new();
            let result = model.fit(&x, &y);
            assert!(result.is_err(), "empty input should be rejected");
        }

        #[test]
        fn identical_targets() {
            let (x, _) = gen_normal_data_nonneg();
            let y = Array1::from_elem(x.nrows(), 1.0);
            let mut model = MultinomialNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x).expect("predict should work");
                for &v in pred.iter() {
                    assert!((v - 1.0).abs() < 1e-10, "expected class 1.0, got {v}");
                }
            }
        }

        #[test]
        fn class_imbalance_99_1() {
            let n = 100;
            let x = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64 * 0.1 + 1.0);
            let y = Array1::from_iter((0..n).map(|i| if i < 99 { 0.0 } else { 1.0 }));
            let mut model = MultinomialNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x).expect("predict should work");
                assert_eq!(pred.len(), n);
            }
        }

        #[test]
        fn extreme_large_nonneg() {
            let n = 20;
            let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i + j) as f64 + 1.0) * 1e12);
            let y = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
            let mut model = MultinomialNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x);
                if let Ok(pred) = pred {
                    for &v in pred.iter() {
                        assert!(v.is_finite(), "prediction should be finite");
                    }
                }
            }
        }
    }

    // =============================================================================
    // Special cases: BernoulliNB (expects binary/boolean features)
    // =============================================================================

    mod bernoulli_nb_edge {
        use super::*;

        #[test]
        fn single_sample_fit() {
            let x = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
            let y = Array1::from_vec(vec![1.0]);
            let mut model = BernoulliNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x);
                assert!(pred.is_ok());
            }
        }

        #[test]
        fn single_feature() {
            let n = 20;
            let x = Array2::from_shape_fn((n, 1), |(i, _)| if i % 2 == 0 { 1.0 } else { 0.0 });
            let y = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
            let mut model = BernoulliNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x).expect("predict should work");
                assert_eq!(pred.len(), n);
            }
        }

        #[test]
        fn nan_input_rejected() {
            let x_nan = gen_nan_features();
            let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);
            let mut model = BernoulliNB::new();
            let result = model.fit(&x_nan, &y);
            assert!(result.is_err(), "NaN features should be rejected");
        }

        #[test]
        fn inf_input_rejected() {
            let x_inf = gen_inf_features();
            let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);
            let mut model = BernoulliNB::new();
            let result = model.fit(&x_inf, &y);
            assert!(result.is_err(), "Inf features should be rejected");
        }

        #[test]
        fn nan_target_rejected() {
            let (x, y) = gen_nan_targets();
            let mut model = BernoulliNB::new();
            let result = model.fit(&x, &y);
            assert!(result.is_err(), "NaN targets should be rejected");
        }

        #[test]
        fn empty_input_rejected() {
            let (x, y) = gen_empty_input();
            let mut model = BernoulliNB::new();
            let result = model.fit(&x, &y);
            assert!(result.is_err(), "empty input should be rejected");
        }

        #[test]
        fn identical_targets() {
            let (x, _y) = gen_binary_data();
            let y_same = Array1::from_elem(x.nrows(), 0.0);
            let mut model = BernoulliNB::new();
            let result = model.fit(&x, &y_same);
            if result.is_ok() {
                let pred = model.predict(&x).expect("predict should work");
                for &v in pred.iter() {
                    assert!((v - 0.0).abs() < 1e-10, "expected class 0.0, got {v}");
                }
            }
        }

        #[test]
        fn class_imbalance_99_1() {
            let n = 100;
            let x =
                Array2::from_shape_fn((n, 3), |(i, j)| if (i + j) % 2 == 0 { 1.0 } else { 0.0 });
            let y = Array1::from_iter((0..n).map(|i| if i < 99 { 0.0 } else { 1.0 }));
            let mut model = BernoulliNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x).expect("predict should work");
                assert_eq!(pred.len(), n);
            }
        }

        #[test]
        fn zero_variance_features() {
            let n = 20;
            let x = Array2::from_elem((n, 2), 1.0);
            let y = Array1::from_iter((0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }));
            let mut model = BernoulliNB::new();
            let result = model.fit(&x, &y);
            if result.is_ok() {
                let pred = model.predict(&x);
                assert!(pred.is_ok());
            }
        }
    }

    // =============================================================================
    // Cross-cutting: predict-time validation (NaN/Inf at predict, not just fit)
    // =============================================================================

    mod predict_time_validation {
        use super::*;

        #[test]
        fn regressor_nan_predict_rejected() {
            let (x_train, y_train, _) = gen_normal_data();
            let mut model = LinearRegression::new();
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_nan =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0]).unwrap();
            let result = model.predict(&x_nan);
            assert!(result.is_err(), "NaN at predict time should be rejected");
        }

        #[test]
        fn regressor_inf_predict_rejected() {
            let (x_train, y_train, _) = gen_normal_data();
            let mut model = LinearRegression::new();
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_inf =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::INFINITY, 5.0, 6.0])
                    .unwrap();
            let result = model.predict(&x_inf);
            assert!(result.is_err(), "Inf at predict time should be rejected");
        }

        #[test]
        fn classifier_nan_predict_rejected() {
            let (x_train, _, y_train) = gen_normal_data();
            let mut model = LogisticRegression::new();
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_nan =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0]).unwrap();
            let result = model.predict(&x_nan);
            assert!(result.is_err(), "NaN at predict time should be rejected");
        }

        #[test]
        fn classifier_inf_predict_rejected() {
            let (x_train, _, y_train) = gen_normal_data();
            let mut model = LogisticRegression::new();
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_inf =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NEG_INFINITY, 5.0, 6.0])
                    .unwrap();
            let result = model.predict(&x_inf);
            assert!(result.is_err(), "Inf at predict time should be rejected");
        }

        #[test]
        fn tree_nan_predict_rejected() {
            let (x_train, _, y_train) = gen_normal_data();
            let mut model = DecisionTreeClassifier::new();
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_nan =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0]).unwrap();
            let result = model.predict(&x_nan);
            assert!(result.is_err(), "NaN at predict time should be rejected");
        }

        #[test]
        fn forest_nan_predict_rejected() {
            let (x_train, _, y_train) = gen_normal_data();
            let mut model = RandomForestClassifier::new()
                .with_n_estimators(5)
                .with_random_state(42);
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_nan =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0]).unwrap();
            let result = model.predict(&x_nan);
            assert!(result.is_err(), "NaN at predict time should be rejected");
        }

        #[test]
        fn scaler_nan_transform_rejected() {
            let n = 30;
            let x_train = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64);
            let mut scaler = StandardScaler::new();
            scaler.fit(&x_train).expect("fit should succeed");

            let x_nan =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0]).unwrap();
            let result = scaler.transform(&x_nan);
            assert!(result.is_err(), "NaN at transform time should be rejected");
        }

        #[test]
        fn scaler_inf_transform_rejected() {
            let n = 30;
            let x_train = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64);
            let mut scaler = StandardScaler::new();
            scaler.fit(&x_train).expect("fit should succeed");

            let x_inf =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::INFINITY, 5.0, 6.0])
                    .unwrap();
            let result = scaler.transform(&x_inf);
            assert!(result.is_err(), "Inf at transform time should be rejected");
        }

        #[test]
        fn knn_nan_predict_rejected() {
            let (x_train, _, y_train) = gen_normal_data();
            let mut model = KNeighborsClassifier::new(3);
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_nan =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0]).unwrap();
            let result = model.predict(&x_nan);
            assert!(result.is_err(), "NaN at predict time should be rejected");
        }

        #[test]
        fn svm_nan_predict_rejected() {
            let (x_train, _, y_train) = gen_normal_data();
            let mut model = SVC::new();
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_nan =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0]).unwrap();
            let result = model.predict(&x_nan);
            assert!(result.is_err(), "NaN at predict time should be rejected");
        }

        #[test]
        fn kmeans_nan_predict_rejected() {
            let n = 50;
            let x_train = Array2::from_shape_fn((n, 3), |(i, j)| (i * 3 + j) as f64 * 0.1);
            let mut model = KMeans::new(2);
            model.fit(&x_train).expect("fit should succeed");

            let x_nan =
                Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0]).unwrap();
            let result = model.predict(&x_nan);
            assert!(result.is_err(), "NaN at predict time should be rejected");
        }
    }

    // =============================================================================
    // Not-fitted validation: predict before fit should error
    // =============================================================================

    mod not_fitted_validation {
        use super::*;

        macro_rules! test_not_fitted {
            ($name:ident, $model_expr:expr) => {
                #[test]
                fn $name() {
                    let model = $model_expr;
                    let x = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f64);
                    let result = model.predict(&x);
                    assert!(result.is_err(), "predict before fit should error");
                }
            };
        }

        test_not_fitted!(linear_regression, LinearRegression::new());
        test_not_fitted!(ridge, RidgeRegression::new(1.0));
        test_not_fitted!(lasso, LassoRegression::new(0.1));
        test_not_fitted!(elastic_net, ElasticNet::new(0.1, 0.5));
        test_not_fitted!(logistic_regression, LogisticRegression::new());
        test_not_fitted!(decision_tree_classifier, DecisionTreeClassifier::new());
        test_not_fitted!(decision_tree_regressor, DecisionTreeRegressor::new());
        test_not_fitted!(
            random_forest_classifier,
            RandomForestClassifier::new().with_n_estimators(5)
        );
        test_not_fitted!(
            random_forest_regressor,
            RandomForestRegressor::new().with_n_estimators(5)
        );
        test_not_fitted!(
            gradient_boosting_classifier,
            GradientBoostingClassifier::new().with_n_estimators(5)
        );
        test_not_fitted!(
            gradient_boosting_regressor,
            GradientBoostingRegressor::new().with_n_estimators(5)
        );
        test_not_fitted!(gaussian_nb, GaussianNB::new());
        test_not_fitted!(multinomial_nb, MultinomialNB::new());
        test_not_fitted!(bernoulli_nb, BernoulliNB::new());
        test_not_fitted!(kneighbors_classifier, KNeighborsClassifier::new(3));
        test_not_fitted!(kneighbors_regressor, KNeighborsRegressor::new(3));
        test_not_fitted!(svc, SVC::new());
        test_not_fitted!(svr, SVR::new());
        test_not_fitted!(linear_svc, LinearSVC::new());
        test_not_fitted!(linear_svr, LinearSVR::new());
        test_not_fitted!(sgd_classifier, SGDClassifier::new());
        test_not_fitted!(sgd_regressor, SGDRegressor::new());

        macro_rules! test_not_fitted_clustering {
            ($name:ident, $model_expr:expr) => {
                #[test]
                fn $name() {
                    let model = $model_expr;
                    let x = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f64);
                    let result = model.predict(&x);
                    assert!(result.is_err(), "predict before fit should error");
                }
            };
        }

        test_not_fitted_clustering!(kmeans, KMeans::new(2));
        test_not_fitted_clustering!(dbscan, DBSCAN::new(0.5, 3));

        macro_rules! test_not_fitted_transformer {
            ($name:ident, $model_expr:expr) => {
                #[test]
                fn $name() {
                    let model = $model_expr;
                    let x = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f64);
                    let result = model.transform(&x);
                    assert!(result.is_err(), "transform before fit should error");
                }
            };
        }

        test_not_fitted_transformer!(standard_scaler, StandardScaler::new());
        test_not_fitted_transformer!(min_max_scaler, MinMaxScaler::new());
        test_not_fitted_transformer!(robust_scaler, RobustScaler::new());
        test_not_fitted_transformer!(pca, PCA::new());
    }

    // =============================================================================
    // Shape mismatch: predict with wrong number of features
    // =============================================================================

    mod shape_mismatch {
        use super::*;

        #[test]
        fn regressor_feature_mismatch() {
            let (x_train, y_train, _) = gen_normal_data(); // 3 features
            let mut model = LinearRegression::new();
            model.fit(&x_train, &y_train).expect("fit should succeed");

            // Predict with 5 features instead of 3
            let x_wrong = Array2::from_shape_fn((5, 5), |(i, j)| (i * 5 + j) as f64);
            let result = model.predict(&x_wrong);
            assert!(
                result.is_err(),
                "predict with wrong feature count should error"
            );
        }

        #[test]
        fn classifier_feature_mismatch() {
            let (x_train, _, y_train) = gen_normal_data(); // 3 features
            let mut model = DecisionTreeClassifier::new();
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_wrong = Array2::from_shape_fn((5, 7), |(i, j)| (i * 7 + j) as f64);
            let result = model.predict(&x_wrong);
            assert!(
                result.is_err(),
                "predict with wrong feature count should error"
            );
        }

        #[test]
        fn scaler_feature_mismatch() {
            let x_train = Array2::from_shape_fn((30, 3), |(i, j)| (i * 3 + j) as f64);
            let mut scaler = StandardScaler::new();
            scaler.fit(&x_train).expect("fit should succeed");

            let x_wrong = Array2::from_shape_fn((5, 5), |(i, j)| (i * 5 + j) as f64);
            let result = scaler.transform(&x_wrong);
            assert!(
                result.is_err(),
                "transform with wrong feature count should error"
            );
        }

        #[test]
        fn fit_xy_shape_mismatch() {
            let x = Array2::from_shape_fn((10, 3), |(i, j)| (i * 3 + j) as f64);
            let y = Array1::from_iter(0..15).mapv(|v| v as f64); // 15 != 10
            let mut model = LinearRegression::new();
            let result = model.fit(&x, &y);
            assert!(result.is_err(), "X rows != y length should error");
        }

        #[test]
        fn forest_feature_mismatch() {
            let (x_train, y_train, _) = gen_normal_data();
            let mut model = RandomForestRegressor::new()
                .with_n_estimators(5)
                .with_random_state(42);
            model.fit(&x_train, &y_train).expect("fit should succeed");

            let x_wrong = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
            let result = model.predict(&x_wrong);
            assert!(
                result.is_err(),
                "predict with wrong feature count should error"
            );
        }

        #[test]
        fn kmeans_feature_mismatch() {
            let x_train = Array2::from_shape_fn((50, 3), |(i, j)| (i * 3 + j) as f64 * 0.1);
            let mut model = KMeans::new(2);
            model.fit(&x_train).expect("fit should succeed");

            let x_wrong = Array2::from_shape_fn((5, 7), |(i, j)| (i * 7 + j) as f64);
            let result = model.predict(&x_wrong);
            assert!(
                result.is_err(),
                "predict with wrong feature count should error"
            );
        }
    }

    // =============================================================================
    // Negative-infinity edge case
    // =============================================================================

    mod neg_infinity {
        use super::*;

        #[test]
        fn neg_inf_features_rejected_regressor() {
            let x =
                Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NEG_INFINITY, 4.0, 5.0, 6.0])
                    .unwrap();
            let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let mut model = LinearRegression::new();
            let result = model.fit(&x, &y);
            assert!(result.is_err(), "-Inf features should be rejected");
        }

        #[test]
        fn neg_inf_features_rejected_classifier() {
            let x =
                Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NEG_INFINITY, 4.0, 5.0, 6.0])
                    .unwrap();
            let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);
            let mut model = LogisticRegression::new();
            let result = model.fit(&x, &y);
            assert!(result.is_err(), "-Inf features should be rejected");
        }

        #[test]
        fn neg_inf_target_rejected() {
            let x = Array2::from_shape_fn((3, 2), |(i, j)| (i + j) as f64);
            let y = Array1::from_vec(vec![1.0, f64::NEG_INFINITY, 3.0]);
            let mut model = LinearRegression::new();
            let result = model.fit(&x, &y);
            assert!(result.is_err(), "-Inf targets should be rejected");
        }

        #[test]
        fn neg_inf_features_rejected_clustering() {
            let x =
                Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NEG_INFINITY, 4.0, 5.0, 6.0])
                    .unwrap();
            let mut model = KMeans::new(2);
            let result = model.fit(&x);
            assert!(result.is_err(), "-Inf features should be rejected");
        }

        #[test]
        fn neg_inf_features_rejected_transformer() {
            let x =
                Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NEG_INFINITY, 4.0, 5.0, 6.0])
                    .unwrap();
            let mut scaler = StandardScaler::new();
            let result = scaler.fit(&x);
            assert!(result.is_err(), "-Inf features should be rejected");
        }
    }
}

#[cfg(feature = "sparse")]
mod sparse_pipeline {
    //! Sparse Pipeline End-to-End Tests (Phase X.5)
    //!
    //! Integration tests verifying that sparse data flows correctly through
    //! pipelines and that sparse model predictions match their dense equivalents.
    //!
    //! Requires the `sparse` feature flag:
    //!   cargo test --features sparse -p ferroml-core --test sparse_pipeline_e2e

    use ferroml_core::models::traits::SparseModel;
    use ferroml_core::models::Model;
    use ferroml_core::pipeline::TextPipeline;
    use ferroml_core::preprocessing::count_vectorizer::{CountVectorizer, TextTransformer};
    use ferroml_core::preprocessing::tfidf::TfidfTransformer;
    use ferroml_core::preprocessing::tfidf_vectorizer::TfidfVectorizer;
    use ferroml_core::preprocessing::SparseTransformer;
    use ferroml_core::preprocessing::Transformer;
    use ferroml_core::sparse::CsrMatrix;
    use ndarray::{Array1, Array2};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // =============================================================================
    // Helpers
    // =============================================================================

    /// Deterministic pseudo-random data generation (no rand crate dependency).
    fn generate_dense_data(n_rows: usize, n_cols: usize, sparsity: f64, seed: u64) -> Array2<f64> {
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for i in 0..n_rows {
            for j in 0..n_cols {
                let mut hasher = DefaultHasher::new();
                (seed, i, j).hash(&mut hasher);
                let hash = hasher.finish();
                let frac = (hash % 10000) as f64 / 10000.0;
                if frac < sparsity {
                    data.push(0.0);
                } else {
                    let mut hasher2 = DefaultHasher::new();
                    (seed + 1, i, j).hash(&mut hasher2);
                    let val_hash = hasher2.finish();
                    data.push(0.1 + (val_hash % 10000) as f64 / 1000.0);
                }
            }
        }
        Array2::from_shape_vec((n_rows, n_cols), data).unwrap()
    }

    /// Generate non-negative count-like data for NB models.
    fn generate_count_data(n_rows: usize, n_cols: usize, sparsity: f64, seed: u64) -> Array2<f64> {
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for i in 0..n_rows {
            for j in 0..n_cols {
                let mut hasher = DefaultHasher::new();
                (seed, i, j).hash(&mut hasher);
                let hash = hasher.finish();
                let frac = (hash % 10000) as f64 / 10000.0;
                if frac < sparsity {
                    data.push(0.0);
                } else {
                    data.push(((hash % 5) + 1) as f64);
                }
            }
        }
        Array2::from_shape_vec((n_rows, n_cols), data).unwrap()
    }

    /// Generate binary labels deterministically.
    fn generate_labels(n: usize, seed: u64) -> Array1<f64> {
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let mut hasher = DefaultHasher::new();
            (seed, i, 999usize).hash(&mut hasher);
            let h = hasher.finish();
            labels.push(if h % 2 == 0 { 0.0 } else { 1.0 });
        }
        Array1::from_vec(labels)
    }

    /// Generate regression targets deterministically.
    fn generate_regression_targets(n: usize, seed: u64) -> Array1<f64> {
        let mut targets = Vec::with_capacity(n);
        for i in 0..n {
            let mut hasher = DefaultHasher::new();
            (seed, i, 888usize).hash(&mut hasher);
            let h = hasher.finish();
            targets.push((h % 1000) as f64 / 100.0);
        }
        Array1::from_vec(targets)
    }

    fn sample_corpus() -> Vec<String> {
        vec![
            "the cat sat on the mat".to_string(),
            "the dog chased the cat".to_string(),
            "a bird flew over the house".to_string(),
            "the fish swam in the pond".to_string(),
            "the cat and dog played together".to_string(),
            "a small bird landed on the mat".to_string(),
            "the dog ran after the bird".to_string(),
            "fish and chips for dinner".to_string(),
        ]
    }

    fn binary_labels_8() -> Array1<f64> {
        Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    }

    const EPSILON: f64 = 1e-6;

    // =============================================================================
    // 1. Text Classification Pipeline: CountVectorizer -> TfidfTransformer -> MultinomialNB
    // =============================================================================

    #[test]
    fn test_text_pipeline_cv_tfidf_multinomial_nb() {
        let corpus = sample_corpus();
        let y = binary_labels_8();

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("clf", ferroml_core::models::MultinomialNB::new());

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();

        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "NaN/Inf prediction");
            assert!(p == 0.0 || p == 1.0, "Unexpected label: {}", p);
        }
    }

    #[test]
    fn test_text_pipeline_matches_manual_steps() {
        // Verify that TextPipeline produces identical results to manual step-by-step
        let corpus = sample_corpus();
        let y = binary_labels_8();

        // --- Manual ---
        let mut cv = CountVectorizer::new();
        cv.fit_text(&corpus).unwrap();
        let sparse_counts = cv.transform_text(&corpus).unwrap();

        let mut tfidf = TfidfTransformer::new();
        SparseTransformer::fit_sparse(&mut tfidf, &sparse_counts).unwrap();
        let sparse_tfidf = SparseTransformer::transform_sparse(&tfidf, &sparse_counts).unwrap();

        let mut mnb = ferroml_core::models::MultinomialNB::new();
        ferroml_core::pipeline::PipelineSparseModel::fit_sparse(&mut mnb, &sparse_tfidf, &y)
            .unwrap();
        let manual_preds =
            ferroml_core::pipeline::PipelineSparseModel::predict_sparse(&mnb, &sparse_tfidf)
                .unwrap();

        // --- Pipeline ---
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("clf", ferroml_core::models::MultinomialNB::new());

        pipeline.fit(&corpus, &y).unwrap();
        let pipeline_preds = pipeline.predict(&corpus).unwrap();

        assert_eq!(manual_preds.len(), pipeline_preds.len());
        for (i, (&m, &p)) in manual_preds.iter().zip(pipeline_preds.iter()).enumerate() {
            assert!(
                (m - p).abs() < 1e-12,
                "Mismatch at index {}: manual={}, pipeline={}",
                i,
                m,
                p
            );
        }
    }

    #[test]
    fn test_text_pipeline_dense_equivalent_match() {
        // Compare text pipeline (sparse path) with manual dense path
        let corpus = sample_corpus();
        let y = binary_labels_8();

        // Sparse path via pipeline
        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("clf", ferroml_core::models::MultinomialNB::new());

        pipeline.fit(&corpus, &y).unwrap();
        let sparse_preds = pipeline.predict(&corpus).unwrap();

        // Dense path: CountVectorizer -> dense -> TfidfTransformer -> dense -> MultinomialNB
        let mut cv = CountVectorizer::new();
        cv.fit_text(&corpus).unwrap();
        let sparse_counts = cv.transform_text(&corpus).unwrap();
        let dense_counts = sparse_counts.to_dense();

        let mut tfidf_dense = TfidfTransformer::new();
        tfidf_dense.fit(&dense_counts).unwrap();
        let dense_tfidf = tfidf_dense.transform(&dense_counts).unwrap();

        let mut mnb_dense = ferroml_core::models::MultinomialNB::new();
        mnb_dense.fit(&dense_tfidf, &y).unwrap();
        let dense_preds = mnb_dense.predict(&dense_tfidf).unwrap();

        assert_eq!(sparse_preds.len(), dense_preds.len());
        for (i, (&s, &d)) in sparse_preds.iter().zip(dense_preds.iter()).enumerate() {
            assert!(
                (s - d).abs() < EPSILON,
                "Sparse vs dense mismatch at {}: sparse={}, dense={}",
                i,
                s,
                d
            );
        }
    }

    // =============================================================================
    // 2. Sparse Linear Pipeline: LogisticRegression
    // =============================================================================

    #[test]
    fn test_sparse_logistic_regression_vs_dense() {
        let dense = generate_dense_data(40, 10, 0.6, 42);
        let y = generate_labels(40, 42);
        let sparse = CsrMatrix::from_dense(&dense);

        // Dense path
        let mut model_dense = ferroml_core::models::LogisticRegression::new();
        model_dense.fit(&dense, &y).unwrap();
        let preds_dense = model_dense.predict(&dense).unwrap();

        // Sparse path
        let mut model_sparse = ferroml_core::models::LogisticRegression::new();
        model_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = model_sparse.predict_sparse(&sparse).unwrap();

        assert_eq!(preds_dense.len(), preds_sparse.len());
        for (i, (&d, &s)) in preds_dense.iter().zip(preds_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "LogisticRegression mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    // =============================================================================
    // 3. Sparse KNN Pipeline: KNeighborsClassifier
    // =============================================================================

    #[test]
    fn test_sparse_knn_classifier_vs_dense() {
        // Two well-separated clusters for reliable classification
        let dense = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.2, 2.0, 1.8, 1.2, 2.0, // cluster 0
                8.0, 8.0, 8.5, 8.2, 9.0, 8.8, 8.2, 9.0, // cluster 1
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let sparse = CsrMatrix::from_dense(&dense);

        // Dense path
        let mut model_dense = ferroml_core::models::KNeighborsClassifier::new(3);
        model_dense.fit(&dense, &y).unwrap();
        let preds_dense = model_dense.predict(&dense).unwrap();

        // Sparse path
        let mut model_sparse = ferroml_core::models::KNeighborsClassifier::new(3);
        model_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = model_sparse.predict_sparse(&sparse).unwrap();

        assert_eq!(preds_dense.len(), preds_sparse.len());
        for (i, (&d, &s)) in preds_dense.iter().zip(preds_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < 1e-10,
                "KNN mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_sparse_knn_regressor_vs_dense() {
        let dense = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 6.0, 7.0, 7.0, 6.0, 8.0, 8.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 1.5, 2.0, 5.0, 5.5, 6.0]);

        let sparse = CsrMatrix::from_dense(&dense);

        let mut model_dense = ferroml_core::models::KNeighborsRegressor::new(3);
        model_dense.fit(&dense, &y).unwrap();
        let preds_dense = model_dense.predict(&dense).unwrap();

        let mut model_sparse = ferroml_core::models::KNeighborsRegressor::new(3);
        model_sparse.fit_sparse(&sparse, &y).unwrap();
        let preds_sparse = model_sparse.predict_sparse(&sparse).unwrap();

        assert_eq!(preds_dense.len(), preds_sparse.len());
        for (i, (&d, &s)) in preds_dense.iter().zip(preds_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "KNN Regressor mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    // =============================================================================
    // 4. Sparse Round-Trip for All Sparse-Enabled Models
    //    Dense -> to_sparse -> fit_sparse -> predict_sparse
    //    vs Dense -> fit -> predict
    // =============================================================================

    #[test]
    fn test_roundtrip_multinomial_nb() {
        let dense = generate_count_data(40, 10, 0.5, 101);
        let y = generate_labels(40, 101);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::MultinomialNB::new();
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::MultinomialNB::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "MultinomialNB roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_bernoulli_nb() {
        // Binary data for BernoulliNB
        let mut data = Vec::new();
        for i in 0..40 {
            for j in 0..10 {
                let mut hasher = DefaultHasher::new();
                (102u64, i, j).hash(&mut hasher);
                let h = hasher.finish();
                data.push(if h % 3 == 0 { 1.0 } else { 0.0 });
            }
        }
        let dense = Array2::from_shape_vec((40, 10), data).unwrap();
        let y = generate_labels(40, 102);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::BernoulliNB::new();
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::BernoulliNB::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "BernoulliNB roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_gaussian_nb() {
        let dense = generate_dense_data(40, 10, 0.5, 103);
        let y = generate_labels(40, 103);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::GaussianNB::new();
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::GaussianNB::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "GaussianNB roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_categorical_nb() {
        let dense = generate_count_data(40, 10, 0.4, 104);
        let y = generate_labels(40, 104);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::CategoricalNB::new();
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::CategoricalNB::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "CategoricalNB roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_logistic_regression() {
        let dense = generate_dense_data(40, 10, 0.5, 105);
        let y = generate_labels(40, 105);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::LogisticRegression::new();
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::LogisticRegression::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "LogisticRegression roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_ridge_regression() {
        let dense = generate_dense_data(40, 10, 0.5, 106);
        let y = generate_regression_targets(40, 106);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::RidgeRegression::new(1.0);
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::RidgeRegression::new(1.0);
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "RidgeRegression roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_linear_svc() {
        let dense = generate_dense_data(40, 10, 0.5, 107);
        let y = generate_labels(40, 107);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::LinearSVC::new();
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::LinearSVC::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "LinearSVC roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_linear_svr() {
        let dense = generate_dense_data(40, 10, 0.5, 108);
        let y = generate_regression_targets(40, 108);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::LinearSVR::new();
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::LinearSVR::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "LinearSVR roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_kneighbors_classifier() {
        let dense = generate_dense_data(30, 8, 0.4, 109);
        let y = generate_labels(30, 109);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::KNeighborsClassifier::new(5);
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::KNeighborsClassifier::new(5);
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "KNeighborsClassifier roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_kneighbors_regressor() {
        let dense = generate_dense_data(30, 8, 0.4, 110);
        let y = generate_regression_targets(30, 110);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::KNeighborsRegressor::new(5);
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::KNeighborsRegressor::new(5);
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "KNeighborsRegressor roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_roundtrip_nearest_centroid() {
        let dense = generate_dense_data(30, 8, 0.4, 111);
        let y = generate_labels(30, 111);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::NearestCentroid::new();
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::NearestCentroid::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "NearestCentroid roundtrip mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    // =============================================================================
    // 5. Mixed Pipeline: Sparse -> Dense -> StandardScaler -> LogisticRegression
    // =============================================================================

    #[test]
    fn test_mixed_sparse_to_dense_pipeline() {
        // Verify that sparse data can be converted to dense mid-pipeline and
        // processed with standard dense transforms + model.
        let corpus = sample_corpus();
        let y = binary_labels_8();

        // Step 1: CountVectorizer -> sparse CsrMatrix
        let mut cv = CountVectorizer::new();
        cv.fit_text(&corpus).unwrap();
        let sparse_counts = cv.transform_text(&corpus).unwrap();

        // Step 2: Convert to dense
        let dense = sparse_counts.to_dense();
        assert_eq!(dense.nrows(), corpus.len());
        assert!(dense.ncols() > 0);

        // Step 3: StandardScaler on dense data
        use ferroml_core::preprocessing::scalers::StandardScaler;

        let mut scaler = StandardScaler::new();
        let scaled = scaler.fit_transform(&dense).unwrap();

        // Step 4: LogisticRegression on scaled dense data
        let mut lr = ferroml_core::models::LogisticRegression::new();
        lr.fit(&scaled, &y).unwrap();
        let preds = lr.predict(&scaled).unwrap();

        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "NaN/Inf in mixed pipeline prediction");
            assert!(p == 0.0 || p == 1.0, "Unexpected label: {}", p);
        }
    }

    #[test]
    fn test_mixed_pipeline_via_text_pipeline_dense_model() {
        // Use TextPipeline's add_dense_model path which automatically
        // densifies sparse data before passing to the model.
        let corpus = sample_corpus();
        let y = binary_labels_8();

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_dense_model("clf", ferroml_core::models::LogisticRegression::new());

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();

        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "NaN/Inf in dense model pipeline prediction");
        }
    }

    // =============================================================================
    // 6. Additional Pipeline Combinations
    // =============================================================================

    #[test]
    fn test_text_pipeline_cv_tfidf_bernoulli_nb() {
        let corpus = sample_corpus();
        let y = binary_labels_8();

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("clf", ferroml_core::models::BernoulliNB::new());

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();

        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite());
            assert!(p == 0.0 || p == 1.0);
        }
    }

    #[test]
    fn test_text_pipeline_cv_tfidf_logistic_regression() {
        let corpus = sample_corpus();
        let y = binary_labels_8();

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("clf", ferroml_core::models::LogisticRegression::new());

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();

        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_text_pipeline_cv_tfidf_linear_svc() {
        let corpus = sample_corpus();
        let y = binary_labels_8();

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("cv", CountVectorizer::new())
            .add_sparse_transformer("tfidf", TfidfTransformer::new())
            .add_sparse_model("clf", ferroml_core::models::LinearSVC::new());

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();

        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_text_pipeline_tfidf_vectorizer_ridge() {
        // Single-step TfidfVectorizer (CountVectorizer + TfidfTransformer combined)
        let corpus = sample_corpus();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let mut pipeline = TextPipeline::new()
            .add_text_transformer("tfidf", TfidfVectorizer::new())
            .add_sparse_model("clf", ferroml_core::models::RidgeRegression::new(1.0));

        pipeline.fit(&corpus, &y).unwrap();
        let preds = pipeline.predict(&corpus).unwrap();

        assert_eq!(preds.len(), corpus.len());
        for &p in preds.iter() {
            assert!(p.is_finite(), "NaN/Inf in ridge regression prediction");
        }
    }

    // =============================================================================
    // 7. TfidfTransformer Sparse Round-Trip
    // =============================================================================

    #[test]
    fn test_tfidf_sparse_native_vs_dense() {
        // Verify TfidfTransformer produces identical output via sparse and dense paths
        let dense_counts = generate_count_data(20, 8, 0.4, 200);
        let sparse_counts = CsrMatrix::from_dense(&dense_counts);

        // Dense path
        let mut tfidf_dense = TfidfTransformer::new();
        tfidf_dense.fit(&dense_counts).unwrap();
        let result_dense = tfidf_dense.transform(&dense_counts).unwrap();

        // Sparse path (fit_sparse + transform_sparse returns dense)
        let mut tfidf_sparse = TfidfTransformer::new();
        tfidf_sparse.fit_sparse(&sparse_counts).unwrap();
        let result_sparse = tfidf_sparse.transform_sparse(&sparse_counts).unwrap();

        assert_eq!(result_dense.shape(), result_sparse.shape());
        for i in 0..result_dense.nrows() {
            for j in 0..result_dense.ncols() {
                assert!(
                    (result_dense[[i, j]] - result_sparse[[i, j]]).abs() < 1e-10,
                    "TF-IDF mismatch at [{},{}]: dense={}, sparse={}",
                    i,
                    j,
                    result_dense[[i, j]],
                    result_sparse[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_tfidf_sparse_native_preserves_sparsity() {
        // Verify transform_sparse_native returns a CsrMatrix (stays in sparse domain)
        let dense_counts = generate_count_data(20, 8, 0.6, 201);
        let sparse_counts = CsrMatrix::from_dense(&dense_counts);

        let mut tfidf = TfidfTransformer::new();
        tfidf.fit_sparse(&sparse_counts).unwrap();
        let result_sparse = tfidf.transform_sparse_native(&sparse_counts).unwrap();

        // Result should be a CsrMatrix with correct shape
        assert_eq!(result_sparse.nrows(), 20);
        assert_eq!(result_sparse.ncols(), 8);
        // Sparse result should have fewer or equal nnz to input (zero TF values stay zero)
        assert!(result_sparse.nnz() <= sparse_counts.nnz());
    }

    // =============================================================================
    // 8. Edge Cases
    // =============================================================================

    #[test]
    fn test_sparse_highly_sparse_data_roundtrip() {
        // >95% sparsity
        let mut data = vec![0.0; 50 * 100];
        // Only ~25 non-zero entries
        for k in 0..25 {
            let mut hasher = DefaultHasher::new();
            (300u64, k).hash(&mut hasher);
            let h = hasher.finish();
            let i = (h % 50) as usize;
            let j = ((h / 50) % 100) as usize;
            data[i * 100 + j] = ((h % 5) + 1) as f64;
        }
        let dense = Array2::from_shape_vec((50, 100), data).unwrap();
        let y = generate_labels(50, 300);
        let sparse = CsrMatrix::from_dense(&dense);

        assert!(sparse.sparsity() > 0.9, "Should be highly sparse");

        let mut m_sparse = ferroml_core::models::MultinomialNB::new();
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        assert_eq!(p_sparse.len(), 50);
        for &p in p_sparse.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_sparse_single_feature_roundtrip() {
        let dense = Array2::from_shape_vec((4, 1), vec![1.0, 0.0, 3.0, 0.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut m_dense = ferroml_core::models::RidgeRegression::new(0.01);
        m_dense.fit(&dense, &y).unwrap();
        let p_dense = m_dense.predict(&dense).unwrap();

        let mut m_sparse = ferroml_core::models::RidgeRegression::new(0.01);
        m_sparse.fit_sparse(&sparse, &y).unwrap();
        let p_sparse = m_sparse.predict_sparse(&sparse).unwrap();

        for (i, (&d, &s)) in p_dense.iter().zip(p_sparse.iter()).enumerate() {
            assert!(
                (d - s).abs() < EPSILON,
                "Single-feature mismatch at {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_sparse_all_zero_rows() {
        // Matrix with some all-zero rows
        let dense = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 2.0, // non-zero
                0.0, 0.0, 0.0, // all-zero
                3.0, 0.0, 1.0, // non-zero
                0.0, 0.0, 0.0, // all-zero
                2.0, 1.0, 0.0, // non-zero
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0]);
        let sparse = CsrMatrix::from_dense(&dense);

        let mut model = ferroml_core::models::MultinomialNB::new();
        model.fit_sparse(&sparse, &y).unwrap();
        let preds = model.predict_sparse(&sparse).unwrap();

        assert_eq!(preds.len(), 5);
        for &p in preds.iter() {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_csr_from_dense_and_back() {
        // Verify CsrMatrix -> to_dense round-trip preserves data exactly
        let dense = generate_dense_data(10, 5, 0.6, 400);
        let sparse = CsrMatrix::from_dense(&dense);
        let recovered = sparse.to_dense();

        assert_eq!(dense.shape(), recovered.shape());
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert!(
                    (dense[[i, j]] - recovered[[i, j]]).abs() < 1e-15,
                    "CsrMatrix round-trip mismatch at [{},{}]",
                    i,
                    j
                );
            }
        }
    }
}
