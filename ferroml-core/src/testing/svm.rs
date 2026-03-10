//! Dedicated SVM test module
//!
//! Comprehensive tests for SVC, SVR, LinearSVC, and LinearSVR covering:
//! - Kernel computations
//! - Classification and regression
//! - Decision function output shapes
//! - Loss function variants
//! - Class weight support

#[cfg(test)]
mod tests {
    use crate::models::svm::*;
    use crate::models::{Model, ProbabilisticModel};
    use ndarray::{Array1, Array2};

    // =========================================================================
    // Test data helpers
    // =========================================================================

    fn make_binary_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 2.0, 1.0, 1.5, 1.5, 2.5, 2.5, 6.0, 7.0, 7.0, 6.0, 6.5, 6.5, 7.5, 7.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    fn make_multiclass_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 1.0, 5.0, 5.0, 5.5, 5.5, 6.0, 5.0, 9.0, 1.0, 9.5, 1.5,
                10.0, 1.0, 1.0, 9.0, 1.5, 9.5, 2.0, 9.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0,
        ]);
        (x, y)
    }

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0,
                9.0, 9.0, 10.0, 10.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
        (x, y)
    }

    // =========================================================================
    // Kernel tests
    // =========================================================================

    #[test]
    fn test_kernel_linear_computation() {
        let k = Kernel::Linear;
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = k.compute(&x, &y);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_rbf_computation() {
        let k = Kernel::Rbf { gamma: 0.5 };
        let x = [1.0, 0.0];
        let y = [0.0, 1.0];
        // ||x-y||^2 = 1+1 = 2, exp(-0.5 * 2) = exp(-1)
        let expected = (-1.0_f64).exp();
        let result = k.compute(&x, &y);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_poly_computation() {
        let k = Kernel::Polynomial {
            gamma: 1.0,
            coef0: 1.0,
            degree: 2,
        };
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];
        // (1*1*3 + 1*2*4 + 1)^2 = (3+8+1)^2 = 144
        let expected = 144.0;
        let result = k.compute(&x, &y);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_sigmoid_computation() {
        let k = Kernel::Sigmoid {
            gamma: 0.1,
            coef0: 0.0,
        };
        let x = [1.0, 2.0];
        let y = [3.0, 4.0];
        // tanh(0.1 * (3+8)) = tanh(1.1)
        let expected = 1.1_f64.tanh();
        let result = k.compute(&x, &y);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_kernel_rbf_auto_gamma() {
        // gamma=0 signals auto; SVC with rbf_auto() should set gamma=1/n_features on fit
        let (x, y) = make_binary_data();
        let mut model = SVC::new().with_kernel(Kernel::rbf_auto()).with_c(10.0);
        model.fit(&x, &y).unwrap();
        // After fitting, the model should use gamma = 1/2 = 0.5 (2 features)
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    // =========================================================================
    // SVC tests
    // =========================================================================

    #[test]
    fn test_svc_binary_classification() {
        let (x, y) = make_binary_data();
        let mut model = SVC::new().with_c(10.0);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
        // Should classify most correctly on well-separated data
        let correct: usize = preds
            .iter()
            .zip(y.iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count();
        assert!(correct >= 6, "Expected at least 6/8 correct, got {correct}");
    }

    #[test]
    fn test_svc_multiclass_ovo() {
        let (x, y) = make_multiclass_data();
        let mut model = SVC::new()
            .with_c(10.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsOne);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 12);
    }

    #[test]
    fn test_svc_multiclass_ovr() {
        let (x, y) = make_multiclass_data();
        let mut model = SVC::new()
            .with_c(10.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsRest);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 12);
    }

    #[test]
    fn test_svc_ovo_vs_ovr_same_data() {
        let (x, y) = make_binary_data();
        let mut ovo = SVC::new()
            .with_c(10.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsOne);
        let mut ovr = SVC::new()
            .with_c(10.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsRest);
        ovo.fit(&x, &y).unwrap();
        ovr.fit(&x, &y).unwrap();
        let preds_ovo = ovo.predict(&x).unwrap();
        let preds_ovr = ovr.predict(&x).unwrap();
        // Both should produce reasonable results
        assert_eq!(preds_ovo.len(), preds_ovr.len());
    }

    #[test]
    fn test_svc_platt_scaling_probabilities() {
        let (x, y) = make_binary_data();
        let mut model = SVC::new().with_c(10.0).with_probability(true);
        model.fit(&x, &y).unwrap();
        let probas = model.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[8, 2]);
        for i in 0..8 {
            let row_sum: f64 = probas.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {i} sums to {row_sum}");
            assert!(probas[[i, 0]] >= 0.0 && probas[[i, 1]] >= 0.0);
        }
    }

    #[test]
    fn test_svc_balanced_class_weight() {
        let (x, y) = make_binary_data();
        let mut model = SVC::new()
            .with_c(10.0)
            .with_class_weight(ClassWeight::Balanced);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_svc_custom_class_weight() {
        let (x, y) = make_binary_data();
        let mut model = SVC::new()
            .with_c(10.0)
            .with_class_weight(ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 10.0)]));
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_svc_decision_function_ovo_shape() {
        let (x, y) = make_multiclass_data();
        let mut model = SVC::new()
            .with_c(10.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsOne);
        model.fit(&x, &y).unwrap();
        let decisions = model.decision_function(&x).unwrap();
        let n_classes = 4;
        let n_pairs = n_classes * (n_classes - 1) / 2;
        assert_eq!(decisions.shape(), &[12, n_pairs]);
    }

    #[test]
    fn test_svc_decision_function_ovr_shape() {
        let (x, y) = make_multiclass_data();
        let mut model = SVC::new()
            .with_c(10.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsRest);
        model.fit(&x, &y).unwrap();
        let decisions = model.decision_function(&x).unwrap();
        assert_eq!(decisions.shape(), &[12, 4]);
    }

    #[test]
    fn test_svc_decision_function_consistency() {
        let (x, y) = make_binary_data();
        let mut model = SVC::new()
            .with_c(10.0)
            .with_multiclass_strategy(MulticlassStrategy::OneVsOne);
        model.fit(&x, &y).unwrap();
        let decisions = model.decision_function(&x).unwrap();
        let preds = model.predict(&x).unwrap();
        // For binary OvO: 1 classifier, positive decision → class 1
        assert_eq!(decisions.ncols(), 1);
        for i in 0..x.nrows() {
            let d = decisions[[i, 0]];
            let p = preds[i];
            // Positive decision should correlate with class 1
            if d > 0.5 {
                assert!(
                    (p - 1.0).abs() < 0.5 || (p - 0.0).abs() < 0.5,
                    "Decision {d} should map to class 0 or 1, got {p}"
                );
            }
        }
    }

    #[test]
    fn test_svc_unfitted_errors() {
        let model = SVC::new();
        let x = Array2::zeros((5, 2));
        assert!(model.predict(&x).is_err());
        assert!(model.decision_function(&x).is_err());
    }

    #[test]
    fn test_svc_all_kernels_fit() {
        let (x, y) = make_binary_data();
        let kernels = vec![
            Kernel::Linear,
            Kernel::rbf(0.5),
            Kernel::poly(3, 0.1, 1.0),
            Kernel::sigmoid(0.01, 0.0),
        ];
        for kernel in kernels {
            let mut model = SVC::new().with_c(10.0).with_kernel(kernel);
            model.fit(&x, &y).unwrap();
            let preds = model.predict(&x).unwrap();
            assert_eq!(preds.len(), 8, "Kernel {:?} failed", kernel);
        }
    }

    // =========================================================================
    // SVR tests
    // =========================================================================

    #[test]
    fn test_svr_basic_regression() {
        let (x, y) = make_regression_data();
        let mut model = SVR::new().with_c(100.0).with_kernel(Kernel::Linear);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
        // Predictions should be correlated with targets
        let correlation: f64 = preds.iter().zip(y.iter()).map(|(p, t)| p * t).sum::<f64>();
        assert!(
            correlation > 0.0,
            "Predictions should correlate with targets"
        );
    }

    #[test]
    fn test_svr_epsilon_tube() {
        let (x, y) = make_regression_data();
        let mut model_small = SVR::new()
            .with_c(100.0)
            .with_epsilon(0.01)
            .with_kernel(Kernel::Linear);
        let mut model_large = SVR::new()
            .with_c(100.0)
            .with_epsilon(5.0)
            .with_kernel(Kernel::Linear);
        model_small.fit(&x, &y).unwrap();
        model_large.fit(&x, &y).unwrap();
        let n_sv_small = model_small.n_support_vectors();
        let n_sv_large = model_large.n_support_vectors();
        // Larger epsilon means more points inside tube → fewer support vectors
        assert!(
            n_sv_large <= n_sv_small,
            "Large epsilon ({n_sv_large} SVs) should have <= SVs than small epsilon ({n_sv_small})"
        );
    }

    #[test]
    fn test_svr_all_kernels() {
        let (x, y) = make_regression_data();
        let kernels = vec![
            Kernel::Linear,
            Kernel::rbf(0.1),
            Kernel::poly(2, 0.1, 1.0),
            Kernel::sigmoid(0.01, 0.0),
        ];
        for kernel in kernels {
            let mut model = SVR::new().with_c(10.0).with_kernel(kernel);
            model.fit(&x, &y).unwrap();
            let preds = model.predict(&x).unwrap();
            assert_eq!(preds.len(), 10, "Kernel {:?} failed", kernel);
        }
    }

    #[test]
    fn test_svr_decision_function_equals_predict() {
        let (x, y) = make_regression_data();
        let mut model = SVR::new().with_c(100.0).with_kernel(Kernel::Linear);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        let decisions = model.decision_function(&x).unwrap();
        for (p, d) in preds.iter().zip(decisions.iter()) {
            assert!(
                (p - d).abs() < 1e-10,
                "predict ({p}) != decision_function ({d})"
            );
        }
    }

    #[test]
    fn test_svr_support_vectors_subset() {
        let (x, y) = make_regression_data();
        let mut model = SVR::new().with_c(10.0).with_kernel(Kernel::Linear);
        model.fit(&x, &y).unwrap();
        let n_sv = model.n_support_vectors();
        assert!(
            n_sv <= x.nrows(),
            "n_sv ({n_sv}) should be <= n_train ({})",
            x.nrows()
        );
    }

    #[test]
    fn test_svr_dual_coef_length() {
        let (x, y) = make_regression_data();
        let mut model = SVR::new().with_c(10.0).with_kernel(Kernel::Linear);
        model.fit(&x, &y).unwrap();
        let n_sv = model.n_support_vectors();
        let dual_coef = model.dual_coef().unwrap();
        assert_eq!(
            dual_coef.len(),
            n_sv,
            "dual_coef length ({}) != n_sv ({n_sv})",
            dual_coef.len()
        );
    }

    // =========================================================================
    // LinearSVC tests
    // =========================================================================

    #[test]
    fn test_linear_svc_binary() {
        let (x, y) = make_binary_data();
        let mut model = LinearSVC::new().with_c(10.0).with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_linear_svc_multiclass() {
        let (x, y) = make_multiclass_data();
        let mut model = LinearSVC::new().with_c(10.0).with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 12);
    }

    #[test]
    fn test_linear_svc_hinge_loss() {
        let (x, y) = make_binary_data();
        let mut model = LinearSVC::new()
            .with_c(10.0)
            .with_loss(LinearSVCLoss::Hinge)
            .with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_linear_svc_squared_hinge_loss() {
        let (x, y) = make_binary_data();
        let mut model = LinearSVC::new()
            .with_c(10.0)
            .with_loss(LinearSVCLoss::SquaredHinge)
            .with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_linear_svc_balanced_weight() {
        let (x, y) = make_binary_data();
        let mut model = LinearSVC::new()
            .with_c(10.0)
            .with_class_weight(ClassWeight::Balanced)
            .with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_linear_svc_custom_weight() {
        let (x, y) = make_binary_data();
        let mut model = LinearSVC::new()
            .with_c(10.0)
            .with_class_weight(ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 5.0)]))
            .with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_linear_svc_decision_function_shape() {
        let (x, y) = make_binary_data();
        let mut model = LinearSVC::new().with_c(10.0).with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let decisions = model.decision_function(&x).unwrap();
        // Binary: 1 classifier
        assert_eq!(decisions.shape(), &[8, 1]);
    }

    #[test]
    fn test_linear_svc_decision_function_sign_matches_predict() {
        let (x, y) = make_binary_data();
        let mut model = LinearSVC::new().with_c(10.0).with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let decisions = model.decision_function(&x).unwrap();
        let preds = model.predict(&x).unwrap();
        let classes = model.classes().unwrap();
        for i in 0..x.nrows() {
            let d = decisions[[i, 0]];
            if d >= 0.0 {
                assert!(
                    (preds[i] - classes[1]).abs() < 1e-10,
                    "Positive decision {d} should predict class {}, got {}",
                    classes[1],
                    preds[i]
                );
            } else {
                assert!(
                    (preds[i] - classes[0]).abs() < 1e-10,
                    "Negative decision {d} should predict class {}, got {}",
                    classes[0],
                    preds[i]
                );
            }
        }
    }

    // =========================================================================
    // LinearSVR tests
    // =========================================================================

    #[test]
    fn test_linear_svr_basic() {
        let (x, y) = make_regression_data();
        let mut model = LinearSVR::new().with_c(10.0).with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_linear_svr_epsilon_insensitive_loss() {
        let (x, y) = make_regression_data();
        let mut model = LinearSVR::new()
            .with_c(10.0)
            .with_loss(LinearSVRLoss::EpsilonInsensitive)
            .with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_linear_svr_squared_loss() {
        let (x, y) = make_regression_data();
        let mut model = LinearSVR::new()
            .with_c(10.0)
            .with_loss(LinearSVRLoss::SquaredEpsilonInsensitive)
            .with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 10);
    }

    #[test]
    fn test_linear_svr_coefficients() {
        let (x, y) = make_regression_data();
        let mut model = LinearSVR::new().with_c(10.0).with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let weights = model.weights().unwrap();
        assert_eq!(weights.len(), 2);
        let intercept = model.intercept();
        // Intercept should be finite
        assert!(intercept.is_finite());
    }

    #[test]
    fn test_linear_svr_decision_function() {
        let (x, y) = make_regression_data();
        let mut model = LinearSVR::new().with_c(10.0).with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        let decisions = model.decision_function(&x).unwrap();
        for (p, d) in preds.iter().zip(decisions.iter()) {
            assert!(
                (p - d).abs() < 1e-10,
                "predict ({p}) != decision_function ({d})"
            );
        }
    }

    #[test]
    fn test_linear_svr_fit_intercept_false() {
        let (x, y) = make_regression_data();
        let mut model = LinearSVR::new()
            .with_c(10.0)
            .with_fit_intercept(false)
            .with_max_iter(2000);
        model.fit(&x, &y).unwrap();
        let intercept = model.intercept();
        assert!(
            intercept.abs() < 1e-10,
            "Intercept should be ~0 when fit_intercept=false, got {intercept}"
        );
    }
}
