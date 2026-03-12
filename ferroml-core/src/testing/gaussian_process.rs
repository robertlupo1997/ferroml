//! Tests for Gaussian Process regression and classification.

#[cfg(test)]
mod tests {
    use crate::models::gaussian_process::*;
    use crate::models::Model;
    use ndarray::{Array1, Array2};

    // =========================================================================
    // Kernel tests
    // =========================================================================

    #[test]
    fn test_rbf_kernel_two_points() {
        let rbf = RBF::new(1.0);
        let x1 = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let k = rbf.compute(&x1, &x2);
        // K = exp(-1 / 2) = 0.6065...
        let expected = (-0.5_f64).exp();
        assert!((k[[0, 0]] - expected).abs() < 1e-10, "got {}", k[[0, 0]]);
    }

    #[test]
    fn test_rbf_kernel_diagonal_all_ones() {
        let rbf = RBF::new(2.0);
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let diag = rbf.diagonal(&x);
        for i in 0..5 {
            assert!((diag[i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rbf_kernel_self_diagonal() {
        let rbf = RBF::new(1.5);
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let k = rbf.compute(&x, &x);
        // Diagonal of K(X, X) should be 1.0
        for i in 0..3 {
            assert!((k[[i, i]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rbf_kernel_length_scale() {
        let rbf1 = RBF::new(1.0);
        let rbf2 = RBF::new(10.0);
        let x1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
        let k1 = rbf1.compute(&x1, &x2)[[0, 0]];
        let k2 = rbf2.compute(&x1, &x2)[[0, 0]];
        // Larger length_scale = broader kernel = higher similarity at same distance
        assert!(k2 > k1, "k2={} should be > k1={}", k2, k1);
    }

    #[test]
    fn test_matern_05_exponential() {
        let m = Matern::new(1.0, 0.5);
        let x1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![2.0]).unwrap();
        let k = m.compute(&x1, &x2)[[0, 0]];
        let expected = (-2.0_f64).exp();
        assert!((k - expected).abs() < 1e-10, "got {}", k);
    }

    #[test]
    fn test_matern_15() {
        let m = Matern::new(1.0, 1.5);
        let x1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let k = m.compute(&x1, &x2)[[0, 0]];
        let s3 = 3.0_f64.sqrt();
        let expected = (1.0 + s3) * (-s3).exp();
        assert!(
            (k - expected).abs() < 1e-10,
            "got {} expected {}",
            k,
            expected
        );
    }

    #[test]
    fn test_matern_25() {
        let m = Matern::new(1.0, 2.5);
        let x1 = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let x2 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let k = m.compute(&x1, &x2)[[0, 0]];
        let s5 = 5.0_f64.sqrt();
        let expected = (1.0 + s5 + s5 * s5 / 3.0) * (-s5).exp();
        assert!(
            (k - expected).abs() < 1e-10,
            "got {} expected {}",
            k,
            expected
        );
    }

    #[test]
    fn test_constant_kernel() {
        let ck = ConstantKernel::new(3.5);
        let x1 = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let x2 = Array2::from_shape_vec((3, 1), vec![2.0, 3.0, 4.0]).unwrap();
        let k = ck.compute(&x1, &x2);
        assert_eq!(k.shape(), &[2, 3]);
        for v in k.iter() {
            assert!((v - 3.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_white_kernel_diagonal() {
        let wk = WhiteKernel::new(0.5);
        let x = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let k = wk.compute(&x, &x);
        for i in 0..3 {
            assert!(
                (k[[i, i]] - 0.5).abs() < 1e-10,
                "diag[{}] = {}",
                i,
                k[[i, i]]
            );
            for j in 0..3 {
                if i != j {
                    assert!(
                        (k[[i, j]]).abs() < 1e-10,
                        "off-diag[{},{}] = {}",
                        i,
                        j,
                        k[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_sum_kernel() {
        let k1 = Box::new(ConstantKernel::new(1.0)) as Box<dyn Kernel>;
        let k2 = Box::new(ConstantKernel::new(2.0)) as Box<dyn Kernel>;
        let sk = SumKernel::new(k1, k2);
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let k = sk.compute(&x, &x);
        for v in k.iter() {
            assert!((v - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_product_kernel() {
        let k1 = Box::new(ConstantKernel::new(2.0)) as Box<dyn Kernel>;
        let k2 = Box::new(ConstantKernel::new(3.0)) as Box<dyn Kernel>;
        let pk = ProductKernel::new(k1, k2);
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let k = pk.compute(&x, &x);
        for v in k.iter() {
            assert!((v - 6.0).abs() < 1e-10);
        }
    }

    // =========================================================================
    // GPR tests
    // =========================================================================

    fn make_sin_data() -> (Array2<f64>, Array1<f64>) {
        let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
        let x = Array2::from_shape_vec((20, 1), xs.clone()).unwrap();
        let y = Array1::from_vec(xs.iter().map(|&v| v.sin()).collect());
        (x, y)
    }

    #[test]
    fn test_gpr_fit_sin_predictions_at_training() {
        let (x, y) = make_sin_data();
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        gpr.fit(&x, &y).unwrap();
        let preds = gpr.predict(&x).unwrap();
        // At training points, predictions should be very close to y
        for i in 0..y.len() {
            assert!(
                (preds[i] - y[i]).abs() < 1e-4,
                "at i={}: pred={}, y={}",
                i,
                preds[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_gpr_interpolation() {
        let (x, y) = make_sin_data();
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        gpr.fit(&x, &y).unwrap();

        // Predict at midpoints between training data
        let x_mid = Array2::from_shape_vec((3, 1), vec![0.15, 0.45, 0.75]).unwrap();
        let preds = gpr.predict(&x_mid).unwrap();
        // Should be reasonable interpolation of sin
        for i in 0..3 {
            let expected = [0.15_f64.sin(), 0.45_f64.sin(), 0.75_f64.sin()][i];
            assert!(
                (preds[i] - expected).abs() < 0.15,
                "at i={}: pred={}, expected={}",
                i,
                preds[i],
                expected
            );
        }
    }

    #[test]
    fn test_gpr_predict_with_std_small_near_training() {
        let (x, y) = make_sin_data();
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        gpr.fit(&x, &y).unwrap();
        let (_mean, std) = gpr.predict_with_std(&x).unwrap();
        // Std should be very small at training points
        for i in 0..y.len() {
            assert!(std[i] < 1e-3, "std at training point {}: {}", i, std[i]);
        }
    }

    #[test]
    fn test_gpr_predict_with_std_large_far_away() {
        let (x, y) = make_sin_data();
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        gpr.fit(&x, &y).unwrap();
        // Far from training data
        let x_far = Array2::from_shape_vec((1, 1), vec![100.0]).unwrap();
        let (_mean, std) = gpr.predict_with_std(&x_far).unwrap();
        // Std should be close to 1.0 (prior variance) far from data
        assert!(std[0] > 0.5, "std far away: {}", std[0]);
    }

    #[test]
    fn test_gpr_normalize_y() {
        let (x, y) = make_sin_data();
        let y_shifted = &y + 100.0;

        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0))).with_normalize_y(true);
        gpr.fit(&x, &y_shifted).unwrap();
        let preds = gpr.predict(&x).unwrap();
        for i in 0..y.len() {
            assert!(
                (preds[i] - y_shifted[i]).abs() < 1e-3,
                "at i={}: pred={}, y={}",
                i,
                preds[i],
                y_shifted[i]
            );
        }
    }

    #[test]
    fn test_gpr_log_marginal_likelihood_negative() {
        let (x, y) = make_sin_data();
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        gpr.fit(&x, &y).unwrap();
        let lml = gpr.log_marginal_likelihood().unwrap();
        // LML should be a finite number (can be positive or negative depending on data)
        assert!(lml.is_finite(), "LML is not finite: {}", lml);
    }

    #[test]
    fn test_gpr_alpha_smoothness() {
        // Larger alpha = smoother predictions (more regularization)
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 10.0, 0.0, 10.0, 0.0]); // noisy/oscillating

        let mut gpr_small =
            GaussianProcessRegressor::new(Box::new(RBF::new(1.0))).with_alpha(1e-10);
        gpr_small.fit(&x, &y).unwrap();
        let preds_small = gpr_small.predict(&x).unwrap();

        let mut gpr_large = GaussianProcessRegressor::new(Box::new(RBF::new(1.0))).with_alpha(10.0);
        gpr_large.fit(&x, &y).unwrap();
        let preds_large = gpr_large.predict(&x).unwrap();

        // With large alpha, predictions should be smoother (closer to mean)
        let range_small = preds_small
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - preds_small.iter().cloned().fold(f64::INFINITY, f64::min);
        let range_large = preds_large
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - preds_large.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            range_large < range_small,
            "large alpha range {} should be < small alpha range {}",
            range_large,
            range_small
        );
    }

    #[test]
    fn test_gpr_single_training_point() {
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let y = Array1::from_vec(vec![5.0]);
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        gpr.fit(&x, &y).unwrap();
        let pred = gpr.predict(&x).unwrap();
        assert!((pred[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_gpr_high_dimensional() {
        let n = 20;
        let d = 10;
        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            for j in 0..d {
                data.push((i * d + j) as f64 * 0.1);
            }
        }
        let x = Array2::from_shape_vec((n, d), data).unwrap();
        let y = Array1::from_vec((0..n).map(|i| (i as f64 * 0.5).sin()).collect());
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(2.0))).with_alpha(1e-5);
        gpr.fit(&x, &y).unwrap();
        let preds = gpr.predict(&x).unwrap();
        assert_eq!(preds.len(), n);
        // Should interpolate well
        for i in 0..n {
            assert!(
                (preds[i] - y[i]).abs() < 0.1,
                "at i={}: pred={}, y={}",
                i,
                preds[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_gpr_not_fitted_error() {
        let gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        assert!(gpr.predict(&x).is_err());
        assert!(gpr.predict_with_std(&x).is_err());
        assert!(gpr.log_marginal_likelihood().is_err());
    }

    #[test]
    fn test_gpr_clone_equivalence() {
        let (x, y) = make_sin_data();
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        gpr.fit(&x, &y).unwrap();
        let gpr2 = gpr.clone();
        let p1 = gpr.predict(&x).unwrap();
        let p2 = gpr2.predict(&x).unwrap();
        for i in 0..p1.len() {
            assert!((p1[i] - p2[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gpr_is_fitted() {
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        assert!(!gpr.is_fitted());
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0]);
        gpr.fit(&x, &y).unwrap();
        assert!(gpr.is_fitted());
    }

    #[test]
    fn test_gpr_with_matern_kernel() {
        let (x, y) = make_sin_data();
        let mut gpr = GaussianProcessRegressor::new(Box::new(Matern::new(1.0, 2.5)));
        gpr.fit(&x, &y).unwrap();
        let preds = gpr.predict(&x).unwrap();
        for i in 0..y.len() {
            assert!(
                (preds[i] - y[i]).abs() < 1e-3,
                "at i={}: pred={}, y={}",
                i,
                preds[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_gpr_predict_with_std_shapes() {
        let (x, y) = make_sin_data();
        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0)));
        gpr.fit(&x, &y).unwrap();
        let x_new = Array2::from_shape_vec((3, 1), vec![0.5, 1.5, 2.5]).unwrap();
        let (mean, std) = gpr.predict_with_std(&x_new).unwrap();
        assert_eq!(mean.len(), 3);
        assert_eq!(std.len(), 3);
        // All std should be non-negative
        for i in 0..3 {
            assert!(std[i] >= 0.0, "std[{}] = {}", i, std[i]);
        }
    }

    // =========================================================================
    // GPC tests
    // =========================================================================

    fn make_linearly_separable() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 1.5, 2.0, 2.0, 1.0, 1.0, 2.5, 2.5, 1.5, 6.0, 6.0, 6.5, 7.0, 7.0, 6.0,
                6.0, 7.5, 7.5, 6.5,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        (x, y)
    }

    #[test]
    fn test_gpc_linearly_separable() {
        let (x, y) = make_linearly_separable();
        let mut gpc = GaussianProcessClassifier::new(Box::new(RBF::new(1.0)));
        gpc.fit(&x, &y).unwrap();
        let preds = gpc.predict(&x).unwrap();
        let mut correct = 0;
        for i in 0..y.len() {
            if (preds[i] - y[i]).abs() < 1e-6 {
                correct += 1;
            }
        }
        assert_eq!(
            correct, 10,
            "Expected 100% accuracy on linearly separable data"
        );
    }

    #[test]
    fn test_gpc_predict_proba_shape_and_values() {
        let (x, y) = make_linearly_separable();
        let mut gpc = GaussianProcessClassifier::new(Box::new(RBF::new(1.0)));
        gpc.fit(&x, &y).unwrap();
        let probas = gpc.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[10, 2]);
        for i in 0..10 {
            let row_sum = probas[[i, 0]] + probas[[i, 1]];
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {} sums to {}",
                i,
                row_sum
            );
            assert!(probas[[i, 0]] >= 0.0 && probas[[i, 0]] <= 1.0);
            assert!(probas[[i, 1]] >= 0.0 && probas[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_gpc_xor_with_rbf() {
        // XOR pattern - not linearly separable but RBF can handle it
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 1.0, 0.0, 1.0, 0.1, 0.9, 0.0, 0.9, 0.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let mut gpc = GaussianProcessClassifier::new(Box::new(RBF::new(0.5)));
        gpc.fit(&x, &y).unwrap();
        let preds = gpc.predict(&x).unwrap();
        // Should get most right with RBF kernel
        let correct: usize = (0..y.len())
            .filter(|&i| (preds[i] - y[i]).abs() < 1e-6)
            .count();
        assert!(
            correct >= 6,
            "Expected at least 6/8 correct, got {}",
            correct
        );
    }

    #[test]
    fn test_gpc_not_fitted_error() {
        let gpc = GaussianProcessClassifier::new(Box::new(RBF::new(1.0)));
        let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        assert!(gpc.predict(&x).is_err());
        assert!(gpc.predict_proba(&x).is_err());
    }

    #[test]
    fn test_gpc_is_fitted() {
        let (x, y) = make_linearly_separable();
        let mut gpc = GaussianProcessClassifier::new(Box::new(RBF::new(1.0)));
        assert!(!gpc.is_fitted());
        gpc.fit(&x, &y).unwrap();
        assert!(gpc.is_fitted());
    }

    #[test]
    fn test_gpc_requires_two_classes() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0]); // only one class
        let mut gpc = GaussianProcessClassifier::new(Box::new(RBF::new(1.0)));
        assert!(gpc.fit(&x, &y).is_err());
    }

    #[test]
    fn test_kernel_clone_box() {
        let rbf: Box<dyn Kernel> = Box::new(RBF::new(1.5));
        let rbf2 = rbf.clone();
        let x = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let k1 = rbf.compute(&x, &x);
        let k2 = rbf2.compute(&x, &x);
        for i in 0..2 {
            for j in 0..2 {
                assert!((k1[[i, j]] - k2[[i, j]]).abs() < 1e-12);
            }
        }
    }
}
