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

    // =========================================================================
    // Inducing point selection tests (Phase S.1)
    // =========================================================================

    #[test]
    fn test_inducing_random_subset_shape() {
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64 * 0.1).collect()).unwrap();
        let kernel = RBF::new(1.0);
        let z = select_inducing_points(
            &x,
            10,
            &InducingPointMethod::RandomSubset { seed: Some(42) },
            &kernel,
        )
        .unwrap();
        assert_eq!(z.shape(), &[10, 2]);
    }

    #[test]
    fn test_inducing_random_subset_from_data() {
        let x = Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64).collect()).unwrap();
        let kernel = RBF::new(1.0);
        let z = select_inducing_points(
            &x,
            5,
            &InducingPointMethod::RandomSubset { seed: Some(42) },
            &kernel,
        )
        .unwrap();
        // Each inducing point should be an actual row from X
        for i in 0..5 {
            let row = z.row(i);
            let found = (0..20).any(|j| {
                let xr = x.row(j);
                (row[0] - xr[0]).abs() < 1e-12 && (row[1] - xr[1]).abs() < 1e-12
            });
            assert!(found, "inducing point {} not found in X", i);
        }
    }

    #[test]
    fn test_inducing_kmeans_shape() {
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64 * 0.1).collect()).unwrap();
        let kernel = RBF::new(1.0);
        let z = select_inducing_points(
            &x,
            10,
            &InducingPointMethod::KMeans {
                max_iter: 100,
                seed: Some(42),
            },
            &kernel,
        )
        .unwrap();
        assert_eq!(z.shape(), &[10, 2]);
    }

    #[test]
    fn test_inducing_kmeans_in_data_range() {
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64 * 0.1).collect()).unwrap();
        let kernel = RBF::new(1.0);
        let z = select_inducing_points(
            &x,
            10,
            &InducingPointMethod::KMeans {
                max_iter: 100,
                seed: Some(42),
            },
            &kernel,
        )
        .unwrap();
        // Centers should be within the range of data
        for i in 0..10 {
            for j in 0..2 {
                let min_x = x.column(j).iter().cloned().fold(f64::INFINITY, f64::min);
                let max_x = x
                    .column(j)
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                assert!(
                    z[[i, j]] >= min_x - 1.0 && z[[i, j]] <= max_x + 1.0,
                    "center [{},{}] = {} outside range [{}, {}]",
                    i,
                    j,
                    z[[i, j]],
                    min_x,
                    max_x
                );
            }
        }
    }

    #[test]
    fn test_inducing_greedy_variance_shape() {
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| i as f64 * 0.1).collect()).unwrap();
        let kernel = RBF::new(1.0);
        let z = select_inducing_points(
            &x,
            10,
            &InducingPointMethod::GreedyVariance { seed: Some(42) },
            &kernel,
        )
        .unwrap();
        assert_eq!(z.shape(), &[10, 2]);
    }

    #[test]
    fn test_inducing_greedy_variance_spread() {
        // Greedy variance should pick spread-out points
        let x = Array2::from_shape_vec((50, 1), (0..50).map(|i| i as f64 * 0.2).collect()).unwrap();
        let kernel = RBF::new(1.0);
        let z = select_inducing_points(
            &x,
            5,
            &InducingPointMethod::GreedyVariance { seed: Some(42) },
            &kernel,
        )
        .unwrap();
        // Check minimum pairwise distance is reasonable
        let mut min_dist = f64::INFINITY;
        for i in 0..5 {
            for j in (i + 1)..5 {
                let d = (z[[i, 0]] - z[[j, 0]]).abs();
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        // With 5 points over range [0, 9.8], min distance should be > 1
        assert!(
            min_dist > 0.5,
            "greedy variance points too close: min_dist = {}",
            min_dist
        );
    }

    #[test]
    fn test_inducing_m_greater_than_n_clamped() {
        let x = Array2::from_shape_vec((5, 2), (0..10).map(|i| i as f64).collect()).unwrap();
        let kernel = RBF::new(1.0);
        let z = select_inducing_points(
            &x,
            100, // m > n
            &InducingPointMethod::RandomSubset { seed: Some(42) },
            &kernel,
        )
        .unwrap();
        assert_eq!(z.nrows(), 5); // clamped to n
    }

    // =========================================================================
    // SparseGPRegressor tests (Phase S.2 / S.3)
    // =========================================================================

    fn make_sin_data_100() -> (Array2<f64>, Array1<f64>) {
        let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.06).collect();
        let x = Array2::from_shape_vec((100, 1), xs.clone()).unwrap();
        let y = Array1::from_vec(xs.iter().map(|&v| v.sin()).collect());
        (x, y)
    }

    fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let mean = y_true.mean().unwrap();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum();
        let ss_tot: f64 = y_true.iter().map(|&v| (v - mean).powi(2)).sum();
        1.0 - ss_res / ss_tot
    }

    #[test]
    fn test_sgpr_fitc_matches_exact_small_data() {
        // With m=n, FITC should closely match exact GP
        let (x, y) = make_sin_data();
        let n = x.nrows();

        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0))).with_alpha(0.01);
        gpr.fit(&x, &y).unwrap();
        let exact_preds = gpr.predict(&x).unwrap();

        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(n)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let sparse_preds = sgpr.predict(&x).unwrap();

        for i in 0..n {
            assert!(
                (exact_preds[i] - sparse_preds[i]).abs() < 0.5,
                "at i={}: exact={}, sparse={}",
                i,
                exact_preds[i],
                sparse_preds[i]
            );
        }
    }

    #[test]
    fn test_sgpr_fitc_prediction_mean_reasonable() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(20)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let preds = sgpr.predict(&x).unwrap();
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.8, "R^2 = {} (expected > 0.8)", r2);
    }

    #[test]
    fn test_sgpr_fitc_uncertainty_near_training_small() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(20)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let (_mean, std) = sgpr.predict_with_std(&x).unwrap();
        let mean_std = std.mean().unwrap();
        assert!(
            mean_std < 1.0,
            "mean std at training points = {} (expected < 1.0)",
            mean_std
        );
    }

    #[test]
    fn test_sgpr_fitc_uncertainty_far_away_large() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(20)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let x_far = Array2::from_shape_vec((1, 1), vec![100.0]).unwrap();
        let (_mean, std) = sgpr.predict_with_std(&x_far).unwrap();
        assert!(
            std[0] > 0.3,
            "std far from data = {} (expected > 0.3)",
            std[0]
        );
    }

    #[test]
    fn test_sgpr_fitc_normalize_y() {
        let (x, y) = make_sin_data_100();
        let y_shifted = &y + 1000.0;
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(20)
            .with_normalize_y(true)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y_shifted).unwrap();
        let preds = sgpr.predict(&x).unwrap();
        let r2 = r_squared(&y_shifted, &preds);
        assert!(r2 > 0.8, "R^2 with normalize_y = {} (expected > 0.8)", r2);
    }

    #[test]
    fn test_sgpr_fitc_log_marginal_likelihood_finite() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(20)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let lml = sgpr.log_marginal_likelihood().unwrap();
        assert!(lml.is_finite(), "LML is not finite: {}", lml);
    }

    #[test]
    fn test_sgpr_fitc_different_kernels() {
        let (x, y) = make_sin_data_100();
        for kernel in [
            Box::new(RBF::new(1.0)) as Box<dyn Kernel>,
            Box::new(Matern::new(1.0, 2.5)) as Box<dyn Kernel>,
        ] {
            let mut sgpr = SparseGPRegressor::new(kernel)
                .with_alpha(0.01)
                .with_n_inducing(20)
                .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
                .with_approximation(SparseApproximation::FITC);
            sgpr.fit(&x, &y).unwrap();
            let preds = sgpr.predict(&x).unwrap();
            let r2 = r_squared(&y, &preds);
            assert!(r2 > 0.5, "R^2 with different kernel = {}", r2);
        }
    }

    #[test]
    fn test_sgpr_fitc_fewer_inducing_still_works() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(5)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let preds = sgpr.predict(&x).unwrap();
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.3, "R^2 with m=5 = {} (expected > 0.3)", r2);
    }

    #[test]
    fn test_sgpr_vfe_matches_exact_small_data() {
        let (x, y) = make_sin_data();
        let n = x.nrows();

        let mut gpr = GaussianProcessRegressor::new(Box::new(RBF::new(1.0))).with_alpha(0.01);
        gpr.fit(&x, &y).unwrap();
        let exact_preds = gpr.predict(&x).unwrap();

        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(n)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::VFE);
        sgpr.fit(&x, &y).unwrap();
        let sparse_preds = sgpr.predict(&x).unwrap();

        for i in 0..n {
            assert!(
                (exact_preds[i] - sparse_preds[i]).abs() < 0.5,
                "VFE at i={}: exact={}, sparse={}",
                i,
                exact_preds[i],
                sparse_preds[i]
            );
        }
    }

    #[test]
    fn test_sgpr_vfe_prediction_reasonable() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(20)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::VFE);
        sgpr.fit(&x, &y).unwrap();
        let preds = sgpr.predict(&x).unwrap();
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.8, "VFE R^2 = {} (expected > 0.8)", r2);
    }

    #[test]
    fn test_sgpr_vfe_more_inducing_better_lml() {
        let (x, y) = make_sin_data_100();
        let mut sgpr10 = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(10)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::VFE);
        sgpr10.fit(&x, &y).unwrap();
        let lml10 = sgpr10.log_marginal_likelihood().unwrap();

        let mut sgpr30 = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(30)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::VFE);
        sgpr30.fit(&x, &y).unwrap();
        let lml30 = sgpr30.log_marginal_likelihood().unwrap();

        assert!(
            lml30 > lml10,
            "more inducing should give higher LML: lml30={}, lml10={}",
            lml30,
            lml10
        );
    }

    #[test]
    fn test_sgpr_single_inducing_point() {
        let (x, y) = make_sin_data();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(1)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        // Should not crash
        sgpr.fit(&x, &y).unwrap();
        let preds = sgpr.predict(&x).unwrap();
        assert_eq!(preds.len(), 20);
    }

    #[test]
    fn test_sgpr_not_fitted_error() {
        let sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)));
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        assert!(sgpr.predict(&x).is_err());
        assert!(sgpr.predict_with_std(&x).is_err());
        assert!(sgpr.log_marginal_likelihood().is_err());
    }

    #[test]
    fn test_sgpr_empty_data_error() {
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)));
        let x = Array2::zeros((0, 1));
        let y = Array1::zeros(0);
        assert!(sgpr.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgpr_shape_mismatch() {
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)));
        let x = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 1.0, 2.0]); // wrong length
        assert!(sgpr.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgpr_random_subset_method() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(15)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(123) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let preds = sgpr.predict(&x).unwrap();
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.5, "random subset R^2 = {}", r2);
    }

    #[test]
    fn test_sgpr_kmeans_method() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(15)
            .with_inducing_method(InducingPointMethod::KMeans {
                max_iter: 100,
                seed: Some(42),
            })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let preds = sgpr.predict(&x).unwrap();
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.5, "kmeans R^2 = {}", r2);
    }

    #[test]
    fn test_sgpr_greedy_variance_method() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(15)
            .with_inducing_method(InducingPointMethod::GreedyVariance { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        sgpr.fit(&x, &y).unwrap();
        let preds = sgpr.predict(&x).unwrap();
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.5, "greedy variance R^2 = {}", r2);
    }

    #[test]
    fn test_sgpr_inducing_points_accessible() {
        let (x, y) = make_sin_data_100();
        let mut sgpr = SparseGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_alpha(0.01)
            .with_n_inducing(15)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) })
            .with_approximation(SparseApproximation::FITC);
        assert!(sgpr.inducing_points().is_none());
        sgpr.fit(&x, &y).unwrap();
        let z = sgpr.inducing_points().unwrap();
        assert_eq!(z.shape(), &[15, 1]);
    }

    // =========================================================================
    // SparseGPClassifier tests (Phase S.4)
    // =========================================================================

    #[test]
    fn test_sgpc_linearly_separable() {
        let (x, y) = make_linearly_separable();
        let mut sgpc = SparseGPClassifier::new(Box::new(RBF::new(1.0)))
            .with_n_inducing(10)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
        sgpc.fit(&x, &y).unwrap();
        let preds = sgpc.predict(&x).unwrap();
        let correct: usize = (0..y.len())
            .filter(|&i| (preds[i] - y[i]).abs() < 1e-6)
            .count();
        assert!(
            correct >= 8,
            "Expected at least 8/10 correct, got {}",
            correct
        );
    }

    #[test]
    fn test_sgpc_predict_proba_valid() {
        let (x, y) = make_linearly_separable();
        let mut sgpc = SparseGPClassifier::new(Box::new(RBF::new(1.0)))
            .with_n_inducing(10)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
        sgpc.fit(&x, &y).unwrap();
        let probas = sgpc.predict_proba(&x).unwrap();
        assert_eq!(probas.shape(), &[10, 2]);
        for i in 0..10 {
            let sum = probas[[i, 0]] + probas[[i, 1]];
            assert!((sum - 1.0).abs() < 1e-6, "row {} sums to {}", i, sum);
            assert!(probas[[i, 0]] >= 0.0 && probas[[i, 0]] <= 1.0);
            assert!(probas[[i, 1]] >= 0.0 && probas[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_sgpc_not_fitted_error() {
        let sgpc = SparseGPClassifier::new(Box::new(RBF::new(1.0)));
        let x = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        assert!(sgpc.predict(&x).is_err());
        assert!(sgpc.predict_proba(&x).is_err());
    }

    #[test]
    fn test_sgpc_requires_two_classes() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let mut sgpc = SparseGPClassifier::new(Box::new(RBF::new(1.0)));
        assert!(sgpc.fit(&x, &y).is_err());
    }

    #[test]
    fn test_sgpc_different_inducing_methods() {
        let (x, y) = make_linearly_separable();
        for method in [
            InducingPointMethod::RandomSubset { seed: Some(42) },
            InducingPointMethod::KMeans {
                max_iter: 50,
                seed: Some(42),
            },
            InducingPointMethod::GreedyVariance { seed: Some(42) },
        ] {
            let mut sgpc = SparseGPClassifier::new(Box::new(RBF::new(1.0)))
                .with_n_inducing(8)
                .with_inducing_method(method);
            sgpc.fit(&x, &y).unwrap();
            let preds = sgpc.predict(&x).unwrap();
            assert_eq!(preds.len(), 10);
        }
    }

    #[test]
    fn test_sgpc_inducing_points_accessible() {
        let (x, y) = make_linearly_separable();
        let mut sgpc = SparseGPClassifier::new(Box::new(RBF::new(1.0)))
            .with_n_inducing(5)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
        assert!(sgpc.inducing_points().is_none());
        sgpc.fit(&x, &y).unwrap();
        let z = sgpc.inducing_points().unwrap();
        assert_eq!(z.shape(), &[5, 2]);
    }

    #[test]
    fn test_sgpc_xor_pattern() {
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 1.0, 0.0, 1.0, 0.1, 0.9, 0.0, 0.9, 0.1,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let mut sgpc = SparseGPClassifier::new(Box::new(RBF::new(0.5)))
            .with_n_inducing(8)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
        sgpc.fit(&x, &y).unwrap();
        let preds = sgpc.predict(&x).unwrap();
        let correct: usize = (0..y.len())
            .filter(|&i| (preds[i] - y[i]).abs() < 1e-6)
            .count();
        assert!(
            correct >= 5,
            "Expected at least 5/8 correct for XOR, got {}",
            correct
        );
    }

    // =========================================================================
    // SVGPRegressor tests (Phase S.5)
    // =========================================================================

    #[test]
    fn test_svgp_basic_sin_regression() {
        let xs: Vec<f64> = (0..200).map(|i| i as f64 * 0.03).collect();
        let x = Array2::from_shape_vec((200, 1), xs.clone()).unwrap();
        let y = Array1::from_vec(xs.iter().map(|&v| v.sin()).collect());

        let mut svgp = SVGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_noise_variance(0.01)
            .with_n_inducing(30)
            .with_n_epochs(50)
            .with_batch_size(200)
            .with_learning_rate(0.1)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
        svgp.fit(&x, &y).unwrap();
        let preds = svgp.predict(&x).unwrap();
        let r2 = r_squared(&y, &preds);
        assert!(r2 > 0.5, "SVGP R^2 = {} (expected > 0.5)", r2);
    }

    #[test]
    fn test_svgp_predict_with_std_shapes() {
        let (x, y) = make_sin_data_100();
        let mut svgp = SVGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_noise_variance(0.01)
            .with_n_inducing(15)
            .with_n_epochs(20)
            .with_batch_size(100)
            .with_learning_rate(0.1)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
        svgp.fit(&x, &y).unwrap();
        let x_test = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0]).unwrap();
        let (mean, std) = svgp.predict_with_std(&x_test).unwrap();
        assert_eq!(mean.len(), 5);
        assert_eq!(std.len(), 5);
        for i in 0..5 {
            assert!(std[i] >= 0.0, "std[{}] = {}", i, std[i]);
        }
    }

    #[test]
    fn test_svgp_not_fitted_error() {
        let svgp = SVGPRegressor::new(Box::new(RBF::new(1.0)));
        let x = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        assert!(svgp.predict(&x).is_err());
        assert!(svgp.predict_with_std(&x).is_err());
    }

    #[test]
    fn test_svgp_different_batch_sizes() {
        let (x, y) = make_sin_data_100();
        for bs in [32, 50, 100] {
            let mut svgp = SVGPRegressor::new(Box::new(RBF::new(1.0)))
                .with_noise_variance(0.01)
                .with_n_inducing(15)
                .with_n_epochs(20)
                .with_batch_size(bs)
                .with_learning_rate(0.1)
                .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
            svgp.fit(&x, &y).unwrap();
            let preds = svgp.predict(&x).unwrap();
            assert_eq!(preds.len(), 100);
        }
    }

    #[test]
    fn test_svgp_inducing_points_accessible() {
        let (x, y) = make_sin_data_100();
        let mut svgp = SVGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_noise_variance(0.01)
            .with_n_inducing(15)
            .with_n_epochs(10)
            .with_batch_size(100)
            .with_learning_rate(0.1)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
        assert!(svgp.inducing_points().is_none());
        svgp.fit(&x, &y).unwrap();
        let z = svgp.inducing_points().unwrap();
        assert_eq!(z.shape(), &[15, 1]);
    }

    #[test]
    fn test_svgp_uncertainty_increases_away() {
        let (x, y) = make_sin_data_100();
        let mut svgp = SVGPRegressor::new(Box::new(RBF::new(1.0)))
            .with_noise_variance(0.01)
            .with_n_inducing(20)
            .with_n_epochs(30)
            .with_batch_size(100)
            .with_learning_rate(0.1)
            .with_inducing_method(InducingPointMethod::RandomSubset { seed: Some(42) });
        svgp.fit(&x, &y).unwrap();

        let x_near = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap(); // within training range
        let x_far = Array2::from_shape_vec((1, 1), vec![50.0]).unwrap(); // far from training
        let (_, std_near) = svgp.predict_with_std(&x_near).unwrap();
        let (_, std_far) = svgp.predict_with_std(&x_far).unwrap();

        assert!(
            std_far[0] > std_near[0],
            "std far ({}) should be > std near ({})",
            std_far[0],
            std_near[0]
        );
    }
}
