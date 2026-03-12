//! Tests for MultiOutputRegressor and MultiOutputClassifier wrappers.
//!
//! These tests verify that the multi-output wrappers correctly delegate
//! to per-column estimator clones and produce correct shapes and values.

#[cfg(test)]
mod tests {
    use crate::models::multioutput::{MultiOutputClassifier, MultiOutputRegressor};
    use crate::models::{
        DecisionTreeRegressor, KNeighborsRegressor, LinearRegression, LogisticRegression, Model,
        RidgeRegression,
    };
    use ndarray::{Array1, Array2};

    /// Generate deterministic multi-output regression data.
    /// y[:, k] = sum_j(x[i,j] * (k+1) * (1 + j*0.5)) + noise
    fn make_multioutput_data(
        n_samples: usize,
        n_features: usize,
        n_outputs: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array2::zeros((n_samples, n_outputs));
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = ((i * 7 + j * 13 + 3) % 100) as f64 / 50.0 - 1.0;
            }
            for k in 0..n_outputs {
                let mut val = 0.0;
                for j in 0..n_features {
                    val += x[[i, j]] * ((k + 1) as f64) * (1.0 + j as f64 * 0.5);
                }
                val += ((i * 3 + k * 7) % 17) as f64 / 170.0;
                y[[i, k]] = val;
            }
        }
        (x, y)
    }

    /// Generate deterministic multi-label classification data (binary labels per column).
    fn make_multilabel_data(
        n_samples: usize,
        n_features: usize,
        n_outputs: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array2::zeros((n_samples, n_outputs));
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = ((i * 11 + j * 7 + 5) % 100) as f64 / 50.0 - 1.0;
            }
            for k in 0..n_outputs {
                // Threshold a linear combination to get binary labels
                let mut val = 0.0;
                for j in 0..n_features {
                    val += x[[i, j]] * ((k + 1) as f64) * (1.0 + j as f64 * 0.3);
                }
                y[[i, k]] = if val > 0.0 { 1.0 } else { 0.0 };
            }
        }
        (x, y)
    }

    #[test]
    fn test_linear_regression_multi_output() {
        let (x, y) = make_multioutput_data(50, 3, 3);
        let mut mo = MultiOutputRegressor::new(LinearRegression::new());
        mo.fit_multi(&x, &y).unwrap();

        let preds = mo.predict_multi(&x).unwrap();
        assert_eq!(preds.shape(), &[50, 3]);

        // Verify that each column matches fitting a single model on that column
        for k in 0..3 {
            let y_col: Array1<f64> = y.column(k).to_owned();
            let mut single = LinearRegression::new();
            single.fit(&x, &y_col).unwrap();
            let single_pred = single.predict(&x).unwrap();
            for i in 0..50 {
                assert!(
                    (preds[[i, k]] - single_pred[i]).abs() < 1e-10,
                    "Mismatch at sample {}, output {}",
                    i,
                    k
                );
            }
        }
    }

    #[test]
    fn test_decision_tree_regressor_multi_output() {
        let (x, y) = make_multioutput_data(40, 2, 2);
        let mut mo =
            MultiOutputRegressor::new(DecisionTreeRegressor::new().with_max_depth(Some(5)));
        mo.fit_multi(&x, &y).unwrap();

        let preds = mo.predict_multi(&x).unwrap();
        assert_eq!(preds.shape(), &[40, 2]);
        assert!(mo.is_fitted());
        assert_eq!(mo.n_outputs(), Some(2));
    }

    #[test]
    fn test_knn_regressor_multi_output() {
        let (x, y) = make_multioutput_data(30, 3, 2);
        let mut mo = MultiOutputRegressor::new(KNeighborsRegressor::new(3));
        mo.fit_multi(&x, &y).unwrap();

        let preds = mo.predict_multi(&x).unwrap();
        assert_eq!(preds.shape(), &[30, 2]);
    }

    #[test]
    fn test_ridge_regression_multi_output() {
        let (x, y) = make_multioutput_data(50, 4, 3);
        let mut mo = MultiOutputRegressor::new(RidgeRegression::new(1.0));
        mo.fit_multi(&x, &y).unwrap();

        let preds = mo.predict_multi(&x).unwrap();
        assert_eq!(preds.shape(), &[50, 3]);

        // Check per-column match
        for k in 0..3 {
            let y_col: Array1<f64> = y.column(k).to_owned();
            let mut single = RidgeRegression::new(1.0);
            single.fit(&x, &y_col).unwrap();
            let single_pred = single.predict(&x).unwrap();
            for i in 0..50 {
                assert!(
                    (preds[[i, k]] - single_pred[i]).abs() < 1e-10,
                    "Ridge mismatch at sample {}, output {}",
                    i,
                    k
                );
            }
        }
    }

    #[test]
    fn test_multioutput_classifier_logistic() {
        let (x, y) = make_multilabel_data(60, 4, 3);
        let mut mo = MultiOutputClassifier::new(LogisticRegression::new());
        mo.fit_multi(&x, &y).unwrap();

        let preds = mo.predict_multi(&x).unwrap();
        assert_eq!(preds.shape(), &[60, 3]);

        // All predictions should be 0 or 1
        for i in 0..60 {
            for k in 0..3 {
                assert!(
                    preds[[i, k]] == 0.0 || preds[[i, k]] == 1.0,
                    "Expected binary prediction, got {}",
                    preds[[i, k]]
                );
            }
        }
    }

    #[test]
    fn test_multioutput_classifier_predict_proba_shape() {
        let (x, y) = make_multilabel_data(40, 3, 2);
        let mut mo = MultiOutputClassifier::new(LogisticRegression::new());
        mo.fit_multi(&x, &y).unwrap();

        let probas = mo.predict_proba_multi(&x).unwrap();
        assert_eq!(probas.len(), 2, "Should have one proba array per output");
        for proba in &probas {
            assert_eq!(proba.nrows(), 40);
            assert!(
                proba.ncols() == 2,
                "Binary classification should yield 2 columns"
            );
            // Each row should sum to ~1
            for i in 0..proba.nrows() {
                let row_sum: f64 = proba.row(i).sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-6,
                    "Proba row {} sums to {}",
                    i,
                    row_sum
                );
            }
        }
    }

    #[test]
    fn test_empty_y_rejected() {
        let x = Array2::zeros((10, 3));
        let y = Array2::zeros((10, 0));
        let mut mo = MultiOutputRegressor::new(LinearRegression::new());
        let result = mo.fit_multi(&x, &y);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("at least one output column"), "Got: {}", msg);
    }

    #[test]
    fn test_single_output_degeneracy() {
        let (x, y) = make_multioutput_data(30, 2, 1);
        let mut mo = MultiOutputRegressor::new(LinearRegression::new());
        mo.fit_multi(&x, &y).unwrap();

        let preds = mo.predict_multi(&x).unwrap();
        assert_eq!(preds.shape(), &[30, 1]);
        assert_eq!(mo.n_outputs(), Some(1));

        // Should match single-model fit
        let y_col: Array1<f64> = y.column(0).to_owned();
        let mut single = LinearRegression::new();
        single.fit(&x, &y_col).unwrap();
        let single_pred = single.predict(&x).unwrap();
        for i in 0..30 {
            assert!((preds[[i, 0]] - single_pred[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_predict_before_fit_raises_not_fitted() {
        let x = Array2::zeros((5, 3));
        let mo = MultiOutputRegressor::new(LinearRegression::new());
        let result = mo.predict_multi(&x);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.to_lowercase().contains("not fitted")
                || msg.to_lowercase().contains("not been fitted"),
            "Expected NotFitted error, got: {}",
            msg
        );
    }

    #[test]
    fn test_classifier_predict_before_fit_raises_not_fitted() {
        let x = Array2::zeros((5, 3));
        let mo = MultiOutputClassifier::new(LogisticRegression::new());
        let result = mo.predict_multi(&x);
        assert!(result.is_err());

        let result_proba = mo.predict_proba_multi(&x);
        assert!(result_proba.is_err());
    }

    #[test]
    fn test_clone_equivalence() {
        let (x, y) = make_multioutput_data(30, 3, 2);
        let base = MultiOutputRegressor::new(LinearRegression::new());

        let mut mo1 = base.clone();
        mo1.fit_multi(&x, &y).unwrap();
        let preds1 = mo1.predict_multi(&x).unwrap();

        let mut mo2 = base.clone();
        mo2.fit_multi(&x, &y).unwrap();
        let preds2 = mo2.predict_multi(&x).unwrap();

        assert_eq!(preds1.shape(), preds2.shape());
        for i in 0..preds1.nrows() {
            for j in 0..preds1.ncols() {
                assert!(
                    (preds1[[i, j]] - preds2[[i, j]]).abs() < 1e-10,
                    "Clone divergence at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_per_output_mse() {
        let (x, y) = make_multioutput_data(50, 3, 3);
        let mut mo = MultiOutputRegressor::new(LinearRegression::new());
        mo.fit_multi(&x, &y).unwrap();
        let preds = mo.predict_multi(&x).unwrap();

        // Compute MSE per output column
        for k in 0..3 {
            let mut mse = 0.0;
            for i in 0..50 {
                let diff = preds[[i, k]] - y[[i, k]];
                mse += diff * diff;
            }
            mse /= 50.0;
            // Linear regression on linearly generated data should fit well
            assert!(mse < 0.01, "MSE for output {} is too high: {}", k, mse);
        }
    }

    #[test]
    fn test_large_multi_output_10_columns() {
        let (x, y) = make_multioutput_data(100, 5, 10);
        let mut mo = MultiOutputRegressor::new(LinearRegression::new());
        mo.fit_multi(&x, &y).unwrap();

        let preds = mo.predict_multi(&x).unwrap();
        assert_eq!(preds.shape(), &[100, 10]);
        assert_eq!(mo.n_outputs(), Some(10));
        assert_eq!(mo.estimators().unwrap().len(), 10);
    }

    #[test]
    fn test_predict_proba_not_supported() {
        // DecisionTreeRegressor does not support predict_proba
        let (x, y) = make_multilabel_data(30, 3, 2);
        let mut mo =
            MultiOutputClassifier::new(DecisionTreeRegressor::new().with_max_depth(Some(3)));
        // It will fit fine (binary labels are valid float targets)
        mo.fit_multi(&x, &y).unwrap();

        // But predict_proba should fail
        let result = mo.predict_proba_multi(&x);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("predict_proba"),
            "Expected NotImplemented for predict_proba, got: {}",
            msg
        );
    }
}
