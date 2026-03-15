//! Cross-library validation: FerroML vs linfa — Linear Models
//!
//! Tests that FerroML's linear model implementations produce results
//! consistent with linfa's implementations on identical data.

use ferroml_core::models::Model;
use ndarray::{Array1, Array2};

// ─── helpers ────────────────────────────────────────────────────────

fn synthetic_regression(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Normal};

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut x = Array2::zeros((n, p));
    for v in x.iter_mut() {
        *v = normal.sample(&mut rng);
    }

    let true_coef: Array1<f64> = (1..=p).map(|i| i as f64).collect();
    let noise: Array1<f64> = (0..n).map(|_| normal.sample(&mut rng) * 0.1).collect();
    let y = x.dot(&true_coef) + &noise;

    (x, y)
}

fn synthetic_classification(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let (x, y_raw) = synthetic_regression(n, p, seed);
    let y = y_raw.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
    (x, y)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0, f64::max)
}

fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred)
        .filter(|(&t, &p)| (t - p).abs() < 0.5)
        .count();
    correct as f64 / y_true.len() as f64
}

// ─── Linear Regression ──────────────────────────────────────────────

mod linear_regression {
    use super::*;
    use linfa::prelude::*;

    fn compare_at_size(n: usize, p: usize, atol: f64) {
        let (x, y) = synthetic_regression(n, p, 42);

        // FerroML
        let mut ferro_model = ferroml_core::models::LinearRegression::new();
        ferro_model.fit(&x, &y).expect("ferroml fit");
        let ferro_pred = ferro_model.predict(&x).expect("ferroml predict");

        // linfa
        let dataset = linfa::Dataset::new(x.clone(), y.clone());
        let linfa_model = linfa_linear::LinearRegression::default();
        let linfa_fitted = linfa_model.fit(&dataset).expect("linfa fit");
        let linfa_pred = linfa_fitted.predict(&x);

        let diff = max_abs_diff(
            ferro_pred.as_slice().unwrap(),
            linfa_pred.as_slice().unwrap(),
        );
        assert!(
            diff < atol,
            "LinearRegression predictions diverge: max_abs_diff={diff:.2e} > atol={atol:.2e} (n={n}, p={p})"
        );
    }

    // Tolerances scale with problem size because the condition number of X'X grows
    // with n and p, amplifying the difference between two closed-form solvers
    // that use different LAPACK routines (FerroML: QR, linfa: SVD-based).
    #[test]
    fn small() {
        compare_at_size(200, 10, 1e-6);
    }

    #[test]
    fn medium() {
        compare_at_size(1000, 50, 1e-4);
    }

    #[test]
    fn large() {
        compare_at_size(5000, 100, 1e-3);
    }
}

// ─── Ridge Regression ───────────────────────────────────────────────

mod ridge_regression {
    use super::*;
    use linfa::prelude::*;

    /// Note: linfa has no dedicated Ridge — we use ElasticNet(l1_ratio=0).
    /// The penalty parameterization differs (linfa uses coordinate descent, FerroML uses closed-form),
    /// so we compare R² and correlation rather than exact predictions.
    fn compare_ridge(n: usize, p: usize, alpha: f64) {
        let (x, y) = synthetic_regression(n, p, 42);

        // FerroML (closed-form Ridge)
        let mut ferro_model = ferroml_core::models::RidgeRegression::new(alpha);
        ferro_model.fit(&x, &y).expect("ferroml ridge fit");
        let ferro_pred = ferro_model.predict(&x).expect("ferroml ridge predict");
        let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa ElasticNet(l1_ratio=0) — coordinate descent approximation to Ridge
        let dataset = linfa::Dataset::new(x.clone(), y.clone());
        let linfa_model = linfa_elasticnet::ElasticNet::params()
            .penalty(alpha)
            .l1_ratio(0.01); // linfa rejects l1_ratio=0, use near-zero
        let linfa_fitted = linfa_model.fit(&dataset).expect("linfa fit");
        let linfa_pred = linfa_fitted.predict(&x);
        let linfa_r2 = r2_score(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

        // FerroML closed-form should achieve at least as good R²
        assert!(
            ferro_r2 > 0.80,
            "FerroML Ridge R² too low: {ferro_r2:.4} (alpha={alpha})"
        );

        // Correlation between predictions should be high
        let ferro_s = ferro_pred.as_slice().unwrap();
        let linfa_s = linfa_pred.as_slice().unwrap();
        let mean_f = ferro_s.iter().sum::<f64>() / n as f64;
        let mean_l = linfa_s.iter().sum::<f64>() / n as f64;
        let cov: f64 = ferro_s
            .iter()
            .zip(linfa_s)
            .map(|(f, l)| (f - mean_f) * (l - mean_l))
            .sum();
        let var_f: f64 = ferro_s.iter().map(|f| (f - mean_f).powi(2)).sum();
        let var_l: f64 = linfa_s.iter().map(|l| (l - mean_l).powi(2)).sum();
        let corr = if var_f > 0.0 && var_l > 0.0 {
            cov / (var_f.sqrt() * var_l.sqrt())
        } else {
            0.0
        };

        assert!(
            corr > 0.95,
            "Ridge prediction correlation too low: {corr:.4} (alpha={alpha}, ferro_r2={ferro_r2:.4}, linfa_r2={linfa_r2:.4})"
        );
    }

    #[test]
    fn alpha_0_01() {
        compare_ridge(500, 20, 0.01);
    }

    #[test]
    fn alpha_0_1() {
        compare_ridge(500, 20, 0.1);
    }

    #[test]
    fn alpha_1_0() {
        compare_ridge(500, 20, 1.0);
    }

    #[test]
    fn alpha_10_0() {
        compare_ridge(500, 20, 10.0);
    }
}

// ─── Lasso Regression ───────────────────────────────────────────────

mod lasso_regression {
    use super::*;
    use linfa::prelude::*;

    fn compare_lasso(n: usize, p: usize, alpha: f64) {
        let (x, y) = synthetic_regression(n, p, 42);

        // FerroML
        let mut ferro_model = ferroml_core::models::LassoRegression::new(alpha);
        ferro_model.fit(&x, &y).expect("ferroml lasso fit");
        let ferro_pred = ferro_model.predict(&x).expect("ferroml lasso predict");
        let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa — ElasticNet with l1_ratio=1.0 is Lasso
        let dataset = linfa::Dataset::new(x.clone(), y.clone());
        let linfa_model = linfa_elasticnet::ElasticNet::params()
            .penalty(alpha)
            .l1_ratio(1.0);
        let linfa_fitted = linfa_model
            .fit(&dataset)
            .expect("linfa elasticnet-as-lasso fit");
        let linfa_pred = linfa_fitted.predict(&x);
        let linfa_r2 = r2_score(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

        // Both should achieve reasonable R²
        assert!(ferro_r2 > 0.5, "FerroML Lasso R² too low: {ferro_r2:.4}");
        assert!(linfa_r2 > 0.5, "linfa Lasso R² too low: {linfa_r2:.4}");

        // R² gap should be small
        assert!(
            (ferro_r2 - linfa_r2).abs() < 0.10,
            "Lasso R² gap too large: ferro={ferro_r2:.4}, linfa={linfa_r2:.4}"
        );
    }

    #[test]
    fn alpha_0_01() {
        compare_lasso(500, 20, 0.01);
    }

    #[test]
    fn alpha_0_1() {
        compare_lasso(500, 20, 0.1);
    }

    #[test]
    fn alpha_1_0() {
        compare_lasso(500, 20, 1.0);
    }
}

// ─── ElasticNet ─────────────────────────────────────────────────────

mod elastic_net {
    use super::*;
    use linfa::prelude::*;

    fn compare_elasticnet(n: usize, p: usize, alpha: f64, l1_ratio: f64) {
        let (x, y) = synthetic_regression(n, p, 42);

        // FerroML
        let mut ferro_model = ferroml_core::models::ElasticNet::new(alpha, l1_ratio);
        ferro_model.fit(&x, &y).expect("ferroml elasticnet fit");
        let ferro_pred = ferro_model.predict(&x).expect("ferroml elasticnet predict");
        let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa
        let dataset = linfa::Dataset::new(x.clone(), y.clone());
        let linfa_model = linfa_elasticnet::ElasticNet::params()
            .penalty(alpha)
            .l1_ratio(l1_ratio);
        let linfa_fitted = linfa_model.fit(&dataset).expect("linfa elasticnet fit");
        let linfa_pred = linfa_fitted.predict(&x);
        let linfa_r2 = r2_score(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

        assert!(
            ferro_r2 > 0.5,
            "FerroML ElasticNet R² too low: {ferro_r2:.4}"
        );
        assert!(linfa_r2 > 0.5, "linfa ElasticNet R² too low: {linfa_r2:.4}");
        assert!(
            (ferro_r2 - linfa_r2).abs() < 0.10,
            "ElasticNet R² gap: ferro={ferro_r2:.4}, linfa={linfa_r2:.4} (alpha={alpha}, l1={l1_ratio})"
        );
    }

    #[test]
    fn l1_ratio_0_1() {
        compare_elasticnet(500, 20, 0.1, 0.1);
    }

    #[test]
    fn l1_ratio_0_5() {
        compare_elasticnet(500, 20, 0.1, 0.5);
    }

    #[test]
    fn l1_ratio_0_9() {
        compare_elasticnet(500, 20, 0.1, 0.9);
    }
}

// ─── Logistic Regression ────────────────────────────────────────────

mod logistic_regression {
    use super::*;
    use linfa::prelude::*;

    fn compare_logreg(n: usize, p: usize) {
        let (x, y) = synthetic_classification(n, p, 42);

        // FerroML
        let mut ferro_model = ferroml_core::models::LogisticRegression::new();
        ferro_model.fit(&x, &y).expect("ferroml logreg fit");
        let ferro_pred = ferro_model.predict(&x).expect("ferroml logreg predict");
        let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa-logistic
        let dataset = linfa::Dataset::new(x.clone(), y.mapv(|v| v > 0.5));
        let linfa_model = linfa_logistic::LogisticRegression::default();
        let linfa_fitted = linfa_model.fit(&dataset).expect("linfa logreg fit");
        let linfa_pred_bool = linfa_fitted.predict(&x);
        let linfa_pred: Array1<f64> = linfa_pred_bool.mapv(|b| if b { 1.0 } else { 0.0 });
        let linfa_acc = accuracy(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

        assert!(
            ferro_acc > 0.80,
            "FerroML LogReg accuracy too low: {ferro_acc:.3}"
        );
        assert!(
            linfa_acc > 0.80,
            "linfa LogReg accuracy too low: {linfa_acc:.3}"
        );
        assert!(
            (ferro_acc - linfa_acc).abs() < 0.10,
            "LogReg accuracy gap: ferro={ferro_acc:.3}, linfa={linfa_acc:.3}"
        );
    }

    #[test]
    fn small() {
        compare_logreg(200, 10);
    }

    #[test]
    fn medium() {
        compare_logreg(1000, 20);
    }

    #[test]
    fn large() {
        compare_logreg(5000, 50);
    }
}
