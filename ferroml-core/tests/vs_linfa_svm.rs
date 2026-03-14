//! Cross-library validation: FerroML vs linfa — SVM

use ferroml_core::models::Model;
use ndarray::{Array1, Array2};

fn synthetic_classification(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Normal};
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut x = Array2::zeros((n, p));
    for i in 0..n {
        let class = if i < n / 2 { 0.0 } else { 1.0 };
        for j in 0..p {
            x[[i, j]] = normal.sample(&mut rng) + class * 2.0;
        }
    }
    let y: Array1<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
    (x, y)
}

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
    let noise: Array1<f64> = (0..n).map(|_| normal.sample(&mut rng) * 0.5).collect();
    let y = x.dot(&true_coef) + &noise;
    (x, y)
}

fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred)
        .filter(|(&t, &p)| (t - p).abs() < 0.5)
        .count();
    correct as f64 / y_true.len() as f64
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

// ─── SVC Linear ─────────────────────────────────────────────────────

mod svc_linear {
    use super::*;
    use linfa::prelude::*;
    use linfa_svm::Svm;

    fn compare(n: usize, p: usize) {
        let (x, y) = synthetic_classification(n, p, 42);

        // FerroML
        let mut ferro = ferroml_core::models::SVC::new()
            .with_kernel(ferroml_core::models::Kernel::Linear)
            .with_c(1.0);
        ferro.fit(&x, &y).unwrap();
        let ferro_pred = ferro.predict(&x).unwrap();
        let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa SVM
        let y_bool: Array1<bool> = y.mapv(|v| v > 0.5);
        let dataset = linfa::Dataset::new(x.clone(), y_bool);
        let linfa_model = Svm::<_, bool>::params()
            .linear_kernel()
            .pos_neg_weights(1.0, 1.0);
        let linfa_fitted = linfa_model.fit(&dataset).unwrap();
        let linfa_pred_bool = linfa_fitted.predict(&x);
        let linfa_pred: Vec<f64> = linfa_pred_bool
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

        assert!(ferro_acc > 0.85, "FerroML SVC(linear) acc: {ferro_acc:.3}");
        assert!(linfa_acc > 0.85, "linfa SVC(linear) acc: {linfa_acc:.3}");
        assert!(
            (ferro_acc - linfa_acc).abs() < 0.10,
            "SVC(linear) gap: ferro={ferro_acc:.3}, linfa={linfa_acc:.3}"
        );
    }

    #[test]
    fn small() {
        compare(200, 5);
    }
    #[test]
    fn medium() {
        compare(500, 10);
    }
}

// ─── SVC RBF ────────────────────────────────────────────────────────

mod svc_rbf {
    use super::*;
    use linfa::prelude::*;
    use linfa_svm::Svm;

    fn compare(n: usize, p: usize) {
        let (x, y) = synthetic_classification(n, p, 42);
        let gamma = 1.0 / p as f64;

        // FerroML
        let mut ferro = ferroml_core::models::SVC::new()
            .with_kernel(ferroml_core::models::Kernel::Rbf { gamma })
            .with_c(1.0);
        ferro.fit(&x, &y).unwrap();
        let ferro_pred = ferro.predict(&x).unwrap();
        let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa SVM with RBF
        let y_bool: Array1<bool> = y.mapv(|v| v > 0.5);
        let dataset = linfa::Dataset::new(x.clone(), y_bool);
        let linfa_model = Svm::<_, bool>::params()
            .gaussian_kernel(1.0 / (2.0 * gamma)) // linfa uses sigma^2, not gamma
            .pos_neg_weights(1.0, 1.0);
        let linfa_fitted = linfa_model.fit(&dataset).unwrap();
        let linfa_pred_bool = linfa_fitted.predict(&x);
        let linfa_pred: Vec<f64> = linfa_pred_bool
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect();
        let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

        assert!(ferro_acc > 0.85, "FerroML SVC(rbf) acc: {ferro_acc:.3}");
        assert!(linfa_acc > 0.85, "linfa SVC(rbf) acc: {linfa_acc:.3}");
    }

    #[test]
    fn small() {
        compare(200, 5);
    }
    #[test]
    fn medium() {
        compare(500, 10);
    }
}

// ─── SVR ────────────────────────────────────────────────────────────

mod svr {
    use super::*;
    use linfa::prelude::*;
    use linfa_svm::Svm;

    fn compare(n: usize, p: usize) {
        let (x, y) = synthetic_regression(n, p, 42);

        // FerroML
        let mut ferro = ferroml_core::models::SVR::new()
            .with_kernel(ferroml_core::models::Kernel::Linear)
            .with_c(1.0)
            .with_epsilon(0.1);
        ferro.fit(&x, &y).unwrap();
        let ferro_pred = ferro.predict(&x).unwrap();
        let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa SVR — use c_svr(C, loss_epsilon) for regression
        let dataset = linfa::Dataset::new(x.clone(), y.clone());
        let linfa_model = Svm::<_, f64>::params()
            .linear_kernel()
            .c_svr(1.0, Some(0.1));
        let linfa_fitted = linfa_model.fit(&dataset).unwrap();
        let linfa_pred = linfa_fitted.predict(&x);
        let linfa_r2 = r2_score(y.as_slice().unwrap(), linfa_pred.as_slice().unwrap());

        assert!(ferro_r2 > 0.5, "FerroML SVR R²: {ferro_r2:.4}");
        assert!(linfa_r2 > 0.5, "linfa SVR R²: {linfa_r2:.4}");
    }

    #[test]
    fn small() {
        compare(200, 5);
    }
    #[test]
    fn medium() {
        compare(500, 10);
    }
}
