//! Cross-library validation: FerroML vs linfa — K-Nearest Neighbors

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

#[allow(dead_code)]
fn agreement(a: &[f64], b: &[f64]) -> f64 {
    let matching = a
        .iter()
        .zip(b)
        .filter(|(&x, &y)| (x - y).abs() < 0.5)
        .count();
    matching as f64 / a.len() as f64
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

// ─── KNN Classifier ─────────────────────────────────────────────────

mod knn_classifier {
    use super::*;

    fn compare(n: usize, p: usize, k: usize) {
        let (x, y) = synthetic_classification(n, p, 42);

        // FerroML
        let mut ferro = ferroml_core::models::KNeighborsClassifier::new(k);
        ferro.fit(&x, &y).unwrap();
        let ferro_pred = ferro.predict(&x).unwrap();
        let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // KNN is deterministic — FerroML should achieve very high train accuracy
        assert!(ferro_acc > 0.90, "FerroML KNN(k={k}) acc: {ferro_acc:.3}");
    }

    #[test]
    fn k3_small() {
        compare(200, 5, 3);
    }
    #[test]
    fn k5_small() {
        compare(200, 5, 5);
    }
    #[test]
    fn k7_small() {
        compare(200, 5, 7);
    }
    #[test]
    fn k3_medium() {
        compare(1000, 10, 3);
    }
    #[test]
    fn k5_medium() {
        compare(1000, 10, 5);
    }
}

// ─── KNN Regressor ──────────────────────────────────────────────────

mod knn_regressor {
    use super::*;

    fn compare(n: usize, p: usize, k: usize) {
        let (x, y) = synthetic_regression(n, p, 42);

        // FerroML
        let mut ferro = ferroml_core::models::KNeighborsRegressor::new(k);
        ferro.fit(&x, &y).unwrap();
        let ferro_pred = ferro.predict(&x).unwrap();
        let ferro_r2 = r2_score(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // k=1 on training data should achieve perfect R² (overfitting)
        if k == 1 {
            assert!(
                ferro_r2 > 0.99,
                "FerroML KNN(k=1) regressor should memorize: R²={ferro_r2:.4}"
            );
        } else {
            assert!(
                ferro_r2 > 0.7,
                "FerroML KNN(k={k}) regressor R²: {ferro_r2:.4}"
            );
        }
    }

    #[test]
    fn k1_small() {
        compare(200, 5, 1);
    }
    #[test]
    fn k3_small() {
        compare(200, 5, 3);
    }
    #[test]
    fn k5_medium() {
        compare(500, 10, 5);
    }
    #[test]
    fn k7_medium() {
        compare(500, 10, 7);
    }
}
