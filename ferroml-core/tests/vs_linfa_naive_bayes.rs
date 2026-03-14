//! Cross-library validation: FerroML vs linfa — Naive Bayes
//!
//! Compares GaussianNB, MultinomialNB, and BernoulliNB predictions.

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

fn positive_data(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Poisson};

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Class 0: high counts in first half of features, low in second half
    // Class 1: low counts in first half of features, high in second half
    let high = Poisson::new(8.0).unwrap();
    let low = Poisson::new(1.0).unwrap();
    let half = p / 2;

    let mut x = Array2::zeros((n, p));
    for i in 0..n {
        let is_class1 = i >= n / 2;
        for j in 0..p {
            let val: f64 = if j < half {
                if !is_class1 {
                    high.sample(&mut rng)
                } else {
                    low.sample(&mut rng)
                }
            } else if is_class1 {
                high.sample(&mut rng)
            } else {
                low.sample(&mut rng)
            };
            x[[i, j]] = val.max(0.0).round();
        }
    }

    let y: Array1<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
    (x, y)
}

fn binary_data(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let (x, y) = synthetic_classification(n, p, seed);
    let x = x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
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

fn agreement(a: &[f64], b: &[f64]) -> f64 {
    let matching = a
        .iter()
        .zip(b)
        .filter(|(&x, &y)| (x - y).abs() < 0.5)
        .count();
    matching as f64 / a.len() as f64
}

// ─── GaussianNB ─────────────────────────────────────────────────────

mod gaussian_nb {
    use super::*;
    use linfa::prelude::*;

    fn compare(n: usize, p: usize) {
        let (x, y) = synthetic_classification(n, p, 42);

        // FerroML
        let mut ferro_model = ferroml_core::models::GaussianNB::new();
        ferro_model.fit(&x, &y).unwrap();
        let ferro_pred = ferro_model.predict(&x).unwrap();
        let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa
        let y_usize: Array1<usize> = y.mapv(|v| v as usize);
        let dataset = linfa::Dataset::new(x.clone(), y_usize);
        let linfa_model = linfa_bayes::GaussianNb::params();
        let linfa_fitted = linfa_model.fit(&dataset).unwrap();
        let linfa_pred_usize = linfa_fitted.predict(&x);
        let linfa_pred: Vec<f64> = linfa_pred_usize.iter().map(|&v| v as f64).collect();
        let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

        let agree = agreement(ferro_pred.as_slice().unwrap(), &linfa_pred);

        assert!(
            ferro_acc > 0.85,
            "FerroML GaussianNB acc too low: {ferro_acc:.3}"
        );
        assert!(
            linfa_acc > 0.85,
            "linfa GaussianNB acc too low: {linfa_acc:.3}"
        );
        assert!(agree > 0.90, "GaussianNB agreement too low: {agree:.3}");
    }

    #[test]
    fn small() {
        compare(200, 5);
    }
    #[test]
    fn medium() {
        compare(1000, 10);
    }
    #[test]
    fn large() {
        compare(5000, 20);
    }
}

// ─── MultinomialNB ──────────────────────────────────────────────────

mod multinomial_nb {
    use super::*;
    use linfa::prelude::*;

    fn compare(n: usize, p: usize) {
        let (x, y) = positive_data(n, p, 42);

        // FerroML
        let mut ferro_model = ferroml_core::models::MultinomialNB::new();
        ferro_model.fit(&x, &y).unwrap();
        let ferro_pred = ferro_model.predict(&x).unwrap();
        let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa
        let y_usize: Array1<usize> = y.mapv(|v| v as usize);
        let dataset = linfa::Dataset::new(x.clone(), y_usize);
        let linfa_model = linfa_bayes::MultinomialNb::params();
        let linfa_fitted = linfa_model.fit(&dataset).unwrap();
        let linfa_pred_usize = linfa_fitted.predict(&x);
        let linfa_pred: Vec<f64> = linfa_pred_usize.iter().map(|&v| v as f64).collect();
        let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

        let agree = agreement(ferro_pred.as_slice().unwrap(), &linfa_pred);

        assert!(
            ferro_acc > 0.60,
            "FerroML MultinomialNB acc too low: {ferro_acc:.3}"
        );
        assert!(
            linfa_acc > 0.60,
            "linfa MultinomialNB acc too low: {linfa_acc:.3}"
        );
        assert!(agree > 0.70, "MultinomialNB agreement too low: {agree:.3}");
    }

    #[test]
    fn small() {
        compare(200, 5);
    }
    #[test]
    fn medium() {
        compare(1000, 10);
    }
}

// ─── BernoulliNB ────────────────────────────────────────────────────

mod bernoulli_nb {
    use super::*;
    use linfa::prelude::*;

    fn compare(n: usize, p: usize) {
        let (x, y) = binary_data(n, p, 42);

        // FerroML
        let mut ferro_model = ferroml_core::models::BernoulliNB::new();
        ferro_model.fit(&x, &y).unwrap();
        let ferro_pred = ferro_model.predict(&x).unwrap();
        let ferro_acc = accuracy(y.as_slice().unwrap(), ferro_pred.as_slice().unwrap());

        // linfa
        let y_usize: Array1<usize> = y.mapv(|v| v as usize);
        let dataset = linfa::Dataset::new(x.clone(), y_usize);
        let linfa_model = linfa_bayes::BernoulliNb::params();
        let linfa_fitted = linfa_model.fit(&dataset).unwrap();
        let linfa_pred_usize = linfa_fitted.predict(&x);
        let linfa_pred: Vec<f64> = linfa_pred_usize.iter().map(|&v| v as f64).collect();
        let linfa_acc = accuracy(y.as_slice().unwrap(), &linfa_pred);

        let agree = agreement(ferro_pred.as_slice().unwrap(), &linfa_pred);

        assert!(
            ferro_acc > 0.60,
            "FerroML BernoulliNB acc too low: {ferro_acc:.3}"
        );
        assert!(
            linfa_acc > 0.60,
            "linfa BernoulliNB acc too low: {linfa_acc:.3}"
        );
        assert!(agree > 0.80, "BernoulliNB agreement too low: {agree:.3}");
    }

    #[test]
    fn small() {
        compare(200, 10);
    }
    #[test]
    fn medium() {
        compare(1000, 20);
    }
}
