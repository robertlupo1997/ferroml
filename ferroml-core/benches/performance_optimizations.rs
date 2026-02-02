//! Benchmarks for P2 Performance Optimizations
//!
//! This benchmark suite measures the impact of performance optimizations:
//! - SIMD histogram subtraction in HistGradientBoosting
//! - Parallel histogram building in HistGradientBoosting
//! - Parallel tree prediction in RandomForest
//!
//! Run with: `cargo bench -p ferroml-core --bench performance_optimizations`
//!
//! Compare with/without parallel:
//! - `cargo bench -p ferroml-core --bench performance_optimizations`
//! - `cargo bench -p ferroml-core --bench performance_optimizations --no-default-features`
//!
//! Compare with SIMD:
//! - `cargo bench -p ferroml-core --bench performance_optimizations --features simd`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferroml_core::datasets::{make_classification, make_regression};
use ferroml_core::models::forest::{RandomForestClassifier, RandomForestRegressor};
use ferroml_core::models::hist_boosting::HistGradientBoostingRegressor;
use ferroml_core::models::tree::DecisionTreeRegressor;
use ferroml_core::models::Model;
use ndarray::{Array1, Array2};

/// Generate synthetic regression data for benchmarking
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let n_informative = n_features / 2;
    let (dataset, _) = make_regression(n_samples, n_features, n_informative, 0.1, Some(42));
    dataset.into_arrays()
}

/// Generate synthetic classification data for benchmarking
fn generate_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let n_informative = n_features / 2;
    let (dataset, _) = make_classification(n_samples, n_features, n_informative, 2, Some(42));
    dataset.into_arrays()
}

// =============================================================================
// HistGradientBoosting Training Benchmarks
// =============================================================================

fn bench_hist_gradient_boosting_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("HistGradientBoosting_Fit");
    group.sample_size(10); // Reduce sample size for longer benchmarks

    for n_samples in [1000, 5000, 10000] {
        let (x, y) = generate_regression_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("regressor", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = HistGradientBoostingRegressor::new()
                        .with_max_iter(10)
                        .with_max_depth(Some(4))
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_hist_gradient_boosting_by_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("HistGradientBoosting_Features");
    group.sample_size(10);

    // Test scaling with number of features (parallel histogram building)
    for n_features in [10, 50, 100] {
        let (x, y) = generate_regression_data(5000, n_features);

        group.bench_with_input(
            BenchmarkId::new("regressor", n_features),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = HistGradientBoostingRegressor::new()
                        .with_max_iter(10)
                        .with_max_depth(Some(4))
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// RandomForest Prediction Benchmarks
// =============================================================================

fn bench_random_forest_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForest_Prediction");

    let (x_train, y_train) = generate_regression_data(1000, 10);
    let (x_test, _) = generate_regression_data(1000, 10);

    // Test scaling with number of trees (parallel prediction)
    for n_estimators in [10, 50, 100] {
        let mut model = RandomForestRegressor::new()
            .with_n_estimators(n_estimators)
            .with_max_depth(Some(5))
            .with_random_state(42);
        model.fit(&x_train, &y_train).unwrap();

        group.bench_with_input(
            BenchmarkId::new("regressor", n_estimators),
            &(&model, &x_test),
            |b, (model, x)| {
                b.iter(|| model.predict(black_box(*x)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_random_forest_classifier_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestClassifier_Prediction");

    let (x_train, y_train) = generate_classification_data(1000, 10);
    let (x_test, _) = generate_classification_data(1000, 10);

    for n_estimators in [10, 50, 100] {
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(n_estimators)
            .with_max_depth(Some(5))
            .with_random_state(42);
        model.fit(&x_train, &y_train).unwrap();

        group.bench_with_input(
            BenchmarkId::new("classifier", n_estimators),
            &(&model, &x_test),
            |b, (model, x)| {
                b.iter(|| model.predict(black_box(*x)).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_random_forest_prediction_batch_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForest_BatchSize");

    let (x_train, y_train) = generate_regression_data(2000, 10);

    // Train a fixed model
    let mut model = RandomForestRegressor::new()
        .with_n_estimators(50)
        .with_max_depth(Some(5))
        .with_random_state(42);
    model.fit(&x_train, &y_train).unwrap();

    // Test scaling with batch size
    for n_samples in [100, 500, 1000, 5000] {
        let (x_test, _) = generate_regression_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("samples", n_samples),
            &(&model, &x_test),
            |b, (model, x)| {
                b.iter(|| model.predict(black_box(*x)).unwrap());
            },
        );
    }

    group.finish();
}

// =============================================================================
// DecisionTree Prediction Benchmarks
// =============================================================================

fn bench_decision_tree_prediction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("DecisionTree_Prediction");

    let (x_train, y_train) = generate_regression_data(5000, 10);

    // Train a fixed model
    let mut model = DecisionTreeRegressor::new()
        .with_max_depth(Some(10))
        .with_random_state(42);
    model.fit(&x_train, &y_train).unwrap();

    // Test scaling with batch size
    for n_samples in [100, 1000, 10000] {
        let (x_test, _) = generate_regression_data(n_samples, 10);

        group.bench_with_input(
            BenchmarkId::new("samples", n_samples),
            &(&model, &x_test),
            |b, (model, x)| {
                b.iter(|| model.predict(black_box(*x)).unwrap());
            },
        );
    }

    group.finish();
}

// =============================================================================
// SIMD Operations Benchmarks (only when simd feature enabled)
// =============================================================================

#[cfg(feature = "simd")]
fn bench_simd_vector_operations(c: &mut Criterion) {
    use ferroml_core::simd::{vector_sub, vector_sub_into};

    let mut group = c.benchmark_group("SIMD_VectorOps");

    for size in [64, 256, 1024] {
        let a: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..size).map(|i| (size - i) as f64).collect();

        group.bench_with_input(BenchmarkId::new("vector_sub", size), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| vector_sub(black_box(a), black_box(b)));
        });

        group.bench_with_input(
            BenchmarkId::new("vector_sub_into", size),
            &(&a, &b),
            |bench, (a, b)| {
                // Create a mutable destination for each iteration
                let mut local_dst = vec![0.0; a.len()];
                bench.iter(|| vector_sub_into(black_box(a), black_box(b), black_box(&mut local_dst)));
            },
        );
    }

    group.finish();
}

#[cfg(feature = "simd")]
criterion_group!(
    benches,
    bench_hist_gradient_boosting_fit,
    bench_hist_gradient_boosting_by_features,
    bench_random_forest_prediction,
    bench_random_forest_classifier_prediction,
    bench_random_forest_prediction_batch_size,
    bench_decision_tree_prediction_scaling,
    bench_simd_vector_operations,
);

#[cfg(not(feature = "simd"))]
criterion_group!(
    benches,
    bench_hist_gradient_boosting_fit,
    bench_hist_gradient_boosting_by_features,
    bench_random_forest_prediction,
    bench_random_forest_classifier_prediction,
    bench_random_forest_prediction_batch_size,
    bench_decision_tree_prediction_scaling,
);

criterion_main!(benches);
