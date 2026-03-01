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
use ferroml_core::clustering::{ClusteringModel, KMeans, DBSCAN};
use ferroml_core::datasets::{make_classification, make_regression};
use ferroml_core::models::forest::{RandomForestClassifier, RandomForestRegressor};
use ferroml_core::models::hist_boosting::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
};
use ferroml_core::models::knn::KNeighborsClassifier;
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

        group.bench_with_input(
            BenchmarkId::new("vector_sub", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| vector_sub(black_box(a), black_box(b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vector_sub_into", size),
            &(&a, &b),
            |bench, (a, b)| {
                // Create a mutable destination for each iteration
                let mut local_dst = vec![0.0; a.len()];
                bench.iter(|| {
                    vector_sub_into(black_box(a), black_box(b), black_box(&mut local_dst))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Parallel Speedup Verification: RandomForest Training
// =============================================================================

/// Benchmark RandomForest training at different scales to verify parallel speedup
///
/// This benchmark measures how RandomForest training time scales with dataset size.
/// With parallel histogram building and parallel tree construction, we expect
/// near-linear scaling on multi-core machines.
fn bench_random_forest_parallel_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel_RF_Scaling");
    group.sample_size(10);

    let n_features = 20;
    for n_samples in [500, 1000, 2000, 5000] {
        let (x, y) = generate_classification_data(n_samples, n_features);

        group.bench_with_input(
            BenchmarkId::new("fit", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = RandomForestClassifier::new()
                        .with_n_estimators(50)
                        .with_max_depth(Some(8))
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Scaling Benchmarks: Verify computational complexity
// =============================================================================

/// Benchmark KMeans clustering scaling with sample count
///
/// KMeans has O(n * k * p * iter) complexity, so timing should scale linearly with n.
fn bench_kmeans_sample_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling_KMeans_Samples");
    group.sample_size(10);

    let n_features = 20;
    for n_samples in [500, 1000, 2000, 5000, 10000] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        group.bench_with_input(BenchmarkId::new("fit", n_samples), &x, |b, x| {
            b.iter(|| {
                let mut model = KMeans::new(5).random_state(42);
                model.fit(black_box(x)).unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark DBSCAN clustering scaling with sample count
///
/// DBSCAN has O(n^2 * p) complexity in the worst case (brute-force range queries).
/// Timing should scale quadratically with n.
fn bench_dbscan_sample_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling_DBSCAN_Samples");
    group.sample_size(10);

    let n_features = 20;
    for n_samples in [200, 500, 1000, 2000] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        group.bench_with_input(BenchmarkId::new("fit", n_samples), &x, |b, x| {
            b.iter(|| {
                let mut model = DBSCAN::new(0.5, 5);
                model.fit(black_box(x)).unwrap()
            })
        });
    }

    group.finish();
}

/// Benchmark KNN classification scaling with sample count
///
/// KNN fit+predict has O(n^2 * p) complexity (brute-force).
fn bench_knn_sample_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling_KNN_Samples");
    group.sample_size(10);

    let n_features = 20;
    for n_samples in [200, 500, 1000, 2000, 5000] {
        let (x, y) = generate_classification_data(n_samples, n_features);

        group.bench_with_input(
            BenchmarkId::new("fit_predict", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = KNeighborsClassifier::new(5);
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark RandomForest fit scaling with sample count
///
/// RandomForest has O(trees * n * p * log(n)) complexity.
fn bench_random_forest_fit_sample_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling_RF_Fit_Samples");
    group.sample_size(10);

    let n_features = 20;
    for n_samples in [500, 1000, 2000, 5000] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.bench_with_input(
            BenchmarkId::new("fit", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = RandomForestRegressor::new()
                        .with_n_estimators(20)
                        .with_max_depth(Some(8))
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark HistGradientBoosting predict scaling with sample count
///
/// Prediction is O(n * trees * depth), should scale linearly with n.
fn bench_hist_gb_predict_sample_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling_HistGB_Predict_Samples");
    group.sample_size(10);

    let n_features = 20;
    let (x_train, y_train) = generate_classification_data(2000, n_features);

    let mut model = HistGradientBoostingClassifier::new()
        .with_max_iter(50)
        .with_max_depth(Some(5))
        .with_random_state(42);
    model.fit(&x_train, &y_train).unwrap();

    for n_samples in [1000, 5000, 10000, 50000] {
        let (x_test, _) = generate_classification_data(n_samples, n_features);

        group.bench_with_input(BenchmarkId::new("predict", n_samples), &x_test, |b, x| {
            b.iter(|| model.predict(black_box(x)).unwrap())
        });
    }

    group.finish();
}

/// Benchmark HistGradientBoosting at large scale to verify parallel histogram speedup
fn bench_hist_gb_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel_HistGB_LargeScale");
    group.sample_size(10);

    for n_samples in [5000, 10000, 20000] {
        let (x, y) = generate_regression_data(n_samples, 20);

        group.bench_with_input(
            BenchmarkId::new("fit", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = HistGradientBoostingRegressor::new()
                        .with_max_iter(20)
                        .with_max_depth(Some(5))
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
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
    bench_random_forest_parallel_scaling,
    bench_hist_gb_large_scale,
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
    bench_random_forest_parallel_scaling,
    bench_hist_gb_large_scale,
);

criterion_group!(
    scaling_benches,
    bench_kmeans_sample_scaling,
    bench_dbscan_sample_scaling,
    bench_knn_sample_scaling,
    bench_random_forest_fit_sample_scaling,
    bench_hist_gb_predict_sample_scaling,
);

criterion_main!(benches, scaling_benches);
