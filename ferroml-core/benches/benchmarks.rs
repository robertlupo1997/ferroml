//! FerroML Performance Benchmark Suite
//!
//! This module benchmarks FerroML's performance characteristics including:
//! - Training time across different dataset sizes
//! - Prediction latency
//! - Scaling behavior
//!
//! Run with: `cargo bench`
//!
//! ## Comparison with sklearn
//!
//! These benchmarks provide timing data that can be compared against sklearn.
//! To generate sklearn reference timings, run the Python script in
//! `benchmarks/sklearn_timing.py` (TODO: create this script).
//!
//! ## Comparison with XGBoost/LightGBM
//!
//! The gradient boosting benchmarks can be compared against XGBoost and LightGBM.
//! Run `benchmarks/xgboost_lightgbm_timing.py` to generate reference timings.
//!
//! Key differences to note:
//! - **XGBoost**: Highly optimized C++, histogram-based by default since 1.0
//! - **LightGBM**: Leaf-wise growth, histogram-based, fastest for large datasets
//! - **FerroML GradientBoosting**: Standard CART-based gradient boosting
//! - **FerroML HistGradientBoosting**: Histogram-based like LightGBM, with
//!   monotonic constraints and feature interaction constraints
//!
//! Expected relative performance:
//! - Small datasets (<10K): FerroML competitive, XGBoost/LightGBM 2-5x faster
//! - Medium datasets (10K-100K): XGBoost/LightGBM 5-10x faster
//! - Large datasets (>100K): XGBoost/LightGBM 10-50x faster (due to SIMD, cache optimization)
//!
//! FerroML's advantages:
//! - Pure Rust, no external dependencies
//! - Statistical rigor (feature importance with CIs)
//! - Native monotonic constraints
//! - Feature interaction constraints
//! - Seamless integration with FerroML pipeline
//!
//! ## Benchmark Categories
//!
//! 1. **Linear Models**: OLS, Ridge, Lasso - O(n·p²) or O(n·p) complexity
//! 2. **Tree Models**: Decision trees, Random forests - O(n·p·log(n)) complexity
//! 3. **Gradient Boosting**: Standard and histogram-based - O(n·p·d·trees) complexity
//! 4. **Preprocessing**: Scalers - O(n·p) complexity
//! 5. **Prediction**: All models - typically O(n·p) or O(n·trees·depth)
//!
//! ## Dataset Sizes
//!
//! - Small: 100 samples, 10 features
//! - Medium: 1,000 samples, 50 features
//! - Large: 10,000 samples, 100 features

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ferroml_core::datasets::{make_classification, make_regression};
use ferroml_core::models::boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
use ferroml_core::models::forest::{RandomForestClassifier, RandomForestRegressor};
use ferroml_core::models::hist_boosting::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
};
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::regularized::{LassoRegression, RidgeRegression};
use ferroml_core::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use ferroml_core::models::Model;
use ferroml_core::preprocessing::scalers::{
    MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler,
};
use ferroml_core::preprocessing::Transformer;
use ndarray::{Array1, Array2};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate synthetic regression data for benchmarking
fn generate_regression_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let n_informative = n_features / 2;
    let (dataset, _) = make_regression(n_samples, n_features, n_informative, 0.1, Some(42));
    dataset.into_arrays()
}

/// Generate synthetic classification data for benchmarking
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<f64>) {
    let n_informative = n_features / 2;
    let (dataset, _) =
        make_classification(n_samples, n_features, n_informative, n_classes, Some(42));
    dataset.into_arrays()
}

// =============================================================================
// LINEAR MODEL BENCHMARKS
// =============================================================================

/// Benchmark LinearRegression training time
fn bench_linear_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearRegression/fit");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (10000, 100)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = LinearRegression::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark LinearRegression prediction time
fn bench_linear_regression_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearRegression/predict");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (10000, 100)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        // Train model once
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| model.predict(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark RidgeRegression training time
fn bench_ridge_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("RidgeRegression/fit");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (10000, 100)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = RidgeRegression::new(1.0);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark LassoRegression training time
fn bench_lasso_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("LassoRegression/fit");

    // Lasso uses iterative coordinate descent, so we use smaller sizes
    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = LassoRegression::new(0.1);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// TREE MODEL BENCHMARKS
// =============================================================================

/// Benchmark DecisionTreeClassifier training time
fn bench_decision_tree_classifier_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("DecisionTreeClassifier/fit");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = DecisionTreeClassifier::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark DecisionTreeClassifier prediction time
fn bench_decision_tree_classifier_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("DecisionTreeClassifier/predict");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (10000, 100)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        // Train model once
        let mut model = DecisionTreeClassifier::new();
        model.fit(&x, &y).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| model.predict(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark DecisionTreeRegressor training time
fn bench_decision_tree_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("DecisionTreeRegressor/fit");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = DecisionTreeRegressor::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark RandomForestClassifier training time
fn bench_random_forest_classifier_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestClassifier/fit");
    group.sample_size(10); // Random forests are slow, reduce sample size

    for (n_samples, n_features) in [(100, 10), (500, 20), (1000, 20)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    // Use smaller forest for benchmarking
                    let mut model = RandomForestClassifier::new()
                        .with_n_estimators(10)
                        .with_max_depth(Some(10))
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark RandomForestClassifier prediction time
fn bench_random_forest_classifier_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestClassifier/predict");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        // Train model once with small forest
        let mut model = RandomForestClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(10))
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| model.predict(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark RandomForestRegressor training time
fn bench_random_forest_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestRegressor/fit");
    group.sample_size(10); // Random forests are slow, reduce sample size

    for (n_samples, n_features) in [(100, 10), (500, 20), (1000, 20)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    // Use smaller forest for benchmarking
                    let mut model = RandomForestRegressor::new()
                        .with_n_estimators(10)
                        .with_max_depth(Some(10))
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// GRADIENT BOOSTING BENCHMARKS
// =============================================================================

/// Benchmark GradientBoostingClassifier training time
///
/// Compares against XGBoost/LightGBM. Expected to be slower due to:
/// - No histogram-based split finding
/// - Pure Rust without SIMD optimizations
fn bench_gradient_boosting_classifier_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("GradientBoostingClassifier/fit");
    group.sample_size(10); // Gradient boosting is slow, reduce sample size

    for (n_samples, n_features) in [(100, 10), (500, 20), (1000, 20)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    // Use small model for benchmarking (10 trees, max depth 3)
                    let mut model = GradientBoostingClassifier::new()
                        .with_n_estimators(10)
                        .with_max_depth(Some(3))
                        .with_learning_rate(0.1)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark GradientBoostingClassifier prediction time
fn bench_gradient_boosting_classifier_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("GradientBoostingClassifier/predict");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        // Train model once
        let mut model = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(Some(3))
            .with_learning_rate(0.1)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| model.predict(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark GradientBoostingRegressor training time
fn bench_gradient_boosting_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("GradientBoostingRegressor/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(100, 10), (500, 20), (1000, 20)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = GradientBoostingRegressor::new()
                        .with_n_estimators(10)
                        .with_max_depth(Some(3))
                        .with_learning_rate(0.1)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark GradientBoostingRegressor prediction time
fn bench_gradient_boosting_regressor_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("GradientBoostingRegressor/predict");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        let mut model = GradientBoostingRegressor::new()
            .with_n_estimators(10)
            .with_max_depth(Some(3))
            .with_learning_rate(0.1)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| model.predict(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

// =============================================================================
// HISTOGRAM GRADIENT BOOSTING BENCHMARKS (LightGBM-style)
// =============================================================================

/// Benchmark HistGradientBoostingClassifier training time
///
/// This is FerroML's histogram-based gradient boosting, similar to LightGBM.
/// Expected to be faster than standard GradientBoosting but slower than
/// LightGBM due to pure Rust implementation without SIMD.
fn bench_hist_gradient_boosting_classifier_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("HistGradientBoostingClassifier/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(100, 10), (500, 20), (1000, 20), (2000, 30)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = HistGradientBoostingClassifier::new()
                        .with_max_iter(10)
                        .with_max_depth(Some(3))
                        .with_learning_rate(0.1)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark HistGradientBoostingClassifier prediction time
fn bench_hist_gradient_boosting_classifier_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("HistGradientBoostingClassifier/predict");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        let mut model = HistGradientBoostingClassifier::new()
            .with_max_iter(10)
            .with_max_depth(Some(3))
            .with_learning_rate(0.1)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| model.predict(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark HistGradientBoostingRegressor training time
fn bench_hist_gradient_boosting_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("HistGradientBoostingRegressor/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(100, 10), (500, 20), (1000, 20), (2000, 30)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = HistGradientBoostingRegressor::new()
                        .with_max_iter(10)
                        .with_max_depth(Some(3))
                        .with_learning_rate(0.1)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark HistGradientBoostingRegressor prediction time
fn bench_hist_gradient_boosting_regressor_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("HistGradientBoostingRegressor/predict");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        let mut model = HistGradientBoostingRegressor::new()
            .with_max_iter(10)
            .with_max_depth(Some(3))
            .with_learning_rate(0.1)
            .with_random_state(42);
        model.fit(&x, &y).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| model.predict(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

/// Benchmark gradient boosting scaling with number of trees
///
/// This shows how training time scales as we add more trees.
/// Important for understanding the trade-off between model complexity and training time.
fn bench_gradient_boosting_tree_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling/GradientBoosting/Trees");
    group.sample_size(10);

    let n_samples = 500;
    let n_features = 20;
    let (x, y) = generate_classification_data(n_samples, n_features, 2);

    for n_trees in [5, 10, 20, 50] {
        group.throughput(Throughput::Elements(n_trees as u64));
        group.bench_with_input(
            BenchmarkId::new("n_estimators", n_trees),
            &n_trees,
            |b, &n_trees| {
                b.iter(|| {
                    let mut model = GradientBoostingClassifier::new()
                        .with_n_estimators(n_trees)
                        .with_max_depth(Some(3))
                        .with_learning_rate(0.1)
                        .with_random_state(42);
                    model.fit(black_box(&x), black_box(&y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark HistGradientBoosting scaling with number of trees
fn bench_hist_gradient_boosting_tree_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling/HistGradientBoosting/Trees");
    group.sample_size(10);

    let n_samples = 500;
    let n_features = 20;
    let (x, y) = generate_classification_data(n_samples, n_features, 2);

    for n_trees in [5, 10, 20, 50, 100] {
        group.throughput(Throughput::Elements(n_trees as u64));
        group.bench_with_input(
            BenchmarkId::new("max_iter", n_trees),
            &n_trees,
            |b, &n_trees| {
                b.iter(|| {
                    let mut model = HistGradientBoostingClassifier::new()
                        .with_max_iter(n_trees)
                        .with_max_depth(Some(3))
                        .with_learning_rate(0.1)
                        .with_random_state(42);
                    model.fit(black_box(&x), black_box(&y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark gradient boosting scaling with dataset size
///
/// This shows O(n) scaling for histogram-based vs O(n log n) for standard.
fn bench_gradient_boosting_sample_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling/GradientBoosting/Samples");
    group.sample_size(10);

    let n_features = 20;
    for n_samples in [100, 250, 500, 1000] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("standard", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = GradientBoostingClassifier::new()
                        .with_n_estimators(10)
                        .with_max_depth(Some(3))
                        .with_learning_rate(0.1)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("histogram", n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = HistGradientBoostingClassifier::new()
                        .with_max_iter(10)
                        .with_max_depth(Some(3))
                        .with_learning_rate(0.1)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark gradient boosting prediction time comparison
///
/// Prediction time for gradient boosting is O(n_samples * n_trees * depth).
/// This benchmark compares standard vs histogram-based prediction.
fn bench_gradient_boosting_predict_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison/GradientBoosting/Predict");

    let n_train = 500;
    let n_features = 20;
    let (x_train, y_train) = generate_classification_data(n_train, n_features, 2);

    // Train models
    let mut standard = GradientBoostingClassifier::new()
        .with_n_estimators(50)
        .with_max_depth(Some(5))
        .with_learning_rate(0.1)
        .with_random_state(42);
    standard.fit(&x_train, &y_train).unwrap();

    let mut hist = HistGradientBoostingClassifier::new()
        .with_max_iter(50)
        .with_max_depth(Some(5))
        .with_learning_rate(0.1)
        .with_random_state(42);
    hist.fit(&x_train, &y_train).unwrap();

    for n_samples in [100, 1000, 5000, 10000] {
        let (x_test, _) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(BenchmarkId::new("standard", n_samples), &x_test, |b, x| {
            b.iter(|| standard.predict(black_box(x)).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("histogram", n_samples), &x_test, |b, x| {
            b.iter(|| hist.predict(black_box(x)).unwrap())
        });
    }

    group.finish();
}

// =============================================================================
// PREPROCESSING BENCHMARKS
// =============================================================================

/// Benchmark StandardScaler fit_transform time
fn bench_standard_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("StandardScaler/fit_transform");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (10000, 100), (100000, 100)] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut scaler = StandardScaler::new();
                    scaler.fit_transform(black_box(x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark MinMaxScaler fit_transform time
fn bench_minmax_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("MinMaxScaler/fit_transform");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (10000, 100), (100000, 100)] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut scaler = MinMaxScaler::new();
                    scaler.fit_transform(black_box(x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark RobustScaler fit_transform time
fn bench_robust_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("RobustScaler/fit_transform");

    // RobustScaler uses median/IQR which requires sorting
    for (n_samples, n_features) in [(100, 10), (1000, 50), (10000, 100)] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut scaler = RobustScaler::new();
                    scaler.fit_transform(black_box(x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark MaxAbsScaler fit_transform time
fn bench_maxabs_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("MaxAbsScaler/fit_transform");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (10000, 100), (100000, 100)] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut scaler = MaxAbsScaler::new();
                    scaler.fit_transform(black_box(x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark scaler transform time (pre-fitted)
fn bench_scaler_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaler/transform");

    let (x, _) = generate_regression_data(10000, 100);

    // Pre-fit scalers
    let mut standard = StandardScaler::new();
    standard.fit(&x).unwrap();

    let mut minmax = MinMaxScaler::new();
    minmax.fit(&x).unwrap();

    group.throughput(Throughput::Elements(10000));

    group.bench_function("StandardScaler", |b| {
        b.iter(|| standard.transform(black_box(&x)).unwrap())
    });

    group.bench_function("MinMaxScaler", |b| {
        b.iter(|| minmax.transform(black_box(&x)).unwrap())
    });

    group.finish();
}

// =============================================================================
// SCALING BENCHMARKS
// =============================================================================

/// Benchmark how LinearRegression scales with sample size
fn bench_linear_regression_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling/LinearRegression");

    let n_features = 50;
    for n_samples in [100, 500, 1000, 5000, 10000] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = LinearRegression::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark how DecisionTree scales with sample size
fn bench_decision_tree_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling/DecisionTree");

    let n_features = 20;
    for n_samples in [100, 500, 1000, 2000, 5000] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = DecisionTreeClassifier::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark how prediction time scales with sample size
fn bench_prediction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling/Predict");

    let n_features = 50;

    // Train models on medium dataset
    let (x_train, y_train) = generate_regression_data(1000, n_features);

    let mut linear = LinearRegression::new();
    linear.fit(&x_train, &y_train).unwrap();

    let mut ridge = RidgeRegression::new(1.0);
    ridge.fit(&x_train, &y_train).unwrap();

    let mut tree = DecisionTreeRegressor::new();
    tree.fit(&x_train, &y_train).unwrap();

    for n_samples in [100, 1000, 10000, 50000] {
        let (x_test, _) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("LinearRegression", n_samples),
            &x_test,
            |b, x| b.iter(|| linear.predict(black_box(x)).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("RidgeRegression", n_samples),
            &x_test,
            |b, x| b.iter(|| ridge.predict(black_box(x)).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("DecisionTree", n_samples),
            &x_test,
            |b, x| b.iter(|| tree.predict(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    linear_models,
    bench_linear_regression_fit,
    bench_linear_regression_predict,
    bench_ridge_regression_fit,
    bench_lasso_regression_fit,
);

criterion_group!(
    tree_models,
    bench_decision_tree_classifier_fit,
    bench_decision_tree_classifier_predict,
    bench_decision_tree_regressor_fit,
    bench_random_forest_classifier_fit,
    bench_random_forest_classifier_predict,
    bench_random_forest_regressor_fit,
);

criterion_group!(
    gradient_boosting,
    bench_gradient_boosting_classifier_fit,
    bench_gradient_boosting_classifier_predict,
    bench_gradient_boosting_regressor_fit,
    bench_gradient_boosting_regressor_predict,
);

criterion_group!(
    hist_gradient_boosting,
    bench_hist_gradient_boosting_classifier_fit,
    bench_hist_gradient_boosting_classifier_predict,
    bench_hist_gradient_boosting_regressor_fit,
    bench_hist_gradient_boosting_regressor_predict,
);

criterion_group!(
    preprocessing,
    bench_standard_scaler,
    bench_minmax_scaler,
    bench_robust_scaler,
    bench_maxabs_scaler,
    bench_scaler_transform,
);

criterion_group!(
    scaling,
    bench_linear_regression_scaling,
    bench_decision_tree_scaling,
    bench_prediction_scaling,
    bench_gradient_boosting_tree_scaling,
    bench_hist_gradient_boosting_tree_scaling,
    bench_gradient_boosting_sample_scaling,
    bench_gradient_boosting_predict_comparison,
);

criterion_main!(
    linear_models,
    tree_models,
    gradient_boosting,
    hist_gradient_boosting,
    preprocessing,
    scaling
);
