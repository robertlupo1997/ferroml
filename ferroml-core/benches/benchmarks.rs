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
use ferroml_core::clustering::agglomerative::AgglomerativeClustering;
use ferroml_core::clustering::DBSCAN;
use ferroml_core::clustering::{ClusteringModel, KMeans};
use ferroml_core::datasets::{make_classification, make_regression};
use ferroml_core::decomposition::{FactorAnalysis, LDA};
use ferroml_core::decomposition::{TruncatedSVD, PCA};
use ferroml_core::ensemble::{BaggingClassifier, StackingClassifier, VotingClassifier};
use ferroml_core::models::adaboost::{AdaBoostClassifier, AdaBoostRegressor};
use ferroml_core::models::boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
use ferroml_core::models::calibration::{CalibrableClassifier, CalibratedClassifierCV};
use ferroml_core::models::extra_trees::{ExtraTreesClassifier, ExtraTreesRegressor};
use ferroml_core::models::forest::{RandomForestClassifier, RandomForestRegressor};
use ferroml_core::models::hist_boosting::{
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
};
use ferroml_core::models::knn::NearestCentroid;
use ferroml_core::models::knn::{KNeighborsClassifier, KNeighborsRegressor};
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::logistic::LogisticRegression;
use ferroml_core::models::naive_bayes::GaussianNB;
use ferroml_core::models::naive_bayes::{BernoulliNB, MultinomialNB};
use ferroml_core::models::quantile::QuantileRegression;
use ferroml_core::models::regularized::{
    ElasticNet, ElasticNetCV, LassoCV, LassoRegression, RidgeCV, RidgeClassifier, RidgeRegression,
};
use ferroml_core::models::robust::RobustRegression;
use ferroml_core::models::sgd::{PassiveAggressiveClassifier, SGDClassifier, SGDRegressor};
use ferroml_core::models::svm::{LinearSVC, LinearSVR, SVC, SVR};
use ferroml_core::models::tree::{DecisionTreeClassifier, DecisionTreeRegressor};
use ferroml_core::models::Model;
use ferroml_core::neural::{Activation, MLPClassifier, MLPRegressor, NeuralModel, Solver};
use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
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
// SVM BENCHMARKS
// =============================================================================

fn bench_svc_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVC/fit_predict");
    group.sample_size(10);
    for (n_samples, n_features) in [(100, 10), (500, 20)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = SVC::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_linear_svc_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearSVC/fit_predict");
    group.sample_size(10);
    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = LinearSVC::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_svr_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVR/fit_predict");
    group.sample_size(10);
    for (n_samples, n_features) in [(100, 10), (500, 20)] {
        let (x, y) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = SVR::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// KNN BENCHMARKS
// =============================================================================

fn bench_knn_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNeighborsClassifier/fit_predict");
    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
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

fn bench_knn_regressor_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("KNeighborsRegressor/fit_predict");
    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = KNeighborsRegressor::new(5);
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// LOGISTIC REGRESSION BENCHMARKS
// =============================================================================

fn bench_logistic_regression_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("LogisticRegression/fit_predict");
    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = LogisticRegression::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// SGD BENCHMARKS
// =============================================================================

fn bench_sgd_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("SGDClassifier/fit_predict");
    for (n_samples, n_features) in [(1000, 50), (5000, 100)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = SGDClassifier::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_sgd_regressor_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("SGDRegressor/fit_predict");
    for (n_samples, n_features) in [(1000, 50), (5000, 100)] {
        let (x, y) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = SGDRegressor::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// ADABOOST BENCHMARKS
// =============================================================================

fn bench_adaboost_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("AdaBoostClassifier/fit_predict");
    group.sample_size(10);
    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = AdaBoostClassifier::new(10)
                        .with_max_depth(3)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_adaboost_regressor_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("AdaBoostRegressor/fit_predict");
    group.sample_size(10);
    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = AdaBoostRegressor::new(10)
                        .with_max_depth(3)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// CLUSTERING BENCHMARKS
// =============================================================================

fn bench_kmeans_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("KMeans/fit");
    group.sample_size(10);
    for (n_samples, n_features) in [(1000, 20), (5000, 50)] {
        let (x, _) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut model = KMeans::new(5).random_state(42);
                    model.fit(black_box(x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_agglomerative_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("AgglomerativeClustering/fit");
    group.sample_size(10);
    for (n_samples, n_features) in [(500, 20), (1000, 20)] {
        let (x, _) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut model = AgglomerativeClustering::new(5);
                    model.fit(black_box(x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// DECOMPOSITION BENCHMARKS
// =============================================================================

fn bench_pca_fit_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCA/fit_transform");
    for (n_samples, n_features) in [(1000, 50), (5000, 100)] {
        let (x, _) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut pca = PCA::new().with_n_components(10);
                    pca.fit_transform(black_box(x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_truncated_svd_fit_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("TruncatedSVD/fit_transform");
    for (n_samples, n_features) in [(1000, 50), (5000, 100)] {
        let (x, _) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut svd = TruncatedSVD::new().with_n_components(10);
                    svd.fit_transform(black_box(x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// NAIVE BAYES BENCHMARKS
// =============================================================================

fn bench_gaussian_nb_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("GaussianNB/fit_predict");
    for (n_samples, n_features) in [(1000, 20), (5000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = GaussianNB::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// EXTRA TREES BENCHMARKS
// =============================================================================

/// Benchmark ExtraTreesClassifier training time
fn bench_extra_trees_classifier_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExtraTreesClassifier/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(500, 20), (1000, 20)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = ExtraTreesClassifier::new()
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

/// Benchmark ExtraTreesRegressor training time
fn bench_extra_trees_regressor_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExtraTreesRegressor/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(500, 20), (1000, 20)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = ExtraTreesRegressor::new()
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
// ELASTICNET BENCHMARKS
// =============================================================================

/// Benchmark ElasticNet training time
fn bench_elastic_net_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("ElasticNet/fit");

    for (n_samples, n_features) in [(100, 10), (1000, 50), (5000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = ElasticNet::new(0.1, 0.5);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// LINEAR SVR BENCHMARKS
// =============================================================================

/// Benchmark LinearSVR training and prediction time
fn bench_linear_svr_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearSVR/fit_predict");
    group.sample_size(10);
    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = LinearSVR::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// NEAREST CENTROID BENCHMARKS
// =============================================================================

/// Benchmark NearestCentroid training and prediction time
fn bench_nearest_centroid_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("NearestCentroid/fit_predict");
    for (n_samples, n_features) in [(1000, 20), (5000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 5);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = NearestCentroid::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// BERNOULLI AND MULTINOMIAL NB BENCHMARKS
// =============================================================================

/// Benchmark BernoulliNB training and prediction time
fn bench_bernoulli_nb_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("BernoulliNB/fit_predict");
    for (n_samples, n_features) in [(1000, 20), (5000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = BernoulliNB::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Benchmark MultinomialNB training and prediction time
///
/// MultinomialNB requires non-negative features, so we use absolute values
fn bench_multinomial_nb_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultinomialNB/fit_predict");
    for (n_samples, n_features) in [(1000, 20), (5000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        // Make features non-negative for MultinomialNB
        let x = x.mapv(|v| v.abs());
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = MultinomialNB::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// ENSEMBLE BENCHMARKS (Voting, Stacking, Bagging)
// =============================================================================

/// Benchmark VotingClassifier with 3 heterogeneous estimators
fn bench_voting_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("VotingClassifier/fit_predict");
    group.sample_size(10);
    for (n_samples, n_features) in [(500, 20), (1000, 20)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let estimators: Vec<(
                        String,
                        Box<dyn ferroml_core::ensemble::voting::VotingClassifierEstimator>,
                    )> = vec![
                        ("lr".to_string(), Box::new(LogisticRegression::new())),
                        ("gnb".to_string(), Box::new(GaussianNB::new())),
                        ("dt".to_string(), Box::new(DecisionTreeClassifier::new())),
                    ];
                    let mut model = VotingClassifier::new(estimators);
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Benchmark StackingClassifier with 3 base estimators + LogisticRegression meta
fn bench_stacking_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("StackingClassifier/fit_predict");
    group.sample_size(10);
    let (x, y) = generate_classification_data(500, 20, 2);
    group.throughput(Throughput::Elements(500));
    group.bench_function("500x20", |b| {
        b.iter(|| {
            let estimators: Vec<(
                String,
                Box<dyn ferroml_core::ensemble::voting::VotingClassifierEstimator>,
            )> = vec![
                ("lr".to_string(), Box::new(LogisticRegression::new())),
                ("gnb".to_string(), Box::new(GaussianNB::new())),
                ("dt".to_string(), Box::new(DecisionTreeClassifier::new())),
            ];
            let mut model = StackingClassifier::new(estimators);
            model.fit(black_box(&x), black_box(&y)).unwrap();
            model.predict(black_box(&x)).unwrap()
        })
    });
    group.finish();
}

/// Benchmark BaggingClassifier with DecisionTree base estimator
fn bench_bagging_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("BaggingClassifier/fit_predict");
    group.sample_size(10);
    for (n_samples, n_features) in [(500, 20), (1000, 20)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let base: Box<dyn ferroml_core::ensemble::voting::VotingClassifierEstimator> =
                        Box::new(DecisionTreeClassifier::new());
                    let mut model = BaggingClassifier::new(base)
                        .with_n_estimators(10)
                        .with_random_state(42);
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// MLP (NEURAL NETWORK) BENCHMARKS
// =============================================================================

/// Benchmark MLPClassifier training and prediction time
fn bench_mlp_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("MLPClassifier/fit_predict");
    group.sample_size(10);

    let (x, y) = generate_classification_data(500, 20, 2);
    group.throughput(Throughput::Elements(500));
    group.bench_function("500x20_h64", |b| {
        b.iter(|| {
            let mut model = MLPClassifier::new()
                .hidden_layer_sizes(&[64])
                .activation(Activation::ReLU)
                .solver(Solver::Adam)
                .max_iter(50)
                .random_state(42);
            model.fit(black_box(&x), black_box(&y)).unwrap();
            model.predict(black_box(&x)).unwrap()
        })
    });
    group.finish();
}

/// Benchmark MLPRegressor training and prediction time
fn bench_mlp_regressor_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("MLPRegressor/fit_predict");
    group.sample_size(10);

    let (x, y) = generate_regression_data(500, 20);
    group.throughput(Throughput::Elements(500));
    group.bench_function("500x20_h64", |b| {
        b.iter(|| {
            let mut model = MLPRegressor::new()
                .hidden_layer_sizes(&[64])
                .activation(Activation::ReLU)
                .solver(Solver::Adam)
                .max_iter(50)
                .random_state(42);
            model.fit(black_box(&x), black_box(&y)).unwrap();
            model.predict(black_box(&x)).unwrap()
        })
    });
    group.finish();
}

// =============================================================================
// DBSCAN BENCHMARKS
// =============================================================================

/// Benchmark DBSCAN clustering
fn bench_dbscan_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("DBSCAN/fit");
    group.sample_size(10);
    for (n_samples, n_features) in [(500, 20), (1000, 20), (2000, 20)] {
        let (x, _) = generate_regression_data(n_samples, n_features);
        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut model = DBSCAN::new(0.5, 5);
                    model.fit(black_box(x)).unwrap()
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// POLYNOMIAL FEATURES BENCHMARKS
// =============================================================================

/// Benchmark PolynomialFeatures fit_transform time
fn bench_polynomial_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("PolynomialFeatures/fit_transform");

    // PolynomialFeatures expands combinatorially, so use smaller feature counts
    for (n_samples, n_features, degree) in [(1000, 10, 2), (1000, 10, 3), (5000, 5, 2)] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "config",
                format!("{}x{}_d{}", n_samples, n_features, degree),
            ),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut poly = PolynomialFeatures::new(degree);
                    poly.fit_transform(black_box(x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// PCA AS TRANSFORMER BENCHMARKS
// =============================================================================

/// Benchmark PCA used as a pure transformer (separate fit + transform)
fn bench_pca_transform_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCA/transform_only");

    for (n_samples, n_features) in [(1000, 50), (5000, 100)] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        // Pre-fit PCA
        let mut pca = PCA::new().with_n_components(10);
        pca.fit(&x).unwrap();

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| b.iter(|| pca.transform(black_box(x)).unwrap()),
        );
    }

    group.finish();
}

// =============================================================================
// CV-VARIANT LINEAR MODEL BENCHMARKS
// =============================================================================

/// Benchmark RidgeCV (cross-validated alpha selection)
fn bench_ridge_cv_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("RidgeCV/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = RidgeCV::new(vec![0.01, 0.1, 1.0, 10.0], 3);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark LassoCV (cross-validated alpha selection)
fn bench_lasso_cv_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("LassoCV/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = LassoCV::new(5, 3);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark ElasticNetCV (cross-validated alpha + l1_ratio selection)
fn bench_elastic_net_cv_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("ElasticNetCV/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = ElasticNetCV::new(5, vec![0.1, 0.5, 0.9], 3);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// ROBUST AND QUANTILE REGRESSION BENCHMARKS
// =============================================================================

/// Benchmark RobustRegression (Huber M-estimator) training time
fn bench_robust_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("RobustRegression/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = RobustRegression::new();
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark QuantileRegression training time (median regression, quantile=0.5)
/// Note: bootstrap inference is disabled (n_bootstrap=0) to benchmark the core
/// IRLS solver. With default n_bootstrap=200, each fit takes ~200x longer.
fn bench_quantile_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("QuantileRegression/fit");
    group.sample_size(10);

    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = QuantileRegression::new(0.5).with_n_bootstrap(0);
                    model.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// PASSIVE AGGRESSIVE AND RIDGE CLASSIFIER BENCHMARKS
// =============================================================================

/// Benchmark PassiveAggressiveClassifier training and prediction time
fn bench_passive_aggressive_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("PassiveAggressiveClassifier/fit_predict");

    for (n_samples, n_features) in [(1000, 50), (5000, 100)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = PassiveAggressiveClassifier::new(1.0);
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark RidgeClassifier training and prediction time
fn bench_ridge_classifier_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("RidgeClassifier/fit_predict");

    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, y) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut model = RidgeClassifier::new(1.0);
                    model.fit(black_box(*x), black_box(*y)).unwrap();
                    model.predict(black_box(*x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// CALIBRATED CLASSIFIER CV BENCHMARKS
// =============================================================================

/// Benchmark CalibratedClassifierCV with LogisticRegression base estimator
fn bench_calibrated_classifier_cv_fit_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("CalibratedClassifierCV/fit_predict");
    group.sample_size(10);

    let (x, y) = generate_classification_data(500, 20, 2);
    group.throughput(Throughput::Elements(500));
    group.bench_function("500x20_lr_sigmoid", |b| {
        b.iter(|| {
            let base: Box<dyn CalibrableClassifier> = Box::new(LogisticRegression::new());
            let mut model = CalibratedClassifierCV::new(base);
            model.fit(black_box(&x), black_box(&y)).unwrap();
            model.predict(black_box(&x)).unwrap()
        })
    });

    group.finish();
}

// =============================================================================
// LDA AND FACTOR ANALYSIS BENCHMARKS
// =============================================================================

/// Benchmark LDA (Linear Discriminant Analysis) fit_transform time
///
/// LDA is supervised dimensionality reduction, so it needs both X and y.
fn bench_lda_fit_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("LDA/fit_transform");
    group.sample_size(10);

    for (n_samples, n_features, n_classes) in [(1000, 50, 5), (5000, 100, 10)] {
        let (x, y) = generate_classification_data(n_samples, n_features, n_classes);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "samples",
                format!("{}x{}_c{}", n_samples, n_features, n_classes),
            ),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let mut lda = LDA::new().with_n_components(n_classes - 1);
                    lda.fit_transform(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark FactorAnalysis fit_transform time
fn bench_factor_analysis_fit_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("FactorAnalysis/fit_transform");
    group.sample_size(10);

    for (n_samples, n_features) in [(500, 20), (1000, 50)] {
        let (x, _) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("samples", format!("{}x{}", n_samples, n_features)),
            &x,
            |b, x| {
                b.iter(|| {
                    let mut fa = FactorAnalysis::new().with_n_factors(5);
                    fa.fit(black_box(x)).unwrap();
                    fa.transform(black_box(x)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// LARGE-SCALE BENCHMARKS (100K+ samples)
// =============================================================================

/// Benchmark LinearRegression on large datasets (100K+ samples)
///
/// Verifies O(n*p^2) scaling and establishes baseline for large-scale perf.
fn bench_large_scale_linear_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("LargeScale/LinearRegression");
    group.sample_size(10);

    for n_samples in [10000, 50000, 100000] {
        let n_features = 20;
        let (x, y) = generate_regression_data(n_samples, n_features);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("fit", n_samples),
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

/// Benchmark RandomForest prediction on large datasets (100K+ samples)
///
/// Tests prediction throughput which should be close to O(n * trees * depth).
fn bench_large_scale_random_forest_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("LargeScale/RandomForest/predict");
    group.sample_size(10);

    let n_features = 20;
    let (x_train, y_train) = generate_classification_data(2000, n_features, 2);

    let mut model = RandomForestClassifier::new()
        .with_n_estimators(50)
        .with_max_depth(Some(10))
        .with_random_state(42);
    model.fit(&x_train, &y_train).unwrap();

    for n_samples in [10000, 50000, 100000] {
        let (x_test, _) = generate_classification_data(n_samples, n_features, 2);

        group.throughput(Throughput::Elements(n_samples as u64));
        group.bench_with_input(BenchmarkId::new("samples", n_samples), &x_test, |b, x| {
            b.iter(|| model.predict(black_box(x)).unwrap())
        });
    }

    group.finish();
}

// =============================================================================
// SVC THRESHOLD SWEEP BENCHMARKS
// =============================================================================

/// Benchmark SVC training across dataset sizes to find the optimal
/// FULL_MATRIX_THRESHOLD crossover point between full kernel matrix and LRU cache.
fn bench_svc_threshold_sweep(c: &mut Criterion) {
    use ferroml_core::models::svm::Kernel;

    let mut group = c.benchmark_group("SVC_Threshold_Sweep");
    group.sample_size(10); // SVC training is slow
                           // Test at sizes that straddle the current threshold (2000)
    for n_samples in [1000, 1500, 2000, 2500, 3000] {
        let (x, y) = generate_classification_data(n_samples, 20, 2);
        group.bench_with_input(
            BenchmarkId::new("fit", n_samples),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    let mut svc = SVC::new()
                        .with_kernel(Kernel::Rbf { gamma: 0.1 })
                        .with_max_iter(200);
                    svc.fit(black_box(x), black_box(y)).unwrap()
                });
            },
        );
    }
    group.finish();
}

// =============================================================================
// LINEARSVC PERFORMANCE BENCHMARKS
// =============================================================================

/// Benchmark LinearSVC training across dataset sizes to establish baseline
/// performance and verify shrinking effectiveness.
fn bench_linear_svc_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearSVC_Performance");
    group.sample_size(10);
    for n_samples in [1000, 2000, 5000] {
        let (x, y) = generate_classification_data(n_samples, 50, 2);
        group.bench_with_input(
            BenchmarkId::new("fit", format!("{}x50", n_samples)),
            &(x.clone(), y.clone()),
            |b, (x, y)| {
                b.iter(|| {
                    let mut svc = LinearSVC::new().with_max_iter(1000);
                    svc.fit(black_box(x), black_box(y)).unwrap()
                });
            },
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

criterion_group!(
    svm_models,
    bench_svc_fit_predict,
    bench_linear_svc_fit_predict,
    bench_svr_fit_predict,
);

criterion_group!(
    knn_models,
    bench_knn_classifier_fit_predict,
    bench_knn_regressor_fit_predict,
);

criterion_group!(logistic_models, bench_logistic_regression_fit_predict,);

criterion_group!(
    sgd_models,
    bench_sgd_classifier_fit_predict,
    bench_sgd_regressor_fit_predict,
);

criterion_group!(
    adaboost_models,
    bench_adaboost_classifier_fit_predict,
    bench_adaboost_regressor_fit_predict,
);

criterion_group!(
    clustering_benches,
    bench_kmeans_fit,
    bench_agglomerative_fit,
);

criterion_group!(
    decomposition_benches,
    bench_pca_fit_transform,
    bench_truncated_svd_fit_transform,
);

criterion_group!(
    naive_bayes_benches,
    bench_gaussian_nb_fit_predict,
    bench_bernoulli_nb_fit_predict,
    bench_multinomial_nb_fit_predict,
);

criterion_group!(
    extra_trees_benches,
    bench_extra_trees_classifier_fit,
    bench_extra_trees_regressor_fit,
);

criterion_group!(elastic_net_benches, bench_elastic_net_fit,);

criterion_group!(svm_extended_benches, bench_linear_svr_fit_predict,);

criterion_group!(svc_threshold_benches, bench_svc_threshold_sweep,);

criterion_group!(linear_svc_perf_benches, bench_linear_svc_sizes,);

criterion_group!(knn_extended_benches, bench_nearest_centroid_fit_predict,);

criterion_group!(
    ensemble_benches,
    bench_voting_classifier_fit_predict,
    bench_stacking_classifier_fit_predict,
    bench_bagging_classifier_fit_predict,
);

criterion_group!(
    neural_benches,
    bench_mlp_classifier_fit_predict,
    bench_mlp_regressor_fit_predict,
);

criterion_group!(dbscan_benches, bench_dbscan_fit,);

criterion_group!(
    preprocessing_extended,
    bench_polynomial_features,
    bench_pca_transform_only,
);

criterion_group!(
    large_scale_benches,
    bench_large_scale_linear_regression,
    bench_large_scale_random_forest_predict,
);

criterion_group!(
    cv_linear_benches,
    bench_ridge_cv_fit,
    bench_lasso_cv_fit,
    bench_elastic_net_cv_fit,
);

criterion_group!(
    robust_quantile_benches,
    bench_robust_regression_fit,
    bench_quantile_regression_fit,
);

criterion_group!(
    additional_classifier_benches,
    bench_passive_aggressive_fit_predict,
    bench_ridge_classifier_fit_predict,
    bench_calibrated_classifier_cv_fit_predict,
);

criterion_group!(
    decomposition_extended_benches,
    bench_lda_fit_transform,
    bench_factor_analysis_fit_transform,
);

criterion_main!(
    linear_models,
    tree_models,
    gradient_boosting,
    hist_gradient_boosting,
    preprocessing,
    scaling,
    svm_models,
    knn_models,
    logistic_models,
    sgd_models,
    adaboost_models,
    clustering_benches,
    decomposition_benches,
    naive_bayes_benches,
    extra_trees_benches,
    elastic_net_benches,
    svm_extended_benches,
    knn_extended_benches,
    ensemble_benches,
    neural_benches,
    dbscan_benches,
    preprocessing_extended,
    large_scale_benches,
    cv_linear_benches,
    robust_quantile_benches,
    additional_classifier_benches,
    decomposition_extended_benches,
    svc_threshold_benches,
    linear_svc_perf_benches
);
