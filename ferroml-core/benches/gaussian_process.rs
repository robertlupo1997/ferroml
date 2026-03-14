//! Gaussian Process benchmarks: Exact GP vs Sparse GP (FITC/VFE) vs SVGP.
//!
//! Run with: `cargo bench -p ferroml-core --bench gaussian_process`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ferroml_core::models::gaussian_process::{
    select_inducing_points, GaussianProcessRegressor, InducingPointMethod, SVGPRegressor,
    SparseApproximation, SparseGPRegressor, RBF,
};
use ferroml_core::models::Model;
use ndarray::{Array1, Array2};

/// Generate synthetic sin data for GP benchmarks.
fn make_sin_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    use rand::prelude::*;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let x_vals: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..10.0)).collect();
    let x = Array2::from_shape_vec((n, 1), x_vals).unwrap();
    let y = x.column(0).mapv(|v| v.sin() + 0.1 * rng.random::<f64>());
    (x, y)
}

// =============================================================================
// Exact GP benchmarks
// =============================================================================

fn bench_exact_gpr_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExactGPR/fit");

    for n in [100, 500, 1000] {
        let (x, y) = make_sin_data(n);
        group.bench_with_input(BenchmarkId::new("n", n), &(&x, &y), |b, (x, y)| {
            b.iter(|| {
                let kernel = RBF::new(1.0);
                let mut gpr = GaussianProcessRegressor::new(Box::new(kernel)).with_alpha(0.01);
                gpr.fit(black_box(*x), black_box(*y)).unwrap()
            })
        });
    }

    group.finish();
}

// =============================================================================
// FITC benchmarks
// =============================================================================

fn bench_fitc_gpr_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("FITC_GPR/fit");

    for (n, m) in [(1000, 50), (5000, 100)] {
        let (x, y) = make_sin_data(n);
        group.bench_with_input(
            BenchmarkId::new("n_m", format!("{n}_{m}")),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let kernel = RBF::new(1.0);
                    let mut sgpr = SparseGPRegressor::new(Box::new(kernel))
                        .with_n_inducing(m)
                        .with_alpha(0.01)
                        .with_approximation(SparseApproximation::FITC);
                    sgpr.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_fitc_gpr_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("FITC_GPR/predict");

    let (x, y) = make_sin_data(1000);
    let kernel = RBF::new(1.0);
    let mut sgpr = SparseGPRegressor::new(Box::new(kernel))
        .with_n_inducing(50)
        .with_alpha(0.01)
        .with_approximation(SparseApproximation::FITC);
    sgpr.fit(&x, &y).unwrap();

    group.bench_function(BenchmarkId::new("n", 1000), |b| {
        b.iter(|| sgpr.predict(black_box(&x)).unwrap())
    });

    group.finish();
}

// =============================================================================
// VFE benchmarks
// =============================================================================

fn bench_vfe_gpr_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("VFE_GPR/fit");

    for (n, m) in [(1000, 50), (5000, 100)] {
        let (x, y) = make_sin_data(n);
        group.bench_with_input(
            BenchmarkId::new("n_m", format!("{n}_{m}")),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let kernel = RBF::new(1.0);
                    let mut sgpr = SparseGPRegressor::new(Box::new(kernel))
                        .with_n_inducing(m)
                        .with_alpha(0.01)
                        .with_approximation(SparseApproximation::VFE);
                    sgpr.fit(black_box(*x), black_box(*y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// SVGP benchmarks
// =============================================================================

fn bench_svgp_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVGP/fit");
    group.sample_size(10); // SVGP is slower, reduce iterations

    let (x, y) = make_sin_data(10000);
    group.bench_function(BenchmarkId::new("n10000_m100_e1", ""), |b| {
        b.iter(|| {
            let kernel = RBF::new(1.0);
            let mut svgp = SVGPRegressor::new(Box::new(kernel))
                .with_n_inducing(100)
                .with_noise_variance(0.01)
                .with_n_epochs(1)
                .with_batch_size(256);
            svgp.fit(black_box(&x), black_box(&y)).unwrap()
        })
    });

    group.finish();
}

// =============================================================================
// Inducing point selection benchmarks
// =============================================================================

fn bench_inducing_point_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("InducingPoints/select");

    let (x, _) = make_sin_data(1000);
    let kernel = RBF::new(1.0);

    group.bench_function(BenchmarkId::new("kmeans", "1000_m50"), |b| {
        b.iter(|| {
            let method = InducingPointMethod::KMeans {
                max_iter: 100,
                seed: Some(42),
            };
            select_inducing_points(
                black_box(&x),
                50,
                &method,
                &kernel as &dyn ferroml_core::models::gaussian_process::Kernel,
            )
            .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("greedy", "1000_m50"), |b| {
        b.iter(|| {
            let method = InducingPointMethod::GreedyVariance { seed: Some(42) };
            select_inducing_points(
                black_box(&x),
                50,
                &method,
                &kernel as &dyn ferroml_core::models::gaussian_process::Kernel,
            )
            .unwrap()
        })
    });

    group.bench_function(BenchmarkId::new("random", "1000_m50"), |b| {
        b.iter(|| {
            let method = InducingPointMethod::RandomSubset { seed: Some(42) };
            select_inducing_points(
                black_box(&x),
                50,
                &method,
                &kernel as &dyn ferroml_core::models::gaussian_process::Kernel,
            )
            .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    gp_benches,
    bench_exact_gpr_fit,
    bench_fitc_gpr_fit,
    bench_fitc_gpr_predict,
    bench_vfe_gpr_fit,
    bench_svgp_fit,
    bench_inducing_point_selection,
);
criterion_main!(gp_benches);
