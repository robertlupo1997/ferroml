//! GPU vs CPU benchmark comparisons
//!
//! Run with: `cargo bench --bench gpu_benchmarks -p ferroml-core --features gpu`
//!
//! Benchmarks skip gracefully if no GPU adapter is available.

#![cfg(feature = "gpu")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ferroml_core::gpu::{GpuBackend, WgpuBackend};
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

fn make_random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>())
}

// =============================================================================
// RAW GEMM BENCHMARKS
// =============================================================================

fn bench_gemm_cpu_vs_gpu(c: &mut Criterion) {
    let gpu = match WgpuBackend::try_new() {
        Some(b) => Arc::new(b),
        None => {
            eprintln!("No GPU adapter found, skipping GPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("GEMM/CPU_vs_GPU");
    group.sample_size(10);

    for size in [256, 512, 1024] {
        let a = make_random_matrix(size, size, 42);
        let b = make_random_matrix(size, size, 43);

        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_with_input(
            BenchmarkId::new("CPU", format!("{}x{}", size, size)),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(*a).dot(black_box(*b))),
        );

        let gpu_ref = gpu.clone();
        group.bench_with_input(
            BenchmarkId::new("GPU", format!("{}x{}", size, size)),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| gpu_ref.matmul(black_box(a), black_box(b)).unwrap()),
        );
    }

    group.finish();
}

// =============================================================================
// MLP FORWARD PASS BENCHMARKS
// =============================================================================

fn bench_mlp_forward_cpu_vs_gpu(c: &mut Criterion) {
    use ferroml_core::neural::{Activation, Layer, WeightInit};

    let gpu = match WgpuBackend::try_new() {
        Some(b) => Arc::new(b),
        None => return,
    };

    let mut group = c.benchmark_group("MLP_Forward/CPU_vs_GPU");
    group.sample_size(10);

    for (batch, hidden) in [(100, 256), (1000, 256), (5000, 1024)] {
        let input = make_random_matrix(batch, hidden, 42);
        let mut layer_cpu = Layer::new(
            hidden,
            hidden,
            Activation::ReLU,
            WeightInit::HeUniform,
            Some(42),
        );
        let mut layer_gpu = layer_cpu.clone();

        group.throughput(Throughput::Elements(batch as u64));

        group.bench_with_input(
            BenchmarkId::new("CPU", format!("{}x{}", batch, hidden)),
            &input,
            |bench, x| bench.iter(|| layer_cpu.forward(black_box(x), false).unwrap()),
        );

        let gpu_ref = gpu.clone();
        group.bench_with_input(
            BenchmarkId::new("GPU", format!("{}x{}", batch, hidden)),
            &input,
            |bench, x| {
                bench.iter(|| {
                    layer_gpu
                        .forward_gpu(black_box(x), false, gpu_ref.as_ref())
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// KMEANS DISTANCE BENCHMARKS
// =============================================================================

fn bench_kmeans_distance_cpu_vs_gpu(c: &mut Criterion) {
    let gpu = match WgpuBackend::try_new() {
        Some(b) => Arc::new(b),
        None => return,
    };

    let mut group = c.benchmark_group("KMeans_Distance/CPU_vs_GPU");
    group.sample_size(10);

    let n_clusters = 10;
    let n_features = 20;

    for n_samples in [1_000, 10_000, 50_000] {
        let x = make_random_matrix(n_samples, n_features, 42);
        let centers = make_random_matrix(n_clusters, n_features, 43);

        group.throughput(Throughput::Elements(n_samples as u64));

        group.bench_with_input(
            BenchmarkId::new("CPU", n_samples),
            &(&x, &centers),
            |bench, (x, centers)| {
                bench.iter(|| {
                    // CPU: compute distances row by row
                    let mut dists = Array2::zeros((x.nrows(), centers.nrows()));
                    for i in 0..x.nrows() {
                        for j in 0..centers.nrows() {
                            let mut d = 0.0;
                            for f in 0..x.ncols() {
                                let diff = x[[i, f]] - centers[[j, f]];
                                d += diff * diff;
                            }
                            dists[[i, j]] = d;
                        }
                    }
                    black_box(dists)
                })
            },
        );

        let gpu_ref = gpu.clone();
        group.bench_with_input(
            BenchmarkId::new("GPU", n_samples),
            &(&x, &centers),
            |bench, (x, centers)| {
                bench.iter(|| {
                    gpu_ref
                        .pairwise_distances(black_box(x), black_box(centers))
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    gpu_benchmarks,
    bench_gemm_cpu_vs_gpu,
    bench_mlp_forward_cpu_vs_gpu,
    bench_kmeans_distance_cpu_vs_gpu,
);

criterion_main!(gpu_benchmarks);
