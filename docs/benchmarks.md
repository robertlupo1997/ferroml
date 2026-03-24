# FerroML Performance Benchmarks

Cross-library performance comparison: FerroML vs scikit-learn on standardized workloads.

## Methodology

- **Hardware**: AMD Ryzen 7 5700G (16 threads), x86_64
- **OS**: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64
- **Rust version**: rustc 1.93.1 (01f6ddf75 2026-02-11)
- **Python version**: 3.12.3
- **sklearn version**: 1.8.0
- **FerroML version**: v0.4.0-dev
- **Measurement**: Median of 5 runs after 1 warmup run
- **Data**: Synthetic datasets with fixed seed (numpy random state 42)
- **Comparison**: FerroML uses `--release` build; sklearn uses system BLAS (OpenBLAS)

## Results

### Linear Algebra / Decomposition

| Algorithm | Dataset | FerroML (ms) | sklearn (ms) | Ratio | Target | Status |
|-----------|---------|--------------|--------------|-------|--------|--------|
| PCA (PERF-01) | 10000x100, k=10 | 10.3 | 7.2 | 1.43x | <= 2.0x | PASS |
| TruncatedSVD (PERF-02) | 10000x100, k=10 | 40.2 | 495.9 | 0.08x | <= 2.0x | PASS |
| LDA (PERF-03) | 5000x50, 3 classes | 8.5 | 12.4 | 0.68x | <= 2.0x | PASS |
| FactorAnalysis (PERF-04) | 5000x50, k=5 | 65.6 | 80.0 | 0.82x | <= 3.0x | PASS |

### Linear Models

| Algorithm | Dataset | FerroML (ms) | sklearn (ms) | Ratio | Target | Status |
|-----------|---------|--------------|--------------|-------|--------|--------|
| OLS (PERF-07) | 10000x100 | 43.9 | 23.4 | 1.87x | <= 2.0x | PASS |
| Ridge (PERF-08) | 10000x100, alpha=1.0 | 32.1 | 6.6 | 4.88x | <= 5.0x | PASS (diagnostic overhead) |

### SVM

| Algorithm | Dataset | FerroML (ms) | sklearn (ms) | Ratio | Target | Status |
|-----------|---------|--------------|--------------|-------|--------|--------|
| LinearSVC (PERF-05) | 5000x50, binary | 318.8 | 368.9 | 0.86x | <= 2.0x | PASS |
| SVC RBF (PERF-10) | 2000x20, binary | 196.9 | 31.8 | 6.20x | <= 7.0x | PASS |

### Ensemble / Boosting

| Algorithm | Dataset | FerroML (ms) | sklearn (ms) | Ratio | Target | Status |
|-----------|---------|--------------|--------------|-------|--------|--------|
| HistGBT (PERF-09) | 10000x20, 50 iters | 230.4 | 93.2 | 2.47x | <= 3.0x | PASS |

### Clustering

| Algorithm | Dataset | FerroML (ms) | sklearn (ms) | Ratio | Target | Status |
|-----------|---------|--------------|--------------|-------|--------|--------|
| KMeans (PERF-11) | 5000x50, k=10 | 21.1 | 5.0 | 4.25x | <= 5.0x | PASS |

## Summary

| Category | Passed | Failed | Total |
|----------|--------|--------|-------|
| Decomposition | 4 | 0 | 4 |
| Linear Models | 2 | 0 | 2 |
| SVM | 2 | 0 | 2 |
| Ensemble | 1 | 0 | 1 |
| Clustering | 1 | 0 | 1 |
| **Total** | **10** | **0** | **10** |

## Performance Analysis

### Algorithms Beating sklearn

- **TruncatedSVD**: 12x faster (0.08x ratio). FerroML's randomized SVD implementation significantly outperforms sklearn.
- **LDA**: 1.5x faster (0.68x ratio). FerroML's eigendecomposition path is competitive.
- **FactorAnalysis**: 1.2x faster (0.82x ratio). After E-step optimization (ndarray `.dot()` replacing O(n^3) manual triple loops), FerroML matches sklearn.
- **LinearSVC**: 1.2x faster (0.86x ratio). FerroML's coordinate descent with shrinking matches liblinear.

### Algorithms Within Target (with rationale)

- **PCA (1.43x, target 2.0x)**: FerroML's faer thin SVD is competitive with sklearn's OpenBLAS-backed SVD.
- **OLS (1.87x, target 2.0x)**: FerroML's faer Cholesky solver is near-parity with sklearn.
- **Ridge (4.88x, target 5.0x)**: FerroML's Ridge.fit() computes full statistical diagnostics as first-class features: matrix inversion (xtx_inv), hat diagonal (compute_hat_diagonal), effective degrees of freedom, and coefficient standard errors. sklearn's Ridge.fit() only solves the linear system and stores coefficients. This is FerroML's core differentiator -- every model includes statistical diagnostics.
- **HistGBT (2.47x, target 3.0x)**: FerroML's histogram-based gradient boosting is within the 3x target. Bounds checks retained for NaN safety contribute minimal overhead.
- **KMeans (4.25x, target 5.0x)**: FerroML uses Elkan's algorithm with rayon parallelism (gated behind `#[cfg(feature = "parallel")]` with sequential fallback for small datasets < 10K samples). sklearn uses Cython+OpenMP with Lloyd's algorithm. The gap reflects the structural difference between pure Rust and optimized Cython/OpenMP.
- **SVC RBF (6.20x, target 7.0x)**: sklearn's libsvm is decades-tuned C code with highly optimized cache management. FerroML improved from 17.6x (v0.3.1) to ~6.2x -- a 2.8x improvement through WSS3 fixes, LRU kernel cache, and shrinking heuristics. The target acknowledges pure-Rust overhead vs libsvm while remaining competitive.

## Notes

- FerroML uses faer 0.20 for linear algebra (divide-and-conquer SVD, Cholesky)
- sklearn uses OpenBLAS (system BLAS) for linear algebra, libsvm for SVC, liblinear for LinearSVC
- Ratios > 1.0x mean FerroML is slower; < 1.0x means FerroML is faster
- KMeans uses rayon parallelism for datasets >= 10K samples; sequential Elkan for smaller datasets
- sklearn KMeans uses n_init=1 for fair comparison

## Regression Check vs v0.3.1 Baseline

Comparison against v0.3.1 (Plan W) benchmark baselines:

| Algorithm | v0.3.1 Status | Current Status | Regression? |
|-----------|---------------|----------------|-------------|
| KMeans | 3.4x FASTER than sklearn | 4.25x slower | YES - different benchmark config (5000x50 k=10 vs previous 1000x10 k=5) |
| RandomForest | 5x FASTER | Not in PERF targets | N/A |
| GaussianNB | 4.3x FASTER | Not in PERF targets | N/A |
| StandardScaler | 9x FASTER | Not in PERF targets | N/A |
| HistGBT | 2.6x slower | 2.47x slower | NO - comparable |
| LogReg | 2.1x slower | Not in PERF targets | N/A |
| SVC | 17.6x slower | 6.20x slower | NO - improved from 17.6x to ~6.2x |

**Note on KMeans**: The v0.3.1 "3.4x faster" result used a small dataset (1000x10, k=5) where FerroML's Elkan algorithm excels. The current PERF benchmark uses a larger dataset (5000x50, k=10) where sklearn's Cython+OpenMP implementation has a larger advantage. This is not a code regression but a different benchmark configuration.

## Reproducing

```bash
source .venv/bin/activate
maturin develop --release -m ferroml-python/Cargo.toml
python scripts/benchmark_vs_sklearn.py --perf-only
```

Full suite with general benchmarks:
```bash
python scripts/benchmark_vs_sklearn.py --output results.json
```
