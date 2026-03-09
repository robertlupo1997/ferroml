# FerroML Benchmark Results — Post-Optimization (Plan N)

**Date**: 2026-03-09
**Commit**: 55ae057 (Plan N phases N.1-N.8)
**Hardware**: WSL2 Linux 6.6.87.2, x86_64
**Rust**: stable, `--release` (Criterion bench profile)
**Features**: default (parallel, onnx, simd) — SIMD now default
**Total benchmarks**: 253 across 3 bench files

## Optimization Summary

| Algorithm | Baseline | Optimized | Speedup | Technique |
|-----------|----------|-----------|---------|-----------|
| **DBSCAN/fit/500x20** | 4.01 ms | 1.65 ms | **2.43x** | Squared distances + SIMD |
| **DBSCAN/fit/1000x20** | 15.80 ms | 6.60 ms | **2.39x** | Squared distances + SIMD |
| **DBSCAN/fit/2000x20** | 62.95 ms | 26.56 ms | **2.37x** | Squared distances + SIMD |
| **RF Classifier/fit/1000x20** | 2.00 ms | 1.38 ms | **1.45x** | Deferred index collection |
| **GB Classifier/fit/1000x20** | 15.81 ms | 10.93 ms | **1.45x** | Pre-allocated buffers, in-place ops |
| **HistGB Regressor/fit/1000x20** | 90.16 ms | 64.95 ms | **1.39x** | Eliminated to_vec(), buffer reuse |
| **DT Classifier/fit/5000x50** | 29.02 ms | 21.28 ms | **1.36x** | Deferred index collection |
| **HistGB predict/10000** | 29.54 ms | 24.38 ms | **1.21x** | Eliminated per-sample allocations |
| **Lasso/fit/5000x50** | 1.46 ms | 1.14 ms | **1.28x** | SIMD default |
| **LinearReg/fit/10000x100** | 239.05 ms | 200.27 ms | **1.19x** | SIMD default |
| **GB Classifier/predict/5000x50** | 1.19 ms | 1.02 ms | **1.17x** | In-place prediction accumulation |
| **GB Regressor/fit/1000x20** | 33.95 ms | 29.14 ms | **1.17x** | Pre-allocated buffers |
| **DT Regressor/fit/5000x50** | 160.07 ms | 144.98 ms | **1.10x** | Deferred index collection |
| **t-SNE (Barnes-Hut)** | O(N^2) exact | O(N log N) | **~10x** at N=5000 | VP-tree + QuadTree |

### Algorithms Meeting 2x Target
1. **DBSCAN** — 2.37-2.43x consistently across all dataset sizes

### New Capabilities (not direct speedups)
2. **Barnes-Hut t-SNE** — O(N log N) approximation, makes t-SNE feasible for N > 5000
3. **Parallel GB/KNN predict** — scales with cores for large prediction batches (>256/128 samples)

## Optimizations Applied

### Phase N.2: SIMD Default
- SIMD (`wide::f64x4`) now enabled by default for all distance computations
- DBSCAN uses squared distances (avoids sqrt) + SIMD in region_query hot loop

### Phase N.3: Tree Splitting
- Deferred index collection: only allocate left/right index vectors for the winning split
- Avoids Vec allocation on every candidate split improvement

### Phase N.4: Gradient Boosting
- Pre-allocated subsample buffers reused across boosting iterations
- In-place prediction updates via `zip_mut_with` (avoids 2 temporary arrays per tree)

### Phase N.5: Barnes-Hut t-SNE
- VP-tree for O(N log N) nearest neighbor search (sparse P matrix)
- QuadTree for Barnes-Hut force approximation with configurable theta
- Auto-selects Barnes-Hut for N > 1000

### Phase N.6: Memory Reduction
- HistGB: eliminated `to_vec()` per sample per tree in predict (→ `as_slice()`)
- Pre-allocated gradient/hessian buffers in both boosting variants
- KMeans: buffer pre-allocation outside iteration loop
- Inline softmax computation without temporary Vec

### Phase N.7: Parallel Prediction
- GB regressor/classifier predict: rayon `par_chunks` for batches > 256 samples
- KNN classifier/regressor predict: rayon `par_iter` for batches > 128 samples
- SVM: incremental error cache update O(n) vs O(n * n_sv) per SMO step

### Phase N.8: Benchmark CI
- `.github/workflows/benchmarks.yml`: automatic regression detection on PRs
- `scripts/check_bench_regressions.py`: parses Criterion output, flags > 15% regressions

## Key Benchmark Results (All Algorithms)

### Distance-Based
| Benchmark | Time |
|-----------|------|
| DBSCAN/fit/500x20 | 1.65 ms |
| DBSCAN/fit/1000x20 | 6.60 ms |
| DBSCAN/fit/2000x20 | 26.56 ms |
| KNeighborsClassifier/1000x50 | 22.14 ms |
| KNeighborsRegressor/1000x50 | 41.76 ms |
| KMeans/fit/1000x20 | 155.98 ms |
| KMeans/fit/5000x50 | 695.90 ms |

### Tree-Based
| Benchmark | Time |
|-----------|------|
| DecisionTreeClassifier/fit/100x10 | 40.86 µs |
| DecisionTreeClassifier/fit/1000x50 | 3.17 ms |
| DecisionTreeClassifier/fit/5000x50 | 21.28 ms |
| DecisionTreeRegressor/fit/1000x50 | 19.80 ms |
| DecisionTreeRegressor/fit/5000x50 | 144.98 ms |
| RandomForestClassifier/fit/1000x20 | 1.38 ms |
| RandomForestRegressor/fit/1000x20 | 5.31 ms |

### Gradient Boosting
| Benchmark | Time |
|-----------|------|
| GradientBoostingClassifier/fit/1000x20 | 10.93 ms |
| GradientBoostingRegressor/fit/1000x20 | 29.14 ms |
| GB Classifier/predict/5000x50 | 1.02 ms |
| GB Regressor/predict/5000x50 | 1.49 ms |
| HistGBClassifier/fit/1000x20 | 50.53 ms |
| HistGBRegressor/fit/1000x20 | 64.95 ms |
| HistGB predict/10000 | 24.38 ms |

### SVM
| Benchmark | Time |
|-----------|------|
| SVC/fit_predict/500x20 | 60.16 ms |
| SVR/fit_predict/500x20 | 25.73 ms |
| LinearSVC/fit_predict/1000x50 | 49.17 ms |
| LinearSVR/fit_predict/1000x50 | 45.26 ms |
