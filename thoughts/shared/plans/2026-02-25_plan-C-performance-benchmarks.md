# Plan C: Performance Benchmark Hardening

**Date:** 2026-02-25
**Priority:** MEDIUM (no bugs, infrastructure exists, needs expansion)
**Files:** `ferroml-core/benches/` (2,415 lines, 45 benchmark functions)
**Estimated New Benchmarks:** ~15 functions
**Parallel-Safe:** Yes (no overlap with clustering/neural/preprocessing plans)

## Overview

FerroML has solid Criterion benchmark infrastructure covering ~23 of 60+ models (38%). All benchmarks compile and run. This plan expands coverage to key missing models, adds baseline tracking for regression detection, and verifies SIMD/parallel optimizations actually help.

## Current State

### What Exists
- `benchmarks.rs` (1,407 lines): 45 benchmark functions across linear, tree, boosting, SVM, KNN, preprocessing
- `memory_benchmarks.rs` (553 lines): Memory profiling for 6 categories
- `performance_optimizations.rs` (270 lines): SIMD, parallel, scaling benchmarks
- `gpu_benchmarks.rs` (185 lines): GPU vs CPU (requires `gpu` feature)
- Criterion v0.5 configured with `harness = false`
- All compile cleanly

### What's Missing
- **37 models unbenchmarked** (~62%): ExtraTrees, ElasticNet/CV variants, RobustRegression, PassiveAggressive, BernoulliNB/MultinomialNB, NearestCentroid, RidgeClassifier, CalibratedClassifierCV, Voting/Stacking/Bagging, MLP, DBSCAN
- **No baseline tracking**: No `baseline.json`, no CI regression detection
- **No cross-library comparison**: No sklearn timing baselines
- **Limited scale**: Most benchmarks use 100-5000 samples, no 100K+ tests

### Benchmark Coverage Map

| Category | Benchmarked | Missing |
|----------|------------|---------|
| Linear Models | LinearRegression, Ridge, Lasso | ElasticNet, ElasticNetCV, LassoCV, RidgeCV, RobustRegression, QuantileRegression |
| Tree Models | DT, RF (classifier+regressor) | ExtraTreesClassifier, ExtraTreesRegressor |
| Boosting | GB, HistGB (classifier+regressor) | AdaBoostClassifier, AdaBoostRegressor |
| SVM | SVC, LinearSVC, SVR | LinearSVR |
| KNN | KNeighborsClassifier, KNeighborsRegressor | NearestCentroid |
| Naive Bayes | GaussianNB | BernoulliNB, MultinomialNB |
| Ensemble | — | VotingClassifier, StackingClassifier, BaggingClassifier |
| Neural | — | MLPClassifier, MLPRegressor |
| Clustering | KMeans, Agglomerative | DBSCAN |
| Preprocessing | StandardScaler, MinMax, Robust, MaxAbs | PolynomialFeatures, PCA (as transformer) |
| Decomposition | PCA, TruncatedSVD | LDA, FactorAnalysis |

## Desired End State

- 60+ benchmark functions covering all major models
- Baseline JSON for regression detection
- SIMD/parallel speedup verified with data
- Documentation of FerroML performance characteristics

## Implementation Phases

### Phase C.1: Add Missing Model Benchmarks (~15 functions)

**File:** `ferroml-core/benches/benchmarks.rs` (MODIFY)

Priority additions (highest impact models first):

```rust
// 1. ExtraTrees (popular sklearn model, large codebase)
fn bench_extra_trees_classifier(c: &mut Criterion) {
    let (x, y) = make_classification(1000, 20);
    c.bench_function("ExtraTreesClassifier fit (1000x20)", |b| {
        b.iter(|| {
            let mut model = ExtraTreesClassifier::new()
                .n_estimators(100)
                .random_state(42);
            model.fit(&x, &y).unwrap();
        })
    });
}

// 2. ExtraTreesRegressor
// 3. AdaBoostClassifier
// 4. AdaBoostRegressor
// 5. MLPClassifier (small: [64], medium: [128, 64])
// 6. MLPRegressor
// 7. DBSCAN
// 8. ElasticNet
// 9. VotingClassifier (3 estimators)
// 10. StackingClassifier (3 base + meta)
// 11. BaggingClassifier
// 12. BernoulliNB
// 13. MultinomialNB
// 14. PolynomialFeatures (degree=2, degree=3)
// 15. LDA (decomposition)
```

Each benchmark uses standardized dataset sizes:
- **Small**: 1,000 samples x 20 features
- **Medium**: 10,000 samples x 50 features
- **Large** (select models): 100,000 samples x 100 features

### Phase C.2: Add Scaling Benchmarks

**File:** `ferroml-core/benches/benchmarks.rs` (MODIFY)

Add sample-count scaling for key models to verify O(n) vs O(n log n) vs O(n²):

```rust
fn bench_scaling_linear_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearRegression_scaling");
    for n in [100, 1000, 10000, 100000] {
        let (x, y) = make_regression(n, 20);
        group.bench_with_input(
            BenchmarkId::new("fit", n),
            &(x, y),
            |b, (x, y)| { b.iter(|| LinearRegression::new().fit(x, y).unwrap()) },
        );
    }
    group.finish();
}
```

Add scaling benchmarks for: LinearRegression, RandomForest, HistGradientBoosting, KMeans, DBSCAN, KNN.

### Phase C.3: Baseline Tracking

**File:** `benchmarks/baseline.json` (NEW)

After running all benchmarks once, save results:
```json
{
    "timestamp": "2026-02-25T00:00:00Z",
    "commit": "0160c35",
    "benchmarks": {
        "LinearRegression fit (1000x20)": {
            "mean_ns": 450000,
            "std_ns": 12000,
            "throughput_samples_per_sec": 2222222
        },
        ...
    }
}
```

**File:** `scripts/check_benchmark_regression.sh` (NEW)
```bash
#!/bin/bash
# Run benchmarks and compare against baseline
# Exit 1 if any benchmark regressed >20%
cargo bench -p ferroml-core --bench benchmarks -- --output-format=bencher 2>/dev/null | \
    python3 scripts/compare_benchmarks.py benchmarks/baseline.json
```

### Phase C.4: SIMD Verification

**File:** `ferroml-core/benches/performance_optimizations.rs` (MODIFY)

Add explicit SIMD vs scalar comparison:
```rust
fn bench_simd_vs_scalar_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");
    let a = Array1::random(10000, Uniform::new(-1.0, 1.0));
    let b = Array1::random(10000, Uniform::new(-1.0, 1.0));

    group.bench_function("scalar_euclidean_10k", |bench| {
        bench.iter(|| scalar_euclidean_distance(&a, &b))
    });

    #[cfg(feature = "simd")]
    group.bench_function("simd_euclidean_10k", |bench| {
        bench.iter(|| simd_euclidean_distance(&a, &b))
    });

    group.finish();
}
```

### Phase C.5: Cross-Library Comparison Script

**File:** `benchmarks/sklearn_comparison.py` (NEW)

Python script that runs the same benchmarks with sklearn and outputs timing JSON:

```python
import time, json
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_classification, make_regression

results = {}
for name, Model, make_data, kwargs in BENCHMARKS:
    X, y = make_data(n_samples=1000, n_features=20, random_state=42)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        Model(**kwargs).fit(X, y)
        times.append(time.perf_counter() - start)
    results[name] = {
        "mean_s": statistics.mean(times),
        "std_s": statistics.stdev(times),
    }

with open("benchmarks/sklearn_baseline.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Success Criteria

- [ ] `cargo bench -p ferroml-core` completes without errors
- [ ] 60+ benchmark functions (up from 45)
- [ ] All major model categories have at least one benchmark
- [ ] Scaling benchmarks show expected complexity (linear for OLS, n log n for trees)
- [ ] `baseline.json` generated and committed
- [ ] sklearn comparison script runs and produces timing data
- [ ] SIMD benchmarks show measurable speedup (>1.5x on 10K+ vectors)

## Dependencies

- Criterion v0.5 (existing)
- Python 3.10+ with sklearn, numpy for comparison script
- No new Rust crate dependencies

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Benchmarks take too long in CI | Use `--sample-size 10` for CI, full runs locally |
| SIMD speedup varies by CPU | Document expected speedup range, test on multiple CPUs |
| sklearn comparison is apples-to-oranges | Document methodology clearly, note cold/warm start |
| Some models may be too slow for 100K samples | Only scale-test models with O(n log n) or better complexity |
