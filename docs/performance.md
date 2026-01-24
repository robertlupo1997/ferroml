# FerroML Performance Guide

This document describes FerroML's performance characteristics, memory requirements, and optimization strategies.

## Table of Contents

1. [Memory Requirements](#memory-requirements)
2. [Time Complexity](#time-complexity)
3. [Memory Profiling](#memory-profiling)
4. [Performance Regression Testing](#performance-regression-testing)
5. [Optimization Guidelines](#optimization-guidelines)
6. [Benchmarking](#benchmarking)

---

## Memory Requirements

### Overview

FerroML is designed for memory efficiency while maintaining statistical rigor. Memory usage varies significantly by model type and dataset size.

### Model-Specific Memory Requirements

#### Linear Models

| Model | Training Memory | Inference Memory | Notes |
|-------|-----------------|------------------|-------|
| LinearRegression | O(n*p + p^2) | O(p) | QR decomposition requires p^2 for normal equations |
| RidgeRegression | O(n*p + p^2) | O(p) | Similar to OLS with regularization |
| LassoRegression | O(n*p) | O(p) | Coordinate descent is memory efficient |
| LogisticRegression | O(n*p) | O(p*k) | k = number of classes |

**Practical Estimates:**
- 1K samples, 100 features: ~10 MB
- 10K samples, 100 features: ~100 MB
- 100K samples, 100 features: ~1 GB

#### Tree-Based Models

| Model | Training Memory | Inference Memory | Notes |
|-------|-----------------|------------------|-------|
| DecisionTree | O(n*p + tree_size) | O(tree_size) | tree_size depends on depth |
| RandomForest | O(n*p + n_trees*tree_size) | O(n_trees*tree_size) | Scales linearly with n_trees |
| GradientBoosting | O(n*p + n_trees*tree_size) | O(n_trees*tree_size) | Sequential training |
| HistGradientBoosting | O(n*bins + n_trees*leaves) | O(n_trees*leaves) | More memory efficient |

**Practical Estimates (100 trees, max_depth=10):**
- 1K samples: ~50 MB
- 10K samples: ~200 MB
- 100K samples: ~500 MB

#### Histogram-Based Gradient Boosting

HistGradientBoosting is optimized for memory efficiency:

```
Memory = O(n_samples * n_bins * sizeof(u8))  # Binned data
       + O(n_trees * n_leaves * sizeof(f64))  # Tree structure
       + O(n_samples * sizeof(f64))           # Gradients/Hessians
```

**Key advantage:** Uses 8-bit binned features instead of 64-bit floats, reducing memory by 8x for feature storage.

### Memory-Constrained Environments

For systems with limited memory:

1. **Use HistGradientBoosting** instead of RandomForest or GradientBoosting
2. **Reduce max_depth** to limit tree size
3. **Use subsample < 1.0** for stochastic training
4. **Process data in batches** for very large datasets
5. **Use memory-mapped files** via `datasets::mmap` module

### Dataset Size Guidelines

| RAM Available | Max Samples (100 features) | Recommended Models |
|---------------|---------------------------|-------------------|
| 2 GB | 50K | Linear, DecisionTree |
| 4 GB | 200K | Linear, HistGradientBoosting |
| 8 GB | 500K | All models (small ensembles) |
| 16 GB | 1M+ | All models |
| 32 GB+ | 5M+ | Full ensembles, large hyperparameter search |

---

## Time Complexity

### Training Complexity

| Model | Time Complexity | Notes |
|-------|-----------------|-------|
| LinearRegression | O(n * p^2) | QR decomposition |
| RidgeRegression | O(n * p^2) | Direct solution |
| LassoRegression | O(n * p * iterations) | Coordinate descent |
| DecisionTree | O(n * p * log(n)) | Recursive splitting |
| RandomForest | O(n_trees * n * p * log(n)) | Parallelizable |
| GradientBoosting | O(n_trees * n * p * log(n)) | Sequential |
| HistGradientBoosting | O(n_trees * n * bins) | Histogram-based |

### Prediction Complexity

| Model | Time Complexity | Notes |
|-------|-----------------|-------|
| LinearRegression | O(n * p) | Matrix-vector product |
| DecisionTree | O(n * depth) | Tree traversal |
| RandomForest | O(n * n_trees * depth) | Parallelizable |
| GradientBoosting | O(n * n_trees * depth) | Sequential traversal |

### Scaling Characteristics

**Linear scaling with samples (O(n)):**
- Preprocessing (StandardScaler, MinMaxScaler)
- HistGradientBoosting training
- All model predictions

**Superlinear scaling (O(n log n)):**
- DecisionTree training (sorting)
- RobustScaler (median computation)
- RandomForest training

**Quadratic scaling (O(n^2)):**
- KNN (without spatial index)
- Some kernel methods

---

## Memory Profiling

### Running Memory Benchmarks

```bash
# Run memory profiling benchmarks
cargo bench --bench memory_benchmarks -p ferroml-core

# Output includes:
# - Peak memory usage
# - Memory delta (allocation during operation)
# - Bytes per sample
# - Duration
```

### Understanding Memory Profile Output

```
Operation                                  Samples     Delta (MB)  Bytes/Sample  Time (ms)
------------------------------------------------------------------------------------------
[LINEAR_REGRESSION]
LinearRegression/fit                         1000          2.50        2621.4        5.2
LinearRegression/fit                        10000         25.00        2621.4       45.1
LinearRegression/fit                       100000        250.00        2621.4      450.0

[RANDOM_FOREST]
RandomForestClassifier/fit(n_est=10)         1000         10.20       10700.8       52.3
RandomForestClassifier/fit(n_est=100)        1000         98.50      103321.6      485.2
```

### Using dhat for Detailed Heap Analysis

For detailed heap profiling:

```bash
# Compile with dhat allocator
DHAT_LOG=dhat-heap.json cargo bench --bench memory_benchmarks

# Analyze with dhat-viewer
# Visit: https://nnethercote.github.io/dh_view/dh_view.html
# Load dhat-heap.json
```

### Memory Metrics Explained

- **Peak Memory**: Maximum memory used during operation
- **Delta Memory**: Memory difference before/after operation
- **Bytes per Sample**: Memory efficiency metric (delta / n_samples)
- **Resident Set Size (RSS)**: Total physical memory used by process

---

## Performance Regression Testing

### CI Integration

FerroML includes automatic performance regression detection in CI:

1. **On Pull Requests**: Compares PR benchmarks against base branch
2. **Threshold**: 10% regression triggers CI failure
3. **Artifacts**: Benchmark results saved for 30 days

### How It Works

```yaml
# The CI workflow:
# 1. Checks out base branch and runs benchmarks
# 2. Checks out PR branch and runs benchmarks
# 3. Compares results with 10% threshold
# 4. Fails if regression detected, comments on PR
```

### Local Performance Testing

```bash
# Run benchmarks and save baseline
cargo bench -p ferroml-core -- --save-baseline main

# Make changes...

# Compare against baseline
cargo bench -p ferroml-core -- --save-baseline new
critcmp main new
```

### Understanding Regression Reports

```
============================================================
PERFORMANCE COMPARISON REPORT
============================================================
Regression threshold: 10.0%

REGRESSIONS DETECTED:
----------------------------------------
  RandomForest/fit
    Base: 50000000ns -> PR: 60000000ns (+20.0%)

IMPROVEMENTS:
----------------------------------------
  LinearRegression/fit
    Base: 5000000ns -> PR: 4000000ns (-20.0%)

Summary: 1 regressions, 1 improvements, 15 unchanged
```

### Baseline Management

Performance baselines are stored in `benchmarks/baseline.json`:

```json
{
  "metadata": {
    "version": "0.1.0",
    "threshold_percent": 10
  },
  "timing_benchmarks": {
    "LinearRegression": {
      "fit": {
        "1000x50": { "mean_ns": 500000, "std_ns": 50000 }
      }
    }
  }
}
```

---

## Optimization Guidelines

### General Best Practices

1. **Choose the right model**: HistGradientBoosting for large datasets
2. **Tune hyperparameters**: Reduce n_estimators, max_depth when possible
3. **Use parallelism**: Enable rayon for multi-threaded training
4. **Batch predictions**: Process predictions in batches for memory efficiency

### Model-Specific Optimizations

#### RandomForest

```rust
// Memory-efficient configuration
let model = RandomForestClassifier::new()
    .with_n_estimators(50)       // Fewer trees = less memory
    .with_max_depth(Some(10))    // Limit tree depth
    .with_max_features(Some(10)) // Subset of features per split
    .with_bootstrap(true);       // Bootstrap sampling
```

#### HistGradientBoosting

```rust
// Fast, memory-efficient configuration
let model = HistGradientBoostingClassifier::new()
    .with_max_iter(100)
    .with_max_depth(Some(6))     // Shallow trees
    .with_max_bins(255)          // Default is optimal
    .with_early_stopping(true)   // Stop when validation plateaus
    .with_validation_fraction(0.1);
```

### Preprocessing Optimizations

```rust
// In-place transformations (when possible)
let mut scaler = StandardScaler::new();
scaler.fit(&x)?;
let x_scaled = scaler.transform(&x)?;  // Returns new array

// For very large data, consider chunked processing
for chunk in data.chunks(10000) {
    let scaled_chunk = scaler.transform(&chunk)?;
    process(scaled_chunk);
}
```

### Memory-Mapped Datasets

For datasets larger than RAM:

```rust
use ferroml_core::datasets::mmap::MappedDataset;

// Memory-map a large CSV file
let dataset = MappedDataset::from_csv("large_data.csv")?;

// Data is loaded on-demand, not all at once
for batch in dataset.batches(1000) {
    model.partial_fit(&batch)?;
}
```

---

## Benchmarking

### Running Full Benchmark Suite

```bash
# All benchmarks (may take 10+ minutes)
cargo bench -p ferroml-core

# Specific benchmark groups
cargo bench -p ferroml-core -- linear_models
cargo bench -p ferroml-core -- tree_models
cargo bench -p ferroml-core -- gradient_boosting
cargo bench -p ferroml-core -- preprocessing
cargo bench -p ferroml-core -- scaling

# Memory benchmarks
cargo bench --bench memory_benchmarks -p ferroml-core
```

### Benchmark Groups

| Group | Description | Typical Duration |
|-------|-------------|------------------|
| linear_models | OLS, Ridge, Lasso | ~30 seconds |
| tree_models | DecisionTree, RandomForest | ~2 minutes |
| gradient_boosting | GB, HistGB | ~3 minutes |
| preprocessing | Scalers, encoders | ~30 seconds |
| scaling | Performance vs dataset size | ~2 minutes |

### Interpreting Results

Criterion reports three timing values:

```
LinearRegression/fit    time:   [485.21 us 492.33 us 499.87 us]
                        ^^^^^^   ^^^^^^^   ^^^^^^^   ^^^^^^^
                        name     lower CI  estimate  upper CI
```

- **Lower CI**: 95% confidence lower bound
- **Estimate**: Point estimate (typically median)
- **Upper CI**: 95% confidence upper bound

### Comparing with Other Libraries

Reference comparisons (see `benchmarks/xgboost_lightgbm_timing.py`):

| Operation | FerroML | XGBoost | LightGBM | Notes |
|-----------|---------|---------|----------|-------|
| GB fit (1K samples) | 100ms | 20ms | 15ms | XGBoost/LightGBM have SIMD |
| GB fit (10K samples) | 1s | 100ms | 80ms | Gap increases with size |
| Prediction (10K) | 10ms | 8ms | 6ms | Similar performance |

FerroML focuses on statistical rigor over raw speed. For applications requiring maximum throughput, consider using FerroML for model selection/validation and exporting to ONNX for production inference.

---

## Profiling Tools

### CPU Profiling

```bash
# Using flamegraph (Linux/macOS)
cargo install flamegraph
cargo flamegraph --bench benchmarks -p ferroml-core

# Using perf (Linux)
perf record cargo bench -p ferroml-core
perf report
```

### Memory Profiling

```bash
# Using heaptrack (Linux)
heaptrack cargo bench --bench memory_benchmarks -p ferroml-core
heaptrack_gui heaptrack.*.gz

# Using valgrind/massif
valgrind --tool=massif cargo bench --bench memory_benchmarks -p ferroml-core
ms_print massif.out.*
```

### Quick Performance Checks

```rust
use std::time::Instant;

let start = Instant::now();
model.fit(&x, &y)?;
let fit_duration = start.elapsed();

let start = Instant::now();
let predictions = model.predict(&x)?;
let predict_duration = start.elapsed();

println!("Fit: {:?}, Predict: {:?}", fit_duration, predict_duration);
```

---

## Performance FAQ

### Q: Why is FerroML slower than XGBoost/LightGBM?

XGBoost and LightGBM are highly optimized C++ libraries with:
- Hand-tuned SIMD vectorization
- Cache-optimized memory layouts
- Years of performance tuning

FerroML prioritizes:
- Pure Rust (no unsafe, no C dependencies)
- Statistical correctness and diagnostics
- Readable, maintainable code

### Q: How can I speed up hyperparameter search?

1. Use HistGradientBoosting instead of RandomForest
2. Enable early stopping
3. Use MedianPruner or Hyperband scheduler
4. Start with coarse grid, then refine
5. Reduce cv folds for initial exploration

### Q: What's the recommended hardware for production?

| Dataset Size | Recommended RAM | Cores |
|--------------|-----------------|-------|
| < 100K rows | 8 GB | 4+ |
| 100K - 1M | 16-32 GB | 8+ |
| 1M - 10M | 64+ GB | 16+ |

### Q: How do I handle datasets larger than RAM?

1. Use memory-mapped files (`datasets::mmap`)
2. Process in batches with `partial_fit`
3. Use feature selection to reduce dimensionality
4. Sample data for model selection, train on full data for final model
