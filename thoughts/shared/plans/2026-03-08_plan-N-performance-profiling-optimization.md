# Plan N: Performance Profiling & Optimization

## Overview

Profile FerroML's top algorithms, identify bottlenecks, and implement targeted optimizations to achieve 2x improvement on at least 5 key algorithms. Currently has 86+ Criterion benchmarks, rayon parallelism in forests/HistGB/KMeans, optional SIMD for distances, and optional GPU acceleration.

## Current State

- **86+ benchmarks** across 4 bench files (benchmarks.rs, memory_benchmarks.rs, performance_optimizations.rs, gpu_benchmarks.rs)
- **baseline.json** with ~50 expected ranges, 20% regression threshold
- **Rayon parallelism**: RandomForest, ExtraTrees, HistGB, KMeans, Bagging, CV folds, explainability
- **SIMD** (optional `--features simd`): squared_euclidean, euclidean, manhattan distances via `wide::f64x4`
- **GPU** (optional `--features gpu`): GEMM, MLP forward, KMeans distance via wgpu
- **No parallelism**: GradientBoosting (sequential tree building), SVM SMO, t-SNE, DecisionTree, MLP (CPU path)
- **6 ignored tests**: slow-runtime correctness tests (not bugs)
- **Known O(N^2)**: t-SNE, KNN brute-force, SVM kernel matrix, DBSCAN

## Desired End State

- Current baselines established and documented
- Top 10 optimization opportunities identified and ranked
- At least 5 algorithms with 2x+ improvement
- Barnes-Hut t-SNE for O(N log N) approximation
- SIMD enabled by default for distance calculations
- Benchmark CI integration for automatic regression detection
- Performance documentation updated with new results

---

## Phase N.1: Establish Baselines

**Overview**: Run all existing benchmarks, document current performance, identify slowest paths.

**Tasks**:

1. **Run full benchmark suite** on reference hardware:
   ```bash
   cargo bench --bench benchmarks -- --save-baseline current
   cargo bench --bench performance_optimizations -- --save-baseline current
   cargo bench --bench memory_benchmarks -- --save-baseline current
   ```

2. **File**: `docs/benchmark-results-baseline.md` (NEW)
   - Table: algorithm, dataset size, fit_time, predict_time, memory
   - Sorted by absolute time (slowest first)
   - Mark algorithms that are slower than expected

3. **Profile top 10 slowest operations** using `cargo bench` output:
   - Expected slowest: SVM (O(n^2)), t-SNE (O(N^2)), large RandomForest, GradientBoosting, KNN
   - Record median, min, max times

4. **Profile with flamegraph** (if available):
   ```bash
   cargo bench --bench benchmarks -- "random_forest" --profile-time=10
   ```
   - Identify hot functions in tree splitting, distance computation, gradient updates

**Success Criteria**:
- [ ] All 86+ benchmarks run successfully
- [ ] Baseline document generated with all timings
- [ ] Top 10 slowest operations identified and ranked

**Expected Tests**: 0 (profiling phase)

---

## Phase N.2: SIMD Default + Distance Optimization

**Overview**: Make SIMD the default for distance calculations, optimize batch distance computation.

**Changes Required**:

1. **File**: `ferroml-core/Cargo.toml` (EDIT)
   - Change default features: `default = ["parallel", "onnx", "simd"]`
   - This enables SIMD distances for KNN, KMeans, SVM, DBSCAN by default

2. **File**: `ferroml-core/src/simd.rs` (EDIT)
   - Add `batch_manhattan_distance()` — vectorized L1 distance matrix
   - Add `cosine_similarity_batch()` — vectorized cosine similarity
   - Optimize `batch_squared_euclidean()`:
     - Pre-compute squared norms: ||a||^2 + ||b||^2 - 2*a.b
     - Use BLAS-like dot product for inner loop
   - Add `minkowski_distance()` with SIMD for p=1,2 fast paths

3. **File**: `ferroml-core/src/models/neighbors.rs` (EDIT)
   - Replace manual distance loops with SIMD batch functions
   - Pre-compute training set norms at fit time (amortize across predictions)

4. **File**: `ferroml-core/src/models/clustering.rs` (EDIT — KMeans)
   - Replace per-point distance computation with batch SIMD call
   - Pre-compute centroid norms, use ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x.c trick

5. **File**: `ferroml-core/src/models/svm.rs` (EDIT)
   - Use SIMD dot products for linear kernel
   - Cache kernel diagonals (used in SMO working set selection)

**Benchmarks to beat**:
- KNN predict: 2x improvement target (distance computation dominant)
- KMeans fit: 1.5x improvement target (assignment step dominant)
- DBSCAN fit: 2x improvement target (pairwise distances)

**Success Criteria**:
- [ ] `cargo test -p ferroml-core` — all pass (no regressions)
- [ ] `cargo bench --bench benchmarks -- "knn"` — 2x faster than baseline
- [ ] `cargo bench --bench benchmarks -- "kmeans"` — 1.5x faster
- [ ] `cargo bench --bench benchmarks -- "dbscan"` — 2x faster

**Expected Tests**: ~10 new unit tests for SIMD functions

---

## Phase N.3: Tree Splitting Optimization

**Overview**: Optimize decision tree split finding — the inner loop of all tree-based models.

**Changes Required**:

1. **File**: `ferroml-core/src/models/tree.rs` (EDIT)
   - **Pre-sort features**: Sort feature values once at the start, reuse sorted indices
   - **Histogram-based splitting**: For continuous features with many unique values, bin into histograms (similar to HistGB)
   - **Cache split statistics**: Maintain running sums as we scan sorted values, avoid recomputing from scratch
   - **Subtraction trick**: For binary splits, compute right child stats as total - left (one less sum)

2. **File**: `ferroml-core/src/models/forest.rs` (EDIT)
   - **Column subsampling**: Only sort/evaluate `max_features` columns per split, not all p
   - **Shared sorted indices**: Pre-sort once, share across trees (read-only)

3. **File**: `ferroml-core/src/models/extra_trees.rs` (EDIT)
   - ExtraTrees already uses random splits — verify it's not sorting unnecessarily

**Benchmarks to beat**:
- DecisionTree fit (1000x50): 2x improvement target
- RandomForest fit (1000x20, 100 trees): 1.5x improvement target

**Success Criteria**:
- [ ] `cargo test -p ferroml-core` — all pass
- [ ] `cargo bench --bench benchmarks -- "decision_tree"` — 2x faster
- [ ] `cargo bench --bench benchmarks -- "random_forest.*fit"` — 1.5x faster
- [ ] Tree predictions unchanged (correctness preserved)

**Expected Tests**: ~5 regression tests verifying split quality unchanged

---

## Phase N.4: Gradient Boosting Sequential Optimization

**Overview**: GradientBoosting builds trees sequentially (can't parallelize across rounds), but we can optimize within each round.

**Changes Required**:

1. **File**: `ferroml-core/src/models/boosting.rs` (EDIT)
   - **Pre-allocate residual/gradient arrays**: Reuse across iterations instead of reallocating
   - **Avoid cloning init_predictions**: Use `to_owned()` only when necessary
   - **Inline loss computation**: Reduce function call overhead for hot path
   - **Cache leaf predictions**: Store as contiguous array, not tree traversal per sample

2. **File**: `ferroml-core/src/models/hist_boosting.rs` (EDIT)
   - **Histogram subtraction trick**: Compute child histogram = parent - sibling (halves histogram work)
   - **Reduce Vec::new() allocations**: Pre-allocate histogram vectors with correct capacity
   - **Parallel gradient computation**: Compute gradients/hessians across samples in parallel (rayon)
   - **Memory layout**: Ensure histograms are cache-line aligned (64-byte boundary)

3. **File**: `ferroml-core/src/models/boosting.rs` (EDIT — predict path)
   - **Batch prediction**: Traverse all trees for one sample at a time (cache-friendly) vs one tree at a time
   - **SIMD leaf accumulation**: Sum leaf values across trees using SIMD

**Benchmarks to beat**:
- GradientBoosting fit (1000x20, 100 trees): 1.5x improvement
- HistGB fit (10Kx20, 100 iter): 2x improvement
- GradientBoosting predict: 2x improvement (batch traversal)

**Success Criteria**:
- [ ] `cargo test -p ferroml-core` — all pass
- [ ] `cargo bench --bench benchmarks -- "gradient_boosting"` — 1.5x faster
- [ ] `cargo bench --bench benchmarks -- "hist_gradient_boosting"` — 2x faster
- [ ] Correctness tests unchanged (predictions match within tolerance)

**Expected Tests**: ~5 regression tests for prediction consistency

---

## Phase N.5: Barnes-Hut t-SNE

**Overview**: Implement O(N log N) approximate t-SNE using Barnes-Hut tree (VP-tree + quad/octree).

**Changes Required**:

1. **File**: `ferroml-core/src/decomposition/tsne.rs` (EDIT)
   - Add `method` parameter: `"exact"` (current) or `"barnes_hut"` (new, default for n > 1000)
   - Add `theta` parameter: Barnes-Hut approximation accuracy (default 0.5)

2. **File**: `ferroml-core/src/decomposition/vptree.rs` (NEW)
   - Vantage-point tree for nearest neighbor search
   - Used for computing sparse P matrix (perplexity-based affinities)

3. **File**: `ferroml-core/src/decomposition/quadtree.rs` (NEW)
   - Quad-tree (2D output) for Barnes-Hut force approximation
   - Octree extension for 3D output
   - `summarize()` method for cell mass and center computation
   - `compute_non_edge_forces()` with theta threshold

4. **File**: `ferroml-core/src/decomposition/tsne.rs` (EDIT — continued)
   - **Sparse P matrix**: Only store k nearest neighbors (k = 3 * perplexity)
   - **Barnes-Hut gradient**: Replace O(N^2) gradient loop with tree traversal O(N log N)
   - **Early exaggeration**: Multiply P by 12 for first 250 iterations (standard)

5. **File**: `ferroml-python/src/decomposition.rs` (EDIT)
   - Expose `method` and `theta` parameters in PyTSNE

6. **File**: `ferroml-python/python/ferroml/decomposition/__init__.py` (EDIT)
   - Update TSNE docstring with new parameters

**Benchmarks**:
- Exact t-SNE (1000 points): current baseline
- Barnes-Hut t-SNE (1000 points): comparable quality, ~2x faster
- Barnes-Hut t-SNE (5000 points): 10x faster than exact
- Barnes-Hut t-SNE (10000 points): feasible (exact is not)

**Success Criteria**:
- [ ] `cargo test -p ferroml-core -- tsne` — all pass (exact tests unchanged)
- [ ] New tests: Barnes-Hut produces kNN preservation > 0.9 on iris
- [ ] Barnes-Hut 5000 points completes in < 30s
- [ ] `pytest ferroml-python/tests/test_tsne.py -v` — all pass

**Expected Tests**: ~15 new tests (VP-tree, quad-tree, Barnes-Hut accuracy)

---

## Phase N.6: Memory Allocation Reduction

**Overview**: Reduce unnecessary allocations in hot paths across all algorithms.

**Changes Required**:

1. **Audit and fix `clone()` calls in hot paths**:
   - `ferroml-core/src/models/hist_boosting.rs`: Remove unnecessary `classes.clone()`, `init_predictions.clone()`
   - `ferroml-core/src/models/boosting.rs`: Avoid cloning feature matrices during tree building
   - `ferroml-core/src/models/forest.rs`: Share feature data across trees (Arc<Array2> or slice views)

2. **Pre-allocate working buffers**:
   - Tree splitting: allocate sort buffers once, reuse across splits
   - KMeans: allocate distance buffer once, reuse across iterations
   - MLP: allocate activation buffers in `initialize()`, reuse in forward/backward

3. **Use `ndarray` views instead of owned copies**:
   - Replace `x.slice(s![..]).to_owned()` with `x.slice(s![..])` where possible
   - Use `ArrayView2` instead of `Array2` for read-only parameters

4. **Arena allocator for tree nodes** (advanced):
   - Currently: each node is a Box allocation
   - Improvement: pre-allocate Vec<TreeNode> and use indices instead of pointers
   - Reduces cache misses during tree traversal

**Benchmarks to beat**:
- Memory profiling: 20% reduction in peak memory for RandomForest, GradientBoosting
- Fit time: 1.2x improvement from reduced allocation pressure

**Success Criteria**:
- [ ] `cargo test -p ferroml-core` — all pass
- [ ] `cargo bench --bench memory_benchmarks` — 20% memory reduction on at least 3 models
- [ ] No correctness regressions

**Expected Tests**: ~5 tests verifying memory-efficient paths produce same results

---

## Phase N.7: Parallel Gradient Boosting Prediction + SVM Optimization

**Overview**: Parallelize prediction paths and optimize SVM.

**Changes Required**:

1. **File**: `ferroml-core/src/models/boosting.rs` (EDIT)
   - Parallel prediction: split samples across threads, each thread traverses all trees
   - Use rayon `par_chunks()` on input rows

2. **File**: `ferroml-core/src/models/svm.rs` (EDIT)
   - **Working set selection**: Use second-order (WSS3) instead of first-order for faster convergence
   - **Kernel caching**: LRU cache for kernel evaluations (avoid recomputing k(x_i, x_j))
   - **Shrinking**: Remove bounded support vectors from optimization (standard SMO technique)
   - **Parallel kernel computation**: Use rayon for batch kernel matrix rows

3. **File**: `ferroml-core/src/models/neural/mlp.rs` (EDIT)
   - **Batch matrix multiply**: Use ndarray's built-in BLAS for matmul (ensure linked to openblas/mkl)
   - **Parallel batch processing**: Split mini-batches across threads for forward pass

**Benchmarks to beat**:
- GradientBoosting predict: 2x improvement (parallel traversal)
- SVM fit (1000 points): 2x improvement (WSS3 + caching)
- MLP fit (1000 points): 1.5x improvement (BLAS optimization)

**Success Criteria**:
- [ ] `cargo test -p ferroml-core` — all pass
- [ ] `cargo bench --bench benchmarks -- "gradient_boosting.*predict"` — 2x faster
- [ ] `cargo bench --bench benchmarks -- "svc"` — 2x faster
- [ ] `cargo bench --bench benchmarks -- "mlp"` — 1.5x faster

**Expected Tests**: ~5 tests verifying parallel prediction matches sequential

---

## Phase N.8: Benchmark CI Integration

**Overview**: Integrate benchmark regression detection into CI pipeline.

**Changes Required**:

1. **File**: `.github/workflows/benchmarks.yml` (NEW)
   ```yaml
   name: Benchmark Regression Detection
   on:
     pull_request:
       branches: [master]
   jobs:
     benchmark:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Install Rust
           uses: dtolnay/rust-toolchain@stable
         - name: Run benchmarks
           run: cargo bench --bench benchmarks -- --save-baseline pr
         - name: Compare against baseline
           run: cargo bench --bench benchmarks -- --baseline current --load-baseline pr
         - name: Check for regressions > 20%
           run: python scripts/check_bench_regressions.py
   ```

2. **File**: `scripts/check_bench_regressions.py` (NEW)
   - Parse Criterion JSON output from `target/criterion/`
   - Flag any benchmark that regressed > 20% from baseline
   - Output summary table: benchmark, old, new, change%
   - Exit code 1 if any regression detected

3. **File**: `ferroml-core/benches/baseline.json` (UPDATE)
   - Update with new baselines after optimization phases complete
   - Tighten regression threshold from 20% to 15% for optimized algorithms

4. **File**: `docs/performance.md` (UPDATE)
   - Add section on optimization results
   - Update time complexity table with new parallelism status
   - Add "how to run benchmarks" developer guide

**Success Criteria**:
- [ ] CI workflow triggers on PRs and detects regressions
- [ ] `scripts/check_bench_regressions.py` correctly identifies > 20% regressions
- [ ] Updated performance documentation

**Expected Tests**: ~3 tests for the regression checking script

---

## Phase N.9: Final Profiling & Documentation

**Overview**: Re-run all benchmarks, document improvements, update baselines.

**Tasks**:

1. **Re-run full benchmark suite**:
   ```bash
   cargo bench --bench benchmarks -- --save-baseline optimized
   cargo bench --bench performance_optimizations -- --save-baseline optimized
   cargo bench --bench memory_benchmarks -- --save-baseline optimized
   ```

2. **Compare against Phase N.1 baselines**:
   ```bash
   cargo bench --bench benchmarks -- --baseline current --load-baseline optimized
   ```

3. **File**: `docs/benchmark-results-optimized.md` (NEW)
   - Side-by-side: before vs after for all 86+ benchmarks
   - Highlight algorithms with 2x+ improvement
   - Memory usage comparison

4. **File**: `docs/performance.md` (UPDATE)
   - Add optimization history section
   - Update recommended feature flags
   - Add Barnes-Hut t-SNE documentation

5. **File**: `ferroml-core/benches/baseline.json` (UPDATE)
   - New baselines reflecting optimized performance
   - Tighter thresholds for critical paths

6. **Verify 2x target achieved on 5+ algorithms**:
   - Target: KNN, DBSCAN, DecisionTree, GradientBoosting predict, SVM, t-SNE (Barnes-Hut)

**Success Criteria**:
- [ ] 2x improvement on at least 5 algorithms (documented with before/after)
- [ ] All tests pass: `cargo test -p ferroml-core`
- [ ] All Python tests pass: `pytest ferroml-python/tests/ -q`
- [ ] Updated baselines committed
- [ ] Performance documentation complete

**Expected Tests**: 0 (documentation phase)

---

## Summary

| Phase | Focus | New Tests | Target Improvement |
|---|---|---|---|
| N.1 | Establish Baselines | 0 | Document current state |
| N.2 | SIMD Default + Distance Optimization | ~10 | KNN 2x, KMeans 1.5x, DBSCAN 2x |
| N.3 | Tree Splitting Optimization | ~5 | DecisionTree 2x, RandomForest 1.5x |
| N.4 | Gradient Boosting Optimization | ~5 | GB 1.5x, HistGB 2x, GB predict 2x |
| N.5 | Barnes-Hut t-SNE | ~15 | t-SNE 10x on large datasets |
| N.6 | Memory Allocation Reduction | ~5 | 20% memory reduction on 3+ models |
| N.7 | Parallel Prediction + SVM | ~5 | GB predict 2x, SVM 2x, MLP 1.5x |
| N.8 | Benchmark CI Integration | ~3 | Automated regression detection |
| N.9 | Final Profiling & Docs | 0 | Verify 2x on 5+ algorithms |
| **Total** | | **~48** | **2x on 5+ algorithms** |

## Algorithms Targeted for 2x Improvement

1. **KNN** (SIMD distances) — Phase N.2
2. **DBSCAN** (SIMD distances) — Phase N.2
3. **DecisionTree** (pre-sorted splits) — Phase N.3
4. **HistGradientBoosting** (histogram subtraction) — Phase N.4
5. **t-SNE** (Barnes-Hut) — Phase N.5
6. **SVM** (WSS3 + kernel cache) — Phase N.7
7. **GradientBoosting predict** (parallel traversal) — Phase N.7

## Verification Commands

```bash
# Run all benchmarks
cargo bench --bench benchmarks 2>&1 | tee bench_results.txt
cargo bench --bench memory_benchmarks 2>&1 | tee memory_results.txt
cargo bench --bench performance_optimizations 2>&1 | tee scaling_results.txt

# Compare against baseline
cargo bench --bench benchmarks -- --baseline current

# Run all tests (verify no regressions)
cargo test -p ferroml-core
pytest ferroml-python/tests/ -q

# Profile specific algorithm
cargo bench --bench benchmarks -- "kmeans" --verbose
```

## Dependencies

- No new crate dependencies for N.1-N.4, N.6-N.9
- Phase N.5 (Barnes-Hut): no new deps (pure Rust implementation)
- `wide` crate already in optional deps (becomes default)
- `criterion` 0.5 already configured
