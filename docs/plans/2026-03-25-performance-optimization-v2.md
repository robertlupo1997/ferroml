# Performance Optimization v2 -- Design Document

**Date:** 2026-03-25
**Status:** Ready for implementation
**Builds on:** KMeans Phases A-C (sqrt-free, GEMM batch distances, Hamerly algorithm)

## Summary

| Work Area | Current Perf | Target | Estimated Effort | Priority |
|-----------|-------------|--------|-----------------|----------|
| Benchmark Refresh | KMeans only in cross-library | All algorithms at multiple sizes | Small | 1 (do first) |
| MiniBatchKMeans | N/A (not implemented) | Within 2.0x of sklearn | Medium | 2 |
| LogisticRegression | ~2.1x slower (estimated) | Within 1.5x of sklearn | Medium | 3 |
| HistGradientBoosting | 2.46x slower (PERF-09) | Within 2.0x of sklearn | Large | 4 |

---

## Section 1: Full Benchmark Refresh (Baseline)

### Current State

The cross-library benchmark (`scripts/benchmark_cross_library.py`) already supports 18 algorithms:
LinearRegression, Ridge, Lasso, LogisticRegression, DecisionTree (reg/clf),
RandomForest (reg/clf), GradientBoosting (reg/clf), HistGradientBoosting (reg/clf),
KNN, SVC, GaussianNB, StandardScaler, PCA, KMeans.

However, the **cross-library results JSON** (`docs/benchmark_cross_library_results.json`) currently only contains KMeans results. The perf-target benchmark (`docs/benchmark_results.json`) covers 10 algorithms but only at fixed sizes.

### What Needs to Change

1. **Run cross-library benchmark at multiple sizes:** 1000, 5000, 10000, 50000 samples
2. **Run at multiple feature counts:** 10, 20, 50 features
3. **Update both output files:**
   - `docs/benchmark_cross_library_results.json` -- full cross-library results
   - `docs/cross-library-benchmark.md` -- human-readable summary tables
4. **Add missing algorithms to cross-library:** LinearSVC, LDA, FactorAnalysis, TruncatedSVD, OLS (these are in perf targets but not in the cross-library script)

### Current Perf Target Results (from benchmark_results.json)

| Algorithm | Dataset | FerroML (ms) | sklearn (ms) | Ratio | Target | Pass? |
|-----------|---------|-------------|-------------|-------|--------|-------|
| PCA | 10000x100, k=10 | 10.9 | 4.8 | 2.29x | 2.0x | No |
| TruncatedSVD | 10000x100, k=10 | 39.9 | 465.8 | 0.09x | 2.0x | Yes (11x faster!) |
| LDA | 5000x50, 3 classes | 9.7 | 65.8 | 0.15x | 2.0x | Yes (6.8x faster!) |
| FactorAnalysis | 5000x50, k=5 | 65.0 | 188.0 | 0.35x | 3.0x | Yes (2.9x faster!) |
| LinearSVC | 5000x50, binary | 315.9 | 362.7 | 0.87x | 2.0x | Yes (1.1x faster!) |
| OLS | 10000x100 | 42.7 | 21.9 | 1.94x | 2.0x | Yes |
| Ridge | 10000x100, alpha=1.0 | 31.9 | 6.7 | 4.80x | 5.0x | Yes |
| KMeans | 5000x50, k=10 | 34.0 | 5.0 | 6.84x | 5.0x | No |
| HistGBT | 10000x20, 50 iters | 230.5 | 93.7 | 2.46x | 3.0x | Yes |
| SVC (RBF) | 2000x20, binary | 246.7 | 32.0 | 7.70x | 7.0x | No |

### Cross-Library KMeans Results (post-Hamerly)

| Size | Features | FerroML (ms) | sklearn (ms) | Ratio |
|------|----------|-------------|-------------|-------|
| 1000 | 20 | 1.2 | 13.4 | 10.98x faster |
| 5000 | 20 | 8.4 | 19.7 | 2.35x faster |
| 10000 | 20 | 8.4 | 24.0 | 2.84x faster |

### Success Criteria

- Cross-library results JSON covers all 18+ algorithms at 3+ sizes
- Markdown summary updated with ratio tables
- Establishes baseline for subsequent optimization work

### Estimated Effort: Small

Script changes are minor (just run with more args). Wall-clock time for full benchmark suite at 4 sizes x 3 feature counts may be 30-60 minutes.

---

## Section 2: MiniBatchKMeans Implementation

### Current State

MiniBatchKMeans does **not exist** in FerroML. The `ferroml-core/src/clustering/` directory contains: `agglomerative.rs`, `dbscan.rs`, `diagnostics.rs`, `gmm.rs`, `hdbscan.rs`, `kmeans.rs`, `metrics.rs`, `mod.rs`. No mini-batch variant.

### Design

MiniBatchKMeans is a **new model**, not an optimization of existing KMeans. It trades clustering quality for speed by processing random mini-batches instead of the full dataset each iteration.

#### API Surface (matching sklearn)

```rust
pub struct MiniBatchKMeans {
    pub n_clusters: usize,
    pub batch_size: usize,       // default: 1024
    pub max_iter: usize,          // default: 100
    pub n_init: usize,            // default: 3
    pub tol: f64,                 // default: 0.0 (no early stopping by default)
    pub reassignment_ratio: f64,  // default: 0.01
    pub random_state: Option<u64>,
    pub init: KMeansInit,         // reuse existing enum (KMeansPlusPlus, Random)
}
```

#### Key Implementation Details

1. **Initialization:** Reuse `KMeans::kmeans_plus_plus_init()` from existing `kmeans.rs` -- already implemented and tested.

2. **Mini-batch update rule:**
   - Each iteration: sample `batch_size` points uniformly at random
   - Assign each point to nearest center (use `batch_squared_distances()` -- already exists)
   - Update centers with learning rate decay: `center[j] = center[j] + (1/(count[j]+1)) * (x[i] - center[j])`
   - This is the Sculley (2010) mini-batch k-means algorithm

3. **Center reassignment:** If a center's accumulated count falls below `reassignment_ratio * batch_size`, reassign it to a randomly chosen point from the current batch. Prevents dead centers.

4. **Convergence detection:** Track ewa_inertia (exponentially weighted average of batch inertia). Converge when `|ewa_new - ewa_old| / ewa_old < tol`.

5. **GEMM batch distances:** The existing `batch_squared_distances()` function computes `||x_i - c_j||^2` via GEMM decomposition. This works directly for mini-batch assignment since batch sizes are typically 256-2048.

6. **Hamerly/Elkan bounds:** These are NOT applicable to mini-batch since centers change every iteration (bounds would be invalidated every batch). Use simple Lloyd-style assignment with GEMM distances.

7. **IncrementalModel trait:** Implement `partial_fit()` for streaming scenarios. Each `partial_fit()` call processes one batch.

#### File Changes Required

| File | Change |
|------|--------|
| `ferroml-core/src/clustering/kmeans.rs` | Add `MiniBatchKMeans` struct + impl (same file as KMeans, shares utilities) |
| `ferroml-core/src/clustering/mod.rs` | Re-export `MiniBatchKMeans` |
| `ferroml-python/src/clustering.rs` | Add `PyMiniBatchKMeans` wrapper |
| `ferroml-python/python/ferroml/clustering/__init__.py` | Re-export `MiniBatchKMeans` |

#### Performance Strategy

- Mini-batch update is O(batch_size * k * d) per iteration vs O(n * k * d) for full KMeans
- For n=50K, batch_size=1024, this is ~50x less work per iteration
- Total work: max_iter * batch_size * k * d = 100 * 1024 * k * d
- sklearn's MiniBatchKMeans uses Cython with similar algorithmic approach
- Target: within 2.0x of sklearn (our GEMM distances + Rust loops should be competitive)

### Success Criteria

- `MiniBatchKMeans` produces correct cluster assignments (verified against sklearn on synthetic data)
- Within 2.0x of sklearn MiniBatchKMeans on 50K+ samples
- Implements `ClusteringModel` trait (fit, predict, labels, cluster_centers)
- Implements `IncrementalModel` trait (partial_fit)
- Python bindings complete with proper `__init__.py` re-export
- Inertia matches sklearn within 5% tolerance (mini-batch is stochastic, exact match not expected)

### Risk Assessment

- **Testing:** New model requires new tests, bindings tests, and correctness verification
- **Serialization:** Need serialization support in Python bindings
- **Stochastic output:** Results vary by seed; tests must use fixed seeds and tolerance-based assertions

### Estimated Effort: Medium

Core algorithm is straightforward (simpler than Elkan/Hamerly). Most effort is in tests, bindings, and edge cases (empty clusters, reassignment, convergence). Approximately 400-600 lines of Rust + 100 lines Python bindings.

---

## Section 3: LogisticRegression Performance

### Current State

LogisticRegression is estimated at **~2.1x slower** than sklearn based on Phase 04 analysis. No cross-library benchmark results exist yet for LogisticRegression (only perf-target results cover OLS at 1.94x).

The perf-target benchmark does not include LogisticRegression directly, but the cross-library benchmark script does (bench_logistic_regression function exists).

### Root Cause Analysis

#### 1. Diagnostic Overhead (Primary Bottleneck)

Every `fit()` call computes `FittedLogisticData`, which includes:
- **Covariance matrix:** O(d^3) Cholesky decomposition of the information matrix
- **Standard errors:** Derived from covariance diagonal
- **Deviance statistics:** Log-likelihood, null deviance, residual deviance
- **AIC/BIC:** Information criteria
- **Pseudo R-squared:** McFadden's and variants

This runs **every fit**, even when the user only wants `predict()`. sklearn computes none of this by default.

This is FerroML's key differentiator (statistical diagnostics as first-class features), but it adds significant overhead for pure prediction workloads.

#### 2. Solver Selection Mismatch

Auto-solver logic (`resolve_solver()`):
- `n_samples >= 100,000` -> SAG
- `n_features < 50 AND n_samples <= 5,000` -> IRLS
- else -> L-BFGS

For a typical benchmark config (5000 samples, 20 features), FerroML selects **IRLS** (O(d^3) per iteration, exact Hessian). sklearn defaults to **L-BFGS** (quasi-Newton, O(d) per iteration with line search).

IRLS converges in fewer iterations but each iteration is more expensive. For d=20, IRLS should be competitive, but the Cholesky decomposition overhead adds up.

#### 3. Benchmark Configuration Asymmetry

- sklearn: `LogisticRegression(max_iter=1000)` with C=1.0 (L2 regularization)
- FerroML: `LogisticRegression()` with max_iter=100, l2_penalty=0.0 (no regularization)

Different regularization affects convergence speed. sklearn's L2 regularization (C=1.0 -> lambda=1.0) helps condition the problem, potentially reducing iteration count.

### Proposed Optimizations

#### Optimization 1: Lazy Diagnostics (High Impact)

**Speedup:** 1.3-2.0x | **Effort:** Medium | **Risk:** Medium

Change `FittedLogisticData` computation from eager (during `fit()`) to lazy (on first access of diagnostic methods).

**Implementation:**
- Store `x_design`, `y`, and fitted coefficients during `fit()`
- Add `LazyCell` or `OnceCell` wrapper around `FittedLogisticData`
- Compute diagnostics only when `summary()`, `diagnostics()`, `coefficient_info()`, etc. are called
- For fit-predict workloads, this eliminates the entire diagnostic overhead

**Risk:** Tests that call `model.fit().summary()` immediately will still work. Tests that access `fitted_data` directly need internal refactoring. The key risk is that storing `x_design` and `y` increases memory usage during the model's lifetime.

**Alternative:** Compute diagnostics during `fit()` but behind a `compute_diagnostics: bool` flag (default true for backward compatibility, false for benchmark mode). Less elegant but lower risk.

#### Optimization 2: Benchmark Fairness (Low Effort)

**Speedup:** N/A (correctness) | **Effort:** Small | **Risk:** None

Ensure benchmark compares like-for-like:
- Add LogisticRegression to perf-target benchmark with explicit solver and regularization settings
- Document that FerroML's diagnostics are a feature, not overhead (report both with/without diagnostics)

#### Optimization 3: L-BFGS Tuning (Incremental)

**Speedup:** 1.1-1.3x | **Effort:** Small | **Risk:** Low

Current L-BFGS uses argmin crate with `MoreThuenteLineSearch`. Tuning opportunities:
- **Memory size:** argmin default is 10; scipy L-BFGS-B uses 10 as well. Likely not a bottleneck.
- **Line search parameters:** Check c1/c2 Wolfe condition parameters match scipy
- **Gradient tolerance:** Ensure convergence criteria are comparable

#### Optimization 4: Auto-Solver Threshold Tuning

**Speedup:** Variable | **Effort:** Small | **Risk:** Low

The current IRLS threshold (`n_features < 50 AND n_samples <= 5,000`) may be too aggressive. Profile IRLS vs L-BFGS at the benchmark configuration (5000x20) to see which is actually faster, and adjust thresholds if needed.

### Success Criteria

- LogisticRegression within 1.5x of sklearn for default configurations (fit + predict)
- Diagnostic methods still work correctly (no regression in statistical output)
- Cross-library benchmark includes LogisticRegression results at multiple sizes

### Risk Assessment

- **Lazy diagnostics:** May break tests that assume diagnostics are available immediately after fit
- **Memory:** Storing x_design for lazy computation increases model size
- **API compatibility:** Must not change public API behavior

### Estimated Effort: Medium

Lazy diagnostics is the biggest change (~200 lines refactoring). L-BFGS tuning and solver threshold changes are small. Total: 1-2 sessions.

---

## Section 4: HistGradientBoosting Performance

### Current State

HistGradientBoosting is **2.46x slower** than sklearn (PERF-09: 230.5ms vs 93.7ms on 10000x20, 50 iterations).

Phase 04 analysis concluded:
- Bounds checks RETAINED due to NaN handling (BinMapper produces out-of-range bin indices for missing values)
- debug_assert added for gradient/hessian index validation (zero cost in release)
- The histogram subtraction trick is **already implemented** (`compute_by_subtraction()` method)

### Architecture Analysis

The current implementation includes:
1. **BinMapper:** Bins continuous features into discrete histograms (up to 256 bins)
2. **Two tree builders:** `grow_tree_col_major` (optimized, column-major layout) and `grow_tree_leafwise` (standard, row-major)
3. **Histogram subtraction:** Already uses the parent - sibling trick for the larger child
4. **Rayon parallelism:** Feature-parallel histogram building via `par_iter()` in `build_histograms_col_major`
5. **Column-major data layout:** `x_col_major: Vec<Vec<u8>>` for cache-friendly histogram accumulation

### Root Cause Analysis

Given that histogram subtraction is already implemented, the remaining performance gap comes from:

#### 1. Histogram Accumulation Inner Loop

The `build_histograms_col_major` function iterates over samples for each feature:
```rust
for &idx in sample_indices {
    let bin = x_col[idx] as usize;
    if bin < n_bins {  // bounds check for NaN handling
        hist.bins[bin].sum_gradients += gradients[idx];
        hist.bins[bin].sum_hessians += hessians[idx];
        hist.bins[bin].count += 1;
    }
}
```

sklearn uses Cython with explicit SIMD intrinsics (SSE2/AVX2) for this loop. The Rust compiler may auto-vectorize parts of this, but the indirect indexing (`bins[bin]`) prevents full SIMD utilization.

#### 2. Split Evaluation

The `find_best_split` function scans all bins for each feature to find the optimal split point. sklearn uses prefix sums and running sums to reduce redundant computation.

#### 3. Data Layout

sklearn uses a contiguous `uint8` array for binned features with explicit memory alignment. FerroML uses `Vec<Vec<u8>>` (column-major) which has pointer indirection per feature.

#### 4. Parallelism Granularity

Both FerroML and sklearn parallelize across features during histogram building. sklearn uses OpenMP with lower overhead than Rayon's work-stealing scheduler for small tasks.

### Proposed Optimizations

#### Optimization 1: Contiguous Column-Major Storage (Medium Impact)

**Speedup:** 1.1-1.2x | **Effort:** Medium | **Risk:** Low

Replace `Vec<Vec<u8>>` with a single contiguous `Vec<u8>` with manual stride-based indexing. This eliminates pointer indirection and improves cache behavior.

```rust
struct ColMajorBins {
    data: Vec<u8>,       // n_features * n_samples contiguous
    n_samples: usize,
    n_features: usize,
}
impl ColMajorBins {
    fn get(&self, feature: usize, sample: usize) -> u8 {
        self.data[feature * self.n_samples + sample]
    }
}
```

#### Optimization 2: Prefetch-Friendly Histogram Loop (Medium Impact)

**Speedup:** 1.1-1.3x | **Effort:** Medium | **Risk:** Low

Restructure the histogram accumulation to be more auto-vectorization-friendly:
1. Separate gradient/hessian accumulation into two passes (allows SIMD on each)
2. Use a flat `[f64; 256*3]` array instead of `Vec<HistBin>` for L1-cache-friendly layout
3. Add software prefetch hints for the binned data array

#### Optimization 3: Feature-Parallel Split Finding (Low Impact)

**Speedup:** 1.05-1.1x | **Effort:** Small | **Risk:** Low

Currently, split finding scans features sequentially within `find_best_split`. Parallelize across features using Rayon, since each feature's optimal split is independent.

Note: This has diminishing returns at typical feature counts (20 features, 8+ cores) since the per-feature work is small.

#### Optimization 4: Reduce Rayon Overhead for Small Tasks (Low-Medium Impact)

**Speedup:** 1.1-1.2x | **Effort:** Small | **Risk:** Low

Add a parallelism threshold: only use Rayon when `n_samples * n_features > THRESHOLD`. For small nodes (few samples), the sequential path is faster due to Rayon's work-stealing overhead.

This is especially important during later tree building stages when nodes contain few samples.

### What Is NOT Worth Pursuing

1. **Removing bounds checks:** Phase 04 confirmed these are needed for NaN handling. The branch predictor makes them effectively free for non-NaN data.
2. **Explicit SIMD intrinsics:** The maintenance burden outweighs the ~1.2x gain. Let the compiler auto-vectorize.
3. **Unsafe code for histogram accumulation:** Risk of correctness bugs outweighs marginal performance gain.

### Success Criteria

- HistGBT within 2.0x of sklearn (from current 2.46x) on PERF-09 benchmark
- No regression in accuracy or correctness
- NaN handling continues to work correctly
- All existing tests pass

### Risk Assessment

- **Contiguous storage:** Changes internal data representation. All code touching `x_col_major` needs updating.
- **Histogram layout:** Flat array approach changes the `Histogram` struct. Tests and serialization need updating.
- **Parallelism thresholds:** Wrong threshold values could hurt rather than help. Needs profiling.

### Estimated Effort: Large

Contiguous storage is a significant refactor (~300 lines). Histogram loop optimization is medium (~200 lines). Total: 2-3 sessions. This is the lowest ROI work area and should be done last.

---

## Section 5: Work Order and Dependencies

### Execution Order

```
Phase 1: Benchmark Refresh (baseline)
    |
    v
Phase 2: MiniBatchKMeans     Phase 3: LogisticRegression     Phase 4: HistGBT
(independent)                 (independent)                    (independent, largest)
    |                             |                                |
    v                             v                                v
Phase 5: Final Benchmark Refresh (after all optimizations)
```

### Dependency Matrix

| Work Area | Depends On | Blocks |
|-----------|-----------|--------|
| Benchmark Refresh | Nothing | All optimization work (provides baseline) |
| MiniBatchKMeans | Benchmark Refresh (baseline) | Final benchmark |
| LogisticRegression | Benchmark Refresh (baseline) | Final benchmark |
| HistGBT | Benchmark Refresh (baseline) | Final benchmark |
| Final Benchmark | All optimizations complete | Release |

### Estimated Timeline

| Phase | Estimated Sessions | Cumulative |
|-------|-------------------|-----------|
| 1. Benchmark Refresh | 1 | 1 |
| 2. MiniBatchKMeans | 2-3 | 3-4 |
| 3. LogisticRegression | 1-2 | 4-6 |
| 4. HistGBT | 2-3 | 6-9 |
| 5. Final Benchmark | 1 | 7-10 |

---

## Section 6: Risk Assessment

### Overall Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| LogReg lazy diagnostics breaks tests | Medium | Medium | Run full test suite before/after; add feature flag |
| HistGBT refactor introduces correctness bugs | Medium | High | Keep old path behind flag; diff test outputs |
| MiniBatchKMeans stochastic results hard to test | Low | Low | Fixed seeds, tolerance-based assertions |
| Benchmark at 50K samples takes too long | Low | Low | Cap at 10K if wall-clock exceeds 2 hours |
| KMeans perf-target still fails after optimizations | Medium | Low | KMeans cross-library already shows 2-10x faster; perf-target uses harder config (50 features, k=10) |

### Correctness Safeguards

For each optimization:
1. **Before:** Run full `cargo test` + `pytest` to establish baseline
2. **During:** Each change gets its own commit with passing tests
3. **After:** Run cross-library benchmark to verify no regression
4. **Validation:** Compare outputs against sklearn on identical inputs

### Performance Measurement Protocol

- All benchmarks use median of 5 timed runs with 1 warmup
- Same machine, same data seed (42), same data generation
- Report both fit time and predict time separately
- For stochastic algorithms (MiniBatchKMeans), use fixed random_state

---

## Appendix: Current Architecture Reference

### KMeans (for MiniBatchKMeans reuse)

Key functions available for reuse:
- `KMeans::kmeans_plus_plus_init()` -- initialization
- `batch_squared_distances()` -- GEMM-based distance computation
- `compute_row_norms()` -- row norm precomputation
- `KMeansAlgorithm` enum -- algorithm selection (Auto/Lloyd/Elkan/Hamerly)

File: `ferroml-core/src/clustering/kmeans.rs` (1915 lines)

### LogisticRegression Solver Selection

```
resolve_solver(n_features, n_samples):
  n_samples >= 100,000  -> SAG
  n_features < 50 AND n_samples <= 5,000  -> IRLS
  else -> L-BFGS
```

File: `ferroml-core/src/models/logistic.rs` (2795 lines)

### HistGBT Tree Building

Two tree builders:
1. `grow_tree_col_major()` -- column-major layout, used when `x_col_major` is available
2. `grow_tree_leafwise()` -- row-major fallback

Both use histogram subtraction (`compute_by_subtraction()`).

File: `ferroml-core/src/models/hist_boosting.rs` (4367 lines)
