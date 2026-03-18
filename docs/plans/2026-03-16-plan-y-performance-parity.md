# Plan Y: Performance Parity

**Goal**: Close the performance gap with sklearn for the three slowest models.
**Target**: SVC 17.6x→2-3x, HistGBT 2.6x→1.3-1.8x, LogReg 2.1x→1.0-1.3x.

## Phase 1: SVC Kernel Cache Rewrite (Critical)

**Problem**: SVC is 17.6x slower than sklearn due to a HashMap-based LRU kernel cache with O(n) eviction scans.

**Root Causes** (all in `ferroml-core/src/models/svm.rs`):
1. **O(n) contains() in eviction loop** (line 153): `self.order.contains(&evicted)` scans the entire VecDeque on every eviction — quadratic behavior in hot path
2. **HashMap overhead**: Hashing on every kernel access adds constant overhead vs libsvm's flat array
3. **Fragmented memory**: Each cached row is a separate `Vec<f64>` heap allocation (line 108: `training_data: Vec<Vec<f64>>`), destroying CPU cache locality
4. **Row-count cache sizing**: `DEFAULT_CACHE_SIZE = 1000` rows (line 84) vs libsvm's 200MB byte budget — underutilizes available memory
5. **First-order j-selection**: Platt's heuristic (max |E_i - E_j|) replaced WSS3 — converges in more iterations, meaning more kernel evaluations

**Changes**:

### 1a. Contiguous Slab Cache (Expected: 3-5x speedup)
Replace `HashMap<usize, Vec<f64>>` + `VecDeque<usize>` with:
- Single contiguous `Vec<f64>` allocation (slab), sized in bytes (default 200MB)
- Intrusive doubly-linked list for O(1) LRU eviction (no contains() scan)
- Direct index mapping: `row_offset[i]` → offset into slab (or -1 if not cached)
- Eliminate per-row heap allocation entirely

```rust
struct SlabKernelCache {
    /// Contiguous memory for all cached rows
    slab: Vec<f64>,
    /// Offset into slab for each sample (-1 if not cached)
    row_offset: Vec<i64>,
    /// Doubly-linked list nodes for LRU ordering
    lru_prev: Vec<i32>,
    lru_next: Vec<i32>,
    lru_head: i32,  // least recently used
    lru_tail: i32,  // most recently used
    /// Row length (n_samples)
    row_len: usize,
    /// Max rows that fit in slab
    capacity: usize,
}
```

Reference: Study `libsvm-rs` (github.com/ricardofrantz/libsvm-rs) — pure Rust port of libsvm v337 with numerical parity. Their cache.rs implements this pattern.

### 1b. Restore WSS3 Second-Order j-Selection (Expected: 2-3x speedup)
After selecting i by maximum KKT violation (first-order), select j to maximize objective decrease:

```
j = argmax_t  (grad_i - grad_t)^2 / (Q_ii + Q_tt - 2*Q_it)
```

This requires precomputing Q_ii (diagonal of kernel matrix) — trivial O(n) cost. WSS3 converges in far fewer iterations than Platt's heuristic. Reference: Fan, Chen, Lin (JMLR 2005).

Note: We previously had WSS3 but replaced it with Platt's heuristic during the RBF convergence fix. The convergence bug was in j-selection for ill-conditioned kernels — WSS3 actually handles this better than Platt when implemented correctly (the denominator regularization prevents division by near-zero).

### 1c. Active-Set-Only Gradient Updates (Expected: 1.5-2x speedup)
Currently the error cache update iterates over all n samples after each SMO step. With shrinking, only iterate over the active set (variables not at bounds). libsvm swaps indices so active variables are contiguous at the front of the array.

### 1d. Byte-Based Cache Sizing (Expected: 1.2-1.5x speedup)
Replace `DEFAULT_CACHE_SIZE: usize = 1000` (row count) with byte-based sizing:
```rust
const DEFAULT_CACHE_SIZE_MB: usize = 200;  // Match sklearn/libsvm default
```
Compute capacity as `cache_bytes / (n_samples * 8)` rows. For n=5000: 200MB / 40KB = 5,000 rows (full matrix cached!).

### Phase 1 Validation
- Benchmark: `cargo bench --bench benchmarks -- svc` before/after
- Cross-library: `pytest ferroml-python/tests/test_comparison_remaining.py -k svc`
- Regression: All 37+ SVC Rust tests + 100+ Python SVC tests must pass
- Target: < 3x of sklearn on n=5000, 2-feature RBF classification

---

## Phase 2: HistGBT Inner Loop Optimization (Medium)

**Problem**: HistGBT is 2.6x slower than sklearn. The histogram building inner loop is the bottleneck.

**Root Causes** (all in `ferroml-core/src/models/hist_boosting.rs`):
1. **SoA histogram layout** (line 392-398): Three separate `Vec`s (`sum_gradients`, `sum_hessians`, `counts`) cause 3 cache misses per random bin access
2. **Bounds check in inner loop** (line 1287/1314): `if bin < n_bins` is redundant when BinMapper guarantees valid bins
3. **No loop unrolling**: Processing one sample at a time prevents CPU pipelining
4. **Low Rayon threshold** (line 1276): 1,000 samples is at Rayon's break-even point

**Changes**:

### 2a. Array-of-Structs Histogram Layout (Expected: 1.3-1.5x speedup)
Replace:
```rust
struct Histogram {
    sum_gradients: Vec<f64>,
    sum_hessians: Vec<f64>,
    counts: Vec<usize>,
}
```
With:
```rust
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct HistogramBin {
    sum_gradients: f64,
    sum_hessians: f64,
    count: u32,  // u32 saves 4 bytes, fits in same cache line
    _pad: u32,   // align to 24 bytes
}

struct Histogram {
    bins: Vec<HistogramBin>,  // all fields for bin[k] in one cache line
}
```
This packs all 3 fields for a single bin into 24 bytes (fits in one cache line). Random access by bin index now hits one cache line instead of three.

### 2b. Remove Bounds Check (Expected: 1.2-1.4x speedup)
Add a `debug_assert!(bin < n_bins)` and use unsafe indexing in release builds, or pre-validate that BinMapper output is always in range (add a contract/invariant check in BinMapper).

```rust
// Before: bounds check every iteration
if bin < n_bins {
    hist.sum_gradients[bin] += gradients[idx];
    ...
}

// After: pre-validated, use get_unchecked in release
unsafe {
    let b = hist.bins.get_unchecked_mut(bin);
    b.sum_gradients += *gradients.get_unchecked(idx);
    b.sum_hessians += *hessians.get_unchecked(idx);
    b.count += 1;
}
```

### 2c. Loop Unrolling — 4 Samples Per Iteration (Expected: 1.3-1.5x speedup)
Process 4 samples per iteration to allow CPU to pipeline memory fetches:
```rust
let chunks = indices.chunks_exact(4);
let remainder = chunks.remainder();

for chunk in chunks {
    let b0 = col[chunk[0]] as usize;
    let b1 = col[chunk[1]] as usize;
    let b2 = col[chunk[2]] as usize;
    let b3 = col[chunk[3]] as usize;
    // Accumulate all 4 before committing — CPU prefetches ahead
    unsafe {
        hist.bins.get_unchecked_mut(b0).sum_gradients += gradients[chunk[0]];
        hist.bins.get_unchecked_mut(b1).sum_gradients += gradients[chunk[1]];
        // ... etc
    }
}
// Handle remainder
for &idx in remainder { ... }
```

### 2d. Raise Rayon Threshold (Trivial)
Change line 1276 from `indices.len() > 1_000` to `indices.len() > 5_000`. Rayon overhead dominates for small workloads.

### Phase 2 Validation
- Benchmark: `cargo bench --bench benchmarks -- hist` before/after
- Cross-library: `pytest ferroml-python/tests/test_comparison_trees.py -k hist`
- Regression: All HistGBT Rust + Python tests must pass
- Target: < 1.8x of sklearn on n=10000, 10-feature classification

---

## Phase 3: SAG/SAGA Solver for LogisticRegression (Nice-to-Have)

**Problem**: LogReg is 2.1x slower than sklearn. IRLS uses full Hessian inversion O(d^3), L-BFGS is slower than sklearn's SAG for large n.

**Root Causes** (in `ferroml-core/src/models/logistic.rs`):
1. **IRLS for d<50**: Full Hessian via Cholesky is O(d^3) per iteration — exact but expensive
2. **No SAG/SAGA**: sklearn uses SAG for large-n regime (O(d) per iteration with linear convergence)
3. **No warm starting across CV paths**: LogisticRegressionCV refits from scratch for each C value

**Changes**:

### 3a. Implement SAG Solver (Expected: 2-4x for n>10K)
SAG (Stochastic Average Gradient, Schmidt/Le Roux/Bach 2013) maintains a table of per-sample gradients:

```rust
pub enum LogisticSolver {
    Irls,
    Lbfgs,
    Sag,    // NEW: L2-only, O(d) per iteration
    Saga,   // NEW: L1/L2/elastic-net via proximal operator
    Auto,
}
```

Algorithm outline:
1. Initialize gradient table `G[i]` for each sample (n × d memory)
2. Each iteration: pick random sample j, compute new gradient g_j
3. Update: `w -= lr * (g_j - G[j] + mean(G))` (SAG update rule)
4. Store: `G[j] = g_j`
5. Convergence: linear rate like full gradient, but O(d) per iteration

Auto solver selection:
- d < 50, n < 10K: IRLS (exact, fast for small problems)
- d >= 50, n < 10K: L-BFGS
- n >= 10K: SAG (L2) or SAGA (L1/elastic-net)

### 3b. SAGA Extension (Expected: same + L1 support)
SAGA (Defazio et al., NIPS 2014) adds an unbiased correction term and supports proximal operators for L1/elastic-net penalties. Implementation is a small modification of SAG.

### 3c. Warm Starting for CV Paths (Expected: 2-3x for LogisticRegressionCV)
When fitting across a regularization grid (e.g., C = [0.01, 0.1, 1, 10, 100]):
- Sort C values
- Use converged weights from C[k] as initialization for C[k+1]
- Reduces total iterations dramatically

### Phase 3 Validation
- Benchmark: `cargo bench --bench benchmarks -- logistic` before/after
- Cross-library: `pytest ferroml-python/tests/test_comparison_remaining.py -k logistic`
- Regression: All LogisticRegression Rust + Python tests must pass
- Convergence test: SAG must converge to same coefficients as IRLS (within 1e-6)
- Target: < 1.3x of sklearn on n=50000, 20-feature binary classification

---

## Execution Order

| Phase | Model | Current | Target | Effort | Priority |
|-------|-------|---------|--------|--------|----------|
| 1 | SVC | 17.6x | 2-3x | 4 days | Critical |
| 2 | HistGBT | 2.6x | 1.3-1.8x | 2 days | Medium |
| 3 | LogReg | 2.1x | 1.0-1.3x | 4 days | Nice-to-have |

**Total: ~10 days across 3 phases.**

Each phase is independent and can be committed separately. Phase 1 is the clear priority — SVC performance is the biggest gap in the library.

## References

- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf) — Chang & Lin
- [Working Set Selection Using Second Order Information](https://www.jmlr.org/papers/volume6/fan05a/fan05a.pdf) — Fan, Chen, Lin (JMLR 2005)
- [libsvm-rs: Pure Rust LIBSVM port](https://github.com/ricardofrantz/libsvm-rs) — reference implementation
- [Minimizing Finite Sums with SAG](https://arxiv.org/abs/1309.2388) — Schmidt, Le Roux, Bach
- [SAGA: Fast Incremental Gradient Method](https://www.di.ens.fr/~fbach/Defazio_NIPS2014.pdf) — Defazio et al. (NIPS 2014)
- [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf) — Fan et al.
- [LightGBM: A Highly Efficient GBDT](https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Ke et al. (NeurIPS 2017)
- [The State of SIMD in Rust in 2025](https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d) — Shnatsel
