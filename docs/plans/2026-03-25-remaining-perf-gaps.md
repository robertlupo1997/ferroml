# Remaining Performance Gaps — Execution Plan

**Date:** 2026-03-25
**Status:** Ready for implementation
**Prereqs:** Commits 36c87b9 (MiniBatchKMeans + lazy diagnostics + HistGBT contiguous) and 5bf432c (LogReg IRLS opt)

## Current State (post-optimization)

| Algorithm | FerroML (ms) | sklearn (ms) | Ratio | Target |
|-----------|-------------|-------------|-------|--------|
| LogisticRegression | 12.1 | 5.2 | 2.3x slower | 1.5x |
| HistGBT Regressor | 415.6 | 146.6 | 2.8x slower | 2.0x |
| HistGBT Classifier | 399.9 | 142.7 | 2.8x slower | 2.0x |
| SVC (RBF) | 1015.4 | 99.8 | 10.2x slower | 5.0x |

Benchmark config: n=10000, features=20, seed=42.

---

## Gap 1: LogisticRegression (2.3x → 1.5x target)

### What was already done
- `compute_diagnostics=False` skips Fisher info / covariance (1.18x speedup)
- Auto-solver now picks IRLS for n<=10K, d<50 (IRLS beats L-BFGS here)
- IRLS inner loop buffers pre-allocated (eta, mu, wz, scaled_x)

### Remaining bottleneck
- IRLS: O(n*p^2 + p^3) per iteration for X'WX GEMM + Cholesky solve
- sklearn uses liblinear (C, highly optimized coordinate descent) or scipy L-BFGS-B (Fortran)
- Our IRLS converges in ~6 iterations vs sklearn's L-BFGS ~20, but each iteration is heavier

### Proposed optimizations (diminishing returns)

1. **Warm-start from previous iteration** (small win): Use previous beta as starting point for Cholesky refinement
2. **Avoid full X'WX recomputation**: Since W changes slowly between iterations, use rank-1 update of Cholesky factor (Cholmod-style). Complex to implement.
3. **BLAS-3 for scaled_x computation**: Replace row-by-row scaling with diagonal matrix multiply (might auto-vectorize better)

### Effort estimate: Medium. ~1.2-1.5x additional improvement realistic. Getting to 1.5x may require replacing IRLS with a custom coordinate descent solver.

---

## Gap 2: HistGradientBoosting (2.8x → 2.0x target)

### What was already done
- Contiguous column-major storage (Vec<Vec<u8>> → ColMajorBins)
- 4-sample unrolled histogram accumulation
- Rayon parallelism threshold at 5K samples
- Histogram subtraction trick (parent - sibling)

### Remaining bottleneck
- sklearn uses Cython with explicit SIMD (SSE2/AVX2) for histogram accumulation
- The inner loop has indirect indexing (bins[bin_value]) which prevents auto-vectorization
- Rayon overhead for feature-parallel histogram building at moderate feature counts

### Proposed optimizations

1. **Sequential threshold for small nodes** (medium win):
   - When `n_samples_in_node < 500`, skip Rayon entirely and use sequential path
   - Currently threshold is 5000 — too high. Rayon setup cost (~2μs) dominates for small nodes
   - File: `ferroml-core/src/models/hist_boosting.rs:1291` — change `5_000` to `500`

2. **Separate gradient/hessian accumulation** (medium win):
   - Current: accumulates grad + hess + count in one pass with AoS layout
   - Alternative: Two separate passes — first accumulate gradients into f64 array, then hessians
   - Separate passes may auto-vectorize better since each is a simple scatter-add
   - Risk: Two passes over sample indices = 2x memory traffic. May not help.

3. **Split finding optimization** (small win):
   - Pre-compute cumulative sums during histogram building (running sum approach)
   - Avoids a separate scan in find_best_split
   - File: `ferroml-core/src/models/hist_boosting.rs` — `find_best_split_from_histogram()`

4. **Reduce tree overhead** (small win):
   - Profile to check if sample index partition (left/right split) is significant
   - Currently copies indices into new vecs — could use in-place partition

### Effort estimate: Medium-Large. Optimization 1 is quick (~5 min). Others need profiling. 2.0x target is achievable with #1 + #3.

---

## Gap 3: SVC (10.2x → 5.0x target)

### What was already done
- Platt's first-order WSS3 heuristic for working set selection
- LRU kernel cache (cache_size configurable)
- Shrinking heuristics
- Band-aid 10K sample threshold for cache strategy

### Remaining bottleneck
- sklearn wraps libsvm (C code, decades of optimization)
- libsvm's kernel cache is highly optimized with column-major access patterns
- Our SMO implementation uses Array2-based kernel matrix which has poor cache locality
- Working set selection scans all samples every iteration

### Proposed optimizations

1. **Kernel computation caching** (high impact):
   - Pre-compute kernel diagonal (used in every WSS iteration)
   - Cache recently used kernel columns (not full rows)
   - File: `ferroml-core/src/models/svm/` — kernel cache module

2. **Reduce gradient update cost** (medium impact):
   - After SMO step, only update gradients for non-zero alpha samples
   - Currently updates all n gradients even though many alphas are at bounds
   - Use shrinking more aggressively to reduce working set size

3. **Optimized RBF kernel computation** (medium impact):
   - Current: computes exp(-gamma * ||x_i - x_j||^2) per pair
   - Use GEMM trick: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
   - Pre-compute norms, use BLAS for the dot product matrix
   - File: `ferroml-core/src/models/svm/kernels.rs`

4. **Parallel kernel column computation** (medium impact):
   - When computing a new kernel column, parallelize across rows with Rayon
   - Only useful for large n (>5K)

### Effort estimate: Large. SVC is the hardest to optimize. 5.0x target requires #1 + #2 + #3. This is 2-3 sessions.

---

## Execution Order

```
Session 1: HistGBT quick wins (#1 Rayon threshold, #3 split finding)
           → Build + benchmark → commit

Session 2: SVC kernel optimization (#1 diagonal cache, #3 GEMM-based RBF)
           → Build + benchmark → commit

Session 3: SVC working set + shrinking (#2)
           LogReg coordinate descent (if needed for 1.5x target)
           → Final benchmark refresh → commit
```

## Files to modify

| File | Changes |
|------|---------|
| `ferroml-core/src/models/hist_boosting.rs` | Rayon threshold, split finding |
| `ferroml-core/src/models/svm/solver.rs` | Gradient update, shrinking |
| `ferroml-core/src/models/svm/kernels.rs` | GEMM-based RBF, diagonal cache |
| `ferroml-core/src/models/logistic.rs` | Coordinate descent (if needed) |
