# KMeans Performance Optimization — Design Document

**Date:** 2026-03-23
**Status:** Research complete, ready for implementation
**Current:** 4.25x slower than sklearn (5000x50, k=10)
**Target:** Within 1.5x of sklearn

## Current Implementation

FerroML's KMeans uses Elkan's algorithm with:
- Triangle inequality bounds (O(n*k) lower bounds)
- Rayon parallelism gated at PARALLEL_MIN_SAMPLES=10,000
- kmeans++ initialization
- Auto algorithm selection (Elkan when k<n/2 and k<=256, else Lloyd)
- `squared_euclidean_distance()` with optional SIMD feature

**File:** `ferroml-core/src/clustering/kmeans.rs` (1593 lines)

## Gap Analysis vs sklearn

sklearn's KMeans uses Cython + OpenMP with these advantages:

| Technique | sklearn | FerroML | Impact |
|-----------|---------|---------|--------|
| BLAS GEMM for batch distances | Yes (level-3 GEMM) | No (point-by-point) | High |
| sqrt-free inner loop | Yes (squared distances throughout) | No (.sqrt() every call) | High |
| Norm precomputation | Yes (||x||^2 once, ||c||^2 per iter) | No | Medium |
| 256-sample chunking | Yes (L2-cache sized) | No | Medium |
| Convergence on center movement | Yes (||C_new - C_old||_F) | No (recomputes inertia) | Low-med |
| Per-iteration allocations | None | Vec<Vec<f64>> per iter | Low |
| Parallel threshold | Low (~256 samples) | 10,000 samples | Low |

## Optimization Catalog

### Tier 1: High Impact (recommended first)

#### 1. Eliminate sqrt from inner loop
**Speedup:** 1.3-1.5x | **Effort:** Small | **Risk:** Low

All Elkan bound comparisons work with squared distances. The triangle inequality
lemma `d(c,c') >= 2*d(x,c') => d(x,c) >= d(x,c')` holds for squared distances
since the comparison direction is preserved.

**Changes:**
- Store all bounds (upper, lower, s, center_dists) as squared distances
- Remove every `.sqrt()` call in `run_elkan` and `run_lloyd`
- Bound update Step 5 changes from additive to multiplicative form:
  - Current: `lower[j] = (lower[j] - delta[j]).max(0.0)`
  - New: work with squared bounds throughout, update via squared arithmetic
- Compute final inertia (already squared) only after convergence
- sqrt only needed for final inertia output if caller expects non-squared

**Validation:** Output labels and inertia must match current implementation within f64 epsilon.

#### 2. BLAS GEMM for batch distance computation
**Speedup:** 1.5-2.5x | **Effort:** Medium | **Risk:** Medium

Decompose distances algebraically:
```
||x_i - c_j||^2 = ||x_i||^2 + ||c_j||^2 - 2 * x_i . c_j
```

- Precompute `x_norms[i] = sum(x[i,:]^2)` once at fit start — O(n*d)
- Each iteration: compute `c_norms[j] = sum(c[j,:]^2)` — O(k*d)
- Compute `X @ C^T` via ndarray `.dot()` or faer GEMM — one (n*d)*(d*k) matmul
- Broadcast: `distances[i,j] = x_norms[i] + c_norms[j] - 2 * XCT[i,j]`
- Clamp to 0.0 for numerical stability (GEMM can produce tiny negatives)

**Where it applies:**
- Initial assignment (replaces per-point loop) — full benefit
- Lloyd's assignment step — full benefit
- Elkan's initial assignment — full benefit
- Elkan's Step 2 — partial benefit (bounds skip most computations, but
  when a distance IS needed, it's still per-point; could batch the non-skipped
  points but adds complexity)

**New function:**
```rust
fn batch_squared_distances(
    x: &Array2<f64>,
    centers: &Array2<f64>,
    x_norms: &Array1<f64>,
) -> Array2<f64>
```

**Dependencies:** Benefits from #12 (norm caching).

#### 3. Remove per-iteration allocations
**Speedup:** 1.1-1.2x | **Effort:** Small | **Risk:** Low

Three allocation sites in the hot loop:

1. `center_data: Vec<Vec<f64>>` — k heap allocations per iteration (line 489, 393)
   - Fix: pre-allocate flat `Vec<f64>` of size k*d, reuse across iterations
   - Copy with `copy_from_slice` instead of `to_vec()`

2. `lowers: vec![0.0f64; k]` — one allocation per sample per iteration in parallel path (line 500)
   - Fix: restructure parallel map to write directly into pre-allocated lower bounds
   - Or: use `collect_into()` pattern with thread-local reusable buffers

3. `results: Vec<(i32, f64, Vec<f64>)>` — collects n tuples with k-sized Vecs (line 493)
   - Fix: write results directly into `labels`, `upper`, `lower` slices using
     `par_chunks_mut` + index-based access instead of collect-then-scatter

#### 4. Convergence on center movement
**Speedup:** 1.05-1.15x | **Effort:** Small | **Risk:** Low

Replace the inertia recomputation loop (lines 712-720) with:
```rust
let max_delta_sq: f64 = deltas.iter().map(|d| d * d).fold(0.0, f64::max);
if max_delta_sq < self.tol * self.tol {
    break;
}
```

Compute final inertia only once after convergence (or at max_iter).
The `deltas` vector is already computed in Step 4.

Matches sklearn behavior (convergence on Frobenius norm of center change).

### Tier 2: Moderate Impact

#### 5. Hamerly's algorithm
**Speedup:** 1.2-1.5x over Elkan for k<=20 | **Effort:** Medium | **Risk:** Medium

Single lower bound per point instead of k bounds. Memory: O(n) vs O(n*k).

**Key insight for k=10 benchmark:** Elkan's lower bounds (400KB) exceed L1 cache.
Hamerly's bounds (80KB) fit in L1. The cache advantage compensates for slightly
lower skip rate at small k.

**Implementation:**
- Add `KMeansAlgorithm::Hamerly` variant
- Track `upper[i]` and `lower[i]` (single value = distance to second-closest)
- Auto-select: Hamerly for k <= 20, Elkan for k > 20

**Algorithm per iteration:**
1. Compute s[j] = 0.5 * min_{j'!=j} d(c_j, c_j') (same as Elkan Step 1)
2. For each point: if upper[i] <= lower[i], skip. Otherwise recompute all k distances.
3. Update centers (same as current)
4. Update bounds: upper[i] += delta[assigned]; lower[i] -= max(delta[all])

#### 6. SoA center layout
**Speedup:** 1.1-1.3x | **Effort:** Medium | **Risk:** Low-medium

Store centers transposed (d x k) so all k values of feature f are contiguous.
Benefits GEMM path directly (X @ C^T is natural). For per-point distance,
benefit is smaller.

#### 7. Lower parallel threshold
**Speedup:** 1.1-1.2x | **Effort:** Tiny | **Risk:** Very low

Lower `PARALLEL_MIN_SAMPLES` from 10,000 to 2,000. At n=5000 with 16 threads,
chunk_size=312 is viable for rayon. Benchmark to verify overhead is acceptable.

#### 8. Raw slice pattern for centers
**Speedup:** 1.05-1.1x | **Effort:** Small | **Risk:** Very low

Extract center data as contiguous slice once per iteration (same pattern as x_data):
```rust
let centers_data = centers.as_slice().expect("contiguous");
let center_row = |j: usize| -> &[f64] {
    &centers_data[j * n_features..(j+1) * n_features]
};
```

Eliminates repeated `centers.row(j).as_slice()` in inner loop.

### Tier 3: Algorithmic Alternatives & Future

#### 9. Yinyang KMeans (large k)
**Speedup:** 2-3x over Elkan for k>=50 | **Effort:** High | **Risk:** Medium

Groups centroids into t=k/10 groups, uses three-level filtering (global, group,
local). Dominant algorithm for large k in literature. Not useful at k=10 benchmark
but important for production use cases (image quantization, codebook generation).

#### 10. Mini-batch KMeans (new model)
**Speedup:** 10-100x for large n | **Effort:** Medium | **Risk:** Medium

Separate `MiniBatchKMeans` struct with SGD-style center updates on random
mini-batches. Trades ~1-3% clustering quality for massive speed on large datasets.
Important for sklearn API parity.

#### 11. SIMD distance tuning
**Speedup:** 1.3-1.8x | **Effort:** Medium | **Risk:** Low

Verify auto-vectorization with `target-cpu=native`. Consider explicit AVX2
(`std::arch::x86_64`) for the distance function if auto-vectorization is
insufficient. Moot for the GEMM path but matters for Elkan's per-point calls.

#### 12. Norm caching
**Speedup:** 1.05-1.1x | **Effort:** Small | **Risk:** Low

Precompute `x_norms[i] = ||x_i||^2` once. Per-point distance becomes:
`x_norms[i] + c_norms[j] - 2 * dot(x_i, c_j)` — one dot product instead of
d subtractions + d multiplications + d additions.

Prerequisite for GEMM optimization (#2).

#### 13. Drake/Hamerly adaptive
**Speedup:** 1.1-1.3x over Hamerly | **Effort:** Medium | **Risk:** Medium

Maintain b=max(1, k/8) lower bounds per point. Auto-tunes between Hamerly and
Elkan. Only worth implementing after Hamerly (#5) exists.

## Recommended Implementation Order

**Phase A — Quick wins (est. 2.0-2.5x combined):**
1. Eliminate sqrt (#1)
2. Skip inertia recomputation (#4)
3. Raw slice for centers (#8)
4. Remove per-iter allocations (#3)
5. Lower parallel threshold (#7)

**Phase B — GEMM path (est. 1.5-2.5x additional):**
6. Norm caching (#12)
7. BLAS GEMM distances (#2)

**Phase C — Algorithm selection (est. 1.2-1.5x additional):**
8. Hamerly algorithm (#5)
9. Auto-select Hamerly vs Elkan based on k

**Phase D — Future (as needed):**
10. Yinyang for large-k workloads (#9)
11. Mini-batch KMeans (#10)
12. SIMD tuning (#11)

## Compound Speedup Estimate

| After Phase | Conservative | Optimistic | vs sklearn |
|-------------|-------------|------------|------------|
| Current | 1.0x | 1.0x | 4.25x slower |
| Phase A | 2.0x | 2.5x | 2.1x / 1.7x slower |
| Phase B | 3.0x | 6.0x | 1.4x / 0.7x (faster!) |
| Phase C | 3.6x | 9.0x | 1.2x / 0.5x (faster!) |

Phases A+B should bring FerroML within 1.5x of sklearn. Phase C could make
it competitive or faster on some workloads.

## References

- [Elkan, "Using the Triangle Inequality to Accelerate k-Means" (ICML 2003)](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)
- [Hamerly, "Making k-means Even Faster" (SDM 2010)](https://www.ccs.neu.edu/home/radivojac/classes/2024fallcs6140/hamerly_sdm_2010.pdf)
- [Drake & Hamerly, "Accelerated k-means" (Book Chapter 2014)](https://www.ccs.neu.edu/home/radivojac/classes/2024fallcs6140/hamerly_bookchapter_2014.pdf)
- [Ding et al., "Yinyang K-Means" (ICML 2015)](http://proceedings.mlr.press/v37/ding15.pdf)
- [sklearn KMeans implementation blog](https://scikit-learn.fondation-inria.fr/implementing-a-faster-kmeans-in-scikit-learn-0-23/)
- [Bottesch et al., "Speeding up KMeans via Block Vectors" (ICML 2016)](http://proceedings.mlr.press/v48/bottesch16.pdf)
- [Colfax CFXKMeans optimization](https://colfaxresearch.com/cfxkmeans/)
