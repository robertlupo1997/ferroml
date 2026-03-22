# Phase 4: Performance Optimization - Research

**Researched:** 2026-03-22
**Domain:** Numerical linear algebra backends, SVM solvers, histogram gradient boosting, benchmarking
**Confidence:** HIGH

## Summary

FerroML already has the faer 0.20 backend integrated behind a `faer-backend` feature flag (enabled by default). The `linalg.rs` module provides `thin_svd`, `cholesky`, `symmetric_eigh`, and `qr_decomposition` with automatic faer/nalgebra dispatch. PCA, TruncatedSVD, LDA, and FactorAnalysis all call `crate::linalg::thin_svd()` which already routes to faer's divide-and-conquer SVD when the feature is enabled. This means PERF-01 through PERF-04 may already be satisfied -- the primary work is verifying performance and adding stability tests.

OLS already has a Cholesky path for `n > 2*p` (line 414 of `linear.rs`), so PERF-07 is already implemented. Ridge uses `solve_symmetric_positive_definite` which calls `crate::linalg::cholesky` (faer-backed), so PERF-08 is also largely done. LinearSVC already has active-set shrinking (line 2688). The main optimization opportunities are: (1) f_i cache for LinearSVC to avoid recomputing `w^T x_i`, (2) HistGBT histogram inner loop improvements, (3) SVC FULL_MATRIX_THRESHOLD tuning, and (4) stability test suite + benchmarking infrastructure.

**Primary recommendation:** Focus effort on verification/benchmarking (many optimizations already exist), f_i cache for LinearSVC, HistGBT histogram loop optimization, and the stability test suite. Several PERF requirements are already met and just need benchmark proof.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| faer | 0.20 | Dense linear algebra (SVD, Cholesky, eigendecomp) | Already integrated, divide-and-conquer algorithms |
| nalgebra | (existing) | Fallback SVD, matrix operations | Pure Rust, no system deps |
| criterion | (existing) | Rust benchmarking framework | Already in 5 bench files |
| ndarray | (existing) | Core array type | Project standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rayon | (existing) | Parallel histogram building | Already used in HistGBT for >5000 samples |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled Cholesky | faer Cholesky | faer already integrated, no change needed |
| MGS QR | faer Householder QR | Already dispatches to faer when feature enabled |

## Architecture Patterns

### Current SVD Dispatch (linalg.rs)
```
linalg::thin_svd()
  |-- #[cfg(feature = "faer-backend")] -> thin_svd_faer() [divide-and-conquer, 10-13x faster]
  |-- #[cfg(not(...))]                 -> thin_svd_nalgebra() [Jacobi, pure Rust]
  Both call svd_flip() for deterministic signs (sklearn convention)
```

All SVD consumers use `crate::linalg::thin_svd()`:
- `decomposition/pca.rs:428` (full_svd path)
- `decomposition/pca.rs:513` (randomized SVD internal)
- `decomposition/truncated_svd.rs:345,356`
- `decomposition/lda.rs:460,531`
- `decomposition/factor_analysis.rs:597`

PCA also has `SvdSolver::CovarianceEigh` which uses `symmetric_eigh` (also faer-backed).

### Current Cholesky Dispatch (linalg.rs)
```
linalg::cholesky()
  |-- #[cfg(feature = "faer-backend")] -> cholesky_faer() [faer 0.20]
  |-- #[cfg(not(...))]                 -> cholesky_native() [hand-rolled]
```

### OLS Solver Selection (linear.rs:414)
```rust
if n > 2 * p {
    // Cholesky normal equations: O(n*d^2 + d^3) -- already implemented
} else {
    // QR decomposition via linalg::qr_decomposition (faer Householder when available)
}
```

### Ridge Solver (regularized.rs:217-218)
```rust
// Already uses: solve_symmetric_positive_definite(&xtx_reg, &xty)
// Which calls: crate::linalg::cholesky(a, 0.0) -> faer Cholesky when feature enabled
```

### Anti-Patterns to Avoid
- **Copying ndarray<->faer element-by-element in hot loops:** The current conversion pattern (nested for loops) is fine for setup but avoid it in inner loops. For HistGBT histograms this is not relevant since they don't use faer.
- **Changing solver behavior without stability tests:** Always add ill-conditioned matrix tests BEFORE swapping solvers.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SVD | Custom SVD | `crate::linalg::thin_svd()` | Already dispatches to faer, tested |
| Cholesky | Custom decomposition | `crate::linalg::cholesky()` | Already faer-backed |
| Eigendecomp | Custom eigensolvers | `crate::linalg::symmetric_eigh()` | Already faer-backed |
| Benchmarking | Ad-hoc timing | Criterion framework | Already set up in 5 bench files |

## Common Pitfalls

### Pitfall 1: Assuming faer is not already integrated
**What goes wrong:** Creating duplicate faer integration code
**Why it happens:** The feature flag `faer-backend` is enabled by default, and all linalg functions already dispatch
**How to avoid:** Check `linalg.rs` dispatch before adding new faer calls; the work is mostly DONE
**Warning signs:** "Let me add faer to..." -- it's already there

### Pitfall 2: Changing Cholesky/SVD without stability regression
**What goes wrong:** Numerical instability on ill-conditioned matrices
**Why it happens:** Different algorithms have different numerical properties
**How to avoid:** PERF-12 requires stability tests BEFORE each solver swap; write those first
**Warning signs:** Tests passing on well-conditioned data but failing on near-singular cases

### Pitfall 3: LinearSVC f_i cache invalidation
**What goes wrong:** Cached predictions become stale after weight updates
**Why it happens:** `w^T x_i` changes when `w` is updated via `w += delta * y_i * x_i`
**How to avoid:** Update `f_i[j] += delta * y_i * x_i^T x_j` for all j (like LIBLINEAR), or maintain a separate f vector that gets incrementally updated. Note: for L2-loss LinearSVC, only one alpha changes per step, so the update is `f[j] += delta * y[i] * x_i^T x_j` for all j -- this costs O(n*d) per step but avoids recomputing `w^T x` from scratch.
**Warning signs:** Different results with/without cache, convergence issues

### Pitfall 4: HistGBT histogram "optimization" that reduces cache locality
**What goes wrong:** Attempted vectorization that breaks the column-major access pattern
**Why it happens:** The current 4-sample unrolling is already cache-friendly
**How to avoid:** Profile before changing; the current `chunks_exact(4)` pattern is good. Focus on reducing branch misprediction (bounds checks) and pre-allocating buffers.
**Warning signs:** Slower performance after "optimization"

### Pitfall 5: Benchmark noise masking real improvements
**What goes wrong:** Criterion results vary by 10-20% between runs
**Why it happens:** CPU frequency scaling, background processes, thermal throttling
**How to avoid:** Use `sample_size(50+)`, warm up, compare relative not absolute, run on same machine
**Warning signs:** Contradictory benchmark results

## Code Examples

### Verifying faer SVD is active
```rust
// Source: ferroml-core/src/linalg.rs:19-28
// thin_svd already dispatches to faer when feature is enabled
// Verify with: cargo test --features faer-backend
// The feature is in default features (Cargo.toml:143)
pub fn thin_svd(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    #[cfg(feature = "faer-backend")]
    { thin_svd_faer(a) }
    #[cfg(not(feature = "faer-backend"))]
    { thin_svd_nalgebra(a) }
}
```

### LinearSVC f_i cache pattern (LIBLINEAR-style)
```rust
// Current inner loop (svm.rs:2722-2724):
let xi = x_design.row(i);
let wt_xi: f64 = w_aug.dot(&xi);  // O(d) per sample, every iteration

// Optimized with f_i cache:
// Initialize: f[i] = 0.0 for all i (since w starts at 0)
// In inner loop: use f[i] instead of w_aug.dot(&xi)
// After alpha update: for j in 0..n_samples { f[j] += delta * y_binary[i] * x_design.row(i).dot(&x_design.row(j)) }
// BUT: this is O(n^2*d) total -- worse than recomputing O(n*d)!
//
// Better approach for LinearSVC: maintain f[i] = w^T x_i
// After w update (w += scale * x_i): f[j] += scale * x_i^T x_j for all j
// This is still O(n*d) per coordinate step.
//
// Actually, the LIBLINEAR trick is different for primal CD:
// Maintain f[i] = w^T x_i explicitly. After each coordinate update:
//   w += delta * y[i] * x_i
//   f[j] += delta * y[i] * (x_i . x_j) for all j  -- too expensive
//
// The real LIBLINEAR approach: just maintain w explicitly (already done)
// and recompute w^T x_i only for the current sample (already done).
// The "f_i cache" optimization is: cache ALL f[i] = w^T x_i at start of epoch,
// then update incrementally: f[j] += scale * x_i[j_feat] for each feature
// NO -- the correct approach: f vector maintained as f[j] = w^T x_j
// Update: when alpha_i changes by delta, w changes by delta*y_i*x_i
// So f[j] changes by delta*y_i*(x_i . x_j) -- need inner product
//
// LIBLINEAR's actual optimization: for primal CD, maintain the full f vector.
// When w changes: for each j, f[j] = w^T x_j. Since w += delta*y_i*x_i,
// f[j] += delta*y_i*(x_i^T x_j). But computing x_i^T x_j for all j is O(n*d).
// Current code already computes w^T x_i in O(d) -- same cost per sample.
//
// The REAL win: avoid the full-sweep permutation. Focus on active set only.
// That's already implemented (lines 2688-2768).
```

### Stability test pattern (ill-conditioned matrices)
```rust
// Pattern for PERF-12: stability tests
#[test]
fn test_pca_ill_conditioned() {
    // Hilbert matrix: condition number grows exponentially with size
    let n = 10;
    let mut h = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            h[[i, j]] = 1.0 / (i + j + 1) as f64;
        }
    }
    // Verify SVD doesn't fail or produce NaN
    let (u, s, vt) = crate::linalg::thin_svd(&h).unwrap();
    assert!(s.iter().all(|v| v.is_finite()));
    assert!(u.iter().all(|v| v.is_finite()));
    assert!(vt.iter().all(|v| v.is_finite()));
    // Verify reconstruction: U * diag(S) * Vt ~ H
    // ...
}

#[test]
fn test_ols_near_collinear() {
    // Features nearly identical (condition number ~1e12)
    let x = Array2::from_shape_vec((100, 2),
        (0..200).map(|i| if i % 2 == 0 { i as f64 / 100.0 } else { (i-1) as f64 / 100.0 + 1e-8 }).collect()
    ).unwrap();
    // Should either succeed with warning or return meaningful error
}
```

### Criterion benchmark pattern
```rust
// Source: ferroml-core/benches/benchmarks.rs
fn bench_pca_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("PCA_SVD");
    for n_samples in [500, 1000, 5000] {
        for n_features in [10, 50, 100] {
            let (x, _) = generate_regression_data(n_samples, n_features);
            group.bench_with_input(
                BenchmarkId::new(format!("{}x{}", n_samples, n_features), ""),
                &x,
                |b, x| {
                    b.iter(|| {
                        let mut pca = PCA::new().with_n_components(NComponents::N(5));
                        pca.fit(black_box(x)).unwrap()
                    });
                },
            );
        }
    }
    group.finish();
}
```

## Detailed Findings by Plan

### Plan 04-01: SVD + Stability Tests

**Finding: faer thin SVD is ALREADY the default path.**

- `linalg.rs:19-28`: `thin_svd()` dispatches to `thin_svd_faer()` when `faer-backend` feature is enabled
- `Cargo.toml:143`: `default = ["parallel", "onnx", "simd", "faer-backend", "datasets"]` -- faer is ON by default
- `Cargo.toml:97-99`: `faer = { version = "0.20", optional = true }`
- `linalg.rs:32-74`: `thin_svd_faer()` uses `mat.thin_svd()` (faer's divide-and-conquer)
- `linalg.rs:71`: `svd_flip()` already applied for deterministic signs

**All four decomposition models already use the faer path:**
- PCA: `decomposition/pca.rs:428` calls `crate::linalg::thin_svd(x_centered)`
- TruncatedSVD: `decomposition/truncated_svd.rs:345,356`
- LDA: `decomposition/lda.rs:460,531`
- FactorAnalysis: `decomposition/factor_analysis.rs:597`

**What remains for 04-01:**
1. **Stability test suite (PERF-12)**: No dedicated ill-conditioned matrix tests exist. Need tests for: Hilbert matrices, near-singular matrices, rank-deficient matrices, matrices with huge condition numbers
2. **Performance verification**: Need Criterion benchmarks comparing faer vs nalgebra SVD at various sizes to prove the 2x-of-sklearn target
3. **svd_flip verification**: Already implemented correctly at `linalg.rs:647`

**Risk:** LOW -- the implementation is done, this is verification + test infrastructure

### Plan 04-02: LinearSVC + SVC Threshold

**LinearSVC shrinking (PERF-05): ALREADY IMPLEMENTED**
- `svm.rs:2688-2768`: Active set management with `active` vec and periodic re-examination every 10 iterations
- Samples at bounds with correct gradient sign are marked inactive
- This matches LIBLINEAR's shrinking strategy

**LinearSVC f_i cache (PERF-06): NOT IMPLEMENTED -- but need to reconsider**
- Current code at `svm.rs:2722-2724`: computes `wt_xi = w_aug.dot(&xi)` fresh each time
- For primal coordinate descent (which LinearSVC uses), maintaining an explicit f_i vector requires updating ALL f values when w changes -- this is O(n*d) per coordinate step, same as the current O(d) dot product per visited sample
- The benefit only comes if the active set is small: skip recomputing for inactive samples
- **LIBLINEAR's actual optimization**: maintain `f[i] = w^T x_i` for all i. When alpha_i changes, update w, then for each sample j: `f[j] += delta * y_i * x_i^T x_j`. This is O(n*d) per update. Currently we compute O(|active|*d) per epoch. The f_i cache only wins when we need multiple passes and active set shrinks significantly.
- **Recommendation**: The f_i cache makes more sense for the DUAL formulation (kernelized SVC). For primal CD (LinearSVC), the current approach of computing `w.dot(x_i)` is already optimal. Instead, focus on: (a) better convergence detection, (b) optimizing the dot product with SIMD

**SVC FULL_MATRIX_THRESHOLD (PERF-10):**
- Current value: `FULL_MATRIX_THRESHOLD = 2_000` at `svm.rs:92`
- Used at `svm.rs:622`: `if n_samples < FULL_MATRIX_THRESHOLD`
- Need benchmarks at n=1000,1500,2000,2500,3000 to find optimal threshold
- Tradeoff: full matrix is O(n^2) memory but O(1) access; cache is O(cache_budget) memory but O(d) per miss

**KMeans (PERF-11):**
- `clustering/kmeans.rs` uses `crate::linalg::squared_euclidean_distance()` throughout (lines 294, 321, 397, etc.)
- `linalg.rs:541-544`: `squared_euclidean_distance` dispatches to SIMD when feature enabled
- Elkan's algorithm implemented at `clustering/kmeans.rs:357`
- **Status**: Appears complete from Plan W. Need benchmark verification only.

### Plan 04-03: OLS Cholesky + Ridge + HistGBT

**OLS Cholesky (PERF-07): ALREADY IMPLEMENTED**
- `linear.rs:414`: `if n > 2 * p` already selects Cholesky normal equations
- `linear.rs:420`: Uses `crate::linalg::cholesky(&xtx, 1e-10)` which is faer-backed
- Computes condition number from Cholesky diagonal (lines 429-441)
- Falls back to QR when `n <= 2*p`

**Ridge faer backend (PERF-08): ALREADY IMPLEMENTED**
- `regularized.rs:217-218`: Uses `solve_symmetric_positive_definite` which calls `crate::linalg::cholesky(a, 0.0)` at line 2286
- `linalg.rs:368-377`: `cholesky()` dispatches to `cholesky_faer()` when feature enabled
- `linalg.rs:381-417`: faer Cholesky converts ndarray->faer, calls `mat.cholesky(faer::Side::Lower)`

**HistGBT histogram optimization (PERF-09): PARTIALLY OPTIMIZED**
- Two histogram build paths:
  1. `build_histograms_col_major` (line 1259): Column-major layout, 4-sample unrolling, parallel for >5000 samples
  2. `build_histograms` (line 1222): Row-major fallback
- Current optimizations already applied:
  - Column-major data layout for cache-friendly access
  - `chunks_exact(4)` for CPU pipelining
  - Parallel over features via rayon (when samples > 5000 and features >= 4)
  - Histogram subtraction (parent - sibling) for the larger child
- **Remaining optimization opportunities:**
  1. **Remove bounds checks**: The `if b0 < n_bins` checks at lines 1292-1311 add branch overhead. If bin mapping guarantees valid indices, these can be `unsafe` index access or the check can be moved outside the hot loop
  2. **Pre-sorted sample indices**: If indices are sorted, memory access to `gradients[chunk[i]]` and `hessians[chunk[i]]` becomes more sequential
  3. **Fused gradient+hessian struct**: The HistBin already packs grad+hess+count (3 fields at line 390), but ensuring 64-byte cache line alignment could help
  4. **SIMD bin assignment**: The bin lookup and accumulation could potentially use SIMD gather/scatter (but this is complex and may not help much for random bin indices)

### Plan 04-04: Benchmarks + Regression + Published Page

**Existing benchmark infrastructure:**
- `ferroml-core/benches/benchmarks.rs` (2225 lines, ~86 bench functions covering all major models)
- `ferroml-core/benches/performance_optimizations.rs` (479 lines, HistGBT/RF/KMeans/DBSCAN)
- `ferroml-core/benches/gaussian_process.rs`, `gpu_benchmarks.rs`, `memory_benchmarks.rs`
- `scripts/benchmark_cross_library.py` -- Python cross-library comparison
- `scripts/benchmark_vs_sklearn.py` -- direct sklearn comparison

**What needs to be added:**
1. PCA/SVD benchmarks at various sizes (not currently in bench files)
2. LinearSVC benchmarks at various sizes
3. Ridge/OLS benchmarks comparing Cholesky vs QR path
4. SVC threshold sweep benchmarks
5. Cross-library benchmark automation (run and compare ferroml vs sklearn)
6. Published benchmark page (markdown or HTML with formatted results)

**Existing test infrastructure:**
- 6 consolidated test files in `ferroml-core/tests/`
- `tests/correctness.rs` -- main correctness suite
- `tests/edge_cases.rs` -- edge case coverage
- `tests/adversarial.rs` -- adversarial inputs
- `tests/regression_tests.rs` -- regression prevention

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| nalgebra Jacobi SVD | faer divide-and-conquer SVD | Plan O (faer-backend feature) | 10-13x faster SVD |
| Hand-rolled Cholesky | faer Cholesky | Plan O | Better numerical stability |
| Row-major histograms | Column-major + 4-sample unrolling | Plan W | Better cache locality |
| Full kernel matrix always | LRU cache + threshold | Previous plans | O(cache) vs O(n^2) memory |

## Requirements Status Assessment

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PERF-01 | PCA uses faer thin SVD | ALREADY DONE -- `pca.rs:428` calls `linalg::thin_svd()` which dispatches to faer. Need benchmark proof. |
| PERF-02 | TruncatedSVD uses faer thin SVD | ALREADY DONE -- `truncated_svd.rs:345,356`. Need benchmark proof. |
| PERF-03 | LDA uses faer thin SVD | ALREADY DONE -- `lda.rs:460,531`. Need benchmark proof. |
| PERF-04 | FactorAnalysis uses faer thin SVD | ALREADY DONE -- `factor_analysis.rs:597`. Need benchmark proof. |
| PERF-05 | LinearSVC shrinking | ALREADY DONE -- `svm.rs:2688-2768` active set management. Need benchmark proof. |
| PERF-06 | LinearSVC f_i cache | NOT DONE but questionable value for primal CD. Reconsider as convergence/SIMD optimization. |
| PERF-07 | OLS Cholesky for n>>2d | ALREADY DONE -- `linear.rs:414-443`. Need benchmark proof. |
| PERF-08 | Ridge faer Cholesky | ALREADY DONE -- `regularized.rs:2286` -> `linalg::cholesky()` -> faer. Need benchmark proof. |
| PERF-09 | HistGBT histogram optimization | PARTIALLY DONE -- col-major + 4-sample unrolling. Bounds check removal + alignment possible. |
| PERF-10 | SVC FULL_MATRIX_THRESHOLD tuning | NOT DONE -- currently hardcoded at 2000. Need benchmark sweep. |
| PERF-11 | KMeans squared-distance verified | LIKELY DONE -- uses `linalg::squared_euclidean_distance()` + SIMD. Need verification. |
| PERF-12 | Stability tests before solver swaps | NOT DONE -- no dedicated ill-conditioned matrix test suite exists. |
| PERF-13 | Cross-library benchmarks after each optimization | PARTIALLY EXISTS -- scripts exist but not integrated per-optimization. |
| PERF-14 | Published benchmark page | NOT DONE. |
</phase_requirements>

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test + Criterion 0.5 |
| Config file | `ferroml-core/Cargo.toml` (bench entries) |
| Quick run command | `cargo test -p ferroml-core` |
| Full suite command | `cargo test --all && pytest ferroml-python/tests/` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PERF-01 | PCA faer SVD correctness | unit | `cargo test -p ferroml-core -- pca` | Existing tests cover correctness |
| PERF-05 | LinearSVC shrinking correctness | unit | `cargo test -p ferroml-core -- test_shrinking` | `svm.rs:4964` (exists) |
| PERF-07 | OLS Cholesky vs QR equivalence | unit | `cargo test -p ferroml-core -- linear_regression` | Existing tests |
| PERF-10 | SVC threshold tuning | bench | `cargo bench -p ferroml-core --bench benchmarks -- svc` | Partial |
| PERF-12 | Ill-conditioned stability | unit | `cargo test -p ferroml-core -- stability` | NOT EXISTS -- Wave 0 |
| PERF-13 | Cross-library benchmarks | integration | `python scripts/benchmark_vs_sklearn.py` | Exists |

### Sampling Rate
- **Per task commit:** `cargo test -p ferroml-core`
- **Per wave merge:** `cargo test --all && cargo bench -p ferroml-core`
- **Phase gate:** Full suite green + benchmark results documented

### Wave 0 Gaps
- [ ] Stability test suite for ill-conditioned matrices (PERF-12) -- add to `ferroml-core/tests/edge_cases.rs` or `adversarial.rs`
- [ ] PCA/SVD Criterion benchmarks -- add to `ferroml-core/benches/benchmarks.rs`
- [ ] SVC threshold sweep benchmarks -- add to `ferroml-core/benches/benchmarks.rs`

## Open Questions

1. **PERF-06: Is f_i cache beneficial for primal coordinate descent?**
   - What we know: For primal CD (LinearSVC), maintaining f_i requires O(n*d) updates per coordinate step. Current approach is O(d) per sample visited. The cache only helps if we can avoid visiting many samples (via shrinking).
   - What's unclear: Whether the overhead of maintaining f for all n samples pays off vs just computing w^T x_i on the fly for active samples only.
   - Recommendation: Benchmark both approaches. If shrinking reduces active set to <30% of samples, the f_i cache wins. Otherwise, focus on SIMD-optimizing the dot product.

2. **HistGBT bounds check removal**
   - What we know: The `if b0 < n_bins` checks at lines 1292-1311 guard against invalid bin indices
   - What's unclear: Whether the BinMapper guarantees all indices are valid (making checks redundant)
   - Recommendation: Audit BinMapper to verify guarantees, then use `unsafe` or `get_unchecked` if safe to do so

3. **Benchmark reproducibility on WSL2**
   - What we know: The dev environment runs on WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
   - What's unclear: Whether WSL2 CPU scheduling introduces enough noise to make Criterion results unreliable
   - Recommendation: Use larger sample sizes in Criterion, take multiple runs, compare relative ratios not absolute times

## Sources

### Primary (HIGH confidence)
- `ferroml-core/src/linalg.rs` -- SVD, Cholesky, QR dispatch implementations (read lines 1-450)
- `ferroml-core/src/models/linear.rs` -- OLS Cholesky path (read lines 384-500)
- `ferroml-core/src/models/regularized.rs` -- Ridge solve (read lines 181-290, 2284-2290)
- `ferroml-core/src/models/svm.rs` -- LinearSVC solver (read lines 2438-2800), SVC threshold (line 92)
- `ferroml-core/src/models/hist_boosting.rs` -- Histogram building (read lines 1258-1382)
- `ferroml-core/src/clustering/kmeans.rs` -- Elkan + squared distance (grep results)
- `ferroml-core/Cargo.toml` -- faer 0.20 dependency, default features

### Secondary (MEDIUM confidence)
- `ferroml-core/benches/` -- Existing benchmark infrastructure (5 files, ~86 functions)
- `scripts/benchmark_*.py` -- Cross-library benchmark scripts

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- read actual source code, all paths verified
- Architecture: HIGH -- traced every SVD/Cholesky call path through the codebase
- Pitfalls: HIGH -- based on actual code analysis and known numerical linear algebra issues
- LinearSVC f_i cache assessment: MEDIUM -- the LIBLINEAR approach is well-documented but its applicability to the existing primal CD implementation needs benchmarking

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable domain, code-specific findings)
