# Plan W — Fix the Library: Performance, Dependencies, Positioning

## Overview

FerroML has 95+ algorithms and 5,099 tests but zero real-world users. This plan fixes the critical issues that would prevent adoption: performance gaps that make the library uncompetitive, unnecessary dependency bloat, and missing materials that demonstrate FerroML's unique value (statistical rigor).

**Goal**: Make FerroML credibly competitive on the algorithms people actually use, slim the build, and ship tutorials that showcase the statistical rigor differentiator.

## Current State

- v0.3.1 tagged and pushed, GitHub billing blocks CI/CD
- HistGBT is 15x slower than sklearn (bin assignment uses linear search, no Elkan's)
- SVC is 4x slower (computes full kernel matrix upfront instead of LRU cache)
- KMeans is 2.5x slower (no triangle inequality / Elkan's algorithm)
- LogReg is 2.5x slower at scale (IRLS with O(d³) Cholesky vs sklearn's L-BFGS)
- PCA is 3x slower on tall-thin data (randomized SVD threshold wrong)
- Polars is a mandatory dependency in ferroml-core, used only for CSV/Parquet loading
- No tutorial notebooks demonstrating the statistical rigor angle
- 5 Python models have untested `fit_dataframe()` methods, rest don't

## Desired End State

- HistGBT within 3x of sklearn (from 15x)
- SVC within 1.5x of sklearn (from 4x)
- KMeans within 1.5x of sklearn (from 2.5x)
- LogReg within 1.5x of sklearn at all dataset sizes (from 2.5x at n=5K)
- PCA within 1.5x of sklearn on tall-thin data (from 3x)
- Polars optional behind feature flag
- 3-4 tutorial notebooks demonstrating statistical rigor
- Ready for r/rust and r/machinelearning announcement

## Implementation Phases

### Phase W.1: PCA Quick Fix (5 min)

**Overview**: Fix the randomized SVD auto-selector threshold — currently only triggers when BOTH n_samples > 500 AND n_features > 500. Should also trigger for tall-thin data (d >> n).

**Changes Required**:
1. **File**: `ferroml-core/src/decomposition/pca.rs` (~line 396-404)
   - Change auto-selector logic:
   ```rust
   // Before: only randomize when both dimensions > 500
   // After: also randomize when n_features > 100 && n_features > 2 * n_samples
   SvdSolver::Auto => {
       if (n_samples > 500 && n_features > 500)
           || (n_features > 100 && n_features > 2 * n_samples) {
           // use randomized
       } else {
           // use full
       }
   }
   ```

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core -- pca --quiet` passes
- [ ] Automated: Benchmark PCA with n=1000, d=5000 shows >2x improvement

---

### Phase W.2: KMeans — Elkan's Algorithm (medium effort)

**Overview**: Implement Elkan's algorithm with triangle inequality bounds to avoid redundant distance computations. This is the standard optimization sklearn uses.

**Changes Required**:
1. **File**: `ferroml-core/src/clustering/kmeans.rs`
   - Add center-to-center distance cache: `center_dists: Array2<f64>` (k×k matrix, recomputed each iteration)
   - Add per-sample upper bound `u[i]` (distance to assigned center) and lower bounds `l[i][j]` (distance to each center)
   - Modify assignment step: skip distance computation when `u[i] <= l[i][j]` (triangle inequality guarantees current assignment is optimal)
   - Add `s[j] = 0.5 * min_{j' != j} d(c_j, c_j')` — half the min inter-center distance
   - Skip step: if `u[i] <= s[assigned[i]]`, skip sample entirely
   - Update bounds after center movement: `l[i][j] = max(l[i][j] - delta_j, 0)`, `u[i] += delta_assigned[i]`

2. **File**: `ferroml-core/src/clustering/kmeans.rs`
   - Add `algorithm` parameter: `Lloyd` (current) or `Elkan` (new, default)
   - Keep Lloyd as fallback for sparse data or very high-dimensional data where bounds overhead > savings

**Target**: Within 1.5x of sklearn (from 2.5x)

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core -- kmeans --quiet` passes
- [ ] Automated: All existing clustering tests pass
- [ ] Manual: Benchmark KMeans n=5000 shows speedup from 33.8ms toward ~20ms

---

### Phase W.3: SVC — LRU Kernel Cache (medium effort)

**Overview**: Replace the full O(n²) kernel matrix with an LRU cache that stores ~min(n, 1000) kernel rows. This is how libsvm (and sklearn's SVC wrapper) works.

**Changes Required**:
1. **File**: `ferroml-core/src/models/svm.rs`
   - Add `KernelCache` struct:
     ```rust
     struct KernelCache {
         cache: HashMap<usize, Vec<f64>>,  // row_index -> kernel values
         order: VecDeque<usize>,           // LRU eviction order
         capacity: usize,                   // max cached rows
     }
     ```
   - Replace `compute_kernel_matrix()` call with on-demand `cache.get_row(i, x)` during SMO
   - Cache capacity: `min(n_samples, 1000)` rows (configurable via `cache_size` parameter)
   - Modify `select_j()` and `examine_example()` to use cached kernel evaluations

2. **File**: `ferroml-core/src/models/svm.rs`
   - Remove full kernel matrix precomputation for `fit()`
   - Keep precomputed matrix option for small datasets (n < 500) where cache overhead isn't worth it

**Target**: Within 1.5x of sklearn (from 4x at n=5000)

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core -- svm --quiet` passes
- [ ] Automated: All cross-library SVM validation tests pass
- [ ] Manual: Benchmark SVC n=5000 shows speedup from 406ms toward ~150ms

---

### Phase W.4: HistGBT — Critical Optimizations (high effort, biggest impact)

**Overview**: Three targeted fixes to close the 15x gap. We won't match LightGBM (50-60x gap is a C++ optimization war), but we should get within 3x of sklearn.

**Changes Required**:

**Fix 1: Binary search for bin assignment** (biggest single fix)
1. **File**: `ferroml-core/src/models/hist_boosting.rs` (~line 310-336)
   - Replace `.iter().position(|&e| value < e)` with proper binary search:
   ```rust
   // Before: O(n_bins) linear scan per sample per feature
   let bin = edges.iter().position(|&e| value < e).unwrap_or(edges.len()).saturating_sub(1);

   // After: O(log n_bins) binary search
   let bin = match edges.binary_search_by(|e| e.partial_cmp(&value).unwrap()) {
       Ok(pos) => pos,
       Err(pos) => pos.saturating_sub(1),
   };
   ```
   - Expected: ~4-5x speedup on bin assignment alone (log₂(256) = 8 vs 256 comparisons)

**Fix 2: Lower parallelism threshold**
2. **File**: `ferroml-core/src/models/hist_boosting.rs` (~line 1271)
   - Change `indices.len() > 10_000 && n_features >= 8` to `indices.len() > 1_000 && n_features >= 4`
   - Most real datasets hit this earlier

**Fix 3: Avoid column-major copy**
3. **File**: `ferroml-core/src/models/hist_boosting.rs` (~line 372-384)
   - `to_col_major()` copies entire binned matrix into Vec<Vec<u8>>
   - Replace with column slice views into the original Array2<u8> (use `.column(f).to_vec()` only once, or pass column views directly)
   - Alternatively, store binned data in column-major layout from the start

**Target**: Within 3x of sklearn (from 15x)

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core -- hist --quiet` passes
- [ ] Automated: Cross-library HistGBT tests pass
- [ ] Manual: Benchmark HistGBClassifier n=5000 shows speedup from 1852ms toward ~400ms

---

### Phase W.5: LogisticRegression — L-BFGS Solver (medium-high effort)

**Overview**: Add L-BFGS as an alternative solver to IRLS. IRLS uses O(d³) Cholesky decomposition per iteration; L-BFGS uses O(d) per iteration with a low-rank Hessian approximation. FerroML already has the `argmin` crate as a dependency, which includes L-BFGS.

**Changes Required**:
1. **File**: `ferroml-core/src/models/logistic.rs`
   - Add `solver` parameter: `Irls` (current default for small d), `Lbfgs` (new default for d > 50)
   - Implement L-BFGS solver using `argmin::solver::linesearch::MoreThuenteLineSearch` + `argmin::solver::quasinewton::LBFGS`
   - Define `LogisticCost` struct implementing argmin's `CostFunction` + `Gradient` traits
   - Auto-select: IRLS for d < 50 (exact Hessian is cheap), L-BFGS for d >= 50

2. **File**: `ferroml-python/src/linear.rs`
   - Expose `solver` parameter in Python bindings

**Target**: Within 1.5x of sklearn at all dataset sizes (from 2.5x at n=5K)

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core -- logistic --quiet` passes
- [ ] Automated: Cross-library LogReg validation tests pass
- [ ] Manual: Benchmark LogReg n=5000 shows speedup from 13.6ms toward ~8ms

---

### Phase W.6: Make Polars Optional (30 min)

**Overview**: Gate Polars behind a `datasets` feature flag so users who don't need file I/O get a much lighter build.

**Changes Required**:
1. **File**: `ferroml-core/Cargo.toml`
   - Move `polars` to optional: `polars = { workspace = true, optional = true }`
   - Add feature: `datasets = ["dep:polars"]`
   - Add `datasets` to `default` features (so existing users aren't broken)

2. **File**: `ferroml-core/src/datasets/loaders.rs`
   - Gate the entire module: `#[cfg(feature = "datasets")]`

3. **File**: `ferroml-core/src/datasets/mod.rs`
   - Gate the `loaders` module import with `#[cfg(feature = "datasets")]`

4. **File**: `ferroml-python/src/linear.rs`
   - Remove or keep the 5 `fit_dataframe()` methods — they're already feature-gated and untested. Consider removing entirely and documenting `df.to_numpy()` workaround.

**Success Criteria**:
- [ ] Automated: `cargo build -p ferroml-core --no-default-features` compiles without polars
- [ ] Automated: `cargo build -p ferroml-core` (with defaults) still works
- [ ] Automated: All tests pass with default features

---

### Phase W.7: Remove Dead DataFrame Methods (15 min)

**Overview**: The 5 `fit_dataframe()`/`predict_dataframe()` methods on linear models are untested and incomplete (only 5 of 55+ models). Remove them and document the `df.to_numpy()` pattern instead.

**Changes Required**:
1. **File**: `ferroml-python/src/linear.rs`
   - Remove `fit_dataframe()` and `predict_dataframe()` methods from LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
2. **File**: `ferroml-python/src/polars_utils.rs`
   - Can be removed entirely if no other consumers exist
3. **File**: README or docs
   - Add one-liner showing `df.to_numpy()` pattern for Polars users

**Success Criteria**:
- [ ] Automated: `maturin develop --release -m ferroml-python/Cargo.toml && pytest ferroml-python/tests/ -q` passes

---

### Phase W.8: Tutorial Notebooks — Statistical Rigor (high effort, high value)

**Overview**: Create 3-4 Jupyter notebooks that demonstrate FerroML's unique value: statistical rigor that sklearn doesn't provide. These are the primary marketing material for the announcement.

**Notebooks to Create**:

1. **`notebooks/01_model_comparison.ipynb`** — "Are These Models Actually Different?"
   - Train 3-4 models on a dataset
   - Use `score()` + corrected resampled t-test to determine if differences are statistically significant
   - Show confidence intervals on CV scores
   - Contrast with sklearn: "sklearn tells you model A scored 0.85 and model B scored 0.84. But is that difference real?"

2. **`notebooks/02_prediction_uncertainty.ipynb`** — "How Confident Is This Prediction?"
   - LinearRegression with prediction intervals
   - GaussianProcess with uncertainty bands
   - Show when the model knows it doesn't know

3. **`notebooks/03_assumption_checking.ipynb`** — "Is Your Linear Model Actually Valid?"
   - Fit LinearRegression
   - Run assumption tests (normality, homoscedasticity, multicollinearity)
   - Show residual diagnostics, influential observations
   - Demonstrate what happens when assumptions are violated

4. **`notebooks/04_fair_model_selection.ipynb`** — "AutoML That Tells You Why"
   - Use AutoML with `statistical_tests: true`
   - Show how it handles multiple testing correction
   - Demonstrate that "best model" includes uncertainty

**Success Criteria**:
- [ ] Manual: Each notebook runs end-to-end without errors
- [ ] Manual: Notebooks tell a compelling story about why statistical rigor matters

---

### Phase W.9: Benchmarks and Announcement (after billing fix)

**Overview**: Regenerate benchmarks with the performance fixes, then announce.

**Steps**:
1. Fix GitHub billing (manual, user action)
2. Verify `pip install ferroml` works after PyPI publish
3. Re-run `python scripts/benchmark_cross_library.py` with performance fixes
4. Regenerate feature parity scorecard
5. Write announcement post for r/rust and r/machinelearning
   - Lead with the statistical rigor angle
   - Include benchmark table (only show algorithms where FerroML is competitive or better)
   - Link to tutorial notebooks
   - Be honest about maturity (beta, seeking feedback)

**Success Criteria**:
- [ ] Manual: `pip install ferroml` works from PyPI
- [ ] Manual: Benchmark results reflect performance improvements
- [ ] Manual: Announcement draft reviewed

---

## Phase Ordering and Dependencies

```
W.1 (PCA fix)          ──┐
W.2 (KMeans Elkan)     ──┤
W.3 (SVC kernel cache) ──┤── All independent, can parallelize
W.4 (HistGBT fixes)    ──┤
W.5 (LogReg L-BFGS)    ──┤
W.6 (Polars optional)  ──┤
W.7 (Remove DataFrame) ──┘
                          │
                          ▼
W.8 (Tutorial notebooks) ── needs working library
                          │
                          ▼
W.9 (Benchmarks + announce) ── needs perf fixes + billing fix + tutorials
```

W.1-W.7 are independent and can be done in any order or in parallel.
W.8 depends on a working library (ideally after perf fixes for credible demos).
W.9 is the capstone — depends on everything else.

## Effort Estimates

| Phase | Effort | Impact |
|-------|--------|--------|
| W.1 PCA threshold | 5 min | 3x→1.5x on tall-thin data |
| W.2 KMeans Elkan | 3-4 hours | 2.5x→1.5x |
| W.3 SVC kernel cache | 3-4 hours | 4x→1.5x |
| W.4 HistGBT fixes | 4-6 hours | 15x→3x (target) |
| W.5 LogReg L-BFGS | 4-5 hours | 2.5x→1.5x at scale |
| W.6 Polars optional | 30 min | Lighter builds |
| W.7 Remove DataFrame | 15 min | Cleaner API |
| W.8 Tutorials | 4-6 hours | Primary marketing material |
| W.9 Announce | 2-3 hours | Get users |

**Total**: ~25-30 hours of work across 9 phases.

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| HistGBT can't reach 3x target | Medium | Even 5x is much better than 15x; document remaining gap honestly |
| Elkan's KMeans increases memory usage (O(n·k) bounds) | Low | Fall back to Lloyd for very large k or n; memory is cheap |
| L-BFGS convergence differs from IRLS | Medium | Keep IRLS as default for small d; validate against sklearn fixtures |
| SVC kernel cache correctness | Low | Existing cross-library validation tests will catch regressions |
| Tutorial notebooks reveal bugs | Medium | Actually a feature — better to find bugs now than after announcement |
| GitHub billing not fixed | High | User must resolve; everything else can proceed without it |

## What This Plan Does NOT Include

- New algorithms (ComplementNB, Spectral Clustering, etc.)
- sklearn migration guide
- v1.0.0 release
- Deep learning / GPU training
- DataFrame-native API
- Matching LightGBM's HistGBT speed (50-60x gap is unrealistic to close)
