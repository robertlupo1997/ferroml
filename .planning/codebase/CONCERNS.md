# Codebase Concerns

**Analysis Date:** 2026-03-15

---

## Tech Debt

### RidgeCV Predict Returns NaN

**Issue:** `RidgeCV().fit(X, y).predict(X)` returns all NaN values instead of valid predictions.

**Files:** `ferroml-core/src/models/regularized.rs` (RidgeCV implementation, ~lines 1430-1614)

**Impact:** RidgeCV is unusable for production. `score()` and cross-validation workflows will fail silently. This is a critical blocker for users relying on cross-validated ridge regression.

**Status:** Pre-existing bug documented in Plan V. Cause believed to be in alpha selection or inner Ridge delegation. Requires mathematical debugging.

**Fix approach:**
1. Add test case to verify the exact failure point
2. Check alpha selection logic in `_fit_cv` method
3. Verify inner `self.model.predict()` works with selected alpha
4. Add instrumentation to log intermediate values (alphas, coefficients)

**Test location:** `ferroml-python/tests/test_score_all_models.py` — test is marked with `@pytest.mark.skip` due to this bug.

---

### ONNX RandomForestClassifier Roundtrip Mismatch

**Issue:** One pre-existing ONNX test failure persists: `test_onnx_roundtrip_RandomForestClassifier`.

**Files:** `ferroml-python/tests/test_onnx_roundtrip.py`, `ferroml-core/src/onnx/tree.rs`

**Impact:** RandomForest classifier predictions can mismatch when exported to ONNX and re-imported via onnxruntime, though both Rust and ONNX predictions are internally consistent.

**Status:** Addressed in robustness audit (2026-03-14). TreeEnsembleRegressor + ArgMax + normalized leaf probabilities should handle this. If failures persist, likely due to f32 precision loss in probability normalization for rare classes (5/20 random seeds show 1/50 mismatches at 0.5/0.5 ties).

**Fix approach:**
1. Verify normalized leaf probabilities are computed correctly in `onnx/tree.rs`
2. Increase test tolerance from exact match to within 1 class if predictions are near decision boundary
3. If still failing, add f32 precision documentation and accept the mismatch for rare-class predictions

**Test tolerances:** Regressors use `atol=1e-5`, classification labels exact match, probabilities `atol=1e-4`.

---

## Performance Bottlenecks

### HistGradientBoosting 15x Slower Than sklearn

**Problem:** HistGBT is 15x slower than sklearn's implementation on n=5000, p=20 datasets.

**Files:** `ferroml-core/src/models/hist_boosting.rs`

**Cause:** Three identified inefficiencies:
1. **Bin assignment uses linear search** (O(n_bins) per sample) instead of binary search (O(log n_bins))
2. **Full kernel matrix precomputation** — evaluates all splits without caching frequently-used rows
3. **Column-major data copy overhead** in `to_col_major()` (lines 372-384)

**Current metrics:** 1852ms on benchmark dataset, sklearn ~120ms

**Improvement path:** Plan W.4 targets three fixes:
1. Replace `.position()` with `.binary_search_by()` for O(log 256) vs O(256) per bin assignment (~4-5x speedup alone)
2. Lower parallelism threshold from `10_000 && n_features >= 8` to `1_000 && n_features >= 4`
3. Replace column-major copy with column views or in-place column-major layout

**Target:** Within 3x of sklearn (400ms range) — realistic given sklearn uses highly optimized C code.

**Risk:** Even after fixes, HistGBT will likely remain 2-3x slower than sklearn due to architecture differences (numpy array operations vs hand-tuned C).

---

### SVC 4x Slower Than sklearn

**Problem:** SVC training is 4x slower than sklearn at n=5000.

**Files:** `ferroml-core/src/models/svm.rs`

**Cause:** Full O(n²) kernel matrix precomputation in memory, no LRU caching. libsvm (used by sklearn) caches only ~min(n, 1000) kernel rows, saving memory and L-cache pressure.

**Current metrics:** 406ms on benchmark, sklearn ~100ms

**Improvement path:** Plan W.3 requires implementing LRU kernel cache:
1. Add `KernelCache` struct with HashMap<usize, Vec<f64>> + VecDeque for LRU order
2. Replace full `compute_kernel_matrix()` with on-demand `cache.get_row(i, x)` in SMO
3. Set capacity to `min(n_samples, 1000)`

**Target:** Within 1.5x of sklearn (~150ms).

**Implementation complexity:** Medium — requires plumbing LRU requests through SMO inner loops without breaking existing tests.

---

### KMeans 2.5x Slower Than sklearn

**Problem:** KMeans fitting is 2.5x slower than sklearn.

**Files:** `ferroml-core/src/clustering/kmeans.rs`

**Cause:** Lloyd's algorithm with full distance matrix recalculation every iteration. sklearn uses Elkan's algorithm with triangle inequality bounds to skip redundant distance computations.

**Current metrics:** 33.8ms on benchmark, sklearn ~13.5ms

**Improvement path:** Plan W.2 implements Elkan's algorithm:
1. Maintain center-to-center distance cache (k×k matrix, recomputed each iteration)
2. Track per-sample upper bounds `u[i]` and lower bounds `l[i][j]` to the assigned and alternative centers
3. Skip distance computation when `u[i] <= l[i][j]` (triangle inequality guarantees optimality)
4. Update bounds after center movement using delta tracking

**Target:** Within 1.5x of sklearn (~20ms).

**Memory overhead:** O(n·k) for bounds storage. Acceptable for typical k < 100.

---

### LogisticRegression 2.5x Slower Than sklearn at Scale

**Problem:** LogReg IRLS solver is 2.5x slower at n=5000 compared to sklearn's liblinear.

**Files:** `ferroml-core/src/models/logistic.rs`

**Cause:** IRLS computes O(d³) Cholesky decomposition per iteration; liblinear uses L-BFGS with O(d) per iteration and low-rank Hessian approximation.

**Current metrics:** 13.6ms on benchmark, sklearn ~5.4ms

**Improvement path:** Plan W.5 adds L-BFGS solver:
1. Add `solver` parameter: `Irls` (current, default for d < 50) or `Lbfgs` (auto-selected for d >= 50)
2. Implement `LogisticCost` struct with argmin crate's `CostFunction` + `Gradient` traits
3. Use `argmin::solver::linesearch::MoreThuenteLineSearch` + `argmin::solver::quasinewton::LBFGS`

**Target:** Within 1.5x of sklearn at all dataset sizes (~8ms).

**Implementation complexity:** Medium — argmin crate already available as dependency; primarily integration work.

---

### PCA 3x Slower on Tall-Thin Data

**Problem:** PCA randomized SVD is not triggered for tall-thin data (n=1000, p=5000).

**Files:** `ferroml-core/src/decomposition/pca.rs` (lines 396-404)

**Cause:** Auto-selector only triggers when **both** n_samples > 500 AND n_features > 500. Should also trigger for tall-thin data where n_features >> n_samples.

**Current behavior:** Uses full SVD (O(n·p²)) even when randomized SVD (O(n·p·k)) would be faster.

**Fix approach:** Change auto-selector logic:
```rust
// Before: only randomize when both dimensions > 500
// After: also randomize when n_features > 100 && n_features > 2 * n_samples
SvdSolver::Auto => {
    if (n_samples > 500 && n_features > 500)
        || (n_features > 100 && n_features > 2 * n_samples) {
        // use randomized
    }
}
```

**Target:** >2x improvement for tall-thin PCA.

**Complexity:** Trivial (5-line change). Plan W.1.

---

## Fragile Areas

### Complex Models with Large File Size

**Naive Bayes (4,680 lines):**
- **Files:** `ferroml-core/src/models/naive_bayes.rs`
- **Why fragile:** Implements 4 NB variants (Gaussian, Multinomial, Bernoulli, Categorical) with dense and sparse paths, ONNX export, and log-probability computations. High density of edge cases (zero probability guards, smoothing parameters).
- **Safe modification:** Changes to log calculations need finite-value guards. Add guard tests before any `.ln()` or `.sqrt()` operations. Cross-reference `audit-report.md` fixes (lines 472-473, 968) for guard patterns.
- **Test coverage:** Comprehensive — 26 correctness tests in `correctness_bayes.rs` validate against scipy/sklearn.

**SVM (4,531 lines):**
- **Files:** `ferroml-core/src/models/svm.rs`
- **Why fragile:** Platt scaling sigmoid, kernel computations, SMO algorithm, sparse data handling, ONNX export. Numerical stability is critical (overflow/underflow in sigmoid, gradient scaling).
- **Safe modification:** All sigmoid computations should use `stable_sigmoid()` helper (line 681, 726). Kernel cache addition (Plan W.3) touches SMO inner loops — high risk of introducing correctness bugs.
- **Test coverage:** 56 Rust tests + cross-library validation vs linfa. Add regression tests for any kernel cache changes.

**HistGradientBoosting (4,279 lines):**
- **Files:** `ferroml-core/src/models/hist_boosting.rs`
- **Why fragile:** Bin assignment, histogram construction, greedy split finding, tree building, ONNX export. Changes to bin assignment algorithm (Plan W.4 Fix 1) are high-risk — small errors cascade to wrong predictions.
- **Safe modification:** Always test bin assignment separately (unit test with known thresholds). Verify histograms match sklearn on toy dataset after changes. Use `atol=1e-5` in ONNX validation.
- **Test coverage:** ONNX roundtrip + 44 correctness tests. Plan W.4 will require extensive benchmarking to verify speedup doesn't break accuracy.

---

### Preprocessing Complexity

**Sampling (3,759 lines):**
- **Files:** `ferroml-core/src/preprocessing/sampling.rs`
- **Why fragile:** Stratified sampling, temporal splits, GroupKFold with group edge cases (empty groups, single-sample groups, mismatched group arrays).
- **Safe modification:** Changes to stratification logic need 100% test coverage. Test with edge cases: single group with n > 1, group with n=1, empty groups after filtering.

**Pipeline (3,086 lines):**
- **Files:** `ferroml-core/src/pipeline/mod.rs`
- **Why fragile:** Chained transformers, ColumnTransformer with mixed sparse/dense, fit→transform→predict flows. State tracking across steps can introduce off-by-one errors or column name mismatches.
- **Safe modification:** Any changes to step execution order need integration tests covering all combination types (dense→dense, sparse→dense, etc.). ColumnTransformer creates dummy 1-row arrays for output shape inference — acceptable risk (transformers validate inputs).

---

## Scaling Limits

### No Active Caching for Large-N Problems

**Current capacity:** Full kernel matrix stored in memory for SVC (O(n²) memory).

**Limit:** Breaks for n > ~20,000 (gigabytes of RAM needed for f64 kernel matrix).

**Scaling path:** Implement LRU kernel cache (Plan W.3) — reduces memory footprint to O(min(n, 1000) · d).

---

### Tree Depth and Recursion

**Files:** `ferroml-core/src/models/tree.rs`

**Current limit:** No enforced maximum tree depth. Deep trees (>1000 levels) can cause stack overflow during recursive predict.

**Risk:** Stack exhaustion on adversarial datasets (e.g., decision stumps on linearly-separable data).

**Mitigation:** Trees grown via DecisionTreeRegressor/Classifier naturally stop at max_depth parameter (default 20). ONNX export has 10M node cap (`check_tree_node_limit` in `onnx/tree.rs`).

**Recommendation:** Add depth limit to tree building to fail gracefully rather than OOM/stack overflow.

---

### PolynomialFeatures Output Explosion

**Files:** `ferroml-core/src/preprocessing/polynomial.rs`

**Current limit:** 1M output features cap (added in robustness audit).

**Scaling formula:** Output dimensions = C(n_features + degree, degree) = (n_features+degree)! / (n_features! · degree!).

**Risk cases:**
- degree=3, n_features=100: C(103, 3) = 176k ✓ (under cap)
- degree=3, n_features=300: C(303, 3) = 4.6M ✗ (exceeds cap, rejected)

**Current behavior:** Returns error when exceeding 1M. Acceptable.

---

### ONNX f32 Precision Loss

**Problem:** ONNX export converts all f64 to f32, losing precision.

**Files:** `ferroml-core/src/onnx/*.rs`

**At-risk areas:**
- **Tree leaf values**: Normalized probabilities for rare classes lose precision (0.001 → 0 in f32)
- **SVM dual coefficients**: Large-magnitude coefficients lose precision
- **Kernel parameters**: Auto-gamma `1/n_features` loses precision for p > 500
- **HistGBT bin thresholds**: `next_down_f32` ULP nudge may not preserve <= semantics near f32 subnormals
- **Tie-breaking**: ~5/20 random seeds show 1/50 mismatch at 0.5/0.5 probability ties

**Current mitigation:** Test tolerances use `atol=1e-5` for regressors, exact match for labels, `atol=1e-4` for probabilities. 44/44 ONNX roundtrip tests pass within these tolerances.

**Known limitation:** Documented but not fixed (acceptable tradeoff for ONNX interoperability).

---

## Dependencies at Risk

### Polars as Mandatory Dependency

**Risk:** ferroml-core depends on polars, used only for `load_csv()` and `load_parquet()`.

**Impact:** Adds ~50MB to minimal ferroml-core builds even for users who don't use file I/O. Increases compile time.

**Migration plan:** Plan W.6 moves polars to optional behind `datasets` feature. Default features include it (backward compatible).

**Complexity:** Simple — gate module with `#[cfg(feature = "datasets")]`.

---

### ngrams Dependency for Text Preprocessing

**Risk:** Unused or minimal usage — check if `ngrams` crate is still imported.

**Status:** No immediate concern if CountVectorizer uses it correctly. Robustness audit (pass 2) fixed CountVectorizer empty vocab issue.

---

## Known Bugs Summary

| Bug | Status | Impact | Fix Priority |
|-----|--------|--------|--------------|
| RidgeCV predict → NaN | Pre-existing | Critical: model unusable | High |
| ONNX RF classifier mismatch | Pre-existing | Low: rare-class precision loss | Low |
| HistGBT 15x slower | Known perf gap | Medium: uncompetitive | High (Plan W.4) |
| SVC 4x slower | Known perf gap | Medium: uncompetitive | High (Plan W.3) |
| KMeans 2.5x slower | Known perf gap | Medium: uncompetitive | Medium (Plan W.2) |
| LogReg 2.5x slower | Known perf gap | Medium: uncompetitive | Medium (Plan W.5) |
| PCA tall-thin 3x slower | Known perf gap | Low: edge case | Trivial (Plan W.1) |

---

## Test Coverage Gaps

### AutoML System Tests

**What's not tested:** Full AutoML workflows with statistical significance testing.

**Files:** `ferroml-core/src/testing/automl.rs`

**Risk:** HPO parameter importance, best trial selection, and multi-trial comparison logic may have subtle bugs.

**Status:** 26 AutoML tests exist but are marked `#[ignore]` due to being slow (system tests that run hundreds of model fits). Uncommented tests don't reach them. No active complaints reported.

**Recommendation:** Keep ignored but periodically run in CI (e.g., nightly or before releases).

---

### Sparse Data End-to-End Testing

**What's not tested:** Full pipelines with sparse input (sparse X → sparse transformations → sparse models).

**Files:** Sparse support scattered across preprocessing, models, pipeline.

**Risk:** Sparse type conversions or shape mismatches may silently drop dimensions.

**Status:** Individual sparse tests exist for most models. No comprehensive sparse-only pipeline test.

**Recommendation:** Add `test_sparse_pipeline_end_to_end.rs` covering: sparse input → sparse scaler → sparse model → predictions match dense equivalent.

---

### Edge Case Combinations

**What's not tested:** Interactions between edge cases (e.g., single-sample data + single-feature data + class imbalance).

**Risk:** Rare but possible combinations may trigger untested code paths.

**Status:** Individual edge cases tested (n=1, p=1, class imbalance). Combinations untested.

**Recommendation:** Fuzz testing or property-based tests would catch these. Low priority given comprehensive individual tests.

---

## Missing Critical Features

### get_params() / set_params() for Hyperparameter Tuning

**Problem:** sklearn's GridSearchCV/RandomSearchCV rely on `get_params()` and `set_params()` to clone and modify models. FerroML uses Rust builders instead.

**Files:** All model structs in `ferroml-core/src/models/`

**Blocks:** GridSearchCV-style workflows from Python. Users must use FerroML's HPO module instead.

**Status:** By-design architectural difference (Rust trait system). Not a bug, but an API gap vs sklearn.

**Workaround:** `ferroml.hpo.GridSearch` and `ferroml.hpo.RandomSearch` provide equivalent functionality.

---

## Last Known State

**Audited:** 2026-03-14 (robustness audit complete — 35/36 issues fixed, 1 by-design)

**Passing Tests:**
- Rust: 3,160+ tests (26 ignored slow AutoML tests)
- Python: ~1,923 tests
- Cross-library: 164 tests (vs linfa, sklearn, xgboost, lightgbm, scipy, statsmodels)
- ONNX roundtrip: 44/44 passing

**Unresolved:** RidgeCV NaN bug + 5-7 performance gaps scheduled in Plan W.

---

*Concerns audit: 2026-03-15*
