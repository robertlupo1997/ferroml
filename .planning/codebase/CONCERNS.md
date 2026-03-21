# Codebase Concerns

**Analysis Date:** 2026-03-20

## Tech Debt

### Defensive Coding: 6,195+ unwrap()/expect() Calls
- **Issue**: 169 Rust files in `ferroml-core/src` contain unwrap() or expect() calls. Many safe (doctests, construction), but some in critical paths risk panics on edge cases.
- **Files**: `ferroml-core/src/models/svm.rs`, `ferroml-core/src/stats/math.rs`, `ferroml-core/src/stats/bootstrap.rs`, and 166 others
- **Impact**: User code hitting unvalidated edge cases (empty data, malformed input, rare numerical conditions) can trigger unhandled panics instead of returning `FerroError`.
- **Fix approach**:
  - Audit high-impact files (SVM, stats, regularized, boosting models) for unsafe unwraps in hot paths
  - Replace with error handling: `as_slice()?.unwrap_or_default()` → `as_slice()?` (array layout guaranteed after `.as_standard_layout()`)
  - Add input validation layer in `validate_fit_input()` to reject NaN/Inf before model fit
  - Phase: Post-launch stability hardening

### Input Validation: NaN/Inf Propagation (Not Rejected)
- **Issue**: Linear models and most Rust code do not validate for NaN/Inf at fit time. These values propagate through linear algebra, producing invalid results.
- **Files**: `ferroml-core/src/models/linear.rs`, `ferroml-core/src/models/regularized.rs`, `ferroml-core/src/models/logistic.rs`
- **Current test note**: `ferroml-core/src/models/compliance_tests/compliance.rs` explicitly documents: "Linear models: NaN/Inf propagate through linear algebra (valid but not sklearn-compatible). Future enhancement: add input validation to reject NaN/Inf."
- **Impact**: Models produce silent NaN-filled predictions. No error raised.
- **Fix approach**:
  - Add `is_finite()` check in `validate_fit_input()` — reject X or y with NaN/Inf immediately
  - Python binding layer (`ferroml-python/src/`) should validate NumPy arrays before passing to Rust
  - Tests exist in `ferroml-python/tests/test_errors.py` but some are skipped (see Known Issues)
  - Priority: HIGH (correctness-blocking)

### Empty Data Handling
- **Issue**: Empty data (n_samples=0) causes panics or invalid errors. `ferroml-python/tests/test_errors.py` line 22 skipped with note: "FerroML panics on empty data - acceptable error handling but not a clean exception."
- **Files**: Multiple models, especially those with array indexing
- **Impact**: Users get panic instead of clean FerroError
- **Fix approach**: Add early check in fit methods: `if x.nrows() == 0 { return Err(FerroError::invalid_input("Empty dataset")) }`
- **Priority**: MEDIUM (rare in practice but user-facing)

---

## Known Bugs

### Skip Markers in Python Tests: 8 Failures Pre-Existing
- **Symptoms**:
  1. `test_errors.py::test_empty_data_raises` — SKIPPED (panics instead of raising)
  2. `test_errors.py::test_parameter_validation_*` (4 tests) — SKIPPED (no constructor-time validation)
  3. `test_automl.py::test_extremely_imbalanced_data` — SKIPPED (single-class CV fold)
  4. `test_reproducibility.py::test_random_forest_determinism_*` (4 tests) — SKIPPED (parallel randomness)
  5. `test_score_all_models.py::test_score_ridge_cv_predict` — SKIPPED (RidgeCV predict NaN — **PRE-EXISTING BUG**)
  6. `test_vs_sklearn_gaps_phase2.py` — 6 pre-existing failures (TemperatureScaling, IncrementalPCA)
- **Files**: `ferroml-python/tests/test_*.py` (multiple)
- **Impact**: These features silently broken or partially implemented
- **Workaround**: Use alternatives or file issue
- **Note**: Plans Y+Z marked these resolved, but they remain in test suite. User should investigate before production use.

### RandomForest Non-Determinism in Parallel Mode
- **Issue**: 4 tests in `test_reproducibility.py` skipped because RandomForest fitting produces different trees when parallel=true due to non-deterministic work-stealing across threads.
- **Files**: `ferroml-core/src/models/tree.rs` (ensemble RandomForest)
- **Root cause**: rayon work-stealing in split selection causes different execution order across runs. Seeding only controls initial randomness, not worker scheduling.
- **Impact**: Cannot guarantee reproducibility with `random_state` when using parallel training.
- **Fix approach**:
  - Document in API: "RandomForest is not bitwise reproducible in parallel mode; use `n_jobs=1` if reproducibility is required"
  - Consider deterministic task scheduling (e.g., static work partitioning) but performance cost unknown
- **Priority**: LOW (acceptable limitation; users can force single-threaded)

---

## Performance Bottlenecks

### SVC (RBF kernel) — Still ~3-5x Slower Than sklearn (Was 17.6x)
- **Problem**: Performance regression partially fixed in Plan Y (17.6x→2-3x target, actual ~3-5x observed)
- **Files**: `ferroml-core/src/models/svm.rs` (lines 80-330: KernelCache implementation; lines 614-625: FULL_MATRIX_THRESHOLD logic)
- **Root cause**: LRU cache misses and band-aid threshold at `FULL_MATRIX_THRESHOLD = 2000`. For n=3000-5000, the slab cache has higher overhead than expected, especially with shrinking interactions.
- **Observed vs sklearn**:
  - Small n (n<500): ~0.5x (faster, acceptable)
  - Medium n (n=1K-2K): ~1.5-2x (acceptable)
  - Large n (n=3K-5K): ~3-5x (regressed from target 2-3x)
- **Evidence**: Plan W performance report shows SVC still slow despite fixes
- **Fix approach**:
  - Tune `FULL_MATRIX_THRESHOLD` — try 1500 or even 1000 to force cache earlier
  - Profile cache hit rates with shrinking enabled (unlikely the bottleneck is LRU, more likely WSS3 convergence)
  - Consider second-order working set selection (WSS3) as planned in Plan Z (not yet implemented)
  - Benchmark before/after with representative datasets
- **Priority**: MEDIUM (acceptable for now, but blocks "competitive with sklearn" claim)
- **Blocked on**: Plan Z (2nd-order WSS3 + shrinking tuning)

### HistGBT — ~2.6x Slower Than XGBoost
- **Problem**: Histogram-based gradient boosting slower than expected
- **Files**: `ferroml-core/src/models/hist_boosting.rs`
- **Root cause**: Histogram binning inner loop not optimized; may involve allocation or cache misses
- **Current status**: Partial fixes in Plan W, but still 2.6x gap
- **Fix approach**:
  - Profile histogram construction with `perf record`
  - Consider pre-allocated histogram buffers
  - Vectorize bin assignment loops
  - Benchmark with Plan Z approach (check for unnecessary allocations, copy operations)
- **Priority**: MEDIUM (acceptable for typical use, but limits XGBoost replacement capability)

### LinearSVC — ~9.6x Slower Than sklearn (Plan Z Target: 1.5-2x)
- **Problem**: Coordinate descent solver for linear SVM slower than LIBLINEAR implementation
- **Files**: `ferroml-core/src/models/svm.rs` (fit_binary method for LinearSVC, lines ~2494-2625)
- **Root cause**: No shrinking (active set management) + no cached predictions (f_i = w^T x_i). Every iteration scans all n samples and recomputes dot products.
- **Expected fix** (Plan Z Phase 2): Shrinking + f_i cache should close gap to ~1.5-2x
- **Priority**: HIGH (critical for large-scale linear SVM use cases)
- **Blocked on**: Plan Z Phase 2 (not yet implemented in codebase)

### PCA — Nalgebra Jacobi SVD 13.8x Slower for n=10K, d=50
- **Problem**: PCA uses nalgebra's Jacobi SVD (O(d³) iterations) instead of faer's thin SVD
- **Files**: `ferroml-core/src/decomposition/pca.rs` (lines 426-429, 494-497, 956-961, 1007-1012)
- **Root cause**: nalgebra Jacobi SVD: 1024×1024 matrix = 3.95s, faer thin SVD = 298ms (13.3x difference)
- **Expected fix** (Plan Z Phase 1): Replace nalgebra::DMatrix::svd() with faer thin_svd_faer()
- **Also affects**: `truncated_svd.rs`, `lda.rs`, `factor_analysis.rs`
- **Priority**: HIGH (correctness not affected, but user experience poor for dimensionality reduction)
- **Blocked on**: Plan Z Phase 1 (not yet implemented)

### OLS/Ridge — 3.3-3.8x Slower (Plan Z Fixes Planned)
- **Problem**:
  - OLS uses MGS QR + scalar triangular solve on tall matrices (n >> d) — inefficient
  - Ridge uses hand-rolled Cholesky instead of faer-backed version
- **Files**: `ferroml-core/src/models/linear.rs` (fit method, lines 384-502), `ferroml-core/src/models/regularized.rs` (lines 182-270, 2254-2302)
- **Expected fix** (Plan Z Phase 1): Use Cholesky normal equations for n >> 2d, faer Cholesky for Ridge
- **Priority**: MEDIUM (acceptable gap but fixable)
- **Blocked on**: Plan Z Phase 1

### KMeans — 3.2x Slower (Partial Fix in Plan W)
- **Problem**: Squared distances computation includes sqrt() in hot path + multiple Vec copies
- **Files**: `ferroml-core/src/models/clustering/kmeans.rs`
- **Expected fix** (Plan W): Use squared distances directly, zero-copy row slicing
- **Status**: Plan W claims fix, but verify if fully implemented
- **Priority**: LOW (accepted as "within 3x for typical use")

---

## Fragile Areas

### PyO3 Bindings: 55+ Models Must Stay in Sync
- **Files**:
  - Rust: `ferroml-core/src/models/` (55+ implementations)
  - PyO3 wrappers: `ferroml-python/src/` (50+ .rs wrapper files)
  - Python re-exports: `ferroml-python/python/ferroml/<submodule>/__init__.py` (14 submodules)
- **Fragility**: Adding a new model requires 3 updates:
  1. Core impl in `ferroml-core/src/models/model_name.rs`
  2. PyO3 wrapper in `ferroml-python/src/model_name.rs`
  3. Re-export in `ferroml-python/python/ferroml/<submodule>/__init__.py`
  - Incomplete bindings = user-facing "Model not available in Python" or import errors
  - Test coverage: `ferroml-python/tests/test_bindings_correctness.py` validates 30+ bindings but not exhaustive
- **Consequence**: Easy to break during refactoring
- **Safe modification**:
  - Always update all 3 places atomically (single commit)
  - Run `pytest ferroml-python/tests/test_bindings_correctness.py` to verify
  - Spot-check Python import: `python -c "from ferroml.model_name import Model; print(Model.__doc__)"`
- **Priority**: MEDIUM (process issue, not code issue)

### SVM Cache: Slab-Based LRU Has Subtle Interactions With Shrinking
- **Files**: `ferroml-core/src/models/svm.rs` (lines 97-150: KernelCache, lines 614-625: selection logic)
- **Why fragile**:
  - Cache correctness depends on intrusive linked list invariants (lru_prev, lru_next, lru_head, lru_tail)
  - Shrinking (marking samples inactive) doesn't reset cache state — unused rows may be evicted mid-iteration
  - FULL_MATRIX_THRESHOLD boundary is tuned but not validated across datasets
  - Small changes to eviction order or active set tracking can cause correctness bugs or performance cliffs
- **Safe modification**:
  - Add assertions for LRU invariants in cache::get_row() (check linked list is consistent)
  - Unit test cache with shrinking enabled (currently missing)
  - Benchmark on edge cases: n=1900, n=2100 (near threshold), tiny cache (capacity=10)
- **Test coverage**: SVM tests in `ferroml-core/tests/correctness.rs` verify predict() but not cache internals
- **Priority**: HIGH (correctness-critical)

### ONNX Export: Uses ort 2.0.0-rc.11 (Release Candidate)
- **Issue**: Dependency on ONNX Runtime RC version, not stable
- **Files**: `ferroml-core/Cargo.toml` (line 92: `ort = "2.0.0-rc.11"`)
- **Risk**:
  - RC versions may have breaking changes before 2.0.0 stable
  - If ort 2.0.0 stable released with API changes, integration breaks
  - Only optional feature, but `default = ["onnx", ...]` means users get RC by default
- **Current test coverage**: `ferroml-python/tests/test_onnx_roundtrip.py` — all 118 tests pass as of Plan V
- **Fix approach**:
  - Monitor ort releases; upgrade to 2.0.0 stable when available
  - Consider pinning to exact version in Cargo.toml if stability critical
  - Add note in README: "ONNX export uses RC runtime; stability not guaranteed until ort 2.0.0 stable release"
- **Priority**: LOW (ONNX is optional, well-tested as of Plan V)

### Numerical Stability: Partial Factor for Regularized Models
- **Issue**: Ridge/Lasso/ElasticNet use hand-rolled Cholesky in some paths; may have numerical issues on ill-conditioned matrices
- **Files**: `ferroml-core/src/models/regularized.rs` (lines 182-270: solve_symmetric_positive_definite)
- **Fix approach** (Plan Z Phase 1): Replace with faer Cholesky (more stable, faster)
- **Current status**: No reported failures in test suite
- **Priority**: LOW (mitigation in place via regularization; fix planned for Plan Z)

---

## Scaling Limits

### SVM Kernel Cache: O(cache_size * n_features) Memory
- **Constraint**: `FULL_MATRIX_THRESHOLD = 2000` means:
  - n < 2000: Full precomputed kernel matrix = O(n²) memory (4 GB @ n=50K)
  - n ≥ 2000: LRU cache = O(cache_size * n) memory (200 MB * 50K = 10 GB for default cache)
- **Practical limit**: n > 100K not feasible even with cache
- **Recommendation for users**: Use LinearSVC for large n (no kernel cache needed)
- **Priority**: ACCEPTED (documented limitation; LinearSVC is workaround)

### AutoML: 26 Ignored Tests (Too Slow for CI)
- **Issue**: `ferroml-core/tests/correctness.rs` has 26 ignored AutoML tests marked `#[ignore]`
- **Reason**: AutoML can run for hours on large datasets with many hyperparameters
- **Impact**: These tests only run on manual `cargo test -- --ignored`, not in CI
- **Consequence**: Regressions in AutoML may not be caught by CI
- **Fix approach**:
  - Add time limit to AutoML config (already exists: `time_budget_seconds`)
  - Create fast CI tests: small dataset, single model, 10 seconds max
  - Full tests run nightly or pre-release only
- **Priority**: LOW (acceptable for long-running feature)

---

## Dependencies at Risk

### ort 2.0.0-rc.11 (ONNX Runtime) — Pre-Release Status
- **Risk**: RC version may have API-breaking changes before stable
- **Impact**: If breaking changes occur, ONNX export fails
- **Mitigation**: ONNX is opt-in feature (`onnx` flag, `onnx-validation` for testing)
- **Action**: Plan upgrade path when ort 2.0.0 stable is released
- **Priority**: LOW (optional feature, well-tested)

### polars 0.46 (Optional, Data Loading)
- **Status**: Active development, generally stable
- **Risk**: Low — used only for optional CSV/Parquet loading
- **Action**: Monitor for breaking changes
- **Priority**: LOW

### nalgebra 0.33 (Linear Algebra) — Jacobi SVD Slow
- **Status**: nalgebra maintained, but Jacobi SVD not competitive
- **Risk**: Plans Z relies on faer as faster alternative; nalgebra not replaceable yet
- **Workaround**: faer 0.20 already added as default dependency; gradual migration in progress
- **Action**: Complete migration from nalgebra SVD to faer thin_svd in Plan Z Phase 1
- **Priority**: MEDIUM (performance, not correctness)

---

## Test Coverage Gaps

### SVM Cache Internals Not Validated
- **What's not tested**: Slab-based LRU cache correctness with shrinking, eviction order, hit rates
- **Files**: `ferroml-core/src/models/svm.rs` (cache implementation, lines 97-320)
- **Risk**: Cache correctness bug could silently produce wrong predictions
- **Safe modification**: Add unit test for cache with mock kernel and shrinking enabled
- **Priority**: HIGH (critical path)

### Empty Data Edge Case
- **What's not tested**: Empty or single-sample datasets across all models
- **Current**: `test_errors.py::test_empty_data_raises` is SKIPPED
- **Fix approach**: Add unit tests for n=0, n=1 edge cases in compliance suite
- **Priority**: MEDIUM (rare but user-facing)

### Extreme Imbalance in Cross-Validation
- **What's not tested**: CV folds with single class (can break stratification)
- **Current**: `test_automl.py::test_extremely_imbalanced_data` is SKIPPED
- **Impact**: AutoML may crash or produce invalid results on highly imbalanced data
- **Fix approach**: Add CV fold validation: skip or merge single-class folds
- **Priority**: MEDIUM (real-world scenario)

### Parallel Non-Determinism in RandomForest
- **What's not tested**: Cannot verify reproducibility because parallel randomness
- **Files**: `ferroml-core/src/models/tree.rs`
- **Workaround**: Document limitation; users can force `n_jobs=1`
- **Priority**: LOW (acceptable limitation)

### TemperatureScaling Calibrator Quality (6 Pre-Existing Failures)
- **What's not tested**: Calibration quality vs sklearn CalibratedClassifierCV
- **Files**: `ferroml-python/tests/test_vs_sklearn_gaps_phase2.py`
- **Status**: 6 tests failing (pre-existing as of Plan V)
- **Action**: Investigate and fix before launch, or document known gap
- **Priority**: HIGH (if marketing "statistical rigor")

### IncrementalPCA (6 Pre-Existing Failures)
- **What's not tested**: Batch vs full PCA equivalence
- **Files**: `ferroml-python/tests/test_vs_sklearn_gaps_phase2.py`
- **Status**: 6 tests failing (pre-existing)
- **Action**: Same as TemperatureScaling
- **Priority**: HIGH (correctness-critical)

---

## Security Considerations

### No Input Sanitization for File I/O
- **Risk**: Loading untrusted pickle/bincode files could execute code or corrupt state
- **Files**: Serialization in `ferroml-core/src/serialization/` (implicit in serde usage)
- **Mitigation**: Serde does not execute code, but malformed files could cause panics or allocation bombs
- **Recommendation for users**: Only load trusted model files; treat as untrusted code
- **Priority**: LOW (user responsibility, documented in security best practices)

### No Rate Limiting on AutoML
- **Risk**: Long-running AutoML can be used for DoS if exposed as a service
- **Current**: AutoML has `time_budget_seconds` parameter, limits execution
- **Recommendation**: Application using FerroML should enforce timeouts and resource limits
- **Priority**: LOW (not library's responsibility; user must configure)

---

## Missing Critical Features

### Parameter Validation at Construction
- **Issue**: Models do not validate hyperparameters at creation time (e.g., `SVC::new().with_c(-1.0)` does not error until `fit()`)
- **Files**: All model constructors in `ferroml-core/src/models/`
- **Impact**: Invalid config discovered late during fit, blocking pipeline
- **Workaround**: Catch error at fit time
- **Fix approach**:
  - Add validation method `fn validate_params() -> Result<()>` called in `fit()`
  - Call at construction for eager errors (builder pattern could support this)
  - Cost: minimal (few comparisons per model)
- **Tests marked SKIPPED**: `test_errors.py::test_parameter_validation_*` (4 tests) expect this
- **Priority**: MEDIUM (user experience improvement)

### Deterministic Parallel Training for RandomForest
- **Issue**: Cannot guarantee reproducibility with `random_state` + `n_jobs > 1`
- **Root cause**: rayon work-stealing introduces non-determinism
- **Workaround**: Force single-threaded: `n_jobs=1`
- **Fix approach**:
  - Document clearly in API
  - Consider deterministic task scheduler (performance cost unknown)
- **Priority**: LOW (acceptable as documented limitation)

### Sparse Model Support for All Algorithms
- **Current**: Sparse matrix types defined in `ferroml-core/src/sparse.rs` but only KNN/SVM use them
- **Missing**: Sparse support for Lasso, Ridge, LogisticRegression, etc.
- **Impact**: Text/NLP features (99% sparse) require dense conversion (huge memory waste)
- **Fix approach**: Add sparse variants or universal dense-to-sparse fallback in preprocessing
- **Priority**: LOW (sparse features work, just inefficient; workaround is to use dedicated sparse libraries)

---

## Launch Readiness Gaps

### 6 Pre-Existing Test Failures (Must Address Before Public Release)
- **Location**: `ferroml-python/tests/test_vs_sklearn_gaps_phase2.py`
- **Models**: TemperatureScalingCalibrator (3 tests), IncrementalPCA (3 tests)
- **Action**:
  1. Run tests, document failure details
  2. Either fix the implementation or mark as known limitation with explanation
  3. Update docs with any gaps
- **Consequence**: If not fixed, users will discover these bugs immediately upon release
- **Priority**: CRITICAL (blocks launch)

### Cliff.toml GitHub URL and Python 3.13 Classifier (Plan Y Target)
- **Status**: Documented in `launch-readiness.md` as fixed in Plan Y, but verify:
  1. `cliff.toml` line 59: Should be `robertlupo1997/ferroml` (not `ferroml/ferroml`)
  2. `ferroml-python/pyproject.toml` line 37: Remove `"Programming Language :: Python :: 3.13"` (CI doesn't test it)
- **Priority**: LOW (config issue, not functional)

---

## Summary Table: Issue Priority & Effort

| Category | Issue | Priority | Effort | Blocks |
|----------|-------|----------|--------|--------|
| **Correctness** | NaN/Inf input validation | HIGH | 2h | Production use |
| **Correctness** | 6 pre-existing test failures (Temp, IncrementalPCA) | CRITICAL | 1d | Launch |
| **Correctness** | SVM cache + shrinking interactions | HIGH | 1d | Model correctness |
| **Performance** | PCA 13.8x (nalgebra SVD) | HIGH | 4h | User experience |
| **Performance** | LinearSVC 9.6x | HIGH | 2d | Large-scale SVM |
| **Performance** | SVC 3-5x (was 17.6x) | MEDIUM | 1d | sklearn parity claim |
| **Performance** | HistGBT 2.6x | MEDIUM | 1d | XGBoost replacement |
| **Safety** | 6,195+ unwrap() calls | MEDIUM | 3d | Robustness |
| **Safety** | Empty data panics | MEDIUM | 4h | Edge case handling |
| **API** | Parameter validation at construction | MEDIUM | 4h | User experience |
| **Bindings** | PyO3 sync with 55+ models | MEDIUM | Process | Maintenance |
| **Documentation** | Document known gaps (TemperatureScaling, RandomForest parallel) | MEDIUM | 2h | Transparency |

---

*Concerns audit: 2026-03-20*
