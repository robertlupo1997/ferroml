# Codebase Concerns

**Analysis Date:** 2026-03-16

## Performance Regressions

**SVC Kernel Caching (Critical — Urgent Fix Required):**
- Issue: SVC performance REGRESSED to 17.6x slower than sklearn (up from 4x) due to LRU kernel cache implementation
- Files: `ferroml-core/src/models/svm.rs` (lines 80-227 for cache logic, lines 91 FULL_MATRIX_THRESHOLD, 84 DEFAULT_CACHE_SIZE)
- Impact: Medium-large datasets (n=5000) now timeout or fail to complete in reasonable time; production use blocked for this scale
- Current mitigation: FULL_MATRIX_THRESHOLD raised to 10K (band-aid only), LRU cache still active for 4K-10K range
- Root cause: LRU cache with shrinking reduces cache hit rates and adds hash lookup overhead on each kernel access; doesn't match libsvm's contiguous memory access pattern
- Fix approach (documented in `docs/plans/2026-03-15-svc-performance-fix.md`):
  1. Implement libsvm-style shrinking (actively manage working set, remove bound variables)
  2. Replace random j-selection with WSS3 (weighted second-order working set selection) for faster convergence
  3. Optimize error cache update to only iterate over active set (not full n samples)
  4. Benchmark & lower FULL_MATRIX_THRESHOLD back to 4K
  - Expected outcome: 17.6x → 2-3x of sklearn (competitive)

**HistGradientBoosting Performance Baseline Gap:**
- Issue: HistGBT Classifier 2.6x slower than sklearn (improved from 15x, but still not competitive)
- Files: `ferroml-core/src/models/hist_boosting.rs`
- Impact: Low-medium (acceptable for non-latency-critical production, but penalizes tree-based ensemble choice)
- Cause: Binary search for bin assignment, Rust overhead in gradient computation vs sklearn's Cython
- Improvement path: SIMD binning, parallel tree building, custom allocator for histogram arrays
- Status: Acceptable as-is (no urgent fix needed)

**LogisticRegression Solver Choice Gap:**
- Issue: LogReg 2.1x slower than sklearn (was 2.5x pre-Plan W)
- Files: `ferroml-core/src/models/logistic.rs`
- Impact: Low (linear models are fast baseline; 2x slower is still <100ms for typical datasets)
- Root cause: IRLS solver used for d<20 (correct numerically), L-BFGS for d>=50 (correct but slower than sklearn's SAG). sklearn uses different solver strategies per case.
- Trade-off: Current approach prioritizes numerical stability (IRLS is exact) over speed
- Improvement path: Implement SAG (Stochastic Average Gradient) for warm-start compatibility

---

## Known Test Failures

**6 Pre-Existing Failures in test_vs_sklearn_gaps_phase2.py:**
- Files: `ferroml-python/tests/test_vs_sklearn_gaps_phase2.py`
- Failures:
  1. `TemperatureScalingCalibrator` — calibration quality vs sklearn CalibratedClassifierCV (sigmoid method)
     - Issue: Slightly different loss minimization strategy during temperature fitting
     - Status: Low priority (calibration works, just not 100% parity)
  2. `IncrementalPCA` — batch vs full PCA equivalence, transformation quality
     - Issue: Accumulated rounding errors in streaming PCA (documented in sklearn as well)
     - Status: Acceptable (incremental PCA inherently has this issue; users warned in docs)
- Impact: Minimal (both models function correctly, edge-case differences)
- No critical bugs in core functionality

---

## Panic & Unwrap Usage (Defensive Programming Gaps)

**Risk Count: 6,429 defensive patterns**
- Files affected: All 200 Rust source files
- Patterns: ~6,429 occurrences of `unwrap()`, `expect()`, `panic!()`, `as_slice().unwrap()` across codebase
- Primary locations:
  - `ferroml-core/src/models/svm.rs` (kernel row slicing — lines 128, 219, 858-900)
  - Documentation examples (safe doctest context)
  - Performance-critical inner loops (justified but risky in edge cases)
- Impact: Low-medium — panics only occur with malformed inputs or violated internal invariants
- Mitigation: Input validation on public APIs; internal invariants maintained by type system
- Improvement path: Replace performance-critical unwraps with debug_assert! or Result types in error paths

---

## Numerical Stability & Edge Cases

**Fisher Z-Transform NaN/Inf Risk:**
- Files: `ferroml-core/src/stats/mod.rs` (line 502 comment notes r=±1 causes NaN/Inf in Fisher z)
- Issue: Correlation coefficients at boundaries (r approaching ±1.0) cause division by zero in z-transform: `z = 0.5 * ln((1+r)/(1-r))`
- Impact: Stats module functions that use z-transform (confidence intervals on correlations) can return NaN
- Fix: Clamp r to [-0.9999, 0.9999] before z-transform or use alternative stable form
- Status: Handled with checks in code, but not universally applied

**Durbin-Watson Edge Cases:**
- Files: `ferroml-core/src/stats/diagnostics.rs` (lines 355, 370)
- Issue: DW returns NaN for n<2 or all-zero residuals (mathematically correct, but can confuse users)
- Impact: Low (documented in docstrings)

**LOF Infinite LRD (Local Reachability Density):**
- Files: `ferroml-core/src/models/lof.rs`
- Issue: If k-NN distance is 0 (duplicate points), LRD becomes infinite; checked (lines 1191, 1193) but can propagate
- Impact: Low (models gracefully return Inf for degenerate cases, not NaN)

---

## Complex Modules (Maintenance & Testing Risk)

**SVM Implementation (4,654 lines):**
- Files: `ferroml-core/src/models/svm.rs`
- Complexity: SMO algorithm with LRU kernel cache, Platt scaling, multiclass strategies (OvO/OvR), class weights
- Issues:
  - Performance regression (discussed above)
  - Dense algorithm implementation (hard to debug)
  - Shrinking mechanism needed but not yet implemented
- Risk: Medium (core algorithm, but well-tested)
- Test coverage: 37+ Rust tests + 100+ Python tests

**HistGradientBoosting (4,286 lines):**
- Files: `ferroml-core/src/models/hist_boosting.rs`
- Complexity: Histogram-based tree building, missing value handling, L2 regularization, per-sample weights
- Issues:
  - Performance gap vs sklearn (2.6x slower)
  - 36 unwrap() calls in binning logic (assumes correct bin indexing)
- Risk: Medium (ensemble algorithm, 200+ lines per method)
- Test coverage: 80+ tests

**Regularized Regression (3,197 lines):**
- Files: `ferroml-core/src/models/regularized.rs`
- Complexity: Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV with coordinate descent, LARS solvers
- Issues:
  - Coordinate descent convergence depends on feature scaling
  - RidgeCV fixed NaN bug in Plan V (ln(-4.0) issue) — suggests fragility in CV pathfinding logic
- Risk: Medium (numerical algorithm)
- Test coverage: 120+ tests

---

## Fragile Areas (Sensitive to Input Changes)

**Pipeline Composition (3,086 lines):**
- Files: `ferroml-core/src/pipeline/mod.rs`
- Why fragile:
  - Strict transformer→estimator ordering (no validation)
  - Feature count mismatch between steps can panic downstream
  - Complex fit/predict chains with state (fitted_ vs unfitted_ models)
- Safe modification: Always validate X.ncols() matches expected features after each transform
- Test coverage: 45+ tests, but gaps in multi-step pipelines with missing values

**Gaussian Process Models (2,288 lines):**
- Files: `ferroml-core/src/models/gaussian_process.rs`
- Why fragile:
  - Cholesky decomposition can fail on poorly conditioned kernels
  - Regularization parameter (alpha) critically affects stability
  - Marginal likelihood optimization can get stuck in local minima
- Safe modification: Always check condition number before Cholesky; use `with_alpha(1e-6)` default (not 1e-8)
- Test coverage: 60+ tests, but no condition-number sensitivity tests

**Tree-Based Node Selection:**
- Files: `ferroml-core/src/models/tree.rs` (2,593 lines)
- Why fragile:
  - Best split selection via `partial_cmp()` (line 828: `.unwrap_or(Ordering::Equal)`) assumes NaN/Inf don't occur in gain
  - Ties broken arbitrarily (first-wins), can differ from sklearn
- Safe modification: Pre-check for NaN/Inf gains; add deterministic tie-breaking (feature index, threshold order)
- Test coverage: 85+ tests, 56 vs sklearn tests pass

---

## Dependency Risks

**linfa 0.8.1 (Cross-Library Validation Only):**
- Usage: Dev dependency for cross-library tests only (`ferroml-core/Cargo.toml` line 78)
- Risk: Low (not in production path)
- Note: Uses ndarray 0.16 (compatible with FerroML)

**Optional Feature Fragmentation:**
- Features: `datasets`, `parallel`, `simd`, `sparse`, `onnx`, `onnx-validation`, `faer-backend`, `gpu`
- Risk: Medium (8 optional feature combinations can have untested interactions)
- Examples:
  - `sparse` models + `gpu` (no GPU sparse kernels implemented)
  - `onnx` + `datasets` (Polars loading may fail with ONNX export)
- Mitigation: CI tests default features + all-features; docs warn of unsupported combinations

**ONNX Runtime (ort 2.0.0-rc.11):**
- Status: Release candidate (not stable)
- Risk: Low (optional feature, ONNX 118 roundtrip tests passing)
- Upgrade path: Upgrade to ort 2.0.0 final when released

---

## Missing Critical Validations

**Input Shape Validation Gaps:**
- Issue: Some models don't fully validate X during fit/predict
- Examples:
  - KNN doesn't validate single-sample datasets
  - Tree-based models skip validation if n_samples < 3
- Files: `ferroml-core/src/models/mod.rs` (validate_fit_input, validate_predict_input functions)
- Risk: Low (models degrade gracefully or error clearly)
- Improvement: Expand `check_*` validators to cover all edge cases (p=1, n=1, n=2)

**Missing Value Handling Inconsistency:**
- Issue: Some models accept NaN, others reject; behavior not consistent
- Examples:
  - Tree-based models handle NaN (go left heuristic)
  - KNN rejects NaN (no neighbor defined)
  - AutoML preprocessing auto-fills NaN
- Files: Multiple model files + `ferroml-core/src/automl/preprocessing.rs`
- Risk: Low (documented per model)
- Improvement: Add schema-based validation option (already exists but not used everywhere)

---

## Build & Deployment Concerns

**Debug Build Size (14 GB):**
- Issue: Post-consolidation (Plans A-X), debug build is ~14 GB vs 350 GB pre-consolidation
- Impact: CI disk usage acceptable, local development slow
- Workaround: Use `cargo build --release` or `maturin develop --release`
- Status: Acceptable (test consolidation fixed root issue)

**Pre-Commit Hooks Enforcement:**
- Hooks: cargo fmt, clippy -D warnings, cargo test (quick)
- Risk: Medium — clippy is strict (blocks commits on warnings), slow test runner can timeout
- Mitigation: Tests are quick (only lib tests + edge cases, not integration tests)

**GitHub Billing Issue (Blocks Release):**
- Status: From memory, this was noted as blocking production deployment
- Action required: Verify GitHub runner limits and billing setup before public release

---

## Test Coverage Gaps

**High-Dimensionality Edge Cases:**
- Missing: Explicit tests for p >> n (tall-thin matrices)
- Affected models: PCA, Ridge, LogisticRegression, LinearSVC
- Impact: Low (models handle via regularization, but no explicit coverage)
- Files to add tests: `ferroml-core/tests/test_edge_cases.rs`

**Class Imbalance with Class Weights:**
- Missing: Tests combining imbalanced datasets + custom class weights
- Affected models: SVC, LogisticRegression, DecisionTree
- Impact: Low (both features tested separately)
- Files: `ferroml-core/tests/test_edge_cases.rs`

**Sparse + GPU Pipeline:**
- Missing: No tests for SparseModel + GPU dispatch together
- Impact: Medium (feature advertised but untested interaction)
- Files: `ferroml-core/tests/test_sparse_models.rs` + GPU tests

---

## Documentation Gaps

**Performance Baseline Documentation:**
- Missing: Explicit guidance on when to use which solver (IRLS vs L-BFGS for LogReg)
- Files: `ferroml-core/src/models/logistic.rs` (docstring only mentions L-BFGS default)
- Impact: Users may not understand 2x overhead of IRLS

**Calibration Quality Caveats:**
- Missing: Notes on TemperatureScalingCalibrator differences vs sklearn
- Files: `ferroml-core/src/models/calibration.rs`
- Impact: Low (docstring exists, but could be more explicit)

**IncrementalPCA Rounding Error Accumulation:**
- Missing: Warning in docstring about accumulated rounding errors
- Files: `ferroml-core/src/decomposition/pca.rs`
- Impact: Low (sklearn documentation also notes this)

---

## Scaling Limits

**KNN & DBSCAN Memory Usage:**
- Constraint: Full distance matrix O(n²) in memory for all-pairs distance calculation
- Limit: ~50K samples (requires 20GB for float64 matrix)
- Scaling path: Implement ball-tree / kd-tree for approximate NN search (medium effort)
- Status: Acceptable for typical use (<100K samples)

**SVM Kernel Matrix Memory:**
- Constraint: Full matrix O(n²) for n < 4K, then LRU cache with O(cache_size * n)
- Limit: Effective max ~100K with cache, but slow (performance regression)
- Scaling path: Implement random features / Nystroem approximation (high effort)

**AutoML Portfolio Size:**
- Constraint: Portfolio has 80+ algorithm variants; time budget splits across them
- Limit: ~300 algorithms total (portfolio + HPO combinations); diminishing returns
- Status: Acceptable (no plans to expand further)

---

## Security Considerations

**ONNX Model Execution (Optional Feature):**
- Risk: Untrusted ONNX models can execute arbitrary code via ONNX runtime
- Mitigation: ort runtime runs in-process (no sandbox); document as "only load trusted ONNX"
- Files: `ferroml-core/src/onnx/` + Python pickle support
- Recommendation: Add warning in docs: "ONNX files from untrusted sources may execute code"

**Pickle Security (Python Bindings):**
- Risk: Python pickle can execute arbitrary code
- Files: `ferroml-python/src/pickle.rs`
- Mitigation: Document "only unpickle models you created or trust"
- Status: Standard Python limitation, not FerroML-specific

**No Explicit Random Seed Management in Production:**
- Risk: AutoML may not be deterministic if seed is None and async parallelism is enabled
- Mitigation: Memory docs note "seed: Some(42) for reproducibility"
- Recommendation: Warn in docs that production deployments should always set seed

---

## Technical Debt Summary

| Item | Severity | Effort | Impact | Priority |
|------|----------|--------|--------|----------|
| SVC 17.6x regression | High | 2-3 days | Blocks medium datasets | **Critical** |
| Shrinking + WSS3 for SMO | High | 2-3 days | 17.6x → 2-3x | **Critical** |
| Input validation gaps | Medium | 1-2 days | Better error messages | High |
| Sparse + GPU tests | Medium | 1 day | Feature confidence | Medium |
| IncrementalPCA rounding docs | Low | 2 hours | User expectation setting | Low |
| Optional feature matrix testing | Medium | 2-3 days | Deployment confidence | High |
| Panic/unwrap audit | Low | 1-2 days | Robustness | Medium |
| LogReg solver optimization | Low | 3-5 days | Speed improvement | Low |

---

*Concerns audit: 2026-03-16*
