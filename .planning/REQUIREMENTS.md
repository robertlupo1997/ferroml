# Requirements: FerroML v0.4.0 Launch Hardening

**Defined:** 2026-03-20
**Core Value:** Every model produces correct results with proper error handling — no silent NaN propagation, no panics on edge cases, no known test failures.

## v1 Requirements

Requirements for launch-ready release. Each maps to roadmap phases.

### Input Validation

- [x] **VALID-01**: All models reject NaN/Inf values in X at fit time with clear FerroError
- [x] **VALID-02**: All models reject NaN/Inf values in y at fit time with clear FerroError
- [x] **VALID-03**: All models reject NaN/Inf values in X at predict time with clear FerroError
- [x] **VALID-04**: All models return clean FerroError on empty dataset (n=0) instead of panicking
- [x] **VALID-05**: All models handle single-sample input (n=1) — either succeed or return clean error
- [x] **VALID-06**: All models validate n_features_in_ at predict time matches fit time
- [x] **VALID-07**: All models enforce NotFitted guard on predict/transform (audit completeness)
- [x] **VALID-08**: Hyperparameters validated at construction time with actionable error messages (e.g., "Parameter C must be positive, got -1.0")
- [x] **VALID-09**: NaN/Inf validation implemented as single shared function in validation layer, not per-model
- [x] **VALID-10**: Python binding layer validates NumPy arrays before passing to Rust

### Correctness

- [x] **CORR-01**: Fix TemperatureScaling calibrator — 3 failing tests pass or model removed from public API
- [x] **CORR-02**: Fix IncrementalPCA — 3 failing tests pass or model removed from public API
- [x] **CORR-03**: SVM kernel cache has unit tests for correctness with shrinking enabled
- [x] **CORR-04**: SVM kernel cache has unit tests for eviction order and hit rates
- [x] **CORR-05**: Post-predict sanity check detects NaN in model output and raises warning/error
- [x] **CORR-06**: Log-sum-exp used in all probability computations (LogisticRegression, NaiveBayes variants, GMM)
- [x] **CORR-07**: Cholesky solver adds jitter fallback on ill-conditioned matrices (condition number > 1e10)
- [x] **CORR-08**: SVD sign normalization (svd_flip) in linalg.rs — consistent signs across nalgebra/faer backends
- [x] **CORR-09**: All 55+ models have at least basic cross-library correctness test (expand from 200+ to full coverage)
- [x] **CORR-10**: Convergence reporting returns result + warning mode (like sklearn ConvergenceWarning) instead of only hard error

### Performance

- [x] **PERF-01**: PCA uses faer thin SVD instead of nalgebra Jacobi SVD — target within 2x of sklearn
- [x] **PERF-02**: TruncatedSVD uses faer thin SVD — same optimization as PCA
- [x] **PERF-03**: LDA uses faer thin SVD — same optimization as PCA
- [x] **PERF-04**: FactorAnalysis uses faer thin SVD — same optimization as PCA
- [x] **PERF-05**: LinearSVC implements shrinking (active set management) — target within 2x of sklearn
- [x] **PERF-06**: LinearSVC implements f_i cache (cached predictions) — complement to shrinking
- [x] **PERF-07**: OLS uses Cholesky normal equations for n >> 2d (instead of MGS QR)
- [x] **PERF-08**: Ridge uses faer-backed Cholesky instead of hand-rolled
- [x] **PERF-09**: HistGBT histogram binning inner loop optimized (pre-allocated buffers, vectorized bin assignment)
- [x] **PERF-10**: SVC FULL_MATRIX_THRESHOLD tuned with benchmarks at n=1000,1500,2000,2500,3000
- [x] **PERF-11**: KMeans squared-distance optimization verified complete (Plan W)
- [x] **PERF-12**: Stability tests added BEFORE each solver swap (ill-conditioned matrix test suite)
- [x] **PERF-13**: Cross-library performance benchmarks run after each optimization to catch regressions
- [x] **PERF-14**: Published benchmark comparison page (Criterion results formatted for public consumption)

### Robustness

- [x] **ROBU-01**: Unwrap/expect calls in fit/predict paths of SVM models replaced with proper error handling
- [x] **ROBU-02**: Unwrap/expect calls in fit/predict paths of stats modules replaced with proper error handling
- [x] **ROBU-03**: Unwrap/expect calls in fit/predict paths of boosting models replaced with proper error handling
- [x] **ROBU-04**: Unwrap/expect calls in fit/predict paths of linear models replaced with proper error handling
- [x] **ROBU-05**: Unwrap/expect calls in fit/predict paths of tree models replaced with proper error handling
- [x] **ROBU-06**: Unwrap/expect calls in fit/predict paths of preprocessing modules replaced with proper error handling
- [x] **ROBU-07**: Unwrap/expect calls in fit/predict paths of clustering models replaced with proper error handling
- [x] **ROBU-08**: Unwrap/expect calls in remaining modules triaged — safe ones documented, unsafe ones fixed
- [x] **ROBU-09**: clippy::unwrap_used lint enabled as warning in CI (not deny, to allow gradual cleanup)
- [x] **ROBU-10**: Error messages across all FerroError variants are actionable (what went wrong + what to do)
- [x] **ROBU-11**: Python exception mapping verified complete (FerroError variants → appropriate Python exceptions)
- [x] **ROBU-12**: Model serialization version checking verified on all load paths (is_compatible_with)
- [x] **ROBU-13**: Pickle roundtrip verified for all 55+ Python-exposed models

### Documentation

- [ ] **DOCS-01**: All 55+ Python model classes have complete docstrings (description, parameters, examples)
- [ ] **DOCS-02**: All constructor parameters documented with type, default, valid range
- [ ] **DOCS-03**: Known limitations documented in README (RF parallel non-determinism, sparse limits, ONNX RC)
- [ ] **DOCS-04**: Per-model known limitations documented in docstrings where applicable
- [ ] **DOCS-05**: Upgrade ort dependency status documented (RC vs stable, user expectations)
- [ ] **DOCS-06**: Published performance benchmark page with methodology and results

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Sparse Support

- **SPARSE-01**: Sparse algorithm variants for Lasso, Ridge, LogisticRegression
- **SPARSE-02**: Universal sparse-to-dense fallback with memory warning

### Advanced Performance

- **APERF-01**: SVC 2nd-order WSS3 working set selection
- **APERF-02**: SIMD-accelerated NaN validation scan for large datasets
- **APERF-03**: Deterministic parallel RandomForest (static work partitioning)

### Extended Testing

- **TEST-01**: Property-based testing with proptest for all models (NaN/Inf/empty strategies)
- **TEST-02**: Fuzzing for serialization/deserialization paths

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full sparse algorithm support | Massive scope; dense conversion works for moderate data; v0.5+ |
| Deterministic parallel RandomForest | Requires replacing rayon scheduler; unknown perf cost; document instead |
| GPU training for all models | Marginal gain for classical ML; keep optional shaders |
| WASM/mobile targets | Different compilation targets; not where ML users are |
| Custom loss functions / plugin system | Power users write Rust directly; not worth complexity |
| DataFrame native input (pandas/polars) | NumPy arrays are lingua franca; conversion is user responsibility |
| Distributed training | Entirely different architecture; out of scope for single-machine library |
| sklearn check_estimator compliance | Different language, different constraints; follow conventions, don't mimic |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| VALID-01 | Phase 1 | Complete |
| VALID-02 | Phase 1 | Complete |
| VALID-03 | Phase 1 | Complete |
| VALID-04 | Phase 1 | Complete |
| VALID-05 | Phase 1 | Complete |
| VALID-06 | Phase 1 | Complete |
| VALID-07 | Phase 1 | Complete |
| VALID-08 | Phase 1 | Complete |
| VALID-09 | Phase 1 | Complete |
| VALID-10 | Phase 1 | Complete |
| CORR-01 | Phase 2 | Complete |
| CORR-02 | Phase 2 | Complete |
| CORR-03 | Phase 2 | Complete |
| CORR-04 | Phase 2 | Complete |
| CORR-05 | Phase 2 | Complete |
| CORR-06 | Phase 2 | Complete |
| CORR-07 | Phase 2 | Complete |
| CORR-08 | Phase 2 | Complete |
| CORR-09 | Phase 2 | Complete |
| CORR-10 | Phase 2 | Complete |
| PERF-01 | Phase 4 | Complete |
| PERF-02 | Phase 4 | Complete |
| PERF-03 | Phase 4 | Complete |
| PERF-04 | Phase 4 | Complete |
| PERF-05 | Phase 4 | Complete |
| PERF-06 | Phase 4 | Complete |
| PERF-07 | Phase 4 | Complete |
| PERF-08 | Phase 4 | Complete |
| PERF-09 | Phase 4 | Complete |
| PERF-10 | Phase 4 | Complete |
| PERF-11 | Phase 4 | Complete |
| PERF-12 | Phase 4 | Complete |
| PERF-13 | Phase 4 | Complete |
| PERF-14 | Phase 4 | Complete |
| ROBU-01 | Phase 3 | Complete |
| ROBU-02 | Phase 3 | Complete |
| ROBU-03 | Phase 3 | Complete |
| ROBU-04 | Phase 3 | Complete |
| ROBU-05 | Phase 3 | Complete |
| ROBU-06 | Phase 3 | Complete |
| ROBU-07 | Phase 3 | Complete |
| ROBU-08 | Phase 3 | Complete |
| ROBU-09 | Phase 3 | Complete |
| ROBU-10 | Phase 3 | Complete |
| ROBU-11 | Phase 3 | Complete |
| ROBU-12 | Phase 3 | Complete |
| ROBU-13 | Phase 3 | Complete |
| DOCS-01 | Phase 5 | Pending |
| DOCS-02 | Phase 5 | Pending |
| DOCS-03 | Phase 5 | Pending |
| DOCS-04 | Phase 5 | Pending |
| DOCS-05 | Phase 5 | Pending |
| DOCS-06 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 53 total
- Mapped to phases: 53
- Unmapped: 0

---
*Requirements defined: 2026-03-20*
*Last updated: 2026-03-20 after roadmap creation*
