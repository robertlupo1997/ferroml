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

- [ ] **CORR-01**: Fix TemperatureScaling calibrator — 3 failing tests pass or model removed from public API
- [ ] **CORR-02**: Fix IncrementalPCA — 3 failing tests pass or model removed from public API
- [ ] **CORR-03**: SVM kernel cache has unit tests for correctness with shrinking enabled
- [ ] **CORR-04**: SVM kernel cache has unit tests for eviction order and hit rates
- [ ] **CORR-05**: Post-predict sanity check detects NaN in model output and raises warning/error
- [ ] **CORR-06**: Log-sum-exp used in all probability computations (LogisticRegression, NaiveBayes variants, GMM)
- [ ] **CORR-07**: Cholesky solver adds jitter fallback on ill-conditioned matrices (condition number > 1e10)
- [ ] **CORR-08**: SVD sign normalization (svd_flip) in linalg.rs — consistent signs across nalgebra/faer backends
- [ ] **CORR-09**: All 55+ models have at least basic cross-library correctness test (expand from 200+ to full coverage)
- [ ] **CORR-10**: Convergence reporting returns result + warning mode (like sklearn ConvergenceWarning) instead of only hard error

### Performance

- [ ] **PERF-01**: PCA uses faer thin SVD instead of nalgebra Jacobi SVD — target within 2x of sklearn
- [ ] **PERF-02**: TruncatedSVD uses faer thin SVD — same optimization as PCA
- [ ] **PERF-03**: LDA uses faer thin SVD — same optimization as PCA
- [ ] **PERF-04**: FactorAnalysis uses faer thin SVD — same optimization as PCA
- [ ] **PERF-05**: LinearSVC implements shrinking (active set management) — target within 2x of sklearn
- [ ] **PERF-06**: LinearSVC implements f_i cache (cached predictions) — complement to shrinking
- [ ] **PERF-07**: OLS uses Cholesky normal equations for n >> 2d (instead of MGS QR)
- [ ] **PERF-08**: Ridge uses faer-backed Cholesky instead of hand-rolled
- [ ] **PERF-09**: HistGBT histogram binning inner loop optimized (pre-allocated buffers, vectorized bin assignment)
- [ ] **PERF-10**: SVC FULL_MATRIX_THRESHOLD tuned with benchmarks at n=1000,1500,2000,2500,3000
- [ ] **PERF-11**: KMeans squared-distance optimization verified complete (Plan W)
- [ ] **PERF-12**: Stability tests added BEFORE each solver swap (ill-conditioned matrix test suite)
- [ ] **PERF-13**: Cross-library performance benchmarks run after each optimization to catch regressions
- [ ] **PERF-14**: Published benchmark comparison page (Criterion results formatted for public consumption)

### Robustness

- [ ] **ROBU-01**: Unwrap/expect calls in fit/predict paths of SVM models replaced with proper error handling
- [ ] **ROBU-02**: Unwrap/expect calls in fit/predict paths of stats modules replaced with proper error handling
- [ ] **ROBU-03**: Unwrap/expect calls in fit/predict paths of boosting models replaced with proper error handling
- [ ] **ROBU-04**: Unwrap/expect calls in fit/predict paths of linear models replaced with proper error handling
- [ ] **ROBU-05**: Unwrap/expect calls in fit/predict paths of tree models replaced with proper error handling
- [ ] **ROBU-06**: Unwrap/expect calls in fit/predict paths of preprocessing modules replaced with proper error handling
- [ ] **ROBU-07**: Unwrap/expect calls in fit/predict paths of clustering models replaced with proper error handling
- [ ] **ROBU-08**: Unwrap/expect calls in remaining modules triaged — safe ones documented, unsafe ones fixed
- [ ] **ROBU-09**: clippy::unwrap_used lint enabled as warning in CI (not deny, to allow gradual cleanup)
- [ ] **ROBU-10**: Error messages across all FerroError variants are actionable (what went wrong + what to do)
- [ ] **ROBU-11**: Python exception mapping verified complete (FerroError variants → appropriate Python exceptions)
- [ ] **ROBU-12**: Model serialization version checking verified on all load paths (is_compatible_with)
- [ ] **ROBU-13**: Pickle roundtrip verified for all 55+ Python-exposed models

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
| CORR-01 | Phase 2 | Pending |
| CORR-02 | Phase 2 | Pending |
| CORR-03 | Phase 2 | Pending |
| CORR-04 | Phase 2 | Pending |
| CORR-05 | Phase 2 | Pending |
| CORR-06 | Phase 2 | Pending |
| CORR-07 | Phase 2 | Pending |
| CORR-08 | Phase 2 | Pending |
| CORR-09 | Phase 2 | Pending |
| CORR-10 | Phase 2 | Pending |
| PERF-01 | Phase 4 | Pending |
| PERF-02 | Phase 4 | Pending |
| PERF-03 | Phase 4 | Pending |
| PERF-04 | Phase 4 | Pending |
| PERF-05 | Phase 4 | Pending |
| PERF-06 | Phase 4 | Pending |
| PERF-07 | Phase 4 | Pending |
| PERF-08 | Phase 4 | Pending |
| PERF-09 | Phase 4 | Pending |
| PERF-10 | Phase 4 | Pending |
| PERF-11 | Phase 4 | Pending |
| PERF-12 | Phase 4 | Pending |
| PERF-13 | Phase 4 | Pending |
| PERF-14 | Phase 4 | Pending |
| ROBU-01 | Phase 3 | Pending |
| ROBU-02 | Phase 3 | Pending |
| ROBU-03 | Phase 3 | Pending |
| ROBU-04 | Phase 3 | Pending |
| ROBU-05 | Phase 3 | Pending |
| ROBU-06 | Phase 3 | Pending |
| ROBU-07 | Phase 3 | Pending |
| ROBU-08 | Phase 3 | Pending |
| ROBU-09 | Phase 3 | Pending |
| ROBU-10 | Phase 3 | Pending |
| ROBU-11 | Phase 3 | Pending |
| ROBU-12 | Phase 3 | Pending |
| ROBU-13 | Phase 3 | Pending |
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
