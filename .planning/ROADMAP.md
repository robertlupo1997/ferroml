# Roadmap: FerroML v0.4.0 Launch Hardening

## Overview

This milestone hardens FerroML for public release by addressing all known concerns from the codebase audit. The work follows strict sequential dependencies: input validation first (reduces unwrap audit surface), then correctness fixes (establishes clean baseline), then robustness hardening (unwrap audit with validated inputs), then performance optimization (with safety net from prior phases), and finally documentation (captures final state). Every phase delivers an observable improvement in library quality.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Input Validation** - All models reject invalid inputs (NaN/Inf/empty) with clear errors instead of silent corruption
- [ ] **Phase 2: Correctness Fixes** - Fix all known bugs, add numerical safeguards, achieve zero test failures
- [ ] **Phase 3: Robustness Hardening** - Replace unsafe unwraps in critical paths, harden error handling and serialization
- [ ] **Phase 4: Performance Optimization** - Close performance gaps vs sklearn (PCA, LinearSVC, OLS/Ridge, HistGBT, SVC)
- [ ] **Phase 5: Documentation and Release** - Complete docstrings, document limitations, publish benchmarks

## Phase Details

### Phase 1: Input Validation
**Goal**: Users get clear, actionable errors when passing invalid data -- no silent NaN propagation, no panics on empty/degenerate inputs
**Depends on**: Nothing (first phase)
**Requirements**: VALID-01, VALID-02, VALID-03, VALID-04, VALID-05, VALID-06, VALID-07, VALID-08, VALID-09, VALID-10
**Success Criteria** (what must be TRUE):
  1. Passing a NumPy array containing NaN or Inf to any model's fit() raises a clear error naming the offending values
  2. Passing an empty dataset (n=0) to any model's fit() returns a FerroError instead of panicking
  3. Calling predict() on a fitted model with wrong number of features raises a shape mismatch error
  4. Constructing a model with invalid hyperparameters (e.g., negative C, n_clusters=0) raises an error at construction time with an actionable message
  5. Calling predict() on an unfitted model raises NotFitted error for all 55+ models
**Plans**: 3 plans

Plans:
- [ ] 01-01-PLAN.md — Shared validation infrastructure + clustering/decomposition adoption
- [ ] 01-02-PLAN.md — Universal adoption audit for all 55+ models (fit/predict/transform validation, NotFitted, hyperparams)
- [ ] 01-03-PLAN.md — Python binding layer validation (check_array_finite) and Python test suite

### Phase 2: Correctness Fixes
**Goal**: All known correctness bugs are fixed, numerical safeguards are in place, and the test suite is green with zero known failures
**Depends on**: Phase 1
**Requirements**: CORR-01, CORR-02, CORR-03, CORR-04, CORR-05, CORR-06, CORR-07, CORR-08, CORR-09, CORR-10
**Success Criteria** (what must be TRUE):
  1. All 5,650+ tests pass with zero known failures (TemperatureScaling and IncrementalPCA either fixed or removed from public API)
  2. SVM kernel cache has unit tests covering shrinking correctness, eviction order, and hit rates
  3. Models that compute probabilities (LogReg, NaiveBayes variants, GMM) use numerically stable log-sum-exp
  4. SVD decomposition produces consistent component signs regardless of backend (svd_flip implemented in linalg.rs)
  5. Convergence issues produce warnings with iteration count rather than only hard errors
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Fix TemperatureScaling and IncrementalPCA failures (diagnose root cause, fix or remove)
- [ ] 02-02-PLAN.md — SVM cache tests, numerical stability (logsumexp, Cholesky jitter, svd_flip)
- [ ] 02-03-PLAN.md — Output sanity checks, convergence warnings, cross-library test expansion

### Phase 3: Robustness Hardening
**Goal**: Critical code paths cannot panic on any user input -- unwraps replaced with proper error handling, error messages are actionable, serialization is verified
**Depends on**: Phase 2
**Requirements**: ROBU-01, ROBU-02, ROBU-03, ROBU-04, ROBU-05, ROBU-06, ROBU-07, ROBU-08, ROBU-09, ROBU-10, ROBU-11, ROBU-12, ROBU-13
**Success Criteria** (what must be TRUE):
  1. No unwrap() or expect() call in any fit/predict/transform path can be reached with validated user input -- verified by triage audit
  2. clippy::unwrap_used lint is enabled at warning level in CI configuration
  3. Every FerroError variant includes both what went wrong and what the user should do about it
  4. All 55+ Python-exposed models survive pickle roundtrip (serialize then deserialize produces identical predictions)
  5. Python exceptions map correctly to FerroError variants (e.g., ShapeMismatch -> ValueError, NotFitted -> RuntimeError)
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md — Unwrap triage and fix for SVM, boosting, and stats modules (Tier 1-2)
- [ ] 03-02-PLAN.md — Unwrap fixes for linear, tree, preprocessing, clustering + clippy lint setup
- [ ] 03-03-PLAN.md — Error mapping, pickle roundtrip, serialization verification, error message audit

### Phase 4: Performance Optimization
**Goal**: FerroML is within 2-3x of sklearn on all major algorithms, with stability tests proving correctness is maintained
**Depends on**: Phase 3
**Requirements**: PERF-01, PERF-02, PERF-03, PERF-04, PERF-05, PERF-06, PERF-07, PERF-08, PERF-09, PERF-10, PERF-11, PERF-12, PERF-13, PERF-14
**Success Criteria** (what must be TRUE):
  1. PCA on a 10000x100 matrix runs within 2x of sklearn time (faer thin SVD replaces nalgebra Jacobi)
  2. LinearSVC on a 5000-sample binary classification runs within 2x of sklearn (shrinking + f_i cache)
  3. OLS/Ridge on tall datasets (n >> 2d) uses Cholesky path and runs within 2x of sklearn
  4. Stability tests for ill-conditioned matrices (condition number > 1e10) pass BEFORE and AFTER each solver swap
  5. Cross-library performance benchmark results are published and show no regressions from v0.3.1
**Plans**: TBD

Plans:
- [ ] 04-01: Stability test suite + faer thin SVD in linalg.rs with svd_flip (PCA, TruncatedSVD, LDA, FactorAnalysis)
- [ ] 04-02: LinearSVC shrinking + f_i cache, SVC threshold tuning, KMeans verification
- [ ] 04-03: OLS Cholesky path, Ridge faer backend, HistGBT histogram optimization
- [ ] 04-04: Cross-library benchmarks, regression checks, published benchmark page

### Phase 5: Documentation and Release
**Goal**: Every public API element is documented, known limitations are transparent, and the library is ready for public consumption
**Depends on**: Phase 4
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04, DOCS-05, DOCS-06
**Success Criteria** (what must be TRUE):
  1. Every Python model class has a docstring with description, parameter table (type + default + valid range), and usage example
  2. README documents known limitations: RandomForest parallel non-determinism, sparse algorithm limits, ort RC status
  3. Per-model docstrings note model-specific limitations where applicable (e.g., SVC scaling sensitivity)
  4. A public benchmark comparison page exists with methodology, dataset descriptions, and formatted results
**Plans**: TBD

Plans:
- [ ] 05-01: Python docstring audit and completion for all 55+ models
- [ ] 05-02: README limitations section, per-model limitations, ort status documentation, benchmark page

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Input Validation | 0/3 | Planned | - |
| 2. Correctness Fixes | 0/3 | Not started | - |
| 3. Robustness Hardening | 0/3 | Not started | - |
| 4. Performance Optimization | 0/4 | Not started | - |
| 5. Documentation and Release | 0/2 | Not started | - |
