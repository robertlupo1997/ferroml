# FerroML v0.4.0 — Launch Hardening

## What This Is

FerroML is a Rust ML library with Python bindings competing with scikit-learn, statsmodels, and XGBoost. It differentiates by making statistical diagnostics (residual analysis, assumption tests, confidence intervals) first-class features in every model. The library has 55+ models, 22+ preprocessors, and 5,650+ tests cross-validated against sklearn, scipy, linfa, and statsmodels.

This milestone addresses all known concerns from the codebase audit to make FerroML launch-ready: fix correctness bugs, close performance gaps, harden robustness, and prepare for public release.

## Core Value

Every model produces correct results with proper error handling — no silent NaN propagation, no panics on edge cases, no known test failures.

## Requirements

### Validated

- ✓ 55+ ML models with fit/predict API — existing
- ✓ Statistical diagnostics on all models (StatisticalModel trait) — existing
- ✓ Python bindings for all models via PyO3 — existing
- ✓ Pipeline, ColumnTransformer, FeatureUnion composition — existing
- ✓ AutoML with Bayesian optimization and ensemble selection — existing
- ✓ Cross-validation framework (KFold, Stratified, TimeSeries, Group) — existing
- ✓ ONNX export for all 118 model types — existing
- ✓ Sparse matrix support (CountVectorizer, TF-IDF, TextPipeline) — existing
- ✓ Comprehensive metrics (classification, regression, clustering) — existing
- ✓ GPU shader support (optional) — existing
- ✓ Cross-library correctness tests vs sklearn/scipy/linfa/statsmodels — existing

### Active

- [ ] NaN/Inf input validation — reject at fit time instead of silent propagation
- [ ] Fix 6 pre-existing test failures (TemperatureScaling, IncrementalPCA)
- [ ] SVM kernel cache correctness with shrinking — unit tests for cache internals
- [ ] Empty data handling — clean FerroError instead of panic
- [ ] PCA performance — replace nalgebra Jacobi SVD with faer thin SVD
- [ ] LinearSVC performance — add shrinking + f_i cache (target: 1.5-2x vs sklearn)
- [ ] OLS/Ridge performance — Cholesky normal equations + faer backend
- [ ] HistGBT performance — optimize histogram binning inner loop
- [ ] SVC performance — tune FULL_MATRIX_THRESHOLD, improve WSS3 convergence
- [ ] KMeans performance — verify Plan W squared-distance fix is complete
- [ ] Unwrap/expect audit — replace unsafe unwraps in critical paths with proper error handling
- [ ] Parameter validation at construction time — eager errors on invalid hyperparameters
- [ ] Document known gaps (RandomForest parallel non-determinism, sparse limitations)
- [ ] Upgrade ort to stable 2.0.0 when available (or document RC status)

### Out of Scope

- Full sparse support for all algorithms — workaround exists (dense conversion), too large for this milestone
- Deterministic parallel RandomForest — rayon work-stealing is fundamental; document as known limitation
- Rate limiting / DoS protection for AutoML — application-level responsibility
- Mobile/WASM targets — not relevant for initial launch
- New model implementations — this milestone is hardening, not feature addition

## Context

- Codebase: ~183K lines Rust, two-crate workspace (ferroml-core + ferroml-python)
- Plans A-X complete — all feature work done, this is polish/hardening
- Codebase map exists at `.planning/codebase/` (ARCHITECTURE.md, STACK.md, CONCERNS.md, etc.)
- Key dependency: nalgebra for linalg (Jacobi SVD is performance bottleneck), faer already added as faster alternative
- Key dependency: ort 2.0.0-rc.11 for ONNX (RC, not stable)
- Pre-commit hooks enforce cargo fmt, clippy -D warnings, quick tests
- Integration tests consolidated into 6 files to reduce build bloat (~14 GB debug builds)

## Constraints

- **No new dependencies**: Avoid adding crates; use faer (already present) to replace nalgebra hot paths
- **No ndarray-linalg**: Removed to avoid OpenBLAS system dependency — do not re-add
- **Backward compatibility**: Python API must remain compatible with v0.3.x usage
- **Test bar**: All existing 5,650+ tests must continue passing; fix the 6 failing ones
- **Build time**: Don't add new test binaries (tests/ consolidation rule)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use faer for SVD/Cholesky hot paths | Already a dependency, faster than nalgebra Jacobi SVD | — Pending |
| Reject NaN/Inf at fit time | sklearn-compatible behavior, prevents silent corruption | — Pending |
| Document RandomForest parallel non-determinism | Fixing requires deterministic scheduler, unknown perf cost | — Pending |
| Target "launch-ready" not "zero issues" | LOW priority items acceptable as documented limitations | — Pending |

---
*Last updated: 2026-03-20 after initialization*
