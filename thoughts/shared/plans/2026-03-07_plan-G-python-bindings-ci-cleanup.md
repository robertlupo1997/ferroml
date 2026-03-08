# Plan G: Python Bindings Completion + CI Hardening + Code Quality

## Overview

Bundle all small, quick-win tasks: expose 14 Rust models to Python, fix README claims, harden CI, and clean up code quality issues. Each phase is independent and can be done in any order.

## Current State

- 14 fully-implemented Rust models have NO Python bindings
- README claims some models are available that aren't actually exposed
- CI has permissive `|| true` and `continue-on-error` patterns
- 5 code quality issues (unused imports/vars, missing docs, clippy suggestions)

## Desired End State

- All 14 models exposed to Python with tests
- README accurately reflects what's available
- CI fails on real issues (clippy strict, benchmarks meaningful)
- Code quality issues resolved

---

## Phase G.1: Expose Naive Bayes Models to Python

**Overview**: Add PyO3 bindings for GaussianNB, MultinomialNB, BernoulliNB.

**Changes Required**:

1. **File**: `ferroml-python/src/naive_bayes.rs` (NEW)
   - Create PyO3 wrappers: `PyGaussianNB`, `PyMultinomialNB`, `PyBernoulliNB`
   - Each needs: `__new__`, `fit`, `predict`, `predict_proba`, `__repr__`
   - Reference Rust structs:
     - `ferroml-core/src/models/naive_bayes.rs:113` (GaussianNB)
     - `ferroml-core/src/models/naive_bayes.rs:701` (MultinomialNB)
     - `ferroml-core/src/models/naive_bayes.rs:1205` (BernoulliNB)
   - Follow existing pattern from `ferroml-python/src/linear.rs` for fit/predict wrappers

2. **File**: `ferroml-python/src/lib.rs`
   - Add `mod naive_bayes;` declaration
   - Register submodule in `PyModule`

3. **File**: `ferroml-python/python/ferroml/naive_bayes/__init__.py` (NEW)
   - Re-export all three classes with docstrings
   - Add `__all__` list

4. **File**: `ferroml-python/python/ferroml/__init__.py`
   - Add `naive_bayes` to imports and `__all__`

5. **File**: `ferroml-python/tests/test_naive_bayes.py` (NEW)
   - ~30 tests: fit/predict/predict_proba for each variant
   - Test with appropriate data (Gaussian: continuous, Multinomial: counts, Bernoulli: binary)

**Success Criteria**:
- [ ] `cargo check -p ferroml-python`
- [ ] `pytest ferroml-python/tests/test_naive_bayes.py -v` — all pass

---

## Phase G.2: Expose Regularized CV Models to Python

**Overview**: Add PyO3 bindings for RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier.

**Changes Required**:

1. **File**: `ferroml-python/src/linear.rs` (EDIT)
   - Add PyO3 wrappers: `PyRidgeCV`, `PyLassoCV`, `PyElasticNetCV`, `PyRidgeClassifier`
   - Rust structs at:
     - `ferroml-core/src/models/regularized.rs:1412` (RidgeCV)
     - `ferroml-core/src/models/regularized.rs:1578` (LassoCV)
     - `ferroml-core/src/models/regularized.rs:1793` (ElasticNetCV)
     - `ferroml-core/src/models/regularized.rs:2012` (RidgeClassifier)
   - Each needs: `__new__` with params, `fit`, `predict`, `__repr__`
   - RidgeClassifier also needs `predict_proba` (it's a classifier)
   - CV variants should expose `alpha_` (best alpha found)

2. **File**: `ferroml-python/python/ferroml/linear/__init__.py` (EDIT)
   - Add re-exports for all 4 new classes

3. **File**: `ferroml-python/tests/test_linear.py` (EDIT or NEW `test_regularized_cv.py`)
   - ~25 tests: fit/predict, alpha_ accessor, cross-validation behavior

**Success Criteria**:
- [ ] `cargo check -p ferroml-python`
- [ ] `pytest ferroml-python/tests/test_regularized_cv.py -v` — all pass

---

## Phase G.3: Expose Specialized Regression Models to Python

**Overview**: Add PyO3 bindings for RobustRegression, QuantileRegression, NearestCentroid, Perceptron.

**Changes Required**:

1. **File**: `ferroml-python/src/linear.rs` (EDIT)
   - Add `PyRobustRegression` — Rust: `ferroml-core/src/models/robust.rs:310`
     - Params: loss function, max_iter, tol
   - Add `PyQuantileRegression` — Rust: `ferroml-core/src/models/quantile.rs:78`
     - Params: quantile, alpha, max_iter
   - Add `PyPerceptron` — Rust: `ferroml-core/src/models/sgd.rs:729`
     - Params: max_iter, tol, eta0, penalty

2. **File**: `ferroml-python/src/neighbors.rs` (EDIT)
   - Add `PyNearestCentroid` — Rust: `ferroml-core/src/models/knn.rs:1446`
     - Params: metric, shrink_threshold

3. **File**: `ferroml-python/python/ferroml/linear/__init__.py` (EDIT)
   - Add RobustRegression, QuantileRegression, Perceptron

4. **File**: `ferroml-python/python/ferroml/neighbors/__init__.py` (EDIT)
   - Add NearestCentroid

5. **File**: `ferroml-python/tests/test_specialized_models.py` (NEW)
   - ~25 tests covering all 4 models

**Success Criteria**:
- [ ] `cargo check -p ferroml-python`
- [ ] `pytest ferroml-python/tests/test_specialized_models.py -v` — all pass

---

## Phase G.4: Expose Linear SVM and Calibration Models to Python

**Overview**: Add PyO3 bindings for LinearSVC, LinearSVR, TemperatureScalingCalibrator.

**Changes Required**:

1. **File**: `ferroml-python/src/linear.rs` (EDIT) or new `ferroml-python/src/svm.rs`
   - Add `PyLinearSVC` — Rust: `ferroml-core/src/models/svm.rs:1778`
     - Params: C, max_iter, tol, penalty
   - Add `PyLinearSVR` — Rust: `ferroml-core/src/models/svm.rs:2319`
     - Params: C, epsilon, max_iter, tol

2. **File**: `ferroml-python/src/ensemble.rs` or new file (EDIT)
   - Add `PyTemperatureScalingCalibrator` — Rust: `ferroml-core/src/models/calibration.rs:657`
     - Params: max_iter, lr
     - Methods: fit, predict_proba, temperature_ accessor

3. **File**: Python `__init__.py` files (EDIT)
   - Add to appropriate module exports

4. **File**: `ferroml-python/tests/test_linear_svm.py` (NEW)
   - ~20 tests: LinearSVC, LinearSVR, TemperatureScalingCalibrator

**Success Criteria**:
- [ ] `cargo check -p ferroml-python`
- [ ] `pytest ferroml-python/tests/test_linear_svm.py -v` — all pass

---

## Phase G.5: Fix README and Documentation Discrepancies

**Overview**: README claims models exist that aren't exposed. Fix after G.1-G.4 complete.

**Changes Required**:

1. **File**: `ferroml-python/README.md` (EDIT)
   - Verify every model listed is actually importable from Python
   - Add newly exposed models from G.1-G.4
   - Remove any claims about models that still don't exist

2. **File**: `README.md` (root) (EDIT)
   - Sync model list with actual Python exports

**Success Criteria**:
- [ ] Every model listed in README can be `from ferroml import X` successfully

---

## Phase G.6: CI Hardening

**Overview**: Make CI stricter — clippy should fail builds, remove unnecessary `|| true`.

**Changes Required**:

1. **File**: `.github/workflows/ci.yml` (EDIT)
   - Line ~72: Remove `|| true` from clippy step — make it strict
   - Line ~15: Consider enabling `RUSTFLAGS: -D warnings` (or keep commented with rationale)
   - Line ~230: Remove `|| true` from `cargo install cargo-tarpaulin` — use `cargo binstall` or cache

2. **File**: `.github/workflows/publish-pypi.yml` (EDIT)
   - Line ~82: Remove `|| true` from pytest — Python tests should block publishing

**Success Criteria**:
- [ ] `cargo clippy -p ferroml-core --all-features -- -D warnings` passes locally
- [ ] CI clippy job would fail on real warnings

---

## Phase G.7: Code Quality Fixes

**Overview**: Resolve the 5 TASK-IMP items from the implementation plan.

**Changes Required**:

1. **TASK-IMP-001**: Fix unused import warnings in `callbacks.rs` (TrainingHistory)
2. **TASK-IMP-002**: Fix unused variables in test files (`counter`, `next_rand`, `perm_result`, `pdp_ratio`)
3. **TASK-IMP-003**: Add missing documentation for struct fields in `callbacks.rs` (31 warnings)
4. **TASK-IMP-004**: Address unused function `edge_case_matrix_strategy` in `properties.rs`
5. **TASK-IMP-005**: Run `cargo fix --lib -p ferroml-core --tests` to auto-fix clippy suggestions

**Success Criteria**:
- [ ] `cargo clippy -p ferroml-core --all-features -- -D warnings` — 0 warnings
- [ ] `cargo test -p ferroml-core` — all pass (no regressions)

---

## Execution Order

```
G.7 (code quality)        — 10 min, unblocks G.6
G.6 (CI hardening)        — 10 min, depends on G.7
G.1 (Naive Bayes)         — 20 min, independent
G.2 (Regularized CV)      — 20 min, independent
G.3 (Specialized models)  — 20 min, independent
G.4 (Linear SVM + Calib)  — 20 min, independent
G.5 (README fix)          — 10 min, depends on G.1-G.4
```

G.1-G.4 are fully independent and can run in parallel.

## Dependencies

- None — all models already implemented in Rust

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Some Rust models may have APIs incompatible with PyO3 | Check each struct's constructor pattern; adapt as needed |
| Making clippy strict may surface new warnings | Run clippy locally first; fix all warnings in G.7 |
| CV models may need special handling for cross-validation params | Follow existing patterns from ensemble.rs |
