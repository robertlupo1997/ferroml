# Handoff: Plan G — Python Bindings Completion + CI Hardening

**Date**: 2026-03-07
**Tasks**: TASK-G01 through TASK-G07 (all 7 complete)

## What Was Done

### G.1: Naive Bayes (3 models)
- **New file**: `ferroml-python/src/naive_bayes.rs` — PyGaussianNB, PyMultinomialNB, PyBernoulliNB
- **New file**: `ferroml-python/python/ferroml/naive_bayes/__init__.py`
- **New file**: `ferroml-python/tests/test_naive_bayes.py` — 30 tests
- **Edited**: `lib.rs`, `__init__.py`

### G.2: Regularized CV Models (4 models)
- **Edited**: `ferroml-python/src/linear.rs` — added PyRidgeCV, PyLassoCV, PyElasticNetCV, PyRidgeClassifier
- **Edited**: `ferroml-python/python/ferroml/linear/__init__.py`
- **New file**: `ferroml-python/tests/test_regularized_cv.py` — 36 tests

### G.3: Specialized Models (4 models)
- **Edited**: `ferroml-python/src/linear.rs` — added PyRobustRegression, PyQuantileRegression, PyPerceptron
- **Edited**: `ferroml-python/src/neighbors.rs` — added PyNearestCentroid
- **Edited**: `linear/__init__.py`, `neighbors/__init__.py`
- **New file**: `ferroml-python/tests/test_specialized_models.py` — 27 tests

### G.4: Linear SVM + Calibration (3 models)
- **New file**: `ferroml-python/src/svm.rs` — PyLinearSVC, PyLinearSVR
- **New file**: `ferroml-python/src/calibration.rs` — PyTemperatureScalingCalibrator
- **New dirs**: `ferroml-python/python/ferroml/svm/`, `ferroml-python/python/ferroml/calibration/`
- **New file**: `ferroml-python/tests/test_linear_svm.py` — 22 tests (covers SVM + calibration)
- **Edited**: `lib.rs`, `__init__.py`

### G.5: README Fix
- **Edited**: `ferroml-python/README.md` — synced model list with actual exports, added naive_bayes, svm, calibration, neural modules

### G.6: CI Hardening
- **Edited**: `.github/workflows/ci.yml`:
  - Enabled `RUSTFLAGS: -D warnings`
  - Made clippy strict: `-- -D warnings` (removed `|| true`)
  - Changed `cargo install cargo-tarpaulin` to use `--locked` (removed `|| true`)
- **Edited**: `.github/workflows/publish-pypi.yml`:
  - Made clippy strict (removed `|| true`)
  - Made pytest strict (removed `|| true`)

### G.7: Code Quality
- Verified: `cargo clippy -p ferroml-core --all-features --tests -- -D warnings` passes clean
- All TASK-IMP issues were already resolved by prior refactoring commits

## Bugs Fixed
- **RidgeCV NaN panic** (`regularized.rs:1529`): `partial_cmp().unwrap()` panicked when cv_scores contained NaN. Fixed to use `unwrap_or(Ordering::Less)` to treat NaN as worst score.

## Test Results
- **Python tests**: 946 passed, 18 skipped (up from 809)
- **137 new tests** across 4 test files
- **Rust tests**: All 2,471 passing

## Models Now Exposed to Python (total: 36+)
All 14 previously unexposed models are now available:
1. GaussianNB, MultinomialNB, BernoulliNB (naive_bayes)
2. RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier (linear)
3. RobustRegression, QuantileRegression, Perceptron (linear)
4. NearestCentroid (neighbors)
5. LinearSVC, LinearSVR (svm)
6. TemperatureScalingCalibrator (calibration)

## What's Next
- Plans H-L remain for v0.2.0 (GMM, IsolationForest, LOF, t-SNE, QDA, IsotonicRegression, testing)
