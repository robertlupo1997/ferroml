# Simplify: PyO3 Binding Deduplication

**Date**: 2026-03-02
**Commit**: `05c2bb1`
**Scope**: `ferroml-python/src/ensemble.rs`, `ferroml-python/src/preprocessing.rs`, `ferroml-python/python/ferroml/preprocessing/__init__.py`
**Net change**: -66 lines (113 added, 179 removed)
**Tests**: 809 Python tests passing, 18 skipped (unchanged)

## What was done

### 1. Fixed 13 missing preprocessing re-exports (bug fix)

`preprocessing/__init__.py` only re-exported 9 of 22 registered Rust classes. Added: `TargetEncoder`, `KNNImputer`, `PowerTransformer`, `QuantileTransformer`, `PolynomialFeatures`, `KBinsDiscretizer`, `VarianceThreshold`, `SelectKBest`, `SelectFromModel`, `SMOTE`, `ADASYN`, `RandomUnderSampler`, `RandomOverSampler`.

### 2. Extracted `parse_sampling_strategy` (preprocessing.rs)

Replaced 4 near-identical functions (`apply_sampling_strategy`, `apply_sampling_strategy_adasyn`, `apply_sampling_strategy_undersampler`, `apply_sampling_strategy_oversampler`) with 1 shared parser returning `SamplingStrategy`. ~80 lines removed.

### 3. Extracted `parse_handle_unknown` (preprocessing.rs)

Deduplicated identical match blocks in `OneHotEncoder::new()` and `OrdinalEncoder::new()`.

### 4. Extracted `parse_penalty` (ensemble.rs)

Deduplicated identical match blocks in `PySGDClassifier::new()` and `PySGDRegressor::new()`.

### 5. Changed `String` fields to `&'static str` (ensemble.rs, preprocessing.rs)

`base_estimator_name` in `PyBaggingClassifier` and `PyBaggingRegressor`, and `estimator_name` in `PyRFE` — all only ever assigned string literals. Eliminated 30 `.to_string()` heap allocations.

### 6. Standardized error messages (ensemble.rs)

Changed `"Model not fitted."` to `"Model not fitted. Call fit() first."` in both Bagging `feature_importances_` getters.

## What was reviewed but not changed

| Pattern | Location | Why skipped |
|---------|----------|-------------|
| 31 near-identical explainability wrappers (KernelSHAP, permutation importance, PDP, ICE) | `explainability.rs` | PyO3 requires concrete types at `#[pyfunction]` boundary; shared helpers already extracted |
| fit/transform/fit_transform/inverse_transform boilerplate across 10+ transformer classes | `preprocessing.rs` | PyO3 `#[pymethods]` can't share via traits; macro refactor is a separate effort |
| 237x `.map_err(\|e\| PyErr::new::<PyRuntimeError, _>(e.to_string()))` | all binding files | Proper fix is `From<FerroError> for PyErr` but that couples ferroml-core to PyO3 |
| Eager submodule imports in root `__init__.py` | `__init__.py` | Lazy loading is a feature/behavior change, not a simplification |
| `ArrayView2` instead of `&Array2` in core API | `ferroml-core` | Cross-crate API change, out of scope |
| `BaggingConfig` struct for 7 repeated common params | `ensemble.rs` | Changes Python-facing API |

## Review methodology

Three parallel agents reviewed the `git diff 23247e0..HEAD` range (4 commits, ~6700 lines):
- **Code reuse agent**: searched for duplicate logic and existing utilities
- **Code quality agent**: checked for copy-paste, redundant state, stringly-typed code, parameter sprawl
- **Efficiency agent**: checked for unnecessary copies, missed concurrency, hot-path bloat

Files reviewed: `ensemble.rs`, `explainability.rs`, `preprocessing.rs`, 5 `__init__.py` wrappers, 4 test files, 2 CI workflow files. Adjacent files grepped for context (pickle.rs, core trait definitions, other binding files).
