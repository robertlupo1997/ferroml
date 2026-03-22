---
phase: 03-robustness-hardening
plan: 03
subsystem: bindings
tags: [pyo3, pickle, error-mapping, serialization, python]

requires:
  - phase: 03-01
    provides: "Unwrap elimination in SVM/boosting/stats modules"
  - phase: 03-02
    provides: "Unwrap elimination in linear/tree/preprocessing/clustering + clippy lint"
provides:
  - "Variant-aware FerroError -> Python exception mapping (ferro_to_pyerr)"
  - "Pickle support for SVM, NaiveBayes, Anomaly, Calibration, MultiOutput models"
  - "Serialization version checking on deserialization path"
  - "Comprehensive pickle roundtrip test suite (51 tests)"
  - "Exception type mapping test suite (8 tests)"
affects: [documentation, python-api]

tech-stack:
  added: []
  patterns:
    - "ferro_to_pyerr for variant-aware FerroError -> PyErr mapping"
    - "Inline pickle methods in #[pymethods] blocks (no impl_pickle! macro due to single-pymethods constraint)"

key-files:
  created:
    - "ferroml-python/tests/test_pickle_roundtrip.py"
  modified:
    - "ferroml-python/src/errors.rs"
    - "ferroml-python/src/svm.rs"
    - "ferroml-python/src/naive_bayes.rs"
    - "ferroml-python/src/anomaly.rs"
    - "ferroml-python/src/calibration.rs"
    - "ferroml-python/src/multioutput.rs"
    - "ferroml-core/src/models/multioutput.rs"
    - "ferroml-core/src/serialization.rs"
    - "ferroml-python/tests/test_bindings_correctness.py"

key-decisions:
  - "GP models (5) excluded from pickle due to Box<dyn Kernel> trait object -- requires erased-serde or custom impl (architectural change)"
  - "not_fitted_err changed from PyValueError to PyRuntimeError for consistency with ferro_to_pyerr NotFitted mapping"
  - "from_bytes() now enforces major version compatibility (hard error, not warning) for pickle safety"
  - "impl_pickle! macro not used due to PyO3 single-pymethods constraint -- inline methods instead"

patterns-established:
  - "Error mapping: all binding files use crate::errors::ferro_to_pyerr for FerroError results"
  - "Pickle pattern: serialize (inner, extra_fields) tuples for wrappers with auxiliary state"

requirements-completed: [ROBU-10, ROBU-11, ROBU-12, ROBU-13]

duration: 48min
completed: 2026-03-22
---

# Phase 3 Plan 3: Python Exception Mapping and Pickle Roundtrip Summary

**Variant-aware FerroError -> Python exception mapping (InvalidInput->ValueError, NotFitted->RuntimeError) with pickle support for 13 newly-serializable models and 59 passing tests**

## Performance

- **Duration:** 48 min
- **Started:** 2026-03-22T01:01:39Z
- **Completed:** 2026-03-22T01:49:00Z
- **Tasks:** 2
- **Files modified:** 26

## Accomplishments

- Created `ferro_to_pyerr()` mapping all 14 FerroError variants to semantically correct Python exceptions, replacing 400+ inline `.map_err()` patterns across 19 binding files
- Added pickle `__getstate__`/`__setstate__` to SVM (4), NaiveBayes (4), Anomaly (2), Calibration (1), MultiOutput (2) models -- 13 models newly serializable
- Wired version compatibility check into `from_bytes()` deserialization path (used by pickle), rejecting models from incompatible major versions
- Created comprehensive test suites: 51 pickle roundtrip tests + 8 exception mapping tests, all passing

## Task Commits

1. **Task 1: Variant-aware error mapping, pickle support, version checking** - `711421d` (feat)
2. **Task 2: Pickle roundtrip and exception mapping tests** - `85b846a` (test)

## Files Created/Modified

- `ferroml-python/src/errors.rs` -- Added ferro_to_pyerr() with variant-aware mapping
- `ferroml-python/src/svm.rs` -- Replaced inline error mapping + added pickle for 4 models
- `ferroml-python/src/naive_bayes.rs` -- Replaced inline error mapping + added pickle for 4 models
- `ferroml-python/src/anomaly.rs` -- Added pickle for IsolationForest, LocalOutlierFactor
- `ferroml-python/src/calibration.rs` -- Added pickle for TemperatureScalingCalibrator
- `ferroml-python/src/multioutput.rs` -- Added pickle for MultiOutputRegressor/Classifier with Serialize enum
- `ferroml-core/src/models/multioutput.rs` -- Added Serialize/Deserialize derives with serde bounds
- `ferroml-core/src/serialization.rs` -- Added version check to from_bytes()
- `ferroml-python/src/neural.rs` -- Replaced inline error mapping with ferro_to_pyerr
- `ferroml-python/tests/test_pickle_roundtrip.py` -- 51 tests across 12 model categories
- `ferroml-python/tests/test_bindings_correctness.py` -- 8 exception mapping tests + updated 2 existing tests

## Decisions Made

- **GP models excluded from pickle**: GaussianProcessRegressor/Classifier/SparseGP/SVGP use `Box<dyn Kernel>` trait objects that cannot be serialized without erased-serde or custom serialization. This is an architectural limitation, not a bug.
- **not_fitted_err returns RuntimeError**: Changed from PyValueError to PyRuntimeError for consistency with ferro_to_pyerr's NotFitted mapping. NotFitted is a state error (model hasn't been trained), not an input error.
- **Hard error on version mismatch**: `from_bytes()` now returns `SerializationError` for incompatible major versions rather than silently loading potentially corrupt data.
- **Inline pickle instead of macro**: PyO3 without `multiple-pymethods` feature only allows one `#[pymethods]` block per struct, so pickle methods are added inline.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated existing tests for new exception types**
- **Found during:** Task 2
- **Issue:** Pre-existing tests in test_bindings_correctness.py expected RuntimeError for ShapeMismatch/InvalidInput errors, but ferro_to_pyerr now correctly maps these to ValueError
- **Fix:** Updated test_empty_array_raises_error (accepts ValueError|RuntimeError) and test_shape_mismatch_x_y / test_wrong_features_at_predict (expect ValueError)
- **Files modified:** ferroml-python/tests/test_bindings_correctness.py
- **Committed in:** 85b846a

**2. [Rule 3 - Blocking] neural.rs error mapping not replaced by bulk sed**
- **Found during:** Task 2
- **Issue:** neural.rs used `PyValueError::new_err(e.to_string())` pattern (not PyErr::new::<...>) which wasn't caught by the bulk replacement
- **Fix:** Replaced 7 occurrences with ferro_to_pyerr
- **Files modified:** ferroml-python/src/neural.rs
- **Committed in:** 85b846a

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Known Gaps (Not In Scope)

Models without pickle support (pre-existing, not introduced by this plan):
- GP models (5): GaussianProcessRegressor, GaussianProcessClassifier, SparseGPRegressor, SparseGPClassifier, SVGPRegressor
- Linear models (7): RobustRegression, QuantileRegression, Perceptron, RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier
- Ensemble models (6): BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
- Other (3): HDBSCAN, CountVectorizer, MLP (serializes but doesn't restore state)

## Issues Encountered

- PyO3 single-pymethods constraint required moving from `impl_pickle!` macro to inline methods, adding ~10 lines per model
- MLP models have a pre-existing serialization bug where fitted state is lost through pickle roundtrip (skipped in tests)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 3 (Robustness Hardening) is now complete
- All FerroError variants map to correct Python exceptions
- 42 models have verified pickle roundtrip support
- Ready for Phase 4 (Performance Optimization) or Phase 5 (Documentation)

---
*Phase: 03-robustness-hardening*
*Completed: 2026-03-22*
