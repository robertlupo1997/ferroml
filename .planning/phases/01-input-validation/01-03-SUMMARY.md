---
phase: 01-input-validation
plan: 03
subsystem: python-bindings
tags: [pyo3, numpy, validation, nan, inf, valueerror]

# Dependency graph
requires:
  - phase: 01-input-validation/01-02
    provides: "Rust-core universal validation (all 55+ models)"
provides:
  - "Python-side check_array_finite/check_array1_finite in array_utils.rs"
  - "ValueError (not RuntimeError) for NaN/Inf in all Python bindings"
  - "151 parametrized Python validation tests"
affects: [02-robustness, 05-documentation]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Python-side NaN/Inf check before Rust conversion", "ValueError for input errors (sklearn convention)"]

key-files:
  created:
    - ferroml-python/tests/test_input_validation.py
  modified:
    - ferroml-python/src/array_utils.rs
    - ferroml-python/src/linear.rs
    - ferroml-python/src/svm.rs
    - ferroml-python/src/trees.rs
    - ferroml-python/src/clustering.rs
    - ferroml-python/src/decomposition.rs
    - ferroml-python/src/preprocessing.rs
    - ferroml-python/src/naive_bayes.rs
    - ferroml-python/src/neighbors.rs
    - ferroml-python/src/ensemble.rs
    - ferroml-python/src/gaussian_process.rs
    - ferroml-python/src/anomaly.rs
    - ferroml-python/src/multioutput.rs
    - ferroml-python/src/neural.rs
    - ferroml-python/tests/test_bindings_correctness.py

key-decisions:
  - "Defense-in-depth: Python validation adds ValueError layer, Rust validation remains as safety net"
  - "Exempt NaN-handling models from finite check: HistGBT, SimpleImputer, KNNImputer"
  - "Models with PyAny y-parameter get NaN check only on X (Rust handles y validation)"

patterns-established:
  - "check_array_finite(&x)?; before to_owned_array_2d(x) in every binding method"
  - "check_array1_finite(&y)?; before to_owned_array_1d(y) where y is PyReadonlyArray1"
  - "ValueError with position info for input validation errors"

requirements-completed: [VALID-10, VALID-03, VALID-07]

# Metrics
duration: 110min
completed: 2026-03-21
---

# Phase 01 Plan 03: Python Binding Validation Summary

**check_array_finite validation in all 14 Python binding files with 151 parametrized tests, raising ValueError with position info for NaN/Inf inputs**

## Performance

- **Duration:** 110 min
- **Started:** 2026-03-21T04:20:46Z
- **Completed:** 2026-03-21T06:10:46Z
- **Tasks:** 2
- **Files modified:** 16

## Accomplishments
- Added `check_array_finite` and `check_array1_finite` to `array_utils.rs` with position-aware error messages
- Inserted validation calls in all 14 Python binding files (464 total check calls)
- Created 151 parametrized Python tests covering NaN, Inf, -Inf, unfitted models, score, predict_proba, decision_function
- Zero regressions: all 2490 Python tests and 3550+ Rust tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Add check_array_finite to array_utils.rs and adopt in all Python bindings** - `e571ad3` (feat)
2. **Task 2: Create comprehensive Python input validation test suite** - `21e137a` (test)

## Files Created/Modified
- `ferroml-python/src/array_utils.rs` - Added check_array_finite and check_array1_finite helpers
- `ferroml-python/src/linear.rs` - Validation before all fit/predict/transform/score calls
- `ferroml-python/src/svm.rs` - Validation before all fit/predict/predict_proba/decision_function calls
- `ferroml-python/src/trees.rs` - Validation (exempting HistGBT which handles NaN natively)
- `ferroml-python/src/clustering.rs` - Validation before all fit/predict/fit_predict calls
- `ferroml-python/src/decomposition.rs` - Validation before all fit/transform calls
- `ferroml-python/src/preprocessing.rs` - Validation (exempting SimpleImputer/KNNImputer)
- `ferroml-python/src/naive_bayes.rs` - Validation before all fit/predict calls
- `ferroml-python/src/neighbors.rs` - Validation before all fit/predict calls
- `ferroml-python/src/ensemble.rs` - Validation before all fit/predict calls
- `ferroml-python/src/gaussian_process.rs` - Validation before all fit/predict calls
- `ferroml-python/src/anomaly.rs` - Validation before all fit/predict calls
- `ferroml-python/src/multioutput.rs` - Validation before all fit/predict calls
- `ferroml-python/src/neural.rs` - Validation before all fit/predict/predict_proba calls
- `ferroml-python/tests/test_input_validation.py` - 151 parametrized validation tests
- `ferroml-python/tests/test_bindings_correctness.py` - Updated to expect ValueError

## Decisions Made
- Defense-in-depth: Python validation adds ValueError layer on top of Rust-side validation (not replacing it)
- Exempted HistGradientBoosting models from finite check (they natively handle NaN as missing values)
- Exempted SimpleImputer and KNNImputer from finite check (designed to process NaN)
- Models with `y: &Bound<PyAny>` (KNeighborsClassifier/Regressor, SVC, etc.) get NaN check only on X; Rust core handles y validation
- Used ValueError with "(row, col)" position info to match sklearn conventions

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_bindings_correctness.py expected exception type**
- **Found during:** Task 2 (test verification)
- **Issue:** Existing tests expected RuntimeError for NaN input, but we now raise ValueError
- **Fix:** Updated 3 test assertions from RuntimeError to ValueError
- **Files modified:** ferroml-python/tests/test_bindings_correctness.py
- **Verification:** All 2490 Python tests pass
- **Committed in:** d4b921e

**2. [Rule 1 - Bug] Exempted NaN-handling models from finite check**
- **Found during:** Task 2 (regression testing)
- **Issue:** SimpleImputer, KNNImputer, and HistGBT models are designed to handle NaN; the finite check blocked legitimate use
- **Fix:** Removed check_array_finite calls from SimpleImputer (3 methods), KNNImputer (3 methods), HistGBT Classifier (7 methods), HistGBT Regressor (6 methods)
- **Files modified:** ferroml-python/src/preprocessing.rs, ferroml-python/src/trees.rs
- **Verification:** test_handles_missing_values and test_comparison_preprocessing pass
- **Committed in:** de686a5

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for correctness. NaN-handling models must accept NaN by design.

## Issues Encountered
- neural.rs used `x_owned`/`y_owned` variable names instead of `x_arr`/`y_arr` -- initial script missed these, fixed with targeted second pass
- 3 unused import warnings for `check_array1_finite` in files without 1D y parameters (anomaly, clustering, multioutput) -- removed unused imports

## Next Phase Readiness
- Phase 01 (Input Validation) is complete with all 3 plans executed
- All models have both Rust-core validation (01-01/01-02) and Python-binding validation (01-03)
- Ready for Phase 02 (Robustness)

---
*Phase: 01-input-validation*
*Completed: 2026-03-21*
