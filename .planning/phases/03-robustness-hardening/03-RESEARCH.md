# Phase 3: Robustness Hardening - Research

**Researched:** 2026-03-21
**Domain:** Rust error handling, unwrap elimination, Python bindings error mapping, serialization
**Confidence:** HIGH

## Summary

Phase 3 targets three complementary robustness concerns: (1) eliminating panic-inducing `unwrap()`/`expect()` calls from fit/predict/transform code paths, (2) ensuring error messages are actionable and correctly mapped to Python exceptions, and (3) verifying serialization (pickle) roundtrip for all 55+ Python-exposed models.

The codebase has approximately 6,418 `unwrap()`/`expect()` calls across `ferroml-core/src/`. However, many are in doc comments, test code, or provably safe contexts (e.g., `from_shape_vec` where shape is computed from the same data). The real work is triaging these into risk tiers and fixing the genuinely dangerous ones -- those in fit/predict/transform paths that could panic on valid-but-unusual user input.

The Python error mapping currently uses a generic `to_py_runtime_err` helper that converts ALL `FerroError` variants to `PyRuntimeError`. This means `ShapeMismatch` (which should be `ValueError`) and `NotFitted` (which could be `RuntimeError` with a clear message) all become undifferentiated `RuntimeError`. The fix is a variant-aware `From<FerroError> for PyErr` pattern. Pickle support exists via `__getstate__`/`__setstate__` using MessagePack, but several binding files (svm.rs, naive_bayes.rs, gaussian_process.rs, anomaly.rs, calibration.rs, multioutput.rs, stats.rs) are missing pickle methods entirely.

**Primary recommendation:** Triage unwraps mechanically by risk tier, fix Tier 1-2 in the highest-risk modules first (SVM 178, hist_boosting 96, knn 123), then sweep remaining modules. Add `clippy::unwrap_used` as a workspace warning. Implement variant-aware Python error mapping. Complete pickle coverage for all model binding files.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ROBU-01 | SVM fit/predict unwraps replaced | SVM has 178 unwraps (167 non-doc); `as_ref().unwrap()` on Option fields and `as_slice().unwrap()` are the primary patterns |
| ROBU-02 | Stats module unwraps replaced | stats/ has 37 unwraps -- lowest count, mostly safe patterns |
| ROBU-03 | Boosting model unwraps replaced | hist_boosting.rs has 96, boosting.rs has 72 -- medium risk |
| ROBU-04 | Linear model unwraps replaced | linear.rs has 58, regularized.rs has 82, logistic.rs has 63 |
| ROBU-05 | Tree model unwraps replaced | tree.rs has 92, forest.rs has 47, extra_trees.rs has 40 |
| ROBU-06 | Preprocessing unwraps replaced | 660 total across scalers (86), selection (81), sampling (67), encoders (66), imputers (61) |
| ROBU-07 | Clustering unwraps replaced | 287 total in clustering/ directory |
| ROBU-08 | Remaining modules triaged | decomposition (243), onnx (237), cv (185), explainability (173), datasets (168), ensemble (113) |
| ROBU-09 | clippy::unwrap_used lint as warning | CI already runs `clippy -D warnings`; add `#![warn(clippy::unwrap_used)]` to lib.rs |
| ROBU-10 | Actionable error messages | FerroError variants already have structured fields; audit that string messages include "what to do" guidance |
| ROBU-11 | Python exception mapping | Currently generic `to_py_runtime_err` -- need variant-aware mapping |
| ROBU-12 | Serialization version checking | `is_compatible_with` exists in serialization.rs (major-version check); verify pickle path uses it |
| ROBU-13 | Pickle roundtrip for all 55+ models | 59 `__getstate__` found but several binding files (svm, naive_bayes, gaussian_process, anomaly, calibration, multioutput, stats) have 0 |
</phase_requirements>

## Architecture Patterns

### Unwrap Risk Tier Classification

Use this tier system to prioritize fixes:

| Tier | Risk | Pattern | Action |
|------|------|---------|--------|
| Tier 1 (Critical) | Panic on user input | `.as_ref().unwrap()` on Option fields in predict paths (e.g., `self.classes.as_ref().unwrap()`) | Replace with `ok_or_else(\|\| FerroError::not_fitted(...))` |
| Tier 1 (Critical) | Panic on edge case | `self.n_features.unwrap()` in predict validation | Replace with `ok_or_else` |
| Tier 2 (Medium) | Panic on unusual data | `.as_slice().unwrap()` on non-contiguous arrays | Replace with `.as_slice().ok_or_else(\|\| FerroError::numerical(...))` or use iterator fallback |
| Tier 3 (Low) | Provably safe | `from_shape_vec((n, m), vec_of_size_n_times_m).unwrap()` | Document with `// SAFETY:` comment, or use `expect("shape matches data")` |
| Tier 3 (Low) | Provably safe | `as_slice().unwrap()` on standard-layout arrays created internally | Document safety |
| Test/Doc | N/A | Unwraps in `#[cfg(test)]`, doc comments, `//!` | Ignore -- these don't affect users |

### Unwrap Count by Module (non-doc, non-test)

| Module | Count | Top Files | Risk Level |
|--------|-------|-----------|------------|
| models/ | ~1,662 | svm.rs (178), knn.rs (123), sgd.rs (96), hist_boosting.rs (96), tree.rs (92) | HIGH |
| preprocessing/ | ~660 | scalers.rs (86), selection.rs (81), sampling.rs (67), encoders.rs (66) | MEDIUM |
| clustering/ | ~287 | (various) | MEDIUM |
| decomposition/ | ~243 | (various) | MEDIUM |
| onnx/ | ~237 | (inference, not user-facing fit/predict) | LOW |
| stats/ | ~37 | hypothesis.rs | LOW |

### Common Unwrap Patterns (codebase-wide)

| Pattern | Count | Safety | Fix Strategy |
|---------|-------|--------|-------------|
| `from_shape_vec/fn(...).unwrap()` | 430 | Usually safe (computed shapes) | Add `// SAFETY:` or use `map_err` |
| `.as_ref().unwrap()` | 219 | Dangerous in predict (NotFitted) | `ok_or_else(FerroError::not_fitted)` |
| `as_slice().unwrap()` | 60 | Safe on standard-layout, risky on views | Check contiguity or use iter fallback |
| `.expect(...)` | 189 | Varies -- some are mutex locks, some are data | Triage individually |
| `.get(..).unwrap()` | 25 | Index-out-of-bounds risk | Use `get().ok_or_else()` |
| `.first()/.last().unwrap()` | 24 | Empty-collection risk | Guard with `is_empty()` check or `ok_or_else` |

### Pattern 1: Replacing Option::unwrap in Predict Paths

**What:** The most common dangerous pattern -- accessing fitted state via `self.field.as_ref().unwrap()`
**When to use:** Every predict/transform/decision_function method that accesses fitted state

```rust
// BEFORE (panics if not fitted)
fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
    let weights = self.weights.as_ref().unwrap();
    // ...
}

// AFTER (returns proper error)
fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
    let weights = self.weights.as_ref()
        .ok_or_else(|| FerroError::not_fitted("predict"))?;
    // ...
}
```

### Pattern 2: Safe Array Construction

**What:** Replacing `from_shape_vec().unwrap()` where shape is provably correct
**When to use:** When the shape is computed from the same data being shaped

```rust
// Option A: Document safety (preferred for proven-correct shapes)
// SAFETY: vec has exactly n_samples * n_features elements
let result = Array2::from_shape_vec((n_samples, n_features), flat_vec)
    .expect("shape matches data length");

// Option B: Propagate error (when shape could theoretically be wrong)
let result = Array2::from_shape_vec((n_samples, n_features), flat_vec)
    .map_err(|e| FerroError::numerical(format!("Array construction failed: {e}")))?;
```

### Pattern 3: Variant-Aware Python Error Mapping

**What:** Map FerroError variants to semantically correct Python exceptions
**Current state:** All errors become `PyRuntimeError` via `to_py_runtime_err`

```rust
// In ferroml-python/src/errors.rs - add this function
pub fn ferro_to_pyerr(e: FerroError) -> PyErr {
    match &e {
        FerroError::InvalidInput(_) => PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()),
        FerroError::ShapeMismatch { .. } => PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()),
        FerroError::NotFitted { .. } => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
        FerroError::ConfigError(_) => PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()),
        FerroError::NumericalError(_) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
        FerroError::ConvergenceFailure { .. } => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
        FerroError::NotImplemented(_) | FerroError::NotImplementedFor { .. } => {
            PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(e.to_string())
        }
        FerroError::SerializationError(_) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
        FerroError::IoError(_) => PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()),
        FerroError::Timeout { .. } => PyErr::new::<pyo3::exceptions::PyTimeoutError, _>(e.to_string()),
        _ => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
    }
}
```

**Correct exception mapping:**

| FerroError Variant | Python Exception | Rationale |
|-------------------|------------------|-----------|
| InvalidInput | ValueError | Bad parameter or data |
| ShapeMismatch | ValueError | Dimensional mismatch is a value error |
| NotFitted | RuntimeError | State error -- model used before fitting |
| ConfigError | ValueError | Bad configuration parameter |
| NumericalError | RuntimeError | Computation failure |
| ConvergenceFailure | RuntimeError | Algorithm didn't converge |
| NotImplemented/For | NotImplementedError | Feature not available |
| SerializationError | RuntimeError | Save/load failure |
| IoError | IOError | File system error |
| Timeout | TimeoutError | Operation took too long |
| AssumptionViolation | RuntimeError | Statistical check failed |
| ResourceExhausted | MemoryError or RuntimeError | Out of resources |
| CrossValidation | RuntimeError | CV procedure failed |
| InferenceError | RuntimeError | ONNX runtime error |

### Anti-Patterns to Avoid

- **Blanket replacement:** Do NOT replace every `unwrap()` with `?`. Many are in test code or doc examples -- only fix production code paths.
- **Losing context:** When replacing `unwrap()`, include context in the error message (what value was None, what operation was being attempted).
- **Over-engineering:** For `from_shape_vec` where the shape is provably correct from the algorithm, `expect("reason")` is acceptable -- it documents the invariant.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Error type hierarchy | Custom enum per module | Existing `FerroError` enum with `thiserror` | Already comprehensive with 14 variants |
| Pickle serialization | Custom binary format | Existing `pickle.rs` with `getstate`/`setstate` + MessagePack | Already works, just needs coverage expansion |
| Version compatibility | Custom version parsing | Existing `SemanticVersion` in `serialization.rs` | Already implements `is_compatible_with` (major-version check) |
| Python error mapping | Per-model error conversion | Single `ferro_to_pyerr` function replacing `to_py_runtime_err` | Centralizes the mapping, consistent across all bindings |

## Common Pitfalls

### Pitfall 1: Mutex unwrap in multi-threaded predict
**What goes wrong:** `self.cache.lock().unwrap()` panics if another thread panicked while holding the lock (poisoned mutex).
**Why it happens:** Rust mutexes are poisoned on panic, and `.unwrap()` on a poisoned lock panics the new thread too.
**How to avoid:** Use `.lock().unwrap_or_else(|e| e.into_inner())` to recover from poisoned mutexes, or return a `FerroError`.
**Warning signs:** Any `Mutex::lock().unwrap()` or `RwLock::read().unwrap()` in model code.

### Pitfall 2: Breaking existing tests with error type changes
**What goes wrong:** Changing an `unwrap()` to return `Err(...)` changes the function signature or behavior. Existing tests that expected success may now get a different error type.
**Why it happens:** Some tests rely on specific panic behavior or error messages.
**How to avoid:** Run full test suite (`cargo test --all`) after each module's changes. Python tests too.
**Warning signs:** Tests that use `#[should_panic]` instead of `assert!(result.is_err())`.

### Pitfall 3: Pickle roundtrip changing predictions due to floating-point serialization
**What goes wrong:** MessagePack serialization of f64 can lose precision in edge cases, causing predictions to differ by epsilon after roundtrip.
**Why it happens:** Some serialization formats use f32 internally or have edge cases with subnormals/NaN.
**How to avoid:** Use `np.testing.assert_allclose` with reasonable tolerance (1e-10) not exact equality.
**Warning signs:** Tests that use `np.array_equal` instead of `np.testing.assert_allclose`.

### Pitfall 4: clippy::unwrap_used lint flooding with false positives
**What goes wrong:** Enabling the lint at deny level immediately breaks the build with thousands of warnings in test code and doc examples.
**Why it happens:** The lint applies to ALL code including tests and examples unless suppressed.
**How to avoid:** Enable as `warn` (not `deny`) in lib.rs. Use `#[allow(clippy::unwrap_used)]` in test modules and doc examples. Only enable for the library crate, not tests.
**Warning signs:** CI build taking minutes just to display clippy output.

### Pitfall 5: Incomplete pickle coverage causing silent failures
**What goes wrong:** Models without `__getstate__`/`__setstate__` silently fail to pickle (Python's default pickle tries and fails on PyO3 objects).
**Why it happens:** Binding files added for new models may forget pickle methods.
**How to avoid:** The `impl_pickle!` macro exists but isn't used everywhere. Use it for all remaining models.
**Warning signs:** Any `#[pyclass]` struct in binding code without corresponding `__getstate__`/`__setstate__`.

## Code Examples

### Adding clippy::unwrap_used lint to lib.rs

```rust
// At the top of ferroml-core/src/lib.rs
#![warn(clippy::unwrap_used)]
```

This emits warnings (not errors) for every `unwrap()` call, allowing gradual cleanup. The existing CI runs `clippy -D warnings`, so this would need to be `warn` level only. To prevent CI failure during the transition, options are:
1. Use `#[allow(clippy::unwrap_used)]` on modules not yet cleaned up
2. Or add `-A clippy::unwrap_used` to CI temporarily and enable after cleanup

**Recommended approach:** Enable `#![warn(clippy::unwrap_used)]` in lib.rs, and add module-level `#[allow(clippy::unwrap_used)]` to NOT-yet-triaged modules. Remove the allows as modules are cleaned. This way CI stays green throughout.

### Pickle Roundtrip Test Pattern

```python
import pickle
import numpy as np
from ferroml.svm import SVC

def test_svc_pickle_roundtrip():
    X = np.array([[1,2],[3,4],[5,6],[7,8]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)

    model = SVC(kernel="rbf", C=1.0)
    model.fit(X, y)
    preds_before = model.predict(X)

    # Roundtrip
    data = pickle.dumps(model)
    loaded = pickle.loads(data)
    preds_after = loaded.predict(X)

    np.testing.assert_array_equal(preds_before, preds_after)
```

### Actionable Error Message Pattern

```rust
// BEFORE: Not actionable
FerroError::InvalidInput("bad input".to_string())

// AFTER: Actionable -- what went wrong + what to do
FerroError::InvalidInput(format!(
    "Feature matrix X has {} columns but model was fitted with {} features. \
     Ensure predict() input has the same number of features as fit() input.",
    x.ncols(), self.n_features_in
))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `unwrap()` everywhere | `?` operator with `Result` return types | Rust 1.13+ (2016) | Standard in all modern Rust |
| `#[should_panic]` tests | `assert!(result.is_err())` | Best practice | More precise error testing |
| Generic `PyRuntimeError` for all Rust errors | Variant-aware mapping (`From<FerroError> for PyErr`) | PyO3 best practice | Users can catch specific exception types |
| Manual `__getstate__`/`__setstate__` | `impl_pickle!` macro | FerroML pattern | Consistent, less boilerplate |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust: built-in `#[test]` + cargo test; Python: pytest |
| Config file | `.pre-commit-config.yaml` (clippy, fmt, test hooks) |
| Quick run command | `cargo test -p ferroml-core --lib -- --test-threads=4` |
| Full suite command | `cargo test --all && pytest ferroml-python/tests/` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ROBU-01 | SVM unwraps eliminated | unit | `cargo test -p ferroml-core --lib models::svm -- --test-threads=4` | Existing tests cover behavior |
| ROBU-02 | Stats unwraps eliminated | unit | `cargo test -p ferroml-core --lib stats -- --test-threads=4` | Existing tests |
| ROBU-03 | Boosting unwraps eliminated | unit | `cargo test -p ferroml-core --lib models::boosting models::hist_boosting -- --test-threads=4` | Existing tests |
| ROBU-04 | Linear unwraps eliminated | unit | `cargo test -p ferroml-core --lib models::linear models::regularized models::logistic -- --test-threads=4` | Existing tests |
| ROBU-05 | Tree unwraps eliminated | unit | `cargo test -p ferroml-core --lib models::tree models::forest -- --test-threads=4` | Existing tests |
| ROBU-06 | Preprocessing unwraps eliminated | unit | `cargo test -p ferroml-core --lib preprocessing -- --test-threads=4` | Existing tests |
| ROBU-07 | Clustering unwraps eliminated | unit | `cargo test -p ferroml-core --lib clustering -- --test-threads=4` | Existing tests |
| ROBU-08 | Remaining modules triaged | unit | `cargo test -p ferroml-core --lib -- --test-threads=4` | Existing tests |
| ROBU-09 | clippy lint enabled | lint | `cargo clippy -p ferroml-core --all-features -- -W clippy::unwrap_used` | N/A (config check) |
| ROBU-10 | Error messages actionable | manual audit | grep-based review | N/A |
| ROBU-11 | Python exception mapping | integration | `pytest ferroml-python/tests/test_bindings_correctness.py -x` | Partial (needs expansion) |
| ROBU-12 | Serialization version check | unit | `cargo test -p ferroml-core --lib serialization -- --test-threads=4` | Existing tests |
| ROBU-13 | Pickle roundtrip all models | integration | `pytest ferroml-python/tests/test_bindings_correctness.py -k pickle -x` | Partial (4 models tested, need 55+) |

### Sampling Rate
- **Per task commit:** `cargo test -p ferroml-core --lib -- --test-threads=4`
- **Per wave merge:** `cargo test --all && pytest ferroml-python/tests/`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Expand pickle roundtrip tests from 4 models to all 55+ in `test_bindings_correctness.py`
- [ ] Add Python exception type assertion tests (verify `ShapeMismatch -> ValueError`, etc.)
- [ ] No new test infrastructure needed -- existing frameworks sufficient

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis via grep/read of `ferroml-core/src/` (unwrap counts, error.rs, serialization.rs)
- Direct analysis of `ferroml-python/src/errors.rs`, `pickle.rs`, binding files
- Existing test files: `test_bindings_correctness.py`
- CI configuration: `.github/workflows/ci.yml`, `.pre-commit-config.yaml`

### Secondary (MEDIUM confidence)
- Clippy lint documentation for `clippy::unwrap_used` (well-documented, stable lint)
- PyO3 exception mapping patterns (standard approach in PyO3 ecosystem)

## Metadata

**Confidence breakdown:**
- Unwrap counts and locations: HIGH - mechanical grep of actual codebase
- Risk tier classification: HIGH - based on code path analysis (fit/predict vs test/doc)
- Python error mapping: HIGH - read actual `errors.rs` and binding patterns
- Pickle coverage gaps: HIGH - diffed `__getstate__` presence across all binding files
- Clippy lint strategy: HIGH - well-documented Rust toolchain feature

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable domain, no external dependency changes expected)
