# Phase 1: Input Validation - Research

**Researched:** 2026-03-20
**Domain:** Input validation infrastructure for Rust ML library with PyO3 Python bindings
**Confidence:** HIGH

## Summary

Phase 1 adds systematic input validation to all 55+ models, 22+ preprocessors, and clustering/decomposition modules. The good news: substantial infrastructure already exists. The `validate_fit_input` function in `ferroml-core/src/models/mod.rs` (line 1235) already checks NaN/Inf in X, NaN/Inf in y, empty input, and X-y row mismatch. It is already called by ~20 model files covering most supervised models. The `validate_predict_input` function (line 1300) checks feature count mismatch, empty input, and NaN/Inf at predict time. The `check_is_fitted` helper (line 1227) provides a standard NotFitted guard.

The primary gaps are: (1) clustering modules (KMeans, DBSCAN, GMM, HDBSCAN, Agglomerative) do inline validation instead of using shared functions -- needs refactoring per VALID-09; (2) decomposition modules (PCA, t-SNE, LDA, TruncatedSVD, FactorAnalysis) have no NaN/Inf validation at all; (3) several supervised models (GaussianProcess, IsolationForest, IsotonicRegression, LOF, QDA, MultiOutput) don't call `validate_fit_input`; (4) hyperparameter validation at construction time uses silent clamping (e.g., `SVC::with_c` clamps to 1e-10) instead of returning errors per VALID-08; (5) the Python binding layer (`ferroml-python/src/`) has zero NumPy array validation -- all validation happens in Rust after conversion; (6) the existing not_fitted test suite covers only ~25 of 55+ models.

**Primary recommendation:** Build shared validation functions (for unsupervised/transformer fit, predict, and transform), systematically adopt them across all modules, convert builder methods to return `Result` or validate in `fit()`, and add Python-side array validation in `array_utils.rs`.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| VALID-01 | All models reject NaN/Inf in X at fit time | `validate_fit_input` exists, needs adoption in ~12 missing modules (clustering, decomposition, GP, IF, Isotonic, LOF, QDA, MultiOutput) |
| VALID-02 | All models reject NaN/Inf in y at fit time | Same `validate_fit_input` covers y; unsupervised models need X-only variant |
| VALID-03 | All models reject NaN/Inf in X at predict time | `validate_predict_input` exists, needs audit for adoption across all predict paths |
| VALID-04 | All models return clean FerroError on empty dataset (n=0) | Empty check exists in `validate_fit_input`; clustering already handles inline; needs audit |
| VALID-05 | All models handle single-sample (n=1) gracefully | Edge case tests exist for some models via macro; needs universal coverage |
| VALID-06 | All models validate n_features_in_ at predict time | `validate_predict_input` checks this; needs adoption audit |
| VALID-07 | All models enforce NotFitted guard | `check_is_fitted` helper exists; ~25 of 55+ models tested; needs completeness audit |
| VALID-08 | Hyperparameters validated at construction with actionable messages | Currently uses silent clamping (e.g., `c.max(1e-10)`); needs conversion to error-returning validation |
| VALID-09 | NaN/Inf validation as single shared function | `validate_fit_input` exists for supervised; need `validate_unsupervised_input` and `validate_transform_input` variants; clustering modules need refactoring from inline to shared |
| VALID-10 | Python binding layer validates NumPy arrays before Rust | No validation exists in `ferroml-python/src/array_utils.rs`; need to add `check_finite` before array conversion |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ndarray | (existing) | Array types Array2/Array1 | Already used throughout; validation iterates over these |
| thiserror | (existing) | FerroError derive | Already used for error enum |
| PyO3 | (existing) | Python bindings | Already used; need numpy array validation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy (PyO3) | (existing) | NumPy array access from Python | For VALID-10 Python-side validation |

### Alternatives Considered
None -- this phase uses existing infrastructure exclusively. No new dependencies needed.

**Installation:**
No new dependencies required.

## Architecture Patterns

### Existing Validation Function Locations
```
ferroml-core/src/
  models/mod.rs          # validate_fit_input, validate_predict_input, check_is_fitted (lines 1227-1316)
  preprocessing/mod.rs   # check_finite (line 374) -- X-only, no y
  clustering/            # Inline validation per model (KMeans, DBSCAN, etc.)
  decomposition/         # NO validation currently
  neural/                # NO validation currently

ferroml-python/src/
  errors.rs              # to_py_runtime_err, not_fitted_err helpers
  array_utils.rs         # Array conversion only, NO validation
```

### Pattern 1: Shared Validation Functions (VALID-09)
**What:** Centralize all NaN/Inf/empty checks into reusable functions
**When to use:** Every fit(), predict(), transform() entry point
**Example:**
```rust
// Already exists in models/mod.rs:
pub fn validate_fit_input(x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(FerroError::shape_mismatch(...));
    }
    if x.is_empty() || y.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input("X contains NaN or infinite values"));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input("y contains NaN or infinite values"));
    }
    Ok(())
}

// NEEDED: Unsupervised variant (no y parameter)
pub fn validate_unsupervised_input(x: &Array2<f64>) -> Result<()> {
    if x.is_empty() || x.nrows() == 0 {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input("X contains NaN or infinite values"));
    }
    Ok(())
}

// NEEDED: Transform input validator
pub fn validate_transform_input(x: &Array2<f64>, expected_features: usize) -> Result<()> {
    if x.ncols() != expected_features {
        return Err(FerroError::shape_mismatch(...));
    }
    if x.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input("X contains NaN or infinite values"));
    }
    Ok(())
}
```

### Pattern 2: Hyperparameter Validation at Construction (VALID-08)
**What:** Validate hyperparameters in builder methods or at fit-time, raise errors instead of silently clamping
**When to use:** Every `with_*` builder method and `new()` constructor
**Current problematic pattern:**
```rust
// BAD: Silent clamping in SVC::with_c (svm.rs:1266)
pub fn with_c(mut self, c: f64) -> Self {
    self.c = c.max(1e-10);  // Silently changes negative to 1e-10
    self
}
```
**Recommended pattern:**
```rust
// OPTION A: Validate in fit() -- simpler, preserves builder ergonomics
fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if self.c <= 0.0 {
        return Err(FerroError::invalid_input(
            "Parameter C must be positive, got {}", self.c
        ));
    }
    validate_fit_input(x, y)?;
    // ... rest of fit
}

// OPTION B: Validate in builder -- changes API signature, more disruptive
pub fn with_c(mut self, c: f64) -> Result<Self, FerroError> {
    if c <= 0.0 {
        return Err(FerroError::invalid_input(...));
    }
    self.c = c;
    Ok(self)
}
```
**Recommendation:** Use Option A (validate in fit). Changing builder methods to return Result would break the entire builder chain pattern (`Model::new().with_a(1).with_b(2)`) used universally. Validate at fit-time instead, which catches invalid params before computation starts.

### Pattern 3: Python-Side Array Validation (VALID-10)
**What:** Check NumPy arrays for NaN/Inf before converting to Rust
**When to use:** In every PyO3 fit/predict/transform binding
**Example:**
```rust
// Add to ferroml-python/src/array_utils.rs:
pub fn check_array_finite(x: &numpy::PyReadonlyArray2<f64>) -> PyResult<()> {
    let arr = x.as_array();
    if arr.iter().any(|v| !v.is_finite()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array contains NaN or Inf values"
        ));
    }
    Ok(())
}
```

### Anti-Patterns to Avoid
- **Inline validation per model:** Each clustering model currently copies the same NaN/empty check. Replace with shared function calls.
- **Silent clamping:** `c.max(1e-10)` hides user errors. Validate and report.
- **Checking `is_fitted()` inconsistently:** Some models check `self.coefficients.is_some()`, others use the `check_is_fitted` helper, others don't check at all. Standardize.
- **Duplicate validation in Python + Rust:** The Python layer should do a fast check that produces good Python errors; the Rust layer keeps its checks as defense-in-depth. Don't remove Rust-side validation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NaN/Inf checking | Per-model `x.iter().any(...)` | Shared `validate_fit_input` / `validate_unsupervised_input` | Already exists, proven across 20+ models |
| NotFitted guard | Ad-hoc `if self.coef.is_none()` | `check_is_fitted(&self.field, "predict")` | Standardized error messages |
| Error type mapping | Custom Python exception construction | `to_py_runtime_err` / map FerroError variants | Already centralized in errors.rs |

**Key insight:** Most validation infrastructure already exists -- the work is adoption and coverage, not invention.

## Common Pitfalls

### Pitfall 1: Breaking Existing Tests
**What goes wrong:** Adding validation to a model's fit() path that previously accepted certain inputs (e.g., a model that silently handled NaN by propagating it) now rejects those inputs, breaking tests that relied on the old behavior.
**Why it happens:** The testing module `nan_inf_validation.rs` has checks that PASS when a model either rejects NaN OR handles it gracefully. Changing to strict rejection may break tests that expected graceful handling.
**How to avoid:** Run `cargo test --test edge_cases` and `cargo test --lib` after each model change. The existing edge_case_matrix macro tests expect `assert!(result.is_err())` for NaN/Inf, so strict rejection is the expected behavior.
**Warning signs:** Test failures in `testing::nan_inf_validation` or `edge_cases::edge_case_matrix`.

### Pitfall 2: Clustering vs Model Trait Differences
**What goes wrong:** Clustering models use `ClusteringModel` trait with `fit(&mut self, x: &Array2<f64>)` (no y parameter), while supervised models use `Model` trait with `fit(&mut self, x: &Array2<f64>, y: &Array1<f64>)`. Using `validate_fit_input` (which takes y) in clustering code causes compilation errors.
**Why it happens:** Two separate trait hierarchies for supervised vs unsupervised.
**How to avoid:** Create a separate `validate_unsupervised_input(x)` function for clustering and decomposition modules. Also `check_finite(x)` already exists in `preprocessing/mod.rs` for this purpose but only checks finite, not empty.
**Warning signs:** Compile errors about wrong number of arguments.

### Pitfall 3: Builder Pattern vs Validation
**What goes wrong:** Attempting to make builder methods like `with_c()` return `Result` breaks the fluent builder chain pattern used everywhere.
**Why it happens:** `Model::new().with_c(1.0).with_kernel(...)` requires each method to return `Self`, not `Result<Self>`.
**How to avoid:** Validate hyperparameters at the START of `fit()`, not in the builder. This preserves the ergonomic API while catching invalid params before computation.
**Warning signs:** API changes that break downstream code in Python bindings and tests.

### Pitfall 4: Performance Impact of Validation
**What goes wrong:** Adding `x.iter().any(|v| !v.is_finite())` to every fit/predict adds O(n*p) scan overhead.
**Why it happens:** Linear scan over entire array before computation.
**How to avoid:** This is acceptable for correctness. The scan is cheap compared to actual model training. For predict with small arrays this is negligible. Keep it simple.
**Warning signs:** Only matters if benchmarks show regression on tiny-dataset microbenchmarks.

### Pitfall 5: Python Validation Must Map to Correct Exception Types
**What goes wrong:** Using `PyRuntimeError` for all validation errors instead of `PyValueError`.
**Why it happens:** The existing `to_py_runtime_err` helper maps everything to RuntimeError.
**How to avoid:** Input validation errors should raise `PyValueError` (consistent with sklearn). `NotFitted` should raise `RuntimeError` or a custom `NotFittedError`. The error mapping will be audited more thoroughly in Phase 3 (ROBU-11), but Python-side validation in Phase 1 should use `PyValueError` for NaN/Inf/empty/shape errors.
**Warning signs:** Python tests that `pytest.raises(RuntimeError)` when they should `pytest.raises(ValueError)`.

## Code Examples

### Existing validate_fit_input (models/mod.rs:1235-1255)
```rust
// Source: ferroml-core/src/models/mod.rs line 1235
pub fn validate_fit_input(x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if x.nrows() != y.len() {
        return Err(FerroError::shape_mismatch(
            format!("X has {} rows", x.nrows()),
            format!("y has {} elements", y.len()),
        ));
    }
    if x.is_empty() || y.is_empty() {
        return Err(FerroError::invalid_input("Empty input data"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input("X contains NaN or infinite values"));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::invalid_input("y contains NaN or infinite values"));
    }
    Ok(())
}
```

### Existing check_is_fitted (models/mod.rs:1227-1232)
```rust
// Source: ferroml-core/src/models/mod.rs line 1227
pub fn check_is_fitted<T>(fitted_data: &Option<T>, operation: &str) -> Result<()> {
    if fitted_data.is_none() {
        return Err(FerroError::not_fitted(operation));
    }
    Ok(())
}
```

### Existing edge_case_matrix test macro (tests/edge_cases.rs:247-277)
```rust
// Source: ferroml-core/tests/edge_cases.rs line 247
#[test]
fn nan_input_rejected() {
    let x_nan = gen_nan_features();
    let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let mut model = $model_expr;
    let result = model.fit(&x_nan, &y);
    assert!(result.is_err(), "NaN features should be rejected");
}
```

### Inline Clustering Validation (to be replaced)
```rust
// Source: ferroml-core/src/clustering/kmeans.rs line 843-853
// This inline validation should be replaced with shared function call
fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
    if x.is_empty() || x.nrows() == 0 {
        return Err(FerroError::InvalidInput("Input array cannot be empty".to_string()));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(FerroError::InvalidInput("Input contains NaN or infinite values".to_string()));
    }
    // ... model-specific validation continues
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No validation (early Plans) | Per-model inline validation | Plan X (edge case matrix) | ~20 models covered |
| No NaN/Inf testing | `testing::nan_inf_validation` module + edge_cases macros | Plan X | Test infrastructure exists |
| Silent clamping of params | Still current | - | Needs fixing per VALID-08 |
| No Python-side validation | Still current | - | Needs adding per VALID-10 |

## Inventory of Gaps

### Models Already Calling validate_fit_input (COVERED)
LinearRegression, RidgeRegression, LassoRegression, ElasticNet, ElasticNetCV, LassoCV, RidgeCV, RidgeClassifier, LogisticRegression, DecisionTree (Regressor+Classifier), RandomForest (Regressor+Classifier), GradientBoosting (Regressor+Classifier), ExtraTrees (Regressor+Classifier), AdaBoost (Regressor+Classifier), HistGradientBoosting (Regressor+Classifier), KNeighbors (Regressor+Classifier), NearestCentroid, SVC, SVR, LinearSVC, LinearSVR, SGD (Regressor+Classifier), GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, QuantileRegression, RobustRegression, Calibration models, BaggingRegressor, BaggingClassifier, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor

### Models MISSING validate_fit_input (GAPS)
- **GaussianProcessRegressor** / **GaussianProcessClassifier** -- no validation
- **IsolationForest** -- no validation (unsupervised, no y)
- **IsotonicRegression** -- no validation
- **LocalOutlierFactor** -- no validation (unsupervised)
- **QuadraticDiscriminantAnalysis** -- no validation
- **MultiOutputRegressor** / **MultiOutputClassifier** -- delegates to inner, but no outer validation

### Clustering Models (need shared function)
- **KMeans** -- has inline NaN/empty checks, needs refactoring to shared
- **DBSCAN** -- has inline NaN/empty checks, needs refactoring to shared
- **GMM** -- has sample count check, missing NaN/Inf check
- **HDBSCAN** -- has empty check, missing NaN/Inf check
- **Agglomerative** -- has empty check, missing NaN/Inf check

### Decomposition Models (no validation at all)
- **PCA** -- no NaN/Inf/empty validation
- **t-SNE** -- no NaN/Inf/empty validation
- **LDA** -- no NaN/Inf/empty validation
- **TruncatedSVD** -- no NaN/Inf/empty validation
- **FactorAnalysis** -- no NaN/Inf/empty validation

### Preprocessing/Transformers
- **check_finite** exists in `preprocessing/mod.rs` but usage is limited
- Scalers (StandardScaler, MinMaxScaler, RobustScaler) have some inline validation
- Other preprocessors need audit

### NotFitted Test Coverage
- Currently tested: ~25 models (edge_cases.rs lines 1207-1272)
- Missing from tests: GaussianProcess, IsolationForest, Isotonic, LOF, QDA, MultiOutput, Perceptron, PassiveAggressive, AdaBoost, ExtraTrees, HistGBT, most preprocessors, most clustering models beyond KMeans/DBSCAN
- Total gap: ~30 models/transformers need NotFitted tests

### Hyperparameter Validation (VALID-08)
Models using silent clamping (needs conversion to validation):
- SVC: `with_c(c)` -> `c.max(1e-10)`
- SVC: `with_tol(tol)` -> `tol.max(1e-10)`
- Many models likely follow similar pattern -- needs full audit

## Open Questions

1. **Where to place shared unsupervised validation?**
   - What we know: `validate_fit_input` is in `models/mod.rs` which is supervised-specific. Clustering uses `clustering/mod.rs`.
   - What's unclear: Should we put shared validation in a top-level `validation.rs` module, or keep variants near their trait definitions?
   - Recommendation: Create a top-level `ferroml-core/src/validation.rs` module that both `models/mod.rs` and `clustering/mod.rs` import from. This is the cleanest approach per VALID-09.

2. **How detailed should NaN/Inf error messages be?**
   - What we know: Current messages say "X contains NaN or infinite values" without position info.
   - What's unclear: Should we report the first offending position (row, col)?
   - Recommendation: Report count and first position: "X contains 3 NaN values (first at row 5, column 2)". This helps debugging without O(n*p) string formatting overhead (just track first occurrence during the scan).

3. **Python-side validation: before or after array conversion?**
   - What we know: NumPy arrays are converted to ndarray views in PyO3. Validation could happen on the NumPy side or the ndarray side.
   - What's unclear: Whether PyO3's numpy integration makes it easier to validate before or after conversion.
   - Recommendation: Validate after conversion to ndarray view (inside the Rust binding functions). This avoids Python overhead and uses the same `is_finite()` logic as Rust-side. Add a helper in `array_utils.rs` that takes `PyReadonlyArray2<f64>` and checks `.as_array().iter().any(|v| !v.is_finite())`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test + cargo test + pytest |
| Config file | Cargo.toml (workspace), ferroml-core/Cargo.toml |
| Quick run command | `cargo test --test edge_cases` |
| Full suite command | `cargo test && pytest ferroml-python/tests/` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| VALID-01 | NaN/Inf in X at fit rejected | integration | `cargo test --test edge_cases nan_input_rejected` | Partial (covers ~22 models) |
| VALID-02 | NaN/Inf in y at fit rejected | integration | `cargo test --test edge_cases nan_target_rejected` | Partial (covers ~22 models) |
| VALID-03 | NaN/Inf in X at predict rejected | integration | `cargo test --test edge_cases predict_time_validation` | Partial (covers ~8 models) |
| VALID-04 | Empty dataset returns FerroError | integration | `cargo test --test edge_cases empty_input_rejected` | Partial (covers ~22 models) |
| VALID-05 | Single-sample handled gracefully | integration | `cargo test --test edge_cases single_sample` | Partial (covers ~22 models) |
| VALID-06 | n_features_in_ validated at predict | integration | `cargo test --test edge_cases shape_mismatch` | Partial (covers ~4 models) |
| VALID-07 | NotFitted guard on all models | integration | `cargo test --test edge_cases not_fitted` | Partial (covers ~25 models) |
| VALID-08 | Hyperparameter validation at construction | unit | `cargo test --lib` (new tests needed) | No |
| VALID-09 | Shared validation function | unit | `cargo test --lib validation` | Partial (validate_fit_input exists) |
| VALID-10 | Python array validation | Python test | `pytest ferroml-python/tests/test_input_validation.py` | No |

### Sampling Rate
- **Per task commit:** `cargo test --test edge_cases -x`
- **Per wave merge:** `cargo test && pytest ferroml-python/tests/`
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps
- [ ] `ferroml-core/src/validation.rs` -- shared validation module (VALID-09)
- [ ] New test entries in `edge_cases.rs` for all missing models (VALID-01 through VALID-07)
- [ ] `ferroml-python/tests/test_input_validation.py` -- Python-side validation tests (VALID-10)
- [ ] Hyperparameter validation unit tests in model source files (VALID-08)

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis of `ferroml-core/src/models/mod.rs` lines 1227-1316 (validation functions)
- Direct codebase analysis of `ferroml-core/tests/edge_cases.rs` (existing test coverage)
- Direct codebase analysis of `ferroml-core/src/clustering/` (inline validation patterns)
- Direct codebase analysis of `ferroml-core/src/decomposition/` (no validation found)
- Direct codebase analysis of `ferroml-python/src/errors.rs` and `array_utils.rs` (Python error handling)

### Secondary (MEDIUM confidence)
- CLAUDE.md project guidelines for coding conventions
- REPO_MAP.md referenced for model inventory (not fully loaded, used memory)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies, using existing Rust/PyO3 infrastructure
- Architecture: HIGH - patterns clearly established by existing `validate_fit_input`
- Pitfalls: HIGH - identified from direct codebase analysis of current validation patterns
- Gap inventory: HIGH - mechanical grep/search across all source files

**Research date:** 2026-03-20
**Valid until:** 2026-04-20 (stable -- validation patterns don't change rapidly)
