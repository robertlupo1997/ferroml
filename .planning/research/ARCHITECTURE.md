# Architecture Patterns: Production Hardening for FerroML

**Domain:** ML library production hardening (Rust + Python bindings)
**Researched:** 2026-03-20

## Recommended Architecture

FerroML's hardening work maps onto four architectural layers that already exist but need deepening. The key insight: FerroML already has the *skeleton* of production-quality validation (the `validate_fit_input` function, the `FeatureSchema` system, the `FerroError` enum), but coverage is inconsistent and the performance-critical paths bypass safety checks.

### Hardening Layer Architecture

```
  Python User Code
       |
  [Layer 1: PyO3 Boundary Validation]
       |  - NumPy array type/shape checks
       |  - Error translation (FerroError -> Python exceptions)
       |
  [Layer 2: Input Validation Gate]
       |  - validate_fit_input() / validate_predict_input()
       |  - NaN/Inf rejection (or allow-nan for tree models)
       |  - Empty data rejection
       |  - Shape consistency checks
       |  - Parameter validation (hyperparameter bounds)
       |
  [Layer 3: Algorithm Core]
       |  - Linear algebra (linalg.rs, nalgebra, faer)
       |  - Model fitting (models/, clustering/, decomposition/)
       |  - Statistical computation (stats/)
       |  - NO unwrap() on user-derived data in this layer
       |
  [Layer 4: Numerical Safety Net]
       |  - Output sanity checks (predictions finite?)
       |  - Convergence monitoring (max iterations, tolerance)
       |  - Numerical guard rails (regularization fallbacks, condition number checks)
       |
  Results returned to user
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **PyO3 Boundary** (`ferroml-python/src/`) | Array conversion, error translation, Python type safety | Input Validation Gate, Algorithm Core |
| **Input Validation Gate** (`models/mod.rs::validate_fit_input`, `schema.rs`, `preprocessing/mod.rs::check_finite`) | Reject invalid data before computation begins | All model fit/predict entry points |
| **Algorithm Core** (`models/`, `clustering/`, `decomposition/`, `linalg.rs`) | Mathematical computation, model fitting, prediction | Linalg Foundation, Stats Foundation |
| **Linalg Foundation** (`linalg.rs`, `simd.rs`, nalgebra, faer) | Matrix decomposition, solving, distance computation | Algorithm Core |
| **Stats Foundation** (`stats/`) | Hypothesis testing, diagnostics, confidence intervals | Algorithm Core, Metrics |
| **Error System** (`error.rs`) | Typed error variants with context for all failure modes | Every layer |
| **Performance Layer** (`simd.rs`, rayon, faer) | SIMD vectorization, parallelism, optimized backends | Algorithm Core, Linalg |

### Data Flow

**Training path (with hardening annotations):**

```
User calls model.fit(X, y)
  |
  +--[GATE] validate_fit_input(X, y)
  |    +-- Shape check: X.nrows() == y.len()
  |    +-- Empty check: X not empty, y not empty
  |    +-- Finite check: X.iter().all(is_finite), y.iter().all(is_finite)
  |    +-- (MISSING) Parameter check: hyperparameters in valid ranges
  |    +-- Returns FerroError on failure
  |
  +--[CORE] Algorithm-specific fitting
  |    +-- Linear algebra operations (QR, Cholesky, SVD)
  |    +-- Iterative solvers (convergence checks needed)
  |    +-- Statistical computation
  |    +-- (CONCERN) unwrap() calls on intermediate results
  |
  +--[OUTPUT] Fitted model state stored
       +-- (MISSING) Output sanity check on fitted parameters
```

**Prediction path:**

```
User calls model.predict(X)
  |
  +--[GATE] check_is_fitted() -- ensures model was trained
  +--[GATE] validate_predict_input(X) -- shape + finite checks
  |
  +--[CORE] model.predict internal
  |    +-- Transform input, compute predictions
  |    +-- (CONCERN) unwrap() on matrix operations
  |
  +--[OUTPUT] Array1<f64> predictions
       +-- (MISSING) assertion that predictions are finite
```

## Current State Assessment

### What Already Exists (and works)

1. **`validate_fit_input()`** in `models/mod.rs` -- shape checks, empty data rejection, NaN/Inf rejection. Called by ~40+ model fit methods across the codebase.

2. **`validate_fit_input_allow_nan()`** -- variant for tree-based models (HistGBT) that can handle missing values. Rejects Inf but allows NaN in X. Called by HistGBT models.

3. **`validate_predict_input()`** and `validate_predict_input_allow_nan()` -- predict-time validation with feature count checking.

4. **`check_is_fitted()`** -- guards predict/transform calls on unfitted models.

5. **`FeatureSchema`** in `schema.rs` -- rich validation system with feature types, ranges, validation modes (Strict/Warn/Permissive). Currently optional/underused.

6. **`check_finite()`** in `preprocessing/mod.rs` -- standalone finite check for preprocessing.

7. **`FerroError` enum** -- well-designed with `InvalidInput`, `ShapeMismatch`, `NotFitted`, `ConvergenceFailure`, `NumericalError`, `AssumptionViolation` variants. Good error messages via constructor methods.

### What Is Missing or Inconsistent

1. **Parameter validation at construction time.** Models accept invalid hyperparameters silently (e.g., `SVC::new().with_c(-1.0)` succeeds, fails at fit time). sklearn validates eagerly in `__init__` or at `set_params`.

2. **Predict-time output validation.** No check that predictions are finite after computation. A model that converges to NaN coefficients will produce NaN predictions silently.

3. **unwrap() in critical paths.** 6,195+ unwrap/expect calls across 169 files. Many are safe (array construction, iterator operations on validated data), but some in hot paths (SVM cache, stats, regularized models) can panic on edge cases.

4. **Clustering/decomposition validation gaps.** Clustering models (KMeans, DBSCAN, GMM, HDBSCAN) and decomposition models (PCA, t-SNE, TruncatedSVD) use their own validation that may not consistently call the centralized functions.

5. **Performance-critical path optimization.** The `x.iter().any(|v| !v.is_finite())` scan is O(n*d) and runs on every fit call. For large datasets this adds overhead. Could be SIMD-accelerated or sampled.

## Patterns to Follow

### Pattern 1: Centralized Validation Gate (sklearn model)

**What:** All validation flows through a small set of functions. Models call `validate_fit_input()` at the top of `fit()`. sklearn uses `validate_data()` (formerly `_validate_data`) which wraps `check_array()`.

**Why:** Single place to add new checks, consistent behavior across models, easy to audit.

**FerroML status:** Already partially implemented. `validate_fit_input()` exists and is called by ~40 model fit methods. The gap is that some models (particularly in clustering and decomposition) may have their own ad-hoc validation.

**Action:** Audit all `fn fit()` entry points to ensure they call the centralized validator. No new code needed for the validator itself -- just ensure universal adoption.

```rust
// Already exists in models/mod.rs -- this is the RIGHT pattern
pub fn validate_fit_input(x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if x.nrows() != y.len() { return Err(FerroError::shape_mismatch(...)); }
    if x.is_empty() || y.is_empty() { return Err(FerroError::invalid_input("Empty input data")); }
    if x.iter().any(|v| !v.is_finite()) { return Err(FerroError::invalid_input("X contains NaN or infinite values")); }
    if y.iter().any(|v| !v.is_finite()) { return Err(FerroError::invalid_input("y contains NaN or infinite values")); }
    Ok(())
}
```

### Pattern 2: Eager Parameter Validation (Builder Guards)

**What:** Validate hyperparameters when they are set, not when `fit()` is called. sklearn validates in `__init__` (constructor) or `set_params`.

**Why:** Fail fast. Users discover configuration errors immediately, not 30 minutes into a training run.

**FerroML approach:** Add a `validate_params()` method to models, called at the end of the builder chain or at the start of `fit()`. Since FerroML uses the builder pattern (`Model::new().with_param(val)`), the cleanest approach is validation in `fit()` before any computation, since builder methods are chainable and validating mid-chain complicates the API.

```rust
// Recommended: validate at fit time (keeps builder ergonomic)
pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    self.validate_params()?;  // NEW: check C > 0, gamma > 0, etc.
    validate_fit_input(x, y)?;
    // ... algorithm
}

fn validate_params(&self) -> Result<()> {
    if self.c <= 0.0 { return Err(FerroError::invalid_input("C must be positive")); }
    if self.max_iter == 0 { return Err(FerroError::invalid_input("max_iter must be > 0")); }
    Ok(())
}
```

### Pattern 3: Unwrap Audit by Risk Tier

**What:** Categorize unwrap() calls by risk level and fix from highest risk down.

**Why:** 6,195 unwrap calls is too many to fix at once. Prioritize by impact.

**Risk tiers:**
- **Tier 1 (CRITICAL):** unwrap on user-provided data or data derived from user input without prior validation. These can panic in production.
- **Tier 2 (HIGH):** unwrap in iterative algorithm loops (SVM, boosting, optimization). Edge cases in convergence can trigger these.
- **Tier 3 (MEDIUM):** unwrap on intermediate computed values (matrix decomposition results, array slicing).
- **Tier 4 (LOW):** unwrap on construction (e.g., `Array2::zeros((n,m))` which cannot fail), or in test code.

**Action:** Focus on Tier 1 and Tier 2. Tier 3 is nice-to-have. Tier 4 is acceptable.

### Pattern 4: Performance Backend Swapping (nalgebra to faer)

**What:** Replace performance-critical linear algebra operations with faer while keeping nalgebra for non-hot paths.

**Why:** nalgebra's Jacobi SVD is 13.8x slower than faer for PCA-scale matrices. The faer crate is already a dependency.

**Architecture for the swap:**

```
linalg.rs (facade)
  +-- qr_decomposition()       -> delegates to qr_decomposition_faer() [already done]
  +-- cholesky()                -> delegates to cholesky_faer()          [already done]
  +-- thin_svd()                -> NEW: delegate to faer thin SVD
  +-- eigendecomposition()      -> NEW: delegate to faer

models use linalg.rs, NOT nalgebra directly
```

**Key:** The `linalg.rs` module already acts as a facade with both native and faer variants. Extend this pattern to SVD and eigendecomposition. Models should never call nalgebra directly for decompositions -- always go through `linalg.rs`.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Validation in the Wrong Layer

**What:** Performing NaN checks inside the PyO3 binding layer (Python side) instead of in Rust.

**Why bad:** Validation would be duplicated. Rust API users (using ferroml-core directly) would not get validation. Maintenance burden doubles.

**Instead:** All validation happens in the Rust core. PyO3 layer only handles type conversion (NumPy -> ndarray) and error translation (FerroError -> Python exception).

### Anti-Pattern 2: Per-Model Custom Validation

**What:** Each model implementing its own shape/finite/empty checks instead of calling the centralized function.

**Why bad:** Inconsistent behavior. When a new check is added (e.g., minimum sample count), it has to be added to every model.

**Instead:** All models call `validate_fit_input()` (or `validate_fit_input_allow_nan()` for NaN-tolerant models). Model-specific validation (e.g., "need at least 2 classes for classification") goes in a separate model-specific check AFTER the centralized one.

### Anti-Pattern 3: Silent Fallback on Numerical Issues

**What:** Catching a numerical error (singular matrix, non-convergence) and returning a default value or continuing silently.

**Why bad:** Produces subtly wrong results. Users trust the output without knowing computation failed.

**Instead:** Return `FerroError::NumericalError` or `FerroError::ConvergenceFailure` with context. Let the user decide how to handle it. This is FerroML's strength: statistical rigor means surfacing problems, not hiding them.

### Anti-Pattern 4: Blanket unwrap Replacement

**What:** Mechanically replacing every `unwrap()` with `?` or `.ok_or_else()` without understanding whether the unwrap is actually reachable.

**Why bad:** Adds noise. Many unwraps are provably safe (e.g., `array.as_slice().unwrap()` on a contiguous array just validated). Converting these adds error branches that can never execute, complicating the code.

**Instead:** Use the tiered audit approach (Pattern 3). Focus on unwraps that can actually be reached by user input.

## Scalability Considerations

| Concern | At 1K samples | At 100K samples | At 1M samples |
|---------|--------------|-----------------|---------------|
| **NaN/Inf scan** | <1ms (trivial) | ~10ms (acceptable) | ~100ms (consider SIMD or sampling) |
| **SVM kernel cache** | Full matrix (fast) | LRU cache (slower) | Infeasible (use LinearSVC) |
| **PCA SVD** | <10ms | Depends on d (nalgebra slow) | Must use faer thin SVD |
| **HistGBT histogram** | <10ms | Allocation-heavy | Need pre-allocated buffers |
| **Validation overhead** | Negligible | 1-2% of fit time | <1% of fit time |

## Suggested Build Order (Dependencies Between Components)

The hardening work has clear dependencies that dictate phase ordering:

### Phase 1: Input Validation Completeness (Foundation)

**Must come first** because all other hardening work assumes inputs are validated.

- Audit all `fit()` entry points for `validate_fit_input()` calls
- Add parameter validation (`validate_params()`) to models
- Fix empty data panics
- Fix the 6 pre-existing test failures (TemperatureScaling, IncrementalPCA)

**Dependencies:** None (pure validation code)
**Unlocks:** Everything else (clean inputs means unwrap audit is meaningful)

### Phase 2: Unwrap Audit (Safety)

**Comes after validation** because many Tier 1 unwraps become Tier 4 (safe) once inputs are validated.

- Audit Tier 1 (user-data-derived) and Tier 2 (iterative loops) unwraps
- Replace with proper error propagation
- Add SVM cache unit tests (correctness-critical fragile area)

**Dependencies:** Phase 1 (validation reduces the unwrap audit surface)
**Unlocks:** Confidence in edge case handling

### Phase 3: Performance Optimization (Speed)

**Comes after safety** because performance changes are risky and need the safety net of validation + error handling to catch regressions.

- Replace nalgebra Jacobi SVD with faer thin SVD in PCA/TruncatedSVD/LDA/FactorAnalysis
- Add Cholesky normal equations for OLS (n >> 2d case)
- Add faer Cholesky for Ridge
- LinearSVC shrinking + f_i cache
- HistGBT histogram optimization
- SVC threshold tuning

**Dependencies:** Phase 2 (need clean error handling to safely refactor hot paths)
**Unlocks:** Performance parity claims

### Phase 4: Documentation and Polish

**Comes last** because it documents the final state.

- Document known limitations (RandomForest parallel non-determinism, sparse limitations)
- Update benchmarks
- Verify Python API backward compatibility

**Dependencies:** Phases 1-3 (document final state, not intermediate)

### Dependency Graph

```
Phase 1 (Validation)
    |
    v
Phase 2 (Unwrap Audit)
    |
    v
Phase 3 (Performance)
    |
    v
Phase 4 (Documentation)
```

This is strictly sequential because each phase depends on the prior phase being complete. Attempting performance optimization before the unwrap audit risks introducing panics in refactored code. Attempting the unwrap audit before validation means you cannot distinguish "this unwrap is safe because inputs are validated" from "this unwrap is dangerous."

## Sources

- [scikit-learn check_array documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html) -- sklearn's centralized input validation, the model FerroML's `validate_fit_input` follows (HIGH confidence)
- [scikit-learn validate_data documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.validate_data.html) -- sklearn's newer validation API replacing `_validate_data` (HIGH confidence)
- [scikit-learn ensure_all_finite rename](https://github.com/scikit-learn/scikit-learn/issues/29262) -- API evolution context (MEDIUM confidence)
- FerroML codebase analysis: `models/mod.rs`, `schema.rs`, `preprocessing/mod.rs`, `error.rs`, `linalg.rs` (HIGH confidence, direct code inspection)
- FerroML `.planning/codebase/CONCERNS.md` -- existing issue inventory (HIGH confidence)
- FerroML `.planning/codebase/ARCHITECTURE.md` -- existing architecture documentation (HIGH confidence)

---

*Architecture research: 2026-03-20*
