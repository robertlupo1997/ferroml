# Domain Pitfalls

**Domain:** Numerical/ML library hardening (Rust with Python bindings)
**Researched:** 2026-03-20

## Critical Pitfalls

Mistakes that cause correctness regressions, data corruption, or rewrites.

### Pitfall 1: Cholesky Normal Equations Squaring the Condition Number

**What goes wrong:** When replacing QR-based solvers with Cholesky normal equations (X^T X) for performance, the condition number of the problem is squared. A matrix with condition number 10^6 becomes 10^12 after forming normal equations, causing catastrophic cancellation on ill-conditioned datasets. Tests pass on well-conditioned synthetic data, then users hit silent numerical garbage on real-world data with correlated features.

**Why it happens:** Cholesky on normal equations is genuinely 2-3x faster than QR for n >> d. The performance win is real, so developers switch without realizing the stability tradeoff. Standard test suites use well-conditioned data (condition numbers < 10^4) that never triggers the failure mode.

**Consequences:** OLS/Ridge predictions silently become inaccurate on collinear features. Coefficients diverge from sklearn. No error raised -- the computation completes, just with wrong answers. Regression tests pass because they use clean synthetic data.

**Prevention:**
- Add explicit tests with ill-conditioned matrices (condition number 10^8+, near-rank-deficient X)
- Use Cholesky ONLY when condition number is checked first, or when regularization guarantees stability (Ridge with alpha > 0)
- For OLS: keep QR as fallback; use Cholesky only when `n > 5*d` AND `d < threshold` (e.g., d < 500)
- Compare coefficient outputs against QR-solved results on the same data as a regression test
- sklearn uses scipy.linalg.lstsq (SVD-based) precisely because it is numerically stable -- match their behavior on edge cases

**Detection:** Cross-library tests against sklearn on datasets with `np.linalg.cond(X) > 1e8` will catch this. Add at least 3 ill-conditioned regression tests before switching any solver.

**Phase relevance:** Performance optimization phase (OLS/Ridge faer backend). Must add stability tests BEFORE swapping solvers.

**Confidence:** HIGH -- this is a well-documented numerical analysis result.

---

### Pitfall 2: Unwrap Audit Scope Creep and Behavioral Regressions

**What goes wrong:** Systematically replacing 6,195+ unwrap()/expect() calls introduces behavioral changes that break downstream code. An `unwrap()` on `Array2::from_shape_vec()` that "can never fail" gets replaced with `?` propagation, changing the function signature from infallible to fallible, which breaks callers. Or worse: an unwrap on an `Option` from `.get()` was correct because the index was always valid, but the replacement introduces an unnecessary error path that degrades performance in hot loops.

**Why it happens:** Bulk unwrap removal treats all unwraps as equal. But unwraps fall into distinct categories:
1. **Mathematically guaranteed safe:** e.g., `array.get(i).unwrap()` where `i` was just validated to be in bounds
2. **Safe by construction:** e.g., `Array2::from_shape_vec(shape, vec).unwrap()` where shape and vec length are computed together
3. **Actually unsafe:** e.g., `kernel_cache.get(&key).unwrap()` where cache eviction may have removed the entry
4. **Test code:** `unwrap()` in `#[test]` blocks (should be left alone)

Blindly converting category 1-2 unwraps to `?` propagation adds overhead, changes function signatures, and introduces error paths that will never fire but must now be handled by callers.

**Consequences:** Function signature changes cascade through the codebase. Performance degrades from unnecessary error checking in hot paths (SVM has ~50 non-test unwraps, many in inner loops). New error variants confuse users ("ShapeMismatch" errors from internal array construction that was always valid).

**Prevention:**
- Triage unwraps into the 4 categories BEFORE replacing any
- Category 1-2: Convert to `expect("reason this is safe")` with documentation, NOT to `?`
- Category 3: Convert to proper error handling with `?`
- Category 4: Leave alone (test code)
- Focus on high-impact files first: `svm.rs` (50 non-test unwraps), `regularized.rs` (26), `hist_boosting.rs` (75 total)
- Run full test suite after EACH file is modified, not in batch

**Detection:** Any function signature change (adding `Result<>` where there was none) is a red flag. Review each such change for whether the error path is actually reachable.

**Phase relevance:** Unwrap audit phase. Triage before replacing.

**Confidence:** HIGH -- based on codebase analysis showing 6,195 unwraps across 169 files.

---

### Pitfall 3: NaN Validation Breaking Legitimate NaN-Producing Workflows

**What goes wrong:** Adding `is_finite()` checks at fit() entry rejects ALL NaN/Inf inputs, but some users rely on NaN as a missing value indicator (common in pandas/polars workflows). Or: validation is added at fit() but not predict(), so a model fitted on clean data silently produces NaN when given NaN at predict time. Or: validation catches NaN in X but not in y, and NaN targets silently corrupt the loss function.

**Why it happens:** sklearn rejects NaN at fit() by default but supports it with imputers in pipelines. The "just reject NaN" approach is correct but incomplete -- it must be consistent across all entry points (fit, predict, transform, partial_fit, score) and all inputs (X, y, sample_weight).

**Consequences:** Users with missing-value workflows get blocked. Inconsistent validation (fit rejects, predict doesn't) creates confusing behavior. Partial validation (X checked, y not) still allows silent corruption.

**Prevention:**
- Create a single `validate_inputs(x, y)` function called at ALL entry points, not ad-hoc checks per model
- Validate BOTH X and y (targets with NaN are a common but overlooked source of corruption)
- Validate sample_weight if provided
- Validate at predict() too, not just fit()
- Error message must be specific: "Input contains NaN at row 42, column 7" not just "Invalid input"
- Document that NaN rejection is the default; point users to SimpleImputer for missing data workflows
- Add tests for NaN in every position: X, y, sample_weight; at fit, predict, transform

**Detection:** Test matrix: {fit, predict, transform, partial_fit} x {NaN in X, NaN in y, Inf in X, -Inf in y} = 16+ test cases per model. If any combination silently succeeds with corrupt output, that's a bug.

**Phase relevance:** Input validation phase. Must be systematic, not per-model.

**Confidence:** HIGH -- sklearn's own history shows this took multiple releases to get right (check_array evolution from 0.18 to 1.0).

---

### Pitfall 4: faer SVD Sign Convention Differences Causing Silent Result Differences

**What goes wrong:** When replacing nalgebra's Jacobi SVD with faer's thin SVD for PCA, the singular vectors may have different sign conventions. SVD is unique up to sign flips of corresponding left/right singular vector pairs. nalgebra and faer may choose different sign conventions, causing PCA components to flip signs. This doesn't affect reconstruction error but changes `components_` signs, which breaks: (a) cross-library comparison tests vs sklearn, (b) serialized model compatibility (old models saved with nalgebra signs), (c) user code that depends on component sign interpretation.

**Why it happens:** IEEE does not mandate a sign convention for SVD. Different implementations legitimately produce different signs. sklearn enforces a convention via `svd_flip()` (largest absolute value in each component is positive). If FerroML doesn't apply the same convention after switching backends, results silently differ.

**Consequences:** PCA components flip sign between library versions (v0.3 nalgebra vs v0.4 faer). Cross-library tests against sklearn fail unpredictably. Users comparing components between versions get confused. Deserialized old PCA models produce different transform() outputs.

**Prevention:**
- Implement `svd_flip()` equivalent that normalizes sign convention AFTER any SVD call, regardless of backend
- Apply sign normalization in the SVD wrapper function (linalg.rs), not in each consumer
- Test that `pca.components_` signs match sklearn on reference datasets BEFORE and AFTER the switch
- Add a migration note: "PCA components may have different signs between v0.3 and v0.4; use `np.allclose(abs(a), abs(b))` for comparison"
- Test with `assert_allclose(np.abs(ferro_components), np.abs(sklearn_components))` first, then verify exact sign match

**Detection:** If any PCA/TruncatedSVD/LDA/FactorAnalysis test breaks after the faer switch with "values differ by sign only," this pitfall is the cause.

**Phase relevance:** Performance optimization phase (PCA faer SVD replacement). Must add sign normalization BEFORE switching backends.

**Confidence:** HIGH -- this is a known issue in every SVD backend migration.

---

### Pitfall 5: Performance Optimization Introducing Correctness Regressions in Rarely-Tested Paths

**What goes wrong:** Optimizing the HistGBT histogram inner loop, KMeans distance computation, or LinearSVC solver changes numerical behavior on edge cases. Example: replacing `sqrt(sum((a-b)^2))` with `sum((a-b)^2)` in KMeans saves a sqrt, but if downstream code (e.g., convergence check, tolerance comparison) assumed Euclidean distance (not squared), the convergence threshold is effectively squared, causing premature convergence or infinite loops on some datasets.

**Why it happens:** Performance optimizations change the numerical representation of intermediate values. The optimization is correct in isolation but breaks implicit contracts. The test suite verifies final predictions but not intermediate convergence behavior.

**Consequences:** Models converge to different solutions on real data. Benchmark numbers improve but edge-case correctness degrades. Regressions appear only on specific data distributions, not in test suite.

**Prevention:**
- For EVERY performance change, identify ALL consumers of the modified value
- When changing distance metrics (Euclidean to squared), update ALL thresholds that compare against them
- Run cross-library correctness tests (not just unit tests) after each optimization
- Add convergence behavior tests: "model converges in N iterations on dataset D" -- not just "predictions match"
- Keep the slow-but-correct path available as a debug/validation mode

**Detection:** Run the full `test_vs_sklearn_*.py` and `test_vs_linfa.py` suites after EACH performance change, not just at the end of the optimization phase. Watch for increased iteration counts or changed convergence behavior.

**Phase relevance:** Every performance optimization phase. Run cross-library tests after each change.

**Confidence:** HIGH -- FerroML already experienced this with the SVC LRU cache optimization (17.6x regression) and KMeans squared-distance fix in Plan W.

---

## Moderate Pitfalls

### Pitfall 6: Empty/Degenerate Input Panics Hidden by Array Indexing

**What goes wrong:** Empty datasets (n=0), single-sample datasets (n=1), single-feature datasets (d=1), or constant-feature datasets cause index-out-of-bounds panics deep in model internals. The panic stack trace is incomprehensible to users.

**Prevention:**
- Add early validation in `validate_fit_input()`: reject n=0, warn on n=1, check d >= model minimum
- For n=1: StandardScaler division by zero (std=0), KFold can't split, PCA can't compute covariance
- For constant features: VarianceThreshold selects nothing, StandardScaler produces NaN
- Test matrix: n={0, 1, 2, d+1} x d={1, 2} x {constant column, all identical rows}
- Return `FerroError::InvalidInput` with clear message, not a panic

**Detection:** Run `pytest test_errors.py` -- the currently-SKIPPED empty data test should pass after fix.

**Phase relevance:** Input validation phase.

**Confidence:** HIGH -- CONCERNS.md documents this as a known issue with a SKIPPED test.

---

### Pitfall 7: Parameter Validation Breaking Builder Pattern Ergonomics

**What goes wrong:** Adding eager validation at construction time (e.g., `SVC::new().with_c(-1.0)` errors immediately) breaks the builder pattern if validation depends on multiple parameters together. Example: `ElasticNet::new().with_l1_ratio(1.5)` -- is 1.5 invalid? Only if you know l1_ratio must be in [0,1]. But `with_l1_ratio` is called before `with_alpha`, and some parameter combinations are only invalid together.

**Prevention:**
- Validate individual parameters at builder methods (range checks that don't depend on other params)
- Validate inter-parameter constraints at fit() time, not construction time
- Use `validate_params()` method called at fit() entry that checks all constraints together
- Never panic in builder methods -- return the builder with the value stored, validate later
- Model the 4 SKIPPED `test_parameter_validation_*` tests as the acceptance criteria

**Detection:** If builder methods start returning `Result`, that's a design smell. Builders should be infallible; validation at fit() should be fallible.

**Phase relevance:** Parameter validation phase.

**Confidence:** MEDIUM -- design choice, but sklearn validates at fit() time, so matching that behavior is safest.

---

### Pitfall 8: PyO3 Binding Sync Breakage During Refactoring

**What goes wrong:** Changing a model's Rust API (adding a parameter, changing a return type, renaming a method) without updating the PyO3 wrapper causes: (a) compile errors caught by CI, or worse (b) the binding silently wraps the old API, producing wrong results or missing the new feature.

**Prevention:**
- The 3-file update rule (core impl, PyO3 wrapper, Python `__init__.py`) must be enforced for every API change
- Run `pytest test_bindings_correctness.py` after ANY model change
- Consider adding a compile-time check: if a model gains a new public method, the PyO3 wrapper must also expose it (possibly via a macro or code generation)
- During hardening: if you change fit() validation, the PyO3 fit() wrapper must also propagate the new errors correctly

**Detection:** `test_bindings_correctness.py` covers 30+ models but isn't exhaustive. Spot-check any changed model with `python -c "from ferroml.X import Y; print(dir(Y))"`.

**Phase relevance:** Every phase that touches model APIs.

**Confidence:** HIGH -- CONCERNS.md identifies this as the most common mistake.

---

### Pitfall 9: Regularization Masking Numerical Instability During Testing

**What goes wrong:** Ridge, Lasso, ElasticNet always add regularization (alpha > 0), which artificially improves condition numbers. Tests pass because regularization hides the underlying numerical instability. Then OLS (alpha=0) is tested on the same "well-conditioned" dataset and passes too. But real-world OLS on collinear data hits the instability that regularization was masking.

**Prevention:**
- Test OLS separately from regularized models on ill-conditioned data
- Don't assume "Ridge works therefore OLS works" -- they have fundamentally different stability profiles
- Include VIF > 100 collinear features in OLS test data
- Compare coefficient magnitude between OLS and Ridge(alpha=1e-10) -- if they differ significantly, the data is ill-conditioned and OLS is unreliable

**Detection:** OLS coefficients with magnitude > 10^6 on standardized data is a warning sign.

**Phase relevance:** Performance phase (OLS Cholesky switch) and correctness phase.

**Confidence:** MEDIUM -- theoretical concern, but FerroML currently uses QR which is stable.

---

### Pitfall 10: Cross-Library Test Tolerance Gaps Hiding Real Regressions

**What goes wrong:** Correctness tests against sklearn use `atol=1e-4` or `rtol=1e-3` tolerances. An optimization changes results at the 5th decimal place -- tests still pass, but the model is now solving a slightly different optimization problem (e.g., different convergence criterion, different initialization). Over multiple optimizations, tolerance debt accumulates until results are qualitatively different.

**Prevention:**
- Document the expected tolerance for each cross-library test and WHY (e.g., "1e-6 because both use same algorithm, 1e-2 because different solver")
- Tighten tolerances where possible after each optimization pass
- Track tolerance as a metric: if a change loosens required tolerance, that's a regression flag
- Keep a "tolerance audit" tracking which tests needed tolerance loosening and why
- Never loosen tolerance without a comment explaining the numerical reason

**Detection:** `git log` showing tolerance changes (`atol`, `rtol`, `assert_allclose`) is a red flag. Each such change should have a justification.

**Phase relevance:** Every phase. Tolerance discipline is ongoing.

**Confidence:** MEDIUM -- this is a process issue, not a specific bug.

---

## Minor Pitfalls

### Pitfall 11: Debug Build Size Explosion from New Test Binaries

**What goes wrong:** Adding a new `.rs` file in `ferroml-core/tests/` creates a separate test binary. Each binary links the entire crate, adding ~2 GB to debug builds. The consolidated 6-file structure was specifically chosen to prevent this.

**Prevention:** Never add new files to `ferroml-core/tests/`. Add tests to existing files or to module-level `#[cfg(test)]` blocks. This is documented in CLAUDE.md.

**Detection:** Debug build size > 16 GB after a change (currently ~14 GB).

**Phase relevance:** All phases. Hard rule.

**Confidence:** HIGH -- documented constraint from Plan consolidation.

---

### Pitfall 12: ort RC Dependency Breaking on Stable Release

**What goes wrong:** `ort 2.0.0-rc.11` has API differences from the eventual stable `ort 2.0.0`. When stable releases, Cargo may auto-upgrade (if using `^2.0.0-rc.11`), breaking ONNX export silently.

**Prevention:**
- Pin exact version: `ort = "=2.0.0-rc.11"` in Cargo.toml
- Monitor ort releases; upgrade deliberately when stable ships
- All 118 ONNX roundtrip tests must pass after any ort upgrade

**Detection:** CI ONNX tests fail after dependency update.

**Phase relevance:** Dependency management. Not a hardening phase issue per se.

**Confidence:** MEDIUM -- depends on ort release timeline.

---

### Pitfall 13: Parallel RandomForest Non-Determinism Documented Incorrectly

**What goes wrong:** Documentation says "use `n_jobs=1` for reproducibility" but the Rust API parameter might be named `parallel: bool` or `n_threads: usize`, not `n_jobs`. Users reading Python-style docs try a parameter that doesn't exist or has different semantics.

**Prevention:**
- Use consistent naming between Rust API, Python bindings, and documentation
- Test that the documented workaround actually produces deterministic results
- Add a doctest showing reproducible results with the correct parameter name

**Detection:** Review docs against actual API signatures.

**Phase relevance:** Documentation phase.

**Confidence:** LOW -- specific to FerroML naming, needs verification.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| NaN/Inf input validation | Inconsistent validation (fit vs predict vs transform) | Single `validate_inputs()` function used at ALL entry points |
| NaN/Inf input validation | NaN in y (targets) not checked | Validate y same as X |
| Unwrap audit | Bulk replacement changing function signatures | Triage into 4 categories first; only category 3 gets `?` |
| Unwrap audit | Performance regression from unnecessary error checking in hot loops | Benchmark hot-path files (svm.rs, hist_boosting.rs) before/after |
| PCA faer SVD | Sign convention differences | Add `svd_flip()` before switching backend |
| PCA faer SVD | Different convergence/accuracy characteristics | Compare singular values (not just vectors) against nalgebra output |
| OLS/Ridge Cholesky | Condition number squaring | Add ill-conditioned matrix tests BEFORE switching |
| OLS/Ridge Cholesky | Regularization masking instability in tests | Test OLS on collinear data separately |
| HistGBT optimization | Changed convergence from intermediate value changes | Test iteration count stability, not just final predictions |
| LinearSVC shrinking | Active set management introducing convergence oscillation | Test convergence on non-separable, overlapping-class data |
| KMeans optimization | Squared vs Euclidean distance threshold mismatch | Audit ALL consumers of distance values |
| SVC tuning | FULL_MATRIX_THRESHOLD boundary creating performance cliffs | Test at n=threshold-1, threshold, threshold+1 |
| Empty data handling | Index-out-of-bounds panic on n=0 before validation runs | Validate FIRST thing in fit(), before any array access |
| Parameter validation | Builder returning Result breaks ergonomics | Validate at fit(), not at builder methods |
| 6 pre-existing failures | Fixing TemperatureScaling/IncrementalPCA introduces new regressions | Fix in isolation, run full suite after each |

## Sources

- [Cholesky decomposition - Wikipedia](https://en.wikipedia.org/wiki/Cholesky_decomposition) -- condition number squaring
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) -- edge case handling patterns
- [Replacing unwrap() and avoiding panics in Rust](https://klau.si/blog/replacing-unwrap-and-avoiding-panics-in-rust/) -- systematic unwrap removal
- [Using unwrap() in Rust is Okay - Andrew Gallant](https://burntsushi.net/unwrap/) -- when unwrap is acceptable
- [ndarray-linalg NaN handling issue](https://github.com/rust-ndarray/ndarray-linalg/issues/179) -- NaN propagation in Rust linalg
- [sklearn LinearRegression performance issues](https://github.com/scikit-learn/scikit-learn/issues/22855) -- solver choice tradeoffs
- [faer-rs repository](https://github.com/sarah-quinones/faer-rs) -- faer design and differences from nalgebra
- [Numerical methods for linear least squares - Wikipedia](https://en.wikipedia.org/wiki/Numerical_methods_for_linear_least_squares) -- stability hierarchy (SVD > QR > Cholesky)
- FerroML `.planning/codebase/CONCERNS.md` -- internal audit of 6,195 unwraps, known bugs, performance gaps
- FerroML `.planning/PROJECT.md` -- active requirements and constraints

---

*Pitfalls research: 2026-03-20*
