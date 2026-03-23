# Phase 5: Documentation and Release - Research

**Researched:** 2026-03-22
**Domain:** Python docstrings, README documentation, benchmark publishing
**Confidence:** HIGH

## Summary

This phase is documentation-only: no code changes beyond adding/editing docstrings in PyO3 Rust files and updating README.md. The project has 107 `#[pyclass]` definitions across 15 binding files, of which roughly 65 are missing Examples sections and all but 1 are missing Notes sections. The gold-standard template already exists (LinearRegression in `linear.rs`) with full NumPy-style docstrings. docs/benchmarks.md already exists with full methodology and results (118 lines) -- DOCS-06 just needs verification and polish.

The primary work is mechanical: audit each model's docstring against the template, add Parameters/Examples/Notes sections where missing, then add a Known Limitations section to README.md. Docstrings live in Rust code (`ferroml-python/src/*.rs`) as `///` doc-comments above `#[pyclass]` definitions.

**Primary recommendation:** Batch docstring work by module file (15 files), prioritizing high-traffic modules (linear, trees, svm, clustering, preprocessing) first. Use LinearRegression as the canonical template.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Follow sklearn-style NumPy docstring format (already established in LinearRegression, KMeans)
- Sections: description, Parameters, Attributes, Examples, Notes (for limitations)
- Every parameter must include type, default value, and valid range where applicable
- Examples should be runnable (import + fit + predict/transform)
- Models that already have complete docstrings should be audited but not rewritten
- Add a "Known Limitations" section to README.md covering: RandomForest parallel non-determinism, sparse algorithm limits, ort RC status
- Add Notes section to docstrings where applicable (not every model)
- Key models needing notes: SVC (scaling sensitivity, RBF kernel tuning), RandomForest (parallel non-determinism), HistGBT (NaN handling behavior), GP models (no pickle support)
- docs/benchmarks.md already exists from Phase 4 -- verify and polish, don't recreate
- Docstrings should highlight FerroML's differentiator: statistical diagnostics (summary(), confidence intervals, p-values)
- Benchmark page should emphasize where FerroML wins
- README should position the library as "production-ready with known limitations"

### Claude's Discretion
- Exact docstring wording and example datasets
- Which models get "Notes" sections vs just basic docstrings
- README structure and ordering of limitations
- Whether to add type hints to Python stubs

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DOCS-01 | All 55+ Python model classes have complete docstrings (description, parameters, examples) | Audit found 65 models missing Examples; template exists in LinearRegression; docstrings live in `ferroml-python/src/*.rs` |
| DOCS-02 | All constructor parameters documented with type, default, valid range | Parameters sections exist in most models but need audit for completeness (type + default + valid range) |
| DOCS-03 | Known limitations documented in README (RF parallel non-determinism, sparse limits, ONNX RC) | README.md currently has no "Known Limitations" section; ort version is 2.0.0-rc.11 |
| DOCS-04 | Per-model known limitations documented in docstrings where applicable | Only 1 file (linear.rs) has a Notes section currently; key models identified in CONTEXT.md |
| DOCS-05 | Upgrade ort dependency status documented (RC vs stable, user expectations) | ort 2.0.0-rc.11 is used; optional dependency behind `onnx-validation` feature flag |
| DOCS-06 | Published performance benchmark page with methodology and results | docs/benchmarks.md already exists (118 lines) with full methodology, results table, and analysis -- needs verification only |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyO3 | (workspace) | Python bindings | Docstrings are `///` comments above `#[pyclass]` |
| NumPy docstring format | N/A | Docstring convention | sklearn standard, already used in LinearRegression/KMeans |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| N/A | N/A | No new dependencies | This phase is documentation-only |

### Alternatives Considered
None -- no libraries needed for documentation work.

## Architecture Patterns

### Docstring Location and Format

Docstrings live in PyO3 Rust code as `///` doc-comments directly above `#[pyclass]` definitions. They are NOT in Python `__init__.py` files (those just re-export).

**File:** `ferroml-python/src/<module>.rs`

```rust
/// Model description paragraph.
///
/// Extended description with FerroML differentiator mention.
///
/// Parameters
/// ----------
/// param_name : type, optional (default=value)
///     Description. Valid range: [min, max].
///
/// Attributes
/// ----------
/// attr_name : type
///     Description.
///
/// Examples
/// --------
/// >>> from ferroml.<module> import <Model>
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [3, 4], [5, 6]])
/// >>> y = np.array([0, 1, 0])
/// >>> model = Model()
/// >>> model.fit(X, y)
/// >>> model.predict(X)
///
/// Notes
/// -----
/// - Limitation 1 description.
/// - Limitation 2 description.
#[pyclass(name = "Model", module = "ferroml.<module>")]
pub struct PyModel { ... }
```

### Gold Standard Template (LinearRegression)

Located at `ferroml-python/src/linear.rs` lines 52-90. Includes:
- One-line summary + extended description mentioning statistical diagnostics
- Parameters with type, optional marker, default value
- Attributes with type and shape annotation
- Runnable example (import, create data, fit, predict, summary())
- Notes section (only model with one currently)

### Module File Map (15 files, 107 pyclass definitions)

| File | Models | Has Examples | Missing Examples |
|------|--------|-------------|-----------------|
| linear.rs | 13 | 5 | RobustRegression, QuantileRegression, Perceptron, RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier, IsotonicRegression |
| preprocessing.rs | 26 | 14 | PowerTransformer, QuantileTransformer, PolynomialFeatures, KBinsDiscretizer, VarianceThreshold, SelectKBest, KNNImputer, TargetEncoder, ADASYN, RandomUnderSampler, RandomOverSampler, Normalizer |
| ensemble.rs | 13 | 0 | ALL 13 (ExtraTrees x2, AdaBoost x2, SGD x2, PassiveAggressive, Bagging x2, Voting x2, Stacking x2) |
| trees.rs | 8 | 8 | None (audit only) |
| gaussian_process.rs | 9 | 0 | ALL 9 (4 kernels, GPR, GPC, SparseGPR, SparseGPC, SVGP) |
| decomposition.rs | 7 | 2 | IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis, + 1 more |
| clustering.rs | 5 | 5 | None (audit only -- AgglomerativeClustering may need check) |
| cv.rs | 8 | 0 | ALL 8 |
| svm.rs | 4 | 0 | ALL 4 |
| naive_bayes.rs | 4 | 0 | ALL 4 |
| neighbors.rs | 3 | 2 | NearestCentroid |
| neural.rs | 2 | 2 | None (audit only) |
| multioutput.rs | 2 | 2 | None (audit only) |
| anomaly.rs | 2 | 0 | IsolationForest, LocalOutlierFactor |
| calibration.rs | 1 | 0 | TemperatureScalingCalibrator |

**Summary:** ~65 models need Examples added, ~all need Notes where applicable, Parameters sections need audit for type+default+range completeness.

### Anti-Patterns to Avoid
- **Writing docstrings in Python `__init__.py`**: These are re-export files only. Docstrings go in Rust.
- **Overly long examples**: Keep examples to 5-8 lines (import, data, fit, predict/transform). Don't demonstrate every method.
- **Documenting every model's Notes section**: Only add Notes for models with user-facing gotchas. Don't pad with obvious information.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Docstring format | Custom format | NumPy/sklearn docstring convention | Industry standard, tools (pdoc, sphinx) parse it |
| Benchmark page | New benchmark suite | Existing docs/benchmarks.md | Already has methodology, results, analysis |
| API reference | Manual API docs | `cargo doc` / pdoc (already configured) | Auto-generated from docstrings |

## Common Pitfalls

### Pitfall 1: Forgetting the Rust rebuild
**What goes wrong:** Docstrings in Rust don't appear in Python until rebuilt
**Why it happens:** Docstrings are compiled into the .so/.abi3.so file
**How to avoid:** After editing any `ferroml-python/src/*.rs` file, run `maturin develop --release`
**Warning signs:** `help(Model)` shows old or no docstring

### Pitfall 2: Examples that don't actually run
**What goes wrong:** Example code in docstrings has import errors, wrong shapes, or missing data
**Why it happens:** No automated doctest runner for PyO3 Rust docstrings
**How to avoid:** Manually test each example in Python after rebuild, or write a verification script
**Warning signs:** Copy-paste from one model to another without adjusting imports/class names

### Pitfall 3: Parameter documentation drift
**What goes wrong:** Documented default doesn't match actual Rust default
**Why it happens:** Parameters changed during development, docstring not updated
**How to avoid:** Cross-reference each documented default with the `#[new]` implementation
**Warning signs:** `Model()` produces different behavior than documented

### Pitfall 4: Inconsistent optional/default notation
**What goes wrong:** Some use "optional (default=X)", others use "default: X", others omit
**Why it happens:** Different authors at different times
**How to avoid:** Use consistent format: `param : type, optional (default=value)` everywhere
**Warning signs:** Mixed formats in same file

### Pitfall 5: README benchmark numbers don't match docs/benchmarks.md
**What goes wrong:** README shows old Plan W numbers, benchmarks.md shows Phase 4 numbers
**Why it happens:** README was updated in Plan W, benchmarks.md was updated in Phase 4
**How to avoid:** Use benchmarks.md as source of truth, update README to reference it
**Warning signs:** Different ratios for same algorithm in different files

## Code Examples

### Docstring Template (classifier)
```rust
// Source: adapted from ferroml-python/src/linear.rs LinearRegression
/// Support Vector Classifier.
///
/// Implements C-Support Vector Classification using SMO optimization with
/// kernel functions. FerroML provides decision_function() for confidence
/// scores alongside standard predict().
///
/// Parameters
/// ----------
/// C : float, optional (default=1.0)
///     Regularization parameter. Must be positive.
/// kernel : str, optional (default="rbf")
///     Kernel type: "linear", "rbf", "poly", "sigmoid".
/// gamma : float, optional (default=auto)
///     Kernel coefficient for "rbf", "poly", "sigmoid".
///     If not set, uses 1/n_features.
/// tol : float, optional (default=1e-3)
///     Tolerance for stopping criterion. Must be positive.
/// max_iter : int, optional (default=1000)
///     Maximum number of iterations. -1 for no limit.
///
/// Attributes
/// ----------
/// support_vectors_ : ndarray of shape (n_sv, n_features)
///     Support vectors.
/// n_support_ : int
///     Number of support vectors.
/// n_features_in_ : int
///     Number of features seen during fit.
///
/// Examples
/// --------
/// >>> from ferroml.svm import SVC
/// >>> import numpy as np
/// >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
/// >>> y = np.array([0, 0, 1, 1])
/// >>> clf = SVC(kernel="rbf")
/// >>> clf.fit(X, y)
/// >>> clf.predict(np.array([[1.5, 1.5]]))
///
/// Notes
/// -----
/// - Feature scaling is strongly recommended. SVC with RBF kernel is
///   sensitive to feature magnitudes. Use StandardScaler before SVC.
/// - For datasets > 3000 samples, consider LinearSVC which scales better.
/// - Parallel non-determinism: results may vary slightly across runs
///   when using parallel kernel computation.
#[pyclass(name = "SVC", module = "ferroml.svm")]
```

### Docstring Template (transformer)
```rust
/// Standard Scaler: zero mean and unit variance normalization.
///
/// Parameters
/// ----------
/// with_mean : bool, optional (default=True)
///     If True, center the data before scaling.
/// with_std : bool, optional (default=True)
///     If True, scale the data to unit variance.
///
/// Attributes
/// ----------
/// mean_ : ndarray of shape (n_features,)
///     Per-feature mean. None if with_mean=False.
/// scale_ : ndarray of shape (n_features,)
///     Per-feature standard deviation. None if with_std=False.
///
/// Examples
/// --------
/// >>> from ferroml.preprocessing import StandardScaler
/// >>> import numpy as np
/// >>> X = np.array([[1, 2], [3, 4], [5, 6]])
/// >>> scaler = StandardScaler()
/// >>> X_scaled = scaler.fit_transform(X)
```

### README Known Limitations Section
```markdown
## Known Limitations

### RandomForest Parallel Non-Determinism
RandomForest and ExtraTrees use Rayon for parallel tree construction. Due to
work-stealing scheduling, results may vary slightly between runs even with
the same `random_state`. For reproducible results, set `n_jobs=1`.

### Sparse Algorithm Support
12 models support sparse input (CSR format) via the `fit_sparse()`/`predict_sparse()`
API. For unsupported models, sparse matrices are converted to dense automatically.
Very large sparse datasets (>100K features) may cause memory issues during conversion.

### ONNX Export (ort RC)
ONNX export depends on `ort 2.0.0-rc.11` (release candidate). The API is stable
and all 118 roundtrip tests pass, but users should be aware this is a pre-release
dependency. Pin your ort version to avoid surprises on upgrade.

### Per-Model Notes
See individual model docstrings for model-specific limitations (e.g., SVC scaling
sensitivity, GP models without pickle support, HistGBT NaN handling behavior).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No docstrings | Partial docstrings (linear, trees, clustering) | Plans A-X | ~40 models have Parameters, ~42 have Examples |
| Benchmarks in multiple files | Consolidated docs/benchmarks.md | Phase 4 | Single source of truth |
| No limitations documented | README has "Common Gotchas" section | Plan V | Gotchas cover dev issues, not user-facing limitations |

## Open Questions

1. **CV splitters as "models" for DOCS-01?**
   - What we know: 8 CV splitter classes (KFold, StratifiedKFold, etc.) are pyclass definitions
   - What's unclear: Whether DOCS-01's "55+ model classes" includes utility classes like CV splitters and GP kernels
   - Recommendation: Include them -- they're public API and users will call `help()` on them

2. **Kernel classes (RBF, Matern, etc.) docstrings?**
   - What we know: 4 kernel classes in gaussian_process.rs, each has minimal docs
   - What's unclear: Level of detail needed (they're building blocks, not standalone models)
   - Recommendation: Brief docstring with Parameters only, no Examples (kernels are passed to GP models)

3. **Existing README benchmark numbers**
   - What we know: README has Plan W numbers (RF 5x, GaussianNB 4.3x, etc.), benchmarks.md has Phase 4 numbers
   - What's unclear: Whether to update README to match benchmarks.md or keep both
   - Recommendation: README should reference benchmarks.md for full details, show a summary table with current Phase 4 numbers

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (Python) |
| Config file | ferroml-python/pytest.ini or pyproject.toml |
| Quick run command | `pytest ferroml-python/tests/test_docstrings.py -x` (to be created) |
| Full suite command | `pytest ferroml-python/tests/ -x --timeout=300` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOCS-01 | All models have docstrings with description, params, examples | smoke | `python -c "from ferroml.linear import LinearRegression; assert 'Parameters' in LinearRegression.__doc__"` | No -- Wave 0 |
| DOCS-02 | Parameters have type, default, range | smoke | Same script, check for "default=" pattern | No -- Wave 0 |
| DOCS-03 | README has Known Limitations | manual-only | Visual inspection of README.md | N/A |
| DOCS-04 | Per-model limitations in Notes | smoke | Check `__doc__` for "Notes" in key models | No -- Wave 0 |
| DOCS-05 | ort status documented | manual-only | Visual inspection of README.md | N/A |
| DOCS-06 | Benchmark page exists | manual-only | `test -f docs/benchmarks.md` | Already exists |

### Sampling Rate
- **Per task commit:** Rebuild with maturin, spot-check 3 models' `help()` output
- **Per wave merge:** Run full pytest suite to ensure no regressions
- **Phase gate:** Full suite green + manual README review

### Wave 0 Gaps
- [ ] `ferroml-python/tests/test_docstrings.py` -- automated docstring completeness check (verify all models have Parameters, Examples, Attributes sections in `__doc__`)
- [ ] Verification script that imports every model and checks `__doc__` is not None and contains required sections

## Sources

### Primary (HIGH confidence)
- `ferroml-python/src/linear.rs` -- Gold standard docstring template (LinearRegression)
- `ferroml-python/src/*.rs` -- All 15 binding files audited for docstring completeness
- `docs/benchmarks.md` -- Existing benchmark page (118 lines, complete)
- `README.md` -- Current state (390 lines, no Known Limitations section)
- `ferroml-core/Cargo.toml` -- ort version confirmed as 2.0.0-rc.11

### Secondary (MEDIUM confidence)
- NumPy docstring convention (numpydoc) -- well-established standard

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, pure documentation work
- Architecture: HIGH -- docstring location and format verified from existing code
- Pitfalls: HIGH -- based on direct audit of current codebase state

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable -- documentation conventions don't change)
