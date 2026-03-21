# Feature Landscape

**Domain:** Production-ready numerical ML library (Rust + Python bindings)
**Researched:** 2026-03-20
**Mode:** Launch hardening for existing 55+ model library

## Table Stakes

Features users expect from a production ML library. Missing = users leave or file bugs immediately.

### Input Validation & Data Handling

| Feature | Why Expected | Complexity | FerroML Status | Notes |
|---------|--------------|------------|----------------|-------|
| NaN/Inf rejection at fit time | sklearn rejects by default via `check_array(ensure_all_finite=True)`. Silent NaN propagation is the #1 production data bug. | Low | MISSING | `validate_fit_input()` with `is_finite()` check on X and y. Apply at trait level so every model inherits it. |
| Empty dataset errors (n=0) | Users pass empty DataFrames during filtering pipelines. Panic = unrecoverable in production. | Low | MISSING (panics) | Early guard: `if x.nrows() == 0 { return Err(FerroError::invalid_input("...")) }` in fit/transform. |
| Shape mismatch errors (X vs y, fit vs predict) | Users transpose matrices, forget columns. Clear error beats cryptic index-out-of-bounds. | Low | PARTIAL (exists but not universal) | Verify `n_features_in_` at predict time matches fit time. sklearn stores this as fitted attribute. |
| NotFitted guard on predict/transform | Calling predict before fit must raise `NotFittedError`, not panic or return garbage. | Low | EXISTS | Already have `FerroError::NotFitted`. Verify every model enforces it. |
| Single-sample handling (n=1) | Edge case in CV, online learning, debugging. Must not panic. | Low | UNTESTED | Add edge case tests for n=1 across all models. Some models (KMeans k>1) should error cleanly. |

### Error Message Quality

| Feature | Why Expected | Complexity | FerroML Status | Notes |
|---------|--------------|------------|---------------|-------|
| Actionable error messages | Error should say what went wrong AND what to do. "Expected 10 features, got 8" not "ShapeMismatch". | Low | PARTIAL | FerroError has structured variants with context fields. Audit messages for actionability. |
| Error context includes parameter names | "Parameter `C` must be positive, got -1.0" not "InvalidInput". | Low | MISSING (no param validation at construction) | Add validation in builder `.with_c()` or in `validate_params()` at fit time. |
| Python exception mapping | Rust FerroError must map to appropriate Python exceptions (ValueError, TypeError, RuntimeError). | Low | EXISTS | PyO3 error translation in `ferroml-python/src/errors.rs`. Verify mapping completeness. |

### Numerical Stability

| Feature | Why Expected | Complexity | FerroML Status | Notes |
|---------|--------------|------------|----------------|-------|
| No silent NaN in predictions | If a model produces NaN predictions, it should error or warn, never silently return NaN arrays. | Med | MISSING | Post-predict check: `if result.iter().any(f64::is_nan) { warn or error }`. |
| Regularization fallbacks for ill-conditioned matrices | Ridge/Lasso must not blow up on collinear features. This is the entire reason regularization exists. | Med | EXISTS (regularization works) | Verify Cholesky fallback adds jitter on failure. Test with condition number > 1e10. |
| Log-sum-exp for probability computations | Naive exp() overflows. All predict_proba must use numerically stable log-sum-exp. | Med | PARTIAL | Verify in LogisticRegression, NaiveBayes, GMM. Known pattern, likely implemented but needs audit. |
| Convergence reporting | Iterative solvers (LogReg, SVM, KMeans) must report if they converged. Users need to know if max_iter was hit. | Low | EXISTS (ConvergenceFailure error) | Currently errors on non-convergence. Consider adding a warning mode (return result + warning) like sklearn's `ConvergenceWarning`. |

### Test Correctness & Cross-Validation

| Feature | Why Expected | Complexity | FerroML Status | Notes |
|---------|--------------|------------|----------------|-------|
| Zero known test failures | Shipping with known failures signals "not ready". Users will find them immediately. | High | 6 FAILURES (TemperatureScaling, IncrementalPCA) | Fix or document as known limitation with explicit API deprecation/removal. |
| Cross-library correctness verification | Claims of "competing with sklearn" require proof. Need numerical agreement on standard datasets. | Med | EXISTS (200+ cross-library tests) | Strong position. Expand to cover all 55+ models at basic level. |
| Deterministic results with seed | `random_state=42` must produce identical results across runs (single-threaded). | Low | EXISTS (except parallel RF) | Document RF parallel non-determinism. All other models should be deterministic with seed. |

### Documentation

| Feature | Why Expected | Complexity | FerroML Status | Notes |
|---------|--------------|------------|----------------|-------|
| Docstrings on all public Python classes | Users discover API via `help()` and IDE tooltips. Missing docstrings = invisible API. | Med | UNKNOWN | Audit all 55+ PyO3 wrappers for `#[pyo3(text_signature = "...")]` and `/// docstring`. |
| Parameter documentation | Each constructor parameter needs type, default, valid range, and what it does. | Med | UNKNOWN | Critical for discoverability. sklearn's biggest strength is parameter docs. |
| Usage examples in docstrings | At least one example per model showing fit/predict. Doctested. | High | PARTIAL (4 tutorial notebooks exist) | Notebooks are good but per-class examples matter more for daily use. |
| Known limitations documented | RandomForest parallel non-determinism, sparse limitations, ONNX RC status. | Low | MISSING | Add to model docstrings and a LIMITATIONS section in README. |

### Serialization & Deployment

| Feature | Why Expected | Complexity | FerroML Status | Notes |
|---------|--------------|------------|----------------|-------|
| Pickle support for Python models | `pickle.dumps(model)` must work. This is how sklearn users save models. | Low | EXISTS | Verify all 55+ models pickle correctly via `test_bindings_correctness.py`. |
| Model versioning on load | Loading a model saved with v0.3 into v0.4 must either work or give a clear error, not silent corruption. | Med | EXISTS (SemanticVersion checking) | `is_compatible_with()` check exists. Verify it's called on every load path. |
| ONNX export | Production deployment path. Users export to ONNX for serving in non-Python environments. | Low | EXISTS (118 models) | RC dependency (ort 2.0.0-rc.11). Document RC status prominently. |

### Performance

| Feature | Why Expected | Complexity | FerroML Status | Notes |
|---------|--------------|------------|----------------|-------|
| Within 3x of sklearn for common models | Users won't adopt a slower library. Rust should be competitive or faster. | High | PARTIAL (some gaps: LinearSVC 9.6x, PCA 13.8x) | PCA and LinearSVC gaps are dealbreakers for those model users. Must fix before launch. |
| Performance benchmarks published | Users want evidence, not claims. Published benchmarks build trust. | Med | EXISTS (86+ Criterion benchmarks) | Need a public-facing benchmark page, not just internal Criterion results. |
| Memory efficiency on large datasets | ML datasets can be 10GB+. Library must not 10x memory usage. | Med | PARTIAL | Streaming I/O exists. Dense-only for most models means sparse data wastes memory. Document limits. |

## Differentiators

Features that set FerroML apart. Not expected, but create competitive advantage.

| Feature | Value Proposition | Complexity | FerroML Status | Notes |
|---------|-------------------|------------|----------------|-------|
| Statistical diagnostics on every model | No other library makes CIs, residual analysis, assumption tests first-class. sklearn requires 3rd-party packages. statsmodels has them but only for linear models. | Already built | EXISTS (StatisticalModel trait) | This IS the differentiator. Must be prominently documented and marketed. |
| Uncertainty quantification (predict_with_uncertainty) | Production ML needs confidence intervals, not just point predictions. sklearn doesn't offer this natively. | Already built | EXISTS (PredictionWithUncertainty) | Unique selling point. Needs prominent examples and docs. |
| Assumption testing built-in | Users can check normality, homoscedasticity, multicollinearity before trusting results. Statsmodels-level rigor in an sklearn-style API. | Already built | EXISTS | Market as "models that tell you when they're wrong". |
| Rust-native performance with Python UX | Best of both worlds: Rust speed + Python usability. Only competitor is polars (different domain). | Already built | EXISTS | Lean into Rust speed advantage where it exists (GaussianNB 4.3x, RF 5x, StandardScaler 9x). |
| Comprehensive ONNX export (118 models) | More models exportable to ONNX than sklearn. Deploy anywhere. | Already built | EXISTS | Unique breadth. Document which models export and which don't. |
| AutoML with statistical rigor | AutoML that uses paired t-tests for model comparison, not just "best CV score". Academically rigorous model selection. | Already built | EXISTS | Differentiated from auto-sklearn/FLAML which lack statistical testing. |
| Feature schema validation | Declare expected input schema at training time, validate at prediction time. Catches data drift. | Already built | EXISTS (FeatureSchema) | Production-oriented feature. Market for MLOps use cases. |
| Explainability built-in (SHAP, PDP, ICE) | TreeSHAP, KernelSHAP, permutation importance, PDP, ICE without separate library. sklearn added partial support only recently. | Already built | EXISTS | Reduces dependency count for users. |

## Anti-Features

Features to explicitly NOT build for this launch milestone.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Full sparse algorithm support | Massive scope (every model needs sparse variant). Dense conversion works for moderate data. | Document which models accept sparse natively (CountVectorizer, TF-IDF). Recommend dense conversion for others. Sparse support is a v0.5+ feature. |
| Deterministic parallel RandomForest | Requires replacing rayon's work-stealing scheduler. Unknown performance cost, no user has requested it. | Document: "Use `n_jobs=1` for reproducible results." This is an acceptable and widely-understood tradeoff. |
| GPU training for all models | GPU only helps for large-scale deep learning. Classical ML on GPU is marginal gain for enormous complexity. | Keep optional GPU shaders for specific bottlenecks. Don't promise GPU acceleration broadly. |
| WASM/mobile targets | Different compilation targets, different constraints. Not where ML library users are. | Focus on Linux/macOS/Windows x86_64. Add WASM in a future milestone if demanded. |
| Streaming/online learning for all models | Only a few models (SGD, NaiveBayes) naturally support online learning. Forcing it on batch algorithms is wrong. | Existing `IncrementalModel` trait + `partial_fit()` on supported models is correct scoping. |
| Custom loss functions / user-defined models | Plugin architecture is complex and rarely used. Power users write Rust directly. | Keep search_space() for HPO integration. Don't build a plugin system. |
| DataFrame input (pandas/polars native) | Adds dependency, conversion overhead, API surface. NumPy arrays are the lingua franca. | Accept numpy arrays. Users convert from DataFrame to numpy themselves (standard practice). Polars optional integration already exists for data loading. |
| Distributed training | Completely different architecture (message passing, data sharding). Out of scope for a single-machine library. | Single-machine focus. Users who need distributed use Spark MLlib or Dask-ML. |
| scikit-learn estimator API wrapper | Making FerroML models pass `check_estimator` would require mimicking sklearn's exact API surface. | Follow sklearn conventions (fit/predict/transform) but don't try to be a drop-in replacement. Different language, different constraints. |

## Feature Dependencies

```
NaN/Inf validation --> Empty data handling (same validation layer)
                   --> Parameter validation (same validation framework)

Parameter validation at construction --> Better error messages
                                     --> 4 skipped Python tests pass

Fix 6 test failures --> Clean test suite --> Publishable quality signal

PCA faer SVD --> Performance benchmarks update
LinearSVC shrinking --> Performance benchmarks update
OLS/Ridge Cholesky --> Performance benchmarks update

Docstrings audit --> Usage examples --> Published API reference
                 --> Parameter docs

Performance fixes --> Published benchmarks --> Marketing material
```

## MVP Recommendation

### Must fix before launch (blocks credibility):

1. **NaN/Inf input validation** - Silent corruption is unacceptable. Every competitor rejects NaN at fit time. This is the single most important robustness feature.
2. **Empty data handling** - Panics in a library are unacceptable. Convert to clean FerroError.
3. **Fix or remove 6 failing tests** - Either fix TemperatureScaling/IncrementalPCA or explicitly mark as experimental/remove from public API. Zero known failures at launch.
4. **Parameter validation** - Users will immediately try `SVC(C=-1)` and get a confusing error at fit time. Validate at construction.
5. **PCA performance** (13.8x gap) - This is a commonly-used algorithm. 13.8x slower than sklearn is a dealbreaker. faer thin SVD is already a dependency.
6. **LinearSVC performance** (9.6x gap) - Linear SVM on large datasets is a core use case. Must be competitive.
7. **Unwrap audit in critical paths** - Replace unwrap/expect in fit/predict hot paths with proper error handling. Panics = crashes in production Python code.

### Should do before launch (improves adoption):

8. **Docstring audit** - Ensure all 55+ Python models have complete parameter documentation.
9. **Known limitations documentation** - Honest docs build trust. Document RF non-determinism, ONNX RC status, sparse limitations.
10. **Published performance benchmarks** - Turn Criterion results into a public comparison page.

### Defer to post-launch:

- Full sparse algorithm support
- SVC performance tuning (3-5x is acceptable, not great)
- HistGBT performance (2.6x vs XGBoost acceptable for initial launch)
- Convergence warning mode (error is fine for now)
- Additional cross-library tests beyond the existing 200+

## Sources

- [scikit-learn check_array documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_array.html) - Input validation reference (ensure_all_finite parameter)
- [scikit-learn check_estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html) - Estimator compliance test suite
- [scikit-learn developer guidelines](https://scikit-learn.org/stable/developers/develop.html) - API design patterns, validation conventions
- [scikit-learn check_is_fitted](https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html) - Fitted state validation pattern
- [Microsoft ML Production Checklist](https://microsoft.github.io/code-with-engineering-playbook/machine-learning/ml-model-checklist/) - Production readiness requirements
- [Microsoft ML Fundamentals Checklist](https://microsoft.github.io/code-with-engineering-playbook/machine-learning/ml-fundamentals-checklist/) - Schema, monitoring, SLA requirements
- FerroML codebase audit: `.planning/codebase/CONCERNS.md`, `.planning/codebase/ARCHITECTURE.md`
- FerroML project context: `.planning/PROJECT.md`
