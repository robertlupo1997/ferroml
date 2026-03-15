# Plan U: sklearn API Parity (score, partial_fit, decision_function) + v0.3.0 Release

**Date:** 2026-03-15
**Status:** COMPLETE
**Depends on:** Plan T (complete)

---

## Overview

Close the top-3 sklearn API gaps identified by the feature parity scorecard, then release v0.3.0.
The `Model` trait already has a default `score()` implementation, and most models already compute
`decision_function` and `partial_fit` internally — the work is primarily **exposing** existing
functionality to the Python API.

---

## Current State

- **score()**: Trait method exists on `Model` (mod.rs:229-251) with default accuracy implementation.
  MLPRegressor overrides it with R². Only 2 of 59 sklearn-equivalent models expose it in Python.
- **partial_fit**: 7 Rust models have it (SGD×2, NB×4, IncrementalPCA). Only CategoricalNB exposes
  it in Python. `IncrementalModel` trait exists (traits.rs:28-42).
- **decision_function**: 5 Rust models have it (LogReg, SVC, LinearSVC, SGDClassifier,
  RidgeClassifier). 4 expose it in Python (SVC, LinearSVC, LinearSVR, SVR).
- **Version**: v0.2.0 released 2026-03-11. CHANGELOG.md has `[Unreleased]` section.
- **CI/CD**: Fully configured — tag push auto-publishes to crates.io, PyPI, and GitHub Releases.

## Desired End State

- `score(X, y)` available on all 59+ sklearn-equivalent models in Python (R² for regressors, accuracy for classifiers)
- `partial_fit` available on all 16 sklearn-equivalent models in Python
- `decision_function` available on all 13 applicable classifiers in Python
- v0.3.0 released with updated CHANGELOG, version bump, and marketing materials

---

## Phase U.1 — Add `score(X, y)` to All Models

**Overview**: The `Model` trait already provides a default `score()` that returns accuracy.
For regressors, we need R² instead. The main work is (a) overriding `score()` for all
regressor types in Rust to return R², and (b) adding the `score()` Python binding to every model.

### U.1a — Rust: Override score() for Regression Models

The default `Model::score()` returns accuracy (fraction of exact matches), which is correct
for classifiers but wrong for regressors. Override it on all regressors to return R².

**Pattern** (reuse `crate::metrics::r2_score` from metrics/regression.rs:114-137):
```rust
fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
    let predictions = self.predict(x)?;
    crate::metrics::r2_score(y, &predictions)
}
```

**Models to override** (~20 regressors):

| Model | File | Notes |
|-------|------|-------|
| LinearRegression | models/linear.rs | Has `r_squared()` already — add `score()` override |
| RidgeRegression | models/regularized.rs | Same pattern |
| LassoRegression | models/regularized.rs | Same pattern |
| ElasticNet | models/regularized.rs | Same pattern |
| RidgeCV | models/regularized.rs | Delegate to inner Ridge |
| LassoCV | models/regularized.rs | Delegate to inner Lasso |
| ElasticNetCV | models/regularized.rs | Delegate to inner ElasticNet |
| RobustRegression | models/robust.rs | Same pattern |
| QuantileRegression | models/quantile.rs | Same pattern |
| IsotonicRegression | models/isotonic.rs | Same pattern |
| DecisionTreeRegressor | models/tree.rs | Same pattern |
| RandomForestRegressor | models/forest.rs | Same pattern |
| ExtraTreesRegressor | models/extra_trees.rs | Same pattern |
| GradientBoostingRegressor | models/boosting.rs | Same pattern |
| HistGradientBoostingRegressor | models/hist_boosting.rs | Same pattern |
| AdaBoostRegressor | models/adaboost.rs | Same pattern |
| SVR | models/svm.rs | Same pattern |
| LinearSVR | models/svm.rs | Same pattern |
| KNeighborsRegressor | models/knn.rs | Same pattern |
| GaussianProcessRegressor | models/gaussian_process.rs | Same pattern |
| SparseGPRegressor | models/gaussian_process.rs | Same pattern |
| SVGPRegressor | models/gaussian_process.rs | Same pattern |
| BaggingRegressor | ensemble/bagging.rs | Same pattern |

**Clustering models** — special case:
- KMeans: return negative inertia (sklearn convention) — `score()` is `-inertia_`
- GaussianMixture: return mean log-likelihood (already has this as `score()`)
- DBSCAN, HDBSCAN, AgglomerativeClustering: sklearn doesn't have `score()` on these

### U.1b — Python: Expose score() on All Model Bindings

Add `score(x, y)` method to every PyO3 `#[pymethods]` block. Pattern:

```rust
/// Evaluate the model on test data.
///
/// For classifiers, returns accuracy. For regressors, returns R².
///
/// Parameters
/// ----------
/// x : numpy.ndarray of shape (n_samples, n_features)
///     Test features.
/// y : numpy.ndarray of shape (n_samples,)
///     True labels or values.
///
/// Returns
/// -------
/// float
///     Score (accuracy or R²).
fn score<'py>(
    &self,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = to_owned_array_1d(y);
    self.inner
        .score(&x_arr, &y_arr)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
```

**Files to edit** (14 Python binding files):

| File | Models |
|------|--------|
| ferroml-python/src/linear.rs | LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, RobustRegression, QuantileRegression, IsotonicRegression, RidgeClassifier, Perceptron |
| ferroml-python/src/logistic.rs | LogisticRegression |
| ferroml-python/src/trees.rs | DT×2, RF×2, GB×2, HistGBT×2 |
| ferroml-python/src/ensemble.rs | ExtraTrees×2, AdaBoost×2, Bagging×2, SGD×2, PassiveAggressive, Voting×2, Stacking×2 |
| ferroml-python/src/svm.rs | SVC, SVR, LinearSVC, LinearSVR |
| ferroml-python/src/naive_bayes.rs | GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB |
| ferroml-python/src/neural.rs | MLPClassifier (MLPRegressor already has it) |
| ferroml-python/src/clustering.rs | KMeans, GaussianMixture |
| ferroml-python/src/neighbors.rs | KNeighborsClassifier, KNeighborsRegressor, NearestCentroid |
| ferroml-python/src/gaussian_process.rs | GP×5 |
| ferroml-python/src/decomposition.rs | PCA, IncrementalPCA, TruncatedSVD, LDA, QDA, FactorAnalysis, TSNE |
| ferroml-python/src/anomaly.rs | IsolationForest, LocalOutlierFactor |
| ferroml-python/src/calibration.rs | TemperatureScalingCalibrator |
| ferroml-python/src/multioutput.rs | MultiOutputRegressor, MultiOutputClassifier |

**Estimated scope**: ~20 Rust score overrides + ~57 Python binding additions (boilerplate-heavy but mechanical)

### U.1c — Tests

- Add Python test `test_score_all_models.py` that verifies every model has `score()` and returns reasonable values
- Test: classifier score in [0, 1], regressor R² in [-inf, 1]
- Test: regressor score matches `sklearn.metrics.r2_score` on same predictions

**Success Criteria**:
- [ ] `python -c "import ferroml; m = ferroml.linear.LinearRegression(); m.fit(X,y); print(m.score(X,y))"` works
- [ ] All 57 previously-missing models now have score()
- [ ] Re-run feature_parity_scorecard.py — score gap = 0

---

## Phase U.2 — Add `partial_fit` to Missing Models

**Overview**: 7 models have `partial_fit` in Rust, but only 1 (CategoricalNB) exposes it in Python.
The main work is (a) exposing existing Rust implementations in Python, (b) adding `partial_fit`
to Perceptron and PassiveAggressiveClassifier in Rust, and (c) adding Python bindings.

### U.2a — Expose Existing Rust partial_fit in Python

These models already have `partial_fit` in Rust — just need Python bindings:

| Model | Rust Location | Python File |
|-------|--------------|-------------|
| SGDClassifier | models/sgd.rs:559-703 (IncrementalModel trait) | ensemble.rs |
| SGDRegressor | models/sgd.rs:1039+ | ensemble.rs |
| GaussianNB | models/naive_bayes.rs:271-308 | naive_bayes.rs |
| MultinomialNB | models/naive_bayes.rs:836-927 | naive_bayes.rs |
| BernoulliNB | models/naive_bayes.rs:1346+ | naive_bayes.rs |
| IncrementalPCA | decomposition/pca.rs:833+ | decomposition.rs |

**Python binding pattern** (from CategoricalNB):
```rust
#[pyo3(signature = (x, y, classes=None))]
fn partial_fit<'py>(
    mut slf: PyRefMut<'py, Self>,
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: &Bound<'py, PyAny>,
    classes: Option<Vec<f64>>,
) -> PyResult<PyRefMut<'py, Self>> {
    let x_arr = to_owned_array_2d(x);
    let y_arr = py_array_to_f64_1d(py, y)?;
    slf.inner
        .partial_fit(&x_arr, &y_arr, classes)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(slf)
}
```

### U.2b — Implement partial_fit for Perceptron and PassiveAggressiveClassifier

**Perceptron** (models/sgd.rs:1104-1175):
- Already wraps SGDClassifier internally
- Just delegate: `self.inner.partial_fit_with_classes(x, y, classes)`

**PassiveAggressiveClassifier** (models/sgd.rs:1189-1340):
- Stores `coef`, `intercept`, `classes`
- PA-I/PA-II update is already per-sample — restructure to support batches
- Pattern: on first call, initialize from `classes`; on subsequent calls, run PA update over batch

### U.2c — Tests

- Add Python test `test_partial_fit.py`:
  - Verify partial_fit on batches ≈ fit on full data (within tolerance)
  - Verify partial_fit preserves state across calls
  - Verify classes parameter validation
  - Test each of: SGD×2, NB×4, Perceptron, PassiveAggressive, IncrementalPCA

**Success Criteria**:
- [ ] All 16 sklearn-equivalent models have partial_fit in Python
- [ ] `partial_fit` on 3 equal batches ≈ `fit` on full data (R² within 0.1)
- [ ] Re-run scorecard — partial_fit gap = 0

---

## Phase U.3 — Add `decision_function` to Missing Classifiers

**Overview**: 5 models have decision_function in Rust. 9 more classifiers need it.
Most already compute the raw scores internally during `predict_proba()` — we just need
to extract them before the sigmoid/softmax step.

### U.3a — Implement decision_function in Rust

**Quick wins (delegate to existing):**

| Model | File | Implementation |
|-------|------|---------------|
| Perceptron | models/sgd.rs | `self.inner.decision_function(x)` (delegates to SGDClassifier) |

**Extract raw scores from predict_proba:**

| Model | File | Raw Score Location |
|-------|------|-------------------|
| GradientBoostingClassifier | models/boosting.rs:1134-1207 | `raw_predictions` before sigmoid/softmax |
| HistGradientBoostingClassifier | models/hist_boosting.rs:1669-1687 | `raw_predictions` before sigmoid/softmax |
| AdaBoostClassifier | models/adaboost.rs:236-245 | `class_scores` (weighted votes) |
| ExtraTreesClassifier | models/extra_trees.rs:210-237 | Pre-normalization tree vote counts |
| RandomForestClassifier | models/forest.rs:377-450 | Pre-normalization average leaf counts |
| DecisionTreeClassifier | models/tree.rs:563+ | Leaf node class counts |

**New coef-based implementation:**

| Model | File | Implementation |
|-------|------|---------------|
| PassiveAggressiveClassifier | models/sgd.rs | `X @ coef.T + intercept` (same as SGDClassifier) |

**Return type convention** (matching sklearn):
- Binary: `Array1<f64>` of shape `(n_samples,)` — distance to decision boundary
- Multiclass: `Array2<f64>` of shape `(n_samples, n_classes)` — per-class scores

### U.3b — Expose in Python Bindings

Add to `#[pymethods]` blocks for all 13 classifiers. Pattern:
```rust
fn decision_function<'py>(
    &self,
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<PyObject> {
    let x_arr = to_owned_array_2d(x);
    let result = self.inner
        .decision_function(&x_arr)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Return Array1 or Array2 depending on shape
    Ok(result.into_pyarray(py).into())
}
```

**Files**: linear.rs, logistic.rs, trees.rs, ensemble.rs, svm.rs (some already done)

### U.3c — Tests

- Add Python test `test_decision_function.py`:
  - Verify output shape (n_samples,) for binary, (n_samples, n_classes) for multiclass
  - Verify sign of decision_function matches predicted class for linear models
  - Verify decision_function output is consistent with predict_proba (monotonic relationship)

**Success Criteria**:
- [ ] All 13 classifiers have decision_function in Python
- [ ] Binary classifiers: `np.sign(clf.decision_function(X)) == clf.predict(X)` (approximately)
- [ ] Re-run scorecard — decision_function gap = 0

---

## Phase U.4 — v0.3.0 Release

**Overview**: Version bump, changelog, marketing materials, tag and release.

### U.4a — Version Bump

| File | Change |
|------|--------|
| Cargo.toml (workspace) | `version = "0.2.0"` → `"0.3.0"` |
| ferroml-python/pyproject.toml | `version = "0.2.0"` → `"0.3.0"` |
| pyproject.toml classifiers | `"Development Status :: 3 - Alpha"` → `"4 - Beta"` |

### U.4b — CHANGELOG.md Update

Convert `[Unreleased]` section to `[0.3.0]` with date. Add entries for:

**Added:**
- `score(X, y)` on all 59+ models (R² for regressors, accuracy for classifiers)
- `partial_fit` on 16 models (SGD, NaiveBayes, Perceptron, PassiveAggressive, IncrementalPCA)
- `decision_function` on 13 classifiers
- `warm_start` on 13 models (from 7 in v0.2.0)
- Feature parity scorecard script and documentation

**Performance:**
- HistGradientBoosting: 6x faster (1.9x vs sklearn, was 12x slower)
- KMeans: 8x faster (now 2x faster than sklearn)
- LogisticRegression: 1.4x faster (1.76x vs sklearn, was 2.5x slower)

### U.4c — Update Marketing Docs

1. Re-run `scripts/benchmark_cross_library.py` → update docs/cross-library-benchmark.md
2. Re-run `scripts/feature_parity_scorecard.py` → update docs/feature-parity-scorecard.md
3. Update docs/ROADMAP.md with v0.3.0 status
4. Update README.md highlights section

### U.4d — Tag and Release

```bash
git tag v0.3.0
git push origin v0.3.0
# CI/CD auto-triggers: GitHub Release, crates.io, PyPI
```

**Success Criteria**:
- [ ] `cargo build --release` succeeds with version 0.3.0
- [ ] `maturin build --release` succeeds
- [ ] All 3,160+ Rust tests pass
- [ ] All 1,855+ Python tests pass
- [ ] Feature parity scorecard shows 0 gaps for score, partial_fit, decision_function
- [ ] GitHub Release created with changelog
- [ ] PyPI package available as `ferroml==0.3.0`

---

## Execution Order

| Phase | Priority | Effort | Dependencies |
|-------|----------|--------|-------------|
| U.1   | High     | Medium | None |
| U.2   | High     | Medium | None |
| U.3   | High     | Medium | None |
| U.4   | High     | Small  | U.1, U.2, U.3 |

**Recommended approach**: U.1 first (biggest impact — 57 models), then U.2 and U.3 in parallel,
then U.4 last. U.1/U.2/U.3 are independent and could run as parallel agent tasks.

**Estimated total effort**: Medium — mostly boilerplate Python binding additions with some
Rust `score()` overrides and `decision_function` extractions.

---

## Risks & Mitigations

1. **Risk**: Score override returns wrong metric type (accuracy vs R²)
   **Mitigation**: Comprehensive test that checks each model returns the correct metric

2. **Risk**: partial_fit state corruption across batches
   **Mitigation**: Existing Rust tests in testing/incremental.rs; add Python equivalence tests

3. **Risk**: decision_function sign convention differs from sklearn
   **Mitigation**: Compare against sklearn on same data in cross-library tests

4. **Risk**: Python binding boilerplate errors (copy-paste bugs)
   **Mitigation**: Test every model in test_score_all_models.py before release
