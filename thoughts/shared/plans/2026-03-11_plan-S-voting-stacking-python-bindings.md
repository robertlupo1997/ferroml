# Plan S: Voting & Stacking Ensemble Python Bindings

## Overview

Expose the four existing Rust ensemble meta-learners (VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor) to Python via PyO3 bindings. The Rust implementations are complete and tested; this plan covers only the Python binding layer, the `__init__.py` re-exports, and comprehensive Python tests (including correctness tests against sklearn fixtures).

## Current State

### Rust (complete, no changes needed)

| Model | File | Estimator Trait | Impls |
|---|---|---|---|
| `VotingClassifier` | `ferroml-core/src/ensemble/voting.rs:127` | `VotingClassifierEstimator` | LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GaussianNB, MultinomialNB, BernoulliNB, KNeighborsClassifier, SVC, GradientBoostingClassifier, HistGradientBoostingClassifier (10 total) |
| `VotingRegressor` | `ferroml-core/src/ensemble/voting.rs:507` | `VotingRegressorEstimator` | LinearRegression, RidgeRegression, LassoRegression, ElasticNet, DecisionTreeRegressor, RandomForestRegressor, KNeighborsRegressor, SVR, GradientBoostingRegressor, HistGradientBoostingRegressor (10 total) |
| `StackingClassifier` | `ferroml-core/src/ensemble/stacking.rs:105` | `VotingClassifierEstimator` (base) + `Model` (final) | Same 10 classifiers as base; any Model as final |
| `StackingRegressor` | `ferroml-core/src/ensemble/stacking.rs:566` | `VotingRegressorEstimator` (base) + `Model` (final) | Same 10 regressors as base; any Model as final |

Rust tests: inline `#[cfg(test)]` in `voting.rs` and `stacking.rs`, plus `testing/ensemble_advanced.rs`.

### Python (bindings missing)

- `ferroml-python/src/ensemble.rs` exposes: ExtraTrees{Classifier,Regressor}, AdaBoost{Classifier,Regressor}, SGD{Classifier,Regressor}, PassiveAggressiveClassifier, Bagging{Classifier,Regressor}
- `ferroml-python/python/ferroml/ensemble/__init__.py` re-exports those 9 classes
- **VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor have NO Python bindings**

### Design Challenge

The Rust APIs accept `Vec<(String, Box<dyn VotingClassifierEstimator>)>` — trait objects that cannot cross the PyO3 boundary. The established pattern in this codebase is **factory methods** (used by BaggingClassifier/BaggingRegressor at `ensemble.rs:833+`) or **string-based dispatch** (used by MultiOutputRegressor at `multioutput.rs:48+`).

For voting/stacking, users need to specify **multiple heterogeneous estimators** with names. The best pattern is a **string-list-based constructor** where users pass estimator specifications as a list of `(name, estimator_type)` tuples, plus a **fluent builder** for additional config. This is more Pythonic than the factory approach for multi-estimator ensembles.

## Desired End State

- `from ferroml.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor`
- All four classes support the full parameter sets from their Rust counterparts
- Users specify base estimators via string names (matching sklearn-style API)
- Comprehensive Python test suite (~60+ tests) including sklearn fixture comparisons
- All existing tests continue passing

---

## Implementation Phases

### Phase S.1: VotingClassifier & VotingRegressor Python Bindings

**Overview**: Add PyO3 wrappers for both voting ensembles using a string-based estimator specification pattern.

**Changes Required**:

1. **File**: `ferroml-python/src/ensemble.rs` (MODIFY — append after BaggingRegressor section, before `register_ensemble_module`)

   Add a helper function to construct estimator lists from Python-provided specs:

   ```rust
   // =============================================================================
   // Estimator construction helpers (shared by Voting and Stacking)
   // =============================================================================

   /// Build a Vec of named classifier estimators from Python string specs.
   /// Supported estimator types: "logistic_regression", "decision_tree",
   /// "random_forest", "gaussian_nb", "multinomial_nb", "bernoulli_nb",
   /// "knn", "svc", "gradient_boosting", "hist_gradient_boosting"
   fn build_classifier_estimators(
       estimators: Vec<(String, String)>,
   ) -> PyResult<Vec<(String, Box<dyn VotingClassifierEstimator>)>> {
       // ... match on estimator type string, return boxed estimator
   }

   /// Build a Vec of named regressor estimators from Python string specs.
   /// Supported: "linear_regression", "ridge", "lasso", "elastic_net",
   /// "decision_tree", "random_forest", "knn", "svr",
   /// "gradient_boosting", "hist_gradient_boosting"
   fn build_regressor_estimators(
       estimators: Vec<(String, String)>,
   ) -> PyResult<Vec<(String, Box<dyn VotingRegressorEstimator>)>> {
       // ... match on estimator type string, return boxed estimator
   }
   ```

   Add `PyVotingClassifier`:

   ```rust
   #[pyclass(name = "VotingClassifier", module = "ferroml.ensemble")]
   pub struct PyVotingClassifier {
       inner: VotingClassifier,
   }

   #[pymethods]
   impl PyVotingClassifier {
       #[new]
       #[pyo3(signature = (estimators, voting="hard", weights=None))]
       fn new(
           estimators: Vec<(String, String)>,  // [("lr", "logistic_regression"), ("dt", "decision_tree")]
           voting: &str,                        // "hard" or "soft"
           weights: Option<Vec<f64>>,
       ) -> PyResult<Self> { ... }

       fn fit(...) -> PyResult<PyRefMut<Self>> { ... }
       fn predict(...) -> PyResult<Py<PyArray1<f64>>> { ... }
       fn predict_proba(...) -> PyResult<Py<PyArray2<f64>>> { ... }  // soft voting only
       fn __repr__(&self) -> String { ... }
       fn __getstate__(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> { ... }
       fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> { ... }

       // Properties
       #[getter] fn estimator_names(&self) -> Vec<String> { ... }
       #[getter] fn voting_method(&self) -> String { ... }
   }
   ```

   Add `PyVotingRegressor`:

   ```rust
   #[pyclass(name = "VotingRegressor", module = "ferroml.ensemble")]
   pub struct PyVotingRegressor {
       inner: VotingRegressor,
   }

   #[pymethods]
   impl PyVotingRegressor {
       #[new]
       #[pyo3(signature = (estimators, weights=None))]
       fn new(
           estimators: Vec<(String, String)>,
           weights: Option<Vec<f64>>,
       ) -> PyResult<Self> { ... }

       fn fit(...) -> PyResult<PyRefMut<Self>> { ... }
       fn predict(...) -> PyResult<Py<PyArray1<f64>>> { ... }
       fn __repr__(&self) -> String { ... }
       fn __getstate__(...) / fn __setstate__(...)

       #[getter] fn estimator_names(&self) -> Vec<String> { ... }
   }
   ```

2. **File**: `ferroml-python/src/ensemble.rs` — update `register_ensemble_module` to add the new classes:
   ```rust
   m.add_class::<PyVotingClassifier>()?;
   m.add_class::<PyVotingRegressor>()?;
   ```

3. **File**: `ferroml-python/src/ensemble.rs` — add required imports at the top:
   ```rust
   use ferroml_core::ensemble::{VotingClassifier, VotingRegressor, VotingMethod};
   use ferroml_core::models::{
       // Add any missing: MultinomialNB, BernoulliNB, LassoRegression, ElasticNet
   };
   ```

4. **File**: `ferroml-python/python/ferroml/ensemble/__init__.py` — add re-exports:
   ```python
   VotingClassifier = _native.ensemble.VotingClassifier
   VotingRegressor = _native.ensemble.VotingRegressor
   ```
   Update `__all__` and module docstring.

**Success Criteria**:
- [ ] Automated: `maturin develop --release -m ferroml-python/Cargo.toml && python -c "from ferroml.ensemble import VotingClassifier, VotingRegressor; print('OK')"`
- [ ] Automated: `cargo fmt --all && cargo clippy -p ferroml-python --no-deps`

---

### Phase S.2: StackingClassifier & StackingRegressor Python Bindings

**Overview**: Add PyO3 wrappers for both stacking ensembles. Stacking has additional parameters (final_estimator, cv, passthrough, stack_method) beyond the base estimator list.

**Changes Required**:

1. **File**: `ferroml-python/src/ensemble.rs` (MODIFY — append after VotingRegressor section)

   Add helper to construct the final estimator (meta-learner) from a string:

   ```rust
   /// Build a final estimator (meta-learner) from a string name.
   /// Classifier finals: "logistic_regression" (default), "decision_tree", "random_forest", "gaussian_nb", "knn"
   /// Regressor finals: "linear_regression" (default), "ridge", "lasso", "decision_tree", "random_forest", "knn"
   fn build_classifier_final_estimator(name: &str) -> PyResult<Box<dyn Model>> { ... }
   fn build_regressor_final_estimator(name: &str) -> PyResult<Box<dyn Model>> { ... }
   ```

   Add `PyStackingClassifier`:

   ```rust
   #[pyclass(name = "StackingClassifier", module = "ferroml.ensemble")]
   pub struct PyStackingClassifier {
       inner: StackingClassifier,
   }

   #[pymethods]
   impl PyStackingClassifier {
       #[new]
       #[pyo3(signature = (
           estimators,
           final_estimator="logistic_regression",
           cv=5,
           stack_method="predict_proba",
           passthrough=false,
       ))]
       fn new(
           estimators: Vec<(String, String)>,
           final_estimator: &str,
           cv: usize,
           stack_method: &str,    // "predict" or "predict_proba"
           passthrough: bool,
       ) -> PyResult<Self> { ... }

       fn fit(...) -> PyResult<PyRefMut<Self>> { ... }
       fn predict(...) -> PyResult<Py<PyArray1<f64>>> { ... }
       fn predict_proba(...) -> PyResult<Py<PyArray2<f64>>> { ... }
       fn __repr__(&self) -> String { ... }
       fn __getstate__(...) / fn __setstate__(...)

       #[getter] fn estimator_names(&self) -> Vec<String> { ... }
       #[getter] fn stack_method_name(&self) -> String { ... }
       #[getter] fn passthrough(&self) -> bool { ... }
   }
   ```

   Add `PyStackingRegressor`:

   ```rust
   #[pyclass(name = "StackingRegressor", module = "ferroml.ensemble")]
   pub struct PyStackingRegressor {
       inner: StackingRegressor,
   }

   #[pymethods]
   impl PyStackingRegressor {
       #[new]
       #[pyo3(signature = (
           estimators,
           final_estimator="linear_regression",
           cv=5,
           passthrough=false,
       ))]
       fn new(
           estimators: Vec<(String, String)>,
           final_estimator: &str,
           cv: usize,
           passthrough: bool,
       ) -> PyResult<Self> { ... }

       fn fit(...) -> PyResult<PyRefMut<Self>> { ... }
       fn predict(...) -> PyResult<Py<PyArray1<f64>>> { ... }
       fn __repr__(&self) -> String { ... }
       fn __getstate__(...) / fn __setstate__(...)

       #[getter] fn estimator_names(&self) -> Vec<String> { ... }
       #[getter] fn passthrough(&self) -> bool { ... }
   }
   ```

2. **File**: `ferroml-python/src/ensemble.rs` — update `register_ensemble_module`:
   ```rust
   m.add_class::<PyStackingClassifier>()?;
   m.add_class::<PyStackingRegressor>()?;
   ```

3. **File**: `ferroml-python/src/ensemble.rs` — add required imports:
   ```rust
   use ferroml_core::ensemble::{StackingClassifier, StackingRegressor, StackMethod};
   ```

4. **File**: `ferroml-python/python/ferroml/ensemble/__init__.py` — add re-exports:
   ```python
   StackingClassifier = _native.ensemble.StackingClassifier
   StackingRegressor = _native.ensemble.StackingRegressor
   ```
   Update `__all__` and module docstring.

**Success Criteria**:
- [ ] Automated: `maturin develop --release -m ferroml-python/Cargo.toml && python -c "from ferroml.ensemble import StackingClassifier, StackingRegressor; print('OK')"`
- [ ] Automated: `cargo fmt --all && cargo clippy -p ferroml-python --no-deps`

---

### Phase S.3: Python Unit Tests

**Overview**: Comprehensive test suite covering all four new ensemble bindings. Tests follow the pattern in `ferroml-python/tests/test_ensemble.py` (which tests ExtraTrees, AdaBoost, SGD, PA).

**Changes Required**:

1. **File**: `ferroml-python/tests/test_voting_stacking.py` (NEW, ~400 lines)

   ```python
   """Test FerroML Voting and Stacking ensemble models."""

   import numpy as np
   import pytest
   from ferroml.ensemble import (
       VotingClassifier, VotingRegressor,
       StackingClassifier, StackingRegressor,
   )
   ```

   **VotingClassifier tests (~15 tests)**:
   - `test_hard_voting_basic` — fit/predict, verify predictions are valid class labels
   - `test_soft_voting_basic` — soft voting, verify predictions
   - `test_soft_voting_predict_proba` — verify probability output shape and sums to 1
   - `test_weighted_voting` — different weights produce different results
   - `test_voting_binary_accuracy` — accuracy > 0.7 on easy binary data
   - `test_voting_multiclass` — 3+ classes, verify all predicted classes are valid
   - `test_voting_multiple_estimators` — 3+ estimators (LR, DT, GNB)
   - `test_voting_estimator_names` — verify estimator_names getter
   - `test_voting_unfitted_raises` — predict before fit raises error
   - `test_voting_invalid_estimator_raises` — unknown estimator string raises ValueError
   - `test_voting_repr` — __repr__ contains class name
   - `test_voting_pickle_roundtrip` — pickle.dumps/loads, predict matches
   - `test_voting_single_estimator` — works with just one estimator
   - `test_voting_mismatched_weights_raises` — wrong number of weights raises error
   - `test_voting_hard_vs_soft_differ` — hard and soft voting can produce different results

   **VotingRegressor tests (~10 tests)**:
   - `test_regression_basic` — fit/predict, verify predictions are finite
   - `test_regression_weighted` — weighted averaging
   - `test_regression_r2_positive` — R^2 > 0 on easy data
   - `test_regression_multiple_estimators` — 3+ regressors
   - `test_regression_estimator_names` — getter
   - `test_regression_unfitted_raises` — predict before fit
   - `test_regression_invalid_estimator_raises` — unknown string
   - `test_regression_repr` — __repr__
   - `test_regression_pickle_roundtrip` — serialization
   - `test_regression_single_estimator` — works with one

   **StackingClassifier tests (~15 tests)**:
   - `test_stacking_clf_basic` — fit/predict, verify valid classes
   - `test_stacking_clf_predict_proba` — probability output
   - `test_stacking_clf_with_passthrough` — passthrough=True
   - `test_stacking_clf_predict_method` — stack_method="predict" vs "predict_proba"
   - `test_stacking_clf_custom_final` — non-default final estimator
   - `test_stacking_clf_cv_folds` — cv=3 vs cv=10
   - `test_stacking_clf_binary_accuracy` — accuracy > 0.6 on easy binary data
   - `test_stacking_clf_multiclass` — 3+ classes
   - `test_stacking_clf_multiple_estimators` — 3+ base estimators
   - `test_stacking_clf_estimator_names` — getter
   - `test_stacking_clf_unfitted_raises` — predict before fit
   - `test_stacking_clf_invalid_estimator_raises` — unknown string
   - `test_stacking_clf_repr` — __repr__
   - `test_stacking_clf_pickle_roundtrip` — serialization
   - `test_stacking_clf_vs_single_estimator` — stacking should be >= single estimator (on average)

   **StackingRegressor tests (~12 tests)**:
   - `test_stacking_reg_basic` — fit/predict, verify finite
   - `test_stacking_reg_with_passthrough` — passthrough=True
   - `test_stacking_reg_custom_final` — non-default final estimator (e.g., "ridge")
   - `test_stacking_reg_cv_folds` — cv=3 vs cv=10
   - `test_stacking_reg_r2_positive` — R^2 > 0 on easy data
   - `test_stacking_reg_multiple_estimators` — 3+ base estimators
   - `test_stacking_reg_estimator_names` — getter
   - `test_stacking_reg_unfitted_raises` — predict before fit
   - `test_stacking_reg_invalid_estimator_raises` — unknown string
   - `test_stacking_reg_repr` — __repr__
   - `test_stacking_reg_pickle_roundtrip` — serialization
   - `test_stacking_reg_predictions_finite` — all predictions finite on reasonable data

**Success Criteria**:
- [ ] Automated: `cd ferroml-python && python -m pytest tests/test_voting_stacking.py -v` — all ~52 tests pass

---

### Phase S.4: Sklearn Correctness Comparison Tests

**Overview**: Generate sklearn fixtures and verify FerroML produces comparable results. These are the "gold standard" correctness tests that compare against sklearn's VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor.

**Changes Required**:

1. **File**: `ferroml-python/tests/fixtures/generate_voting_stacking_fixtures.py` (NEW, ~200 lines)

   Generates JSON fixture files with sklearn outputs:

   ```python
   """Generate sklearn fixtures for voting/stacking ensemble correctness tests."""
   from sklearn.ensemble import (
       VotingClassifier, VotingRegressor,
       StackingClassifier, StackingRegressor,
   )
   from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
   from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
   from sklearn.naive_bayes import GaussianNB
   from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
   ```

   Fixtures to generate:
   - **voting_clf_hard.json**: VotingClassifier(hard) with LR+DT+GNB on binary data → predictions
   - **voting_clf_soft.json**: VotingClassifier(soft) with LR+DT+GNB on binary data → predictions + probas
   - **voting_clf_weighted.json**: weighted soft voting → predictions
   - **voting_reg.json**: VotingRegressor with LR+DT+KNN → predictions
   - **voting_reg_weighted.json**: weighted regression → predictions
   - **stacking_clf.json**: StackingClassifier with DT+GNB, final=LR → predictions
   - **stacking_reg.json**: StackingRegressor with LR+DT, final=Ridge → predictions

   Each fixture includes: `X_train`, `y_train`, `X_test`, `sklearn_predictions`, `sklearn_accuracy`/`sklearn_r2`, `sklearn_probas` (where applicable). Use `random_state=42` everywhere for reproducibility.

   **Note on tolerance**: Voting ensembles should match exactly (same base model types). Stacking may differ due to CV implementation details; use relaxed tolerance (accuracy within 0.15, R^2 within 0.3) to verify "comparable" rather than "identical."

2. **File**: `ferroml-python/tests/test_voting_stacking_correctness.py` (NEW, ~250 lines)

   ```python
   """Correctness tests: FerroML voting/stacking vs sklearn fixtures."""
   import json
   import numpy as np
   import pytest
   from pathlib import Path
   ```

   Tests (~10):
   - `test_voting_clf_hard_vs_sklearn` — predictions match within tolerance (accuracy within 0.1 of sklearn)
   - `test_voting_clf_soft_vs_sklearn` — soft predictions comparable
   - `test_voting_clf_soft_probas_shape` — probas shape matches
   - `test_voting_clf_weighted_vs_sklearn` — weighted predictions comparable
   - `test_voting_reg_vs_sklearn` — R^2 within tolerance of sklearn
   - `test_voting_reg_weighted_vs_sklearn` — weighted regression comparable
   - `test_stacking_clf_vs_sklearn` — accuracy within tolerance
   - `test_stacking_reg_vs_sklearn` — R^2 within tolerance
   - `test_ensemble_all_predictions_valid` — all predictions finite and valid class labels
   - `test_ensemble_probas_sum_to_one` — probability outputs sum to ~1.0

**Success Criteria**:
- [ ] Automated: `cd ferroml-python/tests/fixtures && python generate_voting_stacking_fixtures.py` — generates JSON files
- [ ] Automated: `cd ferroml-python && python -m pytest tests/test_voting_stacking_correctness.py -v` — all ~10 tests pass

---

### Phase S.5: Documentation & Integration

**Overview**: Update all module docs, CHANGELOG, and verify the full test suite passes.

**Changes Required**:

1. **File**: `ferroml-python/python/ferroml/__init__.py`
   - Ensure top-level `__init__.py` mentions voting/stacking in the ensemble module docstring
   - Verify the ensemble submodule is properly importable

2. **File**: `CHANGELOG.md`
   - Add Plan S entry under v0.2.1 or "Unreleased":
     ```
     ### Added
     - VotingClassifier Python bindings (hard/soft voting, weighted, 10 classifier types)
     - VotingRegressor Python bindings (weighted averaging, 10 regressor types)
     - StackingClassifier Python bindings (CV-based meta-features, passthrough, custom meta-learner)
     - StackingRegressor Python bindings (CV-based stacking, passthrough, custom meta-learner)
     - 62+ new Python tests including sklearn correctness comparisons
     ```

3. **File**: `ferroml-python/python/ferroml/ensemble/__init__.py`
   - Final docstring polish with examples for all four new classes

4. **Verify full test suite**:
   - All existing ~3,211 Rust tests pass
   - All existing ~1,570+ Python tests pass
   - All ~62 new Python tests pass
   - `cargo fmt --all` clean
   - `cargo clippy --workspace --no-deps` clean

**Success Criteria**:
- [ ] Automated: `cargo test --workspace` — all Rust tests pass
- [ ] Automated: `cd ferroml-python && python -m pytest tests/ -v --tb=short` — all Python tests pass
- [ ] Automated: `cargo fmt --all --check && cargo clippy --workspace --no-deps` — clean

---

## Dependencies

- **No Rust changes needed**: All four ensemble types already exist in `ferroml-core/src/ensemble/`
- **Phase S.2 depends on S.1**: Stacking uses the same `build_classifier_estimators`/`build_regressor_estimators` helpers
- **Phase S.3 depends on S.1+S.2**: Tests need the bindings to exist
- **Phase S.4 depends on S.3**: Correctness tests build on unit test fixtures
- **Phase S.5 depends on all**: Final integration check

Recommended execution order: S.1 → S.2 → S.3 → S.4 → S.5

## Risks & Mitigations

### 1. Pickle/serialization for trait-object-based structs
**Risk**: `VotingClassifier` and `StackingClassifier` contain `Box<dyn VotingClassifierEstimator>` which may not trivially serialize via serde.
**Mitigation**: The Rust structs already use `#[derive(Serialize, Deserialize)]` with `#[serde(skip)]` or custom impls for trait objects (check existing patterns). If serialization is not supported, skip `__getstate__`/`__setstate__` for now and document the limitation. The BaggingClassifier binding stores the inner `BaggingClassifier` which also contains trait objects — check if its pickle works and follow the same pattern.

### 2. Estimator configuration options
**Risk**: String-based constructors create estimators with default parameters (e.g., `LogisticRegression::new()`). Users may want to configure base estimator hyperparameters.
**Mitigation**: Phase S.1-S.2 use default configs. A future enhancement (not in this plan) could accept `dict` kwargs per estimator. For now, document that base estimators use default settings, which matches sklearn's default behavior when you pass `LogisticRegression()` without args.

### 3. Stacking CV differences vs sklearn
**Risk**: FerroML's KFold CV may split differently from sklearn's, causing different stacking results.
**Mitigation**: Use relaxed tolerances in correctness tests (accuracy within 0.15, R^2 within 0.3). The tests verify "competitive" rather than "identical" behavior.

### 4. VotingClassifier predict_proba for hard voting
**Risk**: sklearn raises an error if you call predict_proba with hard voting. FerroML should do the same.
**Mitigation**: In `PyVotingClassifier::predict_proba`, check the voting method and raise `PyValueError` if hard voting is set.

## Estimated Scope

- **Lines of Rust (bindings)**: ~600 in `ensemble.rs` (comparable to BaggingClassifier+BaggingRegressor which are ~700 lines)
- **Lines of Python (tests)**: ~850 across 3 test files
- **New Python tests**: ~62
- **New models exposed to Python**: 4 (VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor)
- **Total estimator factory methods**: 0 (uses string-based dispatch, not factory statics)
