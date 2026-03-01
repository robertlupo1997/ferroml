# Plan D: Python Bindings Completion

**Date:** 2026-02-25
**Priority:** CRITICAL (only 35% of features accessible from Python)
**Module:** `ferroml-python/src/` (11,278 lines)
**Estimated New Code:** ~2,000 lines
**Parallel-Safe:** Yes (different crate, no Rust core changes)

## Overview

FerroML's Python bindings currently expose only ~35% of the library's features. Major gaps: explainability (0% exposed), decomposition (0% exposed), most preprocessing transformers, several important models. This plan completes the bindings to make FerroML a viable sklearn alternative from Python.

## Current State

### What's Exposed
Reading `ferroml-python/src/lib.rs` and submodules reveals:

**Models (mostly complete):**
- Linear: LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
- Trees: DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor
- Boosting: GradientBoostingClassifier/Regressor, HistGradientBoostingClassifier/Regressor
- SVM: SVC, SVR, LinearSVC
- KNN: KNeighborsClassifier, KNeighborsRegressor
- Naive Bayes: GaussianNB
- Ensemble: VotingClassifier, StackingClassifier

**Preprocessing (partial):**
- StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
- LabelEncoder, OrdinalEncoder, OneHotEncoder
- SimpleImputer

**Clustering (partial):**
- KMeans, DBSCAN

### What's Missing

**Explainability (0% — ~15 functions):**
- `permutation_importance()`
- `partial_dependence()`
- `individual_conditional_expectation()`
- `tree_shap_values()`
- `kernel_shap_values()`
- `h_statistic()`
- `feature_importance_summary()`

**Decomposition (0% — ~5 models):**
- PCA
- IncrementalPCA
- TruncatedSVD
- LDA (LinearDiscriminantAnalysis)
- FactorAnalysis

**Models (missing ~8):**
- ExtraTreesClassifier, ExtraTreesRegressor
- AdaBoostClassifier, AdaBoostRegressor
- BaggingClassifier, BaggingRegressor
- SGDClassifier, SGDRegressor

**Preprocessing (missing ~12):**
- PowerTransformer, QuantileTransformer
- PolynomialFeatures
- KBinsDiscretizer
- VarianceThreshold, SelectKBest, SelectFromModel, RFE
- KNNImputer
- TargetEncoder
- SMOTE, ADASYN (resampling)

**Clustering (missing ~1):**
- AgglomerativeClustering

**Metrics (partial — expand):**
- Clustering metrics (silhouette, ARI, NMI, etc.)
- Additional classification/regression metrics

**AutoML:**
- AutoMLClassifier, AutoMLRegressor (if not exposed)

## Desired End State

- 90%+ of ferroml-core features accessible from Python
- All major model families complete
- Explainability fully exposed (key differentiator vs sklearn)
- Integration tests verify Python API works end-to-end

## Implementation Phases

### Phase D.1: Explainability Bindings (~400 lines)

**File:** `ferroml-python/src/explainability.rs` (NEW)

```rust
use pyo3::prelude::*;
use ferroml_core::explainability::*;

#[pyfunction]
fn permutation_importance(
    py: Python,
    model: &PyAny,  // Accept any fitted model
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    n_repeats: Option<usize>,
    random_state: Option<u64>,
) -> PyResult<PyObject> { ... }

#[pyfunction]
fn partial_dependence(
    x: PyReadonlyArray2<f64>,
    model: &PyAny,
    features: Vec<usize>,
    grid_resolution: Option<usize>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> { ... }

// ... tree_shap_values, kernel_shap_values, etc.
```

**Challenge:** Models need to implement a common trait for the Python `model` parameter. May need a wrapper enum or trait object.

### Phase D.2: Decomposition Bindings (~300 lines)

**File:** `ferroml-python/src/decomposition.rs` (NEW)

```rust
#[pyclass]
pub struct PyPCA {
    inner: ferroml_core::decomposition::PCA,
}

#[pymethods]
impl PyPCA {
    #[new]
    fn new(n_components: Option<usize>) -> Self { ... }
    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> { ... }
    fn transform(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> { ... }
    fn fit_transform(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> { ... }
    fn inverse_transform(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> { ... }
    #[getter] fn explained_variance_ratio(&self) -> PyResult<Py<PyArray1<f64>>> { ... }
    #[getter] fn components(&self) -> PyResult<Py<PyArray2<f64>>> { ... }
}

// Similarly: PyTruncatedSVD, PyLDA, PyFactorAnalysis, PyIncrementalPCA
```

### Phase D.3: Missing Model Bindings (~500 lines)

**File:** `ferroml-python/src/models.rs` (MODIFY)

Add 8 missing models following existing pattern:
```rust
#[pyclass] pub struct PyExtraTreesClassifier { ... }
#[pyclass] pub struct PyExtraTreesRegressor { ... }
#[pyclass] pub struct PyAdaBoostClassifier { ... }
#[pyclass] pub struct PyAdaBoostRegressor { ... }
#[pyclass] pub struct PyBaggingClassifier { ... }
#[pyclass] pub struct PyBaggingRegressor { ... }
#[pyclass] pub struct PySGDClassifier { ... }
#[pyclass] pub struct PySGDRegressor { ... }
```

### Phase D.4: Missing Preprocessing Bindings (~500 lines)

**File:** `ferroml-python/src/preprocessing.rs` (MODIFY)

Add 12 missing transformers:
```rust
#[pyclass] pub struct PyPowerTransformer { ... }
#[pyclass] pub struct PyQuantileTransformer { ... }
#[pyclass] pub struct PyPolynomialFeatures { ... }
#[pyclass] pub struct PyKBinsDiscretizer { ... }
#[pyclass] pub struct PyVarianceThreshold { ... }
#[pyclass] pub struct PySelectKBest { ... }
#[pyclass] pub struct PyKNNImputer { ... }
#[pyclass] pub struct PyTargetEncoder { ... }
```

### Phase D.5: Clustering & Metrics Bindings (~200 lines)

**File:** `ferroml-python/src/clustering.rs` (MODIFY)

```rust
#[pyclass] pub struct PyAgglomerativeClustering { ... }

// Clustering metrics as module-level functions
#[pyfunction] fn silhouette_score(...) -> PyResult<f64> { ... }
#[pyfunction] fn adjusted_rand_score(...) -> PyResult<f64> { ... }
#[pyfunction] fn normalized_mutual_info_score(...) -> PyResult<f64> { ... }
// etc.
```

### Phase D.6: Module Registration & Integration Tests

**File:** `ferroml-python/src/lib.rs` (MODIFY)
Register all new submodules and classes.

**File:** `ferroml-python/tests/test_bindings.py` (NEW)
```python
import ferroml
import numpy as np

def test_pca_roundtrip():
    X = np.random.randn(100, 5)
    pca = ferroml.PCA(n_components=3)
    X_t = pca.fit_transform(X)
    assert X_t.shape == (100, 3)
    X_r = pca.inverse_transform(X_t)
    assert X_r.shape == (100, 5)

def test_permutation_importance():
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + np.random.randn(100) * 0.1
    model = ferroml.LinearRegression()
    model.fit(X, y)
    imp = ferroml.permutation_importance(model, X, y)
    assert imp.shape == (5,)
    assert imp[0] > imp[1]  # Feature 0 should be most important

# ... 20+ integration tests
```

## Success Criteria

- [ ] `cargo check -p ferroml-python` compiles
- [ ] `maturin develop` builds wheel successfully
- [ ] `pytest ferroml-python/tests/` — all integration tests pass
- [ ] 90%+ of ferroml-core public API accessible from Python
- [ ] Explainability functions work end-to-end (key differentiator)
- [ ] All decomposition models expose fit/transform/inverse_transform

## Dependencies

- PyO3 (existing)
- maturin (existing)
- numpy, pytest for testing
- No new Rust crate dependencies

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| PyO3 type conversion for complex return types | Use PyDict for complex results, Py<PyArray> for arrays |
| Explainability needs generic model parameter | Create PyModelWrapper enum or use trait objects |
| ferroml-python won't link without libpython | Use `cargo check` for compilation, `maturin develop` for full build |
| Some core APIs may not be Python-friendly | Add Python-specific convenience wrappers where needed |
