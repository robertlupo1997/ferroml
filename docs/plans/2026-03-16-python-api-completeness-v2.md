# Python API Completeness v2 — Fill Every Gap

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** After this plan, a Python ML practitioner can do their entire workflow — split data, train, evaluate, tune, diagnose — without ever importing sklearn.

**Architecture:** Add functions/classes to existing PyO3 binding files (`cv.rs`, `metrics.rs`, `datasets.rs`) and create one new module (`model_selection.rs`). Each wraps existing Rust core APIs.

---

## Phase 1: train_test_split (CRITICAL)

The #1 sklearn import. Currently requires awkward `Dataset.train_test_split()`.

### Task 1.1: Add standalone train_test_split function

**Files:**
- Create: `ferroml-python/src/model_selection.rs`
- Modify: `ferroml-python/src/lib.rs`
- Create: `ferroml-python/python/ferroml/model_selection/__init__.py`
- Modify: `ferroml-python/python/ferroml/__init__.py`

**Step 1: Create model_selection.rs**

Implement a `train_test_split` function that takes raw numpy arrays and returns `(X_train, X_test, y_train, y_test)`:

```rust
#[pyfunction]
#[pyo3(signature = (x, y, test_size=0.25, shuffle=true, random_state=None, stratify=None))]
fn train_test_split(
    py: Python<'_>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    test_size: f64,
    shuffle: bool,
    random_state: Option<u64>,
    stratify: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject)>
```

Implementation approach:
- If `stratify` is None: Use `ShuffleSplit::new(1).with_test_size(test_size).with_seed(seed)` from `ferroml_core::cv::ShuffleSplit` to get one split, then index into X and y arrays
- If `stratify` is Some: Use `StratifiedKFold` with 2 folds or implement manual stratified shuffle
- Return `(X_train, X_test, y_train, y_test)` as numpy arrays — matching sklearn's return order

Also add a `model_selection` re-export module that bundles train_test_split + all CV classes + cross_val_score for sklearn-like imports.

**Step 2: Register in lib.rs**

```rust
mod model_selection;
// ...
model_selection::register_model_selection_module(m)?;
```

**Step 3: Create Python re-export**

`ferroml-python/python/ferroml/model_selection/__init__.py` should re-export:
- `train_test_split` (from native)
- `KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut` (from ferroml.cv)
- `cross_val_score` (from ferroml.cv)
- `GridSearchCV, RandomSearchCV` (from ferroml.hpo)

This gives users `from ferroml.model_selection import train_test_split, cross_val_score, GridSearchCV` — exactly like sklearn.

**Step 4: Build and test**

```bash
source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml
python3 -c "
from ferroml.model_selection import train_test_split
import numpy as np
X = np.random.randn(100, 5)
y = (X[:, 0] > 0).astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {X_train.shape}, Test: {X_test.shape}')
assert X_train.shape[0] == 80
assert X_test.shape[0] == 20
print('train_test_split works!')
"
```

**Step 5: Commit**

---

## Phase 2: ROC/PR Curves + Metric Curves (CRITICAL)

### Task 2.1: Expose roc_curve and precision_recall_curve

**Files:**
- Modify: `ferroml-python/src/metrics.rs`
- Modify: `ferroml-python/python/ferroml/metrics/__init__.py`

**Step 1: Add curve functions to metrics.rs**

Use existing `RocCurve::compute()` and `PrCurve::compute()` from `ferroml_core::metrics::probabilistic`:

```rust
/// Returns dict with keys: fpr (array), tpr (array), thresholds (array), auc (float)
#[pyfunction]
#[pyo3(signature = (y_true, y_score))]
fn roc_curve(py: Python<'_>, y_true: PyReadonlyArray1<f64>, y_score: PyReadonlyArray1<f64>) -> PyResult<PyObject>

/// Returns dict with keys: precision (array), recall (array), thresholds (array), auc (float), average_precision (float)
#[pyfunction]
#[pyo3(signature = (y_true, y_score))]
fn precision_recall_curve(py: Python<'_>, y_true: PyReadonlyArray1<f64>, y_score: PyReadonlyArray1<f64>) -> PyResult<PyObject>
```

Both return dicts with numpy arrays for the curve points and scalar AUC values.

**Step 2: Register in metrics module and update Python re-export**

Add `roc_curve` and `precision_recall_curve` to `register_metrics_module` and to `__init__.py`.

**Step 3: Build and test**

```python
from ferroml.metrics import roc_curve, precision_recall_curve
import numpy as np
y_true = np.array([0., 0., 1., 1.])
y_score = np.array([0.1, 0.4, 0.8, 0.9])
roc = roc_curve(y_true, y_score)
print(f"ROC AUC: {roc['auc']:.3f}, FPR points: {len(roc['fpr'])}")
pr = precision_recall_curve(y_true, y_score)
print(f"PR AUC: {pr['auc']:.3f}")
```

**Step 4: Commit**

---

## Phase 3: Additional CV Splitters (IMPORTANT)

### Task 3.1: Expose remaining splitters

**Files:**
- Modify: `ferroml-python/src/cv.rs`
- Modify: `ferroml-python/python/ferroml/cv/__init__.py`

Add these classes following the exact same pattern as PyKFold:

1. **`RepeatedKFold`** — `__init__(n_folds=5, n_repeats=10, random_state=None)`, `split(X, y=None)`
   - Wraps `ferroml_core::cv::RepeatedKFold`

2. **`ShuffleSplit`** — `__init__(n_splits=10, test_size=0.1, train_size=None, random_state=None)`, `split(X, y=None)`
   - Wraps `ferroml_core::cv::ShuffleSplit`

3. **`GroupKFold`** — `__init__(n_folds=5)`, `split(X, y=None, groups=None)`
   - Wraps `ferroml_core::cv::GroupKFold`
   - Note: `groups` parameter is `Array1<i64>` in Rust, accept as numpy int array

4. **`LeavePOut`** — `__init__(p=2)`, `split(X, y=None)`
   - Wraps `ferroml_core::cv::LeavePOut`

Register all in `register_cv_module` and update Python `__init__.py`.

**Build and test:**

```python
from ferroml.cv import RepeatedKFold, ShuffleSplit, GroupKFold
import numpy as np
X = np.random.randn(100, 5)
y = np.random.randn(100)

rkf = RepeatedKFold(n_folds=5, n_repeats=2, random_state=42)
splits = rkf.split(X)
assert len(splits) == 10  # 5 folds * 2 repeats

ss = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
splits = ss.split(X)
assert len(splits) == 3

groups = np.array([i // 10 for i in range(100)])
gkf = GroupKFold(n_folds=5)
splits = gkf.split(X, groups=groups)
assert len(splits) == 5
```

**Commit**

---

## Phase 4: Learning Curve & Validation Curve (IMPORTANT)

These are harder because `learning_curve` and `validation_curve` in Rust require the `Estimator` trait and `ParameterSettable` trait, which PyO3 Python objects don't implement.

### Task 4.1: Python-side learning_curve and validation_curve

**Best approach:** Implement these in pure Python (in the `__init__.py`) using the existing `cross_val_score` infrastructure, rather than trying to bridge the Rust trait boundary.

**Files:**
- Modify: `ferroml-python/python/ferroml/cv/__init__.py`

**learning_curve implementation:**

```python
def learning_curve(model, X, y, cv=5, train_sizes=None, scoring="accuracy"):
    """Compute train/test scores for varying training set sizes.

    Returns
    -------
    train_sizes_abs : list of int
    train_scores : list of float (mean per size)
    test_scores : list of float (mean per size)
    """
    if train_sizes is None:
        train_sizes = [0.1, 0.33, 0.55, 0.78, 1.0]

    n = X.shape[0]
    kf = KFold(n_folds=cv)
    splits = kf.split(X)

    results_train = []
    results_test = []
    sizes_abs = []

    for frac in train_sizes:
        fold_train_scores = []
        fold_test_scores = []
        for train_idx, test_idx in splits:
            # Subsample training set
            n_train = max(1, int(len(train_idx) * frac))
            sub_train_idx = train_idx[:n_train]

            X_tr = X[sub_train_idx]
            y_tr = y[sub_train_idx]
            X_te = X[test_idx]
            y_te = y[test_idx]

            m = model.__class__()
            m.fit(X_tr, y_tr)

            # Score
            fold_train_scores.append(_score(m, X_tr, y_tr, scoring))
            fold_test_scores.append(_score(m, X_te, y_te, scoring))

        sizes_abs.append(int(len(splits[0][0]) * frac))
        results_train.append(np.mean(fold_train_scores))
        results_test.append(np.mean(fold_test_scores))

    return sizes_abs, results_train, results_test
```

**validation_curve implementation:**

```python
def validation_curve(model, X, y, param_name, param_range, cv=5, scoring="accuracy"):
    """Compute train/test scores for varying hyperparameter values.

    Returns
    -------
    param_range : array-like
    train_scores : list of float (mean per value)
    test_scores : list of float (mean per value)
    """
    train_scores = []
    test_scores = []

    for value in param_range:
        m = model.__class__(**{param_name: value})
        scores_train = []
        scores_test = []

        kf = KFold(n_folds=cv)
        for train_idx, test_idx in kf.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            m_fold = model.__class__(**{param_name: value})
            m_fold.fit(X_tr, y_tr)
            scores_train.append(_score(m_fold, X_tr, y_tr, scoring))
            scores_test.append(_score(m_fold, X_te, y_te, scoring))

        train_scores.append(np.mean(scores_train))
        test_scores.append(np.mean(scores_test))

    return list(param_range), train_scores, test_scores
```

Helper `_score` function computes accuracy/mse/r2 using ferroml.metrics.

**Build and test:**

```python
from ferroml.cv import learning_curve, validation_curve
from ferroml.linear import LogisticRegression
import numpy as np

X = np.random.randn(200, 5)
y = (X[:, 0] > 0).astype(float)

sizes, train_scores, test_scores = learning_curve(LogisticRegression(), X, y, cv=3)
print(f"Learning curve: {len(sizes)} points")

param_range, train_s, test_s = validation_curve(
    LogisticRegression(), X, y, param_name="l2_penalty", param_range=[0.001, 0.01, 0.1, 1.0, 10.0], cv=3
)
print(f"Validation curve: {len(param_range)} points")
```

**Commit**

---

## Phase 5: Normalizer + model_selection re-exports (NICE-TO-HAVE)

### Task 5.1: Expose Normalizer in preprocessing

**Files:**
- Modify: `ferroml-python/src/preprocessing.rs`
- Modify: `ferroml-python/python/ferroml/preprocessing/__init__.py`

Add `PyNormalizer` class wrapping `ferroml_core::preprocessing::scalers::Normalizer`:
- `__init__(norm="l2")` — accepts "l1", "l2", "max"
- `fit(X)`, `transform(X)`, `fit_transform(X)`

### Task 5.2: Add convenience functions to model_selection

**Files:**
- Modify: `ferroml-python/python/ferroml/model_selection/__init__.py`

Add Python-side convenience wrappers:
- `cross_validate(model, X, y, cv=5, scoring="accuracy", return_train_score=False)` → dict with keys: test_score, train_score, fit_time, score_time
- Re-export everything from `ferroml.cv` and `ferroml.hpo` so users can do `from ferroml.model_selection import *`

**Commit**

---

## Phase 6: Test Suite (CRITICAL)

### Task 6.1: Comprehensive tests for all new APIs

**Files:**
- Create: `ferroml-python/tests/test_model_selection.py`
- Modify: `ferroml-python/tests/test_metrics.py` (add curve tests)
- Modify: `ferroml-python/tests/test_cv.py` (add new splitter tests)

**test_model_selection.py tests:**
- `test_train_test_split_shapes` — verify X_train.shape[0] + X_test.shape[0] == n
- `test_train_test_split_reproducibility` — same random_state → same split
- `test_train_test_split_no_overlap` — train/test have no shared rows
- `test_train_test_split_stratified` — class ratios preserved
- `test_train_test_split_vs_sklearn` — numerical comparison

**metrics curve tests:**
- `test_roc_curve_keys` — returns fpr, tpr, thresholds, auc
- `test_roc_curve_perfect` — AUC = 1.0 for perfect predictions
- `test_roc_curve_vs_sklearn` — compare fpr/tpr arrays
- `test_precision_recall_curve_keys` — returns precision, recall, thresholds
- `test_precision_recall_curve_vs_sklearn`

**CV splitter tests:**
- `test_repeated_kfold_count` — n_folds * n_repeats splits
- `test_shuffle_split_sizes` — test set is ~test_size fraction
- `test_group_kfold_no_group_leakage` — no group appears in both train/test
- `test_leave_p_out_basic`

**learning_curve / validation_curve tests:**
- `test_learning_curve_increasing` — test score should generally increase with more data
- `test_validation_curve_returns_correct_length`

**Run all tests:**
```bash
python3 -m pytest ferroml-python/tests/test_model_selection.py ferroml-python/tests/test_metrics.py ferroml-python/tests/test_cv.py -v
```

**Commit**

---

## Summary

| Phase | What | Functions/Classes | Priority |
|-------|------|-------------------|----------|
| 1 | train_test_split | 1 function | CRITICAL |
| 2 | ROC/PR curves | 2 functions | CRITICAL |
| 3 | CV splitters | 4 classes (RepeatedKFold, ShuffleSplit, GroupKFold, LeavePOut) | IMPORTANT |
| 4 | Learning/validation curves | 2 Python functions | IMPORTANT |
| 5 | Normalizer + model_selection | 1 class + re-exports | NICE-TO-HAVE |
| 6 | Tests | ~25 tests | CRITICAL |

**After completion:** `from ferroml.model_selection import train_test_split, cross_val_score, GridSearchCV` works exactly like sklearn. Users can split data, cross-validate, tune hyperparameters, plot learning curves, and evaluate with ROC/PR curves — all without sklearn.
