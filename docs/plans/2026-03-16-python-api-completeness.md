# Python API Completeness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose every Rust module a Python ML practitioner needs — metrics, cross-validation, HPO, and model comparison — so users never have to import sklearn for evaluation or tuning.

**Architecture:** Create 3 new PyO3 binding files (`metrics.rs`, `cv.rs`, `hpo.rs`) following the same function-based pattern as `stats.rs`. Each function takes numpy arrays, calls Rust core, returns Python dicts or arrays. Python re-export packages provide clean `from ferroml.metrics import accuracy_score` imports.

**Tech Stack:** Rust, PyO3, numpy, ndarray, ferroml-core

---

## Phase 1: Metrics Module (CRITICAL)

### Task 1: Classification metrics

**Files:**
- Create: `ferroml-python/src/metrics.rs`
- Modify: `ferroml-python/src/lib.rs`
- Create: `ferroml-python/python/ferroml/metrics/__init__.py`

**Step 1: Create the metrics binding file**

Read `ferroml-core/src/metrics/classification.rs` and `ferroml-core/src/metrics/regression.rs` to understand the exact function signatures. Then create `ferroml-python/src/metrics.rs` exposing:

Classification:
- `accuracy_score(y_true, y_pred)` → float
- `precision_score(y_true, y_pred, average="binary")` → float
- `recall_score(y_true, y_pred, average="binary")` → float
- `f1_score(y_true, y_pred, average="binary")` → float
- `matthews_corrcoef(y_true, y_pred)` → float
- `confusion_matrix(y_true, y_pred)` → 2D numpy array
- `classification_report(y_true, y_pred)` → dict with per-class metrics

Regression:
- `mse(y_true, y_pred)` → float (mean squared error)
- `rmse(y_true, y_pred)` → float
- `mae(y_true, y_pred)` → float
- `r2_score(y_true, y_pred)` → float
- `explained_variance(y_true, y_pred)` → float
- `max_error(y_true, y_pred)` → float
- `mape(y_true, y_pred)` → float

Probabilistic:
- `roc_auc_score(y_true, y_score)` → float
- `log_loss(y_true, y_prob)` → float
- `brier_score(y_true, y_prob)` → float

Model comparison:
- `paired_ttest(scores1, scores2)` → dict {statistic, p_value, ci_lower, ci_upper}
- `corrected_resampled_ttest(scores1, scores2, n_train, n_test)` → dict
- `mcnemar_test(y_true, pred1, pred2)` → dict
- `wilcoxon_test(scores1, scores2)` → dict

Follow the exact pattern from `stats.rs`: functions return Python dicts via `PyDict::new()`.

**Step 2: Register module in lib.rs**

Add `mod metrics;` and `metrics::register_metrics_module(m)?;`

**Step 3: Create Python re-export**

Create `ferroml-python/python/ferroml/metrics/__init__.py` re-exporting all functions.

Update `ferroml-python/python/ferroml/__init__.py` to add `from ferroml import metrics`.

**Step 4: Build and test**

```bash
source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml
python3 -c "
from ferroml.metrics import accuracy_score, f1_score, r2_score, mse, confusion_matrix
import numpy as np
y_true = np.array([0., 1., 1., 0., 1.])
y_pred = np.array([0., 1., 0., 0., 1.])
print('accuracy:', accuracy_score(y_true, y_pred))
print('f1:', f1_score(y_true, y_pred))
print('confusion_matrix:', confusion_matrix(y_true, y_pred))

y_true_r = np.array([1., 2., 3., 4., 5.])
y_pred_r = np.array([1.1, 2.2, 2.8, 4.1, 5.0])
print('r2:', r2_score(y_true_r, y_pred_r))
print('mse:', mse(y_true_r, y_pred_r))
"
```

**Step 5: Commit**

```bash
git add ferroml-python/src/metrics.rs ferroml-python/src/lib.rs \
       ferroml-python/python/ferroml/metrics/__init__.py \
       ferroml-python/python/ferroml/__init__.py
git commit -m "feat: expose metrics module in Python (classification, regression, probabilistic, comparison)"
```

---

## Phase 2: Cross-Validation Module (CRITICAL)

### Task 2: CV splitters and cross_val_score

**Files:**
- Create: `ferroml-python/src/cv.rs`
- Modify: `ferroml-python/src/lib.rs`
- Create: `ferroml-python/python/ferroml/cv/__init__.py`

**Step 1: Create the CV binding file**

Read `ferroml-core/src/cv/kfold.rs`, `ferroml-core/src/cv/stratified.rs`, `ferroml-core/src/cv/mod.rs` to understand exact APIs.

Expose:
- `KFold` class: `__init__(n_splits=5, shuffle=False, seed=None)`, `split(X, y)` → list of (train_indices, test_indices) tuples
- `StratifiedKFold` class: same interface, preserves class distribution
- `TimeSeriesSplit` class: `__init__(n_splits=5)`, `split(X)`
- `LeaveOneOut` class: `split(X)`
- `cross_val_score(model, X, y, cv=5, scoring="accuracy")` → numpy array of scores

The `cross_val_score` function is the most important. It needs to:
1. Accept any FerroML model (via a trait object or enum dispatch)
2. Split data using the CV strategy
3. Fit/predict/score on each fold
4. Return array of scores

**Challenge:** `cross_val_score` needs to accept any model type. Options:
- (a) Accept model as a Python object, call `.fit()` and `.predict()` via PyO3 — most flexible
- (b) Create separate functions per model type — verbose but simple
- (c) Accept model name string + params dict, construct internally

Option (a) is best for Python ergonomics. Use `PyAny` for the model parameter and call Python methods.

**Step 2: Register and re-export**

Same pattern as metrics.

**Step 3: Build and test**

```bash
python3 -c "
from ferroml.cv import KFold, StratifiedKFold, cross_val_score
from ferroml.linear import LogisticRegression
import numpy as np
X = np.random.randn(100, 5)
y = (X[:, 0] > 0).astype(float)
scores = cross_val_score(LogisticRegression(), X, y, cv=5)
print('CV scores:', scores)
print('Mean:', np.mean(scores))
"
```

**Step 4: Commit**

```bash
git commit -m "feat: expose cross-validation in Python (KFold, StratifiedKFold, cross_val_score)"
```

---

## Phase 3: HPO Module (IMPORTANT)

### Task 3: Hyperparameter optimization

**Files:**
- Create: `ferroml-python/src/hpo.rs`
- Create: `ferroml-python/python/ferroml/hpo/__init__.py`

**Step 1: Create HPO bindings**

Read `ferroml-core/src/hpo/search_space.rs`, `ferroml-core/src/hpo/samplers.rs`, `ferroml-core/src/hpo/bayesian.rs`.

Expose the most useful subset:
- `GridSearchCV` class: `__init__(model, param_grid, cv=5, scoring="accuracy")`, `.fit(X, y)`, `.best_params_`, `.best_score_`, `.cv_results_`
- `RandomSearchCV` class: same interface but random sampling
- `SearchSpace` class: `add_float(name, low, high)`, `add_int(name, low, high)`, `add_categorical(name, choices)`

**Alternative simpler approach:** Expose just the search space and study:
- `Study` class: `__init__(direction="maximize")`, `.optimize(objective, n_trials=100)`, `.best_params`, `.best_value`
- Let users write their own objective function in Python

**Step 2: Register and re-export**

**Step 3: Build and test**

**Step 4: Commit**

```bash
git commit -m "feat: expose HPO module in Python (GridSearchCV, RandomSearchCV, Study)"
```

---

## Phase 4: Extended Metrics Tests

### Task 4: Test suite for all new Python APIs

**Files:**
- Create: `ferroml-python/tests/test_metrics.py`
- Create: `ferroml-python/tests/test_cv.py`

**Step 1: Create metrics tests**

```python
import numpy as np
from ferroml.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mse, rmse, mae, r2_score, roc_auc_score,
    paired_ttest, corrected_resampled_ttest
)
from sklearn.metrics import (
    accuracy_score as sk_acc, f1_score as sk_f1,
    r2_score as sk_r2, mean_squared_error as sk_mse
)

def test_accuracy_vs_sklearn():
    y_true = np.array([0., 1., 1., 0., 1., 0., 1., 1.])
    y_pred = np.array([0., 1., 0., 0., 1., 1., 1., 1.])
    assert_allclose(accuracy_score(y_true, y_pred), sk_acc(y_true, y_pred))
```

**Step 2: Create CV tests**

```python
def test_cross_val_score_5fold():
    scores = cross_val_score(LogisticRegression(), X, y, cv=5)
    assert len(scores) == 5
    assert all(0 <= s <= 1 for s in scores)
```

**Step 3: Run and fix**

**Step 4: Commit**

```bash
git commit -m "test: metrics and CV verification tests vs sklearn"
```

---

## Summary

| Phase | Module | Functions | Priority | Impact |
|-------|--------|-----------|----------|--------|
| 1 | metrics | ~20 functions | CRITICAL | Users can evaluate models without sklearn |
| 2 | cv | ~5 classes + cross_val_score | CRITICAL | Users can do proper model evaluation |
| 3 | hpo | GridSearchCV, RandomSearchCV | IMPORTANT | Users can tune hyperparameters |
| 4 | tests | ~30 tests | IMPORTANT | Verify correctness vs sklearn |

**After completion:** A Python user can do the full ML workflow — load data, preprocess, train, evaluate (metrics), cross-validate (cv), tune (hpo), compare models (metrics.paired_ttest), and compute statistical inference (stats) — all without importing sklearn.

## What We're NOT Exposing (By Design)

- **linalg** — users use numpy/scipy for this
- **gpu/simd** — internal optimization, feature-gated
- **schema** — nice-to-have, not critical for workflows
- **inference/onnx** — users export ONNX via `.export_onnx()` method already on models
- **get_params/set_params** — FerroML uses builder pattern (`.with_c(1.0)`) by design
- **Advanced explainability variants** — 2D PDP, model-specific SHAP variants — the main functions are already exposed
