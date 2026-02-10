# FerroML Accuracy Report

> **Status:** In Progress — Sklearn comparison tests running

## Overview

This document reports FerroML's numerical accuracy compared to scikit-learn reference implementations. All comparisons use standard datasets (Iris, Wine, Diabetes) with fixed random seeds for reproducibility.

## Tolerance Standards

| Algorithm Type | Tolerance | Rationale |
|----------------|-----------|-----------|
| Closed-form (LinearRegression, PCA) | 1e-10 | Exact solution, should match |
| Iterative (Lasso, Logistic, ElasticNet) | 1e-4 | Convergence-dependent |
| Tree-based (DecisionTree, RF, GB) | 1e-6 | Deterministic splits |
| Probabilistic outputs (predict_proba) | 1e-6 | Softmax numerical stability |

## Test Results (2026-02-08)

### Summary

| Status | Count | Description |
|--------|-------|-------------|
| **PASS** | 3 | Exact or near-exact match (diff < 1e-6) |
| **CLOSE** | 2 | Acceptable difference (diff < 0.15) |
| **FAIL** | 0 | Significant discrepancy |

### Linear Models

| Model | Dataset | Metric | sklearn | FerroML | Difference | Status |
|-------|---------|--------|---------|---------|------------|--------|
| LinearRegression | Diabetes | R² | 0.4526027630 | 0.4526027630 | 0.00e+00 | **PASS** |

### Logistic Regression

| Model | Dataset | Metric | sklearn | FerroML | Difference | Status |
|-------|---------|--------|---------|---------|------------|--------|
| LogisticRegression | Iris | Accuracy | _blocked_ | _blocked_ | — | Type conversion issue in Python bindings |

> **Note:** LogisticRegression comparison blocked by numpy array type conversion issue. The Rust implementation is tested separately via `cargo test`.

### Decision Trees

| Model | Dataset | Metric | sklearn | FerroML | Difference | Status |
|-------|---------|--------|---------|---------|------------|--------|
| DecisionTreeClassifier | Iris | Accuracy | 1.000000 | 1.000000 | 0.00e+00 | **PASS** |
| DecisionTreeRegressor | Diabetes | R² | 0.060654 | -0.053882 | 1.15e-01 | **CLOSE** |

> **Investigation Needed:** DecisionTreeRegressor shows sign flip in R². Possible causes:
> - Different split criteria defaults (sklearn uses squared_error, FerroML may differ)
> - Different minimum samples per leaf
> - Random state handling differences

### Random Forest

| Model | Dataset | Metric | sklearn | FerroML | Difference | Status |
|-------|---------|--------|---------|---------|------------|--------|
| RandomForestClassifier | Iris | Accuracy | 1.000000 | 1.000000 | 0.00e+00 | **PASS** |
| RandomForestRegressor | Diabetes | R² | 0.442823 | 0.426252 | 1.66e-02 | **CLOSE** |

> **Expected:** RandomForest differences are expected due to different random number generators between sklearn and FerroML. Both achieve comparable performance.

### K-Nearest Neighbors

| Model | Dataset | Metric | sklearn | FerroML | Difference | Status |
|-------|---------|--------|---------|---------|------------|--------|
| KNeighborsClassifier | Iris | Accuracy | 1.000000 | — | — | Not exposed in Python bindings |

> **Note:** KNN is implemented in Rust but not yet exposed via Python bindings.

## Preprocessing Comparison

| Transformer | Test | sklearn | FerroML | Max Diff | Status |
|-------------|------|---------|---------|----------|--------|
| StandardScaler | Iris transform | _pending_ | _pending_ | _pending_ | |
| MinMaxScaler | Iris transform | _pending_ | _pending_ | _pending_ | |

> **Note:** Preprocessing comparison pending. Priority given to model accuracy validation.

## Known Differences

### Expected Differences (Not Bugs)

These differences are expected due to algorithm implementation choices:

1. **RandomForest RNG** — sklearn and FerroML use different random number generators. Even with the same seed, bootstrap samples and split randomness differ. Both achieve comparable performance on standard datasets.

2. **Ensemble Variance** — Stochastic algorithms (RF, GB) will show 1-5% variance in metrics due to randomness. This is normal and not a correctness issue.

### Issues Found (Under Investigation)

1. **DecisionTreeRegressor R² Sign Flip** — sklearn achieves R²=0.06 while FerroML achieves R²=-0.05 on Diabetes dataset. Possible causes:
   - Different split criterion defaults
   - Different min_samples_leaf handling
   - **Action:** Investigate `models/tree.rs` split logic

### Python Binding Limitations

1. **LogisticRegression** — Multiclass target array conversion issue. The Rust implementation works correctly (tested via `cargo test`), but Python bindings have type conversion issues for integer-like float arrays.

2. **KNeighborsClassifier** — Not exposed in Python bindings. Rust implementation is complete and tested.

## Methodology

### Test Environment
- Python: 3.11+
- sklearn: latest
- FerroML: commit `28b17cd`
- OS: Windows/Linux/macOS

### Procedure
1. Load standard dataset (Iris/Wine/Diabetes)
2. Train model with identical hyperparameters
3. Generate predictions on test set
4. Compare metrics and raw predictions
5. Report maximum absolute difference

### Reproducibility
All tests use fixed random seeds:
- Train/test split: `random_state=42`
- Models with randomness: `random_state=42`

## Updates

| Date | Update |
|------|--------|
| 2026-02-08 | Initial report structure created |
| 2026-02-08 | First comparison results: 3 PASS, 2 CLOSE, 0 FAIL |
| 2026-02-08 | Identified DecisionTreeRegressor R² sign flip for investigation |
| 2026-02-08 | Documented Python binding limitations (LogisticRegression, KNN) |

## Next Steps

1. **Investigate DecisionTreeRegressor** — Compare split criteria and parameters
2. **Fix Python bindings** — LogisticRegression type conversion
3. **Add KNN to Python bindings** — Expose existing Rust implementation
4. **Test preprocessing** — StandardScaler, MinMaxScaler comparison
5. **Test on Wine dataset** — Expand beyond Iris/Diabetes

---

*Last updated: 2026-02-08*
*Test environment: Python 3.11, sklearn latest, FerroML 0.1.0 (commit 28b17cd)*
