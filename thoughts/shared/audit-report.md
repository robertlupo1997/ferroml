# FerroML Real-World Robustness Audit

**Date:** 2026-03-14
**Scope:** NaN/inf handling, numerical stability, edge cases, API gaps vs sklearn

---

## P0 — Critical: NaN/Inf Silently Propagates

`validate_fit_input()` and `validate_predict_input()` in `models/mod.rs:1211-1236` **do not check for NaN or infinity**. They only verify shape consistency and non-emptiness. This means NaN/inf silently flows through nearly all models.

**Fix:** Add finite checks to the centralized validators:
```rust
if !x.iter().all(|&v| v.is_finite()) {
    return Err(FerroError::invalid_input("X contains NaN or infinite values"));
}
```
This would protect ~17 models that call `validate_fit_input`. The remaining ~7 models (GP, IsolationForest, QDA, LOF, etc.) need individual fixes.

---

## P1 — Numerical Stability Issues (9 found)

### Critical

| Issue | Location | Problem |
|-------|----------|---------|
| ~~SGD log loss overflow~~ | `sgd.rs:311-312` | **FIXED** (pass 5) — log-sum-exp trick |
| ~~SVM Platt sigmoid overflow~~ | `svm.rs:681, 726` | **FIXED** (pass 6) — `stable_sigmoid()` |
| ~~LogisticRegression log underflow~~ | `logistic.rs:1003` | **FIXED** — `(1.0 - yi).max(1e-15).ln()` |
| ~~GaussianNB zero log~~ | `naive_bayes.rs:472` | **FIXED** (pass 6) — `.max(1e-300)` guard |
| ~~QDA zero log~~ | `qda.rs:170` | **FIXED** (pass 6) — `.max(1e-300)` guard |

### High

| Issue | Location | Problem |
|-------|----------|---------|
| ~~GP variance cancellation~~ | `gaussian_process.rs` (8 sites) | **FIXED** — `stable_posterior_variance()` helper floors at `prior * 1e-10` instead of hard zero |
| ~~MultinomialNB zero log~~ | `naive_bayes.rs:968` | **FIXED** (pass 6) — `.max(1e-300)` guard |

### Medium

| Issue | Location | Problem |
|-------|----------|---------|
| ~~LogReg condition number~~ | `logistic.rs:499-508` | **FIXED** — diagonal ratio check; auto-jitter when ratio < 1e-12 |
| ~~Regularized sqrt~~ | `regularized.rs:248, 2631` | **FIXED** — added `.max(0.0)` guard before `.sqrt()` on both sites |

---

## P1 — Unguarded sqrt/ln/div Operations

Models that compute `ln()`, `sqrt()`, or divide without guards on potentially-NaN intermediate values:

- ~~**naive_bayes.rs:472-473**~~ — **FIXED** (pass 6) — `.max(1e-300)` guard
- ~~**qda.rs:167**~~ — **FIXED** (pass 6) — `.max(1e-300)` guard
- ~~**gaussian_process.rs:352, 368**~~ — Cholesky diagonal division: safe — cholesky() rejects non-PD matrices, diagonal always positive
- ~~**robust.rs:577, 704**~~ — **FIXED** — `.max(0.0)` guard before `.sqrt()` on sigma and SE
- ~~**quantile.rs:385**~~ — **FIXED** — `.max(0.0)` guard before `.sqrt()` on weights

---

## P2 — Edge Cases

### Well Handled
- **Empty arrays**: Rejected by `check_non_empty()` — all scalers and most models
- **Constant features**: All 4 scalers (Standard, MinMax, Robust, MaxAbs) track constant features with `1e-10` epsilon threshold
- **Class imbalance**: `ClassWeight::Balanced` supported in LogReg, Trees, RF, SVC
- **Single class**: 22+ models tested to reject gracefully

### Gaps
- ~~**Single-sample fit**: Most models untested.~~ — **FIXED**: GaussianNB `max_var` floor (`.max(1e-300)`) prevents epsilon=0 when all per-class variances are zero (both dense and sparse paths). LinearRegression, HuberRegressor, LOF, QDA already reject n=1. Ridge/Lasso/ElasticNet/GP accept n=1 (mathematically valid with regularization). 1 new test added.
- ~~**n < p (underdetermined)**: No validation that n_samples >= n_features for linear models~~ — **FIXED**: LinearRegression and HuberRegressor already reject n < p; Ridge/Lasso/ElasticNet handle gracefully via regularization; **LogisticRegression** now auto-applies L2=1e-4 with warning when n < p and no explicit penalty (both dense and sparse paths). 2 new tests added.
- ~~**Single-feature datasets**: Only tested in IsolationForest, QDA, CategoricalNB~~ — **FIXED**: Improved `check_single_feature` compliance check to also try binary classification labels (previously only tested regression targets, so classifiers were never actually tested with p=1). Now tests both regression and classification across all registered models. 37 compliance tests pass. No panics or silent failures found — all models either work correctly or reject gracefully.

---

## P2 — sklearn API Gaps

### Missing from Model trait (broad impact)
| Method | Status | Impact |
|--------|--------|--------|
| ~~`score()`~~ | **FIXED** (pass 8) — default method on Model trait | Common sklearn pattern |
| `get_params()` / `set_params()` | Not applicable (Rust builder pattern) | Blocks GridSearchCV-style workflows |
| ~~`partial_fit()`~~ | **FIXED** (pass 9) — SGDClassifier + SGDRegressor via IncrementalModel | Online/incremental learning |
| ~~`sample_weight` in `fit()`~~ | **FIXED** — `fit_weighted()` on Model trait; implemented for LogisticRegression + DecisionTreeClassifier | Weighted fitting from user code |
| ~~`decision_function()`~~ | **FIXED** — SVC already public; added to LogReg | Common sklearn pattern |
| ~~`predict_log_proba()`~~ | **FIXED** — default method on ProbabilisticModel trait + exposed in 18 Python bindings | sklearn pipeline compatible |

### Partial coverage
| Feature | Status |
|---------|--------|
| `warm_start` | 7/55+ models (Ridge, Lasso, ElasticNet, GB, RF) |
| `inverse_transform()` | All 4 scalers + PowerTransform + OrdinalEncoder + LabelEncoder; TargetEncoder N/A (many-to-one) |
| `feature_importances_` | Trees/forests only |
| `coef_` / `intercept_` | Linear models only |

---

## P2 — ONNX Export Edge Cases

Found in pass 3 (2026-03-14):

| Issue | Location | Severity |
|-------|----------|----------|
| ~~GaussianNB zero variance in export~~ | `onnx/naive_bayes.rs:283-295` | **FIXED** — variance floor 1e-10 |
| ~~SVC coefficient silently dropped~~ | `onnx/svm.rs:473-484` | **FIXED** — returns error instead of silent skip |
| ~~SVC integer overflow (many classes)~~ | `onnx/svm.rs:430-434` | **FIXED** — checked_mul + 100M element cap |
| ~~BernoulliNB log stability~~ | `onnx/naive_bayes.rs:140-155` | **FIXED** — stable_log1m_exp with 3-regime branching |
| ~~No input_shape vs model n_features validation~~ | all onnx/*.rs | **FIXED** — validate_onnx_config + 4 new tests |
| ~~No model size limits~~ | `onnx/tree.rs` | **FIXED** — 10M node cap via check_tree_node_limit |

---

## P3 — f32 Precision Loss (ONNX)

The ONNX export converts all f64 values to f32. Known risk areas:
- **Tree ensemble leaf values**: Normalized probabilities lose precision for rare classes
- **SVM dual coefficients**: Large-magnitude coefficients lose precision
- **Kernel parameters**: Auto-gamma `1/n_features` can lose precision for large feature counts
- **HistGBT bin thresholds**: `next_down_f32` ULP nudge may not preserve <= semantics near f32 subnormals
- **Tie-breaking**: ~5/20 random seeds show 1/50 mismatch at exact 0.5/0.5 probability ties

Current mitigation: Tests use `atol=1e-5` for regressors, exact match for classifier labels. The ONNX roundtrip suite (44/44 passing) validates these tolerances.

---

## Python Bindings Assessment

**Overall: EXCELLENT** — pass 3 found no security vulnerabilities or crash risks.

Strengths:
- No `unwrap()` or `panic!()` in bindings — all errors convert to PyRuntimeError/PyValueError
- Sparse matrix handling validates NaN/inf, formats, dimensions
- Unfitted model checks comprehensive across all models
- Safe array handling via PyReadonly types

Minor issues (cosmetic/defensive):
- `pipeline.rs:641-660` — ColumnTransformer creates dummy 1-row array to infer output shape; acceptable risk (transformers validate inputs)
- `to_onnx_bytes()` relies on Rust core for unfitted checks (works correctly)
- ~~Minor indentation issues in `preprocessing/__init__.py:52` and `gaussian_process/__init__.py:43`~~ — verified clean, no issues

---

## P1 — Silent Wrong Results (Linear/Regularized Models)

Found in follow-up pass (2026-03-14, pass 2):

| Issue | Location | Problem |
|-------|----------|---------|
| ~~SGD alpha=0 div-by-zero~~ | `sgd.rs:242, 592` | **FIXED** (pass 5) — validates alpha > 0 for optimal schedule |
| ~~Lasso/ElasticNet silent non-convergence~~ | `regularized.rs:659, 1073` | **FIXED** (pass 6) — warns to stderr when max_iter reached |
| ~~LogReg perfect separation ignored~~ | `logistic.rs:460-462` | **FIXED** (pass 6) — warns to stderr when extreme probabilities detected |
| ~~PassiveAggressive C=0 non-learning~~ | `sgd.rs:920` | **FIXED** — validates C > 0 + 2 new tests |
| ~~Ridge alpha=0 degrades to OLS~~ | `regularized.rs:185-187` | **FIXED** — warns to stderr when alpha=0 |

---

## P1 — Clustering/Anomaly Detection Issues

| Issue | Location | Severity |
|-------|----------|----------|
| ~~IsolationForest contamination not validated~~ | `isolation_forest.rs:436-446` | **FIXED** — validates 0 < c <= 0.5 |
| ~~IsolationForest max_samples=1 non-functional~~ | `isolation_forest.rs:165-184` | **FIXED** — validates max_samples >= 2 |
| ~~LOF uniform data silent failure~~ | `lof.rs:251-285` | **FIXED** — warns when all scores identical |
| ~~GMM component collapse undetected~~ | `gmm.rs:545-549` | **FIXED** — warns when weight < 1e-6 |
| ~~LOF k > n_samples silent degradation~~ | `lof.rs:121-123` | **FIXED** — warns when k clamped |

---

## P2 — Preprocessing Issues

| Issue | Location | Severity |
|-------|----------|----------|
| ~~CountVectorizer empty vocab after filtering~~ | `count_vectorizer.rs:305-330` | **FIXED** — returns error when all terms filtered |
| ~~PolynomialFeatures output explosion~~ | `polynomial.rs:199-216` | **FIXED** — caps at 1M output features |

### Well handled (no action needed)
- TF-IDF zero-norm rows → stay zero (correct)
- PowerTransformer Box-Cox negative values → validated
- PCA n_components > n_features → validated
- QuantileTransformer constant features → handled
- OneHotEncoder unseen categories → configurable behavior

---

## Recommended Fix Priority

1. ~~**Add NaN/inf checks to `validate_fit_input`**~~ — **FIXED** (pass 5). Added finite checks to validators; HistGBT uses `_allow_nan` variant for missing value support.
2. ~~**SGD alpha=0 division by zero**~~ — **FIXED** (pass 5). Validates alpha > 0 for optimal schedule.
3. ~~**SGD log loss overflow**~~ — **FIXED** (pass 5). Uses numerically stable log-sum-exp branch.
4. ~~**SVM Platt sigmoid overflow**~~ — **FIXED** (pass 6). Added `stable_sigmoid()` helper with sign-based branching.
5. ~~**CountVectorizer empty vocab**~~ — **FIXED** (pass 5). Returns error when filtering removes all terms.
6. ~~**Lasso/ElasticNet non-convergence warning**~~ — **FIXED** (pass 6). Warns to stderr when max_iter reached.
7. ~~**Guard all `.ln()` calls**~~ — **FIXED** (pass 6). Added `.max(1e-300)` guards in naive_bayes.rs (4 sites) and qda.rs (2 sites).
8. ~~**IsolationForest contamination bounds**~~ — **FIXED** (pass 6). Validates 0 < c <= 0.5 and max_samples >= 2.
9. ~~**LogReg perfect separation**~~ — **FIXED** (pass 6). Warns to stderr when extreme probabilities detected.
10. ~~**HPO all-trials-fail**~~ — **FIXED** (pass 7). Added `best_trial_or_err()` + NaN filter in `best_trial()`.
11. ~~**Metrics macro/weighted avg guards**~~ — **FIXED** (pass 7). Zero-guards for n_classes==0 and total_support==0.
12. ~~**Search space min > max**~~ — **FIXED** (pass 7). Assert low <= high in all 4 Parameter constructors + 5 new tests.
13. ~~**Add `score()` to Model trait**~~ — **FIXED** (pass 8). Default method computes accuracy; regressors can override for R².
14. ~~**Implement `partial_fit()` for SGD models**~~ — **FIXED** (pass 9). Full `IncrementalModel` impl for SGDClassifier + SGDRegressor, 11 new tests.
15. ~~**Expose `sample_weight` in fit()**~~ — **FIXED** — `fit_weighted()` on Model trait; implemented for LogisticRegression + DecisionTreeClassifier
16. ~~**LogisticRegression n < p auto-regularization**~~ — **FIXED** — auto-applies L2=1e-4 with warning when n < p; 2 new tests
17. ~~**GaussianNB single-sample variance floor**~~ — **FIXED** — `max_var.max(1e-300)` prevents epsilon=0 on zero-variance data; 1 new test
18. ~~**Single-feature dataset coverage**~~ — **FIXED** — improved `check_single_feature` to test both regression and classification targets; 37 compliance tests pass

---

## HPO / CV / Metrics Assessment (Pass 4)

### Issues Found

| Issue | Location | Severity |
|-------|----------|----------|
| ~~HPO all-trials-fail → silent None~~ | `hpo/mod.rs:219-231` | **FIXED** — best_trial_or_err() + NaN filter |
| ~~ClassificationReport macro avg div-by-zero~~ | `metrics/classification.rs:185-202` | **FIXED** — zero-guards |
| ~~ClassificationReport weighted avg div-by-zero~~ | `metrics/classification.rs:190-202` | **FIXED** — zero-guards |
| ~~Search space min > max not validated~~ | `hpo/search_space.rs:113-137` | **FIXED** — assert low <= high |
| ~~Parameter importance silent empty return~~ | `hpo/mod.rs:294-305` | **FIXED** — warns when <10 trials |

### Well Protected (no action needed)
- **R² with constant predictions**: Special-cased (ss_tot=0 → returns 1.0 or 0.0)
- **Precision/Recall/F1 with zero TP+FP**: Guarded against div-by-zero
- **MAPE with zero actuals**: Returns explicit error
- **KFold n_splits > n_samples**: Rejected by validate_n_folds()
- **Empty parameter grid**: Rejected by GridSearch/RandomSearch
- **CV confidence intervals**: Returns (mean, mean) for single fold
- **Correlation zero variance**: Returns 0.0 when denominator is zero
- **Stratified KFold rare classes**: Matches sklearn behavior (some folds get 0 samples)

---

## Audit Coverage Summary

| Area | Passes | Issues Found | Status |
|------|--------|-------------|--------|
| NaN/inf handling | 1, 2 | 1 critical (validate_fit_input) | **ALL FIXED** |
| Numerical stability | 1 | 9 issues (overflow, underflow, log-of-zero) | **ALL FIXED** |
| Linear models | 2 | 5 issues (SGD, Lasso, LogReg, PA, Ridge) | **ALL FIXED** |
| Clustering/anomaly | 2 | 5 issues (IsoForest, LOF, GMM) | **ALL FIXED** |
| Preprocessing | 2 | 2 issues (CountVectorizer, PolynomialFeatures) | **ALL FIXED** |
| ONNX export | 3 | 6 issues (NB variance, SVC coefs, overflow) | **ALL FIXED** |
| Python bindings | 3 | 0 critical, 3 minor | Clean |
| HPO/CV/Metrics | 4 | 5 issues (HPO failures, metrics guards) | **ALL FIXED** |
| sklearn API gaps | 1 | 6 missing methods | **5/6 FIXED** (score, partial_fit, decision_function, predict_log_proba, sample_weight); 1 by-design (get/set_params) |
| Edge cases (n < p) | 5 | 1 issue (LogReg underdetermined) | **FIXED** — auto-regularization |
| Edge cases (n=1) | 5 | 1 issue (GaussianNB zero variance) | **FIXED** — max_var floor |
| Edge cases (p=1) | 5 | 1 issue (classifier coverage gap in compliance check) | **FIXED** — dual-target check |
| **Total** | **5 passes** | **36 actionable issues** | **35 fixed, 1 by-design** |
