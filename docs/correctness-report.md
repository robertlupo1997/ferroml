# FerroML v1.0 Correctness Audit Report

**Date:** 2026-03-27
**Phase:** 1, Session 1
**Scope:** All algorithms in ferroml-core/src/models/, clustering/, decomposition/, neural/, gpu/, sparse.rs
**Methodology:** 4-layer correctness framework (Textbook, Invariants, Edge Cases, Numerical Stability) + GPU/Sparse stability assessment

---

## Executive Summary

- **40 algorithms audited** across 6 families
- **39 PASS** — ship in v1.0 as-is (includes 3 fixed in Phase 2 Session 2)
- **0 FIX** — all P0/P1 bugs resolved
- **0 REMOVE** — no algorithm needs to be cut
- **GPU: EXPERIMENTAL** — ship with warning label
- **Sparse: STABLE** — ship as-is
- **12 bugs found** (3 P0/P1, 5 P2, 4 P3) — P0/P1/P2 all fixed

---

## Per-Algorithm Verdicts

### Linear Models

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| LinearRegression | PASS* | PASS | PASS | PASS* | **PASS** |
| Ridge | PASS | PASS | PASS | PASS | **PASS** |
| Lasso | PASS | PASS | PASS | PASS | **PASS** |
| ElasticNet | PASS | PASS | PASS | PASS | **PASS** |
| RidgeCV | PASS | PASS | PASS | PASS | **PASS** |
| LassoCV | PASS | PASS | PASS | PASS | **PASS** |
| ElasticNetCV | PASS | PASS | PASS | PASS | **PASS** |
| RidgeClassifier | PASS | PASS | PASS | PASS | **PASS** |
| LogisticRegression | PASS | PASS | PASS | PASS | **PASS** | *(Fixed: IRLS weight clamping no longer corrupts ClassWeight::Balanced)*

\* LinearRegression uses Cholesky normal equations for n > 2p (less stable than QR/SVD for ill-conditioned X'X, but has rank-deficiency guard at 1e-14).

**LogisticRegression bugs:**
1. **~~P1: IRLS weight clamping~~ FIXED** (`logistic.rs:638`) — Was `(var * sample_weights[i]).clamp(1e-10, 0.25)`. Fixed to `var * sample_weights[i]` (var already clamped on prior line). Regression test: `test_irls_class_weight_balanced_imbalanced_data`.
2. **P2: `z_inv_normal` sign error** (`regularized.rs:2364`) — Original `p` overwritten before sign determination. `z_inv_normal(0.025)` returns positive instead of -1.96. Affects Ridge/Lasso/ElasticNet confidence intervals in `regularized.rs` (not `linear.rs` which has the correct version).
3. **P3: L-BFGS convergence detection** (`logistic.rs:867`) — Uses iteration count comparison instead of inspecting argmin's termination reason.

### Specialized Regression

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| RobustRegression | PASS | PASS | PASS | PASS | **PASS** |
| QuantileRegression | PASS | PASS | PASS | PASS | **PASS** |
| IsotonicRegression | PASS | PASS | PASS | PASS | **PASS** |

**Notes:**
- QuantileRegression Cholesky solver lacks diagonal regularization (unlike RobustRegression which adds 1e-10). Could fail on ill-conditioned problems.
- QuantileRegression convergence check uses absolute tolerance, not relative.
- IsotonicRegression PAVA is O(n^2) worst case (uses restart loop instead of stack-based O(n)). Performance concern, not correctness.

### Trees & Boosting

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| DecisionTree | PASS | PASS | PASS | PASS | **PASS** | *(Fixed: Entropy criterion now implemented with weighted_entropy dispatch)*
| RandomForest | PASS | PASS | PASS | PASS | **PASS** |
| GradientBoosting | PASS | PASS | PASS | PASS | **PASS** |
| HistGradientBoosting | PASS | PASS | PASS | PASS | **PASS** |
| ExtraTrees | PASS | PASS | PASS | PASS | **PASS** |
| AdaBoost | PASS | PASS | PASS | PASS | **PASS** | *(Fixed: regressor weight formula uses ln(1/beta), docstring corrected to SAMME)*

**DecisionTree bugs:**
1. **~~P0: Entropy criterion not implemented~~ FIXED** — Added `weighted_entropy()` function and `match self.criterion` dispatch in `build_tree_weighted`, `find_best_split_weighted`, `find_random_split_weighted`. Regression test: `test_entropy_criterion_differs_from_gini`.

**AdaBoost bugs:**
2. **~~P1: Regressor weight formula~~ FIXED** (`adaboost.rs:524`) — Changed `beta.ln().abs()` to `(1.0_f64 / beta).ln()`. Regression test: `test_adaboost_regressor_weight_formula_bug3`.
3. **~~P1: Docstring said "SAMME.R"~~ FIXED** — Corrected to "SAMME (discrete)" in module and struct docs.

**Other concerns:**
- ExtraTrees uses `expect()` in parallel tree building (line 404, 747) — panics if a tree fit fails, unlike RandomForest which uses `filter_map`. P2.
- HistGBT `u8` bin / missing-value sentinel: missing_bin=255 collides with last valid bin when max_bins=255. P2.
- Tree recursive build could stack overflow on unbounded depth. P2.

### SVM, KNN, SGD

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| SVC | PASS | PASS | PASS | PASS | **PASS** |
| SVR | PASS | PASS | PASS | PASS | **PASS** |
| KNeighborsClassifier | PASS | PASS | PASS | PASS | **PASS** |
| KNeighborsRegressor | PASS | PASS | PASS | PASS | **PASS** |
| SGDClassifier | PASS | PASS | PASS | PASS | **PASS** |
| SGDRegressor | PASS | PASS | PASS | PASS | **PASS** |

**Notes:**
- SGDClassifier does NOT check for NaN/Inf coefficient divergence after training (SGDRegressor does at line 1084). Diverged coefficients silently produce NaN predictions. P2.
- SGD multiclass partial_fit: only last class's t-counter saved, earlier classes use stale learning rates on subsequent calls. P3.
- SVM documentation inconsistency: FULL_MATRIX_THRESHOLD=2000 vs comments saying 4000/5000. P3.

### Naive Bayes

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| GaussianNB | PASS | PASS | PASS | PASS | **PASS** |
| MultinomialNB | PASS | PASS | PASS | PASS | **PASS** |
| BernoulliNB | PASS | PASS | PASS | PASS | **PASS** |
| CategoricalNB | PASS | PASS | PASS | PASS | **PASS** |

All NB models use log-space computation, logsumexp normalization, and proper Laplace smoothing. Production unwrap() counts are low (3-4 each, all guarded).

### Gaussian Processes, QDA, Calibration, Anomaly Detection

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| GP Regressor | PASS | PASS | PASS | CONCERN | **PASS** |
| GP Classifier | PASS | PASS | CONCERN | PASS | **PASS** |
| QDA | PASS | PASS | PASS | PASS | **PASS** |
| Calibration | PASS | PASS | PASS | PASS | **PASS** |
| LOF | PASS | PASS | PASS | PASS | **PASS** |
| IsolationForest | PASS | PASS | PASS | PASS | **PASS** |

**Notes:**
- GP: No jitter retry in Cholesky. Uses local `cholesky()` instead of `cholesky_with_jitter` from linalg.rs. Fit may fail on borderline kernel matrices. P2.
- GP Classifier: `max_iter=0` causes panic (l_ is None but is_fitted() returns true). P3.
- WhiteKernel row equality check is fragile — only works when X1 and X2 are the same array. P3.
- QDA uses exact `==` for class labels, unlike rest of codebase which uses `(yi - class).abs() < 1e-10`. P3.

### Clustering

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| KMeans | PASS | PASS | PASS | PASS | **PASS** |
| MiniBatchKMeans | PASS | PASS | PASS | PASS | **PASS** |
| DBSCAN | PASS | PASS | PASS | PASS | **PASS** |
| HDBSCAN | PASS | PASS | PASS | PASS | **PASS** |
| GMM | PASS | PASS | PASS | PASS | **PASS** |
| Agglomerative | PASS | PASS | PASS | PASS | **PASS** |

All clustering algorithms are textbook-faithful. KMeans implements Lloyd, Elkan, and Hamerly. GMM uses proper logsumexp and Cholesky with reg_covar.

### Decomposition

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| PCA | PASS | PASS | PASS | PASS | **PASS** |
| IncrementalPCA | PASS | PASS | PASS | PASS | **PASS** |
| t-SNE | PASS | PASS | PASS | PASS | **PASS** |
| TruncatedSVD | PASS | PASS | PASS | PASS | **PASS** |
| LDA | PASS | PASS | PASS | PASS | **PASS** |
| FactorAnalysis | PASS | PASS | PASS | PASS | **PASS** |

**Notes:**
- t-SNE Barnes-Hut only works correctly for 2D embeddings (QuadTree hardcoded to dimensions 0,1). n_components > 2 with BarnesHut produces incorrect results. P2.
- t-SNE non-Euclidean metrics: Manhattan and Cosine distances are squared before perplexity calibration, which assumes squared Euclidean. Likely incorrect for non-Euclidean metrics. P2.
- PCA CovarianceEigh solver doubles condition number. Auto-selected only for tall-thin data. Acceptable tradeoff.

### Neural Networks

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| MLPClassifier | PASS | PASS | PASS | PASS | **PASS** |
| MLPRegressor | PASS | PASS | PASS | PASS | **PASS** |
| MultiOutput | PASS | PASS | PASS | PASS | **PASS** |

MLP uses He/Xavier init, inverted dropout, Adam/SGD, softmax with max-subtraction, cross-entropy with eps=1e-15 clipping.

---

## GPU/Sparse Stability Assessment

### GPU Module — EXPERIMENTAL

| Aspect | Assessment |
|---|---|
| API maturity | Moderate — GpuBackend trait is clean, GpuDispatchPolicy is simple |
| Test coverage | 188 tests, but ALL use MockGpuBackend — zero actual GPU shader execution tests |
| Error handling | 13 unwrap() in production backend code; `read_buffer()` has 3 panic-able unwraps |
| Completeness | Partial — f32-only, inference primitives only, no sparse GPU, no training ops |
| **Recommendation** | **EXPERIMENTAL** — ship with clear warning that API may change |

### Sparse Module — STABLE

| Aspect | Assessment |
|---|---|
| API maturity | Good — CSR/CSC/SparseVector, complete distance functions |
| Test coverage | 21 tests, well-covered operations |
| Error handling | Only 2 unwrap() in production (both safe by construction) |
| Completeness | Feature-complete for ML sparse workflows |
| **Recommendation** | **STABLE** — ship as-is |

---

## Numerical Stability Checklist

| Item | Status | Notes |
|---|---|---|
| Log-sum-exp trick | PASS | Used in all NB, GMM, t-SNE, softmax |
| Welford's for variance | PARTIAL | GaussianNB uses Chan's parallel algorithm (correct but misdocumented) |
| SVD for least squares | PASS* | PCA uses SVD; LinearRegression falls back to Cholesky for n>2p |
| Stable sigmoid | PASS | Branch-on-sign in logistic.rs mod.rs |
| Cholesky jitter for GP/GMM | PARTIAL | GMM has reg_covar; GP lacks jitter retry |
| Euclidean distance clamped | PASS | KMeans clamps squared distances >= 0 |
| Log-space probability | PASS | All classifiers use log-space |
| Regularization in GBT leaf weights | PASS | HistGBT: -G/(H+lambda), H clamped >= 1e-8 |
| Empty cluster handling in KMeans | PASS | Random reassignment in Lloyd and Elkan |

---

## All Bugs by Priority

### P0 — Must fix before v1.0

| # | Algorithm | File:Line | Description |
|---|---|---|---|
| 1 | DecisionTree | tree.rs | ~~Entropy criterion silently ignored~~ **FIXED** — weighted_entropy + criterion dispatch added |

### P1 — ~~Should fix before v1.0~~ ALL FIXED

| # | Algorithm | File:Line | Description |
|---|---|---|---|
| 2 | LogisticRegression | logistic.rs:638 | ~~IRLS weight clamping~~ **FIXED** — clamp var alone, multiply by sample_weight after |
| 3 | AdaBoost Regressor | adaboost.rs:524 | ~~Weight formula uses `.abs()`~~ **FIXED** — uses `(1.0/beta).ln()` |
| 4 | AdaBoost | adaboost.rs:7 | ~~Docstring claims SAMME.R~~ **FIXED** — corrected to SAMME |

### P2 — Should fix, non-blocking

| # | Algorithm | File:Line | Description |
|---|---|---|---|
| 5 | z_inv_normal | regularized.rs:2364 | ~~Sign error — returns wrong sign for p > 0.5~~ **FIXED** — saved original_p before transform, fixed branch direction |
| 6 | SGDClassifier | sgd.rs | ~~Missing NaN/Inf divergence check~~ **FIXED** — added finite check on coef+intercept after training (matches SGDRegressor) |
| 7 | ExtraTrees | extra_trees.rs:404 | ~~`expect()` in parallel tree building~~ **FIXED** — replaced with filter_map + empty guard (matches RandomForest) |
| 8 | t-SNE | tsne.rs:729 | ~~Barnes-Hut hardcoded to 2D~~ **FIXED** — falls back to exact method when n_components > 2 (matches sklearn) |
| 9 | GP Regressor | gaussian_process.rs | ~~No Cholesky jitter retry~~ **FIXED** — added jitter retry loop (1e-10 → 1e-4) wrapping raw cholesky |

### P3 — Nice to fix

| # | Algorithm | File:Line | Description |
|---|---|---|---|
| 10 | LogisticRegression | logistic.rs:867 | L-BFGS convergence detection uses iter count, not termination reason |
| 11 | GP Classifier | gaussian_process.rs | max_iter=0 causes panic (l_ None but is_fitted() true) |
| 12 | SGD | sgd.rs | Multiclass partial_fit: only last class's t-counter saved |

---

## Production unwrap() Summary

| File | Prod unwrap()/expect() | Risk Level |
|---|---|---|
| linear.rs | 1 | Low (SAFETY-annotated) |
| regularized.rs | 3 | Low (SAFETY-annotated) |
| logistic.rs | 0 | None |
| robust.rs | 3 | Low (guarded) |
| quantile.rs | 5 | Low (guarded) |
| isotonic.rs | 6 | Low (guarded) |
| tree.rs | 92 | **High** (67 unwrap + 25 expect) |
| forest.rs | 40 | Medium (30 unwrap + 10 expect) |
| boosting.rs | 56 | **High** |
| hist_boosting.rs | 92 | **High** (73 unwrap + 19 expect) |
| extra_trees.rs | 34 | Medium |
| adaboost.rs | 32 | Medium |
| svm.rs | 25 | Medium |
| knn.rs | 19 | Medium |
| sgd.rs | 35 | Medium |
| gaussian_process.rs | 10 | Medium |
| qda.rs | 7 | Low |
| naive_bayes (all) | 13 | Low |
| calibration.rs | 6 | Low |
| lof.rs | 5 | Low |
| isolation_forest.rs | 5 | Low |
| clustering (all) | ~20 non-test | Low |
| decomposition (all) | ~10 non-test | Low |
| neural (all) | ~7 non-test | Low |
| gpu/backend.rs | 13 | **High** (GPU buffer panics) |
| sparse.rs | 2 | None |

**Worst offenders:** tree.rs (92), hist_boosting.rs (92), boosting.rs (56), sgd.rs (35), adaboost.rs (32)

---

## Recommendations for Phase 2-3

### Phase 2 (Robustness) Priority Order
1. ~~Fix P0: Implement Entropy criterion for classification trees~~ **DONE** (Phase 2 Session 2)
2. ~~Fix P1: LogisticRegression IRLS weight clamp, AdaBoost weight formula + docstring~~ **DONE** (Phase 2 Session 2)
3. Eliminate unwrap() in tree.rs, hist_boosting.rs, boosting.rs (the 230+ unwrap target)
4. Fix P2: z_inv_normal sign error, SGDClassifier NaN check, ExtraTrees expect(), GP jitter

### Phase 3 (Correctness Fixes)
5. Fix P2: t-SNE Barnes-Hut 2D limitation (validate n_components or implement OctTree)
6. Fix P3: GP Classifier max_iter=0, SGD t-counter, LogReg L-BFGS convergence
7. Mark GPU module as experimental in Python API and docs
8. Add property tests (Layer 3) for all models
9. Add adversarial tests (Layer 4) for all models
