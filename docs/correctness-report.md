# FerroML v1.0 Correctness Audit Report

**Date:** 2026-03-27 (Phase 1), updated 2026-03-28 (Phase 2 complete)
**Phase:** 1 (Audit) + 2 (Bug Fixes + Robustness Hardening) — COMPLETE
**Scope:** All algorithms in ferroml-core/src/models/, clustering/, decomposition/, neural/, gpu/, sparse.rs
**Methodology:** 4-layer correctness framework (Textbook, Invariants, Edge Cases, Numerical Stability) + GPU/Sparse stability assessment

---

## Executive Summary

- **40 algorithms audited** across 6 families
- **40 PASS** — all ship in v1.0
- **0 FIX** — all 12 bugs resolved (3 P0/P1 + 5 P2 + 4 P3)
- **0 REMOVE** — no algorithm needs to be cut
- **GPU: EXPERIMENTAL** — ship with warning label
- **Sparse: STABLE** — ship as-is

### Phase 2 Hardening Summary

| Work Item | Scope | Sessions |
|---|---|---|
| P0/P1 bug fixes | 3 correctness bugs (Entropy, IRLS, AdaBoost) | Session 2 |
| P2 bug fixes | 5 bugs (z_inv_normal, SGD NaN, ExtraTrees, t-SNE 3D, GP jitter) | Session 3a |
| P3 bug fixes | 4 bugs (SGD t-counter, GP max_iter=0, LogReg L-BFGS, QuantileRegression assert!) | Session 4b |
| unwrap() elimination | 9 files (tree, hist_boosting, boosting, sgd, svm, knn, forest, extra_trees, adaboost) | Sessions 3b, 4a |
| Parameter validation | 19 model files, 30+ fit() methods | Session 4b |
| Empty data guards | isotonic, qda, calibration standardized | Session 4b |
| Layer 3 property tests | 25 tests across 7 categories | Session 5 |
| Layer 4 adversarial tests | 36 tests across 8 categories | Session 5 |

### Test Suite (as of Phase 2 completion)

| Suite | Count | Status |
|---|---|---|
| Library unit tests | 3,224 | All passing |
| Correctness tests | 297 | All passing |
| Edge case tests | 553 | All passing |
| **Rust total** | **4,074** | **All passing** |
| Python tests | ~2,100 | All passing |
| **Grand total** | **~6,174** | **All passing** |

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
| LogisticRegression | PASS | PASS | PASS | PASS | **PASS** |

\* LinearRegression uses Cholesky normal equations for n > 2p (less stable than QR/SVD for ill-conditioned X'X, but has rank-deficiency guard at 1e-14).

**LogisticRegression bugs (all fixed):**
1. **~~P1: IRLS weight clamping~~ FIXED** (Session 2) — Was `(var * sample_weights[i]).clamp(1e-10, 0.25)`. Fixed to `var * sample_weights[i]` (var already clamped on prior line). Regression test: `test_irls_class_weight_balanced_imbalanced_data`.
2. **~~P2: `z_inv_normal` sign error~~ FIXED** (Session 3a) — Original `p` overwritten before sign determination in `regularized.rs:2364`. Fixed by saving `original_p` before transform. Regression test verifies `z_inv_normal(0.025) ≈ -1.96`.
3. **~~P3: L-BFGS convergence detection~~ FIXED** (Session 4b) — Now uses argmin `TerminationReason` instead of iteration count heuristic.

### Specialized Regression

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| RobustRegression | PASS | PASS | PASS | PASS | **PASS** |
| QuantileRegression | PASS | PASS | PASS | PASS | **PASS** |
| IsotonicRegression | PASS | PASS | PASS | PASS | **PASS** |

**Notes:**
- QuantileRegression: `assert!` in constructor replaced with `FerroError::InvalidInput` at fit() (Session 4b).
- IsotonicRegression PAVA is O(n^2) worst case (uses restart loop instead of stack-based O(n)). Performance concern, not correctness.

### Trees & Boosting

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| DecisionTree | PASS | PASS | PASS | PASS | **PASS** |
| RandomForest | PASS | PASS | PASS | PASS | **PASS** |
| GradientBoosting | PASS | PASS | PASS | PASS | **PASS** |
| HistGradientBoosting | PASS | PASS | PASS | PASS | **PASS** |
| ExtraTrees | PASS | PASS | PASS | PASS | **PASS** |
| AdaBoost | PASS | PASS | PASS | PASS | **PASS** |

**DecisionTree bugs (all fixed):**
1. **~~P0: Entropy criterion not implemented~~ FIXED** (Session 2) — Added `weighted_entropy()` function and `match self.criterion` dispatch in `build_tree_weighted`, `find_best_split_weighted`, `find_random_split_weighted`. Regression test: `test_entropy_criterion_differs_from_gini`.

**AdaBoost bugs (all fixed):**
2. **~~P1: Regressor weight formula~~ FIXED** (Session 2) — Changed `beta.ln().abs()` to `(1.0_f64 / beta).ln()`. Regression test: `test_adaboost_regressor_weight_formula_bug3`.
3. **~~P1: Docstring said "SAMME.R"~~ FIXED** (Session 2) — Corrected to "SAMME (discrete)" in module and struct docs.

**ExtraTrees bugs (all fixed):**
4. **~~P2: `expect()` in parallel tree building~~ FIXED** (Session 3a) — Replaced with `filter_map` + empty guard (matches RandomForest pattern).

**Other tree concerns (addressed in Phase 2):**
- HistGBT `u8` bin / missing-value sentinel: missing_bin=255 collides with last valid bin when max_bins=255. Known limitation, documented.
- Tree recursive build depth: bounded by max_depth parameter. Default unbounded depth is a user configuration choice, not a bug.
- unwrap() eliminated/SAFETY-annotated across tree.rs (14 eliminated, 11 SAFETY), hist_boosting.rs (1 eliminated, 16 SAFETY), boosting.rs (2 eliminated, 4 SAFETY).

### SVM, KNN, SGD

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| SVC | PASS | PASS | PASS | PASS | **PASS** |
| SVR | PASS | PASS | PASS | PASS | **PASS** |
| KNeighborsClassifier | PASS | PASS | PASS | PASS | **PASS** |
| KNeighborsRegressor | PASS | PASS | PASS | PASS | **PASS** |
| SGDClassifier | PASS | PASS | PASS | PASS | **PASS** |
| SGDRegressor | PASS | PASS | PASS | PASS | **PASS** |

**SGD bugs (all fixed):**
5. **~~P2: SGDClassifier NaN/Inf divergence check~~ FIXED** (Session 3a) — Added finite check on coef+intercept after training (matches SGDRegressor).
6. **~~P3: Multiclass partial_fit t-counter~~ FIXED** (Session 4b) — t-counter now accumulates across all OvR classes instead of only saving last class.

**Notes:**
- SVM documentation inconsistency (FULL_MATRIX_THRESHOLD): cosmetic, non-blocking.
- unwrap() eliminated/SAFETY-annotated in sgd.rs, svm.rs, knn.rs.

### Naive Bayes

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| GaussianNB | PASS | PASS | PASS | PASS | **PASS** |
| MultinomialNB | PASS | PASS | PASS | PASS | **PASS** |
| BernoulliNB | PASS | PASS | PASS | PASS | **PASS** |
| CategoricalNB | PASS | PASS | PASS | PASS | **PASS** |

All NB models use log-space computation, logsumexp normalization, and proper Laplace smoothing.

### Gaussian Processes, QDA, Calibration, Anomaly Detection

| Algorithm | Textbook Match | Invariants | Edge Cases | Numerical Stability | Verdict |
|---|---|---|---|---|---|
| GP Regressor | PASS | PASS | PASS | PASS | **PASS** |
| GP Classifier | PASS | PASS | PASS | PASS | **PASS** |
| QDA | PASS | PASS | PASS | PASS | **PASS** |
| Calibration | PASS | PASS | PASS | PASS | **PASS** |
| LOF | PASS | PASS | PASS | PASS | **PASS** |
| IsolationForest | PASS | PASS | PASS | PASS | **PASS** |

**GP bugs (all fixed):**
7. **~~P2: GP Regressor Cholesky jitter retry~~ FIXED** (Session 3a) — Added jitter retry loop (1e-10 → 1e-4) wrapping raw Cholesky.
8. **~~P3: GP Classifier max_iter=0~~ FIXED** (Session 4b) — max_iter=0 rejected at fit(); is_fitted() checks l_.is_some() as defense-in-depth.

**Notes:**
- WhiteKernel row equality check is fragile — only works when X1 and X2 are the same array. Known limitation.
- QDA empty data guard added (Session 4b).
- GP Regressor numerical stability upgraded from CONCERN to PASS after jitter retry fix.

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

**t-SNE bugs (all fixed):**
9. **~~P2: Barnes-Hut hardcoded to 2D~~ FIXED** (Session 3a) — Falls back to exact method when n_components > 2 (matches sklearn behavior).

**Notes:**
- t-SNE non-Euclidean metrics: Manhattan and Cosine distances are squared before perplexity calibration, which assumes squared Euclidean. Known limitation, documented.
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
| Cholesky jitter for GP/GMM | PASS | GMM has reg_covar; GP jitter retry added (Session 3a) |
| Euclidean distance clamped | PASS | KMeans clamps squared distances >= 0 |
| Log-space probability | PASS | All classifiers use log-space |
| Regularization in GBT leaf weights | PASS | HistGBT: -G/(H+lambda), H clamped >= 1e-8 |
| Empty cluster handling in KMeans | PASS | Random reassignment in Lloyd and Elkan |

Note: "Cholesky jitter for GP/GMM" upgraded from PARTIAL to PASS after GP jitter retry fix in Session 3a.

---

## All Bugs by Priority — ALL FIXED

### P0 — ~~Must fix before v1.0~~ FIXED

| # | Algorithm | Description | Fixed In |
|---|---|---|---|
| 1 | DecisionTree | ~~Entropy criterion silently ignored~~ — weighted_entropy + criterion dispatch added | Session 2 |

### P1 — ~~Should fix before v1.0~~ ALL FIXED

| # | Algorithm | Description | Fixed In |
|---|---|---|---|
| 2 | LogisticRegression | ~~IRLS weight clamping~~ — clamp var alone, multiply by sample_weight after | Session 2 |
| 3 | AdaBoost Regressor | ~~Weight formula uses `.abs()`~~ — uses `(1.0/beta).ln()` | Session 2 |
| 4 | AdaBoost | ~~Docstring claims SAMME.R~~ — corrected to SAMME | Session 2 |

### P2 — ~~Should fix, non-blocking~~ ALL FIXED

| # | Algorithm | Description | Fixed In |
|---|---|---|---|
| 5 | z_inv_normal | ~~Sign error for p > 0.5~~ — saved original_p before transform | Session 3a |
| 6 | SGDClassifier | ~~Missing NaN/Inf divergence check~~ — added finite check matching SGDRegressor | Session 3a |
| 7 | ExtraTrees | ~~`expect()` in parallel tree building~~ — replaced with filter_map + empty guard | Session 3a |
| 8 | t-SNE | ~~Barnes-Hut hardcoded to 2D~~ — falls back to exact method when n_components > 2 | Session 3a |
| 9 | GP Regressor | ~~No Cholesky jitter retry~~ — added jitter retry loop (1e-10 → 1e-4) | Session 3a |

### P3 — ~~Nice to fix~~ ALL FIXED

| # | Algorithm | Description | Fixed In |
|---|---|---|---|
| 10 | LogisticRegression | ~~L-BFGS convergence uses iter count~~ — uses argmin TerminationReason | Session 4b |
| 11 | GP Classifier | ~~max_iter=0 causes panic~~ — rejected at fit(); is_fitted() checks l_.is_some() | Session 4b |
| 12 | SGD | ~~Multiclass partial_fit t-counter~~ — accumulates across all OvR classes | Session 4b |

### Additional fix (found during Phase 2)

| # | Algorithm | Description | Fixed In |
|---|---|---|---|
| 13 | QuantileRegression | `assert!` panic in constructor — replaced with FerroError::InvalidInput at fit() | Session 4b |

---

## Phase 2 Robustness Hardening

### unwrap() Elimination (Sessions 3b, 4a)

All production `unwrap()`/`expect()` calls in model code have been reviewed. Dangerous calls were replaced with proper error propagation (`?` operator, `ok_or(FerroError::...)`). Remaining calls are annotated with `// SAFETY:` comments explaining why they cannot fail.

| File | Before | After (eliminated / SAFETY-annotated) |
|---|---|---|
| tree.rs | 92 | 14 eliminated, 11 SAFETY |
| hist_boosting.rs | 92 | 1 eliminated, 16 SAFETY |
| boosting.rs | 56 | 2 eliminated, 4 SAFETY |
| sgd.rs | 35 | Reviewed, SAFETY-annotated |
| svm.rs | 25 | Reviewed, SAFETY-annotated |
| knn.rs | 19 | Reviewed, SAFETY-annotated |
| forest.rs | 40 | Reviewed, SAFETY-annotated |
| extra_trees.rs | 34 | Reviewed, SAFETY-annotated |
| adaboost.rs | 32 | Reviewed, SAFETY-annotated |

### Parameter Validation (Session 4b)

Added fit()-time parameter validation to **19 model files** (30+ fit methods):
- Reject invalid hyperparameters (negative alpha, zero max_iter, bad learning_rate) with `FerroError::InvalidInput`
- Standardized empty/NaN/Inf validation (isotonic, qda, calibration now use `validate_fit_input`)
- Calibration helpers check NaN/Inf on probability inputs

Files validated: linear.rs, regularized.rs, logistic.rs, robust.rs, quantile.rs, isotonic.rs, tree.rs, forest.rs, boosting.rs, hist_boosting.rs, extra_trees.rs, adaboost.rs, svm.rs, knn.rs, sgd.rs, gaussian_process.rs, qda.rs, calibration.rs, naive_bayes/*.rs

### Pre-Existing Test Failures (Session 4a)

The 6 pre-existing failures (TemperatureScaling, IncrementalPCA) from `test_vs_sklearn_gaps_phase2.py` now **all pass** (12/12 tests).

---

## Layer 3: Property/Invariant Tests (Session 5)

25 tests in `correctness.rs` verifying mathematical invariants:

| Category | Tests | What's Verified |
|---|---|---|
| Classifier predict_proba | 8 | Probabilities sum to 1.0 for LogReg, RF, GBT, GaussianNB, SVC, AdaBoost, ExtraTrees, MLP |
| Iterative convergence | 3 | KMeans inertia decreases, GBT loss decreases, LogReg converges |
| PCA properties | 4 | Components orthogonal, variance sorted descending, unit norms, explained_variance_ratio sums ≤ 1 |
| Tree feature importances | 3 | Non-negative, sum to ~1 for DecisionTree, RandomForest, GradientBoosting |
| Anomaly scores | 2 | Finite scores for IsolationForest, LOF |
| Ridge shrinkage | 2 | alpha≈0 matches OLS, alpha→∞ coefficients approach 0 |
| SVM validity | 3 | Valid class labels, finite decision function |

## Layer 4: Adversarial/Edge Case Tests (Session 5)

36 tests in `edge_cases.rs` verifying robustness on pathological inputs:

| Category | Tests | What's Tested |
|---|---|---|
| Near-collinear features | 5 | condition number > 1e12 (LinearReg, Ridge, Lasso, ElasticNet, LogReg) |
| Single sample | 5 | n=1 (Ridge, KNN, DecisionTree, GaussianNB, KMeans) |
| Constant features | 4 | All features identical (PCA, StandardScaler, DecisionTree, LogReg) |
| Single class | 5 | Only one class label (LogReg, RF, GBT, SVC, GaussianNB) |
| Very large values (1e15) | 4 | StandardScaler, MinMaxScaler, Ridge, KMeans |
| Very small values (1e-15) | 3 | GaussianNB, PCA, DecisionTree |
| Rank-deficient (p > n) | 4 | More features than samples (Ridge, Lasso, ElasticNet, PCA) |
| Single feature | 6 | p=1 (LinearReg, LogReg, DecisionTree, KNN, SVC, KMeans) |

---

## Recommendations for Phase 3

Phase 2 is complete. All 12 audit bugs are fixed, all P3 included. The next phase per the master design is:

### Phase 3: Frankenstein Tests + Final Validation (Sessions 6-8)
1. **Build Layer 5 Frankenstein test suite** — pipeline composition, stateful interactions, AutoML end-to-end, serialization under composition, cross-module performance
2. Fix RandomForest non-determinism
3. Fix MLP serialization
4. Fix performance regressions exceeding 5x (SVC 7.7x, KMeans 6.84x)
5. Thread safety verification (concurrent predict() from Python threads)
6. Final cross-library validation sweep
7. Update this report with all 5 layers
