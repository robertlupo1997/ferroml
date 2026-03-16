# Plan T: Remaining Work — Feature Parity, Performance, Polish

**Date:** 2026-03-14
**Status:** PENDING
**Depends on:** Plan S (complete)

---

## Overview

With the audit complete (35/36 fixed) and cross-library validation passing (164 tests),
this plan covers the remaining work items: feature parity scorecard, performance
optimization, and API coverage expansion.

---

## Phase T.1 — Feature Parity Scorecard (Plan S.8)

**Goal:** Auto-generate a comprehensive feature comparison matrix: FerroML vs sklearn vs
statsmodels vs XGBoost vs LightGBM.

**Tasks:**
1. Create `scripts/feature_parity_scorecard.py` that introspects FerroML's Python API
   and compares method availability against sklearn equivalents
2. Cover all 55+ models and 22+ preprocessors
3. Check for: `fit`, `predict`, `predict_proba`, `score`, `transform`,
   `inverse_transform`, `partial_fit`, `decision_function`, `fit_weighted`,
   `to_onnx_bytes`, `warm_start`, `feature_importances_`, `coef_`, `intercept_`
4. Output: Markdown table + JSON (save to `docs/feature-parity-scorecard.md`)
5. Identify top-5 missing features by impact

**Estimated scope:** 1 script, 1 doc file

---

## Phase T.2 — HistGradientBoosting Performance (8-15x slower than sklearn)

**Goal:** Close the largest performance gap. sklearn's HistGBT uses highly optimized
histogram-based splitting with SIMD. Target: within 3x of sklearn.

**Investigation tasks:**
1. Profile `HistGradientBoostingRegressor::fit()` with `cargo flamegraph` on the
   cross-library benchmark dataset (n=5000, p=20)
2. Identify hotspots: histogram construction, split finding, tree building
3. Potential optimizations:
   - Parallel histogram construction (rayon)
   - SIMD-accelerated bin counting
   - Reduce allocations in inner loops
   - Pre-sorted feature indices (like sklearn)
4. Benchmark before/after each optimization

**Files:** `ferroml-core/src/models/hist_boosting.rs`
**Estimated scope:** Medium-large — algorithmic changes to inner loops

---

## Phase T.3 — KMeans Performance (2x slower than sklearn)

**Goal:** Close the KMeans gap. sklearn uses Elkan's algorithm and C-optimized distance
computations.

**Investigation tasks:**
1. Profile `KMeans::fit()` on benchmark dataset
2. Potential optimizations:
   - Elkan's triangle inequality algorithm (skip distance computations)
   - Parallel assignment step (rayon)
   - Mini-batch KMeans variant for large datasets
   - SIMD distance computations
3. Benchmark before/after

**Files:** `ferroml-core/src/clustering/kmeans.rs`
**Estimated scope:** Medium — algorithmic improvement

---

## Phase T.4 — LogisticRegression Performance (2.5x slower than liblinear)

**Goal:** Close the LogReg gap. sklearn wraps liblinear (coordinate descent in C) or
lbfgs.

**Investigation tasks:**
1. Profile IRLS vs liblinear approach
2. Potential optimizations:
   - L-BFGS solver option (more efficient for large p)
   - Sparse-aware IRLS (avoid dense X'WX when X is sparse)
   - Convergence acceleration (Anderson mixing)
3. Benchmark before/after

**Files:** `ferroml-core/src/models/logistic.rs`
**Estimated scope:** Medium — alternative solver

---

## Phase T.5 — warm_start Expansion

**Goal:** Expand `warm_start` from 7 models to key models where it matters most.

**Priority models:**
1. LogisticRegression — reuse converged beta as initial guess
2. SVC/SVR — reuse support vectors for similar data
3. KMeans — reuse centroids for streaming scenarios
4. GaussianMixture — reuse component parameters
5. AdaBoost — append new weak learners

**Estimated scope:** Small per model — add `warm_start` field + conditional init

---

## Phase T.6 — feature_importances_ Expansion

**Goal:** Expose `feature_importances_` beyond trees/forests.

**Priority models:**
1. LinearRegression/Ridge/Lasso — absolute coefficient values (standardized)
2. LogisticRegression — absolute coefficient values
3. GradientBoosting — already has it via trees, verify exposure
4. SVC (linear kernel) — coefficient magnitudes

**Estimated scope:** Small — add accessor methods

---

## Execution Order

| Phase | Priority | Effort | Dependencies |
|-------|----------|--------|-------------|
| T.1   | High     | Small  | None        |
| T.2   | High     | Large  | Profiling tools |
| T.3   | Medium   | Medium | None        |
| T.4   | Medium   | Medium | None        |
| T.5   | Low      | Small  | None        |
| T.6   | Low      | Small  | None        |

**Recommended approach:** T.1 first (quick win, provides data for prioritizing T.2-T.6),
then T.2 (largest impact), then T.3/T.4 in parallel.

---

## Success Criteria

- [ ] Feature parity scorecard generated and saved
- [ ] HistGBT within 3x of sklearn on benchmark
- [ ] KMeans within 1.5x of sklearn on benchmark
- [ ] LogReg within 2x of sklearn on benchmark
- [ ] warm_start on 12+ models (from 7)
- [ ] feature_importances_ on linear models
- [ ] All existing tests still pass (3,186 Rust + ~1,678 Python)
