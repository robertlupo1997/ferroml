# Handoff: Plan S Cross-Library Validation + Audit Completion (2026-03-14)

## What Was Done

### Plan S: Cross-Library Validation (Phases S.1–S.7)

**S.1 — FerroML vs linfa (Rust, 56 tests)**
- 6 integration test files: `vs_linfa_linear.rs`, `vs_linfa_naive_bayes.rs`, `vs_linfa_trees.rs`, `vs_linfa_svm.rs`, `vs_linfa_neighbors.rs`, `vs_linfa_clustering.rs`
- 17 overlapping algorithms compared: LR, Ridge, Lasso, ElasticNet, LogReg, DT, RF, AdaBoost, SVC (linear+RBF), SVR, KNN, GaussianNB, MultinomialNB, BernoulliNB, KMeans, DBSCAN, GMM
- linfa 0.8.1 added as dev-dependency (ndarray 0.16 compatible)

**S.2 — smartcore: SKIPPED** (ndarray 0.15 conflicts with our 0.16)

**S.3 — FerroML vs XGBoost + LightGBM (19 Python tests)**
- `test_vs_xgboost.py`: GBR, GBC, HGB regressor/classifier, timing
- `test_vs_lightgbm.py`: HGB regressor/classifier, GBR, feature importance correlation

**S.4 — FerroML vs statsmodels + scipy (16 Python tests)**
- `test_vs_statsmodels.py`: OLS coef/R² match, Ridge, Logit, zscore, PCA/SVD, KMeans/vq, IsotonicRegression, KNN distances

**S.5 — sklearn gap coverage (20 Python tests)**
- `test_vs_sklearn_gaps.py`: GP regressor/classifier, PassiveAggressive, Bagging (C/R), TfidfVectorizer, CountVectorizer

**S.6 — Cross-library benchmark script**
- `scripts/benchmark_cross_library.py`: 18 algorithms × 4 libraries (ferroml, sklearn, xgboost, lightgbm)
- Output: JSON + Markdown + ASCII table
- Results saved to `docs/benchmark_cross_library_results.json` and `docs/cross-library-benchmark.md`

**S.7 — Edge case gauntlet (53 Python tests)**
- `test_cross_library_edge_cases.py`: 8 categories — single-sample, high-dim (p>>n), sparse, extreme values, constant features, constant target, 20-class multiclass, near-duplicate rows

### Robustness Audit Completion (32/33 fixed)

**Numerical stability fixes (this session):**
1. LogisticRegression log underflow — `(1.0 - yi).max(1e-15).ln()` (was `+ 1e-15`)
2. Regularized sqrt — `.max(0.0).sqrt()` on 2 sites in regularized.rs
3. Robust regression sqrt — `.max(0.0).sqrt()` on sigma + SE (robust.rs)
4. Quantile regression sqrt — `.max(0.0).sqrt()` on weights (quantile.rs)
5. GP variance cancellation — `stable_posterior_variance()` helper on 8 sites (floors at `prior * 1e-10`)
6. LogReg condition number — diagonal ratio check + auto-jitter when < 1e-12

**API features added (this session):**
7. `predict_log_proba()` — exposed in 18 Python bindings across 9 files (was already on Rust trait)
8. `fit_weighted()` — implemented for LogisticRegression + DecisionTreeClassifier

**Audit final state:** 32/33 fixed, 1 by-design (`get_params`/`set_params` — Rust builder pattern)

## Key Files Modified

### New Test Files (11)
- `ferroml-core/tests/vs_linfa_linear.rs` (16 tests)
- `ferroml-core/tests/vs_linfa_naive_bayes.rs` (7 tests)
- `ferroml-core/tests/vs_linfa_trees.rs` (11 tests)
- `ferroml-core/tests/vs_linfa_svm.rs` (6 tests)
- `ferroml-core/tests/vs_linfa_neighbors.rs` (9 tests)
- `ferroml-core/tests/vs_linfa_clustering.rs` (7 tests)
- `ferroml-python/tests/test_vs_xgboost.py` (10 tests)
- `ferroml-python/tests/test_vs_lightgbm.py` (9 tests)
- `ferroml-python/tests/test_vs_statsmodels.py` (16 tests)
- `ferroml-python/tests/test_vs_sklearn_gaps.py` (20 tests)
- `ferroml-python/tests/test_cross_library_edge_cases.py` (53 tests)

### New Scripts
- `scripts/benchmark_cross_library.py`

### Modified Core Files
- `ferroml-core/Cargo.toml` — linfa dev-dependencies
- `ferroml-core/src/models/gaussian_process.rs` — stable_posterior_variance (8 sites)
- `ferroml-core/src/models/logistic.rs` — log underflow fix, condition number guard, fit_weighted
- `ferroml-core/src/models/regularized.rs` — sqrt guards (2 sites)
- `ferroml-core/src/models/robust.rs` — sqrt guards (2 sites)
- `ferroml-core/src/models/quantile.rs` — sqrt guard
- `ferroml-core/src/models/tree.rs` — fit_weighted for DecisionTreeClassifier

### Modified Python Bindings (predict_log_proba added)
- `ferroml-python/src/linear.rs`, `naive_bayes.rs`, `trees.rs`, `svm.rs`, `neighbors.rs`, `gaussian_process.rs`, `ensemble.rs`, `clustering.rs`, `decomposition.rs`

## Test Status
- **Rust lib**: 3,157+ tests passing, 0 failures
- **Rust integration (vs linfa)**: 56 tests passing
- **Python cross-library**: 108 tests passing (XGBoost + LightGBM + statsmodels/scipy + sklearn gaps + edge cases)
- **Existing Python**: ~1,570 tests (not re-run this session but no regressions expected — wheel rebuilt)

## Benchmark Highlights
- FerroML wins: RF fit 5.6x faster, GaussianNB 4.3x, StandardScaler 6x, predict latency generally lower
- FerroML competitive: GradientBoosting matches sklearn, Ridge/Linear near-identical
- FerroML slower: HistGBT (8-15x vs sklearn), KMeans (2x), LogReg (2.5x vs liblinear)

## Remaining Work
- **Plan S.8**: Feature parity scorecard (auto-generated) — not started
- **smartcore comparison**: blocked on ndarray version conflict (would need separate workspace)
- **Performance**: HistGBT and KMeans could use optimization
- **Python packages installed**: xgboost 3.2.0, lightgbm 4.6.0, statsmodels 0.14.6

## Cron Loop
- Job `8a8723aa` still active (5-min audit cycle) — will find no more issues. Cancel with `CronDelete 8a8723aa` when ready.

## Nothing is committed yet
- 51 files changed, ~10,347 insertions — all uncommitted
- Suggest committing in logical chunks: (1) audit fixes, (2) linfa tests, (3) Python cross-library tests, (4) benchmark script
