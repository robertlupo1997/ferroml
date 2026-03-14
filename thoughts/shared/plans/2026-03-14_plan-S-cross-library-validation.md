# Plan S: Cross-Library Validation — FerroML vs The World

## Overview

Exhaustive correctness and performance validation of FerroML against every competing ML library with overlapping algorithms. This proves FerroML produces correct results and documents its competitive position across the entire ML ecosystem.

**Libraries Under Test:**
- **Rust**: linfa (v0.7.1), smartcore (v0.4.9)
- **Python**: scikit-learn (v1.8.0), XGBoost, LightGBM, statsmodels, scipy
- **Benchmarks**: Performance comparison at 1K / 10K / 100K samples

**Testing Dimensions:**
1. Correctness — predictions match within justified tolerances
2. Performance — fit/predict timing at multiple scales
3. Edge cases — NaN, single-sample, high-dim, sparse, degenerate
4. API completeness — feature parity scorecard

## Current State

- 450+ Python comparison tests exist against sklearn (5 test files)
- `scripts/benchmark_vs_sklearn.py` covers 10 models × 3 sizes
- `conftest_comparison.py` provides dataset loaders, comparison helpers, timing utilities
- Rust Criterion benchmarks: 86+ functions across 5 bench files
- **NO tests against linfa, smartcore, XGBoost, LightGBM, statsmodels, or scipy**
- linfa/smartcore have no Python bindings — must test via Rust dev-dependencies

## Desired End State

- Rust tests comparing FerroML vs linfa on all 17 overlapping algorithms
- Rust tests comparing FerroML vs smartcore on all 14 overlapping algorithms
- Python tests comparing FerroML vs XGBoost on gradient boosting (3 variants)
- Python tests comparing FerroML vs LightGBM on histogram boosting (2 variants)
- Python tests comparing FerroML vs statsmodels on linear/GLM models (6 algorithms)
- Python tests comparing FerroML vs scipy on statistical methods (4 algorithms)
- Expanded sklearn comparison to cover ALL 55+ FerroML models (currently 40)
- Cross-library performance benchmark with JSON + Markdown output
- Feature parity scorecard (API gap matrix)

---

## Implementation Phases

### Phase S.1: Rust — FerroML vs linfa (17 algorithms)

**Overview**: Add linfa subcrates as dev-dependencies and write Rust integration tests that train both FerroML and linfa on identical data, then compare predictions.

**Changes Required**:

1. **File**: `ferroml-core/Cargo.toml`
   - Add dev-dependencies:
     ```toml
     [dev-dependencies]
     linfa = "0.7"
     linfa-linear = "0.7"
     linfa-elasticnet = "0.7"
     linfa-logistic = "0.7"
     linfa-trees = "0.7"
     linfa-svm = "0.7"
     linfa-nn = "0.7"
     linfa-bayes = "0.7"
     linfa-reduction = "0.7"
     linfa-preprocessing = "0.7"
     linfa-clustering = "0.7"
     ```

2. **File**: `ferroml-core/tests/vs_linfa_linear.rs` (NEW)
   - LinearRegression: predictions atol=1e-6
   - Ridge: predictions atol=1e-5 across alpha=[0.01, 0.1, 1.0, 10.0]
   - Lasso: predictions atol=1.0 (coordinate descent divergence), sparsity pattern ±3
   - ElasticNet: predictions atol=1.0, l1_ratio sweep [0.1, 0.5, 0.9]
   - LogisticRegression: accuracy within 5%, probabilities within 0.1
   - Each at 3 dataset sizes: 200×10, 1000×50, 5000×100

3. **File**: `ferroml-core/tests/vs_linfa_trees.rs` (NEW)
   - DecisionTreeClassifier: accuracy within 5%
   - DecisionTreeRegressor: R² within 0.05
   - RandomForestClassifier: accuracy within 10% (stochastic)
   - RandomForestRegressor: R² within 0.10
   - AdaBoostClassifier: accuracy within 10%
   - Each at 200×10 and 1000×50

4. **File**: `ferroml-core/tests/vs_linfa_svm.rs` (NEW)
   - SVC (linear kernel): accuracy within 5%
   - SVC (RBF kernel): accuracy within 10%
   - SVR (linear kernel): R² within 0.10
   - SVR (RBF kernel): R² within 0.15

5. **File**: `ferroml-core/tests/vs_linfa_neighbors.rs` (NEW)
   - KNN classifier k=3,5,7: exact prediction match (same algorithm)
   - KNN regressor: predictions atol=1e-6

6. **File**: `ferroml-core/tests/vs_linfa_naive_bayes.rs` (NEW)
   - GaussianNB: predictions exact match, log-proba atol=1e-6
   - MultinomialNB: predictions exact match
   - BernoulliNB: predictions exact match

7. **File**: `ferroml-core/tests/vs_linfa_preprocessing.rs` (NEW)
   - StandardScaler: transform atol=1e-10
   - MinMaxScaler: transform atol=1e-10
   - PCA: explained_variance_ratio atol=1e-6, components sign-invariant
   - CountVectorizer: vocabulary match, transform match
   - TF-IDF: transform atol=1e-8

8. **File**: `ferroml-core/tests/vs_linfa_clustering.rs` (NEW)
   - KMeans: ARI > 0.8 on well-separated data, inertia within 10%
   - DBSCAN: cluster count match, noise point count match
   - GMM: BIC within 10%, component count match

**Success Criteria**:
- [ ] `cargo test --test vs_linfa_linear` — all pass
- [ ] `cargo test --test vs_linfa_trees` — all pass
- [ ] `cargo test --test vs_linfa_svm` — all pass
- [ ] `cargo test --test vs_linfa_neighbors` — all pass
- [ ] `cargo test --test vs_linfa_naive_bayes` — all pass
- [ ] `cargo test --test vs_linfa_preprocessing` — all pass
- [ ] `cargo test --test vs_linfa_clustering` — all pass
- [ ] ~60 new Rust tests total

---

### Phase S.2: Rust — FerroML vs smartcore (14 algorithms)

**Overview**: Add smartcore as dev-dependency and write comparison tests for all overlapping algorithms.

**Changes Required**:

1. **File**: `ferroml-core/Cargo.toml`
   - Add: `smartcore = "0.4"` to dev-dependencies

2. **File**: `ferroml-core/tests/vs_smartcore_linear.rs` (NEW)
   - LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
   - Same tolerance strategy as linfa tests
   - 3 dataset sizes each

3. **File**: `ferroml-core/tests/vs_smartcore_trees.rs` (NEW)
   - DecisionTree (C/R), RandomForest (C/R), ExtraTrees (C/R)
   - Accuracy/R² comparison

4. **File**: `ferroml-core/tests/vs_smartcore_svm.rs` (NEW)
   - SVC, SVR with linear and RBF kernels

5. **File**: `ferroml-core/tests/vs_smartcore_neighbors.rs` (NEW)
   - KNN classifier/regressor k=3,5,7

6. **File**: `ferroml-core/tests/vs_smartcore_naive_bayes.rs` (NEW)
   - GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB

7. **File**: `ferroml-core/tests/vs_smartcore_unsupervised.rs` (NEW)
   - PCA, StandardScaler, KMeans, DBSCAN

**Success Criteria**:
- [ ] `cargo test --test vs_smartcore_*` — all pass
- [ ] ~50 new Rust tests total

---

### Phase S.3: Python — FerroML vs XGBoost + LightGBM

**Overview**: Install XGBoost and LightGBM, compare gradient/histogram boosting models.

**Changes Required**:

1. **Install**: `pip install xgboost lightgbm`

2. **File**: `ferroml-python/tests/test_vs_xgboost.py` (NEW)
   - GradientBoostingRegressor vs XGBRegressor
     - R² within 0.05, correlation > 0.95
     - Dataset sizes: 1K, 10K, 100K
     - Feature importance rank correlation > 0.7
   - GradientBoostingClassifier vs XGBClassifier
     - Accuracy within 5%
     - Probabilities: KL divergence < 0.1
   - HistGradientBoostingRegressor vs XGBRegressor (hist)
     - Same tolerances
   - AdaBoostClassifier vs XGBClassifier (n_estimators=1, depth=1 stumps)
   - Performance timing: fit + predict at 1K/10K/100K

3. **File**: `ferroml-python/tests/test_vs_lightgbm.py` (NEW)
   - HistGradientBoostingRegressor vs LGBMRegressor
     - R² within 0.05
     - Dataset sizes: 1K, 10K, 100K
   - HistGradientBoostingClassifier vs LGBMClassifier
     - Accuracy within 5%
   - GradientBoostingRegressor vs LGBMRegressor (exact mode)
   - Performance timing with JSON output
   - Feature importance comparison

**Success Criteria**:
- [ ] `pytest tests/test_vs_xgboost.py -v` — all pass
- [ ] `pytest tests/test_vs_lightgbm.py -v` — all pass
- [ ] ~30 new Python tests total

---

### Phase S.4: Python — FerroML vs statsmodels + scipy

**Overview**: Compare linear/GLM models against statsmodels, and statistical methods against scipy.

**Changes Required**:

1. **Install**: `pip install statsmodels` (scipy already installed)

2. **File**: `ferroml-python/tests/test_vs_statsmodels.py` (NEW)
   - LinearRegression: coefficients atol=1e-6, R² atol=1e-8
   - Ridge (vs OLS with L2): coefficients atol=1e-4
   - LogisticRegression: coefficients atol=1e-3, predictions match
   - RobustRegression (Huber): coefficients atol=1e-2
   - QuantileRegression: coefficients atol=0.1 (different solvers)
   - Diagnostic comparison: p-values, standard errors, confidence intervals
   - Statsmodels summary() stats vs FerroML StatisticalModel output

3. **File**: `ferroml-python/tests/test_vs_scipy.py` (NEW)
   - StandardScaler vs scipy.stats.zscore: atol=1e-10
   - PCA vs scipy.linalg.svd: singular values match
   - KMeans vs scipy.cluster.vq.kmeans2: inertia within 10%
   - GMM vs scipy.stats.multivariate_normal: log-likelihood within 5%
   - IsotonicRegression vs scipy.interpolate: predictions atol=1e-6
   - Distance metrics: verify KNN distances match scipy.spatial.distance

**Success Criteria**:
- [ ] `pytest tests/test_vs_statsmodels.py -v` — all pass
- [ ] `pytest tests/test_vs_scipy.py -v` — all pass
- [ ] ~25 new Python tests total

---

### Phase S.5: Expand sklearn Coverage to ALL FerroML Models

**Overview**: The current sklearn comparison covers ~40 models. FerroML has 55+. Fill the gaps.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_comparison_gaps.py` (NEW)
   - Models not yet compared against sklearn:
     - GaussianProcessRegressor vs sklearn.gaussian_process.GaussianProcessRegressor
     - GaussianProcessClassifier vs sklearn.gaussian_process.GaussianProcessClassifier
     - MultiOutputRegressor vs sklearn.multioutput.MultiOutputRegressor
     - MultiOutputClassifier vs sklearn.multioutput.MultiOutputClassifier
     - SGDClassifier vs sklearn.linear_model.SGDClassifier (already partial)
     - SGDRegressor vs sklearn.linear_model.SGDRegressor
     - PassiveAggressiveClassifier vs sklearn.linear_model.PassiveAggressiveClassifier
     - PassiveAggressiveRegressor vs sklearn.linear_model.PassiveAggressiveRegressor
     - BaggingClassifier vs sklearn.ensemble.BaggingClassifier
     - BaggingRegressor vs sklearn.ensemble.BaggingRegressor
     - VotingClassifier vs sklearn.ensemble.VotingClassifier
     - StackingClassifier vs sklearn.ensemble.StackingClassifier
     - CategoricalNB (already covered but verify completeness)
     - HDBSCAN vs sklearn.cluster.HDBSCAN
     - TfidfVectorizer vs sklearn.feature_extraction.text.TfidfVectorizer
     - TfidfTransformer vs sklearn.feature_extraction.text.TfidfTransformer
   - Each model: fit, predict, score comparison
   - GP kernels: RBF, Matern, verify hyperparameter optimization

**Success Criteria**:
- [ ] `pytest tests/test_comparison_gaps.py -v` — all pass
- [ ] ~40 new Python tests
- [ ] Every FerroML public model has at least one sklearn comparison test

---

### Phase S.6: Cross-Library Performance Benchmark Suite

**Overview**: Unified benchmark script that times FerroML vs ALL libraries on overlapping algorithms.

**Changes Required**:

1. **File**: `scripts/benchmark_cross_library.py` (NEW)
   - Architecture:
     ```python
     @dataclass
     class BenchmarkResult:
         algorithm: str
         library: str
         task: str  # classification/regression/clustering/preprocessing
         n_samples: int
         n_features: int
         fit_time_ms: float
         predict_time_ms: float
         score: float  # accuracy, R², ARI, etc.
     ```
   - Libraries: ferroml, sklearn, xgboost, lightgbm, statsmodels
   - Algorithms (all overlapping):
     - Linear: LR, Ridge, Lasso, ElasticNet, LogReg (ferro, sklearn, statsmodels)
     - Trees: DT, RF, GB, HGB, ET, AdaBoost (ferro, sklearn, xgboost, lightgbm)
     - SVM: SVC, SVR (ferro, sklearn)
     - KNN: classifier, regressor (ferro, sklearn)
     - NB: Gaussian, Multinomial, Bernoulli (ferro, sklearn)
     - Preprocessing: StandardScaler, PCA, CountVectorizer, TF-IDF (ferro, sklearn)
     - Clustering: KMeans, DBSCAN, GMM (ferro, sklearn)
     - Anomaly: IsolationForest, LOF (ferro, sklearn)
     - GP: GPR, GPC (ferro, sklearn)
   - Dataset sizes: 1K, 10K, 100K samples
   - Features: 10, 50, 100
   - Runs: median of 5
   - Output: JSON, Markdown table, ASCII table

2. **File**: `scripts/benchmark_rust_libraries.rs` (NEW, standalone binary)
   - Criterion-based benchmark comparing FerroML vs linfa vs smartcore
   - Overlapping algorithms only
   - Same dataset sizes as Python benchmark
   - Output: JSON results + Criterion HTML reports

3. **File**: `docs/cross-library-benchmark.md` (NEW)
   - Auto-generated from benchmark JSON output
   - Tables: Algorithm × Library × Dataset Size → fit_time, predict_time, score
   - Speedup ratios vs sklearn baseline
   - Feature parity matrix

**Success Criteria**:
- [ ] `python scripts/benchmark_cross_library.py` completes without error
- [ ] JSON output saved to `docs/benchmark_cross_library_results.json`
- [ ] Markdown report generated at `docs/cross-library-benchmark.md`
- [ ] All algorithms produce valid scores (no NaN, no crashes)

---

### Phase S.7: Edge Case Gauntlet (Cross-Library)

**Overview**: Test edge cases that commonly cause ML libraries to diverge or crash.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_cross_library_edge_cases.py` (NEW)
   - **NaN handling** (all libraries):
     - FerroML rejects NaN (by design) — verify clean error
     - Compare: sklearn impute then fit vs ferroml manual impute then fit
   - **Single sample** (all models):
     - Fit on 1 sample, predict on 1 sample
     - Verify: error or valid prediction for each library
   - **High dimensional** (p >> n):
     - 50 samples, 500 features
     - Ridge, Lasso, ElasticNet, PCA, SVM
     - Compare: which libraries handle this, which don't
   - **Sparse data** (>90% zeros):
     - CountVectorizer output → classification
     - Compare sparse handling across libraries
   - **Extreme values** (1e±308):
     - Verify no overflow in predictions
   - **Constant target**:
     - All regressors: verify behavior (error vs constant prediction)
   - **All-same features**:
     - Verify: PCA, StandardScaler, tree splits handle gracefully
   - **Binary vs multiclass**:
     - Same model, different n_classes — verify consistent API
   - **Large class count** (100 classes):
     - LogReg, RF, SVM — verify scaling

**Success Criteria**:
- [ ] `pytest tests/test_cross_library_edge_cases.py -v` — all pass
- [ ] ~50 new edge case tests
- [ ] Each test documents expected behavior per library

---

### Phase S.8: Feature Parity Scorecard & Final Report

**Overview**: Generate a comprehensive feature parity matrix and final validation report.

**Changes Required**:

1. **File**: `scripts/generate_parity_scorecard.py` (NEW)
   - Programmatically check which methods each FerroML model exposes
   - Compare against sklearn's API for same model
   - Generate matrix: Model × Method → {present, missing, N/A}
   - Methods: fit, predict, predict_proba, decision_function, score, transform,
     inverse_transform, get_params, partial_fit, feature_importances_, coef_, intercept_

2. **File**: `docs/feature-parity-scorecard.md` (NEW, auto-generated)
   - Algorithm count comparison: FerroML vs linfa vs smartcore vs sklearn
   - Method coverage: what % of sklearn's API does FerroML implement per model
   - Performance summary: where FerroML wins, where it loses
   - Correctness summary: tolerance ranges achieved
   - Unique features: what only FerroML offers

3. **File**: `docs/cross-library-validation-report.md` (NEW, auto-generated)
   - Executive summary
   - Test counts by library: N tests passing
   - Correctness table: algorithm × library × tolerance achieved
   - Performance table: algorithm × library × speedup
   - Edge case results: what breaks where
   - Recommendations for improvement

**Success Criteria**:
- [ ] `python scripts/generate_parity_scorecard.py` runs cleanly
- [ ] Scorecard covers all 55+ FerroML models
- [ ] Final report includes all benchmark + test data
- [ ] Zero unexplained test failures

---

## Dependencies

- Phase S.1 requires: linfa crates compile (may need specific Rust toolchain features)
- Phase S.2 requires: smartcore compiles with ndarray compatibility
- Phase S.3 requires: `pip install xgboost lightgbm`
- Phase S.4 requires: `pip install statsmodels` (scipy already present)
- Phase S.5 requires: nothing new
- Phase S.6 requires: S.1-S.5 complete (uses all libraries)
- Phase S.7 requires: S.1-S.5 complete
- Phase S.8 requires: S.1-S.7 complete

**Parallelization**: S.1 and S.2 can run in parallel. S.3, S.4, S.5 can run in parallel. S.6 and S.7 can run in parallel after S.1-S.5.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| linfa API incompatible with ndarray version | S.1 blocked | Pin ndarray version or use compatibility shim |
| smartcore uses different matrix type | S.2 blocked | Convert via `.as_slice()` + reshape |
| XGBoost/LightGBM install fails (no C compiler) | S.3 blocked | Use pre-built wheels (`pip install --only-binary`) |
| Tolerance disagreements on stochastic algorithms | False failures | Use distribution-based tests (ARI, correlation) not exact match |
| 100K samples too slow for some algorithms | S.6 timeout | Cap slow algorithms at 10K, note in report |
| linfa subcrate versions mismatch | Compile errors | Pin all linfa-* to same minor version |

## Estimated Test Counts

| Phase | New Tests | Cumulative |
|-------|-----------|------------|
| S.1: vs linfa | ~60 | 60 |
| S.2: vs smartcore | ~50 | 110 |
| S.3: vs XGBoost/LightGBM | ~30 | 140 |
| S.4: vs statsmodels/scipy | ~25 | 165 |
| S.5: sklearn gaps | ~40 | 205 |
| S.6: benchmarks | ~0 (scripts) | 205 |
| S.7: edge cases | ~50 | 255 |
| S.8: scorecard | ~0 (scripts) | 255 |
| **Total** | **~255 new tests** | |

Combined with existing 4,780 tests → **~5,035 total tests**.
