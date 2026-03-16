# Plan X — Production-Ready: Correctness, Accuracy, Verification

## Mission

Make every algorithm in FerroML **provably correct, numerically stable, and battle-tested** against reference implementations. This is not about adding algorithms — it's about making what we have trustworthy enough that someone can bet their production workload on it.

## Current State (2026-03-15, updated after Plan W completion)

- 95+ algorithms, 5,193+ tests passing (3,193 Rust + ~2,000 Python)
- **Plan W COMPLETE** — W.1-W.7 done, W.8 notebooks done (4/4), W.9 benchmarks run
- **5 models** at Tier 1 confidence (3+ validation sources)
- **16 models** at Tier 2 (2 sources)
- **35 models** at Tier 3 (1 source, Python-only)
- **16 models** with ZERO external validation (ensemble, sampling, HDBSCAN)
- 6 medium-severity numerical stability issues
- Sparse end-to-end testing missing
- naive_bayes.rs (4,680 lines) needs splitting

### Post-W Benchmark Results (n=5000, f=20)
| Algorithm | vs sklearn | Status |
|-----------|-----------|--------|
| KMeans | **3.4x FASTER** | Elkan's algorithm — huge win |
| RF fit | **5x FASTER** | Already was fast |
| GaussianNB | **4.3x FASTER** | Already was fast |
| StandardScaler | **9x FASTER** | Already was fast |
| HistGBT | 2.6x slower (was 15x) | Binary search bins — major improvement |
| HistGBT Regressor | 1.7x slower | Competitive |
| LogReg | 2.1x slower (was 2.5x) | IRLS used at d=20 (correct), L-BFGS for d>=50 |
| PCA | 2.1x slower | Benchmark didn't trigger tall-thin path |
| SVC | **REGRESSED to 17.6x** | LRU cache hurts at n=5000; threshold raised to 10K as band-aid |

### SVC Issue (needs proper fix, not just threshold)
The LRU kernel cache (W.3) made SVC slower, not faster. Raising FULL_MATRIX_THRESHOLD to 10K is a band-aid.
The real fix requires matching libsvm's working set selection strategy, not just adding LRU on top of random access.
Options: (a) revert to full matrix always, (b) implement libsvm-style shrinking, (c) accept 4x slower and document it.
**Decision**: For now, full matrix up to 10K samples. Revisit SVC cache for v0.5.0.

## Success Criteria

- Every model verified against sklearn or equivalent reference to within documented tolerances
- Every model handles all 13 edge cases (n=1, p=1, p>n, zero variance, NaN, Inf, empty, etc.)
- All 6 numerical stability issues fixed
- Sparse pipeline E2E tests passing
- 0 `unwrap()` in non-test code, 0 `todo!()`, 0 silent failures
- Cross-library validation expanded from 164 to 300+ tests

---

## Phase X.1: Numerical Stability Fixes (6 issues, ~1 hour)

**Goal**: Fix all 6 medium-severity numerical issues found in audit.

### Changes:

1. **Linear regression diagnostics leverage guard** (`models/linear.rs` lines 252, 267, 287, 301)
   - Clamp `(1.0 - h_i).max(1e-10)` before division in Cook's distance, standardized/studentized residuals
   - Add test: `test_cooks_distance_high_leverage_point()`

2. **GMM diagonal covariance log guard** (`clustering/gmm.rs` lines 497, 506)
   - Change to `diag_cov[[c, j]].max(self.reg_covar).ln()`
   - Add test: `test_gmm_near_zero_variance_component()`

3. **GMM spherical covariance log guard** (`clustering/gmm.rs` lines 523, 532)
   - Change to `sph_cov[c].max(self.reg_covar).ln()`
   - Add test: `test_gmm_spherical_degenerate()`

4. **CategoricalNB log guard** (`models/naive_bayes.rs` line 2057)
   - Change to `prob.max(1e-300).ln()`
   - Add test: `test_categorical_nb_zero_count_no_nan()`

5. **Cholesky log-determinant assertion** (`linalg.rs` line 282)
   - Add `debug_assert!(l[[i, i]] > 0.0, "Cholesky diagonal must be positive")`

6. **t-SNE exponential overflow protection** (`decomposition/tsne.rs` lines 348, 661)
   - Clamp exponent: `(-distances_i[j] * beta).max(-700.0).exp()`
   - Add test: `test_tsne_large_distance_no_overflow()`

### Verification:
- `cargo test -p ferroml-core --lib -- numerical_stability` — all new tests pass
- `cargo clippy -p ferroml-core -- -D warnings` — zero warnings

---

## Phase X.2: Critical Validation Gaps — Ensemble & Meta-Learners (~4 hours)

**Goal**: Add cross-library validation for the 16 models with ZERO external verification.

### X.2a: Ensemble Methods (Voting, Stacking, Bagging)

Create `ferroml-python/tests/test_vs_sklearn_ensemble.py`:

| Model | Test | Validation |
|-------|------|------------|
| VotingClassifier | Fit 3 classifiers, compare predictions vs sklearn VotingClassifier | Exact label match |
| VotingRegressor | Fit 3 regressors, compare predictions vs sklearn VotingRegressor | R² within 0.05 |
| StackingClassifier | 2 base + LR meta, compare vs sklearn StackingClassifier | Accuracy within 5% |
| StackingRegressor | 2 base + Ridge meta, compare vs sklearn StackingRegressor | R² within 0.1 |
| BaggingClassifier | RF-equivalent config, compare vs sklearn BaggingClassifier | Accuracy within 5% |
| BaggingRegressor | RF-equivalent config, compare vs sklearn BaggingRegressor | R² within 0.1 |

### X.2b: MultiOutput Wrappers

Create `ferroml-python/tests/test_vs_sklearn_multioutput.py`:

| Model | Test | Validation |
|-------|------|------------|
| MultiOutputRegressor | LinearRegression base, 3 targets | Per-target R² within 0.01 of sklearn |
| MultiOutputClassifier | LogisticRegression base, 3 targets | Per-target accuracy within 3% |

### X.2c: Sampling Methods

Create `ferroml-python/tests/test_vs_imbalanced_learn.py`:

| Model | Test | Validation |
|-------|------|------------|
| SMOTE | Generate synthetic minority, compare class distribution | Same output shape, balanced ratio within 5% |
| ADASYN | Same | Same output shape, similar distribution |
| RandomOverSampler | Verify exact duplication of minority samples | Exact match on class counts |
| RandomUnderSampler | Verify correct removal of majority samples | Exact match on class counts |

### X.2d: HDBSCAN

Create `ferroml-python/tests/test_vs_hdbscan.py`:

| Model | Test | Validation |
|-------|------|------------|
| HDBSCAN | Compare cluster labels on blobs dataset | Adjusted Rand Index > 0.8 vs hdbscan library |
| HDBSCAN | Noise detection on outlier-injected data | Similar noise point count (within 20%) |

### X.2e: Calibration & IncrementalPCA

Add to `ferroml-python/tests/test_vs_sklearn_gaps.py`:

| Model | Test | Validation |
|-------|------|------------|
| CalibratedClassifierCV | Sigmoid calibration on SVC | Brier score within 0.02 of sklearn |
| IncrementalPCA | Batch vs full PCA equivalence | Explained variance within 1e-3 |

### Verification:
- All new Python tests pass: `pytest tests/test_vs_sklearn_ensemble.py tests/test_vs_sklearn_multioutput.py tests/test_vs_imbalanced_learn.py tests/test_vs_hdbscan.py -v`
- Cross-library test count increases from 164 to ~200+

---

## Phase X.3: Tier 3 Models — Upgrade to Cross-Library Validation (~6 hours)

**Goal**: Upgrade 35 models from "Python-only" or "internal-only" to proper cross-library validation.

### X.3a: SGD Variants (Rust cross-library)

Create `ferroml-core/tests/vs_linfa_sgd.rs`:
- SGDClassifier vs linfa-elasticnet (shared objective)
- SGDRegressor vs linfa-elasticnet
- Perceptron vs linfa (if available)
- Verify convergence behavior on linearly separable data

### X.3b: Linear Models Missing Rust Validation

Add to existing `vs_linfa_linear.rs`:
- RidgeCV: verify best alpha selection matches linfa
- LassoCV: verify best alpha selection
- ElasticNetCV: verify best alpha/l1_ratio selection

### X.3c: SVM Missing Validation

Add to existing `vs_linfa_svm.rs`:
- LinearSVC vs linfa-svm (linear kernel)
- LinearSVR vs linfa-svm

### X.3d: Tree Models Missing Validation

Create `ferroml-python/tests/test_vs_sklearn_trees.py`:
- ExtraTreesClassifier vs sklearn: accuracy within 5%
- ExtraTreesRegressor vs sklearn: R² within 0.05
- GradientBoostingClassifier vs sklearn: accuracy within 5%
- GradientBoostingRegressor vs sklearn: R² within 0.05
- HistGradientBoostingClassifier vs sklearn: accuracy within 5%
- HistGradientBoostingRegressor vs sklearn: R² within 0.05

### X.3e: Clustering Missing Validation

Create `ferroml-python/tests/test_vs_sklearn_clustering.py`:
- AgglomerativeClustering vs sklearn: ARI > 0.9 on well-separated blobs
- DBSCAN vs sklearn: exact label match on standard epsilon

### X.3f: Decomposition Missing Validation

Create `ferroml-python/tests/test_vs_sklearn_decomposition.py`:
- TruncatedSVD vs sklearn: explained variance within 1e-3
- FactorAnalysis vs sklearn: loadings correlation > 0.95
- IncrementalPCA vs sklearn: components within 1e-3
- LDA (decomposition) vs sklearn: if applicable

### X.3g: Preprocessing Missing Validation

Add to existing Python tests:
- PowerTransformer vs sklearn: transform output within 1e-4
- QuantileTransformer vs sklearn: transform output within 1e-3
- KNNImputer vs sklearn: imputed values within 1e-3
- KBinsDiscretizer vs sklearn: bin assignments exact match
- TargetEncoder vs sklearn: encoded values within 1e-2
- CountVectorizer vs sklearn: exact vocabulary + counts match
- TfidfVectorizer vs sklearn: TF-IDF values within 1e-5

### X.3h: Special Models

Create `ferroml-python/tests/test_vs_sklearn_special.py`:
- QDA vs sklearn: accuracy within 3%
- IsotonicRegression vs sklearn: predictions within 1e-6
- QuantileRegression vs sklearn/statsmodels: quantile predictions within 0.05
- RobustRegression vs statsmodels: coefficients within 0.1
- GaussianProcessRegressor vs sklearn: mean predictions within 1e-3
- GaussianProcessClassifier vs sklearn: probabilities within 0.05
- IsolationForest vs sklearn: anomaly scores correlation > 0.9
- LOF vs sklearn: anomaly scores correlation > 0.9

### Verification:
- All new tests pass
- Cross-library test count reaches 300+

---

## Phase X.4: Systematic Edge Case Testing (~4 hours)

**Goal**: Every model handles all 13 standard edge cases with documented behavior.

### Edge Case Matrix

Create `ferroml-core/tests/edge_case_matrix.rs` with systematic tests:

```
For EVERY model that implements Model trait:
  1. test_{model}_single_sample          — n=1, should not panic
  2. test_{model}_single_feature         — p=1, should work or error clearly
  3. test_{model}_high_dimensional       — p > n (p=100, n=10)
  4. test_{model}_zero_variance          — all features constant
  5. test_{model}_identical_targets      — y = [1,1,1,1,1]
  6. test_{model}_nan_input_rejected     — NaN in X → InvalidInput error
  7. test_{model}_inf_input_rejected     — Inf in X → InvalidInput error
  8. test_{model}_nan_target_rejected    — NaN in y → InvalidInput error
  9. test_{model}_empty_input_rejected   — n=0 → error
  10. test_{model}_extreme_large_values  — x = 1e15, should not overflow
  11. test_{model}_extreme_small_values  — x = 1e-15, should not underflow
  12. test_{model}_class_imbalance       — 99:1 ratio (classifiers only)
  13. test_{model}_multicollinear        — duplicate features (linear models)
```

### Models to cover (minimum):

**Already covered (verify)**: LinearRegression, LogisticRegression, Ridge, DecisionTree, RandomForest, GaussianNB, KNN, SVC, StandardScaler, PCA

**Must add**: Lasso, ElasticNet, SGDClassifier, SGDRegressor, GradientBoosting, HistGradientBoosting, ExtraTrees, AdaBoost, MultinomialNB, BernoulliNB, CategoricalNB, SVR, LinearSVC, LinearSVR, KMeans, DBSCAN, GMM, HDBSCAN, Agglomerative, TruncatedSVD, t-SNE, MinMaxScaler, RobustScaler, OneHotEncoder, PolynomialFeatures, IsolationForest, LOF, QDA, Perceptron, PassiveAggressive

### Special edge cases by category:

**Naive Bayes**: test_multinomial_nb_negative_counts_rejected (should error)
**KMeans**: test_kmeans_k_equals_n (k=n should work), test_kmeans_all_identical_points
**DBSCAN**: test_dbscan_no_core_points (eps too small)
**PCA**: test_pca_n_components_exceeds_rank
**StandardScaler**: test_scaler_all_nan_column (with allow_nan)

### Verification:
- `cargo test -p ferroml-core -- edge_case` — all pass
- Every model has at least 6 of the 13 edge case tests

---

## Phase X.5: Sparse End-to-End Pipeline Testing (~2 hours)

**Goal**: Verify sparse data flows correctly through the entire pipeline.

### Rust Tests

Create `ferroml-core/tests/sparse_pipeline_e2e.rs`:

1. **Text classification pipeline**:
   - CountVectorizer → TfidfTransformer → MultinomialNB
   - Verify predictions match dense equivalent within 1e-6

2. **Sparse linear pipeline**:
   - Sparse CSR input → LogisticRegression (fit_sparse/predict_sparse)
   - Verify predictions match dense equivalent

3. **Sparse KNN pipeline**:
   - Sparse CSR input → KNeighborsClassifier
   - Verify labels match dense equivalent

4. **Sparse round-trip**:
   - Dense → to_sparse → model.fit_sparse → predict_sparse vs Dense → model.fit → predict
   - For all 11 sparse-enabled models

5. **Mixed pipeline**:
   - Sparse input → dense conversion → StandardScaler → LogisticRegression
   - Verify full pipeline produces valid predictions

### Python Tests

Expand `ferroml-python/tests/test_sparse_pipeline.py`:

1. CountVectorizer → TfidfTransformer → each sparse model (MultinomialNB, BernoulliNB, LinearSVC, LogisticRegression)
2. scipy.sparse.random() → fit_sparse → predict_sparse for all 11 models
3. Verify no silent densification (memory check)

### Verification:
- `cargo test -p ferroml-core -- sparse_pipeline` — all pass
- `pytest tests/test_sparse_pipeline.py -v` — all pass

---

## Phase X.6: Code Quality — Split naive_bayes.rs (~2 hours)

**Goal**: Split 4,680-line naive_bayes.rs into 5 files for maintainability.

### New structure:
```
ferroml-core/src/models/naive_bayes/
├── mod.rs              — re-exports, shared helpers (compute_variance, z_critical)
├── gaussian.rs         — GaussianNB (~560 lines)
├── multinomial.rs      — MultinomialNB (~470 lines)
├── bernoulli.rs        — BernoulliNB (~470 lines)
└── categorical.rs      — CategoricalNB (~580 lines)
```

### Steps:
1. Create `naive_bayes/` directory
2. Move each variant's struct + impl blocks to its own file
3. Extract shared helpers to `mod.rs`
4. Move SparseModel/PipelineSparseModel impls alongside their variants
5. Move inline tests with their variants
6. Update `models/mod.rs` to use `mod naive_bayes` instead of single file
7. Verify all re-exports work: `use crate::models::{GaussianNB, MultinomialNB, ...}`

### Verification:
- `cargo test -p ferroml-core --lib -- naive_bayes` — all 66 tests pass
- `cargo test -p ferroml-core` — full suite passes (no broken imports)
- Python tests pass (bindings unchanged)

---

## Phase X.7: Tolerance Audit & Calibration (~1 hour)

**Goal**: Ensure every cross-library test uses algorithm-appropriate tolerances.

### Review:
1. Audit all `assert_approx_eq!` calls in `tests/correctness_*.rs` and `tests/vs_linfa_*.rs`
2. Verify tolerance constants match algorithm type:
   - Closed-form solvers (LinearRegression, Ridge): `CLOSED_FORM` (1e-10)
   - Iterative solvers (Lasso, ElasticNet, LogReg): `ITERATIVE` (1e-4)
   - Tree models: `TREE` (1e-12) — verify this is appropriate or relax
   - Stochastic models (SGD, TSNE): `PROBABILISTIC` (1e-2)
   - Cross-library: `SKLEARN_COMPAT` (1e-5) — verify per algorithm
3. Document tolerance rationale in `assertions.rs` comments
4. Fix any tests using wrong tolerance tier

### Verification:
- All tests still pass with calibrated tolerances
- No tests that pass only because tolerance is too loose

---

## Phase X.8: Python Bindings Correctness Audit (~3 hours)

**Goal**: Verify Python bindings faithfully expose Rust behavior — no silent data corruption.

### Tests to add in `ferroml-python/tests/test_bindings_correctness.py`:

1. **Array conversion fidelity**:
   - Verify float64 arrays round-trip without precision loss
   - Verify int arrays convert correctly
   - Verify C-contiguous vs F-contiguous arrays handled
   - Verify empty arrays produce correct errors

2. **Error propagation**:
   - Every FerroError variant maps to a Python exception
   - Error messages preserve diagnostic information
   - Stack traces are useful

3. **State management**:
   - Fitted model persists state between calls
   - Unfitted model raises on predict
   - Re-fitting replaces old state completely

4. **Serialization round-trip**:
   - pickle.dumps → pickle.loads preserves model state
   - Predictions match pre/post serialization

5. **Thread safety**:
   - Concurrent predict calls don't corrupt state
   - Concurrent fit calls on different models work

### Verification:
- `pytest tests/test_bindings_correctness.py -v` — all pass

---

## Phase X.9: Benchmark Verification (~2 hours)

**Goal**: Verify Plan W performance improvements with actual benchmarks.

### Run:
1. `maturin develop --release -m ferroml-python/Cargo.toml`
2. `python scripts/benchmark_cross_library.py`
3. Compare results against pre-W baseline:

| Algorithm | Pre-W | Post-W Target | Actual |
|-----------|-------|---------------|--------|
| HistGBT | 1852ms | <600ms | ? |
| SVC | 406ms | <200ms | ? |
| KMeans | 33.8ms | <20ms | ? |
| LogReg | 13.6ms | <8ms | ? |
| PCA (tall-thin) | ~3x sklearn | <1.5x sklearn | ? |

4. If any target missed, investigate and fix
5. Update benchmark results in docs

### Verification:
- All algorithms within target range or documented justification for gap

---

## Phase X.10: Tutorial Notebooks (~4 hours)

**Goal**: Complete the 4 tutorial notebooks that demonstrate statistical rigor.

### Notebooks:

1. **`01_model_comparison.ipynb`** — "Are These Models Actually Different?"
   - Train LinearRegression, Ridge, Lasso, ElasticNet on California housing
   - Use `score()` + corrected resampled t-test
   - Show confidence intervals on CV scores
   - Demonstrate: "sklearn says 0.85 vs 0.84 — is that real?"

2. **`02_prediction_uncertainty.ipynb`** — Already exists, verify runs

3. **`03_assumption_checking.ipynb`** — Already exists, verify runs

4. **`04_fair_model_selection.ipynb`** — "AutoML That Tells You Why"
   - Use AutoML with statistical testing
   - Show multiple testing correction
   - Demonstrate uncertainty in "best model" selection

### Verification:
- Each notebook runs end-to-end: `jupyter nbconvert --execute --to notebook`
- No cells produce errors
- Output tells a compelling story

---

## Phase X.11: Documentation & Release Prep (~2 hours)

**Goal**: Update all documentation to reflect production-ready state.

### Tasks:
1. Update README with current test counts and verification levels
2. Update ROADMAP with completed Plan W and X status
3. Regenerate feature parity scorecard
4. Write CHANGELOG entries for v0.4.0
5. Fix GitHub billing (user action) → verify CI/CD works
6. Tag v0.4.0 with "production-ready" milestone

---

## Phase Ordering & Dependencies

```
DONE (from Plan W):
  W.8 (Notebooks)            ── 4/4 created and verified
  W.9 (Benchmarks)           ── run, results captured above

LAYER 1 (independent, parallelize):
  X.1 (Numerical fixes)        ── 1 hour
  X.6 (Split naive_bayes.rs)   ── 2 hours
  X.7 (Tolerance audit)        ── 1 hour

LAYER 2 (the bulk — independent, parallelize):
  X.2 (Critical validation)    ── 4 hours
  X.3 (Tier 3 upgrades)        ── 6 hours
  X.4 (Edge case matrix)       ── 4 hours
  X.5 (Sparse E2E)             ── 2 hours

LAYER 3 (needs layers 1-2 done):
  X.8 (Bindings audit)         ── 3 hours

LAYER 4 (capstone):
  X.9 (Benchmarks)             ── DONE (Plan W)
  X.10 (Notebooks)             ── DONE (Plan W)
  X.11 (Release)               ── 2 hours (needs GitHub billing fix)
```

## Effort Summary

| Phase | Effort | Impact | Tests Added | Status |
|-------|--------|--------|-------------|--------|
| X.1 Numerical fixes | 1 hour | Critical safety | ~8 | TODO |
| X.2 Critical validation | 4 hours | Fills biggest gaps | ~40 | TODO |
| X.3 Tier 3 upgrades | 6 hours | Broadest coverage | ~80 | TODO |
| X.4 Edge case matrix | 4 hours | Systematic safety | ~200+ | TODO |
| X.5 Sparse E2E | 2 hours | Pipeline confidence | ~20 | TODO |
| X.6 Split naive_bayes | 2 hours | Maintainability | 0 (move existing) | TODO |
| X.7 Tolerance audit | 1 hour | Test accuracy | 0 (fix existing) | TODO |
| X.8 Bindings audit | 3 hours | Python trust | ~30 | TODO |
| X.9 Benchmarks | — | Performance proof | 0 | **DONE** (Plan W) |
| X.10 Notebooks | — | Marketing | 0 | **DONE** (Plan W) |
| X.11 Release | 2 hours | Ship it | 0 | BLOCKED (GitHub billing) |

**Remaining: ~23 hours across 8 phases**
**Tests to add: ~378 new tests** (5,193 → ~5,571)
**Cross-library tests: 164 → 300+**

## What This Achieves

After Plan X:
- **Every model** verified against sklearn or equivalent
- **Every model** handles degenerate inputs without panicking
- **Every numerical operation** has appropriate guards
- **Sparse pipeline** tested end-to-end
- **Code quality** improved (naive_bayes split, tolerances calibrated)
- **Performance** verified with benchmarks (**DONE**)
- **Tutorials** demonstrate the value proposition (**DONE**)
- **Ready to ship v0.4.0** as production-ready

## What This Does NOT Include

- New algorithms (ComplementNB, SpectralClustering, etc.)
- Semi-supervised learning module
- DataFrame-native API
- Deep learning / GPU training
- v1.0.0 release
- Matching sklearn's full algorithm catalog
- SVC kernel cache optimization (deferred to v0.5.0)

The goal is **depth over breadth**: make what we have bulletproof.

---

## Beyond Plan X — What Comes Next

### Plan Y — Algorithm Expansion (for after X is complete)

Once correctness is proven, expand the algorithm catalog for credibility:

**Quick wins (1-2 days each)**:
- ComplementNB (trivial, extends existing NB)
- MiniBatchKMeans (modify KMeans with mini-batch updates)
- BayesianRidge (extend Ridge with ARD prior)
- NuSVC/NuSVR (extend SVC/SVR with nu-parameterization)
- PassiveAggressiveRegressor (extend existing PA classifier)
- LabelBinarizer, MultiLabelBinarizer (preprocessing utilities)
- TransformedTargetRegressor (compose wrapper)
- ~10 missing metrics (balanced_accuracy, cohen_kappa, matthews_corrcoef, etc.)
- ~5 missing dataset generators (make_moons, make_circles, make_swiss_roll, etc.)

**Core credibility (3-7 days each)**:
- NMF (Non-negative Matrix Factorization)
- FastICA (Independent Component Analysis)
- KernelPCA
- SpectralClustering
- MeanShift
- OPTICS
- Isomap, MDS (manifold learning)
- GLMs: PoissonRegressor, GammaRegressor, TweedieRegressor
- OneVsRestClassifier, OneVsOneClassifier (meta-estimators)

**Adoption accelerators (1-2 weeks each)**:
- Semi-supervised module (LabelPropagation, LabelSpreading, SelfTraining)
- Lars/LassoLars (regularization path algorithms)
- TheilSenRegressor, RANSACRegressor (robust regression)
- BayesianGaussianMixture
- BisectingKMeans, Birch, AffinityPropagation

### Plan Z — Community & Ecosystem

- Fix GitHub billing → CI/CD → auto-publish to PyPI/crates.io
- Write announcement post (r/rust, r/machinelearning, Hacker News)
- Create migration guide: "Coming from sklearn"
- Add `ferroml` to Are We Learning Yet? (arewelearningyet.com)
- Set up Discord/Matrix for community
- Respond to issues and PRs from early adopters
- v1.0.0 when community feedback is incorporated
