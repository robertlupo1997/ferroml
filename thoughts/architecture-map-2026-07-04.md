# FerroML Architecture Map — 2026-07-04

> Dense, factual, file-referenced. No marketing. The task: what is actually here?

---

## 1. Workspace Layout and Crate Responsibilities

**Workspace** (`Cargo.toml`): resolver = "2", members = `["ferroml-core", "ferroml-python"]`.
`default-members = ["ferroml-core"]` (python bindings excluded from bare `cargo build/test`).
`ferroml-plugin/` is a Claude Code plugin directory (agents/, commands/, hooks/, skills/) used
during "ralph-loop" agentic development — it is not a Rust crate.

### ferroml-core (`ferroml-core/src/` — ~207,504 LOC)

The entire ML engine. Modules:

| Module | Content |
|---|---|
| `lib.rs` | Core trait definitions (`Predictor`, `Estimator`, `Transformer`), `AutoML` struct and `AutoMLConfig`, top-level types (`Task`, `Metric`, `MultipleTesting`) |
| `automl/` | CASH loop: portfolio, time-budget bandit, preprocessing selection, ensemble, meta-features, warm-start, transfer |
| `models/` | All supervised estimators (linear, tree, boosting, SVM, kNN, NB, GP, calibration, anomaly, QDA, isotonic, etc.) |
| `clustering/` | KMeans, DBSCAN, HDBSCAN, GMM, Agglomerative + metrics, diagnostics |
| `decomposition/` | PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis, TSNE, VPTree, QuadTree |
| `ensemble/` | BaggingClassifier/Regressor, VotingClassifier/Regressor, StackingClassifier/Regressor |
| `preprocessing/` | 20+ transformers (scalers, encoders, imputers, samplers, polynomial features, feature selectors) |
| `pipeline/` | Pipeline, FeatureUnion, ColumnTransformer with HPO search-space composition |
| `cv/` | 10+ cross-validation strategies, `cross_val_score`, nested CV, learning/validation curves |
| `hpo/` | Search spaces, samplers (Random, Grid, Bayesian/TPE), schedulers (Median, Hyperband, ASHA) |
| `stats/` | Hypothesis testing, CIs, effect sizes, bootstrap, power analysis, multiple-testing correction |
| `metrics/` | Classification, regression, probabilistic + model comparison tests |
| `explainability/` | TreeSHAP, permutation importance, partial dependence |
| `neural/` | Neural network models |
| `gpu/` | wgpu compute backend (optional feature) |
| `simd.rs` | 30+ SIMD-accelerated vector ops via `wide` crate (optional feature) |
| `sparse.rs` | CSR/CSC matrix types, sparse distance functions |
| `schema.rs` | Feature schema validation with `FeatureSchema`/`ValidationResult` |
| `serialization.rs` | JSON/MessagePack/Bincode save/load with metadata and streaming |
| `onnx/` | ONNX export |
| `linalg.rs` | QR, Cholesky, triangular solve, basic distances — pure Rust, no BLAS |
| `testing/` | 35+ testing utility modules (estimator checks, properties, mutation, fairness, drift, etc.) |
| `datasets/` | `Dataset` struct, toy datasets (iris/wine/diabetes), CSV/Parquet loaders, mmap'd arrays, synthetic generators |
| `recommend.rs` | Algorithm recommendation |
| `model_card.rs` | Model card generation |
| `validation.rs`, `array_utils.rs` | Input validation utilities |

### ferroml-python (`ferroml-python/src/` — ~35,956 LOC)

PyO3 bindings exposing all core modules as a `ferroml` Python package with submodules
(`ferroml.linear`, `ferroml.trees`, `ferroml.clustering`, `ferroml.automl`, etc.). Also
includes Pandas/Polars interop layers (feature-gated), sparse utils, and pickle support.

### ferroml-plugin

Not a Rust crate. Contains Claude Code plugin scaffolding (agents, commands, hooks, skills)
used for the ralph-loop agentic development workflow. No ML code.

---

## 2. Key Abstractions

### Core Traits (`ferroml-core/src/lib.rs`)

```rust
pub trait Estimator: Send + Sync {
    fn fit(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted>;
    fn search_space(&self) -> SearchSpace;
}

pub trait Predictor: Send + Sync {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
    fn predict_with_uncertainty(&self, x: &Array2<f64>, confidence: f64) -> Result<PredictionWithUncertainty>;
}

pub trait Transformer: Send + Sync {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
}
```

The `Model` trait (`ferroml-core/src/models/traits.rs`) is the runtime-polymorphism layer
(object-safe, used in `Box<dyn Model>` inside the AutoML fit loop). `Estimator` is the
generic/static-dispatch version used in CV harnesses.

### Data Structures

- **Dense**: `ndarray::Array2<f64>` / `Array1<f64>` everywhere. No f32 in the Rust API.
- **Sparse**: Custom `CsrMatrix`/`CscMatrix`/`SparseVector` in `sparse.rs`, wrapping `sprs`
  (optional feature). DBSCAN has `fit_sparse()`. SVM, text pipeline, NB, TruncatedSVD have
  sparse paths.
- **Dataset**: `ferroml-core/src/datasets/mod.rs` — convenience wrapper with feature/target
  names, train/test split, class counts.
- **MemmappedDataset**: `datasets/mmap.rs` — memory-mapped arrays for large datasets.

### Algorithm Plugin Interface

Algorithms plug in by implementing `Model` (or `Estimator` + `Transformer`). There is no
plugin registry — new algorithms are added by editing `models/` source files and adding to
the `AlgorithmType` enum in `automl/portfolio.rs`. `create_model()` in `automl/fit.rs:1146`
has a `match` arm per algorithm type that calls `new()` with default parameters.

### Error Handling

Uniform `FerroError` enum (`error.rs`) with named constructors:
`invalid_input`, `shape_mismatch`, `assumption_violation`, `convergence_failure`,
`not_fitted`, `not_implemented_for`, `numerical`, `cross_validation`. `Result<T>` alias
throughout. Python layer maps `FerroError` → `PyErr` in `ferroml-python/src/errors.rs`.

---

## 3. Algorithm Inventory

### AutoML Portfolio (`automl/portfolio.rs`, `AlgorithmType` enum — 32 variants)

**Classification (11):**
LogisticRegression, GaussianNB, MultinomialNB, CategoricalNB, KNeighborsClassifier,
SVC, LinearSVC, DecisionTreeClassifier, RandomForestClassifier,
GradientBoostingClassifier, HistGradientBoostingClassifier.

**Regression (11):**
LinearRegression, Ridge, Lasso, ElasticNet, QuantileRegression, RobustRegression,
KNeighborsRegressor, SVR, LinearSVR, DecisionTreeRegressor, RandomForestRegressor,
GradientBoostingRegressor, HistGradientBoostingRegressor.
(Actually 13 with all regression variants; enum has ~32 total across tasks.)

### Standalone Algorithms (not in AutoML portfolio)

**Clustering (`clustering/`):** KMeans, DBSCAN, HDBSCAN, GaussianMixture (GMM),
AgglomerativeClustering.

**Decomposition (`decomposition/`):** PCA, IncrementalPCA, TruncatedSVD, LDA,
FactorAnalysis, TSNE.

**Ensemble (`ensemble/`):** BaggingClassifier, BaggingRegressor, VotingClassifier,
VotingRegressor, StackingClassifier, StackingRegressor. (Plus AdaBoost, ExtraTrees,
SGD, HistGradientBoosting via models/ — already counted above.)

**Anomaly Detection (`models/isolation_forest.rs`, `models/lof.rs`):**
IsolationForest, LocalOutlierFactor.

**Other (`models/`):** GaussianProcess (classifier + regressor), QDA, IsotonicRegression,
MultiOutputClassifier/Regressor, PassiveAggressive, Perceptron, SGD variants,
RidgeClassifier, NearestCentroid.

**Neural (`neural/`):** Present as a module — depth not fully mapped, but bindings registered.

**Total:** ~55 distinct algorithm implementations. The "55+" claim is accurate if counting
all leaf implementations across all modules.

---

## 4. What "AutoML" Exists Today

**Short answer: A real AutoML loop exists, but hyperparameter optimization is missing from it.**

`AutoML::fit()` (`automl/fit.rs:839`) is a genuine end-to-end orchestration:

1. Analyzes data characteristics (`DataCharacteristics::from_data`).
2. Selects an algorithm portfolio (`AlgorithmPortfolio::for_classification/regression`) and
   adapts search spaces to data size/type.
3. Allocates time budget via a multi-armed bandit (`TimeBudgetAllocator`, UCB1/Thompson/
   EpsilonGreedy/SuccessiveHalving strategies).
4. Runs stratified k-fold CV for each algorithm via `run_trial()`.
5. Builds an ensemble from successful trials (`EnsembleBuilder`).
6. Reports a leaderboard with confidence intervals, pairwise statistical comparisons, and
   aggregated feature importance.

**Critical gap: `run_trial()` (`automl/fit.rs:1064`) calls `create_model(algorithm)` which
uses `new()` defaults.** No hyperparameter grid or Bayesian sampling is applied inside the
AutoML loop. The `HPO` module (`hpo/`) with its `SearchSpace`, `BayesianSampler`, TPE, and
`Study` is complete and unit-tested, but is not wired into `AutoML::fit()`. The
`AlgorithmConfig.search_space` field and `adapt_to_data()` are defined but the loop does not
sample from them.

**Meta-learning / warm-start:** `MetaLearningStore` and `WarmStartResult`
(`automl/warmstart.rs`) exist with full similarity search and configuration retrieval. The
store starts empty at runtime (no pre-populated datasets shipped with the library). Warm-start
configs feed into `WarmStartSampler` but this sampler is also not invoked inside `AutoML::fit()`.

**Transfer learning:** `TransferConfig` and `TransferredSearchSpace` (`automl/transfer.rs`)
are implemented. Same issue: not wired into the fit loop.

**Summary:** AutoML is an algorithm-selection-with-ensemble system, not a CASH (Combined
Algorithm Selection and Hyperparameter optimization) system. The H in CASH is stubbed out.

---

## 5. Statistical Rigor Layer

**Implemented and substantive:**

- **Confidence intervals** (`stats/confidence.rs`): t-distribution CIs on CV fold scores.
  `CVResult.ci_lower`/`ci_upper` populated everywhere.
- **Hypothesis testing** (`stats/hypothesis.rs`): t-test, Wilcoxon, McNemar, 5×2cv paired
  t-test, corrected resampled t-test (Bouckaert). `metrics/` includes `paired_ttest`,
  `corrected_resampled_ttest`.
- **Effect sizes** (`stats/effect_size.rs`): Cohen's d, Glass's Δ, Hedges' g.
- **Multiple testing** (`stats/multiple_testing.rs` + `lib.rs::MultipleTesting` enum):
  Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli. Applied to model comparison
  pairwise p-values in `AutoMLResult.model_comparisons`.
- **Bootstrap** (`stats/bootstrap.rs`): `Bootstrap` struct with `BootstrapResult`.
- **Power analysis** (`stats/power.rs`): power calculation module present.
- **Residual diagnostics** (`stats/diagnostics.rs`): Normality tests, Durbin-Watson.
- **Assumption testing** (`FerroError::assumption_violation` + `AssumptionTestResult`):
  integrated into linear model diagnostics.

**Not present:** Conformal prediction / prediction sets. No conformal calibration layer.
Prediction intervals exist for linear models via OLS residuals, but no distribution-free
conformal guarantees for arbitrary models.

---

## 6. Test Architecture

### Module structure (`ferroml-core/src/testing/`)

35 test utility submodules: `assertions`, `checks`, `properties`, `automl`, `callbacks`,
`categorical`, `cv_advanced`, `drift`, `ensemble_advanced`, `explainability`, `fairness`,
`gaussian_process`, `gpu_fallback`, `hpo`, `incremental`, `metrics`, `metrics_custom`,
`multioutput`, `multioutput_models`, `mutation`, `nan_inf_validation`, `onnx`,
`probabilistic`, `regression`, `serialization`, `sparse_models`, `sparse_tests`,
`svm`, `text_pipeline*`, `transformer`, `utils`, `weights`.

### "5-Layer" verification — what it actually is

The docs reference 5 layers. Mapping to code:
1. **Reference-Match**: Algorithm outputs compared against known-correct values (e.g., cross-
   validating against sklearn reference on toy datasets). Located in individual test modules
   and `testing/regression.rs`.
2. **Textbook**: Analytical results (e.g., OLS coefficient formula, closed-form Ridge).
   Checked in `testing/assertions.rs` with calibrated tolerances (`tolerances::CLOSED_FORM`,
   `tolerances::ITERATIVE`).
3. **Property/Invariant**: `testing/properties.rs` — idempotency, scale-invariance,
   output-range checks.
4. **Adversarial**: `testing/nan_inf_validation.rs`, edge cases — all-identical features,
   single sample, NaN/Inf inputs.
5. **"Frankenstein"**: Cross-component integration tests — `text_pipeline_integration.rs`,
   `cv_advanced.rs`, `ensemble_advanced.rs`, `multioutput_models.rs`.

### Coverage and quantity

`grep -r "#\[test\]\|fn test_" ferroml-core/src/ | wc -l` → 7,581 (counts each `#[test]`
and `fn test_` annotation separately; approximately 3,700–4,000 unique test functions).
Implementation plan (2026-03-08) reported 2,701 Rust + 1,006 Python tests; actual count has
grown since. The 6,500+ tests claim in the repo description appears to be roughly accurate
(Rust + Python combined, plus property-based tests that expand into multiple scenarios).

### Mutation testing

`mutants.toml` configures `cargo-mutants` targeting `ferroml-core`, 60s timeout, 2 parallel
jobs. GPU, serialization, ONNX, and test modules are excluded. Mutation workflow present in
`.github/workflows/mutation.yml`.

### Compliance tests

`ferroml-core/src/preprocessing/compliance_tests/compliance.rs` (504 LOC) — systematic
sklearn-compatibility checks for preprocessing transformers.

`testing/checks.rs` provides 30+ individual `check_*()` functions (not-fitted error,
n_features_in tracking, repeated fit, consistent predictions, etc.) composable via
`check_estimator()`.

---

## 7. Python Binding Surface (PyO3)

### Exposed modules

`ferroml.linear`, `ferroml.trees`, `ferroml.ensemble`, `ferroml.neighbors`,
`ferroml.naive_bayes`, `ferroml.neural`, `ferroml.clustering`, `ferroml.preprocessing`,
`ferroml.decomposition`, `ferroml.explainability`, `ferroml.pipeline`, `ferroml.automl`,
`ferroml.datasets`, `ferroml.svm`, `ferroml.calibration`, `ferroml.anomaly`,
`ferroml.gaussian_process`, `ferroml.multioutput`, `ferroml.stats`, `ferroml.metrics`,
`ferroml.cv`, `ferroml.hpo`, `ferroml.model_selection`.
Top-level: `recommend()`, `PyRecommendation`, `PyModelCard`.

### Sklearn compatibility level

- `fit()`, `predict()`, `score()` (R² or accuracy computed inline): present on all
  supervised models.
- `predict_proba()`: present on probabilistic classifiers, calibrated classifiers, GPs.
- `decision_function()`: present on SVM and LogisticRegression wrappers
  (`ferroml-python/src/linear.rs:1210`).
- `partial_fit()`: present on NaiveBayes variants and SGD models (`IncrementalModel` trait).
- `transform()` / `fit_transform()` / `inverse_transform()`: present on all transformers.
- `Pipeline`, `ColumnTransformer`, `FeatureUnion`: exposed.
- `get_params()` / `set_params()` with `step__param` syntax: implemented in Rust pipeline.

**Not present:** `clone()` (Python-side object copy), `get_feature_names_out()` returning
Python strings (partially — some transformers expose it), `BaseEstimator` subclassing so
sklearn utilities like `set_params` grid search work natively.

### Array handling

Read-only zero-copy via `PyReadonlyArray2<f64>`. Output via `into_pyarray()` (zero-copy
ownership transfer). Copy only on `fit()` where owned arrays are required.

### Packaging

abi3 wheels (`pyo3 = { features = ["abi3-py310"] }`), targeting Python 3.10+.
`publish-pypi.yml` CI workflow. `ferroml` is on PyPI at v1.0.1 per workspace version.

**Pandas support** (feature-gated): `fit_pandas()` / `predict_pandas()` adapters via
`pandas_utils.rs`. **Polars support** (feature-gated): `pyo3-polars` integration.
**Sparse support** (feature-gated): `sparse_utils.rs` — converts scipy CSR to `CsrMatrix`.

---

## 8. Performance Layer

### SIMD (`simd.rs`, `wide` crate — optional feature)

30+ functions: `squared_euclidean_distance`, `batch_squared_euclidean`,
`cosine_similarity_batch`, `matrix_vector_mul`, `axpy`/`axpby`, `row_norms_l2`,
`normalize_rows_l2`, `col_means`, etc. Used internally in KNN, clustering, SVM.
`wide` is portable SIMD (compiles to target's best SIMD width). No target-specific
intrinsics.

### Rayon (`rayon` crate — always on)

Used in: `cross_val_score` (parallel folds), `RandomForest` tree building, `FeatureUnion`
(parallel transformers), `ColumnTransformer` (parallel columns). The main AutoML trial loop
is sequential (one algorithm at a time per bandit selection).

### GPU (`gpu/` — optional `gpu` feature, wgpu)

Operations: matmul, pairwise distances, relu, sigmoid, softmax, row_sum, row_max,
bias_add, relu_grad, sigmoid_grad, elementwise_mul.

Precision: f64→f32 on upload, f32→f64 on download (wgpu shaders are f32-only; ~1e-7
relative precision loss documented in mod.rs).

GPU dispatch policy: `Auto { min_elements: 4096 }` (64×64 threshold).
Algorithms with `with_gpu()` hooks: KMeans, DBSCAN, HDBSCAN, TSNE.

**Assessment**: GPU acceleration is architecturally wired but excluded from mutation testing
and has no CI coverage (`exclude_globs` in `mutants.toml` explicitly excludes `gpu/**`
with the note "no tests"). The dispatch is conditional; GPU is a bonus path, not the primary
one.

### Linear algebra

No BLAS/LAPACK. `ndarray-linalg` was removed due to the OpenBLAS build requirement.
QR decomposition (Gram-Schmidt and modified-GS), Cholesky, triangular solve, log-det —
all implemented from scratch in `linalg.rs` using `nalgebra` helpers. This is correct for
small-to-medium matrices but will be slower than BLAS for large problems.

---

## 9. Honest Gaps Assessment

### Gap 1 (Critical): HPO not wired into AutoML::fit()

`automl/fit.rs:1086` calls `create_model(algorithm)` then `model.fit(x_train, y_train)`
with default hyperparameters only. The `SearchSpace` per algorithm is defined in
`AlgorithmConfig`, the TPE/Bayesian sampler exists in `hpo/`, but `run_trial()` never
samples from the search space. Every algorithm is evaluated exactly once per bandit
selection with its defaults. This is a "random portfolio evaluation" loop, not hyperparameter
optimization. **Implication: calling this CASH or AutoML in the sklearn-AutoML / Auto-sklearn
sense is inaccurate until HPO sampling is integrated.**

### Gap 2 (High): MetaLearning store ships empty

`MetaLearningStore::new()` returns an empty in-memory store. No pre-populated dataset
records are bundled. Warm-starting requires the user to populate the store from their own
prior runs. This negates the warm-start benefit for first-time users and makes the
"meta-learning" feature invisible in practice.

### Gap 3 (High): No preprocessing pipeline applied during AutoML trials

`AutoML::fit()` does not apply the `PreprocessingSelector` / `PreprocessingPipelineSpec`
it can build (`automl/preprocessing.rs`). The preprocessing module is complete, but
`run_trial()` passes raw `x_train`/`y_train` to each model without any automatic scaling,
imputation, or encoding. Models that require scaling (SVM, KNN, logistic regression) will
underperform on unscaled data.

### Gap 4 (Medium): No data profiling or schema inference at fit() entry

There is a `FeatureSchema` (`schema.rs`) and `DetectedCharacteristics` struct
(`automl/preprocessing.rs:514`) that can infer missing values, categorical indices, skewness.
Neither is called at the start of `AutoML::fit()` to validate or profile the input. The user
gets no warning about missing values, constant features, or severe imbalance before the loop
runs.

### Gap 5 (Medium): GPU tested without CI

`mutants.toml` explicitly excludes `gpu/**`: "GPU/backend code — feature-gated, no tests,
~12K lines". The GPU module is not tested in the mutation workflow or the standard CI. This
means ~12K lines of compute shader code have unknown correctness in practice.

### Additional gaps (not in top 5 but material):

- **Documentation**: No rendered docs site. `docs.yml` exists but outputs to pages not
  confirmed live. Docstrings in Python bindings are partial (LinearRegression has them;
  many other classes do not).
- **Time series / survival**: `Task::TimeSeries` and `Task::Survival` are defined in the
  `Task` enum but `AutoML::fit()` defaults both to regression treatment
  (`_ => AlgorithmPortfolio::for_regression(preset)`).
- **No conformal prediction**: `PredictionWithUncertainty` uses OLS residuals or bootstrap;
  no distribution-free coverage guarantee.
- **Auto-ensemble is greedy forward selection with default config**: `EnsembleConfig` is
  hardcoded inside `fit()` (max_models=10, selection_iterations=50); no user tuning.
- **Neural network depth**: `neural/` is registered in Python bindings but not part of the
  AutoML portfolio and its implementation scope is unclear from external inspection.

---

## 10. Size and Health Stats

### Lines of Code

| Crate | LOC |
|---|---|
| ferroml-core/src | ~207,500 |
| ferroml-python/src | ~35,960 |
| Total | ~243,460 |

### Test Counts

| Location | Count (approx) |
|---|---|
| ferroml-core `#[test]` functions | ~3,700–4,000 (7,581 grep hits counting both `#[test]` attr and `fn test_`) |
| ferroml-python pytest (per impl plan) | 1,006 as of 2026-03-08 |
| Total combined | ~4,700–5,000+ |

The 6,500+ claim in the README includes property-based test scenarios (each property test
expands to many runs) and likely counts the March 2026 baseline plus subsequent additions.

### CI Workflows (`.github/workflows/`)

| Workflow | Purpose |
|---|---|
| `ci.yml` | Main: build, test, clippy, fmt |
| `benchmarks.yml` | Benchmark regression detection on PRs |
| `mutation.yml` | `cargo-mutants` on ferroml-core |
| `publish.yml` / `publish-pypi.yml` | Crates.io + PyPI publishing |
| `release.yml` | Release automation |
| `changelog.yml` | CHANGELOG generation via git-cliff |
| `docs.yml` | Documentation build |
| `fuzz.yml` | Fuzz testing |

### Cargo Features

`ferroml-core` optional features: `gpu` (wgpu), `simd` (wide), `sparse` (sprs),
`datasets` (polars, arrow), `gpu-diagnostics`. Default features not specified explicitly
(all off unless enabled).

---

*Report written 2026-07-04. Source: repository at `/home/tlupo/experiments/ferroml`.
No build was run; counts are from static analysis and grep.*
