# Codebase Structure

**Analysis Date:** 2026-03-15

## Directory Layout

```
ferroml/                               # Cargo workspace root
├── ferroml-core/                      # Core Rust library (primary)
│   ├── src/
│   │   ├── lib.rs                     # Main library entry, module declarations
│   │   ├── error.rs                   # FerroError enum, Result type alias
│   │   ├── linalg.rs                  # Linear algebra utilities (SVD, QR, etc.)
│   │   ├── schema.rs                  # Feature schema validation
│   │   ├── serialization.rs           # Model persistence (bincode, JSON, ONNX)
│   │   ├── simd.rs                    # SIMD operations (wide crate)
│   │   ├── sparse.rs                  # Sparse matrix types (sprs CSR/CSC)
│   │   │
│   │   ├── models/                    # Machine learning algorithms (55+)
│   │   │   ├── mod.rs                 # Module exports, Model trait
│   │   │   ├── traits.rs              # Model, StatisticalModel, ProbabilisticModel traits
│   │   │   ├── linear.rs              # LinearRegression with diagnostics
│   │   │   ├── regularized.rs         # Ridge, Lasso, ElasticNet with CV variants
│   │   │   ├── logistic.rs            # LogisticRegression with odds ratios
│   │   │   ├── sgd.rs                 # SGDClassifier, SGDRegressor, Perceptron
│   │   │   ├── tree.rs                # DecisionTreeClassifier/Regressor
│   │   │   ├── forest.rs              # RandomForestClassifier/Regressor
│   │   │   ├── extra_trees.rs         # ExtraTreesClassifier/Regressor
│   │   │   ├── boosting.rs            # GradientBoostingClassifier/Regressor
│   │   │   ├── hist_boosting.rs       # HistGradientBoostingClassifier/Regressor
│   │   │   ├── adaboost.rs            # AdaBoostClassifier/Regressor
│   │   │   ├── svm.rs                 # SVC, SVR, LinearSVC, LinearSVR
│   │   │   ├── knn.rs                 # KNeighborsClassifier/Regressor, KDTree, BallTree
│   │   │   ├── naive_bayes.rs         # GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
│   │   │   ├── gaussian_process.rs    # GaussianProcessClassifier/Regressor with kernels
│   │   │   ├── qda.rs                 # QuadraticDiscriminantAnalysis
│   │   │   ├── isotonic.rs            # IsotonicRegression
│   │   │   ├── quantile.rs            # QuantileRegression (multi-quantile)
│   │   │   ├── robust.rs              # RobustRegression (Huber, M-estimators)
│   │   │   ├── multioutput.rs         # MultiOutputClassifier, MultiOutputRegressor
│   │   │   ├── calibration.rs         # CalibratedClassifierCV with Sigmoid/Isotonic
│   │   │   ├── lof.rs                 # LocalOutlierFactor (anomaly detection)
│   │   │   ├── isolation_forest.rs    # IsolationForest (anomaly detection)
│   │   │   └── compliance_tests/      # Internal trait compliance test suite
│   │   │
│   │   ├── clustering/                # Clustering algorithms
│   │   │   ├── mod.rs                 # ClusteringModel trait
│   │   │   ├── kmeans.rs              # KMeans with kmeans++ init
│   │   │   ├── dbscan.rs              # DBSCAN with epsilon tuning
│   │   │   ├── agglomerative.rs       # Agglomerative with linkage variants
│   │   │   ├── gmm.rs                 # Gaussian Mixture Model
│   │   │   ├── hdbscan.rs             # Hierarchical DBSCAN
│   │   │   ├── diagnostics.rs         # Silhouette, CalinskiHarabasz, Davies-Bouldin
│   │   │   └── metrics.rs             # Adjusted Rand, NMI, Hopkins statistic
│   │   │
│   │   ├── preprocessing/             # Feature transformation
│   │   │   ├── mod.rs                 # Transformer trait
│   │   │   ├── scalers.rs             # StandardScaler, MinMax, RobustScaler, Normalizer
│   │   │   ├── encoders.rs            # OneHot, Ordinal, Target, Binary encoders
│   │   │   ├── imputers.rs            # SimpleImputer, KNNImputer, IterativeImputer
│   │   │   ├── selection.rs           # Variance, SelectKBest, RFE, RFECV
│   │   │   ├── power.rs               # BoxCox, YeoJohnson transforms
│   │   │   ├── polynomial.rs          # PolynomialFeatures generator
│   │   │   ├── discretizers.rs        # KBinsDiscretizer, QuantileDiscretizer
│   │   │   ├── quantile.rs            # QuantileTransformer
│   │   │   ├── sampling.rs            # RandomUnderSampler, SMOTE
│   │   │   ├── tfidf.rs               # TfidfTransformer (dense)
│   │   │   ├── count_vectorizer.rs    # CountVectorizer (sparse, gated)
│   │   │   ├── tfidf_vectorizer.rs    # TfidfVectorizer (sparse, gated)
│   │   │   └── compliance_tests/      # Internal test suite
│   │   │
│   │   ├── decomposition/             # Dimensionality reduction
│   │   │   ├── mod.rs                 # Module exports
│   │   │   ├── pca.rs                 # PCA, IncrementalPCA
│   │   │   ├── truncated_svd.rs       # TruncatedSVD
│   │   │   ├── nmf.rs                 # Non-Negative Matrix Factorization
│   │   │   ├── lda.rs                 # Linear Discriminant Analysis
│   │   │   ├── factor_analysis.rs     # FactorAnalysis
│   │   │   ├── tsne.rs                # t-SNE (Barnes-Hut optimized)
│   │   │   └── umap.rs                # UMAP (if available)
│   │   │
│   │   ├── pipeline/                  # Workflow composition
│   │   │   ├── mod.rs                 # Pipeline, FeatureUnion, ColumnTransformer
│   │   │   └── text_pipeline.rs       # TextPipeline for text data (sparse, gated)
│   │   │
│   │   ├── cv/                        # Cross-validation
│   │   │   ├── mod.rs                 # CVFold, CrossValidator trait
│   │   │   ├── kfold.rs               # KFold, RepeatedKFold
│   │   │   ├── stratified.rs          # StratifiedKFold, RepeatedStratifiedKFold
│   │   │   ├── timeseries.rs          # TimeSeriesSplit
│   │   │   ├── group.rs               # GroupKFold, StratifiedGroupKFold
│   │   │   ├── loo.rs                 # LeaveOneOut, LeavePOut, ShuffleSplit
│   │   │   ├── nested.rs              # nested_cv_score
│   │   │   ├── curves.rs              # learning_curve, validation_curve
│   │   │   └── search.rs              # GridSearchCV, RandomizedSearchCV
│   │   │
│   │   ├── hpo/                       # Hyperparameter optimization
│   │   │   ├── mod.rs                 # Trial, TrialState, ParameterValue
│   │   │   ├── search_space.rs        # SearchSpace, Parameter, ParameterType
│   │   │   ├── bayesian.rs            # BayesianOptimizer, GaussianProcessRegressor
│   │   │   ├── samplers.rs            # Sampler trait, GridSampler, RandomSampler, TPESampler
│   │   │   └── schedulers.rs          # HyperbandScheduler, ASHAScheduler, BOHBScheduler
│   │   │
│   │   ├── metrics/                   # Performance metrics
│   │   │   ├── mod.rs                 # Metric trait
│   │   │   ├── classification.rs      # Accuracy, Precision, Recall, F1, ROC-AUC, etc.
│   │   │   ├── regression.rs          # MSE, MAE, R², RMSE
│   │   │   ├── probabilistic.rs       # Log-loss, Brier score
│   │   │   └── comparison.rs          # Paired tests, bootstrap CIs
│   │   │
│   │   ├── stats/                     # Statistical testing
│   │   │   ├── mod.rs                 # Module exports
│   │   │   └── diagnostics.rs         # Shapiro-Wilk, Durbin-Watson, etc.
│   │   │
│   │   ├── automl/                    # Automated machine learning
│   │   │   ├── mod.rs                 # AutoML struct, fit orchestration
│   │   │   ├── selector.rs            # Algorithm selection logic
│   │   │   ├── ensemble.rs            # Ensemble construction
│   │   │   └── utils.rs               # Utilities
│   │   │
│   │   ├── explainability/            # Model interpretability
│   │   │   ├── mod.rs                 # Module exports
│   │   │   ├── shap.rs                # TreeSHAP for tree models
│   │   │   ├── permutation.rs         # Permutation importance
│   │   │   └── pdp.rs                 # Partial dependence plots, ALE
│   │   │
│   │   ├── ensemble/                  # Ensemble utilities
│   │   │   ├── mod.rs                 # Ensemble trait
│   │   │   └── stacking.rs            # StackingClassifier/Regressor
│   │   │
│   │   ├── testing/                   # Test utilities (30+ modules)
│   │   │   ├── mod.rs                 # Module exports
│   │   │   ├── sklearn_fixtures.rs    # Sklearn fixture loading
│   │   │   ├── correctness.rs         # Correctness utilities
│   │   │   └── [24 more test helper modules]
│   │   │
│   │   ├── neural/                    # Neural networks
│   │   │   ├── mod.rs                 # MLPClassifier, MLPRegressor
│   │   │   └── mlp.rs                 # MLP implementation
│   │   │
│   │   ├── datasets/                  # Dataset utilities
│   │   │   ├── mod.rs                 # Module exports
│   │   │   ├── synthetic.rs           # make_classification, make_regression
│   │   │   └── loaders.rs             # Dataset loading
│   │   │
│   │   ├── gpu/                       # GPU acceleration (feature-gated)
│   │   │   ├── mod.rs                 # GpuDispatcher, kernel registry
│   │   │   └── shaders/               # Wgpu compute shaders
│   │   │
│   │   ├── inference/                 # ONNX inference (feature-gated)
│   │   │   ├── mod.rs                 # InferenceSession, Tensor types
│   │   │   └── session.rs             # ONNX runtime wrapping
│   │   │
│   │   ├── onnx/                      # ONNX export (feature-gated)
│   │   │   ├── mod.rs                 # OnnxExportable trait
│   │   │   └── exporter.rs            # Model → ONNX conversion
│   │   │
│   │   └── neural/                    # Neural networks
│   │       ├── mod.rs
│   │       └── mlp.rs
│   │
│   ├── tests/                         # Integration tests (26 test files)
│   │   ├── correctness_*.rs           # Cross-library correctness tests vs sklearn/scipy
│   │   ├── vs_linfa_*.rs              # linfa 0.8.1 compatibility tests
│   │   ├── adversarial_*.rs           # Edge case testing
│   │   ├── real_dataset_validation.rs # Real-world dataset testing
│   │   └── regression/                # Specific regression test files
│   │
│   ├── benches/                       # Criterion.rs benchmarks (86+ functions)
│   │   ├── benches_*.rs               # Performance benchmarks
│   │   └── fixtures/                  # Benchmark data
│   │
│   └── Cargo.toml                     # Core crate manifest
│
├── ferroml-python/                    # PyO3 Python bindings
│   ├── src/
│   │   ├── lib.rs                     # PyModule registration entry point
│   │   ├── array_utils.rs             # Zero-copy array handling
│   │   ├── errors.rs                  # Rust→Python error translation
│   │   ├── pickle.rs                  # Custom pickling for models
│   │   ├── pandas_utils.rs            # Pandas interop (feature-gated)
│   │   ├── polars_utils.rs            # Polars interop (feature-gated)
│   │   ├── sparse_utils.rs            # Sparse matrix handling (feature-gated)
│   │   │
│   │   ├── linear.rs                  # ferroml.linear module
│   │   ├── trees.rs                   # ferroml.trees module (GB models here, not ensemble)
│   │   ├── ensemble.rs                # ferroml.ensemble module (non-GB models)
│   │   ├── neighbors.rs               # ferroml.neighbors module
│   │   ├── naive_bayes.rs             # ferroml.naive_bayes module
│   │   ├── svm.rs                     # ferroml.svm module
│   │   ├── clustering.rs              # ferroml.clustering module
│   │   ├── decomposition.rs           # ferroml.decomposition module
│   │   ├── preprocessing.rs           # ferroml.preprocessing module
│   │   ├── gaussian_process.rs        # ferroml.gaussian_process module
│   │   ├── calibration.rs             # ferroml.calibration module
│   │   ├── anomaly.rs                 # ferroml.anomaly module
│   │   ├── multioutput.rs             # ferroml.multioutput module
│   │   ├── explainability.rs          # ferroml.explainability module
│   │   ├── pipeline.rs                # ferroml.pipeline module
│   │   ├── automl.rs                  # ferroml.automl module
│   │   ├── datasets.rs                # ferroml.datasets module
│   │   └── neural.rs                  # ferroml.neural module
│   │
│   ├── python/
│   │   └── ferroml/                   # Python package (pure Python)
│   │       ├── __init__.py            # ferroml top-level re-exports
│   │       ├── linear/
│   │       ├── trees/
│   │       ├── ensemble/
│   │       ├── neighbors/
│   │       ├── naive_bayes/
│   │       ├── svm/
│   │       ├── clustering/
│   │       ├── decomposition/
│   │       ├── preprocessing/
│   │       ├── gaussian_process/
│   │       ├── calibration/
│   │       ├── anomaly/
│   │       ├── multioutput/
│   │       ├── explainability/
│   │       ├── pipeline/
│   │       ├── automl/
│   │       ├── datasets/
│   │       ├── neural/
│   │       └── [14 __init__.py files re-exporting Rust bindings]
│   │
│   ├── tests/                         # Python tests (50+ files)
│   │   ├── test_*.py                  # Pytest test files
│   │   ├── test_vs_*.py               # Cross-library validation (xgboost, lightgbm, sklearn, statsmodels, scipy)
│   │   ├── test_score_all_models.py   # sklearn.score() API parity (Plan U)
│   │   ├── test_partial_fit.py        # Incremental learning tests (Plan U)
│   │   └── test_decision_function.py  # Decision function tests (Plan U)
│   │
│   ├── examples/
│   │   └── notebooks/                 # Jupyter notebooks
│   │
│   ├── python/ferroml/
│   │   ├── [14 submodules with pure Python wrappers]
│   │   └── ferroml.abi3.so            # Compiled PyO3 extension
│   │
│   └── Cargo.toml                     # Python crate manifest (PyO3)
│
├── scripts/                           # Utility scripts
│   ├── benchmark_cross_library.py     # Multi-library performance benchmarks
│   └── [other automation scripts]
│
├── benchmarks/                        # Criterion.rs benchmark configs
│   └── fixtures/                      # Benchmark data files
│
├── thoughts/shared/                   # Implementation planning docs
│   ├── plans/                         # Phase plans A-U (completed)
│   ├── audit-report.md                # Robustness audit (Plan O, complete)
│   └── handoffs/                      # Phase completion handoffs
│
├── .planning/codebase/                # GSD documentation (this directory)
│   ├── ARCHITECTURE.md                # This document
│   ├── STRUCTURE.md                   # This document
│   └── [other analysis docs]
│
├── Cargo.toml                         # Workspace manifest
├── Cargo.lock                         # Dependency lock file
├── README.md                          # Project overview
└── .gitignore                         # Git exclusions
```

## Directory Purposes

**ferroml-core/src/**
- Purpose: Core Rust library implementation
- Contains: 55+ ML algorithms, preprocessing, CV, HPO, metrics, AutoML
- Entry point: `lib.rs` - declares all modules, re-exports public API

**ferroml-core/src/models/**
- Purpose: Machine learning algorithms with statistical diagnostics
- Key pattern: Each model is a struct implementing `Model`, `StatisticalModel`, or `ProbabilisticModel` trait
- Build pattern: Builder-style methods for configuration (e.g., `LinearRegression::new().with_fit_intercept(false)`)

**ferroml-core/src/preprocessing/**
- Purpose: Feature transformations following sklearn API
- Key pattern: Each transformer implements `Transformer` trait with `fit()` and `transform()` methods
- Feature: Stateless after fit (all state stored in struct fields), reusable

**ferroml-core/src/pipeline/**
- Purpose: Composable workflows (Pipeline, FeatureUnion, ColumnTransformer)
- Key pattern: Acts as proxy `Model` by delegating fit/predict to internal steps
- Use case: Chain scalers + models, parallel feature extraction, column-specific transforms

**ferroml-core/src/cv/**
- Purpose: Cross-validation splitters and search methods
- Key pattern: Splitters generate `CVFold` vectors; search methods iterate folds evaluating models
- Re-exports: KFold, StratifiedKFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

**ferroml-core/src/hpo/**
- Purpose: Hyperparameter optimization with multiple strategies
- Strategies: Bayesian (GP-based), Grid, Random, TPE, ASHA, BOHB
- Pattern: Trial abstraction with state tracking (Running, Complete, Pruned, Failed)

**ferroml-core/src/metrics/**
- Purpose: Performance evaluation metrics
- Organization: classification.rs, regression.rs, probabilistic.rs, comparison.rs
- Pattern: Each metric is a function `fn metric(y_true, y_pred) -> f64`

**ferroml-core/src/stats/**
- Purpose: Statistical hypothesis testing and diagnostics
- Key: Normality tests (Shapiro-Wilk), autocorrelation (Durbin-Watson), assumption validation
- Used by: Linear regression diagnostics, model validation

**ferroml-core/tests/**
- Purpose: Integration tests and cross-library validation
- Correctness tests: Verify against sklearn/scipy fixtures (26 test files)
- Linfa tests: Verify compatibility with linfa 0.8.1 (6 test files)
- Adversarial tests: Edge cases, numerical stability, robustness

**ferroml-python/src/**
- Purpose: PyO3 bindings exposing Rust library to Python
- Organization: One module per Python submodule (linear.rs → ferroml.linear)
- Pattern: Each Rust struct wrapped as Python class with methods bound via `#[pymethods]`

**ferroml-python/python/ferroml/**
- Purpose: Pure Python wrappers and re-exports
- Organization: 14 submodules mirroring Rust structure
- Pattern: Each `__init__.py` re-exports classes from compiled `.so` extension

**ferroml-python/tests/**
- Purpose: Python test suite
- Cross-library tests: Validate against xgboost, lightgbm, sklearn, statsmodels, scipy
- API tests: Verify sklearn compatibility (score, partial_fit, decision_function)
- Coverage: ~1,923 tests across 50+ test files

## Key File Locations

**Entry Points:**
- `ferroml-core/src/lib.rs`: Rust library entry (module declarations, trait re-exports)
- `ferroml-python/src/lib.rs`: Python module entry (PyO3 registration, submodule setup)
- `ferroml-python/python/ferroml/__init__.py`: Python package entry (high-level imports)

**Configuration:**
- `Cargo.toml` (root): Workspace manifest, workspace dependencies, profiles
- `Cargo.toml` (ferroml-core): Core library manifest, features (onnx, gpu, sparse, simd)
- `Cargo.toml` (ferroml-python): Python bindings manifest, PyO3 config
- `pyproject.toml` (if exists): Python build metadata, maturin config

**Core Logic:**
- `ferroml-core/src/models/linear.rs`: LinearRegression reference implementation
- `ferroml-core/src/models/forest.rs`: RandomForest reference implementation
- `ferroml-core/src/preprocessing/scalers.rs`: StandardScaler reference preprocessing
- `ferroml-core/src/pipeline/mod.rs`: Pipeline composition engine
- `ferroml-core/src/cv/search.rs`: GridSearchCV / RandomizedSearchCV optimization loops

**Error Handling:**
- `ferroml-core/src/error.rs`: FerroError enum, Result type alias
- `ferroml-python/src/errors.rs`: Rust→Python error translation

**Testing & Validation:**
- `ferroml-core/src/testing/mod.rs`: Test utilities (30+ helper modules)
- `ferroml-core/tests/correctness_*.rs`: Cross-library validation (sklearn/scipy)
- `ferroml-core/tests/vs_linfa_*.rs`: linfa compatibility tests
- `ferroml-python/tests/test_vs_*.py`: Python cross-library tests

**Serialization & Persistence:**
- `ferroml-core/src/serialization.rs`: Model save/load (bincode, JSON, ONNX)
- `ferroml-python/src/pickle.rs`: Custom pickling for Python objects

**Performance & Optimization:**
- `ferroml-core/src/linalg.rs`: Linear algebra primitives (SVD, QR, Cholesky)
- `ferroml-core/src/simd.rs`: SIMD operations via `wide` crate (feature-gated)
- `ferroml-core/src/sparse.rs`: Sparse matrix types using `sprs` (feature-gated)
- `ferroml-core/src/gpu/mod.rs`: GPU acceleration framework (feature-gated)

## Naming Conventions

**Files:**
- Model implementations: `{lowercase_algorithm_name}.rs` (e.g., `linear.rs`, `forest.rs`)
- Submodule aggregators: `mod.rs` at directory root (re-exports, trait definitions)
- Test files: `test_*.py` (Python), inline in Rust source with `#[cfg(test)]` modules
- Utility modules: Descriptive names like `scalers.rs`, `encoders.rs`, `diagnostics.rs`

**Directories:**
- Algorithm groups: `{category}` (models, preprocessing, clustering, decomposition)
- Test suites: `tests/` at crate root
- Internal organization: Flat structure with clear module boundaries (no deeply nested dirs)

**Rust Types:**
- Structs: PascalCase (e.g., `LinearRegression`, `StandardScaler`, `KMeans`)
- Methods: snake_case (e.g., `fit()`, `predict()`, `transform()`)
- Traits: PascalCase (e.g., `Model`, `Transformer`, `ClusteringModel`)
- Enums: PascalCase variants (e.g., `FerroError::ShapeMismatch`, `TrialState::Complete`)

**Python Classes:**
- Mimic sklearn naming: PascalCase (e.g., `LinearRegression`, `StandardScaler`)
- Maintain submodule organization: `ferroml.linear.LinearRegression`, `ferroml.preprocessing.StandardScaler`
- Methods: snake_case (e.g., `fit()`, `predict()`, `transform()`)

## Where to Add New Code

**New ML Algorithm:**
1. Create file: `ferroml-core/src/models/{algorithm_name}.rs`
2. Implement struct with configuration fields
3. Implement `Model` trait: `fn fit(&mut self, x, y) -> Result<()>`, `fn predict(&self, x) -> Result<Array1<f64>>`
4. Implement `StatisticalModel` or `ProbabilisticModel` if applicable
5. Add `fn search_space(&self) -> SearchSpace` for HPO
6. Add to `ferroml-core/src/models/mod.rs` re-exports
7. Add Python bindings: `ferroml-python/src/ensemble.rs` or appropriate module
8. Add tests: `ferroml-core/tests/correctness_*.rs` and `ferroml-python/tests/test_*.py`

**New Preprocessing Transformer:**
1. Create file: `ferroml-core/src/preprocessing/{transformer_name}.rs` or add to existing file (e.g., `scalers.rs`)
2. Implement struct with configuration and learned parameters
3. Implement `Transformer` trait: `fn fit(&mut self, x) -> Result<()>`, `fn transform(&self, x) -> Result<Array2<f64>>`
4. Implement `inverse_transform()` if applicable
5. Add to `ferroml-core/src/preprocessing/mod.rs` re-exports
6. Add Python bindings: `ferroml-python/src/preprocessing.rs`
7. Add tests: `ferroml-core/tests/correctness_preprocessing.rs` and `ferroml-python/tests/test_preprocessing.py`

**New Clustering Algorithm:**
1. Create file: `ferroml-core/src/clustering/{algorithm_name}.rs`
2. Implement struct with configuration
3. Implement `ClusteringModel` trait: `fn fit(&mut self, x) -> Result<()>`, `fn predict(&self, x) -> Result<Array1<usize>>`
4. Add diagnostics in `ferroml-core/src/clustering/diagnostics.rs` if needed
5. Add to `ferroml-core/src/clustering/mod.rs` re-exports
6. Add Python bindings: `ferroml-python/src/clustering.rs`
7. Add tests: `ferroml-core/tests/correctness_clustering.rs`

**New Metric:**
1. Add to `ferroml-core/src/metrics/{category}.rs` (classification/regression/probabilistic)
2. Function signature: `pub fn metric_name(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64>`
3. Add to `ferroml-core/src/metrics/mod.rs` re-exports and `Metric` enum if applicable
4. Add tests: inline in metrics file with `#[cfg(test)]` module

**New Cross-Validation Strategy:**
1. Create file: `ferroml-core/src/cv/{strategy_name}.rs`
2. Implement struct
3. Implement `CrossValidator` trait: `fn split(...) -> Result<Vec<CVFold>>`
4. Add to `ferroml-core/src/cv/mod.rs` re-exports
5. Add tests: inline in cv module or dedicated test file

**New Hyperparameter Optimization Method:**
1. Add to `ferroml-core/src/hpo/samplers.rs` (for sampling strategies) or `schedulers.rs` (for multi-fidelity)
2. Implement `Sampler` or `Scheduler` trait
3. Add to module re-exports
4. Add tests: inline in HPO module

## Special Directories

**ferroml-core/src/models/compliance_tests/ (Generated tests)**
- Purpose: Automatically generated compliance tests for all models
- Generated: Yes (by internal test utility)
- Committed: Yes
- Usage: Ensures all models satisfy required trait contracts

**ferroml-core/src/preprocessing/compliance_tests/ (Generated tests)**
- Purpose: Automatically generated compliance tests for all transformers
- Generated: Yes
- Committed: Yes
- Usage: Ensures all transformers satisfy Transformer trait contract

**ferroml-core/src/testing/ (30+ test helper modules)**
- Purpose: Shared test utilities (sklearn fixtures, assertion helpers, test data generators)
- Generated: No
- Committed: Yes
- Key modules: `sklearn_fixtures.rs`, `correctness.rs`, `datasets.rs`

**ferroml-core/tests/regression/ (Regression test suite)**
- Purpose: Dedicated regression testing infrastructure
- Generated: No
- Committed: Yes
- Contains: Baseline models, guards against numerical regressions

**ferroml-python/python/ferroml/ (Python package)**
- Purpose: Pure Python wrappers re-exporting from compiled Rust bindings
- Generated: No (but compiled `.so` is generated)
- Committed: Yes (source), No (compiled `.so`)
- Pattern: Each `__init__.py` re-exports from `ferroml` (PyO3 module)

**benchmarks/ and scripts/ (Performance infrastructure)**
- Purpose: Criterion.rs benchmarks, cross-library performance comparison
- Generated: No
- Committed: Yes
- Usage: Performance tracking, regression detection

**thoughts/shared/ (Planning documentation)**
- Purpose: Implementation plans (Plans A-U), audit reports, handoffs
- Generated: No (manual)
- Committed: Yes
- Pattern: Completed plans preserved for reference

---

*Structure analysis: 2026-03-15*
