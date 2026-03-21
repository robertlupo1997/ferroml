# Codebase Structure

**Analysis Date:** 2026-03-20

## Directory Layout

```
ferroml/
├── ferroml-core/                    # Rust library (all algorithms)
│   ├── src/
│   │   ├── lib.rs                   # Module exports, top-level traits
│   │   ├── error.rs                 # FerroError enum
│   │   ├── schema.rs                # Feature validation schema
│   │   ├── serialization.rs         # Model persistence (JSON/MessagePack/Bincode)
│   │   ├── linalg.rs                # Linear algebra (QR, Cholesky, SVD via nalgebra)
│   │   ├── models/                  # 25+ ML algorithms
│   │   │   ├── mod.rs               # Re-exports all models
│   │   │   ├── traits.rs            # Specialized traits (LinearModel, TreeModel, etc.)
│   │   │   ├── linear.rs            # LinearRegression
│   │   │   ├── regularized.rs       # Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
│   │   │   ├── logistic.rs          # LogisticRegression, LogisticSolver
│   │   │   ├── sgd.rs               # SGD classifier/regressor
│   │   │   ├── svm.rs               # SVC, SVR, LinearSVC, LinearSVR, SVC kernels
│   │   │   ├── tree.rs              # DecisionTreeClassifier, DecisionTreeRegressor
│   │   │   ├── forest.rs            # RandomForestClassifier, RandomForestRegressor
│   │   │   ├── extra_trees.rs       # ExtraTreesClassifier, ExtraTreesRegressor
│   │   │   ├── boosting.rs          # GradientBoostingClassifier, GradientBoostingRegressor
│   │   │   ├── hist_boosting.rs     # HistGradientBoostingClassifier, HistGradientBoostingRegressor
│   │   │   ├── adaboost.rs          # AdaBoostClassifier, AdaBoostRegressor
│   │   │   ├── knn.rs               # KNeighborsClassifier, KNeighborsRegressor, NearestCentroid
│   │   │   ├── gaussian_process.rs  # GaussianProcessClassifier, GaussianProcessRegressor
│   │   │   ├── qda.rs               # QuadraticDiscriminantAnalysis
│   │   │   ├── naive_bayes/         # Naive Bayes implementations
│   │   │   │   └── mod.rs           # GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
│   │   │   ├── calibration.rs       # CalibratedClassifierCV, calibration methods
│   │   │   ├── isolation_forest.rs  # IsolationForest
│   │   │   ├── lof.rs               # LocalOutlierFactor
│   │   │   ├── multioutput.rs       # MultiOutputClassifier, MultiOutputRegressor
│   │   │   ├── quantile.rs          # QuantileRegression
│   │   │   ├── robust.rs            # HuberRegressor, RANSACRegressor
│   │   │   ├── isotonic.rs          # IsotonicRegression
│   │   │   └── compliance_tests/    # Model correctness validators (sklearn parity)
│   │   ├── preprocessing/           # Feature transformers
│   │   │   ├── mod.rs               # Re-exports all transformers
│   │   │   ├── scalers.rs           # StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
│   │   │   ├── encoders.rs          # OneHotEncoder, OrdinalEncoder, TargetEncoder, BinaryEncoder
│   │   │   ├── imputers.rs          # SimpleImputer, KNNImputer
│   │   │   ├── selection.rs         # SelectKBest, SelectPercentile, VarianceThreshold, RFE, RFECV
│   │   │   ├── polynomial.rs        # PolynomialFeatures
│   │   │   ├── power.rs             # PowerTransformer, Yeo-Johnson
│   │   │   ├── quantile.rs          # QuantileTransformer
│   │   │   ├── sampling.rs          # SMOTE, ADASYN, RandomUnderSampler, OverSampler, TomekLinks
│   │   │   ├── count_vectorizer.rs  # CountVectorizer (sparse text)
│   │   │   ├── tfidf.rs             # TfidfTransformer
│   │   │   ├── tfidf_vectorizer.rs  # TfidfVectorizer (sparse text)
│   │   │   ├── discretizers.rs      # KBinsDiscretizer, Binarizer
│   │   │   └── compliance_tests/    # Transformer correctness validators
│   │   ├── stats/                   # Statistical foundations
│   │   │   ├── mod.rs               # StatisticalResult, AssumptionTest
│   │   │   ├── hypothesis.rs        # HypothesisTest, TwoSampleTest, t-test, F-test
│   │   │   ├── diagnostics.rs       # ResidualDiagnostics, NormalityTest, residual plots
│   │   │   ├── confidence.rs        # ConfidenceInterval, Wilson score intervals
│   │   │   ├── effect_size.rs       # CohensD, HedgesG, GlasssDelta
│   │   │   ├── bootstrap.rs         # Bootstrap, BootstrapResult
│   │   │   ├── multiple_testing.rs  # Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli
│   │   │   ├── power.rs             # Statistical power analysis
│   │   │   └── math.rs              # Distribution functions, special functions
│   │   ├── metrics/                 # Model evaluation metrics
│   │   │   ├── mod.rs               # Metric enum, scoring functions
│   │   │   ├── classification.rs    # Accuracy, precision, recall, F1, AUC-ROC, AUC-PR, confusion matrix
│   │   │   ├── regression.rs        # MSE, RMSE, MAE, R², MAPE, SMAPE, MASE, median AE
│   │   │   └── clustering.rs        # Silhouette, Calinski-Harabasz, Davies-Bouldin, homogeneity, completeness, V-measure
│   │   ├── cv/                      # Cross-validation strategies
│   │   │   ├── mod.rs               # CVFold, CrossValidator trait
│   │   │   ├── kfold.rs             # KFold, RepeatedKFold
│   │   │   ├── stratified.rs        # StratifiedKFold, RepeatedStratifiedKFold
│   │   │   ├── timeseries.rs        # TimeSeriesSplit
│   │   │   ├── group.rs             # GroupKFold, StratifiedGroupKFold
│   │   │   ├── loo.rs               # LeaveOneOut, LeavePOut, ShuffleSplit
│   │   │   ├── search.rs            # GridSearchCV, RandomizedSearchCV, SearchResult
│   │   │   ├── nested.rs            # nested_cv_score, NestedCVConfig
│   │   │   └── curves.rs            # learning_curve, validation_curve, LearningCurveResult
│   │   ├── pipeline/                # Workflow composition
│   │   │   ├── mod.rs               # Pipeline, FeatureUnion, ColumnTransformer
│   │   │   └── text_pipeline.rs     # TextPipeline (sparse text + models, feature="sparse")
│   │   ├── hpo/                     # Hyperparameter optimization
│   │   │   ├── mod.rs               # SearchSpace, ParameterValue, sampling
│   │   │   ├── bayesian.rs          # Bayesian optimization (GP-based)
│   │   │   ├── bandit.rs            # Multi-armed bandit algorithm selection
│   │   │   ├── hyperband.rs         # Hyperband, ASHA for resource allocation
│   │   │   └── sampler.rs           # Grid, random, TPE samplers
│   │   ├── automl/                  # End-to-end ML automation
│   │   │   ├── mod.rs               # AutoML, AutoMLConfig, AutoMLResult
│   │   │   ├── search.rs            # Algorithm portfolio, ensemble selection
│   │   │   ├── portfolio.rs         # Model portfolios (Balanced, Aggressive, Conservative)
│   │   │   ├── ensemble.rs          # Ensemble aggregation
│   │   │   ├── comparisons.rs       # Pairwise statistical comparisons
│   │   │   └── result.rs            # AutoMLResult, LeaderboardEntry, statistics aggregation
│   │   ├── clustering/              # Unsupervised learning
│   │   │   ├── mod.rs               # Re-exports clustering models
│   │   │   ├── kmeans.rs            # KMeans, KMeans++, Elkan algorithm
│   │   │   ├── dbscan.rs            # DBSCAN (density-based)
│   │   │   ├── hierarchical.rs      # AgglomerativeClustering (linkage methods)
│   │   │   ├── gaussian_mixture.rs  # GaussianMixture (expectation-maximization)
│   │   │   ├── hdbscan.rs           # HDBSCAN (hierarchical density-based)
│   │   │   └── metrics.rs           # Silhouette, Calinski-Harabasz, Davies-Bouldin, V-measure
│   │   ├── decomposition/           # Dimensionality reduction
│   │   │   ├── mod.rs               # Re-exports all decomposers
│   │   │   ├── pca.rs               # PCA, IncrementalPCA
│   │   │   ├── truncated_svd.rs     # TruncatedSVD
│   │   │   ├── lda.rs               # LinearDiscriminantAnalysis
│   │   │   ├── tsne.rs              # t-SNE (t-distributed stochastic neighbor embedding)
│   │   │   ├── umap.rs              # UMAP (uniform manifold approximation and projection)
│   │   │   ├── factor_analysis.rs   # FactorAnalysis
│   │   │   └── kernel_pca.rs        # KernelPCA
│   │   ├── explainability/          # Model interpretation
│   │   │   ├── mod.rs               # Explainer trait
│   │   │   ├── tree_shap.rs         # TreeSHAP (tree model explanations)
│   │   │   ├── kernel_shap.rs       # KernelSHAP (model-agnostic)
│   │   │   ├── permutation.rs       # Permutation importance
│   │   │   ├── partial_dependence.rs # Partial dependence plots, ICE plots
│   │   │   └── diagnostics.rs       # Residual plots, prediction error analysis
│   │   ├── neural/                  # Neural networks
│   │   │   ├── mod.rs               # MLPClassifier, MLPRegressor
│   │   │   ├── layers.rs            # Dense, activation functions
│   │   │   └── solver.rs            # SGD, Adam, LBFGS solvers
│   │   ├── inference/               # ONNX inference (feature="onnx")
│   │   │   ├── mod.rs               # InferenceSession, Tensor conversion
│   │   │   └── session.rs           # Runtime session management
│   │   ├── onnx/                    # ONNX export (feature="onnx")
│   │   │   ├── mod.rs               # OnnxExportable trait, OnnxConfig
│   │   │   └── export.rs            # Model → ONNX serialization
│   │   ├── gpu/                     # GPU acceleration (feature="gpu")
│   │   │   ├── mod.rs               # GpuDispatcher, shader compilation
│   │   │   └── shaders.rs           # GPU kernel implementations
│   │   ├── sparse/                  # Sparse matrix support (feature="sparse")
│   │   │   ├── mod.rs               # CsrMatrix, CscMatrix wrappers
│   │   │   └── models.rs            # SparseModel implementations
│   │   ├── simd/                    # SIMD operations (feature="simd")
│   │   │   └── mod.rs               # SIMD distance, dot product
│   │   ├── testing/                 # Test utilities
│   │   │   ├── mod.rs               # Fixture loaders, test helpers
│   │   │   └── fixtures.rs          # Test data generation
│   │   └── datasets/                # Built-in datasets
│   │       ├── mod.rs               # load_iris, load_wine, load_diabetes
│   │       ├── synthetic.rs         # make_classification, make_regression, make_blobs
│   │       └── hub.rs               # HuggingFace Hub integration
│   ├── tests/                       # Consolidated integration tests (6 files)
│   │   ├── correctness.rs           # Model correctness vs sklearn/scipy/linfa
│   │   ├── adversarial.rs           # Edge cases, numerical stability
│   │   ├── regression_tests.rs      # Known bug verification
│   │   ├── vs_linfa.rs              # Cross-library comparison with linfa
│   │   ├── edge_cases.rs            # Boundary conditions, empty inputs
│   │   └── integration.rs           # End-to-end workflows
│   ├── benches/                     # Criterion benchmarks
│   │   ├── bench_*.rs               # 5+ benchmark files (86+ functions)
│   │   └── fixtures/                # Benchmark datasets
│   ├── examples/                    # Usage examples
│   └── Cargo.toml                   # Crate manifest
│
├── ferroml-python/                  # PyO3 bindings layer
│   ├── src/                         # Rust bindings code
│   │   ├── lib.rs                   # Main PyO3 module, submodule registration
│   │   ├── array_utils.rs           # Zero-copy ndarray ↔ numpy conversion
│   │   ├── errors.rs                # FerroError → Python exception translation
│   │   ├── pickle.rs                # Model pickling support
│   │   ├── linear/                  # LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression bindings
│   │   ├── trees/                   # DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting bindings
│   │   ├── ensemble/                # AdaBoost, GradientBoosting, SGD, PassiveAggressive bindings
│   │   ├── neighbors/               # KNN, NearestCentroid bindings
│   │   ├── clustering/              # KMeans, DBSCAN, Agglomerative, GaussianMixture, HDBSCAN bindings
│   │   ├── decomposition/           # PCA, t-SNE, TruncatedSVD, LDA bindings
│   │   ├── naive_bayes/             # Naive Bayes bindings
│   │   ├── svm/                     # SVC, SVR, LinearSVC, LinearSVR bindings
│   │   ├── preprocessing/           # Scaler, Encoder, Imputer bindings
│   │   ├── explainability/          # TreeSHAP, permutation importance bindings
│   │   ├── pipeline/                # Pipeline, FeatureUnion, ColumnTransformer bindings
│   │   ├── automl/                  # AutoML, AutoMLConfig bindings
│   │   ├── datasets/                # Dataset loading bindings
│   │   ├── cv/                      # CV splitter bindings
│   │   ├── hpo/                     # HPO, SearchSpace bindings
│   │   ├── metrics/                 # Metric computation bindings
│   │   ├── calibration/             # CalibratedClassifierCV bindings
│   │   ├── anomaly/                 # IsolationForest, LOF bindings
│   │   ├── gaussian_process/        # GP classifier/regressor bindings
│   │   ├── multioutput/             # MultiOutput wrapper bindings
│   │   ├── neural/                  # MLP bindings
│   │   ├── stats/                   # Statistical test bindings
│   │   ├── pandas_utils.rs          # Pandas DataFrame ↔ Array conversion (feature="pandas")
│   │   ├── polars_utils.rs          # Polars DataFrame ↔ Array conversion (feature="polars")
│   │   ├── sparse_utils.rs          # Sparse matrix ↔ scipy.sparse conversion (feature="sparse")
│   │   └── model_selection/         # GridSearchCV, RandomizedSearchCV bindings
│   ├── python/                      # Python package
│   │   └── ferroml/                 # ferroml package
│   │       ├── __init__.py          # Package init, submodule re-exports
│   │       ├── linear/__init__.py   # Linear model re-exports
│   │       ├── trees/__init__.py    # Tree model re-exports
│   │       ├── ensemble/__init__.py # Ensemble re-exports
│   │       ├── neighbors/           # KNN re-exports
│   │       ├── clustering/          # Clustering re-exports
│   │       ├── preprocessing/       # Preprocessing re-exports
│   │       ├── decomposition/       # Decomposition re-exports
│   │       ├── svm/                 # SVM re-exports
│   │       ├── pipeline/            # Pipeline re-exports
│   │       ├── automl/              # AutoML re-exports
│   │       ├── datasets/            # Dataset re-exports
│   │       ├── cv/                  # CV re-exports
│   │       ├── hpo/                 # HPO re-exports
│   │       ├── metrics/             # Metrics re-exports
│   │       ├── calibration/         # Calibration re-exports
│   │       ├── anomaly/             # Anomaly re-exports
│   │       ├── gaussian_process/    # GP re-exports
│   │       ├── multioutput/         # MultiOutput re-exports
│   │       ├── naive_bayes/         # Naive Bayes re-exports
│   │       ├── neural/              # Neural re-exports
│   │       ├── stats/               # Stats re-exports
│   │       ├── explainability/      # Explainability re-exports
│   │       └── model_selection/     # Model selection re-exports
│   ├── tests/                       # Python binding tests
│   │   ├── test_vs_sklearn.py       # sklearn parity tests
│   │   ├── test_vs_xgboost.py       # XGBoost comparison tests
│   │   ├── test_vs_scipy.py         # scipy comparison tests
│   │   ├── test_vs_statsmodels.py   # statsmodels comparison tests
│   │   ├── test_bindings_correctness.py # Array fidelity, pickling, threading
│   │   ├── test_*_feature.py        # Model-specific tests (60+ files)
│   │   └── conftest.py              # pytest fixtures
│   ├── Cargo.toml                   # PyO3 crate manifest
│   └── pyproject.toml               # Python package metadata (maturin)
│
├── scripts/                         # Build and utility scripts
│   ├── benchmark_cross_library.py   # Multi-library benchmark runner
│   └── ci_*.py                      # CI-related automation
│
├── benchmarks/                      # Benchmark data and scripts
│   ├── fixtures/                    # Benchmark datasets
│   │   └── adversarial/             # Edge case datasets
│   └── analysis.py                  # Benchmark analysis
│
├── docs/                            # Documentation
│   ├── plans/                       # Implementation plans (Plans A-X completed)
│   ├── python-api/                  # Auto-generated API docs
│   └── tutorials/                   # Tutorial notebooks
│
├── notebooks/                       # Jupyter notebooks
│   ├── quickstart.ipynb             # Basic usage
│   ├── automl_tutorial.ipynb        # AutoML workflow
│   ├── statistical_diagnostics.ipynb # Stats features demo
│   └── performance_tuning.ipynb     # Optimization techniques
│
├── Cargo.toml                       # Workspace manifest
├── Cargo.lock                       # Dependency lock file
├── REPO_MAP.md                      # Structural skeleton (4,699 lines)
├── CLAUDE.md                        # Project instructions
└── README.md                        # Project overview
```

## Directory Purposes

**ferroml-core/src/:**
- Purpose: All Rust implementations of algorithms, preprocessing, stats, metrics, optimization
- Key files: lib.rs (module exports), error.rs (error types), linalg.rs (matrix ops)
- Dependency rules: Inner modules use lower-level modules; no circular dependencies

**ferroml-core/src/models/:**
- Purpose: 25+ ML algorithms (classification, regression, clustering, anomaly detection)
- Contains: One `.rs` file per model or family (e.g., regularized.rs has Ridge, Lasso, ElasticNet)
- Traits: All implement `Model` + specialized traits (LinearModel, TreeModel, etc.)
- Tests: Compliance tests in subdirectory verify sklearn parity

**ferroml-core/src/preprocessing/:**
- Purpose: Feature transformations (scaling, encoding, imputation, sampling)
- Contains: One `.rs` file per transformer category
- Traits: All implement `Transformer` trait (fit_transform, inverse_transform)
- File order: scalers, encoders, imputers, selectors, then special-purpose (polynomial, power, quantile)

**ferroml-core/src/stats/:**
- Purpose: Statistical foundation for diagnostics, hypothesis testing, CIs, effect sizes
- Contains: hypothesis.rs (tests), diagnostics.rs (residuals), confidence.rs (intervals), effect_size.rs
- Design: Every statistical operation includes assumptions checking, effect sizes, CIs

**ferroml-core/src/cv/:**
- Purpose: Data splitting and model evaluation strategies
- Contains: One `.rs` per strategy (kfold, stratified, timeseries, group, loo)
- search.rs: GridSearchCV, RandomizedSearchCV for hyperparameter tuning
- curves.rs: learning_curve, validation_curve for model analysis

**ferroml-python/src/:**
- Purpose: PyO3 bindings that expose Rust code to Python
- File pattern: One submodule per feature area (matching ferroml-core structure)
- Responsibility: Array conversion, error translation, docstring wrapping
- Zero-copy semantics: Input uses readonly arrays; output uses owned arrays

**ferroml-python/python/ferroml/:**
- Purpose: Python package that re-exports native bindings
- Structure: Each submodule (linear/, trees/, etc.) has __init__.py with re-exports
- Usage: User imports from ferroml.linear, not directly from Rust

## Key File Locations

**Entry Points:**

- `ferroml-core/src/lib.rs`: Rust library exports, main traits definition
- `ferroml-python/src/lib.rs`: PyO3 module registration
- `ferroml-python/python/ferroml/__init__.py`: Python package init
- `Cargo.toml`: Workspace manifest (2 crates)
- `Cargo.lock`: Pinned dependency versions

**Configuration:**

- `ferroml-core/Cargo.toml`: Core library dependencies
- `ferroml-python/Cargo.toml`: PyO3 + binding-specific deps
- `ferroml-python/pyproject.toml`: Python package metadata (maturin config)
- `.pre-commit-config.yaml`: Pre-commit hooks (cargo fmt, clippy, tests)

**Core Logic:**

- `ferroml-core/src/models/mod.rs`: All model re-exports
- `ferroml-core/src/preprocessing/mod.rs`: All transformer re-exports
- `ferroml-core/src/automl/mod.rs`: AutoML orchestrator
- `ferroml-core/src/pipeline/mod.rs`: Pipeline and composition
- `ferroml-core/src/cv/search.rs`: Grid/random search CV

**Testing:**

- `ferroml-core/tests/*.rs`: 6 consolidated integration test files (correctness, adversarial, regression_tests, vs_linfa, edge_cases, integration)
- `ferroml-python/tests/test_*.py`: 60+ Python test files
- `ferroml-core/benches/bench_*.rs`: Criterion benchmarks

**Documentation:**

- `REPO_MAP.md`: Full structural skeleton (4,699 lines, all pub API)
- `docs/plans/`: Implementation plan history (Plans A-X completed)
- `docs/python-api/`: Auto-generated API docs
- `notebooks/`: Tutorial Jupyter notebooks

## Naming Conventions

**Files:**

- Model files: `[model_name].rs` (e.g., `linear.rs`, `logistic.rs`, `forest.rs`)
- Related models in one file: `[family_name].rs` (e.g., `regularized.rs` has Ridge/Lasso/ElasticNet)
- Preprocessing: `[transformer_type].rs` (e.g., `scalers.rs`, `encoders.rs`)
- Tests: `tests/[test_category].rs` (correctness, vs_linfa, edge_cases, etc.)
- Benchmarks: `benches/bench_[area].rs` (bench_tree.rs, bench_linear.rs, etc.)
- Examples: `examples/[example_name].rs`

**Directories:**

- Feature areas: kebab-case (models, preprocessing, decomposition, neural)
- Python submodules: snake_case (linear, trees, ensemble, svm)
- Configuration: dot-prefix (.planning, .github, .claude)
- Build artifacts: `target/` (ignored)

**Functions & Types:**

- Models: PascalCase (LinearRegression, RandomForestClassifier)
- Functions: snake_case (fit, predict, transform)
- Constants: UPPER_SNAKE_CASE (MAX_ITERATIONS, RANDOM_STATE)
- Traits: PascalCase (Model, Predictor, Estimator, Transformer)
- Private helpers: Leading underscore or module-private scoping

## Where to Add New Code

**New Model Implementation:**

1. Create new file in `ferroml-core/src/models/[model_name].rs` OR add to existing family file (e.g., regularized.rs for Ridge variants)
2. Implement trait: `Model` (base) + specialized traits as needed (LinearModel, TreeModel, ProbabilisticModel)
3. Add re-export to `ferroml-core/src/models/mod.rs`
4. Add compliance test in `ferroml-core/src/models/compliance_tests/test_[model_name].rs`
5. Create PyO3 wrapper in `ferroml-python/src/[category]/[model_name].rs`
6. Add re-export to `ferroml-python/src/[category]/mod.rs` and register in `ferroml-python/src/lib.rs`
7. Add Python re-export in `ferroml-python/python/ferroml/[category]/__init__.py`

**New Transformer Implementation:**

1. Create new file in `ferroml-core/src/preprocessing/[transformer_name].rs`
2. Implement `Transformer` trait (fit_transform, inverse_transform)
3. Add re-export to `ferroml-core/src/preprocessing/mod.rs`
4. Follow same PyO3 binding pattern as models (steps 5-7 above, but in preprocessing category)

**New Statistical Test or Metric:**

- **Tests**: Add to `ferroml-core/src/stats/hypothesis.rs` or new file (e.g., stats/manova.rs)
- **Metrics**: Add to `ferroml-core/src/metrics/[classification|regression|clustering].rs`
- **Diagnostics**: Add to `ferroml-core/src/stats/diagnostics.rs`

**New Utility or Helper:**

- **Shared helpers**: `ferroml-core/src/testing/` (for test utilities)
- **Math functions**: `ferroml-core/src/stats/math.rs` (for special functions)
- **Linear algebra**: `ferroml-core/src/linalg.rs` (for matrix operations)

## Special Directories

**ferroml-core/src/models/compliance_tests/:**
- Purpose: Validate correctness against sklearn/scipy/linfa
- Generated: No (hand-written)
- Committed: Yes
- Pattern: Test each model against reference implementation with multiple datasets

**ferroml-core/tests/:**
- Purpose: Integration tests consolidated to reduce binary bloat
- Files: 6 (correctness, adversarial, regression_tests, vs_linfa, edge_cases, integration)
- Generated: No
- Committed: Yes
- Note: Do not create new test binaries; add tests to one of the 6 files

**ferroml-core/benches/:**
- Purpose: Criterion benchmarks for performance tracking
- Files: 5+ (bench_tree.rs, bench_linear.rs, etc.)
- Generated: No
- Committed: Yes
- Output: Stored in `target/criterion/` (gitignored)

**ferroml-core/fixtures/:**
- Purpose: Test datasets and fixtures
- Generated: No (fixed reference data)
- Committed: Yes
- Usage: Loaded by correctness tests and benchmarks

**ferroml-python/tests/:**
- Purpose: Python-level tests (cross-library validation, binding correctness)
- Files: 60+ (test_vs_sklearn.py, test_vs_xgboost.py, test_[model_name]_feature.py, etc.)
- Generated: No
- Committed: Yes
- Pattern: Each model gets dedicated test_[model_name]_feature.py file

**benchmarks/fixtures/:**
- Purpose: Benchmark datasets (large, adversarial)
- Generated: Partially (some via make_* functions)
- Committed: No (stored on disk, loaded at benchmark time)
- Usage: Loaded by scripts/benchmark_cross_library.py

**docs/plans/:**
- Purpose: Implementation plan history (Plans A-X completed)
- Files: 24+ (PLAN_A.md, PLAN_B.md, etc.)
- Generated: No (authored during planning phases)
- Committed: Yes
- Usage: Reference for future phases

**.planning/codebase/:**
- Purpose: Codebase documentation (ARCHITECTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md, STACK.md, INTEGRATIONS.md, STRUCTURE.md)
- Generated: Yes (by gsd:map-codebase)
- Committed: Yes
- Usage: Reference by gsd:plan-phase and gsd:execute-phase

---

*Structure analysis: 2026-03-20*
