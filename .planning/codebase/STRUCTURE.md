# Codebase Structure

**Analysis Date:** 2026-03-16

## Directory Layout

```
ferroml/
├── ferroml-core/                    # Core Rust library
│   ├── src/
│   │   ├── lib.rs                   # Root module, public API re-exports
│   │   ├── automl/                  # Automated machine learning
│   │   ├── clustering/              # Clustering algorithms (KMeans, DBSCAN, etc.)
│   │   ├── cv/                      # Cross-validation splitters
│   │   ├── datasets/                # Dataset loading (Iris, HuggingFace, synthetic)
│   │   ├── decomposition/           # Dimensionality reduction (PCA, t-SNE, LDA, etc.)
│   │   ├── ensemble/                # Ensemble methods (RandomForest, GradientBoosting, Bagging)
│   │   ├── error.rs                 # FerroError type and Result alias
│   │   ├── explainability/          # Model explanations (TreeSHAP, SHAP, permutation importance)
│   │   ├── gpu/                     # GPU shader implementations (when feature enabled)
│   │   ├── hpo/                     # Hyperparameter optimization (Bayesian, Hyperband, ASHA)
│   │   ├── inference/               # ONNX model inference (when feature enabled)
│   │   ├── linalg.rs                # Linear algebra wrappers (nalgebra, QR decomposition)
│   │   ├── metrics/                 # Evaluation metrics (accuracy, ROC-AUC, MSE, etc.)
│   │   ├── models/                  # Supervised learning models
│   │   │   ├── mod.rs               # Model trait, re-exports
│   │   │   ├── linear.rs            # LinearRegression with diagnostics
│   │   │   ├── logistic.rs          # LogisticRegression, multinomial
│   │   │   ├── svm.rs               # SVC, SVR, LinearSVC, LinearSVR
│   │   │   ├── tree.rs              # DecisionTreeClassifier, Regressor
│   │   │   ├── forest.rs            # RandomForestClassifier, Regressor
│   │   │   ├── extra_trees.rs       # ExtraTreesClassifier, Regressor
│   │   │   ├── knn.rs               # KNeighborsClassifier, Regressor, NearestCentroid
│   │   │   ├── gaussian_process.rs  # GaussianProcessRegressor, Classifier
│   │   │   ├── ensemble/ (moved to ensemble module)
│   │   │   ├── boosting.rs          # GradientBoostingClassifier, Regressor
│   │   │   ├── hist_boosting.rs     # HistGradientBoosting (histogram-based)
│   │   │   ├── adaboost.rs          # AdaBoostClassifier, Regressor
│   │   │   ├── naive_bayes/         # GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
│   │   │   ├── regularized.rs       # Ridge, Lasso, ElasticNet + CV variants
│   │   │   ├── robust.rs            # RobustRegression (M-estimators)
│   │   │   ├── quantile.rs          # QuantileRegression
│   │   │   ├── sgd.rs               # SGDClassifier, SGDRegressor, Perceptron
│   │   │   ├── neural.rs            # MLPClassifier, MLPRegressor (simple)
│   │   │   ├── isotonic.rs          # IsotonicRegression
│   │   │   ├── isolation_forest.rs  # IsolationForest (anomaly detection)
│   │   │   ├── lof.rs               # LocalOutlierFactor (anomaly detection)
│   │   │   ├── qda.rs               # QuadraticDiscriminantAnalysis
│   │   │   ├── calibration.rs       # CalibratedClassifierCV, TemperatureScaling
│   │   │   ├── multioutput.rs       # MultiOutputRegressor, MultiOutputClassifier
│   │   │   ├── traits.rs            # Extended traits (LinearModel, IncrementalModel, etc.)
│   │   │   └── compliance_tests/    # Model compliance test fixtures
│   │   ├── neural/                  # Neural network utilities (activation, loss)
│   │   ├── onnx/                    # ONNX model export (when feature enabled)
│   │   ├── pipeline/                # Pipeline and FeatureUnion
│   │   │   └── text_pipeline.rs     # Text pipeline for sparse features
│   │   ├── preprocessing/           # Feature transformers
│   │   │   ├── mod.rs               # Transformer trait, helpers
│   │   │   ├── scalers.rs           # StandardScaler, MinMaxScaler, RobustScaler, etc.
│   │   │   ├── encoders.rs          # OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder
│   │   │   ├── imputers.rs          # SimpleImputer, KNNImputer, IterativeImputer
│   │   │   ├── selection.rs         # VarianceThreshold, SelectKBest, RFE
│   │   │   ├── polynomial.rs        # PolynomialFeatures
│   │   │   ├── power.rs             # PowerTransformer (Box-Cox, Yeo-Johnson)
│   │   │   ├── quantile.rs          # QuantileTransformer
│   │   │   ├── discretizers.rs      # KBinsDiscretizer, Binarizer
│   │   │   ├── sampling.rs          # RandomUnderSampler, SMOTE, etc.
│   │   │   ├── tfidf.rs             # TfidfTransformer
│   │   │   ├── count_vectorizer.rs  # CountVectorizer (sparse text)
│   │   │   ├── tfidf_vectorizer.rs  # TfidfVectorizer (sparse text)
│   │   │   └── compliance_tests/    # Transformer compliance test fixtures
│   │   ├── schema.rs                # FeatureSchema, validation modes
│   │   ├── serialization.rs         # Model save/load (bincode, msgpack, JSON, ONNX)
│   │   ├── simd.rs                  # SIMD operations (feature gated)
│   │   ├── sparse.rs                # Sparse matrix types (CSR, CsrMatrix wrapper)
│   │   ├── stats/                   # Statistical tests and diagnostics
│   │   ├── testing/                 # Test utilities and synthetic data generators
│   │   └── linalg.rs                # Linear algebra (QR, SVD, eigendecomposition)
│   ├── tests/                       # Integration tests (6 consolidated files)
│   │   ├── correctness.rs           # Correctness tests (3,000+ assertions)
│   │   ├── adversarial.rs           # Edge cases and adversarial inputs
│   │   ├── edge_cases.rs            # Boundary conditions, numerical edge cases
│   │   ├── integration.rs           # End-to-end workflows
│   │   ├── vs_linfa.rs              # Cross-library comparison (linfa)
│   │   ├── regression_tests.rs      # Regression tracking for fixed bugs
│   │   ├── regression/              # Per-issue regression fixtures
│   │   └── sklearn_comparison.py    # Python comparison runner script
│   ├── fuzz/                        # Fuzzing harnesses (optional)
│   └── Cargo.toml                   # Core library manifest
│
├── ferroml-python/                  # PyO3 Python bindings
│   ├── src/
│   │   ├── lib.rs                   # pymodule root, submodule registration
│   │   ├── linear.rs                # Linear models wrapper
│   │   ├── trees.rs                 # Tree models wrapper
│   │   ├── ensemble.rs              # Ensemble models wrapper
│   │   ├── svm.rs                   # SVM models wrapper
│   │   ├── neighbors.rs             # KNN models wrapper
│   │   ├── naive_bayes.rs           # Naive Bayes models wrapper
│   │   ├── neural.rs                # Neural network models wrapper
│   │   ├── clustering.rs            # Clustering models + metrics wrapper
│   │   ├── decomposition.rs         # Decomposition models wrapper
│   │   ├── preprocessing.rs         # Preprocessing transformers wrapper
│   │   ├── explainability.rs        # Explainability methods wrapper
│   │   ├── pipeline.rs              # Pipeline and FeatureUnion wrapper
│   │   ├── automl.rs                # AutoML wrapper
│   │   ├── datasets.rs              # Dataset loading wrapper
│   │   ├── calibration.rs           # Calibration wrapper
│   │   ├── anomaly.rs               # Anomaly detection wrapper
│   │   ├── gaussian_process.rs      # Gaussian process wrapper
│   │   ├── multioutput.rs           # Multi-output wrapper
│   │   ├── stats.rs                 # Statistical tests wrapper
│   │   ├── metrics.rs               # Metrics wrapper
│   │   ├── cv.rs                    # CV splitters wrapper
│   │   ├── hpo.rs                   # HPO wrapper
│   │   ├── model_selection.rs       # Model selection utilities (train_test_split, cross_validate)
│   │   ├── array_utils.rs           # Array marshalling (numpy ↔ ndarray)
│   │   ├── pandas_utils.rs          # Pandas DataFrame conversion (feature gated)
│   │   ├── polars_utils.rs          # Polars DataFrame conversion (feature gated)
│   │   ├── sparse_utils.rs          # Sparse matrix utils (feature gated)
│   │   ├── pickle.rs                # Pickle protocol support
│   │   ├── errors.rs                # Python error translation
│   │   └── Cargo.toml               # Python bindings manifest
│   ├── tests/                       # Python integration tests (60+ files, 2,100+ tests)
│   │   ├── test_*.py                # Per-module tests (linear, clustering, preprocessing, etc.)
│   │   ├── test_bindings_correctness.py  # Binding-specific tests (pickle, threads, state)
│   │   ├── test_comparison_*.py     # Cross-library comparison vs sklearn, linfa, xgboost, etc.
│   │   ├── test_cross_library_edge_cases.py  # Adversarial inputs across libraries
│   │   ├── conftest.py              # pytest fixtures
│   │   └── conftest_comparison.py   # Comparison test configuration
│   └── python/                      # Pure Python wrapper code
│       ├── ferroml/
│       │   ├── __init__.py          # Package root, module re-exports
│       │   ├── linear/
│       │   ├── trees/
│       │   ├── ensemble/
│       │   ├── preprocessing/
│       │   ├── pipeline/
│       │   └── [14 submodules]      # Each matches ferroml-python/src/
│       └── setup.py                 # Installation metadata
│
├── .planning/                       # GSD planning system
│   ├── codebase/                    # This file and analysis docs
│   └── phase-*/                     # Implementation phase details
│
├── benchmarks/                      # Criterion benchmark suite
│   ├── benches/
│   │   ├── linear_bench.rs          # LinearRegression, Ridge, Lasso benchmarks
│   │   ├── tree_bench.rs            # DecisionTree, RandomForest benchmarks
│   │   ├── ensemble_bench.rs        # GradientBoosting, AdaBoost benchmarks
│   │   ├── clustering_bench.rs      # KMeans, DBSCAN, Hierarchical benchmarks
│   │   └── preprocessing_bench.rs   # Scaler, Encoder, Imputer benchmarks
│   └── Cargo.toml
│
├── tests/                           # Top-level integration tests
│   ├── correctness/ (consolidated into ferroml-core/tests/)
│   └── vs_sklearn_gaps_phase2.py    # Cross-library compatibility gaps
│
├── notebooks/                       # Tutorial and usage notebooks
│   ├── 01-getting-started.ipynb
│   ├── 02-statistical-diagnostics.ipynb
│   ├── 03-automated-ml.ipynb
│   └── 04-production-deployment.ipynb
│
├── docs/                            # Documentation
│   ├── plans/                       # Implementation plans (Phase A-X complete)
│   └── guides/                      # User guides and API documentation
│
├── scripts/                         # Utility scripts
│   ├── benchmark_cross_library.py   # Multi-library performance comparison
│   ├── generate_fixtures.py         # Test fixture generation
│   └── validate_fixtures.py         # Fixture validation
│
├── REPO_MAP.md                      # Complete structural skeleton (4,699 lines)
├── IMPLEMENTATION_PLAN.md           # Master implementation plan
├── README.md                        # Project overview
├── Cargo.toml                       # Workspace manifest
├── Cargo.lock                       # Dependency lock
├── .pre-commit-config.yaml          # Pre-commit hooks (cargo fmt, clippy)
├── CHANGELOG.md                     # Release notes
└── [Standard files]                 # LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, etc.
```

## Directory Purposes

**ferroml-core/src/models/:**
- Purpose: Supervised learning algorithm implementations
- Contains: 25+ model types, each in its own .rs file (linear.rs, svm.rs, tree.rs, etc.)
- Key files:
  - `mod.rs`: Model trait, re-exports for public API
  - `traits.rs`: Extended traits (LinearModel, TreeModel, IncrementalModel, SparseModel, OutlierDetector, WarmStartModel)
  - Individual files: Each model in isolated implementation (150-3,000 lines depending on complexity)
- Responsibility: Algorithm correctness, statistical diagnostics, hyperparameter search spaces

**ferroml-core/src/preprocessing/:**
- Purpose: Feature preprocessing and transformation
- Contains: Scalers (5 types), encoders (5 types), imputers (3 types), selectors, polynomial, power, quantile
- Key files:
  - `mod.rs`: Transformer trait, validation helpers, common operations
  - `scalers.rs`: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
  - `encoders.rs`: OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder, BinaryEncoder
- Responsibility: Fit/transform pattern, feature name tracking, sparse matrix support (optional)

**ferroml-core/src/stats/:**
- Purpose: Statistical testing, diagnostics, confidence intervals
- Contains: Hypothesis tests (Shapiro-Wilk, Breusch-Pagan, Durbin-Watson), assumption tests, diagnostic plots
- Responsibility: Assumption validation, effect size computation, p-value calculation

**ferroml-core/src/cv/:**
- Purpose: Cross-validation splitter implementations
- Contains: KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut, GroupKFold, ShuffleSplit
- Responsibility: Fold generation, stratification, group/time series handling

**ferroml-core/src/metrics/:**
- Purpose: Evaluation metrics and scoring functions
- Contains: Classification metrics (accuracy, ROC-AUC, F1, log loss, MCC), regression metrics (MSE, MAE, R²), clustering metrics
- Responsibility: Metric computation, threshold-independent scoring

**ferroml-core/src/pipeline/:**
- Purpose: Composite model building
- Contains: Pipeline (sequential steps), FeatureUnion (parallel features), TextPipeline (sparse text)
- Responsibility: Step sequencing, hyperparameter nesting, cross-validation integration

**ferroml-core/tests/:**
- Purpose: Comprehensive testing
- Contains: 6 consolidated test files (consolidation from 19 files in previous iterations)
- Files:
  - `correctness.rs`: 3,000+ tests verifying algorithm correctness against known results
  - `adversarial.rs`: Edge cases and pathological inputs
  - `edge_cases.rs`: Boundary conditions (1 sample, all NaN, single feature, etc.)
  - `integration.rs`: End-to-end workflows (pipeline → AutoML → evaluation)
  - `vs_linfa.rs`: Cross-library comparison with linfa 0.8.1
  - `regression_tests.rs`: Tracking for previously fixed bugs
- Responsibility: Quality assurance, regression prevention, cross-library validation

**ferroml-python/src/:**
- Purpose: PyO3 bindings to expose Rust models to Python
- Contains: 24 module files, each wrapping one Rust module
- Pattern: Each file mirrors core module structure:
  - `register_X_module()` function for initialization
  - `#[pyclass]` wrappers for public types
  - `#[pymethods]` implementations of fit/predict/transform/score
  - Array marshalling via numpy interop
- Responsibility: Python API surface, error translation, array semantics

**ferroml-python/tests/:**
- Purpose: Python integration and cross-library testing
- Contains: 60+ .py files, 2,100+ tests
- File types:
  - `test_*.py`: Per-module tests (test_linear.py, test_clustering.py, etc.)
  - `test_comparison_*.py`: sklearn parity tests (features, output, edge cases)
  - `test_bindings_correctness.py`: Binding-specific tests (state, pickle, threads)
  - `conftest.py`: Fixtures (toy datasets, model instances)
- Responsibility: Python API correctness, cross-library compatibility

## Key File Locations

**Entry Points:**

- `ferroml-core/src/lib.rs`: Root of Rust library, public API surface (508 lines)
- `ferroml-python/src/lib.rs`: Root of Python extension, pymodule registration (104 lines)
- `ferroml-python/python/ferroml/__init__.py`: Python package root, re-exports
- `notebooks/01-getting-started.ipynb`: First user touch point

**Configuration:**

- `Cargo.toml` (workspace root): Workspace members, shared dependencies
- `ferroml-core/Cargo.toml`: Core library features (sparse, gpu, onnx, simd)
- `ferroml-python/Cargo.toml`: Python extension metadata (extension-module, abi3-py310)
- `ferroml-python/python/setup.py`: pip installation metadata
- `.pre-commit-config.yaml`: Linting/formatting hooks (cargo fmt, clippy -D warnings)

**Core Logic:**

- `ferroml-core/src/models/linear.rs`: LinearRegression with diagnostics (2,000+ lines)
- `ferroml-core/src/models/svm.rs`: SVM implementations (3,500+ lines)
- `ferroml-core/src/models/tree.rs`: DecisionTree implementation (2,200+ lines)
- `ferroml-core/src/models/forest.rs`: RandomForest ensemble (1,900+ lines)
- `ferroml-core/src/models/hist_boosting.rs`: HistGradientBoosting (3,700+ lines, most complex)
- `ferroml-core/src/models/boosting.rs`: GradientBoosting (2,400+ lines)
- `ferroml-core/src/preprocessing/scalers.rs`: Feature scaling (1,600+ lines)

**Testing:**

- `ferroml-core/tests/correctness.rs`: Comprehensive algorithm tests (6,000+ lines)
- `ferroml-core/tests/edge_cases.rs`: Boundary condition tests (2,400+ lines)
- `ferroml-core/tests/vs_linfa.rs`: Cross-library validation (1,600+ lines)
- `ferroml-python/tests/test_bindings_correctness.py`: Binding fidelity (400+ lines)
- `ferroml-python/tests/test_comparison_*.py`: sklearn parity (5,000+ combined lines)

**Serialization & Schema:**

- `ferroml-core/src/serialization.rs`: save_model/load_model with formats (2,000+ lines)
- `ferroml-core/src/schema.rs`: FeatureSchema validation (1,400+ lines)

## Naming Conventions

**Files:**

- Model implementations: snake_case matching struct name (linear.rs for LinearRegression, svm.rs for SVC)
- Transformer implementations: snake_case matching type group (scalers.rs for all scaler types)
- Test files: test_*.py for Python, module_name.rs for Rust (tests in single file per logical group)
- Utilities: short names (error.rs, schema.rs, linalg.rs, simd.rs)

**Directories:**

- Feature grouping: domain-specific (models/, preprocessing/, stats/, clustering/, decomposition/)
- Python binding mirrors: Same names as core modules (ferroml-python/src/linear.rs ↔ ferroml-core/src/models/linear.rs)
- Test directory: tests/ at repo root for both crates
- Module subdirs: Only for large feature areas (models/naive_bayes/, preprocessing/compliance_tests/)

**Code Symbols:**

- Model types: PascalCase (LinearRegression, RandomForestClassifier)
- Hyperparameters: snake_case (n_estimators, max_depth, learning_rate)
- Methods: snake_case (fit, predict, transform, score, fit_predict)
- Traits: PascalCase (Model, Transformer, StatisticalModel, ProbabilisticModel)
- Internal helpers: snake_case with leading _ if private (e.g., _compute_residuals)
- Constants: SCREAMING_SNAKE_CASE (DEFAULT_RANDOM_STATE, MAX_ITERATIONS)

## Where to Add New Code

**New Model:**
- Primary implementation: `ferroml-core/src/models/{model_name}.rs`
- Register in: `ferroml-core/src/models/mod.rs` (pub use, pub mod statements)
- Python binding: Add file `ferroml-python/src/{domain}.rs` if new domain, or extend existing
- Register Python: Add to `ferroml-python/src/lib.rs` pymodule function
- Tests: Add correctness tests to `ferroml-core/tests/correctness.rs` module
- Python tests: Add to `ferroml-python/tests/test_{domain}.py`

**New Transformer:**
- Primary implementation: `ferroml-core/src/preprocessing/{group}.rs` (e.g., add to scalers.rs)
- Register in: `ferroml-core/src/preprocessing/mod.rs` (pub use statement)
- Python binding: Add wrapper in `ferroml-python/src/preprocessing.rs`
- Tests: Add correctness tests in `ferroml-core/tests/correctness.rs` → preprocessing module
- Python tests: Add to `ferroml-python/tests/test_comparison_preprocessing.py`

**New Metric:**
- Primary implementation: `ferroml-core/src/metrics/mod.rs`
- Register in: Public fn in metrics module
- Python binding: Add wrapper in `ferroml-python/src/metrics.rs`
- Tests: Add to `ferroml-core/tests/correctness.rs` → metrics module

**Utilities:**
- Shared helpers: `ferroml-core/src/testing/mod.rs` (fixtures, generators)
- Validation: Add check_* function to relevant module's mod.rs
- Linear algebra: Add to `ferroml-core/src/linalg.rs`

## Special Directories

**ferroml-core/fuzz/:**
- Purpose: Fuzzing harnesses for finding edge cases
- Generated: No (static fuzzing code)
- Committed: Yes
- Run: `cargo +nightly fuzz`

**benchmarks/:**
- Purpose: Criterion performance benchmarks (86+ functions)
- Generated: No (static benchmark code), results/ generated at runtime
- Committed: Yes (code only, not results)
- Run: `cargo bench --release`

**notebooks/:**
- Purpose: Tutorial and usage examples
- Generated: No (static .ipynb files with outputs)
- Committed: Yes
- Usage: Jupyter/JupyterLab for interactive learning

**docs/plans/:**
- Purpose: Implementation phase details (Plans A-X complete)
- Generated: No (created during planning phase)
- Committed: Yes
- Reference: Each phase document describes work scope, testing strategy

**.planning/:**
- Purpose: GSD (Guided Structured Development) planning metadata
- Generated: Yes (created by /gsd:map-codebase and /gsd:plan-phase commands)
- Committed: Yes
- Structure: codebase/ (analysis), phase-*/ (implementation details)

**ferroml-core/tests/regression/:**
- Purpose: Per-issue fixtures tracking previously fixed bugs
- Generated: No (static fixtures)
- Committed: Yes
- Pattern: Directory per regression (e.g., regression/issue-42-svc-rbf-convergence/)

---

*Structure analysis: 2026-03-16*
