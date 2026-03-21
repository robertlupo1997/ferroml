# Architecture

**Analysis Date:** 2026-03-20

## Pattern Overview

**Overall:** Layered, trait-based ML library with strict separation between core algorithms (Rust) and Python bindings (PyO3).

**Key Characteristics:**
- **Trait-driven design**: All components implement standard traits (`Model`, `Predictor`, `Estimator`, `Transformer`, `Estimator`) for composability
- **Statistical rigor as first-class citizen**: Every model includes uncertainty quantification, assumption testing, and diagnostics built-in
- **Workspace architecture**: Two crates (ferroml-core library + ferroml-python PyO3 bindings) with clear separation of concerns
- **Performance-oriented**: Uses ndarray + nalgebra for numerical computing (not ndarray-linalg to avoid OpenBLAS system dependency), parallel processing with rayon, SIMD via wide crate
- **Correctness-verified**: 5,650+ tests cross-validated against sklearn, scipy, linfa, and statsmodels

## Layers

**Statistics Foundation:**
- Purpose: Provide hypothesis testing, confidence intervals, effect sizes, assumption checking, power analysis
- Location: `ferroml-core/src/stats/`
- Contains: `bootstrap.rs`, `hypothesis.rs`, `diagnostics.rs`, `effect_size.rs`, `multiple_testing.rs`, `confidence.rs`, `power.rs`, `math.rs`
- Depends on: ndarray, statrs, rand
- Used by: All models via `StatisticalModel` trait; CV, metrics, and diagnostics modules

**Linear Algebra Foundation:**
- Purpose: Matrix decomposition, solving, and linear operations (QR, Cholesky, SVD, eigendecomposition)
- Location: `ferroml-core/src/linalg.rs`
- Contains: QR decomposition (native + MGS + faer variants), Cholesky (native + faer), triangular solves, distance computations
- Depends on: nalgebra (replaces ndarray-linalg to avoid OpenBLAS dependency), faer for performance
- Used by: Linear models, decomposition, GP models, SVM solvers

**Core Models & Algorithms:**
- Purpose: Implement ML algorithms with statistical diagnostics
- Location: `ferroml-core/src/models/`
- Contains: 25+ models across linear (Linear, Ridge, Lasso, ElasticNet, LogisticRegression, Regularized), trees (DecisionTree, RandomForest, ExtraTrees, IsolationForest), boosting (GradientBoosting, HistGradientBoosting, AdaBoost), SVM (SVC, SVR, LinearSVC, LinearSVR), neighbors (KNN, NearestCentroid, LOF), Gaussian process (GPC, GPR), QDA, naive Bayes (Gaussian, Multinomial, Bernoulli, Categorical), clustering (KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture, HDBSCAN), decomposition (PCA, t-SNE, TruncatedSVD, LDA, FactorAnalysis), calibration (TemperatureScaling, IsotonicCalibrator, SigmoidCalibrator)
- Depends on: stats, linalg, metrics, preprocessing (for data validation)
- Used by: Pipeline, AutoML, Python bindings, cross-validation

**Preprocessing & Feature Engineering:**
- Purpose: Transform raw features into ML-ready data
- Location: `ferroml-core/src/preprocessing/`
- Contains: Scalers (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer), Encoders (OneHotEncoder, OrdinalEncoder, TargetEncoder, BinaryEncoder), Imputers (SimpleImputer, KNNImputer), Selectors (SelectKBest, SelectPercentile, VarianceThreshold, RecursiveFeatureElimination), Transformers (PolynomialFeatures, PowerTransformer, QuantileTransformer), Samplers (SMOTE, ADASYN, RandomUnderSampler, RandomOverSampler, TomekLinks), Vectorizers (CountVectorizer, TFIDFVectorizer)
- Depends on: linalg, stats, models (for some selectors)
- Used by: Pipeline, AutoML, direct user code

**Metrics & Evaluation:**
- Purpose: Compute performance metrics and evaluation curves
- Location: `ferroml-core/src/metrics/`
- Contains: Classification metrics (accuracy, precision, recall, F1, AUC-ROC, AUC-PR, Matthews correlation, confusion matrix), regression metrics (MSE, RMSE, MAE, R², MAPE, SMAPE, MASE), clustering metrics (silhouette, Calinski-Harabasz, Davies-Bouldin, homogeneity, completeness, V-measure), multilabel metrics
- Depends on: stats (for some tests)
- Used by: CV, AutoML, model evaluation, model selection

**Cross-Validation Framework:**
- Purpose: Split data and evaluate models with statistical guarantees
- Location: `ferroml-core/src/cv/`
- Contains: Splitters (KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, LeaveOneOut, ShuffleSplit, StratifiedGroupKFold), Searchers (GridSearchCV, RandomizedSearchCV), Curves (learning_curve, validation_curve), Nested CV
- Depends on: metrics, models, stats (for bootstrap CIs)
- Used by: AutoML, model selection, hyperparameter optimization

**Pipeline & Composition:**
- Purpose: Chain transformers and models into reusable workflows
- Location: `ferroml-core/src/pipeline/`
- Contains: `Pipeline` (sequential steps), `FeatureUnion` (parallel feature extraction), `ColumnTransformer` (column-specific transformations), `TextPipeline` (for text + sparse features with feature="sparse")
- Depends on: models, preprocessing, cv
- Used by: AutoML, user workflows

**Hyperparameter Optimization:**
- Purpose: Tune model hyperparameters and perform algorithm selection
- Location: `ferroml-core/src/hpo/`
- Contains: Search space definition (`SearchSpace`, `ParameterValue`), optimization algorithms (Bayesian optimization, ASHA, Hyperband), sampler (grid, random, TPE)
- Depends on: models, metrics, cv, stats
- Used by: AutoML, CV search

**AutoML Orchestration:**
- Purpose: End-to-end machine learning automation with statistical rigor
- Location: `ferroml-core/src/automl/`
- Contains: `AutoML` orchestrator, `AutoMLConfig`, algorithm portfolios, ensemble selection, result aggregation, feature importance aggregation, pairwise model comparison
- Depends on: All other modules (models, preprocessing, pipeline, cv, metrics, stats, hpo)
- Used by: User's top-level API

**Schema Validation:**
- Purpose: Validate input data against declared feature types and ranges
- Location: `ferroml-core/src/schema.rs`
- Contains: `FeatureSchema`, `FeatureSpec`, `ValidationResult`, validation modes (Strict/Warn/Permissive)
- Depends on: error handling
- Used by: Preprocessing, models, pipeline (optional)

**Serialization & Persistence:**
- Purpose: Save/load models in multiple formats with metadata and versioning
- Location: `ferroml-core/src/serialization.rs`
- Contains: Multiple format support (JSON, JSON Pretty, MessagePack, Bincode), metadata tracking, streaming I/O, version checking
- Depends on: serde, serde_json, bincode, rmp-serde
- Used by: User model persistence, AutoML result storage

**ONNX Export & Inference:**
- Purpose: Export models to ONNX format and perform inference
- Location: `ferroml-core/src/onnx/` and `ferroml-core/src/inference/`
- Contains: ONNX serialization, session management, tensor conversion
- Depends on: models, prost (for protobuf)
- Used by: Cross-platform deployment, production inference

**Explainability & Diagnostics:**
- Purpose: Interpret model predictions and understand decision boundaries
- Location: `ferroml-core/src/explainability/`
- Contains: TreeSHAP, KernelSHAP, permutation importance, partial dependence plots, individual conditional expectation (ICE), residual diagnostics
- Depends on: models, stats
- Used by: Model analysis, debugging

**GPU Acceleration (optional feature="gpu"):**
- Purpose: Accelerate specific operations via GPU shaders
- Location: `ferroml-core/src/gpu/`
- Contains: GPU dispatcher, shader compilation, memory management
- Depends on: GPU libraries (feature-gated)
- Used by: Specific models when feature enabled

**Sparse Matrix Support (optional feature="sparse"):**
- Purpose: Efficient computation on sparse data (text, high-dimensional)
- Location: `ferroml-core/src/sparse/`
- Contains: CSR/CSC matrix wrappers, sparse transformers, sparse models
- Depends on: sprs
- Used by: Text pipeline, sparse models

## Data Flow

**Training Pipeline:**

1. **Input**: User provides `X: Array2<f64>` (n_samples × n_features), `y: Array1<f64>` (n_samples)
2. **Validation**: `FeatureSchema` validates input shape, types, ranges (optional)
3. **Preprocessing**: Pipeline applies transformers in sequence:
   - Transformer.fit_transform(X) → transformed X
   - Each step receives output of previous step
4. **Model Fitting**: Last step (Model) receives preprocessed data
   - Model.fit(&transformed_X, &y) → fitted Model
5. **Cross-Validation** (if enabled): CV splitter generates train/test indices
   - For each fold: fit on train indices, evaluate on test indices
   - Metrics computed for each fold
   - Stats module computes bootstrap CIs across folds
6. **Output**: Fitted model ready for prediction

**Prediction Pipeline:**

1. **Input**: User provides X (same schema as training)
2. **Validation**: Schema validation (if enabled)
3. **Preprocessing**: Transformers applied in same order as training:
   - Transformer.transform(X) → transformed X
   - Uses fitted parameters from training
4. **Model Prediction**:
   - Model.predict(&transformed_X) → Array1<f64>
   - Optional: Model.predict_with_uncertainty(...) → Array1 + CIs
5. **Output**: Predictions with optional confidence intervals

**Hyperparameter Optimization Loop (GridSearchCV/RandomizedSearchCV):**

1. **Setup**: Define parameter grid, CV strategy, scoring metric
2. **Enumeration**: Generate all param combinations (grid) or sample (random)
3. **Iteration**: For each param combination:
   - Create new model with params
   - Run CV: fit on train fold → predict on test fold → compute metric
   - Store metric scores across folds
4. **Aggregation**: Stats module computes mean ± CI across folds
5. **Selection**: Choose params with best mean metric
6. **Output**: Best model, best params, all results with CIs

**AutoML Orchestration Loop:**

1. **Configuration**: User specifies task, metric, time budget, CV strategy, etc.
2. **Algorithm Portfolio**: AutoML selects subset of algorithms to try (Bayesian/Bandit/Uniform)
3. **Preprocessing Search**: Try different preprocessing pipelines
4. **Model Search**: For each (preprocessor, model, params) tuple:
   - Create pipeline: preprocessor → model
   - Run CV to evaluate
   - Log results (metric ± CI)
5. **Ensemble**: Top N models combined into meta-learner
6. **Statistical Testing**: Compare best model vs others via paired t-test
7. **Output**: AutoMLResult with best model, leaderboard, comparisons, diagnostics

**State Management:**

- **Unfitted state**: Model created but not trained (fit() not called)
- **Fitted state**: Model has learned parameters; predict() calls blocked until fit()
- **NotFitted error**: Enforced at runtime; design enables serialization of both states
- **Immutability where possible**: Transformers and models preserve input data (no mutations)

## Key Abstractions

**Model Hierarchy:**

```
Model (base trait)
├── Predictor: predict(&X) → Array1<f64>
├── Estimator: fit(&X, &y) → fitted Model + search_space()
├── StatisticalModel: coefficients, std_errors, CIs, diagnostics
├── ProbabilisticModel: predict_proba(), log_proba()
├── TreeModel: feature_importances, tree_depths
├── LinearModel: coefficients, intercept
├── IncrementalModel: partial_fit()
├── WeightedModel: fit_weighted(X, y, sample_weight)
├── WarmStartModel: warm start for ensemble
└── OutlierDetector: score_samples, decision_function (for anomaly detection)
```

**Transformer Hierarchy:**

```
Transformer (base trait)
├── transform(&X) → Array2<f64>
├── fit_transform(&mut self, &X) → Array2<f64>
├── inverse_transform(&X) → Array2<f64> (optional)
└── PipelineTransformer: extends for pipeline integration
```

**Search Space:**

All models expose `search_space() → SearchSpace` defining hyperparameter ranges:
- `SearchSpace` contains `HashMap<String, ParameterValue>` for each param
- `ParameterValue` variants: Categorical, IntRange, FloatRange
- Used by GridSearchCV, RandomizedSearchCV, AutoML for exploration

**Error Handling:**

`FerroError` enum with variants:
- `InvalidInput`: Schema validation, shape mismatches
- `ShapeMismatch`: X and y dimensions don't align
- `NotFitted`: Called predict/transform before fit
- `ConvergenceFailure`: Optimization didn't converge
- `AssumptionViolation`: Statistical test failed
- `NumericalError`: NaN/Inf in computation
- `ConfigError`: Invalid configuration
- `SerializationError`: Save/load issues

## Entry Points

**Rust Library (`ferroml-core/src/lib.rs`):**
- Main re-exports: AutoML, AutoMLConfig, traits, error types
- Modules public: models, preprocessing, cv, metrics, stats, hpo, pipeline, automl, etc.
- Direct usage: `let model = ferroml_core::models::LinearRegression::new()`

**Python Package (`ferroml-python/python/ferroml/__init__.py`):**
- Imports native extension: `from ferroml import ferroml as _native`
- Re-exports submodules: `from ferroml import linear, trees, ensemble, ...`
- Usage: `from ferroml.linear import LinearRegression`

**PyO3 Bindings (`ferroml-python/src/lib.rs`):**
- Registers all submodules via `#[pymodule]`
- Each submodule has `register_*_module(m: &PyModule)` function
- Array conversion via `array_utils` module (zero-copy input, owned output)
- Error translation in `errors` module

## Cross-Cutting Concerns

**Logging:**
- Framework: `tracing` crate (optional, feature-gated)
- Pattern: Used in AutoML, long-running processes, debugging
- File: No dedicated logging module; integrated into modules

**Validation:**
- **Input validation**: `FeatureSchema` validates X before fit/predict
- **Parameter validation**: Model::new() builder checks parameter ranges
- **Output validation**: Metrics validate class distributions (for classification)

**Reproducibility:**
- **Random seeds**: Most stochastic components accept `Option<u64>` seed
- **Default seeding**: Uses `rand` crate with unseeded RNG if seed not provided
- **Parallelization**: rayon used for data-parallel operations (deterministic)

**Serialization:**
- **Format flexibility**: JSON, JSON Pretty, MessagePack, Bincode
- **Metadata tracking**: Version, model type, timestamp, custom description
- **Streaming I/O**: For large models, use `StreamingWriter`/`StreamingReader`

**Performance Optimization:**
- **Zero-copy arrays**: ndarray views passed by reference; only copies on ownership transfer
- **Parallelization**: rayon for CV folds, ensemble predictions, some preprocessing
- **Memory mapping**: memmap2 for loading large datasets
- **SIMD**: wide crate for portable SIMD operations
- **Linear algebra**: nalgebra (not ndarray-linalg) to avoid OpenBLAS dependency

---

*Architecture analysis: 2026-03-20*
