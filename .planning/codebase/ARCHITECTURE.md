# Architecture

**Analysis Date:** 2026-03-15

## Pattern Overview

**Overall:** Layered trait-based architecture with clear separation of concerns across statistical modeling, transformation, hyperparameter optimization, and cross-validation layers.

**Key Characteristics:**
- **Trait-driven design**: Core abstractions (Model, Transformer, ClusteringModel, etc.) define contracts for all implementations
- **Fit/Transform pattern**: Stateful learning phase followed by deterministic transformation/prediction phase
- **Statistical rigor first**: Every model includes diagnostics, uncertainty quantification, and assumption testing
- **Pipeline composability**: Transformers and models chain through a unified interface
- **Zero-copy Python integration**: PyO3 bindings with optimized array handling between Python and Rust
- **Multi-fidelity optimization**: Hyperparameter optimization supports Bayesian, grid, random, ASHA, and BOHB strategies

## Layers

**Core Traits (foundational contracts):**
- Location: `ferroml-core/src/models/traits.rs`, `ferroml-core/src/preprocessing/mod.rs`, `ferroml-core/src/clustering/mod.rs`
- Purpose: Define interfaces for models, transformers, and clustering algorithms
- Contains: `Model` (fit/predict), `StatisticalModel` (diagnostics), `ProbabilisticModel` (probabilities), `Transformer` (fit/transform), `ClusteringModel` (fit/predict labels)
- Depends on: ndarray, serde
- Used by: All algorithm implementations, pipelines, cross-validation, hyperparameter optimization

**Models Layer:**
- Location: `ferroml-core/src/models/`
- Purpose: Machine learning algorithms with full statistical diagnostics
- Contains:
  - Linear models (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, Perceptron)
  - Tree-based (DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost)
  - Probabilistic (GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, GaussianProcess, QDA)
  - Support Vector Machines (SVC, SVR, LinearSVC, LinearSVR)
  - Ensemble (SGD, PassiveAggressive, MultiOutput wrappers)
  - Specialized (KNeighbors, IsotonicRegression, QuantileRegression, RobustRegression)
- Depends on: Core traits, stats module, metrics, linalg
- Used by: Pipelines, cross-validation, AutoML, Python bindings

**Preprocessing Layer:**
- Location: `ferroml-core/src/preprocessing/`
- Purpose: Feature transformations with statistically rigorous handling
- Contains:
  - Scalers (StandardScaler, MinMaxScaler, RobustScaler, Normalizer)
  - Encoders (OneHotEncoder, OrdinalEncoder, TargetEncoder, BinaryEncoder)
  - Imputers (SimpleImputer, KNNImputer, IterativeImputer)
  - Feature selection (VarianceThreshold, SelectKBest, RFE, RFECV)
  - Power transforms (BoxCox, YeoJohnson)
  - Polynomial features, discretizers, quantile transformers
  - Text processors (CountVectorizer, TfidfVectorizer - sparse feature)
- Depends on: Core traits, sparse module (optional)
- Used by: Pipelines, feature engineering workflows, Python bindings

**Clustering Layer:**
- Location: `ferroml-core/src/clustering/`
- Purpose: Clustering algorithms with statistical quality assessment
- Contains: KMeans, DBSCAN, Agglomerative, HDBSCAN, GaussianMixture, diagnostics, metrics
- Depends on: Core traits, metrics
- Used by: AutoML, Python bindings

**Dimensionality Reduction Layer:**
- Location: `ferroml-core/src/decomposition/`
- Purpose: Feature extraction and dimensionality reduction
- Contains: PCA, IncrementalPCA, TruncatedSVD, NMF, LDA, FactorAnalysis, TSNE, UMAP
- Depends on: Core traits, linalg
- Used by: Pipelines, feature engineering, Python bindings

**Pipeline Layer:**
- Location: `ferroml-core/src/pipeline/`
- Purpose: Composition of transformers and estimators with unified fit/predict interface
- Contains:
  - `Pipeline`: Sequential chaining of transformers + final estimator
  - `FeatureUnion`: Parallel feature extraction with concatenation
  - `ColumnTransformer`: Column-specific transformations
  - `TextPipeline` (sparse): Text-specific preprocessing pipeline
- Depends on: Models, preprocessing, core traits
- Used by: AutoML, end-user workflows, Python bindings

**Cross-Validation Layer:**
- Location: `ferroml-core/src/cv/`
- Purpose: Statistical validation strategies with reproducibility
- Contains:
  - Splitters (KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, LeaveOneOut, ShuffleSplit)
  - Search methods (GridSearchCV, RandomizedSearchCV)
  - Curves (learning_curve, validation_curve)
  - Nested CV for unbiased performance estimation
- Depends on: Models, metrics
- Used by: Model evaluation, hyperparameter optimization selection

**Hyperparameter Optimization Layer:**
- Location: `ferroml-core/src/hpo/`
- Purpose: Statistical hyperparameter optimization with uncertainty quantification
- Contains:
  - `SearchSpace`: Parameter definitions
  - `BayesianOptimizer`: Gaussian Process-based optimization
  - `GridSampler`, `RandomSampler`, `TPESampler`: Parameter samplers
  - `HyperbandScheduler`, `ASHAScheduler`, `BOHBScheduler`: Multi-fidelity schedulers
- Depends on: Models, core traits
- Used by: AutoML, model hyperparameter tuning, Python bindings

**Statistical Testing Layer:**
- Location: `ferroml-core/src/stats/`
- Purpose: Hypothesis testing and statistical diagnostics
- Contains: Normality tests (Shapiro-Wilk), hypothesis tests, confidence intervals, diagnostics
- Depends on: distributions (statrs), ndarray
- Used by: Models, preprocessing, validation workflows

**Metrics Layer:**
- Location: `ferroml-core/src/metrics/`
- Purpose: Performance evaluation and model comparison
- Contains:
  - Classification metrics (accuracy, precision, recall, F1, ROC-AUC, log-loss)
  - Regression metrics (MSE, MAE, R², RMSE)
  - Probabilistic metrics (log-loss, Brier score)
  - Comparison utilities (paired tests, bootstrap confidence intervals)
- Depends on: Core traits, stats
- Used by: Cross-validation, AutoML, model evaluation, Python bindings

**Explainability Layer:**
- Location: `ferroml-core/src/explainability/`
- Purpose: Model interpretability tools
- Contains: TreeSHAP, permutation importance, partial dependence, ALE plots
- Depends on: Models, preprocessing
- Used by: Model analysis, Python bindings

**AutoML Layer:**
- Location: `ferroml-core/src/automl/`
- Purpose: Automated machine learning with statistical rigor
- Contains: Algorithm selection, ensemble construction, feature importance aggregation, statistical testing
- Depends on: All model/preprocessing layers, CV, HPO, metrics
- Used by: High-level API, Python bindings

**Serialization Layer:**
- Location: `ferroml-core/src/serialization/`
- Purpose: Model persistence with metadata tracking
- Contains: Binary serialization (bincode), JSON export, ONNX export (feature-gated)
- Depends on: Models, serde
- Used by: Model persistence, inference, Python pickling

**Python Bindings Layer:**
- Location: `ferroml-python/src/`
- Purpose: PyO3 bindings exposing FerroML to Python ecosystem
- Contains: 14 submodules mirroring Rust structure (linear, trees, ensemble, etc.)
- Depends on: ferroml-core, PyO3, numpy, polars (optional)
- Used by: Python users via `import ferroml`

## Data Flow

**Training Flow (fit → fit_transform → predict):**

1. **Input validation**: Data enters through model/transformer `fit()` method
   - Shape validation: `validate_fit_input()` ensures (n_samples, n_features)
   - Type checking: Array bounds and dtype verification
   - Feature name tracking: Optional feature names propagated through pipeline

2. **Fit phase (Parameter Learning):**
   - Transformer learns statistics: means, stds, encodings, etc. (stored in struct fields)
   - Model estimates parameters: coefficients, tree splits, cluster centers, etc.
   - Diagnostics calculated: residuals, covariance matrices, assumption tests
   - State transitions from unfitted → fitted (checked by `check_is_fitted()`)

3. **Transform/Prediction phase (Deterministic Application):**
   - Transformer applies learned parameters: centering, scaling, encoding
   - Model uses fitted parameters to predict: dot product (linear), tree traversal, etc.
   - Output validation: predictions checked for NaN/inf (hardened in Plan V)
   - Shape preservation: (n_test, n_features_in) → (n_test, n_features_out) or (n_test,)

4. **Pipeline Flow** (composite operations):
   ```
   Input Data (X, y)
        ↓
   Transformer 1.fit(X) → stores params
        ↓
   X_transformed = Transformer 1.transform(X)
        ↓
   Transformer 2.fit(X_transformed) → stores params
        ↓
   X_transformed2 = Transformer 2.transform(X_transformed)
        ↓
   Model.fit(X_transformed2, y) → stores coefficients, diagnostics
        ↓
   Predictions = Model.predict(new_X → T1.transform → T2.transform)
   ```

5. **Cross-Validation Flow:**
   - CV splitter generates k (train_idx, test_idx) pairs
   - For each fold:
     - Model.fit(X[train_idx], y[train_idx])
     - Predictions = Model.predict(X[test_idx])
     - Metric(y[test_idx], predictions) → fold score
   - Aggregate fold scores → mean, std, CI via bootstrap

6. **Hyperparameter Optimization Flow:**
   - Sampler generates hyperparameter values from SearchSpace
   - Trial: CV evaluates model with params → fold scores → aggregated metric
   - Optimizer observes metric, updates belief (Bayesian), suggests next params
   - Repeat until budget exhausted or convergence

**State Management:**

- **Unfitted**: Created via `Model::new()`, no parameters stored
- **In-Progress**: `fit()` called, learnable parameters being calculated
- **Fitted**: All parameters stored (coefficients, centers, encodings), ready for prediction
- **Prediction**: Deterministic application of learned parameters

State is verified by `check_is_fitted()` which panics if predict called before fit.

## Key Abstractions

**Model Trait Family:**
- Location: `ferroml-core/src/models/traits.rs`
- Purpose: Define core contract for trainable algorithms
- Examples: `LinearRegression`, `RandomForest`, `KMeans`, all implement `Model` or `ClusteringModel`
- Pattern: `fit(&mut self, X, y) → Result<()>` mutates self with learned params; `predict(&self, X) → Result<Array1<f64>>` applies them

**Transformer Trait:**
- Location: `ferroml-core/src/preprocessing/mod.rs`
- Purpose: Define contract for feature transformations
- Examples: `StandardScaler`, `OneHotEncoder`, all implement `Transformer`
- Pattern: `fit(&mut self, X) → Result<()>` learns params (no y needed); `transform(&self, X) → Array2<f64>` applies

**SearchSpace:**
- Location: `ferroml-core/src/hpo/search_space.rs`
- Purpose: Parameterize hyperparameter search domain
- Examples: `SearchSpace::new().add_int("n_estimators", 10, 500)`
- Pattern: Every `Model` exposes `search_space()` for HPO integration

**CVFold:**
- Location: `ferroml-core/src/cv/mod.rs`
- Purpose: Represent single train/test split
- Contains: `train_indices: Vec<usize>`, `test_indices: Vec<usize>`
- Pattern: CV splitter generates vector of CVFold, executor iterates them

**FerroError Enum:**
- Location: `ferroml-core/src/error.rs`
- Purpose: Structured error handling with statistical context
- Variants: `ShapeMismatch`, `AssumptionViolation`, `ConvergenceFailure`, `NotFitted`, `NumericalError`
- Pattern: All operations return `Result<T> = std::result::Result<T, FerroError>`

**Pipeline:**
- Location: `ferroml-core/src/pipeline/mod.rs`
- Purpose: Compose transformers and models into workflows
- Contains: `steps: Vec<(String, PipelineStep)>` (named, ordered)
- Pattern: Acts as proxy `Model` (implements fit/predict) by delegating to steps

**ModelContainer:**
- Location: `ferroml-core/src/serialization/mod.rs`
- Purpose: Wrap models with metadata for persistence
- Contains: Model bytes, feature names, schema, version, creation date
- Pattern: Enables model export/import with reproducibility tracking

## Entry Points

**Rust API Entry Points:**

**Library root (`ferroml-core/src/lib.rs`):**
- Location: `ferroml-core/src/lib.rs`
- Triggers: `use ferroml_core::{models::..., preprocessing::..., cv::..., hpo::..., metrics::...}`
- Responsibilities: Re-exports all public modules, defines core traits, defines PredictionWithUncertainty

**AutoML High-Level API (`ferroml-core/src/automl/mod.rs`):**
- Location: `ferroml-core/src/automl/mod.rs`
- Triggers: `automl.fit(X, y, cv=5)` in Python or direct instantiation in Rust
- Responsibilities: Algorithm selection, ensemble construction, cross-validation orchestration, statistical testing

**Individual Model Constructors:**
- Location: Each model file (e.g., `ferroml-core/src/models/linear.rs`)
- Triggers: `LinearRegression::new()`, `RandomForest::new()`
- Responsibilities: Default initialization, builder pattern chain setup

**Python Entry Point:**

**PyO3 Module (`ferroml-python/src/lib.rs`):**
- Location: `ferroml-python/src/lib.rs`, function `ferroml(m: &Bound<'_, PyModule>)`
- Triggers: `import ferroml` in Python
- Responsibilities: Register all 14 submodules, expose `__version__`, initialize PyO3 runtime
- Creates: Python modules ferroml.linear, ferroml.trees, ferroml.ensemble, ferroml.clustering, etc.

**Submodule Registration:**
- Location: Each `ferroml-python/src/{module_name}.rs` file
- Triggers: Called by `ferroml()` via `register_{module}_module(m)?`
- Responsibilities: Create Python classes, bind methods with PyO3 decorators, register with parent module

## Error Handling

**Strategy:** Fail-fast with structured, actionable errors. All operations return `Result<T>`.

**Patterns:**

1. **Input Validation Errors:**
   ```rust
   // In Model::fit()
   validate_fit_input(x, y)?;  // Raises ShapeMismatch if x.nrows() != y.len()
   validate_predict_input(x)?;  // Raises NotFitted if model.coefficients.is_none()
   ```

2. **Numerical Stability Errors:**
   ```rust
   // Hardened output validation (Plan V)
   if prediction.iter().any(|p| p.is_nan() || p.is_infinite()) {
       return Err(FerroError::NumericalError("NaN in predictions".into()));
   }
   ```

3. **Statistical Assumption Violations:**
   ```rust
   // LinearRegression::diagnostics()
   if shapiro_wilk_p < 0.05 {
       return Err(FerroError::assumption_violation(
           "Normality of residuals",
           "Shapiro-Wilk",
           shapiro_wilk_p
       ));
   }
   ```

4. **Convergence Failures:**
   ```rust
   // LogisticRegression::fit()
   if !converged && iterations >= max_iterations {
       return Err(FerroError::convergence_failure(iterations, "Loss did not decrease"));
   }
   ```

5. **Not Fitted Check:**
   ```rust
   // Before any prediction
   check_is_fitted(self.coefficients.as_ref(), "fit")?;
   ```

**Python Error Translation:**
- Location: `ferroml-python/src/errors.rs`
- Rust `FerroError::*` variants mapped to Python exceptions via `PyErr::from()`

## Cross-Cutting Concerns

**Logging:**
- Framework: tracing crate for structured logging
- Implementation: Models log convergence progress, CV logs fold completion
- Not heavily used yet; available for debugging

**Validation:**
- Centralized: `ferroml-core/src/schema.rs` defines `ValidationMode`, `ValidationIssue`
- Applied to: Input data, feature names, model parameters
- Returns: `ValidationResult` with list of issues and severity levels

**Authentication (N/A):**
- Not applicable; library is self-contained

**Thread Safety:**
- All trait objects require `Send + Sync`
- Data parallelization via rayon (e.g., `par_iter()` in random forests)
- CV and HPO use rayon for fold/trial parallel execution
- No mutable shared state across threads

**Reproducibility:**
- All models accept optional `seed` parameter for deterministic behavior
- CV splitters accept `random_state` for reproducible fold generation
- HPO trials tracked with IDs for result reproduction
- Serialization preserves all state via `ModelContainer` with metadata

---

*Architecture analysis: 2026-03-15*
