# Architecture

**Analysis Date:** 2026-03-16

## Pattern Overview

**Overall:** Layered architecture with trait-based polymorphism

**Key Characteristics:**
- Core-shell design: `ferroml-core` library + `ferroml-python` PyO3 bindings
- Trait-based model abstraction enabling diverse algorithm implementations
- Fit/transform separation with consistent API across all transformers
- Zero-copy array handling between Python and Rust where possible
- Comprehensive error types with ML-specific variants (FerroError)
- Statistical diagnostics integrated into model layer (not bolted-on)

## Layers

**Presentation Layer (Python Bindings):**
- Purpose: Python API surface via PyO3
- Location: `ferroml-python/src/`
- Contains: 24 PyO3 module files (linear.rs, trees.rs, preprocessing.rs, etc.)
- Depends on: ferroml-core types, numpy/pyo3
- Used by: End users via `import ferroml`
- Responsibility: Array marshalling, error translation, module registration

**Core Algorithm Layer (Models):**
- Purpose: ML algorithm implementations
- Location: `ferroml-core/src/models/`
- Contains: 25+ model files (linear.rs, svm.rs, tree.rs, forest.rs, etc.)
- Depends on: traits, linalg, preprocessing (for pipelines)
- Used by: Pipeline, AutoML, Python bindings
- Responsibility: Model fitting, prediction, statistical diagnostics

**Feature Engineering Layer (Preprocessing):**
- Purpose: Feature transformers and scaling
- Location: `ferroml-core/src/preprocessing/`
- Contains: scalers, encoders, imputers, selection, sampling, polynomial, power, quantile, tfidf, count_vectorizer
- Depends on: ndarray, sparse matrices (optional)
- Used by: Pipelines, feature engineering workflows
- Responsibility: Fit/transform stateless operations, feature name tracking

**Orchestration Layer (Pipeline & AutoML):**
- Purpose: Composite model building and hyperparameter optimization
- Location: `ferroml-core/src/pipeline/` and `ferroml-core/src/automl/`
- Contains: Pipeline, FeatureUnion, TextPipeline, AutoML
- Depends on: models, preprocessing, CV, HPO, metrics
- Used by: End users building composite workflows
- Responsibility: Step sequencing, search space merging, cross-validation integration

**Statistical Foundation Layer:**
- Purpose: Hypothesis tests, statistical diagnostics, metrics
- Location: `ferroml-core/src/stats/`, `ferroml-core/src/metrics/`, `ferroml-core/src/cv/`
- Contains: Diagnostics (normality, homoscedasticity), confidence intervals, CV splitters, scoring functions
- Depends on: statrs, ndarray
- Used by: Models (for residual analysis), evaluation functions
- Responsibility: Test implementation, assumption validation

**Support Layer:**
- Purpose: Cross-cutting concerns
- Location: `ferroml-core/src/{error, schema, serialization, linalg, simd, sparse, testing}`
- Contains: Error types, input validation, serialization formats, linear algebra wrappers, SIMD ops
- Depends on: thiserror, serde, ndarray, nalgebra
- Used by: All other layers
- Responsibility: Type safety, numerical correctness, data persistence

## Data Flow

**Training Flow:**

1. **Input** → User provides Array2<f64> features (X) and Array1<f64> targets (y)
2. **Validation** → `validate_fit_input()` checks shapes, NaN/Inf values, dimensionality
3. **Model Fitting** → Model trait's `fit(&mut self, x, y)` implementation runs
   - Linear models: QR decomposition via nalgebra
   - Tree models: Recursive splitting via entropy/gini calculation
   - Distance-based: Spatial indexing (KDTree, BallTree) or clustering (KMeans)
4. **Diagnostics** → If statistical model, compute residuals, test assumptions, generate warnings
5. **State Storage** → Store coefficients, thresholds, means/stds in model struct fields
6. **Return** → `Result<()>` with error context if convergence/numerical issues occur

**Prediction Flow:**

1. **Input** → User provides Array2<f64> test features
2. **Validation** → `validate_predict_input()` checks fit status, feature count
3. **Schema Check** (optional) → `FeatureSchema::validate()` against training distribution
4. **Transformation** → If in pipeline, apply all preprocessing steps sequentially
5. **Prediction** → Model trait's `predict(x)` returns Array1<f64> predictions
6. **Uncertainty** (optional) → `predict_with_uncertainty()` returns PredictionWithUncertainty (confidence intervals)
7. **Return** → Array as PyArray via numpy interop in Python layer

**Preprocessing Flow:**

1. **Fit** → `fit(&mut self, x)` learns statistics (mean, std, categories, thresholds) and stores as struct fields
2. **Transform** → `transform(&self, x)` applies fitted parameters stateless
   - Can be called concurrently (Sync trait)
   - Validates input shape against n_features_in()
3. **Fit+Transform** → `fit_transform()` chains fit→transform efficiently
4. **Inverse** (optional) → `inverse_transform()` reverses transformation for reversible ops (scaling, rotation)

**HPO Flow (Hyperparameter Optimization):**

1. **Search Space** → `search_space()` returns SearchSpace with parameter bounds
   - Each model exposes its own search space (n_estimators, learning_rate, etc.)
   - Pipelines merge component search spaces with "__" naming
2. **Iteration** → HPO sampler (Bayesian, random, grid) suggests parameter combinations
3. **Evaluation** → Cross-validation loop:
   - For each fold: fit on train split, score on validation split
   - Collect scores across folds, compute mean+std
   - Return metric value to optimizer
4. **Best Model** → HPO selects model with best cross-validated metric

## State Management

**Model State:**
- Before fit: Option::None for all learned parameters
- After fit: Option::Some with concrete values (coefficients, intercepts, statistics)
- Fitness check: Models implement `is_fitted() -> bool` checking Option state
- Fitted data: Some models cache training statistics for diagnostics (residuals, leverage, VIF)

**Transformer State:**
- Before fit: None, with `is_fitted() = false`
- After fit: Stores learned parameters (means, stds, category maps, thresholds)
- Feature tracking: Optional feature_names stored, n_features_in cached
- Thread-safe: Multiple transform() calls on fitted transformer are concurrent-safe (Sync)

**Pipeline State:**
- Steps are stored as trait objects (Box<dyn PipelineTransformer>, Box<dyn PipelineModel>)
- Each step manages its own fit state independently
- Pipeline.fit() calls fit() on each transformer sequentially, passing transformed output to next
- Pipeline.predict() calls transform() on transformers, then predict() on final model

## Key Abstractions

**Model Trait:**
- Purpose: Unified interface for all supervised learning algorithms
- Examples: `ferroml-core/src/models/linear.rs`, `ferroml-core/src/models/svm.rs`, `ferroml-core/src/models/tree.rs`
- Pattern: fit(&mut self, x, y) → Result<()>, predict(&self, x) → Result<Array1<f64>>
- Implemented by: 50+ model types (LinearRegression, SVC, RandomForestClassifier, etc.)

**Transformer Trait:**
- Purpose: Unified interface for feature transformers
- Examples: `ferroml-core/src/preprocessing/scalers.rs`, `ferroml-core/src/preprocessing/encoders.rs`
- Pattern: fit(&mut self, x) → (), transform(&self, x) → Array2<f64>
- Implemented by: 20+ transformer types (StandardScaler, OneHotEncoder, SimpleImputer, etc.)

**Statistical Model Trait (extending Model):**
- Purpose: Models that provide residual diagnostics and assumption tests
- Examples: `ferroml-core/src/models/linear.rs` implements StatisticalModel
- Pattern: Additional methods for coefficient_std_errors(), assumption_tests(), diagnostics()
- Distinguishes: Models with full R-style statistical output vs. black-box predictions

**Probabilistic Model Trait (extending Model):**
- Purpose: Classifiers that output class probabilities
- Examples: LogisticRegression, RandomForestClassifier, SVC with probability=True
- Pattern: predict_proba(&self, x) → Result<Array2<f64>> (n_samples × n_classes)
- Used by: ROC curves, calibration, soft voting ensembles

**PipelineTransformer & PipelineModel Traits:**
- Purpose: Wrapping trait objects for runtime polymorphism in pipelines
- Pattern: Box<dyn PipelineTransformer>, Box<dyn PipelineModel>
- Contains: search_space(), clone(), n_estimators_fitted() (ensemble models)
- Enables: Heterogeneous step composition, nested hyperparameter access

**CrossValidator Trait:**
- Purpose: CV splitter interface
- Examples: `ferroml-core/src/cv/` implements KFold, StratifiedKFold, TimeSeriesSplit
- Pattern: split(n_samples, groups, times) → Vec<(train_indices, test_indices)>
- Used by: cross_val_score, AutoML evaluation loops

## Entry Points

**Rust Entry Points:**

**ferroml-core/src/lib.rs:**
- Location: Root of algorithm library
- Triggers: `use ferroml_core::models::LinearRegression; let mut m = LinearRegression::new();`
- Responsibilities: Module re-exports, public API surface, top-level types (AutoMLConfig, Task, Metric)

**ferroml-core/src/models/mod.rs:**
- Location: Model module registry
- Triggers: Creating any supervised learning model
- Responsibilities: Trait definitions (Model, StatisticalModel, ProbabilisticModel), model re-exports

**ferroml-core/src/preprocessing/mod.rs:**
- Location: Transformer module registry
- Triggers: Creating any feature transformer
- Responsibilities: Transformer trait, module re-exports, validation helpers (check_is_fitted, check_shape)

**Python Entry Points:**

**ferroml-python/src/lib.rs (pymodule):**
- Location: Root of Python extension
- Triggers: `import ferroml`
- Responsibilities: Module initialization, submodule registration (linear, trees, preprocessing, etc.)

**ferroml-python/src/linear.rs:**
- Location: Linear models Python wrapper
- Triggers: `from ferroml.linear import LinearRegression`
- Responsibilities: PyClass wrapping of Rust models, pyo3 methods (__init__, fit, predict, summary)

**ferroml-python/src/preprocessing.rs:**
- Location: Preprocessing transformers Python wrapper
- Triggers: `from ferroml.preprocessing import StandardScaler`
- Responsibilities: Wrapping transformer types, fit/transform/fit_transform methods, feature name tracking

## Error Handling

**Strategy:** Result<T> type alias wrapping FerroError enum

**Patterns:**

1. **Input Validation Errors:**
   - `FerroError::InvalidInput(String)` - Empty arrays, mismatched dimensions
   - `FerroError::ShapeMismatch { expected, actual }` - Feature count mismatch at predict time
   - Code: `validate_fit_input(x, y)` in models, `check_shape(x, n_features)` in transformers

2. **Statistical Errors:**
   - `FerroError::AssumptionViolation { assumption, test, p_value }` - Normality/homoscedasticity failed
   - `FerroError::NumericalError(String)` - Singular matrix, zero variance, numerical instability
   - Code: Logged in LinearRegression diagnostics, doesn't fail fit (warnings)

3. **Convergence Errors:**
   - `FerroError::ConvergenceFailure { iterations, reason }` - Optimizer didn't converge
   - Code: SVM, LogisticRegression iterative solvers; raised if iterations exhausted

4. **State Errors:**
   - `FerroError::NotFitted { operation }` - predict() called before fit()
   - Code: `check_is_fitted(self.intercept.is_some(), "predict")?` in model.predict()

5. **Not Implemented Errors:**
   - `FerroError::NotImplemented(String)` - Feature not yet in library
   - `FerroError::NotImplementedFor { feature, model }` - Feature for specific model
   - Code: `inverse_transform()` on feature selectors, some model methods

6. **Configuration Errors:**
   - `FerroError::ConfigError(String)` - Invalid hyperparameter combination
   - Code: Solver + penalty incompatibility checks in logistic/SVM

**Error Propagation:**

- Errors bubble up as `Result<T>` through layers
- Python bindings catch Result errors, translate to PyErr (pyo3 automatic via impl From)
- User sees Python exception with error message from .to_string()

**Example Error Flow:**
```
LinearRegression::fit() fails due to singular matrix
  ↓ raises FerroError::NumericalError("Singular matrix in QR decomposition")
  ↓ returns Err(FerroError) from fit()
  ↓ Python binding receives Err, pyo3 converts to PyException
  ↓ User sees: "Exception: Numerical error: Singular matrix in QR decomposition"
```

## Cross-Cutting Concerns

**Logging:** Uses `tracing` crate (integrated but not active by default)
- Models log convergence progress, warn on assumption violations
- No stdout spam by default; requires tracing-subscriber initialization

**Validation:** Defensive checks at layer boundaries
- fit/predict input validation in models
- shape/fit checks in transformers and pipelines
- Feature schema optional validation before prediction

**Serialization:** serde traits + custom Format enum
- Models serialize with bincode (binary), msgpack (compact), JSON (human-readable)
- Preserves fitted state: coefficients, thresholds, learned statistics
- Round-trip: save_model(path) → load_model(path) with version checking

**Numerical Stability:**
- Linear algebra via nalgebra (more stable than ndarray-linalg)
- QR decomposition for LinearRegression (more stable than normal equations)
- Welford's algorithm for mean/variance (numerical stability)
- Epsilon comparisons in comparisons (not direct f64 == checks)

**Thread Safety:**
- All trait objects are Send + Sync
- Models are Send (safe to move between threads)
- Transformers are Sync (same transformer instance can be used in parallel transform() calls)
- Pipelines use rayon for parallel FeatureUnion execution
- HPO uses rayon for parallel fold evaluation

---

*Architecture analysis: 2026-03-16*
