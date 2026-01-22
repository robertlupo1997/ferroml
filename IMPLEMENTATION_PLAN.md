# FerroML Implementation Plan

> **Vision**: The greatest ML library - combining sklearn's completeness, statsmodels' rigor, and Rust's performance.
>
> This file is the shared state for the Ralph loop. Update after each task.

## Current Phase: Phase 12 - CI/CD & Release

**Last Updated**: TASK-006 fully completed - Added remaining model comparison tests:
- Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
- 5x2cv paired t-test (Dietterich, 1998) for classifier comparison
- All comparison tests now in `ferroml-core/src/metrics/comparison.rs`

Previously: TASK-137 completed - GitHub Release creation with artifacts:
- Created `.github/workflows/release.yml` for automated GitHub Release creation
  - Triggers on version tags (v*) and manual workflow_dispatch with tag input
  - Generates changelog using git-cliff for release notes
  - Builds release artifacts:
    - Rust library files (rlib, static lib)
    - Rust crate package (.crate file)
    - Python wheels for Linux (x86_64), macOS (arm64), Windows (x86_64)
  - Creates GitHub Release with softprops/action-gh-release@v2
  - Auto-detects prereleases from version tags (-alpha, -beta, -rc)
  - Generates SHA256 checksums for all artifacts
  - Includes installation instructions for PyPI and crates.io
  - Supports draft release creation via workflow_dispatch

### Phase 12 Complete!
All CI/CD & Release tasks have been completed:
- [x] TASK-130: GitHub Actions: cargo check, clippy, fmt
- [x] TASK-131: GitHub Actions: cargo test on Linux, macOS, Windows
- [x] TASK-132: GitHub Actions: Python binding tests
- [x] TASK-133: Code coverage with codecov.io
- [x] TASK-134: Automated crates.io publishing on tag
- [x] TASK-135: Automated PyPI wheel building via maturin
- [x] TASK-136: Changelog generation from commits
- [x] TASK-137: GitHub Release creation with artifacts

FerroML v0.1.0 is ready for release!

---

## Implementation Status Summary

### ✅ Implemented
- **Statistics Module** (`ferroml-core/src/stats/`): Hypothesis tests, CIs, effect sizes, multiple testing, power analysis, bootstrap
- **HPO Core** (`ferroml-core/src/hpo/`): Search space, Random/Grid/TPE samplers, MedianPruner/Hyperband/ASHA schedulers, Study management
- **Core Traits**: Predictor, Estimator, Transformer in `lib.rs`
- **AutoML Config**: Basic struct (no fit method yet)
- **Metrics Module** (`ferroml-core/src/metrics/`):
  - Core trait structure (Metric, ProbabilisticMetric, MetricValue, MetricValueWithCI)
  - Classification: Accuracy, Balanced Accuracy, Precision, Recall, F1, Confusion Matrix
  - Correlation-based: Matthews Correlation Coefficient (MCC), Cohen's Kappa
  - Regression: MSE, RMSE, MAE, R², Explained Variance, Max Error, Median Absolute Error, MAPE
  - Probabilistic: ROC-AUC (with CI), PR-AUC, Average Precision, Log Loss, Brier Score
  - Comparison: Paired t-test, Corrected Resampled t-test, McNemar's test, Wilcoxon signed-rank test, 5x2cv paired t-test

- **CV Module** (`ferroml-core/src/cv/`):
  - CrossValidator trait (core abstraction for all CV strategies)
  - CVFold, CVFoldResult, CVResult structs with statistical rigor
  - t-distribution confidence intervals for CV scores
  - CVConfig for execution configuration
  - Utility functions for stratification, shuffling, group handling
  - KFold and RepeatedKFold implementations
  - StratifiedKFold and RepeatedStratifiedKFold (preserves class distribution)
  - TimeSeriesSplit (temporal CV with expanding/sliding window, gap support)
  - GroupKFold (ensures groups never split between train/test)
  - StratifiedGroupKFold (groups + class distribution preservation)
  - LeaveOneOut (each sample as test set, n folds for n samples)
  - LeavePOut (each combination of p samples as test set)
  - ShuffleSplit (random train/test splits with configurable sizes)
  - cross_val_score, cross_val_score_simple, cross_val_score_array (parallel execution via rayon, CIs)
  - NestedCV (outer loop for evaluation, inner loop for HPO, prevents data leakage)
  - learning_curve (performance vs training set size, with CIs)
  - validation_curve (performance vs hyperparameter value, with CIs)

- **Preprocessing Module** (`ferroml-core/src/preprocessing/`):
  - Transformer trait (fit, transform, fit_transform, inverse_transform, is_fitted, get_feature_names_out)
  - Validation utilities (check_is_fitted, check_shape, check_non_empty)
  - Column-wise statistics (mean, std, min, max, median, quantile)
  - Welford's algorithm for numerically stable variance computation
  - UnknownCategoryHandling enum for encoders
  - FitStatistics struct for diagnostics
  - Scalers: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
  - Power transforms: PowerTransformer (Box-Cox, Yeo-Johnson with MLE lambda optimization)
  - Quantile transforms: QuantileTransformer (uniform, normal distribution via empirical CDF)
  - Categorical encoders: OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder (with smoothing and CV-based leakage prevention)
  - Imputation: SimpleImputer (mean, median, mode, constant strategies with missing value indicator support), KNNImputer (k-nearest neighbors imputation with uniform/distance weights, euclidean/manhattan metrics)
  - Feature expansion: PolynomialFeatures (polynomial and interaction features with configurable degree, interaction_only, include_bias)
  - Discretization: KBinsDiscretizer (uniform, quantile, kmeans binning strategies with ordinal and one-hot encoding)
  - Feature selection: VarianceThreshold (variance-based filtering), SelectKBest (f_classif, f_regression, chi2 scoring), SelectFromModel (importance-based selection with Mean/Median/Value/MeanPlusStd thresholds), RecursiveFeatureElimination (RFE with FeatureImportanceEstimator trait, ClosureEstimator, configurable step size, feature rankings)
  - Sampling: SMOTE (k-NN interpolation, multiple strategies), BorderlineSMOTE (borderline sample focus), ADASYN (adaptive density-based sampling), RandomUnderSampler (random majority removal), RandomOverSampler (random minority duplication with optional shrinkage), SMOTETomek (SMOTE + Tomek links cleaning), SMOTEENN (SMOTE + ENN cleaning), Resampler trait

- **Models Module** (`ferroml-core/src/models/`):
  - Core traits: Model, StatisticalModel, ProbabilisticModel
  - Supporting structs: ModelSummary, CoefficientInfo, FitStatistics, Diagnostics, ResidualStatistics
  - Assumption testing: Assumption enum, AssumptionTestResult
  - Influential observation detection: InfluentialObservation with Cook's distance, leverage, DFFITS
  - Prediction intervals: PredictionInterval with coverage calculation
  - Validation utilities: validate_fit_input, validate_predict_input, check_is_fitted
  - LinearRegression: OLS via QR decomposition, coefficient inference (t-stats, p-values, CIs), R²/adjusted R²/F-statistic, VIF, residual diagnostics (normality, Durbin-Watson), Cook's distance, leverage, prediction intervals
  - LogisticRegression: IRLS fitting, coefficient inference (z-stats, p-values, CIs), odds ratios with CIs, pseudo R², likelihood ratio test, deviance/null deviance, AIC/BIC, Pearson/deviance residuals, ROC-AUC with bootstrap CI, L2 regularization
  - RidgeRegression: L2 regularization via closed-form solution (Cholesky), effective degrees of freedom, coefficient standard errors, prediction intervals
  - LassoRegression: L1 regularization via coordinate descent, sparse solutions, soft thresholding, warm start support
  - ElasticNet: L1+L2 combined regularization, configurable l1_ratio
  - RidgeCV, LassoCV, ElasticNetCV: Cross-validated alpha (and l1_ratio) selection
  - lasso_path, elastic_net_path: Regularization path computation for visualization
  - QuantileRegression: IRLS fitting with pinball loss, bootstrap inference for CIs, multiple quantile estimation (0.25, 0.5, 0.75 etc.), pseudo R² (Koenker-Machado), robust to outliers and heteroscedasticity
  - RobustRegression: Multiple M-estimators (Huber, Bisquare/Tukey, Hampel, Andrew's Wave), IRLS fitting, MAD-based scale estimation, asymptotic covariance-based coefficient inference, breakdown point and efficiency information
  - GaussianNB: Gaussian Naive Bayes classifier with class prior estimation, Gaussian likelihood, partial_fit for incremental learning, variance smoothing
  - MultinomialNB: Multinomial Naive Bayes for discrete count features (e.g., word counts), Laplace/Lidstone smoothing, partial_fit, fit_prior option
  - BernoulliNB: Bernoulli Naive Bayes for binary features, configurable binarization threshold, partial_fit, explicit absence penalty
  - KNeighborsClassifier: K-nearest neighbors classification with uniform/distance weighting, Euclidean/Manhattan/Minkowski metrics, KD-Tree/Ball Tree optimization, predict_proba
  - KNeighborsRegressor: K-nearest neighbors regression with uniform/distance weighting, Euclidean/Manhattan/Minkowski metrics, KD-Tree/Ball Tree optimization
  - SVC: Support Vector Classification with Linear/RBF/Polynomial/Sigmoid kernels, SMO algorithm, Platt scaling probability estimates, One-vs-One/One-vs-Rest multiclass, class weights
  - SVR: Support Vector Regression with Linear/RBF/Polynomial/Sigmoid kernels, epsilon-insensitive loss, coordinate descent optimization, sample weights
  - LinearSVC: Fast linear SVM classification with primal coordinate descent, O(n·d) memory, Hinge/SquaredHinge loss, multiclass via OvR, class weights, feature importance
  - LinearSVR: Fast linear SVM regression with primal coordinate descent, O(n·d) memory, EpsilonInsensitive/SquaredEpsilonInsensitive loss, feature importance
  - DecisionTreeClassifier: CART classification with Gini/Entropy criteria, feature importance, cost-complexity pruning, tree visualization (DOT format)
  - DecisionTreeRegressor: CART regression with MSE/MAE criteria, feature importance, cost-complexity pruning, tree visualization (DOT format)
  - RandomForestClassifier: Bootstrap aggregating, random feature subsampling (sqrt/log2/fraction), OOB error estimation, feature importance with bootstrap CIs, parallel tree building via rayon
  - RandomForestRegressor: Bootstrap aggregating, random feature subsampling, OOB error estimation, feature importance with bootstrap CIs, parallel tree building via rayon
  - GradientBoostingClassifier: Deviance loss, learning rate scheduling (constant/linear/exponential decay), early stopping, stochastic gradient boosting, feature importance, binary/multiclass support
  - GradientBoostingRegressor: SquaredError/AbsoluteError/Huber loss, learning rate scheduling, early stopping, stochastic gradient boosting, feature importance, staged predictions
  - HistGradientBoostingClassifier: Histogram-based split finding (O(n)), native missing value handling, monotonic constraints, feature interaction constraints, leaf-wise growth, L1/L2 regularization, early stopping
  - HistGradientBoostingRegressor: Multiple loss functions (SquaredError/AbsoluteError/Huber), histogram-based split finding (O(n)), native missing value handling, monotonic constraints, feature interaction constraints, leaf-wise growth, L1/L2 regularization, early stopping

- **Decomposition Module** (`ferroml-core/src/decomposition/`):
  - PCA: Full and randomized SVD, explained variance ratios, component loadings, whitening, covariance reconstruction
  - IncrementalPCA: Memory-efficient batch processing with partial_fit(), running mean/variance updates
  - TruncatedSVD: Sparse matrix support (no centering), randomized SVD algorithm, configurable power iterations, reproducible results
  - LDA: Supervised dimensionality reduction, SVD and Eigen solvers, maximizes class separability, between-class/within-class scatter, shrinkage regularization, classification capabilities (predict, predict_proba), explained variance ratios
  - FactorAnalysis: EM algorithm MLE estimation, factor loadings and communalities, explained variance per factor, noise variance estimation, rotation methods (Varimax, Quartimax, Promax/oblique), factor correlation matrix, Bartlett factor scores

- **Ensemble Module** (`ferroml-core/src/ensemble/`):
  - VotingClassifier: Hard voting (majority class), Soft voting (probability averaging), weighted voting, named estimators, feature importance aggregation
  - VotingRegressor: Simple averaging, weighted averaging, individual predictions access, named estimators, feature importance aggregation
  - StackingClassifier: CV-based meta-feature generation, configurable meta-learner (default: RidgeRegression), passthrough option, soft/hard stacking methods, named estimators
  - StackingRegressor: CV-based meta-feature generation, configurable meta-learner (default: RidgeRegression), passthrough option, individual predictions, named estimators
  - BaggingClassifier: Bootstrap aggregating for classifiers, OOB error estimation, feature/sample subsampling, parallel training via rayon, warm start support
  - BaggingRegressor: Bootstrap aggregating for regressors, OOB R² estimation, feature/sample subsampling, parallel training via rayon, individual predictions access

- **Pipeline Module** (`ferroml-core/src/pipeline/`):
  - Pipeline: Named steps, fit/transform/predict sequencing, memory caching, combined search space for HPO, set_params with "step__param" syntax
  - PipelineTransformer trait: Extends Transformer with search_space, clone_boxed, set_param
  - PipelineModel trait: Extends Model with clone_boxed, set_param
  - PipelineCache: Hash-based cache validation, invalidation on parameter changes
  - FeatureUnion: Parallel feature extraction with concatenation, named transformers, weights support, combined search space, parallel execution via rayon
  - ColumnTransformer: Apply different transformers to different column subsets, column selection (indices, mask, all), remainder handling (drop, passthrough), combined search space, parallel execution via rayon

- **Calibration Module** (`ferroml-core/src/models/calibration.rs`):
  - CalibratedClassifierCV: CV-based probability calibration wrapper, prevents data leakage
  - SigmoidCalibrator: Platt scaling with Newton-Raphson optimization
  - IsotonicCalibrator: PAVA algorithm with monotonic interpolation
  - TemperatureScalingCalibrator: Single-parameter calibration for multi-class (divides logits by learned temperature T), gradient descent optimization, preserves accuracy while improving calibration
  - MulticlassCalibrator trait: Multi-class calibration interface for 2D probability arrays
  - CalibrationResult: ECE (Expected Calibration Error), MCE, Brier score
  - calibration_curve: Reliability diagram data generation
  - CalibrableClassifier trait: Implemented for LogisticRegression, GaussianNB, MultinomialNB, BernoulliNB, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, KNeighborsClassifier

- **Explainability Module** (`ferroml-core/src/explainability/`):
  - permutation_importance: Model-agnostic feature importance via permutation shuffling
  - permutation_importance_parallel: Parallel version using rayon for large datasets
  - PermutationImportanceResult: Mean importance, std, CI per feature, sorted_indices(), top_k(), is_significant()
  - Configurable n_repeats, random_state, confidence_level
  - Works with any Model trait and custom scoring functions
  - partial_dependence: 1D PDP showing marginal feature effects
  - partial_dependence_2d: 2D PDP for feature interactions
  - partial_dependence_parallel, partial_dependence_2d_parallel: Parallel versions via rayon
  - partial_dependence_multi, partial_dependence_multi_parallel: Compute PDPs for multiple features
  - PDPResult: Grid values, PDP values, std (heterogeneity measure), optional ICE curves, monotonicity checks, effect range
  - PDP2DResult: 2D grid values and PDP matrix for feature interactions
  - GridMethod: Percentile (default, handles non-uniform distributions) or Uniform spacing
  - individual_conditional_expectation: Per-sample feature effect curves (ICE)
  - individual_conditional_expectation_parallel: Parallel ICE via rayon
  - ICEConfig: Builder pattern for n_grid_points, grid_method, centering, derivatives, sample subset
  - ICEResult: Raw ICE curves, centered ICE (c-ICE), derivative ICE (d-ICE), PDP values, heterogeneity analysis
  - center_ice_curves: Center ICE at reference point for easier comparison
  - compute_derivative_ice: Compute d-ICE for non-linear effect detection
  - ice_from_curves: Convert raw curves to full ICEResult with analysis
  - ice_multi, ice_multi_parallel: Compute ICE for multiple features
  - h_statistic, h_statistic_parallel: Friedman's H-statistic for pairwise feature interaction strength
  - h_statistic_matrix, h_statistic_matrix_parallel: Compute H² for all feature pairs
  - h_statistic_overall: Total interaction strength for a single feature
  - HStatisticConfig: Builder pattern for n_grid_points, grid_method, bootstrap CI, permutation p-value, max_samples
  - HStatisticResult: H² value, CI, std_error, p_value, interpretation (No/Weak/Moderate/Strong/Very strong)
  - HStatisticMatrix: Pairwise H² matrix with top_k() for strongest interactions
  - TreeExplainer: TreeSHAP for tree-based models (DecisionTree*, RandomForest*, GradientBoosting*)
  - explain(), explain_batch(), explain_batch_parallel(): Compute SHAP values for samples
  - SHAPResult: base_value, shap_values, prediction reconstruction, sorted_indices(), top_k(), summary()
  - SHAPBatchResult: Batch SHAP values with mean_abs_shap() global importance, global_importance_sorted()
  - KernelExplainer: KernelSHAP for model-agnostic SHAP (works with any Model trait)
  - KernelSHAPConfig: Builder pattern for n_samples, max_background_samples, random_state, regularization, paired_sampling
  - Weighted linear regression approximation using SHAP kernel weights
  - Background data for reference distribution, Cholesky solver with regularized fallback
  - SHAPSummary: Comprehensive SHAP visualization data aggregation
  - FeatureSHAPStats: Per-feature statistics (mean, std, percentiles, correlation)
  - BarPlotData/BarPlotEntry: Global importance with CIs, cumulative importance
  - BeeswarmPlotData/BeeswarmFeatureData/BeeswarmPoint: Detailed beeswarm visualization data
  - DependencePlotData: Scatter plot data with interaction feature support
  - find_best_interaction(): Automatic interaction detection

- **AutoML Module** (`ferroml-core/src/automl/`):
  - AlgorithmPortfolio: Manages collections of algorithms with presets (Quick/Balanced/Thorough)
  - AlgorithmConfig: Pairs algorithms with search spaces and preprocessing requirements
  - AlgorithmType: 23 supported algorithms (10 classifiers, 13 regressors)
  - AlgorithmComplexity: Low/Medium/High complexity classification for time budget allocation
  - PreprocessingRequirement: Scaling, HandleMissing, EncodeCategorical, NonNegative, DimensionalityReduction
  - DataCharacteristics: Data-aware configuration (n_samples, n_features, n_classes, missing values, variance ratio, imbalance)
  - PortfolioPreset: Quick (4 algorithms), Balanced (8 algorithms), Thorough (10+ algorithms)
  - Adaptive search space generation based on dataset characteristics
  - Combined search space with algorithm selection for CASH optimization
  - TimeBudgetAllocator: Bandit-based time budget allocation (UCB1, Thompson Sampling, Epsilon-Greedy, Successive Halving)
  - TimeBudgetConfig: Configuration for total budget, strategy, warmup trials, early stopping threshold
  - AlgorithmArm: Per-algorithm statistics tracking (trials, scores, time, success/failure counts)
  - ArmSelection: Selection results with trial budget and selection reason
  - Early stopping for underperforming algorithms based on performance threshold
  - Complexity-weighted budget allocation respecting algorithm time factors
  - PreprocessingSelector: Data-aware automatic preprocessing selection
  - PreprocessingConfig: Builder pattern configuration (strategy, scaler type, imputation, encoding)
  - PreprocessingStrategy: Auto, Conservative, Standard, Thorough, Passthrough strategies
  - PreprocessingPipelineSpec: Ordered list of preprocessing steps with parameters
  - DetectedCharacteristics: Data analysis (missing values, variance, skewness, class distribution)
  - Automatic imputation (SimpleImputer, KNNImputer), scaling, encoding selection
  - Class imbalance handling (SMOTE, RandomUnderSampler) with configurable threshold
  - Dimensionality reduction for high-dimensional data (PCA)
  - Selection reasons documentation for explainability
  - EnsembleBuilder: Greedy ensemble selection from trial results (auto-sklearn style)
  - TrialResult: Stores AutoML trial results with OOF predictions for ensemble construction
  - EnsembleConfig: Configuration for max models, selection iterations, diversity weighting, weight optimization
  - EnsembleMember/EnsembleResult: Selected models with weights, improvement metrics, score history
  - Diversity-based selection using algorithm type distribution
  - Coordinate descent weight optimization for ensemble members
  - DatasetMetafeatures: Comprehensive dataset characterization for meta-learning
  - SimpleMetafeatures: n_samples, n_features, n_classes, dimensionality ratio, imbalance ratio, missing ratio
  - StatisticalMetafeatures: mean, std, skewness, kurtosis aggregations, outlier ratio, correlation statistics
  - InformationMetafeatures: target entropy, feature entropies, mutual information, noise-to-signal ratio
  - LandmarkingMetafeatures: 1-NN, decision stump, Naive Bayes, linear model scores for meta-learning
  - MetafeatureConfig: Configuration for landmarking CV folds and subsampling
  - Dataset similarity computation for warm-starting from similar datasets
  - MetaLearningStore: In-memory store for dataset metafeatures and configurations
  - DatasetRecord: Stores metafeatures, task type, and best configurations with timestamps
  - ConfigurationRecord: Stores algorithm type, params, score, rank for each configuration
  - WarmStartConfig: Configuration for k_nearest, min_similarity, max_configs, normalization
  - WarmStartResult: Similar datasets with weighted configurations for HPO initialization
  - WeightedConfiguration: Configuration with similarity-based priority weights
  - Cosine similarity with optional metafeature normalization, JSON serialization, store merging
  - AutoMLResult: Comprehensive result object with leaderboard, ensemble, feature importance, model comparisons
  - AggregatedFeatureImportance: Weighted feature importance across models with CIs
  - ModelComparisonResults: Statistical significance tests using corrected t-test with Holm-Bonferroni correction
  - PairwiseComparison: Detailed comparison results between model pairs

- **Python Bindings** (`ferroml-python/src/`):
  - Linear models submodule (`ferroml.linear`): LinearRegression, LogisticRegression, RidgeRegression, LassoRegression, ElasticNet
  - Tree models submodule (`ferroml.trees`): DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor, HistGradientBoostingClassifier/Regressor
  - Preprocessing submodule (`ferroml.preprocessing`): StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, SimpleImputer
  - Pipeline submodule (`ferroml.pipeline`): Pipeline, ColumnTransformer, FeatureUnion
  - AutoML submodule (`ferroml.automl`): AutoMLConfig, AutoML, AutoMLResult, LeaderboardEntry, EnsembleResult, EnsembleMember
  - NumPy array support via pyo3/numpy crate with zero-copy optimization
  - array_utils module: centralized utilities for input conversion (to_owned_array), read-only views (as_array_view), and output transfer (into_pyarray)
  - sklearn-compatible API (fit, predict, coef_, intercept_)
  - Full statistical methods exposed (summary, coefficients_with_ci, predict_interval, odds_ratios, etc.)
  - Python pickle support via __getstate__/__setstate__ using MessagePack serialization

- **Serialization Module** (`ferroml-core/src/serialization.rs`):
  - Format enum: Json, JsonPretty, MessagePack, Bincode
  - SerializationMetadata: Version, model type, timestamp, description
  - ModelContainer wrapper for versioned serialization
  - save_model/load_model file I/O with auto-format detection
  - to_bytes/from_bytes for in-memory serialization
  - ModelSerialize extension trait for all serializable models
  - Magic bytes (FRML) for bincode format validation

- **Inference Module** (`ferroml-core/src/inference/`):
  - InferenceSession: Load and execute ONNX models without Python
  - Tensor: Multi-dimensional f32 array with shape, strides, matmul, add, sigmoid, softmax
  - TensorI64: Multi-dimensional i64 array for integer outputs (labels)
  - Value enum: Tensor, TensorI64, SequenceMapI64F32 for different output types
  - Standard ONNX operators: MatMul, Add, Squeeze, Sigmoid, Softmax, Flatten, Reshape
  - ONNX-ML operators: TreeEnsembleRegressor, TreeEnsembleClassifier
  - Supports all FerroML-exported models (linear, logistic, ridge, lasso, elastic net, decision trees, random forests)

- **Schema Module** (`ferroml-core/src/schema.rs`):
  - FeatureSchema: Expected input structure (n_features, feature specs, validation mode)
  - FeatureSpec: Per-feature specification (type, missing policy, range, categories)
  - FeatureType: Continuous, Integer, Categorical, Binary, Unknown
  - ValidationMode: Strict (all fail), Warn (log issues), Permissive (shape only)
  - ValidationIssue: ShapeMismatch, MissingValue, ValueBelowMin/AboveMax, UnknownCategory, InvalidBinary, NonInteger
  - ValidationResult: Pass/fail with issues, issue_counts(), severity filtering
  - SchemaValidated trait: Models can expose expected schema
  - from_array() for automatic schema inference from training data

- **Datasets Module** (`ferroml-core/src/datasets/`):
  - Dataset struct: Features (X) and targets (y) with optional metadata
  - DatasetInfo: Dataset metadata (name, description, task type, feature/target names, source, license)
  - DatasetLoader trait: Interface for custom dataset loaders
  - LoadOptions: Configuration for shuffling, random state, feature/sample selection
  - train_test_split: Split dataset into train/test with shuffle support
  - Task inference: Automatically detect classification vs regression
  - Synthetic data generators: make_classification, make_regression, make_blobs, make_moons, make_circles
  - Dataset statistics: describe() for feature-wise statistics (mean, std, min, max, median, missing)
  - Built-in toy datasets: load_iris (150×4, 3 classes), load_wine (178×13, 3 classes), load_diabetes (260×10, regression), load_linnerud (20×3, regression)
  - File loading: load_csv, load_parquet, load_file (auto-detect format)
  - CsvOptions: Delimiter, header, skip_rows, n_rows, column selection, null values, encoding
  - ParquetOptions: Column selection, parallel reading configuration

- **Sparse Module** (`ferroml-core/src/sparse.rs`, feature-gated):
  - CsrMatrix: Compressed Sparse Row format with row slicing, transpose, matrix-vector ops
  - CscMatrix: Compressed Sparse Column format for column operations
  - SparseVector: Sparse 1D vector, SparseRowView: Zero-copy row view
  - Sparse distance calculations: euclidean, manhattan, cosine (O(nnz) complexity)
  - sparse_pairwise_distances: Batch distance with SparseDistanceMetric enum
  - Matrix ops: normalize_rows_l2, column_sums/means, row_norms, scale_rows
  - Utilities: sparse_eye, sparse_diag, sparse_vstack, sparse_hstack
  - SparseMatrixInfo: Memory analysis, sparsity metrics, storage recommendations
  - Uses sprs crate, 23 comprehensive tests

### 🔲 Not Implemented (Stubs Only)
- (None in current phase)

---

## Phase 1: Foundation (Core Infrastructure)

### 1.1 Metrics Module (CRITICAL - Required for CV and Model Evaluation)
- [x] **TASK-001**: Create `metrics/mod.rs` module structure with `Metric` trait
- [x] **TASK-002**: Implement classification metrics
  - Accuracy, Precision, Recall, F1 (micro/macro/weighted)
  - Confusion matrix with derived metrics
  - Balanced Accuracy (for imbalanced data)
- [x] **TASK-003**: Implement probabilistic classification metrics
  - ROC-AUC with CI (using bootstrap from stats module)
  - PR-AUC, Average Precision
  - Log Loss, Brier score (calibration)
- [x] **TASK-004**: Implement correlation-based classification metrics
  - Matthews Correlation Coefficient
  - Cohen's Kappa
- [x] **TASK-005**: Implement regression metrics
  - MSE, RMSE, MAE, MAPE, R², Adjusted R²
  - Explained variance, Max error, Median absolute error
- [x] **TASK-006**: Implement model comparison statistics
  - Paired t-test for CV scores
  - Corrected resampled t-test (Nadeau & Bengio)
  - McNemar's test for classifier comparison
  - Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
  - 5x2cv paired t-test (Dietterich, 1998) for classifier comparison

### 1.2 Cross-Validation Module
- [x] **TASK-007**: Implement `CrossValidator` trait in `cv/mod.rs`
- [x] **TASK-008**: Implement `KFold`, `RepeatedKFold`
- [x] **TASK-009**: Implement `StratifiedKFold`, `RepeatedStratifiedKFold`
- [x] **TASK-010**: Implement `TimeSeriesSplit`
- [x] **TASK-011**: Implement `GroupKFold`, `StratifiedGroupKFold`
- [x] **TASK-012**: Implement `LeaveOneOut`, `LeavePOut`, `ShuffleSplit`
- [x] **TASK-013**: Implement `cross_val_score` with CIs and parallel execution (rayon)
- [x] **TASK-014**: Implement `NestedCV` (proper model selection)
- [x] **TASK-015**: Implement `learning_curve` and `validation_curve` data generation

### 1.3 Preprocessing Module
- [x] **TASK-016**: Implement `Transformer` trait in `preprocessing/mod.rs`
- [x] **TASK-017**: Implement `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler`
- [x] **TASK-018**: Implement `PowerTransformer` (Box-Cox, Yeo-Johnson)
- [x] **TASK-019**: Implement `QuantileTransformer`
- [x] **TASK-020**: Implement `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`
- [x] **TASK-021**: Implement `TargetEncoder` (with smoothing)
- [x] **TASK-022**: Implement `SimpleImputer` (mean, median, mode, constant)
- [x] **TASK-023**: Implement `KNNImputer`
- [x] **TASK-024**: Implement `PolynomialFeatures` (interactions, powers)
- [x] **TASK-025**: Implement `KBinsDiscretizer` (uniform, quantile, kmeans)
- [x] **TASK-026**: Implement feature selection: `VarianceThreshold`, `SelectKBest`, `SelectFromModel`
- [x] **TASK-027**: Implement `RecursiveFeatureElimination` (RFE)

### 1.4 Model Traits & Linear Models
- [x] **TASK-028**: Implement `Model`, `StatisticalModel`, `ProbabilisticModel` traits in `models/mod.rs`
- [x] **TASK-029**: Implement `LinearRegression` with full diagnostics
  - OLS fitting via QR decomposition
  - Coefficient standard errors, t-statistics, p-values
  - R², adjusted R², F-statistic
  - VIF (Variance Inflation Factor) for multicollinearity
  - Residual diagnostics (Shapiro-Wilk, Breusch-Pagan, Durbin-Watson)
  - Cook's distance, leverage (hat values)
  - Prediction intervals and confidence bands
- [x] **TASK-030**: Implement `LogisticRegression`
  - Maximum likelihood via IRLS
  - Odds ratios with CIs
  - Wald tests, likelihood ratio test
  - ROC-AUC with CI (uses metrics module)
- [x] **TASK-031**: Implement `RidgeRegression`, `LassoRegression`, `ElasticNet`
  - Regularization path visualization data (lasso_path, elastic_net_path)
  - Cross-validated alpha selection (RidgeCV, LassoCV, ElasticNetCV)
- [x] **TASK-032**: Implement `QuantileRegression` (statsmodels parity)
  - IRLS fitting with asymmetric pinball loss
  - Bootstrap-based coefficient inference (standard errors, confidence intervals)
  - Support for multiple quantile estimation (fit_quantiles)
  - Pseudo R² (Koenker-Machado, 1999)
- [x] **TASK-033**: Implement `RobustRegression` (Huber, RLM with M-estimators)
  - IRLS fitting with robust M-estimator weight functions
  - Multiple M-estimators: Huber, Bisquare (Tukey), Hampel, Andrew's Wave
  - MAD-based robust scale estimation
  - Asymptotic covariance-based coefficient inference (standard errors, CIs, z-tests)
  - Breakdown point and efficiency information for each estimator
  - Pseudo R-squared for model fit assessment

---

## Phase 2: Core ML Algorithms

### 2.1 Probabilistic Models
- [x] **TASK-034**: Implement `GaussianNB` (Gaussian Naive Bayes)
- [x] **TASK-035**: Implement `MultinomialNB`, `BernoulliNB`
- [x] **TASK-036**: Implement `KNeighborsClassifier`, `KNeighborsRegressor`
  - Distance metrics: euclidean, manhattan, minkowski
  - Weighted voting (uniform, distance-based)
  - Ball tree / KD tree optimization

### 2.2 Support Vector Machines
- [x] **TASK-037**: Implement `SVC` (Support Vector Classification)
  - Kernels: linear, rbf, poly, sigmoid
  - Probability estimates via Platt scaling
  - Multiclass support: One-vs-One, One-vs-Rest
  - Class weights: uniform, balanced, custom
- [x] **TASK-038**: Implement `SVR` (Support Vector Regression)
  - Reuses kernel infrastructure from SVC (Linear, RBF, Polynomial, Sigmoid)
  - Epsilon-insensitive loss function with coordinate descent optimization
  - Regularization parameter C, epsilon tube width parameter
  - Sample weights support, intercept computation for free/bounded SVs
- [x] **TASK-039**: Implement `LinearSVC`, `LinearSVR` (faster for large datasets)
  - Primal coordinate descent optimization (no kernel matrix needed)
  - O(n·d) memory vs O(n²) for kernelized versions
  - LinearSVC: Hinge and SquaredHinge loss, multiclass via OvR, class weights
  - LinearSVR: EpsilonInsensitive and SquaredEpsilonInsensitive loss
  - Feature importance from absolute weight magnitudes

### 2.3 Tree Models
- [x] **TASK-040**: Implement `DecisionTreeClassifier`, `DecisionTreeRegressor` (CART)
  - Feature importance (Gini, information gain)
  - Tree visualization/export data
  - Cost-complexity pruning
- [x] **TASK-041**: Implement `RandomForestClassifier`, `RandomForestRegressor`
  - OOB (Out-of-Bag) error estimation
  - Feature importance with CI (via bootstrap)
  - Parallel tree building via rayon
- [x] **TASK-042**: Implement `GradientBoostingClassifier`, `GradientBoostingRegressor`
  - Multiple loss functions (SquaredError, AbsoluteError, Huber for regression; Deviance for classification)
  - Learning rate scheduling (constant, linear decay, exponential decay)
  - Early stopping with validation set
  - Stochastic gradient boosting via subsample
  - Feature importance from impurity decrease
  - Staged predictions for learning curves

### 2.4 Histogram-Based Gradient Boosting (LightGBM-style) - KEY DIFFERENTIATOR
- [x] **TASK-043**: Implement `HistGradientBoostingClassifier`
  - Histogram-based split finding (O(n) complexity via bin mapper)
  - Native missing value handling (learns optimal direction)
  - **Monotonic constraints** (domain knowledge enforcement)
  - **Feature interaction constraints** (control allowed interactions)
  - Leaf-wise growth option (LightGBM-style best-first growth)
  - L1/L2 regularization
- [x] **TASK-044**: Implement `HistGradientBoostingRegressor`
  - Multiple loss functions (SquaredError, AbsoluteError, Huber)
  - Reuses histogram infrastructure from HistGradientBoostingClassifier
  - Native missing value handling
  - Monotonic constraints and feature interaction constraints
  - Leaf-wise growth strategy
  - L1/L2 regularization
  - Early stopping with validation set
- [x] **TASK-045**: Implement native categorical feature handling (CatBoost-style)
  - Ordered target encoding to prevent leakage (CatBoost-style)
  - CategoricalFeatureHandler for ordered target encoding
  - CategoricalBinMapper for integrated binning of categorical + continuous features
  - BinMapperInfo trait for unified histogram building
  - Integration with HistGradientBoostingClassifier and HistGradientBoostingRegressor
  - with_categorical_features() and with_categorical_smoothing() builder methods

### 2.5 Dimensionality Reduction
- [x] **TASK-046**: Implement `PCA` (Principal Component Analysis)
  - Explained variance ratios with cumulative variance calculation
  - Component loadings with interpretation (scaled by sqrt of variance)
  - Incremental PCA for large datasets with partial_fit()
  - Whitening option for uncorrelated unit-variance output
  - Multiple SVD solvers (Full, Randomized auto-selection)
  - Noise variance estimation when components < features
  - Covariance matrix reconstruction
- [x] **TASK-047**: Implement `TruncatedSVD` (for sparse matrices)
  - Randomized SVD algorithm (Halko et al. 2011)
  - No data centering (preserves sparsity)
  - Configurable power iterations for accuracy
  - Reproducible via random_state parameter
- [x] **TASK-048**: Implement `LDA` (Linear Discriminant Analysis)
- [x] **TASK-049**: Implement `FactorAnalysis`

---

## Phase 3: Ensemble & Pipeline

### 3.1 Ensemble Methods
- [x] **TASK-050**: Implement `VotingClassifier`, `VotingRegressor`
  - Hard and soft voting
  - Weighted voting
- [x] **TASK-051**: Implement `StackingClassifier`, `StackingRegressor`
  - Configurable meta-learner (default: RidgeRegression)
  - CV-based stacking (prevents data leakage)
  - Passthrough option (include original features)
  - Soft/hard stacking methods for classifiers
- [x] **TASK-052**: Implement `BaggingClassifier`, `BaggingRegressor`
  - OOB samples for validation
  - Feature subsampling
  - Sample subsampling support
  - Parallel training via rayon
  - Warm start support

### 3.2 Pipeline Module
- [x] **TASK-053**: Implement `Pipeline` struct
  - Named steps with get/set
  - Caching of intermediate transformations
  - Combined search space for HPO
  - set_params with "step__param" syntax
- [x] **TASK-054**: Implement combined search space for HPO (included in TASK-053)
- [x] **TASK-055**: Implement `FeatureUnion` (parallel feature extraction)
  - Named transformers with add_transformer() builder pattern
  - Parallel fitting and transforming via rayon
  - Horizontal concatenation of transformed features
  - Combined search space from all transformers
  - set_params with "transformer__param" syntax
  - Optional weights for transformer outputs
  - Can be used as a PipelineTransformer in Pipeline
- [x] **TASK-056**: Implement `ColumnTransformer` (column-wise preprocessing)

---

## Phase 4: Probability & Class Imbalance

### 4.1 Probability Calibration - CRITICAL FOR PRODUCTION
- [x] **TASK-057**: Implement `CalibratedClassifierCV`
  - CV-based probability calibration wrapper
  - SigmoidCalibrator (Platt scaling) with Newton-Raphson optimization
  - IsotonicCalibrator (PAVA algorithm with linear interpolation)
  - CalibrationResult with ECE, MCE, Brier score metrics
  - calibration_curve function for reliability diagrams
  - CalibrableClassifier trait and implementations for all classifiers
- [x] **TASK-058**: Implement Platt scaling (sigmoid calibration) (included in TASK-057)
- [x] **TASK-059**: Implement isotonic regression calibration (included in TASK-057)
- [x] **TASK-060**: Implement temperature scaling (multi-class)
  - TemperatureScalingCalibrator with gradient descent optimization
  - MulticlassCalibrator trait for 2D probability arrays
  - Works with binary (via Calibrator trait) and multi-class classifiers
  - Preserves accuracy (argmax unchanged) while improving calibration
  - Integration with CalibratedClassifierCV
- [x] **TASK-061**: Implement calibration curve data generation (reliability diagrams) (included in TASK-057)

### 4.2 Class Imbalance Handling
- [x] **TASK-062**: Add `class_weight` parameter to all classifiers
  - 'balanced' mode
  - Custom weights dictionary
- [x] **TASK-063**: Implement `SMOTE` (Synthetic Minority Oversampling)
  - Resampler trait for oversampling/undersampling techniques
  - SMOTE: k-NN based synthetic sample generation via interpolation
  - BorderlineSMOTE: Focus on borderline minority samples
  - Multiple sampling strategies (Auto, Ratio, TargetCounts, Classes)
  - Reproducible via random_state parameter
- [x] **TASK-064**: Implement `ADASYN` (Adaptive Synthetic Sampling)
  - Adaptive density-based weighting (more synthetics for harder minority instances)
  - Configurable imbalance threshold to skip balanced classes
  - Density ratio diagnostics for analysis
  - k-NN based synthetic generation via interpolation
- [x] **TASK-065**: Implement `RandomUnderSampler`, `RandomOverSampler`
  - RandomUnderSampler: Randomly remove samples from majority classes to match minority
  - RandomOverSampler: Randomly duplicate samples from minority classes to match majority
  - Support for multiple sampling strategies (Auto, Ratio, TargetCounts, Classes)
  - random_state for reproducibility
  - RandomOverSampler includes optional shrinkage for noise injection
  - Diagnostics: sample_indices, n_samples_removed/added
- [x] **TASK-066**: Implement `SMOTE-Tomek`, `SMOTE-ENN` (combined sampling)
  - SMOTETomek: SMOTE oversampling + Tomek links cleaning (removes majority samples forming Tomek links)
  - SMOTEENN: SMOTE oversampling + Edited Nearest Neighbors cleaning (removes samples misclassified by k-NN)
  - ENNKind enum for ENN strategy (All: remove if all neighbors differ, Mode: remove if majority differ)
  - Diagnostics: n_tomek_links, n_samples_removed, n_synthetic_samples

---

## Phase 5: Explainability & Interpretability

### 5.1 Feature Importance
- [x] **TASK-067**: Implement `permutation_importance` (model-agnostic)
  - With confidence intervals via repeated shuffling
  - Created `explainability` module with sequential and parallel versions
  - `PermutationImportanceResult` struct with mean, std, CI per feature
  - Supports any Model trait with custom scoring functions
- [x] **TASK-068**: Implement Partial Dependence Plot (PDP) data generation
  - 1D and 2D PDP support with parallel versions via rayon
  - `PDPResult` with grid values, PDP values, std (heterogeneity), ICE curves option
  - `PDP2DResult` for feature interactions
  - GridMethod enum (Percentile/Uniform), monotonicity checks, effect range
  - Multi-feature support via `partial_dependence_multi()` and parallel version
- [x] **TASK-069**: Implement Individual Conditional Expectation (ICE) data
- [x] **TASK-070**: Implement feature interaction detection (H-statistic)

### 5.2 SHAP Integration
- [x] **TASK-071**: Implement TreeSHAP for tree-based models
- [x] **TASK-072**: Implement KernelSHAP hooks for model-agnostic explanations
- [x] **TASK-073**: Implement SHAP summary plot data generation

---

## Phase 6: Hyperparameter Optimization (Completion)

### 6.1 Bayesian Optimization Completion
- [x] **TASK-074**: Implement Gaussian Process regression
  - RBF, Matern52, Matern32 kernels with length_scale/variance parameters
  - Hyperparameter optimization via marginal log-likelihood grid search
  - Cholesky decomposition for numerical stability
  - Input normalization and target standardization
  - Posterior mean and variance prediction
  - Integration with BayesianOptimizer.suggest()
- [x] **TASK-075**: Implement acquisition functions (EI, PI, UCB, LCB) - included in TASK-074
- [x] **TASK-076**: Implement acquisition optimization (L-BFGS-B)
  - L-BFGS-B algorithm with two-loop recursion and projected gradient bounds
  - Multi-start optimization with configurable restarts
  - Armijo backtracking line search
  - Acquisition function gradients (EI, PI, UCB, LCB)
  - GP gradient computation (dμ/dx, dσ/dx) via kernel gradients
  - AcquisitionOptimizer with gradient and random search modes
  - Integration with BayesianOptimizer via with_lbfgsb() builder

### 6.2 Multi-Fidelity Optimization
- [x] **TASK-077**: Enhance Hyperband implementation
  - Proper bracket management with successive halving
  - FidelityParameter support (discrete: epochs, continuous: data_fraction)
  - EarlyStoppingCallback trait for pruning events
  - HyperbandMetrics and RungMetrics for performance collection
  - Trial-to-bracket assignment and value reporting
  - MedianPruner percentile threshold option
- [x] **TASK-078**: Implement BOHB (Bayesian Optimization + Hyperband)

---

## Phase 7: AutoML Orchestration

### 7.1 Core AutoML
- [x] **TASK-079**: Implement algorithm portfolio with presets
- [x] **TASK-080**: Implement time budget allocation (bandit-based)
- [x] **TASK-081**: Implement automatic preprocessing selection
- [x] **TASK-082**: Implement ensemble construction from trials
- [x] **TASK-083**: Implement `AutoML.fit()` end-to-end

### 7.2 Meta-Learning (Auto-sklearn style)
- [x] **TASK-084**: Implement dataset metafeature extraction
  - Statistical: mean, std, skewness, kurtosis
  - Information-theoretic: entropy, mutual information
  - Landmarking: performance of simple models
- [x] **TASK-085**: Implement warm-starting from similar datasets
  - MetaLearningStore: In-memory store for dataset metafeatures and configurations
  - DatasetRecord: Stores metafeatures, task type, and best configurations
  - ConfigurationRecord: Stores algorithm type, params, score, rank
  - WarmStartConfig: Configuration for k_nearest, min_similarity, max_configs
  - WarmStartResult: Similar datasets with weighted configurations
  - Similarity computation via cosine similarity with optional normalization
  - Serialization support via serde (to_json/from_json)
  - Store merging, statistics, task filtering
  - 16 comprehensive tests covering all functionality
- [x] **TASK-086**: Implement configuration space transfer
  - TransferConfig: Configuration for shrink_factor, adapt_bounds, confidence_level
  - PriorKnowledge: Extracts priors from warm-start configurations
  - ParameterPrior: Normal, LogNormal, Categorical, Boolean priors
  - TransferredSearchSpace: Adapts search space bounds based on prior knowledge
  - WarmStartSampler: Sampler that uses warm-start configs as initial trials
  - initialize_study_with_warmstart: Pre-populate study with warm-start results
  - algorithm_priorities_from_warmstart: Compute algorithm weights from prior performance
  - 14 comprehensive tests covering all functionality

### 7.3 AutoML Results
- [x] **TASK-087**: Implement comprehensive AutoML result object
  - Leaderboard with CIs
  - Ensemble composition
  - Feature importance aggregated across models
  - Statistical significance of model differences

---

## Phase 8: Python Bindings & Serialization

### 8.1 Core Bindings
- [x] **TASK-088**: Export all linear models to Python
- [x] **TASK-089**: Export all tree models to Python
- [x] **TASK-090**: Export preprocessing transformers to Python
- [x] **TASK-091**: Export Pipeline and ColumnTransformer to Python
- [x] **TASK-092**: Export AutoML to Python
- [x] **TASK-093**: Add maturin build configuration

### 8.2 Data Interchange
- [x] **TASK-094**: NumPy zero-copy with `numpy` crate
  - Created `array_utils.rs` module with documented zero-copy utilities
  - `to_owned_array_1d`, `to_owned_array_2d` for input conversion
  - `as_array_view_1d`, `as_array_view_2d` for read-only access (zero-copy)
  - `into_pyarray` wrappers for ownership transfer to Python (no data copy)
  - Shape and contiguity check utilities
  - Comprehensive documentation on zero-copy semantics
  - Updated all binding modules to use centralized utilities
- [x] **TASK-095**: Polars DataFrame support (zero-copy)
  - Added optional `polars` feature flag to ferroml-python (enabled by default)
  - Created `polars_utils.rs` module with DataFrame conversion utilities
  - `extract_xy_from_pydf()`: Extract features and target from PyDataFrame
  - `extract_x_from_pydf()`: Extract features only for prediction
  - `PolarsConversionError` enum with descriptive error variants
  - Null value detection and numeric type validation
  - Added `fit_dataframe()` and `predict_dataframe()` methods to all linear models
  - Uses pyo3-polars with `derive` feature for type re-exports
  - Feature names preserved from DataFrame columns
- [x] **TASK-096**: Pandas DataFrame support (via Arrow)
  - Added optional `pandas` feature flag to ferroml-python (enabled by default)
  - Created `pandas_utils.rs` module with DataFrame conversion utilities
  - `extract_xy_from_pandas()`: Extract features and target from Pandas DataFrame
  - `extract_x_from_pandas()`: Extract features only for prediction
  - `PandasConversionError` enum with descriptive error variants
  - Null/NaN value detection and numeric type validation
  - Added `fit_pandas()` and `predict_pandas()` methods to all linear models
  - Uses Pandas `.to_numpy()` and `.select_dtypes()` for efficient conversion
- [x] **TASK-097**: Sparse matrix support (CSR, CSC)
  - Added optional `sparse` feature flag to ferroml-python (enabled by default)
  - Created `sparse_utils.rs` module with CSR/CSC conversion utilities
  - `sparse_to_dense()`: Convert via scipy's `toarray()` method
  - `sparse_to_dense_efficient()`: Direct conversion using data/indices/indptr arrays
  - `SparseConversionError` enum with descriptive error variants
  - `SparseMatrixInfo` struct for matrix metadata and sparsity analysis
  - Added `fit_sparse()` and `predict_sparse()` methods to all linear models
  - LogisticRegression includes `predict_proba_sparse()` method
  - Supports CSR, CSC, and auto-conversion of other formats

### 8.3 Serialization & Deployment
- [x] **TASK-098**: Implement model save/load (serde JSON, MessagePack, bincode)
  - Created `serialization` module with comprehensive save/load functionality
  - Format enum: Json, JsonPretty, MessagePack, Bincode
  - SerializationMetadata: ferroml_version, model_type, format, timestamp, description
  - ModelContainer wrapper for versioned serialization
  - File I/O: save_model, load_model, load_model_auto (auto-detect format)
  - In-memory: to_bytes, from_bytes, from_bytes_with_metadata
  - ModelSerialize trait: Extension trait with save_json, save_msgpack, save_bincode, etc.
  - Magic bytes for bincode format validation (FRML prefix)
  - 17 comprehensive tests covering all functionality
- [x] **TASK-099**: Python pickle compatibility via `__getstate__`/`__setstate__`
- [x] **TASK-100**: Implement ONNX export for deployment
  - Added `onnx` module with ONNX protobuf definitions using prost
  - OnnxExportable trait with export_onnx() and to_onnx() methods
  - Linear model ONNX export: LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
  - Tree model ONNX export: DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor
  - Uses MatMul+Add ops for linear models, TreeEnsembleRegressor/Classifier from ONNX-ML domain for trees
  - 26 comprehensive tests covering all ONNX export functionality
- [x] **TASK-101**: Implement pure-Rust inference mode (no Python needed)
  - Added `inference` module with pure-Rust ONNX runtime
  - InferenceSession: Load ONNX models from bytes or file, execute inference
  - Tensor: Multi-dimensional f32 array with shape, strides, operations
  - TensorI64: Multi-dimensional i64 array for integer outputs
  - Supported operators: MatMul, Add, Squeeze, Sigmoid, Softmax, Flatten, Reshape
  - TreeEnsembleRegressor/Classifier operators for tree-based models
  - Value enum for different output types (Tensor, TensorI64, SequenceMap)
  - 19 comprehensive tests covering tensor ops, operators, and full inference
- [x] **TASK-102**: Implement feature schema validation
  - FeatureSchema: Captures expected structure (n_features, feature specs, validation mode)
  - FeatureSpec: Per-feature specification (type, missing policy, range, categories)
  - FeatureType: Continuous, Integer, Categorical, Binary, Unknown
  - ValidationMode: Strict (all fail), Warn (log issues), Permissive (shape only)
  - ValidationIssue: ShapeMismatch, MissingValue, ValueBelowMin/AboveMax, UnknownCategory, etc.
  - ValidationResult: Pass/fail with issues, issue_counts(), severity filtering
  - SchemaValidated trait: Models can expose expected schema
  - 22 comprehensive tests covering all validation scenarios

---

## Phase 9: Datasets & Benchmarks

### 9.1 Dataset Module
- [x] **TASK-103**: Create `datasets` module
- [x] **TASK-104**: Built-in toy datasets (iris, wine, diabetes, linnerud)
  - load_iris: 150 samples, 4 features, 3 classes (Fisher's Iris)
  - load_wine: 178 samples, 13 features, 3 classes (Wine recognition)
  - load_diabetes: 260 samples (subset), 10 features (regression)
  - load_linnerud: 20 samples, 3 features (multi-output regression)
  - Complete metadata: feature names, target names, descriptions, sources, licenses
- [x] **TASK-105**: HuggingFace Hub integration via Python bindings
  - Created `ferroml.datasets` Python submodule
  - `load_huggingface()` function using Python's `datasets` library
  - `Dataset` wrapper with: x, y, n_samples, n_features, feature_names, train_test_split, describe
  - `DatasetInfo` wrapper with full metadata support
  - Exposed built-in datasets and synthetic generators to Python
  - Automatic numeric column detection, split/config support, caching option
- [x] **TASK-106**: CSV/Parquet loading utilities
  - `load_csv()`, `load_csv_with_options()`: CSV file loading with CsvOptions
  - `load_parquet()`, `load_parquet_with_options()`: Parquet file loading with ParquetOptions
  - `load_file()`: Auto-detect format from file extension
  - Automatic numeric column detection and target column selection
  - Integration with Dataset and DatasetInfo structures

### 9.2 Benchmarking
- [x] **TASK-107**: Benchmark suite vs sklearn (correctness)
  - Created `ferroml_core/tests/sklearn_correctness.rs` integration test suite
  - 20 tests comparing FerroML outputs to sklearn reference values
  - Covers linear models (LinearRegression, Ridge, Lasso), tree models (DecisionTree), and scalers
  - Documents tolerance levels and intentional differences from sklearn
- [x] **TASK-108**: Benchmark suite vs sklearn (performance)
  - Updated `ferroml_core/benches/benchmarks.rs` with comprehensive performance benchmarks
  - Linear models: LinearRegression, RidgeRegression, LassoRegression fit/predict timing
  - Tree models: DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor fit/predict timing
  - Preprocessing: All scalers (Standard, MinMax, Robust, MaxAbs) fit_transform/transform timing
  - Scaling analysis: Measures how performance scales with dataset size (100 to 100K samples)
  - Throughput metrics: Elements/sec for comparative analysis with sklearn
- [x] **TASK-109**: Benchmark vs XGBoost/LightGBM for gradient boosting
  - Added gradient boosting benchmarks to `ferroml_core/benches/benchmarks.rs`
  - GradientBoostingClassifier/Regressor fit and predict benchmarks
  - HistGradientBoostingClassifier/Regressor fit and predict benchmarks (LightGBM-style)
  - Tree scaling benchmarks: Training time vs number of trees (5, 10, 20, 50, 100)
  - Sample scaling benchmarks: Standard vs histogram-based with dataset size
  - Prediction comparison: Standard vs histogram across different batch sizes
  - Python comparison script: `benchmarks/xgboost_lightgbm_timing.py`
  - Documentation of expected performance tradeoffs and FerroML's advantages
- [x] **TASK-110**: Publish benchmark results to HuggingFace Hub
  - Created `benchmarks/publish_to_huggingface.py` script
  - Parses Criterion benchmark output (Rust benchmarks)
  - Collects XGBoost/LightGBM comparison results
  - System metadata collection (OS, CPU, Rust version, FerroML version)
  - HuggingFace Hub dataset publishing with dataset card
  - Local JSON output option for offline use
  - Automatic README generation with benchmark documentation

---

## Phase 10: Performance Optimization

### 10.1 Parallelization
- [x] **TASK-111**: Rayon parallelism for CV folds
- [x] **TASK-112**: Rayon parallelism for ensemble training
- [x] **TASK-113**: Parallel tree building in Random Forest (already implemented with rayon par_iter in forest.rs)

### 10.2 Low-Level Optimization
- [x] **TASK-114**: SIMD acceleration for distance calculations
  - Created `simd` module using `wide` crate for portable SIMD
  - Euclidean, Manhattan, Minkowski distance with f64x4 vectorization
  - Dot product, sum, sum of squares, cosine similarity
  - Integrated with KNN via conditional compilation
  - 26 unit tests, works with and without `simd` feature
- [x] **TASK-115**: SIMD acceleration for matrix operations
  - Matrix-vector multiplication: `matrix_vector_mul`, `matrix_vector_mul_into`, `vector_matrix_mul`
  - Element-wise operations: `vector_add`, `vector_sub`, `vector_mul`, `vector_div`, scalar variants
  - BLAS-like operations: `axpy`, `axpby`, `outer_product`
  - Row/column aggregations: `sum_rows`, `sum_cols`, `row_means`, `col_means`, `row_norms_l2`, `row_norms_l1`
  - In-place transforms: `normalize_rows_l2`, `center_rows`, `scale_rows_by_std`
  - 29 new matrix operation tests (55 total SIMD tests)
- [x] **TASK-116**: Memory-mapped datasets for large data
  - `MemmappedDataset`: Zero-copy access to features and targets via memory-mapped files
  - `MemmappedArray2`, `MemmappedArray1`, `MemmappedArray2Mut`: Low-level mmap arrays
  - `MemmappedDatasetBuilder`: Fluent API for creating mmap datasets
  - Binary format with "FRMD" magic, version, shape metadata
  - Batch and sample iterators for processing large datasets
  - `peek_mmap_info()` for quick metadata inspection
  - 21 comprehensive tests covering all functionality
- [x] **TASK-117**: Sparse matrix optimization throughout
  - `CsrMatrix`, `CscMatrix`, `SparseVector`: Native sparse matrix types
  - Sparse distance calculations: euclidean, manhattan, cosine (O(nnz) complexity)
  - `sparse_pairwise_distances`: Batch distance with `SparseDistanceMetric` enum
  - Matrix operations: normalize_rows_l2, column_sums/means, vstack/hstack
  - Utility functions: sparse_eye, sparse_diag, dense conversion
  - `SparseMatrixInfo`: Memory analysis and storage recommendations
  - 23 comprehensive tests, uses `sprs` crate, feature-gated as `sparse`

---

## Phase 11: Documentation & Examples

### 11.1 Rust Examples
- [x] **TASK-118**: `examples/linear_regression.rs` - Diagnostics showcase
- [x] **TASK-119**: `examples/classification.rs` - Full workflow
- [x] **TASK-120**: `examples/pipeline.rs` - Preprocessing + model
- [x] **TASK-121**: `examples/automl.rs` - AutoML with statistical output
- [x] **TASK-122**: `examples/gradient_boosting.rs` - Monotonic constraints

### 11.2 Python Examples
- [x] **TASK-123**: Jupyter notebook: "Getting Started with FerroML"
- [x] **TASK-124**: Jupyter notebook: "Statistical Diagnostics Deep Dive"
- [x] **TASK-125**: Jupyter notebook: "FerroML vs sklearn Comparison"
- [x] **TASK-126**: Jupyter notebook: "Production Deployment Guide"

### 11.3 API Documentation
- [x] **TASK-127**: Comprehensive doc comments for all public APIs
- [x] **TASK-128**: Host documentation on docs.rs / GitHub Pages
  - Added docs.rs metadata with all-features enabled
  - Created GitHub Actions workflow for GitHub Pages deployment
  - Documentation accessible at https://user.github.io/ferroml/ferroml_core/ (after pages setup)
  - docs.rs will auto-build when published to crates.io
- [x] **TASK-129**: User guide with conceptual explanations
  - Created `docs/user-guide.md` with comprehensive documentation
  - Covers philosophy, core concepts, all modules, deployment, best practices
  - Includes glossary and further reading references

---

## Phase 12: CI/CD & Release

### 12.1 Continuous Integration
- [x] **TASK-130**: GitHub Actions: cargo check, clippy, fmt
- [x] **TASK-131**: GitHub Actions: cargo test on Linux, macOS, Windows
- [x] **TASK-132**: GitHub Actions: Python binding tests
- [x] **TASK-133**: Code coverage with codecov.io

### 12.2 Release Automation
- [x] **TASK-134**: Automated crates.io publishing on tag
- [x] **TASK-135**: Automated PyPI wheel building via maturin
- [x] **TASK-136**: Changelog generation from commits
- [x] **TASK-137**: GitHub Release creation with artifacts

---

## Completed Tasks

### Statistics Module (Pre-existing)
- [x] T-tests (one-sample, two-sample, paired) with effect sizes
- [x] Mann-Whitney U test (non-parametric)
- [x] Effect sizes (Cohen's d, Hedges' g, rank-biserial)
- [x] Confidence intervals (normal, Student's t, bootstrap)
- [x] Multiple testing correction (Bonferroni, Holm, Hochberg, BH, BY)
- [x] Descriptive statistics
- [x] Correlation with CI
- [x] Power analysis (sample size, power calculation)

### HPO Module (Pre-existing)
- [x] Search space definition (Int, Float, Categorical, Bool with log scale)
- [x] Random sampler
- [x] Grid sampler
- [x] TPE (Tree-Parzen Estimator) sampler
- [x] Median pruner
- [x] Hyperband scheduler
- [x] ASHA scheduler
- [x] Study management (ask/tell interface)
- [x] Parameter importance calculation

---

## Blocked Tasks

_None_

---

## Notes

### Critical Dependencies (Must respect order)
- **Phase 1.1 (Metrics)** before **Phase 1.2 (CV)** - CV needs scoring functions
- **Phase 1.2 (CV)** before **Phase 1.4 (Models)** - Models need CV for validation
- **Phase 1 (Foundation)** before **Phase 2 (Algorithms)** - Algorithms use traits
- **Phase 2 (Algorithms)** before **Phase 3 (Ensemble)** - Ensembles wrap algorithms
- **Phase 4 (Calibration)** before **Phase 7 (AutoML)** - AutoML should auto-calibrate
- **All Rust** before **Phase 8 (Python)** - Bindings wrap Rust

### Validation Checklist (Run after each task)
```bash
cargo check && cargo clippy -- -D warnings && cargo test && cargo fmt --check
```

### FerroML's Unique Positioning
1. **Statistical rigor** - Every model includes assumption tests, CIs, diagnostics
2. **Calibrated by default** - Automatic probability calibration
3. **Explainable AutoML** - Feature importance, PDPs, significance tests on results
4. **Rust-native inference** - Deploy without Python runtime
5. **Production-ready** - ONNX export, schema validation, versioning

### All Tasks Complete!
FerroML v0.1.0 implementation is complete. All 137 tasks across 12 phases have been implemented:
- Phase 1: Foundation (Core Infrastructure)
- Phase 2: Core ML Algorithms
- Phase 3: Ensemble & Pipeline
- Phase 4: Probability & Class Imbalance
- Phase 5: Explainability & Interpretability
- Phase 6: Hyperparameter Optimization
- Phase 7: AutoML Orchestration
- Phase 8: Python Bindings & Serialization
- Phase 9: Datasets & Benchmarks
- Phase 10: Performance Optimization
- Phase 11: Documentation & Examples
- Phase 12: CI/CD & Release
