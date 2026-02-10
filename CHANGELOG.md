# Changelog

All notable changes to FerroML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Progress
- Sklearn accuracy comparison testing (validating numerical correctness)
- Comprehensive documentation updates

### Planned
- Clustering algorithms (KMeans, DBSCAN) with statistical extensions
- Neural networks (MLPClassifier, MLPRegressor) with diagnostics
- 59 doctest fixes

---

## [0.1.0-alpha] - 2026-02-08

### Quality Hardening Release

This release represents a major quality hardening effort across 6 phases, fixing algorithmic correctness issues and rewriting key components to match reference implementations.

**Test Status:** 2287 tests passing, Clippy clean

### Phase 1: Quick Correctness Fixes (commit `d4e60c1`)

#### Added
- Fisher z-transform guard for r=±1 edge case (`stats/mod.rs:308-309`)
- Log-scale HPO parameter bounds validation - asserts bounds > 0 (`hpo/search_space.rs:123,142`)
- Box-Muller clamp to prevent NaN from log(0) (`hpo/samplers.rs:577`)
- Correlation NaN guard for zero-variance inputs (`hpo/mod.rs:380`)
- Reshape -1 dimension inference in ONNX inference (`inference/operators.rs:203-254`)
- Bootstrap CI percentile rounding (`stats/bootstrap.rs:106-107`)
- PR-AUC precision interpolation for monotonicity (`metrics/probabilistic.rs:244`)
- INT32/DOUBLE tensor support in inference engine (`inference/session.rs`)

#### Fixed
- Unknown data type now returns proper error instead of empty tensor

### Phase 2: Lentz's Incomplete Beta (commit `d4e60c1`)

#### Changed
- Replaced naive continued fraction with Lentz algorithm in 4 files for numerical stability:
  - `stats/mod.rs:370-396`
  - `stats/confidence.rs:218-242`
  - `stats/hypothesis.rs:364-388`
  - `metrics/comparison.rs:418-443`

#### Removed
- Duplicate dead `beta()` functions from 3 files (consolidated to single implementation)

### Phase 3: LDA Eigenvalue Solver (commit `d4e60c1`)

#### Changed
- LDA now uses symmetric transformation for eigenvalue problem
- Cholesky primary path with SVD fallback for non-positive-definite S_w
- Eigen and SVD solvers now produce consistent predictions

### Phase 4: Serialization Improvements (commit `d4e60c1`)

#### Added
- `SemanticVersion` type with proper Ord implementation for version comparison
- CRC32 integrity checksums on all bincode read/write paths
- Corruption detection for serialized models

### Phase 5: TPE Sampler Rewrite (commit `e42dacc`)

#### Added
- `OneDimensionalKDE` struct for 1D Gaussian kernel KDE with Scott's rule bandwidth
- `CategoricalDistribution` with Laplace smoothing for unseen categories
- `log_add_exp()` for numerically stable log-space operations
- Configurable `n_ei_candidates` parameter (default 24)
- 4 new tests for TPE correctness

#### Changed
- TPE sampler now uses true l(x)/g(x) density ratio algorithm (Bergstra et al.)
- Bad trials now contribute to g(x) density instead of being discarded
- Categorical parameters use frequency distribution instead of random sampling

### Phase 6: TreeSHAP Rewrite (commit `28b17cd`)

#### Added
- Complete rewrite using Lundberg 2018 Algorithm 2
- Sentinel element at path[0] for correct pweight calculations
- Hot/cold branch traversal (visits both paths, not just taken branch)
- `extend_path()` - updates polynomial pweights using recurrence relation
- `unwind_path()` - handles repeated features correctly (shifts feature/fraction but NOT pweights)
- `unwound_path_sum()` - separate formulas for hot vs cold branches
- Path cloning before recursive calls for state preservation
- Depth-2 tree test with hand-computed Shapley values

#### Changed
- SHAP values now sum exactly to (prediction - base_value) without normalization
- TreeSHAP tolerance tightened from 1.0 to 1e-6

### Earlier Bug Fixes (commit `553f921`)

#### Fixed
- z_inv_normal sign inversion in `models/linear.rs`
- t_critical df<5 dead code branch in `models/linear.rs`
- VIF used covariance diagonal instead of R² in `models/linear.rs`
- MAE leaf used mean instead of median in `models/tree.rs`
- OOB score misaligned class indices in `models/forest.rs`
- SMO bounds used wrong index in `models/svm.rs`
- KNN tie-breaking returned last class instead of first in `models/knn.rs`

---

## [0.0.x] - Pre-release

### Features

- Initial release of FerroML - a statistically rigorous AutoML library in Rust

### Added

#### Core Infrastructure
- Statistics module with hypothesis tests, CIs, effect sizes, multiple testing, power analysis, bootstrap
- HPO core with search space, Random/Grid/TPE samplers, MedianPruner/Hyperband/ASHA schedulers
- Metrics module with classification, regression, and probabilistic metrics
- Cross-validation module with KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, NestedCV, and more
- Preprocessing module with scalers, encoders, imputers, and feature selectors
- Model traits and linear models with full statistical diagnostics

#### Machine Learning Models
- Linear models: LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, QuantileRegression, RobustRegression
- Probabilistic models: GaussianNB, MultinomialNB, BernoulliNB
- K-Nearest Neighbors: KNeighborsClassifier, KNeighborsRegressor
- Support Vector Machines: SVC, SVR, LinearSVC, LinearSVR
- Decision Trees: DecisionTreeClassifier, DecisionTreeRegressor
- Random Forests: RandomForestClassifier, RandomForestRegressor with OOB error and feature importance with CIs
- Gradient Boosting: GradientBoostingClassifier, GradientBoostingRegressor
- Histogram Gradient Boosting: HistGradientBoostingClassifier, HistGradientBoostingRegressor with monotonic constraints

#### Dimensionality Reduction
- PCA with explained variance ratios and whitening
- IncrementalPCA for large datasets
- TruncatedSVD for sparse matrices
- LDA (Linear Discriminant Analysis)
- FactorAnalysis with rotation methods

#### Ensemble Methods
- VotingClassifier and VotingRegressor
- StackingClassifier and StackingRegressor with CV-based stacking
- BaggingClassifier and BaggingRegressor with OOB estimation

#### Pipeline Module
- Pipeline with named steps and caching
- FeatureUnion for parallel feature extraction
- ColumnTransformer for column-wise preprocessing

#### Probability Calibration
- CalibratedClassifierCV with Platt scaling and isotonic regression
- Temperature scaling for multi-class calibration
- Calibration curve generation for reliability diagrams

#### Class Imbalance Handling
- SMOTE, BorderlineSMOTE, ADASYN oversampling
- RandomUnderSampler, RandomOverSampler
- SMOTE-Tomek and SMOTE-ENN combined sampling

#### Explainability
- Permutation importance with confidence intervals
- Partial Dependence Plots (PDP) 1D and 2D
- Individual Conditional Expectation (ICE) curves
- H-statistic for feature interaction detection
- TreeSHAP for tree-based models
- KernelSHAP for model-agnostic explanations
- SHAP summary plot data generation

#### AutoML
- Algorithm portfolio with Quick/Balanced/Thorough presets
- Bandit-based time budget allocation (UCB1, Thompson Sampling, Successive Halving)
- Automatic preprocessing selection
- Ensemble construction from trials
- Dataset metafeature extraction
- Meta-learning warm-start from similar datasets
- Configuration space transfer

#### Python Bindings
- Linear, tree, preprocessing, pipeline, and AutoML submodules
- NumPy zero-copy array support
- Polars and Pandas DataFrame support
- Sparse matrix (CSR, CSC) support
- Python pickle compatibility

#### Serialization & Deployment
- JSON, MessagePack, and Bincode serialization formats
- ONNX export for linear and tree models
- Pure-Rust inference mode (no Python needed)
- Feature schema validation

#### Datasets
- Built-in toy datasets: iris, wine, diabetes, linnerud
- Synthetic data generators: make_classification, make_regression, make_blobs, make_moons, make_circles
- CSV and Parquet file loading
- HuggingFace Hub integration via Python

#### Performance
- SIMD acceleration for distance calculations and matrix operations
- Memory-mapped datasets for large data
- Sparse matrix optimization
- Rayon parallelism throughout

#### CI/CD
- GitHub Actions for CI (check, clippy, fmt, test)
- Cross-platform testing (Linux, macOS, Windows)
- Automated crates.io publishing
- Automated PyPI wheel building via maturin
- Code coverage with codecov.io
- Documentation hosting on GitHub Pages

<!-- Generated by git-cliff -->
