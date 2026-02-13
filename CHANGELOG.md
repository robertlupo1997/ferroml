# Changelog

All notable changes to FerroML are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

#### Plan 6: Advanced Features (2026-02-11)
- Bootstrap BCa confidence intervals — bias-corrected and accelerated CIs via jackknife acceleration (`compute_bca()`)
- Probit activation — standard normal CDF transform for inference operators (`PostTransform::Probit`)
- SaveOptions/LoadOptions API — configurable serialization with metadata toggle, progress callbacks, version compatibility
- Streaming serialization — chunked bincode format with `StreamingWriter`/`StreamingReader`, CRC32 integrity verification

#### Plan 4: Neural Networks (2026-02-10)
- MLPClassifier — multi-layer perceptron for classification with softmax output, early stopping, training diagnostics
- MLPRegressor — multi-layer perceptron for regression with target normalization
- 7 activation functions: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, Linear
- Optimizers: SGD with momentum, Adam with bias correction
- Learning rate schedules: Constant, InverseScaling, Adaptive
- Training diagnostics: convergence detection, gradient statistics, dead neuron detection
- Uncertainty estimation: MC Dropout, confidence intervals, calibration analysis (ECE, MCE)
- Python bindings: `PyMLPClassifier`, `PyMLPRegressor`

#### Plan 3: Clustering (2026-02-09)
- KMeans — k-means++ initialization, silhouette score, Davies-Bouldin index, inertia, Calinski-Harabasz index
- DBSCAN — eps-neighborhood density clustering with core point detection
- Clustering metrics: inertia, silhouette, Davies-Bouldin, Calinski-Harabasz
- Clustering diagnostics: cluster size distribution, separation metrics, stability analysis
- Python bindings: `PyKMeans`, `PyDBSCAN`, 7 metric functions

#### Plan 2: Doctest Fixes (2026-02-09)
- Enabled 19 module-level doctests with proper test data
- Doctests: 82 passing, 0 failing

#### Plan 1: Sklearn Accuracy Testing (2026-02-09)
- Added KNN Python bindings (`KNeighborsClassifier`, `KNeighborsRegressor`)
- Accuracy validation: LinearRegression, DecisionTree, RandomForest match sklearn within tolerances

### Changed

#### Plan 5: Code Quality (2026-02-11)
- Removed unused `portfolio` and `study` fields from AutoML
- Removed unused tree functions (`gini_impurity`, `entropy`, `build_tree`, `find_best_split`)
- Removed unused linear model functions (`breusch_pagan_test`, `chi_squared_cdf`, gamma functions)
- Removed `gpu = []` feature flag
- Reduced `#[allow(dead_code)]` from 18 to 1 (only `KernelDensityEstimator` justified)

### Fixed

#### Bug Audit & Quality Hardening (2026-02-06 — 2026-02-08)

**Critical fixes:**
- LogisticRegression.predict_proba — returned shape `(n, 1)` instead of `(n, 2)`
- RandomForestClassifier OOB — out-of-bounds array access when bootstrap misses classes
- LinearRegression — `z_inv_normal` variable shadowing, VIF formula error, coefficient sign bug
- DecisionTreeRegressor MAE — used mean instead of median for leaf values
- SVM SMO — incorrect bounds calculation with class weights
- Pipeline — cache key used original input (defeating caching), `fit_transform` computed twice
- StackingClassifier — ignored user's `final_estimator`
- StandardScaler — used sample variance (n-1) instead of population variance (n)

**Algorithm rewrites:**
- TreeSHAP — complete rewrite using Lundberg's Algorithm 2 (2018)
- TPE sampler — true l(x)/g(x) density ratio with KDE and Laplace-smoothed categoricals
- LDA eigenvalue solver — symmetric Cholesky + SVD fallback
- Incomplete beta function — Lentz's continued fraction algorithm

**Precision improvements:**
- Applied `mul_add()` optimization (287 clippy suboptimal_flops warnings eliminated)
- Eliminated ~163 `unwrap`/`expect`/`panic` calls in library code
- Added CRC32 integrity checksums on all bincode serialization paths
- Added `SemanticVersion` type with proper Ord implementation
- Added Fisher z-transform r=+/-1 guard, log-scale bounds, Box-Muller clamp

### Removed
- Dead code: unused tree, linear, HPO, ensemble, and serialization functions (Plan 5)
- `gpu = []` feature flag (Plan 5)
- Redundant `beta()` functions replaced by Lentz's algorithm

## [0.1.0] - 2026-01

### Added
- Initial release with 37+ ML algorithms
- **Linear models**: LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, QuantileRegression, RobustRegression
- **Tree-based models**: DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting (classifiers and regressors)
- **Distance-based**: KNeighborsClassifier, KNeighborsRegressor
- **SVM**: SVC, SVR, LinearSVC, LinearSVR
- **Bayesian**: GaussianNB, MultinomialNB, BernoulliNB
- **Preprocessing**: 23+ transformers (scalers, encoders, imputers, feature selection, SMOTE variants)
- **Explainability**: TreeSHAP, KernelSHAP, PDP, ICE, permutation importance, H-statistic
- **HPO**: TPE, Hyperband, ASHA, BOHB, Bayesian optimization, grid/random search
- **AutoML**: Automated model selection with statistical model comparison
- **Statistics**: Hypothesis testing, confidence intervals, effect sizes, multiple testing correction, bootstrap
- **Pipeline**: DAG-based execution with caching and parallelization
- **Cross-validation**: K-Fold, Stratified, GroupK-Fold, TimeSeriesSplit, Nested CV
- **Metrics**: Classification (ROC-AUC, F1, MCC, PR-AUC) and regression (R², RMSE, MAE)
- **ONNX export**: Model serialization to ONNX format
- **Python bindings**: PyO3-based with NumPy/Polars integration
- **Datasets**: Iris, Diabetes, Wine, synthetic generators
- **Performance**: SIMD acceleration, Rayon parallelism, sparse matrix support
