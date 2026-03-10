# Changelog

All notable changes to FerroML are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

#### Plans M-O: Validation & Optimization (2026-03-08 — 2026-03-09)

**Plan M — Real-World Validation (2026-03-08):**
- 279 side-by-side comparison tests against sklearn on real datasets (iris, wine, breast_cancer, diabetes)
- Performance benchmarks: fit/predict time comparison across 15 models at 4 dataset sizes
- Edge case testing: high dimensionality, class imbalance, near-constant features, multicollinearity
- Zero bugs found during validation

**Plan N — Performance Optimization (2026-03-08):**
- DBSCAN 2.4x faster, KNN 1.75x faster, DecisionTree 1.5x faster
- RandomForest predict 1.92x faster, GradientBoosting fit 1.46x faster
- Barnes-Hut t-SNE: O(N log N) via VP-tree + QuadTree (was O(N²))
- Memory allocation reduction in HistGB predict, gradient buffers, KMeans
- Parallel predict for GradientBoosting and KNN via rayon
- SVM incremental error update optimization
- Benchmark CI workflow with regression detection script
- SIMD batch distance computation

**Plan O — System Validation + Features (2026-03-09):**
- CategoricalNB: Categorical Naive Bayes (~450 lines, 25 Rust + 13 Python tests) — completes NB family 4/4
- HDBSCAN: Hierarchical density-based clustering (~780 lines, 33 Rust + 12 Python tests, VP-tree k-NN, Prim's MST, condensed tree, excess-of-mass extraction)
- 20 AutoML Rust system tests (end-to-end pipeline validation)
- 17 AutoML Python system tests
- LogReg/Quantile X'WX triple loop replaced with ndarray .dot() (BLAS-backed GEMM, ~2-3x speedup)
- faer-backend enabled by default with faer Cholesky dispatch
- CategoricalNB wired into AutoML classification portfolio

#### Plans F-L: v0.1.0 Completion (2026-03-02 — 2026-03-08)

**Plan F — CI Fixes & Tests (2026-03-02):**
- Fixed Python virtualenv detection in CI (`maturin develop` failures across 9 jobs)
- Fixed Rust linker OOM / disk exhaustion on CI runners
- Updated `bytes` crate to fix RUSTSEC-2026-0007 security advisory
- Fixed `cargo deny check advisories` and license compliance jobs
- Fixed GitHub Pages docs deployment step
- Added 120 Python tests for BaggingRegressor and RFE

**Plan G — Python Bindings for 14 Models + CI Hardening (2026-03-07):**
- Naive Bayes bindings: GaussianNB, MultinomialNB, BernoulliNB (30 Python tests)
- Regularized CV bindings: RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier (36 Python tests)
- Specialized model bindings: RobustRegression, QuantileRegression, Perceptron, NearestCentroid (27 Python tests)
- LinearSVC, LinearSVR, TemperatureScalingCalibrator bindings (22 Python tests)
- CI hardening: strict clippy (`-D warnings`), removed `|| true` patterns, strict pytest
- README synced with actual Python exports

**Plan H — Gaussian Mixture Models (2026-03-07):**
- `GaussianMixture` implementing `ClusteringModel` trait with full EM algorithm
- 4 covariance types: Full, Tied, Diagonal, Spherical
- BIC/AIC for model selection, `predict_proba()` for soft clustering, `sample()` for generation
- Cholesky decomposition and triangular solvers added to `linalg.rs`
- 29 Rust tests + 30 Python tests

**Plan I — Anomaly Detection (2026-03-07):**
- `IsolationForest` with random isolation trees, anomaly scoring, contamination threshold
- `LocalOutlierFactor` with k-nearest neighbor density estimation
- New `OutlierDetector` trait for shared anomaly detection interface
- Python bindings for both models + anomaly module
- 58 Rust tests + 30 Python tests

**Plan J — t-SNE (2026-03-07):**
- `TSNE` implementing `Transformer` trait for nonlinear dimensionality reduction
- Exact O(N^2) algorithm with gradient descent and momentum
- 3 distance metrics (Euclidean, Manhattan, Cosine), PCA/random initialization
- Configurable perplexity, learning rate, early exaggeration
- 28 Rust tests + 19 Python tests

**Plan K — QDA + IsotonicRegression (2026-03-07):**
- `QuadraticDiscriminantAnalysis` with per-class covariance matrices, regularization
- `IsotonicRegression` wrapping PAVA algorithm (generalized beyond [0,1] clipping)
- Python bindings + tests for both
- 43 Rust tests + 24 Python tests

**Plan L — Testing Phases 23-28 (2026-03-08):**
- 6 new test modules: multioutput, cv_advanced, ensemble_advanced, categorical, incremental, metrics_custom
- 218 Rust tests covering multi-output prediction, advanced CV (learning_curve, validation_curve), ensemble stacking, categorical features (HistGradientBoosting, ColumnTransformer), warm start / incremental learning, custom metrics

**Final Completion (2026-03-08):**
- Phases 29-32: Fairness testing, drift detection, regression baselines, mutation testing
- GPU backend tests (67 tests) for wgpu acceleration
- Kernel SVM Python bindings (SVC/SVR)
- Code deduplication: shared math/helper modules, deduplicated PyO3 bindings
- `cargo fmt --all` and clippy clean across all new files

### Fixed

- RidgeCV NaN panic: `partial_cmp().unwrap()` on NaN cv_scores (Plan G)
- Class extraction epsilon bug in model deduplication (post-Plan L)

#### Plans A-E: Hardening & Correctness (2026-03-01)

**Plan A — Clustering Hardening:**
- 102 correctness tests for KMeans, DBSCAN, AgglomerativeClustering
- Tests cover all linkage methods (Ward, complete, average, single), clustering metrics (silhouette, ARI, NMI, CH, DB, Hopkins, Dunn), edge cases, and diagnostics

**Plan B — Neural Network Hardening:**
- 49 correctness tests for MLPClassifier, MLPRegressor
- Tests cover training convergence, activation functions, multi-class/multi-output, diagnostics, and edge cases

**Plan C — Performance Benchmarks:**
- 15 new Criterion benchmark functions (86+ total)
- Added benchmarks for RidgeCV, LassoCV, ElasticNetCV, RobustRegression, QuantileRegression, PassiveAggressive, RidgeClassifier, CalibratedClassifierCV, LDA, FactorAnalysis, and 5 scaling benchmarks
- Created `baseline.json` for regression detection

**Plan D — Python Bindings Completion:**
- Decomposition module: PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis
- Ensemble module: ExtraTrees, AdaBoost, SGD, PassiveAggressive
- Explainability module: TreeSHAP, permutation importance (10 model types), PDP, 2D PDP, ICE, H-statistic
- Preprocessing additions: SelectFromModel, SMOTE, ADASYN, RandomUnderSampler, RandomOverSampler
- AgglomerativeClustering bindings
- Overall Python binding coverage: 35% → 85%

**Plan E — Preprocessing Correctness Tests:**
- 101 correctness tests covering all transformers, scalers, encoders, imputers, and resamplers

### Fixed

#### Plans A-B: Bug Fixes (2026-03-01)

**Clustering (3 bugs):**
- Ward linkage Lance-Williams formula: was squaring already-squared distances (d^4 → d^2)
- Empty cluster handling: now re-initializes to random data point instead of keeping stale center
- KMeans predict() validation: now returns ShapeMismatch error instead of panicking

**Neural Networks (7 bugs):**
- Double gradient application: output layer was applying activation derivative on top of loss gradient
- Softmax derivative: was returning sigmoid derivative — now bypassed for output layer
- Multi-output MSE: now divides by n_samples × n_outputs (was missing n_outputs)
- Cross-entropy clipping: now only clips for log(), gradient uses unclipped predictions
- ELU derivative boundary: changed `>` to `>=` at x=0
- RNG reseeding: MLP now has persistent RNG field instead of recreating every forward pass
- Empty input validation: added n_samples==0 checks in fit/predict

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
