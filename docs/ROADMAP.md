# FerroML Roadmap

> **Last Updated:** 2026-03-08

## Current Status: v0.1.0 Complete

FerroML has completed 18 plans of development and hardening (Plans 1-6, A-L) with ~3,500 Rust tests and ~1,047 Python tests passing. All core ML algorithms are implemented, validated against sklearn, and hardened with correctness tests. Python binding coverage is ~99%.

## Completed Plans

### Phase 1-6: Foundation (2026-02-06 — 2026-02-11)

| Plan | Description | Status |
|------|-------------|--------|
| Plan 1: Sklearn Accuracy Testing | 58 fixture-based comparisons | Complete |
| Plan 2: Doctest Fixes | 82 doctests passing | Complete |
| Plan 3: Clustering | KMeans, DBSCAN with statistical extensions | Complete |
| Plan 4: Neural Networks | MLP + diagnostics, uncertainty | Complete |
| Plan 5: Code Quality | Dead code removal, lint cleanup | Complete |
| Plan 6: Advanced Features | BCa CI, streaming serialization | Complete |

### Phase A-E: Hardening (2026-02-25 — 2026-03-01)

| Plan | Description | Status |
|------|-------------|--------|
| Plan A: Clustering Hardening | 3 bug fixes + 102 correctness tests | Complete |
| Plan B: Neural Network Hardening | 7 bug fixes + 49 correctness tests | Complete |
| Plan C: Performance Benchmarks | 15 new benchmarks (86+ total) | Complete |
| Plan D: Python Bindings | Coverage 35% → 85% | Complete |
| Plan E: Preprocessing Tests | 101 correctness tests | Complete |

### Phase F-L: Completion (2026-03-02 — 2026-03-08)

| Plan | Description | Status |
|------|-------------|--------|
| Plan F: CI Fixes & Tests | CI green + 120 BaggingRegressor/RFE tests | Complete |
| Plan G: Python Bindings + CI | 14 models exposed, strict CI, README sync | Complete |
| Plan H: Gaussian Mixture Models | EM, 4 covariance types, BIC/AIC, 59 tests | Complete |
| Plan I: Anomaly Detection | IsolationForest, LOF, OutlierDetector trait, 88 tests | Complete |
| Plan J: t-SNE | Exact O(N²), 3 metrics, PCA init, 47 tests | Complete |
| Plan K: QDA + IsotonicRegression | Per-class covariance, PAVA, 67 tests | Complete |
| Plan L: Testing Phases 23-28 | 6 test modules, 218 tests | Complete |

## Feature Completion Matrix

### Algorithms

| Category | sklearn | FerroML | Notes |
|----------|---------|---------|-------|
| Linear Models | 10+ | 10 | Ridge, Lasso, ElasticNet, Quantile, Robust, RidgeCV, LassoCV, ElasticNetCV, SGD, PassiveAggressive |
| Trees | 2 | 4 | Decision trees + ExtraTrees (classifier/regressor) |
| Ensembles | 6+ | 10 | RF, GB, HistGB, AdaBoost, ExtraTrees, Bagging, Stacking, Voting |
| SVM | 4 | 4 | SVC, SVR, LinearSVC, LinearSVR — full parity |
| KNN | 2 | 3 | + NearestCentroid |
| Naive Bayes | 4 | 3 | Missing ComplementNB |
| Clustering | 8+ | 4 | KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture |
| Anomaly Detection | 2 | 2 | IsolationForest, LocalOutlierFactor |
| Decomposition | 7+ | 6 | PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis, t-SNE |
| Discriminant Analysis | 2 | 2 | LDA, QDA |
| Neural Networks | 2 | 2 | MLPClassifier, MLPRegressor |
| Calibration | 2 | 3 | + Temperature scaling |
| Isotonic | 1 | 1 | IsotonicRegression |

### Statistical Features (FerroML Differentiators)

| Feature | sklearn | FerroML | Notes |
|---------|---------|---------|-------|
| Confidence Intervals | No | Yes | On all predictions |
| Effect Sizes | No | Yes | Cohen's d, Hedges' g, etc. |
| Multiple Testing | No | Yes | Bonferroni, Holm, BH, BY |
| Power Analysis | No | Yes | Sample size calculations |
| Model Diagnostics | Minimal | Extensive | VIF, Cook's D, residuals |
| Fairness Metrics | Via fairlearn | Built-in | 5 metrics + intersectional |
| Assumption Testing | No | Yes | Normality, homoscedasticity |

### Explainability

| Method | sklearn | FerroML | Notes |
|--------|---------|---------|-------|
| Permutation Importance | Yes | Yes | With CI |
| TreeSHAP | Via shap | Built-in | Lundberg 2018 Algorithm 2 |
| KernelSHAP | Via shap | Built-in | Model-agnostic, 10 typed variants |
| PDP | Yes | Yes | 1D and 2D |
| ICE | Yes | Yes | With centering |
| H-Statistic | No | Yes | Interaction detection |

### Python Bindings Coverage (~99%)

| Module | Coverage | Details |
|--------|----------|---------|
| Models | ~99% | 37+ models exposed (linear, trees, ensemble, SVM, NB, anomaly, QDA, isotonic) |
| Preprocessing | ~95% | 21 transformers + 5 resamplers + RFE |
| Decomposition | 100% | PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis, t-SNE |
| Explainability | ~95% | TreeSHAP, KernelSHAP (10 variants), permutation importance, PDP, ICE, H-statistic |
| Clustering | 100% | KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture |
| Anomaly | 100% | IsolationForest, LocalOutlierFactor |
| Naive Bayes | 100% | GaussianNB, MultinomialNB, BernoulliNB |
| SVM | 100% | SVC, SVR, LinearSVC, LinearSVR |
| Calibration | 100% | CalibratedClassifierCV, TemperatureScalingCalibrator |
| Ensemble | 100% | BaggingClassifier (8 factories), BaggingRegressor (9 factories), Stacking, Voting |

## Next Steps

### Post-v0.1.0 Priorities

#### Documentation & Tutorials
- [ ] Comprehensive API documentation (rustdoc + Python docstrings)
- [ ] Tutorial notebooks (classification, regression, clustering, anomaly detection)
- [ ] Migration guide for sklearn users
- [ ] Performance comparison guide

#### Real-World Validation
- [ ] Published benchmarks vs XGBoost/LightGBM/sklearn on standard datasets
- [ ] End-to-end examples on Kaggle-style problems
- [ ] Community feedback integration

#### Performance
- [ ] GPU acceleration (wgpu backend exists, needs hardening)
- [ ] Barnes-Hut t-SNE for large datasets (O(N log N))
- [ ] Sparse matrix optimizations for text/NLP workloads

#### Coverage Gaps
- [ ] ComplementNB (Naive Bayes)
- [ ] HDBSCAN, Spectral Clustering
- [ ] stats/power.rs and stats/diagnostics.rs test coverage
- [ ] GPU backend tests (67 exist, more needed)

### v1.0.0 (Future)

- [ ] Production-ready stability
- [ ] Full sklearn parity for core algorithms
- [ ] Publish to crates.io and PyPI
- [ ] CI/CD pipeline for automated releases
- [ ] Comprehensive documentation and tutorials
