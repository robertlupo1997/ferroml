# FerroML Roadmap

> **Last Updated:** 2026-03-01

## Current Status: v0.1.0 Feature-Complete

FerroML has completed 11 plans of quality hardening with 2,949 tests passing. All core ML algorithms are implemented, validated against sklearn, and hardened with correctness tests.

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

## Feature Completion Matrix

### Algorithms

| Category | sklearn | FerroML | Notes |
|----------|---------|---------|-------|
| Linear Models | 10+ | 10 | Ridge, Lasso, ElasticNet, Quantile, Robust, RidgeCV, LassoCV, ElasticNetCV, SGD, PassiveAggressive |
| Trees | 2 | 4 | Decision trees + ExtraTrees (classifier/regressor) |
| Ensembles | 6+ | 10 | RF, GB, HistGB, AdaBoost, ExtraTrees, Bagging, Stacking, Voting |
| SVM | 4 | 4 | Full parity |
| KNN | 2 | 3 | + NearestCentroid |
| Naive Bayes | 4 | 3 | Missing ComplementNB |
| Clustering | 8+ | 3 | KMeans, DBSCAN, AgglomerativeClustering |
| Neural Networks | 2 | 2 | MLPClassifier, MLPRegressor |
| Calibration | 2 | 3 | + Temperature scaling |
| Decomposition | 7+ | 5 | PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis |

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
| KernelSHAP | Via shap | Built-in | Model-agnostic |
| PDP | Yes | Yes | 1D and 2D |
| ICE | Yes | Yes | With centering |
| H-Statistic | No | Yes | Interaction detection |

### Python Bindings Coverage (~85%)

| Module | Coverage | Details |
|--------|----------|---------|
| Models | ~95% | 22 models exposed |
| Preprocessing | ~85% | 21 transformers + 5 resamplers |
| Decomposition | 100% | PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis |
| Explainability | ~85% | TreeSHAP, permutation importance, PDP, ICE, H-statistic |
| Clustering | 100% | KMeans, DBSCAN, AgglomerativeClustering |
| Not exposed | — | BaggingClassifier (trait object), RFE (trait object), KernelSHAP (lifetime) |

## Next Steps

### v0.1.0 Release

- [ ] Python integration tests (`maturin develop && pytest`)
- [ ] Fix remaining minor warnings
- [ ] Publish to crates.io and PyPI
- [ ] CI/CD pipeline for automated releases

### v0.2.0 (Future)

- [ ] Expose BaggingClassifier via factory pattern
- [ ] KernelSHAP bindings (owned model storage)
- [ ] ComplementNB
- [ ] GPU acceleration (optional feature)
- [ ] Published benchmarks vs XGBoost/LightGBM

### v1.0.0 (Future)

- [ ] Production-ready stability
- [ ] Full sklearn parity for core algorithms
- [ ] Comprehensive documentation and tutorials
