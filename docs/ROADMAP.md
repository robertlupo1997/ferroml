# FerroML Roadmap

> **Last Updated:** 2026-02-08
> **Reference:** `thoughts/shared/research/2026-02-08_ferroml-comprehensive-assessment.md`

## Current Status: Early Alpha (v0.1.0-alpha)

FerroML has completed 6 phases of quality hardening with 2287 tests passing. The core ML algorithms are implemented and being validated against sklearn.

## Implementation Plans

All plans are documented in `thoughts/shared/plans/` with detailed task breakdowns.

### Phase A: Validation & Documentation (High Priority)

| Plan | Tasks | Description | Status |
|------|-------|-------------|--------|
| [Plan 1: Sklearn Accuracy Testing](../thoughts/shared/plans/2026-02-08_plan-1-sklearn-accuracy-testing.md) | 8 | Validate all models against sklearn | In Progress |
| [Plan 2: Doctest Fixes](../thoughts/shared/plans/2026-02-08_plan-2-doctest-fixes.md) | 10 | Fix 59 failing doctests | Planned |
| [Plan 7: Documentation](../thoughts/shared/plans/2026-02-08_plan-7-documentation.md) | 8 | Tutorials, API docs, accuracy report | Planned |

### Phase B: Major Features (High Priority)

| Plan | Tasks | Description | Status |
|------|-------|-------------|--------|
| [Plan 3: Clustering](../thoughts/shared/plans/2026-02-08_plan-3-clustering-implementation.md) | 10 | KMeans, DBSCAN + statistical extensions | Planned |
| [Plan 4: Neural Networks](../thoughts/shared/plans/2026-02-08_plan-4-neural-networks.md) | 10 | MLP + diagnostics, uncertainty | Planned |

### Phase C: Quality & Completion (Medium Priority)

| Plan | Tasks | Description | Status |
|------|-------|-------------|--------|
| [Plan 5: Code Quality](../thoughts/shared/plans/2026-02-08_plan-5-code-quality.md) | 8 | Remove dead code, reduce suppressions | Planned |
| [Plan 6: Advanced Features](../thoughts/shared/plans/2026-02-08_plan-6-advanced-features.md) | 8 | BCa CI, GPU, streaming serialization | Planned |

## Feature Completion Matrix

### Algorithms

| Category | sklearn | FerroML | Notes |
|----------|---------|---------|-------|
| Linear Models | 10+ | 7 | Complete for core use cases |
| Trees | 2 | 2 | Full parity |
| Ensembles | 6+ | 8 | Exceeds sklearn (stacking, voting, bagging) |
| SVM | 4 | 4 | Full parity |
| KNN | 2 | 2 | Full parity |
| Naive Bayes | 4 | 3 | Missing ComplementNB |
| **Clustering** | **8+** | **0** | **Major gap - Plan 3** |
| **Neural Networks** | **2** | **0** | **Gap - Plan 4** |
| Calibration | 2 | 3 | Exceeds (temp scaling) |
| Decomposition | 7+ | 5 | Core complete |

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

### HPO

| Method | sklearn | FerroML | Notes |
|--------|---------|---------|-------|
| Grid Search | Yes | Yes | |
| Random Search | Yes | Yes | |
| TPE | Via optuna | Built-in | True l/g ratio |
| Bayesian Optimization | Via optuna | Built-in | GP-based |
| Hyperband | Via optuna | Built-in | |
| ASHA | Via optuna | Built-in | |
| BOHB | Via optuna | Built-in | |

## Milestones

### v0.1.0-beta (Target: Q1 2026)

- [ ] All sklearn accuracy tests passing
- [ ] 59 doctests fixed
- [ ] Accuracy report published
- [ ] User tutorials complete

### v0.2.0 (Target: Q2 2026)

- [ ] KMeans + DBSCAN with stability metrics
- [ ] MLPClassifier/Regressor with diagnostics
- [ ] Dead code removed
- [ ] GPU support (optional feature)

### v1.0.0 (Target: Q3 2026)

- [ ] Production-ready stability
- [ ] Full sklearn parity for core algorithms
- [ ] Published benchmarks vs XGBoost/LightGBM
- [ ] Comprehensive documentation

## Contributing

See individual plan documents for task details. Each plan is designed to be completed in 1-2 focused sessions without hitting context limits.

Priority order for contributors:
1. Sklearn accuracy testing (immediate value)
2. Doctest fixes (quick wins)
3. Clustering (major feature gap)
4. Code quality (maintainability)
