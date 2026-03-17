# FerroML Roadmap

> **Last Updated:** 2026-03-16

## Current Status: v0.3.1

FerroML has completed 30 plans of development and hardening (Plans 1-6, A-X) with 3,550+ Rust tests and 2,100+ Python tests passing (5,650+ total). All core ML algorithms are implemented, validated against sklearn/scipy/xgboost/lightgbm/statsmodels, and hardened with correctness tests. Python binding coverage is ~99%.

## Release History

### v0.3.1 (2026-03-16) — Production Ready

- SVC RBF kernel convergence fix (WSS3 j-selection bug)
- RidgeCV NaN bug fix
- ONNX RandomForest roundtrip fix
- Numerical stability and input validation hardening (Plan X)
- cross_validate(), train_test_split, ROC/PR curves in Python
- 200+ cross-library correctness tests
- PyPI publishing workflow with TestPyPI support

### v0.3.0 (2026-03-15) — sklearn API Parity

- `score(X, y)` on 56 models (R² for regressors, accuracy for classifiers)
- `partial_fit` on 10 models (SGD, NaiveBayes, Perceptron, PassiveAggressive, IncrementalPCA)
- `decision_function` on 13 classifiers
- Performance optimization (HistGBT, KMeans, LogReg)
- Feature parity scorecard
- Cross-library validation (164 tests vs 6 libraries)
- 5-pass robustness audit (36 issues found, 35 fixed)

### v0.2.0 (2026-03-11) — Feature Completion

- CountVectorizer, TfidfVectorizer
- GaussianProcessRegressor/Classifier (RBF, Matern, Constant, White kernels)
- Sparse GP variants (FITC, VFE, SVGP)
- MultiOutput wrappers
- GPU shaders (12 WGSL shaders, GpuDispatcher)
- SparseModel trait (12 models)
- Voting/Stacking ensembles
- ONNX export for all models

### v0.1.0 (2026-03-09) — Initial Release

- 55+ ML algorithms across 13 categories
- Python bindings via PyO3
- Statistical features (CI, effect sizes, power analysis, fairness)
- Explainability (TreeSHAP, KernelSHAP, PDP, ICE)
- 86+ Criterion benchmarks

## Completed Plans

| Plans | Description | Status |
|-------|-------------|--------|
| 1-6 | Foundation (sklearn tests, clustering, neural, code quality) | Complete |
| A-E | Hardening (correctness tests, benchmarks, Python bindings) | Complete |
| F-L | Feature completion (GMM, anomaly, t-SNE, QDA, test suites) | Complete |
| M-O | Validation & optimization (SIMD, Barnes-Hut, HDBSCAN) | Complete |
| P | SVM polish (decision_function, class weights) | Complete |
| Q | GPU shaders, SparseModel trait | Complete |
| R | CountVectorizer, GP, MultiOutput, v0.2.0 release | Complete |
| S | Cross-library validation (linfa, xgboost, lightgbm, statsmodels, scipy) | Complete |
| T | Performance optimization, feature parity scorecard, warm_start | Complete |
| U | sklearn API parity (score, partial_fit, decision_function), v0.3.0 | Complete |
| V | Open-source polish, bug fixes, v0.3.1 | Complete |
| W | Performance fixes (Elkan KMeans, HistGBT, L-BFGS LogReg, tutorials) | Complete |
| X | Production-ready (numerical stability, input validation, cross-library tests) | Complete |

## Next Steps

### Future

- [ ] ComplementNB (Naive Bayes variant)
- [ ] Spectral Clustering
- [ ] sklearn migration guide
- [ ] Tutorial notebooks
- [ ] Published benchmarks on standard datasets
- [ ] v1.0.0 stability release
