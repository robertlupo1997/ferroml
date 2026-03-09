# FerroML Benchmark Results — Pre-Optimization Baseline

**Date**: 2026-03-08
**Commit**: 78b72fd (Plan M)
**Hardware**: WSL2 Linux 6.6.87.2, x86_64
**Rust**: stable, `--release` (Criterion bench profile)
**Features**: default (parallel, onnx) — SIMD **not** default yet
**Total benchmarks**: 179 across 3 bench files

## Top 10 Slowest Operations (Optimization Targets)

| # | Benchmark | Median Time | Complexity | Target |
|---|-----------|-------------|------------|--------|
| 1 | KMeans/fit/5000x50 | 700.44 ms | O(n·k·p·iter) | SIMD distances (N.2) |
| 2 | HistGB/fit/max_iter=100 | 592.11 ms | O(trees·n·bins) | Histogram subtraction (N.4) |
| 3 | AgglomerativeClustering/fit/1000x20 | 497.74 ms | O(n²·p) | Distance matrix (N.2) |
| 4 | HistGB/fit/max_iter=50 | 286.40 ms | O(trees·n·bins) | Histogram subtraction (N.4) |
| 5 | LinearRegression/fit/10000x100 | 239.05 ms | O(n·p²) | BLAS (out of scope) |
| 6 | DecisionTreeRegressor/fit/5000x50 | 160.07 ms | O(n·p·log n) | Pre-sorted splits (N.3) |
| 7 | KMeans/fit/1000x20 | 154.24 ms | O(n·k·p·iter) | SIMD distances (N.2) |
| 8 | HistGB Regressor/fit/2000x30 | 99.86 ms | O(trees·n·bins) | Histogram subtraction (N.4) |
| 9 | AdaBoostRegressor/fit/1000x50 | 82.03 ms | O(trees·n·p·log n) | Tree splitting (N.3) |
| 10 | HistGB Regressor/fit/1000x20 | 90.16 ms | O(trees·n·bins) | Histogram subtraction (N.4) |

## Key Algorithm Baselines (Optimization Targets)

### Distance-based (Phase N.2 targets)
| Benchmark | Median | Target |
|-----------|--------|--------|
| KMeans/fit/5000x50 | 700.44 ms | 1.5x → <467 ms |
| KMeans/fit/1000x20 | 154.24 ms | 1.5x → <103 ms |
| DBSCAN/fit/2000x20 | 62.95 ms | 2x → <31.5 ms |
| DBSCAN/fit/1000x20 | 15.80 ms | 2x → <7.9 ms |
| DBSCAN/fit/500x20 | 4.01 ms | 2x → <2.0 ms |
| KNeighborsClassifier/fit_predict/1000x50 | 22.77 ms | 2x → <11.4 ms |
| KNeighborsRegressor/fit_predict/1000x50 | 42.73 ms | 2x → <21.4 ms |

### Tree-based (Phase N.3 targets)
| Benchmark | Median | Target |
|-----------|--------|--------|
| DecisionTreeClassifier/fit/5000x50 | 29.02 ms | 2x → <14.5 ms |
| DecisionTreeClassifier/fit/1000x50 | 3.45 ms | 2x → <1.73 ms |
| DecisionTreeRegressor/fit/5000x50 | 160.07 ms | 2x → <80 ms |
| DecisionTreeRegressor/fit/1000x50 | 21.32 ms | 2x → <10.7 ms |
| RandomForestClassifier/fit/1000x20 | 2.00 ms | 1.5x → <1.33 ms |
| RandomForestRegressor/fit/1000x20 | 5.84 ms | 1.5x → <3.89 ms |

### Gradient Boosting (Phase N.4 targets)
| Benchmark | Median | Target |
|-----------|--------|--------|
| GradientBoostingRegressor/fit/1000x20 | 33.95 ms | 1.5x → <22.6 ms |
| GradientBoostingClassifier/fit/1000x20 | 15.81 ms | 1.5x → <10.5 ms |
| HistGB Regressor/fit/1000x20 | 90.16 ms | 2x → <45.1 ms |
| HistGB Classifier/fit/1000x20 | 53.04 ms | 2x → <26.5 ms |
| GB Regressor/predict/5000x50 | 1.65 ms | 2x → <0.83 ms |
| GB Classifier/predict/5000x50 | 1.19 ms | 2x → <0.60 ms |
| HistGB predict/10000 | 29.54 ms | 2x → <14.8 ms |

### SVM (Phase N.7 targets)
| Benchmark | Median | Target |
|-----------|--------|--------|
| SVC/fit_predict/500x20 | 60.26 ms | 2x → <30.1 ms |
| SVR/fit_predict/500x20 | 25.72 ms | 2x → <12.9 ms |
| LinearSVC/fit_predict/1000x50 | 49.56 ms | — |
| LinearSVR/fit_predict/1000x50 | 46.92 ms | — |

### MLP (Phase N.7 targets)
| Benchmark | Median | Target |
|-----------|--------|--------|
| MLPClassifier/fit_predict/500x20_h64 | 15.79 ms | 1.5x → <10.5 ms |
| MLPRegressor/fit_predict/500x20_h64 | 14.34 ms | 1.5x → <9.6 ms |

## Full Results (sorted by time, descending)

| # | Benchmark | Median Time |
|---|-----------|-------------|
| 1 | KMeans/fit/samples/5000x50 | 700.44 ms |
| 2 | Scaling/HistGradientBoosting/Trees/max_iter/100 | 592.11 ms |
| 3 | AgglomerativeClustering/fit/samples/1000x20 | 497.74 ms |
| 4 | Scaling/HistGradientBoosting/Trees/max_iter/50 | 286.40 ms |
| 5 | LinearRegression/fit/samples/10000x100 | 239.05 ms |
| 6 | MinMaxScaler/fit_transform/samples/100000x100 | 214.36 ms |
| 7 | DecisionTreeRegressor/fit/samples/5000x50 | 160.07 ms |
| 8 | StandardScaler/fit_transform/samples/100000x100 | 156.12 ms |
| 9 | KMeans/fit/samples/1000x20 | 154.24 ms |
| 10 | MaxAbsScaler/fit_transform/samples/100000x100 | 127.79 ms |
| 11 | Scaling/HistGradientBoosting/Trees/max_iter/20 | 119.95 ms |
| 12 | HistGradientBoostingRegressor/fit/samples/2000x30 | 99.86 ms |
| 13 | HistGradientBoostingRegressor/fit/samples/1000x20 | 90.16 ms |
| 14 | AdaBoostRegressor/fit_predict/samples/1000x50 | 82.03 ms |
| 15 | HistGradientBoostingRegressor/fit/samples/500x20 | 72.64 ms |
| 16 | LogisticRegression/fit_predict/samples/1000x50 | 71.13 ms |
| 17 | Scaling/HistGradientBoosting/Trees/max_iter/10 | 66.44 ms |
| 18 | AgglomerativeClustering/fit/samples/500x20 | 63.02 ms |
| 19 | DBSCAN/fit/samples/2000x20 | 62.95 ms |
| 20 | HistGradientBoostingClassifier/fit/samples/2000x30 | 60.88 ms |
| 21 | SVC/fit_predict/samples/500x20 | 60.26 ms |
| 22 | RobustScaler/fit_transform/samples/10000x100 | 57.85 ms |
| 23 | HistGradientBoostingClassifier/fit/samples/500x20 | 57.62 ms |
| 24 | PCA/fit_transform/samples/5000x100 | 57.22 ms |
| 25 | GradientBoosting/Samples/histogram/500 | 54.83 ms |
| 26 | HistGradientBoostingClassifier/fit/samples/1000x20 | 53.04 ms |
| 27 | GradientBoosting/Samples/histogram/1000 | 51.50 ms |
| 28 | LinearSVC/fit_predict/samples/1000x50 | 49.56 ms |
| 29 | GradientBoosting/Samples/histogram/250 | 47.75 ms |
| 30 | LinearSVR/fit_predict/samples/1000x50 | 46.92 ms |
