# FerroML

[![CI](https://github.com/robertlupo1997/ferroml/actions/workflows/ci.yml/badge.svg)](https://github.com/robertlupo1997/ferroml/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/ferroml-core.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-4%2C351%20passing-brightgreen)](https://github.com/robertlupo1997/ferroml)

**Statistically rigorous AutoML in Rust with Python bindings.**

> **Status: v0.1.0 Feature-Complete** — 50+ ML algorithms, 4,351 tests passing (2,933 Rust + 1,418 Python), validated against sklearn with 252+ correctness tests. Plans A-O complete. See [Project Status](#project-status) for details.

FerroML is a high-performance machine learning library that prioritizes statistical rigor over black-box automation. Unlike traditional AutoML tools that hide statistical assumptions, FerroML makes them explicit and testable.

## Key Features

- **Statistical Rigor First** — Confidence intervals on all predictions, hypothesis testing for model comparison, multiple testing correction (Bonferroni, Holm, Benjamini-Hochberg)
- **Transparent Assumptions** — All statistical assumptions are documented and tested. No hidden magic.
- **Reproducible Results** — Deterministic by default with explicit randomness control
- **High Performance** — Written in Rust with SIMD acceleration, parallel processing via Rayon, and optional sparse matrix support
- **Python Bindings** — Seamless NumPy/Polars integration via PyO3

## Installation

### Rust

```toml
[dependencies]
ferroml-core = "0.1"
```

### Python

```bash
pip install ferroml
```

## Quick Start

### Rust

```rust
use ferroml_core::{AutoML, AutoMLConfig, Metric, Task};
use ferroml_core::datasets::load_iris;

fn main() -> ferroml_core::Result<()> {
    // Load dataset
    let (dataset, _info) = load_iris();
    let (train, _test) = dataset.train_test_split(0.2, true, Some(42))?;
    let (x, y) = train.into_arrays();

    // Configure AutoML with statistical controls
    let config = AutoMLConfig {
        task: Task::Classification,
        metric: Metric::Accuracy,
        time_budget_seconds: 60,
        cv_folds: 5,
        statistical_tests: true,
        confidence_level: 0.95,
        seed: Some(42),
        ..Default::default()
    };

    // Run AutoML
    let automl = AutoML::new(config);
    let result = automl.fit(&x, &y)?;

    // Results include confidence intervals and statistical tests
    if let Some(best) = result.best_model() {
        println!("Best: {:?}", best.algorithm);
        println!("Score: {:.4} ± {:.4}", best.cv_score, best.cv_std);
        println!("95% CI: [{:.4}, {:.4}]", best.ci_lower, best.ci_upper);
    }

    Ok(())
}
```

### Python

```python
import ferroml as fml

# Load data
dataset, info = fml.datasets.load_iris()
X, y = dataset.x, dataset.y

# Create and fit a model
config = fml.AutoMLConfig(
    task="classification",
    metric="accuracy",
    time_budget_seconds=60,
)
automl = fml.AutoML(config)
result = automl.fit(X, y)

# Results with statistical guarantees
best = result.best_model()
print(f"Best: {best.algorithm}")
print(f"Score: {best.cv_score:.4f} ± {best.cv_std:.4f}")
print(f"95% CI: [{best.ci_lower:.4f}, {best.ci_upper:.4f}]")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FerroML Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   stats     │  │   models    │  │    hpo      │        │
│  │             │  │             │  │             │        │
│  │ Hypothesis  │  │ Linear      │  │ Bayesian    │        │
│  │ Tests       │  │ Tree-based  │  │ Optim       │        │
│  │ Confidence  │  │ Ensemble    │  │ Hyperband   │        │
│  │ Intervals   │  │ Boosting    │  │ ASHA        │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐      │
│  │              preprocessing                       │      │
│  │  Imputation, Encoding, Scaling, Selection       │      │
│  └─────────────────────────────────────────────────┘      │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐      │
│  │                 pipeline                         │      │
│  │  DAG execution, Caching, Parallelization        │      │
│  └─────────────────────────────────────────────────┘      │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐      │
│  │                    cv                            │      │
│  │  K-Fold, Stratified, TimeSeries, Nested         │      │
│  └─────────────────────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `stats` | Hypothesis testing, confidence intervals, effect sizes, multiple testing correction |
| `models` | Linear, logistic, ridge, lasso, elastic net, decision trees, random forests, gradient boosting, SVM (kernel + linear), KNN, naive bayes (Gaussian/Multinomial/Bernoulli), extra trees, AdaBoost, SGD, QDA, isotonic regression, isolation forest, LOF |
| `ensemble` | Bagging, stacking, voting classifiers with diversity-weighted selection |
| `hpo` | Bayesian optimization, Hyperband, ASHA, random/grid search |
| `preprocessing` | Imputation, one-hot/target encoding, standard/robust scaling, feature selection, SMOTE/ADASYN resampling |
| `clustering` | KMeans (k-means++), DBSCAN, HDBSCAN, AgglomerativeClustering (Ward/complete/average/single), GaussianMixture (EM, 4 covariance types) |
| `neural` | MLPClassifier, MLPRegressor with training diagnostics, MC Dropout uncertainty |
| `decomposition` | PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis, t-SNE |
| `pipeline` | DAG-based pipeline execution with caching and parallel processing |
| `cv` | K-Fold, Stratified K-Fold, Group K-Fold, Time Series Split, Nested CV |
| `automl` | Automated model selection with statistical model comparison |
| `explainability` | Permutation importance, SHAP values, partial dependence |
| `metrics` | Classification (ROC-AUC, F1, MCC) and regression (R², RMSE, MAE) metrics |
| `onnx` | Export trained models to ONNX format for deployment |
| `datasets` | Built-in datasets (Iris, Diabetes, etc.) for testing and examples |

## Statistical Features

FerroML distinguishes itself through rigorous statistical methodology:

- **Corrected Resampled t-test** (Nadeau-Bengio) for comparing cross-validated models
- **Multiple Testing Correction** — Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR
- **Confidence Intervals** on all CV scores and predictions
- **Assumption Testing** — Normality, homoscedasticity, independence checks
- **Effect Size Reporting** — Cohen's d, confidence intervals for differences

## Examples

Run the included examples:

```bash
cargo run --example automl
cargo run --example linear_regression
cargo run --example classification
cargo run --example gradient_boosting
cargo run --example pipeline
```

## Feature Flags

```toml
[dependencies]
ferroml-core = { version = "0.1", features = ["simd", "sparse", "onnx"] }
```

| Feature | Description |
|---------|-------------|
| `parallel` | Parallel processing (enabled by default) |
| `simd` | SIMD acceleration for distance calculations |
| `sparse` | Native sparse matrix operations |
| `onnx` | ONNX model export (enabled by default) |
| `faer-backend` | BLAS-backed linear algebra via faer (enabled by default) |
| `gpu` | GPU acceleration via wgpu (GEMM, distance matrix) |

## Performance

FerroML leverages Rust's performance characteristics:

- **Zero-cost abstractions** — No runtime overhead for safety
- **Rayon parallelism** — Automatic parallel iteration
- **SIMD operations** — Vectorized distance calculations
- **Memory-mapped files** — Efficient handling of large datasets
- **faer-backend** — BLAS-accelerated linear algebra (Cholesky, matrix solve) via faer
- **Barnes-Hut t-SNE** — O(N log N) visualization with VP-tree and QuadTree
- **LTO optimization** — Link-time optimization in release builds

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Project Status

### Current State (v0.1.0)

FerroML is **feature-complete for v0.1.0**, hardened through 21 plans (1-6, A-O) of correctness and quality work:

| Metric | Status |
|--------|--------|
| **Tests** | 4,351 passing (2,933 Rust + 1,418 Python), 0 failing, 26 ignored (slow AutoML system tests) |
| **Correctness Tests** | 252+ (clustering: 102, neural: 49, preprocessing: 101) |
| **Sklearn Match** | 58 comparisons passing (32 models + preprocessing) |
| **Python Test Files** | 23 end-to-end test files |
| **Python Bindings** | ~99% coverage (50+ models, 21 preprocessors, 6 decomposition, 37 explainability) |
| **Benchmarks** | 86+ Criterion benchmarks with CI regression baseline |

### What's Implemented

- **Linear Models**: LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier, Quantile, Robust, Perceptron, SGD
- **Trees & Ensembles**: DecisionTree, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, AdaBoost, Bagging, Stacking, Voting
- **Instance-Based**: KNN (classifier + regressor) with KD-Tree/Ball-Tree acceleration, NearestCentroid
- **SVM**: SVC, SVR (kernel), LinearSVC, LinearSVR
- **Probabilistic**: Gaussian/Multinomial/Bernoulli/Categorical Naive Bayes, QuadraticDiscriminantAnalysis (QDA)
- **Anomaly Detection**: IsolationForest, LocalOutlierFactor (LOF)
- **Clustering**: KMeans (k-means++), DBSCAN, HDBSCAN, AgglomerativeClustering (4 linkage methods), GaussianMixture (EM, 4 covariance types, BIC/AIC)
- **Calibration**: TemperatureScaling, Sigmoid (Platt), Isotonic
- **Regression**: IsotonicRegression (monotonic constraints)
- **Neural Networks**: MLPClassifier, MLPRegressor with training diagnostics, MC Dropout
- **Decomposition**: PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis, t-SNE
- **Preprocessing**: 23+ transformers (scalers, encoders, imputers, SMOTE variants, RFE)
- **Explainability**: TreeSHAP (Lundberg 2018), KernelSHAP (10 typed variants), PDP, ICE, H-statistic
- **HPO**: TPE, Hyperband, ASHA, BOHB, Bayesian optimization

### Accuracy Validation

FerroML is validated against scikit-learn with 58 fixture-based comparisons and 252 correctness tests:

| Category | Tests | Status |
|----------|-------|--------|
| Models (sklearn comparison) | 17 | All passing |
| Preprocessing (sklearn comparison) | 15 | All passing |
| Clustering correctness | 102 | All passing |
| Neural network correctness | 49 | All passing |
| Preprocessing correctness | 101 | All passing |

| Algorithm Type | Tolerance |
|----------------|-----------|
| Closed-form (Linear, PCA) | 1e-8 |
| Iterative (Lasso, Logistic) | 1e-4 |
| Tree-based | 1e-6 |
| Ensemble (RF, GB) | 5% (RNG variance) |

Full accuracy report: [docs/accuracy-report.md](docs/accuracy-report.md)

See [CHANGELOG.md](CHANGELOG.md) for detailed change history.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

### Development Setup

1. **Install pre-commit hooks** (recommended):

   ```bash
   # Install pre-commit (if not already installed)
   pip install pre-commit

   # Install the git hooks
   pre-commit install
   ```

   Pre-commit hooks will automatically run on every commit to ensure:
   - Code is formatted with `cargo fmt`
   - Code passes `cargo clippy` lints
   - Quick unit tests pass
   - No debug macros (`todo!()`, `unimplemented!()`, `dbg!()`) in production code
   - No large files (>1MB) are accidentally committed

2. **Run hooks manually** (useful before pushing):

   ```bash
   pre-commit run --all-files
   ```

3. **Skip hooks temporarily** (use sparingly):

   ```bash
   git commit --no-verify -m "WIP: work in progress"
   ```
