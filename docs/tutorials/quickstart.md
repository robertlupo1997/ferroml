# Quick Start Guide

Learn how to get started with FerroML in under 10 minutes. By the end of this tutorial, you'll know how to train a regression model, a classifier, and apply preprocessing pipelines.

## Learning Objectives

- Install FerroML (Rust and Python)
- Train and evaluate a linear regression model
- Train a classifier with probability outputs
- Build a preprocessing pipeline

## Installation

### Rust

Add `ferroml-core` to your `Cargo.toml`:

```toml
[dependencies]
ferroml-core = "0.3"
```

Optional feature flags:

```toml
[dependencies]
ferroml-core = { version = "0.1", features = ["simd", "sparse"] }
```

| Feature | Description |
|---------|-------------|
| `parallel` | Rayon-based parallelism (enabled by default) |
| `simd` | SIMD-accelerated distance calculations |
| `sparse` | Sparse matrix operations via sprs |
| `onnx` | ONNX model export (enabled by default) |

### Python

```bash
pip install ferroml
```

## Your First Regression Model

Let's predict diabetes progression using `LinearRegression`.

```rust
use ferroml_core::models::linear::LinearRegression;
use ferroml_core::models::Model;
use ferroml_core::datasets::load_diabetes;
use ndarray::Array2;

fn main() -> ferroml_core::Result<()> {
    // Load the diabetes dataset (442 samples, 10 features)
    let (dataset, _info) = load_diabetes();
    let (train, test) = dataset.train_test_split(0.2, true, Some(42))?;
    let (x_train, y_train) = train.into_arrays();
    let (x_test, y_test) = test.into_arrays();

    // Create and fit the model
    let mut model = LinearRegression::new();
    model.fit(&x_train, &y_train)?;

    // Make predictions
    let predictions = model.predict(&x_test)?;

    // Access coefficients
    let coefs = model.coefficients()?;
    let intercept = model.intercept()?;
    println!("Intercept: {:.4}", intercept);
    println!("Coefficients: {:?}", coefs);

    Ok(())
}
```

### Statistical Diagnostics

FerroML models go beyond basic fit/predict. Every `LinearRegression` provides a full statistical summary, similar to R's `lm()`:

```rust
use ferroml_core::models::StatisticalModel;

// Get an R-style summary with all statistics
let summary = model.summary();
println!("{}", summary);

// Access individual fit statistics
let r2 = model.r_squared()?;
let adj_r2 = model.adjusted_r_squared()?;
let (f_stat, f_pvalue) = model.f_statistic()?;
println!("R-squared: {:.4}", r2);
println!("Adjusted R-squared: {:.4}", adj_r2);
println!("F-statistic: {:.4} (p={:.4e})", f_stat, f_pvalue);

// Get coefficients with confidence intervals
let coef_info = model.coefficients_with_ci(0.95);
for ci in &coef_info {
    println!("{}: {:.4} [{:.4}, {:.4}] (p={:.4e})",
        ci.name, ci.estimate, ci.ci_lower, ci.ci_upper, ci.p_value);
}
```

### Prediction Intervals

Unlike most ML libraries, FerroML provides prediction intervals out of the box:

```rust
use ferroml_core::models::ProbabilisticModel;

let intervals = model.predict_interval(&x_test, 0.95)?;
for i in 0..5 {
    println!("Predicted: {:.2} [{:.2}, {:.2}]",
        intervals.predictions[i],
        intervals.lower[i],
        intervals.upper[i]);
}
```

## Your First Classifier

Let's classify iris species using `LogisticRegression`.

```rust
use ferroml_core::models::logistic::LogisticRegression;
use ferroml_core::models::{Model, ProbabilisticModel};
use ferroml_core::datasets::load_iris;

fn main() -> ferroml_core::Result<()> {
    let (dataset, _info) = load_iris();
    let (train, test) = dataset.train_test_split(0.2, true, Some(42))?;
    let (x_train, y_train) = train.into_arrays();
    let (x_test, y_test) = test.into_arrays();

    // Create classifier with explicit configuration
    let mut model = LogisticRegression::new()
        .with_max_iter(100)
        .with_tol(1e-8);

    model.fit(&x_train, &y_train)?;

    // Predict classes
    let predictions = model.predict(&x_test)?;

    // Get class probabilities
    let probabilities = model.predict_proba(&x_test)?;
    for i in 0..5 {
        println!("Predicted: {} | Probabilities: [{:.3}, {:.3}]",
            predictions[i],
            probabilities[[i, 0]],
            probabilities[[i, 1]]);
    }

    Ok(())
}
```

### Handling Imbalanced Classes

FerroML supports class weighting for imbalanced datasets:

```rust
use ferroml_core::models::logistic::{LogisticRegression, ClassWeight};

let mut model = LogisticRegression::new()
    .with_class_weight(ClassWeight::Balanced);

model.fit(&x_train, &y_train)?;
```

## Preprocessing Pipelines

FerroML provides sklearn-compatible preprocessing transformers.

### Scaling Features

```rust
use ferroml_core::preprocessing::scalers::StandardScaler;
use ferroml_core::preprocessing::Transformer;

// StandardScaler: zero mean, unit variance
let mut scaler = StandardScaler::new();
let x_scaled = scaler.fit_transform(&x_train)?;

// Transform test data using training statistics
let x_test_scaled = scaler.transform(&x_test)?;

// Reverse the transformation
let x_original = scaler.inverse_transform(&x_scaled)?;
```

Other scalers follow the same `Transformer` interface:

```rust
use ferroml_core::preprocessing::scalers::{MinMaxScaler, RobustScaler, MaxAbsScaler};

// Scale to [0, 1]
let mut mm = MinMaxScaler::new();
let x_mm = mm.fit_transform(&x_train)?;

// Robust to outliers (uses median and IQR)
let mut robust = RobustScaler::new();
let x_robust = robust.fit_transform(&x_train)?;

// Scale by maximum absolute value
let mut maxabs = MaxAbsScaler::new();
let x_maxabs = maxabs.fit_transform(&x_train)?;
```

### Handling Missing Data

```rust
use ferroml_core::preprocessing::imputers::{SimpleImputer, ImputeStrategy};
use ferroml_core::preprocessing::Transformer;

// Replace NaN values with column means
let mut imputer = SimpleImputer::new(ImputeStrategy::Mean);
let x_imputed = imputer.fit_transform(&x_train)?;
```

### Encoding Categorical Features

```rust
use ferroml_core::preprocessing::encoders::OneHotEncoder;
use ferroml_core::preprocessing::Transformer;

let mut encoder = OneHotEncoder::new();
let x_encoded = encoder.fit_transform(&x_categorical)?;
```

## Choosing a Model

FerroML implements 45+ algorithms. Here's a quick guide:

| Task | Model | When to Use |
|------|-------|-------------|
| Regression | `LinearRegression` | Interpretable, small-medium data |
| Regression | `RandomForestRegressor` | Non-linear, feature importance needed |
| Regression | `GradientBoostingRegressor` | Best accuracy, larger datasets |
| Classification | `LogisticRegression` | Binary/multi-class, interpretable |
| Classification | `RandomForestClassifier` | Non-linear, robust to overfitting |
| Classification | `GradientBoostingClassifier` | Best accuracy, larger datasets |
| Classification | `KNeighborsClassifier` | Small data, no training needed |
| Clustering | `KMeans` | Known number of clusters |
| Clustering | `DBSCAN` | Unknown clusters, arbitrary shapes |
| Clustering | `HDBSCAN` | Variable-density clusters, automatic k |
| Clustering | `GaussianMixture` | Soft clustering, overlapping clusters |
| Anomaly Detection | `IsolationForest` | Unsupervised outlier detection |
| Anomaly Detection | `LocalOutlierFactor` | Density-based outlier detection |
| Classification | `LinearSVC` / `SVC` | Maximum-margin classifier (linear or kernel) |
| Classification | `GaussianNB` | Fast probabilistic baseline |
| Classification | `CategoricalNB` | Discrete categorical features |
| Dimensionality Reduction | `TSNE` | Visualization of high-dimensional data |

## Anomaly Detection

FerroML includes two anomaly detectors that follow sklearn's sign conventions (+1 for inliers, -1 for outliers).

```rust
use ferroml_core::models::isolation_forest::IsolationForest;
use ferroml_core::models::OutlierDetector;

// IsolationForest isolates anomalies via random recursive partitioning
let mut iforest = IsolationForest::new()
    .with_n_estimators(100)
    .with_random_state(Some(42));

iforest.fit_outlier_detector(&x_train)?;

// predict returns +1 (inlier) or -1 (outlier)
let labels = iforest.predict_outliers(&x_test)?;

// score_samples returns anomaly scores (lower = more anomalous)
let scores = iforest.score_samples(&x_test)?;
```

## Soft Clustering with GaussianMixture

When you need probabilistic cluster assignments or overlapping clusters, use `GaussianMixture`:

```rust
use ferroml_core::clustering::{GaussianMixture, CovarianceType, ClusteringModel};

let mut gmm = GaussianMixture::new(3)
    .covariance_type(CovarianceType::Full)
    .random_state(42);

gmm.fit(&x)?;

// Hard cluster assignments
let labels = gmm.predict(&x)?;

// Soft assignments: probability of each cluster
let proba = gmm.predict_proba(&x)?;

// Model selection with BIC/AIC
let bic = gmm.bic(&x)?;
let aic = gmm.aic(&x)?;
println!("BIC: {:.2}, AIC: {:.2}", bic, aic);
```

## Next Steps

- [Statistical Features Tutorial](statistical-features.md) — Model diagnostics, confidence intervals, hypothesis testing
- [Explainability Tutorial](explainability.md) — TreeSHAP, KernelSHAP, partial dependence
- [API Reference](https://docs.rs/ferroml-core) — Full API documentation
