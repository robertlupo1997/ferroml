# FerroML — AI Agent Reference

> Statistically rigorous machine learning in Rust with Python bindings.
> 55+ models, statistical diagnostics, confidence intervals, AutoML.

## Installation

```bash
pip install ferroml            # core (numpy only)
pip install ferroml[cli]       # + CLI (typer, rich, polars)
pip install ferroml[all]       # everything
```

## CLI Quick Reference

All commands support `--json` for structured agent output.

```bash
# Recommend algorithms for a dataset
ferroml recommend --data data.csv --target y --json

# Train a model
ferroml train --model LinearRegression --data data.csv --target y --output model.pkl --json

# Train with hyperparameters
ferroml train --model RidgeRegression --data data.csv --target y --params '{"alpha": 0.5}' --output model.pkl

# Train with hold-out evaluation
ferroml train --model LinearRegression --data data.csv --target y --test-size 0.2 --output model.pkl --json

# Predict
ferroml predict --model model.pkl --data new_data.csv --target y --json

# Evaluate on labeled data
ferroml evaluate --model model.pkl --data test.csv --target y --json
ferroml evaluate --model model.pkl --data test.csv --target y --metrics rmse,r2,mae --json

# Statistical diagnostics
ferroml diagnose --model model.pkl --data data.csv --target y --json

# Compare multiple models
ferroml compare --models LinearRegression,RidgeRegression,LassoRegression --data data.csv --target y --json

# Model metadata
ferroml info LinearRegression --json
ferroml info --all --json

# AutoML search
ferroml automl --data data.csv --target y --task regression --time-budget 60 --json

# Export to ONNX
ferroml export --model model.pkl --output model.onnx --n-features 4
```

## Python API Quick Reference

```python
import numpy as np
import ferroml

# Algorithm recommendation
recs = ferroml.recommend(X, y, task="regression")
for r in recs:
    print(r.algorithm, r.reason, r.score)

# Model training
from ferroml.linear import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Statistical diagnostics (first-class feature)
print(model.summary())
print(model.r_squared(), model.adjusted_r_squared())
print(model.f_statistic())           # (F, p-value)
print(model.coefficients_with_ci())  # coefficients + confidence intervals
print(model.aic(), model.bic())

# Model card metadata
card = LinearRegression.model_card()
print(card.task, card.complexity, card.strengths)

# Metrics
from ferroml.metrics import r2_score, rmse, accuracy_score, f1_score

# Stats
from ferroml.stats import durbin_watson, normality_test, ttest_ind

# Preprocessing
from ferroml.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from ferroml.preprocessing import CountVectorizer, TfidfVectorizer

# AutoML
from ferroml.automl import AutoML, AutoMLConfig
config = AutoMLConfig(task="regression", metric="rmse", time_budget_seconds=60)
result = AutoML(config).fit(X, y)
print(result.summary())

# Pipelines
from ferroml.pipeline import Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
pipe.fit(X_train, y_train)

# ONNX export
model.export_onnx("model.onnx")

# Serialization (ML ecosystem standard via MessagePack under the hood)
model.save("model.ferroml")
loaded = LinearRegression.load("model.ferroml")
```

## Model Selection Guide

| Task | Start With | When to Upgrade |
|------|-----------|-----------------|
| Regression (small) | `LinearRegression`, `RidgeRegression` | Non-linear patterns -> `GradientBoostingRegressor` |
| Regression (large) | `HistGradientBoostingRegressor` | Need interpretability -> `LinearRegression` |
| Classification (small) | `LogisticRegression` | Non-linear -> `RandomForestClassifier` |
| Classification (large) | `HistGradientBoostingClassifier` | Need probabilities -> `GaussianProcessClassifier` |
| Clustering | `KMeans` | Unknown k -> `HDBSCAN` |
| Anomaly detection | `IsolationForest` | Need local density -> `LocalOutlierFactor` |
| Dimensionality reduction | `PCA` | Non-linear -> `TSNE` |
| Text features | `CountVectorizer` + `TfidfVectorizer` | |
| Don't know | `ferroml recommend --data X --target y --json` | |

## Available Models (55+)

**Linear:** LinearRegression, LogisticRegression, RidgeRegression, LassoRegression, ElasticNet, RobustRegression, QuantileRegression, Perceptron, RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier, IsotonicRegression

**Trees:** DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor, HistGradientBoostingClassifier/Regressor

**Ensemble:** ExtraTreesClassifier/Regressor, AdaBoostClassifier/Regressor, SGDClassifier/Regressor, VotingClassifier/Regressor, StackingClassifier/Regressor, BaggingClassifier/Regressor

**Neighbors:** KNeighborsClassifier/Regressor, NearestCentroid

**Naive Bayes:** GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB

**SVM:** LinearSVC, LinearSVR, SVC, SVR

**Neural:** MLPClassifier, MLPRegressor

**Gaussian Process:** GaussianProcessRegressor, GaussianProcessClassifier

**Clustering:** KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, GaussianMixture

**Anomaly:** IsolationForest, LocalOutlierFactor

**Decomposition:** PCA, TruncatedSVD, IncrementalPCA, TSNE, FactorAnalysis

## Error Handling

All errors are `FerroError` variants with `.hint()` remediation:

```python
try:
    model.fit(X, y)
except ValueError as e:
    print(e)  # includes hint automatically
    # "Shape mismatch: expected 3 features, got 4 features
    #  Hint: Ensure X.shape[0] == y.shape[0]..."
```

Key error types: `ShapeMismatch`, `NotFitted`, `ConvergenceFailure`, `InvalidInput`, `AssumptionViolation`

## Build & Test

```bash
# Rust
cargo test                              # core library (4,400+ tests)
cargo test --test correctness           # correctness suite

# Python
source .venv/bin/activate
maturin develop --release -m ferroml-python/Cargo.toml
pytest ferroml-python/tests/            # 2,900+ tests
```

## Key Imports by Submodule

```python
ferroml.linear       # Linear/logistic regression, ridge, lasso, elastic net
ferroml.trees        # Decision trees, random forests, gradient boosting
ferroml.ensemble     # Extra trees, AdaBoost, SGD, voting, stacking, bagging
ferroml.neighbors    # KNN, nearest centroid
ferroml.naive_bayes  # Gaussian, multinomial, Bernoulli, categorical NB
ferroml.svm          # Linear SVC/SVR, kernel SVC/SVR
ferroml.neural       # MLP classifier/regressor
ferroml.gaussian_process  # GP regressor/classifier
ferroml.clustering   # KMeans, DBSCAN, HDBSCAN, GMM, agglomerative
ferroml.anomaly      # Isolation forest, LOF
ferroml.decomposition    # PCA, SVD, t-SNE, factor analysis
ferroml.preprocessing    # Scalers, encoders, vectorizers
ferroml.metrics      # Regression, classification, clustering metrics
ferroml.stats        # Statistical tests, confidence intervals
ferroml.automl       # AutoML, AutoMLConfig
ferroml.pipeline     # Pipeline, FeatureUnion
ferroml.model_selection  # Cross-validation, train/test split
```
