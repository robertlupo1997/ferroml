# FerroML

[![PyPI](https://img.shields.io/pypi/v/ferroml.svg)](https://pypi.org/project/ferroml/)
[![CI](https://github.com/robertlupo1997/ferroml/actions/workflows/ci.yml/badge.svg)](https://github.com/robertlupo1997/ferroml/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/ferroml-core.svg)](https://github.com/robertlupo1997/ferroml)

**High-performance ML in Rust with a scikit-learn-compatible Python API.**

FerroML is a machine learning library written in Rust that provides 55+ algorithms with statistical rigor built in: confidence intervals on predictions, hypothesis testing for model comparison, and assumption checks on every model. On the latest committed cross-library benchmark (2026-03-27, docs/cross-library-benchmark.md) it is up to ~7x faster than scikit-learn on predict for many models — though slower on HistGBT and KNN predict at scale — and, in that recorded run, up to ~20x faster on fit for tree/ensemble models (RandomForest fit; that run predates the max_features parity fix in scripts/benchmark_cross_library.py, so its comparison figures await a rerun).

## Installation

```bash
pip install ferroml
```

Requires Python 3.10+. Pre-built wheels available for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64).

## Quick Start

```python
from ferroml.linear import LinearRegression
import numpy as np

# Linear regression with full statistical diagnostics
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

model = LinearRegression()
model.fit(X, y)
print(model.summary())  # R-style output: coefficients, std errors, p-values, R²
```

```python
from ferroml.trees import RandomForestClassifier
from ferroml.preprocessing import StandardScaler
from ferroml.pipeline import Pipeline

# scikit-learn-compatible pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

```python
from ferroml.preprocessing import CountVectorizer, TfidfTransformer
from ferroml.naive_bayes import MultinomialNB

# Text classification
cv = CountVectorizer()
X_counts = cv.fit_transform(documents)
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_counts)
clf = MultinomialNB()
clf.fit(X_tfidf, y)
```

## Performance vs scikit-learn

Figures below are from the v0.2.0 run (2026-03-11, docs/benchmark-vs-sklearn.md); the newer cross-library benchmark (2026-03-27, docs/cross-library-benchmark.md) supersedes them. Test scores match for most models but differ where model defaults differed. Speedup >1x = FerroML is faster.

| Model | N | Fit | Predict |
|-------|--:|----:|--------:|
| RandomForest | 1K | **9.2x** | **7.8x** |
| Ridge | 1K | **5.3x** | **19.3x** |
| DecisionTree | 5K | **1.4x** | **16.4x** |
| GradientBoosting | 1K | **1.5x** | **1.2x** |
| LogisticRegression | 10K | **1.5x** | **13.7x** |

FerroML is faster on **predict for most models**, though scikit-learn wins on HistGBT and KNN predict at scale, and FerroML is faster on **fit for tree/ensemble models** (Rayon parallel construction). scikit-learn wins on fit for LAPACK/MKL-backed linear algebra.

## Available Models

| Module | Models |
|--------|--------|
| `linear` | LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier, QuantileRegression, RobustRegression, Perceptron, IsotonicRegression |
| `trees` | DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor, HistGradientBoostingClassifier/Regressor |
| `ensemble` | ExtraTrees, AdaBoost, Bagging, Stacking, Voting (classifiers + regressors), SGD, PassiveAggressive |
| `naive_bayes` | GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB |
| `svm` | SVC, SVR, LinearSVC, LinearSVR |
| `neighbors` | KNeighborsClassifier/Regressor, NearestCentroid |
| `neural` | MLPClassifier, MLPRegressor |
| `gaussian_process` | GaussianProcessClassifier, GaussianProcessRegressor |
| `clustering` | KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, GaussianMixture |
| `anomaly` | IsolationForest, LocalOutlierFactor |
| `decomposition` | PCA, IncrementalPCA, TruncatedSVD, LDA, QDA, FactorAnalysis, TSNE |
| `preprocessing` | 22+ transformers: scalers, encoders, imputers, SMOTE/ADASYN, CountVectorizer, TfidfTransformer |
| `explainability` | TreeSHAP, KernelSHAP, permutation importance, PDP, ICE, H-statistic |
| `multioutput` | MultiOutputClassifier, MultiOutputRegressor |
| `calibration` | TemperatureScaling, Sigmoid (Platt), Isotonic |
| `pipeline` | Pipeline, ColumnTransformer, FeatureUnion |
| `model_selection` | train_test_split, cross_validate, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit |
| `metrics` | ROC-AUC, F1, MCC, R², RMSE, MAE, roc_curve, precision_recall_curve |
| `automl` | AutoML with statistical model comparison |
| `datasets` | Iris, Diabetes, Wine, Linnerud, synthetic generators |

## sklearn API Compatibility

FerroML supports the scikit-learn API conventions:

- `fit()` / `predict()` / `transform()` on all models
- `score()` on 56 models (R² for regressors, accuracy for classifiers)
- `partial_fit()` on 10 models for incremental learning
- `decision_function()` on 18 estimators
- `predict_proba()` on probabilistic classifiers
- Pipeline and ColumnTransformer composition
- NumPy array input/output

## Testing

7,600+ tests collected (4,600+ Rust + 3,000+ Python; docs/correctness-report.md), validated against scikit-learn, scipy, xgboost, lightgbm, and statsmodels with 489+ cross-library correctness tests.

## License

MIT OR Apache-2.0
