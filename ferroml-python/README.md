# FerroML

[![PyPI](https://img.shields.io/pypi/v/ferroml.svg)](https://pypi.org/project/ferroml/)
[![CI](https://github.com/robertlupo1997/ferroml/actions/workflows/ci.yml/badge.svg)](https://github.com/robertlupo1997/ferroml/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/ferroml.svg)](https://github.com/robertlupo1997/ferroml)

**High-performance ML in Rust with a scikit-learn-compatible Python API.**

FerroML is a machine learning library written in Rust that provides 55+ algorithms with statistical rigor built in: confidence intervals on cross-validation scores, prediction intervals on the models that support them, hypothesis testing for model comparison, and regression diagnostics. New in 1.0.2: AutoML that picks the simplest model not statistically worse than the best, and says so when models are indistinguishable.

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
predictions = pipe.predict(X_test)
```

```python
from ferroml.automl import AutoML, AutoMLConfig

# Statistically-grounded model selection (new in 1.0.2)
config = AutoMLConfig(task="classification", metric="accuracy",
                      time_budget_seconds=15, statistical_tests=True, seed=42)
result = AutoML(config).fit(X, y)
entry, rationale = result.statistically_best_model()
print(rationale.explanation)
print(result.benchmark_report())
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

Fit time relative to scikit-learn from the 1.0.2 cross-library run (10K × 20, median of three runs; lower is faster):

| Model | vs sklearn |
|-------|-----------:|
| StandardScaler | **0.26x** |
| GaussianNB | **0.32x** |
| RandomForest | **0.36–0.41x** |
| DecisionTree | 0.58–0.81x |
| GradientBoosting | 0.65–0.69x |
| KMeans | 0.73x |
| SVC RBF (5K) | 1.3x |
| HistGBT | 1.8–2.0x |
| LogisticRegression | 2.3x |
| KNN | 3.1x |

Speed depends on data shape, and timings under ~30 ms are noisy. Scores match sklearn within 0.01 for every model except GradientBoostingClassifier (0.906 vs 0.929). RandomForest now matches sklearn (R² 0.785 vs 0.786, accuracy 0.925 vs 0.930) after the 1.0.2 feature-sampling fix. Full numbers, including predict times and scores: [cross-library benchmark](https://github.com/robertlupo1997/ferroml/blob/master/docs/cross-library-benchmark.md).

## Available Models

| Module | Models |
|--------|--------|
| `linear` | LinearRegression, LogisticRegression, RidgeRegression, LassoRegression, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier, QuantileRegression, RobustRegression, Perceptron, IsotonicRegression |
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
| `calibration` | TemperatureScalingCalibrator (Sigmoid/Platt and Isotonic calibration are available in the Rust crate) |
| `pipeline` | Pipeline, ColumnTransformer, FeatureUnion |
| `model_selection` | train_test_split, cross_validate, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit |
| `metrics` | ROC-AUC, F1, MCC, R², RMSE, MAE, roc_curve, precision_recall_curve |
| `automl` | AutoML with statistical model comparison |
| `datasets` | Iris, Diabetes, Wine, California Housing, synthetic generators |

## sklearn API Compatibility

FerroML supports the scikit-learn API conventions:

- `fit()` / `predict()` / `transform()` on all models
- `score()` on 56 models (R² for regressors, accuracy for classifiers)
- `partial_fit()` on 10 models for incremental learning
- `decision_function()` on 13 classifiers
- `predict_proba()` on probabilistic classifiers
- Pipeline and ColumnTransformer composition
- NumPy array input/output

## Testing

8,196 tests passing at 1.0.2 (5,225 Rust + 2,971 Python; [recorded run](https://github.com/robertlupo1997/ferroml/blob/master/docs/test-runs/SUMMARY.md)), validated against scikit-learn, scipy, xgboost, lightgbm, statsmodels and linfa with 489+ cross-library correctness tests.

## License

MIT OR Apache-2.0
