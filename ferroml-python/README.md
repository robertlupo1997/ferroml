# FerroML

Statistically rigorous AutoML in Rust with Python bindings.

## Installation

```bash
pip install ferroml
```

## Quick Start

```python
from ferroml.linear import LinearRegression
from ferroml.automl import AutoML, AutoMLConfig
import numpy as np

# Linear Regression with full statistical diagnostics
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

model = LinearRegression()
model.fit(X, y)
print(model.summary())  # R-style statistical output

# AutoML with statistical significance testing
config = AutoMLConfig(
    task="classification",
    metric="roc_auc",
    time_budget_seconds=300,
)
automl = AutoML(config)
result = automl.fit(X_train, y_train)
print(result.summary())
```

## Available Modules

| Module | Models / Features |
|--------|-------------------|
| `linear` | LinearRegression, LogisticRegression, RidgeRegression, LassoRegression, ElasticNet, RidgeCV, LassoCV, ElasticNetCV, RidgeClassifier, QuantileRegression, RobustRegression, Perceptron |
| `trees` | DecisionTreeClassifier, DecisionTreeRegressor |
| `ensemble` | RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor, SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, BaggingClassifier, BaggingRegressor |
| `naive_bayes` | GaussianNB, MultinomialNB, BernoulliNB |
| `svm` | LinearSVC, LinearSVR |
| `neighbors` | KNeighborsClassifier, KNeighborsRegressor, NearestCentroid |
| `neural` | MLPClassifier, MLPRegressor |
| `clustering` | KMeans, DBSCAN, AgglomerativeClustering |
| `preprocessing` | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder, SimpleImputer, KNNImputer, PolynomialFeatures, PowerTransformer, QuantileTransformer, VarianceThreshold, SelectKBest, SelectFromModel, RecursiveFeatureElimination, KBinsDiscretizer, SMOTE, BorderlineSMOTE, ADASYN, RandomUnderSampler |
| `decomposition` | PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis |
| `explainability` | TreeSHAP, KernelSHAP, permutation importance, PDP, 2D PDP, ICE, H-statistic |
| `calibration` | TemperatureScalingCalibrator |
| `pipeline` | Pipeline, ColumnTransformer, FeatureUnion |
| `automl` | AutoML with statistical model comparison |
| `datasets` | Iris, Diabetes, Wine, synthetic generators |

## Features

- **Statistical Rigor**: Confidence intervals, effect sizes, and assumption tests
- **Linear Models**: Full diagnostics (R-style summary, VIF, residual analysis)
- **Tree Models**: RandomForest, GradientBoosting, HistGradientBoosting, ExtraTrees, AdaBoost
- **Clustering**: KMeans, DBSCAN, AgglomerativeClustering with full metrics
- **Neural Networks**: MLPClassifier, MLPRegressor with training diagnostics
- **Preprocessing**: 24+ transformers including SMOTE/ADASYN resampling
- **Explainability**: TreeSHAP, permutation importance, PDP, ICE, H-statistic
- **AutoML**: Automatic model selection with statistical significance testing

## License

MIT OR Apache-2.0
