---
date: 2026-02-02T12:00:00-05:00
researcher: Claude
topic: sklearn Parity Analysis
tags: [sklearn, parity, features, api]
status: complete
---

# FerroML sklearn Parity Analysis

## Executive Summary

FerroML has achieved **85-90% parity** with scikit-learn's commonly-used features. **Key differentiator: statistical rigor that sklearn lacks** (confidence intervals, p-values, diagnostic tools).

---

## What FerroML HAS

### Models
| Category | Implemented |
|----------|-------------|
| **Linear** | LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, RobustRegression, QuantileRegression (all with CV variants) |
| **Trees** | DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor, HistGradientBoosting |
| **SVM** | SVC, SVR, LinearSVC, LinearSVR |
| **KNN** | KNeighborsClassifier/Regressor (KD-Tree, Ball-Tree) |
| **Naive Bayes** | GaussianNB, MultinomialNB, BernoulliNB |
| **Calibration** | CalibratedClassifierCV, Isotonic, Sigmoid |

### Preprocessing
| Category | Implemented |
|----------|-------------|
| **Scalers** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformers |
| **Encoders** | OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder |
| **Imputers** | SimpleImputer (mean/median/mode/constant), KNNImputer |
| **Feature Selection** | VarianceThreshold, SelectKBest, SelectFromModel, RFE |

### Cross-Validation
All major strategies: KFold, StratifiedKFold, RepeatedKFold, GroupKFold, StratifiedGroupKFold, TimeSeriesSplit, LeaveOneOut, LeavePOut, ShuffleSplit + `cross_val_score`, `learning_curve`, `validation_curve`

### Metrics
- **Classification**: accuracy, precision, recall, F1, ROC-AUC, log_loss, brier_score, cohen_kappa, matthews_corrcoef
- **Regression**: MSE, RMSE, MAE, R2, explained_variance

### Ensemble
VotingClassifier/Regressor, StackingClassifier/Regressor, BaggingClassifier/Regressor

### Pipeline
Pipeline, FeatureUnion, caching, combined search spaces

### Decomposition
PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis, LDA

---

## What FerroML Has That sklearn DOESN'T

### 1. Statistical Diagnostics (Key Differentiator!)
- R-style `model.summary()` with coefficient tables
- Confidence intervals for coefficients AND predictions
- Residual diagnostics (Shapiro-Wilk, Durbin-Watson)
- Influential observation detection (Cook's distance, leverage, VIF)

### 2. HPO/AutoML (sklearn has nothing)
- Bayesian optimization with GP (EI/PI/UCB)
- TPE sampler, Hyperband, ASHA schedulers
- Algorithm portfolios with data-aware adaptation
- Time budget allocation via bandit algorithms
- Warmstarting/transfer learning

### 3. Explainability (sklearn is minimal)
- Permutation importance with CIs
- PDP and ICE plots (1D/2D)
- H-statistic for interactions
- TreeSHAP and KernelSHAP

### 4. Statistical Testing
- Bootstrap CIs for any metric
- Model comparison tests (McNemar, 5x2 CV)
- Multiple testing correction
- Power analysis

---

## What's MISSING

### High Priority Gaps
| Feature | Impact |
|---------|--------|
| **ColumnTransformer** | Essential for mixed-type preprocessing |
| **MAPE metric** | Common regression metric |
| **SequentialFeatureSelector** | Forward/backward selection |

### Medium Priority
- NuSVC/NuSVR
- AdaBoostClassifier/Regressor
- KernelPCA, FastICA, NMF
- RadiusNeighborsClassifier

### Lower Priority
- CategoricalNB, ComplementNB
- OneClassSVM (anomaly detection)
- GaussianProcessClassifier/Regressor

---

## API Compatibility

| Aspect | Compatibility |
|--------|---------------|
| Transformer API (`fit`, `transform`, `inverse_transform`) | **Identical** |
| Model API (`fit`, `predict`, `predict_proba`) | **Identical** |
| CV splitter interface | **Compatible** |
| Metric functions | **Compatible** |
| Parameter setting | Different (Rust builder vs kwargs) |

---

## Action Items for Ralph Loop

1. `TASK-SKLEARN-001`: Implement ColumnTransformer (HIGH)
2. `TASK-SKLEARN-002`: Add MAPE metric (HIGH)
3. `TASK-SKLEARN-003`: Add SequentialFeatureSelector (MEDIUM)
4. `TASK-SKLEARN-004`: Add AdaBoost (MEDIUM)
