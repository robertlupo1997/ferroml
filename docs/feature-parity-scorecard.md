# FerroML Feature Parity Scorecard

*Generated: 2026-03-14 16:48*

**Legend:** Y = supported, GAP = sklearn has it / FerroML missing, Y+ = FerroML-only, - = neither has it

## Summary

- **FerroML models:** 94
- **sklearn equivalents checked:** 87
- **Models in both:** 87
- **Total gaps:** 132

## Top Missing Features (by Impact)

1. **`score`** — missing in 57 models: LinearRegression, RidgeRegression, LassoRegression, ElasticNet, RidgeCV, ... (+52 more)
2. **`partial_fit`** — missing in 16 models: Perceptron, SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, GaussianNB, ... (+11 more)
3. **`decision_function`** — missing in 13 models: LogisticRegression, RidgeClassifier, Perceptron, GradientBoostingClassifier, HistGradientBoostingClassifier, ... (+8 more)
4. **`inverse_transform`** — missing in 10 models: IncrementalPCA, TruncatedSVD, SimpleImputer, KBinsDiscretizer, VarianceThreshold, ... (+5 more)
5. **`fit_transform`** — missing in 9 models: IsotonicRegression, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, ... (+4 more)

## Method Parity Summary

| Method | FerroML has | sklearn has | Parity | Gap |
|--------|------------|------------|--------|-----|
| `fit` | 90 | 87 | 87 | 0 |
| `predict` | 60 | 59 | 56 | 3 |
| `predict_proba` | 20 | 26 | 19 | 7 |
| `predict_log_proba` | 18 | 18 | 11 | 7 |
| `decision_function` | 7 | 18 | 5 | 13 |
| `score` | 2 | 59 | 2 | 57 |
| `transform` | 28 | 32 | 26 | 6 |
| `fit_transform` | 24 | 33 | 24 | 9 |
| `inverse_transform` | 10 | 20 | 10 | 10 |
| `partial_fit` | 1 | 17 | 1 | 16 |
| `feature_importances_` | 12 | 10 | 8 | 2 |
| `coef_` | 9 | 2 | 0 | 2 |
| `intercept_` | 10 | 0 | 0 | 0 |
| `to_onnx_bytes` | 34 | 0 | 0 | 0 |
| `export_onnx` | 34 | 0 | 0 | 0 |
| `warm_start` | 0 | 23 | 0 | 23 |
| `get_params` | 0 | 87 | 0 | 87 |
| `set_params` | 0 | 87 | 0 | 87 |

## Models

### Linear

| Model | fit | pred | proba | log_p | dec_fn | score | xform | fit_x | p_fit | coef | icpt | Extras |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LinearRegression | Y | Y | - | - | - | GAP | - | - | - | Y+ | Y+ | summary, predict_interval, fit_sparse +6 |
| RidgeRegression | Y | Y | - | - | - | GAP | - | - | - | Y+ | Y+ | summary, fit_sparse, predict_sparse +4 |
| LassoRegression | Y | Y | - | - | - | GAP | - | - | - | Y+ | Y+ | summary, fit_sparse, predict_sparse +4 |
| ElasticNet | Y | Y | - | - | - | GAP | - | - | - | Y+ | Y+ | summary, fit_sparse, predict_sparse +4 |
| RidgeCV | Y | Y | - | - | - | GAP | - | - | - | - | - |  |
| LassoCV | Y | Y | - | - | - | GAP | - | - | - | - | - |  |
| ElasticNetCV | Y | Y | - | - | - | GAP | - | - | - | - | - |  |
| RobustRegression | Y | Y | - | - | - | GAP | - | - | - | Y+ | Y+ |  |
| QuantileRegression | Y | Y | - | - | - | GAP | - | - | - | Y+ | Y+ |  |
| IsotonicRegression | Y | Y | - | - | - | GAP | GAP | GAP | - | - | - |  |
| LogisticRegression | Y | Y | Y | Y | GAP | GAP | - | - | - | Y+ | Y+ | summary, fit_sparse, predict_sparse +4 |
| RidgeClassifier | Y | Y | - | - | GAP | GAP | - | - | - | - | - |  |
| Perceptron | Y | Y | - | - | GAP | GAP | - | - | GAP | - | - |  |

### Trees

| Model | fit | pred | proba | log_p | dec_fn | score | f_imp | Extras |
|---|---|---|---|---|---|---|---|---|
| DecisionTreeClassifier | Y | Y | Y | Y | - | GAP | Y |  |
| DecisionTreeRegressor | Y | Y | - | - | - | GAP | Y |  |
| RandomForestClassifier | Y | Y | Y | Y | - | GAP | Y |  |
| RandomForestRegressor | Y | Y | - | - | - | GAP | Y |  |
| GradientBoostingClassifier | Y | Y | Y | Y | GAP | GAP | Y |  |
| GradientBoostingRegressor | Y | Y | - | - | - | GAP | Y |  |
| HistGradientBoostingClassifier | Y | Y | Y | Y+ | GAP | GAP | Y+ |  |
| HistGradientBoostingRegressor | Y | Y | - | - | - | GAP | Y+ |  |

### Ensemble

| Model | fit | pred | proba | log_p | dec_fn | score | xform | fit_x | p_fit | f_imp | coef | icpt | Extras |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ExtraTreesClassifier | Y | Y | GAP | GAP | - | GAP | - | - | - | Y | - | - |  |
| ExtraTreesRegressor | Y | Y | - | - | - | GAP | - | - | - | Y | - | - |  |
| AdaBoostClassifier | Y | Y | GAP | GAP | GAP | GAP | - | - | - | GAP | - | - |  |
| AdaBoostRegressor | Y | Y | - | - | - | GAP | - | - | - | GAP | - | - |  |
| BaggingClassifier | Y | Y | Y | Y | GAP | GAP | - | - | - | Y+ | - | - |  |
| BaggingRegressor | Y | Y | - | - | - | GAP | - | - | - | Y+ | - | - |  |
| VotingClassifier | Y | Y | Y | Y+ | - | GAP | GAP | GAP | - | - | - | - |  |
| VotingRegressor | Y | Y | - | - | - | GAP | GAP | GAP | - | - | - | - |  |
| StackingClassifier | Y | Y | Y | Y+ | GAP | GAP | GAP | GAP | - | - | - | - |  |
| StackingRegressor | Y | Y | - | - | - | GAP | GAP | GAP | - | - | - | - |  |
| SGDClassifier | Y | Y | GAP | GAP | GAP | GAP | - | - | GAP | - | - | - |  |
| SGDRegressor | Y | Y | - | - | - | GAP | - | - | GAP | - | Y+ | Y+ |  |
| PassiveAggressiveClassifier | Y | Y | - | - | GAP | GAP | - | - | GAP | - | - | - |  |

### Neighbors

| Model | fit | pred | proba | log_p | dec_fn | score | Extras |
|---|---|---|---|---|---|---|---|
| KNeighborsClassifier | Y | Y | Y | Y+ | - | GAP |  |
| KNeighborsRegressor | Y | Y | - | - | - | GAP |  |
| NearestCentroid | Y | Y | GAP | GAP | GAP | GAP |  |

### SVM

| Model | fit | pred | proba | log_p | dec_fn | score | coef | icpt | Extras |
|---|---|---|---|---|---|---|---|---|---|
| LinearSVC | Y | Y | - | - | Y | GAP | - | - | fit_sparse, predict_sparse |
| LinearSVR | Y | Y | - | - | Y+ | GAP | Y+ | Y+ | fit_sparse, predict_sparse |
| SVC | Y | Y | Y | Y | Y | GAP | GAP | - |  |
| SVR | Y | Y | - | - | Y+ | GAP | GAP | Y+ |  |

### Naive Bayes

| Model | fit | pred | proba | log_p | score | p_fit | Extras |
|---|---|---|---|---|---|---|---|
| GaussianNB | Y | Y | Y | Y | GAP | GAP |  |
| MultinomialNB | Y | Y | Y | Y | GAP | GAP | fit_sparse, predict_sparse |
| BernoulliNB | Y | Y | Y | Y | GAP | GAP | fit_sparse, predict_sparse |
| CategoricalNB | Y | Y | Y | Y | GAP | Y |  |

### Neural

| Model | fit | pred | proba | log_p | score | p_fit | Extras |
|---|---|---|---|---|---|---|---|
| MLPClassifier | Y | Y | Y | GAP | GAP | GAP |  |
| MLPRegressor | Y | Y | - | - | Y | GAP |  |

### Clustering

| Model | fit | pred | proba | log_p | score | xform | fit_x | Extras |
|---|---|---|---|---|---|---|---|---|
| KMeans | Y | Y | - | - | GAP | GAP | GAP |  |
| DBSCAN | Y | Y+ | - | - | - | - | - |  |
| AgglomerativeClustering | Y | - | - | - | - | - | - |  |
| GaussianMixture | Y | Y | Y | Y+ | Y | - | - |  |
| HDBSCAN | Y | - | - | - | - | - | - |  |

### Decomposition

| Model | fit | pred | proba | log_p | dec_fn | score | xform | fit_x | inv_x | p_fit | Extras |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PCA | Y | - | - | - | - | GAP | Y | Y | Y | - |  |
| IncrementalPCA | Y | - | - | - | - | - | Y | Y | GAP | GAP |  |
| TruncatedSVD | Y | - | - | - | - | - | Y | Y | GAP | - |  |
| LDA | Y | GAP | GAP | GAP | GAP | GAP | Y | GAP | - | - |  |
| QuadraticDiscriminantAnalysis | Y | Y | Y | Y | Y | GAP | - | - | - | - |  |
| FactorAnalysis | Y | - | - | - | - | GAP | Y | Y | - | - |  |
| TSNE | Y | - | - | - | - | - | Y+ | Y | - | - |  |

### Anomaly

| Model | fit | pred | dec_fn | Extras |
|---|---|---|---|---|
| IsolationForest | Y | Y | Y |  |
| LocalOutlierFactor | Y | Y | Y |  |

### Gaussian Process

| Model | fit | pred | proba | log_p | score | Extras |
|---|---|---|---|---|---|---|
| GaussianProcessRegressor | Y | Y | - | - | GAP | predict_with_std |
| GaussianProcessClassifier | Y | Y | Y | Y+ | GAP |  |
| SparseGPRegressor * | Y | Y | - | - | - | predict_with_std |
| SparseGPClassifier * | Y | Y | Y | Y | - |  |
| SVGPRegressor * | Y | Y | - | - | - | predict_with_std |

### Calibration

| Model | fit | pred | proba | score | xform | Extras |
|---|---|---|---|---|---|---|
| TemperatureScalingCalibrator | Y | GAP | GAP | GAP | Y+ |  |

### MultiOutput

| Model | fit | pred | proba | score | p_fit | Extras |
|---|---|---|---|---|---|---|
| MultiOutputRegressor | Y | Y | - | GAP | GAP |  |
| MultiOutputClassifier | Y | Y | Y | GAP | GAP |  |

## Preprocessors

### Preprocessing

| Model | fit | pred | proba | log_p | dec_fn | score | xform | fit_x | inv_x | p_fit | Extras |
|---|---|---|---|---|---|---|---|---|---|---|---|
| StandardScaler | Y | - | - | - | - | - | Y | Y | Y | GAP |  |
| MinMaxScaler | Y | - | - | - | - | - | Y | Y | Y | GAP |  |
| RobustScaler | Y | - | - | - | - | - | Y | Y | Y | - |  |
| MaxAbsScaler | Y | - | - | - | - | - | Y | Y | Y | GAP |  |
| OneHotEncoder | Y | - | - | - | - | - | Y | Y | Y | - |  |
| OrdinalEncoder | Y | - | - | - | - | - | Y | Y | Y | - |  |
| LabelEncoder | Y | - | - | - | - | - | Y | Y | Y | - |  |
| TargetEncoder | Y | - | - | - | - | - | Y | GAP | - | - |  |
| SimpleImputer | Y | - | - | - | - | - | Y | Y | GAP | - |  |
| KNNImputer | Y | - | - | - | - | - | Y | Y | - | - |  |
| PowerTransformer | Y | - | - | - | - | - | Y | Y | Y | - |  |
| QuantileTransformer | Y | - | - | - | - | - | Y | Y | Y | - |  |
| PolynomialFeatures | Y | - | - | - | - | - | Y | Y | - | - |  |
| KBinsDiscretizer | Y | - | - | - | - | - | Y | Y | GAP | - |  |
| VarianceThreshold | Y | - | - | - | - | - | Y | Y | GAP | - |  |
| SelectKBest | Y | - | - | - | - | - | Y | GAP | GAP | - |  |
| SelectFromModel | Y | - | - | - | - | - | Y | Y | GAP | GAP |  |
| RecursiveFeatureElimination | Y | GAP | GAP | GAP | GAP | GAP | Y | Y | GAP | - |  |

### Text

| Model | fit | xform | fit_x | inv_x | Extras |
|---|---|---|---|---|---|
| CountVectorizer | Y | Y | Y | GAP |  |
| TfidfTransformer | Y | Y | Y | - | fit_sparse, transform_sparse |
| TfidfVectorizer | Y | Y | Y | GAP |  |

### Sampling

| Model |  | Extras |
|---|---|
| SMOTE * |  |  |
| ADASYN * |  |  |
| RandomUnderSampler * |  |  |
| RandomOverSampler * |  |  |

---

*Models marked with `*` have no sklearn equivalent.*
