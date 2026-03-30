# FerroML API Cheatsheet

Complete API reference organized by submodule. Every class and function in `import ferroml`.

## ferroml.linear

| Class | Constructor params | Key methods | Quick usage |
|-------|-------------------|-------------|-------------|
| `LinearRegression` | `fit_intercept=True` | `fit(X,y)`, `predict(X)`, `summary()`, `coefficients_with_ci()`, `r_squared()`, `f_statistic()`, `aic()`, `bic()` | `m = LinearRegression(); m.fit(X, y); m.summary()` |
| `LogisticRegression` | `C=1.0, max_iter=100, solver="lbfgs", penalty="l2"` | `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `summary()`, `coefficients_with_ci()` | `m = LogisticRegression(C=0.1); m.fit(X, y)` |
| `RidgeRegression` | `alpha=1.0, fit_intercept=True` | `fit(X,y)`, `predict(X)`, `summary()`, `coefficients_with_ci()` | `m = RidgeRegression(alpha=0.5); m.fit(X, y)` |
| `LassoRegression` | `alpha=1.0, max_iter=1000` | `fit(X,y)`, `predict(X)`, `summary()` | `m = LassoRegression(alpha=0.01); m.fit(X, y)` |
| `ElasticNet` | `alpha=1.0, l1_ratio=0.5, max_iter=1000` | `fit(X,y)`, `predict(X)`, `summary()` | `m = ElasticNet(l1_ratio=0.7); m.fit(X, y)` |
| `RobustRegression` | `max_iter=100` | `fit(X,y)`, `predict(X)`, `summary()` | `m = RobustRegression(); m.fit(X, y)` |
| `QuantileRegression` | `quantile=0.5, alpha=0.0` | `fit(X,y)`, `predict(X)` | `m = QuantileRegression(quantile=0.9); m.fit(X, y)` |
| `Perceptron` | `max_iter=1000, eta0=1.0` | `fit(X,y)`, `predict(X)` | `m = Perceptron(); m.fit(X, y)` |
| `RidgeCV` | `alphas=[0.1,1.0,10.0]` | `fit(X,y)`, `predict(X)`, `best_alpha()` | `m = RidgeCV(); m.fit(X, y); m.best_alpha()` |
| `LassoCV` | `alphas=None, cv=5` | `fit(X,y)`, `predict(X)`, `best_alpha()` | `m = LassoCV(); m.fit(X, y)` |
| `ElasticNetCV` | `l1_ratio=0.5, cv=5` | `fit(X,y)`, `predict(X)`, `best_alpha()` | `m = ElasticNetCV(); m.fit(X, y)` |
| `RidgeClassifier` | `alpha=1.0` | `fit(X,y)`, `predict(X)` | `m = RidgeClassifier(); m.fit(X, y)` |
| `IsotonicRegression` | `increasing=True` | `fit(X,y)`, `predict(X)` | `m = IsotonicRegression(); m.fit(x, y)` |

## ferroml.trees

**Note:** Gradient boosting models live here, NOT in `ferroml.ensemble`.

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `DecisionTreeClassifier` | `max_depth=None, min_samples_split=2, criterion="gini"` | `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `feature_importances()` |
| `DecisionTreeRegressor` | `max_depth=None, min_samples_split=2, criterion="mse"` | `fit(X,y)`, `predict(X)`, `feature_importances()` |
| `RandomForestClassifier` | `n_estimators=100, max_depth=None, n_jobs=1, random_state=None` | `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `feature_importances()` |
| `RandomForestRegressor` | `n_estimators=100, max_depth=None, n_jobs=1` | `fit(X,y)`, `predict(X)`, `feature_importances()` |
| `GradientBoostingClassifier` | `n_estimators=100, learning_rate=0.1, max_depth=3` | `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `feature_importances()` |
| `GradientBoostingRegressor` | `n_estimators=100, learning_rate=0.1, max_depth=3` | `fit(X,y)`, `predict(X)`, `feature_importances()` |
| `HistGradientBoostingClassifier` | `max_iter=100, learning_rate=0.1, max_depth=None, max_bins=255` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `HistGradientBoostingRegressor` | `max_iter=100, learning_rate=0.1, max_depth=None, max_bins=255` | `fit(X,y)`, `predict(X)` |

## ferroml.ensemble

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `ExtraTreesClassifier` | `n_estimators=100, max_depth=None` | `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `feature_importances()` |
| `ExtraTreesRegressor` | `n_estimators=100, max_depth=None` | `fit(X,y)`, `predict(X)`, `feature_importances()` |
| `AdaBoostClassifier` | `n_estimators=50, learning_rate=1.0` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `AdaBoostRegressor` | `n_estimators=50, learning_rate=1.0` | `fit(X,y)`, `predict(X)` |
| `SGDClassifier` | `loss="hinge", alpha=0.0001, max_iter=1000, learning_rate="optimal"` | `fit(X,y)`, `predict(X)`, `partial_fit(X,y)` |
| `SGDRegressor` | `loss="squared_error", alpha=0.0001, max_iter=1000` | `fit(X,y)`, `predict(X)`, `partial_fit(X,y)` |
| `PassiveAggressiveClassifier` | `C=1.0, max_iter=1000` | `fit(X,y)`, `predict(X)`, `partial_fit(X,y)` |
| `BaggingClassifier` | `base_estimator="decision_tree", n_estimators=10` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `BaggingRegressor` | `base_estimator="decision_tree", n_estimators=10` | `fit(X,y)`, `predict(X)` |
| `VotingClassifier` | `estimators=[...], voting="hard"` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` (soft voting) |
| `VotingRegressor` | `estimators=[...]` | `fit(X,y)`, `predict(X)` |
| `StackingClassifier` | `estimators=[...], final_estimator=None` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `StackingRegressor` | `estimators=[...], final_estimator=None` | `fit(X,y)`, `predict(X)` |

## ferroml.neighbors

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `KNeighborsClassifier` | `n_neighbors=5, metric="euclidean"` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `KNeighborsRegressor` | `n_neighbors=5, metric="euclidean"` | `fit(X,y)`, `predict(X)` |
| `NearestCentroid` | `metric="euclidean"` | `fit(X,y)`, `predict(X)` |

## ferroml.naive_bayes

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `GaussianNB` | `var_smoothing=1e-9` | `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `partial_fit(X,y)` |
| `MultinomialNB` | `alpha=1.0` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `BernoulliNB` | `alpha=1.0, binarize=0.0` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `CategoricalNB` | `alpha=1.0` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |

## ferroml.svm

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `LinearSVC` | `C=1.0, max_iter=1000` | `fit(X,y)`, `predict(X)`, `decision_function(X)` |
| `LinearSVR` | `C=1.0, epsilon=0.1, max_iter=1000` | `fit(X,y)`, `predict(X)` |
| `SVC` | `C=1.0, kernel="rbf", gamma="scale"` | `fit(X,y)`, `predict(X)`, `predict_proba(X)`, `decision_function(X)` |
| `SVR` | `C=1.0, kernel="rbf", gamma="scale", epsilon=0.1` | `fit(X,y)`, `predict(X)` |

## ferroml.neural

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `MLPClassifier` | `hidden_layer_sizes=[100], activation="relu", max_iter=200, learning_rate_init=0.001` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `MLPRegressor` | `hidden_layer_sizes=[100], activation="relu", max_iter=200, learning_rate_init=0.001` | `fit(X,y)`, `predict(X)` |

## ferroml.gaussian_process

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `GaussianProcessRegressor` | `kernel=None, alpha=1e-10, n_restarts=0` | `fit(X,y)`, `predict(X)`, `predict_with_uncertainty(X, confidence=0.95)` |
| `GaussianProcessClassifier` | `kernel=None, n_restarts=0` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `SparseGPRegressor` | `kernel=None, n_inducing=50` | `fit(X,y)`, `predict(X)`, `predict_with_uncertainty(X, confidence=0.95)` |
| `SparseGPClassifier` | `kernel=None, n_inducing=50` | `fit(X,y)`, `predict(X)`, `predict_proba(X)` |
| `SVGPRegressor` | `kernel=None, n_inducing=50` | `fit(X,y)`, `predict(X)` |

### Kernels

| Kernel | Params | Usage |
|--------|--------|-------|
| `RBF` | `length_scale=1.0` | `RBF(length_scale=1.0)` |
| `Matern` | `length_scale=1.0, nu=1.5` | `Matern(nu=2.5)` |
| `ConstantKernel` | `constant_value=1.0` | `ConstantKernel(1.0)` |
| `WhiteKernel` | `noise_level=1.0` | `WhiteKernel(0.1)` |

## ferroml.clustering

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `KMeans` | `n_clusters=8, max_iter=300, n_init=10, random_state=None` | `fit(X)`, `predict(X)`, `labels()`, `cluster_centers()`, `inertia()` |
| `MiniBatchKMeans` | `n_clusters=8, batch_size=100, max_iter=100` | `fit(X)`, `predict(X)`, `partial_fit(X)`, `labels()` |
| `DBSCAN` | `eps=0.5, min_samples=5` | `fit(X)`, `labels()` |
| `AgglomerativeClustering` | `n_clusters=2, linkage="ward"` | `fit(X)`, `labels()` |
| `GaussianMixture` | `n_components=1, max_iter=100, covariance_type="full"` | `fit(X)`, `predict(X)`, `predict_proba(X)`, `aic(X)`, `bic(X)` |
| `HDBSCAN` | `min_cluster_size=5, min_samples=None` | `fit(X)`, `labels()`, `probabilities()` |

### Clustering metrics (functions)

| Function | Signature | Returns |
|----------|-----------|---------|
| `silhouette_score` | `(X, labels)` | `float` (-1 to 1, higher is better) |
| `calinski_harabasz_score` | `(X, labels)` | `float` (higher is better) |
| `davies_bouldin_score` | `(X, labels)` | `float` (lower is better) |
| `adjusted_rand_index` | `(labels_true, labels_pred)` | `float` (-1 to 1) |
| `normalized_mutual_info` | `(labels_true, labels_pred)` | `float` (0 to 1) |
| `hopkins_statistic` | `(X)` | `float` (>0.7 suggests clusters exist) |

## ferroml.anomaly

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `IsolationForest` | `n_estimators=100, contamination=0.1, random_state=None` | `fit(X)`, `predict(X)`, `score_samples(X)` |
| `LocalOutlierFactor` | `n_neighbors=20, contamination=0.1` | `fit(X)`, `predict(X)`, `score_samples(X)` |

## ferroml.decomposition

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `PCA` | `n_components=None` | `fit(X)`, `transform(X)`, `fit_transform(X)`, `explained_variance_ratio()`, `inverse_transform(X)` |
| `IncrementalPCA` | `n_components=None, batch_size=None` | `fit(X)`, `transform(X)`, `partial_fit(X)` |
| `TruncatedSVD` | `n_components=2` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `LDA` | `n_components=None` | `fit(X, y)`, `transform(X)`, `fit_transform(X, y)` |
| `QuadraticDiscriminantAnalysis` | `reg_param=0.0` | `fit(X, y)`, `predict(X)`, `predict_proba(X)` |
| `FactorAnalysis` | `n_components=None, max_iter=1000` | `fit(X)`, `transform(X)` |
| `TSNE` | `n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000` | `fit_transform(X)` |

## ferroml.preprocessing

### Scalers

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `StandardScaler` | `with_mean=True, with_std=True` | `fit(X)`, `transform(X)`, `fit_transform(X)`, `inverse_transform(X)` |
| `MinMaxScaler` | `feature_range=(0.0, 1.0)` | `fit(X)`, `transform(X)`, `fit_transform(X)`, `inverse_transform(X)` |
| `RobustScaler` | `with_centering=True, with_scaling=True` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `MaxAbsScaler` | — | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `Normalizer` | `norm="l2"` | `transform(X)` |
| `PowerTransformer` | `method="yeo-johnson"` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `QuantileTransformer` | `n_quantiles=1000, output_distribution="uniform"` | `fit(X)`, `transform(X)`, `fit_transform(X)` |

### Encoders

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `OneHotEncoder` | `sparse=False` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `OrdinalEncoder` | — | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `LabelEncoder` | — | `fit(y)`, `transform(y)`, `fit_transform(y)`, `inverse_transform(y)` |
| `TargetEncoder` | `smooth=1.0` | `fit(X, y)`, `transform(X)` |

### Imputation

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `SimpleImputer` | `strategy="mean"` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `KNNImputer` | `n_neighbors=5` | `fit(X)`, `transform(X)`, `fit_transform(X)` |

### Feature selection

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `VarianceThreshold` | `threshold=0.0` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `SelectKBest` | `k=10` | `fit(X, y)`, `transform(X)`, `scores()` |
| `SelectFromModel` | `estimator, threshold=None` | `fit(X, y)`, `transform(X)` |
| `RecursiveFeatureElimination` | `estimator, n_features_to_select=None` | `fit(X, y)`, `transform(X)`, `ranking()` |

### Feature engineering

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `PolynomialFeatures` | `degree=2, interaction_only=False, include_bias=True` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `KBinsDiscretizer` | `n_bins=5, strategy="quantile", encode="ordinal"` | `fit(X)`, `transform(X)`, `fit_transform(X)` |

### Sampling

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `SMOTE` | `k_neighbors=5, random_state=None` | `fit_resample(X, y)` |
| `ADASYN` | `n_neighbors=5, random_state=None` | `fit_resample(X, y)` |
| `RandomUnderSampler` | `random_state=None` | `fit_resample(X, y)` |
| `RandomOverSampler` | `random_state=None` | `fit_resample(X, y)` |

### Text

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `TfidfTransformer` | `norm="l2", use_idf=True` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `CountVectorizer` | `max_features=None, min_df=1, max_df=1.0` | `fit(texts)`, `transform(texts)`, `fit_transform(texts)`, `vocabulary()` |
| `TfidfVectorizer` | `max_features=None, min_df=1, max_df=1.0` | `fit(texts)`, `transform(texts)`, `fit_transform(texts)` |

## ferroml.metrics

### Classification metrics

| Function | Signature | Returns |
|----------|-----------|---------|
| `accuracy_score` | `(y_true, y_pred)` | `float` |
| `precision_score` | `(y_true, y_pred, average="binary")` | `float` |
| `recall_score` | `(y_true, y_pred, average="binary")` | `float` |
| `f1_score` | `(y_true, y_pred, average="binary")` | `float` |
| `matthews_corrcoef` | `(y_true, y_pred)` | `float` (-1 to 1) |
| `cohen_kappa_score` | `(y_true, y_pred)` | `float` (-1 to 1) |
| `balanced_accuracy_score` | `(y_true, y_pred)` | `float` |
| `confusion_matrix` | `(y_true, y_pred)` | `list[list[int]]` |
| `classification_report` | `(y_true, y_pred)` | `str` |

### Regression metrics

| Function | Signature | Returns |
|----------|-----------|---------|
| `mse` | `(y_true, y_pred)` | `float` |
| `rmse` | `(y_true, y_pred)` | `float` |
| `mae` | `(y_true, y_pred)` | `float` |
| `r2_score` | `(y_true, y_pred)` | `float` |
| `explained_variance` | `(y_true, y_pred)` | `float` |
| `max_error` | `(y_true, y_pred)` | `float` |
| `mape` | `(y_true, y_pred)` | `float` |
| `median_absolute_error` | `(y_true, y_pred)` | `float` |

### Probabilistic metrics

| Function | Signature | Returns |
|----------|-----------|---------|
| `roc_auc_score` | `(y_true, y_score)` | `float` (0 to 1) |
| `pr_auc_score` | `(y_true, y_score)` | `float` (0 to 1) |
| `average_precision_score` | `(y_true, y_score)` | `float` (0 to 1) |
| `log_loss` | `(y_true, y_pred_proba)` | `float` (lower is better) |
| `brier_score` | `(y_true, y_pred_proba)` | `float` (0 to 1, lower is better) |
| `brier_skill_score` | `(y_true, y_pred_proba)` | `float` (<0 worse than climatology) |

### Curves

| Function | Signature | Returns |
|----------|-----------|---------|
| `roc_curve` | `(y_true, y_score)` | `(fpr, tpr, thresholds)` |
| `precision_recall_curve` | `(y_true, y_score)` | `(precision, recall, thresholds)` |

### Model comparison

| Function | Signature | Returns |
|----------|-----------|---------|
| `paired_ttest` | `(scores_a, scores_b)` | `(t_stat, p_value)` |
| `corrected_resampled_ttest` | `(scores_a, scores_b, n_train, n_test)` | `(t_stat, p_value)` |
| `mcnemar_test` | `(y_true, pred_a, pred_b)` | `(stat, p_value)` |
| `wilcoxon_test` | `(scores_a, scores_b)` | `(stat, p_value)` |

## ferroml.stats

| Function | Signature | Returns |
|----------|-----------|---------|
| `ttest_ind` | `(a, b)` | `(t_stat, p_value)` |
| `welch_ttest` | `(a, b)` | `(t_stat, p_value)` |
| `mann_whitney` | `(a, b)` | `(u_stat, p_value)` |
| `cohens_d` | `(a, b)` | `float` |
| `hedges_g` | `(a, b)` | `float` |
| `confidence_interval` | `(data, confidence=0.95)` | `(lower, upper)` |
| `bootstrap_ci` | `(data, n_bootstrap=1000, confidence=0.95)` | `(lower, upper)` |
| `adjust_pvalues` | `(p_values, method="bonferroni")` | `list[float]` |
| `sample_size_for_power` | `(effect_size, alpha=0.05, power=0.8)` | `int` |
| `power_for_sample_size` | `(effect_size, n, alpha=0.05)` | `float` |
| `durbin_watson` | `(residuals)` | `float` (0 to 4) |
| `descriptive_stats` | `(data)` | `dict` |
| `normality_test` | `(data)` | `(stat, p_value)` |
| `correlation` | `(a, b)` | `float` |

## ferroml.pipeline

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `Pipeline` | `steps=[("name", transformer_or_model), ...]` | `fit(X, y)`, `predict(X)`, `transform(X)`, `fit_predict(X, y)` |
| `ColumnTransformer` | `transformers=[("name", transformer, columns), ...]` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `FeatureUnion` | `transformer_list=[("name", transformer), ...]` | `fit(X)`, `transform(X)`, `fit_transform(X)` |
| `TextPipeline` | `steps=[...]` | `fit(texts, y)`, `predict(texts)`, `transform(texts)` |

## ferroml.automl

| Class | Constructor params | Key methods |
|-------|-------------------|-------------|
| `AutoMLConfig` | `task="regression"`, `metric="rmse"`, `max_trials=50`, `timeout=300`, `seed=None` | — (config object) |
| `AutoML` | `config` | `fit(X, y)`, `predict(X)`, `leaderboard()`, `best_model()`, `best_params()` |
| `AutoMLResult` | — (returned by AutoML) | `.best_score`, `.best_model`, `.leaderboard` |
| `LeaderboardEntry` | — | `.model_name`, `.score`, `.params`, `.fit_time` |
| `EnsembleResult` | — | `.members`, `.score` |
| `EnsembleMember` | — | `.model_name`, `.weight` |

## Top-level (ferroml)

| Name | Type | Signature / Usage |
|------|------|-------------------|
| `recommend` | function | `recommend(X, y, task="auto")` -- returns `list[Recommendation]` (top 5 models with reasoning) |
| `Recommendation` | class | `.model_name`, `.reasoning`, `.suggested_params` |
| `ModelCard` | class | `ModelCard.model_card()` (staticmethod on every model) -- `.name`, `.task`, `.complexity`, `.strengths`, `.weaknesses` |

## Universal model interface

All models share:

```python
model.fit(X, y)                    # Train
model.predict(X)                   # Predict
model.search_space()               # HPO parameter ranges
ModelClass.model_card()            # Structured metadata (static)
```

Statistical models add:

```python
model.summary()                    # Full OLS-style summary
model.coefficients_with_ci()       # Coefficients with confidence intervals
model.r_squared()                  # R-squared
model.f_statistic()                # F-statistic
model.aic()                        # Akaike information criterion
model.bic()                        # Bayesian information criterion
```

Probabilistic models add:

```python
model.predict_proba(X)             # Class probabilities
model.predict_with_uncertainty(X, confidence=0.95)  # Predictions + CIs
```

## Error handling

All errors are `ValueError` or `RuntimeError` in Python, with `.hint()` remediation text.

| Error variant | When raised | Typical hint |
|---------------|-------------|--------------|
| `InvalidInput` | Bad parameter values, wrong types | "Parameter X must be positive" |
| `ShapeMismatch` | X and y have incompatible shapes | "X has N samples but y has M" |
| `ConvergenceFailure` | Model did not converge | "Increase max_iter or reduce learning_rate" |
| `NotFitted` | predict() called before fit() | "Call fit() before predict()" |
| `AssumptionViolation` | Statistical assumption not met | "Residuals are not normally distributed" |
| `NumericalError` | NaN/Inf in computation | "Input contains NaN -- use SimpleImputer" |
