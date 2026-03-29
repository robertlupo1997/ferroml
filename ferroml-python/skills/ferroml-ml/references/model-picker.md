# Model Selection Guide

## Decision Framework

### Step 1: What's your task?

| Task | How to detect | Go to |
|------|--------------|-------|
| Regression | Target is continuous (price, temperature, revenue) | Regression Models |
| Binary Classification | Target has 2 classes (yes/no, spam/not, churn/stay) | Classification Models |
| Multiclass Classification | Target has 3-20 classes (species, category, rating) | Classification Models |
| Clustering | No target — find groups in data | Clustering Models |
| Anomaly Detection | Find outliers or unusual patterns | Anomaly Models |
| Dimensionality Reduction | Reduce features for visualization or speed | Decomposition Models |

### Step 2: How much data?

| Size | Rows | Best approach |
|------|------|--------------|
| Tiny | <100 | Simple models (Linear, KNN). Risk of overfitting with complex models. |
| Small | 100-1K | Linear models, small Random Forests. Use strong regularization. |
| Medium | 1K-100K | Full toolkit available. This is the sweet spot. |
| Large | 100K-1M | Hist-based models, MiniBatchKMeans. Avoid SVC (O(n^2)). |
| Very Large | >1M | HistGradientBoosting, SGD, IncrementalPCA. Subsample for other models. |

### Step 3: What matters most?

| Priority | Choose | Avoid |
|----------|--------|-------|
| Interpretability | LinearRegression, LogisticRegression, DecisionTree | MLP, SVC (kernel), GBT |
| Raw accuracy | HistGradientBoosting, RandomForest, Stacking | Linear models (if non-linear) |
| Speed (training) | LinearRegression, GaussianNB, SGD | SVC, GP, large RandomForest |
| Speed (prediction) | Linear models, DecisionTree | KNN (lazy), GP (kernel) |
| Uncertainty estimates | GaussianProcess, bootstrap_ci | Point-estimate-only models |
| Statistical diagnostics | LinearRegression (full OLS summary) | Tree-based (no p-values) |

## Regression Models

### LinearRegression
- **When:** Linear relationship, need statistical diagnostics (p-values, CIs, F-stat)
- **Strengths:** Full OLS summary, fast, interpretable
- **Weaknesses:** Assumes linearity, sensitive to multicollinearity
- **FerroML bonus:** `.summary()`, `.coefficients_with_ci()`, `.f_statistic()`, `.aic()`, `.bic()`

### RidgeRegression / LassoRegression / ElasticNet
- **When:** Multicollinearity (Ridge), feature selection (Lasso), both (ElasticNet)
- **Use RidgeCV/LassoCV/ElasticNetCV** for automatic alpha selection

### GradientBoostingRegressor / HistGradientBoostingRegressor
- **When:** Non-linear patterns, medium-large data
- **Hist variant** is 5-10x faster on >10K samples
- **Strengths:** Handles mixed features, robust to outliers

### RandomForestRegressor
- **When:** Non-linear, want feature importance, want robustness
- **Strengths:** Minimal tuning needed, resistant to overfitting

### GaussianProcessRegressor
- **When:** Small data (<1K), need uncertainty on predictions
- **Weaknesses:** O(n^3) scaling, fails on large datasets

### SVR
- **When:** Small-medium data, complex non-linear patterns
- **Weaknesses:** Needs careful hyperparameter tuning (C, gamma)

## Classification Models

### LogisticRegression
- **When:** Linear decision boundary, need probabilities, interpretability
- **FerroML bonus:** Statistical diagnostics, coefficients with CIs

### RandomForestClassifier
- **When:** Non-linear, want feature importance, robust default
- **Strengths:** Works well out of the box, handles mixed features

### GradientBoostingClassifier / HistGradientBoostingClassifier
- **When:** Maximum accuracy on tabular data
- **Note:** Hist variant for >10K samples

### GaussianNB
- **When:** Fast baseline, text classification, very small data
- **Strengths:** Extremely fast, works with few samples

### SVC
- **When:** Small-medium data, complex boundaries
- **Note:** Use LinearSVC for large datasets (much faster)

### KNeighborsClassifier
- **When:** Local decision boundaries, small data, need simplicity
- **Weaknesses:** Slow prediction on large datasets, sensitive to scaling

## Clustering Models

### KMeans / MiniBatchKMeans
- **When:** Known number of clusters, spherical clusters
- **MiniBatch** for >50K samples

### DBSCAN / HDBSCAN
- **When:** Unknown number of clusters, arbitrary shapes, outlier detection
- **HDBSCAN** is more robust (no epsilon parameter)

### GaussianMixture
- **When:** Soft clustering (probability of membership), elliptical clusters

### AgglomerativeClustering
- **When:** Hierarchical structure, small-medium data

## Anomaly Models

### IsolationForest
- **When:** General anomaly detection, medium-large data
- **Strengths:** Fast, handles high dimensions

### LocalOutlierFactor
- **When:** Density-based anomalies, need local context
- **Strengths:** Catches local outliers that global methods miss
