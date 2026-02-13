# FerroML Accuracy Report

Comparison of FerroML against scikit-learn across models and preprocessing transformers.

*Last updated: 2026-02-11 | Test environment: Python 3.11, sklearn latest, FerroML 0.1.0 (commit 3d693b9)*

## Summary

| Category | Tested | Passing | Match Rate |
|----------|--------|---------|------------|
| Models | 6 | 6 | 100% |
| Preprocessing | 8 | 8 | 100% |
| **Total** | **14** | **14** | **100%** |

## Tolerance Standards

| Algorithm Type | Tolerance | Rationale |
|----------------|-----------|-----------|
| Closed-form (Linear, PCA) | 1e-10 | Exact analytic solution |
| Iterative (Lasso, Logistic) | 1e-4 | Convergence-dependent |
| Tree-based | 1e-6 | Deterministic splits |
| Probabilistic outputs | 1e-6 | Softmax stability |
| Ensemble (RF, GB) | 5% | RNG implementation differences |

## Model Accuracy

| Model | Dataset | Metric | sklearn | FerroML | Difference | Status |
|-------|---------|--------|---------|---------|------------|--------|
| LinearRegression | Diabetes | R² | 0.4526 | 0.4526 | 0.00e+00 | **PASS** |
| DecisionTreeClassifier | Iris | Accuracy | 1.0000 | 1.0000 | 0.00e+00 | **PASS** |
| DecisionTreeClassifier | Wine | Accuracy | 1.0000 | 1.0000 | 0.00e+00 | **PASS** |
| RandomForestClassifier | Iris | Accuracy | 1.0000 | 1.0000 | 0.00e+00 | **PASS** |
| RandomForestClassifier | Wine | Accuracy | 1.0000 | 1.0000 | 0.00e+00 | **PASS** |
| RandomForestRegressor | Diabetes | R² | 0.4428 | 0.4263 | 1.66e-02 | **PASS*** |

*RandomForestRegressor difference (3.75%) is expected variance from different RNG implementations between Rust and Python. Both use the same algorithm; random seed sequences differ across languages.

### Model Notes

- **DecisionTreeRegressor**: R² = 0.26 (FerroML) vs 0.29 (sklearn). Difference is due to tie-breaking in split selection. Fixed in Plan 1 with epsilon tie-breaking.
- **LogisticRegression**: Functional and convergent. Slight coefficient differences due to optimizer implementation (IRLS vs L-BFGS). Python binding type conversion fixed in Plan 1.

## Preprocessing Accuracy

All preprocessing transformers tested against sklearn with a tolerance of 1e-10.

| Transformer | Max Difference | Status |
|-------------|---------------|--------|
| StandardScaler | 0.00e+00 | **EXACT MATCH** |
| MinMaxScaler | 0.00e+00 | **EXACT MATCH** |
| RobustScaler | 0.00e+00 | **EXACT MATCH** |
| MaxAbsScaler | 0.00e+00 | **EXACT MATCH** |
| OneHotEncoder | 0.00e+00 | **EXACT MATCH** |
| OrdinalEncoder | 0.00e+00 | **EXACT MATCH** |
| LabelEncoder | consistent | **MATCH** |
| SimpleImputer (mean) | 0.00e+00 | **EXACT MATCH** |

### Notes

- **StandardScaler** uses population variance (ddof=0) matching sklearn's default. Fixed in the Bug Audit phase.
- **LabelEncoder** produces consistent, invertible 1-to-1 mappings. Category ordering may differ from sklearn.

## Implemented Algorithms

### Models (28)

| Category | Models |
|----------|--------|
| Linear | LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, QuantileRegression, RobustRegression |
| Tree-Based | DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor |
| Distance-Based | KNeighborsClassifier, KNeighborsRegressor |
| SVM | SVC, SVR, LinearSVC, LinearSVR |
| Bayesian | GaussianNB, MultinomialNB, BernoulliNB |
| Clustering | KMeans, DBSCAN |
| Neural Networks | MLPClassifier, MLPRegressor |

### Preprocessing (23+)

| Category | Transformers |
|----------|-------------|
| Scalers | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler |
| Encoders | OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder |
| Imputers | SimpleImputer, KNNImputer |
| Decomposition | PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis |
| Feature Selection | VarianceThreshold, SelectKBest, SelectFromModel, RFE |
| Resampling | SMOTE, BorderlineSMOTE, ADASYN, RandomUnderSampler, RandomOverSampler, SMOTETomek, SMOTEENN |

## Test Infrastructure

| Metric | Value |
|--------|-------|
| Unit tests | 2395 passing |
| Doctests | 82 passing |
| Clippy | Clean (0 warnings) |
| Platforms | Windows, macOS, Linux |

### Test Datasets

| Dataset | Type | Samples | Features | Classes |
|---------|------|---------|----------|---------|
| Iris | Classification | 150 | 4 | 3 |
| Wine | Classification | 178 | 13 | 3 |
| Diabetes | Regression | 442 | 10 | - |

## Reproducing Results

### Sklearn comparison script

```bash
cd ferroml-core/tests
python sklearn_comparison.py
```

### Preprocessing comparison

```bash
cd ferroml-python/tests
python sklearn_preprocessing_comparison.py
```

### All unit tests

```bash
cargo test -p ferroml-core --lib
```
