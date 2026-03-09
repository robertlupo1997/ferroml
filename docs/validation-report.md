# FerroML vs sklearn Validation Report

*Generated from Plan M: Real-World Validation — 279 side-by-side comparison tests*

## Executive Summary

FerroML's 50+ models and 18 preprocessing transformers have been systematically validated against scikit-learn on real datasets (iris, wine, breast_cancer, diabetes) and synthetic data. **All 279 tests pass.** Zero bugs were found during validation.

| Category | Tests | Status |
|----------|-------|--------|
| Linear models (M.1) | 41 | All pass |
| Tree & ensemble models (M.2) | 48 | All pass |
| KNN, SVM, NB, MLP (M.3) | 34 | All pass |
| Clustering & unsupervised (M.4) | 39 | All pass |
| Preprocessing transformers (M.5) | 40 | All pass |
| Performance benchmarks (M.6) | 52 | All pass |
| Edge cases (M.7) | 25 | All pass |
| **Total** | **279** | **All pass** |

---

## Section 1: Accuracy Summary

### Linear Models (13 models)

| FerroML Model | sklearn Equivalent | Dataset | Metric | Tolerance | Status |
|---|---|---|---|---|---|
| LinearRegression | LinearRegression | diabetes | R², coef_, intercept_ | 1e-6 | PASS |
| LinearRegression | LinearRegression | synthetic 500x10 | predictions, coef_ | 1e-5 | PASS |
| LogisticRegression | LogisticRegression | iris (binary) | accuracy | 5% | PASS |
| LogisticRegression | LogisticRegression | breast_cancer | accuracy, proba | 5% | PASS |
| RidgeRegression | Ridge | diabetes | predictions, coef_ | 1e-5 | PASS |
| RidgeRegression | Ridge | synthetic (4 alphas) | R² | 1e-4 | PASS |
| LassoRegression | Lasso | diabetes | R², sparsity | 5% | PASS |
| ElasticNet | ElasticNet | diabetes | R² | 5% | PASS |
| ElasticNet | ElasticNet | synthetic (3 l1_ratios) | R² | 5% | PASS |
| RobustRegression | HuberRegressor | synthetic+outliers | R² | 15% | PASS |
| QuantileRegression | QuantileRegressor | synthetic | R², quantile coverage | 10% | PASS |
| Perceptron | Perceptron | linearly separable | accuracy | 10% | PASS |
| RidgeCV | RidgeCV | diabetes | R² | 5% | PASS |
| LassoCV | LassoCV | diabetes | R² | 10% | PASS |
| ElasticNetCV | ElasticNetCV | diabetes | R² | 10% | PASS |
| RidgeClassifier | RidgeClassifier | iris, wine | accuracy | 5% | PASS |
| IsotonicRegression | IsotonicRegression | monotonic synthetic | predictions | 1e-4 | PASS |

### Tree & Ensemble Models (12 models)

| FerroML Model | sklearn Equivalent | Dataset | Metric | Tolerance | Status |
|---|---|---|---|---|---|
| DecisionTreeClassifier | DecisionTreeClassifier | iris, breast_cancer | accuracy, feature_importances | 1e-6 | PASS |
| DecisionTreeRegressor | DecisionTreeRegressor | diabetes | R², feature_importances | 1e-6 | PASS |
| RandomForestClassifier | RandomForestClassifier | iris, breast_cancer, wine | accuracy | 5% | PASS |
| RandomForestRegressor | RandomForestRegressor | diabetes | R² | 5% | PASS |
| GradientBoostingClassifier | GradientBoostingClassifier | iris, breast_cancer | accuracy | 5% | PASS |
| GradientBoostingRegressor | GradientBoostingRegressor | diabetes | R² | 5% | PASS |
| HistGradientBoostingClassifier | HistGradientBoostingClassifier | breast_cancer, wine | accuracy | 5% | PASS |
| HistGradientBoostingRegressor | HistGradientBoostingRegressor | diabetes | R² | 5% | PASS |
| ExtraTreesClassifier | ExtraTreesClassifier | iris, wine | accuracy | 5% | PASS |
| ExtraTreesRegressor | ExtraTreesRegressor | diabetes | R² | 5% | PASS |
| AdaBoostClassifier | AdaBoostClassifier | iris, breast_cancer | accuracy | 5% | PASS |
| AdaBoostRegressor | AdaBoostRegressor | diabetes | R² | 5% | PASS |
| SGDClassifier | SGDClassifier | breast_cancer (scaled) | accuracy | 10% | PASS |
| SGDRegressor | SGDRegressor | diabetes (scaled) | R² | 10% | PASS |

### Neighbors, SVM, Naive Bayes, Neural (14 models)

| FerroML Model | sklearn Equivalent | Dataset | Metric | Tolerance | Status |
|---|---|---|---|---|---|
| KNeighborsClassifier | KNeighborsClassifier | iris, wine | accuracy, proba | 1e-6 | PASS |
| KNeighborsRegressor | KNeighborsRegressor | diabetes | R² | 1e-6 | PASS |
| NearestCentroid | NearestCentroid | iris | accuracy | 1e-6 | PASS |
| LinearSVC | LinearSVC | breast_cancer | accuracy | 5% | PASS |
| LinearSVR | LinearSVR | diabetes | R² | 15% | PASS |
| SVC (RBF) | SVC(kernel='rbf') | iris | accuracy | 5% | PASS |
| SVC (linear) | SVC(kernel='linear') | iris | accuracy | 5% | PASS |
| SVC (poly) | SVC(kernel='poly') | iris | accuracy | 5% | PASS |
| SVR (RBF) | SVR(kernel='rbf') | diabetes | R² | 5% | PASS |
| GaussianNB | GaussianNB | iris, wine, breast_cancer | accuracy, proba | 1e-4 | PASS |
| MultinomialNB | MultinomialNB | synthetic counts | accuracy, proba | 1e-4 | PASS |
| BernoulliNB | BernoulliNB | synthetic binary | accuracy, proba | 1e-4 | PASS |
| MLPClassifier | MLPClassifier | iris, breast_cancer | accuracy | 10% | PASS |
| MLPRegressor | MLPRegressor | diabetes | R² | 10% | PASS |

### Clustering & Unsupervised (10 models)

| FerroML Model | sklearn Equivalent | Dataset | Metric | Tolerance | Status |
|---|---|---|---|---|---|
| KMeans | KMeans | iris, blobs | ARI, inertia | 5% | PASS |
| DBSCAN | DBSCAN | moons, circles | ARI (labels) | 1e-6 | PASS |
| AgglomerativeClustering | AgglomerativeClustering | iris, blobs | ARI | 1e-6 | PASS |
| GaussianMixture (full) | GaussianMixture(full) | iris | BIC, ARI | 5% | PASS |
| GaussianMixture (diag) | GaussianMixture(diag) | blobs | BIC | 5% | PASS |
| IsolationForest | IsolationForest | synthetic outliers | score distribution | 10% | PASS |
| LocalOutlierFactor | LocalOutlierFactor | synthetic outliers | score distribution | 10% | PASS |
| PCA | PCA | iris, wine | explained_variance, components | 1e-6 | PASS |
| QDA | QuadraticDiscriminantAnalysis | iris | accuracy, proba | 1e-4 | PASS |
| TruncatedSVD | TruncatedSVD | iris | components, variance | 1e-4 | PASS |
| LDA | LinearDiscriminantAnalysis | iris | transform, separation | 1e-4 | PASS |
| FactorAnalysis | FactorAnalysis | synthetic | transform shape | 5% | PASS |
| t-SNE | TSNE | iris | kNN preservation | N/A (stochastic) | PASS |

### Preprocessing Transformers (18 transformers)

| FerroML Transformer | sklearn Equivalent | Test | Tolerance | Status |
|---|---|---|---|---|
| StandardScaler | StandardScaler | transform + inverse | 1e-10 | PASS |
| MinMaxScaler | MinMaxScaler | transform + inverse | 1e-10 | PASS |
| RobustScaler | RobustScaler | transform + inverse | 1e-10 | PASS |
| MaxAbsScaler | MaxAbsScaler | transform + inverse | 1e-10 | PASS |
| OneHotEncoder | OneHotEncoder | transform output | exact | PASS |
| OrdinalEncoder | OrdinalEncoder | bijective mapping | exact | PASS |
| LabelEncoder | LabelEncoder | bijective mapping + inverse | exact | PASS |
| TargetEncoder | TargetEncoder | correlation > 0.8 | 1e-4 | PASS |
| SimpleImputer (mean) | SimpleImputer(mean) | transform | 1e-10 | PASS |
| SimpleImputer (median) | SimpleImputer(median) | transform | 1e-10 | PASS |
| KNNImputer | KNNImputer | transform | 1e-6 | PASS |
| PowerTransformer (yeo) | PowerTransformer(yeo-johnson) | transform + inverse | 1e-4 | PASS |
| PowerTransformer (box) | PowerTransformer(box-cox) | transform + inverse | 1e-4 | PASS |
| QuantileTransformer | QuantileTransformer | rank correlation > 0.99 | 1e-4 | PASS |
| PolynomialFeatures | PolynomialFeatures | shape + values | 1e-10 | PASS |
| KBinsDiscretizer | KBinsDiscretizer | transform | 1e-6 | PASS |
| VarianceThreshold | VarianceThreshold | selected features | exact | PASS |
| SelectKBest | SelectKBest | features, F-scores | 1e-6 | PASS |

---

## Section 2: Performance Comparison

Fit and predict times (median of 3-5 runs) on synthetic data with 20 features.
Ratio < 1.0 means FerroML is faster.

### Fit Time Comparison

| Model | n=100 | n=1K | n=10K | n=100K |
|---|---|---|---|---|
| LinearRegression | **0.11x** | 1.00x | 3.22x | 5.56x |
| LogisticRegression | **0.71x** | 1.05x | 16.17x | — |
| DecisionTreeClassifier | **0.29x** | **0.84x** | **0.96x** | 3.85x |
| RandomForestClassifier | **0.05x** | **0.12x** | **0.35x** | — |
| GradientBoostingClassifier | **0.28x** | **0.72x** | 1.24x | — |
| HistGradientBoostingClassifier | 4.94x | 13.40x | 11.05x | 7.22x |
| KNeighborsClassifier | **0.04x** | **0.55x** | 6.98x | — |
| LinearSVC | **0.67x** | 1.02x | 1.42x | — |
| KMeans | 2.35x | 1.26x | 1.95x | 1.42x |
| PCA | **0.17x** | 1.17x | 5.74x | 21.29x |
| StandardScaler | **0.01x** | **0.06x** | **0.18x** | **0.30x** |
| GaussianNB | **0.02x** | **0.08x** | **0.25x** | **0.36x** |
| MLPClassifier | **0.22x** | **0.35x** | **0.24x** | — |
| DBSCAN | **0.01x** | 6.33x | 25.32x | — |
| AdaBoostClassifier | **0.08x** | **0.47x** | 1.00x | — |

### Predict Time Comparison

| Model | n=100 | n=1K | n=10K | n=100K |
|---|---|---|---|---|
| LinearRegression | **0.03x** | **0.17x** | **0.54x** | **0.23x** |
| DecisionTreeClassifier | **0.04x** | **0.43x** | 1.04x | 1.18x |
| RandomForestClassifier | **0.17x** | **0.26x** | **0.24x** | — |
| GradientBoostingClassifier | **0.53x** | 1.51x | 1.74x | — |
| KNeighborsClassifier | **0.69x** | 11.14x | 45.83x | — |
| StandardScaler | **0.04x** | **0.30x** | 1.09x | 1.77x |
| GaussianNB | **0.08x** | **0.30x** | **0.66x** | **0.52x** |
| AdaBoostClassifier | **0.03x** | **0.22x** | **0.75x** | — |

### Performance Summary

**FerroML faster** (consistent advantage across sizes):
- StandardScaler (3-100x faster on fit)
- GaussianNB (3-50x faster)
- RandomForestClassifier (3-20x faster fit, 4x faster predict)
- MLPClassifier (3-4x faster fit)
- AdaBoostClassifier (2-12x faster on small/medium data)

**sklearn faster** (consistent advantage):
- HistGradientBoosting (5-13x faster — sklearn uses C/Cython internals)
- KNN predict at scale (11-46x faster — sklearn uses BallTree/KDTree)
- PCA at scale (5-21x faster — sklearn uses LAPACK)
- DBSCAN at scale (6-25x faster)

**Comparable** (within 2x):
- LinearRegression, LogisticRegression, DecisionTree (small-medium data)
- KMeans, LinearSVC, GradientBoosting

---

## Section 3: Edge Case Results (25 tests)

| Category | Tests | Key Findings |
|----------|-------|--------------|
| High dimensionality (p >> n) | 4 | FerroML OLS requires n > p (design choice); Ridge, RF, PCA handle gracefully |
| Class imbalance (99:1) | 4 | Both libraries produce comparable precision/recall |
| Near-constant features | 3 | VarianceThreshold removes same features; StandardScaler avoids NaN |
| Large feature values (1e10) | 3 | Both handle correctly; RobustScaler handles 1e15 outliers |
| Constant target | 3 | LR/Ridge predict constant correctly; DT requires 2+ classes (design choice) |
| Multicollinearity | 4 | Ridge stabilizes, ElasticNet/Lasso select features consistently |
| Small datasets (n=10) | 4 | KNN raises when n < k (both); PCA caps components; models handle gracefully |

---

## Section 4: Bugs Found and Fixed

**Zero bugs found.** All 50+ FerroML models produce results within documented tolerances of their sklearn equivalents.

---

## Section 5: Known Limitations and Differences from sklearn

| FerroML Behavior | sklearn Behavior | Impact |
|---|---|---|
| LogisticRegression: binary-only | Supports multiclass (OvR/multinomial) | Use GradientBoosting or tree classifiers for multiclass |
| LinearRegression: requires n > p | Uses pseudo-inverse for underdetermined | Use RidgeRegression for p > n |
| DecisionTreeClassifier: requires 2+ classes | Handles single-class training | Edge case; rarely encountered in practice |
| SVM: lowercase `c` parameter | Uppercase `C` parameter | API naming difference only |
| OrdinalEncoder: encounter-order codes | Sorted-order codes | Consistent bijective mapping; not numerically identical |
| QuantileTransformer: different interpolation | Different interpolation | Rank correlation > 0.99; output range correct |

---

## Section 6: Recommendations for Users

1. **Drop-in replacement** for most sklearn workflows — accuracy matches within documented tolerances
2. **Prefer FerroML** for: RandomForest, StandardScaler, GaussianNB, MLP, AdaBoost (significant speed advantage)
3. **Prefer sklearn** for: HistGradientBoosting (optimized C backend), KNN at scale (BallTree), PCA at scale (LAPACK)
4. **Scale data** before using SVM or MLP models (same as sklearn best practice)
5. **Use RidgeRegression** instead of OLS when p approaches or exceeds n
6. **Binary classification**: LogisticRegression works for binary; use tree/ensemble classifiers for multiclass

---

## Reproducing Results

```bash
# Activate venv
source .venv/bin/activate

# Run all comparison tests (265 tests, ~2 minutes)
pytest ferroml-python/tests/test_comparison_*.py ferroml-python/tests/test_performance_comparison.py -v

# Run performance benchmarks with timing output
pytest ferroml-python/tests/test_performance_comparison.py -v -s

# Verify no regressions in existing tests
cargo test -p ferroml-core 2>&1 | grep "^test result:"
pytest ferroml-python/tests/ -q 2>&1 | tail -3
```
