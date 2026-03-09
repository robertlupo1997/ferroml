# Plan M: Real-World Validation (FerroML vs sklearn)

## Overview

Systematic accuracy and performance comparison of ALL 50+ FerroML models against their sklearn equivalents on real and synthetic datasets. Currently, FerroML has 58 Rust-level sklearn fixture tests and 1,108 Python tests — but **zero Python-level side-by-side sklearn comparison tests**. This plan fills that gap with comprehensive validation.

## Current State

- 58 Rust correctness tests compare against sklearn JSON fixtures (tolerance: 1e-8 to 5%)
- 1,108 Python tests verify FerroML models work (functional, not comparative)
- Built-in datasets: iris (150x4), wine (178x13), diabetes (442x10), linnerud (20x3)
- Synthetic generators: make_classification, make_regression, make_blobs, make_moons, make_circles
- No breast_cancer or california_housing datasets built-in (will use sklearn's)
- docs/accuracy-report.md has tolerance standards per algorithm type

## Desired End State

- Every FerroML model with a sklearn equivalent has a side-by-side comparison test
- Comprehensive validation report (markdown) with pass/fail per model
- Performance comparison (fit time, predict time) on varying dataset sizes
- Edge case validation for robustness
- All comparison tests re-runnable as regression suite
- Any bugs found are documented and fixed

---

## Phase M.1: Comparison Infrastructure + Linear Models

**Overview**: Build the comparison test framework and validate all 13 linear models.

**Changes Required**:

1. **File**: `ferroml-python/tests/conftest_comparison.py` (NEW)
   - Helper functions: `compare_predictions(ferro, sklearn, X, y, tol)`, `compare_probabilities(...)`, `compare_transforms(...)`
   - Timing helpers: `timed_fit(model, X, y)`, `timed_predict(model, X)`
   - Dataset loaders wrapping sklearn: `get_iris()`, `get_wine()`, `get_breast_cancer()`, `get_diabetes()`, `get_california_housing()`
   - Synthetic dataset generators at varying sizes: `get_classification_data(n=1000, p=20)`, `get_regression_data(n=1000, p=20)`
   - Result collector: `ComparisonResult(model, dataset, metric, ferro_value, sklearn_value, tolerance, passed)`
   - Report generator: `generate_report(results) -> str` (markdown table)

2. **File**: `ferroml-python/tests/test_comparison_linear.py` (NEW)
   - ~45 tests covering all 13 linear models:

   | FerroML Model | sklearn Equivalent | Datasets | Metrics | Tolerance |
   |---|---|---|---|---|
   | LinearRegression | LinearRegression | diabetes, california, synthetic | R², MSE, coef_ | 1e-8 |
   | LogisticRegression | LogisticRegression | iris, breast_cancer, wine | accuracy, proba | 1e-4 |
   | RidgeRegression | Ridge | diabetes, synthetic | R², coef_ | 1e-6 |
   | LassoRegression | Lasso | diabetes, synthetic | R², coef_ | 1e-4 |
   | ElasticNet | ElasticNet | diabetes, synthetic | R², coef_ | 1e-4 |
   | RobustRegression | HuberRegressor | synthetic (with outliers) | R² | 5% |
   | QuantileRegression | QuantileRegressor | synthetic | predictions | 5% |
   | Perceptron | Perceptron | iris (linearly separable subset) | accuracy | 10% |
   | RidgeCV | RidgeCV | diabetes, synthetic | R², alpha_ | 1e-4 |
   | LassoCV | LassoCV | diabetes, synthetic | R², alpha_ | 1e-4 |
   | ElasticNetCV | ElasticNetCV | diabetes, synthetic | R², alpha_ | 1e-4 |
   | RidgeClassifier | RidgeClassifier | iris, wine | accuracy | 5% |
   | IsotonicRegression | IsotonicRegression | synthetic monotonic | predictions | 1e-6 |

   - Each test: fit both models, compare predictions, compare coefficients where applicable
   - Parametrized over datasets where relevant

**Success Criteria**:
- [ ] `pytest ferroml-python/tests/test_comparison_linear.py -v` — all pass
- [ ] Report shows all linear models within tolerance

**Expected Tests**: ~45

---

## Phase M.2: Tree & Ensemble Models

**Overview**: Validate all 8 tree models and 9 ensemble models against sklearn.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_comparison_trees.py` (NEW)
   - ~40 tests covering tree + ensemble models:

   | FerroML Model | sklearn Equivalent | Datasets | Metrics | Tolerance |
   |---|---|---|---|---|
   | DecisionTreeClassifier | DecisionTreeClassifier | iris, breast_cancer | accuracy, feature_importances | 1e-6 |
   | DecisionTreeRegressor | DecisionTreeRegressor | diabetes, california | R², MSE | 1e-6 |
   | RandomForestClassifier | RandomForestClassifier | iris, breast_cancer, wine | accuracy, proba | 5% (RNG) |
   | RandomForestRegressor | RandomForestRegressor | diabetes, california | R² | 5% |
   | GradientBoostingClassifier | GradientBoostingClassifier | iris, breast_cancer | accuracy, proba | 5% |
   | GradientBoostingRegressor | GradientBoostingRegressor | diabetes, california | R² | 5% |
   | HistGradientBoostingClassifier | HistGradientBoostingClassifier | breast_cancer, wine | accuracy | 5% |
   | HistGradientBoostingRegressor | HistGradientBoostingRegressor | california, diabetes | R² | 5% |
   | ExtraTreesClassifier | ExtraTreesClassifier | iris, wine | accuracy | 5% |
   | ExtraTreesRegressor | ExtraTreesRegressor | diabetes | R² | 5% |
   | AdaBoostClassifier | AdaBoostClassifier | iris, breast_cancer | accuracy | 5% |
   | AdaBoostRegressor | AdaBoostRegressor | diabetes | R² | 5% |
   | SGDClassifier | SGDClassifier | breast_cancer (scaled) | accuracy | 10% |
   | SGDRegressor | SGDRegressor | diabetes (scaled) | R² | 10% |

   - For ensemble models: compare prediction distributions rather than exact predictions (RNG differs)
   - Test with `random_state=42` where both support it
   - Compare feature importances for tree models

**Success Criteria**:
- [ ] `pytest ferroml-python/tests/test_comparison_trees.py -v` — all pass
- [ ] All models achieve comparable accuracy (within tolerance)

**Expected Tests**: ~40

---

## Phase M.3: Neighbors, SVM, Naive Bayes, Neural

**Overview**: Validate remaining supervised models.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_comparison_remaining.py` (NEW)
   - ~35 tests:

   | FerroML Model | sklearn Equivalent | Datasets | Metrics | Tolerance |
   |---|---|---|---|---|
   | KNeighborsClassifier | KNeighborsClassifier | iris, wine | accuracy, proba | 1e-6 |
   | KNeighborsRegressor | KNeighborsRegressor | diabetes | R² | 1e-6 |
   | NearestCentroid | NearestCentroid | iris | accuracy | 1e-6 |
   | LinearSVC | LinearSVC | breast_cancer (scaled) | accuracy | 5% |
   | LinearSVR | LinearSVR | diabetes (scaled) | R² | 5% |
   | SVC (RBF) | SVC(kernel='rbf') | iris (scaled) | accuracy, proba | 5% |
   | SVC (linear) | SVC(kernel='linear') | iris | accuracy | 5% |
   | SVC (poly) | SVC(kernel='poly') | iris | accuracy | 5% |
   | SVR (RBF) | SVR(kernel='rbf') | diabetes (scaled) | R² | 5% |
   | GaussianNB | GaussianNB | iris, wine | accuracy, proba | 1e-4 |
   | MultinomialNB | MultinomialNB | synthetic (counts) | accuracy, proba | 1e-4 |
   | BernoulliNB | BernoulliNB | synthetic (binary) | accuracy, proba | 1e-4 |
   | MLPClassifier | MLPClassifier | iris (scaled) | accuracy | 10% |
   | MLPRegressor | MLPRegressor | diabetes (scaled) | R² | 10% |

   - KNN should match exactly (deterministic)
   - SVM: scale data first, compare decision boundaries
   - MLP: wider tolerance due to initialization/optimization differences

**Success Criteria**:
- [ ] `pytest ferroml-python/tests/test_comparison_remaining.py -v` — all pass

**Expected Tests**: ~35

---

## Phase M.4: Clustering & Unsupervised Models

**Overview**: Validate clustering, decomposition, anomaly detection, and GMM.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_comparison_unsupervised.py` (NEW)
   - ~30 tests:

   | FerroML Model | sklearn Equivalent | Datasets | Metrics | Tolerance |
   |---|---|---|---|---|
   | KMeans | KMeans | iris, blobs | inertia, ARI | 5% |
   | DBSCAN | DBSCAN | moons, circles | labels (ARI) | 1e-6 |
   | AgglomerativeClustering | AgglomerativeClustering | iris, blobs | labels (ARI) | 1e-6 |
   | GaussianMixture (full) | GaussianMixture(full) | iris | BIC, ARI | 5% |
   | GaussianMixture (diag) | GaussianMixture(diag) | blobs | BIC | 5% |
   | IsolationForest | IsolationForest | synthetic outliers | anomaly_score dist | 10% |
   | LocalOutlierFactor | LocalOutlierFactor | synthetic outliers | anomaly_score dist | 10% |
   | PCA | PCA | iris, wine | explained_variance, components | 1e-6 |
   | TruncatedSVD | TruncatedSVD | synthetic sparse | explained_variance | 1e-4 |
   | LDA | LinearDiscriminantAnalysis | iris | transform, accuracy | 1e-4 |
   | QDA | QuadraticDiscriminantAnalysis | iris, wine | accuracy, proba | 1e-4 |
   | TSNE | TSNE | iris (small) | structural (kNN preservation) | N/A (stochastic) |
   | FactorAnalysis | FactorAnalysis | synthetic | components | 5% |

   - Clustering: compare via ARI/NMI (label permutation invariant)
   - PCA: compare up to sign flip (eigenvector sign ambiguity)
   - t-SNE: compare kNN preservation ratio, not coordinates
   - Anomaly: compare score distributions, not exact scores

**Success Criteria**:
- [ ] `pytest ferroml-python/tests/test_comparison_unsupervised.py -v` — all pass

**Expected Tests**: ~30

---

## Phase M.5: Preprocessing Comparison

**Overview**: Validate all 21 preprocessing transformers against sklearn.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_comparison_preprocessing.py` (NEW)
   - ~35 tests:

   | FerroML Transformer | sklearn Equivalent | Test | Tolerance |
   |---|---|---|---|
   | StandardScaler | StandardScaler | transform + inverse | 1e-10 |
   | MinMaxScaler | MinMaxScaler | transform + inverse | 1e-10 |
   | RobustScaler | RobustScaler | transform + inverse | 1e-10 |
   | MaxAbsScaler | MaxAbsScaler | transform + inverse | 1e-10 |
   | OneHotEncoder | OneHotEncoder | transform output | 1e-10 |
   | OrdinalEncoder | OrdinalEncoder | transform output | 1e-10 |
   | LabelEncoder | LabelEncoder | transform output | exact |
   | TargetEncoder | TargetEncoder | transform output | 1e-4 |
   | SimpleImputer (mean) | SimpleImputer(mean) | transform output | 1e-10 |
   | SimpleImputer (median) | SimpleImputer(median) | transform output | 1e-10 |
   | KNNImputer | KNNImputer | transform output | 1e-6 |
   | PowerTransformer (yeo) | PowerTransformer(yeo-johnson) | transform | 1e-4 |
   | PowerTransformer (box) | PowerTransformer(box-cox) | transform | 1e-4 |
   | QuantileTransformer | QuantileTransformer | transform | 1e-4 |
   | PolynomialFeatures | PolynomialFeatures | transform shape + values | 1e-10 |
   | KBinsDiscretizer | KBinsDiscretizer | transform | 1e-6 |
   | VarianceThreshold | VarianceThreshold | selected features | exact |
   | SelectKBest | SelectKBest | selected features, scores | 1e-6 |

   - Inverse transform roundtrip where applicable
   - Test on iris, diabetes, and synthetic data with NaN for imputers
   - Compare feature counts and output shapes

**Success Criteria**:
- [ ] `pytest ferroml-python/tests/test_comparison_preprocessing.py -v` — all pass

**Expected Tests**: ~35

---

## Phase M.6: Performance Comparison (Timing)

**Overview**: Compare fit/predict times between FerroML and sklearn on varying dataset sizes.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_performance_comparison.py` (NEW)
   - Not a pytest file — a benchmark script that generates a report
   - ~20 model comparisons at 4 dataset sizes each

   **Models to benchmark** (priority order):
   1. LinearRegression (100, 1K, 10K, 100K rows x 20 features)
   2. LogisticRegression (100, 1K, 10K, 100K rows)
   3. DecisionTreeClassifier (100, 1K, 10K, 100K rows)
   4. RandomForestClassifier (100, 1K, 10K, 50K rows — 100 trees)
   5. GradientBoostingClassifier (100, 1K, 10K, 50K rows — 100 trees)
   6. HistGradientBoostingClassifier (100, 1K, 10K, 100K rows)
   7. KNeighborsClassifier (100, 1K, 10K rows)
   8. SVC (100, 1K, 5K rows — O(n^2))
   9. KMeans (100, 1K, 10K, 100K rows, k=5)
   10. PCA (100, 1K, 10K, 100K rows)
   11. StandardScaler (100, 1K, 10K, 100K rows)
   12. GaussianNB (100, 1K, 10K, 100K rows)
   13. MLPClassifier (100, 1K, 10K rows)
   14. DBSCAN (100, 1K, 5K rows)
   15. AdaBoostClassifier (100, 1K, 10K rows)

   **Metrics per model per size**:
   - fit_time_ms (median of 5 runs)
   - predict_time_ms (median of 5 runs)
   - ferro_vs_sklearn_ratio (< 1.0 = FerroML faster)

2. **File**: `ferroml-python/scripts/run_comparison.py` (NEW)
   - Orchestrator script that runs all comparison tests and generates the report
   - Output: `docs/validation-report.md`

3. **File**: `docs/validation-report.md` (NEW, generated)
   - Model-by-model results table
   - Performance comparison charts (text-based)
   - Summary: pass/fail per model, any bugs found

**Success Criteria**:
- [ ] `python ferroml-python/scripts/run_comparison.py` completes without error
- [ ] `docs/validation-report.md` generated with all results
- [ ] FerroML competitive with sklearn on at least 10/15 models

**Expected Tests**: ~20 benchmark scenarios (not pytest)

---

## Phase M.7: Edge Case Validation

**Overview**: Test robustness on difficult data scenarios.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_comparison_edge_cases.py` (NEW)
   - ~25 tests:

   **High Dimensionality** (p >> n):
   - LinearRegression on 50x500 data (underdetermined)
   - PCA on 100x200 data
   - RandomForest on 200x1000 data

   **Class Imbalance** (99:1 ratio):
   - LogisticRegression, RandomForest, GradientBoosting
   - Compare precision/recall/F1, not just accuracy

   **Near-Constant Features**:
   - VarianceThreshold, StandardScaler (should handle gracefully)
   - Models with near-zero-variance columns

   **Large Feature Counts**:
   - KNN with 500 features (curse of dimensionality)
   - PCA dimensionality reduction then classification

   **Extreme Values**:
   - Models with features at 1e10 scale
   - RobustScaler vs StandardScaler robustness

   **Single Class / Degenerate Cases**:
   - Classifiers with only 1 class in training data
   - Regression with constant target

   **Multicollinear Features**:
   - Ridge vs OLS on perfectly correlated features
   - ElasticNet feature selection behavior

**Success Criteria**:
- [ ] `pytest ferroml-python/tests/test_comparison_edge_cases.py -v` — all pass
- [ ] Both FerroML and sklearn handle edge cases consistently (or both raise errors)

**Expected Tests**: ~25

---

## Phase M.8: Validation Report & Bug Fixes

**Overview**: Consolidate all findings, fix any bugs discovered, generate final report.

**Changes Required**:

1. **Bug fixes** (if any found in M.1-M.7):
   - File bugs with exact reproduction steps
   - Fix in ferroml-core, update Rust tests
   - Rebuild Python bindings, verify fix in comparison test

2. **File**: `docs/validation-report.md` (FINAL)
   - Section 1: Accuracy Summary Table (50+ models, pass/fail, tolerance)
   - Section 2: Performance Summary Table (15 models x 4 sizes, ratio vs sklearn)
   - Section 3: Edge Case Results
   - Section 4: Bugs Found and Fixed
   - Section 5: Known Limitations and Differences from sklearn
   - Section 6: Recommendations for Users

3. **File**: `docs/accuracy-report.md` (UPDATE)
   - Add new comparison results to existing accuracy report
   - Update tolerance standards if any adjustments needed

**Success Criteria**:
- [ ] All comparison tests pass: `pytest ferroml-python/tests/test_comparison_*.py -v`
- [ ] Validation report complete with 50+ model entries
- [ ] Zero new bugs remaining (all found bugs fixed)
- [ ] `cargo test -p ferroml-core` — all pass (no regressions)
- [ ] `pytest ferroml-python/tests/ -q` — all pass

**Expected Tests**: 0 new (consolidation phase)

---

## Summary

| Phase | Focus | New Tests | Key Output |
|---|---|---|---|
| M.1 | Infrastructure + Linear Models | ~45 | conftest_comparison.py, 13 model comparisons |
| M.2 | Tree & Ensemble Models | ~40 | 14 model comparisons |
| M.3 | Neighbors, SVM, NB, Neural | ~35 | 14 model comparisons |
| M.4 | Clustering & Unsupervised | ~30 | 13 model comparisons |
| M.5 | Preprocessing | ~35 | 18 transformer comparisons |
| M.6 | Performance Timing | ~20 | validation-report.md (performance section) |
| M.7 | Edge Cases | ~25 | Robustness validation |
| M.8 | Report & Bug Fixes | 0 | Final validation-report.md |
| **Total** | | **~230** | **50+ models validated against sklearn** |

## Verification Commands

```bash
# Run all comparison tests
pytest ferroml-python/tests/test_comparison_*.py -v

# Run performance benchmarks
python ferroml-python/scripts/run_comparison.py

# Verify no regressions
cargo test -p ferroml-core 2>&1 | grep "^test result:"
pytest ferroml-python/tests/ -q 2>&1 | tail -3
```

## Dependencies

- sklearn (already in .venv: `pip install scikit-learn`)
- imblearn (for SMOTE/ADASYN comparison if needed)
- numpy, time (stdlib)
- No new Rust dependencies
