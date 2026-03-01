# Plan E: Preprocessing Module Correctness Testing

**Date:** 2026-02-25
**Priority:** HIGH (15,875 lines of untested preprocessing code)
**Module:** `ferroml-core/src/preprocessing/` (15,875 lines)
**Estimated New Tests:** ~40
**Parallel-Safe:** Yes (no overlap with clustering/neural/benchmark/python plans)

## Overview

The preprocessing module is the largest untested module by code volume. While scalers (StandardScaler, MinMaxScaler, etc.) have basic correctness tests, the majority of transformers — PolynomialFeatures, KBinsDiscretizer, feature selectors, power transforms, and imputers — have zero sklearn comparison tests. Since preprocessing is the entry point for every ML pipeline, bugs here silently corrupt all downstream models.

## Current State

### Module Structure
```
preprocessing/
├── scalers.rs          — StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
├── encoders.rs         — LabelEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder
├── imputers.rs         — SimpleImputer, KNNImputer
├── polynomial.rs       — PolynomialFeatures
├── discretizers.rs     — KBinsDiscretizer
├── selection.rs        — VarianceThreshold, SelectKBest, SelectFromModel, RFE
├── power.rs            — PowerTransformer (Box-Cox, Yeo-Johnson)
├── quantile.rs         — QuantileTransformer
├── resampling.rs       — SMOTE, ADASYN, RandomOverSampler, RandomUnderSampler
├── text.rs             — TF-IDF, CountVectorizer
├── mod.rs              — Re-exports, Transformer trait
└── ...
```

### Existing Test Coverage

| Transformer | Inline Tests | Correctness Tests | Status |
|------------|-------------|-------------------|--------|
| StandardScaler | ~5 | Yes (phases 2-7) | **Tested** |
| MinMaxScaler | ~3 | Yes | **Tested** |
| RobustScaler | ~3 | Yes | **Tested** |
| MaxAbsScaler | ~2 | Yes | **Tested** |
| LabelEncoder | ~3 | Partial | Needs more |
| OrdinalEncoder | ~3 | Partial | Needs more |
| OneHotEncoder | ~4 | Partial (bug found+fixed) | Needs more |
| TargetEncoder | ~2 | No | **UNTESTED** |
| SimpleImputer | ~3 | No | **UNTESTED** |
| KNNImputer | ~2 | No | **UNTESTED** |
| PolynomialFeatures | ~3 | Partial (bug found+fixed) | Needs more |
| KBinsDiscretizer | ~2 | No | **UNTESTED** |
| VarianceThreshold | ~2 | No | **UNTESTED** |
| SelectKBest | ~2 | No (was panicking, fixed) | **UNTESTED** |
| SelectFromModel | ~1 | No | **UNTESTED** |
| RFE | ~1 | No | **UNTESTED** |
| PowerTransformer | ~2 | No | **UNTESTED** |
| QuantileTransformer | ~2 | No | **UNTESTED** |
| SMOTE | ~2 | No | **UNTESTED** |
| ADASYN | ~1 | No | **UNTESTED** |
| TF-IDF | ~2 | No | **UNTESTED** |

**Summary:** 4 of 20+ transformers have sklearn-level testing. 16 are untested.

### Known Bugs (Previously Found & Fixed)
- StandardScaler: ddof=1 vs ddof=0 (fixed)
- OneHotEncoder: extra column (fixed)
- PolynomialFeatures: missing interaction terms (fixed)
- SelectKBest: panic on certain inputs (fixed)

### Suspected Remaining Issues
- TargetEncoder: smoothing implementation unverified
- KNNImputer: distance weighting correctness unclear
- PowerTransformer: Box-Cox lambda optimization unverified
- QuantileTransformer: quantile interpolation method unverified
- SMOTE: synthetic sample generation correctness unverified

## Desired End State

- Every transformer tested against sklearn equivalent
- 40+ correctness tests with fixtures
- Edge cases: NaN handling, single-feature, constant-feature, empty data
- Pipeline integration tests (scaler -> selector -> model)

## Implementation Phases

### Phase E.1: Python Fixture Generation

**File:** `benchmarks/preprocessing_fixtures.py` (NEW)

```python
from sklearn.preprocessing import (
    PolynomialFeatures, KBinsDiscretizer, PowerTransformer,
    QuantileTransformer, LabelEncoder, OrdinalEncoder, OneHotEncoder,
    TargetEncoder
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    SelectFromModel, RFE
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.datasets import make_classification, load_iris
import numpy as np

# Each fixture: input data + expected output after fit_transform

# 1. PolynomialFeatures degree=2 on Iris (4 features -> 15)
# 2. PolynomialFeatures degree=3 on 2 features
# 3. PolynomialFeatures interaction_only=True
# 4. KBinsDiscretizer n_bins=5, strategy='uniform'
# 5. KBinsDiscretizer n_bins=3, strategy='quantile'
# 6. KBinsDiscretizer strategy='kmeans'
# 7. PowerTransformer method='yeo-johnson'
# 8. PowerTransformer method='box-cox' (positive data only)
# 9. QuantileTransformer n_quantiles=100, output_distribution='uniform'
# 10. QuantileTransformer output_distribution='normal'
# 11. VarianceThreshold threshold=0.1
# 12. SelectKBest k=3, score_func=f_classif
# 13. SimpleImputer strategy='mean'
# 14. SimpleImputer strategy='median'
# 15. SimpleImputer strategy='most_frequent'
# 16. KNNImputer n_neighbors=5
# 17. LabelEncoder on string categories
# 18. OrdinalEncoder on mixed categories
# 19. OneHotEncoder sparse=False, drop=None
# 20. OneHotEncoder drop='first'
# 21. TargetEncoder (if sklearn >= 1.3)
# 22. SMOTE on imbalanced binary (100:10 ratio)
# 23. ADASYN on imbalanced binary
```

### Phase E.2: Correctness Test File

**File:** `ferroml-core/tests/correctness_preprocessing.rs` (NEW)

```rust
mod polynomial_tests {
    #[test] fn test_polynomial_features_degree2_iris_vs_sklearn() { ... }
    #[test] fn test_polynomial_features_degree3_vs_sklearn() { ... }
    #[test] fn test_polynomial_features_interaction_only_vs_sklearn() { ... }
    #[test] fn test_polynomial_features_single_feature() { ... }
    #[test] fn test_polynomial_features_output_shape() { ... }
}

mod discretizer_tests {
    #[test] fn test_kbins_uniform_vs_sklearn() { ... }
    #[test] fn test_kbins_quantile_vs_sklearn() { ... }
    #[test] fn test_kbins_kmeans_vs_sklearn() { ... }
    #[test] fn test_kbins_single_feature() { ... }
}

mod power_transform_tests {
    #[test] fn test_yeo_johnson_vs_sklearn() { ... }
    #[test] fn test_box_cox_vs_sklearn() { ... }
    #[test] fn test_power_transform_zero_variance_column() { ... }
    #[test] fn test_power_transform_already_normal() { ... }
}

mod quantile_tests {
    #[test] fn test_quantile_uniform_vs_sklearn() { ... }
    #[test] fn test_quantile_normal_vs_sklearn() { ... }
    #[test] fn test_quantile_few_samples() { ... }
}

mod selection_tests {
    #[test] fn test_variance_threshold_vs_sklearn() { ... }
    #[test] fn test_variance_threshold_removes_constant() { ... }
    #[test] fn test_select_k_best_f_classif_vs_sklearn() { ... }
    #[test] fn test_select_k_best_f_regression_vs_sklearn() { ... }
    #[test] fn test_select_k_best_k_equals_n_features() { ... }
}

mod imputer_tests {
    #[test] fn test_simple_imputer_mean_vs_sklearn() { ... }
    #[test] fn test_simple_imputer_median_vs_sklearn() { ... }
    #[test] fn test_simple_imputer_most_frequent_vs_sklearn() { ... }
    #[test] fn test_knn_imputer_vs_sklearn() { ... }
    #[test] fn test_imputer_all_nan_column() { ... }
    #[test] fn test_imputer_no_missing_values() { ... }
}

mod encoder_tests {
    #[test] fn test_label_encoder_vs_sklearn() { ... }
    #[test] fn test_ordinal_encoder_vs_sklearn() { ... }
    #[test] fn test_onehot_encoder_vs_sklearn() { ... }
    #[test] fn test_onehot_encoder_drop_first_vs_sklearn() { ... }
    #[test] fn test_target_encoder_vs_sklearn() { ... }
    #[test] fn test_encoder_unseen_category() { ... }
}

mod resampling_tests {
    #[test] fn test_smote_class_balance() { ... }
    #[test] fn test_smote_sample_count() { ... }
    #[test] fn test_adasyn_class_balance() { ... }
    #[test] fn test_random_oversampler_exact_count() { ... }
}

mod edge_cases {
    #[test] fn test_single_sample() { ... }
    #[test] fn test_single_feature() { ... }
    #[test] fn test_constant_feature() { ... }
    #[test] fn test_all_nan_input() { ... }
    #[test] fn test_empty_input() { ... }
}
```

**Total: ~40 tests**

### Phase E.3: Bug Fixes (As Discovered)

Based on Phases 2-7 pattern, expect to find 5-10 bugs during correctness testing:
- Off-by-one errors in binning
- Incorrect lambda optimization in PowerTransformer
- Wrong interpolation in QuantileTransformer
- Edge cases in feature selection scoring

Each bug: fix in source, add regression test, mark as found.

## Success Criteria

- [ ] `cargo test -p ferroml-core --test correctness_preprocessing` — all pass
- [ ] PolynomialFeatures output matches sklearn within 1e-10
- [ ] KBinsDiscretizer bin edges match sklearn within 1e-8
- [ ] PowerTransformer lambdas match sklearn within 1e-4
- [ ] Feature selectors choose same features as sklearn
- [ ] Imputers produce same fill values as sklearn within 1e-10
- [ ] SMOTE/ADASYN produce correct class ratios

## Dependencies

- Python 3.10+ with sklearn >= 1.3 for fixture generation
- Existing fixture infrastructure in `benchmarks/fixtures/`
- No new Rust crate dependencies

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| PowerTransformer lambda optimization uses different algorithm than sklearn | Compare outputs not internals; use looser tolerance (1e-4) |
| QuantileTransformer interpolation differs | Test at exact quantile boundaries where no interpolation needed |
| SMOTE synthetic samples are stochastic | Test class ratios and sample counts, not exact synthetic points |
| Some transformers may have deep bugs requiring major rewrites | Document as ignored tests first, fix in dedicated follow-up |
