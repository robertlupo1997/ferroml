# Specification: Preprocessing

**Status**: 🔲 NOT IMPLEMENTED

## Overview

Feature preprocessing transformers with proper fit/transform separation.

## Requirements

### Imputers
- [ ] SimpleImputer (mean, median, mode, constant)
- [ ] KNNImputer

### Encoders
- [ ] OneHotEncoder (with drop_first option)
- [ ] OrdinalEncoder
- [ ] TargetEncoder (with smoothing)
- [ ] LabelEncoder

### Scalers
- [ ] StandardScaler (z-score)
- [ ] MinMaxScaler
- [ ] RobustScaler (median/IQR)
- [ ] MaxAbsScaler

### Feature Selection
- [ ] VarianceThreshold
- [ ] SelectKBest (with statistical tests)
- [ ] SelectFromModel
- [ ] RecursiveFeatureElimination

### Transformer Trait

```rust
pub trait Transformer: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()>;
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn fit_transform(&mut self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn inverse_transform(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn get_feature_names_out(&self) -> Vec<String>;
}
```

## Implementation Priority

1. StandardScaler (most common)
2. OneHotEncoder (categorical handling)
3. SimpleImputer (missing values)
4. SelectKBest (feature selection with statistics)

## Acceptance Criteria

- [ ] All transformers implement Transformer trait
- [ ] inverse_transform works correctly where applicable
- [ ] Feature names tracked through transformations
- [ ] Encoders handle unknown categories gracefully

## Implementation Location

`ferroml-core/src/preprocessing/`
