# Specification: Cross-Validation

**Status**: 🔲 NOT IMPLEMENTED

## Overview

CV strategies with proper statistical handling of results.

## Requirements

### CV Strategies
- [ ] KFold
- [ ] StratifiedKFold (preserves class distribution)
- [ ] TimeSeriesSplit (respects temporal ordering)
- [ ] GroupKFold (respects group boundaries)
- [ ] LeaveOneOut
- [ ] ShuffleSplit

### Nested CV
- [ ] NestedCV (outer for evaluation, inner for HPO)
  - Prevents data leakage
  - Returns unbiased performance estimate

### CV Results with Statistics

```rust
pub struct CVResult {
    pub scores: Array1<f64>,
    pub mean: f64,
    pub std: f64,
    pub ci: (f64, f64),  // 95% CI for mean score
    pub fold_results: Vec<FoldResult>,
}

pub struct FoldResult {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub train_score: f64,
    pub test_score: f64,
    pub fit_time: f64,
    pub score_time: f64,
}
```

### CrossValidator Trait

```rust
pub trait CrossValidator: Send + Sync {
    fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
    fn n_splits(&self) -> usize;
}
```

## Implementation Priority

1. KFold (baseline)
2. StratifiedKFold (classification)
3. TimeSeriesSplit (time series)
4. NestedCV (proper model selection)

## Acceptance Criteria

- [ ] Splits are identical to sklearn given same seed
- [ ] CI calculations validated against R
- [ ] NestedCV prevents data leakage
- [ ] Parallel fold execution with rayon

## Implementation Location

`ferroml-core/src/cv/`
