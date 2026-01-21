# Specification: Ensemble Methods

**Status**: 🔲 NOT IMPLEMENTED

## Overview

Meta-learners for combining models.

## Requirements

### Voting
- [ ] VotingClassifier (hard/soft voting)
- [ ] VotingRegressor (averaging)
- [ ] Weighted voting

### Stacking
- [ ] StackingClassifier
- [ ] StackingRegressor
- [ ] Uses CV to prevent leakage
- [ ] Configurable meta-learner

### Bagging
- [ ] BaggingClassifier
- [ ] BaggingRegressor
- [ ] OOB score estimation
- [ ] Bootstrap sampling

### Ensemble Traits

```rust
pub struct VotingClassifier {
    pub estimators: Vec<Box<dyn Model>>,
    pub voting: Voting,  // Hard, Soft
    pub weights: Option<Vec<f64>>,
}

pub struct StackingClassifier {
    pub estimators: Vec<Box<dyn Model>>,
    pub final_estimator: Box<dyn Model>,
    pub cv: Box<dyn CrossValidator>,
    pub passthrough: bool,  // Include original features
}
```

## Implementation Priority

1. VotingClassifier (simplest)
2. StackingClassifier (powerful)
3. BaggingClassifier (with OOB)

## Acceptance Criteria

- [ ] Stacking uses proper CV to prevent leakage
- [ ] OOB scores computed for bagging
- [ ] Ensemble weights have CIs
- [ ] Parallel training of base estimators

## Implementation Location

`ferroml-core/src/ensemble/`
