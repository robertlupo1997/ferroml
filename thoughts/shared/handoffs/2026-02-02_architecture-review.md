---
date: 2026-02-02T12:00:00-05:00
researcher: Claude
topic: Architecture Review & Code Quality Analysis
tags: [architecture, code-quality, technical-debt, bugs]
status: complete
---

# FerroML Architecture Review

## Executive Summary

FerroML has solid foundations with excellent statistical diagnostics, but contains **one critical bug** and ~400 lines of code duplication. The trait hierarchy is well-designed, but extended traits are underutilized.

## CRITICAL BUG FOUND

### Decision Tree MAE Criterion - WRONG ARRAY
**Location**: `ferroml-core/src/models/tree.rs`, lines 1537-1538

```rust
SplitCriterion::Mae => mae(&right_values),  // BUG: Should be &left_values
```

The MAE criterion for **left child impurity** incorrectly uses `right_values` instead of `left_values`. This breaks all MAE-based decision tree splits.

**Fix**: Change `&right_values` to `&left_values`

---

## Architecture Strengths

### 1. Well-Designed Trait Hierarchy
Three-tier model trait system in `models/mod.rs`:
- `Model` (base) - fit/predict/is_fitted
- `StatisticalModel` - summary/diagnostics/coefficients_with_ci
- `ProbabilisticModel` - predict_proba/predict_interval

### 2. Comprehensive Statistical Diagnostics
Linear/logistic models provide R-style output: coefficient SEs, t/z-statistics, p-values, assumption tests, influential observation detection. **Genuine differentiator vs sklearn**.

### 3. Consistent Builder Pattern
All models use fluent `with_*` builder APIs.

### 4. Structured Error Handling
`FerroError` provides domain-specific variants with context.

### 5. Preprocessing Trait Consistency
`Transformer` trait: fit/transform/inverse_transform + feature name tracking.

---

## Code Smells and Technical Debt

### 1. Duplicate ClassWeight Definition
Defined in both `svm.rs:196-209` AND exported from `mod.rs`. Confusing.

### 2. Duplicated Statistical Functions (~300 lines)
- `linear.rs`: t_critical, t_cdf_approx, f_cdf, chi_squared_cdf, ln_gamma
- `logistic.rs`: z_critical, normal_cdf, chi_squared_cdf (separate impl!)

Should consolidate into `stats/distributions.rs`.

### 3. Duplicate Bootstrap Code in Random Forest
`generate_bootstrap_indices()` defined twice (lines 367-379 AND 874-886).

### 4. Inconsistent `#[must_use]` Annotations
LogisticRegression has them, LinearRegression doesn't.

### 5. Underutilized Extended Traits
`traits.rs` defines LinearModel, IncrementalModel, WeightedModel, TreeModel, WarmStartModel - barely implemented.

---

## Inconsistencies

| Issue | Details |
|-------|---------|
| Probability output shapes | LogisticRegression: (n,1), others: (n, n_classes) |
| Validation helpers | Models vs preprocessing use different functions |
| Feature importance naming | `feature_importance()` vs `feature_importances_with_ci()` |
| Search space validation | No validation that param names match struct fields |

---

## Priority Improvements

### HIGH (Do First)
1. **Fix tree.rs MAE bug** - 5 min, correctness issue
2. **Consolidate ClassWeight** - 30 min, reduces confusion

### MEDIUM
3. Extract statistical distributions module (-300 lines)
4. Standardize probability outputs
5. Extract common ensemble code

### LOW
6. Add `#[must_use]` consistently
7. Implement extended traits fully
8. Unify validation helpers

---

## Action Items for Ralph Loop

1. `TASK-BUG-001`: Fix tree.rs line 1538 MAE bug (CRITICAL)
2. `TASK-REFACTOR-001`: Move ClassWeight to single location
3. `TASK-REFACTOR-002`: Extract stats functions to stats/distributions.rs
4. `TASK-REFACTOR-003`: Dedupe generate_bootstrap_indices()
