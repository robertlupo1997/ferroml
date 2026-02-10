# Plan 2: Doctest Fixes

**Date:** 2026-02-08
**Reference:** `thoughts/shared/research/2026-02-08_ferroml-comprehensive-assessment.md`
**Priority:** High
**Estimated Tasks:** 10

## Objective

Fix all 59 failing doctests to ensure documentation examples are accurate and runnable.

## Context

From research:
- 59 pre-existing doctest failures
- 142 doctests currently ignored
- Doctests provide executable documentation

## Tasks

### Task 2.1: Inventory all failing doctests
**Command:** `cargo test --doc 2>&1 | grep -E "(FAILED|error)" > doctest_failures.txt`
**Output:** List of all failing doctests with file:line locations

### Task 2.2: Fix models/ doctests
**Files:** `models/*.rs`
**Description:** Fix doctests for LinearRegression, LogisticRegression, tree models, etc.
**Estimate:** ~15 doctests

### Task 2.3: Fix preprocessing/ doctests
**Files:** `preprocessing/*.rs`
**Description:** Fix doctests for scalers, encoders, imputers.
**Estimate:** ~10 doctests

### Task 2.4: Fix metrics/ doctests
**Files:** `metrics/*.rs`
**Description:** Fix doctests for classification, regression, probabilistic metrics.
**Estimate:** ~8 doctests

### Task 2.5: Fix cv/ doctests
**Files:** `cv/*.rs`
**Description:** Fix doctests for cross-validation methods.
**Estimate:** ~6 doctests

### Task 2.6: Fix explainability/ doctests
**Files:** `explainability/*.rs`
**Description:** Fix doctests for SHAP, PDP, ICE, permutation importance.
**Estimate:** ~5 doctests

### Task 2.7: Fix hpo/ doctests
**Files:** `hpo/*.rs`
**Description:** Fix doctests for samplers, schedulers, Bayesian optimization.
**Estimate:** ~5 doctests

### Task 2.8: Fix stats/ doctests
**Files:** `stats/*.rs`
**Description:** Fix doctests for hypothesis tests, confidence intervals, effect sizes.
**Estimate:** ~5 doctests

### Task 2.9: Fix automl/ doctests
**Files:** `automl/*.rs`
**Description:** Fix doctests for AutoML components.
**Estimate:** ~3 doctests

### Task 2.10: Verify all doctests pass
**Command:** `cargo test --doc`
**Success:** 0 failures, reduced ignored count

## Success Criteria

- [ ] `cargo test --doc` shows 0 failures
- [ ] Each doctest demonstrates actual usage
- [ ] Doctests use realistic example data
- [ ] No doctests left ignored without reason

## Common Fixes Expected

1. Missing imports in doctest examples
2. Outdated API (method signatures changed)
3. Wrong expected output values
4. Missing `# use` statements for dependencies
5. Examples that panic instead of returning Result
