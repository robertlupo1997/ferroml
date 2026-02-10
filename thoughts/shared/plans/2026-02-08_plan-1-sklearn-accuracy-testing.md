# Plan 1: Sklearn Accuracy Testing

**Date:** 2026-02-08
**Reference:** `thoughts/shared/research/2026-02-08_ferroml-comprehensive-assessment.md`
**Priority:** High
**Estimated Tasks:** 8

## Objective

Create comprehensive accuracy comparison tests between FerroML and sklearn to validate numerical correctness. Record all results for documentation.

## Context

Current state (from research):
- 20 existing sklearn_correctness tests in Rust
- Python tests only check API compatibility, not numerical correctness
- No actual sklearn datasets loaded (synthetic only)
- Tolerances vary: 1e-10 (strict) to 1e-5 (loose)

## Tolerance Standards

| Algorithm Type | Tolerance | Rationale |
|----------------|-----------|-----------|
| Closed-form (Linear, PCA) | 1e-10 | Exact solution |
| Iterative (Lasso, Logistic) | 1e-4 | Convergence-dependent |
| Tree-based | 1e-6 | Deterministic splits |
| Probabilistic outputs | 1e-6 | Softmax stability |

## Tasks

### Task 1.1: Create sklearn baseline generator script
**File:** `benchmarks/sklearn_baselines.py`
**Description:** Python script that runs sklearn models on Iris, Wine, Diabetes and saves predictions/scores to JSON for Rust test comparison.
**Output:** `ferroml-core/tests/fixtures/sklearn_baselines.json`

### Task 1.2: Add actual sklearn datasets to Rust tests
**File:** `ferroml-core/src/datasets/toy.rs`
**Description:** Verify load_iris(), load_wine(), load_diabetes() match sklearn exactly (same feature values, same targets).
**Test:** Compare first 5 rows against sklearn values.

### Task 1.3: LinearRegression accuracy test
**File:** `ferroml-core/tests/sklearn_correctness.rs`
**Description:** Compare coefficients, intercept, R² on Diabetes dataset.
**Tolerance:** 1e-10
**Metrics:** coef_, intercept_, score()

### Task 1.4: LogisticRegression accuracy test
**File:** `ferroml-core/tests/sklearn_correctness.rs`
**Description:** Compare on Iris and Wine (multiclass).
**Tolerance:** 1e-4 (iterative)
**Metrics:** coef_, intercept_, predict(), predict_proba(), accuracy

### Task 1.5: DecisionTree accuracy tests
**File:** `ferroml-core/tests/sklearn_correctness.rs`
**Description:** Classifier on Iris/Wine, Regressor on Diabetes.
**Tolerance:** 1e-6
**Metrics:** feature_importances_, predict(), max_depth behavior

### Task 1.6: RandomForest accuracy tests
**File:** `ferroml-core/tests/sklearn_correctness.rs`
**Description:** With fixed random_state for reproducibility.
**Tolerance:** 1e-6
**Metrics:** feature_importances_, oob_score_, predict()

### Task 1.7: KNeighbors accuracy tests
**File:** `ferroml-core/tests/sklearn_correctness.rs`
**Description:** Classifier on Iris/Wine with various k values.
**Tolerance:** 1e-10 (deterministic)
**Metrics:** predict(), predict_proba(), kneighbors()

### Task 1.8: Create accuracy report generator
**File:** `benchmarks/generate_accuracy_report.py`
**Description:** Script that runs all comparisons and generates markdown report.
**Output:** `docs/accuracy-report.md`

## Success Criteria

- [ ] All core models tested against sklearn
- [ ] Results documented with exact tolerances achieved
- [ ] Any discrepancies explained (algorithm differences, not bugs)
- [ ] Accuracy report generated and saved

## Dependencies

- Python 3.10+ with sklearn installed
- FerroML Python bindings (ferroml-python)
