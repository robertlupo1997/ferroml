# Plan L: Testing Phases 23-28

## Overview

Execute the 6 remaining testing phases from the implementation plan. Each phase creates a new test module in `ferroml-core/src/testing/`. These test existing functionality — no new features needed.

## Current State

- Testing infrastructure: 23 test modules in `ferroml-core/src/testing/`
- 2,471 Rust tests passing
- Phases 16-22 complete; Phases 23-28 not started
- Each phase has 6-7 tasks (create module, register, write test categories)

## Desired End State

- 6 new test modules created and registered
- ~150 additional Rust tests
- All pass with `cargo test -p ferroml-core`

---

## Phase L.1: Multi-output Prediction Tests (Plan Phase 23)

**Overview**: Test multi-output prediction capabilities across models.

**Changes Required**:

1. **File**: `ferroml-core/src/testing/multioutput.rs` (NEW, ~200 lines)
   - Test models that support multi-output (if any — check which models handle it)
   - Test prediction shape validation
   - Test per-output metric computation
   - If no models support multi-output natively, test the workaround pattern (fitting separate models)

2. **File**: `ferroml-core/src/testing/mod.rs` (EDIT)
   - Add `pub mod multioutput;`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core testing::multioutput` — all pass
- [ ] ~20-25 tests

---

## Phase L.2: Advanced Cross-validation Tests (Plan Phase 24)

**Overview**: Test advanced CV strategies: NestedCV, GroupKFold, TimeSeriesSplit, learning_curve, validation_curve.

**Changes Required**:

1. **File**: `ferroml-core/src/testing/cv_advanced.rs` (NEW, ~300 lines)
   - NestedCV: verify no data leakage between inner/outer folds
   - GroupKFold: verify group integrity (same group never in both train and test)
   - TimeSeriesSplit: verify temporal ordering preserved
   - learning_curve: verify train sizes increase and scores are valid
   - validation_curve: verify parameter range is tested correctly

2. **File**: `ferroml-core/src/testing/mod.rs` (EDIT)
   - Add `pub mod cv_advanced;`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core testing::cv_advanced` — all pass
- [ ] ~25-30 tests

---

## Phase L.3: Ensemble Stacking Tests (Plan Phase 25)

**Overview**: Test StackingClassifier/StackingRegressor meta-learner behavior.

**Changes Required**:

1. **File**: `ferroml-core/src/testing/ensemble_advanced.rs` (NEW, ~250 lines)
   - StackingClassifier: verify meta-learner uses base learner predictions
   - StackingRegressor: same for regression
   - Passthrough option: verify original features are included
   - CV-based stacking: verify no data leakage in cross-validated predictions
   - Ensemble weights: verify weighted combination works

2. **File**: `ferroml-core/src/testing/mod.rs` (EDIT)
   - Add `pub mod ensemble_advanced;`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core testing::ensemble_advanced` — all pass
- [ ] ~25-30 tests

---

## Phase L.4: Categorical Feature Handling Tests (Plan Phase 26)

**Overview**: Test categorical feature handling across preprocessing and models.

**Changes Required**:

1. **File**: `ferroml-core/src/testing/categorical.rs` (NEW, ~250 lines)
   - HistGradientBoosting native categorical handling
   - Ordered target encoding correctness
   - Unknown category handling (OneHotEncoder, OrdinalEncoder)
   - Mixed categorical/numeric feature pipelines
   - ColumnTransformer with different encoders per column type

2. **File**: `ferroml-core/src/testing/mod.rs` (EDIT)
   - Add `pub mod categorical;`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core testing::categorical` — all pass
- [ ] ~20-25 tests

---

## Phase L.5: Warm Start / Incremental Learning Tests (Plan Phase 27)

**Overview**: Test warm start and incremental learning capabilities.

**Changes Required**:

1. **File**: `ferroml-core/src/testing/incremental.rs` (NEW, ~200 lines)
   - partial_fit for NaiveBayes models (if supported)
   - warm_start for ensemble models (if WarmStartModel trait is implemented)
   - Knowledge preservation: warm-started model retains previous learning
   - Online learning scenarios: sequential batch updates
   - Note: If WarmStartModel trait exists but isn't implemented, test what IS available

2. **File**: `ferroml-core/src/testing/mod.rs` (EDIT)
   - Add `pub mod incremental;`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core testing::incremental` — all pass
- [ ] ~15-20 tests

---

## Phase L.6: Custom Metrics Tests (Plan Phase 28)

**Overview**: Test custom scoring functions in CV and HPO.

**Changes Required**:

1. **File**: `ferroml-core/src/testing/metrics_custom.rs` (NEW, ~250 lines)
   - Custom scoring function works in cross_val_score
   - Custom metric works in GridSearchCV/RandomizedSearchCV
   - Multi-objective optimization (if supported)
   - Metric with confidence intervals (bootstrap-based)
   - Standard metrics (accuracy, f1, r2) produce expected values

2. **File**: `ferroml-core/src/testing/mod.rs` (EDIT)
   - Add `pub mod metrics_custom;`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core testing::metrics_custom` — all pass
- [ ] ~20-25 tests

---

## Execution Order

All 6 phases are **independent** and can run in parallel:
```
L.1 (multioutput)         — 20 min
L.2 (cv_advanced)         — 25 min
L.3 (ensemble_advanced)   — 25 min
L.4 (categorical)         — 20 min
L.5 (incremental)         — 20 min
L.6 (metrics_custom)      — 20 min
```

## Dependencies

- All test existing functionality — no new implementations needed
- Must understand which features actually exist (some tasks may be N/A if the feature isn't implemented)

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Some tested features may not exist | Write tests for what's available; document gaps as future work |
| WarmStartModel may not be implemented on any models | Test whatever incremental learning IS available |
| Multi-output may not be natively supported | Test single-output models and document the gap |
| Custom metrics API may differ from plan expectations | Read actual API before writing tests |

## Important Note for Implementation

Before writing each test module, **read the actual code** for the features being tested. The implementation plan was written before all code was complete — some features may work differently than described in the plan tasks. Adapt tests to match actual API, not plan assumptions.
