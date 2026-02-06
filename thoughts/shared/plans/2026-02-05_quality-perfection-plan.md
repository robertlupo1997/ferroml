# FerroML Quality Perfection Plan

## Overview

This plan addresses **165 issues** found across 14 validation agents, plus completes 2 remaining validations. The goal is to achieve production-quality statistical rigor, mathematical correctness, and robust error handling throughout the FerroML codebase.

## Current State

### Issues Found by Module

| Module | CRITICAL | HIGH | MEDIUM | LOW | Total |
|--------|----------|------|--------|-----|-------|
| Stats | 6 | 9 | 8 | 4 | 27 |
| HPO | 2 | 4 | 5 | 1 | 12 |
| Preprocessing | 3 | 4 | 6 | 0 | 13 |
| AutoML | 5 | 2 | 4 | 2 | 13 |
| Decomposition | 3 | 3 | 4 | 3 | 13 |
| Ensemble | 2 | 3 | 3 | 2 | 10 |
| Pipeline | 2 | 2 | 4 | 2 | 10 |
| Python Bindings | 2 | 2 | 5 | 3 | 12 |
| CV | 1 | 3 | 3 | 3 | 10 |
| Serialization | 1 | 2 | 6 | 3 | 12 |
| Explainability | 1 | 0 | 2 | 4 | 7 |
| ONNX/Inference | 1 | 2 | 3 | 4 | 10 |
| Metrics | 0 | 0 | 1 | 3 | 4 |
| Quality Hardening | 1 | 2 | 6 | 3 | 12 |
| **TOTAL** | **30** | **38** | **60** | **37** | **165** |

### Incomplete Validations
- Models validation (context exhausted)
- Testing Plan validation (context exhausted)

## Desired End State

- All 30 CRITICAL bugs fixed with mathematical correctness verified
- All 38 HIGH bugs fixed with proper error handling
- All MEDIUM/LOW issues addressed or documented as accepted limitations
- 100% of model implementations validated against specs
- Full test plan validation complete
- All tests passing
- Clippy clean with `suboptimal_flops` addressed

---

## Implementation Phases

### Stream A: Critical Mathematical Fixes (Parallel Stream)

#### Phase A.1: Statistical Foundations Overhaul
**Priority**: P0 - Blocks everything
**Estimated Effort**: 16 hours
**Parallel Agents**: 2

**Changes Required**:

1. **File**: `ferroml-core/src/stats/mod.rs`
   - Fix incomplete beta function using Lentz's continued fraction algorithm
   - Fix Fisher z-transform division by zero guard for r=±1
   - Fix skewness/kurtosis to use sample formula (Fisher's adjusted)
   - Add division by zero guards when std=0

2. **File**: `ferroml-core/src/stats/hypothesis.rs`
   - Fix Cohen's d variance formula: use (n1+n2-2) denominator
   - Fix t-test CI: use t-critical not z=1.96
   - Add power calculation using non-central t-distribution

3. **File**: `ferroml-core/src/stats/diagnostics.rs`
   - Implement proper D'Agostino-Pearson normality test
   - Fix quartile calculation with interpolation
   - Fix homoscedasticity test with proper F-distribution p-value

4. **File**: `ferroml-core/src/stats/bootstrap.rs`
   - Implement BCa (bias-corrected accelerated) confidence intervals
   - Fix percentile index calculation (off-by-one)

5. **File**: `ferroml-core/src/stats/multiple_testing.rs`
   - Fix Hochberg procedure multiplier formula

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core stats::`
- [ ] Automated: Compare p-values against scipy.stats reference
- [ ] Manual: Verify incomplete beta matches Wolfram Alpha for edge cases

---

#### Phase A.2: HPO Algorithm Fixes
**Priority**: P0 - Core optimization broken
**Estimated Effort**: 12 hours
**Parallel Agents**: 2

**Changes Required**:

1. **File**: `ferroml-core/src/hpo/samplers.rs`
   - **REWRITE TPE Sampler**: Implement true Tree-Parzen Estimator
     - Fit KDE l(x) for good trials, g(x) for bad trials
     - Sample from l(x), return max l(x)/g(x) candidate
     - Reuse KDE from BOHB in schedulers.rs
   - Fix log-scale parameter validation (bounds > 0)
   - Fix GridSampler division by zero when grid_points=1
   - Fix Box-Muller transform: clamp u1 > MIN_POSITIVE
   - Fix RandomSampler RNG seeding

2. **File**: `ferroml-core/src/hpo/schedulers.rs`
   - Fix Hyperband: change ceil() to floor().max(1)

3. **File**: `ferroml-core/src/hpo/mod.rs`
   - Fix correlation() NaN on zero variance
   - Fix parameter importance CI calculation

4. **File**: `ferroml-core/src/hpo/bayesian.rs`
   - Fix Cholesky epsilon threshold

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core hpo::`
- [ ] Automated: TPE outperforms random on benchmark functions
- [ ] Manual: Compare Hyperband bracket sizes against paper

---

#### Phase A.3: AutoML Orchestration Fixes
**Priority**: P0 - AutoML completely broken
**Estimated Effort**: 20 hours
**Parallel Agents**: 2

**Changes Required**:

1. **File**: `ferroml-core/src/lib.rs`
   - Add missing fields to AutoML struct: portfolio, study
   - Add AlgorithmSelection enum: Uniform, Bayesian, Bandit
   - Add to AutoMLConfig: algorithm_selection, ensemble_size, preset

2. **File**: `ferroml-core/src/automl/fit.rs`
   - **FIX BANDIT SELECTION**: Use allocation.algorithm_index, not priority order
   - Fix wrong index in time_allocator.update()
   - Add best_model to AutoMLResult
   - Use config.multiple_testing_correction instead of hardcoded
   - Use config.preset instead of hardcoded Balanced

3. **File**: `ferroml-core/src/automl/time_budget.rs`
   - **IMPLEMENT SUCCESSIVE HALVING**: Proper bracket-based elimination
     - Use min_resource, max_resource, eta parameters
     - Eliminate bottom 1/eta configs each round
     - Multiply resource by eta

4. **File**: `ferroml-core/src/automl/ensemble.rs`
   - Unify ParamValue and ParameterValue types

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core automl::`
- [ ] Automated: Verify bandit arm selection distribution
- [ ] Manual: AutoML.fit() returns usable best_model

---

#### Phase A.4: Decomposition Mathematical Fixes
**Priority**: P0 - LDA completely wrong
**Estimated Effort**: 10 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-core/src/decomposition/lda.rs`
   - **FIX EIGENVALUE SOLVER**: Use Schur decomposition or QZ algorithm
     - S_w^{-1} S_b is NOT symmetric
   - Fix between-class scatter: use actual class counts, not priors
   - Document non-standard shrinkage formula

2. **File**: `ferroml-core/src/decomposition/pca.rs`
   - Fix whitening division by zero for near-zero singular values
   - Fix IncrementalPCA mean inconsistency in batch updates
   - Fix IncrementalPCA whitening sample count
   - Fix Gram-Schmidt to true modified (project against current v)

3. **File**: `ferroml-core/src/decomposition/factor_analysis.rs`
   - Fix log-determinant singular matrix handling
   - Add parameter convergence check to EM
   - Document Promax rotation variant

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core decomposition::`
- [ ] Automated: Compare LDA with sklearn on iris dataset
- [ ] Manual: Verify PCA whitening produces unit variance

---

### Stream B: Core Module Fixes (Parallel Stream)

#### Phase B.1: Preprocessing Fixes
**Priority**: P1
**Estimated Effort**: 12 hours
**Parallel Agents**: 2

**Changes Required**:

1. **File**: `ferroml-core/src/preprocessing/imputers.rs`
   - Fix KNNImputer distance: sqrt(sum * n_features / valid_count)
   - Fix SimpleImputer mode: use ordered float wrapper

2. **File**: `ferroml-core/src/preprocessing/scalers.rs` & `mod.rs`
   - Fix StandardScaler: use population variance (n), not sample (n-1)

3. **File**: `ferroml-core/src/preprocessing/selection.rs`
   - Fix Chi2 test: require count data or implement binning
   - Implement full RFE algorithm
   - Fix F-distribution CDF for edge cases

4. **File**: `ferroml-core/src/preprocessing/encoders.rs`
   - Add sparse matrix support to OneHotEncoder
   - Fix TargetEncoder CV: add shuffling

5. **File**: `ferroml-core/src/preprocessing/power.rs`
   - Fix Yeo-Johnson inverse transform branch detection

6. **File**: `ferroml-core/src/preprocessing/sampling.rs`
   - Implement Borderline-2 SMOTE correctly

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core preprocessing::`
- [ ] Automated: StandardScaler matches sklearn output exactly
- [ ] Manual: KNNImputer distance correct for mixed missing patterns

---

#### Phase B.2: Ensemble Fixes
**Priority**: P1
**Estimated Effort**: 8 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-core/src/ensemble/stacking.rs`
   - **FIX**: Actually use user's final_estimator (currently ignored!)
   - Fix default meta-learner: LogisticRegression for classifier
   - Fix predict_proba: return actual probabilities, not one-hot

2. **File**: `ferroml-core/src/ensemble/bagging.rs`
   - **FIX**: Clone user's base_estimator (currently always DecisionTree)

3. **File**: `ferroml-core/src/ensemble/voting.rs`
   - Fix tie-breaking: prefer smaller index (sklearn behavior)

4. **Add**: Ensemble weight confidence intervals

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core ensemble::`
- [ ] Automated: StackingClassifier uses custom final_estimator
- [ ] Manual: BaggingRegressor with LinearRegression base works

---

#### Phase B.3: Pipeline Fixes
**Priority**: P1
**Estimated Effort**: 8 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-core/src/pipeline/mod.rs`
   - **FIX CACHE KEY**: Use input BEFORE transformation, not original x
   - **FIX fit_transform**: Single pass, don't double-compute
   - Fix FeatureUnion: O(n) lookup instead of O(n²)
   - Fix Remainder selector: update used_columns after each
   - Fix ColumnSelector::Mask length validation
   - Fix hash_array: consider hashing all values or documenting limitation
   - Add CacheStrategy::Disk option per spec

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core pipeline::`
- [ ] Automated: Cache hit rate > 0 for multi-step pipeline
- [ ] Manual: fit_transform performance matches single pass

---

#### Phase B.4: CV Fixes
**Priority**: P1
**Estimated Effort**: 6 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-core/src/cv/mod.rs`
   - Fix CI calculation for RepeatedKFold: apply Nadeau-Bengio correction
   - Add scores field/accessor to CVResult per spec

2. **File**: `ferroml-core/src/cv/stratified.rs`
   - Fix stratification to match sklearn zigzag interleaving

3. **File**: `ferroml-core/src/cv/loo.rs`
   - Fix n_choose_k overflow: use checked arithmetic
   - Fix ShuffleSplit rounding: use floor()

4. **File**: `ferroml-core/src/cv/timeseries.rs`
   - Move gap validation before loop

5. **File**: `ferroml-core/src/cv/nested.rs`
   - Pass groups to inner CV loop

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core cv::`
- [ ] Automated: StratifiedKFold matches sklearn splits
- [ ] Manual: RepeatedKFold CI width is reasonable

---

### Stream C: Interface & Export Fixes (Parallel Stream)

#### Phase C.1: Python Bindings Fixes
**Priority**: P1
**Estimated Effort**: 10 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-python/src/automl.rs`
   - Add predict() method to PyAutoMLResult
   - Add cv parameter to AutoML.fit()
   - Fix multiple_testing_from_str: return error on invalid

2. **File**: `ferroml-python/src/pipeline.rs`
   - Fix fit_transform behavior when last step is model

3. **File**: `ferroml-python/src/sparse_utils.rs`
   - Fix index overflow: use try_from with error handling

4. **File**: `ferroml-python/src/array_utils.rs`
   - Fix usize to i64 overflow

5. **File**: `ferroml-python/src/pickle.rs`
   - Add __reduce__ for full pickle support

**Success Criteria**:
- [ ] Automated: `maturin develop && pytest`
- [ ] Automated: result.predict(X_test) works after AutoML
- [ ] Manual: Sparse matrix with negative index gives clear error

---

#### Phase C.2: ONNX/Inference Fixes
**Priority**: P1
**Estimated Effort**: 6 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-core/src/onnx/tree.rs`
   - **FIX**: Change aggregate_function from "SUM" to "AVERAGE"

2. **File**: `ferroml-core/src/inference/operators.rs`
   - Fix Reshape: handle -1 dimension properly
   - Fix SqueezeOp: squeeze all dims of size 1 when axes empty
   - Fix BranchEq: use relative tolerance

3. **File**: `ferroml-core/src/inference/session.rs`
   - Fix unknown data type: return error, not zero tensor
   - Add support for DOUBLE, INT32 data types

4. **File**: `ferroml-core/src/onnx/linear.rs`
   - Add warning for f64->f32 precision loss

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core onnx:: inference::`
- [ ] Automated: RF ONNX predictions match native predictions
- [ ] Manual: External ONNX model with reshape -1 loads

---

#### Phase C.3: Serialization Fixes
**Priority**: P1
**Estimated Effort**: 8 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-core/src/serialization.rs`
   - Add version compatibility validation on load
   - Add deserialization size limits (100MB default)
   - Add NaN/Inf validation for JSON format
   - Add model type validation
   - Add checksum support (optional)
   - Implement efficient peek_metadata for JSON
   - Add streaming serialization API

2. **File**: `ferroml-core/src/models/linear.rs`
   - Add compact serialization option (skip diagnostic data)

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core serialization::`
- [ ] Automated: Version mismatch gives clear error
- [ ] Manual: JSON with NaN gives helpful error message

---

### Stream D: Explainability & Metrics (Parallel Stream)

#### Phase D.1: Explainability Fixes
**Priority**: P2
**Estimated Effort**: 16 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-core/src/explainability/treeshap.rs`
   - **REWRITE**: Implement true TreeSHAP algorithm (Lundberg 2018)
     - Extended path tracking with weight vectors
     - Recursive EXTEND, UNWIND operations
     - Remove post-hoc normalization workaround

2. **File**: `ferroml-core/src/explainability/kernelshap.rs`
   - Fix coalition sampling: use importance sampling proportional to kernel weights

3. **File**: `ferroml-core/src/explainability/permutation.rs`
   - Add multi-output support

4. **File**: `ferroml-core/src/explainability/summary.rs`
   - Fix variance: use sample variance (n-1)

5. **File**: `ferroml-core/src/explainability/h_statistic.rs`
   - Document H-statistic variant used

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core explainability::`
- [ ] Automated: TreeSHAP values sum correctly without normalization
- [ ] Manual: Compare SHAP values with shap library

---

#### Phase D.2: Metrics Fixes
**Priority**: P2
**Estimated Effort**: 2 hours
**Parallel Agents**: 1

**Changes Required**:

1. **File**: `ferroml-core/src/metrics/probabilistic.rs`
   - Fix PR-AUC: add precision interpolation for monotonicity

2. **File**: `ferroml-core/src/metrics/mod.rs`
   - Fix bootstrap CI percentile indexing

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core metrics::`
- [ ] Automated: PR-AUC matches sklearn average_precision_score

---

### Stream E: Quality & Validation Completion

#### Phase E.1: Complete Remaining Validations
**Priority**: P1
**Estimated Effort**: 8 hours
**Parallel Agents**: 4

**Work Required**:

1. **Models Validation (focused)**
   - Validate Linear models (OLS, Ridge, Lasso, Logistic)
   - Validate Tree models (DecisionTree, RandomForest, GradientBoosting)
   - Validate SVM models
   - Validate Naive Bayes
   - Validate KNN

2. **Testing Plan Validation (focused)**
   - Verify all 32 phases implemented
   - Check test quality and coverage
   - Identify gaps in edge case coverage

**Success Criteria**:
- [ ] Automated: All validation agents complete
- [ ] Manual: Issues documented and added to task list

---

#### Phase E.2: Quality Hardening - unwrap() Elimination
**Priority**: P1
**Estimated Effort**: 40+ hours
**Parallel Agents**: 4 (by module)

**Changes Required**:

1. **Models Module** (999 unwraps - highest priority)
   - `naive_bayes.rs`: 170 unwraps → Result propagation
   - `svm.rs`: 141 unwraps → Result propagation
   - `knn.rs`: 89 unwraps → Result propagation
   - `hist_boosting.rs`: 89 unwraps → Result propagation
   - `tree.rs`: 85 unwraps → Result propagation

2. **Other Modules** (~1,776 unwraps)
   - Systematic conversion to ? operator
   - Add context with .context() from anyhow

3. **expect() Messages** (112 calls)
   - Add descriptive context to all expect() calls

4. **panic!() Removal** (58 calls)
   - Convert to Result::Err in library code

**Success Criteria**:
- [ ] Automated: `cargo clippy -- -A clippy::unwrap_used` passes
- [ ] Automated: No panics in library code (test with panic=abort)
- [ ] Manual: Error messages are actionable

---

#### Phase E.3: Numerical Stability - mul_add Optimization
**Priority**: P2
**Estimated Effort**: 8 hours
**Parallel Agents**: 2

**Changes Required**:

1. Address 285 remaining `suboptimal_flops` warnings
2. Priority files:
   - `simd.rs` (performance critical)
   - `bayesian.rs` (numerical precision)
   - `linear.rs` (core computations)
   - `stats/*.rs` (statistical accuracy)

**Success Criteria**:
- [ ] Automated: `cargo clippy -- -W clippy::suboptimal_flops` passes
- [ ] Manual: Numerical accuracy tests pass

---

#### Phase E.4: Remaining TODOs and Polish
**Priority**: P3
**Estimated Effort**: 8 hours
**Parallel Agents**: 1

**Changes Required**:

1. Complete 4 remaining TODOs:
   - `models/linear.rs:403` - Condition number computation
   - `models/compliance_tests/compliance.rs:169` - check_classifier function
   - `testing/incremental.rs:1056` - WarmStartModel tests
   - `stats/hypothesis.rs:193` - Power calculation

2. Enable NaN/Inf handling in model compliance tests

3. Add missing examples in public API documentation

**Success Criteria**:
- [ ] Automated: `grep -r "TODO" src/ | wc -l` returns 0
- [ ] Manual: All public APIs have examples

---

## Parallelization Strategy

### Wave 1 (Critical Fixes - Week 1-2)
Run in parallel:
- Stream A: Phases A.1, A.2, A.3, A.4 (4 parallel agents)
- Stream E.1: Validation completion (4 parallel agents)

### Wave 2 (Core Fixes - Week 2-3)
Run in parallel:
- Stream B: Phases B.1, B.2, B.3, B.4 (4 parallel agents)
- Stream C: Phases C.1, C.2, C.3 (3 parallel agents)

### Wave 3 (Polish - Week 3-4)
Run in parallel:
- Stream D: Phases D.1, D.2 (2 parallel agents)
- Stream E: Phases E.2, E.3, E.4 (3 parallel agents)

---

## Dependencies

```
A.1 (Stats) ──────────┐
A.2 (HPO) ────────────┼──→ E.2 (Quality)
A.3 (AutoML) ─────────┤
A.4 (Decomposition) ──┘

B.1 (Preprocessing) ──┐
B.2 (Ensemble) ───────┼──→ C.1 (Python)
B.3 (Pipeline) ───────┤
B.4 (CV) ─────────────┘

C.2 (ONNX) ───────────→ Final Integration Tests
C.3 (Serialization) ──→ Final Integration Tests

E.1 (Validation) ─────→ May discover more issues
```

---

## Risks & Mitigations

### Risk 1: TreeSHAP rewrite is complex
**Mitigation**: Reference Lundberg's original C++ implementation, start with single-tree case

### Risk 2: unwrap() elimination is massive (3,807 calls)
**Mitigation**: Prioritize by module, automate with sed where patterns are clear

### Risk 3: Some fixes may break existing tests
**Mitigation**: Run full test suite after each phase, fix regressions immediately

### Risk 4: Additional issues found during remaining validations
**Mitigation**: Reserve 20% buffer time, create new tasks as discovered

---

## Verification Commands

```bash
# Full test suite
cargo test -p ferroml-core

# Clippy with all warnings
cargo clippy -p ferroml-core -- -D warnings -W clippy::suboptimal_flops

# Python bindings
cd ferroml-python && maturin develop && pytest

# Specific module tests
cargo test -p ferroml-core stats::
cargo test -p ferroml-core hpo::
cargo test -p ferroml-core automl::
# ... etc

# Compare with sklearn (integration)
python tests/sklearn_comparison.py
```

---

## Estimated Total Effort

| Stream | Phases | Hours |
|--------|--------|-------|
| A (Critical Math) | 4 | 58 |
| B (Core Modules) | 4 | 34 |
| C (Interfaces) | 3 | 24 |
| D (Explain/Metrics) | 2 | 18 |
| E (Quality) | 4 | 64 |
| **Total** | **17** | **198 hours** |

With 4 parallel agents, estimated calendar time: **~2-3 weeks**
