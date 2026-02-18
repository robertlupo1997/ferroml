# Plan: Fix All Vibecoded Issues

## Overview
Comprehensive fix for all ~30 issues discovered in the codebase audit — silent bugs, placeholder implementations, incorrect algorithms, weak tests, and missing trait implementations.

## Current State
The codebase compiles and passes 2469 lib tests + 215 doctests, but contains several implementations that silently produce wrong results or are incomplete stubs. Commit `ef0b46b` is latest on master.

## Desired End State
Every public API produces correct results. No placeholders, no fake computations, no silently dropped parameters.

## Verification Commands
```bash
cargo test -p ferroml-core --lib    # 2469+ tests pass
cargo clippy -p ferroml-core -- -D warnings  # clean
```

---

## Phase 1: Algorithm Correctness Bugs (Critical)

These silently produce wrong results and must be fixed first.

### 1a: Tree MAE impurity uses median instead of mean

**File**: `ferroml-core/src/models/tree.rs` (~line 145-159)
- `mae()` computes mean absolute deviation from **median**
- CART trees should use `Σ|y - mean(y)|/n` per partition
- Fix: change reference point from median to mean
- Add test: `test_mae_impurity_uses_mean`

### 1b: AdaBoost SAMME learning rate doesn't scale ln(K-1)

**File**: `ferroml-core/src/models/adaboost.rs` (~line 176-177)
- Current: `lr * ln((1-err)/err) + ln(K-1)` — learning rate only scales first term
- Correct: `lr * (ln((1-err)/err) + ln(K-1))` — scale entire expression
- Bug masked in binary classification (K=2) since ln(1)=0
- Add test: `test_samme_multiclass_learning_rate`

### 1c: StackingClassifier.predict_proba() returns one-hot instead of probabilities

**File**: `ferroml-core/src/ensemble/stacking.rs` (~line 268-297)
- Converts discrete predictions to `[0,1,0]` one-hot vectors
- Should call probabilistic interface on the final estimator
- Add test: `test_stacking_predict_proba_returns_probabilities`

### 1d: KNNImputer distance scaling inflates distances

**File**: `ferroml-core/src/preprocessing/imputers.rs` (~line 687-694)
- Scales distances by `n_features/valid_count` when features are missing
- This artificially inflates distances when many features are missing
- Fix: remove the scaling factor, use raw valid-feature distances
- Add test: `test_knn_imputer_missing_features_distance`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib` passes
- [ ] New tests for each fix

---

## Phase 2: AutoML Fixes (Critical)

### 2a: Feature importance always returns uniform 1/n

**File**: `ferroml-core/src/automl/ensemble.rs`
- Add `feature_importances: Option<Vec<f64>>` field to `TrialResult`
- Add `with_feature_importances()` builder method

**File**: `ferroml-core/src/automl/fit.rs`
- In `run_trial()` (~line 1050): after fitting on last fold, call `model.feature_importance()` and store via `.with_feature_importances()`
- In `compute_aggregated_feature_importance()` (~line 1267): use actual importances when available, fall back to uniform only for models that don't support it

### 2b: Dead code cleanup — unused parameters

**File**: `ferroml-core/src/automl/fit.rs`
- Remove `_maximize` parameter from `run_trial()` (direction already in `MetricAdapter`)
- Update call site (~line 855)

**File**: `ferroml-core/src/automl/ensemble.rs`
- Document `params` field: "Reserved for future HPO integration; currently always empty"

**File**: `ferroml-core/src/automl/transfer.rs`
- Remove `_search_space` from `WarmStartSampler::new()` and `from_prior()` signatures (callers supply at `sample()` time)

### 2c: Wire RobustRegression into AutoML

**File**: `ferroml-core/src/automl/fit.rs` (~line 1148)
- Replace `NotImplemented` error with `Ok(Box::new(RobustRegression::new()))`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib` passes
- [ ] `cargo clippy -p ferroml-core -- -D warnings` clean

---

## Phase 3: Statistical Tests — Correct Formulas (Critical)

### 3a: Rename Shapiro-Wilk proxy

**File**: `ferroml-core/src/stats/diagnostics.rs` (~line 97)
- Current formula uses skewness/kurtosis — this is a Jarque-Bera variant, not Shapiro-Wilk
- Rename from `"Shapiro-Wilk (approximation)"` to `"Normality (skewness-kurtosis)"`
- Fix p-value: current `(-5.0 * (1.0 - w)).exp()` is arbitrary
- Use chi-squared survival with df=2: compute `k2 = z_s² + z_k²`, then `p = exp(-k2/2)`

### 3b: Fix homoscedasticity p-value

**File**: `ferroml-core/src/stats/diagnostics.rs` (~line 198)
- Replace hardcoded `p_value: 0.5` with proper F-distribution p-value
- Need incomplete beta function for F-CDF — check if `stats/` already has one
- If not, implement `f_test_pvalue(f_stat, df1, df2)` using regularized incomplete beta
- Update `is_homoscedastic` to use `p_value > 0.05` instead of `f_ratio < 3.0`
- Add test: `test_homoscedasticity_pvalue_varies`

### 3c: Fix D'Agostino-Pearson comment

**File**: `ferroml-core/src/stats/hypothesis.rs` (~line 324)
- The formula `(-k2 / 2.0).exp()` is actually correct for χ²(2) survival
- Change comment from "Simplified" to "chi-squared survival function with df=2"

### 3d: Verify t_cdf sign handling

**File**: `ferroml-core/src/stats/hypothesis.rs` (~line 358)
- The `copysign(t)` logic for negative t values may produce wrong p-values
- Add test with known values: verify `t_cdf(-2.0, 10)` ≈ 0.0367 (scipy reference)
- Fix if incorrect

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib` passes
- [ ] New tests for homoscedasticity p-value and t_cdf

---

## Phase 4: HPO — Acquisition Function Bugs (High)

### 4a: LCB acquisition not negated for L-BFGS minimizer

**File**: `ferroml-core/src/hpo/bayesian.rs` (~line 1843-1880)
- UCB is correctly negated (max→min for L-BFGS), but LCB is passed through raw
- L-BFGS minimizes `μ - κσ` toward -∞ instead of finding useful lower confidence bounds
- Fix: negate LCB like UCB, or clarify semantics and fix accordingly
- Add test: `test_lcb_acquisition_finds_reasonable_point`

### 4b: L-BFGS unused convergence variables

**File**: `ferroml-core/src/hpo/bayesian.rs` (~line 1248-1300)
- `x_prev`/`g_prev` updated every iteration then `let _ = (x_prev, g_prev)`
- Add parameter-change convergence check: `if norm(x - x_prev) < xtol { break; }`
- Remove `let _ =` suppression
- Add `xtol` to L-BFGS config (default 1e-8)

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib` passes
- [ ] No `let _ =` warning suppressions in L-BFGS

---

## Phase 5: Robust Regression — HuberProposal2 (High)

**File**: `ferroml-core/src/models/robust.rs` (~line 758)
- `ScaleMethod::HuberProposal2` match arm just calls `compute_mad()` — identical to MAD
- Implement iterative Huber Proposal 2 scale estimator:
  1. Initialize σ₀ = MAD
  2. Iterate: σ² = (1/n) · Σ ρ(rᵢ/σ) / E[ρ(Z)] where Z~N(0,1)
  3. Converge when |σ_new - σ_old| < tol, cap at 20 iterations
- Tuning constant k from existing `LossFunction` (Huber default k=1.345)
- Add test: `test_huber_proposal2_differs_from_mad`

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib` passes
- [ ] New test passes

---

## Phase 6: Missing Trait Implementations (Medium)

### 6a: LinearModel on LinearRegression

**File**: `ferroml-core/src/models/regularized.rs`
- Implement `LinearModel` trait: `coefficients()` → return fitted weights, `intercept()` → return bias
- `coefficient_std_errors()` and `coefficient_intervals()` can return `None`

**File**: `ferroml-core/src/testing/properties.rs` (~line 790)
- Uncomment the two LinearModel tests

### 6b: WarmStartModel on ensemble models

**Files**: `ferroml-core/src/models/forest.rs`, `ferroml-core/src/models/boosting.rs`
- Implement `WarmStartModel` on RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
- `set_warm_start()`: set flag, `warm_start()`: get flag, `n_estimators_fitted()`: return tree count

**File**: `ferroml-core/src/testing/incremental.rs`
- Update module doc to remove "not yet implemented" note

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib` passes
- [ ] LinearModel tests uncommented and passing
- [ ] WarmStartModel on ≥2 ensemble models

---

## Phase 7: Edge Cases & Statistical Quality (Medium)

### 7a: Nested CV seed derivation

**File**: `ferroml-core/src/cv/nested.rs` (~line 517)
- Current: `seed + outer_fold_index` creates linearly correlated RNG sequences
- Fix: `seed.wrapping_mul(6364136223846793005).wrapping_add(outer_fold_index as u64)`

### 7b: SimpleImputer mode tie-breaking

**File**: `ferroml-core/src/preprocessing/imputers.rs` (~line 305-332)
- HashMap iteration is non-deterministic; tied modes return arbitrary values
- Fix: when multiple modes tie, return the smallest value (matches scikit-learn)

### 7c: TargetEncoder target leakage for tiny datasets

**File**: `ferroml-core/src/preprocessing/encoders.rs` (~line 1013-1029)
- When `n_samples < 2`, falls back to transform without out-of-fold protection
- Fix: return error for n_samples < 2, or use leave-one-out encoding

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib` passes

---

## Phase 8: Weak Tests & Cleanup (Low)

1. **`cv/group.rs:794`** — Replace `let _ = all_same` with `assert!(!all_same, "Different seeds should produce different folds")`
2. **`cv/loo.rs:644`** — Add `assert_eq!(cv, cv2)` after deserialization
3. **`preprocessing/mod.rs:52`** — Remove stale comment "to be implemented in subsequent tasks"
4. **`stats/power.rs:1-2`** — Remove "Placeholder" from module doc
5. **`cv/stratified.rs:153-159`** — Derive shuffle seed from class value instead of class index for stability

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --lib` passes
- [ ] `cargo clippy -p ferroml-core -- -D warnings` clean

---

## Execution Strategy

**Parallel groups** (no dependencies between groups):
- **Group A**: Phase 1 (algorithm bugs) — 4 independent fixes
- **Group B**: Phase 2 (AutoML) — coupled changes
- **Group C**: Phase 3 (stats) + Phase 5 (robust) — both in stats/models
- **Group D**: Phase 4 (HPO) — self-contained
- **Group E**: Phase 6 (traits) — self-contained
- **Group F**: Phase 7 + 8 (edge cases, cleanup) — do last

**Within each group**, use parallel agents where files don't overlap.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| F-distribution CDF needs incomplete beta function | Check if `stats/` already has one; if not, use Lentz's continued fraction |
| Tree MAE change may shift existing test expectations | Update test assertions to match correct behavior |
| KNNImputer distance change affects neighbor selection | Existing tests may need threshold updates |
| WarmStartModel changes could break ensemble behavior | Warm start is opt-in (default false), no behavioral change |
| Removing `_maximize`/`_search_space` changes signatures | Internal only, not public API |
| HuberProposal2 iteration may not converge | Cap at 20 iterations, fall back to MAD |
| AdaBoost SAMME fix changes multiclass weights | Existing binary tests unaffected; add multiclass test |
| LCB acquisition fix changes Bayesian optimization behavior | Add test with known optimum to verify |
