# FerroML Remaining Fixes Implementation Plan

**Date:** 2026-02-06
**Predecessor:** quality-perfection-plan (17/17 phases complete, commit `553f921`)
**Scope:** 19 remaining items (1 dropped from original 20, 1 more dropped after research)

## Overview

After the 17-phase quality perfection plan, 20 documented-but-not-applied items remain. Deep codebase research by 5 parallel agents confirmed 18 are still needed, 1 is already correct (LDA between-class scatter), and 1 proposed fix (Hochberg multiplier) needs verification — the agent found the current code's sorting order may make `(rank+1)` correct for the iteration direction used. The full streaming serialization is dropped as over-engineering (replaced with CRC32 only).

## Current State

- **Tests:** 2262 passed, 0 failed, 6 ignored
- **Clippy:** Clean (`-D warnings`)
- **Commit:** `553f921`

## Desired End State

- All critical/high correctness bugs fixed
- TPE sampler implements true l(x)/g(x) algorithm
- TreeSHAP implements Lundberg 2018 Algorithm 2
- ONNX export and inference produce correct results
- Serialization has integrity checking and structured versioning
- Tests increase by ~200+ (regression tests for each fix)

## Implementation Phases

### Phase 1: Quick Correctness Fixes (10 items, ~1 session)

All low-complexity, high-impact changes.

#### 1.1: Fisher z-transform guard
**File:** `ferroml-core/src/stats/mod.rs:304-317`
**Change:** Guard r=±1 before `((1+r)/(1-r)).ln()` which produces NaN/Inf.
```rust
// Before the z-transform computation, add:
let (r_lower, r_upper) = if r.abs() >= 1.0 - 1e-10 {
    (r.clamp(-1.0, 1.0), r.clamp(-1.0, 1.0))
} else {
    // existing z-transform computation
};
```

#### 1.2: Hochberg multiplier (VERIFY FIRST)
**File:** `ferroml-core/src/stats/multiple_testing.rs:107`
**Action:** The agent found the iteration order is descending (line 100: `b.partial_cmp(&a)`), which means rank=0 is the largest p-value. Verify against the Hochberg step-up definition before changing. If `(rank+1)` is correct for descending iteration, skip this fix.

#### 1.3: Log-scale parameter validation
**File:** `ferroml-core/src/hpo/search_space.rs:122-146`
**Change:** Add bounds validation in `float_log()` and `int_log()`.
```rust
pub fn float_log(low: f64, high: f64) -> Self {
    assert!(low > 0.0 && high > 0.0, "log-scale bounds must be > 0");
    // ... existing code
}
```
Note: Use `assert!` since this is a parameter constructor (fail-fast), not a runtime computation.

#### 1.4: Box-Muller clamp in samplers.rs
**File:** `ferroml-core/src/hpo/samplers.rs:333`
**Change:** `rng.random()` → `rng.random::<f64>().max(1e-10)` (schedulers.rs already fixed).

#### 1.5: Correlation NaN guard
**File:** `ferroml-core/src/hpo/mod.rs:361`
**Change:** Guard zero-variance denominator.
```rust
let denom = sum_x2.sqrt() * sum_y2.sqrt();
if denom == 0.0 { return 0.0; }
sum_xy / denom
```

#### 1.6: RF ONNX aggregate SUM→AVERAGE
**File:** `ferroml-core/src/onnx/tree.rs:185`
**Change:** `"SUM"` → `"AVERAGE"` in the aggregate_function attribute string.

#### 1.7: Reshape -1 dimension
**File:** `ferroml-core/src/inference/operators.rs:203-222`
**Change:** Parse shape values as i64, resolve -1 by inferring dimension from total element count.
```rust
let total_elements = input.data.len();
let mut inferred_idx = None;
let mut known_product = 1usize;
for (i, &dim) in shape_i64.iter().enumerate() {
    if dim == -1 {
        inferred_idx = Some(i);
    } else {
        known_product *= dim as usize;
    }
}
if let Some(idx) = inferred_idx {
    new_shape[idx] = total_elements / known_product;
}
```

#### 1.8: Unknown data type → error + DOUBLE/INT32 support
**File:** `ferroml-core/src/inference/session.rs:237-278`
**Change:** Add `int32_data`/`double_data` named field handling, add raw data type 6 (INT32) and 11 (DOUBLE), change default fallback from empty tensor to `Err(FerroError::invalid_input(...))`.

#### 1.9: Bootstrap CI percentile index
**File:** `ferroml-core/src/stats/bootstrap.rs:106-107, 185-186`
**Change:** Add `.round()` before `as usize` cast in both locations.

#### 1.10: PR-AUC precision interpolation
**File:** `ferroml-core/src/metrics/probabilistic.rs:244`
**Change:** After the for loop, before `Ok(...)`, insert backward pass:
```rust
for i in (0..precision.len() - 1).rev() {
    precision[i] = precision[i].max(precision[i + 1]);
}
```

**Success Criteria Phase 1:**
- [ ] `cargo test -p ferroml-core --lib` passes (2262+ tests)
- [ ] `cargo clippy -p ferroml-core -- -D warnings` clean
- [ ] Each fix has at least 1 regression test

---

### Phase 2: Lentz's Incomplete Beta (1 item, ~1 session)

#### 2.1: Replace naive continued fraction in 4 files
**Files:**
- `ferroml-core/src/stats/mod.rs:370-396` (primary)
- `ferroml-core/src/stats/confidence.rs:218-242` (duplicate)
- `ferroml-core/src/stats/hypothesis.rs:364-388` (duplicate)
- `ferroml-core/src/metrics/comparison.rs:418-443` (duplicate)

**Key insight:** A correct Lentz implementation already exists in `models/linear.rs:1030-1112`. Use it as the template.

**Algorithm:**
1. Symmetry check: if x < (a+1)/(a+b+2), use CF directly; else use I_x(a,b) = 1 - I_{1-x}(b,a)
2. Log-space prefix: `exp(a*ln(x) + b*ln(1-x) + lngamma(a+b) - lngamma(a) - lngamma(b))`
3. Lentz's method with even/odd steps, fpmin=1e-30, eps=3e-12, max_iter=200

**Tests:** Known values: I_0.5(1,1)=0.5, I_0.5(2,2)=0.5, I_0.3(2,5)≈0.5282, edge cases 0 and 1.

**Success Criteria Phase 2:**
- [ ] All 4 copies replaced with Lentz algorithm
- [ ] Test with known incomplete beta values passes
- [ ] Existing t_cdf tests still pass

---

### Phase 3: LDA Eigenvalue Solver (1 item, ~1 session)

#### 3.1: Use symmetric transformation for eigenvalue problem
**File:** `ferroml-core/src/decomposition/lda.rs:673-716`

**Problem:** `symmetric_eigen()` called on non-symmetric `S_w^{-1} * S_b`.

**Fix:** Two approaches tried in order:
1. **Cholesky path:** S_w = L*L^T → M = L^{-1} * S_b * L^{-T} (symmetric) → eigendecompose M → transform back w = L^{-T} * v
2. **SVD fallback:** If S_w not positive definite → S_w^{-1/2} via SVD → M = S_w^{-1/2} * S_b * S_w^{-1/2} (symmetric) → eigendecompose → transform back

**Tests:** 3-class dataset comparing Eigen solver vs SVD solver predictions.

**Success Criteria Phase 3:**
- [ ] LDA Eigen solver produces correct predictions on 3+ class data
- [ ] Eigen and SVD solvers agree on predictions
- [ ] Existing LDA tests pass

---

### Phase 4: Serialization Improvements (2 items, ~0.5 session)

#### 4.1: SemanticVersion type
**File:** `ferroml-core/src/serialization.rs` (add before line 90)

New struct with major/minor/patch, Display, FromStr, Ord, Serialize/Deserialize (as string for backward compat). Replace `ferroml_version: String` with `ferroml_version: SemanticVersion`. Add `is_compatible_with()` method.

**Wire format:** Unchanged (serializes as "0.1.0" string). Backward compatible.

#### 4.2: CRC32 checksum for bincode format
**File:** `ferroml-core/src/serialization.rs` (all bincode read/write paths)
**Dependency:** Add `crc32fast = "1.5"` to Cargo.toml (already in lockfile as transitive dep).

Format: `[MAGIC_BYTES:4][SERIALIZED_DATA:N][CRC32:4 LE]`

CRC32 computed over SERIALIZED_DATA only. On load, verify before deserializing. Error on mismatch.

**Breaking change:** Old bincode files without CRC will fail to load. Acceptable for 0.1.0.

**Tests:** Round-trip, corruption detection (flip byte in middle), truncation detection.

**Success Criteria Phase 4:**
- [ ] SemanticVersion parses, compares, serializes correctly
- [ ] CRC32 detects corruption in bincode files
- [ ] Existing serialization tests pass (with updated type checks)

---

### Phase 5: TPE Sampler Rewrite (~1 session)

**File:** `ferroml-core/src/hpo/samplers.rs:197-337`

**Current state:** Single Gaussian around good values. Bad trials discarded entirely. Categorical falls back to random. No candidate scoring.

**Key insight:** KDE infrastructure already exists in `schedulers.rs:968-1082` (BOHB sampler). Extract and reuse.

**Design:**
1. Create `OneDimensionalKDE` (60 lines) — log_pdf, sample, bandwidth via Scott's rule
2. Add `CategoricalDistribution` helper (30 lines) — frequency-based sampling + log_pdf
3. Rewrite `sample()` method (90 lines):
   - Split trials into good/bad by gamma percentile
   - Build per-dimension l(x) KDE from good, g(x) KDE from bad
   - Generate n_ei_candidates (default 24) from l(x)
   - Score each by sum of log(l) - log(g) across dimensions
   - Return highest-scoring candidate
4. Handle log-scale parameters (transform to log space before KDE)
5. Handle categorical parameters (frequency distribution, not KDE)

**Estimated:** ~330 new lines, -90 removed = +240 net.

**No interface changes.** `Sampler` trait unchanged. All existing tests should pass.

**Success Criteria Phase 5:**
- [ ] TPE uses l(x)/g(x) density ratio
- [ ] Bad trials contribute to g(x) (not discarded)
- [ ] Categorical parameters use frequency, not random
- [ ] n_ei_candidates scored and best returned
- [ ] Existing HPO tests pass
- [ ] New test: TPE outperforms random on simple 1D quadratic

---

### Phase 6: TreeSHAP Rewrite (~1-2 sessions)

**File:** `ferroml-core/src/explainability/treeshap.rs:665-808`

**Current state:** Single-path Saabas method with post-hoc normalization hack. Not true Shapley values.

**Design (Lundberg 2018, Algorithm 2):**
1. New `PathElement` struct (8 lines): feature_index, zero_fraction, one_fraction, pweight
2. `extend_path` (25 lines): Add feature to path, update combinatorial pweights
3. `unwind_path` (25 lines): Compute contribution sum by removing a feature from path
4. `tree_shap_recursive` (55 lines): Recurse into BOTH children (not just taken branch), compute contributions at leaves via unwind
5. New `tree_shap_values` (15 lines): Setup + call recursive. **No normalization.**

**Estimated:** ~138 new lines, -97 removed = +41 net.

**Key risk:** pweight index arithmetic is subtle. Mitigate with hand-computed test on depth-2 tree.

**Preserved infrastructure:** SHAPResult, SHAPBatchResult, InternalTree, TreeExplainer constructors, explain/explain_batch/explain_batch_parallel, all ~570 lines unchanged.

**Success Criteria Phase 6:**
- [ ] SHAP values naturally sum to prediction - base_value (no normalization)
- [ ] Hand-computed test on 3-node tree matches exact Shapley values
- [ ] Duplicate feature splits handled correctly
- [ ] Existing tests pass with tightened tolerances (1.0 → 1e-6)
- [ ] `explain_batch_parallel` still works with rayon

---

### Deferred / Skipped Items

| Item | Decision | Rationale |
|------|----------|-----------|
| Full streaming serialization | **SKIP** | Over-engineering. Models must fit in RAM for inference, so they fit for serialization. BufWriter already provides I/O buffering. |
| SqueezeOp axis type safety | **DEFER** | Partial issue, low impact. Only matters for malformed ONNX models. |
| LoadOptions/SaveOptions API | **DEFER** | Nice-to-have, not a correctness issue. |
| Progress callbacks for serialization | **DEFER** | Small value-add, can be done anytime. ~40 lines. |

## Dependencies

- Phase 1 has no dependencies (all independent fixes)
- Phase 2 has no dependencies
- Phase 3 has no dependencies
- Phase 4 requires adding `crc32fast` dependency
- Phase 5 has no dependencies (can run parallel with Phase 6)
- Phase 6 has no dependencies (can run parallel with Phase 5)

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hochberg fix may be wrong | Would break multiple testing | Verify against reference implementation before applying |
| CRC32 breaks old bincode files | Can't load pre-CRC files | Acceptable for 0.1.0; could add fallback if needed |
| LDA Cholesky may fail on edge cases | Wrong eigendecomposition | SVD fallback path handles non-PD S_w |
| TPE behavioral change | HPO results differ | Correct — old results were suboptimal |
| TreeSHAP pweight arithmetic | Incorrect SHAP values | Hand-computed test on minimal tree |
| Lentz's algorithm 4 copies to update | Inconsistency | Could consolidate to single shared function (future refactor) |

## Estimated Total Effort

| Phase | Items | Estimated |
|-------|-------|-----------|
| Phase 1: Quick fixes | 10 | ~1 session |
| Phase 2: Incomplete beta | 1 (4 files) | ~1 session |
| Phase 3: LDA eigen | 1 | ~1 session |
| Phase 4: Serialization | 2 | ~0.5 session |
| Phase 5: TPE sampler | 1 | ~1 session |
| Phase 6: TreeSHAP | 1 | ~1-2 sessions |
| **Total** | **16 items** | **~5.5-6.5 sessions** |
