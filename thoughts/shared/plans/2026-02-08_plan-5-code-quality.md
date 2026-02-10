# Plan 5: Code Quality Cleanup

**Date:** 2026-02-08
**Reference:** `thoughts/shared/research/2026-02-08_ferroml-comprehensive-assessment.md`
**Priority:** Medium
**Estimated Tasks:** 8

## Objective

Remove dead code, reduce clippy suppressions, and improve overall code quality.

## Context

From research:
- 18 dead code items with #[allow(dead_code)]
- 39 clippy suppressions
- Unused GPU feature flag
- Unused AutoML fields (portfolio, study)

## Tasks

### Task 5.1: Audit dead code in lib.rs
**File:** `ferroml-core/src/lib.rs`
**Description:**
- `portfolio: Option<automl::AlgorithmPortfolio>` - Decide: implement or remove
- `study: Option<hpo::Study>` - Decide: implement or remove
**Decision needed:** Are these planned features or abandoned?

### Task 5.2: Remove dead code in models/tree.rs
**File:** `ferroml-core/src/models/tree.rs`
**Lines:** 106, 138, 609, 909
**Description:** Remove or implement 4 dead code items.

### Task 5.3: Remove dead code in models/linear.rs
**File:** `ferroml-core/src/models/linear.rs`
**Lines:** 383, 1020, 1137, 1143, 1173
**Description:** Remove or implement 5 dead code items.

### Task 5.4: Clean up automl dead code
**Files:** `automl/transfer.rs:706`, `automl/ensemble.rs:324`
**Description:** Remove dead code paths in AutoML modules.

### Task 5.5: Clean up explainability dead code
**File:** `ferroml-core/src/explainability/treeshap.rs`
**Lines:** 271, 342, 345
**Description:** Remove 3 dead code items from TreeSHAP.

### Task 5.6: Address clippy suppressions
**Files:** Various
**Description:** Review 39 clippy suppressions, fix where possible:
- `many_single_char_names` (8) - Consider renaming variables
- `cast_precision_loss` (8) - Add explicit handling or document
- `too_many_arguments` (4) - Consider builder pattern or config struct
- `type_complexity` (1) - Consider type alias
**Target:** Reduce by 50% (20 suppressions)

### Task 5.7: Remove unused GPU feature flag
**File:** `ferroml-core/Cargo.toml`
**Description:** Remove `gpu = []` feature flag until GPU support is implemented.

### Task 5.8: Final clippy and dead code audit
**Command:** `cargo clippy --all-targets --all-features -- -D warnings`
**Description:** Verify no new warnings, document remaining justified suppressions.

## Success Criteria

- [ ] Dead code reduced from 18 to <5 (justified remainder documented)
- [ ] Clippy suppressions reduced from 39 to <20
- [ ] GPU feature flag removed
- [ ] No new warnings introduced
- [ ] All remaining suppressions have inline comments explaining why

## Decision Points

### AutoML Fields (Task 5.1)
**Options:**
1. **Implement:** Wire up portfolio and study fields to AutoML fit()
2. **Remove:** Delete fields if not planned for near-term
3. **Document:** Keep with explicit "future feature" comment

**Recommendation:** Option 2 (Remove) - Clean code now, add back when implementing

### Clippy many_single_char_names (Task 5.6)
**Locations:** hpo/bayesian.rs (8 instances)
**Context:** Mathematical formulas where short names are conventional
**Recommendation:** Keep suppression with comment: "Mathematical notation"
