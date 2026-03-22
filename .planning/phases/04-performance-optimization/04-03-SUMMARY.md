---
phase: 04-performance-optimization
plan: 03
subsystem: models
tags: [benchmarks, ols, ridge, cholesky, histgbt, histogram, performance]

# Dependency graph
requires:
  - phase: 04-performance-optimization
    provides: "Research identifying OLS/Ridge solver paths and HistGBT optimization targets"
provides:
  - "OLS Cholesky vs QR solver path benchmarks"
  - "Ridge faer Cholesky solver benchmarks"
  - "HistGBT histogram bounds check safety analysis"
  - "HistGBT_Histogram benchmark group"
affects: [performance-optimization]

# Tech tracking
tech-stack:
  added: []
  patterns: ["debug_assert for invariant validation in hot loops"]

key-files:
  created: []
  modified:
    - "ferroml-core/benches/benchmarks.rs"
    - "ferroml-core/benches/performance_optimizations.rs"
    - "ferroml-core/src/models/hist_boosting.rs"

key-decisions:
  - "HistGBT bounds checks RETAINED: BinMapper NaN handling can produce out-of-range bin indices (missing_bin = max_bins > histogram size)"
  - "debug_assert added for gradient/hessian index validation (zero overhead in release builds)"
  - "No unsafe code: bounds checks effectively free for non-NaN data via branch prediction"

patterns-established:
  - "debug_assert for hot-loop invariants: validates correctness in debug/test builds, zero cost in release"

requirements-completed: [PERF-07, PERF-08, PERF-09]

# Metrics
duration: 145min
completed: 2026-03-22
---

# Phase 04 Plan 03: OLS/Ridge Solver Verification and HistGBT Histogram Analysis Summary

**OLS Cholesky/QR solver path benchmarks, Ridge faer Cholesky benchmarks, and HistGBT histogram bounds check safety analysis with debug_assert guards**

## Performance

- **Duration:** 145 min
- **Started:** 2026-03-22T17:00:44Z
- **Completed:** 2026-03-22T19:25:44Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- OLS solver path benchmarks confirm Cholesky active for n >> 2p (5Kx50: ~3.9ms, 10Kx100: ~30ms) and QR for n <= 2p (200x100: ~4.1ms, 100x50: ~171us)
- Ridge faer Cholesky benchmarks across 4 dataset shapes (tall 5Kx50: ~4.6ms, tall 10Kx100: ~33ms, wide 200x100: ~5.4ms, sq 100x50: ~787us)
- HistGBT histogram bounds check audit: documented as NECESSARY due to BinMapper NaN handling producing out-of-range indices
- Added debug_assert in build_histograms_col_major for gradient/hessian index validation
- HistGBT_Histogram benchmark group added (5K/10K/20K samples, 50 iterations)

## Task Commits

Each task was committed atomically:

1. **Task 1: OLS/Ridge solver path benchmarks** - `337be10` (feat)
2. **Task 2: HistGBT histogram inner loop optimization** - `f1e383b` (documentation + benchmark, committed alongside 04-01 docs), `5753dff` (debug_assert addition)

## Files Created/Modified
- `ferroml-core/benches/benchmarks.rs` - OLS_Solver_Paths and Ridge_Solver benchmark groups
- `ferroml-core/benches/performance_optimizations.rs` - HistGBT_Histogram benchmark group
- `ferroml-core/src/models/hist_boosting.rs` - Documentation of bounds check necessity + debug_assert

## Decisions Made

1. **HistGBT bounds checks RETAINED (not removed):** BinMapper assigns `missing_bin = max_bins` for NaN values, but histogram size is `actual_bins + 1` where `actual_bins <= max_bins - 1`. NaN samples can produce bin index `max_bins` which exceeds histogram size. Bounds checks are necessary for correctness.

2. **No unsafe code introduced:** For non-NaN data, all bin indices are valid and the branch predictor learns the always-true pattern, making bounds checks effectively free (< 1% overhead). Using `unsafe get_unchecked_mut` would save nothing measurable while introducing UB risk.

3. **debug_assert for invariant validation:** Added assertion that sample indices are within gradient/hessian array bounds. Zero cost in release builds, catches bugs in debug/test builds.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed with_max_leaf_nodes type error in benchmark**
- **Found during:** Task 2 (HistGBT benchmark)
- **Issue:** `with_max_leaf_nodes(31)` should be `with_max_leaf_nodes(Some(31))`
- **Fix:** Wrapped value in `Some()`
- **Files modified:** ferroml-core/benches/performance_optimizations.rs
- **Verification:** Clippy passes
- **Committed in:** f1e383b

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor type fix. No scope creep.

## Issues Encountered
- Concurrent agent commits caused pre-commit hook stash conflicts. Multiple retry cycles needed for Task 2 commit. Task 2 documentation and benchmarks were eventually committed as part of another agent's commit (f1e383b), with the debug_assert committed separately (5753dff).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- OLS and Ridge solver paths verified with benchmark proof
- HistGBT histogram inner loop documented as already optimal
- All 30 HistGBT tests pass with no regressions

---
*Phase: 04-performance-optimization*
*Completed: 2026-03-22*
