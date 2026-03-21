---
phase: 02-correctness-fixes
plan: 02
subsystem: numerical-stability
tags: [logsumexp, cholesky, svd, lru-cache, svm, naive-bayes, gmm]

# Dependency graph
requires:
  - phase: 01-input-validation
    provides: validated inputs reduce numerical edge case surface
provides:
  - Shared logsumexp utility in linalg.rs used by GMM and all NaiveBayes variants
  - cholesky_with_jitter wrapper for automatic jitter retry on ill-conditioned matrices
  - svd_flip inside thin_svd for deterministic SVD signs across backends
  - SVM KernelCache unit tests (9 tests) covering LRU eviction, hit promotion, shrinking
  - Fix for KernelCache evict_lru len tracking bug
affects: [03-unwrap-audit, decomposition, gaussian-process, pca]

# Tech tracking
tech-stack:
  added: []
  patterns: [shared-numerical-utilities, svd-sign-normalization, jitter-retry]

key-files:
  created: []
  modified:
    - ferroml-core/src/linalg.rs
    - ferroml-core/src/models/svm.rs
    - ferroml-core/src/clustering/gmm.rs
    - ferroml-core/src/models/naive_bayes/gaussian.rs
    - ferroml-core/src/models/naive_bayes/multinomial.rs
    - ferroml-core/src/models/naive_bayes/bernoulli.rs
    - ferroml-core/src/models/naive_bayes/categorical.rs

key-decisions:
  - "logsumexp takes &[f64] not &Array1 for maximum flexibility across callers"
  - "svd_flip applied inside thin_svd (not as separate call) so all consumers get consistent signs automatically"
  - "cholesky_with_jitter does not add log crate dependency -- jitter is applied silently"
  - "KernelCache evict_lru bug fixed: len was decremented on eviction but slot immediately reused, causing slot collisions"

patterns-established:
  - "Shared numerical utilities in linalg.rs: logsumexp, logsumexp_rows, cholesky_with_jitter, svd_flip"
  - "SVD sign normalization applied inside thin_svd for all backends (nalgebra, faer)"

requirements-completed: [CORR-03, CORR-04, CORR-06, CORR-07, CORR-08]

# Metrics
duration: 42min
completed: 2026-03-21
---

# Phase 2 Plan 02: Numerical Stability Summary

**SVM kernel cache unit tests (9 tests + eviction bug fix), shared logsumexp/cholesky_with_jitter/svd_flip utilities in linalg.rs with all 6 consumers updated**

## Performance

- **Duration:** 42 min
- **Started:** 2026-03-21T15:43:48Z
- **Completed:** 2026-03-21T16:25:48Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Added 9 SVM KernelCache unit tests covering LRU eviction order, hit promotion, cache hits, shrinking invalidation, symmetric lookup, repeated access, empty/small cache, and full eviction cycles
- Discovered and fixed a real bug in KernelCache::evict_lru where len was decremented but the slot was immediately reused, causing slot collisions when the cache re-entered the free-slot allocation path
- Added shared logsumexp(), logsumexp_rows(), cholesky_with_jitter(), and svd_flip() to linalg.rs with 13 unit tests
- Updated GMM and all 4 NaiveBayes variants to use shared logsumexp instead of duplicated inline code
- Applied svd_flip inside thin_svd_nalgebra and thin_svd_faer so all SVD consumers get deterministic signs automatically
- All 3203 lib tests, 272 correctness tests, and 36 regression tests pass with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Add SVM KernelCache unit tests** - `b551c4f` (feat: 9 cache tests + eviction bug fix)
2. **Task 2: Add numerical stability utilities** - `9a41665` (feat: logsumexp, cholesky_with_jitter, svd_flip + consumer updates)

## Files Created/Modified
- `ferroml-core/src/linalg.rs` - Added logsumexp, logsumexp_rows, cholesky_with_jitter, svd_flip; applied svd_flip in thin_svd_nalgebra and thin_svd_faer; 13 new tests
- `ferroml-core/src/models/svm.rs` - Fixed evict_lru len tracking bug; added 9 KernelCache unit tests in cache_tests module
- `ferroml-core/src/clustering/gmm.rs` - Replaced local logsumexp with crate::linalg::logsumexp at 4 call sites
- `ferroml-core/src/models/naive_bayes/gaussian.rs` - Replaced inline log-sum-exp with shared logsumexp
- `ferroml-core/src/models/naive_bayes/multinomial.rs` - Replaced inline log-sum-exp with shared logsumexp
- `ferroml-core/src/models/naive_bayes/bernoulli.rs` - Replaced inline log-sum-exp with shared logsumexp
- `ferroml-core/src/models/naive_bayes/categorical.rs` - Replaced inline log-sum-exp with shared logsumexp

## Decisions Made
- logsumexp takes `&[f64]` rather than `&Array1<f64>` for maximum flexibility -- callers convert via `.to_vec()` or `.as_slice()`
- svd_flip is applied inside `thin_svd()` itself (not as a separate post-processing step) per sklearn convention, ensuring all PCA/LDA/FA/TruncatedSVD consumers get consistent signs automatically
- cholesky_with_jitter does not add the `log` crate as a dependency (the crate doesn't use logging currently); jitter is applied silently with a code comment documenting the intent
- KernelCache evict_lru bug was fixed by removing the `self.len -= 1` since the freed slot is always immediately reused by get_row

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed KernelCache evict_lru len tracking**
- **Found during:** Task 1 (KernelCache unit tests)
- **Issue:** `evict_lru()` decremented `self.len` but `get_row()` immediately reused the freed slot. After eviction, `len` was `capacity - 1`, causing the next `get_row` to allocate slot `self.len` (already occupied!) instead of evicting again, corrupting the cache.
- **Fix:** Removed `self.len -= 1` from `evict_lru()` since the slot is always immediately reused
- **Files modified:** `ferroml-core/src/models/svm.rs`
- **Verification:** All 9 cache tests pass; all 63 existing SVM tests pass
- **Committed in:** b551c4f

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix was discovered through the cache tests (exactly what the plan intended). No scope creep.

## Issues Encountered
None beyond the cache bug described above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All numerical stability utilities are in place for future consumers (GP models can adopt cholesky_with_jitter)
- SVD sign normalization ensures consistent PCA/LDA components across runs and backends
- Ready for Phase 2 Plan 03 (convergence reporting and output checks)

## Self-Check: PASSED

All 7 modified files exist. Both commit hashes (b551c4f, 9a41665) verified. All 4 key functions (logsumexp, cholesky_with_jitter, svd_flip, cache_tests module) confirmed present.

---
*Phase: 02-correctness-fixes*
*Completed: 2026-03-21*
