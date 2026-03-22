---
phase: 04-performance-optimization
plan: 01
subsystem: testing
tags: [linalg, svd, cholesky, qr, eigendecomposition, pca, criterion, benchmarks, faer]

# Dependency graph
requires:
  - phase: 03-robustness-hardening
    provides: "Unwrap elimination and error handling safety net"
provides:
  - "15 stability tests for ill-conditioned matrices covering all linalg primitives"
  - "Criterion benchmarks for PCA, TruncatedSVD, LDA, FactorAnalysis at production sizes"
  - "Baseline timing: PCA 10Kx100 in 10.7ms with faer SVD backend"
affects: [04-02, 04-03, 04-04]

# Tech tracking
tech-stack:
  added: []
  patterns: ["LCG-based deterministic PRNG for reproducible ill-conditioned matrix generation"]

key-files:
  created: []
  modified:
    - "ferroml-core/tests/edge_cases.rs"
    - "ferroml-core/benches/benchmarks.rs"

key-decisions:
  - "Used LCG PRNG instead of rand crate for test matrix generation (avoids additional dependency in test code)"
  - "Cholesky tests use regularization parameter rather than expecting raw success on near-singular SPD"

patterns-established:
  - "stability_tests module in edge_cases.rs for numerical conditioning tests"
  - "SVD perf benchmark group pattern with sample_size(10) for large data"

requirements-completed: [PERF-01, PERF-02, PERF-03, PERF-04, PERF-12]

# Metrics
duration: 67min
completed: 2026-03-22
---

# Phase 04 Plan 01: Stability Tests and Decomposition Benchmarks Summary

**15 ill-conditioned matrix stability tests plus Criterion benchmarks proving faer SVD completes PCA on 10Kx100 in 10.7ms**

## Performance

- **Duration:** 67 min
- **Started:** 2026-03-22T17:00:37Z
- **Completed:** 2026-03-22T18:07:43Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- 15 stability tests covering SVD (4), Cholesky (3), QR (2), eigendecomposition (2), and decomposition models (4) on ill-conditioned inputs
- Criterion benchmarks for PCA, TruncatedSVD, LDA, FactorAnalysis at production-relevant sizes
- PCA on 10000x100 completes in 10.7ms with faer backend confirmed active
- Zero test regressions (pre-commit hooks verified on both commits)

## Task Commits

Each task was committed atomically:

1. **Task 1: Ill-conditioned matrix stability test suite** - `89d7396` (test)
2. **Task 2: Decomposition Criterion benchmarks and faer verification** - `8f593e9` (feat)

## Files Created/Modified
- `ferroml-core/tests/edge_cases.rs` - Added stability_tests module with 15 tests for ill-conditioned matrices
- `ferroml-core/benches/benchmarks.rs` - Added 4 benchmark functions (PCA_SVD_Performance, TruncatedSVD_Performance, LDA_Performance, FactorAnalysis_Performance)

## Benchmark Results

| Model | Size | Time |
|-------|------|------|
| PCA | 1000x50 | 1.7ms |
| PCA | 5000x100 | 6.3ms |
| PCA | 10000x100 | 10.7ms |
| TruncatedSVD | 1000x50 | 5.1ms |
| TruncatedSVD | 5000x100 | 29.4ms |
| TruncatedSVD | 10000x100 | 53.6ms |
| LDA | 1000x50 | 3.1ms |
| LDA | 5000x100 | 36.0ms |
| FactorAnalysis | 1000x50 | 26.0ms |
| FactorAnalysis | 5000x50 | 130.1ms |

## Decisions Made
- Used LCG-based deterministic PRNG for reproducible ill-conditioned matrix generation (avoids rand dependency in tests)
- Cholesky tests use regularization parameter rather than expecting raw success on near-singular SPD matrices

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed clippy let-and-return warning**
- **Found during:** Task 1
- **Issue:** Clippy rejected unnecessary let binding in Cholesky well-conditioned test
- **Fix:** Inlined the expression
- **Files modified:** ferroml-core/tests/edge_cases.rs
- **Committed in:** 89d7396

**2. [Rule 1 - Bug] Fixed TruncatedSVD::new() and FactorAnalysis API**
- **Found during:** Task 1
- **Issue:** TruncatedSVD::new() takes no args (use with_n_components), FactorAnalysis uses with_n_factors not with_n_components
- **Fix:** Used correct builder API calls
- **Files modified:** ferroml-core/tests/edge_cases.rs
- **Committed in:** 89d7396

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Minor API mismatch corrections. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Stability safety net is in place for all linalg primitives
- Baseline benchmark timings established for all decomposition models
- Ready for solver tuning and optimization in Plans 04-02 through 04-04

---
*Phase: 04-performance-optimization*
*Completed: 2026-03-22*
