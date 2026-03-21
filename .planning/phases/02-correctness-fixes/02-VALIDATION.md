---
phase: 2
slug: correctness-fixes
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust: built-in #[test] + cargo test; Python: pytest |
| **Config file** | ferroml-python/pyproject.toml |
| **Quick run command** | `cargo test` |
| **Full suite command** | `cargo test --all && pytest ferroml-python/tests/` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test`
- **After every plan wave:** Run `cargo test --all && pytest ferroml-python/tests/`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | CORR-01 | integration | `pytest ferroml-python/tests/test_vs_sklearn_gaps_phase2.py::TestTemperatureScalingVsSklearn -x` | Exists (failing) | ⬜ pending |
| 02-01-02 | 01 | 1 | CORR-02 | integration | `pytest ferroml-python/tests/test_vs_sklearn_gaps_phase2.py::TestIncrementalPCAVsSklearn -x` | Exists (failing) | ⬜ pending |
| 02-02-01 | 02 | 1 | CORR-03 | unit | `cargo test svm::cache_tests::test_shrinking -p ferroml-core` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 1 | CORR-04 | unit | `cargo test svm::cache_tests -p ferroml-core` | ❌ W0 | ⬜ pending |
| 02-02-03 | 02 | 1 | CORR-06 | unit | `cargo test logsumexp -p ferroml-core` | Partial (GMM) | ⬜ pending |
| 02-02-04 | 02 | 1 | CORR-07 | unit | `cargo test cholesky_jitter -p ferroml-core` | ❌ W0 | ⬜ pending |
| 02-02-05 | 02 | 1 | CORR-08 | unit | `cargo test svd_flip -p ferroml-core` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | CORR-05 | unit | `cargo test validate_output -p ferroml-core` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 2 | CORR-10 | unit + integration | `cargo test convergence -p ferroml-core` | ❌ W0 | ⬜ pending |
| 02-03-03 | 03 | 2 | CORR-09 | integration | `pytest ferroml-python/tests/test_vs_sklearn*.py` | Partial | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] SVM `KernelCache` unit tests stubs (CORR-03, CORR-04) — inline in `svm.rs`
- [ ] `validate_output` utility function stubs (CORR-05) — in `models/mod.rs`
- [ ] `svd_flip` function stubs in `linalg.rs` (CORR-08)
- [ ] `cholesky_with_jitter` wrapper stubs (CORR-07) — in `linalg.rs`
- [ ] `ConvergenceStatus` enum stubs (CORR-10) — in `error.rs`
- [ ] Shared `logsumexp` utility stubs (CORR-06) — in `linalg.rs` or `stats/mod.rs`

*Existing infrastructure covers TemperatureScaling (CORR-01), IncrementalPCA (CORR-02), and cross-library tests (CORR-09).*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
