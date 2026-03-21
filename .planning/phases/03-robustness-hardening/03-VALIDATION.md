---
phase: 3
slug: robustness-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-21
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust: built-in #[test] + cargo test; Python: pytest |
| **Config file** | .pre-commit-config.yaml (clippy, fmt, test hooks) |
| **Quick run command** | `cargo test -p ferroml-core --lib -- --test-threads=4` |
| **Full suite command** | `cargo test --all && pytest ferroml-python/tests/` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p ferroml-core --lib -- --test-threads=4`
- **After every plan wave:** Run `cargo test --all && pytest ferroml-python/tests/`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | ROBU-01 | unit | `cargo test -p ferroml-core --lib models::svm` | ✅ | ⬜ pending |
| 03-01-02 | 01 | 1 | ROBU-02 | unit | `cargo test -p ferroml-core --lib stats` | ✅ | ⬜ pending |
| 03-01-03 | 01 | 1 | ROBU-03 | unit | `cargo test -p ferroml-core --lib models::boosting models::hist_boosting` | ✅ | ⬜ pending |
| 03-02-01 | 02 | 1 | ROBU-04 | unit | `cargo test -p ferroml-core --lib models::linear models::regularized models::logistic` | ✅ | ⬜ pending |
| 03-02-02 | 02 | 1 | ROBU-05 | unit | `cargo test -p ferroml-core --lib models::tree models::forest` | ✅ | ⬜ pending |
| 03-02-03 | 02 | 1 | ROBU-06 | unit | `cargo test -p ferroml-core --lib preprocessing` | ✅ | ⬜ pending |
| 03-02-04 | 02 | 1 | ROBU-07 | unit | `cargo test -p ferroml-core --lib clustering` | ✅ | ⬜ pending |
| 03-02-05 | 02 | 1 | ROBU-08, ROBU-09 | lint | `cargo clippy -p ferroml-core -- -W clippy::unwrap_used` | ✅ | ⬜ pending |
| 03-03-01 | 03 | 2 | ROBU-10 | audit | `grep -rn "FerroError" ferroml-core/src/error.rs` | ✅ | ⬜ pending |
| 03-03-02 | 03 | 2 | ROBU-11 | integration | `pytest ferroml-python/tests/test_bindings_correctness.py -x` | Partial | ⬜ pending |
| 03-03-03 | 03 | 2 | ROBU-12, ROBU-13 | integration | `pytest ferroml-python/tests/test_bindings_correctness.py -k pickle -x` | Partial | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Expand pickle roundtrip tests from 4 models to all 55+ in `test_bindings_correctness.py`
- [ ] Add Python exception type assertion tests (ShapeMismatch -> ValueError, etc.)

*Existing test infrastructure covers all unwrap-triage requirements (ROBU-01 through ROBU-09).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Error messages are actionable | ROBU-10 | Subjective quality assessment | Review each FerroError variant message for "what happened" + "what to do" |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
