---
phase: 1
slug: input-validation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-20
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test + cargo test + pytest |
| **Config file** | Cargo.toml (workspace), ferroml-core/Cargo.toml |
| **Quick run command** | `cargo test --test edge_cases` |
| **Full suite command** | `cargo test && pytest ferroml-python/tests/` |
| **Estimated runtime** | ~120 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --test edge_cases -x`
- **After every plan wave:** Run `cargo test && pytest ferroml-python/tests/`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | VALID-09 | unit | `cargo test --lib validation` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | VALID-01, VALID-02 | integration | `cargo test --test edge_cases nan_input` | ✅ partial | ⬜ pending |
| 1-01-03 | 01 | 1 | VALID-04 | integration | `cargo test --test edge_cases empty_input` | ✅ partial | ⬜ pending |
| 1-01-04 | 01 | 1 | VALID-05 | integration | `cargo test --test edge_cases single_sample` | ✅ partial | ⬜ pending |
| 1-01-05 | 01 | 1 | VALID-08 | unit | `cargo test --lib` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 2 | VALID-01, VALID-02 | integration | `cargo test --test edge_cases nan_input` | ✅ partial | ⬜ pending |
| 1-02-02 | 02 | 2 | VALID-03 | integration | `cargo test --test edge_cases predict_validation` | ✅ partial | ⬜ pending |
| 1-02-03 | 02 | 2 | VALID-06 | integration | `cargo test --test edge_cases shape_mismatch` | ✅ partial | ⬜ pending |
| 1-02-04 | 02 | 2 | VALID-07 | integration | `cargo test --test edge_cases not_fitted` | ✅ partial | ⬜ pending |
| 1-03-01 | 03 | 3 | VALID-10 | Python test | `pytest ferroml-python/tests/test_input_validation.py` | ❌ W0 | ⬜ pending |
| 1-03-02 | 03 | 3 | VALID-07 | Python test | `pytest ferroml-python/tests/test_errors.py` | ✅ partial | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `ferroml-core/src/validation.rs` — shared validation module for unsupervised/transformer variants (VALID-09)
- [ ] New test entries in `edge_cases.rs` for all missing models (VALID-01 through VALID-07)
- [ ] `ferroml-python/tests/test_input_validation.py` — Python-side validation tests (VALID-10)
- [ ] Hyperparameter validation unit tests in model source files (VALID-08)

*Existing infrastructure covers supervised model validation; gaps are in clustering, decomposition, and Python binding layer.*

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
