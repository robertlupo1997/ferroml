---
phase: 05
slug: documentation-and-release
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-23
---

# Phase 05 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (Python) |
| **Config file** | ferroml-python/pytest.ini |
| **Quick run command** | `pytest ferroml-python/tests/test_docstrings.py -x` |
| **Full suite command** | `pytest ferroml-python/tests/ -x --timeout=300` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Rebuild with maturin, spot-check 3 models' `help()` output
- **After every plan wave:** Run `pytest ferroml-python/tests/ -x --timeout=300`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | DOCS-01, DOCS-02 | smoke | `pytest ferroml-python/tests/test_docstrings.py -x` | No -- Wave 0 | pending |
| 05-01-02 | 01 | 1 | DOCS-04 | smoke | `pytest ferroml-python/tests/test_docstrings.py -k notes -x` | No -- Wave 0 | pending |
| 05-02-01 | 02 | 1 | DOCS-03, DOCS-05 | manual-only | Visual inspection of README.md | N/A | pending |
| 05-02-02 | 02 | 1 | DOCS-06 | manual-only | `test -f docs/benchmarks.md` | Already exists | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `ferroml-python/tests/test_docstrings.py` — automated docstring completeness check (verify all models have Parameters, Examples, Attributes in `__doc__`)
- [ ] Verification script that imports every model and checks `__doc__` is not None and contains required sections

*Existing test infrastructure covers regression testing; Wave 0 adds docstring-specific checks.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README Known Limitations section | DOCS-03 | Content quality requires human judgment | Review README.md for accuracy, completeness, tone |
| ort status documentation | DOCS-05 | Content quality | Verify README mentions RC status and user expectations |
| Benchmark page polish | DOCS-06 | Already exists, needs review | Verify docs/benchmarks.md has methodology, results, analysis |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
