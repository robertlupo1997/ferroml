# Launch Readiness Plan

## Overview
Prepare FerroML for open-source launch: run full Python smoke test, fix remaining config issues, document the launch checklist.

## Current State
- 55+ models, 5,650+ tests, Plans A-Z complete
- Performance: every major model within 2x of sklearn or faster
- CI/CD: 12 GitHub Actions workflows (tests, fuzzing, mutation, benchmarks, docs, publishing)
- Docs: CHANGELOG, ROADMAP, README updated for Plans Y+Z
- Tags: cleaned (no version tags — fresh start for v1.0)
- Known issues: cliff.toml wrong GitHub URL, Python 3.13 untested but in classifiers

## Desired End State
- Zero test failures across Rust and Python
- All config files pointing to correct repo
- Clear pre-launch checklist for when user enables public access

---

## Phase 1: Python Smoke Test (~15 min)

**Overview**: Run full Python test suite to catch any binding regressions from Plans Y+Z.

**Commands**:
```bash
source .venv/bin/activate
maturin develop --release -m ferroml-python/Cargo.toml
pytest ferroml-python/tests/ -x -q --tb=short
```

**Success Criteria**:
- [ ] Automated: `pytest ferroml-python/tests/` — ~2,100 tests pass (6 pre-existing now fixed)
- [ ] Automated: `cargo test --doc -p ferroml-core` — all doctests pass (multioutput fixed)

---

## Phase 2: Config Fixes (~5 min)

### 2a. Fix cliff.toml GitHub URL
**File**: `cliff.toml` line 59
- **Change**: `ferroml/ferroml` → `robertlupo1997/ferroml`

### 2b. Remove Python 3.13 classifier (not tested in CI)
**File**: `ferroml-python/pyproject.toml` line 37
- **Remove**: `"Programming Language :: Python :: 3.13",`
- **Reason**: CI only tests 3.10, 3.11, 3.12. Don't claim 3.13 support without testing.

**Success Criteria**:
- [ ] Automated: `grep "robertlupo1997" cliff.toml` — URL correct
- [ ] Automated: `grep "3.13" ferroml-python/pyproject.toml` — no 3.13 classifier

---

## Phase 3: Commit, Push, Document Launch Checklist (~5 min)

### 3a. Commit and push all fixes

### 3b. Document launch checklist
When going public, user needs to:

1. **GitHub repo settings**:
   - Settings → General → Visibility → Public
   - Settings → Pages → Source → "GitHub Actions"

2. **GitHub secrets** (Settings → Secrets and variables → Actions):
   - `PYPI_API_TOKEN` — from pypi.org account
   - `TEST_PYPI_API_TOKEN` — from test.pypi.org account
   - `CRATES_IO_TOKEN` — from crates.io account

3. **First release**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
   This triggers: CI tests → PyPI publish → crates.io publish → GitHub Release with changelog + artifacts

4. **Verify**:
   - GitHub Pages: `https://robertlupo1997.github.io/ferroml/`
   - PyPI: `pip install ferroml`
   - crates.io: `cargo add ferroml-core`

---

## Dependencies
- Python venv with sklearn, ferroml installed
- Release build of ferroml-python (maturin develop --release)

## Risks & Mitigations
- **Risk**: Python tests fail on a binding that changed in Plan Z
  - **Mitigation**: Fix immediately — most likely a signature change or missing re-export
- **Risk**: PyPI name `ferroml` already taken
  - **Mitigation**: Check `pip install ferroml` — if taken, use `ferro-ml`
