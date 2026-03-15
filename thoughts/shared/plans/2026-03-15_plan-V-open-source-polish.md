# Plan V: Open-Source Polish Pass

**Date:** 2026-03-15
**Status:** PENDING
**Goal:** Make ferroml presentable, installable, and trustworthy for open-source launch.

---

## Overview

ferroml has excellent bones: 183K lines of Rust, 5,000+ tests, 55+ models, sklearn-compatible
Python API, GPU shaders, ONNX export. But first impressions matter. This plan fixes the two
known bugs, updates all version references, adds community files, and does a final API surface
audit — so that `pip install ferroml` on day one doesn't embarrass us.

---

## Current State

- **Version**: v0.3.0 tagged and pushed, CI blocked by billing
- **Bugs**: RidgeCV predict→NaN, ONNX RF classifier roundtrip mismatch
- **Docs**: Excellent content but version references stuck at v0.1.0/v0.2.0
- **Community files**: Missing CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, issue/PR templates
- **README**: Good structure but outdated version refs and test counts
- **pyproject.toml**: Production-ready metadata, `Development Status :: 3 - Alpha`

## Desired End State

- 0 known bugs
- All version references say v0.3.0
- Community files in place (CONTRIBUTING, CoC, SECURITY, templates)
- README is accurate, compelling, and has working install instructions
- `Development Status :: 4 - Beta` in pyproject.toml
- Feature parity scorecard regenerated post-Plan U
- API surface audit complete — no panics, no misleading methods

---

## Phase V.1 — Fix Known Bugs

### V.1a — RidgeCV NaN Bug

**Symptom**: `RidgeCV().fit(X, y).predict(X)` returns all NaN.

**Root cause investigation needed**: The fit() stores `self.model = Some(model)` and predict()
delegates to `self.model.predict()`. The NaN likely comes from:
- Cross-validation fold scoring producing NaN scores (div by zero in MSE?)
- Best alpha selection picking a model that wasn't fitted
- The inner RidgeRegression producing NaN for extreme alpha values

**Files**: `ferroml-core/src/models/regularized.rs` (RidgeCV impl, ~line 1430-1614)

**Success Criteria**:
- [ ] `RidgeCV().fit(X, y).predict(X)` returns finite values
- [ ] `RidgeCV().fit(X, y).score(X, y)` returns R² > 0.5 on linear data
- [ ] Existing test_regularized_cv.py still passes
- [ ] New test with the exact failing data added

### V.1b — ONNX RandomForestClassifier Roundtrip

**Symptom**: ONNX export → onnxruntime inference produces different labels than native predict.

**Root cause investigation needed**: The test uses `assert_classifier_label_roundtrip` which
compares native FerroML predictions against onnxruntime on the same data. Likely:
- Tree node threshold precision loss (f64 → f32)
- Leaf value aggregation differs (voting vs probability)
- Class label mapping issue

**Files**:
- `ferroml-core/src/onnx/` (tree export logic)
- `ferroml-python/tests/test_onnx_roundtrip.py` (test ~line 258)

**Success Criteria**:
- [ ] `pytest test_onnx_roundtrip.py::TestTreeClassifierRoundtrip::test_random_forest_classifier` passes
- [ ] All other ONNX roundtrip tests still pass

---

## Phase V.2 — Version & Docs Sync

### V.2a — Update All Version References

| File | Current | Target |
|------|---------|--------|
| README.md (line 27) | `ferroml-core = "0.2"` | `ferroml-core = "0.3"` |
| README.md (line 200-201) | `0.2` references | `0.3` |
| README.md status section | v0.2.0, ~4,700 tests | v0.3.0, ~5,083 tests |
| docs/ROADMAP.md | v0.1.0 current | v0.3.0 current, Plans A-U complete |
| docs/tutorials/quickstart.md | v0.1.0 feature flags | v0.3.0 |
| docs/feature-parity-scorecard.md | Pre-Plan U | Regenerate post-Plan U |

### V.2b — Regenerate Feature Parity Scorecard

Run `scripts/feature_parity_scorecard.py` to update the scorecard reflecting Plan U additions
(score, partial_fit, decision_function). This is the most visible proof of sklearn parity.

### V.2c — Update README

- Update test counts (3,160 Rust + 1,923 Python = 5,083 total)
- Add v0.3.0 highlights (score, partial_fit, decision_function)
- Ensure install instructions work: `pip install ferroml` (once billing fixed)
- Add badge for PyPI version, license, CI status
- Add a "What's New in v0.3.0" section or update the highlights

**Success Criteria**:
- [ ] `grep -r "0\.1\.\|0\.2\." README.md docs/ROADMAP.md docs/tutorials/quickstart.md` returns 0 hits
- [ ] Feature parity scorecard reflects Plan U completions
- [ ] README test counts match reality

---

## Phase V.3 — Community Files

### V.3a — CONTRIBUTING.md

Standard open-source contributing guide:
- How to report bugs (use issue templates)
- Development setup (Rust toolchain, Python venv, maturin, pre-commit)
- Testing requirements (cargo test, pytest, cargo fmt, clippy)
- Commit message convention (feat/fix/docs prefix)
- PR process
- Where to ask questions

### V.3b — CODE_OF_CONDUCT.md

Adopt Contributor Covenant v2.1 (industry standard).

### V.3c — SECURITY.md

Security vulnerability reporting process. Email contact for responsible disclosure.

### V.3d — Issue & PR Templates

**.github/ISSUE_TEMPLATE/bug_report.yml** (YAML form):
- FerroML version, Python version, OS
- Steps to reproduce
- Expected vs actual behavior
- Minimal reproducing code

**.github/ISSUE_TEMPLATE/feature_request.yml**:
- What sklearn/other library does this exist in?
- Use case description
- Proposed API

**.github/PULL_REQUEST_TEMPLATE.md**:
- Description of changes
- Checklist: tests added, docs updated, cargo fmt, clippy clean

**Success Criteria**:
- [ ] All 6 files exist and are well-formatted
- [ ] Issue templates render correctly on GitHub (test via `gh issue create --web`)

---

## Phase V.4 — API Surface Audit

### V.4a — Python Binding Safety Audit

Verify no panicking paths exist in Python-facing code:
- `grep -r "unwrap()" ferroml-python/src/*.rs` — audit each for safety
- `grep -r "panic!\|todo!\|unimplemented!" ferroml-python/src/*.rs` — must be 0
- `grep -r "expect(" ferroml-python/src/*.rs` — audit each

Any unsafe unwrap must be converted to `.map_err(|e| PyErr::new::<...>(...))`.

### V.4b — Python `__init__.py` Completeness

Verify all new methods (score, partial_fit, decision_function) are accessible through
the public Python API. Check each submodule's `__init__.py` re-exports.

### V.4c — Docstring Spot Check

Spot check 10 random Python-facing models:
- Do they have accurate parameter docs?
- Do they have a working example in the docstring?
- Do they list the correct return types?

### V.4d — pyproject.toml Final Polish

- Bump `Development Status :: 3 - Alpha` → `Development Status :: 4 - Beta`
- Verify all project URLs are correct and reachable
- Verify classifiers are comprehensive

**Success Criteria**:
- [ ] 0 unwrap/panic/todo in Python-facing code (or all audited as safe)
- [ ] `python -c "from ferroml.linear import LinearRegression; help(LinearRegression)"` shows clean docs
- [ ] All submodule imports work: `python -c "from ferroml import linear, trees, ensemble, svm, ..."`

---

## Phase V.5 — Final Validation

### V.5a — Full Test Suite

```bash
cargo test --lib -p ferroml-core
source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml
pytest ferroml-python/tests/ -v
```

All tests must pass (0 failures, 0 errors).

### V.5b — Clean Install Test

```bash
# In a fresh venv
python -m venv /tmp/ferroml-test
source /tmp/ferroml-test/bin/activate
pip install /path/to/ferroml-0.3.0-*.whl
python -c "
import ferroml
from ferroml.linear import LinearRegression
import numpy as np
X = np.random.randn(100, 3)
y = X @ [1, 2, 3] + np.random.randn(100) * 0.1
m = LinearRegression()
m.fit(X, y)
print(f'R² = {m.score(X, y):.4f}')
print('ferroml is working!')
"
```

### V.5c — Commit, Re-tag, Push

If bugs were fixed, bump to v0.3.1 (patch release):
```bash
# Update version to 0.3.1 in Cargo.toml + pyproject.toml
git commit -am "fix: RidgeCV NaN + ONNX RF roundtrip + open-source polish"
git tag v0.3.1
git push origin master v0.3.1
```

**Success Criteria**:
- [ ] 0 test failures
- [ ] Clean install works in fresh venv
- [ ] All community files present
- [ ] All version references synchronized
- [ ] GitHub Actions run successfully (once billing fixed)

---

## Phase V.6 — Close Plan T Gaps

Plan T.5 and T.6 were marked complete in commit but investigation shows they're incomplete.

### V.6a — warm_start Expansion (Plan T.5 Gap)

Currently ~7 models have `warm_start`. Plan T.5 targeted 12+. Add to:

| Model | File | Implementation |
|-------|------|---------------|
| KMeans | clustering/kmeans.rs | Reuse previous cluster_centers as initial centers |
| GaussianMixture | clustering/gmm.rs | Reuse previous means/covariances as initial params |
| MLPClassifier | neural/classifier.rs | Continue training from current weights |
| MLPRegressor | neural/mlp.rs | Continue training from current weights |
| SGDClassifier | models/sgd.rs | Already incremental — flag to skip re-init |
| SGDRegressor | models/sgd.rs | Same |

**Success Criteria**:
- [ ] 12+ models have `pub warm_start: bool`
- [ ] Each warm_start model has a test verifying re-fit reuses state

### V.6b — feature_importances_ for Linear Models (Plan T.6 Gap)

Add `feature_importances_` (absolute coefficient magnitudes, normalized) to:

| Model | File | Implementation |
|-------|------|---------------|
| LinearRegression | models/linear.rs | `abs(coef) / sum(abs(coef))` |
| RidgeRegression | models/regularized.rs | Same pattern |
| LassoRegression | models/regularized.rs | Same (sparse coefs make this especially useful) |
| ElasticNet | models/regularized.rs | Same |
| LogisticRegression | models/logistic.rs | Same |
| SGDClassifier | models/sgd.rs | Same |
| SGDRegressor | models/sgd.rs | Same |

Expose via Python `feature_importances_` property on each binding.

**Success Criteria**:
- [ ] 7+ linear models have `feature_importances_` returning normalized abs(coef)
- [ ] Python bindings expose `feature_importances_` property
- [ ] Test verifying Lasso feature_importances_ has zeros for dropped features

### V.6c — Regenerate Benchmarks

Re-run `scripts/benchmark_cross_library.py` to verify Plan T performance claims:
- HistGBT: claimed 1.9x vs sklearn (was 8-15x)
- KMeans: claimed 2x faster than sklearn (was 2x slower)
- LogReg: claimed 1.76x vs sklearn (was 2.5x slower)

Update `docs/cross-library-benchmark.md` with fresh numbers.

**Success Criteria**:
- [ ] Benchmark results file updated with fresh data
- [ ] Performance claims in CHANGELOG match reality

---

## Execution Order

| Phase | Priority | Effort | Dependencies |
|-------|----------|--------|-------------|
| V.1   | Critical | Medium | None — bugs must be fixed first |
| V.2   | High     | Small  | V.1 (need accurate test counts after fixes) |
| V.3   | High     | Small  | None — boilerplate community files |
| V.4   | High     | Small  | None — audit only, minimal code changes |
| V.6   | High     | Medium | None — independent feature work |
| V.5   | Critical | Small  | V.1-V.4, V.6 all complete |

**Recommended approach**: V.1 first (bug fixes), then V.2-V.4 + V.6 in parallel,
then V.5 last (final validation).

**Estimated total effort**: Medium — 2 bug fixes, 2 feature gaps, docs sync, community files.

---

## Risks & Mitigations

1. **Risk**: RidgeCV bug is deep in the math (not a simple fix)
   **Mitigation**: If complex, can mark RidgeCV as experimental/known limitation in docs

2. **Risk**: ONNX RF fix breaks other ONNX roundtrip tests
   **Mitigation**: Run full ONNX test suite after fix, use tolerance-based assertions

3. **Risk**: GitHub billing not fixed, blocking CI/CD
   **Mitigation**: User must resolve billing independently; code changes don't depend on it

4. **Risk**: Feature parity scorecard script is broken or outdated
   **Mitigation**: Run it early in V.2; if broken, fix or regenerate manually

5. **Risk**: warm_start for KMeans/GMM changes convergence behavior
   **Mitigation**: Test that warm_start=True with same data produces same result as cold start

6. **Risk**: Performance benchmark numbers differ from Plan T claims
   **Mitigation**: Update CHANGELOG/docs to match actual numbers; honesty > marketing
