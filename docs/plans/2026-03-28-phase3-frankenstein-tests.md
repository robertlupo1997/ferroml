# Phase 3: Frankenstein Tests + Final Validation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Layer 5 Frankenstein test suite verifying that individually-correct FerroML components compose into globally-correct systems. Fix remaining medium-priority issues (RF non-determinism, MLP serialization). Update correctness report.

**Architecture:** Frankenstein tests are primarily Python-side (ferroml-python/tests/test_frankenstein.py) because the Python Pipeline wrapper accepts any ferroml model object — far more flexible than the Rust Pipeline which only supports the 2 models implementing PipelineModel. Rust-side composition tests go in ferroml-core/tests/correctness.rs (Layer 5 section). MLP serialization fix is Rust-side. RF determinism documented/tested.

**Tech Stack:** Python (pytest, numpy, ferroml), Rust (ndarray, serde, rayon), threading module for concurrency tests.

**Constraints:**
- Do NOT add new test binaries in ferroml-core/tests/ (consolidation rule)
- Frankenstein Rust tests append to correctness.rs
- Python tests go in a single new file: test_frankenstein.py
- Thread safety tests go in test_frankenstein.py (Python threading module)

---

## Session 7 Scope (this session)

This session covers Tasks 1-5 (Frankenstein tests + fixes). Session 8-9 handle final validation sweep and correctness report update.

---

### Task 1: Python Frankenstein Tests — Pipeline Composition

**Files:**
- Create: `ferroml-python/tests/test_frankenstein.py`

**Step 1: Write pipeline composition tests**

Create `test_frankenstein.py` with these test classes covering 7 categories:

1. **Pipeline composition correctness** (~6 tests):
   - Pipeline(StandardScaler, PCA, LogisticRegression) end-to-end vs manual step-by-step — exact match
   - Pipeline(MinMaxScaler, LinearRegression) — scaler state doesn't corrupt model
   - Pipeline(StandardScaler, Ridge) — regression pipeline R² > 0.9
   - Pipeline predict before fit raises ValueError
   - Pipeline(StandardScaler, DecisionTreeClassifier) — works with tree models
   - Pipeline(StandardScaler, SVC) — SVM with scaling

2. **Stateful interaction bugs** (~6 tests):
   - Fit/predict/refit/predict cycle — classifier state fully replaced
   - Fit/predict/refit/predict cycle — regressor state fully replaced
   - Pipeline refit replaces all state
   - deepcopy independence — clone doesn't affect original
   - RandomForest refit independence
   - DecisionTree refit — no stale nodes

3. **Ensemble composition** (~8 tests):
   - VotingClassifier hard voting — majority correct
   - VotingRegressor — manual average matches
   - StackingClassifier — meta-learner accuracy
   - StackingRegressor — R² check
   - BaggingClassifier — bootstrap aggregation
   - BaggingRegressor — regression bagging
   - VotingClassifier with Pipeline estimators — nested composition
   - Ensemble refit independence

4. **AutoML end-to-end** (~5 tests):
   - Classification end-to-end: raw data → AutoML → predictions
   - Regression end-to-end
   - Reproducibility: same seed → same results
   - Refit independence: two AutoML runs independent
   - Leaderboard sorted by rank

5. **Serialization under composition** (~5 tests):
   - Pickle Pipeline round-trip — same predictions
   - Pickle VotingClassifier round-trip
   - Pickle BaggingClassifier round-trip
   - Pickle StackingClassifier round-trip
   - Pickle RandomForest round-trip

   Note: pickle is used here for testing FerroML's own model serialization in
   a controlled test environment, not for loading untrusted external data.

6. **Thread safety** (~4 tests):
   - Concurrent predict() on LogisticRegression from 8 threads
   - Concurrent predict() on RandomForest from 4 threads
   - Concurrent predict() on fitted Pipeline from 4 threads
   - Concurrent predict() on GradientBoosting from 4 threads

7. **Performance composition** (~2 tests):
   - 100 repeated predict() calls — no degradation
   - Pipeline overhead < 10x vs manual steps

**Step 2: Run the tests to see which pass/fail**

Run: `cd /home/tlupo/ferroml && source .venv/bin/activate && pytest ferroml-python/tests/test_frankenstein.py -v --tb=short 2>&1 | tail -80`

Note: Some tests may fail due to API differences (e.g., AutoMLConfig missing `random_state` param, pickle not supported for certain models, etc.). Record failures for Task 2 fixes.

**Step 3: Fix any import/API issues in the test file**

Adjust test code based on actual ferroml Python API (constructor params, method names, etc.). The tests above are best-effort based on exploration — may need minor adjustments to match the actual binding API.

**Step 4: Run again and verify passing tests**

Run: `pytest ferroml-python/tests/test_frankenstein.py -v --tb=short 2>&1 | tail -80`

**Step 5: Commit**

```bash
git add ferroml-python/tests/test_frankenstein.py
git commit -m "test: Layer 5 Frankenstein tests — pipeline, stateful, ensemble, automl, serialization, threads"
```

---

### Task 2: Fix Frankenstein Test Failures

**Files:**
- Modify: `ferroml-python/tests/test_frankenstein.py` (adjust tests to match actual API)
- Potentially modify: `ferroml-python/src/pipeline.rs`, `ferroml-python/src/ensemble.rs`

**Step 1: Analyze test failures from Task 1**

Categorize failures:
- API mismatches (wrong constructor params) → fix tests
- Missing pickle support → fix bindings or mark xfail
- Thread safety issues → investigate and fix
- Genuine composition bugs → fix in Rust core

**Step 2: Fix API mismatches in tests**

Adjust constructor parameters, method names, import paths to match actual ferroml API.

**Step 3: Handle pickle failures**

If pickle is not supported for certain models, either:
a. Implement `__getstate__`/`__setstate__` in the PyO3 binding, or
b. Mark those tests as `@pytest.mark.xfail(reason="pickle not yet supported for X")`

**Step 4: Run full test suite**

Run: `pytest ferroml-python/tests/test_frankenstein.py -v --tb=short`

**Step 5: Commit fixes**

```bash
git add -u
git commit -m "fix: resolve Frankenstein test API mismatches and failures"
```

---

### Task 3: Rust-Side Frankenstein Tests (Pipeline Composition in correctness.rs)

**Files:**
- Modify: `ferroml-core/tests/correctness.rs` (append Layer 5 section)

**Step 1: Add Layer 5 module to correctness.rs**

Append a `layer5_frankenstein` module with ~6 tests:

- `test_pipeline_scaler_linreg_matches_manual` — Pipeline vs manual step-by-step, exact match
- `test_pipeline_minmax_linreg` — MinMaxScaler + LinearRegression, R² > 0.9
- `test_pipeline_scaler_logreg` — StandardScaler + LogisticRegression, accuracy > 0.85
- `test_pipeline_predict_before_fit_errors` — predict before fit returns Err
- `test_pipeline_refit_replaces_state` — second fit completely replaces first
- `test_pipeline_scaler_pca_logreg` — 3-step pipeline matches manual reproduction

Uses helper functions `make_clf_data(seed)` and `make_reg_data(seed)` for deterministic data generation.

**Step 2: Run correctness tests**

Run: `cargo test --test correctness layer5 -- --nocapture 2>&1 | tail -30`

**Step 3: Fix compilation errors**

Adjust imports, trait bounds, constructor signatures to match actual Rust API.

**Step 4: Run and verify all pass**

Run: `cargo test --test correctness layer5`

**Step 5: Commit**

```bash
git add ferroml-core/tests/correctness.rs
git commit -m "test: Layer 5 Rust Frankenstein tests — pipeline composition and state"
```

---

### Task 4: Fix MLP Serialization

**Files:**
- Modify: `ferroml-core/src/neural/mlp.rs` — remove `#[serde(skip)]` on `layers`, implement custom serialization for `Layer`

**Step 1: Read MLP struct and Layer struct**

Read `ferroml-core/src/neural/mlp.rs` to understand `Layer`, `SGDState`, `AdamState` structs and what needs serialization.

**Step 2: Add Serialize/Deserialize to Layer and optimizer state types**

The key fields with `#[serde(skip)]`:
- `layers: Vec<Layer>` — MUST serialize (contains weights)
- `sgd_state` / `adam_state` — CAN skip (optimizer state not needed for inference)
- `rng` — CAN skip (only needed for training)

Add `#[derive(Serialize, Deserialize)]` to `Layer` struct (and any inner types it depends on like `Activation`). Then remove `#[serde(skip)]` from `layers`.

**Step 3: Write test for MLP serialization**

Add to correctness.rs or edge_cases.rs:

```rust
#[test]
fn test_mlp_serialization_preserves_predictions() {
    // Fit MLP, serialize, deserialize, verify predictions match
}
```

**Step 4: Run test and fix**

Run: `cargo test --test correctness test_mlp_serialization`

**Step 5: Commit**

```bash
git add ferroml-core/src/neural/mlp.rs ferroml-core/tests/correctness.rs
git commit -m "fix: MLP serialization — serialize layers/weights for save/load"
```

---

### Task 5: RandomForest Determinism Documentation + Test

**Files:**
- Modify: `ferroml-python/tests/test_frankenstein.py` (add RF determinism test)

The master design lists RF non-determinism as a medium issue. The non-determinism comes from Rayon thread scheduling — inherent to parallel floating-point. sklearn has the same behavior. Fix: document and test that `n_jobs=1` + `random_state` gives deterministic results.

**Step 1: Add RF determinism tests**

Add `TestRandomForestDeterminism` class:
- `test_rf_deterministic_sequential` — n_jobs=1 + random_state is fully deterministic
- `test_rf_parallel_close_results` — parallel RF with same seed, >95% agreement

**Step 2: Run tests**

Run: `pytest ferroml-python/tests/test_frankenstein.py::TestRandomForestDeterminism -v`

**Step 3: Commit**

```bash
git add ferroml-python/tests/test_frankenstein.py
git commit -m "test: RF determinism verification (sequential deterministic, parallel close)"
```

---

## Session 8 Scope (next session)

- Performance regression investigation (SVC, KMeans — check if Plan W already fixed these)
- Final cross-library validation sweep
- Update correctness report with Layer 5 results
- Update MEMORY.md

---

## Test Count Estimate

| Category | New Tests | Location |
|----------|-----------|----------|
| Pipeline composition | 6 | test_frankenstein.py |
| Stateful interactions | 6 | test_frankenstein.py |
| Ensemble composition | 8 | test_frankenstein.py |
| AutoML end-to-end | 5 | test_frankenstein.py |
| Serialization | 5 | test_frankenstein.py |
| Thread safety | 4 | test_frankenstein.py |
| Performance | 2 | test_frankenstein.py |
| RF determinism | 2 | test_frankenstein.py |
| Rust pipeline composition | 6 | correctness.rs |
| MLP serialization | 1 | correctness.rs |
| **Total** | **~45** | |
