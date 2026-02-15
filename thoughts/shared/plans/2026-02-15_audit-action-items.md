# Audit Action Items: CI Green + v0.1 Polish

## Overview
Execute remaining action items from the FerroML Audit Report. CI fix is already done (committed `18b6f1e`). This plan covers: README corrections, dead code removal, license files, AutoML algorithm wiring, AutoMLResult.predict() in Python, and a benchmark script.

## Current State
- CI yml fixed (check/test scoped to `-p ferroml-core`) ✓
- `cargo test -p ferroml-core --all-features`: 2,608 pass, 0 fail ✓
- `pytest`: 151 pass, 17 skipped, 0 fail ✓
- README has inaccurate test count, broken Quick Start, badges for unpublished crate
- `stats/distributions.rs` is a 37-line unused placeholder
- No LICENSE-APACHE or LICENSE-MIT files in repo root
- 5 fully-implemented models (SVC, SVR, LinearSVC, LinearSVR, QuantileRegression) not wired into AutoML dispatch
- AutoMLResult has no predict() on Rust or Python side

## Desired End State
- README is accurate and Quick Start examples actually work
- No dead code placeholders
- Proper license files present
- AutoML can dispatch to all implemented models
- AutoMLResult.predict() works in Python
- Benchmark script demonstrates ferroml vs sklearn

---

## Implementation Phases

### Phase 1: README Update (parallel agent 1)
**Overview**: Fix inaccurate claims, broken Quick Start, and misleading badges.

**Changes Required**:
1. **File**: `README.md`
   - Line 7: Change test badge from `2395` to `2608`
   - Line 11: Change "2395 tests passing" to "2608 tests passing"
   - Lines 3-4: Remove crates.io and docs.rs badges (not published). Keep CI and License badges.
   - Lines 82-103: Rewrite Python Quick Start to match actual API:
     ```python
     import ferroml as fml

     # Load data
     dataset, info = fml.datasets.load_iris()
     X, y = dataset.x, dataset.y

     # Create and fit a model
     config = fml.AutoMLConfig(
         task="classification",
         metric="accuracy",
         time_budget_seconds=60,
     )
     automl = fml.AutoML(config)
     result = automl.fit(X, y)

     # Results with statistical guarantees
     best = result.best_model()
     print(f"Best: {best.algorithm}")
     print(f"Score: {best.cv_score:.4f} ± {best.cv_std:.4f}")
     print(f"95% CI: [{best.ci_lower:.4f}, {best.ci_upper:.4f}]")
     ```
   - Line 228: Update test count in Project Status table to 2,608

**Success Criteria**:
- [ ] Automated: `grep -c "2608" README.md` returns matches
- [ ] Manual: No references to unpublished crate badges

---

### Phase 2: Dead Code & License Files (parallel agent 2)
**Overview**: Delete unused distributions placeholder, add dual license files.

**Changes Required**:
1. **File**: `ferroml-core/src/stats/distributions.rs` — DELETE entirely
2. **File**: `ferroml-core/src/stats/mod.rs` line 17 — Remove `pub mod distributions;`
3. **File**: `LICENSE-MIT` — Create with standard MIT text, copyright "2025-2026 Robert Lupo"
4. **File**: `LICENSE-APACHE` — Create with Apache 2.0 text, copyright "2025-2026 Robert Lupo"

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --lib` still passes
- [ ] Automated: `cargo clippy -p ferroml-core -- -D warnings` clean
- [ ] Manual: LICENSE-MIT and LICENSE-APACHE exist in repo root

---

### Phase 3: Wire 5 Models into AutoML Dispatch (parallel agent 3)
**Overview**: SVC, SVR, LinearSVC, LinearSVR, QuantileRegression are fully implemented but return NotImplemented in AutoML's `create_model()`. Wire them in.

**Changes Required**:
1. **File**: `ferroml-core/src/automl/fit.rs`
   - Line 51-56: Add imports:
     ```rust
     use crate::models::svm::{SVC, SVR, LinearSVC, LinearSVR};
     use crate::models::quantile::QuantileRegression;
     ```
     (Check exact module paths — may need `crate::models::regularized::QuantileRegression` or `crate::models::quantile::QuantileRegression`)
   - Lines 1055-1064: Replace the NotImplemented arm with actual constructors:
     ```rust
     AlgorithmType::SVC => Ok(Box::new(SVC::new())),
     AlgorithmType::SVR => Ok(Box::new(SVR::new())),
     AlgorithmType::LinearSVC => Ok(Box::new(LinearSVC::new())),
     AlgorithmType::LinearSVR => Ok(Box::new(LinearSVR::new())),
     AlgorithmType::QuantileRegression => Ok(Box::new(QuantileRegression::new(0.5))),
     AlgorithmType::RobustRegression => Err(FerroError::NotImplemented(format!(
         "Algorithm {:?} not yet available for AutoML", algorithm
     ))),
     ```
   - Note: Keep RobustRegression as NotImplemented unless it exists and is complete.

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --lib` passes
- [ ] Automated: `cargo clippy -p ferroml-core -- -D warnings` clean
- [ ] Automated: Test that AutoML with thorough portfolio doesn't error on SVC/SVR

---

### Phase 4: AutoMLResult.predict() in Python (parallel agent 4)
**Overview**: Add predict() to AutoMLResult on both Rust and Python sides. The result needs to store the best fitted model and delegate predict to it.

**Changes Required**:
1. **File**: `ferroml-core/src/automl/fit.rs`
   - Add a `best_fitted_model: Option<Box<dyn Model>>` field to `AutoMLResult` struct (lines 71-101)
   - During fit, after selecting best model, store a clone of it in the result
   - Add `pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>` method that delegates to stored model
   - **Complication**: `Box<dyn Model>` may not be Clone or Serialize. May need to re-fit best model on full data, or store the trial's fitted model. Investigate how `TrialResult` stores models.

2. **File**: `ferroml-python/src/automl.rs`
   - Add `#[pyo3(name = "predict")]` method to `PyAutoMLResult` that calls the Rust predict
   - Accept numpy array, return numpy array

**Risks**: The `Model` trait may not be object-safe for cloning/serialization. May need to store the model differently. This phase may require more investigation.

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --lib` passes
- [ ] Automated: Python test — `result = automl.fit(X, y); preds = result.predict(X)` works
- [ ] Manual: Predictions are reasonable (accuracy > random)

---

### Phase 5: Benchmark Script (parallel agent 5)
**Overview**: Write a Python benchmark comparing ferroml vs sklearn on Iris classification and a regression dataset.

**Changes Required**:
1. **File**: `benchmarks/ferroml_vs_sklearn.py` — CREATE
   - Compare LinearRegression, Ridge, RandomForest on Iris (classification) and diabetes (regression)
   - Time fit + predict for both libraries
   - Report accuracy/R² and wall-clock time
   - Output a markdown table

**Success Criteria**:
- [ ] Automated: `python benchmarks/ferroml_vs_sklearn.py` runs without error
- [ ] Manual: Results show competitive accuracy

---

## Parallelization Strategy

All 5 phases are independent. Run as parallel agents:

| Agent | Phase | Estimated Context |
|-------|-------|-------------------|
| 1 | README update | Light (single file edit) |
| 2 | Dead code + licenses | Light (delete file, create 2 files) |
| 3 | AutoML wiring | Medium (imports + match arm edit, verify compilation) |
| 4 | AutoMLResult.predict() | Heavy (Rust + Python changes, new trait requirements) |
| 5 | Benchmark script | Medium (new Python file, needs sklearn installed) |

After all agents complete: run `cargo test -p ferroml-core --lib` and `pytest` to verify nothing broke, then commit all changes.

## Dependencies
- Phase 4 (predict) benefits from Phase 3 (wiring) being done first, but they touch different parts of the file and can be developed in parallel.
- Phase 5 (benchmark) ideally runs after Phase 4, but can be written to skip predict if not available.

## Risks & Mitigations
1. **Phase 4 Model storage**: `dyn Model` may not support Clone. Mitigation: re-fit best model on full training data inside `predict()`, or store serialized bytes.
2. **Phase 3 constructor signatures**: SVC::new() etc. may require different args. Mitigation: agent checks actual `new()` signatures before editing.
3. **Phase 5 sklearn dependency**: sklearn may not be installed. Mitigation: script checks for sklearn and prints instructions if missing.
