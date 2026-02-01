# FerroML Ralph Loop - Self-Correcting AI Development

> **Purpose**: This document enables AI assistants to autonomously implement the FerroML testing suite and continuously improve the project. It is designed to be resumed, self-correcting, and publicly verifiable.

---

## Quick Start (For AI Agents)

```
Resume ferroml development from RALPH_LOOP.md
```

**First Actions:**
1. Read this entire document
2. Check `PHASE_STATUS` section for current progress
3. Run verification commands to assess project state
4. Pick up the next incomplete phase
5. Update this document after completing work

---

## Phase Status Tracker

<!-- AI: Update this section after completing each phase -->

| Phase | Name | Status | Tests Added | Committed |
|-------|------|--------|-------------|-----------|
| 16 | AutoML Time Budget | IN_PROGRESS | 177 | NO |
| 17 | HPO Correctness | IN_PROGRESS | 82+ | NO |
| 18 | Early Stopping & Callbacks | IN_PROGRESS | ~50 | NO |
| 19 | Explainability Tests | IN_PROGRESS | 118+ | NO |
| 20 | ONNX Parity | IN_PROGRESS | 26+ | NO |
| 21 | Sample Weights | NOT_STARTED | 0 | NO |
| 22 | Sparse Data Support | NOT_STARTED | 0 | NO |
| 23 | Multi-output Predictions | NOT_STARTED | 0 | NO |
| 24 | Advanced Cross-validation | NOT_STARTED | 0 | NO |
| 25 | Ensemble Stacking | NOT_STARTED | 0 | NO |
| 26 | Categorical Features | NOT_STARTED | 0 | NO |
| 27 | Warm Start / Incremental | NOT_STARTED | 0 | NO |
| 28 | Custom Metrics | NOT_STARTED | 0 | NO |
| 29 | Fairness Testing | NOT_STARTED | 0 | NO |
| 30 | Drift Detection | NOT_STARTED | 0 | NO |
| 31 | Regression Test Suite | NOT_STARTED | 0 | NO |
| 32 | Mutation Testing | NOT_STARTED | 0 | NO |

**Current Focus**: Phases 16-20 (files exist, need module registration and commits)

---

## Self-Correction Protocol

Before starting any work, run these verification commands:

```bash
# 1. Check project compiles
cargo check -p ferroml-core 2>&1 | head -20

# 2. Check test count
cargo test -p ferroml-core 2>&1 | grep -E "^test result"

# 3. Check for uncommitted work
git status --short

# 4. Check current branch
git branch --show-current

# 5. Check recent commits
git log --oneline -5
```

### Error Recovery

**If compilation fails:**
1. Read the error message carefully
2. Check if a module is missing from mod.rs
3. Check for missing imports or dependencies
4. Fix the issue before proceeding

**If tests fail:**
1. Run the specific failing test with `cargo test <test_name> -- --nocapture`
2. Understand why it fails
3. Either fix the test or fix the implementation
4. Document any intentional deviations

**If git is in a bad state:**
1. Run `git status` to understand current state
2. If there are uncommitted changes that should be committed, commit them
3. If there are changes that should be discarded, ask the user first
4. Never force push or reset without explicit user permission

---

## Phase Implementation Guide

### Phases 16-20: Current Priority (Files Exist)

These test files already exist but need to be properly integrated:

```
ferroml-core/src/testing/
├── automl.rs        # 1357 lines - Phase 16
├── callbacks.rs     # 1670 lines - Phase 18
├── explainability.rs # 1444 lines - Phase 19
├── hpo.rs           # 1153 lines - Phase 17
└── onnx.rs          # 1702 lines - Phase 20
```

**Step 1: Register modules in mod.rs**

Add to `ferroml-core/src/testing/mod.rs` (after line 48):
```rust
pub mod callbacks;
pub mod explainability;
pub mod hpo;
pub mod onnx;
```

**Step 2: Verify compilation**
```bash
cargo check -p ferroml-core
```

**Step 3: Run all tests**
```bash
cargo test -p ferroml-core automl callback hpo explain onnx
```

**Step 4: Create commits for each phase**

```bash
# Phase 16
git add ferroml-core/src/testing/automl.rs
git commit -m "test(automl): Phase 16 - Time budget and trial management tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Phase 17
git add ferroml-core/src/testing/hpo.rs
git commit -m "test(hpo): Phase 17 - HPO sampler and pruner correctness tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Phase 18
git add ferroml-core/src/testing/callbacks.rs ferroml-core/src/models/boosting.rs
git commit -m "test(callbacks): Phase 18 - Early stopping and callback tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Phase 19
git add ferroml-core/src/testing/explainability.rs
git commit -m "test(explain): Phase 19 - SHAP, PDP, and feature importance tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Phase 20
git add ferroml-core/src/testing/onnx.rs
git commit -m "test(onnx): Phase 20 - ONNX export/import parity tests

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

# Module registration
git add ferroml-core/src/testing/mod.rs
git commit -m "chore(testing): Register test modules for phases 16-20

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Phases 21-28: Advanced Features (Need Implementation)

For each phase, follow this pattern:

1. **Research**: Read the existing implementation in the relevant module
2. **Create test file**: `ferroml-core/src/testing/{phase_name}.rs`
3. **Register module**: Add to `mod.rs`
4. **Write tests**: Follow existing patterns in the testing directory
5. **Verify**: `cargo test -p ferroml-core {phase_name}`
6. **Commit**: Follow commit message pattern

#### Phase 21: Sample Weights & Class Weights

**File**: `ferroml-core/src/testing/weights.rs`

Test coverage needed:
- [ ] Test sample_weight parameter for all models that support it
- [ ] Test class_weight='balanced' produces correct weights
- [ ] Test custom class weights dictionary
- [ ] Test that weighted metrics match expectations
- [ ] Test edge cases: zero weights, negative weights (should error)

Reference implementation: Check `Model` trait for `fit_weighted` method.

#### Phase 22: Sparse Data Support

**File**: `ferroml-core/src/testing/sparse.rs`

Test coverage needed:
- [ ] Test sparse input handling for LinearRegression, RidgeRegression
- [ ] Test sparse input for tree models (if supported)
- [ ] Test CSR and CSC format compatibility
- [ ] Test sparse/dense consistency (same results)
- [ ] Test memory efficiency (sparse should use less memory)

Reference: `ferroml-core/src/sparse.rs`

#### Phase 23: Multi-output Predictions

**File**: `ferroml-core/src/testing/multioutput.rs`

Test coverage needed:
- [ ] Test multi-output regression (multiple y columns)
- [ ] Test multi-label classification
- [ ] Test prediction shape matches y shape
- [ ] Test per-output metrics

#### Phase 24: Advanced Cross-validation

**File**: `ferroml-core/src/testing/cv_advanced.rs`

Test coverage needed:
- [ ] Test NestedCV prevents data leakage
- [ ] Test GroupKFold keeps groups together
- [ ] Test TimeSeriesSplit respects temporal order
- [ ] Test learning_curve produces valid data
- [ ] Test validation_curve produces valid data

Reference: `ferroml-core/src/cv/`

#### Phase 25: Ensemble Stacking

**File**: `ferroml-core/src/testing/ensemble_advanced.rs`

Test coverage needed:
- [ ] Test StackingClassifier with various meta-learners
- [ ] Test StackingRegressor with various meta-learners
- [ ] Test passthrough option
- [ ] Test that stacking doesn't leak data (CV-based)
- [ ] Test ensemble weights optimization

Reference: `ferroml-core/src/ensemble/`

#### Phase 26: Categorical Feature Handling

**File**: `ferroml-core/src/testing/categorical.rs`

Test coverage needed:
- [ ] Test native categorical handling in HistGradientBoosting
- [ ] Test ordered target encoding (CatBoost-style)
- [ ] Test unknown category handling
- [ ] Test mixed categorical/numeric features

#### Phase 27: Warm Start / Incremental

**File**: `ferroml-core/src/testing/incremental.rs`

Test coverage needed:
- [ ] Test partial_fit for NaiveBayes classifiers
- [ ] Test warm_start for ensemble models
- [ ] Test incremental learning preserves previous knowledge
- [ ] Test online learning scenarios

#### Phase 28: Custom Metrics

**File**: `ferroml-core/src/testing/metrics_custom.rs`

Test coverage needed:
- [ ] Test custom scoring function in cross_val_score
- [ ] Test custom metric in HPO
- [ ] Test multi-objective optimization
- [ ] Test metric with confidence intervals

---

### Phases 29-32: Advanced Quality (Industry-Leading)

These phases add features unique to FerroML in the Rust ecosystem.

#### Phase 29: Fairness Testing

**File**: `ferroml-core/src/testing/fairness.rs`

Create module with:
```rust
//! Fairness testing for ML models
//!
//! Implements fairness metrics and bias detection:
//! - Demographic parity
//! - Equalized odds
//! - Calibration across groups

pub fn check_demographic_parity<M: Model>(...) -> f64 { ... }
pub fn check_disparate_impact<M: Model>(...) -> f64 { ... }
pub fn audit_fairness<M: Model>(...) -> FairnessReport { ... }
```

Test coverage:
- [ ] Test fair model passes demographic parity
- [ ] Test biased model is detected
- [ ] Test disparate impact ratio calculation
- [ ] Test group accuracy reporting

#### Phase 30: Drift Detection

**File**: `ferroml-core/src/testing/drift.rs`

Create module with:
```rust
//! Drift detection testing
//!
//! Test data and concept drift detection capabilities.

pub enum DriftType { DataDrift, ConceptDrift, LabelDrift }
pub fn detect_data_drift(reference: &Array2<f64>, current: &Array2<f64>) -> DriftResult { ... }
```

Test coverage:
- [ ] Test no drift detected for same distribution
- [ ] Test drift detected for shifted distribution
- [ ] Test feature-wise drift identification

#### Phase 31: Regression Test Suite

**File**: `ferroml-core/tests/regression/baselines.json` + `test_baselines.rs`

Create performance baselines:
```json
{
  "LinearRegression": { "r2_min": 0.74, "fit_ms_max": 50 },
  "RandomForestRegressor": { "r2_min": 0.85, "fit_ms_max": 500 }
}
```

Test coverage:
- [ ] Test all models meet accuracy baselines
- [ ] Test all models meet performance baselines
- [ ] Test baselines are versioned and reproducible

#### Phase 32: Mutation Testing

**File**: `.github/workflows/mutation.yml`

Add mutation testing workflow:
- [ ] Configure cargo-mutants
- [ ] Set mutation score threshold (80%)
- [ ] Run weekly on schedule
- [ ] Report mutation score trends

---

## Continuous Improvement Protocol

After completing all testing phases, the ralph loop continues with:

### Quality Improvements

1. **Increase test coverage**: Target 90%+ code coverage
2. **Add property-based tests**: Use proptest for invariant testing
3. **Add fuzzing tests**: Use cargo-fuzz for security testing
4. **Performance regression detection**: Track benchmark results over time

### Feature Improvements

1. **Review GitHub issues**: Address user-reported bugs and feature requests
2. **sklearn compatibility**: Ensure API matches sklearn where possible
3. **Documentation**: Keep docs current with implementation
4. **Examples**: Add examples for new features

### Community Engagement

1. **PR reviews**: Review and merge community contributions
2. **Issue triage**: Label and prioritize issues
3. **Release notes**: Document changes for each release

---

## Commit Message Format

All commits should follow this format:

```
<type>(<scope>): <description>

<optional body>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `test`: Adding or updating tests
- `docs`: Documentation
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

---

## Reference Documents

- `IMPLEMENTATION_PLAN.md` - Full implementation status (Phase 1-12 complete)
- `thoughts/shared/plans/2026-01-22_comprehensive-testing.md` - Testing plan details (32 phases)
- `thoughts/shared/handoffs/` - Previous session handoffs

---

## Success Metrics

Track these metrics to measure progress:

| Metric | Current | Target |
|--------|---------|--------|
| Unit Tests | ~1,700 | 3,500+ |
| Code Coverage | ~73% | 90%+ |
| Property Tests | ~200 | 500+ |
| Integration Tests | ~50 | 100+ |
| sklearn Compat | ~80% | 100% |
| Mutation Score | Unknown | 80%+ |

---

## Session Logging

<!-- AI: Add a log entry after each session -->

### Session Log

**2026-02-01**: Initial ralph loop creation
- Analyzed phases 16-20 status (files exist, need registration)
- Created comprehensive self-correcting prompt
- Documented all 32 testing phases

---

## Emergency Recovery

If the project gets into a bad state:

```bash
# Check what changed
git diff HEAD~5

# See all uncommitted changes
git status

# Stash changes temporarily
git stash

# Return to last known good state
git checkout HEAD~1

# Or create a fresh branch from main
git checkout -b fix/recovery master
```

**Never run destructive commands without user approval:**
- `git reset --hard`
- `git push --force`
- `git clean -f`
- `rm -rf`

---

## How to Use This Document

1. **Starting a new session**: Read this document, check phase status, run verification
2. **During work**: Update phase status as you complete tasks
3. **Ending a session**: Add session log entry, commit this file if updated
4. **Handing off**: The next session can pick up exactly where you left off

This document is designed to be:
- **Self-contained**: All information needed to continue work
- **Self-correcting**: Verification steps catch problems early
- **Publicly verifiable**: Anyone can check progress via git history
- **AI-friendly**: Clear instructions for autonomous operation

---

*"The greatest ML library testing framework ever designed for Rust"*
