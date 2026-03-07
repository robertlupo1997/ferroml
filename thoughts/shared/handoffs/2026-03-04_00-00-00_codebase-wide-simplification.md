# Codebase-Wide Simplification via 20 Parallel Review Agents

**Date**: 2026-03-04
**Base commit**: `05c2bb1`
**Branch**: `master`
**Status**: partially complete — changes uncommitted, all 3,169 tests passing

## What was done

### Approach

20 parallel `code-simplifier` agents were launched across the entire codebase (~148K lines), partitioned into 8 logical areas with 3 review types each (reuse, quality, efficiency):

| Area | Files | Reuse | Quality | Efficiency |
|------|-------|-------|---------|------------|
| Core models (18 files, ~33K lines) | models/ | Full report | Rate-limited | Rate-limited |
| Preprocessing + pipeline (9 files, ~18K lines) | preprocessing/, pipeline/ | Rate-limited | Rate-limited | Rate-limited |
| Ensemble + neural + decomposition (18 files, ~16K lines) | ensemble/, neural/, decomposition/ | Rate-limited | Rate-limited | Rate-limited |
| Explainability + stats + metrics (20 files, ~16K lines) | explainability/, stats/, metrics/ | Context-limited | Context-limited | Context-limited |
| AutoML + HPO + datasets (18 files, ~18K lines) | automl/, hpo/, datasets/ | Rate-limited | Rate-limited | — |
| Infrastructure (17 files, ~17K lines) | linalg, simd, onnx, etc. | Rate-limited | Rate-limited | — |
| PyO3 bindings (18 files, ~18K lines) | ferroml-python/src/ | Rate-limited | **Completed (3 fixes)** | Rate-limited |
| Python __init__.py wrappers (12 files, ~874 lines) | ferroml-python/python/ | **Completed (3 findings)** | — | — |

4 agents completed fully. ~12 agents did substantial work (50-100+ tool calls) before hitting API rate limits. Several agents conflicted by editing the same files, causing some changes to be clobbered.

### Surviving changes (7 modified files, 1 new file, net -117 lines)

#### 1. `ferroml-core/src/stats/math.rs` (NEW, 424 lines)

Centralized numerical algorithms previously duplicated across `stats/hypothesis.rs`, `stats/mod.rs`, and other files:
- `gamma_ln`, `beta`, `ln_beta`, `incomplete_beta`, `beta_cf`
- `t_cdf`, `t_pdf`, `t_critical`, `normal_cdf`, `z_critical`, `erf`
- `chi2_cdf`, `incomplete_gamma`
- `percentile_ci`, `bootstrap_std_error`, `percentile`, `norm_ppf`, `pearson_r`
- Includes its own test module (4 tests)

#### 2. `ferroml-core/src/stats/hypothesis.rs` (-96 lines)

Removed duplicated `normal_cdf`, `erf`, `t_cdf`, `incomplete_beta`, `beta_cf`, `gamma_ln`. Replaced with thin delegates to `super::math::*`.

#### 3. `ferroml-core/src/stats/mod.rs` (+1 line)

Added `pub mod math;` registration.

#### 4. `ferroml-core/src/ensemble/bagging.rs` (-56 lines net)

Extracted `generate_bootstrap_indices()` and `generate_feature_indices()` from both `BaggingClassifier` and `BaggingRegressor` into module-level free functions. Identical implementations were previously duplicated in both structs. Call sites changed from `Self::generate_bootstrap_indices(...)` to `generate_bootstrap_indices(...)`.

#### 5. `ferroml-core/src/preprocessing/sampling.rs` (+46 lines net, restructured)

Extracted 3 shared helpers for resamplers:
- `validate_resampler_input(x, y)` — validates non-empty + shape match
- `count_classes(y)` — returns `(HashMap<i64, usize>, HashMap<i64, Vec<usize>>)`
- `assemble_resampled(all_x, all_y, n_features)` — builds output arrays

Also simplified BorderlineSMOTE: removed dead `_effective_k` variable, unified Borderline-1/Borderline-2 logic (both used identical neighbor search), removed redundant comments.

#### 6. `ferroml-core/src/models/knn.rs` (efficiency)

BallTree optimizations — eliminated unnecessary `Vec<f64>` allocations:
- `compute_radius`: use `data.row(idx)` view instead of `.to_vec()`
- `find_widest_spread_dim`: streaming min/max instead of collecting to Vec
- `query_recursive`: use `row.as_slice().unwrap()` instead of `.to_vec()`

#### 7. `ferroml-core/src/preprocessing/scalers.rs` (efficiency)

`StandardScaler::transform` and `inverse_transform`: fused center + scale into a single column-wise pass. Previously did two separate passes (`result = result - mean` creating a temporary, then scaling columns). Now iterates each column once.

#### 8. `ferroml-core/src/models/mod.rs` (+26 lines)

Added shared helpers (not yet wired to call sites):
- `pub fn get_feature_name(feature_names: &Option<Vec<String>>, idx: usize) -> String` — duplicated 7x across models
- `pub fn sorted_median(sorted: &[f64]) -> f64` — duplicated in multiple places

#### 9. `ferroml-core/src/automl/metafeatures.rs` (efficiency)

- `compute_feature_statistics`: replaced `sort+dedup` with `HashSet` for unique counting
- `mean_finite` / `std_finite`: streaming computation instead of collecting to intermediate `Vec<f64>`
- Removed unused `HashSet` import

### Orphaned files (created by agents but not yet integrated)

These were created by agents whose companion changes (modifying `lib.rs`, `trees.rs`, etc.) were clobbered by conflicting agents:

- **`ferroml-python/src/errors.rs`** (48 lines) — Shared PyO3 error helpers:
  - `to_py_runtime_err(e)` — replaces 237 occurrences of `.map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))`
  - `not_fitted_err(entity)` — replaces ~100 occurrences of "not fitted" error construction
  - Not registered in `lib.rs` yet

- **`ferroml-python/python/ferroml/datasets/__init__.py`** (94 lines) — Missing datasets wrapper exposing 2 classes + 9 functions. The `datasets` module was registered in Rust but had no Python wrapper package.

### Changes that were clobbered (agent conflicts)

The following agents completed changes that were overwritten by later agents editing the same files:

- **PyO3 quality agent**: Extracted `parse_knn_weights`/`parse_distance_metric`/`parse_knn_algorithm` in `neighbors.rs`, removed `IntoOk` trait in `decomposition.rs`, removed spurious `sys.modules` in `neural.rs` — all reverted
- **PyO3 reuse agent**: Added shared criterion/loss parsers to `trees.rs`, referenced `errors.rs` from `lib.rs` — reverted
- **Python wrappers agent**: Updated root `__init__.py` to export all 12 submodules (was only 7), improved preprocessing docstring — reverted
- **Infra efficiency agent**: Added tree ensemble helpers to `inference/operators.rs` — manually reverted (dead code)

## Key findings NOT yet implemented

### From models reuse agent (full report, 11 findings)

| # | Pattern | Instances | Files | Priority |
|---|---------|-----------|-------|----------|
| 1 | `sigmoid` function duplicated | 4 | boosting, hist_boosting, logistic, calibration | HIGH |
| 2 | Inline class extraction (should use `get_unique_classes`) | 12 | 8 files | HIGH (also fixes `.dedup()` vs epsilon-aware `.dedup_by()`) |
| 3 | `raw_to_proba` (sigmoid+softmax) duplicated | 4 | boosting, hist_boosting | HIGH |
| 4 | `get_feature_name` duplicated (helper added, call sites not updated) | 7 | 5 files | MEDIUM |
| 5 | `compute_log_loss` duplicated | 2 | boosting, hist_boosting | MEDIUM |
| 6 | `sample_indices` duplicated within boosting.rs | 2 | 1 file | MEDIUM |
| 7 | Tree feature importance duplicated (classifier vs regressor) | 2 | tree.rs | MEDIUM |
| 8 | Ensemble importance + CI duplicated | 4 | forest.rs, extra_trees.rs | MEDIUM |
| 9 | TSS/RSS inline computation | 4 | linear.rs, regularized.rs | MEDIUM |
| 10 | KNN non-SIMD distance duplicates `linalg::squared_euclidean_distance` | 2 | knn.rs | LOW |
| 11 | `r_squared()` method duplicated 3x in regularized.rs | 3 | regularized.rs | LOW |

### From PyO3 quality agent (3 findings fixed, 6 documented)

Documented but not fixable due to PyO3 constraints:
- 31+ typed function wrappers in `explainability.rs` (PyO3 requires concrete types)
- Bagging factory parameter sprawl (PyO3 staticmethod signatures must be spelled out)
- RFE ClosureEstimator boilerplate (different closure types per factory)
- Stringly-typed parameters at Python boundary (correct sklearn-compatible pattern)
- Pickle macro exists but unused (manual implementations are clearer)

## Verification

```bash
# Compile check (0 warnings)
cargo check 2>&1

# Full test suite (3,169 pass, 0 fail, 7 ignored)
cargo test 2>&1 | grep 'test result:'

# Python bindings compile
cargo check -p ferroml-python 2>&1
```

## Action items (priority order)

1. [ ] **Commit surviving changes** — 7 modified + 1 new file, all tests pass
2. [ ] **Re-apply clobbered PyO3 changes** — `neighbors.rs` parsers, `decomposition.rs` IntoOk removal, `neural.rs` sys.modules cleanup, root `__init__.py` fixes
3. [ ] **Integrate orphaned files** — register `errors.rs` in `lib.rs`, add `datasets/__init__.py`
4. [ ] **Wire `get_feature_name` helper** — replace 7 duplicate implementations across 5 model files
5. [ ] **Extract `sigmoid` to `models/mod.rs`** — replace 4 duplicates (finding #1)
6. [ ] **Replace 12 inline class extractions with `get_unique_classes`** — also fixes epsilon comparison bug (finding #2)
7. [ ] **Extract `raw_to_proba`** — replace 4 duplicates in boosting variants (finding #3)
8. [ ] **Remaining medium-priority deduplication** — findings #5-9

## Lessons learned

- **20 parallel agents is too many when they share files** — agents clobbered each other's changes. Better approach: partition by file ownership (no overlap) or use worktree isolation with merge at the end.
- **Rate limits hit hard with 20 concurrent agents** — most agents ran 50-100+ tool calls but hit API rate limits before finishing. Stagger launches or use fewer agents.
- **`code-simplifier` agents make direct edits** — some broke compilation (removed functions without adding replacements). Need to either constrain agents to report-only mode or run compilation checks after each agent.
- **Background agents can't get interactive permission** — tools in "ask" list are effectively denied. Ensure Edit/Write/Bash are in allow list.
