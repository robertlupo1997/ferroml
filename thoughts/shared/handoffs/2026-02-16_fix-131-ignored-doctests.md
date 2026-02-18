---
date: 2026-02-16T12:00:00-06:00
researcher: Claude
git_commit: 1f76d7c (uncommitted changes on top)
git_branch: master
repository: ferroml
topic: Fix 131 ignored doctests
tags: [doctests, documentation, quality]
status: complete
---

# Handoff: Fix 131 Ignored Doctests

## Task Status

### Current Phase
Complete - all 131 ignored doctests converted to runnable.

### Progress
- [x] Audit all 131 ```` ```ignore ```` doctests across 71 files
- [x] Fix models/ batch (27 doctests across 12 files)
- [x] Fix cv/ batch (21 doctests across 8 files)
- [x] Fix explainability/ + automl/ batch (22 doctests across 16 files)
- [x] Fix preprocessing/ + pipeline/ batch (18 doctests across 6 files)
- [x] Fix ensemble/ + clustering/ + decomposition/ batch (9 doctests across 6 files)
- [x] Fix datasets/ batch (9 doctests across 3 files)
- [x] Fix testing/ batch (13 doctests across 8 files)
- [x] Fix serialization.rs + schema.rs (4 doctests across 2 files)
- [x] Fix neural/ + inference/ + onnx/ + simd + sparse + gpu + metrics batch (14 doctests across 10 files)
- [x] Verify all 215 doctests pass (0 ignored, 0 failed)
- [x] Verify clippy clean
- [x] Verify 2469 lib tests pass (no regressions)
- [ ] **NOT YET COMMITTED** — changes are unstaged

## Critical References

1. `ferroml-core/src/` — all 71 modified files are in this directory

## Recent Changes

71 files changed, 1071 insertions(+), 384 deletions(-)

Key categories of changes:
- **models/**: svm.rs, hist_boosting.rs, calibration.rs, regularized.rs, knn.rs, mod.rs, logistic.rs, naive_bayes.rs, tree.rs, adaboost.rs, quantile.rs, robust.rs
- **cv/**: timeseries.rs, curves.rs, mod.rs, group.rs, loo.rs, nested.rs, stratified.rs, search.rs
- **explainability/**: mod.rs, h_statistic.rs, ice.rs, kernelshap.rs, partial_dependence.rs, permutation.rs, summary.rs, treeshap.rs
- **automl/**: fit.rs, ensemble.rs, metafeatures.rs, mod.rs, preprocessing.rs, time_budget.rs, transfer.rs, warmstart.rs
- **preprocessing/**: sampling.rs, mod.rs, power.rs, quantile.rs, selection.rs
- **pipeline/**: mod.rs
- **ensemble/**: bagging.rs, stacking.rs, voting.rs
- **clustering/**: agglomerative.rs, mod.rs
- **decomposition/**: mod.rs
- **datasets/**: loaders.rs, mmap.rs, mod.rs
- **testing/**: assertions.rs, mod.rs, mutation.rs, callbacks.rs, nan_inf_validation.rs, onnx.rs, properties.rs, serialization.rs
- **Other**: serialization.rs, schema.rs, neural/{classifier,mod,regressor}.rs, inference/{mod,session}.rs, onnx/mod.rs, simd.rs, sparse.rs, gpu/mod.rs, metrics/mod.rs

## Key Learnings

### What Worked
- Parallel agents (7 batches) for initial pass, covering all 71 files concurrently
- Common fix patterns: Result wrappers, hidden imports, hidden data setup with `# ` prefix

### What Didn't Work
- Sequential `(0..N)` data causes collinear features → `sin()` used for non-collinear data
- `SVC` doesn't implement `CalibrableClassifier` → used `LogisticRegression` instead
- `PCA`/`PolynomialFeatures` don't implement `PipelineTransformer` → used `StandardScaler`/`MinMaxScaler`
- `CategoricalFeatureHandler::transform()` panics with `is_training=true` and `permutation=None` → used `is_training=false`
- One agent hit "Prompt is too long" → split into 3 smaller agents

### Important Discoveries
- `FeatureUnion::fit_transform()` requires `use ferroml_core::preprocessing::Transformer` trait import
- Only scalers and imputers implement `PipelineTransformer` (not PCA, PolynomialFeatures, OneHotEncoder, etc.)
- AutoML doctests use `no_run` (3600s time budget would timeout)
- GPU/inference doctests use `no_run` (hardware-dependent)
- Some visible doctest code had wrong API signatures (fixed: missing args, wrong method names)

## Verification Commands

```bash
# Full doctest suite — expect 215 passed, 0 failed, 0 ignored
cargo test -p ferroml-core --doc

# Clippy — expect clean
cargo clippy -p ferroml-core -- -D warnings

# Lib tests — expect 2469 passed, 0 failed
cargo test -p ferroml-core --lib

# Confirm zero ignores remain
grep -rn '```ignore' ferroml-core/src/ --include="*.rs"
```

## Action Items & Next Steps

Priority order:
1. [ ] Commit the changes (71 files, ~1071 insertions)
2. [ ] Consider adding more `PipelineTransformer` impls (PCA, PolynomialFeatures, OneHotEncoder) to enable richer pipeline examples
3. [ ] Consider adding `CalibrableClassifier` impl for SVC

## Other Notes

- Doctest compilation is very slow on this Windows machine (~15 min for full suite)
- Some agents introduced LF line endings on Windows (CRLF warnings in git diff)
- The `sparse.rs` doctest had a visible code change (added `use ndarray::array;`) — all other visible code preserved or minimally adjusted
