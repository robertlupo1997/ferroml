---
date: 2026-03-13T21:35:00-05:00
researcher: Claude
git_commit: 9368660474f355c42a6e97459ddaffb51776aa07
git_branch: master
repository: ferroml
topic: Plan S ONNX — S.8 complete, fix 9 xfail models, then S.9/S.10
tags: [plan-S, onnx, s8, s8-fixes]
status: in-progress
---

# Handoff: Plan S — ONNX Round-Trip Validation Complete, Fix xfail Models

## Task Status

### Current Phase
S.8: External Round-Trip Validation with onnxruntime — **COMPLETE**

### Progress
- [x] S.1–S.6: Rust ONNX export for all 34 models
- [x] S.7: Python bindings (export_onnx/to_onnx_bytes) for all 34 models, 42 tests passing
- [x] S.8: Round-trip validation — 44 pytest tests (35 pass, 9 xfail), validation script
- [ ] **FIX xfail models** (9 models with known ONNX export bugs — research complete, fixes designed below)
- [ ] S.9: MultiOutput composite ONNX export
- [ ] S.10: Documentation and compatibility matrix

### S.8 Results Summary
- **25/34 models pass** onnxruntime round-trip perfectly
- **9 models xfail** with documented root causes and researched fixes (see below)
- Test file: `ferroml-python/tests/test_onnx_roundtrip.py` (44 tests)
- Validation script: `scripts/validate_onnx_exports.py` (outputs compatibility matrix)

## Critical References

1. `thoughts/shared/plans/2026-03-11_plan-S-onnx-export-coverage.md` — Full plan
2. `ferroml-python/tests/test_onnx_roundtrip.py` — Round-trip tests (remove xfail as bugs are fixed)
3. `scripts/validate_onnx_exports.py` — Run to verify all 34 models
4. `ferroml-core/src/onnx/` — All ONNX export implementations

## Recent Changes (This Session)

- `ferroml-python/tests/test_onnx_roundtrip.py` — NEW: 44 round-trip tests
- `scripts/validate_onnx_exports.py` — NEW: validation script with compatibility matrix
- Installed `onnxruntime 1.24.3` in `.venv`

---

## The 9 xfail Models: Root Causes and Fixes

### Bug 1: TreeEnsembleClassifier output type (fixes 4 models)

**Affected models:** DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

**Root cause:** `create_proba_output()` in `ferroml-core/src/onnx/tree.rs:404-436` declares the second output as `sequence<map<int64, tensor(float)>>`. The ONNX spec says TreeEnsembleClassifier's second output is `tensor(float)` with shape `[N, n_classes]`. The `sequence<map<...>>` type is only produced by the optional `ZipMap` post-processor, which we don't use.

**Fix:**
1. In `tree.rs`, replace `create_proba_output()` with a call to `create_tensor_output()` (the 2D version from `mod.rs:296`) with `elem_type: Float` and shape `[batch_size, n_classes]`.
2. The node's second output should be named something like `"{output_name}_probabilities"`.
3. The node still has 2 outputs: `[label_name, proba_name]`.
4. The first output (labels) is already correct: `tensor(int64)` with shape `[N]`.

**Reference:** sklearn-onnx's `random_forest.py` converter uses `FloatTensorType(shape=[N, n_classes])` for the second output. [ONNX TreeEnsembleClassifier spec](https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html) confirms output Z is `tensor(float)`.

**Key code location:** `ferroml-core/src/onnx/tree.rs:344-436` (`create_tree_classifier_graph` and `create_proba_output`)

**After fix:** Update round-trip tests to verify both label output AND probability output match. The ORT result will have 2 arrays: `result[0]` = labels (int64), `result[1]` = probabilities (float32 [N, n_classes]).

---

### Bug 2: SVMClassifier missing second output (fixes 1 model)

**Affected model:** SVC

**Root cause:** `ferroml-core/src/onnx/svm.rs:552-614` declares only 1 output (Int64 labels) but the ONNX SVMClassifier spec **requires exactly 2 outputs**: labels + scores/probabilities. The node also only lists 1 output name. ORT rejects the graph as invalid.

**Fix:**
1. Add a second output: `tensor(float)` with shape `[N, n_classes]` for scores.
2. Update `NodeProto.output` to have 2 entries: `[label_name, scores_name]`.
3. Update `GraphProto.output` to have 2 `ValueInfoProto` entries.
4. The label output stays `tensor(int64)` shape `[N]`.
5. `post_transform` can stay `"NONE"` for raw scores, or use `"LOGISTIC"` for probability calibration if `probability=True`.

**Reference:** sklearn-onnx's `support_vector_machines.py` always declares 2 outputs. [ONNX SVMClassifier spec](https://onnx.ai/onnx/operators/onnx_aionnxml_SVMClassifier.html).

**Key code location:** `ferroml-core/src/onnx/svm.rs:331-625` (SVC impl)

**After fix:** ORT result will have `result[0]` = labels (int64), `result[1]` = scores (float32). Compare `result[0]` to `model.predict()`.

---

### Bug 3: BernoulliNB missing neg_prob term (fixes 1 model)

**Affected model:** BernoulliNB

**Root cause:** `ferroml-core/src/onnx/naive_bayes.rs:98-178` uses the same formula as MultinomialNB (`X @ feature_log_prob.T + class_log_prior`). BernoulliNB requires a different formula that accounts for absent features:

```
neg_prob = log(1 - exp(feature_log_prob))
jll = X @ (feature_log_prob - neg_prob).T + sum(neg_prob, axis=1) + class_log_prior
```

Without the `neg_prob` correction, predictions are wrong whenever features have value 0.

**Fix:**
1. Add ONNX nodes to compute `neg_prob`:
   - `Exp(feature_log_prob_initializer)` → intermediate
   - `Sub(constant_1.000000001, exp_result)` → `one_minus_prob` (use 1.000000001 to avoid log(0))
   - `Log(one_minus_prob)` → `neg_prob`
2. Compute the difference matrix: `Sub(feature_log_prob, neg_prob)` → `diff_matrix`
3. Compute sum of neg_prob over features: `ReduceSum(neg_prob, axes=[0])` → `sum_neg_prob` (shape [1, n_classes])
4. The main computation becomes: `MatMul(X, diff_matrix)` → `Add(_, sum_neg_prob)` → `Add(_, class_log_prior)` → `ArgMax` → `Cast`

**Reference:** sklearn-onnx's `naive_bayes.py` converter (`_joint_log_likelihood_bernoulli` function). sklearn source: `sklearn/naive_bayes.py` BernoulliNB._joint_log_likelihood.

**Key code location:** `ferroml-core/src/onnx/naive_bayes.rs:98-178`

**Also consider:** If FerroML's BernoulliNB has a `binarize` threshold (like sklearn defaults to 0.0), the ONNX graph needs a `Greater` + `Cast(float)` at the front to binarize input features.

---

### Bug 4: HistGradientBoosting bin-threshold edge cases (fixes 2 models)

**Affected models:** HistGradientBoostingRegressor, HistGradientBoostingClassifier

**Root cause:** `ferroml-core/src/onnx/hist_boosting.rs:68-111` (`add_hist_tree`) calls `bin_mapper.bin_threshold_to_real(feature_idx, bin)` to convert u8 bin indices to float thresholds. Some edge cases in the bin-to-threshold mapping produce incorrect routing for ~4-12% of samples.

**Symptoms:** 6/50 regressor samples mismatch (max_diff=1.24), 2/50 classifier samples mismatch. The correlation is still very high (r²=0.995) suggesting most tree structure is correct but a few nodes route wrongly.

**Investigation needed:**
1. Examine `bin_threshold_to_real()` — likely in `ferroml-core/src/models/hist_boosting.rs` or similar
2. Compare the thresholds used by the native Rust predict vs the ONNX-exported thresholds
3. The issue may be off-by-one in bin boundaries or incorrect handling of the "missing bin" case
4. Check if the bin mapper's thresholds are inclusive vs exclusive (ONNX TreeEnsemble uses `BRANCH_LEQ` mode = less-than-or-equal)

**Key code locations:**
- `ferroml-core/src/onnx/hist_boosting.rs:68-111` (ONNX tree building)
- `ferroml-core/src/models/hist_boosting.rs` (native model, bin mapper)

---

### Bug 5: AdaBoostRegressor weighted median (1 model, may be unfixable)

**Affected model:** AdaBoostRegressor

**Root cause:** AdaBoost.R2 uses weighted median for prediction, which is NOT expressible in standard ONNX operators. The current export uses weighted-sum approximation (r²=0.995 but max_diff=0.82).

**Options:**
1. **Accept approximation** — document that AdaBoostRegressor ONNX export is an approximation, keep xfail or loosen tolerance
2. **Use custom operator** — define a custom ONNX op for weighted median (breaks portability)
3. **Use weighted mean instead** — simpler approximation, may be closer for some distributions

**Recommendation:** Accept the approximation, document it clearly, change from xfail to a test with large tolerance (atol=1.0 or rtol=0.5) and add a docstring noting the limitation.

---

## Recommended Fix Order (by impact)

1. **Bug 1 — TreeEnsembleClassifier** (fixes 4 models, ~30 min, low risk)
   - Change `create_proba_output()` to use `tensor(float) [N, n_classes]` instead of `sequence<map<...>>`

2. **Bug 2 — SVMClassifier** (fixes 1 model, ~30 min, low risk)
   - Add second output to SVC ONNX export

3. **Bug 3 — BernoulliNB** (fixes 1 model, ~45 min, medium risk)
   - Add neg_prob computation to ONNX graph

4. **Bug 4 — HistGradientBoosting** (fixes 2 models, ~1-2 hr, needs investigation)
   - Debug bin-threshold mapping edge cases

5. **Bug 5 — AdaBoostRegressor** (1 model, ~10 min, accept approximation)
   - Document limitation, adjust test tolerance

## Verification Commands

```bash
# Build Python bindings
source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml

# Run round-trip tests
cd ferroml-python && python -m pytest tests/test_onnx_roundtrip.py -v

# Run validation script (quick compatibility matrix)
cd /home/tlupo/ferroml && python scripts/validate_onnx_exports.py

# Run all ONNX tests (export + roundtrip)
cd ferroml-python && python -m pytest tests/test_onnx_export.py tests/test_onnx_roundtrip.py -v

# Run Rust ONNX tests
cd /home/tlupo/ferroml && cargo test --package ferroml-core onnx -- --nocapture
```

## Action Items & Next Steps

Priority order:
1. [ ] Fix Bug 1 (TreeEnsembleClassifier output type) — `tree.rs:404-436`
2. [ ] Fix Bug 2 (SVMClassifier 2 outputs) — `svm.rs:552-614`
3. [ ] Fix Bug 3 (BernoulliNB neg_prob) — `naive_bayes.rs:98-178`
4. [ ] Investigate + fix Bug 4 (HistGBRT bin thresholds) — `hist_boosting.rs:68-111`
5. [ ] Resolve Bug 5 (AdaBoostRegressor approximation) — adjust test tolerance
6. [ ] Remove xfail markers from fixed tests in `test_onnx_roundtrip.py`
7. [ ] Update `scripts/validate_onnx_exports.py` to remove `known_issue` for fixed models
8. [ ] Proceed to S.9 (MultiOutput ONNX) and S.10 (documentation)

## Other Notes

- onnxruntime 1.24.3 was installed in `.venv` this session
- The `cargo fmt --all` pre-commit hook will run — always format before committing
- GradientBoosting classifiers (non-hist) work correctly because they use TreeEnsembleRegressor + Sigmoid/ArgMax (not TreeEnsembleClassifier)
- LogisticRegression ONNX exports sigmoid probabilities (not labels) — the round-trip test uses threshold comparison
- All 4 preprocessing scalers pass round-trip perfectly (max_diff ~1e-7)
