# Tree-Ensemble Structural Parity — Design Pass

**Status:** design only; no implementation in this pass.
**Branch:** `pilot/ox-alpha`, HEAD `0e44d1a` at time of writing.
**Anchors:** `PARITY-CONTRACT.md` (the four-layer contract and the FM-003 pattern),
the Naive-Bayes slice (`ferroml-python/tests/parity/certify_naive_bayes.py`,
`test_naive_bayes.py`, `docs/test-runs/parity-certification-naive-bayes.json`),
and the audit history (`AUDIT-OX.md`, `FIXES-OX.md`, `VERIFY-OX.md`). Recorded
decisions there are inherited, not re-derived.

Every claim about Rust capability below cites file:line verified at HEAD `0e44d1a`.

## 1. The problem

Tree ensembles are the largest remaining parity family. The NB slice established the
slice pattern: a certification harness measures discrepancies over a preregistered
seeds x configs matrix; tolerances are derived from the measurements
(`max(10 x observed max diff, 1e-12)`); a pytest gate enforces them; the artifact
records environment, matrix, policy, channels, cells, and notes.

Trees break the NB pattern in one fundamental way. NB has no training randomness:
given identical data, both implementations compute closed-form statistics, so
near-exact parity is achievable and array gates carry real weight. Trees are trained
by randomized recursive partitioning, and ferroml's random streams are not sklearn's:

- ferroml's per-split feature subsampling uses an inline LCG,
  `rng.wrapping_mul(1103515245).wrapping_add(12345)`
  (ferroml-core/src/models/tree.rs:940);
- forest bootstrap samples and per-tree seeds come from the rand crate's
  `StdRng` (ferroml-core/src/models/forest.rs:469,
  ferroml-core/src/models/forest.rs:677);
- sklearn uses MT19937 via `RandomState`.

Different RNG streams mean different feature orders at ties and different bootstrap
samples, so two independently fitted forests diverge tree-by-tree from the first
split. **Fitted-structure parity for ensembles is therefore impossible by
construction**, exactly as PARITY-CONTRACT.md records for independently trained
ensembles ("do not promise a universal score epsilon"). The design question is which
channels remain that still deserve the word *structural*.

## 2. Capability inventory — what the Rust side can expose

The structural surface exists in ferroml-core but is mostly unreachable from Python.
This gap defines the slice's prerequisite work.

**Exposed today (Python):** only `get_depth` and `get_n_leaves`
(ferroml-python/src/trees.rs:236, ferroml-python/src/trees.rs:243). Two scalars per
tree. Not enough for any structural channel worth the name.

**Exists in core, not exposed:** `TreeStructure` (ferroml-core/src/models/tree.rs:247)
carries every `TreeNode` (ferroml-core/src/models/tree.rs:174): feature index,
threshold, impurity, sample counts, leaf values with class counts, children, depth.
`decision_path` (ferroml-core/src/models/tree.rs:343) walks a sample to its leaf and
returns the node-id path. Forests expose their estimators
(`estimators()`, ferroml-core/src/models/forest.rs:360 area) so per-tree structure of
a whole forest is reachable once bindings exist.

**Inference internals relevant to gating:**

- `predict_proba` normalizes the raw leaf value vector to probabilities
  (ferroml-core/src/models/tree.rs:582); forest `predict_proba` averages aligned
  per-tree probabilities (ferroml-core/src/models/forest.rs:377).
- Split thresholds are midpoints between adjacent distinct feature values
  (ferroml-core/src/models/tree.rs:1016).
- Classifier split search breaks gain ties by strict `>` over sorted features
  (ferroml-core/src/models/tree.rs:1041); the regressor sweep instead requires
  `gain > best_gain + 1e-10` (ferroml-core/src/models/tree.rs:1853) — an internal
  inconsistency this design flags but does not change.
- Growth stops early on `depth >= 1000` and `impurity < 1e-10`
  (ferroml-core/src/models/tree.rs:707, ferroml-core/src/models/tree.rs:708);
  sklearn's stopping semantics differ in detail (open question Q1).
- `min_samples_leaf` is enforced *after* the split search, by rejecting the winning
  split, unlike sklearn's integrated constraint (design observation; see Q2).
- Gradient boosting draws stochastic subsample indices from `StdRng`
  (ferroml-core/src/models/boosting.rs:355, ferroml-core/src/models/boosting.rs:643);
  HistGB bins features on quantile edges computed by its own `BinMapper`
  (ferroml-core/src/models/hist_boosting.rs:238,
  ferroml-core/src/models/hist_boosting.rs:285), whose edge placement differs from
  sklearn's `_BinMapper`.
- ExtraTrees delegates to the same tree builder with
  `SplitStrategy::Random` (ferroml-core/src/models/extra_trees.rs:417).

## 3. Channel classification by family

### 3.1 Decision trees (DecisionTreeClassifier/Regressor)

With `max_features=None` both implementations run deterministic CART on the same
data, same midpoint thresholds (sklearn also splits at midpoints), same criterion
formulas. Structure comparison becomes plausible — but only on data engineered so
that no split is decided by a near-tie or by the epsilon details above (ferroml's
`1e-10` tolerances in equal-value checks and tie-breaks vs sklearn's exact
comparisons can flip a split even without randomness). Therefore:

- **Channel `dt_structure` (layer 1, conditional).** For each certified config, fit
  both trees and compare flattened structure: per-node `(feature_index, threshold,
  depth)` sequences and leaf count. Applied only to configs marked
  tie-free in the matrix design (see 5); configs that fail a pre-screen for
  near-tied gains fall back to agreement channels only. This pre-screen is itself
  recorded in the artifact, so the set of structure-gated configs is evidence, not
  taste.
- **Channels `dt_proba`, `dt_predict` (layers 2-3).** Two-sided proba gates with
  derived tolerances plus label agreement, unconditional across all configs.

### 3.2 Forests and ExtraTrees (all four estimators)

Bootstrap resampling differs (different RNG), so per-tree structures differ.
Structural channels compare **within ferroml**, against sklearn only through
aggregates:

- **Channel `rf_self_consistency` (layers 1+3).** Using the new bindings: for each
  tree in the fitted ferroml forest, `decision_path` output must agree with the
  model's own `predict` routing, and the average of per-tree probas must equal the
  forest's `predict_proba` to within 1e-12 absolute (recomputed independently in the
  test from exposed per-tree predictions). This gates ferroml's aggregation logic —
  averaging, class alignment — which is where real bugs live, without pretending two
  RNG streams produce the same forest.
- **Channels `rf_proba`, `rf_agreement` (layers 2-3).** Two-sided proba gate with
  derived tolerance; label agreement on fit and holdout data. Certification mandates
  `n_jobs=Some(1)`: parallel execution is documented non-deterministic even under a
  fixed seed (ferroml-core/src/models/forest.rs:125).
- ExtraTrees inherits the same channels; its random-threshold strategy makes
  `dt_structure`'s conditional structural gate unavailable, which the matrix records
  explicitly rather than silently omitting.

### 3.3 Gradient boosting family (GradientBoosting*, HistGradientBoosting*, AdaBoost*)

PARITY-CONTRACT.md's survey found no sklearn-equivalence tolerance for gradient
boosting anywhere in the ecosystem, because XGBoost/LightGBM/sklearn implement
different training procedures despite shared parameter names. Ferroml's own internals
confirm divergence (own loss schedules, own bin mapper). Declared honestly:

- **No structural or array channel vs sklearn for GBM/HGB fitted models.**
  Channels are layer-3 only: label/probability agreement with derived-agreement
  thresholds, plus layer-1 invariants (shapes, finiteness, monotone staged
  improvement recorded via `training_history`). AdaBoost additionally exposes
  `estimator_weights`, gated for internal consistency (non-negative, sums sane) —
  self-consistency again, not cross-implementation structure.
- HGB binning gets a layer-1 invariant: every training value must land in a recorded
  bin edge interval, and bin counts must match `n_bins_per_feature`. This catches
  binning regressions without claiming sklearn equivalence.

## 4. Tolerance policy

Inherited unchanged from the NB slice and applied per channel:

```
derived_epsilon(channel) = max(10 x max observed |diff|, 1e-12)
rtol(channel)            = derived_epsilon / max_ref_abs
agreement_threshold      = max(0, min observed agreement - 0.01)
```

Two-sided form everywhere: `|mine - ref| <= atol + rtol*|ref|`. No hand-picked
numbers. The gate recomputes every derived value from the artifact and fails if they
disagree (policy self-consistency tests, as in NB). Negative controls prove the
gates bite: tightening atol 1000x must violate on real diffs.

## 5. Tail coverage — the tree analog of the NB escape (note 1)

The NB tail was deep log-proba cells where a documented floor made one-sided
checking the only honest option, and the wave-two finding was that replacing tail
values with -20 produced zero failures — the gate could not see wrongness in the
region it had declared ungateable. Trees have two analogous regions:

1. **Near-tie decision margins.** Where a sample's top-2 class probabilities are
   nearly equal, an infinitesimal numeric difference flips the argmax while every
   array gate passes comfortably. A pure label-agreement gate would catch flips only
   statistically, and a mutation that biases decisions near ties could hide inside
   the agreement threshold's slack.
2. **Deep/rare leaves.** Leaves holding few samples have quantized, high-variance
   leaf values; proba differences concentrate there, and a tolerance derived over
   the whole matrix is dominated by them — the opposite failure, where the tail
   inflates epsilon for everyone.

Design response:

- **Margin channel `*_margin_tail` (new, preregistered).** During certification,
  record per estimator/config/seed the distribution of `p_top1 - p_top2` over a
  holdout set. Preregister a margin threshold M = the 1st percentile observed for
  sklearn, floored at 1e-9, before looking at any ferroml results. The gate then
  asserts, per cell: (a) outside the margin tail, label agreement >= the derived
  threshold computed over in-margin samples only; (b) inside the margin tail, both
  implementations emit valid distributions (finite, simplex), and every disagreement
  is counted and reported in the artifact. Disagreement rate inside the tail gets its
  own derived bound: max observed rate x10, floor 1e-12 — a rate gate, not per-sample.
- **Rare-leaf channel `*_leaf_value` (conditional structural).** For single trees on
  tie-free configs, leaf `value` vectors are compared elementwise (they are raw class
  counts, integer-valued, comparable far more tightly than probas). For ensembles,
  leaf-depth and leaf-size histograms are recorded per tree as layer-1 metadata;
  gross divergence (e.g., all trees collapsing to stumps) fails a shape-level bound
  derived from the matrix.
- **Honest one-sidedness statement, same wording discipline as the NB artifact.**
  The margin channel bounds how often ferroml disagrees inside the tail, measured on
  the certified matrix; it does NOT certify that any particular tail decision is
  correct, and it cannot detect a correlated bias that flips ferroml and sklearn
  identically (both wrong the same way is invisible to any cross-implementation
  gate). This residual risk goes verbatim into the artifact's `notes.tail_coverage`.
  That is the demonstrated-escape lesson applied up front: when the sixth NB
  mutation escaped, the correct response was to narrow the gate's stated claim, not
  widen the gate. If a tree mutation demonstrates an escape here, the fix is the
  same: restate what the gate actually proves.

## 6. Stale comments — documentation generated from the gate code (note 2)

The NB slice found a comment asserting a check that did not exist. Rule for this
slice, designed in from the start:

- Every factual claim in the gate module's docstring (channel names, thresholds,
  env fields, artifact path) must be *generated or validated from the loaded
  artifact and code objects at import time*. Concretely: the module builds its
  channel list by iterating `CERT["channels"]`; a test enumerates the gate functions'
  consumed parameters and asserts the sets match — no orphan channels (declared in
  the artifact but ungated) and no phantom claims (documented but absent). Docstring
  text is assembled from the same dict, so a doc/code drift is a failing import, not
  a stale comment.
- The design doc itself carries no numbers that the certification harness computes;
  it names formulas, not values.

## 7. Environment scoping from day one (note 3)

The Layer-4 skip compares the WHOLE certified environment block, all seven fields
(python, numpy, sklearn, scipy, ferroml, platform, dtype) — implemented as a dict
equality against the live environment built from the same keys, exactly as the
current NB gate does post-fix. The tree gate copies that mechanism in its first
commit; a mismatch skips with the differing fields named. Additionally, a guard test
asserts the live-env construction covers every key present in the artifact's
environment block, so a future eighth field cannot silently drop out of the
comparison. No field is optional from day one.

## 8. Certification protocol

- **Matrix:** seeds x configs, following the NB convention: seeds `[42, 7, 123,
  2026]`; per-family configs spanning small/large n, low/high dimension, 2-5
  classes, engineered tie-free variants for the conditional structural channels.
  Regression families get their own configs (targets from fixed-seed generators).
  float64 throughout.
- **Procedure:** per cell, fit both implementations on identical data, measure every
  channel, write per-cell rows into the artifact. Derived tolerances computed after
  the full sweep, never during.
- **Artifact:** `docs/test-runs/parity-certification-trees.json` with the NB
  artifact's schema: environment (7 fields), matrix, policy, channels, cells, notes
  (including `tail_coverage`). Gate reads only this file.
- **Trivial probes allowed now:** confirming sklearn exposes `tree_.feature`,
  `tree_.threshold`, `children_left/right`, `value` arrays needed for `dt_structure`
  comparison. No comparison runs beyond that in this pass.

## 9. Mutation plan (note 4)

Preregistered mutations, each naming its expected catcher. Run at gate-implementation
time, results recorded in the artifact's notes.

| # | Mutation (in a scratch copy of the model wrapper, never shipped) | Expected catcher |
|---|---|---|
| M1 | Perturb one leaf value by +1 in a single tree | `rf_self_consistency` (averaged proba no longer equals recomputed aggregate) and `rf_proba` |
| M2 | Change traversal comparison `<=` to `<` at internal nodes | `rf_self_consistency` (decision_path disagrees with predict routing); `dt_structure` on tie-free configs |
| M3 | Return lower endpoint instead of midpoint as threshold | `dt_structure` threshold comparison; `dt_proba` |
| M4 | Drop one env field from the skip check | day-one env-scope guard test (section 7) |
| M5 | Widen tolerance floor `1e-12` -> `1e-6` in the gate | policy self-consistency test recomputing `max(10x, 1e-12)` |
| M6 | Bias predict toward class 0 only when `p_top1 - p_top2 < 1e-7` | expected catch: PARTIAL — caught only if the flipped-rate exceeds the tail rate bound; otherwise escapes |

Expected outcome: five of six caught cleanly, mirroring the NB experience. **What a
demonstrated escape means here is defined in advance:** an escaping mutation narrows
the gate's published claim (as M6 already does — the gate claims bounded tail
disagreement rate, not per-sample equivalence) and adds the escape case to
`notes.tail_coverage`; it does not trigger tolerance inflation. The mutation plan is
part of the design precisely so this accounting happens against preregistered
expectations rather than after the fact.

## 10. What CANNOT be gated structurally — honest limits

- **Fitted ensemble structure vs sklearn, ever.** Different RNG streams (LCG at
  tree.rs:940, StdRng bootstrap at forest.rs:469/677, vs MT19937) make per-tree
  divergence certain from the first split. Any gate claiming per-tree structural
  equality for forests would be theater.
- **The gradient boosting family vs sklearn, structurally or with tight numeric
  tolerance.** Different training procedures; the ecosystem-wide absence of any
  equivalence tolerance confirms it. Agreement-only, as section 3.3 declares.
- **HGB bin boundaries vs sklearn's.** Independent quantile-edge computations;
  compared only by ferroml-internal invariants.
- **Any channel under `n_jobs != 1`.** Documented thread-scheduling
  non-determinism (forest.rs:125). Gates pin `n_jobs=Some(1)`; behavior under
  parallelism is outside every claim.
- **Per-sample correctness inside the margin tail.** Covered only as a bounded rate
  (section 5). Residual risk stated in the artifact.

Residual risks accepted: correlated errors invisible to cross-implementation gates;
epsilon-detail split flips on near-tie data (mitigated by the tie-free pre-screen,
not eliminated); sklearn-version drift invalidating derived tolerances (mitigated by
the full-environment skip, which turns drift into an explicit skip, not a false pass).

## 11. Prerequisite work and open questions

Implementation prerequisites (future slices, listed because the design depends on
them): Python bindings exposing `TreeStructure` node arrays and `decision_path` for
trees and forest estimators; per-tree proba access for the self-consistency channel.

Open questions (facts not settled by sources read in this pass):

- **Q1:** sklearn's exact growth-stopping semantics (impurity epsilon, depth cap)
  versus ferroml's `depth >= 1000` / `impurity < 1e-10` stops (tree.rs:707-708) —
  settle by reading sklearn's splitter source at implementation time; affects only
  how wide the tie-free pre-screen must be.
- **Q2:** whether sklearn's integrated `min_samples_leaf` handling versus ferroml's
  post-hoc rejection changes reachable tree shapes on tie-free configs — settle with
  the trivial probes in section 8.
- **Q3:** whether any worker transcript holds measured tree-vs-sklearn discrepancy
  magnitudes usable for pre-sizing the matrix — deliberately not consulted per
  brief; the certification run answers it empirically.
