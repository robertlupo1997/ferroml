# Tree-Ensemble Structural Parity — Design Pass

> ## PARKED — do not implement from this document yet
>
> Three adversarial review rounds converged the mechanics, and this revision passes its
> gate 26/26. It is parked anyway, on one unresolved question that no further revision
> round can settle:
>
> **The tie-free pre-screen is named twelve times in this document and defined nowhere.**
> It decides which configurations get structural gating at all — `dt_structure`,
> `dt_leaf_value`, `dt_proba`, and `rf_structure_det` are all conditional on it — and the
> document specifies no gain-gap formula, no numeric cutoff, no independent
> implementation, no fixed corpus, no preregistration timing, and no minimum surviving
> cell count. Q1 (section 11) defers its width to implementation time.
>
> An undefined screen is an escape hatch. An implementation can widen it until the
> configurations that would have failed are classified as near-tied, watch them fall back
> to agreement channels, and still call the programme structural parity. The artifact
> would not look wrong. That is the exact failure this programme exists to prevent.
>
> The screen's width is downstream of a decision about what the parity claim promises,
> which is a product call and not this document's to make. Written up with the options
> and their costs in **`project-ops/thoughts/shared/2026-08-24-ferroml-tie-free-prescreen-decision.md`**.
> Settle that first; the width follows from it.
>
> **Also outstanding**, from the round-3 review
> (`coord/inbox/2026-08-24-codex-ferroml-design-review-round3.md`), all mechanical and
> all unaddressed here:
>
> 1. Equality probes still escape when an **ancestor splits on the same feature** as the
>    target node. The round-3 fix handles non-repeated ancestors only, so section 3's
>    "guaranteed to arrive at N" is too strong and the M2 coverage claim is unsupported.
> 2. The "complete TreeNode representation" omits `impurity_decrease`, a public field at
>    `ferroml-core/src/models/tree.rs:198`. This document mentions it zero times while
>    claiming completeness — the same defect class as the round-2 environment-block
>    finding, in a second place.
> 3. No serialization channel, though `PARITY-CONTRACT.md:63` requires
>    "structure and serialized inference where applicable" for trees, and
>    `ferroml-core/src/models/tree.rs:66` derives `Serialize`/`Deserialize` — so it applies.
>
> What survived three rounds and is reusable regardless of the decision: the capability
> inventory (section 2), the preregistered tolerance ceilings (section 4), the full
> environment block (section 7), the mutation plan (section 9), and the honest limits
> (section 10).

**Status:** design only; no implementation in this pass. **PARKED — see the banner above.** **Round-3 revision:** revised
in place again after a second independent adversarial review from a different model
family — five new high findings, none a repeat of round 1 — of commit `b71a571`
(`coord/inbox/2026-08-24-codex-ferroml-design-review-round2.md`); see section 0B for
the per-finding disposition. **Round-2 revision:** against the first Codex review
(`coord/inbox/2026-08-24-codex-ferroml-design-review.md`) of commit `c22a05e`; see
section 0. Round-1 and round-2 dispositions are kept verbatim below.
**Branch:** `pilot/ox-alpha`, HEAD `c22a05e` at time of writing.
**Anchors:** `PARITY-CONTRACT.md` (the four-layer contract and the FM-003 pattern),
the Naive-Bayes slice (`ferroml-python/tests/parity/certify_naive_bayes.py`,
`test_naive_bayes.py`, `docs/test-runs/parity-certification-naive-bayes.json`),
and the audit history (`AUDIT-OX.md`, `FIXES-OX.md`, `VERIFY-OX.md`). Recorded
decisions there are inherited, not re-derived.

Every claim about Rust capability below cites file:line re-verified at HEAD
`b71a571` (the design passes added no code; these files are unchanged since
`0e44d1a`, where the round-1 citations were first verified).

## 0. Round-2 review disposition

The adversarial review returned four high findings. Each was re-derived from primary
sources before being accepted; none is folded in on trust.

1. **Derived tolerances with no ceiling — ACCEPTED (confirmed by arithmetic).**
   Round-1 section 4 derived every bound from ferroml's own measurements. An observed
   probability discrepancy of 0.5 becomes epsilon 5 on a quantity bounded in [0,1];
   zero label agreement becomes an agreement threshold of 0. Worse, the round-1
   policy dropped half of PARITY-CONTRACT.md's own formula (lines 68-76): the
   contract derives epsilon from the observed matrix discrepancy **plus a
   preregistered numerical margin**, scoped to a declared corpus. Round 1 kept the
   observed term and silently discarded the preregistered term. Section 4 restores
   it as hard ceilings and floors, calibrated only on sklearn-side noise.
2. **Incomplete certified environment block — ACCEPTED (confirmed against
   PARITY-CONTRACT.md:65-66).** Round-1 section 7 called seven incidental runtime
   fields "the WHOLE certified environment block" while the contract scopes the
   matrix to sklearn version, crate version, dtype, solver, seed, thread count, BLAS
   implementation, and CPU/GPU backend. Five of those eight were absent. This is the
   same defect class as wave-two note 3, which round 1 claimed to address. Section 7
   now enumerates every contract dimension with an explicit artifact field, records
   N/A explicitly rather than omitting, and puts seed and thread count into each
   certified cell.
3. **M2 not catchable by the proposed gates — ACCEPTED (verified in source).**
   `decision_path` and predict routing share the same traversal code semantics:
   `x[feature] <= threshold` appears at ferroml-core/src/models/tree.rs:357
   (decision_path), ferroml-core/src/models/tree.rs:646 and
   ferroml-core/src/models/tree.rs:1557 (predict routing). A mutation flipping `<=`
   to `<` moves both identically, so `decision_path` versus `predict` is not an
   independent oracle; and `dt_structure` compares fitted arrays, which such a
   mutation never touches. Round-1's claim that five of six mutations are caught was
   unsupported for M2. Section 9 replaces the M2 catcher with an independently
   implemented traversal reconstruction plus exact-equality probes.
4. **Regressor channels defined in classifier-only terms — ACCEPTED (verified in
   source).** Round-1 section 3.2 claimed coverage of all four forest estimators but
   named only per-tree probabilities, `predict_proba`, and label agreement. The
   aggregation mechanics do generalize (the forest regressor averages per-tree
   predictions exactly as the classifier averages aligned probas,
   ferroml-core/src/models/forest.rs:1329-1367), but generalization asserted is not
   a channel designed. Section 3.2 now defines regressor channels explicitly and
   section 3.4 adds the estimator-by-channel coverage matrix the gate validates.

No finding is rejected. Where the review's mechanism needed correction, the
correction is stated inline (for example, the aggregation channel survives finding 4
on new evidence, but only because ferroml-core/src/models/forest.rs:1329-1367 shows
value-averaging mirroring proba-averaging).

## 0B. Round-3 review disposition

The second adversarial pass returned five high findings, all new relative to round 1.
Each was re-derived from primary sources before being folded in; none is accepted on
trust, and one mechanism correction is stated inline.

1. **Deterministic forests excluded from structural parity — ACCEPTED, with a
   configuration correction.** The round-2 claim "fitted ensemble structure vs
   sklearn, ever" (section 10) was false as a blanket rule. With bootstrap sampling
   disabled and all features considered at every split, every forest member is a
   deterministic CART tree fitted on the identical training set and IS structurally
   comparable to sklearn, tree by tree. One correction to the review's mechanism: in
   ferroml, `max_features: None` means the *default* `MaxFeatures::Sqrt`
   (ferroml-core/src/models/forest.rs:201), not "all features" — a `None`-configured
   forest is still randomized. The genuinely deterministic configuration is
   `with_bootstrap(false)` (the fit path then uses all samples for every tree,
   ferroml-core/src/models/forest.rs:689) combined with `MaxFeatures::All`, whose
   `compute` returns `n_features` (ferroml-core/src/models/forest.rs:175). The
   per-tree builder passes that value down (ferroml-core/src/models/forest.rs:724),
   and the split search's LCG feature-selection branch then selects a de-duplicated,
   sorted set equal to `(0..n_features)` — bit-identical to no subsampling
   (ferroml-core/src/models/tree.rs:934-948). The per-tree seeds drawn from `StdRng`
   (ferroml-core/src/models/forest.rs:677) become inert under this configuration:
   they feed only bootstrap generation and feature subsampling, neither of which can
   alter the result. On the sklearn side the same configuration is spelled
   `bootstrap=False, max_features=n_features`; sklearn removed `max_features=None`.
   Section 3.2 adds the deterministic-forest slice, the coverage matrix gates it,
   and section 10 narrows the impossibility claim to randomized configurations.

2. **Incomplete `dt_structure` — ACCEPTED.** Comparing `(feature_index, threshold,
   depth)` sequences plus leaf count verifies a projection of the tree, not the
   tree: a corrupted child link, wrong node count, wrong impurity or sample-count
   metadata, or reordered class values can all survive it while every compared field
   agrees. Section 3.1 now compares the complete exported representations — child
   links, node count and node ids, impurity, sample and weighted-sample counts,
   value arrays with their shapes, depth, leaf set, and class ordering via
   `classes_` — exact checks for discrete fields, preregistered tolerances for
   floats (section 4).

3. **Threshold-equality probes do not guarantee the mutated node is exercised —
   ACCEPTED.** A probe whose other feature values are arbitrary can be routed away
   at an ancestor sharing the split feature and never reach the mutated comparison,
   so the M2 catcher could pass while M2 escapes. Worse, the assertion as written
   was type-confused: a reconstructed leaf id cannot equal a `predict` output or a
   probability vector. Section 3.1 rebuilds each probe from a training sample known
   to reach the target node, asserts reachability before asserting behavior, fails
   certification for unprobeable nodes instead of skipping them, and separates the
   assertions: reconstructed leaf id against `decision_path`'s terminal node, then
   the exported leaf value mapped separately onto classifier probabilities/classes
   or regression output.

4. **Regression tail defined with classifier-only assertions — ACCEPTED.** The
   round-2 margin channel told regressors to reuse a procedure requiring label
   agreement outside the tail and simplex probability distributions inside it;
   neither quantity exists for a regressor, so the channel was uninstantiable for
   half its declared coverage. Section 5 now defines a separate regression-tail
   channel in regression terms: tail selected by sklearn residual magnitude over a
   preregistered percentile during sklearn-only calibration, normalized per-sample
   error statistic, finite-output checks, and preregistered rate caps. The coverage
   matrix no longer applies the margin tail to regressor rows.

5. **Coverage promised without prerequisite APIs — ACCEPTED, split by family.**
   GradientBoosting and AdaBoost already expose their internal estimators
   (`estimators()` returning `&[DecisionTreeRegressor]`,
   ferroml-core/src/models/boosting.rs:512 and
   ferroml-core/src/models/boosting.rs:1114; `&[DecisionTreeClassifier]`,
   ferroml-core/src/models/adaboost.rs:128 and
   ferroml-core/src/models/adaboost.rs:437), so for those families the defect is in
   this document's scope statement, not the codebase: section 11 extends the binding
   work to those accessors and the matrix marks the affected cells as requiring it.
   HGB is different in kind: its trees are `HistTree`s whose nodes split on a u8 bin
   index plus a `missing_go_left` flag over quantile-binned features with
   ferroml-private edges (ferroml-core/src/models/hist_boosting.rs:518,
   ferroml-core/src/models/hist_boosting.rs:463-477), so traversal reconstruction
   over raw-feature node arrays is not implementable there and cross-implementation
   traversal parity is meaningless against a differently-binned reference. The HGB
   traversal cell becomes N/A and the residual risk is stated in section 10 rather
   than silently dropped. New guard rule (section 3.4): no coverage cell without a
   concrete executable channel survives — every cell must name a concrete executable
   channel or an N/A with a reason tied to a stated representation limit.

## 1. The problem

Tree ensembles are the largest remaining parity family. The NB slice established the
slice pattern: a certification harness measures discrepancies over a preregistered
seeds x configs matrix; tolerances are derived from the measurements; a pytest gate
enforces them; the artifact records environment, matrix, policy, channels, cells, and
notes. Trees inherit the pattern **including its preregistration obligation**
(PARITY-CONTRACT.md:68-76), which round 1 under-weighted and section 4 now makes
central.

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

- Traversal semantics: internal nodes route left on `x[feature] <= threshold`,
  identically in `decision_path` (ferroml-core/src/models/tree.rs:357) and in
  predict routing for classifiers and regressors
  (ferroml-core/src/models/tree.rs:646, ferroml-core/src/models/tree.rs:1557). These
  share one comparison convention, which is why neither can oracle the other for a
  traversal-semantics mutation (section 0, finding 3).
- Classifier `predict_proba` normalizes the raw leaf value vector to probabilities
  (ferroml-core/src/models/tree.rs:582); forest classifier `predict_proba` sums
  class-aligned per-tree probabilities then divides by the tree count
  (ferroml-core/src/models/forest.rs:377, alignment loop from
  ferroml-core/src/models/forest.rs:395).
- Forest regressor `predict` sums per-tree predictions then divides by the tree
  count (ferroml-core/src/models/forest.rs:1329-1367) — value-averaging that mirrors
  the proba-averaging, and the hook for the regressor self-consistency channel.
- Split thresholds are midpoints between adjacent distinct feature values
  (ferroml-core/src/models/tree.rs:1016). A stored threshold is an f64, so a probe
  sample can be constructed with a feature value *exactly equal* to a stored
  threshold — the input class on which `<=` versus `<` actually differs.
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

### 3.1 Single decision trees (DecisionTreeClassifier / DecisionTreeRegressor)

With `max_features=None` both implementations run deterministic CART on the same
data, same midpoint thresholds (sklearn also splits at midpoints), same criterion
formulas. Structure comparison becomes plausible — but only on data engineered so
that no split is decided by a near-tie or by the epsilon details above (ferroml's
`1e-10` tolerances in equal-value checks and tie-breaks vs sklearn's exact
comparisons can flip a split even without randomness). Therefore:

- **Channel `dt_structure` (layer 1, conditional).** For each certified config, fit
  both trees and compare the COMPLETE exported representations node-by-node, not a
  projection. Round-2 compared only `(feature_index, threshold, depth)` sequences
  and leaf count; a tree with a corrupted child link or wrong metadata passed that
  gate. The round-3 comparison covers every field both sides expose — ferroml's
  `TreeNode` carries id, feature index, threshold, impurity, sample count,
  weighted sample count, value vector, child ids, depth, and leaf flag
  (ferroml-core/src/models/tree.rs:174-199), and sklearn's `tree_` arrays carry the
  same information:
  - topology: `children_left`/`children_right` versus `left_child`/`right_child`
    for every node — child LINKS, not just the attribute sequences;
  - node count and node id sets (root must be id 0 on both sides);
  - discrete fields exact: feature index, depth, leaf flag, node count, class count;
  - float fields under preregistered tolerances (section 4): thresholds,
    impurity per node;
  - counts: `n_node_samples` and `weighted_n_node_samples` per node
    (`weighted_n_samples`, ferroml-core/src/models/tree.rs:186);
  - values: the full `value` array with its shape `(n_nodes, 1, k)` compared
    elementwise. Normalization differs by convention — sklearn stores raw class
    counts (or means), ferroml stores class probabilities (tree.rs:187) — so the
    test normalizes BOTH sides to probabilities before comparing, after asserting
    class ordering matches via each model's `classes_`;
  - derived aggregates recomputed from the arrays and asserted equal: leaf count,
    maximum depth, and per-node depth from root.
  Applied only to configs marked tie-free in the matrix design (see 5); configs
  that fail a pre-screen for near-tied gains fall back to agreement channels only.
  This pre-screen is itself recorded in the artifact, so the set of
  structure-gated configs is evidence, not taste. Scope statement: `dt_structure`
  verifies *fitted arrays*. It cannot see any inference-time defect — most
  importantly a traversal-semantics change (section 9, M2) — and is never cited as
  the catcher for one. The corrupted-topology negative control (M7, section 9)
  exists precisely because the round-2 projection could not see one.
- **Channel `traversal_reconstruction` (layers 1+3, unconditional where the node
  arrays are reachable; see the coverage matrix for per-family scoping).** The test
  implements traversal itself, in test code, from the exported node arrays: left on
  `x[f] <= t`. It never calls ferroml's routing to decide routing.

  Probe construction (round-3 revision — round-2 probes could be routed away at an
  ancestor and never exercise the target node): for each internal node N, first
  compute, with the TEST-OWNED traversal over fit and holdout samples, the set of
  samples whose path reaches N. A node with no reaching sample is UNPROBEABLE and
  fails certification outright — it is recorded and failed, never skipped. From a
  reaching sample s, build the equality probe by copying all of s's feature values
  and setting only the split feature of N to N's stored threshold bit-for-bit
  (constructible because thresholds are f64 midpoints,
  ferroml-core/src/models/tree.rs:1016). Because every ancestor constraint is
  satisfied by construction — s reached N through them, and no other feature moved —
  the probe is guaranteed to arrive at N, so flipping N's comparison from `<=` to
  `<` cannot escape detection. Reachability is re-asserted with the test-owned
  traversal before any behavioral assertion; a probe that fails its own reachability
  check fails the cell rather than producing a vacuous pass. Ordinary fit/holdout
  samples remain as probe class (b).

  Assertions per probe, now type-separated (round-2 conflated them): (a) the
  reconstructed leaf id must equal the terminal node id of ferroml's
  `decision_path`; (b) SEPARATELY, the exported leaf value must map onto the model's
  outputs — classifier: normalized leaf value vector equals `predict_proba` output
  and its argmax equals `predict`, under the section-4 ceilings; regressor: leaf
  mean equals `predict` output to within 1e-12. This is the M2 catcher.
- **Channels `dt_proba`, `dt_predict` (layers 2-3, classifier).** Two-sided proba
  gates with derived tolerances under preregistered ceilings (section 4), plus label
  agreement, unconditional across all configs.
- **Channel `dt_leaf_value` (conditional structural).** On tie-free configs, single-tree
  leaf `value` vectors are compared elementwise against sklearn's `tree_.value` —
  raw counts (classifier) or raw means (regressor), comparable far more tightly than
  probabilities. The regressor variant is the regression analog of the round-1
  leaf-value channel, now defined instead of implied.

### 3.2 Forests and ExtraTrees (RandomForest*, ExtraTrees* — classifier AND regressor)

Bootstrap resampling differs (different RNG), so per-tree structures differ
**for randomized configurations**. Round-3 correction: that is not a blanket rule.
Structural channels compare **within ferroml** for randomized configs, against
sklearn only through aggregates — and structurally against sklearn for the
deterministic slice below. Round-1 defined the randomized channels with classifier
vocabulary only; both variants are now explicit.

The deterministic-forest slice (round 3). With `with_bootstrap(false)` and
`MaxFeatures::All`, every member of a ferroml forest is a CART tree on the full,
identical training set with no random influence: bootstrap generation is bypassed
(ferroml-core/src/models/forest.rs:689), `MaxFeatures::All.compute` returns
`n_features` (ferroml-core/src/models/forest.rs:175), and the LCG feature-selection
branch then yields all features regardless of stream state
(ferroml-core/src/models/tree.rs:934-948). The same configuration exists in sklearn
as `bootstrap=False, max_features=n_features`. On engineered tie-free configs
(the same pre-screen as `dt_structure`), this makes every estimator tree
structurally comparable to its sklearn counterpart.

- **Channel `rf_structure_det` (layer 1, conditional on the tie-free pre-screen).
  For each certified deterministic config, compare EVERY estimator tree of the
  ferroml forest against its indexed sklearn counterpart using the full
  representation comparison defined for `dt_structure` (topology/child links, node
  count and ids, impurity, sample and weighted-sample counts, value arrays and
  shapes, classes ordering, depth, leaf set) under the section-4 ceilings. A single
  mismatched node in any member fails the cell. Scope limits, stated so nobody
  over-reads the certification: this slice covers ONLY `bootstrap=False,
  max_features=MaxFeatures::All` ferroml configurations matched to
  `bootstrap=False, max_features=n_features` sklearn configurations; ExtraTrees is
  excluded (random thresholds); it does not license structural claims for any
  randomized configuration, which remain covered by the self-consistency and
  aggregate channels below.

Classifier channels:

- **Channel `rf_self_consistency` (layers 1+3).** Using the new bindings: for each
  tree in the fitted ferroml forest, `decision_path` output must agree with the
  model's own `predict` routing, and the average of per-tree probas must equal the
  forest's `predict_proba` to within 1e-12 absolute (recomputed independently in the
  test from exposed per-tree predictions). This gates ferroml's aggregation logic —
  summing, class alignment (ferroml-core/src/models/forest.rs:395), division —
  which is where real bugs live, without pretending two RNG streams produce the same
  forest. Note the honest scope limit: this consistency pair shares traversal
  semantics (section 2), so it gates aggregation, not traversal; traversal is gated
  by `traversal_reconstruction`.
- **Channels `rf_proba`, `rf_agreement` (layers 2-3).** Two-sided proba gate with
  derived tolerance under a preregistered ceiling; label agreement on fit and holdout
  data with a preregistered floor. Certification mandates `n_jobs=Some(1)`:
  parallel execution is documented non-deterministic even under a fixed seed
  (ferroml-core/src/models/forest.rs:125).

Regressor channels (new in round 2; previously implied):

- **Channel `rf_self_consistency_reg` (layers 1+3).** Same shape as the classifier
  version over values instead of probas: the arithmetic mean of exposed per-tree
  predictions must equal the forest regressor `predict` output to within 1e-12
  absolute, recomputed in test code (ferroml-core/src/models/forest.rs:1329-1367 is
  the logic under test).
- **Channel `rf_pred` (layers 2-3).** Per-sample prediction-array discrepancy
  distribution between ferroml and sklearn on identical fit data: report mean |diff|,
  max |diff|, RMSE; the derived tolerance applies under the preregistered relative
  ceiling of section 4. Replaces "label agreement", which has no regressor meaning.
- **Channel `rf_score_floor` (layer 3).** On engineered easy targets (fixed-seed
  generators, preregistered signal-to-noise), ferroml's held-out R² must meet a
  preregistered floor (0.85) and residuals must be finite. This is a sanity floor,
  not an equivalence claim — it catches a regressor predicting garbage while saying
  nothing about matching sklearn closely.
- ExtraTrees inherits every channel in both variants; its random-threshold strategy
  makes `dt_structure` unavailable, which the coverage matrix records explicitly
  rather than silently omitting.

### 3.3 Gradient boosting family (GradientBoosting*, HistGradientBoosting*, AdaBoost*)

PARITY-CONTRACT.md's survey found no sklearn-equivalence tolerance for gradient
boosting anywhere in the ecosystem, because XGBoost/LightGBM/sklearn implement
different training procedures despite shared parameter names. Ferroml's own internals
confirm divergence (own loss schedules, own bin mapper). Declared honestly:

- **No structural or array channel vs sklearn for GBM/HGB fitted models.**
  Channels are layer-3 only, now split by task: classifiers get label/probability
  agreement with derived-agreement thresholds under preregistered floors; **regressors
  get prediction-discrepancy distributions (`gb_pred`) under the same ceiling
  machinery plus the `rf_score_floor` channel** — round-1 offered regressors nothing
  here. Both variants additionally get layer-1 invariants (shapes, finiteness,
  monotone staged improvement recorded via `training_history`). AdaBoost exposes
  `estimator_weights`, gated for internal consistency (non-negative, sums sane) —
  self-consistency again, not cross-implementation structure.
- HGB binning gets a layer-1 invariant: every training value must land in a recorded
  bin edge interval, and bin counts must match `n_bins_per_feature`. This catches
  binning regressions without claiming sklearn equivalence.

### 3.4 Estimator-by-channel coverage matrix

The gate validates this table against the artifact's declared channels: every
estimator row must carry at least one cross-implementation channel or an explicit
N/A with reason; every channel declared in the artifact must appear in at least one
row. A channel with no estimator, or an estimator with no channel and no stated
reason, fails the gate. **Round-3 rule, closing the round-3 finding 5 defect class:
no coverage cell without a concrete executable channel survives review — every cell
must name a concrete executable channel implementable from the specified
prerequisite work, or an N/A whose reason cites a representation limit (e.g., HGB's
`HistTree`). The guard rejects any coverage-matrix cell that promises a channel its
prerequisites cannot deliver; mandatory-but-unimplementable coverage is how gates
get silently dropped at implementation time.** `X` = applies, `S` = applies
conditionally (tie-free configs only).

| Channel | DTC | DTR | RFC | RFR | ETC | ETR | GBC | GBR | HGB* | ABC | ABR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dt_structure` | S | S | S via `rf_structure_det` only | S (same slice) | N/A (random thresholds) | N/A (same) | N/A (own training procedure) | N/A (same) | N/A (own HistTree representation) | N/A (stochastic sample reweighting) | N/A (same) |
| `rf_structure_det` | — | — | S (deterministic slice, §3.2; full-representation comparison per estimator tree) | S (same) | N/A (random thresholds) | N/A (same) | N/A (own training procedure) | N/A (same) | N/A (HistTree bins) | N/A (same) | N/A (same) |
| `traversal_reconstruction` | X | X | X (per estimator tree) | X | X | X | X (requires §11 binding over estimators(), boosting.rs:512/1114) | X (same) | N/A (HistTree routes on u8 bin indices over private bin edges — not implementable from specified work; see §10) | X (requires §11 binding over estimators(), adaboost.rs:128/437) | X (same) |
| `dt_proba` / `dt_predict` | X | dt_predict only (values) | — | — | — | — | — | — | — | — | — |
| `dt_leaf_value` | S | S | — | — | — | — | — | — | — | — | — |
| `clf_margin_tail` (§5) | X | — | X | — | X | — | X | — | X (classifier variant) | X | — |
| `reg_residual_tail` (§5) | — | X | — | X | — | X | — | X | X (regressor variant) | — | X |
| `rf_self_consistency` | — | — | X | — | X | — | — | — | — | — | — |
| `rf_self_consistency_reg` | — | — | — | X | — | X | — | — | — | — | — |
| `rf_proba` | — | — | X | — | X | — | — | — | — | — | — |
| `rf_agreement` | — | — | X | — | X | — | — | — | — | — | — |
| `rf_pred` | — | — | — | X | — | X | — | — | — | — | — |
| `rf_score_floor` | — | X | — | X | — | X | — | X | X | — | X |
| GBM layer-3 agreement (`gb_proba`/`gb_pred`) | — | — | — | — | — | — | X | X | X | X | X |
| HGB binning invariant | — | — | — | — | — | — | — | — | X | — | — |
| AdaBoost weight consistency | — | — | — | — | — | — | — | — | — | X | X |

Legend: DTC/DTR DecisionTree*, RFC/RFR RandomForest*, ETC/ETR ExtraTrees*,
GBC/GBR GradientBoosting*, HGB* HistGradientBoosting{Classifier,Regressor},
ABC/ABR AdaBoost*. Every cross-implementation channel (anything naming sklearn) is
subject to section 4's preregistered ceilings; within-ferroml consistency channels
use their own fixed tolerances (1e-12) because they assert arithmetic identity, not
cross-implementation agreement.

## 4. Tolerance policy — derived bounds under preregistered ceilings

Finding 1 accepted: deriving every bound from the implementation under test, with no
independent ceiling, certifies whatever that implementation does. The policy keeps
the NB derivation but fences it:

```
raw_epsilon(channel)     = max(10 x max observed |ferroml - sklearn|, 1e-12)
ceiling(channel)         = preregistered absolute maximum, committed to the artifact
                           BEFORE any ferroml cell is measured
derived_epsilon(channel) = raw_epsilon; the cell FAILS certification if
                           raw_epsilon > ceiling. The bound is never widened to fit.

agreement_threshold(cell) = max(floor(channel), min observed agreement - 0.01)
floor(channel)            = preregistered minimum; if min observed agreement - 0.01
                            falls below the floor, the cell FAILS. The threshold
                            never sinks to 0.

tail_rate_bound           = min(max(10 x observed tail disagreement rate, floor),
                                cap)
any computed bound >= cap FAILS as non-discriminating: a rate gate that tolerates
more than the cap proves nothing about the tail.
```

Two-sided form everywhere: `|mine - ref| <= atol + rtol*|ref|`. The gate recomputes
every derived value from the artifact and fails if they disagree (policy
self-consistency tests, as in NB).

**Ceiling calibration uses sklearn only.** Ceilings and floors are fixed from
sklearn-side experiments — sklearn rerun across the seed matrix gives a reference
self-noise scale — plus the design constants below. Ferroml measurements enter only
`raw_epsilon`; they can fail a cell but can never raise a pass mark. If ferroml
cannot meet a preregistered floor, the correct outcome is failed certification, not
a wider bound.

**Preregistered constants (design commitments, recorded before measurement; changing
one later follows PARITY-CONTRACT.md:75-76 — never after a production failure):**

| Channel | Quantity | Ceiling / floor | Rationale |
|---|---|---|---|
| `dt_proba` (tie-free) | abs proba diff | ceiling 1e-9 | equal structures imply equal integer class counts, hence rational probas agreeing to division rounding (~1e-16); 1e-9 leaves slack for summation order only |
| `dt_leaf_value` (tie-free) | per-leaf count/mean diff | ceiling 1e-9 (counts: exact 0) | integer counts must match exactly; regressor leaf means agree to float rounding on identical structures |
| `rf_proba` | mean abs proba diff | ceiling = 10 x sklearn seed-noise scale, hard cap 0.05 | different bootstraps give genuine forest-to-forest variation; the scale is measured from sklearn vs itself, and 0.05 mean abs error on [0,1] is already a materially different model |
| `rf_agreement` (outside margin tail) | label agreement rate | floor 0.90 | below 90% agreement outside near-ties means systematic disagreement, not RNG noise |
| `rf_pred` | RMSE / target std | ceiling = 10 x sklearn seed-noise scale, hard cap 0.25 | same construction as `rf_proba`, in regression units |
| `rf_score_floor` | held-out R² | floor 0.85 on engineered easy targets | engineered signal guarantees a correct learner exceeds this; a lower score means the implementation is broken, regardless of sklearn |
| `clf_margin_tail` / `reg_residual_tail` rate | tail disagreement / excess rate | cap 0.10, floor 1e-12 | a tail gate tolerating >10% disagreement certifies nothing |
| `gb_pred` / GBM layer-3 | same forms as `rf_pred` / agreement | same caps and floors | no structural claim is made, so the aggregate bounds match the forest aggregates |
| `rf_structure_det` | full node-array comparison (§3.1 field list) | discrete fields exact 0; float fields ceiling 1e-12 absolute | identical training set, deterministic split search on both sides; only summation-order rounding can differ |
| `reg_residual_tail` | mean per-sample error ratio in tail; share of tail samples with ferroml error > 2x sklearn error | ratio ceiling 1.5; excess-rate cap 0.10 (same cap logic as the margin tail) | inside a deep residual tail both models are at their worst; tolerating more than these bounds certifies nothing about the region that matters |
| `traversal_reconstruction` value mapping | leaf-value-to-output mapping error | ceiling 1e-12 absolute (self-consistency arithmetic) | asserts internal identity between exported arrays and model outputs, not cross-implementation agreement |

**Negative controls, upgraded.** Round 1's only control (tightening atol 1000x must
violate) shows the gate discriminates against a tightened copy of itself; it says
nothing about whether the gate detects real wrongness. Two controls now:

1. *Tightened-copy control* (kept): scaling every atol down 1000x must violate on
   the measured diffs — the gates are actually consuming the tolerances.
2. *Corrupted-model control* (new): every mutation in section 9 marked "caught" is
   applied to a scratch copy and must violate at least one preregistered gate. This
   is the direct answer to finding 1's objection: the gate must reject a deliberately
   degraded model against fixed standards, not merely reproduce its own baseline.

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
   inflates epsilon for everyone. Under the revised policy a tail-inflated
   `raw_epsilon` hits the preregistered ceiling and fails loudly instead of quietly
   licensing everyone (section 4).

Design response:

- **Margin channel `clf_margin_tail` (preregistered; CLASSIFIERS ONLY).** During
  certification, record per estimator/config/seed the distribution of
  `p_top1 - p_top2` over a holdout set. Preregister a margin threshold M = the 1st
  percentile observed for sklearn, floored at 1e-9, before looking at any ferroml
  results. The gate then asserts, per cell: (a) outside the margin tail, label
  agreement >= the agreement_threshold of section 4 — which has a preregistered
  floor and cannot sink to 0; (b) inside the margin tail, both implementations emit
  valid distributions (finite, simplex), and every disagreement is counted and
  reported in the artifact. Disagreement rate inside the tail gets the capped rate
  bound of section 4 (cap 0.10): a computed bound at or above the cap fails the cell
  as non-discriminating rather than passing vacuously.
- **Regression-tail channel `reg_residual_tail` (preregistered; round-3 addition —
  round 2 borrowed label agreement and simplex probabilities here, neither of which
  exists for a regressor).** Defined entirely in regression terms:
  - *Tail-selection corpus.* Per cell, compute sklearn's per-sample absolute
    residuals `|sklearn_pred_i - y_i|` on the holdout set during the sklearn-only
    calibration phase (section 4), BEFORE any ferroml result is read. The tail
    threshold tau = the preregistered 99th percentile of that distribution, floored
    at 1e-9. The tail corpus is `{i : |sklearn_pred_i - y_i| >= tau}`.
  - *Per-sample error statistic.* For tail sample i, the normalized per-sample
    error ratio `rho_i = |ferroml_pred_i - y_i| / max(|sklearn_pred_i - y_i|,
    1e-9)`. The gate asserts the mean of `rho` over the tail is <= 1.5 and that the
    share of tail samples with ferroml error exceeding 2x sklearn error is under
    the capped rate bound of section 4 (cap 0.10); a bound at or above the cap
    fails the cell as non-discriminating, exactly like the classifier margin tail.
  - *Finite-output checks.* Every ferroml prediction on every corpus (tail and
    non-tail) must be finite; NaN or inf anywhere fails the cell outright,
    independent of all tolerances.
  - *Outside the tail*, regressors are covered by the prediction-discrepancy
    channels already defined for them (`rf_pred`, `gb_pred`) — no label-agreement
    quantity is invoked anywhere in this channel.
- **Rare-leaf channel `*_leaf_value` (conditional structural).** For single trees on
  tie-free configs, leaf `value` vectors are compared elementwise under the section-4
  ceiling (counts exact; means to 1e-9). For ensembles, leaf-depth and leaf-size
  histograms are recorded per tree as layer-1 metadata; gross divergence (e.g., all
  trees collapsing to stumps) fails a shape-level bound derived from the sklearn-only
  noise scale.
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
  a stale comment. The coverage matrix of section 3.4 joins this mechanism: the
  artifact's channel list and the matrix rows must agree.
- The design doc itself carries no numbers that the certification harness computes;
  it names formulas and preregistered constants, not measured values.

## 7. Environment scoping — the full contract dimension set (note 3, round 2)

PARITY-CONTRACT.md:65-66 scopes every tolerance to eight dimensions: sklearn
version, crate version, dtype, solver, seed, thread count, BLAS implementation, and
CPU/GPU backend. Round 1 called seven unrelated runtime fields "the WHOLE certified
environment block"; five contract dimensions were absent. The corrected block maps
every contract dimension to an explicit artifact field:

| Contract dimension | Artifact field | Value / mechanism |
|---|---|---|
| sklearn version | `sklearn_version` | from `sklearn.__version__` |
| crate version | `crate_version` | ferroml core version string |
| dtype | `dtype` | `float64` throughout the matrix |
| solver | `solver` | literal `"n/a"` for tree families — no iterative solver participates; recorded explicitly, never omitted |
| seed | `seed` | per certified cell (matrix dimension, section 8) |
| thread count | `thread_count` | per certified cell; pinned `n_jobs=Some(1)` per ferroml-core/src/models/forest.rs:125 |
| BLAS implementation | `blas_implementation` | resolved at runtime from the numpy/sklearn BLAS config |
| CPU/GPU backend | `compute_backend` | `"cpu"` — ferroml is CPU-only; recorded explicitly so a GPU-sklearn environment cannot masquerade |

Diagnostic fields beyond the contract (python, numpy, scipy versions, platform) are
retained in the artifact but are no longer the completeness claim; completeness is
stated as "all eight PARITY-CONTRACT.md:65-66 dimensions present", checked by an
enumerated list in the guard test, not by counting fields.

Mechanics: the Layer-4 skip compares the full block (contract dimensions first,
diagnostic fields after) as dict equality against a live environment built from the
same keys, exactly as the current NB gate does post-fix. The tree gate copies that
mechanism in its first commit; a mismatch skips with the differing fields named.
Guard tests, day one: (a) the live-env construction covers every key present in the
artifact's environment block, so a future field cannot silently drop out; (b) the
field list contains all eight contract dimensions by name — an N/A is a value, not
an absence. No field is optional from day one.

## 8. Certification protocol

- **Matrix:** seeds x configs, following the NB convention: seeds `[42, 7, 123,
  2026]`; per-family configs spanning small/large n, low/high dimension, 2-5
  classes, engineered tie-free variants for the conditional structural channels, and
  **explicit regressor configs per family** (targets from fixed-seed generators with
  preregistered signal-to-noise) so the section-3.2/3.3 regressor channels have
  cells of their own. Seed and thread count are recorded in every cell row, per the
  contract's scoping (section 7). float64 throughout.
- **Procedure:** per cell, fit both implementations on identical data, measure every
  channel, write per-cell rows into the artifact. Order of operations is normative:
  preregistered ceilings/floors and the sklearn-only noise calibration are committed
  to the artifact before any ferroml cell is read; derived tolerances are computed
  after the full sweep, never during, and only ever tighten toward failure.
- **Artifact:** `docs/test-runs/parity-certification-trees.json` with the NB
  artifact's schema extended by section 7's environment block, the section-3.4
  coverage matrix, and the section-4 preregistered constants block. Gate reads only
  this file.
- **Trivial probes allowed now:** confirming sklearn exposes `tree_.feature`,
  `tree_.threshold`, `children_left/right`, `value` arrays needed for `dt_structure`
  and `traversal_reconstruction`. No comparison runs beyond that in this pass.

## 9. Mutation plan (note 4)

Preregistered mutations, each naming its expected catcher. Run at
gate-implementation time, results recorded in the artifact's notes. Every mutation
marked "caught" doubles as the corrupted-model negative control of section 4: if it
does not violate a gate, the gate plan — not just the mutation — has failed.

| # | Mutation (in a scratch copy of the model wrapper, never shipped) | Expected catcher |
|---|---|---|
| M1 | Perturb one leaf value by +1 in a single tree | `rf_self_consistency` / `rf_self_consistency_reg` (averaged output no longer equals recomputed aggregate) and `rf_proba` / `rf_pred` |
| M2 | Change traversal comparison `<=` to `<` at internal nodes | `traversal_reconstruction`: each probe is built from a sample that reaches the target node, satisfying every ancestor constraint by construction, and carries the split feature exactly at the stored threshold — the mutated comparison flips that probe's leaf, which the test-owned traversal does not do. Assertions: reconstructed leaf id vs `decision_path` terminal node; separately, leaf value vs mapped model output. NOT `dt_structure` (fitted arrays unchanged) and NOT `decision_path`-vs-`predict` (shared semantics, ferroml-core/src/models/tree.rs:357/646/1557 — round 1 wrongly counted this as an independent oracle) |
| M7 | Corrupt one child link (swap a left/right child id) in a single exported tree, leaving feature/threshold arrays untouched | `dt_structure` full-representation comparison (round 3): topology/child-link comparison catches what the round-2 `(feature, threshold, depth)` projection provably could not; also caught by `rf_structure_det` on the deterministic slice. This mutation is the negative control for finding 2 of section 0B |
| M3 | Return lower endpoint instead of midpoint as threshold | `dt_structure` threshold comparison; `dt_proba`; `traversal_reconstruction` probes shift consistently, so the structural channels carry the catch |
| M4 | Drop one env field from the skip check | day-one env-scope guard tests, including the eight-contract-dimension enumeration (section 7) |
| M5 | Widen tolerance floor `1e-12` -> `1e-6` in the gate | policy self-consistency test recomputing `max(10x, 1e-12)` |
| M6 | Bias predict toward class 0 only when `p_top1 - p_top2 < 1e-7` | expected catch: PARTIAL — caught only if the flipped-rate exceeds the capped tail rate bound (cap 0.10, section 4); otherwise escapes |

Expected outcome: six of seven caught cleanly, M6 partial by preregistration —
mirroring the NB experience. M7 exists because round 2's structure gate compared a
projection of the tree; a gate that cannot see corrupted topology must be able to
fail on it, so M7 is applied as a negative control at gate-implementation time like
every other "caught" mutation. Unlike round 1, the M2 expectation rests on an oracle
computed independently of the code under test (test-owned traversal over exported
arrays), so the count is supported rather than assumed. **What a demonstrated escape
means here is defined in advance:** an escaping mutation narrows the gate's published
claim (as M6 already does — the gate claims bounded tail disagreement rate, not
per-sample equivalence) and adds the escape case to `notes.tail_coverage`; it does
not trigger tolerance inflation. The mutation plan is part of the design precisely so
this accounting happens against preregistered expectations rather than after the fact.

## 10. What CANNOT be gated structurally — honest limits

- **Fitted ensemble structure vs sklearn for RANDOMIZED configurations.**
  Different RNG streams (LCG at tree.rs:940, StdRng bootstrap at forest.rs:469/677,
  vs MT19937) make per-tree divergence certain from the first split whenever
  bootstrap or feature subsampling is active. Any gate claiming per-tree structural
  equality for those configs would be theater. Round-3 correction: this limit does
  NOT extend to the deterministic slice (`bootstrap=False` +
  `MaxFeatures::All`, section 3.2), whose members are plain CART trees on identical
  data and ARE gated structurally via `rf_structure_det`. The round-2 blanket "ever"
  was wrong and is narrowed here rather than quietly edited.
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
- **Traversal defects beyond probed inputs, until bindings land.** The
  `traversal_reconstruction` oracle needs the section-11 bindings exposing node
  arrays. Before those land, an M2-class inference mutation is uncovered by every
  planned gate — stated plainly rather than papered over by `dt_structure`, which
  provably cannot see it.
- **Equality-probe completeness.** Probes place one ancestor-constrained sample
  exactly on each internal threshold (section 3.1); they prove the comparison
  convention at each probed node under the asserted reachability, not exhaustive
  routing correctness over the input space. Combined with fit/holdout agreement
  channels this is the practical best available; it is not a proof of routing
  equivalence.
- **HGB traversal defects.** `traversal_reconstruction` is N/A for HGB (finding 5,
  section 0B): its `HistTree` nodes route on u8 bin indices plus a missing-goes-left
  flag over ferroml-private quantile edges (hist_boosting.rs:518, :463-477), and no
  binding in the specified prerequisite work exposes a raw-feature traversal path
  for it. An M2-class inference mutation inside HGB is uncovered by every planned
  gate; HGB keeps only the layer-1 binning invariant and aggregate agreement
  channels. This is a stated limitation, accepted because an implementable-looking
  cell that silently gets dropped is worse than an honest N/A.
- **Regressor cross-implementation parity is bounded discrepancy plus score floors,
  not equivalence.** `rf_pred`/`gb_pred` say how far apart predictions are on the
  certified corpus; they cannot say the models are interchangeable.

Residual risks accepted: correlated errors invisible to cross-implementation gates;
epsilon-detail split flips on near-tie data (mitigated by the tie-free pre-screen,
not eliminated); sklearn-version drift invalidating derived tolerances (mitigated by
the full-environment skip, which turns drift into an explicit skip, not a false pass);
preregistered floors chosen partly from design judgement — if ferroml fails one, that
is a genuine parity result to investigate, and the floor may be revisited only
before measurement, per section 4.

## 11. Prerequisite work and open questions

Implementation prerequisites (future slices, listed because the design depends on
them): Python bindings exposing `TreeStructure` node arrays and `decision_path` for
trees, forest estimators, **and — extended in round 3 — the estimators already
reachable on the boosting side: GradientBoosting's `estimators()`
(ferroml-core/src/models/boosting.rs:512, ferroml-core/src/models/boosting.rs:1114)
and AdaBoost's `estimators()` (ferroml-core/src/models/adaboost.rs:128,
ferroml-core/src/models/adaboost.rs:437)**; per-tree proba access for the classifier
self-consistency channel; per-tree prediction access for the regressor
self-consistency channel. The traversal-reconstruction matrix cells for GBC/GBR/ABC/
ABR are mandatory ONLY once these bindings exist; if a future slice ships without
them, those cells must be re-marked in the artifact as N/A with this reason — they
may not be silently dropped while the certification still reads as covered. HGB is
deliberately NOT added to the prerequisite list: its `HistTree` representation has
no raw-feature node arrays to expose (section 0B, finding 5), so no binding work is
specified for it in this design. The traversal-reconstruction oracle for trees and
forests is blocked on the node-array bindings — until then M2-class defects are
uncovered there too (section 10).

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
- **Q4 (round 2):** empirical scale of sklearn forest seed-to-seed noise, needed to
  instantiate the `rf_proba`/`rf_pred` ceilings at implementation time — measured
  sklearn-only during calibration, before any ferroml cell is read (section 4).
