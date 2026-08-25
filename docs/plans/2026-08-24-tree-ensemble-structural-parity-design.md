# Tree-Ensemble Structural Parity — Design Pass

> ## RESOLVED 2026-08-25 — pre-screen question settled; this document is unparked
>
> The parked question was decided as **option D plus the D-prime refinement**, with three
> additions from a blind Codex review. Full reasoning:
> **`coord/inbox/2026-08-25-prescreen-decision-SETTLED.md`**. The decision was produced by
> two adversarial reviews: Codex, which saw the problem blind and was not told which option
> was recommended, and Prime Agent's red-team of option A from the implementer's seat
> (`coord/results/ferroml-prescreen-redteam.md`), which proved A is not buildable. The two
> reviews conflict on mechanism and agree on the product; D-prime is their reconciliation.
>
> What changed in this revision:
>
> - The **a-priori tie-free pre-screen is retired** — stated plainly here and again at each
>   place it used to condition a channel, not quietly deleted. It is not buildable: its
>   gain-gap screen needs candidate split gains, and neither implementation retains them
>   (sklearn 1.9.0's `tree_` exposes NO gain or candidate attribute; ferroml keeps only the
>   winner — `if gain > best_gain` at `ferroml-core/src/models/tree.rs:902`,
>   `ferroml-core/src/models/tree.rs:1041`, `ferroml-core/src/models/tree.rs:1759`,
>   `ferroml-core/src/models/tree.rs:1853`). Every channel that was conditional on the
>   screen is now conditional on nothing, because adjudication happens after the fit
>   (section 12).
> - New section **"12. Divergence adjudication (D + D-prime)"**: first-divergent-node
>   protocol, same-criterion margin measurement, preregistered cutoff, stratified minimum
>   surviving cells, two independent evaluators, separate structural reporting, and pinned
>   epsilon constants.
> - The three mechanical round-3 findings recorded in the former park banner are fixed:
>   same-feature-ancestor probe escape (section 3.1 probe construction), `impurity_decrease`
>   (`ferroml-core/src/models/tree.rs:198`) added to the complete-representation comparison,
>   and a serialization channel `dt_serialized_inference` per `PARITY-CONTRACT.md:63`
>   (`ferroml-core/src/models/tree.rs:66` imports `Serialize`/`Deserialize`).

**Status:** design only; no implementation in this pass. **UNPARKED — see the resolution note above.** **Round-4 revision:** settled the parked pre-screen decision as option D + D-prime with Codex's three additions, wrote it in as section 12, and retired the a-priori tie-free pre-screen everywhere it conditioned a channel. **Round-3 revision:** revised
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

## 0C. Round-6 review disposition

The round-6 Codex adversarial review returned No-ship with three findings; all three are
addressed in place and none reopens settled round-4 decisions:

- **F1 (M3 escape):** D-prime certifies a midpoint-to-endpoint threshold mutation as a
  zero-margin tie because identical training partitions give identical same-criterion
  gain — while unseen inputs between the thresholds route differently. Resolution: the
  new inference-boundary probe rule in section 12 ("Threshold differences are never
  waived by gain alone") makes any routing difference found by probes strictly between
  the two thresholds an outright failure regardless of margin; the M3 row of section 9
  names that probe — not `dt_structure`/`dt_proba`/`traversal_reconstruction` — as the
  catcher, and the six-of-seven claim is restated as conditional on it.
- **F2 (numbers-less minimum):** section 12's stratified minimum now carries immutable
  numeric minima — cells per family stratum, cells per config class, seeds per cell,
  maximum tie rate, maximum failure rate, minimum examined-node fraction — committed in
  the design rather than deferred to implementation.
- **F3 (malformed matrix):** the coverage matrix is repaired to exactly one cell per
  estimator column in every row, HGB's serialization cell is an explicit reasoned N/A
  instead of an em dash, and the schema rule the guard enforces (exact column
  cardinality, recognized cell states, cardinality checked before semantics) is stated
  in section 3.4.

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
formulas. Structure comparison is well-defined on EVERY certified config. Where a split IS decided
by a near-tie or by the epsilon details above (ferroml's `1e-10` tolerances in equal-value
checks and tie-breaks vs sklearn's exact comparisons can flip a split even without
randomness), the fitted trees diverge from that node down — and that divergence is not
excluded and not screened out; it is adjudicated after the fit under section 12. Therefore:

- **Channel `dt_structure` (layer 1, unconditional).** For each certified config, fit
  both trees and compare the COMPLETE exported representations node-by-node, not a
  projection. Round-2 compared only `(feature_index, threshold, depth)` sequences
  and leaf count; a tree with a corrupted child link or wrong metadata passed that
  gate. The round-3 comparison covers every field both sides expose — ferroml's
  `TreeNode` carries id, feature index, threshold, impurity, sample count,
  weighted sample count, value vector, child ids, depth, leaf flag, AND
  `impurity_decrease` (ferroml-core/src/models/tree.rs:198) — round 4 added it after the
  round-3 review found the "complete representation" claim omitted a public field — and
  sklearn's `tree_` arrays carry the same information (the sklearn-side counterpart of
  `impurity_decrease` is recomputed from `impurity` and `weighted_n_node_samples`;
  because its exact semantics — raw gain versus weight-scaled contribution — are not
  documented, they are pinned once at binding time against that recomputation and the
  chosen semantic is recorded in the artifact):
  - topology: `children_left`/`children_right` versus `left_child`/`right_child`
    for every node — child LINKS, not just the attribute sequences;
  - node count and node id sets (root must be id 0 on both sides);
  - discrete fields exact: feature index, depth, leaf flag, node count, class count;
  - float fields under preregistered tolerances (section 4): thresholds,
    impurity per node, and `impurity_decrease` per node
    (ferroml-core/src/models/tree.rs:198);
  - counts: `n_node_samples` and `weighted_n_node_samples` per node
    (`weighted_n_samples`, ferroml-core/src/models/tree.rs:186);
  - values: the full `value` array with its shape `(n_nodes, 1, k)` compared
    elementwise. Normalization differs by convention — sklearn stores raw class
    counts (or means), ferroml stores class probabilities (tree.rs:187) — so the
    test normalizes BOTH sides to probabilities before comparing, after asserting
    class ordering matches via each model's `classes_`;
  - derived aggregates recomputed from the arrays and asserted equal: leaf count,
    maximum depth, and per-node depth from root.
  Round-4 disposition of the former tie-free pre-screen: RETIRED, stated plainly where the
  screen used to live rather than quietly deleted. The screen classified configs as
  structure-gateable BEFORE fitting by comparing candidate split gains; it is not
  buildable, because candidate gains do not survive either fit (sklearn 1.9.0's `tree_`
  exposes no gain or candidate attribute; ferroml keeps only the winner at
  ferroml-core/src/models/tree.rs:902, :1041, :1759, :1853). `dt_structure` now carries no
  precondition: it runs on every certified config. Identical structures pass outright;
  divergent structures go to the first-divergent-node adjudication of section 12 instead
  of falling back to agreement channels. There is no longer any knob that widens the set
  of structure-gated configs, because there is no longer a screen to widen. Scope statement: `dt_structure`
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
  ferroml-core/src/models/tree.rs:1016). Round-4 correction of a round-3 escape: copying s
  is NOT sufficient when an ANCESTOR OF N SPLITS ON THE SAME FEATURE as N. Setting that
  feature to N's threshold can violate the ancestor's constraint — s reached N through
  x[f] <= t_anc while equality at N demands x[f] = t_N, which may be > t_anc — so the probe
  would be routed away before reaching N and the mutated comparison would escape. The probe
  builder therefore first enumerates every ancestor of N that splits on f and intersects its
  constraint interval with {t_N}. If the intersection is empty, NO equality probe exists
  through this path and node N is UNPROBEABLE — recorded and failed like any other
  unprobeable node, never skipped. When the intersection is non-empty, the copied-and-pinned
  sample satisfies every ancestor constraint on every feature by construction, the probe is
  guaranteed to arrive at N, and flipping N's comparison from `<=` to `<` cannot escape
  detection. Reachability is re-asserted with the test-owned
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
- **Channel `dt_leaf_value` (structural).** On every certified config whose fitted
  structures are identical (divergent structures are adjudicated under section 12;
  adjudicated-tie configs are reported separately and never counted inside a structural
  headline), single-tree leaf `value` vectors are compared elementwise against sklearn's
  `tree_.value` —
  raw counts (classifier) or raw means (regressor), comparable far more tightly than
  probabilities. The regressor variant is the regression analog of the round-1
  leaf-value channel, now defined instead of implied.
- **Channel `dt_serialized_inference` (layers 1+3; round-4 addition).** `PARITY-CONTRACT.md:63`
  requires "structure and serialized inference where applicable" for trees, and the tree
  model derives `Serialize`/`Deserialize` (serde imported at
  ferroml-core/src/models/tree.rs:66; the forest containers likewise at
  ferroml-core/src/models/forest.rs:104 and :844), so it applies. Per certified config:
  serialize the fitted model through the standard save path, deserialize into a fresh
  instance, then assert (a) the deserialized model's exported representation equals the
  pre-serialization representation under the full `dt_structure` field comparison — a
  round-trip identity check against ITSELF, not a cross-implementation claim — and
  (b) predictions on the holdout set are bit-identical to the pre-serialization model's.
  A digest of the serialized artifact is recorded in the certification artifact. Forest
  rows apply the same round trip once the §11 estimator bindings exist.

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
as `bootstrap=False, max_features=n_features`. This makes every estimator tree structurally comparable to its sklearn counterpart on
every certified deterministic config; member trees that diverge are adjudicated under
section 12 exactly like single trees (round 4 removed the former engineered-config /
pre-screen precondition).

- **Channel `rf_structure_det` (layer 1; scoped to the deterministic slice — round 4
  removed the former tie-free pre-screen condition).
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
get silently dropped at implementation time.**

**Round-6 schema rule the guard enforces:** every row of this matrix carries EXACTLY
one cell per header column — twelve fields including the channel name, eleven
estimator columns, no more and no fewer. Recognized cell states are `X`, `S`, and
`N/A (reason)`; an em dash is not a recognized state and is rejected wherever an
estimator has no channel, because a bare dash cannot say WHY coverage is absent.
The cardinality check runs BEFORE any semantic validation: a row whose field count
differs from the header's is a malformed document and the guard rejects it, since a
parser could otherwise silently truncate surplus cells or mis-map coverage while the
document claims to be machine-validated. Every markdown
table in this document is checked by the guard independently: each of its rows must
carry its own header's cell count, and no table can pass or fail because of another. `X` = applies, `S` = applies on a scoped slice (the deterministic-forest slice of §3.2;
round 4 retired the tie-free pre-screen that formerly scoped these cells, so the scoping
is configurational, not a screen).

| Channel | DTC | DTR | RFC | RFR | ETC | ETR | GBC | GBR | HGB* | ABC | ABR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dt_structure` | X | X | X via `rf_structure_det` only | S (same slice) | N/A (random thresholds) | N/A (same) | N/A (own training procedure) | N/A (same) | N/A (own HistTree representation) | N/A (stochastic sample reweighting) | N/A (same) |
| `rf_structure_det` | — | — | S (deterministic slice, §3.2; full-representation comparison per estimator tree) | S (same) | N/A (random thresholds) | N/A (same) | N/A (own training procedure) | N/A (same) | N/A (HistTree bins) | N/A (same) | N/A (same) |
| `traversal_reconstruction` | X | X | X (per estimator tree) | X | X | X | X (requires §11 binding over estimators(), boosting.rs:512/1114) | X (same) | N/A (HistTree routes on u8 bin indices over private bin edges — not implementable from specified work; see §10) | X (requires §11 binding over estimators(), adaboost.rs:128/437) | X (same) |
| `dt_proba` / `dt_predict` | X | dt_predict only (values) | — | — | — | — | — | — | — | — | — |
| `dt_leaf_value` | X | X | — | — | — | — | — | — | — | — | — |
| `dt_serialized_inference` (§3.1) | X | X | X (per estimator tree, §11 bindings) | X (same) | X (same) | X (same) | X (via estimators(), boosting.rs:512/1114) | X (same) | N/A (HistTree serialization semantics unstated; representation limit per §0B finding 5) | X (via estimators(), adaboost.rs:128/437) | X (same) |
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
| `dt_proba` (structure-matched) | abs proba diff | ceiling 1e-9 | equal structures imply equal integer class counts, hence rational probas agreeing to division rounding (~1e-16); 1e-9 leaves slack for summation order only |
| `dt_leaf_value` (structure-matched) | per-leaf count/mean diff | ceiling 1e-9 (counts: exact 0) | integer counts must match exactly; regressor leaf means agree to float rounding on identical structures |
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
- **Rare-leaf channel `*_leaf_value` (structural).** For single trees on every certified
  config whose fitted structures are identical (section 12 governs divergences), leaf
  `value` vectors are compared elementwise under the section-4 ceiling (counts exact;
  means to 1e-9). For ensembles, leaf-depth and leaf-size
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
  classes, engineered low-near-tie variants alongside ordinary configs for the structural
  channels (round 4: these are corpus design, NOT a screen — the pre-screen is retired,
  and divergent fits are adjudicated per section 12 rather than excluded), and
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
| M3 | Return lower endpoint instead of midpoint as threshold | REASSIGNED in round 6 after the previous claim was shown false: NOT `dt_structure`, NOT `dt_proba`, NOT `traversal_reconstruction`. The structural delta routes into section-12 adjudication, where the endpoint and the midpoint partition the TRAINING data identically — identical same-criterion gain, margin exactly zero below the 1e-9 cutoff — so adjudication records a TIE, not a failure. Training predictions are unchanged (`dt_proba` is structure-matched and sees nothing), and `traversal_reconstruction` follows the exported mutated threshold and stays self-consistent. The real catcher is the inference-boundary probe of section 12 ("Threshold differences are never waived by gain alone"): probes placed STRICTLY BETWEEN the two thresholds against an independent reference boundary change routing, which fails certification regardless of gain margin. |
| M4 | Drop one env field from the skip check | day-one env-scope guard tests, including the eight-contract-dimension enumeration (section 7) |
| M5 | Widen tolerance floor `1e-12` -> `1e-6` in the gate | policy self-consistency test recomputing `max(10x, 1e-12)` |
| M6 | Bias predict toward class 0 only when `p_top1 - p_top2 < 1e-7` | expected catch: PARTIAL — caught only if the flipped-rate exceeds the capped tail rate bound (cap 0.10, section 4); otherwise escapes |

Expected outcome: six of seven caught cleanly, M6 partial by preregistration —
mirroring the NB experience. Round 6 states precisely what that claim now is, because
M3 was re-assigned. Five of the six clean catches rest on channels that predate round
6; M3's catch rests SOLELY on the inference-boundary probe added to section 12 this
round. Before that rule existed, the honest expectation was five of seven caught
cleanly, M6 partial, and M3 ESCAPING as a zero-margin adjudicated tie while changing
routing for unseen inputs between adjacent training values — reproduced numerically
before this revision (training values [1,2,3,4,5,6], adjacent pair 3.0/4.0;
thresholds 3.5 and 3.0 give identical partitions [1,2,3] and identical gains
0.50000000000000000; D-prime margin 0.00e+00 against the 1e-9 cutoff; yet x=3.2 and
x=3.4 route LEFT under 3.5 and RIGHT under 3.0). The six-of-seven claim is therefore
CONDITIONAL on the boundary probe being implemented and exercised as part of the
mutation harness; a run reporting six of seven without exercising it is misreporting,
not success. M7 exists because round 2's structure gate compared a
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

- **Any structural claim stronger than section 12 permits.** Round 5 narrowed the permitted
  claim to exactly this sentence, replacing the round-4 wording ("every structural
  divergence ... individually adjudicated"), which the procedure beneath it — first-
  divergent-node adjudication only — did not support: *"On a preregistered corpus, at the
  FIRST divergent node of every non-identical config, ferroml and scikit-learn's chosen
  splits were individually adjudicated under one test-owned criterion; none exceeded the
  preregistered margin; N configs were recorded as ties with their margins published,
  M configs had identical structure, and F failed outright (above-cutoff or shape
  divergence). Structure below an adjudicated tie is NOT covered by this claim."*
  Nothing in this design supports "structurally identical", any unqualified parity
  headline, or any structural statement about territory below an adjudicated tie. Structural results and prediction-only
  results are reported separately; adjudicated-tie cells are published with their margins
  and never inflate a structural headline.

Residual risks accepted: correlated errors invisible to cross-implementation gates;
epsilon-detail split flips on near-tie data (round 4 retired the tie-free pre-screen,
so nothing is screened out up front; each divergence is adjudicated at the FIRST DIVERGENT
NODE under section 12; an above-cutoff divergence fails certification outright with no
fallback, and so does a shape divergence, which has no second split to score; structure
below an adjudicated tie is unexamined and outside every structural claim); sklearn-version drift invalidating derived tolerances (mitigated by
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
  settle by reading sklearn's splitter source at implementation time. Round 4 removed the
  purpose this question used to serve — sizing the tie-free pre-screen, which is retired
  (section 12) — so the answer now matters only for predicting which certified configs
  produce divergent trees that adjudication must classify.
- **Q2:** whether sklearn's integrated `min_samples_leaf` handling versus ferroml's
  post-hoc rejection changes reachable tree shapes on tie-free configs — settle with
  the trivial probes in section 8.
- **Q3:** whether any worker transcript holds measured tree-vs-sklearn discrepancy
  magnitudes usable for pre-sizing the matrix — deliberately not consulted per
  brief; the certification run answers it empirically.
- **Q4 (round 2):** empirical scale of sklearn forest seed-to-seed noise, needed to
  instantiate the `rf_proba`/`rf_pred` ceilings at implementation time — measured
  sklearn-only during calibration, before any ferroml cell is read (section 4).

## 12. Divergence adjudication (D + D-prime)

Round 4 settled the question that parked this document. The decision is **option D plus
the D-prime refinement, with three additions from a blind Codex review**; full reasoning,
including why option A died, is in `coord/inbox/2026-08-25-prescreen-decision-SETTLED.md`,
produced by two adversarial reviews (Codex blind; Prime Agent red-teaming option A from
the implementer's seat). This section is the normative replacement for the retired
a-priori tie-free pre-screen, and the round-4 gate reads THIS SECTION ONLY.

### Why there is no pre-screen anymore

The a-priori tie-free pre-screen is RETIRED, not revised. It required candidate split
gains from both implementations before fitting, and those do not exist: sklearn 1.9.0's
`tree_` exposes NO gain attribute and no candidate or runner-up splits (verified by
probing its full public surface), and ferroml retains only the winning split
(`if gain > best_gain` at ferroml-core/src/models/tree.rs:902,
ferroml-core/src/models/tree.rs:1041, ferroml-core/src/models/tree.rs:1759,
ferroml-core/src/models/tree.rs:1853). Reconstructing either side's candidate set means
cloning its splitter — which produces YOUR floats, least accurate exactly on the
near-ties the screen existed to classify. A screen that cannot be built honestly must
not be simulated.

### The protocol

1. **Fit both trees on every certified config.** No config is excluded up front and no
   ties are classified in advance. If the two fitted structures are IDENTICAL, every
   structural channel passes outright for that config — no screen needed, no ties to
   classify.
2. **At the FIRST DIVERGENT NODE, adjudicate.** Walk both trees in lockstep from the
   root; the first node where the two structures differ is the adjudication point,
   and everything below it is downstream of one split decision. A node DIVERGES when
   feature choice or threshold differs — OR WHEN EXACTLY ONE SIDE IS A LEAF. That
   second case is a SHAPE divergence: one implementation splits a node where the
   other stops. It must be detected as a divergence event even though no feature or
   threshold differs there; a walk that compares only the split attributes of two
   internal nodes walks straight past a shape divergence without seeing anything.
3. **Above the cutoff, certification FAILS OUTRIGHT.** This is stated explicitly because
   it is the load-bearing rule: a divergence whose measured margin exceeds the
   preregistered cutoff is a certification failure with NO FALLBACK to agreement
   channels. There is no path by which a large-margin structural divergence becomes a
   passing cell.
4. **Below the cutoff, record an adjudicated tie**: margin, node id, config id, and both
   chosen splits go into the artifact, and the margin histogram is published — EXCEPT
   that a tie whose divergence is a THRESHOLD difference additionally requires the
   inference-boundary probe of the next subsection. Equal gain never waives a
   routing-changing threshold.

### Threshold differences are never waived by gain alone (round 6)

The M3 defect class, found by the round-6 Codex review and reproduced numerically before
this revision: two thresholds that partition the TRAINING data identically have
IDENTICAL same-criterion gain under D-prime, so their margin is exactly zero and rule 4
records an adjudicated tie — while the two boundaries route unseen inputs differently.
Verified reproduction: training values [1,2,3,4,5,6], adjacent pair 3.0/4.0; thresholds
3.5 (midpoint) and 3.0 (lower endpoint) induce the identical left partition [1,2,3],
score gains of 0.50000000000000000 under the same criterion on both, D-prime margin
0.00e+00 against the 1e-9 cutoff — yet unseen x=3.2 and x=3.4 route LEFT under 3.5 and
RIGHT under 3.0. Gain equality means the two splits are equally good ON THE TRAINING
PARTITION; it says nothing about the inference boundary each threshold induces over the
open interval between them.

The rule: **when the adjudication point is a THRESHOLD difference (same feature,
different threshold), same-criterion gain adjudication alone can NEVER waive it.**
Before such a divergence is recorded as a tie, the harness runs the mandatory
inference-boundary probe:

- Probe inputs are drawn STRICTLY BETWEEN the two thresholds — t_min < x < t_max,
  exclusive on both ends — including values arbitrarily close to each endpoint.
- Each probe is evaluated against an INDEPENDENT REFERENCE BOUNDARY: the test-owned
  traversal of the reference (sklearn-side) tree, computed independently of ferroml's
  exported arrays. Neither the artifact's own routing compared to itself nor any oracle
  derived from the mutated arrays qualifies as the reference.
- Probes inherit the ancestor constraints of section 3.1's equality probes: each is
  built from (or constrained to) input that reaches the adjudicated node.
- If ANY probe routes differently under the two boundaries, certification FAILS for that
  config outright — regardless of how small the measured gain margin is. This failure
  has the same load-bearing status as rule 3 and shape divergence: no fallback to
  agreement channels, no waiver by margin.
- If NO f64 value lies strictly between the two thresholds, the boundaries are provably
  routing-equivalent over every valid input and the tie stands without the probe.

Why this is sound where gain adjudication was not: the probe evaluates exactly what the
margin cannot see — whether the two certified boundaries induce the same decision
function on the interval separating them. A threshold difference that changes no
routing there changes nothing inferable and may pass as a tie; a threshold difference
that does change routing is a materially different model, and arithmetic over training
partitions must not be able to certify it.

### Shape divergence: one side has no split to score

A shape divergence is not an edge case in this corpus; it is the MOST LIKELY divergence.
Ferroml's hard-coded stopping rules (`depth >= 1000`, `impurity < 1e-10`,
ferroml-core/src/models/tree.rs:707-708) sit against sklearn's impurity-epsilon and
depth-cap semantics (open question Q1), and ferroml enforces `min_samples_leaf` as a
post-hoc rejection of candidate splits rather than integrating it into splitting
(ferroml-core/src/models/tree.rs:733-735, open question Q2). Both are stopping-rule
differences, and a stopping-rule difference surfaces first as a node that one side
splits while the other stops and makes it a leaf.

The rule for it is explicit: **a shape divergence cannot be adjudicated by margin, so
it FAILS THE CONFIG OUTRIGHT — same load-bearing status as rule 3, no fallback to
agreement channels.**

Why there is no margin: D-prime scores TWO CHOSEN SPLITS under one test-owned criterion
over the same node's samples. At a shape-divergent node the stopping side has no split
at all — there is no second split to score. Scoring "split versus no-split" would need
the stopped side's best candidate gain, which means enumerating candidates — exactly
the retired option A, dead for the reasons recorded above; neither implementation
exports its candidate set or runner-up gains.

Why failure rather than something softer: a below-cutoff margin certifies that two
implementations picking different splits on identical data picked splits of equivalent
quality — float noise, not semantic disagreement. A shape divergence is the opposite:
the implementations disagree about whether to keep growing the tree at all. That is a
genuine stopping-semantics difference (Q1/Q2) — precisely the defect class this
certification exists to surface — and recording it as a tie would launder a semantic
difference through arithmetic built only for near-tie gains over identical samples.

The artifact reports shape-divergence failures as their own category, distinct from
above-cutoff failures, so a corpus-wide pattern of them reads as what it is: evidence
that stopping semantics need reconciliation before structural parity is even askable.
Because Q1/Q2 make shape divergence likely, this rule should be expected to bind often;
that pressure is intended.

### D-prime: the margin is a measurement, not a proxy

Do NOT compare sklearn's winning gain against ferroml's winning gain. Those numbers come
from two different implementations, so their difference carries a cross-implementation
confound — it is a proxy for a tie, not a measurement of one. Instead:

- Score BOTH sides' chosen splits under ONE test-owned criterion, using node membership
  taken from the traversal the design already reconstructs (section 3.1): the test
  computes each split's impurity decrease over the same node's samples with the same
  formula, so the gap between sklearn's split and ferroml's split is a true
  same-criterion quantity.
- Verified buildable: reconstructing a winning gain from exported fields agrees with an
  independent test-code computation to 1.11e-16, and a NON-winning split can be scored
  the same way.
- Evaluating two KNOWN splits is not enumerating candidates. That distinction is exactly
  why this is buildable when option A was not: option A needed every candidate's gain;
  adjudication needs only the two splits the fits actually chose.

### Preregistered cutoff

- **Cutoff: 1e-9 absolute**, preregistered before any measurement.
- **Derivation:** the float cancellation floor of the reconstruction itself — parent
  impurity minus weighted child impurities cancels to roughly 1e-14 at Gini/entropy
  magnitudes, so margins below ~1e-14 are arithmetic noise; 1e-9 sits five orders above
  that floor and far below any decision-relevant gain difference.
- **Timing:** the cutoff, the corpus and config ids, the criterion implementation, and
  the epsilon pins below are committed to the artifact PREREGISTERED BEFORE any ferroml
  measurement is read. Changing any of them afterward follows PARITY-CONTRACT.md:75-76:
  never after a production failure.
- **Publication:** the artifact publishes the full margin histogram over adjudicated
  ties, so a reader can see how close the corpus sits to the cutoff rather than only the
  count N.

### Minimum surviving cells: stratified

The minimum is STRATIFIED, not a bare count, and it carries IMMUTABLE NUMBERS committed
here in the design — not deferred to implementation. Certification fails when ANY of
them is missed. The numbers are preregistered before measurement; changing any of them
afterward follows PARITY-CONTRACT.md:75-76 — never after a production failure. A corpus
that survives only inside one family or one easy config class does not support the claim
regardless of its size, and a total alone would let it.

- **Cells per family stratum:** at least 12 cells per family stratum must survive — the
  two strata are single trees and deterministic forests (section 3.2) — so at least 24
  surviving cells corpus-wide. Strata are enumerated sets; one easy cell per named
  stratum does not satisfy a minimum of 12.
- **Cells per config-class stratum:** at least 3 cells per config class must survive,
  inside EACH family stratum's count, across the four classes small-n, large-n,
  low-dimension, high-dimension; plus at least 6 regressor-variant surviving cells
  corpus-wide (at least 3 per family stratum), because regressor coverage rests on
  structurally different channels than classifier coverage.
- **Seeds per cell:** at least 5 seeds per cell. A cell survives only if EVERY seed
  reaches the same classification under BOTH independent evaluators; a cell whose
  classification varies across seeds is a failed cell, not a partial credit.
- **Maximum tie rate:** adjudicated ties may not exceed 40% of certified cells (cells
  excluding representation-limited exclusions). Ties carry unexamined subtrees (see
  below); a corpus dominated by ties supports almost no structural territory no matter
  how many cells it has.
- **Maximum outright-failure rate:** above-cutoff and shape-divergence failures may not
  exceed 5% of certified cells corpus-wide, and may not exceed 25% within any single
  family stratum. These caps ADD to rule 3 and the shape rule — they never soften them;
  each individual failure still fails its own config outright with no fallback.
- **Minimum examined-node fraction:** at least 60% of total tree nodes corpus-wide must
  be EXAMINED — all nodes of identical-structure configs, plus every node at or above
  the adjudication point of each tie cell, counted over both implementations' trees.
  The fraction is computed from the per-tie visibility data required below (depth of
  adjudication point, subtree node counts) and published alongside the margin
  histogram. Coverage of CELLS is not coverage of TREES: a run meeting every cell
  minimum but falling below this fraction fails certification.

### Two independent evaluators

TWO INDEPENDENT evaluator implementations — written separately against this section, not
one implementation checked against itself — must agree on EVERY classification
(identical / adjudicated-tie / fail). Any disagreement between evaluators fails the cell
outright. One evaluator is one point of failure; a mis-classified above-cutoff divergence
is precisely the defect class this programme exists to catch.

### Separate reporting

Structural results and prediction-only results are reported SEPARATELY, as distinct
claims in the artifact. Configs excluded for representation limits and configs recorded
as adjudicated ties are listed explicitly and MUST NOT be combined into, counted inside,
or used to inflate a structural headline. The structural claim stands only on the cells
it actually covers, and the prediction-only claim never borrows structural language.

### Pinned epsilon constants

Ferroml's epsilon constants are pinned in the artifact, with file:line and values read
at the certified commit: ferroml-core/src/models/tree.rs:1041 (classifier strict-greater
tie-break), ferroml-core/src/models/tree.rs:1853 (regressor
`gain > best_gain + 1e-10`), ferroml-core/src/models/tree.rs:1012 (equal-value gap
1e-10). These are the sharpest cheat knob under ANY option: widening
them until ferroml picks whatever sklearn picks would make the whole corpus agree, the
gate green, and the shipped model silently changed. The pins make that mutation visible
in the artifact diff; any change to a pinned constant invalidates the certification and
forces a full re-run against re-preregistered constants.

### What an adjudicated tie covers — and what it does not

External review (round 5) found the protocol silent on subtree coverage: after an
adjudicated tie, both subtrees below the tie are built on different data and nothing
examines them, so one adjudicated tie can pass a config while everything beneath it
goes unchecked. Two resolutions existed — recurse into the subtrees, or narrow the
claim — and the resolution chosen is **(b): NARROW THE CLAIM**, with the reason against
recursion recorded.

Recursion was rejected on substance, not cost. After an adjudicated tie at node v,
ferroml and sklearn route v's samples through DIFFERENT splits, so every corresponding
position below v is fitted on a different sample set. Any further structural difference
below v is then guaranteed data divergence, not implementation noise. The 1e-9 cutoff
derives its meaning from the ~1e-14 cancellation floor of scoring two splits over THE
SAME samples; applied across different sample sets it measures nothing about
implementation fidelity. Recursing would convert those guaranteed differences into
manufactured "adjudications" and inflate N with numbers that carry no evidence.
Margin adjudication does not transfer below a tie, so recursion would make "every
divergence adjudicated" true in words and meaningless in substance.

What the procedure therefore does and does not cover, stated plainly:

- Per config, exactly one of three outcomes: identical structure (full structural
  coverage for that config); an adjudicated tie at the FIRST divergent node, where
  NOTHING BELOW THE TIE IS STRUCTURALLY COMPARED; or outright failure (above-cutoff or
  shape divergence). The subtrees below an adjudicated tie are unexamined territory for
  every structural channel. Behavior there is covered only by the prediction-only
  channels, which still run on the whole corpus and are reported separately per this
  section.
- The artifact makes the uncovered region VISIBLE per tie cell: depth of the
  adjudication point, node counts of both unexamined subtrees, and the fraction of each
  tree's nodes below the tie — plus a corpus-level aggregate of examined vs total nodes.
  The more ties a corpus records, the smaller its structurally covered fraction; that
  fraction is published alongside the margin histogram rather than buried under the tie
  count N.

### Standing rule: claims are scoped by the procedure beneath them

This is the THIRD instance in this document's history of a completeness claim outrunning
the procedure underneath it: the round-2 "whole certified environment block" (seven of
ten PARITY-CONTRACT dimensions), the round-3 representation comparison that skipped
`impurity_decrease` (ferroml-core/src/models/tree.rs:198), and now round-5's "every
structural divergence" over a first-divergence-only protocol. The standing rule adopted:
any artifact claim containing "every", "all", or "whole" must name the procedure step
whose enumeration defines its scope, and the claim's scope IS that enumeration — never
wider. If no such step exists, the claim is rewritten at the procedure's actual
granularity. A reviewer finding a fourth instance should treat the pattern as the
finding, not just the case.

### The permitted claim

The ONLY structural claim this certification supports, stated once and narrowed to what
D + D-prime actually examines:

> "On a preregistered corpus, at the FIRST divergent node of every non-identical
> config, ferroml and scikit-learn's chosen splits were individually adjudicated under
> one test-owned criterion; none exceeded the preregistered margin; N configs were
> recorded as ties with their margins published, M configs had identical structure, and
> F failed outright (above-cutoff or shape divergence). Structure below an adjudicated
> tie is NOT covered by this claim."

No artifact text may state or imply anything stronger — not "structurally identical",
not unqualified "structural parity", and not the round-4 wording ("every structural
divergence ... individually adjudicated"), which this round replaces because the
procedure beneath it adjudicates only the first divergence and cannot support the
unqualified quantifier. Section 10 records the same narrowing.
