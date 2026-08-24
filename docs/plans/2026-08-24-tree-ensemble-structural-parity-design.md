# Tree-Ensemble Structural Parity — Design Pass

**Status:** design only; no implementation in this pass. **Round-2 revision:** revised
in place against the Codex adversarial review
(`coord/inbox/2026-08-24-codex-ferroml-design-review.md`) of commit `c22a05e`; see
section 0 for the per-finding disposition.
**Branch:** `pilot/ox-alpha`, HEAD `c22a05e` at time of writing.
**Anchors:** `PARITY-CONTRACT.md` (the four-layer contract and the FM-003 pattern),
the Naive-Bayes slice (`ferroml-python/tests/parity/certify_naive_bayes.py`,
`test_naive_bayes.py`, `docs/test-runs/parity-certification-naive-bayes.json`),
and the audit history (`AUDIT-OX.md`, `FIXES-OX.md`, `VERIFY-OX.md`). Recorded
decisions there are inherited, not re-derived.

Every claim about Rust capability below cites file:line verified at HEAD `c22a05e`
(the tree-ensemble design pass added no code; these files are unchanged from
`0e44d1a`, where the round-1 citations were verified).

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
  both trees and compare flattened structure: per-node `(feature_index, threshold,
  depth)` sequences and leaf count. Applied only to configs marked
  tie-free in the matrix design (see 5); configs that fail a pre-screen for
  near-tied gains fall back to agreement channels only. This pre-screen is itself
  recorded in the artifact, so the set of structure-gated configs is evidence, not
  taste. Scope statement: `dt_structure` verifies *fitted arrays*. It cannot see any
  inference-time defect — most importantly a traversal-semantics change (section 9,
  M2) — and is never cited as the catcher for one.
- **Channel `traversal_reconstruction` (layers 1+3, unconditional).** The test
  implements traversal itself, in test code, from the exported node arrays: left on
  `x[f] <= t`. It never calls ferroml's routing to decide routing. Two probes per
  internal node: (a) a sample whose feature value equals the stored threshold
  bit-for-bit (constructible because thresholds are f64 midpoints,
  ferroml-core/src/models/tree.rs:1016) — the only input distinguishing `<=` from
  `<`; (b) ordinary fit and holdout samples. Assertions per probe: reconstructed
  leaf id equals `decision_path`'s terminal node, equals the tree `predict` output,
  and (classifier) equals `predict_proba`'s implied leaf. This is the M2 catcher.
- **Channels `dt_proba`, `dt_predict` (layers 2-3, classifier).** Two-sided proba
  gates with derived tolerances under preregistered ceilings (section 4), plus label
  agreement, unconditional across all configs.
- **Channel `dt_leaf_value` (conditional structural).** On tie-free configs, single-tree
  leaf `value` vectors are compared elementwise against sklearn's `tree_.value` —
  raw counts (classifier) or raw means (regressor), comparable far more tightly than
  probabilities. The regressor variant is the regression analog of the round-1
  leaf-value channel, now defined instead of implied.

### 3.2 Forests and ExtraTrees (RandomForest*, ExtraTrees* — classifier AND regressor)

Bootstrap resampling differs (different RNG), so per-tree structures differ.
Structural channels compare **within ferroml**, against sklearn only through
aggregates. Round-1 defined these channels with classifier vocabulary only; both
variants are now explicit.

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
reason, fails the gate. `X` = applies, `S` = applies conditionally (tie-free configs
only).

| Channel | DTC | DTR | RFC | RFR | ETC | ETR | GBC | GBR | HGB* | ABC | ABR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dt_structure` | S | S | N/A (bootstrap RNG differs) | N/A (same) | N/A (random thresholds) | N/A (same) | N/A (own training procedure) | N/A (same) | N/A (same) | N/A (same) | N/A (same) |
| `traversal_reconstruction` | X | X | X (per estimator tree) | X | X | X | X | X | X | X | X |
| `dt_proba` / `dt_predict` | X | dt_predict only (values) | — | — | — | — | — | — | — | — | — |
| `dt_leaf_value` | S | S | — | — | — | — | — | — | — | — | — |
| `*_margin_tail` | X | X (pred-margin analog) | X | X | X | X | X | X | X | X | X |
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
| `*_margin_tail` rate | tail disagreement rate | cap 0.10, floor 1e-12 | a tail gate tolerating >10% disagreement certifies nothing |
| `gb_pred` / GBM layer-3 | same forms as `rf_pred` / agreement | same caps and floors | no structural claim is made, so the aggregate bounds match the forest aggregates |

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

- **Margin channel `*_margin_tail` (preregistered).** During certification,
  record per estimator/config/seed the distribution of `p_top1 - p_top2` over a
  holdout set (regressors: the analogous residual-magnitude tail). Preregister a
  margin threshold M = the 1st percentile observed for sklearn, floored at 1e-9,
  before looking at any ferroml results. The gate then asserts, per cell: (a) outside
  the margin tail, label agreement >= the agreement_threshold of section 4 — which
  has a preregistered floor and cannot sink to 0; (b) inside the margin tail, both
  implementations emit valid distributions (finite, simplex), and every disagreement
  is counted and reported in the artifact. Disagreement rate inside the tail gets
  the capped rate bound of section 4 (cap 0.10): a computed bound at or above the
  cap fails the cell as non-discriminating rather than passing vacuously.
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
| M2 | Change traversal comparison `<=` to `<` at internal nodes | `traversal_reconstruction`: the test-owned traversal disagrees with mutated `predict`/`decision_path` on exact-threshold equality probes. NOT `dt_structure` (fitted arrays unchanged) and NOT `decision_path`-vs-`predict` (shared semantics, ferroml-core/src/models/tree.rs:357/646/1557 — round 1 wrongly counted this as an independent oracle) |
| M3 | Return lower endpoint instead of midpoint as threshold | `dt_structure` threshold comparison; `dt_proba`; `traversal_reconstruction` probes shift consistently, so the structural channels carry the catch |
| M4 | Drop one env field from the skip check | day-one env-scope guard tests, including the eight-contract-dimension enumeration (section 7) |
| M5 | Widen tolerance floor `1e-12` -> `1e-6` in the gate | policy self-consistency test recomputing `max(10x, 1e-12)` |
| M6 | Bias predict toward class 0 only when `p_top1 - p_top2 < 1e-7` | expected catch: PARTIAL — caught only if the flipped-rate exceeds the capped tail rate bound (cap 0.10, section 4); otherwise escapes |

Expected outcome: five of six caught cleanly, M6 partial by preregistration —
mirroring the NB experience. Unlike round 1, the M2 expectation rests on an oracle
computed independently of the code under test (test-owned traversal over exported
arrays), so the count is supported rather than assumed. **What a demonstrated escape
means here is defined in advance:** an escaping mutation narrows the gate's published
claim (as M6 already does — the gate claims bounded tail disagreement rate, not
per-sample equivalence) and adds the escape case to `notes.tail_coverage`; it does
not trigger tolerance inflation. The mutation plan is part of the design precisely so
this accounting happens against preregistered expectations rather than after the fact.

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
- **Traversal defects beyond probed inputs, until bindings land.** The
  `traversal_reconstruction` oracle needs the section-11 bindings exposing node
  arrays. Before those land, an M2-class inference mutation is uncovered by every
  planned gate — stated plainly rather than papered over by `dt_structure`, which
  provably cannot see it.
- **Equality-probe completeness.** Probes put one sample exactly on each internal
  threshold with arbitrary values elsewhere; they prove the comparison convention at
  each node, not exhaustive routing correctness over the input space. Combined with
  fit/holdout agreement channels this is the practical best available; it is not a
  proof of routing equivalence.
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
trees and forest estimators; per-tree proba access for the classifier
self-consistency channel; per-tree prediction access for the regressor
self-consistency channel. The traversal-reconstruction oracle is blocked on the node
array bindings — until then M2-class defects are uncovered (section 10).

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
