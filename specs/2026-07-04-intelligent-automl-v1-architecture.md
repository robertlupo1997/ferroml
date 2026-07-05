# FerroML Intelligent-AutoML v1 — Rebuild Architecture

**Date:** 2026-07-04 (Fable architecture pass)
**Inputs:** `thoughts/architecture-map-2026-07-04.md` (Sonnet mapping agent, read-only);
career design spec (Ornith repo `docs/superpowers/specs/2026-07-04-career-direction-design.md`).
**Execution:** post-2026-07-07 on Opus (plan/review) + Sonnet (implement) per the routing table.
**v1 deadline:** Aug 29, 2026 (career spec P3 done bar).

## The verdict that reframes the project

Trey's brief was "redo it all to make it actually usable and intelligent." The map says the
intelligence already exists as *parts* — what's missing is the *wiring*:

- `AutoML::fit()` (`automl/fit.rs:839`) is a real orchestration loop (data characteristics →
  portfolio → bandit budget → CV trials → ensemble → statistical leaderboard).
- But `run_trial()` (`fit.rs:1064`) evaluates every algorithm ONCE with `new()` defaults.
  The complete, unit-tested `hpo/` module (TPE, Bayesian, Hyperband, ASHA) is never sampled.
  **Today FerroML is algorithm selection with ensembling — the H in CASH is unwired.**
- Preprocessing selection (`automl/preprocessing.rs`) is built but trials receive raw data —
  scale-sensitive models (SVM/KNN/logistic) are systematically handicapped.
- Meta-learning/warm-start/transfer modules exist; the store ships empty; none are invoked.
- No conformal prediction anywhere — uncertainty comes from OLS residuals/bootstrap only.

**Therefore v1 is NOT a rewrite.** It is five surgical workstreams into a 243k-LOC codebase
that already has the parts, the test culture (≈4,700–5,000 real tests, mutation CI), and the
statistical vocabulary.

## v1 Workstreams (priority order)

### WS1 — Wire HPO into the trial loop (the CASH milestone)
`run_trial()` samples hyperparameters from each `AlgorithmConfig.search_space` via the
existing `hpo::Study` (TPE default), instead of `create_model()` defaults; the bandit's
per-algorithm budget buys N trials, not 1. Success: on a fixed-seed benchmark dataset,
CASH beats defaults-portfolio by a statistically significant margin (the repo's own
`corrected_resampled_ttest`).
Anchors: `automl/fit.rs:1064,1086,1146`; `hpo/`; `automl/portfolio.rs` (`AlgorithmType`).

### WS2 — Wire preprocessing into trials
Each trial runs `PreprocessingSelector`'s pipeline spec (scaling/imputation/encoding chosen
per algorithm family + `DetectedCharacteristics`) inside the CV fold (fit on train fold only
— no leakage). Success: SVM/KNN/logistic portfolio scores improve on unscaled datasets;
leakage guard proven by a test where preprocessing-on-full-data would inflate the score.
Anchors: `automl/preprocessing.rs` (incl. `DetectedCharacteristics:514`), `pipeline/`.

### WS3 — Data profiling gate at fit() entry
`AutoML::fit()` opens with schema/profile: missing values, constant features, imbalance,
high cardinality — surfaced as structured warnings on `AutoMLResult` (and printed by the
Python layer). Success: feeding a pathological dataset yields actionable warnings, not
silent degradation. Anchors: `schema.rs`, `automl/preprocessing.rs`.

### WS4 — Conformal prediction layer (the differentiator)
Split-conformal (regression intervals; classification prediction sets via nonconformity
scores) wrapping ANY `Predictor` — distribution-free coverage guarantees, the one rigor
feature the "statistically rigorous" library lacks and the author's personal signature
(conformal CIs in production at his day job). Success: empirical coverage ≈ nominal (e.g.,
90% ± CI) across portfolio models on held-out benchmarks; exposed in Python as
`ConformalPredictor(model).predict_interval(x, alpha)`.
Anchors: new `ferroml-core/src/conformal/`; `Predictor::predict_with_uncertainty` (lib.rs).

### WS5 — Honest benchmark + model card + claim hygiene
Benchmark v1 against auto-sklearn and FLAML on a small public suite (5–8 OpenML datasets,
fixed budget, CV, statistical comparison with the repo's own multiple-testing correction).
Publish results in README **whatever they say**. Fix over-claims found by the map: test-count
wording (≈4.7–5k functions, not "6,500+ tests" unless counting property expansions —
say which), don't call it CASH/AutoML-in-the-auto-sklearn-sense until WS1 lands, note GPU
module is untested (`mutants.toml` excludes `gpu/**`) or gate it behind a doc warning.
Anchors: `model_card.rs`, `benchmarks/`, README.

## Sequencing to Aug 29

| Phase | Weeks | Content |
|---|---|---|
| 1 | Jul 7–18 | WS1 (HPO wiring) + WS5's claim-hygiene edits (cheap, do early) |
| 2 | Jul 18–Aug 1 | WS2 (preprocessing in trials) + WS3 (profiling gate) |
| 3 | Aug 1–15 | WS4 (conformal) |
| 4 | Aug 15–29 | WS5 benchmark runs + model card + README/writeup; buffer |

Each phase gets its own detailed implementation plan (superpowers:writing-plans format,
bite-sized TDD tasks) written by the Opus planner AT PHASE START, using this doc + the map
as inputs — per-task Rust diffs are deliberately NOT specified here; the codebase is too
large to plan blind, and the map gives the planner exact anchors.

## Out of scope for v1 (post-v1 backlog)

Meta-learning store seeding (needs a corpus of runs v1 will generate), transfer learning
wiring, GPU CI + correctness (12k untested lines — quarantine, don't advertise), time-series/
survival tasks (enum variants currently alias to regression), BLAS-optional linalg,
`get_feature_names_out()`/`clone()` sklearn parity gaps.

## Risks

- **Rust iteration speed on a 243k-LOC crate** — mitigation: phases are narrow; the test
  suite + mutation CI catch regressions; Sonnet implements against per-phase plans with
  exact anchors.
- **Benchmark embarrassment** (auto-sklearn wins) — acceptable by design: the story is
  rigor + honesty, and WS1/WS2 should close most of the gap; publish regardless.
- **Scope creep toward "redo it all"** — this doc is the fence: five workstreams, hard
  deadline, everything else is backlog.
