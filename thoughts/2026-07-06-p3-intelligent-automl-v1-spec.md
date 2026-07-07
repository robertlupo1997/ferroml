# P3 design spec — Intelligent AutoML v1 (statistically-grounded selection)

**Date:** 2026-07-06 · **Author:** Claude (Fable 5), per the career-direction spec's "Fable architects" step · **Status:** DRAFT for Trey's approval · **Deadline:** Aug 29 (career spec P3) · **Input:** `thoughts/2026-07-06-p3-architecture-map.md` (verified architecture, cited there)

## What P3 proves (career spec)

"The stats degree made software": FerroML stops *annotating* statistics onto AutoML results and starts *deciding* with them. Today `best_model()` returns leaderboard top-1 by CV mean; the Nadeau-Bengio corrected t-tests are computed and attached but never drive selection. v1 closes that loop and renders the evidence.

## The v1 slice — three additive features, one story

### 1. `AutoMLResult::statistically_best_model()` (~20 lines, `ferroml-core/src/automl/fit.rs`)
Selection rule: take `competitive_models()` (already BH-corrected pairwise non-inferiority vs the top scorer) → among them pick the LOWEST `AlgorithmComplexity` (exists in `portfolio.rs`), tie-break by training time. Returns the entry **plus a `SelectionRationale`**: top-1 score ± std, chosen model's score ± CI, the p-value(s) that established non-inferiority, complexity classes compared. The rationale is the product — "simplest model not statistically distinguishable from the best" is the sentence interviewers remember.

### 2. `AutoMLResult::render_benchmark_report()` (~50 lines, same file)
Serializes `ModelComparisonResults` to Markdown and JSON: leaderboard (CV mean ± std, bootstrap CI, train time), pairwise significance matrix (corrected resampled t-test, BH-adjusted), and explicit **"no significant difference"** statements where that's the truth. Extends the existing `summary()` text table. Honesty is the feature: the report must be equally comfortable saying "these five models are statistically indistinguishable on this dataset."

### 3. `ModelCardGenerator::from_automl_result()` (~80 lines, `ferroml-core/src/model_card.rs`)
Merges static `HasModelCard` data (strengths/limitations/references) with run data: `DataCharacteristics`, CV protocol, score ± std + CI, training time, and the `SelectionRationale` from (1). Output: one Markdown model card for the selected model.

### 4. Python surface (3-step binding rule per CLAUDE.md)
`result.statistically_best_model()`, `result.benchmark_report(format="markdown"|"json")`, `result.model_card()` — PyO3 wrappers in `ferroml-python/src/automl.rs` + `model_card.rs`, re-exports in `python/ferroml/automl/__init__.py`.

## Testing (repo rules honored: no new integration binaries)

- Rust: add to existing automl test surface — synthetic fixtures where (a) a simple model is non-inferior to a complex top-1 → selection MUST prefer simple; (b) simple is significantly worse → MUST NOT; (c) single-model degenerate case; (d) report renders both "significant" and "indistinguishable" phrasings correctly.
- Python: extend `test_automl_system.py` — end-to-end fit → all three methods return; markdown parses; JSON schema stable.
- Property: selection is invariant to leaderboard order permutations.

## Phases (session-sized)

1. **P3.1** `SelectionRationale` + `statistically_best_model()` + Rust tests (RED first).
2. **P3.2** `render_benchmark_report()` (md+json) + tests.
3. **P3.3** `ModelCardGenerator::from_automl_result()` + tests.
4. **P3.4** PyO3 exposure (3-step rule) + Python tests + docs example in README ("Intelligent selection" section with a real run's output).
5. **P3.5** Release: CHANGELOG, version bump (1.1.0), full CI green, PyPI publish — **Trey gate before publish**. Also refreshes the repo's "maintained" signal (last real commit Apr 30).

## Non-goals (v1)

Meta-learning upgrade of `recommend.rs` (v2, uses `metafeatures.rs`+`warmstart.rs`); GPU CI; `clippy::unwrap_used` triage of `automl`/`cv`/etc. (separate hygiene track); `ferroml.model_selection` sklearn-parity gap (separate); `ort` RC pin.

## Done bar (maps to career spec P3)

Auto-select with statistical guarantees ✔ (1) · honest benchmark ✔ (2) · model card ✔ (3) · working v1 slice on PyPI ✔ (P3.5). Estimated total ≈ 150 lines core Rust + bindings + tests — well inside the Aug 29 budget; realistic finish mid-July→early-August on Opus via subagent-driven development.
