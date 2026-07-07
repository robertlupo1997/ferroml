# FerroML architecture map — P3 (intelligent AutoML v1) groundwork

**Date:** 2026-07-06 · **Produced by:** code-explorer agent (report-only pass, no edits), dispatched per the career-direction spec's P3 sequence (`~/experiments/Ornith/docs/superpowers/specs/2026-07-04-career-direction-design.md`). Saved by the mastermind session; feeds the P3 architecture/spec step. P3 done bar: auto-select + honest benchmark + model card, due **Aug 29**.

## Executive summary

1. Two-crate workspace: `ferroml-core` (Rust, 55+ models, full CASH AutoML) + `ferroml-python` (PyO3, 24 submodules, ~2,425 Python tests).
2. AutoML selection today is pure CV-score ranking — the corrected Nadeau-Bengio t-test is computed and annotated on results but does NOT drive winner selection; `best_model()` returns leaderboard top-1.
3. Auto-select plugin point: `AutoMLResult::statistically_best_model()` in `ferroml-core/src/automl/fit.rs` filtering `competitive_models()` (already BH-corrected) by the existing `AlgorithmComplexity` — pick the simplest non-inferior model. ~20 lines, no new statistical machinery.
4. Honest benchmark: pieces exist (`metrics/comparison.rs`, `stats/multiple_testing.rs`, `stats/bootstrap.rs`); missing only `render_benchmark_report()` on `AutoMLResult` serializing `ModelComparisonResults` to Markdown/JSON (~50 lines; `summary()` already renders a text table).
5. Model card: `model_card.rs` + `HasModelCard` implemented for most models, `PyModelCard` exposed. Missing: `ModelCardGenerator::from_automl_result()` combining static card data with run data (CV score ± std, CI, training time, `DataCharacteristics`) (~80 lines).
6. Biggest gap: the three features are independent stubs with no unified "select winner + benchmark + card" pipeline.
7. Rot: build clean, zero `todo!()`/`unimplemented!()`. But `clippy::unwrap_used` untriaged in `automl`, `cv`, `neural`, `hpo`, `metrics` + 10 more modules (bare panics possible on malformed input).
8. `ort v2.0.0-rc.11` (ONNX) is a release candidate; 5 RustSec advisories suppressed in CI audit — review periodically.
9. `gpu/` feature-gated, not exercised in main CI (benchmark workflow is continue-on-error) — correctness unverified.
10. Python parity gap: `ferroml.model_selection` exposes only `train_test_split` — no `cross_val_score`/`GridSearchCV` despite full Rust `cv/` machinery.

## v1 slice — lowest-friction build order

1. `ferroml-core/src/automl/fit.rs` — `AutoMLResult::statistically_best_model()` (~20 lines)
2. `ferroml-core/src/automl/fit.rs` — `AutoMLResult::render_benchmark_report()` (~50 lines)
3. `ferroml-core/src/model_card.rs` — `ModelCardGenerator::from_automl_result()` (~80 lines)
4. `ferroml-python/src/automl.rs` + `ferroml-python/src/model_card.rs` — expose all three (3-step binding rule in CLAUDE.md: Rust → PyO3 wrapper → `python/ferroml/<submodule>/__init__.py` re-export)

Stretch (v2, not v1): upgrade `recommend.rs` from rule-based to meta-learning-driven using `metafeatures.rs` + `warmstart.rs`.

## Key architecture facts (for the spec author)

- CASH loop: `AutoML::fit()` in `automl/fit.rs` — portfolio (`portfolio.rs`, Balanced/Fast/HighAccuracy presets) → data-aware adaptation → bandit time allocation (`time_budget.rs`: UCB1/EpsilonGreedy/ThompsonSampling) → StratifiedKFold CV per trial → `EnsembleBuilder::build_from_trials()` → pairwise `corrected_resampled_ttest()` with Benjamini-Hochberg → `ModelComparisonResults`.
- Pre-fit recommender: `recommend.rs` — rule-based `DatasetProfile` heuristics, ≤5 ranked suggestions, no fitting.
- Meta-learning: `warmstart.rs` (`MetaLearningStore`) + `transfer.rs` influence initial HPO params only, not algorithm selection.
- Tests: ~4,584 Rust `#[test]` + ~2,425 Python. 6 integration binaries in `ferroml-core/tests/` (adding new ones forbidden by CLAUDE.md): correctness, integration, vs_linfa, adversarial, edge_cases, regression_tests. 6 cargo-fuzz targets. CI: 3 OS × Rust + 3 OS × Python 3.10-3.12, tarpaulin 70/75%, Criterion benchmark regression at 10%, cargo audit/deny, MSRV 1.75.
- `default-members = ["ferroml-core"]` — plain `cargo test` skips the Python crate.
- nalgebra (not ndarray-linalg) for linalg — no OpenBLAS system dep. Default features: parallel, onnx, simd, faer-backend, datasets.
