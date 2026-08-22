# FIXES-OX — Corrections Applied from AUDIT-OX

Scope: every AUDIT-OX row marked **FALSE** or **UNPROVABLE** was either fixed or
explicitly dispositioned below. SUPPORTED rows were left untouched. All file:line
citations below were re-verified against the working tree before this file was
written (`grep -n` / `sed -n`, not estimates).

Policy notes:

- Where a number depended on an unrecorded run (test totals, benchmark figures),
  it was replaced only with statically verifiable counts, marked as pending a
  rerun, or covered by a dated supersession/erratum banner in the document itself.
- The benchmark-configuration defect is summarized in its own section at the end.

---

## README.md (root)

### Row 3 (badge "tests-7,600+ passing") — UNPROVABLE — FIXED
- README.md:4 (badge URL text)
- Old: `[![Tests](https://img.shields.io/badge/tests-7%2C600%2B%20passing-brightgreen)](...)`
- New: `[![Tests](https://img.shields.io/badge/tests-7%2C600%2B%20collected-brightgreen)](...)`
- Rationale: "passing" requires a recorded test run, which no artifact provides;
  "collected" is backed by docs/correctness-report.md (4,602 Rust + 3,039 Python).

### Rows 3, 5 ("7,600+ tests passing (…)", "3,000+ Python") — UNPROVABLE — FIXED
- README.md:10 and README.md:334 (Project Status table)
- Old (L10): `7,600+ tests passing (4,600+ Rust + 3,000+ Python)`
- New (L10): `7,600+ tests collected (4,600+ Rust + 3,000+ Python; docs/correctness-report.md)`
- Old (L334): `| **Tests** | 7,600+ passing (4,600+ Rust + 3,000+ Python), 0 failing, 26 ignored (slow AutoML system tests) |`
- New (L334): `| **Tests** | 7,600+ collected (4,600+ Rust + 3,000+ Python; docs/correctness-report.md), 31 ignored (21 slow AutoML system tests) |`
- Rationale: verified `grep -rE '^\s*def test_' ferroml-python/tests/*.py | wc -l` = 2,428 static defs;
  the 3,000+ magnitude rests on correctness-report.md's collection record, so the wording now says
  "collected". "0 failing" deleted (no artifact). Ignored count corrected 26 -> 31 (21 AutoML),
  verified by `grep -rn '#\[ignore' ferroml-core/src ferroml-core/tests | wc -l`.

### Row 8 ("100% correctness") — FALSE — FIXED
- README.md:292
- Old: `All benchmarks produce matching predictions (100% correctness). Results from cross-library benchmark (2026-03-27, 10K samples):`
- New: `Benchmarks train on identical data per workload. In the recorded run (2026-03-27, 10K samples), test scores match for most models but differ where model defaults differed — most visibly RandomForestRegressor, whose comparison pinned different \`max_features\` values on the two sides (fixed in scripts/benchmark_cross_library.py; affected numbers pending a rerun):`
- Evidence: docs/cross-library-benchmark.md RF regressor row (ferroml 0.4494 vs sklearn 0.7863);
  root cause confirmed in scripts/benchmark_cross_library.py and pinned identically on both sides
  (see benchmark-defect section).

### Rows 10, 11, 13 ("5x slower", "predict universally", "parity or better on linear algebra") — FALSE — FIXED
- README.md:305
- Old: `No model is more than 5x slower than sklearn. FerroML wins on **predict universally** (zero Python overhead), on **tree/ensemble models** (Rayon parallel construction), and achieves **parity or better on linear algebra** thanks to faer SVD and Cholesky solvers. Full results ...`
- New: `In the recorded cross-library benchmark no measured workload is more than ~4.2x slower than sklearn, though other repo benchmarks show larger gaps for KNN predict (45.8x at n=10K), DBSCAN fit, and SVC RBF (docs/validation-report.md, docs/benchmarks.md). FerroML wins on predict for most models but not universally — sklearn is faster on HistGBT and KNN predict at scale — and on tree/ensemble fit (Rayon parallel construction); on linear-model fits it is within ~2x of sklearn and at parity on PCA. Full results ...`
- Evidence: docs/cross-library-benchmark.md KNN fit 0.24x (L82); HistGBT/KNN predict slower (rows at L43-52);
  LinearRegression/Ridge/LogisticRegression fit 0.48x-0.59x (L70-73); docs/validation-report.md KNN predict 45.83x,
  DBSCAN 25.32x; docs/benchmarks.md SVC RBF 6.20x.

### Row 14 ("8 WGSL shaders") — FALSE — FIXED
- README.md:311
- Old: `(8 WGSL shaders with auto CPU/GPU dispatch)` → New: `(10 WGSL shaders with auto CPU/GPU dispatch)`
- Verified: `grep -c 'SHADER: &str' ferroml-core/src/gpu/kernels.rs` = 10.

### Row 16 ("12 models sparse") — FALSE — FIXED
- README.md:312 and README.md:437
- Old: `CsrMatrix operations for 12 models` / `12 models support sparse input (CSR format)`
- New: `...for 11 models` / `11 models support sparse input (CSR format)`
- Verified: 11 `traits::SparseModel for` implementations in ferroml-core/src.

### Row 17 ("26 ignored") — FALSE — FIXED (see Rows 3/5 edit of the same table row)
- Old: `26 ignored (slow AutoML system tests)` → New: `31 ignored (21 slow AutoML system tests)`
- Verified: 31 total `#[ignore]`, 21 in ferroml-core/src/testing/automl.rs.

### Row 18 (correctness-test breakdown) — FALSE — FIXED
- README.md:335
- Old: `1,140 (553 edge_cases + 304 correctness + 88 adversarial + 103 integration + 36 regression + 56 vs_linfa)`
- New: `807 (238 edge_cases + 304 correctness + 89 adversarial + 103 integration + 17 regression + 56 vs_linfa)`
- Verified per file with `grep -rc '#\[test\]' ferroml-core/tests/*.rs`; sum = 807.

### Row 19 ("37 Python + 8 Rust") — FALSE — FIXED
- README.md:337
- Old: `37 Python + 8 Rust composition tests (...)` → New: `37 Python + 7 Rust composition tests (...)`
- Verified: 37 `def test_` in test_frankenstein.py; 7 `#[test]` after layer5_frankenstein in
  ferroml-core/tests/correctness.rs (awk from line 7636).

### Row 21 ("~99% coverage", "6 decomposition") — UNPROVABLE/FALSE — FIXED
- README.md:340
- Old: `~99% coverage (55+ models, 22+ preprocessors, 6 decomposition, 37 explainability)`
- New: `55+ models, 22+ preprocessors, 7 decomposition classes, 37 explainability tools (coverage report not yet published)`
- Verified: ferroml.decomposition `__all__` has exactly 7 entries. The unverifiable percentage was
  deleted rather than softened.

### Row 22 ("decision_function on 13 classifiers") — FALSE — FIXED
- README.md:341
- Old: `` `decision_function` on 13 classifiers`` → New: `` `decision_function` on 18 estimators``
- Verified: `grep -rn 'fn decision_function' ferroml-python/src --include='*.rs' | wc -l` = 18.

### Row 24 ("58 fixture-based comparisons") — UNPROVABLE — FIXED
- README.md:367
- Old: `validated against scikit-learn with 58 fixture-based comparisons and 489+ cross-library correctness tests:`
- New: `validated against scikit-learn on committed fixture data (benchmarks/fixtures/: 17 model fixtures, 11 transformer fixtures, 12 adversarial fixtures) and 489+ cross-library correctness tests:`
- Verified by walking the committed JSON fixtures (17 model cases across 8 files; preprocessing.json
  holds 11 transformer fixtures; benchmarks/fixtures/adversarial/ holds 12 files). No artifact
  supports 58, so the count was replaced by the verifiable inventory.

### Row 25 (per-category accuracy table 17/15/102/49/101) — UNPROVABLE — ANNOTATED
- README.md:369-375 — a note was inserted between the accuracy table and the tolerance table:
  `*Note: the per-category counts in this table come from an earlier suite snapshot and could not be reproduced from current test IDs; they are pending regeneration from collected test IDs. The committed fixture data covers 17 model and 11 preprocessing fixtures.*`
- The original numbers were left in place inside the annotated block because no current artifact can
  regenerate them (test_clustering.py has 63 defs vs claimed 102; test_neural.py 70 vs 49); inventing
  replacements was ruled out.

### Row 27 ("Real output … 420 trials") — UNPROVABLE — FIXED
- README.md:225-226
- Old: `Real (excerpted) output from the run above — 420 trials on a 250 × 8 synthetic classification dataset:`
- New: `Illustrative example output (the excerpt predates v1.0.2; the run that produced it is not stored in this repository, so the trial count is unverified):`

### Row 34 ("all 118 roundtrip tests pass") — UNPROVABLE — FIXED
- README.md:441
- Old: `The API is stable and all 118 roundtrip tests pass, but users should be aware this is a pre-release dependency.`
- New: `The API is stable — ferroml-core/src/onnx carries 88 unit tests — but users should be aware this is a pre-release dependency.`
- Verified: 88 `#[test]` under ferroml-core/src/onnx/*.rs. The unsubstantiated 118/"all pass" claim
  was deleted, not softened into a vaguer passing claim.

### Row 35 ("No model is more than 5x slower", Known Limitations) — FALSE — FIXED
- README.md:449
- Old: `FerroML achieves competitive or better performance than scikit-learn across most algorithms. No model is more than 5x slower. Highlights: ...`
- New: `FerroML achieves competitive or better performance than scikit-learn across most algorithms. Within the cross-library benchmark workloads the largest gap is ~4.2x (KNN predict); other repo benchmarks show larger gaps for KNN predict at scale, DBSCAN fit, and SVC RBF (docs/validation-report.md, docs/benchmarks.md). Highlights: ...`
- The supported highlights sentence (row 36) was preserved unchanged.

## ferroml-python/README.md

### Row 39 ("2-40x faster on predict") — FALSE — FIXED
- ferroml-python/README.md:9
- Old: `It's 2-40x faster than scikit-learn on predict and up to 9x faster on fit for tree/ensemble models.`
- New: `On the latest committed cross-library benchmark (2026-03-27, docs/cross-library-benchmark.md) it is up to ~7x faster than scikit-learn on predict for many models — though slower on HistGBT and KNN predict at scale — and up to 9x faster on fit for tree/ensemble models.`
- Note: AUDIT-OX's own suggested replacement said "~19x faster on predict"; recomputing from the
  committed data shows the maximum *predict* speedup is RandomForestRegressor 7.15x (19.87x is a
  *fit*-time figure, cross-library-benchmark.md:76). The fix uses ~7x accordingly.

### Row 43 ("faster on predict universally") — FALSE — FIXED
- ferroml-python/README.md:73
- Old: `FerroML is faster on **predict universally** (zero Python overhead) and on **fit for tree/ensemble models** (Rayon parallel construction). scikit-learn wins on fit for LAPACK/MKL-backed linear algebra.`
- New: `FerroML is faster on **predict for most models**, though scikit-learn wins on HistGBT and KNN predict at scale, and FerroML is faster on **fit for tree/ensemble models** (Rayon parallel construction). scikit-learn wins on fit for LAPACK/MKL-backed linear algebra.`

### Row 42 staleness (v0.2.0 performance table intro) — SUPERSESSION NOTE ADDED
- ferroml-python/README.md:63
- Old: `All benchmarks produce matching predictions. Speedup >1x = FerroML is faster.`
- New: `Figures below are from the v0.2.0 run (2026-03-11, docs/benchmark-vs-sklearn.md); the newer cross-library benchmark (2026-03-27, docs/cross-library-benchmark.md) supersedes them. Test scores match for most models but differ where model defaults differed. Speedup >1x = FerroML is faster.`
- The table values themselves stay (row 42 verdict SUPPORTED against their source); a rerun against
  v1.0.2 is needed to refresh them.

### Row 46 ("ensemble: RandomForest …") — FALSE — FIXED
- ferroml-python/README.md:81 and :80
- ensemble row: removed `RandomForest, ` (RandomForest* lives in `ferroml.trees`).
- trees row: `DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingClassifier/Regressor, HistGradientBoostingClassifier/Regressor`
  became `DecisionTreeClassifier/Regressor, RandomForestClassifier/Regressor, GradientBoostingClassifier/Regressor, HistGradientBoostingClassifier/Regressor`
  (verified against trees/__init__.py exports).

### Row 49 ("California Housing") — FALSE — FIXED
- ferroml-python/README.md:98
- Old: `` | `datasets` | Iris, Diabetes, Wine, California Housing, synthetic generators | ``
- New: `` | `datasets` | Iris, Diabetes, Wine, Linnerud, synthetic generators | ``
- Verified: datasets/__init__.py exposes load_iris/load_wine/load_diabetes/load_linnerud; no
  California loader exists.

### Row 50 ("decision_function() on 13 classifiers") — FALSE — FIXED
- ferroml-python/README.md:107
- Old: `- \`decision_function()\` on 13 classifiers` → New: `- \`decision_function()\` on 18 estimators`

### Row 51 (stale v0.3.x test counts) — FALSE — FIXED
- ferroml-python/README.md:114
- Old: `5,650+ tests passing (3,550+ Rust + 2,100+ Python), validated against ... with 200+ cross-library correctness tests.`
- New: `7,600+ tests collected (4,600+ Rust + 3,000+ Python; docs/correctness-report.md), validated against scikit-learn, scipy, xgboost, lightgbm, and statsmodels with 489+ cross-library correctness tests.`
- Wording matches the corrected root README (collected, not passing); 489+ is the audited root figure
  (517 cross-library tests counted >= 489).

### Row 52 (Pipeline.score()) — FALSE — FIXED
- ferroml-python/README.md:45
- Old: `score = pipe.score(X_test, y_test)` → New: `y_pred = pipe.predict(X_test)`
- Pipeline has no score(); predict() exists (ferroml-python/src/pipeline.rs), so the example runs.

## docs/*

### Row 67 (ferroml-core = "0.1") — FALSE — FIXED
- docs/user-guide.md:74: `ferroml-core = "0.1"` → `ferroml-core = "1.0"` (Cargo.toml workspace version 1.0.2).

### Row 68 (HistGradientBoosting example) — FALSE — FIXED
- docs/user-guide.md:335-347
- Old constructor kwargs `n_estimators=100, max_depth=6, learning_rate=0.1, monotonic_constraints=[1,0,-1], early_stopping=True, validation_fraction=0.1`
- New:
  ```python
  model = HistGradientBoostingRegressor(
      max_iter=100,       # the Python class uses max_iter, not n_estimators
      max_depth=6,
      learning_rate=0.1,
  )
  model.fit(X_train, y_train)
  ```
  plus an inline note that monotonic/interaction constraints live in the Rust core
  (ferroml-core/src/models/hist_boosting.rs) and early stopping / validation_fraction are not
  parameters of the Python class. Verified against trees/__init__.pyi signature.

### Row 66 ("Threshold: 10% regression triggers CI failure") — FALSE — FIXED
- docs/performance.md:191
- Old: `2. **Threshold**: 10% regression triggers CI failure`
- New: `2. **Threshold**: 15% regression flags a benchmark regression (\`regression_threshold_percent\` in ferroml-core/benches/baseline.json)`
- Verified: baseline.json:214 `"regression_threshold_percent": 15`.

### Row 70 (ROADMAP status banner v0.3.1) — FALSE — FIXED
- docs/ROADMAP.md:5-7
- Banner updated to `## Current Status: v1.0.2`; counts updated to 4,600+ Rust `#[test]` attributes +
  3,000+ Python collected (7,600+ total; docs/correctness-report.md); the unverifiable
  "Python binding coverage is ~99%" clause was deleted; release entries below remain as dated history.

### Row 62 (validation-report 281/265 inconsistency) — FALSE — ERRATUM ADDED
- docs/validation-report.md:5 (inserted after the title block)
- An erratum states the report's totals are internally inconsistent (281 vs its own reproduce-block
  265) and disagree with accuracy-report.md (279/265); the current suite has 245 comparison test
  functions (verified: `cat ferroml-python/tests/test_comparison_*.py ferroml-python/tests/test_performance_comparison.py | grep -c 'def test_'`); counts must be regenerated before citing.
- Original body text preserved; historical numbers cannot be reconciled from artifacts.

### Row 63 (accuracy-report Test Infrastructure stale table) — FALSE — FIXED WITH VERIFIED COUNTS
- docs/accuracy-report.md:145-151 (+ note inserted above the table)
- Old rows (~3,400 Rust / ~2,930 src / ~480 integration / 58 sklearn / ~1,385 Python / 35 files)
  replaced with verified static counts: 4,602 total / 3,795 src / 807 integration /
  17+11 fixtures / 2,428 static Python defs (3,039 collected per correctness-report) / 85 files.
- A note marks the update as part of the v1.0.2 claims audit.

### Row 64 (Plan M 279 vs 265 self-inconsistency) — FALSE — ERRATUM ADDED
- docs/accuracy-report.md:33 (after the Plan M heading)
- Erratum documents the 279/265 disagreement and validation-report.md's 281/265; directs readers to
  regenerate per-phase counts from collected test IDs. Historical numbers left untouched.

### Row 65 (feature-parity-scorecard stale parity figures) — FALSE — HISTORICAL BANNER ADDED
- docs/feature-parity-scorecard.md:5 (after the generation stamp)
- Banner: pre-v0.3.0 snapshot; current bindings have score() on 56 models, partial_fit() on 10,
  decision_function() on 18 estimators; regenerate via scripts/feature_parity_scorecard.py before citing.
- Regenerating here was not possible (needs a built extension).

### Row 61 (docs/benchmarks.md v0.4.0-dev results) — UNPROVABLE/STALE — SUPERSESSION BANNER ADDED
- docs/benchmarks.md:4 (after the intro line)
- Banner: results recorded at FerroML v0.4.0-dev, raw samples not committed, superseded for current
  claims by docs/cross-library-benchmark.md (2026-03-27). Historical measurements preserved in place.

### Row 71 (benchmark-results-baseline/-optimized) — UNPROVABLE — HISTORICAL BANNERS ADDED
- docs/benchmark-results-baseline.md:3 and docs/benchmark-results-optimized.md:3
- Each got a banner naming its commit/date, noting no machine-readable results are committed, and
  pointing to docs/cross-library-benchmark.md for current claims.

### Row 72 (benchmark-vs-sklearn headline claims) — UNPROVABLE/CONTRADICTED — BANNER ADDED
- docs/benchmark-vs-sklearn.md:38 (after the "All 26 benchmarks produce matching predictions." sentence)
- Banner marks the document as a historical v0.2.0 run superseded by docs/cross-library-benchmark.md
  and states that the newer data contradicts the headline generalizations (HistGBT/KNN predict slower;
  RF-regressor score gap traced to unequal max_features defaults, since fixed in the script).

---

## Benchmark-configuration defect (missed by AUDIT-OX)

### What I found

- ferroml-core/src/models/forest.rs:903 — `RandomForestRegressor::new()` sets
  `max_features: Some(MaxFeatures::Fraction(1.0 / 3.0))` (Breiman randomForest p/3 default; a
  legitimate choice — the defect is the comparison, not this default).
- Nuance the task brief did not cover: the PyO3 wrapper never reaches that default.
  ferroml-python/src/trees.rs:1112 defaults the Python `max_features` to `"sqrt"`, and the accepted
  values are only `"sqrt"`/`"log2"`/`"none"` (trees.rs:1123-1129). Because `"none"` maps to `None`,
  the core's fit-time fallback kicks in: forest.rs:1235-1237 `.unwrap_or(MaxFeatures::Fraction(1.0/3.0))`
  (regressor) and forest.rs:663-667 `.unwrap_or(MaxFeatures::Sqrt)` (classifier). So all-features
  sampling is unreachable from Python entirely — noted as a follow-up bug; NOT changed, per the
  task rule against altering forest.rs defaults.
- scripts/benchmark_cross_library.py pinned `n_estimators`/`max_depth` on both RF sides but NOT
  `max_features`. Effective configuration in the recorded run:
  RandomForestRegressor — ferroml sqrt(p) vs sklearn ALL features (sklearn default max_features=1.0);
  RandomForestClassifier — both sqrt(p) already, but xgboost (colsample 1.0) and lightgbm
  (feature_fraction 1.0) sampled all features while ferroml/sklearn sampled sqrt(p);
  LogisticRegression — ferroml ran unpinned defaults max_iter=100/tol=1e-8 while sklearn was pinned
  to max_iter=1000 (tol 1e-4).
- Checked for the same defect elsewhere in the script and found the comparisons already equal:
  DecisionTree (max_features all on both sides, tree.rs:459/1443), GradientBoosting
  (subsample 1.0 on both sides, boosting.rs:396), HistGradientBoosting (train size <= 8000 < 10000,
  so sklearn's `early_stopping='auto'` stays off), KMeans (ferroml default n_init=10 equals the
  pinned sklearn n_init=10, kmeans.rs:185).

### What I changed (scripts/benchmark_cross_library.py)

- Added `import math`.
- bench_random_forest_regressor / bench_random_forest_classifier:
  - sklearn side now pins `max_features="sqrt"` explicitly (matching FerroML's fixed sqrt behavior);
  - ferroml side now pins `random_state=42` like every other library;
  - xgboost side pins `colsample_bynode=sqrt(n_features)/n_features`;
  - lightgbm rf side pins `feature_fraction_bynode=sqrt(n_features)/n_features`
    (per-split feature sampling, matching RF semantics).
- bench_logistic_regression: ferroml pinned `max_iter=1000, tol=1e-4`; sklearn pinned
  `tol=1e-4` alongside its existing `max_iter=1000`.
- Script compiles: `python3 -m py_compile scripts/benchmark_cross_library.py` passes.

### What still needs a rerun to settle

- Every number derived from the affected comparisons: RandomForestRegressor fit/predict/score,
  RandomForestClassifier xgboost/lightgbm rows, LogisticRegression fit time. Until then:
  - README.md:292 and docs/benchmark-vs-sklearn.md state the affected numbers are pending a rerun;
  - docs/cross-library-benchmark.md (2026-03-27) still contains the unequal-comparison figures;
    it should be regenerated with
    `python scripts/benchmark_cross_library.py --output-md docs/cross-library-benchmark.md`
    once a built extension is available. I did not edit its recorded tables — they are the raw
    output of the old script revision, and rewriting generated data in place would falsify provenance.

## Audit rows intentionally NOT acted on

- Row 32 (ferroml-plugin bundles 5 commands / agent / hooks): claim about an external repository
  (github.com/robertlupo1997/ferroml-plugin); nothing in this repo can verify it. AUDIT-OX itself
  recommends verifying against the plugin repo before release. Text left unchanged.
- Row 59 (PyPI badge): whether a release is published on pypi.org is an external registry fact;
  AUDIT-OX marked it "None (external fact)". Badge links left unchanged.
