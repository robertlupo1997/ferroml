# VERIFY-OX — Adversarial Re-Verification of FIXES-OX

Verifier: ox-alpha (independent of the FIXES-OX author).

Method: every number in FIXES-OX was re-derived from the repository's own
artifacts (committed benchmark data, `.pyi` stubs, Rust sources, static
greps) without trusting either AUDIT-OX or FIXES-OX. Where FIXES cites a
line or count, that citation itself was re-checked. A repo-wide sweep for
claims both passes missed was run on top (constructor-kwarg validation of
every Python snippet in user-facing docs against committed `.pyi` stubs,
plus pattern searches for superseded claims).

Verdict key: **SUPPORTED** = new wording matches independently re-derived
values. **STILL WRONG** = new wording remains false. **OVERCORRECTED** = a
true claim was deleted/weakened without cause. **NEW ERROR** = the fix
itself introduced an inaccuracy.

---

## README.md (root)

### Rows 3, 3/5 (badge "collected"; L10 wording; L334 table)
**SUPPORTED** (with one caveat, see Q3). Badge now reads `tests-7%2C600%2B%20collected`; L334 reads `31 ignored (21 slow AutoML system tests)`.

Independently re-derived:
- Total `#[test]`: **4,602** (`grep -rn '#\[test\]' ferroml-core/src ferroml-core/tests | wc -l`) ✓
- `#[ignore]`: **31 total**, 21 in `ferroml-core/src/testing/automl.rs`, remaining 10 spread over recommend.rs (1), models/compliance (3), preprocessing/compliance (3), testing/properties.rs (1), tests/adversarial.rs (1), tests/correctness.rs (1) ✓
- Python static defs: **2,428** ✓

Proof:
```sh
grep -n 'tests-' README.md | head -1 && grep -rn '#\[ignore' ferroml-core/src ferroml-core/tests | wc -l && grep -rc '#\[ignore' ferroml-core/src/testing/automl.rs
```

### Row 8 ("100% correctness" → defaults-mismatch wording)
**SUPPORTED.** docs/cross-library-benchmark.md really does show differing scores for identical N (RFRegressor 0.4494 vs 0.7863 at L26-27; RFClassifier 0.8910 vs 0.9300 at L30-31), and the script defect is real (see benchmark-defect section below).

```sh
grep -n 'RandomForestRegressor' docs/cross-library-benchmark.md | head -4
```

### Rows 10, 11, 13 (L304 performance paragraph)
**SUPPORTED.** Recomputed every ferroml/sklearn ratio from the summary table of docs/cross-library-benchmark.md:
- Worst slowdown: KNeighborsClassifier **fit** 2.0ms vs 0.5ms = **4.17x** ≈ "~4.2x" ✓ (HistGBTReg fit 2.17x, KNN predict 2.13x — nothing larger)
- sklearn faster on HistGBT (fit AND predict) and KNN predict ✓
- Linear-model fits: LogReg 16.8 vs 8.0ms (~2.08x slower), LinReg ~1.9x, Ridge ~1.7x — "within ~2x" holds approximately ✓; PCA predict/fit parity (1.01x) ✓
- Cited cross-references verified: docs/validation-report.md:164 KNN predict **45.83x**, :153 DBSCAN **25.32x**; docs/benchmarks.md:42 SVC RBF **6.20x** ✓

```sh
grep -E 'KNeighborsClassifier|45\.83|25\.32|SVC RBF' docs/validation-report.md docs/benchmarks.md | head -6
```

### Row 14 (8 → 10 WGSL shaders)
**SUPPORTED.** `grep -c 'SHADER: &str' ferroml-core/src/gpu/kernels.rs` = **10**.

### Row 16 (12 → 11 sparse models)
**SUPPORTED.** Exactly **11** impls of the pure trait: `grep -rn 'traits::SparseModel for' ferroml-core/src | wc -l` = 11 (LogisticRegression, Ridge, LinearSVC, LinearSVR, 2xKNN, NearestCentroid, 4xNB). Note for future auditors: a broad grep `impl.*SparseModel for` returns 23 because it also matches the *different* trait `PipelineSparseModel` (11) and a test mock — the 11 figure is correct for the CSR model API.

### Row 17 (26 → 31 ignored)
**SUPPORTED** — see rows 3/5 above.

### Row 18 (807 breakdown)
**SUPPORTED.** Per-file `#[test]` counts re-derived: edge_cases.rs **238**, correctness.rs **304**, adversarial.rs **89**, integration.rs **103**, regression_tests.rs **17**, vs_linfa.rs **56**; sum = **807** exactly as printed.

```sh
for f in ferroml-core/tests/*.rs; do echo "$f $(grep -c '#\[test\]' $f)"; done
```

### Row 19 (37 Python + 8 → 7 Rust composition tests)
**SUPPORTED.** test_frankenstein.py has **37** `def test_`. tests/correctness.rs has 304 total minus the 297 preceding... re-derived directly: `awk` split confirmed 7 `#[test]` after the layer5_frankenstein marker region (304 total − 297 pre-marker = 7; independent awk recount agrees).

```sh
grep -c 'def test_' ferroml-python/tests/test_frankenstein.py && grep -c '#\[test\]' ferroml-core/tests/correctness.rs
```

### Row 21 (~99% coverage removed; 6 → 7 decomposition)
**SUPPORTED, not overcorrected.** `ferroml-python/python/ferroml/decomposition/__init__.py` `__all__` has exactly **7** entries (PCA, IncrementalPCA, TruncatedSVD, LDA, QDA, FactorAnalysis, TSNE); explainability `__all__` = **37** ✓. No artifact anywhere records a coverage percentage; deletion is justified.

### Row 22 (13 → 18 decision_function)
**SUPPORTED both places.** `grep -rn 'fn decision_function' ferroml-python/src --include='*.rs' | wc -l` = **18**; independent `.pyi` count `grep -rh 'def decision_function(' ferroml-python/python/ferroml/*/__init__.pyi ...` = **18** ✓.

### Row 24 (58 → fixture inventory)
**SUPPORTED.** Independently walked benchmarks/fixtures/: 8 model-fixture files containing **17** model cases (adaboost 2, gradient_boosting 2, knn 2, logistic_regression 1, naive_bayes 3, random_forest 2, regularized 3, svm 2), preprocessing.json = **11** transformer fixtures, adversarial/ = **12** files. All three numbers match the new wording.

```sh
find benchmarks/fixtures -type f | wc -l && python3 -c "import json;d=json.load(open('benchmarks/fixtures/preprocessing.json'));print(len(d))"
```

### Row 25 (per-category table annotated, numbers kept)
**SUPPORTED and correctly dispositioned.** Current IDs do not reproduce 102 clustering / 49 neural etc., and inventing substitutes would be fabrication. The inserted note is present between the two tables (README.md ~L376). Not overcorrection: deleting the historical table was not required once annotated.

### Row 27 (420-trial excerpt)
**SUPPORTED.** New hedged sentence present at README.md:225. No artifact stores the AutoML run output; the hedge is accurate.

### Row 34 (118 roundtrip → 88 onnx unit tests)
**SUPPORTED.** `grep -rn '#\[test\]' ferroml-core/src/onnx/*.rs | wc -l` = **88**.

### Row 35 (Known Limitations paragraph)
**NEW ERROR INTRODUCED.** README.md:450 now reads "the largest gap is ~4.2x (**KNN predict**)". The 4.2x magnitude is real, but it is the KNN **fit-time** gap (ferroml 2.0ms vs sklearn 0.5ms; fit-speedup row 0.24x). The KNN **predict** gap in the same recorded run is only 2.13x (27.5ms vs 12.9ms). The parenthetical misattributes the number. (README.md:304, the row-10 edit, makes no attribution and is fine.)

Proof:
```sh
grep -n 'KNeighborsClassifier' docs/cross-library-benchmark.md | head -2   # fit=2.0 pred=27.5 vs fit=0.5 pred=12.9
sed -n '82p' docs/cross-library-benchmark.md                               # 0.24x is under "Fit-Time Speedup"
sed -n '450p' README.md
```

## ferroml-python/README.md

### Row 39 ("2-40x predict" rewrite)
**PARTIALLY SUPPORTED — one clause STILL WRONG.**
- "up to ~7x faster … on predict": recomputed all predict ratios from the summary table; max is RandomForestRegressor 18.6/2.6 = **7.15x** ✓ (next: RFClf 6.46x, KMeans 4.0x).
- "slower on HistGBT and KNN predict at scale" ✓ (HGBR 2.10x, HGBC 2.05x, KNN 2.13x slower).
- **"up to 9x faster on fit for tree/ensemble models"**: the same sentence anchors to docs/cross-library-benchmark.md (2026-03-27), whose fit-speedup table shows RandomForestRegressor at **19.87x** vs sklearn (line 76). "Up to 9x" contradicts the cited source. The old document supported 9x; after re-anchoring the sentence to the new file, the retained figure became false (an understated maximum). Should read e.g. "up to ~20x (RandomForest fit; affected-comparison figures pending a rerun)" or drop the clause.

Proof:
```sh
sed -n '9p' ferroml-python/README.md && sed -n '76p' docs/cross-library-benchmark.md
```

### Row 43 ("predict universally" softening)
**SUPPORTED.** From the recorded data FerroML is strictly faster on predict for 7 of 17 workloads (RF x2, GB x2, SVC, GaussianNB, KMeans), at parity on 7, slower on 3 — "faster on predict for most models" is defensible; the named exceptions are correct.

### Row 42 (v0.2.0 supersession note)
**SUPPORTED.** Note present at ferroml-python/README.md:63; table left intact as historical record; the table's values do trace to docs/benchmark-vs-sklearn.md.

### Row 46 (ensemble/trees module lists)
**SUPPORTED.** `ferroml-python/python/ferroml/trees/__init__.py` exports exactly the 8 listed classes including RandomForest*; ensemble row no longer lists RandomForest.

### Row 49 (California Housing → Linnerud)
**SUPPORTED.** datasets docstring lists iris/wine/diabetes/linnerud only; `grep -rni california ferroml-python/python/ferroml/` returns nothing.

### Row 50 (13 → 18 estimators)
**SUPPORTED** — see row 22.

### Row 51 (stale 5,650+ test counts)
**SUPPORTED.** New wording mirrors the corrected root README; "489+" holds against 461 Python comparison defs + 56 vs_linfa = 517 ≥ 489 (re-derived).

### Row 52 (Pipeline.score() → predict())
**SUPPORTED.** ferroml-python/src/pipeline.rs defines `fn predict` (L312, L1323) and no score method; trees .pyi confirms predict exists.

## docs/*

### Row 67 (ferroml-core = "1.0")
**SUPPORTED.** Cargo.toml workspace version 1.0.2; `"1.0"` requirement covers it.

### Row 68 (HistGradientBoosting example)
**SUPPORTED.** trees/__init__.pyi signature is `(max_iter, learning_rate, loss, max_depth, max_leaf_nodes, max_bins, min_samples_leaf, l2_regularization, random_state)` — every kwarg in the new example exists; n_estimators/monotonic_constraints/early_stopping/validation_fraction genuinely absent; monotonic constraints confirmed in ferroml-core/src/models/hist_boosting.rs:648.

### Row 66 (10% → 15% threshold)
**SUPPORTED.** ferroml-core/benches/baseline.json:214 `"regression_threshold_percent": 15`.

### Row 70 (ROADMAP banner)
**SUPPORTED.** Banner reads v1.0.2 with the same corrected counts; the residual "`decision_function` on 13 classifiers" at docs/ROADMAP.md:25 sits inside the v0.3.0 release-history section covered by the banner's "dated history" clause — acceptable.

### Row 62 (validation-report erratum), Row 64 (accuracy-report Plan M erratum)
**SUPPORTED.** Both errata present; the 245 comparison-test-function count reproduces exactly (`cat ferroml-python/tests/test_comparison_*.py ferroml-python/tests/test_performance_comparison.py | grep -c 'def test_'` = 245).

### Row 63 (accuracy-report Test Infrastructure table)
**SUPPORTED.** Every replaced value re-derived: 4,602 total ✓, 3,795 src (`grep -rn '#\[test\]' ferroml-core/src | wc -l`) ✓, 807 integration ✓, 17+11 fixtures ✓, 2,428 static / 3,039 collected Python ✓, **85** `test_*.py` files ✓ (`ls ferroml-python/tests/test_*.py | wc -l`).

### Row 65 (feature-parity-scorecard banner)
**SUPPORTED.** Banner counts reproduce from committed stubs today: exact `def score(` across all `__init__.pyi` = 57, of which 1 is `Recommendation` (not an ML model) → **56** ✓; `def partial_fit(` = **10** ✓; decision_function = **18** ✓.

### Row 61, 71, 72 (supersession/historical banners)
**SUPPORTED.** All four banners present (docs/benchmarks.md:4, benchmark-results-baseline.md:3, benchmark-results-optimized.md:3, benchmark-vs-sklearn.md:38). One nit: the optimized-results banner dates commit 55ae057 as "(2026-03-09)", but git records author and commit date **2026-03-08** for both 55ae057 and 78b72fd (`git log --format='%h %ad %cd' --date=short -1 55ae057`). The body's own pre-existing date line was repeated unverified. Cosmetic.

## Benchmark-defect section (script changes)

**SUPPORTED.**
- forest.rs:903 `Fraction(1.0/3.0)` default ✓; classifier fallback `Sqrt` at :663-667 ✓; regressor fallback at :1235-1237 ✓; PyO3 wrapper default `"sqrt"` with only sqrt/auto/log2/none/all accepted (trees.rs:1112-1129; the FIXES text omits the undocumented aliases "auto"/"all", immaterial to the conclusion) ✓.
- Script diff matches the description exactly; `python3 -m py_compile scripts/benchmark_cross_library.py` passes ✓.
- "Comparisons already equal elsewhere": tree.rs max_features `None` (=all) both sides ✓; boosting.rs subsample 1.0 ✓; HistGB train size 0.8×10000 = 8000 < 10000 so sklearn `early_stopping='auto'` stays off ✓; KMeans ferroml default n_init=10 (kmeans.rs) equals pinned sklearn n_init=10 ✓.
- Not overcorrected: leaving the recorded tables in docs/cross-library-benchmark.md untouched preserves provenance; the pending-rerun flags are in place.

---

## Q1 — Claims STILL FALSE, missed by BOTH passes

Yes — three, all found by this pass:

1. **docs/correctness-report.md:412 (also :42, :451): "No model exceeds 5x slower than sklearn."** This is the exact generalization AUDIT-OX marked FALSE in two READMEs, but it survives untouched in a doc whose header claims currency ("v1.0.2 Correctness Audit Report", "updated 2026-07-12") and which FIXES-OX cites as its source of authority for test totals. It is contradicted by the repository's own data: docs/validation-report.md:164 (sklearn 45.83x faster on KNN predict), :153 (DBSCAN 25.32x), docs/benchmarks.md:42 (SVC RBF 6.20x). Needs the same erratum/banner treatment the other perf docs got.
   ```sh
   sed -n '412p' docs/correctness-report.md && grep -n '45.83' docs/validation-report.md
   ```
2. **docs/correctness-report.md:49 and :428 carry stale suite internals**: "Library unit tests 3,224 … (26 ignored)" vs static reality 3,795 src tests / 31 ignores; the Final-suite table's "Edge case tests 553 / Adversarial 88" contradicts the actual per-file counts (238/89) used in the corrected README. The grand totals (4,602/3,039) are fine; the sub-rows are not, and the file got neither a fix nor a note.
   ```sh
   sed -n '47,57p' docs/correctness-report.md && grep -rc '#\[test\]' ferroml-core/tests/edge_cases.rs ferroml-core/tests/adversarial.rs
   ```
3. **docs/user-guide.md:1169: `AutoMLConfig(multiple_testing_correction="benjamini_hochberg", ...)`** — no such parameter. The class accepts `multiple_testing` (python/ferroml/automl/__init__.pyi:25; ferroml-python/src/automl.rs:187). The example raises TypeError as written. Found by validating every constructor call in user-facing docs against the committed stubs; this was the only mismatch.
   ```sh
   sed -n '1168,1171p' docs/user-guide.md && sed -n '25p' ferroml-python/python/ferroml/automl/__init__.pyi
   ```

## Q2 — Did any fix introduce a NEW inaccuracy?

Yes, two:

1. **README.md:450 "~4.2x (KNN predict)"** — the magnitude is right, the attribution is wrong: 4.2x is the KNN **fit** gap; KNN predict is 2.13x in the same data. Introduced by the Row-35 edit (and repeated in FIXES-OX's own Row-35 text).
2. **ferroml-python/README.md:9 "up to 9x faster on fit for tree/ensemble models"** — kept verbatim while the sentence's anchor moved to docs/cross-library-benchmark.md, which records 19.87x RF-regressor fit speedup vs sklearn. Against its own new citation, "up to 9x" is false.

(Minor third: the benchmark-results-optimized banner repeats the body's "2026-03-09" date for commit 55ae057, which git dates 2026-03-08.)

## Q3 — Is any corrected number itself unreproducible from this repository today?

Two, partially:

1. **"7,600+ collected (… 3,000+ Python)"** (badge, root README:10/334, ferroml-python/README:114, ROADMAP banner, accuracy-report table). Reproducible today only as: 4,602 static `#[test]` attrs (verified) plus docs/correctness-report.md's *recorded claim* of 3,039 collected Python. `pytest --collect-only` needs a built extension; none can be built here without a full maturin release build. The wording honestly names the source, but the 3,000+ half is citation, not derivation — and its cited host (correctness-report.md) carries the stale sub-counts listed in Q1.
2. **"pending a rerun" figures** (README:292 and the RF/xgboost/lightgbm/LogReg rows): by design, no number in this repo currently substantiates the post-fix comparisons; they cannot be reproduced until scripts/benchmark_cross_library.py is rerun against a built extension. Correctly flagged, but a reader should treat every affected performance figure as provisional.

Everything else this pass re-derived — 10 shaders, 11 sparse impls, 31/21 ignores, 807 breakdown, 37+7 frankenstein, 7 decomposition, 37 explainability, 18 decision_function, 88 onnx, 245/461+56 comparison tests, 85 test files, 17+11+12 fixtures, 56/10 parity counts, 4,602/3,795/2,428, 15% threshold — is statically reproducible today.

VERDICT: NOT-SAFE — the fixes are mostly sound and evidence-based, but they introduce two new inaccuracies (README:450 "KNN predict" misattribution; the "up to 9x fit" clause contradicted by its own cited source), leave a live ">5x slower" falsehood and stale suite counts standing in docs/correctness-report.md, and miss a broken AutoMLConfig example in docs/user-guide.md.
