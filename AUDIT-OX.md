# FerroML Public-Claims Audit (AUDIT-OX)

Auditor: ox-alpha.

Scope: every verifiable factual claim in `README.md`, `ferroml-python/README.md`, `docs/*.md`, `Cargo.toml` / `pyproject.toml` metadata, and the README badges.

Method notes:

- Every verdict is grounded in a file in this repository. Runtime measurements are UNPROVABLE unless a committed data file or statically countable artifact settles them.
- Rust test counts are static counts of `#[test]` attributes (4,602 total), matching docs/correctness-report.md's collected figure. Python collected counts require a built extension; this worktree has none, so Python 'passing' counts cannot be settled here.
- Benchmark comparisons were checked for configuration parity: the root README cites the 2026-03-27 cross-library benchmark (`docs/cross-library-benchmark.md`); ferroml-python/README's table traces to the older v0.2.0 run (`docs/benchmark-vs-sklearn.md`, 2026-03-11). The newer file supersedes the older one; where they disagree, the newer data decides.
- A claim repeated by several documents is traced to generated data or code, never to the repeating document alone.

Supersession map (stale vs current):

| Stale artifact | Superseded by |
|---|---|
| docs/benchmark-vs-sklearn.md (v0.2.0, 2026-03-11) | docs/cross-library-benchmark.md (2026-03-27) |
| docs/feature-parity-scorecard.{md,json} (2026-03-14, pre-v0.3.0) | current PyO3 bindings + ROADMAP.md Plan U |
| docs/accuracy-report.md Test-Infrastructure table (~v0.x) | docs/correctness-report.md (updated 2026-07-12) + root README |
| docs/ROADMAP.md status banner (v0.3.1) | Cargo.toml 1.0.2 + CHANGELOG.md + correctness-report.md |
| ferroml-python/README.md test counts (5,650+) | root README / correctness-report.md (7,600+ collected) |

---

## README.md (root)

### [1] README.md:10

- **CLAIM:** "**Status: v1.0.2**"
- **LOCATION:** `README.md:10`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** Cargo.toml:15 `version = "1.0.2"` (workspace.package); ferroml-python/pyproject.toml:7 `version = "1.0.2"`; CHANGELOG.md:7 `## [1.0.2] — 2026-07-12`.
- **REPRO:**
  ```sh
  grep -n 'version = "1.0.2"' Cargo.toml ferroml-python/pyproject.toml
  ```
- **FIX:** None needed.

### [2] README.md:10

- **CLAIM:** "55+ ML algorithms"
- **LOCATION:** `README.md:10`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** docs/feature-parity-scorecard.json `"total_ferroml_models": 94` (>= 55). Understates; current bindings export 90+ estimators/transformers.
- **REPRO:**
  ```sh
  python3 -c "import json; print(json.load(open('docs/feature-parity-scorecard.json'))['summary']['total_ferroml_models'])"
  ```
- **FIX:** Optionally update to '90+ models' per scorecard; '55+' is not false.

### [3] README.md:4

- **CLAIM:** Badge "tests-7,600+ passing"
- **LOCATION:** `README.md:4`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** No artifact in the repo records a passing run. Static count supports magnitude (4,602 Rust #[test] attrs; docs/correctness-report.md:53-55 claims 4,602 Rust + 3,039 Python collected = 7,641), but 'passing' can only be shown by running the suites. Artifact needed: `cargo test --all-features` output plus `pytest` run against a built extension (`maturin develop --release`).
- **REPRO:**
  ```sh
  grep -rn '#\[test\]' ferroml-core/src ferroml-core/tests | wc -l   # 4602; full check needs cargo test + pytest
  ```
- **FIX:** Keep badge only if CI publishes the number; otherwise link to CI runs.

### [4] README.md:10,334

- **CLAIM:** "4,600+ Rust" (test count)
- **LOCATION:** `README.md:10,334`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** Exactly 4,602 `#[test]` attributes under ferroml-core/src + ferroml-core/tests; matches docs/correctness-report.md:53 'Rust total 4,602 Collected tests'. Note: includes feature-gated GPU tests that run only with --all-features, which CI uses (.github/workflows/ci.yml:126 `cargo test -p ferroml-core --all-features --release`).
- **REPRO:**
  ```sh
  grep -rn '#\[test\]' ferroml-core/src ferroml-core/tests | wc -l
  ```
- **FIX:** None needed.

### [5] README.md:10,334

- **CLAIM:** "3,000+ Python" (test count)
- **LOCATION:** `README.md:10,334`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** Static grep finds 2,428 `def test_` functions (+64 `@pytest.mark.parametrize` decorators that can expand the collected count). The collected total (docs/correctness-report.md:54 claims 3,039) requires importing the compiled `ferroml` extension, which is not built in this repo state. Artifact needed: `pytest --collect-only` output from a built wheel.
- **REPRO:**
  ```sh
  grep -rE '^\s*def test_' ferroml-python/tests/*.py | wc -l   # 2428
  ```
- **FIX:** Either publish collection output or lower the claim to the statically verifiable 2,400+.

### [6] README.md:10,336

- **CLAIM:** "489+ cross-library correctness tests"
- **LOCATION:** `README.md:10,336`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** 461 `def test_` functions across ferroml-python/tests/test_vs_*.py, test_comparison_*.py, test_cross_library_edge_cases.py plus 56 `#[test]` in ferroml-core/tests/vs_linfa.rs = 517 >= 489.
- **REPRO:**
  ```sh
  cat ferroml-python/tests/test_vs_*.py ferroml-python/tests/test_comparison_*.py ferroml-python/tests/test_cross_library_edge_cases.py | grep -c 'def test_' && grep -c '#\[test\]' ferroml-core/tests/vs_linfa.rs
  ```
- **FIX:** None needed.

### [7] README.md:10

- **CLAIM:** "validated against sklearn/scipy/xgboost/lightgbm/statsmodels/linfa"
- **LOCATION:** `README.md:10`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** Test files exist for each library: test_vs_xgboost.py, test_vs_lightgbm.py, test_vs_statsmodels.py, ferroml-core/tests/vs_linfa.rs, scipy imports in test_statistical_*.py / test_vs_sklearn_gaps.py, plus 19 test_vs_sklearn_*.py files.
- **REPRO:**
  ```sh
  ls ferroml-python/tests/test_vs_*.py ferroml-core/tests/vs_linfa.rs
  ```
- **FIX:** None needed.

### [8] README.md:292

- **CLAIM:** "All benchmarks produce matching predictions (100% correctness)"
- **LOCATION:** `README.md:292`
- **VERDICT:** **FALSE**
- **EVIDENCE:** The cited data file contradicts it: docs/cross-library-benchmark.md:145-152 shows different test scores for the same N — RandomForestRegressor ferroml 0.4494 vs sklearn 0.7863; RandomForestClassifier ferroml 0.8910 vs sklearn 0.9300; DecisionTreeClassifier ferroml 0.8660 vs sklearn 0.8580.
- **REPRO:**
  ```sh
  sed -n '145,152p' docs/cross-library-benchmark.md
  ```
- **FIX:** 'Benchmarks use identical train/test data; test scores match for most models but differ where model defaults differ (e.g., RandomForest, DecisionTree). See docs/cross-library-benchmark.md.'

### [9] README.md:295-303

- **CLAIM:** Benchmark table: "KMeans 0.26x / 3.9x FASTER ... KNN 4.0x"
- **LOCATION:** `README.md:295-303`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** Matches docs/cross-library-benchmark.md fit-time speedups: KMeans 3.90x (L87), StandardScaler 3.44x (L85), GaussianNB 2.72x (L84), DecisionTree 1.14x/1.29x (L64-65), SVC 0.65x -> 1.54x slower (L83), LogReg 0.48x -> 2.1x slower (L73), KNN 0.24x -> ~4.2x slower (L82). Minor: the SVC row is N=5000, not 10K as the header says.
- **REPRO:**
  ```sh
  sed -n '73p;82,87p' docs/cross-library-benchmark.md
  ```
- **FIX:** Annotate the SVC row with N=5000 or drop it from the 10K table.

### [10] README.md:305

- **CLAIM:** "No model is more than 5x slower than sklearn."
- **LOCATION:** `README.md:305`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Repo artifacts show larger gaps: docs/validation-report.md:162 KNN predict at n=10K ratio 45.83x (sklearn faster); :151 DBSCAN fit 25.32x; :143 HistGBT fit 13.40x; docs/benchmarks.md:40 SVC RBF 6.20x slower than sklearn.
- **REPRO:**
  ```sh
  sed -n '143p;151p;162p' docs/validation-report.md && sed -n '40p' docs/benchmarks.md
  ```
- **FIX:** 'No model in the 2026-03-27 cross-library benchmark is more than ~4.2x slower on the measured workloads; other repo benchmarks show larger gaps for KNN predict, DBSCAN fit and SVC RBF (see docs/validation-report.md, docs/benchmarks.md).'

### [11] README.md:305

- **CLAIM:** "FerroML wins on **predict universally** (zero Python overhead)"
- **LOCATION:** `README.md:305`
- **VERDICT:** **FALSE**
- **EVIDENCE:** docs/cross-library-benchmark.md (2026-03-27, the cited source) shows sklearn faster on predict for HistGBT (ferroml 4.2/4.1 ms vs sklearn 2.0 ms, L43-44/47-48) and KNN (27.5 ms vs 12.9 ms, L51-52).
- **REPRO:**
  ```sh
  sed -n '43,48p;51,52p' docs/cross-library-benchmark.md
  ```
- **FIX:** 'FerroML is faster on predict for most models, but not universally: sklearn is faster on HistGBT and KNN predict at scale.'

### [12] README.md:305

- **CLAIM:** "on **tree/ensemble models** (Rayon parallel construction)"
- **LOCATION:** `README.md:305`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** docs/cross-library-benchmark.md: RandomForestRegressor fit 19.87x faster than sklearn, RandomForestClassifier 4.90x (fit-time speedup table).
- **REPRO:**
  ```sh
  sed -n '60,66p' docs/cross-library-benchmark.md
  ```
- **FIX:** None needed.

### [13] README.md:305

- **CLAIM:** "achieves **parity or better on linear algebra** thanks to faer SVD and Cholesky solvers"
- **LOCATION:** `README.md:305`
- **VERDICT:** **FALSE**
- **EVIDENCE:** docs/cross-library-benchmark.md: LinearRegression fit 4.2 ms vs sklearn 2.2 ms (0.52x), Ridge 1.7 vs 1.0 (0.59x), LogisticRegression 16.8 vs 8.0 (0.48x) — FerroML is ~2x slower on dense linear-algebra fits; only PCA is at parity (1.01x).
- **REPRO:**
  ```sh
  sed -n '133,140p;72,76p' docs/cross-library-benchmark.md
  ```
- **FIX:** 'FerroML is within ~2x of sklearn on linear-model fits (LAPACK-backed) and at parity on PCA; it wins on TruncatedSVD (docs/benchmarks.md).'

### [14] README.md:311

- **CLAIM:** "8 WGSL shaders with auto CPU/GPU dispatch"
- **LOCATION:** `README.md:311`
- **VERDICT:** **FALSE**
- **EVIDENCE:** ferroml-core/src/gpu/kernels.rs defines 10 WGSL shader constants: MATMUL, RELU, SIGMOID, SOFTMAX, ROW_REDUCE, BIAS_ADD, RELU_GRAD, SIGMOID_GRAD, ELEMENTWISE_MUL, DISTANCE (`grep -c 'SHADER: &str'` = 10).
- **REPRO:**
  ```sh
  grep -n 'SHADER: &str' ferroml-core/src/gpu/kernels.rs
  ```
- **FIX:** '10 WGSL shaders with auto CPU/GPU dispatch' (or list the 8 dispatched ones if two are unused).

### [15] README.md:343

- **CLAIM:** "~188 tests for wgpu shader acceleration"
- **LOCATION:** `README.md:343`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** `#[test]` counts: ferroml-core/src/gpu/kernels.rs 114 + src/gpu/dispatch.rs 16 + src/gpu/mod.rs 58 = 188.
- **REPRO:**
  ```sh
  grep -rc '#\[test\]' ferroml-core/src/gpu/*.rs ferroml-core/src/gpu/mod.rs
  ```
- **FIX:** None needed.

### [16] README.md:312

- **CLAIM:** "CsrMatrix operations for 12 models"
- **LOCATION:** `README.md:312`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Exactly 11 types implement `traits::SparseModel`: LogisticRegression, RidgeRegression, LinearSVC, LinearSVR, KNeighborsClassifier, KNeighborsRegressor, NearestCentroid, GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB. TfidfTransformer adds sparse support separately but is a transformer, not a model.
- **REPRO:**
  ```sh
  grep -rn 'traits::SparseModel for' ferroml-core/src --include='*.rs' | awk -F'for ' '{print $2}' | sort -u
  ```
- **FIX:** 'CsrMatrix operations for 11 models, TfidfTransformer, and sparse scalers' / '11 models support sparse input'.

### [17] README.md:334

- **CLAIM:** "26 ignored (slow AutoML system tests)"
- **LOCATION:** `README.md:334`
- **VERDICT:** **FALSE**
- **EVIDENCE:** 31 `#[ignore]` attributes exist: 21 in ferroml-core/src/testing/automl.rs, 3 in models/compliance_tests/compliance.rs, 3 in preprocessing/compliance_tests/compliance.rs, plus 4 others. No grouping yields 26.
- **REPRO:**
  ```sh
  grep -rn '#\[ignore' ferroml-core/src ferroml-core/tests | wc -l
  ```
- **FIX:** '31 ignored (21 slow AutoML system tests)'.

### [18] README.md:335

- **CLAIM:** "Correctness Tests | 1,140 (553 edge_cases + 304 correctness + 88 adversarial + 103 integration + 36 regression + 56 vs_linfa)"
- **LOCATION:** `README.md:335`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Actual `#[test]` counts: edge_cases.rs 238 (not 553), correctness.rs 304, adversarial.rs 89 (not 88), integration.rs 103, regression_tests.rs 17 (not 36), vs_linfa.rs 56 — sum 807, not 1,140. The overall Rust figure 4,602 is correct because it includes src/ unit tests.
- **REPRO:**
  ```sh
  grep -rc '#\[test\]' ferroml-core/tests/*.rs
  ```
- **FIX:** 'Correctness-focused integration tests: 807 (238 edge_cases + 304 correctness + 89 adversarial + 103 integration + 17 regression + 56 vs_linfa)'.

### [19] README.md:337

- **CLAIM:** "Frankenstein Tests | 37 Python + 8 Rust composition tests"
- **LOCATION:** `README.md:337`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Python side correct: tests/test_frankenstein.py has 37 `def test_`. Rust side wrong: the `layer5_frankenstein` module in ferroml-core/tests/correctness.rs (L7639) contains 7 `#[test]` functions, not 8 (docs/correctness-report.md:38 also claims 8).
- **REPRO:**
  ```sh
  grep -c 'def test_' ferroml-python/tests/test_frankenstein.py && awk 'NR>7636 && /#\[test\]/' ferroml-core/tests/correctness.rs | wc -l
  ```
- **FIX:** '37 Python + 7 Rust composition tests'.

### [20] README.md:339

- **CLAIM:** "Python Test Files | 60+ end-to-end test files"
- **LOCATION:** `README.md:339`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** 85 `test_*.py` files exist in ferroml-python/tests/ (docs/accuracy-report.md:150 still says 35 — stale).
- **REPRO:**
  ```sh
  ls ferroml-python/tests/test_*.py | wc -l
  ```
- **FIX:** None needed.

### [21] README.md:340

- **CLAIM:** "Python Bindings | ~99% coverage (55+ models, 22+ preprocessors, 6 decomposition, 37 explainability)"
- **LOCATION:** `README.md:340`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** '~99% coverage' needs a coverage report (artifact needed: maturin build + coverage run). Spot-check: preprocessing exports 26 objects (>= 22 OK), but ferroml.decomposition exports 7 classes (PCA, IncrementalPCA, TruncatedSVD, LDA, QuadraticDiscriminantAnalysis, FactorAnalysis, TSNE), not 6.
- **REPRO:**
  ```sh
  grep -c '"' <(grep -A60 '__all__' ferroml-python/python/ferroml/preprocessing/__init__.py) && grep -A10 '__all__' ferroml-python/python/ferroml/decomposition/__init__.py
  ```
- **FIX:** Publish a coverage artifact; correct decomposition count to 7 (or define the counting rule).

### [22] README.md:341; ferroml-python/README.md:105-107

- **CLAIM:** "`score()` on 56 models, `partial_fit` on 10, `decision_function` on 13 classifiers"
- **LOCATION:** `README.md:341; ferroml-python/README.md:105-107`
- **VERDICT:** **FALSE**
- **EVIDENCE:** PyO3 method counts in ferroml-python/src: score = 56 (model score() methods; matches), partial_fit = 10 (matches), decision_function = 18 (ensemble.rs 4, anomaly.rs 2, svm.rs 4, trees.rs 4, linear.rs 3, decomposition.rs 1) — not 13. The 13 figure is stale (docs/feature-parity-scorecard.md:30 shows the pre-v0.3.0 state of 7).
- **REPRO:**
  ```sh
  grep -rn 'fn decision_function' ferroml-python/src --include='*.rs' | wc -l   # 18
  ```
- **FIX:** '`decision_function` on 18 estimators' (or recount per current bindings).

### [23] README.md:342

- **CLAIM:** "Benchmarks | 86+ Criterion benchmarks with CI regression baseline"
- **LOCATION:** `README.md:342`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** ferroml-core/benches/benchmarks.rs alone generates >140 benchmark cases via sized loops (141 loop iterations with BenchmarkId), plus performance_optimizations.rs (15), gaussian_process.rs (11), gpu_benchmarks.rs (3). CI regression baseline exists: .github/workflows/benchmarks.yml:50-55 + ferroml-core/benches/baseline.json (`regression_threshold_percent: 15`, 48 tracked cases).
- **REPRO:**
  ```sh
  grep -c 'BenchmarkId::new' ferroml-core/benches/benchmarks.rs && grep -n 'regression_threshold_percent' ferroml-core/benches/baseline.json
  ```
- **FIX:** None needed.

### [24] README.md:367

- **CLAIM:** "validated against scikit-learn with 58 fixture-based comparisons"
- **LOCATION:** `README.md:367`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** benchmarks/fixtures/*.json contain 39 fixture cases in total; docs/accuracy-report.md:148 claims 'Sklearn correctness tests | 58 passing' and its reproduce command (L160) references `cargo test --test sklearn_correctness`, a target that does not exist in ferroml-core/tests/. Artifact needed: the historical sklearn_correctness test target or its saved output.
- **REPRO:**
  ```sh
  python3 -c "import json,glob; print(sum(len(json.load(open(f))) if isinstance(json.load(open(f)),list) else len(json.load(open(f)).get('cases',{})) for f in glob.glob('benchmarks/fixtures/*.json')))"   # 39; ls ferroml-core/tests/
  ```
- **FIX:** Restore/regenerate the fixture-comparison test and its report, or restate the number from the current suite.

### [25] README.md:369-375

- **CLAIM:** Accuracy table: "Models (sklearn comparison) 17 / Preprocessing 15 / Clustering correctness 102 / Neural 49 / Preprocessing correctness 101"
- **LOCATION:** `README.md:369-375`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** No artifact produces these per-category counts. Nearest traces: ferroml-python/tests/test_clustering.py has 63 test defs (not 102), test_neural.py 70 (not 49); the numbers are repeated from docs/accuracy-report.md, itself a document, not generated data. Artifact needed: test IDs per category (e.g., pytest -k clustering --collect-only).
- **REPRO:**
  ```sh
  grep -c 'def test_' ferroml-python/tests/test_clustering.py ferroml-python/tests/test_neural.py
  ```
- **FIX:** Regenerate the table from collected test IDs, or cite the test files that produce each count.

### [26] README.md:377-382

- **CLAIM:** Tolerance table: "Closed-form 1e-8 / Iterative 1e-4 / Tree-based 1e-6 / Ensemble 5%"
- **LOCATION:** `README.md:377-382`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** Tolerance values are used in the tests: e.g. ferroml-core/tests/correctness.rs:948 `.abs() < 1e-8`; ferroml-python/tests/test_comparison_trees.py:64 `PERF_TOL = 0.05  # 5% performance tolerance for stochastic models`; 1e-4/1e-6 tolerances appear across test_comparison_*.py.
- **REPRO:**
  ```sh
  grep -rn '1e-8' ferroml-core/tests/correctness.rs | head -1 && grep -n 'PERF_TOL' ferroml-python/tests/test_comparison_trees.py
  ```
- **FIX:** None needed.

### [27] README.md:225-260

- **CLAIM:** "Real (excerpted) output from the run above — 420 trials on a 250 × 8 synthetic classification dataset"
- **LOCATION:** `README.md:225-260`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** This is runtime AutoML output; no artifact in the repo stores this report or the trial count. Artifact needed: re-running the documented snippet (`AutoMLConfig(time_budget_seconds=15, statistical_tests=True, seed=42)`) and saving `result.benchmark_report()`.
- **REPRO:**
  ```sh
  python3 -c "from ferroml.automl import AutoML, AutoMLConfig"   # requires built extension; full repro needs the snippet in README.md:216-222
  ```
- **FIX:** Commit the JSON report (format='json') that produced the excerpt, or label the output as illustrative.

### [28] README.md:203-232

- **CLAIM:** "result.statistically_best_model() / result.benchmark_report() / result.model_card()"
- **LOCATION:** `README.md:203-232`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** All three exist as PyO3 methods: ferroml-python/src/automl.rs:643 `fn statistically_best_model`, :722 `fn benchmark_report(format: &str)`, :736 `fn model_card`; Rust side per CHANGELOG.md:9-18.
- **REPRO:**
  ```sh
  grep -n 'fn statistically_best_model\|fn benchmark_report\|fn model_card' ferroml-python/src/automl.rs
  ```
- **FIX:** None needed.

### [29] README.md:273-281

- **CLAIM:** Feature flags table (parallel/simd/sparse/onnx/faer-backend/gpu, defaults)
- **LOCATION:** `README.md:273-281`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** ferroml-core/Cargo.toml `[features]`: `default = ["parallel", "onnx", "simd", "faer-backend", "datasets"]`; simd, sparse, gpu, faer-backend all defined as claimed. (Default also includes an undocumented `datasets` feature.)
- **REPRO:**
  ```sh
  sed -n '/\[features\]/,/\[\/p' ferroml-core/Cargo.toml | head -12
  ```
- **FIX:** Document the `datasets` default feature in the table.

### [30] README.md:255-264

- **CLAIM:** "cargo run --example automl|linear_regression|classification|gradient_boosting|pipeline"
- **LOCATION:** `README.md:255-264`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** All five files exist in ferroml-core/examples/: automl.rs, linear_regression.rs, classification.rs, gradient_boosting.rs, pipeline.rs.
- **REPRO:**
  ```sh
  ls ferroml-core/examples/
  ```
- **FIX:** None needed.

### [31] README.md:~100

- **CLAIM:** "Python (3.10-3.13)"
- **LOCATION:** `README.md:~100`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** ferroml-python/pyproject.toml: `requires-python = ">=3.10"`; classifiers list 3.10/3.11/3.12/3.13; maturin config `abi3-py310`.
- **REPRO:**
  ```sh
  grep -n 'requires-python\|3.13' ferroml-python/pyproject.toml
  ```
- **FIX:** None needed.

### [32] README.md:~125

- **CLAIM:** "The plugin bundles 5 slash commands, an autonomous ML agent, and hooks..." (ferroml-plugin)
- **LOCATION:** `README.md:~125`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** Claims about the external repository github.com/robertlupo1997/ferroml-plugin; nothing in this repo can verify the command count, agent, or hooks. Artifact needed: the plugin repo's manifest/command definitions.
- **REPRO:**
  ```sh
  grep -rn 'ferroml-plugin' README.md   # only the external URL is present
  ```
- **FIX:** Keep, but verify against the plugin repo before release; or link its command list.

### [33] README.md:441

- **CLAIM:** "ONNX export depends on `ort 2.0.0-rc.11` (release candidate)"
- **LOCATION:** `README.md:441`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** ferroml-core/Cargo.toml:91-96 `[dependencies.ort] version = "2.0.0-rc.11"` (optional, via `onnx-validation` feature).
- **REPRO:**
  ```sh
  sed -n '91,96p' ferroml-core/Cargo.toml
  ```
- **FIX:** None needed.

### [34] README.md:441

- **CLAIM:** "all 118 roundtrip tests pass"
- **LOCATION:** `README.md:441`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** No artifact shows 118 roundtrip tests. Nearest counts: 88 `#[test]` in ferroml-core/src/onnx/*.rs; 71 functions named *roundtrip* repo-wide; scripts/validate_onnx_exports.py validates 34 exportable models. Artifact needed: `cargo test -p ferroml-core --features onnx-validation` output.
- **REPRO:**
  ```sh
  grep -rc '#\[test\]' ferroml-core/src/onnx/*.rs | awk -F: '{s+=$2} END {print s}'   # 88
  ```
- **FIX:** Recount from the actual ONNX test run and cite the number, or link the CI job.

### [35] README.md:449

- **CLAIM:** "FerroML achieves competitive or better performance... No model is more than 5x slower." (Known Limitations)
- **LOCATION:** `README.md:449`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Same contradiction as README.md:305 — docs/validation-report.md:162 shows KNN predict 45.83x slower at n=10K; docs/benchmarks.md:40 shows SVC RBF 6.20x slower.
- **REPRO:**
  ```sh
  sed -n '162p' docs/validation-report.md
  ```
- **FIX:** Delete the 'No model is more than 5x slower' sentence or scope it to the cross-library benchmark workloads.

### [36] README.md:449

- **CLAIM:** "Highlights: KMeans 3.9x faster, StandardScaler 3.4x faster, GaussianNB 2.7x faster"
- **LOCATION:** `README.md:449`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** docs/cross-library-benchmark.md:84-87: GaussianNB 2.72x, StandardScaler 3.44x, KMeans 3.90x (fit-time speedups).
- **REPRO:**
  ```sh
  sed -n '84,87p' docs/cross-library-benchmark.md
  ```
- **FIX:** None needed.

### [37] README.md:174-191

- **CLAIM:** Modules table (stats, models, ensemble, hpo, preprocessing, clustering, onnx, datasets, ...)
- **LOCATION:** `README.md:174-191`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** All listed modules exist: ferroml-core/src/{stats,models,hpo,preprocessing,clustering,pipeline,cv,automl,explainability,metrics,onnx,datasets}; spot-checks confirm HDBSCAN, GaussianMixture, TfidfTransformer exports. Note: Python `ensemble` module contents differ from this table (see ferroml-python/README.md row below).
- **REPRO:**
  ```sh
  ls ferroml-core/src/
  ```
- **FIX:** None needed.

## ferroml-python/README.md

### [38] ferroml-python/README.md:9

- **CLAIM:** "55+ algorithms"
- **LOCATION:** `ferroml-python/README.md:9`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** docs/feature-parity-scorecard.json `total_ferroml_models: 94`.
- **REPRO:**
  ```sh
  python3 -c "import json; print(json.load(open('docs/feature-parity-scorecard.json'))['summary']['total_ferroml_models'])"
  ```
- **FIX:** None needed.

### [39] ferroml-python/README.md:9

- **CLAIM:** "It's 2-40x faster than scikit-learn on predict"
- **LOCATION:** `ferroml-python/README.md:9`
- **VERDICT:** **FALSE**
- **EVIDENCE:** The newest benchmark data in the repo contradicts it: docs/cross-library-benchmark.md (2026-03-27) shows FerroML SLOWER on predict for HistGBT (4.2 ms vs sklearn 2.0 ms) and KNN (27.5 ms vs 12.9 ms). The 2-40x range traces only to the stale docs/benchmark-vs-sklearn.md:247 ('Predict across the board: 2-40x faster'), recorded on 2026-03-11 against FerroML v0.2.0. The newer cross-library-benchmark.md supersedes it.
- **REPRO:**
  ```sh
  sed -n '43,44p;51,52p' docs/cross-library-benchmark.md && sed -n '247p' docs/benchmark-vs-sklearn.md
  ```
- **FIX:** 'Up to ~19x faster on predict for many models, but slower than sklearn for HistGBT and KNN predict at scale (docs/cross-library-benchmark.md).' — or DELETE and point to the current benchmark doc.

### [40] ferroml-python/README.md:9

- **CLAIM:** "up to 9x faster on fit for tree/ensemble models"
- **LOCATION:** `ferroml-python/README.md:9`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** docs/benchmark-vs-sklearn.md:217 RandomForest fit 9.24x at N=1000; newer docs/cross-library-benchmark.md even shows RandomForestRegressor fit 19.87x vs sklearn, so 'up to 9x' is conservative, not false.
- **REPRO:**
  ```sh
  sed -n '217p' docs/benchmark-vs-sklearn.md && grep -n 'RandomForestRegressor | 10000 | 19.87x' docs/cross-library-benchmark.md
  ```
- **FIX:** Optionally update to 'up to ~20x' per the 2026-03-27 data.

### [41] ferroml-python/README.md:17

- **CLAIM:** "Pre-built wheels available for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64)."
- **LOCATION:** `ferroml-python/README.md:17`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** .github/workflows/publish-pypi.yml builds exactly these targets: x86_64/aarch64-unknown-linux-gnu (L80-82), x86_64/aarch64-apple-darwin (L112-114), x86_64-pc-windows-msvc (L146).
- **REPRO:**
  ```sh
  grep -n 'target:' .github/workflows/publish-pypi.yml
  ```
- **FIX:** None needed.

### [42] ferroml-python/README.md:65-71

- **CLAIM:** Performance table "RandomForest 9.2x/7.8x, Ridge 5.3x/19.3x, DecisionTree 1.4x/16.4x, GradientBoosting 1.5x/1.2x, LogisticRegression 1.5x/13.7x"
- **LOCATION:** `ferroml-python/README.md:65-71`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** Values match docs/benchmark-vs-sklearn.md (2026-03-11): RF 9.24x/7.78x (L217), Ridge@1K 5.30x/19.30x (L214), DT@5K 1.38x/16.38x (L223), GB@1K 1.49x/1.17x (L225), LogReg@10K 1.45x/13.71x (L232). STALENESS NOTE: that source documents FerroML v0.2.0; the newer cross-library-benchmark.md (v0.4-era, 2026-03-27) shows different fit results (e.g., LogisticRegression fit 0.48x = slower at 10K). The newer file supersedes the older one.
- **REPRO:**
  ```sh
  sed -n '214p;217p;223p;225p;232p' docs/benchmark-vs-sklearn.md && sed -n '73p' docs/cross-library-benchmark.md
  ```
- **FIX:** Re-run benchmarks against v1.0.2 and refresh this table from the new data file.

### [43] ferroml-python/README.md:73

- **CLAIM:** "FerroML is faster on **predict universally**"
- **LOCATION:** `ferroml-python/README.md:73`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Same contradiction as root README.md:305 — docs/cross-library-benchmark.md:43-52 shows sklearn faster on predict for HistGBT and KNN.
- **REPRO:**
  ```sh
  sed -n '43,44p;51,52p' docs/cross-library-benchmark.md
  ```
- **FIX:** 'Faster on predict for most models; sklearn wins on HistGBT and KNN predict at scale.'

### [44] ferroml-python/README.md:73

- **CLAIM:** "scikit-learn wins on fit for LAPACK/MKL-backed linear algebra."
- **LOCATION:** `ferroml-python/README.md:73`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** docs/cross-library-benchmark.md: LinearRegression fit 0.52x, Ridge 0.59x, LogisticRegression 0.48x relative to sklearn (fit-time speedup table).
- **REPRO:**
  ```sh
  sed -n '60,66p' docs/cross-library-benchmark.md
  ```
- **FIX:** None needed.

### [45] ferroml-python/README.md:79

- **CLAIM:** `linear` model list (LinearRegression ... IsotonicRegression)
- **LOCATION:** `ferroml-python/README.md:79`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** ferroml-python/python/ferroml/linear/__init__.py exports all 13 listed classes — under the names RidgeRegression/LassoRegression rather than 'Ridge'/'Lasso'.
- **REPRO:**
  ```sh
  grep -A16 '__all__' ferroml-python/python/ferroml/linear/__init__.py
  ```
- **FIX:** Use exact export names (RidgeRegression, LassoRegression) in the table.

### [46] ferroml-python/README.md:81

- **CLAIM:** `ensemble` | "RandomForest, ExtraTrees, AdaBoost, Bagging, Stacking, Voting (classifiers + regressors), SGD, PassiveAggressive"
- **LOCATION:** `ferroml-python/README.md:81`
- **VERDICT:** **FALSE**
- **EVIDENCE:** ferroml-python/python/ferroml/ensemble/__init__.py `__all__` contains ExtraTrees*, AdaBoost*, SGD*, PassiveAggressive*, Bagging*, Voting*, Stacking* — NO RandomForest*. RandomForestClassifier/Regressor are exported by ferroml.trees (trees/__init__.py), consistent with the root README's own gotcha that trees live in `ferroml.trees`.
- **REPRO:**
  ```sh
  grep -A16 '__all__' ferroml-python/python/ferroml/ensemble/__init__.py && grep -A10 '__all__' ferroml-python/python/ferroml/trees/__init__.py
  ```
- **FIX:** Remove 'RandomForest' from the ensemble row (it is listed under `trees`).

### [47] ferroml-python/README.md:89

- **CLAIM:** `decomposition` | "PCA, IncrementalPCA, TruncatedSVD, LDA, QDA, FactorAnalysis, TSNE"
- **LOCATION:** `ferroml-python/README.md:89`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** ferroml-python/python/ferroml/decomposition/__init__.py `__all__` contains exactly PCA, IncrementalPCA, TruncatedSVD, LDA, QuadraticDiscriminantAnalysis, FactorAnalysis, TSNE.
- **REPRO:**
  ```sh
  grep -A10 '__all__' ferroml-python/python/ferroml/decomposition/__init__.py
  ```
- **FIX:** None needed.

### [48] ferroml-python/README.md:90

- **CLAIM:** `preprocessing` | "22+ transformers"
- **LOCATION:** `ferroml-python/README.md:90`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** ferroml-python/python/ferroml/preprocessing/__init__.py exports 26 objects.
- **REPRO:**
  ```sh
  grep -A60 '__all__' ferroml-python/python/ferroml/preprocessing/__init__.py | grep -c '"'
  ```
- **FIX:** None needed.

### [49] ferroml-python/README.md:98

- **CLAIM:** `datasets` | "Iris, Diabetes, Wine, California Housing, synthetic generators"
- **LOCATION:** `ferroml-python/README.md:98`
- **VERDICT:** **FALSE**
- **EVIDENCE:** No California Housing loader exists. ferroml-python/python/ferroml/datasets/__init__.py exposes load_iris, load_wine, load_diabetes, load_linnerud plus synthetic generators; 'california' appears only in test helpers that fetch it FROM sklearn (tests/conftest_comparison.py:44-47) and a Rust lookalike dataset (ferroml-core/tests/integration.rs:116-117).
- **REPRO:**
  ```sh
  grep -rn 'load_' ferroml-python/python/ferroml/datasets/__init__.py | head && grep -rin california ferroml-core/src/datasets/ | head -3
  ```
- **FIX:** "Iris, Diabetes, Wine, Linnerud, synthetic generators".

### [50] ferroml-python/README.md:105-107

- **CLAIM:** "score() on 56 models / partial_fit() on 10 / decision_function() on 13 classifiers" (sklearn API list)
- **LOCATION:** `ferroml-python/README.md:105-107`
- **VERDICT:** **FALSE**
- **EVIDENCE:** score=56 and partial_fit=10 verified in bindings; decision_function=18 PyO3 methods (see README.md:341 row). The quick-start at :45 additionally uses Pipeline.score(), which does not exist (own row below).
- **REPRO:**
  ```sh
  grep -rn 'fn decision_function' ferroml-python/src --include='*.rs' | wc -l
  ```
- **FIX:** 'decision_function() on 18 estimators'.

### [51] ferroml-python/README.md:114

- **CLAIM:** "5,650+ tests passing (3,550+ Rust + 2,100+ Python), validated against ... with 200+ cross-library correctness tests."
- **LOCATION:** `ferroml-python/README.md:114`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Stale v0.3.x numbers superseded by newer repo artifacts: ferroml-core has 4,602 `#[test]` attrs (> 3,550); docs/correctness-report.md:53-55 (updated 2026-07-12) and the root README state 4,600+ Rust + 3,000+ Python; cross-library tests total 517 (> 200). Which supersedes: docs/correctness-report.md + root README v1.0.2 figures.
- **REPRO:**
  ```sh
  grep -rn '#\[test\]' ferroml-core/src ferroml-core/tests | wc -l
  ```
- **FIX:** '7,600+ tests passing (4,600+ Rust + 3,000+ Python) with 480+ cross-library correctness tests' — matching the audited root figures.

### [52] ferroml-python/README.md:40-45

- **CLAIM:** Quick start: "pipe = Pipeline([...]); pipe.fit(X_train, y_train); score = pipe.score(X_test, y_test)"
- **LOCATION:** `ferroml-python/README.md:40-45`
- **VERDICT:** **FALSE**
- **EVIDENCE:** The native Pipeline class exposes no score method: ferroml-python/src/pipeline.rs defines only new/fit/transform/fit_transform/predict/named_steps/params/etc., and python/ferroml/pipeline/__init__.py aliases the native class without adding score(). This call would raise AttributeError.
- **REPRO:**
  ```sh
  grep -n 'fn ' ferroml-python/src/pipeline.rs | grep -c score   # 0
  ```
- **FIX:** Either add Pipeline.score() to the bindings, or change the example to `clf.score(X_test, y_test)`.

## Cargo.toml / pyproject.toml metadata

### [53] Cargo.toml:15,18

- **CLAIM:** Cargo.toml workspace version "1.0.2" and package description
- **LOCATION:** `Cargo.toml:15,18`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** Consistent with ferroml-python/pyproject.toml version 1.0.2 and CHANGELOG.md top entry [1.0.2] — 2026-07-12.
- **REPRO:**
  ```sh
  grep -n 'version' Cargo.toml | head -3 && head -8 CHANGELOG.md
  ```
- **FIX:** None needed.

### [54] Cargo.toml:33

- **CLAIM:** Comment: "ndarray-linalg removed due to OpenBLAS requirement - using nalgebra for linalg instead"
- **LOCATION:** `Cargo.toml:33`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** Zero ndarray-linalg entries in Cargo.lock; nalgebra present as workspace dependency (Cargo.toml:32).
- **REPRO:**
  ```sh
  grep -c ndarray-linalg Cargo.lock   # 0
  ```
- **FIX:** None needed.

### [55] ferroml-python/pyproject.toml:14, classifiers

- **CLAIM:** pyproject requires-python >=3.10, classifiers 3.10-3.13, Typing :: Typed
- **LOCATION:** `ferroml-python/pyproject.toml:14, classifiers`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** `requires-python = ">=3.10"`; CPython 3.10-3.13 classifiers present; py.typed marker exists (ferroml-python/python/ferroml/py.typed); maturin abi3-py310 matches.
- **REPRO:**
  ```sh
  ls ferroml-python/python/ferroml/py.typed && grep -n 'requires-python' ferroml-python/pyproject.toml
  ```
- **FIX:** None needed.

### [56] ferroml-python/pyproject.toml

- **CLAIM:** [project.scripts] ferroml = "ferroml.cli:app"
- **LOCATION:** `ferroml-python/pyproject.toml`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** ferroml-python/python/ferroml/cli/__init__.py:6 defines `app = typer.Typer(...)` with commands registered (train, predict, evaluate, compare, etc.).
- **REPRO:**
  ```sh
  grep -n 'app = typer.Typer' ferroml-python/python/ferroml/cli/__init__.py
  ```
- **FIX:** None needed.

### [57] Cargo.toml:17; pyproject license-files

- **CLAIM:** License metadata MIT OR Apache-2.0 + license files
- **LOCATION:** `Cargo.toml:17; pyproject license-files`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** LICENSE-MIT and LICENSE-APACHE exist at repo root; both license classifiers declared; dual licensing text matches root README License section.
- **REPRO:**
  ```sh
  ls LICENSE-MIT LICENSE-APACHE
  ```
- **FIX:** None needed.

## Badges

### [58] README.md:3; ferroml-python/README.md:4

- **CLAIM:** CI badge -> .github/workflows/ci.yml
- **LOCATION:** `README.md:3; ferroml-python/README.md:4`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** .github/workflows/ci.yml exists and runs cargo clippy/test (`cargo test -p ferroml-core --all-features --release`, L126).
- **REPRO:**
  ```sh
  ls .github/workflows/ci.yml
  ```
- **FIX:** None needed.

### [59] README.md:3; ferroml-python/README.md:3

- **CLAIM:** PyPI badge (published version of ferroml on pypi.org)
- **LOCATION:** `README.md:3; ferroml-python/README.md:3`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** Whether a release is actually published on PyPI cannot be settled from repository artifacts. Publishing workflows exist (.github/workflows/publish-pypi.yml). Artifact needed: PyPI registry response for https://pypi.org/pypi/ferroml/json.
- **REPRO:**
  ```sh
  curl -s https://pypi.org/pypi/ferroml/json   # outside repo scope
  ```
- **FIX:** None (external fact).

### [60] README.md:3; ferroml-python/README.md:5

- **CLAIM:** License badge (crates/pypi license)
- **LOCATION:** `README.md:3; ferroml-python/README.md:5`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** License is verifiable in-repo: Cargo.toml `license = "MIT OR Apache-2.0"` + LICENSE files present.
- **REPRO:**
  ```sh
  grep -n 'license' Cargo.toml | head -2
  ```
- **FIX:** None needed.

## docs/*.md

### [61] docs/benchmarks.md:21-63

- **CLAIM:** docs/benchmarks.md performance results (SVC RBF 6.20x, KMeans 4.25x, Ridge 4.88x, ...) and run metadata (rustc 1.93.1, sklearn 1.8.0, FerroML v0.4.0-dev)
- **LOCATION:** `docs/benchmarks.md:21-63`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** These are recorded measurements of one run; no data file in the repo stores the underlying samples, and reproducing requires the documented hardware/build (`scripts/benchmark_vs_sklearn.py --perf-only`). Metadata is stale relative to v1.0.2 (L12 says v0.4.0-dev). Artifact needed: a rerun saving JSON output, committed alongside the md.
- **REPRO:**
  ```sh
  python3 scripts/benchmark_vs_sklearn.py --perf-only --output /tmp/perf.json   # requires built extension + sklearn
  ```
- **FIX:** Regenerate with current version and commit the raw JSON next to the markdown.

### [62] docs/validation-report.md:3,7,185

- **CLAIM:** docs/validation-report.md: '281 side-by-side comparison tests... All 281 tests pass. Zero bugs were found.'
- **LOCATION:** `docs/validation-report.md:3,7,185`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Internally contradictory and stale: docs/accuracy-report.md:29 says the same Plan M suite is 279 tests and its own heading (L32) says 265; validation-report.md itself says '265 tests' in its reproduce block (L238). Current code has 245 comparison test defs (223 in test_comparison_*.py + 22 in test_performance_comparison.py). None of 265/279/281 matches the current suite; the three documents disagree.
- **REPRO:**
  ```sh
  cat ferroml-python/tests/test_comparison_*.py ferroml-python/tests/test_performance_comparison.py | grep -c 'def test_' && grep -n '265\|279\|281' docs/validation-report.md docs/accuracy-report.md | head
  ```
- **FIX:** Pick one count regenerated from the current suite and reconcile accuracy-report.md, validation-report.md, and the README.

### [63] docs/accuracy-report.md:145-151

- **CLAIM:** docs/accuracy-report.md Test Infrastructure: '~3,400 Rust passing, 6 ignored / ~1,385 Python passing / Python test files 35'
- **LOCATION:** `docs/accuracy-report.md:145-151`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Stale v0.x snapshot: actual counts are 4,602 Rust #[test], 31 #[ignore], 85 Python test files. Superseded by docs/correctness-report.md:45-55 and root README Project Status.
- **REPRO:**
  ```sh
  grep -rc '#\[test\]' ferroml-core/src ferroml-core/tests | awk -F: '{s+=$2} END {print s}'; ls ferroml-python/tests/test_*.py | wc -l
  ```
- **FIX:** Delete the section or regenerate it from the current tree.

### [64] docs/accuracy-report.md:29,32

- **CLAIM:** docs/accuracy-report.md Plan M totals: summary 279 vs heading '(265 tests)'
- **LOCATION:** `docs/accuracy-report.md:29,32`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Self-inconsistent within one document: phase rows sum to 41+48+34+39+40+52+25 = 279 while the heading says 265; validation-report.md uses M.3=35/M.4=40 and total 281.
- **REPRO:**
  ```sh
  awk 'NR>=37 && NR<=46' docs/accuracy-report.md
  ```
- **FIX:** Recompute per-phase counts from test IDs and use them consistently.

### [65] docs/feature-parity-scorecard.md:30-35 (+ .json)

- **CLAIM:** docs/feature-parity-scorecard.md method parity (score: FerroML 2 / gap 57; partial_fit: 1 / gap 16; decision_function: 7)
- **LOCATION:** `docs/feature-parity-scorecard.md:30-35 (+ .json)`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Stale snapshot generated 2026-03-14 (L3), before Plan U added the API (ROADMAP.md:63). Current bindings have score on 56 models, partial_fit on 10, decision_function on 18. The scorecard no longer describes the shipped API; ROADMAP.md and the current code supersede it.
- **REPRO:**
  ```sh
  grep -rn 'fn score\|fn partial_fit\|fn decision_function' ferroml-python/src --include='*.rs' | awk -F'fn ' '{print $2}' | cut -d'(' -f1 | sort | uniq -c
  ```
- **FIX:** Regenerate via scripts/feature_parity_scorecard.py and commit fresh JSON+MD, or mark the file as a historical snapshot.

### [66] docs/performance.md:~191

- **CLAIM:** docs/performance.md: 'Threshold: 10% regression triggers CI failure'
- **LOCATION:** `docs/performance.md:~191`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Configured threshold is 15%: ferroml-core/benches/baseline.json:214 `"regression_threshold_percent": 15` ('tightened from 20% to 15%'); scripts/check_bench_regressions.py default is 20%. No artifact uses 10%.
- **REPRO:**
  ```sh
  grep -n 'regression_threshold_percent' ferroml-core/benches/baseline.json && sed -n '9p;44,45p' scripts/check_bench_regressions.py
  ```
- **FIX:** 'Threshold: 15% regression triggers the CI benchmark-regression job (baseline.json regression_threshold_percent).'

### [67] docs/user-guide.md:74

- **CLAIM:** docs/user-guide.md install snippet: ferroml-core = "0.1"
- **LOCATION:** `docs/user-guide.md:74`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Current version is 1.0.2 (Cargo.toml:15); "0.1" never matches any release in CHANGELOG.md.
- **REPRO:**
  ```sh
  sed -n '72,76p' docs/user-guide.md && grep -n 'version' Cargo.toml | head -2
  ```
- **FIX:** ferroml-core = "1.0"

### [68] docs/user-guide.md:334-347

- **CLAIM:** docs/user-guide.md HistGradientBoosting example with monotonic_constraints=[1,0,-1], n_estimators=100, early_stopping=True, validation_fraction=0.1
- **LOCATION:** `docs/user-guide.md:334-347`
- **VERDICT:** **FALSE**
- **EVIDENCE:** The Python class does not accept these arguments: ferroml-python/python/ferroml/trees/__init__.pyi __init__ signature is (max_iter, learning_rate, max_depth, max_leaf_nodes, max_bins, min_samples_leaf, l2_regularization, random_state) — no monotonic_constraints, early_stopping, validation_fraction, or n_estimators. Monotonic/interaction constraints exist only in the Rust core (ferroml-core/src/models/hist_boosting.rs:648,650) and are not exposed to Python; the example would raise TypeError.
- **REPRO:**
  ```sh
  sed -n '221,231p' ferroml-python/python/ferroml/trees/__init__.pyi && grep -n 'monotonic_constraints' ferroml-core/src/models/hist_boosting.rs | head -2
  ```
- **FIX:** Show the real signature (max_iter etc.) and note monotonic constraints are Rust-only today, or expose them in the bindings first.

### [69] docs/user-guide.md:~365

- **CLAIM:** docs/user-guide.md: 'All [Naive Bayes variants] support incremental learning via partial_fit()'
- **LOCATION:** `docs/user-guide.md:~365`
- **VERDICT:** **SUPPORTED**
- **EVIDENCE:** All four NB wrappers expose partial_fit: ferroml-python/src/naive_bayes.rs:217,568,955,1311 (GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB).
- **REPRO:**
  ```sh
  grep -n 'fn partial_fit' ferroml-python/src/naive_bayes.rs
  ```
- **FIX:** None needed.

### [70] docs/ROADMAP.md:5-7

- **CLAIM:** docs/ROADMAP.md: 'Current Status: v0.3.1 ... 3,550+ Rust tests and 2,100+ Python tests passing (5,650+ total)'
- **LOCATION:** `docs/ROADMAP.md:5-7`
- **VERDICT:** **FALSE**
- **EVIDENCE:** Stale: repo is at v1.0.2 (Cargo.toml:15, CHANGELOG.md:7) with 4,602 Rust #[test]; docs/correctness-report.md and the root README supersede with 7,600+ collected. Release-history entries below remain valid history.
- **REPRO:**
  ```sh
  grep -n 'version = "1.0.2"' Cargo.toml && sed -n '5,7p' docs/ROADMAP.md
  ```
- **FIX:** Update the status banner to v1.0.2 with current counts, or move the banner into a dated 'Release History' entry.

### [71] docs/benchmark-results-*.md

- **CLAIM:** docs/benchmark-results-baseline.md / -optimized.md (179/253 Criterion timings at commits 78b72fd / 55ae057)
- **LOCATION:** `docs/benchmark-results-*.md`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** Historical measurement snapshots tied to specific commits and dates (2026-03-08/09); no machine-readable result files are committed for them and the commits are not resolvable in this worktree state. Artifact needed: the Criterion output at those commits. They are superseded for current claims by docs/cross-library-benchmark.md.
- **REPRO:**
  ```sh
  git log --oneline -- docs/benchmark-results-baseline.md   # locate recording commit
  ```
- **FIX:** Keep as dated history; add links to stored Criterion outputs if available.

### [72] docs/benchmark-vs-sklearn.md:36,241,247

- **CLAIM:** docs/benchmark-vs-sklearn.md: 'Speedup >1x means FerroML is faster... All 26 benchmarks produce matching predictions.' + key takeaways ('2-40x faster on every model')
- **LOCATION:** `docs/benchmark-vs-sklearn.md:36,241,247`
- **VERDICT:** **UNPROVABLE**
- **EVIDENCE:** A 2026-03-11 run against FerroML v0.2.0; not re-runnable from current artifacts, and its headline generalization is already contradicted by the repo's own newer data (docs/cross-library-benchmark.md: HistGBT/KNN predict slower; RF scores differ). The newer file supersedes this one for current claims. Artifact needed: rerun at the pinned version.
- **REPRO:**
  ```sh
  head -1 docs/benchmark-vs-sklearn.md && sed -n '247p' docs/benchmark-vs-sklearn.md
  ```
- **FIX:** Mark as historical (v0.2.0) and link docs/cross-library-benchmark.md as current.

---

TOTALS: supported=34 false=26 unprovable=12
