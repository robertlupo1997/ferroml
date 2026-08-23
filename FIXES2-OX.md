# FIXES2-OX — Corrections Addressing VERIFY-OX

Each entry: file:line, old text, new text, and the VERIFY-OX.md finding it
satisfies. Every citation below was re-proven with the shown command before
being written. No commits were made; all changes are working-tree only.

---

## 1. README.md:450 — "KNN predict" misattribution (VERIFY-OX Q2-1, Row-35 NEW ERROR)

Proof before editing:

```sh
grep -n 'KNeighborsClassifier' docs/cross-library-benchmark.md | head -2
# | KNeighborsClassifier | ferroml | ... | 2.0 | 27.5 | 0.9425 |
# | KNeighborsClassifier | sklearn | ... | 0.5 | 12.9 | 0.9425 |
sed -n '82p' docs/cross-library-benchmark.md   # fit-speedup row: KNeighborsClassifier 0.24x (=> sklearn ~4.17x faster on FIT)
```

The ~4.2x magnitude is the KNN **fit-time** gap (ferroml 2.0ms vs sklearn
0.5ms; the recorded speedup row is 0.24x = 1/4.17). The KNN **predict** gap in
the same run is 27.5/12.9 = **2.13x**.

- Old: `the largest gap is ~4.2x (KNN predict); other repo benchmarks show larger gaps for KNN predict at scale, DBSCAN fit, and SVC RBF`
- New: `the largest gap is ~4.2x (KNN fit); other repo benchmarks show larger gaps for KNN predict at scale, DBSCAN fit, and SVC RBF`

## 2. ferroml-python/README.md:9 — "up to 9x faster on fit" contradicts its own cited source (VERIFY-OX Q2-2, Row-39 STILL WRONG clause)

Proof before editing:

```sh
sed -n '76p' docs/cross-library-benchmark.md
# | RandomForestRegressor | 10000 | 19.87x | 6.97x | 0.33x |
```

The sentence anchors to docs/cross-library-benchmark.md, whose fit-speedup
table records RandomForestRegressor at **19.87x** vs sklearn. "Up to 9x" is an
understated maximum against that source. Restored the true magnitude (~20x)
and added the pending-rerun caveat, because the recorded run predates the
max_features parity fix in scripts/benchmark_cross_library.py (it compared
FerroML sqrt vs sklearn ALL-features), so every comparison figure in it is
provisional until a rerun.

- Old: `— and up to 9x faster on fit for tree/ensemble models.`
- New: `— and, in that recorded run, up to ~20x faster on fit for tree/ensemble models (RandomForest fit; that run predates the max_features parity fix in scripts/benchmark_cross_library.py, so its comparison figures await a rerun).`

## 3. docs/correctness-report.md:8 — live ">5x slower" falsehood + stale suite sub-counts (VERIFY-OX Q1-1 and Q1-2)

Proofs before editing:

```sh
grep -n '5x slower\|exceeds 5x' docs/correctness-report.md   # lines 42, 412, 451
grep -n '45.83' docs/validation-report.md                     # line 164: KNeighborsClassifier predict 45.83x slower
sed -n '153p' docs/validation-report.md                       # DBSCAN 25.32x
grep -E 'SVC RBF' docs/benchmarks.md                          # 6.20x slower
echo "src tests: $(grep -rn '#\[test' ferroml-core/src | wc -l)"        # 3795 (doc says 3,224)
grep -rn '#\[ignore' ferroml-core/src ferroml-core/tests | wc -l       # 31 (doc says 26 ignored)
grep -rc '#\[test\]' ferroml-core/tests/edge_cases.rs ferroml-core/tests/adversarial.rs  # 238 and 89 (doc says 553 and 88)
```

Treatment mirrors the erratum-banner pattern already applied to
docs/validation-report.md (a historical record is not rewritten for
provenance; a banner marks what no longer holds).

- Added after the header block (now docs/correctness-report.md:8):
  `> **Erratum (v1.0.2 claims audit):** two classes of statements in this historical report do not hold against committed benchmark data and current static test counts. (1) The ">5x slower than sklearn" performance claims (the Session 8 table row, the Executive Summary, and the Phase 3 conclusion) are contradicted by the repository's own recorded results: docs/validation-report.md records sklearn 45.83x faster on KNeighborsClassifier predict and 25.32x faster on DBSCAN, and docs/benchmarks.md records SVC RBF at 6.20x slower. (2) The suite sub-counts below ("Library unit tests 3,224 / 26 ignored", "Edge case tests 553", "Adversarial tests 88") are stale: static counting today gives 3,795 \`#[test]\` attributes in ferroml-core/src with 31 \`#[ignore]\` attributes repo-wide, and per-file counts of 238 (edge_cases.rs) and 89 (adversarial.rs). The grand totals (4,602 Rust / 3,039 Python) remain as recorded. Treat the sub-counts and performance claims here as historical; regenerate before citing.`

Lines 42, 49, 412, 428, 451 are left verbatim as historical record, per the
same disposition VERIFY-OX accepted ("SUPPORTED and correctly dispositioned")
for README's annotated per-category table.

## 4. docs/user-guide.md:1169 — broken AutoMLConfig example (VERIFY-OX Q1-3)

Proofs before editing:

```sh
sed -n '1168,1171p' docs/user-guide.md                       # multiple_testing_correction="benjamini_hochberg"
sed -n '25p' ferroml-python/python/ferroml/automl/__init__.pyi   # multiple_testing: str = "bh"
grep -n 'multiple_testing_from_str' -A4 ferroml-python/src/automl.rs  # "bh" | "benjamini-hochberg" | "fdr" accepted
```

`multiple_testing_correction` is not a parameter; as written the example
raises TypeError. Fixed to the canonical accepted value.

- Old:
  ```python
  config = AutoMLConfig(
      multiple_testing_correction="benjamini_hochberg",
      ...
  )
  ```
- New:
  ```python
  config = AutoMLConfig(
      multiple_testing="bh",
      ...
  )
  ```

## 5. Open question settled — which max_features default the cross-library benchmark exercises

**Answer: `"sqrt"` on BOTH sides.** The benchmark runs the Python wrapper
default, not the Rust core's Fraction(1/3).

Proof chain (each command run):

```sh
# (a) The script constructs FerroML forests WITHOUT max_features:
grep -n 'FerroRF(' scripts/benchmark_cross_library.py
#   -> FerroRF(n_estimators=100, max_depth=10, random_state=42)   (both regressor :360 and classifier :391 pre-fix)

# (b) The PyO3 wrapper signature defaults to "sqrt":
sed -n '1112p' ferroml-python/src/trees.rs    # regressor: max_features="sqrt"
sed -n '747p'  ferroml-python/src/trees.rs    # classifier: max_features="sqrt"

# (c) That default becomes Some(MaxFeatures::Sqrt) and IS applied:
sed -n '1123,1139p' ferroml-python/src/trees.rs   # "sqrt" => Some(MaxFeatures::Sqrt); .with_max_features(...)

# (d) with_max_features unconditionally overwrites the field, so the core default never survives the wrapper:
sed -n '903p;961,962p' ferroml-core/src/models/forest.rs
#   903: max_features: Some(MaxFeatures::Fraction(1.0 / 3.0)),  // core default
#   961: pub fn with_max_features(mut self, max_features: Option<MaxFeatures>) -> Self {
#   962:     self.max_features = max_features;

# (e) sklearn side is pinned to sqrt:
grep -n "SkRF(n_estimators=100, max_depth=10, max_features=\"sqrt\"" scripts/benchmark_cross_library.py
```

With the committed benchmark size of n_features=20 (`--features` default 20,
scripts/benchmark_cross_library.py:818; feature column "20" in the committed
tables), sqrt gives ceil(sqrt(20)) = **5** features per split while p/3 gives
ceil(20/3) = **7**. The benchmark exercises **5**: the wrapper's Sqrt.

**Change made so both sides match by construction, not by implicit default**
(satisfies the instruction "make both sides match whichever one the benchmark
really runs"): pinned `max_features="sqrt"` explicitly on both FerroML
constructors and updated the parity comments.

- scripts/benchmark_cross_library.py:362 (regressor) and :394 (classifier):
  - Old: `constructors["ferroml"] = lambda: FerroRF(n_estimators=100, max_depth=10, random_state=42)`
  - New: `constructors["ferroml"] = lambda: FerroRF(n_estimators=100, max_depth=10, max_features="sqrt", random_state=42)`
- scripts/benchmark_cross_library.py:355-359 parity comment now states the
  wrapper-vs-core distinction ("Python wrapper default ... overrides the Rust
  core regressor default of Fraction(1/3)") and that sqrt is pinned on both
  sides.

Validation: `python3 -m py_compile scripts/benchmark_cross_library.py` passes.
No rerun was attempted: running the benchmark requires a built extension
(`maturin develop --release`), which would mean installing/building packages —
out of scope here.

## 6. Cosmetic NEW ERROR — banner date for commit 55ae057 (VERIFY-OX Row 61/71/72 nit)

Proof before editing:

```sh
git log --format='%h %ad %cd' --date=short -1 55ae057   # 55ae057 2026-03-08 2026-03-08
```

- docs/benchmark-results-optimized.md:3
  - Old: `> **Historical snapshot:** Criterion timings recorded at commit 55ae057 (2026-03-09).`
  - New: `> **Historical snapshot:** Criterion timings recorded at commit 55ae057 (2026-03-08).`

## Declined changes, with reasons

1. **docs/correctness-report.md body rows (42, 49-57, 412, 428, 451) left
   unedited.** Rewriting a dated historical report's figures destroys
   provenance; VERIFY-OX itself endorsed banner-over-rewrite elsewhere
   ("leaving the recorded tables untouched preserves provenance"). The new
   erratum names each false/stale figure explicitly, including its line-level
   values, so a reader cannot miss them.
2. **docs/benchmark-results-optimized.md:5 body date "2026-03-09" left as-is.**
   It is the document's own pre-existing historical header (not written by
   FIXES-OX); git proves 2026-03-08, but altering the original snapshot's own
   date line is outside this audit's remit. The banner we authored is now
   correct.
3. **No edit to docs/cross-library-benchmark.md tables.** Confirmed correct
   disposition by VERIFY-OX: the tables are the recorded output of the run as
   it actually happened; the pending-rerun flags stay.
4. **README badge "7,600+ collected" wording left unchanged.** VERIFY-OX Q3-1
   flags it as partially unreproducible today but explicitly notes "the
   wording honestly names the source". Nothing inaccurate to remove without a
   pytest collection run requiring a built extension.
5. **No restoration of any OVERCORRECTED item.** Every item VERIFY-OX examined
   under that lens was ruled SUPPORTED ("Row 21 SUPPORTED, not overcorrected";
   benchmark-defect section "Not overcorrected"); no true claim was found
   deleted without cause, so there was nothing to restore.

## Verification summary

All four NOT-SAFE reasons addressed:

| VERIFY-OX reason | Fix |
|---|---|
| Q2-1 README:450 "KNN predict" misattribution | §1 — corrected to "KNN fit" |
| Q2-2 "up to 9x fit" contradicts cited source | §2 — corrected to ~20x with pending-rerun caveat |
| Q1-1/Q1-2 correctness-report ">5x" falsehood + stale counts | §3 — erratum banner with re-derived figures |
| Q1-3 broken AutoMLConfig example | §4 — kwarg fixed to `multiple_testing="bh"` |

Plus: open max_features question settled (§5 — benchmark runs sqrt on both
sides; FerroML constructors now pin it explicitly) and the cosmetic banner
date fixed (§6).
