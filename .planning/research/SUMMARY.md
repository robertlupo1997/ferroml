# Project Research Summary

**Project:** FerroML v0.4.0 Launch Hardening
**Domain:** Production-ready numerical ML library (Rust + Python bindings)
**Researched:** 2026-03-20
**Confidence:** HIGH

## Executive Summary

FerroML is a mature ML library with 55+ models, 5,650+ tests, and 183K lines of Rust, but it is not yet production-hardened. The library has the architecture and tooling of a production system -- centralized validation functions, a well-designed error enum, property-based testing frameworks, and performance profiling tools -- but coverage is inconsistent. The critical gap is not missing infrastructure; it is incomplete application of existing infrastructure. NaN/Inf inputs can silently corrupt results, 6,195 unwrap() calls create panic risk in production Python code, and two algorithms (PCA at 13.8x, LinearSVC at 9.6x slower than sklearn) have performance gaps that undermine credibility.

The recommended approach is a strictly sequential four-phase hardening sprint: input validation first (because it reduces the unwrap audit surface), then unwrap safety (because it enables safe refactoring), then performance optimization (because it depends on the safety net from phases 1-2), and finally documentation (because it captures the final state). No new runtime dependencies are needed. Every tool required -- proptest, approx, rstest, criterion, dhat, faer -- is already in the dependency tree. This is a code-quality milestone, not a stack-change milestone.

The key risks are: (1) Cholesky normal equations squaring the condition number when optimizing OLS/Ridge, causing silent numerical garbage on collinear data; (2) faer SVD sign conventions differing from nalgebra, silently breaking PCA cross-library tests; and (3) unwrap audit scope creep, where mechanical replacement of safe unwraps cascades function signature changes through the codebase. All three are preventable with the testing-first approach detailed below: write the stability/correctness tests before making the optimization, and triage unwraps by risk tier before replacing any.

## Key Findings

### Recommended Stack

No new runtime dependencies. The hardening stack is entirely code patterns applied to existing tools. See [STACK.md](STACK.md) for full details.

**Core patterns and tools:**
- `f64::is_finite()` validation in `validate_fit_input()` -- rejects NaN/Inf at every entry point, zero-cost Rust stdlib
- faer 0.20 thin SVD/Cholesky (already a dependency) -- replaces nalgebra in performance-critical decompositions (13x faster SVD)
- `clippy::unwrap_used` + `clippy::expect_used` lints -- systematic unwrap auditing with `allow-unwrap-in-tests = true`
- cargo-semver-checks 0.45 -- catches accidental API breakage before crates.io publish
- cargo-flamegraph 0.6.11 -- CPU profiling for identifying remaining performance bottlenecks
- proptest/rstest/approx (all already present) -- edge-case and property-based test coverage

**Critical constraint:** No ndarray-linalg (forbidden, requires OpenBLAS system dep). No new runtime crates per PROJECT.md.

### Expected Features

See [FEATURES.md](FEATURES.md) for full analysis with FerroML status annotations.

**Must have (table stakes -- blocks launch credibility):**
- NaN/Inf input rejection at fit/predict/transform (MISSING -- silent corruption today)
- Empty dataset handling without panics (MISSING -- panics today)
- Parameter validation with actionable error messages (MISSING -- silent acceptance of invalid hyperparameters)
- Zero known test failures (6 FAILURES exist -- TemperatureScaling, IncrementalPCA)
- PCA within 3x of sklearn (currently 13.8x slower -- dealbreaker)
- LinearSVC within 3x of sklearn (currently 9.6x slower -- dealbreaker)
- Unwrap-free critical paths (6,195 unwrap calls, unknown count in user-reachable paths)

**Should have (improves adoption):**
- Complete Python docstrings on all 55+ models with parameter documentation
- Known limitations documented (RF non-determinism, ONNX RC status, sparse limits)
- Published performance benchmarks (internal Criterion results exist, need public page)

**Defer (post-launch):**
- Full sparse algorithm support (massive scope, dense conversion works for moderate data)
- Deterministic parallel RandomForest (document `n_jobs=1` workaround)
- GPU training expansion, WASM targets, distributed training
- Custom loss functions / plugin architecture

**Key differentiators already built (market these):**
- Statistical diagnostics on every model (StatisticalModel trait) -- no competitor offers this
- Uncertainty quantification (PredictionWithUncertainty) -- sklearn lacks this natively
- 118-model ONNX export -- broader than sklearn
- AutoML with statistical rigor (paired t-tests for model comparison)

### Architecture Approach

FerroML's hardening maps onto four existing architectural layers that need deepening, not creation. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full layer diagram.

**Major components:**
1. **PyO3 Boundary** (`ferroml-python/src/`) -- NumPy-to-ndarray conversion and FerroError-to-Python exception translation. Validation must NOT happen here (belongs in Rust core).
2. **Input Validation Gate** (`models/mod.rs::validate_fit_input`, `schema.rs`) -- Centralized NaN/Inf/empty/shape validation. Already exists and is called by ~40 models, but adoption is not universal. Clustering and decomposition modules have gaps.
3. **Algorithm Core** (`models/`, `clustering/`, `decomposition/`, `linalg.rs`) -- Mathematical computation. Must have zero unwrap() calls on user-derived data. Performance backends route through `linalg.rs` facade.
4. **Linalg Facade** (`linalg.rs`) -- Already delegates QR and Cholesky to faer. Extend to SVD and eigendecomposition. Models must never call nalgebra directly for decompositions.

**Key pattern:** The `linalg.rs` facade pattern (already established) is the right architecture for the nalgebra-to-faer backend swap. Add `thin_svd()` and `eigendecomposition()` to the facade, with `svd_flip()` sign normalization built in.

### Critical Pitfalls

See [PITFALLS.md](PITFALLS.md) for all 13 pitfalls with detailed prevention strategies.

1. **Cholesky normal equations squaring condition number** -- When switching OLS/Ridge to Cholesky for performance, ill-conditioned matrices (cond > 10^6) produce silently wrong coefficients. Prevention: add ill-conditioned matrix tests BEFORE switching solvers; use Cholesky only with regularization (Ridge) or when n > 5d.
2. **faer SVD sign convention differences** -- SVD is unique only up to sign flips. Switching from nalgebra to faer will flip PCA component signs, breaking cross-library tests and serialized model compatibility. Prevention: implement `svd_flip()` in `linalg.rs` BEFORE switching backends.
3. **Unwrap audit scope creep** -- Mechanically replacing 6,195 unwraps changes function signatures, breaks callers, and degrades hot-path performance. Prevention: triage into 4 risk tiers; only tier 3 (actually unsafe) gets `?` replacement; tiers 1-2 get documented `expect()`.
4. **NaN validation inconsistency** -- Adding NaN checks at fit() but not predict()/transform() creates confusing partial safety. Prevention: single `validate_inputs()` function called at ALL entry points; validate X, y, AND sample_weight.
5. **Performance optimization correctness regressions** -- Changing distance metrics or convergence criteria silently alters model behavior on edge cases. Prevention: run full cross-library test suite after EACH optimization, not just at phase end.

## Implications for Roadmap

Based on research, the hardening work has strict sequential dependencies. Four phases, each unlocking the next.

### Phase 1: Input Validation and Test Cleanup
**Rationale:** Foundation for all other work. Validated inputs reduce the unwrap audit surface (many unwraps become provably safe once inputs are validated). Fixing test failures establishes a clean baseline.
**Delivers:** All models reject NaN/Inf/empty inputs consistently; parameter validation at fit time; 6 pre-existing test failures fixed or removed; zero known failures.
**Addresses:** NaN/Inf rejection, empty data handling, parameter validation, test cleanup (FEATURES.md table stakes)
**Avoids:** NaN validation inconsistency (Pitfall 3), empty/degenerate panics (Pitfall 6), parameter validation breaking builders (Pitfall 7)
**Estimated scope:** Audit ~55 model fit() entry points, add validate_params() to models with bounded hyperparameters, fix TemperatureScaling/IncrementalPCA or remove from public API.

### Phase 2: Unwrap Audit and Safety Hardening
**Rationale:** Depends on Phase 1 -- with validated inputs, many Tier 1 unwraps become Tier 4 (safe by validation). Without Phase 1, the audit cannot distinguish safe from unsafe unwraps.
**Delivers:** All Tier 1 (user-data-derived) and Tier 2 (iterative loop) unwraps replaced with proper error handling or documented expect(); clippy::unwrap_used lint enabled (warn level); output sanity checks (predictions finite).
**Addresses:** Unwrap-free critical paths (FEATURES.md), predict-time output validation (FEATURES.md)
**Avoids:** Unwrap audit scope creep (Pitfall 2), blanket replacement anti-pattern (ARCHITECTURE.md)
**Estimated scope:** Triage 6,195 unwraps; focus on ~10 high-risk files (svm.rs, regularized.rs, hist_boosting.rs, clustering modules); benchmark hot-path files before/after.

### Phase 3: Performance Optimization
**Rationale:** Depends on Phase 2 -- refactoring hot paths without proper error handling risks introducing panics. The safety net from phases 1-2 catches regressions from numerical changes.
**Delivers:** PCA within 2x of sklearn (faer thin SVD), LinearSVC within 3x (shrinking + cache), OLS/Ridge Cholesky path for n >> d, HistGBT histogram optimization.
**Addresses:** PCA 13.8x gap, LinearSVC 9.6x gap (FEATURES.md performance table stakes)
**Avoids:** Cholesky condition number squaring (Pitfall 1), faer SVD sign conventions (Pitfall 4), performance optimization correctness regressions (Pitfall 5), regularization masking instability (Pitfall 9)
**Estimated scope:** Implement svd_flip() in linalg.rs; add faer thin_svd() facade; add ill-conditioned matrix regression tests; swap PCA/TruncatedSVD/LDA/FactorAnalysis; LinearSVC shrinking heuristic; run full cross-library suite after each change.

### Phase 4: Documentation and Release Polish
**Rationale:** Documents the final state after all changes are complete. Cannot be done earlier because the API surface and performance numbers change in phases 1-3.
**Delivers:** Complete Python docstrings on all 55+ models, known limitations documented, updated benchmark results, cargo-semver-checks verification, SemVer compatibility confirmed.
**Addresses:** Docstring audit, limitations documentation, published benchmarks (FEATURES.md should-haves)
**Avoids:** PyO3 binding sync breakage (Pitfall 8), RF non-determinism documentation gap (Pitfall 13)
**Estimated scope:** Audit all PyO3 wrappers for text_signature and docstrings; write LIMITATIONS section; run cargo-semver-checks against v0.3.1.

### Phase Ordering Rationale

- **Strictly sequential:** Each phase depends on the prior phase being complete. This is not parallelizable.
- **Validation before audit:** Input validation makes many unwraps provably safe, reducing audit scope by an estimated 30-40%.
- **Safety before performance:** Performance refactoring without error handling is how panics get introduced. The unwrap audit provides the safety net.
- **Documentation last:** Documenting intermediate state wastes effort. Document the final state once.
- **Testing-first within each phase:** For performance optimizations, write the correctness/stability tests BEFORE making the change. This is the single most important process discipline.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Performance):** faer thin SVD API specifics, svd_flip() implementation details, LinearSVC shrinking heuristic design, Cholesky condition number threshold selection. These are algorithmic decisions that benefit from targeted research.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Validation):** Well-documented sklearn patterns; FerroML's existing validate_fit_input() is the right architecture, just needs universal adoption.
- **Phase 2 (Unwrap Audit):** Mechanical triage process; clippy lint configuration is well-documented.
- **Phase 4 (Documentation):** Straightforward audit and writing work.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new dependencies needed; all tools already in Cargo.toml or are standard cargo-install utilities. Sources are official Rust documentation and crates.io. |
| Features | HIGH | Feature priorities derived from sklearn API patterns (authoritative), Microsoft production checklists, and direct FerroML codebase audit. Clear status annotations per feature. |
| Architecture | HIGH | Architecture analysis based on direct codebase inspection. The 4-layer model matches existing code structure. validate_fit_input() already exists and works. |
| Pitfalls | HIGH | Pitfalls grounded in numerical analysis theory (condition number squaring), FerroML's own history (SVC LRU regression, KMeans squared-distance bug), and established Rust ecosystem patterns (unwrap auditing). |

**Overall confidence:** HIGH

### Gaps to Address

- **Exact unwrap triage count per tier:** The 6,195 total is known, but the breakdown into risk tiers (how many are Tier 1 vs Tier 4) needs a mechanical grep/categorization pass at the start of Phase 2. This determines phase scope.
- **Python docstring coverage:** Marked as UNKNOWN in FEATURES.md. Need an audit of all 55+ PyO3 wrappers to scope Phase 4 accurately.
- **TemperatureScaling/IncrementalPCA root cause:** The 6 pre-existing test failures need diagnosis before deciding fix vs. remove from API. This is Phase 1 scope work.
- **LinearSVC performance gap root cause:** The 9.6x gap needs profiling (flamegraph) to determine whether the fix is algorithmic (shrinking heuristic) or implementation-level (allocation, cache). Targeted research in Phase 3 planning.
- **ort 2.0.0-rc.11 stability:** The RC dependency could break on stable release. Not a hardening issue per se, but should be pinned with exact version (`=2.0.0-rc.11`) as a Phase 4 task.

## Sources

### Primary (HIGH confidence)
- Rust stdlib `f64::is_finite()` documentation -- input validation pattern
- Clippy `unwrap_used`/`expect_used` lint documentation -- unwrap audit tooling
- scikit-learn `check_array`, `validate_data`, `check_is_fitted` documentation -- validation patterns
- scikit-learn developer guidelines -- API design conventions
- Cholesky decomposition numerical analysis (Wikipedia, textbooks) -- condition number squaring
- faer-rs repository and benchmarks -- performance characteristics vs nalgebra
- FerroML codebase direct inspection -- `.planning/codebase/CONCERNS.md`, `ARCHITECTURE.md`, source files

### Secondary (MEDIUM confidence)
- cargo-semver-checks 2025 year-in-review blog -- tool capabilities and lint count
- Microsoft ML Production/Fundamentals Checklists -- feature prioritization
- Rust auditing/static analysis tool comparisons (2025 surveys) -- tool selection
- Andrew Gallant "unwrap is okay" blog post -- unwrap triage philosophy

### Tertiary (LOW confidence)
- ort 2.0.0-rc.11 release timeline -- uncertain when stable ships
- RandomForest parallel parameter naming -- needs verification against actual API

---
*Research completed: 2026-03-20*
*Ready for roadmap: yes*
