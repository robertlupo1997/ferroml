# Technology Stack: Launch Hardening

**Project:** FerroML v0.4.0 Launch Hardening
**Researched:** 2026-03-20
**Focus:** Input validation, numerical stability, performance profiling, unwrap auditing, edge-case testing

## Context

FerroML already has a mature stack (ndarray 0.16, nalgebra 0.33, faer 0.20, PyO3 0.24, etc.). This document covers the **hardening tools and patterns** needed for production release -- not the core stack itself. Per PROJECT.md constraint: **no new runtime dependencies**. All recommendations here are dev-dependencies, cargo-install tools, clippy configuration, or code patterns applied to the existing codebase.

## Recommended Hardening Stack

### Input Validation (Code Patterns, No New Dependencies)

| Pattern | Purpose | Confidence |
|---------|---------|------------|
| `f64::is_finite()` check in `validate_fit_input()` | Reject NaN/Inf at fit time (sklearn-compatible) | HIGH |
| `x.nrows() == 0` early return | Reject empty datasets with `FerroError::InvalidInput` | HIGH |
| Builder-pattern `validate_params()` | Eager hyperparameter validation at construction | HIGH |
| `ndarray` `.iter().all(|v| v.is_finite())` | Vectorized finite check over entire array | HIGH |

**Why `is_finite()` not `is_nan()` + `is_infinite()`:** Single call covers both NaN and Inf. Standard Rust f64 method, no dependency needed. This is the standard pattern used by numerical Rust libraries.

**Why NOT a validation crate (e.g., `validator`, `garde`):** These are for struct field validation (web forms, API inputs). FerroML needs array-level numerical validation -- a simple `is_finite()` loop is faster and more appropriate than any generic validation framework.

**Confidence:** HIGH -- `f64::is_finite()` is Rust stdlib, IEEE 754 compliant, zero-cost.

### Numerical Stability (Code Patterns Using Existing faer 0.20)

| Pattern | Purpose | When to Use | Confidence |
|---------|---------|-------------|------------|
| faer thin SVD via `faer::Mat::thin_svd()` | Replace nalgebra Jacobi SVD (13.8x faster) | PCA, TruncatedSVD, LDA, FactorAnalysis | HIGH |
| faer Cholesky via existing `cholesky_faer()` | Replace hand-rolled Cholesky in Ridge/OLS | Normal equations path when n >> 2d | HIGH |
| Tikhonov regularization (add lambda*I) | Stabilize ill-conditioned Gram matrices | Ridge, ElasticNet, any X^T X solve | HIGH |
| Condition number monitoring via singular values | Detect ill-conditioned inputs, warn user | Optional diagnostic, log via tracing | MEDIUM |
| Pivoted QR (faer) over MGS QR | More stable for rank-deficient matrices | OLS fallback path | MEDIUM |

**Why faer, not nalgebra for hot paths:** faer 0.20 is already a dependency. Its thin SVD is 13x faster than nalgebra's Jacobi SVD. Cholesky is also faster. Use faer for performance-critical decompositions, keep nalgebra for everything else (wide API surface, serde support).

**Why NOT add ndarray-linalg:** Explicitly forbidden -- requires OpenBLAS system dependency. faer provides the same performance benefit without system deps.

**Confidence:** HIGH -- faer already in Cargo.toml, performance validated in Plan W benchmarks.

### Performance Profiling Tools

| Tool | Version | Purpose | Install | Confidence |
|------|---------|---------|---------|------------|
| Criterion | 0.5 (already present) | Statistical benchmarking with regression detection | Already in dev-deps | HIGH |
| cargo-flamegraph | 0.6.11 | CPU profiling via perf, generates SVG flamegraphs | `cargo install flamegraph` | HIGH |
| dhat | 0.3 (already present) | Heap allocation profiling (count allocs, find hotspots) | Already in dev-deps | HIGH |
| `perf record` + `perf report` | system | Linux CPU profiling (sampling-based) | System package | HIGH |
| samply | latest | macOS/Linux profiler with Firefox Profiler UI | `cargo install samply` | MEDIUM |

**Why keep Criterion at 0.5 (not upgrade to 0.8):** FerroML has 86+ Criterion benchmark functions already written for 0.5. Criterion 0.8 is a rewrite with API changes. The migration cost is not justified for a hardening milestone -- benchmarks work fine. Upgrade in a future milestone if needed.

**Why cargo-flamegraph over alternatives:** Standard Rust ecosystem tool. Works on Linux (perf-based) which is FerroML's primary dev platform. Generates self-contained SVG files. No complex setup.

**Why dhat (already present) over heaptrack:** dhat integrates directly into Rust test harness. Already used in memory benchmarks. Heaptrack requires external tooling and is Linux-only.

**Why NOT Parca/eBPF profilers:** Overkill for library profiling. These are for production service profiling (always-on, low-overhead). FerroML is a library, not a service.

**Confidence:** HIGH -- all tools are standard Rust ecosystem, already proven in FerroML's existing benchmarks.

### Unwrap/Expect Audit Tooling

| Tool/Approach | Purpose | Configuration | Confidence |
|---------------|---------|---------------|------------|
| `clippy::unwrap_used` lint | Detect `.unwrap()` calls in non-test code | `#![deny(clippy::unwrap_used)]` in lib.rs | HIGH |
| `clippy::expect_used` lint | Detect `.expect()` calls in non-test code | `#![deny(clippy::expect_used)]` in lib.rs | HIGH |
| `clippy.toml` with `allow-unwrap-in-tests = true` | Allow unwrap in test code only | Create `clippy.toml` at workspace root | HIGH |
| `grep -rn "\.unwrap()" --include="*.rs" src/` | Manual audit for remaining unwraps | One-time triage, categorize as safe/unsafe | HIGH |

**Recommended approach -- phased, not big-bang:**

1. **Triage first:** Run grep to categorize 6,195 unwrap/expect calls into:
   - Safe (array indexing after bounds check, `Option` from infallible construction) -- annotate with `#[allow(clippy::unwrap_used)]` + comment
   - Unsafe (user input paths, error-prone operations) -- replace with `?` or `ok_or_else()`
   - Test-only -- automatically allowed via `clippy.toml`

2. **Enable lint incrementally:** Start with `#![warn(clippy::unwrap_used)]` (warning), fix critical paths, then escalate to `#![deny()]` once count is manageable.

3. **clippy.toml configuration:**
```toml
# Allow unwrap/expect in test code
allow-unwrap-in-tests = true
```

**Why NOT a third-party unwrap-finder tool:** Clippy's built-in `unwrap_used` and `expect_used` lints are the standard solution. They integrate with CI, support allow-lists, and understand test vs. non-test context as of the February 2025 update. No external tool needed.

**Confidence:** HIGH -- clippy lints are official, widely used, actively maintained.

### Testing Frameworks for Edge Cases

| Library | Version | Purpose | Status | Confidence |
|---------|---------|---------|--------|------------|
| proptest | 1.5 (already present) | Property-based testing, random input generation | Already in dev-deps | HIGH |
| approx | 0.5 (already present) | Floating-point approximate equality assertions | Already in dev-deps | HIGH |
| rstest | 0.23 (already present) | Parameterized tests (matrix of inputs) | Already in dev-deps | HIGH |
| test-case | 3.3 (already present) | Attribute-based test parameterization | Already in dev-deps | HIGH |

**Key observation: FerroML already has every testing library needed.** The gap is not tooling -- it is test coverage for edge cases. The hardening work is writing tests, not adding dependencies.

**Recommended edge-case test patterns using existing tools:**

```rust
// proptest: Random valid inputs stay finite
proptest! {
    #[test]
    fn predictions_always_finite(
        x in prop::array::uniform20(prop::num::f64::NORMAL),
        y in prop::array::uniform10(prop::num::f64::NORMAL),
    ) {
        let model = LinearRegression::new();
        let fitted = model.fit(&x_array, &y_array)?;
        let preds = fitted.predict(&x_array)?;
        prop_assert!(preds.iter().all(|v| v.is_finite()));
    }
}

// proptest: NaN/Inf inputs are rejected
proptest! {
    #[test]
    fn nan_inputs_rejected(row in 0..100usize, col in 0..10usize) {
        let mut x = Array2::ones((100, 10));
        x[[row, col]] = f64::NAN;
        let model = LinearRegression::new();
        prop_assert!(model.fit(&x, &y).is_err());
    }
}

// rstest: Parameterized edge cases
#[rstest]
#[case(0, 10)]   // 0 samples
#[case(1, 10)]   // 1 sample
#[case(10, 0)]   // 0 features (if applicable)
#[case(10, 1)]   // 1 feature
fn edge_case_dimensions(#[case] n: usize, #[case] d: usize) {
    // Test model handles gracefully
}
```

**Why NOT add `quickcheck`:** proptest supersedes quickcheck with better shrinking, strategies, and Rust integration. FerroML already uses proptest.

**Why NOT add `float-cmp`:** The `approx` crate (already present) provides `assert_relative_eq!` and `assert_abs_diff_eq!` which cover FerroML's needs. `float-cmp` adds ULP-based comparison which is rarely needed for ML (relative tolerance is the standard).

**Confidence:** HIGH -- all libraries already in Cargo.toml, just need more tests.

### CI/Release Quality Tools (cargo install, not dependencies)

| Tool | Version | Purpose | Install | Confidence |
|------|---------|---------|---------|------------|
| cargo-semver-checks | 0.45 | Lint for SemVer violations before publish | `cargo install cargo-semver-checks` | HIGH |
| cargo-audit | latest | Check for known vulnerabilities in deps | `cargo install cargo-audit` | HIGH |
| cargo-deny | latest | License compliance, duplicate deps, advisories | `cargo install cargo-deny` | MEDIUM |

**Why cargo-semver-checks:** FerroML is publishing to crates.io. With 242 lints as of 2025, this catches accidental API breakage. Run before any version bump. Essential for a library with downstream users.

**Why cargo-audit:** Standard security practice. Checks Cargo.lock against RustSec advisory database. FerroML uses `ort 2.0.0-rc.11` (pre-release) which may have advisories.

**Why cargo-deny is MEDIUM:** Useful but not blocking for launch. FerroML has MIT/Apache-2.0 license -- cargo-deny validates all deps are compatible. Nice-to-have, not must-have for v0.4.0.

**Confidence:** HIGH for semver-checks and audit (industry standard), MEDIUM for deny (optional).

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| SVD backend | faer 0.20 (existing) | ndarray-linalg + OpenBLAS | Explicitly forbidden (system dep) |
| Unwrap auditing | clippy::unwrap_used | cargo-geiger, custom lint | Clippy lint is built-in, no extra tool |
| Input validation | `f64::is_finite()` pattern | `validator` crate | Wrong abstraction (web forms, not arrays) |
| Property testing | proptest 1.5 (existing) | quickcheck | proptest has better shrinking, already used |
| Float comparison | approx 0.5 (existing) | float-cmp | ULP comparison rarely needed for ML |
| Benchmarking | Criterion 0.5 (existing) | Criterion 0.8, codspeed | Migration cost not justified for hardening |
| CPU profiling | cargo-flamegraph 0.6.11 | samply, perf directly | flamegraph is simpler, generates SVG |
| Heap profiling | dhat 0.3 (existing) | heaptrack, valgrind massif | dhat integrates with Rust harness |
| SemVer checking | cargo-semver-checks 0.45 | Manual review | 242 automated lints > human review |

## What NOT to Add

| Do Not Add | Reason |
|------------|--------|
| ndarray-linalg | System dep (OpenBLAS), explicitly forbidden |
| Any new runtime dependencies | PROJECT.md constraint: "No new dependencies" |
| Criterion 0.8 | API-breaking upgrade, 86+ benchmarks would need rewriting |
| Miri | For unsafe memory bugs -- FerroML has minimal unsafe code; clippy + proptest cover more ground |
| Kani (formal verification) | Academic-grade tool, overkill for ML library hardening |
| cargo-mutants | Mutation testing is valuable but slow; not appropriate for a hardening sprint |
| wasm-bindgen tests | WASM out of scope per PROJECT.md |

## Installation

```bash
# Dev tools (cargo install, one-time)
cargo install flamegraph           # CPU profiling (0.6.11)
cargo install cargo-semver-checks  # SemVer linting (0.45)
cargo install cargo-audit          # Vulnerability scanning

# Everything else is already in ferroml-core/Cargo.toml [dev-dependencies]:
# criterion = "0.5"
# approx = "0.5"
# proptest = "1.5"
# rstest = "0.23"
# test-case = "3.3"
# dhat = "0.3"
```

```toml
# Create clippy.toml at workspace root
allow-unwrap-in-tests = true
```

```rust
// Add to ferroml-core/src/lib.rs (after triage, not immediately)
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
```

## Key Insight

**FerroML's hardening gap is not tooling -- it is application of existing tools.** The project already has proptest, approx, rstest, criterion, and dhat. The work is:
1. Writing edge-case tests with proptest (NaN, Inf, empty, single-sample, ill-conditioned)
2. Adding `is_finite()` validation to fit methods
3. Triaging 6,195 unwrap calls and replacing unsafe ones
4. Profiling hot paths with criterion + flamegraph to close performance gaps
5. Using faer (already a dep) to replace nalgebra SVD/Cholesky in bottleneck paths

No new crate additions needed. This is a code-quality milestone, not a stack-change milestone.

## Sources

- [Rust f64::is_finite() documentation](https://doc.rust-lang.org/std/primitive.f64.html) -- HIGH confidence
- [Clippy unwrap_used lint](https://rust-lang.github.io/rust-clippy/master/index.html) -- HIGH confidence
- [Clippy PR #14200: unwrap/expect compile-time linting](https://github.com/rust-lang/rust-clippy/pull/14200) -- HIGH confidence (Feb 2025)
- [cargo-semver-checks 2025 year in review](https://predr.ag/blog/cargo-semver-checks-2025-year-in-review/) -- HIGH confidence
- [cargo-flamegraph 0.6.11](https://github.com/flamegraph-rs/flamegraph) -- HIGH confidence
- [Criterion.rs](https://crates.io/crates/criterion) -- HIGH confidence
- [proptest on crates.io](https://crates.io/crates/proptest) -- HIGH confidence
- [Rust Auditing Tools in 2025](https://markaicode.com/rust-auditing-tools-2025-automated-security-scanning/) -- MEDIUM confidence
- [Rust Static Analysis Tools Comparison 2025](https://markaicode.com/rust-static-analysis-tools-comparison-2025/) -- MEDIUM confidence
