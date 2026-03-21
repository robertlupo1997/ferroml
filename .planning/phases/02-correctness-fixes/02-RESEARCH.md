# Phase 2: Correctness Fixes - Research

**Researched:** 2026-03-21
**Domain:** Numerical correctness, calibration, SVD sign stability, SVM cache testing, convergence reporting
**Confidence:** HIGH

## Summary

Phase 2 addresses 10 correctness requirements across four domains: (1) fixing 6 known test failures in TemperatureScaling and IncrementalPCA, (2) adding SVM kernel cache unit tests, (3) hardening numerical stability (log-sum-exp, Cholesky jitter, svd_flip), and (4) improving convergence reporting and output sanity checks.

The codebase is well-structured for these changes. GMM already has a correct `logsumexp` implementation in `clustering/gmm.rs`. NaiveBayes variants (Gaussian, Multinomial, Categorical, Bernoulli) already use inline log-sum-exp in their `predict_proba` methods. LogisticRegression uses sigmoid (binary case), which is inherently stable. The SVM kernel cache (`KernelCache` struct in `svm.rs`) has a complete LRU implementation but zero unit tests. `linalg.rs` has `cholesky` and `thin_svd` but no jitter fallback or sign normalization.

**Primary recommendation:** Fix the 6 failing tests first (TemperatureScaling/IncrementalPCA), then add SVM cache tests and numerical safeguards, then convergence warnings and output checks.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CORR-01 | Fix TemperatureScaling calibrator -- 3 failing tests pass or model removed | TemperatureScaling implementation found in `calibration.rs:657-927`. Gradient descent optimizer has no line search backtracking and clamps T >= 0.01. Tests are in `test_vs_sklearn_gaps_phase2.py`. Fix optimizer or adjust test expectations. |
| CORR-02 | Fix IncrementalPCA -- 3 failing tests pass or model removed | IncrementalPCA in `decomposition/pca.rs:763-1030+`. Uses augmented SVD approach following Ross et al. Tests compare batch vs full PCA and vs sklearn. Variance tracking may have incremental formula issues. |
| CORR-03 | SVM kernel cache unit tests for shrinking correctness | `KernelCache` struct at `svm.rs:95-300+`. Shrinking logic at `svm.rs:716-745`. Active set management with periodic shrinking. No existing unit tests for cache. |
| CORR-04 | SVM kernel cache unit tests for eviction order and hit rates | LRU linked list implementation at `svm.rs:107-200+`. `lru_remove`, `lru_push_back` methods. Need tests verifying FIFO eviction and hit rate tracking. |
| CORR-05 | Post-predict sanity check for NaN in output | No existing output validation. Add to `Model` trait default impl or as utility function in `models/mod.rs`. |
| CORR-06 | Log-sum-exp in all probability computations | GMM: already has `logsumexp()` function. NaiveBayes (4 variants): already use inline log-sum-exp. LogReg: uses sigmoid (binary, stable). Main gap: consolidate into shared utility, ensure all paths use it. |
| CORR-07 | Cholesky jitter fallback on ill-conditioned matrices | `linalg.rs:364-458`. Both `cholesky_faer` and `cholesky_native` take `reg` param but no automatic jitter retry. GP at `gaussian_process.rs` uses cholesky. |
| CORR-08 | SVD sign normalization (svd_flip) in linalg.rs | `linalg.rs` has `thin_svd` (lines 19-28) dispatching to faer/nalgebra. No sign normalization exists. PCA, LDA, FactorAnalysis, TruncatedSVD all call `thin_svd`. |
| CORR-09 | All 55+ models have cross-library correctness tests | Currently ~200+ cross-library tests across 12 Python test files. Need inventory of untested models and expand coverage. |
| CORR-10 | Convergence warnings instead of only hard errors | `FerroError::ConvergenceFailure` exists but is a hard error. Need a convergence warning mechanism (e.g., `FitResult` wrapper or `log::warn`). |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ndarray | 0.16 | Array types (Array1, Array2) | Core data structures throughout |
| nalgebra | current | SVD fallback (Jacobi) | Used when faer-backend disabled |
| faer | current | High-perf SVD, Cholesky | Primary linear algebra backend |
| serde | 1.x | Serialization | Model persistence |
| log | 0.4 | Logging framework | For convergence warnings (CORR-10) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| approx | 0.5 | Float comparison in tests | All numerical assertion tests |
| argmin | current | L-BFGS optimizer | Used by LogisticRegression |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `log::warn` for convergence | Custom `FitResult<T>` enum | `log::warn` is simpler, but callers can't programmatically detect warnings. Use struct return for convergence status. |

## Architecture Patterns

### Relevant Code Structure
```
ferroml-core/src/
├── linalg.rs             # svd_flip goes here (CORR-08), cholesky jitter (CORR-07)
├── models/
│   ├── mod.rs            # Output sanity check utility (CORR-05)
│   ├── calibration.rs    # TemperatureScaling fix (CORR-01)
│   ├── svm.rs            # KernelCache tests (CORR-03, CORR-04)
│   ├── logistic.rs       # Log-sum-exp audit (CORR-06)
│   └── naive_bayes/      # 4 variants, all need audit (CORR-06)
├── clustering/
│   └── gmm.rs            # Has reference logsumexp() implementation
├── decomposition/
│   └── pca.rs            # IncrementalPCA fix (CORR-02)
├── error.rs              # ConvergenceFailure, convergence warnings (CORR-10)
└── validation.rs         # Input validation (Phase 1 complete)
```

### Pattern 1: svd_flip (Sign Normalization)
**What:** Ensure deterministic SVD component signs by making the largest absolute value in each row of Vt positive.
**When to use:** After every `thin_svd` call, before returning results.
**Example:**
```rust
// sklearn's svd_flip algorithm:
// For each row of Vt, find the column with max absolute value.
// If that value is negative, flip the signs of that row of Vt and corresponding column of U.
pub fn svd_flip(u: &mut Array2<f64>, vt: &mut Array2<f64>) {
    let k = vt.nrows();
    for i in 0..k {
        let row = vt.row(i);
        let max_abs_col = row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        if vt[[i, max_abs_col]] < 0.0 {
            vt.row_mut(i).mapv_inplace(|v| -v);
            u.column_mut(i).mapv_inplace(|v| -v);
        }
    }
}
```

### Pattern 2: Cholesky with Jitter Fallback
**What:** On Cholesky failure, retry with increasing diagonal jitter.
**When to use:** In `linalg::cholesky` wrapper.
**Example:**
```rust
pub fn cholesky_with_jitter(a: &Array2<f64>, reg: f64) -> Result<Array2<f64>> {
    // Try without jitter first
    if let Ok(l) = cholesky(a, reg) {
        return Ok(l);
    }
    // Retry with increasing jitter: 1e-10, 1e-8, 1e-6, 1e-4
    for &jitter in &[1e-10, 1e-8, 1e-6, 1e-4] {
        if let Ok(l) = cholesky(a, reg + jitter) {
            log::warn!("Cholesky required jitter={:.0e} for numerical stability", jitter);
            return Ok(l);
        }
    }
    Err(FerroError::numerical("Matrix not positive definite even with jitter"))
}
```

### Pattern 3: Convergence Warning via FitResult
**What:** Return convergence status alongside fitted model state.
**When to use:** Models with iterative solvers (KMeans, GMM, LogReg, SVM).
**Example:**
```rust
/// Convergence status returned from iterative fitting.
pub enum ConvergenceStatus {
    Converged { iterations: usize },
    NotConverged { iterations: usize, final_change: f64 },
}

// In fit method: store status, log warning, but don't error
if !converged {
    log::warn!(
        "{} did not converge after {} iterations (change={:.2e}). Consider increasing max_iter.",
        model_name, max_iter, final_change
    );
    self.convergence_status = Some(ConvergenceStatus::NotConverged { iterations: max_iter, final_change });
}
```

### Anti-Patterns to Avoid
- **Hard error on non-convergence when partial result is usable:** KMeans, GMM, and similar should return the best-so-far result with a warning, not fail. SVM with no support vectors is a legitimate hard error.
- **Per-model logsumexp implementations:** The GMM has a standalone `logsumexp()` fn. The NaiveBayes models have inline versions. Consolidate into a shared utility.
- **Testing kernel cache through full SVM fit:** Cache tests should construct `KernelCache` directly and test LRU mechanics in isolation.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Log-sum-exp | Inline per-model implementations | Shared `logsumexp()` utility in linalg or stats module | Already exists in GMM; 4 NaiveBayes variants have duplicated inline code |
| SVD sign normalization | Per-decomposition sign fixes | `svd_flip()` in linalg.rs called from `thin_svd` | sklearn's standard approach; ensures consistency across PCA/LDA/FA/TruncatedSVD |
| Convergence warning system | Custom logging per model | `ConvergenceStatus` enum + `log::warn` | Consistent API across all iterative models |

**Key insight:** The GMM's `logsumexp` function (lines 998-1005) is the reference implementation. It correctly handles the max-subtract trick. The NaiveBayes variants do the same thing inline. Consolidate, don't duplicate.

## Common Pitfalls

### Pitfall 1: TemperatureScaling Gradient Descent Instability
**What goes wrong:** The optimizer uses fixed learning rate (0.01) with no line search or adaptive step. Temperature is clamped to >= 0.01 but gradient can oscillate.
**Why it happens:** Temperature scaling is a 1D convex optimization problem, but the gradient `d(NLL)/d(T) = (1/T^2) * sum(...)` has a `1/T^2` factor that amplifies near small T values.
**How to avoid:** Use bisection search, Newton's method with Hessian, or at minimum adaptive learning rate. sklearn's CalibratedClassifierCV uses L-BFGS for Platt scaling. Consider using argmin's L-BFGS (already a dependency) for this 1D problem or implement golden-section search.
**Warning signs:** Temperature converges to 0.01 (the clamp), or oscillates without reducing NLL.

### Pitfall 2: IncrementalPCA Variance Tracking
**What goes wrong:** Incremental variance update formula (Welford's) can accumulate numerical error over many batches. The variance combination formula in `partial_fit` (lines 881-892) is an approximation.
**Why it happens:** The comment says "approximation" -- the mean correction term `old_n * n_samples / (new_n * new_n)` should be `old_n * n_samples / new_n` (without extra division by new_n).
**How to avoid:** Cross-validate against sklearn's IncrementalPCA which uses a proper two-pass or Chan-Golub-Welford formula. The critical path is the augmented SVD matrix construction, not the variance tracking.
**Warning signs:** Explained variance ratios don't sum to <= 1.0, or components diverge from batch PCA.

### Pitfall 3: svd_flip Must Be Applied Inside thin_svd
**What goes wrong:** If svd_flip is a separate function called after thin_svd, callers might forget to call it.
**Why it happens:** Decomposition modules (PCA, LDA, FA, TruncatedSVD) all call `thin_svd` directly.
**How to avoid:** Apply svd_flip inside `thin_svd()` itself, so all callers get consistent signs automatically. This matches sklearn where `_fit_full` and `_fit_truncated` both call `svd_flip`.

### Pitfall 4: Cholesky Jitter Can Mask Real Problems
**What goes wrong:** Adding jitter to make Cholesky succeed can hide genuinely ill-conditioned problems.
**Why it happens:** Some matrices are ill-conditioned because the underlying model is wrong, not because of floating point.
**How to avoid:** Log a warning when jitter is needed, and cap jitter at 1e-4. If even 1e-4 doesn't work, it's a real problem. GP models (gaussian_process.rs) are the primary consumer.

### Pitfall 5: Integration Test Binary Consolidation Rule
**What goes wrong:** Adding new test files in `ferroml-core/tests/` creates new binaries, bloating debug build size.
**Why it happens:** Each `.rs` file in `tests/` is a separate binary.
**How to avoid:** SVM cache tests go as `#[cfg(test)] mod tests` inside `svm.rs` or in existing `ferroml-core/tests/correctness.rs`. Do NOT create new test files.

## Code Examples

### Shared logsumexp Utility
```rust
// Move from clustering/gmm.rs to a shared location (e.g., linalg.rs or stats/mod.rs)
/// Compute log(sum(exp(x))) in a numerically stable way.
pub fn logsumexp(x: &[f64]) -> f64 {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum_exp: f64 = x.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

// Row-wise variant for predict_proba
pub fn logsumexp_rows(log_probs: &Array2<f64>) -> Array1<f64> {
    let n = log_probs.nrows();
    let mut result = Array1::zeros(n);
    for i in 0..n {
        let row: Vec<f64> = log_probs.row(i).to_vec();
        result[i] = logsumexp(&row);
    }
    result
}
```

### Output NaN Check (CORR-05)
```rust
/// Check model output for NaN/Inf values. Returns error if any found.
pub fn validate_output(predictions: &Array1<f64>, model_name: &str) -> Result<()> {
    if let Some(pos) = predictions.iter().position(|v| !v.is_finite()) {
        return Err(FerroError::numerical(format!(
            "{} produced non-finite output at index {} (value: {}). \
             This indicates a numerical issue in the model.",
            model_name, pos, predictions[pos]
        )));
    }
    Ok(())
}
```

### SVM KernelCache Isolated Test
```rust
#[cfg(test)]
mod cache_tests {
    use super::KernelCache;

    #[test]
    fn test_lru_eviction_order() {
        // Small cache: 3 slots, 5 features
        let mut cache = KernelCache::new(5, 3);
        // Insert rows 0, 1, 2 (fills cache)
        cache.get_or_insert(0, |row, buf| { /* fill */ });
        cache.get_or_insert(1, |row, buf| { /* fill */ });
        cache.get_or_insert(2, |row, buf| { /* fill */ });
        // Insert row 3 -- should evict row 0 (LRU)
        cache.get_or_insert(3, |row, buf| { /* fill */ });
        assert!(!cache.contains(0)); // evicted
        assert!(cache.contains(1));  // still present
    }

    #[test]
    fn test_cache_hit_promotes_to_mru() {
        let mut cache = KernelCache::new(5, 3);
        cache.get_or_insert(0, |_, _| {});
        cache.get_or_insert(1, |_, _| {});
        cache.get_or_insert(2, |_, _| {});
        // Access row 0 again -- promotes it, so row 1 becomes LRU
        cache.get_or_insert(0, |_, _| {});
        cache.get_or_insert(3, |_, _| {}); // evicts row 1 (now LRU)
        assert!(cache.contains(0));  // promoted, not evicted
        assert!(!cache.contains(1)); // evicted
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hard error on non-convergence | Warning + partial result (sklearn pattern) | sklearn 0.21+ (ConvergenceWarning) | Users get useful results even without convergence |
| No SVD sign normalization | svd_flip applied after every SVD | Always standard in sklearn | Deterministic components across runs/backends |
| Fixed regularization for Cholesky | Automatic jitter retry | Standard in GPy, GPflow, sklearn GP | Handles ill-conditioned kernels automatically |
| Per-model logsumexp | Shared utility | N/A (consolidation) | Reduces code duplication, ensures consistency |

## Open Questions

1. **TemperatureScaling: fix or remove?**
   - What we know: The implementation exists and compiles. The optimizer may have gradient issues.
   - What's unclear: Whether the tests fail due to optimizer convergence or correctness of the algorithm itself.
   - Recommendation: Diagnose first. If optimizer is the issue, switch to L-BFGS (argmin already available). If algorithm is wrong, fix. Removal is last resort since the model is useful.

2. **IncrementalPCA: exact cause of test failures?**
   - What we know: The variance combination formula has a suspicious extra division by `new_n`. The augmented SVD approach looks correct.
   - What's unclear: Whether the 8 test failures (not 3 as originally stated -- there are 8 IncrementalPCA tests) are due to variance formula or something else.
   - Recommendation: Build module, run tests, diagnose from actual error output. The variance formula at line 889 looks like the primary suspect.

3. **CORR-09 scope: what counts as "basic" cross-library test?**
   - What we know: 200+ cross-library tests exist across 12 files. 55+ models exposed.
   - What's unclear: Exactly which models lack any cross-library test.
   - Recommendation: Generate inventory during planning. A "basic" test = fit on small dataset, compare predictions to sklearn within tolerance.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust: built-in #[test] + cargo test; Python: pytest |
| Config file | pytest: ferroml-python/pytest.ini or pyproject.toml |
| Quick run command | `cargo test` (Rust core only) |
| Full suite command | `cargo test --all && pytest ferroml-python/tests/` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORR-01 | TemperatureScaling calibration tests pass | integration | `pytest ferroml-python/tests/test_vs_sklearn_gaps_phase2.py::TestTemperatureScalingVsSklearn -x` | Exists (currently failing) |
| CORR-02 | IncrementalPCA tests pass | integration | `pytest ferroml-python/tests/test_vs_sklearn_gaps_phase2.py::TestIncrementalPCAVsSklearn -x` | Exists (currently failing) |
| CORR-03 | SVM cache shrinking correctness | unit | `cargo test svm::cache_tests::test_shrinking -p ferroml-core` | Wave 0 |
| CORR-04 | SVM cache eviction + hit rates | unit | `cargo test svm::cache_tests -p ferroml-core` | Wave 0 |
| CORR-05 | NaN output detection | unit | `cargo test validate_output -p ferroml-core` | Wave 0 |
| CORR-06 | Log-sum-exp in probability models | unit | `cargo test logsumexp -p ferroml-core` | Partial (GMM has tests) |
| CORR-07 | Cholesky jitter fallback | unit | `cargo test cholesky_jitter -p ferroml-core` | Wave 0 |
| CORR-08 | SVD sign normalization | unit | `cargo test svd_flip -p ferroml-core` | Wave 0 |
| CORR-09 | Cross-library test expansion | integration | `pytest ferroml-python/tests/test_vs_sklearn*.py` | Partial |
| CORR-10 | Convergence warnings | unit + integration | `cargo test convergence -p ferroml-core` | Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test` (quick Rust tests)
- **Per wave merge:** `cargo test --all && pytest ferroml-python/tests/`
- **Phase gate:** Full suite green before verify-work

### Wave 0 Gaps
- [ ] SVM `KernelCache` unit tests (CORR-03, CORR-04) -- no existing cache tests
- [ ] `validate_output` utility function and tests (CORR-05)
- [ ] `svd_flip` function and tests in `linalg.rs` (CORR-08)
- [ ] `cholesky_with_jitter` wrapper and tests (CORR-07)
- [ ] `ConvergenceStatus` enum and convergence warning tests (CORR-10)
- [ ] Shared `logsumexp` utility tests (CORR-06)

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `ferroml-core/src/models/calibration.rs` (TemperatureScaling implementation)
- Direct code inspection of `ferroml-core/src/decomposition/pca.rs` (IncrementalPCA implementation)
- Direct code inspection of `ferroml-core/src/models/svm.rs` (KernelCache, shrinking logic)
- Direct code inspection of `ferroml-core/src/linalg.rs` (thin_svd, cholesky, no svd_flip)
- Direct code inspection of `ferroml-core/src/clustering/gmm.rs` (logsumexp reference impl)
- Direct code inspection of `ferroml-core/src/models/naive_bayes/*.rs` (inline log-sum-exp patterns)
- Direct code inspection of `ferroml-core/src/models/logistic.rs` (sigmoid-based predict_proba)
- Direct code inspection of `ferroml-core/src/error.rs` (ConvergenceFailure variant)
- Direct test execution of `test_vs_sklearn_gaps_phase2.py` (12 failures confirmed)

### Secondary (MEDIUM confidence)
- sklearn source code patterns (svd_flip, ConvergenceWarning, jitter retry)
- Memory/project notes on test counts and architecture

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in Cargo.toml, inspected directly
- Architecture: HIGH - all relevant files read and patterns identified
- Pitfalls: HIGH - identified from direct code inspection (variance formula, optimizer issues)

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable domain, no external dependency changes expected)
