# FerroML Quality Hardening Plan

**Date**: 2026-02-03
**Status**: Draft
**Scope**: Address AI-loop failure patterns, apply Rust best practices, implement remaining features

---

## Executive Summary

This plan addresses quality issues commonly found in rapid AI development loops ("ralph loops") and implements remaining work from the bug audit. Research indicates AI-generated code has 1.75x more logic errors, 1.57x more security issues, and struggles with recursion, edge cases, and numerical precision.

---

## Part 1: AI-Loop Failure Pattern Audit

### 1.1 Edge Case & Boundary Validation
**Priority**: High
**Rationale**: AI code often produces syntactically correct code that fails under edge cases

| Check | Files to Audit | Action |
|-------|----------------|--------|
| Empty input handling | All `fit()` and `predict()` methods | Verify guards exist |
| Single-element arrays | Tree splits, scalers, metrics | Test edge behavior |
| NaN/Inf propagation | All numerical code | Ensure explicit handling |
| Zero-division guards | Metrics, normalization | Check denominators |
| Integer overflow (usize) | Loop indices, array sizing | Verify `.saturating_*` or checks |

### 1.2 Recursion & Termination Logic
**Priority**: High
**Rationale**: AI frequently fails on recursive termination and stack depth

| Check | Location | Action |
|-------|----------|--------|
| Tree depth limits | `tree.rs`, `hist_boosting.rs` | Verify max_depth enforcement |
| Recursive splits | Decision tree building | Check base cases |
| Stack safety | Deep tree traversal | Consider iterative alternatives |

### 1.3 Error Handling Quality
**Priority**: Medium
**Rationale**: Generic error messages obscure root causes

| Pattern | Current State | Target |
|---------|--------------|--------|
| `unwrap()` usage | ~50 occurrences | Replace with `?` or context |
| `expect("...")` | Generic messages | Descriptive context |
| Silent failures | Ensemble error swallowing | Proper propagation |
| Panic in library code | `unreachable!()` | Return `Result` |

### 1.4 Test Quality Audit
**Priority**: High
**Rationale**: AI tests often test assumptions, not intent

| Check | Action |
|-------|--------|
| Edge case coverage | Add tests for empty, single, boundary inputs |
| Domain constraints | Verify ML-specific invariants (probabilities sum to 1) |
| Integration tests | Test full pipelines, not just units |
| Regression suite | Capture known-good outputs |

---

## Part 2: Rust Best Practices Application

### 2.1 Idiomatic Patterns
| Pattern | Current | Target |
|---------|---------|--------|
| Error handling | Mixed `unwrap`/`expect` | `Result` propagation with `?` |
| String parameters | `String` ownership | Accept `impl AsRef<str>` |
| Boolean params | `bool` flags | Enums for clarity |
| Iterator usage | Some imperative loops | Functional chains where cleaner |

### 2.2 Performance Patterns
| Issue | Location | Fix |
|-------|----------|-----|
| `VecDeque` for FIFO | Check queue usage | Replace `Vec::remove(0)` |
| `mul_add()` precision | 1,625 opportunities | Apply in numerical hotspots |
| Clone reduction | Completed in P3 | Maintain discipline |
| HashSet efficiency | Completed in P3 | Maintain pattern |

### 2.3 API Design
| Guideline | Application |
|-----------|-------------|
| Builder pattern | Already used for models |
| Extension traits | Consider for preprocessing pipelines |
| Trait bounds | Use `impl Trait` for ergonomic APIs |
| Documentation | Examples in all public APIs |

---

## Part 3: Implementation Tasks

### Phase 1: Ignored Tests (5 tests)
**Location**: `ferroml-core/src/testing/properties.rs`
**Issue**: Property tests marked `#[ignore]` due to long runtime

| Test | Lines | What It Tests |
|------|-------|---------------|
| `test_tree_predictions_bounded` | 955-963 | Tree predictions within training range |
| `test_forest_predictions_bounded` | 965-977 | Forest predictions bounded |
| `test_logistic_regression_probabilities_sum_to_one` | 980-990 | Proba normalization |
| `test_logistic_regression_probabilities_in_range` | 992-1000 | Proba in [0,1] |
| `test_gaussian_nb_probabilities_sum_to_one` | 1002-1008 | GaussianNB normalization |

**Actions**:
1. Profile each test to identify slowness source
2. Reduce sample/iteration counts for faster execution
3. Consider proptest `#![proptest_config]` with fewer cases
4. Enable in CI with `--ignored` on nightly/weekly runs
5. Add quick smoke-test versions that run in standard suite

### Phase 2: Slow Compliance Test
**Test**: `test_random_forest_classifier_compliance`
**Location**: `compliance.rs:218-239`

**Root Causes Identified**:
- 10 trees × 25 checks × 100 samples = excessive computation
- OOB score computed unnecessarily during compliance testing
- `check_subset_invariance` does 100 individual predictions instead of batch
- Cholesky decomposition in feature importance CI calculation

**Optimizations**:
1. Reduce `n_estimators` from 10 → 5
2. Reduce `n_samples` from 100 → 50
3. Skip `check_fit_idempotent` (random forests are non-deterministic)
4. Disable OOB score computation in test config
5. Batch predictions in `check_subset_invariance`

**Expected Speedup**: 3-5x

### Phase 3: Missing Features

#### 3.1 TargetEncoder (TASK-021)
**Location**: `preprocessing/encoders.rs`
**Priority**: High (referenced in module docs)

**Implementation**:
```rust
pub struct TargetEncoder {
    encoding_map: HashMap<String, HashMap<String, f64>>,
    smoothing: f64,
    global_mean: f64,
}
```

**Key Methods**:
- `fit(X: &[Vec<String>], y: &[f64])` - Compute target means per category
- `transform(X: &[Vec<String>])` - Replace categories with smoothed means
- Smoothing: `(category_mean * n + global_mean * m) / (n + m)`

**Tests**:
- High-cardinality handling
- Unseen categories (fallback to global mean)
- Multiclass target support

#### 3.2 KNNImputer
**Location**: `preprocessing/imputers.rs`
**Priority**: Medium

**Implementation**:
```rust
pub struct KNNImputer {
    n_neighbors: usize,
    weights: WeightFunction, // uniform or distance
    metric: DistanceMetric,
    fitted_data: Option<Array2<f64>>,
}
```

**Key Considerations**:
- Handle NaN in distance calculation (skip NaN features)
- Efficient neighbor search (consider KD-tree for large datasets)
- Memory: store training data for imputation

#### 3.3 Other TODOs
| Item | Location | Complexity |
|------|----------|------------|
| `check_classifier` function | compliance.rs:169 | Low |
| Statistical power calculation | hypothesis.rs:193 | Medium |
| Condition number in LinearRegression | linear.rs:403 | Low |
| sklearn timing script | benches/benchmarks.rs:14 | Low |

### Phase 4: mul_add Optimization
**Scope**: 1,625 opportunities across 15+ files
**Priority**: Low (micro-optimization, but improves precision)

**Critical Files** (do first):
1. `simd.rs` (214 opportunities) - SIMD remainder loops
2. `hpo/bayesian.rs` (78 opportunities) - Cholesky decomposition
3. `models/linear.rs` (55 opportunities) - Regression coefficients

**Pattern Transformation**:
```rust
// Before
sum += a[i] * b[i];

// After (better precision, often faster)
sum = a[i].mul_add(b[i], sum);
```

**Automation**: Use `cargo clippy` with `suboptimal_flops` lint or custom script

---

## Part 4: AutoML Best Practices

### 4.1 Transparency (from research)
| Principle | Current | Target |
|-----------|---------|--------|
| Decision logging | Minimal | Add search history export |
| Performance comparison | Basic | Show improvement deltas |
| Component visibility | Good | Maintain |

### 4.2 Extensibility
| Principle | Current | Target |
|-----------|---------|--------|
| Custom models | Trait-based | Good, maintain |
| Custom metrics | Supported | Good, maintain |
| Search space config | Builder pattern | Good, maintain |

### 4.3 Efficiency (FLAML-inspired)
| Principle | Current | Target |
|-----------|---------|--------|
| Cost-aware optimization | Unknown | Consider budget limits |
| Early stopping | Implemented | Verify effectiveness |
| Warm starting | Implemented | Good |

---

## Implementation Order

### Week 1: Foundation
1. [ ] Audit edge cases (1.1) - 4 hours
2. [ ] Audit recursion safety (1.2) - 2 hours
3. [ ] Fix slow compliance test (Phase 2) - 2 hours
4. [ ] Enable ignored tests with optimizations (Phase 1) - 3 hours

### Week 2: Features
5. [ ] Implement TargetEncoder (3.1) - 4 hours
6. [ ] Implement `check_classifier` function - 1 hour
7. [ ] Add condition number computation - 1 hour
8. [ ] Statistical power calculation - 2 hours

### Week 3: Polish
9. [ ] KNNImputer implementation (3.2) - 4 hours
10. [ ] mul_add optimization in simd.rs - 2 hours
11. [ ] mul_add optimization in bayesian.rs - 2 hours
12. [ ] mul_add optimization in linear.rs - 1 hour

### Week 4: Quality
13. [ ] Error handling audit (1.3) - 3 hours
14. [ ] Test quality audit (1.4) - 4 hours
15. [ ] Documentation review - 2 hours
16. [ ] Final integration testing - 2 hours

---

## Verification Commands

```bash
# Full test suite
cargo test -p ferroml-core --lib

# Include ignored (slow) tests
cargo test -p ferroml-core -- --ignored

# Clippy with all warnings
cargo clippy -p ferroml-core -- -D warnings

# Check for mul_add opportunities
cargo clippy -p ferroml-core -- -W clippy::suboptimal_flops

# Benchmark compliance test
cargo test -p ferroml-core test_random_forest_classifier_compliance -- --nocapture
```

---

## Success Criteria

1. **All 5 ignored tests enabled** with reasonable runtime (<30s each)
2. **Compliance test runtime reduced** by 3x or more
3. **TargetEncoder implemented** with full test coverage
4. **KNNImputer implemented** with full test coverage
5. **Critical mul_add patterns** applied in simd.rs, bayesian.rs, linear.rs
6. **Zero clippy warnings** maintained
7. **Edge case tests added** for all major algorithms

---

## Sources

- [IEEE Spectrum: AI Coding Degrades](https://spectrum.ieee.org/ai-coding-degrades)
- [CodeRabbit: AI vs Human Code Generation Report](https://www.coderabbit.ai/blog/state-of-ai-vs-human-code-generation-report)
- [Addy Osmani: LLM Coding Workflow 2026](https://addyosmani.com/blog/ai-coding-workflow/)
- [Idiomatic Rust Collection](https://github.com/mre/idiomatic-rust)
- [AutoML Design Patterns](https://link.springer.com/chapter/10.1007/978-3-032-07986-2_5)
- [H2O AutoML Framework](https://openml.github.io/automlbenchmark/frameworks.html)
