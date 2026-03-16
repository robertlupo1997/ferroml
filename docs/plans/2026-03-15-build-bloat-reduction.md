# Build Bloat Reduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce debug build disk usage from ~350 GB to ~50 GB by consolidating integration test files from 19 separate binaries down to 6, and making heavy optional dependencies non-default.

**Architecture:** Each `.rs` file in `tests/` compiles as its own binary, re-linking the entire dependency tree. By merging related test files into single files with `mod` blocks, we drastically reduce link jobs. We also gate the heaviest dependencies (`polars`, `wgpu`, `faer`) behind non-default features so casual `cargo test` doesn't pull them in.

**Tech Stack:** Rust, Cargo, ndarray, ferroml-core

---

## Consolidation Map

| New File | Old Files Merged In | Test Count |
|----------|-------------------|------------|
| `correctness.rs` | `correctness_clustering.rs` (111), `correctness_explainability.rs` (20), `correctness_neural.rs` (48), `correctness_preprocessing.rs` (99) | ~278 |
| `adversarial.rs` | `adversarial_models.rs` (56), `adversarial_preprocessing_metrics.rs` (39) | ~95 |
| `regression.rs` | `regression_baselines.rs` (1), `regression_guards.rs` (16) | ~17 |
| `vs_linfa.rs` | `vs_linfa_clustering.rs` (7), `vs_linfa_linear.rs` (14), `vs_linfa_naive_bayes.rs` (7), `vs_linfa_neighbors.rs` (9), `vs_linfa_svm.rs` (6), `vs_linfa_trees.rs` (11) | ~54 |
| `edge_cases.rs` | `edge_case_matrix.rs` (98), `sparse_pipeline_e2e.rs` (30) | ~128 |
| `integration.rs` | `integration_uci_datasets.rs` (15), `real_dataset_validation.rs` (31), `sklearn_correctness.rs` (69) | ~115 |

**Result:** 19 binaries -> 6 binaries (68% reduction)

---

### Task 1: Consolidate `vs_linfa_*.rs` -> `vs_linfa.rs`

These 6 files are the simplest — small, uniform structure, no feature gates, all import from `ferroml_core` and `linfa*`.

**Files:**
- Delete: `ferroml-core/tests/vs_linfa_clustering.rs` (239 lines)
- Delete: `ferroml-core/tests/vs_linfa_linear.rs` (350 lines)
- Delete: `ferroml-core/tests/vs_linfa_naive_bayes.rs` (231 lines)
- Delete: `ferroml-core/tests/vs_linfa_neighbors.rs` (157 lines)
- Delete: `ferroml-core/tests/vs_linfa_svm.rs` (200 lines)
- Delete: `ferroml-core/tests/vs_linfa_trees.rs` (224 lines)
- Create: `ferroml-core/tests/vs_linfa.rs`

**Step 1: Create the consolidated file**

Create `ferroml-core/tests/vs_linfa.rs` with this structure:

```rust
//! Cross-library validation tests: FerroML vs linfa
//!
//! Consolidated from vs_linfa_{clustering,linear,naive_bayes,neighbors,svm,trees}.rs

mod clustering {
    // paste entire contents of vs_linfa_clustering.rs here (minus any top-level use/extern)
}

mod linear {
    // paste entire contents of vs_linfa_linear.rs here
}

mod naive_bayes {
    // paste entire contents of vs_linfa_naive_bayes.rs here
}

mod neighbors {
    // paste entire contents of vs_linfa_neighbors.rs here
}

mod svm {
    // paste entire contents of vs_linfa_svm.rs here
}

mod trees {
    // paste entire contents of vs_linfa_trees.rs here
}
```

Each old file's contents go inside its `mod` block. Move `use` statements that were at the top of each file into their respective `mod` block (so each module has its own imports).

**Step 2: Verify it compiles**

Run: `cargo test -p ferroml-core --test vs_linfa --no-run 2>&1 | tail -5`
Expected: `Compiling` then `Finished`

**Step 3: Run all the tests**

Run: `cargo test -p ferroml-core --test vs_linfa -- --quiet 2>&1 | tail -5`
Expected: `54 passed; 0 failed` (approximately)

**Step 4: Delete the old files**

```bash
rm ferroml-core/tests/vs_linfa_clustering.rs
rm ferroml-core/tests/vs_linfa_linear.rs
rm ferroml-core/tests/vs_linfa_naive_bayes.rs
rm ferroml-core/tests/vs_linfa_neighbors.rs
rm ferroml-core/tests/vs_linfa_svm.rs
rm ferroml-core/tests/vs_linfa_trees.rs
```

**Step 5: Verify no old binaries referenced**

Run: `cargo test -p ferroml-core --test vs_linfa -- --quiet 2>&1 | tail -5`
Expected: Same pass count as step 3

**Step 6: Commit**

```bash
git add ferroml-core/tests/
git commit -m "refactor: consolidate 6 vs_linfa test files into one binary"
```

---

### Task 2: Consolidate `adversarial_*.rs` -> `adversarial.rs`

Two files with similar structure (adversarial/edge-case inputs).

**Files:**
- Rename: `ferroml-core/tests/adversarial_models.rs` -> becomes `mod models` inside new file
- Delete: `ferroml-core/tests/adversarial_preprocessing_metrics.rs` -> becomes `mod preprocessing_metrics`
- Create: `ferroml-core/tests/adversarial.rs`

**Step 1: Create the consolidated file**

Create `ferroml-core/tests/adversarial.rs`:

```rust
//! Adversarial and edge-case input tests
//!
//! Consolidated from adversarial_models.rs and adversarial_preprocessing_metrics.rs

mod models {
    // paste adversarial_models.rs contents here (with its use statements)
}

mod preprocessing_metrics {
    // paste adversarial_preprocessing_metrics.rs contents here (with its use statements)
}
```

**Step 2: Verify it compiles**

Run: `cargo test -p ferroml-core --test adversarial --no-run 2>&1 | tail -5`
Expected: Compiles successfully

**Step 3: Run all the tests**

Run: `cargo test -p ferroml-core --test adversarial -- --quiet 2>&1 | tail -5`
Expected: ~95 passed; 0 failed

**Step 4: Delete old files**

```bash
rm ferroml-core/tests/adversarial_models.rs
rm ferroml-core/tests/adversarial_preprocessing_metrics.rs
```

**Step 5: Verify**

Run: `cargo test -p ferroml-core --test adversarial -- --quiet 2>&1 | tail -5`
Expected: Same pass count

**Step 6: Commit**

```bash
git add ferroml-core/tests/
git commit -m "refactor: consolidate 2 adversarial test files into one binary"
```

---

### Task 3: Consolidate `regression_*.rs` -> `regression.rs`

Two files. Note: there's a `tests/regression/` directory with `baselines.json` — this is a data directory, not a Rust module. The new `regression.rs` file will sit alongside it (Cargo treats `tests/regression.rs` and `tests/regression/` as a module with submodules — this is a Rust 2018+ path issue). **If there's a conflict**, rename the data directory to `tests/regression_data/` and update the path reference in the code.

**Files:**
- Delete: `ferroml-core/tests/regression_baselines.rs`
- Delete: `ferroml-core/tests/regression_guards.rs`
- Create: `ferroml-core/tests/regression_tests.rs` (use `_tests` suffix to avoid conflict with `regression/` dir)

**Step 1: Check for path conflict**

Run: `ls -la ferroml-core/tests/regression/`
If directory exists, we name our file `regression_tests.rs` to avoid the Rust module path ambiguity.

**Step 2: Create the consolidated file**

Create `ferroml-core/tests/regression_tests.rs`:

```rust
//! Regression baseline and guard tests
//!
//! Consolidated from regression_baselines.rs and regression_guards.rs

mod baselines {
    // paste regression_baselines.rs contents here
}

mod guards {
    // paste regression_guards.rs contents here
}
```

**Step 3: Verify it compiles**

Run: `cargo test -p ferroml-core --test regression_tests --no-run 2>&1 | tail -5`

**Step 4: Run tests**

Run: `cargo test -p ferroml-core --test regression_tests -- --quiet 2>&1 | tail -5`
Expected: ~17 passed

**Step 5: Delete old files**

```bash
rm ferroml-core/tests/regression_baselines.rs
rm ferroml-core/tests/regression_guards.rs
```

**Step 6: Commit**

```bash
git add ferroml-core/tests/
git commit -m "refactor: consolidate 2 regression test files into one binary"
```

---

### Task 4: Consolidate `correctness_*.rs` -> `correctness.rs`

The biggest merge — 4 files totaling ~7,000 lines. Same mechanical process.

**Files:**
- Delete: `ferroml-core/tests/correctness_clustering.rs` (2,080 lines)
- Delete: `ferroml-core/tests/correctness_explainability.rs` (634 lines)
- Delete: `ferroml-core/tests/correctness_neural.rs` (1,437 lines)
- Delete: `ferroml-core/tests/correctness_preprocessing.rs` (2,827 lines)
- Create: `ferroml-core/tests/correctness.rs`

**Step 1: Create the consolidated file**

Create `ferroml-core/tests/correctness.rs`:

```rust
//! Correctness tests for all major modules
//!
//! Consolidated from correctness_{clustering,explainability,neural,preprocessing}.rs

mod clustering {
    // paste correctness_clustering.rs contents here
}

mod explainability {
    // paste correctness_explainability.rs contents here
}

mod neural {
    // paste correctness_neural.rs contents here
}

mod preprocessing {
    // paste correctness_preprocessing.rs contents here
}
```

**Step 2: Verify it compiles**

Run: `cargo test -p ferroml-core --test correctness --no-run 2>&1 | tail -5`
Expected: Compiles (may take a minute)

**Step 3: Run tests**

Run: `cargo test -p ferroml-core --test correctness -- --quiet 2>&1 | tail -5`
Expected: ~278 passed

**Step 4: Delete old files**

```bash
rm ferroml-core/tests/correctness_clustering.rs
rm ferroml-core/tests/correctness_explainability.rs
rm ferroml-core/tests/correctness_neural.rs
rm ferroml-core/tests/correctness_preprocessing.rs
```

**Step 5: Commit**

```bash
git add ferroml-core/tests/
git commit -m "refactor: consolidate 4 correctness test files into one binary"
```

---

### Task 5: Consolidate `edge_case_matrix.rs` + `sparse_pipeline_e2e.rs` -> `edge_cases.rs`

**Files:**
- Delete: `ferroml-core/tests/edge_case_matrix.rs` (1,408 lines)
- Delete: `ferroml-core/tests/sparse_pipeline_e2e.rs` (908 lines)
- Create: `ferroml-core/tests/edge_cases.rs`

**Step 1: Create the consolidated file**

```rust
//! Edge case and sparse pipeline tests
//!
//! Consolidated from edge_case_matrix.rs and sparse_pipeline_e2e.rs

mod edge_case_matrix {
    // paste edge_case_matrix.rs contents here
}

mod sparse_pipeline {
    // paste sparse_pipeline_e2e.rs contents here
}
```

**Step 2: Verify compiles and tests pass**

Run: `cargo test -p ferroml-core --test edge_cases --no-run 2>&1 | tail -5`
Then: `cargo test -p ferroml-core --test edge_cases -- --quiet 2>&1 | tail -5`
Expected: ~128 passed

**Step 3: Delete old files and commit**

```bash
rm ferroml-core/tests/edge_case_matrix.rs
rm ferroml-core/tests/sparse_pipeline_e2e.rs
git add ferroml-core/tests/
git commit -m "refactor: consolidate edge case and sparse pipeline tests into one binary"
```

---

### Task 6: Consolidate remaining integration tests -> `integration.rs`

**Files:**
- Delete: `ferroml-core/tests/integration_uci_datasets.rs` (499 lines)
- Delete: `ferroml-core/tests/real_dataset_validation.rs` (558 lines)
- Delete: `ferroml-core/tests/sklearn_correctness.rs` (1,446 lines)
- Create: `ferroml-core/tests/integration.rs`

**Step 1: Create the consolidated file**

```rust
//! Integration tests: UCI datasets, real dataset validation, sklearn fixtures
//!
//! Consolidated from integration_uci_datasets.rs, real_dataset_validation.rs, sklearn_correctness.rs

mod uci_datasets {
    // paste integration_uci_datasets.rs contents here
}

mod real_datasets {
    // paste real_dataset_validation.rs contents here
}

mod sklearn_correctness {
    // paste sklearn_correctness.rs contents here
}
```

**Step 2: Verify compiles and tests pass**

Run: `cargo test -p ferroml-core --test integration --no-run 2>&1 | tail -5`
Then: `cargo test -p ferroml-core --test integration -- --quiet 2>&1 | tail -5`
Expected: ~115 passed

**Step 3: Delete old files and commit**

```bash
rm ferroml-core/tests/integration_uci_datasets.rs
rm ferroml-core/tests/real_dataset_validation.rs
rm ferroml-core/tests/sklearn_correctness.rs
git add ferroml-core/tests/
git commit -m "refactor: consolidate 3 integration test files into one binary"
```

---

### Task 7: Full verification

After all consolidations, verify the complete test suite.

**Step 1: List remaining test files**

Run: `ls ferroml-core/tests/*.rs`
Expected: 6 files:
- `adversarial.rs`
- `correctness.rs`
- `edge_cases.rs`
- `integration.rs`
- `regression_tests.rs`
- `vs_linfa.rs`

**Step 2: Run full test suite**

Run: `cargo test -p ferroml-core 2>&1 | tail -10`
Expected: All ~3,160 tests pass (lib tests + integration tests combined)

**Step 3: Check that no tests were lost**

Count tests before and after. Before: ~500 integration test functions across 19 files. After: ~500 across 6 files. The number should match exactly.

Run: `cargo test -p ferroml-core --test '*' -- --list 2>&1 | grep '::.*: test$' | wc -l`

**Step 4: Commit the final state if needed**

```bash
git add -A
git commit -m "refactor: verify all integration tests pass after consolidation"
```

---

### Task 8: Make heavy optional deps non-default

This reduces compile time for `cargo test` when GPU/Polars aren't needed.

**Files:**
- Modify: `ferroml-core/Cargo.toml`

**Step 1: Change default features**

In `ferroml-core/Cargo.toml`, change the `[features]` section:

```toml
[features]
default = ["parallel", "onnx", "simd", "faer-backend"]
# Remove "datasets" from default — Polars is heavy and only needed for CSV/Parquet loading
datasets = ["dep:polars", "dep:arrow"]
parallel = []
simd = ["wide"]
sparse = ["sprs"]
onnx = []
onnx-validation = ["ort"]
faer-backend = ["faer"]
gpu = ["wgpu", "bytemuck", "pollster"]
# New: "all" feature for CI or full builds
all = ["datasets", "sparse", "gpu", "onnx-validation"]
```

Key change: `datasets` removed from `default`. Users who need `load_iris()` etc. add `features = ["datasets"]`.

**Step 2: Verify default build works without Polars**

Run: `cargo build -p ferroml-core 2>&1 | tail -5`
Expected: Compiles without Polars/Arrow

**Step 3: Verify full build still works**

Run: `cargo build -p ferroml-core --features datasets 2>&1 | tail -5`
Expected: Compiles with Polars/Arrow

**Step 4: Check which tests need `datasets` feature**

Run: `grep -rl 'load_iris\|load_wine\|load_diabetes\|load_california\|datasets::' ferroml-core/tests/ | sort`

These tests will need `#[cfg(feature = "datasets")]` gates OR we run tests with `--features datasets`.

**Step 5: Gate dataset-dependent tests**

For each test function that uses `ferroml_core::datasets::*`, add:
```rust
#[cfg(feature = "datasets")]
```
above the test function or the entire module if all tests in it use datasets.

**Step 6: Verify tests pass both ways**

Run: `cargo test -p ferroml-core -- --quiet 2>&1 | tail -5` (without datasets)
Run: `cargo test -p ferroml-core --features datasets -- --quiet 2>&1 | tail -5` (with datasets)

**Step 7: Commit**

```bash
git add ferroml-core/Cargo.toml ferroml-core/tests/
git commit -m "build: make polars/datasets a non-default feature to reduce build size"
```

---

## Summary

| Change | Binaries Before | Binaries After | Estimated Disk Savings |
|--------|----------------|---------------|----------------------|
| Consolidate tests | 19 | 6 | ~65% of test binary disk |
| Remove Polars from default | N/A | N/A | ~30s compile time + ~200 crates skipped |
| **Total** | **29 targets** | **16 targets** | **~200 GB in debug builds** |

## Risk Mitigation

- **No test logic changes** — purely mechanical moves into `mod` blocks
- **Each task is independently verifiable** — run tests after each merge
- **Rollback is trivial** — `git revert` any commit
- **Module name conflicts** — each old file becomes its own `mod`, no naming collisions
- **`use` statement scoping** — imports move inside their `mod` block, so no cross-contamination
