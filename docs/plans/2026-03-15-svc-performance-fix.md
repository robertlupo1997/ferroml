# SVC Performance Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix SVC's 17.6x performance regression by implementing libsvm-style shrinking and WSS3 (second-order working set selection), bringing SVC to within 2-3x of sklearn.

**Architecture:** Three targeted changes to the existing SMO loop in `svm.rs`: (1) add a shrinking mechanism that removes bound-variable samples from the active set, reducing the effective problem size from `n` to `~n_sv`; (2) replace the first-order `select_j` heuristic with second-order WSS3 that uses curvature information; (3) optimize the error cache update to only iterate over active samples. The kernel cache and KernelProvider abstraction stay as-is since they work fine with shrinking.

**Tech Stack:** Rust, ndarray, existing ferroml-core SVM module

---

### Task 1: Add shrinking infrastructure

Add the data structures and helper methods that track which samples are "active" vs "shrunk" (at bounds and unlikely to change).

**Files:**
- Modify: `ferroml-core/src/models/svm.rs:558-698` (the `smo` method)

**Step 1: Add active set tracking to the SMO method**

At the start of the `smo` method (after line 574), add:

```rust
// Shrinking: track which samples are "active" (may change) vs "shrunk" (at bounds)
let mut active: Vec<bool> = vec![true; n_samples];
let mut active_set: Vec<usize> = (0..n_samples).collect();
let shrink_interval = 100.min(self.max_iter); // Re-evaluate shrinking every 100 iters
```

**Step 2: Add the shrinking helper method**

Add a new method to `impl SVC` (after `select_j`, around line 740):

```rust
/// Shrink samples that are firmly at their bounds and unlikely to change.
/// A sample is shrinkable if:
/// - alpha[i] == 0 and y[i] * error[i] >= 0 (correctly classified, won't enter)
/// - alpha[i] == c[i] and y[i] * error[i] <= 0 (at upper bound, won't decrease)
fn shrink_active_set(
    active: &mut [bool],
    active_set: &mut Vec<usize>,
    alpha: &ndarray::ArrayViewMut1<f64>,
    errors: &Array1<f64>,
    y: &Array1<f64>,
    c: &Array1<f64>,
    tol: f64,
) {
    let threshold = 1e-8;
    for &i in active_set.iter() {
        if !active[i] {
            continue;
        }
        let ri = errors[i] * y[i]; // KKT residual

        // At lower bound (alpha = 0) and gradient says "stay"
        let at_lower = alpha[i] < threshold && ri >= -tol;
        // At upper bound (alpha = C) and gradient says "stay"
        let at_upper = (alpha[i] - c[i]).abs() < threshold && ri <= tol;

        if at_lower || at_upper {
            active[i] = false;
        }
    }
    // Rebuild active_set from active flags
    active_set.retain(|&i| active[i]);
}
```

**Step 3: Run tests to verify no regressions**

Run: `cargo test -p ferroml-core --lib -- svm --quiet 2>&1 | tail -5`
Expected: All existing SVM tests pass (no functional changes yet)

**Step 4: Commit**

```bash
git add ferroml-core/src/models/svm.rs
git commit -m "feat(svm): add shrinking infrastructure for SMO"
```

---

### Task 2: Integrate shrinking into the SMO loop

Wire the shrinking into the main SMO iteration loop so it periodically removes bound samples from the active set.

**Files:**
- Modify: `ferroml-core/src/models/svm.rs:558-698` (the `smo` method)

**Step 1: Apply shrinking periodically and use active_set for iteration**

Replace the SMO loop body (lines 580-694) with this updated version:

```rust
while (n_changed > 0 || examine_all) && iter < self.max_iter {
    n_changed = 0;

    // Periodically shrink the active set
    if iter > 0 && iter % shrink_interval == 0 && !examine_all {
        Self::shrink_active_set(
            &mut active, &mut active_set, &alpha, &errors, y, c, self.tol,
        );
    }

    let indices: Vec<usize> = if examine_all {
        // Full pass: unshrink all samples to re-evaluate
        for i in 0..n_samples {
            active[i] = true;
        }
        active_set = (0..n_samples).collect();
        (0..n_samples).collect()
    } else {
        // Only examine non-bound active examples
        active_set
            .iter()
            .copied()
            .filter(|&i| alpha[i] > 0.0 && alpha[i] < c[i])
            .collect()
    };

    for &i in &indices {
        let ei = errors[i];
        let ri = ei * y[i];

        // Check KKT conditions
        if (ri < -self.tol && alpha[i] < c[i]) || (ri > self.tol && alpha[i] > 0.0) {
            // Select j using heuristic (only from active samples)
            let j = self.select_j_active(i, &errors, &alpha, c, &active);

            if let Some(j) = j {
                // ... (rest of the alpha update logic stays identical)
```

The key changes are:
- `shrink_active_set` is called every `shrink_interval` iterations during non-examine-all passes
- `examine_all = true` unshrinks everything (re-checks all samples)
- Non-bound iteration filters from `active_set` instead of `0..n_samples`
- `select_j` is replaced with `select_j_active` that only considers active samples

**Step 2: Update the error cache update to only iterate active samples**

Replace the error cache update block (lines 666-681):

```rust
// Incremental error cache update — only active samples
{
    let di = (alpha[i] - alpha_i_old) * y[i];
    let dj = (alpha[j] - alpha_j_old) * y[j];
    let db = b - b_old;

    kernel.fill_row(i, &mut row_i_buf);
    kernel.fill_row(j, &mut row_j_buf);

    // Update errors for all samples (including shrunk, for correctness
    // when unshrinking). This is the same cost as before but shrinking
    // reduces the number of KKT checks and select_j searches.
    for k in 0..n_samples {
        errors[k] += di * row_i_buf[k] + dj * row_j_buf[k] + db;
    }
}
```

Note: We keep updating errors for ALL samples (not just active) so that when we unshrink, the error cache is still valid. The speedup comes from fewer KKT checks and cheaper select_j, not from the error update.

**Step 3: Run tests**

Run: `cargo test -p ferroml-core --lib -- svm --quiet 2>&1 | tail -5`
Expected: All SVM tests pass

**Step 4: Commit**

```bash
git add ferroml-core/src/models/svm.rs
git commit -m "feat(svm): integrate shrinking into SMO loop"
```

---

### Task 3: Implement WSS3 (second-order working set selection)

Replace the first-order `select_j` (max |Ei - Ej|) with second-order WSS3 that uses curvature information to pick the pair that makes the most progress per iteration.

**Files:**
- Modify: `ferroml-core/src/models/svm.rs:700-740` (the `select_j` method)

**Step 1: Add the new `select_j_active` method**

Add this method to `impl SVC`, replacing the old `select_j`:

```rust
/// WSS3: Second-order working set selection (libsvm-style).
///
/// Given i (the first index, selected by max KKT violation), find j that
/// maximizes the objective function decrease. Uses second-order information
/// (curvature via kernel diagonal) for better convergence.
///
/// Objective decrease for pair (i,j) ≈ -(Ei - Ej)^2 / (Kii + Kjj - 2*Kij)
/// We want to maximize this, which means maximizing (Ei - Ej)^2 / eta.
fn select_j_active(
    &self,
    i: usize,
    errors: &Array1<f64>,
    alpha: &ndarray::ArrayViewMut1<f64>,
    c: &Array1<f64>,
    active: &[bool],
    kernel: &mut KernelProvider,
) -> Option<usize> {
    let ei = errors[i];
    let n_samples = errors.len();
    let k_ii = kernel.get(i, i);

    let mut best_j = None;
    let mut best_gain = 0.0_f64;

    for j in 0..n_samples {
        if j == i || !active[j] {
            continue;
        }

        let ej = errors[j];
        let diff = ei - ej;

        // Skip if no error difference (no progress possible)
        if diff.abs() < 1e-12 {
            continue;
        }

        // Second-order approximation of objective decrease
        let k_jj = kernel.get(j, j);
        let k_ij = kernel.get(i, j);
        let eta = k_ii + k_jj - 2.0 * k_ij;

        // If eta > 0 (the usual case), gain = diff^2 / eta
        // If eta <= 0 (degenerate), fall back to first-order: gain = diff^2
        let gain = if eta > 1e-12 {
            diff * diff / eta
        } else {
            diff * diff
        };

        if gain > best_gain {
            best_gain = gain;
            best_j = Some(j);
        }
    }

    best_j
}
```

**Step 2: Update `smo` to pass `kernel` to `select_j_active`**

Change the call in the SMO loop from:
```rust
let j = self.select_j(i, &errors, &alpha, c);
```
to:
```rust
let j = self.select_j_active(i, &errors, &alpha, c, &active, &mut kernel_provider);
```

Wait — `kernel` is already `&mut KernelProvider` in `smo`. The signature needs to thread it through. Update the call:

```rust
let j = self.select_j_active(i, &errors, &alpha, c, &active, kernel);
```

**Step 3: Keep the old `select_j` method but mark it deprecated**

```rust
#[deprecated(note = "Use select_j_active with WSS3 instead")]
fn select_j(/* ... */) -> Option<usize> { /* ... */ }
```

Or just delete it if no other code calls it.

**Step 4: Run tests**

Run: `cargo test -p ferroml-core --lib -- svm --quiet 2>&1 | tail -5`
Expected: All SVM tests pass (WSS3 converges at least as well as first-order)

Run: `cargo test -p ferroml-core --test vs_linfa -- svm --quiet 2>&1 | tail -5`
Expected: Cross-library SVM validation tests pass

**Step 5: Commit**

```bash
git add ferroml-core/src/models/svm.rs
git commit -m "feat(svm): implement WSS3 second-order working set selection"
```

---

### Task 4: Lower the full-matrix threshold

Now that shrinking + WSS3 make the LRU cache viable, lower the threshold from 10K back to a reasonable value so larger datasets benefit from the cache.

**Files:**
- Modify: `ferroml-core/src/models/svm.rs:87` (FULL_MATRIX_THRESHOLD constant)

**Step 1: Lower threshold**

Change:
```rust
const FULL_MATRIX_THRESHOLD: usize = 10_000;
```
to:
```rust
/// Threshold below which the full kernel matrix is precomputed.
/// Above this, the LRU cache is used with shrinking to limit memory.
/// Full matrix costs O(n^2) memory but O(1) access; cache costs O(cache_size * n)
/// but O(n_features) per miss. With shrinking, cache hit rates are high because
/// the active set converges to ~n_sv (support vectors).
const FULL_MATRIX_THRESHOLD: usize = 4_000;
```

**Step 2: Run tests at various scales**

Run: `cargo test -p ferroml-core --lib -- svm --quiet 2>&1 | tail -5`
Expected: All pass

Run: `cargo test -p ferroml-core --test vs_linfa -- svm --quiet 2>&1 | tail -5`
Expected: Cross-library validation pass

**Step 3: Commit**

```bash
git add ferroml-core/src/models/svm.rs
git commit -m "perf(svm): lower full-matrix threshold to 4K (shrinking makes cache viable)"
```

---

### Task 5: Benchmark and validate

Run the cross-library benchmark to measure the actual speedup.

**Files:**
- Read: `scripts/benchmark_cross_library.py`

**Step 1: Build release bindings**

```bash
source .venv/bin/activate
maturin develop --release -m ferroml-python/Cargo.toml
```

**Step 2: Run SVC benchmark**

```bash
python3 -c "
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC as SkSVC
from ferroml.svm import SVC as FerroSVC

X, y = make_classification(n_samples=5000, n_features=20, random_state=42)

# sklearn
start = time.time()
sk = SkSVC(kernel='rbf', C=1.0, gamma='scale')
sk.fit(X, y)
sk_time = time.time() - start

# ferroml
start = time.time()
ferro = FerroSVC(kernel='rbf', c=1.0, gamma=1.0/20)
ferro.fit(X, y)
ferro_time = time.time() - start

ratio = ferro_time / sk_time
print(f'sklearn:  {sk_time*1000:.1f}ms')
print(f'ferroml:  {ferro_time*1000:.1f}ms')
print(f'ratio:    {ratio:.1f}x')
print(f'target:   <3x')
"
```

**Step 3: Verify accuracy hasn't degraded**

```bash
python3 -c "
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC as SkSVC
from sklearn.metrics import accuracy_score
from ferroml.svm import SVC as FerroSVC

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

sk = SkSVC(kernel='rbf', C=1.0, gamma='scale')
sk.fit(X_train, y_train)
sk_acc = accuracy_score(y_test, sk.predict(X_test))

ferro = FerroSVC(kernel='rbf', c=1.0, gamma=1.0/10)
ferro.fit(X_train, y_train)
ferro_acc = accuracy_score(y_test, ferro.predict(X_test))

print(f'sklearn accuracy:  {sk_acc:.3f}')
print(f'ferroml accuracy:  {ferro_acc:.3f}')
print(f'gap:               {abs(sk_acc - ferro_acc):.3f} (target: <0.05)')
"
```

**Step 4: Commit final state**

```bash
git add -A
git commit -m "perf(svm): shrinking + WSS3 — SVC now within Nx of sklearn (was 17.6x)"
```

---

## Summary

| Task | What | Expected Impact |
|------|------|----------------|
| 1 | Shrinking infrastructure | Foundation |
| 2 | Integrate shrinking into SMO | 3-5x speedup (fewer KKT checks, smaller active set) |
| 3 | WSS3 second-order selection | 1.5-2x speedup (fewer iterations to converge) |
| 4 | Lower full-matrix threshold | Memory savings for n>4K |
| 5 | Benchmark & validate | Verify <3x target |

**Combined target:** 17.6x -> 2-3x of sklearn.

**Risk:** If shrinking is too aggressive, convergence may slow (unshrinking handles this). If WSS3 kernel.get() calls are expensive with cache, the overhead may eat the convergence gains. Both are mitigated by the full-matrix path for n<4K.
