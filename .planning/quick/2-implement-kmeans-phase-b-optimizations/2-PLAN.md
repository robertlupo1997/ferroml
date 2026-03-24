---
phase: quick-2
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - ferroml-core/src/clustering/kmeans.rs
autonomous: true
requirements: [PERF-KMEANS-B]
must_haves:
  truths:
    - "KMeans uses GEMM-based batch distance computation for initial assignment and Lloyd's assign step"
    - "x_norms are precomputed once per fit, c_norms recomputed per iteration"
    - "All existing KMeans tests pass with identical results (labels, inertia)"
  artifacts:
    - path: "ferroml-core/src/clustering/kmeans.rs"
      provides: "batch_squared_distances fn, norm caching in run_elkan and run_lloyd"
      contains: "fn batch_squared_distances"
  key_links:
    - from: "batch_squared_distances"
      to: "ndarray .dot()"
      via: "X @ C^T matmul"
      pattern: "\\.dot\\("
---

<objective>
Implement KMeans Phase B optimizations: norm caching (#12) and BLAS GEMM batch distance computation (#2).

Purpose: These two optimizations work together to replace per-point distance loops with a single matrix multiply decomposition: ||x_i - c_j||^2 = ||x_i||^2 + ||c_j||^2 - 2 * (X @ C^T). This is the highest-impact remaining optimization, expected to yield 1.5-2.5x additional speedup.

Output: Modified kmeans.rs with batch distance computation used in initial assignments and Lloyd's cpu_assign. Elkan's Step 2 inner loop remains per-point (bounds skip most computations anyway).
</objective>

<execution_context>
@/home/tlupo/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlupo/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@ferroml-core/src/clustering/kmeans.rs
@docs/plans/2026-03-23-kmeans-performance-optimization.md

Key existing patterns:
- `crate::linalg::squared_euclidean_distance(a, b)` computes ||a-b||^2 (no sqrt)
- `run_elkan` and `run_lloyd` both use contiguous slice extraction (x_data, centers_data)
- `cpu_assign` is a static method used by Lloyd for assignment+inertia
- Elkan initial assignment (lines 410-458) and Step 2 (lines 500-607) use per-point loops
- All distance comparisons in Elkan use Euclidean (with .sqrt()), not squared
- ndarray Array2 `.dot()` calls BLAS DGEMM when available
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add batch_squared_distances helper and norm caching</name>
  <files>ferroml-core/src/clustering/kmeans.rs</files>
  <action>
1. Add a module-level helper function `batch_squared_distances`:

```rust
/// Compute all pairwise squared distances between rows of X and rows of C
/// using the GEMM decomposition: ||x_i - c_j||^2 = x_norms[i] + c_norms[j] - 2*(X@C^T)[i,j]
/// Returns Array2 of shape (n_samples, k) with squared distances clamped to >= 0.
fn batch_squared_distances(
    x: &Array2<f64>,
    centers: &Array2<f64>,
    x_norms: &Array1<f64>,
) -> Array2<f64> {
    let k = centers.nrows();
    // c_norms[j] = sum of squares of center j
    let c_norms: Array1<f64> = centers.rows().into_iter()
        .map(|row| row.dot(&row))
        .collect();
    // XC^T via BLAS GEMM: (n, d) @ (d, k) = (n, k)
    let xct = x.dot(&centers.t());
    let n = x.nrows();
    let mut dists = Array2::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let d = x_norms[i] + c_norms[j] - 2.0 * xct[[i, j]];
            dists[[i, j]] = d.max(0.0); // clamp for numerical stability
        }
    }
    dists
}
```

2. Add a helper to compute row norms:

```rust
/// Compute squared L2 norms for each row: x_norms[i] = ||x_i||^2
fn compute_row_norms(x: &Array2<f64>) -> Array1<f64> {
    x.rows().into_iter().map(|row| row.dot(&row)).collect()
}
```

3. Modify `cpu_assign` to use batch distances instead of per-point loops:
   - Compute x_norms via `compute_row_norms(x)` (note: ideally caller caches this, but cpu_assign is a static method called only for final inertia, so computing here is fine)
   - Call `batch_squared_distances(x, centers, &x_norms)` to get (n, k) squared distance matrix
   - Find argmin and min per row to produce labels and inertia (squared distances, no sqrt needed since inertia IS sum of squared distances)
   - Remove the old per-point loop and the `center_data: Vec<Vec<f64>>` allocation
   - Keep both parallel and serial paths but use the batch distance matrix for both

4. Modify `run_lloyd` to precompute x_norms once before the iteration loop:
   - Add `let x_norms = compute_row_norms(x);` before the loop
   - In the assignment step, replace `Self::cpu_assign(x, &centers, ...)` with batch distance matrix approach
   - Actually, since cpu_assign now uses batch_squared_distances internally, this will happen automatically. BUT for optimal perf, pass x_norms to avoid recomputing. Either:
     (a) Add an x_norms parameter to cpu_assign, or
     (b) Create a separate `batch_assign` method that takes x_norms
   - Choose option (a): add `x_norms: Option<&Array1<f64>>` parameter to cpu_assign. When None, compute internally. Update all call sites.

5. Modify `run_elkan` initial assignment (lines 410-458) to use batch distances:
   - Precompute `let x_norms = compute_row_norms(x_ref);` once before initial assignment
   - Replace the per-point initial assignment loop with:
     ```
     let init_dists = batch_squared_distances(x_ref, &centers, &x_norms);
     ```
   - Then derive labels, upper bounds, and lower bounds from init_dists:
     - For each row i: find min squared dist -> label. Take .sqrt() of each dist for bounds (Elkan uses Euclidean bounds, not squared).
     - lower[i*k+j] = init_dists[[i,j]].sqrt()
     - upper[i] = init_dists[[i, label]].sqrt()
   - Remove both the parallel and serial per-point initial assignment blocks, replace with a single block that works from the distance matrix.

6. Do NOT modify Elkan Step 2 (the inner bound-checking loop). The bounds skip most distance computations there, so batch GEMM would compute many unnecessary distances. The per-point approach is correct for Step 2.

7. For Elkan's final inertia computation (lines 734-756), use batch distances with x_norms (already computed).

Run `cargo test -p ferroml-core -- kmeans` after completing changes to verify correctness.
  </action>
  <verify>
    <automated>cd /home/tlupo/ferroml && cargo test -p ferroml-core -- kmeans 2>&1 | tail -5</automated>
  </verify>
  <done>
    - batch_squared_distances and compute_row_norms functions exist
    - x_norms precomputed once per fit call in both run_elkan and run_lloyd
    - Initial assignment in run_elkan uses batch GEMM distances
    - cpu_assign uses batch GEMM distances
    - All existing KMeans tests pass with no regressions
  </done>
</task>

<task type="auto">
  <name>Task 2: Full test suite and benchmark validation</name>
  <files>ferroml-core/src/clustering/kmeans.rs</files>
  <action>
1. Run the full correctness test suite (not just kmeans unit tests):
   ```
   cargo test --test correctness -- kmeans
   ```
   This runs cross-validation tests that compare FerroML KMeans output against expected fixtures.

2. Run the edge case tests:
   ```
   cargo test --test edge_cases -- kmeans
   ```

3. If any test fails, diagnose and fix. Common issues with GEMM distance decomposition:
   - Tiny negative values from floating point (should be clamped to 0.0 already)
   - Sqrt of clamped values producing slightly different bounds than direct computation
   - Ensure inertia is computed as sum of SQUARED distances (no sqrt), which is what sklearn reports

4. Once all Rust tests pass, rebuild Python bindings and run the cross-library benchmark:
   ```
   source .venv/bin/activate
   maturin develop --release -m ferroml-python/Cargo.toml
   python scripts/benchmark_cross_library.py --models KMeans --sizes 5000
   ```

5. Record the benchmark result. Expected: significant improvement from Phase A baseline (was ~2.1x after Phase A, targeting within 1.5x of sklearn).

6. If benchmark shows regression or no improvement, check:
   - That ndarray .dot() is actually calling BLAS (check Cargo.toml for blas feature flags)
   - That the batch path is being used (not accidentally falling through to old per-point code)
  </action>
  <verify>
    <automated>cd /home/tlupo/ferroml && cargo test -p ferroml-core -- kmeans 2>&1 | tail -3 && cargo test --test correctness -- kmeans 2>&1 | tail -3</automated>
  </verify>
  <done>
    - All kmeans unit tests pass
    - All kmeans correctness tests pass
    - All kmeans edge case tests pass
    - Python benchmark executed and result recorded
    - Performance improvement measured vs Phase A baseline
  </done>
</task>

</tasks>

<verification>
- `cargo test -p ferroml-core -- kmeans` -- all unit tests pass
- `cargo test --test correctness -- kmeans` -- all correctness tests pass
- `cargo test --test edge_cases -- kmeans` -- all edge case tests pass
- Python benchmark shows KMeans performance improvement
</verification>

<success_criteria>
1. batch_squared_distances function implements ||x||^2 + ||c||^2 - 2*X@C^T decomposition using ndarray .dot()
2. x_norms computed once per fit, c_norms per iteration
3. Initial assignment in both Elkan and Lloyd uses batch GEMM distances
4. All existing tests pass with no regressions (labels and inertia match)
5. Benchmark shows measurable speedup over Phase A baseline
</success_criteria>

<output>
After completion, create `.planning/quick/2-implement-kmeans-phase-b-optimizations/2-SUMMARY.md`
</output>
