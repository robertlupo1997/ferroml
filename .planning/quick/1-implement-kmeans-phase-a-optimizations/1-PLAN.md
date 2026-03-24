---
phase: quick-kmeans-phase-a
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - ferroml-core/src/clustering/kmeans.rs
autonomous: true
requirements: [PERF-KMEANS-A]
must_haves:
  truths:
    - "All existing KMeans tests pass (cargo test) after each optimization"
    - "KMeans benchmark shows measurable improvement over 4.25x baseline"
    - "Cluster labels and inertia remain numerically equivalent to pre-optimization output"
  artifacts:
    - path: "ferroml-core/src/clustering/kmeans.rs"
      provides: "Optimized KMeans with 5 Phase A improvements"
  key_links:
    - from: "run_elkan squared bounds"
      to: "all bound comparisons"
      via: "All bounds stored as squared distances, sqrt eliminated from inner loops"
      pattern: "squared_euclidean_distance.*without.*sqrt"
---

<objective>
Implement 5 KMeans Phase A optimizations from the design document to reduce the performance gap from 4.25x to ~2x slower than sklearn.

Purpose: These are quick-win optimizations that collectively provide 2.0-2.5x speedup through eliminating unnecessary computation (sqrt), avoiding redundant work (inertia recomputation), reducing allocation overhead, and enabling parallelism at lower thresholds.

Output: Optimized kmeans.rs with all tests passing and benchmark results.
</objective>

<execution_context>
@/home/tlupo/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlupo/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@docs/plans/2026-03-23-kmeans-performance-optimization.md
@ferroml-core/src/clustering/kmeans.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement all 5 Phase A optimizations</name>
  <files>ferroml-core/src/clustering/kmeans.rs</files>
  <action>
Apply 5 optimizations to `run_elkan` (and `run_lloyd`/`cpu_assign` where applicable) in this order, running `cargo test -p ferroml-core` after each to catch regressions immediately:

**Optimization 1 — Eliminate sqrt from Elkan inner loop (design doc #1):**
Store ALL bounds (upper, lower, s, center_dists) as SQUARED distances. This means:
- Remove every `.sqrt()` call in `run_elkan`. There are ~10 occurrences (lines 418, 442, 475, 524, 534, 577, 588, 678).
- All comparisons already work with squared distances since if `a < b` then `a^2 < b^2` for non-negative values.
- `s[j]` becomes `0.5^2 * min(center_dists_sq)` = `0.25 * min(center_dists_sq)`. Update the comparison `ub <= s[ai]` to `ub_sq <= s_sq[ai]`.
- For center-to-center half-distance comparisons: `ub <= center_dists[ai*k+j] * 0.5` becomes `ub_sq <= center_dists_sq[ai*k+j] * 0.25`.
- Step 5 bound update changes: since bounds are squared, `lower_sq[j] = (sqrt(lower_sq[j]) - delta).max(0.0)^2` but this requires sqrt. Instead use the simpler SAFE approximation: for the lower bound, `new_lower_sq >= (sqrt(old_lower_sq) - delta)^2 = old_lower_sq - 2*delta*sqrt(old_lower_sq) + delta^2`. Since we want a LOWER bound (must not overestimate), use `new_lower_sq = (sqrt(old_lower_sq) - delta).max(0.0).powi(2)`. This still needs one sqrt per bound. ALTERNATIVE: keep deltas as non-squared (euclidean distances, one sqrt per center per iteration = k sqrts total, cheap). Then: `lower[j] = (lower[j].sqrt() - deltas[j]).max(0.0).powi(2)` and `upper[i] = (upper[i].sqrt() + deltas[ai]).powi(2)`. This moves sqrt from O(n*k) inner loop to O(n*k) bound update, BUT we can do better: store deltas squared AND keep a `deltas_unsq` vector (k values, cheap). Actually the simplest correct approach: keep deltas as euclidean (non-squared) — they're only k values computed once per iteration. Then bound updates become:
  - `lower_sq[j] = (lower_sq[j].sqrt() - deltas[j]).max(0.0).powi(2)` — still O(n*k) sqrts in bound update
  This negates the benefit. SO: the correct approach is to store bounds as EUCLIDEAN (non-squared) but eliminate sqrt from the DISTANCE COMPUTATION by using `squared_euclidean_distance` and only taking sqrt once when storing into bounds. Wait — that's what the code already does.

  REVISED APPROACH: The real win is eliminating sqrt from the COMPARISON paths where bounds allow skipping. The key insight: most points are SKIPPED by bounds (70%+ skip rate). For skipped points, no sqrt is needed at all. For non-skipped points, we still compute sqrt. The optimization is: store everything as squared, compare squared values, and the bound update formula in squared space is:
  - `new_lower_sq[j] >= (sqrt(old_lower_sq[j]) - delta[j])^2` — needs sqrt, defeats purpose.

  FINAL CORRECT APPROACH per sklearn source: sklearn stores bounds as EUCLIDEAN distances (not squared) but uses `squared_euclidean_distance` (no sqrt) for the actual distance computation. The sqrt is taken ONLY when a distance is actually computed (non-skipped point), not for comparisons. This is already what our code does. The actual optimization sklearn does differently is using BLAS GEMM for batch distances.

  SKIP this optimization — the current code already only takes sqrt when computing actual distances, not for bound comparisons. The sqrt on computed distances is O(1) per call and is required for correct bound arithmetic.

**Optimization 1 (REVISED) — Raw slice pattern for centers (design doc #8):**
At the start of each iteration, extract centers as a contiguous flat slice, similar to x_data pattern:
```rust
let centers_contig;
let centers_ref = if centers.is_standard_layout() {
    &centers
} else {
    centers_contig = centers.as_standard_layout().into_owned();
    &centers_contig
};
let centers_data = centers_ref.as_slice().expect("SAFETY: standard-layout");
let center_row = |j: usize| -> &[f64] { &centers_data[j * n_features..(j + 1) * n_features] };
```
Replace all `centers.row(j).as_slice().expect(...)` with `center_row(j)` in the serial path (Step 1 center-to-center distances, Step 2 serial path, Step 4 deltas). This eliminates repeated ndarray row extraction overhead.

**Optimization 2 — Convergence on center movement instead of inertia (design doc #4):**
Replace lines 711-728 (inertia computation + convergence check) with:
```rust
let max_delta_sq: f64 = deltas.iter().map(|d| d * d).fold(0.0f64, f64::max);
if max_delta_sq < self.tol * self.tol {
    // Compute final inertia only on convergence
    let inertia: f64 = (0..n_samples)
        .map(|i| {
            let ci = labels[i] as usize;
            let xi = x_row(i);
            let c_s = center_row(ci);  // uses new_centers after swap
            crate::linalg::squared_euclidean_distance(xi, c_s)
        })
        .sum();
    return (centers, labels, inertia, iter + 1);
}
```
IMPORTANT: The `std::mem::swap(&mut centers, &mut new_centers)` must happen BEFORE this check so that `centers` has the new values. Move the swap to just after Step 4 (deltas computation). The center_row closure must be re-bound after swap to point to the new centers. Remove `prev_inertia` variable entirely.

Also apply same convergence approach to `run_lloyd`: compute deltas between old and new centers, check max_delta_sq < tol^2, compute inertia only on convergence/max_iter.

**Optimization 3 — Remove per-iteration Vec<Vec<f64>> allocations (design doc #3):**
In the parallel Step 2 path (lines 489-490), `center_data` is reallocated every iteration:
```rust
let center_data: Vec<Vec<f64>> = (0..k).map(|j| centers.row(j).to_vec()).collect();
```
Replace with the `center_row` closure from Optimization 1 which uses the flat slice — zero allocations. The closure captures `centers_data` which is already extracted at iteration start.

For the parallel initial assignment (lines 407-427), the `Vec<f64>` per-sample (`lowers`) allocation: replace the collect-then-scatter pattern with `par_chunks_mut` on the output arrays. Use indices to write directly:
```rust
use rayon::prelude::*;
let chunk_size = (n_samples / rayon::current_num_threads().max(1)).max(64);
labels.as_slice_mut().expect("contiguous")
    .par_chunks_mut(1)
    .zip(upper.par_chunks_mut(1))
    .zip(lower.par_chunks_mut(k))
    .enumerate()
    .with_min_len(chunk_size)
    .for_each(|(i, ((label_chunk, ub_chunk), lower_chunk))| {
        let xi = &x_data[i * n_features..(i + 1) * n_features];
        let mut min_dist = f64::MAX;
        let mut min_idx = 0i32;
        for j in 0..k {
            let dist = crate::linalg::squared_euclidean_distance(xi, center_row(j)).sqrt();
            lower_chunk[j] = dist;
            if dist < min_dist {
                min_dist = dist;
                min_idx = j as i32;
            }
        }
        label_chunk[0] = min_idx;
        ub_chunk[0] = min_dist;
    });
```

For parallel Step 2 (lines 493-550), similarly replace collect-then-scatter with in-place mutation via `par_chunks_mut` indexed access. Each thread writes directly to its slice of labels/upper/lower.

**Optimization 4 — Lower PARALLEL_MIN_SAMPLES from 10000 to 2000 (design doc #7):**
Change line 41: `const PARALLEL_MIN_SAMPLES: usize = 2_000;`

**Run `cargo test -p ferroml-core` after completing all changes.** If any test fails, diagnose: the most likely issue is the convergence tolerance change (center movement vs inertia). The `tol` default is 1e-4 in our code — center movement convergence with `tol^2 = 1e-8` may converge differently than inertia convergence. If tests fail, check whether switching to `max_delta_sq < self.tol` (comparing euclidean delta, not squared) matches expected behavior better.

IMPORTANT: Do NOT change the `cpu_assign` function signature or its return of inertia — it's used by Lloyd and by the initial parallel path. Only change convergence checking logic in the iteration loop.
  </action>
  <verify>
    <automated>cd /home/tlupo/ferroml && cargo test -p ferroml-core 2>&1 | tail -5</automated>
  </verify>
  <done>All ~3550 Rust tests pass. KMeans produces equivalent labels and inertia values.</done>
</task>

<task type="auto">
  <name>Task 2: Run correctness tests and benchmark</name>
  <files>ferroml-core/src/clustering/kmeans.rs</files>
  <action>
1. Run the full correctness suite to verify no regressions:
   ```bash
   cargo test --test correctness 2>&1 | tail -20
   ```

2. Run KMeans-specific tests to verify clustering quality:
   ```bash
   cargo test -p ferroml-core kmeans 2>&1 | tail -20
   ```

3. Run the Python benchmark to measure improvement:
   ```bash
   source /home/tlupo/ferroml/.venv/bin/activate
   maturin develop --release -m ferroml-python/Cargo.toml
   python scripts/benchmark_vs_sklearn.py --perf-only 2>&1 | grep -A2 -i kmeans
   ```

4. Report the KMeans result vs the 4.25x baseline.

If benchmark shows no improvement or regression, investigate whether the parallel threshold change caused overhead at the benchmark's sample size, or whether the convergence change altered iteration count. Revert individual changes if needed to isolate.
  </action>
  <verify>
    <automated>cd /home/tlupo/ferroml && cargo test --test correctness 2>&1 | tail -3</automated>
  </verify>
  <done>Correctness suite passes. Benchmark result reported showing improvement from 4.25x baseline. Target: within 3.0x of sklearn.</done>
</task>

</tasks>

<verification>
1. `cargo test -p ferroml-core` — all ~3550 tests pass
2. `cargo test --test correctness` — full correctness suite passes
3. Python benchmark shows KMeans improvement from 4.25x baseline
</verification>

<success_criteria>
- All existing tests pass with zero regressions
- KMeans benchmark shows measurable improvement (target: better than 3.5x, stretch: within 2.5x)
- Code changes are limited to kmeans.rs only
</success_criteria>

<output>
After completion, create `.planning/quick/1-implement-kmeans-phase-a-optimizations/1-SUMMARY.md`
</output>
