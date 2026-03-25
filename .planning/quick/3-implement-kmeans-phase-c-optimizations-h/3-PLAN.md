---
phase: quick-3
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - ferroml-core/src/clustering/kmeans.rs
  - ferroml-python/src/clustering.rs
autonomous: true
requirements: [PERF-KMEANS-C]

must_haves:
  truths:
    - "KMeans with k<=20 auto-selects Hamerly algorithm"
    - "KMeans with k>20 auto-selects Elkan algorithm"
    - "Hamerly uses O(n) memory for bounds instead of O(n*k)"
    - "All existing KMeans tests pass with identical results"
    - "KMeans benchmark within 1.5x of sklearn at 5000 samples"
  artifacts:
    - path: "ferroml-core/src/clustering/kmeans.rs"
      provides: "Hamerly algorithm variant and updated auto-selection"
      contains: "fn run_hamerly"
  key_links:
    - from: "KMeansAlgorithm::Hamerly"
      to: "run_hamerly"
      via: "run_kmeans dispatch"
      pattern: "Hamerly.*run_hamerly"
---

<objective>
Implement KMeans Phase C optimizations: Hamerly's algorithm with single lower bound per point (O(n) vs O(n*k) memory) and auto-select Hamerly for k<=20, Elkan for k>20.

Purpose: Hamerly's bounds (80KB for n=5000) fit in L1 cache vs Elkan's (400KB for k=10), giving better cache performance at small k. This is the final optimization phase targeting within 1.5x of sklearn.
Output: Updated kmeans.rs with Hamerly variant, updated Python bindings exposing algorithm parameter.
</objective>

<execution_context>
@/home/tlupo/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlupo/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@ferroml-core/src/clustering/kmeans.rs
@docs/plans/2026-03-23-kmeans-performance-optimization.md

<interfaces>
<!-- Key types and contracts the executor needs -->

From ferroml-core/src/clustering/kmeans.rs:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KMeansAlgorithm {
    Lloyd,
    Elkan,
    Auto,
}

// Display impl: Lloyd => "lloyd", Elkan => "elkan", Auto => "auto"

// resolve_algorithm: Auto => Elkan when k<=256 && k*2<=n, else Lloyd
// run_kmeans dispatches: Elkan => run_elkan, Lloyd|Auto => run_lloyd

// run_elkan signature:
fn run_elkan(&self, x: &Array2<f64>, initial_centers: Array2<f64>)
    -> (Array2<f64>, Array1<i32>, f64, usize)
// Returns: (centers, labels, inertia, n_iter)

// Existing helper functions:
fn compute_row_norms(x: &Array2<f64>) -> Array1<f64>;
fn batch_squared_distances(x: &Array2<f64>, centers: &Array2<f64>, x_norms: &Array1<f64>) -> Array2<f64>;
```

From ferroml-python/src/clustering.rs:
```rust
#[pyclass(name = "KMeans", module = "ferroml.clustering")]
pub struct PyKMeans { inner: KMeans }

#[pyo3(signature = (n_clusters=8, max_iter=300, tol=1e-4, random_state=None, n_init=10))]
fn new(...) -> Self
// NOTE: Does not currently expose algorithm parameter
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement Hamerly's algorithm and update auto-selection in kmeans.rs</name>
  <files>ferroml-core/src/clustering/kmeans.rs</files>
  <action>
1. Add `Hamerly` variant to `KMeansAlgorithm` enum:
   - Add `Hamerly` variant with doc comment: "Hamerly's algorithm with single lower bound per point (O(n) memory)"
   - Update `Display` impl to include `KMeansAlgorithm::Hamerly => write!(f, "hamerly")`
   - Update doc comment on `Auto` to mention Hamerly: "Automatically selects Hamerly for k<=20, Elkan for k<=256, otherwise Lloyd"

2. Update `resolve_algorithm` to use three-tier selection:
   ```rust
   KMeansAlgorithm::Auto => {
       if self.n_clusters <= 20 && self.n_clusters * 2 <= n_samples {
           KMeansAlgorithm::Hamerly
       } else if self.n_clusters <= 256 && self.n_clusters * 2 <= n_samples {
           KMeansAlgorithm::Elkan
       } else {
           KMeansAlgorithm::Lloyd
       }
   }
   ```
   Also add `KMeansAlgorithm::Hamerly => KMeansAlgorithm::Hamerly` to the explicit match.

3. Update `run_kmeans` dispatch to route `Hamerly` to `self.run_hamerly(x, initial_centers)`.

4. Implement `run_hamerly` method. Model it closely on `run_elkan` but with these key differences:
   - **Memory:** `upper: Vec<f64>` of size n (same as Elkan) but `lower: Vec<f64>` of size n (ONE value per point, not n*k). The single lower bound tracks distance to the second-closest center.
   - **Initial assignment:** Use `batch_squared_distances` (same as Elkan). For each point, find closest and second-closest center. Set `upper[i] = sqrt(min_dist)`, `lower[i] = sqrt(second_min_dist)`.
   - **Per-iteration Step 1:** Compute `s[j] = 0.5 * min_{j'!=j} d(c_j, c_j')` (same as Elkan).
   - **Per-iteration Step 2 (the key difference):** For each point i with assigned center a_i:
     - Compute `m = max(s[a_i], lower[i])`. If `upper[i] <= m`, skip this point entirely.
     - Otherwise, tighten upper bound: recompute `upper[i] = d(x_i, c_{a_i})` exactly.
     - If `upper[i] <= m` after tightening, still skip.
     - Otherwise, recompute ALL k distances for this point (no per-center lower bounds to consult). Find new closest center. If assignment changes, update label. Set `upper[i] = d(x_i, new_closest)`, `lower[i] = d(x_i, second_closest)`.
   - **Per-iteration Step 3:** Compute new centers (same accumulation logic as Elkan, both parallel and sequential paths).
   - **Per-iteration Step 4:** Compute deltas (center movement distances, Euclidean not squared).
   - **Per-iteration Step 5 (bound update):**
     - `r = index of max(deltas)` (center that moved most)
     - `r2 = index of second-max(deltas)` (second most moved)
     - For each point i with assigned center a_i:
       - `upper[i] += deltas[a_i]`
       - If `a_i == r`: `lower[i] -= deltas[r2]` (but max with 0.0... actually per Hamerly: `lower[i] = (lower[i] - deltas[r]).max(0.0)` when r != a_i, but if r == a_i use deltas[r2])

       CORRECTION — use the simpler standard Hamerly bound update:
       - `upper[i] += deltas[a_i]`
       - `lower[i] -= max_delta` where `max_delta = deltas.iter().cloned().fold(0.0, f64::max)`
       - Clamp: `lower[i] = lower[i].max(0.0)`

       This is simpler and correct. The max_delta approach is conservative but avoids tracking per-center relationships.
   - **Convergence:** Same as Elkan — `center_shift_total = sum(delta^2) < tol`.
   - **Final inertia:** Use `batch_squared_distances` after convergence (same as Elkan).

   Include both parallel and sequential code paths gated on `PARALLEL_MIN_SAMPLES` (same pattern as `run_elkan`). The parallel path for Step 2 should use `par_chunks_mut` or indexed parallel iteration, writing directly to `labels`, `upper`, `lower` arrays.

   For the parallel Step 2, since each point is independent and we only have scalar upper/lower per point (not k-sized vectors), use a simpler parallel pattern than Elkan:
   ```rust
   // Parallel: zip upper, lower, labels, and process each point
   upper.par_iter_mut()
       .zip(lower.par_iter_mut())
       .zip(labels.as_slice_mut().unwrap().par_iter_mut())
       .enumerate()
       .for_each(|(i, ((ub, lb), label))| {
           // ... Hamerly step 2 logic for point i
       });
   ```

5. Update the module doc comment at the top to mention Hamerly's algorithm alongside Elkan.

6. Run `cargo fmt --all` after all changes.
  </action>
  <verify>
    <automated>cd /home/tlupo/ferroml && cargo test -p ferroml-core -- clustering::kmeans 2>&1 | tail -20 && cargo test --test correctness -- kmeans 2>&1 | tail -20</automated>
  </verify>
  <done>
    - KMeansAlgorithm::Hamerly variant exists and is routed in run_kmeans
    - run_hamerly implements single-lower-bound algorithm with both parallel and sequential paths
    - Auto selection: Hamerly for k<=20, Elkan for k>20 (up to 256), Lloyd beyond
    - All existing KMeans tests pass with same results (labels, inertia within f64 epsilon)
    - cargo fmt clean, no clippy warnings
  </done>
</task>

<task type="auto">
  <name>Task 2: Expose algorithm parameter in Python bindings and run benchmark</name>
  <files>ferroml-python/src/clustering.rs</files>
  <action>
1. Update `PyKMeans::new` to accept an `algorithm` parameter:
   - Add `algorithm: &str` parameter with default `"auto"` to the pyo3 signature
   - Parse string to `KMeansAlgorithm`: "auto" => Auto, "lloyd" => Lloyd, "elkan" => Elkan, "hamerly" => Hamerly
   - Return `PyErr` (ValueError) for unknown algorithm strings
   - Call `inner = inner.algorithm(parsed_algorithm)` (the builder method `with_algorithm` or `algorithm` on KMeans — check the builder API; if it does not exist, it needs to be added in Task 1's file as `pub fn algorithm(mut self, algo: KMeansAlgorithm) -> Self { self.algorithm = algo; self }`)
   - Update the docstring to document the algorithm parameter: `algorithm : str, optional (default="auto") — Algorithm variant: "auto", "lloyd", "elkan", "hamerly". Auto selects Hamerly for k<=20, Elkan for k<=256, Lloyd otherwise.`

2. Rebuild Python bindings:
   ```bash
   source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml
   ```

3. Run the cross-library benchmark:
   ```bash
   source .venv/bin/activate && python scripts/benchmark_cross_library.py --algorithms KMeans --sizes 5000
   ```

4. Run Python KMeans tests to ensure no regressions:
   ```bash
   source .venv/bin/activate && pytest ferroml-python/tests/ -k kmeans -x --tb=short
   ```

5. Run `cargo fmt --all` if any Rust changes were made.
  </action>
  <verify>
    <automated>cd /home/tlupo/ferroml && source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml 2>&1 | tail -5 && pytest ferroml-python/tests/ -k kmeans -x --tb=short 2>&1 | tail -10</automated>
  </verify>
  <done>
    - Python KMeans accepts algorithm="auto"|"lloyd"|"elkan"|"hamerly"
    - All Python KMeans tests pass
    - Benchmark results recorded showing KMeans performance vs sklearn at n=5000
    - Target: within 1.5x of sklearn (or documented explanation if not met)
  </done>
</task>

</tasks>

<verification>
1. `cargo test -p ferroml-core -- clustering::kmeans` — all unit tests pass
2. `cargo test --test correctness -- kmeans` — correctness suite passes
3. `pytest ferroml-python/tests/ -k kmeans` — Python tests pass
4. Benchmark: `python scripts/benchmark_cross_library.py --algorithms KMeans --sizes 5000` shows improvement
5. `cargo clippy -p ferroml-core -- -D warnings` — no warnings
</verification>

<success_criteria>
- Hamerly's algorithm implemented with O(n) lower bounds (single value per point)
- Auto-selection routes k<=20 to Hamerly, k<=256 to Elkan, k>256 to Lloyd
- All ~50+ existing KMeans tests pass unchanged
- KMeans benchmark at 5000x50 k=10 is within 1.5x of sklearn
- Python bindings expose algorithm parameter
</success_criteria>

<output>
After completion, create `.planning/quick/3-implement-kmeans-phase-c-optimizations-h/3-SUMMARY.md`
</output>
