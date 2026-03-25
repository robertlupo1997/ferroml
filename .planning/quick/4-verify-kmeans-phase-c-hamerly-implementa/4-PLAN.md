---
phase: quick-4
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/benchmark_cross_library_results.json
  - docs/cross-library-benchmark.md
autonomous: true
requirements: [VERIFY-KMEANS-C]

must_haves:
  truths:
    - "All Rust KMeans tests pass (unit + correctness + vs_linfa)"
    - "All Python KMeans tests pass"
    - "Hamerly auto-selects for k<=20 and Elkan for k>20"
    - "run_hamerly handles edge cases correctly (k=1, k=2, empty clusters)"
    - "Parallel path in run_hamerly has no correctness issues"
    - "Benchmark results documented at multiple sizes (1000, 5000, 10000)"
  artifacts:
    - path: "docs/benchmark_cross_library_results.json"
      provides: "Updated benchmark results with multi-size KMeans data"
    - path: "docs/cross-library-benchmark.md"
      provides: "Human-readable benchmark report"
  key_links:
    - from: "KMeansAlgorithm::Auto"
      to: "resolve_algorithm"
      via: "Three-tier dispatch"
      pattern: "Hamerly.*Elkan.*Lloyd"
---

<objective>
Verify the KMeans Phase C (Hamerly) implementation for correctness, edge case handling, parallel safety, and performance. Run the full test suite, review the parallel path for race conditions, and benchmark at multiple dataset sizes.

Purpose: Ensure the Hamerly implementation is production-quality before considering it shipped. The implementation was done in quick-3 -- this task verifies it thoroughly.
Output: Verified test results, code review findings, and updated benchmark results in docs/.
</objective>

<execution_context>
@/home/tlupo/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlupo/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@ferroml-core/src/clustering/kmeans.rs
@.planning/quick/3-implement-kmeans-phase-c-optimizations-h/3-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Run full test suite and review Hamerly implementation for correctness</name>
  <files>ferroml-core/src/clustering/kmeans.rs</files>
  <action>
1. Run the full Rust KMeans test suite:
   ```bash
   cargo test -p ferroml-core -- clustering::kmeans 2>&1
   cargo test --test correctness -- kmeans 2>&1
   ```

2. Build Python bindings and run Python KMeans tests:
   ```bash
   source .venv/bin/activate
   maturin develop --release -m ferroml-python/Cargo.toml
   pytest ferroml-python/tests/ -k kmeans -x -v --tb=short
   ```

3. Review the `run_hamerly` method in kmeans.rs (starts around line 748) for:
   a. **Edge case: k=1** -- When k=1, there is no second-closest center. Verify that the initial assignment and bound update logic handles this correctly (lower bound should be infinity or very large).
   b. **Edge case: k=2** -- Minimal case with exactly one alternative center. Verify lower bound is set to the distance to that single other center.
   c. **Empty cluster handling** -- If a cluster loses all points mid-iteration, verify the code does not panic (division by zero in centroid computation, zero-length slices, etc.).
   d. **Parallel path correctness** -- Read the parallel Step 2 code using `par_iter_mut().zip()`. Verify:
      - No shared mutable state between parallel iterations
      - Each point's upper/lower/label update is independent
      - The center accumulation (fold+reduce) correctly aggregates across threads
      - No data races on the labels array (should be safe since each index is unique in zip)
   e. **Bound update correctness** -- Verify the conservative max_delta update:
      - `upper[i] += deltas[assigned[i]]` (correct: upper bound grows by how much assigned center moved)
      - `lower[i] -= max_delta` then clamp to 0.0 (correct: conservative -- any center could have gotten closer)
   f. **Convergence check** -- Verify it uses sum of squared center shifts < tol (matching sklearn), not per-iteration inertia.

4. Verify algorithm auto-selection by checking `resolve_algorithm`:
   - k <= 20 AND k*2 <= n => Hamerly
   - 20 < k <= 256 AND k*2 <= n => Elkan
   - Otherwise => Lloyd
   Confirm these thresholds match the documented three-tier selection in the summary.

5. Run clippy to confirm no warnings:
   ```bash
   cargo clippy -p ferroml-core -- -D warnings 2>&1 | tail -20
   ```

If any issues are found in the review, fix them and re-run tests. Document all findings.
  </action>
  <verify>
    <automated>cd /home/tlupo/ferroml && cargo test -p ferroml-core -- clustering::kmeans 2>&1 | tail -5 && cargo test --test correctness -- kmeans 2>&1 | tail -5 && source .venv/bin/activate && pytest ferroml-python/tests/ -k kmeans -x --tb=short 2>&1 | tail -5</automated>
  </verify>
  <done>
    - All Rust KMeans tests pass (unit + correctness + vs_linfa)
    - All Python KMeans tests pass
    - Code review complete: edge cases (k=1, k=2, empty clusters) verified safe
    - Parallel path verified: no shared mutable state, correct fold+reduce accumulation
    - Bound update logic verified correct (conservative max_delta approach)
    - Algorithm auto-selection thresholds verified matching documentation
    - Clippy clean with -D warnings
  </done>
</task>

<task type="auto">
  <name>Task 2: Benchmark at multiple sizes and update results documentation</name>
  <files>docs/benchmark_cross_library_results.json, docs/cross-library-benchmark.md</files>
  <action>
1. Run the cross-library benchmark at three sizes to characterize scaling behavior:
   ```bash
   source .venv/bin/activate
   python scripts/benchmark_cross_library.py --algorithms KMeans --sizes 1000 5000 10000
   ```
   This will write results to docs/benchmark_cross_library_results.json and docs/cross-library-benchmark.md automatically.

2. Examine the results:
   - At each size, record FerroML time vs sklearn time and the ratio
   - Verify Hamerly auto-selects (the default algorithm="auto" should pick Hamerly since benchmark uses k=8 which is <= 20)
   - Check that performance scales reasonably (should be roughly linear in n)

3. Verify that the benchmark results show FerroML KMeans is competitive with sklearn (target: within 3.0x per the relaxed PERF-11 target from Phase 04).

4. If the benchmark script does not already include multiple sizes in a single JSON output, ensure all three size results are captured.

5. Document final observations as comments in the summary (Task 1 findings + benchmark numbers).
  </action>
  <verify>
    <automated>cd /home/tlupo/ferroml && cat docs/benchmark_cross_library_results.json | python -c "import json,sys; data=json.load(sys.stdin); kmeans=[r for r in data.get('results',data) if 'KMeans' in str(r.get('algorithm',''))]; print(f'KMeans results: {len(kmeans)} entries'); [print(f\"  n={r.get('size','?')}: ferro={r.get('ferroml_time','?'):.4f}s, sklearn={r.get('sklearn_time','?'):.4f}s, ratio={r.get('ratio','?'):.2f}x\") for r in kmeans]" 2>&1</automated>
  </verify>
  <done>
    - Benchmark results exist for KMeans at sizes 1000, 5000, and 10000
    - docs/benchmark_cross_library_results.json updated with all three sizes
    - docs/cross-library-benchmark.md updated with human-readable report
    - FerroML KMeans performance is within 3.0x of sklearn at all sizes (or deviation documented)
    - Hamerly auto-selected for the benchmark's k=8 configuration
  </done>
</task>

</tasks>

<verification>
1. `cargo test -p ferroml-core -- clustering::kmeans` -- all unit tests pass
2. `cargo test --test correctness -- kmeans` -- correctness suite passes
3. `pytest ferroml-python/tests/ -k kmeans` -- Python tests pass
4. `cargo clippy -p ferroml-core -- -D warnings` -- no warnings
5. Benchmark results in docs/benchmark_cross_library_results.json include 3 sizes
6. Code review confirms: no race conditions, correct bounds, edge cases handled
</verification>

<success_criteria>
- All KMeans tests pass (Rust + Python, 60+ tests total)
- Code review complete with no blocking issues found (or issues fixed)
- Benchmark results documented at 1000, 5000, 10000 samples
- FerroML KMeans within 3.0x of sklearn at all benchmark sizes
- Hamerly correctly auto-selected for k<=20
</success_criteria>

<output>
After completion, create `.planning/quick/4-verify-kmeans-phase-c-hamerly-implementa/4-SUMMARY.md`
</output>
