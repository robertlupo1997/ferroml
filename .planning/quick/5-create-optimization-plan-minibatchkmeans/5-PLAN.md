---
phase: quick-5
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - docs/plans/2026-03-25-performance-optimization-v2.md
autonomous: true
must_haves:
  truths:
    - "Plan document covers all 4 work areas with concrete strategies"
    - "Plan includes profiled baseline numbers from current benchmarks"
    - "Plan defines measurable success targets for each optimization"
    - "Plan orders work logically (baseline -> optimizations -> final benchmark)"
  artifacts:
    - path: "docs/plans/2026-03-25-performance-optimization-v2.md"
      provides: "Comprehensive performance optimization plan v2"
      min_lines: 200
  key_links: []
---

<objective>
Create a comprehensive performance optimization plan document covering 4 work areas:
MiniBatchKMeans optimization, LogisticRegression performance, HistGradientBoosting performance,
and a full benchmark refresh.

Purpose: Provide a detailed, actionable plan for the next round of performance work,
building on the KMeans Phases A-C optimizations already completed.

Output: docs/plans/2026-03-25-performance-optimization-v2.md
</objective>

<execution_context>
@/home/tlupo/.claude/get-shit-done/workflows/execute-plan.md
@/home/tlupo/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@ferroml-core/src/clustering/kmeans.rs (1915 lines — KMeans with Hamerly/Elkan/GEMM for reference patterns)
@ferroml-core/src/models/logistic.rs (2795 lines — LogReg with IRLS/L-BFGS/SAG/SAGA solvers)
@ferroml-core/src/models/hist_boosting.rs (4367 lines — HistGBT classifier/regressor)
@scripts/benchmark_cross_library.py (870 lines — benchmark harness)
@docs/benchmark_results.json (perf target results — HistGBT 2.46x, KMeans 6.84x at 5000x50)
@docs/benchmark_cross_library_results.json (cross-library — KMeans only, post-Hamerly: 2.35-10.98x FASTER)
@docs/plans/2026-03-23-kmeans-performance-optimization.md (KMeans optimization design doc — reference for format/depth)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Research and profile current performance gaps</name>
  <files>
    ferroml-core/src/clustering/kmeans.rs
    ferroml-core/src/models/logistic.rs
    ferroml-core/src/models/hist_boosting.rs
    scripts/benchmark_cross_library.py
    docs/benchmark_results.json
    docs/benchmark_cross_library_results.json
  </files>
  <action>
Research the codebase to gather data for the plan document. For each optimization area:

**1. MiniBatchKMeans:**
- Confirm MiniBatchKMeans does NOT exist yet in ferroml (no file in clustering/)
- Read kmeans.rs to catalog which techniques (Hamerly, Elkan, GEMM batch distances, sqrt-free) could transfer to a MiniBatchKMeans implementation
- Note sklearn's MiniBatchKMeans API surface (partial_fit, n_init, batch_size, reassignment_ratio)

**2. LogisticRegression (2.1x slower):**
- Read logistic.rs focusing on the L-BFGS solver path (uses argmin crate's LBFGS with MoreThuenteLineSearch)
- Identify the solver auto-selection logic (IRLS for d<50, L-BFGS for d>=50, SAG/SAGA for large n)
- Note that the benchmark uses default constructor `LogisticRegression()` which likely uses Auto solver
- Check what sklearn defaults to (lbfgs solver, C=1.0 regularization) vs ferroml defaults (no regularization, IRLS)
- Profile hot path: the FittedLogisticData computation (covariance matrix, residuals, diagnostics) runs EVERY fit — this diagnostic overhead is a key differentiator but also a performance cost
- Check if the benchmark is comparing apples-to-apples (sklearn max_iter=1000 vs ferroml max_iter=100)

**3. HistGradientBoosting (2.6x slower):**
- Read hist_boosting.rs focusing on BinMapper, histogram building, and split finding
- Check for the bounds-check decision from Phase 04 (bounds checks RETAINED due to NaN handling)
- Identify the histogram accumulation loop and split evaluation inner loop
- Note parallelism usage (rayon for tree building)
- Compare with sklearn's approach: OpenMP parallelism, SIMD histogram accumulation, histogram subtraction trick

**4. Benchmark infrastructure:**
- Review benchmark_cross_library.py to understand current coverage (which algorithms, what sizes)
- Note it currently only has KMeans results in the cross-library JSON
- Identify what algorithms SHOULD be benchmarked but aren't in the cross-library results
- Check the perf target results for current ratios

Record findings as notes (mental model) to inform Task 2. Do NOT write any files yet.
  </action>
  <verify>
    <automated>echo "Research task — no file output. Verify by checking Task 2 produces comprehensive plan."</automated>
  </verify>
  <done>All 4 optimization areas researched with specific bottleneck identification, current numbers gathered from benchmark JSONs, and sklearn comparison points identified.</done>
</task>

<task type="auto">
  <name>Task 2: Write the comprehensive optimization plan document</name>
  <files>docs/plans/2026-03-25-performance-optimization-v2.md</files>
  <action>
Write a detailed plan document at docs/plans/2026-03-25-performance-optimization-v2.md following the format established by docs/plans/2026-03-23-kmeans-performance-optimization.md.

Structure the document with these sections:

**Header:**
- Title: "Performance Optimization v2 — Design Document"
- Date, Status (ready for implementation)
- Summary table of all 4 work areas with current perf, target perf, estimated effort

**Section 1: Full Benchmark Refresh (do FIRST as baseline)**
- Expand benchmark_cross_library.py to cover ALL algorithms currently in the perf targets (PCA, TruncatedSVD, LDA, FactorAnalysis, LinearSVC, OLS, Ridge, KMeans, HistGBT, SVC) plus LogisticRegression
- Run at multiple sizes: 1000, 5000, 10000, 50000
- Include multiple feature counts: 10, 20, 50
- Update both JSON and markdown output files
- Success criteria: comprehensive baseline covering all models at all sizes
- Estimated effort: Small (script changes + run time)

**Section 2: MiniBatchKMeans Implementation**
- Note this is a NEW model, not an optimization of existing code
- Design based on sklearn's MiniBatchKMeans API: batch_size=1024, max_iter=100, n_init=3, reassignment_ratio=0.01
- Key implementation details:
  - Reuse KMeans::kmeans_plus_plus_init from existing kmeans.rs
  - Mini-batch update rule: for each batch, assign points to nearest center, then update centers with learning rate decay (1/(count+1))
  - Reassignment of low-count centers (if count < reassignment_ratio * batch_size)
  - Support IncrementalModel trait for partial_fit
  - Apply GEMM batch distances from kmeans.rs (batch_squared_distances function already exists)
  - Apply Hamerly-style bounds if applicable (may not help for mini-batch since centers change every batch)
- Required files: clustering/kmeans.rs (add MiniBatchKMeans struct), Python bindings, __init__.py re-export
- Success criteria: Within 2x of sklearn MiniBatchKMeans on 50K+ samples, correct cluster assignments
- Estimated effort: Medium (new model but reuses existing infrastructure)

**Section 3: LogisticRegression Performance**
- Current state: 2.1x slower (per memory/known metrics — need to verify with cross-library benchmark)
- Root cause analysis:
  1. Diagnostic overhead: FittedLogisticData computes covariance matrix (d^3 Cholesky), residuals, deviance — runs every fit even when user only wants predict
  2. Solver mismatch: sklearn defaults to lbfgs with C=1.0 (L2 reg); ferroml Auto may select IRLS for small d, which is O(d^3) per iteration
  3. Benchmark fairness: sklearn uses max_iter=1000 vs ferroml max_iter=100 — different convergence behavior
- Proposed optimizations:
  1. Lazy diagnostics: compute FittedLogisticData only when diagnostic methods are called, not during fit()
  2. Ensure Auto solver matches sklearn's choice for benchmark sizes (L-BFGS for the standard benchmark config)
  3. L-BFGS tuning: check argmin's LBFGS defaults (memory size, line search params) vs scipy's L-BFGS-B
  4. Add C parameter (inverse regularization) to match sklearn API
- Success criteria: Within 1.5x of sklearn for default configurations
- Estimated effort: Medium (lazy diagnostics is the main win, L-BFGS tuning is incremental)

**Section 4: HistGradientBoosting Performance**
- Current state: 2.46x slower (PERF-09: 230ms vs 94ms on 10000x20, 50 iters)
- Phase 04 context: bounds checks retained due to NaN handling producing out-of-range bin indices
- Root cause analysis:
  1. Histogram accumulation: per-feature serial loop vs sklearn's SIMD-optimized (SSE/AVX intrinsics)
  2. Split evaluation: brute-force scan vs sklearn's optimized scan with early stopping
  3. Missing histogram subtraction trick: sklearn computes child histogram as parent - sibling (O(bins) vs O(samples))
  4. Tree building parallelism: compare rayon task granularity vs sklearn's OpenMP
- Proposed optimizations:
  1. Histogram subtraction trick: For binary splits, compute only the smaller child's histogram, derive the other by subtraction. This halves histogram computation for balanced splits.
  2. Vectorized histogram accumulation: Use explicit SIMD or ensure auto-vectorization with proper data layout (SoA vs AoS for gradient/hessian arrays)
  3. Split evaluation optimization: pre-sum gradients/hessians, use running sums instead of re-scanning
  4. Feature-parallel tree building: parallelize across features during split finding (each feature's histogram is independent)
- Success criteria: Within 2.0x of sklearn (from current 2.46x)
- Estimated effort: Large (histogram subtraction is the biggest win, SIMD is complex)

**Section 5: Work Order and Dependencies**
1. Benchmark refresh (baseline) — no dependencies, do first
2. MiniBatchKMeans — independent of other optimizations
3. LogisticRegression — independent, lazy diagnostics is the key change
4. HistGBT — independent but largest effort, do last
5. Final benchmark refresh — after all optimizations, update all docs

**Section 6: Risk Assessment**
- LogReg: lazy diagnostics may break tests that call diagnostic methods immediately after fit
- HistGBT: histogram subtraction with NaN handling adds complexity (missing_bin edge cases)
- MiniBatchKMeans: new model means new tests, bindings, serialization support
- Benchmark: larger sizes (50K) may take significant wall-clock time

Include estimated complexity for each area using a simple scale: Small / Medium / Large / XL.
  </action>
  <verify>
    <automated>test -f docs/plans/2026-03-25-performance-optimization-v2.md && wc -l docs/plans/2026-03-25-performance-optimization-v2.md | awk '{if ($1 >= 200) print "PASS: " $1 " lines"; else print "FAIL: only " $1 " lines"}'</automated>
  </verify>
  <done>Plan document exists at docs/plans/2026-03-25-performance-optimization-v2.md with 200+ lines covering all 4 work areas, each with current numbers, root cause analysis, proposed optimizations, success criteria, and estimated effort. Work is ordered logically with dependencies noted.</done>
</task>

</tasks>

<verification>
- Plan document exists and is 200+ lines
- All 4 work areas covered: MiniBatchKMeans, LogReg, HistGBT, benchmark refresh
- Each area has: current perf numbers, root cause, proposed optimizations, success criteria, effort estimate
- Work order is specified with dependencies
</verification>

<success_criteria>
- docs/plans/2026-03-25-performance-optimization-v2.md exists with comprehensive content
- Document follows format of existing plan docs (see 2026-03-23-kmeans-performance-optimization.md)
- Each optimization area has actionable, specific strategies (not vague "make it faster")
- Success targets are measurable (e.g., "within 2.0x of sklearn")
</success_criteria>

<output>
After completion, create `.planning/quick/5-create-optimization-plan-minibatchkmeans/5-SUMMARY.md`
</output>
