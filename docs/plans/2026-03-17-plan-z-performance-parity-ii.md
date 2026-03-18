# Plan Z: Performance Parity II — Close All sklearn Gaps

**Goal**: Make every model competitive with sklearn (<2x). Currently 6 models are slower.
**Approach**: faer SVD (the nuclear option), algorithmic fixes, eliminate wasteful copies.

## Current State (vs sklearn, lower is better)

| Model | Ratio | Root Cause | Fix |
|-------|-------|------------|-----|
| PCA (n=10K, d=50) | 13.8x | nalgebra Jacobi SVD | faer thin_svd |
| LinearSVC (n=5K, d=20) | 9.6x | No shrinking, no f_i cache | LIBLINEAR-style DCD |
| OLS (n=50K, d=50) | 3.3x | MGS QR + scalar triangular solve | Cholesky normal eqs |
| Ridge (n=50K, d=50) | 3.8x | Hand-rolled Cholesky (not faer) | Use faer Cholesky |
| KMeans (n=10K, d=10) | 3.2x | sqrt in hot path, 3× Vec copies | Squared distances, zero-copy |
| SVC RBF (n=3K) | 6.8x | Boundary between full/cache | Lower FULL_MATRIX_THRESHOLD |

## The Nuclear Option: faer SVD

faer 0.20 is **already a default dependency** but only used for QR and Cholesky in `linalg.rs`.
All SVD calls go through **nalgebra's Jacobi SVD** — 10-13x slower than faer at typical sizes.

faer benchmarks (11th Gen i5):
| n | faer SVD | nalgebra SVD | Speedup |
|---|----------|--------------|---------|
| 256 | 7.8ms | 47.4ms | **6.1x** |
| 512 | 45.3ms | 472.1ms | **10.4x** |
| 1024 | 298ms | 3.95s | **13.3x** |

Replacing nalgebra SVD → faer SVD closes the PCA gap entirely and benefits LDA, FactorAnalysis, TruncatedSVD.

---

## Phase 1: faer SVD + Unified Linear Algebra (Critical — 3 days)

**Expected impact**: PCA 13.8x→~1.5x, OLS 3.3x→~1.5x, Ridge 3.8x→~1.5x

### 1a. Add `svd_faer` to `linalg.rs`

Add a faer-backed SVD wrapper following the existing `qr_decomposition_faer` pattern:

```rust
/// Thin SVD via faer: returns (U, S, Vt) where U is m×k, S is k, Vt is k×n.
#[cfg(feature = "faer-backend")]
pub fn thin_svd_faer(a: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let mat = faer::Mat::from_fn(m, n, |i, j| a[[i, j]]);
    let svd = mat.thin_svd();
    // Convert U, S, V back to ndarray
    ...
}
```

**Files**: `ferroml-core/src/linalg.rs`

### 1b. Wire PCA to use faer SVD

Replace all `nalgebra::DMatrix::svd()` calls in PCA with `crate::linalg::thin_svd()`:

**Files**: `ferroml-core/src/decomposition/pca.rs` (lines 426-429, 494-497, 956-961, 1007-1012)

Also replace in:
- `ferroml-core/src/decomposition/truncated_svd.rs` (lines 346, 372)
- `ferroml-core/src/decomposition/lda.rs` (lines 462, 544, 694)
- `ferroml-core/src/decomposition/factor_analysis.rs` (lines 599, 1191)

### 1c. OLS: Switch to Cholesky Normal Equations for n >> d

When n_samples >> n_features (which is the common case), solving via Cholesky on X'X is faster than QR:
- QR: O(n·d²) — wastes work on tall matrices
- Cholesky of X'X: O(n·d² + d³) — but the d³ term is tiny for d=50

```rust
// When n_samples > 2 * n_features, use normal equations
if n > 2 * p {
    let xtx = x_design.t().dot(&x_design);  // O(n·d²)
    let xty = x_design.t().dot(y);           // O(n·d)
    // Add regularization to diagonal for numerical stability
    let beta = cholesky_solve(&xtx, &xty)?;  // O(d³)
} else {
    // Underdetermined: use QR (current code)
}
```

Use `crate::linalg::cholesky()` (which already uses faer) instead of the hand-rolled version.

**Files**: `ferroml-core/src/models/linear.rs` (fit method, lines 384-502)

### 1d. Ridge: Use faer-backed Cholesky

The Ridge `solve_symmetric_positive_definite` (line 2255 of regularized.rs) has its own hand-rolled Cholesky with scalar loops. Replace with `crate::linalg::cholesky()` + triangular solve.

**Files**: `ferroml-core/src/models/regularized.rs` (lines 182-270, 2254-2302)

### 1e. Add faer-backed triangular solve to `linalg.rs`

Current `solve_upper_triangular` is scalar loops. Add:
```rust
#[cfg(feature = "faer-backend")]
pub fn solve_triangular_faer(l: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    // Use faer's optimized triangular solve
}
```

### Phase 1 Validation
- `cargo test -p ferroml-core -- pca linear regularized`
- `pytest ferroml-python/tests/ -k "pca or linear or ridge"`
- Benchmark: PCA n=10K d=50 should be <2x sklearn
- Benchmark: OLS/Ridge n=50K d=50 should be <2x sklearn

---

## Phase 2: LinearSVC — LIBLINEAR-style Optimization (2 days)

**Expected impact**: 9.6x→~1.5-2x

### 2a. Shrinking (Active Set Management)

Track which samples are "bounded" (alpha_i = 0 or alpha_i = C) and skip them.
After a few iterations, typically 50-70% of samples are bounded.

```rust
let mut active = vec![true; n_samples];
let mut active_count = n_samples;

// Only process active samples
for &i in &perm {
    if !active[i] { continue; }
    // ... existing DCD update ...

    // Mark as inactive if firmly at bound
    if alpha_new < 1e-12 && g > 0.0 {
        active[i] = false;
        active_count -= 1;
    }
}

// Periodically re-examine all (every 10 iterations)
if iter % 10 == 0 {
    for i in 0..n { active[i] = true; }
    active_count = n;
}
```

**Files**: `ferroml-core/src/models/svm.rs` (`fit_binary` for LinearSVC, lines 2494-2625)

### 2b. Cache Predictions `f_i = w^T x_i`

Instead of recomputing the full dot product every iteration, maintain cached predictions:

```rust
let mut f_cache: Vec<f64> = vec![0.0; n_samples]; // f_i = w^T x_i

// After alpha update with delta:
let delta = alpha_new - alpha[i];
if delta.abs() > 1e-15 {
    let scale = delta * y_binary[i];
    // Update w
    for j in 0..d { w_aug[j] += scale * x_aug[i][j]; }
    // Update ALL cached predictions: f_k += scale * x_k^T x_i
    // This is O(n*d) per UPDATE, but updates are rare after convergence starts
    // Alternative: only update f_i, recompute others lazily
}
```

Better approach — only update `f[i]` on demand:
```rust
// Before processing sample i, recompute f[i] = w^T x[i]
f_cache[i] = w_aug.iter().zip(x_aug[i].iter()).map(|(&w, &x)| w * x).sum();
```
This avoids the O(n*d) full update while still being correct.

### 2c. Direct ndarray Access (Remove Vec Copy)

Replace the `Vec<Vec<f64>>` copy with direct ndarray row access:

```rust
// Before (lines 2511-2521):
let x_aug: Vec<Vec<f64>> = ...  // Copies ALL data

// After: work directly with ndarray slices
let xi = x_design.row(i);
let wt_xi: f64 = xi.dot(&w);  // ndarray's optimized dot product
```

### Phase 2 Validation
- All LinearSVC Rust + Python tests pass
- Benchmark: LinearSVC n=5K d=20 should be <2x sklearn

---

## Phase 3: KMeans Elkan Optimization (1 day)

**Expected impact**: 3.2x→~1.2-1.5x

### 3a. Squared Distances Throughout

Remove ALL `.sqrt()` calls from the Elkan hot path. Use squared distances for comparisons.

**Lines to change** (in `ferroml-core/src/models/clustering.rs` or equivalent):
- Center-to-center distances: use squared, only sqrt for final output
- Point-to-center distances: use squared
- Triangle inequality: works with squared distances if you square the bounds too

```rust
// Before:
let dist = squared_euclidean_distance(&x_rows[i], &center_data[j]).sqrt();

// After:
let dist_sq = squared_euclidean_distance(&x_rows[i], &center_data[j]);
// Compare with squared bounds
```

### 3b. Zero-Copy Architecture

Replace the 3× per-iteration `Vec<Vec<f64>>` copies with direct ndarray access:

```rust
// Before (3 copies per iteration):
let x_rows: Vec<Vec<f64>> = (0..n).map(|i| x.row(i).to_vec()).collect();
let center_data: Vec<Vec<f64>> = (0..k).map(|j| centers.row(j).to_vec()).collect();

// After: use ndarray slices directly
let xi = x.row(i).as_slice().unwrap();
let cj = centers.row(j).as_slice().unwrap();
let dist_sq = squared_euclidean_distance(xi, cj);
```

### 3c. Allocate Bounds Once

Move `upper` and `lower` bound allocations outside the iteration loop:

```rust
// Before: allocated inside loop
let mut lower = vec![vec![0.0f64; k]; n_samples]; // 80K floats per iter

// After: allocated once, reset per iteration
let mut lower = vec![0.0f64; n_samples * k]; // flat layout, better cache
```

### Phase 3 Validation
- All KMeans Rust + Python tests pass
- Benchmark: KMeans k=8 n=10K d=10 should be <2x sklearn

---

## Phase 4: SVC Boundary Fix (Quick Win — 0.5 day)

**Expected impact**: 6.8x at n=3K→~2-3x

### 4a. Lower FULL_MATRIX_THRESHOLD

Currently 4000. The slab cache kicks in above this, but n=3000 uses the full precomputed matrix which is slower than expected. Lower to 2000 or even 1000:

```rust
const FULL_MATRIX_THRESHOLD: usize = 2_000;
```

For n=3000, the slab cache (200MB = 6250 rows at d=3000) will cache the full matrix anyway but with better access patterns.

### Phase 4 Validation
- All SVC tests pass
- Benchmark: SVC n=3K should drop from 6.8x to ~2-3x

---

## Execution Order

| Phase | Model | Current | Target | Effort | Priority |
|-------|-------|---------|--------|--------|----------|
| 1 | PCA/OLS/Ridge | 3.3-13.8x | <2x | 3 days | Critical |
| 2 | LinearSVC | 9.6x | <2x | 2 days | High |
| 3 | KMeans | 3.2x | <1.5x | 1 day | Medium |
| 4 | SVC boundary | 6.8x | <3x | 0.5 day | Quick win |

**Total: ~6.5 days across 4 phases.**

Phase 1 is the highest-leverage change — faer SVD alone closes the biggest gap (PCA 14x→~1.5x) and the infrastructure benefits OLS, Ridge, LDA, TruncatedSVD, FactorAnalysis, and IncrementalPCA.

## Key Constraint

**No system BLAS dependency.** faer is pure Rust — no OpenBLAS, no MKL, no system packages.
This is why faer is the right choice: LAPACK-competitive performance, pure Rust, already in our Cargo.toml.

## References

- [faer: High-Performance Linear Algebra in Rust](https://github.com/sarah-quinones/faer-rs) — benchmarks show near-LAPACK SVD
- [LIBLINEAR: A Library for Large Linear Classification](https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf) — shrinking + cached predictions
- [Elkan's K-Means](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf) — triangle inequality with squared distances
- [faer 0.20 benchmark tables](https://github.com/sarah-quinones/faer-rs/blob/main/paper.md) — SVD: 10-13x faster than nalgebra
