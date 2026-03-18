# PCA covariance_eigh Solver

**Goal**: Close the PCA performance gap. Currently 10x slower than sklearn because sklearn uses eigendecomposition of X'X (d×d) instead of SVD on X (n×d) when n >> d.

**Expected impact**: PCA 10x → ~1.5x (benchmarked: eigh is 9.5x faster than SVD at n=10K, d=50)

## The Insight

When n_samples >> n_features, eigendecomposing the d×d covariance matrix is much faster than SVD on the n×d data matrix:

| Method | Complexity | n=10K, d=50 |
|--------|-----------|-------------|
| Full SVD of X | O(n·d²) | ~15ms (numpy LAPACK) |
| Eigh of X'X | O(n·d² + d³) | ~1.6ms (numpy LAPACK) |

The n·d² term (forming X'X) is the same, but d³ eigendecomposition of a 50×50 matrix is trivial compared to SVD of a 10K×50 matrix.

**Trade-off**: Doubles the condition number (X'X squares singular values). Fine for well-conditioned data, risky for near-rank-deficient data.

## Implementation — 1 Phase, ~2 hours

### Step 1: Add `symmetric_eigh` to `linalg.rs`

Add faer-backed symmetric eigendecomposition following the existing pattern:

```rust
pub fn symmetric_eigh(a: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)>
```

Returns `(eigenvalues, eigenvectors)` with eigenvalues in **descending** order (PCA convention) and eigenvectors as columns.

**faer API**: `mat.selfadjoint_eigendecomposition(faer::Side::Lower)` → `.s()` for eigenvalues, `.u()` for eigenvectors.

**nalgebra fallback**: `mat.symmetric_eigen()` → `.eigenvalues`, `.eigenvectors`.

Both return eigenvalues ascending — reverse before returning.

**Files**: `ferroml-core/src/linalg.rs` (insert after SVD section, ~line 99)

### Step 2: Add `CovarianceEigh` variant to PCA

**a)** Add to `SvdSolver` enum:
```rust
pub enum SvdSolver {
    Auto,
    Full,
    Randomized,
    CovarianceEigh,  // NEW: eigendecomposition of X'X
}
```

**b)** Add `covariance_eigh` method:
```rust
fn covariance_eigh(
    &self,
    x_centered: &Array2<f64>,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (n_samples, n_features) = x_centered.dim();

    // Form covariance matrix: C = X'X (d×d)
    let cov = x_centered.t().dot(x_centered);

    // Eigendecompose (returns descending order)
    let (eigenvalues, eigenvectors) = crate::linalg::symmetric_eigh(&cov)?;

    // Clamp negative eigenvalues (numerical noise) to zero
    let singular_values = eigenvalues.mapv(|λ| λ.max(0.0).sqrt());

    // Components = eigenvectors transposed (each row is a principal direction)
    let vt = eigenvectors.t().to_owned();

    // U is not computed (not needed for transform, which uses components directly)
    let u_dummy = Array2::zeros((0, 0));

    Ok((u_dummy, singular_values, vt))
}
```

**c)** Update auto-selection in `compute_svd`:
```rust
SvdSolver::Auto => {
    if n_features <= 500 && n_samples > 2 * n_features {
        // Covariance eigh: O(n·d² + d³) — fast for tall-and-thin
        SvdSolver::CovarianceEigh
    } else if (n_samples > 500 && n_features > 500)
        || (n_features > 100 && n_features > 2 * n_samples)
    {
        SvdSolver::Randomized
    } else {
        SvdSolver::Full
    }
}
```

**d)** Update dispatch match:
```rust
match solver {
    SvdSolver::Full | SvdSolver::Auto => self.full_svd(x_centered),
    SvdSolver::Randomized => self.randomized_svd(x_centered),
    SvdSolver::CovarianceEigh => self.covariance_eigh(x_centered),
}
```

**Files**: `ferroml-core/src/decomposition/pca.rs`

### Step 3: Update Python bindings

Add `"covariance_eigh"` as a valid `svd_solver` string in the PyO3 wrapper.

**Files**: `ferroml-python/src/decomposition.rs` (search for SvdSolver string matching)

### Step 4: Test

- All existing PCA tests must pass (the auto-selection will now route to eigh for most test cases since they're tall-and-thin)
- Cross-validate: `PCA(svd_solver="full")` vs `PCA(svd_solver="covariance_eigh")` produce same components (up to sign flip) and same explained variance
- Benchmark: PCA n=10K d=50 should drop from 10x to ~1.5x

### Validation
```bash
cargo test -p ferroml-core -- pca
cargo test --test correctness -- pca
pytest ferroml-python/tests/ -k pca
python /tmp/bench_plan_z.py  # re-run benchmark
```

## Risk

Low. The covariance_eigh solver is a well-understood technique (sklearn has used it since v1.5). The only risk is numerical stability on ill-conditioned data, which is why it's only auto-selected when n >> d (condition number doubling matters less with regularization from averaging).
