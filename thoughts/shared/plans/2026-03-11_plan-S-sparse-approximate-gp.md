# Plan S: Sparse/Approximate Gaussian Processes

## Overview

The current GP implementation uses exact inference via full Cholesky decomposition, which is O(n^3) in time and O(n^2) in memory. This limits practical use to ~5K samples. Sparse GP methods use a small set of m inducing points to approximate the full posterior, reducing complexity to O(nm^2) time and O(nm) memory. This plan adds two industry-standard sparse GP approximations (FITC and VFE/SGPR), automatic inducing point selection, and a Sparse Variational GP (SVGP) for stochastic mini-batch training on datasets with 100K+ samples.

## Current State

- **Exact GP Regressor**: `ferroml-core/src/models/gaussian_process.rs:428-643` — Cholesky-based, stores full n x n kernel matrix, O(n^3) fit
- **Exact GP Classifier**: same file `:671-928` — Laplace approximation with Newton's method, also O(n^3)
- **Kernel trait**: `:18-27` — `compute()` returns full `Array2<f64>`, `diagonal()` returns `Array1<f64>`
- **6 kernels**: RBF, Matern (0.5/1.5/2.5), Constant, White, Sum, Product
- **Linear algebra**: hand-rolled Cholesky, forward/backward substitution (`:329-402`)
- **Python bindings**: `ferroml-python/src/gaussian_process.rs` — PyGPR, PyGPC, 4 kernel wrappers, `parse_kernel()` helper
- **Tests**: `ferroml-core/src/testing/gaussian_process.rs` — 32 Rust tests (14 kernel, 12 GPR, 6 GPC)
- **KMeans clustering**: `ferroml-core/src/clustering/kmeans.rs` — available for inducing point initialization
- **Sparse matrix**: `ferroml-core/src/sparse.rs` — CsrMatrix wrapper (not directly relevant here, "sparse" in sparse GP refers to inducing point approximations, not sparse matrices)

## Desired End State

- `SparseGPRegressor` with FITC and VFE approximations, O(nm^2) complexity
- `SparseGPClassifier` with FITC approximation + Laplace
- `SVGPRegressor` with stochastic variational inference for 100K+ datasets
- Automatic inducing point selection: k-means, random subset, greedy variance
- All three share the existing `Kernel` trait and all 6 kernels
- API-compatible with exact GP (same `Model` trait, `predict_with_std()`, etc.)
- Python bindings for all new classes
- Correctness verified against GPy/sklearn on synthetic + real datasets
- Performance benchmarks demonstrating scaling advantage

---

## Background: Sparse GP Methods

### FITC (Fully Independent Training Conditional)

Snelson & Ghahramani (2006). Approximates the full GP by assuming training points are conditionally independent given inducing points.

Given:
- X (n training), Z (m inducing), K_mm = K(Z,Z), K_nm = K(X,Z), K_nn_diag = diag(K(X,X))
- Lambda = diag(K_nn_diag - diag(Q_nn)) + sigma^2 * I, where Q_nn = K_nm @ K_mm^{-1} @ K_mn

**Training** (O(nm^2)):
1. Cholesky of K_mm: L_m = chol(K_mm + jitter*I), O(m^3)
2. V = L_m^{-1} @ K_mn, shape (m, n), O(nm^2)
3. Lambda_diag = K_nn_diag - colnorms(V)^2 + sigma^2, shape (n,)
4. Lambda_inv_half = 1/sqrt(Lambda_diag)
5. B = I_m + V @ diag(Lambda_inv) @ V^T, shape (m, m), O(nm^2)
6. L_B = chol(B), O(m^3)
7. beta = Lambda_inv * y
8. alpha = L_B^{-T} @ L_B^{-1} @ V @ beta, shape (m,)

**Prediction** (O(m^2) per test point):
- K_star_m = K(X_*, Z), shape (n_*, m)
- mean = K_star_m @ K_mm^{-1} @ K_mn @ Lambda_inv @ y (simplified via cached alpha)
- var = K_star_star_diag - diag(K_star_m @ (K_mm^{-1} - B^{-1}) @ K_m_star)

### VFE (Variational Free Energy) / SGPR

Titsias (2009). Treats inducing points as variational parameters and derives a lower bound on the marginal likelihood. Mathematically cleaner than FITC — uses a modified noise term.

The key difference from FITC: Lambda = sigma^2 * I (isotropic) plus a trace penalty term:
- trace_term = (1/(2*sigma^2)) * (tr(K_nn) - tr(Q_nn))
- This penalizes the approximation error, encouraging good inducing point placement

Training is structurally identical to FITC but with Lambda = sigma^2 * I.

### SVGP (Sparse Variational GP)

Hensman et al. (2013, 2015). Extends VFE to use stochastic optimization on mini-batches. Introduces variational parameters mu (m,) and S (m,m) for the inducing point posterior q(u) = N(mu, S).

**ELBO** (computed on mini-batch of size b):
- E_q[log p(y|f)] - KL[q(u) || p(u)]
- The first term decomposes over data points (hence mini-batch compatible)
- KL term is O(m^3) and independent of n

**Training**: gradient descent (L-BFGS or Adam) on variational parameters + inducing locations.

---

## Implementation Phases

### Phase S.1: Inducing Point Selection

**Overview**: Implement three strategies for selecting m inducing points from n training points. These are used by all subsequent sparse GP models.

**Changes Required**:

1. **File**: `ferroml-core/src/models/gaussian_process.rs` (extend, ~120 lines)
   - Add `InducingPointMethod` enum:
     ```rust
     #[derive(Debug, Clone)]
     pub enum InducingPointMethod {
         /// Random subset of training data
         RandomSubset { seed: Option<u64> },
         /// K-means cluster centers
         KMeans { max_iter: usize, seed: Option<u64> },
         /// Greedy variance maximization (iteratively pick point with highest posterior variance)
         GreedyVariance { seed: Option<u64> },
     }
     ```
   - Add `select_inducing_points()` function:
     ```rust
     pub fn select_inducing_points(
         x: &Array2<f64>,
         m: usize,
         method: &InducingPointMethod,
         kernel: &dyn Kernel,
     ) -> Result<Array2<f64>>
     ```
   - **RandomSubset**: use Fisher-Yates shuffle (import from existing RNG or use `rand` crate — check what the codebase already uses)
   - **KMeans**: call into `crate::clustering::kmeans::KMeans` with `n_clusters=m`, extract `cluster_centers()`
   - **GreedyVariance**: start with random point, iteratively select the point that maximizes kernel diagonal minus current posterior variance. O(nm^2) total — fine since it runs once.

2. **File**: `ferroml-core/src/testing/gaussian_process.rs` (extend, ~50 lines)
   - Test `select_inducing_points` with each method:
     - Output shape is (m, d)
     - m <= n constraint enforced
     - KMeans centers are in data-space range
     - GreedyVariance points are spread out (check minimum pairwise distance)
     - RandomSubset points are actual rows from X

**Success Criteria**:
- [ ] Automated: `cargo test --lib gaussian_process -- inducing` passes 6+ tests
- [ ] Automated: `cargo clippy` clean

---

### Phase S.2: FITC Sparse GP Regressor

**Overview**: Core sparse GP regressor using FITC approximation. This is the workhorse — handles 10K-50K samples with m=100-500 inducing points.

**Changes Required**:

1. **File**: `ferroml-core/src/models/gaussian_process.rs` (extend, ~350 lines)
   - Add `SparseApproximation` enum:
     ```rust
     #[derive(Debug, Clone, Copy, PartialEq)]
     pub enum SparseApproximation {
         FITC,
         VFE,
     }
     ```
   - Add `SparseGPRegressor` struct:
     ```rust
     #[derive(Debug, Clone)]
     pub struct SparseGPRegressor {
         kernel: Box<dyn Kernel>,
         alpha: f64,                           // noise variance sigma^2
         normalize_y: bool,
         n_inducing: usize,                    // m
         inducing_method: InducingPointMethod,
         approximation: SparseApproximation,   // FITC or VFE
         // Fitted state
         inducing_points_: Option<Array2<f64>>,   // Z, shape (m, d)
         l_m_: Option<Array2<f64>>,               // chol(K_mm + jitter)
         l_b_: Option<Array2<f64>>,               // chol(B)
         woodbury_vec_: Option<Array1<f64>>,       // cached vector for prediction mean
         y_train_mean_: Option<f64>,
         y_train_std_: Option<f64>,
         log_marginal_likelihood_: Option<f64>,
     }
     ```
   - **`fit()` implementation** (FITC path):
     1. Select inducing points Z via `select_inducing_points()`
     2. K_mm = kernel.compute(Z, Z) + jitter*I (jitter = 1e-8)
     3. L_m = cholesky(K_mm)
     4. K_nm = kernel.compute(X, Z), shape (n, m)
     5. V = solve_lower_triangular_mat(L_m, K_nm^T), shape (m, n) — L_m^{-1} @ K_mn
     6. K_nn_diag = kernel.diagonal(X)
     7. Q_nn_diag = column_norms_squared(V) — diag of V^T @ V
     8. Lambda_diag = K_nn_diag - Q_nn_diag + sigma^2 (FITC) or sigma^2 * ones (VFE)
     9. Lambda_diag = Lambda_diag.mapv(|v| v.max(1e-10)) — numerical safety
     10. Lambda_inv = 1.0 / Lambda_diag
     11. V_scaled = V * Lambda_inv_sqrt (broadcast columns)
     12. B = I_m + V_scaled @ V_scaled^T, shape (m, m)
     13. L_B = cholesky(B)
     14. beta = Lambda_inv * y_fit
     15. woodbury_vec = solve_cholesky(L_B, V @ beta)
     16. Compute log marginal likelihood:
         - For FITC: -0.5 * (y^T Lambda_inv y - beta^T V^T woodbury_vec + log|B| + sum(log Lambda_diag) + n*log(2pi))
         - For VFE: same but add trace penalty: -(1/(2*sigma^2)) * (sum(K_nn_diag) - sum(Q_nn_diag))
     17. Store all fitted state
   - **`predict()` and `predict_with_std()` implementation**:
     1. K_star_m = kernel.compute(X_*, Z)
     2. mean = K_star_m @ L_m^{-T} @ (L_m^{-1} @ K_mn @ Lambda_inv @ y - ...)
        Simplify: mean via woodbury_vec (precomputed)
     3. For variance:
        - v1 = L_m^{-1} @ K_m_star^T
        - v2 = L_B^{-1} @ v1 (forward solve)
        - var = K_star_star_diag - colnorms(v1)^2 + colnorms(v2)^2
   - **`Model` trait impl**: standard `fit`, `predict`, `is_fitted`, `n_features`, `search_space`
   - Builder pattern: `with_alpha()`, `with_normalize_y()`, `with_n_inducing()`, `with_inducing_method()`, `with_approximation()`

2. **File**: `ferroml-core/src/models/mod.rs`
   - No new module needed — everything stays in `gaussian_process.rs`
   - Add re-export if needed

3. **Helper functions** (same file, ~60 lines):
   - `fn mat_vec_product(a: &Array2<f64>, v: &Array1<f64>) -> Array1<f64>` — if not already available
   - `fn column_norms_squared(v: &Array2<f64>) -> Array1<f64>` — sum of squares per column
   - `fn solve_cholesky(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64>` — L L^T x = b

**Success Criteria**:
- [ ] Automated: `cargo test --lib gaussian_process -- sparse_gpr` passes 15+ tests
- [ ] Automated: `cargo clippy` clean
- [ ] Automated: `cargo fmt --check` clean

---

### Phase S.3: Sparse GP Regressor Tests & Correctness Validation

**Overview**: Thorough testing to ensure FITC and VFE produce correct results. Test against exact GP on small datasets (where they should nearly match), and verify scaling on larger datasets.

**Changes Required**:

1. **File**: `ferroml-core/src/testing/gaussian_process.rs` (extend, ~250 lines)

   **Correctness tests (FITC)**:
   - `test_sgpr_fitc_matches_exact_small_data`: With m=n inducing points (all training data), FITC should reproduce exact GP predictions to within tolerance (~1e-3). Use 20-point sin data.
   - `test_sgpr_fitc_prediction_mean_reasonable`: 100 training points, m=20 inducing. Predictions at training points should have R^2 > 0.9 for sin function.
   - `test_sgpr_fitc_uncertainty_near_training_small`: Std at training points should be small (< data range / 2).
   - `test_sgpr_fitc_uncertainty_far_away_large`: Std far from training data should approach prior variance.
   - `test_sgpr_fitc_normalize_y`: Same as exact GP normalize_y test with shifted targets.
   - `test_sgpr_fitc_log_marginal_likelihood_finite`: LML is finite and negative for typical data.
   - `test_sgpr_fitc_different_kernels`: Works with RBF, Matern 2.5, and SumKernel.
   - `test_sgpr_fitc_fewer_inducing_still_works`: m=5 still gives reasonable (R^2 > 0.5) predictions.

   **Correctness tests (VFE)**:
   - `test_sgpr_vfe_matches_exact_small_data`: Same as FITC counterpart.
   - `test_sgpr_vfe_prediction_reasonable`: 100 points, m=20, R^2 > 0.9.
   - `test_sgpr_vfe_trace_penalty_lml`: VFE LML <= FITC LML (VFE is a proper lower bound).
   - `test_sgpr_vfe_more_inducing_better_lml`: LML(m=30) > LML(m=10) for same data.

   **Edge case tests**:
   - `test_sgpr_m_equals_n_exact_match`: When m=n, sparse and exact give same results.
   - `test_sgpr_m_greater_than_n_clamped`: If m > n, automatically use m=n.
   - `test_sgpr_single_inducing_point`: m=1, should not crash, gives mean-like prediction.
   - `test_sgpr_not_fitted_error`: predict before fit gives NotFitted.
   - `test_sgpr_empty_data_error`: fit with empty data gives InvalidInput.
   - `test_sgpr_shape_mismatch`: X and y have different lengths.

   **Inducing method tests**:
   - `test_sgpr_random_subset_method`: Works, predictions are stable across seeds.
   - `test_sgpr_kmeans_method`: Uses k-means init, gives good predictions.
   - `test_sgpr_greedy_variance_method`: Greedy selection, predictions are at least as good as random.

2. **File**: `ferroml-python/tests/test_sparse_gp.py` (NEW, ~120 lines)
   - Python-side tests matching the Rust tests:
     - `test_sparse_gpr_fitc_basic`
     - `test_sparse_gpr_vfe_basic`
     - `test_sparse_gpr_predict_with_std`
     - `test_sparse_gpr_matches_exact_when_m_equals_n`
     - `test_sparse_gpr_different_kernels`
     - `test_sparse_gpr_normalize_y`
     - `test_sparse_gpr_inducing_methods`

**Success Criteria**:
- [ ] Automated: `cargo test --lib gaussian_process` — all 50+ tests pass
- [ ] Automated: `pytest ferroml-python/tests/test_sparse_gp.py` — all 7+ tests pass

---

### Phase S.4: Sparse GP Classifier

**Overview**: Extend FITC to classification via Laplace approximation on the inducing point posterior. Mirrors the exact GPC's Newton iteration but operates on the m-dimensional inducing space.

**Changes Required**:

1. **File**: `ferroml-core/src/models/gaussian_process.rs` (extend, ~300 lines)
   - Add `SparseGPClassifier` struct:
     ```rust
     #[derive(Debug, Clone)]
     pub struct SparseGPClassifier {
         kernel: Box<dyn Kernel>,
         n_inducing: usize,
         inducing_method: InducingPointMethod,
         max_iter: usize,
         tol: f64,
         // Fitted state
         inducing_points_: Option<Array2<f64>>,
         x_train_: Option<Array2<f64>>,
         classes_: Option<Vec<f64>>,
         f_hat_: Option<Array1<f64>>,           // mode at training points (n,)
         l_m_: Option<Array2<f64>>,             // chol(K_mm)
         k_nm_: Option<Array2<f64>>,            // K(X, Z)
     }
     ```
   - **`fit()` implementation**:
     1. Select inducing points Z
     2. K_mm = kernel.compute(Z, Z) + jitter
     3. K_nm = kernel.compute(X, Z)
     4. L_m = cholesky(K_mm)
     5. Newton iteration in the approximate posterior:
        - f = K_nm @ K_mm^{-1} @ K_mn @ a (using Woodbury for efficiency)
        - At each iteration, compute pi = sigmoid(f), W = pi*(1-pi)
        - Use FITC-structured matrix: Q_ff = K_nm @ K_mm^{-1} @ K_mn
        - B_approx = I_m + L_m^{-1} @ K_mn @ W_diag @ K_nm @ L_m^{-T}
        - Update a via Newton step
     6. Store fitted state
   - **`predict()` and `predict_proba()`**:
     - Compute posterior mean at test points via inducing point representation
     - Probit approximation for probabilities (same formula as exact GPC)
   - Builder: `with_n_inducing()`, `with_inducing_method()`, `with_max_iter()`, `with_tol()`

2. **File**: `ferroml-core/src/testing/gaussian_process.rs` (extend, ~100 lines)
   - `test_sgpc_linearly_separable`: 100% accuracy on well-separated classes
   - `test_sgpc_predict_proba_valid`: Probabilities sum to 1, in [0,1]
   - `test_sgpc_matches_exact_when_m_equals_n`: Same predictions as exact GPC
   - `test_sgpc_xor_pattern`: Non-linear decision boundary
   - `test_sgpc_not_fitted_error`
   - `test_sgpc_requires_two_classes`
   - `test_sgpc_different_inducing_methods`

**Success Criteria**:
- [ ] Automated: `cargo test --lib gaussian_process -- sparse_gpc` passes 7+ tests
- [ ] Automated: `cargo clippy` clean

---

### Phase S.5: SVGP (Sparse Variational GP) Regressor

**Overview**: Stochastic variational inference for GP regression. Unlike FITC/VFE which require all data in memory, SVGP processes mini-batches, enabling 100K+ datasets. Uses L-BFGS or simple gradient descent to optimize the ELBO.

**Changes Required**:

1. **File**: `ferroml-core/src/models/gaussian_process.rs` (extend, ~400 lines)
   - Add `SVGPRegressor` struct:
     ```rust
     #[derive(Debug, Clone)]
     pub struct SVGPRegressor {
         kernel: Box<dyn Kernel>,
         noise_variance: f64,                  // sigma^2
         n_inducing: usize,
         inducing_method: InducingPointMethod,
         n_epochs: usize,                      // number of passes through data
         batch_size: usize,                    // mini-batch size
         learning_rate: f64,                   // step size for natural gradient
         normalize_y: bool,
         // Fitted state
         inducing_points_: Option<Array2<f64>>,   // Z (m, d)
         mu_: Option<Array1<f64>>,                // variational mean (m,)
         l_s_: Option<Array2<f64>>,               // chol of variational covariance S (m, m)
         l_mm_: Option<Array2<f64>>,              // chol(K_mm) for prediction
         y_train_mean_: Option<f64>,
         y_train_std_: Option<f64>,
     }
     ```
   - **ELBO computation** for a mini-batch B of size b:
     ```
     ELBO = (n/b) * sum_{i in B} E_q[log N(y_i | f_i, sigma^2)] - KL[q(u) || p(u)]

     where:
     E_q[log N(y_i | f_i, sigma^2)] = -0.5 * (log(2pi*sigma^2) + (y_i - mu_i)^2/sigma^2 + trace_i/sigma^2)
     mu_i = K_ib @ K_mm^{-1} @ mu    (predictive mean at point i)
     trace_i = K_ii - K_ib @ K_mm^{-1} @ K_bi + K_ib @ K_mm^{-1} @ S @ K_mm^{-1} @ K_bi

     KL = 0.5 * (tr(K_mm^{-1} @ S) + mu^T K_mm^{-1} mu - m + log|K_mm| - log|S|)
     ```
   - **Natural gradient updates** (Hensman et al. 2013):
     - Natural gradients for mu and S have closed-form per mini-batch
     - This avoids slow standard gradient convergence
     - mu_new = mu + lr * S @ (K_mm^{-1} @ K_mb @ Lambda_inv_b @ (y_b - K_bm @ K_mm^{-1} @ mu) - K_mm^{-1} @ mu)
     - S_new^{-1} = S^{-1} + lr * (K_mm^{-1} @ K_mb @ Lambda_inv_b @ K_bm @ K_mm^{-1} - K_mm^{-1})
     - In practice: update the Cholesky factor L_S of S directly for stability
   - **`fit()` implementation**:
     1. Select inducing points Z
     2. Initialize mu = zeros(m), L_S = I_m
     3. For each epoch:
        a. Shuffle training data
        b. For each mini-batch of size b:
           - Compute K_mb = kernel.compute(X_b, Z)
           - Compute ELBO gradient (or use natural gradient)
           - Update mu, L_S
     4. Store fitted state
   - **`predict()` and `predict_with_std()`**:
     - K_star_m = kernel.compute(X_*, Z)
     - alpha = K_mm^{-1} @ mu
     - mean = K_star_m @ alpha
     - For variance: use S and K_mm to compute posterior variance
   - Builder: `with_noise_variance()`, `with_n_inducing()`, `with_n_epochs()`, `with_batch_size()`, `with_learning_rate()`, `with_normalize_y()`

2. **File**: `ferroml-core/src/testing/gaussian_process.rs` (extend, ~120 lines)
   - `test_svgp_basic_sin_regression`: Fits sin(x) with 200 points, m=30, R^2 > 0.8
   - `test_svgp_converges_to_vfe`: With batch_size=n (no stochasticity), should match VFE result
   - `test_svgp_predict_with_std_shapes`: Correct output shapes
   - `test_svgp_large_dataset`: 10K points, m=100, completes in < 10s
   - `test_svgp_uncertainty_calibration`: Std increases away from data
   - `test_svgp_not_fitted_error`
   - `test_svgp_different_batch_sizes`: batch_size=32, 64, 128 all converge
   - `test_svgp_learning_rate_effect`: Higher LR converges faster but may oscillate

**Success Criteria**:
- [ ] Automated: `cargo test --lib gaussian_process -- svgp` passes 8+ tests
- [ ] Automated: Large dataset test (10K points) completes in < 10s

---

### Phase S.6: Python Bindings

**Overview**: Expose all new sparse GP models to Python with sklearn-compatible API.

**Changes Required**:

1. **File**: `ferroml-python/src/gaussian_process.rs` (extend, ~350 lines)
   - Add `PyInducingPointMethod` enum-like class (or string parameter):
     - Accept string: "random", "kmeans", "greedy_variance"
   - Add `PySparseGPRegressor`:
     ```python
     class SparseGPRegressor:
         def __init__(self, kernel=None, alpha=1e-2, n_inducing=100,
                      inducing_method="kmeans", approximation="fitc",
                      normalize_y=False):
         def fit(self, X, y) -> self:
         def predict(self, X) -> ndarray:
         def predict_with_std(self, X) -> (ndarray, ndarray):
         @property
         def log_marginal_likelihood_(self) -> float:
         @property
         def inducing_points_(self) -> ndarray:
     ```
   - Add `PySparseGPClassifier`:
     ```python
     class SparseGPClassifier:
         def __init__(self, kernel=None, n_inducing=100,
                      inducing_method="kmeans", max_iter=50):
         def fit(self, X, y) -> self:
         def predict(self, X) -> ndarray:
         def predict_proba(self, X) -> ndarray:
         @property
         def inducing_points_(self) -> ndarray:
     ```
   - Add `PySVGPRegressor`:
     ```python
     class SVGPRegressor:
         def __init__(self, kernel=None, noise_variance=1.0, n_inducing=100,
                      inducing_method="kmeans", n_epochs=100, batch_size=256,
                      learning_rate=0.01, normalize_y=False):
         def fit(self, X, y) -> self:
         def predict(self, X) -> ndarray:
         def predict_with_std(self, X) -> (ndarray, ndarray):
         @property
         def inducing_points_(self) -> ndarray:
     ```
   - Update `parse_kernel()` — no changes needed (already supports all 4 kernels)
   - Update `register_gaussian_process_module()`:
     ```rust
     m.add_class::<PySparseGPRegressor>()?;
     m.add_class::<PySparseGPClassifier>()?;
     m.add_class::<PySVGPRegressor>()?;
     ```

2. **File**: `ferroml-python/python/ferroml/gaussian_process/__init__.py` (or wherever re-exports live)
   - Add re-exports: `SparseGPRegressor`, `SparseGPClassifier`, `SVGPRegressor`

3. **File**: `ferroml-python/python/ferroml/__init__.py`
   - Update docstring to mention sparse GP models

**Success Criteria**:
- [ ] Automated: `maturin develop --release -m ferroml-python/Cargo.toml` succeeds
- [ ] Automated: `python -c "from ferroml.gaussian_process import SparseGPRegressor, SVGPRegressor"` works

---

### Phase S.7: Python Tests & Correctness Validation

**Overview**: Comprehensive Python-side testing, including comparison against exact GP and large-scale benchmarks.

**Changes Required**:

1. **File**: `ferroml-python/tests/test_sparse_gp.py` (NEW or extend from S.3, ~250 lines)

   **SparseGPRegressor tests**:
   - `test_sparse_gpr_fitc_sin_regression`: R^2 > 0.9 on sin data
   - `test_sparse_gpr_vfe_sin_regression`: R^2 > 0.9 on sin data
   - `test_sparse_gpr_predict_with_std_shapes`: Correct shapes
   - `test_sparse_gpr_matches_exact_m_eq_n`: Nearly matches exact GPR
   - `test_sparse_gpr_inducing_points_accessible`: `model.inducing_points_` returns ndarray
   - `test_sparse_gpr_normalize_y_shifted_target`: Works with y = sin(x) + 1000
   - `test_sparse_gpr_all_kernels`: RBF, Matern, ConstantKernel work
   - `test_sparse_gpr_inducing_methods`: "random", "kmeans", "greedy_variance" all work

   **SparseGPClassifier tests**:
   - `test_sparse_gpc_linearly_separable`: 100% accuracy
   - `test_sparse_gpc_predict_proba_valid`: Sum to 1, in [0,1]
   - `test_sparse_gpc_matches_exact_m_eq_n`: Nearly matches exact GPC

   **SVGPRegressor tests**:
   - `test_svgp_basic_regression`: R^2 > 0.7 on sin data
   - `test_svgp_predict_with_std`: Returns valid std
   - `test_svgp_large_dataset`: 20K points completes in < 30s
   - `test_svgp_batch_size_parameter`: Different batch sizes work

   **Comparison tests** (generate sklearn GP fixtures):
   - `test_sparse_gpr_vs_sklearn_gpr`: On 200-point dataset, FITC predictions within 0.2 of sklearn exact GP
   - `test_svgp_vs_sklearn_gpr`: On 500-point dataset, SVGP predictions within 0.3 of sklearn exact GP

**Success Criteria**:
- [ ] Automated: `pytest ferroml-python/tests/test_sparse_gp.py -v` — all 18+ tests pass
- [ ] Automated: No test takes more than 30s

---

### Phase S.8: Performance Benchmarks & Documentation

**Overview**: Demonstrate scaling advantage and document the new models.

**Changes Required**:

1. **File**: `scripts/benchmark_sparse_gp.py` (NEW, ~150 lines)
   - Benchmark: Exact GPR vs FITC vs VFE vs SVGP
   - Dataset sizes: 500, 1K, 2K, 5K, 10K, 20K, 50K
   - Inducing points: m=50, 100, 200
   - Metrics: fit time, predict time, R^2, memory (approximate)
   - Output: JSON + markdown table
   - Expected results:
     - Exact GP: works up to ~5K, O(n^3) scaling
     - FITC/VFE: works up to ~50K, O(nm^2) scaling
     - SVGP: works up to 100K+, near-constant per-epoch time

2. **File**: `ferroml-core/benches/gaussian_process.rs` (NEW or extend, ~80 lines)
   - Criterion benchmarks:
     - `bench_exact_gpr_fit_100`, `bench_exact_gpr_fit_500`, `bench_exact_gpr_fit_1000`
     - `bench_fitc_gpr_fit_1000_m50`, `bench_fitc_gpr_fit_5000_m100`
     - `bench_vfe_gpr_fit_1000_m50`, `bench_vfe_gpr_fit_5000_m100`
     - `bench_svgp_fit_10000_m100_epoch1`
     - `bench_fitc_gpr_predict_1000`
     - `bench_inducing_kmeans_1000_m50`, `bench_inducing_greedy_1000_m50`

3. **File**: Update CHANGELOG.md
   - Add Plan S entry documenting sparse GP models

**Success Criteria**:
- [ ] Automated: `python scripts/benchmark_sparse_gp.py` produces results
- [ ] Automated: `cargo bench -- gaussian_process` runs without error
- [ ] Manual: Review benchmark numbers — FITC 5K should be >5x faster than exact 5K

---

## Dependencies

- **Existing**: Kernel trait, Cholesky/triangular solvers, KMeans clustering, Model trait
- **No new crate dependencies** — all linear algebra is hand-rolled (consistent with existing GP code)
- **Phase ordering**: S.1 -> S.2 -> S.3 (can start S.4 after S.2). S.5 independent of S.4. S.6 needs S.2+S.4+S.5. S.7 needs S.6. S.8 can start after S.6.

```
S.1 (inducing points)
  |
  v
S.2 (FITC/VFE regressor) -----> S.4 (sparse GPC) -------+
  |                                                        |
  v                                                        v
S.3 (regressor tests) ---> S.5 (SVGP) ------> S.6 (Python bindings)
                                                    |
                                                    v
                                               S.7 (Python tests)
                                                    |
                                                    v
                                               S.8 (benchmarks)
```

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Numerical instability in FITC Woodbury** | Wrong predictions, NaN | Add jitter (1e-8) to K_mm diagonal; clamp Lambda_diag to >= 1e-10; test with ill-conditioned kernels |
| **SVGP natural gradients diverge** | Training fails | Add learning rate warmup; clamp S to remain positive definite by working with Cholesky factor L_S; default conservative LR=0.01 |
| **K-means inducing points cluster poorly for non-Euclidean kernels** | Suboptimal approximation | Offer greedy variance as alternative; document that k-means is Euclidean |
| **SVGP mini-batch noise** | Slow convergence | Default batch_size=256 (empirically good); allow users to increase; full-batch fallback |
| **Sparse GPC Newton iteration fails to converge** | fit() error | Use damped Newton (step size < 1.0); increase max_iter default; fall back to fixed-point iteration |
| **Large m makes sparse GP as slow as exact** | No speedup | Validate m < n at fit time; warn if m > n/2; document recommended m ranges |
| **VFE trace penalty makes LML too negative** | Misleading model comparison | This is expected behavior — VFE is a proper lower bound. Document this. |

## Test Count Estimate

| Phase | Rust Tests | Python Tests | Total |
|-------|-----------|--------------|-------|
| S.1   | 6         | 0            | 6     |
| S.2   | 0         | 0            | 0     |
| S.3   | 20        | 7            | 27    |
| S.4   | 7         | 0            | 7     |
| S.5   | 8         | 0            | 8     |
| S.6   | 0         | 0            | 0     |
| S.7   | 0         | 18           | 18    |
| S.8   | 0         | 0            | 0     |
| **Total** | **41** | **25**      | **66** |

## API Summary

After Plan S, the GP module offers four levels of scalability:

| Model | Complexity | Max n | Method |
|-------|-----------|-------|--------|
| `GaussianProcessRegressor` | O(n^3) | ~5K | Exact Cholesky |
| `SparseGPRegressor(approx="fitc")` | O(nm^2) | ~50K | FITC inducing points |
| `SparseGPRegressor(approx="vfe")` | O(nm^2) | ~50K | Variational (Titsias) |
| `SVGPRegressor` | O(bm^2) per step | 100K+ | Stochastic variational |
| `GaussianProcessClassifier` | O(n^3) | ~5K | Exact Laplace |
| `SparseGPClassifier` | O(nm^2) | ~50K | FITC + Laplace |
