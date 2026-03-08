# Plan H: Gaussian Mixture Models (GMM)

## Overview

Implement GaussianMixture with full EM algorithm, multiple covariance types, and BIC/AIC model selection. Add to clustering module following existing KMeans/DBSCAN patterns.

## Current State

- Clustering module has KMeans, DBSCAN, AgglomerativeClustering
- `ClusteringModel` trait: `fit(x)`, `predict(x)`, `fit_predict(x)`, `labels()`, `is_fitted()`
- `ClusteringStatistics` trait: `cluster_stability()`, `silhouette_with_ci()`
- Linear algebra: QR decomposition, SIMD distances, but NO Cholesky in linalg.rs (exists privately in `hpo/bayesian.rs:576`)
- GaussianNB shows per-class Gaussian computation pattern (`naive_bayes.rs:113`)
- nalgebra used for SVD/eigendecomposition in PCA (`decomposition/pca.rs:410`)

## Desired End State

- `GaussianMixture` struct implementing `ClusteringModel` trait
- 4 covariance types: full, tied, diagonal, spherical
- EM algorithm with convergence monitoring
- BIC/AIC for model selection
- `predict_proba()` for soft clustering
- Python bindings + tests
- ~30 Rust tests, ~20 Python tests

---

## Phase H.1: Linear Algebra Utilities

**Overview**: Extract and generalize Cholesky decomposition, add log-determinant computation.

**Changes Required**:

1. **File**: `ferroml-core/src/linalg.rs` (EDIT)
   - Add `cholesky_decomposition(a: &Array2<f64>) -> Result<Array2<f64>>`
     - Extract from `hpo/bayesian.rs:576` and convert from `Vec<Vec<f64>>` to ndarray
   - Add `log_determinant_from_cholesky(l: &Array2<f64>) -> f64`
     - Sum of 2 * log(diagonal elements)
   - Add `solve_triangular_system(l: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>`
     - Forward substitution for L * X = B

**Success Criteria**:
- [ ] `cargo test -p ferroml-core linalg` — new tests pass
- [ ] Cholesky of known SPD matrix matches expected result

---

## Phase H.2: GaussianMixture Core Implementation

**Overview**: Implement the GaussianMixture struct with EM algorithm.

**Changes Required**:

1. **File**: `ferroml-core/src/clustering/gmm.rs` (NEW, ~800 lines)

   **Struct**:
   ```rust
   pub enum CovarianceType { Full, Tied, Diagonal, Spherical }
   pub enum GmmInit { KMeans, KMeansPlusPlus, Random, RandomFromData }

   pub struct GaussianMixture {
       // Config (match sklearn defaults exactly)
       n_components: usize,          // default 1
       covariance_type: CovarianceType, // default Full
       max_iter: usize,              // default 100
       tol: f64,                     // default 1e-3
       n_init: usize,                // default 1
       init_params: GmmInit,         // default KMeans
       reg_covar: f64,               // default 1e-6
       warm_start: bool,             // default false
       random_state: Option<u64>,
       // Fitted state
       weights_: Option<Array1<f64>>,         // mixing weights (n_components,)
       means_: Option<Array2<f64>>,           // (n_components, n_features)
       covariances_: Option<Vec<Array2<f64>>>,// per-component covariance (Full)
       precisions_cholesky_: Option<...>,     // precomputed for efficiency
       labels_: Option<Array1<i32>>,
       n_iter_: Option<usize>,
       lower_bound_: Option<f64>,             // final log-likelihood
       converged_: Option<bool>,
       n_features_in_: Option<usize>,
   }
   ```

   **Methods**:
   - `new(n_components)` + builder methods
   - `fit(x: &Array2<f64>) -> Result<()>` — run EM with n_init restarts
   - `predict(x: &Array2<f64>) -> Result<Array1<i32>>` — hard assignment (argmax of responsibilities)
   - `predict_proba(x: &Array2<f64>) -> Result<Array2<f64>>` — soft assignment (responsibilities)
   - `score_samples(x: &Array2<f64>) -> Result<Array1<f64>>` — per-sample log-likelihood
   - `score(x: &Array2<f64>) -> Result<f64>` — total log-likelihood
   - `bic(x: &Array2<f64>) -> Result<f64>` — Bayesian Information Criterion
   - `aic(x: &Array2<f64>) -> Result<f64>` — Akaike Information Criterion
   - `sample(n_samples: usize) -> Result<(Array2<f64>, Array1<i32>)>` — generate from model

   **Private methods**:
   - `e_step()` — compute responsibilities
   - `m_step()` — update weights, means, covariances
   - `compute_log_likelihood()` — evaluate convergence
   - `initialize_parameters()` — KMeans or random init

2. **File**: `ferroml-core/src/clustering/mod.rs` (EDIT)
   - Add `mod gmm;`
   - Add `pub use gmm::GaussianMixture;`
   - Add `pub use gmm::CovarianceType;`

**Success Criteria**:
- [ ] `cargo check -p ferroml-core`
- [ ] GMM fits 2D Gaussian mixture data and recovers component means

---

## Phase H.3: GaussianMixture Tests

**Overview**: Comprehensive Rust tests for GMM.

**Changes Required**:

1. **File**: `ferroml-core/src/clustering/gmm.rs` (EDIT — add `#[cfg(test)]` module)
   - ~30 tests covering:
     - Basic fit/predict on well-separated Gaussians
     - All 4 covariance types produce valid results
     - predict_proba sums to 1.0 per sample
     - BIC/AIC decrease with correct n_components
     - Convergence within max_iter
     - n_init picks best initialization
     - Edge cases: single component, single sample, single feature
     - score_samples returns finite values
     - Reproducibility with random_state
     - sample() generates valid data

**Success Criteria**:
- [ ] `cargo test -p ferroml-core gmm` — all ~30 tests pass

---

## Phase H.4: Python Bindings + Tests

**Overview**: Expose GaussianMixture to Python.

**Changes Required**:

1. **File**: `ferroml-python/src/clustering.rs` (EDIT)
   - Add `PyGaussianMixture` wrapper
   - Methods: `__new__`, `fit`, `predict`, `predict_proba`, `score_samples`, `bic`, `aic`, `sample`
   - Accessors: `weights_`, `means_`, `covariances_`, `n_iter_`, `converged_`

2. **File**: `ferroml-python/python/ferroml/clustering/__init__.py` (EDIT)
   - Add GaussianMixture to exports

3. **File**: `ferroml-python/tests/test_gmm.py` (NEW)
   - ~20 tests: fit/predict, predict_proba shape, BIC/AIC, covariance types, sample

**Success Criteria**:
- [ ] `cargo check -p ferroml-python`
- [ ] `pytest ferroml-python/tests/test_gmm.py -v` — all pass

---

## Execution Order

```
H.1 (linalg utilities)     — 15 min
H.2 (GMM core)             — 45 min, depends on H.1
H.3 (Rust tests)           — 20 min, depends on H.2
H.4 (Python bindings)      — 20 min, depends on H.2
```

H.3 and H.4 can run in parallel after H.2.

## Dependencies

- Phase H.1 (linalg) must come first
- nalgebra already available as dependency

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Numerical instability in Cholesky | Add reg_covar parameter (small diagonal regularization, default 1e-6) |
| EM convergence to local optima | n_init parameter runs multiple restarts, picks best log-likelihood |
| Full covariance scales poorly | Diagonal/spherical types avoid O(d^3) per-component |
| Cholesky fails for non-PD matrices | Add graceful error with suggestion to increase reg_covar |
