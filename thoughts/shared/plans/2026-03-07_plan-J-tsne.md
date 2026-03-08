# Plan J: t-SNE (t-distributed Stochastic Neighbor Embedding)

## Overview

Implement t-SNE for nonlinear dimensionality reduction / visualization. Start with exact O(N^2) algorithm, optionally add Barnes-Hut O(N log N) approximation for large datasets.

## Current State

- Decomposition module: PCA, TruncatedSVD, LDA, FactorAnalysis — all implement `Transformer` trait
- `Transformer` trait: `fit(x)`, `transform(x)`, `fit_transform(x)`, `inverse_transform(x)`
- SIMD-accelerated distances: `squared_euclidean_distance`, `batch_squared_euclidean` in `simd.rs:59`
- SGD/Adam optimizers in `neural/optimizers.rs`
- PCA uses nalgebra for SVD (`decomposition/pca.rs:410`)
- No spatial trees (Barnes-Hut, VP-tree) exist

## Desired End State

- `TSNE` struct implementing `Transformer` trait (fit_transform only — transform on new data is not standard for t-SNE)
- Configurable: perplexity, learning_rate, n_iter, n_components, early_exaggeration
- Exact algorithm for datasets up to ~5K points
- Optional PCA initialization for better convergence
- Python bindings + tests
- ~25 Rust tests, ~15 Python tests

---

## Phase J.1: t-SNE Core Implementation

**Overview**: Implement exact t-SNE with gradient descent.

**Changes Required**:

1. **File**: `ferroml-core/src/decomposition/tsne.rs` (NEW, ~600 lines)

   **Struct**:
   ```rust
   pub struct TSNE {
       // Config
       n_components: usize,          // output dimensions (default 2)
       perplexity: f64,              // effective number of neighbors (default 30.0)
       learning_rate: LearningRate,   // "auto" = max(N/early_exaggeration/4, 50) or fixed float
       max_iter: usize,              // max iterations (default 1000, sklearn renamed from n_iter)
       early_exaggeration: f64,      // factor for first 250 iters (default 12.0)
       min_grad_norm: f64,           // convergence threshold (default 1e-7)
       metric: TsneMetric,           // distance metric (default Euclidean)
       init: TsneInit,              // initialization method (default PCA)
       random_state: Option<u64>,
       // Fitted
       embedding_: Option<Array2<f64>>,  // (n_samples, n_components)
       kl_divergence_: Option<f64>,      // final KL divergence
       n_iter_final_: Option<usize>,     // iterations actually run
       n_features_in_: Option<usize>,
   }

   pub enum TsneInit { Random, Pca }
   pub enum TsneMetric { Euclidean, Manhattan, Cosine }
   ```

   **Algorithm (exact)**:
   1. Compute pairwise distances D_ij using SIMD utilities
   2. Binary search for sigma_i per point to match target perplexity
   3. Compute conditional probabilities p_{j|i} = exp(-D_ij / 2*sigma_i^2) / sum
   4. Symmetrize: P_ij = (p_{j|i} + p_{i|j}) / (2N)
   5. Initialize Y (PCA or random)
   6. Gradient descent loop:
      - Compute Student-t kernel: q_ij = (1 + ||y_i - y_j||^2)^(-1) / sum
      - Gradient: dY = 4 * sum((p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^(-1))
      - Apply momentum-based update (momentum = 0.5 for iter < 250, 0.8 after)
      - Early exaggeration: multiply P by factor for first 250 iterations

   **Key private methods**:
   - `compute_pairwise_distances(x)` — use `simd::batch_squared_euclidean`
   - `binary_search_perplexity(distances, target_perplexity)` — find sigma per point
   - `compute_joint_probabilities(distances, sigmas)` — P matrix
   - `compute_gradient(p, y)` — KL divergence gradient
   - `kl_divergence(p, q)` — loss function

   **Implements Transformer trait** (match sklearn: NO standalone transform):
   - `fit(x)` — compute embedding, store in embedding_
   - `fit_transform(x)` — main entry point (this is what users call)
   - `transform(x)` — return stored embedding_ (t-SNE is transductive, no out-of-sample)
   - `inverse_transform()` — return NotImplemented error
   - Note: sklearn has NO `transform()` method at all, only `fit_transform()`.
     We can add `transform()` that just returns the stored embedding for pipeline compat.

2. **File**: `ferroml-core/src/decomposition/mod.rs` (EDIT)
   - Add `mod tsne;`
   - Add `pub use tsne::TSNE;`

**Success Criteria**:
- [ ] `cargo check -p ferroml-core`
- [ ] t-SNE separates well-separated clusters in 2D output

---

## Phase J.2: t-SNE Tests

**Changes Required**:

1. **File**: `ferroml-core/src/decomposition/tsne.rs` (EDIT — `#[cfg(test)]` module)
   - ~25 tests:
     - Separates 3 well-separated Gaussian clusters
     - Output shape is (n_samples, n_components)
     - KL divergence decreases over iterations
     - Perplexity affects neighborhood size
     - PCA vs random initialization
     - Reproducibility with random_state
     - n_components = 3 works
     - Single cluster (should stay compact)
     - Early exaggeration phase works
     - learning_rate affects convergence speed
     - Edge cases: very small dataset, very large perplexity
     - Cosine/Manhattan metrics work

**Success Criteria**:
- [ ] `cargo test -p ferroml-core tsne` — all pass

---

## Phase J.3: Python Bindings + Tests

**Changes Required**:

1. **File**: `ferroml-python/src/decomposition.rs` (EDIT)
   - Add `PyTSNE` wrapper
   - Methods: `__new__`, `fit`, `transform`, `fit_transform`, `__repr__`
   - Accessors: `embedding_`, `kl_divergence_`, `n_iter_`

2. **File**: `ferroml-python/python/ferroml/decomposition/__init__.py` (EDIT)
   - Add TSNE to exports

3. **File**: `ferroml-python/tests/test_tsne.py` (NEW)
   - ~15 tests: fit_transform output shape, cluster separation, perplexity param, random_state

**Success Criteria**:
- [ ] `cargo check -p ferroml-python`
- [ ] `pytest ferroml-python/tests/test_tsne.py -v` — all pass

---

## Execution Order

```
J.1 (t-SNE core)        — 45 min
J.2 (Rust tests)         — 20 min, depends on J.1
J.3 (Python bindings)    — 15 min, depends on J.1
```

J.2 and J.3 can run in parallel.

## Dependencies

- SIMD distance utilities in `simd.rs` (already exist)
- PCA for initialization (already exists in `decomposition/pca.rs`)

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| O(N^2) is slow for N > 5K | Document limitation; suggest PCA preprocessing; Barnes-Hut is future work |
| Perplexity binary search may not converge | Set max iterations (50) with tolerance; warn if not converged |
| Numerical instability in probability computation | Use log-space arithmetic; add epsilon to denominators |
| t-SNE is non-deterministic without seed | Require random_state for reproducible tests |
| Early exaggeration timing is critical | Use standard sklearn defaults (250 iters, factor 12.0) |

## Future Work (NOT in this plan)

- Barnes-Hut approximation for O(N log N) — requires VP-tree or quad-tree
- UMAP (different algorithm entirely, not a t-SNE variant)
- GPU acceleration for pairwise distances
