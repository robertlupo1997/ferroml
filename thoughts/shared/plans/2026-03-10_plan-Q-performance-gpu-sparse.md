# Plan Q: Performance — GPU Hardening & Sparse Integration

## Overview

Expand FerroML's performance story across two tracks: (1) harden the existing GPU/wgpu backend with more shaders, model integrations, and robustness, and (2) wire the existing sparse matrix module into models for O(nnz) text/NLP workloads.

## Current State

### GPU Backend (complete but minimal)
- **Trait**: `GpuBackend` with 2 operations: `matmul()`, `pairwise_distances()` (`gpu/mod.rs:33-44`)
- **Implementation**: `WgpuBackend` using wgpu v23 (`gpu/backend.rs`, 327 lines)
- **Shaders**: 2 WGSL shaders — tiled 16×16 matmul + pairwise distance (`gpu/kernels.rs`, 346 lines)
- **Model integrations**: KMeans (distance computation), MLP (forward/backward matmul)
- **Tests**: 67 total (35 mock, 24 shader validation, 8 CPU fallback)
- **Feature flag**: `--features gpu` (wgpu + bytemuck + pollster)

### Sparse Module (complete but not integrated)
- **Data structures**: `CsrMatrix`, `CscMatrix`, `SparseVector`, `SparseRowView` (`sparse.rs`, 1,282 lines)
- **Distance functions**: Euclidean, Manhattan, Cosine, batch pairwise (`sparse.rs:528-715`)
- **Operations**: dot, row norms, column sums/means, normalize, vstack/hstack, eye, diag
- **Trait**: `SparseModel` defined but **zero implementations** (`models/traits.rs:55-63`)
- **Tests**: 25 unit tests in `sparse.rs` + 65 in `testing/sparse_tests.rs`
- **Feature flag**: `--features sparse` (sprs crate)
- **Python**: `sparse_utils.rs` converts scipy.sparse → dense immediately (no native sparse compute)
- **No TfidfTransformer** exists anywhere in the codebase

## Desired End State

- GPU: 6+ shader operations, 6+ model integrations, automatic CPU/GPU dispatch, robust error handling
- Sparse: 8+ models implement `SparseModel` trait, TfidfTransformer, Python sparse pipeline end-to-end
- ~135 new tests across 8 phases

---

## Implementation Phases

### Phase Q.1: GPU Shader Expansion

**Overview**: Add element-wise and reduction shaders to enable full neural network forward/backward on GPU.

**Changes Required**:

1. **File**: `ferroml-core/src/gpu/kernels.rs`
   - Add `RELU_SHADER`: element-wise max(0, x), workgroup_size(256), 1D dispatch
   - Add `SIGMOID_SHADER`: element-wise 1/(1+exp(-x)), workgroup_size(256)
   - Add `SOFTMAX_SHADER`: row-wise softmax (two-pass: max reduction then exp/sum), workgroup_size(256)
   - Add `ROW_REDUCE_SHADER`: row-wise sum/max reduction, workgroup_size(256)
   - Add `BIAS_ADD_SHADER`: broadcast add bias vector to each row, workgroup_size(256)
   - Each shader: uniform dims struct, storage buffers for input/output
   - Shader validation tests (binding count, workgroup size, WGSL syntax) — same pattern as existing 24 tests

2. **File**: `ferroml-core/src/gpu/mod.rs`
   - Expand `GpuBackend` trait with new methods:
     ```rust
     fn relu(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
     fn sigmoid(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
     fn softmax(&self, x: &Array2<f64>) -> Result<Array2<f64>>;  // row-wise
     fn row_sum(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
     fn row_max(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
     fn bias_add(&self, x: &Array2<f64>, bias: &Array1<f64>) -> Result<Array2<f64>>;
     ```
   - Add mock implementations in `MockGpuBackend` (CPU-based)
   - Add trait object dispatch tests (Arc, Box, &dyn)

3. **File**: `ferroml-core/src/gpu/backend.rs`
   - Create compute pipelines for each new shader in `try_new_async()`
   - Implement each new trait method following the matmul pattern:
     - Create uniform buffer with dims
     - Upload f64→f32 input
     - Dispatch workgroups
     - Download f32→f64 output
   - Add `relu_pipeline`, `sigmoid_pipeline`, `softmax_pipeline`, `reduce_pipeline`, `bias_pipeline` fields

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features gpu -- gpu` passes
- [ ] Automated: `cargo clippy -p ferroml-core --features gpu -- -D warnings` clean
- [ ] ~20 new tests (5 shaders × 2 validation tests + 6 mock parity tests + 4 edge cases)

---

### Phase Q.2: GPU Model Integrations

**Overview**: Wire GPU pairwise distances and matmul into DBSCAN, HDBSCAN, t-SNE, and GradientBoosting.

**Changes Required**:

1. **File**: `ferroml-core/src/clustering/dbscan.rs`
   - Add `#[cfg(feature = "gpu")] gpu_backend: Option<Arc<dyn GpuBackend>>` field (line ~53)
   - Add `with_gpu()` builder method
   - In `region_query()` (line 133): if GPU available, compute full pairwise distance matrix once via `gpu.pairwise_distances()`, cache it, use for all neighborhood queries
   - Fallback to existing CPU path if GPU absent or errors

2. **File**: `ferroml-core/src/clustering/hdbscan.rs`
   - Add `#[cfg(feature = "gpu")]` field + builder
   - In core distance computation: use `gpu.pairwise_distances()` for the mutual reachability distance matrix
   - This is the O(N²) bottleneck — GPU gives biggest win here

3. **File**: `ferroml-core/src/decomposition/tsne.rs`
   - Add `#[cfg(feature = "gpu")]` field + builder
   - In exact t-SNE gradient computation: the O(N²) pairwise distance can dispatch to GPU
   - Barnes-Hut path stays CPU (tree-based, not GPU-friendly)

4. **File**: `ferroml-core/src/models/gradient_boosting.rs` (or hist variant)
   - Add `#[cfg(feature = "gpu")]` field + builder
   - In batch `predict()`: use `gpu.matmul()` for leaf-value aggregation if beneficial
   - Only wire if matrix dimensions justify GPU dispatch

5. **File**: `ferroml-core/src/testing/gpu_fallback.rs`
   - Add fallback tests for each new integration (DBSCAN, HDBSCAN, t-SNE, GB)
   - Pattern: build model without GPU → fit/predict works; build with mock GPU → same results

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features gpu -- gpu` passes
- [ ] Automated: `cargo test -p ferroml-core -- dbscan hdbscan tsne gradient_boosting` (CPU path unbroken)
- [ ] ~16 new tests (4 models × 4 tests: with-gpu, without-gpu, mock-parity, error-fallback)

---

### Phase Q.3: GPU Neural Network Full Pipeline

**Overview**: Wire Q.1 shaders into MLP so the full forward+backward pass runs on GPU without CPU roundtrips.

**Changes Required**:

1. **File**: `ferroml-core/src/neural/layers.rs`
   - Update `forward_gpu()` (line 306): replace CPU activation with `gpu.relu()` / `gpu.sigmoid()`
   - Current code: `z = gpu.matmul(input, weights) + biases` then CPU activation
   - New code: `z = gpu.bias_add(gpu.matmul(input, weights), biases)` then `gpu.relu(z)` / `gpu.sigmoid(z)`
   - For output layer with softmax: use `gpu.softmax(z)`
   - Update `backward_gpu()` (line 338): add GPU activation gradient computation
     - ReLU gradient: element-wise mask (z > 0)
     - Sigmoid gradient: σ(z) * (1 - σ(z)) — reuse forward output

2. **File**: `ferroml-core/src/gpu/kernels.rs`
   - Add `RELU_GRAD_SHADER`: element-wise (z > 0) ? 1.0 : 0.0
   - Add `SIGMOID_GRAD_SHADER`: element-wise output * (1 - output)

3. **File**: `ferroml-core/src/gpu/mod.rs`
   - Add to trait:
     ```rust
     fn relu_grad(&self, z: &Array2<f64>) -> Result<Array2<f64>>;
     fn sigmoid_grad(&self, output: &Array2<f64>) -> Result<Array2<f64>>;
     fn elementwise_mul(&self, a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>>;
     ```

4. **File**: `ferroml-core/src/gpu/backend.rs`
   - Implement the 3 new trait methods + pipelines

5. **File**: `ferroml-core/src/neural/mlp.rs`
   - Update training loop GPU path (line 329-375): use new activation shaders
   - Ensure gradient caching still works (last_z, last_output needed for backward)

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features gpu -- mlp neural` passes
- [ ] Automated: Mock GPU MLP produces same loss trajectory as CPU MLP (within f32 tolerance)
- [ ] ~12 new tests (shader validation, mock forward parity, mock backward parity, full train loop parity, edge cases)

---

### Phase Q.4: GPU Robustness & Auto-Dispatch

**Overview**: Add error recovery, automatic CPU/GPU dispatch based on matrix size, memory limit checks, and timeout handling.

**Changes Required**:

1. **File**: `ferroml-core/src/gpu/mod.rs`
   - Add `GpuDispatchPolicy` enum:
     ```rust
     pub enum GpuDispatchPolicy {
         Always,           // Always use GPU (fail on error)
         Auto { min_elements: usize },  // GPU if matrix > threshold, else CPU
         Never,            // CPU only (useful for testing)
     }
     ```
   - Add `dispatch_policy` field to trait or as wrapper struct
   - Add `GpuMemoryInfo` struct: `max_buffer_size`, `max_storage_buffer_binding_size`

2. **File**: `ferroml-core/src/gpu/backend.rs`
   - Query adapter limits in `try_new_async()`:
     ```rust
     let limits = adapter.limits();
     let max_buffer = limits.max_buffer_size;
     let max_storage = limits.max_storage_buffer_binding_size;
     ```
   - Store limits in `WgpuBackend` struct
   - Add `pub fn memory_info(&self) -> GpuMemoryInfo`
   - Pre-flight check in each operation: if input buffer size > max, return `Err` with clear message
   - Add device-lost callback: `device.on_uncaptured_error(|err| ...)` → log warning
   - Add timeout: `device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx))` with deadline

3. **File**: `ferroml-core/src/gpu/dispatch.rs` (NEW)
   - `GpuDispatcher` struct wrapping `Arc<dyn GpuBackend>` + policy
   - `matmul()` method: check element count vs threshold, dispatch accordingly
   - `pairwise_distances()`: same pattern
   - Crossover point constants (tunable): `MATMUL_GPU_THRESHOLD = 4096` (64×64)
   - All element-wise ops: threshold based on total elements

4. **File**: `ferroml-core/benches/gpu_benchmarks.rs`
   - Add benchmarks for new shaders (relu, sigmoid, softmax)
   - Add crossover-point benchmark: sweep matrix sizes [32, 64, 128, 256, 512, 1024, 2048]
   - Measure CPU vs GPU at each size to validate dispatch thresholds

5. **File**: `ferroml-core/src/gpu/mod.rs` (tests section)
   - Error path tests: oversized buffer, device-lost simulation, timeout
   - Auto-dispatch tests: small matrix → CPU, large matrix → GPU
   - Memory info tests

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features gpu -- gpu dispatch` passes
- [ ] Automated: `cargo clippy -p ferroml-core --features gpu -- -D warnings` clean
- [ ] ~15 new tests (dispatch policy, memory limits, error paths, timeout, auto-dispatch threshold)

---

### Phase Q.5: Sparse KNN + NearestCentroid + DBSCAN

**Overview**: Wire existing sparse distance functions into KNN, NearestCentroid, and DBSCAN via the `SparseModel` trait.

**Changes Required**:

1. **File**: `ferroml-core/src/models/traits.rs`
   - The `SparseModel` trait (line 55-63) uses `sprs::CsMat<f64>` directly
   - Update to use wrapper: `&crate::sparse::CsrMatrix` instead (more ergonomic, hides sprs dependency)
     ```rust
     #[cfg(feature = "sparse")]
     pub trait SparseModel {
         fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &Array1<f64>) -> Result<()>;
         fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> Result<Array1<f64>>;
     }
     ```

2. **File**: `ferroml-core/src/models/knn.rs`
   - `impl SparseModel for KNeighborsClassifier`:
     - `fit_sparse()`: store `CsrMatrix` alongside or instead of dense `x_train`
       - Add field: `#[cfg(feature = "sparse")] x_train_sparse: Option<CsrMatrix>`
     - `predict_sparse()`: for each query row, compute distances via `sparse_pairwise_distances()`, find k-nearest
     - Skip KDTree/BallTree for sparse (brute-force is standard for high-dim sparse)
   - `impl SparseModel for KNeighborsRegressor`: same pattern
   - `impl SparseModel for NearestCentroid`:
     - `fit_sparse()`: compute per-class centroids from sparse rows (iterate nnz only)
     - `predict_sparse()`: sparse distance to each centroid

3. **File**: `ferroml-core/src/clustering/dbscan.rs`
   - Add `fit_sparse(&self, x: &CsrMatrix) -> Result<Array1<i32>>` method (not SparseModel trait — clustering uses different signature)
   - Sparse `region_query()`: use `sparse_squared_euclidean_distance()` per pair
   - For large datasets: precompute sparse pairwise distance matrix once

4. **File**: `ferroml-core/src/testing/sparse_tests.rs` (or new `testing/sparse_models.rs`)
   - Equivalence tests: fit/predict on dense vs sparse representation → same results
   - Text-like data tests (99% sparse, 10K features)
   - Edge cases: empty rows, single-feature sparse, all-zero query

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features sparse -- sparse knn nearest_centroid dbscan` passes
- [ ] Automated: Dense vs sparse predictions match within tolerance
- [ ] ~20 new tests (KNN: 6, KNNReg: 4, NearestCentroid: 4, DBSCAN: 4, edge cases: 2)

---

### Phase Q.6: Sparse Naive Bayes (all 4)

**Overview**: Implement `SparseModel` for all 4 Naive Bayes variants. MultinomialNB and BernoulliNB are the classic text classifiers — sparse support is essential.

**Changes Required**:

1. **File**: `ferroml-core/src/models/naive_bayes.rs`

   **MultinomialNB** (~line 702):
   - `impl SparseModel for MultinomialNB`:
     - `fit_sparse()`: iterate sparse rows per class, accumulate feature counts via nnz indices only
       ```rust
       for (row_idx, &label) in y.iter().enumerate() {
           let row = x.row(row_idx);
           for (&col, &val) in row.indices().iter().zip(row.data().iter()) {
               feature_count[class_idx][col] += val;
           }
       }
       ```
     - `predict_sparse()`: log-probability via sparse dot product: `log_prob[c] = class_log_prior[c] + sparse_row · feature_log_prob[c]`

   **BernoulliNB** (~line 1206):
   - `impl SparseModel for BernoulliNB`:
     - `fit_sparse()`: count non-zero feature occurrences per class (presence/absence)
     - `predict_sparse()`: binary log-likelihood, only iterate nnz indices

   **GaussianNB**:
   - `impl SparseModel for GaussianNB`:
     - `fit_sparse()`: compute per-class mean/variance from sparse rows
       - Mean: sum nnz values per column per class, divide by class count
       - Variance: second pass or online Welford on sparse rows
     - `predict_sparse()`: Gaussian log-likelihood — only non-zero features contribute differently from mean

   **CategoricalNB**:
   - `impl SparseModel for CategoricalNB`:
     - `fit_sparse()`: count category occurrences from sparse integer features
     - `predict_sparse()`: categorical log-likelihood from sparse feature indices

2. **File**: `ferroml-core/src/testing/sparse_models.rs` (NEW or append to sparse_tests.rs)
   - Per-variant: train on dense, train on sparse, compare predictions
   - Text classification scenario: 1000 docs, 5000 vocab, 99% sparse
   - Partial fit sparse (MultinomialNB, BernoulliNB)

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features sparse -- sparse naive_bayes` passes
- [ ] Dense vs sparse predictions identical (within f64 epsilon)
- [ ] ~16 new tests (4 variants × 4: basic, text-data, partial_fit, edge cases)

---

### Phase Q.7: Sparse Linear Models

**Overview**: Wire sparse matrix-vector products into LinearSVC, LinearSVR, LogisticRegression, and Ridge for O(nnz) prediction and training.

**Changes Required**:

1. **File**: `ferroml-core/src/models/svm.rs`

   **LinearSVC** (~line 1802):
   - `impl SparseModel for LinearSVC`:
     - `fit_sparse()`: SGD/coordinate descent operating on sparse rows
       - Gradient: `∇ = -y_i * x_i` when hinge loss active — only touch nnz indices
     - `predict_sparse()`: `CsrMatrix::dot(weights)` + intercept → argmax
       - Uses existing `CsrMatrix::dot(&Array1<f64>)` from sparse.rs:254

   **LinearSVR** (~line 2362):
   - `impl SparseModel for LinearSVR`:
     - `fit_sparse()`: same SGD pattern with epsilon-insensitive loss
     - `predict_sparse()`: sparse dot + intercept

2. **File**: `ferroml-core/src/models/logistic.rs`
   - `impl SparseModel for LogisticRegression`:
     - `predict_sparse()`: sparse X @ w + b → sigmoid → threshold
     - `fit_sparse()`: IRLS with sparse X'WX computation
       - Key optimization: X'WX where X is sparse — iterate only nnz per row
       - X'Wz: sparse transpose-vector product
     - Fisher information matrix: sparse row outer products

3. **File**: `ferroml-core/src/models/regularized.rs` (Ridge)
   - `impl SparseModel for Ridge`:
     - `fit_sparse()`: solve (X'X + λI)β = X'y with sparse X'X
     - `predict_sparse()`: sparse dot product

4. **File**: `ferroml-core/src/sparse.rs`
   - Add `CsrMatrix::transpose_dot(y: &Array1<f64>) -> Array1<f64>` — X' @ y efficiently
   - Add `CsrMatrix::gram_matrix() -> Array2<f64>` — X' @ X for normal equations
   - Add `CsrMatrix::weighted_gram(w: &Array1<f64>) -> Array2<f64>` — X' @ diag(w) @ X

5. **File**: `ferroml-core/src/testing/sparse_models.rs`
   - Linear model sparse equivalence tests
   - High-dimensional sparse data (10K features, 100 nnz per row)
   - Regularization interaction with sparse

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features sparse -- sparse linear svm logistic ridge` passes
- [ ] Dense vs sparse predictions match within tolerance
- [ ] ~16 new tests (LinearSVC: 4, LinearSVR: 4, LogReg: 4, Ridge: 4)

---

### Phase Q.8: Sparse Preprocessing & Python Bindings

**Overview**: Add TfidfTransformer, sparse-aware preprocessing, Python CsrMatrix bindings, and end-to-end NLP pipeline test.

**Changes Required**:

1. **File**: `ferroml-core/src/preprocessing/tfidf.rs` (NEW)
   - `TfidfTransformer` struct:
     ```rust
     pub struct TfidfTransformer {
         norm: TfidfNorm,          // L1, L2, None
         use_idf: bool,
         smooth_idf: bool,         // Add 1 to df to prevent zero division
         sublinear_tf: bool,       // Replace tf with 1 + log(tf)
         idf_: Option<Array1<f64>>, // Learned IDF weights
     }
     ```
   - `impl Transformer for TfidfTransformer`:
     - `fit()`: compute IDF from dense input (document frequencies)
     - `transform()`: TF-IDF weighting on dense input
   - `fit_sparse()` / `transform_sparse()` methods:
     - IDF computation: `sparse_column_nnz()` → `log((1+n)/(1+df)) + 1`
     - TF-IDF: multiply each nnz value by IDF weight (stays sparse)
     - Optional L2 normalization via `sparse_normalize_rows_l2()`
   - `CountVectorizer` is out of scope (requires tokenization) — users pass pre-tokenized sparse matrices

2. **File**: `ferroml-core/src/preprocessing/scalers.rs`
   - Add `fit_sparse()` / `transform_sparse()` to `StandardScaler`:
     - Mean: `sparse_column_means()`
     - Variance: two-pass on sparse columns (account for implicit zeros)
     - Transform: subtract mean (densifies!), scale by std — or skip mean for sparse (common pattern)
   - Add `transform_sparse()` to `MaxAbsScaler` (scales by max absolute, preserves sparsity)
   - Add `transform_sparse()` to `Normalizer` (row-wise L1/L2, already have `sparse_normalize_rows_l2`)

3. **File**: `ferroml-core/src/preprocessing/mod.rs`
   - Register `tfidf` module, re-export `TfidfTransformer`

4. **File**: `ferroml-python/src/sparse_utils.rs`
   - Add `py_csr_to_ferro(py_sparse: &Bound<'_, PyAny>) -> PyResult<CsrMatrix>`:
     - Extract `.data`, `.indices`, `.indptr` arrays from scipy.sparse.csr_matrix
     - Construct `CsrMatrix::new(shape, indptr, indices, data)`
   - Add `ferro_csr_to_py(matrix: &CsrMatrix, py: Python<'_>) -> PyResult<PyObject>`:
     - Construct scipy.sparse.csr_matrix from components
   - Update model wrappers to accept sparse input and dispatch to `fit_sparse()` / `predict_sparse()`

5. **File**: `ferroml-python/src/preprocessing.rs`
   - Add `PyTfidfTransformer` class with `fit()`, `transform()`, `fit_transform()`
   - Accept both dense numpy and scipy.sparse inputs

6. **File**: `ferroml-python/python/ferroml/preprocessing/__init__.py`
   - Re-export `TfidfTransformer`

7. **File**: `ferroml-python/tests/test_sparse_pipeline.py` (NEW)
   - End-to-end: generate sparse text data → TfidfTransformer → MultinomialNB → predict
   - Verify scipy.sparse → FerroML roundtrip preserves data
   - Compare TF-IDF output vs sklearn.feature_extraction.text.TfidfTransformer
   - Benchmark: sparse pipeline vs dense pipeline on 10K-feature data

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features sparse -- tfidf sparse scaler` passes
- [ ] Automated: `cd ferroml-python && python -m pytest tests/test_sparse_pipeline.py -v` passes
- [ ] TF-IDF output matches sklearn within tolerance
- [ ] ~20 new tests (TfidfTransformer: 8, sparse scalers: 4, Python roundtrip: 4, pipeline: 4)

---

## Dependencies

| Phase | Depends On | Reason |
|-------|-----------|--------|
| Q.1 | — | Self-contained shader work |
| Q.2 | — | Uses existing GpuBackend trait (2 methods suffice) |
| Q.3 | Q.1 | Needs new shaders (relu, sigmoid, softmax, bias_add) |
| Q.4 | Q.1 | Needs all shaders to benchmark/dispatch |
| Q.5 | — | Uses existing sparse module |
| Q.6 | Q.5 | Uses SparseModel trait pattern from Q.5 |
| Q.7 | Q.5 | Uses SparseModel trait + new sparse ops |
| Q.8 | Q.5, Q.6 or Q.7 | Needs at least one SparseModel impl for pipeline test |

**Recommended execution order**: Q.1 → Q.2 → Q.3 → Q.4 → Q.5 → Q.6 → Q.7 → Q.8

GPU and sparse tracks are independent — could interleave (Q.1, Q.5, Q.2, Q.6, ...) if preferred.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| No GPU hardware in CI | GPU-specific tests can't run | All GPU tests use MockGpuBackend; real GPU tests behind `#[ignore]` |
| f32 precision in GPU shaders | Numerical divergence from CPU | Tolerance-based assertions (1e-5 relative); document precision limitations |
| sprs API changes | Sparse module breaks | Pin sprs version in Cargo.toml; wrapper types insulate |
| Sparse StandardScaler mean-centering densifies data | Defeats purpose of sparse | Default to `with_mean=false` for sparse (sklearn convention) |
| TfidfTransformer without CountVectorizer | Users need pre-tokenized input | Document clearly; CountVectorizer is a separate future feature |
| SparseModel trait proliferation | API surface grows | Keep trait minimal (fit_sparse + predict_sparse); optional predict_proba_sparse later |
| GpuBackend trait growing too large | Hard to mock, implement | Group related ops; consider sub-traits if > 15 methods |

## Test Count Summary

| Phase | New Tests | Running Total |
|-------|----------|---------------|
| Q.1 | ~20 | 20 |
| Q.2 | ~16 | 36 |
| Q.3 | ~12 | 48 |
| Q.4 | ~15 | 63 |
| Q.5 | ~20 | 83 |
| Q.6 | ~16 | 99 |
| Q.7 | ~16 | 115 |
| Q.8 | ~20 | 135 |

**Grand total: ~135 new tests**
