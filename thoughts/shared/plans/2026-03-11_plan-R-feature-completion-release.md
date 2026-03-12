# Plan R: Feature Completion & v0.2.0 Release

## Overview

Close remaining feature gaps and prepare for v0.2.0 release. Six independent tracks that can all execute in parallel.

## Current State

Plans G-Q complete. 50+ models, 3,854 Rust tests, 1,507+ Python tests, 9 CI workflows, comprehensive docs. The following gaps remain:

1. **No CountVectorizer** — TfidfTransformer exists but users must provide pre-tokenized count matrices
2. **Python sparse boundary converts to dense** — scipy.sparse → dense → fit, losing O(nnz) benefit
3. **No Gaussian Processes** — GP-based Bayesian optimization exists in HPO but no GP model
4. **No native MultiOutput wrapper** — pattern-based workaround only
5. **CHANGELOG/docs don't cover Plans P-Q** — last entry is Plan O
6. **Repo URL placeholder** — `github.com/user/ferroml` in workspace Cargo.toml

## Desired End State

- CountVectorizer → TfidfTransformer → MultinomialNB pipeline works end-to-end
- Python sparse data flows through to Rust SparseModel without densification
- GaussianProcessRegressor + GaussianProcessClassifier with RBF/Matern kernels
- MultiOutputRegressor/MultiOutputClassifier wrappers in Python
- All docs, CHANGELOG, README updated for v0.2.0
- Published benchmark comparison vs sklearn

---

## Implementation Phases

### Phase R.1: CountVectorizer

**Overview**: Text tokenization + vocabulary management, completing the NLP pipeline.

**Changes Required**:

1. **File**: `ferroml-core/src/preprocessing/count_vectorizer.rs` (NEW, ~500 lines)
   - `CountVectorizer` struct:
     ```rust
     pub struct CountVectorizer {
         max_features: Option<usize>,
         min_df: MinDF,           // MinCount(usize) | MinFraction(f64)
         max_df: MaxDF,           // MaxCount(usize) | MaxFraction(f64)
         ngram_range: (usize, usize),  // (1, 1) for unigrams, (1, 2) for uni+bigrams
         binary: bool,            // If true, all non-zero counts are set to 1
         lowercase: bool,         // Default: true
         token_pattern: TokenPattern,  // Word (alphanumeric 2+ chars) or Custom(regex)
         vocabulary_: Option<HashMap<String, usize>>,  // Learned vocab
         stop_words: Option<HashSet<String>>,
     }
     ```
   - `impl Transformer for CountVectorizer`:
     - `fit()` takes `&[String]` (documents) — not Array2, since input is text
     - `transform()` returns `CsrMatrix` (sparse count matrix)
     - Custom trait `TextTransformer` since Transformer expects Array2:
       ```rust
       pub trait TextTransformer {
           fn fit_text(&mut self, documents: &[String]) -> Result<()>;
           fn transform_text(&self, documents: &[String]) -> Result<CsrMatrix>;
           fn fit_transform_text(&mut self, documents: &[String]) -> Result<CsrMatrix>;
       }
       ```
   - Tokenization: split on non-alphanumeric, filter by token_pattern
   - N-gram generation: sliding window over tokens
   - Vocabulary building: HashMap<String, usize> sorted by document frequency
   - min_df/max_df filtering: remove too-rare or too-common terms
   - max_features: keep top-N by document frequency

2. **File**: `ferroml-core/src/preprocessing/mod.rs`
   - Add `pub mod count_vectorizer;`
   - Re-export `CountVectorizer`, `TextTransformer`

3. **File**: `ferroml-python/src/preprocessing.rs`
   - `PyCountVectorizer` class:
     - `fit(documents: Vec<String>)` — Python list of strings
     - `transform(documents: Vec<String>) -> numpy.ndarray` — returns dense (or scipy.sparse if R.2 done)
     - `fit_transform(documents: Vec<String>)`
     - `vocabulary_` property → dict
     - `get_feature_names_out() -> list[str]`

4. **File**: `ferroml-python/python/ferroml/preprocessing/__init__.py`
   - Re-export `CountVectorizer`

5. **File**: `ferroml-core/src/testing/text_pipeline.rs` (NEW)
   - Basic tokenization tests
   - N-gram tests (unigrams, bigrams, trigrams)
   - min_df/max_df filtering tests
   - max_features cap tests
   - Stop words tests
   - Binary mode tests
   - Lowercase tests
   - CountVectorizer → TfidfTransformer pipeline test
   - Dense vs sparse output equivalence

6. **File**: `ferroml-python/tests/test_count_vectorizer.py` (NEW)
   - Basic usage, sklearn comparison for simple corpus
   - N-gram range tests
   - Vocabulary properties
   - Pipeline: CountVectorizer → TfidfTransformer → MultinomialNB

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core --features sparse -- count_vectorizer text_pipeline` passes
- [ ] Automated: `cd ferroml-python && python -m pytest tests/test_count_vectorizer.py -v` passes
- [ ] Pipeline: CountVectorizer → TfidfTransformer → MultinomialNB classifies 20newsgroups-like data
- [ ] ~25 new tests (15 Rust + 10 Python)

---

### Phase R.2: Python Sparse Round-Trip

**Overview**: Enable scipy.sparse matrices to flow through to Rust `CsrMatrix` without densification, so `fit_sparse()` / `predict_sparse()` get true O(nnz) performance from Python.

**Changes Required**:

1. **File**: `ferroml-python/src/sparse_utils.rs` (MODIFY)
   - Add `py_csr_to_ferro()`: extract `.data`, `.indices`, `.indptr` from scipy CSR → construct `CsrMatrix::new()`
     ```rust
     pub fn py_csr_to_ferro(py_sparse: &Bound<'_, PyAny>) -> PyResult<CsrMatrix> {
         let data: Vec<f64> = py_sparse.getattr("data")?.extract()?;
         let indices: Vec<usize> = py_sparse.getattr("indices")?.extract()?;
         let indptr: Vec<usize> = py_sparse.getattr("indptr")?.extract()?;
         let shape: (usize, usize) = py_sparse.getattr("shape")?.extract()?;
         CsrMatrix::new(shape, indptr, indices, data).map_err(|e| ...)
     }
     ```
   - Add `ferro_csr_to_py()`: construct scipy.sparse.csr_matrix from components
     ```rust
     pub fn ferro_csr_to_py(matrix: &CsrMatrix, py: Python<'_>) -> PyResult<PyObject> {
         let scipy_sparse = py.import("scipy.sparse")?;
         let data = matrix.data().to_pyarray(py);
         let indices = matrix.indices().to_pyarray(py);
         let indptr = matrix.indptr().to_pyarray(py);
         let shape = matrix.shape().to_object(py);
         scipy_sparse.call_method1("csr_matrix", ((data, indices, indptr), shape))
     }
     ```
   - Add `is_sparse_matrix()` helper: detect scipy.sparse types

2. **File**: `ferroml-python/src/linear.rs` (MODIFY)
   - Update `fit()` on PyLinearRegression, PyLogisticRegression, PyRidge:
     - Auto-detect sparse input: `if is_sparse_matrix(x) { fit_sparse(...) } else { fit_dense(...) }`
     - Or add explicit `fit_sparse()` Python method that calls `py_csr_to_ferro()` → `model.fit_sparse()`
   - Same for `predict()` → `predict_sparse()` dispatch

3. **File**: `ferroml-python/src/naive_bayes.rs` (MODIFY)
   - Same sparse dispatch pattern for MultinomialNB, BernoulliNB, GaussianNB, CategoricalNB

4. **File**: `ferroml-python/src/svm.rs` (MODIFY)
   - Same for LinearSVC, LinearSVR

5. **File**: `ferroml-python/src/preprocessing.rs` (MODIFY)
   - TfidfTransformer: accept scipy.sparse input, return scipy.sparse output via `ferro_csr_to_py()`
   - StandardScaler/MaxAbsScaler/Normalizer: sparse dispatch

6. **File**: `ferroml-python/tests/test_sparse_roundtrip.py` (NEW)
   - scipy.sparse → CsrMatrix → scipy.sparse preserves data, indices, indptr
   - End-to-end: scipy.sparse input → fit_sparse → predict_sparse → numpy output
   - Pipeline: scipy.sparse → TfidfTransformer.fit_sparse → MultinomialNB.fit_sparse → predict
   - Performance comparison: sparse vs dense path on 10K-feature data (timing)
   - Edge cases: empty rows, single-element sparse, CSC auto-conversion

**Success Criteria**:
- [ ] Automated: `cd ferroml-python && python -m pytest tests/test_sparse_roundtrip.py -v` passes
- [ ] scipy.sparse CSR survives round-trip with exact data preservation
- [ ] fit(scipy_sparse) dispatches to fit_sparse() without densification
- [ ] ~15 new tests

---

### Phase R.3: Gaussian Processes

**Overview**: Add GaussianProcessRegressor and GaussianProcessClassifier with RBF and Matern kernels.

**Changes Required**:

1. **File**: `ferroml-core/src/models/gaussian_process.rs` (NEW, ~800 lines)
   - **Kernel trait**:
     ```rust
     pub trait Kernel: Send + Sync {
         fn compute(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64>;
         fn diagonal(&self, x: &Array2<f64>) -> Array1<f64>;
         fn n_params(&self) -> usize;
         fn get_params(&self) -> Vec<f64>;
         fn set_params(&mut self, params: &[f64]);
         fn clone_box(&self) -> Box<dyn Kernel>;
     }
     ```
   - **Kernels**:
     - `RBF { length_scale: f64 }` — K(x, x') = exp(-||x-x'||² / 2l²)
     - `Matern { length_scale: f64, nu: f64 }` — nu = 0.5, 1.5, 2.5
     - `ConstantKernel { constant: f64 }` — K = c
     - `WhiteKernel { noise_level: f64 }` — K = σ²I (diagonal)
     - `ProductKernel`, `SumKernel` — kernel algebra
   - **GaussianProcessRegressor**:
     ```rust
     pub struct GaussianProcessRegressor {
         kernel: Box<dyn Kernel>,
         alpha: f64,              // Noise level (regularization on diagonal)
         optimizer: GPOptimizer,   // LML optimization method
         n_restarts: usize,       // Random restarts for optimizer
         normalize_y: bool,
         // Fitted state
         x_train_: Option<Array2<f64>>,
         y_train_: Option<Array1<f64>>,
         alpha_: Option<Array1<f64>>,    // K^{-1} y (precomputed)
         l_: Option<Array2<f64>>,        // Cholesky factor of K
         log_marginal_likelihood_: Option<f64>,
     }
     ```
     - `fit()`: compute K(X,X) + αI, Cholesky decompose, solve for α_ = L\(L\y)
     - `predict()`: K(X*, X) @ α_ for mean; optionally K(X*, X*) - v^T v for variance
     - `predict_with_std()`: return (mean, std) tuple
     - `log_marginal_likelihood()`: -½ y^T α - ½ log|K| - n/2 log(2π)
     - `optimize_kernel_params()`: maximize LML over kernel hyperparameters (L-BFGS or grid)
   - **GaussianProcessClassifier** (Laplace approximation):
     - Binary: f ~ GP, p(y|f) = sigmoid(f), Laplace approx for posterior
     - `fit()`: Newton iterations to find f_hat (mode of posterior)
     - `predict()`: Bernoulli threshold
     - `predict_proba()`: probit approximation of predictive posterior
   - `impl Model for GaussianProcessRegressor` + `impl Model for GaussianProcessClassifier`
   - `search_space()` for HPO integration

2. **File**: `ferroml-core/src/models/mod.rs`
   - Add `pub mod gaussian_process;`
   - Re-export types

3. **File**: `ferroml-core/src/testing/gaussian_process.rs` (NEW)
   - RBF kernel computation vs manual calculation
   - Matern kernel for nu=0.5 (exponential), 1.5, 2.5
   - GPR fit/predict on simple function (sin, quadratic)
   - GPR uncertainty: std increases away from training data
   - GPR normalize_y test
   - Log-marginal likelihood computation
   - Kernel hyperparameter optimization
   - GPC on linearly separable data
   - GPC predict_proba range [0, 1]
   - Kernel algebra: Sum, Product
   - Edge cases: single point, duplicate points, high dimensions

4. **File**: `ferroml-python/src/gaussian_process.rs` (NEW)
   - PyGaussianProcessRegressor, PyGaussianProcessClassifier
   - PyRBF, PyMatern, PyConstantKernel, PyWhiteKernel
   - predict_with_std() returning (ndarray, ndarray)

5. **File**: `ferroml-python/python/ferroml/gaussian_process/__init__.py` (NEW)
   - Re-export all GP classes

6. **File**: `ferroml-python/tests/test_gaussian_process.py` (NEW)
   - GPR basic regression, predict_with_std
   - GPC basic classification, predict_proba
   - sklearn comparison on simple functions
   - Kernel hyperparameter learning

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core -- gaussian_process` passes
- [ ] Automated: `cd ferroml-python && python -m pytest tests/test_gaussian_process.py -v` passes
- [ ] GPR predictions match sklearn within 1e-4 on sin(x) regression
- [ ] ~35 new tests (25 Rust + 10 Python)

---

### Phase R.4: MultiOutput Wrappers

**Overview**: Native MultiOutputRegressor and MultiOutputClassifier wrappers, promoting the existing pattern-based approach to first-class models.

**Changes Required**:

1. **File**: `ferroml-core/src/models/multioutput.rs` (NEW, ~300 lines)
   - `MultiOutputRegressor<M: Model + Clone>`:
     ```rust
     pub struct MultiOutputRegressor<M: Model + Clone> {
         base_estimator: M,
         estimators_: Option<Vec<M>>,   // One per target column
         n_outputs_: Option<usize>,
     }
     ```
     - `fit(x: &Array2<f64>, y: &Array2<f64>)` — note: y is 2D (n_samples × n_outputs)
     - For each column of y: clone base_estimator, fit on (x, y_col)
     - `predict(x: &Array2<f64>) -> Result<Array2<f64>>` — stack predictions column-wise
     - Parallel fitting via rayon
   - `MultiOutputClassifier<M: Model + Clone>` — same pattern
     - `predict_proba()` if base model supports it

2. **File**: `ferroml-core/src/models/mod.rs`
   - Add `pub mod multioutput;`

3. **File**: `ferroml-python/src/multioutput.rs` (NEW)
   - `PyMultiOutputRegressor`:
     - Takes estimator name string + params dict (factory pattern like Bagging)
     - `fit(x: ndarray, y: ndarray)` — y must be 2D
     - `predict(x: ndarray) -> ndarray` — returns 2D
   - `PyMultiOutputClassifier`: same pattern

4. **File**: `ferroml-python/python/ferroml/multioutput/__init__.py` (NEW)
   - Re-export wrappers

5. **File**: `ferroml-core/src/testing/multioutput_models.rs` (NEW)
   - LinearRegression multi-output
   - DecisionTree multi-output
   - KNN multi-output
   - Ridge multi-output
   - Shape validation (1D y rejected, 2D accepted)
   - Single output degeneracy test
   - Per-output metrics (MSE, R²)

6. **File**: `ferroml-python/tests/test_multioutput.py` (NEW)
   - Basic multi-output regression
   - Multi-output classification
   - sklearn comparison

**Success Criteria**:
- [ ] Automated: `cargo test -p ferroml-core -- multioutput` passes
- [ ] Automated: `cd ferroml-python && python -m pytest tests/test_multioutput.py -v` passes
- [ ] ~20 new tests (12 Rust + 8 Python)

---

### Phase R.5: Documentation & Release Prep

**Overview**: Update all docs for Plans P-Q, fix metadata, prepare for v0.2.0 tag.

**Changes Required**:

1. **File**: `Cargo.toml` (workspace root)
   - Fix repository URL: `github.com/user/ferroml` → `github.com/robertlupo1997/ferroml`
   - Bump version to `0.2.0`

2. **File**: `ferroml-python/pyproject.toml`
   - Bump version to `0.2.0`

3. **File**: `CHANGELOG.md`
   - Add Plan P entry (SVM polish: decision_function, loss params, class weights)
   - Add Plan Q entry (GPU shader expansion, native sparse O(nnz) for 12 models)
   - Add Plan R entries as they complete
   - Create `## [0.2.0] - 2026-03-11` section

4. **File**: `README.md`
   - Update test counts (3,854 Rust + 1,507 Python)
   - Add GPU + Sparse sections
   - Add CountVectorizer → TfidfTransformer pipeline example (after R.1)
   - Update "What's Implemented" section with GP, MultiOutput, CountVectorizer

5. **File**: `docs/user-guide.md`
   - Add GPU usage section (with_gpu() builder pattern)
   - Add sparse data section (CsrMatrix, SparseModel trait)
   - Add text/NLP pipeline tutorial (CountVectorizer → TF-IDF → NB)
   - Add Gaussian Process section

6. **File**: `docs/accuracy-report.md`
   - Update with Plans P-Q test counts
   - Add sparse equivalence test results
   - Add GP accuracy results (after R.3)

**Success Criteria**:
- [ ] Manual: README accurately reflects current capabilities
- [ ] Manual: CHANGELOG covers all plans through R
- [ ] Manual: Repo URL is correct in all Cargo.toml/pyproject.toml
- [ ] Automated: `cargo doc --all-features --no-deps` builds without warnings
- [ ] Manual: Version is 0.2.0 everywhere

---

### Phase R.6: Published Benchmark Suite

**Overview**: Create a reproducible benchmark comparing FerroML vs sklearn on standard datasets and publish results.

**Changes Required**:

1. **File**: `scripts/benchmark_vs_sklearn.py` (NEW, ~300 lines)
   - Benchmark harness:
     ```python
     BENCHMARKS = [
         ("LinearRegression", "boston-like", [100, 1000, 10000]),
         ("RandomForest", "classification", [1000, 5000, 10000]),
         ("GradientBoosting", "regression", [1000, 5000]),
         ("KNN", "classification", [1000, 5000, 10000]),
         ("MultinomialNB", "text-sparse", [1000, 5000, 20000]),
         ("DBSCAN", "clustering", [500, 2000, 5000]),
         ("PCA", "decomposition", [1000, 5000]),
         ("t-SNE", "decomposition", [500, 2000]),
         ("StandardScaler", "preprocessing", [1000, 10000, 50000]),
         ("TF-IDF Pipeline", "text-nlp", [1000, 5000]),
     ]
     ```
   - For each: measure fit time, predict time, memory usage
   - Compare accuracy (predictions must match within tolerance)
   - Generate markdown report table
   - Generate JSON results for CI

2. **File**: `scripts/generate_benchmark_report.py` (NEW)
   - Read JSON results → generate `docs/benchmark-vs-sklearn.md`
   - ASCII bar charts for speedup ratios
   - Memory comparison table
   - Accuracy verification summary

3. **File**: `docs/benchmark-vs-sklearn.md` (NEW, generated)
   - Speedup table: FerroML vs sklearn fit/predict times
   - Memory usage comparison
   - Accuracy parity verification
   - Hardware/environment info

4. **File**: `.github/workflows/benchmarks.yml` (MODIFY)
   - Add sklearn comparison job (weekly, not on every PR)
   - Archive results as workflow artifact

**Success Criteria**:
- [ ] Automated: `python scripts/benchmark_vs_sklearn.py` completes without error
- [ ] Manual: Benchmark report shows FerroML competitive with sklearn
- [ ] Automated: All accuracy checks pass (predictions within tolerance)
- [ ] ~10 benchmark scenarios × 3 sizes = 30 data points

---

## Dependencies

| Phase | Depends On | Reason |
|-------|-----------|--------|
| R.1 | — | Self-contained text processing |
| R.2 | — | Self-contained Python binding work |
| R.3 | — | Self-contained model implementation |
| R.4 | — | Self-contained wrapper implementation |
| R.5 | R.1-R.4 (partial) | Needs to document new features |
| R.6 | R.1 (optional) | NLP pipeline benchmark needs CountVectorizer |

**All of R.1-R.4 can execute in parallel.** R.5 can start immediately (CHANGELOG, repo URL fix) and finalize after R.1-R.4. R.6 can start immediately for non-NLP benchmarks.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CountVectorizer regex dependency | Adds external crate | Use `regex` crate (already common in Rust ecosystem) |
| scipy.sparse dtype variations | int32 vs int64 indices break conversion | Handle both dtypes in py_csr_to_ferro() |
| GP Cholesky failure on ill-conditioned K | fit() crashes | Add jitter (α on diagonal), increase if Cholesky fails |
| GP scaling O(n³) | Slow for n > 5000 | Document limitation, suggest n < 5000 or sparse GP (future) |
| MultiOutput generic type erasure in Python | Can't use generics across PyO3 boundary | Factory pattern with string estimator name (like Bagging) |
| Benchmark results machine-dependent | Not reproducible | Document hardware, use relative speedup ratios |

## Test Count Summary

| Phase | New Tests | Running Total |
|-------|----------|---------------|
| R.1 | ~25 | 25 |
| R.2 | ~15 | 40 |
| R.3 | ~35 | 75 |
| R.4 | ~20 | 95 |
| R.5 | 0 (docs) | 95 |
| R.6 | ~30 (benchmarks) | 125 |

**Grand total: ~125 new tests + 30 benchmark data points**
