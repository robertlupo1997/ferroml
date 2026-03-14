---
date: 2026-03-13T03:02:11Z
researcher: Claude
git_commit: 9368660
git_branch: master
repository: ferroml
topic: Plan S - Pipeline + TextTransformer Integration (S.4-S.6 remaining)
tags: [plan-s, text-pipeline, tfidf-vectorizer, sparse-transformer, pipeline-trait-impls]
status: in-progress
---

# Handoff: Plan S - Text Pipeline Integration (Phases S.4-S.6 Remaining)

## Task Status

### Completed Phases
- **S.1** (commit `d0d0ebc`): SparseTransformer trait + TfidfTransformer integration — 20 new tests
- **S.2** (commit `9368660`): TfidfVectorizer (CountVectorizer + TfidfTransformer combined) — 18 tests
- **S.3** (commit `59a3212`): TextPipeline Rust core + 3 pipeline traits — 41 tests

### Remaining Phases
- **S.4**: Pipeline trait impls for all model/transformer types (~20 tests)
- **S.5**: Python bindings for TfidfVectorizer + TextPipeline (~40 tests)
- **S.6**: Integration tests + correctness fixtures (~15 tests)

## Critical References

1. **Plan file**: `thoughts/shared/plans/2026-03-11_plan-S-pipeline-text-transformer-integration.md` — Full 6-phase plan
2. **SparseTransformer trait**: `ferroml-core/src/preprocessing/mod.rs:232-258`
3. **TfidfTransformer**: `ferroml-core/src/preprocessing/tfidf.rs` — has Transformer, PipelineTransformer, SparseTransformer impls + `transform_sparse_native()`
4. **TfidfVectorizer**: `ferroml-core/src/preprocessing/tfidf_vectorizer.rs` — wraps CountVectorizer + TfidfTransformer, implements TextTransformer
5. **TextPipeline**: `ferroml-core/src/pipeline/text_pipeline.rs` — TextPipeline struct + PipelineTextTransformer, PipelineSparseTransformer, PipelineSparseModel traits
6. **Pipeline re-exports**: `ferroml-core/src/pipeline/mod.rs:54-61` — re-exports TextPipeline, all 3 traits, TextPipelineStep
7. **Existing SparseModel impls**: 11 models across naive_bayes.rs, knn.rs, svm.rs, logistic.rs, regularized.rs
8. **Python Pipeline**: `ferroml-python/src/pipeline.rs` — duck-typing approach with PyObject
9. **Python preprocessing**: `ferroml-python/src/preprocessing.rs` — PyCountVectorizer, PyTfidfTransformer
10. **Sparse utils**: `ferroml-python/src/sparse_utils.rs` — py_csr_to_ferro, ferro_csr_to_py

## Phase S.4 Details (Next Up)

### What to implement

**PipelineTextTransformer** (2 files):
- `ferroml-core/src/preprocessing/count_vectorizer.rs` — `impl PipelineTextTransformer for CountVectorizer`
  - `clone_boxed()`, `name()` -> "CountVectorizer"
  - `set_param()` for: max_features, binary, lowercase (use `value.as_usize()`, `value.as_bool()`)
  - `n_features_out()` -> `self.vocabulary().map(|v| v.len())`
  - Gate behind `#[cfg(feature = "sparse")]`

- `ferroml-core/src/preprocessing/tfidf_vectorizer.rs` — `impl PipelineTextTransformer for TfidfVectorizer`
  - Same pattern, support all combined CV+TFIDF params
  - Already behind sparse feature gate (whole module is)

**PipelineSparseTransformer** (1 file):
- `ferroml-core/src/preprocessing/tfidf.rs` — `impl PipelineSparseTransformer for TfidfTransformer`
  - Reuse existing PipelineTransformer::set_param logic (norm, use_idf, smooth_idf, sublinear_tf)
  - Gate behind `#[cfg(feature = "sparse")]`

**PipelineSparseModel** (5 model files, 11 models):

All follow this pattern:
```rust
#[cfg(feature = "sparse")]
impl crate::pipeline::PipelineSparseModel for ModelName {
    fn fit_sparse(&mut self, x: &crate::sparse::CsrMatrix, y: &ndarray::Array1<f64>) -> crate::Result<()> {
        crate::models::traits::SparseModel::fit_sparse(self, x, y)
    }
    fn predict_sparse(&self, x: &crate::sparse::CsrMatrix) -> crate::Result<ndarray::Array1<f64>> {
        crate::models::traits::SparseModel::predict_sparse(self, x)
    }
    fn search_space(&self) -> crate::hpo::SearchSpace { crate::models::Model::search_space(self) }
    fn clone_boxed(&self) -> Box<dyn crate::pipeline::PipelineSparseModel> { Box::new(self.clone()) }
    fn set_param(&mut self, name: &str, value: &crate::hpo::ParameterValue) -> crate::Result<()> { ... }
    fn name(&self) -> &str { "ModelName" }
    fn is_fitted(&self) -> bool { crate::models::Model::is_fitted(self) }
}
```

Models needing PipelineSparseModel:
| File | Model | set_param keys |
|------|-------|---------------|
| `naive_bayes.rs` | MultinomialNB | "alpha" (f64) |
| `naive_bayes.rs` | BernoulliNB | "alpha" (f64), "binarize" (f64) |
| `naive_bayes.rs` | CategoricalNB | "alpha" (f64) |
| `naive_bayes.rs` | GaussianNB | (none) |
| `logistic.rs` | LogisticRegression | "fit_intercept" (bool), "l2_penalty" (f64), "max_iter" (i64->usize) |
| `svm.rs` | LinearSVC | "C"/"c" (f64), "max_iter" (i64->usize) |
| `svm.rs` | LinearSVR | "C"/"c" (f64), "epsilon" (f64), "max_iter" (i64->usize) |
| `knn.rs` | KNeighborsClassifier | "n_neighbors" (i64->usize) |
| `knn.rs` | KNeighborsRegressor | "n_neighbors" (i64->usize) |
| `knn.rs` | NearestCentroid | (none) |
| `regularized.rs` | RidgeRegression | "alpha" (f64) |

### is_fitted() dispatch

All models implement `Model` trait which has `is_fitted()`. Use `crate::models::Model::is_fitted(self)`.

### Tests for S.4

Create `ferroml-core/src/testing/text_pipeline_trait_impls.rs` and register in `testing/mod.rs`.

Test each model type in a TextPipeline with CountVectorizer (smoke test: fit + predict). Plus set_param, clone_boxed, name tests for transformers.

## Phase S.5 Details (Python Bindings)

**PyTfidfVectorizer** in `ferroml-python/src/preprocessing.rs`:
- `#[pyclass(name = "TfidfVectorizer", module = "ferroml.preprocessing")]`
- `fit(documents: list[str])` -> returns `PyRefMut<Self>` for chaining
- `transform(documents: list[str])` -> returns scipy.sparse.csr_matrix (use `ferro_csr_to_py`)
- `fit_transform(documents: list[str])` -> returns scipy.sparse.csr_matrix
- `transform_dense(documents: list[str])` -> returns numpy array
- Properties: `vocabulary_`, `idf_`, `get_feature_names_out()`
- Register in `register_preprocessing_module()`

**PyTextPipeline** in `ferroml-python/src/pipeline.rs`:
- Duck-typing approach (like existing PyPipeline)
- `fit(documents: list[str], y: np.ndarray)` -> `PyRefMut<Self>`
- `predict(documents: list[str])` -> np.ndarray
- `transform(documents: list[str])` -> scipy.sparse.csr_matrix
- Steps detected by checking for `fit_text`/`transform_text` vs `fit`/`transform` vs `fit`/`predict`
- Register in `register_pipeline_module()`

**__init__.py exports**:
- `ferroml-python/python/ferroml/preprocessing/__init__.py` — add TfidfVectorizer
- `ferroml-python/python/ferroml/pipeline/__init__.py` — add TextPipeline

**Python tests** (~40):
- `ferroml-python/tests/test_tfidf_vectorizer.py` (~20 tests)
- `ferroml-python/tests/test_text_pipeline.py` (~20 tests)

## Phase S.6 Details (Integration Tests)

- `ferroml-core/src/testing/text_pipeline_integration.rs` — end-to-end with real models
- `scripts/generate_text_pipeline_fixtures.py` — sklearn reference data
- `ferroml-core/src/testing/text_pipeline_correctness.rs` — correctness vs sklearn
- Register modules in `testing/mod.rs`

## Key Architecture Decisions

- **TextPipeline is separate from Pipeline** — doesn't modify existing Pipeline at all
- **Python TextPipeline uses duck-typing** — not Rust TextPipeline wrapper, for sklearn compatibility
- **SparseTransformer returns CsrMatrix** — true sparse-in/sparse-out, no densification
- **TfidfVectorizer wraps CountVectorizer + TfidfTransformer** — uses fit_sparse + transform_sparse_native

## Existing Patterns to Follow

### PipelineModel impl pattern (from logistic.rs:1137):
```rust
impl PipelineModel for LogisticRegression {
    fn clone_boxed(&self) -> Box<dyn PipelineModel> { Box::new(self.clone()) }
    fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()> { ... }
    fn name(&self) -> &str { "LogisticRegression" }
}
```

### Python binding pattern (from preprocessing.rs PyCountVectorizer):
- `#[pyclass(name = "CountVectorizer", module = "ferroml.preprocessing")]`
- `#[new]` with `#[pyo3(signature = (...))]` for default params
- Returns `PyRefMut<'py, Self>` from fit methods for chaining
- Registers class in `register_preprocessing_module()`

## Verification Commands

```bash
# Compile with sparse feature
cargo build --features sparse -p ferroml-core

# Run S.1-S.3 tests
cargo test --features sparse -p ferroml-core -- tfidf
cargo test --features sparse -p ferroml-core -- pipeline::text_pipeline

# Run all Rust tests
cargo test --workspace

# Build Python bindings
source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml

# Run Python tests
python -m pytest ferroml-python/tests/ -x -q

# Format (required by pre-commit hook)
cargo fmt --all
```

## Parallel Agent Strategy

S.4 should be done first (sequential, touches 7 files). Then S.5 (Python bindings, touches different files). Then S.6 (tests).

S.4 could use a single agent since it's mostly boilerplate impls.
S.5 could use 2 agents if desired (one for PyTfidfVectorizer, one for PyTextPipeline), but file conflicts in pipeline.rs and preprocessing.rs make sequential safer.

## Other Notes

- The user calls this "Plan U" in conversation, but the plan file says "Plan S"
- Pre-existing uncommitted changes from Plan S (Voting/Stacking) exist in CHANGELOG.md, ensemble __init__.py, and ensemble.rs — these should be committed separately or together with final Plan S work
- `cargo fmt --all` is required by pre-commit hook — always run before committing
- Total expected tests when complete: ~143 across all 6 phases (currently ~79 done: 20 + 18 + 41)
- Build command for Python: `source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml`
