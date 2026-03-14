---
date: 2026-03-13T00:19:30Z
researcher: Claude
git_commit: 9a5f3c9ca8dbf68b28654dc85b0c43365a134e6c
git_branch: master
repository: ferroml
topic: Plan U - Pipeline + TextTransformer Integration
tags: [plan-u, text-pipeline, tfidf-vectorizer, sparse-transformer]
status: in-progress
---

# Handoff: Plan U - Pipeline + TextTransformer Integration (Phase S.1 partially done)

## Task Status

### Current Phase
Phase S.1: SparseTransformer Trait + TfidfTransformer Integration — **~80% complete**

### Progress

**Phase S.1** (in progress):
- [x] Added `SparseTransformer` trait to `ferroml-core/src/preprocessing/mod.rs`
- [x] Added `pub mod tfidf_vectorizer` declaration (file not yet created — expected, done in S.2)
- [x] Implemented `Transformer` trait for `TfidfTransformer` (delegates to existing fit/transform)
- [x] Implemented `PipelineTransformer` for `TfidfTransformer` (clone_boxed, set_param, name)
- [x] Added `transform_sparse_native()` — CsrMatrix->CsrMatrix path (O(nnz), no densification)
- [x] Implemented `SparseTransformer` trait for `TfidfTransformer`
- [x] Added getter methods: `norm()`, `use_idf()`, `smooth_idf()`, `sublinear_tf()`, `n_features()`
- [ ] **NOT DONE**: Add tests for S.1 (SparseTransformer, Transformer trait, PipelineTransformer)
- [ ] **NOT DONE**: Compile and verify `cargo test --features sparse -p ferroml-core -- tfidf`

**Phase S.2** (not started): TfidfVectorizer
**Phase S.3** (not started): TextPipeline (Rust core)
**Phase S.4** (not started): Pipeline trait impls for all types
**Phase S.5** (not started): Python bindings
**Phase S.6** (not started): Integration tests + correctness

## Plan File
`thoughts/shared/plans/2026-03-11_plan-S-pipeline-text-transformer-integration.md`

Despite the filename saying "plan-S", the user refers to this as "Plan U" in conversation.

## Critical References

1. **Plan file**: `thoughts/shared/plans/2026-03-11_plan-S-pipeline-text-transformer-integration.md` — Full 6-phase plan with ~143 tests
2. **SparseTransformer trait**: `ferroml-core/src/preprocessing/mod.rs:229-255` (new)
3. **TfidfTransformer with new impls**: `ferroml-core/src/preprocessing/tfidf.rs` (heavily modified)
4. **Existing Pipeline**: `ferroml-core/src/pipeline/mod.rs` — PipelineTransformer, PipelineModel, PipelineStep
5. **SparseModel trait**: `ferroml-core/src/models/traits.rs:61-67`
6. **Existing SparseModel impls**: 12 models across naive_bayes.rs, knn.rs, svm.rs, logistic.rs, regularized.rs
7. **Python Pipeline**: `ferroml-python/src/pipeline.rs` — duck-typing approach with PyObject
8. **Python preprocessing**: `ferroml-python/src/preprocessing.rs` — PyCountVectorizer, PyTfidfTransformer
9. **Sparse utils**: `ferroml-python/src/sparse_utils.rs` — py_csr_to_ferro, ferro_csr_to_py

## Recent Changes (this session, uncommitted)

Files modified:
- `ferroml-core/src/preprocessing/mod.rs:64-67` — Added `pub mod tfidf_vectorizer` (behind sparse feature gate)
- `ferroml-core/src/preprocessing/mod.rs:229-255` — Added `SparseTransformer` trait
- `ferroml-core/src/preprocessing/tfidf.rs:200-344` — Added Transformer impl, PipelineTransformer impl, getter methods
- `ferroml-core/src/preprocessing/tfidf.rs:347-420` — Added `transform_sparse_native()` (CsrMatrix->CsrMatrix)
- `ferroml-core/src/preprocessing/tfidf.rs:453-470` — Added `SparseTransformer` impl

Note: There are also pre-existing uncommitted changes from Plan S (Voting/Stacking):
- `CHANGELOG.md` — Has Plan S entries
- `ferroml-python/python/ferroml/ensemble/__init__.py` — Voting/Stacking exports
- `ferroml-python/src/ensemble.rs` — Voting/Stacking Python bindings

## Key Architecture Decisions

### SparseTransformer trait (new)
```rust
#[cfg(feature = "sparse")]
pub trait SparseTransformer: Send + Sync {
    fn fit_sparse(&mut self, x: &CsrMatrix) -> Result<()>;
    fn transform_sparse(&self, x: &CsrMatrix) -> Result<CsrMatrix>;  // sparse -> sparse!
    fn fit_transform_sparse(&mut self, x: &CsrMatrix) -> Result<CsrMatrix>;
    fn is_fitted(&self) -> bool;
    fn n_features_out(&self) -> Option<usize>;
}
```

### TextPipeline design (Phase S.3)
- **Separate struct** from existing `Pipeline` (option C from plan)
- Steps: TextToSparse -> SparseToSparse -> SparseModel/DenseModel
- Three new pipeline traits: `PipelineTextTransformer`, `PipelineSparseTransformer`, `PipelineSparseModel`
- File: `ferroml-core/src/pipeline/text_pipeline.rs` (to be created)

### Python TextPipeline (Phase S.5)
- Duck-typing approach (like existing PyPipeline), NOT Rust TextPipeline wrapper
- Accepts `list[str]` input, flows through steps checking for `fit_text`/`transform_text` vs `fit`/`transform`
- Intermediate data can be scipy.sparse

## Implementation Approach for Parallel Agents

The plan has 6 phases with dependencies:
```
S.1 (done) -> S.2 + S.3 (can be parallel) -> S.4 -> S.5 -> S.6
```

**Recommended parallel agent strategy:**
1. **Agent 1**: Finish S.1 tests → then S.2 (TfidfVectorizer)
2. **Agent 2**: S.3 (TextPipeline core)
3. After both complete: S.4 (trait impls — depends on S.3's traits)
4. Then S.5 (Python bindings — depends on S.2 + S.4)
5. Finally S.6 (integration tests)

**Important**: Agents 1 and 2 can run in parallel since S.2 and S.3 only depend on S.1 (done). S.3 creates new traits in `pipeline/text_pipeline.rs` while S.2 creates `preprocessing/tfidf_vectorizer.rs` — no file conflicts.

## Existing Patterns to Follow

### PipelineModel impl pattern (from logistic.rs):
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

### Python __init__.py pattern:
```python
from ferroml import ferroml as _native
TfidfVectorizer = _native.preprocessing.TfidfVectorizer
```

## Files That Will Be Created (remaining phases)

| Phase | File | Purpose |
|-------|------|---------|
| S.2 | `ferroml-core/src/preprocessing/tfidf_vectorizer.rs` | TfidfVectorizer (CountVectorizer + TfidfTransformer) |
| S.3 | `ferroml-core/src/pipeline/text_pipeline.rs` | TextPipeline struct + pipeline traits |
| S.6 | `ferroml-core/src/testing/text_pipeline_integration.rs` | Integration tests |
| S.6 | `ferroml-core/src/testing/text_pipeline_correctness.rs` | Correctness fixtures vs sklearn |
| S.6 | `scripts/generate_text_pipeline_fixtures.py` | Generate sklearn reference data |
| S.5 | `ferroml-python/tests/test_tfidf_vectorizer.py` | Python TfidfVectorizer tests |
| S.5 | `ferroml-python/tests/test_text_pipeline.py` | Python TextPipeline tests |

## Files That Will Be Modified (remaining phases)

| Phase | File | Changes |
|-------|------|---------|
| S.2 | `ferroml-core/src/preprocessing/mod.rs` | Already has `pub mod tfidf_vectorizer` |
| S.3 | `ferroml-core/src/pipeline/mod.rs` | Add `pub mod text_pipeline`, re-exports |
| S.4 | `ferroml-core/src/preprocessing/count_vectorizer.rs` | PipelineTextTransformer impl |
| S.4 | `ferroml-core/src/preprocessing/tfidf_vectorizer.rs` | PipelineTextTransformer impl |
| S.4 | `ferroml-core/src/preprocessing/tfidf.rs` | PipelineSparseTransformer impl |
| S.4 | `ferroml-core/src/models/naive_bayes.rs` | PipelineSparseModel for 4 NB types |
| S.4 | `ferroml-core/src/models/logistic.rs` | PipelineSparseModel |
| S.4 | `ferroml-core/src/models/svm.rs` | PipelineSparseModel for LinearSVC/SVR |
| S.4 | `ferroml-core/src/models/knn.rs` | PipelineSparseModel for 3 KNN types |
| S.4 | `ferroml-core/src/models/regularized.rs` | PipelineSparseModel for Ridge |
| S.5 | `ferroml-python/src/preprocessing.rs` | Add PyTfidfVectorizer class + register |
| S.5 | `ferroml-python/src/pipeline.rs` | Add PyTextPipeline class + register |
| S.5 | `ferroml-python/python/ferroml/preprocessing/__init__.py` | Add TfidfVectorizer export |
| S.5 | `ferroml-python/python/ferroml/pipeline/__init__.py` | Add TextPipeline export |
| S.6 | `ferroml-core/src/testing/mod.rs` | Already has `pub mod text_pipeline` |

## Verification Commands

```bash
# Compile with sparse feature
cargo build --features sparse -p ferroml-core

# Run tfidf tests
cargo test --features sparse -p ferroml-core -- tfidf

# Run all Rust tests
cargo test --workspace

# Build Python bindings
source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml

# Run Python tests
python -m pytest ferroml-python/tests/ -x -q

# Format (required by pre-commit hook)
cargo fmt --all
```

## Build Notes

- Always run `cargo fmt --all` before committing (pre-commit hook)
- Build command: `source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml`
- The `tfidf_vectorizer.rs` file doesn't exist yet — compiler will error on `pub mod tfidf_vectorizer` until Phase S.2 creates it. This is fine; just create the file early if needed.
- `testing/mod.rs` already declares `pub mod text_pipeline` — that module already exists with basic CountVectorizer+TfidfTransformer tests from a prior plan.

## Other Notes

- The user calls this "Plan U" even though the plan file says "Plan S" — follow the user's naming.
- The user wants subagent/parallel agent driven development in the next session.
- There are pre-existing uncommitted changes from Plan S (Voting/Stacking bindings) in ensemble.rs and CHANGELOG.md — these should be committed together with Plan U work or separately first.
- Total expected tests: ~143 across all 6 phases.
- The `ParameterValue` type is in `crate::hpo` — it has `as_bool()`, `as_f64()`, `as_str()`, `as_usize()` methods.
