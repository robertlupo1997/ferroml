# Plan S: Pipeline + TextTransformer Integration

## Overview

Build a unified pipeline system that handles the full text-to-prediction workflow: raw text documents -> sparse term-count features -> TF-IDF weighting -> model prediction. Today, CountVectorizer (TextTransformer) and TfidfTransformer operate outside Pipeline because Pipeline only accepts `Transformer` (Array2 -> Array2). This plan bridges that gap with an enum-based intermediate data type, a `SparseTransformer` trait, a `TfidfVectorizer` convenience struct, and a new `TextPipeline` that chains it all together.

## Current State

### What exists:
- **Pipeline** (`ferroml-core/src/pipeline/mod.rs:300`): Sequential chain of `PipelineTransformer` steps + optional `PipelineModel`. All data flows as `Array2<f64>`. Has FeatureUnion, ColumnTransformer, caching, HPO search space merging.
- **Transformer trait** (`ferroml-core/src/preprocessing/mod.rs:114`): `fit(&mut self, x: &Array2<f64>)`, `transform(&self, x: &Array2<f64>) -> Array2<f64>`, plus `is_fitted`, `get_feature_names_out`, `n_features_in/out`.
- **TextTransformer trait** (`ferroml-core/src/preprocessing/count_vectorizer.rs:60`): `fit_text(&mut self, &[String])`, `transform_text(&self, &[String]) -> CsrMatrix`. Implemented by CountVectorizer.
- **TfidfTransformer** (`ferroml-core/src/preprocessing/tfidf.rs`): Standalone struct with `fit/transform` (Array2 -> Array2) and `fit_sparse/transform_sparse` (CsrMatrix -> Array2). Does NOT implement Transformer trait.
- **SparseModel trait** (`ferroml-core/src/models/traits.rs:61`): `fit_sparse(&mut self, x: &CsrMatrix, y: &Array1<f64>)`, `predict_sparse(&self, x: &CsrMatrix) -> Array1<f64>`. Implemented by 12 models (MultinomialNB, LogisticRegression, KNN variants, LinearSVC/SVR, Ridge, etc.).
- **CsrMatrix** (`ferroml-core/src/sparse.rs:67`): Wrapper around `sprs::CsMat<f64>` with `to_dense()`, `from_dense()`, `nrows()`, `ncols()`, `nnz()`.
- **Python Pipeline** (`ferroml-python/src/pipeline.rs`): Dynamic dispatch via PyObject method calls. Accepts `PyReadonlyArray2<f64>` for fit/transform/predict.

### The gap:
1. CountVectorizer produces `CsrMatrix` from `&[String]` but Pipeline expects `Array2<f64>` input.
2. TfidfTransformer doesn't implement `Transformer` trait, so it can't be a pipeline step either.
3. Pipeline has no concept of sparse intermediate data -- densifying defeats the purpose of sparse support.
4. No `TfidfVectorizer` (CountVectorizer + TfidfTransformer combined).
5. Python Pipeline only accepts numpy arrays, not lists of strings or scipy.sparse.

## Desired End State

```python
# Python: Full text classification pipeline in ~5 lines
from ferroml.pipeline import TextPipeline
from ferroml.preprocessing import TfidfVectorizer
from ferroml.naive_bayes import MultinomialNB

pipe = TextPipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', MultinomialNB()),
])
pipe.fit(documents_train, y_train)
predictions = pipe.predict(documents_test)
```

```rust
// Rust: Same workflow
let mut pipe = TextPipeline::new()
    .add_text_transformer("tfidf", TfidfVectorizer::new())
    .add_sparse_model("model", MultinomialNB::new());
pipe.fit(&documents, &y)?;
let predictions = pipe.predict(&documents_test)?;
```

Specifically:
- `TextPipeline` accepts `&[String]` input, flows data as `CsrMatrix` through sparse-aware steps, and routes to SparseModel for prediction.
- `TfidfVectorizer` = CountVectorizer + TfidfTransformer in a single step (text -> CsrMatrix -> TF-IDF weighted CsrMatrix).
- `SparseTransformer` trait for CsrMatrix -> CsrMatrix operations (TfidfTransformer, Normalizer, etc.).
- `Densifier` step to bridge sparse -> dense when the final model doesn't support SparseModel.
- Existing `Pipeline` is untouched -- full backward compatibility.
- 80+ Rust tests, 40+ Python tests.

---

## Implementation Phases

### Phase S.1: SparseTransformer Trait + TfidfTransformer Integration

**Overview**: Define the `SparseTransformer` trait for CsrMatrix-to-CsrMatrix transformations and make TfidfTransformer implement both `Transformer` (for backward compat) and `SparseTransformer`. Also make TfidfTransformer implement `PipelineTransformer`.

**Changes Required**:

1. **File**: `ferroml-core/src/preprocessing/mod.rs`
   - Add new trait `SparseTransformer`:
     ```rust
     /// Trait for transformers that operate on sparse matrices natively.
     ///
     /// Unlike `Transformer` (Array2 -> Array2), SparseTransformer operates
     /// in the sparse domain: CsrMatrix -> CsrMatrix, preserving sparsity.
     #[cfg(feature = "sparse")]
     pub trait SparseTransformer: Send + Sync {
         fn fit_sparse(&mut self, x: &CsrMatrix) -> Result<()>;
         fn transform_sparse(&self, x: &CsrMatrix) -> Result<CsrMatrix>;
         fn fit_transform_sparse(&mut self, x: &CsrMatrix) -> Result<CsrMatrix> {
             self.fit_sparse(x)?;
             self.transform_sparse(x)
         }
         fn is_fitted(&self) -> bool;
         fn n_features_out(&self) -> Option<usize>;
     }
     ```
   - Re-export `SparseTransformer`

2. **File**: `ferroml-core/src/preprocessing/tfidf.rs`
   - Implement `Transformer` for TfidfTransformer (delegates to existing `fit`/`transform` methods)
   - Implement `SparseTransformer` for TfidfTransformer:
     - `fit_sparse`: delegates to existing `fit_sparse` method
     - `transform_sparse`: returns CsrMatrix (currently returns Array2 -- need new `transform_sparse_to_sparse` that returns CsrMatrix instead of densifying). The current `transform_sparse` densifies, so we need a new native sparse-in/sparse-out path.
   - Implement `PipelineTransformer` for TfidfTransformer
   - Add `transform_sparse_native(&self, x: &CsrMatrix) -> Result<CsrMatrix>` that applies TF weighting, IDF weighting, and normalization directly on sparse structure, returning CsrMatrix.

3. **File**: `ferroml-core/src/preprocessing/count_vectorizer.rs`
   - CountVectorizer already returns CsrMatrix from `transform_text`. No trait changes needed here. But we should consider whether CountVectorizer should also implement `SparseTransformer` -- it should NOT, because its input is `&[String]`, not `CsrMatrix`. The `TextTransformer` trait already handles this correctly.

**Success Criteria**:
- [ ] Automated: `cargo test --features sparse -p ferroml-core -- tfidf` passes (existing + new tests)
- [ ] Automated: TfidfTransformer can be used as a `PipelineTransformer` step in existing Pipeline
- [ ] Automated: `SparseTransformer::transform_sparse` on TfidfTransformer returns CsrMatrix with correct values matching dense output
- [ ] Manual: Verify that existing TfidfTransformer API is unchanged

**Tests** (~20 tests):
- TfidfTransformer as Transformer trait (fit/transform with Array2)
- TfidfTransformer as SparseTransformer (fit_sparse/transform_sparse returning CsrMatrix)
- Sparse TF-IDF output matches dense TF-IDF output (round-trip verification)
- TfidfTransformer as PipelineTransformer in existing Pipeline
- L1/L2/None normalization on sparse path
- sublinear_tf on sparse path
- Not-fitted errors on sparse path

---

### Phase S.2: TfidfVectorizer (CountVectorizer + TfidfTransformer Combined)

**Overview**: Create `TfidfVectorizer` that wraps CountVectorizer + TfidfTransformer into a single step. Implements `TextTransformer` (text -> CsrMatrix of TF-IDF values).

**Changes Required**:

1. **File**: `ferroml-core/src/preprocessing/tfidf_vectorizer.rs` (NEW, ~300 lines)
   ```rust
   pub struct TfidfVectorizer {
       count_vectorizer: CountVectorizer,
       tfidf_transformer: TfidfTransformer,
   }

   impl TfidfVectorizer {
       pub fn new() -> Self { ... }
       // Builder methods forwarding to CountVectorizer and TfidfTransformer:
       pub fn with_max_features(mut self, n: usize) -> Self { ... }
       pub fn with_ngram_range(mut self, range: (usize, usize)) -> Self { ... }
       pub fn with_min_df(mut self, min_df: DocFrequency) -> Self { ... }
       pub fn with_max_df(mut self, max_df: DocFrequency) -> Self { ... }
       pub fn with_binary(mut self, binary: bool) -> Self { ... }
       pub fn with_stop_words(mut self, sw: Vec<String>) -> Self { ... }
       pub fn with_norm(mut self, norm: TfidfNorm) -> Self { ... }
       pub fn with_use_idf(mut self, use_idf: bool) -> Self { ... }
       pub fn with_smooth_idf(mut self, smooth: bool) -> Self { ... }
       pub fn with_sublinear_tf(mut self, sub: bool) -> Self { ... }
       // Getters forwarding to inner components:
       pub fn vocabulary(&self) -> Option<&HashMap<String, usize>> { ... }
       pub fn get_feature_names(&self) -> Option<&[String]> { ... }
       pub fn idf(&self) -> Option<&Array1<f64>> { ... }
   }

   impl TextTransformer for TfidfVectorizer {
       fn fit_text(&mut self, documents: &[String]) -> Result<()> {
           self.count_vectorizer.fit_text(documents)?;
           let counts = self.count_vectorizer.transform_text(documents)?;
           self.tfidf_transformer.fit_sparse(&counts)?;
           Ok(())
       }

       fn transform_text(&self, documents: &[String]) -> Result<CsrMatrix> {
           let counts = self.count_vectorizer.transform_text(documents)?;
           self.tfidf_transformer.transform_sparse_native(&counts)
       }
   }
   ```

2. **File**: `ferroml-core/src/preprocessing/mod.rs`
   - Add `pub mod tfidf_vectorizer;`
   - Re-export `TfidfVectorizer`

**Success Criteria**:
- [ ] Automated: `cargo test --features sparse -p ferroml-core -- tfidf_vectorizer` passes
- [ ] Automated: TfidfVectorizer output matches manual CountVectorizer + TfidfTransformer pipeline
- [ ] Automated: Vocabulary, feature names, IDF weights all accessible

**Tests** (~18 tests):
- Basic fit_transform_text with default params
- TfidfVectorizer output matches CountVectorizer -> TfidfTransformer manual chain
- All builder methods (max_features, ngram_range, min_df, max_df, stop_words, norm, use_idf, smooth_idf, sublinear_tf)
- Binary mode + TF-IDF
- Empty corpus error
- Not-fitted error
- Unseen terms in transform
- Sparse output shape and value correctness
- Dense convenience method (transform_text_dense)

---

### Phase S.3: TextPipeline (Rust Core)

**Overview**: Build `TextPipeline`, a new pipeline type that accepts `&[String]` input and chains text transformers, sparse transformers, and a final model (either `SparseModel` or `Model` with auto-densification).

**Design Decision -- Why a new struct instead of modifying Pipeline**:

The existing `Pipeline` is typed around `Array2<f64>` input/output at every step. Modifying it to support three different data types (`&[String]`, `CsrMatrix`, `Array2<f64>`) would require either:
- (a) An enum wrapping all three types flowing through steps (complex, runtime overhead checking variants at each step, type errors deferred to runtime)
- (b) Generics (would break the existing `dyn PipelineTransformer` trait object model)
- (c) A new pipeline struct purpose-built for the text workflow

Option (c) is cleanest: `TextPipeline` is a focused struct that understands the text -> sparse -> model flow. It has fewer moving parts than making Pipeline generic, and doesn't risk breaking the 3,211 existing tests.

**Changes Required**:

1. **File**: `ferroml-core/src/pipeline/text_pipeline.rs` (NEW, ~600 lines)

   ```rust
   /// A step in a text pipeline.
   pub enum TextPipelineStep {
       /// Text -> CsrMatrix (e.g., CountVectorizer, TfidfVectorizer)
       TextToSparse(Box<dyn PipelineTextTransformer>),
       /// CsrMatrix -> CsrMatrix (e.g., TfidfTransformer, SparseNormalizer)
       SparseToSparse(Box<dyn PipelineSparseTransformer>),
       /// CsrMatrix -> Array2 (densification bridge)
       SparseToDense,
       /// Final model that accepts sparse input
       SparseModel(Box<dyn PipelineSparseModel>),
       /// Final model that accepts dense input (auto-densifies)
       DenseModel(Box<dyn PipelineModel>),
   }

   /// Trait for text transformers usable in TextPipeline.
   pub trait PipelineTextTransformer: TextTransformer {
       fn search_space(&self) -> SearchSpace { SearchSpace::new() }
       fn clone_boxed(&self) -> Box<dyn PipelineTextTransformer>;
       fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>;
       fn name(&self) -> &str;
       fn n_features_out(&self) -> Option<usize>;
   }

   /// Trait for sparse transformers usable in TextPipeline.
   pub trait PipelineSparseTransformer: SparseTransformer {
       fn search_space(&self) -> SearchSpace { SearchSpace::new() }
       fn clone_boxed(&self) -> Box<dyn PipelineSparseTransformer>;
       fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>;
       fn name(&self) -> &str;
   }

   /// Trait for models that accept sparse input in pipelines.
   pub trait PipelineSparseModel: Send + Sync {
       fn fit_sparse(&mut self, x: &CsrMatrix, y: &Array1<f64>) -> Result<()>;
       fn predict_sparse(&self, x: &CsrMatrix) -> Result<Array1<f64>>;
       fn search_space(&self) -> SearchSpace;
       fn clone_boxed(&self) -> Box<dyn PipelineSparseModel>;
       fn set_param(&mut self, name: &str, value: &ParameterValue) -> Result<()>;
       fn name(&self) -> &str;
       fn is_fitted(&self) -> bool;
   }

   /// A pipeline for text classification/regression workflows.
   ///
   /// Accepts raw text documents as input and chains:
   /// 1. TextTransformer steps (text -> sparse)
   /// 2. SparseTransformer steps (sparse -> sparse)
   /// 3. A final model (sparse or dense)
   ///
   /// ## Example
   /// ```
   /// let mut pipe = TextPipeline::new()
   ///     .add_text_transformer("tfidf", TfidfVectorizer::new())
   ///     .add_sparse_model("nb", MultinomialNB::new());
   /// pipe.fit(&documents, &labels)?;
   /// let preds = pipe.predict(&test_docs)?;
   /// ```
   pub struct TextPipeline {
       steps: Vec<(String, TextPipelineStep)>,
       fitted: bool,
   }

   impl TextPipeline {
       pub fn new() -> Self { ... }

       /// Add a text-to-sparse transformer (must be the first step(s)).
       pub fn add_text_transformer<T: PipelineTextTransformer + 'static>(
           mut self, name: impl Into<String>, t: T
       ) -> Self { ... }

       /// Add a sparse-to-sparse transformer.
       pub fn add_sparse_transformer<T: PipelineSparseTransformer + 'static>(
           mut self, name: impl Into<String>, t: T
       ) -> Self { ... }

       /// Add a final model that accepts sparse input (no densification needed).
       pub fn add_sparse_model<M: PipelineSparseModel + 'static>(
           mut self, name: impl Into<String>, m: M
       ) -> Self { ... }

       /// Add a final dense model (will auto-densify before feeding).
       pub fn add_dense_model<M: PipelineModel + 'static>(
           mut self, name: impl Into<String>, m: M
       ) -> Self { ... }

       /// Fit the pipeline on text documents and labels.
       pub fn fit(&mut self, documents: &[String], y: &Array1<f64>) -> Result<()> {
           // 1. First text transformer: fit_text + transform_text -> CsrMatrix
           // 2. Subsequent sparse transformers: fit_sparse + transform_sparse -> CsrMatrix
           // 3. Final model:
           //    - SparseModel: fit_sparse(csr, y)
           //    - DenseModel: densify, then fit(dense, y)
       }

       /// Predict from text documents.
       pub fn predict(&self, documents: &[String]) -> Result<Array1<f64>> {
           // Transform through text->sparse->sparse chain, then predict
       }

       /// Transform text to features (without final model).
       pub fn transform(&self, documents: &[String]) -> Result<CsrMatrix> { ... }

       /// Transform text to dense features.
       pub fn transform_dense(&self, documents: &[String]) -> Result<Array2<f64>> { ... }

       /// Get combined search space from all steps.
       pub fn search_space(&self) -> SearchSpace { ... }

       /// Set hyperparameters with "step__param" convention.
       pub fn set_params(&mut self, params: &HashMap<String, ParameterValue>) -> Result<()> { ... }

       pub fn step_names(&self) -> Vec<&str> { ... }
       pub fn n_steps(&self) -> usize { ... }
       pub fn is_fitted(&self) -> bool { ... }
   }
   ```

2. **File**: `ferroml-core/src/pipeline/mod.rs`
   - Add `pub mod text_pipeline;`
   - Re-export `TextPipeline` and related traits
   - Re-export from `ferroml-core/src/lib.rs` pipeline module

3. **Step validation rules** (enforced at build time via the add_ methods):
   - First step(s) MUST be TextToSparse (text transformers)
   - Only one TextToSparse step is typical, but multiple are allowed (chained: text -> sparse -> text transformer that re-tokenizes? Unlikely, so validate: at most one TextToSparse, and it must be first)
   - SparseToSparse steps come after TextToSparse
   - Model must be last step
   - Cannot add steps after model

**Success Criteria**:
- [ ] Automated: `cargo test --features sparse -p ferroml-core -- text_pipeline` passes
- [ ] Automated: TextPipeline with TfidfVectorizer + MultinomialNB fits and predicts correctly
- [ ] Automated: TextPipeline with CountVectorizer + TfidfTransformer + LogisticRegression (dense path) works
- [ ] Automated: search_space() merges params from all steps
- [ ] Automated: set_params() dispatches to correct step

**Tests** (~30 tests):
- Construction: text_transformer -> sparse_model
- Construction: text_transformer -> sparse_transformer -> sparse_model
- Construction: text_transformer -> dense_model (auto-densification)
- Construction: text_transformer -> sparse_transformer -> dense_model
- fit + predict end-to-end with TfidfVectorizer + MultinomialNB
- fit + predict with CountVectorizer + TfidfTransformer (sparse) + MultinomialNB
- fit + predict with TfidfVectorizer + LinearRegression (dense path)
- transform() returns CsrMatrix
- transform_dense() returns Array2
- Not-fitted errors on predict/transform
- Empty pipeline error
- Empty documents error
- Single document
- Large corpus (100 docs, 1000 vocab)
- search_space() includes all step parameters
- set_params() updates correct step
- Step ordering validation (model must be last, text must be first)
- Prediction correctness: compare TextPipeline output vs manual step-by-step execution
- Sparse output shape matches expectations

---

### Phase S.4: PipelineTextTransformer / PipelineSparseTransformer / PipelineSparseModel Implementations

**Overview**: Implement the pipeline trait adapters for all relevant existing types so they can plug into TextPipeline.

**Changes Required**:

1. **File**: `ferroml-core/src/preprocessing/count_vectorizer.rs`
   - `impl PipelineTextTransformer for CountVectorizer`:
     - `clone_boxed()`, `name()` -> "CountVectorizer"
     - `set_param()` for: max_features, min_df, max_df, ngram_min, ngram_max, binary, lowercase
     - `search_space()` for relevant hyperparameters
     - `n_features_out()` -> vocabulary size if fitted

2. **File**: `ferroml-core/src/preprocessing/tfidf_vectorizer.rs`
   - `impl PipelineTextTransformer for TfidfVectorizer`:
     - `clone_boxed()`, `name()` -> "TfidfVectorizer"
     - `set_param()` for all combined params (CV + TFIDF)
     - `search_space()`
     - `n_features_out()`

3. **File**: `ferroml-core/src/preprocessing/tfidf.rs`
   - `impl PipelineSparseTransformer for TfidfTransformer`:
     - `clone_boxed()`, `name()` -> "TfidfTransformer"
     - `set_param()` for norm, use_idf, smooth_idf, sublinear_tf
     - `search_space()`

4. **File**: `ferroml-core/src/models/naive_bayes.rs`
   - `impl PipelineSparseModel for MultinomialNB`
   - `impl PipelineSparseModel for BernoulliNB`
   - `impl PipelineSparseModel for CategoricalNB`
   - `impl PipelineSparseModel for GaussianNB`
   - Each: delegates to existing SparseModel impl, adds search_space/set_param/clone_boxed

5. **File**: `ferroml-core/src/models/logistic.rs`
   - `impl PipelineSparseModel for LogisticRegression`

6. **File**: `ferroml-core/src/models/svm.rs`
   - `impl PipelineSparseModel for LinearSVC`
   - `impl PipelineSparseModel for LinearSVR`

7. **File**: `ferroml-core/src/models/knn.rs`
   - `impl PipelineSparseModel for KNeighborsClassifier`
   - `impl PipelineSparseModel for KNeighborsRegressor`
   - `impl PipelineSparseModel for NearestCentroid`

8. **File**: `ferroml-core/src/models/regularized.rs`
   - `impl PipelineSparseModel for RidgeRegression`

**Success Criteria**:
- [ ] Automated: All 12 SparseModel types can be used in TextPipeline
- [ ] Automated: CountVectorizer and TfidfVectorizer usable as PipelineTextTransformer
- [ ] Automated: TfidfTransformer usable as PipelineSparseTransformer
- [ ] Automated: `cargo test --features sparse -p ferroml-core` passes all existing + new tests

**Tests** (~20 tests):
- Each model type in a TextPipeline with TfidfVectorizer (smoke test: fit + predict doesn't panic)
- set_param round-trip for CountVectorizer, TfidfVectorizer, TfidfTransformer
- search_space includes correct parameters for each
- clone_boxed produces working clone

---

### Phase S.5: Python Bindings

**Overview**: Expose TextPipeline, TfidfVectorizer, and SparseTransformer to Python.

**Changes Required**:

1. **File**: `ferroml-python/src/preprocessing.rs`
   - Add `PyTfidfVectorizer` class:
     ```python
     class TfidfVectorizer:
         def __init__(self, *, max_features=None, ngram_range=(1,1),
                      min_df=1, max_df=1.0, binary=False, lowercase=True,
                      stop_words=None, norm='l2', use_idf=True,
                      smooth_idf=True, sublinear_tf=False): ...
         def fit(self, documents: list[str]) -> 'TfidfVectorizer': ...
         def transform(self, documents: list[str]) -> scipy.sparse.csr_matrix: ...
         def fit_transform(self, documents: list[str]) -> scipy.sparse.csr_matrix: ...
         def transform_dense(self, documents: list[str]) -> np.ndarray: ...
         @property
         def vocabulary_(self) -> dict: ...
         def get_feature_names_out(self) -> list[str]: ...
         @property
         def idf_(self) -> np.ndarray: ...
     ```

2. **File**: `ferroml-python/src/pipeline.rs`
   - Add `PyTextPipeline` class:
     ```python
     class TextPipeline:
         def __init__(self, steps: list[tuple[str, object]]): ...
         def fit(self, documents: list[str], y: np.ndarray) -> 'TextPipeline': ...
         def predict(self, documents: list[str]) -> np.ndarray: ...
         def transform(self, documents: list[str]) -> scipy.sparse.csr_matrix: ...
         def fit_predict(self, documents: list[str], y: np.ndarray) -> np.ndarray: ...
         @property
         def named_steps(self) -> dict: ...
         def get_params(self) -> dict: ...
         def set_params(self, **params) -> 'TextPipeline': ...
     ```
   - The Python TextPipeline uses duck typing (like the existing PyPipeline): checks for `fit_text`/`transform_text` or `fit`/`transform` methods on each step object. This allows mixing FerroML text transformers with any sklearn-compatible text transformer.
   - For the first step: call `fit_transform(documents)` (list of str). If it returns scipy.sparse, pass sparse forward. If it returns ndarray, wrap as dense.
   - For middle steps: call `fit_transform(X)` where X is whatever came from the previous step.
   - For the final model: call `fit(X, y)` / `predict(X)`.

3. **File**: `ferroml-python/python/ferroml/__init__.py`
   - Re-export `TfidfVectorizer` from preprocessing
   - Re-export `TextPipeline` from pipeline

4. **File**: `ferroml-python/python/ferroml/preprocessing/__init__.py`
   - Add `TfidfVectorizer` to exports

5. **File**: `ferroml-python/python/ferroml/pipeline/__init__.py`
   - Add `TextPipeline` to exports

**Implementation note on Python TextPipeline approach**: Rather than wrapping the Rust TextPipeline directly (which would require converting all Python step objects to Rust trait objects), the Python TextPipeline should use the same duck-typing approach as the existing PyPipeline. This is simpler and more compatible with the Python ecosystem. The key difference from PyPipeline is:
- `fit()` accepts `list[str]` for documents instead of `np.ndarray`
- Intermediate data can be `scipy.sparse.csr_matrix` (not just ndarray)
- Steps are detected by checking for `fit_text`/`transform_text` (text transformers) vs `fit`/`transform` (regular transformers) vs `fit`/`predict` (models)

**Success Criteria**:
- [ ] Automated: `pytest tests/test_tfidf_vectorizer.py` passes
- [ ] Automated: `pytest tests/test_text_pipeline.py` passes
- [ ] Automated: TfidfVectorizer produces scipy.sparse output
- [ ] Automated: TextPipeline end-to-end with TfidfVectorizer + MultinomialNB
- [ ] Manual: `from ferroml.preprocessing import TfidfVectorizer` works
- [ ] Manual: `from ferroml.pipeline import TextPipeline` works

**Python Tests** (~40 tests across two test files):

`tests/test_tfidf_vectorizer.py` (~20 tests):
- Basic fit/transform with default params
- Output is scipy.sparse.csr_matrix
- Dense output via transform_dense
- vocabulary_ property
- get_feature_names_out()
- idf_ property
- All builder params (ngram_range, max_features, min_df, max_df, stop_words, binary, norm, use_idf, smooth_idf, sublinear_tf)
- Match against sklearn.feature_extraction.text.TfidfVectorizer output (correctness fixture)
- Empty input error
- Not-fitted error
- Unseen terms in transform

`tests/test_text_pipeline.py` (~20 tests):
- TfidfVectorizer + MultinomialNB (text classification)
- TfidfVectorizer + LogisticRegression (text classification)
- CountVectorizer + MultinomialNB
- TfidfVectorizer + LinearSVC
- fit + predict round-trip
- transform returns sparse
- Predictions have correct shape
- Empty documents error
- Not-fitted error
- Single document predict
- Large corpus (100+ docs)
- get_params / set_params
- named_steps access
- Accuracy sanity check (20 newsgroups subset or similar small fixture)

---

### Phase S.6: Testing & Verification

**Overview**: Integration tests, correctness fixtures against sklearn, and documentation.

**Changes Required**:

1. **File**: `ferroml-core/src/testing/text_pipeline_integration.rs` (NEW)
   - End-to-end integration tests:
     - TextPipeline with every SparseModel type
     - TextPipeline with dense model fallback
     - Prediction correctness vs manual step-by-step execution
     - Large-scale test (1000 docs, 5000 vocab features)
     - Serialization round-trip (if applicable)

2. **File**: `ferroml-core/src/testing/mod.rs`
   - Add `pub mod text_pipeline_integration;`

3. **File**: `scripts/generate_text_pipeline_fixtures.py` (NEW)
   - Generate sklearn reference outputs for:
     - TfidfVectorizer with various params
     - TfidfVectorizer + MultinomialNB pipeline accuracy
     - CountVectorizer + TfidfTransformer + LogisticRegression pipeline accuracy
   - Save as JSON fixtures

4. **File**: `ferroml-core/src/testing/text_pipeline_correctness.rs` (NEW)
   - Load fixtures and verify FerroML matches sklearn within tolerance

**Success Criteria**:
- [ ] Automated: `cargo test --features sparse -p ferroml-core` -- all pass
- [ ] Automated: `pytest tests/` -- all pass
- [ ] Automated: `cargo fmt --all --check` passes
- [ ] Automated: `cargo clippy --all-targets --features sparse` clean
- [ ] Automated: Correctness tests match sklearn within rtol=1e-6

**Tests** (~15 additional integration/correctness tests):
- TfidfVectorizer output matches sklearn fixture
- TextPipeline accuracy matches sklearn pipeline accuracy (within tolerance)
- Sparse intermediate shapes match expectations
- Memory efficiency: sparse pipeline uses less memory than dense (basic check via nnz vs full shape)

---

## Dependencies

- **Phase S.1** has no dependencies (only adds trait + impl to existing code)
- **Phase S.2** depends on S.1 (TfidfVectorizer needs SparseTransformer for TfidfTransformer)
- **Phase S.3** depends on S.1 (TextPipeline uses SparseTransformer trait)
- **Phase S.4** depends on S.3 (implements PipelineSparseModel etc. defined in S.3)
- **Phase S.5** depends on S.2 + S.4 (Python wraps TfidfVectorizer + TextPipeline with all impls)
- **Phase S.6** depends on S.5 (correctness tests need everything working)

Execution order: S.1 -> S.2 + S.3 (parallel) -> S.4 -> S.5 -> S.6

## Risks & Mitigations

### Risk 1: TfidfTransformer sparse-to-sparse path is non-trivial
The current `transform_sparse` densifies to Array2. Writing a true CsrMatrix-to-CsrMatrix path requires row-wise normalization on sparse data, which means computing row norms from sparse structure.

**Mitigation**: The IDF weighting step only scales columns (multiply each nnz value by the corresponding IDF weight) -- this is O(nnz) and straightforward. The TF weighting (sublinear) is element-wise on nnz values. Only the normalization step is tricky: computing row L2 norms from sparse data requires summing squares of nnz values per row, then dividing. This is well-defined and O(nnz). Implementation: iterate CSR row-by-row, compute norm from non-zero entries, divide each entry by the norm.

### Risk 2: Python duck typing may fail for some step types
The Python TextPipeline uses method detection to determine step types. If a step has both `transform` and `transform_text`, it's ambiguous.

**Mitigation**: Priority order: check for `fit_text` first (text transformer), then `predict` (model), then `transform` (regular transformer). FerroML classes won't have conflicting methods. Document that user-provided steps should follow the convention.

### Risk 3: PipelineSparseModel trait explosion
Adding another pipeline trait for each model is boilerplate.

**Mitigation**: Use a macro `impl_pipeline_sparse_model!` that generates the boilerplate for each model, similar to how PipelineModel/PipelineTransformer impls are likely structured. This keeps per-model code to ~5 lines.

### Risk 4: Feature gating complexity
`SparseTransformer`, `TextPipeline`, and `PipelineSparseModel` all depend on the `sparse` feature flag.

**Mitigation**: Gate the entire `text_pipeline` module behind `#[cfg(feature = "sparse")]`. The `sparse` feature is already well-established in the codebase. TextPipeline inherently requires sparse support since text features are sparse.

### Risk 5: Backward compatibility
Existing Pipeline, PipelineTransformer, PipelineModel traits must not change.

**Mitigation**: All new types are additive. No existing signatures change. TextPipeline is a separate struct. The only modification to existing code is adding trait impls to existing types (TfidfTransformer gets Transformer + SparseTransformer + PipelineTransformer impls; models get PipelineSparseModel impls). These are purely additive.

## Summary

| Phase | Scope | New/Modified Files | Estimated Tests |
|-------|-------|-------------------|-----------------|
| S.1 | SparseTransformer trait + TfidfTransformer impls | 2 modified | ~20 |
| S.2 | TfidfVectorizer | 1 new, 1 modified | ~18 |
| S.3 | TextPipeline (Rust core) | 1 new, 1 modified | ~30 |
| S.4 | Pipeline trait impls for all relevant types | 8 modified | ~20 |
| S.5 | Python bindings | 4-5 modified | ~40 |
| S.6 | Integration tests + correctness fixtures | 3 new, 1 modified | ~15 |
| **Total** | | **~6 new, ~15 modified** | **~143** |
