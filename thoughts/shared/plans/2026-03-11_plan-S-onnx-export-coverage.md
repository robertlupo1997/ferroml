# Plan S: ONNX Export Coverage Expansion

## Overview

Expand FerroML's ONNX export from 9 models (5 linear + 4 tree) to 40+ models covering all major model families and preprocessing transformers. This makes FerroML models deployable via ONNX Runtime, TensorRT, CoreML, and other ONNX-compatible inference engines without any Rust or Python dependency at prediction time.

## Current State

### What exists now

**OnnxExportable trait** (`ferroml-core/src/onnx/mod.rs:153`):
- `to_onnx(&self, config: &OnnxConfig) -> Result<Vec<u8>>` — serialize to bytes
- `export_onnx(path, config)` — write to file (default impl)
- `onnx_n_features()`, `onnx_n_outputs()`, `onnx_output_type()` — metadata

**ONNX protobuf layer** (`ferroml-core/src/onnx/protos.rs`):
- Hand-rolled prost definitions for ModelProto, GraphProto, NodeProto, TensorProto, etc.
- Supports both standard ONNX domain and `ai.onnx.ml` domain
- IR version 9, opset 18, ML opset 3

**9 models currently support ONNX export:**

| Model | File | ONNX Ops Used |
|-------|------|---------------|
| LinearRegression | `onnx/linear.rs:70` | MatMul + Add + Squeeze |
| RidgeRegression | `onnx/linear.rs:97` | MatMul + Add + Squeeze |
| LassoRegression | `onnx/linear.rs:123` | MatMul + Add + Squeeze |
| ElasticNet | `onnx/linear.rs:149` | MatMul + Add + Squeeze |
| LogisticRegression | `onnx/linear.rs:227` | MatMul + Add + Sigmoid |
| DecisionTreeRegressor | `onnx/tree.rs:435` | TreeEnsembleRegressor (ML) |
| DecisionTreeClassifier | `onnx/tree.rs:458` | TreeEnsembleClassifier (ML) |
| RandomForestRegressor | `onnx/tree.rs:486` | TreeEnsembleRegressor (ML) |
| RandomForestClassifier | `onnx/tree.rs:517` | TreeEnsembleClassifier (ML) |

**Pure-Rust inference engine** (`ferroml-core/src/inference/`):
- Supports: MatMul, Add, Squeeze, Sigmoid, Softmax, Flatten, Reshape, TreeEnsembleRegressor, TreeEnsembleClassifier
- Round-trip tests exist in `ferroml-core/src/testing/onnx.rs`

**Python bindings**: No ONNX export exposed to Python yet (`ferroml-python/src/` has no `export_onnx`/`to_onnx` references).

**Feature flags** (`ferroml-core/Cargo.toml:130-131`):
- `onnx` — enabled by default, gates export code
- `onnx-validation` — optional, pulls `ort` crate for round-trip testing

### What doesn't exist

- ONNX export for 46+ models (boosting, SVM, NB, KNN, clustering, etc.)
- ONNX export for any preprocessing transformers
- Python bindings for ONNX export
- External round-trip validation (onnxruntime via Python)
- Pipeline/composite ONNX graph construction
- Several ONNX operator implementations needed for inference engine

## Desired End State

1. **40+ models** support `OnnxExportable` (all models where ONNX representation is feasible)
2. **8+ preprocessing transformers** export as ONNX sub-graphs
3. **Python bindings** expose `export_onnx(path)` and `to_onnx_bytes()` on all supported models
4. **Round-trip validation** via `onnxruntime` Python tests (export, load in ORT, compare predictions)
5. **Inference engine** extended with new operators (Normalizer, Scaler, SVMClassifier, SVMRegressor, etc.)
6. **Documentation** listing all supported models and their ONNX operator mappings

## Model Audit: Full Classification

### Tier 1 — Straightforward (linear algebra ops, already-proven patterns)
These use MatMul + Add + activation, directly analogous to existing linear exports.

| Model | Internal State | ONNX Strategy | Priority |
|-------|---------------|---------------|----------|
| RidgeClassifier | weights + intercept | Gemm + ArgMax | HIGH |
| RobustRegression | coefficients + intercept | MatMul + Add + Squeeze | HIGH |
| QuantileRegression | coefficients + intercept | MatMul + Add + Squeeze | HIGH |
| SGDClassifier | weights + intercept | Gemm + Sigmoid/Softmax | HIGH |
| SGDRegressor | weights + intercept | MatMul + Add + Squeeze | HIGH |
| PassiveAggressiveClassifier | weights + intercept | Gemm + ArgMax | HIGH |
| LinearSVC | weights + intercepts (OvR) | LinearClassifier (ML) | HIGH |
| LinearSVR | weights + intercept | LinearRegressor (ML) or MatMul+Add | HIGH |

### Tier 2 — Tree ensembles (reuse existing TreeEnsembleBuilder)
These all store `Vec<DecisionTree{Classifier,Regressor}>` internally and can reuse tree.rs.

| Model | Internal State | ONNX Strategy | Priority |
|-------|---------------|---------------|----------|
| GradientBoostingRegressor | `Vec<DecisionTreeRegressor>` + init_prediction + learning_rate | TreeEnsembleRegressor (weighted SUM) | HIGH |
| GradientBoostingClassifier | `Vec<Vec<DecisionTreeRegressor>>` + init + classes | TreeEnsembleRegressor per class + post-processing | HIGH |
| AdaBoostClassifier | `Vec<DecisionTreeClassifier>` + estimator_weights | TreeEnsembleClassifier (weighted) | HIGH |
| AdaBoostRegressor | `Vec<DecisionTreeRegressor>` + estimator_weights | TreeEnsembleRegressor (weighted) | HIGH |
| ExtraTreesClassifier | `Vec<DecisionTreeClassifier>` | TreeEnsembleClassifier (AVERAGE) | HIGH |
| ExtraTreesRegressor | `Vec<DecisionTreeRegressor>` | TreeEnsembleRegressor (AVERAGE) | HIGH |

### Tier 3 — ONNX-ML domain operators (NB, SVM with kernels)

| Model | Internal State | ONNX Strategy | Priority |
|-------|---------------|---------------|----------|
| GaussianNB | theta (means), sigma (variances), class_prior | Custom graph: log-likelihood computation | MEDIUM |
| MultinomialNB | feature_log_prob, class_log_prior | MatMul + Add + ArgMax | MEDIUM |
| BernoulliNB | feature_log_prob, class_log_prior | MatMul + Add + ArgMax | MEDIUM |
| SVC (linear kernel) | weights via support vectors | SVMClassifier (ML) | MEDIUM |
| SVR (linear kernel) | dual_coef, support vectors | SVMRegressor (ML) | MEDIUM |

### Tier 4 — Preprocessing transformers (Scaler/Normalizer ops)

| Transformer | Internal State | ONNX Strategy | Priority |
|-------------|---------------|---------------|----------|
| StandardScaler | mean, scale | Sub + Div (or Scaler ML op) | HIGH |
| MinMaxScaler | data_min, data_range, feature_range | Sub + Div + Mul + Add | HIGH |
| RobustScaler | center, scale | Sub + Div | HIGH |
| MaxAbsScaler | max_abs (scale) | Div | HIGH |
| OneHotEncoder | categories | OneHotEncoder (ML) | MEDIUM |
| OrdinalEncoder | categories | LabelEncoder (ML) per feature | MEDIUM |
| LabelEncoder | classes | LabelEncoder (ML) | MEDIUM |
| TfidfTransformer | idf weights | Mul (element-wise by idf) | MEDIUM |

### Tier 5 — Complex but valuable

| Model | Internal State | ONNX Strategy | Priority |
|-------|---------------|---------------|----------|
| KNeighborsClassifier | training data (stored) | Not feasible for ONNX (requires storing full dataset) | SKIP |
| KNeighborsRegressor | training data (stored) | Not feasible for ONNX | SKIP |
| SVC (RBF/poly kernel) | support vectors + dual_coef | SVMClassifier (ML) with kernel params | LOW |
| SVR (RBF/poly kernel) | support vectors + dual_coef | SVMRegressor (ML) with kernel params | LOW |
| GaussianProcessRegressor | kernel + training data | Not standard ONNX op, skip | SKIP |
| GaussianProcessClassifier | kernel + training data | Not standard ONNX op, skip | SKIP |
| HistGradientBoostingRegressor | HistTree (binned thresholds) + bin_mapper | Convert bin thresholds to real thresholds, use TreeEnsembleRegressor | MEDIUM |
| HistGradientBoostingClassifier | HistTree + bin_mapper | Same, TreeEnsembleClassifier | MEDIUM |
| IsolationForest | tree ensemble | TreeEnsembleRegressor + post-processing | LOW |
| MultiOutputRegressor | inner estimators | Sequence of sub-graphs + Concat | MEDIUM |
| MultiOutputClassifier | inner estimators | Sequence of sub-graphs + Concat | MEDIUM |
| CalibratedClassifierCV | base classifier + calibrator | Sub-graph chaining | LOW |

### Models to SKIP (no feasible ONNX representation)

| Model | Reason |
|-------|--------|
| KNeighborsClassifier | Requires full training dataset at inference |
| KNeighborsRegressor | Requires full training dataset at inference |
| GaussianProcessRegressor | Requires kernel evaluation over training data |
| GaussianProcessClassifier | Same |
| LocalOutlierFactor | Requires training data |
| DBSCAN | Transductive — no predict on new data |
| HDBSCAN | Same |
| AgglomerativeClustering | Same |
| KMeans | Could do (centroids → distance), but non-standard |
| GaussianMixture | Could do (means/covs), but complex custom graph |
| TSNE | Transductive — no predict |
| PCA | Possible (MatMul) but lower priority |
| LDA | Possible but lower priority |
| QuadraticDiscriminantAnalysis | Complex custom graph |
| IsotonicRegression | Piecewise linear — needs LinearRegressor ML op |
| CategoricalNB | Complex per-category log prob |
| CountVectorizer | Text → sparse — not standard ONNX |
| KBinsDiscretizer | Binning — custom graph |
| PowerTransformer | Box-Cox/Yeo-Johnson — custom ops |
| QuantileTransformer | Requires interpolation table |

## Implementation Phases

---

### Phase S.1: Accessor Methods & Internal Plumbing
**Overview**: Add missing public accessor methods on models that need them for ONNX export, and expose ONNX via Python bindings infrastructure.

**Changes Required**:

1. **File**: `ferroml-core/src/models/boosting.rs`
   - Add `pub fn estimators(&self) -> Option<&[Vec<DecisionTreeRegressor>]>` on `GradientBoostingClassifier`
   - Add `pub fn init_predictions(&self) -> Option<&Array1<f64>>` on both
   - Add `pub fn learning_rate_at(&self, i: usize) -> f64` helper or expose schedule

2. **File**: `ferroml-core/src/models/adaboost.rs`
   - Add `pub fn estimators(&self) -> Option<&[DecisionTreeClassifier]>` on `AdaBoostClassifier`
   - Add `pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]>` on `AdaBoostRegressor`

3. **File**: `ferroml-core/src/models/extra_trees.rs`
   - Add `pub fn estimators(&self) -> Option<&[DecisionTreeClassifier]>` on `ExtraTreesClassifier`
   - Add `pub fn estimators(&self) -> Option<&[DecisionTreeRegressor]>` on `ExtraTreesRegressor`

4. **File**: `ferroml-core/src/models/sgd.rs`
   - Add `pub fn weights(&self) -> Option<&Array1<f64>>` and `pub fn intercept_value(&self) -> Option<f64>` on `SGDRegressor` (if not existing)
   - Verify `SGDClassifier` exposes weights for multi-class case

5. **File**: `ferroml-core/src/models/svm.rs`
   - Add `pub fn weights(&self) -> Option<&Vec<Array1<f64>>>` and `pub fn intercepts(&self) -> Option<&Vec<f64>>` on `LinearSVC` (verify existing)
   - Add `pub fn weights(&self) -> Option<&Array1<f64>>` and `pub fn intercept(&self) -> f64` accessors on `LinearSVR` (verify existing)

6. **File**: `ferroml-core/src/models/hist_boosting.rs`
   - Add `pub fn trees(&self) -> Option<&[Vec<HistTree>]>` on `HistGradientBoostingClassifier`
   - Add `pub fn trees(&self) -> Option<&[HistTree]>` on `HistGradientBoostingRegressor`
   - Add `pub fn bin_mapper(&self) -> Option<&CategoricalBinMapper>` on both
   - Add method to convert HistTree bin thresholds to real-valued thresholds for ONNX

**Success Criteria**:
- [ ] `cargo test --lib` passes with new accessor methods
- [ ] No breaking API changes

**Estimated effort**: ~100 lines, low risk

---

### Phase S.2: Linear Model Family ONNX Export
**Overview**: Export all remaining linear-like models (8 models). These all follow the same MatMul + Add + activation pattern.

**Changes Required**:

1. **File**: `ferroml-core/src/onnx/linear.rs` (extend existing)
   - `impl OnnxExportable for RidgeClassifier` — Gemm + ArgMax (multi-class: weights matrix is [n_features, n_classes])
   - `impl OnnxExportable for RobustRegression` — MatMul + Add + Squeeze (same as LinearRegression)
   - `impl OnnxExportable for QuantileRegression` — MatMul + Add + Squeeze
   - `impl OnnxExportable for SGDRegressor` — MatMul + Add + Squeeze
   - `impl OnnxExportable for PassiveAggressiveClassifier` — Gemm + ArgMax (similar to RidgeClassifier)

2. **File**: `ferroml-core/src/onnx/sgd.rs` (new file)
   - `impl OnnxExportable for SGDClassifier` — binary: Gemm + Sigmoid; multi-class: Gemm + Softmax + ArgMax
   - Needs to handle both binary and multi-class cases

3. **File**: `ferroml-core/src/onnx/svm.rs` (new file)
   - `impl OnnxExportable for LinearSVC` — Use `ai.onnx.ml.LinearClassifier` operator (or multi-class Gemm + ArgMax)
   - `impl OnnxExportable for LinearSVR` — Use `ai.onnx.ml.LinearRegressor` operator (or MatMul + Add)

4. **File**: `ferroml-core/src/onnx/mod.rs`
   - Add `mod sgd;` and `mod svm;`
   - Add helper `create_argmax_node()` for classifier output
   - Add helper `create_gemm_node_multiclass()` for multi-class linear models

5. **File**: `ferroml-core/src/inference/operators.rs`
   - Add `ArgMaxOp` operator for inference engine
   - Add `LinearClassifierOp` if using ML domain
   - Add `LinearRegressorOp` if using ML domain

6. **File**: `ferroml-core/src/inference/session.rs`
   - Register new operators in `compile_operator()`

**New ONNX helpers needed in mod.rs**:
```rust
pub fn create_argmax_node(input: &str, output: &str, name: &str, axis: i64, keepdims: i64) -> NodeProto
pub fn create_cast_node(input: &str, output: &str, name: &str, to_type: TensorProtoDataType) -> NodeProto
```

**Success Criteria**:
- [ ] `cargo test -p ferroml-core --features onnx` — all 8 new OnnxExportable impls have unit tests
- [ ] Round-trip test: export → inference engine → compare with native predict (within f32 tolerance)
- [ ] ~16 new tests (2 per model: export validity + round-trip parity)

**Estimated effort**: ~600 lines, medium risk

---

### Phase S.3: Tree Ensemble ONNX Export (Boosting + ExtraTrees)
**Overview**: Export GradientBoosting (regressor + classifier), AdaBoost (regressor + classifier), and ExtraTrees (regressor + classifier) — 6 models total. All reuse the existing `TreeEnsembleBuilder`.

**Changes Required**:

1. **File**: `ferroml-core/src/onnx/tree.rs` (extend existing)

   **GradientBoostingRegressor**:
   - Collect all estimator trees via `estimators()`
   - Each tree's leaf values need to be scaled by learning_rate
   - `base_values` set to `init_prediction`
   - Aggregate function: `SUM`

   **GradientBoostingClassifier**:
   - Binary case: single sequence of trees, base_value = init_prediction, SUM aggregate, then Sigmoid post-transform
   - Multi-class case: `n_classes` separate tree sequences, each outputs raw score, then Softmax
   - This requires multiple TreeEnsembleRegressor nodes or a single one with `n_targets = n_classes`

   **AdaBoostClassifier**:
   - Weighted ensemble: multiply each tree's class votes by `estimator_weight`
   - Use TreeEnsembleClassifier with weighted class outputs

   **AdaBoostRegressor**:
   - Weighted median: complex. Best approximation: weighted SUM with TreeEnsembleRegressor
   - Note: exact AdaBoost.R2 prediction uses weighted median, which is NOT expressible in standard ONNX. Export the weighted-sum approximation with a doc warning.

   **ExtraTreesClassifier / ExtraTreesRegressor**:
   - Identical to RandomForest export (AVERAGE aggregate)
   - Internal estimators are DecisionTreeClassifier/Regressor

2. **File**: `ferroml-core/src/onnx/mod.rs`
   - Extend `TreeEnsembleBuilder` to support per-tree weight scaling (for learning_rate and AdaBoost weights)
   - Add `base_values` parameter to `create_tree_regressor_graph()`

**Key complexity**: GradientBoostingClassifier multi-class case. The internal representation is `Vec<Vec<DecisionTreeRegressor>>` where outer is iterations and inner is per-class trees. ONNX TreeEnsembleRegressor with `n_targets = n_classes` can represent this: flatten all trees, mark tree_id such that each class gets its own target_id.

**Success Criteria**:
- [ ] 6 new `impl OnnxExportable` with unit tests
- [ ] Round-trip parity tests for all 6 models (within f32 tolerance)
- [ ] GradientBoosting binary + multi-class both tested
- [ ] ~18 new tests (3 per model: export, round-trip, edge cases)

**Estimated effort**: ~800 lines, high complexity for GradientBoostingClassifier

---

### Phase S.4: Preprocessing Transformer ONNX Export
**Overview**: Export 8 preprocessing transformers as ONNX graphs. These are critical for deploying full pipelines.

**Design**: Add a new trait `OnnxTransformExportable` (or reuse `OnnxExportable` with transform semantics — input and output are both feature tensors).

**Changes Required**:

1. **File**: `ferroml-core/src/onnx/preprocessing.rs` (new file)

   **StandardScaler**: `(X - mean) / scale`
   - Nodes: Sub(X, mean_tensor) → Div(_, scale_tensor) → output
   - 2 initializers: mean [n_features], scale [n_features]

   **MinMaxScaler**: `(X - data_min) / data_range * (max - min) + min`
   - Nodes: Sub → Div → Mul → Add
   - 4 initializers: data_min, data_range, feature_range_scale, feature_range_min

   **RobustScaler**: `(X - center) / scale`
   - Nodes: Sub(X, center_tensor) → Div(_, scale_tensor)
   - 2 initializers

   **MaxAbsScaler**: `X / max_abs`
   - Nodes: Div(X, max_abs_tensor)
   - 1 initializer

   **OneHotEncoder**: Use `ai.onnx.ml.OneHotEncoder` operator
   - Attributes: cats_int64s or cats_strings, zeros

   **OrdinalEncoder**: Use `ai.onnx.ml.LabelEncoder` per feature → Concat
   - Multiple LabelEncoder nodes + Concat

   **LabelEncoder**: Use `ai.onnx.ml.LabelEncoder` operator
   - Attributes: keys_strings/keys_int64s, values_int64s

   **TfidfTransformer**: Element-wise multiply by IDF vector
   - Nodes: Mul(X, idf_tensor)
   - 1 initializer: idf [n_features]

2. **File**: `ferroml-core/src/onnx/mod.rs`
   - Add `mod preprocessing;`
   - Add helpers: `create_sub_node()`, `create_div_node()`, `create_mul_node()`, `create_concat_node()`

3. **File**: `ferroml-core/src/inference/operators.rs`
   - Add `SubOp`, `DivOp`, `MulOp`, `ConcatOp` operators
   - Add `OneHotEncoderOp` (ML domain)
   - Add `LabelEncoderOp` (ML domain)

**Success Criteria**:
- [ ] 8 new OnnxExportable impls with unit tests
- [ ] Round-trip: transform → export → inference → compare outputs
- [ ] ~24 new tests (3 per transformer)

**Estimated effort**: ~700 lines, medium complexity

---

### Phase S.5: Naive Bayes ONNX Export
**Overview**: Export GaussianNB, MultinomialNB, and BernoulliNB. These require custom computation graphs.

**Changes Required**:

1. **File**: `ferroml-core/src/onnx/naive_bayes.rs` (new file)

   **MultinomialNB / BernoulliNB**:
   - Both use: `log P(y=k|X) = X @ feature_log_prob.T + class_log_prior`
   - Graph: MatMul(X, feature_log_prob_transposed) + Add(class_log_prior) → ArgMax for labels, Softmax(Exp()) for probabilities
   - Straightforward linear-algebra graph

   **GaussianNB**:
   - `log P(x_j|y=k) = -0.5 * ((x_j - theta_k_j)^2 / sigma_k_j + log(2π * sigma_k_j))`
   - Need ops: Sub, Mul, Div, Log, Exp, ReduceSum, Add, ArgMax
   - More complex graph but all standard ONNX ops
   - Pre-compute `log_var = log(2π * sigma)` as initializer
   - Graph: Sub(X, theta) → Square → Div(sigma) → Add(log_var) → Mul(-0.5) → ReduceSum(axis=1) → Add(log_prior)

2. **File**: `ferroml-core/src/onnx/mod.rs`
   - Add `mod naive_bayes;`
   - Add helpers: `create_reduce_sum_node()`, `create_log_node()`, `create_exp_node()`, `create_neg_node()`

3. **File**: `ferroml-core/src/inference/operators.rs`
   - Add `ReduceSumOp`, `LogOp`, `ExpOp`, `NegOp`, `SquareOp` (or use Mul self)

**Success Criteria**:
- [ ] 3 new OnnxExportable impls
- [ ] Round-trip parity for all 3 (within f32 tolerance — NB is numerically sensitive)
- [ ] ~9 new tests

**Estimated effort**: ~500 lines, medium complexity (GaussianNB graph is intricate)

---

### Phase S.6: Kernel SVM & HistGradientBoosting ONNX Export
**Overview**: Export SVC/SVR with kernel support using ONNX-ML SVMClassifier/SVMRegressor, and HistGradientBoosting models by converting binned thresholds to real thresholds.

**Changes Required**:

1. **File**: `ferroml-core/src/onnx/svm.rs` (extend from Phase S.2)

   **SVC with kernel**:
   - Use `ai.onnx.ml.SVMClassifier` operator
   - Attributes: kernel_type, kernel_params (gamma, coef0, degree), support_vectors, coefficients, rho
   - Only feasible for models that expose support vectors and dual coefficients
   - Binary OvO/OvR decomposition needs careful handling

   **SVR with kernel**:
   - Use `ai.onnx.ml.SVMRegressor` operator
   - Attributes: kernel_type, kernel_params, support_vectors, coefficients, rho

2. **File**: `ferroml-core/src/onnx/hist_boosting.rs` (new file)

   **HistGradientBoosting{Regressor,Classifier}**:
   - Need to convert HistTree + CategoricalBinMapper into standard TreeEnsemble
   - For each internal node: `bin_threshold` (u8) → actual float threshold via `bin_mapper.bin_thresholds[feature][bin]`
   - Leaf values remain as-is
   - Then use the same TreeEnsembleBuilder from tree.rs
   - Requires S.1 accessor methods

3. **File**: `ferroml-core/src/inference/operators.rs`
   - Add `SVMClassifierOp` and `SVMRegressorOp` (ML domain)

**Success Criteria**:
- [ ] SVC/SVR kernel export + round-trip (at least RBF kernel)
- [ ] HistGBRT export + round-trip for both regressor and classifier
- [ ] ~12 new tests

**Estimated effort**: ~800 lines, high complexity (SVM kernel parameter mapping, HistTree threshold conversion)

---

### Phase S.7: Python Bindings for ONNX Export
**Overview**: Expose `export_onnx(path)` and `to_onnx_bytes()` methods on all Python-bound models that implement OnnxExportable.

**Changes Required**:

1. **File**: `ferroml-python/src/linear.rs`, `trees.rs`, `ensemble.rs`, `svm.rs`, `naive_bayes.rs`, `preprocessing.rs`, etc.
   - Add `export_onnx(&self, path: &str, model_name: Option<String>) -> PyResult<()>` method to each PyO3 class
   - Add `to_onnx_bytes(&self, model_name: Option<String>) -> PyResult<Vec<u8>>` method
   - Pattern:
     ```rust
     fn export_onnx(&self, path: &str, model_name: Option<String>) -> PyResult<()> {
         let config = OnnxConfig::new(model_name.unwrap_or_else(|| "ferroml_model".into()));
         self.inner.export_onnx(path, &config).map_err(to_py_err)
     }
     ```

2. **File**: `ferroml-python/python/ferroml/__init__.py`
   - Export `OnnxConfig` if desired (or keep it simple with just kwargs)

3. **File**: `ferroml-python/tests/test_onnx_export.py` (new)
   - Test export_onnx() for all supported models
   - Test to_onnx_bytes() returns valid protobuf
   - Test round-trip with onnxruntime (if available, else skip)

**Python API design**:
```python
from ferroml import LinearRegression, OnnxConfig

model = LinearRegression()
model.fit(X, y)

# Simple export
model.export_onnx("model.onnx")

# With options
model.export_onnx("model.onnx", model_name="my_lr", input_name="features")

# Get bytes
onnx_bytes = model.to_onnx_bytes()

# Round-trip with onnxruntime
import onnxruntime as ort
session = ort.InferenceSession(onnx_bytes)
ort_pred = session.run(None, {"input": X_test.astype(np.float32)})[0]
```

**Success Criteria**:
- [ ] `export_onnx()` and `to_onnx_bytes()` available on 40+ Python model classes
- [ ] Python tests for each model family (linear, tree, boosting, svm, nb, preprocessing)
- [ ] Optional onnxruntime round-trip tests (skip if ort not installed)
- [ ] ~60 new Python tests

**Estimated effort**: ~800 lines Rust + 400 lines Python, medium complexity (mostly mechanical)

---

### Phase S.8: External Round-Trip Validation with onnxruntime
**Overview**: Comprehensive validation that exported ONNX models produce correct predictions when loaded in onnxruntime (the reference ONNX runtime).

**Changes Required**:

1. **File**: `ferroml-python/tests/test_onnx_roundtrip.py` (new)
   - For each ONNX-exportable model:
     1. Fit model on known data
     2. Get native FerroML predictions
     3. Export to ONNX bytes
     4. Load in onnxruntime
     5. Run inference on same data
     6. Assert predictions match within tolerance (f32 precision: atol=1e-5 for regressors, atol=1e-4 for classifiers)
   - Skip if onnxruntime not installed
   - Cover edge cases: single sample, large batch, near-zero coefficients

2. **File**: `scripts/validate_onnx_exports.py` (new convenience script)
   - Automated validation script that tests all models
   - Outputs a compatibility matrix (model × status)
   - Can be run in CI

3. **File**: `ferroml-core/src/testing/onnx.rs` (extend existing)
   - Add Rust-side round-trip tests using the built-in inference engine
   - Test matrix: all OnnxExportable models × {small, medium} data sizes

**Tolerance guidelines**:
- Regression models: atol=1e-5, rtol=1e-4 (f64→f32 precision loss)
- Classification labels: exact match
- Classification probabilities: atol=1e-4
- Tree models: exact match (no floating-point arithmetic in traversal)
- Preprocessing: atol=1e-6 (simple arithmetic)

**Success Criteria**:
- [ ] All 40+ models pass onnxruntime round-trip (when ort available)
- [ ] All models pass built-in inference engine round-trip
- [ ] CI integration (optional: `pytest -m onnx` marker)
- [ ] ~80 new tests total (40 Python + 40 Rust)

**Estimated effort**: ~600 lines Python + 400 lines Rust

---

### Phase S.9: MultiOutput & Composite Model ONNX Export
**Overview**: Export MultiOutputRegressor, MultiOutputClassifier, and lay groundwork for Pipeline ONNX export.

**Changes Required**:

1. **File**: `ferroml-core/src/onnx/multioutput.rs` (new file)

   **MultiOutputRegressor<M: OnnxExportable>**:
   - Export each inner estimator as its own sub-graph
   - Combine outputs with Concat along axis=1
   - Input shared across all sub-graphs

   **MultiOutputClassifier<M: OnnxExportable>**:
   - Same pattern, but each sub-graph outputs a class label
   - Concat labels into [batch, n_outputs]

2. **File**: `ferroml-core/src/onnx/mod.rs`
   - Add `mod multioutput;`
   - Add `create_concat_node()` helper
   - Consider adding `SubGraphBuilder` for composing multiple model graphs

3. **File**: `ferroml-core/src/inference/operators.rs`
   - Add `ConcatOp` if not already present

**Key challenge**: ONNX doesn't have a built-in "run N models in parallel" concept. Each estimator's graph nodes must be embedded in the single graph with unique node names (suffix `_output_0`, `_output_1`, etc.) and shared input.

**Success Criteria**:
- [ ] MultiOutputRegressor/Classifier export when inner model is OnnxExportable
- [ ] Round-trip tests with LinearRegression as inner model
- [ ] ~8 new tests

**Estimated effort**: ~400 lines, high complexity (graph composition)

---

### Phase S.10: Documentation & Compatibility Matrix
**Overview**: Document all ONNX-supported models, opset compatibility, and known limitations.

**Changes Required**:

1. **File**: `docs/onnx-export.md` (new)
   - Full compatibility matrix: model → ONNX support (yes/no/partial)
   - Opset version requirements
   - Known precision limitations (f64→f32)
   - Example code (Rust + Python)
   - Pipeline export instructions

2. **File**: `CHANGELOG.md` (update)
   - Document ONNX expansion under v0.3.0

3. **File**: `ferroml-core/src/onnx/mod.rs` (update module docs)
   - Update supported model list
   - Add examples for new model families

4. **File**: `README.md` (update)
   - Add ONNX export section to feature list

**Success Criteria**:
- [ ] All ONNX-supported models documented with examples
- [ ] Compatibility matrix accurate and tested

**Estimated effort**: ~200 lines documentation

---

## Phase Execution Order & Dependencies

```
S.1 (Accessors)
 ├── S.2 (Linear family)      — no deps beyond S.1
 ├── S.3 (Tree ensembles)     — needs S.1 for boosting accessors
 ├── S.4 (Preprocessing)      — no deps beyond S.1
 └── S.5 (Naive Bayes)        — no deps beyond S.1

S.2 + S.3 + S.4 + S.5 complete
 ├── S.6 (SVM kernels + Hist) — needs S.1 accessors + S.3 tree infrastructure
 ├── S.7 (Python bindings)    — needs S.2-S.5 for models to bind
 └── S.9 (MultiOutput)        — needs S.2+ for inner model support

S.6 + S.7 + S.9 complete
 ├── S.8 (Round-trip validation) — needs everything above
 └── S.10 (Documentation)       — needs everything above
```

**Recommended execution**: S.1 → S.2 → S.3 → S.4 (parallel with S.5) → S.7 → S.6 → S.9 → S.8 → S.10

## Dependencies

- **prost** crate (already in workspace) — ONNX protobuf serialization
- **ort** crate (optional, already in Cargo.toml) — Rust-side ONNX validation
- **onnxruntime** Python package (optional) — Python-side round-trip tests
- **onnx** Python package (optional) — model validation via `onnx.checker`
- No new Cargo dependencies required for core export

## Risks & Mitigations

### Risk 1: f64 → f32 precision loss
- FerroML uses f64 internally; ONNX standard uses f32
- **Mitigation**: Use DOUBLE (f64) ONNX tensors where possible; document precision expectations; use generous tolerances in tests; add optional f64 mode to OnnxConfig

### Risk 2: GradientBoostingClassifier multi-class complexity
- Multi-class GBC stores `Vec<Vec<DecisionTreeRegressor>>` (iterations × classes), which maps awkwardly to ONNX TreeEnsemble
- **Mitigation**: Flatten into single TreeEnsembleRegressor with n_targets=n_classes, using tree_id indexing to separate class contributions. Validate carefully with 3+ class datasets.

### Risk 3: HistGradientBoosting bin threshold conversion
- HistGBRT uses binned (u8) thresholds, not real-valued. Need bin_mapper to convert.
- **Mitigation**: Add `to_real_threshold()` method on bin_mapper; verify converted trees produce identical predictions to original binned trees before ONNX export.

### Risk 4: AdaBoost.R2 weighted median
- The exact AdaBoost.R2 prediction algorithm uses weighted median, which has no ONNX equivalent.
- **Mitigation**: Export as weighted sum with explicit documentation warning about the approximation. Users who need exact parity should use the native Rust/Python API.

### Risk 5: SVC OvO multi-class
- SVC with One-vs-One stores K*(K-1)/2 binary classifiers. ONNX-ML SVMClassifier supports this but the attribute layout is complex.
- **Mitigation**: Start with linear kernel (which uses LinearClassifier), add kernel SVM support as stretch goal. Test with 2, 3, and 5+ classes.

### Risk 6: Python onnxruntime availability
- onnxruntime may not be installed in all environments.
- **Mitigation**: Make ort tests optional (`pytest.importorskip("onnxruntime")`). Core ONNX export tests use only the built-in Rust inference engine.

## Test Count Summary

| Phase | Rust Tests | Python Tests | Total |
|-------|-----------|-------------|-------|
| S.1   | 10        | 0           | 10    |
| S.2   | 16        | 0           | 16    |
| S.3   | 18        | 0           | 18    |
| S.4   | 24        | 0           | 24    |
| S.5   | 9         | 0           | 9     |
| S.6   | 12        | 0           | 12    |
| S.7   | 0         | 60          | 60    |
| S.8   | 40        | 40          | 80    |
| S.9   | 8         | 0           | 8     |
| S.10  | 0         | 0           | 0     |
| **Total** | **137** | **100** | **237** |

## ONNX Operator Coverage After Plan S

### Standard ONNX Domain (opset 18)
Currently: MatMul, Add, Squeeze, Sigmoid, Softmax, Flatten, Reshape

Adding: Sub, Div, Mul, ArgMax, Cast, Concat, ReduceSum, Log, Exp, Neg, Reciprocal

### ONNX-ML Domain (opset 3)
Currently: TreeEnsembleRegressor, TreeEnsembleClassifier

Adding: LinearClassifier, LinearRegressor, SVMClassifier, SVMRegressor, OneHotEncoder, LabelEncoder, Scaler, Normalizer
