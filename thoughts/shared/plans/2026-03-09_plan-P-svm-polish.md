# Plan P: SVM Polish

**Goal**: Close remaining gaps in SVM API surface — loss function exposure, custom class weights, `decision_function`, dedicated Rust test module, docstring fixes.

**Scope**: 5 phases, each independently committable. Estimated ~800 lines Rust, ~200 lines Python, ~400 lines tests.

---

## P.1: Fix SVR Docstring + Expose Loss Parameters in Python

### P.1a: Fix SVR Docstring Bug

**File**: `ferroml-python/python/ferroml/svm/__init__.py`

Line 19 says `"supports epsilon-insensitive and Huber loss"` — Huber loss does NOT exist in the Rust core. Fix to:
```
"supports epsilon-insensitive loss"
```

### P.1b: Expose `loss` Parameter on PyLinearSVC

**File**: `ferroml-python/src/svm.rs` (PyLinearSVC, lines 34-113)

Current constructor: `__new__(c=1.0, max_iter=1000, tol=1e-4)`

Change to: `__new__(c=1.0, loss="squared_hinge", max_iter=1000, tol=1e-4)`

**Implementation**:
1. Add `loss: String` field to PyLinearSVC struct (store for repr)
2. Add `loss` param to `__new__` with default `"squared_hinge"`
3. Parse string to `LinearSVCLoss`:
   - `"hinge"` → `LinearSVCLoss::Hinge`
   - `"squared_hinge"` → `LinearSVCLoss::SquaredHinge`
   - anything else → `ValueError`
4. Call `.with_loss(parsed_loss)` on the Rust LinearSVC — **VERIFY** this builder method exists; if not, add it
5. Update `__repr__` to include loss

**Rust side check**: LinearSVC has a `loss` field (line 1790) and Default sets it to `SquaredHinge`. Need to verify a `.with_loss()` builder exists. If not, add one:
```rust
pub fn with_loss(mut self, loss: LinearSVCLoss) -> Self {
    self.loss = loss;
    self
}
```

### P.1c: Expose `loss` Parameter on PyLinearSVR

**File**: `ferroml-python/src/svm.rs` (PyLinearSVR, lines 135-232)

Current constructor: `__new__(c=1.0, epsilon=0.0, max_iter=1000, tol=1e-4)`

Change to: `__new__(c=1.0, epsilon=0.0, loss="epsilon_insensitive", max_iter=1000, tol=1e-4)`

**Implementation**:
1. Add `loss: String` field to PyLinearSVR struct
2. Add `loss` param to `__new__` with default `"epsilon_insensitive"`
3. Parse string to `LinearSVRLoss`:
   - `"epsilon_insensitive"` → `LinearSVRLoss::EpsilonInsensitive`
   - `"squared_epsilon_insensitive"` → `LinearSVRLoss::SquaredEpsilonInsensitive`
   - anything else → `ValueError`
4. Call `.with_loss(parsed_loss)` — same as above, verify/add builder
5. Update `__repr__` to include loss

### P.1d: Update `__init__.py` Docstrings

Update `ferroml-python/python/ferroml/svm/__init__.py` docstrings for LinearSVC and LinearSVR to mention loss parameter options.

### P.1 Tests (Python)

Add to `ferroml-python/tests/test_linear_svm.py`:
- `test_linear_svc_loss_hinge` — fit with `loss="hinge"`, verify works
- `test_linear_svc_loss_squared_hinge` — fit with `loss="squared_hinge"`, verify works
- `test_linear_svc_loss_invalid` — `loss="invalid"` raises `ValueError`
- `test_linear_svc_loss_in_repr` — `repr(model)` includes loss
- `test_linear_svr_loss_epsilon_insensitive` — fit with default loss
- `test_linear_svr_loss_squared` — fit with `loss="squared_epsilon_insensitive"`
- `test_linear_svr_loss_invalid` — raises `ValueError`
- `test_linear_svr_loss_in_repr` — repr includes loss

**~8 new Python tests**

---

## P.2: Expose Custom Class Weights in Python

### P.2a: Update PySVC to Accept Dict Class Weights

**File**: `ferroml-python/src/svm.rs` (PySVC, lines 317-472)

Current: `class_weight` param accepts `None` (Uniform) or `"balanced"`.

Change to also accept a Python dict `{class_label: weight}`:
- `None` → `ClassWeight::Uniform`
- `"balanced"` → `ClassWeight::Balanced`
- `{0.0: 1.0, 1.0: 5.0}` → `ClassWeight::Custom(vec![(0.0, 1.0), (1.0, 5.0)])`

**Implementation**:
1. In `__new__`, change `class_weight` extraction to try:
   - `None` → Uniform
   - String `"balanced"` → Balanced
   - Dict → extract items, convert to `Vec<(f64, f64)>`, create Custom
   - Anything else → `ValueError`
2. Use `PyAny` type for the parameter, then branch on type checking

### P.2b: Update PyLinearSVC to Accept Class Weights

Currently PyLinearSVC does NOT expose `class_weight` at all. Add it:
- Add `class_weight` param to `__new__` with default `None`
- Same parsing as PySVC (None / "balanced" / dict)
- Wire to Rust `LinearSVC::with_class_weight()`

**Rust side check**: LinearSVC has `class_weight: ClassWeight` field (line 1798) and needs `.with_class_weight()` builder. Verify or add.

### P.2 Tests (Python)

Add to `ferroml-python/tests/test_kernel_svm.py`:
- `test_svc_custom_class_weight_dict` — `class_weight={0.0: 1.0, 1.0: 10.0}` works, predictions shift toward minority
- `test_svc_custom_class_weight_invalid` — non-dict/non-string raises ValueError

Add to `ferroml-python/tests/test_linear_svm.py`:
- `test_linear_svc_class_weight_balanced` — `class_weight="balanced"` works
- `test_linear_svc_class_weight_dict` — custom dict works
- `test_linear_svc_class_weight_invalid` — invalid raises ValueError

**~5 new Python tests**

---

## P.3: Add `decision_function` to Python Bindings

### P.3a: Add Public `decision_function` Methods to Rust Core

The internal methods exist but aren't public on the main structs. Add public wrappers:

**SVC** (add to `impl SVC`):
```rust
/// Compute decision function values for each sample.
///
/// For OvO: returns Array2 shape (n_samples, n_classifiers) where
///   n_classifiers = n_classes * (n_classes - 1) / 2
/// For OvR: returns Array2 shape (n_samples, n_classes)
pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
    check_is_fitted(&self.classes, "decision_function")?;
    validate_predict_input(x, self.n_features.unwrap())?;
    match self.multiclass_strategy {
        MulticlassStrategy::OneVsOne => self.decision_function_ovo(x),
        MulticlassStrategy::OneVsRest => self.decision_function_ovr(x),
    }
}
```

Then implement `decision_function_ovo` and `decision_function_ovr`:

**OvO**: For each binary classifier, call `clf.decision_function(x)` → collect into Array2 columns.
**OvR**: Already done in `predict_ovr` lines 1066-1073 — extract into a method.

**SVR** (add to `impl SVR`):
```rust
/// Compute decision function values (same as predict for regression).
pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
    self.predict(x)  // For SVR, decision_function == predict
}
```

**LinearSVC** (add to `impl LinearSVC`):
```rust
/// Compute decision function values.
/// Returns Array2 shape (n_samples, n_classes) for multiclass,
/// or Array2 shape (n_samples, 1) for binary.
pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
    check_is_fitted(&self.classes, "decision_function")?;
    validate_predict_input(x, self.n_features.unwrap())?;
    let weights = self.weights.as_ref().unwrap();
    let intercepts = self.intercepts.as_ref().unwrap();
    let n_samples = x.nrows();
    let n_classifiers = weights.len();
    let mut decisions = Array2::zeros((n_samples, n_classifiers));
    for (clf_idx, (w, &b)) in weights.iter().zip(intercepts.iter()).enumerate() {
        let col = self.decision_function_binary(x, w, b);
        decisions.column_mut(clf_idx).assign(&col);
    }
    Ok(decisions)
}
```

**LinearSVR**: decision_function == predict (single value). Add thin wrapper returning Array1.

### P.3b: Expose in Python Bindings

**File**: `ferroml-python/src/svm.rs`

Add `decision_function` method to each of the 4 Python classes:

**PySVC**:
```rust
fn decision_function(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let x = x.as_array();
    let result = self.inner.decision_function(&x.to_owned())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Python::with_gil(|py| Ok(result.into_pyarray_bound(py).unbind()))
}
```

**PySVR**: Returns 1D array (same as predict).
**PyLinearSVC**: Returns 2D array (n_samples, n_classes).
**PyLinearSVR**: Returns 1D array (same as predict).

### P.3c: Update `__init__.py` Docstrings

Add `decision_function` to each class description in the svm `__init__.py`.

### P.3 Tests (Python)

Add to `ferroml-python/tests/test_kernel_svm.py`:
- `test_svc_decision_function_binary_shape` — shape is (n_samples, 1) for 2 classes OvO
- `test_svc_decision_function_multiclass_ovo_shape` — shape is (n_samples, n_pairs) for OvO
- `test_svc_decision_function_multiclass_ovr_shape` — shape is (n_samples, n_classes) for OvR
- `test_svc_decision_function_unfitted` — raises RuntimeError
- `test_svr_decision_function_shape` — shape is (n_samples,), values match predict
- `test_svr_decision_function_unfitted` — raises RuntimeError

Add to `ferroml-python/tests/test_linear_svm.py`:
- `test_linear_svc_decision_function_shape` — correct 2D shape
- `test_linear_svc_decision_function_unfitted` — raises RuntimeError
- `test_linear_svr_decision_function_shape` — 1D, matches predict output
- `test_linear_svr_decision_function_unfitted` — raises RuntimeError

**~10 new Python tests**

---

## P.4: Dedicated Rust SVM Test Module

### P.4a: Create `ferroml-core/src/testing/svm.rs`

Comprehensive test module covering areas NOT tested in `sklearn_correctness.rs`:

**Kernel Tests** (~5 tests):
- `test_kernel_linear_computation` — K(x,y) = x·y for known vectors
- `test_kernel_rbf_computation` — K(x,y) = exp(-γ||x-y||²) for known values
- `test_kernel_poly_computation` — K(x,y) = (γx·y + c₀)^d
- `test_kernel_sigmoid_computation` — K(x,y) = tanh(γx·y + c₀)
- `test_kernel_rbf_auto_gamma` — gamma=0 triggers 1/n_features

**SVC Tests** (~12 tests):
- `test_svc_binary_classification` — basic fit/predict
- `test_svc_multiclass_ovo` — 3+ classes with OvO
- `test_svc_multiclass_ovr` — 3+ classes with OvR
- `test_svc_ovo_vs_ovr_same_data` — both strategies produce reasonable results
- `test_svc_platt_scaling_probabilities` — probabilities sum to 1, range [0,1]
- `test_svc_balanced_class_weight` — shifts decision boundary toward majority
- `test_svc_custom_class_weight` — custom weights affect predictions
- `test_svc_decision_function_ovo_shape` — correct output dimensions
- `test_svc_decision_function_ovr_shape` — correct output dimensions
- `test_svc_decision_function_consistency` — positive decisions → positive class
- `test_svc_unfitted_errors` — predict/decision_function fail before fit
- `test_svc_all_kernels_fit` — all 4 kernels produce fitted model on same data

**SVR Tests** (~6 tests):
- `test_svr_basic_regression` — fit/predict on linear data
- `test_svr_epsilon_tube` — larger ε → fewer support vectors
- `test_svr_all_kernels` — all 4 kernels fit
- `test_svr_decision_function_equals_predict` — decision_function == predict
- `test_svr_support_vectors_subset` — n_sv ≤ n_train
- `test_svr_dual_coef_length` — len(dual_coef) == n_sv

**LinearSVC Tests** (~8 tests):
- `test_linear_svc_binary` — basic fit/predict
- `test_linear_svc_multiclass` — 3+ classes
- `test_linear_svc_hinge_loss` — fit with Hinge loss
- `test_linear_svc_squared_hinge_loss` — fit with SquaredHinge loss
- `test_linear_svc_balanced_weight` — class_weight=Balanced
- `test_linear_svc_custom_weight` — class_weight=Custom
- `test_linear_svc_decision_function_shape` — correct 2D output
- `test_linear_svc_decision_function_sign_matches_predict` — positive → class 1

**LinearSVR Tests** (~6 tests):
- `test_linear_svr_basic` — fit/predict on linear data
- `test_linear_svr_epsilon_insensitive_loss` — default loss
- `test_linear_svr_squared_loss` — SquaredEpsilonInsensitive loss
- `test_linear_svr_coefficients` — coef + intercept accessible
- `test_linear_svr_decision_function` — equals predict output
- `test_linear_svr_fit_intercept_false` — intercept is ~0

**~37 new Rust tests**

### P.4b: Register Module

Add `pub mod svm;` to `ferroml-core/src/testing/mod.rs`.

---

## P.5: Verify + Add Missing Rust Builder Methods

Before any Python binding changes, verify these builder methods exist on the Rust structs:

### LinearSVC Builders Needed
- `.with_loss(LinearSVCLoss)` — may not exist, add if missing
- `.with_class_weight(ClassWeight)` — may not exist, add if missing

### LinearSVR Builders Needed
- `.with_loss(LinearSVRLoss)` — may not exist, add if missing

### Verification
```bash
grep "fn with_loss\|fn with_class_weight" ferroml-core/src/models/svm.rs
```

If any are missing, add them following the existing builder pattern (e.g., `.with_c()`, `.with_tol()`).

---

## Execution Order

**Phase order matters because of dependencies:**

1. **P.5 first** — add missing Rust builders (prerequisite for P.1, P.2)
2. **P.4 second** — Rust test module (tests builders + decision_function, catches Rust bugs early)
3. **P.3 third** — add `decision_function` to Rust public API + Python bindings
4. **P.1 fourth** — loss parameter exposure + docstring fix
5. **P.2 last** — custom class weights in Python

Each phase is independently committable. Total: ~5 commits.

---

## Summary

| Phase | Scope | New Tests | Key Files |
|-------|-------|-----------|-----------|
| P.5 | Rust builders | 0 (tested in P.4) | svm.rs |
| P.4 | Rust SVM test module | ~37 Rust | testing/svm.rs |
| P.3 | decision_function | ~10 Python | svm.rs (core + python) |
| P.1 | loss params + docfix | ~8 Python | svm.rs (python), __init__.py |
| P.2 | custom class weights | ~5 Python | svm.rs (python) |
| **Total** | | **~37 Rust + ~23 Python** | |
