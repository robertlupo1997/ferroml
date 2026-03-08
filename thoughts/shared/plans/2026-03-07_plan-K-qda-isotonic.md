# Plan K: QDA + IsotonicRegression

## Overview

Two small, independent models that leverage existing infrastructure. QDA extends LDA with per-class covariance matrices. IsotonicRegression wraps the existing PAVA algorithm from IsotonicCalibrator.

## Current State

- **LDA**: Fully implemented at `decomposition/lda.rs:141` with fit, predict, predict_proba, decision_function
  - Has per-class means, shared covariance, eigendecomposition via nalgebra
  - Implements both Transformer and Model traits
- **IsotonicCalibrator**: Fully implemented at `models/calibration.rs:315`
  - PAVA algorithm at `calibration.rs:367-428`
  - Linear interpolation at `calibration.rs:431-471`
  - Currently clipped to [0, 1] for probability calibration
- **ColumnTransformer**: Already exists in `pipeline/mod.rs:1390` with 16 tests — NOT MISSING

## Desired End State

- `QuadraticDiscriminantAnalysis` struct implementing Model trait
- `IsotonicRegression` struct implementing Model trait
- Python bindings + tests for both
- ~20 Rust tests per model, ~10 Python tests per model

---

## Phase K.1: QuadraticDiscriminantAnalysis

**Overview**: QDA fits per-class covariance matrices instead of a shared one. Decision boundary is quadratic.

**Changes Required**:

1. **File**: `ferroml-core/src/models/qda.rs` (NEW, ~400 lines)

   **Struct**:
   ```rust
   pub struct QuadraticDiscriminantAnalysis {
       // Config (match sklearn defaults)
       reg_param: f64,                // default 0.0 — regularizes scaling: S2 = (1-r)*S2 + r*I
       priors: Option<Vec<f64>>,      // class priors (None = estimated from data)
       store_covariance: bool,        // default false — whether to store full covariance matrices
       tol: f64,                      // default 1e-4
       // Fitted
       means_: Option<Array2<f64>>,           // (n_classes, n_features)
       covariances_: Option<Vec<Array2<f64>>>, // per-class covariance (only if store_covariance=true)
       priors_: Option<Array1<f64>>,           // fitted class priors
       classes_: Option<Vec<f64>>,
       rotations_: Option<Vec<Array2<f64>>>,   // principal axes per class
       scalings_: Option<Vec<Array1<f64>>>,    // Gaussian scaling along principal axes
       n_features_in_: Option<usize>,
   }
   ```

   **Key differences from LDA** (`decomposition/lda.rs`):
   - LDA: S_w (within-class scatter) is shared → linear decision boundary
   - QDA: Each class k has its own Sigma_k → quadratic decision boundary
   - Decision function: log P(k) - 0.5 * log|Sigma_k| - 0.5 * (x - mu_k)^T Sigma_k^{-1} (x - mu_k)

   **Methods**:
   - `new()` + builder methods
   - `fit(x, y)` — compute per-class means and covariance matrices
   - `predict(x)` — argmax of decision function
   - `predict_proba(x)` — softmax of decision function values
   - `decision_function(x)` — discriminant value per class

   **Implementation details**:
   - Use nalgebra for eigendecomposition of per-class covariance (following PCA pattern)
   - Store eigendecomposition for efficient prediction (rotations_ + scalings_)
   - Handle rank-deficient covariances with reg_param

2. **File**: `ferroml-core/src/models/mod.rs` (EDIT)
   - Add `pub mod qda;`
   - Re-export `QuadraticDiscriminantAnalysis`

**Success Criteria**:
- [ ] `cargo check -p ferroml-core`
- [ ] QDA achieves higher accuracy than LDA on non-linearly-separable data

---

## Phase K.2: QDA Tests

**Changes Required**:

1. **File**: `ferroml-core/src/models/qda.rs` (EDIT — `#[cfg(test)]` module)
   - ~20 tests:
     - Basic fit/predict on 2-class data
     - predict_proba sums to 1.0
     - Matches LDA results when covariances are equal
     - Outperforms LDA when class covariances differ
     - Multi-class (3+ classes) works
     - reg_param prevents singular covariance issues
     - Custom priors affect predictions
     - Edge cases: single feature, unbalanced classes
     - decision_function shape is correct
     - Reproducibility

**Success Criteria**:
- [ ] `cargo test -p ferroml-core qda` — all pass

---

## Phase K.3: IsotonicRegression

**Overview**: Standalone isotonic regression using the existing PAVA algorithm. Fits a non-decreasing (or non-increasing) piecewise-linear function.

**Changes Required**:

1. **File**: `ferroml-core/src/models/isotonic.rs` (NEW, ~250 lines)

   **Struct**:
   ```rust
   pub struct IsotonicRegression {
       // Config (match sklearn defaults)
       increasing: Increasing,     // true, false, or "auto" (Spearman correlation)
       y_min: Option<f64>,         // default None (-inf)
       y_max: Option<f64>,         // default None (+inf)
       out_of_bounds: OutOfBounds, // default Nan
       // Fitted
       x_min_: Option<f64>,            // min training X
       x_max_: Option<f64>,            // max training X
       x_thresholds_: Option<Vec<f64>>,  // unique ascending X knots
       y_thresholds_: Option<Vec<f64>>,  // de-duplicated Y knots
       increasing_: Option<bool>,        // inferred increasing direction
       n_features_in_: Option<usize>,    // must be 1
   }

   pub enum Increasing { True, False, Auto }
   pub enum OutOfBounds { Nan, Clip, Raise }
   ```

   **Implementation**:
   - Reuse PAVA algorithm from `calibration.rs:367-428` — extract to shared utility or call directly
   - Reuse interpolation from `calibration.rs:431-471`
   - Remove [0, 1] clipping — use configurable y_min/y_max
   - Input validation: x must be 1D (single feature)

   **Methods**:
   - `new()` + builder
   - `fit(x, y)` — requires x to be single-column; sorts by x then applies PAVA
   - `predict(x)` — interpolate at new x values
   - Also implement `Transformer` trait: `transform(x)` = `predict(x)` for pipeline use

2. **File**: `ferroml-core/src/models/mod.rs` (EDIT)
   - Add `pub mod isotonic;`
   - Re-export `IsotonicRegression`

**Success Criteria**:
- [ ] `cargo check -p ferroml-core`
- [ ] IsotonicRegression produces monotonically non-decreasing output

---

## Phase K.4: IsotonicRegression Tests

**Changes Required**:

1. **File**: `ferroml-core/src/models/isotonic.rs` (EDIT — `#[cfg(test)]` module)
   - ~20 tests:
     - Output is monotonically non-decreasing
     - With `increasing: false`, output is non-increasing
     - Fits perfectly on already-monotone data
     - Handles tied x values
     - y_min/y_max bounds work
     - out_of_bounds: NaN for extrapolation, Clip clips to range, Raise errors
     - Single point, two points
     - Multi-column input raises error (must be 1D)
     - Predictions interpolate between knots

**Success Criteria**:
- [ ] `cargo test -p ferroml-core isotonic` — all pass

---

## Phase K.5: Python Bindings + Tests

**Changes Required**:

1. **File**: `ferroml-python/src/linear.rs` (EDIT) — or new files as appropriate
   - Add `PyQuadraticDiscriminantAnalysis`
   - Add `PyIsotonicRegression`

2. **File**: Python `__init__.py` files (EDIT)
   - Add QDA to `decomposition` or `linear` module
   - Add IsotonicRegression to `linear` or `preprocessing` module

3. **File**: `ferroml-python/tests/test_qda_isotonic.py` (NEW)
   - ~20 tests total

**Success Criteria**:
- [ ] `pytest ferroml-python/tests/test_qda_isotonic.py -v` — all pass

---

## Execution Order

```
K.1 (QDA core)           — 25 min
K.2 (QDA tests)          — 15 min, depends on K.1
K.3 (IsotonicRegression) — 20 min, independent of K.1
K.4 (Isotonic tests)     — 10 min, depends on K.3
K.5 (Python bindings)    — 15 min, depends on K.1 + K.3
```

K.1+K.2 and K.3+K.4 can run in parallel.

## Dependencies

- LDA code for reference (decomposition/lda.rs)
- IsotonicCalibrator PAVA algorithm (models/calibration.rs:367)
- nalgebra for eigendecomposition

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| QDA singular covariance for small classes | reg_param adds diagonal regularization |
| PAVA reuse requires refactoring calibration.rs | Can copy/adapt instead if refactoring is risky |
| IsotonicRegression with tied x-values | Average y-values for same x before PAVA |
| QDA memory scales O(k * d^2) for k classes | Acceptable for typical use; document for very high-dimensional data |
