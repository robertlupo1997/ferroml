# Coding Conventions

**Analysis Date:** 2026-03-16

## Language-Specific Patterns

### Rust Code Style

**File Naming:**
- Module files: `snake_case` (e.g., `adaboost.rs`, `linear_regression.rs`)
- Public struct/trait: PascalCase (e.g., `AdaBoostClassifier`, `LinearRegression`)
- Private helper: snake_case (e.g., `fit_internal()`, `validate_input()`)

**Function Naming:**
- Public API: `snake_case` (e.g., `fit()`, `predict()`, `with_learning_rate()`)
- Constructor: `new()` or builder with `with_*()` pattern
- Getter: `snake_case` with trailing `_` for fields (e.g., `classes()`, `theta_()`)
- Internal check: prefix with `check_` (e.g., `check_is_fitted()`)

**Variable Naming:**
- Model parameters: single descriptors (e.g., `n_estimators`, `learning_rate`, `max_depth`)
- Array/matrix: explicit types (e.g., `x` for features Array2, `y` for targets Array1)
- Math notation: single letter allowed in ML algorithms (e.g., `w` for weights, `b` for bias)

**Type/Trait Naming:**
- Traits: descriptive adjectives ending in action (e.g., `Model`, `Transformer`, `CrossValidator`)
- Error variants: PascalCase with context (e.g., `ShapeMismatch`, `ConvergenceFailure`, `NotFitted`)

**Code Organization:**
- Modules: one concern per file (e.g., `adaboost.rs` contains only AdaBoost implementations)
- Section separators: `// =============================================================================`
- Block comments: `//! ` for module-level docs, `/// ` for item-level docs
- Visibility: public types at top, helpers/state lower

**Example from `adaboost.rs`:**
```rust
/// AdaBoost classifier using SAMME.R algorithm.
///
/// Fits an ensemble of weighted decision stumps, where each subsequent
/// estimator focuses on the samples that previous estimators got wrong.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaBoostClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub random_state: Option<u64>,
    pub warm_start: bool,

    // Private fitted state
    estimators: Option<Vec<DecisionTreeClassifier>>,
    estimator_weights: Option<Array1<f64>>,
    classes: Option<Array1<f64>>,
    n_features: Option<usize>,
}

impl AdaBoostClassifier {
    pub fn new(n_estimators: usize) -> Self { /* ... */ }
    pub fn with_learning_rate(mut self, lr: f64) -> Self { /* ... */ }
    pub fn estimator_weights(&self) -> Option<&Array1<f64>> { /* ... */ }
    pub fn classes(&self) -> Option<&Array1<f64>> { /* ... */ }
}
```

### Python Code Style

**File Naming:**
- Test files: `test_*.py` (e.g., `test_naive_bayes.py`)
- Module files: `snake_case.py`
- Conftest: `conftest.py`

**Function/Class Naming:**
- Public classes: PascalCase (e.g., `TestGaussianNB`, `GaussianNB`)
- Test methods: `test_*` pattern describing behavior (e.g., `test_fit_predict_basic`, `test_predict_proba_sums_to_one`)
- Fixtures: descriptive snake_case (e.g., `regression_data`, `classification_data`)
- Helpers: lowercase with leading underscore if private (e.g., `_fit_both()`)

**Test Class Organization:**
- Class per component (e.g., `class TestGaussianNB`)
- Group related tests by section (e.g., `# --------- GaussianNB tests --------`)
- One assertion focus per test method

**Example from `test_naive_bayes.py`:**
```python
class TestGaussianNB:
    def test_fit_predict_basic(self):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0],
                      [6.0, 7.0], [7.0, 6.0], [8.0, 8.0]])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (6,)

    def test_predict_proba_sums_to_one(self):
        X = np.array([[1.0, 2.0], [2.0, 1.0], [6.0, 7.0], [7.0, 6.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        probas = model.predict_proba(X)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)
```

## Formatting & Linting

**Rust Formatting:**
- Tool: Built-in `cargo fmt` (Rust 2021 edition)
- Applied on commit (pre-commit hook)
- No custom `.rustfmt.toml` — uses edition defaults
- Line length: implicit via formatter (typically 100 chars)

**Rust Linting:**
- Tool: `cargo clippy` with `-D warnings` (deny all warnings)
- Clippy config: ~50+ lint allowances in `lib.rs` for ML-specific exceptions:
  - `#![allow(clippy::too_many_lines)]` — Complex algorithms need space
  - `#![allow(clippy::too_many_arguments)]` — ML models have many parameters
  - `#![allow(clippy::many_single_char_names)]` — Math notation uses single letters
  - `#![allow(clippy::float_cmp)]` — Epsilon comparisons used where needed
  - `#![allow(clippy::return_self_not_must_use)]` — Builder pattern heavy codebase
  - See `/home/tlupo/ferroml/ferroml-core/src/lib.rs` lines 88-149 for full list

**Python Formatting:**
- Tool: pytest-standard (no explicit formatter configured)
- Convention: PEP 8 style observed in test files
- Import order: stdlib → third-party → local (implicit)

## Import Organization

**Rust Pattern:**
```rust
// 1. Crate imports
use crate::models::{check_is_fitted, validate_fit_input, Model};
use crate::{FerroError, Result};

// 2. External crates (std first, then workspace deps, then other crates)
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// 3. Internal modules
mod clustering { ... }
```

**Python Pattern:**
```python
# 1. Standard library
import numpy as np
import pytest
from typing import Tuple

# 2. Third-party (ferroml)
from ferroml.naive_bayes import GaussianNB, MultinomialNB

# 3. Local imports
from conftest_comparison import get_iris, get_wine
```

## Error Handling

**Rust Error Strategy:**
- Error type: `FerroError` enum with variants for ML-specific cases
- File: `/home/tlupo/ferroml/ferroml-core/src/error.rs`
- Pattern: Use `Result<T> = std::result::Result<T, FerroError>`
- Error variants include:
  - `InvalidInput(String)` — Invalid input data
  - `ShapeMismatch { expected, actual }` — Array shape mismatch
  - `AssumptionViolation { assumption, test, p_value }` — Statistical assumption failed
  - `ConvergenceFailure { iterations, reason }` — Optimization didn't converge
  - `NotFitted { operation }` — Model not fitted before predict

**Error Creation:**
```rust
// Preferred: use helper methods
FerroError::shape_mismatch("(n, m)", "(n, k)")
FerroError::invalid_input("n_estimators must be > 0")

// Or: direct enum construction
Err(FerroError::InvalidInput("expected 2D array".into()))
```

**Python Error Strategy:**
- Propagation: Rust errors become Python exceptions
- Testing: Check error messages contain relevant keywords
- Pattern in tests: `with pytest.raises(Exception)` + message assertion

**Example from `test_errors.py`:**
```python
def test_nan_in_features_raises(self):
    model = LinearRegression()
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(Exception) as exc_info:
        model.fit(X, y)

    error_msg = str(exc_info.value).lower()
    assert 'nan' in error_msg or 'invalid' in error_msg
```

## Logging

**Framework:** `tracing` crate (structured logging)

**Pattern:**
- Used in critical sections and diagnostics
- Not verbose in hot paths
- Levels: debug for algorithm steps, warn for assumptions, error for failures

## Comments & Documentation

**When to Comment (Rust):**
- Algorithm details with references (especially papers)
- Non-obvious design decisions
- Mathematical notation explanations
- Workarounds with issue references

**Doc Patterns:**
- Module-level: `//! ` with architecture diagrams and examples
- Items: `/// ` with doc-comments, reference papers, examples
- Inline: `// ` for implementation notes only

**Example from `adaboost.rs`:**
```rust
//! AdaBoost Ensemble Methods
//!
//! Implements AdaBoost (Adaptive Boosting) for classification and regression.
//!
//! ## References
//! - Hastie, Tibshirani, Friedman (2009). "Elements of Statistical Learning"
//! - Drucker (1997). "Improving Regressors using Boosting Techniques"

/// AdaBoost classifier using SAMME.R algorithm.
///
/// Fits an ensemble of weighted decision stumps, where each subsequent
/// estimator focuses on the samples that previous estimators got wrong.
///
/// ## Example
///
/// ```
/// let mut clf = AdaBoostClassifier::new(50);
/// clf.fit(&x, &y)?;
/// ```
pub struct AdaBoostClassifier { ... }
```

**Python Documentation:**
- Module docstrings: describe purpose and patterns
- Class docstrings: behavior and test focus
- Method docstrings: only when non-obvious

**Example from `conftest.py`:**
```python
"""
Shared fixtures for FerroML Python tests.

This module provides:
- Sample datasets (regression, classification, multiclass)
- Model factories for common test patterns
- Utility functions for testing
"""

@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simple regression data.

    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + noise

    Returns:
        Tuple of (X, y) with shape (100, 3) and (100,)
    """
```

## Function Design

**Size Guidelines (Rust):**
- Clippy allows `#![allow(clippy::too_many_lines)]` for complex algorithms
- Typical: 50-300 lines for algorithm implementations
- Preference: Break into helper functions by logical step

**Parameters:**
- Input arrays: `&Array2<f64>` (features), `&Array1<f64>` (targets)
- Output arrays: `Array2<f64>` or `Array1<f64>` (owned)
- Configs: builder pattern with `with_*()` methods
- No default parameters — use builders

**Return Values:**
- Success: `Result<T>` for public API
- Arrays: owned (Array2<f64>, Array1<f64>)
- References: for accessors (e.g., `estimators()` returns `Option<&[...]>`)

**Example:**
```rust
// Builder pattern for initialization
pub fn new(n_estimators: usize) -> Self { ... }
pub fn with_learning_rate(mut self, lr: f64) -> Self { ... }
pub fn with_max_depth(mut self, max_depth: usize) -> Self { ... }

// Fit/predict pattern
pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> { ... }
pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> { ... }

// Accessors
pub fn classes(&self) -> Option<&Array1<f64>> { ... }
pub fn estimator_weights(&self) -> Option<&Array1<f64>> { ... }
```

## Module Design

**Public Exports:**
- Traits and primary types: always public
- Helpers and validation: often public for library users
- Internals: private with `pub(crate)` when needed by other modules

**Barrel Files (re-exports):**
- Used in preprocessing and models submodules
- File: `/home/tlupo/ferroml/ferroml-core/src/preprocessing/mod.rs`
- Pattern: `pub use self::scalers::*; pub use self::encoders::*;`

**Python Re-exports:**
- Via `__init__.py` in submodules
- Example: `/home/tlupo/ferroml/ferroml-python/src` wraps all Rust types
- Bindings expose full sklearn-compatible API

## Constants & Magic Numbers

**Conventions:**
- Named constants for tuning parameters (e.g., `const DEFAULT_VARIANCE_SMOOTHING: f64 = 1e-9;`)
- Documented magic numbers in comments when needed
- Use `const` for compile-time known values
- Use `lazy_static` or runtime initialization for non-const values

## Trait Implementations

**Standard Traits:**
- `Debug`: Always derived
- `Clone`: Always derived for models (enables HPO re-runs)
- `Serialize`/`Deserialize`: Always derived (serde framework)
- `Default`: Implemented when sensible (e.g., `AdaBoostClassifier` default has n_estimators=50)

**ML-Specific Traits (in `traits.rs`):**
- `Model`: `fit()`, `predict()`, `is_fitted()`, `n_features()`, `model_name()`
- `Transformer`: `fit()`, `transform()`, `fit_transform()`, `inverse_transform()`
- `StatisticalModel`: adds `confidence_interval()`, `get_params()`
- `ProbabilisticModel`: adds `predict_proba()`

## Testing Conventions

**Inline Module Tests (Rust):**
- Location: module file with `#[cfg(test)] mod tests { }`
- Pattern: Simple unit tests for single functions
- Use `#[test]` attribute
- Example from `error.rs`:
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_error_creation() {
          let err = FerroError::shape_mismatch("expected", "actual");
          assert!(matches!(err, FerroError::ShapeMismatch { .. }));
      }
  }
  ```

**Integration Tests (Rust):**
- Location: `/home/tlupo/ferroml/ferroml-core/tests/` (separate binaries)
- Consolidated files: `correctness.rs`, `integration.rs`, `edge_cases.rs`, `adversarial.rs`, `vs_linfa.rs`, `regression_tests.rs`
- Each file is a single test binary (reduced from 19 to 6 files for faster builds)
- Run with: `cargo test --test correctness`

**Fuzzing:**
- Framework: libfuzzer-sys
- Location: `/home/tlupo/ferroml/ferroml-core/fuzz/fuzz_targets/`
- Test serialization, preprocessing, ONNX loading
- Run with: `cargo +nightly fuzz run <fuzzer_name>`

---

*Convention analysis: 2026-03-16*
