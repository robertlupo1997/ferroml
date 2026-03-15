# Testing Patterns

**Analysis Date:** 2026-03-15

## Test Framework

**Rust Test Runner:**
- Framework: Built-in `cargo test` (no external test framework needed)
- Config: `ferroml-core/Cargo.toml` defines test dependencies
- Run Commands:
```bash
cargo test -p ferroml-core --lib              # Unit tests only
cargo test -p ferroml-core                    # Unit + integration tests
cargo test -p ferroml-core -- --test-threads=1  # Serial execution (debugging)
cargo test -- --ignored                       # Run ignored tests (slow suite)
```

**Python Test Runner:**
- Framework: `pytest` (installed in `.venv`)
- Config: `ferroml-python/tests/conftest.py` provides shared fixtures
- Run Commands:
```bash
cd ferroml-python && python -m pytest tests/              # All tests
python -m pytest tests/test_clustering.py -v             # Single file, verbose
python -m pytest tests/test_clustering.py::TestKMeans    # Single class
python -m pytest -k "score" --tb=short                   # Filter by name
```

**Assertion Libraries:**
- Rust: Built-in `assert!`, `assert_eq!` + `approx` crate for floats
- Python: Built-in `assert` + `numpy.testing` utilities

## Test File Organization

**Rust Location Patterns:**
- Unit tests: `#[cfg(test)] mod tests { ... }` in source files (e.g., `src/models/linear.rs`)
- Integration tests: Separate `.rs` files in `/tests/` directory (e.g., `tests/correctness_preprocessing.rs`)
- Helper modules: `src/testing/` with reusable check functions (e.g., `assertions.rs`, `checks.rs`)

**Python Location Patterns:**
- Test files: `ferroml-python/tests/test_*.py` (pytest convention)
- Fixtures: `conftest.py` provides shared datasets and utilities
- Test classes: Grouped by feature (e.g., `TestKMeans`, `TestClassifierScore`)

**File Naming:**
- Rust integration: `[feature_area].rs` (e.g., `correctness_preprocessing.rs`, `sklearn_correctness.rs`)
- Cross-library: `vs_[library].rs` (e.g., `vs_linfa_linear.rs`, `vs_linfa_clustering.rs`)
- Python: `test_[feature].py` (e.g., `test_clustering.py`, `test_partial_fit.py`)

**Rust Directory Tree:**
```
ferroml-core/
├── src/
│   ├── testing/                    # Shared test utilities
│   │   ├── assertions.rs          # Tolerance constants and macros
│   │   ├── checks.rs              # 30+ estimator validation checks
│   │   ├── serialization.rs       # Model roundtrip testing
│   │   ├── probabilistic.rs       # Probability calibration tests
│   │   └── ...
│   ├── models/
│   │   ├── linear.rs
│   │   │   └── #[cfg(test)] mod tests { ... }  # unit tests
│   │   └── ...
│   └── ...
├── tests/
│   ├── correctness_preprocessing.rs     # 101 preprocessing tests
│   ├── correctness_clustering.rs        # KMeans, DBSCAN, hierarchical
│   ├── sklearn_correctness.rs           # sklearn fixture comparisons
│   ├── vs_linfa_linear.rs               # linfa cross-library validation
│   ├── vs_linfa_clustering.rs           # linfa vs FerroML parity
│   └── ...
└── benches/
    ├── benchmarks.rs                    # Criterion benchmarks
    ├── memory_benchmarks.rs
    └── gpu_benchmarks.rs
```

**Python Directory Tree:**
```
ferroml-python/tests/
├── conftest.py                          # Shared fixtures
├── test_clustering.py                   # KMeans, DBSCAN, hierarchical
├── test_score_all_models.py             # score() method across 55+ models
├── test_partial_fit.py                  # incremental learning
├── test_decision_function.py            # decision_function() for classifiers
├── test_vs_sklearn.py                   # sklearn comparison tests
└── ...
```

## Test Structure

**Rust Suite Organization:**

```rust
//! Module documentation explaining what is tested

use ferroml_core::preprocessing::Transformer;
use ndarray::{array, Array1, Array2};

// =============================================================================
// Helper functions
// =============================================================================

/// Assert two f64 values approximately equal within tolerance
fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: expected {}, got {}, diff = {}",
        msg,
        expected,
        actual,
        (actual - expected).abs()
    );
}

/// Assert two Array2<f64> values approximately equal element-wise
fn assert_array2_approx(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64, msg: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{}: shape mismatch", msg);
    for ((i, j), &a) in actual.indexed_iter() {
        let e = expected[[i, j]];
        assert!(
            (a - e).abs() < tol,
            "{}: at [{},{}] expected {}, got {}",
            msg, i, j, e, a
        );
    }
}

// =============================================================================
// Test Section (organized by feature)
// =============================================================================

#[test]
fn polynomial_features_degree2_two_features() {
    // Setup
    use ferroml_core::preprocessing::polynomial::PolynomialFeatures;
    let mut poly = PolynomialFeatures::new(2);
    let x = array![[1.0, 2.0], [3.0, 4.0]];

    // Execute
    let result = poly.fit_transform(&x).unwrap();

    // Assert (multiple assertions per test is OK for feature validation)
    assert_eq!(result.ncols(), 6, "degree=2 with 2 features => 6 output cols");
    assert_eq!(result.nrows(), 2);
    let expected = array![
        [1.0, 1.0, 2.0, 1.0, 2.0, 4.0],
        [1.0, 3.0, 4.0, 9.0, 12.0, 16.0]
    ];
    assert_array2_approx(&result, &expected, 1e-10, "poly_degree2");
}

#[ignore]  // Mark slow or broken tests
#[test]
fn slow_automl_system_test() {
    // This test takes >30 seconds - only run manually
    // Reason: exhaustive hyperparameter search
}
```

**Python Suite Organization:**

```python
"""Tests that every model exposes score(X, y) and returns sensible values."""

import numpy as np
import pytest
from typing import Tuple

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def reg_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple regression data. y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + noise"""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y

@pytest.fixture
def clf_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple binary classification data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y

# =============================================================================
# Helpers
# =============================================================================

def _fit_and_score_clf(model, X, y):
    """Fit a classifier and return score. Score should be in [0, 1]."""
    model.fit(X, y)
    s = model.score(X, y)
    assert isinstance(s, float), f"{type(model).__name__}: score returned {type(s)}"
    assert 0.0 <= s <= 1.0, f"{type(model).__name__}: accuracy={s} not in [0,1]"
    return s

# =============================================================================
# Test Classes
# =============================================================================

class TestClassifierScore:
    def test_logistic_regression(self, clf_data):
        from ferroml.linear import LogisticRegression
        _fit_and_score_clf(LogisticRegression(), *clf_data)

    def test_ridge_classifier(self, clf_data):
        from ferroml.linear import RidgeClassifier
        _fit_and_score_clf(RidgeClassifier(), *clf_data)

class TestRegressorScore:
    def test_linear_regression(self, reg_data):
        from ferroml.linear import LinearRegression
        model = LinearRegression()
        model.fit(*reg_data)
        r2 = model.score(*reg_data)
        assert r2 <= 1.0, f"R² should be ≤ 1, got {r2}"
```

**Patterns:**
- Setup/execute/assert structure (AAA pattern)
- Helper functions for common operations
- Fixtures for shared data
- Use `#[ignore]` for slow tests (30+ seconds)
- Python: Use pytest classes to group related tests
- Rust: Use comment sections with `// =====` to organize

## Mocking

**Framework:**
- Rust: No mocking framework; use trait-based testing instead
- Python: `unittest.mock` built-in (used sparingly)

**Patterns:**

**Rust - Trait Substitution (Preferred):**
```rust
// Instead of mocking, create a test-specific implementation
#[cfg(test)]
mod tests {
    use super::*;

    struct MockPreprocessor;

    impl Transformer for MockPreprocessor {
        fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
            // Return controlled result
            Ok(())
        }

        fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
            Ok(x.clone())  // Identity transform for testing
        }
    }
}
```

**Python - unittest.mock (Rarely Used):**
```python
from unittest.mock import Mock, patch

# Mock external service
with patch('ferroml.models.external_model') as mock_model:
    mock_model.return_value = 0.95
    # Test code
```

**What to Mock:**
- External API calls (rare in ferroml-core; more in bindings)
- Heavy computation for unit tests (prefer creating small datasets)
- Random number generation for reproducibility

**What NOT to Mock:**
- Model internals (test actual computation)
- Array operations (test with real arrays)
- Preprocessing steps (test full pipeline)

## Fixtures and Factories

**Rust Fixtures (rstest macro):**
```rust
use rstest::rstest;

#[rstest]
#[case::degree_2([1.0, 2.0], 6)]           // name, input, expected_cols
#[case::degree_3([1.0], 4)]
fn test_polynomial_output_size(#[case] input: [f64; 1], #[case] expected_cols: usize) {
    // Test runs twice with different parameters
}
```

**Python Fixtures (pytest):**
```python
@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
    return X, y

@pytest.fixture
def iris_like_data():
    """Generate iris-like dataset (3 classes, 4 features)."""
    np.random.seed(42)
    n_per_class = 50
    X_list = []
    y_list = []

    # Class 0
    X_list.append(np.random.randn(n_per_class, 4) * 0.5)
    y_list.append(np.zeros(n_per_class))

    # Class 1
    X_list.append(np.random.randn(n_per_class, 4) * 0.5 + 2)
    y_list.append(np.ones(n_per_class))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = np.random.permutation(len(y))
    return X[idx], y[idx]
```

**Test Data Location:**
- Rust: Defined inline in test files or in helper functions
- Python: Fixtures in `conftest.py` for cross-file reuse
- Real datasets: `ferroml-core/datasets/` module (loaded at runtime)

**Dataset Patterns:**
- Synthetic blob data: `make_blobs_3c()`, `make_blobs_2c()` for clustering
- Regression data: `y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + noise` (linear with known coefs)
- Classification data: `y = (X[:, 0] + X[:, 1] > 0)` (linear boundary)
- Multiclass: 3 classes, 50 samples each, gaussian clouds

## Coverage

**Requirements:** Not enforced (no explicit coverage target)

**View Coverage (Python):**
```bash
pip install pytest-cov
pytest --cov=ferroml_python --cov-report=html tests/
# View in htmlcov/index.html
```

**View Coverage (Rust):**
```bash
# Using tarpaulin
cargo install cargo-tarpaulin
cargo tarpaulin --out Html --output-dir coverage/
```

**Coverage Strategy:**
- Aim for >80% on core algorithms
- 100% coverage not required (test boundary cases, not defensive coding)
- Skip coverage for: error handling branches, deprecated features, GPU fallbacks

## Test Types

**Unit Tests:**
- Scope: Single function or method
- Location: `#[cfg(test)]` module in source file (Rust) or same package (Python)
- Example: `test_linear_regression_fit_basic()` tests just the fit() method
- Pattern: Setup minimal data, call one method, assert output
- Rust location: `src/models/linear.rs` contains `#[cfg(test)] mod tests { ... }`

**Integration Tests:**
- Scope: Multiple components working together (e.g., preprocessor → model → metric)
- Location: `/tests/*.rs` (Rust) or `tests/test_integration.py` (Python)
- Example: `test_pipeline_preprocessing_then_fit()` tests scaler + regression
- Pattern: Use realistic data, full end-to-end workflow, assert final output
- Rust location: `ferroml-core/tests/correctness_preprocessing.rs` (101 tests)

**Cross-Library Validation Tests:**
- Scope: Compare FerroML against sklearn/linfa/scipy reference implementations
- Location: `tests/vs_*.rs` (Rust) or `tests/test_vs_*.py` (Python)
- Purpose: Verify correctness against battle-tested implementations
- Pattern: Load sklearn fixture → run in FerroML → assert within sklearn tolerance
- Rust files: `vs_linfa_linear.rs`, `vs_linfa_clustering.rs`, `vs_linfa_svm.rs` (6 files, 56 tests)
- Python files: `test_vs_sklearn.py`, `test_vs_xgboost.py`, `test_vs_statsmodels.py` (5+ files, 108 tests)

**E2E Tests:**
- Scope: Full application flow (rarely used in ferroml-core)
- Status: Not used for library testing
- Note: AutoML system tests exist but marked `#[ignore]` (too slow for CI)

## Common Patterns

**Async Testing (Not Used):**
- FerroML is synchronous (no async/await)
- No async test patterns

**Error Testing - Rust:**
```rust
#[test]
fn test_shape_mismatch_error() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = array![1.0, 2.0, 3.0];  // wrong shape

    let mut model = LinearRegression::new();
    match model.fit(&x, &y) {
        Err(FerroError::ShapeMismatch { .. }) => {
            // Success: got expected error
        }
        Err(e) => panic!("Expected ShapeMismatch, got {:?}", e),
        Ok(()) => panic!("Expected error, fit succeeded"),
    }
}
```

**Error Testing - Python:**
```python
def test_shape_mismatch_error(clf_data):
    from ferroml.linear import LogisticRegression
    X, y = clf_data
    X_wrong = X[:, :1]  # wrong number of features

    model = LogisticRegression()
    with pytest.raises(RuntimeError, match="shape"):
        model.fit(X_wrong, y)
```

**Floating-Point Comparison - Rust:**
```rust
use ferroml_core::testing::assertions::{tolerances, assert_approx_eq, assert_array_approx_eq};

// Use calibrated tolerance for algorithm type
assert_approx_eq!(coef, expected_coef, tolerances::CLOSED_FORM);
assert_array_approx_eq!(predictions, expected_preds, tolerances::SKLEARN_COMPAT);
```

**Floating-Point Comparison - Python:**
```python
import numpy as np

# numpy testing utilities
np.testing.assert_allclose(predictions, expected, rtol=1e-5, atol=1e-8)
```

**Tolerance Constants (Defined in `src/testing/assertions.rs`):**
- `CLOSED_FORM = 1e-10` - Linear regression, QR decomposition, direct solve
- `ITERATIVE = 1e-4` - Gradient descent, coordinate descent, IRLS
- `TREE = 1e-12` - Decision trees, random forests (very tight)
- `PROBABILISTIC = 1e-2` - Sampling-based, MCMC
- `NEURAL = 1e-3` - Neural networks
- `SKLEARN_COMPAT = 1e-5` - Comparing against sklearn reference
- `SCALER = 1e-10` - StandardScaler, normalization
- `DECOMPOSITION = 1e-8` - PCA, SVD
- `DISTANCE = 1e-10` - KNN, clustering distances
- `PROBABILITY = 1e-6` - Probability predictions (should sum to 1)
- `METRIC = 1e-10` - Evaluation metrics (R², MSE, accuracy)

## Test Utilities

**Rust Estimation Checks (`src/testing/checks.rs`):**
```rust
use ferroml_core::testing::check_estimator;

// 30+ individual checks for Model contract
pub fn check_not_fitted<M: Model>(model: &M) -> CheckResult
pub fn check_n_features_in<M: Model>(model: &M) -> CheckResult
pub fn check_output_shapes<M: Model>(model: &M) -> CheckResult
pub fn check_array_mutability<M: Model>(model: &M) -> CheckResult
pub fn check_no_nan_predictions<M: Model>(model: &M) -> CheckResult
// ... more checks
```

**Rust Assertion Macros (`src/testing/assertions.rs`):**
```rust
// Use in tests
assert_approx_eq!(actual, expected, tolerances::CLOSED_FORM);
assert_array_approx_eq!(actual_array, expected_array, tolerances::SKLEARN_COMPAT);
```

**Rust Serialization Testing (`src/testing/serialization.rs`):**
```rust
check_model_serialization(
    LinearRegression::new(),
    SerializationTestConfig::default(),
)
```

**Python Test Helpers (`ferroml-python/tests/conftest.py`):**
- `regression_data`: Simple linear regression fixture
- `classification_data`: Binary classification fixture
- `multiclass_data`: 3-class classification fixture
- `iris_like_data`: Iris dataset replica
- `diabetes_like_data`: Regression with 10 features

## Pre-commit Test Hook

**What Runs:**
```bash
cargo test -p ferroml-core --lib -- --test-threads=4
```

**Timeout:** Must complete in ~30 seconds (pre-commit timeout)

**Excluded:**
- Integration tests in `/tests/` (too slow)
- Ignored tests (marked `#[ignore]`)
- GPU tests (require GPU)
- Cross-library tests (optional dependencies)

**To Run Full Suite:**
```bash
cargo test -p ferroml-core                    # All tests
cargo test -p ferroml-core -- --ignored       # Slow tests
cargo test -p ferroml-python                  # Python tests (requires .venv)
```

## Known Test Issues

**Pre-existing Failures:**
- 1 ONNX roundtrip test failure (RandomForestClassifier, tracked)
- RidgeCV `predict()` returns NaN in some cases (known issue, pre-existing)

**Ignored Tests:**
- 26 ignored tests (slow AutoML system tests, >30 seconds each)
- Run with: `cargo test -- --ignored --test-threads=1`

## Test Authorship Best Practices

**When Adding Tests:**
1. Follow existing structure (section comments, helper functions)
2. Use calibrated tolerance constants, not hardcoded values
3. Name tests descriptively: `test_[feature]_[scenario]_[expected_result]`
4. Document non-obvious test logic with comments
5. Use fixtures for any data used by multiple tests
6. Prefer integration tests for feature validation
7. Mark slow tests (>5 seconds) with `#[ignore]`
8. Ensure all Rust tests pass `cargo fmt` and `cargo clippy`
9. Ensure all Python tests pass `pytest` and follow PEP 8

---

*Testing analysis: 2026-03-15*
