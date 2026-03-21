# Testing Patterns

**Analysis Date:** 2026-03-20

## Test Framework

**Rust Runner:**
- Framework: `cargo test` (built-in libtest harness)
- Config: No explicit test config (uses Cargo.toml test settings)
- Note: Tests run with `--lib` by default to skip integration tests during dev (faster)

**Python Runner:**
- Framework: pytest (invoked via `pytest ferroml-python/tests/`)
- Config: `ferroml-python/pyproject.toml` with pytest sections
- pytest plugins: parametrize, fixtures

**Assertion Library:**
- Rust: `assert!()`, `assert_eq!()` macros + `approx` crate for float comparisons
- Python: `np.testing.assert_allclose()`, `assert` statements, pytest's `assert`

**Run Commands:**

```bash
# Rust - all tests
cargo test --all

# Rust - lib only (fast, pre-commit default)
cargo test --lib

# Rust - integration tests only
cargo test --test correctness
cargo test --test edge_cases
cargo test --test integration
cargo test --test vs_linfa
cargo test --test regression_tests
cargo test --test adversarial

# Rust - with ignored tests
cargo test -- --ignored

# Python - all tests
source .venv/bin/activate
pytest ferroml-python/tests/

# Python - specific file
pytest ferroml-python/tests/test_bindings_correctness.py -v

# Python - specific test class
pytest ferroml-python/tests/test_bindings_correctness.py::TestArrayConversionFidelity -v
```

## Test File Organization

**Location - Rust:**
- Unit tests: inline in source files (`#[cfg(test)]` module at end)
- Integration tests: `/ferroml-core/tests/{feature}.rs` (6 consolidated files, see below)

**Location - Python:**
- All tests: `/ferroml-python/tests/test_*.py`
- Patterns: `test_bindings_correctness.py`, `test_comparison_*.py`, `test_vs_*.py`

**Naming - Rust:**
- Test functions: `test_{subject}_{scenario}()` or `test_{feature}()`
- Examples: `test_kmeans_blobs_finds_correct_clusters()`, `test_linear_regression_simple()`
- Modules: lowercase `mod module_name { #[test] fn test_... }`

**Naming - Python:**
- Test functions: `test_{scenario}()` (lowercase)
- Test classes: `Test{Feature}` (PascalCase, e.g., `TestLinearRegressionComparison`)
- Fixtures: lowercase with underscores (`regression_data`, `classification_data`)

**Structure - Test Discovery:**
```
ferroml-core/tests/          # Integration tests (6 files)
├── correctness.rs           # Cross-library validation
├── adversarial.rs           # Adversarial inputs
├── edge_cases.rs            # Degenerate cases
├── integration.rs           # E2E pipelines, serialization
├── regression_tests.rs      # Plan-by-plan regression
└── vs_linfa.rs              # linfa cross-validation

ferroml-core/src/            # Unit tests inline
├── models/
│   ├── linear.rs            # #[cfg(test)] mod tests {...}
│   └── forest.rs            # #[cfg(test)] mod tests {...}
└── stats/
    └── hypothesis.rs        # #[cfg(test)] mod tests {...}

ferroml-python/tests/        # Python tests
├── conftest.py              # Shared fixtures
├── test_bindings_correctness.py
├── test_comparison_linear.py
├── test_comparison_trees.py
└── ... (12+ comparison/cross-library files)
```

## Test Structure

**Rust Suite Organization:**

```rust
//! Module-level doc comment explaining test scope

mod clustering {
    //! Submodule doc comment for organization

    use ferroml_core::clustering::{...};  // Imports at top
    use ndarray::{array, Array1, Array2};
    use std::collections::HashSet;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Helper function with doc comment
    fn make_blobs_3c() -> (Array2<f64>, Array1<i32>) {
        // Generates test data with known structure
        let x = Array2::from_shape_vec((15, 2), vec![...]).unwrap();
        let labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
        (x, labels)
    }

    /// Check test invariants
    fn clusterings_agree(a: &Array1<i32>, b: &Array1<i32>) -> bool {
        let ari = adjusted_rand_index(a, b).unwrap();
        ari > 0.99
    }

    // =========================================================================
    // Feature Tests
    // =========================================================================

    #[test]
    fn test_kmeans_blobs_finds_correct_clusters() {
        let (x, true_labels) = make_blobs_3c();
        let mut kmeans = KMeans::new(3).random_state(42).n_init(5);
        kmeans.fit(&x).unwrap();

        let pred_labels = kmeans.labels().unwrap();
        assert!(
            clusterings_agree(&true_labels, pred_labels),
            "KMeans labels do not match ground truth (ARI={:.4})",
            adjusted_rand_index(&true_labels, pred_labels).unwrap()
        );
    }
}
```

**Key Patterns:**
- Section headers with `// =========================================================================`
- Helper functions with descriptive names
- Assertions with detailed error messages (use format strings for debugging)
- Use builder pattern for model instantiation: `.with_param(value)`
- Always specify `random_state` for reproducibility

**Python Suite Organization:**

```python
"""
Module docstring explaining test scope and cross-library coverage.

Tests all X models on real and synthetic datasets, comparing
predictions, coefficients, R², and accuracy within documented tolerances.
"""

import numpy as np
import pytest
from conftest import regression_data, classification_data  # Import fixtures

class TestLinearRegressionComparison:
    """FerroML vs sklearn LinearRegression comparison."""

    def _fit_both(self, X, y):
        """Helper: fit FerroML and sklearn versions, return both."""
        from ferroml.linear import LinearRegression as FerroLR
        from sklearn.linear_model import LinearRegression as SkLR

        ferro = FerroLR()
        ferro.fit(X, y)
        sk = SkLR()
        sk.fit(X, y)
        return ferro, sk

    def test_diabetes_predictions(self):
        """Predictions match sklearn within tolerance."""
        X, y = get_diabetes()  # from conftest_comparison
        ferro, sk = self._fit_both(X, y)
        fp = ferro.predict(X)
        sp = sk.predict(X)
        np.testing.assert_allclose(fp, sp, atol=1e-6)

    @pytest.mark.parametrize("n_samples,n_features", [
        (100, 5),
        (1000, 20),
        (10000, 100),
    ])
    def test_varying_shapes(self, n_samples, n_features):
        """Test across various data shapes."""
        X = np.random.randn(n_samples, n_features)
        y = X @ np.random.randn(n_features) + 0.01 * np.random.randn(n_samples)
        ferro, sk = self._fit_both(X, y)
        np.testing.assert_allclose(ferro.predict(X), sk.predict(X), rtol=1e-4)
```

## Mocking

**Framework:**
- Rust: No external mock library; tests use in-memory data structures
- Python: `unittest.mock` (built-in) or pytest fixtures for test isolation

**Patterns - Rust:**
- Create minimal valid inputs: `make_blobs_3c()` generates known-good test data
- Avoid actual I/O: use `Array2::from_shape_vec()` with in-memory vectors
- Don't mock traits; test concrete implementations
- Use `random_state` to make stochastic algorithms deterministic

**Patterns - Python:**
- Fixtures in `conftest.py` provide reusable datasets (see lines 20-85)
- No mocking of FerroML models (test real behavior)
- Mock sklearn/linfa only if testing cross-library behavior with special conditions
- Example: `regression_data` fixture (lines 20-34) creates y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + noise

**What to Mock:**
- External libraries in cross-validation tests (sklearn, linfa, xgboost)
- Example: `conftest_comparison.py` has factory functions: `get_iris()`, `get_diabetes()`, `get_regression_data()`

**What NOT to Mock:**
- FerroML model behavior (test real implementations)
- Array conversions (test actual NumPy ↔ Rust boundaries)
- Serialization (pickle real objects, verify roundtrip)

## Fixtures and Factories

**Rust Test Data:**

See `ferroml-core/tests/correctness.rs` (lines 22-105) for patterns:

```rust
/// Generate well-separated 2D blob data with known cluster structure.
/// Returns (X, true_labels) where clusters are at:
///   cluster 0: center (0, 0)
///   cluster 1: center (10, 10)
///   cluster 2: center (10, 0)
fn make_blobs_3c() -> (Array2<f64>, Array1<i32>) {
    let x = Array2::from_shape_vec(
        (15, 2),
        vec![
            // Cluster 0 near (0, 0)
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.0,
            // Cluster 1 near (10, 10)
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, 10.1, 10.1, 9.9, 10.0,
            // Cluster 2 near (10, 0)
            10.0, 0.0, 10.1, 0.0, 10.0, 0.1, 10.1, 0.1, 9.9, 0.0,
        ],
    )
    .unwrap();
    let labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
    (x, labels)
}

/// Generate high-dimensional blob data
fn make_blobs_high_dim(n_features: usize) -> (Array2<f64>, Array1<i32>) {
    let n_per_cluster = 10;
    let n_clusters = 3;
    let n_total = n_per_cluster * n_clusters;
    let mut data = vec![0.0; n_total * n_features];

    for c in 0..n_clusters {
        for i in 0..n_per_cluster {
            let row = c * n_per_cluster + i;
            for j in 0..n_features {
                data[row * n_features + j] =
                    (c as f64) * 10.0 + ((i * 7 + j * 3) % 10) as f64 * 0.01;
            }
        }
    }
    let labels = Array1::from_vec(
        (0..n_clusters)
            .flat_map(|c| vec![c as i32; n_per_cluster])
            .collect(),
    );
    (Array2::from_shape_vec((n_total, n_features), data).unwrap(), labels)
}
```

**Python Fixtures:**

See `ferroml-python/tests/conftest.py` (lines 20-100):

```python
@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simple regression data.

    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + noise

    Returns:
        Tuple of (X, y) with shape (100, 3) and (100,)
    """
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def classification_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simple binary classification data.

    Linearly separable with decision boundary at X[:, 0] + X[:, 1] = 0

    Returns:
        Tuple of (X, y) with shape (100, 3) and (100,)
    """
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y
```

**Location:**
- Rust: Helper functions at top of each test module
- Python: `conftest.py` for shared fixtures, `conftest_comparison.py` for cross-library helpers

## Coverage

**Requirements:** No enforced coverage target, but >90% achieved on core modules

**View Coverage:**

```bash
# Rust - basic coverage (requires tarpaulin or llvm-cov)
cargo tarpaulin --out Html

# Python - coverage with pytest-cov
pip install pytest-cov
pytest ferroml-python/tests/ --cov=ferroml --cov-report=html
open htmlcov/index.html
```

**Coverage Focus:**
- Core model algorithms: >95%
- Edge cases: 100% (by design, see edge_cases.rs)
- Cross-library validation: 100% (representative models)
- Error paths: >90%

## Test Types

**Unit Tests:**
- Scope: Single function or method in isolation
- Approach: Fast, deterministic, in-process
- Location: Inline in source files with `#[cfg(test)]` modules
- Example: `ferroml-core/src/stats/hypothesis.rs` tests individual hypothesis tests
- Run time: < 1 second per test
- Emphasis: Mathematical correctness, edge cases

**Integration Tests:**
- Scope: Multiple components together (fit → predict → diagnostics, etc.)
- Approach: End-to-end workflows, realistic data, cross-library comparison
- Location: `/ferroml-core/tests/{feature}.rs`
- Files:
  - `correctness.rs` — 200+ tests, covers all major models vs sklearn/linfa fixtures
  - `edge_cases.rs` — 13-scenario matrix per model (single sample, high-dim, NaN, etc.)
  - `integration.rs` — Pipelines, CV, serialization end-to-end
  - `regression_tests.rs` — Plan-by-plan regression (ensures Plans A-X still pass)
  - `adversarial.rs` — Robustness to pathological inputs
  - `vs_linfa.rs` — Cross-validation against linfa implementations
- Run time: ~30-60 seconds total
- Emphasis: Numerical agreement with other libraries, statistical correctness

**Cross-Library Tests:**
- Scope: FerroML vs sklearn, xgboost, lightgbm, linfa, scipy, statsmodels
- Approach: Train both, compare predictions/coefficients/scores
- Location: Python: `ferroml-python/tests/test_vs_sklearn_gaps_phase2.py`, `test_comparison_*.py` (12+ files)
- Tolerance: Documented in each test (e.g., `atol=1e-6` for coefficients, `rtol=1e-4` for predictions)
- Pre-existing failures (known, acceptable):
  - `TemperatureScaling` — sklearn uses different calibration approach
  - `IncrementalPCA` — partial fit not yet implemented

**Python Bindings Tests:**
- Scope: PyO3 boundary correctness (NumPy ↔ Rust, state management, pickle)
- File: `ferroml-python/tests/test_bindings_correctness.py` (30 tests)
- Classes:
  - `TestArrayConversionFidelity` — float64 precision, C/F contiguous arrays
  - `TestErrorPropagation` — Rust errors → Python exceptions
  - `TestStateManagement` — Fit/predict order enforcement, refitting
  - `TestSerialization` — pickle roundtrip fidelity
  - `TestThreadSafety` — concurrent predict calls
- Emphasis: Array safety, error handling, state consistency

## Common Patterns

**Async Testing (Rust):**
- No async in FerroML core (all blocking algorithms)
- Parallelism via rayon within algorithms, not async/await

**Error Testing (Rust):**

```rust
#[test]
fn test_predict_before_fit_fails() {
    let mut model = LinearRegression::new();
    let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    // Should return NotFitted error
    let result = model.predict(&x);
    assert!(result.is_err());

    match result {
        Err(FerroError::NotFitted { operation }) => {
            assert_eq!(operation, "predict");
        }
        _ => panic!("Expected NotFitted error"),
    }
}
```

**Error Testing (Python):**

```python
def test_predict_before_fit_raises():
    """predict() before fit() raises informative error."""
    m = LinearRegression()
    X = np.random.randn(5, 3)

    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict(X)
```

**Parametrized Tests (Python):**

```python
@pytest.mark.parametrize("n_estimators,max_depth", [
    (10, 5),
    (50, 10),
    (100, None),
])
def test_forest_hyperparams(n_estimators, max_depth):
    """Test across hyperparameter combinations."""
    X, y = regression_data()
    m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    m.fit(X, y)
    pred = m.predict(X)
    assert len(pred) == len(y)
```

**Floating Point Comparisons:**

```rust
// Use approx crate
use approx::assert_relative_eq;

#[test]
fn test_coefficients_relative_tolerance() {
    // ...
    assert_relative_eq!(predicted, expected, max_relative = 1e-5);
}
```

```python
# Use np.testing
np.testing.assert_allclose(predicted, expected, rtol=1e-5, atol=1e-8)
```

**Slow Tests:**

```rust
#[test]
#[ignore]  // Skip in normal test runs
fn test_automl_slow() {
    // Takes 10+ seconds
    // Run with: cargo test -- --ignored
}
```

```python
@pytest.mark.slow
def test_automl_slow():
    """Takes 10+ seconds, skip in CI."""
    # Run with: pytest -m slow
```

## Test Coverage Strategy

**Coverage Matrix:**
- **Correctness**: Every model has at least 5 tests in `correctness.rs`
- **Edge cases**: 13 scenarios per model in `edge_cases.rs` (single sample, high-dim, NaN, etc.)
- **Cross-library**: All major models compared vs sklearn/linfa in Python tests
- **Bindings**: 30 tests in `test_bindings_correctness.py` covering PyO3 boundary
- **Regression**: Every plan (Plans A-X) has at least one regression test in `regression_tests.rs`

**Skipped/Known Failures:**
- 26 tests ignored (marked `#[ignore]`, slow AutoML benchmarks)
- 6 pre-existing failures in `test_vs_sklearn_gaps_phase2.py` (TemperatureScaling, IncrementalPCA — documented as acceptable)

---

*Testing analysis: 2026-03-20*
