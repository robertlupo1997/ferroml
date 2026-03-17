# Testing Patterns

**Analysis Date:** 2026-03-16

## Test Framework

### Rust Testing

**Runner:**
- Built-in `cargo test` (standard Rust test framework)
- Harness: false for benchmarks (Criterion)

**Assertion Library:**
- Built-in `assert!`, `assert_eq!`, `assert_ne!`
- Cross-library: `approx` crate for floating-point comparisons
- Array assertions: ndarray `.assert_*()` methods and custom helpers

**Run Commands:**
```bash
cargo test                           # Run all tests in workspace (default = ferroml-core)
cargo test --lib                     # Library tests only
cargo test --test integration        # Integration test binary
cargo test --test correctness        # Correctness test binary
cargo test --all                     # All workspace members
cargo test -- --nocapture            # Show println! output
cargo test -- --test-threads=1       # Single-threaded (determinism)
cargo test -- --ignored              # Run ignored (slow) tests only
```

**Test Files Location:**
- Inline tests: `src/**/*.rs` with `#[cfg(test)] mod tests { }`
- Integration tests: `/home/tlupo/ferroml/ferroml-core/tests/`
  - `correctness.rs` — Major functionality validation (~3,500 tests)
  - `integration.rs` — Cross-module workflows
  - `edge_cases.rs` — Boundary conditions and degenerate cases
  - `adversarial.rs` — Pathological inputs and adversarial data
  - `vs_linfa.rs` — Cross-library validation (linfa 0.8.1)
  - `regression_tests.rs` — Regression tracking for known issues

### Python Testing

**Runner:**
- pytest (pytest framework)
- Configuration: implicit (no pytest.ini, uses defaults)

**Run Commands:**
```bash
pytest ferroml-python/tests/                    # Run all Python tests
pytest ferroml-python/tests/test_naive_bayes.py # Single module
pytest ferroml-python/tests/test_naive_bayes.py::TestGaussianNB::test_fit_predict_basic # Single test
pytest -v                                       # Verbose output
pytest -k "gaussian"                            # Filter by name pattern
pytest --tb=short                               # Brief tracebacks
pytest -m "not slow"                            # Skip slow tests
pytest --co                                     # List tests without running
pytest -x                                       # Stop on first failure
pytest --pdb                                    # Drop to debugger on failure
```

**Test Files Location:**
- `/home/tlupo/ferroml/ferroml-python/tests/`
- 60+ test files, ~2,100 passing tests
- Patterns:
  - `test_*.py` — Main test modules
  - `test_comparison_*.py` — Cross-library validation (sklearn, xgboost, lightgbm, statsmodels)
  - `test_vs_*.py` — Library-specific comparison
  - `conftest.py` — Shared fixtures and configuration
  - `conftest_comparison.py` — Comparison helper fixtures

## Test File Organization

### Rust Structure

**Location Pattern:**
- Inline: Same file as implementation (`src/models/adaboost.rs` contains mod tests)
- Integration: Separate binary files in `/home/tlupo/ferroml/ferroml-core/tests/`

**File Size & Scope:**
- Consolidated to 6 main test binaries (reduced from 19 for faster builds)
- `correctness.rs`: 252KB (~3,500 tests) — All major model tests
- `integration.rs`: 92KB — Cross-module workflows
- `edge_cases.rs`: 92KB — Boundary conditions
- `adversarial.rs`: 51KB — Adversarial inputs
- `vs_linfa.rs`: 49KB — Cross-library validation
- `regression_tests.rs`: 37KB — Known issue tracking

**Test Naming:**
```rust
mod clustering {
    mod tests {
        fn make_blobs_3c() -> (Array2<f64>, Array1<i32>) { ... }

        #[test]
        fn test_kmeans_basic() { ... }

        #[test]
        fn test_kmeans_convergence() { ... }
    }
}
```

### Python Structure

**Location Pattern:**
- Test files: `/home/tlupo/ferroml/ferroml-python/tests/test_*.py`
- Fixtures: `/home/tlupo/ferroml/ferroml-python/tests/conftest.py`
- Comparison helpers: `/home/tlupo/ferroml/ferroml-python/tests/conftest_comparison.py`

**File Organization:**
```python
"""Module docstring explaining test scope."""

import numpy as np
import pytest
from ferroml.module import Class

# ============================================================================
# Section Comment
# ============================================================================

class TestClassName:
    """Test group for a specific component."""

    def test_basic_functionality(self):
        """Test name describing expected behavior."""
        ...

    def test_edge_case_description(self):
        """Specific edge case with setup."""
        ...

# ============================================================================
# Another Section
# ============================================================================

@pytest.fixture
def fixture_name():
    """Fixture docstring."""
    ...

class TestAnotherClass:
    ...
```

**Test Naming:**
```python
class TestGaussianNB:
    def test_fit_predict_basic(self):
        """Basic fit/predict workflow."""

    def test_predict_proba_sums_to_one(self):
        """Mathematical invariant: probabilities sum to 1."""

    def test_predict_proba_values_in_range(self):
        """Probability bounds check: 0 <= p <= 1."""

    def test_var_smoothing_effect(self):
        """Test variance smoothing parameter effect."""
```

## Test Suite Structure

### Rust Suite Organization

**Module Grouping Pattern:**
```rust
// File: ferroml-core/tests/correctness.rs

mod clustering {
    //! Clustering module tests
    use ferroml_core::clustering::{...};

    // Helper functions first
    fn make_blobs_3c() -> (Array2<f64>, Array1<i32>) { ... }
    fn make_blobs_2c() -> (Array2<f64>, Array1<i32>) { ... }

    // Tests by category
    mod kmeans {
        #[test]
        fn test_basic_fit() { ... }
        #[test]
        fn test_convergence() { ... }
    }

    mod dbscan {
        #[test]
        fn test_density_connectivity() { ... }
    }
}

mod models {
    //! Model tests (classification, regression, ensemble)
    ...
}

mod preprocessing {
    //! Feature preprocessing tests
    ...
}
```

**Assertion Patterns:**
```rust
// Property checks
assert_eq!(labels.len(), n_samples);
assert!(labels.iter().all(|&l| l >= 0));
assert!(inertia >= 0.0);

// Floating-point comparisons
assert!(score > 0.9);  // Loose bounds for ML
approx::assert_relative_eq!(actual, expected, epsilon = 1e-10);

// Array comparisons
ndarray::assert_allclose!(&actual, &expected, epsilon = 1e-10);
```

### Python Suite Organization

**Fixture Pattern:**
```python
# conftest.py - Shared fixtures
@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple regression data (100, 3) -> (100,)"""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y

@pytest.fixture
def classification_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate binary classification data (100, 3) -> (100,) with classes {0, 1}"""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return X, y

@pytest.fixture
def multiclass_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate 3-class data (150, 4) -> (150,) with classes {0, 1, 2}"""
    ...

@pytest.fixture
def large_regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate larger dataset for performance tests (10000, 10) -> (10000,)"""
    ...
```

**Test Class Pattern:**
```python
class TestGaussianNB:
    """Tests for GaussianNB classifier."""

    def test_fit_predict_basic(self):
        """Basic workflow: fit then predict."""
        X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0],
                      [6.0, 7.0], [7.0, 6.0], [8.0, 8.0]])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (6,)
        assert np.all(np.isin(preds, [0.0, 1.0]))

    def test_predict_proba_sums_to_one(self):
        """Mathematical check: probabilities sum to 1.0 per sample."""
        X = np.array([[1.0, 2.0], [2.0, 1.0], [6.0, 7.0], [7.0, 6.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = GaussianNB()
        model.fit(X, y)
        probas = model.predict_proba(X)
        assert probas.shape == (4, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)
```

**Comparison Helper Pattern (from conftest_comparison.py):**
```python
def _fit_both(X, Y=None):
    """Fit both FerroML and sklearn models, return (ferro_model, sk_model)."""
    from ferroml.preprocessing import StandardScaler as FerroSS
    from sklearn.preprocessing import StandardScaler as SkSS

    ferro = FerroSS()
    ferro.fit(X)
    sk = SkSS()
    sk.fit(X)
    return ferro, sk

class TestStandardScalerComparison:
    def test_iris_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        np.testing.assert_allclose(ferro.transform(X), sk.transform(X), atol=1e-10)
```

## Mocking & Fixtures

### Rust Fixtures

**Pattern:**
- No mock framework used; instead use real data helpers
- Fixtures are inline functions, not framework macros

**Data Generators:**
```rust
/// Generate well-separated 2D blob data with known cluster structure.
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
    ).unwrap();
    let labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];
    (x, labels)
}

/// Generate half-moon shaped data (simplified version).
fn make_moons() -> Array2<f64> {
    let mut data = Vec::new();
    for i in 0..10 {
        let angle = std::f64::consts::PI * i as f64 / 9.0;
        data.push(angle.cos());
        data.push(angle.sin());
    }
    Array2::from_shape_vec((20, 2), data).unwrap()
}
```

**Location:**
- In each test module: `/home/tlupo/ferroml/ferroml-core/tests/correctness.rs` lines 22-95 (clustering helpers)
- Same for each domain (preprocessing, models, etc.)

### Python Fixtures

**Framework:** pytest

**Scope:** Function-level (default), class-level when expensive

```python
@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate regression dataset (100 samples, 3 features)."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y
```

**Fixture Usage in Tests:**
```python
class TestTrainTestSplit:
    def test_shapes(self, regression_data):  # fixture injected
        from ferroml.model_selection import train_test_split
        X, y = regression_data  # Use fixture
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        assert X_train.shape[0] + X_test.shape[0] == 100
```

## Mocking Strategy

**What to Mock (Rust):**
- Nothing explicitly (use real data)
- Use carefully constructed test data instead

**What to Mock (Python):**
- External libraries: not done — tests compare against real sklearn/xgboost/lightgbm
- File I/O: use tmpdir fixture from pytest
- Network: not needed (library is CPU-local)

**What NOT to Mock:**
- Rust model internals: test actual behavior
- Python model internals: test actual behavior
- Cross-library behavior: real sklearn/linfa implementations

## Coverage

**Requirements:**
- No strict coverage target configured
- Test-driven approach: write tests alongside code

**View Coverage (Rust):**
```bash
cargo tarpaulin --out Html --exclude-files benches tests  # HTML report
cargo tarpaulin --timeout 600 --exclude-files benches tests
```

**Coverage Baseline (as of v0.3.1):**
- ~3,550 tests in ferroml-core (Rust)
- ~2,100 tests in ferroml-python (Python)
- 200+ cross-library validation tests
- 118 ONNX round-trip tests

## Test Types & Scopes

### Unit Tests (Rust)

**Location:** `src/**/*.rs` with `#[cfg(test)] mod tests`

**Scope:** Single functions/methods in isolation

**Example from `error.rs`:**
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

### Integration Tests (Rust)

**Location:** `/home/tlupo/ferroml/ferroml-core/tests/` (separate binaries)

**Scope:** Multi-component workflows, cross-module interactions

**Structure:** Consolidated to 6 files:
- `correctness.rs`: Model fit/predict workflows
- `integration.rs`: Pipeline, cross-validation, serialization
- `edge_cases.rs`: Boundary conditions, degenerate cases
- `adversarial.rs`: Pathological inputs, numerical instability
- `vs_linfa.rs`: Cross-library validation against linfa 0.8.1
- `regression_tests.rs`: Known issue tracking

### Integration Tests (Python)

**Location:** `/home/tlupo/ferroml/ferroml-python/tests/`

**Scope:**
- End-to-end workflows (fit/predict/transform chains)
- Cross-library comparisons (vs sklearn, xgboost, lightgbm, statsmodels)
- Feature parity validation

**Test Files:**
```
test_comparison_preprocessing.py   # 18 preprocessing transformers vs sklearn
test_comparison_trees.py           # Decision trees, RF, GB, XGB, LGB
test_comparison_unsupervised.py    # KMeans, DBSCAN, GMM, ICA, PCA vs sklearn
test_comparison_remaining.py       # Linear, logistic, SVM vs sklearn
test_comparison_edge_cases.py      # Adversarial, degenerate, boundary cases
test_naive_bayes.py                # Naive Bayes classifier variants
test_gaussian_process.py           # GP regression and classification
test_model_selection.py            # train_test_split, CV splitters
test_automl.py                     # AutoML system integration
test_sparse_roundtrip.py           # Sparse matrix handling
test_sparse_pipeline.py            # Sparse pipelines
test_rfe.py                        # Recursive feature elimination
test_kernel_shap.py                # SHAP feature importance
test_text_pipeline.py              # Text preprocessing pipeline
```

### Adversarial/Edge Case Tests

**Rust Location:** `/home/tlupo/ferroml/ferroml-core/tests/edge_cases.rs` (92KB)

**Examples:**
```rust
#[test]
fn test_single_sample() { /* Model with 1 sample */ }

#[test]
fn test_single_feature() { /* Model with 1 feature */ }

#[test]
fn test_constant_feature() { /* All features identical */ }

#[test]
fn test_large_feature_scales() { /* Features with vastly different magnitudes */ }

#[test]
fn test_highly_imbalanced_classification() { /* Extreme class imbalance */ }
```

**Python Location:** `/home/tlupo/ferroml/ferroml-python/tests/test_comparison_edge_cases.py`

### Fuzzing Tests (Rust)

**Framework:** libfuzzer-sys

**Location:** `/home/tlupo/ferroml/ferroml-core/fuzz/fuzz_targets/`

**Test Targets:**
```
fuzz_serialization_bincode.rs      # Bincode deserialization robustness
fuzz_serialization_json.rs         # JSON parsing edge cases
fuzz_serialization_msgpack.rs      # MessagePack parsing
fuzz_preprocessing_input_validation.rs  # Preprocessing input bounds
fuzz_preprocessing_scalers.rs       # Scaler numerical stability
fuzz_onnx_model_loading.rs         # ONNX model loading
```

**Run Fuzzing:**
```bash
cargo +nightly fuzz run fuzz_serialization_bincode -- -max_len=10000
cargo +nightly fuzz run fuzz_preprocessing_scalers -- -timeout=5
```

## Test Patterns & Best Practices

### Property-Based Assertions

**Mathematical Invariants (Rust):**
```rust
// Probabilities should sum to 1
np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-10)

// Inertia should be non-negative
assert!(inertia >= 0.0, "Inertia should be non-negative");

// Cluster labels should be contiguous from 0
let unique_labels: HashSet<_> = labels.iter().copied().collect();
assert_eq!(unique_labels.len(), expected_clusters);
```

### Async Testing

**Pattern (Python with pytest-asyncio):**
Not currently used in FerroML tests (library is synchronous).

### Error Testing

**Rust Pattern:**
```rust
#[test]
#[should_panic(expected = "assertion failed")]
fn test_panics_on_invalid() {
    let _ = Model::new(invalid_param);
}

// Or using Result assertions:
#[test]
fn test_error_handling() {
    let result = Model::fit(&invalid_data, &targets);
    assert!(result.is_err());
    match result {
        Err(FerroError::ShapeMismatch { expected, actual }) => {
            assert_eq!(expected, "(n, m)");
        }
        _ => panic!("Unexpected error type"),
    }
}
```

**Python Pattern:**
```python
def test_nan_in_features_raises(self):
    """Test that NaN in features raises an error."""
    model = LinearRegression()
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(Exception) as exc_info:
        model.fit(X, y)

    error_msg = str(exc_info.value).lower()
    assert 'nan' in error_msg or 'invalid' in error_msg
```

### Cross-Library Validation

**Pattern (Python):**
```python
class TestStandardScalerComparison:
    """FerroML StandardScaler vs sklearn StandardScaler."""

    def _fit_both(self, X):
        from ferroml.preprocessing import StandardScaler as FerroSS
        from sklearn.preprocessing import StandardScaler as SkSS
        ferro = FerroSS()
        ferro.fit(X)
        sk = SkSS()
        sk.fit(X)
        return ferro, sk

    def test_iris_transform(self):
        X, _ = get_iris()
        ferro, sk = self._fit_both(X)
        np.testing.assert_allclose(ferro.transform(X), sk.transform(X), atol=1e-10)
```

**Tolerance Strategy:**
- Default: `atol=1e-10` for double precision
- Relaxed: `atol=1e-8` for complex operations or single precision
- Very loose: `atol=1e-5` for probabilistic outputs

### Benchmarking

**Framework:** Criterion (Rust), custom Python scripts

**Rust Benchmarks:**
- Location: `/home/tlupo/ferroml/ferroml-core/benches/`
- Files: `benchmarks.rs`, `memory_benchmarks.rs`, `performance_optimizations.rs`, `gpu_benchmarks.rs`, `gaussian_process.rs`
- Run: `cargo bench --bench benchmarks`
- 86+ benchmark functions across 5 files

**Python Benchmarks:**
- Script: `scripts/benchmark_cross_library.py`
- Compares FerroML vs sklearn/xgboost/lightgbm/statsmodels
- Time per model, memory usage tracking

---

*Testing analysis: 2026-03-16*
