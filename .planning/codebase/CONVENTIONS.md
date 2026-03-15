# Coding Conventions

**Analysis Date:** 2026-03-15

## Naming Patterns

**Files:**
- Rust modules: `snake_case` (e.g., `linear.rs`, `standard_scaler.rs`, `kmeans.rs`)
- Python test files: `test_*.py` (e.g., `test_score_all_models.py`, `test_clustering.py`)
- Integration test files: `[feature_area].rs` (e.g., `correctness_preprocessing.rs`, `vs_linfa_linear.rs`)

**Functions:**
- Rust public APIs: `snake_case` (e.g., `fit()`, `predict()`, `with_fit_intercept()`, `cooks_distance()`)
- Builder pattern methods: `with_*` (e.g., `with_confidence_level()`, `with_feature_names()`)
- Helper/validation functions: `check_*`, `compute_*`, `generate_*`
- Python test functions: `test_*` (e.g., `test_fit_basic()`, `test_labels_shape_and_dtype()`)
- Python private utilities: `_*` prefix (e.g., `_fit_and_score_clf()`, `_partial_fit_classifier()`)

**Variables:**
- Rust data arrays: `x`, `y` for features and targets following ndarray conventions
- Rust private fields: `snake_case` with optional prefix for clarity (e.g., `n_samples`, `fitted_data`, `residuals`)
- Python fixtures: `*_data` suffix (e.g., `regression_data`, `clf_data`, `iris_like_data`)
- Model state: `fitted_*` or `n_*` conventions (e.g., `fitted_values`, `n_features`, `n_clusters`)

**Types:**
- Rust structs/enums: `PascalCase` (e.g., `LinearRegression`, `StandardScaler`, `ClusteringModel`)
- Rust trait names: `PascalCase` (e.g., `Model`, `Transformer`, `StatisticalModel`)
- Python classes: `PascalCase` (e.g., `LogisticRegression`, `DecisionTreeClassifier`)

**Constants:**
- Rust: `SCREAMING_SNAKE_CASE` (used sparingly in tolerances module: `CLOSED_FORM`, `SKLEARN_COMPAT`)
- Python: `SCREAMING_SNAKE_CASE` for module-level constants

## Code Style

**Formatting:**
- Rust: `cargo fmt` (enforced by pre-commit hook)
- Configuration: `.pre-commit-config.yaml` runs `cargo fmt --all -- --check`
- Linting: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- All Rust code must pass clippy with `-D warnings` (warnings are errors)

**Linting:**
- Tool: `cargo clippy`
- Config: Workspace-wide, warnings treated as errors (`-D warnings` flag)
- Pre-commit: Runs on all Rust files before commit
- Common fixes: address all clippy warnings in submission

**Line Length:**
- Rust: 100-120 characters (standard Rust convention, no enforced limit in code)
- Python: 100 characters (PEP 8 extended)
- Comments: wrap to prevent horizontal scroll

**Indentation:**
- Rust: 4 spaces
- Python: 4 spaces

## Import Organization

**Rust:**

Order:
1. Standard library imports (`std::*`)
2. Crate imports (`crate::*`)
3. External crate imports (alphbetical within groups)
4. Re-exports at module level

Example from `linear.rs`:
```rust
use crate::hpo::{ParameterValue, SearchSpace};
use crate::models::{check_is_fitted, validate_fit_input, ...};
use crate::pipeline::PipelineModel;
use crate::stats::diagnostics::{...};
use crate::{FerroError, Result};
use ndarray::{s, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
```

**Path Aliases:**
- `crate::models::*` - for model trait and implementations
- `crate::preprocessing::*` - for transformers
- `crate::testing::*` - for test utilities and assertions
- `ndarray::*` - for array operations
- `serde::*` - for serialization

**Python:**
```python
# Standard library
import numpy as np
import pytest
from typing import Tuple

# Local/relative imports at end
from ferroml.linear import LogisticRegression
```

## Error Handling

**Strategy:** Typed errors using `thiserror` crate

**Pattern - Return Result with FerroError:**
```rust
pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    validate_fit_input(x, y)?;
    // ... implementation
    Ok(())
}
```

**Error Variants (defined in `src/error.rs`):**
- `InvalidInput(String)` - malformed input data
- `ShapeMismatch { expected, actual }` - dimension mismatch with details
- `AssumptionViolation { assumption, test, p_value }` - statistical test failure
- `NumericalError(String)` - NaN/inf handling, overflow
- `ConvergenceFailure { iterations, reason }` - optimization failed
- `NotImplemented(String)` - feature not available
- `NotFitted { operation }` - model not fitted before use
- `ConfigError(String)` - invalid parameters
- `Timeout { operation, elapsed_seconds, budget_seconds }` - timeout exceeded

**Pattern - Use descriptive error contexts:**
```rust
// Include what failed and why
return Err(FerroError::ShapeMismatch {
    expected: format!("({}, {})", n_samples, n_features),
    actual: format!("({}, {})", x.nrows(), x.ncols()),
});
```

**No panics in library code:**
- Use `Result` types instead of `expect()`
- Panics acceptable only in test code or examples
- Use `.map_err()` to provide context

## Logging

**Framework:** `tracing` crate

**Patterns:**
- Use debug! for development: `tracing::debug!("Starting fit with {} samples", n)`
- Use warn! for recoverable issues: `tracing::warn!("Zero variance feature {}", i)`
- Use info! for major milestones: `tracing::info!("Model fitted successfully")`
- Not extensively used in current code (minimal instrumentation)

## Comments

**When to Comment:**
- Complex mathematical operations (e.g., QR decomposition steps)
- Non-obvious algorithmic choices (e.g., "Use Barnes-Hut approximation for O(n log n)")
- Assumptions and edge cases (e.g., "Features with zero variance handled separately")
- TODOs and FIXMEs should be rare; use issue tracking instead

**JSDoc/RustDoc:**
- All public APIs documented with `///` comments
- Module-level docs with `//!` block comments
- Examples included for main types and functions
- Example in docstrings:

```rust
/// Ordinary Least Squares Linear Regression with full statistical diagnostics
///
/// Fits a linear model: y = Xβ + ε
///
/// ## Statistical Assumptions
///
/// 1. **Linearity**: The relationship between X and y is linear
/// 2. **Independence**: Observations are independent
/// ...
///
/// ## Example
///
/// ```
/// use ferroml_core::models::linear::LinearRegression;
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y)?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression { ... }
```

**Comment Style:**
- Line comments: `//` for single lines
- Block comments: `/* ... */` rarely used
- Test comments: Explain what is being tested and why
- Use markdown in doc comments for structure

## Function Design

**Size:**
- Prefer functions <150 lines (guideline, not hard limit)
- Large implementations split into private helper methods
- Each function should have single responsibility

**Parameters:**
- Use builder pattern for complex initialization (e.g., `LinearRegression::new().with_fit_intercept(false)`)
- Avoid boolean parameters in public APIs (use builder methods instead)
- Array parameters always `&Array2<f64>`, `&Array1<f64>` (immutable references)
- Result type: `Result<T>` where T is return value

**Return Values:**
- Main algorithms return `Result<()>` for fit, `Result<Array1<f64>>` for predict
- Getter methods return `Option<T>` for potentially unfitted state
- Statistical functions return `Option<(f64, f64)>` for paired values (e.g., F-statistic, p-value)

**Pattern - Check is_fitted before accessing state:**
```rust
pub fn cooks_distance(&self) -> Option<Array1<f64>> {
    let data = self.fitted_data.as_ref()?;
    // ... compute from data
    Some(result)
}
```

## Module Design

**Exports:**
- Public API types and trait re-exported at module level
- Implementation details marked `pub(crate)` or private
- Sub-modules for large features (e.g., `models/linear.rs`, `preprocessing/scalers.rs`)

**Barrel Files:**
- `mod.rs` files re-export all public types
- Example in `preprocessing/mod.rs`:
```rust
pub use self::scalers::{MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler};
pub use self::polynomial::PolynomialFeatures;
pub use self::encoders::{OneHotEncoder, OrdinalEncoder, LabelEncoder};
```

**Trait Organization:**
- Core traits in `traits` module: `Model`, `Transformer`, `StatisticalModel`, `ProbabilisticModel`
- Implementations alongside business logic
- Trait implementations separate from struct definition by blank line + `impl` block

**Feature Flags:**
- Sparse operations: `#[cfg(feature = "sparse")]`
- GPU acceleration: `#[cfg(feature = "gpu")]`
- ONNX export: `#[cfg(feature = "onnx")]`
- Default features: `parallel`, `onnx`, `simd`, `faer-backend`

## API Design Patterns

**Builder Pattern (Common):**
```rust
let mut model = LinearRegression::new()
    .with_fit_intercept(true)
    .with_confidence_level(0.95)
    .with_feature_names(vec!["x1".to_string(), "x2".to_string()]);
```

**Trait-Based Polymorphism:**
- All models implement `Model` trait
- Common preprocessing via `Transformer` trait
- Optional: `StatisticalModel`, `ProbabilisticModel`, `IncrementalModel` for specialized behavior

**Sklearn API Parity:**
- Core methods: `fit(X, y)`, `predict(X)`, `score(X, y)` available on 55+ models
- Optional: `partial_fit()` for 10+ incremental learners
- Optional: `decision_function()` for 13+ classifiers
- Parameters validated in builders, not in fit()

## Type System

**Numerical Arrays:**
- Features: `&Array2<f64>` with shape `(n_samples, n_features)`
- Targets: `&Array1<f64>` with shape `(n_samples,)`
- Outputs: `Array1<f64>` for vectors, `Array2<f64>` for matrices
- No generic numeric types; always `f64` for stability

**Serialization:**
- Serde with `Serialize`, `Deserialize` derives on model structs
- Formats: JSON, MessagePack, Bincode, Protocol Buffers (optional)
- All models must be serializable

**Error Type:**
- Return `Result<T>` which is alias for `std::result::Result<T, FerroError>`
- Never `Option` for operation failures (use `Result`)
- Use `Option` for optional state (e.g., `Option<Array1<f64>>` for fitted coefficients)

## Testing Conventions

See TESTING.md for comprehensive patterns. Key points:
- Unit tests in `#[cfg(test)]` modules within source files
- Integration tests in `/tests/` directory as separate compilation units
- Test names start with `test_` or follow pattern `test_[feature]_[scenario]`
- Use fixtures extensively (pytest in Python, rstest in Rust)
- Assertions use specialized tolerance constants: `CLOSED_FORM`, `ITERATIVE`, `SKLEARN_COMPAT`

## Pre-commit Hooks

**Enforced Checks:**
1. `cargo fmt --all` - Format all Rust code (fails if not formatted)
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings` - Lint check (warnings are errors)
3. `cargo test -p ferroml-core --lib -- --test-threads=4` - Quick unit tests
4. Python script `scripts/check_debug_macros.py` - No `todo!()`, `unimplemented!()`, `dbg!()` in production code
5. Python script `scripts/check_cargo_toml.py` - Cargo.toml validation
6. Pre-commit hooks: trailing whitespace, line endings (LF), YAML/TOML checks, private key detection

**Workflow:**
```bash
# Before committing, pre-commit automatically:
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p ferroml-core --lib -- --test-threads=4
```

---

*Convention analysis: 2026-03-15*
