# Coding Conventions

**Analysis Date:** 2026-03-20

## Naming Patterns

**Files:**
- Module files: lowercase with underscores (`linear_regression.rs`, `count_vectorizer.rs`)
- Crate root: `lib.rs`, `main.rs`, or `mod.rs`
- Test integration files: `{feature_name}.rs` in `ferroml-core/tests/` directory
- Python test files: `test_{feature}.py` or `test_vs_{library}.py` (e.g., `test_comparison_linear.py`, `test_vs_sklearn_gaps_phase2.py`)

**Types (Rust structs/enums):**
- PascalCase for public types: `LinearRegression`, `RandomForestClassifier`, `StandardScaler`, `ShapeMismatch`, `FerroError`
- Suffix models with task-specific name when applicable: `RandomForestRegressor` vs `RandomForestClassifier`, `GradientBoostingRegressor` vs `GradientBoostingClassifier`
- Error enum variants: PascalCase describing the error (`InvalidInput`, `ShapeMismatch`, `ConvergenceFailure`, `NotFitted`, `AssumptionViolation`)

**Functions & Methods:**
- snake_case for all functions: `fit()`, `predict()`, `predict_proba()`, `fit_transform()`, `with_parameter()`
- Builder pattern methods start with `with_`: `.with_n_estimators(10)`, `.with_fit_intercept(true)`, `.with_random_state(42)`, `.with_confidence_level(0.95)`
- Getter methods: `coefficients()`, `intercept()`, `cluster_centers()`, not `get_coefficients()`
- Statistical diagnostic methods use full descriptive names: `standardized_residuals()`, `studentized_residuals()`, `cooks_distance()`, `dffits()`, `vif()`
- Test functions: `test_{feature_being_tested}()` or `test_{subject}_{specific_case}()` (e.g., `test_kmeans_blobs_finds_correct_clusters()`, `test_linear_regression_simple()`)

**Variables:**
- snake_case: `x_train`, `y_test`, `max_depth`, `learning_rate`, `n_samples`, `n_features`
- Mathematical/domain-specific single letters acceptable in formulas and math-heavy code: `i`, `j`, `k` for loops; `x`, `y` for features/targets; `m`, `n` for dimensions
- Prefix `_` for intentionally unused parameters: `_unused_param`
- Boolean flags read clearly: `fit_intercept`, `warm_start`, `use_feature_names` (not just `intercept`, `start`, `names`)

**Constants:**
- SCREAMING_SNAKE_CASE at module level: `DEFAULT_MAX_ITER = 1000`, `MIN_SAMPLES_LEAF = 1`

## Code Style

**Formatting:**
- Rust: `cargo fmt --all` enforced by pre-commit hook (edition 2021)
- Python: Follows PEP 8 conventions (no pre-commit formatter enforced, but consistent 4-space indentation)
- Line length: No strict limit, but prefer clarity (Rust files can exceed 100 chars for documentation examples)

**Linting:**
- Rust: `cargo clippy -D warnings` enforced by pre-commit hook
- Clippy lints explicitly allowed with justification in `lib.rs` (lines 88-150): `allow(clippy::too_many_arguments)`, `allow(clippy::too_many_lines)`, `allow(clippy::cast_precision_loss)`, etc.
- Rationale: ML codebases have many parameters, complex algorithms, numeric conversions; pedantic lints often too noisy

**Indentation & Whitespace:**
- Rust: 4 spaces (standard Rust)
- Python: 4 spaces

## Import Organization

**Order (Rust):**
1. Standard library: `use std::...`
2. External crates: `use ndarray::..., use serde::...`
3. Internal crate: `use crate::...`
4. Relative imports: `use super::...`

**Example from `ferroml-core/src/models/logistic.rs`:**
```rust
use crate::hpo::{ParameterValue, SearchSpace};
use crate::metrics::probabilistic::{roc_auc_score, roc_auc_with_ci};
use crate::models::{...};
use crate::pipeline::PipelineModel;
use crate::{FerroError, Result};
use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
```

**Path Aliases:**
- No custom path aliases in ferroml-core (uses standard `crate::`, `super::`)
- PyO3 bindings use explicit module paths in `ferroml-python/src/` to expose to Python: `pub use ferroml_core::linear::{...}`

**Wildcards:**
- Acceptable in test modules: `use super::*;`
- Avoid in library code

## Error Handling

**Pattern - Result Type:**
- All fallible operations return `Result<T>` (alias for `std::result::Result<T, FerroError>`)
- Defined in `ferroml-core/src/error.rs`: `pub type Result<T> = std::result::Result<T, FerroError>;`

**Pattern - Error Enum:**
- Use `FerroError` enum with meaningful variants (see `error.rs` lines 12-100):
  - `InvalidInput(String)` — validation failures
  - `ShapeMismatch { expected, actual }` — array dimension mismatches
  - `AssumptionViolation { assumption, test, p_value }` — statistical tests fail
  - `NumericalError(String)` — NaN/Inf/convergence issues
  - `ConvergenceFailure { iterations, reason }` — optimizer didn't converge
  - `NotFitted { operation }` — model used before fit()
  - `NotImplemented(String)` — missing feature
  - `ConfigError(String)` — invalid configuration

**Pattern - Propagation:**
- Use `?` operator for early returns
- Wrap external errors with context: `std::io::Error` wrapped as `FerroError::IoError`

**Pattern - Validation:**
- Input validation happens at model entry points (fit, predict)
- Helper: `validate_fit_input(x, y)` checks shapes and finite values
- Helper: `validate_predict_input(x)` checks shapes and finite values
- See `ferroml-core/src/models/traits.rs` for validation trait

## Logging

**Framework:** No structured logging in library code; internal `tracing` crate for future instrumentation

**Patterns:**
- Algorithms use `println!` in examples only (not in library)
- No info/warn logs in models themselves (library is silent)
- Diagnostics returned as struct fields: `DiagnosticsResult { assumption_tests: Vec<...>, residuals: Array1<...> }`
- Python side logs via pytest output if needed

## Comments

**When to Comment:**
- Complex statistical formulas: show the equation and cite the source
- Non-obvious algorithm choices: explain why (e.g., "Use QR decomposition for numerical stability")
- Hard-coded thresholds: explain the justification (e.g., "Band-aid threshold at 10K samples for SVC kernel to prevent memory explosion")
- Workarounds for edge cases: mark with `// HACK:` or `// TODO:` if temporary

**When NOT to Comment:**
- Self-documenting code (good function/variable names)
- What is obvious from the code itself

**Doc Comments (Rust):**
- All public types and methods: `///` style doc comments with examples
- Module-level: `//!` with design philosophy and architecture diagrams (see `lib.rs` lines 1-85)
- Example: LinearRegression module (lines 1-49) shows full example with imports

**Examples in Doc Comments:**
- Runnable examples using `use` statements
- Show typical builder pattern usage: `.with_parameter(value)`
- Mark slow examples with `#[ignore]` or mention runtime

## Function Design

**Size:**
- Prefer functions < 200 lines (complex models may exceed this)
- Break long algorithms into helper functions with descriptive names
- Tests can be longer (up to 300-400 lines for complex test suites)

**Parameters:**
- Use builder pattern for configurable models (see `with_*` methods)
- Maximum ~10 parameters before extracting to config struct
- All models implement Default and accept `()` for `new()`: `Model::new()` == `Model::default()`

**Return Values:**
- Result<T> for fallible operations
- Option<T> for optional results (e.g., `r_squared()` returns `Option<f64>` because requires fitted model)
- Tuple for multiple related outputs: `(f_statistic, p_value)` from `f_statistic()`
- Struct with named fields for complex outputs: `ModelSummary { ... }`, `DiagnosticsResult { ... }`

**Trait Implementation:**
- Core traits: `Model` (fit/predict), `Transformer` (fit/transform), `StatisticalModel`, `ProbabilisticModel`
- All regression models: implement `Model`, `StatisticalModel`
- All classifiers: implement `Model`, `ProbabilisticModel`
- All preprocessors: implement `Transformer`
- See `ferroml-core/src/models/traits.rs` for full trait hierarchy

## Module Design

**Exports:**
- Re-export public types in parent module: `pub use submodule::{Type1, Type2}`
- Example: `models/mod.rs` lines 62-100 re-export all model types
- This allows `use ferroml_core::models::{LinearRegression, RandomForestClassifier}` (not `use ferroml_core::models::linear::LinearRegression`)

**Barrel Files:**
- `mod.rs` aggregates and re-exports submodules
- Pattern: `pub mod submodule;` followed by `pub use submodule::{...};`
- Keeps public API clean and organized

**Module Documentation:**
- Each module starts with `//!` doc comment explaining purpose
- Lists key types and traits provided
- Usually includes a design philosophy section
- Example from `models/mod.rs` (lines 1-26) and `preprocessing/mod.rs` (lines 1-38)

**Test Organization:**
- Unit tests live in same file as implementation: `#[cfg(test)]` at end
- Integration tests consolidated in `/ferroml-core/tests/` (6 files total):
  - `correctness.rs` — Cross-library validation, correctness tests
  - `adversarial.rs` — Adversarial inputs, robustness
  - `edge_cases.rs` — Degenerate inputs, 13-scenario matrix
  - `integration.rs` — Pipeline, CV, serialization end-to-end
  - `regression_tests.rs` — Plan-by-plan regression suite
  - `vs_linfa.rs` — Cross-validation against linfa library

**No new test binaries added** — consolidation reduced build from 350 GB to ~14 GB debug size

## Serialization

**Pattern:**
- All models derive `Serialize, Deserialize` using serde
- Use `serde_json` for human-readable dumps
- Use `bincode` for compact binary serialization
- Python pickle support via PyO3 (see `ferroml-python/src/pickle.rs`)

**Example:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression { ... }
```

## Type Signatures

**Array Types:**
- Features: `&Array2<f64>` (2D, owned or borrowed)
- Targets: `&Array1<f64>` (1D, owned or borrowed)
- Return types: `Array1<f64>` or `Array2<f64>` (owned, caller moves)
- No generic `<T>` for array types — FerroML is f64-only for ML safety

**Borrowed vs Owned:**
- Input parameters: borrowed references `&Array2<f64>`
- Return values: owned arrays (caller decides memory)
- Fit methods: mutable `&mut self` for internal state

## Pre-commit Hooks

**Enforced by `.git/hooks/pre-commit`:**
1. `cargo fmt --all` — Code must be formatted
2. `cargo clippy -D warnings` — No clippy warnings
3. `cargo test --lib` — Quick library tests (integration tests skipped)

**Commit failure means:**
- Fix formatting: `cargo fmt --all`
- Fix clippy: address warnings or add to allowed list in `lib.rs`
- Re-stage and commit again (do NOT amend previous commit if test fails)

---

*Convention analysis: 2026-03-20*
