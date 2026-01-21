# FerroML Development Guide

## Project Overview

FerroML is a statistically rigorous AutoML library in Rust with Python bindings.
**Core differentiator**: Statistical rigor first - assumption testing, effect sizes, CIs.

## Repository Structure

```
ferroml/
├── Cargo.toml                    # Workspace config
├── ferroml-core/                 # Core Rust library
│   └── src/
│       ├── lib.rs                # Entry point, AutoMLConfig
│       ├── error.rs              # FerroError types
│       ├── stats/                # ✅ COMPLETE - Statistical foundations
│       ├── hpo/                  # ✅ COMPLETE - Hyperparameter optimization
│       ├── models/               # ✅ COMPLETE - ML models (Linear, Tree, SVM, KNN, Naive Bayes, etc.)
│       ├── preprocessing/        # ✅ COMPLETE - Feature transformers (Scalers, Encoders, Imputers, etc.)
│       ├── cv/                   # ✅ COMPLETE - Cross-validation (KFold, Stratified, TimeSeries, Nested, etc.)
│       ├── ensemble/             # ✅ COMPLETE - Meta-learners (Voting, Stacking, Bagging)
│       ├── pipeline/             # ✅ COMPLETE - DAG execution (Pipeline, FeatureUnion, ColumnTransformer)
│       ├── automl/               # ✅ COMPLETE - AutoML orchestration (Portfolio, TimeBudget, MetaLearning)
│       ├── explainability/       # ✅ COMPLETE - Model interpretability (SHAP, PDP, ICE, Permutation)
│       ├── decomposition/        # ✅ COMPLETE - Dimensionality reduction (PCA, LDA, TruncatedSVD)
│       ├── metrics/              # ✅ COMPLETE - Evaluation metrics (Classification, Regression, Probabilistic)
│       └── serialization.rs      # ✅ COMPLETE - Model persistence (JSON, MessagePack, Bincode)
├── ferroml-python/               # PyO3 Python bindings
├── specs/                        # Requirement specifications
├── IMPLEMENTATION_PLAN.md        # Current task list (Ralph state)
└── thoughts/shared/plans/        # PRD and planning docs
```

## Build Commands

```bash
# Check compilation (fast)
cargo check

# Build debug
cargo build

# Build release (optimized)
cargo build --release

# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Clippy linting
cargo clippy -- -D warnings

# Format code
cargo fmt

# Generate docs
cargo doc --open
```

## Test Commands

```bash
# Run all tests
cargo test --workspace

# Run ferroml-core tests only
cargo test -p ferroml-core

# Run with all features
cargo test --all-features
```

## Validation Checklist (Run After Each Task)

1. `cargo check` - Must pass with no errors
2. `cargo clippy` - Should have no warnings (ideally)
3. `cargo test` - All tests must pass
4. `cargo fmt --check` - Code must be formatted

## Key Design Patterns

### Trait-based Polymorphism
```rust
pub trait Model: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
}
```

### Result Types
All fallible operations return `Result<T, FerroError>`.

### Statistical Results
Every statistical operation returns full context:
- Test statistic
- p-value
- Effect size with interpretation
- Confidence interval
- Assumption test results

## Dependencies (already configured)

- `ndarray` - NumPy-like arrays
- `nalgebra` - Linear algebra
- `polars` - DataFrames
- `statrs` - Statistical distributions
- `rayon` - Parallelism
- `serde` - Serialization
- `pyo3` - Python bindings

## Don't Assume Not Implemented

Before implementing any feature:
1. Search the codebase for existing implementations
2. Check if similar functionality exists in a different module
3. Look for TODOs or placeholder code

## Code Style

- Use `///` doc comments for public items
- Include examples in doc comments
- Add `#[cfg(test)]` module for unit tests
- Prefer `Result` over panics
- Use explicit types for public APIs
