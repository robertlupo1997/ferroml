# FerroML Agent Operations Reference

> This file contains project-specific commands, patterns, and learnings for autonomous agents.
> Updated by agents as they discover useful information.

## Project Overview

FerroML is a comprehensive machine learning library for Rust with Python bindings.
- **Vision**: The greatest ML library - combining sklearn's completeness, statsmodels' rigor, and Rust's performance
- **Current State**: Phase 1-12 complete (137 tasks), Testing phases 16-20 in progress

## Directory Structure

```
ferroml/
├── ferroml-core/           # Main Rust library
│   ├── src/
│   │   ├── lib.rs          # Library root, re-exports
│   │   ├── models/         # ML models (linear, tree, ensemble, etc.)
│   │   ├── preprocessing/  # Transformers (scalers, encoders, etc.)
│   │   ├── metrics/        # Evaluation metrics
│   │   ├── cv/             # Cross-validation
│   │   ├── hpo/            # Hyperparameter optimization
│   │   ├── automl/         # AutoML orchestration
│   │   ├── explainability/ # SHAP, PDP, feature importance
│   │   ├── testing/        # Test utilities and frameworks
│   │   ├── ensemble/       # Voting, stacking, bagging
│   │   ├── pipeline/       # Pipeline and ColumnTransformer
│   │   └── ...
│   ├── tests/              # Integration tests
│   └── benches/            # Benchmarks
├── ferroml-python/         # Python bindings (PyO3/Maturin)
├── thoughts/shared/        # Plans and handoffs
│   ├── plans/              # Implementation plans
│   └── handoffs/           # Session handoff documents
└── IMPLEMENTATION_PLAN.md  # Master task tracking
```

## Essential Commands

### Build & Check
```bash
# Quick check (fastest)
cargo check -p ferroml-core

# Check with all features
cargo check -p ferroml-core --all-features

# Clippy lints
cargo clippy -p ferroml-core -- -D warnings

# Format check
cargo fmt --all -- --check
```

### Testing
```bash
# Run all tests
cargo test -p ferroml-core

# Run specific test module
cargo test -p ferroml-core testing::automl
cargo test -p ferroml-core testing::hpo
cargo test -p ferroml-core testing::callbacks
cargo test -p ferroml-core testing::explainability
cargo test -p ferroml-core testing::onnx

# Run with output
cargo test -p ferroml-core [pattern] -- --nocapture

# Run ignored tests
cargo test -p ferroml-core -- --ignored

# Count tests
cargo test -p ferroml-core 2>&1 | grep "^test result"
```

### Documentation
```bash
# Generate docs
cargo doc -p ferroml-core --no-deps --open

# Check doc tests
cargo test -p ferroml-core --doc
```

### Python Bindings
```bash
cd ferroml-python
maturin develop
pytest tests/
```

## Code Patterns

### Adding a New Test Module
1. Create `ferroml-core/src/testing/{name}.rs`
2. Add `pub mod {name};` to `ferroml-core/src/testing/mod.rs` (alphabetical order)
3. Write tests using existing patterns from `automl.rs` or `hpo.rs`
4. Run `cargo test -p ferroml-core testing::{name}`

### Test Module Structure
```rust
//! Description of what this module tests
//!
//! This module provides tests for:
//! - Feature A
//! - Feature B

use crate::models::SomeModel;
use ndarray::{Array1, Array2};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_a() {
        // Test implementation
    }
}
```

### Model Pattern
All models implement the `Model` trait:
```rust
pub trait Model: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
    fn n_features(&self) -> Option<usize>;
}
```

## Current Priorities

1. **Testing Phases 16-20**: Files exist, need verification and completion
2. **Testing Phases 21-28**: Advanced features tests (not started)
3. **Testing Phases 29-32**: Fairness, drift, regression, mutation testing

## Known Issues

- Pre-commit hooks may fail due to clippy warnings in existing code
- Use `--no-verify` for commits if hooks block valid changes
- Some test files have unused helper functions (ok for now)

## Learnings

<!-- Agents: Add discoveries here as you work -->

### 2026-02-01
- Testing modules must be registered in `mod.rs` to be compiled
- Phases 16-20 test files created but only automl was registered
- All 5 modules now registered: automl, callbacks, explainability, hpo, onnx

## Git Workflow

```bash
# Check status
git status --short

# Stage specific files (preferred)
git add path/to/file.rs

# Commit with co-author
git commit -m "type(scope): description

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push
git push origin master
```

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Unit Tests | ~1,700+ | 3,500+ |
| Code Coverage | ~73% | 90%+ |
| Testing Phases Complete | 5/17 | 17/17 |
