# FerroML Agent Operations Reference

> Operational commands and patterns only. No status updates or progress notes.

## Directory Structure

```
ferroml/
├── ferroml-core/src/       # Main library
│   ├── models/             # ML models
│   ├── preprocessing/      # Transformers
│   ├── testing/            # Test modules
│   ├── hpo/                # Hyperparameter optimization
│   └── automl/             # AutoML orchestration
├── IMPLEMENTATION_PLAN.md  # Task tracking
└── thoughts/shared/        # Plans and handoffs
```

## Essential Commands

```bash
# Quick check
cargo check -p ferroml-core

# Run all tests
cargo test -p ferroml-core

# Run specific test module
cargo test -p ferroml-core testing::automl
cargo test -p ferroml-core testing::hpo

# With output
cargo test -p ferroml-core [pattern] -- --nocapture

# Clippy
cargo clippy -p ferroml-core -- -D warnings
```

## Adding a Test Module

1. Create `ferroml-core/src/testing/{name}.rs`
2. Add `pub mod {name};` to `ferroml-core/src/testing/mod.rs`
3. Run `cargo test -p ferroml-core testing::{name}`

## Model Trait Pattern

```rust
pub trait Model: Send + Sync {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;
}
```

## Git Workflow

```bash
git add path/to/file.rs
git commit -m "type(scope): description

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

Commit types: feat, fix, test, docs, refactor, perf, chore
