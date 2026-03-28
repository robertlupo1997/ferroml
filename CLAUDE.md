# FerroML

A machine learning library written in Rust with Python bindings, designed to compete with scikit-learn, statsmodels, and XGBoost. What sets it apart: every model includes statistical diagnostics (residual analysis, assumption tests, confidence intervals) as first-class features, not bolted-on extras. All implementations are correctness-verified against sklearn, scipy, linfa, and statsmodels via 5,000+ cross-validated tests.

## Workspace

- `ferroml-core/` — Rust library. Models, preprocessing, stats, metrics, pipelines, AutoML.
- `ferroml-python/` — PyO3 bindings. 55+ models exposed via `import ferroml`.

## Build & Test

```bash
# Rust
cargo test                    # core library tests
cargo test --test correctness # main correctness suite
cargo fmt --all               # required before commit (pre-commit hook)

# Python (activate venv first)
source .venv/bin/activate
maturin develop --release -m ferroml-python/Cargo.toml
pytest ferroml-python/tests/
```

## Core Types

- Features: `Array2<f64>`, Targets: `Array1<f64>` (ndarray)
- Traits: `Model` (fit/predict), `Transformer` (fit/transform), `StatisticalModel`, `ProbabilisticModel`
- Errors: `FerroError` — `ShapeMismatch`, `NotFitted`, `ConvergenceFailure`, `InvalidInput`
- Builder pattern: `Model::new().with_param(val)`, all models expose `search_space()` for HPO

## Rules

- Correctness is the top priority. Every change must pass existing tests. New code needs tests.
- Adding/changing a model requires 3 updates: Rust impl in `ferroml-core/src/` → PyO3 wrapper in `ferroml-python/src/` → re-export in `ferroml-python/python/ferroml/<submodule>/__init__.py`. Incomplete bindings are the most common mistake.
- `cargo test` only tests `ferroml-core` (workspace default-members). Use `cargo test -p ferroml-python` or `cargo test --all` to include binding-side Rust code.
- Linear algebra uses `nalgebra`, NOT `ndarray-linalg`. We removed ndarray-linalg to avoid the OpenBLAS system dependency. Do not re-add it.
- Gradient boosting models live in `ferroml.trees`, NOT `ferroml.ensemble`.
- Pre-commit hooks enforce `cargo fmt`, `clippy -D warnings`, and quick tests.
- Integration tests are consolidated into 6 files in `ferroml-core/tests/` — do not add new test binaries.

## Reference

- `REPO_MAP.md` — structural skeleton of all source files (pub API surfaces)
- `.planning/codebase/` — architecture, stack, conventions, testing, concerns docs
- `docs/plans/` — implementation phase history (Plans A-X)
