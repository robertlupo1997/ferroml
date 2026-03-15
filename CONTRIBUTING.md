# Contributing to FerroML

Thank you for your interest in contributing to FerroML! This guide will help you get started.

## Reporting Bugs

Please use the [bug report template](https://github.com/robertlupo1997/ferroml/issues/new?template=bug_report.yml) to file issues. Include a minimal reproducing example whenever possible.

## Development Setup

### Prerequisites

- **Rust**: stable toolchain (install via [rustup](https://rustup.rs/))
- **Python**: 3.10 or later
- **maturin**: `pip install maturin`

### Clone and Build

```bash
git clone https://github.com/robertlupo1997/ferroml.git
cd ferroml

# Build the Rust library
cargo build

# Build the Python bindings (requires a virtual environment)
python -m venv .venv
source .venv/bin/activate
maturin develop --release -m ferroml-python/Cargo.toml
```

### Pre-commit Hooks

Install pre-commit hooks to catch formatting and lint issues before committing:

```bash
pip install pre-commit
pre-commit install
```

## Testing

```bash
# Rust unit tests
cargo test --lib -p ferroml-core

# Python tests
pytest ferroml-python/tests/
```

## Code Quality

All code must pass formatting and lint checks before merging:

```bash
cargo fmt --all
cargo clippy -- -D warnings
```

## Commit Messages

Use conventional commit prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation changes
- `test:` — adding or updating tests
- `refactor:` — code restructuring without behavior change

Example: `feat: add Lasso regression model`

## Pull Request Process

1. Fork the repository and create a feature branch from `master`.
2. Make your changes and add tests.
3. Ensure all checks pass (`cargo fmt`, `cargo clippy`, `cargo test`, `pytest`).
4. Open a pull request against `master` using the PR template.
5. A maintainer will review your PR. Please be patient and responsive to feedback.

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project: MIT OR Apache-2.0.
