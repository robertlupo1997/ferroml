# Technology Stack

**Analysis Date:** 2026-03-16

## Languages

**Primary:**
- Rust 1.75.0+ (MSRV) - Core library (ferroml-core): ~183K lines. Edition 2021. All numerical algorithms, models, and ML implementations.
- Python 3.10+ - PyO3 bindings (ferroml-python). Python type hints, integration tests, test fixtures.

**Secondary:**
- Python 3.10-3.13 - Test frameworks and CI scripts (pytest, mypy, ruff). Optional dependencies (scikit-learn, scipy, pandas, polars for cross-library validation).

## Runtime

**Rust Environment:**
- Rust Toolchain: Stable channel (tests on 1.75.0 MSRV, stable, and nightly via CI)
- Targets: x86_64-unknown-linux-gnu, x86_64-apple-darwin, aarch64-apple-darwin, x86_64-pc-windows-msvc (tested in CI on Linux, macOS, Windows)
- Edition: 2021

**Python Environment:**
- CPython 3.10, 3.11, 3.12, 3.13
- Virtual environment: Standard `python -m venv`
- Maturin builds: Extension modules (abi3-py310) for backward compatibility

**Package Managers:**
- Cargo (Rust) - workspace with 2 crates (ferroml-core, ferroml-python). Lockfile: `Cargo.lock`
- pip (Python) - for test dependencies and optional sklearn/pandas/polars
- PyPI (distribution) - wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)

## Frameworks

**Core ML/Numerical:**
- `ndarray` 0.16 - N-dimensional arrays (Array2<f64> for features, Array1<f64> for targets). BLAS-less (uses nalgebra for linalg). With rayon and serde features.
- `ndarray-stats` 0.6 - Statistical functions on ndarray
- `nalgebra` 0.33 - Linear algebra (decompositions, solvers). Replaces ndarray-linalg to avoid OpenBLAS dependency.
- `statrs` 0.18 - Statistical distributions (normal, chi-square, t, F, binomial, uniform, gamma, beta, exponential, logistic)
- `rand` 0.9 + `rand_distr` 0.5 - Random number generation and distributions
- `argmin` 0.10 + `argmin-math` 0.4 - Optimization algorithms (L-BFGS, Nelder-Mead, Powell, Adam, etc.)

**Parallel & Concurrent:**
- `rayon` 1.10 - Data parallelism for ensemble models, cross-validation loops, batch predictions
- `crossbeam` 0.8 - Thread synchronization primitives (used in parallel model training)
- `tokio` 1.x (full features) - Async runtime for I/O-bound operations (currently used in test infrastructure)

**SIMD & Performance:**
- `wide` 0.7 - Portable SIMD operations for distance calculations (optional feature `simd`)

**GPU Acceleration:**
- `wgpu` 23 - GPU compute via WebGPU (optional feature `gpu`)
- `bytemuck` 1.x - GPU memory interop with safety guarantees
- `pollster` 0.4 - GPU executor

**Data Processing & I/O:**
- `polars` 0.46 - DataFrame operations, CSV/Parquet I/O (optional feature `datasets`, enabled by default). Supports lazy evaluation.
- `arrow` 54 - Columnar Arrow format for data exchange. Zero-copy array views.
- `memmap2` 0.9 - Memory-mapped file I/O for datasets that don't fit in RAM

**Sparse Matrices:**
- `sprs` 0.11 - Compressed sparse row/column (CSR/CSC) formats. Exposed via SparseModel trait (optional feature `sparse`).

**Serialization:**
- `serde` 1.0 - Serialization framework (derive macros). Models implement Serialize/Deserialize.
- `serde_json` 1.0 - JSON encoding/decoding
- `bincode` 1.3 - Efficient binary serialization (model snapshots)
- `rmp-serde` 1.3 - MessagePack binary format
- `prost` 0.13 - Protocol Buffers (for ONNX export support)

**Error Handling:**
- `thiserror` 2.0 - Derive error types with Display/std::error::Error
- `anyhow` 1.0 - Flexible error context and backtraces

**Logging & Observability:**
- `tracing` 0.1 - Structured logging framework (spans, events)
- `tracing-subscriber` 0.3 - Log backend and formatting

**Python Interop:**
- `pyo3` 0.24 - Python bindings via FFI. Features: extension-module, abi3-py310 (stable ABI for backward compatibility)
- `numpy` 0.24 - NumPy array interop (Array2<f64> ↔ numpy.ndarray)
- `pyo3-polars` 0.21 - Zero-copy Polars/PyArrow interchange

**ONNX Support:**
- ONNX export support enabled via default feature flag (no inference; export only)
- `ort` 2.0.0-rc.11 - ONNX Runtime for round-trip validation in tests (optional feature `onnx-validation`)

**Advanced Linear Algebra:**
- `faer` 0.20 - High-performance linear algebra backend (optional feature `faer-backend`, enabled by default). Used for decompositions, solves.

## Key Dependencies

**Critical:**
- `ndarray` 0.16 - All models operate on ndarray arrays. Core data structure of the library.
- `nalgebra` 0.33 - Linear algebra backbone (SVD, QR, Cholesky, eigenvalues). Replaces ndarray-linalg to avoid OpenBLAS.
- `pyo3` 0.24 - Python bindings FFI. All 55+ models exposed to Python via PyO3.
- `polars` 0.46 - CSV/Parquet loading for datasets feature. Uses Arrow for zero-copy data exchange.

**Infrastructure:**
- `rayon` 1.10 - Parallel computation. Used in ensemble methods, cross-validation, distance calculations.
- `statrs` 0.18 - Statistical functions (probability distributions, hypothesis tests, diagnostics).
- `argmin` 0.10 - Optimization solvers (L-BFGS for LogisticRegression, Nelder-Mead for custom objectives).
- `serde` 1.0 - Model persistence and interop.
- `faer` 0.20 - High-performance matrix decompositions (SVD, QR, eigenvalues) used in dimensionality reduction, PCA, ICA.

**Optional (Feature-Gated):**
- `wide` 0.7 - SIMD distance calculations (feature: `simd`)
- `sprs` 0.11 - Sparse matrix support (feature: `sparse`)
- `wgpu` 23 - GPU compute (feature: `gpu`)
- `ort` 2.0.0-rc.11 - ONNX round-trip validation (feature: `onnx-validation`)
- `polars`, `arrow` - CSV/Parquet loading (feature: `datasets`, enabled by default)

## Testing Dependencies

**Rust:**
- `criterion` 0.5 - Benchmarking framework (86+ benchmarks across 5 bench files). Outputs to `target/criterion/`
- `approx` 0.5 - Approximate float equality comparisons (tolerance-based)
- `proptest` 1.5 - Property-based testing (randomized edge case generation)
- `tempfile` 3.15 - Temporary directories for test data
- `test-case` 3.3 - Parameterized test macros
- `rstest` 0.23 - Fixture-based testing (reusable test data)
- `dhat` 0.3 - Heap allocation profiling
- `memory-stats` 1.2 - Memory usage tracking during benchmarks

**Cross-library validation (dev-dependencies):**
- `linfa` 0.8 + linfa-* (0.8) - Validated against 12 linfa models (linear, trees, SVM, clustering, PCA, etc.)

**Python:**
- `pytest` 7.0+ - Test runner
- `pytest-cov` 4.0+ - Coverage reporting
- `pytest-xdist` 3.0+ - Parallel test execution
- `mypy` 1.0+ - Type checking
- `ruff` 0.1+ - Linting (line-length 100, targets E/F/W/I/UP/B/C4/SIM)

**Optional Python test dependencies:**
- `scikit-learn` 1.0+ - Cross-library validation (60+ sklearn vs ferroml tests)
- `pandas` 1.5+ - DataFrame I/O and DataFrame tests
- `polars` 0.19+ - Native DataFrame support

## Build & Compilation

**Profile Settings:**
- Release: thin LTO, codegen-units=1, opt-level=3
- Bench: thin LTO (same as release for accurate benchmarks)

**Python Binding Build:**
- Builder: `maturin` 1.4+ (PyO3 build tool)
- Bindings: PyO3 via FFI
- Extension Module: cdylib (compiled .so/.pyd)
- Wheel Compatibility: abi3-py310 (stable ABI, compatible with Python 3.10-3.13)
- Strip: True (smaller wheel size)

**Compilation Targets:**
- Rust: x86_64-unknown-linux-gnu, x86_64-apple-darwin, aarch64-apple-darwin, x86_64-pc-windows-msvc
- Python wheels: manylinux (auto detection), macOS, Windows

## Configuration Files

**Cargo Workspace:**
- `Cargo.toml` - Workspace manifest with shared dependencies
- `ferroml-core/Cargo.toml` - Core library config (143 feature combinations tested)
- `ferroml-python/Cargo.toml` - Python bindings config (features: polars, pandas, sparse)

**Rust Configuration:**
- `rust-toolchain.toml` - MSRV 1.75.0, stable channel, clippy, rustfmt, cross-compilation targets
- `.cargo/config.toml` (if exists) - Cargo behavior customization
- `deny.toml` - Security audit (advisories, licenses, bans, sources). Ignores 3 unmaintained but non-critical crates.

**Python Configuration:**
- `ferroml-python/pyproject.toml` - maturin build system, pytest config, mypy config, ruff config
  - Build system: maturin 1.4+
  - Requires: Python 3.10+
  - Dependencies: numpy 1.21+
  - Optional: scikit-learn, polars, pandas (for tests)

**Code Quality:**
- `.pre-commit-config.yaml` - 10 hooks: cargo fmt (check), cargo clippy (-D warnings), cargo test (quick), custom debug macro check, large file check, Cargo.toml validation, trailing whitespace, YAML validation, line ending normalization, secret detection

## CI/CD Infrastructure

**GitHub Actions Workflows:**
- `.github/workflows/ci.yml` - Main CI: check, clippy, fmt, test (3 OS), Python tests (3 OS × 3 Python versions), coverage (70% threshold, 75% target), benchmarks (PR only), memory profiling (main only), security audit, license compliance, MSRV test
- `.github/workflows/publish-pypi.yml` - PyPI publishing: validate, build wheels (Linux/macOS/Windows, multiarch), build sdist, publish to PyPI (trusted publishing) or TestPyPI
- `.github/workflows/benchmarks.yml` - Criterion benchmark tracking
- `.github/workflows/mutation.yml` - Mutation testing (not detailed here)
- `.github/workflows/fuzz.yml` - Fuzzing tests
- `.github/workflows/release.yml` - Version release workflow
- `.github/workflows/docs.yml` - Documentation generation (docs.rs)
- `.github/workflows/changelog.yml` - CHANGELOG automation

**Environment Variables:**
- CARGO_TERM_COLOR=always (CI)
- RUSTFLAGS=-D warnings (CI)

## Dependency Management

**Security Scanning:**
- `cargo deny` - Advisory database from rustsec (https://github.com/rustsec/advisory-db)
- License compliance: Accepts MIT, Apache-2.0, BSD-2/3-Clause, BSL-1.0, ISC, Zlib, 0BSD, CC0-1.0, Unicode, MPL-2.0, OpenSSL
- Confidence threshold: 0.9 (license detection)

**Stability:**
- Cargo.lock committed (reproducible builds)
- MSRV enforced: 1.75.0 (tested in CI)
- Dependency tree pruning: No unused dependencies (checked by clippy)

**Crate Publishing:**
- ferroml-core: Published to crates.io
- ferroml-python: Published to PyPI (wheels + sdist)
- Repository: github.com/robertlupo1997/ferroml
- License: MIT OR Apache-2.0

## Platform Requirements

**Development:**
- Rust 1.75.0+ (MSRV enforced)
- Python 3.10+ (for bindings and tests)
- Linux (Ubuntu 20.04+), macOS (11+), Windows (10/11 with MSVC toolchain)
- Disk space: 14 GB for debug builds (after consolidation from 350 GB)

**Production:**
- Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)
- No external runtime dependencies (all Rust, statically linked)
- Python 3.10-3.13 (for PyPI packages)
- NumPy 1.21+ (Python bindings only)

---

*Stack analysis: 2026-03-16*
