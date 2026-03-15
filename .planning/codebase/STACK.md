# Technology Stack

**Analysis Date:** 2026-03-15

## Languages

**Primary:**
- Rust 1.75.0 (MSRV) — Core ML algorithms and numerical computing engine
- Python 3.10+ — Language bindings via PyO3, user-facing API

**Supporting:**
- TOML — Cargo and project configuration
- YAML — GitHub Actions CI/CD workflows and pre-commit hooks
- Bash — Development scripts (benchmarking, data collection)

## Runtime

**Environment:**
- Rust: `1.75.0` (stable channel with rustfmt, clippy, cross-platform targets)
- Python: CPython 3.10, 3.11, 3.12, 3.13
- Platforms: Linux (x86_64), macOS (x86_64, aarch64), Windows (x86_64 MSVC)

**Package Manager:**
- Cargo — Rust dependencies and workspace management
- pip/maturin — Python package distribution and building
- npm — Claude CLI tooling (development only, in Ralph Wiggum Docker environment)

## Frameworks

**Core Numerical Computing:**
- `ndarray` 0.16 — Dense array operations (features: rayon, serde)
- `ndarray-stats` 0.6 — Statistical operations on arrays
- `nalgebra` 0.33 — Linear algebra with serialization support
- `wide` 0.7 — Portable SIMD for distance calculations (optional feature)
- `faer` 0.20 — High-performance linear algebra backend (optional, default enabled)

**Data Processing:**
- `polars` 0.46 — DataFrame operations with lazy evaluation, Parquet/JSON support
- `arrow` 54 — Apache Arrow columnar format support
- `sprs` 0.11 — Sparse matrix operations (CSR/CSC formats, optional feature)
- `memmap2` 0.9 — Memory-mapped file access for large datasets

**Statistical Distribution & Optimization:**
- `statrs` 0.18 — Statistical distributions and probability functions
- `rand` 0.9 + `rand_distr` 0.5 — Random number generation and distributions
- `rand_chacha` 0.9 — ChaCha PRNG for reproducible randomness
- `argmin` 0.10 + `argmin-math` 0.4 — Optimization algorithms framework

**Parallel Processing:**
- `rayon` 1.10 — Data parallelism for arrays, model fitting
- `crossbeam` 0.8 — Thread synchronization primitives
- `tokio` 1.x — Async runtime (full feature set, for future extensibility)

**GPU Acceleration (Optional):**
- `wgpu` 23 — WebGPU compute shaders for GPU acceleration (optional feature `gpu`)
- `bytemuck` 1.x — Memory layout transformation for GPU data
- `pollster` 0.4 — Async GPU operations blocking

**ONNX Export & Validation:**
- `prost` 0.13 — Protocol buffers for ONNX serialization format
- `ort` 2.0.0-rc.11 — ONNX Runtime for round-trip validation (optional feature `onnx-validation`)
- `crc32fast` 1.4 — CRC32 checksums for ONNX file integrity

**Serialization:**
- `serde` 1.0 — Serialization framework (derive macros)
- `serde_json` 1.0 — JSON serialization
- `bincode` 1.3 — Binary serialization
- `rmp-serde` 1.3 — MessagePack serialization

**Error Handling:**
- `thiserror` 2.0 — Error type derivation macros
- `anyhow` 1.0 — Flexible error handling with context

**Logging & Observability:**
- `tracing` 0.1 — Structured tracing framework (infrastructure only, not active)
- `tracing-subscriber` 0.3 — Tracing output formatting

**Python Bindings (PyO3):**
- `pyo3` 0.24 — Python C extension bindings (features: extension-module, abi3-py310)
- `numpy` 0.24 — NumPy array interop
- `pyo3-polars` 0.21 — Polars DataFrame interop with derive exports

## Key Dependencies

**Critical for Core Operations:**
- `ndarray` + `nalgebra` — All 55+ ML algorithms depend on dense/sparse arrays
- `polars` + `arrow` — Dataset loading and preprocessing
- `rayon` — Parallel model fitting and batch predictions
- `statrs` — Statistical distributions for NB, GP, mixture models
- `argmin` — Optimization for gradient boosting, SVM, logistic regression

**Performance-Critical:**
- `wide` (SIMD) — Distance calculations in clustering, KNN
- `faer` (linear algebra) — Matrix decompositions (PCA, SVD) and solve operations
- `wgpu` (GPU, optional) — GPU-accelerated kernels for large-scale fitting

**Data Integration:**
- `pyo3-polars` — Direct Polars DataFrame exchange with Python
- `numpy` — NumPy array zero-copy interop

## Configuration

**Environment:**
- No external environment variables required for core library
- `.env` files: Not used
- Configuration approach: Rust builder patterns in `Config` structs (AutoMLConfig, ModelConfig)

**Build Configuration:**
- `Cargo.toml` — Workspace and package metadata (root, `ferroml-core/`, `ferroml-python/`)
- `rust-toolchain.toml` — Pinned to 1.75.0 stable
- `pyproject.toml` — Python wheel build configuration via maturin
- `.pre-commit-config.yaml` — Git hooks: cargo fmt, cargo clippy, cargo test
- `profile.release` — LTO: thin, codegen-units: 1, opt-level: 3

**Feature Flags:**
- `default = ["parallel", "onnx", "simd", "faer-backend"]`
- `parallel` — Rayon data parallelism (always enabled by default)
- `simd` — Wide SIMD for distance metrics
- `sparse` — Sparse matrix support via sprs
- `onnx` — ONNX export metadata generation
- `onnx-validation` — Full ONNX round-trip testing with ort
- `faer-backend` — High-performance linear algebra
- `gpu` — GPU acceleration via wgpu

## Platform Requirements

**Development:**
- Rust 1.75.0 or later (stable)
- Python 3.10+ dev environment for PyO3 bindings
- C compiler (for Rust dependencies, MSVC on Windows)
- Pre-commit hooks: Python 3.6+

**Production (Python wheels):**
- CPython 3.10, 3.11, 3.12, 3.13
- Linux (x86_64 glibc 2.28+)
- macOS (x86_64, aarch64)
- Windows (x86_64 MSVC)
- Binary wheels with bundled Rust code (no Rust build required)

**Production (Rust crate):**
- Any platform with Rust toolchain support
- No system dependencies (all numerical libs bundled)

## CI/CD Infrastructure

**Build System:**
- GitHub Actions — Multi-platform CI (ubuntu-latest, macos-latest, windows-latest)
- Maturin — Python wheel builder for PyPI distribution
- Dependabot — Automated dependency updates (Cargo, pip, GitHub Actions)

**Artifact Management:**
- PyPI — Python package repository (automatic publish on version tags)
- crates.io — Rust crate registry (manual publish)
- GitHub Releases — Binary distribution and changelog

**Quality Gates:**
- `cargo check` — Full compile check on all features
- `cargo clippy` — Linting with `-D warnings` (all warnings are errors)
- `cargo fmt` — Code formatting enforcement
- `cargo test` — Full test suite with linfa cross-library validation
- `cargo build --all-features` — All feature combinations
- `pytest` — Python integration tests with pytest-cov

---

*Stack analysis: 2026-03-15*
