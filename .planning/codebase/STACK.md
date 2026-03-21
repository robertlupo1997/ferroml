# Technology Stack

**Analysis Date:** 2026-03-20

## Languages

**Primary:**
- Rust 1.75.0 (MSRV) - Core ML library in `ferroml-core/`
- Python 3.10+ - Bindings and user-facing API in `ferroml-python/`

**Secondary:**
- GLSL/WGSL - GPU shader code (optional GPU support)

## Runtime

**Environment:**
- Cargo workspace with two crates: `ferroml-core` (library) and `ferroml-python` (PyO3 bindings)
- Python 3.10, 3.11, 3.12 supported (from `pyproject.toml`)

**Package Managers:**
- Cargo - Rust dependency management
- pip/maturin - Python wheel building and distribution
- Lockfile: `Cargo.lock` present at workspace root

**Python Build System:**
- maturin 1.4+ (`maturin develop --release -m ferroml-python/Cargo.toml`)
- Wheel compatibility: abi3-py310 (stable ABI, forward compatible to Python 3.15+)

## Frameworks

**Core Numerical:**
- ndarray 0.16 (with rayon, serde features) - Dense array operations and main data structure
- nalgebra 0.33 (with serde-serialize) - Linear algebra (SVD, Cholesky, eigenvalue decomposition)
- statrs 0.18 - Statistical distributions (Beta, Gamma, Normal, Student-t, etc.)

**Note:** ndarray-linalg removed (required OpenBLAS system dependency). Uses nalgebra instead for all linear algebra.

**Data Processing:**
- polars 0.46 (lazy, parquet, json, dtype-full) - Optional DataFrames and data loading
- arrow 54 - Columnar format support (optional, bundled with Polars)

**Parallel Computing:**
- rayon 1.10 - Data parallelism (default enabled)
- crossbeam 0.8 - Thread synchronization primitives
- tokio 1 (full features) - Async runtime (optional, dev-dependency)

**Optimization & Algorithms:**
- argmin 0.10 + argmin-math 0.4 - Optimization algorithms (L-BFGS, SAG, SAGA)
- wide 0.7 - Portable SIMD for distance calculations (optional feature)
- faer 0.20 - High-performance linear algebra backend (optional, default enabled)

**Sparse Matrices:**
- sprs 0.11 - Sparse matrix operations (CSR/CSC formats, optional feature)

**Memory & Storage:**
- memmap2 0.9 - Memory-mapped files for large datasets
- bincode 1.3 - Binary serialization (model persistence)
- rmp-serde 1.3 - MessagePack serialization (alternative to bincode)

**Serialization & Export:**
- serde 1.0 (with derive) - Serialization framework
- serde_json 1.0 - JSON support
- prost 0.13 - Protocol buffers (for ONNX export)

**Error Handling:**
- thiserror 2.0 - Structured error definitions
- anyhow 1.0 - Generic error context wrapper

**Logging:**
- tracing 0.1 - Structured logging framework
- tracing-subscriber 0.3 - Log filtering and formatting

## Python Bindings

**PyO3 Integration:**
- pyo3 0.24 (extension-module, abi3-py310 features) - Python FFI bindings
- numpy 0.24 - NumPy array interop
- pyo3-polars 0.21 (with derive feature) - Polars DataFrame integration for Python

**Exposed to Python:**
- 55+ models across 14 submodules (linear, trees, ensemble, clustering, etc.)
- 22+ preprocessors (scalers, encoders, imputers, CountVectorizer, etc.)
- Full statistical diagnostics (residual analysis, assumption tests)
- ONNX export support

## Optional Features

**Default Features:**
- `parallel` - Rayon-based parallelism (enabled by default)
- `onnx` - ONNX model export support
- `simd` - SIMD acceleration via wide
- `faer-backend` - High-performance linear algebra
- `datasets` - CSV/Parquet loading via Polars

**Opt-in Features:**
- `sparse` - Native sparse matrix operations (enabled for Python bindings)
- `gpu` - GPU acceleration via wgpu + bytemuck + pollster
- `onnx-validation` - ONNX round-trip testing via ort 2.0.0-rc.11

## Key Dependencies (Critical)

**Essential:**
- ndarray 0.16 - All matrix operations depend on this
- nalgebra 0.33 - All decompositions (SVD, Cholesky, eigenvalue) depend on this
- statrs 0.18 - Statistical distributions for model implementations
- rand 0.9 + rand_distr 0.5 - Random number generation
- pyo3 0.24 - Python bindings compilation

**For Cross-Library Validation (dev-only):**
- linfa 0.8.x (+ linfa-* sub-crates) - Cross-validation against linfa algorithms

## Configuration

**Rust Compiler Settings:**
- MSRV: 1.75.0 (set in `ferroml-core/Cargo.toml`)
- Edition: 2021
- Release profile: thin LTO, codegen-units=1, opt-level=3
- Clippy: `-D warnings` enforced (all warnings treated as errors)

**Rustfmt/Clippy:**
- Required components: rustfmt, clippy
- Pre-commit hook enforces `cargo fmt --all`
- All targets built for: x86_64-unknown-linux-gnu, x86_64-apple-darwin, aarch64-apple-darwin, x86_64-pc-windows-msvc

**Python Package Configuration:**
- Maturin build backend
- Python source directory: `python/`
- Module name: `ferroml`
- Strip debug symbols for smaller wheels
- docs.rs: all features enabled

**Feature Flag Structure:**
- Default gates around expensive optional deps (polars, faer, wgpu)
- Each model can be compiled with/without optional features
- Tests use `linfa` (dev-dependency only) for correctness validation

## Platform Requirements

**Development:**
- Rust 1.75.0+ (stable)
- Python 3.10+ (3.12 recommended for dev)
- C compiler toolchain (for Rust compilation)
- ~14 GB disk for debug builds (optimized from 350 GB via test consolidation)

**Production:**
- Linux (x86_64, aarch64), macOS (x86_64, aarch64), Windows (x86_64)
- CPython 3.10+ (abi3 wheels work with 3.10-3.15+)
- No external system dependencies for linear algebra (ndarray-linalg removed)
- Optional: Polars for CSV/Parquet support; wgpu for GPU features

**Deployment:**
- PyPI: wheels published for CPython 3.10-3.12 (Linux, macOS, Windows)
- crates.io: ferroml-core published for Rust consumers
- GitHub Releases: source archives + compiled wheels

---

*Stack analysis: 2026-03-20*
