# External Integrations

**Analysis Date:** 2026-03-20

## APIs & External Services

**None Currently Implemented**

FerroML is a self-contained ML library. It does not depend on external ML services, cloud APIs, or remote computation endpoints. All computation occurs locally (CPU or optional GPU).

## Data Storage

**Databases:**
- Not applicable - FerroML is an in-memory ML library
- Supports optional loading from CSV/Parquet via Polars (`datasets` feature)

**File Storage:**
- Local filesystem only
- Memory-mapped file support via `memmap2` for large datasets
- Binary serialization: bincode and MessagePack via `bincode` and `rmp-serde`
- ONNX export: `prost` for protocol buffers

**Caching:**
- In-process caching only (no external cache service)
- Model parameters cached in memory during training
- MLP layer forward-pass caching for batch inference

## Authentication & Identity

**Not applicable** - FerroML has no user authentication, API keys, or identity management.

## Monitoring & Observability

**Error Tracking:**
- None - errors propagate as `FerroError` enum variants to caller

**Logs:**
- Structured logging via `tracing` 0.1
- Console/file output via `tracing-subscriber` 0.3
- No external log aggregation (logs are caller-controlled)
- Example: AutoML verbose mode logs model evaluations to stderr

**Profiling:**
- Benchmarking: Criterion.rs via `benches/` (dev-only)
- Memory profiling: dhat and memory-stats (dev-only)

## Cross-Library Validation (Development Only)

**Verification Against:**
- scikit-learn 1.0+ (Python tests compare predictions, coefficients, probabilities)
- scipy - distributions and special functions
- linfa 0.8.1 (Rust tests for algorithms, dev-dependency)
- xgboost, lightgbm (cross-library performance benchmarks in `scripts/benchmark_cross_library.py`)
- statsmodels (statistical test validation)

**Test Infrastructure:**
- conftest_comparison.py - Shared fixtures for sklearn dataset loading and comparison
- test_vs_sklearn_*.py files - 12 cross-library validation test files
- test_vs_linfa.rs - Rust-side cross-validation against linfa
- ONNX round-trip validation (118 tests) - verifies export/import fidelity

## CI/CD & Deployment

**Hosting:**
- GitHub (robertlupo1997/ferroml) - source repository
- GitHub Releases - version tags and release artifacts

**CI Pipeline:**
- GitHub Actions (8 workflows in `.github/workflows/`)

**Workflows:**
1. `ci.yml` - Check, clippy, fmt, tests (on push/PR to main/master)
2. `publish-pypi.yml` - Build and publish wheels to PyPI (on version tags or manual dispatch)
3. `publish.yml` - Publish crate to crates.io (Rust)
4. `benchmarks.yml` - Run Criterion benchmarks and compare against baseline
5. `release.yml` - Automated version bumping and release notes
6. `changelog.yml` - Generate CHANGELOG via git-cliff
7. `docs.yml` - Build and publish docs to docs.rs
8. `fuzz.yml` - Fuzzing tests (libfuzzer)
9. `mutation.yml` - Mutation testing for test quality

**Deployment Targets:**
- PyPI: `ferroml` package (CPython 3.10-3.12, wheels for Linux/macOS/Windows)
- crates.io: `ferroml-core` crate (Rust library)
- docs.rs: Rust API documentation (all features enabled)

**Release Management:**
- Conventional commits (feat:, fix:, perf:, etc.)
- git-cliff for changelog generation (`cliff.toml` config)
- Semantic versioning (v0.3.1 as of 2026-03-20)
- Stable Rust channel required

## Environment Configuration

**Required Environment Variables:**
- None - FerroML requires no environment configuration
- Optional: RUSTFLAGS (for Rust compilation), CARGO_TERM_COLOR (for colored output)

**Build Configuration:**
- Rust: `rust-toolchain.toml` - MSRV 1.75.0, stable channel
- Cargo: `Cargo.toml` (workspace root) - all dependencies pinned
- Python: `pyproject.toml` - maturin build backend, Python 3.10+ requirement

**Secrets Location:**
- Not applicable - no API keys or credentials used
- Note: `.env` files MUST NOT be committed (covered by `.gitignore`)

## Webhooks & Callbacks

**Incoming:**
- None - FerroML is not a server

**Outgoing:**
- None - FerroML does not call external services

## Development Dependencies (Optional)

**Testing Ecosystem:**
- pytest 7.0+ - Python test runner
- pytest-cov 4.0+ - Coverage reporting
- pytest-xdist 3.0+ - Parallel test execution
- scikit-learn 1.0+ (optional, for cross-library tests)
- pandas 1.5+ (optional, for DataFrame integration tests)
- polars 0.19+ (optional, for Polars integration tests)

**Code Quality:**
- ruff 0.1+ - Python linter
- mypy 1.0+ - Python type checking
- cargo clippy - Rust linter (in-tree, enabled via Rust toolchain)
- cargo fmt - Rust formatter (in-tree)

**Documentation & Changelog:**
- git-cliff - Changelog generation from conventional commits
- Sphinx (optional, for extended docs in `docs/` if needed)

**Performance Analysis:**
- Criterion 0.5 - Rust benchmarking framework
- dhat 0.3 - Heap allocation profiling
- memory-stats 1.2 - Memory usage tracking

**Property Testing:**
- proptest 1.5 - Property-based testing
- test-case 3.3 - Parameterized tests
- rstest 0.23 - Fixture-based testing

## Data Format Support

**Input Formats:**
- NumPy arrays (np.ndarray) - primary
- Polars DataFrames (optional, via `datasets` feature)
- Pandas DataFrames (interop via PyArrow in Python)
- CSV/Parquet (via Polars optional dependency)

**Output/Export Formats:**
- NumPy arrays
- ONNX models (all 55+ models, validation via ort 2.0.0-rc.11)
- Bincode (binary serialization, not human-readable)
- MessagePack (via rmp-serde)
- JSON (serde_json, for some metadata)

## Sparse Matrix Support

**Integration:**
- sprs 0.11 (CSR/CSC formats, optional `sparse` feature)
- Exposed in Python bindings for 12 models
- SparseModel trait for algorithms supporting sparse inputs
- SciPy sparse matrix interop (Python-side, via PyO3)

## Known Integration Boundaries

**Not Integrated:**
- Database connectors (users must load via Polars/Pandas)
- Web frameworks (Flask, FastAPI) - users wrap models themselves
- Cloud platforms (AWS SageMaker, GCP Vertex AI, Azure ML) - no native SDKs
- Distributed computing (Spark, Dask, Ray) - single-machine only
- GPU frameworks (TensorFlow, PyTorch) - has own optional wgpu GPU layer

**Design Pattern:**
FerroML follows Unix philosophy: does one thing (ML algorithms) exceptionally well. Users compose it with their choice of data loading, serving, and orchestration tools.

---

*Integration audit: 2026-03-20*
