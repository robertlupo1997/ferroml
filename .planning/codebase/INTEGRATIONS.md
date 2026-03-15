# External Integrations

**Analysis Date:** 2026-03-15

## APIs & External Services

**GitHub:**
- Integration: Automatic releases and PyPI publishing via GitHub Actions
- Workflows: `ci.yml`, `publish-pypi.yml`, `publish.yml`, `release.yml`
- Artifacts: Automatic wheel distribution on git tags matching `v*`

## Data Storage

**Databases:**
- Not integrated — FerroML is an in-memory ML library
- No relational database (PostgreSQL, MySQL)
- No NoSQL database (MongoDB, DynamoDB)
- No data warehousing integrations

**File Storage:**
- Local filesystem only — `memmap2` for memory-mapped large arrays
- Arrow/Parquet support via `polars` for reading/writing columnar data
- No cloud storage integration (S3, GCS, Azure Blob)
- No data versioning (DVC)

**Caching:**
- None — All computations in-memory or disk via mmap
- No distributed caching (Redis)

## Data Processing & Exchange

**Python Ecosystem Interop:**
- NumPy arrays — Direct zero-copy interop via `numpy` crate
- Polars DataFrames — Native PyO3 bindings via `pyo3-polars` with derive support
- Pandas DataFrames — Not directly supported; users convert via PyArrow/Polars
- SciPy sparse matrices — PyO3 Python interop for scipy.sparse (not statically typed)

**Columnar Data Format:**
- Apache Arrow 54 — Via `arrow` crate and Polars integration
- Parquet — Via Polars' lazy execution engine
- JSON — Via Polars I/O operations
- NumPy binary format — Via ndarray serialization

## Authentication & Identity

**Auth Provider:**
- None — FerroML is a library, not a service
- No API authentication required
- GitHub Actions uses OIDC tokens for PyPI publishing (automatic via trusted publisher)

## Monitoring & Observability

**Error Tracking:**
- None — Library-level error handling via `thiserror` and `anyhow`
- No external error aggregation (Sentry, Rollbar)

**Logs:**
- Tracing infrastructure present but inactive — `tracing` 0.1 framework available
- Runtime logging: Console output via `println!` macros (no structured logging)
- Test output: Pytest verbose mode captures all printed output

**Metrics:**
- In-memory only — No external metrics collection (Prometheus, Datadog)
- Benchmarking: Criterion.rs for internal performance measurement

## CI/CD & Deployment

**Hosting:**
- GitHub — Repository and issue tracking
- PyPI — Python package index for wheel distribution
- crates.io — Rust crate registry

**CI Pipeline:**
- GitHub Actions — Multi-platform testing and publishing
- Dependabot — Automated dependency update PRs
- Pre-commit hooks — Local linting and formatting checks

**Build & Test Infrastructure:**
- Criterion.rs — Benchmark harness for performance tracking (85+ benchmarks)
- Proptest 1.5 — Property-based testing for robustness
- pytest + pytest-xdist — Python test parallelization

## Environment Configuration

**Required env vars:**
- None for normal operation
- GitHub Actions secrets: `PYPI_API_TOKEN` for PyPI publishing (via trusted publisher)

**Secrets location:**
- GitHub Actions: Protected branch secrets (trusted publisher for PyPI)
- No .env file usage in production

**Cargo.lock:**
- Committed to git — Reproducible dependency locking for library

## Testing & Validation

**Cross-Library Validation:**
- `linfa` 0.8 ecosystem — 56 integration tests in `ferroml-core/tests/vs_linfa_*.rs`
  - `linfa-linear`, `linfa-elasticnet`, `linfa-logistic`, `linfa-trees`, `linfa-svm`, `linfa-nn`, `linfa-bayes`, `linfa-reduction`, `linfa-preprocessing`, `linfa-clustering`
- `scikit-learn` 1.0+ — 44+ Python tests comparing predictions and scores
- `XGBoost` 1.x — 18 Python tests for gradient boosting models
- `LightGBM` 3.x — 18 Python tests for histogram boosting
- `scipy` — 12 Python tests for statistical validation
- `statsmodels` — 8 Python tests for linear/logistic models

**Fixtures & Test Data:**
- `ferroml-core/src/datasets/` — Toy datasets (Iris, Boston, Wine, Breast Cancer, California housing, Diabetes)
- `proptest` — Property-based data generation for fuzz testing
- Correctness tests — 26 test files with sklearn fixture validation (vibecoded)

**Benchmark Comparisons:**
- Cross-library benchmark script: `scripts/benchmark_cross_library.py`
- Compares: FerroML vs sklearn vs XGBoost vs LightGBM on 18 algorithms
- Results: JSON + Markdown output to `docs/benchmark_cross_library_results.json`

## Webhooks & Callbacks

**Incoming:**
- GitHub webhooks — Automatic trigger on push/PR for CI
- None in FerroML itself (library, not a service)

**Outgoing:**
- None — Library does not make external requests

## Package Distribution

**Python (PyPI):**
- Wheel format: `.whl` (binary, platform-specific)
- Compatibility: abi3 (stable ABI) for CPython 3.10+
- Build: Maturin 1.4+ with LTO and debug symbol stripping
- Publish trigger: Git tag matching `v*` in `publish-pypi.yml`
- Upload: PyPI via trusted publisher (OIDC, no token in secrets)

**Rust (crates.io):**
- Manual publish via `cargo publish`
- Metadata in workspace `Cargo.toml` with MIT OR Apache-2.0 dual license

## Known External Dependencies

**Development-Only (not in releases):**
- `dhat` 0.3 — Memory profiling (benchmarks only)
- `memory-stats` 1.2 — Runtime memory measurement
- `tempfile` 3.15 — Temporary directory creation for tests
- `test-case` 3.3 — Parameterized testing
- `rstest` 0.23 — Fixtures for pytest-like Rust testing
- `approx` 0.5 — Floating-point assertion helpers

**Optional in Production:**
- `wgpu` 23 + `bytemuck` 1 + `pollster` 0.4 — GPU acceleration (feature: `gpu`)
- `ort` 2.0.0-rc.11 — ONNX Runtime validation (feature: `onnx-validation`)

## Security Considerations

**No External APIs:**
- No token or credential handling required
- No network I/O in core library

**Dependency Supply Chain:**
- Dependabot monitors all Cargo and pip dependencies weekly
- Pre-release versions excluded to avoid experimental code
- Cargo.lock committed for reproducibility

**Release Process:**
- GitHub trusted publisher for PyPI (no hardcoded tokens)
- Pre-commit hooks detect private keys (blocks commits)
- Large file detection (>1MB warning)

---

*Integration audit: 2026-03-15*
