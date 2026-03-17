# External Integrations

**Analysis Date:** 2026-03-16

## APIs & External Services

**Not Detected**

FerroML is a pure machine learning library with no runtime external API integrations. It does not call out to cloud services, remote APIs, or third-party compute platforms during normal operation.

However, **optional cross-library validation** is available for development/testing:
- **scikit-learn** - Correctness validation against 60+ sklearn models (test_vs_sklearn*.py)
- **XGBoost** - GBM comparison (test_vs_xgboost.py)
- **LightGBM** - GBDT comparison (test_vs_lightgbm.py)
- **statsmodels** - Statistical model validation (test_vs_statsmodels.py)
- **scipy** - Distribution and numerical algorithm validation
- **linfa 0.8** - 12 Rust cross-library validation tests (linear, trees, SVM, clustering)

These are **optional dev-dependencies only**, not required for production use.

## Data Storage

**File Formats Supported:**

**CSV:**
- Location: `ferroml-core/src/datasets/file.rs`
- Function: `load_csv()`, `load_csv_with_options()`
- Backend: Polars with arrow-backed columnar format
- Config: `CsvOptions` (delimiter, encoding, null handling)
- Type inference: Automatic (Polars type inference)
- Feature-gated: `datasets` (enabled by default)

**Parquet:**
- Location: `ferroml-core/src/datasets/file.rs`
- Function: `load_parquet()`, `load_parquet_with_options()`
- Backend: Polars + Arrow
- Config: `ParquetOptions` (compression, row group size)
- Zero-copy: Arrow columnar data directly to ndarray
- Feature-gated: `datasets` (enabled by default)

**Memory-Mapped Files:**
- Location: `ferroml-core/src/datasets/mmap.rs`
- Types: `MemmappedDataset`, `MemmappedArray2`, `MemmappedArray1`
- Backend: `memmap2` 0.9
- Use case: Datasets > RAM (zero-copy file access)
- Format: Custom `.fmm` binary format (native ndarray layout via bincode)
- Example: `ferroml-core/tests/test_mmap_dataset.rs`

**NumPy Interop:**
- Via pyo3: Python numpy.ndarray ↔ ndarray conversions
- No external NumPy calls from Rust code; pure data exchange in Python bindings
- Location: `ferroml-python/src/` (PyO3 wrappers)

**Serialization Formats:**
- Binary: `bincode` 1.3 (model snapshots)
- MessagePack: `rmp-serde` 1.3 (cross-language model exchange)
- JSON: `serde_json` 1.0 (config, metadata)
- Protocol Buffers: `prost` 0.13 (ONNX export)

**Local Filesystem Only:**
- No cloud storage (S3, GCS, Azure Blob)
- No database connections (MySQL, PostgreSQL, MongoDB)
- No network file systems (NFS, SMB)

## Caching

**Not Detected**

FerroML does not use external caching systems. All caching is in-process:
- Model warm-start state (cached normalization params in `automl/warmstart.rs`)
- SVC kernel cache (LRU cache with 10K sample threshold in `models/svm/svc.rs`)
- Ensemble bootstrap cache (cached decision functions for AdaBoost)

## Authentication & Identity

**Not Applicable**

No authentication or authorization system. FerroML is a pure compute library without user/tenant isolation. No cloud authentication (AWS IAM, Azure AD, Okta) needed.

## Monitoring & Observability

**Logging:**
- Framework: `tracing` 0.1 (structured logging)
- Backend: `tracing-subscriber` 0.3 (configurable formatter)
- Initialization: Optional in Python bindings (`init_tracing()` if exposed)
- Usage: Event and span logging for debug/development (not required for production)

**Error Tracking:**
- Not detected. FerroML returns `Result<T, FerroError>` with detailed error types:
  - `ShapeMismatch` - Array dimension mismatches
  - `AssumptionViolation` - Statistical assumption failures
  - `ConvergenceFailure` - Optimizer/solver non-convergence
  - `InvalidInput` - Data validation errors
  - Location: `ferroml-core/src/error.rs`

**Performance Metrics:**
- Criterion benchmarks to `target/criterion/` (local file output)
- Memory profiling via `dhat` (heap allocation tracking)
- No external APM (New Relic, DataDog, Prometheus)

**Coverage Reporting:**
- `cargo tarpaulin` (Rust code coverage)
- CI uploads to Codecov (via `codecov/codecov-action@v4`)
- Threshold: 70% minimum, 75% target

## CI/CD & Deployment

**Hosting:**
- GitHub (primary repository: github.com/robertlupo1997/ferroml)
- PyPI (package distribution: pypi.org/project/ferroml/)
- crates.io (Rust crate: crates.io/crates/ferroml-core)
- GitHub Pages (documentation, if enabled)

**CI Platform:**
- GitHub Actions (workflows in `.github/workflows/`)
- Runners: ubuntu-latest, macos-latest, windows-latest
- Caching: GitHub Actions cache API

**Build Infrastructure:**
- `maturin` 1.4+ - PyO3 wheel building
- `PyO3/maturin-action` - GitHub Actions workflow for cross-platform wheel builds
- `sccache` - Rust compilation caching in CI

**Publishing:**
- PyPI: `pypa/gh-action-pypi-publish` (trusted publishing via OIDC)
- Fallback: `PYPI_API_TOKEN` secret (if trusted publishing not configured)
- TestPyPI: Dry-run publishing before production (test.pypi.org)
- Crates.io: Manual `cargo publish` (not automated in CI yet)

**Release Management:**
- Git tags trigger PyPI publish (v-prefixed semver: v0.3.1)
- Changelog: `cliff.toml` (automated generation from conventional commits)
- Version sync: Workspace-level in `Cargo.toml`, `pyproject.toml` mirror required

## Environment Configuration

**Required Environment Variables:**
- None at runtime (pure compute library)
- Optional during build:
  - `CARGO_TERM_COLOR` - Set to `always` in CI for colored output
  - `RUSTFLAGS` - Set to `-D warnings` in CI to enforce no warnings

**Build-Time Secrets (for CI/CD only):**
- `CODECOV_TOKEN` - For coverage upload (optional)
- `PYPI_API_TOKEN` - For PyPI publishing (fallback to trusted publishing)
- `TEST_PYPI_API_TOKEN` - For TestPyPI publishing (optional, for testing)
- **Note:** These are GitHub Actions secrets, not application environment variables

**Secrets Location:**
- GitHub Actions: `Settings → Secrets and variables → Actions`
- CI workflow references: `${{ secrets.PYPI_API_TOKEN }}`
- Never committed to repository (`.gitignore` includes `.env*`)

**No Configuration Files for App Secrets:**
- `.env`, `.env.local`, `config/secrets.json` - Not used
- All configuration is code-based (Cargo.toml, pyproject.toml)

## Webhooks & Callbacks

**Not Detected**

FerroML does not expose webhooks, callbacks, or event subscription mechanisms. It is a pure compute library:
- No HTTP servers exposed
- No event streams
- No pub/sub integrations
- No async event handlers (though `tokio` is available for potential future features)

---

## Summary: Standalone Library with File I/O

FerroML is a **self-contained ML library with no external runtime dependencies**:
- ✅ Works offline (no cloud API calls)
- ✅ Deterministic and reproducible (pure compute)
- ✅ No secrets required at runtime
- ✅ Local data processing only (CSV, Parquet, memory-mapped files)
- ✅ Optional cross-library validation for development

### Deployment Model:
1. **Rust crate** (ferroml-core) - Link against Rust projects, zero external deps
2. **Python package** (ferroml) - Install via pip, depends only on NumPy 1.21+
3. **Distribution** - Wheels for Linux/macOS/Windows, source distribution for edge cases

All models and utilities are self-contained and require no external service integrations.

---

*Integration audit: 2026-03-16*
