# Plan: Fix FerroML PyPI Publish Workflow

**Date**: 2026-03-31
**Status**: Ready for execution
**Goal**: Make `publish-pypi.yml` pass dry-run and successfully publish to TestPyPI, then PyPI

---

## Root Cause Analysis

We attempted 5 dry runs. All failed during the **Validate** job. The Validate job tried to be a mini-CI — running clippy, cargo test, maturin develop, and the full pytest suite. This is fundamentally wrong.

### Every major PyO3 project (pydantic-core, polars, ruff, watchfiles) does NOT run tests in the publish workflow.

They rely on CI passing before the tag is pushed. The publish workflow only:
1. Builds wheels for all platforms
2. Smoke-tests: `pip install <wheel> && python -c "import pkg"`
3. Publishes

Our Validate job has been a dependency-management nightmare because:
- **Run 1**: `--all-features` compiled wgpu/ort (45 min timeout)
- **Run 2**: 29 test files needed sklearn/scipy (not installed)
- **Run 3**: 13 more test files needed sklearn/statsmodels (missed in ignore list)
- **Run 4**: test_cli.py needed typer (not installed)
- **Run 5**: test_cli.py's train/predict commands need polars at runtime (not installed) — 18 failures, 2009 passed

Each fix just revealed the next missing dependency. The test suite has 90 files with a web of dependencies (sklearn, scipy, statsmodels, xgboost, lightgbm, polars, typer, rich). Trying to run a subset in the publish workflow is a game of whack-a-mole.

---

## The Fix: Follow Industry Standard

Replace the entire Validate job with what pydantic-core and polars do:

### 1. Version check only (no clippy, no cargo test, no pytest)

The CI workflow already runs all of these on every push. By the time someone pushes a tag, the code has passed 22 CI jobs. Re-running tests in publish is redundant and fragile.

### 2. Smoke test AFTER building wheels (not before)

Instead of `maturin develop` + pytest (which compiles Rust in debug mode), install the actual built wheel and verify it imports. This tests what users will actually get.

### 3. Keep the existing build matrix

The Linux/macOS/Windows wheel builds via maturin-action are correct and don't need changes.

---

## Implementation Plan

### Step 1: Rewrite the Validate job

Replace the current 90-line Validate job with a ~20-line version check:

```yaml
validate:
  name: Validate
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Extract version from tag
      if: github.event_name == 'push'
      id: version
      run: |
        TAG=${GITHUB_REF#refs/tags/v}
        echo "version=$TAG" >> $GITHUB_OUTPUT

    - name: Verify version matches pyproject.toml
      if: github.event_name == 'push'
      run: |
        PYPROJECT_VERSION=$(grep '^version' ferroml-python/pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
        CARGO_VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
        TAG_VERSION=${{ steps.version.outputs.version }}
        if [ "$PYPROJECT_VERSION" != "$TAG_VERSION" ]; then
          echo "Error: Tag ($TAG_VERSION) != pyproject.toml ($PYPROJECT_VERSION)"
          exit 1
        fi
        if [ "$CARGO_VERSION" != "$TAG_VERSION" ]; then
          echo "Error: Tag ($TAG_VERSION) != Cargo.toml ($CARGO_VERSION)"
          exit 1
        fi
        echo "Version verified: $TAG_VERSION"
```

This takes **~10 seconds** instead of 60+ minutes.

For `workflow_dispatch` (dry runs), skip the version check since there's no tag.

### Step 2: Add smoke test to Dry Run Check job

After building all wheels, verify the Linux wheel actually works:

```yaml
dry-run-check:
  # ... (existing artifact download) ...
  - name: Smoke test wheel
    run: |
      WHEEL=$(ls dist/*linux*x86_64*.whl 2>/dev/null | head -1)
      if [ -n "$WHEEL" ]; then
        pip install "$WHEEL"
        python -c "import ferroml; print(f'FerroML {ferroml.__version__} imported successfully')"
        python -c "from ferroml.linear import LinearRegression; print('LinearRegression OK')"
        python -c "from ferroml.trees import RandomForestClassifier; print('RandomForestClassifier OK')"
        pip uninstall -y ferroml
      fi
```

### Step 3: Remove unnecessary fields

- Remove `RUSTFLAGS: -D warnings` from env (not building Rust in this workflow anymore)
- Remove Rust toolchain setup from validate job
- Remove cargo cache from validate job

### Step 4: Verify dry run passes

```bash
gh workflow run publish-pypi.yml
# Wait ~15 min (just wheel builds, no Rust compile in validate)
gh run view <id>
```

### Step 5: Test publish to TestPyPI

```bash
gh workflow run publish-pypi.yml -f dry_run=false -f publish_target=testpypi
```

Verify at https://test.pypi.org/project/ferroml/

### Step 6: Tag and publish to PyPI

```bash
git tag v1.0.0
git push origin v1.0.0
```

This triggers the publish workflow with the tag, which runs version check + builds + publishes.

---

## What Changes

| File | Change |
|------|--------|
| `.github/workflows/publish-pypi.yml` | Rewrite validate job (remove clippy/test/pytest, keep version check). Add smoke test to dry-run-check. Clean up env vars. |

**One file. One commit.**

## What Doesn't Change

- Build jobs (linux, macos, windows, sdist) — already correct
- Publish jobs (pypi, testpypi) — already correct with trusted publishing
- Dry run check job — keep, just add smoke test
- CI workflow — continues to run all 22 jobs on every push

## Risk Assessment

- **Low risk**: We're removing redundant testing, not adding new functionality
- **CI already gates**: All 22 jobs pass on every push to master
- **Smoke test catches real issues**: If the wheel is broken, the import will fail
- **Version check catches tag mistakes**: Prevents publishing wrong version

## Time Estimate

- Rewrite: 5 minutes (it's mostly deletion)
- Dry run: ~15 minutes (wheel builds only)
- TestPyPI: ~15 minutes
- PyPI: ~15 minutes + tag push

---

## Appendix: All 5 Failure Details

| Run | Duration | Failure | Root Cause |
|-----|----------|---------|------------|
| 23769801810 | 45m (cancelled) | Timeout in compile | `--all-features` compiled wgpu + ort |
| 23771369551 | 65m | 29 import errors | sklearn/scipy not installed, 29 test files need them |
| 23773344401 | 65m | 13 import errors | More sklearn/statsmodels files missed by ignore patterns |
| 23775128504 | 65m | typer not found | test_cli.py imports typer via subprocess |
| 23776960312 | 70m | 18 CLI failures | CLI train/predict need polars at runtime, not installed |

Pattern: Each fix revealed the next missing dependency. The test dependency graph is too complex for the publish workflow.

## Appendix: Industry Comparison

| Project | Tests in publish? | Validate time | Total publish time |
|---------|-------------------|---------------|-------------------|
| pydantic-core | No (import only) | ~10s | ~15 min |
| polars | No (import only) | ~10s | ~20 min |
| ruff | No | ~10s | ~10 min |
| watchfiles | No (import only) | ~10s | ~10 min |
| **FerroML (current)** | **Yes (2000+ tests)** | **60+ min** | **Never completes** |
| **FerroML (proposed)** | **No (import only)** | **~10s** | **~15 min** |
