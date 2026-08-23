# Test-run artifact — FerroML v1.0.2

Produced 2026-08-22 on WSL2 (Linux 6.18.33.2-microsoft-standard-WSL2), Python 3.12.3,
against the release-mode extension built by `maturin develop --release`.

This file exists to settle one question: **are FerroML's published test numbers true?**

**Provenance.** The cargo log was recorded by the pilot iteration at 17:39 on 2026-08-22.
The maturin build (21:47) and the Python run (22:20) were produced by a later Claude Code
session on the same machine and the same commit. No suite was re-run to obtain a better
number; the Python suite was run twice only because the first run lacked `typer`, and both
logs are described below.

## Result

| Suite | Passed | Failed | Other | Exit |
|---|---|---|---|---|
| Rust — `cargo test --all-features` | **5,225** | 0 | 30 ignored | 0 |
| Python — `pytest` (built wheel, all extras) | **2,971** | 0 | 21 skipped, 4 xfailed | 0 |
| **Total passing** | **8,196** | **0** | | |

## What each figure means

**Rust: 5,225.** `cargo-test-all-features.log` contains two complete, identical runs
concatenated (an append, not a fresh write). One run is 8 result blocks:

```
3819 + 88 + 304 + 582 + 103 + 36 + 56 + 237(doc-tests) = 5225 passed, 0 failed, 30 ignored
```

Do not sum the whole file — that double-counts to 10,450. `CARGO_TEST_EXIT_CODE=0`.

**Python: 2,971.** `pytest-built-wheel.log`. The suite collects 2,996 tests in total, so
2,971 passing is the ceiling for this suite as it stands. `EXIT_CODE=0`.

## Why earlier runs failed, and why none of it was a code defect

| Earlier symptom | Real cause | Fix |
|---|---|---|
| `maturin failed: Failed to find pip` | maturin ran without the venv active; `.venv/bin/pip` exists | activate `.venv` first |
| 19 collection errors | `ModuleNotFoundError: No module named 'lightgbm'` | `pip install lightgbm xgboost` |
| 23 `test_cli.py` failures | `ModuleNotFoundError: No module named 'typer'` | install the `cli` extra |

The `typer` gap was a genuine packaging defect, not a local accident: `test` resolved to
`ferroml[dev,sklearn,pandas]`, which never pulls in `cli`, while `ferroml/cli/__init__.py`
imports `typer` at module scope. A clean `pip install -e ".[test]" && pytest` therefore
failed 23 tests on any machine, CI included. Fixed in this branch — `test` now resolves to
`ferroml[dev,sklearn,pandas,cli]`.

## Reproducing

```sh
cd <repo>
source .venv/bin/activate
pip install -e "ferroml-python[test]"
pip install lightgbm xgboost          # benchmark comparisons only
cd ferroml-python && maturin develop --release
python -m pytest -q
cd .. && cargo test --all-features
```

## What this settles

`AUDIT-OX.md` rated "7,600+ tests **passing**" and "3,000+ Python" UNPROVABLE for want of
a recorded run. This is that record.

- "tests passing" — **now supported.** 8,196 pass, 0 fail, both suites exit 0.
- "4,600+ Rust" — **supported.** 5,225 passed.
- "3,000+ Python" — **still false.** 2,971 passed, and the suite only collects 2,996, so
  the figure is unreachable without writing new tests.

The honest headline is larger than the claim it replaces: **8,196**, not "7,600+".
