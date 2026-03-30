# Phase 5: CLI Tool + AI Agent Discoverability — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `ferroml` CLI that lets AI agents (and humans) train, predict, evaluate, and diagnose ML models from the terminal with structured JSON output.

**Architecture:** Typer-based CLI in `ferroml-python/python/ferroml/cli/` with 9 subcommands. Each command loads data via polars, calls existing FerroML Python APIs, and outputs either Rich tables (human) or JSON (agent). Model persistence via pickle (NOTE: pickle is used here intentionally — it is the ML ecosystem standard for model serialization, used by scikit-learn/joblib. FerroML models already implement `__getstate__`/`__setstate__` via MessagePack under the hood, so pickle is the established pattern in this codebase. Only user-trained models are serialized/loaded — no untrusted content). Installed as `ferroml[cli]` optional dependency group.

**Tech Stack:** Typer (CLI framework), Rich (human-readable output), Polars (CSV/Parquet I/O), pickle (model persistence — established pattern in this codebase)

---

## Task 1: Scaffold CLI Package + Entry Point

**Files:**
- Create: `ferroml-python/python/ferroml/cli/__init__.py`
- Create: `ferroml-python/python/ferroml/cli/__main__.py`
- Create: `ferroml-python/python/ferroml/cli/_io.py`
- Create: `ferroml-python/python/ferroml/cli/_format.py`
- Modify: `ferroml-python/pyproject.toml`
- Create: `ferroml-python/tests/test_cli.py`

**Step 1: Write the failing test**

```python
# ferroml-python/tests/test_cli.py
"""Tests for the ferroml CLI."""
import subprocess
import sys

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the ferroml CLI as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "ferroml.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestCLIEntryPoint:
    def test_help(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "ferroml" in result.stdout.lower()

    def test_version(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "1.0.0" in result.stdout

    def test_no_args_shows_help(self):
        result = run_cli()
        assert result.returncode == 0
```

**Step 2: Run test to verify it fails**

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestCLIEntryPoint -v`
Expected: FAIL (no module ferroml.cli)

**Step 3: Add dependencies to pyproject.toml**

In `ferroml-python/pyproject.toml`, add under `[project.optional-dependencies]`:

```toml
cli = [
    "typer>=0.9",
    "rich>=13.0",
    "polars>=0.19",
]
```

And add under `[project.scripts]`:

```toml
[project.scripts]
ferroml = "ferroml.cli:app"
```

And update `all` to include cli:

```toml
all = [
    "ferroml[polars,pandas,sklearn,cli]",
]
```

**Step 4: Install CLI deps**

Run: `source .venv/bin/activate && pip install typer rich polars`

**Step 5: Create the CLI scaffold**

```python
# ferroml-python/python/ferroml/cli/__init__.py
"""FerroML command-line interface."""
import typer

from ferroml import __version__

app = typer.Typer(
    name="ferroml",
    help="FerroML: Statistically rigorous ML from the command line.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"ferroml {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit.", callback=version_callback, is_eager=True,
    ),
):
    """FerroML: Statistically rigorous ML from the command line."""


# __main__ support: python -m ferroml.cli
def cli_main():
    app()
```

```python
# ferroml-python/python/ferroml/cli/__main__.py
"""Allow running CLI as `python -m ferroml.cli`."""
from ferroml.cli import cli_main

if __name__ == "__main__":
    cli_main()
```

**Step 6: Create the I/O helpers**

```python
# ferroml-python/python/ferroml/cli/_io.py
"""Data loading and model persistence for the CLI."""
from __future__ import annotations

import pickle  # Used intentionally: ML ecosystem standard, only loads user's own models
import sys
from pathlib import Path

import numpy as np


def load_data(path: str, target: str | None = None) -> tuple:
    """Load CSV or Parquet file, split into X (and optionally y).

    Returns (X, y, feature_names) if target is given, else (X, None, feature_names).
    """
    import polars as pl

    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    if p.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif p.suffix in (".csv", ".tsv"):
        df = pl.read_csv(path)
    else:
        print(f"Error: unsupported file format: {p.suffix} (use .csv or .parquet)", file=sys.stderr)
        raise SystemExit(1)

    if target and target not in df.columns:
        print(f"Error: target column '{target}' not found. Columns: {df.columns}", file=sys.stderr)
        raise SystemExit(1)

    if target:
        y = df[target].to_numpy().astype(np.float64)
        X = df.drop(target).to_numpy().astype(np.float64)
        feature_names = [c for c in df.columns if c != target]
    else:
        X = df.to_numpy().astype(np.float64)
        y = None
        feature_names = df.columns

    return X, y, feature_names


def save_model(model, path: str) -> None:
    """Save a fitted model to disk via pickle.

    Note: FerroML models implement __getstate__/__setstate__ using MessagePack
    serialization under the hood. Pickle is used as the standard ML ecosystem
    interface (compatible with joblib, sklearn patterns).
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    """Load a fitted model from disk via pickle.

    Only loads models that were saved by this CLI — not for untrusted content.
    """
    p = Path(path)
    if not p.exists():
        print(f"Error: model file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301
```

**Step 7: Create the format helpers**

```python
# ferroml-python/python/ferroml/cli/_format.py
"""Output formatting for the CLI (JSON vs Rich tables)."""
from __future__ import annotations

import json
import sys


def output(data: dict | list, json_mode: bool) -> None:
    """Print data as JSON (to stdout) or as a Rich table (to stderr + stdout)."""
    if json_mode:
        print(json.dumps(data, indent=2, default=str))
    else:
        _print_rich(data)


def _print_rich(data: dict | list) -> None:
    """Pretty-print data using Rich."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console(stderr=True)

        if isinstance(data, list) and data and isinstance(data[0], dict):
            table = Table(show_header=True)
            for key in data[0]:
                table.add_column(str(key))
            for row in data:
                table.add_row(*[str(row.get(k, "")) for k in data[0]])
            console.print(table)
        elif isinstance(data, dict):
            table = Table(show_header=True)
            table.add_column("Key")
            table.add_column("Value")
            for k, v in data.items():
                table.add_row(str(k), str(v))
            console.print(table)
        else:
            console.print(data)
    except ImportError:
        # Fallback if rich not installed
        print(json.dumps(data, indent=2, default=str))
```

**Step 8: Run tests to verify they pass**

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestCLIEntryPoint -v`
Expected: PASS (3 tests)

**Step 9: Commit**

```bash
git add ferroml-python/python/ferroml/cli/ ferroml-python/tests/test_cli.py ferroml-python/pyproject.toml
git commit -m "feat(cli): scaffold CLI package with entry point, I/O, and format helpers"
```

---

## Task 2: Model Registry + `ferroml train`

**Files:**
- Create: `ferroml-python/python/ferroml/cli/_registry.py`
- Create: `ferroml-python/python/ferroml/cli/train.py`
- Modify: `ferroml-python/python/ferroml/cli/__init__.py`
- Modify: `ferroml-python/tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `test_cli.py`:

```python
import json
import os
import tempfile

import numpy as np


def _make_csv(tmp_path: str, n: int = 50, task: str = "regression") -> str:
    """Create a simple CSV dataset for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, 3)
    if task == "regression":
        y = X[:, 0] * 2 + X[:, 1] + rng.randn(n) * 0.1
    else:
        y = (X[:, 0] > 0).astype(float)
    path = os.path.join(tmp_path, "data.csv")
    header = "f0,f1,f2,target"
    rows = [f"{x[0]},{x[1]},{x[2]},{yi}" for x, yi in zip(X, y)]
    with open(path, "w") as f:
        f.write(header + "\n" + "\n".join(rows))
    return path


class TestTrain:
    def test_train_linear_regression(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        model_path = str(tmp_path / "model.pkl")
        result = run_cli("train", "--model", "LinearRegression", "--data", csv_path,
                         "--target", "target", "--output", model_path)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(model_path)

    def test_train_json_output(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        model_path = str(tmp_path / "model.pkl")
        result = run_cli("train", "--model", "LinearRegression", "--data", csv_path,
                         "--target", "target", "--output", model_path, "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["model"] == "LinearRegression"
        assert data["status"] == "fitted"

    def test_train_with_params(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        model_path = str(tmp_path / "model.pkl")
        result = run_cli("train", "--model", "RidgeRegression", "--data", csv_path,
                         "--target", "target", "--output", model_path,
                         "--params", '{"alpha": 0.5}')
        assert result.returncode == 0, result.stderr

    def test_train_with_test_size(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        model_path = str(tmp_path / "model.pkl")
        result = run_cli("train", "--model", "LinearRegression", "--data", csv_path,
                         "--target", "target", "--output", model_path,
                         "--test-size", "0.2", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "test_score" in data

    def test_train_unknown_model(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        result = run_cli("train", "--model", "NoSuchModel", "--data", csv_path,
                         "--target", "target")
        assert result.returncode != 0

    def test_train_classifier(self, tmp_path):
        csv_path = _make_csv(str(tmp_path), task="classification")
        model_path = str(tmp_path / "model.pkl")
        result = run_cli("train", "--model", "LogisticRegression", "--data", csv_path,
                         "--target", "target", "--output", model_path, "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["model"] == "LogisticRegression"
```

**Step 2: Run tests to verify they fail**

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestTrain -v`
Expected: FAIL

**Step 3: Build the model registry**

```python
# ferroml-python/python/ferroml/cli/_registry.py
"""Model registry: resolve model class name strings to constructors."""
from __future__ import annotations

import sys
from typing import Any


# Lazy registry — maps model name -> (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    # linear
    "LinearRegression": ("ferroml.linear", "LinearRegression"),
    "LogisticRegression": ("ferroml.linear", "LogisticRegression"),
    "RidgeRegression": ("ferroml.linear", "RidgeRegression"),
    "LassoRegression": ("ferroml.linear", "LassoRegression"),
    "ElasticNet": ("ferroml.linear", "ElasticNet"),
    "RobustRegression": ("ferroml.linear", "RobustRegression"),
    "QuantileRegression": ("ferroml.linear", "QuantileRegression"),
    "Perceptron": ("ferroml.linear", "Perceptron"),
    "RidgeCV": ("ferroml.linear", "RidgeCV"),
    "LassoCV": ("ferroml.linear", "LassoCV"),
    "ElasticNetCV": ("ferroml.linear", "ElasticNetCV"),
    "RidgeClassifier": ("ferroml.linear", "RidgeClassifier"),
    "IsotonicRegression": ("ferroml.linear", "IsotonicRegression"),
    # trees
    "DecisionTreeClassifier": ("ferroml.trees", "DecisionTreeClassifier"),
    "DecisionTreeRegressor": ("ferroml.trees", "DecisionTreeRegressor"),
    "RandomForestClassifier": ("ferroml.trees", "RandomForestClassifier"),
    "RandomForestRegressor": ("ferroml.trees", "RandomForestRegressor"),
    "GradientBoostingClassifier": ("ferroml.trees", "GradientBoostingClassifier"),
    "GradientBoostingRegressor": ("ferroml.trees", "GradientBoostingRegressor"),
    "HistGradientBoostingClassifier": ("ferroml.trees", "HistGradientBoostingClassifier"),
    "HistGradientBoostingRegressor": ("ferroml.trees", "HistGradientBoostingRegressor"),
    # ensemble
    "ExtraTreesClassifier": ("ferroml.ensemble", "ExtraTreesClassifier"),
    "ExtraTreesRegressor": ("ferroml.ensemble", "ExtraTreesRegressor"),
    "AdaBoostClassifier": ("ferroml.ensemble", "AdaBoostClassifier"),
    "AdaBoostRegressor": ("ferroml.ensemble", "AdaBoostRegressor"),
    "SGDClassifier": ("ferroml.ensemble", "SGDClassifier"),
    "SGDRegressor": ("ferroml.ensemble", "SGDRegressor"),
    # neighbors
    "KNeighborsClassifier": ("ferroml.neighbors", "KNeighborsClassifier"),
    "KNeighborsRegressor": ("ferroml.neighbors", "KNeighborsRegressor"),
    "NearestCentroid": ("ferroml.neighbors", "NearestCentroid"),
    # clustering
    "KMeans": ("ferroml.clustering", "KMeans"),
    "MiniBatchKMeans": ("ferroml.clustering", "MiniBatchKMeans"),
    "DBSCAN": ("ferroml.clustering", "DBSCAN"),
    "AgglomerativeClustering": ("ferroml.clustering", "AgglomerativeClustering"),
    "GaussianMixture": ("ferroml.clustering", "GaussianMixture"),
    "HDBSCAN": ("ferroml.clustering", "HDBSCAN"),
    # naive_bayes
    "GaussianNB": ("ferroml.naive_bayes", "GaussianNB"),
    "MultinomialNB": ("ferroml.naive_bayes", "MultinomialNB"),
    "BernoulliNB": ("ferroml.naive_bayes", "BernoulliNB"),
    "CategoricalNB": ("ferroml.naive_bayes", "CategoricalNB"),
    # svm
    "LinearSVC": ("ferroml.svm", "LinearSVC"),
    "LinearSVR": ("ferroml.svm", "LinearSVR"),
    "SVC": ("ferroml.svm", "SVC"),
    "SVR": ("ferroml.svm", "SVR"),
    # neural
    "MLPClassifier": ("ferroml.neural", "MLPClassifier"),
    "MLPRegressor": ("ferroml.neural", "MLPRegressor"),
    # gaussian_process
    "GaussianProcessRegressor": ("ferroml.gaussian_process", "GaussianProcessRegressor"),
    "GaussianProcessClassifier": ("ferroml.gaussian_process", "GaussianProcessClassifier"),
    # anomaly
    "IsolationForest": ("ferroml.anomaly", "IsolationForest"),
    "LocalOutlierFactor": ("ferroml.anomaly", "LocalOutlierFactor"),
    # decomposition
    "PCA": ("ferroml.decomposition", "PCA"),
    "TruncatedSVD": ("ferroml.decomposition", "TruncatedSVD"),
    "IncrementalPCA": ("ferroml.decomposition", "IncrementalPCA"),
    "TSNE": ("ferroml.decomposition", "TSNE"),
    "FactorAnalysis": ("ferroml.decomposition", "FactorAnalysis"),
}


def get_model_class(name: str) -> type:
    """Resolve a model name string to its class. Raises SystemExit if not found."""
    if name not in _REGISTRY:
        print(f"Error: unknown model '{name}'.", file=sys.stderr)
        print(f"Available models: {', '.join(sorted(_REGISTRY))}", file=sys.stderr)
        raise SystemExit(1)

    module_path, class_name = _REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def list_models() -> list[str]:
    """Return sorted list of all available model names."""
    return sorted(_REGISTRY)


def construct_model(name: str, params: dict[str, Any] | None = None):
    """Construct a model instance by name, with optional constructor params."""
    cls = get_model_class(name)
    if params:
        return cls(**params)
    return cls()
```

**Step 4: Build the train command**

```python
# ferroml-python/python/ferroml/cli/train.py
"""ferroml train — fit a model on a dataset."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, save_model
from ferroml.cli._registry import construct_model


def train(
    model: str = typer.Option(..., "--model", "-m", help="Model class name (e.g. LinearRegression)."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    output_path: str = typer.Option("model.pkl", "--output", "-o", help="Path to save fitted model."),
    params: Optional[str] = typer.Option(None, "--params", "-p", help="Model params as JSON string."),
    test_size: Optional[float] = typer.Option(None, "--test-size", help="Hold-out fraction for scoring."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Fit a model on a dataset and save to disk."""
    import json as json_mod
    import numpy as np

    parsed_params = json_mod.loads(params) if params else None
    mdl = construct_model(model, parsed_params)

    X, y, feature_names = load_data(data, target)

    result_data: dict = {"model": model, "n_samples": X.shape[0], "n_features": X.shape[1]}

    if test_size:
        split_idx = int(X.shape[0] * (1 - test_size))
        indices = np.random.RandomState(42).permutation(X.shape[0])
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        if len(np.unique(y)) <= 20:
            from ferroml.metrics import accuracy_score
            score = accuracy_score(y_test, preds)
            result_data["test_metric"] = "accuracy"
        else:
            from ferroml.metrics import r2_score
            score = r2_score(y_test, preds)
            result_data["test_metric"] = "r2"
        result_data["test_score"] = round(float(score), 6)
        result_data["test_samples"] = len(test_idx)
    else:
        mdl.fit(X, y)

    save_model(mdl, output_path)
    result_data["status"] = "fitted"
    result_data["output"] = output_path

    output(result_data, json_mode)
```

**Step 5: Wire train into the main app**

Update `ferroml-python/python/ferroml/cli/__init__.py` to register the train command:

```python
"""FerroML command-line interface."""
import typer

from ferroml import __version__

app = typer.Typer(
    name="ferroml",
    help="FerroML: Statistically rigorous ML from the command line.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"ferroml {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit.", callback=version_callback, is_eager=True,
    ),
):
    """FerroML: Statistically rigorous ML from the command line."""


# Register subcommands
from ferroml.cli.train import train  # noqa: E402
app.command(name="train")(train)


def cli_main():
    app()
```

**Step 6: Run tests to verify they pass**

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestTrain -v`
Expected: PASS (6 tests)

**Step 7: Commit**

```bash
git add ferroml-python/python/ferroml/cli/ ferroml-python/tests/test_cli.py
git commit -m "feat(cli): model registry + ferroml train command"
```

---

## Task 3: `ferroml predict`

**Files:**
- Create: `ferroml-python/python/ferroml/cli/predict.py`
- Modify: `ferroml-python/python/ferroml/cli/__init__.py`
- Modify: `ferroml-python/tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `test_cli.py`:

```python
class TestPredict:
    def _train_model(self, tmp_path):
        """Helper: train a model and return (csv_path, model_path)."""
        csv_path = _make_csv(str(tmp_path))
        model_path = str(tmp_path / "model.pkl")
        run_cli("train", "--model", "LinearRegression", "--data", csv_path,
                "--target", "target", "--output", model_path)
        return csv_path, model_path

    def test_predict_to_stdout(self, tmp_path):
        csv_path, model_path = self._train_model(tmp_path)
        result = run_cli("predict", "--model", model_path, "--data", csv_path)
        assert result.returncode == 0, result.stderr
        assert len(result.stdout.strip()) > 0

    def test_predict_json(self, tmp_path):
        csv_path, model_path = self._train_model(tmp_path)
        result = run_cli("predict", "--model", model_path, "--data", csv_path, "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "predictions" in data
        assert len(data["predictions"]) == 50

    def test_predict_to_file(self, tmp_path):
        csv_path, model_path = self._train_model(tmp_path)
        out_csv = str(tmp_path / "preds.csv")
        result = run_cli("predict", "--model", model_path, "--data", csv_path, "--output", out_csv)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(out_csv)
```

**Step 2: Run tests to verify they fail**

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestPredict -v`

**Step 3: Implement predict command**

```python
# ferroml-python/python/ferroml/cli/predict.py
"""ferroml predict — generate predictions with a fitted model."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, load_model


def predict(
    model: str = typer.Option(..., "--model", "-m", help="Path to fitted model (.pkl)."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Save predictions to CSV."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Generate predictions with a fitted model."""
    mdl = load_model(model)
    X, _, feature_names = load_data(data)

    preds = mdl.predict(X)
    preds_list = [round(float(p), 6) for p in preds]

    if output_path:
        import polars as pl
        pl.DataFrame({"prediction": preds_list}).write_csv(output_path)

    result_data = {
        "n_samples": len(preds_list),
        "predictions": preds_list,
    }
    output(result_data, json_mode)
```

**Step 4: Register in `__init__.py`**

Add to `cli/__init__.py`:

```python
from ferroml.cli.predict import predict  # noqa: E402
app.command(name="predict")(predict)
```

**Step 5: Run tests**

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestPredict -v`
Expected: PASS

**Step 6: Commit**

```bash
git add ferroml-python/python/ferroml/cli/predict.py ferroml-python/python/ferroml/cli/__init__.py ferroml-python/tests/test_cli.py
git commit -m "feat(cli): ferroml predict command"
```

---

## Task 4: `ferroml evaluate`

**Files:**
- Create: `ferroml-python/python/ferroml/cli/evaluate.py`
- Modify: `ferroml-python/python/ferroml/cli/__init__.py`
- Modify: `ferroml-python/tests/test_cli.py`

**Step 1: Write the failing tests**

```python
class TestEvaluate:
    def _train_model(self, tmp_path, task="regression"):
        csv_path = _make_csv(str(tmp_path), task=task)
        model_path = str(tmp_path / "model.pkl")
        model_name = "LinearRegression" if task == "regression" else "LogisticRegression"
        run_cli("train", "--model", model_name, "--data", csv_path,
                "--target", "target", "--output", model_path)
        return csv_path, model_path

    def test_evaluate_default_metrics(self, tmp_path):
        csv_path, model_path = self._train_model(tmp_path)
        result = run_cli("evaluate", "--model", model_path, "--data", csv_path,
                         "--target", "target", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "metrics" in data

    def test_evaluate_specific_metrics(self, tmp_path):
        csv_path, model_path = self._train_model(tmp_path)
        result = run_cli("evaluate", "--model", model_path, "--data", csv_path,
                         "--target", "target", "--metrics", "rmse,r2,mae", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "rmse" in data["metrics"]
        assert "r2" in data["metrics"]
        assert "mae" in data["metrics"]

    def test_evaluate_classifier(self, tmp_path):
        csv_path, model_path = self._train_model(tmp_path, task="classification")
        result = run_cli("evaluate", "--model", model_path, "--data", csv_path,
                         "--target", "target", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "metrics" in data
```

**Step 2: Implement evaluate command**

```python
# ferroml-python/python/ferroml/cli/evaluate.py
"""ferroml evaluate — score a fitted model on labeled data."""
from __future__ import annotations

from typing import Optional

import numpy as np
import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, load_model


_REGRESSION_METRICS = {
    "rmse": "rmse",
    "mse": "mse",
    "mae": "mae",
    "r2": "r2_score",
    "explained_variance": "explained_variance",
    "max_error": "max_error",
    "mape": "mape",
    "median_absolute_error": "median_absolute_error",
}

_CLASSIFICATION_METRICS = {
    "accuracy": "accuracy_score",
    "precision": "precision_score",
    "recall": "recall_score",
    "f1": "f1_score",
    "mcc": "matthews_corrcoef",
    "balanced_accuracy": "balanced_accuracy_score",
}


def _compute_metric(name: str, y_true, y_pred) -> float:
    from ferroml import metrics
    all_metrics = {**_REGRESSION_METRICS, **_CLASSIFICATION_METRICS}
    fn_name = all_metrics.get(name)
    if fn_name is None:
        print(f"Error: unknown metric '{name}'. Available: {', '.join(sorted(all_metrics))}")
        raise SystemExit(1)
    fn = getattr(metrics, fn_name)
    return float(fn(y_true, y_pred))


def evaluate(
    model: str = typer.Option(..., "--model", "-m", help="Path to fitted model (.pkl)."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    metrics_str: Optional[str] = typer.Option(None, "--metrics", help="Comma-separated metric names."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Score a fitted model on labeled data."""
    mdl = load_model(model)
    X, y, _ = load_data(data, target)
    preds = mdl.predict(X)

    if metrics_str:
        metric_names = [m.strip() for m in metrics_str.split(",")]
    else:
        if len(np.unique(y)) <= 20:
            metric_names = ["accuracy", "f1", "precision", "recall"]
        else:
            metric_names = ["rmse", "r2", "mae"]

    scores = {}
    for name in metric_names:
        scores[name] = round(_compute_metric(name, y, preds), 6)

    result_data = {
        "n_samples": X.shape[0],
        "metrics": scores,
    }
    output(result_data, json_mode)
```

**Step 3: Register in `__init__.py` and run tests**

Add: `from ferroml.cli.evaluate import evaluate; app.command(name="evaluate")(evaluate)`

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestEvaluate -v`
Expected: PASS

**Step 4: Commit**

```bash
git add ferroml-python/python/ferroml/cli/evaluate.py ferroml-python/python/ferroml/cli/__init__.py ferroml-python/tests/test_cli.py
git commit -m "feat(cli): ferroml evaluate command"
```

---

## Task 5: `ferroml recommend` + `ferroml info`

**Files:**
- Create: `ferroml-python/python/ferroml/cli/recommend.py`
- Create: `ferroml-python/python/ferroml/cli/info.py`
- Modify: `ferroml-python/python/ferroml/cli/__init__.py`
- Modify: `ferroml-python/tests/test_cli.py`

**Step 1: Write the failing tests**

```python
class TestRecommend:
    def test_recommend_regression(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        result = run_cli("recommend", "--data", csv_path, "--target", "target", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0

    def test_recommend_classification(self, tmp_path):
        csv_path = _make_csv(str(tmp_path), task="classification")
        result = run_cli("recommend", "--data", csv_path, "--target", "target",
                         "--task", "classification", "--json")
        assert result.returncode == 0, result.stderr


class TestInfo:
    def test_info_model(self):
        result = run_cli("info", "LinearRegression", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["name"] == "LinearRegression"
        assert "task" in data

    def test_info_unknown_model(self):
        result = run_cli("info", "NoSuchModel")
        assert result.returncode != 0

    def test_info_list_all(self):
        result = run_cli("info", "--all", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert len(data) > 10
```

**Step 2: Implement recommend**

```python
# ferroml-python/python/ferroml/cli/recommend.py
"""ferroml recommend — suggest algorithms for a dataset."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data


def recommend(
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    task: Optional[str] = typer.Option(None, "--task", help="Task type: classification or regression."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Suggest the best algorithms for a dataset."""
    import numpy as np
    from ferroml import recommend as _recommend

    X, y, _ = load_data(data, target)

    if task is None:
        task = "classification" if len(np.unique(y)) <= 20 else "regression"

    recs = _recommend(X, y, task=task)

    result_data = {
        "task": task,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "recommendations": [
            {
                "algorithm": r.algorithm,
                "reason": r.reason,
                "estimated_fit_time": r.estimated_fit_time,
                "params": r.params,
                "score": r.score,
            }
            for r in recs
        ],
    }
    output(result_data, json_mode)
```

**Step 3: Implement info**

```python
# ferroml-python/python/ferroml/cli/info.py
"""ferroml info — show model card metadata."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._registry import get_model_class, list_models


def info(
    model_name: Optional[str] = typer.Argument(None, help="Model class name (e.g. LinearRegression)."),
    all_models: bool = typer.Option(False, "--all", help="List all available models with metadata."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Show model card metadata for a model class."""
    if all_models:
        cards = []
        for name in list_models():
            cls = get_model_class(name)
            if hasattr(cls, "model_card"):
                card = cls.model_card()
                cards.append({"name": name, "task": card.task, "complexity": card.complexity})
            else:
                cards.append({"name": name, "task": [], "complexity": "unknown"})
        output(cards, json_mode)
        return

    if not model_name:
        typer.echo("Error: provide a model name or --all", err=True)
        raise typer.Exit(1)

    cls = get_model_class(model_name)
    if not hasattr(cls, "model_card"):
        typer.echo(f"Error: {model_name} does not have a model card.", err=True)
        raise typer.Exit(1)

    card = cls.model_card()
    result_data = {
        "name": card.name,
        "task": card.task,
        "complexity": card.complexity,
        "interpretability": card.interpretability,
        "supports_sparse": card.supports_sparse,
        "supports_incremental": card.supports_incremental,
        "supports_sample_weight": card.supports_sample_weight,
        "strengths": card.strengths,
        "limitations": card.limitations,
        "references": card.references,
    }
    output(result_data, json_mode)
```

**Step 4: Register both in `__init__.py` and run tests**

Add:
```python
from ferroml.cli.recommend import recommend  # noqa: E402
from ferroml.cli.info import info  # noqa: E402
app.command(name="recommend")(recommend)
app.command(name="info")(info)
```

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestRecommend tests/test_cli.py::TestInfo -v`

**Step 5: Commit**

```bash
git add ferroml-python/python/ferroml/cli/recommend.py ferroml-python/python/ferroml/cli/info.py ferroml-python/python/ferroml/cli/__init__.py ferroml-python/tests/test_cli.py
git commit -m "feat(cli): ferroml recommend + info commands"
```

---

## Task 6: `ferroml compare`

**Files:**
- Create: `ferroml-python/python/ferroml/cli/compare.py`
- Modify: `ferroml-python/python/ferroml/cli/__init__.py`
- Modify: `ferroml-python/tests/test_cli.py`

**Step 1: Write the failing tests**

```python
class TestCompare:
    def test_compare_regressors(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        result = run_cli("compare", "--models", "LinearRegression,RidgeRegression,LassoRegression",
                         "--data", csv_path, "--target", "target", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "leaderboard" in data
        assert len(data["leaderboard"]) == 3

    def test_compare_classifiers(self, tmp_path):
        csv_path = _make_csv(str(tmp_path), task="classification")
        result = run_cli("compare", "--models", "LogisticRegression,GaussianNB",
                         "--data", csv_path, "--target", "target", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert len(data["leaderboard"]) == 2
```

**Step 2: Implement compare**

```python
# ferroml-python/python/ferroml/cli/compare.py
"""ferroml compare — train and compare multiple models on a dataset."""
from __future__ import annotations

import numpy as np
import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data
from ferroml.cli._registry import construct_model


def compare(
    models: str = typer.Option(..., "--models", help="Comma-separated model names."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    test_size: float = typer.Option(0.2, "--test-size", help="Hold-out fraction for scoring."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Train multiple models and compare their performance."""
    import time
    from ferroml import metrics

    X, y, _ = load_data(data, target)
    is_classification = len(np.unique(y)) <= 20

    split_idx = int(X.shape[0] * (1 - test_size))
    indices = np.random.RandomState(42).permutation(X.shape[0])
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model_names = [m.strip() for m in models.split(",")]
    leaderboard = []

    for name in model_names:
        try:
            mdl = construct_model(name)
            t0 = time.perf_counter()
            mdl.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0
            preds = mdl.predict(X_test)

            if is_classification:
                score = float(metrics.accuracy_score(y_test, preds))
                metric_name = "accuracy"
            else:
                score = float(metrics.r2_score(y_test, preds))
                metric_name = "r2"

            leaderboard.append({
                "model": name,
                "score": round(score, 6),
                "metric": metric_name,
                "fit_time_seconds": round(fit_time, 4),
            })
        except Exception as e:
            leaderboard.append({"model": name, "error": str(e)})

    leaderboard.sort(key=lambda x: x.get("score", float("-inf")), reverse=True)

    result_data = {
        "n_models": len(model_names),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "leaderboard": leaderboard,
    }
    output(result_data, json_mode)
```

**Step 3: Register, test, commit**

Add to `__init__.py`: `from ferroml.cli.compare import compare; app.command(name="compare")(compare)`

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestCompare -v`

```bash
git add ferroml-python/python/ferroml/cli/compare.py ferroml-python/python/ferroml/cli/__init__.py ferroml-python/tests/test_cli.py
git commit -m "feat(cli): ferroml compare command"
```

---

## Task 7: `ferroml diagnose`

**Files:**
- Create: `ferroml-python/python/ferroml/cli/diagnose.py`
- Modify: `ferroml-python/python/ferroml/cli/__init__.py`
- Modify: `ferroml-python/tests/test_cli.py`

**Step 1: Write the failing tests**

```python
class TestDiagnose:
    def test_diagnose_linear_regression(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        model_path = str(tmp_path / "model.pkl")
        run_cli("train", "--model", "LinearRegression", "--data", csv_path,
                "--target", "target", "--output", model_path)
        result = run_cli("diagnose", "--model", model_path, "--data", csv_path,
                         "--target", "target", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "diagnostics" in data
```

**Step 2: Implement diagnose**

```python
# ferroml-python/python/ferroml/cli/diagnose.py
"""ferroml diagnose — run statistical diagnostics on a fitted model."""
from __future__ import annotations

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, load_model


def diagnose(
    model: str = typer.Option(..., "--model", "-m", help="Path to fitted model (.pkl)."),
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Run statistical diagnostics on a fitted model."""
    import numpy as np
    from ferroml import metrics, stats

    mdl = load_model(model)
    X, y, feature_names = load_data(data, target)
    preds = mdl.predict(X)
    residuals = y - preds

    diag: dict = {}

    # Always compute: residual stats
    diag["residual_mean"] = round(float(np.mean(residuals)), 6)
    diag["residual_std"] = round(float(np.std(residuals)), 6)
    diag["durbin_watson"] = round(float(stats.durbin_watson(residuals)), 6)
    normality = stats.normality_test(residuals)
    diag["residual_normality"] = normality

    # Model-specific diagnostics
    if hasattr(mdl, "summary"):
        diag["summary"] = mdl.summary()

    if hasattr(mdl, "r_squared"):
        diag["r_squared"] = round(float(mdl.r_squared()), 6)

    if hasattr(mdl, "adjusted_r_squared"):
        diag["adjusted_r_squared"] = round(float(mdl.adjusted_r_squared()), 6)

    if hasattr(mdl, "f_statistic"):
        diag["f_statistic"] = round(float(mdl.f_statistic()), 6)

    if hasattr(mdl, "coefficients_with_ci"):
        diag["coefficients"] = mdl.coefficients_with_ci()

    if hasattr(mdl, "aic"):
        diag["aic"] = round(float(mdl.aic()), 6)

    if hasattr(mdl, "bic"):
        diag["bic"] = round(float(mdl.bic()), 6)

    if hasattr(mdl, "log_likelihood"):
        diag["log_likelihood"] = round(float(mdl.log_likelihood()), 6)

    # Basic metrics
    is_classification = len(np.unique(y)) <= 20
    if is_classification:
        diag["accuracy"] = round(float(metrics.accuracy_score(y, preds)), 6)
    else:
        diag["rmse"] = round(float(metrics.rmse(y, preds)), 6)
        diag["r2"] = round(float(metrics.r2_score(y, preds)), 6)

    result_data = {"diagnostics": diag}
    output(result_data, json_mode)
```

**Step 3: Register, test, commit**

Add to `__init__.py`: `from ferroml.cli.diagnose import diagnose; app.command(name="diagnose")(diagnose)`

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestDiagnose -v`

```bash
git add ferroml-python/python/ferroml/cli/diagnose.py ferroml-python/python/ferroml/cli/__init__.py ferroml-python/tests/test_cli.py
git commit -m "feat(cli): ferroml diagnose command"
```

---

## Task 8: `ferroml automl` + `ferroml export`

**Files:**
- Create: `ferroml-python/python/ferroml/cli/automl_cmd.py`
- Create: `ferroml-python/python/ferroml/cli/export.py`
- Modify: `ferroml-python/python/ferroml/cli/__init__.py`
- Modify: `ferroml-python/tests/test_cli.py`

**Step 1: Write the failing tests**

```python
class TestAutoML:
    @pytest.mark.slow
    def test_automl_regression(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        result = run_cli("automl", "--data", csv_path, "--target", "target",
                         "--task", "regression", "--time-budget", "10", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "leaderboard" in data


class TestExport:
    def test_export_onnx(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        model_path = str(tmp_path / "model.pkl")
        run_cli("train", "--model", "LinearRegression", "--data", csv_path,
                "--target", "target", "--output", model_path)
        onnx_path = str(tmp_path / "model.onnx")
        result = run_cli("export", "--model", model_path, "--output", onnx_path,
                         "--n-features", "3")
        assert result.returncode == 0, result.stderr
        assert os.path.exists(onnx_path)
```

**Step 2: Implement automl command**

```python
# ferroml-python/python/ferroml/cli/automl_cmd.py
"""ferroml automl — run automated machine learning search."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_data, save_model


def automl(
    data: str = typer.Option(..., "--data", "-d", help="Path to CSV or Parquet file."),
    target: str = typer.Option(..., "--target", "-t", help="Target column name."),
    task: str = typer.Option("classification", "--task", help="Task: classification or regression."),
    time_budget: int = typer.Option(60, "--time-budget", help="Time budget in seconds."),
    metric: Optional[str] = typer.Option(None, "--metric", help="Evaluation metric."),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Save best model to disk."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Run AutoML to find the best model for a dataset."""
    from ferroml.automl import AutoML, AutoMLConfig

    X, y, _ = load_data(data, target)

    if metric is None:
        metric = "roc_auc" if task == "classification" else "rmse"

    config = AutoMLConfig(
        task=task,
        metric=metric,
        time_budget_seconds=time_budget,
    )
    aml = AutoML(config)
    result = aml.fit(X, y)

    leaderboard = []
    for entry in result.leaderboard:
        leaderboard.append({
            "rank": entry.rank,
            "algorithm": entry.algorithm,
            "cv_score": round(float(entry.cv_score), 6),
            "cv_std": round(float(entry.cv_std), 6),
            "ci_lower": round(float(entry.ci_lower), 6),
            "ci_upper": round(float(entry.ci_upper), 6),
            "training_time": round(float(entry.training_time_seconds), 4),
        })

    if output_path and result.best_model():
        from ferroml.cli._registry import construct_model
        best = result.best_model()
        mdl = construct_model(best.algorithm, best.params)
        mdl.fit(X, y)
        save_model(mdl, output_path)

    result_data = {
        "task": task,
        "metric": metric,
        "n_trials": result.n_successful_trials,
        "total_time_seconds": round(float(result.total_time_seconds), 2),
        "leaderboard": leaderboard,
        "summary": result.summary(),
    }
    output(result_data, json_mode)
```

**Step 3: Implement export command**

```python
# ferroml-python/python/ferroml/cli/export.py
"""ferroml export — export a model to ONNX format."""
from __future__ import annotations

from typing import Optional

import typer

from ferroml.cli._format import output
from ferroml.cli._io import load_model


def export(
    model: str = typer.Option(..., "--model", "-m", help="Path to fitted model (.pkl)."),
    output_path: str = typer.Option(..., "--output", "-o", help="Path for ONNX output."),
    n_features: int = typer.Option(..., "--n-features", help="Number of input features."),
    model_name: Optional[str] = typer.Option(None, "--name", help="ONNX model name."),
    json_mode: bool = typer.Option(False, "--json", help="Output JSON for agent consumption."),
):
    """Export a fitted model to ONNX format."""
    mdl = load_model(model)

    if not hasattr(mdl, "export_onnx"):
        typer.echo("Error: this model does not support ONNX export.", err=True)
        raise typer.Exit(1)

    mdl.export_onnx(output_path, model_name=model_name)

    result_data = {
        "status": "exported",
        "format": "onnx",
        "output": output_path,
        "n_features": n_features,
    }
    output(result_data, json_mode)
```

**Step 4: Register both, test, commit**

Add to `__init__.py`:
```python
from ferroml.cli.automl_cmd import automl as automl_cmd  # noqa: E402
from ferroml.cli.export import export  # noqa: E402
app.command(name="automl")(automl_cmd)
app.command(name="export")(export)
```

Run: `cd ferroml-python && python -m pytest tests/test_cli.py::TestExport -v` (skip automl — it's slow)

```bash
git add ferroml-python/python/ferroml/cli/automl_cmd.py ferroml-python/python/ferroml/cli/export.py ferroml-python/python/ferroml/cli/__init__.py ferroml-python/tests/test_cli.py
git commit -m "feat(cli): ferroml automl + export commands"
```

---

## Task 9: AGENTS.md Rewrite

**Files:**
- Modify: `AGENTS.md`

**Step 1: Write AGENTS.md**

Rewrite the existing AGENTS.md as an AI agent discoverability document. No test needed — this is documentation.

Content should cover:
- What FerroML is (1-2 sentences)
- Installation (`pip install ferroml` or `pip install ferroml[cli]`)
- CLI quick reference — every subcommand with a one-liner example
- Python API quick reference — key imports by task
- Model selection guidance table (task -> recommended models)
- Build/test instructions
- Error handling patterns (FerroError variants + .hint())

Keep it under 300 lines — agents have token budgets.

**Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "docs: rewrite AGENTS.md for AI agent discoverability"
```

---

## Task 10: llms.txt

**Files:**
- Create: `llms.txt`

**Step 1: Write llms.txt**

Standard format from llmstxt.org:
```
# FerroML

> Statistically rigorous machine learning in Rust with Python bindings.

FerroML is a machine learning library ...

## Docs

- [README](https://github.com/robertlupo1997/ferroml#readme)
- [AGENTS.md](https://github.com/robertlupo1997/ferroml/blob/master/AGENTS.md): Agent operations reference
- [API Stubs](https://github.com/robertlupo1997/ferroml/tree/master/ferroml-python/python/ferroml): Type stubs for all 55+ models
```

**Step 2: Commit**

```bash
git add llms.txt
git commit -m "docs: add llms.txt for LLM discoverability"
```

---

## Task 11: Final Validation

**Step 1: Run full Python test suite**

Run: `cd ferroml-python && python -m pytest tests/ -v --tb=short`
Expected: All existing tests + new CLI tests pass.

**Step 2: Run Rust tests**

Run: `cargo test`
Expected: All 3,266+ tests pass.

**Step 3: Manual CLI smoke test**

```bash
# Create test data
python -c "
import numpy as np
rng = np.random.RandomState(42)
X = rng.randn(100, 4)
y = X[:, 0] * 2 + X[:, 1] + rng.randn(100) * 0.1
import polars as pl
df = pl.DataFrame({'a': X[:,0], 'b': X[:,1], 'c': X[:,2], 'd': X[:,3], 'target': y})
df.write_csv('/tmp/test_data.csv')
"

ferroml --version
ferroml recommend --data /tmp/test_data.csv --target target --json
ferroml train --model LinearRegression --data /tmp/test_data.csv --target target --output /tmp/model.pkl --test-size 0.2 --json
ferroml predict --model /tmp/model.pkl --data /tmp/test_data.csv --json | python -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d[\"predictions\"])} predictions')"
ferroml evaluate --model /tmp/model.pkl --data /tmp/test_data.csv --target target --json
ferroml diagnose --model /tmp/model.pkl --data /tmp/test_data.csv --target target --json
ferroml compare --models LinearRegression,RidgeRegression,LassoRegression --data /tmp/test_data.csv --target target --json
ferroml info LinearRegression --json
ferroml export --model /tmp/model.pkl --output /tmp/model.onnx --n-features 4
```

**Step 4: Commit any fixes, then verify clean state**

No version bump needed (already 1.0.0). Just ensure everything is clean.

---

## Execution Order

Tasks 1-8 are sequential (each builds on previous).
Tasks 9 and 10 (AGENTS.md, llms.txt) can be done in parallel with any task.
Task 11 is the final gate.

**Estimated: 8-10 commits across 11 tasks.**
