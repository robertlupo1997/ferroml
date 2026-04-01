"""Tests for the ferroml CLI."""
import json
import os
import subprocess
import sys

import numpy as np
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
        assert "ferroml" in result.stdout and any(v in result.stdout for v in ["1.0.", "1.1."])

    def test_no_args_shows_help(self):
        result = run_cli()
        # Typer's no_args_is_help returns exit code 0 or 2 depending on version
        assert "ferroml" in result.stdout.lower()


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
        result = run_cli("predict", "--model", model_path, "--data", csv_path, "--target", "target")
        assert result.returncode == 0, result.stderr
        # Rich output goes to stderr, so check either stdout or stderr has content
        assert len(result.stdout.strip()) > 0 or len(result.stderr.strip()) > 0

    def test_predict_json(self, tmp_path):
        csv_path, model_path = self._train_model(tmp_path)
        result = run_cli("predict", "--model", model_path, "--data", csv_path, "--target", "target", "--json")
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "predictions" in data
        assert len(data["predictions"]) == 50

    def test_predict_to_file(self, tmp_path):
        csv_path, model_path = self._train_model(tmp_path)
        out_csv = str(tmp_path / "preds.csv")
        result = run_cli("predict", "--model", model_path, "--data", csv_path, "--target", "target", "--output", out_csv)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(out_csv)


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


class TestAutoML:
    @pytest.mark.slow
    def test_automl_regression(self, tmp_path):
        csv_path = _make_csv(str(tmp_path))
        result = subprocess.run(
            [sys.executable, "-m", "ferroml.cli", "automl", "--data", csv_path,
             "--target", "target", "--task", "regression", "--time-budget", "5", "--json"],
            capture_output=True, text=True, timeout=60,
        )
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
