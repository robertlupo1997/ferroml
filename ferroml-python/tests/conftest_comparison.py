"""
Shared infrastructure for FerroML vs sklearn comparison tests.

Provides:
- Dataset loaders (sklearn real datasets + synthetic generators)
- Comparison helpers (predictions, probabilities, transforms)
- Timing helpers
- Result collection and markdown report generation
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def get_iris() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_iris
    data = load_iris()
    return data.data, data.target.astype(np.float64)


def get_wine() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_wine
    data = load_wine()
    return data.data, data.target.astype(np.float64)


def get_breast_cancer() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    return data.data, data.target.astype(np.float64)


def get_diabetes() -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    return data.data, data.target


def get_california_housing(n_max: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Load california housing, subsampled for speed."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y = data.data, data.target
    if len(y) > n_max:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y), n_max, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def get_classification_data(n: int = 1000, p: int = 20, n_classes: int = 2,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=min(p, 10),
        n_redundant=min(p - min(p, 10), 5), n_classes=n_classes,
        random_state=random_state,
    )
    return X, y.astype(np.float64)


def get_regression_data(n: int = 1000, p: int = 20,
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=n, n_features=p, n_informative=min(p, 10),
                           noise=10.0, random_state=random_state)
    return X, y


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_predictions(ferro_preds: np.ndarray, sklearn_preds: np.ndarray,
                        tol: float = 1e-6) -> Tuple[bool, float]:
    """Compare prediction arrays. Returns (passed, max_diff)."""
    diff = np.max(np.abs(ferro_preds - sklearn_preds))
    return diff <= tol, float(diff)


def compare_r2(ferro_r2: float, sklearn_r2: float, tol: float = 0.05) -> Tuple[bool, float]:
    """Compare R² scores. Returns (passed, diff)."""
    diff = abs(ferro_r2 - sklearn_r2)
    return diff <= tol, diff


def compare_accuracy(ferro_preds: np.ndarray, sklearn_preds: np.ndarray,
                     y_true: np.ndarray, tol: float = 0.05) -> Tuple[bool, float, float]:
    """Compare classification accuracy. Returns (passed, ferro_acc, sklearn_acc)."""
    ferro_acc = np.mean(ferro_preds == y_true)
    sklearn_acc = np.mean(sklearn_preds == y_true)
    return abs(ferro_acc - sklearn_acc) <= tol, float(ferro_acc), float(sklearn_acc)


def compare_probabilities(ferro_proba: np.ndarray, sklearn_proba: np.ndarray,
                          tol: float = 1e-4) -> Tuple[bool, float]:
    """Compare probability matrices. Returns (passed, max_diff)."""
    diff = np.max(np.abs(ferro_proba - sklearn_proba))
    return diff <= tol, float(diff)


def compare_transforms(ferro_out: np.ndarray, sklearn_out: np.ndarray,
                       tol: float = 1e-10) -> Tuple[bool, float]:
    """Compare transform outputs. Returns (passed, max_diff)."""
    diff = np.max(np.abs(ferro_out - sklearn_out))
    return diff <= tol, float(diff)


def compare_coef(ferro_coef: np.ndarray, sklearn_coef: np.ndarray,
                 tol: float = 1e-6) -> Tuple[bool, float]:
    """Compare coefficient arrays. Returns (passed, max_diff)."""
    diff = np.max(np.abs(ferro_coef - sklearn_coef))
    return diff <= tol, float(diff)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def timed_fit(model, X, y, n_runs: int = 3):
    """Fit model and return (fitted_model, median_time_ms)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.fit(X, y)
        times.append((time.perf_counter() - t0) * 1000)
    return model, float(np.median(times))


def timed_predict(model, X, n_runs: int = 5):
    """Predict and return (predictions, median_time_ms)."""
    times = []
    preds = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        preds = model.predict(X)
        times.append((time.perf_counter() - t0) * 1000)
    return preds, float(np.median(times))


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    model: str
    dataset: str
    metric: str
    ferro_value: float
    sklearn_value: float
    tolerance: float
    passed: bool

    def to_row(self) -> str:
        status = "PASS" if self.passed else "**FAIL**"
        return (f"| {self.model} | {self.dataset} | {self.metric} | "
                f"{self.ferro_value:.6f} | {self.sklearn_value:.6f} | "
                f"{self.tolerance} | {status} |")


@dataclass
class ComparisonReport:
    results: List[ComparisonResult] = field(default_factory=list)

    def add(self, result: ComparisonResult):
        self.results.append(result)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def generate_markdown(self) -> str:
        lines = [
            "# FerroML vs sklearn Validation Report",
            "",
            f"**Total**: {len(self.results)} comparisons | "
            f"**Passed**: {self.n_passed} | **Failed**: {self.n_failed}",
            "",
            "| Model | Dataset | Metric | FerroML | sklearn | Tolerance | Status |",
            "|-------|---------|--------|---------|---------|-----------|--------|",
        ]
        for r in self.results:
            lines.append(r.to_row())
        return "\n".join(lines)
