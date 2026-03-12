#!/usr/bin/env python3
"""Benchmark FerroML vs scikit-learn on standard ML tasks.

Compares fit and predict (or transform) times across a suite of common models
and preprocessing steps, checks that predictions agree within tolerance, and
optionally writes results to JSON and/or a Markdown report.

Usage:
    python scripts/benchmark_vs_sklearn.py
    python scripts/benchmark_vs_sklearn.py --output results.json
    python scripts/benchmark_vs_sklearn.py --markdown docs/benchmark-vs-sklearn.md
    python scripts/benchmark_vs_sklearn.py --output results.json --markdown docs/benchmark-vs-sklearn.md
"""

import argparse
import json
import time
import platform
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, List

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Stores timing and correctness data for a single benchmark run."""
    name: str
    task: str
    n_samples: int
    n_features: int
    ferroml_fit_time: float
    sklearn_fit_time: float
    ferroml_predict_time: float
    sklearn_predict_time: float
    fit_speedup: float        # sklearn / ferroml  (>1 => ferroml faster)
    predict_speedup: float
    predictions_match: bool   # within tolerance
    tolerance: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    """Collect basic system metadata."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor() or "unknown",
        "numpy_version": np.__version__,
        "timestamp": datetime.now().isoformat(),
    }


def time_fn(fn: Callable[[], Any], n_repeats: int = 3):
    """Time *fn*, returning (median_seconds, last_result)."""
    times: list[float] = []
    result = None
    for _ in range(n_repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return float(np.median(times)), result


def speedup(sklearn_time: float, ferroml_time: float) -> float:
    """Compute speedup ratio (>1 means FerroML is faster)."""
    return sklearn_time / max(ferroml_time, 1e-10)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_regression_data(n_samples: int, n_features: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    coef = rng.randn(n_features)
    y = X @ coef + rng.randn(n_samples) * 0.1
    return X, y


def make_classification_data(n_samples: int, n_features: int, seed: int = 42):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, 5),
        n_redundant=min(n_features - min(n_features, 5), 2),
        random_state=seed,
    )
    return X, y.astype(float)


def make_clustering_data(n_samples: int, n_features: int, n_clusters: int = 5,
                         seed: int = 42):
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=seed,
    )
    return X


# ---------------------------------------------------------------------------
# Individual benchmark functions
# ---------------------------------------------------------------------------

def benchmark_linear_regression(n_samples: int) -> BenchmarkResult:
    n_features = 10
    X, y = make_regression_data(n_samples, n_features)
    X_test = X[:200]

    from ferroml.linear import LinearRegression as FerroLR
    from sklearn.linear_model import LinearRegression as SkLR

    # FerroML
    def ferro_fit():
        m = FerroLR()
        m.fit(X, y)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_predict_time, ferro_preds = time_fn(lambda: ferro_model.predict(X_test))

    # sklearn
    def sk_fit():
        m = SkLR()
        m.fit(X, y)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_predict_time, sk_preds = time_fn(lambda: sk_model.predict(X_test))

    tol = 1e-4
    match = bool(np.allclose(ferro_preds, sk_preds, atol=tol, rtol=tol))

    return BenchmarkResult(
        name="LinearRegression", task="regression",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_predict_time, sklearn_predict_time=sk_predict_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_predict_time, ferro_predict_time),
        predictions_match=match, tolerance=tol,
    )


def benchmark_ridge(n_samples: int) -> BenchmarkResult:
    n_features = 10
    X, y = make_regression_data(n_samples, n_features)
    X_test = X[:200]

    from ferroml.linear import RidgeRegression as FerroRidge
    from sklearn.linear_model import Ridge as SkRidge

    def ferro_fit():
        m = FerroRidge(alpha=1.0)
        m.fit(X, y)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_predict_time, ferro_preds = time_fn(lambda: ferro_model.predict(X_test))

    def sk_fit():
        m = SkRidge(alpha=1.0)
        m.fit(X, y)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_predict_time, sk_preds = time_fn(lambda: sk_model.predict(X_test))

    tol = 1e-4
    match = bool(np.allclose(ferro_preds, sk_preds, atol=tol, rtol=tol))

    return BenchmarkResult(
        name="Ridge", task="regression",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_predict_time, sklearn_predict_time=sk_predict_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_predict_time, ferro_predict_time),
        predictions_match=match, tolerance=tol,
    )


def benchmark_logistic_regression(n_samples: int) -> BenchmarkResult:
    n_features = 10
    X, y = make_classification_data(n_samples, n_features)
    X_test = X[:200]

    from ferroml.linear import LogisticRegression as FerroLR
    from sklearn.linear_model import LogisticRegression as SkLR

    def ferro_fit():
        m = FerroLR(max_iter=200)
        m.fit(X, y)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_predict_time, ferro_preds = time_fn(lambda: ferro_model.predict(X_test))

    def sk_fit():
        m = SkLR(max_iter=200)
        m.fit(X, y)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_predict_time, sk_preds = time_fn(lambda: sk_model.predict(X_test))

    # Logistic is iterative; predictions may differ slightly
    tol = 0.1
    agree_frac = np.mean(ferro_preds == sk_preds)
    match = bool(agree_frac >= 0.90)

    return BenchmarkResult(
        name="LogisticRegression", task="classification",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_predict_time, sklearn_predict_time=sk_predict_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_predict_time, ferro_predict_time),
        predictions_match=match, tolerance=tol,
        notes=f"agreement={agree_frac:.2%}",
    )


def benchmark_decision_tree(n_samples: int) -> BenchmarkResult:
    n_features = 10
    X, y = make_classification_data(n_samples, n_features)
    X_test = X[:200]

    from ferroml.trees import DecisionTreeClassifier as FerroTree
    from sklearn.tree import DecisionTreeClassifier as SkTree

    def ferro_fit():
        m = FerroTree(max_depth=10, random_state=42)
        m.fit(X, y)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_predict_time, ferro_preds = time_fn(lambda: ferro_model.predict(X_test))

    def sk_fit():
        m = SkTree(max_depth=10, random_state=42)
        m.fit(X, y)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_predict_time, sk_preds = time_fn(lambda: sk_model.predict(X_test))

    # Different splitting heuristics may produce different trees
    tol = 0.1
    agree_frac = np.mean(ferro_preds == sk_preds)
    match = bool(agree_frac >= 0.85)

    return BenchmarkResult(
        name="DecisionTree", task="classification",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_predict_time, sklearn_predict_time=sk_predict_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_predict_time, ferro_predict_time),
        predictions_match=match, tolerance=tol,
        notes=f"agreement={agree_frac:.2%}",
    )


def benchmark_random_forest(n_samples: int) -> BenchmarkResult:
    n_features = 10
    X, y = make_classification_data(n_samples, n_features)
    X_test = X[:200]

    from ferroml.trees import RandomForestClassifier as FerroRF
    from sklearn.ensemble import RandomForestClassifier as SkRF

    def ferro_fit():
        m = FerroRF(n_estimators=50, max_depth=5, random_state=42)
        m.fit(X, y)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_predict_time, ferro_preds = time_fn(lambda: ferro_model.predict(X_test))

    def sk_fit():
        m = SkRF(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
        m.fit(X, y)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_predict_time, sk_preds = time_fn(lambda: sk_model.predict(X_test))

    # Stochastic — allow generous tolerance
    tol = 0.1
    agree_frac = np.mean(ferro_preds == sk_preds)
    match = bool(agree_frac >= 0.80)

    return BenchmarkResult(
        name="RandomForest", task="classification",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_predict_time, sklearn_predict_time=sk_predict_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_predict_time, ferro_predict_time),
        predictions_match=match, tolerance=tol,
        notes=f"agreement={agree_frac:.2%}",
    )


def benchmark_gradient_boosting(n_samples: int) -> BenchmarkResult:
    n_features = 10
    X, y = make_regression_data(n_samples, n_features)
    X_test = X[:200]

    from ferroml.trees import GradientBoostingRegressor as FerroGB
    from sklearn.ensemble import GradientBoostingRegressor as SkGB

    def ferro_fit():
        m = FerroGB(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
        m.fit(X, y)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_predict_time, ferro_preds = time_fn(lambda: ferro_model.predict(X_test))

    def sk_fit():
        m = SkGB(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
        m.fit(X, y)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_predict_time, sk_preds = time_fn(lambda: sk_model.predict(X_test))

    # Different tree building — compare via correlation
    tol = 0.1
    corr = float(np.corrcoef(ferro_preds, sk_preds)[0, 1])
    match = corr > 0.90

    return BenchmarkResult(
        name="GradientBoosting", task="regression",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_predict_time, sklearn_predict_time=sk_predict_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_predict_time, ferro_predict_time),
        predictions_match=match, tolerance=tol,
        notes=f"corr={corr:.4f}",
    )


def benchmark_knn(n_samples: int) -> BenchmarkResult:
    n_features = 10
    X, y = make_classification_data(n_samples, n_features)
    X_test = X[:200]

    from ferroml.neighbors import KNeighborsClassifier as FerroKNN
    from sklearn.neighbors import KNeighborsClassifier as SkKNN

    def ferro_fit():
        m = FerroKNN(n_neighbors=5)
        m.fit(X, y)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_predict_time, ferro_preds = time_fn(lambda: ferro_model.predict(X_test))

    def sk_fit():
        m = SkKNN(n_neighbors=5, algorithm="brute")
        m.fit(X, y)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_predict_time, sk_preds = time_fn(lambda: sk_model.predict(X_test))

    # KNN should match closely (same algorithm on same data)
    tol = 0.0
    agree_frac = np.mean(ferro_preds == sk_preds)
    match = bool(agree_frac >= 0.95)

    return BenchmarkResult(
        name="KNN", task="classification",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_predict_time, sklearn_predict_time=sk_predict_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_predict_time, ferro_predict_time),
        predictions_match=match, tolerance=tol,
        notes=f"agreement={agree_frac:.2%}",
    )


def benchmark_standard_scaler(n_samples: int) -> BenchmarkResult:
    n_features = 20
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features) * 10 + 5

    from ferroml.preprocessing import StandardScaler as FerroScaler
    from sklearn.preprocessing import StandardScaler as SkScaler

    def ferro_fit():
        m = FerroScaler()
        result = m.fit_transform(X)
        return m, result
    ferro_fit_time, (ferro_model, ferro_result) = time_fn(ferro_fit)

    def sk_fit():
        m = SkScaler()
        result = m.fit_transform(X)
        return m, result
    sk_fit_time, (sk_model, sk_result) = time_fn(sk_fit)

    # For preprocessing, fit_transform is the main operation; use it for both timings
    tol = 1e-10
    match = bool(np.allclose(ferro_result, sk_result, atol=tol))

    return BenchmarkResult(
        name="StandardScaler", task="preprocessing",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=0.0, sklearn_predict_time=0.0,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=0.0,
        predictions_match=match, tolerance=tol,
        notes="fit_transform comparison",
    )


def benchmark_pca(n_samples: int) -> BenchmarkResult:
    n_features = 20
    n_components = 5
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)

    from ferroml.decomposition import PCA as FerroPCA
    from sklearn.decomposition import PCA as SkPCA

    def ferro_fit():
        m = FerroPCA(n_components=n_components)
        m.fit(X)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_transform_time, ferro_result = time_fn(lambda: ferro_model.transform(X))

    def sk_fit():
        m = SkPCA(n_components=n_components)
        m.fit(X)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_transform_time, sk_result = time_fn(lambda: sk_model.transform(X))

    # PCA components may have sign flips — compare absolute values
    tol = 1e-4
    match = bool(np.allclose(np.abs(ferro_result), np.abs(sk_result), atol=tol, rtol=tol))

    return BenchmarkResult(
        name="PCA", task="decomposition",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_transform_time,
        sklearn_predict_time=sk_transform_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_transform_time, ferro_transform_time),
        predictions_match=match, tolerance=tol,
        notes="transform comparison; sign-flip invariant",
    )


def benchmark_kmeans(n_samples: int) -> BenchmarkResult:
    n_features = 10
    n_clusters = 5
    X = make_clustering_data(n_samples, n_features, n_clusters=n_clusters)

    from ferroml.clustering import KMeans as FerroKM
    from sklearn.cluster import KMeans as SkKM

    def ferro_fit():
        m = FerroKM(n_clusters=n_clusters, random_state=42)
        m.fit(X)
        return m
    ferro_fit_time, ferro_model = time_fn(ferro_fit)
    ferro_predict_time, ferro_labels = time_fn(lambda: ferro_model.predict(X[:200]))

    def sk_fit():
        m = SkKM(n_clusters=n_clusters, random_state=42, n_init=1)
        m.fit(X)
        return m
    sk_fit_time, sk_model = time_fn(sk_fit)
    sk_predict_time, sk_labels = time_fn(lambda: sk_model.predict(X[:200]))

    # Compare inertia (sensitive to initialization, so allow 10% slack)
    ferro_inertia = ferro_model.inertia_
    sk_inertia = sk_model.inertia_
    tol = 0.1
    rel_diff = abs(ferro_inertia - sk_inertia) / max(abs(sk_inertia), 1e-10)
    match = rel_diff < tol

    return BenchmarkResult(
        name="KMeans", task="clustering",
        n_samples=n_samples, n_features=n_features,
        ferroml_fit_time=ferro_fit_time, sklearn_fit_time=sk_fit_time,
        ferroml_predict_time=ferro_predict_time,
        sklearn_predict_time=sk_predict_time,
        fit_speedup=speedup(sk_fit_time, ferro_fit_time),
        predict_speedup=speedup(sk_predict_time, ferro_predict_time),
        predictions_match=match, tolerance=tol,
        notes=f"inertia_diff={rel_diff:.4f}",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARKS = [
    # (name, function, sizes)
    ("LinearRegression", benchmark_linear_regression, [1000, 5000, 10000]),
    ("Ridge", benchmark_ridge, [1000, 5000, 10000]),
    ("LogisticRegression", benchmark_logistic_regression, [1000, 5000, 10000]),
    ("DecisionTree", benchmark_decision_tree, [1000, 5000, 10000]),
    ("RandomForest", benchmark_random_forest, [1000, 5000]),
    ("GradientBoosting", benchmark_gradient_boosting, [1000, 5000]),
    ("KNN", benchmark_knn, [1000, 5000, 10000]),
    ("StandardScaler", benchmark_standard_scaler, [1000, 10000, 50000]),
    ("PCA", benchmark_pca, [1000, 5000]),
    ("KMeans", benchmark_kmeans, [1000, 5000]),
]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_table(results: List[BenchmarkResult]) -> None:
    """Print results as a formatted ASCII table to stdout."""
    header = f"{'Model':<22} {'N':>7} {'Fit(FML)':>10} {'Fit(SK)':>10} {'Fit x':>8} " \
             f"{'Pred(FML)':>10} {'Pred(SK)':>10} {'Pred x':>8} {'Match':>6}"
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for r in results:
        fit_mark = f"{r.fit_speedup:.2f}x"
        pred_mark = f"{r.predict_speedup:.2f}x" if r.predict_speedup > 0 else "n/a"
        match_mark = "Y" if r.predictions_match else "N"
        print(
            f"{r.name:<22} {r.n_samples:>7} "
            f"{r.ferroml_fit_time:>10.5f} {r.sklearn_fit_time:>10.5f} {fit_mark:>8} "
            f"{r.ferroml_predict_time:>10.5f} {r.sklearn_predict_time:>10.5f} {pred_mark:>8} "
            f"{match_mark:>6}"
        )
    print(sep)
    print("Times in seconds. Speedup >1 means FerroML is faster.")
    print()


def write_json(results: List[BenchmarkResult], sys_info: dict, path: str) -> None:
    """Write results and system info to a JSON file."""
    payload = {
        "system_info": sys_info,
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON results written to {path}")


def write_markdown(results: List[BenchmarkResult], sys_info: dict, path: str) -> None:
    """Write a Markdown benchmark report."""
    lines: list[str] = []
    lines.append("# FerroML vs scikit-learn Benchmark Results\n")

    lines.append("## System Information\n")
    lines.append(f"- **Platform**: {sys_info['platform']}")
    lines.append(f"- **Python**: {sys_info['python_version'].split(chr(10))[0]}")
    lines.append(f"- **CPU**: {sys_info['cpu']}")
    lines.append(f"- **NumPy**: {sys_info['numpy_version']}")
    lines.append("")

    lines.append("## Results\n")
    lines.append("| Model | N | Fit Speedup | Predict Speedup | Match | Notes |")
    lines.append("|-------|---|-------------|-----------------|-------|-------|")
    for r in results:
        fit_s = f"{r.fit_speedup:.2f}x"
        pred_s = f"{r.predict_speedup:.2f}x" if r.predict_speedup > 0 else "n/a"
        match_s = "Y" if r.predictions_match else "N"
        notes = r.notes or ""
        lines.append(f"| {r.name} | {r.n_samples} | {fit_s} | {pred_s} | {match_s} | {notes} |")
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- **Speedup > 1.0** means FerroML is faster than scikit-learn")
    lines.append("- **Match** indicates predictions agree within the specified tolerance")
    lines.append("- Times are median of 3 runs using `time.perf_counter()`")
    lines.append("- RandomForest and GradientBoosting use n_estimators=50, max_depth=5")
    lines.append("- sklearn RandomForest uses n_jobs=1 for fair single-threaded comparison")
    lines.append("- sklearn KMeans uses n_init=1 for fair comparison")
    lines.append(f"- Generated on {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown report written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmarks() -> List[BenchmarkResult]:
    """Execute all registered benchmarks, returning results."""
    # Verify libraries are available
    try:
        import ferroml  # noqa: F401
    except ImportError:
        print("ERROR: ferroml is not installed. Build with:")
        print("  source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml")
        sys.exit(1)

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("ERROR: scikit-learn is not installed. Install with:")
        print("  pip install scikit-learn")
        sys.exit(1)

    results: List[BenchmarkResult] = []
    total = sum(len(sizes) for _, _, sizes in BENCHMARKS)
    done = 0

    for name, bench_fn, sizes in BENCHMARKS:
        for n in sizes:
            done += 1
            tag = f"[{done}/{total}]"
            print(f"{tag} Running {name} n={n}...", end=" ", flush=True)
            try:
                result = bench_fn(n)
                results.append(result)
                fit_s = f"fit={result.fit_speedup:.2f}x"
                pred_s = f"pred={result.predict_speedup:.2f}x" if result.predict_speedup > 0 else ""
                match_s = "match" if result.predictions_match else "MISMATCH"
                print(f"{fit_s} {pred_s} {match_s}")
            except Exception as exc:
                print(f"FAILED: {exc}")
                traceback.print_exc()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FerroML vs scikit-learn on standard ML tasks."
    )
    parser.add_argument(
        "--output", "-o",
        type=str, default=None,
        help="Path to write JSON results (e.g. results.json)",
    )
    parser.add_argument(
        "--markdown", "-m",
        type=str, default=None,
        help="Path to write Markdown report (e.g. docs/benchmark-vs-sklearn.md)",
    )
    args = parser.parse_args()

    sys_info = get_system_info()
    print("FerroML vs scikit-learn Benchmark Suite")
    print(f"Platform: {sys_info['platform']}")
    print(f"Python:   {sys_info['python_version'].split(chr(10))[0]}")
    print()

    results = run_benchmarks()

    if results:
        print_table(results)

    if args.output and results:
        write_json(results, sys_info, args.output)

    if args.markdown and results:
        write_markdown(results, sys_info, args.markdown)

    if not results:
        print("No benchmark results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
