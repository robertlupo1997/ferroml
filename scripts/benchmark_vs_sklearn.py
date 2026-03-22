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


def time_fn(fn: Callable[[], Any], n_repeats: int = 3, n_warmup: int = 0):
    """Time *fn*, returning (median_seconds, last_result).

    Args:
        fn: callable to time
        n_repeats: number of timed runs (median is reported)
        n_warmup: number of untimed warmup runs before timing
    """
    # Warmup runs (not timed)
    for _ in range(n_warmup):
        fn()

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
# PERF-target benchmark functions (Phase 04 cross-library comparison)
# ---------------------------------------------------------------------------

@dataclass
class PerfBenchmarkResult:
    """Stores timing data for a single PERF-target benchmark."""
    name: str
    perf_id: str
    dataset: str
    ferroml_ms: float
    sklearn_ms: float
    ratio: float       # ferroml / sklearn (>1 means ferroml slower)
    target: float      # max acceptable ratio
    passed: bool
    notes: str = ""


def perf_benchmark_pca() -> PerfBenchmarkResult:
    """PERF-01: PCA on 10000x100, n_components=10."""
    rng = np.random.RandomState(42)
    X = rng.randn(10000, 100)

    from ferroml.decomposition import PCA as FerroPCA
    from sklearn.decomposition import PCA as SkPCA

    ferro_time, _ = time_fn(lambda: FerroPCA(n_components=10).fit(X), n_repeats=5, n_warmup=1)
    sk_time, _ = time_fn(lambda: SkPCA(n_components=10).fit(X), n_repeats=5, n_warmup=1)

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="PCA", perf_id="PERF-01", dataset="10000x100, k=10",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=2.0, passed=ratio <= 2.0,
    )


def perf_benchmark_truncated_svd() -> PerfBenchmarkResult:
    """PERF-02: TruncatedSVD on 10000x100, n_components=10."""
    rng = np.random.RandomState(42)
    X = rng.randn(10000, 100)

    from ferroml.decomposition import TruncatedSVD as FerroSVD
    from sklearn.decomposition import TruncatedSVD as SkSVD

    ferro_time, _ = time_fn(lambda: FerroSVD(n_components=10).fit(X), n_repeats=5, n_warmup=1)
    sk_time, _ = time_fn(lambda: SkSVD(n_components=10).fit(X), n_repeats=5, n_warmup=1)

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="TruncatedSVD", perf_id="PERF-02", dataset="10000x100, k=10",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=2.0, passed=ratio <= 2.0,
    )


def perf_benchmark_lda() -> PerfBenchmarkResult:
    """PERF-03: LDA on 5000x50, 3 classes, n_components=2."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=5000, n_features=50, n_informative=10,
                                n_classes=3, n_clusters_per_class=1, random_state=42)
    y = y.astype(float)

    from ferroml.decomposition import LDA as FerroLDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA

    def ferro_fit():
        m = FerroLDA(n_components=2)
        m.fit(X, y)
        return m

    def sk_fit():
        m = SkLDA(n_components=2)
        m.fit(X, y)
        return m

    ferro_time, _ = time_fn(ferro_fit, n_repeats=5, n_warmup=1)
    sk_time, _ = time_fn(sk_fit, n_repeats=5, n_warmup=1)

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="LDA", perf_id="PERF-03", dataset="5000x50, 3 classes, k=2",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=2.0, passed=ratio <= 2.0,
    )


def perf_benchmark_factor_analysis() -> PerfBenchmarkResult:
    """PERF-04: FactorAnalysis on 5000x50, n_components=5."""
    rng = np.random.RandomState(42)
    X = rng.randn(5000, 50)

    from ferroml.decomposition import FactorAnalysis as FerroFA
    from sklearn.decomposition import FactorAnalysis as SkFA

    ferro_time, _ = time_fn(lambda: FerroFA(n_factors=5).fit(X), n_repeats=5, n_warmup=1)
    sk_time, _ = time_fn(lambda: SkFA(n_components=5).fit(X), n_repeats=5, n_warmup=1)

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="FactorAnalysis", perf_id="PERF-04", dataset="5000x50, k=5",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=3.0, passed=ratio <= 3.0,
    )


def perf_benchmark_linear_svc() -> PerfBenchmarkResult:
    """PERF-05: LinearSVC on 5000x50, binary classification."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=5000, n_features=50, n_informative=10,
                                random_state=42)
    y = y.astype(float)

    from ferroml.svm import LinearSVC as FerroLSVC
    from sklearn.svm import LinearSVC as SkLSVC

    ferro_time, _ = time_fn(lambda: FerroLSVC().fit(X, y), n_repeats=5, n_warmup=1)
    sk_time, _ = time_fn(lambda: SkLSVC(dual=True, max_iter=1000).fit(X, y), n_repeats=5, n_warmup=1)

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="LinearSVC", perf_id="PERF-05", dataset="5000x50, binary",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=2.0, passed=ratio <= 2.0,
    )


def perf_benchmark_ols() -> PerfBenchmarkResult:
    """PERF-07: OLS (LinearRegression) on 10000x100."""
    rng = np.random.RandomState(42)
    X = rng.randn(10000, 100)
    coef = rng.randn(100)
    y = X @ coef + rng.randn(10000) * 0.1

    from ferroml.linear import LinearRegression as FerroLR
    from sklearn.linear_model import LinearRegression as SkLR

    ferro_time, _ = time_fn(lambda: FerroLR().fit(X, y), n_repeats=5, n_warmup=1)
    sk_time, _ = time_fn(lambda: SkLR().fit(X, y), n_repeats=5, n_warmup=1)

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="OLS", perf_id="PERF-07", dataset="10000x100",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=2.0, passed=ratio <= 2.0,
    )


def perf_benchmark_ridge() -> PerfBenchmarkResult:
    """PERF-08: Ridge on 10000x100, alpha=1.0."""
    rng = np.random.RandomState(42)
    X = rng.randn(10000, 100)
    coef = rng.randn(100)
    y = X @ coef + rng.randn(10000) * 0.1

    from ferroml.linear import RidgeRegression as FerroRidge
    from sklearn.linear_model import Ridge as SkRidge

    ferro_time, _ = time_fn(lambda: FerroRidge(alpha=1.0).fit(X, y), n_repeats=5, n_warmup=1)
    sk_time, _ = time_fn(lambda: SkRidge(alpha=1.0).fit(X, y), n_repeats=5, n_warmup=1)

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="Ridge", perf_id="PERF-08", dataset="10000x100, alpha=1.0",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=2.0, passed=ratio <= 2.0,
    )


def perf_benchmark_kmeans() -> PerfBenchmarkResult:
    """PERF-11: KMeans on 5000x50, n_clusters=10."""
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=5000, n_features=50, centers=10, random_state=42)

    from ferroml.clustering import KMeans as FerroKM
    from sklearn.cluster import KMeans as SkKM

    ferro_time, _ = time_fn(
        lambda: FerroKM(n_clusters=10, random_state=42).fit(X),
        n_repeats=5, n_warmup=1,
    )
    sk_time, _ = time_fn(
        lambda: SkKM(n_clusters=10, random_state=42, n_init=1).fit(X),
        n_repeats=5, n_warmup=1,
    )

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="KMeans", perf_id="PERF-11", dataset="5000x50, k=10",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=2.0, passed=ratio <= 2.0,
    )


def perf_benchmark_hist_gbt() -> PerfBenchmarkResult:
    """PERF-09: HistGradientBoosting on 10000x20, 50 iterations."""
    rng = np.random.RandomState(42)
    X = rng.randn(10000, 20)
    coef = rng.randn(20)
    y = X @ coef + rng.randn(10000) * 0.5

    from ferroml.trees import HistGradientBoostingRegressor as FerroHGBT
    from sklearn.ensemble import HistGradientBoostingRegressor as SkHGBT

    ferro_time, _ = time_fn(
        lambda: FerroHGBT(max_iter=50, max_depth=5, learning_rate=0.1).fit(X, y),
        n_repeats=5, n_warmup=1,
    )
    sk_time, _ = time_fn(
        lambda: SkHGBT(max_iter=50, max_depth=5, learning_rate=0.1).fit(X, y),
        n_repeats=5, n_warmup=1,
    )

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="HistGBT", perf_id="PERF-09", dataset="10000x20, 50 iters",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=3.0, passed=ratio <= 3.0,
    )


def perf_benchmark_svc_rbf() -> PerfBenchmarkResult:
    """PERF-10: SVC (RBF kernel) on 2000x20, binary."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                                random_state=42)
    y = y.astype(float)

    from ferroml.svm import SVC as FerroSVC
    from sklearn.svm import SVC as SkSVC

    ferro_time, _ = time_fn(
        lambda: FerroSVC(kernel="rbf").fit(X, y),
        n_repeats=5, n_warmup=1,
    )
    sk_time, _ = time_fn(
        lambda: SkSVC(kernel="rbf").fit(X, y),
        n_repeats=5, n_warmup=1,
    )

    ratio = ferro_time / max(sk_time, 1e-10)
    return PerfBenchmarkResult(
        name="SVC (RBF)", perf_id="PERF-10", dataset="2000x20, binary",
        ferroml_ms=ferro_time * 1000, sklearn_ms=sk_time * 1000,
        ratio=ratio, target=3.0, passed=ratio <= 3.0,
    )


PERF_BENCHMARKS = [
    ("PCA", perf_benchmark_pca),
    ("TruncatedSVD", perf_benchmark_truncated_svd),
    ("LDA", perf_benchmark_lda),
    ("FactorAnalysis", perf_benchmark_factor_analysis),
    ("LinearSVC", perf_benchmark_linear_svc),
    ("OLS", perf_benchmark_ols),
    ("Ridge", perf_benchmark_ridge),
    ("KMeans", perf_benchmark_kmeans),
    ("HistGBT", perf_benchmark_hist_gbt),
    ("SVC (RBF)", perf_benchmark_svc_rbf),
]


def run_perf_benchmarks() -> List[PerfBenchmarkResult]:
    """Execute all PERF-target benchmarks, returning results."""
    results: List[PerfBenchmarkResult] = []
    total = len(PERF_BENCHMARKS)
    for i, (name, bench_fn) in enumerate(PERF_BENCHMARKS, 1):
        print(f"[{i}/{total}] PERF: {name}...", end=" ", flush=True)
        try:
            result = bench_fn()
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"{result.ferroml_ms:.1f}ms vs {result.sklearn_ms:.1f}ms "
                  f"= {result.ratio:.2f}x (target <={result.target:.1f}x) [{status}]")
        except Exception as exc:
            print(f"ERROR: {exc}")
            traceback.print_exc()
    return results


def print_perf_table(results: List[PerfBenchmarkResult]) -> None:
    """Print PERF benchmark results as a formatted table."""
    print()
    print("=" * 90)
    print("PERF-TARGET CROSS-LIBRARY BENCHMARK RESULTS")
    print("=" * 90)
    header = f"{'PERF':>7} {'Algorithm':<16} {'Dataset':<22} {'FerroML':>10} {'sklearn':>10} {'Ratio':>8} {'Target':>8} {'Status':>6}"
    print(header)
    print("-" * 90)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.perf_id:>7} {r.name:<16} {r.dataset:<22} "
              f"{r.ferroml_ms:>8.1f}ms {r.sklearn_ms:>8.1f}ms "
              f"{r.ratio:>7.2f}x {r.target:>6.1f}x {status:>6}")
    print("-" * 90)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} benchmarks")

    # Check for regressions (> 3x)
    regressions = [r for r in results if r.ratio > 3.0]
    if regressions:
        print(f"\nREGRESSIONS (> 3.0x):")
        for r in regressions:
            print(f"  {r.perf_id} {r.name}: {r.ratio:.2f}x (target {r.target:.1f}x)")
    print()


def write_perf_json(results: List[PerfBenchmarkResult], sys_info: dict, path: str) -> None:
    """Write PERF benchmark results to JSON."""
    payload = {
        "system_info": sys_info,
        "benchmark_type": "perf_targets",
        "methodology": {
            "warmup_runs": 1,
            "timed_runs": 5,
            "aggregation": "median",
            "data_seed": 42,
        },
        "results": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"PERF JSON results written to {path}")


# ---------------------------------------------------------------------------
# Registry (original general-purpose benchmarks)
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
    parser.add_argument(
        "--perf-only",
        action="store_true",
        help="Only run PERF-target benchmarks (skip general suite)",
    )
    parser.add_argument(
        "--perf-json",
        type=str, default=None,
        help="Path to write PERF benchmark JSON (e.g. docs/benchmark_results.json)",
    )
    args = parser.parse_args()

    sys_info = get_system_info()
    print("FerroML vs scikit-learn Benchmark Suite")
    print(f"Platform: {sys_info['platform']}")
    print(f"Python:   {sys_info['python_version'].split(chr(10))[0]}")
    print()

    # Always run PERF-target benchmarks
    print("--- PERF-Target Benchmarks (Phase 04) ---")
    perf_results = run_perf_benchmarks()
    if perf_results:
        print_perf_table(perf_results)

    perf_json_path = args.perf_json or "docs/benchmark_results.json"
    if perf_results:
        write_perf_json(perf_results, sys_info, perf_json_path)

    if not args.perf_only:
        # Run general benchmark suite
        print("--- General Benchmark Suite ---")
        results = run_benchmarks()
        if results:
            print_table(results)
        if args.output and results:
            write_json(results, sys_info, args.output)
        if args.markdown and results:
            write_markdown(results, sys_info, args.markdown)

    if not perf_results:
        print("No PERF benchmark results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
