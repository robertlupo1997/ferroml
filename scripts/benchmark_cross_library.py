#!/usr/bin/env python3
"""Cross-library benchmark: FerroML vs scikit-learn vs XGBoost vs LightGBM.

Benchmarks all overlapping algorithms across libraries on synthetic data,
measuring fit time, predict/transform time, and score (R2 or accuracy).

Usage:
    python scripts/benchmark_cross_library.py
    python scripts/benchmark_cross_library.py --sizes 1000 5000
    python scripts/benchmark_cross_library.py --features 50
    python scripts/benchmark_cross_library.py --output-json docs/benchmark_cross_library_results.json
    python scripts/benchmark_cross_library.py --output-md docs/cross-library-benchmark.md
"""

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Stores timing and score data for a single benchmark run."""
    algorithm: str
    library: str
    task: str          # regression, classification, clustering, preprocessing
    n_samples: int
    n_features: int
    fit_time_ms: float
    predict_time_ms: float  # or transform_time_ms for preprocessing/clustering
    score: Optional[float]  # R2 for regression, accuracy for classification, inertia for clustering, None for preprocessing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    """Collect basic system metadata."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor() or "unknown",
        "numpy_version": np.__version__,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    try:
        import xgboost
        info["xgboost_version"] = xgboost.__version__
    except ImportError:
        pass
    try:
        import lightgbm
        info["lightgbm_version"] = lightgbm.__version__
    except ImportError:
        pass
    return info


def time_fn(fn: Callable[[], Any], n_repeats: int = 3) -> Tuple[float, Any]:
    """Time *fn* n_repeats times, return (median_ms, last_result)."""
    times: list[float] = []
    result = None
    for _ in range(n_repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000.0)  # convert to ms
    return float(np.median(times)), result


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_regression_data(n_samples: int, n_features: int, seed: int = 42):
    """Generate synthetic regression data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    coef = rng.randn(n_features)
    y = X @ coef + rng.randn(n_samples) * 0.1
    return X, y


def make_classification_data(n_samples: int, n_features: int, seed: int = 42):
    """Generate synthetic binary classification data."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, 10),
        n_redundant=min(max(n_features - 10, 0), 5),
        random_state=seed,
    )
    return X, y.astype(np.float64)


def make_clustering_data(n_samples: int, n_features: int, n_clusters: int = 5,
                         seed: int = 42):
    """Generate synthetic clustering data."""
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=seed,
    )
    return X


# ---------------------------------------------------------------------------
# Library availability checks
# ---------------------------------------------------------------------------

_LIB_AVAILABLE: Dict[str, bool] = {}


def _check_lib(name: str) -> bool:
    if name not in _LIB_AVAILABLE:
        try:
            __import__(name)
            _LIB_AVAILABLE[name] = True
        except ImportError:
            _LIB_AVAILABLE[name] = False
    return _LIB_AVAILABLE[name]


def has_ferroml() -> bool:
    return _check_lib("ferroml")


def has_sklearn() -> bool:
    return _check_lib("sklearn")


def has_xgboost() -> bool:
    return _check_lib("xgboost")


def has_lightgbm() -> bool:
    return _check_lib("lightgbm")


# ---------------------------------------------------------------------------
# Benchmark runner for a single algorithm across libraries
# ---------------------------------------------------------------------------

def run_single(
    algorithm: str,
    task: str,
    n_samples: int,
    n_features: int,
    constructors: Dict[str, Callable],
    X_train: np.ndarray,
    y_train: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: Optional[np.ndarray],
    is_transform: bool = False,
    is_clustering: bool = False,
) -> List[BenchmarkResult]:
    """Benchmark one algorithm across all provided library constructors."""
    results = []
    for lib_name, make_model in constructors.items():
        try:
            # Fit
            def do_fit():
                m = make_model()
                if y_train is not None:
                    m.fit(X_train, y_train)
                else:
                    m.fit(X_train)
                return m

            fit_ms, model = time_fn(do_fit)

            # Predict / transform
            if is_transform:
                pred_ms, output = time_fn(lambda: model.transform(X_test))
                score = None
            elif is_clustering:
                pred_ms, preds = time_fn(lambda: model.predict(X_test))
                # Use inertia if available, else None
                score = getattr(model, "inertia_", None)
                if score is None:
                    score = getattr(model, "inertia", None)
            elif task == "regression":
                pred_ms, preds = time_fn(lambda: model.predict(X_test))
                # R2 score
                ss_res = np.sum((y_test - preds) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                score = 1.0 - ss_res / max(ss_tot, 1e-10)
            elif task == "classification":
                pred_ms, preds = time_fn(lambda: model.predict(X_test))
                if hasattr(preds, 'astype'):
                    preds = preds.astype(np.float64)
                score = float(np.mean(preds == y_test))
            else:
                pred_ms, _ = time_fn(lambda: model.predict(X_test))
                score = None

            results.append(BenchmarkResult(
                algorithm=algorithm,
                library=lib_name,
                task=task,
                n_samples=n_samples,
                n_features=n_features,
                fit_time_ms=round(fit_ms, 3),
                predict_time_ms=round(pred_ms, 3),
                score=round(score, 6) if score is not None else None,
            ))
        except Exception as e:
            print(f"  WARNING: {lib_name}/{algorithm} failed: {e}")
            traceback.print_exc()
    return results


# ---------------------------------------------------------------------------
# Algorithm benchmark definitions
# ---------------------------------------------------------------------------

def bench_linear_regression(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.linear import LinearRegression as FerroLR
        constructors["ferroml"] = lambda: FerroLR()
    if has_sklearn():
        from sklearn.linear_model import LinearRegression as SkLR
        constructors["sklearn"] = lambda: SkLR()

    return run_single("LinearRegression", "regression", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_ridge(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.linear import RidgeRegression as FerroRidge
        constructors["ferroml"] = lambda: FerroRidge(alpha=1.0)
    if has_sklearn():
        from sklearn.linear_model import Ridge as SkRidge
        constructors["sklearn"] = lambda: SkRidge(alpha=1.0)

    return run_single("Ridge", "regression", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_lasso(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.linear import LassoRegression as FerroLasso
        constructors["ferroml"] = lambda: FerroLasso(alpha=1.0)
    if has_sklearn():
        from sklearn.linear_model import Lasso as SkLasso
        constructors["sklearn"] = lambda: SkLasso(alpha=1.0)

    return run_single("Lasso", "regression", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_logistic_regression(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_classification_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.linear import LogisticRegression as FerroLR
        constructors["ferroml"] = lambda: FerroLR(compute_diagnostics=False)
    if has_sklearn():
        from sklearn.linear_model import LogisticRegression as SkLR
        constructors["sklearn"] = lambda: SkLR(max_iter=1000)

    return run_single("LogisticRegression", "classification", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_decision_tree_regressor(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.trees import DecisionTreeRegressor as FerroDTO
        constructors["ferroml"] = lambda: FerroDTO(max_depth=10)
    if has_sklearn():
        from sklearn.tree import DecisionTreeRegressor as SkDTO
        constructors["sklearn"] = lambda: SkDTO(max_depth=10)

    return run_single("DecisionTreeRegressor", "regression", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_decision_tree_classifier(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_classification_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.trees import DecisionTreeClassifier as FerroDTO
        constructors["ferroml"] = lambda: FerroDTO(max_depth=10)
    if has_sklearn():
        from sklearn.tree import DecisionTreeClassifier as SkDTO
        constructors["sklearn"] = lambda: SkDTO(max_depth=10)

    return run_single("DecisionTreeClassifier", "classification", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_random_forest_regressor(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.trees import RandomForestRegressor as FerroRF
        constructors["ferroml"] = lambda: FerroRF(n_estimators=100, max_depth=10)
    if has_sklearn():
        from sklearn.ensemble import RandomForestRegressor as SkRF
        constructors["sklearn"] = lambda: SkRF(n_estimators=100, max_depth=10, random_state=42)
    if has_xgboost():
        import xgboost as xgb
        constructors["xgboost"] = lambda: xgb.XGBRFRegressor(
            n_estimators=100, max_depth=10, random_state=42, verbosity=0)
    if has_lightgbm():
        import lightgbm as lgb
        constructors["lightgbm"] = lambda: lgb.LGBMRegressor(
            boosting_type="rf", n_estimators=100, max_depth=10,
            subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)

    return run_single("RandomForestRegressor", "regression", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_random_forest_classifier(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_classification_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.trees import RandomForestClassifier as FerroRF
        constructors["ferroml"] = lambda: FerroRF(n_estimators=100, max_depth=10)
    if has_sklearn():
        from sklearn.ensemble import RandomForestClassifier as SkRF
        constructors["sklearn"] = lambda: SkRF(n_estimators=100, max_depth=10, random_state=42)
    if has_xgboost():
        import xgboost as xgb
        constructors["xgboost"] = lambda: xgb.XGBRFClassifier(
            n_estimators=100, max_depth=10, random_state=42, verbosity=0,
            use_label_encoder=False, eval_metric="logloss")
    if has_lightgbm():
        import lightgbm as lgb
        constructors["lightgbm"] = lambda: lgb.LGBMClassifier(
            boosting_type="rf", n_estimators=100, max_depth=10,
            subsample=0.8, subsample_freq=1, random_state=42, verbose=-1)

    return run_single("RandomForestClassifier", "classification", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_gradient_boosting_regressor(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.trees import GradientBoostingRegressor as FerroGB
        constructors["ferroml"] = lambda: FerroGB(n_estimators=100, max_depth=5, learning_rate=0.1)
    if has_sklearn():
        from sklearn.ensemble import GradientBoostingRegressor as SkGB
        constructors["sklearn"] = lambda: SkGB(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    if has_xgboost():
        import xgboost as xgb
        constructors["xgboost"] = lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
    if has_lightgbm():
        import lightgbm as lgb
        constructors["lightgbm"] = lambda: lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)

    return run_single("GradientBoostingRegressor", "regression", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_gradient_boosting_classifier(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_classification_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.trees import GradientBoostingClassifier as FerroGB
        constructors["ferroml"] = lambda: FerroGB(n_estimators=100, max_depth=5, learning_rate=0.1)
    if has_sklearn():
        from sklearn.ensemble import GradientBoostingClassifier as SkGB
        constructors["sklearn"] = lambda: SkGB(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    if has_xgboost():
        import xgboost as xgb
        constructors["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
            verbosity=0, use_label_encoder=False, eval_metric="logloss")
    if has_lightgbm():
        import lightgbm as lgb
        constructors["lightgbm"] = lambda: lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)

    return run_single("GradientBoostingClassifier", "classification", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_hist_gradient_boosting_regressor(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.trees import HistGradientBoostingRegressor as FerroHGB
        constructors["ferroml"] = lambda: FerroHGB(max_iter=100, max_depth=5, learning_rate=0.1)
    if has_sklearn():
        from sklearn.ensemble import HistGradientBoostingRegressor as SkHGB
        constructors["sklearn"] = lambda: SkHGB(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
    if has_xgboost():
        import xgboost as xgb
        constructors["xgboost"] = lambda: xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, tree_method="hist",
            random_state=42, verbosity=0)
    if has_lightgbm():
        import lightgbm as lgb
        constructors["lightgbm"] = lambda: lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)

    return run_single("HistGradientBoostingRegressor", "regression", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_hist_gradient_boosting_classifier(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_classification_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.trees import HistGradientBoostingClassifier as FerroHGB
        constructors["ferroml"] = lambda: FerroHGB(max_iter=100, max_depth=5, learning_rate=0.1)
    if has_sklearn():
        from sklearn.ensemble import HistGradientBoostingClassifier as SkHGB
        constructors["sklearn"] = lambda: SkHGB(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
    if has_xgboost():
        import xgboost as xgb
        constructors["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, tree_method="hist",
            random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
    if has_lightgbm():
        import lightgbm as lgb
        constructors["lightgbm"] = lambda: lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)

    return run_single("HistGradientBoostingClassifier", "classification", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_knn_classifier(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_classification_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.neighbors import KNeighborsClassifier as FerroKNN
        constructors["ferroml"] = lambda: FerroKNN(n_neighbors=5)
    if has_sklearn():
        from sklearn.neighbors import KNeighborsClassifier as SkKNN
        constructors["sklearn"] = lambda: SkKNN(n_neighbors=5)

    return run_single("KNeighborsClassifier", "classification", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_svc(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    # SVC is O(n^2)-O(n^3), cap training size for large datasets
    cap = min(n_samples, 5000)
    X, y = make_classification_data(cap, n_features)
    split = int(0.8 * cap)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.svm import SVC as FerroSVC
        constructors["ferroml"] = lambda: FerroSVC()
    if has_sklearn():
        from sklearn.svm import SVC as SkSVC
        constructors["sklearn"] = lambda: SkSVC()

    return run_single("SVC", "classification", cap, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_gaussian_nb(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, y = make_classification_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.naive_bayes import GaussianNB as FerroGNB
        constructors["ferroml"] = lambda: FerroGNB()
    if has_sklearn():
        from sklearn.naive_bayes import GaussianNB as SkGNB
        constructors["sklearn"] = lambda: SkGNB()

    return run_single("GaussianNB", "classification", n_samples, n_features,
                      constructors, X_train, y_train, X_test, y_test)


def bench_standard_scaler(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, _ = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.preprocessing import StandardScaler as FerroSS
        constructors["ferroml"] = lambda: FerroSS()
    if has_sklearn():
        from sklearn.preprocessing import StandardScaler as SkSS
        constructors["sklearn"] = lambda: SkSS()

    return run_single("StandardScaler", "preprocessing", n_samples, n_features,
                      constructors, X_train, None, X_test, None, is_transform=True)


def bench_pca(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X, _ = make_regression_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    n_components = min(10, n_features)

    constructors = {}
    if has_ferroml():
        from ferroml.decomposition import PCA as FerroP
        constructors["ferroml"] = lambda: FerroP(n_components=n_components)
    if has_sklearn():
        from sklearn.decomposition import PCA as SkP
        constructors["sklearn"] = lambda: SkP(n_components=n_components)

    return run_single("PCA", "preprocessing", n_samples, n_features,
                      constructors, X_train, None, X_test, None, is_transform=True)


def bench_kmeans(n_samples: int, n_features: int) -> List[BenchmarkResult]:
    X = make_clustering_data(n_samples, n_features)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]

    constructors = {}
    if has_ferroml():
        from ferroml.clustering import KMeans as FerroKM
        constructors["ferroml"] = lambda: FerroKM(n_clusters=5)
    if has_sklearn():
        from sklearn.cluster import KMeans as SkKM
        constructors["sklearn"] = lambda: SkKM(n_clusters=5, random_state=42, n_init=10)

    return run_single("KMeans", "clustering", n_samples, n_features,
                      constructors, X_train, None, X_test, None, is_clustering=True)


# ---------------------------------------------------------------------------
# Registry of all benchmarks
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = [
    # Linear models (ferroml, sklearn)
    ("LinearRegression", bench_linear_regression),
    ("Ridge", bench_ridge),
    ("Lasso", bench_lasso),
    ("LogisticRegression", bench_logistic_regression),
    # Trees / Ensembles (ferroml, sklearn, xgboost, lightgbm)
    ("DecisionTreeRegressor", bench_decision_tree_regressor),
    ("DecisionTreeClassifier", bench_decision_tree_classifier),
    ("RandomForestRegressor", bench_random_forest_regressor),
    ("RandomForestClassifier", bench_random_forest_classifier),
    ("GradientBoostingRegressor", bench_gradient_boosting_regressor),
    ("GradientBoostingClassifier", bench_gradient_boosting_classifier),
    ("HistGradientBoostingRegressor", bench_hist_gradient_boosting_regressor),
    ("HistGradientBoostingClassifier", bench_hist_gradient_boosting_classifier),
    # Other supervised (ferroml, sklearn)
    ("KNeighborsClassifier", bench_knn_classifier),
    ("SVC", bench_svc),
    ("GaussianNB", bench_gaussian_nb),
    # Preprocessing (ferroml, sklearn)
    ("StandardScaler", bench_standard_scaler),
    ("PCA", bench_pca),
    # Clustering (ferroml, sklearn)
    ("KMeans", bench_kmeans),
]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_ascii_table(results: List[BenchmarkResult]) -> None:
    """Print results as a formatted ASCII table to stdout."""
    if not results:
        print("No results to display.")
        return

    # Column headers and widths
    headers = ["Algorithm", "Library", "Task", "N", "Feat", "Fit(ms)", "Pred(ms)", "Score"]
    rows = []
    for r in results:
        score_str = f"{r.score:.4f}" if r.score is not None else "N/A"
        rows.append([
            r.algorithm, r.library, r.task,
            str(r.n_samples), str(r.n_features),
            f"{r.fit_time_ms:.1f}", f"{r.predict_time_ms:.1f}",
            score_str,
        ])

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def fmt_row(vals):
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(vals))

    separator = "-+-".join("-" * w for w in col_widths)

    print()
    print("=" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    print("  CROSS-LIBRARY BENCHMARK RESULTS")
    print("=" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    print()
    print(fmt_row(headers))
    print(separator)

    prev_algo = None
    for row in rows:
        if prev_algo is not None and row[0] != prev_algo:
            print(separator)
        prev_algo = row[0]
        print(fmt_row(row))

    print()
    print(f"Total benchmarks: {len(results)}")


def write_json(results: List[BenchmarkResult], path: str) -> None:
    """Write results as JSON to the given file path."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "system_info": get_system_info(),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON results written to {path}")


def write_markdown(results: List[BenchmarkResult], path: str) -> None:
    """Write results as a Markdown table to the given file path."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    info = get_system_info()
    lines = [
        "# Cross-Library Benchmark Results",
        "",
        f"Generated: {info['timestamp']}",
        f"Platform: {info['platform']}",
        f"Python: {info['python_version'].split()[0]}",
        f"NumPy: {info['numpy_version']}",
    ]
    for lib in ["sklearn", "xgboost", "lightgbm"]:
        key = f"{lib}_version"
        if key in info:
            lines.append(f"{lib}: {info[key]}")
    lines.append("")

    # Group results by algorithm for comparison tables
    from collections import OrderedDict
    algo_groups: Dict[str, List[BenchmarkResult]] = OrderedDict()
    for r in results:
        key = f"{r.algorithm} (n={r.n_samples})"
        algo_groups.setdefault(key, []).append(r)

    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Algorithm | Library | Task | N | Features | Fit (ms) | Predict (ms) | Score |")
    lines.append("|-----------|---------|------|---|----------|----------|--------------|-------|")

    for r in results:
        score_str = f"{r.score:.4f}" if r.score is not None else "N/A"
        lines.append(
            f"| {r.algorithm} | {r.library} | {r.task} | {r.n_samples} | "
            f"{r.n_features} | {r.fit_time_ms:.1f} | {r.predict_time_ms:.1f} | {score_str} |"
        )

    # Speedup comparison section
    lines.append("")
    lines.append("## FerroML vs Others: Fit-Time Speedup")
    lines.append("")
    lines.append("Values > 1.0 mean FerroML is faster.")
    lines.append("")
    lines.append("| Algorithm | N | vs sklearn | vs xgboost | vs lightgbm |")
    lines.append("|-----------|---|-----------|-----------|------------|")

    # Group by (algorithm, n_samples) for speedup calc
    from collections import defaultdict
    grouped: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(dict)
    for r in results:
        grouped[(r.algorithm, r.n_samples)][r.library] = r.fit_time_ms

    for (algo, n), lib_times in grouped.items():
        ferro_t = lib_times.get("ferroml")
        if ferro_t is None or ferro_t < 0.001:
            continue
        sk = lib_times.get("sklearn")
        xgb = lib_times.get("xgboost")
        lgb = lib_times.get("lightgbm")
        sk_str = f"{sk / ferro_t:.2f}x" if sk else "N/A"
        xgb_str = f"{xgb / ferro_t:.2f}x" if xgb else "N/A"
        lgb_str = f"{lgb / ferro_t:.2f}x" if lgb else "N/A"
        lines.append(f"| {algo} | {n} | {sk_str} | {xgb_str} | {lgb_str} |")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown report written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-library benchmark: FerroML vs scikit-learn vs XGBoost vs LightGBM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/benchmark_cross_library.py
  python scripts/benchmark_cross_library.py --sizes 1000 5000
  python scripts/benchmark_cross_library.py --features 50 --output-json results.json
  python scripts/benchmark_cross_library.py --output-md docs/cross-library-benchmark.md
        """,
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1000, 5000, 10000],
        help="Dataset sizes to benchmark (default: 1000 5000 10000)")
    parser.add_argument(
        "--features", type=int, default=20,
        help="Number of features (default: 20)")
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to write JSON results (default: docs/benchmark_cross_library_results.json)")
    parser.add_argument(
        "--output-md", type=str, default=None,
        help="Path to write Markdown report (default: docs/cross-library-benchmark.md)")
    parser.add_argument(
        "--algorithms", type=str, nargs="+", default=None,
        help="Only run specific algorithms (by name substring match)")

    args = parser.parse_args()

    # Print library availability
    print("Library availability:")
    print(f"  ferroml:  {'YES' if has_ferroml() else 'NO'}")
    print(f"  sklearn:  {'YES' if has_sklearn() else 'NO'}")
    print(f"  xgboost:  {'YES' if has_xgboost() else 'NO'}")
    print(f"  lightgbm: {'YES' if has_lightgbm() else 'NO'}")
    print()

    if not has_ferroml():
        print("ERROR: ferroml is not importable. Build with:")
        print("  source .venv/bin/activate && maturin develop --release -m ferroml-python/Cargo.toml")
        sys.exit(1)

    # Filter benchmarks if --algorithms specified
    benchmarks = ALL_BENCHMARKS
    if args.algorithms:
        benchmarks = [
            (name, fn) for name, fn in benchmarks
            if any(a.lower() in name.lower() for a in args.algorithms)
        ]
        if not benchmarks:
            print(f"ERROR: No algorithms matched filters: {args.algorithms}")
            sys.exit(1)

    all_results: List[BenchmarkResult] = []
    total = len(benchmarks) * len(args.sizes)
    idx = 0

    for size in args.sizes:
        for name, bench_fn in benchmarks:
            idx += 1
            print(f"[{idx}/{total}] {name} (n={size}, f={args.features}) ...", end=" ", flush=True)
            try:
                results = bench_fn(size, args.features)
                all_results.extend(results)
                libs = ", ".join(f"{r.library}={r.fit_time_ms:.1f}ms" for r in results)
                print(f"done ({libs})")
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()

    # Output
    print_ascii_table(all_results)

    json_path = args.output_json if args.output_json else "docs/benchmark_cross_library_results.json"
    md_path = args.output_md if args.output_md else "docs/cross-library-benchmark.md"

    write_json(all_results, json_path)
    write_markdown(all_results, md_path)


if __name__ == "__main__":
    main()
