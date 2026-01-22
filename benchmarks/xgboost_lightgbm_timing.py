#!/usr/bin/env python3
"""
XGBoost and LightGBM Benchmark Script

This script generates timing data for XGBoost and LightGBM gradient boosting
implementations to compare against FerroML's benchmarks.

Usage:
    pip install xgboost lightgbm scikit-learn numpy
    python benchmarks/xgboost_lightgbm_timing.py

The output can be compared against FerroML benchmarks run via:
    cargo bench --bench benchmarks -- gradient_boosting
    cargo bench --bench benchmarks -- hist_gradient_boosting

Note: FerroML is expected to be slower than XGBoost/LightGBM due to:
- Pure Rust implementation without SIMD optimizations
- XGBoost/LightGBM have years of optimization for production use
- FerroML prioritizes statistical rigor and code clarity

FerroML's advantages:
- Pure Rust with no external dependencies
- Feature importance with bootstrap confidence intervals
- Native monotonic constraints
- Feature interaction constraints
- Full statistical inference capabilities
"""

import time
import json
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost benchmarks.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Skipping LightGBM benchmarks.")

try:
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Error: scikit-learn not installed. Required for benchmark data generation.")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    library: str
    model_type: str
    operation: str
    n_samples: int
    n_features: int
    n_estimators: int
    max_depth: int
    time_seconds: float
    samples_per_second: Optional[float] = None


def generate_classification_data(n_samples: int, n_features: int, n_classes: int = 2, seed: int = 42):
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=seed
    )
    return X.astype(np.float64), y.astype(np.float64)


def generate_regression_data(n_samples: int, n_features: int, seed: int = 42):
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        noise=0.1,
        random_state=seed
    )
    return X.astype(np.float64), y.astype(np.float64)


def benchmark_fit(model_fn, X, y, n_runs: int = 5) -> float:
    """Benchmark fit time, returning median of n_runs."""
    times = []
    for _ in range(n_runs):
        model = model_fn()
        start = time.perf_counter()
        model.fit(X, y)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.median(times)


def benchmark_predict(model, X, n_runs: int = 10) -> float:
    """Benchmark predict time, returning median of n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.median(times)


def run_xgboost_benchmarks(results: list[BenchmarkResult]):
    """Run XGBoost benchmarks."""
    if not HAS_XGBOOST:
        return

    print("\n=== XGBoost Benchmarks ===")

    # Standard settings matching FerroML benchmarks
    n_estimators = 10
    max_depth = 3
    learning_rate = 0.1

    for n_samples, n_features in [(100, 10), (500, 20), (1000, 20), (2000, 30)]:
        X_clf, y_clf = generate_classification_data(n_samples, n_features)
        X_reg, y_reg = generate_regression_data(n_samples, n_features)

        # Classification
        def xgb_clf():
            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )

        fit_time = benchmark_fit(xgb_clf, X_clf, y_clf)
        model = xgb_clf()
        model.fit(X_clf, y_clf)
        predict_time = benchmark_predict(model, X_clf)

        print(f"XGBoost Classifier fit {n_samples}x{n_features}: {fit_time*1000:.2f}ms")
        print(f"XGBoost Classifier predict {n_samples}x{n_features}: {predict_time*1000:.2f}ms")

        results.append(BenchmarkResult(
            library="xgboost",
            model_type="classifier",
            operation="fit",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=fit_time,
            samples_per_second=n_samples / fit_time
        ))

        results.append(BenchmarkResult(
            library="xgboost",
            model_type="classifier",
            operation="predict",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=predict_time,
            samples_per_second=n_samples / predict_time
        ))

        # Regression
        def xgb_reg():
            return xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                verbosity=0
            )

        fit_time = benchmark_fit(xgb_reg, X_reg, y_reg)
        model = xgb_reg()
        model.fit(X_reg, y_reg)
        predict_time = benchmark_predict(model, X_reg)

        print(f"XGBoost Regressor fit {n_samples}x{n_features}: {fit_time*1000:.2f}ms")
        print(f"XGBoost Regressor predict {n_samples}x{n_features}: {predict_time*1000:.2f}ms")

        results.append(BenchmarkResult(
            library="xgboost",
            model_type="regressor",
            operation="fit",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=fit_time,
            samples_per_second=n_samples / fit_time
        ))

        results.append(BenchmarkResult(
            library="xgboost",
            model_type="regressor",
            operation="predict",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=predict_time,
            samples_per_second=n_samples / predict_time
        ))


def run_lightgbm_benchmarks(results: list[BenchmarkResult]):
    """Run LightGBM benchmarks."""
    if not HAS_LIGHTGBM:
        return

    print("\n=== LightGBM Benchmarks ===")

    n_estimators = 10
    max_depth = 3
    learning_rate = 0.1

    for n_samples, n_features in [(100, 10), (500, 20), (1000, 20), (2000, 30)]:
        X_clf, y_clf = generate_classification_data(n_samples, n_features)
        X_reg, y_reg = generate_regression_data(n_samples, n_features)

        # Classification
        def lgb_clf():
            return lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                verbosity=-1
            )

        fit_time = benchmark_fit(lgb_clf, X_clf, y_clf)
        model = lgb_clf()
        model.fit(X_clf, y_clf)
        predict_time = benchmark_predict(model, X_clf)

        print(f"LightGBM Classifier fit {n_samples}x{n_features}: {fit_time*1000:.2f}ms")
        print(f"LightGBM Classifier predict {n_samples}x{n_features}: {predict_time*1000:.2f}ms")

        results.append(BenchmarkResult(
            library="lightgbm",
            model_type="classifier",
            operation="fit",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=fit_time,
            samples_per_second=n_samples / fit_time
        ))

        results.append(BenchmarkResult(
            library="lightgbm",
            model_type="classifier",
            operation="predict",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=predict_time,
            samples_per_second=n_samples / predict_time
        ))

        # Regression
        def lgb_reg():
            return lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                verbosity=-1
            )

        fit_time = benchmark_fit(lgb_reg, X_reg, y_reg)
        model = lgb_reg()
        model.fit(X_reg, y_reg)
        predict_time = benchmark_predict(model, X_reg)

        print(f"LightGBM Regressor fit {n_samples}x{n_features}: {fit_time*1000:.2f}ms")
        print(f"LightGBM Regressor predict {n_samples}x{n_features}: {predict_time*1000:.2f}ms")

        results.append(BenchmarkResult(
            library="lightgbm",
            model_type="regressor",
            operation="fit",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=fit_time,
            samples_per_second=n_samples / fit_time
        ))

        results.append(BenchmarkResult(
            library="lightgbm",
            model_type="regressor",
            operation="predict",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=predict_time,
            samples_per_second=n_samples / predict_time
        ))


def run_sklearn_benchmarks(results: list[BenchmarkResult]):
    """Run sklearn benchmarks for reference."""
    print("\n=== sklearn Benchmarks ===")

    n_estimators = 10
    max_depth = 3
    learning_rate = 0.1

    for n_samples, n_features in [(100, 10), (500, 20), (1000, 20)]:
        X_clf, y_clf = generate_classification_data(n_samples, n_features)
        X_reg, y_reg = generate_regression_data(n_samples, n_features)

        # Standard GradientBoosting
        def sklearn_gb_clf():
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )

        fit_time = benchmark_fit(sklearn_gb_clf, X_clf, y_clf)
        model = sklearn_gb_clf()
        model.fit(X_clf, y_clf)
        predict_time = benchmark_predict(model, X_clf)

        print(f"sklearn GradientBoosting Classifier fit {n_samples}x{n_features}: {fit_time*1000:.2f}ms")
        print(f"sklearn GradientBoosting Classifier predict {n_samples}x{n_features}: {predict_time*1000:.2f}ms")

        results.append(BenchmarkResult(
            library="sklearn",
            model_type="gradient_boosting_classifier",
            operation="fit",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=fit_time,
            samples_per_second=n_samples / fit_time
        ))

        # HistGradientBoosting (sklearn's LightGBM-style implementation)
        def sklearn_hist_clf():
            return HistGradientBoostingClassifier(
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )

        fit_time = benchmark_fit(sklearn_hist_clf, X_clf, y_clf)
        model = sklearn_hist_clf()
        model.fit(X_clf, y_clf)
        predict_time = benchmark_predict(model, X_clf)

        print(f"sklearn HistGradientBoosting Classifier fit {n_samples}x{n_features}: {fit_time*1000:.2f}ms")
        print(f"sklearn HistGradientBoosting Classifier predict {n_samples}x{n_features}: {predict_time*1000:.2f}ms")

        results.append(BenchmarkResult(
            library="sklearn",
            model_type="hist_gradient_boosting_classifier",
            operation="fit",
            n_samples=n_samples,
            n_features=n_features,
            n_estimators=n_estimators,
            max_depth=max_depth,
            time_seconds=fit_time,
            samples_per_second=n_samples / fit_time
        ))


def run_tree_scaling_benchmark(results: list[BenchmarkResult]):
    """Benchmark how training time scales with number of trees."""
    print("\n=== Tree Scaling Benchmarks ===")

    n_samples = 500
    n_features = 20
    max_depth = 3
    X, y = generate_classification_data(n_samples, n_features)

    for n_trees in [5, 10, 20, 50, 100]:
        if HAS_XGBOOST:
            def xgb_clf():
                return xgb.XGBClassifier(
                    n_estimators=n_trees,
                    max_depth=max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=0
                )

            fit_time = benchmark_fit(xgb_clf, X, y)
            print(f"XGBoost {n_trees} trees: {fit_time*1000:.2f}ms")

            results.append(BenchmarkResult(
                library="xgboost",
                model_type="classifier",
                operation="tree_scaling",
                n_samples=n_samples,
                n_features=n_features,
                n_estimators=n_trees,
                max_depth=max_depth,
                time_seconds=fit_time,
                samples_per_second=n_samples / fit_time
            ))

        if HAS_LIGHTGBM:
            def lgb_clf():
                return lgb.LGBMClassifier(
                    n_estimators=n_trees,
                    max_depth=max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=-1
                )

            fit_time = benchmark_fit(lgb_clf, X, y)
            print(f"LightGBM {n_trees} trees: {fit_time*1000:.2f}ms")

            results.append(BenchmarkResult(
                library="lightgbm",
                model_type="classifier",
                operation="tree_scaling",
                n_samples=n_samples,
                n_features=n_features,
                n_estimators=n_trees,
                max_depth=max_depth,
                time_seconds=fit_time,
                samples_per_second=n_samples / fit_time
            ))


def run_sample_scaling_benchmark(results: list[BenchmarkResult]):
    """Benchmark how training time scales with dataset size."""
    print("\n=== Sample Scaling Benchmarks ===")

    n_features = 20
    n_estimators = 10
    max_depth = 3

    for n_samples in [100, 250, 500, 1000, 2500, 5000]:
        X, y = generate_classification_data(n_samples, n_features)

        if HAS_XGBOOST:
            def xgb_clf():
                return xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=0
                )

            fit_time = benchmark_fit(xgb_clf, X, y)
            print(f"XGBoost {n_samples} samples: {fit_time*1000:.2f}ms")

            results.append(BenchmarkResult(
                library="xgboost",
                model_type="classifier",
                operation="sample_scaling",
                n_samples=n_samples,
                n_features=n_features,
                n_estimators=n_estimators,
                max_depth=max_depth,
                time_seconds=fit_time,
                samples_per_second=n_samples / fit_time
            ))

        if HAS_LIGHTGBM:
            def lgb_clf():
                return lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=-1
                )

            fit_time = benchmark_fit(lgb_clf, X, y)
            print(f"LightGBM {n_samples} samples: {fit_time*1000:.2f}ms")

            results.append(BenchmarkResult(
                library="lightgbm",
                model_type="classifier",
                operation="sample_scaling",
                n_samples=n_samples,
                n_features=n_features,
                n_estimators=n_estimators,
                max_depth=max_depth,
                time_seconds=fit_time,
                samples_per_second=n_samples / fit_time
            ))


def main():
    """Run all benchmarks and output results."""
    print("=" * 60)
    print("XGBoost / LightGBM / sklearn Gradient Boosting Benchmarks")
    print("=" * 60)
    print("\nThese results can be compared against FerroML benchmarks:")
    print("  cargo bench --bench benchmarks -- gradient_boosting")
    print("  cargo bench --bench benchmarks -- hist_gradient_boosting")
    print()

    results: list[BenchmarkResult] = []

    run_xgboost_benchmarks(results)
    run_lightgbm_benchmarks(results)
    run_sklearn_benchmarks(results)
    run_tree_scaling_benchmark(results)
    run_sample_scaling_benchmark(results)

    # Output results as JSON for further analysis
    output_path = "benchmarks/gradient_boosting_results.json"
    results_dict = [
        {
            "library": r.library,
            "model_type": r.model_type,
            "operation": r.operation,
            "n_samples": r.n_samples,
            "n_features": r.n_features,
            "n_estimators": r.n_estimators,
            "max_depth": r.max_depth,
            "time_seconds": r.time_seconds,
            "samples_per_second": r.samples_per_second
        }
        for r in results
    ]

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n\nResults saved to {output_path}")
    print("\nSummary:")
    print("-" * 60)

    # Print comparison table
    print(f"{'Library':<15} {'Model':<25} {'Op':<10} {'Size':<12} {'Time (ms)':<12}")
    print("-" * 60)
    for r in results[:20]:  # First 20 results
        size = f"{r.n_samples}x{r.n_features}"
        print(f"{r.library:<15} {r.model_type:<25} {r.operation:<10} {size:<12} {r.time_seconds*1000:>8.2f}")


if __name__ == "__main__":
    main()
