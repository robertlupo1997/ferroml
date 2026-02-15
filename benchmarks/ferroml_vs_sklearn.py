#!/usr/bin/env python3
"""
FerroML vs scikit-learn Benchmark Comparison

This script compares FerroML's AutoML interface against scikit-learn models on:
- Iris dataset (classification)
- Diabetes dataset (regression)

Metrics:
- Accuracy/R2 score
- Training time
- Prediction time (if available)
"""

import sys
import time
import warnings
import numpy as np

# Check for required dependencies
try:
    from sklearn.datasets import load_iris, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not available")
    print("Install with: pip install scikit-learn")
    print()

try:
    from ferroml.automl import AutoML, AutoMLConfig
    FERROML_AVAILABLE = True
except ImportError:
    FERROML_AVAILABLE = False
    print("WARNING: FerroML not available")
    print("Install with: cd ferroml-python && pip install -e .")
    print()

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore')


def format_time(seconds):
    """Format time in a readable way."""
    if seconds < 0.001:
        return f"{seconds*1000000:.1f}us"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"


def benchmark_classification():
    """Benchmark on Iris dataset (classification)."""
    print("=" * 80)
    print("CLASSIFICATION BENCHMARK (Iris Dataset)")
    print("=" * 80)
    print()

    if not SKLEARN_AVAILABLE:
        print("Skipping: scikit-learn not available\n")
        return None

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")
    print()

    results = []

    # Sklearn models
    sklearn_models = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    for name, model in sklearn_models:
        # Training
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Prediction
        start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start

        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            'library': 'sklearn',
            'model': name,
            'accuracy': accuracy,
            'train_time': train_time,
            'pred_time': pred_time,
        })

    # FerroML AutoML
    if FERROML_AVAILABLE:
        config = AutoMLConfig(
            task="classification",
            metric="accuracy",
            time_budget_seconds=30,  # Short budget for quick comparison
            cv_folds=3,
            statistical_tests=False,  # Skip for faster benchmarking
        )
        automl = AutoML(config)

        # Training
        start = time.time()
        result = automl.fit(X_train, y_train.astype(np.float64))
        train_time = time.time() - start

        # Get best model info
        best = result.best_model()
        best_score = best.cv_score if best else 0.0

        # Prediction using result.predict(X_train, y_train, X_test)
        pred_time = None
        accuracy = best_score  # Use CV score as fallback

        predict_available = False
        try:
            start = time.time()
            y_pred = result.predict(
                X_train, y_train.astype(np.float64), X_test
            )
            pred_time = time.time() - start
            accuracy = accuracy_score(y_test, np.round(y_pred).astype(int))
            predict_available = True
        except Exception:
            pass

        results.append({
            'library': 'FerroML',
            'model': f"AutoML ({best.algorithm if best else 'N/A'})",
            'accuracy': accuracy,
            'train_time': train_time,
            'pred_time': pred_time,
            'note': '' if predict_available else 'CV score',
        })

    return results


def benchmark_regression():
    """Benchmark on Diabetes dataset (regression)."""
    print("=" * 80)
    print("REGRESSION BENCHMARK (Diabetes Dataset)")
    print("=" * 80)
    print()

    if not SKLEARN_AVAILABLE:
        print("Skipping: scikit-learn not available\n")
        return None

    # Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    print()

    results = []

    # Sklearn models
    sklearn_models = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0, random_state=42)),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]

    for name, model in sklearn_models:
        # Training
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Prediction
        start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start

        r2 = r2_score(y_test, y_pred)

        results.append({
            'library': 'sklearn',
            'model': name,
            'r2': r2,
            'train_time': train_time,
            'pred_time': pred_time,
        })

    # FerroML AutoML
    if FERROML_AVAILABLE:
        config = AutoMLConfig(
            task="regression",
            metric="r2",
            time_budget_seconds=30,  # Short budget for quick comparison
            cv_folds=3,
            statistical_tests=False,  # Skip for faster benchmarking
        )
        automl = AutoML(config)

        # Training
        start = time.time()
        result = automl.fit(X_train, y_train)
        train_time = time.time() - start

        # Get best model info
        best = result.best_model()
        best_score = best.cv_score if best else 0.0

        # Prediction using result.predict(X_train, y_train, X_test)
        pred_time = None
        r2 = best_score  # Use CV score as fallback

        predict_available = False
        try:
            start = time.time()
            y_pred = result.predict(X_train, y_train, X_test)
            pred_time = time.time() - start
            r2 = r2_score(y_test, y_pred)
            predict_available = True
        except Exception:
            pass

        results.append({
            'library': 'FerroML',
            'model': f"AutoML ({best.algorithm if best else 'N/A'})",
            'r2': r2,
            'train_time': train_time,
            'pred_time': pred_time,
            'note': '' if predict_available else 'CV score',
        })

    return results


def print_classification_results(results):
    """Print classification benchmark results as a markdown table."""
    if not results:
        return

    print("\n## Classification Results\n")
    print("| Library | Model | Accuracy | Train Time | Predict Time | Notes |")
    print("|---------|-------|----------|------------|--------------|-------|")

    for r in results:
        pred_time_str = format_time(r['pred_time']) if r['pred_time'] is not None else "N/A"
        note = r.get('note', '')
        print(f"| {r['library']:8s} | {r['model']:25s} | {r['accuracy']:8.4f} | "
              f"{format_time(r['train_time']):>10s} | {pred_time_str:>12s} | {note} |")
    print()


def print_regression_results(results):
    """Print regression benchmark results as a markdown table."""
    if not results:
        return

    print("\n## Regression Results\n")
    print("| Library | Model | R² Score | Train Time | Predict Time | Notes |")
    print("|---------|-------|----------|------------|--------------|-------|")

    for r in results:
        pred_time_str = format_time(r['pred_time']) if r['pred_time'] is not None else "N/A"
        note = r.get('note', '')
        print(f"| {r['library']:8s} | {r['model']:25s} | {r['r2']:8.4f} | "
              f"{format_time(r['train_time']):>10s} | {pred_time_str:>12s} | {note} |")
    print()


def main():
    """Run all benchmarks."""
    print()
    print("=" * 80)
    print("FerroML vs scikit-learn Benchmark")
    print("=" * 80)
    print()

    if not SKLEARN_AVAILABLE and not FERROML_AVAILABLE:
        print("ERROR: Neither sklearn nor FerroML is available. Cannot run benchmarks.")
        print()
        print("Install dependencies:")
        print("  pip install scikit-learn")
        print("  cd ferroml-python && pip install -e .")
        sys.exit(1)

    if not SKLEARN_AVAILABLE:
        print("WARNING: scikit-learn not available - only FerroML will be tested")
        print()

    if not FERROML_AVAILABLE:
        print("WARNING: FerroML not available - only sklearn will be tested")
        print()

    # Run benchmarks
    classification_results = benchmark_classification()
    regression_results = benchmark_regression()

    # Print results
    print("\n")
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    if classification_results:
        print_classification_results(classification_results)

    if regression_results:
        print_regression_results(regression_results)

    print("\n## Notes")
    print()
    print("- FerroML uses AutoML with a 30-second time budget per task")
    print("- FerroML shows the best model selected by AutoML")
    print("- Prediction time marked as 'N/A' means predict() is not yet implemented")
    print("- 'CV score' means cross-validation score is shown (test set not evaluated)")
    print("- sklearn models use default or standard hyperparameters")
    print()

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
