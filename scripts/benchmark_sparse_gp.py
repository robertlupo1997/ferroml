#!/usr/bin/env python3
"""Benchmark Exact GP vs Sparse GP (FITC/VFE) vs SVGP.

Compares fit time, predict time, and R^2 across dataset sizes for the four
GP regression variants in FerroML.

Usage:
    python scripts/benchmark_sparse_gp.py
    python scripts/benchmark_sparse_gp.py --output results.json
    python scripts/benchmark_sparse_gp.py --markdown docs/benchmark-sparse-gp.md
"""

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List

import numpy as np

from ferroml.gaussian_process import (
    GaussianProcessRegressor,
    RBF,
    SparseGPRegressor,
    SVGPRegressor,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GPBenchmarkResult:
    model: str
    n_samples: int
    n_inducing: int
    fit_time: float
    predict_time: float
    r2: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor() or "unknown",
        "numpy_version": np.__version__,
        "timestamp": datetime.now().isoformat(),
    }


def make_sin_data(n: int, seed: int = 42) -> tuple:
    """Synthetic 1-D sin data with noise."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 10, (n, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(n)
    return X, y


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

DATASET_SIZES = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
INDUCING_COUNTS = [50, 100, 200]
# Skip exact GP above this threshold (O(n^3) is too slow)
EXACT_GP_MAX = 5_000


def bench_exact_gp(X: np.ndarray, y: np.ndarray) -> GPBenchmarkResult:
    kernel = RBF(1.0)
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.01)

    t0 = time.perf_counter()
    model.fit(X, y)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X)
    predict_time = time.perf_counter() - t0

    return GPBenchmarkResult(
        model="ExactGP",
        n_samples=len(X),
        n_inducing=0,
        fit_time=fit_time,
        predict_time=predict_time,
        r2=r2_score(y, preds),
    )


def bench_sparse_gp(
    X: np.ndarray, y: np.ndarray, n_inducing: int, approx: str,
) -> GPBenchmarkResult:
    kernel = RBF(1.0)
    model = SparseGPRegressor(
        kernel=kernel, alpha=0.01, n_inducing=n_inducing, approximation=approx.lower()
    )

    t0 = time.perf_counter()
    model.fit(X, y)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X)
    predict_time = time.perf_counter() - t0

    return GPBenchmarkResult(
        model=approx,
        n_samples=len(X),
        n_inducing=n_inducing,
        fit_time=fit_time,
        predict_time=predict_time,
        r2=r2_score(y, preds),
    )


def bench_svgp(
    X: np.ndarray, y: np.ndarray, n_inducing: int
) -> GPBenchmarkResult:
    kernel = RBF(1.0)
    model = SVGPRegressor(
        kernel=kernel,
        noise_variance=0.01,
        n_inducing=n_inducing,
        n_epochs=20,
        batch_size=256,
        learning_rate=0.05,
    )

    t0 = time.perf_counter()
    model.fit(X, y)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X)
    predict_time = time.perf_counter() - t0

    return GPBenchmarkResult(
        model="SVGP",
        n_samples=len(X),
        n_inducing=n_inducing,
        fit_time=fit_time,
        predict_time=predict_time,
        r2=r2_score(y, preds),
    )


def run_benchmarks() -> List[GPBenchmarkResult]:
    results: List[GPBenchmarkResult] = []

    for n in DATASET_SIZES:
        X, y = make_sin_data(n)
        print(f"\n--- n={n} ---")

        # Exact GP (skip large sizes)
        if n <= EXACT_GP_MAX:
            try:
                r = bench_exact_gp(X, y)
                print(f"  ExactGP:  fit={r.fit_time:.3f}s  pred={r.predict_time:.3f}s  R2={r.r2:.4f}")
                results.append(r)
            except Exception as e:
                print(f"  ExactGP:  FAILED ({e})")

        for m in INDUCING_COUNTS:
            if m >= n:
                continue

            # FITC
            try:
                r = bench_sparse_gp(X, y, m, "FITC")
                print(f"  FITC m={m}: fit={r.fit_time:.3f}s  pred={r.predict_time:.3f}s  R2={r.r2:.4f}")
                results.append(r)
            except Exception as e:
                print(f"  FITC m={m}: FAILED ({e})")

            # VFE
            try:
                r = bench_sparse_gp(X, y, m, "VFE")
                print(f"  VFE  m={m}: fit={r.fit_time:.3f}s  pred={r.predict_time:.3f}s  R2={r.r2:.4f}")
                results.append(r)
            except Exception as e:
                print(f"  VFE  m={m}: FAILED ({e})")

            # SVGP
            try:
                r = bench_svgp(X, y, m)
                print(f"  SVGP m={m}: fit={r.fit_time:.3f}s  pred={r.predict_time:.3f}s  R2={r.r2:.4f}")
                results.append(r)
            except Exception as e:
                print(f"  SVGP m={m}: FAILED ({e})")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def results_to_markdown(results: List[GPBenchmarkResult]) -> str:
    lines = [
        "# Sparse GP Benchmark Results",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Model | n | m | Fit (s) | Predict (s) | R^2 |",
        "|-------|--:|--:|--------:|------------:|----:|",
    ]
    for r in results:
        m_str = str(r.n_inducing) if r.n_inducing > 0 else "-"
        lines.append(
            f"| {r.model} | {r.n_samples} | {m_str} | {r.fit_time:.4f} | {r.predict_time:.4f} | {r.r2:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Benchmark Sparse GP models")
    parser.add_argument("--output", type=str, help="Path for JSON output")
    parser.add_argument("--markdown", type=str, help="Path for Markdown output")
    args = parser.parse_args()

    print("=" * 60)
    print("FerroML Sparse GP Benchmark")
    print("Exact GP vs FITC vs VFE vs SVGP")
    print("=" * 60)

    results = run_benchmarks()

    payload = {
        "system_info": get_system_info(),
        "results": [asdict(r) for r in results],
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON results written to {args.output}")

    if args.markdown:
        md = results_to_markdown(results)
        with open(args.markdown, "w") as f:
            f.write(md)
        print(f"Markdown report written to {args.markdown}")

    if not args.output and not args.markdown:
        print("\n" + results_to_markdown(results))


if __name__ == "__main__":
    main()
