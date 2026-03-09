#!/usr/bin/env python3
"""
Check Criterion benchmark results against baseline expected ranges.

Reads Criterion output from target/criterion/ and compares point estimates
against the expected ranges defined in ferroml-core/benches/baseline.json.

A benchmark is flagged as a regression if it exceeds the upper bound of
its expected range by more than the configured threshold (default 20%).

Exit codes:
  0 - No regressions detected (or no data available)
  1 - One or more regressions detected
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_project_root() -> Path:
    """Find the project root by looking for ferroml-core/benches/baseline.json."""
    # Try common locations relative to this script
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent,  # scripts/ -> project root
        Path.cwd(),
        Path.cwd().parent,
    ]
    for candidate in candidates:
        if (candidate / "ferroml-core" / "benches" / "baseline.json").exists():
            return candidate
    return Path.cwd()


def load_baseline(baseline_path: Path) -> Tuple[Dict, float]:
    """Load baseline.json and return (benchmarks dict, threshold percent)."""
    with open(baseline_path, "r") as f:
        data = json.load(f)
    benchmarks = data.get("benchmarks", {})
    threshold = data.get("regression_threshold_percent", 20.0)
    return benchmarks, threshold


def load_criterion_results(criterion_dir: Path) -> Dict[str, float]:
    """
    Load Criterion benchmark results from target/criterion/.

    Each benchmark has a directory structure like:
      target/criterion/<group>/<bench_name>/new/estimates.json

    The point_estimate in estimates.json is in nanoseconds.
    Returns a dict mapping benchmark name -> time in microseconds.
    """
    results = {}
    if not criterion_dir.exists():
        return results

    for estimates_file in criterion_dir.rglob("new/estimates.json"):
        try:
            with open(estimates_file, "r") as f:
                estimates = json.load(f)

            # Extract point estimate (nanoseconds)
            point_estimate_ns = None
            if "mean" in estimates:
                point_estimate_ns = estimates["mean"].get("point_estimate")
            elif "median" in estimates:
                point_estimate_ns = estimates["median"].get("point_estimate")
            elif "slope" in estimates:
                point_estimate_ns = estimates["slope"].get("point_estimate")

            if point_estimate_ns is None:
                continue

            # Convert ns to us
            point_estimate_us = point_estimate_ns / 1000.0

            # Build benchmark name from directory path
            # e.g. target/criterion/LinearRegression/fit/100x10/new/estimates.json
            # -> LinearRegression/fit/100x10
            rel = estimates_file.relative_to(criterion_dir)
            # Remove "new/estimates.json" suffix (2 levels)
            parts = list(rel.parts[:-2])
            bench_name = "/".join(parts)

            if bench_name:
                results[bench_name] = point_estimate_us

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  Warning: Could not parse {estimates_file}: {e}")
            continue

    return results


def check_regressions(
    baseline: Dict,
    results: Dict[str, float],
    threshold_pct: float,
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Compare results against baseline ranges.

    Returns (regressions, within_range, missing) where each is a list of tuples.
    A regression is when the measured value exceeds upper_bound * (1 + threshold/100).
    """
    regressions = []
    within_range = []
    missing = []

    for bench_name, spec in baseline.items():
        expected_range = spec.get("expected_range_us")
        if expected_range is None or len(expected_range) != 2:
            continue

        lower_us, upper_us = expected_range

        if bench_name not in results:
            missing.append((bench_name, lower_us, upper_us))
            continue

        measured_us = results[bench_name]
        regression_limit = upper_us * (1.0 + threshold_pct / 100.0)

        if measured_us > regression_limit:
            pct_over = ((measured_us - upper_us) / upper_us) * 100.0
            regressions.append((bench_name, lower_us, upper_us, measured_us, pct_over))
        else:
            within_range.append((bench_name, lower_us, upper_us, measured_us))

    return regressions, within_range, missing


def format_time(us: float) -> str:
    """Format microseconds into a human-readable string."""
    if us < 1000:
        return f"{us:.1f} us"
    elif us < 1_000_000:
        return f"{us / 1000:.2f} ms"
    else:
        return f"{us / 1_000_000:.2f} s"


def print_summary(
    regressions: List[Tuple],
    within_range: List[Tuple],
    missing: List[Tuple],
    threshold_pct: float,
) -> None:
    """Print a formatted summary table."""
    print("=" * 78)
    print("BENCHMARK REGRESSION CHECK")
    print(f"Regression threshold: >{threshold_pct:.0f}% above upper expected range")
    print("=" * 78)
    print()

    # Summary counts
    total_checked = len(regressions) + len(within_range)
    print(f"Checked:     {total_checked}")
    print(f"Passed:      {len(within_range)}")
    print(f"Regressions: {len(regressions)}")
    print(f"Missing:     {len(missing)}")
    print()

    if regressions:
        print("-" * 78)
        print("REGRESSIONS DETECTED:")
        print("-" * 78)
        print(f"{'Benchmark':<45} {'Expected':<15} {'Measured':<12} {'Over':<8}")
        print(f"{'':.<45} {'(upper)':.<15} {'':.<12} {'':.<8}")
        for name, _low, upper, measured, pct in sorted(regressions, key=lambda x: -x[4]):
            print(f"  {name:<43} {format_time(upper):<15} {format_time(measured):<12} +{pct:.1f}%")
        print()

    if within_range:
        print("-" * 78)
        print("PASSED:")
        print("-" * 78)
        print(f"{'Benchmark':<45} {'Range':<22} {'Measured':<12}")
        for name, low, upper, measured in sorted(within_range, key=lambda x: x[0]):
            range_str = f"{format_time(low)} - {format_time(upper)}"
            print(f"  {name:<43} {range_str:<22} {format_time(measured)}")
        print()

    if missing:
        print("-" * 78)
        print("MISSING (not found in Criterion output, skipped):")
        print("-" * 78)
        for name, _low, _upper in sorted(missing, key=lambda x: x[0]):
            print(f"  {name}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Criterion benchmark results against baseline expected ranges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                              # Use default paths
  %(prog)s --criterion-dir target/criterion
  %(prog)s --baseline ferroml-core/benches/baseline.json
  %(prog)s --threshold 30               # Allow 30%% over upper bound
""",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline.json (default: ferroml-core/benches/baseline.json)",
    )
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=None,
        help="Path to Criterion output directory (default: target/criterion)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Regression threshold percentage (default: from baseline.json, typically 20)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write results as JSON to this file",
    )

    args = parser.parse_args()

    # Resolve paths
    root = find_project_root()

    baseline_path = args.baseline or (root / "ferroml-core" / "benches" / "baseline.json")
    criterion_dir = args.criterion_dir or (root / "target" / "criterion")

    # Load baseline
    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        return 1

    baseline, default_threshold = load_baseline(baseline_path)
    threshold = args.threshold if args.threshold is not None else default_threshold

    print(f"Baseline:      {baseline_path}")
    print(f"Criterion dir: {criterion_dir}")
    print(f"Benchmarks in baseline: {len(baseline)}")
    print()

    # Load Criterion results
    results = load_criterion_results(criterion_dir)

    if not results:
        print("No Criterion benchmark results found.")
        print("Run `cargo bench -p ferroml-core` first to generate results.")
        print()
        print("Exiting with success (no data to compare).")
        return 0

    print(f"Benchmarks found in Criterion output: {len(results)}")
    print()

    # Check for regressions
    regressions, within_range, missing = check_regressions(baseline, results, threshold)

    # Print summary
    print_summary(regressions, within_range, missing, threshold)

    # Optional JSON output
    if args.json_output:
        output = {
            "threshold_percent": threshold,
            "total_checked": len(regressions) + len(within_range),
            "regressions_count": len(regressions),
            "passed_count": len(within_range),
            "missing_count": len(missing),
            "regressions": [
                {
                    "benchmark": name,
                    "expected_upper_us": upper,
                    "measured_us": measured,
                    "percent_over": round(pct, 1),
                }
                for name, _low, upper, measured, pct in regressions
            ],
            "passed": [
                {
                    "benchmark": name,
                    "expected_range_us": [low, upper],
                    "measured_us": round(measured, 1),
                }
                for name, low, upper, measured in within_range
            ],
            "missing": [name for name, _, _ in missing],
        }
        with open(args.json_output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"JSON results written to: {args.json_output}")

    # Exit code
    if regressions:
        print(f"FAILED: {len(regressions)} regression(s) detected.")
        return 1
    else:
        print("PASSED: No regressions detected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
