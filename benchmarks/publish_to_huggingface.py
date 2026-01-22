#!/usr/bin/env python3
"""
Publish FerroML Benchmark Results to HuggingFace Hub

This script collects benchmark results from FerroML's Criterion benchmarks
and publishes them as a dataset to HuggingFace Hub.

Usage:
    # Run all benchmarks and publish (requires HF_TOKEN env var or huggingface-cli login)
    python benchmarks/publish_to_huggingface.py

    # Use existing benchmark results (skip running benchmarks)
    python benchmarks/publish_to_huggingface.py --skip-benchmarks

    # Specify custom repository
    python benchmarks/publish_to_huggingface.py --repo-id username/ferroml-benchmarks

Requirements:
    pip install huggingface_hub datasets pandas
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, create_repo
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("Warning: huggingface_hub not installed. Run: pip install huggingface_hub")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Run: pip install pandas")


@dataclass
class BenchmarkResult:
    """A single benchmark result."""
    library: str              # ferroml, sklearn, xgboost, lightgbm
    model_type: str           # LinearRegression, DecisionTree, etc.
    operation: str            # fit, predict, fit_transform
    n_samples: int            # Number of samples
    n_features: int           # Number of features
    time_seconds: float       # Median time in seconds
    time_std_seconds: Optional[float] = None  # Standard deviation
    throughput: Optional[float] = None  # Samples per second
    n_estimators: Optional[int] = None  # For ensemble methods
    max_depth: Optional[int] = None     # For tree methods


@dataclass
class BenchmarkMetadata:
    """Metadata about the benchmark run."""
    ferroml_version: str
    rust_version: str
    os: str
    os_version: str
    cpu: str
    timestamp: str
    commit_hash: Optional[str] = None


def get_rust_version() -> str:
    """Get the installed Rust version."""
    try:
        result = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_ferroml_version() -> str:
    """Get FerroML version from Cargo.toml."""
    try:
        cargo_toml = Path(__file__).parent.parent / "Cargo.toml"
        content = cargo_toml.read_text()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip()[:8]
    except Exception:
        return None


def get_cpu_info() -> str:
    """Get CPU information."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        elif platform.system() == "Windows":
            return platform.processor()
    except Exception:
        pass
    return platform.processor() or "unknown"


def collect_metadata() -> BenchmarkMetadata:
    """Collect system metadata for the benchmark run."""
    return BenchmarkMetadata(
        ferroml_version=get_ferroml_version(),
        rust_version=get_rust_version(),
        os=platform.system(),
        os_version=platform.release(),
        cpu=get_cpu_info(),
        timestamp=datetime.utcnow().isoformat() + "Z",
        commit_hash=get_git_commit()
    )


def run_cargo_bench() -> str:
    """Run cargo bench and return the output."""
    print("Running FerroML benchmarks (this may take several minutes)...")

    result = subprocess.run(
        ["cargo", "bench", "--bench", "benchmarks"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    if result.returncode != 0:
        print(f"Warning: cargo bench failed with code {result.returncode}")
        print(f"stderr: {result.stderr}")

    return result.stdout + result.stderr


def parse_criterion_output(output: str) -> list[BenchmarkResult]:
    """Parse Criterion benchmark output into BenchmarkResult objects."""
    results = []

    # Pattern: "GroupName/SubName   time:   [1.234 ms 1.456 ms 1.678 ms]"
    # More specific patterns for FerroML benchmark output
    patterns = [
        # Standard format: Model/operation/samples/{size}
        r"(?P<group>\w+)/(?P<op>\w+)/samples/(?P<size>\d+x\d+)\s+"
        r"time:\s+\[[\d.]+ [mµn]s\s+(?P<median>[\d.]+)\s+(?P<unit>[mµn]s)",

        # Scaling format: Scaling/Model/{n_samples}
        r"Scaling/(?P<model>\w+)/(?P<param>\d+)\s+"
        r"time:\s+\[[\d.]+ [mµn]s\s+(?P<median>[\d.]+)\s+(?P<unit>[mµn]s)",

        # Tree scaling format: Scaling/GradientBoosting/Trees/n_estimators/{n}
        r"Scaling/(?P<model>\w+)/(?P<param_name>\w+)/(?P<param_type>\w+)/(?P<value>\d+)\s+"
        r"time:\s+\[[\d.]+ [mµn]s\s+(?P<median>[\d.]+)\s+(?P<unit>[mµn]s)",
    ]

    # Map units to seconds multipliers
    unit_to_seconds = {
        "s": 1.0,
        "ms": 1e-3,
        "µs": 1e-6,
        "us": 1e-6,
        "ns": 1e-9,
    }

    # Parse standard benchmark lines
    for line in output.split('\n'):
        # Standard format: LinearRegression/fit/samples/100x10
        match = re.search(
            r"(\w+)/(\w+)/samples/(\d+)x(\d+)\s+time:\s+\[[\d.]+ [mµn]s\s+([\d.]+)\s+([mµn]s)",
            line
        )
        if match:
            model, op, n_samples, n_features, time_val, unit = match.groups()
            time_seconds = float(time_val) * unit_to_seconds.get(unit, 1e-3)

            results.append(BenchmarkResult(
                library="ferroml",
                model_type=model,
                operation=op,
                n_samples=int(n_samples),
                n_features=int(n_features),
                time_seconds=time_seconds,
                throughput=int(n_samples) / time_seconds if time_seconds > 0 else None
            ))
            continue

        # Scaling format: Scaling/LinearRegression/1000
        match = re.search(
            r"Scaling/(\w+)/(\d+)\s+time:\s+\[[\d.]+ [mµn]s\s+([\d.]+)\s+([mµn]s)",
            line
        )
        if match:
            model, n_samples, time_val, unit = match.groups()
            time_seconds = float(time_val) * unit_to_seconds.get(unit, 1e-3)

            results.append(BenchmarkResult(
                library="ferroml",
                model_type=model,
                operation="scaling",
                n_samples=int(n_samples),
                n_features=50,  # Default for scaling benchmarks
                time_seconds=time_seconds,
                throughput=int(n_samples) / time_seconds if time_seconds > 0 else None
            ))
            continue

        # Gradient boosting tree scaling: Scaling/GradientBoosting/Trees/n_estimators/50
        match = re.search(
            r"Scaling/(\w+)/Trees/(\w+)/(\d+)\s+time:\s+\[[\d.]+ [mµn]s\s+([\d.]+)\s+([mµn]s)",
            line
        )
        if match:
            model, param_name, n_trees, time_val, unit = match.groups()
            time_seconds = float(time_val) * unit_to_seconds.get(unit, 1e-3)

            results.append(BenchmarkResult(
                library="ferroml",
                model_type=model,
                operation="tree_scaling",
                n_samples=500,  # Fixed for tree scaling
                n_features=20,
                time_seconds=time_seconds,
                n_estimators=int(n_trees)
            ))
            continue

        # Sample scaling comparison: Scaling/GradientBoosting/Samples/standard/1000
        match = re.search(
            r"Scaling/(\w+)/Samples/(\w+)/(\d+)\s+time:\s+\[[\d.]+ [mµn]s\s+([\d.]+)\s+([mµn]s)",
            line
        )
        if match:
            model, variant, n_samples, time_val, unit = match.groups()
            time_seconds = float(time_val) * unit_to_seconds.get(unit, 1e-3)

            results.append(BenchmarkResult(
                library="ferroml",
                model_type=f"{model}_{variant}",
                operation="sample_scaling",
                n_samples=int(n_samples),
                n_features=20,
                time_seconds=time_seconds,
                throughput=int(n_samples) / time_seconds if time_seconds > 0 else None
            ))
            continue

        # Scaler benchmarks: StandardScaler/fit_transform/samples/100x10
        match = re.search(
            r"(\w+Scaler)/(\w+)/samples/(\d+)x(\d+)\s+time:\s+\[[\d.]+ [mµn]s\s+([\d.]+)\s+([mµn]s)",
            line
        )
        if match:
            scaler, op, n_samples, n_features, time_val, unit = match.groups()
            time_seconds = float(time_val) * unit_to_seconds.get(unit, 1e-3)

            results.append(BenchmarkResult(
                library="ferroml",
                model_type=scaler,
                operation=op,
                n_samples=int(n_samples),
                n_features=int(n_features),
                time_seconds=time_seconds,
                throughput=int(n_samples) / time_seconds if time_seconds > 0 else None
            ))
            continue

    return results


def load_xgboost_lightgbm_results() -> list[BenchmarkResult]:
    """Load pre-generated XGBoost/LightGBM benchmark results."""
    results_path = Path(__file__).parent / "gradient_boosting_results.json"

    if not results_path.exists():
        print(f"No XGBoost/LightGBM results found at {results_path}")
        print("Run benchmarks/xgboost_lightgbm_timing.py first to generate them.")
        return []

    with open(results_path) as f:
        data = json.load(f)

    results = []
    for item in data:
        results.append(BenchmarkResult(
            library=item["library"],
            model_type=item["model_type"],
            operation=item["operation"],
            n_samples=item["n_samples"],
            n_features=item["n_features"],
            time_seconds=item["time_seconds"],
            throughput=item.get("samples_per_second"),
            n_estimators=item.get("n_estimators"),
            max_depth=item.get("max_depth")
        ))

    return results


def results_to_dataframe(results: list[BenchmarkResult], metadata: BenchmarkMetadata):
    """Convert results to a pandas DataFrame."""
    if not HAS_PANDAS:
        raise ImportError("pandas is required for DataFrame conversion")

    # Convert to dicts
    data = [asdict(r) for r in results]
    df = pd.DataFrame(data)

    # Add metadata columns
    df["benchmark_date"] = metadata.timestamp
    df["ferroml_version"] = metadata.ferroml_version
    df["rust_version"] = metadata.rust_version
    df["os"] = metadata.os
    df["cpu"] = metadata.cpu
    if metadata.commit_hash:
        df["commit_hash"] = metadata.commit_hash

    return df


def publish_to_huggingface(
    results: list[BenchmarkResult],
    metadata: BenchmarkMetadata,
    repo_id: str,
    private: bool = False
) -> str:
    """Publish benchmark results to HuggingFace Hub."""
    if not HAS_HF_HUB:
        raise ImportError("huggingface_hub is required for publishing")
    if not HAS_PANDAS:
        raise ImportError("pandas is required for publishing")

    api = HfApi()

    # Create or get the repository
    try:
        create_repo(
            repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"Repository '{repo_id}' ready.")
    except Exception as e:
        print(f"Note: Could not create repository: {e}")

    # Convert results to DataFrame
    df = results_to_dataframe(results, metadata)

    # Save to temporary files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = Path(__file__).parent / "hf_upload"
    local_dir.mkdir(exist_ok=True)

    # Save current results
    current_csv = local_dir / f"benchmarks_{timestamp}.csv"
    df.to_csv(current_csv, index=False)

    # Save as parquet for efficiency
    current_parquet = local_dir / f"benchmarks_{timestamp}.parquet"
    df.to_parquet(current_parquet, index=False)

    # Create/update a latest.csv for easy access
    latest_csv = local_dir / "latest.csv"
    df.to_csv(latest_csv, index=False)

    # Create metadata JSON
    metadata_json = local_dir / "metadata.json"
    with open(metadata_json, "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    # Create README with dataset card
    readme_content = f"""---
license: apache-2.0
language:
- en
tags:
- benchmark
- machine-learning
- rust
- ferroml
size_categories:
- 1K<n<10K
---

# FerroML Benchmark Results

This dataset contains performance benchmarks for [FerroML](https://github.com/ferroml/ferroml),
a statistically rigorous AutoML library in Rust.

## Dataset Description

The benchmarks compare FerroML against:
- **scikit-learn**: Python ML library
- **XGBoost**: Gradient boosting library
- **LightGBM**: Microsoft's gradient boosting library

## Benchmark Categories

1. **Linear Models**: LinearRegression, Ridge, Lasso
2. **Tree Models**: DecisionTree, RandomForest
3. **Gradient Boosting**: Standard and Histogram-based
4. **Preprocessing**: Scalers (Standard, MinMax, Robust, MaxAbs)

## Latest Benchmark Run

- **Date**: {metadata.timestamp}
- **FerroML Version**: {metadata.ferroml_version}
- **Rust Version**: {metadata.rust_version}
- **OS**: {metadata.os} {metadata.os_version}
- **CPU**: {metadata.cpu}
{f"- **Commit**: {metadata.commit_hash}" if metadata.commit_hash else ""}

## Usage

```python
from datasets import load_dataset

# Load the latest benchmarks
dataset = load_dataset("{repo_id}")

# Or load specific benchmark file
import pandas as pd
df = pd.read_csv("hf://datasets/{repo_id}/latest.csv")

# Filter for FerroML results only
ferroml_results = df[df["library"] == "ferroml"]

# Compare gradient boosting performance
gb_comparison = df[df["model_type"].str.contains("Boosting")]
```

## Key Findings

FerroML's advantages:
- **Pure Rust**: No external dependencies, easy deployment
- **Statistical rigor**: Feature importance with confidence intervals
- **Native constraints**: Monotonic and interaction constraints built-in
- **Full integration**: Seamless pipeline and AutoML integration

Expected relative performance (vs XGBoost/LightGBM):
- Small datasets (<10K samples): FerroML competitive, 2-5x slower
- Medium datasets (10K-100K): 5-10x slower
- Large datasets (>100K): 10-50x slower (due to SIMD optimizations in XGBoost/LightGBM)

## License

Apache 2.0
"""

    readme_path = local_dir / "README.md"
    readme_path.write_text(readme_content)

    # Upload all files
    print(f"Uploading benchmarks to {repo_id}...")

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update benchmarks: {timestamp}"
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"Successfully published to: {url}")

    return url


def main():
    parser = argparse.ArgumentParser(
        description="Publish FerroML benchmark results to HuggingFace Hub"
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip running benchmarks, use existing results"
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip loading XGBoost/LightGBM results"
    )
    parser.add_argument(
        "--repo-id",
        default="ferroml/benchmarks",
        help="HuggingFace Hub repository ID (default: ferroml/benchmarks)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only save results locally, don't upload to HuggingFace"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results.json",
        help="Output path for local results JSON"
    )

    args = parser.parse_args()

    # Collect metadata
    print("Collecting system metadata...")
    metadata = collect_metadata()
    print(f"  FerroML: {metadata.ferroml_version}")
    print(f"  Rust: {metadata.rust_version}")
    print(f"  OS: {metadata.os} {metadata.os_version}")
    print(f"  CPU: {metadata.cpu}")

    results: list[BenchmarkResult] = []

    # Run or load FerroML benchmarks
    if not args.skip_benchmarks:
        print("\nRunning FerroML benchmarks...")
        output = run_cargo_bench()
        ferroml_results = parse_criterion_output(output)
        results.extend(ferroml_results)
        print(f"  Collected {len(ferroml_results)} FerroML benchmark results")

    # Load XGBoost/LightGBM results
    if not args.skip_xgboost:
        print("\nLoading XGBoost/LightGBM results...")
        xgb_lgb_results = load_xgboost_lightgbm_results()
        results.extend(xgb_lgb_results)
        print(f"  Loaded {len(xgb_lgb_results)} XGBoost/LightGBM results")

    if not results:
        print("\nNo benchmark results collected!")
        print("Either run the benchmarks or ensure existing results are available.")
        sys.exit(1)

    print(f"\nTotal benchmark results: {len(results)}")

    # Save results locally
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        "metadata": asdict(metadata),
        "results": [asdict(r) for r in results]
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved results to {output_path}")

    # Publish to HuggingFace Hub
    if not args.local_only:
        if not HAS_HF_HUB:
            print("\nSkipping HuggingFace upload (huggingface_hub not installed)")
        elif not HAS_PANDAS:
            print("\nSkipping HuggingFace upload (pandas not installed)")
        else:
            print(f"\nPublishing to HuggingFace Hub: {args.repo_id}")
            try:
                url = publish_to_huggingface(results, metadata, args.repo_id, args.private)
                print(f"\nBenchmark results published: {url}")
            except Exception as e:
                print(f"\nFailed to publish to HuggingFace Hub: {e}")
                print("Results are still saved locally.")
                sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
