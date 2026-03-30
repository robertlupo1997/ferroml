"""Capture a complete reproducibility snapshot of an ML experiment.

Usage: Claude runs this after training to record everything needed to reproduce results.
Output: JSON snapshot with model, data, metrics, environment, and diff capabilities.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def snapshot(
    model: object,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    metrics: dict[str, float],
    seed: int = 42,
    notes: str = "",
) -> dict:
    """Capture a full reproducibility snapshot.

    Parameters
    ----------
    model : object
        Fitted FerroML model.
    model_name : str
        Human-readable model identifier.
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training targets.
    metrics : dict
        Evaluation metrics (e.g. {"r2": 0.95, "rmse": 1.2}).
    seed : int
        Random seed used for training.
    notes : str
        Optional experiment notes.

    Returns
    -------
    dict — complete reproducibility snapshot
    """
    # Data fingerprint
    x_hash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
    y_hash = hashlib.sha256(y.tobytes()).hexdigest()[:16]

    # Feature statistics
    feature_stats = []
    for j in range(X.shape[1]):
        col = X[:, j]
        col_clean = col[np.isfinite(col)]
        if len(col_clean) > 0:
            feature_stats.append({
                "index": j,
                "mean": round(float(np.mean(col_clean)), 6),
                "std": round(float(np.std(col_clean)), 6),
                "min": round(float(np.min(col_clean)), 6),
                "max": round(float(np.max(col_clean)), 6),
                "nan_count": int(np.isnan(col).sum()),
            })
        else:
            feature_stats.append({
                "index": j,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "nan_count": int(len(col)),
            })

    # Model parameters
    model_params = {}
    try:
        card = model.model_card()
        if isinstance(card, dict):
            model_params = card
    except Exception:
        pass

    if not model_params:
        try:
            space = model.search_space()
            if isinstance(space, dict):
                model_params = {"search_space": space}
        except Exception:
            model_params = {"class": type(model).__name__}

    # Environment
    ferroml_version = _get_ferroml_version()
    numpy_version = np.__version__

    snap = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_class": type(model).__name__,
        "model_params": model_params,
        "data": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "x_hash": x_hash,
            "y_hash": y_hash,
            "target_mean": round(float(np.nanmean(y)), 6),
            "target_std": round(float(np.nanstd(y)), 6),
        },
        "feature_stats": feature_stats,
        "metrics": {k: round(float(v), 8) for k, v in metrics.items()},
        "seed": seed,
        "environment": {
            "ferroml_version": ferroml_version,
            "python_version": sys.version,
            "numpy_version": numpy_version,
            "platform": sys.platform,
        },
        "notes": notes,
    }

    return snap


def save_snapshot(snap: dict, path: str) -> str:
    """Write snapshot to JSON file.

    Parameters
    ----------
    snap : dict
        Snapshot from snapshot().
    path : str
        Output file path.

    Returns
    -------
    str — path written
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(snap, f, indent=2, default=str)
    return str(out)


def load_snapshot(path: str) -> dict:
    """Load a snapshot from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_snapshots(snap1: dict, snap2: dict) -> dict:
    """Compare two snapshots and report differences.

    Parameters
    ----------
    snap1 : dict
        First snapshot (typically the baseline).
    snap2 : dict
        Second snapshot (typically the new experiment).

    Returns
    -------
    dict with differences by category
    """
    diffs: dict[str, list[dict]] = {
        "data": [],
        "metrics": [],
        "model": [],
        "environment": [],
    }

    # Data differences
    d1 = snap1.get("data", {})
    d2 = snap2.get("data", {})
    for key in ["n_samples", "n_features", "x_hash", "y_hash"]:
        if d1.get(key) != d2.get(key):
            diffs["data"].append({
                "field": key,
                "snap1": d1.get(key),
                "snap2": d2.get(key),
            })

    # Metric differences
    m1 = snap1.get("metrics", {})
    m2 = snap2.get("metrics", {})
    all_metric_keys = set(m1.keys()) | set(m2.keys())
    for key in sorted(all_metric_keys):
        v1 = m1.get(key)
        v2 = m2.get(key)
        if v1 != v2:
            delta = None
            if v1 is not None and v2 is not None:
                try:
                    delta = round(float(v2) - float(v1), 8)
                except (TypeError, ValueError):
                    pass
            diffs["metrics"].append({
                "metric": key,
                "snap1": v1,
                "snap2": v2,
                "delta": delta,
            })

    # Model differences
    if snap1.get("model_name") != snap2.get("model_name"):
        diffs["model"].append({
            "field": "model_name",
            "snap1": snap1.get("model_name"),
            "snap2": snap2.get("model_name"),
        })
    if snap1.get("model_class") != snap2.get("model_class"):
        diffs["model"].append({
            "field": "model_class",
            "snap1": snap1.get("model_class"),
            "snap2": snap2.get("model_class"),
        })
    if snap1.get("seed") != snap2.get("seed"):
        diffs["model"].append({
            "field": "seed",
            "snap1": snap1.get("seed"),
            "snap2": snap2.get("seed"),
        })

    # Environment differences
    e1 = snap1.get("environment", {})
    e2 = snap2.get("environment", {})
    for key in ["ferroml_version", "python_version", "numpy_version"]:
        if e1.get(key) != e2.get(key):
            diffs["environment"].append({
                "field": key,
                "snap1": e1.get(key),
                "snap2": e2.get(key),
            })

    # Summary
    total_diffs = sum(len(v) for v in diffs.values())
    data_same = len(diffs["data"]) == 0
    reproducible = data_same and len(diffs["metrics"]) == 0 and len(diffs["model"]) == 0

    return {
        "diffs": diffs,
        "total_differences": total_diffs,
        "data_identical": data_same,
        "fully_reproducible": reproducible,
        "summary": (
            "Experiments are identical" if reproducible
            else f"{total_diffs} difference(s) found across {sum(1 for v in diffs.values() if v)} categories"
        ),
    }


def _get_ferroml_version() -> str:
    """Get installed ferroml version."""
    try:
        import ferroml
        return getattr(ferroml, "__version__", "unknown")
    except ImportError:
        return "not installed"


def print_snapshot(snap: dict) -> None:
    """Print human-readable snapshot summary."""
    print(f"\n{'='*60}")
    print(f"REPRODUCIBILITY SNAPSHOT")
    print(f"{'='*60}")
    print(f"  Timestamp:  {snap['timestamp']}")
    print(f"  Model:      {snap['model_name']} ({snap['model_class']})")
    print(f"  Data:       {snap['data']['n_samples']} samples x {snap['data']['n_features']} features")
    print(f"  Data hash:  X={snap['data']['x_hash']}  y={snap['data']['y_hash']}")
    print(f"  Seed:       {snap['seed']}")
    print(f"\n  Metrics:")
    for k, v in snap["metrics"].items():
        print(f"    {k}: {v}")
    print(f"\n  Environment:")
    for k, v in snap["environment"].items():
        print(f"    {k}: {v}")
    if snap.get("notes"):
        print(f"\n  Notes: {snap['notes']}")
    print()


def print_comparison(result: dict) -> None:
    """Print human-readable snapshot comparison."""
    print(f"\n{'='*60}")
    print(f"SNAPSHOT COMPARISON  |  {result['summary']}")
    print(f"{'='*60}")

    for category, items in result["diffs"].items():
        if not items:
            continue
        print(f"\n  {category.upper()}:")
        for item in items:
            if "delta" in item and item["delta"] is not None:
                print(f"    {item.get('metric', item.get('field'))}: "
                      f"{item['snap1']} -> {item['snap2']} (delta={item['delta']:+})")
            else:
                field = item.get("metric", item.get("field"))
                print(f"    {field}: {item['snap1']} -> {item['snap2']}")
    print()
