"""Data loading and model persistence for the CLI."""
from __future__ import annotations

import pickle  # Used intentionally: ML ecosystem standard, only loads user's own models
import sys
from pathlib import Path

import numpy as np


def load_data(path: str, target: str | None = None) -> tuple:
    """Load CSV or Parquet file, split into X (and optionally y).

    Returns (X, y, feature_names) if target is given, else (X, None, feature_names).
    """
    import polars as pl

    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    if p.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif p.suffix in (".csv", ".tsv"):
        df = pl.read_csv(path)
    else:
        print(f"Error: unsupported file format: {p.suffix} (use .csv or .parquet)", file=sys.stderr)
        raise SystemExit(1)

    if target and target not in df.columns:
        print(f"Error: target column '{target}' not found. Columns: {df.columns}", file=sys.stderr)
        raise SystemExit(1)

    if target:
        y = df[target].to_numpy().astype(np.float64)
        X = df.drop(target).to_numpy().astype(np.float64)
        feature_names = [c for c in df.columns if c != target]
    else:
        X = df.to_numpy().astype(np.float64)
        y = None
        feature_names = df.columns

    return X, y, feature_names


def save_model(model, path: str) -> None:
    """Save a fitted model to disk via pickle.

    Note: FerroML models implement __getstate__/__setstate__ using MessagePack
    serialization under the hood. Pickle is used as the standard ML ecosystem
    interface (compatible with joblib, sklearn patterns).
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    """Load a fitted model from disk via pickle.

    Only loads models that were saved by this CLI — not for untrusted content.
    """
    p = Path(path)
    if not p.exists():
        print(f"Error: model file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301
