"""PCA-family parity certification harness (PARITY-CONTRACT.md layers 2-4).

PCA components carry two arbitrary symmetries -- per-component sign flips AND
(when eigenvalues are degenerate) any rotation inside the retained subspace --
so the contract (layer 3) forbids raw component-sign comparison.  This harness
measures, over a preregistered matrix (seeds x data configurations):

  * ``subspace_distance`` -- ||sin(theta)||_2 over the principal angles between
    the span of FerroML ``components_`` rows and the span of sklearn's
    (sign/permutation invariant).  Gate = max(10 x max observed, 1e-12).
  * ``transform_aligned`` -- max |mine - ref| on ``transform(X)`` after the
    orthogonal Procrustes alignment Q (from svd of ref @ mine.T) that maps my
    basis onto the reference one.  Sign symmetry handled exactly here.
  * ``reconstruction``    -- max |mine - ref| on
    ``inverse_transform(fit_transform(X))`` (naturally invariant to both
    symmetries).
  * ``explained_variance_ratio_`` -- max |mine - ref| elementwise (ratios are
    sign/permutation free because they follow the descending spectrum).

Recorded maxima feed the DERIVED tolerances exactly as in certify_linear.py /
certify_clustering.py:

    derived_epsilon(channel)   = max(10 x max observed diff, 1e-12)
    subspace_gate              = max(10 x max observed distance, 1e-12)

No tolerance in this file is chosen by hand beyond the preregistered margin
policy above.

Known scope note (recorded, not certified here): FerroML TruncatedSVD matches
sklearn's arpack solution at subspace distance ~4.5e-7, but its
``explained_variance_ratio_`` differs from sklearn's by ~4.2e-2 on the probe
matrix -- a definitional mismatch (different normalization base), not a
tolerance question.  It is therefore NOT given a parity epsilon in this slice;
fixing/clarifying the ratio convention belongs to the Rust-side follow-up.

Run:
    python -m tests.parity.certify_pca      # from ferroml-python/
Artifact:
    docs/test-runs/parity-certification-pca.json  (in repo root)
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_blobs, make_low_rank_matrix
from sklearn.decomposition import PCA as SkPCA

from ferroml import __version__ as ferroml_version
from ferroml.decomposition import PCA as FePCA

# --- Preregistered certification matrix --------------------------------------
SEEDS = [42, 7, 123, 2026]
# (name, n_samples, n_features, n_components, generator, param)
CONFIGS = [
    ("blobs-easy", 200, 10, 4, "blobs", 1.0),
    ("blobs-medium", 500, 10, 6, "blobs", 1.0),
    ("blobs-wide", 1000, 20, 8, "blobs", 1.5),
    ("lowrank-noise", 500, 20, 5, "lowrank", 0.0),
]
DTYPE = np.float64

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Discrepancy:
    """Two-sided-form components for one output channel."""
    max_abs_diff: float = 0.0
    max_ref_abs: float = 0.0
    where: str = ""

    def update(self, mine: np.ndarray, ref: np.ndarray, where: str) -> None:
        mine = np.asarray(mine, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        assert mine.shape == ref.shape, f"shape mismatch at {where}"
        d = float(np.abs(mine - ref).max())
        if d > self.max_abs_diff:
            self.max_abs_diff = d
            self.where = where
        r = float(np.abs(ref).max())
        if r > self.max_ref_abs:
            self.max_ref_abs = r


@dataclass
class MaxTracker:
    """Tracks the maximum scalar observed across the whole matrix."""
    max_value: float = 0.0
    where: str = ""

    def update(self, value: float, where: str) -> None:
        if value > self.max_value:
            self.max_value = float(value)
            self.where = where


@dataclass
class CertResult:
    cells: list = field(default_factory=list)
    evr: Discrepancy = field(default_factory=Discrepancy)
    transform_aligned: Discrepancy = field(default_factory=Discrepancy)
    reconstruction: Discrepancy = field(default_factory=Discrepancy)
    subspace_distance: MaxTracker = field(default_factory=MaxTracker)


def _matrix(seed: int, n_samples: int, n_features: int,
            generator: str, param: float) -> np.ndarray:
    if generator == "blobs":
        X, _ = make_blobs(
            n_samples=n_samples,
            centers=max(2, n_features // 2),
            n_features=n_features,
            cluster_std=param,
            random_state=seed,
        )
        return X.astype(DTYPE)
    if generator == "lowrank":
        # Deterministic low-rank signal plus seeded noise.
        base_seed = seed % (2**32)
        rng = np.random.RandomState(base_seed)
        signal = make_low_rank_matrix(
            n_samples=n_samples, n_features=n_features,
            effective_rank=int(param) if param else 5,
            random_state=base_seed,
        )
        noise = rng.normal(scale=0.05, size=signal.shape)
        return (signal + noise).astype(DTYPE)
    raise ValueError(f"unknown generator {generator}")


def principal_angle_distance(A: np.ndarray, B: np.ndarray) -> float:
    """||sin(theta)||_2 between the row spans of A (k x d) and B (k x d).

    Invariant to per-row signs and to any orthogonal mixing within either
    span -- exactly the symmetry the contract says clustering-like checks must
    respect for PCA.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    assert A.shape == B.shape
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    theta = np.arccos(np.clip(s, -1.0, 1.0))
    return float(np.linalg.norm(np.sin(theta)))


def procrustes_alignment(C_mine: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    """Orthogonal Q (k x k) minimizing ||Q @ C_mine - C_ref||_F.

    Q = V U^T where U S Vt = svd(C_ref @ C_mine.T).  Rows of Q @ C_mine then
    correspond one-to-one (same order) with C_ref rows, signs included.
    """
    U, _, Vt = np.linalg.svd(C_ref @ C_mine.T)
    return Vt.T @ U.T


def main() -> int:
    res = CertResult()
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "ferroml": ferroml_version,
        "platform": platform.platform(),
        "dtype": "float64",
    }

    for seed in SEEDS:
        for (cfg_name, n, d, k, gen, param) in CONFIGS:
            X = _matrix(seed, n, d, gen, param)
            tag = f"{cfg_name}/seed={seed}"

            sk = SkPCA(n_components=k).fit(X)
            fe = FePCA(n_components=k).fit(X)

            C_ref = np.asarray(sk.components_, dtype=DTYPE)
            C_mine = np.asarray(fe.components_, dtype=DTYPE)
            assert C_mine.shape == (k, d), f"{tag}: components shape {C_mine.shape}"

            # --- layer-3 invariant: subspace distance (sign/permutation free)
            dist = principal_angle_distance(C_mine, C_ref)
            res.subspace_distance.update(dist, f"{tag}:components")

            # --- Procrustes-aligned component comparison ----------------------
            Q = procrustes_alignment(C_mine, C_ref)

            # --- explained variance ratios (spectrum order, symmetry free) ----
            evr_mine = np.asarray(fe.explained_variance_ratio_, dtype=DTYPE)
            evr_ref = np.asarray(sk.explained_variance_ratio_, dtype=DTYPE)
            assert evr_mine.shape == (k,)
            res.evr.update(evr_mine, evr_ref, f"{tag}:evr")

            # --- transform channel, aligned through the same rotation ---------
            T_mine = np.asarray(fe.transform(X), dtype=DTYPE)
            T_ref = np.asarray(sk.transform(X), dtype=DTYPE)
            T_aligned = T_mine @ Q.T  # columns now correspond 1:1 with T_ref
            res.transform_aligned.update(T_aligned, T_ref, f"{tag}:transform")

            # --- reconstruction (invariant by construction) --------------------
            R_mine = np.asarray(fe.inverse_transform(fe.fit_transform(X)),
                                dtype=DTYPE)
            R_ref = np.asarray(sk.inverse_transform(sk.transform(X)), dtype=DTYPE)
            res.reconstruction.update(R_mine, R_ref, f"{tag}:reconstruction")

            res.cells.append({
                "config": cfg_name, "seed": seed, "n": n, "k": k,
                "n_features": d, "generator": gen,
                "subspace_distance": dist,
                "evr_max_abs_diff":
                    float(np.abs(evr_mine - evr_ref).max()),
                "transform_aligned_max_abs_diff":
                    float(np.abs(T_aligned - T_ref).max()),
                "reconstruction_max_abs_diff":
                    float(np.abs(R_mine - R_ref).max()),
                "evr_ref_max": float(np.abs(evr_ref).max()),
            })
            c = res.cells[-1]
            print(f"[{tag}] sub_dist={dist:.3e} evr={c['evr_max_abs_diff']:.3e} "
                  f"transform={c['transform_aligned_max_abs_diff']:.3e} "
                  f"recon={c['reconstruction_max_abs_diff']:.3e}")

    # --- Derived tolerances (preregistered policy) ---------------------------
    MARGIN = 10.0
    FLOOR = 1e-12
    out = {
        "environment": env,
        "matrix": {"seeds": SEEDS,
                   "configs": [c[0] for c in CONFIGS]},
        "policy": {
            "derived_epsilon": "max(10 * max_abs_diff, 1e-12)",
            "rtol": "derived_epsilon / max_ref_abs",
            "subspace_gate": "max(10 * max_observed_subspace_distance, 1e-12)",
        },
        "channels": {
            "explained_variance_ratio": {
                "max_abs_diff": res.evr.max_abs_diff,
                "max_ref_abs": res.evr.max_ref_abs,
                "where": res.evr.where,
                "derived_epsilon": max(MARGIN * res.evr.max_abs_diff, FLOOR),
            },
            "transform_aligned": {
                "max_abs_diff": res.transform_aligned.max_abs_diff,
                "max_ref_abs": res.transform_aligned.max_ref_abs,
                "where": res.transform_aligned.where,
                "derived_epsilon":
                    max(MARGIN * res.transform_aligned.max_abs_diff, FLOOR),
            },
            "reconstruction": {
                "max_abs_diff": res.reconstruction.max_abs_diff,
                "max_ref_abs": res.reconstruction.max_ref_abs,
                "where": res.reconstruction.where,
                "derived_epsilon":
                    max(MARGIN * res.reconstruction.max_abs_diff, FLOOR),
            },
            "subspace_distance": {
                "max_observed": res.subspace_distance.max_value,
                "where": res.subspace_distance.where,
                "gate": max(MARGIN * res.subspace_distance.max_value, FLOOR),
            },
        },
        "scope_exclusions": {
            "truncated_svd_evr":
                "FerroML TruncatedSVD explained_variance_ratio_ uses a "
                "different normalization than sklearn (observed ~4.2e-2 "
                "discrepancy at subspace distance ~4.5e-7); definitional "
                "mismatch, no parity epsilon claimed",
            "gradient_boosting":
                "declared scope exclusion: algorithms differ from sklearn's",
        },
        "cells": res.cells,
    }
    path = REPO_ROOT / "docs" / "test-runs" / "parity-certification-pca.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")
    for ch, v in out["channels"].items():
        key = "gate" if "gate" in v else "derived_epsilon"
        obs = v.get("max_observed", v.get("max_abs_diff", 0.0))
        print(f"  {ch}: {key}={v[key]:.3e} (max observed={obs:.3e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
