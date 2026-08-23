"""Clustering-family parity certification harness (PARITY-CONTRACT.md layers 2-4).

KMeans is NOT compared by raw output equality: label assignments carry an
arbitrary permutation symmetry, so the contract (layer 3) requires
invariant-aware checks.  This harness measures, over a preregistered matrix
(seeds x data configurations):

  * ``labels_fit_ari``     -- Adjusted Rand Index between FerroML and sklearn
                              fit-time labels (permutation invariant).
  * ``labels_predict_ari`` -- ARI between FerroML ``predict(X)`` assignments
                              and sklearn fit-time labels.
  * ``centroids_matched``  -- max |mine - ref| over OPTIMALLY MATCHED centroids
                              (scipy.optimize.linear_sum_assignment on the
                              pairwise distance matrix), plus max |ref| so a
                              two-sided atol/rtol can be fit afterwards.
  * ``inertia``            -- max |mine - ref| and max |ref| for the final
                              within-cluster sum of squares.

Recorded maxima feed the DERIVED tolerances exactly as in certify_linear.py:

    derived_epsilon(channel) = max(10 x max observed diff, 1e-12)
    ari_threshold            = max(0.0, min observed ARI - 0.01)

No tolerance in this file is chosen by hand beyond the preregistered margin
policy above.

Run:
    python -m tests.parity.certify_clustering      # from ferroml-python/
Artifact:
    docs/test-runs/parity-certification-clustering.json  (in repo root)
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from ferroml import __version__ as ferroml_version
from ferroml.clustering import KMeans as FeKMeans

# --- Preregistered certification matrix --------------------------------------
SEEDS = [42, 7, 123, 2026]
# (name, n_samples, n_clusters, n_features, cluster_std)
CONFIGS = [
    ("blobs-easy", 200, 4, 10, 1.0),
    ("blobs-medium", 500, 6, 10, 1.0),
    ("blobs-wide", 1000, 8, 20, 1.5),
    ("blobs-overlap", 500, 3, 5, 3.0),
]
DTYPE = np.float64
N_INIT = 10

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
class AriTracker:
    """Tracks the minimum ARI observed across the whole matrix."""
    min_ari: float = 1.0
    where: str = ""

    def update(self, ari: float, where: str) -> None:
        if ari < self.min_ari:
            self.min_ari = float(ari)
            self.where = where


@dataclass
class CertResult:
    cells: list = field(default_factory=list)
    centroids_matched: Discrepancy = field(default_factory=Discrepancy)
    inertia: Discrepancy = field(default_factory=Discrepancy)
    labels_fit_ari: AriTracker = field(default_factory=AriTracker)
    labels_predict_ari: AriTracker = field(default_factory=AriTracker)


def _cluster_data(seed: int, n_samples: int, n_features: int,
                  cluster_std: float, centers: int) -> np.ndarray:
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=seed,
    )
    return X.astype(DTYPE)


def match_centroids(C_ref: np.ndarray, C_mine: np.ndarray):
    """Optimal one-to-one matching of my centroids onto reference ones."""
    cost = np.linalg.norm(C_ref[:, None, :] - C_mine[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    return row, col  # ref index row[i] matches my index col[i]


def main() -> int:
    res = CertResult()
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "scipy": __import__("scipy").__version__,
        "ferroml": ferroml_version,
        "platform": platform.platform(),
        "dtype": "float64",
    }

    for seed in SEEDS:
        for (cfg_name, n, k, d, std) in CONFIGS:
            X = _cluster_data(seed, n, d, std, k)
            tag = f"{cfg_name}/seed={seed}"

            sk = SkKMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)
            fe = FeKMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)

            C_ref = np.asarray(sk.cluster_centers_, dtype=DTYPE)
            C_mine = np.asarray(fe.cluster_centers_, dtype=DTYPE)
            assert C_mine.shape == (k, d), f"{tag}: centers shape {C_mine.shape}"

            # --- layer-3 invariant: ARI on fit labels (permutation free) ------
            ari_fit = adjusted_rand_score(sk.labels_, np.asarray(fe.labels_))
            res.labels_fit_ari.update(ari_fit, f"{tag}:labels")

            # --- optimal centroid permutation ---------------------------------
            row, col = match_centroids(C_ref, C_mine)
            res.centroids_matched.update(C_mine[col], C_ref[row],
                                         f"{tag}:centroids")

            # --- inertia -------------------------------------------------------
            res.inertia.update(np.asarray([float(fe.inertia_)]),
                               np.asarray([float(sk.inertia_)]),
                               f"{tag}:inertia")

            # --- predict channel: ARI against reference fit labels -------------
            fe_pred = np.asarray(fe.predict(X))
            ari_pred = adjusted_rand_score(sk.labels_, fe_pred)
            res.labels_predict_ari.update(ari_pred, f"{tag}:predict")

            res.cells.append({
                "config": cfg_name, "seed": seed, "n": n, "k": k,
                "n_features": d, "cluster_std": std,
                "ari_fit": ari_fit,
                "ari_predict": ari_pred,
                "centroid_max_abs_diff_matched":
                    float(np.abs(C_mine[col] - C_ref[row]).max()),
                "inertia_abs_diff":
                    abs(float(fe.inertia_) - float(sk.inertia_)),
                "inertia_ref": float(sk.inertia_),
                "n_iter_mine": int(fe.n_iter_), "n_iter_ref": int(sk.n_iter_),
            })
            print(f"[{tag}] ari_fit={ari_fit:.6f} ari_pred={ari_pred:.6f} "
                  f"centroid_diff={res.cells[-1]['centroid_max_abs_diff_matched']:.3e} "
                  f"inertia_diff={res.cells[-1]['inertia_abs_diff']:.3e}")

    # --- Derived tolerances (preregistered policy) ---------------------------
    MARGIN = 10.0
    FLOOR = 1e-12
    out = {
        "environment": env,
        "matrix": {"seeds": SEEDS, "configs": [c[0] for c in CONFIGS],
                   "n_init": N_INIT},
        "policy": {
            "derived_epsilon": "max(10 * max_abs_diff, 1e-12)",
            "rtol": "derived_epsilon / max_ref_abs",
            "ari_threshold": "max(0, min_observed_ari - 0.01)",
        },
        "channels": {
            "centroids_matched": {
                "max_abs_diff": res.centroids_matched.max_abs_diff,
                "max_ref_abs": res.centroids_matched.max_ref_abs,
                "where": res.centroids_matched.where,
                "derived_epsilon": max(MARGIN * res.centroids_matched.max_abs_diff,
                                       FLOOR),
            },
            "inertia": {
                "max_abs_diff": res.inertia.max_abs_diff,
                "max_ref_abs": res.inertia.max_ref_abs,
                "where": res.inertia.where,
                "derived_epsilon": max(MARGIN * res.inertia.max_abs_diff, FLOOR),
            },
            "labels_fit_ari": {
                "min_observed": res.labels_fit_ari.min_ari,
                "where": res.labels_fit_ari.where,
                "threshold": max(0.0, res.labels_fit_ari.min_ari - 0.01),
            },
            "labels_predict_ari": {
                "min_observed": res.labels_predict_ari.min_ari,
                "where": res.labels_predict_ari.where,
                "threshold": max(0.0, res.labels_predict_ari.min_ari - 0.01),
            },
        },
        "cells": res.cells,
    }

    out_path = REPO_ROOT / "docs" / "test-runs" / \
        "parity-certification-clustering.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    print(f"centroids_matched derived_epsilon = "
          f"{out['channels']['centroids_matched']['derived_epsilon']:.3e}")
    print(f"inertia derived_epsilon           = "
          f"{out['channels']['inertia']['derived_epsilon']:.3e}")
    print(f"labels_fit_ari threshold          = "
          f"{out['channels']['labels_fit_ari']['threshold']}")
    print(f"labels_predict_ari threshold      = "
          f"{out['channels']['labels_predict_ari']['threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
