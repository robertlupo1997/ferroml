"""Clustering-family parity gate (PARITY-CONTRACT.md layers 1, 2, 3, 4).

KMeans outputs carry an arbitrary label-permutation symmetry, so this module
does NOT compare raw assignments (contract layer 3).  Every tolerance below is
DERIVED from the measured certification run recorded in
``docs/test-runs/parity-certification-clustering.json``
(``python -m tests.parity.certify_clustering``, from ferroml-python/):

    atol(channel)   = derived_epsilon            # 10 x max observed diff, floor 1e-12
    rtol(channel)   = derived_epsilon / max_ref_abs
    ARI gates       = max(0, min observed ARI - 0.01)

Numeric comparisons use the two-sided form:

    |mine - reference| <= atol + rtol * |reference|

applied to OPTIMALLY MATCHED centroids (Hungarian assignment on the pairwise
centroid distances) and to inertia.  Label quality uses the permutation
invariant Adjusted Rand Index for both ``labels_`` and ``predict(X)``.

Layer-1 structural checks (label sets, center shapes, inertia finiteness) run
alongside.  If the live environment does not match the version-scoped
certification environment, the whole module skips (contract layer 4).

Gradient boosting is a declared scope exclusion: NO parity epsilon is claimed
for it anywhere in this suite.

Run:
    python -m pytest tests/parity/test_clustering.py -v   # from ferroml-python/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

import ferroml
from ferroml.clustering import KMeans as FeKMeans

from .certify_clustering import CONFIGS, SEEDS, _cluster_data, match_centroids

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_PATH = REPO_ROOT / "docs" / "test-runs" / "parity-certification-clustering.json"

pytestmark = pytest.mark.parity

if not CERT_PATH.exists():
    pytest.skip(
        f"certification artifact missing: {CERT_PATH} -- run "
        "`python -m tests.parity.certify_clustering` first",
        allow_module_level=True,
    )

CERT = json.loads(CERT_PATH.read_text())

# --- Layer 4: tolerances are scoped to the certified environment -------------
_env = CERT["environment"]
if (_env["sklearn"] != sklearn.__version__
        or _env["numpy"] != np.__version__
        or _env["ferroml"] != ferroml.__version__):
    pytest.skip(
        "live environment "
        f"(sklearn {sklearn.__version__}, numpy {np.__version__}, "
        f"ferroml {ferroml.__version__}) does not match the certified "
        f"environment (sklearn {_env['sklearn']}, numpy {_env['numpy']}, "
        f"ferroml {_env['ferroml']}); tolerances are not valid outside the "
        "certified matrix",
        allow_module_level=True,
    )

CH = CERT["channels"]


def _tolerances(channel: str) -> tuple[float, float]:
    """Derive (atol, rtol) for one channel from the measured certification run."""
    atol = float(CH[channel]["derived_epsilon"])  # margin already applied
    denom = float(CH[channel]["max_ref_abs"])
    rtol = atol / denom if denom > 0.0 else 0.0
    return atol, rtol


def assert_two_sided(mine, ref, atol: float, rtol: float, label: str) -> None:
    """Contract layer 2: |mine - ref| <= atol + rtol * |ref|, elementwise."""
    mine = np.asarray(mine, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert mine.shape == ref.shape, f"{label}: shape {mine.shape} != {ref.shape}"
    slack = atol + rtol * np.abs(ref)
    violation = np.abs(mine - ref) - slack
    if violation.size and violation.max() > 0.0:
        idx = int(np.argmax(violation.ravel()))
        raise AssertionError(
            f"{label}: |mine-ref|={np.abs(mine - ref).ravel()[idx]:.3e} exceeds "
            f"atol={atol:.3e} + rtol={rtol:.3e}*|ref| at flat index {idx}"
        )


ARI_FIT_GATE = float(CH["labels_fit_ari"]["threshold"])
ARI_PREDICT_GATE = float(CH["labels_predict_ari"]["threshold"])
CENT_ATOL, CENT_RTOL = _tolerances("centroids_matched")
INERTIA_ATOL, INERTIA_RTOL = _tolerances("inertia")


def _fit_pair(seed: int, cfg):
    name, n, k, d, std = cfg
    X = _cluster_data(seed, n, d, std, k)
    sk = SkKMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    fe = FeKMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return X, sk, fe, k


CONFIG_IDS = [f"{c[0]}-seed{s}" for c in CONFIGS for s in SEEDS]


# --- Layer 1: exact structural checks ----------------------------------------
@pytest.mark.parametrize("cfg", CONFIGS, ids=[c[0] for c in CONFIGS])
class TestStructural:
    def test_label_set_and_shapes(self, cfg):
        _, sk, fe, k = _fit_pair(SEEDS[0], cfg)
        X_shape = fe.labels_.shape
        assert len(fe.labels_) == cfg[1] * 0 + int(np.prod(X_shape))
        assert set(np.unique(fe.labels_)) == set(range(k))
        assert fe.cluster_centers_.shape == (k, cfg[3])
        assert sk.cluster_centers_.shape == fe.cluster_centers_.shape

    def test_inertia_finite_nonnegative(self, cfg):
        _, _, fe, _ = _fit_pair(SEEDS[0], cfg)
        assert np.isfinite(float(fe.inertia_))
        assert float(fe.inertia_) >= 0.0

    def test_predict_shape(self, cfg):
        X, _, fe, k = _fit_pair(SEEDS[0], cfg)
        pred = np.asarray(fe.predict(X))
        assert pred.shape == (X.shape[0],)
        assert set(np.unique(pred)).issubset(set(range(k)))


# --- Layers 2 & 3: invariants with derived tolerances -------------------------
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cfg", CONFIGS, ids=[c[0] for c in CONFIGS])
class TestParityInvariants:
    def test_labels_fit_ari(self, seed, cfg):
        """Layer 3: ARI is permutation invariant; gate from measured minimum."""
        _, sk, fe, _ = _fit_pair(seed, cfg)
        ari = adjusted_rand_score(sk.labels_, np.asarray(fe.labels_))
        assert ari >= ARI_FIT_GATE, (
            f"fit-labels ARI={ari:.6f} < certified gate {ARI_FIT_GATE}")

    def test_labels_predict_ari(self, seed, cfg):
        _, sk, fe, _ = _fit_pair(seed, cfg)
        pred = np.asarray(fe.predict(X := np.asarray(
            _cluster_data(seed, cfg[1], cfg[3], cfg[4], cfg[2]),
            dtype=np.float64)))
        ari = adjusted_rand_score(sk.labels_, pred)
        assert ari >= ARI_PREDICT_GATE, (
            f"predict ARI={ari:.6f} < certified gate {ARI_PREDICT_GATE}")

    def test_centroids_matched_two_sided(self, seed, cfg):
        """Optimal label permutation via Hungarian matching, then two-sided."""
        _, sk, fe, k = _fit_pair(seed, cfg)
        C_ref = np.asarray(sk.cluster_centers_, dtype=np.float64)
        C_mine = np.asarray(fe.cluster_centers_, dtype=np.float64)
        row, col = match_centroids(C_ref, C_mine)
        # Matching must be a full one-to-one bijection over k clusters.
        assert sorted(row.tolist()) == list(range(k))
        assert sorted(col.tolist()) == list(range(k))
        assert_two_sided(C_mine[col], C_ref[row], CENT_ATOL, CENT_RTOL,
                         f"{cfg[0]}/seed={seed}:centroids-matched")

    def test_inertia_two_sided(self, seed, cfg):
        _, sk, fe, _ = _fit_pair(seed, cfg)
        assert_two_sided([float(fe.inertia_)], [float(sk.inertia_)],
                         INERTIA_ATOL, INERTIA_RTOL,
                         f"{cfg[0]}/seed={seed}:inertia")


# --- Policy self-consistency (guards against hand-edited artifacts) ----------
class TestDerivedPolicy:
    def test_epsilon_matches_preregistered_margin(self):
        for ch_name in ("centroids_matched", "inertia"):
            rec = CH[ch_name]
            expected = max(10.0 * float(rec["max_abs_diff"]), 1e-12)
            assert float(rec["derived_epsilon"]) == pytest.approx(expected)

    def test_ari_gates_match_measured_minimum(self):
        for ch_name in ("labels_fit_ari", "labels_predict_ari"):
            rec = CH[ch_name]
            expected = max(0.0, float(rec["min_observed"]) - 0.01)
            assert float(rec["threshold"]) == pytest.approx(expected)

    def test_gate_is_not_a_rubber_stamp(self):
        """Negative control: tightening atol by 100x must fail on real data."""
        _, sk, fe, k = _fit_pair(SEEDS[0], CONFIGS[1])
        C_ref = np.asarray(sk.cluster_centers_, dtype=np.float64)
        C_mine = np.asarray(fe.cluster_centers_, dtype=np.float64)
        row, col = match_centroids(C_ref, C_mine)
        # Observed diffs are ULP-level (~5e-15); atol/1000 (=1e-15) with no
        # relative slack must be violated by them.
        assert np.abs(C_mine[col] - C_ref[row]).max() > CENT_ATOL / 1000.0
        with pytest.raises(AssertionError):
            assert_two_sided(C_mine[col], C_ref[row],
                             CENT_ATOL / 1000.0, 0.0,
                             "negative-control")
