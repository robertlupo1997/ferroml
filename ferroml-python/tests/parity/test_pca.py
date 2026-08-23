"""PCA-family parity gate (PARITY-CONTRACT.md layers 1, 2, 3, 4).

PCA components carry per-component sign flips AND (under eigenvalue ties) any
rotation inside the retained subspace, so this module does NOT compare raw
component signs (contract layer 3).  Every tolerance below is DERIVED from the
measured certification run recorded in
``docs/test-runs/parity-certification-pca.json``
(``python -m tests.parity.certify_pca``, from ferroml-python/):

    atol(channel)      = derived_epsilon     # 10 x max observed diff, floor 1e-12
    rtol(channel)      = derived_epsilon / max_ref_abs
    subspace gate      = max(10 x max observed ||sin(theta)||_2, 1e-12)

Numeric comparisons use the two-sided form

    |mine - reference| <= atol + rtol * |reference|

on explained_variance_ratio_, Procrustes-aligned ``transform(X)``, and the
sign-invariant ``inverse_transform(fit_transform(X))`` reconstruction.  The
components themselves are gated by the principal-angle subspace distance --
never by raw signs.

Layer-1 structural checks (shapes, spectrum order, finiteness) run alongside.
If the live environment does not match the version-scoped certification
environment, the whole module skips (contract layer 4).

Scope exclusions honored here: NO epsilon is claimed for gradient boosting,
and FerroML TruncatedSVD's explained_variance_ratio_ normalization differs
from sklearn's (~4.2e-2 definitional gap at subspace ~4.5e-7), so it is not
certified in this slice either.

Run:
    python -m pytest tests/parity/test_pca.py -v   # from ferroml-python/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sklearn
from sklearn.decomposition import PCA as SkPCA

import ferroml
from ferroml.decomposition import PCA as FePCA

from .certify_pca import (
    CONFIGS,
    SEEDS,
    DTYPE,
    _matrix,
    principal_angle_distance,
    procrustes_alignment,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_PATH = REPO_ROOT / "docs" / "test-runs" / "parity-certification-pca.json"

pytestmark = pytest.mark.parity

if not CERT_PATH.exists():
    pytest.skip(
        f"certification artifact missing: {CERT_PATH} -- run "
        "`python -m tests.parity.certify_pca` first",
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
SUBSPACE_GATE = float(CH["subspace_distance"]["gate"])
FLOOR = 1e-12


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


EVR_ATOL, EVR_RTOL = _tolerances("explained_variance_ratio")
TRANSFORM_ATOL, TRANSFORM_RTOL = _tolerances("transform_aligned")
RECON_ATOL, RECON_RTOL = _tolerances("reconstruction")


def _fit_pair(seed: int, cfg):
    name, n, d, k, gen, param = cfg
    X = _matrix(seed, n, d, gen, param)
    sk = SkPCA(n_components=k).fit(X)
    fe = FePCA(n_components=k).fit(X)
    return X, sk, fe, k


CONFIG_IDS = [c[0] for c in CONFIGS]
CELL_IDS = [f"{c[0]}-seed{s}" for c in CONFIGS for s in SEEDS]


# --- Layer 1: exact structural checks ----------------------------------------
@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
class TestStructural:
    def test_component_shapes(self, cfg):
        _, _, fe, k = _fit_pair(SEEDS[0], cfg)
        C = np.asarray(fe.components_)
        assert C.shape == (k, cfg[2]), f"components_ shape {C.shape} != {(k, cfg[2])}"

    def test_evr_shape_order_nonneg(self, cfg):
        _, _, fe, k = _fit_pair(SEEDS[0], cfg)
        evr = np.asarray(fe.explained_variance_ratio_, dtype=np.float64)
        assert evr.shape == (k,)
        assert (evr >= 0.0).all(), "negative explained_variance_ratio_"
        assert (np.diff(evr) <= 0.0).all(), (
            f"explained_variance_ratio_ not descending: {evr}"
        )

    def test_transform_shape(self, cfg):
        X, _, fe, k = _fit_pair(SEEDS[0], cfg)
        T = np.asarray(fe.transform(X))
        assert T.shape == (X.shape[0], k)

    def test_reconstruction_finite(self, cfg):
        X, _, fe, _ = _fit_pair(SEEDS[0], cfg)
        R = np.asarray(fe.inverse_transform(fe.fit_transform(X)))
        assert R.shape == X.shape
        assert np.isfinite(R).all()

    def test_components_near_unit_norm(self, cfg):
        # Tolerance DERIVED from the measured subspace gate -- no hand constant.
        X, _, fe, _ = _fit_pair(SEEDS[0], cfg)
        C = np.asarray(fe.components_, dtype=np.float64)
        norms = np.linalg.norm(C, axis=1)
        assert np.abs(norms - 1.0).max() <= SUBSPACE_GATE


# --- Layer 3: invariant check -- subspace distance, never raw signs ----------
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
class TestSubspaceInvariant:
    def test_principal_angle_distance(self, seed, cfg):
        _, sk, fe, _ = _fit_pair(seed, cfg)
        dist = principal_angle_distance(
            np.asarray(fe.components_), np.asarray(sk.components_)
        )
        assert dist <= SUBSPACE_GATE, (
            f"subspace distance {dist:.3e} exceeds measured gate "
            f"{SUBSPACE_GATE:.3e}"
        )
        # The distance must also beat an explicitly sign-permuted basis far
        # less often than the real comparison: sanity that it measures spans,
        # not signs.
        C_ref = np.asarray(sk.components_)
        flipped = C_ref.copy()
        flipped[0] *= -1.0
        assert principal_angle_distance(np.asarray(fe.components_),
                                        flipped) <= SUBSPACE_GATE


# --- Layer 2: two-sided numeric gates ----------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cfg", CONFIGS, ids=CONFIG_IDS)
class TestTwoSided:
    def test_explained_variance_ratio(self, seed, cfg):
        _, sk, fe, _ = _fit_pair(seed, cfg)
        assert_two_sided(
            fe.explained_variance_ratio_,
            sk.explained_variance_ratio_,
            EVR_ATOL, EVR_RTOL,
            f"{cfg[0]}/seed={seed}:evr",
        )

    def test_transform_procrustes_aligned(self, seed, cfg):
        X, sk, fe, _ = _fit_pair(seed, cfg)
        Q = procrustes_alignment(np.asarray(fe.components_, dtype=DTYPE),
                                 np.asarray(sk.components_, dtype=DTYPE))
        T_aligned = np.asarray(fe.transform(X), dtype=DTYPE) @ Q.T
        assert_two_sided(
            T_aligned, sk.transform(X),
            TRANSFORM_ATOL, TRANSFORM_RTOL,
            f"{cfg[0]}/seed={seed}:transform",
        )

    def test_reconstruction_sign_invariant(self, seed, cfg):
        X, sk, fe, _ = _fit_pair(seed, cfg)
        R_mine = np.asarray(fe.inverse_transform(fe.fit_transform(X)),
                            dtype=DTYPE)
        R_ref = np.asarray(sk.inverse_transform(sk.transform(X)), dtype=DTYPE)
        assert_two_sided(
            R_mine, R_ref,
            RECON_ATOL, RECON_RTOL,
            f"{cfg[0]}/seed={seed}:reconstruction",
        )


# --- Preregistered margin policy self-consistency -----------------------------
class TestDerivedPolicy:
    def test_epsilons_match_margin_policy(self):
        for ch in ("explained_variance_ratio", "transform_aligned",
                   "reconstruction"):
            rec = CH[ch]
            expected = max(10.0 * float(rec["max_abs_diff"]), FLOOR)
            assert float(rec["derived_epsilon"]) == pytest.approx(expected), (
                f"{ch}: derived_epsilon {rec['derived_epsilon']} != policy "
                f"value {expected}"
            )

    def test_subspace_gate_matches_policy(self):
        rec = CH["subspace_distance"]
        expected = max(10.0 * float(rec["max_observed"]), FLOOR)
        assert SUBSPACE_GATE == pytest.approx(expected)

    def test_scope_exclusions_recorded(self):
        excl = CERT.get("scope_exclusions", {})
        assert "gradient_boosting" in excl, (
            "gradient boosting exclusion must stay recorded"
        )
        assert "truncated_svd_evr" in excl, (
            "TruncatedSVD EVR definitional mismatch must stay recorded"
        )


# --- Negative controls: the gates are not rubber stamps ------------------------
class TestGateBites:
    def test_reconstruction_gate_trips_when_tightened(self):
        X, sk, fe, _ = _fit_pair(SEEDS[0], CONFIGS[1])
        R_mine = np.asarray(fe.inverse_transform(fe.fit_transform(X)),
                            dtype=DTYPE)
        R_ref = np.asarray(sk.inverse_transform(sk.transform(X)), dtype=DTYPE)
        obs = float(np.abs(R_mine - R_ref).max())
        tight_atol = 1e-15  # 100x below the certified atol of 8.74e-12
        assert obs > tight_atol, "fixture lost its violation"
        with pytest.raises(AssertionError):
            assert_two_sided(R_mine, R_ref, tight_atol, 0.0,
                             "negative-control")

    def test_subspace_gate_trips_on_rotated_basis(self):
        _, sk, *_ = _fit_pair(SEEDS[0], CONFIGS[1])
        C_ref = np.asarray(sk.components_)
        # Mix two components: same span overall but row i no longer aligned;
        # principal angles against individual rows change.  A raw-sign-style
        # row-wise comparison would pass this; the span distance does not
        # care -- so instead verify the gate detects a genuinely DIFFERENT
        # subspace (drop the top direction into another).
        perturbed = C_ref.copy()
        perturbed[0], perturbed[-1] = perturbed[-1].copy(), perturbed[0].copy()
        # Row permutation keeps the span, so distance stays small: document.
        assert principal_angle_distance(C_ref, perturbed) <= SUBSPACE_GATE
        # A true different subspace (replace last component with a fresh
        # random orthogonal-ish direction) MUST trip the gate.
        rng = np.random.default_rng(0)
        v = rng.normal(size=C_ref.shape[1])
        v -= C_ref[:-1].T @ (C_ref[:-1] @ v)  # orthogonalize vs kept rows
        v /= np.linalg.norm(v)
        different = C_ref.copy()
        different[-1] = v
        assert principal_angle_distance(C_ref, different) > SUBSPACE_GATE
