"""Naive-Bayes-family parity gate (PARITY-CONTRACT.md layers 1, 2, 3, 4).

Naive Bayes has no output symmetry (no label permutation, no sign ambiguity),
so channels compare directly.  Every tolerance below is DERIVED from the
measured certification run recorded in
``docs/test-runs/parity-certification-naive-bayes.json``
(``python -m tests.parity.certify_naive_bayes``, from ferroml-python/):

    atol(channel) = derived_epsilon            # 10 x max observed diff, floor 1e-12
    rtol(channel) = derived_epsilon / max_ref_abs
    agreement     >= max(0, min observed agreement - 0.01)

Numeric comparisons use the two-sided form:

    |mine - reference| <= atol + rtol * |reference|

KNOWN DEVIATION, gated structurally -- not hidden behind an epsilon:
FerroML's GaussianNB.predict_log_proba floors class probabilities at 1e-15
before the logarithm, so deep-tail cells return ~ln(1e-15) ~= -34.54 while
scikit-learn returns the true joint log likelihood (~-85 in the certified
matrix).  A single epsilon there would be a rubber stamp (~5e2).  Instead:

  * in-range cells (reference > REF_TAIL = -30) get a real two-sided gate;
  * tail cells are checked structurally: all finite, and at or above the
    measured FerroML floor recorded in the certification artifact. This
    coverage is ONE-SIDED: tail values wrong upward are invisible to it
    (see the artifact's ``notes.tail_coverage``).

Layer-1 structural checks (shapes, class lists, finiteness) run alongside.
If the live environment does not match the version-scoped certification
environment, the whole module skips (contract layer 4).

Run:
    python -m pytest tests/parity/test_naive_bayes.py -v   # from ferroml-python/
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy
import sklearn

import ferroml
from ferroml.naive_bayes import (
    BernoulliNB as FeBernoulliNB,
    GaussianNB as FeGaussianNB,
    MultinomialNB as FeMultinomialNB,
)
from sklearn.naive_bayes import (
    BernoulliNB as SkBernoulliNB,
    GaussianNB as SkGaussianNB,
    MultinomialNB as SkMultinomialNB,
)

from .certify_naive_bayes import (
    BNB_CONFIGS,
    GNB_CONFIGS,
    MNB_CONFIGS,
    REF_TAIL,
    SEEDS,
    _binary_data,
    _count_data,
    _gnb_data,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_PATH = REPO_ROOT / "docs" / "test-runs" / "parity-certification-naive-bayes.json"

pytestmark = pytest.mark.parity

if not CERT_PATH.exists():
    pytest.skip(
        f"certification artifact missing: {CERT_PATH} -- run "
        "`python -m tests.parity.certify_naive_bayes` first",
        allow_module_level=True,
    )

CERT = json.loads(CERT_PATH.read_text())

# --- Layer 4: tolerances are scoped to the certified environment -------------
# The WHOLE recorded environment is load-bearing: interpreter version, the
# BLAS-linked numpy/scipy builds, the platform's floating-point behaviour, and
# the dtype all feed the measured max-diffs the tolerances were derived from,
# so every field the artifact records is compared, not just package versions.
_env = CERT["environment"]
_live = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "sklearn": sklearn.__version__,
    "scipy": scipy.__version__,
    "ferroml": ferroml.__version__,
    "platform": platform.platform(),
    "dtype": "float64",
}
if _live != _env:
    _mismatched = ", ".join(
        f"{k} (certified {_env[k]!r}, live {_live.get(k)!r})"
        for k in _env
        if _env[k] != _live.get(k)
    )
    pytest.skip(
        f"live environment does not match the certified environment; "
        f"differing fields: {_mismatched}. Tolerances are not valid "
        "outside the certified matrix",
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


THETA_ATOL, THETA_RTOL = _tolerances("gnb_theta")
VAR_ATOL, VAR_RTOL = _tolerances("gnb_var")
GNB_PROBA_ATOL, GNB_PROBA_RTOL = _tolerances("gnb_predict_proba")
GNB_LOGP_ATOL, GNB_LOGP_RTOL = _tolerances("gnb_predict_log_proba_inrange")
MNB_PROBA_ATOL, MNB_PROBA_RTOL = _tolerances("mnb_predict_proba")
BNB_PROBA_ATOL, BNB_PROBA_RTOL = _tolerances("bnb_predict_proba")

GNB_AGREE_GATE = float(CH["gnb_label_agreement"]["threshold"])
MNB_AGREE_GATE = float(CH["mnb_label_agreement"]["threshold"])
BNB_AGREE_GATE = float(CH["bnb_label_agreement"]["threshold"])

TAIL = CH["gnb_log_proba_tail"]
TAIL_FLOOR = float(TAIL["derived_floor"])

GNB_IDS = [f"{c[0]}-seed{s}" for c in GNB_CONFIGS for s in SEEDS]


def _fit_gnb(seed: int, cfg):
    X, y = _gnb_data(seed, *cfg[1:])
    sk = SkGaussianNB().fit(X, y)
    fe = FeGaussianNB().fit(X, y)
    return X, y, sk, fe


# --- Layer 1: exact structural checks ----------------------------------------
class TestStructural:
    def test_gnb_class_list_and_shapes(self):
        X, y, sk, fe = _fit_gnb(SEEDS[0], GNB_CONFIGS[1])
        assert np.array_equal(np.asarray(fe.classes_), sk.classes_)
        assert fe.theta_.shape == sk.theta_.shape
        assert fe.var_.shape == sk.var_.shape

    def test_gnb_outputs_finite(self):
        X, _, _, fe = _fit_gnb(SEEDS[0], GNB_CONFIGS[2])
        assert np.isfinite(fe.theta_).all()
        assert np.isfinite(fe.var_).all()
        assert (np.asarray(fe.var_) > 0.0).all()
        proba = np.asarray(fe.predict_proba(X))
        assert np.isfinite(proba).all()
        assert ((proba >= 0.0) & (proba <= 1.0)).all()

    def test_mnb_class_list_and_proba_simplex(self):
        X, y = _count_data(SEEDS[0], *MNB_CONFIGS[1][1:])
        sk = SkMultinomialNB().fit(X, y)
        fe = FeMultinomialNB().fit(X, y)
        assert np.array_equal(np.asarray(fe.classes_), sk.classes_)
        proba = np.asarray(fe.predict_proba(X))
        assert np.isfinite(proba).all()
        assert ((proba >= 0.0) & (proba <= 1.0)).all()
        assert np.abs(proba.sum(axis=1) - 1.0).max() < 1e-9

    def test_bnb_class_list_and_proba_simplex(self):
        X, y = _binary_data(SEEDS[0], *BNB_CONFIGS[1][1:])
        sk = SkBernoulliNB().fit(X, y)
        fe = FeBernoulliNB().fit(X, y)
        assert np.array_equal(np.asarray(fe.classes_), sk.classes_)
        proba = np.asarray(fe.predict_proba(X))
        assert np.isfinite(proba).all()
        assert np.abs(proba.sum(axis=1) - 1.0).max() < 1e-9


# --- Layers 2 & 3: GaussianNB with derived tolerances -------------------------
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cfg", GNB_CONFIGS, ids=[c[0] for c in GNB_CONFIGS])
class TestGaussianNBParity:
    def test_theta_two_sided(self, seed, cfg):
        """Per-class means were EXACT across the whole certified matrix."""
        X, y, sk, fe = _fit_gnb(seed, cfg)
        assert_two_sided(fe.theta_, sk.theta_, THETA_ATOL, THETA_RTOL,
                         f"{cfg[0]}/seed={seed}:theta")

    def test_var_two_sided(self, seed, cfg):
        X, y, sk, fe = _fit_gnb(seed, cfg)
        assert_two_sided(fe.var_, sk.var_, VAR_ATOL, VAR_RTOL,
                         f"{cfg[0]}/seed={seed}:var")

    def test_predict_proba_two_sided(self, seed, cfg):
        X, y, sk, fe = _fit_gnb(seed, cfg)
        assert_two_sided(fe.predict_proba(X), sk.predict_proba(X),
                         GNB_PROBA_ATOL, GNB_PROBA_RTOL,
                         f"{cfg[0]}/seed={seed}:proba")

    def test_predict_log_proba_inrange_two_sided(self, seed, cfg):
        """Two-sided gate on cells BOTH implementations compute directly."""
        X, y, sk, fe = _fit_gnb(seed, cfg)
        lsk = np.asarray(sk.predict_log_proba(X), dtype=np.float64)
        lfe = np.asarray(fe.predict_log_proba(X), dtype=np.float64)
        in_range = lsk > REF_TAIL
        assert in_range.any(), "no in-range log-proba cells"
        assert_two_sided(lfe[in_range], lsk[in_range],
                         GNB_LOGP_ATOL, GNB_LOGP_RTOL,
                         f"{cfg[0]}/seed={seed}:log_proba_inrange")

    def test_predict_log_proba_tail_structural(self, seed, cfg):
        """Tail deviation is real and bounded by FerroML's measured floor.

        This documents the KNOWN DEVIATION instead of pretending an epsilon
        covers it: wherever sklearn goes below REF_TAIL, FerroML must stay
        finite and at or above its measured clamp floor.
        """
        X, _, _, fe = _fit_gnb(seed, cfg)
        lfe = np.asarray(fe.predict_log_proba(X), dtype=np.float64)
        assert np.isfinite(lfe).all()
        assert (lfe >= TAIL_FLOOR - 1e-6).all(), (
            f"log_proba below the certified FerroML floor {TAIL_FLOOR}")

    def test_label_agreement_fit_and_holdout(self, seed, cfg):
        """Layer 3: classification agreement, both in-sample and holdout."""
        X, y, sk, fe = _fit_gnb(seed, cfg)
        agree_fit = float(np.mean(
            np.asarray(fe.predict(X)) == sk.predict(X)))
        assert agree_fit >= GNB_AGREE_GATE
        n, d, n_inf, k, sep = cfg[1:]
        Xte, _ = _gnb_data(seed + 10_000, n // 2, d, n_inf, k, sep)
        agree_pred = float(np.mean(
            np.asarray(fe.predict(Xte)) == sk.predict(Xte)))
        assert agree_pred >= GNB_AGREE_GATE, (
            f"holdout agreement={agree_pred:.4f} < {GNB_AGREE_GATE}")


# --- Layers 2 & 3: MultinomialNB / BernoulliNB --------------------------------
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cfg", MNB_CONFIGS, ids=[c[0] for c in MNB_CONFIGS])
class TestMultinomialNBParity:
    def test_predict_proba_two_sided(self, seed, cfg):
        X, y = _count_data(seed, *cfg[1:])
        sk = SkMultinomialNB().fit(X, y)
        fe = FeMultinomialNB().fit(X, y)
        assert_two_sided(fe.predict_proba(X), sk.predict_proba(X),
                         MNB_PROBA_ATOL, MNB_PROBA_RTOL,
                         f"{cfg[0]}/seed={seed}:proba")

    def test_label_agreement(self, seed, cfg):
        X, y = _count_data(seed, *cfg[1:])
        sk = SkMultinomialNB().fit(X, y)
        fe = FeMultinomialNB().fit(X, y)
        agree = float(np.mean(np.asarray(fe.predict(X)) == sk.predict(X)))
        assert agree >= MNB_AGREE_GATE


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cfg", BNB_CONFIGS, ids=[c[0] for c in BNB_CONFIGS])
class TestBernoulliNBParity:
    def test_predict_proba_two_sided(self, seed, cfg):
        X, y = _binary_data(seed, *cfg[1:])
        sk = SkBernoulliNB().fit(X, y)
        fe = FeBernoulliNB().fit(X, y)
        assert_two_sided(fe.predict_proba(X), sk.predict_proba(X),
                         BNB_PROBA_ATOL, BNB_PROBA_RTOL,
                         f"{cfg[0]}/seed={seed}:proba")

    def test_label_agreement(self, seed, cfg):
        X, y = _binary_data(seed, *cfg[1:])
        sk = SkBernoulliNB().fit(X, y)
        fe = FeBernoulliNB().fit(X, y)
        agree = float(np.mean(np.asarray(fe.predict(X)) == sk.predict(X)))
        assert agree >= BNB_AGREE_GATE


# --- Policy self-consistency (guards against hand-edited artifacts) ----------
class TestDerivedPolicy:
    def test_epsilon_matches_preregistered_margin(self):
        for ch_name in ("gnb_theta", "gnb_var", "gnb_predict_proba",
                        "gnb_predict_log_proba_inrange", "mnb_predict_proba",
                        "bnb_predict_proba"):
            rec = CH[ch_name]
            expected = max(10.0 * float(rec["max_abs_diff"]), 1e-12)
            assert float(rec["derived_epsilon"]) == pytest.approx(expected)

    def test_rtol_matches_measured_ratio(self):
        for ch_name in ("gnb_var", "gnb_predict_proba",
                        "gnb_predict_log_proba_inrange"):
            rec = CH[ch_name]
            eps = float(rec["derived_epsilon"])
            denom = float(rec["max_ref_abs"])
            if denom > 0.0:
                assert float(rec["derived_rtol"]) == pytest.approx(eps / denom)

    def test_agreement_gates_match_measured_minimum(self):
        for ch_name in ("gnb_label_agreement", "mnb_label_agreement",
                        "bnb_label_agreement"):
            rec = CH[ch_name]
            expected = max(0.0, float(rec["min_observed"]) - 0.01)
            assert float(rec["threshold"]) == pytest.approx(expected)

    def test_gate_is_not_a_rubber_stamp(self):
        """Negative controls: tightened tolerances must fail on real data."""
        X, y, sk, fe = _fit_gnb(SEEDS[0], GNB_CONFIGS[3])
        # var diffs are ~3.4e-08; tightening atol 1000x must violate them.
        var_diff = float(np.abs(fe.var_ - sk.var_).max())
        assert var_diff > VAR_ATOL / 1000.0
        with pytest.raises(AssertionError):
            assert_two_sided(fe.var_, sk.var_, VAR_ATOL / 1000.0, 0.0,
                             "negative-control-var")
        # proba diffs are ~2.5e-10 here; same for the probability channel.
        p_diff = float(np.abs(
            np.asarray(fe.predict_proba(X)) - sk.predict_proba(X)).max())
        assert p_diff > GNB_PROBA_ATOL / 1000.0
        with pytest.raises(AssertionError):
            assert_two_sided(fe.predict_proba(X), sk.predict_proba(X),
                             GNB_PROBA_ATOL / 1000.0, 0.0,
                             "negative-control-proba")
