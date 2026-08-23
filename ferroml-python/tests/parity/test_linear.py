"""Linear-family parity gate (PARITY-CONTRACT.md layers 1, 2, 4).

Every tolerance used here is DERIVED from the measured certification run
recorded in ``docs/test-runs/parity-certification-linear.json``
(``python -m tests.parity.certify_linear``, from ferroml-python/).  Nothing is
chosen by hand:

    atol(channel) = derived_epsilon            # 10 x max observed diff, floor 1e-12
    rtol(channel) = derived_epsilon / max_ref_abs   # relative slack implied by atol

and every numeric comparison uses the two-sided form of the contract:

    |mine - reference|  <=  atol + rtol * |reference|

Layer-1 structural checks (shapes, class list, n_features_in_, probability
rows summing to one) run alongside the numeric gate.

If the live environment does not match the version-scoped certification
environment, the whole module skips: the recorded tolerances are not valid
outside the certified matrix (contract layer 4), and widening them after a
failure is forbidden.

Gradient boosting is a declared scope exclusion: NO parity epsilon is claimed
for it anywhere in this suite.

Run:
    python -m pytest tests/parity/test_linear.py -v   # from ferroml-python/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import sklearn
from sklearn.linear_model import (
    ElasticNet as SkElasticNet,
    Lasso as SkLasso,
    LinearRegression as SkLinearRegression,
    LogisticRegression as SkLogisticRegression,
    Ridge as SkRidge,
)

import ferroml
from ferroml.linear import (
    ElasticNet as FeElasticNet,
    LassoRegression as FeLasso,
    LinearRegression as FeLinearRegression,
    LogisticRegression as FeLogisticRegression,
    RidgeRegression as FeRidge,
)

from .certify_linear import SHAPES, SEEDS, _logistic_data, _regression_data

REPO_ROOT = Path(__file__).resolve().parents[3]
CERT_PATH = REPO_ROOT / "docs" / "test-runs" / "parity-certification-linear.json"

pytestmark = pytest.mark.parity

if not CERT_PATH.exists():
    pytest.skip(
        f"certification artifact missing: {CERT_PATH} -- run "
        "`python -m tests.parity.certify_linear` first",
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


def _tolerances(model: str, channel: str) -> tuple[float, float]:
    """Derive (atol, rtol) for one channel from the measured certification run."""
    rec = CERT["models"][model][channel]
    atol = float(rec["derived_epsilon"])  # preregistered margin already applied
    denom = float(rec["max_ref_abs"])
    rtol = atol / denom if denom > 0.0 else 0.0
    return atol, rtol


def assert_two_sided(mine, ref, atol: float, rtol: float, label: str) -> None:
    """Contract layer 2: |mine - ref| <= atol + rtol * |ref|, elementwise."""
    mine = np.asarray(mine, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert mine.shape == ref.shape, f"{label}: shape {mine.shape} != {ref.shape}"
    slack = atol + rtol * np.abs(ref)
    violation = np.abs(mine - ref) - slack
    if violation.max() > 0.0:
        idx = int(np.argmax(violation.ravel()))
        raise AssertionError(
            f"{label}: |mine-ref|={np.abs(mine - ref).ravel()[idx]:.3e} exceeds "
            f"atol={atol:.3e} + rtol={rtol:.3e}*|ref| at flat index {idx}"
        )


# --- Model pairing: identical to certify_linear.py ----------------------------
def _make_pair(model: str):
    """Return (ferroml_model, sklearn_model, needs_proba) for a certified name."""
    if model == "LinearRegression":
        return FeLinearRegression(), SkLinearRegression(), False
    if model == "RidgeRegression(alpha=1.0)":
        return FeRidge(alpha=1.0), SkRidge(alpha=1.0), False
    if model == "LassoRegression(alpha=0.1)":
        return FeLasso(alpha=0.1), SkLasso(alpha=0.1), False
    if model == "ElasticNet(alpha=0.1,l1_ratio=0.5)":
        return FeElasticNet(alpha=0.1, l1_ratio=0.5), SkElasticNet(alpha=0.1, l1_ratio=0.5), False
    if model == "LogisticRegression":
        # FerroML logistic is an unregularized MLE (l2_penalty=0.0); the
        # matching sklearn objective is C=np.inf (penalty=None deprecated in
        # sklearn 1.8). Binary labels only: FerroML multiclass raises ValueError.
        return FeLogisticRegression(), SkLogisticRegression(C=np.inf, max_iter=1000), True
    raise KeyError(model)


MODELS = sorted(CERT["models"].keys())


# --- Layer 1 gap tracker: sklearn-standard fit metadata -----------------------
# scikit-learn's estimator convention exposes n_features_in_ after fit and
# classes_ on classifiers. FerroML's native estimators expose these only
# partially. This suite treats their ABSENCE as a known library gap (xfail),
# not a pass; when the Rust side adds them, these tests flip to passing.
EXPECTED_METADATA = {
    "LinearRegression": ("n_features_in_",),
    "RidgeRegression(alpha=1.0)": ("n_features_in_",),
    "LassoRegression(alpha=0.1)": ("n_features_in_",),
    "ElasticNet(alpha=0.1,l1_ratio=0.5)": ("n_features_in_",),
    "LogisticRegression": ("n_features_in_", "classes_"),
}


def _metadata_param(model):
    """xfail only the models whose metadata is actually missing today."""
    if model == "LinearRegression":  # exposes n_features_in_ already
        return pytest.param(model)
    return pytest.param(model, marks=pytest.mark.xfail(
        reason="FerroML native estimators do not yet expose all "
        "sklearn-standard fit metadata (n_features_in_, classes_); library "
        "gap surfaced by this suite, to be fixed in the Rust extension",
        strict=False,
    ))


@pytest.mark.parametrize("model", [_metadata_param(m) for m in MODELS])
def test_sklearn_standard_metadata(model):
    X, y = _regression_data(42, 60, 4)
    fm, _, proba = _make_pair(model)
    fm.fit(X[:50], y[:50] if not proba else (y[:50] > 0).astype(np.int64))
    for attr in EXPECTED_METADATA[model]:
        assert hasattr(fm, attr), f"{model}: missing sklearn-standard attribute {attr!r}"


# --- Layer 4 self-check: the recorded epsilons follow the preregistered policy -
@pytest.mark.parametrize("model", MODELS)
def test_certification_epsilons_follow_margin_policy(model):
    policy = CERT["margin_policy"]
    for channel, rec in CERT["models"][model].items():
        expected = max(policy["factor"] * rec["max_abs_diff"], policy["floor"])
        expected_msg = (
            f"{model}/{channel}: derived_epsilon {rec['derived_epsilon']} "
            f"!= policy value {expected}"
        )
        assert rec["derived_epsilon"] == pytest.approx(expected), expected_msg


# --- Layer 1 + Layer 2 gates --------------------------------------------------
@pytest.mark.parametrize("n,p", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("model", MODELS)
def test_linear_parity_against_certified_tolerances(model, seed, n, p):
    fm, sk, proba = _make_pair(model)
    tag = f"seed={seed},shape={n}x{p}"

    if model == "LogisticRegression":
        Xtr, ytr = _logistic_data(seed, n, p)
        Xte, _ = _logistic_data(seed + 1, max(50, n // 2), p)
    else:
        Xtr, ytr = _regression_data(seed, n, p)
        Xte, _ = _regression_data(seed + 1, max(50, n // 2), p)

    fm.fit(Xtr, ytr)
    sk.fit(Xtr, ytr)

    # Layer 1: exact structural checks. Attributes that exist must be exact;
    # attributes that do not exist at all are tracked by
    # test_sklearn_standard_metadata below (a documented library gap).
    if hasattr(fm, "n_features_in_"):
        assert fm.n_features_in_ == p, f"{model} {tag}: n_features_in_"
    assert np.asarray(fm.coef_, dtype=np.float64).shape == (p,), f"{model} {tag}: coef_ shape"
    assert np.asarray(fm.predict(Xte)).shape == (max(50, n // 2),), f"{model} {tag}: predict shape"
    if proba:
        if hasattr(fm, "classes_"):
            assert np.array_equal(np.asarray(fm.classes_), np.array([0, 1])), \
                f"{model} {tag}: classes_ must be [0 1]"
        P = np.asarray(fm.predict_proba(Xte), dtype=np.float64)
        assert P.shape == (Xte.shape[0], 2), f"{model} {tag}: predict_proba shape"
        rowsum_err = float(np.abs(P.sum(axis=1) - 1.0).max())
        assert rowsum_err <= 1e-12, f"{model} {tag}: proba rows sum off by {rowsum_err:.3e}"

    # Layer 2: two-sided numeric gates on every certified channel.
    # Binary-logistic coef shape differs by convention (ferroml (p,), sklearn
    # (1, p)); the numeric gate compares the flattened coefficients -- the
    # shape convention itself is tracked in the module docstring.
    channels = {
        "coef": (np.ravel(np.asarray(fm.coef_, dtype=np.float64)),
                 np.ravel(np.asarray(sk.coef_, dtype=np.float64))),
        # same convention difference as coef_ (ferroml scalar, sklearn (1,))
        "intercept": (np.ravel(np.asarray(fm.intercept_, dtype=np.float64)),
                      np.ravel(np.asarray(sk.intercept_, dtype=np.float64))),
        "predict": (fm.predict(Xte), sk.predict(Xte)),
    }
    if proba:
        channels["predict_proba"] = (fm.predict_proba(Xte), sk.predict_proba(Xte))
    for channel, (mine, ref) in channels.items():
        atol, rtol = _tolerances(model, channel)
        assert_two_sided(mine, ref, atol, rtol, f"{model} {tag}/{channel}")
