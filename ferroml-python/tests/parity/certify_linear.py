"""Linear-family parity certification harness (PARITY-CONTRACT.md, layers 2 & 4).

Measures the maximum observed discrepancy between FerroML and scikit-learn on
the linear family over a preregistered matrix (seeds x shapes). The recorded
maxima are the raw material from which per-model atol/rtol tolerances are
DERIVED (epsilon_family = max observed + preregistered margin); no tolerance
in this file is chosen by hand.

Run:
    python -m tests.parity.certify_linear          # from ferroml-python/
Artifact:
    docs/test-runs/parity-certification-linear.json  (in repo root)
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import (
    ElasticNet as SkElasticNet,
    Lasso as SkLasso,
    LinearRegression as SkLinearRegression,
    LogisticRegression as SkLogisticRegression,
    Ridge as SkRidge,
)

from ferroml.linear import (
    ElasticNet as FeElasticNet,
    LassoRegression as FeLasso,
    LinearRegression as FeLinearRegression,
    LogisticRegression as FeLogisticRegression,
    RidgeRegression as FeRidge,
)

# --- Preregistered certification matrix -------------------------------------
SEEDS = [42, 7, 123, 2026]
SHAPES = [(200, 10), (500, 20), (1000, 50)]
DTYPE = np.float64
NOISE = 0.1

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Discrepancy:
    """Two-sided-form components for one output channel: we record both the
    max absolute difference and the max |reference| so atol/rtol can be fit
    to the measured cloud afterwards."""
    max_abs_diff: float = 0.0
    max_ref_abs: float = 0.0
    n_compared: int = 0
    where: str = ""

    def update(self, mine: np.ndarray, ref: np.ndarray, where: str) -> None:
        mine = np.asarray(mine, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        d = float(np.abs(mine - ref).max())
        if d > self.max_abs_diff:
            self.max_abs_diff = d
            self.where = where
        r = float(np.abs(ref).max())
        if r > self.max_ref_abs:
            self.max_ref_abs = r
        self.n_compared += mine.size


@dataclass
class ModelCert:
    name: str
    channels: dict = field(default_factory=dict)  # channel -> Discrepancy

    def chan(self, ch: str) -> Discrepancy:
        return self.channels.setdefault(ch, Discrepancy())


def _regression_data(seed: int, n: int, p: int):
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=p)
    X = rng.normal(size=(n, p)).astype(DTYPE)
    y = (X @ beta + 0.5 + rng.normal(scale=NOISE, size=n)).astype(DTYPE)
    return X, y


def _logistic_data(seed: int, n: int, p: int):
    """Labels drawn from a true Bernoulli-logistic response.

    Center-overlap generators become separable in moderate dimensions, and
    unregularized MLE has no finite optimum on separable data — IRLS and
    L-BFGS then diverge along different paths and any comparison is
    meaningless. A generative-label design keeps the MLE well posed.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(DTYPE)
    beta = rng.normal(size=p) * 0.5
    prob = 1.0 / (1.0 + np.exp(-(X @ beta + 0.3)))
    y = rng.binomial(1, prob).astype(np.int64)
    return X, y


def certify_pair(model_cert: ModelCert, tag: str, ferroml_model, sk_model,
                 Xtr, ytr, Xte, *, proba: bool = False) -> None:
    ferroml_model.fit(Xtr, ytr)
    sk_model.fit(Xtr, ytr)

    fm_coef = np.asarray(ferroml_model.coef_, dtype=np.float64).ravel()
    sk_coef = np.asarray(sk_model.coef_, dtype=np.float64).ravel()
    model_cert.chan("coef").update(fm_coef, sk_coef, tag)
    fm_int = np.asarray(ferroml_model.intercept_, dtype=np.float64).ravel()
    sk_int = np.asarray(sk_model.intercept_, dtype=np.float64).ravel()
    model_cert.chan("intercept").update(fm_int, sk_int, tag)

    fm_pred = np.asarray(ferroml_model.predict(Xte), dtype=np.float64)
    sk_pred = np.asarray(sk_model.predict(Xte), dtype=np.float64)
    model_cert.chan("predict").update(fm_pred, sk_pred, tag)

    if proba:
        fm_p = np.asarray(ferroml_model.predict_proba(Xte), dtype=np.float64)
        sk_p = np.asarray(sk_model.predict_proba(Xte), dtype=np.float64)
        model_cert.chan("predict_proba").update(fm_p, sk_p, tag)


def main() -> int:
    certs = {
        "LinearRegression": ModelCert("LinearRegression"),
        "RidgeRegression(alpha=1.0)": ModelCert("RidgeRegression(alpha=1.0)"),
        "LassoRegression(alpha=0.1)": ModelCert("LassoRegression(alpha=0.1)"),
        "ElasticNet(alpha=0.1,l1_ratio=0.5)": ModelCert("ElasticNet(alpha=0.1,l1_ratio=0.5)"),
        "LogisticRegression": ModelCert("LogisticRegression"),
    }

    for seed in SEEDS:
        for n, p in SHAPES:
            # --- regression models ---
            Xtr, ytr = _regression_data(seed, n, p)
            Xte, _ = _regression_data(seed + 1, max(50, n // 2), p)
            tag = f"seed={seed},shape={n}x{p}"

            pairs = [
                ("LinearRegression", lambda: FeLinearRegression(), SkLinearRegression),
                ("RidgeRegression(alpha=1.0)", lambda: FeRidge(alpha=1.0), lambda: SkRidge(alpha=1.0)),
                ("LassoRegression(alpha=0.1)", lambda: FeLasso(alpha=0.1), lambda: SkLasso(alpha=0.1)),
                ("ElasticNet(alpha=0.1,l1_ratio=0.5)",
                 lambda: FeElasticNet(alpha=0.1, l1_ratio=0.5),
                 lambda: SkElasticNet(alpha=0.1, l1_ratio=0.5)),
            ]
            for name, fmake, smake in pairs:
                certify_pair(certs[name], tag, fmake(), smake(), Xtr, ytr, Xte)

            # --- logistic: FerroML supports binary labels only (multiclass
            # raises ValueError), so the certification matrix is binary-only.
            Xc, yc = _logistic_data(seed, n, p)
            Xce, _ = _logistic_data(seed + 1, max(50, n // 2), p)
            # FerroML logistic = unregularized IRLS MLE (l2_penalty=0.0);
            # the matching sklearn objective is C=np.inf (penalty=None was
            # deprecated in sklearn 1.8). Comparing against sklearn's
            # regularized default would be a different-objective comparison.
            certify_pair(certs["LogisticRegression"], f"{tag},classes=2",
                         FeLogisticRegression(),
                         SkLogisticRegression(C=np.inf, max_iter=1000),
                         Xc, yc, Xce, proba=True)

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "ferroml": __import__("ferroml").__version__,
        "dtype": "float64",
        "threads_env": {k: os.environ.get(k, "") for k in
                        ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS")},
        "seeds": SEEDS,
        "shapes": SHAPES,
    }

    # Preregistered margin policy (PARITY-CONTRACT.md layer 4):
    # epsilon_channel = 10 x max observed discrepancy, floored at 1e-12.
    MARGIN_FACTOR, EPS_FLOOR = 10.0, 1e-12

    out = {"contract": "PARITY-CONTRACT.md layers 2+4 (certification run, not tolerances)",
           "margin_policy": {"factor": MARGIN_FACTOR, "floor": EPS_FLOOR,
                             "formula": "eps = max(factor*max_abs_diff, floor)"},
           "environment": env, "models": {}}
    print(f"{'model':<36} {'channel':<15} {'max |mine-ref|':>14}  {'worst case':<28}")
    for name, cert in certs.items():
        out["models"][name] = {
            ch: {"max_abs_diff": d.max_abs_diff, "max_ref_abs": d.max_ref_abs,
                 "n_compared": d.n_compared, "argmax_case": d.where,
                 "derived_epsilon": max(MARGIN_FACTOR * d.max_abs_diff, EPS_FLOOR)}
            for ch, d in sorted(cert.channels.items())
        }
        for ch, d in sorted(cert.channels.items()):
            eps = max(MARGIN_FACTOR * d.max_abs_diff, EPS_FLOOR)
            print(f"{name:<36} {ch:<15} {d.max_abs_diff:>14.3e}  "
                  f"derived_eps={eps:.3e}  {d.where}")

    dest = REPO_ROOT / "docs" / "test-runs" / "parity-certification-linear.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
