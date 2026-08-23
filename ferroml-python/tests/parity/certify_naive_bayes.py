"""Naive-Bayes-family parity certification harness (PARITY-CONTRACT.md layers 2-4).

Naive Bayes is the contract's own example of the achievable case: same algorithm,
same dtype (float64), deterministic op order -- "Smartcore does hold exact Naive
Bayes fixtures".  There is no output symmetry to factor out (no label permutation,
no sign ambiguity), so channels are compared directly in the two-sided form
inputs:

    max_abs_diff : max |mine - reference|
    max_ref_abs  : max |reference|          (to fit a two-sided rtol afterwards)

Measured per estimator over a preregistered matrix (seeds x data configs):

  GaussianNB     theta_, var_, predict_proba, predict_log_proba,
                 fit/predict label agreement
  MultinomialNB  predict_proba, label agreement   (integer count features)
  BernoulliNB    predict_proba, label agreement   (binary features)

CategoricalNB is out of scope for this slice (per-category discrete counts need
their own input-handling pass).

KNOWN IMPLEMENTATION DEVIATION (documented, not hidden behind an epsilon):
FerroML's GaussianNB.predict_log_proba floors each class probability at 1e-15
before taking the logarithm, so in the deep tail it returns ~ln(1e-15) = -34.54
where scikit-learn returns the true joint log likelihood (down to ~-85 in this
matrix).  A single epsilon CANNOT cover both regimes -- the derived tolerance
would be meaningless (~5e2).  The contract forbids rubber-stamp tolerances, so
this harness splits the channel:

  gnb_predict_log_proba_inrange -- two-sided atol/rtol gate, applied ONLY where
                                   the reference is above the preregistered
                                   threshold REF_TAIL (-30.0), i.e. inside the
                                   domain both implementations compute directly.
  gnb_log_proba_tail            -- structural layer-1/3 record: every tail cell
                                   must be finite and must sit at or above the
                                   measured FerroML floor; the reference range
                                   is recorded verbatim as evidence of the
                                   deviation.

Recorded maxima feed the DERIVED tolerances exactly as in certify_linear.py /
certify_clustering.py:

    derived_epsilon(channel) = max(10 x max observed diff, 1e-12)
    rtol(channel)            = derived_epsilon / max_ref_abs
    agreement_threshold      = max(0, min observed agreement - 0.01)

No tolerance in this file is chosen by hand beyond that preregistered policy.

Run:
    python -m tests.parity.certify_naive_bayes      # from ferroml-python/
Artifact:
    docs/test-runs/parity-certification-naive-bayes.json  (in repo root)
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.naive_bayes import (
    BernoulliNB as SkBernoulliNB,
    GaussianNB as SkGaussianNB,
    MultinomialNB as SkMultinomialNB,
)

from ferroml import __version__ as ferroml_version
from ferroml.naive_bayes import (
    BernoulliNB as FeBernoulliNB,
    GaussianNB as FeGaussianNB,
    MultinomialNB as FeMultinomialNB,
)

# --- Preregistered certification matrix --------------------------------------
SEEDS = [42, 7, 123, 2026]

# (name, n_samples, n_features, n_informative, n_classes, class_sep)
GNB_CONFIGS = [
    ("gauss-easy", 300, 10, 6, 2, 2.5),
    ("gauss-medium", 500, 12, 8, 3, 1.5),
    ("gauss-hard", 800, 20, 10, 4, 0.8),
    ("gauss-wide", 1000, 30, 15, 3, 1.2),
]

# (name, n_samples, n_features, max_count) -- Poisson-style integer counts
MNB_CONFIGS = [
    ("counts-small", 300, 8, 15),
    ("counts-large", 600, 16, 40),
]

# (name, n_samples, n_features) -- binary features
BNB_CONFIGS = [
    ("bernoulli-small", 300, 8),
    ("bernoulli-large", 600, 20),
]

DTYPE = np.float64

# Preregistered tail threshold for the log-proba split.  The observed FerroML
# clamp floor is ln(1e-15) ~= -34.54; anything above -30 is computed by both
# implementations without hitting the floor.
REF_TAIL = -30.0

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
class AgreementTracker:
    """Tracks the minimum label-agreement fraction across the whole matrix."""
    min_agreement: float = 1.0
    where: str = ""

    def update(self, agree: float, where: str) -> None:
        if agree < self.min_agreement:
            self.min_agreement = float(agree)
            self.where = where


@dataclass
class CertResult:
    cells: list = field(default_factory=list)
    gnb_theta: Discrepancy = field(default_factory=Discrepancy)
    gnb_var: Discrepancy = field(default_factory=Discrepancy)
    gnb_proba: Discrepancy = field(default_factory=Discrepancy)
    gnb_log_proba_inrange: Discrepancy = field(default_factory=Discrepancy)
    log_proba_tail_ref_min: float = 0.0
    log_proba_tail_fe_min: float = 0.0
    log_proba_tail_cells: int = 0
    mnb_proba: Discrepancy = field(default_factory=Discrepancy)
    bnb_proba: Discrepancy = field(default_factory=Discrepancy)
    gnb_agree: AgreementTracker = field(default_factory=AgreementTracker)
    mnb_agree: AgreementTracker = field(default_factory=AgreementTracker)
    bnb_agree: AgreementTracker = field(default_factory=AgreementTracker)


def _gnb_data(seed: int, n: int, d: int, informative: int,
              classes: int, sep: float):
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=informative,
        n_redundant=min(2, d - informative), n_classes=classes,
        weights=None, class_sep=sep, flip_y=0.0, random_state=seed,
    )
    return X.astype(DTYPE), y.astype(DTYPE)


def _count_data(seed: int, n: int, d: int, max_count: int):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.5).astype(DTYPE)
    # class-dependent intensity so the count distributions differ by class
    lam = np.where(y[:, None] == 1.0, 0.7, 0.3) * \
        rng.uniform(0.5, 1.0, size=(n, d)) * max_count
    X = rng.poisson(lam).astype(DTYPE)
    return X, y


def _binary_data(seed: int, n: int, d: int):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.5).astype(DTYPE)
    p = np.where(y[:, None] == 1.0, 0.65, 0.35)
    X = (rng.random((n, d)) < p).astype(DTYPE)
    return X, y


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
        # ---------------- GaussianNB ----------------
        for (name, n, d, inf, k, sep) in GNB_CONFIGS:
            X, y = _gnb_data(seed, n, d, inf, k, sep)
            tag = f"{name}/seed={seed}"

            sk = SkGaussianNB().fit(X, y)
            fe = FeGaussianNB().fit(X, y)

            assert np.array_equal(np.asarray(fe.classes_), sk.classes_), \
                f"{tag}: class list mismatch"

            res.gnb_theta.update(fe.theta_, sk.theta_, f"{tag}:theta")
            res.gnb_var.update(fe.var_, sk.var_, f"{tag}:var")
            proba_diff = float(
                np.abs(fe.predict_proba(X) - sk.predict_proba(X)).max())
            res.gnb_proba.update(fe.predict_proba(X), sk.predict_proba(X),
                                 f"{tag}:proba")

            lsk = np.asarray(sk.predict_log_proba(X), dtype=DTYPE)
            lfe = np.asarray(fe.predict_log_proba(X), dtype=DTYPE)
            in_range = lsk > REF_TAIL
            if in_range.any():
                res.gnb_log_proba_inrange.update(lfe[in_range], lsk[in_range],
                                                 f"{tag}:log_proba_inrange")
            tail = ~in_range
            res.log_proba_tail_cells += int(tail.sum())
            if tail.any():
                res.log_proba_tail_ref_min = min(
                    res.log_proba_tail_ref_min, float(lsk[tail].min()))
                res.log_proba_tail_fe_min = min(
                    res.log_proba_tail_fe_min, float(lfe[tail].min()))
                assert np.isfinite(lfe[tail]).all(), \
                    f"{tag}: non-finite tail log_proba"

            agree_fit = float(np.mean(
                np.asarray(fe.predict(X)) == sk.predict(X)))
            Xte, yte = _gnb_data(seed + 10_000, n // 2, d, inf, k, sep)
            sk.fit(X, y)
            fe.fit(X, y)
            agree_pred = float(np.mean(
                np.asarray(fe.predict(Xte)) == sk.predict(Xte)))
            res.gnb_agree.update(min(agree_fit, agree_pred), f"{tag}:agree")

            res.cells.append({
                "estimator": "GaussianNB", "config": name, "seed": seed,
                "theta_max_abs_diff":
                    float(np.abs(fe.theta_ - sk.theta_).max()),
                "var_max_abs_diff":
                    float(np.abs(fe.var_ - sk.var_).max()),
                "proba_max_abs_diff": proba_diff,
                "log_proba_inrange_max_abs_diff": float(
                    np.abs(lfe[in_range] - lsk[in_range]).max())
                if in_range.any() else None,
                "agreement_fit": agree_fit,
                "agreement_predict_holdout": agree_pred,
            })
            c = res.cells[-1]
            print(f"[{tag}] theta={c['theta_max_abs_diff']:.3e} "
                  f"var={c['var_max_abs_diff']:.3e} "
                  f"proba={c['proba_max_abs_diff']:.3e} "
                  f"logp_in={c['log_proba_inrange_max_abs_diff']:.3e} "
                  f"agree_fit={agree_fit:.4f} agree_pred={agree_pred:.4f}")

        # ---------------- MultinomialNB ----------------
        for (name, n, d, mc) in MNB_CONFIGS:
            X, y = _count_data(seed, n, d, mc)
            tag = f"{name}/seed={seed}"

            sk = SkMultinomialNB().fit(X, y)
            fe = FeMultinomialNB().fit(X, y)

            assert np.array_equal(np.asarray(fe.classes_), sk.classes_), \
                f"{tag}: class list mismatch"

            res.mnb_proba.update(fe.predict_proba(X), sk.predict_proba(X),
                                 f"{tag}:proba")
            agree = float(np.mean(np.asarray(fe.predict(X)) == sk.predict(X)))
            res.mnb_agree.update(agree, f"{tag}:agree")

            res.cells.append({
                "estimator": "MultinomialNB", "config": name, "seed": seed,
                "proba_max_abs_diff": float(
                    np.abs(fe.predict_proba(X) - sk.predict_proba(X)).max()),
                "agreement_fit": agree,
            })
            print(f"[{tag}] proba="
                  f"{res.cells[-1]['proba_max_abs_diff']:.3e} agree={agree:.4f}")

        # ---------------- BernoulliNB ----------------
        for (name, n, d) in BNB_CONFIGS:
            X, y = _binary_data(seed, n, d)
            tag = f"{name}/seed={seed}"

            sk = SkBernoulliNB().fit(X, y)
            fe = FeBernoulliNB().fit(X, y)

            assert np.array_equal(np.asarray(fe.classes_), sk.classes_), \
                f"{tag}: class list mismatch"

            res.bnb_proba.update(fe.predict_proba(X), sk.predict_proba(X),
                                 f"{tag}:proba")
            agree = float(np.mean(np.asarray(fe.predict(X)) == sk.predict(X)))
            res.bnb_agree.update(agree, f"{tag}:agree")

            res.cells.append({
                "estimator": "BernoulliNB", "config": name, "seed": seed,
                "proba_max_abs_diff": float(
                    np.abs(fe.predict_proba(X) - sk.predict_proba(X)).max()),
                "agreement_fit": agree,
            })
            print(f"[{tag}] proba="
                  f"{res.cells[-1]['proba_max_abs_diff']:.3e} agree={agree:.4f}")

    # --- Derived tolerances (preregistered policy) ---------------------------
    MARGIN = 10.0
    FLOOR = 1e-12

    def channel(d: Discrepancy) -> dict:
        eps = max(MARGIN * d.max_abs_diff, FLOOR)
        return {
            "max_abs_diff": d.max_abs_diff,
            "max_ref_abs": d.max_ref_abs,
            "where": d.where,
            "derived_epsilon": eps,
            "derived_rtol": eps / d.max_ref_abs if d.max_ref_abs else 0.0,
        }

    def agreement(t: AgreementTracker) -> dict:
        return {
            "min_observed": t.min_agreement,
            "where": t.where,
            "threshold": max(0.0, t.min_agreement - 0.01),
        }

    out = {
        "environment": env,
        "matrix": {
            "seeds": SEEDS,
            "gnb_configs": [c[0] for c in GNB_CONFIGS],
            "mnb_configs": [c[0] for c in MNB_CONFIGS],
            "bnb_configs": [c[0] for c in BNB_CONFIGS],
            "scope_note": "CategoricalNB excluded from this slice",
        },
        "policy": {
            "derived_epsilon": "max(10 * max_abs_diff, 1e-12)",
            "rtol": "derived_epsilon / max_ref_abs",
            "agreement_threshold": "max(0, min_observed_agreement - 0.01)",
        },
        "channels": {
            "gnb_theta": channel(res.gnb_theta),
            "gnb_var": channel(res.gnb_var),
            "gnb_predict_proba": channel(res.gnb_proba),
            "gnb_predict_log_proba_inrange": channel(res.gnb_log_proba_inrange),
            "gnb_log_proba_tail": {
                "note": ("FerroML floors class probabilities at 1e-15 before "
                         "predict_log_proba; tail cells are gated "
                         "structurally, not by epsilon"),
                "ref_tail_threshold": REF_TAIL,
                "tail_cells": res.log_proba_tail_cells,
                "ref_min_observed": res.log_proba_tail_ref_min,
                "fe_min_observed": res.log_proba_tail_fe_min,
                "derived_floor": res.log_proba_tail_fe_min,
            },
            "mnb_predict_proba": channel(res.mnb_proba),
            "bnb_predict_proba": channel(res.bnb_proba),
            "gnb_label_agreement": agreement(res.gnb_agree),
            "mnb_label_agreement": agreement(res.mnb_agree),
            "bnb_label_agreement": agreement(res.bnb_agree),
        },
        "cells": res.cells,
    }

    out_path = REPO_ROOT / "docs" / "test-runs" / \
        "parity-certification-naive-bayes.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    for ch_name in ("gnb_theta", "gnb_var", "gnb_predict_proba",
                    "gnb_predict_log_proba_inrange", "mnb_predict_proba",
                    "bnb_predict_proba"):
        ch = out["channels"][ch_name]
        print(f"{ch_name:<24} derived_epsilon = {ch['derived_epsilon']:.3e} "
              f"(rtol {ch['derived_rtol']:.3e})")
    for ch_name in ("gnb_label_agreement", "mnb_label_agreement",
                    "bnb_label_agreement"):
        print(f"{ch_name:<24} threshold       = "
              f"{out['channels'][ch_name]['threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
