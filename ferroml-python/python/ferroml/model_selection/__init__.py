"""FerroML Model Selection -- sklearn-compatible entry point.

Provides train/test splitting, cross-validation splitters, scoring utilities,
and hyperparameter optimization in one convenient module.

Functions
---------
train_test_split(X, y, test_size=0.25, shuffle=True, random_state=None, stratify=None)
    Split arrays into random train and test subsets.
cross_val_score(model, X, y, cv=5, scoring="accuracy")
    Evaluate a model with cross-validation, returning per-fold scores.

Classes
-------
KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut
    Cross-validation splitters.
RepeatedKFold, ShuffleSplit, GroupKFold, LeavePOut
    Additional cross-validation splitters.
GridSearchCV, RandomSearchCV
    Hyperparameter search with cross-validation.
"""

from ferroml.ferroml import model_selection as _ms

# train_test_split from native
train_test_split = _ms.train_test_split

# Re-export cross-validation splitters
from ferroml.cv import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    LeaveOneOut,
    RepeatedKFold,
    ShuffleSplit,
    GroupKFold,
    LeavePOut,
    cross_val_score,
)

import time
import numpy as np

# Re-export HPO classes
from ferroml.hpo import GridSearchCV, RandomSearchCV

# Re-export learning/validation curves
from ferroml.cv import learning_curve, validation_curve, _score


def cross_validate(model, X, y, cv=5, scoring="accuracy", return_train_score=False):
    """Evaluate a model with cross-validation, returning a dict of results.

    Parameters
    ----------
    model : estimator
        A FerroML model with fit() and predict() methods.
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target array.
    cv : int, optional (default=5)
        Number of cross-validation folds.
    scoring : str, optional (default="accuracy")
        Scoring metric: "accuracy", "mse", "mae", or "r2".
    return_train_score : bool, optional (default=False)
        Whether to include training scores.

    Returns
    -------
    dict
        Keys: test_score (list), fit_time (list), score_time (list).
        If return_train_score=True, also includes train_score (list).
    """
    kf = KFold(n_folds=cv)
    splits = kf.split(X)

    test_scores = []
    train_scores = []
    fit_times = []
    score_times = []

    for train_idx, test_idx in splits:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        m = model.__class__()

        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        fit_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        test_scores.append(_score(m, X_te, y_te, scoring))
        score_times.append(time.perf_counter() - t0)

        if return_train_score:
            train_scores.append(_score(m, X_tr, y_tr, scoring))

    result = {
        "test_score": test_scores,
        "fit_time": fit_times,
        "score_time": score_times,
    }
    if return_train_score:
        result["train_score"] = train_scores
    return result


__all__ = [
    "train_test_split",
    "cross_val_score",
    "KFold",
    "StratifiedKFold",
    "TimeSeriesSplit",
    "LeaveOneOut",
    "RepeatedKFold",
    "ShuffleSplit",
    "GroupKFold",
    "LeavePOut",
    "GridSearchCV",
    "RandomSearchCV",
    "cross_validate",
    "learning_curve",
    "validation_curve",
]
