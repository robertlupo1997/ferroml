"""FerroML Cross-Validation -- splitters and scoring utilities.

This module provides sklearn-compatible cross-validation strategies
for model evaluation with proper train/test splitting.

Classes
-------
KFold(n_folds=5, shuffle=False, random_state=None)
    Standard k-fold cross-validation.
StratifiedKFold(n_folds=5, shuffle=False, random_state=None)
    Stratified k-fold preserving class distribution.
TimeSeriesSplit(n_splits=5)
    Time-series aware cross-validation.
LeaveOneOut()
    Leave-one-out cross-validation.

Functions
---------
cross_val_score(model, X, y, cv=5, scoring="accuracy")
    Evaluate a model with cross-validation, returning per-fold scores.
"""

import numpy as np

from ferroml.ferroml import cv as _cv

# Splitters
KFold = _cv.KFold
StratifiedKFold = _cv.StratifiedKFold
TimeSeriesSplit = _cv.TimeSeriesSplit
LeaveOneOut = _cv.LeaveOneOut
RepeatedKFold = _cv.RepeatedKFold
ShuffleSplit = _cv.ShuffleSplit
GroupKFold = _cv.GroupKFold
LeavePOut = _cv.LeavePOut

# Utility functions
cross_val_score = _cv.cross_val_score


def _score(model, X, y, scoring):
    """Compute a scoring metric."""
    preds = model.predict(X)
    if scoring == "accuracy":
        return float(np.mean(np.abs(preds - y) < 1e-10))
    elif scoring == "mse":
        return -float(np.mean((preds - y) ** 2))
    elif scoring == "mae":
        return -float(np.mean(np.abs(preds - y)))
    elif scoring == "r2":
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)
    else:
        raise ValueError(f"Unknown scoring '{scoring}'. Use 'accuracy', 'mse', 'mae', or 'r2'.")


def learning_curve(model, X, y, cv=5, train_sizes=None, scoring="accuracy"):
    """Compute train/test scores for varying training set sizes.

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
    train_sizes : list of float, optional
        Fractions of training data to use (default: [0.1, 0.33, 0.55, 0.78, 1.0]).
    scoring : str, optional (default="accuracy")
        Scoring metric.

    Returns
    -------
    train_sizes_abs : list of int
    train_scores : list of float (mean per size)
    test_scores : list of float (mean per size)
    """
    if train_sizes is None:
        train_sizes = [0.1, 0.33, 0.55, 0.78, 1.0]

    kf = KFold(n_folds=cv)
    splits = kf.split(X)

    results_train = []
    results_test = []
    sizes_abs = []

    for frac in train_sizes:
        fold_train_scores = []
        fold_test_scores = []
        for train_idx, test_idx in splits:
            n_train = max(1, int(len(train_idx) * frac))
            sub_train_idx = train_idx[:n_train]

            X_tr = X[sub_train_idx]
            y_tr = y[sub_train_idx]
            X_te = X[test_idx]
            y_te = y[test_idx]

            m = model.__class__()
            m.fit(X_tr, y_tr)

            fold_train_scores.append(_score(m, X_tr, y_tr, scoring))
            fold_test_scores.append(_score(m, X_te, y_te, scoring))

        sizes_abs.append(max(1, int(len(splits[0][0]) * frac)))
        results_train.append(float(np.mean(fold_train_scores)))
        results_test.append(float(np.mean(fold_test_scores)))

    return sizes_abs, results_train, results_test


def validation_curve(model, X, y, param_name, param_range, cv=5, scoring="accuracy"):
    """Compute train/test scores for varying hyperparameter values.

    Parameters
    ----------
    model : estimator
        A FerroML model with fit() and predict() methods.
    X : numpy.ndarray
        Feature matrix.
    y : numpy.ndarray
        Target array.
    param_name : str
        Name of the hyperparameter to vary.
    param_range : list
        Values of the hyperparameter to evaluate.
    cv : int, optional (default=5)
        Number of cross-validation folds.
    scoring : str, optional (default="accuracy")
        Scoring metric.

    Returns
    -------
    param_range : list
    train_scores : list of float (mean per value)
    test_scores : list of float (mean per value)
    """
    train_scores = []
    test_scores = []

    for value in param_range:
        scores_train = []
        scores_test = []

        kf = KFold(n_folds=cv)
        for train_idx, test_idx in kf.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            m_fold = model.__class__(**{param_name: value})
            m_fold.fit(X_tr, y_tr)
            scores_train.append(_score(m_fold, X_tr, y_tr, scoring))
            scores_test.append(_score(m_fold, X_te, y_te, scoring))

        train_scores.append(float(np.mean(scores_train)))
        test_scores.append(float(np.mean(scores_test)))

    return list(param_range), train_scores, test_scores


__all__ = [
    "KFold",
    "StratifiedKFold",
    "TimeSeriesSplit",
    "LeaveOneOut",
    "RepeatedKFold",
    "ShuffleSplit",
    "GroupKFold",
    "LeavePOut",
    "cross_val_score",
    "learning_curve",
    "validation_curve",
]
