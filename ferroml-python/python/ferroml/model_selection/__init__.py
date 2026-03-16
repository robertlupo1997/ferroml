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

# Re-export HPO classes
from ferroml.hpo import GridSearchCV, RandomSearchCV

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
]
