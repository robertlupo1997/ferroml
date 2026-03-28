from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable

from ferroml.cv import (
    KFold as KFold,
    StratifiedKFold as StratifiedKFold,
    TimeSeriesSplit as TimeSeriesSplit,
    LeaveOneOut as LeaveOneOut,
    RepeatedKFold as RepeatedKFold,
    ShuffleSplit as ShuffleSplit,
    GroupKFold as GroupKFold,
    LeavePOut as LeavePOut,
    cross_val_score as cross_val_score,
    learning_curve as learning_curve,
    validation_curve as validation_curve,
)

from ferroml.hpo import (
    GridSearchCV as GridSearchCV,
    RandomSearchCV as RandomSearchCV,
)

def train_test_split(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    test_size: float = 0.25,
    shuffle: bool = True,
    random_state: int | None = None,
    stratify: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

def cross_validate(
    model: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    cv: int = 5,
    scoring: str = "accuracy",
    return_train_score: bool = False,
) -> dict[str, list[float]]: ...

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
