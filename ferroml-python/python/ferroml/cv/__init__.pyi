from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any

class KFold:
    def __init__(self, n_folds: int = 5, shuffle: bool = False, random_state: int | None = None) -> None: ...
    def split(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

class StratifiedKFold:
    def __init__(self, n_folds: int = 5, shuffle: bool = False, random_state: int | None = None) -> None: ...
    def split(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

class TimeSeriesSplit:
    def __init__(self, n_splits: int = 5) -> None: ...
    def split(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

class LeaveOneOut:
    def __init__(self) -> None: ...
    def get_n_splits(self, x: NDArray[np.float64]) -> int: ...
    def split(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

class RepeatedKFold:
    def __init__(self, n_folds: int = 5, n_repeats: int = 10, random_state: int | None = None) -> None: ...
    def split(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

class ShuffleSplit:
    def __init__(self, n_splits: int = 10, test_size: float = 0.1, train_size: float | None = None, random_state: int | None = None) -> None: ...
    def split(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

class GroupKFold:
    def __init__(self, n_folds: int = 5) -> None: ...
    def split(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None, groups: NDArray[np.float64] | None = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

class LeavePOut:
    def __init__(self, p: int = 2) -> None: ...
    def split(self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]: ...

def cross_val_score(model: Any, x: NDArray[np.float64], y: NDArray[np.float64], cv: int = 5, scoring: str = "accuracy") -> NDArray[np.float64]: ...

def learning_curve(
    model: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    cv: int = 5,
    train_sizes: list[float] | None = None,
    scoring: str = "accuracy",
) -> tuple[list[int], list[float], list[float]]: ...

def validation_curve(
    model: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    param_name: str,
    param_range: list[Any],
    cv: int = 5,
    scoring: str = "accuracy",
) -> tuple[list[Any], list[float], list[float]]: ...

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
