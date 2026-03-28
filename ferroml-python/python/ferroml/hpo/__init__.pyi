from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable

class GridSearchCV:
    best_params_: dict[str, Any]
    best_score_: float
    cv_results_: dict[str, Any]
    def __init__(self, model: Any, param_grid: dict[str, list[Any]], cv: int = 5, scoring: str = "accuracy") -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> GridSearchCV: ...

class RandomSearchCV:
    best_params_: dict[str, Any]
    best_score_: float
    cv_results_: dict[str, Any]
    def __init__(self, model: Any, param_distributions: dict[str, list[Any]], n_iter: int = 10, cv: int = 5, scoring: str = "accuracy", seed: int | None = None) -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> RandomSearchCV: ...

class Study:
    best_params: dict[str, Any]
    best_value: float
    trials: list[dict[str, Any]]
    n_trials: int
    def __init__(self, direction: str = "minimize", seed: int | None = None) -> None: ...
    def optimize(self, objective: Callable[[dict[str, Any]], float], n_trials: int, search_space: dict[str, Any] | None = None) -> Study: ...

__all__ = [
    "GridSearchCV",
    "RandomSearchCV",
    "Study",
]
