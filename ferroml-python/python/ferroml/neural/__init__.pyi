from __future__ import annotations

from ferroml import ModelCard
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MLPClassifier",
    "MLPRegressor",
]

class MLPClassifier:
    classes_: NDArray[np.float64] | None
    n_layers_: int
    loss_curve_: list[float] | None
    def __init__(
        self,
        hidden_layer_sizes: list[int] | None = None,
        activation: str = "relu",
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int | None = None,
        alpha: float = 0.0001,
        batch_size: int = 200,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> MLPClassifier: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def is_fitted(self) -> bool: ...

class MLPRegressor:
    n_layers_: int
    loss_curve_: list[float] | None
    def __init__(
        self,
        hidden_layer_sizes: list[int] | None = None,
        activation: str = "relu",
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int | None = None,
        alpha: float = 0.0001,
        batch_size: int = 200,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> MLPRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def is_fitted(self) -> bool: ...
