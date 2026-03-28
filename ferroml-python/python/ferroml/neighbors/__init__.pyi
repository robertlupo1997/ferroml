from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "NearestCentroid",
]

class KNeighborsClassifier:
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
        algorithm: str = "auto",
        p: float = 2.0,
        leaf_size: int = 30,
    ) -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> KNeighborsClassifier: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_log_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...

class KNeighborsRegressor:
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "euclidean",
        algorithm: str = "auto",
        p: float = 2.0,
        leaf_size: int = 30,
    ) -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> KNeighborsRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...

class NearestCentroid:
    def __init__(
        self,
        metric: str = "euclidean",
        shrink_threshold: float | None = None,
    ) -> None: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NearestCentroid: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
