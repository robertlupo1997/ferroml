from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "IsolationForest",
    "LocalOutlierFactor",
]

class IsolationForest:
    offset_: float
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = "auto",
        contamination: str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def fit(self, x: NDArray[np.float64]) -> IsolationForest: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def decision_function(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score_samples(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

class LocalOutlierFactor:
    negative_outlier_factor_: NDArray[np.float64]
    offset_: float
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: str = "auto",
        metric: str = "euclidean",
        algorithm: str = "auto",
        novelty: bool = False,
    ) -> None: ...
    def fit(self, x: NDArray[np.float64]) -> LocalOutlierFactor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def decision_function(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score_samples(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
