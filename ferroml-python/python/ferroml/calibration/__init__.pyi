from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "TemperatureScalingCalibrator",
]

class TemperatureScalingCalibrator:
    def __init__(self, max_iter: int = 100, learning_rate: float = 0.01) -> None: ...
    def fit(self, y_prob: NDArray[np.float64], y_true: NDArray[np.float64]) -> TemperatureScalingCalibrator: ...
    def transform(self, y_prob: NDArray[np.float64]) -> NDArray[np.float64]: ...
