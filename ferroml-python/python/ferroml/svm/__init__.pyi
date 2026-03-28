from __future__ import annotations

from ferroml import ModelCard

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "LinearSVC",
    "LinearSVR",
    "SVC",
    "SVR",
]

class LinearSVC:
    def __init__(
        self,
        c: float = 1.0,
        loss: str = "squared_hinge",
        max_iter: int = 1000,
        tol: float = 1e-4,
        class_weight: str | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> LinearSVC: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def decision_function(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def fit_sparse(self, x: Any, y: NDArray[np.float64]) -> LinearSVC: ...
    def predict_sparse(self, x: Any) -> NDArray[np.float64]: ...
    def export_onnx(
        self,
        path: str,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> None: ...
    def to_onnx_bytes(
        self,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> bytes: ...

class LinearSVR:
    def __init__(
        self,
        c: float = 1.0,
        epsilon: float = 0.0,
        loss: str = "epsilon_insensitive",
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> LinearSVR: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def decision_function(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def fit_sparse(self, x: Any, y: NDArray[np.float64]) -> LinearSVR: ...
    def predict_sparse(self, x: Any) -> NDArray[np.float64]: ...
    def export_onnx(
        self,
        path: str,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> None: ...
    def to_onnx_bytes(
        self,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> bytes: ...

class SVC:
    def __init__(
        self,
        kernel: str = "rbf",
        c: float = 1.0,
        gamma: float = 0.0,
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        probability: bool = False,
        multiclass: str = "ovo",
        class_weight: str | None = None,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> SVC: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_log_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def decision_function(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def export_onnx(
        self,
        path: str,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> None: ...
    def to_onnx_bytes(
        self,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> bytes: ...

class SVR:
    def __init__(
        self,
        kernel: str = "rbf",
        c: float = 1.0,
        epsilon: float = 0.1,
        gamma: float = 0.0,
        degree: int = 3,
        coef0: float = 0.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> SVR: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def decision_function(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def export_onnx(
        self,
        path: str,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> None: ...
    def to_onnx_bytes(
        self,
        model_name: str | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> bytes: ...
