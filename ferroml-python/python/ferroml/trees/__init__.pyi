from __future__ import annotations

from ferroml import ModelCard

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
]

class DecisionTreeClassifier:
    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        ccp_alpha: float = 0.0,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> DecisionTreeClassifier: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_log_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def decision_function(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def get_depth(self) -> int: ...
    def get_n_leaves(self) -> int: ...
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

class DecisionTreeRegressor:
    def __init__(
        self,
        criterion: str = "mse",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        ccp_alpha: float = 0.0,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> DecisionTreeRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    def get_depth(self) -> int: ...
    def get_n_leaves(self) -> int: ...
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

class RandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = True,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> RandomForestClassifier: ...
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

class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = True,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> RandomForestRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
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

class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> GradientBoostingClassifier: ...
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

class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        loss: str = "squared_error",
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> GradientBoostingRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
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

class HistGradientBoostingClassifier:
    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = None,
        max_leaf_nodes: int = 31,
        max_bins: int = 255,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> HistGradientBoostingClassifier: ...
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

class HistGradientBoostingRegressor:
    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        loss: str = "squared_error",
        max_depth: int | None = None,
        max_leaf_nodes: int = 31,
        max_bins: int = 255,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> HistGradientBoostingRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
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
