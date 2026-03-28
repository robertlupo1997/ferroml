from __future__ import annotations

from ferroml import ModelCard

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "SGDClassifier",
    "SGDRegressor",
    "PassiveAggressiveClassifier",
    "BaggingClassifier",
    "BaggingRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
]

class ExtraTreesClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> ExtraTreesClassifier: ...
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

class ExtraTreesRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> ExtraTreesRegressor: ...
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

class AdaBoostClassifier:
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> AdaBoostClassifier: ...
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

class AdaBoostRegressor:
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: str = "linear",
        max_depth: int = 3,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> AdaBoostRegressor: ...
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

class SGDClassifier:
    def __init__(
        self,
        loss: str = "hinge",
        penalty: str = "l2",
        alpha: float = 0.0001,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> SGDClassifier: ...
    def partial_fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        classes: NDArray[np.float64] | None = None,
    ) -> SGDClassifier: ...
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

class SGDRegressor:
    def __init__(
        self,
        loss: str = "squared_error",
        penalty: str = "l2",
        alpha: float = 0.0001,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> SGDRegressor: ...
    def partial_fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> SGDRegressor: ...
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

class PassiveAggressiveClassifier:
    def __init__(
        self,
        c: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> PassiveAggressiveClassifier: ...
    def partial_fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        classes: NDArray[np.float64] | None = None,
    ) -> PassiveAggressiveClassifier: ...
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

class BaggingClassifier:
    def __init__(self) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> BaggingClassifier: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_log_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    @staticmethod
    def with_decision_tree(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> BaggingClassifier: ...
    @staticmethod
    def with_gaussian_nb(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
    ) -> BaggingClassifier: ...
    @staticmethod
    def with_gradient_boosting(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        gb_n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ) -> BaggingClassifier: ...
    @staticmethod
    def with_hist_gradient_boosting(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        hgb_max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = None,
    ) -> BaggingClassifier: ...
    @staticmethod
    def with_knn(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        n_neighbors: int = 5,
    ) -> BaggingClassifier: ...
    @staticmethod
    def with_logistic_regression(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        max_iter: int = 100,
    ) -> BaggingClassifier: ...
    @staticmethod
    def with_random_forest(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        rf_n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> BaggingClassifier: ...
    @staticmethod
    def with_svc(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        c: float = 1.0,
    ) -> BaggingClassifier: ...

class BaggingRegressor:
    def __init__(self) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> BaggingRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
    @staticmethod
    def with_decision_tree(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> BaggingRegressor: ...
    @staticmethod
    def with_extra_trees(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        et_n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> BaggingRegressor: ...
    @staticmethod
    def with_gradient_boosting(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        gb_n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
    ) -> BaggingRegressor: ...
    @staticmethod
    def with_hist_gradient_boosting(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        hgb_max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = None,
    ) -> BaggingRegressor: ...
    @staticmethod
    def with_knn(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        n_neighbors: int = 5,
    ) -> BaggingRegressor: ...
    @staticmethod
    def with_linear_regression(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        fit_intercept: bool = True,
    ) -> BaggingRegressor: ...
    @staticmethod
    def with_random_forest(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        rf_n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> BaggingRegressor: ...
    @staticmethod
    def with_ridge_regression(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        alpha: float = 1.0,
    ) -> BaggingRegressor: ...
    @staticmethod
    def with_svr(
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        random_state: int | None = None,
        warm_start: bool = False,
        c: float = 1.0,
        epsilon: float = 0.1,
    ) -> BaggingRegressor: ...

class VotingClassifier:
    def __init__(
        self,
        estimators: list[tuple[str, str]],
        voting: str = "hard",
        weights: list[float] | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> VotingClassifier: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_log_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...

class VotingRegressor:
    def __init__(
        self,
        estimators: list[tuple[str, str]],
        weights: list[float] | None = None,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> VotingRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...

class StackingClassifier:
    def __init__(
        self,
        estimators: list[tuple[str, str]],
        final_estimator: str = "logistic_regression",
        cv: int = 5,
        stack_method: str = "predict_proba",
        passthrough: bool = False,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> StackingClassifier: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_log_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...

class StackingRegressor:
    def __init__(
        self,
        estimators: list[tuple[str, str]],
        final_estimator: str = "linear_regression",
        cv: int = 5,
        passthrough: bool = False,
    ) -> None: ...
    @staticmethod
    def model_card() -> ModelCard: ...

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> StackingRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> float: ...
