from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any

class Dataset:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    n_samples: int
    n_features: int
    shape: tuple[int, int]
    feature_names: list[str] | None
    target_names: list[str] | None
    def __init__(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None: ...
    def class_counts(self) -> dict[float, int]: ...
    def describe(self) -> dict[str, Any]: ...
    def infer_task(self) -> str: ...
    def is_binary(self) -> bool: ...
    def is_multiclass(self) -> bool: ...
    def set_feature_names(self, names: list[str]) -> None: ...
    def set_target_names(self, names: list[str]) -> None: ...
    def train_test_split(self, test_size: float, shuffle: bool = True, random_state: int | None = None) -> tuple[Dataset, Dataset]: ...
    def unique_classes(self) -> NDArray[np.float64]: ...

class DatasetInfo:
    name: str
    description: str
    task: str
    n_samples: int
    n_features: int
    n_classes: int | None
    feature_names: list[str]
    target_names: list[str] | None
    source: str | None
    url: str | None
    license: str | None
    def __init__(self, name: str, task: str, n_samples: int, n_features: int) -> None: ...

def load_iris() -> tuple[Dataset, DatasetInfo]: ...
def load_wine() -> tuple[Dataset, DatasetInfo]: ...
def load_diabetes() -> tuple[Dataset, DatasetInfo]: ...
def load_linnerud() -> tuple[Dataset, DatasetInfo]: ...

def make_classification(
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 5,
    n_classes: int = 2,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def make_regression(
    n_samples: int = 100,
    n_features: int = 10,
    n_informative: int = 5,
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def make_blobs(
    n_samples: int = 100,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def make_moons(
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def make_circles(
    n_samples: int = 100,
    noise: float = 0.1,
    factor: float = 0.5,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def load_huggingface(
    dataset_name: str,
    target_column: str,
    split: str = "train",
    feature_columns: list[str] | None = None,
    config_name: str | None = None,
    cache_dir: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[Dataset, DatasetInfo]: ...

__all__ = [
    "Dataset",
    "DatasetInfo",
    "load_iris",
    "load_wine",
    "load_diabetes",
    "load_linnerud",
    "make_classification",
    "make_regression",
    "make_blobs",
    "make_moons",
    "make_circles",
    "load_huggingface",
]
