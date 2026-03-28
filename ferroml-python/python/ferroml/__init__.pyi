from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any

from ferroml import linear as linear
from ferroml import trees as trees
from ferroml import neighbors as neighbors
from ferroml import clustering as clustering
from ferroml import preprocessing as preprocessing
from ferroml import pipeline as pipeline
from ferroml import automl as automl
from ferroml import datasets as datasets
from ferroml import decomposition as decomposition
from ferroml import ensemble as ensemble
from ferroml import explainability as explainability
from ferroml import naive_bayes as naive_bayes
from ferroml import neural as neural
from ferroml import svm as svm
from ferroml import calibration as calibration
from ferroml import anomaly as anomaly
from ferroml import multioutput as multioutput
from ferroml import gaussian_process as gaussian_process
from ferroml import stats as stats
from ferroml import metrics as metrics
from ferroml import cv as cv
from ferroml import hpo as hpo
from ferroml import model_selection as model_selection

__version__: str

class Recommendation:
    @property
    def algorithm(self) -> str: ...
    @property
    def reason(self) -> str: ...
    @property
    def estimated_fit_time(self) -> str: ...
    @property
    def params(self) -> dict[str, Any]: ...
    @property
    def score(self) -> float: ...

def recommend(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    task: str = "classification",
) -> list[Recommendation]: ...

class ModelCard:
    @property
    def name(self) -> str: ...
    @property
    def task(self) -> list[str]: ...
    @property
    def complexity(self) -> str: ...
    @property
    def interpretability(self) -> str: ...
    @property
    def supports_sparse(self) -> bool: ...
    @property
    def supports_incremental(self) -> bool: ...
    @property
    def supports_sample_weight(self) -> bool: ...
    @property
    def strengths(self) -> list[str]: ...
    @property
    def limitations(self) -> list[str]: ...
    @property
    def references(self) -> list[str]: ...

__all__ = [
    "__version__",
    "recommend",
    "Recommendation",
    "ModelCard",
    "linear",
    "trees",
    "neighbors",
    "clustering",
    "preprocessing",
    "pipeline",
    "automl",
    "datasets",
    "decomposition",
    "ensemble",
    "explainability",
    "naive_bayes",
    "neural",
    "svm",
    "calibration",
    "anomaly",
    "multioutput",
    "gaussian_process",
    "stats",
    "metrics",
    "cv",
    "hpo",
    "model_selection",
]
