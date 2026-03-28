from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any

__all__ = [
    "KMeans",
    "MiniBatchKMeans",
    "DBSCAN",
    "AgglomerativeClustering",
    "GaussianMixture",
    "HDBSCAN",
    "silhouette_score",
    "silhouette_samples",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "adjusted_rand_index",
    "normalized_mutual_info",
    "hopkins_statistic",
]

class KMeans:
    cluster_centers_: NDArray[np.float64]
    labels_: NDArray[np.int32]
    inertia_: float
    n_iter_: int
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
        n_init: int = 10,
        algorithm: str = "auto",
    ) -> None: ...
    def fit(self, x: NDArray[np.float64]) -> KMeans: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64]) -> float: ...
    def cluster_stability(self, x: NDArray[np.float64], n_bootstrap: int = 100) -> NDArray[np.float64]: ...
    def silhouette_with_ci(self, x: NDArray[np.float64], confidence: float = 0.95) -> tuple[float, float, float]: ...
    @staticmethod
    def optimal_k(
        x: NDArray[np.float64],
        k_min: int = 1,
        k_max: int = 10,
        n_refs: int = 10,
        random_state: int | None = None,
    ) -> dict[str, Any]: ...
    @staticmethod
    def elbow(
        x: NDArray[np.float64],
        k_min: int = 1,
        k_max: int = 10,
        random_state: int | None = None,
    ) -> dict[str, Any]: ...

class MiniBatchKMeans:
    cluster_centers_: NDArray[np.float64]
    labels_: NDArray[np.int32]
    inertia_: float
    n_iter_: int
    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 1024,
        max_iter: int = 100,
        n_init: int = 3,
        tol: float = 0.0,
        reassignment_ratio: float = 0.01,
        random_state: int | None = None,
        init: str = "k-means++",
    ) -> None: ...
    def fit(self, x: NDArray[np.float64]) -> MiniBatchKMeans: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def partial_fit(self, x: NDArray[np.float64]) -> MiniBatchKMeans: ...
    def score(self, x: NDArray[np.float64]) -> float: ...

class DBSCAN:
    labels_: NDArray[np.int32]
    core_sample_indices_: list[int]
    components_: NDArray[np.float64]
    n_clusters_: int
    n_noise_: int
    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None: ...
    def fit(self, x: NDArray[np.float64]) -> DBSCAN: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def noise_analysis(self, x: NDArray[np.float64]) -> dict[str, Any]: ...
    @staticmethod
    def optimal_eps(x: NDArray[np.float64], min_samples: int = 5) -> dict[str, Any]: ...
    @staticmethod
    def cluster_persistence(
        x: NDArray[np.float64],
        eps_values: list[float],
        min_samples: int = 5,
    ) -> list[tuple[float, int, int]]: ...

class AgglomerativeClustering:
    labels_: NDArray[np.int32]
    def __init__(self, n_clusters: int = 2, linkage: str = "ward") -> None: ...
    def fit(self, x: NDArray[np.float64]) -> AgglomerativeClustering: ...
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

class GaussianMixture:
    weights_: NDArray[np.float64]
    means_: NDArray[np.float64]
    labels_: NDArray[np.int32]
    n_iter_: int
    converged_: bool
    lower_bound_: float
    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        max_iter: int = 100,
        tol: float = 1e-3,
        n_init: int = 1,
        init_params: str = "kmeans",
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None: ...
    def fit(self, x: NDArray[np.float64]) -> GaussianMixture: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_log_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def score(self, x: NDArray[np.float64]) -> float: ...
    def score_samples(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def aic(self, x: NDArray[np.float64]) -> float: ...
    def bic(self, x: NDArray[np.float64]) -> float: ...
    def sample(self, n_samples: int = 1) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...

class HDBSCAN:
    labels_: NDArray[np.int32]
    probabilities_: NDArray[np.float64]
    n_clusters_: int
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
        allow_single_cluster: bool = False,
    ) -> None: ...
    def fit(self, x: NDArray[np.float64]) -> HDBSCAN: ...
    def fit_predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

def silhouette_score(x: NDArray[np.float64], labels: NDArray[np.float64]) -> float: ...
def silhouette_samples(x: NDArray[np.float64], labels: NDArray[np.float64]) -> NDArray[np.float64]: ...
def calinski_harabasz_score(x: NDArray[np.float64], labels: NDArray[np.float64]) -> float: ...
def davies_bouldin_score(x: NDArray[np.float64], labels: NDArray[np.float64]) -> float: ...
def adjusted_rand_index(labels_true: NDArray[np.float64], labels_pred: NDArray[np.float64]) -> float: ...
def normalized_mutual_info(labels_true: NDArray[np.float64], labels_pred: NDArray[np.float64]) -> float: ...
def hopkins_statistic(
    x: NDArray[np.float64],
    sample_size: int | None = None,
    random_state: int | None = None,
) -> float: ...
