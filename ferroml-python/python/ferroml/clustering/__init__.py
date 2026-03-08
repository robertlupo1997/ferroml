"""
FerroML Clustering Algorithms

Clustering algorithms with statistical extensions beyond sklearn.

Classes
-------
KMeans
    K-Means clustering with kmeans++ initialization
DBSCAN
    Density-based spatial clustering with noise detection
AgglomerativeClustering
    Hierarchical agglomerative clustering with Ward, complete, average,
    and single linkage strategies
GaussianMixture
    Gaussian Mixture Model with EM algorithm, multiple covariance types,
    BIC/AIC model selection, and soft clustering via predict_proba

Functions
---------
silhouette_score
    Mean silhouette coefficient for clustering quality
silhouette_samples
    Per-sample silhouette coefficients
calinski_harabasz_score
    Variance ratio criterion (higher is better)
davies_bouldin_score
    Average similarity between clusters (lower is better)
adjusted_rand_index
    Similarity between clusterings, adjusted for chance
normalized_mutual_info
    Normalized mutual information between clusterings
hopkins_statistic
    Clustering tendency (>0.5 suggests clusters)

Example
-------
>>> from ferroml.clustering import KMeans, silhouette_score
>>> import numpy as np
>>>
>>> X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
>>> kmeans = KMeans(n_clusters=2, random_state=42)
>>> kmeans.fit(X)
>>> print(f"Labels: {kmeans.labels_}")
>>> print(f"Inertia: {kmeans.inertia_}")
>>>
>>> # Evaluate clustering quality
>>> score = silhouette_score(X, kmeans.labels_)
>>> print(f"Silhouette score: {score:.4f}")
>>>
>>> # Find optimal k using gap statistic
>>> result = KMeans.optimal_k(X, k_min=1, k_max=5)
>>> print(f"Optimal k: {result['optimal_k']}")
"""

# Import from the native extension's clustering submodule
from ferroml import ferroml as _native

# Classes
KMeans = _native.clustering.KMeans
DBSCAN = _native.clustering.DBSCAN
AgglomerativeClustering = _native.clustering.AgglomerativeClustering
GaussianMixture = _native.clustering.GaussianMixture

# Metric functions
silhouette_score = _native.clustering.silhouette_score
silhouette_samples = _native.clustering.silhouette_samples
calinski_harabasz_score = _native.clustering.calinski_harabasz_score
davies_bouldin_score = _native.clustering.davies_bouldin_score
adjusted_rand_index = _native.clustering.adjusted_rand_index
normalized_mutual_info = _native.clustering.normalized_mutual_info
hopkins_statistic = _native.clustering.hopkins_statistic

__all__ = [
    # Classes
    "KMeans",
    "DBSCAN",
    "AgglomerativeClustering",
    "GaussianMixture",
    # Metrics
    "silhouette_score",
    "silhouette_samples",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "adjusted_rand_index",
    "normalized_mutual_info",
    "hopkins_statistic",
]
