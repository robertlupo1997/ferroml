"""
FerroML Dimensionality Reduction and Matrix Decomposition

Linear and probabilistic methods for reducing feature dimensionality
and extracting latent structure from data.

Classes
-------
PCA
    Principal Component Analysis via eigen-decomposition; supports
    explained-variance ratio and cumulative variance queries.
IncrementalPCA
    Online PCA that processes data in mini-batches; suitable for
    datasets too large to fit in memory at once.
TruncatedSVD
    Randomized truncated singular-value decomposition; works on both
    dense and sparse matrices (also known as LSA when applied to
    term-frequency matrices).
LDA
    Linear Discriminant Analysis; finds the projection that maximises
    class separability while minimising within-class scatter.
FactorAnalysis
    Probabilistic factor analysis that models observed variables as
    linear combinations of latent factors plus Gaussian noise.

Example
-------
>>> from ferroml.decomposition import PCA, TruncatedSVD
>>> import numpy as np
>>>
>>> X = np.random.default_rng(0).standard_normal((200, 10))
>>> pca = PCA(n_components=3)
>>> pca.fit(X)
>>> X_reduced = pca.transform(X)
>>> print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
"""

# Import from the native extension's decomposition submodule
from ferroml import ferroml as _native

PCA = _native.decomposition.PCA
IncrementalPCA = _native.decomposition.IncrementalPCA
TruncatedSVD = _native.decomposition.TruncatedSVD
LDA = _native.decomposition.LDA
FactorAnalysis = _native.decomposition.FactorAnalysis

__all__ = [
    "PCA",
    "IncrementalPCA",
    "TruncatedSVD",
    "LDA",
    "FactorAnalysis",
]
