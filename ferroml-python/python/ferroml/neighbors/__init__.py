"""
FerroML K-Nearest Neighbors

K-Nearest Neighbors classification and regression with efficient
spatial data structures (KD-Tree, Ball Tree).

Classes
-------
KNeighborsClassifier
    K-Nearest Neighbors classifier with majority voting
KNeighborsRegressor
    K-Nearest Neighbors regressor with weighted averaging

Example
-------
>>> from ferroml.neighbors import KNeighborsClassifier
>>> import numpy as np
>>>
>>> X = np.array([[1, 2], [2, 1], [3, 3], [6, 7], [7, 6], [8, 8]])
>>> y = np.array([0, 0, 0, 1, 1, 1])
>>> model = KNeighborsClassifier(n_neighbors=3)
>>> model.fit(X, y)
>>> predictions = model.predict(X)
>>> probas = model.predict_proba(X)
"""

# Import from the native extension's neighbors submodule
from ferroml import ferroml as _native

KNeighborsClassifier = _native.neighbors.KNeighborsClassifier
KNeighborsRegressor = _native.neighbors.KNeighborsRegressor
NearestCentroid = _native.neighbors.NearestCentroid

__all__ = [
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "NearestCentroid",
]
