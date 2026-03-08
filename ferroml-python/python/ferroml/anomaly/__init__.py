"""
FerroML Anomaly Detection

Unsupervised anomaly and outlier detection algorithms.

Classes
-------
IsolationForest
    Isolation Forest anomaly detector using random recursive partitioning.
    Scores samples based on the average path length in an ensemble of
    isolation trees; shorter paths indicate anomalies.
LocalOutlierFactor
    Local Outlier Factor for density-based anomaly/novelty detection.
    Compares the local density of a point to its neighbors; points with
    substantially lower density are flagged as outliers.

Example
-------
>>> from ferroml.anomaly import IsolationForest
>>> import numpy as np
>>>
>>> rng = np.random.default_rng(42)
>>> X_normal = rng.standard_normal((100, 2))
>>> X_outlier = rng.uniform(low=-6, high=6, size=(5, 2))
>>> X = np.vstack([X_normal, X_outlier])
>>> model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
>>> model.fit(X)
>>> labels = model.predict(X)  # 1 for inliers, -1 for outliers
>>> scores = model.score_samples(X)  # lower = more anomalous
"""

from ferroml import ferroml as _native

IsolationForest = _native.anomaly.IsolationForest
LocalOutlierFactor = _native.anomaly.LocalOutlierFactor

__all__ = [
    "IsolationForest",
    "LocalOutlierFactor",
]
