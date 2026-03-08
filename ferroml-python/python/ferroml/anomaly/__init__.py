"""
Anomaly detection models.

Classes
-------
IsolationForest
    Isolation Forest anomaly detector using random recursive partitioning.
LocalOutlierFactor
    Local Outlier Factor for density-based anomaly/novelty detection.
"""

from ferroml import ferroml as _native

IsolationForest = _native.anomaly.IsolationForest
LocalOutlierFactor = _native.anomaly.LocalOutlierFactor

__all__ = [
    "IsolationForest",
    "LocalOutlierFactor",
]
