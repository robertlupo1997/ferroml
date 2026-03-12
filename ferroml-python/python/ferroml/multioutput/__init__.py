"""
Multi-output wrappers for single-output estimators.

Classes
-------
MultiOutputRegressor
    Fits one regressor per target column.
MultiOutputClassifier
    Fits one classifier per target column.
"""

from ferroml import ferroml as _native

MultiOutputRegressor = _native.multioutput.MultiOutputRegressor
MultiOutputClassifier = _native.multioutput.MultiOutputClassifier

__all__ = ["MultiOutputRegressor", "MultiOutputClassifier"]
