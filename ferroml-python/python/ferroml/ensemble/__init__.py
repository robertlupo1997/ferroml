"""
FerroML Ensemble and Online Learning Models

Ensemble methods and stochastic gradient descent models for classification
and regression tasks.

Classes
-------
ExtraTreesClassifier
    Extremely randomized trees classifier with feature importances
ExtraTreesRegressor
    Extremely randomized trees regressor with feature importances
AdaBoostClassifier
    Adaptive boosting classifier using decision stumps
AdaBoostRegressor
    Adaptive boosting regressor with multiple loss functions
SGDClassifier
    Linear classifier fitted with stochastic gradient descent
SGDRegressor
    Linear regressor fitted with stochastic gradient descent
PassiveAggressiveClassifier
    Online passive-aggressive algorithm for classification

Example
-------
>>> from ferroml.ensemble import ExtraTreesClassifier, SGDClassifier
>>> import numpy as np
>>>
>>> model = ExtraTreesClassifier(n_estimators=100, random_state=42)
>>> model.fit(X_train, y_train)
>>> print(f"Feature importances: {model.feature_importances_}")
"""

# Import from the native extension's ensemble submodule
from ferroml import ferroml as _native

ExtraTreesClassifier = _native.ensemble.ExtraTreesClassifier
ExtraTreesRegressor = _native.ensemble.ExtraTreesRegressor
AdaBoostClassifier = _native.ensemble.AdaBoostClassifier
AdaBoostRegressor = _native.ensemble.AdaBoostRegressor
SGDClassifier = _native.ensemble.SGDClassifier
SGDRegressor = _native.ensemble.SGDRegressor
PassiveAggressiveClassifier = _native.ensemble.PassiveAggressiveClassifier

__all__ = [
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "SGDClassifier",
    "SGDRegressor",
    "PassiveAggressiveClassifier",
]
