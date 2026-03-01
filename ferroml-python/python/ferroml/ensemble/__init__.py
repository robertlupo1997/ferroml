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
BaggingClassifier
    Bootstrap aggregating classifier with factory constructors for base estimators
BaggingRegressor
    Bootstrap aggregating regressor with factory constructors for base estimators

Example
-------
>>> from ferroml.ensemble import ExtraTreesClassifier, SGDClassifier, BaggingClassifier, BaggingRegressor
>>> import numpy as np
>>>
>>> model = ExtraTreesClassifier(n_estimators=100, random_state=42)
>>> model.fit(X_train, y_train)
>>> print(f"Feature importances: {model.feature_importances_}")
>>>
>>> # BaggingClassifier uses factory methods for different base estimators
>>> bag = BaggingClassifier.with_decision_tree(n_estimators=10, max_depth=5)
>>> bag.fit(X_train, y_train)
>>> predictions = bag.predict(X_test)
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
BaggingClassifier = _native.ensemble.BaggingClassifier
BaggingRegressor = _native.ensemble.BaggingRegressor

__all__ = [
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "SGDClassifier",
    "SGDRegressor",
    "PassiveAggressiveClassifier",
    "BaggingClassifier",
    "BaggingRegressor",
]
