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
VotingClassifier
    Voting classifier combining multiple classifiers via hard or soft voting
VotingRegressor
    Voting regressor combining multiple regressors by averaging predictions
StackingClassifier
    Stacking classifier combining multiple classifiers with a meta-learner
StackingRegressor
    Stacking regressor combining multiple regressors with a meta-learner

Example
-------
>>> from ferroml.ensemble import VotingClassifier, StackingRegressor
>>> import numpy as np
>>>
>>> # Voting: combine classifiers with soft voting
>>> voter = VotingClassifier(
...     [("lr", "logistic_regression"), ("dt", "decision_tree"), ("nb", "gaussian_nb")],
...     voting="soft",
... )
>>> voter.fit(X_train, y_train)
>>> predictions = voter.predict(X_test)
>>>
>>> # Stacking: use cross-validation meta-features with a meta-learner
>>> stacker = StackingRegressor(
...     [("lr", "linear_regression"), ("dt", "decision_tree")],
...     final_estimator="ridge",
... )
>>> stacker.fit(X_train, y_train)
>>> predictions = stacker.predict(X_test)
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
VotingClassifier = _native.ensemble.VotingClassifier
VotingRegressor = _native.ensemble.VotingRegressor
StackingClassifier = _native.ensemble.StackingClassifier
StackingRegressor = _native.ensemble.StackingRegressor

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
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
]
