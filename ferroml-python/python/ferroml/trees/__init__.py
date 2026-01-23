"""
FerroML Tree-Based Models

Decision tree, random forest, and gradient boosting models with
feature importance and out-of-bag error estimation.

Classes
-------
DecisionTreeClassifier
    CART classifier with Gini/entropy criteria and cost-complexity pruning
DecisionTreeRegressor
    CART regressor with MSE/MAE criteria
RandomForestClassifier
    Bootstrap aggregating with OOB error and feature importance CIs
RandomForestRegressor
    Bootstrap aggregating for regression tasks
GradientBoostingClassifier
    Gradient boosting with learning rate scheduling and early stopping
GradientBoostingRegressor
    Gradient boosting with multiple loss functions (squared, absolute, Huber)
HistGradientBoostingClassifier
    LightGBM-style histogram-based boosting with native missing value handling
HistGradientBoostingRegressor
    Histogram-based boosting for regression tasks

Example
-------
>>> from ferroml.trees import RandomForestClassifier
>>> import numpy as np
>>>
>>> model = RandomForestClassifier(n_estimators=100, random_state=42)
>>> model.fit(X_train, y_train)
>>> print(f"OOB Score: {model.oob_score_:.4f}")
>>> print(f"Feature importances: {model.feature_importances_}")
"""

# Import from the native extension's trees submodule
from ferroml import ferroml as _native

DecisionTreeClassifier = _native.trees.DecisionTreeClassifier
DecisionTreeRegressor = _native.trees.DecisionTreeRegressor
RandomForestClassifier = _native.trees.RandomForestClassifier
RandomForestRegressor = _native.trees.RandomForestRegressor
GradientBoostingClassifier = _native.trees.GradientBoostingClassifier
GradientBoostingRegressor = _native.trees.GradientBoostingRegressor
HistGradientBoostingClassifier = _native.trees.HistGradientBoostingClassifier
HistGradientBoostingRegressor = _native.trees.HistGradientBoostingRegressor

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
]
