"""
FerroML Neural Network Models

Multi-layer perceptron models for classification and regression with
support for early stopping, multiple activations, and various solvers.

Classes
-------
MLPClassifier
    Multi-layer perceptron classifier with backpropagation
MLPRegressor
    Multi-layer perceptron regressor with backpropagation

Example
-------
>>> from ferroml.neural import MLPClassifier, MLPRegressor
>>> import numpy as np
>>>
>>> model = MLPClassifier(hidden_layer_sizes=[100, 50], activation="relu",
...                       max_iter=200, random_state=42)
>>> model.fit(X_train, y_train)
>>> print(f"Training loss curve length: {len(model.loss_curve_)}")
>>> proba = model.predict_proba(X_test)
"""

# Import from the native extension's neural submodule
from ferroml import ferroml as _native

MLPClassifier = _native.neural.MLPClassifier
MLPRegressor = _native.neural.MLPRegressor

__all__ = [
    "MLPClassifier",
    "MLPRegressor",
]
