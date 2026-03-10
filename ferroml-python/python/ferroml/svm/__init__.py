"""
FerroML Support Vector Machines

Support Vector Machines for classification and regression,
including both linear and kernel-based variants.

Classes
-------
LinearSVC
    Linear Support Vector Classification using coordinate descent on
    the primal formulation; efficient for large datasets.
LinearSVR
    Linear Support Vector Regression using epsilon-insensitive loss.
SVC
    Support Vector Classification with kernel methods (RBF, poly,
    sigmoid, linear); uses the SMO algorithm for training.
SVR
    Support Vector Regression with kernel methods (RBF, poly,
    sigmoid, linear); supports epsilon-insensitive loss.

Example
-------
>>> from ferroml.svm import SVC, LinearSVC
>>> import numpy as np
>>>
>>> X = np.array([[1, 2], [2, 1], [3, 3], [6, 7], [7, 6], [8, 8]])
>>> y = np.array([0, 0, 0, 1, 1, 1])
>>> model = SVC(kernel="rbf", C=1.0)
>>> model.fit(X, y)
>>> predictions = model.predict(X)
>>>
>>> # Linear SVM (faster for large datasets)
>>> linear_model = LinearSVC(C=1.0)
>>> linear_model.fit(X, y)
"""

from ferroml import ferroml as _native

LinearSVC = _native.svm.LinearSVC
LinearSVR = _native.svm.LinearSVR
SVC = _native.svm.SVC
SVR = _native.svm.SVR

__all__ = [
    "LinearSVC",
    "LinearSVR",
    "SVC",
    "SVR",
]
