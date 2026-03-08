"""
FerroML Support Vector Machines

Support Vector Machines for classification and regression,
including both linear and kernel-based variants.

Classes
-------
LinearSVC
    Linear Support Vector Classification
LinearSVR
    Linear Support Vector Regression
SVC
    Support Vector Classification with kernel methods (RBF, poly, sigmoid, linear)
SVR
    Support Vector Regression with kernel methods (RBF, poly, sigmoid, linear)
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
