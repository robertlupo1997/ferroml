"""
FerroML Gaussian Processes

Gaussian Process models for regression and classification with
built-in uncertainty quantification.

Classes
-------
GaussianProcessRegressor
    Exact GP regression with predict_with_std for uncertainty estimates.
GaussianProcessClassifier
    Binary GP classification via Laplace approximation.
RBF
    Radial Basis Function (squared exponential) kernel.
Matern
    Matern kernel (nu = 0.5, 1.5, or 2.5).
ConstantKernel
    Constant-valued kernel.
WhiteKernel
    White noise kernel (diagonal only).

Example
-------
>>> from ferroml.gaussian_process import GaussianProcessRegressor, RBF
>>> import numpy as np
>>>
>>> X = np.linspace(0, 5, 20).reshape(-1, 1)
>>> y = np.sin(X).ravel()
>>> gpr = GaussianProcessRegressor(kernel=RBF(1.0))
>>> gpr.fit(X, y)
>>> mean, std = gpr.predict_with_std(X)
"""

from ferroml import ferroml as _native

GaussianProcessRegressor = _native.gaussian_process.GaussianProcessRegressor
GaussianProcessClassifier = _native.gaussian_process.GaussianProcessClassifier
SparseGPRegressor = _native.gaussian_process.SparseGPRegressor
SparseGPClassifier = _native.gaussian_process.SparseGPClassifier
SVGPRegressor = _native.gaussian_process.SVGPRegressor
RBF = _native.gaussian_process.RBF
Matern = _native.gaussian_process.Matern
ConstantKernel = _native.gaussian_process.ConstantKernel
WhiteKernel = _native.gaussian_process.WhiteKernel

__all__ = [
    "GaussianProcessRegressor",
    "GaussianProcessClassifier",
    "SparseGPRegressor",
    "SparseGPClassifier",
    "SVGPRegressor",
    "RBF",
    "Matern",
    "ConstantKernel",
    "WhiteKernel",
]
