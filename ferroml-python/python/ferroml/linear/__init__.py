"""
FerroML Linear Models

Statistical linear models with full diagnostics including coefficient
standard errors, t-statistics, p-values, confidence intervals, and more.

Classes
-------
LinearRegression
    OLS regression with R-style summary output, VIF, and residual diagnostics
LogisticRegression
    Maximum likelihood logistic regression with odds ratios and pseudo R-squared
RidgeRegression
    L2 regularized regression with effective degrees of freedom
LassoRegression
    L1 regularized regression with sparse solutions
ElasticNet
    Combined L1/L2 regularization
RidgeCV
    Ridge regression with built-in cross-validated alpha selection
LassoCV
    Lasso regression with built-in cross-validated alpha selection
ElasticNetCV
    Elastic net with built-in cross-validated alpha/l1_ratio selection
RidgeClassifier
    Ridge regression adapted for classification
RobustRegression
    Regression robust to outliers (Huber loss)
QuantileRegression
    Regression for estimating conditional quantiles
Perceptron
    Single-layer perceptron for linear classification
IsotonicRegression
    Non-parametric monotonic regression

Example
-------
>>> from ferroml.linear import LinearRegression
>>> import numpy as np
>>>
>>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
>>> y = np.array([3.1, 4.9, 7.2, 9.0, 10.8])
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> print(model.summary())  # Full statistical output
>>> print(f"R-squared: {model.r_squared():.4f}")
"""

# Import from the native extension's linear submodule
from ferroml import ferroml as _native

LinearRegression = _native.linear.LinearRegression
LogisticRegression = _native.linear.LogisticRegression
RidgeRegression = _native.linear.RidgeRegression
LassoRegression = _native.linear.LassoRegression
ElasticNet = _native.linear.ElasticNet
RobustRegression = _native.linear.RobustRegression
QuantileRegression = _native.linear.QuantileRegression
Perceptron = _native.linear.Perceptron
RidgeCV = _native.linear.RidgeCV
LassoCV = _native.linear.LassoCV
ElasticNetCV = _native.linear.ElasticNetCV
RidgeClassifier = _native.linear.RidgeClassifier
IsotonicRegression = _native.linear.IsotonicRegression

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
    "RobustRegression",
    "QuantileRegression",
    "Perceptron",
    "RidgeCV",
    "LassoCV",
    "ElasticNetCV",
    "RidgeClassifier",
    "IsotonicRegression",
]
