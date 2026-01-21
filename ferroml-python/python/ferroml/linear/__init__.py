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

from ferroml.ferroml.linear import (
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNet,
)

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
]
