"""
FerroML Naive Bayes Classifiers

Naive Bayes classifiers with different feature distribution assumptions.
Each variant models the likelihood P(x|y) differently to suit the nature
of the input features.

Classes
-------
GaussianNB
    Gaussian Naive Bayes for continuous features; assumes each feature
    follows a normal distribution within each class.
MultinomialNB
    Multinomial Naive Bayes for count/frequency features; suitable for
    text classification with word counts or TF-IDF.
BernoulliNB
    Bernoulli Naive Bayes for binary/boolean features; models each feature
    as a Bernoulli random variable.
CategoricalNB
    Categorical Naive Bayes for discrete categorical features; models each
    feature as a categorical distribution.

Example
-------
>>> from ferroml.naive_bayes import GaussianNB
>>> import numpy as np
>>>
>>> X = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
>>> y = np.array([0, 0, 1, 1])
>>> model = GaussianNB()
>>> model.fit(X, y)
>>> predictions = model.predict(X)
>>> probas = model.predict_proba(X)
"""

from ferroml import ferroml as _native

GaussianNB = _native.naive_bayes.GaussianNB
MultinomialNB = _native.naive_bayes.MultinomialNB
BernoulliNB = _native.naive_bayes.BernoulliNB
CategoricalNB = _native.naive_bayes.CategoricalNB

__all__ = [
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    "CategoricalNB",
]
