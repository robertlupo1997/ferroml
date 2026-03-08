"""
FerroML Naive Bayes Classifiers

Naive Bayes classifiers with different feature distribution assumptions.

Classes
-------
GaussianNB
    Gaussian Naive Bayes for continuous features
MultinomialNB
    Multinomial Naive Bayes for count/frequency features
BernoulliNB
    Bernoulli Naive Bayes for binary/boolean features
"""

from ferroml import ferroml as _native

GaussianNB = _native.naive_bayes.GaussianNB
MultinomialNB = _native.naive_bayes.MultinomialNB
BernoulliNB = _native.naive_bayes.BernoulliNB

__all__ = [
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
]
