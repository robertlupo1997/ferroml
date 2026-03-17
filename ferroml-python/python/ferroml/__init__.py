"""
FerroML: Statistically Rigorous AutoML in Rust

FerroML is a machine learning library that prioritizes statistical rigor,
providing confidence intervals, effect sizes, and assumption tests alongside
every model prediction.

Submodules
----------
linear
    Linear models with full statistical diagnostics (R-style output)
trees
    Decision tree classifiers and regressors
ensemble
    Ensemble methods (RandomForest, GradientBoosting, AdaBoost, Bagging, etc.)
neighbors
    K-Nearest Neighbors for classification, regression, and nearest centroid
clustering
    Clustering algorithms (KMeans, DBSCAN, Agglomerative, GaussianMixture) with metrics
anomaly
    Anomaly detection (IsolationForest, LocalOutlierFactor)
naive_bayes
    Naive Bayes classifiers (Gaussian, Multinomial, Bernoulli)
svm
    Support Vector Machines (LinearSVC, LinearSVR, SVC, SVR)
neural
    Multi-layer perceptron classifiers and regressors
preprocessing
    Data preprocessing transformers (scalers, encoders, imputers, samplers)
decomposition
    Dimensionality reduction (PCA, t-SNE, TruncatedSVD, LDA, QDA, FactorAnalysis)
explainability
    Model explanations (TreeSHAP, KernelSHAP, PDP, ICE, permutation importance)
calibration
    Post-hoc probability calibration (TemperatureScaling)
pipeline
    Pipeline construction (Pipeline, ColumnTransformer, FeatureUnion)
datasets
    Built-in datasets (Iris, Diabetes, Wine) and synthetic generators
gaussian_process
    Gaussian Process models (GaussianProcessRegressor, GaussianProcessClassifier)
multioutput
    Multi-output wrappers (MultiOutputRegressor, MultiOutputClassifier)
automl
    Automated Machine Learning with statistical testing

Example
-------
>>> from ferroml.linear import LinearRegression
>>> from ferroml.automl import AutoML, AutoMLConfig
>>> import numpy as np
>>>
>>> # Simple linear regression with full diagnostics
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> print(model.summary())  # R-style statistical output
>>>
>>> # AutoML with statistical significance testing
>>> config = AutoMLConfig(task="classification", metric="roc_auc")
>>> automl = AutoML(config)
>>> result = automl.fit(X, y)
>>> print(result.summary())
"""

# Import from the native extension
from ferroml import ferroml as _native
__version__ = _native.__version__

# Re-export submodules for cleaner imports
from ferroml import linear
from ferroml import trees
from ferroml import neighbors
from ferroml import clustering
from ferroml import preprocessing
from ferroml import pipeline
from ferroml import automl
from ferroml import datasets
from ferroml import decomposition
from ferroml import ensemble
from ferroml import explainability
from ferroml import naive_bayes
from ferroml import neural
from ferroml import svm
from ferroml import calibration
from ferroml import anomaly
from ferroml import multioutput
from ferroml import gaussian_process
from ferroml import stats
from ferroml import metrics
from ferroml import cv
from ferroml import hpo
from ferroml import model_selection

__all__ = [
    "__version__",
    "linear",
    "trees",
    "neighbors",
    "clustering",
    "preprocessing",
    "pipeline",
    "automl",
    "datasets",
    "decomposition",
    "ensemble",
    "explainability",
    "naive_bayes",
    "neural",
    "svm",
    "calibration",
    "anomaly",
    "multioutput",
    "gaussian_process",
    "stats",
    "metrics",
    "cv",
    "hpo",
    "model_selection",
]
