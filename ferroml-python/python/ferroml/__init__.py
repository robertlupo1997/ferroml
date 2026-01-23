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
    Tree-based models including RandomForest, GradientBoosting, and HistGradientBoosting
preprocessing
    Data preprocessing transformers (scalers, encoders, imputers)
pipeline
    Pipeline construction (Pipeline, ColumnTransformer, FeatureUnion)
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
from ferroml import preprocessing
from ferroml import pipeline
from ferroml import automl

__all__ = [
    "__version__",
    "linear",
    "trees",
    "preprocessing",
    "pipeline",
    "automl",
]
