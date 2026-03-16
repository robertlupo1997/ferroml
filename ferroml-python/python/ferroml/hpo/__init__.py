"""FerroML Hyperparameter Optimization -- grid search, random search, and Optuna-like Study.

This module provides two styles of hyperparameter optimization:

1. **sklearn-like API**: ``GridSearchCV`` and ``RandomSearchCV`` accept a model and
   parameter grid, performing k-fold cross-validation to find the best combination.

2. **Optuna-like API**: ``Study`` provides an ``optimize(objective, n_trials)`` method
   backed by the TPE (Tree-Parzen Estimator) sampler from FerroML's Rust HPO engine.

Classes
-------
GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    Exhaustive search over all parameter combinations with cross-validation.
RandomSearchCV(model, param_distributions, n_iter=10, cv=5, scoring="accuracy", seed=None)
    Randomized search sampling parameter combinations with cross-validation.
Study(direction="minimize", sampler="tpe", seed=None)
    Optuna-like hyperparameter study with ask/tell interface and optimize method.

Examples
--------
>>> from ferroml.hpo import GridSearchCV, RandomSearchCV, Study
>>> from ferroml.trees import RandomForestClassifier
>>> import numpy as np
>>>
>>> # sklearn-like grid search
>>> model = RandomForestClassifier()
>>> grid = GridSearchCV(model, {"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10]})
>>> grid.fit(X, y)
>>> print(grid.best_params_)
>>>
>>> # Optuna-like study
>>> def objective(params):
...     model = RandomForestClassifier(n_estimators=params["n_estimators"])
...     model.fit(X_train, y_train)
...     preds = model.predict(X_test)
...     return -np.mean(preds == y_test)  # minimize negative accuracy
...
>>> study = Study(direction="minimize", sampler="tpe", seed=42)
>>> study.optimize(objective, n_trials=20, search_space={
...     "n_estimators": {"type": "int", "low": 10, "high": 200},
... })
>>> print(study.best_params)
"""

from ferroml.ferroml import hpo as _hpo

GridSearchCV = _hpo.GridSearchCV
RandomSearchCV = _hpo.RandomSearchCV
Study = _hpo.Study

__all__ = [
    "GridSearchCV",
    "RandomSearchCV",
    "Study",
]
