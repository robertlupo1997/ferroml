"""
FerroML AutoML

Automated Machine Learning with statistical rigor.

AutoML automatically searches for the best machine learning pipeline,
providing confidence intervals, statistical significance tests, and
ensemble construction.

Classes
-------
AutoMLConfig
    Configuration for AutoML search (task, metric, time budget, etc.)
AutoML
    Main AutoML class that orchestrates the search
AutoMLResult
    Comprehensive result object with leaderboard, ensemble, and diagnostics
LeaderboardEntry
    Entry in the AutoML leaderboard with CV scores and confidence intervals
EnsembleResult
    Result of ensemble construction with member weights
EnsembleMember
    Individual member of the ensemble

Example
-------
>>> from ferroml.automl import AutoML, AutoMLConfig
>>> import numpy as np
>>>
>>> # Configure AutoML
>>> config = AutoMLConfig(
...     task="classification",
...     metric="roc_auc",
...     time_budget_seconds=300,
...     cv_folds=5,
...     statistical_tests=True,
... )
>>>
>>> # Run AutoML
>>> automl = AutoML(config)
>>> result = automl.fit(X, y)
>>>
>>> # Check results
>>> print(result.summary())
>>> best = result.best_model()
>>> print(f"Best: {best.algorithm} (score: {best.cv_score:.4f})")
>>>
>>> # Get models not significantly worse than best
>>> competitive = result.competitive_models()
>>> print(f"Competitive models: {len(competitive)}")
"""

# Import from the native extension's automl submodule
from ferroml import ferroml as _native

AutoMLConfig = _native.automl.AutoMLConfig
AutoML = _native.automl.AutoML
AutoMLResult = _native.automl.AutoMLResult
LeaderboardEntry = _native.automl.LeaderboardEntry
EnsembleResult = _native.automl.EnsembleResult
EnsembleMember = _native.automl.EnsembleMember

__all__ = [
    "AutoMLConfig",
    "AutoML",
    "AutoMLResult",
    "LeaderboardEntry",
    "EnsembleResult",
    "EnsembleMember",
]
