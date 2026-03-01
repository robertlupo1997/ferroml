"""
FerroML Model Explainability

Post-hoc explainability tools: SHAP tree explanations, permutation
feature importance, partial dependence plots (1-D and 2-D), individual
conditional expectation (ICE) curves, and H-statistic interaction
strength.

Classes
-------
TreeExplainer
    SHAP TreeExplainer for decision-tree-based models (random forest,
    gradient boosting, extra trees).  Returns per-sample SHAP values
    that sum to the model output.

Permutation Importance
----------------------
permutation_importance_rf_reg
    Permutation importance for RandomForestRegressor.
permutation_importance_rf_clf
    Permutation importance for RandomForestClassifier.
permutation_importance_dt_reg
    Permutation importance for DecisionTreeRegressor.
permutation_importance_dt_clf
    Permutation importance for DecisionTreeClassifier.
permutation_importance_gb_reg
    Permutation importance for GradientBoostingRegressor.
permutation_importance_gb_clf
    Permutation importance for GradientBoostingClassifier.
permutation_importance_linear
    Permutation importance for linear regression models.
permutation_importance_logistic
    Permutation importance for logistic regression models.
permutation_importance_et_clf
    Permutation importance for ExtraTreesClassifier.
permutation_importance_et_reg
    Permutation importance for ExtraTreesRegressor.

Partial Dependence (1-D)
------------------------
partial_dependence_rf_reg
    Partial dependence for RandomForestRegressor.
partial_dependence_rf_clf
    Partial dependence for RandomForestClassifier.
partial_dependence_gb_reg
    Partial dependence for GradientBoostingRegressor.
partial_dependence_gb_clf
    Partial dependence for GradientBoostingClassifier.
partial_dependence_dt_reg
    Partial dependence for DecisionTreeRegressor.
partial_dependence_dt_clf
    Partial dependence for DecisionTreeClassifier.
partial_dependence_linear
    Partial dependence for linear regression models.

Partial Dependence (2-D)
------------------------
partial_dependence_2d_rf_reg
    Joint 2-D PDP for two features using RandomForestRegressor.
partial_dependence_2d_gb_reg
    Joint 2-D PDP for two features using GradientBoostingRegressor.

Individual Conditional Expectation
-----------------------------------
ice_rf_reg
    ICE curves for RandomForestRegressor.
ice_gb_reg
    ICE curves for GradientBoostingRegressor.
ice_dt_reg
    ICE curves for DecisionTreeRegressor.
ice_linear
    ICE curves for linear regression models.

H-Statistic (Interaction Strength)
------------------------------------
h_statistic_rf_reg
    Pairwise H-statistic for RandomForestRegressor.
h_statistic_gb_reg
    Pairwise H-statistic for GradientBoostingRegressor.
h_statistic_matrix_rf_reg
    Full interaction matrix (all feature pairs) for RandomForestRegressor.

KernelSHAP (Model-Agnostic SHAP)
----------------------------------
kernel_shap_rf_reg
    KernelSHAP for RandomForestRegressor.
kernel_shap_rf_clf
    KernelSHAP for RandomForestClassifier.
kernel_shap_dt_reg
    KernelSHAP for DecisionTreeRegressor.
kernel_shap_dt_clf
    KernelSHAP for DecisionTreeClassifier.
kernel_shap_gb_reg
    KernelSHAP for GradientBoostingRegressor.
kernel_shap_gb_clf
    KernelSHAP for GradientBoostingClassifier.
kernel_shap_linear
    KernelSHAP for LinearRegression.
kernel_shap_logistic
    KernelSHAP for LogisticRegression.
kernel_shap_et_clf
    KernelSHAP for ExtraTreesClassifier.
kernel_shap_et_reg
    KernelSHAP for ExtraTreesRegressor.

Example
-------
>>> from ferroml.explainability import TreeExplainer, permutation_importance_rf_reg
>>> import numpy as np
>>>
>>> # SHAP values
>>> explainer = TreeExplainer(rf_model)
>>> shap_values = explainer.shap_values(X_test)
>>> print(f"SHAP shape: {shap_values.shape}")
>>>
>>> # Permutation importance
>>> importances = permutation_importance_rf_reg(rf_model, X_val, y_val,
...                                             n_repeats=10, random_state=0)
>>> print(f"Feature importances: {importances['importances_mean']}")
"""

# Import from the native extension's explainability submodule
from ferroml import ferroml as _native

# Classes
TreeExplainer = _native.explainability.TreeExplainer

# Permutation importance
permutation_importance_rf_reg = _native.explainability.permutation_importance_rf_reg
permutation_importance_rf_clf = _native.explainability.permutation_importance_rf_clf
permutation_importance_dt_reg = _native.explainability.permutation_importance_dt_reg
permutation_importance_dt_clf = _native.explainability.permutation_importance_dt_clf
permutation_importance_gb_reg = _native.explainability.permutation_importance_gb_reg
permutation_importance_gb_clf = _native.explainability.permutation_importance_gb_clf
permutation_importance_linear = _native.explainability.permutation_importance_linear
permutation_importance_logistic = _native.explainability.permutation_importance_logistic
permutation_importance_et_clf = _native.explainability.permutation_importance_et_clf
permutation_importance_et_reg = _native.explainability.permutation_importance_et_reg

# Partial dependence (1-D)
partial_dependence_rf_reg = _native.explainability.partial_dependence_rf_reg
partial_dependence_rf_clf = _native.explainability.partial_dependence_rf_clf
partial_dependence_gb_reg = _native.explainability.partial_dependence_gb_reg
partial_dependence_gb_clf = _native.explainability.partial_dependence_gb_clf
partial_dependence_dt_reg = _native.explainability.partial_dependence_dt_reg
partial_dependence_dt_clf = _native.explainability.partial_dependence_dt_clf
partial_dependence_linear = _native.explainability.partial_dependence_linear

# Partial dependence (2-D)
partial_dependence_2d_rf_reg = _native.explainability.partial_dependence_2d_rf_reg
partial_dependence_2d_gb_reg = _native.explainability.partial_dependence_2d_gb_reg

# Individual conditional expectation
ice_rf_reg = _native.explainability.ice_rf_reg
ice_gb_reg = _native.explainability.ice_gb_reg
ice_dt_reg = _native.explainability.ice_dt_reg
ice_linear = _native.explainability.ice_linear

# H-statistic
h_statistic_rf_reg = _native.explainability.h_statistic_rf_reg
h_statistic_gb_reg = _native.explainability.h_statistic_gb_reg
h_statistic_matrix_rf_reg = _native.explainability.h_statistic_matrix_rf_reg

# KernelSHAP (model-agnostic SHAP values)
kernel_shap_rf_reg = _native.explainability.kernel_shap_rf_reg
kernel_shap_rf_clf = _native.explainability.kernel_shap_rf_clf
kernel_shap_dt_reg = _native.explainability.kernel_shap_dt_reg
kernel_shap_dt_clf = _native.explainability.kernel_shap_dt_clf
kernel_shap_gb_reg = _native.explainability.kernel_shap_gb_reg
kernel_shap_gb_clf = _native.explainability.kernel_shap_gb_clf
kernel_shap_linear = _native.explainability.kernel_shap_linear
kernel_shap_logistic = _native.explainability.kernel_shap_logistic
kernel_shap_et_clf = _native.explainability.kernel_shap_et_clf
kernel_shap_et_reg = _native.explainability.kernel_shap_et_reg

__all__ = [
    # Classes
    "TreeExplainer",
    # Permutation importance
    "permutation_importance_rf_reg",
    "permutation_importance_rf_clf",
    "permutation_importance_dt_reg",
    "permutation_importance_dt_clf",
    "permutation_importance_gb_reg",
    "permutation_importance_gb_clf",
    "permutation_importance_linear",
    "permutation_importance_logistic",
    "permutation_importance_et_clf",
    "permutation_importance_et_reg",
    # Partial dependence (1-D)
    "partial_dependence_rf_reg",
    "partial_dependence_rf_clf",
    "partial_dependence_gb_reg",
    "partial_dependence_gb_clf",
    "partial_dependence_dt_reg",
    "partial_dependence_dt_clf",
    "partial_dependence_linear",
    # Partial dependence (2-D)
    "partial_dependence_2d_rf_reg",
    "partial_dependence_2d_gb_reg",
    # Individual conditional expectation
    "ice_rf_reg",
    "ice_gb_reg",
    "ice_dt_reg",
    "ice_linear",
    # H-statistic
    "h_statistic_rf_reg",
    "h_statistic_gb_reg",
    "h_statistic_matrix_rf_reg",
    # KernelSHAP
    "kernel_shap_rf_reg",
    "kernel_shap_rf_clf",
    "kernel_shap_dt_reg",
    "kernel_shap_dt_clf",
    "kernel_shap_gb_reg",
    "kernel_shap_gb_clf",
    "kernel_shap_linear",
    "kernel_shap_logistic",
    "kernel_shap_et_clf",
    "kernel_shap_et_reg",
]
