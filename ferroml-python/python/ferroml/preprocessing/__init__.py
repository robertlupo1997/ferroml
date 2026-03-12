"""
FerroML Preprocessing Transformers

Data preprocessing utilities including scalers, encoders, and imputers.
All transformers follow the sklearn-compatible fit/transform API.

Classes
-------
StandardScaler
    Standardize features by removing mean and scaling to unit variance
MinMaxScaler
    Scale features to a given range (default [0, 1])
RobustScaler
    Scale using statistics robust to outliers (median/IQR)
MaxAbsScaler
    Scale by maximum absolute value, preserving sparsity
OneHotEncoder
    Encode categorical features as one-hot numeric arrays
OrdinalEncoder
    Encode categorical features as integers
LabelEncoder
    Encode target labels as integers
SimpleImputer
    Impute missing values with mean, median, mode, or constant

Example
-------
>>> from ferroml.preprocessing import StandardScaler, SimpleImputer
>>> import numpy as np
>>>
>>> # Scaling
>>> scaler = StandardScaler()
>>> X_scaled = scaler.fit_transform(X)
>>> print(f"Mean: {X_scaled.mean(axis=0)}")  # ~0
>>> print(f"Std: {X_scaled.std(axis=0)}")    # ~1
>>>
>>> # Imputation
>>> imputer = SimpleImputer(strategy='mean')
>>> X_imputed = imputer.fit_transform(X_with_missing)
"""

# Import from the native extension's preprocessing submodule
from ferroml import ferroml as _native

StandardScaler = _native.preprocessing.StandardScaler
MinMaxScaler = _native.preprocessing.MinMaxScaler
RobustScaler = _native.preprocessing.RobustScaler
MaxAbsScaler = _native.preprocessing.MaxAbsScaler
OneHotEncoder = _native.preprocessing.OneHotEncoder
OrdinalEncoder = _native.preprocessing.OrdinalEncoder
LabelEncoder = _native.preprocessing.LabelEncoder
TargetEncoder = _native.preprocessing.TargetEncoder
SimpleImputer = _native.preprocessing.SimpleImputer
KNNImputer = _native.preprocessing.KNNImputer
PowerTransformer = _native.preprocessing.PowerTransformer
QuantileTransformer = _native.preprocessing.QuantileTransformer
PolynomialFeatures = _native.preprocessing.PolynomialFeatures
KBinsDiscretizer = _native.preprocessing.KBinsDiscretizer
VarianceThreshold = _native.preprocessing.VarianceThreshold
SelectKBest = _native.preprocessing.SelectKBest
SelectFromModel = _native.preprocessing.SelectFromModel
SMOTE = _native.preprocessing.SMOTE
ADASYN = _native.preprocessing.ADASYN
RandomUnderSampler = _native.preprocessing.RandomUnderSampler
RandomOverSampler = _native.preprocessing.RandomOverSampler
RecursiveFeatureElimination = _native.preprocessing.RecursiveFeatureElimination
TfidfTransformer = _native.preprocessing.TfidfTransformer
CountVectorizer = _native.preprocessing.CountVectorizer

__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "TargetEncoder",
    "SimpleImputer",
    "KNNImputer",
    "PowerTransformer",
    "QuantileTransformer",
    "PolynomialFeatures",
    "KBinsDiscretizer",
    "VarianceThreshold",
    "SelectKBest",
    "SelectFromModel",
    "SMOTE",
    "ADASYN",
    "RandomUnderSampler",
    "RandomOverSampler",
    "RecursiveFeatureElimination",
    "TfidfTransformer",
    "CountVectorizer",
]
