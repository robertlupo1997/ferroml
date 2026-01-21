"""
FerroML Pipeline Components

Pipeline construction utilities for chaining transformers and models.

Classes
-------
Pipeline
    Chain transformers and a final model, ensuring proper fit/transform sequencing
ColumnTransformer
    Apply different transformers to different column subsets
FeatureUnion
    Apply multiple transformers in parallel and concatenate outputs

Example
-------
>>> from ferroml.pipeline import Pipeline, ColumnTransformer
>>> from ferroml.preprocessing import StandardScaler, OneHotEncoder
>>> from ferroml.linear import LinearRegression
>>>
>>> # Simple pipeline
>>> pipe = Pipeline([
...     ('scaler', StandardScaler()),
...     ('model', LinearRegression()),
... ])
>>> pipe.fit(X_train, y_train)
>>> predictions = pipe.predict(X_test)
>>>
>>> # Column transformer for mixed data types
>>> ct = ColumnTransformer([
...     ('numeric', StandardScaler(), [0, 1, 2]),
...     ('categorical', OneHotEncoder(), [3, 4]),
... ])
>>> X_transformed = ct.fit_transform(X)
"""

from ferroml.ferroml.pipeline import (
    Pipeline,
    ColumnTransformer,
    FeatureUnion,
)

__all__ = [
    "Pipeline",
    "ColumnTransformer",
    "FeatureUnion",
]
