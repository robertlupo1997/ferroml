"""
FerroML Datasets

Dataset loading utilities including built-in toy datasets, synthetic
data generators, and HuggingFace Hub integration.

Classes
-------
Dataset
    Container for feature matrix and target vector
DatasetInfo
    Metadata describing a loaded dataset (name, feature names, etc.)

Built-in Datasets
-----------------
load_iris
    Fisher's iris dataset (150 samples, 4 features, 3 classes)
load_wine
    Wine recognition dataset (178 samples, 13 features, 3 classes)
load_diabetes
    Diabetes regression dataset (442 samples, 10 features)
load_linnerud
    Linnerud physiological dataset (20 samples, 3 features)

Synthetic Generators
--------------------
make_classification
    Generate a random classification problem
make_regression
    Generate a random regression problem
make_blobs
    Generate isotropic Gaussian blobs for clustering
make_moons
    Generate two interleaving half-circle datasets
make_circles
    Generate concentric circle datasets

HuggingFace Hub
---------------
load_huggingface
    Load a dataset from HuggingFace Hub

Example
-------
>>> from ferroml.datasets import load_iris, make_classification
>>>
>>> dataset, info = load_iris()
>>> print(f"Iris: {dataset.X.shape}, targets: {dataset.y.shape}")
>>>
>>> X, y = make_classification(n_samples=200, n_features=5, random_state=42)
"""

# Import from the native extension's datasets submodule
from ferroml import ferroml as _native

# Classes
Dataset = _native.datasets.Dataset
DatasetInfo = _native.datasets.DatasetInfo

# Built-in datasets
load_iris = _native.datasets.load_iris
load_wine = _native.datasets.load_wine
load_diabetes = _native.datasets.load_diabetes
load_linnerud = _native.datasets.load_linnerud

# Synthetic generators
make_classification = _native.datasets.make_classification
make_regression = _native.datasets.make_regression
make_blobs = _native.datasets.make_blobs
make_moons = _native.datasets.make_moons
make_circles = _native.datasets.make_circles

# HuggingFace Hub
load_huggingface = _native.datasets.load_huggingface

__all__ = [
    # Classes
    "Dataset",
    "DatasetInfo",
    # Built-in datasets
    "load_iris",
    "load_wine",
    "load_diabetes",
    "load_linnerud",
    # Synthetic generators
    "make_classification",
    "make_regression",
    "make_blobs",
    "make_moons",
    "make_circles",
    # HuggingFace Hub
    "load_huggingface",
]
