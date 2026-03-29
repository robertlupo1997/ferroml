"""Model registry: resolve model class name strings to constructors."""
from __future__ import annotations

import sys
from typing import Any


# Lazy registry — maps model name -> (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    # linear
    "LinearRegression": ("ferroml.linear", "LinearRegression"),
    "LogisticRegression": ("ferroml.linear", "LogisticRegression"),
    "RidgeRegression": ("ferroml.linear", "RidgeRegression"),
    "LassoRegression": ("ferroml.linear", "LassoRegression"),
    "ElasticNet": ("ferroml.linear", "ElasticNet"),
    "RobustRegression": ("ferroml.linear", "RobustRegression"),
    "QuantileRegression": ("ferroml.linear", "QuantileRegression"),
    "Perceptron": ("ferroml.linear", "Perceptron"),
    "RidgeCV": ("ferroml.linear", "RidgeCV"),
    "LassoCV": ("ferroml.linear", "LassoCV"),
    "ElasticNetCV": ("ferroml.linear", "ElasticNetCV"),
    "RidgeClassifier": ("ferroml.linear", "RidgeClassifier"),
    "IsotonicRegression": ("ferroml.linear", "IsotonicRegression"),
    # trees
    "DecisionTreeClassifier": ("ferroml.trees", "DecisionTreeClassifier"),
    "DecisionTreeRegressor": ("ferroml.trees", "DecisionTreeRegressor"),
    "RandomForestClassifier": ("ferroml.trees", "RandomForestClassifier"),
    "RandomForestRegressor": ("ferroml.trees", "RandomForestRegressor"),
    "GradientBoostingClassifier": ("ferroml.trees", "GradientBoostingClassifier"),
    "GradientBoostingRegressor": ("ferroml.trees", "GradientBoostingRegressor"),
    "HistGradientBoostingClassifier": ("ferroml.trees", "HistGradientBoostingClassifier"),
    "HistGradientBoostingRegressor": ("ferroml.trees", "HistGradientBoostingRegressor"),
    # ensemble
    "ExtraTreesClassifier": ("ferroml.ensemble", "ExtraTreesClassifier"),
    "ExtraTreesRegressor": ("ferroml.ensemble", "ExtraTreesRegressor"),
    "AdaBoostClassifier": ("ferroml.ensemble", "AdaBoostClassifier"),
    "AdaBoostRegressor": ("ferroml.ensemble", "AdaBoostRegressor"),
    "SGDClassifier": ("ferroml.ensemble", "SGDClassifier"),
    "SGDRegressor": ("ferroml.ensemble", "SGDRegressor"),
    # neighbors
    "KNeighborsClassifier": ("ferroml.neighbors", "KNeighborsClassifier"),
    "KNeighborsRegressor": ("ferroml.neighbors", "KNeighborsRegressor"),
    "NearestCentroid": ("ferroml.neighbors", "NearestCentroid"),
    # clustering
    "KMeans": ("ferroml.clustering", "KMeans"),
    "MiniBatchKMeans": ("ferroml.clustering", "MiniBatchKMeans"),
    "DBSCAN": ("ferroml.clustering", "DBSCAN"),
    "AgglomerativeClustering": ("ferroml.clustering", "AgglomerativeClustering"),
    "GaussianMixture": ("ferroml.clustering", "GaussianMixture"),
    "HDBSCAN": ("ferroml.clustering", "HDBSCAN"),
    # naive_bayes
    "GaussianNB": ("ferroml.naive_bayes", "GaussianNB"),
    "MultinomialNB": ("ferroml.naive_bayes", "MultinomialNB"),
    "BernoulliNB": ("ferroml.naive_bayes", "BernoulliNB"),
    "CategoricalNB": ("ferroml.naive_bayes", "CategoricalNB"),
    # svm
    "LinearSVC": ("ferroml.svm", "LinearSVC"),
    "LinearSVR": ("ferroml.svm", "LinearSVR"),
    "SVC": ("ferroml.svm", "SVC"),
    "SVR": ("ferroml.svm", "SVR"),
    # neural
    "MLPClassifier": ("ferroml.neural", "MLPClassifier"),
    "MLPRegressor": ("ferroml.neural", "MLPRegressor"),
    # gaussian_process
    "GaussianProcessRegressor": ("ferroml.gaussian_process", "GaussianProcessRegressor"),
    "GaussianProcessClassifier": ("ferroml.gaussian_process", "GaussianProcessClassifier"),
    # anomaly
    "IsolationForest": ("ferroml.anomaly", "IsolationForest"),
    "LocalOutlierFactor": ("ferroml.anomaly", "LocalOutlierFactor"),
    # decomposition
    "PCA": ("ferroml.decomposition", "PCA"),
    "TruncatedSVD": ("ferroml.decomposition", "TruncatedSVD"),
    "IncrementalPCA": ("ferroml.decomposition", "IncrementalPCA"),
    "TSNE": ("ferroml.decomposition", "TSNE"),
    "FactorAnalysis": ("ferroml.decomposition", "FactorAnalysis"),
}


def get_model_class(name: str) -> type:
    """Resolve a model name string to its class. Raises SystemExit if not found."""
    if name not in _REGISTRY:
        print(f"Error: unknown model '{name}'.", file=sys.stderr)
        print(f"Available models: {', '.join(sorted(_REGISTRY))}", file=sys.stderr)
        raise SystemExit(1)

    module_path, class_name = _REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def list_models() -> list[str]:
    """Return sorted list of all available model names."""
    return sorted(_REGISTRY)


def construct_model(name: str, params: dict[str, Any] | None = None):
    """Construct a model instance by name, with optional constructor params."""
    cls = get_model_class(name)
    if params:
        return cls(**params)
    return cls()
