"""Tests for ModelCard structured metadata."""

import pytest

import ferroml
from ferroml import ModelCard


# ---------------------------------------------------------------------------
# All model classes that should have model_card()
# ---------------------------------------------------------------------------
ALL_MODEL_CLASSES = []

# Linear
from ferroml.linear import (
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNet,
    RobustRegression,
    QuantileRegression,
    Perceptron,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
    RidgeClassifier,
    IsotonicRegression,
)

ALL_MODEL_CLASSES += [
    LinearRegression,
    LogisticRegression,
    RidgeRegression,
    LassoRegression,
    ElasticNet,
    RobustRegression,
    QuantileRegression,
    Perceptron,
    RidgeCV,
    LassoCV,
    ElasticNetCV,
    RidgeClassifier,
    IsotonicRegression,
]

# Trees
from ferroml.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

ALL_MODEL_CLASSES += [
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
]

# Ensemble
from ferroml.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    SGDClassifier,
    SGDRegressor,
    PassiveAggressiveClassifier,
    BaggingClassifier,
    BaggingRegressor,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)

ALL_MODEL_CLASSES += [
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    SGDClassifier,
    SGDRegressor,
    PassiveAggressiveClassifier,
    BaggingClassifier,
    BaggingRegressor,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
]

# SVM
from ferroml.svm import LinearSVC, LinearSVR, SVC, SVR

ALL_MODEL_CLASSES += [LinearSVC, LinearSVR, SVC, SVR]

# Neighbors
from ferroml.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestCentroid

ALL_MODEL_CLASSES += [KNeighborsClassifier, KNeighborsRegressor, NearestCentroid]

# Naive Bayes
from ferroml.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB

ALL_MODEL_CLASSES += [GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB]

# Neural
from ferroml.neural import MLPClassifier, MLPRegressor

ALL_MODEL_CLASSES += [MLPClassifier, MLPRegressor]

# Gaussian Process
from ferroml.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier

ALL_MODEL_CLASSES += [GaussianProcessRegressor, GaussianProcessClassifier]

# Clustering
from ferroml.clustering import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    GaussianMixture,
    HDBSCAN,
    MiniBatchKMeans,
)

ALL_MODEL_CLASSES += [
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    GaussianMixture,
    HDBSCAN,
    MiniBatchKMeans,
]

# Decomposition
from ferroml.decomposition import PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis, TSNE

ALL_MODEL_CLASSES += [PCA, IncrementalPCA, TruncatedSVD, LDA, FactorAnalysis, TSNE]

# Anomaly
from ferroml.anomaly import IsolationForest, LocalOutlierFactor

ALL_MODEL_CLASSES += [IsolationForest, LocalOutlierFactor]

# Calibration
from ferroml.calibration import TemperatureScalingCalibrator

ALL_MODEL_CLASSES += [TemperatureScalingCalibrator]

# MultiOutput
from ferroml.multioutput import MultiOutputRegressor, MultiOutputClassifier

ALL_MODEL_CLASSES += [MultiOutputRegressor, MultiOutputClassifier]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

VALID_TASKS = {
    "classification",
    "regression",
    "clustering",
    "dimensionality_reduction",
    "outlier_detection",
    "calibration",
    "ensemble",
}


def test_model_card_linear_regression():
    card = LinearRegression.model_card()
    assert isinstance(card, ModelCard)
    assert card.name == "LinearRegression"
    assert card.task == ["regression"]
    assert card.interpretability == "high"
    assert card.complexity == "O(n*p^2)"
    assert not card.supports_sparse
    assert not card.supports_incremental
    assert len(card.strengths) >= 2
    assert len(card.limitations) >= 1
    assert len(card.references) >= 1


def test_model_card_random_forest():
    card = RandomForestClassifier.model_card()
    assert card.name == "RandomForestClassifier"
    assert card.task == ["classification"]
    assert card.interpretability == "medium"


def test_model_card_all_models_have_card():
    """Every model class has a model_card() static method returning ModelCard."""
    for cls in ALL_MODEL_CLASSES:
        card = cls.model_card()
        assert isinstance(card, ModelCard), f"{cls.__name__}.model_card() did not return ModelCard"


def test_model_card_fields_not_empty():
    """All fields of every model card are non-empty."""
    for cls in ALL_MODEL_CLASSES:
        card = cls.model_card()
        assert card.name, f"name is empty for {cls.__name__}"
        assert card.task, f"task is empty for {cls.__name__}"
        assert card.complexity, f"complexity is empty for {cls.__name__}"
        assert card.interpretability, f"interpretability is empty for {cls.__name__}"
        assert card.strengths, f"strengths is empty for {cls.__name__}"
        assert card.limitations, f"limitations is empty for {cls.__name__}"
        assert card.references, f"references is empty for {cls.__name__}"


def test_model_card_task_valid():
    """All task values are from the valid set."""
    for cls in ALL_MODEL_CLASSES:
        card = cls.model_card()
        for task in card.task:
            assert task in VALID_TASKS, (
                f"Invalid task '{task}' for {cls.__name__}. "
                f"Valid: {VALID_TASKS}"
            )


def test_model_card_repr():
    card = LinearRegression.model_card()
    r = repr(card)
    assert "ModelCard" in r
    assert "LinearRegression" in r
    assert "regression" in r


def test_model_card_incremental_models():
    """Models that support partial_fit should have supports_incremental=True."""
    incremental_models = [SGDClassifier, SGDRegressor, GaussianNB]
    for cls in incremental_models:
        card = cls.model_card()
        assert card.supports_incremental, f"{cls.__name__} should have supports_incremental=True"


def test_model_card_interpretability_levels():
    """Interpretability follows expected levels for known model types."""
    # High: linear, tree, naive_bayes
    for cls in [LinearRegression, DecisionTreeClassifier, GaussianNB]:
        card = cls.model_card()
        assert card.interpretability == "high", f"{cls.__name__} should be 'high'"

    # Medium: ensemble
    for cls in [RandomForestClassifier, GradientBoostingClassifier]:
        card = cls.model_card()
        assert card.interpretability == "medium", f"{cls.__name__} should be 'medium'"

    # Low: neural, SVM (RBF)
    for cls in [MLPClassifier, SVC]:
        card = cls.model_card()
        assert card.interpretability == "low", f"{cls.__name__} should be 'low'"


def test_model_card_count():
    """Ensure we have cards for a reasonable number of models."""
    assert len(ALL_MODEL_CLASSES) >= 50, (
        f"Expected at least 50 model classes, got {len(ALL_MODEL_CLASSES)}"
    )
