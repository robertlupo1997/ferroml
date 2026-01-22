"""Test that all FerroML imports work correctly."""

import pytest


def test_import_ferroml():
    """Test top-level ferroml import."""
    import ferroml
    assert hasattr(ferroml, '__version__')


def test_import_linear():
    """Test linear models import."""
    from ferroml.linear import (
        LinearRegression,
        LogisticRegression,
        RidgeRegression,
        LassoRegression,
        ElasticNet,
    )
    # Check classes can be instantiated
    assert LinearRegression is not None
    assert LogisticRegression is not None
    assert RidgeRegression is not None
    assert LassoRegression is not None
    assert ElasticNet is not None


def test_import_trees():
    """Test tree models import."""
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
    assert DecisionTreeClassifier is not None
    assert DecisionTreeRegressor is not None
    assert RandomForestClassifier is not None
    assert RandomForestRegressor is not None
    assert GradientBoostingClassifier is not None
    assert GradientBoostingRegressor is not None
    assert HistGradientBoostingClassifier is not None
    assert HistGradientBoostingRegressor is not None


def test_import_preprocessing():
    """Test preprocessing import."""
    from ferroml.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        MaxAbsScaler,
        OneHotEncoder,
        OrdinalEncoder,
        LabelEncoder,
        SimpleImputer,
    )
    assert StandardScaler is not None
    assert MinMaxScaler is not None
    assert RobustScaler is not None
    assert MaxAbsScaler is not None
    assert OneHotEncoder is not None
    assert OrdinalEncoder is not None
    assert LabelEncoder is not None
    assert SimpleImputer is not None


def test_import_pipeline():
    """Test pipeline import."""
    from ferroml.pipeline import (
        Pipeline,
        ColumnTransformer,
        FeatureUnion,
    )
    assert Pipeline is not None
    assert ColumnTransformer is not None
    assert FeatureUnion is not None


def test_import_automl():
    """Test automl import."""
    from ferroml.automl import (
        AutoMLConfig,
        AutoML,
        AutoMLResult,
        LeaderboardEntry,
        EnsembleResult,
        EnsembleMember,
    )
    assert AutoMLConfig is not None
    assert AutoML is not None
    assert AutoMLResult is not None
    assert LeaderboardEntry is not None
    assert EnsembleResult is not None
    assert EnsembleMember is not None
