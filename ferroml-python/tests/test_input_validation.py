"""Test input validation across all Python-exposed models (VALID-10).

Verifies that:
- NaN in X raises ValueError (not RuntimeError)
- Inf in X raises ValueError (not RuntimeError)
- NaN in y raises ValueError for supervised models
- Error messages include position info (row, col)
- Unfitted model predict/transform raises error
"""

import numpy as np
import pytest

import ferroml


# ---------------------------------------------------------------------------
# Model lists -- representative models from each binding file
# ---------------------------------------------------------------------------

# (name, constructor, kwargs) -- supervised models that take fit(X, y)
SUPERVISED_MODELS = [
    # linear.rs
    ("LinearRegression", ferroml.linear.LinearRegression, {}),
    ("RidgeRegression", ferroml.linear.RidgeRegression, {}),
    ("LogisticRegression", ferroml.linear.LogisticRegression, {}),
    ("LassoRegression", ferroml.linear.LassoRegression, {}),
    ("ElasticNet", ferroml.linear.ElasticNet, {}),
    ("RidgeClassifier", ferroml.linear.RidgeClassifier, {}),
    ("Perceptron", ferroml.linear.Perceptron, {}),
    # svm.rs
    ("LinearSVC", ferroml.svm.LinearSVC, {}),
    ("LinearSVR", ferroml.svm.LinearSVR, {}),
    ("SVC", ferroml.svm.SVC, {}),
    ("SVR", ferroml.svm.SVR, {}),
    # trees.rs (in ferroml.trees, NOT ferroml.ensemble)
    ("DecisionTreeClassifier", ferroml.trees.DecisionTreeClassifier, {}),
    ("DecisionTreeRegressor", ferroml.trees.DecisionTreeRegressor, {}),
    ("RandomForestClassifier", ferroml.trees.RandomForestClassifier, {}),
    ("RandomForestRegressor", ferroml.trees.RandomForestRegressor, {}),
    ("GradientBoostingClassifier", ferroml.trees.GradientBoostingClassifier, {}),
    ("GradientBoostingRegressor", ferroml.trees.GradientBoostingRegressor, {}),
    # naive_bayes.rs
    ("GaussianNB", ferroml.naive_bayes.GaussianNB, {}),
    # neighbors.rs
    ("KNeighborsClassifier", ferroml.neighbors.KNeighborsClassifier, {}),
    ("KNeighborsRegressor", ferroml.neighbors.KNeighborsRegressor, {}),
    # ensemble.rs
    ("SGDClassifier", ferroml.ensemble.SGDClassifier, {}),
    ("SGDRegressor", ferroml.ensemble.SGDRegressor, {}),
    # gaussian_process.rs
    ("GaussianProcessRegressor", ferroml.gaussian_process.GaussianProcessRegressor, {}),
    # neural.rs
    ("MLPClassifier", ferroml.neural.MLPClassifier, {}),
    ("MLPRegressor", ferroml.neural.MLPRegressor, {}),
]

# Unsupervised models that take fit(X) only
UNSUPERVISED_MODELS = [
    # clustering.rs
    ("KMeans", ferroml.clustering.KMeans, {"n_clusters": 2}),
    ("DBSCAN", ferroml.clustering.DBSCAN, {}),
    ("AgglomerativeClustering", ferroml.clustering.AgglomerativeClustering, {"n_clusters": 2}),
    ("GaussianMixture", ferroml.clustering.GaussianMixture, {"n_components": 2}),
    # decomposition.rs
    ("PCA", ferroml.decomposition.PCA, {"n_components": 2}),
    ("TruncatedSVD", ferroml.decomposition.TruncatedSVD, {"n_components": 2}),
    # anomaly.rs
    ("IsolationForest", ferroml.anomaly.IsolationForest, {}),
    ("LocalOutlierFactor", ferroml.anomaly.LocalOutlierFactor, {}),
]

# Transformers that have fit(X)/transform(X)
TRANSFORMER_MODELS = [
    ("StandardScaler", ferroml.preprocessing.StandardScaler, {}),
    ("MinMaxScaler", ferroml.preprocessing.MinMaxScaler, {}),
    ("MaxAbsScaler", ferroml.preprocessing.MaxAbsScaler, {}),
    ("RobustScaler", ferroml.preprocessing.RobustScaler, {}),
    ("Normalizer", ferroml.preprocessing.Normalizer, {}),
    ("PolynomialFeatures", ferroml.preprocessing.PolynomialFeatures, {}),
]

# All models that accept 2D X
ALL_MODELS = SUPERVISED_MODELS + UNSUPERVISED_MODELS + TRANSFORMER_MODELS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def X_clean():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])


@pytest.fixture
def y_clean():
    return np.array([0.0, 1.0, 0.0, 1.0, 0.0])


@pytest.fixture
def X_nan():
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    return X


@pytest.fixture
def X_inf():
    X = np.array([[1.0, 2.0], [3.0, np.inf], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    return X


@pytest.fixture
def X_neginf():
    X = np.array([[1.0, 2.0], [3.0, -np.inf], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    return X


@pytest.fixture
def y_nan():
    return np.array([0.0, np.nan, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Test NaN in X raises ValueError for supervised models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cls,kwargs", SUPERVISED_MODELS, ids=[m[0] for m in SUPERVISED_MODELS])
def test_supervised_nan_x_raises_value_error(name, cls, kwargs, X_nan, y_clean):
    """NaN in X should raise ValueError before Rust conversion."""
    model = cls(**kwargs)
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X_nan, y_clean)


@pytest.mark.parametrize("name,cls,kwargs", SUPERVISED_MODELS, ids=[m[0] for m in SUPERVISED_MODELS])
def test_supervised_inf_x_raises_value_error(name, cls, kwargs, X_inf, y_clean):
    """Inf in X should raise ValueError before Rust conversion."""
    model = cls(**kwargs)
    with pytest.raises(ValueError, match="Inf"):
        model.fit(X_inf, y_clean)


@pytest.mark.parametrize("name,cls,kwargs", SUPERVISED_MODELS, ids=[m[0] for m in SUPERVISED_MODELS])
def test_supervised_neginf_x_raises_value_error(name, cls, kwargs, X_neginf, y_clean):
    """-Inf in X should raise ValueError before Rust conversion."""
    model = cls(**kwargs)
    with pytest.raises(ValueError, match="Inf"):
        model.fit(X_neginf, y_clean)


# ---------------------------------------------------------------------------
# Test NaN in X raises ValueError for unsupervised models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cls,kwargs", UNSUPERVISED_MODELS, ids=[m[0] for m in UNSUPERVISED_MODELS])
def test_unsupervised_nan_x_raises_value_error(name, cls, kwargs, X_nan):
    """NaN in X should raise ValueError for unsupervised models."""
    model = cls(**kwargs)
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X_nan)


@pytest.mark.parametrize("name,cls,kwargs", UNSUPERVISED_MODELS, ids=[m[0] for m in UNSUPERVISED_MODELS])
def test_unsupervised_inf_x_raises_value_error(name, cls, kwargs, X_inf):
    """Inf in X should raise ValueError for unsupervised models."""
    model = cls(**kwargs)
    with pytest.raises(ValueError, match="Inf"):
        model.fit(X_inf)


# ---------------------------------------------------------------------------
# Test NaN in X raises ValueError for transformers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cls,kwargs", TRANSFORMER_MODELS, ids=[m[0] for m in TRANSFORMER_MODELS])
def test_transformer_nan_x_fit_raises_value_error(name, cls, kwargs, X_nan):
    """NaN in X should raise ValueError for transformer fit."""
    model = cls(**kwargs)
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X_nan)


@pytest.mark.parametrize("name,cls,kwargs", TRANSFORMER_MODELS, ids=[m[0] for m in TRANSFORMER_MODELS])
def test_transformer_nan_x_transform_raises_value_error(name, cls, kwargs, X_clean, X_nan):
    """NaN in X should raise ValueError for transformer transform."""
    model = cls(**kwargs)
    model.fit(X_clean)
    with pytest.raises(ValueError, match="NaN"):
        model.transform(X_nan)


@pytest.mark.parametrize("name,cls,kwargs", TRANSFORMER_MODELS, ids=[m[0] for m in TRANSFORMER_MODELS])
def test_transformer_inf_x_fit_raises_value_error(name, cls, kwargs, X_inf):
    """Inf in X should raise ValueError for transformer fit."""
    model = cls(**kwargs)
    with pytest.raises(ValueError, match="Inf"):
        model.fit(X_inf)


# ---------------------------------------------------------------------------
# Test NaN in y raises ValueError for supervised models
# ---------------------------------------------------------------------------

# Models that accept y as PyReadonlyArray1<f64> (direct check_array1_finite)
SUPERVISED_WITH_DIRECT_Y = [
    ("LinearRegression", ferroml.linear.LinearRegression, {}),
    ("RidgeRegression", ferroml.linear.RidgeRegression, {}),
    ("LassoRegression", ferroml.linear.LassoRegression, {}),
    ("ElasticNet", ferroml.linear.ElasticNet, {}),
    ("LinearSVR", ferroml.svm.LinearSVR, {}),
    ("SVR", ferroml.svm.SVR, {}),
    ("DecisionTreeRegressor", ferroml.trees.DecisionTreeRegressor, {}),
    ("RandomForestRegressor", ferroml.trees.RandomForestRegressor, {}),
    ("GradientBoostingRegressor", ferroml.trees.GradientBoostingRegressor, {}),
    ("SGDRegressor", ferroml.ensemble.SGDRegressor, {}),
    ("MLPRegressor", ferroml.neural.MLPRegressor, {}),
    ("MLPClassifier", ferroml.neural.MLPClassifier, {}),
]


@pytest.mark.parametrize("name,cls,kwargs", SUPERVISED_WITH_DIRECT_Y, ids=[m[0] for m in SUPERVISED_WITH_DIRECT_Y])
def test_nan_y_raises_value_error(name, cls, kwargs, X_clean, y_nan):
    """NaN in y should raise ValueError for supervised models with direct y."""
    model = cls(**kwargs)
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X_clean, y_nan)


# ---------------------------------------------------------------------------
# Test predict on NaN input raises ValueError
# ---------------------------------------------------------------------------

# Subset of supervised models to test predict-side validation
PREDICT_MODELS = [
    ("RidgeRegression", ferroml.linear.RidgeRegression, {}),
    ("LogisticRegression", ferroml.linear.LogisticRegression, {}),
    ("SVC", ferroml.svm.SVC, {}),
    ("RandomForestClassifier", ferroml.trees.RandomForestClassifier, {}),
    ("GaussianNB", ferroml.naive_bayes.GaussianNB, {}),
    ("KNeighborsClassifier", ferroml.neighbors.KNeighborsClassifier, {}),
]


@pytest.mark.parametrize("name,cls,kwargs", PREDICT_MODELS, ids=[m[0] for m in PREDICT_MODELS])
def test_predict_nan_raises_value_error(name, cls, kwargs, X_clean, y_clean, X_nan):
    """NaN in X during predict should raise ValueError."""
    model = cls(**kwargs)
    model.fit(X_clean, y_clean)
    with pytest.raises(ValueError, match="NaN"):
        model.predict(X_nan)


@pytest.mark.parametrize("name,cls,kwargs", PREDICT_MODELS, ids=[m[0] for m in PREDICT_MODELS])
def test_predict_inf_raises_value_error(name, cls, kwargs, X_clean, y_clean, X_inf):
    """Inf in X during predict should raise ValueError."""
    model = cls(**kwargs)
    model.fit(X_clean, y_clean)
    with pytest.raises(ValueError, match="Inf"):
        model.predict(X_inf)


# ---------------------------------------------------------------------------
# Test error messages include position information
# ---------------------------------------------------------------------------

def test_nan_error_message_has_row_col():
    """ValueError for NaN should include (row, col) position."""
    X = np.zeros((3, 4))
    X[2, 3] = np.nan
    y = np.array([0.0, 1.0, 0.0])
    model = ferroml.linear.LinearRegression()
    with pytest.raises(ValueError, match=r"\(2, 3\)"):
        model.fit(X, y)


def test_inf_error_message_has_row_col():
    """ValueError for Inf should include (row, col) position."""
    X = np.zeros((3, 4))
    X[1, 2] = np.inf
    y = np.array([0.0, 1.0, 0.0])
    model = ferroml.linear.LinearRegression()
    with pytest.raises(ValueError, match=r"\(1, 2\)"):
        model.fit(X, y)


def test_nan_y_error_message_has_position():
    """ValueError for NaN in y should include position."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0.0, np.nan, 1.0])
    model = ferroml.linear.LinearRegression()
    with pytest.raises(ValueError, match="position 1"):
        model.fit(X, y)


# ---------------------------------------------------------------------------
# Test error type is ValueError, NOT RuntimeError
# ---------------------------------------------------------------------------

def test_nan_raises_value_error_not_runtime_error():
    """Ensure NaN raises ValueError specifically, not RuntimeError."""
    X = np.array([[np.nan, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])
    model = ferroml.linear.LinearRegression()
    # Should be ValueError
    with pytest.raises(ValueError):
        model.fit(X, y)
    # Should NOT be RuntimeError
    try:
        model.fit(X, y)
    except ValueError:
        pass  # Correct
    except RuntimeError:
        pytest.fail("Got RuntimeError instead of ValueError for NaN input")


def test_inf_raises_value_error_not_runtime_error():
    """Ensure Inf raises ValueError specifically, not RuntimeError."""
    X = np.array([[np.inf, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])
    model = ferroml.linear.LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)
    try:
        model.fit(X, y)
    except ValueError:
        pass
    except RuntimeError:
        pytest.fail("Got RuntimeError instead of ValueError for Inf input")


# ---------------------------------------------------------------------------
# Test unfitted model predict raises error
# ---------------------------------------------------------------------------

UNFITTED_PREDICT_MODELS = [
    ("LinearRegression", ferroml.linear.LinearRegression, {}),
    ("LogisticRegression", ferroml.linear.LogisticRegression, {}),
    ("SVC", ferroml.svm.SVC, {}),
    ("RandomForestClassifier", ferroml.trees.RandomForestClassifier, {}),
    ("GaussianNB", ferroml.naive_bayes.GaussianNB, {}),
    ("KNeighborsClassifier", ferroml.neighbors.KNeighborsClassifier, {}),
    ("PCA", ferroml.decomposition.PCA, {"n_components": 2}),
    ("StandardScaler", ferroml.preprocessing.StandardScaler, {}),
    ("MLPClassifier", ferroml.neural.MLPClassifier, {}),
]


@pytest.mark.parametrize("name,cls,kwargs", UNFITTED_PREDICT_MODELS, ids=[m[0] for m in UNFITTED_PREDICT_MODELS])
def test_unfitted_predict_raises(name, cls, kwargs, X_clean):
    """Calling predict/transform on unfitted model should raise."""
    model = cls(**kwargs)
    with pytest.raises((RuntimeError, ValueError)):
        if hasattr(model, "predict"):
            model.predict(X_clean)
        elif hasattr(model, "transform"):
            model.transform(X_clean)


# ---------------------------------------------------------------------------
# Test score with NaN raises ValueError
# ---------------------------------------------------------------------------

def test_score_nan_x_raises_value_error():
    """Score with NaN in X should raise ValueError."""
    X_clean = np.array([[1.0, 2.0], [3.0, 5.0], [5.0, 1.0], [7.0, 3.0], [9.0, 8.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    model = ferroml.linear.RidgeRegression()
    model.fit(X_clean, y)

    X_nan = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    with pytest.raises(ValueError, match="NaN"):
        model.score(X_nan, y)


def test_score_nan_y_raises_value_error():
    """Score with NaN in y should raise ValueError."""
    X = np.array([[1.0, 2.0], [3.0, 5.0], [5.0, 1.0], [7.0, 3.0], [9.0, 8.0]])
    y_clean = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    model = ferroml.linear.RidgeRegression()
    model.fit(X, y_clean)

    y_nan = np.array([0.0, np.nan, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="NaN"):
        model.score(X, y_nan)


# ---------------------------------------------------------------------------
# Test decision_function and predict_proba with NaN
# ---------------------------------------------------------------------------

def test_predict_proba_nan_raises_value_error():
    """predict_proba with NaN should raise ValueError."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model = ferroml.svm.SVC(probability=True)
    model.fit(X, y)

    X_nan = np.array([[np.nan, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="NaN"):
        model.predict_proba(X_nan)


def test_decision_function_nan_raises_value_error():
    """decision_function with NaN should raise ValueError."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model = ferroml.svm.LinearSVC()
    model.fit(X, y)

    X_nan = np.array([[np.nan, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="NaN"):
        model.decision_function(X_nan)
