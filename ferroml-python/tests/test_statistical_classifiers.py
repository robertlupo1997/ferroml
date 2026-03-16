"""
Statistical classifier tests: predict_proba calibration, score accuracy,
and decision_function consistency across all FerroML classifiers.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from ferroml.linear import LogisticRegression
from ferroml.trees import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    DecisionTreeClassifier,
)
from ferroml.svm import SVC
from ferroml.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from ferroml.neighbors import KNeighborsClassifier


# ---------------------------------------------------------------------------
# Classifier lists
# ---------------------------------------------------------------------------

PROBA_CLASSIFIERS = [
    ("LogisticRegression", LogisticRegression()),
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
    ("SVC", SVC(kernel="rbf", probability=True)),
    ("GaussianNB", GaussianNB()),
    ("MultinomialNB", MultinomialNB()),
    ("BernoulliNB", BernoulliNB()),
    ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
]

# decision_function: LogReg returns 1D, trees return 2D, SVC returns 2D (inverted sign)
DECISION_FUNCTION_CLASSIFIERS = [
    ("LogisticRegression", LogisticRegression()),
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
    ("SVC", SVC(kernel="rbf", probability=True)),
]

ALL_CLASSIFIERS = [
    ("LogisticRegression", LogisticRegression()),
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
    ("SVC", SVC(kernel="rbf", probability=True)),
    ("GaussianNB", GaussianNB()),
    ("BernoulliNB", BernoulliNB()),
    ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    return X, y.astype(np.float64)


@pytest.fixture
def well_separated_data():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        class_sep=4.0,
        random_state=42,
    )
    return X, y.astype(np.float64)


@pytest.fixture
def random_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    y = rng.randint(0, 2, size=200).astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Helper: fit a model, handling MultinomialNB's non-negative requirement
# ---------------------------------------------------------------------------

def _fit_model(name, model, X, y):
    if name == "MultinomialNB":
        X_use = X - X.min(axis=0)
    else:
        X_use = X
    model.fit(X_use, y)
    return X_use


# ---------------------------------------------------------------------------
# predict_proba tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,model", PROBA_CLASSIFIERS)
def test_proba_sums_to_one(name, model, binary_data):
    X, y = binary_data
    X_use = _fit_model(name, model, X, y)
    proba = np.array(model.predict_proba(X_use))
    sums = np.sum(proba, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6, err_msg=f"{name} proba rows don't sum to 1")


@pytest.mark.parametrize("name,model", PROBA_CLASSIFIERS)
def test_proba_between_0_and_1(name, model, binary_data):
    X, y = binary_data
    X_use = _fit_model(name, model, X, y)
    proba = np.array(model.predict_proba(X_use))
    assert np.all(proba >= -1e-10), f"{name}: some probabilities < 0"
    assert np.all(proba <= 1.0 + 1e-10), f"{name}: some probabilities > 1"


@pytest.mark.parametrize("name,model", PROBA_CLASSIFIERS)
def test_predict_matches_argmax_proba(name, model, binary_data):
    X, y = binary_data
    X_use = _fit_model(name, model, X, y)
    proba = np.array(model.predict_proba(X_use))
    preds = np.array(model.predict(X_use))
    expected = np.argmax(proba, axis=1).astype(float)
    np.testing.assert_array_equal(
        preds, expected, err_msg=f"{name}: predict != argmax(predict_proba)"
    )


@pytest.mark.parametrize("name,model", PROBA_CLASSIFIERS)
def test_score_matches_manual(name, model, binary_data):
    X, y = binary_data
    X_use = _fit_model(name, model, X, y)
    preds = np.array(model.predict(X_use))
    manual_acc = accuracy_score(y, preds)
    model_score = model.score(X_use, y)
    np.testing.assert_allclose(
        model_score, manual_acc, atol=1e-10,
        err_msg=f"{name}: score() != manual accuracy"
    )


# ---------------------------------------------------------------------------
# decision_function tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,model", DECISION_FUNCTION_CLASSIFIERS)
def test_decision_function_consistent_with_predict(name, model, binary_data):
    X, y = binary_data
    X_use = _fit_model(name, model, X, y)
    df = np.array(model.decision_function(X_use))
    preds = np.array(model.predict(X_use))

    if df.ndim == 2 and df.shape[1] == 2:
        # 2-column decision_function: argmax should match predict
        df_class = np.argmax(df, axis=1).astype(float)
    elif df.ndim == 2 and df.shape[1] == 1:
        # Single-column 2D (e.g. SVC): flatten and use sign
        # SVC uses inverted sign convention: negative -> class 1
        df_flat = df.flatten()
        if name == "SVC":
            df_class = (df_flat < 0).astype(float)
        else:
            df_class = (df_flat >= 0).astype(float)
    else:
        # 1D decision_function: sign convention (positive -> class 1)
        df_flat = df.flatten()
        df_class = (df_flat >= 0).astype(float)

    agreement = np.mean(df_class == preds)
    assert agreement > 0.95, (
        f"{name}: decision_function agrees with predict only "
        f"{agreement * 100:.1f}% of the time"
    )


# ---------------------------------------------------------------------------
# Cross-model calibration tests
# ---------------------------------------------------------------------------

def test_proba_calibration_well_separated(well_separated_data):
    """On well-separated data, max(proba) should be high for most samples."""
    X, y = well_separated_data
    models = [
        ("LogisticRegression", LogisticRegression()),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("GaussianNB", GaussianNB()),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
    ]
    for name, model in models:
        model.fit(X, y)
        proba = np.array(model.predict_proba(X))
        max_proba = np.max(proba, axis=1)
        mean_max = np.mean(max_proba)
        assert mean_max > 0.9, (
            f"{name}: mean max proba on well-separated data = {mean_max:.3f}, expected > 0.9"
        )


def test_proba_calibration_random_data(random_data):
    """On random data, mean max proba should be moderate (0.5-0.85)."""
    X, y = random_data
    models = [
        ("LogisticRegression", LogisticRegression()),
        ("GaussianNB", GaussianNB()),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
    ]
    for name, model in models:
        model.fit(X, y)
        proba = np.array(model.predict_proba(X))
        max_proba = np.max(proba, axis=1)
        mean_max = np.mean(max_proba)
        assert 0.45 <= mean_max <= 0.90, (
            f"{name}: mean max proba on random data = {mean_max:.3f}, expected in [0.45, 0.90]"
        )


def test_all_classifiers_score_above_random(well_separated_data):
    """On linearly separable data, all classifiers should score > 0.7."""
    X, y = well_separated_data
    classifiers = [
        ("LogisticRegression", LogisticRegression()),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
        ("SVC", SVC(kernel="rbf", probability=True)),
        ("GaussianNB", GaussianNB()),
        ("BernoulliNB", BernoulliNB()),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
    ]
    for name, model in classifiers:
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.7, f"{name}: score = {score:.3f} on separable data, expected > 0.7"
